"""Multi-turn agent loop with crash/retry recovery (the 90%-Verified run used this).

Drives an OpenAI-compatible endpoint (vLLM serving Gemma 4 31B) through
localize -> propose edits -> run tests -> iterate, until tests pass or the
turn/budget is exhausted. The full message history (including assistant
``reasoning_content`` preserved across turns, per K2.7-Code) is returned as a
trajectory dict — the same schema collected for Phase-D SFT.

Requires the `openai` package only at runtime (lazy import) so the pure-Python
tools above stay importable without it.
"""
from __future__ import annotations

import json
import subprocess
import time
from dataclasses import asdict, dataclass, field

from . import edit_tool, localizer, test_loop

SYSTEM_PROMPT = """You are a coding agent fixing a bug in a repository.
Work in small steps: localize, inspect, make a minimal edit, run the tests, and
read the failures. Use the provided tools. When the tests pass, stop. Do not
guess at edits — anchor every search_replace with enough surrounding context to
be unique."""

# Tool surface exposed to the model.
TOOLS = edit_tool.TOOL_SCHEMAS + [
    {
        "type": "function",
        "function": {
            "name": "localize",
            "description": "Rank repo files relevant to the issue. Call this first.",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string"}, "top_k": {"type": "integer", "default": 10}},
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read a file (optionally a line range) to inspect before editing.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "start": {"type": "integer", "default": 1},
                    "end": {"type": "integer", "default": 0},
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_tests",
            "description": "Run the task test command and return pass/fail + failing test ids.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
]


@dataclass
class Trajectory:
    instance_id: str
    messages: list[dict] = field(default_factory=list)
    resolved: bool = False
    n_turns: int = 0
    final_patch: str = ""
    error: str | None = None


def _dispatch(name: str, args: dict, *, repo_dir: str, test_cmd: str, docker_container: str | None) -> str:
    """Execute a tool call and return a string tool result for the model."""
    if name == "localize":
        cands = localizer.localize(repo_dir, args["query"], top_k=int(args.get("top_k", 10)))
        return json.dumps([asdict(c) for c in cands]) if cands else "no candidates found"
    if name == "read_file":
        from pathlib import Path

        p = Path(repo_dir) / args["path"]
        if not p.is_file():
            return f"no such file: {args['path']}"
        lines = p.read_text(encoding="utf-8", errors="ignore").splitlines()
        start = max(int(args.get("start", 1)), 1)
        end = int(args.get("end", 0)) or len(lines)
        chunk = lines[start - 1 : end]
        return "\n".join(f"{start + i:>5}  {ln}" for i, ln in enumerate(chunk))[:6000]
    if name == "search_replace":
        r = edit_tool.search_replace(
            str(_abs(repo_dir, args["path"])), args["find"], args["replace"],
            expected_count=int(args.get("expected_count", 1)),
        )
        return _edit_msg(r)
    if name == "ast_edit":
        r = edit_tool.ast_edit(
            str(_abs(repo_dir, args["path"])), args["target_symbol"], args["new_source"],
        )
        return _edit_msg(r)
    if name == "run_tests":
        tr = test_loop.run_tests(repo_dir, test_cmd, docker_container=docker_container)
        return tr.feedback()
    return f"unknown tool: {name}"


def _abs(repo_dir: str, path: str):
    from pathlib import Path

    p = Path(path)
    return p if p.is_absolute() else Path(repo_dir) / p


def _edit_msg(r: edit_tool.EditResult) -> str:
    if not r.ok:
        return f"EDIT FAILED: {r.message}"
    return f"EDIT OK ({r.n_replacements} change(s)).\n{r.diff[:2000]}"


def _git_diff(repo_dir: str, docker_container: str | None) -> str:
    cmd = ["git", "-C", repo_dir, "diff"]
    if docker_container:
        cmd = ["docker", "exec", "-w", repo_dir, docker_container, "git", "diff"]
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=60).stdout
    except Exception:  # noqa: BLE001
        return ""


def solve_task(
    repo_dir: str,
    problem_statement: str,
    test_cmd: str,
    instance_id: str = "task",
    *,
    model: str = "google/gemma-4-31B-it",
    base_url: str = "http://localhost:8000/v1",
    api_key: str = "EMPTY",
    max_turns: int = 40,
    max_retries: int = 3,
    docker_container: str | None = None,
    temperature: float = 0.0,
) -> Trajectory:
    """Run the agent loop and return the trajectory.

    `docker_container`: if set, edits/tests target a running SWE-bench container.
    Preserves assistant ``reasoning_content`` across turns (carried forward in the
    message history) so collected traces match how the model is served.
    """
    from openai import OpenAI  # lazy: keeps tools importable without the SDK

    client = OpenAI(base_url=base_url, api_key=api_key)
    traj = Trajectory(instance_id=instance_id)
    traj.messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Repository at {repo_dir}.\n\nIssue:\n{problem_statement}"},
    ]

    for turn in range(max_turns):
        traj.n_turns = turn + 1
        resp = _chat_with_retry(
            client, model, traj.messages, temperature=temperature, max_retries=max_retries
        )
        if resp is None:
            traj.error = "model call failed after retries"
            break

        msg = resp.choices[0].message
        # Preserve reasoning across turns (vLLM exposes reasoning_content for thinking models).
        assistant_entry: dict = {"role": "assistant", "content": msg.content or ""}
        reasoning = getattr(msg, "reasoning_content", None)
        if reasoning:
            assistant_entry["reasoning_content"] = reasoning
        if msg.tool_calls:
            assistant_entry["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {"name": tc.function.name, "arguments": tc.function.arguments},
                }
                for tc in msg.tool_calls
            ]
        traj.messages.append(assistant_entry)

        if not msg.tool_calls:
            break  # model is done talking

        for tc in msg.tool_calls:
            try:
                args = json.loads(tc.function.arguments or "{}")
            except json.JSONDecodeError as e:
                result = f"TOOL ARG PARSE ERROR (fix your JSON): {e}"
            else:
                try:
                    result = _dispatch(
                        tc.function.name, args,
                        repo_dir=repo_dir, test_cmd=test_cmd, docker_container=docker_container,
                    )
                except Exception as e:  # noqa: BLE001  (crash recovery: report, keep going)
                    result = f"TOOL CRASHED: {type(e).__name__}: {e}"
            traj.messages.append({"role": "tool", "tool_call_id": tc.id, "content": result})
            if tc.function.name == "run_tests" and result.startswith("ALL TESTS PASSED"):
                traj.resolved = True

        if traj.resolved:
            break

    traj.final_patch = _git_diff(repo_dir, docker_container)
    return traj


def _chat_with_retry(client, model, messages, *, temperature, max_retries):
    """Crash/retry recovery around the server call (the 90% run relied on this)."""
    for attempt in range(max_retries):
        try:
            return client.chat.completions.create(
                model=model,
                messages=messages,
                tools=TOOLS,
                tool_choice="auto",
                temperature=temperature,
            )
        except Exception as e:  # noqa: BLE001
            wait = 2 ** attempt
            print(f"[agent] model call failed ({e}); retry {attempt + 1}/{max_retries} in {wait}s")
            time.sleep(wait)
    return None
