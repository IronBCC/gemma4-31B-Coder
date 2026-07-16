#!/usr/bin/env python3
"""Fable-5 teacher-trace collector (P1 pilot, subscription mode).

Per instance: start the SWE-smith container -> run headless `claude` (bash-only,
visible terse reasoning per step, all repo work via `docker exec`) -> harvest
git diff -> score F2P in-container -> persist the FULL stream-json as the raw
trace plus a compact record.

Visible-reasoning pivot (2026-07-15): hidden thinking is API-redacted for this
model generation; the teacher is instructed to externalize 1-3 terse sentences
of reasoning before each command — capturable and style-prescribed at source.

Run on the box:
  .venv-eval/bin/python phaseD_sft/teacher_trace_driver.py \
      --tasks data/teacher_pilot_tasks.jsonl --out runs/teacher_pilot_fable5 --limit 3
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "phaseE_rl"))
from verified_reward import f2p_invocations  # noqa: E402

CLAUDE = str(Path.home() / ".local/bin/claude")
# codex CLI is not installed on the box; npx resolves @latest (one-time per
# instance, ~10s on a 300s task). Runs from a scratch cwd with the git guard
# skipped so it never wanders the real repo (all work is via docker exec).
CODEX_NPX = ["npx", "--yes", "@openai/codex@latest"]
CODEX_SCRATCH = "/tmp/codex_teacher_work"
ENV_BOOTSTRAP = "source /opt/miniconda3/bin/activate testbed 2>/dev/null || true; "
CREDIT_EXHAUSTED_ERROR = "out_of_credits"
# Claude prints this even with subtype=success; codex quota strings are UNVERIFIED
# (never hit the wall in probing) — kept generic and confirmed on first real batch.
_CREDIT_TEXTS = ("out of usage credits", "usage limit reached", "rate limit",
                 "quota exceeded", "insufficient_quota", "you've hit your usage limit")

PROMPT = """You are fixing one repository bug inside a Docker container.

<pr_description>
{problem}
</pr_description>

Rules:
- The repository is at /testbed INSIDE container {cid}. Interact with it ONLY via:
  docker exec {cid} bash -c "cd /testbed && <your command>"
- Never run commands outside `docker exec {cid}`. Never touch other containers,
  services, host files, or user accounts. Stay inside /testbed — do not clone
  upstream copies or compare external checkouts.
- Before EACH command, write 1-3 terse sentences of reasoning. No filler words,
  no restating the task, no pleasantries. Just the decision logic.
- BUDGET: you have at most 30 commands. Make your FIRST source edit within your
  first 12 commands. Never repeat a read command.
- Make a small correct source edit (sed -i or a heredoc patch), then verify with
  ONE focused test run. Do not modify tests.
- CRITICAL: leave your fix APPLIED in the worktree — `git diff` in /testbed must
  show your change when you finish. Then reply with the single word DONE."""


def sh(cmd: list[str], timeout: int = 300, input_text: str | None = None):
    p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout,
                       input=input_text, check=False)
    return (p.stdout + p.stderr), p.returncode


def credit_exhausted_in_stream(stream: str) -> bool:
    """Detect the credit/quota wall (Claude reports subtype=success anyway)."""
    low = stream.lower()
    return any(t in low for t in _CREDIT_TEXTS)


def build_teacher_cmd(backend: str, prompt: str, model: str, max_turns: int) -> tuple[list[str], str | None]:
    """Return (argv, cwd) for the chosen teacher CLI. Both are single-shot,
    bash-only, and instructed in `prompt` to act solely via docker exec."""
    if backend == "claude":
        return ([CLAUDE, "-p", prompt, "--model", model, "--allowedTools", "Bash",
                 "--output-format", "stream-json", "--verbose",
                 "--max-turns", str(max_turns)], None)
    if backend == "codex":
        Path(CODEX_SCRATCH).mkdir(parents=True, exist_ok=True)
        argv = CODEX_NPX + ["exec", "--json", "--skip-git-repo-check",
                            "--sandbox", "danger-full-access",
                            "-c", f"model_reasoning_effort=high"]
        if model:
            argv += ["-c", f'model="{model}"']
        argv.append(prompt)
        return (argv, CODEX_SCRATCH)
    raise ValueError(f"unknown backend {backend}")


def parse_teacher_stream(backend: str, raw_path: Path) -> tuple[int, float]:
    """Count agent turns and cost from the backend's JSONL stream."""
    n_turns = 0
    cost = 0.0
    for line in open(raw_path, errors="replace"):
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            d = json.loads(line)
        except json.JSONDecodeError:
            continue
        if backend == "claude":
            if d.get("type") == "assistant":
                n_turns += 1
            if d.get("type") == "result":
                cost = d.get("total_cost_usd") or cost
        else:  # codex: item.completed with item.type command_execution/agent_message
            if d.get("type") == "item.completed":
                if (d.get("item") or {}).get("type") in ("command_execution", "agent_message"):
                    n_turns += 1
    return n_turns, cost


def next_credit_streak(previous: int, error: str | None) -> int:
    return previous + 1 if error == CREDIT_EXHAUSTED_ERROR else 0


def collect_one(row: dict, outdir: Path, max_turns: int, model: str, backend: str) -> dict:
    iid = row["instance_id"]
    t0 = time.time()
    out, rc = sh(["docker", "run", "-d", row["image_name"], "sleep", "infinity"], 120)
    if rc:
        return {"instance_id": iid, "error": f"container: {out[-200:]}", "resolved": False}
    cid = out.strip().splitlines()[-1][:12]
    rec = {"instance_id": iid, "image": row["image_name"], "model": model, "backend": backend}
    try:
        prompt = PROMPT.format(problem=row["problem_statement"], cid=cid)
        raw_path = outdir / f"{iid}.stream.jsonl"
        argv, cwd = build_teacher_cmd(backend, prompt, model, max_turns)
        with open(raw_path, "w") as raw:
            p = subprocess.run(argv, stdout=raw, stderr=subprocess.DEVNULL, text=True,
                               timeout=3600, check=False, cwd=cwd)
        rec["cli_rc"] = p.returncode
        n_turns, cost = parse_teacher_stream(backend, raw_path)
        rec.update(n_assistant_events=n_turns, cost_usd=cost)
        if credit_exhausted_in_stream(raw_path.read_text(errors="replace")):
            rec.update(error=CREDIT_EXHAUSTED_ERROR, resolved=False, skipped=True)
        else:
            diff, _ = sh(["docker", "exec", cid, "bash", "-c", "cd /testbed && git diff"], 120)
            rec["patch_len"] = len(diff.strip())
            (outdir / f"{iid}.patch").write_text(diff)
            resolved = False
            if diff.strip():
                ok = True
                invocations = f2p_invocations(list(row.get("FAIL_TO_PASS") or []))
                rec["f2p_cmds"] = len(invocations)
                for tc in invocations or ["false"]:
                    t, trc = sh(["docker", "exec", cid, "bash", "-c",
                                 f"cd /testbed && {ENV_BOOTSTRAP}{tc}"], 600)
                    if trc != 0:
                        ok = False
                        break
                resolved = ok and bool(invocations)
            rec["resolved"] = resolved
    except subprocess.TimeoutExpired:
        rec.update(error="collector timeout", resolved=False)
    finally:
        sh(["docker", "rm", "-f", cid], 60)
    rec["wall_s"] = round(time.time() - t0, 1)
    return rec


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=3)
    ap.add_argument("--max-turns", type=int, default=40)
    ap.add_argument("--backend", choices=("claude", "codex"), default="claude")
    ap.add_argument("--model", default=None,
                    help="teacher model; default claude-fable-5 (claude) / gpt-5.6-terra (codex)")
    ap.add_argument("--max-consecutive-credit-hits", type=int, default=3,
                    help="Stop early after this many consecutive out-of-credit responses.")
    a = ap.parse_args()
    model = a.model or ("gpt-5.6-terra" if a.backend == "codex" else "claude-fable-5")

    outdir = Path(a.out)
    outdir.mkdir(parents=True, exist_ok=True)
    rows = [json.loads(l) for l in open(a.tasks)][: a.limit]
    results = []
    credit_streak = 0
    for row in rows:
        rec = collect_one(row, outdir, a.max_turns, model, a.backend)
        results.append(rec)
        print(json.dumps({k: rec.get(k) for k in
                          ("instance_id", "n_assistant_events", "patch_len",
                           "resolved", "cost_usd", "wall_s", "error")}), flush=True)
        with open(outdir / "results.jsonl", "w") as fh:
            for x in results:
                fh.write(json.dumps(x) + "\n")
        credit_streak = next_credit_streak(credit_streak, rec.get("error"))
        if credit_streak >= a.max_consecutive_credit_hits:
            print(f"[teacher-pilot] stopping after {credit_streak} consecutive credit hits", flush=True)
            break
    n = sum(1 for x in results if x.get("resolved"))
    print(f"[teacher-pilot] instances={len(results)} resolved={n}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
