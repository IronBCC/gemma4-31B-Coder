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
import re
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from phaseE_rl.verified_reward import f2p_invocations  # noqa: E402
try:
    from teacher_platform.generic_trace_replay import (  # noqa: E402
        ReplayContractError,
        _run_suite,
        build_task_contract,
        parse_patch_paths,
        protected_paths,
        sha256_bytes,
        strict_test_run_failed,
        verify_candidate_patch,
    )
    from teacher_platform.trace_gate import (  # noqa: E402
        TracePreflightError,
        preflight_trainable_trace,
    )
except ModuleNotFoundError:
    sys.path.insert(0, str(ROOT / "teacher_platform"))
    from generic_trace_replay import (  # type: ignore[no-redef]  # noqa: E402
        ReplayContractError,
        _run_suite,
        build_task_contract,
        parse_patch_paths,
        protected_paths,
        sha256_bytes,
        strict_test_run_failed,
        verify_candidate_patch,
    )
    from trace_gate import (  # type: ignore[no-redef]  # noqa: E402
        TracePreflightError,
        preflight_trainable_trace,
    )

CLAUDE = str(Path.home() / ".local/bin/claude")
# codex CLI is not installed on the box; npx resolves @latest (one-time per
# instance, ~10s on a 300s task). Runs from a scratch cwd with the git guard
# skipped so it never wanders the real repo (all work is via docker exec).
CODEX_NPX = ["npx", "--yes", "@openai/codex@latest"]
CODEX_SCRATCH = "/tmp/codex_teacher_work"
ENV_BOOTSTRAP = "source /opt/miniconda3/bin/activate testbed 2>/dev/null || true; "
CREDIT_EXHAUSTED_ERROR = "out_of_credits"
ARTIFACT_COLLISION_ERROR = "artifact_collision: raw stream or patch already exists"
# Claude prints this even with subtype=success; codex quota strings are UNVERIFIED
# (never hit the wall in probing) — kept generic and confirmed on first real batch.
_CREDIT_TEXTS = ("out of usage credits", "usage limit reached", "rate limit",
                 "quota exceeded", "insufficient_quota", "you've hit your usage limit", "hit your session limit")

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
- The bug-inducing mutation is already applied and committed as current HEAD.
  Treat that HEAD as the task baseline; do not reset or check out its parent.
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


def capture_worktree_patch(cid: str) -> str:
    """Capture tracked and untracked changes without mutating the real Git index."""

    command = (
        "set -e; cd /testbed; "
        "idx=$(mktemp); rm -f \"$idx\"; "
        "trap 'rm -f \"$idx\"' EXIT; "
        "GIT_INDEX_FILE=\"$idx\" git read-tree HEAD; "
        "GIT_INDEX_FILE=\"$idx\" git add -A -- .; "
        "GIT_INDEX_FILE=\"$idx\" git diff --cached --binary --no-renames"
    )
    output, returncode = sh(
        ["docker", "exec", cid, "bash", "-c", command],
        180,
    )
    if returncode != 0:
        raise RuntimeError(f"worktree patch capture failed: {output[-300:]}")
    return output


def establish_mutation_baseline(cid: str, row: dict) -> dict:
    """Apply and commit the SWE-smith bug mutation before the teacher runs."""

    contract = build_task_contract(row)
    mutation_patch = str(row["patch"])
    mutation_paths = parse_patch_paths(mutation_patch)
    if protected_paths(mutation_paths):
        raise ReplayContractError("task mutation touches protected test or harness paths")
    reference_f2p = _run_suite(
        cid,
        contract["f2p_commands"],
        run=sh,
        timeout=900,
        env_bootstrap=ENV_BOOTSTRAP,
    )
    reference_p2p = _run_suite(
        cid,
        contract["p2p_commands"],
        run=sh,
        timeout=900,
        env_bootstrap=ENV_BOOTSTRAP,
    )
    if not reference_f2p["passed"] or not reference_p2p["passed"]:
        raise ReplayContractError("clean reference controls did not pass exactly")
    apply_output, apply_rc = sh(
        [
            "docker",
            "exec",
            "-i",
            cid,
            "bash",
            "-c",
            "cd /testbed && git apply --whitespace=nowarn -",
        ],
        180,
        mutation_patch,
    )
    if apply_rc != 0:
        raise ReplayContractError(
            f"task mutation failed to apply: {apply_output[-300:]}"
        )
    f2p = _run_suite(
        cid,
        contract["f2p_commands"],
        run=sh,
        timeout=900,
        env_bootstrap=ENV_BOOTSTRAP,
    )
    reproduced = bool(f2p["runs"]) and all(
        strict_test_run_failed(
            run["command"],
            run["output_tail"],
            run["returncode"],
        )
        for run in f2p["runs"]
    )
    if not reproduced:
        raise ReplayContractError("task mutation did not reproduce an exact F2P failure")
    mutation_p2p = _run_suite(
        cid,
        contract["p2p_commands"],
        run=sh,
        timeout=900,
        env_bootstrap=ENV_BOOTSTRAP,
    )
    if not mutation_p2p["passed"]:
        raise ReplayContractError("task mutation regressed PASS_TO_PASS controls")
    commit_output, commit_rc = sh(
        [
            "docker",
            "exec",
            cid,
            "bash",
            "-c",
            (
                "cd /testbed && git add -A -- . && "
                "git -c user.name=teacher-platform "
                "-c user.email=teacher-platform@invalid "
                "commit --no-verify -m teacher-task-mutation >/dev/null && "
                "git rev-parse HEAD"
            ),
        ],
        180,
    )
    commit = commit_output.strip().splitlines()[-1] if commit_output.strip() else ""
    if commit_rc != 0 or not re.fullmatch(r"[0-9a-f]{40}(?:[0-9a-f]{24})?", commit):
        raise ReplayContractError(
            f"task mutation baseline commit failed: {commit_output[-300:]}"
        )
    return {
        "task_contract_sha256": contract["contract_sha256"],
        "mutation_patch_sha256": contract["mutation_patch_sha256"],
        "mutation_patch_paths": list(mutation_paths),
        "reference_controls_passed": True,
        "reference_f2p_runs": reference_f2p["runs"],
        "reference_p2p_runs": reference_p2p["runs"],
        "mutation_f2p_reproduced": True,
        "mutation_f2p_runs": f2p["runs"],
        "mutation_p2p_passed": True,
        "mutation_p2p_runs": mutation_p2p["runs"],
        "mutation_baseline_commit": commit,
    }


def credit_exhausted_in_stream(stream: str) -> bool:
    """Detect the credit/quota wall (Claude reports subtype=success anyway)."""
    low = stream.lower()
    if any(t in low for t in _CREDIT_TEXTS):
        return True
    # Robust: any 429 rate/quota/session wall regardless of message wording
    return ('api_error_status": 429' in low) or ('api_error_status":429' in low)


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


def remaining_task_rows(rows: list[dict], existing_results: list[dict]) -> list[dict]:
    """Return task rows not already present in a restartable result ledger."""
    completed_ids: set[str] = set()
    for result in existing_results:
        instance_id = str(result.get("instance_id") or "")
        if not instance_id:
            raise ValueError("existing result is missing instance_id")
        if instance_id in completed_ids:
            raise ValueError(f"duplicate existing result: {instance_id}")
        completed_ids.add(instance_id)
    return [row for row in rows if str(row.get("instance_id") or "") not in completed_ids]


def trailing_credit_streak(results: list[dict]) -> int:
    """Restore the consecutive credit-wall guard when resuming a ledger."""
    streak = 0
    for result in reversed(results):
        if result.get("error") != CREDIT_EXHAUSTED_ERROR:
            break
        streak += 1
    return streak


def collect_one(row: dict, outdir: Path, max_turns: int, model: str, backend: str) -> dict:
    iid = row["instance_id"]
    t0 = time.time()
    raw_path = outdir / f"{iid}.stream.jsonl"
    patch_path = outdir / f"{iid}.patch"
    if raw_path.exists() or patch_path.exists():
        return {
            "instance_id": iid,
            "image": row["image_name"],
            "model": model,
            "backend": backend,
            "executed": False,
            "f2p_pass": False,
            "training_admitted": False,
            "resolved": False,
            "error": ARTIFACT_COLLISION_ERROR,
            "wall_s": round(time.time() - t0, 1),
        }
    out, rc = sh(["docker", "run", "-d", row["image_name"], "sleep", "infinity"], 120)
    if rc:
        return {"instance_id": iid, "error": f"container: {out[-200:]}", "resolved": False}
    cid = out.strip().splitlines()[-1][:12]
    rec = {
        "instance_id": iid,
        "image": row["image_name"],
        "model": model,
        "backend": backend,
        "executed": False,
        "f2p_pass": False,
    }
    try:
        rec.update(establish_mutation_baseline(cid, row))
        prompt = PROMPT.format(problem=row["problem_statement"], cid=cid)
        argv, cwd = build_teacher_cmd(backend, prompt, model, max_turns)
        with open(raw_path, "x") as raw:
            p = subprocess.run(argv, stdout=raw, stderr=subprocess.DEVNULL, text=True,
                               timeout=3600, check=False, cwd=cwd)
        rec["cli_rc"] = p.returncode
        n_turns, cost = parse_teacher_stream(backend, raw_path)
        rec.update(n_assistant_events=n_turns, cost_usd=cost)
        if credit_exhausted_in_stream(raw_path.read_text(errors="replace")):
            rec.update(error=CREDIT_EXHAUSTED_ERROR, resolved=False, skipped=True)
        else:
            diff = capture_worktree_patch(cid)
            rec["patch_len"] = len(diff.strip())
            with patch_path.open("x") as patch_handle:
                patch_handle.write(diff)
            rec["patch_sha256"] = sha256_bytes(diff.encode("utf-8"))
            rec["stream_sha256"] = sha256_bytes(raw_path.read_bytes())
            rec["resolved"] = False
            if diff.strip():
                try:
                    evidence = verify_candidate_patch(
                        row,
                        diff,
                        run=sh,
                        env_bootstrap=ENV_BOOTSTRAP,
                    )
                except ReplayContractError as exc:
                    rec.update(
                        error=f"strict_verification: {str(exc)[:300]}",
                        training_admitted=False,
                        rejection_reasons=["strict_verification_error"],
                    )
                else:
                    rec.update(evidence)
                    rec["executed"] = True
                    rec["f2p_pass"] = evidence["candidate_passed_twice"]
                    rec["p2p_pass"] = evidence["candidate_passed_twice"]
                    try:
                        preflight = preflight_trainable_trace(
                            row,
                            backend,
                            raw_path,
                            rec,
                        )
                    except TracePreflightError as exc:
                        rec.update(
                            training_admitted=False,
                            resolved=False,
                            rejection_reasons=[
                                f"trace_preflight:{exc.code}"
                            ],
                            trace_preflight_error=str(exc)[:300],
                        )
                    else:
                        rec.update(
                            trace_preflight_source_sha256=(
                                preflight.source_sha256
                            ),
                            trace_preflight_retained_steps=(
                                preflight.retained_steps
                            ),
                            resolved=True,
                        )
    except ReplayContractError as exc:
        rec.update(
            error=f"mutation_baseline: {str(exc)[:300]}",
            training_admitted=False,
            resolved=False,
        )
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
    ap.add_argument("--resume", action="store_true",
                    help="Continue an existing results.jsonl, skipping completed instance IDs.")
    a = ap.parse_args()
    model = a.model or ("gpt-5.6-terra" if a.backend == "codex" else "claude-fable-5")

    outdir = Path(a.out)
    outdir.mkdir(parents=True, exist_ok=True)
    rows = [json.loads(l) for l in open(a.tasks)][: a.limit]
    ledger = outdir / "results.jsonl"
    if ledger.exists() and ledger.stat().st_size:
        if not a.resume:
            ap.error(f"{ledger} already exists; pass --resume to continue it")
        results = [json.loads(line) for line in ledger.read_text().splitlines() if line.strip()]
    else:
        results = []
    rows = remaining_task_rows(rows, results)
    credit_streak = trailing_credit_streak(results)
    if credit_streak >= a.max_consecutive_credit_hits:
        print(f"[teacher-pilot] refusing resume: trailing credit streak={credit_streak}", flush=True)
        return 2
    for row in rows:
        rec = collect_one(row, outdir, a.max_turns, model, a.backend)
        results.append(rec)
        print(json.dumps({k: rec.get(k) for k in
                          ("instance_id", "n_assistant_events", "patch_len",
                           "resolved", "cost_usd", "wall_s", "error")}), flush=True)
        with open(ledger, "w") as fh:
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
