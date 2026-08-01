#!/usr/bin/env python3
"""Teacher-trace PLATFORM — one CLI for the full capability-gap data pipeline.

Pipeline (each step is one subcommand; `run` chains them):

  pool     Build a decontaminated candidate pool from SWE-bench/SWE-smith
           (only vetted repos = the repos already in an existing decontaminated
           pool; excludes every already-attempted instance id).
  hard     Select HARD tasks = instances the BASE model FAILED (from a base
           k-sampling results ledger). Teacher traces are only worth collecting
           where base fails; base-solvable tasks teach nothing.
  collect  Run a TEACHER over a task file and F2P-verify each patch in-container.
           Backends: claude / codex (subscription CLIs, $0) or openrouter
           (OPENROUTER_API_KEY, any model). Self-healing: quota-walls only pause,
           disk-guarded, fully resumable.
  merge    Consolidate all collect batches -> results.jsonl + resolved.jsonl
           (trainable positives) and rejected.jsonl (contrastive negatives),
           with copied artifacts and a manifest.
  prepare  Render RESOLVED teacher traces into mini-SWE SFT format (one-bash-
           per-turn, THOUGHT + tool_call, edit-first). Shape-safe: strips the
           `docker exec` wrapper so the student never learns teacher-harness
           shape (the data-mirror regression that sank 4 prior adapters).
  smoke-train  Validate a prepared dataset is trainable: format-loss gate
           (0 failures required) + a 2-step LoRA micro-run.

Backends reuse phaseD_sft/teacher_trace_driver.py (claude/codex) and add an
in-process openrouter agent loop with the identical record schema, so hard /
merge / prepare treat all three uniformly.

Examples:
  python phaseD_sft/teacher_platform.py pool  --out data/pool.jsonl --per-repo 48
  python phaseD_sft/teacher_platform.py hard  --labels runs/base_k4/results.jsonl \
        --pool data/pool.jsonl --out data/hard.jsonl
  python phaseD_sft/teacher_platform.py collect --tasks data/hard.jsonl \
        --out-dir runs/teacher_claude --backend claude --loop
  python phaseD_sft/teacher_platform.py merge  --glob 'runs/teacher_*' \
        --out-dir runs/teacher_merged
  python phaseD_sft/teacher_platform.py prepare --merged runs/teacher_merged \
        --out data/teacher_sft.jsonl
  python phaseD_sft/teacher_platform.py smoke-train --data data/teacher_sft.jsonl
"""
from __future__ import annotations

import argparse
import glob as globmod
import hashlib
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
import time
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent            # teacher_platform/
ROOT = HERE.parent                                # repo root
sys.path.insert(0, str(ROOT / "phaseD_sft"))      # teacher_trace_driver lives here
import teacher_trace_driver as ttd  # noqa: E402  (claude/codex collect + credit detector + F2P)
if __package__:
    from .generic_trace_replay import (
        ReplayContractError,
        admission_is_exact,
        assess_controls,
        build_task_contract,
        canonical_json_bytes,
        parse_patch_paths,
        sha256_bytes,
        verify_candidate_patch,
    )
    from .success_trace_distill import (
        ReplayStep,
        ReplayTrace,
        command_mutates_source,
        distill_success_path,
    )
    from .trace_gate import (
        NormalizedTrace,
        TraceNormalizationError,
        TracePreflightError,
        _strip_docker_exec,
        normalize_steps,
        preflight_trainable_trace,
    )
else:
    from generic_trace_replay import (
        ReplayContractError,
        admission_is_exact,
        assess_controls,
        build_task_contract,
        canonical_json_bytes,
        parse_patch_paths,
        sha256_bytes,
        verify_candidate_patch,
    )
    from success_trace_distill import (
        ReplayStep,
        ReplayTrace,
        command_mutates_source,
        distill_success_path,
    )
    from trace_gate import (
        NormalizedTrace,
        TraceNormalizationError,
        TracePreflightError,
        _strip_docker_exec,
        normalize_steps,
        preflight_trainable_trace,
    )

DEFAULT_SEED = 4218


# --------------------------------------------------------------------------- #
# helpers                                                                      #
# --------------------------------------------------------------------------- #
def _read_jsonl(path):
    with open(path) as fh:
        return [json.loads(line) for line in fh if line.strip()]


def _write_jsonl(path, rows):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")


def _sha256_path(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _publish_directory_atomic(out: Path, build) -> None:
    """Build a directory privately and atomically publish it without overwrite."""

    from phaseD_sft.filter_edit_decisive_dataset import _rename_noreplace

    out = Path(out)
    if os.path.lexists(out):
        raise FileExistsError(f"refusing to overwrite {out}")
    out.parent.mkdir(parents=True, exist_ok=True)
    stage_root = Path(tempfile.mkdtemp(prefix=f".{out.name}.", dir=out.parent))
    payload = stage_root / "payload"
    try:
        build(payload)
        _rename_noreplace(payload, out)
        stage_root.rmdir()
    except BaseException:
        shutil.rmtree(stage_root, ignore_errors=True)
        raise


def _is_real_attempt(rec: dict) -> bool:
    """Return whether a ledger row contains real teacher work.

    Claude can emit a long stream before reporting an explicit credit wall, so
    event count alone cannot override ``out_of_credits`` when no patch exists.
    """
    patch_len = int(rec.get("patch_len", 0) or 0)
    if patch_len > 0:
        return True
    if rec.get("error") == ttd.CREDIT_EXHAUSTED_ERROR:
        return False
    return int(rec.get("n_assistant_events", 0) or 0) > 2


def _non_real_attempt_reason(rec: dict) -> str | None:
    if _is_real_attempt(rec):
        return None
    if rec.get("error") == ttd.CREDIT_EXHAUSTED_ERROR:
        return "credit_wall"
    return "empty_or_dud"


def _consumes_collection_task(rec: dict) -> bool:
    if _is_real_attempt(rec):
        return True
    error = str(rec.get("error") or "")
    return (
        error.startswith("mutation_baseline:")
        or error == ttd.ARTIFACT_COLLISION_ERROR
    )


def _attempted_ids(*paths_or_globs) -> set:
    ids = set()
    for p in paths_or_globs:
        for f in ([p] if os.path.exists(p) else globmod.glob(p)):
            try:
                for rec in _read_jsonl(f):
                    if "instance_id" in rec:
                        ids.add(rec["instance_id"])
            except (FileNotFoundError, json.JSONDecodeError):
                pass
    return ids


# --------------------------------------------------------------------------- #
# pool                                                                         #
# --------------------------------------------------------------------------- #
def cmd_pool(a) -> int:
    """Build a decontaminated candidate pool from SWE-smith, restricted to the
    repos already vetted in --vetted-pool (which inherits Lite-30/hard30
    exclusion), minus every already-attempted instance id."""
    from datasets import load_dataset

    vetted_rows = _read_jsonl(a.vetted_pool)
    vetted_repos = {r["repo"] for r in vetted_rows}
    attempted = _attempted_ids(*a.exclude)
    attempted |= {r["instance_id"] for r in vetted_rows}  # base already labeled these
    print(f"[pool] vetted_repos={len(vetted_repos)} excluded_ids={len(attempted)}", flush=True)

    ds = load_dataset(a.dataset, split=a.split)
    cols = ("instance_id", "patch", "FAIL_TO_PASS", "PASS_TO_PASS", "image_name", "repo", "problem_statement")
    eligible = [
        {k: r[k] for k in cols}
        for r in ds
        if r["repo"] in vetted_repos and r["instance_id"] not in attempted
    ]
    # deterministic per-repo cap for a balanced pool
    import random

    byrepo = defaultdict(list)
    for r in eligible:
        byrepo[r["repo"]].append(r)
    rnd = random.Random(a.seed)
    out = []
    for repo in sorted(byrepo):
        rows = sorted(byrepo[repo], key=lambda r: r["instance_id"])
        rnd.shuffle(rows)
        out.extend(rows[: a.per_repo] if a.per_repo else rows)
    rnd.shuffle(out)
    _write_jsonl(a.out, out)
    man = {
        "source": a.dataset,
        "vetted_repos": sorted(vetted_repos),
        "eligible_total": len(eligible),
        "selected": len(out),
        "per_repo_cap": a.per_repo,
        "excluded_ids": len(attempted),
        "per_repo_selected": dict(Counter(r["repo"] for r in out)),
        "decontam": "repo-level, inherited from vetted pool (Lite-30/hard30 excluded)",
    }
    Path(str(a.out) + ".manifest.json").write_text(json.dumps(man, indent=1))
    print(f"[pool] wrote {len(out)} tasks -> {a.out} (eligible universe {len(eligible)})", flush=True)
    return 0


# --------------------------------------------------------------------------- #
# hard                                                                         #
# --------------------------------------------------------------------------- #
def cmd_hard(a) -> int:
    """Select base-FAILED tasks. --labels is a base k-sampling ledger with
    per-rollout {source_instance_id|instance_id, resolved}. An instance is HARD
    iff it was NEVER resolved across its k samples. Joins full task metadata
    from --pool so the output feeds `collect` directly."""
    labels = _read_jsonl(a.labels)
    grouped = defaultdict(list)
    for r in labels:
        iid = r.get("source_instance_id") or r.get("instance_id")
        if not iid:
            continue
        grouped[iid].append(r)
    complete = set()
    solved = set()
    invalid = Counter()
    for instance_id, attempts in grouped.items():
        if len(attempts) != a.samples_per_instance:
            invalid["wrong_sample_count"] += 1
            continue
        if any(
            type(row.get("resolved")) is not bool
            or row.get("error")
            or row.get("pull_failed")
            for row in attempts
        ):
            invalid["invalid_or_infrastructure_result"] += 1
            continue
        complete.add(instance_id)
        if any(row["resolved"] is True for row in attempts):
            solved.add(instance_id)
    failed = complete - solved
    pool = {r["instance_id"]: r for r in _read_jsonl(a.pool)}
    exclude = _attempted_ids(*a.exclude) if a.exclude else set()
    hard = [pool[i] for i in sorted(failed) if i in pool and i not in exclude]
    _write_jsonl(a.out, hard)
    man = {
        "labels": str(a.labels),
        "labels_sha256": _sha256_path(a.labels),
        "samples_per_instance": a.samples_per_instance,
        "instances_seen": len(grouped),
        "instances_with_complete_clean_samples": len(complete),
        "invalid_instances": sum(invalid.values()),
        "invalid_reasons": dict(invalid),
        "base_solved": len(solved),
        "base_failed": len(failed),
        "hard_with_metadata": len(hard),
        "excluded": len(failed) - len(hard),
        "note": "hard = base failed ALL k samples; these resolved by a teacher = capability-gap gold",
    }
    Path(str(a.out) + ".manifest.json").write_text(json.dumps(man, indent=1))
    print(f"[hard] base seen={len(grouped)} complete={len(complete)} "
          f"solved={len(solved)} FAILED={len(failed)} "
          f"-> hard tasks with metadata={len(hard)} -> {a.out}", flush=True)
    return 0


# --------------------------------------------------------------------------- #
# collect  (claude / codex via driver; openrouter in-process; self-healing)   #
# --------------------------------------------------------------------------- #
def _docker_free_gib(path="/var/lib/docker") -> int:
    try:
        out = subprocess.run(["df", "--output=avail", "-BG", path], capture_output=True, text=True)
        return int(re.sub(r"\D", "", out.stdout.strip().splitlines()[-1]))
    except Exception:
        return 10 ** 6


def _probe_quota(backend: str, model: str) -> bool:
    """Cheap readiness probe. Subscription CLIs: one-token call. openrouter: key present."""
    if backend == "openrouter":
        return bool(os.environ.get("OPENROUTER_API_KEY"))
    if backend == "claude":
        out = subprocess.run([ttd.CLAUDE, "-p", "Reply exactly READY.", "--model", model],
                             capture_output=True, text=True, timeout=120)
        return "READY" in (out.stdout + out.stderr)
    if backend == "codex":
        return True  # codex npx has no cheap probe; the guard catches walls in-run
    return False


def _attempted_from_run_globs(*run_globs) -> set:
    """Union of instance_ids REALLY attempted across any matching run dirs.
    This is the cross-run no-overlap mechanism: a new campaign excludes every
    problem any prior run already worked."""
    attempted = set()
    for g in run_globs:
        if not g:
            continue
        pat = g if g.endswith("results.jsonl") else g.rstrip("/") + "*/results.jsonl"
        for f in globmod.glob(pat):
            try:
                for rec in _read_jsonl(f):
                    if _consumes_collection_task(rec):
                        attempted.add(rec["instance_id"])
            except (FileNotFoundError, json.JSONDecodeError):
                pass
    return attempted


_FATAL_ERR_MARKERS = ("not a valid model", "invalid model", "no such model",
                      "401", "invalid api key", "no auth credentials", "403 ")


def _is_fatal_error(err) -> bool:
    """Config errors that will fail EVERY task identically (bad model id, bad
    key) — stop immediately instead of burning the whole task list."""
    low = str(err or "").lower()
    return bool(err) and any(m in low for m in _FATAL_ERR_MARKERS)


def _is_rate_limited_error(err) -> bool:
    """True for a provider or router 429 that should pause a loop pass."""
    low = str(err or "").lower()
    return "429" in low and ("rate limit" in low or "rate-limit" in low)


def _remaining(tasks_path: str, *run_globs) -> list:
    pool = _read_jsonl(tasks_path)
    attempted = _attempted_from_run_globs(*run_globs)
    return [r for r in pool if r["instance_id"] not in attempted]


def cmd_collect(a) -> int:
    model = a.model or {"codex": "gpt-5.6-terra", "claude": "claude-fable-5",
                        "openrouter": "anthropic/claude-3.7-sonnet"}[a.backend]
    out_root = Path(a.out_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    # scan sibling batch dirs (out_dir plus <out_dir>_run*) for real attempts;
    # --exclude-runs adds OTHER campaigns so problems never overlap across runs.
    out_glob = str(out_root.parent / (out_root.name + "*/results.jsonl"))
    all_globs = [out_glob, *a.exclude_runs]
    cross_run_done = _attempted_from_run_globs(*a.exclude_runs)

    def one_pass(tasks_file: str, outdir: Path) -> str:
        """Run once and classify why the pass stopped."""
        outdir.mkdir(parents=True, exist_ok=True)
        rows = _read_jsonl(tasks_file)
        ledger = outdir / "results.jsonl"
        results = _read_jsonl(ledger) if ledger.exists() and ledger.stat().st_size else []
        rows = ttd.remaining_task_rows(rows, results)
        rows = [r for r in rows if r["instance_id"] not in cross_run_done]
        streak = ttd.trailing_credit_streak(results)
        if streak >= a.max_consecutive_credit_hits:
            return "credit_walled"
        for row in rows:
            if a.backend == "openrouter":
                rec = collect_one_openrouter(row, outdir, a.max_turns, model)
            else:
                rec = ttd.collect_one(row, outdir, a.max_turns, model, a.backend)
            results.append(rec)
            print(json.dumps({k: rec.get(k) for k in
                              ("instance_id", "n_assistant_events", "patch_len", "resolved", "error")}),
                  flush=True)
            _replace_jsonl_atomic(ledger, results)
            if _is_fatal_error(rec.get("error")):
                raise RuntimeError(f"fatal config error (bad model id / key?), aborting: {rec.get('error')}")
            if _is_rate_limited_error(rec.get("error")):
                print("[collect] upstream rate limited -> pausing pass", flush=True)
                return "rate_limited"
            streak = ttd.next_credit_streak(streak, rec.get("error"))
            if streak >= a.max_consecutive_credit_hits:
                print(f"[collect] {streak} consecutive credit hits -> pausing pass", flush=True)
                return "credit_walled"
        return "completed"

    if not a.loop:
        one_pass(a.tasks, out_root)
    else:
        walls = 0
        it = 0
        while True:
            it += 1
            subprocess.run(["docker", "container", "prune", "-f"], capture_output=True)
            if _docker_free_gib() < a.floor_gib + 2:
                subprocess.run(["docker", "image", "prune", "-f"], capture_output=True)
            rem = _remaining(a.tasks, *all_globs)
            print(f"[collect] iter={it} remaining={len(rem)} walls={walls} "
                  f"free={_docker_free_gib()}G", flush=True)
            if not rem:
                print("[collect] POOL EXHAUSTED", flush=True)
                break
            if walls >= a.max_walls:
                print(f"[collect] GAVE UP after {walls} walls, remaining={len(rem)}", flush=True)
                break
            if _probe_quota(a.backend, model):
                tmp = out_root.parent / f".{out_root.name}_remaining.jsonl"
                _write_jsonl(tmp, rem)
                pass_status = one_pass(
                    str(tmp),
                    out_root.parent / f"{out_root.name}_run{it}",
                )
                if pass_status == "rate_limited":
                    walls += 1
                    if walls < a.max_walls:
                        print(f"[collect] rate limited ({walls}), sleeping {a.rate_limit_sleep}s", flush=True)
                        time.sleep(a.rate_limit_sleep)
                elif pass_status == "credit_walled":
                    walls += 1
                    if walls < a.max_walls:
                        print(f"[collect] credit walled ({walls}), sleeping {a.wall_sleep}s", flush=True)
                        time.sleep(a.wall_sleep)
                else:
                    walls = 0
            else:
                walls += 1
                print(f"[collect] quota walled ({walls}), sleeping {a.wall_sleep}s", flush=True)
                time.sleep(a.wall_sleep)
    # tally
    done = _remaining(a.tasks, *all_globs)
    total = len(_read_jsonl(a.tasks))
    print(f"[collect] attempted+excluded={total - len(done)}/{total}", flush=True)
    return 0


def collect_one_openrouter(row: dict, outdir: Path, max_turns: int, model: str) -> dict:
    """In-process agent loop via OpenRouter (OpenAI-compatible). Same record
    schema + raw stream + F2P scoring as ttd.collect_one, so downstream steps
    treat it identically. The model drives bash; we docker-exec into /testbed."""
    from openai import OpenAI

    iid = row["instance_id"]
    t0 = time.time()
    raw_path = outdir / f"{iid}.stream.jsonl"
    patch_path = outdir / f"{iid}.patch"
    if raw_path.exists() or patch_path.exists():
        return {
            "instance_id": iid,
            "image": row["image_name"],
            "model": model,
            "backend": "openrouter",
            "executed": False,
            "f2p_pass": False,
            "p2p_pass": False,
            "training_admitted": False,
            "resolved": False,
            "error": ttd.ARTIFACT_COLLISION_ERROR,
            "wall_s": round(time.time() - t0, 1),
        }
    out, rc = ttd.sh(["docker", "run", "-d", row["image_name"], "sleep", "infinity"], 300)
    # Robust cid parse: docker may emit WARNING/platform lines (e.g. amd64 image
    # on an arm64 host) that would otherwise be mistaken for the container id.
    cid = next((m.group(0) for line in reversed(out.splitlines())
                for m in [re.fullmatch(r"[0-9a-f]{12,64}", line.strip())] if m), "")
    if rc or not cid:
        return {"instance_id": iid, "error": f"container-start failed (wrong host/arch?): {out[-200:]}",
                "resolved": False}
    cid = cid[:12]
    # confirm the container is actually running before driving it
    st, _ = ttd.sh(["docker", "inspect", "-f", "{{.State.Running}}", cid], 30)
    if "true" not in st.lower():
        ttd.sh(["docker", "rm", "-f", cid], 30)
        return {"instance_id": iid, "error": f"container not running: {st[-120:]}", "resolved": False}
    rec = {
        "instance_id": iid,
        "image": row["image_name"],
        "model": model,
        "backend": "openrouter",
        "executed": False,
        "f2p_pass": False,
        "p2p_pass": False,
        "training_admitted": False,
    }
    stream = open(raw_path, "x")
    events = 0
    try:
        rec.update(ttd.establish_mutation_baseline(cid, row))
        client = OpenAI(base_url="https://openrouter.ai/api/v1",
                        api_key=os.environ["OPENROUTER_API_KEY"])
        # openrouter runs bash directly in /testbed (no docker-exec wrapper in the
        # rendered convention); we exec on its behalf and hand back stdout.
        sysmsg = ttd.PROMPT.format(problem=row["problem_statement"], cid="LOCAL").replace(
            f'docker exec LOCAL bash -c "cd /testbed && <your command>"', "bash: <your command> (run from /testbed)"
        )
        messages = [{"role": "system", "content": sysmsg},
                    {"role": "user", "content": "Begin. Emit one bash command per turn."}]
        tools = [{"type": "function", "function": {
            "name": "bash",
            "description": "Run a bash command in /testbed and return its output.",
            "parameters": {"type": "object", "properties": {"command": {"type": "string"}},
                           "required": ["command"]}}}]
        for _ in range(max_turns):
            resp = client.chat.completions.create(model=model, messages=messages, tools=tools,
                                                  temperature=0.7, max_tokens=2048)
            msg = resp.choices[0].message
            stream.write(json.dumps({"role": "assistant", "content": msg.content,
                                     "tool_calls": [tc.function.arguments for tc in (msg.tool_calls or [])]}) + "\n")
            events += 1
            if not msg.tool_calls:
                break
            messages.append({"role": "assistant", "content": msg.content or "",
                             "tool_calls": [{"id": tc.id, "type": "function",
                                             "function": {"name": "bash", "arguments": tc.function.arguments}}
                                            for tc in msg.tool_calls]})
            for tc in msg.tool_calls:
                try:
                    cmd = json.loads(tc.function.arguments).get("command", "")
                except json.JSONDecodeError:
                    cmd = ""
                obs, tool_returncode = ttd.sh(
                    [
                        "docker",
                        "exec",
                        cid,
                        "bash",
                        "-c",
                        f"cd /testbed && {cmd}",
                    ],
                    300,
                )
                obs = obs[-4000:]
                stream.write(json.dumps({
                    "role": "tool",
                    "command": cmd,
                    "observation": obs,
                    "returncode": tool_returncode,
                }) + "\n")
                messages.append({"role": "tool", "tool_call_id": tc.id, "content": obs})
        stream.close()
        rec["n_assistant_events"] = events
        rec["cost_usd"] = 0.0
        diff = ttd.capture_worktree_patch(cid)
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
                    run=ttd.sh,
                    env_bootstrap=ttd.ENV_BOOTSTRAP,
                )
            except ReplayContractError as exc:
                rec.update(
                    error=f"strict_verification: {str(exc)[:300]}",
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
                        "openrouter",
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
    except Exception as exc:  # noqa: BLE001
        if not stream.closed:
            stream.close()
        rec.update(error=f"openrouter: {str(exc)[:160]}", resolved=rec.get("resolved", False))
    finally:
        ttd.sh(["docker", "rm", "-f", cid], 60)
    rec["wall_s"] = round(time.time() - t0, 1)
    return rec


# --------------------------------------------------------------------------- #
# merge                                                                        #
# --------------------------------------------------------------------------- #
def _resolve_merge_ledgers(
    glob_pattern: str | None,
    inputs: list[str],
) -> list[str]:
    if inputs:
        if glob_pattern:
            raise ValueError("merge accepts either --glob or --input")
        ledgers = []
        for raw in inputs:
            path = Path(raw).resolve()
            ledger = path / "results.jsonl" if path.is_dir() else path
            if not ledger.is_file():
                raise ValueError(f"merge input ledger is missing: {ledger}")
            ledgers.append(str(ledger))
    else:
        if not glob_pattern:
            raise ValueError("merge requires --glob or --input")
        ledgers = [
            str(Path(path).resolve())
            for path in (
                globmod.glob(
                    glob_pattern.rstrip("/") + "/results.jsonl"
                )
                if not glob_pattern.endswith("results.jsonl")
                else globmod.glob(glob_pattern)
            )
        ]
    return sorted(dict.fromkeys(ledgers))


def _merge_selection_exclusions(
    artifacts: list[str],
) -> tuple[set[str], list[dict[str, str]]]:
    instance_ids: set[str] = set()
    bindings = []
    for raw in artifacts:
        path = Path(raw).resolve()
        if not path.is_file():
            raise ValueError(
                f"merge selection exclusion is missing: {path}"
            )
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            values = payload if isinstance(payload, list) else [payload]
        except json.JSONDecodeError:
            values = _read_jsonl(path)
        for value in values:
            if isinstance(value, str):
                instance_ids.add(value)
                continue
            if not isinstance(value, dict):
                raise ValueError(
                    "merge selection exclusion contains an invalid value"
                )
            for key in ("instance_ids", "selected_ids"):
                listed = value.get(key, [])
                if not isinstance(listed, list) or any(
                    not isinstance(item, str) for item in listed
                ):
                    raise ValueError(
                        f"invalid merge selection exclusion field: {key}"
                    )
                instance_ids.update(listed)
            for key in ("instance_id", "source_instance_id"):
                item = value.get(key)
                if isinstance(item, str):
                    instance_ids.add(item)
        bindings.append({
            "path": str(path),
            "sha256": _sha256_path(path),
        })
    return instance_ids, bindings


def cmd_merge(a) -> int:
    real = {}
    ledgers = _resolve_merge_ledgers(
        getattr(a, "glob", None),
        list(getattr(a, "inputs", []) or []),
    )
    excluded_ids, exclusion_bindings = _merge_selection_exclusions(
        list(getattr(a, "exclude_instance_ids", []) or []),
    )
    for ledger in ledgers:
        batch = Path(ledger).parent.name
        src = Path(ledger).parent
        for rec in _read_jsonl(ledger):
            if _is_real_attempt(rec):
                rec = dict(rec, batch=batch, _src=str(src))
                real[rec["instance_id"]] = rec  # last real attempt wins
    selection_excluded = sorted(set(real) & excluded_ids)
    rows = [
        row
        for instance_id, row in real.items()
        if instance_id not in excluded_ids
    ]
    res = [r for r in rows if admission_is_exact(r)]
    rejected = []
    for row in rows:
        if row in res:
            continue
        value = dict(row)
        reasons = list(value.get("rejection_reasons") or [])
        if value.get("resolved") and "missing_exact_admission_evidence" not in reasons:
            reasons.append("missing_exact_admission_evidence")
        value["merge_rejection_reasons"] = reasons or ["not_training_admitted"]
        rejected.append(value)
    out = Path(a.out_dir)
    manifest_holder = {}

    def build(stage: Path) -> None:
        stage.mkdir(mode=0o700)
        clean_rows = [{k: v for k, v in row.items() if k != "_src"} for row in rows]
        clean_res = [{k: v for k, v in row.items() if k != "_src"} for row in res]
        clean_rejected = [
            {k: v for k, v in row.items() if k != "_src"} for row in rejected
        ]
        _write_jsonl(stage / "results.jsonl", clean_rows)
        _write_jsonl(stage / "resolved.jsonl", clean_res)
        _write_jsonl(stage / "rejected.jsonl", clean_rejected)
        bindings = []
        for row in res:
            stream = Path(row["_src"]) / f"{row['instance_id']}.stream.jsonl"
            patch = Path(row["_src"]) / f"{row['instance_id']}.patch"
            if not stream.is_file() or not patch.is_file():
                raise ReplayContractError(
                    f"admitted artifact set is incomplete: {row['instance_id']}"
                )
            actual_stream = _sha256_path(stream)
            actual_patch = _sha256_path(patch)
            if (
                actual_stream != row.get("stream_sha256")
                or actual_patch != row.get("patch_sha256")
                or actual_patch != row.get("candidate_patch_sha256")
                or list(parse_patch_paths(patch.read_text(encoding="utf-8")))
                != row.get("candidate_patch_paths")
            ):
                raise ReplayContractError(
                    f"admitted artifact hash mismatch: {row['instance_id']}"
                )
            shutil.copy2(stream, stage / stream.name)
            shutil.copy2(patch, stage / patch.name)
            bindings.append({
                "instance_id": row["instance_id"],
                "batch": row["batch"],
                "stream_sha256": actual_stream,
                "patch_sha256": actual_patch,
                "admission_evidence_sha256": row["admission_evidence_sha256"],
                "task_contract_sha256": row["task_contract_sha256"],
            })
        rejected_dir = stage / "rejected"
        rejected_dir.mkdir()
        rejected_copied = 0
        for row in rejected:
            for ext in (".stream.jsonl", ".patch"):
                source = Path(row["_src"]) / f"{row['instance_id']}{ext}"
                if source.is_file():
                    shutil.copy2(source, rejected_dir / source.name)
                    rejected_copied += 1
        manifest = {
            "schema_version": 2,
            "complete": True,
            "attempted": len(rows),
            "resolved": len(res),
            "training_admitted": len(res),
            "rejected": len(rejected),
            "resolve_rate": round(len(res) / max(len(rows), 1), 3),
            "per_batch": dict(Counter(row["batch"] for row in rows)),
            "resolved_per_batch": dict(Counter(row["batch"] for row in res)),
            "rejected_per_batch": dict(Counter(row["batch"] for row in rejected)),
            "selection_excluded": len(selection_excluded),
            "selection_excluded_instance_ids": selection_excluded,
            "selection_exclusion_artifacts": exclusion_bindings,
            "input_ledgers": [
                {"path": ledger, "sha256": _sha256_path(ledger)}
                for ledger in ledgers
            ],
            "artifact_bindings": bindings,
            "results_sha256": _sha256_path(stage / "results.jsonl"),
            "resolved_sha256": _sha256_path(stage / "resolved.jsonl"),
            "rejected_sha256": _sha256_path(stage / "rejected.jsonl"),
            "artifacts_copied": len(bindings) * 2,
            "rejected_streams_copied": rejected_copied,
        }
        (stage / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n"
        )
        manifest_holder.update(manifest)

    _publish_directory_atomic(out, build)
    man = manifest_holder
    print(f"[merge] attempted={len(rows)} resolved={len(res)} rejected={len(rejected)} "
          f"({man['resolve_rate']:.0%}) -> {out}", flush=True)
    return 0


# --------------------------------------------------------------------------- #
# revalidate  (strict fresh controls for raw/legacy extraction artifacts)      #
# --------------------------------------------------------------------------- #
def _replace_jsonl_atomic(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def _revalidation_contract(
    *,
    selected: dict[str, tuple[dict, Path]],
    taskmeta: dict[str, dict],
    ledgers: list[str],
    task_inputs: list[dict],
    exclusion_evidence: list[dict],
    test_timeout: int,
) -> dict:
    cases = []
    for instance_id in sorted(selected):
        _raw, source = selected[instance_id]
        meta = taskmeta.get(instance_id)
        stream = source / f"{instance_id}.stream.jsonl"
        patch = source / f"{instance_id}.patch"
        task_contract = None
        image_id = None
        contract_error = None
        try:
            if meta is None:
                raise ReplayContractError("missing task metadata")
            task_contract = build_task_contract(meta)
            output, returncode = ttd.sh(
                [
                    "docker",
                    "image",
                    "inspect",
                    "--format",
                    "{{.Id}}",
                    task_contract["image_name"],
                ],
                60,
            )
            image_id = output.strip()
            if returncode != 0 or not image_id:
                raise ReplayContractError("image identity is unavailable")
        except (OSError, ReplayContractError, subprocess.TimeoutExpired) as exc:
            contract_error = f"{type(exc).__name__}: {str(exc)[:300]}"
        cases.append({
            "instance_id": instance_id,
            "source_run": str(source),
            "stream_sha256": _sha256_path(stream) if stream.is_file() else None,
            "patch_sha256": _sha256_path(patch) if patch.is_file() else None,
            "task_contract": task_contract,
            "image_id": image_id,
            "contract_error": contract_error,
        })
    contract = {
        "schema_version": 1,
        "input_ledgers": [
            {"path": ledger, "sha256": _sha256_path(ledger)}
            for ledger in ledgers
        ],
        "task_inputs": task_inputs,
        "exclusion_artifacts": exclusion_evidence,
        "test_timeout": test_timeout,
        "implementation": {
            "platform_sha256": _sha256_path(Path(__file__)),
            "replay_sha256": _sha256_path(HERE / "generic_trace_replay.py"),
        },
        "cases": cases,
    }
    contract["contract_sha256"] = sha256_bytes(canonical_json_bytes(contract))
    return contract


def _resolve_revalidation_ledgers(values) -> list[str]:
    patterns = [values] if isinstance(values, (str, os.PathLike)) else list(values)
    ledgers: set[str] = set()
    for raw in patterns:
        value = os.fspath(raw)
        pattern = (
            value
            if value.endswith("results.jsonl")
            else value.rstrip("/") + "/results.jsonl"
        )
        ledgers.update(globmod.glob(pattern))
    if not ledgers:
        raise ReplayContractError("no extraction ledgers matched")
    return sorted(ledgers)


_COLLECTOR_PREFLIGHT_EVIDENCE_ERROR = (
    "normalization_error: exact strict admission evidence is required"
)
_COLLECTOR_PREFLIGHT_REJECTION = [
    "trace_preflight:normalization_error"
]
_MUTATION_ADMISSION_FIELDS = (
    "executed",
    "f2p_pass",
    "reference_controls_passed",
    "mutation_f2p_reproduced",
    "mutation_p2p_passed",
)


def _restore_collector_preflight_evidence(record: dict) -> dict:
    if (
        record.get("training_admitted") is not False
        or record.get("rejection_reasons")
        != _COLLECTOR_PREFLIGHT_REJECTION
        or record.get("trace_preflight_error")
        != _COLLECTOR_PREFLIGHT_EVIDENCE_ERROR
        or any(record.get(key) is not True for key in _MUTATION_ADMISSION_FIELDS)
    ):
        raise ReplayContractError(
            "record is not the exact collector preflight evidence bug"
        )
    controls = record.get("controls")
    protected_patch_paths = record.get("protected_patch_paths")
    if not isinstance(controls, dict) or not isinstance(
        protected_patch_paths,
        list,
    ):
        raise ReplayContractError("strict control evidence is incomplete")
    assessment = assess_controls(
        controls,
        protected_patch_paths=protected_patch_paths,
    )
    if assessment.get("training_admitted") is not True:
        raise ReplayContractError("strict verification did not admit the patch")
    restored = dict(record)
    restored.update(assessment)
    if not admission_is_exact(restored):
        raise ReplayContractError(
            "restored strict admission evidence is not exact"
        )
    return restored


def cmd_repreflight(a) -> int:
    """Recover immutable raw artifacts rejected by the collector evidence bug."""

    ledger = Path(a.input)
    if not ledger.is_file():
        raise ReplayContractError(f"input ledger is missing: {ledger}")
    source = ledger.parent
    source_rows = _read_jsonl(ledger)
    identities = [row.get("instance_id") for row in source_rows]
    if (
        any(not isinstance(instance_id, str) or not instance_id for instance_id in identities)
        or len(set(identities)) != len(identities)
    ):
        raise ReplayContractError("input ledger identities are invalid or duplicated")

    taskmeta = {}
    task_inputs = []
    for value in a.tasks:
        files = [value] if os.path.isfile(value) else sorted(globmod.glob(value))
        if not files:
            raise ReplayContractError(
                f"task metadata input did not resolve: {value}"
            )
        for filename in files:
            task_inputs.append({
                "path": filename,
                "sha256": _sha256_path(filename),
            })
            for row in _read_jsonl(filename):
                instance_id = row.get("instance_id")
                if not isinstance(instance_id, str) or not instance_id:
                    raise ReplayContractError(
                        f"task metadata is missing instance_id: {filename}"
                    )
                previous = taskmeta.get(instance_id)
                if (
                    previous is not None
                    and canonical_json_bytes(previous)
                    != canonical_json_bytes(row)
                ):
                    raise ReplayContractError(
                        f"conflicting task metadata: {instance_id}"
                    )
                taskmeta[instance_id] = row

    results = []
    for row in source_rows:
        instance_id = row["instance_id"]
        stream = source / f"{instance_id}.stream.jsonl"
        patch = source / f"{instance_id}.patch"
        result = dict(row)
        result["source_record_sha256"] = sha256_bytes(
            canonical_json_bytes(row)
        )
        try:
            task = taskmeta.get(instance_id)
            if task is None:
                raise ReplayContractError("missing task metadata")
            if not stream.is_file() or not patch.is_file():
                raise ReplayContractError("raw artifact set is incomplete")
            if (
                _sha256_path(stream) != row.get("stream_sha256")
                or _sha256_path(patch) != row.get("patch_sha256")
                or row.get("patch_sha256")
                != row.get("candidate_patch_sha256")
            ):
                raise ReplayContractError("raw artifact hash mismatch")
            result = _restore_collector_preflight_evidence(result)
            preflight = preflight_trainable_trace(
                task,
                result.get("backend", "claude"),
                stream,
                result,
            )
            result.pop("trace_preflight_error", None)
            result.update(
                resolved=True,
                trace_preflight_source_sha256=preflight.source_sha256,
                trace_preflight_retained_steps=preflight.retained_steps,
            )
        except TracePreflightError as exc:
            result.update(
                training_admitted=False,
                resolved=False,
                rejection_reasons=[f"trace_preflight:{exc.code}"],
                trace_preflight_error=str(exc)[:300],
            )
        except (
            OSError,
            ReplayContractError,
            subprocess.TimeoutExpired,
            ValueError,
        ) as exc:
            result.update(
                training_admitted=False,
                resolved=False,
                rejection_reasons=["trace_preflight_recovery_error"],
                trace_preflight_error=(
                    f"{type(exc).__name__}: {str(exc)[:300]}"
                ),
            )
        results.append(result)

    admitted = [row for row in results if admission_is_exact(row)]
    rejected = [row for row in results if row not in admitted]
    out = Path(a.out_dir)
    manifest_holder = {}

    def build(stage: Path) -> None:
        stage.mkdir(mode=0o700)
        _write_jsonl(stage / "source_results.jsonl", source_rows)
        _write_jsonl(stage / "results.jsonl", results)
        _write_jsonl(stage / "resolved.jsonl", admitted)
        _write_jsonl(stage / "rejected.jsonl", rejected)
        admitted_ids = {row["instance_id"] for row in admitted}
        rejected_dir = stage / "rejected"
        rejected_dir.mkdir()
        for row in results:
            target = stage if row["instance_id"] in admitted_ids else rejected_dir
            for extension in (".stream.jsonl", ".patch"):
                artifact = source / f"{row['instance_id']}{extension}"
                if artifact.is_file():
                    shutil.copy2(artifact, target / artifact.name)
        bindings = [
            {
                "instance_id": row["instance_id"],
                "stream_sha256": row["stream_sha256"],
                "patch_sha256": row["patch_sha256"],
                "admission_evidence_sha256": row[
                    "admission_evidence_sha256"
                ],
                "task_contract_sha256": row["task_contract_sha256"],
                "trace_preflight_source_sha256": row[
                    "trace_preflight_source_sha256"
                ],
                "source_record_sha256": row["source_record_sha256"],
            }
            for row in admitted
        ]
        manifest = {
            "schema_version": 1,
            "artifact_type": "trace_preflight_recovery",
            "complete": True,
            "selected": len(results),
            "training_admitted": len(admitted),
            "rejected": len(rejected),
            "source_ledger": {
                "path": str(ledger),
                "sha256": _sha256_path(ledger),
            },
            "source_snapshot_sha256": _sha256_path(
                stage / "source_results.jsonl"
            ),
            "task_inputs": task_inputs,
            "artifact_bindings": bindings,
            "implementation": {
                "platform_sha256": _sha256_path(Path(__file__)),
                "trace_gate_sha256": _sha256_path(HERE / "trace_gate.py"),
                "distillation_sha256": _sha256_path(
                    HERE / "success_trace_distill.py"
                ),
            },
            "results_sha256": _sha256_path(stage / "results.jsonl"),
            "resolved_sha256": _sha256_path(stage / "resolved.jsonl"),
            "rejected_sha256": _sha256_path(stage / "rejected.jsonl"),
        }
        (stage / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n"
        )
        manifest_holder.update(manifest)

    _publish_directory_atomic(out, build)
    print(
        f"[repreflight] selected={len(results)} "
        f"admitted={len(admitted)} rejected={len(rejected)} -> {out}",
        flush=True,
    )
    return 0


def cmd_revalidate(a) -> int:
    """Re-run raw teacher patches against fresh strict controls without mutation."""

    out = Path(a.out_dir)
    if os.path.lexists(out):
        raise FileExistsError(f"refusing to overwrite {out}")
    work = Path(str(out) + ".work")
    work.mkdir(parents=True, exist_ok=True)
    taskmeta = {}
    task_inputs = []
    for value in a.tasks:
        files = [value] if os.path.isfile(value) else sorted(globmod.glob(value))
        if not files:
            raise ReplayContractError(f"task metadata input did not resolve: {value}")
        for filename in files:
            task_inputs.append({"path": filename, "sha256": _sha256_path(filename)})
            for row in _read_jsonl(filename):
                instance_id = row.get("instance_id")
                if not isinstance(instance_id, str):
                    raise ReplayContractError(f"task metadata is missing instance_id: {filename}")
                previous = taskmeta.get(instance_id)
                if previous is not None and canonical_json_bytes(previous) != canonical_json_bytes(row):
                    raise ReplayContractError(f"conflicting task metadata: {instance_id}")
                taskmeta[instance_id] = row
    excluded_ids, excluded_repos, exclusion_evidence = _load_exclusion_contract(
        list(a.exclude)
    )
    ledgers = _resolve_revalidation_ledgers(a.glob)
    selected = {}
    selected_bindings = {}
    superseded_attempts = []
    real_attempts_seen = 0
    extraction_records_seen = 0
    non_real_attempt_reasons = Counter()
    for ledger in ledgers:
        source = Path(ledger).parent
        for row in _read_jsonl(ledger):
            extraction_records_seen += 1
            if "revalidation_contract_sha256" in row:
                raise ReplayContractError(
                    f"derived revalidation ledger cannot be an extraction input: {ledger}"
                )
            if _is_real_attempt(row):
                real_attempts_seen += 1
                instance_id = row.get("instance_id")
                if not isinstance(instance_id, str) or not instance_id:
                    raise ReplayContractError(
                        f"real extraction attempt is missing instance_id: {ledger}"
                    )
                binding = {
                    "instance_id": instance_id,
                    "source_run": str(source),
                    "ledger": ledger,
                    "record_sha256": sha256_bytes(canonical_json_bytes(row)),
                }
                if instance_id in selected_bindings:
                    superseded_attempts.append(selected_bindings[instance_id])
                selected[instance_id] = (dict(row), source)
                selected_bindings[instance_id] = binding
            else:
                non_real_attempt_reasons[_non_real_attempt_reason(row)] += 1
    run_contract = _revalidation_contract(
        selected=selected,
        taskmeta=taskmeta,
        ledgers=ledgers,
        task_inputs=task_inputs,
        exclusion_evidence=exclusion_evidence,
        test_timeout=a.test_timeout,
    )
    contract_path = work / "contract.json"
    if contract_path.exists():
        try:
            previous_contract = json.loads(contract_path.read_text())
        except json.JSONDecodeError as exc:
            raise ReplayContractError("revalidation work contract is invalid") from exc
        if previous_contract != run_contract:
            raise ReplayContractError(
                "revalidation inputs changed; refusing to reuse stale work evidence"
            )
    else:
        _replace_json_atomic(contract_path, run_contract)
    run_contract_sha256 = run_contract["contract_sha256"]
    case_contracts = {
        row["instance_id"]: row for row in run_contract["cases"]
    }
    results_path = work / "results.jsonl"
    results = _read_jsonl(results_path) if results_path.exists() else []
    completed = {row["instance_id"] for row in results}
    if len(completed) != len(results):
        raise ReplayContractError("revalidation work ledger contains duplicate IDs")
    if any(
        row.get("revalidation_contract_sha256") != run_contract_sha256
        for row in results
    ):
        raise ReplayContractError("revalidation work ledger is bound to stale inputs")
    for index, instance_id in enumerate(sorted(selected), start=1):
        if instance_id in completed:
            continue
        raw, source = selected[instance_id]
        meta = taskmeta.get(instance_id)
        stream = source / f"{instance_id}.stream.jsonl"
        patch = source / f"{instance_id}.patch"
        result = {
            key: value
            for key, value in raw.items()
            if key not in {"resolved", "training_admitted", "rejection_reasons", "error"}
        }
        result.update(
            resolved=False,
            training_admitted=False,
            source_run=str(source),
            revalidation_contract_sha256=run_contract_sha256,
        )
        try:
            if meta is None:
                raise ReplayContractError("missing task metadata")
            if instance_id in excluded_ids or str(meta.get("repo", "")).casefold() in excluded_repos:
                raise ReplayContractError("task overlaps an explicit exclusion")
            if not stream.is_file() or not patch.is_file():
                raise ReplayContractError("raw artifact set is incomplete")
            candidate_patch = patch.read_text(encoding="utf-8")
            result["stream_sha256"] = _sha256_path(stream)
            result["patch_sha256"] = _sha256_path(patch)
            evidence = verify_candidate_patch(
                meta,
                candidate_patch,
                run=ttd.sh,
                env_bootstrap=ttd.ENV_BOOTSTRAP,
                timeout=a.test_timeout,
            )
            if evidence.get("image_id") != case_contracts[instance_id]["image_id"]:
                raise ReplayContractError("image identity changed during replay")
            result.update(evidence)
            result["resolved"] = evidence["training_admitted"]
            if result["patch_sha256"] != evidence["candidate_patch_sha256"]:
                raise ReplayContractError("candidate patch identity changed during replay")
        except (OSError, ReplayContractError, subprocess.TimeoutExpired) as exc:
            result.update(
                resolved=False,
                training_admitted=False,
                rejection_reasons=["strict_revalidation_error"],
                error=f"{type(exc).__name__}: {str(exc)[:500]}",
            )
        results.append(result)
        _replace_jsonl_atomic(results_path, results)
        print(
            f"[revalidate] {index}/{len(selected)} {instance_id} "
            f"admitted={result['training_admitted']}",
            flush=True,
        )

    result_by_id = {row["instance_id"]: row for row in results}
    if set(result_by_id) != set(selected):
        raise ReplayContractError("revalidation did not complete the selected artifact set")
    final_contract = _revalidation_contract(
        selected=selected,
        taskmeta=taskmeta,
        ledgers=ledgers,
        task_inputs=task_inputs,
        exclusion_evidence=exclusion_evidence,
        test_timeout=a.test_timeout,
    )
    if final_contract != run_contract:
        raise ReplayContractError("revalidation inputs changed before publication")

    def build(stage: Path) -> None:
        stage.mkdir(mode=0o700)
        ordered = [result_by_id[instance_id] for instance_id in sorted(result_by_id)]
        _write_jsonl(stage / "results.jsonl", ordered)
        admitted = [row for row in ordered if admission_is_exact(row)]
        rejected = [row for row in ordered if row not in admitted]
        _write_jsonl(stage / "resolved.jsonl", admitted)
        _write_jsonl(stage / "rejected.jsonl", rejected)
        bindings = []
        for row in ordered:
            _raw, source = selected[row["instance_id"]]
            for extension in (".stream.jsonl", ".patch"):
                artifact = source / f"{row['instance_id']}{extension}"
                if artifact.is_file():
                    shutil.copy2(artifact, stage / artifact.name)
            if row in admitted:
                bindings.append({
                    "instance_id": row["instance_id"],
                    "stream_sha256": row["stream_sha256"],
                    "patch_sha256": row["patch_sha256"],
                    "admission_evidence_sha256": row["admission_evidence_sha256"],
                    "task_contract_sha256": row["task_contract_sha256"],
                    "revalidation_contract_sha256": run_contract_sha256,
                })
        manifest = {
            "schema_version": 2,
            "complete": True,
            "selected": len(ordered),
            "selection_policy": "last_real_attempt_wins",
            "extraction_records_seen": extraction_records_seen,
            "real_attempts_seen": real_attempts_seen,
            "non_real_attempts_seen": sum(non_real_attempt_reasons.values()),
            "non_real_attempt_reasons": dict(sorted(non_real_attempt_reasons.items())),
            "superseded_attempts": superseded_attempts,
            "training_admitted": len(admitted),
            "rejected": len(rejected),
            "input_ledgers": [
                {"path": ledger, "sha256": _sha256_path(ledger)}
                for ledger in ledgers
            ],
            "task_inputs": task_inputs,
            "exclusions": {
                "instance_ids": len(excluded_ids),
                "repositories": len(excluded_repos),
                "artifacts": exclusion_evidence,
            },
            "artifact_bindings": bindings,
            "revalidation_contract_sha256": run_contract_sha256,
            "results_sha256": _sha256_path(stage / "results.jsonl"),
            "resolved_sha256": _sha256_path(stage / "resolved.jsonl"),
            "rejected_sha256": _sha256_path(stage / "rejected.jsonl"),
        }
        (stage / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n"
        )

    _publish_directory_atomic(out, build)
    shutil.rmtree(work)
    print(
        f"[revalidate] complete selected={len(results)} "
        f"admitted={sum(admission_is_exact(row) for row in results)} -> {out}",
        flush=True,
    )
    return 0


# --------------------------------------------------------------------------- #
# prepare  (render resolved teacher traces -> mini-SWE SFT, shape-safe)        #
# --------------------------------------------------------------------------- #
BASH_TOOL_NAME = "bash"


def _msg(role, content, tool_calls=None, *, loss: bool | None = None):
    """Uniform message schema (every row carries a tool_calls key) matching the
    existing SFT datasets so the HF arrow schema stays consistent and the Gemma
    template renders it."""
    message = {"role": role, "content": content, "tool_calls": tool_calls or []}
    if loss is not None:
        message["loss"] = loss
    return message


def render_sft(
    row: dict,
    steps: list,
    *,
    require_terminal: bool = False,
) -> dict | None:
    """mini-SWE SFT row in the exact existing schema: system + PR problem, then
    per step an assistant turn (THOUGHT in content + one bash tool_call) and a
    user OBSERVATION turn. Edit-first shape inherited from the teacher prompt."""
    if not steps:
        return None
    system = ("You are a practical software engineer using a shell to fix one repository "
              "bug. Prefer a small correct source edit over extended inspection. Before each "
              "command write 1-3 terse sentences of reasoning, then emit one bash tool call.")
    terminal = getattr(steps, "terminal_assistant", "").strip()
    if require_terminal and not terminal:
        raise TraceNormalizationError("missing terminal assistant response")
    messages = [_msg("system", system, loss=False),
                _msg("user", f"<pr_description>\n{row['problem_statement']}\n</pr_description>", loss=False)]
    for i, s in enumerate(steps):
        tc = [{"function": {"arguments": json.dumps({"command": s["command"]}), "name": BASH_TOOL_NAME},
               "id": f"teacher-tool-{i}", "type": "function"}]
        step_loss = s.get("loss", True)
        if type(step_loss) is not bool:
            raise TraceNormalizationError("normalized step loss must be boolean")
        messages.append(_msg("assistant", s["thought"], tc, loss=step_loss))
        messages.append(_msg("user", f"OBSERVATION:\n{s['observation']}", loss=False))
    if terminal:
        messages.append(_msg("assistant", terminal, loss=True))
    return {"instance_id": row["instance_id"], "messages": messages,
            "repo": row.get("repo", ""),
            "source": f"teacher:{row.get('backend', '?')}:{row.get('model', '?')}",
            "n_steps": len(steps)}


def _load_exclusion_contract(paths_or_globs: list[str]) -> tuple[set[str], set[str], list[dict]]:
    files: list[str] = []
    for value in paths_or_globs:
        matches = [value] if os.path.isfile(value) else sorted(globmod.glob(value))
        if not matches:
            raise ReplayContractError(f"exclusion input did not resolve: {value}")
        files.extend(matches)
    if not files:
        raise ReplayContractError("at least one explicit exclusion artifact is required")
    instance_ids: set[str] = set()
    repositories: set[str] = set()
    evidence = []
    for filename in files:
        path = Path(filename)
        text = path.read_text(encoding="utf-8")
        try:
            payload = json.loads(text)
        except json.JSONDecodeError:
            payload = [
                json.loads(line)
                for line in text.splitlines()
                if line.strip()
            ]
        values = payload if isinstance(payload, list) else [payload]
        for value in values:
            if not isinstance(value, dict):
                raise ReplayContractError(f"exclusion artifact has a non-object row: {path}")
            for key in ("instance_ids", "selected_ids"):
                listed = value.get(key, [])
                if not isinstance(listed, list) or any(
                    not isinstance(item, str) for item in listed
                ):
                    raise ReplayContractError(f"invalid exclusion field {key}: {path}")
                instance_ids.update(listed)
            for key in ("instance_id", "source_instance_id"):
                if isinstance(value.get(key), str):
                    instance_ids.add(value[key])
            listed_repos = value.get("repo_denylist", [])
            if not isinstance(listed_repos, list) or any(
                not isinstance(item, str) for item in listed_repos
            ):
                raise ReplayContractError(f"invalid repository exclusions: {path}")
            repositories.update(item.casefold() for item in listed_repos)
        evidence.append({
            "path": str(path),
            "sha256": _sha256_path(path),
        })
    return instance_ids, repositories, evidence


def cmd_prepare(a) -> int:
    merged = Path(a.merged)
    try:
        merge_manifest = json.loads((merged / "manifest.json").read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ReplayContractError("merge manifest is unreadable") from exc
    resolved_path = merged / "resolved.jsonl"
    if (
        merge_manifest.get("schema_version") != 2
        or merge_manifest.get("complete") is not True
        or merge_manifest.get("resolved_sha256") != _sha256_path(resolved_path)
    ):
        raise ReplayContractError("merge manifest is incomplete or does not bind resolved.jsonl")
    resolved = _read_jsonl(merged / "resolved.jsonl")
    bindings = {
        row["instance_id"]: row
        for row in merge_manifest.get("artifact_bindings", [])
        if isinstance(row, dict) and isinstance(row.get("instance_id"), str)
    }
    if len(bindings) != len(merge_manifest.get("artifact_bindings", [])):
        raise ReplayContractError("merge artifact bindings contain duplicate identities")
    if set(bindings) != {row.get("instance_id") for row in resolved}:
        raise ReplayContractError("merge artifact bindings do not match resolved rows")
    excluded_ids, excluded_repos, exclusion_evidence = _load_exclusion_contract(
        list(getattr(a, "exclude", []) or [])
    )
    # result ledgers carry no problem_statement -> join task metadata by id
    taskmeta = {}
    task_evidence = []
    for tp in a.tasks:
        files = [tp] if os.path.isfile(tp) else sorted(globmod.glob(tp))
        if not files:
            raise ReplayContractError(f"task metadata input did not resolve: {tp}")
        for f in files:
            task_evidence.append({"path": f, "sha256": _sha256_path(f)})
            for row in _read_jsonl(f):
                instance_id = row.get("instance_id")
                if not isinstance(instance_id, str):
                    raise ReplayContractError(f"task metadata is missing instance_id: {f}")
                previous = taskmeta.get(instance_id)
                if previous is not None and canonical_json_bytes(previous) != canonical_json_bytes(row):
                    raise ReplayContractError(f"conflicting task metadata: {instance_id}")
                taskmeta[instance_id] = row
    out_rows = []
    distillation_exclusions = []
    stats = Counter()
    seen_content: dict[str, str] = {}
    for r in resolved:
        if not admission_is_exact(r):
            raise ReplayContractError(
                f"resolved row lost exact admission evidence: {r.get('instance_id')}"
            )
        meta = taskmeta.get(r["instance_id"])
        if not meta:
            raise ReplayContractError(f"missing task metadata: {r['instance_id']}")
        if (
            r["instance_id"] in excluded_ids
            or str(meta.get("repo", "")).casefold() in excluded_repos
        ):
            raise ReplayContractError(
                f"admitted row overlaps an explicit exclusion: {r['instance_id']}"
            )
        contract = build_task_contract(meta)
        if (
            contract != r.get("task_contract")
            or contract["contract_sha256"] != r.get("task_contract_sha256")
        ):
            raise ReplayContractError(f"task contract hash mismatch: {r['instance_id']}")
        stream = merged / f"{r['instance_id']}.stream.jsonl"
        patch = merged / f"{r['instance_id']}.patch"
        binding = bindings[r["instance_id"]]
        if not stream.is_file() or not patch.is_file():
            raise ReplayContractError(f"merged artifact is missing: {r['instance_id']}")
        if (
            _sha256_path(stream) != binding.get("stream_sha256")
            or _sha256_path(patch) != binding.get("patch_sha256")
            or binding.get("stream_sha256") != r.get("stream_sha256")
            or binding.get("patch_sha256") != r.get("patch_sha256")
        ):
            raise ReplayContractError(f"merged artifact hash mismatch: {r['instance_id']}")
        try:
            normalized = normalize_steps(r.get("backend", "claude"), stream)
            replay_trace = ReplayTrace(
                steps=tuple(
                    ReplayStep(
                        command=step["command"],
                        observation=step["observation"],
                        returncode=step["returncode"],
                        mutates_source=step["mutates_source"],
                        assistant=step.get("thought", ""),
                        loss=True,
                    )
                    for step in normalized
                ),
                terminal_assistant=normalized.terminal_assistant,
            )
            distilled = distill_success_path(meta, replay_trace, r)
            steps = NormalizedTrace(
                [
                    {
                        "thought": step.assistant,
                        "command": step.command,
                        "observation": step.observation,
                        "returncode": step.returncode,
                        "mutates_source": step.mutates_source,
                        "loss": step.loss,
                    }
                    for step in distilled.steps
                ],
                terminal_assistant=distilled.terminal_assistant,
            )
            # Render off task metadata but tag with the exact teacher identity.
            render_row = dict(meta, backend=r.get("backend"), model=r.get("model"))
            row = render_sft(render_row, steps, require_terminal=True) if steps else None
        except (ValueError, OSError, subprocess.TimeoutExpired) as exc:
            rejection = {
                "instance_id": r["instance_id"],
                "backend": r.get("backend"),
                "model": r.get("model"),
                "stage": "success_path_distillation",
                "reason_type": type(exc).__name__,
                "reason": str(exc)[:1000],
                "stream_sha256": r["stream_sha256"],
                "patch_sha256": r["patch_sha256"],
                "admission_evidence_sha256": r["admission_evidence_sha256"],
                "task_contract_sha256": r["task_contract_sha256"],
            }
            rejection["rejection_sha256"] = sha256_bytes(
                canonical_json_bytes(rejection)
            )
            distillation_exclusions.append(rejection)
            continue
        if row is None:
            raise ReplayContractError(f"trace rendered no commands: {r['instance_id']}")
        content_sha = sha256_bytes(canonical_json_bytes(row["messages"]))
        if content_sha in seen_content:
            raise ReplayContractError(
                f"duplicate rendered content: {seen_content[content_sha]} and {r['instance_id']}"
            )
        seen_content[content_sha] = r["instance_id"]
        row["content_sha256"] = content_sha
        row["stream_sha256"] = r["stream_sha256"]
        row["patch_sha256"] = r["patch_sha256"]
        row["admission_evidence_sha256"] = r["admission_evidence_sha256"]
        row["task_contract_sha256"] = r["task_contract_sha256"]
        row["distilled_source_sha256"] = distilled.source_sha256
        row["retained_commands"] = [step["command"] for step in steps]
        row["retained_loss"] = [step["loss"] for step in steps]
        # edit-first sanity: first source-editing command index
        first_edit = next((i for i, s in enumerate(steps)
                           if re.search(r'\b(sed -i|>|>>|tee |patch |apply|cat <<)', s["command"])), None)
        row["first_edit_cmd"] = first_edit
        stats[r.get("backend", "?")] += 1
        out_rows.append(row)
    if not out_rows:
        print("[prepare] no rows rendered — nothing to write", flush=True)
        return 1
    first_edits = [x.pop("first_edit_cmd", None) for x in out_rows]
    fe = sorted(e for e in first_edits if e is not None)
    manifest_holder = {}

    def build(stage: Path) -> None:
        from datasets import Dataset

        Dataset.from_list(out_rows).save_to_disk(stage)
        _write_jsonl(stage / "train.jsonl", out_rows)
        manifest = {
            "schema_version": 2,
            "complete": True,
            "resolved_in": len(resolved),
            "rendered": len(out_rows),
            "training_admitted": 0,
            "all_training_gates_complete": False,
            "per_backend": dict(stats),
            "median_first_edit_cmd": fe[len(fe) // 2] if fe else None,
            "merge_manifest_sha256": _sha256_path(merged / "manifest.json"),
            "merge_resolved_sha256": _sha256_path(resolved_path),
            "task_inputs": task_evidence,
            "exclusions": {
                "instance_ids": len(excluded_ids),
                "repositories": len(excluded_repos),
                "artifacts": exclusion_evidence,
            },
            "distillation_exclusions": {
                "count": len(distillation_exclusions),
                "artifact_bindings": distillation_exclusions,
            },
            "artifact_bindings": [
                {
                    key: row[key]
                    for key in (
                        "instance_id",
                        "stream_sha256",
                        "patch_sha256",
                        "admission_evidence_sha256",
                        "task_contract_sha256",
                        "content_sha256",
                        "distilled_source_sha256",
                        "retained_commands",
                        "retained_loss",
                    )
                }
                for row in out_rows
            ],
            "train_jsonl_sha256": _sha256_path(stage / "train.jsonl"),
            "success_path_distilled": True,
            "distillation_implementation_sha256": _sha256_path(
                HERE / "success_trace_distill.py"
            ),
            "standard_native_format_loss_gate": {
                "status": "pending_full_dataset_verification",
                "failure_count": None,
            },
            "format": (
                "mini-swe SFT: system+PR, assistant targets loss=true, "
                "conditioning loss=false, terminal assistant preserved"
            ),
        }
        (stage / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n"
        )
        manifest_holder.update(manifest)

    _publish_directory_atomic(Path(a.out), build)
    man = manifest_holder
    print(f"[prepare] resolved={len(resolved)} rendered={len(out_rows)} "
          f"format_gate=pending -> {a.out} (atomic HF dataset dir)", flush=True)
    return 0


# --------------------------------------------------------------------------- #
# ingest  (external trajectory datasets, e.g. nvidia/Open-SWE-Traces)          #
# --------------------------------------------------------------------------- #
class UnsupportedTrajectoryTool(ValueError):
    """A trajectory cannot be represented by the one-bash-tool schema."""


_OPEN_SWE_EDITOR_VIEW_MARKER = "# OPEN_SWE_EDITOR_VIEW"


def _require_positive_int(value: int, name: str) -> int:
    if type(value) is not int or value <= 0:
        raise ValueError(f"{name} must be positive")
    return value


def _require_nonnegative_int(value: int, name: str) -> int:
    if type(value) is not int or value < 0:
        raise ValueError(f"{name} must be nonnegative")
    return value


def _required_trajectory_string(arguments: dict, key: str, *, nonempty: bool = False) -> str:
    value = arguments.get(key)
    if not isinstance(value, str) or (nonempty and not value.strip()):
        raise UnsupportedTrajectoryTool(f"{key} must be a{' nonempty' if nonempty else ''} string")
    return value


def _trajectory_declared_root(msgs: list) -> str | None:
    """Return the single repository root declared by the first upload block."""
    for message in msgs:
        if not isinstance(message, dict) or message.get("role") != "user":
            continue
        content = message.get("content")
        if not isinstance(content, str) or content.lstrip().startswith("OBSERVATION:"):
            return None
        match = re.search(
            r"<uploaded_files>\s*(.*?)\s*</uploaded_files>", content, re.DOTALL
        )
        if match is None:
            return None
        paths = [line.strip() for line in match.group(1).splitlines() if line.strip()]
        if len(paths) != 1:
            raise UnsupportedTrajectoryTool(
                "uploaded_files must declare exactly one repository root"
            )
        raw_root = paths[0]
        if raw_root.startswith("//"):
            raise UnsupportedTrajectoryTool(
                "uploaded_files repository root cannot use a double-leading slash"
            )
        root = os.path.normpath(raw_root)
        if not os.path.isabs(root) or root == "/":
            raise UnsupportedTrajectoryTool(
                "uploaded_files repository root must be an absolute non-root path"
            )
        if root == "/workspace":
            raise UnsupportedTrajectoryTool(
                "uploaded_files repository root is too broad"
            )
        return root
    return None


def _path_is_under(path: str, root: str) -> bool:
    try:
        return os.path.commonpath((path, root)) == root
    except ValueError:
        return False


def _trajectory_testbed_path(arguments: dict, declared_root: str | None = None) -> str:
    path = _required_trajectory_string(arguments, "path", nonempty=True)
    normalized = os.path.normpath(path)
    if _path_is_under(normalized, "/testbed"):
        return normalized
    if declared_root is None or not _path_is_under(normalized, declared_root):
        raise UnsupportedTrajectoryTool(
            "editor path must be under /testbed or the declared repository root"
        )
    relative = os.path.relpath(normalized, declared_root)
    return "/testbed" if relative == "." else f"/testbed/{relative}"


def _lex_trajectory_shell(
    text: str, *, comments: bool, posix: bool = True
) -> list[str]:
    lexer = shlex.shlex(text, posix=posix, punctuation_chars=True)
    lexer.whitespace_split = True
    lexer.commenters = "#" if comments else ""
    try:
        return list(lexer)
    except ValueError as exc:
        raise UnsupportedTrajectoryTool("bash command has ambiguous shell quoting") from exc


_SHELL_COMMAND_SEPARATORS = {";", "&&", "||", "|", "&", "(", ")"}
_SHELL_REDIRECTION_TOKENS = {"<", ">", "<<", ">>", "<>", "<&", ">&", ">|"}
_MAX_TRAJECTORY_HEREDOC_DEPTH = 32


def _heredoc_body_executes_as_shell(tokens: list[str], operator_index: int) -> bool:
    """Recognize only bash/sh stdin consumers; other interpreters stay literal.

    This intentionally supports direct or absolute-path bash/sh commands and a
    simple ``env [assignments] bash|sh`` wrapper. It does not infer execution
    semantics for Python, Ruby, or other language interpreters.
    """
    segment_start = 0
    for index, token in enumerate(tokens[:operator_index]):
        if token in _SHELL_COMMAND_SEPARATORS:
            segment_start = index + 1
    prefix = tokens[segment_start:operator_index]
    while prefix and re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*=.*", prefix[0]):
        prefix = prefix[1:]
    if not prefix:
        raise UnsupportedTrajectoryTool("bash command has ambiguous heredoc consumer")
    executable = os.path.basename(prefix[0])
    if executable in ("bash", "sh"):
        return True
    if executable in ("command", "exec", "source", "."):
        raise UnsupportedTrajectoryTool("bash command has ambiguous heredoc consumer")
    if executable != "env":
        if any(token in _SHELL_REDIRECTION_TOKENS for token in prefix) and any(
            os.path.basename(token) in ("bash", "sh") for token in prefix[1:]
        ):
            raise UnsupportedTrajectoryTool("bash command has ambiguous heredoc consumer")
        return False
    prefix = prefix[1:]
    while prefix and (
        prefix[0] in ("-i", "--ignore-environment")
        or re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*=.*", prefix[0])
    ):
        prefix = prefix[1:]
    if not prefix:
        raise UnsupportedTrajectoryTool("bash command has ambiguous heredoc consumer")
    if prefix[0].startswith("-"):
        raise UnsupportedTrajectoryTool("bash command has ambiguous heredoc consumer")
    return os.path.basename(prefix[0]) in ("bash", "sh")


def _trajectory_shell_parts(
    command: str, *, depth: int = 0
) -> list[tuple[bool, str]]:
    """Split shell-visible lines from well-formed heredoc bodies and delimiters."""
    if depth > _MAX_TRAJECTORY_HEREDOC_DEPTH:
        raise UnsupportedTrajectoryTool(
            "bash command exceeds heredoc nesting depth limit"
        )
    lines = command.splitlines(keepends=True)
    parts: list[tuple[bool, str]] = []
    index = 0
    while index < len(lines):
        header = lines[index]
        parts.append((True, header))
        index += 1
        if "<<" not in header:
            continue
        tokens = _lex_trajectory_shell(header, comments=True, posix=False)
        normalized_tokens = _lex_trajectory_shell(header, comments=True)
        normalized_operators = [
            token_index
            for token_index, token in enumerate(normalized_tokens)
            if token == "<<"
        ]
        delimiters = []
        operator_ordinal = 0
        for token_index, token in enumerate(tokens):
            if token != "<<":
                continue
            if operator_ordinal >= len(normalized_operators):
                raise UnsupportedTrajectoryTool("bash command has ambiguous heredoc")
            if token_index + 1 >= len(tokens):
                raise UnsupportedTrajectoryTool("bash command has malformed heredoc")
            raw_delimiter = tokens[token_index + 1]
            if (
                len(raw_delimiter) >= 2
                and raw_delimiter[0] == raw_delimiter[-1]
                and raw_delimiter[0] in "'\""
            ):
                delimiter = raw_delimiter[1:-1]
            elif raw_delimiter.startswith("\\"):
                delimiter = raw_delimiter[1:]
            else:
                raise UnsupportedTrajectoryTool(
                    "bash command uses an unquoted heredoc delimiter"
                )
            if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", delimiter):
                raise UnsupportedTrajectoryTool("bash command has ambiguous heredoc delimiter")
            scans_body = _heredoc_body_executes_as_shell(
                normalized_tokens, normalized_operators[operator_ordinal]
            )
            delimiters.append((delimiter, scans_body))
            operator_ordinal += 1
        if operator_ordinal != len(normalized_operators):
            raise UnsupportedTrajectoryTool("bash command has ambiguous heredoc")
        for delimiter, scans_body in delimiters:
            body_lines = []
            while index < len(lines):
                body_line = lines[index]
                index += 1
                if body_line.rstrip("\r\n") == delimiter:
                    break
                body_lines.append(body_line)
            else:
                raise UnsupportedTrajectoryTool("bash command has unterminated heredoc")
            if scans_body:
                parts.extend(
                    _trajectory_shell_parts("".join(body_lines), depth=depth + 1)
                )
            else:
                parts.extend((False, body_line) for body_line in body_lines)
            parts.append((False, body_line))
    return parts


def _trajectory_shell_tokens(command: str) -> list[str]:
    """Return executable shell tokens while excluding literal heredoc bodies."""
    visible = "".join(
        part for is_visible, part in _trajectory_shell_parts(command) if is_visible
    )
    return _lex_trajectory_shell(visible, comments=True)


def _shell_token_references_path(token: str, root: str) -> bool:
    """Whether a shell token contains an absolute reference rooted at ``root``."""
    offset = 0
    while (index := token.find(root, offset)) >= 0:
        end = index + len(root)
        has_left_boundary = (
            index == 0
            or token[index - 1].isspace()
            or token[index - 1] in "=:(,[{"
        )
        if has_left_boundary and (end == len(token) or token[end] in "/:;,)]}"):
            return True
        offset = index + 1
    return False


def _normalize_trajectory_bash_command(
    command: str, declared_root: str | None
) -> str:
    """Rewrite declared source paths and reject source-workspace escapes."""
    if declared_root and declared_root != "/testbed":
        root_pattern = re.compile(
            r"(?<![A-Za-z0-9_./~-])"
            + re.escape(declared_root)
            + r"(?=$|/|[\s'\"`;&|<>()])"
        )
        command = "".join(
            root_pattern.sub("/testbed", part) if is_visible else part
            for is_visible, part in _trajectory_shell_parts(command)
        )
    for token in _trajectory_shell_tokens(command):
        if _shell_token_references_path(token, "/workspace") or (
            _shell_token_references_path(token, "/testbed")
            and ".." in token.split("/")
        ):
            raise UnsupportedTrajectoryTool(
                "bash command contains an unsafe workspace path"
            )
    return command


def _reject_trajectory_relative_escape(command: str) -> None:
    for token in _trajectory_shell_tokens(command):
        if _shell_token_references_path(token, "/workspace") or (
            ".." in token.split("/")
        ):
            raise UnsupportedTrajectoryTool(
                "bash command contains an unsafe workspace path"
            )


def _quoted_python_editor(lines: list[str]) -> str:
    return (
        "python3 - <<'OPEN_SWE_PY'\n"
        "# OPEN_SWE_EDITOR_MUTATION\n"
        + "\n".join(lines)
        + "\nOPEN_SWE_PY"
    )


def translate_trajectory_tool_call(
    tool_call: dict, *, declared_root: str | None = None
) -> str | None:
    """Translate one Open-SWE tool call into executable bash, or reject it.

    ``None`` is reserved for ``submit``, a harness control action which should
    not become a training command.
    """
    if not isinstance(tool_call, dict) or not isinstance(tool_call.get("function"), dict):
        raise UnsupportedTrajectoryTool("malformed tool call")
    function = tool_call["function"]
    name = function.get("name")
    raw_arguments = function.get("arguments")
    if not isinstance(name, str) or not isinstance(raw_arguments, str):
        raise UnsupportedTrajectoryTool("tool name and arguments must be strings")
    try:
        arguments = json.loads(raw_arguments)
    except (json.JSONDecodeError, TypeError) as exc:
        raise UnsupportedTrajectoryTool("malformed tool arguments") from exc
    if not isinstance(arguments, dict):
        raise UnsupportedTrajectoryTool("tool arguments must decode to an object")

    if name == "submit":
        return None
    if name in ("bash", "execute_bash"):
        if name == "execute_bash":
            command_keys = [key for key in ("command", "input") if key in arguments]
            if len(command_keys) != 1:
                raise UnsupportedTrajectoryTool(
                    "execute_bash must provide exactly one of command or input"
                )
            command_key = command_keys[0]
        else:
            command_key = "command"
        command = _required_trajectory_string(arguments, command_key, nonempty=True)
        command = _strip_docker_exec(command)
        command = _normalize_trajectory_bash_command(command, declared_root)
        command = _strip_docker_exec(command).strip()
        if not command:
            raise UnsupportedTrajectoryTool("bash command is empty after wrapper cleanup")
        _reject_trajectory_relative_escape(command)
        return command
    if name != "str_replace_editor":
        raise UnsupportedTrajectoryTool(f"unsupported trajectory tool: {name}")

    editor_command = _required_trajectory_string(arguments, "command", nonempty=True)
    path = _trajectory_testbed_path(arguments, declared_root)
    python_path = json.dumps(path)

    if editor_command == "view":
        quoted_path = shlex.quote(path)
        if "view_range" in arguments:
            view_range = arguments["view_range"]
            if (
                not isinstance(view_range, list)
                or len(view_range) != 2
                or any(type(line) is not int for line in view_range)
            ):
                raise UnsupportedTrajectoryTool("view_range must be a two-integer list")
            start, end = view_range
            if start < 1 or (end != -1 and end < start):
                raise UnsupportedTrajectoryTool("invalid view_range")
            last = "$" if end == -1 else str(end)
            sed_range = shlex.quote(f"{start},{last}p")
            command = (
                f"if [ -d {quoted_path} ]; then "
                "echo 'view_range is invalid for a directory' >&2; exit 1; "
                f"elif [ ! -f {quoted_path} ]; then "
                "echo 'view target is not a regular file' >&2; exit 1; "
                "else "
                f"line_count=$(awk 'END {{ print NR }}' {quoted_path}) || exit 1; "
                f"if [ \"$line_count\" -lt {start} ]; then "
                "echo 'view_range starts past end of file' >&2; exit 1; fi; "
                f"set -o pipefail; nl -ba {quoted_path} | sed -n {sed_range}; fi"
            )
        else:
            command = (
                f"if [ -d {quoted_path} ]; then ls -la {quoted_path}; "
                f"else nl -ba {quoted_path}; fi"
            )
        return f"{_OPEN_SWE_EDITOR_VIEW_MARKER}\n{command}"

    if editor_command == "create":
        file_text = _required_trajectory_string(arguments, "file_text")
        return _quoted_python_editor([
            "from pathlib import Path",
            f"path = Path({python_path})",
            f"file_text = {json.dumps(file_text)}",
            "with path.open('x', encoding='utf-8') as handle:",
            "    handle.write(file_text)",
        ])

    if editor_command == "str_replace":
        old_str = _required_trajectory_string(arguments, "old_str", nonempty=True)
        new_str = _required_trajectory_string(arguments, "new_str")
        return _quoted_python_editor([
            "from pathlib import Path",
            f"path = Path({python_path})",
            f"old_str = {json.dumps(old_str)}",
            f"new_str = {json.dumps(new_str)}",
            "text = path.read_text(encoding='utf-8')",
            "matches = text.count(old_str)",
            "if matches != 1:",
            "    raise SystemExit(f'expected exactly one old_str match, found {matches}')",
            "path.write_text(text.replace(old_str, new_str), encoding='utf-8')",
        ])

    if editor_command == "insert":
        insert_line = arguments.get("insert_line")
        if type(insert_line) is not int or insert_line < 0:
            raise UnsupportedTrajectoryTool("insert_line must be a nonnegative integer")
        new_str = _required_trajectory_string(arguments, "new_str")
        return _quoted_python_editor([
            "from pathlib import Path",
            f"path = Path({python_path})",
            f"insert_line = {insert_line}",
            f"new_str = {json.dumps(new_str)}",
            "text = path.read_text(encoding='utf-8')",
            "physical_line_count = len(text.splitlines())",
            "if insert_line > physical_line_count:",
            "    raise SystemExit(",
            "        f'insert_line {insert_line} exceeds {physical_line_count} lines'",
            "    )",
            "lines = text.split('\\n')",
            "lines[insert_line:insert_line] = new_str.split('\\n')",
            "path.write_text('\\n'.join(lines), encoding='utf-8')",
        ])

    raise UnsupportedTrajectoryTool(f"unsupported editor command: {editor_command}")


def steps_from_trajectory(msgs: list, max_obs: int = 2000) -> list:
    """Parse an OpenAI-style trajectory into strict command/result pairs."""
    _require_positive_int(max_obs, "max_obs")
    declared_root = _trajectory_declared_root(msgs)
    steps, pending = [], None
    pending_tool_call_id = None
    omitted_submit = False
    omitted_tool_call_id = None
    for m in msgs:
        if not isinstance(m, dict):
            raise UnsupportedTrajectoryTool("trajectory message must be an object")
        role = m.get("role")
        tcs = m.get("tool_calls") or []
        if role == "assistant" and tcs:
            if pending is not None:
                raise UnsupportedTrajectoryTool("tool call is missing its observation")
            omitted_submit = False
            omitted_tool_call_id = None
            if not isinstance(tcs, list) or len(tcs) != 1:
                raise UnsupportedTrajectoryTool("trajectory turns must contain exactly one tool call")
            tool_call_id = tcs[0].get("id") if isinstance(tcs[0], dict) else None
            if tool_call_id is not None and (
                not isinstance(tool_call_id, str) or not tool_call_id
            ):
                raise UnsupportedTrajectoryTool("tool call id must be a nonempty string")
            thought = next(
                (value.strip() for key in ("content", "reasoning_content", "think")
                 if isinstance((value := m.get(key)), str) and value.strip()),
                "",
            )
            command = translate_trajectory_tool_call(
                tcs[0], declared_root=declared_root
            )
            if command is None:
                omitted_submit = True
                omitted_tool_call_id = tool_call_id
            else:
                pending = {"thought": thought, "command": command, "observation": ""}
                pending_tool_call_id = tool_call_id
        elif role in ("tool", "user") and pending is not None:
            obs = m.get("content")
            if not isinstance(obs, str):
                raise UnsupportedTrajectoryTool("observation content must be a string")
            if role == "user" and not obs.lstrip().startswith("OBSERVATION:"):
                raise UnsupportedTrajectoryTool("user tool result must start with OBSERVATION:")
            result_has_id = "tool_call_id" in m
            result_id = m.get("tool_call_id")
            if result_has_id and (not isinstance(result_id, str) or not result_id):
                raise UnsupportedTrajectoryTool("tool result id must be a nonempty string")
            if result_has_id and result_id != pending_tool_call_id:
                raise UnsupportedTrajectoryTool("tool result id does not match tool call id")
            obs = re.sub(r"^\s*OBSERVATION:\s*", "", obs).strip()
            if len(obs) > max_obs:
                head_chars = max_obs // 2
                tail_chars = max_obs - head_chars
                obs = obs[:head_chars] + "\n...[truncated]...\n" + obs[-tail_chars:]
            steps.append({**pending, "observation": obs})
            pending = None
            pending_tool_call_id = None
        elif role == "tool" or (
            role == "user" and str(m.get("content") or "").lstrip().startswith("OBSERVATION:")
        ):
            if omitted_submit:
                obs = m.get("content")
                if not isinstance(obs, str):
                    raise UnsupportedTrajectoryTool("observation content must be a string")
                result_has_id = "tool_call_id" in m
                result_id = m.get("tool_call_id")
                if result_has_id and (not isinstance(result_id, str) or not result_id):
                    raise UnsupportedTrajectoryTool("tool result id must be a nonempty string")
                if result_has_id and result_id != omitted_tool_call_id:
                    raise UnsupportedTrajectoryTool("tool result id does not match tool call id")
                omitted_submit = False
                omitted_tool_call_id = None
            else:
                raise UnsupportedTrajectoryTool("observation has no matching tool call")
        elif pending is not None:
            raise UnsupportedTrajectoryTool("tool call is missing its observation")
        elif omitted_submit:
            omitted_submit = False
            omitted_tool_call_id = None
    if pending is not None:
        raise UnsupportedTrajectoryTool("trajectory ended before a tool observation")
    return steps


_EDIT_RE = re.compile(
    r'\b(sed -i|>|>>|tee |patch |apply|cat <<|python -c|>\s*/testbed)'
    r'|# OPEN_SWE_EDITOR_MUTATION'
)


def edit_first_trim(steps: list, max_steps: int) -> list:
    """Keep an edit-biased slice while preserving the mutation and verification."""
    _require_nonnegative_int(max_steps, "max_steps")
    if max_steps == 0 or len(steps) <= max_steps:
        return steps
    first_edit = next((i for i, s in enumerate(steps) if _EDIT_RE.search(s["command"])), None)
    if first_edit is None:
        return steps[-max_steps:]

    selected = {first_edit}
    for index in range(min(3, len(steps))):
        if len(selected) < max_steps:
            selected.add(index)
    if first_edit > 0 and len(selected) < max_steps:
        selected.add(first_edit - 1)
    for index in range(len(steps) - 1, first_edit, -1):
        if len(selected) >= max_steps:
            break
        selected.add(index)
    return [steps[index] for index in sorted(selected)]


def _streaming_split_num_examples(split_dataset, split_name: str) -> int | None:
    """Read a streaming split's declared size without loading it again."""
    splits = getattr(getattr(split_dataset, "info", None), "splits", None)
    try:
        split_info = splits[split_name]
    except (KeyError, TypeError):
        return None
    count = getattr(split_info, "num_examples", None)
    if not isinstance(count, int) or isinstance(count, bool) or count < 0:
        return None
    return count


def _ingest_progress_snapshot(
    *,
    scanned: int,
    total: int | None,
    resolved_matched: int,
    ingested: int,
    started_at: float,
    now: float,
) -> dict:
    """Build one stable, machine-readable ingest progress record."""
    elapsed = max(0.0, now - started_at)
    rows_per_second = scanned / elapsed if elapsed else 0.0
    if total is None:
        percent = None
        eta_seconds = None
    elif total == 0:
        percent = 100.0 if scanned == 0 else None
        eta_seconds = 0.0 if scanned == 0 else None
    else:
        percent = min(100.0, scanned * 100.0 / total)
        if scanned >= total:
            eta_seconds = 0.0
        elif rows_per_second:
            eta_seconds = (total - scanned) / rows_per_second
        else:
            eta_seconds = None
    return {
        "scanned": scanned,
        "total": total,
        "percent": None if percent is None else round(percent, 3),
        "resolved_matched": resolved_matched,
        "ingested": ingested,
        "elapsed_seconds": round(elapsed, 3),
        "rows_per_second": round(rows_per_second, 3),
        "eta_seconds": None if eta_seconds is None else round(eta_seconds, 3),
    }


def cmd_ingest(a) -> int:
    """Ingest an external resolved-trajectory dataset into mini-SWE SFT rows."""
    import ast

    _require_nonnegative_int(a.limit, "limit")
    _require_positive_int(a.max_obs_chars, "max_obs_chars")
    _require_nonnegative_int(a.max_steps, "max_steps")
    _require_nonnegative_int(a.max_pr_chars, "max_pr_chars")
    _require_positive_int(a.progress_every, "progress_every")

    from datasets import Dataset, load_dataset

    exclude = _attempted_ids(*a.exclude)
    out, seen = [], set()
    stats = Counter()
    tool_conversion_drop_reasons = Counter()
    scanned = kept_resolved = 0
    started_at = time.monotonic()
    loaded_streams = []
    declared_total = 0
    total_known = True
    for cfg in a.configs:
        d = load_dataset(a.dataset, cfg, streaming=True)
        split_names = tuple(d.keys())
        loaded_streams.append((cfg, d, split_names))
        for split in split_names:
            split_count = _streaming_split_num_examples(d[split], split)
            if split_count is None:
                total_known = False
            else:
                declared_total += split_count
    total_rows = declared_total if total_known else None
    last_progress_scanned = None

    def report_progress():
        nonlocal last_progress_scanned
        print(json.dumps(_ingest_progress_snapshot(
            scanned=scanned,
            total=total_rows,
            resolved_matched=kept_resolved,
            ingested=len(out),
            started_at=started_at,
            now=time.monotonic(),
        )), flush=True)
        last_progress_scanned = scanned

    for cfg, d, split_names in loaded_streams:
        for split in split_names:
            for r in d[split]:
                scanned += 1
                try:
                    if a.language and r.get("language") != a.language:
                        continue
                    if a.resolved_only and str(r.get("resolved")) != "1":
                        continue
                    kept_resolved += 1
                    iid = r["instance_id"]
                    if iid in exclude or iid in seen:
                        continue
                    traj = r["trajectory"]
                    try:
                        msgs = ast.literal_eval(traj) if isinstance(traj, str) else traj
                    except (ValueError, SyntaxError):
                        continue
                    try:
                        steps = steps_from_trajectory(msgs, max_obs=a.max_obs_chars)
                    except UnsupportedTrajectoryTool as exc:
                        tool_conversion_drop_reasons[str(exc)] += 1
                        continue
                    if a.max_steps:
                        steps = edit_first_trim(steps, a.max_steps)
                    pr = next((str(m.get("content")) for m in msgs if m.get("role") == "user"), "")
                    if a.max_pr_chars and len(pr) > a.max_pr_chars:
                        pr = pr[: a.max_pr_chars] + "\n...[truncated]..."
                    task = {"instance_id": iid, "problem_statement": pr, "repo": r.get("repo", ""),
                            "backend": "open-swe", "model": f"{cfg}:{split}"}
                    sft = render_sft(task, steps)
                    if sft:
                        out.append(sft)
                        seen.add(iid)
                        stats[f"{cfg}/{split}"] += 1
                        if a.limit and len(out) >= a.limit:
                            break
                finally:
                    if scanned % a.progress_every == 0:
                        report_progress()
            if a.limit and len(out) >= a.limit:
                break
        if a.limit and len(out) >= a.limit:
            break
    if last_progress_scanned != scanned:
        report_progress()
    man = {"dataset": a.dataset, "configs": a.configs, "language": a.language,
           "resolved_only": a.resolved_only, "scanned": scanned, "resolved_matched": kept_resolved,
           "ingested": len(out), "excluded_ids": len(exclude), "per_source": dict(stats),
           "dropped_tool_conversion": sum(tool_conversion_drop_reasons.values()),
           "tool_conversion_drop_reasons": dict(sorted(tool_conversion_drop_reasons.items())),
           "note": "external teacher traces rendered to mini-SWE SFT; blend with own teacher traces via `blend`"}
    manifest_path = Path(str(a.out) + ".manifest.json")
    if not out:
        manifest_path.write_text(json.dumps(man, indent=1))
        print("[ingest] nothing ingested (check filters/exclude)", flush=True)
        return 1
    Dataset.from_list(out).save_to_disk(a.out)
    _write_jsonl(str(a.out) + ".jsonl", out)
    manifest_path.write_text(json.dumps(man, indent=1))
    print(f"[ingest] scanned={scanned} resolved-matched={kept_resolved} "
          f"ingested={len(out)} -> {a.out}", flush=True)
    return 0


# --------------------------------------------------------------------------- #
# blend  (combine SFT sources -> one training dataset, dedup, per-source cap)  #
# --------------------------------------------------------------------------- #
def cmd_blend(a) -> int:
    """Combine multiple prepared/ingested SFT sources into ONE training dataset
    dir. Dedup by instance_id (earlier sources win, so put your best teacher
    first). Optional --cap limits rows per source to control the mix."""
    rows, seen = [], set()
    stats = Counter()
    source_evidence = []
    for src in a.sources:
        source_path = Path(src)
        source_manifest = source_path / "manifest.json"
        manifest = None
        if source_manifest.is_file():
            try:
                manifest = json.loads(source_manifest.read_text())
            except json.JSONDecodeError as exc:
                raise ReplayContractError(f"source manifest is invalid: {src}") from exc
            if (
                manifest.get("schema_version") == 2
                and manifest.get("all_training_gates_complete") is not True
            ):
                raise ReplayContractError(
                    f"source dataset admission gates are incomplete: {src}"
                )
        srows = _load_rows_any(src)
        taken = 0
        for r in srows:
            iid = r.get("instance_id")
            if iid in seen:
                continue
            if a.cap and taken >= a.cap:
                break
            seen.add(iid)
            rows.append({"instance_id": iid, "messages": r["messages"],
                         "repo": r.get("repo", ""), "source": r.get("source", src)})
            stats[src] += 1
            taken += 1
        evidence = {"path": src, "rows": len(srows), "selected": taken}
        if source_manifest.is_file():
            evidence["manifest_sha256"] = _sha256_path(source_manifest)
        external_manifest = Path(str(src) + ".manifest.json")
        if external_manifest.is_file():
            evidence["external_manifest_sha256"] = _sha256_path(external_manifest)
        source_evidence.append(evidence)
    if not rows:
        print("[blend] no rows", flush=True)
        return 1
    manifest_holder = {}

    def build(stage: Path) -> None:
        from datasets import Dataset

        Dataset.from_list(rows).save_to_disk(stage)
        _write_jsonl(stage / "train.jsonl", rows)
        manifest = {
            "schema_version": 2,
            "kind": "teacher_blend",
            "sources": source_evidence,
            "cap_per_source": a.cap,
            "total_rows": len(rows),
            "rendered": len(rows),
            "training_admitted": 0,
            "all_training_gates_complete": False,
            "per_source": dict(stats),
            "dedup": "by instance_id, earlier source wins",
            "train_jsonl_sha256": _sha256_path(stage / "train.jsonl"),
            "standard_native_format_loss_gate": {
                "status": "pending_full_dataset_verification",
                "failure_count": None,
            },
        }
        (stage / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n"
        )
        manifest_holder.update(manifest)

    _publish_directory_atomic(Path(a.out), build)
    man = manifest_holder
    print(f"[blend] {dict(stats)} -> {len(rows)} rows -> {a.out}", flush=True)
    return 0


# --------------------------------------------------------------------------- #
# smoke-train                                                                  #
# --------------------------------------------------------------------------- #
def _load_rows_any(path):
    """Load message rows from an HF dataset dir (save_to_disk) or a jsonl file."""
    if os.path.isdir(path):
        from datasets import load_from_disk
        return list(load_from_disk(path))
    return _read_jsonl(path)


def _replace_json_atomic(path: Path, value: dict) -> None:
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def _attach_teacher_format_gate(data: Path, report: dict) -> bool:
    manifest_path = data / "manifest.json"
    if not manifest_path.is_file():
        return False
    try:
        manifest = json.loads(manifest_path.read_text())
    except json.JSONDecodeError as exc:
        raise ReplayContractError("prepared dataset manifest is invalid") from exc
    if manifest.get("schema_version") != 2:
        return False
    train_jsonl = data / "train.jsonl"
    if (
        manifest.get("rendered") != report.get("samples")
        or report.get("failure_count") != 0
        or report.get("failures") != []
        or manifest.get("train_jsonl_sha256") != _sha256_path(train_jsonl)
    ):
        raise ReplayContractError("full-dataset format/loss report is incomplete or failed")
    payload = canonical_json_bytes(report) + b"\n"
    digest = sha256_bytes(payload)
    artifact_name = f"format-gate.{digest}.json"
    artifact = data / artifact_name
    descriptor = os.open(artifact, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        artifact.unlink(missing_ok=True)
        raise
    updated = dict(manifest)
    updated["training_admitted"] = manifest["rendered"]
    updated["all_training_gates_complete"] = True
    updated["standard_native_format_loss_gate"] = {
        "status": "passed",
        "samples": report["samples"],
        "failure_count": 0,
        "counts": dict(report.get("counts", {})),
        "supervised_tokens": dict(report.get("supervised_tokens", {})),
        "artifact": {
            "path": artifact_name,
            "sha256": digest,
            "bytes": len(payload),
        },
    }
    _replace_json_atomic(manifest_path, updated)
    return True


def cmd_smoke_train(a) -> int:
    rows = _load_rows_any(a.data)
    print(f"[smoke] dataset rows={len(rows)}", flush=True)
    # 1) structural gate
    bad = 0
    for r in rows:
        m = r.get("messages") or []
        if not m or m[0]["role"] != "system":
            bad += 1
            continue
        if not any(msg["role"] == "assistant" and msg.get("tool_calls") for msg in m):
            bad += 1
    if bad:
        print(f"[smoke] STRUCTURAL FAIL: {bad}/{len(rows)} rows missing system or assistant tool_call", flush=True)
        return 1
    print(f"[smoke] structural gate PASS ({len(rows)} rows)", flush=True)
    # the format verifier and trainer need the TRAIN venv (transformers/unsloth),
    # not the eval venv the platform runs in.
    train_py = a.train_python if os.path.exists(a.train_python) else sys.executable
    # 2) Gemma format-loss gate (reuse existing verifier if present)
    verifier = ROOT / "phaseD_sft" / "verify_gemma_format_loss.py"
    if verifier.exists() and not a.skip_format:
        n = len(rows)
        report_dir = Path(tempfile.mkdtemp(prefix=".teacher-format-gate."))
        report_path = report_dir / "report.json"
        cmd = [
            train_py,
            str(verifier),
            "--data",
            a.data,
            "--samples",
            str(n),
            "--report-out",
            str(report_path),
        ]
        print(f"[smoke] format-loss gate: {' '.join(cmd)}", flush=True)
        try:
            rc = subprocess.run(cmd, cwd=str(ROOT)).returncode
            if rc != 0:
                print("[smoke] FORMAT-LOSS GATE FAILED", flush=True)
                return 1
            try:
                report = json.loads(report_path.read_text())
            except (OSError, json.JSONDecodeError) as exc:
                raise ReplayContractError("format verifier did not publish a valid report") from exc
            attached = (
                _attach_teacher_format_gate(Path(a.data), report)
                if Path(a.data).is_dir()
                else False
            )
            print(
                f"[smoke] format-loss gate PASS ({n}/{n}; manifest_bound={attached})",
                flush=True,
            )
        finally:
            shutil.rmtree(report_dir, ignore_errors=True)
    # 3) optional 2-step micro-train (only if a trainer + GPU are requested)
    if a.micro_train:
        manifest_path = Path(a.data) / "manifest.json"
        if manifest_path.is_file():
            manifest = json.loads(manifest_path.read_text())
            if (
                manifest.get("schema_version") == 2
                and manifest.get("all_training_gates_complete") is not True
            ):
                print("[smoke] MICRO-TRAIN BLOCKED: dataset admission gates are incomplete", flush=True)
                return 1
        trainer = ROOT / "phaseD_sft" / "train_rust_lora.py"
        cmd = [train_py, str(trainer), "--data", a.data, "--out",
               a.micro_out, "--max-seq", str(a.micro_max_seq), "--epochs", "1",
               "--max-steps", "2", "--save-steps", "2", "--rank", "16"]
        print(f"[smoke] micro-train (2 steps): {' '.join(cmd)}", flush=True)
        rc = subprocess.run(cmd, cwd=str(ROOT)).returncode
        if rc != 0:
            print("[smoke] MICRO-TRAIN FAILED", flush=True)
            return 1
        print("[smoke] micro-train PASS", flush=True)
    print("[smoke] ALL GATES PASSED — dataset is training-ready", flush=True)
    return 0


# --------------------------------------------------------------------------- #
# cli                                                                          #
# --------------------------------------------------------------------------- #
def build_parser() -> argparse.ArgumentParser:
    def positive_int(value: str) -> int:
        try:
            parsed = int(value)
            return _require_positive_int(parsed, "value")
        except ValueError as exc:
            raise argparse.ArgumentTypeError(str(exc)) from exc

    def nonnegative_int(value: str) -> int:
        try:
            parsed = int(value)
            return _require_nonnegative_int(parsed, "value")
        except ValueError as exc:
            raise argparse.ArgumentTypeError(str(exc)) from exc

    ap = argparse.ArgumentParser(description="Teacher-trace platform")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("pool", help="build decontaminated candidate pool from SWE-smith")
    p.add_argument("--out", required=True)
    p.add_argument("--vetted-pool", default="data/expert_iter1_pool.jsonl",
                   help="existing decontaminated pool; its repos are reused as vetted set")
    p.add_argument("--exclude", nargs="*", default=[],
                   help="jsonl files/globs whose instance_ids to exclude (already attempted)")
    p.add_argument("--per-repo", type=int, default=48)
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p.add_argument("--dataset", default="SWE-bench/SWE-smith")
    p.add_argument("--split", default="train")
    p.set_defaults(fn=cmd_pool)

    p = sub.add_parser("hard", help="select base-failed hard tasks from a base k-sampling ledger")
    p.add_argument("--labels", required=True, help="base k-sampling results.jsonl (per-rollout resolved)")
    p.add_argument("--pool", required=True, help="task pool with full metadata to join")
    p.add_argument("--out", required=True)
    p.add_argument(
        "--samples-per-instance",
        type=positive_int,
        default=4,
        help="require exactly this many clean base samples before calling a task hard",
    )
    p.add_argument("--exclude", nargs="*", default=[])
    p.set_defaults(fn=cmd_hard)

    p = sub.add_parser("collect", help="run a teacher over tasks, F2P-verify, self-heal on quota walls")
    p.add_argument("--tasks", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--backend", choices=("claude", "codex", "openrouter"), default="claude")
    p.add_argument("--model", default=None)
    p.add_argument("--max-turns", type=int, default=40)
    p.add_argument("--loop", action="store_true", help="self-healing loop until pool exhausted")
    p.add_argument("--max-consecutive-credit-hits", type=int, default=3)
    p.add_argument("--max-walls", type=int, default=3)
    p.add_argument("--wall-sleep", type=int, default=1200)
    p.add_argument("--rate-limit-sleep", type=int, default=300,
                   help="seconds to back off after an upstream HTTP 429")
    p.add_argument("--floor-gib", type=int, default=40)
    p.add_argument("--exclude-runs", nargs="*", default=[],
                   help="run dir globs (e.g. 'runs/teacher_*') whose attempted "
                        "problems to skip — guarantees no overlap across campaigns")
    p.set_defaults(fn=cmd_collect)

    p = sub.add_parser("merge", help="consolidate collect batches into a training-ready set")
    merge_inputs = p.add_mutually_exclusive_group(required=True)
    merge_inputs.add_argument(
        "--glob",
        help="glob of batch dirs (e.g. 'runs/teacher_*')",
    )
    merge_inputs.add_argument(
        "--input",
        dest="inputs",
        action="append",
        help="exact run directory or results.jsonl; repeat for each input",
    )
    p.add_argument(
        "--exclude-instance-ids",
        action="append",
        default=[],
        help="bound JSON/JSONL artifact of source IDs to omit",
    )
    p.add_argument("--out-dir", required=True)
    p.set_defaults(fn=cmd_merge)

    p = sub.add_parser(
        "revalidate",
        help="fresh baseline/reference/candidate-twice replay for raw teacher artifacts",
    )
    p.add_argument(
        "--glob",
        nargs="+",
        required=True,
        help="one or more exact run-dir/results.jsonl paths or globs",
    )
    p.add_argument("--tasks", nargs="+", required=True)
    p.add_argument("--exclude", nargs="+", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--test-timeout", type=positive_int, default=900)
    p.set_defaults(fn=cmd_revalidate)

    p = sub.add_parser(
        "repreflight",
        help="recover immutable traces rejected by the collector evidence bug",
    )
    p.add_argument("--input", required=True, help="exact raw results.jsonl")
    p.add_argument("--tasks", nargs="+", required=True)
    p.add_argument("--out-dir", required=True)
    p.set_defaults(fn=cmd_repreflight)

    p = sub.add_parser("prepare", help="render resolved traces into mini-SWE SFT jsonl")
    p.add_argument("--merged", required=True, help="merge out-dir (has resolved.jsonl + streams)")
    p.add_argument("--tasks", nargs="+", required=True,
                   help="task jsonl(s)/globs with problem_statement to join by instance_id")
    p.add_argument(
        "--exclude",
        nargs="+",
        required=True,
        help="explicit eval/decontamination JSON or JSONL artifacts; hashes are bound",
    )
    p.add_argument("--out", required=True)
    p.set_defaults(fn=cmd_prepare)

    p = sub.add_parser("ingest", help="ingest an external resolved-trajectory dataset (e.g. Open-SWE-Traces) into SFT")
    p.add_argument("--out", required=True)
    p.add_argument("--dataset", default="nvidia/Open-SWE-Traces")
    p.add_argument("--configs", nargs="+", default=["sweagent", "openhands"])
    p.add_argument("--language", default="python")
    p.add_argument("--resolved-only", action="store_true", default=True)
    p.add_argument("--all-outcomes", dest="resolved_only", action="store_false",
                   help="keep unresolved trajectories too (default: resolved only)")
    p.add_argument("--limit", type=nonnegative_int, default=0, help="0 = no cap")
    p.add_argument("--max-obs-chars", type=positive_int, default=800,
                   help="truncate each observation to this many chars (head+tail)")
    p.add_argument("--max-steps", type=nonnegative_int, default=25,
                   help="edit-first trim long trajectories to this many steps (0 = keep all)")
    p.add_argument("--max-pr-chars", type=nonnegative_int, default=6000,
                   help="truncate problem statement (0 = keep all)")
    p.add_argument("--progress-every", type=positive_int, default=10_000,
                   help="emit JSON ingest progress every N scanned rows")
    p.add_argument("--exclude", nargs="*", default=[],
                   help="jsonl files/globs of eval + already-used instance_ids to exclude (decontam)")
    p.set_defaults(fn=cmd_ingest)

    p = sub.add_parser("blend", help="combine SFT sources into one training dataset (dedup, per-source cap)")
    p.add_argument("--sources", nargs="+", required=True,
                   help="prepared/ingested SFT dataset dirs or jsonls; earlier wins on dedup")
    p.add_argument("--out", required=True)
    p.add_argument("--cap", type=int, default=0, help="max rows per source (0 = all)")
    p.set_defaults(fn=cmd_blend)

    p = sub.add_parser("smoke-train", help="validate a prepared dataset is trainable")
    p.add_argument("--data", required=True)
    p.add_argument("--format-samples", type=int, default=200)
    p.add_argument("--skip-format", action="store_true")
    p.add_argument("--train-python", default=str(ROOT / ".venv-train" / "bin" / "python"),
                   help="python with transformers/unsloth for the format+train gates")
    p.add_argument("--micro-train", action="store_true", help="also run a 2-step LoRA micro-run (needs GPU)")
    p.add_argument("--micro-out", default="adapters/_smoke_micro")
    p.add_argument(
        "--micro-max-seq",
        type=positive_int,
        default=8192,
        help="context window for the optional two-step LoRA micro-run",
    )
    p.set_defaults(fn=cmd_smoke_train)

    return ap


def main() -> int:
    a = build_parser().parse_args()
    return a.fn(a)


if __name__ == "__main__":
    raise SystemExit(main())
