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
           (trainable positives) + copied artifacts + manifest.
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
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent            # teacher_platform/
ROOT = HERE.parent                                # repo root
sys.path.insert(0, str(ROOT / "phaseD_sft"))      # teacher_trace_driver lives here
import teacher_trace_driver as ttd  # noqa: E402  (claude/codex collect + credit detector + F2P)

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


def _is_real_attempt(rec: dict) -> bool:
    """A row is a real teacher attempt (not a quota-wall no-op) iff it produced
    agent turns or a patch. Walls are 0-event / 0-patch and MUST NOT count."""
    return int(rec.get("n_assistant_events", 0) or 0) > 2 or int(rec.get("patch_len", 0) or 0) > 0


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
    solved = set()
    seen = set()
    for r in labels:
        iid = r.get("source_instance_id") or r.get("instance_id")
        if not iid:
            continue
        seen.add(iid)
        if r.get("resolved") in (True, "True"):
            solved.add(iid)
    failed = seen - solved
    pool = {r["instance_id"]: r for r in _read_jsonl(a.pool)}
    exclude = _attempted_ids(*a.exclude) if a.exclude else set()
    hard = [pool[i] for i in sorted(failed) if i in pool and i not in exclude]
    _write_jsonl(a.out, hard)
    man = {
        "labels": str(a.labels),
        "instances_seen": len(seen),
        "base_solved": len(solved),
        "base_failed": len(failed),
        "hard_with_metadata": len(hard),
        "excluded": len(failed) - len(hard),
        "note": "hard = base failed ALL k samples; these resolved by a teacher = capability-gap gold",
    }
    Path(str(a.out) + ".manifest.json").write_text(json.dumps(man, indent=1))
    print(f"[hard] base seen={len(seen)} solved={len(solved)} FAILED={len(failed)} "
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
                    if _is_real_attempt(rec):
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

    def one_pass(tasks_file: str, outdir: Path) -> None:
        outdir.mkdir(parents=True, exist_ok=True)
        rows = _read_jsonl(tasks_file)
        ledger = outdir / "results.jsonl"
        results = _read_jsonl(ledger) if ledger.exists() and ledger.stat().st_size else []
        rows = ttd.remaining_task_rows(rows, results)
        rows = [r for r in rows if r["instance_id"] not in cross_run_done]
        streak = ttd.trailing_credit_streak(results)
        for row in rows:
            if a.backend == "openrouter":
                rec = collect_one_openrouter(row, outdir, a.max_turns, model)
            else:
                rec = ttd.collect_one(row, outdir, a.max_turns, model, a.backend)
            results.append(rec)
            print(json.dumps({k: rec.get(k) for k in
                              ("instance_id", "n_assistant_events", "patch_len", "resolved", "error")}),
                  flush=True)
            with open(ledger, "w") as fh:
                for x in results:
                    fh.write(json.dumps(x) + "\n")
            if _is_fatal_error(rec.get("error")):
                raise RuntimeError(f"fatal config error (bad model id / key?), aborting: {rec.get('error')}")
            streak = ttd.next_credit_streak(streak, rec.get("error"))
            if streak >= a.max_consecutive_credit_hits:
                print(f"[collect] {streak} consecutive credit hits -> pausing pass", flush=True)
                break

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
                walls = 0
                tmp = out_root.parent / f".{out_root.name}_remaining.jsonl"
                _write_jsonl(tmp, rem)
                one_pass(str(tmp), out_root.parent / f"{out_root.name}_run{it}")
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
    rec = {"instance_id": iid, "image": row["image_name"], "model": model, "backend": "openrouter"}
    raw_path = outdir / f"{iid}.stream.jsonl"
    stream = open(raw_path, "w")
    events = 0
    try:
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
                obs, _ = ttd.sh(["docker", "exec", cid, "bash", "-c", f"cd /testbed && {cmd}"], 300)
                obs = obs[-4000:]
                stream.write(json.dumps({"role": "tool", "command": cmd, "observation": obs}) + "\n")
                messages.append({"role": "tool", "tool_call_id": tc.id, "content": obs})
        stream.close()
        rec["n_assistant_events"] = events
        rec["cost_usd"] = 0.0
        diff, _ = ttd.sh(["docker", "exec", cid, "bash", "-c", "cd /testbed && git diff"], 120)
        rec["patch_len"] = len(diff.strip())
        (outdir / f"{iid}.patch").write_text(diff)
        resolved = False
        if diff.strip():
            invocations = ttd.f2p_invocations(list(row.get("FAIL_TO_PASS") or []))
            rec["f2p_cmds"] = len(invocations)
            ok = True
            for tc in invocations or ["false"]:
                _, trc = ttd.sh(["docker", "exec", cid, "bash", "-c",
                                 f"cd /testbed && {ttd.ENV_BOOTSTRAP}{tc}"], 600)
                if trc != 0:
                    ok = False
                    break
            resolved = ok and bool(invocations)
        rec["resolved"] = resolved
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
def cmd_merge(a) -> int:
    real = {}
    for ledger in sorted(globmod.glob(a.glob.rstrip("/") + "/results.jsonl")
                         if not a.glob.endswith("results.jsonl") else globmod.glob(a.glob)):
        batch = Path(ledger).parent.name
        src = Path(ledger).parent
        for rec in _read_jsonl(ledger):
            if _is_real_attempt(rec):
                rec = dict(rec, batch=batch, _src=str(src))
                real[rec["instance_id"]] = rec  # last real attempt wins
    rows = list(real.values())
    res = [r for r in rows if r.get("resolved")]
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    _write_jsonl(out / "results.jsonl", [{k: v for k, v in r.items() if k != "_src"} for r in rows])
    _write_jsonl(out / "resolved.jsonl", [{k: v for k, v in r.items() if k != "_src"} for r in res])
    copied = 0
    for r in res:
        for ext in (".stream.jsonl", ".patch"):
            s = Path(r["_src"]) / f"{r['instance_id']}{ext}"
            if s.exists():
                shutil.copy2(s, out / s.name)
                copied += 1
    man = {"attempted": len(rows), "resolved": len(res),
           "resolve_rate": round(len(res) / max(len(rows), 1), 3),
           "per_batch": dict(Counter(r["batch"] for r in rows)),
           "resolved_per_batch": dict(Counter(r["batch"] for r in res)),
           "artifacts_copied": copied}
    (out / "manifest.json").write_text(json.dumps(man, indent=1))
    print(f"[merge] attempted={len(rows)} resolved={len(res)} "
          f"({man['resolve_rate']:.0%}) -> {out}", flush=True)
    return 0


# --------------------------------------------------------------------------- #
# prepare  (render resolved teacher traces -> mini-SWE SFT, shape-safe)        #
# --------------------------------------------------------------------------- #
_DOCKER_EXEC_RE = re.compile(r'^\s*docker\s+exec\s+\S+\s+bash\s+-c\s+["\'](.*)["\']\s*$', re.DOTALL)


def _strip_docker_exec(cmd: str) -> str:
    """Convert a teacher `docker exec cid bash -c "cd /testbed && X"` into the
    bare `X` the student runs directly. Prevents learning teacher-harness shape."""
    m = _DOCKER_EXEC_RE.match(cmd.strip())
    inner = m.group(1) if m else cmd.strip()
    inner = re.sub(r'^\s*cd\s+/testbed\s*&&\s*', "", inner)
    return inner.strip()


def normalize_steps(backend: str, stream_path: Path):
    """Parse a raw teacher stream into ordered [{thought, command, observation}].
    Returns [] if unparseable. THOUGHT is the terse reasoning before each command."""
    steps = []
    try:
        lines = [json.loads(l) for l in open(stream_path, errors="replace") if l.strip().startswith("{")]
    except Exception:
        return steps
    if backend == "openrouter":
        pending = None
        for d in lines:
            if d.get("role") == "assistant":
                pending = {"thought": (d.get("content") or "").strip(),
                           "cmds": list(d.get("tool_calls") or [])}
            elif d.get("role") == "tool" and pending is not None:
                cmd = _strip_docker_exec(d.get("command", ""))
                if cmd:
                    steps.append({"thought": pending["thought"], "command": cmd,
                                  "observation": (d.get("observation") or "")[-2000:]})
    elif backend == "claude":
        thought = ""
        for d in lines:
            if d.get("type") == "assistant":
                for block in (d.get("message", {}) or {}).get("content", []):
                    if block.get("type") == "text":
                        thought = (block.get("text") or "").strip()
                    elif block.get("type") == "tool_use" and block.get("name") == "Bash":
                        cmd = _strip_docker_exec((block.get("input") or {}).get("command", ""))
                        if cmd:
                            steps.append({"thought": thought, "command": cmd, "observation": ""})
                            thought = ""
            elif d.get("type") == "user":
                for block in (d.get("message", {}) or {}).get("content", []):
                    if block.get("type") == "tool_result" and steps and not steps[-1]["observation"]:
                        c = block.get("content")
                        txt = c if isinstance(c, str) else " ".join(
                            b.get("text", "") for b in (c or []) if isinstance(b, dict))
                        steps[-1]["observation"] = (txt or "")[-2000:]
    elif backend == "codex":
        thought = ""
        for d in lines:
            if d.get("type") != "item.completed":
                continue
            item = d.get("item") or {}
            if item.get("type") == "agent_message":
                thought = (item.get("text") or "").strip()
            elif item.get("type") == "command_execution":
                cmd = _strip_docker_exec(item.get("command", ""))
                if cmd:
                    steps.append({"thought": thought, "command": cmd,
                                  "observation": (item.get("aggregated_output") or "")[-2000:]})
                    thought = ""
    return steps


BASH_TOOL_NAME = "bash"


def _msg(role, content, tool_calls=None):
    """Uniform message schema (every row carries a tool_calls key) matching the
    existing SFT datasets so the HF arrow schema stays consistent and the Gemma
    template renders it."""
    return {"role": role, "content": content, "tool_calls": tool_calls or []}


def render_sft(row: dict, steps: list) -> dict | None:
    """mini-SWE SFT row in the exact existing schema: system + PR problem, then
    per step an assistant turn (THOUGHT in content + one bash tool_call) and a
    user OBSERVATION turn. Edit-first shape inherited from the teacher prompt."""
    if not steps:
        return None
    system = ("You are a practical software engineer using a shell to fix one repository "
              "bug. Prefer a small correct source edit over extended inspection. Before each "
              "command write 1-3 terse sentences of reasoning, then emit one bash tool call.")
    messages = [_msg("system", system),
                _msg("user", f"<pr_description>\n{row['problem_statement']}\n</pr_description>")]
    for i, s in enumerate(steps):
        tc = [{"function": {"arguments": json.dumps({"command": s["command"]}), "name": BASH_TOOL_NAME},
               "id": f"teacher-tool-{i}", "type": "function"}]
        messages.append(_msg("assistant", s["thought"], tc))
        messages.append(_msg("user", f"OBSERVATION:\n{s['observation']}"))
    return {"instance_id": row["instance_id"], "messages": messages,
            "repo": row.get("repo", ""),
            "source": f"teacher:{row.get('backend', '?')}:{row.get('model', '?')}",
            "n_steps": len(steps)}


def cmd_prepare(a) -> int:
    merged = Path(a.merged)
    resolved = _read_jsonl(merged / "resolved.jsonl")
    # result ledgers carry no problem_statement -> join task metadata by id
    taskmeta = {}
    for tp in a.tasks:
        for f in ([tp] if os.path.exists(tp) else globmod.glob(tp)):
            for row in _read_jsonl(f):
                taskmeta.setdefault(row["instance_id"], row)
    out_rows, dropped, no_meta = [], 0, 0
    stats = Counter()
    for r in resolved:
        meta = taskmeta.get(r["instance_id"])
        if not meta:
            no_meta += 1
            continue
        stream = merged / f"{r['instance_id']}.stream.jsonl"
        steps = normalize_steps(r.get("backend", "claude"), stream) if stream.exists() else []
        # render off task metadata (problem_statement) but tag with teacher backend/model
        render_row = dict(meta, backend=r.get("backend"), model=r.get("model"))
        row = render_sft(render_row, steps) if steps else None
        if row is None:
            dropped += 1
            continue
        # edit-first sanity: first source-editing command index
        first_edit = next((i for i, s in enumerate(steps)
                           if re.search(r'\b(sed -i|>|>>|tee |patch |apply|cat <<)', s["command"])), None)
        row["first_edit_cmd"] = first_edit
        stats[r.get("backend", "?")] += 1
        out_rows.append(row)
    if not out_rows:
        print("[prepare] no rows rendered — nothing to write", flush=True)
        return 1
    # primary output = HF dataset dir (what verify_gemma_format_loss + trainer consume);
    # drop the transient first_edit_cmd from the saved columns, keep in manifest.
    first_edits = [x.pop("first_edit_cmd", None) for x in out_rows]
    from datasets import Dataset

    Dataset.from_list(out_rows).save_to_disk(a.out)
    _write_jsonl(str(a.out) + ".jsonl", out_rows)  # human-inspectable copy
    fe = sorted(e for e in first_edits if e is not None)
    man = {"resolved_in": len(resolved), "rendered": len(out_rows),
           "dropped_unparseable": dropped, "dropped_no_task_metadata": no_meta,
           "per_backend": dict(stats),
           "median_first_edit_cmd": fe[len(fe) // 2] if fe else None,
           "output": f"HF dataset dir at {a.out} (+ {a.out}.jsonl); columns instance_id/messages/repo/source",
           "format": "mini-swe SFT: system+PR, assistant(THOUGHT+bash tool_call), user(OBSERVATION)"}
    Path(str(a.out) + ".manifest.json").write_text(json.dumps(man, indent=1))
    print(f"[prepare] resolved={len(resolved)} rendered={len(out_rows)} "
          f"dropped={dropped} no_meta={no_meta} -> {a.out} (HF dataset dir)", flush=True)
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


def _trajectory_testbed_path(arguments: dict) -> str:
    path = _required_trajectory_string(arguments, "path", nonempty=True)
    normalized = os.path.normpath(path)
    if normalized != "/testbed" and not normalized.startswith("/testbed/"):
        raise UnsupportedTrajectoryTool("editor path must be under /testbed")
    return normalized


def _quoted_python_editor(lines: list[str]) -> str:
    return (
        "python3 - <<'OPEN_SWE_PY'\n"
        "# OPEN_SWE_EDITOR_MUTATION\n"
        + "\n".join(lines)
        + "\nOPEN_SWE_PY"
    )


def translate_trajectory_tool_call(tool_call: dict) -> str | None:
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
    if name == "bash":
        command = _required_trajectory_string(arguments, "command", nonempty=True)
        command = _strip_docker_exec(command).strip()
        if not command:
            raise UnsupportedTrajectoryTool("bash command is empty after wrapper cleanup")
        return command
    if name != "str_replace_editor":
        raise UnsupportedTrajectoryTool(f"unsupported trajectory tool: {name}")

    editor_command = _required_trajectory_string(arguments, "command", nonempty=True)
    path = _trajectory_testbed_path(arguments)
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
            command = translate_trajectory_tool_call(tcs[0])
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
            if pending_tool_call_id is not None or result_has_id:
                if result_id != pending_tool_call_id:
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
                if omitted_tool_call_id is not None or result_has_id:
                    if result_id != omitted_tool_call_id:
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


def cmd_ingest(a) -> int:
    """Ingest an external resolved-trajectory dataset into mini-SWE SFT rows."""
    import ast

    _require_nonnegative_int(a.limit, "limit")
    _require_positive_int(a.max_obs_chars, "max_obs_chars")
    _require_nonnegative_int(a.max_steps, "max_steps")
    _require_nonnegative_int(a.max_pr_chars, "max_pr_chars")

    from datasets import Dataset, load_dataset

    exclude = _attempted_ids(*a.exclude)
    out, seen = [], set()
    stats = Counter()
    tool_conversion_drop_reasons = Counter()
    scanned = kept_resolved = 0
    for cfg in a.configs:
        d = load_dataset(a.dataset, cfg, streaming=True)
        for split in d.keys():
            for r in d[split]:
                scanned += 1
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
            if a.limit and len(out) >= a.limit:
                break
        if a.limit and len(out) >= a.limit:
            break
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
    from datasets import Dataset

    rows, seen = [], set()
    stats = Counter()
    for src in a.sources:
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
    if not rows:
        print("[blend] no rows", flush=True)
        return 1
    Dataset.from_list(rows).save_to_disk(a.out)
    _write_jsonl(str(a.out) + ".jsonl", rows)
    man = {"sources": a.sources, "cap_per_source": a.cap, "total_rows": len(rows),
           "per_source": dict(stats), "dedup": "by instance_id, earlier source wins"}
    Path(str(a.out) + ".manifest.json").write_text(json.dumps(man, indent=1))
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
        n = min(len(rows), a.format_samples)
        cmd = [train_py, str(verifier), "--data", a.data, "--samples", str(n)]
        print(f"[smoke] format-loss gate: {' '.join(cmd)}", flush=True)
        rc = subprocess.run(cmd, cwd=str(ROOT)).returncode
        if rc != 0:
            print("[smoke] FORMAT-LOSS GATE FAILED", flush=True)
            return 1
        print("[smoke] format-loss gate PASS", flush=True)
    # 3) optional 2-step micro-train (only if a trainer + GPU are requested)
    if a.micro_train:
        trainer = ROOT / "phaseD_sft" / "train_rust_lora.py"
        cmd = [train_py, str(trainer), "--data", a.data, "--out",
               a.micro_out, "--max-seq", "8192", "--epochs", "1",
               "--train-steps", "2", "--save-steps", "2", "--lora-r", "16"]
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
    p.add_argument("--max-walls", type=int, default=48)
    p.add_argument("--wall-sleep", type=int, default=1200)
    p.add_argument("--floor-gib", type=int, default=40)
    p.add_argument("--exclude-runs", nargs="*", default=[],
                   help="run dir globs (e.g. 'runs/teacher_*') whose attempted "
                        "problems to skip — guarantees no overlap across campaigns")
    p.set_defaults(fn=cmd_collect)

    p = sub.add_parser("merge", help="consolidate collect batches into a training-ready set")
    p.add_argument("--glob", required=True, help="glob of batch dirs (e.g. 'runs/teacher_*')")
    p.add_argument("--out-dir", required=True)
    p.set_defaults(fn=cmd_merge)

    p = sub.add_parser("prepare", help="render resolved traces into mini-SWE SFT jsonl")
    p.add_argument("--merged", required=True, help="merge out-dir (has resolved.jsonl + streams)")
    p.add_argument("--tasks", nargs="+", required=True,
                   help="task jsonl(s)/globs with problem_statement to join by instance_id")
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
    p.set_defaults(fn=cmd_smoke_train)

    return ap


def main() -> int:
    a = build_parser().parse_args()
    return a.fn(a)


if __name__ == "__main__":
    raise SystemExit(main())
