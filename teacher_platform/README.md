# teacher_platform

One CLI for the capability-gap teacher-trace pipeline: find problems the **base
model fails**, have a **teacher** (Claude / Codex / OpenRouter) solve and
**F2P-verify** them, and render the verified solutions into a **training-ready
SFT dataset** — then smoke-test that it trains.

## Layout
- `teacher_platform.py` — the CLI:
  `pool | hard | collect | merge | prepare | ingest | blend | smoke-train`
- `INSTRUCTIONS.md` — step-by-step runbook for an operator/agent
- `tests/` — unit tests for the pure logic (selection, merge, render, dud-filter)

## Quick start (on the box, repo root)
```
PY=.venv-eval/bin/python
PLAT="$PY teacher_platform/teacher_platform.py"

$PLAT hard   --labels runs/expert_iter1_rawbase_k4/results.jsonl \
             --pool data/expert_iter1_pool.jsonl --out data/hard_tasks.jsonl
$PLAT collect --tasks data/hard_tasks.jsonl --out-dir runs/teacher_claude --backend claude --loop
$PLAT merge   --glob 'runs/teacher_*' --out-dir runs/teacher_merged
$PLAT prepare --merged runs/teacher_merged --tasks data/hard_tasks.jsonl --out data/teacher_sft
$PLAT ingest  --out data/open_swe_sft --resolved-only --language python --exclude data/lite_eval_ids.jsonl
$PLAT blend   --sources data/teacher_sft data/open_swe_sft --out data/main_train_mix
$PLAT smoke-train --data data/main_train_mix
```

- **No overlap across runs:** `collect --exclude-runs 'runs/teacher_*'` skips any
  problem a prior campaign already attempted (duds ignored). Same globs work as
  `--exclude` on `pool`/`hard`.
- **Blend external traces:** `ingest` pulls verified real-issue trajectories
  (e.g. `nvidia/Open-SWE-Traces`, resolved+Python) into the same SFT format;
  `blend` merges them with your own teacher traces into one training set
  (dedup by instance_id, first source wins).

## Design invariants (each earned a failed adapter — do not remove)
1. **Only base-FAILED tasks** are worth teaching (`hard`).
2. **Verified positives only** (`resolved: true`, F2P in-container).
3. **Dud ≠ failure** — quota-wall no-ops (`n_assistant_events<=2 & patch_len=0`)
   are excluded everywhere.
4. **Shape-safe render** — `prepare` strips the teacher's `docker exec` wrapper
   so the student learns bare `/testbed` commands, not the teacher harness.
5. **Decontaminated** — pool is repo-level filtered against Lite-30 / hard30.

See `INSTRUCTIONS.md` for the full runbook including generating new hard problems
via base k-sampling (GPU).
