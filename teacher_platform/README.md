# teacher_platform

One CLI for the capability-gap teacher-trace pipeline: find problems the **base
model fails**, have a **teacher** (Claude / Codex / OpenRouter) solve and
strictly replay them against clean baseline, reference, and repeated candidate
controls, and render only hash-bound solutions into an SFT dataset. A
full-dataset format/loss gate must pass before the trainer admits it.

## Layout
- `python -m teacher_platform` — the CLI:
  `pool | hard | collect | revalidate | merge | prepare | ingest | blend | smoke-train`
- `INSTRUCTIONS.md` — step-by-step runbook for an operator/agent
- `tests/` — unit tests for the pure logic (selection, merge, render, dud-filter)

## Quick start (on the box, repo root)
```
PY=.venv-eval/bin/python
PLAT="$PY -m teacher_platform"
EXCLUSIONS="data/swe_verified_eval_exclusions_v1.json data/python_eval_repo_denylist_v1.json"

$PLAT hard   --labels runs/expert_iter1_rawbase_k4/results.jsonl \
             --pool data/expert_iter1_pool.jsonl --out data/hard_tasks.jsonl
$PLAT collect --tasks data/hard_tasks.jsonl --out-dir runs/teacher_claude --backend claude --loop
$PLAT merge   --glob 'runs/teacher_*' --out-dir runs/teacher_merged
$PLAT prepare --merged runs/teacher_merged --tasks data/hard_tasks.jsonl \
              --exclude $EXCLUSIONS --out data/teacher_sft
$PLAT smoke-train --data data/teacher_sft
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
2. **Strict verified positives only**: clean baseline F2P failure with P2P
   preserved, reference pass, candidate pass twice, and protected hashes stable.
3. **Dud ≠ failure** — quota-wall no-ops (`n_assistant_events<=2 & patch_len=0`)
   are excluded everywhere.
4. **Shape-safe render** — `prepare` strips the teacher's `docker exec` wrapper
   so the student learns bare `/testbed` commands, not the teacher harness.
5. **Decontaminated** — pool is repo-level filtered against Lite-30 / hard30.
6. **Content-bound and atomic** — task, stream, patch, controls, exclusions,
   rendered JSONL, and format report are SHA-256 bound; final directories never
   overwrite an existing artifact.

See `INSTRUCTIONS.md` for the full runbook including generating new hard problems
via base k-sampling (GPU).
