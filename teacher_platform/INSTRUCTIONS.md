# Teacher-Trace Platform — Agent Runbook

Goal: collect teacher (Claude/Codex/OpenRouter) solutions to problems the **base
model cannot solve**, F2P-verify them, and turn them into a training-ready SFT
dataset — then smoke-test that it trains. One CLI does every step.

## WHERE to run (read this first — it is the #1 failure)

**Run everything ON THE GPU BOX**, never on a laptop/Mac:
`ssh ironbcc@192.168.50.148`, then `cd ~/projects/gemma4-31B-Coder`.

WHY it must be the box, not a Mac: collection starts a Docker container per task
from the SWE-smith **linux/amd64** images and drives it via `docker exec`. On an
arm64 Mac, `docker run` emits a `WARNING: The requested image's platform...` line
and the container never works — every command returns
`Error response from daemon: No such container`, so the "patch" is just that
error string and 0 traces are real. (This exact mistake wasted a whole kimi run.)
The box has the images, the venvs, and matching arch.

```
PY=.venv-eval/bin/python            # eval venv: has datasets/docker deps
PLAT="$PY -m teacher_platform"
```

## Backend model IDs (must be exact — a wrong slug 400s and now fail-fast aborts)

| backend | good `--model` values |
|---|---|
| `claude` (subscription) | `claude-fable-5` |
| `codex` (subscription) | `gpt-5.6-terra` |
| `openrouter` (needs `OPENROUTER_API_KEY`) | **`moonshotai/kimi-k3`** (Kimi3), `moonshotai/kimi-k2.7-code`, `anthropic/claude-3.7-sonnet` |

For OpenRouter, the slug must match `https://openrouter.ai/api/v1/models` exactly.
`moonshotai/kimi-3` is **NOT valid** (it's `kimi-k3`). A bad slug now aborts the run
immediately instead of erroring through every task.

## Resume: just re-run the same `collect` command

`collect --loop` is stateful via the run ledger. If it pauses on a quota wall or
you stop it, **re-run the identical command** — it skips finished tasks and
continues. Nothing else to reset.

Hard rules (do not skip — each earned a failed adapter):
- **Only train on base-FAILED tasks.** Traces on tasks the base already solves
  teach nothing (4 prior adapters regressed this way). Use `hard` to filter.
- **Never let eval leak in.** The pool is decontaminated at the repo level
  (Lite-30 / hard30 excluded). Don't add repos without re-checking.
- **Duds ≠ failures.** A quota-wall row has `n_assistant_events<=2` and
  `patch_len=0`; it means the task never ran. The tools already exclude these —
  never count them as teacher misses.
- **A row is verified only with complete strict evidence**: a fresh baseline
  fails F2P while P2P passes, the reference passes, the candidate passes
  F2P+P2P twice, protected hashes remain stable, and every artifact hash
  matches. Candidate-only F2P labels from older runs are diagnostics.

---

## Step 0 — Get hard problems

You need a set of tasks the base failed. Two ways:

**A) Reuse existing base labels** (fast, no GPU). If a base k-sampling ledger
exists (e.g. `runs/expert_iter1_rawbase_k4/results.jsonl`):

```
$PLAT hard \
  --labels runs/expert_iter1_rawbase_k4/results.jsonl \
  --pool   data/expert_iter1_pool.jsonl \
  --out    data/hard_tasks.jsonl \
  --exclude 'runs/teacher_*/results.jsonl'      # skip anything already attempted
```

**B) Generate new hard problems** (needs a base serve on GPU — see NOTE):

```
# 1. build a bigger decontaminated candidate pool from SWE-smith
$PLAT pool --out data/candidate_pool.jsonl --per-repo 48 \
      --exclude data/expert_iter1_pool.jsonl 'runs/teacher_*/results.jsonl'

# 2. base k-samples the pool (GPU): serve base bf16 as gemma4-rawbase on :8012,
#    then run the sampler (this labels which tasks base fails)
.venv-eval/bin/python phaseD_sft/expert_iteration.py sample \
      --pool data/candidate_pool.jsonl --out runs/base_k4 \
      --api-base http://localhost:8012/v1 --model gemma4-rawbase \
      --samples-per-instance 4 --temperature 1.0 --workers 8

# 3. select the failures
$PLAT hard --labels runs/base_k4/results.jsonl \
      --pool data/candidate_pool.jsonl --out data/hard_tasks.jsonl
```

> NOTE (GPU): base serving lives on GPU1 or a freed GPU0. GPU0 is prod
> (`vllm.service`); freeing it needs `sudo systemctl stop vllm.service` (a human
> runs this) and restoring is `sudo systemctl start vllm.service`. Do not touch
> prod without an explicit instruction.

---

## Step 1 — Extract traces (teacher solves the hard tasks)

Pick a backend. Subscriptions are $0; OpenRouter costs per token.

```
# Claude / Fable-5 (subscription):
$PLAT collect --tasks data/hard_tasks.jsonl --out-dir runs/teacher_claude \
      --backend claude --model claude-fable-5 --loop

# Codex / GPT-5.6 (subscription):
$PLAT collect --tasks data/hard_tasks.jsonl --out-dir runs/teacher_codex \
      --backend codex --model gpt-5.6-terra --loop

# OpenRouter (any model; needs the key):
OPENROUTER_API_KEY=sk-... $PLAT collect --tasks data/hard_tasks.jsonl \
      --out-dir runs/teacher_or --backend openrouter \
      --model anthropic/claude-3.7-sonnet --loop
```

`--loop` is self-healing: it works only the not-yet-attempted tasks, pauses on
quota walls (does not burn the pool), prunes docker between passes, and stops
when the pool is exhausted. Safe to re-run; it resumes.

**No overlap across runs (important).** Within one `--out-dir` family the loop
already skips finished problems. To guarantee a NEW campaign never re-collects a
problem an EARLIER campaign already did (across different out-dirs, teachers, or
task files), pass `--exclude-runs`:

```
# second campaign (e.g. a different teacher) skips everything already attempted:
$PLAT collect --tasks data/hard_tasks.jsonl --out-dir runs/teacher_codex \
      --backend codex --loop --exclude-runs 'runs/teacher_*'
```

`--exclude-runs` unions the *real* attempts (dud/quota-wall rows are ignored)
from every matching run dir and removes them from this run's work list. Use the
same convention when building a fresh task file: `hard`/`pool` take `--exclude`
with the same globs, so each new pool is disjoint from all prior work.

Run it in the background and watch the ledger:
```
nohup $PLAT collect --tasks data/hard_tasks.jsonl --out-dir runs/teacher_claude \
      --backend claude --loop > /tmp/collect.log 2>&1 &
# progress:
python3 -c "import json; rows=[json.loads(l) for l in open('runs/teacher_claude/results.jsonl')]; \
  real=[r for r in rows if r['n_assistant_events']>2 or r['patch_len']>0]; \
  print('attempted',len(real),'resolved',sum(r['resolved'] for r in real))"
```

Collect from **multiple teachers** into sibling dirs (`runs/teacher_claude`,
`runs/teacher_codex`, …); merge pools them.

---

## Step 2 — Revalidate legacy or already-running extractions

New collections run strict controls automatically. Older collections and an
extraction started with pre-admission code must be replayed into a new
directory; never mutate the live extraction:

```
$PLAT revalidate --glob 'runs/teacher_fable_new*' \
      --tasks data/hard_tasks.jsonl \
      --exclude data/swe_verified_eval_exclusions_v1.json \
                data/python_eval_repo_denylist_v1.json \
      --out-dir runs/teacher_fable_new_revalidated
```

The `.work` ledger is resumable. Only the atomically published output is an
eligible merge input.

## Step 3 — Merge (consolidate strictly verified positives)

```
$PLAT merge --glob 'runs/teacher_*' --out-dir runs/teacher_merged
```

Produces `runs/teacher_merged/`:
- `results.jsonl` — every real attempt (all backends, deduped by instance)
- `resolved.jsonl` — strict-control positives with exact evidence
- `<id>.stream.jsonl` + `<id>.patch` — artifacts for each positive
- `manifest.json` — counts, per-backend resolve rates

---

## Step 4 — Prepare for training (render to SFT format)

```
$PLAT prepare --merged runs/teacher_merged --tasks data/hard_tasks.jsonl \
      --exclude data/swe_verified_eval_exclusions_v1.json \
                data/python_eval_repo_denylist_v1.json \
      --out data/teacher_sft
```

This is the **critical** step. It parses each teacher stream into
`(thought, command, observation)` steps and renders them as mini-SWE SFT:
`system` + PR problem, then per step `assistant`(THOUGHT + one `bash` tool_call)
and `user`(OBSERVATION), followed by the real terminal assistant response.
Assistant targets carry `loss=true`; conditioning carries `loss=false`. It
**strips the `docker exec <cid> bash -c` wrapper** so
the student learns bare `/testbed` commands, not the teacher's harness shape —
this is what prevents the data-mirror regression. Preparation rejects an
unparseable or unbound row instead of silently degrading the dataset.

---

## Step 3b — Blend external traces (Open-SWE-Traces) into the mix

Besides your own collected traces, fold in ready-made **verified** real-issue
trajectories and train on them together.

```
# ingest resolved Python trajectories from nvidia/Open-SWE-Traces -> SFT
# (decontaminate: exclude eval ids + anything you already used)
$PLAT ingest --out data/open_swe_sft --resolved-only --language python \
      --exclude 'runs/teacher_*/results.jsonl' data/lite_eval_ids.jsonl

# blend YOUR teacher traces + Open-SWE into ONE training set.
# Sources are deduped by instance_id; the FIRST source wins, so put your own
# (higher-value) teacher traces first. --cap N limits rows per source.
$PLAT blend --sources data/teacher_sft data/open_swe_sft \
      --out data/main_train_mix
```

`ingest` renders each external trajectory shape-safe (same mini-SWE format,
`docker exec` / `cd /testbed` prefixes stripped) and skips unresolved rows.
`blend` writes a single HF dataset dir + `.manifest.json` with per-source counts
and the mix. Decontamination is your responsibility: always pass eval instance
ids to `ingest --exclude`.

## Step 5 — Full-dataset format gate and smoke-test

```
# structural + Gemma format-loss gates (no GPU):
$PLAT smoke-train --data data/teacher_sft

# add a 2-step LoRA micro-run to prove it actually trains (needs a training GPU):
$PLAT smoke-train --data data/teacher_sft --micro-train
```

PASS criteria: every row has a system message + at least one assistant
`bash` tool_call; every row is checked by `verify_gemma_format_loss.py`; its
zero-failure report is immutably bound into `manifest.json`; (optional) the
2-step micro-train completes. `train_rust_lora.py` refuses schema-v2 teacher
datasets while this gate is pending or their JSONL hash has drifted.

---

## End-to-end (once hard tasks exist)

```
$PLAT collect --tasks data/hard_tasks.jsonl --out-dir runs/teacher_claude --backend claude --loop
$PLAT merge   --glob 'runs/teacher_*' --out-dir runs/teacher_merged
$PLAT prepare --merged runs/teacher_merged --tasks data/hard_tasks.jsonl \
      --exclude data/swe_verified_eval_exclusions_v1.json \
                data/python_eval_repo_denylist_v1.json --out data/teacher_sft
$PLAT smoke-train --data data/teacher_sft
$PLAT ingest  --out data/open_swe_sft --resolved-only --language python --exclude data/lite_eval_ids.jsonl
$PLAT blend   --sources data/teacher_sft data/open_swe_sft --out data/main_train_mix
$PLAT smoke-train --data data/main_train_mix
```

## Troubleshooting
- **`collect` writes only 2s duds** → quota wall; the loop pauses automatically.
  Check the wall wording is caught: `credit_exhausted_in_stream` keys on HTTP
  429. If a new wording slips through, add it there.
- **`prepare` rejects a row** → treat it as a provenance or normalization
  defect; check the artifact/task hash and balanced tool-result/terminal ledger.
- **`smoke-train` format gate fails** → inspect the offending row's `messages`;
  usually a tool_call arguments blob isn't valid JSON.
