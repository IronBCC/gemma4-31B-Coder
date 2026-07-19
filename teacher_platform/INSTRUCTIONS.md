# Teacher-Trace Platform — Agent Runbook

Goal: collect teacher (Claude/Codex/OpenRouter) solutions to problems the **base
model cannot solve**, F2P-verify them, and turn them into a training-ready SFT
dataset — then smoke-test that it trains. One CLI does every step.

**All commands run on the box, from the repo root**
(`~/projects/gemma4-31B-Coder`), with `.venv-eval/bin/python`.

```
PY=.venv-eval/bin/python
PLAT="$PY teacher_platform/teacher_platform.py"
```

Hard rules (do not skip — each earned a failed adapter):
- **Only train on base-FAILED tasks.** Traces on tasks the base already solves
  teach nothing (4 prior adapters regressed this way). Use `hard` to filter.
- **Never let eval leak in.** The pool is decontaminated at the repo level
  (Lite-30 / hard30 excluded). Don't add repos without re-checking.
- **Duds ≠ failures.** A quota-wall row has `n_assistant_events<=2` and
  `patch_len=0`; it means the task never ran. The tools already exclude these —
  never count them as teacher misses.
- **A row is verified only if `resolved: true`** (its patch passed the instance's
  FAIL_TO_PASS tests in-container). Everything downstream uses `resolved.jsonl`.

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

## Step 2 — Merge (consolidate verified positives)

```
$PLAT merge --glob 'runs/teacher_*' --out-dir runs/teacher_merged
```

Produces `runs/teacher_merged/`:
- `results.jsonl` — every real attempt (all backends, deduped by instance)
- `resolved.jsonl` — F2P-verified positives = **the trainable set**
- `<id>.stream.jsonl` + `<id>.patch` — artifacts for each positive
- `manifest.json` — counts, per-backend resolve rates

---

## Step 3 — Prepare for training (render to SFT format)

```
$PLAT prepare --merged runs/teacher_merged --tasks data/hard_tasks.jsonl --out data/teacher_sft.jsonl
```

This is the **critical** step. It parses each teacher stream into
`(thought, command, observation)` steps and renders them as mini-SWE SFT:
`system` + PR problem, then per step `assistant`(THOUGHT + one `bash` tool_call)
and `user`(OBSERVATION). It **strips the `docker exec <cid> bash -c` wrapper** so
the student learns bare `/testbed` commands, not the teacher's harness shape —
this is what prevents the data-mirror regression. Check
`data/teacher_sft.jsonl.manifest.json`: `median_first_edit_cmd` should be small
(edit-first); `dropped_unparseable` should be near 0.

---

## Step 4 — Smoke-test for training

```
# structural + Gemma format-loss gates (no GPU):
$PLAT smoke-train --data data/teacher_sft.jsonl

# add a 2-step LoRA micro-run to prove it actually trains (needs a training GPU):
$PLAT smoke-train --data data/teacher_sft.jsonl --micro-train
```

PASS criteria: every row has a system message + at least one assistant
`bash` tool_call; `verify_gemma_format_loss.py` reports 0 failures; (optional)
the 2-step micro-train completes without error. Only then hand the dataset to a
real training run.

---

## End-to-end (once hard tasks exist)

```
$PLAT collect --tasks data/hard_tasks.jsonl --out-dir runs/teacher_claude --backend claude --loop
$PLAT merge   --glob 'runs/teacher_*' --out-dir runs/teacher_merged
$PLAT prepare --merged runs/teacher_merged --tasks data/hard_tasks.jsonl --out data/teacher_sft.jsonl
$PLAT smoke-train --data data/teacher_sft.jsonl
```

## Troubleshooting
- **`collect` writes only 2s duds** → quota wall; the loop pauses automatically.
  Check the wall wording is caught: `credit_exhausted_in_stream` keys on HTTP
  429. If a new wording slips through, add it there.
- **`prepare` drops many rows** → the stream parser didn't recognize the backend
  format; check `normalize_steps` for that backend against a sample
  `.stream.jsonl`.
- **`smoke-train` format gate fails** → inspect the offending row's `messages`;
  usually a tool_call arguments blob isn't valid JSON.
