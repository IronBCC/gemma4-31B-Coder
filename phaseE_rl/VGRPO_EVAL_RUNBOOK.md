# vGRPO checkpoint evaluation runbook

This runbook evaluates the verified-reward continuation `swe_vgrpo_v1` at
checkpoints 100, 200, and 300.  It is deliberately a post-training procedure:
do not merge, serve, run Docker evaluation, or touch GPU1 while the 300-step
training process is still live.

## Policy being evaluated

The intended policy is an ordered LoRA stack over the frozen Gemma-4 base:

1. `swe_edit_v8_49k_s1/checkpoint-30` (SFT)
2. `swe_grpo_v1_s2/checkpoint-100` (promoted standard GRPO)
3. `swe_vgrpo_v1/checkpoint-{100,200,300}` (verified-reward continuation)

Each merge materializes a new, disposable evaluation model; it must never
modify the base model or any adapter checkpoint.  Use distinct output
directories:

| vGRPO checkpoint | Adapter | Merged evaluation model | Served name |
| --- | --- | --- | --- |
| 100 | `adapters/swe_vgrpo_v1/checkpoint-100` | `/media/ironbcc/CrucialX10/models/merged/swe_vgrpo_v1_cp100` | `swe_vgrpo_v1_cp100` |
| 200 | `adapters/swe_vgrpo_v1/checkpoint-200` | `/media/ironbcc/CrucialX10/models/merged/swe_vgrpo_v1_cp200` | `swe_vgrpo_v1_cp200` |
| 300 | `adapters/swe_vgrpo_v1/checkpoint-300` | `/media/ironbcc/CrucialX10/models/merged/swe_vgrpo_v1_cp300` | `swe_vgrpo_v1_cp300` |

## Merge preflight

`phaseE_rl/merge_swe_rlvr_for_eval.py` accepts repeatable `--adapter` flags
and merges them strictly in CLI order.  The required v8 + GRPO + vGRPO triple
stack is therefore materialized with three `--adapter` flags.  The legacy
`--sft-adapter` / `--rl-adapter` pair remains a deprecated two-adapter
compatibility path; do not use it for this lane.

Do not use the already saved `/media/ironbcc/CrucialX10/models/merged/swe_grpo_v1`
as `--base`: that artifact is a text-only `Gemma4ForCausalLM`, while the merger
expects the original multimodal Gemma-4 base and grafts its language model.
Using it as a shortcut is unverified and can silently produce an invalid
composition.

The merger checks every adapter boundary independently: a merge count of 100
or fewer aborts before model save.  It also retains the frozen base path and
refuses to overwrite an output directory.  Before any real checkpoint merge,
run the focused merger tests and confirm the target checkpoint exists.  The
required invocation is:

```bash
# Run from /home/ironbcc/projects/gemma4-31B-Coder after checkpoint-100 exists.
CUDA_VISIBLE_DEVICES=1 PYTHONPATH="$PWD" \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  .venv-rl/bin/python phaseE_rl/merge_swe_rlvr_for_eval.py \
  --adapter adapters/swe_edit_v8_49k_s1/checkpoint-30 \
  --adapter adapters/swe_grpo_v1_s2/checkpoint-100 \
  --adapter adapters/swe_vgrpo_v1/checkpoint-100 \
  --out /media/ironbcc/CrucialX10/models/merged/swe_vgrpo_v1_cp100 \
  --max-shard-size 20GB \
  > /tmp/merge_swe_vgrpo_v1_cp100.log 2>&1
```

For cp200 and cp300, change only the third `--adapter`, `--out`, and log
suffix.  Capture the three merge counts and `done ->` output line in the eval
record.  Preflight disk before each merge, reserve approximately 62 GiB for
the new materialization, and do not delete any prior artifact without explicit
approval.

## GPU and serving isolation

Evaluation begins only after the trainer and every child have exited and GPU1
is below 1 GiB used.  GPU0 and production ports `8000`, `8101`, `8103`, and
`8104` remain off-limits.  If an 8012 process exists, identify its exact PID
and stop only that PID; do not use `pkill`.

The standard-GRPO merged model was successfully served with the following
candidate configuration: text-only merged model, port 8012, 49,152 maximum
length, 0.93 GPU utilization, FP8 KV cache, Gemma-4 tool/reasoning parsers,
and the v2 thinking-open template.  Use the same isolated pattern for every
candidate, changing only `MODEL`, `SERVED`, and `LOG`:

```bash
cd /home/ironbcc/projects/llm/vllm
export CUDA_VISIBLE_DEVICES=1
export PATH=/home/ironbcc/projects/llm/vllm/vllm_env/bin:$PATH

MODEL=/media/ironbcc/CrucialX10/models/merged/swe_vgrpo_v1_cp100
SERVED=swe_vgrpo_v1_cp100
TPL=/home/ironbcc/projects/llm/vllm/tool_chat_template_gemma4_thinkopen_v2.jinja
LOG=/tmp/serve_${SERVED}.log

nohup vllm_env/bin/vllm serve "$MODEL" \
  --served-model-name "$SERVED" --port 8012 \
  --tokenizer-mode auto --gpu-memory-utilization 0.93 --max-model-len 49152 \
  --kv-cache-dtype fp8 --enable-auto-tool-choice --tool-call-parser gemma4 \
  --max-num-seqs 64 --reasoning-parser gemma4 \
  --chat-template "$TPL" \
  --default-chat-template-kwargs '{"enable_thinking": true}' \
  > "$LOG" 2>&1 &
SERVER_PID=$!
echo "server_pid=$SERVER_PID"
```

Wait for `curl -fsS http://127.0.0.1:8012/health` and verify that
`curl -fsS http://127.0.0.1:8012/v1/models` lists `$SERVED`.  Retain the
exact launch command and PID in the checkpoint's artifact directory.

The anchor cannot be applied as a LoRA to a candidate-merged model: that would
measure `candidate + cp20`, not cp20.  Instead, record a fresh same-day anchor
on a separately served raw NVFP4 base plus the `cp20_same_day` LoRA, then stop
that exact server PID and wait for GPU1 below 1 GiB before serving the merged
candidate.  Keep template, parser, context length, and decoding settings
identical between the anchor and candidate.

## Gate sequence for each checkpoint

Run the checkpoints serially in the order cp100, cp200, cp300.  A successful
smoke does not authorize a later checkpoint automatically; each checkpoint has
its own fresh anchor and hard30 floor.

### 1. Fresh same-day cp20 anchor

Serve the raw NVFP4 base on 8012 with only:

```text
--enable-lora
--lora-modules cp20_same_day=/home/ironbcc/projects/gemma4-31B-Coder/adapters/unsloth_agentic_filtered_repair_cp10_to_s10_14336/checkpoint-10
--max-lora-rank 32 --lora-dtype auto
```

and all shared serving arguments above.  Run hard30 through a disposable
proxy, saving the semantic score:

```bash
cd /home/ironbcc/projects/gemma4-31B-Coder
TS=$(date +%Y%m%d_%H%M%S)
MODEL=cp20_same_day
OUT="runs/hard_subset_${MODEL}_vgrpo_${TS}"
PORT=7871
mkdir -p "$OUT"
.venv-train/bin/python phaseD_sft/openai_chat_proxy.py \
  --api http://127.0.0.1:8012/v1 --model "$MODEL" --port "$PORT" \
  > "$OUT/proxy.log" 2>&1 &
PROXY_PID=$!
sleep 3
.venv-train/bin/python phaseD_sft/agent_smoke_eval.py \
  --url "http://127.0.0.1:${PORT}/chat" --jsonl "$OUT/results.jsonl" \
  > "$OUT/eval.log" 2>&1
kill "$PROXY_PID"
grep -E 'semantic score|full score' "$OUT/eval.log"
```

Stop the exact raw-base server PID, wait for GPU1 below 1 GiB, then serve the
merged candidate as described above.

### 2. Candidate hard30 regression floor

Run the identical proxy/evaluator sequence with
`MODEL=swe_vgrpo_v1_cp{100,200,300}` and an output path containing that
checkpoint.  Parse the candidate semantic score and apply the only hard30
promotion rule:

```text
candidate semantic >= fresh same-day cp20 semantic - 2
```

Below the floor is a hard failure: preserve the artifacts, stop the candidate
server by its exact PID, and do not start a SWE-Lite smoke for that checkpoint.
Hard30 is a regression floor, not a +3 promotion gate.

### 3. Decisive 30-case SWE-Lite smoke

If and only if the candidate clears the hard30 floor, leave that candidate
server live and run the fixed temperature-0.7 smoke:

```bash
cd /home/ironbcc/projects/gemma4-31B-Coder
NAME=swe_vgrpo_v1_cp100 PORT=8012 SLICE=0:30 WORKERS=8 TEMPERATURE=0.7 SCORE=1 \
  OUT=runs/smoke_swe_vgrpo_v1_cp100_t07_lite_0_30 \
  bash phaseH_eval/smoke_single.sh

PYTHONPATH="$PWD" .venv-eval/bin/python phaseH_eval/summarize_smoke.py \
  runs/smoke_swe_vgrpo_v1_cp100_t07_lite_0_30/swe_vgrpo_v1_cp100 \
  | tee runs/smoke_swe_vgrpo_v1_cp100_t07_lite_0_30/summary.json
```

Change `cp100` consistently for cp200 and cp300.  Save the raw `preds.json`,
generation log, score log/report, hard30 logs, and `summary.json` together.

The behavior gate is at least 18/30 non-empty patches and format errors below
10%.  Also report resolved count, edit-reaching trajectories, first-edit
statistics, and patch precision.  Compare each summary with the established
SFT references: v7 plateau 13/30 patches, 7/30 resolved, 14/30 edit reach;
v8cp30 11/30 patches, 6/30 resolved.  For verified-reward promotion, the
phase-2 target also requires resolved >7/30 and precision >40%.

## Checkpoint selection and handoff table

Do not pick a winner from training loss.  Once all eligible checkpoints have
completed their artifact-backed gates, make the decision from this table:

| Checkpoint | Merge counts | Anchor / candidate hard30 | Floor | Patches | Format rate | Resolved | Edit reach | Precision | Decision |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| cp100 |  |  |  |  |  |  |  |  |  |
| cp200 |  |  |  |  |  |  |  |  |  |
| cp300 |  |  |  |  |  |  |  |  |  |

Report the completed table and exact artifact paths before any further
training, data changes, or model deletion.
