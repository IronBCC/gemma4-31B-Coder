# v6 Pipeline Runbook — Executable Per-Phase Guide

**Source of truth**: `~/.claude/plans/nifty-purring-pie.md` (approved v6 plan).
**Status record**: `phaseH_eval/SESSION_STATE.md`.
**EXECUTION STATUS 2026-07-11**: The clean compressed/sanitized 49,152 and 40,960 datasets both passed their required 1,000-sample loss-format gates; Gate A at 24,576 passed. The revised-policy 49,152 Gate B then failed because the installed PyTorch 2.10 FlexAttention path fell back to dense math attention and requested 128.13 GiB before microstep 1. The next authorized action is the one-step 40,960 Gate B on its long16 shard, with the same timestamped trainer/launcher progress telemetry.
**Host**: `192.168.50.148` (`ironbccllm.tail0cc1d4.ts.net`), repo root `/home/ironbcc/projects/gemma4-31B-Coder`.

---

## Non-Negotiable Host Constraints (Apply to Every Phase)

| Constraint | Detail |
|---|---|
| GPU0 production vLLM | **OFF-LIMITS**. Never stop or kill `vllm.service` or its GPU0 workers. |
| Production ports | **8000** (router), **8101** (Gemma4), **8103** (nano), **8104** (speech). All must remain green throughout every phase. |
| Training GPU | **GPU1 only**. Always set `CUDA_VISIBLE_DEVICES=1`. |
| RAM / Swap | 61 GiB physical + 255 GiB swap. Bounded launcher mandatory (`recover_swe_edit_v4_smoke_first.sh`). |
| Launcher env var | `ALLOW_64G_TRAINING=I_ACCEPT_BOUNDED_SWAP` (required for any training launch). |
| Cgroup caps | **MemoryHigh=36G**, **MemoryMax=45G**, **MemorySwapMax=16G**. Abort only if host `MemAvailable <2 GiB`; global swap use is recorded but is **not** a stop condition. |
| Kill discipline | By exact numeric PID only (from `ps -eo pid,args \| grep ... \| grep -v grep`). **NEVER use `pkill -f`** — over SSH the pattern matches your own command string and kills the session. |
| Cgroup caps policy | **Never raise cgroup caps**. If OOM or threshold trip → fallback tier (rebuild dataset at lower cap). |

### Before Every Remote Command Block

```bash
# Verify production health before touching GPU1:
for p in 8000 8101 8103 8104; do curl -fsS --max-time 2 http://127.0.0.1:$p/health >/dev/null || echo "port $p FAIL"; done

# Verify GPU1 is free (if no training running):
ssh -o HostKeyAlias=ironbccllm.tail0cc1d4.ts.net ironbcc@192.168.50.148 \
  'nvidia-smi --query-gpu=memory.used,memory.total --id=1 --format=csv,noheader'
```

---

## Phase 0 — Init-Adapter Decision (Parallel with Data Work)

**Purpose**: Decide which adapter initializes v6 training. Evaluate the completed v5-s2 checkpoint-5 on hard30 first; if it passes, use it; otherwise fall back to cp20. This runs in parallel with data processing (Phases 1–3).

### Step 0a — Serve v5-s2 checkpoint-5 on GPU1 at 256k

```bash
# SSH to host:
ssh -o HostKeyAlias=ironbccllm.tail0cc1d4.ts.net ironbcc@192.168.50.148 \
  'cd /home/ironbcc/projects/gemma4-31B-Coder && \
   systemctl --user stop v5-s2-eval.service 2>/dev/null || true'

# Launch isolated eval unit (pattern from v4-s1-eval.service precedent):
ssh -o HostKeyAlias=ironbccllm.tail0cc1d4.ts.net ironbcc@192.168.50.148 \
  'cd /home/ironbcc/projects/gemma4-31B-Coder && \
   mkdir -p logs adapters/unsloth_agentic_swe_edit_trace_v5_cp20_24k_s2_eval && \
   systemctl --user stop v5-s2-eval.service 2>/dev/null || true; \
   if [[ -f "logs/v5_s2_eval.log" ]]; then mv logs/v5_s2_eval.log "logs/v5_s2_eval.log.previous-$(date -u +%Y%m%dT%H%M%SZ)"; fi; \
   systemd-run --user --unit=v5-s2-eval --remain-after-exit \
     --property=MemoryAccounting=yes \
     --property="MemoryHigh=16G" \
     --property="MemoryMax=24G" \
     --property="MemorySwapMax=8G" \
     --property=OOMScoreAdjust=500 \
     --property="StandardOutput=append:logs/v5_s2_eval.log" \
     --property="StandardError=append:logs/v5_s2_eval.log" \
     --working-directory=/home/ironbcc/projects/gemma4-31B-Coder \
     --setenv=CUDA_VISIBLE_DEVICES=1 \
     --setenv=PATH="/home/ironbcc/projects/gemma4-31B-Coder/.venv-train/bin:$PATH" \
     python -m vllm.entrypoints.openai.api_server \
       --model /home/ironbcc/projects/gemma4-31B-Coder/adapters/unsloth_agentic_swe_edit_trace_v5_cp20_24k_s2/checkpoint-5 \
       --served-model-name gemma4-agentic-v5-s2-cp5 \
       --trust-remote-code \
       --max-model-len 262144 \
       --gpu-memory-utilization 0.95 \
       --kv-cache-dtype fp8 \
       --enable-lora --max-lora-rank 32'

# Verify server health:
ssh -o HostKeyAlias=ironbccllm.tail0cc1d4.ts.net ironbcc@192.168.50.148 \
  'curl -fsS http://127.0.0.1:8012/v1/models'

# Verify production still green:
for p in 8000 8101 8103 8104; do curl -fsS --max-time 2 http://127.0.0.1:$p/health >/dev/null || echo "port $p FAIL"; done
```

### Step 0b — Run hard30 eval against v5-s2 cp5

```bash
# On host (using .venv-eval for eval harness):
ssh -o HostKeyAlias=ironbccllm.tail0cc1d4.ts.net ironbcc@192.168.50.148 \
  'cd /home/ironbcc/projects/gemma4-31B-Coder && \
   mkdir -p runs/hard_subset_v5_s2_cp5_256k && \
   .venv-eval/bin/python phaseD_sft/agent_smoke_eval.py \
     --url http://localhost:8012/v1/chat/completions \
     --model gemma4-agentic-v5-s2-cp5 \
     --max-tokens 512 \
     --temperature 0 \
     --jsonl runs/hard_subset_v5_s2_cp5_256k/results.jsonl'

# Check result:
ssh -o HostKeyAlias=ironbccllm.tail0cc1d4.ts.net ironbcc@192.168.50.148 \
  'cat runs/hard_subset_v5_s2_cp5_256k/results.jsonl | python3 -c "import sys,json; rows=[json.loads(l) for l in sys.stdin]; print(f\"semantic: {sum(r.get(\"semantic\",0) for r in rows)}/30\"); print(f\"full: {sum(r.get(\"full\",0) for r in rows)}/30\")"'
```

### Step 0c — Decision Gate

| Result | Action |
|---|---|
| **≥25/30 semantic** | Use `adapters/unsloth_agentic_swe_edit_trace_v5_cp20_24k_s2/checkpoint-5` as v6 init adapter. Record verdict in SESSION_STATE. |
| **<25/30 semantic** | Fall back to cp20: `adapters/unsloth_agentic_filtered_repair_cp10_to_s10_14336/checkpoint-10`. Record verdict in SESSION_STATE. |

### Step 0d — Teardown eval server (if decision is cp20 or after recording v5-s2 result)

```bash
ssh -o HostKeyAlias=ironbccllm.tail0cc1d4.ts.net ironbcc@192.168.50.148 \
  'systemctl --user stop v5-s2-eval.service; systemctl --user disable v5-s2-eval.service'
```

### Artifacts to Record

- Run dir path for hard30 eval.
- Semantic score and full score.
- Verdict: which adapter is the init candidate (v5-s2-cp5 or cp20).
- Server PID, port used, launch flags.

---

## Phase 1 — Dedup + Trim Script + Compact Escape-Hatch Fix

**Purpose**: Build the one new script (`dedup_and_trim_traces.py`) and patch `compact_observations.py` so v6 data is clean before any training. This phase produces code that runs locally (no remote host needed).

### Step 1a — Create `phaseD_sft/dedup_and_trim_traces.py`

The script operates on JSONL or HF datasets:

- **Trim**: while last message role != assistant → drop it. Drop trace entirely if no assistant remains or fewer than 2 messages.
- **Dedup**: sha256 over `json.dumps(messages, sort_keys=True)` **after trimming**. Keep first occurrence. Never dedup by instance_id (window splits are legitimate variants).
- **Manifest output**: rows_in, rows_out, exact_dups_removed, trailing_trimmed, traces_dropped, unique_content_count, per-source counts.

Usage:
```bash
# JSONL input/output:
.venv-train/bin/python phaseD_sft/dedup_and_trim_traces.py \
  --input data/<source>.jsonl \
  --output data/<output>.jsonl \
  --manifest data/<output>_dedup_manifest.json

# HF dataset (if applicable):
.venv-train/bin/python phaseD_sft/dedup_and_trim_traces.py \
  --dataset <hf_dataset_name> \
  --split train \
  --output <local_output_path> \
  --manifest data/<output>_dedup_manifest.json
```

### Step 1b — Patch `phaseD_sft/compact_observations.py`

In `_trim_to_char_budget`: remove the escape hatch. The current code has:
```python
if len(head) <= 2:
    # escape: return candidates over max_chars
```
Replace with hard enforcement of `max_chars` via the head/tail fallback (no length-based escape). This ensures no observation exceeds 2,400 chars.

### Step 1c — Add regression test for compact_observations fix

Add a test in `phaseD_sft/tests/test_compact_observations.py`:
- Create an observation with many important lines that would trigger the escape hatch.
- Verify `_trim_to_char_budget` now enforces hard cap (no output over max_chars).

### Step 1d — Add unit tests for dedup_and_trim_traces.py

Create `phaseD_sft/tests/test_dedup_and_trim_traces.py`:
- Trailing-user trim: trace ending on user → trimmed to last assistant.
- Exact-dup collapse: duplicate content hashes → only one kept.
- Window-variant preservation: same instance_id, different content → both kept.
- Id-reuse-not-deduped: dedup is by content hash, never by instance_id.

### Step 1e — Run tests locally (no remote needed)

```bash
pytest phaseD_sft/tests/test_dedup_and_trim_traces.py phaseD_sft/tests/test_compact_observations.py -v
```

### Pass Gate

- All unit tests pass with exit code 0.
- Script handles edge cases: empty messages list, single-message trace, all-user trace (dropped).

### Artifacts to Record

- Test results output.
- Script location: `phaseD_sft/dedup_and_trim_traces.py`.
- Patch location: `phaseD_sft/compact_observations.py` (note which lines changed).

---

## Phase 2 — Base Datasource Verify + Nebius Audit

**Purpose**: Verify raw dataset integrity, audit nebius rows for reinclude eligibility.

### Step 2a — Schema scan of raw agentic dataset on host

```bash
ssh -o HostKeyAlias=ironbccllm.tail0cc1d4.ts.net ironbcc@192.168.50.148 \
  'cd /home/ironbcc/projects/gemma4-31B-Coder && \
   .venv-train/bin/python phaseD_sft/scan_agentic_dataset.py data/unsloth_agentic_24k_train'
```

This promotes the audit scratch pattern (`/tmp/scan_v5_dataset.py`) to a committed script. Output: per-source counts, schema validation results, row totals vs manifest expectations.

### Step 2b — Verify nebius rows (3,129 rows)

```bash
ssh -o HostKeyAlias=ironbccllm.tail0cc1d4.ts.net ironbcc@192.168.50.148 \
  'cd /home/ironbcc/projects/gemma4-31B-Coder && \
   .venv-train/bin/python phaseD_sft/scan_agentic_dataset.py data/nebius_subset --source nebius'

# Run format gate sample:
ssh -o HostKeyAlias=ironbccllm.tail0cc1d4.ts.net ironbcc@192.168.50.148 \
  'cd /home/ironbcc/projects/gemma4-31B-Coder && \
   .venv-train/bin/python verify_gemma_format_loss.py --data data/nebius_subset --samples 300'

# Run quality filters (same defaults as normalize_agentic_dataset.py):
ssh -o HostKeyAlias=ironbccllm.tail0cc1d4.ts.net ironbcc@192.168.50.148 \
  'cd /home/ironbcc/projects/gemma4-31B-Coder && \
   .venv-train/bin/python phaseD_sft/agentic_trace_filters.py trace_should_keep \
     --data data/nebius_subset \
     --max-first-edit-ratio 0.4 \
     --max-read-streak 6 \
     --require-verify-tail'
```

### Step 2c — Cross-reference with manifests

Compare per-source counts against `data/manifests/unsloth_agentic_24k_train_manifest.json` and raw JSONL chain manifests (`agentic_sft_bashtools*` in `data/manifests/`).

### Nebius Reinclude Decision Gate

| Survivors | Action |
|---|---|
| **≥200 rows** pass format + trim + dedup | Reinclude in v6 mixture build (Phase 3). Record per-source counts. |
| **<200 rows** survive | Skip reinclusion this round. Note in SESSION_STATE and manifest. |

### Artifacts to Record

- Schema scan output (source counts, schema violations if any).
- Nebius format gate results (failure_count from verify_gemma_format_loss.py).
- Quality filter survivor count for nebius.
- Decision: reinclude or skip.

---

## Phase 2b — External Trace-Dataset Scouting (HuggingFace)

**Purpose**: Vet external datasets (Open-SWE-Traces, Kwai-Klear) as potential v6 contributors. Download + audit only; convert and pipeline through same compaction → trim → dedup → format gate. Include in v6 only what passes, capped at ≤50% of total mixture to protect anchor sources.

### Step 2b-1 — Download Open-SWE-Traces (filtered)

```bash
# On host:
ssh -o HostKeyAlias=ironbccllm.tail0cc1d4.ts.net ironbcc@192.168.50.148 \
  'cd /home/ironbcc/projects/gemma4-31B-Coder && \
   .venv-train/bin/python -c "
from datasets import load_dataset
ds = load_dataset(\"nvidia/Open-SWE-Traces\", split=\"train\")
# Filter: resolved==1 AND language in {Python, Rust, C/C++} AND rendered_tokens <= 32768
resolved = ds.filter(lambda x: x[\"resolved\"] == True)
target_langs = resolved.filter(lambda x: x[\"language\"] in [\"Python\", \"Rust\", \"C/C++\"])
print(f\"Open-SWE-Traces resolved+target-lang rows: {len(target_langs)}\")'
```

### Step 2b-2 — Download Kwai-Klear (filtered)

```bash
ssh -o HostKeyAlias=ironbccllm.tail0cc1d4.ts.net ironbcc@192.168.50.148 \
  'cd /home/ironbcc/projects/gemma4-31B-Coder && \
   .venv-train/bin/python -c "
from datasets import load_dataset
ds = load_dataset(\"Kwai-Klear/SWE-smith-mini_swe_agent_plus-trajectories-66k\", split=\"train\")
print(f\"Kwai-Klear rows: {len(ds)}\")'
```

### Step 2b-3 — Convert to v6 schema + pipeline

Convert each dataset's trajectories into the v6 format convention:
- Assistant `tool_calls` → bash JSON args (matching existing CoT convention).
- User `OBSERVATION:` turns.
- THOUGHT text stays as plain assistant content.
- Pipeline: compaction (patched) → trim → dedup (cross-checked against existing corpus hashes via content-hash set) → format gate (`verify_gemma_format_loss.py`).

### Step 2b-4 — Decision Gate

| Outcome | Action |
|---|---|
| Conversion clean, gates pass, ≤50% of mixture share | Include in v6 mixture (Phase 3). Record per-source counts. |
| Conversion lossy or gates fail | Defer external rows to v7; ship v6 fix-only. Note in SESSION_STATE. |

### Artifacts to Record

- Per-dataset row counts before/after filtering.
- Schema conversion success/failure notes.
- Format gate results (verify_gemma_format_loss.py).
- Final inclusion decision and share percentage cap applied.

---

## Phase 3 — v6 Mixture Build @ 32,768 Tokens

**Purpose**: Concatenate all clean sources into the final v6 dataset at the 32k budget cap with full audit trail in manifests.

### Step 3a — Gather inputs from pre-budget parents

Source datasets (from pre-18432 budgets so length-rejected rows return):
1. `data/unsloth_agentic_24k_train_normalized_format_plus_coder_repair` (normalized swe-smith + coder_repair 768).
2. `data/unsloth_agentic_24k_train_swe_edit_trace_v4` (oracle edit traces, regenerated via `build_swe_edit_trace_dataset.py` if needed).
3. Surviving nebius rows from Phase 2 (if reincluded; otherwise skip this source).

For long traces: where a full trace ≤32,768 exists in the pre-split JSONL chain (`agentic_sft_bashtools_compacted_obs*_prunedloops.jsonl`), prefer it over its 16k window splits. Drop corresponding window rows via instance_id + content bookkeeping to avoid double-counting.

### Step 3b — Re-compact observations with patched `compact_observations.py`

```bash
ssh -o HostKeyAlias=ironbccllm.tail0cc1d4.ts.net ironbcc@192.168.50.148 \
  'cd /home/ironbcc/projects/gemma4-31B-Coder && \
   .venv-train/bin/python phaseD_sft/compact_observations.py \
     --input data/<concatenated_input>.jsonl \
     --output data/v6_compacted.jsonl'

# Idempotent on already-compacted rows; enforces the fixed hard cap.
```

### Step 3c — Trim + Dedup with new script from Phase 1

```bash
ssh -o HostKeyAlias=ironbccllm.tail0cc4.ts.net ironbcc@192.168.50.148 \
  'cd /home/ironbcc/projects/gemma4-31B-Coder && \
   .venv-train/bin/python phaseD_sft/dedup_and_trim_traces.py \
     --input data/v6_compacted.jsonl \
     --output data/v6_trimmed_deduped.jsonl \
     --manifest data/v6_pre_budget_manifest.json'
```

### Step 3d — Filter by token budget (≤32,768) with Gemma tokenizer

```bash
ssh -o HostKeyAlias=ironbccllm.tail0cc1d4.ts.net ironbcc@192.168.50.148 \
  'cd /home/ironbcc/projects/gemma4-31B-Coder && \
   .venv-train/bin/python phaseD_sft/filter_token_budget.py \
     --data data/v6_trimmed_deduped.jsonl \
     --output data/unsloth_agentic_32k_train_swe_edit_trace_v6_budget32768 \
     --max-tokens 32768'

# Uses native Gemma template token counts via token_audit.py loader.
# Expect near-zero rejection on clean v6 data.
```

### Step 3e — Extend manifest with full audit trail

The budget manifest at `data/unsloth_agentic_32k_train_swe_edit_trace_v6_budget32768/budget_manifest.json` must include:
- `unique_content_count`: rows in final dataset (must equal `rows_accepted`).
- `dups_removed`: count of exact-duplicate rows removed.
- `trailing_trimmed`: count of traces trimmed or dropped.
- Per-source accepted/rejected counts with source labels.

### Step 3f — Build worst-case long16 shard

```bash
ssh -o HostKeyAlias=ironbccllm.tail0cc1d4.ts.net ironbcc@192.168.50.148 \
  'cd /home/ironbcc/projects/gemma4-31B-Coder && \
   .venv-train/bin/python phaseD_sft/filter_token_budget.py \
     --data data/unsloth_agentic_32k_train_swe_edit_trace_v6_budget32768 \
     --output data/unsloth_agentic_32k_train_swe_edit_trace_v6_budget32768_long16 \
     --longest 16'

# gate_manifest.json included in the long16 shard directory.
```

### Pass Gate

- `unique_content_count == rows_accepted` (no phantom rows).
- Per-source counts explained vs parents (audit trail complete).
- Long16 shard contains exactly 16 rows, all ≤32,768 tokens.

### Artifacts to Record

- Final dataset path: `data/unsloth_agentic_32k_train_swe_edit_trace_v6_budget32768`.
- Budget manifest content (key counts).
- Long16 shard path and gate_manifest.json.

---

## Phase 4 — Format Gate (Hard, Blocking)

**Purpose**: Verify the entire v6 dataset is correctly formatted for Gemma training before any GPU time is spent. This phase blocks all downstream work if it fails.

### Step 4a — Run verify_gemma_format_loss.py on sample

```bash
ssh -o HostKeyAlias=ironbccllm.tail0cc1d4.ts.net ironbcc@192.168.50.148 \
  'cd /home/ironbcc/projects/gemma4-31B-Coder && \
   .venv-train/bin/python verify_gemma_format_loss.py --data data/unsloth_agentic_32k_train_swe_edit_trace_v6_budget32768 --samples 1000'

# Required: exit code 0 AND failure_count = 0.
```

### Step 4b — Run verify_gemma_format_loss.py on full dataset (if runtime allows)

If the host has CPU-free GPU1 time or fast enough CPU inference for this:

```bash
ssh -o HostKeyAlias=ironbccllm.tail0cc1d4.ts.net ironbcc@192.168.50.148 \
  'cd /home/ironbcc/projects/gemma4-31B-Coder && \
   .venv-train/bin/python verify_gemma_format_loss.py --data data/unsloth_agentic_32k_train_swe_edit_trace_v6_budget32768 --samples all'

# If full-sample run is infeasible, at minimum confirm exit 0 on the 1000-sample pass.
```

### Step 4c — Structural scan (re-run from Phase 1 / audit)

Verify these conditions hold across the full dataset:
- **0 exact dups** (content-hash uniqueness).
- **0 traces ending on user OBSERVATION**.
- **All tool_calls are valid JSON `bash` calls**.
- **All observations ≤2,400 chars** (post-fix enforcement).
- Every assistant message inside a supervised `<|turn>model` span.
- Balanced `<|tool_call>`/`<tool_call|>`/`<|tool_response>`.
- No zero-supervision rows.
- Stable prefix rendering.

### Pass Gate — ALL of the following must hold:

1. `verify_gemma_format_loss.py --samples 1000`: exit code 0, failure_count = 0.
2. Structural scan: 0 exact dups, 0 user-tail traces, all obs ≤2400 chars.
3. All tool_calls are valid JSON bash calls with no malformed schemas.

### Fail Response

If gate fails: do not proceed to Phase 5. Diagnose and fix the specific failure mode (re-run compaction/trim/dedup/budget filter as needed). Record diagnosis in SESSION_STATE.

### Artifacts to Record

- verify_gemma_format_loss.py output (both runs if applicable): exit code, failure_count, sample count.
- Structural scan results: counts of dups found, user-tail traces, observations over cap.

---

## Phase 5 — Staged Memory Gates → Training Run

**Purpose**: Prove v6 dataset fits in GPU1 VRAM at the target sequence length(s) under bounded conditions, then run full training with init from Phase 0 verdict.

### Step 5a — Gate A: 24,576 worst-case memory gate

Filter long16 shard to rows ≤24,576 tokens (should be all of them if budget was enforced). Launch single-step smoke on GPU1:

```bash
ssh -o HostKeyAlias=ironbccllm.tail0cc1d4.ts.net ironbcc@192.168.50.148 \
  'cd /home/ironbcc/projects/gemma4-31B-Coder && \
   systemctl --user stop swe-edit-v6-gate-a.service 2>/dev/null || true; \
   if [[ -f "logs/v6_gate_a.log" ]]; then mv logs/v6_gate_a.log "logs/v6_gate_a.log.previous-$(date -u +%Y%m%dT%H%M%SZ)"; fi; \
   systemd-run --user --unit=swe-edit-v6-gate-a --remain-after-exit \
     --property=MemoryAccounting=yes \
     --property="MemoryHigh=36G" \
     --property="MemoryMax=45G" \
     --property="MemorySwapMax=16G" \
     --property=OOMScoreAdjust=500 \
     --property="StandardOutput=append:logs/v6_gate_a.log" \
     --property="StandardError=append:logs/v6_gate_a.log" \
     --working-directory=/home/ironbcc/projects/gemma4-31B-Coder \
     --setenv=CUDA_VISIBLE_DEVICES=1 \
     --setenv=PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
     --setenv=ALLOW_64G_TRAINING=I_ACCEPT_BOUNDED_SWAP \
     /home/ironbcc/projects/gemma4-31B-Coder/.venv-train/bin/python phaseD_sft/train_rust_lora.py \
       --data /home/ironbcc/projects/gemma4-31B-Coder/data/unsloth_agentic_32k_train_swe_edit_trace_v6_budget32768_long16 \
       --out /home/ironbcc/projects/gemma4-31B-Coder/adapters/v6_gate_a_24576 \
       --init-adapter <PHASE_0_INIT_ADAPTER> \
       --max-seq 24576 \
       --max-steps 1 \
       --warmup-steps 0 \
       --bsz 1 \
       --grad-accum 16 \
       --lr 0.0002 \
       --load-4bit \
       --logging-steps 1 \
       --save-steps 1'

# Monitor: check prod health every 30s, cgroup memory peak, GPU1 VRAM, host MemAvailable/swap.
```

### Gate A Pass Criteria

| Metric | Threshold |
|---|---|
| `MemoryPeak` (cgroup) | <45G |
| `MemorySwapPeak` (cgroup) | <16G |
| GPU1 VRAM peak | Recorded, <97.9 GB total |
| Host available RAM min | ≥2 GiB (abort below this) |
| Host swap use max | Recorded only; no global-swap stop |
| Production ports 8000/8101/8103/8104 | Green throughout |

### Step 5b — Gate B: target-cap worst-case memory gate

For the current v6 lane, the target is **49,152**. Use the bounded local-launcher command in `V6_49K_CODEX_HANDOFF.md` §5 with its 49k dataset and `_long16` shard; the 32,768 command below is retained as the historical reference pattern only.

```bash
ssh -o HostKeyAlias=ironbccllm.tail0cc1d4.ts.net ironbcc@192.168.50.148 \
  'cd /home/ironbcc/projects/gemma4-31B-Coder && \
   systemctl --user stop swe-edit-v6-gate-b.service 2>/dev/null || true; \
   if [[ -f "logs/v6_gate_b.log" ]]; then mv logs/v6_gate_b.log "logs/v6_gate_b.log.previous-$(date -u +%Y%m%dT%H%M%SZ)"; fi; \
   systemd-run --user --unit=swe-edit-v6-gate-b --remain-after-exit \
     --property=MemoryAccounting=yes \
     --property="MemoryHigh=36G" \
     --property="MemoryMax=45G" \
     --property="MemorySwapMax=16G" \
     --property=OOMScoreAdjust=500 \
     --property="StandardOutput=append:logs/v6_gate_b.log" \
     --property="StandardError=append:logs/v6_gate_b.log" \
     --working-directory=/home/ironbcc/projects/gemma4-31B-Coder \
     --setenv=CUDA_VISIBLE_DEVICES=1 \
     --setenv=PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
     --setenv=ALLOW_64G_TRAINING=I_ACCEPT_BOUNDED_SWAP \
     /home/ironbcc/projects/gemma4-31B-Coder/.venv-train/bin/python phaseD_sft/train_rust_lora.py \
       --data /home/ironbcc/projects/gemma4-31B-Coder/data/unsloth_agentic_32k_train_swe_edit_trace_v6_budget32768_long16 \
       --out /home/ironbcc/projects/gemma4-31B-Coder/adapters/v6_gate_b_32768 \
       --init-adapter <PHASE_0_INIT_ADAPTER> \
       --max-seq 32768 \
       --max-steps 1 \
       --warmup-steps 0 \
       --bsz 1 \
       --grad-accum 16 \
       --lr 0.0002 \
       --load-4bit \
       --logging-steps 1 \
       --save-steps 1'

# Same monitoring as Gate A. Preserve the launcher watchlog and record model-load duration plus the measured per-step wall-clock duration, peak GPU1 used MiB, `MemoryPeak`, and `MemorySwapPeak`. These are required inputs for the 49k five-step smoke ETA.
```

### Step 5b Fallback: OOM or threshold trip on Gate B

If GPU1 OOMs, cgroup/swap thresholds are tripped, production becomes unhealthy, or host RAM drops below abort floor during Gate B:

**DO NOT raise cgroup caps.** Instead:

1. Record the failure telemetry (MemoryPeak, SwapPeak, VRAM peak at failure point).
2. Rebuild budget dataset at 24,576: `filter_token_budget.py --max-tokens 24576` on the trimmed/deduped v6 corpus → `data/unsloth_agentic_v6_budget24576`.
3. Rerun Gate A shard against the rebuilt 24k dataset (Gate B is not required; Gate A already proved fit at this cap).
4. Proceed with training at max-seq = 24,576 instead of 32,768.

### Step 5c — Full v6 training run

```bash
# Use the recover_swe_edit_v4_smoke_first.sh launcher with bounded mode:
ALLOW_64G_TRAINING=I_ACCEPT_BOUNDED_SWAP \
DATA=data/unsloth_agentic_32k_train_swe_edit_trace_v6_budget32768 \
INIT_ADAPTER=<PHASE_0_INIT_ADAPTER> \
SMOKE_OUT=<new_adapter_output_dir> \
RUN_NAME=swe-edit-v6-training \
SMOKE_UNIT=swe-edit-v6-train \
MAX_SEQ=<HIGHEST_PASSED_GATE> \
TRAIN_STEPS=5 \
WARMUP_STEPS=1 \
TRAIN_LR=0.00002 \
bash phaseD_sft/recover_swe_edit_v4_smoke_first.sh

# If fallback to 24k tier occurred:
MAX_SEQ=24576 \
DATA=data/unsloth_agentic_v6_budget24576 \
...
```

Training hyperparameters (from v6 plan):
- `--max-seq`: highest passed gate (32,768 or 24,576).
- LR lane: `2e-5` peak with cosine schedule + 1 warmup step.
- `--save-steps 1`.
- Bounded unit name: `swe-edit-v6-32k-s1` (or rename per the actual gate outcome).

### Pass Gate — Training Run

| Metric | Threshold |
|---|---|
| Unit Result=success, ExecMainStatus=0 | Required |
| MemoryPeak <45G | Required |
| Production ports green throughout | Required |
| Checkpoint-1 through checkpoint-N exist | Required (N = TRAIN_STEPS) |

### Artifacts to Record

- Gate A telemetry: MemoryPeak, SwapPeak, GPU1 VRAM peak, host min MemAvailable, host max swap use.
- Gate B telemetry: model-load duration, per-step wall-clock duration, MemoryPeak, MemorySwapPeak, GPU1 VRAM peak, host minimum MemAvailable, host maximum swap use; or failure reason if fallback triggered.
- Highest passed gate value (used as --max-seq).
- Final training run dir path, unit name, checkpoint existence verification.

---

## Phase 6 — Eval Gate Sequence (Existing Policy)

**Purpose**: Validate v6 training produced a useful model before scaling to broader evaluation. Three sequential gates; failure at any gate stops the lane and requires rebuild/reweighting of data.

### Step 6a — hard30 on checkpoint-5 (256k serving, FP8 KV, isolated eval unit)

```bash
# Serve checkpoint from training output:
ssh -o HostKeyAlias=ironbccllm.tail0cc1d4.ts.net ironbcc@192.168.50.148 \
  'cd /home/ironbcc/projects/gemma4-31B-Coder && \
   systemctl --user stop v6-eval.service 2>/dev/null || true; \
   if [[ -f "logs/v6_eval.log" ]]; then mv logs/v6_eval.log "logs/v6_eval.log.previous-$(date -u +%Y%m%dT%H%M%SZ)"; fi; \
   systemd-run --user --unit=v6-eval --remain-after-exit \
     --property=MemoryAccounting=yes \
     --property="MemoryHigh=16G" \
     --property="MemoryMax=24G" \
     --property="MemorySwapMax=8G" \
     --property=OOMScoreAdjust=500 \
     --property="StandardOutput=append:logs/v6_eval.log" \
     --property="StandardError=append:logs/v6_eval.log" \
     --working-directory=/home/ironbcc/projects/gemma4-31B-Coder \
     --setenv=CUDA_VISIBLE_DEVICES=1 \
     --setenv=PATH="/home/ironbcc/projects/gemma4-31B-Coder/.venv-train/bin:$PATH" \
     python -m vllm.entrypoints.openai.api_server \
       --model <TRAINING_OUTPUT_DIR>/checkpoint-5 \
       --served-model-name gemma4-agentic-v6-cp5 \
       --trust-remote-code \
       --max-model-len 262144 \
       --gpu-memory-utilization 0.95 \
       --kv-cache-dtype fp8'

# Run hard30:
ssh -o HostKeyAlias=ironbccllm.tail0cc1d4.ts.net ironbcc@192.168.50.148 \
  'cd /home/ironbcc/projects/gemma4-31B-Coder && \
   .venv-eval/bin/python phaseD_sft/agent_smoke_eval.py \
     --url http://localhost:8012/v1/chat/completions \
     --model gemma4-agentic-v6-cp5 \
     --max-tokens 512 \
     --temperature 0 \
     --jsonl runs/hard_subset_v6_cp5_256k/results.jsonl'

# Check result:
ssh -o HostKeyAlias=ironbccllm.tail0cc1d4.ts.net ironbcc@192.168.50.148 \
  'cat runs/hard_subset_v6_cp5_256k/results.jsonl | python3 -c "import sys,json; rows=[json.loads(l) for l in sys.stdin]; print(f\"semantic: {sum(r.get(\"semantic\",0) for r in rows)}/30\"); print(f\"full: {sum(r.get(\"full\",0) for r in rows)}/30\")"'
```

### Gate 6a Pass Criteria

| Metric | Threshold |
|---|---|
| Semantic score on hard30 | **≥ same-day cp20 + 3 semantic** on the same serving path |
| Production ports green during serving | Required |

### Step 6b — Thirty-case SWE-Lite smoke test

```bash
ssh -o HostKeyAlias=ironbccllm.tail0cc1d4.ts.net ironbcc@192.168.50.148 \
  'cd /home/ironbcc/projects/gemma4-31B-Coder && \
   mkdir -p runs/smoke_v6_lite_0_30_256k && \
   .venv-eval/bin/python phaseD_sft/agent_smoke_eval.py \
     --url http://localhost:8012/v1/chat/completions \
     --model gemma4-agentic-v6-cp5 \
     --max-tokens 512 \
     --temperature 0 \
     --jsonl runs/smoke_v6_lite_0_30_256k/results.jsonl \
     --config <30-case SWE-Lite config>'

# After completion, check:
ssh -o HostKeyAlias=ironbccllm.tail0cc1d4.ts.net ironbcc@192.168.50.148 \
  'python3 runs/smoke_v6_lite_0_30_256k/aggregate.py'
```

### Gate 6b Pass Criteria (ALL required)

| Metric | Threshold |
|---|---|
| Non-empty patches | **≥18/30** |
| Format errors | **<10%** of model calls |
| Median first source edit | **<command 10** |
| Scorer evidence for resolved patches | Required (PASS_TO_PASS and FAIL_TO_PASS tests pass) |

### 49k dataset correction checkpoint (2026-07-11)

Before Gate B, use **only** `data/unsloth_agentic_train_swe_edit_trace_v6_budget49152` and its `_long16` shard from `data/v6_build/v6_final_49152_manifest.json`. A first compressed build was structurally rejected for empty assistant placeholders; it is archived under `*_rejected_empty_assistant` and must never be selected by a training command.

The approved source is `data/v6_build/ext_openswe_compact1200_sanitized_clean.jsonl`: `sanitize_agentic_empty_assistant.py` removed 2,954 empty assistant/no-tool messages and merged 1,754 adjacent observation pairs, then `dedup_and_trim_traces.py` trimmed 1,200 terminal observations. The rebuilt corpus has 8,500 rows (4,250 anchor + 2,500 OpenSWE + 1,750 Kwai), passed the structural scan with zero failures, and is awaiting the required 1,000-sample Gemma loss-format gate; do not launch Gate B before that log reports `failure_count: 0`.

### Authoritative cap-fallback record (2026-07-11)

The old global-swap stop is superseded. Under the user-directed `MemAvailable <2 GiB` policy, the true 49,152 run explicitly selected `flex_attention`, reached `train_begin`, then failed at **microstep 0/16**: PyTorch dispatched the path to `sdpa_dense → math_attention` and requested **128.13 GiB**. It ran 123 seconds, had `MemoryPeak=20,434,018,304` bytes and `MemorySwapPeak=0`, wrote no checkpoint, and left production green.

40,960 and 34,816 were then each run after their own required format gate and failed identically before microstep 1: their dense fallback allocations were **91.59 GiB** (with 32.34 GiB resident) and **66.48 GiB** (with 30.48 GiB resident), respectively. The 40k run measured model-load/data-prep `51s/12s`, lifetime 137s, `MemoryPeak=38,655,057,920`, `MemorySwapPeak=0`; the 34k run measured `58s/12s`, lifetime 144s, `MemoryPeak=38,655,205,376`, `MemorySwapPeak=0`.

Run the final 24,576 rung on `data/unsloth_agentic_train_swe_edit_trace_v6_budget49152_long16_le24576` with `--attn-implementation flex_attention`; this is the final sanitized build rather than an older similarly capped shard. The trainer emits timestamped `model_ready`/`data_ready`, gradient-microstep, optimizer-step, wall-clock, and ETA lines; the launcher parses those exact timestamps, so preserve the watchlog even if no optimizer step completes.

### Superseding kernel decision (2026-07-11)

The 49k/40k/34k `sdpa_dense` failures above were generated while a temporary `UNSLOTH_COMPILE_DISABLE=1` debug environment suppressed Unsloth's regional compiler. They are retained as a precise dense-fallback diagnosis, but **must not be used to choose a smaller cap**. Remove that override, retain `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`, and rerun **49,152** from cp20 on the long16 shard with `--attn-implementation flex_attention`.

The user has directly verified fused SDPA-Flash on GPU1 at the exact 49,152-token Gemma4 GQA shape with a 2.3 GiB attention peak. Treat 30–60 minutes of initial CPU-only Inductor work and a near-idle GPU as expected compilation; the launcher now reports its compiler-worker count/CPU usage as well as phase timestamps. A valid gate needs an actual completed GPU step and `checkpoint-1` with no `sdpa_dense`/`_math_attention_inner` traceback; only a failure of this compiled path permits backend reassessment.

### Step 6c — Broader Verified slice (only after gates 6a + 6b pass)

If both gates pass, proceed to a broader SWE-Bench Verified evaluation. Otherwise: **stop the lane**. Rebuild/reweight data; do not scale.

### Artifacts to Record

- hard30 semantic/full scores with run dir path.
- SWE-Lite (30 cases): submissions count, non-empty patches, format error rate, median first edit command number, scorer evidence summary.
- Decision: proceed to Verified slice OR stop and rebuild.
- If stopped: diagnosis of which gate failed and why (record in SESSION_STATE).

---

## What NOT To Do (Summary)

| Action | Reason |
|---|---|
| Stop or kill `vllm.service` / GPU0 workers | Kills production; ports 8000/8101/8103/8104 go red. |
| Raise cgroup caps (MemoryHigh, MemoryMax, MemorySwapMax) above stated values | Host RAM is 61 GiB total; exceeding caps risks system-wide OOM and host reboot (history: 2026-07-09 crash). |
| Use `pkill -f` to kill processes | Pattern matches your own SSH command string over SSH → kills the session. Kill only by exact PID from `ps -eo pid,args \| grep ... \| grep -v grep`. |
| Run training on GPU0 or without `CUDA_VISIBLE_DEVICES=1` | Competes with production vLLM for VRAM; known failure pattern. |
| Skip staged memory gates (Phase 5a/b) before full training | VRAM at 24k context already peaked ~93 GB; 32k is unproven territory. Gates catch OOM before wasting GPU time on multi-step runs. |
| Dedup by instance_id | Window-split variants are legitimate data, not duplicates. Always dedup by content-hash (sha256 of trimmed messages JSON). |
| Proceed to Phase 5 if Phase 4 format gate fails | Training a misformatted dataset wastes GPU1 time and produces unusable adapters. Diagnose and fix first. |
| Scale to SWE-Bench Verified before passing Phase 6 gates | Hard30 semantic regression and low non-empty-patch rate have killed every prior lane. Don't repeat that pattern. |

---

## Quick Reference: Key Paths on Host

```
/home/ironbcc/projects/gemma4-31B-Coder/
├── data/unsloth_agentic_32k_train_swe_edit_trace_v6_budget32768/    # v6 dataset (Phase 3)
│   ├── budget_manifest.json                                          # Full audit manifest
│   └── ...                                                           # JSONL rows
├── data/unsloth_agentic_32k_train_swe_edit_trace_v6_budget32768_long16/  # Worst-case shard (Phase 3)
│   ├── gate_manifest.json
│   └── ...
├── adapters/<training_output_dir>/checkpoint-<N>/                    # v6 training output (Phase 5)
├── phaseD_sft/dedup_and_trim_traces.py                               # New script (Phase 1)
├── phaseD_sft/compact_observations.py                                # Patched (Phase 1)
├── phaseD_sft/recover_swe_edit_v4_smoke_first.sh                     # Bounded launcher (all phases)
├── phaseH_eval/vllm_direct_model.py                                  # Eval harness
└── logs/<run_name>.log                                               # Training/eval logs
```

---

## Recovery: Host Becomes Unreachable Mid-Phase

If host goes unreachable at any point during Phases 0–6:

1. Do NOT launch a new training run or increase context — the current state may still be intact on disk.
2. Ping via LAN (`ping 192.168.50.148`) and Tailscale (`tailscale ping ironbccllm.tail0cc1d4.ts.net`).
3. If TCP port 22 is reachable but SSH banner times out, try `ssh -o ConnectTimeout=15 ironbcc@192.168.50.148`.
4. Check production ports from local machine: `for p in 8000 8101 8103 8104; do curl -fsS --max-time 3 http://192.168.50.148:$p/health >/dev/null || echo "$p FAIL"; done`.
5. If host stays unreachable: send Wake-on-LAN packet to known NIC MAC, wait 2 minutes.
6. After recovery: inspect `journalctl -b -1 -k` for OOM/kernel panic/Xid records before choosing next step.
7. Resume from the last completed gate; do not re-run gates that already passed (verify artifacts exist on disk first).
