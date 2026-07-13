# v6 @ 49,152 — Codex Execution Handoff

**Owner on handoff**: Codex agent (dispatched via Orca orchestration).
**Prereq reading (in order)**: this file → `phaseD_sft/V6_PIPELINE_RUNBOOK.md` → `phaseH_eval/SESSION_STATE.md` (sections dated 2026-07-11). Local repo of record: `/Users/ironbcc/projects/llm` branch `phaseC-verify-harness`; remote training host `192.168.50.148` (alias `ironbccllm.tail0cc1d4.ts.net`), repo `/home/ironbcc/projects/gemma4-31B-Coder`, python `.venv-train/bin/python` with `PYTHONPATH=$PWD`.

---

## 0. MAIN GOAL (north star — never lose sight of this)

Make **Gemma-4-31B the best SOLO SWE-bench coder across Python + Rust + C++**, with improved multi-step agentic reasoning, using a **FROZEN base + LoRA adapter** (no-regress guarantee — base weights never change, Python/C++ ability cannot be destroyed). Every experiment is measured against this: a candidate ships only if it **beats the cp20 baseline** on hard30 semantics AND improves SWE edit-behavior, with **no regression**.

This v6 lane specifically: prove that a **clean-data** LoRA (the dup/trailing-user corruption fixed + external Rust/agentic augmentation) trained at **49,152 context** beats cp20. The dataset corruption fixes are the real hypothesis under test — do not let scope creep bury that.

Init adapter is **cp20**: `adapters/unsloth_agentic_filtered_repair_cp10_to_s10_14336/checkpoint-10`. Do NOT init from the v5-s2 checkpoints (trained on corrupted dup data).

---

## 1. ⚠️ SMOKE-TEST CADENCE (non-negotiable — read before every phase)

**Never jump to full training or full eval. Smoke first, every time.** The user has repeatedly emphasized this. Concretely:

| After you… | Run this smoke BEFORE proceeding | Pass bar |
|---|---|---|
| change compression | format gate (1000 samples) + eyeball 5 compressed traces | `failure_count: 0`, traces still coherent |
| rebuild the dataset | structural scan + format gate | 0 non-intended dups, 0 traces ending on user, `failure_count: 0` |
| pick a new context cap | **1-step memory gate** (Gate B) at that cap | unit `Result=success`, no OOM, prod green |
| finish 5 training steps | **hard30 smoke** (30 cases, ~3 min) — NOT full SWE-bench | ≥ same-day cp20 semantic +3 |
| pass hard30 | **30-case SWE-Lite smoke** | ≥18/30 non-empty patches, <10% format errors, median first edit < cmd 10 |
| pass SWE-Lite smoke | only THEN a broader Verified slice | — |

If any smoke fails: STOP, diagnose, fix the data/config — do NOT scale up hoping it improves. A failed smoke is cheaper than a failed full run.

---

## 2. Context cap decision: 49,152 (user-approved)

Raised from 34,816 → **49,152** to recover openswe Rust/language diversity (yield 3% → 36% before compression; higher after). Consequences you must plan for:

- **VRAM is UNPROVEN at 49,152.** Estimate ~87 GB / 97.9 GB on GPU1 (from the measured 24,576 → 53.4 GB point, ≈1.35 GB/1k-tokens above ~20 GB base). This is tight. **Gate B (§5) is mandatory and may fail.**
- **Fallback ladder if Gate B OOMs or trips a threshold**: 49,152 → 40,960 → 34,816 (already built + format-passed) → 24,576 (Gate A already passed). Rebuild the budgeted dataset at the fallback cap; never raise cgroup caps to force it.
- The 34,816 dataset (`data/unsloth_agentic_32k_train_swe_edit_trace_v6_budget34816`) and its passed format gate are the **safe fallback** — keep them, don't delete.

---

## 3. openswe token compression (new work)

**Goal**: shrink openswe traces so more of the 12,094 rows fit ≤49,152, increasing Rust/C++/Python-agentic diversity. openswe median is 53k tokens (see distribution in SESSION_STATE).

### CRITICAL technical steer — compress conditioning tokens, NOT supervised tokens

The user brainstormed "obfuscation, article drop, etc." — here is the honest engineering reality, follow it:

- **Assistant turns are loss-supervised** (the model learns to reproduce them). Applying lossy transforms to them — dropping articles/stopwords from reasoning, minifying/obfuscating code — **corrupts the exact signal the model must learn** (fluent CoT + clean idiomatic code). **DO NOT do this.** It would train the model to write degraded reasoning and ugly code. This is the wrong lever.
- **Observations (user `OBSERVATION:` turns) are conditioning-only** (masked out of loss — verified: `verify_gemma_format_loss.py` shows they sit outside supervised spans). They are also **the bulk of the tokens** (openswe traces are dominated by tool-output/file-dump observations). **This is where all safe compression lives.**

### Compression techniques, ranked (apply top-down, smoke after each)

**Tier 1 — safe, lossless-ish, do these first (observation-side only):**
1. Tighten `compact_observations.py`: run openswe with `--max-observation-chars 1200 --head-lines 16 --tail-lines 10` (vs the 2400/24/16 default). The hard-cap fix (already in the script) guarantees no observation exceeds the budget.
2. Truncate giant file-view observations: `cat -n`/`sed -n` dumps of whole files → keep only the head + the hunks near the eventual edit (or a fixed head/tail window). These are the single biggest openswe token sink.
3. Strip ANSI escapes, carriage returns, collapse >2 consecutive blank lines, strip trailing whitespace.
4. Verify aggressive duplicate-observation dedup is active (repeated identical tool outputs → `[unchanged observation omitted]`).

**Tier 2 — structural, needs a smoke check (this is the highest-yield lever):**
5. **Window long trajectories**: split a >49,152 trace into ≤cap segments at assistant-turn boundaries (reuse the swe-smith `split16k` precedent already in the raw JSONL chain). Turns one unreachable 53k trace into 1-2 trainable segments. Each segment must still start with the system+task context and end on an assistant turn (run it through `dedup_and_trim_traces.py` after).
6. Forward-prune dead exploration prefixes that never lead to the edit (careful — keep enough context that the edit is motivated).

**Tier 3 — DO NOT (quality-destroying, the "article drop / obfuscation" ideas):**
7. ❌ Stopword/article dropping from assistant reasoning — corrupts supervised CoT.
8. ❌ Code minification/obfuscation — trains degraded code output.
   (If you want to experiment, gate it behind a dedicated A/B smoke where the compressed-CoT variant must MATCH the clean variant on hard30 — it won't; don't waste a full run on it.)

### Target & measurement
After Tier 1 (+ Tier 2 if needed), re-tokenize openswe and report the new yield ≤49,152. Success = a material rise in eligible Rust + Python openswe rows without any `verify_gemma_format_loss` failure. Reuse the safe tokenization pattern (§ machine safety below).

---

## 4. Dataset rebuild @ 49,152 with compressed openswe

1. Reuse cleaned pools in `data/v6_build/`: `anchor_v6.jsonl` (4,250, untouched), `ext_kwai_final.jsonl`, and the NEW compressed `ext_openswe_*` (from §3).
2. Assembly (adapt `/tmp/v6_full_pipeline3.py` on host — change `CAP_TOKENS=49152`, point openswe at the compressed file): single tokenizer load + `ThreadPoolExecutor` (see § machine safety), select rust/cpp/c openswe first, then python, then kwai fill, external hard-capped at 4,250.
3. **Mixture-ratio decision you must make**: at 49,152 with compression, eligible openswe may exceed 4,250 and could crowd out kwai (the proven SWE-agent behavior source). Recommended: cap openswe's share at ~2,000-2,500 so kwai keeps ≥40% of the external half — document the split in the manifest and rationale in SESSION_STATE. Keep anchor at 4,250 (the no-regress backbone).
4. Save as `data/unsloth_agentic_train_swe_edit_trace_v6_budget49152` (+`_long16`, +`_long16_le24576` for Gate A reuse) with a manifest recording per-source counts, `unique_content_count`, `intended_dups_kept`, `max_accepted_tokens`.
5. SMOKE: structural scan (`scan_agentic_dataset.py`) — exact dups must equal `intended_dups_kept`, 0 traces ending on user.

## 5. Format gate + memory gates

- **Format gate (mandatory, blocking)**: `CUDA_VISIBLE_DEVICES=1 .venv-train/bin/python phaseD_sft/verify_gemma_format_loss.py --data <v6 49152 dataset> --samples 1000` → REQUIRED `failure_count: 0`. (Do NOT pass empty `CUDA_VISIBLE_DEVICES` — unsloth crashes on an AMD/HIP path.)
- **Gate A (24,576)**: already PASSED this session (VRAM peak 53.4 GB) — reuse, no rerun needed.
- **Gate B (49,152) — THE critical gate**: bounded launcher, 1 step, cp20 init, on the `_long16` shard (16 longest ≤49,152 rows). Run from LOCAL repo (the launcher SSHes in itself via the host alias):
  ```
  ALLOW_64G_TRAINING=I_ACCEPT_BOUNDED_SWAP \
  DATA=data/unsloth_agentic_train_swe_edit_trace_v6_budget49152_long16 \
  INIT_ADAPTER=adapters/unsloth_agentic_filtered_repair_cp10_to_s10_14336/checkpoint-10 \
  SMOKE_OUT=adapters/v6_gateB_49152_memgate RUN_NAME=swe_edit_v6_memgate_b \
  SMOKE_UNIT=swe-edit-v6-memgate-b MAX_SEQ=49152 TRAIN_STEPS=1 WARMUP_STEPS=1 \
  bash phaseD_sft/recover_swe_edit_v4_smoke_first.sh
  ```
  Record the launcher-watchlog model-load time and measured per-step wall-clock time, plus MemoryPeak / MemorySwapPeak / peak `gpu1_used_mb` / prod health. **If GPU1 OOMs → drop to the §2 fallback ladder.** Never raise cgroup caps.

## 6. Training (SMOKE-FIRST — never a blind full run)

Only after Gate B passes. Bounded launcher, init cp20, `MAX_SEQ=<passed cap>`, `TRAIN_LR=2e-5` (prior 2e-4 caused hard30 regression; low-lr lane is required), unit `swe-edit-v6-49k-s1`, data = the v6 49152 dataset.
1. **5 steps first** (`TRAIN_STEPS=5 WARMUP_STEPS=1`) → serve checkpoint-5 at 256k (distinct LoRA names!) → **hard30 smoke** vs a SAME-DAY cp20 run on the same server. Gate: candidate ≥ cp20 + 3 semantic. (Absolute scores are serving-path-dependent — always re-baseline cp20 same-day; do NOT compare to the historical 25-26/30.)
2. If hard30 passes → **30-case SWE-Lite smoke** (≥18/30 non-empty patches, <10% format errors, median first edit < cmd 10).
3. If SWE smoke passes → extend training (more steps/epoch) and re-smoke at each checkpoint. Only a passing smoke justifies the next scale-up.
4. Only after all smokes pass → broader SWE-bench Verified slice.

## 7. Machine safety (learned the hard way this session — obey)

- **GPU0 production vLLM is OFF-LIMITS.** Never stop/kill `vllm.service` or GPU0 workers. Prod health ports 8000/8101/8103/8104 must stay green through every phase (`for p in 8000 8101 8103 8104; do curl -fsS --max-time 3 http://127.0.0.1:$p/health; done`).
- **Host RAM = 64 GB / 61 GiB usable; historical failures were RAM-related.** For ANY CPU-parallel job (tokenization, compaction): load the tokenizer **ONCE in the main thread** and use `ThreadPoolExecutor` (Gemma tokenizer is `is_fast=True`, releases GIL). **NEVER** use `multiprocessing.Pool` with the tokenizer loaded in `_init_worker` — this session that pattern spawned 16 tokenizer copies and drove host swap 55 GB → 102 GB in 3 minutes, near-crashing the box. Add a `/proc/meminfo` MemAvailable preflight (abort < 15 GB). Check `free -g` within ~30 s of every parallel launch — "it's running" ≠ "running correctly".
- **Kill only by exact numeric PID** (`ps -eo pid,args | grep ... | grep -v grep`). NEVER `pkill -f` / `pgrep -f` over SSH — it matches your own command and kills the session.
- Training is GPU1-only, always via the bounded launcher (`recover_swe_edit_v4_smoke_first.sh`, `ALLOW_64G_TRAINING=I_ACCEPT_BOUNDED_SWAP`, cgroup 36G/45G/16G-swap). Launch it from the LOCAL Mac repo — it manages its own SSH.
- Verify claims about live state with a fresh command in the same turn — never narrate status from a stale proxy.

## 8. Current state at handoff (verify before acting)

- v6 @ 34,816 dataset built + format-gate PASSED (the fallback). Gate A (24,576) memory gate PASSED (VRAM 53.4 GB). Gate A checkpoint dir: `adapters/unsloth_agentic_swe_edit_trace_v6_gateA_24576_memgate`.
- A Gate A run may still be finalizing on `swe-edit-v6-memgate-a.service` — check `systemctl --user show ... -p SubState -p Result` before launching new GPU1 work; confirm GPU1 ≈ 2 MiB free first.
- Cleaned pools + all manifests in `data/v6_build/` on host. Phase-1 pipeline code (dedup/trim/scan/convert/compact) is at 93/93 tests (`.venv/bin/python -m pytest phaseD_sft/tests/ -q` from repo root).
- First deliverable: §3 openswe compression, smoke-validated, then §4 rebuild at 49,152.

---

## 9. Execution update: reproducible OpenSWE compression (2026-07-11, in progress)

### Completed observation-only compression

- Input: `data/v6_build/ext_openswe_merged.jsonl` (12,094 converted OpenSWE traces).
- Run the existing deterministic compressor with **only observation-side changes**:
  ```bash
  export PYTHONPATH=$PWD
  .venv-train/bin/python phaseD_sft/compact_observations.py \
    --in data/v6_build/ext_openswe_merged.jsonl \
    --out data/v6_build/ext_openswe_compact1200.jsonl \
    --manifest data/v6_build/ext_openswe_compact1200_manifest.json \
    --max-observation-chars 1200 --head-lines 16 --tail-lines 10
  .venv-train/bin/python phaseD_sft/dedup_and_trim_traces.py \
    --in data/v6_build/ext_openswe_compact1200.jsonl \
    --out data/v6_build/ext_openswe_compact1200_clean.jsonl \
    --manifest data/v6_build/ext_openswe_compact1200_dedup_manifest.json
  ```
- Result: observations fell from `1,848,651,961` to `668,260,323` characters (a **63.85%** reduction). The cleaned output kept all 12,094 rows, with 0 exact duplicates, 0 trimmed trailing messages, and 0 dropped traces.
- The compressor touched no assistant content or tool calls. Its head/important/tail policy safely compresses large file-view/tool-output observations; do not apply article removal, minification, or obfuscation to assistant turns.

### Required safe measurement recipe

1. Before any CPU-parallel audit, verify the four production health ports, require `/proc/meminfo` `MemAvailable >= 15 GiB`, and run `free -g` within 30 seconds of launch.
2. Load the Gemma tokenizer **once in the main process** and use `ThreadPoolExecutor(max_workers=16)` for row token counts. Never use a tokenizer-loading `multiprocessing.Pool`.
3. Tokenize `ext_openswe_compact1200_clean.jsonl` with the same message-token sum used by the assembly pipeline, then retain the JSON audit beside the run: `/tmp/v6_openswe_compact1200_49152_audit.json`.

At 49,152 tokens this produced **7,855 / 12,094 eligible rows (64.95%)**, versus the prior uncompressed estimate of 4,392 / 12,094 (36%). Eligible language counts: Rust 1,132, C 15, Python 6,708. The token distribution is p50 44,994, p90 60,697, p99 81,112, max 151,510.

### Mixture decision and current gate

- Tier-2 assistant-boundary windowing is **not needed** for this build: Tier 1 alone produces far more than the 2,500 OpenSWE rows we can safely use while preserving Kwai behavior signal.
- The 49k assembly must cap OpenSWE at **2,500** rows (select C/Rust first, then Python) and take **1,750 Kwai** rows. That keeps Kwai at 41.2% of the 4,250-row external half; anchor remains 4,250 rows.
- A 12,094-row HF copy is at `data/v6_build/ext_openswe_compact1200_clean_hf` solely for the required compression format gate. It **passed** at `/tmp/v6_openswe_compact1200_format_gate.log`: 1,000 samples, `failure_count: 0`, 104,735 assistant messages in valid supervised Gemma spans, and `fallback_spans: 0`. The legacy verifier took about 1h50m on this very multi-turn sample; the synced progress telemetry will expose the true ETA from future runs.
- The verifier currently has 53 threads but consumes about one CPU core: its repeated `apply_chat_template` prefix checks are Python/Jinja-dominated. Do **not** parallelize it by spawning separate tokenizer processes; that violates the host-RAM rule. A future shared-tokenizer `ThreadPoolExecutor` fast path is acceptable only after a small benchmark proves real speedup and format-equivalent output.
- The v6 long-running scripts now emit durable progress telemetry: completed/total, it/s, elapsed, ETA, and total estimate. This covers `compact_observations.py`, `dedup_and_trim_traces.py`, `filter_token_budget.py`, `scan_agentic_dataset.py`, and future `verify_gemma_format_loss.py` runs. The bounded launcher also records launch-to-data-ready/model-load timing, step intervals, final trainer runtime/per-step time, GPU1 used MiB, and cgroup peaks in its watchlog.
- The inherited 24,576 Gate A finalized successfully while this work ran: checkpoint present, 28m12s elapsed, `MemoryPeak=38,655,455,232` (~36.0 GiB), `MemorySwapPeak=644,567,040` (~615 MiB), and GPU1 returned to ~2 MiB before the format check. Production remained green.

### Sanitization correction and final 49k rebuild (2026-07-11)

The first compressed 49k assembly correctly caught a separate source defect at the **structural smoke**: 638 selected traces contained empty, non-tool-call assistant placeholders. This is zero-supervision content, so those output directories were archived as `*_rejected_empty_assistant` and are **not trainable**; do not bypass this stop gate.

Use the reproducible observation-side repair below before budgeting the compressed OpenSWE pool. It removes only empty assistant turns, merges the two newly adjacent `OBSERVATION:` user messages, then applies the normal trailing-user trim; it never changes a non-empty assistant response, code, reasoning, or tool call:

```bash
export PYTHONPATH=$PWD
.venv-train/bin/python phaseD_sft/sanitize_agentic_empty_assistant.py \
  --in data/v6_build/ext_openswe_compact1200_clean.jsonl \
  --out data/v6_build/ext_openswe_compact1200_sanitized.jsonl \
  --manifest data/v6_build/ext_openswe_compact1200_sanitized_manifest.json \
  --progress-every 500
.venv-train/bin/python phaseD_sft/dedup_and_trim_traces.py \
  --in data/v6_build/ext_openswe_compact1200_sanitized.jsonl \
  --out data/v6_build/ext_openswe_compact1200_sanitized_clean.jsonl \
  --manifest data/v6_build/ext_openswe_compact1200_sanitized_dedup_manifest.json \
  --progress-every 500
```

Measured repair: all 12,094 rows were retained; 2,954 empty assistant placeholders were removed, 1,754 adjacent observation pairs merged, and 1,200 newly terminal user observations trimmed. A deterministic five-trace hash check (`/tmp/v6_openswe_sanitized_spotcheck.json`) proved that every non-empty assistant-content/tool-call sequence was byte-equivalent before and after repair.

The corrected 49,152 assembly is now complete at `data/unsloth_agentic_train_swe_edit_trace_v6_budget49152` (plus `_long16` and `_long16_le24576`). Its manifest, `data/v6_build/v6_final_49152_manifest.json`, reports 8,500 rows: 4,250 anchor, 1,135 OpenSWE Rust/C, 1,365 OpenSWE Python, and 1,750 Kwai; Kwai is 41.2% of the external half, the largest accepted row is 49,137 tokens, `unique_content_count=6,976`, and only the expected 1,524 intentional duplicates remain. The corrected pool has 7,801 OpenSWE rows eligible at 49,152; the mandatory structural scan passed with `failure_count: 0`, so the next blocking action is the 1,000-sample final dataset format gate.

**PASSED:** the final dataset format gate completed with 1,000 samples and `failure_count: 0` in `/tmp/v6_49152_format_gate.log` (about 29 minutes with the new ten-sample ETA lines). GPU1 returned to ~2 MiB and the four production health ports stayed green; Gate B is now authorized, using the safety-corrected launcher that only stops its explicitly named systemd unit and contains no broad `pgrep`/`pkill` path.

### Gate B result and authorized 40,960 fallback (2026-07-11)

**49,152 did not pass the bounded memory gate.** The one-step cp20-init run reached data-ready in **89 seconds** but was stopped during its first 16-way accumulation before a training step completed: system swap reached **16,866 MiB**, above the fixed **16,384 MiB** safety guard. Do not treat the transient GPU fit as a pass and do not raise any cgroup cap; the only valid per-step result is **unavailable** because the step never finished.

Record these measured 49k values: peak sampled `gpu1_used_mb=90,432` (~88.3 GiB); `MemoryPeak=38,666,403,840` bytes (~36.0 GiB); `MemorySwapPeak=2,291,339,264` bytes (~2.13 GiB). The launcher stopped only `swe-edit-v6-memgate-b.service`; after stop GPU1 returned to 2 MiB, MemAvailable to ~53 GiB, swap recovered, no checkpoint existed, and production ports remained green.

The next ladder rung is built from the **same sanitized-clean source**, not the rejected first assembly: `data/unsloth_agentic_train_swe_edit_trace_v6_budget40960` (+ `_long16`, `_long16_le24576`), manifest `data/v6_build/v6_final_40960_manifest.json`. It contains 8,500 rows: the unchanged 4,250 anchor, 622 OpenSWE Rust/C, 1,878 OpenSWE Python, and 1,750 Kwai (41.2% external); the 40k structural scan passed with `failure_count: 0`. Its 1,000-sample loss-format gate is mandatory and pending before launching `swe-edit-v6-memgate-b-40960`.

### User-directed Gate B retest policy (2026-07-11)

The global swap-used abort was too conservative for this host's 255 GiB swap allocation and has been removed at user direction. The launcher now stops for host pressure **only when `MemAvailable < 2,048 MiB`**; it still enforces GPU1-only execution, the unchanged 36G/45G/16G cgroup limits, explicit named-unit stopping, timeout, and all production-port health checks. Re-test the 49,152 gate first from clean GPU1; do not infer a cap failure from the old global-swap threshold.

### Gate B retest: actual 49,152 attention-kernel failure (2026-07-11)

The revised-policy 49,152 cp20-init one-step run (`swe-edit-v6-memgate-b-flex.service`) is the authoritative Gate B result; it **failed before optimizer microstep 1 and wrote no checkpoint**. This was not a host-memory or swap stop: the explicit post-load `flex_attention` override was applied and logged, but PyTorch 2.10 dispatched it to `torch._higher_order_ops.flex_attention.sdpa_dense`, whose dense math fallback tried to allocate **128.13 GiB** on the 94.97 GiB GPU.

- Unit execution was 123 seconds (17:20:52–17:22:55 UTC), `MemoryPeak=20,434,018,304` bytes, `MemorySwapPeak=0`; GPU1 returned to 2 MiB and production remained green. The previous 60-second watch poll observed `model_load_seconds=64`, but that was an upper bound because model-ready and data-ready were first seen in the same poll; it is not a valid step-time measurement.
- The source traceback reaches Transformers `flex_attention_forward`, then PyTorch `sdpa_dense → math_attention → query @ key.T`. Therefore setting the config name alone does not provide a sparse/flash kernel on this installed Torch/Unsloth stack.
- The trainer and launcher now write timestamped progress events (`model_ready`, `data_ready`, every gradient-accumulation microstep, optimizer-step wall time, ETA). The next gate will report exact model-load/data-prep intervals even at a 60-second launcher check cadence.
- Do not retry 49k with the same Torch 2.10 FlexAttention path. Follow the required ladder with a fresh, fully-gated 40,960 one-step run; if it fails for the same dense allocation, continue 34,816 then 24,576. Do not raise cgroup limits or claim an unmeasured per-step time.

### Ladder measurements: 40,960 and 34,816 (2026-07-11)

Both intermediate rungs were run only after their clean-data structural and 1,000-sample format gates had passed, and both failed at microstep `0/16` for the same dense fallback. At **40,960**, exact trainer timestamps measured model load **51 s** and data preparation **12 s**; it then requested **91.59 GiB** with 32.34 GiB resident, ran 137 s total, and recorded `MemoryPeak=38,655,057,920`, `MemorySwapPeak=0`. At **34,816**, model load was **58 s**, data preparation **12 s**; it requested **66.48 GiB** with 30.48 GiB resident, ran 144 s total, and recorded `MemoryPeak=38,655,205,376`, `MemorySwapPeak=0`.

GPU1 returned to 2 MiB and all production ports remained green after each failure. The only remaining prescribed capability gate is a new **24,576** one-step run on the final sanitized 49k build's `_long16_le24576` shard; do not treat older gate artifacts as proof for a differently selected clean shard.

### Supersession: restore compiled FlexAttention before judging a cap (2026-07-11)

The preceding 49k/40k/34k `sdpa_dense` results are **diagnostic only, not valid cap gates**: they were run with the temporary `UNSLOTH_COMPILE_DISABLE=1` debug setting, which prevented Unsloth's regional `torch.compile` path from producing its fused block-mask kernel. That environment override has been removed; telemetry, the `expandable_segments:True` allocator setting, named-unit safety, cgroup limits, and the no-global-swap-abort policy remain.

The user directly verified GPU1 SDPA-Flash at the exact Gemma4 GQA geometry (32 query heads, 16 KV heads, head dimension 256, 49,152 causal tokens) with a 2.3 GiB peak. Therefore the 128/91/66 GiB requests identify only the dense score-matrix fallback, not an intrinsic 49k hardware limit; **do not step down the cap further**. Re-run the 49,152 cp20-init long16 one-step Gate B with regional compilation enabled, allow an initial 30–60 minute CPU-only Inductor compile (GPU near-idle is expected), and require a checkpoint-1 plus an actual completed GPU step before any cap decision.
