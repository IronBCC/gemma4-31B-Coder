# Gemma 4 31B → Best Coding Model: Maximal Build Spec (Final)

**Base model (fixed):** `google/gemma-4-31B-it` — dense 30.7B, 256K ctx, thinking mode, native function-calling, native MTP draft heads.
**Mandate:** apply every best technique, stacked, to make this the strongest coding/agentic model it can be. Target to beat: **Qwen3.6-27B** (SWE-bench Verified 77.2, SWE-bench Pro 53.5 on Qwen's own scaffold).
**Deploy:** vLLM, NVFP4 (W4A4) on RTX 6000 Pro Blackwell.

---

## 0. Mandate & one honest calibration

Base is locked to Gemma 4 31B; this plan maximizes it and does not consider alternatives. One calibration to keep us rigorous (stated once, then we execute): Gemma 4 31B's native strength is reasoning/math, not code, so the strategy leans on the three levers with the most headroom on a fixed base — **(1) scaffold engineering, (2) execution-reward RL, (3) raising the coding floor via continued pretraining** — rather than on distillation, which is bounded by the base's capacity. Realistic outcome: match/approach Qwen3.6-27B on a fair, identical-scaffold comparison; surpassing it is the stretch and depends mostly on out-engineering the agent stack and RL. We give it every chance.

The comparison to Qwen3.6-27B must be **fair**: Qwen's 53.5 Pro uses its internal scaffold at temp 1.0 / 200K ctx on a *refined* Pro set. We run Qwen3.6-27B ourselves under our identical harness and precision as the live baseline; we do not compare against its self-reported number.

---

## Critical path (start here)

Most of this plan is grant-gated and expensive. The cheap, high-ROI core that needs **no grant** is **Phase A + B on your box**: build the agent scaffold and add best-of-n + a hybrid verifier, then measure against **Qwen3.6-27B run under your identical stack**. This may deliver most of the achievable win (the scaffold alone took a 27B to 90% Verified) and it is the decision gate for everything else: if A+B don't lift over the same-stack Qwen baseline, the ceiling is the base model and the capability phases (C/D-full/E) are not worth the grant spend. Do A+B first, then decide.

---

## 1. The lever stack (all applied, ordered by ROI)

| # | Lever | Effect | Cost | Where |
|---|---|---|---|---|
| 1 | **Agent scaffold engineering** | Largest single gain (scaffold-only took Qwen3.6-27B to 90% Verified, no training) | Low–med | Your box |
| 2 | **Inference-time scaling** (best-of-n + hybrid verifiers) | +10 pts class gains, no weight change | Med (compute at inference) | Your box |
| 3 | **Speculative decoding** (native MTP → EAGLE-3 refresh after FT) | ~1.5–2.5× single-request latency | Free (native); cheap to refit | Your box |
| 3b | **Serving config** (prefix caching, KV-quant, chunked prefill, replicas) | 3–5×+ throughput for collection/RL | Free (vLLM flags) | Your box |
| 4 | **Continued pretraining on code** | Raises Gemma's coding *floor* (its weak axis) | High | Grant |
| 5 | **Agentic SFT (rejection-sampling)** | Teaches the loop + format; RL cold-start | Med (collection) → high (full SFT) | Box (QLoRA) → Grant (full) |
| 6 | **GRPO RL (execution reward)** | The teacher-independent capability push | Highest | Grant |
| 7 | **Distillation augmentation** | Reasoning traces; bounded by base capacity | Med | Optional |
| 8 | **NVFP4 + QAT** | Deploy at 4-bit without handicapping capability | Med | Box/Grant |

The pipeline is **scaffold + inference-time first** (biggest cheap wins, base-agnostic), then **floor-raising + SFT + RL** (capability), with **NVFP4+QAT** as the deployment artifact measured separately.

---

## 2. Fair-eval protocol (define first; gate everything)

- **Run the target yourself.** Stand up `Qwen/Qwen3.6-27B` under *our* scaffold, harness, precision, and turn/cost budget. That is the baseline to beat — not its blog number.
- **Capability vs deployment split.** Make the "beat Qwen" claim in **BF16-vs-BF16** (equal precision). NVFP4 is the *deployment* artifact, measured separately, with the quant gap reported. (Qwen ran its results at FP8; a 4-bit-vs-FP8 claim would be unfair to our own model.)
- **Eval suite ("best coder" ≠ Pro only):**
  - Primary: SWE-bench **Pro** public (731), standardized mini-SWE-agent harness.
  - Dev loop (cheap/fast): SWE-bench **Lite → Verified**.
  - Breadth: **Terminal-Bench 2.0**, **LiveCodeBench**, **multilingual** (SWE-bench Multilingual / Multi-SWE-bench).
- **Report:** resolve rate (pass@1 and best-of-n+reranked), empty-patch rate, tool-call validity, per-repo/per-language tables, latency (tok/s, NVFP4 + speculative decoding), decontamination manifest.
- **Integrity gate:** every training source diffed against Pro's 41 repos + Verified; n-gram overlap check; dropped-count manifest.

---

## 3. Phase A — Scaffold engineering (centerpiece)

This is the highest-ROI work and base-agnostic. The proof: an engineered agent stack took **Qwen3.6-27B-FP8 to 90.0% on SWE-bench Verified with no fine-tune, no distill** — the scaffold, not the weights, drove it. Build the best stack you can around Gemma 4 31B:

- **Fine-grained editing tools** (the documented ~2.6-pt edge of finer edit tools): structured search/replace + AST-aware edits, not blind diff application.
- **Localization** pass (retrieval over repo → candidate files/functions) before editing.
- **Test-execution feedback loop:** run tests, feed failures back, iterate; this is where most resolves come from.
- **Multi-turn budget + crash/retry recovery** (the 90% run used crash-retry recovery).
- **Protocol/format hardening:** lock system prompt + tool schema to Gemma 4's expected format; root-cause the vLLM tool-call parser bug here.

Tune the scaffold on your box against Lite/Verified. Expect this phase alone to be the biggest mover.

---

## 4. Phase B — Inference-time scaling (no weight change)

- **Best-of-n + hybrid verifiers.** Sample N trajectories; rank with a **hybrid verifier** combining execution-based (run tests) and execution-free (trained ORM) signals. R2E-Gym's **Hybrid Test-time Scaling** reached **51% pass@1 on SWE-bench Verified** this way — SOTA for open SWE agents at the time. Train the execution-free ORM on your collected trajectories (Phase D).
- **Speculative decoding (latency).** Gemma 4 ships native MTP draft heads; enable in vLLM for a ~1.5–2.5× single-request speedup so best-of-n stays affordable and the agent loop stays interactive. Full treatment — speculator refresh after fine-tuning and the throughput-vs-latency serving split — is in Phase G.

These compound with the scaffold and require zero weight training.

---

## 5. Phase C — Raise the coding floor: continued pretraining (compute grant)

This is the lever that lifts Gemma's *coding capacity* — its weak axis — before any SFT. Domain-adaptive continued pretraining (a.k.a. mid-training) on a large, clean, deduplicated code corpus:

- **Corpus:** `bigcode/the-stack-v2` filtered to high-quality + target languages (Python, Java, Kotlin, Rust, C++), plus OpenCoder-style mid-training data (`OpenCoder-LLM/opc-sft-stage1/2` annealing mix) and fill-in-the-middle objectives.
- **Objective:** next-token + FIM on code; keep a replay of general/reasoning data to avoid eroding Gemma's reasoning strength.
- **Scale:** this is heavy (full-parameter, many tokens) → grant. It's optional but the highest-leverage capability move for a fixed weaker-coding base.
- **Decontaminate** the corpus against Pro/Verified repos before training.

If grant budget is tight, this is the phase to scale down first — but it's also the one that most directly closes the gap to Qwen on raw coding.

---

## 6. Phase D — Agentic SFT (rejection sampling; RL cold-start)

**Trajectory data:**
- **Primary source: your own execution-verified teacher traces — see D.1 below.** Collect via an R2E-Gym-style Docker fleet (~300–500 MB/image); **keep only test-passing trajectories** (Fail2Pass + Pass2Pass).
- **Preserve reasoning across turns** in collected trajectories — carry `reasoning_content` forward between tool calls rather than discarding it (K2.7-Code's forced preserve-thinking; reported to improve coding-agent performance and it matches how you'll serve).
- **For pure tool-use data, replicate Kimi K2's 3-stage agentic synthesis** (arXiv 2507.20534): start from real MCP tools → expand to a large synthetic tool set → generate diverse agents/tasks/multi-turn rollouts in simulated environments → keep only trajectories that pass a judge rubric. MCP-centric and directly in your wheelhouse.
- Augment with ready sets: `nvidia/SWE-Hero-openhands-trajectories` (34k, permissive/commercial-OK), `SWE-bench/SWE-smith-trajectories`, `ByteDance-Seed/Multi-SWE-bench_trajs` (multilingual).
- Tool/format seasoning: `Salesforce/xlam-function-calling-60k`, `Salesforce/APIGen-MT-5k`, `Team-ACE/ToolACE`, `NousResearch/hermes-function-calling-v1`.

### D.1 — Execution-verified teacher-trace pipeline (key SFT source)

A first-class trace source that sits alongside the gyms: frontier-grade reasoning, execution-gated, license-clean. This is the disciplined version of "distill strong CoT" — same goal as scraped trace dumps, none of the provenance risk, and unlike raw transcripts it actually clears your test-pass quality bar.

**Teacher selection (the gating discipline).**
- Use a teacher whose **license permits training on its outputs**, and **self-host it** — a hosted API adds a second ToS layer that often forbids competitor-training regardless of the weight license; running open weights yourself collapses it to just the (permissive) model license.
- Clean choices, self-hosted: **Qwen/Qwen3.6-27B or Qwen3-Coder** (Apache-2.0; strongest coding teacher — and it's your benchmark target, so a sensible cold-start source), **DeepSeek** (MIT), **gpt-oss** (Apache-2.0). **Kimi K2.6 / K2.7-Code** (Modified-MIT) only after reading the modified clauses re: outputs.
- Avoid: closed-model API outputs and any "scraped preview" trace dumps (license + provenance exposure).
- Record `teacher_model` + `teacher_license` on every trace for auditability.

**Generation loop (agentic, execution-grounded).**
- Drive the teacher with the **same harness you'll deploy** (mini-SWE-agent) over tasks from the executable gyms (R2E-Gym / SWE-Gym / SWE-rebench-V2 / Multi-SWE) inside the Docker env. This yields *agentic trajectories*, not just static CoT.
- Record the full trajectory — messages, tool calls, tool results, reasoning — and **preserve `reasoning_content` across turns**.
- **Best-of-N:** sample N attempts per task (raise N on hard tasks) to lift yield.
- Hard tasks: **gold-patch-assisted** generation (+~38% usable); tag these `synthetic_second_attempt=true` and keep them a **minority aux** slice (real-CoT-main / second-attempt-aux pattern).

**Execution gate (the quality bar).**
- Keep a trajectory **only if** its final patch/solution passes the task's tests (Fail2Pass + Pass2Pass for SWE; unit tests for code-gen) — rejection sampling with a frontier teacher.
- Drop on: test failure, format-invalid tool calls, limit-hit/truncated, or pathological length.

**Dedup + decontamination.**
- Dedup near-duplicate trajectories (MinHash or embedding) so easy/templated solves don't dominate.
- Decontaminate: drop any task whose repo ∈ {Pro 41 ∪ Verified}; 13-gram problem-statement overlap via open-r1 `decontaminate.py`. Keep the dropped-count manifest.

**Schema (extends §12 trajectory format with provenance/quality fields):**
```json
{ "...trajectory + instance fields (see §12.5)...":  "",
  "teacher_model": "Qwen/Qwen3.6-27B",
  "teacher_license": "Apache-2.0",
  "generation_harness": "mini-swe-agent",
  "sampled_k": 8, "passed": true,
  "synthetic_second_attempt": false,
  "n_turns": 14, "n_tokens": 9123 }
```
Serialize into the Gemma 4 chat + tool format at training time (same template/EOS as serving).

**Positioning.** Primary SFT source alongside the gyms, and the RL cold-start. Pair it with the ready sets (D below) for breadth; let this pipeline carry the high-quality core.

**Curriculum (3-stage, validated):**
1. **Format Inception** (<4,096 tok): stabilize `<think>` boundaries + tool-call format.
2. **Complexity Expansion** (4,096–8,192): harder coding + multi-turn agent traces; reason→action→feedback.
3. **Long-Context SFT** (→32K): long/multi-turn trajectories **with short-sample replay**.

**Execution:** QLoRA on your box to validate the pipeline on Lite/Verified (DDP ×2 — only adapter grads sync over PCIe); then **full SFT on the grant** for capability (evidence: full SFT > LoRA on SWE). This SFT checkpoint is the RL cold-start.

**Gemma-4 landmines (mandatory):**
- KV-shared layer / `use_cache` bug → use Unsloth's patched loader (gradient checkpointing forces `use_cache=False` → divergence otherwise).
- Use the `gemma-4-thinking` template; train and serve identical template + EOS.
- Keep ≥75% reasoning-style examples to preserve thinking behavior.

**Hyperparameters (QLoRA validation):** `r=32` (64 if headroom), `alpha=32`, dropout 0, all proj modules, `lr=2e-4` cosine 3–5% warmup, 1–3 epochs (overfits fast), `max_seq_len=8192` (raise in Stage 3).

---

## 7. Phase E — GRPO RL with execution reward (core capability lever)

The only teacher-independent way to push past a strong baseline. DeepSWE trained a 32B purely via this.

- **From** the SFT checkpoint, on-policy **GRPO**.
- **Reward:** sparse outcome — 1 iff patch passes selected Fail2Pass + Pass2Pass tests within a time cap (~5 min for training speed).
- **Token-efficiency term:** lightly penalize tokens-to-solve so the model resolves in fewer thinking tokens without losing reasoning — Kimi K2.7-Code cut thinking-token usage ~30% this way. **Tune the penalty weight as a deliberate sweep**; over-weighting it suppresses reasoning and hurts resolve rate.
- **Reward-design reference:** Kimi K2 combines **verifiable rewards** (rule-based 1/0 — exactly your test-pass signal for coding) with **rubric-driven self-critique** for non-verifiable aspects (arXiv 2507.20534); K2.5 adds a unified agentic-RL environment for parallel rollouts (arXiv 2602.02276). Your coding reward is the verifiable kind; reserve rubric-self-critique for style/quality terms if you add them.
- **Data/envs:** `R2E-Gym` (best for RL per DeepSWE) + Multi-SWE-RL (4,723 instances, cross-language). **Pre-filter** tasks too-easy/too-hard via base rollouts to avoid high solve-none rate.
- **Config:** ~64k max ctx, ~100 max env steps, async rollout framework.
- **Iterate:** optionally cycle SFT (on new RL-discovered successes) ↔ RL.

This is promoted to a **core** phase, not a stretch — it's the lever most likely to move Gemma toward parity-and-beyond on Pro.

---

## 8. Phase F — Distillation augmentation (optional, bounded)

- SFT on frontier-teacher reasoning trajectories can help, but is **bounded by the base's capacity** and can't beat the teacher.
- If used, reconstruct compressed reasoning (Trace-Inversion) into full CoT to avoid "reasoning fractures." Caveat: a community Opus-trace-inversion attempt on a Gemma 4 MoE variant came out unusable — treat CoT-distillation as a supplement to, never a replacement for, execution-verified trajectories + RL.

---

## 9. Phase G — Deployment: NVFP4 + QAT, speculator refresh, serving config

Goal: ship 4-bit with minimal capability loss and maximal speed so the deployed model isn't fighting the "beat Qwen" target with one hand tied. Order of operations: **finalize weights (post-RL) → quantize (G.1) → refresh speculator (G.2) → configure serving per workload (G.3)**.

### G.1 — NVFP4 + QAT
- **PTQ → QAT.** PTQ to NVFP4 (LLM Compressor `scheme="NVFP4"` or ModelOpt `NVFP4_DEFAULT_CFG`), then **QAT / QAD via NVIDIA Model Optimizer** to recover PTQ accuracy loss. Export Unified HF checkpoint → vLLM.
- **Calibrate on your own SWE data** (512–1024 domain-matched samples; >1024 rarely helps >0.1–0.2%).
- **Measure the quant gap** vs the BF16 capability number. If the NVFP4 drop is material on Pro, push QAT harder; keep an **FP8 fallback** for the capability claim and NVFP4 for max-speed deployment.
- RTX 6000 Pro (SM120) supports NVFP4 W4A4 inference (activation quant included).
- **Re-verify tool-call format after every re-quant** — known failure mode.

### G.2 — Speculator refresh (EAGLE-3)
Gemma 4's **native MTP draft heads** were trained on the *original* weights. After continued pretraining + full SFT + RL the base distribution shifts, so the MTP heads drift and acceptance rate drops. Refresh the speculator on the **final** model:

- **QLoRA-only checkpoints:** base barely moves — native MTP is probably still fine; verify acceptance before bothering.
- **Full SFT / continued-pretraining / RL checkpoints:** retrain an **EAGLE-3** draft head on the fine-tuned model. EAGLE-3 reuses the target's top-layer features and predicts the next feature, then uses the target's own LM head for the draft token. It self-distills: the draft head's training data is generated by running the target on *your* examples, so it learns your model's output patterns (generic/out-of-box drafts land at ~0.6–0.8 acceptance on domain/long-context work; a model-specific EAGLE-3 head recovers most of the gap).
  ```bash
  # 1) build training cache from the target's own outputs on your SWE data
  python build_eagle3_dataset_cache.py \
    --target-model-path <final-gemma4-31b> \
    --train-data-path ./data/swe_trajectories.jsonl \
    --chat-template gemma-4-thinking --max-length <long; see cap note>
  # 2) train the draft head (cheap relative to main training)
  torchrun --nproc_per_node 2 train_eagle3.py \
    --target-model-path <final-gemma4-31b> --output-dir ./eagle3-head ...
  ```
- **Tooling:** SpecForge / EAGLE-3 training scripts; NVIDIA Model Optimizer also trains speculators. On Blackwell, stack **P-EAGLE** (parallel drafting) for ~1.05–1.69× over vanilla EAGLE-3.
- **Validate acceptance against the actual deployed (NVFP4) checkpoint**, not just BF16 — quantization slightly shifts outputs and can change acceptance.
- **Context cap:** vLLM speculative models have historically had a ~2048 context cap — confirm/extend for long agent trajectories, or the speculator silently stops helping past 2K.

### G.3 — Serving configuration (choose per workload)
Speculative decoding and high-concurrency batching **partially trade off** (drafting adds verify-compute; at high batch you're already compute-bound, so spec decoding helps less or hurts). Run two distinct serving profiles:

**Throughput profile — trajectory collection / best-of-n / RL rollouts** (the big multipliers; usually *no* spec decoding):
```bash
vllm serve <nvfp4-checkpoint> \
  --max-model-len 65536 \
  --enable-prefix-caching \          # APC: agent loops re-send the whole trajectory → near-free repeated prefills
  --kv-cache-dtype fp8 \             # (or nvfp4 KV if supported) → bigger batches/longer ctx
  --enable-chunked-prefill \
  --max-num-seqs 256 \              # high concurrency = the main throughput lever
  --enable-auto-tool-choice --tool-call-parser <gemma4_parser>
# Run ONE replica per GPU (data-parallel, no TP) behind a router → ~2× aggregate.
```

**Latency profile — interactive single-user coding** (stack spec decoding + Blackwell):
```bash
vllm serve <nvfp4-checkpoint> \
  --max-model-len 65536 \
  --speculative-config '{"method":"eagle3","model":"./eagle3-head","num_speculative_tokens":5}' \
  --enable-prefix-caching \
  --enable-auto-tool-choice --tool-call-parser <gemma4_parser>
# CUDA graphs on (default). Consider thinking-off for the fast local agent (quality/speed tradeoff).
# Free, code-friendly extra: n-gram / prompt-lookup speculation (no training) for copy-heavy patch edits.
```
- **Expected:** throughput profile clears 3–5× (batching + APC + KV-quant + 2 replicas), often more. Latency profile ~2–3× via EAGLE-3 + NVFP4 + CUDA graphs, stretching to 3–5× with P-EAGLE + thinking-off.
- All vLLM flags above are illustrative — confirm exact names against your vLLM version.

---

## 10. Phase H — Benchmark (execute §2 protocol)

1. Deploy candidate on vLLM (BF16 for capability, NVFP4 for deployment number).
2. Run mini-SWE-agent over Pro public + Verified/Lite + Terminal-Bench/LiveCodeBench + multilingual, fixed budget.
3. Apply patches in official Docker harness; run hidden tests.
4. Score and compare vs: (a) your-harness Qwen3.6-27B baseline, (b) Phase-0 Gemma baseline, (c) prior checkpoint.
5. Report pass@1 and best-of-n+reranked.

---

## 11. Sequencing & integrity

| Phase | Where | Gate |
|---|---|---|
| 0 Baseline + fair-eval setup | Box | Reproducible Gemma + Qwen baselines under same harness |
| A Scaffold | Box | Measurable lift over baseline |
| B Inference-time + MTP | Box | Best-of-n+verifier lift; throughput acceptable |
| C Continued pretraining | **Grant** | Coding-floor lift on held-out code evals |
| D Agentic SFT (QLoRA→full) | Box → **Grant** | QLoRA lift on Verified/Lite before full-SFT spend |
| E GRPO RL | **Grant** | SFT stable; envs scaled; difficulty-filtered |
| F Distillation (opt) | Box/Grant | Only if it beats SFT-only on eval |
| G Deploy: NVFP4+QAT, EAGLE-3 refresh, serving config | Box/Grant | Quant gap within BF16 tolerance; speculator acceptance high on the deployed NVFP4 checkpoint |
| H Benchmark | Box | Pro number + full suite + fair comparison |

**Rules:** scaffold + inference-time (A/B) come first and may deliver most of the win cheaply; never spend grant (C/D-full/E) on an unvalidated pipeline; every source decontaminated against Pro/Verified.

---

## 12. Data appendix — complete linked list

> Decontaminate every training source against SWE-bench Pro's 41 repos + Verified before use. Eval-only sets must never enter training.

### 12.1 Executable environments + task corpora (collection / RL — Phases D, E)
The bottleneck is the Docker env fleet. **Nebius SWE-rebench ships 7,500 prebuilt Docker images on Docker Hub** — use these instead of building your own where possible.

- [nebius/SWE-rebench](https://huggingface.co/datasets/nebius/SWE-rebench) — 21k+ Python tasks, rolling, decontaminated, CC-BY-4.0; **7,500 prebuilt Docker images**.
- [nebius/SWE-rebench-V2](https://huggingface.co/datasets/nebius/SWE-rebench-V2) — **32,079 tasks across ~20 languages incl. Python, Java, Kotlin, Rust, C, C++, Go, TS, Scala, Swift**. This is the multilingual agentic backbone (and the Kotlin agentic data I'd earlier said didn't exist).
- [nebius/SWE-bench-extra](https://huggingface.co/datasets/nebius/SWE-bench-extra) — 6,415 Python instances (superseded by SWE-rebench, still useful).
- [SWE-Gym/SWE-Gym](https://huggingface.co/datasets/SWE-Gym/SWE-Gym) — 2,438 Python, exec envs + tests.
- [R2E-Gym (org)](https://huggingface.co/R2E-Gym) → `R2E-Gym/R2E-Gym`, `R2E-Gym/R2E-Gym-Lite`, `R2E-Gym/R2E-Gym-Subset` (Subset is non-overlapping with SWE-bench; best for RL).
- [SWE-bench/SWE-smith](https://huggingface.co/datasets/SWE-bench/SWE-smith) — 50k synthetic tasks across 128 repos.
- [ByteDance-Seed/Multi-SWE-RL](https://huggingface.co/datasets/ByteDance-Seed/Multi-SWE-RL) — 4,723 multilingual RL instances.
- [ByteDance-Seed/Multi-SWE-bench](https://huggingface.co/datasets/ByteDance-Seed/Multi-SWE-bench) — 1,632 instances, 7 languages, Docker envs.
- [Daoguang/Multi-SWE-bench](https://huggingface.co/datasets/Daoguang/Multi-SWE-bench) — SWE-bench-java + Docker harness.

### 12.2 Ready-made agentic trajectories (drop-in SFT — Phase D)
- [nebius/swe-agent-trajectories](https://huggingface.co/datasets/nebius/swe-agent-trajectories) — **80,036 trajectories** (from SWE-bench-extra). Largest ready set.
- [nebius/SWE-rebench-openhands-trajectories](https://huggingface.co/datasets/nebius/SWE-rebench-openhands-trajectories) — multi-turn (Qwen3-Coder-480B + OpenHands), with tool_calls.
- [nvidia/SWE-Hero-openhands-trajectories](https://huggingface.co/datasets/nvidia/SWE-Hero-openhands-trajectories) — 34k, permissive (MIT/Apache/BSD), commercial-OK.
- [SWE-bench/SWE-smith-trajectories](https://huggingface.co/datasets/SWE-bench/SWE-smith-trajectories) — SWE-smith trajectories.
- [ByteDance-Seed/Multi-SWE-bench_trajs](https://huggingface.co/datasets/ByteDance-Seed/Multi-SWE-bench_trajs) — multilingual leaderboard trajectories.

### 12.3 Tool-use / function-calling (Phase D format layer)
- [Salesforce/xlam-function-calling-60k](https://huggingface.co/datasets/Salesforce/xlam-function-calling-60k) — gated, research-use flag.
- [Salesforce/APIGen-MT-5k](https://huggingface.co/datasets/Salesforce/APIGen-MT-5k) — multi-turn FC.
- [Team-ACE/ToolACE](https://huggingface.co/datasets/Team-ACE/ToolACE)
- [NousResearch/hermes-function-calling-v1](https://huggingface.co/datasets/NousResearch/hermes-function-calling-v1)
- [glaiveai/glaive-function-calling-v2](https://huggingface.co/datasets/glaiveai/glaive-function-calling-v2)

### 12.4 Code-gen + reasoning SFT (Phase D seasoning)
- [nvidia/OpenCodeReasoning](https://huggingface.co/datasets/nvidia/OpenCodeReasoning) + [nvidia/OpenCodeReasoning-2](https://huggingface.co/datasets/nvidia/OpenCodeReasoning-2)
- [nvidia/OpenCodeInstruct](https://huggingface.co/datasets/nvidia/OpenCodeInstruct) — 5M, with tests + exec feedback.
- [m-a-p/CodeFeedback-Filtered-Instruction](https://huggingface.co/datasets/m-a-p/CodeFeedback-Filtered-Instruction) — 156k, complexity-filtered.
- [m-a-p/Code-Feedback](https://huggingface.co/datasets/m-a-p/Code-Feedback) — multi-turn w/ execution feedback.
- [ise-uiuc/Magicoder-OSS-Instruct-75K](https://huggingface.co/datasets/ise-uiuc/Magicoder-OSS-Instruct-75K)

### 12.5 Continued-pretraining corpus (Phase C)
- [bigcode/the-stack-v2](https://huggingface.co/datasets/bigcode/the-stack-v2) (+ `bigcode/the-stack-v2-dedup`) — filter to high-quality + target languages; FIM objective.
- [OpenCoder-LLM (org)](https://huggingface.co/OpenCoder-LLM) → `opc-sft-stage1`, `opc-sft-stage2`, plus annealing/pretraining corpora (`opc-annealing-corpus`, `opc-fineweb-code-corpus` — **confirm exact IDs on the org page**).

### 12.6 Multilingual code (Java / Kotlin / Rust / C++ — Phases C/D)
- **Agentic, all four langs:** use **SWE-rebench-V2** (§12.1) — it covers Java, Kotlin, Rust, C++ as real issue-resolution tasks.
- [Multilingual-Multimodal-NLP/McEval-Instruct](https://huggingface.co/datasets/Multilingual-Multimodal-NLP/McEval-Instruct) — 40 langs SFT; CC-BY-SA-4.0 (copyleft).
- [bigcode/commitpack](https://huggingface.co/datasets/bigcode/commitpack) + [bigcode/commitpackft](https://huggingface.co/datasets/bigcode/commitpackft) — commit-based, multilingual.
- Kotlin: [JetBrains/KStack](https://huggingface.co/datasets/JetBrains/KStack), [JetBrains/KStack-clean](https://huggingface.co/datasets/JetBrains/KStack-clean), [JetBrains/KExercises](https://huggingface.co/datasets/JetBrains/KExercises).
- C++/Java heavy (competitive): [deepmind/code_contests](https://huggingface.co/datasets/deepmind/code_contests), [BAAI/TACO](https://huggingface.co/datasets/BAAI/TACO), [codeparrot/apps](https://huggingface.co/datasets/codeparrot/apps).
- Rust: [Neloy262/rust_instruction_dataset](https://huggingface.co/datasets/Neloy262/rust_instruction_dataset).

### 12.7 Eval-only (NEVER train on these)
- **SWE-bench Pro** (primary) — Scale AI; public set via the [Scale SEAL leaderboard](https://labs.scale.com/leaderboard/swe_bench_pro_public) (get the public 731-instance set from Scale; **confirm HF mirror if any**).
- [SWE-bench/SWE-bench_Multilingual](https://huggingface.co/datasets/SWE-bench/SWE-bench_Multilingual) — 300 tasks, 9 langs.
- SWE-bench Verified / Lite — under the [SWE-bench org](https://huggingface.co/SWE-bench) (`SWE-bench/SWE-bench_Verified`, `SWE-bench/SWE-bench_Lite`).
- [ByteDance-Seed/Multi-SWE-bench_mini](https://huggingface.co/datasets/ByteDance-Seed/Multi-SWE-bench_mini) + [Multi-SWE-bench-flash](https://huggingface.co/datasets/ByteDance-Seed/Multi-SWE-bench-flash).
- [nebius/SWE-rebench-leaderboard](https://huggingface.co/datasets/nebius/SWE-rebench-leaderboard) — contamination-free rolling eval.
- [JetBrains/Kotlin_HumanEval](https://huggingface.co/datasets/JetBrains/Kotlin_HumanEval).
- LiveCodeBench, Terminal-Bench 2.0 — benchmark harnesses (not single HF datasets; **confirm current repo/harness**).

### 12.8 To confirm before use
- Exact OpenCoder pretraining/annealing corpus IDs (§12.5).
- Whether SWE-bench Pro has an official HF mirror or must be pulled from Scale (§12.7).
- Current LiveCodeBench dataset repo + Terminal-Bench 2.0 harness (§12.7).

---

## 13. Risks, kill criteria & resourcing

**Honest ceiling.** Gemma 4 31B is the weaker coding base; **parity** with Qwen3.6-27B on a fair, same-stack comparison is the realistic target, with surpassing it riding mostly on out-engineering the scaffold and RL. Hold this expectation; don't let sunk cost inflate it.

**Kill criteria (stop / redirect before spending grant):**
- **A+B show no lift** over Qwen3.6-27B run under your identical stack → the ceiling is the base model; do not fund C/D-full/E. Reassess (better scaffold, or accept "best Gemma-based local agent" as the win).
- **QLoRA SFT (D) doesn't lift Verified/Lite** → the data/trajectories are the problem; fix collection before any full-SFT or RL spend.
- **RL (E) shows high solve-none rate** across GRPO attempts → difficulty curriculum is wrong; re-filter tasks (too-easy/too-hard) before continuing.
- **NVFP4 PTQ+QAT can't get within tolerance of BF16 on Pro** → ship the FP8 fallback for capability; keep NVFP4 only where the speed is worth the drop.

**Resourcing (the real gating constraint).** The grant phases — continued pretraining (C), full SFT (D), GRPO RL (E), plus the Docker env fleet — are the dominant, currently-undefined cost. Size the grant before committing to C/E. **Continued pretraining (C) is the first to cut** if budget is tight; A+B+QLoRA-D give signal without it. The env fleet cost is largely avoidable up front by reusing nebius/SWE-rebench's 7,500 prebuilt Docker images (§12.1).

**Hard integrity gate.** Any overlap between training data and SWE-bench Pro's 41 repos (or Verified) voids the headline number. Decontaminate, keep the dropped-count manifest, and report it alongside every result.

---

## 14. Key references & external validation

- **Kimi K2: Open Agentic Intelligence** — arXiv 2507.20534. The substantive published recipe: large-scale agentic data-synthesis pipeline (real MCP tools → synthetic tools → agents/tasks/rollouts → rubric-judged trajectories) + a joint RL stage with a **Verifiable Rewards Gym** (rule-based 1/0 across Math/STEM/Logic/IF/Coding/etc.) plus rubric self-critique for non-verifiable tasks. Paper-reported (non-thinking): **65.8 SWE-bench Verified, 47.3 SWE-bench Multilingual, 53.7 LiveCodeBench v6**.
- **Kimi K2.5: Visual Agentic Intelligence** — arXiv 2602.02276. Adds a unified agentic-RL environment and parallel-agent RL (agent swarm).
- **Note:** Kimi **K2.7-Code** (the model that prompted these additions) has **no paper yet** — too new (June 2026); its benchmarks are in-house/vendor and it has not been run on SWE-bench. Treat its numbers as vendor-reported.
- **Datasets:** Moonshot released the K2 **weights** and the **Muon optimizer code**, but **not** the agentic training datasets. The transferable artifact is the *documented pipeline*, not data you can download.
- **Why this matters here:** K2's published architecture — synthesize/collect agentic trajectories, train an SFT cold-start, then RL against a verifiable-reward gym of executable environments — is exactly Phases D→E of this plan. A frontier agentic-coding model was built this way; your gym stack (R2E-Gym / SWE-Gym / Multi-SWE-RL + GRPO on test-pass reward) is the same recipe at a feasible scale.
