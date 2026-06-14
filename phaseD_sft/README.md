# Phase D — Agentic SFT (rejection sampling; RL cold-start)

## Trajectory data
- **Primary:** your own execution-verified teacher traces (see `D1_teacher_traces.md`) — keep only
  test-passing trajectories (Fail2Pass + Pass2Pass).
- **Preserve `reasoning_content` across turns** (K2.7-Code forced preserve-thinking).
- **Pure tool-use:** replicate Kimi K2's 3-stage agentic synthesis (arXiv 2507.20534).
- **Ready sets:** `nvidia/SWE-Hero-openhands-trajectories` (34k, commercial-OK),
  `SWE-bench/SWE-smith-trajectories`, `ByteDance-Seed/Multi-SWE-bench_trajs`.
- **Format seasoning:** `Salesforce/xlam-function-calling-60k`, `APIGen-MT-5k`, `ToolACE`,
  `hermes-function-calling-v1`.

## Curriculum (3-stage)
1. **Format Inception** (<4,096 tok) — stabilize `<think>` boundaries + tool-call format.
2. **Complexity Expansion** (4,096–8,192) — harder coding + multi-turn; reason→action→feedback.
3. **Long-Context SFT** (→32K) — long/multi-turn trajectories with short-sample replay.

## Execution
QLoRA on the box to validate on Lite/Verified (DDP ×2), then **full SFT on grant**. This checkpoint is
the RL cold-start.

## Gemma-4 landmines (mandatory)
- KV-shared layer / `use_cache` bug → Unsloth patched loader (gradient checkpointing forces
  `use_cache=False`).
- Use the `gemma-4-thinking` template; train + serve identical template + EOS.
- Keep ≥75% reasoning-style examples to preserve thinking behavior.

## Hyperparameters (QLoRA validation)
`r=32` (64 if headroom), `alpha=32`, dropout 0, all proj modules, `lr=2e-4` cosine 3–5% warmup,
1–3 epochs (overfits fast), `max_seq_len=8192` (raise in Stage 3).

## Gate
QLoRA lift on Verified/Lite before any full-SFT spend.
