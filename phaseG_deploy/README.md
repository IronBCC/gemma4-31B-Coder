# Phase G — Deployment: NVFP4 + QAT, speculator refresh, serving config

Ship 4-bit with minimal capability loss and max speed.
**Order:** finalize weights (post-RL) → quantize (G.1) → refresh speculator (G.2) → serving (G.3).

## G.1 — NVFP4 + QAT
- PTQ → QAT. PTQ to NVFP4 (LLM Compressor `scheme="NVFP4"` or ModelOpt `NVFP4_DEFAULT_CFG`), then
  QAT/QAD via NVIDIA Model Optimizer. Export Unified HF → vLLM.
- Calibrate on your own SWE data (512–1024 domain samples).
- Measure quant gap vs BF16; keep an **FP8 fallback** for the capability claim.
- RTX 6000 Pro (SM120) supports NVFP4 W4A4.
- Re-verify tool-call format after every re-quant.

## G.2 — Speculator refresh (EAGLE-3)
Native MTP heads drift after C/D/E. QLoRA-only → native MTP probably fine (verify acceptance).
Full SFT / continued-pretrain / RL → retrain an **EAGLE-3** head on the final model
(self-distilled on the target's own outputs over your SWE data). See `eagle3_refresh.sh`.
- Tooling: SpecForge / EAGLE-3; NVIDIA Model Optimizer. On Blackwell stack **P-EAGLE** (~1.05–1.69×).
- Validate acceptance against the **deployed NVFP4** checkpoint, not just BF16.
- Context cap: vLLM spec models historically ~2048 — confirm/extend for long trajectories.

## G.3 — Serving config (choose per workload)
Two profiles in `serving/` (`throughput.sh`, `latency.sh`). Spec decoding and high-concurrency batching
partially trade off — throughput profile usually runs *no* spec decoding.

## Gate
Quant gap within BF16 tolerance; speculator acceptance high on the deployed NVFP4 checkpoint.
