# gemma4-31B-Coder

Maximal build pipeline to turn **`google/gemma-4-31B-it`** (dense 30.7B, 256K ctx, thinking mode,
native function-calling, native MTP draft heads) into the strongest coding/agentic model it can be.

**Target to beat:** Qwen3.6-27B, run *under our own identical harness* (not its self-reported number).
**Deploy:** vLLM, NVFP4 (W4A4) on RTX 6000 Pro Blackwell.

> Full spec: [`docs/PLAN.md`](docs/PLAN.md).

## Critical path
Do **Phase A + B on the box first** (scaffold + best-of-n/verifier), measured against a same-stack
Qwen3.6-27B baseline. This is the cheap, no-grant decision gate: if A+B don't lift over that baseline,
the ceiling is the base model and the grant phases (C / D-full / E) are not worth funding.

## Repository layout
| Dir | Phase | Purpose | Where |
|---|---|---|---|
| `phaseA_scaffold/` | A | Agent scaffold: fine-grained edit tools, localization, test-exec loop, crash/retry | Box |
| `phaseB_inference/` | B | Inference-time scaling: best-of-n + hybrid verifiers, speculative decoding (native MTP) | Box |
| `phaseC_pretrain/` | C | Continued pretraining (raise the coding floor) | Grant |
| `phaseD_sft/` | D | Agentic SFT (rejection-sampling), execution-verified teacher traces | Box (QLoRA) → Grant (full) |
| `phaseE_rl/` | E | GRPO RL with execution reward | Grant |
| `phaseF_distill/` | F | Distillation augmentation (optional, bounded) | Box/Grant |
| `phaseG_deploy/` | G | NVFP4 + QAT, EAGLE-3 speculator refresh, serving configs | Box/Grant |
| `phaseH_eval/` | H | Fair-eval protocol + benchmark execution | Box |
| `data/` | — | Dataset manifests + decontamination outputs (raw data git-ignored) | — |
| `configs/` | — | Shared config (model, harness, vLLM) | — |
| `scripts/` | — | Setup / utility scripts | — |

## Eval integrity (hard gate)
Every training source is decontaminated against **SWE-bench Pro (41 repos) + Verified** before use;
keep the dropped-count manifest in `data/manifests/`. Eval-only sets never enter training.
Capability ("beat Qwen") claim is made **BF16-vs-BF16**; NVFP4 is the deployment artifact, measured
separately with the quant gap reported.

## Status
Scaffold initialized. Start in `phaseA_scaffold/` and `phaseH_eval/` (baseline + fair-eval setup).
