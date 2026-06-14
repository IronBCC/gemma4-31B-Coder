# Phase F — Distillation augmentation (optional, bounded)

SFT on frontier-teacher reasoning traces can help but is bounded by base capacity and can't beat the
teacher. If used, reconstruct compressed reasoning (Trace-Inversion) into full CoT to avoid "reasoning
fractures." Treat CoT-distillation as a supplement to — never a replacement for — execution-verified
trajectories + RL.

## Gate
Only ship if it beats SFT-only on eval.
