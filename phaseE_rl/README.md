# Phase E — GRPO RL with execution reward (core capability lever)

The only teacher-independent way to push past a strong baseline (DeepSWE trained a 32B purely this way).

- **From** the SFT checkpoint, on-policy **GRPO**.
- **Reward:** sparse outcome — 1 iff patch passes selected Fail2Pass + Pass2Pass within a time cap (~5 min).
- **Token-efficiency term:** lightly penalize tokens-to-solve (K2.7-Code cut thinking tokens ~30%).
  **Sweep the penalty weight** — over-weighting suppresses reasoning and hurts resolve rate.
- **Reward reference:** Kimi K2 verifiable rewards (rule-based 1/0) + rubric self-critique for
  non-verifiable terms (arXiv 2507.20534); K2.5 unified agentic-RL env (arXiv 2602.02276).
- **Data/envs:** `R2E-Gym` (best for RL) + Multi-SWE-RL (4,723, cross-lang). **Pre-filter**
  too-easy/too-hard tasks via base rollouts.
- **Config:** ~64k max ctx, ~100 max env steps, async rollout.
- Optionally cycle SFT (on new RL successes) ↔ RL.

## Gate
SFT stable; envs scaled; difficulty-filtered (watch solve-none rate).
