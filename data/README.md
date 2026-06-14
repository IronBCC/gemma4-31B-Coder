# Data

Raw datasets/corpora are **git-ignored** — see `docs/PLAN.md` §12 for the full linked source list.
Track only small **manifests** here (decontamination dropped-counts, source licenses, dataset versions)
in `manifests/`.

## Key sources (see PLAN §12 for links + licenses)
- **Executable envs / RL:** nebius/SWE-rebench (7,500 prebuilt Docker images), SWE-rebench-V2 (32k, ~20 langs),
  R2E-Gym (Subset best for RL), SWE-Gym, ByteDance Multi-SWE-RL/bench.
- **Ready trajectories (SFT):** nebius/swe-agent-trajectories (80k), nvidia/SWE-Hero (34k, commercial-OK).
- **Tool-use:** Salesforce xLAM-60k, APIGen-MT-5k, ToolACE, hermes-function-calling-v1.
- **Continued-pretrain:** bigcode/the-stack-v2 (+dedup), OpenCoder annealing mix.
- **Eval-only (NEVER train):** SWE-bench Pro / Verified / Lite, SWE-bench Multilingual, LiveCodeBench, Terminal-Bench 2.0.

## To confirm before use (PLAN §12.8)
- Exact OpenCoder pretraining/annealing corpus IDs.
- Whether SWE-bench Pro has an official HF mirror or must be pulled from Scale.
- Current LiveCodeBench repo + Terminal-Bench 2.0 harness.
