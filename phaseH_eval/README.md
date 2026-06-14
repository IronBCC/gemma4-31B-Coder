# Phase H — Benchmark (execute the fair-eval protocol)

See `fair_eval_protocol.md` for the gating rules (§2). Execution:
1. Deploy candidate on vLLM (BF16 for capability, NVFP4 for deployment number).
2. Run mini-SWE-agent over Pro public + Verified/Lite + Terminal-Bench/LiveCodeBench + multilingual, fixed budget.
3. Apply patches in official Docker harness; run hidden tests.
4. Score vs (a) your-harness Qwen3.6-27B baseline, (b) Phase-0 Gemma baseline, (c) prior checkpoint.
5. Report pass@1 and best-of-n + reranked.
