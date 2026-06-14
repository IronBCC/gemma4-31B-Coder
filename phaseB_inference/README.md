# Phase B — Inference-time scaling (no weight change)

- **Best-of-n + hybrid verifiers** — sample N trajectories; rank with hybrid verifier (execution-based
  tests + execution-free trained ORM). R2E-Gym Hybrid TTS reached 51% pass@1 on Verified. Train the ORM
  on Phase-D collected trajectories.
- **Speculative decoding (latency)** — enable Gemma 4 native MTP draft heads in vLLM (~1.5–2.5×) so
  best-of-n stays affordable. Full speculator-refresh treatment lives in Phase G.

## Gate
Best-of-n + verifier lift over Phase-A pass@1; throughput acceptable.

## TODO
- [ ] Best-of-n sampler over the Phase-A harness.
- [ ] Execution verifier (run tests on candidate patches).
- [ ] Execution-free ORM (trained on Phase-D traces) + reranker.
- [ ] Enable native MTP spec decoding; measure acceptance.
