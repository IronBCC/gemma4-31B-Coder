# Phase B — Inference-time scaling (no weight change)

best-of-n over the Phase-A agent loop + a **hybrid verifier** (execution tests +
execution-free ORM). R2E-Gym Hybrid TTS reached 51% pass@1 on Verified this way.
Plus native MTP speculative decoding for latency (full refresh treatment in Phase G).

## Implemented
- `inference/verifier.py` — `Candidate`, hybrid `rerank`/`select_best`; execution signal dominates,
  ORM breaks ties. `HeuristicORM` is a non-trained fallback (penalizes empty/shotgun patches, prefers
  shorter solves) so the reranker runs end-to-end before a real ORM exists.
- `inference/best_of_n.py` — samples N trajectories (parallel), reranks; `make_agent_solve_fn` wires in
  the real Phase-A `solve_task` (temperature 0.7 so samples differ). Decoupled via a `solve_fn` callable.

## TODO
- [ ] Train the execution-free **ORM** on Phase-D collected trajectories; drop into `rerank(orm=...)`.
- [ ] Enable native MTP spec decoding in vLLM; measure acceptance (see `phaseG_deploy/`).

## Run tests
```bash
cd phaseB_inference && python -m pytest tests/ -q   # 4 tests, no GPU/network
```

## Gate
Best-of-n + verifier lift over Phase-A pass@1; throughput acceptable.
