"""Phase B — inference-time scaling (no weight change).

best-of-n sampling over the Phase-A agent loop, plus a hybrid verifier that
combines an execution-based signal (run the tests) with an execution-free ORM
(an outcome reward model trained on Phase-D trajectories). R2E-Gym's Hybrid TTS
reached 51% pass@1 on SWE-bench Verified this way.
"""
