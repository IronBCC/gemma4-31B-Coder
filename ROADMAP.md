# Roadmap — critical path first

> Rule (PLAN §11): scaffold + inference-time (A/B) come first and may deliver most of the win cheaply.
> Never spend grant (C / D-full / E) on an unvalidated pipeline.

## Phase 0 — Baseline + fair-eval setup  *(box, do first)*
- [ ] Box ready: `scripts/setup_box.sh` (vLLM, mini-SWE-agent, datasets, Docker).
- [ ] Serve Gemma 4 31B (BF16) on vLLM with locked Gemma-4 chat/tool template + EOS.
- [ ] Serve `Qwen/Qwen3.6-27B` under the **identical** stack (this is the real baseline to beat).
- [ ] `phaseH_eval/run_baseline.sh` over SWE-bench Lite → Verified, fixed turn/cost budget.
- [ ] **Gate:** reproducible Gemma + Qwen numbers under the same harness.

## Phase A — Scaffold  *(box, centerpiece)*
- [ ] Fine-grained edit tool (AST-aware search/replace).
- [ ] Repo localization/retrieval pass.
- [ ] Test-execution feedback loop + crash/retry recovery.
- [ ] Multi-turn budget; Gemma-4 tool-call parser pinned.
- [ ] **Gate:** measurable lift over Phase-0 Gemma baseline on Lite/Verified.

## Phase B — Inference-time scaling  *(box)*
- [ ] Best-of-n sampler.
- [ ] Hybrid verifier (execution tests + execution-free ORM reranker).
- [ ] Native MTP speculative decoding enabled; acceptance measured.
- [ ] **Gate:** best-of-n + verifier lift; throughput acceptable.

## ►► DECISION GATE ◄◄
If **A+B show no lift** over same-stack Qwen3.6-27B → the ceiling is the base model.
**Do not fund C / D-full / E.** Accept "best Gemma-based local agent" as the win, or rethink scaffold.
If A+B lift → proceed to grant phases (see `docs/PLAN.md` §5–9 and per-phase READMEs).

## Grant phases (only after the gate)
- [ ] C — continued pretraining (`phaseC_pretrain/`)
- [ ] D — agentic SFT: QLoRA validate → full SFT (`phaseD_sft/`)
- [ ] E — GRPO RL with execution reward (`phaseE_rl/`)
- [ ] G — NVFP4 + QAT, EAGLE-3 refresh, serving (`phaseG_deploy/`)
- [ ] H — full benchmark + fair comparison (`phaseH_eval/`)

## Always-on integrity
- [ ] Decontaminate every training source vs Pro (41 repos) + Verified; keep dropped-count manifest.
- [ ] Capability claim BF16-vs-BF16; NVFP4 reported separately with quant gap.
