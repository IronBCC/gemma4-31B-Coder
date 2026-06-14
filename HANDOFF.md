# HANDOFF — gemma4-31B-Coder

_Last updated: 2026-06-14 · HEAD: `9b6edd6` · remote: https://github.com/IronBCC/gemma4-31B-Coder_

Pick up here on the Blackwell (RTX 6000 Pro) box. The full strategy is in
[`docs/PLAN.md`](docs/PLAN.md); the sequenced checklist is [`ROADMAP.md`](ROADMAP.md).

---

## 1. Get the code on the box
```bash
git clone https://github.com/IronBCC/gemma4-31B-Coder.git
cd gemma4-31B-Coder
bash scripts/setup_box.sh        # venv + deps + GPU/Docker sanity
```
> If you instead rsync the local folder, first delete `.git/*.lock` — the desktop
> sandbox can leave stale zero-byte locks it lacks permission to remove.

## 2. What's already built (commit 9b6edd6)
- **Repo scaffold** — phase dirs A–H, `.gitignore` (datasets/checkpoints ignored), `configs/model.yaml`.
- **Phase A — agent scaffold** (`phaseA_scaffold/scaffold/`), tested:
  - `edit_tool.py` — unique-match `search_replace`; `ast_edit` (Python via stdlib, syntax-validated).
  - `treesitter_edit.py` — `ast_edit` for Java/Kotlin/Rust/C++/Go/TS/JS (optional grammars, graceful fallback).
  - `localizer.py` — dependency-free BM25 + definition-site boost.
  - `test_loop.py` — subprocess/`docker exec` runner, timeout cap, pytest/unittest parsing, `.feedback()`.
  - `agent.py` — multi-turn OpenAI-compatible vLLM loop; crash/retry; `reasoning_content` preserved; returns `Trajectory`.
- **Phase B — inference-time** (`phaseB_inference/inference/`), tested:
  - `verifier.py` — hybrid rerank (execution dominates, ORM breaks ties) + `HeuristicORM` fallback.
  - `best_of_n.py` — parallel N-sampler; `make_agent_solve_fn` wires in the real Phase-A agent.
- **Eval** — `phaseH_eval/run_baseline.sh` (same-stack Gemma vs Qwen3.6-27B), `fair_eval_protocol.md`.
- **Tests** — 15 passing, no GPU/network needed:
  ```bash
  (cd phaseA_scaffold && python -m pytest tests/ -q)   # 11
  (cd phaseB_inference && python -m pytest tests/ -q)   # 4
  ```

## 3. Next steps (in order — see ROADMAP.md)
1. **Phase 0 baseline.** Serve Gemma 4 31B (BF16) on vLLM with the locked `gemma-4-thinking`
   template + EOS; serve `Qwen/Qwen3.6-27B` under the *identical* stack. Run
   `phaseH_eval/run_baseline.sh` on SWE-bench Lite → Verified. **Gate:** reproducible numbers, same harness.
2. **Phase A tune.** Point `agent.solve_task` at the live endpoint; iterate the scaffold on Lite/Verified.
   Pin the Gemma-4 `--tool-call-parser` (root-cause the vLLM parser bug here).
3. **Phase B.** Run `best_of_n` with `make_agent_solve_fn`; confirm best-of-n + verifier lift.
4. **DECISION GATE:** if A+B don't beat same-stack Qwen → ceiling is the base model; do NOT fund C/D-full/E.
5. If lift → Phase D trajectory collection (also produces ORM training data), then E (GRPO), then G (deploy).

## 4. Open TODOs / honest caveats
- Train the execution-free **ORM** on Phase-D traces; drop into `verifier.rerank(orm=...)`.
- Localizer has **no embedding/LLM rerank** yet (BM25 only).
- vLLM flags, CLI flags in `run_baseline.sh`, and some dataset IDs are marked **"confirm against your version"**.
- Gemma-4 landmines (PLAN §6): Unsloth patched loader for the KV/`use_cache` bug; identical template+EOS train/serve; keep ≥75% reasoning-style SFT examples.
- **Integrity:** decontaminate every training source vs SWE-bench Pro (41 repos) + Verified; keep the dropped-count manifest (`scripts/decontaminate.md`).

## 5. Key paths
| What | Path |
|---|---|
| Strategy | `docs/PLAN.md` |
| Checklist | `ROADMAP.md` |
| Agent loop | `phaseA_scaffold/scaffold/agent.py` |
| Reranker | `phaseB_inference/inference/verifier.py` |
| Baseline runner | `phaseH_eval/run_baseline.sh` |
| Box setup | `scripts/setup_box.sh` |
| Model/precision config | `configs/model.yaml` |

## 6. Commit history so far
```
  9b6edd6 Add tree-sitter ast_edit backend + Phase B best-of-n/hybrid verifier
  f677e1f Implement Phase A scaffold components + tests
  d28fd8c Critical-path starter kit (Phase 0/A/B)
  f0c59f4 Initial scaffold: phase-based pipeline for gemma4-31B-Coder
```
