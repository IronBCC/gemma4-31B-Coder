# Phase A — Agent scaffold engineering (centerpiece)

Highest-ROI, base-agnostic work. Proof point: an engineered stack took Qwen3.6-27B-FP8 to
**90.0% SWE-bench Verified** with no fine-tune/distill — the scaffold drove it.

## Build
- **Fine-grained editing tools** — structured search/replace + AST-aware edits (the ~2.6-pt edge), not blind diffs.
- **Localization pass** — retrieval over repo → candidate files/functions before editing.
- **Test-execution feedback loop** — run tests, feed failures back, iterate (most resolves come from here).
- **Multi-turn budget + crash/retry recovery** (the 90% run used crash-retry).
- **Protocol/format hardening** — lock system prompt + tool schema to Gemma 4's format; root-cause the
  vLLM tool-call parser bug.

## Gate
Measurable lift over the Phase-0 baseline on SWE-bench Lite/Verified.

## TODO
- [ ] Adopt mini-SWE-agent as the harness (same one used for collection + eval).
- [ ] Implement AST-aware edit tool.
- [ ] Implement repo localization/retrieval.
- [ ] Wire test-exec feedback + crash/retry.
- [ ] Pin Gemma-4 tool-call parser + system prompt.

## Implemented (this scaffold)
- `scaffold/edit_tool.py` — `search_replace` (unique-match guard) + `ast_edit` (Python; syntax-validated). Exposes `TOOL_SCHEMAS`.
- `scaffold/localizer.py` — dependency-free BM25 + definition-site boost.
- `scaffold/test_loop.py` — subprocess runner (local or `docker exec`), timeout cap, pytest/unittest failure parsing, `.feedback()` for the model.
- `scaffold/agent.py` — multi-turn OpenAI-compatible loop (vLLM), tool dispatch, crash/retry, `reasoning_content` preserved across turns, returns a `Trajectory`.

Still TODO: tree-sitter backend for `ast_edit` (java/kotlin/rust/cpp), embedding/LLM rerank in the localizer, and pinning the Gemma-4 tool-call parser against the live server.

### Run tests
```bash
cd phaseA_scaffold && python -m pytest tests/ -q   # 7 tests, no GPU/network needed
```

### Quick agent smoke (needs a running vLLM endpoint)
```python
from scaffold.agent import solve_task
traj = solve_task(repo_dir="/path/to/repo",
                  problem_statement="...",
                  test_cmd="python -m pytest -x tests/test_foo.py",
                  base_url="http://localhost:8000/v1",
                  model="google/gemma-4-31B-it")
print(traj.resolved, traj.n_turns)
```
