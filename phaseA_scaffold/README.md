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
