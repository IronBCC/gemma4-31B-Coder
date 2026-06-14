# Phase C — Continued pretraining (raise the coding floor)  [GRANT]

Domain-adaptive mid-training to lift Gemma's coding *capacity* before any SFT.

- **Corpus:** `bigcode/the-stack-v2` (high-quality + Python/Java/Kotlin/Rust/C++) + OpenCoder-style
  annealing mix + FIM.
- **Objective:** next-token + FIM; replay general/reasoning data to protect Gemma's reasoning strength.
- **Decontaminate** against Pro/Verified repos first.

First phase to cut if budget is tight (A+B+QLoRA-D give signal without it), but the most direct way to
close the raw-coding gap to Qwen.

## Gate
Coding-floor lift on held-out code evals.
