# Fair-eval protocol (define first; gate everything)

- **Run the target yourself.** Stand up `Qwen/Qwen3.6-27B` under *our* scaffold/harness/precision/budget.
  That — not its blog number — is the baseline to beat.
- **Capability vs deployment split.** "Beat Qwen" claim in **BF16-vs-BF16**. NVFP4 is the deployment
  artifact, measured separately, quant gap reported. (Qwen ran FP8; 4-bit-vs-FP8 would be unfair to us.)
- **Eval suite:**
  - Primary: SWE-bench **Pro** public (731), standardized mini-SWE-agent harness.
  - Dev loop: SWE-bench **Lite → Verified**.
  - Breadth: **Terminal-Bench 2.0**, **LiveCodeBench**, multilingual (SWE-bench Multilingual / Multi-SWE-bench).
- **Report:** resolve rate (pass@1 + best-of-n reranked), empty-patch rate, tool-call validity,
  per-repo/per-language tables, latency (tok/s, NVFP4 + spec decoding), decontamination manifest.
- **Integrity gate:** every training source diffed against Pro's 41 repos + Verified; n-gram overlap;
  dropped-count manifest.
