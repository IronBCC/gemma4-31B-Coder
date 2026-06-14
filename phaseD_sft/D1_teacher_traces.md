# D.1 — Execution-verified teacher-trace pipeline (key SFT source)

Frontier-grade reasoning, execution-gated, license-clean.

## Teacher selection (gating discipline)
- License must permit training on outputs, and **self-host** (collapses hosted-API ToS layer).
- Clean self-hosted choices: **Qwen3.6-27B / Qwen3-Coder** (Apache-2.0), **DeepSeek** (MIT),
  **gpt-oss** (Apache-2.0). Kimi K2.6 / K2.7-Code (Modified-MIT) only after reading output clauses.
- Avoid closed-API outputs + scraped trace dumps.
- Record `teacher_model` + `teacher_license` on every trace.

## Generation loop
- Drive teacher with the **deploy harness** (mini-SWE-agent) over executable gyms inside Docker.
- Record full trajectory; preserve `reasoning_content` across turns.
- Best-of-N per task (raise N on hard tasks).
- Hard tasks: gold-patch-assisted (+~38% usable); tag `synthetic_second_attempt=true`, keep minority aux.

## Execution gate
Keep only if final patch passes tests (Fail2Pass + Pass2Pass). Drop on test fail / format-invalid tool
calls / truncation / pathological length.

## Dedup + decontamination
MinHash/embedding dedup; drop tasks whose repo ∈ {Pro 41 ∪ Verified}; 13-gram problem-statement overlap
via open-r1 `decontaminate.py`. Keep dropped-count manifest.

## Schema
```json
{ "...trajectory + instance fields...": "",
  "teacher_model": "Qwen/Qwen3.6-27B", "teacher_license": "Apache-2.0",
  "generation_harness": "mini-swe-agent",
  "sampled_k": 8, "passed": true, "synthetic_second_attempt": false,
  "n_turns": 14, "n_tokens": 9123 }
```
Serialize into Gemma 4 chat + tool format at training time (same template/EOS as serving).
