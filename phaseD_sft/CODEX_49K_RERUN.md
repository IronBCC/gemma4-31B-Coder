# CODEX RERUN — v6 49k SWE-Lite smoke with clamp+thinking fix

The prior 30-case SWE-Lite smoke failed 0/30, but the verified root cause is a **harness
clamp**, not the model. Two fixes are applied + tested + synced to the host. Re-run the smoke.

## What was wrong (verified)
- hard30 PASSED (+3), so the model is fine. The SWE-Lite 0/30 was driven by:
  1. **`model_clamps.py DEFAULT_MAX_TOKENS = 512`** — 29% of generations hit exactly 512 tokens;
     truncated turns lose their `<|tool_call>` markup → gemma4 tool parser returns no tool call →
     harness "Tool call error: No tool calls found" → 63.19% format-recovery loop → turns burned,
     edits (2/30) rarely reached.
  2. **Thinking was OFF** — the gemma4 chat template defaults `enable_thinking=false`; the model
     ran terse tool-only turns (0 reasoning_content in every traj).
  3. **Docker scoring crashed** — `docker.errors.NotFound: 404 No such container` → `No instances to run`
     (SWE-bench docker-loopback gotcha), so even the 2 edits could not be scored.

## Fixes already applied (local repo + scp'd to host, 15/15 phaseH_eval tests green)
- `phaseH_eval/model_clamps.py`: `DEFAULT_MAX_TOKENS` 512 → **32768** (reasoning + tool call both fit).
- `phaseH_eval/vllm_direct_model.py::_query`: sends
  `extra_body={"chat_template_kwargs": {"enable_thinking": True}}` → **thinking ON**. Gemma-4 then
  renders `<|channel>thought …<channel|>` + `<|tool_call>…<tool_call|>`; the server's
  `--reasoning-parser gemma4` + `--tool-call-parser gemma4` split them and still return structured
  `tool_calls`. (Files already on host — do not re-edit; just use them.)

## Your task
1. **Fix docker scoring first** — before rerunning, verify docker health so scoring can run:
   `docker ps -a`, `docker system df`, check the loopback FS free space (see `disk_watchdog.sh` /
   the eval-docker-disk-watchdog note). Prune dead containers/images if the loopback FS is full.
   If scoring still 404s, the smoke is worthless — resolve this or report it as the blocker.
2. **Re-serve** the same adapter, same verified config: NVFP4 base, LoRA
   `adapters/swe_edit_v6_49k_s1/checkpoint-5`, `--enable-auto-tool-choice --tool-call-parser gemma4
   --reasoning-parser gemma4`, 262k ctx, fp8 KV, **distinct served-model-name vs LoRA module name**
   (name collision invalidates the eval), isolated GPU1 unit, ninja on PATH (`vllm_env/bin`), prod
   ports 8000/8101/8103/8104 green, GPU0/vllm.service off-limits.
3. **Re-run** the 30-case SWE-Lite smoke via `phaseH_eval/smoke_single.sh` (the fixed
   `vllm_direct_model.py` forces thinking; 32k clamp is in `model_clamps.py`).
4. **Report** with per-layer numbers so we can see the delta:
   - format-error rate (was 63.19% — expect a large drop),
   - non-empty `model_patch` count (gate ≥ 18/30),
   - trajectories reaching a source edit (was 2/30),
   - completion_tokens distribution (was 29% at the 512 ceiling — expect no ceiling pile-up now),
   - confirm `tool_calls` are returned structured (thinking-on did NOT swallow the tool call).

## Watch
- Thinking-on + 32k → longer generations: slower smoke, more KV per seq. Monitor GPU1 VRAM + host RAM
  (earlyoom off, ~45 GB train-free now). Kill by exact PID only; never `pkill -f`; never raise cgroup caps.
- If format errors drop sharply and patches appear → clamp was the cause, gate may pass. If format
  stays high even untruncated → there is a second markup-fidelity issue; report trajectories, do NOT
  jump to a data rebuild without evidence.
- `worker_done` to your coordinator with the numbers; `ask` if blocked on the docker layer.
