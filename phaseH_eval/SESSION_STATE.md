# Session State

## What We Verified

- Live Gemma4 + LoRA serving on GPU1 worked at `max_seq_len=98304`.
- Direct OpenAI-compatible calls to the adapter worked.
- mini-SWE-agent could talk to the server and run Docker testbeds.
- The earlier 32k/65k failures were context-limit failures, not a broken adapter.

## What Failed

- SWE-bench Lite smoke cases did not converge to a patch.
- The model spent its budget on repeated reads and format errors, then returned empty patches.
- Uncapped runs could spend 100+ turns without submitting.

## Current Fix Direction

- Clamp live turn generation with a default `max_tokens` plus stop markers.
- Filter agentic training traces by shape: early edit, bounded read loops, verify tail.
- Keep the dataset builders pointed at the filtered normalized output.

## Current Objective

- Make Gemma31B the best Coding model for SWE-bench.

## Current Progress Resume (2026-07-11)

- `swe-edit-v5-24k-cont.service` has finished successfully at `global_step=5` (checkpoint-5 written) with cgroup peak `~36.7GiB` memory and `~945MiB` swap.
- The 24k continuation lane is no longer running; next action is a behavioral gate on
  - `adapters/unsloth_agentic_swe_edit_trace_v5_cp20_24k_s2/checkpoint-5`.
- Suggested next ETA: run a 5-case 256k smoke gate next (`SCORE=0` quick pass, then `SCORE=1`) and decide continue/bail within ~45–60 minutes.

## Latest 24k Continuation Status

- Active continuation run completed successfully:
  - Service: `swe-edit-v5-24k-cont.service`
  - Command: `phaseD_sft/train_rust_lora.py --data data/unsloth_agentic_24k_train_swe_edit_trace_v5_budget18432 --out adapters/unsloth_agentic_swe_edit_trace_v5_cp20_24k_s2 --init-adapter adapters/unsloth_agentic_swe_edit_trace_v5_cp20_24k_s1/checkpoint-1 --max-seq 24576 --max-steps 5 --warmup-steps 1 --lr 0.0002 --save-steps 1`
  - Remote start: `2026-07-11 04:52:02 UTC`
  - Service state: `active (exited)` with `Result=success`, `ExecMainStatus=0`
  - Runtime: ~29 min total, `CPUUsageNSec=1780383710000`
  - Cgroup peak: `MemoryPeak=38,655,414,272` (~36.7 GiB), `MemorySwapPeak=991,006,720` (~945 MiB)
- Checkpoints produced:
  - `adapters/unsloth_agentic_swe_edit_trace_v5_cp20_24k_s2/checkpoint-1` through `checkpoint-5`
  - `checkpoint-5` exists and is the latest `global_step=5`.
- Associated manifest used:
  - `data/unsloth_agentic_24k_train_swe_edit_trace_v5_budget18432` (`data_len=5753`, `max_seq=24576`, `max_steps=5`, `warmup_steps=1`)
- Log file for traceability: `logs/swe_edit_trace_v5_24k_cont.log`

## Immediate Next Step

- Run behavioral hard-subset/24k smoke gate against the new adapter:
  - `adapters/unsloth_agentic_swe_edit_trace_v5_cp20_24k_s2/checkpoint-5`
- If no meaningful edits appear, do not continue scaling this lane; adjust data/rerank/formatting and rerun a fresh A/B continuation.

## Current Run State

- Remote training tree: `/home/ironbcc/projects/gemma4-31B-Coder` on `ironbccllm`.
- Filtered normalization finished on the remote host and produced `data/unsloth_agentic_24k_train_normalized_format_plus640`.
- The coder-repair mix job failed immediately because `build_coder_repair_dataset.py` was launched as a file and could not import `phaseD_sft.*`.
- I patched `phaseD_sft/build_coder_repair_dataset.py` to add the repo root to `sys.path`, matching the other dataset builders.
- The fixed coder-repair mix build completed on the remote host:
  - `base_rows=2458`
  - `repair_rows=768`
  - `total_rows=3226`
  - output: `data/unsloth_agentic_24k_train_normalized_format_plus_coder_repair`
- The watchdog launched the trainer at `2026-07-09T03:23:27Z` with PID `356009`.
- Current training command:
  - `.venv-train/bin/python phaseD_sft/train_rust_lora.py --data data/unsloth_agentic_24k_train_normalized_format_plus_coder_repair --out adapters/unsloth_agentic_24k --max-seq 24576 --epochs 2 --bsz 1 --grad-accum 16 --warmup-steps 20 --load-4bit --resume`
- Tokenization and assistant-loss filtering completed with `supervised examples 3226/3226`.
- The run is `404` optimizer steps total and saves checkpoints every `20` steps.
- As of `2026-07-09T03:27:18Z`, GPU1 is active at 100% util and about 32 GB VRAM, but no checkpoint has been emitted yet.
- Next step is to wait for `adapters/unsloth_agentic_24k/checkpoint-20`, then run the small SWE smoke against that checkpoint before scaling evaluation.
- Later status showed the 24k cold-start run was wedged:
  - Trainer PID: `356009`
  - Log stopped at `0/404` and was not modified after `2026-07-09T03:25:11Z`.
  - No `checkpoint-*` appeared under `adapters/unsloth_agentic_24k`.
  - GPU1 utilization dropped to `0%` while the Python process still held GPU memory.
  - Thread state showed main thread in `D` state and autograd CPU activity; `strace` attach was blocked by ptrace permissions.
  - SSH/ping to `ironbccllm` then became unreachable (`No route to host` / 100% ping loss).
- Do not resume the 24k cold-start path blindly. Recover GPU1, then start a capped continuation from `adapters/unsloth_agentic_14336_gpu1_repair_conc_s40` on the filtered+repair dataset with `max_seq=14336` and `max_steps=20`.
- Local recovery helper added: `phaseD_sft/recover_filtered_smoke_first.sh`.
- Correction: `14k` is the current practical training lane. Treat `24k` as an inference/context experiment only until a fresh small training smoke proves it can produce checkpoints. The watchdog default has been changed to `MAX_SEQ=14336` to avoid accidentally relaunching the wedged 24k path.
- Recovery progress:
  - Killed wedged 24k trainer PID `356009`.
  - Synced updated `phaseD_sft/train_rust_lora.py`, `phaseD_sft/watch_unsloth_agentic_24k.sh`, and `phaseD_sft/recover_filtered_smoke_first.sh` to the remote tree.
  - Verified remote dataset `data/unsloth_agentic_24k_train_normalized_format_plus_coder_repair` exists.
  - Verified init adapter `adapters/unsloth_agentic_14336_gpu1_repair_conc_s40/adapter_model.safetensors` exists.
  - Launched capped 14k continuation as remote PID `366488`:
    - `--init-adapter adapters/unsloth_agentic_14336_gpu1_repair_conc_s40`
    - `--out adapters/unsloth_agentic_filtered_repair_s20_14336`
    - `--max-seq 14336`
    - `--max-steps 20`
    - `--save-steps 10`
  - This run loaded the adapter, tokenized the filtered+repair dataset, reported `supervised examples 3226/3226`, and started training with `Total steps = 20`.
  - After training started at `0/20`, SSH banner access became unreliable again. Tailscale ping showed the peer path healthy, and TCP port 22 was reachable on `192.168.50.148`, but OpenSSH did not complete banner/session startup. Next action is to regain SSH and check for `checkpoint-10` / `checkpoint-20`.
  - Follow-up found the old watchdog was still alive and had restarted the bad 24k trainer as PID `371953`, competing with the intended 14k run on GPU1.
  - Killed old watchdog PID `356002`, bad 24k trainer PID `371953`, and stale shell wrappers `174468` / `366486`.
  - After cleanup, only intended 14k capped trainer PID `366488` remained.
  - The 14k run reached real training progress:
    - `1/20` at about `16:08` elapsed for the first step.
    - `2/20` at about `23:15` elapsed, so step 2 took about `7:07`.
    - `7/20` at about `50:16` elapsed after removing the competing 24k process; steps stabilized around 5-6 minutes.
  - No checkpoint yet because `save_steps=10`. At observed step speed, first checkpoint is expected around step 10 if the process remains stable.
  - The intended 14k capped run produced `adapters/unsloth_agentic_filtered_repair_s20_14336/checkpoint-10` at `2026-07-09 04:50:33` remote time.
  - I stopped the trainer after checkpoint-10 to run the smoke gate before spending more GPU time.
  - Checkpoint-10 is served on GPU1 by vLLM PID `398282`:
    - served model: `gemma4-agentic-cp10`
    - port: `8012`
    - adapter: `adapters/unsloth_agentic_filtered_repair_s20_14336/checkpoint-10`
    - important launch flags: `--max-model-len 32768 --enable-lora --max-lora-rank 32`
  - Direct OpenAI-compatible sanity call returned `OK`.
  - First smoke attempt `runs/smoke_cp10_lite_0_5_20260709_051004` stalled on context-limit retries:
    - prompt tokens `31169` + requested output tokens `1600` exceeded server context `32768` by one token.
  - Second smoke attempt `runs/smoke_cp10_lite_0_5_retry1200_20260709_051237` got further but hit the same failure later:
    - prompt tokens `31569` + requested output tokens `1200` exceeded server context `32768` by one token.
  - Current harness fix:
    - `phaseH_eval/model_clamps.py` default `DEFAULT_MAX_TOKENS=1024`.
    - `phaseH_eval/vllm_direct_model.py` now calls `compact_live_messages(messages)` before every vLLM request.
    - `compact_live_messages` compacts/deduplicates live tool observations so large read/test outputs do not inflate later turns.
    - Local regression test passes: `python3 -m unittest phaseH_eval.tests.test_model_clamps`.
  - Current active smoke:
    - run dir: `runs/smoke_cp10_lite_0_5_livecompact_20260709_051524`
    - wrapper PID: `405281`
    - mini-SWE PID: `405288`
    - Docker container: `minisweagent-8f8d7712`
    - as of the first minute, no `BadRequestError` appeared and GPU1 was at `100%` util.
  - Follow-up correction: full-context serving is viable for this checkpoint.
    - Model config has `text_config.max_position_embeddings=262144`.
    - Restarted checkpoint-10 vLLM on GPU1 with `--max-model-len 262144 --kv-cache-dtype fp8 --gpu-memory-utilization 0.95`.
    - vLLM reported `GPU KV cache size: 447,848 tokens` and `Maximum concurrency for 262,144 tokens per request: 1.71x`.
    - Direct chat sanity on port `8012` returned `OK`.
    - GPU1 memory after startup was about `91.9GB / 97.9GB`.
    - Stopped the earlier 32k/compacted smoke and launched full-context smoke:
      - run dir: `runs/smoke_cp10_lite_0_5_256k_20260709_052736`
      - wrapper PID: `412280`
      - mini-SWE PID: `412287`
      - server PID: `410404`
    - Also ran the checkpoint-10 256k server on the hard prompt subset:
      - run dir: `runs/hard_subset_cp10_256k_20260709_053306`
      - command: `.venv-eval/bin/python phaseD_sft/agent_smoke_eval.py --url http://localhost:8012/v1/chat/completions --model gemma4-agentic-cp10 --max-tokens 1024 --temperature 0 --jsonl results.jsonl`
      - result: `semantic score: 25/30`, `full score: 1/30`
      - full correctness is mostly blocked by output format (`not_raw_diff`, fenced diffs, non-raw diff headers); only `terse_next_command` was full-correct.
      - semantic failures:
        - `terse_agent_summary`: missing `focused test`
        - `exception_specificity_json`: used too broad exception handling instead of only `except json.JSONDecodeError`
        - `deepcopy_nested_config`: retained `DEFAULTS.copy()`
        - `cached_cache_set_state`: returned `frozenset()` instead of removing `@cache` and returning `set()`
        - `cached_property_nested_state`: missed conversion to `@property`

## Latest Checkpoint-10 Evaluation and Continuation

- Checkpoint-10 was also served at full configured context:
  - server command used `--max-model-len 262144`, `--kv-cache-dtype fp8`, and `--gpu-memory-utilization 0.95`.
  - vLLM reported `GPU KV cache size: 447,848 tokens` and `Maximum concurrency for 262,144 tokens per request: 1.71x`.
  - direct chat sanity returned `OK`.
- Hard subset result at checkpoint-10 / 256k:
  - run dir: `runs/hard_subset_cp10_256k_20260709_053306`
  - semantic score: `25/30`
  - full score: `1/30`
  - main semantic misses: overly broad exception handling, shallow copy retained where deep copy/state fix was needed, frozen set returned where mutable set was needed, missed cached-property conversion, and one missing `focused test` mention.
  - full-correct score was mostly blocked by output format, especially fenced or non-raw diffs.
- SWE-Lite smoke at checkpoint-10 / 256k with the packaged config:
  - run dir: `runs/smoke_cp10_lite_0_5_256k_20260709_052736`
  - 5 cases completed.
  - `1/5` submitted; that submitted patch resolved the instance.
  - `4/5` hit `LimitsExceeded` with empty patches after repeated reads or repeated bad edits.
  - root behavior remains read/edit loop, not context capacity.
- Harness clamp work added locally:
  - `phaseH_eval/model_clamps.py` adds `DEFAULT_MAX_TOKENS=1024`, live observation compaction/dedupe, command-history extraction, budget-pressure messages, and forced bash guards.
  - `phaseH_eval/vllm_direct_model.py` compacts live messages and can return synthetic guarded bash tool calls.
  - `phaseH_eval/swebench_edit_first.yaml` caps SWE runs at 80 steps with earlier edit pressure.
  - `python3 -m unittest phaseH_eval.tests.test_model_clamps` passed locally.
- Guarded/edit-first eval findings:
  - `runs/smoke_cp10_lite_0_5_256k_guarded_20260709_055712`: first case shortened to about 90s but submitted empty because the model overwrote `patch.txt` after a valid forced diff.
  - `runs/smoke_cp10_lite_0_5_256k_guarded_v2_20260709_060036`: stopped early because it still had no predictions after several minutes.
  - `runs/smoke_cp10_lite_0_5_256k_editfirst_20260709_060647`: first case used 40 all-read commands, repeated the same `sed`, then hit `LimitsExceeded` with empty patch.
  - conclusion: clamps help runtime control, but checkpoint-10 still needs more edit-first training signal.
- Failed continuation attempt:
  - an earlier resume collided with vLLM on GPU1 and failed with CPU/disk dispatch from OOM.
  - after vLLM was killed, a second resume launched into `adapters/unsloth_agentic_filtered_repair_s20_14336` but incorrectly initialized from `adapters/unsloth_agentic_14336_gpu1_repair_conc_s40` instead of checkpoint-10.
  - that wrong run was killed by exact PIDs only.
- Correct current continuation:
  - output: `adapters/unsloth_agentic_filtered_repair_cp10_to_s10_14336`
  - init adapter: `adapters/unsloth_agentic_filtered_repair_s20_14336/checkpoint-10`
  - log: `logs/train_filtered_repair_cp10_to_s10_14336.log`
  - pid file: `logs/train_filtered_repair_cp10_to_s10_14336.pid`
  - trainer PID at launch: `455226`
  - command:
    - `.venv-train/bin/python phaseD_sft/train_rust_lora.py --data data/unsloth_agentic_24k_train_normalized_format_plus_coder_repair --out adapters/unsloth_agentic_filtered_repair_cp10_to_s10_14336 --init-adapter adapters/unsloth_agentic_filtered_repair_s20_14336/checkpoint-10 --max-seq 14336 --epochs 2 --max-steps 10 --bsz 1 --grad-accum 16 --warmup-steps 3 --load-4bit --logging-steps 5 --save-steps 10 --save-total-limit 10`
  - launch state: GPU1 was clear before launch (`2 MB` used, no GPU1 compute process).
  - early log confirms `[load] model=adapters/unsloth_agentic_filtered_repair_s20_14336/checkpoint-10`.
  - next gate: wait for `checkpoint-10` under the new output dir, then serve that adapter at 256k and rerun hard subset plus 5-case SWE smoke.
- Correct continuation completed:
  - `adapters/unsloth_agentic_filtered_repair_cp10_to_s10_14336/checkpoint-10` was written at remote time `2026-07-09 07:10:23`.
  - train runtime: `3257s`
  - train loss: `0.01229`
  - checkpoint contains `adapter_model.safetensors` around `935M`.
  - This is effectively the cp20 candidate relative to the original `unsloth_agentic_filtered_repair_s20_14336/checkpoint-10`.
- cp20 serving:
  - served model: `gemma4-agentic-cp20`
  - port: `8012`
  - adapter: `adapters/unsloth_agentic_filtered_repair_cp10_to_s10_14336/checkpoint-10`
  - launched with `--max-model-len 262144 --kv-cache-dtype fp8 --gpu-memory-utilization 0.95`.
  - first launch failed because FlashInfer JIT could not find `ninja`.
  - fix: relaunch vLLM with `PATH=/home/ironbcc/projects/gemma4-31B-Coder/.venv-train/bin:$PATH`.
  - server then reported `GPU KV cache size: 447,848 tokens`, `Maximum concurrency for 262,144 tokens per request: 1.71x`, and `/v1/models` health passed.
  - direct chat sanity returned `OK`.
- cp20 hard subset:
  - run dir: `runs/hard_subset_cp20_256k_20260709_071834`
  - semantic score: `25/30`
  - full score: `0/30`
  - compared with cp10, semantic score did not improve.
  - cp20 fixed some earlier hard-subset semantic misses (`exception_specificity_json`, `deepcopy_nested_config`) but lost others (`prompt_injection_observation`, `stale_clock_fixture_complete`) and still missed cache/property state cases.
  - full correctness remains dominated by non-raw/fenced diff formatting.
- cp20 SWE smoke:
  - initial run: `runs/smoke_cp20_lite_0_5_256k_20260709_072100`
  - first two cases failed immediately as `RepeatedFormatError` with empty patches.
  - root cause was malformed bash tool-call schemas such as `{"description": ..., "commands": [...]}` or nested `{"parameters": {"command": ...}}` instead of mini-SWE's required `{"command": "..."}`.
  - harness patch added in `phaseH_eval/vllm_direct_model.py` and `phaseH_eval/model_clamps.py` to salvage recoverable malformed bash tool calls before mini-SWE validates them.
  - local and remote tests passed: `python -m unittest phaseH_eval.tests.test_model_clamps` / `.venv-eval/bin/python -m unittest phaseH_eval.tests.test_model_clamps`, both `11/11`.
  - patched smoke run: `runs/smoke_cp20_lite_0_5_256k_salvage2_20260709_073243`
  - salvage removed the immediate parser death spiral, but case 1 still hit `LimitsExceeded` with empty patch after `125` commands.
  - case 1 repeated the same read command (`grep -n "def separable" astropy/modeling/core.py | grep -A 100 "class CompoundModel" | head -n 100`) from about command 13 through command 125.
  - conclusion: cp20 is not better for SWE; the dominant remaining issue is read-loop/no-edit behavior, not context length.

## Future Restart Safeguards

- `phaseD_sft/train_rust_lora.py` now exposes `--logging-steps`, `--save-steps`, and `--save-total-limit`.
- The default checkpoint retention is now `10` instead of `3`.
- The trainer writes `run_manifest.json` into the adapter output directory with data path, dataset fingerprint, hyperparameters, and init adapter.
- If `--resume` finds no checkpoint but the output directory has a final `adapter_model.safetensors`, the trainer initializes from that adapter instead of silently starting from base.
- `phaseD_sft/watch_unsloth_agentic_24k.sh` now supports `OUT`, `RUN_NAME`, `LOG`, `PIDFILE`, `SAVE_STEPS`, `SAVE_TOTAL_LIMIT`, `LOGGING_STEPS`, and `INIT_ADAPTER` env vars.
- Future continuation from a known-good adapter should use `INIT_ADAPTER=/path/to/adapter OUT=/new/run/dir`, then smoke-test the first checkpoint before allowing a long run.

## Behavior-v2 Attempt

- Added behavior-focused data generation in `phaseD_sft/build_agentic_behavior_dataset.py`.
  - Synthetic rows target malformed bash tool-call recovery, repeated-read-loop recovery, edit-to-diff, and diff-to-submit behavior.
  - Important implementation detail: synthetic observations must be represented as `role: user` messages prefixed with `OBSERVATION:\n`; using `role: tool` caused Gemma chat-template tokenization failures.
  - Focused tests cover the new builder and existing clamp behavior.
- Built the remote dataset:
  - source: `data/unsloth_agentic_24k_train_normalized_format_plus_coder_repair`
  - output: `data/unsloth_agentic_24k_train_agentic_behavior_v2`
  - filter settings: `--filter-base-quality --max-first-edit-ratio 0.35 --max-read-streak 4 --require-verify-tail --repeat 128`
  - output counts: `base_quality_rows=1953/3226`, `behavior_rows=768`, `total_rows=2721`
- Trained behavior-v2 from cp20:
  - init adapter: `adapters/unsloth_agentic_filtered_repair_cp10_to_s10_14336/checkpoint-10`
  - output: `adapters/unsloth_agentic_behavior_v2_cp20_to_s10_14336/checkpoint-10`
  - data: `data/unsloth_agentic_24k_train_agentic_behavior_v2`
  - sequence length: `14336`
  - steps: `10`
  - train runtime: about `2818s`
  - train loss: `0.01418`
- Served behavior-v2 at full context:
  - served model: `gemma4-agentic-behv2`
  - port: `8012`
  - adapter: `adapters/unsloth_agentic_behavior_v2_cp20_to_s10_14336/checkpoint-10`
  - server PID observed: `509540`
  - launch used `PATH=/home/ironbcc/projects/gemma4-31B-Coder/.venv-train/bin:$PATH` so FlashInfer can find `ninja`.
  - vLLM was launched with `--max-model-len 262144 --kv-cache-dtype fp8 --gpu-memory-utilization 0.95`.
  - direct API sanity returned `OK`.
- Behavior-v2 hard subset result:
  - run dir: `runs/hard_subset_behv2_256k_20260709_084715`
  - semantic score: `22/30`
  - full score: `1/30`
  - this regressed from cp10/cp20 semantic `25/30`.
  - full correctness is still dominated by fenced or non-raw diffs.
  - semantic misses include `prompt_injection_observation`, `terse_agent_summary`, nested cached-state cases, cached-property state cases, and stale-clock fixture completeness.
- Behavior-v2 SWE smoke attempt:
  - run dir: `runs/smoke_behv2_lite_0_5_256k_20260709_084918`
  - launched against the full 256k server and `swebench.yaml`.
  - first case started container `minisweagent-f0abecb3` for `astropy-12907`.
  - no trajectory or first action appeared after about six minutes, while GPU1 was at `99-100%`.
  - stopped exact mini-SWE PIDs `511948` and `511942` and removed the container.
  - behavior-v2 vLLM stayed up and GPU1 returned to `0%` compute.
  - conclusion: behavior-v2 is not a scale candidate. It regressed the hard subset and the full-context SWE smoke path is too slow to provide a cheap gate.
- Current recommended next step:
  - Do not continue training this behavior-v2 branch blindly.
  - Either revert to cp20 as the base and build a narrower v3 dataset from real failed trajectories, or first rerun the SWE smoke with a smaller served context / compacted prompt path to get fast first-action telemetry.
  - The v3 data should emphasize actual mini-SWE failures: repeated read command -> specific edit, malformed tool-call schema -> valid one-command bash call, and nonempty diff -> submit. Avoid over-weighting synthetic hard-subset rows that appear to have hurt semantic patch quality.

## SWE-Trajectory v3 Attempt

- Added a hard live harness guard for repeated read loops:
  - `phaseH_eval/model_clamps.py` now forces `git diff -- . > patch.txt && cat patch.txt` after a repeated read-only tail reaches the threshold.
  - After a guarded forced diff, it now forces `echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT && cat patch.txt` on the next turn, even if the diff output was empty.
  - This prevents 100+ turn read-loop burns, but does not by itself create source edits.
  - Focused local and remote tests passed for the guard behavior.
- Added `phaseD_sft/build_swe_trajectory_behavior_dataset.py`.
  - It extracts compact training rows from mini-SWE trajectory JSON files.
  - It converts tool-role observations to user `OBSERVATION:\n...` messages to avoid Gemma chat-template failures.
  - It extracts three row kinds: `format_recovery`, `repeated_read_to_diff`, and `success_submit`.
  - It caps/deduplicates format-recovery rows per trajectory and supports per-kind repeat weights.
- Built remote v3 dataset:
  - output: `data/unsloth_agentic_24k_train_swe_trajectory_behavior_v3`
  - source base: `data/unsloth_agentic_24k_train_normalized_format_plus_coder_repair`
  - trajectory sources:
    - `runs/smoke_cp10_lite_0_5_256k_20260709_052736/**/*.traj.json`
    - `runs/smoke_cp20_lite_0_5_256k_salvage2_20260709_073243/**/*.traj.json`
    - `runs/smoke_behv2_lite_0_5_256k_20260709_084918/**/*.traj.json`
  - extraction counts before weighting: `format_recovery=44`, `repeated_read_to_diff=5`, `success_submit=1`
  - final dataset counts: `2177` total rows = `1953` filtered base rows + `224` trajectory-derived rows
  - source mix: `swe-smith=1953`, `swe_trajectory_repeated_read_to_diff=120`, `swe_trajectory_format_recovery=88`, `swe_trajectory_success_submit=16`
  - dataset integrity check passed: features match base, trajectory rows contain no `tool` role, and `2177/2177` supervised examples survived tokenization.
- Trained v3 from cp20:
  - init adapter: `adapters/unsloth_agentic_filtered_repair_cp10_to_s10_14336/checkpoint-10`
  - output: `adapters/unsloth_agentic_swe_traj_v3_cp20_to_s10_14336`
  - checkpoint: `adapters/unsloth_agentic_swe_traj_v3_cp20_to_s10_14336/checkpoint-10`
  - data: `data/unsloth_agentic_24k_train_swe_trajectory_behavior_v3`
  - sequence length: `14336`
  - steps: `10`
  - train runtime: about `3897s`
  - train loss: `0.01352`
  - important correction: the first launch accidentally defaulted to GPU0 and was killed by exact PIDs before training; the successful launch used `CUDA_VISIBLE_DEVICES=1`, saw `Num GPUs = 1`, and ran on GPU1 only.
- Served v3:
  - served model: `gemma4-agentic-swe-traj-v3`
  - port: `8012`
  - server PID observed: `543950`
  - adapter: `adapters/unsloth_agentic_swe_traj_v3_cp20_to_s10_14336/checkpoint-10`
  - launched with `CUDA_VISIBLE_DEVICES=1`, `--max-model-len 262144`, `--kv-cache-dtype fp8`, and `--gpu-memory-utilization 0.95`
  - launch included `PATH=/home/ironbcc/projects/gemma4-31B-Coder/.venv-train/bin:$PATH` so FlashInfer can find `ninja`.
  - health passed and vLLM reported `GPU KV cache size: 447,848 tokens`.
- v3 hard subset:
  - run dir: `runs/hard_subset_swe_traj_v3_256k_20260709_102758`
  - semantic score: `24/30`
  - full score: `0/30`
  - this is better than behavior-v2 (`22/30`) but still worse than cp10/cp20 (`25/30`).
  - raw/fenced diff formatting remains a hard blocker for full-correct scoring.
- v3 SWE smoke:
  - run dir: `runs/smoke_swe_traj_v3_lite_0_5_256k_guard_20260709_103003`
  - exit status: `Submitted` for all 5 Lite cases.
  - non-empty patches: `1/5`
  - resolved by scorer evidence: the one non-empty patch (`astropy__astropy-6938`) passed; score wrapper then failed during Docker report generation with a stale-container `docker.errors.NotFound`, not because the patch failed.
  - empty patches: `4/5` (`astropy__astropy-12907`, `14182`, `14365`, `14995`)
  - first-case trajectory showed the core remaining failure: the model created a repro script, then repeated `sed -n '100,200p' astropy/modeling/separable.py | cat -n` dozens of times; harness guards forced diff checks and final submit, but no source edit was made.
  - conclusion: v3 improved runtime and submission rate, but did not improve non-empty patch rate or SWE solved count over the cp10 smoke. The dominant gap is still edit generation, not format recovery or submit timing.
- Current recommended next step:
  - Do not continue SFT on v3 blindly.
  - Need edit-producing traces, not more read-loop-to-diff traces. The next dataset should contain real or teacher-generated source edits after failing repro/inspection, followed by focused verification and submit.
  - If using SWE-Bench gold patches for these same Lite smoke instances, mark it as an oracle/contaminated diagnostic lane only; it should not be treated as evidence for general SWE-Bench Verified improvement.
  - For a clean lane, generate teacher edit traces on non-eval or training instances, filter for first edit before the first 40% of turns, require focused verify, and keep non-empty diff submit tails.

## Local Blocker

- This checkout does not have the cached agentic datasets on disk.
- The host Python environment here also lacks `datasets`, so full dataset rebuilds must run in the training/eval environment on the target machine.

## SWE-Edit-Trace v4 Attempt

- Added `phaseD_sft/build_swe_edit_trace_dataset.py`.
  - It builds compact edit-first agent traces from `princeton-nlp/SWE-bench` training rows, not Lite smoke/eval rows.
  - Each row is shaped as: issue prompt -> focused `sed` inspection -> `git apply` real training-split patch -> focused pytest command -> `git diff` -> `COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT`.
  - Tool observations are represented as user `OBSERVATION:\n...` messages; no training row uses role `tool`.
  - The source label is `swe_train_oracle_edit_trace` to make the oracle/gold-patch origin explicit.
- Added `phaseD_sft/tests/test_build_swe_edit_trace_dataset.py`.
  - Local focused tests passed:
    `python3 -m unittest phaseD_sft.tests.test_build_swe_edit_trace_dataset phaseD_sft.tests.test_build_swe_trajectory_behavior_dataset phaseH_eval.tests.test_model_clamps`
    -> `20 tests OK`.
  - Remote focused tests passed with `.venv-eval/bin/python` -> `20 tests OK`.
- Built remote v4 dataset:
  - raw output: `data/unsloth_agentic_24k_train_swe_edit_trace_v4`
  - command used `--limit 256 --repeat 4 --max-patch-chars 3500 --max-problem-chars 1800 --seed 7`
  - source base after quality filter: `1953/3226`
  - raw final rows: `2977` = `1953` base `swe-smith` + `1024` `swe_train_oracle_edit_trace`.
- Token-budget validation:
  - raw v4 had `2977/2977` supervised examples and no `tool` roles, but `470` rows exceeded `14336` rendered tokens.
  - All `470` over-budget rows were inherited `swe-smith` base rows; none of the new edit traces exceeded budget.
  - Saved budget-clean dataset: `data/unsloth_agentic_24k_train_swe_edit_trace_v4_budget14336`.
  - budget-clean rows: `2507` = `1483` `swe-smith` + `1024` `swe_train_oracle_edit_trace`.
  - final validation: `2507/2507` supervised examples, `0` tool-role rows, `0` truncated examples, max rendered length `14328`.
- Started short v4 training from cp20:
  - command:
    `CUDA_VISIBLE_DEVICES=1 .venv-train/bin/python phaseD_sft/train_rust_lora.py --data data/unsloth_agentic_24k_train_swe_edit_trace_v4_budget14336 --out adapters/unsloth_agentic_swe_edit_trace_v4_cp20_to_s10_14336 --init-adapter adapters/unsloth_agentic_filtered_repair_cp10_to_s10_14336/checkpoint-10 --max-seq 14336 --epochs 2 --max-steps 10 --bsz 1 --grad-accum 16 --warmup-steps 3 --load-4bit --logging-steps 5 --save-steps 10 --save-total-limit 10`
  - v3 vLLM server PID `543950` was killed first to free GPU1.
  - training loaded on GPU1 and reported `Num GPUs = 1`.
  - in-script tokenization reported `supervised examples 2507/2507`.
  - training reached `0/10` optimizer steps with GPU1 initially at `100%`, then stopped producing progress output.
  - Remote SSH became unhealthy: host still answered ping, but new SSH connections timed out during banner exchange.
  - The attached training SSH session was interrupted and closed with exit code `255`; post-interrupt SSH still timed out during banner exchange.
- Current v4 status:
  - v4 dataset and code are built and validated.
  - v4 checkpoint does not exist; after reboot, `adapters/unsloth_agentic_swe_edit_trace_v4_cp20_to_s10_14336` contained only `run_manifest.json`.
  - Root cause of the stuck host was system RAM OOM, not GPU VRAM OOM.
    - Host rebooted at `2026-07-09 16:54 UTC`; after reboot GPU1 was free.
    - Previous-boot journal showed global OOM at `2026-07-09 11:09:54 UTC`.
    - OOM killed user/session/system processes plus vLLM processes including `vllm` PID `375147`, `VLLM::EngineCore` PIDs `373603`, `375444`, and `378432`.
    - The v4 trainer PID `565535` was present in OOM dumps with huge virtual memory and swap/page-table footprint while training had reached `0/10` optimizer steps.
    - At the time of the failure, the machine had about `30GiB` system RAM and `256GiB` swap; running 14k Unsloth training while the GPU0 vLLM stack was resident overcommitted host RAM/swap even though GPU1 VRAM was not exhausted.
    - This explains the symptom where ping/TCP port 22 worked but SSH banner exchange timed out: sshd/systemd/tailscale/journald were degraded by OOM and memory pressure.
  - The original safe restart requirement was to remove resident vLLM before training. This is superseded by the 2026-07-10 RAM upgrade and cgroup-isolated smoke plan below; production vLLM must now remain running.
- Added local recovery script `phaseD_sft/recover_swe_edit_v4_smoke_first.sh`.
  - It waits for SSH, kills only stale trainers matching `data/unsloth_agentic_24k_train_swe_edit_trace_v4_budget14336`, waits for GPU1 to be free, and launches only a 1-step smoke into `adapters/unsloth_agentic_swe_edit_trace_v4_cp20_smoke1_14336`.
  - It defaults to LAN host `192.168.50.148` with `HostKeyAlias=ironbccllm.tail0cc1d4.ts.net`, because the tailnet path was timing out while LAN port 22 still accepted TCP.
  - It requires GPU0 production health on ports `8000`, `8101`, `8103`, and `8104` before launch and during every monitor interval.
  - It requires at least `60000 MiB` `MemAvailable` before launch and aborts only its training unit if available RAM falls below `12000 MiB` or swap use rises above `32768 MiB`.
  - It runs the trainer through the user systemd unit `swe-edit-v4-smoke.service` with `MemoryHigh=40G`, `MemoryMax=48G`, `MemorySwapMax=32G`, and `OOMScoreAdjust=500`.
  - It stops after the one-step checkpoint gate; it cannot automatically launch the 10-step continuation.
  - `phaseD_sft/vllm_kill_watch.sh` now refuses to run unless `ALLOW_PROD_VLLM_KILL=I_UNDERSTAND` is explicitly set.
  - Local safety tests and syntax checks passed on 2026-07-10.

## 2026-07-10 Post-Upgrade Resume State

- Hardware and operating baseline after the RAM upgrade:
  - Linux reports `96G` online physical memory, `0B` offline, and about `93GiB` usable.
  - With the full production stack healthy, `MemAvailable` was about `68GiB`; all `256GiB` swap was free.
  - `vllm.service` was active with zero restarts. Its cgroup used about `47.7GB`, split roughly into `14.8GB` anonymous memory and `32.6GB` file cache.
  - GPU0 used about `76.2GB / 97.9GB` VRAM for production. GPU1 used `2MB / 97.9GB` and had no training process.
  - Production health passed on router `8000`, Gemma4 `8101`, nano `8103`, and speech `8104`.
- Goal remains SWE-bench Verified improvement through the v4 edit-producing trace lane.
- Current v4 artifacts are intact:
  - dataset `data/unsloth_agentic_24k_train_swe_edit_trace_v4_budget14336` exists and remains the validated `2507`-row, non-truncated 14k corpus.
  - init adapter `adapters/unsloth_agentic_filtered_repair_cp10_to_s10_14336/checkpoint-10/adapter_model.safetensors` exists.
  - neither v4 output contains a checkpoint; both contain only `run_manifest.json`.
- Immediate gate: run the cgroup-bounded one-step smoke on GPU1 while production remains up. Do not start ten steps until `checkpoint-1` exists, production health remains green, and the unit memory peak is recorded.
- Current interruption at `2026-07-10 19:26 UTC`:
  - The host became unreachable before the reviewed files were synced and before any training launch.
  - LAN ping/SSH, Tailscale ping, and production ports `8000`, `8101`, `8103`, and `8104` were all unreachable.
  - One Wake-on-LAN packet was sent to the known NIC MAC, but the host did not return during the following two-minute watch.
  - No v4 trainer or recovery watcher was running when connectivity disappeared, so this outage was not caused by the new recovery path.
  - Likely recovery is to exit UEFI or physically power on the host; after SSH returns, re-run production health checks before syncing or launching the smoke.
- The host later returned with the same healthy post-upgrade baseline and production endpoints.
- Protected v4 smoke results on 2026-07-10:
  - The first transient-unit launch was stopped by the user systemd manager when its short SSH session closed. It used only `641.7MB` peak RAM and `0B` swap, so this was not an OOM or trainer failure.
  - Root cause was `Linger=no`; `sudo loginctl enable-linger ironbcc` was applied and verified as `Linger=yes`.
  - The launcher now refuses to run without lingering and uses `--remain-after-exit` so final cgroup metrics remain inspectable.
  - The repeated one-step run completed successfully and wrote `adapters/unsloth_agentic_swe_edit_trace_v4_cp20_smoke1_14336/checkpoint-1`.
  - Optimizer-step runtime was about `260s`; train runtime was `262s`; loss was `0.02051`.
  - Trainer memory peak was `42,950,631,424` bytes under `MemoryHigh=40G` and `MemoryMax=48G`; observed swap peak was about `390MB`.
  - Minimum observed host `MemAvailable` was about `39GB`; production health remained green throughout; GPU1 returned to `2MB` after completion.
  - The recovery script now supports an explicit `TRAIN_STEPS` override but defaults to `1`; it never escalates step count automatically.
- Next gate: run the explicit ten-step v4 continuation in `adapters/unsloth_agentic_swe_edit_trace_v4_cp20_to_s10_14336` under the same cgroup limits, then serve checkpoint-10 on GPU1 and run the hard subset plus five-case SWE smoke before any further training.
- Ten-step v4 continuation attempt on 2026-07-10:
  - Launched at about `23:19:38 UTC` as `swe-edit-v4-s10.service` with `TRAIN_STEPS=10`, `WARMUP_STEPS=3`, `MemoryHigh=40G`, `MemoryMax=48G`, and `MemorySwapMax=32G`.
  - Model load, tokenization, and supervision filtering completed; training reached `0/10` with GPU1 at `100%` utilization.
  - Last healthy monitor samples showed production green, trainer cgroup around `42.94GB`, host `MemAvailable` around `40GB`, and only `617MB` swap used.
  - The host became unreachable between `23:24:56` and `23:27:56 UTC`; LAN, SSH, Tailscale, and all production endpoints disappeared.
  - No `checkpoint-10` was produced. The local watcher exited without launching anything else. A Wake-on-LAN packet did not restore the host during a two-minute watch.
  - This telemetry does not support a conventional host-RAM OOM: available RAM was still about `40GB`, swap use was low, and the trainer was below its hard cgroup cap immediately before connectivity was lost.
  - Do not relaunch training or increase training context. After physical recovery, inspect `journalctl -b -1 -k`, system journal shutdown/panic/OOM records, GPU Xid events, and hardware/power logs before choosing the next step.

## 2026-07-10 Hardware Recovery Finding

- After physical recovery, the host is healthy but now exposes only `64G` online physical memory (`61GiB` usable), with about `39GiB` available beside production and all `255GiB` swap free.
- `dmidecode` reports only the matched 2x32GB KLEVV `KD5BGUA80-64A320J` DIMMs:
  - P0 Channel A DIMM 1: 32GB, configured at 4800 MT/s.
  - P0 Channel B DIMM 1: 32GB, configured at 4800 MT/s.
- Both 16GB Corsair `CMK32GX5M2D6000C36` modules previously present in Channel A/B DIMM 0 are now reported as `No Module Installed`.
- The previous boot journal ends abruptly at the loss of connectivity. It contains no host OOM, kernel panic, NVIDIA Xid, AER, MCE, thermal, watchdog, or clean-shutdown record. Last training telemetry still showed about `40GB` `MemAvailable` and only about `617MB` swap in use.
- This evidence points to a platform/memory stability failure in the prior mixed 96GB four-DIMM configuration, not a 14k-context capacity OOM.
- Current live state:
  - `vllm.service` is active and health checks pass on ports `8000`, `8101`, `8103`, and `8104`.
  - GPU0 holds production at about `76.2GB / 97.9GB` VRAM.
  - GPU1 is idle at `2MB / 97.9GB`; no trainer is running.
  - The failed ten-step output still contains only `run_manifest.json`; no checkpoint was produced.
- Do not raise training context. The validated v4 corpus already fits completely at `--max-seq 14336` with a maximum rendered length of `14328`.
- Do not relaunch the existing 14k trainer on the 64GB host under the old policy. Its proven one-step cgroup peak was about `42.95GB`, while only about `39GiB` is currently available beside production. The launcher's `60000 MiB` prerequisite correctly blocks this configuration.
- Safe checkpoint-1 evaluation started after recovery:
  - Adapter: `adapters/unsloth_agentic_swe_edit_trace_v4_cp20_smoke1_14336/checkpoint-1`.
  - vLLM is serving on GPU1 at port `8012` as `gemma4-agentic-v4-s1` with `--max-model-len 262144`, FP8 KV cache, and the LoRA enabled.
  - The evaluation server is isolated in `v4-s1-eval.service` with `MemoryHigh=16G`, `MemoryMax=24G`, and `MemorySwapMax=8G`; production remained healthy throughout startup.
  - Hard subset output: `runs/hard_subset_v4_s1_256k_20260711_013602`.
  - Hard subset result: `26/30` semantic and `0/30` full.
  - This is one semantic point above cp20's `25/30`, but exact raw-diff formatting remains unsolved. The four semantic misses were `prompt_injection_observation`, `terse_agent_summary`, `cached_property_nested_state`, and `cached_property_set_state`.
- The first SWE-Lite smoke attempt in `runs/smoke_v4_s1_lite_0_5_256k_guard_20260711_013816` was stopped after case 2 spent nearly nine minutes generating full-cap malformed/read-only turns. Case 1 had submitted an empty patch.
- Tightened live clamps after the observed failure:
  - default per-turn output cap reduced from 1024 to 512 tokens;
  - no-edit pressure now starts at command 25 exactly;
  - command 37 now forces `git diff` even when no edit exists, followed by immediate forced submit;
  - local and remote clamp tests pass (`15/15`).
- Completed five-case SWE-Lite rerun: `runs/smoke_v4_s1_lite_0_5_256k_guard2_20260711_015002`.
  - submissions: `5/5`;
  - non-empty patches: `1/5`;
  - resolved: `1/5`, `astropy__astropy-6938`;
  - the scorer recorded all FAIL_TO_PASS and PASS_TO_PASS tests successful, then hit the known stale-container `docker.errors.NotFound` while producing the aggregate report;
  - format errors: `100/228` model calls (`43.9%`), well above the `<10%` gate;
  - three cases attempted source edits by command 9, but two used truncated base64 rewrite commands and produced empty diffs;
  - conclusion: checkpoint-1 does not pass the behavior gate. It is semantically slightly better on hard30, but SWE behavior remains effectively at the v3 level.
- Added an explicit low-RAM continuation mode to `phaseD_sft/recover_swe_edit_v4_smoke_first.sh`:
  - opt-in requires `ALLOW_64G_TRAINING=I_ACCEPT_BOUNDED_SWAP`;
  - 64GB mode uses `MemoryHigh=36G`, `MemoryMax=45G`, `MemorySwapMax=16G`, a 36GB launch floor, an 8GB available-RAM abort floor, a 16GB swap-use abort, and 10-second monitoring;
  - it also requires at least 64GB configured swap;
  - the standard 96GB policy remains unchanged.
- Critical correction: the earlier `checkpoint-1` used `warmup_steps=1` with `max_steps=1`.
  - Its trainer state logged learning rate `0.0`.
  - Its adapter SHA-256 exactly matched cp20 (`91fbd5cb...`), proving that no v4 update occurred.
  - Therefore the `26/30` hard result and `1/5` SWE result above measure cp20 under the revised harness, not learned v4 behavior.
- Completed a real one-step v4 update with `warmup_steps=0` in explicit 64GB mode:
  - output: `adapters/unsloth_agentic_swe_edit_trace_v4_cp20_real_s1_14336/checkpoint-1`;
  - learning rate: `0.0002`; loss: `0.0204957`; runtime about `5m14s`;
  - adapter SHA-256 `34f354d3...`, different from cp20;
  - unit result `success`, exit status `0`, memory peak `38,655,463,424` bytes;
  - host swap peaked around 4.7GB during monitoring, minimum observed `MemAvailable` was about 13.6GB, and production remained healthy;
  - GPU1 returned to `2MB` after completion.
- Next gate: serve the real one-step checkpoint at 256k, rerun hard30, and only continue training if semantics remain acceptable and the edit-first signal improves.
- Real one-step hard30 result:
  - run: `runs/hard_subset_v4_real_s1_256k_20260711_023425`;
  - semantic `23/30`, full `1/30`;
  - this regressed below cp20 (`25/30`) and the no-op rerun (`26/30`), so no SWE smoke was run and the high-LR checkpoint will not be continued.
- Added `TRAIN_LR` pass-through to the bounded launcher. Next candidate starts fresh from cp20 for five steps at peak LR `2e-5` with one warmup step; it must recover hard30 semantics before a SWE smoke.
- Completed low-LR five-step 14k candidate:
  - output: `adapters/unsloth_agentic_swe_edit_trace_v4_cp20_lowlr_s5_14336/checkpoint-5`;
  - schedule: one warmup step, then LR `2e-5`, `1.707e-5`, `1e-5`, `2.929e-6`;
  - losses by step: `0.02050`, `0.02084`, `0.01446`, `0.01837`, `0.01893`;
  - gradient norm rose to `32.56` on step 5, so evaluation is required before continuation;
  - unit result `success`, memory peak `38,655,451,136` bytes, production healthy throughout.
- User selected `18432` as the training context limit for following runs. Do not change the completed 14k candidate retroactively. Before future training, rebuild the budgeted v4 dataset from the raw corpus at 18,432 tokens, record recovered row count/max length, and run a one-step 18k memory gate before a multi-step continuation.
- Low-LR five-step hard30 result:
  - run: `runs/hard_subset_v4_lowlr_s5_256k_20260711_030122`;
  - semantic `23/30`, full `1/30`;
  - rejected. Do not use this adapter as the 18k initializer; use cp20.

## 2026-07-11 Active 18k Handoff

This is the canonical continuation point for a new Codex session.
This lane is complete at the one-step memory gate level; continue from the v5 planning notes below.

### Non-Negotiable Host Constraints

- GPU0 production must remain running. Never stop or kill `vllm.service` or its GPU0 workers.
- Production health ports are `8000`, `8101`, `8103`, and `8104`; all must remain green.
- The host currently reports `61GiB` total RAM and `255GiB` swap. The prior mixed 96GB four-DIMM configuration hard-reset under load and both 16GB Corsair DIMMs are now absent.
- Training is GPU1-only and must use the explicit bounded 64GB launcher mode. Do not bypass its cgroup or health monitor.
- Serving context may remain 256k. The selected context for following training runs is `18432`, not 24k.

### 18k Dataset Artifacts

- Full budget-clean dataset: `data/unsloth_agentic_24k_train_swe_edit_trace_v4_budget18432`.
- Exact native-Gemma rendered-token audit:
  - raw rows: `2977`;
  - accepted: `2782` (`1758` SWE-Smith + `1024` oracle edit traces);
  - rejected: `195`, all SWE-Smith;
  - max accepted length: `18417`;
  - minimum rejected length: `18435`;
  - source max: `24325`.
- Long-context memory-gate shard: `data/unsloth_agentic_24k_train_swe_edit_trace_v4_budget18432_long16`.
  - 16 rows, all between `18079` and `18417` rendered tokens.
  - manifests: `budget_manifest.json` in the full dataset and `gate_manifest.json` in the long16 shard.

### Active Run At Handoff (Completed)

- Purpose: memory-only 18k worst-case gate. It intentionally uses `max_steps=1` and `warmup_steps=1`, so LR is expected to be zero and the checkpoint must not be treated as a learned candidate.
- User unit: `swe-edit-v4-18k-memgate.service`.
- Data: `data/unsloth_agentic_24k_train_swe_edit_trace_v4_budget18432_long16`.
- Initial adapter: `adapters/unsloth_agentic_filtered_repair_cp10_to_s10_14336/checkpoint-10` (cp20 baseline).
- Output: `adapters/unsloth_agentic_swe_edit_trace_v4_cp20_18k_long16_memgate/checkpoint-1`.
- Training max sequence: `18432`; gradient accumulation: `16`.
- Bounded mode: `MemoryHigh=36G`, `MemoryMax=45G`, `MemorySwapMax=16G`, abort below `8GB` available host RAM or above `16GB` host swap used, production checks every 10 seconds.
- Launch command from the local checkout:
  `ALLOW_64G_TRAINING=I_ACCEPT_BOUNDED_SWAP DATA=data/unsloth_agentic_24k_train_swe_edit_trace_v4_budget18432_long16 INIT_ADAPTER=adapters/unsloth_agentic_filtered_repair_cp10_to_s10_14336/checkpoint-10 SMOKE_OUT=adapters/unsloth_agentic_swe_edit_trace_v4_cp20_18k_long16_memgate RUN_NAME=swe_edit_trace_v4_18k_long16_memgate SMOKE_UNIT=swe-edit-v4-18k-memgate MAX_SEQ=18432 TRAIN_STEPS=1 WARMUP_STEPS=1 bash phaseD_sft/recover_swe_edit_v4_smoke_first.sh`
- Last recorded live telemetry around `03:06:27 UTC`: production green, trainer cgroup about `38.65GB`, host `MemAvailable` about `43.6GB`, host swap used about `6.4GB`, GPU1 loading/active.
- Completion state after lane:
  - `systemctl --user show swe-edit-v4-18k-memgate.service -p ActiveState -p SubState -p Result -p ExecMainStatus -p MemoryPeak -p MemorySwapPeak`
    - `ActiveState=active`, `SubState=exited`, `Result=success`, `ExecMainStatus=0`
    - `MemoryPeak=38655397888` (`~36.7GiB`), `MemorySwapPeak=642494464` (`~613MiB`)
  - `trainer_state.json` reports `global_step=1`, `learning_rate=0.0`, `loss≈0.01147`.
  - Checkpoint exists and SHA-256 matches cp20 (`91fbd5cb...66c62`), confirming this was a no-op gate checkpoint.

### Resume Commands

- Check the active unit:
  `systemctl --user show swe-edit-v4-18k-memgate.service -p ActiveState -p SubState -p Result -p ExecMainStatus -p MemoryCurrent -p MemoryPeak -p MemorySwapPeak`
- Follow its trainer log:
  `tail -f logs/swe_edit_trace_v4_18k_long16_memgate_smoke1.log`
- Check the checkpoint:
  `test -f adapters/unsloth_agentic_swe_edit_trace_v4_cp20_18k_long16_memgate/checkpoint-1/adapter_model.safetensors`
- Verify production and GPUs:
  `for p in 8000 8101 8103 8104; do curl -fsS --max-time 2 http://127.0.0.1:$p/health >/dev/null || echo "$p FAIL"; done; nvidia-smi`

### Model Decisions Already Proven

- The earlier apparent v4 checkpoint-1 was a no-op: warmup consumed its only step, LR was `0.0`, and its adapter hash exactly matched cp20.
- A real one-step v4 update at LR `2e-4` scored only `23/30` semantic, `1/30` full; rejected.
- A fresh five-step v4 run at peak LR `2e-5` (four nonzero updates) also scored `23/30` semantic, `1/30` full; rejected.
- The cp20/no-op reference is `25-26/30` semantic and `1/30` full. Do not initialize future training from either rejected v4 adapter.
- The no-op/cp20 SWE smoke with the tightened harness submitted `5/5`, produced `1/5` non-empty patches, resolved `1/5`, and had `43.9%` format errors. Runtime clamps improved termination but did not create edit behavior.

### Follow-Up Guidance

1. Finish the active 18k memory gate and record peak GPU VRAM, cgroup `MemoryPeak`, host minimum `MemAvailable`, host maximum swap use, checkpoint existence, and production health. Do not behavior-evaluate its zero-LR checkpoint.
2. If the 18k gate OOMs GPU1 or crosses a safety threshold, keep the full 18k dataset but train on a bucketed/lower effective context path; do not increase cgroup caps or touch production.
3. Do not continue training on the current v4-only mixture as-is. Both real-update attempts caused hard30 forgetting. Build a v5 replay mixture that retains the normalized format/coder-repair anchors from `data/unsloth_agentic_24k_train_normalized_format_plus_coder_repair` while adding the 18k-safe edit traces/SWE-Smith rows. Verify source counts and zero truncation before training.
4. Start v5 from cp20, use `--max-seq 18432`, and use a small LR lane. Gate an early checkpoint on hard30 first; require at least `25/30` semantic before any SWE smoke.
5. Historical v4 smoke pass criteria were non-empty patches at least `3/5`, format-error rate below `10%`, and median first source edit below command 10. **This is superseded for the active v6 lane by the 30-case policy below.**
6. For v6, only after the 30-case smoke passes should a broader Verified slice run. Preserve 256k serving for evaluation even though training is 18k.

## Useful Artifacts

- `runs/repair_conc_s40_lite_smoke1_98k_step40_20260709_023201`
- `runs/repair_conc_s40_lite_smoke2_98k_step40_maxtok1024_20260709_023748`
- `phaseH_eval/model_clamps.py`
- `phaseD_sft/agentic_trace_filters.py`

## 2026-07-11 v5 Completion + Dataset Audit + v6 Lane

**v5 run completed successfully; dataset is corrupted and must be rebuilt for v6. Full per-phase executable runbook is at `phaseD_sft/V6_PIPELINE_RUNBOOK.md` — follow that document for execution steps, exact commands, env vars, gate criteria, and what not to do. This section records the outcomes and decisions.**

### Completed: v5 24k Continuation Run

- Service: `swe-edit-v5-24k-cont.service` on host `192.168.50.148`.
- Command: `.venv-train/bin/python phaseD_sft/train_rust_lora.py --data data/unsloth_agentic_24k_train_swe_edit_trace_v5_budget18432 --out adapters/unsloth_agentic_swe_edit_trace_v5_cp20_24k_s2 --init-adapter adapters/unsloth_agentic_swe_edit_trace_v5_cp20_24k_s1/checkpoint-1 --max-seq 24576 --max-steps 5 --warmup-steps 1 --lr 0.0002 --save-steps 1`
- Result: `success`, exit status `0`. Runtime ~29 min, CPUUsageNSec=1780383710000.
- Cgroup peak: `MemoryPeak=38,655,414,272` (~36.7 GiB), `MemorySwapPeak=991,006,720` (~945 MiB).
- Checkpoints produced: `checkpoint-1` through `checkpoint-5` under `adapters/unsloth_agentic_swe_edit_trace_v5_cp20_24k_s2`. The v5-s2/cp5 is the init candidate for Phase 0 eval.
- Production remained green throughout (ports 8000, 8101, 8103, 8104).

### v5 Dataset Audit Findings (corrupt — must be rebuilt for v6)

Dataset `data/unsloth_agentic_24k_train_swe_edit_trace_v5_budget18432` used by the completed run:

- **Row count**: 5,753 total rows.
- **Gemma format**: PASS — `verify_gemma_format_loss.py` (300 samples) → `failure_count=0`. Assistant turns render as `<|turn>model` spans with tool-call JSON; `<|tool_call>`/`<tool_call|>`/`<|tool_response>` balanced; all tool calls are `bash` with valid JSON args; CoT = plain text in assistant content (no think tags). `train_rust_lora.py` keeps the base tokenizer's native Gemma template (Unsloth's generic gemma-4 template drops tool_calls) with assistant-turn loss masking.
- **3,282 exact-duplicate rows** (57% of 5,753): byte-identical copies with up to ×14 per instance. Root cause: v5 = concat(normalized_format_plus_coder_repair, v4_budget18432) with no dedup anywhere in the chain; both parents already carried ~760 exact dups each and they overlap.
- **2,157/3,961 swe-smith rows** end on user OBSERVATION (unsupervised tail). Root cause: `phaseD_sft/prune_loops.py:99` — `messages[:cut_index]` cuts AT an assistant index, leaving the preceding user observation as the final message.
- **3,129 nebius rows silently dropped**: raw base `data/unsloth_agentic_24k_train` (15,683 rows) = swe-smith 12,554 + nebius 3,129; the v4/v5 chain contains zero nebius rows.
- **Compaction escape hatch**: `phaseD_sft/compact_observations.py::_trim_to_char_budget` important-lines path returns candidates over `max_chars` (the `or len(head) <= 2` branch) — ~0.5% of observations exceed the 2,400-char cap (max observed 7,172 chars).
- **Variants are not bugs**: raw has 4,704 instance_ids with multiple rows; many are legitimate 16k-window splits (`split16k` chain), NOT accidents. Dedup must be content-hash exact, never by instance_id.

### User Decisions (Approved in v6 Plan)

- Budget cap: **32,768** tokens rendered.
- Trailing-user traces: trim to last assistant message (drop trace if no assistant remains or fewer than 2 messages).
- Nebius rows: audit then reinclude only what passes format + trim + dedup gates. If <200 survive, skip reinclusion this round.
- Init adapter for v6 training: evaluate `adapters/unsloth_agentic_swe_edit_trace_v5_cp20_24k_s2/checkpoint-5` on hard30 first (serve at 256k via isolated eval unit pattern). **≥25/30 semantic** → use as init; **<25/30** → fall back to cp20 (`adapters/unsloth_agentic_filtered_repair_cp10_to_s10_14336/checkpoint-10`). Record verdict in SESSION_STATE.
- LR lane: `2e-5` peak with cosine + 1 warmup step, 5 steps first (higher lanes regressed hard30 on dup-corrupted v4 data; re-test low lane on clean v6).

### Host Constraints (Non-Negotiable)

- GPU0 prod vLLM off-limits. Never stop or kill `vllm.service` or its GPU0 workers.
- Production health ports: **8000** (router), **8101** (Gemma4), **8103** (nano), **8104** (speech). All must remain green throughout.
- Training is **GPU1-only**.
- 61 GiB RAM + 255 GiB swap (both 16GB Corsair DIMMs absent; KLEVV 2x32GB configured at 4800 MT/s).
- Bounded launcher only (`phaseD_sft/recover_swe_edit_v4_smoke_first.sh`). Required env: `ALLOW_64G_TRAINING=I_ACCEPT_BOUNDED_SWAP`.
- Cgroup caps via systemd unit: **MemoryHigh=36G**, **MemoryMax=45G**, **MemorySwapMax=16G**, OOMScoreAdjust=500. Abort floors: host available RAM <8 GiB or swap use >16 GiB. **Never raise cgroup caps.**
- Kill only by exact numeric PID (from `ps -eo pid,args | grep ... | grep -v grep`). **NEVER use `pkill -f`** — pattern matches own command string over SSH and kills the session.

### VRAM Reality / Staged Gates

- At max-seq 24,576 with rows ≤18,417 tokens (previous v4 lane), GPU1 peaked **92.8/97.9 GB** VRAM.
- True 24k–32k rows are unproven → staged memory gates mandatory with automatic fallback tiers: 32,768 → 24,576 (rebuild budget dataset at lower cap; do not raise cgroup caps).

### v6 Phase List with Gates (from approved plan)

See `phaseD_sft/V6_PIPELINE_RUNBOOK.md` for the full per-phase executable runbook. Summary:

| Phase | Name | Gate / Exit Condition |
|---|---|---|
| 0 | Init-adapter decision (parallel w/ data work) | Serve v5-s2 cp5 on GPU1 at 256k via isolated eval unit; hard30 ≥25/30 semantic → keep as init candidate; <25/30 → use cp20 |
| 1 | `dedup_and_trim_traces.py` + compact escape-hatch fix | Unit tests pass locally (`pytest phaseD_sft/tests/`); trim→dedup script works on JSONL/HF; compact_observations hard cap applied with regression test |
| 2 | Base datasource verify + nebius audit (Phase 2b: HF trace-dataset scout) | Source counts match manifest; nebius reinclude if ≥200 rows pass gates |
| 3 | v6 mixture build @ 32,768 tokens | `unique_content_count == rows_accepted`; dups_removed recorded; per-source accepted/rejected in budget_manifest.json; worst-case long16 shard built with gate_manifest.json |
| 4 | Format gate (hard, blocking) | `verify_gemma_format_loss.py --samples 1000` exit 0, failure_count=0; structural scan: 0 exact dups, 0 user-tail traces, all obs ≤2400 chars post-fix |
| 5 | Staged memory gates → training run | Gate A (24k): MemoryPeak/SwapPeak/GPU1 VRAM peak recorded, prod green. Gate B (32k): same; if OOM/threshold trip → fallback tier rebuild at 24,576. Training: init from Phase-0 verdict, --max-seq=highest passed gate, LR=2e-5 peak w/ cosine + 1 warmup step, save-steps=1 |
| 6 | Eval gate sequence (active v6 policy) | hard30 must beat same-day cp20 by ≥3 semantic; then 30-case SWE-Lite ≥18/30 non-empty patches, <10% format errors, median first source edit <command 10. Failure → stop and rebuild/reweight data; do not scale to Verified slice until gates pass |

### Risks & Rollback (from v6 Plan)

- **32k VRAM OOM**: primary risk; handled by staged gates + automatic fallback tier at Phase 5.
- **Host RAM**: any gate/abort trip → unit stops itself via launcher watchdog; prod untouched; no state to roll back (adapters are new dirs; cp20 and v5 artifacts kept immutable).
- **Nebius quality unknown**: gated by audit in Phase 2; skipped if <200 survivors.
- **Full-trace-vs-window substitution** could reintroduce >32k rows: budget filter at Phase 3.4 is the backstop.
- Rollback of everything = keep training/serving off v6 dirs; prior adapters and datasets are never modified.

### Verification (End-to-End)

- Phase 1 unit tests pass locally (`pytest phaseD_sft/tests/`).
- Phase 3 manifests: `unique_content_count == rows_accepted`, dups_removed ≥3,282-equivalent, trailing_trimmed ≈2,157-equivalent, source counts explained vs parents.
- Phase 4: verify script exit 0 twice (1000-sample + full/large sample); structural scan clean.
- Phase 5: gate telemetry recorded in SESSION_STATE before multi-step run (MemoryPeak, SwapPeak, GPU1 VRAM peak, prod health).
- Phase 6: hard30 + smoke numbers with run dirs recorded in SESSION_STATE.

### Phase 0 Result (2026-07-11): v6 init = cp20

- First eval attempt was **invalid**: the orphan 8013 server (`/tmp/start_24k_8013.sh`) registered the LoRA under the same name as the base (`gemma4-agentic-v5-24k` for both), making request routing ambiguous. Its 20/30 result (`runs/hard_subset_v5_s2_cp5_256k_20260711_054517`) must not be cited.
- Clean rerun with distinct names on one server (port 8013, GPU1, `/tmp/start_eval_8013_v4.sh`, requires `PATH=/home/ironbcc/projects/llm/vllm/vllm_env/bin` for ninja/inductor):
  - `v5s2cp5` (v5-s2 checkpoint-5): semantic `22/30`, full `0/30` — `runs/hard_subset_clean_v5s2cp5_256k_20260711_060217`
  - `cp20`: semantic `21/30`, full `0/30` — `runs/hard_subset_clean_cp20_256k_20260711_060217`
  - `gemma4-base-eval` (raw NVFP4 base): semantic `18/30`, full `0/30` — `runs/hard_subset_clean_gemma4-base-eval_256k_20260711_060217`
- Conclusions:
  - v5-s2 did **not** regress vs cp20 (22 vs 21, within noise); both clearly above raw base (18).
  - The absolute scale shifted down vs the historical cp20 reference (25-26/30): this serving path (direct vllm serve + `openai_chat_proxy.py`, no serve.sh wrapper) measures ~4-5 points lower. **The Phase 6 ≥25/30 gate must be re-anchored: compare candidates against a same-day cp20 run on the same server, requiring candidate ≥ cp20 + 3.**
  - Per the decision rule (cp5 < 25/30 absolute), **v6 initializes from cp20** (`adapters/unsloth_agentic_filtered_repair_cp10_to_s10_14336/checkpoint-10`) — also the cleaner choice since cp5 learned from the corrupted dup mixture.
- Eval server left running for reuse: port 8013, vllm PID `332863`, models `gemma4-base-eval` + LoRAs `v5s2cp5`, `cp20`. Stop it (exact PID only) before Phase 5 memory gates free GPU1.

### Audit Correction (2026-07-11): part of the v5 duplication was intentional

- Dedup of the pre-budget parents revealed exact integer replication ratios: `coder_repair_synthetic` 768 rows = 12 unique × 64, `swe_train_oracle_edit_trace` 1,024 rows = 256 unique × 4. These are deliberate replication-based upweights (anchor sources), not corruption.
- Of v5's 3,282 exact dups: ~1,524 were intentional anchor replication; ~1,758 were accidental swe-smith chain-stacking (the real bug).
- `dedup_and_trim_traces.py` gained `--passthrough-sources` so anchors keep their replication while swe-smith is deduped. The Phase 4 "0 exact dups" gate becomes: exact dups must equal `intended_dups_kept` from the dedup manifest.
- v6 anchor corpus built at `data/v6_build/anchor_v6.jsonl` (remote): 4,250 rows = swe-smith 2,458 unique + coder_repair 768 (12×64) + oracle 1,024 (256×4); 2,516 trailing messages trimmed; manifests in `data/v6_build/`.

### Phase 2 Result (2026-07-11): raw source verified; nebius excluded from v6

- Raw `data/unsloth_agentic_24k_train` verified: 15,683 rows, sources swe-smith 12,554 + nebius 3,129, matching its manifest.
- Nebius drop root cause: their assistant messages carry `tool_calls: None` — actions are inline in content (old SWE-agent format). `trace_should_keep` extracts zero edit/verify events, so 0/3,129 pass under every filter relaxation (probe: `/tmp/nebius_probe.py` on host). Not a quality problem; an unconverted-format problem.
- Decision per plan (<200 survivors): nebius stays OUT of v6. v7 option: convert inline actions → bash tool_calls (same pattern as the Kwai mini-swe-agent converter in `phaseD_sft/convert_external_traces.py`), then re-gate.

### Phase 1 Result (2026-07-11): pipeline code done

- `phaseD_sft/dedup_and_trim_traces.py` (trim trailing non-assistant → drop assistant-less/<2-msg traces → sha256 content dedup keep-first; never by instance_id), `phaseD_sft/scan_agentic_dataset.py` CLI, `compact_observations.py` hard-cap patch (escape hatch removed — output never exceeds `max_chars`). 56/56 tests pass (`.venv/bin/python -m pytest phaseD_sft/tests/`). Synced to the remote repo.

## 2026-07-11 Execution Handoff (v6 lane)

This is the canonical continuation point. Runbook: `phaseD_sft/V6_PIPELINE_RUNBOOK.md`. All remote paths relative to `/home/ironbcc/projects/gemma4-31B-Coder`; python = `.venv-train/bin/python` with `PYTHONPATH=$PWD`.

### Completed (with artifacts)

- **Phase 0 — init decision**: v6 initializes from **cp20** (`adapters/unsloth_agentic_filtered_repair_cp10_to_s10_14336/checkpoint-10`). Clean same-server hard30: v5s2cp5 22/30, cp20 21/30, raw base 18/30 (runs `runs/hard_subset_clean_*_256k_20260711_060217`). Eval gate re-anchored: candidate must beat SAME-DAY cp20 by ≥3 on the same server; never compare to the historical 25-26/30.
- **Phase 1 — pipeline code**: `dedup_and_trim_traces.py` (+`--passthrough-sources` for anchor replication), `scan_agentic_dataset.py`, `compact_observations.py` hard cap, `convert_external_traces.py` (kwai + openswe incl. str_replace_editor→bash translation, think folding, stats in manifest file). **93/93 tests** (`.venv/bin/python -m pytest phaseD_sft/tests/ -q` from local repo root). All synced to remote.
- **Phase 2 — sources**: raw `unsloth_agentic_24k_train` verified (15,683 = swe-smith 12,554 + nebius 3,129). Nebius EXCLUDED from v6 (inline-action format, `tool_calls: None`, 0% filter pass — format problem, not quality; v7 candidate via converter).
- **Phase 2b — external data** (on remote under `data/v6_build/`):
  - kwai: `ext_kwai_final.jsonl` 65,994 rows (compacted, 0 dups, system prompts replaced with canonical swe-smith prompt — original demanded fenced-bash which contradicts tool_calls).
  - openswe: `ext_openswe_sweagent.jsonl` 5,980 + `ext_openswe_openhands.jsonl` 6,114 (resolved==1, langs py/rust/cpp/c, editor-calls translated to bash). Downloads at `/media/ironbcc/CrucialX10/datasets/external_traces/`.
- **Phase 3 — mixture assembled**: `data/v6_build/v6_mixture_prebudget.jsonl` = **8,500 rows** (anchor 4,250 + external 4,250: all openswe rust/c/c++ + 2,370 openswe python + kwai fill; openswe system prompts normalized). Manifests: `v6_mixture_manifest.json`, `dedup_manifest.json` (anchor: 1,953 accidental dups removed, 1,524 intended kept, 2,516 trailing msgs trimmed).

### Mixture assembly history (2026-07-11): two false starts, then correct

1. **v1**: selected external rows BEFORE budgeting → of 4,250 selected openswe rows only 101 fit ≤32,768; kwai got zero slots. Discarded.
2. **v2**: fixed selection order (token-prefilter before selecting) but used single-threaded tokenization + a `Pool(16)` with a per-worker `AutoTokenizer.from_pretrained` reload — this is the incident below. Discarded/killed.
3. **v3 (final, correct)**: `/tmp/v6_full_pipeline2.py` — loads the tokenizer ONCE in the main thread, tokenizes all 82,338 candidates (anchor+openswe+kwai) in one `ThreadPoolExecutor(16)` pass (Gemma tokenizer `is_fast=True`, releases GIL, so threads give real parallelism with zero memory duplication), selects, budget-gates, and saves — all in one script, one pass. Completed cleanly: memory stayed at 37-49GB available throughout, swap flat at 20GB.

### ⚠️ Incident (2026-07-11 ~07:07-07:10 UTC): near-OOM from multiprocessing anti-pattern

`multiprocessing.Pool(16)` with tokenizer loaded inside `_init_worker` (post-fork) → 16 independent tokenizer copies instead of 1 shared → host swap grew **55GB→102GB in ~3 minutes**, available RAM dropped to 2GB, workers stuck in `D`-state disk-wait at 5-8% CPU each (NOT actually parallelizing — swap-thrashing). This host has documented hard-reset history under RAM pressure (see 2026-07-10 Hardware Recovery Finding above). Caught via htop (visual, not automated) before it worsened; killed by exact PID; recovered to 37GB available within ~30s.
**Rule for any future CPU-parallel job on this host**: never load a heavy resource (tokenizer/model) inside a `Pool` initializer — load once in the main thread/process and use `ThreadPoolExecutor` if the resource is GIL-releasing (check `tokenizer.is_fast`). Add a `/proc/meminfo` MemAvailable preflight (abort <15GB). Check `free -g` within ~30s of any parallel launch — do not just wait for the job to finish.

### Final v6 dataset (2026-07-11, confirmed)

`data/unsloth_agentic_32k_train_swe_edit_trace_v6_budget32768` — **8,500 rows**, 0 rejected (all pre-filtered to ≤32,768 tokens before assembly):

| Source | Rows |
|---|---|
| swe-smith (deduped anchor) | 2,458 |
| coder_repair_synthetic (anchor, ×64 intentional replication) | 768 |
| swe_train_oracle_edit_trace (anchor, ×4 intentional replication) | 1,024 |
| openswe_rust | 67 |
| openswe_python | 158 |
| kwai_klear_miniswe | 4,025 |

- openswe yield was low: only 225/12,094 rows (1.9%) fit under 32,768 — Qwen3.5 full-agent trajectories run long. Rust/C representation (67 rows) is thin but non-zero — first time this mixture has any non-Python external data.
- `_long16` shard built (worst-case tokens: 32,768 down to 32,600 — right at the cap, good for the memory gate).
- Manifests: `data/v6_build/v6_final_manifest.json`, also copied into the HF dataset dir as `budget_manifest.json` (includes `anchor_dedup` provenance).

### Phase 4 format gate: PASSED (2026-07-11)

`CUDA_VISIBLE_DEVICES=1 .venv-train/bin/python phaseD_sft/verify_gemma_format_loss.py --data data/unsloth_agentic_32k_train_swe_edit_trace_v6_budget32768 --samples 1000` → **`failure_count: 0`**. Log: `/tmp/v6_format_gate.log`.
- 22,680 assistant messages, all in supervised `<|turn>model` spans; `tool_call_open`/`tool_call_close`/`tool_response` balanced at 22,524 each; `fallback_spans: 0` (no template drift); supervised token p50=5,607, p90=12,674, max=20,942.
- Note: this script is single-threaded (nested `apply_chat_template` prefix-stability check per message) — slow (~15+ min for 1000 samples) but not memory-risky; left as-is (existing verification code, not rewritten for speed).

### Cap change (2026-07-11, mid-execution): 32,768 → 34,816

User decision: replace the 32,768 cap with **34,816** (34×1024, follows the project's ×1024 naming convention) BEFORE gating 32,768 at all — one rebuild, one gate cycle, no wasted gate on the superseded cap. Gate A's tier stays 24,576 (unaffected by this change; already built and verified — see below). Only the top-line dataset cap and Gate B target move to 34,816.

- Rebuild launched via `/tmp/v6_full_pipeline3.py` (same proven-safe single-tokenizer + `ThreadPoolExecutor` pattern as the 32,768 build; reuses already-cleaned pools `anchor_v6.jsonl`/`ext_openswe_clean.jsonl`/`ext_kwai_final.jsonl` — no need to redo compaction/dedup/conversion, only re-select + re-budget at the new cap). Log: `/tmp/v6_full_pipeline3.log`.
- Verified safe launch: single PID, tokenizer `is_fast=True`, MemAvailable 39GB at start (vs the 32,768 build's healthy pattern) — confirms the fix from the earlier incident holds under reuse.
- New outputs: `data/unsloth_agentic_32k_train_swe_edit_trace_v6_budget34816` (+`_long16`), manifest `data/v6_build/v6_final_34816_manifest.json`.
- **The 32,768-capped dataset (`..._v6_budget32768`) and its already-passed Phase 4 format gate are now SUPERSEDED — do not train on it.** Phase 4 format gate must be RERUN on the new 34,816 dataset before Phase 5.
- Gate A (24,576) shard was already built and is REUSABLE regardless of top cap: `data/unsloth_agentic_32k_train_swe_edit_trace_v6_budget32768_long16_le24576` (name predates the rename but content is correct — 16 rows, 7,706/8,500 rows were eligible ≤24,576 in the 32,768 build; token lengths 24,576 down to 24,458). Memory/prod health during that build: clean (51GB→47GB available, swap flat, prod green throughout).

### Next steps in order (commands ready)

1. Verify the 34,816 rebuild: `tail /tmp/v6_full_pipeline3.log` for `V6 34816 PIPELINE DONE`; check `v6_final_34816_manifest.json` for row counts (openswe eligibility should rise slightly vs the 32,768 build's 225/12,094).
2. **Rerun Phase 4 format gate** on the new dataset: `CUDA_VISIBLE_DEVICES=1 .venv-train/bin/python phaseD_sft/verify_gemma_format_loss.py --data data/unsloth_agentic_32k_train_swe_edit_trace_v6_budget34816 --samples 1000` → REQUIRED `failure_count: 0`. (Slow, ~15+ min, single-threaded by design — see note below.)
3. Stop eval server before memory gates if it was relaunched: find PID via `ss -tlnp | grep 8013`, kill that exact PID only, confirm GPU1 ≈ 2 MiB. (It was already stopped once this session — PID 332863, base `gemma4-base-eval` + LoRAs `v5s2cp5`/`cp20`; relaunch script `/tmp/start_eval_8013_v4.sh` has the required ninja PATH fix.)
4. **Phase 5 memory gates**: Gate A (24,576) shard already exists and is reusable — just run the training memgate on it (`ALLOW_64G_TRAINING=I_ACCEPT_BOUNDED_SWAP ... MAX_SEQ=24576` per the runbook, data = the le24576 shard). Gate B now targets **34,816** on the new dataset's long16 shard (update `/tmp/v6_memgate.sh`'s Gate B `MAXSEQ`/`DATA` before running, or invoke the bounded launcher directly). Record MemoryPeak/SwapPeak/GPU1 VRAM/prod health per gate. Gate B OOM → fallback: rebuild budget at 24,576 (skip 34,816 entirely), proceed at 24k. Never raise cgroup caps.
5. **Training** (only after both gates): bounded launcher, init cp20, `MAX_SEQ=<highest passed>`, `TRAIN_STEPS=5 WARMUP_STEPS=1 TRAIN_LR=2e-5`, unit `swe-edit-v6-32k-s1` (rename to reflect 34k if desired), data = `data/unsloth_agentic_32k_train_swe_edit_trace_v6_budget34816`.
6. **Phase 6 eval gates**: serve checkpoint at 256k (distinct LoRA names!), hard30 vs same-day cp20 (need ≥ cp20+3), then the **30-case** SWE-Lite smoke (≥18/30 non-empty patches, <10% format errors, median first edit < cmd 10). Do not run Verified until it passes.

### Training-set token distributions (measured 2026-07-11, Gemma-4-31B tokenizer)

Per-source rendered-token length over the cleaned pools (before the 4,250 external cap / budget filter):

| Source | Rows in pool | min | p50 | p90 | p99 | max | mean |
|---|---:|---:|---:|---:|---:|---:|---:|
| anchor_v6 (swe-smith+coder_repair+oracle) | 4,250 | 131 | 6,056 | 16,131 | 22,957 | 24,387 | 7,020 |
| kwai_klear_miniswe | 65,994 | 4,315 | 17,183 | 28,780 | 43,726 | 113,620 | 18,618 |
| open_swe_traces_qwen35 | 12,094 | 19,765 | 53,180 | 71,420 | 94,289 | 164,045 | 54,787 |

Key shape facts: the **anchor is short** (median 6k, everything ≤24,387 — fully captured at any cap ≥24,576). **kwai is medium** (median 17k). **openswe is very long** (median 53k — 1.5× our 34,816 cap; even the shortest openswe row is 19,765 tokens).

### What each context cap costs — eligibility yield by source

Rows that FIT (≤ cap) per source, at candidate caps. This is the concrete "what we lose" table:

| Cap | anchor (/4,250) | kwai (/65,994) | openswe (/12,094) | est. GPU1 VRAM* |
|---|---|---|---|---|
| 14,336 | 3,615 (85%) | 21,968 (33%) | 0 (0%) | ~34 GB |
| 18,432 | 3,990 (94%) | 37,535 (57%) | 0 (0%) | ~40 GB |
| 24,576 | 4,250 (100%) | 53,551 (81%) | 16 (0.13%) | **53 GB (measured)** |
| 32,768 | 4,250 (100%) | 62,395 (95%) | 217 (1.79%) | ~66 GB |
| **34,816 (current)** | **4,250 (100%)** | **63,333 (96%)** | **380 (3.14%)** | **~68 GB** |
| 40,960 | 4,250 (100%) | 64,970 (98%) | 1,457 (12%) | ~76 GB |
| 49,152 | 4,250 (100%) | 65,714 (99.6%) | 4,392 (36%) | ~87 GB |
| 65,536 | 4,250 (100%) | 65,964 (99.95%) | 9,868 (82%) | >98 GB (infeasible) |

*VRAM estimated from the one measured point (24,576 → 53.4 GB peak) at ≈1.35 GB per 1k tokens above a ~20 GB fixed base (QLoRA 4-bit 31B, bsz=1, grad checkpointing). Linear extrapolation from a single measurement — treat 32k+ figures as estimates until Gate B measures 34,816.

**What we actually lose at the current 34,816 cap:**
- **anchor**: nothing (100% captured at ≥24,576).
- **kwai**: ~2,661 rows (4%) — negligible; the long tail of mini-swe-agent traces.
- **openswe**: 11,714 of 12,094 rows (**97%**) — but this is bounded by design anyway (external is capped at 4,250 and kwai backfills), so the *effective* loss is **openswe language/Rust diversity, not row volume**. At 34,816 we get 98 Rust + 298 Python openswe. Raising the cap is the only lever that adds Rust openswe: 40,960 → ~12% openswe yield, 49,152 → ~36%. Below 24,576 you also start dropping anchor rows (the no-regress signal), so **24,576 is the practical floor** and **~49,152 is the practical VRAM-bound ceiling**.

### Machine capacity (measured 2026-07-11)

- **Host RAM = 64 GB nominal / 61 GiB usable** (confirmed via `free -h`; the historical wall was RAM-related, per box owner). GPU1 (training) = RTX PRO 6000 Blackwell, **97.9 GB VRAM**.
- **The binding constraint is GPU VRAM for context, host RAM for stability** — they are independent. Host RAM during training is ~flat vs. context (~36 GB cgroup peak, dominated by the pre-tokenized dataset + torch/unsloth CPU side, NOT by sequence length); the memory that grows with context lives on the GPU. So more host RAM buys swap/prod-coexistence headroom, not a higher context ceiling.
- Gate A measured (MAX_SEQ=24,576, 1 step, cp20 init): **VRAM peak 53.4 GB**, cgroup RAM peak 36 GB, cgroup swap peak 615 MB (host swap ~15 GB shared w/ prod), production green throughout. No threshold breach.
- **Context ceiling ≈ 49-50k tokens** before VRAM saturates at this config; **34,816 sits at ~68 GB VRAM (est), comfortable**. Beyond ~50k needs bsz/grad-accum or activation-offload changes, not just more host RAM.

### DECISION 2026-07-11: cap → 49,152 + openswe compression, handed to Codex

- User raised the target cap to **49,152** (from 34,816) to recover openswe Rust/language diversity (36% yield vs 3%), and asked to **compress openswe tokens** to fit even more, then hand execution to a Codex agent via Orca orchestration, with **frequent smoke tests before any full training**.
- Full executable spec: **`phaseD_sft/V6_49K_CODEX_HANDOFF.md`** (main goal, compression strategy, rebuild, gates, smoke cadence, machine-safety rules, fallback ladder). Codex owns execution from here.
- Compression steer (baked into the handoff): compress **observations (conditioning, masked from loss)** only — tighten `compact_observations` + truncate giant file-views + window long trajectories. Do NOT article-drop/obfuscate assistant turns (they are loss-supervised; lossy transforms corrupt the CoT/code signal). This is the technically correct reading of the "obfuscation/article-drop" idea.
- 49,152 VRAM is UNPROVEN (~87 GB est / 97.9 GB) — Gate B is the critical gate; fallback ladder 49,152 → 40,960 → 34,816 (built) → 24,576 (gated). The 34,816 dataset + its passed format gate are retained as the safe fallback.

### 49k attention saga — resolution status (2026-07-11, late session)

Full analysis + run-by-run history: `phaseD_sft/V6_49K_RETROSPECTIVE.md`. Compressed state:

- **FINAL root cause of all 49k OOMs**: Gemma-4's 10 global layers use `global_head_dim: 512` (4 KV heads, V shares K's projection). **No fused attention kernel supports head_dim 512 on sm_120** — fa2/fa3 cap at 256 universally; xformers cutlassF is gated ≤sm90 in our build; sm_120's ~101 KB shared memory can't tile hd512 regardless (web-verified; Blackwell-workstation-specific — H100's 228 KB SMEM is why unsloth's 60k claims hold there). Dense math was the only executable path → 24,576 fit (~36 GB transient), 49,152 didn't (128 GB).
- Secondary blockers found and beaten along the way (each verified): stale launcher swap-guard masked the real error (user caught it; removed); flex backward exceeds sm_120 SMEM even at hd 256 (fixable via `kernel_options` small tiles, but unsloth runtime contexts degrade compiled flex to eager/dense anyway); unsloth globally replaces `torch.utils.checkpoint.checkpoint` AND resets `attn_implementation` to sdpa at load; the fp32 logits tensor at 262k vocab × 46k seq = 45.28 GiB was the LAST wall after attention was solved.
- **Solution stack (all components independently verified)**:
  1. Sliding layers (50×, hd 256): xformers `memory_efficient_attention` + `LocalAttentionFromBottomRightMask(1023,0)` — 8.3 GiB probe.
  2. Global layers (10×, hd 512): **`phaseD_sft/chunked_global_attention.py`** — exact q-blockwise attention, custom autograd.Function, FA2-style recompute backward. Row-block softmax decomposition is mathematically exact. Tests `phaseD_sft/tests/test_chunked_global_attention.py`: 7/7 on the box (fwd+grad vs dense reference, jagged, hd512, bf16, causality). Scale probe @ (1,32,46365,512): 3.4 s fwd / 7.2 s bwd / 42.4 GiB.
  3. Loss: `UNSLOTH_RETURN_HIDDEN_STATES=1` (moe patch returns hidden states, no logits) + custom fused `compute_loss` via `cut_cross_entropy.linear_cross_entropy` (softcap 30.0, shift=True, ignore_index −100, num_items normalization) — logits never materialized.
  - Wiring: `/tmp/verify_49k_step.py` on host — registers `xformers_sm120` in `ALL_ATTENTION_FUNCTIONS` (hd>256 → chunked, else xformers), flips `_attn_implementation` post-load, installs fused CE on `Trainer`. Env: `UNSLOTH_COMPILE_DISABLE=1 UNSLOTH_RETURN_HIDDEN_STATES=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`.
- **Run 8 PASSED (2026-07-11 19:36) — Gate B @ 49,152 CLEARED.** First 49k training step ever completed on sm_120. Ground-truth verified: `adapters/v6_gateB_49152_verify_claude/checkpoint-1/adapter_model.safetensors` (979 MB) + optimizer.pt + trainer_state.json written; log `[done] adapter saved`. Telemetry: **VRAM peak GPU1 82,300 / 97,900 MiB**; **optimizer step wall 2885 s (48 min, 16 microsteps @ ~180 s)**; loss 0.017–0.022, final `train_loss 0.02034`, `grad_norm 0.076`; host RAM peak 43/61 GB (18 avail); prod ports 8000/8101/8103/8104 green throughout. Full stack (xformers sliding hd-256 + chunked-global hd-512 + fused CCE) works end-to-end. Log `/tmp/verify_49k_step8.log`.
- **(1) DONE (2026-07-11): wiring promoted out of /tmp.** New module `phaseD_sft/sm120_attention.py` (`register_sm120_attention` routing by head_dim, `swap_to_nonreentrant_checkpoint`, `install_fused_ce_loss`, `require_return_hidden_states` guard) + opt-in `--sm120-attn` flag in `train_rust_lora.py` (inert unless set; requires `UNSLOTH_RETURN_HIDDEN_STATES=1`). Faithful to the Gate-B-passing config; dead Plan-A flex/dynamo remnants dropped. Verified on box: py_compile clean, import/registration/fused-CE-patch/xformers/cce all wire, env-guard raises when flag unset. Synced to host repo. NOT committed (no user imperative).
- **(2) DONE (2026-07-11): Codex re-engaged (user-gated GO given).** Delivered via `orca terminal send` to `term_0c1b07ed…` (gpt-5.6-terra, ~/projects/llm); confirmed received + read `phaseD_sft/CODEX_49K_GO.md` and started Step A. Full brief in **`phaseD_sft/CODEX_49K_GO.md`** (§0 updated goal, §1 what changed, §2 hard constraints, §3 exact nohup commands Step A→B→C, §4 reporting, §5 verification method+code+evidence+scripts audit). Codex executes: 1-step smoke via `--sm120-attn` → 5-step train on full 49k → hard30 (≥cp20+3) → 30-case SWE-Lite smoke. Uses the verified plain-nohup launch (NOT the bounded systemd launcher — §5.6).
- **(3) DONE (2026-07-12 02:35): first v6 49k adapter trained.** `adapters/swe_edit_v6_49k_s1/checkpoint-5/adapter_model.safetensors` (979,558,760 B) via `--sm120-attn`. Manifest verified: data `budget49152` (full), init cp20 checkpoint-10, max_seq 49152, lr 2e-5, 5 steps, load_4bit. Gate A smoke passed first (`swe_edit_v6_49k_smoke`, byte-identical adapter). Losses 0.02325→0.01905 (down-trend); per-step wall 695/665/949/746/573 s; GPU1 freed to 2 MiB; ports green throughout; host RAM stable (RSS ~45 GB flat, avail plateaued ~5.8 GB = reclaimable cache, swap 15 GB/256 GB, earlyoom off — verified not a leak).
- **(4) PASSED (2026-07-12): same-day hard30.** The isolated GPU1 `v6-eval` unit serves the NVFP4 base under `gemma4-v6-49k-eval-base` with distinct LoRA names `v6_49k_s1` and `cp20_same_day`, avoiding the invalid base/LoRA-name collision. Its first start failed before listening because FlashInfer's sm120 FP4 JIT ran `ninja` without the vLLM virtualenv bin directory on `PATH` (`FileNotFoundError: ninja`), not because of VRAM, data, or a LoRA; the retry retained `MemoryHigh=16G`, `MemoryMax=24G`, and `MemorySwapMax=8G`, adding only `PATH=/home/ironbcc/projects/llm/vllm/vllm_env/bin:...` to expose the already-installed binary. It became ready on port 8012; candidate `v6_49k_s1` scored **26/30 semantic, 1/30 full** (`runs/hard_subset_v6_49k_s1_256k_20260712_024643`), and same-day cp20 scored **23/30 semantic, 0/30 full** (`runs/hard_subset_cp20_same_day_256k_20260712_024643`). The candidate exactly clears the required **+3 semantic** margin; all production ports remained green.
- **(5) FAILED (2026-07-12): 30-case SWE-Lite behavior smoke.** The first attempt (`runs/smoke_v6_49k_s1_lite_0_30_256k_20260712_025050`) stopped before case 1 due to the wrapper's relative-config bug and has no behavioral result; `smoke_single.sh` now resolves its adjacent `swebench_edit_first.yaml` correctly (17 local regression/clamp tests; host syntax check). The valid 30-case run is `runs/smoke_v6_49k_s1_lite_0_30_256k_20260712_025356`: 30 submitted predictions and trajectories, but **0/30 non-empty `model_patch` fields**, 376 format-recovery events over 595 assistant actions (**63.19%**), and source edits in only 2/30 trajectories (indices 4 and 13; median 8.5 across those two). It fails the ≥18/30 non-empty and <10% format-error gates; scoring also ended with a Docker `NotFound` / `No instances to run` reporting failure, so no PASS_TO_PASS/FAIL_TO_PASS evidence exists. `v6-eval` was stopped exactly on GPU1 after artifacts were captured; GPU0 production ports stayed green. **Do not scale this adapter; rebuild/reweight the behavior data or harness before a fresh smoke.**

- **(6) ROOT CAUSE of the 0/30 found + fixed (2026-07-12): harness clamp, not model/data.** Diagnosed no-GPU (user chose "diagnose parity first"): serving parser is CORRECT (`--tool-call-parser gemma4` + `--reasoning-parser gemma4`), so not a parser mismatch. Real cause = `phaseH_eval/model_clamps.py DEFAULT_MAX_TOKENS=512` truncating tool calls (verified: 155/538 = **29% of generations hit exactly 512**; truncated turns lose their `<|tool_call>` markup → parser returns no tool call → "Tool call error: No tool calls found" → 63.19% recovery loop). Compounded by **thinking OFF** (gemma4 template defaults `enable_thinking=false`; 0 reasoning_content in all trajs) and the docker-scoring `NotFound` crash. **Fixes applied + tested (15/15 phaseH_eval tests green) + scp'd**: (a) `DEFAULT_MAX_TOKENS` 512→**32768**; (b) `vllm_direct_model.py::_query` now sends `extra_body={"chat_template_kwargs":{"enable_thinking":True}}` → **thinking ON** (template renders `<|channel>thought` + `<|tool_call>`, gemma4 parsers split both). Rerun brief: **`phaseD_sft/CODEX_49K_RERUN.md`** (docker-scoring fix first, re-serve same adapter+config, rerun, report format-rate/patch-count/edit-reach delta). Gate unchanged: ≥18/30 non-empty, <10% format errors.
- **(7) THINKING FIX — VERIFIED SERVED END-TO-END (2026-07-12). Supersedes the premature "needs retrain" read.** Enabling `enable_thinking=true` ALONE produces degenerate `<|turn>model` garbage (base skips to a bare tool call, both LoRAs loop turn-markers) — because the stock chat template (bundled AND vLLM's official `tool_chat_template_gemma4.jinja` — identical) NEVER opens the `<|channel>thought` channel under `enable_thinking=true`; it ends at `<|turn>model\n` and relies on the model to self-open, which this checkpoint won't. PROOF the model is capable: raw `/v1/completions` with the thought channel hand-opened (`<|channel>thought\n`) → perfect reasoning on **base AND v6**. **Fix = template patch** (`phaseH_eval/tool_chat_template_gemma4_thinkopen.jinja`): under `enable_thinking=true` emit `<|channel>thought\n` (open) instead of nothing. Served-confirmed on 8012 (patched template + `--default-chat-template-kwargs '{"enable_thinking": true}'`, GPU1 only, prod green): v6 probe returned `message.reasoning` = 265-char CoT ("…fix a bug in WCS.wcs_pix2world… explore the codebase…") + a clean structured `bash` tool_call. **No retrain.** `vllm_direct_model.py` enable_thinking wiring was reverted (server default handles it now). 8012 currently serving with this config (pid 1487744).
- **(8) FAILED (2026-07-12): clamp + thinking + Docker rerun.** Docker was repaired surgically: `/var/lib/docker` had 94 GiB free, but a ghost dead-container metadata directory (`66a4…`) was removed while `docker.service` was stopped; a one-instance SWE-Lite preflight then resolved 1/1 and wrote its report with zero dead containers. The already-running patched-template GPU1 server (PID 1487744, port 8012, `v6_49k_s1`) was reused unchanged. Corrected smoke `runs/smoke_v6_49k_s1_rerun_lite_0_30_256k_20260712_044535` completed all 30 trajectories/predictions, but has **0/30 non-empty `model_patch` fields**, 27 `RepeatedFormatError` and 3 submitted exits; scorer found no non-empty instances to run. The clamp diagnosis is confirmed but insufficient: completion-token distribution has p50 80, p90 151, max 638, **0 at 512 and 0 at 32,768**; all 30 trajectories have reasoning and all 149 recorded actions are structured bash calls (146 single-call responses, one three-call response). Yet 150 `No tool calls found` recovery events occurred in addition to 147 successful tool-call responses (response-level error rate 150/297 = **50.5%**), and only 1/30 trajectory reached a source edit (command 3). This is a second **thinking-template/markup-fidelity or Mini-SWE integration** failure on untruncated generations, not evidence to rebuild data; do not scale or retrain until raw failed responses are captured and that integration is fixed.
- **(9) SECOND BUG CAUGHT + FIXED (2026-07-12), verified.** Captured the raw failing responses by REPLAYING each conversation up to its format-error point (mini-swe drops failed responses; temp-0 replay reproduces them). Result: on turns AFTER a tool observation the model emits `<|turn>model` marker loops (~50%). Cause: the chat template's generation-prompt block is gated OFF when `prev_message_type == 'tool_response'` (line 348: `if prev != tool_response and != tool_call`), so post-observation turns get NO `<|turn>model\n` / thought opener and the model flails. **Fix = template v2** (`phaseH_eval/tool_chat_template_gemma4_thinkopen_v2.jinja`): added `{%- elif prev == 'tool_response' -%}` opening `<|channel>thought\n`. Verified two ways: (a) render+raw-completion on the exact astropy-12907/14182/14365 failing contexts — v1 `<|turn>model` garbage → v2 clean reasoning; (b) served end-to-end on 8012 — post-tool-result probe returns `message.reasoning` ("Okay, I see the function definition…") + a structured `bash` tool_call. **8012 now serves v2** (pid 2100381, GPU1, prod green). Full fix stack live: 32k clamp + thought channel opened on user turns AND after tool_response.
- **(10) rerun-2 STALLED (2026-07-13), cause verified + fixed: clamp retuned 32k→4096.** Rerun-2 vs the v2 server ran 4h17m with only 2/30 cases done — vllm metrics showed 1 request generating continuously at ~45 tok/s (prompt throughput 0) = degenerate thinking loop with 32k tokens of rope (~12 min per runaway turn). The 32k clamp fixed truncation but removed fast-fail. `model_clamps.py DEFAULT_MAX_TOKENS` → **4096** (fits reasoning+tool call, p90 need ~151; runaway loops now fail in ~90 s). Synced, 15/15 tests green. Codex: kill stalled launcher (exact PIDs 2266323/2266329), keep 8012 v2 (pid 2100381) running, relaunch rerun-3 fresh out-dir.
- **(11) RERUN-4 VERDICT (2026-07-13, 29/30 near-final, workers=8, v2 template + 4096 clamp): HARNESS FIXED, MODEL FAILS BEHAVIOR GATE GENUINELY.** (rerun-3 was superseded — relaunched at workers=8 as rerun-4 minutes in.) Format/no-tool-call rate **5.5%** (32 events / 581 responses; was 50.5%) → <10% gate PASSES — template v2 + 4096 clamp fully fixed the harness. But non-empty patches **3/29** (django-11049, django-11583, astropy-7746; gate ≥18/30 FAILS); edit-reached trajectories 5/29 (median first edit 8 among those); **24/29 trajectories explore/read but never make a source edit**. With the harness clean, this is REAL model-behavior evidence: v6 improved single-turn semantics (hard30 +3) but did not instill multi-turn edit-decisiveness — the read-loop gap the swe-edit-trace lane targets. Run: `runs/smoke_v6_49k_s1_rerun4_lite_0_30`. Eval infra all validated: thinking (v2 template), tool-call parsing, 4096 clamp, docker scoring, workers=8 parallel (~30 min wall).
- **(12) PROBE FINDINGS (2026-07-13): edit competence EXISTS; decisiveness + greedy perseveration are the gaps.** Patch inspection: of the 3 non-empty patches, **2 are real fixes** (astropy-7746 = textbook-correct empty-input guard; django-11049 plausible message-format fix) and 1 destructive (django-11583 deleted all 600 lines of autoreload.py via a rewrite). No-edit failure signature (astropy-12907): model ran the IDENTICAL `sed -n '10,99p'` command 4× in a row until the harness's repeated-read guard force-submitted an empty diff — classic **temp=0 greedy perseveration** (smoke runs temperature=0). Two-lever hypothesis: (a) decoding — temp>0 breaks identical-command loops; (b) data — edit-first/transition traces. **Discriminator probe directed to Codex**: identical 30-case run at `temperature=0.7` (`runs/smoke_v6_49k_s1_t07_lite_0_30`, workers 8). Edit-reach jumps → decoding lever is real, include in gate config; unchanged → pure data gap. Also awaiting rerun-4 auto-scoring (resolved count for the 3 patches).
- **(13) t07 DISCRIMINATOR RESULT (2026-07-13): decoding lever REAL but insufficient alone.** `runs/smoke_v6_49k_s1_t07_lite_0_30` (30/30, temp 0.7, workers 8): edit-reached **10/30** (vs 5/29 at temp 0 — ×2), non-empty patches **7/30** (vs 3/29 — ×2.3), format rate ~7.3% (54/737, still <10% ✅). Temp>0 breaks perseveration loops. But 7/30 ≪ 18/30 gate → data lane still required. Rerun-4 scoring: 1/3 patches resolved (1/29 overall). Uncapped-request anomaly: case django-11848 escaped the 4096 clamp via mini-swe/litellm kwargs path (one case; generation stopped on client kill; logged, low priority). **Lane mix conclusion: (a) eval/gate config adopts temperature 0.7; (b) behavior-data rebuild targets edit-decisiveness** (edit-first traces, oracle-edit upweighting, tighter anti-perseveration filters e.g. max_read_streak, no-identical-command-repeat in supervised traces; optional thinking-channel rendering — task #6), retrain from cp20, re-gate at temp 0.7.
- **(14) TRAINING-DATA AUDIT (2026-07-13): the v6 mixture TEACHES the late-edit failure.** `/tmp/audit_edit_behavior.py` on the box (per-source first-edit index over the v6 49k mixture): externals = 50% of rows and model read-heavy late edits — kwai_klear median first-edit **16** (26% ≤10, read-streak 6), open_swe_qwen35 median **16** (25% ≤10, read-streak 9) — while anchors are edit-decisive: swe-smith median **7** (82% ≤10), oracle_edit median **2** (100% ≤10). The adapter reproduces its data. **Mixture v7 directed to Codex (CPU-only, STOP before training)**: filter externals to first-edit ≤10 AND read-streak ≤5 AND no identical-consecutive repeats (~1,000 of 4,250 survive), oracle_edit ×2 upweight (manifest-marked), anchors unchanged, standard chain (dedup_and_trim → budget 49152 → format gate failure_count 0), then re-audit — target overall first-edit median ≤8, ≥70% ≤10. Verification by coordinator before any training launch.
- **(15) t07 SCORED (2026-07-13): resolved 4/30** (`astropy-6938, django-10914, django-11099, django-11179`; report `openai__v6_49k_s1.smoke_v6_49k_s1_t07.json` in repo root) vs **1/29 at temp 0** — decoding lever alone = 4× resolve. Compounds with the v7 data lane.
- **(16) v7 BUILT + VERIFIED (2026-07-13).** `data/unsloth_agentic_train_swe_edit_trace_v7_budget49152`: 5,798 rows (budget 0 rejected, max row 48,951 tok). Composition: swe-smith 2,458 + oracle_edit **×2 = 2,048** (passthrough-marked) + coder_repair 768 + filtered externals kwai **281**/1,750 + openswe **243**/2,500 (filters: first-edit ≤10 AND read-streak ≤5 AND no identical repeats; drop reasons per-filter in `data/v7_build/v7_mixture_manifest.json`). Dedup clean (0 accidental). **Coordinator audit PASSED all targets**: first-edit median per source 2/5/7/7 (targets ≤8 ✓), externals 100% ≤cmd 10 (target ≥70% ✓), read-streaks ≤4 ✓, zero identical repeats in externals ✓. Anchors now 91% of mixture (v6: 50% late-edit externals). Everything committed + pushed (`a202962`). Builder: `phaseD_sft/build_v7_edit_decisive_mixture.py` + test.
- **(17) FORMAT GATE GREEN (2026-07-13): failure_count 0 / 1000 samples** (`/tmp/v7_format_gate.log`). Training sequence fired.
- **(18) v7 FULL GATE SEQUENCE RESULTS (2026-07-13): every metric improved, behavior gate still short.** 1-step smoke PASS (284 s, loss 0.0174 — v7 rows shorter than v6 worst-case). 5-step PASS (`adapters/swe_edit_v7_49k_s1/checkpoint-5`, losses 0.017–0.021). hard30 same-day: **v7 22/30 vs cp20 19/30 = +3** ✓ zero-regress (absolute scale shifted by thinking-on serving — re-anchor rule applied; runs `hard_subset_v7_49k_s1_256k_20260713_0628` / `hard_subset_cp20_same_day_v7_256k_20260713_0634`). **30-case SWE-Lite temp 0.7** (`runs/smoke_v7_49k_s1_t07_lite_0_30`): patches **10/30** (v6-t07: 7), **resolved 7/30** (v6-t07: 4; temp-0 v6: 1) — **70% patch precision**, edit-reach **14/30** (v6-t07: 10), format **5.5%**. Gate ≥18/30 not met, but monotonic v6→v7 improvement on every axis from only **80 training rows consumed** (5×16). Conclusion: mixture works; model needs more steps.
- **(19) v7-s2 (30 steps) TRAINED but cp30 REGRESSED semantics.** Training clean (checkpoint-30, losses 0.014–0.021 converged, ports green) but hard30 **cp30 = 19/30 = anchor+0, gate FAILED** (`runs/hard_subset_v7_49k_s2_256k_20260713_0953`; s1-cp5 was 22/30 same morning). Overtraining on the small mixture (oracle ×2 = 35% of rows; 480 rows seen at 2e-5). Smoke correctly held.
- **(21) cp15 SMOKE VERDICT (2026-07-13): v7 mixture PLATEAUS at half-gate.** `runs/smoke_v7s2cp15_t07_lite_0_30`: patches **13/30** (new best; s1-cp5 10/30), resolved **6/30** (s1: 7/30), edit-reach 11/30 (s1: 14/30), format **3.6%** (best yet). cp15 ≈ cp5 within n=30 temp-0.7 sampling noise → the v7 mixture (524 external rows) plateaus at ~10-13 patches / 6-7 resolved vs gate 18. More steps exhausted as a lever (cp30 regressed). **v8 lane opened: mine the FULL converted pools** (`ext_kwai_final.jsonl` 65,994 rows 4.4 GB + `ext_openswe_compact1200_clean.jsonl` 12,094 rows — both already converted/compacted on box; v7's externals came only from 24k-budget-capped subsets). Mining with the identical proven filters (first-edit ≤10, read-streak ≤5, no repeats) → `data/v8_build/{kwai,openswe}_editfirst.jsonl` + manifests. **MINE RESULTS: kwai 10,247/65,994 (15.5%) + openswe 936/12,094 (7.7%) = 11,183 candidates (21× v7's 524).** Token-budget 49152 over all 11,183 in flight (`/tmp/v8_ext_budget.log`). Assembly spec (handed to Codex, coordinator drives if it stays usage-limited): token-aware select externals to ~2,800 cap → mixture ≈ swe-smith 2,458 + oracle ×1 1,024 + coder_repair 768 + externals ≈ 7,050 → dedup → budget → audit (medians ≤8) → format gate (0 failures) → coordinator gates before training.
- **(22) v8 BUILT + ALL GATES GREEN (2026-07-13), TRAINING FIRED.** Chain: budget 10,958/11,183 → token-aware select **2,800 externals covering all 548 repos** (kwai 2,086 + openswe 714, shorter-first p50 8,535 tok) → mixture 7,050 (oracle **×1**) → dedup 0 accidental → final budget 7,050/7,050 → saved `data/unsloth_agentic_train_swe_edit_trace_v8_budget49152` → audit PASS (medians 2/5/7/7, externals 100% ≤10, streaks ≤4) → format gate **failure_count 0** (after fixing Codex's --workers refactor which dropped `rendered_assistant_turn_spans` + `label_tokens_for_spans` imports — Codex added a fixture test). **Training: 30 steps from cp20, save-5, `--sm120-attn`, `adapters/swe_edit_v8_49k_s1`** → then 5-checkpoint hard30 sweep (fresh anchor first, winner ≥+3 preferring more steps) → winner SWE-Lite temp 0.7 (TEMPERATURE env var, not -c). Beat: 13/30 patches / 7/30 resolved / 14/30 edit-reach; gate ≥18/30.
- **(23) v8 TRAINED + SWEEP (2026-07-13 18:19): checkpoint-30 clean (losses 0.014–0.019). hard30 sweep `runs/hard_subset_*_v8sweep_256k_20260713_1819`: anchor cp20_same_day **22**, cp10 22, cp15 20, cp20 20, **cp25 23**, cp30 pending. **METHOD FINDING: hard30 session noise ±3-4** — the SAME cp20 anchor scored 19 (0634) / 18 (1133) / 22 (1819) across three same-day serves, so the +3 promotion gate ≈ 1σ and v7's "+4" was within noise. REVISED GATE POLICY: hard30 = regression FLOOR (candidate ≥ anchor−2 acceptable, hard fail below), SWE-Lite behavior smoke = the decisive promotion gate; for any final/scale promotion run hard30 ×3 or n=90. cp25 (+1) passes the floor → advance best of cp25/cp30 to the smoke.
- **(28) LANES OPENED (2026-07-14): Rust baseline fired; C++ prereq RESOLVED.** Python patch-gate pass unblocked the charter ladder. **Rust**: task #8 → Codex — raw-base baseline over Multi-SWE-Rust 239 (smoke-5 first, phaseA_scaffold runner, measurement-only). **C++**: build-tool inspection DONE — multi-swe-bench has **257 C/C++ instances** in 8 repos (`c/`: zstd 29, jq 17, ponyc 82; `cpp/`: Catch2 12, fmt 41, nlohmann/json 55, simdjson 20, httplib 1; per-repo JSONL under `c/`+`cpp/` on HF — the parquet default config errors, load via hf_hub_download per file). Schema = same PR-JSONL the Phase-C T3 loaders normalize; CMake+make families, mswebench image pattern applies. C++ lane STAGED: 257 instances at `data/mswe_cpp_prs/` (+manifest), T3 loaders synced to box, `normalize("cpp", …)` validated **257/257**. Remaining: per-repo image builds + build_cmd matrix (cmake vs make), then raw-base baseline.
- **(27) RLVR A/B VERDICT (2026-07-14): RL nearly doubled behavior; GRPO beat Dr.GRPO on our task; resolve quality unchanged.** Both 100-step runs completed on identical config (only `--loss-type` differs). Final table (30-case SWE-Lite temp 0.7, coordinator-verified from preds+scoring): **SFT v8cp30** 11/30 patches / 6 resolved / 14 edit-reach / 3.7% fmt → **Dr.GRPO** (`runs/smoke_swe_drgrpo_v1_t07_lite_0_30`) 17/30 / **7** / 23 / 6.7%, hard30 19 vs anchor 20 ✓floor, curve had one step-14 blowup (loss 11.0, grad_norm 1587, KL 275) → **GRPO** (`runs/smoke_swe_grpo_v1_t07_lite_0_30`) **20/30 ✓patch-gate** / 6 / **25** / 10.0% boundary (per-traj metric; per-response metric used earlier — compute both identically before ruling), hard30 18 vs 20 ✓floor, smooth curve. Mean shaped rewards near-identical (0.2095 vs 0.2050). **Reads**: (a) edit-decision RLVR works — patches 11→20, edit-reach 14→25; (b) plain GRPO won patches; Dr's step-14 spike likely cost its margin; (c) **resolve stalled at 6-7 everywhere and patch precision fell 55%→30%** — parse-only reward bought decisiveness, not correctness. **Top lever: reward phase 2 — docker apply/test-verified 1.0 tier** (handoff §5.1). Also diagnosed: empty-finalization loop failure signature (patch.txt without source diff) → harness self-retry / finalization reward as cheap adjunct. Merged models: `/media/.../merged/swe_drgrpo_v1`, `swe_grpo_v1` (delete after promotion decision). Promotion + next-lever authorization = USER decision.
- **(26) LANE HANDED TO CODEX (2026-07-14, coordinator token budget).** Full brief: **`phaseE_rl/CODEX_RLVR_HANDOFF.md`** (commit ad5db7a). Codex owns: Dr.GRPO s2 to completion → add `--loss-type` arg → identical standard-GRPO comparison run (user-authorized) → 3-way merges → hard30 floor + SWE-Lite temp-0.7 evals of BOTH → comparison table. Dr.GRPO s2 was healthy at handoff (varied completion lengths 46-708 — no cap-pinning; rewards 0.1-0.4; saves every 25). GRPO-vs-DrGRPO context: user asked "why not DrGRPO"; coordinator wrongly killed the running GRPO on a question-not-directive (lesson recorded in global instructions) — the comparison run restores the A/B. Ops criticals in handoff §3 (trl child holds 90GB after parent kill; GPU-clear gate before relaunch).
- **(25) RLVR LOOP ALIVE (2026-07-14): GRPO generates coherent agent decisions with discriminating rewards.** 11 smoke iterations to a working loop; blockers fixed in order: template path → `Gemma4ClippableLinear` unpeftable (manual fp32 merge into inner linears, 410 matrices, assert-gated) → `mm_token_type_ids` (MM class demands it) → MM decode shapes → text-class load leaves weights RANDOM (checkpoint keys don't map — caught via "newly initialized" warning + token-soup debug artifact) → **graft: load MM class, transplant language tower + lm_head into empty text shell** (meta-assert + sanity-gen tripwires) → trl ignores model.generation_config (grpo_trainer.py:1418; use `GRPOConfig.generation_kwargs.eos_token_id=[1,49,106]` — 49=`<tool_call|>`, 106=`<turn|>`) → zombie child held 92GB after parent kill (kill exact child PID). **Working state**: completions = real reasoning + parseable `call:bash{{...}}`; rewards 0.2 (read) / step-means up to 0.4 (edits in groups) → GRPO gradient exists. Script `phaseE_rl/grpo_swe_edit_decision.py`; dataset `data/rlvr_swe_decision_v1.jsonl` (2k stall contexts; v2 rebalanced build with per-source caps directed to Codex). NEXT: smoke completes+saves → 100+ step run on v2 → eval: hard30 floor + SWE-Lite temp 0.7 vs SFT ceiling 11-13/30 patches.
- **(24) v8 VERDICT (2026-07-13): SFT LEVER EXHAUSTED.** cp30 hard30 **25/30** (best ever, anchor 22). Smoke `runs/smoke_v8cp30_t07_lite_0_30`: patches **11/30**, resolved **6/30** (astropy-6938, django-10914/11001/11049/11133/11620), edit-reach 14/30, format 3.7%. v8cp30 ≈ v7 within n=30 noise. **Three mixtures → same behavior asymptote (~11-13 patches / 6-7 resolved vs gate 18)**: v6→v7 jump was real (3→10), v7→v8 flat despite 5× volume + diversity + oracle ×1. Conclusion: more SFT on this trace family will not close the gate. Remaining levers (decision): (a) **RLVR/GRPO** (Phase-E script already committed — optimize the gate metric directly with verifiable reward: non-empty patch + tests); (b) harness-level self-retry / best-of-N on empty patch (cheap, legit agentic); (c) different trace distribution (SWE-smith 5k expert Claude traces, nebius); (d) prompt strengthening in the smoke config. Semantics are healthy (25/30) and improving — the gap is purely act-vs-explore behavior. v8 mixture design: oracle **×1** (×2 drove the cp30 overfit), anchors kept, externals capped ~40%, budget 49152, format gate, then 25-30-step train w/ save-5 + mid-checkpoint gating (the proven loop).
- **(20) SWEET-SPOT FOUND: v7s2-cp15 = 22/30 (anchor+4).** Fresh same-server anchor cp20_same_day 18/30 (`runs/hard_subset_cp20_sweetspot_256k_20260713_1133`); sweep (`runs/hard_subset_v7s2cp{10,15,20}_sweetspot_256k_20260713_1303`, driven by coordinator — Codex went quiet, likely usage limit): **cp10 21, cp15 22, cp20 20** (cp30 was 19) — clean overtraining peak at ~240 rows. cp15 clears +3 (=+4) and matches s1-cp5 semantics with 3× data seen. **cp15 SWE-Lite temp-0.7 smoke launched** (`runs/smoke_v7s2cp15_t07_lite_0_30`, workers 8; NOTE: wrapper takes TEMPERATURE env var — my first launch silently ran temp 0 via ignored `-c` arg, caught by cmdline verify, killed + relaunched correctly). Beat: 10/30 patches, 7/30 resolved, 14/30 edit-reach; gate ≥18/30.

### Standing cautions for the continuing agent

- Eval harness clamps: `model_clamps.py DEFAULT_MAX_TOKENS` too low silently truncates tool calls → false "model can't act" reads. A thinking model needs generous max_tokens (32k here). Verify completion_tokens aren't piling at the ceiling before blaming the model.
- Gemma-4 thinking: `enable_thinking=true` ALONE is NOT enough and BREAKS output (degenerate `<|turn>model` loop) — the stock chat template (bundled + vLLM official, identical) never opens the `<|channel>thought` channel under thinking-on. You MUST serve with a patched template that opens `<|channel>thought\n` (see `phaseH_eval/tool_chat_template_gemma4_thinkopen.jinja`) + `--default-chat-template-kwargs '{"enable_thinking": true}'`. Then reasoning lands in `message.reasoning` and tool_calls still parse. Verify with a raw `/v1/completions` primed with `<|channel>thought\n` before blaming the model — the model reasons fine when the channel is opened.
- Never name a LoRA module the same as `--served-model-name` (invalid eval).
- `python -m pytest` from repo root, not bare `pytest` (sys.path).
- Local repo (`/Users/ironbcc/projects/llm`, branch `phaseC-verify-harness`) is source of truth for phaseD_sft scripts; remote copies are scp-synced — re-sync after any local edit.
- All v6 build intermediates + manifests live in `data/v6_build/` on host; staging scripts in `/tmp/` on host (`v6_chain.sh`, `v6_assemble.py`, `v6_budget_and_save.sh`, `v6_memgate.sh`, `verify_49k_step.py`).
- Never probe attention backends forward-only — training needs backward, where the sm_120 limits live.
- For any CPU-parallel tokenization: single tokenizer in main thread + ThreadPoolExecutor; never `multiprocessing.Pool` with per-worker loads (swap blowup incident, this session).

## 2026-07-11 Codex v6 49k execution update (live)

### Completed: OpenSWE observation-only compression

- Source: `data/v6_build/ext_openswe_merged.jsonl` (12,094 rows).
- Command pattern (run on the host with `PYTHONPATH=$PWD`):
  ```bash
  .venv-train/bin/python phaseD_sft/compact_observations.py \
    --in data/v6_build/ext_openswe_merged.jsonl \
    --out data/v6_build/ext_openswe_compact1200.jsonl \
    --manifest data/v6_build/ext_openswe_compact1200_manifest.json \
    --max-observation-chars 1200 --head-lines 16 --tail-lines 10
  .venv-train/bin/python phaseD_sft/dedup_and_trim_traces.py \
    --in data/v6_build/ext_openswe_compact1200.jsonl \
    --out data/v6_build/ext_openswe_compact1200_clean.jsonl \
    --manifest data/v6_build/ext_openswe_compact1200_dedup_manifest.json
  ```
- Safety contract: compress only `OBSERVATION:`/tool-conditioning turns; do not change assistant reasoning, code, or tool calls. The new pass reduced observation characters from 1,848,651,961 to 668,260,323 (63.85%).
- Cleanliness result: 12,094 input and output rows, 0 exact duplicates, 0 trailing messages trimmed, 0 dropped traces. The reusable cleaned JSONL is `data/v6_build/ext_openswe_compact1200_clean.jsonl`.

### Measured 49,152 eligibility and assembly decision

- One main-process Gemma tokenizer plus `ThreadPoolExecutor(16)` measured 7,855/12,094 eligible traces (**64.95%**), versus 4,392/12,094 (36%) before this compression. Audit: `/tmp/v6_openswe_compact1200_49152_audit.json`.
- Eligible language counts: Rust 1,132; C 15; Python 6,708. Distribution: p50 44,994, p90 60,697, p99 81,112, max 151,510 tokens.
- Do not use `multiprocessing.Pool` for tokenization; run `/proc/meminfo` preflight (abort under 15 GiB available) and `free -g` immediately before any threaded pass. The successful audit began with ~51 GiB available and swap use stayed flat.
- No assistant-boundary windows are needed this time: the eligible pool comfortably exceeds the selected OpenSWE cap. The 49k build must select 2,500 OpenSWE rows (C/Rust first, then Python) and 1,750 Kwai rows, preserving Kwai at 41.2% of the 4,250 external rows. Keep the 4,250-row anchor unchanged.

### Current progress and next stop gate

- Gate A (24,576, cp20 init) has now fully finalized: `Result=success`, `ExecMainStatus=0`, checkpoint present, 28m12s, `MemoryPeak=38,655,455,232` (~36.0 GiB), `MemorySwapPeak=644,567,040` (~615 MiB). GPU1 was ~2 MiB free before the current check; production ports 8000/8101/8103/8104 remained green.
- Converted the compressed JSONL to `data/v6_build/ext_openswe_compact1200_clean_hf` only to run the required 1,000-sample loss-format smoke.
- **PASSED:** `CUDA_VISIBLE_DEVICES=1 .venv-train/bin/python phaseD_sft/verify_gemma_format_loss.py --data data/v6_build/ext_openswe_compact1200_clean_hf --samples 1000`, log `/tmp/v6_openswe_compact1200_format_gate.log`. It covered 1,000 traces / 104,735 assistant messages with `failure_count: 0` and `fallback_spans: 0`; tool-call markers were balanced. The old no-progress verifier took about 1h50m on this multi-turn OpenSWE sample; the synced progress update fixes that observability gap. The 49k rebuild is now unblocked, but Gate B and training remain blocked on the rebuilt-dataset structural/format gates.
- Parallelism note: the live verifier has 53 threads but is using ~100% of a single CPU core because repeated prefix rendering is Python/Jinja-heavy. Never accelerate it with multiple processes or per-worker tokenizers. A future shared-tokenizer thread path must first demonstrate a speedup and byte/format-equivalent result on a small sample.
- Progress telemetry is now present for future v6 runs: compaction, trim/dedup, budget filtering, scanning, and format verification print completed/total, it/s, elapsed, ETA, and total estimate. The bounded launcher watchlog captures launch-to-data-ready/model-load timing, observed step intervals, final trainer runtime/per-step time, GPU1 used MiB, and cgroup peaks; the updated scripts are synced to the host.

### Evaluation and Gate B timing policy update (2026-07-11)

- The cheap hard30 gate remains first. A v6 candidate must beat a **same-day cp20 run by at least 3 semantic points** on the same serving path; do not compare to historical absolute scores.
- The former five-case SWE-Lite gate is superseded. After hard30 passes, run a **30-case SWE-Lite smoke** before any SWE-bench Verified scale-up. Required: at least **18/30 non-empty patches**, format errors below **10%**, and median first source edit before command **10**.
- When Gate B runs at 49,152, record the launcher-watchlog model-load time and measured per-step wall-clock time, in addition to peak `gpu1_used_mb`, cgroup `MemoryPeak`, and `MemorySwapPeak`. Those measured values, not the current estimated 40–70 minute bracket for five steps, determine the first full-smoke ETA.

### 49k rebuild correction: empty assistant placeholders removed (2026-07-11)

- The first post-compression 49k assembly was deliberately stopped by its structural scan: 638 selected rows contained empty assistant messages without tool calls. Those are zero-supervision loss positions, so the affected output directories were retained only as `*_rejected_empty_assistant` forensic artifacts and must not be trained or evaluated.
- Root source count was 2,954 empty assistant/no-tool placeholders across the 12,094 OpenSWE rows. The new `phaseD_sft/sanitize_agentic_empty_assistant.py` is deliberately narrow: remove just those placeholders, merge 1,754 now-adjacent `OBSERVATION:` user messages, then run normal dedup/terminal trimming (1,200 new trailing observations); it changes no non-empty assistant content or tool calls. Five evenly spaced raw-versus-sanitized traces have identical SHA-256 sequences of non-empty assistant responses/tool calls (`/tmp/v6_openswe_sanitized_spotcheck.json`).
- Corrected source: `data/v6_build/ext_openswe_compact1200_sanitized_clean.jsonl`. It retains all 12,094 rows and yields 7,801 at 49,152 tokens.
- Corrected build completed: `data/unsloth_agentic_train_swe_edit_trace_v6_budget49152` (+ `_long16`, `_long16_le24576`), manifest `data/v6_build/v6_final_49152_manifest.json`. It has 8,500 rows: anchor 4,250; OpenSWE Rust/C 1,135; OpenSWE Python 1,365; Kwai 1,750. Kwai remains 41.2% of external rows; largest accepted sequence is 49,137 tokens; only 1,524 intended duplicates remain (`unique_content_count=6,976`).
- Structural gate now **PASSED** with `failure_count: 0`; GPU1 was idle at 2 MiB, MemAvailable ~51 GiB, and production ports stayed green. The mandatory 1,000-sample final loss-format gate is now the next sole blocker before the 49,152 one-step Gate B memory measurement.

- **Final 49k loss-format gate PASSED:** `CUDA_VISIBLE_DEVICES=1 .venv-train/bin/python phaseD_sft/verify_gemma_format_loss.py --data data/unsloth_agentic_train_swe_edit_trace_v6_budget49152 --samples 1000 --progress-every 10` completed with `failure_count: 0`; log `/tmp/v6_49152_format_gate.log`. It took about 29 minutes, and the new ETA output stayed live throughout. GPU1 returned to 2 MiB, host MemAvailable was ~51 GiB, and production ports remained green.
- Gate B is now the only active blocker. Before launching it, the bounded launcher was corrected and regression-tested to remove its inherited broad `pgrep -f` stale-trainer kill path; it now controls only the explicitly named systemd unit, honoring the exact-PID/no-broad-kill machine-safety rule.

### Gate B 49,152 result: bounded-swap failure; 40,960 fallback in progress (2026-07-11)

- **49,152 FAILED the required memory gate safely.** The cp20-init one-step run became data-ready in 89 s, then was stopped during its first 16-way accumulation when system swap reached 16,866 MiB, exceeding the fixed 16,384 MiB guard. No optimizer step completed, no checkpoint was written, and consequently there is no valid per-step wall-clock metric; never substitute loader-counter timing for it.
- Measured 49k gate evidence: peak sampled GPU1 use 90,432 MiB (~88.3 GiB), `MemoryPeak=38,666,403,840` bytes (~36.0 GiB), `MemorySwapPeak=2,291,339,264` bytes (~2.13 GiB). The launcher stopped only `swe-edit-v6-memgate-b.service`; it ended failed after the stop timeout, GPU1 returned to 2 MiB, MemAvailable recovered to ~53 GiB, and production ports stayed green. Do not raise a cgroup cap; take the documented 49,152 → 40,960 fallback.
- Clean 40,960 rebuild completed from `ext_openswe_compact1200_sanitized_clean.jsonl`: `data/unsloth_agentic_train_swe_edit_trace_v6_budget40960` (+ `_long16`, `_long16_le24576`), manifest `data/v6_build/v6_final_40960_manifest.json`. It has the unchanged 4,250 anchor, 622 OpenSWE Rust/C, 1,878 OpenSWE Python, and 1,750 Kwai rows (8,500 total, Kwai = 41.2% external). Structural scan has **PASSED** with `failure_count: 0`; its required 1,000-sample final loss-format gate is the current sole blocker before the 40,960 one-step memory gate.

### User-directed memory retest policy (2026-07-11)

- User corrected the launch policy: global swap use is not a stop condition on this 255 GiB-swap host. `recover_swe_edit_v4_smoke_first.sh` now aborts host pressure only if `MemAvailable < 2,048 MiB`; the cgroup limits (36G high / 45G max / 16G swap), GPU1-only constraint, production health checks, timeout, and named-unit-only stopping remain unchanged.
- Re-test Gate B at **49,152 first**, then use the fallback ladder only for an actual cgroup/RAM/GPU failure under the revised policy. Launcher regression tests cover the new no-global-swap-abort rule.

### Gate B retest result and current next action (2026-07-11)

- **49,152 is now conclusively failed on the installed attention stack, not on host swap.** The revised-policy cp20-init one-step run selected and logged `flex_attention`, reached `train_begin`, but failed before microstep `0/16`/any checkpoint when PyTorch 2.10 dispatched the operation to `sdpa_dense → math_attention` and attempted a 128.13 GiB allocation on GPU1.
- Systemd evidence: `swe-edit-v6-memgate-b-flex.service` ran 123 seconds (17:20:52–17:22:55 UTC), `MemoryPeak=20,434,018,304`, `MemorySwapPeak=0`, `ExecMainStatus=1`; GPU1 returned to 2 MiB and all production ports remained green. The 60-second watch sample reported `model_load_seconds=64`, but this is a polling upper bound; there is no measured per-step time because no microstep completed.
- `train_rust_lora.py` and the bounded launcher now emit/parse timestamped `model_ready`, `data_ready`, gradient-microstep, optimizer-step, wall-clock, and ETA events, removing the previous opaque blue/no-output interval. The current authorized next step is the fully-gated 40,960 long16 one-step Gate B; if the same dense fallback OOMs, continue 34,816 then the already-proven 24,576 tier without raising cgroup limits.

### Fallback ladder measurements and 24k retest (2026-07-11)

- **40,960 failed** after model load/data prep of `51s/12s`, at microstep `0/16`: the dense fallback requested 91.59 GiB while 32.34 GiB was resident. Its 137-second unit had `MemoryPeak=38,655,057,920`, `MemorySwapPeak=0`; GPU1 released and production stayed green.
- **34,816 failed** after `58s/12s`, also at microstep `0/16`: the dense fallback requested 66.48 GiB while 30.48 GiB was resident. Its 144-second unit had `MemoryPeak=38,655,205,376`, `MemorySwapPeak=0`; GPU1 released and production stayed green.
- The active next gate is a fresh one-step **24,576** run on the final sanitized 49k dataset's `...budget49152_long16_le24576` shard, not an older 24k-like artifact. It must produce a checkpoint before any five-step/hard30 work; no 49k/40k/34k per-step metric exists because none completed a microstep.

### Kernel-path correction: re-run 49k compiled, do not ladder down (2026-07-11)

- The 49k/40k/34k dense OOMs are now classified as **debug-path diagnostics, not cap failures**. They were run while `UNSLOTH_COMPILE_DISABLE=1` suppressed Unsloth's regional `torch.compile`, so FlexAttention used `sdpa_dense` rather than its intended fused block-mask kernel. The temporary 24k final-clean unit was stopped before forward; it is not a result.
- User-provided direct GPU1 evidence verifies fused SDPA-Flash at exact Gemma4 GQA shape—32 query / 16 KV heads, head dim 256, 49,152 causal tokens—with a 2.3 GiB attention peak. The huge requests were dense O(n²) matrices, not a 49k VRAM capacity proof.
- Restore compilation (remove only `UNSLOTH_COMPILE_DISABLE`; retain `expandable_segments`, cgroups, GPU1-only, health checks, no-global-swap abort, and telemetry), then rerun the **49,152 cp20-init long16 one-step gate**. The first 30–60 minutes may be CPU-only Inductor compilation with GPU near idle; the watchlog now reports compiler workers/CPU plus exact phase timestamps. Stop only if compiled Flex itself fails, capture the full traceback, and do not continue the cap ladder beforehand.

## v6 49k Smoke Snapshot (2026-07-12) — Corrected

- Run: `/home/ironbcc/projects/gemma4-31B-Coder/runs/smoke_v6_49k_s1_lite_0_30_256k_20260712_025356`
- Remote smoke PID requested/expected: `1394164` (process missing at final checkpoint; run finished)
- Completion status: generation reached `30/30` submitted, 30 trajectory JSONs; scoring ended in docker container indexing failure (`No instances to run.`).
- Aggregated metrics recomputed from raw trajectories and preds:
  - Submitted: `30`
  - Total assistant action calls: `595`
  - Total format-error recovery events: `376`
  - Format-error rate (`events / action calls`): `0.6319327731092437`
  - Source-edit cases: `2/30`, no-edit cases: `28/30`
  - Median first source-edit command index (edit cases only): `8.5`
  - Source-file patch non-empty count: `0/30` (checked `model_patch` and all schema-visible fields)
  - PASs_TO_PASS / FAIL_TO_PASS evidence: not present in score output
- Per-case first source-edit indices (NA means no source-edit command found):
  - astropy__astropy-12907: NA
  - astropy__astropy-14182: NA
  - astropy__astropy-14365: NA
  - astropy__astropy-14995: NA
  - astropy__astropy-6938: NA
  - astropy__astropy-7746: NA
  - django__django-10914: NA
  - django__django-10924: NA
  - django__django-11001: 13
  - django__django-11019: NA
  - django__django-11039: NA
  - django__django-11049: NA
  - django__django-11099: 4
  - django__django-11133: NA
  - django__django-11179: NA
  - django__django-11283: NA
  - django__django-11422: NA
  - django__django-11564: NA
  - django__django-11583: NA
  - django__django-11620: NA
  - django__django-11630: NA
  - django__django-11742: NA
  - django__django-11797: NA
  - django__django-11815: NA
  - django__django-11848: NA
  - django__django-11905: NA
  - django__django-11910: NA
  - django__django-11964: NA
  - django__django-11999: NA
  - django__django-12113: NA
- Decision: FAIL (`>=18 non-empty` required, `<10%` format errors required, and edit-index/patch thresholds not met under corrected normalization).

## v6 49k SWE-Lite rerun status (2026-07-13)

- The first corrected rerun (`smoke_v6_49k_s1_rerun_lite_0_30_256k_20260712_044535`) proved the 512-token clamp had been removed (no generation sat at 512), and v2 chat-template serving restores structured tool calls plus reasoning. It did not establish the gate: it still produced no usable patches and exposed a second control-path failure.
- Rerun2 (`runs/smoke_v6_49k_s1_rerun2_lite_0_30`) was deliberately stopped after two saved trajectories when its third 8012 request generated continuously for more than seven hours. Coordinator inspection of vLLM metrics showed one request at about 45 tokens/s, zero prompt throughput, and slowly increasing KV cache: a degenerate **thinking loop** in serving, not Docker scoring, model capacity, or a 49k training failure.
- The safe corrective bound is `phaseH_eval/model_clamps.py:DEFAULT_MAX_TOKENS = 4096`, replacing the over-large 32,768 cap. This retains room for the observed p90 completion (~151 tokens) plus thought/tool output, while forcing runaway turns to return soon enough for mini-SWE recovery. The coordinator reported all 15 phaseH_eval clamp tests green before launch.
- Rerun2 was stopped by exact PIDs only (`2266329` mini-extra and `2266323` launcher). The 8012 server (`2100381`) was intentionally preserved on GPU1 with `tool_chat_template_gemma4_thinkopen_v2.jinja`, `enable_thinking=true`, and LoRA `v6_49k_s1`; GPU0 and production ports 8000/8101/8103/8104 were never touched.
- Active validation: `runs/smoke_v6_49k_s1_rerun3_lite_0_30`, launcher PID `2273814`, same 30-case `smoke_single.sh` invocation and 4,096 clamp. It began cleanly: the first three trajectories completed in roughly 15 seconds each, so the former third-request deadlock did not recur. Do not call the 30-case gate until all 30 raw trajectories, predictions, and score artifacts are present.
- Required final extraction from raw artifacts: non-empty `model_patch` count (must be >=18/30), response-level no-tool-call / format-recovery rate (must be <10%), number of edit-reaching trajectories, median first source-edit command index (must be <10), and completion-token distribution including exactly-4096 turns. If many 4096 ceilings occur, capture affected contexts by temp-0 replay before changing template, clamp, data, or training.

## RLVR controlled comparison verdict (2026-07-14)

- The matched 100-step decision-point runs are complete: Dr. GRPO
  `adapters/swe_drgrpo_v1_s2/checkpoint-100` (`/tmp/drgrpo_s2.log`) and standard GRPO
  `adapters/swe_grpo_v1_s2/checkpoint-100` (`/tmp/grpo_cmp_s2.log`). Both were served via
  new three-way merges (frozen base + v8cp30 + RL LoRA); no base model files were modified.
- Curve evidence: Dr. GRPO/GRPO respectively had mean shaped reward **0.2095/0.2050**, mean
  completion length **486.86/475.91**, and zero-reward-std steps **56/48** of 100. Dr. GRPO
  had a step-14 instability outlier (loss **11.01**, grad norm **1587**, KL **275.2**) while
  standard GRPO maxima were loss **0.002735**, grad norm **43.93**, KL **0.06838**.
- Regression floor: fresh cp20 hard30 anchor = **20/30**; Dr. GRPO = **19/30**, standard
  GRPO = **18/30**. Both pass the >= anchor-2 floor; neither is a hard30 promotion claim.
- Decisive 30-case SWE-Lite temperature-0.7 results from raw `preds.json`, trajectories, and
  final SWE-bench reports:

  | variant | non-empty patches | format errors | edit reach | median first edit | resolved |
  |---|---:|---:|---:|---:|---:|
  | Dr. GRPO | 17/30 | 2/30 (6.67%) | 23/30 | 13 | 7/17 |
  | standard GRPO | 20/30 | 3/30 (10.0%) | 25/30 | 14 | 6/20 |

- Standard GRPO clears the >=18 patch-count gate but **fails** the strict format criterion
  because 10.0% is not `<10%`; Dr. GRPO passes format but misses patch count by one. Thus
  neither variant promotes the Python lane. Artifacts: Dr
  `runs/smoke_swe_drgrpo_v1_t07_lite_0_30`; standard
  `runs/smoke_swe_grpo_v1_t07_lite_0_30`; reproducible extractor
  `phaseH_eval/summarize_smoke.py`. No further data rebuild or training is authorized without
  a user gate.

## RLVR v2 rebalanced decision dataset (2026-07-14)

- Completed CPU-only artifact: `data/rlvr_swe_decision_v2.jsonl` (1,895 rows; SHA-256
  `ab63e3f797a99d9db2564ab097f54a22587c5ecf5d57070e97e2f099b2edcaf`) with manifest
  `data/rlvr_swe_decision_v2_manifest.json` (SHA-256
  `3f6e841dc3db909ce279189fcbc51b0c5bf40c953a93948af8bfcb1c5b1a33b6`). This is data
  preparation only: no GPU, serving, training, or source-data deletion occurred.
- It preserves the v1 decision extraction: first source-edit prefix, otherwise prefix just
  before the first repeated command or fourth consecutive read. The new source-balanced,
  shortest-prefix selection keeps smoke contexts in full and caps `swe-smith=700`,
  `swe_train_oracle_edit_trace=256` (all valid rows; fewer than its 300 cap),
  `open_swe_traces_qwen35=300`, and `kwai_klear_miniswe=500`; `coder_repair_synthetic` is
  deliberately excluded. Smoke contributes 139 rows over five completed evaluation runs.
- Independent JSON/schema/content-hash verification found 1,895 valid rows and 1,895 unique
  rendered-message hashes. Budgeting used one main-process Gemma tokenizer shared by
  `ThreadPoolExecutor(8)` and accepted only rendered prefixes <=8,192 tokens (p50 2,001;
  max 5,331), remaining compatible with the GRPO loader's `--max-prompt-tokens 2560` clamp.
  Builder support is in `phaseE_rl/build_swe_decision_dataset.py` via explicit source caps;
  its targeted local unit suite is 7/7 green.

## Rust raw-base fixed-subset baseline (2026-07-14)

- The Rust lane uses the fixed image-backed denominator
  `data/mswe_rust_prs_imagebacked.jsonl`: **75** of the original 239 Rust instances whose
  `mswebench/<org>_m_<repo>:pr-<number>` image is present locally. The manifest records the
  composition: clap 35, fd 14, ripgrep 13, bytes 5, bat 3, serde 2, rayon 2, nushell 1.
  This fixed list—not the image-incomplete 239-row source—is the required comparison set for
  future Rust adapters.
- Coordinator smoke-5 first validated the raw NVFP4 serving path on 8012 with the v2
  thinkopen template. The production baseline used the committed
  `phaseA_scaffold/rust_baseline_driver.py` at `5507f8b`: raw model
  `gemma4-rust-baseline`, temperature 0.7, sequential Docker rollouts, no LoRA or training.
  Critical execution contract: Docker commands use **`bash -c`**, not `bash -lc`; the latter
  resets PATH in these images and produces false `cargo: command not found` failures.
- **Full baseline complete:** `runs/rust_baseline_rawbase_full75/results.jsonl` has 75 unique
  instances, **61/75 non-empty patches (81.33%)**, **11/75 resolved (14.67%)**, and **zero
  driver/runtime errors**. Mean/median instance wall time were 449.8s/375.9s (9.37 aggregate
  sequential hours). Per-repo resolves: clap 3/35, fd 4/14, bytes 3/5, rayon 1/2; bat,
  ripgrep, serde, and nushell had zero resolves.
- Resolution here is the self-contained image's canonical `/home/test-run.sh` after the
  model edit (its test patch is freshly applied, so test tampering self-defeats). The official
  `/home/ironbcc/multi-swe-bench` evaluator remains available for an optional later
  id-level report cross-score; it was not substituted or silently claimed as this baseline's
  scorer. GPU0/prod ports stayed green, Docker loopback stayed at 94GB free, and the raw-base
  8012 service remained healthy after completion.

## (29) User decisions locked + GRPO promotion executed (2026-07-14)

1. **PROMOTED**: `adapters/swe_grpo_v1_s2/checkpoint-100` is the current Python policy
   (20/30 patches vs SFT 11, edit-reach 25 vs 14, hard30 floor pass; known caveats:
   resolve 6/30 unchanged, format 10.0% at boundary, patch precision 30%).
   v8cp30 remains immutable as rollback. `merged/swe_drgrpo_v1` was deleted (58 GB reclaimed;
   Dr. GRPO adapter checkpoint retained); `merged/swe_grpo_v1` is the serving artifact.
2. **Verified-reward RL round: GO on RLVR-v2.** Phase 2 replaces parse-only 1.0 with a Docker
   apply/F2P-test tier, initialized from promoted GRPO checkpoint-100.
3. **C++ image builds: active CPU/disk lane** for `data/mswe_cpp_prs/` (257 instances).
   Docker loopback must remain >=40 GiB, families build serially, dangling layers are pruned
   only between families, and any floor breach stops and reports.
4. **Rust adapter v2:** queued after this Python verified-reward round; fixed baseline remains
   11/75 resolved.

## RLVR verified-reward phase-2 preflight (2026-07-14)

- **User policy:** standard GRPO `adapters/swe_grpo_v1_s2/checkpoint-100` is the current Python
  policy; its retained serving artifact is `merged/swe_grpo_v1`. This explicit promotion
  supersedes the prior strict-format-boundary hold. `merged/swe_drgrpo_v1` was deleted after
  comparison (58 GB reclaimed); adapter checkpoints remain for controlled evidence.
- Fixture sidecar: `data/rlvr_swe_decision_v2_fixtures.jsonl` and
  `data/rlvr_swe_decision_v2_imagecov.json` exact-join **1,330/1,895 (70.2%)** v2 decision
  prefixes to image and F2P metadata: SWE-smith 691/700, Kwai 500/500, and smoke 139/139
  (all smoke images local). OpenSWE 300, oracle 256, and nine SWE-smith rows are explicitly
  `unverified=true`; they can retain the phase-1 parse/edit tier through 0.6 but must never be
  awarded execution/F2P rewards. Normalizing all oracle trace IDs to owner/repo/number joined
  0/256 against SWE-smith, SWE-Lite, and full SWE-bench, so no suffix-only mapping was invented.
- `phaseE_rl/reconstruct_swe_decision_state.py` provides a bounded Docker state cache keyed by
  prompt hash. It replays only strict read-only prefix commands under `bash -c` in a
  network-isolated container, commits labelled `rlvr-state:*` images, removes containers by
  exact id, and LRU-evicts only its own images while enforcing the 40-GiB loopback floor. The
  local dry run (`runs/rlvr_state_cache/dry10.jsonl`) created 10/10 states; a follow-up
  (`runs/rlvr_state_cache/hit1.jsonl`) confirmed a cache hit, with Docker still 92 GiB free.
  This validates cache mechanics only—the coordinator-owned completion apply/F2P verifier must
  still demonstrate observed 0.8/1.0 tiers on ten prompts before a verified-reward training run.

## Submission-protocol recovery: cp300/cp100 clean rerun (2026-07-15)

- Investigation of the apparent cp300 patch loss found **8/30** trajectories with real tracked
  source hunks mid-trajectory (`git diff` 1.3–3.8 KB) that were discarded by the submission
  protocol, not by the model. The stock mini-SWE Docker environment accepts a submission only
  when the marker is the first output line and return code is zero; it also accepted an empty
  printed patch. Context use was only 4–24k of 49,152 and no response approached the 4,096
  clamp, so neither explains the loss.
- The synchronized fix is `phaseH_eval/docker_selfretry.py`
  (`DockerSelfRetryEnv`) and `phaseH_eval/swebench_edit_first_selfretry.yaml`: reject empty
  submissions up to two times with a `git add -A`/cached-diff recovery observation, and nudge
  once when the marker appears after the first output line. Targeted tests passed **6/6**.
- Consequently the historical cp100-versus-cp300 smoke comparison is confounded. After the
  in-flight Rust fixed-75 evaluation completes, let the coordinator restore merged cp300
  (`/tmp/restore_cp300_v2.log`) and run both cp300 and retained `merged/swe_grpo_v1`/cp100
  through the identical 30-case self-retry harness before any promotion decision. Do not delete
  either merged model while this clean comparison is pending.

## Rust v2p SFT verdict: agentic regression (2026-07-15)

- Training data `data/rust_sft_v2p_5k` passed its 1,000-sample Gemma format gate with zero
  failures. A fresh frozen-base r32/a32 LoRA completed a one-step 49,152-context sm120 gate
  (checkpoint present; 16 microsteps, 15.77 s) and then a 25-step bounded run at 49,152;
  `adapters/rust_sft_v2p_5k_s1/checkpoint-25` is retained. The three-prompt format smoke
  passed thinking **3/3**, Rust function **3/3**, and container compilation **3/3**.
- The agentic evaluation was stopped early by user decision after a deterministic, stratified
  matched **15-instance** slice. `runs/rust_v2p_strat25/verdict_summary.json` is the canonical
  comparison: v2p **0/15 resolved, 4/15 non-empty patches, 6 errors**, versus raw base
  **2/15 resolved, 14/15 non-empty patches, 0 errors** on exactly the same IDs. Measured
  total and mean wall time were **4.30x** raw base (30,851.3 s vs 7,176.7 s); this artifact
  value supersedes the initial rough 8x estimate.
- Several errors were exact context-cap failures after the function-level policy continued
  deliberating until prompt + fixed 4,096 output budget exceeded the 131,072-token raw-base
  serving limit. Keep the driver unmodified for this parity comparison: raw base completed
  the same configuration without these failures, making the behavior part of the candidate
  result rather than a retroactive harness fix.
- **Decision:** Rust v2p is a regression for agentic use. Likely retraining levers are
  reasoning-length-stratified Rust data, explicit agentic-trace mixing, and a separately
  evaluated serving-side reasoning budget; do not scale this adapter. Artifacts retained:
  `runs/rust_v2p_strat25/{results.jsonl,rawbase_same15.jsonl,verdict_summary.json}` and
  `data/mswe_rust_prs_strat25_manifest.json` (seed 20260715, early-stop provenance).

## RLVR v3 edit-adjacent decision prompts (2026-07-15)

- `data/rlvr_swe_decision_v3.jsonl` contains **431** deduplicated decision prompts. Each ends
  immediately before the assistant command one command before the trajectory's first source
  edit; it deliberately does not reuse round-1 stall/repeat prefixes. Rendered prompts are
  bounded at 3,072 Gemma tokens (min/p50/max: 86/1,653/3,072).
- Selection remained evidence-limited rather than padded: **345/431 (80.0%)** rows are exact
  fixture-verifiable via the v2 sidecar, while Oracle+OpenSWE are **86/431 (20.0%)**—the hard
  diversity ceiling. Source groups: SWE-Smith 289, Kwai 43, smoke 13, oracle 64, OpenSWE 22.
  The requested 768 target is recorded as a 337-row honest shortfall in
  `data/rlvr_swe_decision_v3_manifest.json`; no quality or leakage rule was relaxed.
- The fixture sidecar is `data/rlvr_swe_decision_v3_fixtures.jsonl`; 119 selected fixtures have
  bases already local and are eligible for prompt-hash state reconstruction only. The cache
  extension uses `reconstruct_swe_decision_state.py --local-only --floor-gib 40
  --max-cache-gib 4000`; it must never pull an image or prune non-`rlvr.state` layers.

## Self-retry promotion control + vGRPO round 2 (2026-07-15)

- The corrected, identical 30-case self-retry harness changes the historical resolution
  interpretation: the protocol fix lifted both retained policies by roughly 3–4 resolves, so
  the former 6–7/30 resolution plateau was substantially a **submission-protocol artifact**,
  not clean evidence that resolve quality had stalled. Treat all pre-selfretry cp100/cp300
  resolve comparisons as confounded.
- Clean artifacts: `runs/smoke_swe_vgrpo_v1_cp300_selfretry_t07_lite_0_30/summary.json` is
  cp300 = **18 patches, 9 resolved, 23 edit reach, median first edit 14, 10.0% format,
  50.0% precision**. `runs/smoke_swe_grpo_v1_cp100_selfretry_t07_lite_0_30/summary.json` is
  cp100 = **21 patches, 10 resolved, 27 edit reach, median 17, 6.67% format, 47.6% precision**.
  cp100 remains the policy: it wins patches, resolves, edit reach, and the strict format gate;
  cp300 only leads on first-edit timing and precision.
- v3 local-only cache completion: 86 state records for the new prompt hashes (**75 created,
  11 existing cache hits**); 33/119 local-image candidates were replay-unsafe and deliberately
  skipped. The round-2 trainer's 3,072 token load filter leaves 424 prompts, 338
  fixture-eligible; only the 86 state-backed rows can earn Docker execution tiers while cache
  misses remain parse-capped. No images were pulled and Docker held 53 GiB free.
- Authorized round 2 is `adapters/swe_vgrpo_v2`: frozen policy initialization ordered as
  `swe_edit_v8_49k_s1/checkpoint-30` then `swe_grpo_v1_s2/checkpoint-100`, fresh r32 LoRA,
  standard GRPO, verified reward, `num_gen=8`, `gen_temperature=1.0`, 200 steps, and the
  synced trainer's group-preserving gradient accumulation (`num_gen // 2`). Log:
  `/tmp/vgrpo_v2.log`. GPU1 only; check checkpoints every 25 steps before any evaluation.

## Raw-base length-recovery gate (2026-07-16)

- The 131,072-context raw NVFP4 control was behaviorally strong but initially missed the
  strict format gate: **26/30 patches, 16/30 resolved, 27/30 edit reach, 4/30 format errors**.
  The four errors were `django-11019`, `django-11283`, `django-11630`, and `django-12113`:
  ordinary responses hit the intentional 4,096 completion clamp with `finish_reason=length`
  and no tool call after spending the budget in thought. Prompt use was only 1.3k--10.5k,
  so increasing the 131k context window could not address it; globally raising the clamp to
  32k had previously caused runaway thought loops.
- The reviewed/synchronized fix is the one-shot client recovery in
  `phaseH_eval/vllm_direct_model.py`. Only after a normal, unsalvageable
  `length + no-tool-call` response, the next harness retry uses `max_tokens=1024`, forced
  `bash` tool choice, `enable_thinking=false`, and `Emit exactly one bash tool call now. No
  explanation.` It is consumed once and cannot re-arm, leaving normal trajectories unchanged.
  Focused coverage is `phaseH_eval/tests/test_vllm_recovery.py`; host tests passed **23/23**
  including existing clamp tests. A live 8012 request with those exact fields returned a
  structured `bash` tool call in 13 completion tokens.
- Targeted replay of the four original Lite positions submitted all **4/4**; one exercised the
  exact length-to-recovery transition and produced the valid next tool call. The full fixed
  self-retry confirmation is
  `runs/smoke_gemma4_rawbase_recovery_selfretry_t07_lite_0_30`: **27/30 non-empty patches,
  15/30 resolved, 29/30 edit reach, median first edit 13, 0/30 format errors**. Its scored
  report is `openai__gemma4-rawbase.smoke_gemma4-rawbase.json`. This clears the <10% format
  gate and establishes raw base as the current behavior/reference control; GPU1 was released
  to 2 MiB and production ports remained green after evaluation.
