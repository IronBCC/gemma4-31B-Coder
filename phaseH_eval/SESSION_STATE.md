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

- Codex is the main execution driver/co-owner. Drive Python improvements primarily, verify claims
  from artifacts, and incorporate execution-verified successful trajectories into the next native
  Gemma training/reward corpus. After the Python behavior/promotion gate, proceed to Rust, then C++.
- User restated this ownership on 2026-07-17: Codex drives the experiment loop end to end and
  carries learned trajectories through conversion, loss/format gates, training smokes, and
  behavior evaluation; reporting or waiting for an external coordinator is not the terminal state.
- Make Gemma31B the best solo Coding model for SWE-bench while keeping the base frozen and avoiding
  regressions. SWE-Lite is the promotion gate; hard30 is a regression floor.

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

## Expert Iteration 1 corpus: raw-base + recovery (2026-07-16)

- A decontaminated, local-image-only SWE-Smith pool was sampled from raw NVFP4 base through the
  fixed self-retry/recovery agent path at 131,072 context: 150 Python instances across 14
  non-evaluation repositories, four temperature-1.0 rollouts each. The pool has zero overlap
  with the 30-case Lite IDs and zero repo overlap (Lite uses astropy/django); manifest:
  `data/expert_iter1_pool_manifest.json`.
- Sampling completed **600/600** without errors: **371 F2P-resolved (61.8%)**, 532 non-empty
  patches. Only resolved trajectories entered harvest. Eight had no source-edit command and
  were excluded by the structural edit-first gate; 363 traces remained. A v7-style command-10
  cutoff would retain only 1/371 successful trajectories, so this corpus records its deliberate
  source-edit-required/no-position-cutoff policy rather than silently applying that external-mixture
  heuristic.
- Final training dataset is `data/expert_iter1_sft` (363 rows; Arrow SHA256
  `10cbd69a5dba5a61ce7f8adf20ff1a13a674c2cb5e7456ca82a1ed2ae8bca39f`) with audit manifest
  `data/expert_iter1_sft_manifest.json`. It has zero content duplicates, observation-only
  compaction to 2,400 characters (42.64% character reduction), zero budget rejections at 49,152
  tokens (maximum 29,557), and a full 363-sample Gemma format/loss gate with **failure_count=0**.
  Stop here: no Expert Iteration adapter training is authorized until its strict evaluation plan
  is separately approved.

## Expert Iteration 1 edit-first compression (2026-07-16)

- The mechanically clean 363-row Expert Iteration corpus mirrored raw-base's late-edit behavior
  (median first source edit command 29; only about 3% edited by command 10). It is retained at
  `data/expert_iter1_sft` but is **not** the corpus to train first: this is the same late-edit
  shape that made earlier adapters subtractive.
- `phaseD_sft/compress_editfirst.py` rebuilds successful traces around the first source edit:
  it keeps the system/PR turn, up to the last three pre-edit read/observation pairs that name the
  edited path (basename accepted when a directory listing is the evidence), then preserves the
  edit-through-submit suffix—including assistant reasoning—verbatim. It rejects a trace if no
  file-grounding observation can be retained; focused tests cover command-index reduction,
  suffix preservation, edited-file evidence, balanced selected pairs, and compound
  `sed ...; git diff > patch.txt` target parsing.
- New CPU-only dataset: `data/expert_iter1_sft_editfirst`, **317** rows, Arrow SHA256
  `1a2be5bd07c8631e788a95c2e9540b9963d165932bbdf390cdbef494d57e8b29`. First-edit histogram is
  command **1:3, 2:87, 3:87, 4:140** (median **3**). The full chain passed: observations compacted
  to 2,400 chars, zero content duplicates, zero 49,152-token rejections (maximum 16,273), and
  a 317-sample Gemma format/loss gate with **failure_count=0**. Manifest:
  `data/expert_iter1_sft_editfirst_manifest.json`; no training is authorized yet.

## Expert Iteration 1 SFT attribution control (2026-07-17)

- The approved fresh-r32 Expert Iteration adapter trained for three epochs on the 317-row
  edit-first corpus; retained checkpoints are `adapters/expert_iter1_s1/checkpoint-{20,40,60}`.
  All three were merged in bf16 for evaluation and served with the v2 think-open template,
  131,072 context, fp8 KV cache, the fixed self-retry harness, recovery client, and temperature
  0.7 on the same Lite 0:30 slice. GPU1 was released to 2 MiB and production ports remained
  green after the final control.
- Checkpoint results were: **cp20** hard30 semantic 17/30, Lite **28 patches / 17 resolved /
  28 edit-reach / 0% format**; **cp40** 18/30, 27 / 15 / 29 / 0%; **cp60** 16/30, 25 / 12 / 30 /
  0%. Thus cp20 is the adapter-set peak and additional epochs regress resolution. Hard30 uses the
  regression floor; cp60 is exactly the historical 18/30 anchor minus the allowed two points.
  Artifacts are `runs/hard30_expert_iter1_cp{20,40,60}.jsonl` and
  `runs/smoke_expert_iter1_cp{20,40,60}_t07_lite_0_30/`.
- The original raw control was NVFP4 and resolved 15/30, so cp20's apparent +2 required a
  same-precision attribution control. The raw **bf16** base, with no adapter and otherwise the
  identical 131k self-retry/recovery evaluation, scored **27 patches / 17 resolved / 29
  edit-reach / 0% format** (`runs/smoke_gemma4_rawbase_bf16_selfretry_t07_lite_0_30/`, official
  report `openai__gemma4_rawbase_bf16.smoke_gemma4_rawbase_bf16.json`). Cp20 therefore ties raw
  bf16 on the decisive resolve metric (and only adds one patch while losing one edit-reaching
  trajectory); it does **not** prove an adapter improvement. Keep raw-base+recovery as the policy
  and bank Expert Iteration 1 as a clean neutral result rather than promote the adapter.

### Failure-mode audit

- The bf16 raw control makes the remaining limitation explicit: 17 resolved, **10 submitted
  non-empty but unresolved** patches, and only 3 empty submissions. The self-retry/recovery path
  reduced format errors to zero, so it is no longer defensible to attribute the 10 losses to tool
  syntax or the old first-line marker bug. All ten contain real source diffs except
  `django-12113`, which incorrectly edits tests; they are plausible but F2P-incorrect patches.
- Unresolved submitted trajectories are not primarily an edit-decisiveness problem: their median
  first edit is command **7.5** versus **9.5** for resolved traces, but their command median is
  **33.5** versus **24**. The next lever must therefore supply outcome credit/search over the
  correctness of candidate patches (or targeted verification feedback), rather than further
  generic CoT/agent-trace imitation. The Expert corpus was self-distillation from raw-base
  successes (371 F2P successes, 317 compressed training rows); it can preserve workflow but adds
  little novel patch-selection information on held-out issues.

## Python verified-teacher system gate preflight (2026-07-17)

- Before any further Python SFT/RL corpus is collected, the experiment must establish an
  execution-verified advantage for a **teacher system** over the raw bf16 Gemma recovery system.
  This is deliberately not called a pure model comparison: Codex executes shell commands through
  its own CLI loop, while Gemma uses native mini-SWE bash tool calls. Both sides must nevertheless
  use the same fixed task IDs, local Docker image, and in-container F2P scorer; their differing
  agent contracts are recorded in the paired manifest.
- The collector-only smoke `runs/python_teacher_gap_v1_smoke3/` passed mechanically: **2/3
  F2P-resolved, 3/3 non-empty patches, 0 errors**, one F2P invocation each, with Docker at
  **53 GiB free**. It is not evidence of a teacher gap and does not authorize corpus construction.
- Reproducible selector `phaseD_sft/select_python_teacher_probe.py` now builds difficulty-stratified
  screens from the four historical raw rollouts per instance. It emitted
  `data/python_teacher_gap_probe20.jsonl` (**20 rows**, strata `0/1/2/3/4 solved = 3/1/4/7/5`,
  historical raw p@1 **14/20**, p@4 **17/20**) and the disjoint
  `data/python_teacher_gap_probe60.jsonl` (**60 rows**, strata `9/4/11/21/15`, historical raw
  p@1 **38/60**, p@4 **51/60**). The sets have **zero ID overlap**, both cover 14 already
  decontaminated non-evaluation repositories, and manifests record all IDs/quotas/seed `20260717`.
  Historical rollout rates are selection strata only; they are not the missing current bf16 control.
- Required order: run raw bf16 + recovery on probe20 first; then Codex on the exact same 20;
  advance only to the pre-registered, disjoint 60 if the 20 is at least raw +3; collect at least
  500 verified positives across 50 repos only if the 60 is at least raw +3. Conversion is
  chosen-only SFT rendered in native mini-SWE/Gemma format—never Codex-shaped trajectories or
  implicit DPO pairs. Promotion later requires raw +4 on Lite-30 or raw +3 on a held-out Lite-60,
  not a +1 fluctuation. Codex reasoning fields are not a usable CoT source, so no CoT-transfer
  claim is permitted.
- The paired 20-case screen has passed: raw bf16 + native recovery recorded **13/20 F2P-resolved,
  18/20 non-empty, 0 errors** at `runs/python_teacher_gap_v1_raw20/results.jsonl`; Codex
  `gpt-5.6-terra` recorded **18/20 F2P-resolved, 20/20 non-empty, 0 errors/credit-wall** at
  `runs/python_teacher_gap_v1_teacher20/results.jsonl`. The ID sets are exactly equal and the
  teacher-system delta is **+5** (six teacher-only F2P successes). GPU1 was returned to 2 MiB and
  production ports were green after each run. This authorizes only the disjoint 60-case
  confirmation; it does not authorize collection or a training run.

## Python verified-teacher confirmation and corpus build (2026-07-17)

- The pre-registered, disjoint paired-60 confirmation passed decisively. Raw bf16 + native
  recovery recorded **35/60 F2P-resolved, 52/60 non-empty patches, 0 errors**; Codex
  `gpt-5.6-terra` recorded **52/60 F2P-resolved, 60/60 non-empty patches, 0 errors/credit hits**.
  Exact contingency is both 34, teacher-only 18, raw-only 1, neither 7: teacher-system delta
  **+17**, one-sided exact McNemar `p=3.814697265625e-05` (two-sided
  `7.62939453125e-05`). Artifact: `runs/python_teacher_gap_v1_comparison60.json`. This remains
  a teacher-*system* versus raw-recovery-*system* result because the agent contracts differ.
- The available local/decontaminated collection universe is 150 unique SWE-Smith tasks across
  14 repositories, not the earlier hypothetical 500/50 target. The 80 paired runs already
  supply 70 verified positives. `data/python_teacher_collection_v1_remaining70.jsonl` covers
  the other 70 exactly once; its resumable Codex collection is active at
  `runs/python_teacher_collection_v1_remaining70/`. After this first pass, only unresolved
  tasks are retry-eligible, up to four total attempts. `phaseD_sft/build_teacher_retry_pool.py`
  makes that selection auditable and never repeats a resolved task. The hard corpus gate remains
  at least 120 unique verified tasks, target 130; do not pad with duplicate successful attempts.
- `phaseD_sft/build_verified_teacher_sft.py` converts verified Codex streams into chosen-only,
  native mini-SWE patch-decision rows. It discards all Codex `agent_message` prose, unwraps only
  container-scoped `/testbed` commands, keeps up to three read/observation pairs that actually
  name the edited source file, and ends supervision at the first matching source edit. Empty
  synthetic issue text is replaced only with its F2P test names; gold patches are never rendered
  into the prompt. Non-replayable outer-host heredocs and Git-ref patch copying remain rejected.
- Real 80-run conversion smoke: of 70 F2P positives, **67 kept**, with the three exclusions above
  (one Git-ref copy, two outer-host heredocs); zero content duplicates and zero 49,152-token
  rejects. Prefix tokens min/median/max = **724/1,424/8,454**. The full 67-row Gemma loss-format
  gate passed with **failure_count=0**, 190/190 assistant spans and tool-call markers balanced,
  and zero fallback spans. Artifacts: `/tmp/python_verified_teacher_sft_smoke80_manifest.json`,
  `/tmp/python_verified_teacher_sft_smoke80_rejected.jsonl`, and
  `/tmp/python_verified_teacher_sft_smoke80_format.log`. Focused converter/retry/collector tests
  passed **30/30** on the training host. This validates the conversion path but does not authorize
  training until the unique-task corpus gate is met and the final dataset passes the same gates.
- Patch-decision loss is now explicit rather than nominal: each retained read action carries
  `loss=false`, while exactly the final verified source-edit assistant turn carries `loss=true`.
  `train_rust_lora.py` honors this flag with backward-compatible default-on behavior for all
  older datasets, and `verify_gemma_format_loss.py` audits the same selected spans. The rebuilt
  67-row smoke masks **123** read assistant turns and supervises exactly **67** edit turns;
  `failure_count=0`, zero fallback spans, supervised-token min/p50/p90/max = **43/81/175/748**.
  This prevents the chosen-only corpus from silently becoming another read-trajectory imitation.

## Python verified-teacher final corpus and training gate (2026-07-17)

- The collection completed with 124 unique F2P-verified tasks across 13 repositories. The final
  native Gemma dataset is `data/python_verified_teacher_sft_v1` with JSONL mirror
  `data/python_verified_teacher_sft_v1.jsonl`, manifest
  `data/python_verified_teacher_sft_v1_manifest.json`, and rejected-record sidecar
  `data/python_verified_teacher_sft_v1_rejected.jsonl`. It contains 124 content-unique rows,
  zero budget rejects, and prefix-token min/median/max **724/1,505/8,454**. Nineteen empty
  synthetic problem statements use only F2P test names as the prompt fallback; no gold patch or
  hidden `test_patch` enters the prompt or target.
- The complete 124-row loss-format gate passed with `failure_count=0`: **223** read/tool assistant
  turns are masked and exactly **124** verified source-edit turns are supervised, with zero
  fallback spans. Supervised-token min/p50/p90/max is **40/84/194/748**; all **347** tool-call
  markers are balanced. Arrow SHA256 is
  `a28c205ba9f149a29fc9336807aaf23fba963bb7db4623ce66c973e0eccc81a8`.
- The mandatory one-step fresh-r32 LoRA smoke at context 12,288 passed: checkpoint-1 is
  979,558,760 bytes, loss **0.34496**, grad norm **0.54445**, optimizer-step wall **31.73s**,
  cgroup MemoryPeak **1,198,120,960 bytes**, MemorySwapPeak **0**, and GPU1 returned to 2 MiB.
  Production ports remained HTTP 200. This authorizes the preregistered one-epoch run only; it
  is not behavior evidence.
- The one-epoch run is `python-verified-teacher-v1-s1-20260717.service`, output
  `adapters/python_verified_teacher_v1_s1`, log `/tmp/python_verified_teacher_v1_s1.log`, with
  fresh r32/a32 LoRA, frozen 4-bit-loaded base, LR 2e-5, batch 1, accumulation 8, warmup 2,
  context 12,288, and checkpoints every four optimizer steps. Promotion still requires behavior:
  hard30 is a regression floor and fixed self-retry Lite-30 must beat the same-precision raw
  bf16+recovery control of 17/30 by the preregistered margin.

## Python verified-teacher SFT behavior verdict (2026-07-18)

- The one-epoch fresh-r32 run completed all 16 optimizer steps in **283.8s** with checkpoints
  4/8/12/16 and final train loss **0.3233**. Selective loss remained enabled for 124/124 rows.
  Gradients were finite but had isolated pre-clip spikes at steps 2 and 6; this made the exposure
  sweep mandatory rather than treating the final checkpoint as automatically best. The focused
  teacher/converter/selective-loss test suite passes **49/49** in `.venv-eval`.
- Exact bf16, 131,072-context, FP8-KV, think-open-v2, temperature-0.7, fixed-self-retry results:

  | checkpoint | hard30 semantic | non-empty | resolved | edit-reach | format |
  |---|---:|---:|---:|---:|---:|
  | raw bf16 control | n/a | 27/30 | 17/30 | 29/30 | 0% |
  | teacher-SFT cp4 | 18/30 | 26/30 | 15/30 | 29/30 | 0% |
  | teacher-SFT cp8 | 17/30 | 27/30 | **19/30** | 30/30 | 0% |
  | teacher-SFT cp16 | 19/30 | 28/30 | 16/30 | 29/30 | 0% |

- Cp8 is the sweet spot and paired-dominates the raw run on this slice: **17 both resolved,
  0 raw-only, 2 cp8-only, 11 neither**. The two gained cases are `astropy__astropy-14365` and
  `django__django-11422`. This is promising evidence that the learned trajectories transferred,
  but +2/30 remains below the preregistered +4 promotion margin and is not statistically strong
  enough to replace the raw policy. Cp4 and cp16 are negative exposure controls; more of the same
  one-epoch data is not justified by the curve.
- Raw bf16+recovery therefore remains the current policy while cp8 is retained as the candidate.
  The next confirmatory step must be frozen before running either model. A disk-safe option is the
  35 SWE-bench Lite instances already backed by local Docker images, excluding the fixed 30 and
  every teacher-corpus repository; require cp8 to beat raw by at least **4 resolved** on that new
  set before promotion. Do not weaken the threshold or pull new images below the 40-GiB Docker
  floor after seeing results.

## Python teacher-SFT independent confirmation (2026-07-18)

- The confirmation set was frozen before either run at
  `data/python_verified_teacher_holdout35.jsonl`, manifest
  `data/python_verified_teacher_holdout35_manifest.json`, SHA256
  `03a61f86d1a3ff03eb8243ae0278b6d74b8e1d2b30a579e0a8126704e3d0040e`.
  It contains all **35** additional Lite instances already backed by local Docker images:
  django 20, sympy 11, matplotlib 3, flask 1. It has zero fixed-30 ID overlap, zero
  teacher-corpus repository overlap, and requires no image pulls; Docker remained at 53 GiB
  free. `smoke_single.sh` now forwards an optional exact `FILTER` regex, with a focused test,
  so both policies ran precisely the manifest IDs through the unchanged official Lite scorer.
- Confirmatory results at the same bf16/131k/FP8-KV/think-open-v2/self-retry/temp-0.7 contract:

  | policy | non-empty | resolved | edit-reach | format |
  |---|---:|---:|---:|---:|
  | raw bf16 control | 30/35 | **17/35** | 33/35 | 0% |
  | teacher-SFT cp8 | 30/35 | **16/35** | 34/35 | 0% |

- Cp8 failed the frozen `raw +4` confirmation bar and is not promoted. Heldout pairing is
  14 both resolved, 3 raw-only, 2 cp8-only, 16 neither. Across fixed30 + heldout35, raw resolves
  **34/65** and cp8 resolves **35/65**: 31 both, 3 raw-only, 4 cp8-only, 27 neither. The net +1
  is neutral and confirms that the fixed-30 +2 was not a robust gain. Canonical comparison:
  `runs/python_verified_teacher_v1_comparison65.json`.
- **Verdict:** chosen-only SFT on 124 execution-verified teacher edit turns transferred action
  shape but not enough patch-correctness to improve the policy. Do not run a second epoch or
  train more on the same targets. Raw bf16+recovery remains the Python policy. The next primary
  Python lever must use the learned successes *against matched failures*—audit existing raw k=4
  trajectories for same-task, non-empty incorrect edits and build common-prompt preference or
  patch-revision pairs only if the data volume/contract gate passes.

## Python teacher-label credit audit and final-patch correction (2026-07-18)

- The matched-failure inventory is useful but too small for the preregistered preference lane:
  the 124 clean teacher tasks contain 90 raw failed/submitted/F2P-tested negatives across 56
  tasks, but strict same-file, prefix-grounded pairing yields only 44 tasks / 61 pairs at a cap
  of two. This is far below the 500-pair DPO gate, so no DPO training is authorized from this
  bank. Five tasks remain strong teacher-only examples where all four raw samples failed with
  tested edits.
- More importantly, isolated replay disproved the assumption that a resolved final trajectory
  makes its *first* edit a correct SFT target. `phaseD_sft/audit_teacher_edit_credit.py` audited
  every eligible supervised command against the exact local fixture and complete F2P list up to
  the verifier's honest 20-test limit. Of 82 eligible v1 rows, only **58 tier-1.0** commands
  passed F2P, **7 tier-0.8** changed source but failed F2P, and **17 tier-0.6** failed to produce
  an executable source diff. Forty-two rows were excluded because their F2P list exceeded 20;
  the older verifier silently caps larger lists and therefore cannot certify them fully.
- Fifteen of the 17 tier-0.6 labels begin with the teacher-only `apply_patch` helper, which is not
  installed in the mini-SWE deployment container. The other failures are invalid/no-op edits;
  most tier-0.8 rows are plausible partial `sed` edits that do not solve the task in isolation.
  Thus the mechanically perfect Gemma format gate validated syntax, not label execution or
  causal correctness. This explains why better final teacher trajectories transferred action
  shape without improving held-out resolution.
- Corrective builder `phaseD_sft/build_verified_teacher_finalpatch_sft.py` leaves the v1 corpus
  untouched and emits one portable supervised `git apply -` heredoc containing the teacher's
  own final submitted source patch. It retains only successful pre-edit reads, requires an exact
  observation-grounding path for every edited file, preserves that evidence through compaction,
  rejects unsafe patch shapes, caps F2P at 20, hashes the command for stale-ledger protection,
  and can finalize only independently replayed tier-1.0 rows. Focused final-patch/legacy/audit
  tests pass **43/43**.
- The strict pre-replay candidate bank is
  `data/python_verified_teacher_finalpatch_candidates_v2_grounded`: **80** rows across 13 repos,
  zero budget rejects, token min/median/max **788/1,478/8,130**. Drops are 43 F2P-over-cap,
  44 unresolved attempts, four missing exact path grounding, one unparseable command, and one
  unrecognized original edit. Its isolated all-row replay passed **80/80 at tier 1.0**, with zero
  lower-tier rows or audit drops. Credit ledger:
  `data/python_verified_teacher_finalpatch_credit_v2.jsonl` SHA256
  `921a1957d7fe7ee8b5ac5741c9c7666e4ee71217c139ea547b401f5d12c3c39f`.
- Final tier-1-only corpus: `data/python_verified_teacher_finalpatch_sft_v2` plus JSONL/manifest.
  Exact command hashes, credit IDs, full patch paths, and retained grounding were independently
  rechecked for all **80/80** rows. The Gemma loss/format gate passed all 80 with
  `failure_count=0`: 116 read turns masked, exactly 80 final-patch turns supervised, zero fallback
  spans, and supervised-token min/p50/p90/max **151/262/496/1,523**. JSONL SHA256 is
  `f740d7366a6976c7380e753af7a6e99dcbea4fb7e88a0b06d653ba8027513802`; Arrow SHA256 is
  `a80010847ffc6a1beeff472d07fd50f75c8ce4be6df5bcfda1a16febaed40aa9`. This is small but
  causally cleaner than v1 and clears the gate for a one-step fresh-LoRA smoke; it does not
  justify scaling beyond one epoch or weakening the fixed/held-out promotion bars.

## Python final-patch v2 SFT verdict and next correction lane (2026-07-18)

- The bounded one-step smoke passed at context 12,288 with loss **0.2855**, grad norm **0.6088**,
  optimizer-step wall **22.66s**, and a valid 979,558,760-byte checkpoint. The authorized
  one-epoch fresh-r32 run then completed **10/10** steps in **194.1s** with final train loss
  **0.2708** and checkpoints 2/4/6/8/10. The only large gradient was an isolated step-2 value
  of 7.8068; it was not sustained. Cgroup peak was 38,655,193,088 bytes, swap peak zero, and
  GPU1 returned to 2 MiB. Log: `/tmp/python_verified_teacher_finalpatch_v2_s1.log`; adapter:
  `adapters/python_verified_teacher_finalpatch_v2_s1`.
- A precision-matched raw bf16 hard30 anchor scored **16/30**. The preregistered exposure sweep
  evaluated cp6 first (closest to the previous half-epoch sweet spot) and cp8 as the only fallback:

  | checkpoint | hard30 | non-empty | resolved | edit-reach | format |
  |---|---:|---:|---:|---:|---:|
  | raw bf16 fixed30 control | 16/30 | 27/30 | **17/30** | 29/30 | 0% |
  | final-patch v2 cp6 | **19/30** | 26/30 | **14/30** | 30/30 | 0% |
  | final-patch v2 cp8 | 15/30 | 27/30 | **15/30** | 30/30 | 0% |

  Both checkpoints pass the hard30 regression floor, but both fail the decisive Lite gate and
  are below the matched bf16 raw policy. The older 15/30 raw number was NVFP4; it must not be
  used to call cp8 a tie. Artifacts are `runs/hard30_python_finalpatch_{raw_anchor,v2_cp6,v2_cp8}_20260718`
  and `runs/smoke_python_finalpatch_v2_cp{6,8}_t07_lite_0_30`. No further epoch/checkpoint in
  this success-only final-patch lane is authorized. Raw bf16 + recovery remains policy.
- Exact cp6 pairing against raw is both-resolved 14, raw-only 3, cp6-only 0, neither 13. The three
  lost raw wins are `django-11620`, `django-11848`, and `django-11964`. Independent trajectory
  review shows **final-state selection**, not edit volume, as the failure: cp6 submitted empty
  after a successful repro in 11620, chose `datetime.now().year` instead of the mock-safe
  `utcnow().year` in 11848, and explicitly reverted a working fix then submitted empty after the
  repro failed in 11964. Cp6 was more deliberative (median 33 commands versus raw 25); both had
  median two source mutations. The 3-0 discordance is directional but small-sample noisy
  (two-sided exact p=0.25), so the next corpus targets the measured behavior rather than claiming
  a general capability loss.
- CPU inventory of banked learned trajectories found **52** content-unique, nonempty,
  F2P-scored raw failed patches for 33 of the 80 strict teacher tasks. The conservative surface
  is **38 exact edited-path pairs across 24 tasks** (plus four partial-overlap pairs that remain
  excluded by default). These support a verification-conditioned correction row: grounded native
  prefix, raw failed patch and real F2P failure as masked conditioning, then a supervised scoped
  checkout of the failed paths plus the independently tier-1 teacher patch. This is the next
  Python data lane. It must sequentially prove the raw patch applies and fails F2P, then prove the
  correction passes all F2P in the same fixture; no gold patch/test patch may be serialized.

## Python matched-failure revision corpus v1 (2026-07-18)

- Added `phaseD_sft/audit_matched_failure_revisions.py` and
  `phaseD_sft/build_matched_failure_revision_sft.py`, each with focused tests. The audit is
  deliberately single-container (`workers=1`) so the exact-byte Docker floor remains enforceable;
  CPU token validation remains threaded with one shared tokenizer. The combined related suite
  passes **136/136** locally and remotely. Independent review caught and closed two pre-run truth
  gaps: generic tracebacks can no longer count as test failures, and empty rc0 output can no longer
  count as a correction pass. Pytest credit is now fail-closed on reconciled JUnit XML; container
  launch uses a unique name plus cidfile, and any cleanup failure aborts the full audit.
- Holdout exclusion is content-addressed and blocking: the exact union contains **65** fixed30 plus
  frozen-heldout35 IDs and denies the five upstream evaluation repositories astropy, django,
  matplotlib, flask, and sympy. Current corpus overlap is zero IDs and zero normalized repos.
  Canonical exclusion-union SHA256 is
  `f0b97acee4cc39f95ea5b1b2d6448a2ed768d1920de2a019bb991bf13cfd65d8`.
- Sequential same-container replay examined **38** exact-path candidates across **24** tasks in
  47 seconds. It attempted 29 candidates, selected one verified candidate per task, and retained
  **14** corrections across 10 repositories; nine later candidates were skipped after their task
  already verified. Every retained raw patch applied to exactly the teacher paths, executed a real
  F2P failure, then passed the complete F2P contract after scoped checkout plus the independently
  tier-1 teacher patch. Audit JSONL:
  `data/python_matched_failure_revision_audit_v1.jsonl`, SHA256
  `33b13c2d8ca5789c81cb6221a6343222555a92724e58d9ede9364f32fef72c42`.
  Docker returned to **64,928,980,992 bytes** free, with no live audit containers; production ports
  remained green.
- The native revision row preserves the grounded teacher prefix, masks the failed raw patch and
  verifier-derived F2P action, conditions on the real compact failure observation, and supervises
  exactly one scoped rollback-plus-correction action. Final dataset:
  `data/python_matched_failure_revision_sft_v1` plus JSONL/manifest/rejected sidecar. It kept
  **14/14**, with zero content duplicates, zero budget rejects, token min/median/max
  **1,818/2,712/4,896**, and exactly one supervised correction per row. The Gemma format/loss gate
  passed all 14 with `failure_count=0`: 47 masked assistant turns, 14 supervised turns, zero
  fallback spans, and supervised-token min/p50/p90/max **219/324/472/1,544**. JSONL SHA256 is
  `9fd0ecb4d5facee00a44bae6555769ea50450c6debad7642d1cec48b34f67484`.
- The strict verifier rejected 15 attempted negatives before correction because no executed-test
  JUnit proof existed. A counts-only diagnostic over the 13 remaining unverified candidates found
  eleven source-path syntax errors, one source-path import error, and one other pre-test failure;
  these are patch-caused rather than host failures, but they are intentionally excluded from v1.
  They form a separately auditable source-error tier if later evidence justifies it. **Do not train
  the 14-row corpus standalone or silently upweight it.** The next decision is a small, explicit
  augmentation ratio against clean edit-first anchors, with a one-step smoke and the same raw-bf16
  fixed30/heldout promotion contract.

## Python matched-failure revision augmentation verdict (2026-07-18)

- The preregistered mixture is `data/python_teacher_revision_mix_v1`: 80 independently replayed
  tier-1 final-patch rows plus the 14 strict matched-failure revisions once each (**94 rows; 14.9%
  revisions**). It has zero content duplicates and zero overlap with the frozen 65-instance/five-
  repository evaluation denylist. Rendered-token min/median/max is **788/1,619/8,130**. The
  all-row Gemma format/loss gate passed with `failure_count=0`: 163 assistant turns masked,
  exactly 94 supervised, zero fallback spans, and supervised-token min/p50/p90/max
  **151/268/496/1,544**. Canonical JSONL SHA256 is
  `9c5f15b070394b80bf016235892e27a84cb5f5b58526cd8cd822b8f62dbdf724`.
- The fresh-r32/a32, frozen-base, 4-bit-loaded one-step smoke passed with loss **0.24618**, grad
  norm **0.67957**, a 34.11-second optimizer step, and a valid 979,558,760-byte checkpoint. The
  bounded one-epoch run then completed **12/12** steps in **244.7s**, final/mean train loss
  **0.3094/0.2456**, and retained checkpoints 3/6/9/12. Cgroup MemoryPeak was
  **38,655,492,096 bytes** (36.00 GiB), MemorySwapPeak **979,480,576 bytes** (0.912 GiB), GPU1
  returned to 2 MiB, and production remained green. Step 7 had one isolated pre-clip grad-norm
  spike of 24.64 and immediately returned to 0.696; it was not sustained. Log:
  `/tmp/python_teacher_revision_mix_v1_s1.log`; adapter:
  `adapters/python_teacher_revision_mix_v1_s1`.
- Evaluation used the precision/session-matched dynamic-LoRA contract: one bf16 raw-base vLLM
  server at 131,072 context, FP8 KV, think-open-v2, Gemma parsers, plus exactly one r32 candidate.
  Both aliases returned reasoning and a structured bash call with fingerprint
  `vllm-0.24.0-6c5b20bc`. Cp6 was the preregistered first window; cp9 was the only fallback because
  cp6 was neutral rather than a hard regression and had only five nonzero-LR updates. Cp12 and a
  second epoch were not evaluated.

  | checkpoint | same-server raw hard30 | candidate hard30 | non-empty | resolved | edit-reach | format |
  |---|---:|---:|---:|---:|---:|---:|
  | revision-mix cp6 | 19/30 | 18/30 | **26/30** | **17/30** | **29/30** | **0%** |
  | revision-mix cp9 | 19/30 | **15/30** | not run | not run | not run | not run |

- Cp6 passed the hard30 regression floor and produced high-quality action behavior: patch
  precision **17/26 = 65.4%**, 29/30 edit-reaching trajectories, and no format failures. It still
  tied the raw bf16+recovery policy at **17/30 resolved** (raw reference: 27 patches, 17 resolved,
  29 edit-reaching, 0% format), so it failed the strict `>=18` screening threshold and did not
  trigger a same-server raw Lite rerun or heldout35 promotion test. Cp9 then failed hard30 by four
  points versus its fresh raw anchor, below the allowed raw-minus-two floor, so its Lite run was
  correctly skipped.
- Artifacts: cp6 hard30
  `runs/hard30_python_teacher_revision_mix_v1_20260718_090617/`, cp6 Lite
  `runs/smoke_python_teacher_revision_mix_v1_cp6_t07_lite_0_30_20260718_093512/`, and cp9 hard30
  `runs/hard30_python_teacher_revision_mix_v1_cp9_20260718_102909/`. Comparison SHA256 values are
  `cc0646a3def95162a51f54b741e69c84e5d754e15e8811f843b5d8045cd3d076` (cp6 hard30) and
  `8ec774bd8885530dd9c47c41d5e747ab3483ba9f7b2ec39250f73700188bf442` (cp9 hard30); cp6
  `preds.json` SHA256 is
  `ade726d029404191610eeff13aa447bd0e37f79d2462c1d2c0a74083b6d2c71c`.
- **Verdict:** the strict revision rows improved submission decisiveness/precision but did not
  increase Python resolution, while modest extra exposure regressed hard30. Raw bf16+recovery
  remains the Python policy. Bank this clean negative result; do not upweight the 14 revisions,
  add epochs, or sweep cp12. The next Python improvement must add a materially different learning
  signal rather than another small success/revision SFT mixture.

## Python replay-verified outcome-KTO lane (2026-07-18)

- This lane changes the learning signal rather than imitating another successful trajectory shape:
  KTO rows supervise one native mini-SWE patch action, with desirable/undesirable labels accepted
  only after exact Docker apply plus F2P replay. The closed expert banks plus model-patch mutations
  project only about 419 rows/100 undesirable (optimistic 474/143), below the immutable production
  gate of 500/150, so no training has started.
- The second acquisition pool is
  `data/expert_iter2_pool40_v2.jsonl`: 40 nonblank local-image SWE-Smith tasks, zero old/eval/hard
  or evaluation-repository overlap, output SHA256
  `4876b22fe7e3c1ac86c867b89aa68e7021450ec60b5d6b163bbf0042ce8bcab8`.
  Sampling now has a schema-v2 fail-closed contract: one exact pool-byte snapshot, effective
  model/API and ordered rollout hash frozen atomically before requests, exclusive outdir lock,
  exact ledger-metadata validation, fsynced appends, and an exact final-ledger hash. Independent
  adversarial tests reproduce and close the former concurrent-resume/config-drift bypasses.
- Both one-task smokes completed **4/4 resolved, 4/4 nonempty, zero errors**. The schema-v2 smoke
  also published an exact matching ledger hash. Observed rollout walls were sufficiently long that
  64 rollouts at workers 8 project beyond two hours, so collection is tranche-gated rather than
  silently overrunning the standing GPU window.
- First-tranche collection `runs/expert_iter2_kto_pilot8a_v2_strict` was exact-stopped at the
  two-hour boundary with **22/32** fsynced rows across seven tasks: 7 resolved, 14 nonempty patches,
  6 `LimitsExceeded`, zero infrastructure-error fields. Ledger SHA256 is
  `6e8a9204396d6dc667c17778c0754b6f1ebeb3b7a9de5b7b6e2d8a1e3121e6fa`.
- Exact replay of the six structurally eligible tasks passed its preregistered proportional gate:
  `data/python_swe_outcome_kto_pilot8a_partial.jsonl` retained **10 rows = 5 desirable / 5
  undesirable** versus required 9/5, with five rejected
  (`duplicate_prompt_completion=1`, `f2p_execution_unproven=3`, `ledger_label_mismatch=1`). Token
  max is 1,952/4,096; format/loss failures, truncations, duplicates, and leakage hits are all zero.
  Data/manifest/rejected SHA256 values are respectively
  `3882eac4b647f436c3ed4fcda61857ef676c9414ba6bd90f26139c948676675a`,
  `9f67642662726993d63a5ae8b98ba4e2148498a042fd3f9c3e3bb1ecd3daaf9f`, and
  `0ebffe3d7deae496d677e33fc49a008d278ac44ad4e302220636e6804049bcef`.
  Independent audit reconstructed all 15 candidates and passed every hash/arithmetic/leakage check.
- Because 10/5 passed, only the ten missing rollouts from the same eight tasks are now resuming
  under a fresh <=2-hour gate (wrapper/Python PIDs **2781812/2781813**; log
  `/tmp/expert_iter2_kto_pilot8a_v2_strict_resume.log`). On completion, replay the full first-eight
  bank and require at least **12 useful / 7 verified undesirable** before collecting tasks 9--16.
  The eventual combined first-16 gate remains 24/14; production remains fixed at 500/150.

### Outcome-KTO first-eight final gate (2026-07-18)

- The resume completed cleanly at **32/32 unique rollouts**: 12 resolved, 20 nonempty patches,
  24 submitted and 8 limits-exceeded. The schema-v2 complete manifest binds ledger SHA256
  `ca9fa86ce893b9ddac78d2ab8e795f1faf12bb9ff125cc5d4c89c22c8d262160`.
- Full exact replay originally retained **16 rows = 10 desirable / 6 undesirable**, one short of
  the preregistered 12/7 gate. A six-row sanitized diagnostic (SHA256
  `4c610674ee1f2655b095c649249830920e747a3ea4e04c0980b5f9a7788cbc78`) proved three Pydantic
  drops are broken fixtures, two synthetic hard negatives genuinely pass, and one DeepDiff patch
  destroys test collection while its clean fixture runs 4/4 F2P tests successfully.
- `build_swe_outcome_kto.py` now has a fail-closed A/B tier for that last class: expected-negative
  rc2/4 import/collection failure is accepted only when the exact clean fixture executes the same
  F2P contract with a proven semantic outcome. It records sanitized execution hashes/counts and
  `baseline_observed_label`; Pydantic remains rejected. Real `subprocess.TimeoutExpired` is also
  normalized to `infrastructure_timeout`. Independent review passed; remote related tests are
  **201 passed**.
- The unchanged 12/7 replay still **FAILED at 16/6**. The DeepDiff row passed replay under
  `candidate_caused_collection_failure_v1`, but its destructive 42,975-character deletion has a
  **12,363-token completion by itself** (13,052-token theoretical minimum with a one-line source
  prompt), so it cannot fit the immutable 4,096-token cap and was correctly rejected as
  `token_budget_exceeded`. Final drops: duplicate 1, F2P-unproven 3, label-mismatch 2, token-budget
  1. Atomic publication held: no first-eight production dataset/manifest/rejected artifacts exist.
  Failed-run log SHA256 is
  `1b2d5f6fcdd3e7c464fa0d4adf468d942cd11792e412e6a4c575bb3e6d7ec95b`.
- **Decision:** stop; do not collect tasks 9--16 and do not train KTO. Source-context compaction
  cannot recover this row because the completion alone is over 3x the cap. A future attempt must
  obtain additional compact, replay-proven negatives (for example a separately preregistered
  no-gold mutation strategy); it may not raise the 4,096 cap or truncate supervised completions.

### Outcome-KTO compact-negative gate reopened (2026-07-18)

- The preregistered no-gold `patch_mutation_v2` fix enumerates every safe added-line mutation in
  frozen strategy-major/source-minor order, attempts at most 16 variants per task, and applies the
  five-row cap only after exact replay/render/token/dedup gates. Canary selection now happens before
  augmentation, so generation and rejection counters describe only the selected seven-task subset.
  Local related tests passed **215/216** with one optional environment skip; the synced remote suite
  passed **216/216**, and independent final review found no open issues.
- The exact pilot8a replay completed rc0 in **84.3s** and PASSED the unchanged gate at **19 rows =
  10 desirable / 9 undesirable**, versus required 12/7. All 19 rows are exact-replay verified;
  token max is **1,952/4,096**; format/loss failures, leakage hits, duplicate output rows, and
  truncations are all zero.
- Five new compact undesirable rows survived from **three repositories** (Trio, JSONSchema, and
  Pydicom), clearing the frozen >=2 rows / >=2 repositories canary. The builder enumerated 61
  variants, generated/attempted 47, retained 5, deduplicated 12, and ceiling-dropped 2; all retained
  mutations were `rename_called_identifier` and failed the exact `direct_f2p_v1` contract.
- Artifacts: `data/python_swe_outcome_kto_pilot8a_compact_v2.jsonl` SHA256
  `e7bd3f1646e2981d14f7da9b6abbff755392af7954c9ff3ba3dc05c7187ef765`, manifest SHA256
  `231aaae367a66f80d7fb11c3d94a12927a415baf0bd5a5cb6c505157f576fc5c`, rejected SHA256
  `5924874836d82157ef6cfb2f77b616d26a0f3a0c9987f0b0cdbfe415dd07592a`.
  Postflight: no replay containers, Docker loopback **61 GiB free**, and production ports
  8000/8101/8103/8104 all returned HTTP 200.
- **Decision boundary:** tasks 9--16 are now eligible, but remain stopped pending a separately
  authorized GPU acquisition window; no KTO training has started.

### Outcome-KTO pilot8b acquisition launched (2026-07-18)

- The schema-v2 collector now supports immutable original-pool slicing: `--offset` is applied before
  `--limit`, recorded in the resume manifest, and checked on resume. Strict TDD and independent
  review passed; 262 related tests passed locally and again on the host. A host-side gate caught and
  corrected one test sync-path mistake before any rollout process was launched.
- Tasks 9--16 launched from the unchanged `data/expert_iter2_pool40_v2.jsonl` snapshot as
  `runs/expert_iter2_kto_pilot8b_v2_strict`, with `offset=8`, `limit=8`, four samples/task,
  temperature 1.0, workers 8, and model `openai/gemma4-kto-pilot-raw` on the existing 131,072-context
  raw-base server. Frozen pool SHA256 is
  `4876b22fe7e3c1ac86c867b89aa68e7021450ec60b5d6b163bbf0042ce8bcab8`; the exact 32-rollout ID
  SHA256 is `979a5535ee1392d633d6689e5af400b419204353e3d7c0038abd8a5679aaec23`.
- Launch PIDs are outer wrapper **2927848**, bounded wrapper **2927862**, timeout **2927863**, and
  Python collector **2927864**. Log: `/tmp/expert_iter2_kto_pilot8b_v2_strict.log`. The timeout is
  6,600 seconds, reserving ten minutes inside the two-hour rail for exact cleanup and postflight;
  it does not auto-chain a resume.
- Launch preflight: all eight images local, Docker 61 GiB free, root 67 GiB free, RAM 45 GiB
  available, no conflicting acquisition process/container, and production ports
  8000/8101/8103/8104 all HTTP 200. The running manifest exactly matches every frozen identity
  field. Historical throughput projects about 18--24/32 completed rollouts in this window, so a
  separately bounded resume is likely.
- After 32/32, exact-replay the combined first16 bank and require the unchanged **24 useful / 14
  undesirable** gate. Pilot8a contributes 19/9, so pilot8b must add at least 5/5 after combined
  dedup/replay; no KTO training begins before the combined gate.

### Outcome-KTO pilot16 gate passed; production replay running (2026-07-19)

- Pilot8b completed before its cap in **1h46m06s**, rc0: 32/32 unique rollouts over eight tasks,
  21 resolved, 28 nonempty patches, 29 submitted, 3 limits-exceeded, and zero errors. Ledger SHA256
  is `ba9301dd154c33ce81743ab5a3f20c900e7d9a1677e11a5f9a0edb67c1b249ae`; manifest SHA256 is
  `ff791c6bdfea4bc3e8c680510fd9410722ef25e8e1e67c98596972d5c21faddd`. Independent validation
  reconstructed the exact offset-8/limit-8 rollout set, matched the ledger hash and LF termination,
  and found no live sampler/container afterward.
- The immutable combined bank contains all validated pilot8a bytes followed by all validated
  pilot8b bytes: 64 unique rollouts over the exact first 16 pool tasks, with no outcome/content
  filtering or reordering. Pilot8a's pre-offset schema is recorded as
  `legacy_prefix_implied_0` and independently bound by its limit-8 rollout-ID hash rather than
  rewriting its completed manifest. Combined bank SHA256 is
  `448d86469543ea906bf062c9705f84f5010a8040595c746409f03054d557ba4e`; sidecar SHA256 is
  `88934fcfb0ce9d28290d26bf1e581796cd38a4a8bb7b227b950e6940f480a832`.
- The exact combined first16 Docker replay **PASSED** in 157.7s at **47 rows = 28 desirable / 19
  undesirable**, versus frozen 24/14. All 47 are replay verified; max tokens are 2,070/4,096;
  format/loss failures, leakage, duplicate output rows, and truncations are zero. Ten compact
  negatives survived across seven tasks/six repositories. Dataset/manifest/rejected SHA256 values
  are respectively `600cc42c855e2e5a340d0d50af377a370a870bfb95f504375f24d88608a1d79a`,
  `70f6c02ab50af4c7bbcea40bc27de88eb1c5cf20ec2dd5f6ee8f9df3d1bd0811`, and
  `561faba6a3a5fb30a19bac49ab86449105f45b60478956d6c5ce736aba2a5bba`.
- Before spending another GPU window, the full fixed production replay is measuring whether the
  closed k4/raw20/raw60 banks plus pilot16 already clear **500 rows / 150 undesirable** under v2
  compact-negative enumeration. Wrapper/Python PIDs are **3054633/3054637**; log is
  `/tmp/python_swe_outcome_kto_v1_compact_v2.log`. It loaded 1,633 candidates across 14 local
  images. No KTO training has started.
- Post-gate/pre-production safety: Docker 61 GiB free, no stale replay containers, and production
  ports 8000/8101/8103/8104 all HTTP 200. The existing raw 8012 server remains intentionally live.

### Outcome-KTO production gate passed; one-step smoke active (2026-07-19)

- The fixed production replay completed in **1h09m09s** and passed its immutable gate at **518 rows
  = 290 desirable / 228 undesirable**. All 518 rows are Docker replay verified; maximum rendered
  length is 3,206/4,096 tokens; format/loss failures, leakage, duplicate output rows, and truncations
  are zero. The replay used only the original k4 bank after v2 compact-negative enumeration already
  cleared the threshold; raw20, raw60, and pilot16 remained listed but were not needed.
- Dataset/manifest/rejected SHA256 values are respectively
  `12fcc94005d128004904600176c0a3efb3ec6ded0d88b7d223d236049f6f1f56`,
  `cab181f7d8732d605ef8e4c0a085a281eacf16d892816938e2e8e41c47bb0e2b`, and
  `5838dfbe33de5788587290e0d492bc5401dad665a58983fae763161b3284056e`.
- A live fresh-LoRA guard now requires exact enabled-adapter versus disabled-adapter logits before
  data or trainer construction. The first bounded smoke failed closed before comparison because
  Gemma's Unsloth-patched processor interprets a positional string as `images`, leaving `text=None`.
  The minimal `text=` fix is regression-locked by a keyword-only processor test; the full remote
  Phase-E suite passes **269/269** with one optional skip.
- Corrected bounded unit `python-outcome-kto-v1-smoke2.service` launched on GPU1 after a green
  preflight (2 MiB GPU1, 50.5 GiB MemAvailable, 67 GiB root free, port 8012 free, production ports
  all HTTP 200). It retains `MemoryHigh=36G`, `MemoryMax=45G`, `MemorySwapMax=16G`, and a one-hour
  runtime cap. Do not scale until checkpoint-1, finite nonzero update, exact reference equality,
  measured GPU/cgroup peaks, clean GPU release, and production health are all verified.
- The next guarded retry proved the live reference contract: **1,310,720 logits matched exactly**
  with `max_abs_diff=0`, and the 518-row manifest/data gate passed. It then failed before optimizer
  construction because TRL KTO called the multimodal Gemma processor positionally inside dataset
  tokenization, which binds the prompt to `images` and leaves `text=None`. Peak evidence was
  20,170 MiB GPU1, `MemoryPeak=38,655,418,368`, and `MemorySwapPeak=920,694,784`; GPU1 returned to
  2 MiB and production stayed green. The trainer must receive the processor's underlying text
  tokenizer instead. Also force `dataset_num_proc=1`: Unsloth auto-selected 28 process workers,
  violating the established no-multiprocess-tokenizer safety rule and needlessly reaching the
  cgroup high-memory boundary. Do not launch another retry until both boundaries are regression
  tested.
- `python-outcome-kto-v1-smoke4.service` then passed every integration/memory gate in **126.5s**:
  model load 61.43s, sequential trainer preprocessing, one 17.35s optimizer callback, finite loss
  0.47058 and grad norm 92.80, checkpoint-1 (979,558,760-byte adapter), sampled GPU1 peak
  33,616 MiB, `MemoryPeak=27,784,212,480`, `MemorySwapPeak=0`, clean GPU1 release to 2 MiB, and
  production all green. The exact reference check remained 1,310,720 elements / max diff zero.
- **The nonzero-update gate nevertheless failed honestly.** Direct safetensors inspection found all
  410 saved `lora_B` tensors (135,331,840 elements) still exactly zero. `training_args.bin` exposed
  the root cause: Unsloth supplied `warmup_steps=0.1`; at max_steps=1 the optimizer state reached
  step 1 but had zero nonzero first moments, and the scheduler/optimizer LR was 0. Set
  `warmup_steps=0` explicitly and regression-lock a bounded live LoRA-B invariant: all trainable B
  tensors finite/zero before training, the identical set finite with at least one nonzero value
  after training. Do not scale from smoke4 despite its otherwise clean checkpoint.

### Outcome-KTO final smoke passed; bounded 25-step tranche launched (2026-07-19)

- `python-outcome-kto-v1-smoke6.service` ran the exact production code path and passed every
  blocking gate. Model load was 59.98s, the optimizer step was 12.565s, trainer time was 14.33s,
  and total wall time was 121.87s. Loss was 0.470578, grad norm 93.4753, and LR 5e-7.
- The disabled-adapter reference remained bit-exact across 1,310,720 logits (`max_abs_diff=0`).
  All 410 trainable LoRA-B tensors / 135,331,840 elements started finite and exactly zero; after
  the step, 135,330,949 elements were nonzero with max absolute value 4.9998482e-7.
- Atomic publication passed: the sibling `.inprogress` quarantine was renamed only after the live
  update gate, canonical and checkpoint adapter hashes both equal
  `779181b290554995471abaf30e7968e84ef0e259e66e1fb9a683970f4949acf1`, and no quarantine remains.
  Sampled GPU1 peak was 34,440 MiB; framework peak was 34,267,433,984 bytes; GPU1 returned to 2 MiB
  and all production ports stayed HTTP 200. The post-exit systemd `MemoryPeak` reset is not usable,
  so the scale run samples cgroup peaks while the unit is active.
- The bounded scale unit `python-outcome-kto-v1-s1.service` launched with main PID 3392157 and exact
  command shape: physical batch 2, gradient accumulation 8, effective batch 16, 25 optimizer steps,
  checkpoint every 5 steps, and the fixed 518-row dataset. This exposes 400 rows (77.2% of one
  epoch) while respecting the standing 25-step and two-hour bounds. Smoke-derived ETA is about
  44 minutes. Output is `adapters/python_swe_outcome_kto_v1_s1`, log is
  `/tmp/python_outcome_kto_v1_s1.log`, and live resource/GPU telemetry is in the matching
  `/tmp/python_outcome_kto_v1_s1_{resource_monitor.txt,gpu.csv}` files.
- Preflight was green: GPU1 2 MiB, MemAvailable 50.3 GiB, root 58 GiB free, port 8012 free, and
  production ports 8000/8101/8103/8104 all HTTP 200. The unit is capped at
  `MemoryHigh=36G`, `MemoryMax=45G`, `MemorySwapMax=16G`, `RuntimeMaxSec=6300`; no cgroup limit was
  raised.

### Outcome-KTO 25-step tranche passed; evaluation waiting on an external GPU owner (2026-07-19)

- `python-outcome-kto-v1-s1.service` completed all 25 optimizer steps successfully in 1,389s
  (23.15m), with mean loss 0.4326 and final loss/grad norm 0.4465/44.84. Finite isolated gradient
  spikes recovered on later steps; there was no NaN, nonfinite loss, OOM, or production failure.
- The final live update gate passed: all 410 LoRA-B tensors / 135,331,840 elements are nonzero,
  with max absolute value 8.6970314e-05. Checkpoints 5/10/15/20/25 were observed only in the
  `.inprogress` quarantine while training. The canonical
  `adapters/python_swe_outcome_kto_v1_s1` was published after the gate and the quarantine is absent.
  Final and checkpoint-25 adapter hashes both equal
  `2a4e6e738eb4ae3194f0d7b331bdac20e659947d21cc49e50bab39fc35bd73a5`.
- Measured peaks were 51,236 MiB GPU1, 38,655,488,000 bytes cgroup memory, and 651,976,704 bytes
  cgroup swap. Postflight was GPU1 4 MiB, production all HTTP 200, and about 52.7GB root free.
- The first matched BF16 raw+checkpoint-25 vLLM launch correctly failed before serving and is
  retained at `runs/python_swe_outcome_kto_eval_20260719_040057`. An independently launched
  `teacher_sft_v1` process (PID 3405306) took about 26 GiB of GPU1 at 04:00:34, after KTO postflight
  and before vLLM reserved its requested 88.32 GiB. vLLM saw only 76.28/94.97 GiB free and exited;
  this is a resource collision, not an adapter load or evaluation failure. Do not kill or alter the
  external training. Resume the preregistered same-process hard30/fixed-SWE-Lite gate only after
  its exact PID exits and GPU1 is below 1 GiB.
- Independent streaming audit of that external training's input found a blocking semantic defect,
  documented at `.superpowers/sdd/teacher-sft-v1-readonly-audit.md` and reproduced separately.
  `teacher_train_mix_v2` is 1,006 rows: 6 Fable rows and 1,000 Open-SWE Minimax rows. The converter
  discards rich editor-tool payloads and retains only `arguments.command`, so 965/1,000 external
  rows teach bare pseudo-shell calls: 2,811 `view`, 1,285 `str_replace`, 515 `create`, and one
  `insert`. Of those rows, 344 have no detectable retained edit, 527 contain identical consecutive
  commands, and 16,485/24,461 assistant turns (67.4%) have empty content. These counts were
  independently recomputed directly from the live HF dataset using the repository's standard
  `command_trace_quality_report` semantics.
- The live external schedule is fresh r32/a32, full assistant-turn loss, 2 epochs / 126 steps, and
  its measured first step was 306.2s, projecting about 10h38m remaining. This is a high-confidence
  shape-mirroring risk and blocks the ready KTO evaluation for the same interval. Because stopping
  an unowned running GPU job is destructive, require an explicit user decision before terminating
  exact PID 3405306; do not infer promotion from its training loss.

### Outcome-KTO checkpoint-25 missed the behavior gate; midpoint sweep active (2026-07-19)

- The matched same-process BF16 evaluation completed at
  `runs/python_swe_outcome_kto_eval_20260719_060234`. Hard30 was raw **19/30** versus
  checkpoint-25 **17/30**, exactly the allowed raw-minus-two regression floor.
- On the fixed 30-case SWE-Lite recovery/self-retry harness, checkpoint-25 produced **27/30
  non-empty patches**, reached an edit in **29/30**, had **0% format errors**, and resolved
  **15/30** according to the official scorer. Its median first source edit was command 15 and only
  9/30 edited by command 10. It therefore failed the required >=18/30 resolved promotion gate;
  raw+recovery remains the Python policy.
- To distinguish a bad endpoint from an unproductive objective, checkpoints 5/10/15/20 are being
  swept against a fresh raw BF16 hard30 anchor in one 8012 server. The immutable artifact root is
  `runs/python_swe_outcome_kto_midpoint_eval_20260719_073151`; raw repeated at **19/30**. Only the
  best midpoint at or above the raw-minus-two floor will receive one fixed30 evaluation. No further
  KTO training is authorized unless that midpoint clears the behavior gate.

### Open-SWE rich conversion and clean v3b mixture passed all CPU gates (2026-07-19)

- The original `data/open_swe_sft_v3` all-drop result was a converter defect, not absence of usable
  data: 22,253 resolved Python trajectories matched, but the old parser rejected missing SWE-agent
  result IDs, OpenHands `execute_bash`, and editor paths rooted under the declared workspace. The
  fixed converter is fail-closed for ambiguous shell/heredoc/path semantics, preserves source
  reasoning, translates rich editor calls into executable bash, and has machine-readable progress
  with ETA. Local Phase-D/teacher-platform verification is **520 passed / 1 optional skip**; the
  focused host teacher-platform suite is **123 passed**.
- Immutable rebuild `data/open_swe_sft_v3b` scanned **207,489/207,489** rows in 1,153s, matched
  **22,253** resolved Python rows, and ingested **2,756** unique rows. Its sources are 2,459
  SWE-agent Minimax and 297 SWE-agent Qwen rows; later OpenHands rows share task IDs and lose the
  deterministic first-config-wins dedup. Pairing failures and bare editor pseudo-commands are zero.
  The 553 auditable conversion drops remain in `data/open_swe_sft_v3b.manifest.json`.
- Strict edit-decisive filtering retained **1,608/2,756** rows and dropped 1,148: 675 no edit, 307
  first edit after command 10, 244 read-streak violations, and 54 identical-command repeats (a row
  may have multiple reasons). Content duplicates are zero; the filtered content SHA256 is
  `6dae4ccf802a7129a2d97a63ed4ae11eccb9b007874393c0d5a43cbfa24e9d2b`.
- The deterministic 16,384-token mixture is
  `data/python_swe_open_swe_v3b_mix_budget16384` with manifest beside it. It contains **2,145
  unique tasks/content hashes**: 858 Open-SWE (40.0%), 1,157 SWE-Smith, 124 verified patch-decision
  anchors, and six strict expert-iteration anchors across 348 selected external repositories.
  Evaluation overlap is zero, maximum tokens are **16,353**, and JSONL SHA256 is
  `3e1a0a1014bf8865a2e8325fc33918459a085229747d36bdd4ddd4cbaf45e57c`.
- The builder/verifier integration gate first exposed that the builder publishes canonical JSONL
  while the verifier and trainer accepted only HF `save_to_disk` directories. A TDD fix added one
  shared loader for both representations, retained single-process tokenizer preprocessing, and is
  covered by **18 focused / 525 related** passing tests. The canonical JSONL is now the direct
  training input; `.venv-train` loaded all 2,145 rows and its exact 1,000-sample format/loss gate
  passed with **failure_count=0** (`/tmp/open_swe_v3b_mix_format_gate_jsonl.log`). The intermediate
  `_hf` copy made by the newer eval Datasets version is intentionally invalid for training; the
  `_train_hf` copy is compatible but unnecessary. Full behavior audit: first-edit median **5**,
  100% by command 10, read-streak median/max **4/5**, and zero identical consecutive repeats.

### Outcome-KTO midpoint hard30 sweep completed; checkpoint-15 fixed30 active (2026-07-19)

- Same-server hard30 scores were raw **19/30**, checkpoint-5 **17/30**, checkpoint-10 **16/30**,
  checkpoint-15 **17/30**, and checkpoint-20 **17/30**. Checkpoint-10 hard-failed the raw-minus-two
  floor; the other three only tied it. Checkpoint-15 was selected as the center of the tied
  floor-pass region, before checkpoint-25's negative fixed30 result.
- The single decisive recovery/self-retry fixed30 launched with wrapper PID **3645665**, eight
  workers, temperature 0.7, and the exact 131,072-context same-server adapter. Artifact:
  `runs/python_swe_outcome_kto_midpoint_eval_20260719_073151/smoke_kto_cp15_fixed30`. Stop KTO if it
  resolves fewer than 18/30; do not use another KTO training tranche to explain away the result.

### Outcome-KTO checkpoint-15 passed the fixed30 screen; matched raw control active (2026-07-19)

- Checkpoint-15 finished the exact fixed recovery/self-retry 30-case slice at **18/30 resolved**,
  **28/30 non-empty patches**, **30/30 edit reach**, and **0% format errors**. Its first source edit
  remains late: median command 16 and only 8/30 by command 10. Official report:
  `openai__kto_cp15.smoke_kto_cp15.json`; generation/scoring artifacts are under
  `runs/python_swe_outcome_kto_midpoint_eval_20260719_073151/smoke_kto_cp15_fixed30`.
- The 30th case (`django__django-11019`) exposed a still-live submission failure mode: a real
  696-byte one-file patch existed in the container, but the policy spent about 109 minutes on
  repeated generations and ultimately submitted empty. Direct server telemetry proved it was not
  hung (about 19 generated tokens/s, one running request, GPU1 100%). An immutable 29-case snapshot
  already scored 18 resolved / 10 unresolved / one empty, so the final case did not change the
  screen verdict.
- Because 18/30 is only the preregistered screen, not a promotion by itself, a same-server raw-BF16
  fixed30 control launched as wrapper PID **3845246** with the identical 131,072 context, template,
  recovery environment, temperature 0.7, and eight workers. Artifact:
  `runs/python_swe_outcome_kto_midpoint_eval_20260719_073151/smoke_gemma4-kto-raw-bf16_fixed30`.
  Only after the raw control is banked should the heldout35 candidate/raw comparison proceed.

### Outcome-KTO checkpoint-15 passed the preregistered combined-65 promotion gate (2026-07-19)

- The matched same-server BF16 evaluation is complete under
  `runs/python_swe_outcome_kto_midpoint_eval_20260719_073151`. Hard30 remained only a regression
  floor: raw was **19/30** and checkpoint-15 was **17/30**, exactly the permitted raw-minus-two
  boundary. Checkpoint-15 then passed the fixed screen at **18/30 resolved**, versus raw **15/30**.
- On the disjoint registered heldout-35, checkpoint-15 resolved **17/35** with 31 non-empty patches,
  while raw resolved **16/35** with 30 non-empty patches. Both had zero format errors. The final
  preregistered combined result is therefore checkpoint-15 **35/65** versus raw **31/65**, delta
  **+4**, exactly the promotion threshold with no format degradation. Machine-readable verdict:
  `runs/python_swe_outcome_kto_midpoint_eval_20260719_073151/combined65_verdict.json` (SHA256
  `c5cedb01974284128089bc6df4e363a3aeac8a60d611ba17c14a926b32f8295c`). Checkpoint-15 is the
  promoted Python adapter from this lane; later checkpoint-25 remains rejected.
- The raw heldout official report is retained inside the run at
  `smoke_gemma4-kto-raw-bf16_heldout35/report_gemma4-kto-raw-bf16/official_report.json`; its 35-case
  summary records 16 resolved, 30 non-empty, 35 edit-reaching trajectories, first-edit median 23,
  and zero format errors. A TDD harness fix now copies every successful scorer report atomically to
  the corresponding run directory so repeated slices of the same served name cannot overwrite the
  only durable report. The focused test is green **6/6** locally and on the host.
- The next independent Python lever remains the fully gated clean Open-SWE v3b mixture. Run a
  one-step fresh-LoRA smoke first; only a clean live update/checkpoint and resource postflight may
  advance to a checkpointed tranche of at most 25 optimizer steps. KTO promotion does not waive
  the Open-SWE smoke-first gate.

### Clean Open-SWE v3b one-step SFT smoke passed (2026-07-19)

- The bounded fresh-r32 smoke `python-open-swe-v3b-smoke.service` trained directly from the
  canonical JSONL `data/python_swe_open_swe_v3b_mix_budget16384` with max sequence 16,384,
  assistant-turn loss, effective batch 16, LR 2e-5, frozen 4-bit base, and the verified
  `--sm120-attn` path. All **2,145/2,145** rows retained supervised tokens.
- Model load took **83s**. The full optimizer step took **452.37s** and finished at loss
  **0.0450251**, grad norm **0.2229**, with `checkpoint-1` written. Canonical and checkpoint adapter
  SHA256 are both `2e6891cda60df23c2340275ddae76e7c46e0063466240ce4093e42ace00def45`.
  All 410 LoRA-B tensors are finite; 135,330,778/135,331,840 elements became nonzero and max
  absolute value is 1.99994e-05, proving a live update from the fresh zero-B initialization.
- Sampled peaks were **42,018 MiB GPU1**, **38,655,455,232 bytes** cgroup memory, and
  **988,491,776 bytes** cgroup swap; minimum host MemAvailable was about **24.2 GiB**. Postflight
  returned GPU1 to 2 MiB and port 8000 stayed green. The smoke output is
  `adapters/python_swe_open_swe_v3b_smoke` and the run log/telemetry are
  `/tmp/python_open_swe_v3b_smoke{.log,_gpu.csv,_resource.log}`.
- Scaling remains checkpoint-gated. At the measured step time, 15 optimizer steps plus load/save
  overhead project just under two hours and expose 240 examples; retain checkpoints 5/10/15 for
  the hard30-floor and SWE-Lite promotion sweep rather than committing to a longer blind run.

### Open-SWE v3b scale stopped safely at checkpoint-10; behavior gate active (2026-07-19)

- A fresh 15-step scale launched as `python-open-swe-v3b-s1.service` with the smoke-proven config,
  warmup 2, and checkpoints every five steps. Checkpoint-5 and checkpoint-10 are complete at
  `adapters/python_swe_open_swe_v3b_s1/checkpoint-{5,10}`. Through step 10, loss stayed finite at
  0.0450-0.0696; step 6 had one isolated grad-norm spike to 13.61 followed by 0.116-0.539, so there
  was no sustained divergence. Checkpoint-5 adapter SHA256 is
  `f99d9be90148b861e1f33cb9050a4123d4d213016c1d2b30c9945f9262e0f531`; checkpoint-10 is
  `06c7e785924cc3809fcf98ce5067368e57443c2b137157b206a9895bae686751`.
- The run was deliberately terminated by exact PID after retrospective telemetry inspection found
  **48 ten-second samples below the user-set 2-GiB MemAvailable stop floor** during steps 2-3,
  including a 53,368-KiB minimum. Cgroup memory peaked at 38,655,492,096 bytes and cgroup swap at
  2,548,887,552 bytes. Another host workload was active during the pressure interval and was gone
  by discovery, but its exact historical PID is not recoverable; do not attribute the event more
  specifically without evidence. Post-stop MemAvailable recovered to 54,148,976 KiB, swap use to
  611,840 KiB, GPU1 to 2 MiB, and port 8000 stayed green. Future training must use an active
  exact-PID RAM watchdog, not telemetry-only sampling.
- Rather than resume blindly or repeat already-seen rows, checkpoints 5 and 10 advanced to the
  same-server behavior gate against the promoted KTO checkpoint-15 policy. At 131,072 context,
  hard30 was KTO **17/30**, Open-SWE checkpoint-5 **16/30**, and checkpoint-10 **17/30**; both pass
  the KTO-minus-two regression floor. Checkpoint-10 won the tie and is now running the fixed
  30-case SWE-Lite self-retry smoke at temperature 0.7 and eight workers under
  `runs/python_swe_open_swe_v3b_eval_20260719_150700/smoke_open_swe_cp10_fixed30`.

### Open-SWE v3b SFT is subtractive; KTO checkpoint-15 remains Python policy (2026-07-19)

- The official checkpoint-10 fixed30 result is **15/30 resolved**, **29/30 non-empty patches**,
  **30/30 edit reach**, first-edit median 18.5, 10/30 edits by command 10, and **0% format
  errors**. The retained report and reproducible summary are under
  `runs/python_swe_open_swe_v3b_eval_20260719_150700/smoke_open_swe_cp10_fixed30`; summary SHA256 is
  `5df3463ccc934f9d3679ebd83a83f2a5a00248eacfa33e8f7be0cec68c0f7c61` and official-report
  SHA256 is `80336db8331a26c6e16d2aec89660cf8deac4a5bbc314e90e2d4f79e0e1dd6d3`.
- This misses KTO checkpoint-15's 18/30 fixed screen and therefore stops without further training,
  a matched KTO rerun, or heldout expansion. Per-instance comparison is stronger than the aggregate:
  all 15 Open-SWE resolutions are contained inside KTO's 18 resolved IDs; Open-SWE adds **zero**
  unique resolves and loses `astropy__astropy-14365`, `astropy__astropy-14995`, and
  `django__django-11848`. It is strictly dominated on this slice, so adapter merging/crossover with
  this checkpoint has no observed complementarity to harvest.
- Interpretation: aggressive edit-first filtering plus clean rich tool conversion successfully
  taught action frequency and formatting, but not final-patch correctness. More Open-SWE SFT steps
  are unsupported; the next Python experiment must use execution outcomes/negative evidence rather
  than further trajectory-shape imitation. GPU1 was released by exact PID to 2 MiB and port 8000
  remained green.
- `phaseD_sft/ram_watchdog.py` now closes the telemetry-only safety gap for future training: it
  samples MemAvailable, emits flushed progress, and terminates only its configured exact PID on the
  first sub-2-GiB or unreadable sample. It is covered by 6/6 focused tests locally and remotely;
  the full local Phase-D suite is **408 passed / 1 skipped / 5 subtests passed**.

### Outcome-KTO iteration-2 acquisition probe: full trajectories are too slow and negative-starved (2026-07-19)

- The promoted checkpoint-15 policy was sampled only on the decontaminated Expert Iteration
  training pool, never on fixed30, heldout35, hard30, or their denied repositories. An eight-worker
  bounded pilot retained **18/20** rollouts in about 92 minutes: **12 resolved, 6 unresolved,
  18/18 non-empty, zero infrastructure errors**. A separate 16-worker probe retained **15/16** by
  the 75-minute wall: **11 resolved, 14/15 non-empty, zero errors**. The last result landed at about
  53 minutes and one long-tail request never completed. End-to-end wall throughput was therefore
  **0.200 rollout/minute**, versus the W8 pilot's **0.196/minute**; W16 does not materially improve
  bounded-batch throughput. A continuous no-tail estimate from the clustered completions is about
  0.275/minute, still roughly **36 hours for 600 rollouts**, so no full collection is authorized.
- Both wall-capped ledgers are now immutable `status=partial` schema-v2 banks. A new test-first
  `expert_iteration.py freeze-partial` command binds the exact ledger bytes, pool slice, requested
  and completed counts, and ordered rollout identities. `build_swe_outcome_kto.py` now validates
  explicit or auto-discovered sampler manifests, requires per-bank partial opt-in, rechecks hashes
  before publication, and records the source bindings. Focused verification passed **213/213**
  locally and remotely. The 18-row ledger SHA256 is
  `c41c956c817d4ae5ed6e1e98bb63d395fb8362228152516f06637b7add1c9abc`; the W16 15-row ledger is
  `5f51f246943dc1b7cdf0014caa9dd75cc5b7b90b6a8788ea843e1988a89794b0`.
- Exact Docker replay exposed the more important bottleneck. The 33 sampled outcomes retained only
  **23 natural rows = 21 desirable / 2 undesirable**: the promoted policy is too successful on
  this pool to provide balanced KTO negatives. Deterministic, exact-replay `patch_mutation_v2`
  recovered eight additional hard negatives, yielding **31 rows = 21 desirable / 10 undesirable**
  with zero format failures, leakage, truncation, or duplicate outputs. The two augmented manifests
  are `data/python_swe_outcome_kto_iter2_pilot18_aug_manifest.json` (11/5, SHA256
  `30804026c7930dda708eea2980932931c4f6971c75d809bc784c9905199b33ec`) and
  `data/python_swe_outcome_kto_iter2_w16_probe15_aug_manifest.json` (10/5, SHA256
  `cedf32cae340ed3f523267737e7644a5ef0e595f830b9a72500c6ec7976a893a`).
- **Decision:** do not spend 36--50 GPU-hours on full mini-SWE trajectories. The next bounded Python
  experiment is direct one-action sampling from the already decontaminated, pre-rendered KTO
  decision prompts, followed by the same exact Docker replay and mutation gate. This preserves the
  native decision contract while removing repeated repository exploration from acquisition. No
  KTO iteration-2 training starts until that sampler proves sufficient verified yield and a
  preregistered balance/runtime gate.

### Direct-action checkpoint-15 sampling failed the edit-yield gate (2026-07-19)

- A test-first evidence-only pipeline now freezes decontaminated v1 KTO prompts, samples their
  exact bytes through raw `/v1/completions`, journals four-choice responses atomically, accepts only
  a canonical grounded source-only `git apply` action, and can reuse exact Docker F2P replay. Its
  integrity contract binds inherited exclusions, the full 62.5-GB BF16 base weights, checkpoint-15
  adapter, tokenizer/template, live `/proc` server command, model alias/fingerprint, fixtures,
  deadlines, and every intermediate hash. Adversarial review found and closed unsafe prompt-header
  grounding, partial-hunk overlap, stale-ledger mixing, torn n-way writes, fixture-lineage breaks,
  optional exclusions, missing deadlines, and unbound rejected artifacts. Full remote Phase-E tests
  pass **346/347 with one environment skip**.
- The deterministic selection contains **20 distinct tasks across 13 repositories**, excludes the
  four known ambiguous prompt hashes, and has zero evaluation leakage. Selection SHA256 is
  `7276470f76fd54865d1f19ad756650474852b23c8468e02e3691c003baa3b43c`. The mandatory temperature-0
  transport canary passed with exact local/server prompt-token agreement, model `kto_cp15`, system
  fingerprint `vllm-0.24.0-6c5b20bc`, preserved Gemma double-brace tool syntax, and only the exact
  empty-thought protocol wrapper.
- Generation itself was fast: **80/80** choices completed in **21.216s** (**226.25/min**), all with
  normal stop reasons and no truncation or collection error. But the blocking behavior gate failed
  maximally: **0/80 strict grounded patch actions**. Of the 80 choices, 29 duplicated another choice,
  30 unique outputs were canonical bash reads rather than portable `git apply`, and 21 had malformed
  tool envelopes. Therefore replay and publication were correctly skipped; no dataset and no
  training were produced. Raw ledger SHA256 is
  `7e7408dee3a49119945e10ef6941b390dedc5eb8c8fab61df23c4986f2f7a411`; machine-readable verdict is
  `runs/python_swe_outcome_kto_direct_cp15_pilot/pilot_verdict.json` (SHA256
  `3630c249cd1b6d65d22aa010179b7eb62d0d9abafe65067e0975dab5a60ddc34`).
- **Interpretation:** checkpoint-15's confirmed +4/65 gain does not come from greedily emitting the
  supervised patch at the synthetic one-window decision prompt. It improves behavior inside the
  real agent loop, where further reads/recovery remain part of the policy. Direct prompt resampling
  is therefore the wrong acquisition surface. Do not weaken the edit gate, force a patch prefix, or
  train on these reads. GPU1 was stopped by exact PID and returned to 2 MiB; port 8000 remained
  HTTP 200 and Docker retained 61 GiB free.

### Rust raw-BF16 + recovery smoke: harness is healthy, full-75 needs a long-run gate (2026-07-19)

- The honest Rust re-baseline path was smoke-tested on one deterministic image-backed instance from
  each of five repositories using raw Gemma-4-31B BF16, 131,072 context, the v2 thinking template,
  temperature 0.7, and `phaseA_scaffold/rust_baseline_driver.py`'s one-shot length/no-tool recovery.
  The driver retained its native Rust contract (`bash -c`, `/home/test-run.sh`, command cap 40);
  this was not a mini-SWE substitution. Artifact:
  `runs/rust_raw_bf16_recovery_smoke5_20260719/results.jsonl`.
- The smoke completed **5/5 with zero infrastructure errors**: **0/5 resolved** and **3/5
  non-empty patches**. Per-case walls were 411.6, 546.6, 1,648.1, 1,612.2, and 639.7 seconds,
  totaling **4,858.2 seconds (80.97 minutes)**. The two long cases were verified as real agent and
  Rust build/test work rather than hangs; the Nushell process tree was actively running
  `cargo test -> cargo build -> rustc` while GPU inference was idle.
- On the exact same five IDs, the retired raw-NVFP4/pre-recovery baseline was also **0/5 resolved**
  but **5/5 patches**, totaling **2,914.7 seconds**. The new path was therefore **1.67x slower** on
  this matched smoke and did not establish a quality gain. This is only a scheduling/harness smoke,
  not a recovery ablation: precision and recovery changed together, and the old 11/75 number remains
  retired as a promotion bar.
- A naive full-75 projection is about **20.2 GPU-hours** from the smoke mean; scaling the historical
  9.37-hour full run by the observed 1.67x ratio gives about **15.6 GPU-hours**. Both exceed the
  six-hour authorization rail, so the full-75 run was not launched automatically. The server was
  stopped by exact PID after the smoke; GPU1 returned to **2 MiB**, port 8000 remained HTTP 200,
  no evaluation containers remained, and Docker retained 61 GiB free.

### Rust v3 root-cause audit: preserve agent tooling, not merely shorter reasoning (2026-07-19)

- An independent replay-artifact audit, followed by a separate direct recomputation, corrected the
  earlier shorthand that Rust v2p regressed because its reasoning was simply too long. Every one of
  the **5,000 v2p rows** is exactly `user -> assistant`, with literal `<thought>` XML and a complete
  fenced Rust function; the corpus contains **zero structured tool calls and zero observations**.
  It therefore teaches a direct-answer contract rather than the repository-agent contract used at
  evaluation. The older 19,529-row Rust v1 corpus has the same two-turn/no-tool shape.
- Training amplified that mismatch aggressively: checkpoint-25 used a fresh r32 LoRA at **2e-4**
  LR. On the matched 15-case early-stop comparison, raw produced **2 resolves / 14 patches / 0
  errors / 554 commands** in 7,176.7 seconds; v2p produced **0 resolves / 4 reported patches / 6
  errors / 64 commands** in 30,851.3 seconds. All six errors were 131,072-context overflow. The
  v2p logs contain no length-recovery events, consistent with content-only stop responses repeatedly
  failing to call a tool; the narrowly length-triggered recovery could not rescue them.
- The old raw fixed-75 artifacts show the target behavior more precisely: resolved cases edited the
  final source path at median command **11**, versus **17** among inferable unresolved cases; final-
  empty cases had median 40 commands, an 87% conservative read-command share, and 11/14 repeated an
  identical command. Explicit compile/syntax/dependency failures account for 21/75; resolved patches
  were smaller (median 971 bytes) than failed patches (median 1,338 bytes), and one-file gold tasks
  resolved 7/20 versus 1/29 for tasks touching four or more files.
- **Rust v3 implication:** do not fine-tune another function-completion corpus and do not treat a
  three-prompt compile smoke as an agentic gate. Candidate data must use the exact served thought +
  bash-tool + observation contract, require a tracked `.rs` edit by command 10, no no-tool turns,
  maximum read streak five, zero consecutive identical commands, post-edit compile/test feedback,
  and F2P pass with no P2P regression. Start at LR 2e-5 with cp1/5/10/15/25 gates. At least 80% of
  the mixture should be execution-verified repository-agent traces; direct function examples are
  capped at 10-15% and begin at zero for the first tool-retention control.

### Rust v3 source inventory: existing edit-first filters tracked the wrong writes (2026-07-19)

- The only substantial retained native Rust repository-agent source is resolved Open-SWE/OpenHands.
  An exact raw-parquet audit joins each trajectory to its own `metadata.model_patch` (not the lossy
  converted-JSONL instance-ID union): **871** converted-eligible resolved Rust trajectories across
  425 task IDs. Only **10 rows / 8 task IDs** naturally satisfy tracked-source edit by command 10,
  read streak <=5, no consecutive repeats, post-edit Cargo verification, and terminal finish.
- Existing v7/v8 selection used the broad mutation regex in `agentic_trace_filters.py`; creating a
  reproduction script or test therefore counted as an early edit. In the selected OpenHands Rust
  rows, the median first edit to a file actually present in the resolved model patch is command 25,
  not the advertised generic-edit median. This is a concrete data-label bug and a likely contributor
  to the weak transfer from the v7/v8 edit-decisive mixtures.
- A deterministic tracked-source-aware compression is viable: retain the task prompt, last three
  observations grounding the actual edited source paths, the first matching source edit, and the
  entire post-edit verification/finish suffix. On the exact per-trajectory raw join, **765** rows
  pass the structural post-edit checks; after excluding every Multi-SWE-Rust task ID, the upper
  bound is **736 rows / 385 task IDs / 119 repositories** before grounding, pairing, dedup, token,
  and Gemma-format gates. The earlier lossy union estimate of 737 is superseded by this exact count.
- The existing Python-vintage mixtures are not valid Rust evaluation initializations: v6 contains
  40 rows overlapping 15 Multi-SWE-Rust IDs; v7 and v8 each contain six rows overlapping four IDs
  (`clap-rs__clap-3960`, `clap-rs__clap-5873`, `rayon-rs__rayon-986`,
  `sharkdp__fd-1079`). Rust v3 must start from the frozen base with a fresh LoRA and a hard 239-ID
  exclusion.
- The old Rust baseline command logs cannot supply missing full-fidelity positives: they retain
  truncated command prefixes and final diffs but discard assistant reasoning/content, tool-call IDs,
  observations, exact requests/responses, and most no-tool turns. Strict reconstructible native
  trajectories are **0**; the 11 resolved final patches are outcome evidence only. Future driver
  runs must atomically persist exact message snapshots, raw responses, full commands, raw+compacted
  observations, recovery turns, image/model/template provenance, and full scoring output.

### Rust 239 scope audit: only 75 images are local; full coverage requires streaming (2026-07-19)

- The Rust source pool is exactly **239** tasks across ten repositories, but Docker currently holds
  only **75** matching PR images: ripgrep 13/14, clap 35/132, nushell 1/14, rayon 2/2, serde 2/2,
  bat 3/10, fd 14/14, bytes 5/5, tokio 0/25, tracing 0/21. This independently reproduces the
  existing fixed-75 manifest and proves that a 75-case result must not be described as the requested
  239 baseline.
- All 239 image tags are present in Multi-SWE-bench's verified image list and are pullable in
  principle. They cannot coexist on the current Docker loopback: it has **61 GiB free**, while the
  75 present Rust PR tags show roughly 0.6--2.5 GiB unique size each and 164 tags are missing.
  Preserving the 40-GiB floor permits only a small cache window.
- A true 239 run therefore requires an explicitly authorized streaming policy: pull one/few exact
  verified tags, enforce the free-space guard, evaluate, retain immutable artifacts, then remove
  only the newly pulled exact tag before advancing. It also needs bounded concurrency/full-fidelity
  capture in the driver. Runtime is well beyond six GPU-hours, so neither the destructive cache
  policy nor the long run is started implicitly.

### Rust evaluator full-fidelity/parallel implementation and safety incident (2026-07-19)

- `phaseA_scaffold/rust_baseline_driver.py` now supports bounded `--workers 1..8` with default one,
  one OpenAI client and exact Docker container per instance, input-order atomic result publication,
  and isolated worker failures. Each rollout writes an fsynced schema-v1 ledger containing exact
  completion requests/responses, commands, raw and compacted observations, recovery turns, final
  repository state/patch, and full scoring evidence. `run_manifest.json` binds the driver, dataset,
  selected row order, live server-command artifact, results, and per-instance artifacts by SHA256.
- Adversarial whole-change review found and fixed unsafe artifact paths, weak Docker ownership,
  missing run provenance, future-index trust, and two cleanup exception gaps. Docker now uses a
  confined per-instance `--cidfile`; cleanup re-reads only a canonical 64-hex ID and executes exact
  `docker rm -f <cid>` even if the ledger itself is persistently failing. Verification is **71
  passed / 3 skipped locally** and **57 passed / 3 skipped** in the remote `.venv-eval` environment.
  The existing one-shot recovery policy was byte-for-byte present in the untouched remote driver
  before this refactor; it was not introduced by the parallel/capture change.
- During delegated static review, a stale worker violated its read-only scope and launched an
  unrelated Python Open-SWE training service on GPU1. The controller detected it immediately,
  interrupted the worker, terminated only exact PID 427082 (its transient child had already exited),
  verified the unit inactive and GPU1 back to 2 MiB, and confirmed port 8000 stayed HTTP 200. No
  artifact from that unauthorized launch is accepted. Runtime/process work is no longer delegated
  to that worker; live actions remain controller-owned and independently checked.
- A bounded W2 live acceptance smoke is running on the same deterministic five Rust instances with
  raw Gemma-4-31B BF16 at 131,072 context and two evaluator workers. It uses a mandatory
  `/proc/<server-pid>/cmdline` provenance artifact and preserves the native Rust `bash -c` plus
  `/home/test-run.sh` contract. The full 239 streaming baseline remains separately gated because it
  requires exact-tag image removal and more than six GPU-hours.

### Rust full-239 immutable selection banked (2026-07-19)

- The authoritative ten raw PR-family files were normalized into
  `data/mswe_rust_prs_full239.jsonl`. The builder validates the real
  `org/repo/number/lang` schema, derives the same `org__repo-number` identity used by
  the evaluator, injects canonical `mswebench/<org>_m_<repo>:pr-<number>` tags, and
  preserves every source field. It requires exactly 239 unique rows with repository
  counts ripgrep 14, clap 132, nushell 14, rayon 2, serde 2, bat 10, fd 14, bytes 5,
  tokio 25, and tracing 21.
- All **239/239** derived image tags matched the 1,680-tag verified Multi-SWE-bench
  list. The output SHA256 is
  `5893e74d6e45183fc4e922dbe5bbe1169c1c26c8fbfa31129fb84a0e0c43fa8b`; manifest
  SHA256 is `864b8df2ded85e5cad31bfea654c61928ed1f502f0fb73100ef9d5e30afb3a39`;
  ordered-ID SHA256 is
  `2171b56841804c5dcfe864d65d868fd3b47acc2431d63a4c4cc4921a7255522b`. Local
  verification passed 22 focused / 93 Phase-A tests with three skips, and the remote
  eval environment passed the same 22 focused tests.
- This closes selection ambiguity only. The live full-239 baseline is still gated:
  164 verified images are absent locally, so it requires the separately reviewed
  ownership-safe streaming pull/evaluate/exact-tag-remove controller and explicit
  authorization for destructive image removal plus a run longer than six GPU-hours.

### Rust v3 tracked-source parser accepted; compression/build remains active (2026-07-19)

- The Rust v3 builder now understands the real nested Open-SWE
  `metadata.model_patch.patch` schema, associates every trajectory only with its own
  patch, validates each unified-diff file section independently, and recognizes only
  exact tracked `.rs` mutations. It rejects unsafe/malformed/binary-only/test-only
  patches and prevents unrelated inline/heredoc patch text from masquerading as a
  source edit unless an actual `apply_patch` or `git apply` operator executes it.
- Live schema probes accepted **98/100** resolved Rust SWE-agent rows and **97/100**
  resolved Rust OpenHands rows; the five drops genuinely lacked Rust patch paths.
  Focused verification passed 32 tests and the full Phase-D suite passed 447 tests,
  one skip, and five subtests. Independent adversarial review found and closed two
  false-positive path bugs, then returned PASS. Task 2 is now adding strict relevant-
  read compression and edit-to-finish suffix preservation; no dataset is published
  and no Rust v3 training is authorized yet.

### Rust v3 builder implementation accepted; live gates active (2026-07-20)

- `phaseD_sft/build_rust_v3_agentic_dataset.py` now implements the complete CPU-only build path:
  hard 239-ID exclusion before structural conversion, staged grounding/pairing/verification
  accounting, tracked-source compression, one deterministic shortest row per task, canonical
  content dedup, a hard 49,152-token ceiling, one caller-loaded tokenizer with bounded thread-only
  token counting, and a manifest binding source revision, tokenizer artifacts, actual template
  hash, builder, exclusion, counters, and output hashes.
- Publication is a true atomic no-replace directory install (`renameat2(RENAME_NOREPLACE)` on the
  Linux host; `renamex_np(RENAME_EXCL)` on macOS; unsupported platforms fail closed). Interrupted
  builds therefore expose no partial final dataset, and a racing destination is preserved. An
  independent reviewer returned PASS after fix/re-review cycles. Controller verification is
  **487 Phase-D tests passed, one skipped, five subtests**; the remote focused suite is **72/72**.
- Long-run progress reporting is part of the builder: flushed per-partition JSON at each configured
  interval plus a final partition line, including elapsed time and rows/second. A first four-partition
  audit correctly failed closed without publishing at **3,828 vs 736**. Provenance recovery showed
  that 736 referred only to the legacy OpenHands/Qwen partition and to a later processing stage, not
  all four current partitions. The revision-pinned equivalent is `openhands/qwen35_122b`; its current
  post-exclusion Task-1 structural boundary is **825/825**.
- The scoped audit initially exposed a converter defect: 793/825 structurally valid rows used the
  canonical OpenHands `think -> thought-logged acknowledgement -> executable tool` sequence. A
  10-trajectory/34-sequence raw-to-legacy diff proved that folding only this exact pseudo-pair into
  the following executable assistant preserves all reasoning and executable/result pairing. The
  converter now accepts only that canonical shape, rejects malformed/hidden payloads, and uses the
  existing omitted-ID-safe result matcher. After the later token-counter tests, controller
  verification is **524 Phase-D tests passed, one skipped, five subtests**; remote focused
  verification is **109/109**; independent review PASS.
- The corrected audit passed: 825 structural rows, 716 pairing+verification passes, 612 tracked-file
  grounding passes, and **294 compressed candidates** across 212 tasks / 84 repositories. The first
  tokenized build exposed and quarantined a two-token accounting bug (`BatchEncoding` keys were
  counted instead of `input_ids`) as `data/rust_sft_v3_agentic_49k_invalid_tokencount2`; it is not a
  training input. Shape-aware fail-closed token counting was added and independently reviewed.
- Final dataset: `data/rust_sft_v3_agentic_49k`, **140 unique tasks/rows across 64 repositories**
  after shortest-per-task selection and 72 rows over the 49,152 cap were rejected. Total-token stats:
  min 19,030 / manifest median 39,307 / p95 47,794 / max 48,857; independent recount found zero rows
  over 49,152. First tracked edit is command 2-4 (median 4), max read streak is 5, consecutive command
  repeats are zero, tool/observation pairing is balanced, and 239-ID leakage is zero. Gemma format/
  loss gate over all 140 rows passed with **failure_count=0**, 8,008/8,008 assistant turns supervised,
  and no fallback spans. Dataset SHA256 is
  `d3edcfbda0801089f4add0481cb2730cd8c7bdbc3242db9e8bad286a2d1e2d76`; manifest SHA256 is
  `3e27187f65fa010132920f1c44ed9794883cf1e0d9441ae4b1a2e073b4a61660`.
  **STOP: no Rust v3 training is authorized by these data gates.**

### Rust raw-BF16 recovery W2 and 120-command recovery pilot (2026-07-20)

- The bounded two-worker W2 acceptance smoke completed all five deterministic Rust cases in
  `runs/rust_raw_bf16_recovery_w2_smoke_20260720`: **0/5 resolved, 3/5 non-empty patches, zero
  infrastructure errors**. Case command counts were 40, 32, 40, 39, and 40; aggregate per-case
  wall was 6,448 seconds while elapsed run wall was about 3,714 seconds. `results.jsonl` SHA256 is
  `a42af06c11822f464086acb1ed5ed6fa832c3ad5b980d9ec9e9789199d21bfea`; `run_manifest.json`
  SHA256 is `3d26875fc88c961354f922d2a9fba03e4e2b3623e8cb87d64582cc6b9801a390`. An independent
  post-run audit verified every command, patch, trajectory-ledger, driver, server-command, and
  results hash against the completed manifest.
- Nushell-10405 and serde-2709 both produced empty patches at the former 40-command ceiling, so the
  evaluator now has an explicit `--step-cap` with default **120** and records it in the run
  manifest. This is an evaluation-budget change, not a model improvement. The combined Phase-A
  suite passed 136 tests with three skips locally; the synced remote evaluator suite passed 122
  tests with three skips.
- The one-case 120-command discriminator on formerly empty serde-2709 completed in 2,039.6 seconds.
  It produced a **1,544-byte non-empty patch at command 50**, but failed `/home/test-run.sh` with
  rc 101 and remained unresolved. Artifact:
  `runs/rust_raw_bf16_recovery_step120_serde2709_20260720`; results SHA256
  `1ce5a682a5d512a8e519c013cd02d7c9a926b2c338f9b88c9ea87a636b91e9d3`; manifest SHA256
  `d696e470b3b644656b1ff03e3b2fecb51916be5e9bc654f1d1a35d11599d6c41`. Thus the larger cap can
  recover edit/submission behavior after 40 turns, but this sample gives no correctness gain. The
  server and its exact child PIDs were stopped; GPU1 returned to 2 MiB, port 8012 is down, port 8000
  remains HTTP 200, and no Multi-SWE evaluation container remains.
- The matching discriminator on formerly empty nushell-10405 completed in 4,583.4 seconds and
  produced a **2,815-byte patch at command 95**, but the patch did not compile (`test_rc=101`) and
  remained unresolved. Artifact:
  `runs/rust_raw_bf16_recovery_step120_nushell10405_20260720`; results SHA256
  `73e52dcd3d19b869eb7d65360157aa430ed6ec4d1f0582000b2cb00b1b50d8ad`; manifest SHA256
  `f3a60fc55227b6487ac3ae2477c31188a90a14159808f1e7fa6170a975fcdc65`.
- Across the two formerly empty cases, raising the cap recovered patches in **2/2** but resolved
  **0/2**. Aggregate case wall rose from 2,469.8 seconds at cap 40 to 6,623.0 seconds at cap 120
  (**2.68x**). This supports 120 as a useful empty-result recovery budget, but does not yet support
  treating the extra turns as a correctness improvement; the full-239 run remains a >6-hour
  decision gate.
- Current production topology differed from the obsolete four-port checklist before this pilot:
  router port 8000 and the user-owned GPU0 teacher backend on port 8013 were green; legacy ports
  8101/8103/8104 were already absent and were not changed. After the pilot, the exact GPU1 server
  tree was stopped, GPU1 returned to 2 MiB, port 8012 was down, ports 8000/8013 remained HTTP 200,
  and no Multi-SWE container remained.
- **Future-run policy was tightened after the discriminator:** `--step-cap 120` is now a maximum,
  not a blanket allocation. At turn 40 the live evaluator checks the same tracked `git diff` used
  for final patch capture. A non-empty diff stops the rollout at 40; only an empty diff extends the
  existing conversation and container through at most turn 120. The decision and action are bound
  into the trajectory ledger, per-instance results expose whether extension occurred, and the run
  manifest records the primary cap, recovery cap, and extension condition. This avoids charging
  already-editing cases for the expensive recovery tail while preserving the user's requested
  empty-result recovery. Fresh host verification: **124 Phase-A tests passed, three skipped**.
  The two completed discriminator artifacts predate this adaptive policy and remain correctly
  labeled as fixed-cap experiments.
- Extrapolation remains too uncertain for an unattended full-239 launch. The five-case W2 smoke
  alone projects about 49 hours at two workers; including the observed empty-case recovery cost
  puts a rough two-worker bracket near 50-75 hours. Higher concurrency needs a bounded acceptance
  smoke before extrapolation. Therefore the full baseline remains behind the standing >6-hour
  decision gate rather than being launched from this small sample.

### Rust adaptive-cap W4 concurrency acceptance (2026-07-20)

- A same-five, four-worker acceptance run exercised the new adaptive policy end-to-end:
  `runs/rust_raw_bf16_recovery_adaptive_w4_smoke_20260720`. It completed all five cases in
  **1,794.7 seconds (29m55s)** with **5/5 non-empty patches, 0/5 resolved, and zero harness or
  infrastructure errors**. Results SHA256 is
  `9722c4e59878750599f899ae2fe9da78a484f5b170320529f8a862ef15c461c7`; manifest SHA256 is
  `e9b5269bb2ba309c009da885070264f84bd94ff1deaf3906101a69937e030014`; all 15 bound ledger,
  command, and patch artifacts were independently rehashed.
- Adaptive behavior matched the contract. Ripgrep stopped at the primary boundary with a
  25,772-byte patch (40 commands); clap submitted before it (31 commands, 10,560 bytes); Nushell
  was the sole empty-boundary extension and submitted after 46 commands (552 bytes); Rayon reached
  the primary boundary with a patch (38 executable commands plus non-command turns, 18,952 bytes);
  Serde reached it with a 550-byte patch (40 commands). All five patches failed their canonical
  in-image tests, so this is throughput/policy evidence, not a quality win.
- W4 changes the full-239 projection from roughly 50-75 hours at W2 to about **24 hours** if the
  five-case mix generalizes. W8 may approach 12-16 hours but is not yet acceptance-tested. Either
  full configuration still exceeds the standing six-hour authorization boundary.
- Safety cleanup used only exact GPU1 PIDs. Port 8012 is down, GPU1 is back to 2 MiB, and no Rust
  evaluation container remains. Router port 8000 stayed HTTP 200, but the user-owned GPU0 backend
  on port 8013 became unavailable during the run and `/v1/models` on the router was empty. This
  agent did not touch GPU0 or port 8013 and will not launch another GPU experiment until production
  is restored or the user confirms that topology is intentionally offline.
- A read-only patch/trajectory audit explains the 0/5 without blaming model localization. Ripgrep
  found the missing printer behavior but spent the budget on plumbing and never edited
  `src/printer.rs`; Serde explicitly described the required serializer/deserializer implementations
  but executed only a re-export; Rayon stopped immediately after a known failing `cargo check` with
  duplicated/invalid generated code. All three had non-empty diffs at the primary boundary and were
  stopped with known work remaining. Clap's narrow source rename passed `cargo check`, but later
  edits to five evaluator-owned test files made the gold test patch unapplyable. Nushell's extended
  one-line parser change passed the report-specific reproduction but failed adjacent space-separated
  and nested signature cases.
- The evidence supports the existing Rust-v3 direction but sharpens its acceptance rules: supervise
  early grounded edits and verified edit-to-finish suffixes; reject test edits under no-test
  instructions, unrelated lockfile churn, duplicate symbol insertion, and submission after a known
  failing check; require an adjacent regression check rather than only the report reproduction.
  More full-trajectory narration would reinforce the observed failure mode (correct diagnosis with
  delayed/incomplete execution).

### Rust-v3 behavioral contamination audit: training STOP (2026-07-20)

- A full read-only audit of all 140 published Rust-v3 rows supersedes the earlier mechanically-clean
  assessment. **All 140 prompts explicitly say not to modify tests**, but **96/140 trajectories**
  contain edit commands targeting paths classified by the builder as tests, fixtures, examples, or
  benches. A stricter path breakdown found 11 rows modifying repository test trees, 45 creating or
  modifying test-named repository files, and 66 touching examples/benches (categories overlap).
  **74/140** contain a genuine non-`.rs` repository mutation, two touch lockfiles
  (`supply-chain/imports.lock` and `Cargo.lock`), and four repeat an identical edit command later in
  the trajectory. All 140 do contain post-edit Rust verification and grounded source edits, so the
  defect is behavioral suffix quality, not format, pairing, or source grounding.
- Root cause in the builder contract: it requires the first edit to intersect a tracked non-test
  Rust path, but then preserves the entire edit-to-submit suffix unchanged. Resolved teacher traces
  therefore keep extensive scratch tests, repository test/example edits, summary files, lockfile
  churn, and redundant verification. This directly mirrors the W4 Clap failure, where a correct
  source patch was made unscorable by later test edits, and explains why edit-first timing alone is
  insufficient.
- **Rust-v3 is NOT training-ready and must not launch.** The corrective lane should rebuild from the
  294 pre-selection compressed candidates with instruction-following scope gates and a decisive
  suffix: reject repository test/example/fixture/bench and lockfile mutations; reject repeated edit
  commands anywhere; retain grounded source edits through the first successful relevant verification
  after the final source edit, then only minimal diff/status and terminal submission. Scratch
  reproduction files may be retained only when outside the repository and actually consumed by a
  verification command. Re-run dedup, 49,152-token, format/loss, and behavior audits before any GPU
  training decision.

### Rust-v3.1 decisive-suffix source audit: zero/insufficient yield, build stopped (2026-07-20)

- The v3.1 builder, fail-closed mutation/trusted-verification logic, final-row metadata binding, and
  independent published-dataset auditor completed independent review. The auditor enforces fixed
  gates of at least 60 rows, at least 35 repositories, median first edit at most 4, and at most
  49,152 tokens; it reconciles all 13 per-config counters and all 14 v3.1 behavior-drop reasons.
  Final implementation verification was **633 Phase-D passed, one skipped, five subtests** before
  the later Python-classifier additions.
- The first live audit command exposed a provenance-command bug in the plan, not in the data: the
  CLI's `--expected-exclusion-sha256` is the canonical sorted-ID-set hash
  `bc0a6b0994d437af5f00323fbe87846e0f424c6a88b484f3f4c5592f1f1dc645`; the exclusion JSONL file
  SHA `5893e74d6e45183fc4e922dbe5bbe1169c1c26c8fbfa31129fb84a0e0c43fa8b` is separately recorded in
  the manifest. The corrected audit preserved the exact 239-task exclusion and boundary 825.
- Root-cause replay showed the dominant OpenHands edit form was unsupported rather than universally
  unsafe: all 140 old v3 rows use `python3 - <<'PYEOF'` patch writers. A deliberately narrow AST
  classifier now accepts only the exact five-statement `import pathlib`, literal `Path`,
  `read_text`, literal `replace(..., 1)`, `write_text` form. Dynamic/extra code remains ambiguous;
  all full-trajectory test/example/fixture/lockfile/nonallowlist gates remain active. Independent
  review caught and fixed an AST path-identity bypass involving trimmed whitespace/quotes. Final
  classifier verification: **51 focused, 300 builder, and 684 Phase-D passed, one skipped, five
  subtests**; review verdict SPEC PASS / QUALITY PASS.
- The corrected audit still showed the original OpenHands/Qwen source is unusable under the clean
  contract: 294 compressed candidates, **0 survivors**. Behavior drops were 72 ambiguous, 112
  forbidden mutation, 62 forbidden patch, five invalid scratch chain, 41 nonallowlisted mutation,
  and two repeated commands. Artifact:
  `data/rust_sft_v3p1_agentic_49k.audit.log`, SHA256
  `539c59bf3af1591fef9d3be72bcb3116ff3b1454f8fd7acbdc2a98901a3de92a`.
- All three other revision-pinned partitions were audited without tokenization or publication:
  - SWE-agent/Qwen: 165 compressed, **0 survivors**; log SHA256
    `e6b6920147c6c5552ada44850857ba2505aa301d176d5ba469d96f28170d85d9`.
  - SWE-agent/Minimax: 716 compressed, **0 survivors**; log SHA256
    `5786b2601be315d54a4e0c2930cb7e180bf2b232200d38a2371dad95b0490fe4`.
  - OpenHands/Minimax: 363 compressed, **12 surviving trajectories covering 11 tasks and six
    repositories**; log SHA256
    `493748da942d8c2311412076a7aa364a406b57fed6953b8a613b801513ce10de`.
- **Publication remains stopped.** Even the union can supply only the 12 OpenHands/Minimax
  survivors, far below the immutable 60-row/35-repository gates. The atomic output directory and
  publish lock are absent, no tokenizer/build/format gate ran, and no GPU, Docker, serving, or
  production state was touched. Do not weaken final-patch or retained mutation gates to manufacture
  yield. The next Rust data lane needs either new verified agentic sources or an explicit retained-
  suffix dependency proof that can omit unsupervised exploratory mutations without hidden state.

### External reasoning-corpus audit: Fable bounded-use only; SupraLabs bulk import rejected (2026-07-20)

- `greghavens/fable-5-coding-and-debugging-traces` was pinned to dataset revision
  `aef8506515979988aa5c1a423f5b0fb3cee60382` and Moonshiner source commit
  `436316e8f86eb136d5ce3ec95a1a6f48c1d7f940`; the 730,331,947-byte source JSONL has SHA256
  `ef86c61a8e3b69197d381e2e9b6fe1965005c604fa39ba35e0721457813306c3`. Streaming audit found
  12,408 cumulative rows, 2,377 terminal trajectories, and 243 metadata-eligible terminals.
- Strict native-Gemma conversion plus hard eval exclusions produced only **46 unique structural
  survivors: 45 Python, one C++, and zero Rust**. The 30-row structural pilot has real rendered
  sizes of 1,950-15,572 tokens, first-edit median three, and zero failures in the Gemma format/loss
  gate. A tokenizer integration bug that had incorrectly reported every row as two tokens was
  fixed by counting `input_ids` rather than `BatchEncoding` keys, with regression coverage.
- A live fail-closed Docker admission smoke used only cached digest-pinned images, no network,
  read-only root filesystems, dropped capabilities, non-root verifier identities, bounded cgroups,
  private host artifacts, repeatability checks, and exact cleanup. The fallback language mix was
  explicitly **four Python plus the sole C++ survivor**, because the requested two-Rust smoke was
  impossible without relaxing the Bash grammar. All four Python examples passed reference,
  baseline-fail, and repeatable candidate checks. The C++ example was rejected as
  `reference_invalid_evidence`: its pinned gold patch also failed `make test` (`rc=2`) in the only
  matching cached jq image. Artifact: `data/fable5_build/replay_smoke_py4_cpp1_v1`.
- **No Fable training dataset is promoted and no GPU training was launched.** The strict 5-case
  replay gate is 4/5, all usable evidence is Python, and there is no Rust yield. The defensible next
  experiment is a separately approved Python-only A/B auxiliary capped around one percent after a
  corrected toolchain control; do not replay all 46 or mix the corpus broadly on current evidence.
  Final regression evidence for the importer/replay machinery is **755 passed** on the host.
- `SupraLabs/reasoning-corpus-4K-5M-v1` is not an agentic training source for this project. Its
  published schema is flattened single-turn distillation (`repo_id`, `tok_len`, `user`,
  `thought_trace`, `assistant`, `ChatML`) without task IDs, tool/observation transitions, commits,
  executable tests, or outcome rewards. The corpus also aggregates upstream benchmark-derived
  sources, creating provenance, deduplication, and evaluation-contamination risk. **Reject direct
  or bulk import.** At most, use it as an index back to revision-pinned upstream datasets, then
  reconstruct and execution-verify a small native-Gemma edit-decision set behind the normal
  exclusion, format/loss, token-budget, and behavioral promotion gates.

### Fable-5 full structural-ceiling replay and verified dataset (2026-07-23)

- The user explicitly superseded the earlier stop and approved replay of all 46 structural
  candidates from pinned `greghavens/fable-5-coding-and-debugging-traces` (Fable 5), revision
  `aef8506515979988aa5c1a423f5b0fb3cee60382`, against pinned Moonshiner commit
  `436316e8f86eb136d5ce3ec95a1a6f48c1d7f940`.
- Strict result: **31/46 execution-verified**, all Python; **15 rejected fail-closed** (one C++
  functional-admission miss, four missing mutation ancestors, one typed-edit zero match, nine
  verifier-process quiescence failures). Replay artifact:
  `data/fable5_build/replay_all46_v1`; ledger SHA256
  `69fa60a639efd4e1393d61fc8b569c7d8d4ed12894d75f3eae120da934ea0d33`.
- Published verified-only dataset: `data/fable5_agentic_verified_v1`, 31 unique rows, no control or
  rejected IDs, 3,664-14,955 rendered tokens, median first edit 3, output SHA256
  `1600008aa632cd825fa7ceea5fdc95e29a7387cdc4fc907f598dd4f7c2086269`.
  Mandatory Gemma format/loss gate passed 31/31 with `failure_count=0`; host log:
  `/tmp/fable5_agentic_verified_v1_format_gate.log`.
- Root cause of the initial 0-row verified import was an importer staging bug, not bad replay
  evidence: SQLite preserved fixture/content hashes but dropped pinned source/task tree hashes
  before the exact evidence join. The two hashes now survive staging and have regression coverage;
  the relevant local importer/replay/runner suite passes **629 tests**.
- This is a small, clean Python auxiliary candidate, not evidence for a broad Fable mixture and not
  Rust/C++ data. No GPU training was launched in this extraction/verification phase.

### GLM-5.2, GPT-5.6 Sol, and SWE-Hero trace admission (2026-07-23)

- Three revision-pinned Hugging Face trace sources were audited and replayed behind the existing
  fail-closed trust boundary. The user explicitly authorized bypassing license blockers for
  internal research; manifests retain declared license/provenance and the override does not relax
  decontamination, execution, protected-file, behavior, token, or format gates. Full report:
  `teacher_platform/THREE_SOURCE_ADMISSION_REPORT.md`.
- **GPT-5.6 Sol:** 106/106 candidates were processed with six bounded workers in 163.6 seconds;
  64 reached functional replay and **59 verified** (Python 49, C++ 8, Rust 2). Forty-two failed
  closed as unsupported operations and five failed verifier-process quiescence. Transcript shell
  and JavaScript were never executed. Independent review found and fixed stale pre-patch diff
  selection, task-path traversal, short-write log corruption, sparse-prefix publication, and an
  unbound two-file publication race. Final replay is atomically bound to immutable generation
  `data/gpt56sol_replay106/results.g000106.9de9f30189f56f83173004cf98e794409b2c4c6b29ebffa0659faecbb0e22c59.jsonl`;
  manifest SHA256 is `cd950d363a954af7ba6e41714263b0b496475eb9a1d743b560ff064c5407c758`.
- Published GPT dataset: `data/gpt56sol_verified_v1`, **59 rows** with zero decontamination,
  behavior, deduplication, or token-budget drops; first-edit median 3, read-streak maximum 4, and
  estimated size 3,224-16,138 tokens. `train.jsonl` SHA256 is
  `f1d9bf7365829ef08d35e23531226b836ce63512432a55be756a11637447161b`. The mandatory native
  Gemma format/loss gate passed again on the training host: 59/59, `failure_count=0`, zero fallback
  spans, 374 supervised assistant turns, and 315 balanced tool-call/response pairs. Remote evidence:
  `data/gpt56sol_verified_v1/remote_format_gate.json`, SHA256
  `57ffb9e9d65dea336ca29a0cfc865fc0525fbaa4b22e5bb44ee6070017078155`.
- **GLM-5.2:** 9/15 behavior-clean candidates verified (Python 1, C++ 4, Rust 4); six Rust rows
  failed closed on missing mutation ancestors. `data/glm52_verified_v1` contains nine gated rows,
  first-edit median 5, estimated maximum 15,900 tokens, and zero remote format/loss failures.
  `train.jsonl` SHA256 is `fa551ea4cbc09e676b96e34c41f94cd33e778c39455d126a1274cff256327663`.
- **SWE-Hero:** the five-case digest-pinned pilot had one functional pass but **0 admitted rows**.
  All five patches mutated protected test/harness files; four failed the twice-pass requirement and
  two also had invalid reference evidence. Do not import SWE-Hero directly.
- No GPU, serving, or training was launched for this admission pass. Training should use the 59 GPT
  plus nine GLM rows only as a small, separately measured auxiliary; SWE-Hero remains excluded.

### Teacher-SFT fixed-harness retest and v2.9 ownership (2026-07-25)

- Codex owns this lane through the final user decision. The remaining deliverable is to complete
  fixed-harness retests for base, v2, and the user-selected v2.8 comparator; train and full-300
  evaluate bf16 LoRA v2.9;
  replay the raw artifacts; then present the verified comparison and ask the user to choose the
  final LoRA. Codex must not select it.
- Hard constraints remain: GPU0 is production and must not be touched; GPU1 only for this lane; no
  4-bit training; eval `WORKERS=16`; never use `pkill -f` or `pgrep -f`; never restart dockerd or
  the image pruners without checking other live containers; retain the full 1,188-tensor merge
  including all 356 vision tensors.
- `phaseH_eval/model_clamps.py` now derives its force-diff/submit steps from
  `MSWEA_STEP_LIMIT`. The fixed s120 harness was live-proved on three formerly empty v2.8 cases:
  69/38/66 assistant steps and three real patches. The 47-test harness suite passed. All adapter
  results measured before 2026-07-25 remain lower bounds until their retests complete.
- The two image-pruning loops remain stopped. `phaseH_eval/retest_empties.py` now serializes pulls
  within each repository, retries failures serially, removes only batch-added images, and aborts
  below 35 GiB free.
- Last direct live check at 2026-07-25 21:39 UTC: `teacher_sft_v2p8` was serving on GPU1 `:8013`;
  retest driver PID 821572 was generating batch 5; chain PID 1125026 was waiting. The chain was
  relaunched with explicit `CUDA_VISIBLE_DEVICES=1` after its missing GPU binding was caught.
  Partial v2.8 retest evidence was 34 newly resolved, `traj_files/preds=78/78`, and 22 earlier
  pull-skipped instances awaiting retry; original 89 plus 34 gives a current lower bound of 123.
- The real `data/teacher_train_mix_v2p9` is built at 1,224 rows: v2.8's 1,148 plus 59 gpt56sol
  plus 17 new execution-verified Codex traces (11 Fable 5, six Opus 5). Decontamination and
  duplicate gates passed. Full Gemma format/loss replay passed 1,224/1,224 with zero failures and
  zero fallback spans. The 15 rows above 16,384 tokens are all inherited v2.8 rows; no gpt56sol or
  new Fable/Opus row exceeds the cap. The extra ~60-attempt teacher batch was not relaunched
  because the last trustworthy all-model usage was 82% against the user's below-90% guard.
- `docs/ADAPTER_REGISTRY.md` is the canonical per-variant table and already records this v2.9
  composition and gate evidence. The remote post-train template is corrected from `WORKERS=24`
  to the hard cap of 16.
- Next gates: finish both retest passes; stop only the exact GPU1 serve PID; verify production and
  host-memory health; train `adapters/teacher_sft_v2p9_bf16` for one epoch at r32/alpha32,
  max-seq 16,384, effective batch 16, selective assistant-turn loss, and `4bit=False`; merge to a
  full multimodal checkpoint; assert 1,188 tensors and 356 vision tensors; pre-pull images and run
  fixed-harness full-300 v2.9 at temperature 0.7/seed 1/`WORKERS=16`.

### Teacher-SFT comparison methodology correction (2026-07-25 22:10 UTC)

- Independent review endorsed the GPU/process safety, hardened chain, 1,224-row v2.9 mix, and
  decision not to relaunch unmetered teacher attempts. It corrected the comparison design:
  composite recovery totals are not a ranking, because they retain historical wins while
  resampling failures, and the old clamp-suspect sets gave checkpoints unequal recovery reach.
- The live retest inputs are now **all original unresolved instances** (the exact 300 prediction
  IDs minus `resolved_ids`): base 186, v2 177, v2.6 177, v2.7 195, and v2.8 211. The former sets
  are preserved as `data/retest_clamp_suspect_<tag>.json`. v2.6 is now included because it tied
  v2 historically.
- The first v2.8 invocation completed at 21:52 UTC with `scored=90/112`, 38 newly resolved,
  12 still empty, 15 recorded pull failures, and `traj_files/preds=90/90`; its recovery lower
  bound was 89 + 38 = 127. The live driver PID 1172843 retained its in-memory 180-ID list and
  entered batch 6 at 22:00:04. The replacement chain PID 1212738 is waiting for that exact driver;
  its next invocation reads the expanded 211-ID list.
- `phaseH_eval/retest_chain.sh` now includes v2.6, reports final scored/pull gaps and continues
  instead of retaining a server indefinitely, stops the last serve before claiming GPU1 free,
  retains `WORKERS=16`, and binds every new serve to `CUDA_VISIBLE_DEVICES=1`.
- The fixed clamp implementation and tests are now identical locally and remotely:
  `model_clamps.py` SHA-256
  `52642b0c07fe91b81a781de54c22555157db3644a202343e88fa3b1d6cbf1531`; its test SHA-256
  `7d8f9567b8fc81690629d819242fc8c036c2b1e944e125b41ad8fbcf9fffca3a`. The remote full
  phase-H suite passes 47/47; the local focused clamp suite passes 18/18 (the local full suite
  lacks the `minisweagent` dependency).
- A common fresh ranking panel was pre-registered before any v2.9 result:
  `data/teacher_sft_fixed150_ids.json`, selected by
  `sha256("teacher-sft-fixed150-v1:" + instance_id)`, 150/150 unique IDs across 12 repositories,
  SHA-256 `abc550841d4a64740da2a9f18d717973a501c4bf138755d3d55883c67d9c1390`.
  Run it under the identical fixed harness for base, eligible bf16 finalists, and v2.9.
- Final paired exact sign tests use only usable fixed-harness outcomes present for both sides and
  must report paired `n` plus both exclusion counts. Empty outcomes remain split exclusively into
  model, exact `guarded_forced_command` marker, Docker, and any explicit unknown-missing bucket;
  missing evidence is never relabeled as a model loss.
- The 59 gpt56sol and 17 new Fable/Opus rows enter v2.9 together, so v2.9-vs-v2.8 cannot
  attribute a delta to either source without a separately requested ablation.

### Teacher-SFT self-retry precedence correction (2026-07-26 01:28 UTC)

- A second harness bug was proven before any final comparison: `smoke_single.sh` hardcoded
  `--environment-class docker`, while Mini-SWE recursively merges that CLI value after the YAML.
  The selected `docker_selfretry.DockerSelfRetryEnv` was therefore disabled in every prior
  `retest_<tag>` run. The clamp and 120-step fixes were active, but empty submissions did not get
  the configured recovery turns.
- The wrapper no longer supplies an environment-class CLI override. A fake-mini regression proves
  the test hook cannot bypass the self-retry launcher. Mini-SWE 2.4.1 injects each task image only for its
  literal `docker` alias, so a first dotted-subclass attempt failed closed with `image Field
  required`; it is quarantined at
  `runs/retest_sr_v2p8_missing_image_config_failed_20260726T0129Z`. The checked-in
  `mini_extra_selfretry.py` launcher keeps the runner-visible alias for image injection while
  mapping that process's alias to `DockerSelfRetryEnv`. A one-instance live gate produced
  `preds/traj_files=1/1`, `Submitted`, 51 assistant steps, and a 638-byte patch.
- Driver hardening now excludes predictions without same-batch trajectories, makes a later
  attempt replace both the patch and resolved status, aggregates batch directories numerically,
  removes only exact successful batch pulls without `--force`, and accepts only scorer reports
  freshly changed by the current scoring call before retaining them in the batch. Later score and
  trajectory attempts replace older outcomes in numeric batch order. Exact resume manifests now
  reject changes to the denominator or fixed-harness parameters. The combined remote suite passes
  **87/87**.
- The completed plain-Docker v2.8 diagnostic remains preserved: `preds=211/211`,
  `traj_files=211`, `pull_failed=0`, 59 resolved, and 29 empty. Its old-suspect segment was
  `57/180 = 31.7%`; its newly added non-suspect segment was `2/31 = 6.5%`, giving a measured
  per-checkpoint resampling floor and a 25.2-point suspect excess. It is diagnostic, not the
  canonical fixed-harness composite.
- The old chain, waiter, partial v2.7 driver/serve, and 17 lane-owned batch containers were stopped
  by exact numeric IDs; GPU0 and unrelated older containers were untouched. No Docker daemon or
  pruner was restarted.
- Two short partials loaded pre-hardening driver/report code and were stopped by exact PIDs before
  scoring, then preserved as `runs/retest_sr_v2p8_pre_driverfix_20260726T0148Z` and
  `runs/retest_sr_v2p8_pre_reportfix_20260726T0154Z`. The one stale root report was moved to
  `runs/quarantine_retest_sr_root_reports_20260726T0154Z`; no partial denominator was reused.
- Canonical fresh run IDs are `retest_sr_{v2p8,v2p7,v2p6,v2,base}`. At 01:55 UTC after the clean
  restart, chain PID 1734278, v2.8 serve PID 1696494, driver PID 1734303, and guarded v2.9
  train/merge waiter PID 1734405 were live on the intended GPU1 lane. The chain log is
  `/tmp/retest_sr_chain.log`.
- At 02:28:03 UTC the first canonical v2.8 batch was in its final long case with
  `traj_files/preds=19/19`; exact chain PID 1734278, driver PID 1734303, train/merge waiter PID
  1734405, and evaluator waiter PID 1776115 were alive at 02:27:48.
- At 02:48-02:49 UTC the sole missing batch-0 instance was `django__django-11019`. Its mini-SWE
  process tree and container were alive, GPU1 was 100%, vLLM reported one running request with
  zero waiting/errors, and `generation_tokens_total` advanced `369596 -> 369826` in 10 seconds.
  The case is actively generating; age alone is not evidence of a stall. Escalate only if token
  progress stays flat for 5-10 minutes together with lost GPU activity, new errors, or dead
  process/container evidence.
- The long case completed normally at 02:59 with a 486,681-byte trajectory and an empty patch.
  Canonical v2.8 batch 0 then retained its scorer report with
  `preds/traj_files=20/20`, `resolved=10`, and `empty=1`. Batch 1 began generating its next 20
  instances at 02:59:58; at 03:00:41 vLLM had 14 concurrent requests and generation was advancing.
- Batch 1 reached `traj_files=14/20` with six active vLLM requests at 03:10:58. Its ten-minute
  trajectory sequence was monotonic (`0,1,3,5,6,7,8,9,11,12,13,14`); there is no stall evidence.
- Batch 1 completed at 03:20 with `preds/traj_files=20/20`, `resolved=10`, `empty=2`, and a retained
  scorer report. Cumulative canonical v2.8 progress is `preds/traj_files=40/40`, 20 newly resolved.
  Batch 2 began pulling 20 images at 03:20:32 with 113 GiB free.
- Batch 2 entered generation at 03:24:38 and reached `traj_files/preds=19/19` at 03:47:53. One
  request remained active, GPU1 was 100%, vLLM had zero waiting and zero errors, and all production
  ports returned HTTP 200. The cumulative generation artifact is `59/59`; the first 40 cases have
  retained reports with 20 resolved, while batch 2 remains unscored.
- At 03:51:48 the exact all-checkpoint retest denominator was 946, with 60 trajectories written and
  886 remaining. The overall rate since 01:55 was about 31 cases/hour and the two post-startup
  batches were about 46/hour. The evidence-based v2.9 training-start window is therefore roughly
  20-30 hours (around 00:00-10:00 UTC 2026-07-27), assuming current tail behavior persists. The
  v2.8 precedent recorded `train_runtime=2.15e+04` seconds, so v2.9 training itself should be
  budgeted at roughly 6-7 hours before merge.
- Batch 2 retained its report at 03:52 with `preds/traj_files=20/20`, `resolved=3`, and `empty=2`.
  Canonical v2.8 cumulative progress is `preds/traj_files=60/60`, 23 newly resolved, and five
  observed empties. Batch 3 began pulling 20 images across two repositories at 03:52:47 with
  113 GiB free.
- Batch 3 reached `preds/traj_files=19/19` at 04:18. The final `django__django-15695` process and
  container remained alive, GPU1 was 100%, vLLM had one running request with zero waiting/errors,
  and generation advanced 207 tokens in ten seconds. It completed normally, and the retained
  batch-3 report at 04:26 records `preds/traj_files=20/20`, `resolved=4`, `empty=5`, `errors=0`.
  Canonical v2.8 is now `preds/traj_files=80/80`, 27 newly resolved, and ten observed empties.
  Batch 4 began pulling 20 images at 04:27:03 with 113 GiB free. Across the exact 946-case
  five-checkpoint denominator, 80 trajectories are written and 866 remain.
- `phaseH_eval/eval_v2p9_and_fixed150_after_merge.sh` is regression-tested and mirrored. It runs
  v2.9 full-300, then the pre-registered fixed150 panel for base, v2_bf16, v2.7, v2.8, and v2.9.
  It owns only its captured GPU1 serve PID, binds every run to model/config/index and ID hashes,
  validates the full 1188/356 merge, reports incomplete artifacts without promoting them, and
  checks GPU0 production health before and after each serve. Guarded evaluator PID 1787049 is
  waiting on train/merge waiter PID 1734405; log
  `/tmp/eval_v2p9_and_fixed150_after_merge.log`. Its terminal marker includes the evaluator PID
  and Linux process-start ticks. The prior idle evaluator PID 1776115 was stopped by exact ID before
  this hardened waiter was launched; no eval artifact existed yet.
- Report-and-continue after an incomplete retest terminal is intentional: retest gaps do not alter
  the 1224-row training mix, but all missing recovery evidence remains non-promotable.
- `phaseH_eval/build_teacher_sft_final_report.py` is mirrored and regression-tested. It fails
  closed on all canonical ID-list counts and SHA-256 values, embeds exact original/retest/fixed
  specs and per-file artifact hashes, and renders per-checkpoint suspect recovery, checkpoint-local
  non-suspect resampling floor, excess-over-floor estimates, `traj_files/preds`, `pull_failed`,
  model/forced/Docker/unknown empty splits, step distributions, and every common fixed150 paired
  test with `n` and both exclusion counts. It explicitly chooses no adapter. The focused reporter
  suite passes 18/18 locally and is included in the remote 87/87 phase-H gate.
- Guarded final-report waiter PID 1787251 is bound to evaluator PID 1787049 and logs to
  `/tmp/build_teacher_sft_final_report_after_eval.log`. It accepts only a fresh evaluator
  PID/start-tick completion marker, holds a nonblocking publication lock, builds in a temporary
  sibling, validates five recovery runs, five fixed150 runs, ten paired comparisons, and the
  no-adapter-selected flag, then atomically promotes `runs/teacher_sft_final_report`.
- At 04:40 UTC, canonical v2.8 batch 4 exposed that the driver's 35 GiB disk guard was only a
  pre-batch check: 20 pulls began with 113 GiB free and continued to 23 GiB while five images were
  still absent. This was a machine-safety breach, not an age-based restart. Exact pull PID 2020976
  and driver PID 1734303 were stopped; chain PID 1734278 halted normally on the failed child.
  The v2.8 serve and GPU0 production were untouched. Seventeen exact batch-4 image tags that were
  absent before the batch and had zero container references were removed without force; three
  tags had never completed, and free space recovered from 24 GiB to 106 GiB.
- Red-green regressions now prove that `retest_empties.pull()` refuses to start below 50 GiB
  (35 GiB hard floor plus 15 GiB headroom), serializes pulls across different repositories, and
  removes only the exact just-pulled image if a completed pull crosses the hard floor. The three
  tests failed against the old code for the intended reasons, pass against the fix, and the full
  authoritative remote phase-H suite passes 87/87.
- Fresh exact PID lineage after the safety restart: chain 2023879, resumed driver 2023902,
  train/merge waiter 2024092, evaluator waiter 2024162, final-report waiter 2024177, and unchanged
  v2.8 serve 1696494. The driver logged `resume: 80 instances already generated`. Live batch 4
  serialized every pull, refused nine exact images at 47.7 GiB, skipped them instead of producing
  false empty predictions, and began generation on 11 safely available instances at 04:54:25.
  The second instance-level pass retains responsibility for the nine skipped IDs.
- Batch 4 retained `preds/traj_files=11/11`, `resolved=3`, `empty=1`, and `errors=0` at 05:10.
  Its 11 exact batch-pulled images were removed without force, restoring 106 GiB free. Canonical
  v2.8 is now `preds/traj_files=91/91`, 30 newly resolved, and 11 observed empties across five
  retained reports. Batch 5 began pulling 20 images across five repositories at 05:11:28. Across
  the exact 946-case five-checkpoint denominator, 91 trajectories are written and 855 remain.
- Batch 5 retained `preds/traj_files=20/20`, `resolved=3`, `empty=1`, and `errors=0` at 05:37.
  Its exact 20 pulled images were removed without force and free space returned to 106 GiB.
  Canonical v2.8 is now `preds/traj_files=111/111`, 33 newly resolved, and 12 observed empties
  across six retained reports. Batch 6 began pulling 20 images across two repositories at
  05:37:41. Across the exact 946-case five-checkpoint denominator, 111 trajectories are written
  and 835 remain.
- Batch 6 retained `preds/traj_files=20/20`, `resolved=6`, `empty=3`, and `errors=0` at 05:59.
  Its exact 20 pulled images were removed without force and free space returned to 106 GiB.
  Canonical v2.8 is now `preds/traj_files=131/131`, 39 newly resolved, and 15 observed empties
  across seven retained reports. Batch 7 began pulling 18 images across two repositories at
  05:59:18. Across the exact 946-case five-checkpoint denominator, 131 trajectories are written
  and 815 remain.
- At 06:18 UTC the user narrowed the lane: retain only v2.8 from v2.6/v2.7/v2.8, while keeping
  base and v2 as controls and v2.9 as the new candidate. v2.8 is the bf16, full-multimodal direct
  parent of v2.9; v2.6 is 4-bit/text-only and v2.7 is text-only/source-confounded. The active
  canonical denominator is now 575 (`v2p8=211`, `v2=177`, `base=186`); v2.6/v2.7 receive no
  further runs. The fixed150 panel is now base/v2_bf16/v2.8/v2.9 (four runs, six pairs).
- The old chain/waiter lineage 2023879/2024092/2024162/2024177 was stopped by exact numeric PID
  without touching active driver 2023902, v2.8 serve 1696494, GPU0, or Docker. The replacement
  lineage is chain 2262751, train/merge waiter 2262773, evaluator 2262774, and final-report waiter
  2262775. Artifact line: `[06:18:56] waiting for existing retest driver pid=2023902`.
- The narrowed pipeline also fixes paired-panel exclusions: per-side exclusion counts now include
  IDs unusable on both sides and report the shared count, rather than silently reporting `0/0`.
  Focused local and remote gates pass 23/23; the full authoritative remote phase-H suite passes
  89/89.

### Teacher-SFT v2.9 training cutover (2026-07-26 16:06 UTC)

- The user prioritized v2.9 training over completing the remaining fixed-harness base recovery
  pass. The partial base run is preserved at `runs/retest_sr_base` with 89 completed trajectories,
  including 16/20 from the final partial batch; four interrupted cases remain resumable. Retained
  scorer reports cover only the first 73 trajectories, resolve 19, and contain one empty patch;
  the 16 final-batch trajectories are unscored. It is explicitly incomplete and non-promotable. The v2.8 recovery
  pass terminaled at `211/211` scored, 58 resolved, 31 empty, and zero pull failures. The v2 pass
  terminaled at `169/177` scored, 30 resolved, 12 empty, and eight pull failures.
- Only the exact lane-owned base driver, its four live mini-SWE containers, retest chain, and GPU1
  base serve were stopped. GPU0 was untouched. At cutover, port `8013` was closed, GPU1 was free at
  2 MiB, and production ports `8000`, `8101`, `8103`, and `8104` all returned HTTP 200.
- The guarded train/merge waiter accepted the durable incomplete-artifact handoff and started v2.9
  at `2026-07-26 16:06:29 UTC`; waiter PID `2262773`, trainer PID `3340505`, log
  `/tmp/train_v2p9.log`, and output `adapters/teacher_sft_v2p9_bf16`.
- Fresh live gates confirm dataset `data/teacher_train_mix_v2p9` has 1,224 rows and fingerprint
  `611e0ffed37878e6`. The run manifest is bf16/16-bit LoRA with `load_4bit=False`, rank/alpha 32,
  LR `2e-5`, one epoch, batch 1, accumulation 16, max sequence 16,384, warmup 8, and selective
  assistant loss. Tokenization retained `1224/1224` supervised examples and training entered
  optimizer step `0/77` with GPU1 at 100% utilization. The first four gradient microsteps advanced
  through `4/1232` in 69 seconds; the trainer's early ETA was 21,427 seconds (about 5.95 hours).
- Evidence-based ETA remains roughly 6-7 hours for training, followed by the guarded full
  multimodal merge and 1,188-tensor/356-vision-tensor validation. Verify the first optimizer-step
  throughput once, then inspect at checkpoint 20 or on an error/safety alert rather than polling
  continuously. The evaluator and final-report waiters remain bound to this train/merge waiter.

### Teacher-SFT v2.11r2 Fable-51 matched-full300 lane (2026-07-30 PDT)

- Canonical comparison scope is only v2.10 versus v2.11. v2.10 is complete and bound at
  `runs/v2p10_full300_composite.json` plus
  `runs/v2p10_full300_official_score_binding.json`: 157 resolved of the same 300 Lite IDs.
  No v2/base reruns are authorized or required for this verdict.
- v2.11r2 trains from `adapters/teacher_sft_v2p10_bf16` on
  `data/teacher_train_mix_v2p11_fable1262`, a frozen 1,262-row mix: 1,211 inherited v2.10 rows,
  36 safe Stage-A Fable rows, and 15 separately admitted late Fable rows. The 15 late supervised
  targets are exact byte-for-byte `git apply` reproductions of their bound admitted raw patches;
  the full untruncated rendered-context audit records max 26,594 tokens under the configured
  32,768-token limit.
- The apparent historical Fable `43` versus Stage-A `36` discrepancy is an audited disposition,
  not missing training data: five historical rows overlap frozen v2.10 and cannot be duplicated;
  11 fail the required focused passing-test replay gate and remain quarantined; nine newly
  re-preflighted rows are valid replacements. The 11 must never be admitted without fresh focused
  F2P replay plus mutation, execution, rendering, and loss-mask evidence.
- Live artifact snapshot at 2026-07-30 23:58 PDT: trainer unit
  `v2p11r2-fable51-train-gpu0.service` PID 1404827 and watchdog PID 1404830 were active;
  post-train chain `v2p11r2-posttrain-chain-v2.service` PID 1444662 was waiting on the exact
  trainer. The journal recorded optimizer step 74/79, microstep 1192/1264, and 1,650 seconds ETA.
  GPU0 was 100% utilized at 80,390/97,887 MiB; GPU1 was idle at 2 MiB. Do not poll at batch
  boundaries; next live check is the expected training exit near 00:26 PDT or an error/safety alert.
- The post-train chain validates a changed finite adapter and watchdog evidence, performs the
  32,768 context audit, then uses the bounded 12-GiB-RSS merge and portability gate before
  `eval_v2p11_full300_after_merge.sh`. It rejects incomplete acceptance artifacts rather than
  treating their existence as completion. Existing dual-production coexistence code is fail-closed:
  when enabled, GPU0 `vllm.service` plus ports 8000/8101/8103/8104 must remain healthy and GPU1
  serve watchdogs own only their exact candidate PID.
- GPU0 became unavailable after training: at 08:48-08:51 PDT on 2026-07-31 it was occupied by the
  external production vLLM stack (EngineCore PIDs 1618662, 1619429, and 1620344; router PID
  1620662 on :8000). Do not stop or use these processes. The temporary dual-panel candidate path
  was removed before evaluation; v2.11 full300, empty-only correction, join, official score binding,
  and verdict run solely on GPU1/:8013 with the existing 40-GiB admission and exact-PID 12-GiB
  watchdog. Host-wide Docker transaction locking remains enabled for pull/remove safety.
- Training itself completed successfully at 00:21 PDT. The first post-train chain stopped before
  merge because its audit accepted only numeric loss/grad-norm objects while the retained trainer
  journal emits finite numeric strings. The adapter, 820 tensors (410 LoRA-A/410 LoRA-B), 79/79
  optimizer completion, and watchdog exit all passed before that parser assertion. The resumed
  chain reuses only the immutable nonempty journal/watchdog pair, rejects partial capture, converts
  finite numeric strings while rejecting booleans/NaN/infinity, and must not retrain.
- At 08:35 PDT on 2026-07-31, the unit metadata for the already-complete training invocation had
  been garbage-collected. Its retained result remained `success`/`0`; resumption is therefore
  permitted only with both immutable nonempty captured logs, which bind the exact trainer PID,
  optimizer completion, adapter save, and watchdog target exit. The resumed GPU1-only chain is
  `v2p11r2-posttrain-resume-gpu1b.service` PID 1680988. It passed the 24-GiB RAM admission with
  34.6 GiB available, revalidated the 820 changed adapter tensors, and has no GPU0 ownership.
  The threshold is 24 GiB for this invocation because external GPU0 production keeps the prior
  40-GiB threshold unattainable; every merge/serve phase retains the exact 12-GiB watchdog floor.
- At 08:55 PDT, GPU1 portability is active at
  `runs/v2p11r2_v2p10init_fable51_portability_stock/candidate`: 4/10 tasks have submitted, four
  are active, and two are queued. Controller PID 1685038 and the exact candidate vLLM PID 1683328
  are live; the candidate remains on GPU1/:8013 with 200-response evidence and about 81 generated
  tokens/second. There is no traceback, fatal/critical/OOM, HTTP 429/500, dead-engine, kill, or
  watchdog event. Do not intervene merely because a trajectory is long; gate completion requires
  all ten trajectories and the published portability artifact.

### v2.11r2 post-portability GPU boundary (2026-07-31 PDT)

- Portability completed and passed at
  `runs/v2p11r2_v2p10init_fable51_portability_gate.json`: candidate 6/10 resolved, zero empty
  patches and zero format errors; control 5/10 resolved with one empty patch. The candidate
  vLLM/watchdog exited cleanly and GPU1/:8013 is released.
- GPU0 is now externally owned and unavailable to this lane. The active v2.11 runners
  (`run_v2p10_empty_diff_goal.sh`, `run_v2p11_portability_gate.sh`, and
  `eval_v2p11_full300_after_merge.sh`) hard-reject every `GPU_INDEX` other than `1` before any
  model/process action; all GPU0 health/service dependencies were removed. The post-train defaults
  now name GPU1 units/logs only. Focused remote validation: Bash syntax clean and 45 tests passed.
- Full300 has not begun. Before launch, completion provenance must reconcile the passed portability
  gate's candidate model contract with the current merged-model contract; do not weaken that check
  or start evaluation until the exact difference is explained and bound.
- The exact reconciliation found no model change: all 21 model-artifact bindings, path, config hash,
  and index hash matched. The gate omits `served_name` from its artifact contract and binds it
  separately, whereas provenance had incorrectly compared against an enriched contract. The validator
  now compares the native gate contract and separately binds the candidate name; focused remote
  validation passed 48 tests. Provenance was published at
  `runs/v2p11r2_v2p10init_fable51_completion_provenance.json` with `status=complete` and 1,262 rows.
- Canonical full300 launched at 09:46 PDT as `v2p11r2-full300-gpu1.service`, MainPID `1732585`,
  with `GPU_INDEX=1`, port `8013`, and no GPU0 dependency. It will run fixed150, bounded
  empty-only retries, complement150, then immutable join/official score binding/comparison. Do not
  inspect at batch boundaries; next check is the first launch-health/ETA boundary or an error.

### v2.11r2 final evidence gate (2026-07-31 PDT)

- The direct-LoRA evaluator deliberately bypasses the old posttrain completion chain, so its legacy
  `v2p11_goal_completion_audit.py` cannot be used for this candidate. A separate
  `v2p11r2_goal_completion_audit.py` now replays the published full300 comparison and official
  score bindings, revalidates the direct-LoRA provenance (dataset, exact recent-Fable patch targets,
  79-step LoRA run, merge, and portability), and reruns the original Stage-A contract for all 36
  frozen Fable rows. It will publish only for a trustworthy win with no empty regression.
- Run the new audit only after the active evaluator has published its immutable verdict. Its final
  artifact reports both Fable generations separately (36 Stage-A + 15 recent strict rows), full
  32,768-token context evidence, and bound score/failure-analysis artifacts. Focused remote tests:
  `36 passed` across the new audit, r2 provenance, full300 comparison, evaluator runner, and
  Stage-A contract tests. This change does not alter the active full300 unit.
- A CPU-only waiter, `v2p11r2-final-audit-waiter.service` (PID `1983550`, invocation
  `6b737e1f5c054bab868815f4a69dd568`), is active. It waits for the immutable candidate verdict,
  reruns the r2 final audit with the exact candidate/control inputs above, and fails closed without
  writing an audit if the evaluator becomes terminal first or the comparison is not trustworthy.
- Fable reconciliation: the historical `43` strict sources are not 43 appendable rows. Five are
  already inherited by v2.10, 27 non-overlapping rows pass the current downstream gate, and 11 are
  excluded because their distilled traces lack a focused passing task-contract test. Nine
  re-preflighted replacements make the 36 Stage-A additions. The late campaign had 17 raw-strict
  admissions but its authorized, fully rendered/loss-checked/decontaminated selection is 15;
  the two unselected raw-strict rows are not packaged training rows. Thus r2’s 51 additions
  (36 Stage-A + 15 late) include every currently eligible, training-admitted trace within the
  stated 15-row late-Fable cap; no safe packaged trace was omitted.
- Final-audit hardening after independent review: the r2 audit now requires failure-analysis
  coverage for every instance that is not jointly resolved and captures/rechecks immutable input
  bindings before publication. The old CPU waiter had a verdict-publication race and was replaced
  without touching the evaluator by `v2p11r2-final-audit-waiter.service`, PID `2016694`, invocation
  `91c61c88344e4e7cb41e9c5282af8cf3`; it rechecks for a just-published verdict before failing.
  Focused remote validation after this change: `38 passed`.

### Fable reasoning-contract derivative (2026-07-31 PDT)

- Fixed150’s matched controller contract is identical for v2.10 and v2.11r2 (DockerSelfRetry,
  seed 1, temperature 0.7, 120 steps). The preliminary corrected panel is 89/150 resolved for
  v2.10 and 72/150 for v2.11r2. v2.11r2 reduced repeat loops (6 versus 14) and removed all eight
  first-pass empties on retry, but had 17 tool-format errors versus v2.10's seven. This is a
  behavior tradeoff, not a panel/configuration mismatch; full300 remains the sole promotion gate.
- `phaseD_sft/render_fable_reasoning_contract.py` creates a separate, source-bound derivative and
  leaves all raw Fable evidence and the active evaluator unchanged. It preserves non-Fable rows,
  nonblank assistant text, and all tool calls byte-for-byte. For blank supervised Fable `bash` turns
  it adds only a deterministic, command-class-matched generic rationale; any noncanonical/malformed
  Fable tool call fails closed. Local and remote focused tests passed (4 direct renderer tests; 8
  including the existing Fable extension tests).
- The fully sealed derivative is
  `data/teacher_train_mix_v2p11_fable1262_reasoned_v1`: 1,262 rows, 92 Fable rows, 517 canonical
  supervised Fable `bash` turns, and 468 formerly blank turns supplied with the production
  reasoning-plus-tool-call contract. It has zero Lite evaluation overlap, boolean-loss and canonical
  bash structural gates, and the full 1,262-row native format/loss gate passed with zero failures;
  its manifest binds `format-gate.e0fd3cd3440da23367da638f6e06b83415c3e9a784aa0bc4299336644b06355d.json`.

### v2.11r2 cutoff and v2.11r3 corrected continuation (2026-07-31 PDT)

- User-directed decision: fixed150 is sufficient to reject r2 as a promotion candidate. Its matched
  result is 72/150 resolved versus v2.10's 89/150, despite zero residual empty patches after its
  exact eight-task retry. The in-progress complement150 was therefore intentionally stopped rather
  than spending GPU time to produce a no-longer-useful r2 full300 comparison.
- At 15:55 PDT, the exact `v2p11r2-full300-gpu1.service` cgroup (MainPID 1732585) and the exact
  `v2p11r2-final-audit-waiter.service` cgroup (MainPID 2016694) were stopped. Both report
  `ActiveState=inactive`, `MainPID=0`; port 8013 is released. Preserve all r2 fixed150 and partial
  complement artifacts, but do not treat r2 as complete, promotable, or a full300 verdict.
- The next candidate is a clean v2.11r3 continuation from the immutable v2.10 adapter, not a
  continuation from r2. `phaseH_eval/train_v2p11r3_reasoned_gpu1.sh` accepts only GPU1, rechecks
  the sealed 1,262-row reasoning-contract derivative and all 1,262 untruncated rendered contexts,
  then trains from `adapters/teacher_sft_v2p10_bf16` to
  `adapters/teacher_sft_v2p11r3_v2p10init_fable_reasoned_bf16`. Its deliberately matched r2
  optimizer contract is rank/alpha 32/32, lr 2e-6, one epoch, batch 1, grad accumulation 16,
  32,768 tokens, 79 steps, warmup 4, and bounded Unsloth checkpointing. It uses an exact-PID
  memory watchdog only; it has no production or other-GPU health dependency.
- The first r3 launch intentionally failed closed before model loading because the shared trainer
  admission helper recognized only the older `standard_native_format_loss_gate` spelling. The helper
  now accepts the reasoning-contract manifest only when its exact variant, 92 Fable rows, 517
  canonical tool calls, 468 reasoned turns, zero-overlap structural gates, and bound full-dataset
  format report all revalidate. Focused tests passed (30); this does not relax ordinary teacher-data
  admission.
- The corrected r3 launch began at 16:01 PDT under
  `v2p11r3-reasoned-train-gpu1.service` (invocation
  `2685ede85c054d649a3794aa9e7e07ca`). Its full 1,262-row context artifact is
  `runs/v2p11r3_reasoned_context_audit.json`: max rendered length 26,594, zero rows over 32,768.
  Trainer PID is 2526889, bound directly to GPU1; its exact-PID watchdog has a 12-GiB stop floor.
- At 16:09 PDT optimizer step 1/79 completed in 351 seconds. At 16:12 PDT the trainer's own
  remaining-time estimate was 24,810 seconds, projecting completion around 23:05 PDT. Available
  host memory has ranged from 39 to 47 GiB, with no watchdog safety event or restart. Do not poll
  it at batch boundaries; next live check is that ETA or an error/safety alert.
- The GPU-free posttrain handoff is ready at `phaseH_eval/run_v2p11r3_posttrain_chain.sh`. It
  cannot run while the training unit is active and requires its immutable 79/79 journal plus exact
  watchdog evidence. It performs only a bounded 12-GiB-RSS merge, portability gate, checksum-bound
  r3 provenance, then the same fixed150+complement150 controller with empty-only correction.
  `v2p11r3_completion_provenance.py` revalidates the reasoned dataset and delegates the underlying
  Fable source/replay/mutation evidence to the already-validated source dataset; it also reruns the
  production Stage-A contract (1,247 rows = 1,211 frozen v2.10 rows + 36 strict Fable additions).
  The real r3-data provenance preflight passed both the 15 latest strict Fable replay bindings and
  the 36-row Stage-A contract; it does not make r2 a candidate.
- A CPU-only `v2p11r3-posttrain-waiter.service` (PID 2532604, invocation
  `02aa5b77267e4679b36554555b99587f`) started at 16:15 PDT. It checks the active r3 training unit
  only every 60 seconds, then invokes `run_v2p11r3_posttrain_chain.sh`. The chain itself refuses to
  start if training is active and fails closed unless the immutable 79/79 trainer and watchdog
  evidence are complete. No serving, merging, or evaluation is running while the trainer owns GPU1.
- On a successful trustworthy full300 win, the same chain now invokes
  `v2p11r3_goal_completion_audit.py`. It replays the exact 300-instance verdict, both official score
  bindings, the current r3 provenance/model, the 36 strict Stage-A and 15 latest strict Fable source
  evidence, 468 reasoned Fable tool turns, and per-instance failure analysis. It publishes only a
  trustworthy win; a losing full300 verdict remains preserved evidence but cannot be misreported as
  goal completion. Focused local and remote validation passed 23 tests.

### v2.11r3 behavior-poststage handoff (2026-07-31 18:55 PDT)

- The final candidate is now `teacher_sft_v2p11r3_behavior`, not the direct reasoned r3 adapter.
  After the 79-step reasoning SFT, the guarded chain runs a 138-row/105-step recovery SFT, a
  one-step KTO canary, a 25-step coverage-selected KTO pass, the final multimodal merge, the
  controller-free portability gate, and exactly one matched Lite300 evaluation. The final audit
  still requires a complete v2.11 win over the bound v2.10 result of 157/300 with no empty-patch
  regression.
- Model-level empty/loop mitigation is bound into the poststage data: 606 behavior rows include
  33 empty-terminal, 55 repeated-read-loop, and 228 wrong-nonempty negatives. The deterministic
  50-row KTO coverage pass requires 25 desirable patches plus 8 empty, 8 loop, and 9 wrong-edit
  negatives. Recovery SFT uses 32,768 context, rank/alpha 32/32, and 105 optimizer steps.
- Remote production admission revalidated at 18:54 PDT: recovery `138` rows/`105` optimizer steps,
  behavior `606` rows, `707` excluded IDs, `12` excluded repositories, `300` full-evaluation IDs,
  and zero overlap. Sealed SHA-256 values are recovery train
  `eda502d05f736426728651953e76330d24addd39471e82bf5fe423c896ba4b66`, recovery manifest
  `c996953fb810f3e4642bd458dde7226e245bcb99ac02fec249e38a6d9d21fc2a`, behavior data
  `525526bffe58a64a3e42b0d733d87b60dd62e3d2dbaa8362de7b74b00a6699e9`, behavior manifest
  `2b77db5de27de7f3880ac711fd0721efb8c00ffe82ce8ca6f219225c5fb43122`, and exclusions
  `00171a1e7796fac2103317bcf4f05af42e46ff92db0041fd4a26ccbaaa7e4bd8`.
- The integration commits are `b251340`, `f0f3bbb`, `0adcd9f`, and `73f708a`. Local and remote
  focused verification both pass `93/93`; all five shell launchers and all five embedded Python
  heredocs compile. The chain is restartable at immutable training evidence, recovery/KTO phases,
  portability, provenance, full300, and goal-audit boundaries. Existing artifacts are reused only
  after current-input revalidation; partial final-file writes use atomic no-replace publication.
- Live artifact line at 18:51 PDT: training unit `v2p11r3-reasoned-train-gpu1.service` has wrapper
  MainPID `2526707`; exact GPU1 trainer PID `2526889` is at optimizer step `32/79`, microstep
  `525/1264`, GPU1 utilization `98%`, and trainer ETA `14,168` seconds. The evidence-based r3
  training boundary is about 22:48 PDT. No process or query targets GPU0.
- CPU-only waiter `v2p11r3-posttrain-waiter.service` is active as PID `2577442`, invocation
  `b735b35d76754947bb62cebee84ce49f`. It checks only the named training unit every 60 seconds and
  then executes `phaseH_eval/run_v2p11r3_posttrain_chain.sh`; it performs no GPU work while r3 owns
  GPU1. Allow roughly 4-7 hours after r3 for recovery/KTO/merges/portability and roughly 12-14 hours
  for the matched full300 including bounded empty correction. The current final-verdict window is
  approximately 16:00-20:00 PDT on 2026-08-01, subject to actual recovery/KTO throughput.

### v2.11r3 poststage interruption recovery (2026-07-31 19:20 PDT)

- Commit `818a2de` makes incomplete recovery-SFT and KTO outputs restartable without deleting
  evidence. Checkpointless recovery output and every KTO `.inprogress` directory are atomically
  renamed to unique `.interrupted-*` archives; a verified real recovery checkpoint is resumed.
  Top-level and checkpoint symlinks fail closed, archive collisions cannot replace either
  directory, and the launcher branches on the helper's authoritative post-race status.
- The commit contains exactly four files:
  `phaseH_eval/train_v2p11r3_behavior_gpu1.sh`, its launcher test,
  `phaseH_eval/v2p11_poststage_recovery.py`, and its recovery tests. All unrelated tracked and
  untracked work remains outside the commit. Local and remote broad verification each passed
  `99/99`; shell syntax, helper compilation, and checksum parity also passed. The remote test
  interpreter is `.venv-eval/bin/python`; runtime recovery remains bound to `.venv-train/bin/python`.
- After verifying the prior CPU waiter's exact PID and loop command, it was replaced by
  `v2p11r3-posttrain-waiter.service`, PID `2584583`, invocation
  `eafdcf7be89e4b268c70cb1b495781bf`. It retains the 60-second named-unit wait and now has
  `Restart=on-failure`, `RestartSec=60`, `StartLimitBurst=3`, and a 30-minute start-limit interval.
  It performs no GPU query or work while waiting. The training and final-verdict ETA boundaries
  remain approximately 22:48 PDT on 2026-07-31 and 16:00-20:00 PDT on 2026-08-01, respectively.

### v2.11r3 promotion proof gates (2026-07-31 20:16 PDT)

- Commit `042feca` binds the r3 init adapter to the rebuilt canonical v2.10 training-lineage
  contract, including exact adapter weights, `adapter_config.json`, and resolved adapter directory.
  Fresh publication, provenance reuse, evaluator preflight/comparison, and the final goal audit all
  receive the canonical trust root externally; a provenance report cannot nominate its own lineage.
- Commit `d14d763` publishes every unresolved non-empty full300 ID (including classified model
  failures), its paired introduced/eliminated sets and delta, and requires a non-positive delta for
  promotion. The final audit reconstructs `wrong = full - resolved - empty`, checks the failure
  classification split, and requires verdict resolved/empty IDs and counts to match both validated
  official score bindings. The omitted-ID and contradictory-score attacks now fail closed.
- Independent review returned zero findings after both adversarial fixes. The complete tracked
  v2.11 contract suite passed `129/129` locally and `129/129` remotely; Python compilation and both
  r3 shell syntax checks passed. Checksum-only rsync verification is blank for all 15 committed
  proof-gate paths.
- CPU-only remote revalidation published/reused
  `runs/v2p11_v2p10_training_lineage.json` with SHA-256
  `4f7affc19fc56f370987a728c01d21c36290ffdebb9e87e4d8304210f32ade12`. It binds canonical v2.10
  adapter weights SHA-256 `b91b41a5b0b9b82395d5b1aecca3b24536c110663f276d0caaec9e0e6a3bcb45`
  and config SHA-256 `896cd7dc25f89a534058a0c2943884bc92d2fc45068469ebbd39cca361cce4b0`.
- The waiter remains at its last verified identity above; it was not polled during this offline
  proof work. Do not query or touch GPU0. The next live evidence boundary remains 22:48 PDT, when
  the named training unit, exact trainer evidence, waiter/poststage unit, and resulting immutable
  artifacts should be checked once.
- The 20:23 PDT requirement-by-requirement audit found no missing pre-boundary proof logic. A fresh
  CPU-only remote dataset validation re-proved 1,262 admitted rows, 92 Fable rows, 517 canonical
  Fable bash turns, 468 reasoned tool turns, 15 recent strict source IDs, 36 Stage-A additions on
  the frozen 1,211-row v2.10 mix, max rendered length 26,594 under the 32,768-token limit, and the
  current reasoned train SHA-256 `be1246db2e105256138dcaee79f58d7bd0daaf26910380799d8732a81b88b792`.
- Fresh CPU-only poststage admission re-proved 138 recovery rows/105 steps, 606 behavior rows,
  negative counts 33 empty + 55 loop + 228 wrong-nonempty, 707 combined exclusions across 12
  repositories, all 300 evaluation IDs, and zero overlap. The live waiter was not queried; its
  static unit definition still waits on only `v2p11r3-reasoned-train-gpu1.service` and then execs
  the newly synced `phaseH_eval/run_v2p11r3_posttrain_chain.sh`. Remaining requirements are runtime
  completion/merge/portability/full300 artifacts and a score-qualified final audit.

### v2.11r3 full300 verdict and successor ablation (2026-08-01 09:35 PDT)

- Reasoning SFT, recovery SFT, KTO, final merge, portability, and the matched Lite300 evaluation
  are complete. The corrected candidate composite is 300/300 with 124 resolved, two empty, zero
  pull failures, and zero Docker failures; canonical v2.10 is 157 resolved and two empty. The
  candidate therefore loses by 33 resolved tasks and adds 33 wrong-nonempty outcomes. It is not
  promoted and the goal remains active.
- The first-pass comparison is 122/300 with seven empties for r3 versus 148/300 with 28 empties for
  v2.10. Empty/loop mitigation worked, but correctness did not: r3 has 21 repeat loops and five
  tool-format errors versus 24 and ten for v2.10. Corrected paired movement is 41 v2.10-only wins
  versus eight r3-only wins. The next revision must preserve the behavior gain without the patch
  correctness regression.
- The checksum-bound verdict artifacts are
  `runs/v2p11r3_behavior_vs_v2p10_full300.json` SHA-256
  `5fe08709fbcecac493ec272f43457b54580d80e7cdf4e9ac318b217f5b000796` and its Markdown companion
  SHA-256 `1a6f9a9675f602c5ca6ef4928db80c3d1768e3f9825222f954af88ffbbd2d260`.
  The evaluator did not rerun any task while publishing this verdict.
- Verdict publication initially failed because provenance represented the inactive single-file
  hash as `null` while the sharded evaluation composite omitted that optional key. All model paths,
  names, config/index hashes, and 21 weight-artifact bindings were identical. The validator now
  normalizes only a missing inactive hash; changed or extra contract fields still fail closed.
  The focused local and remote suites both pass 26/26.
- The obsolete CPU retry unit `v2p11r3-posttrain-waiter.service` is inactive with `MainPID=0`.
  A single recovery-stage ablation is active on GPU1 as
  `v2p11r3-recovery-ablation50-gpu1.service`, MainPID `3618166`, invocation
  `f5879a7a71784c7298e3d760b8ada030`. Its immutable ID set
  `data/v2p11r3_stage_ablation50_ids.json` contains exactly the 38 first-pass v2.10-only tasks plus
  the 12 first-pass r3-only tasks and has SHA-256
  `581727784f273e54daa8a70833a17a7fa16581a7619a1a3c87733f21866251b3`. Existing v2.10 and final
  r3 outputs cover the same 50 tasks, so only the already-merged recovery checkpoint is running.
  Do not query or touch GPU0. Check next at the ablation completion/error boundary, estimated around
  11:00-12:00 PDT.

### v2.11r3 recovery-stage ablation verdict (2026-08-01 10:53 PDT)

- `v2p11r3-recovery-ablation50-gpu1.service` completed successfully and is now inactive with
  `MainPID=0`, `ExecMainStatus=0`, and `Result=success`; GPU1 had no compute process after cleanup.
  The exact run is `runs/v2p11r3_recovery_ablation50` over the immutable 50-ID artifact
  `data/v2p11r3_stage_ablation50_ids.json` SHA-256
  `581727784f273e54daa8a70833a17a7fa16581a7619a1a3c87733f21866251b3`.
- The run is complete at 50 predictions, 50 unique trajectories, 50 usable outcomes, 23 resolved,
  one model-empty output (`django__django-12915`), zero pull failures, and zero Docker failures.
  `runs/v2p11r3_recovery_ablation50/acceptance.json` has SHA-256
  `ba44b507725c9a1669732f48ba5e6647d90607bcd94ff436601d5a719eeb2595`;
  `summary.json` has SHA-256
  `d9ffc105dbabdffca1c241e9ba9aa6827350a048330d75ee8c190a49e1c9a60f`;
  `preds_all.json` has SHA-256
  `7a11ad106303eacbcc77a90df3b68c1cc664cdb030ad85b27ef371a5e31c268f`.
- On the exact first-pass decision set, canonical v2.10 resolves 38/50 and final r3 resolves 12/50.
  Recovery versus v2.10 is 19 both resolved, four recovery-only, 19 v2.10-only, and eight neither;
  recovery therefore misses the v2.10 floor by 15. Recovery versus final r3 is four both resolved,
  19 recovery-only, eight r3-only, and 19 neither, a net recovery advantage of 11. This isolates the
  main correctness collapse to recovery SFT, with KTO worsening the same set further. Empty/format
  mitigation alone is not sufficient to promote either checkpoint.
- No recovery full300 will run. The next candidate must remain anchored to v2.10 while retaining only
  an attenuated amount of the admitted 92-row Fable and behavior signal already realized in r3; it
  must pass controller-free portability and a matched pre-full300 correctness/empty/loop gate before
  consuming a full300 evaluation. Do not query or touch GPU0.

### v2.11r4 conservative successor live chain (2026-08-01 12:00 PDT)

- The sole candidate is `teacher_sft_v2p11r4_blend25`, defined as
  `0.75 * v2.10 + 0.25 * v2.11r3_behavior`. Canonical v2.10 remains 157/300; r3 remains 124/300.
  Full300 is conditional on controller-free portability10 and a matched fixed150 gate that preserves
  correctness and wrong-nonempty while strictly improving empty patches or repeat loops.
- Gate/chain code is committed as `ba7aeb3`; the padding-only tokenizer compatibility fix is committed
  as `9e96eee`. The first materialization attempt failed closed before writing an output because r3's
  merge had copied adapter padding state. Live comparison proved identical 262,144-token vocabularies,
  special/added-token IDs, chat template, and representative encodings; only
  `tokenizer.json.padding` and `tokenizer_config.json.padding_side` differ. The fix copies the v2.10
  tokenizer, binds both raw hashes plus a normalized semantic hash, and still rejects vocabulary or
  token-ID drift. Focused verification is 97 passed.
- The corrected chain launched at `2026-08-01 11:59:01 PDT` as
  `v2p11r4-blend25-chain-gpu1.service`, exact PID `3798384`. Its CPU-only child is
  `v2p11r4-blend25-materialize.service`, exact PID `3798398`. At 25 seconds the bounded staging output
  was 5.8 GiB and the child was actively CPU/I/O bound at about 2.7 GiB RSS. No final output is yet
  published and no GPU is used during construction. The chain will proceed automatically to GPU1/8013
  portability, fixed150, and only on gate pass full300. Do not query or touch GPU0.
- Next evidence boundary: materialization completion/error, initially expected around 12:10-12:20 PDT.
  If it succeeds, portability is approximately 1-2 hours and fixed150 approximately 6-10 hours; a
  passing full300 then requires another approximately 8-14 hours.

### v2.11r4 portability10 live boundary (2026-08-01 12:25 PDT)

- The interpolated checkpoint is atomically published and lineage-verified at
  `/media/ironbcc/CrucialX10/models/merged/teacher_sft_v2p11r4_blend25`. Its complete manifest has
  SHA-256 `3026c4cd158fc66460f01d10b1bc2288c806de2ca93e125aba0844df54656c66`, 1,188
  tensors, 20 shards, and 62,546,177,752 tensor bytes. The bounded materializer peaked at
  10,018,902,016 bytes RSS. Only the admitted tokenizer padding fields differ between parents; the
  published output copies the canonical v2.10 tokenizer.
- The resumed automatic chain is active as `v2p11r4-blend25-chain-gpu1.service`, exact chain PID
  `3803607`, with portability wrapper PID `3804523`. At `12:25:25 PDT`, exact vLLM PID `3806254`
  became ready on `:8013`; `/v1/models` returns only `teacher_sft_v2p11r4_blend25` rooted at the
  exact published path. Engine PID `3806533` owns GPU1 and the 12-GiB host-memory watchdog is PID
  `3806255`. Do not query or touch GPU0.
- The controller-free stock-harness portability driver is exact PID `3807933` over the immutable
  verified 10-task set. It began at `12:25:29 PDT`; candidate artifacts were still 0 predictions,
  0 trajectories, and 0 batch-prediction files at launch. The already-verified matched v2.10
  control remains 10 predictions, 10 trajectories, and one batch-prediction file. No portability
  gate artifact or fixed150 artifact exists yet.
- Next evidence boundary is portability completion/error, projected around `13:25-14:25 PDT` from
  the prior measured lane. On pass, the same chain starts matched fixed150 automatically; its
  provisional completion bracket is `19:25 PDT` through `00:25 PDT` on 2026-08-02, to be replaced
  by a measured ETA after the first completed fixed150 batch. Full300 remains conditional on that
  gate and would add approximately 8-14 hours.

### v2.11r4 fixed150 control preflight (2026-08-01 12:31 PDT)

- While portability runs, a fresh CPU-only `_complete_run` validation re-hashed and accepted the
  exact paired v2.10 control at `runs/fixed150_v2p10`. The immutable ID set has 150 unique rows and
  SHA-256 `abc550841d4a64740da2a9f18d717973a501c4bf138755d3d55883c67d9c1390`.
  The control has 150 prediction rows, 150 trajectories, nine batch `preds.json` files, 81 resolved,
  18 empty, 51 wrong-nonempty, 12 repeat loops, 11 tool-format errors, and 9,595 assistant responses.
- The revalidated control binds served name `teacher_sft_v2p10`, canonical model path
  `/media/ironbcc/CrucialX10/models/merged/teacher_sft_v2p10_full`, model-index SHA-256
  `77a39432f90e34fd77d1605b2d7c9f8b4427b4f671c8b9148583bfd1ff3487ee`, and config SHA-256
  `e967dd38bc5cfd38bd09a995a7bf4a754075df2b46aba68f7fbb5a791e6d8dd1`.
  Its acceptance SHA-256 is `6fde5e10adb45213836ae56d2f967a3f5b801f377dc2acd8c242c3d821dbc54e`
  and `preds_all.json` SHA-256 is
  `b23f142f0e50d1cb9b9ab0f4815b96a25c06ae78ae968c4dba9b795a4caa5ea6`.
- The candidate fixed150 launcher uses the identical comparison contract: seed 1, temperature 0.7,
  step limit 120, 16 generation/scorer workers, batch 20, five pull workers, and
  `docker_selfretry.DockerSelfRetryEnv`. The successor gate rejects any harness or model-contract
  mismatch before comparing resolution, wrong-nonempty, empty, or repeat-loop metrics. No live
  portability status was polled for this offline preflight; the next live check remains its
  completion/error boundary.

### v2.11r4 full300 and source-provenance preflight (2026-08-01 12:43 PDT)

- The correct joined-panel and official-score validators freshly replayed the canonical v2.10
  full300 evidence. `data/swebench_lite_test_ids.json` remains 300 unique IDs with SHA-256
  `b98fc2b1054dc8fdfcb94f083f43454fd568961a0b3dbf8c388c210b7b868e14`.
  `runs/v2p10_full300_composite.json` remains a complete
  `disjoint_panel_full300_composite`, SHA-256
  `c26838c892684c963202d0ce88c936c4fec522134c1eb145f31c85eb3d9f516e`, with 157 resolved
  and two model-empty outcomes (`pylint-dev__pylint-7080`, `sphinx-doc__sphinx-8801`). Its 29 empty
  retries produced 26 nonempty patches and nine additional resolutions, leaving two empty.
- The canonical v2.10 predictions SHA-256 is
  `d742ad5d25751ea5af6522be3dd361c404ce7791879679f1c0d156226616acca`; training-lineage
  SHA-256 is `4f7affc19fc56f370987a728c01d21c36290ffdebb9e87e4d8304210f32ade12`; and official-score
  binding SHA-256 is `c0717bd8c7fb2b8894267345963f08412ee4c15e11a6c904c76b41e592562a76`.
  All three revalidated against current panels, reports, model identity, and the 1,211-row v2.10
  training contract. No recent-v2.11 Fable row is attributed to v2.10.
- A full rebuild of `runs/v2p11r3_behavior_completion_provenance.json`, SHA-256
  `727d6ed77657d92739c32824ca646bdf0b1dc7949bbbbdf107814c166cc0be8b`, also passed. It
  revalidated the 1,262-row, 32,768-token, 79-step LoRA run with 820 changed tensors; 92 Fable rows;
  36 new strict Stage-A rows plus 15 recent strict exact-patch rows; max rendered length 26,594;
  707 excluded IDs across 12 repositories; and zero overlap with the 300 evaluation IDs. The
  behavior poststage remains bound to 138 recovery rows/105 steps, 606 behavior rows/25 KTO steps,
  and the exact recovery, behavior, exclusion, adapter, journal, watchdog, merge, and model hashes.
- One exploratory command initially applied `_validate_composite_source`, which is intentionally a
  single-panel validator, to the joined full300 composite and therefore returned `source composite
  is incomplete`. Root-cause inspection showed the joined artifact correctly uses type
  `disjoint_panel_full300_composite` and nested fixed/complement `selected_attempts`; the production
  chain does not make that incompatible call. Re-running through `validate_official_score_binding`
  and `validate_v2p10_training_lineage_contract` passed without edits. No canonical artifact was
  changed. The next live check remains the portability completion/error boundary.

### v2.11r4 downstream regression preflight (2026-08-01 12:46 PDT)

- A fresh focused regression pass covering the successor gate, automatic r4 chain, empty-retry
  panel construction, joined full300 composite, official score binding, v2.10 lineage, full300
  comparison/evaluator, and r3/Fable provenance passed `59/59` locally and `59/59` in the remote
  `.venv-eval` environment. Bash syntax is clean for
  `run_v2p11r4_blend25_chain.sh`, `run_v2p11_portability_gate.sh`,
  `run_v2p10_empty_diff_goal.sh`, and `eval_v2p11_full300_after_merge.sh`.
- SHA-256 parity was checked for the four production shell runners, the six Python gate/composite/
  lineage/provenance modules, and all nine focused test files; all 19 local and remote hashes match
  exactly. No live portability status was polled and no runtime or canonical artifact was changed.
