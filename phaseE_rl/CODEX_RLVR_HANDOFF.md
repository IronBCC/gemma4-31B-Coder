# CODEX HANDOFF — RLVR lane (verified-reward round)

The controlled Dr. GRPO-versus-GRPO comparison is complete. On 2026-07-14 the user promoted
standard GRPO as the current Python policy and authorized the next Python lever: a
Docker-verified reward round on RLVR-v2. The base remains frozen; do not start unrelated
training or alter the promoted serving artifact.

## 0. Why this lane exists
SFT plateaued: v7/v8 mixtures both peaked at ~11-13/30 non-empty patches on the 30-case
SWE-Lite temp-0.7 smoke (gate ≥18/30), resolved 6-7/30, while hard30 semantics hit 25/30
(best-ever). The model READS but doesn't EDIT (perseveration). RLVR optimizes the edit
decision directly. User picked "why not DrGRPO" (arXiv:2503.20783) mid-run — its two fixes
(no length normalization: stops subsidizing rambling 0-reward completions; no σ-scaling:
stable advantages on our discrete rewards) map onto our observed pathologies.

## 1. Current verified state and promotion decision (2026-07-14)
- Both matched 100-step runs are complete: Dr. GRPO at
  `adapters/swe_drgrpo_v1_s2/checkpoint-100` (`/tmp/drgrpo_s2.log`) and standard GRPO at
  `adapters/swe_grpo_v1_s2/checkpoint-100` (`/tmp/grpo_cmp_s2.log`). The only intentional
  config difference is `--loss-type dr_grpo` versus `--loss-type grpo`.
- Dr. GRPO has been three-way merged without altering the frozen base at
  `/media/ironbcc/CrucialX10/models/merged/swe_drgrpo_v1`. Its hard30 result is **19/30**;
  the fresh same-day cp20 anchor is **20/30**, so it passes the regression floor (>=18), not
  a promotion claim. Its completed 30-case temp-0.7 SWE-Lite artifact is
  `runs/smoke_swe_drgrpo_v1_t07_lite_0_30`: **17/30** non-empty patches (promotion miss by
  one), **2/30** format errors (6.67%), **23/30** edit-reaching, median first edit command
  **13**, and **7/17 resolved** non-empty predictions. Do not relaunch it.
- Standard GRPO was merged to `/media/ironbcc/CrucialX10/models/merged/swe_grpo_v1` with
  the same frozen-base + v8cp30 + RL-LoRA three-way merge. Its fresh hard30 result is
  **18/30** against the same-day cp20 anchor at **20/30**, exactly passing the regression
  floor (>=18). Its completed 30-case temp-0.7 SWE-Lite artifact is
  `runs/smoke_swe_grpo_v1_t07_lite_0_30`: **20/30** non-empty patches, **3/30** format
  errors (**10.0%**, so it misses the strict `<10%` format gate), **25/30** edit-reaching,
  median first edit command **14**, and **6/20 resolved** non-empty predictions.
- **USER PROMOTION:** `adapters/swe_grpo_v1_s2/checkpoint-100` is the current Python policy,
  and `/media/ironbcc/CrucialX10/models/merged/swe_grpo_v1` is retained as its serving
  artifact. The non-promoted `merged/swe_drgrpo_v1` was deleted (58 GB reclaimed); both
  adapter checkpoints remain for controlled comparison. The former strict-format ambiguity is
  superseded by this explicit promotion decision.
- Curve caveat to include in the comparison: Dr. GRPO has one step-14 outlier (loss 11.007,
  grad norm 1587.5, KL 275.2); standard GRPO peaks at loss 0.00274, grad norm 43.9, and KL
  0.0684. Both completed 100 steps, so behavioral artifacts remain the decision evidence.

### Reproducible smoke extraction

`phaseH_eval/summarize_smoke.py` reads raw `*.traj.json`, `preds.json`, and
`exit_statuses_*.yaml`; `preds.json` is authoritative for the non-empty-patch gate. Run:

```bash
PYTHONPATH=$PWD .venv-eval/bin/python phaseH_eval/summarize_smoke.py \
  runs/smoke_<name>_t07_lite_0_30/<name>
```

It reports attempted cases, non-empty patches, format errors/rate, edit reach, first
source-edit distribution, and resolved count once the harness final report exists. Tests: `PYTHONPATH=$PWD .venv-eval/bin/python -m unittest
phaseH_eval.tests.test_summarize_smoke phaseH_eval.tests.test_smoke_single`.

## 2. Reward, dataset, and integration state — do NOT re-hit prior fixes
- **Reward** (shaped, parse-only phase 1): 0.0 no tool call / 0.2 read / 0.6 edit /
  1.0 edit touching a file named in context. Regexes in the script; 8/8 unit tests
  (`phaseE_rl/tests/test_grpo_swe_reward.py`).
- **RLVR-v2 dataset COMPLETE:** `data/rlvr_swe_decision_v2.jsonl` (1,895 rows) plus
  `data/rlvr_swe_decision_v2_manifest.json`. Counts: swe-smith 700, oracle 256, OpenSWE 300,
  Kwai 500, smoke 139; `coder_repair_synthetic` excluded. It retains v1 first-edit/stall-point
  extraction, has unique message hashes, and raw rendered prefixes <=8,192 tokens (p50 2,001;
  max 5,331), compatible with the 2,560-token loader clamp. Source-cap builder tests: 7/7.
- **Reward phase 2 is GO:** replace the parse-only reward's 1.0 tier with a Docker
  apply/F2P-test-verified tier. CPU-only harness work starts on v2; any training still requires
  the normal smoke gate, frozen-base constraint, GPU1-only launch, and artifact verification.
- **Fixture join and cache preflight (2026-07-14):**
  `data/rlvr_swe_decision_v2_fixtures.jsonl` joins **1,330/1,895 (70.2%)** rows to an exact
  image plus F2P list: SWE-smith 691/700, Kwai 500/500, and all 139 smoke prefixes. The 565
  unresolved rows (OpenSWE 300, oracle 256, nine SWE-smith) stay in the dataset with
  `unverified=true` and may receive the existing parse/edit reward through **0.6**, never an
  execution tier. `data/rlvr_swe_decision_v2_imagecov.json` separately reports local images
  (all smoke rows are present) from fixture-backed-but-not-local SWE-smith rows.
  Oracle mapping was checked by normalizing `swe-edit-trace-<owner>__<repo>-<number>-000` to
  `<owner>__<repo>-<number>`: it joins **0/256** to the 59,136-row SWE-smith pool, 300-row
  SWE-Lite, and 2,294-row SWE-bench set, so there is no safe suffix-only upgrade.
- **State cache:** `phaseE_rl/reconstruct_swe_decision_state.py` creates only labelled
  `rlvr-state:<prompt_hash>` images. It executes prefix commands with `bash -c` in a
  network-isolated container, accepts a strict read-only replay subset, removes containers by
  exact id, LRU-evicts only its own labelled images, and refuses work below the 40-GiB Docker
  floor. The local-state dry run is **10/10 created** (`runs/rlvr_state_cache/dry10.jsonl`),
  with 22 replayed commands (19 rc=0, 3 expected nonzero inspection commands), followed by a
  confirmed cache hit (`runs/rlvr_state_cache/hit1.jsonl`); loopback remained 92 GiB free.
  This proves reconstruction/cache mechanics only. The **full** 10-prompt gate still needs the
  coordinator-owned completion apply + F2P verifier to observe 0.8/1.0 tiers; do not call the
  dry run a reward pass or prebuild the whole cache without the disk guard.

All in `phaseE_rl/grpo_swe_edit_decision.py` (committed, 4ea4bea + later edits synced to box):
1. `Gemma4ClippableLinear` is un-peft-able → SFT adapter (v8cp30) is MANUALLY merged into
   inner `nn.Linear` weights (fp32 B@A, assert >100 matrices).
2. Text-class direct load leaves weights RANDOM (MM checkpoint keys don't map) → load
   `Gemma4ForConditionalGeneration`, GRAFT `.model.language_model` + `.lm_head` into an
   empty `Gemma4ForCausalLM` shell (meta-assert + sanity-gen tripwires in script).
3. Adapter keys are MM-layout → `model.language_model.` → `model.` remap in merge.
4. trl ignores `model.generation_config` (grpo_trainer.py:1418) → stops via
   `GRPOConfig.generation_kwargs.eos_token_id=[1, 49, 106]` (49=`<tool_call|>`, 106=`<turn|>`).
   Without this, completions pin at the cap and ALL rewards are 0.
5. LoRA targets computed at runtime (inner `.linear` paths).
6. OOM fixes: prompt cap 2560, completion 1024, `per_device_train_batch_size=2` +
   `gradient_accumulation_steps=2` (GRPO group of 4 preserved), gradient_checkpointing,
   `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`.

## 3. Ops gotchas (cost us three restarts)
- Killing the launcher PID leaves a trl CHILD holding ~90GB. Kill BOTH (ps for
  grpo_swe_edit, kill each PID), then LOOP until `nvidia-smi` shows GPU1 < 1GB before
  any relaunch.
- kill-then-launch in one ssh command drops the connection (exit 255) — separate them.
- `.venv-rl` is the ONLY venv where trl imports (unsloth-free by design).
- Never `pkill -f`. GPU0/prod vLLM off-limits; ports 8000/8101/8103/8104 stay green.

## 4. Completed comparison and next authorized sequence
1. **Completed:** Dr. GRPO s2 and the matched standard-GRPO run both reached checkpoint-100;
   both three-way serving merges and both hard30 + 30-case temp-0.7 evaluations are complete.
   Do not restart either run or alter their data/configuration; they are the controlled pair.
2. **Completed comparison:** (a) training curves — reward_fn/mean trajectory, completion
   mean_length trajectory, frac_reward_zero_std; (b) behavioral evaluation for both final
   checkpoints:
   - 3-way merge for serving: base is FROZEN — merge (bf16 base + v8cp30 + RL LoRA) into a
     NEW dir (e.g. /media/ironbcc/CrucialX10/models/merged/swe_<variant>_v1) using the same
     manual-merge approach as the script (RL LoRA keys are standard peft over the grafted
     text model; v8cp30 keys need the language_model remap). Disk check first (needs ~62GB
     per merge; delete after eval).
   - Serve merged model on 8012 (v2 thinkopen template, --reasoning-parser gemma4
     --tool-call-parser gemma4, GPU1, prod green), hard30 floor (≥ same-day cp20 anchor −2),
     then 30-case SWE-Lite: NAME=<variant> PORT=8012 SLICE=0:30 WORKERS=8 TEMPERATURE=0.7
     SCORE=1 bash phaseH_eval/smoke_single.sh.
   - Baselines to beat: v8cp30 = 11/30 patches, 6/30 resolved, 14/30 edit-reach;
     v7s1cp5 = 13/30 patches, 7/30 resolved. Gate ≥18/30 non-empty, <10% format.
3. **Promotion decision:** standard GRPO is promoted by the user despite its exact 10.0%
   format-boundary result; it is the current Python policy. Do not re-run the comparison.
4. **Next:** implement and test the CPU-side verified reward on `rlvr_swe_decision_v2.jsonl`:
   apply the model patch in the instance Docker image, run the canonical F2P test tier, and use
   that verification—not a parse-only filename match—for the reward's top tier. Keep failed or
   infra-flake classifications explicit; no training launch is implied by this authorization.

## 5. Parallel C++ image-build lane (CPU/disk only)

- The staged C++ pool is `data/mswe_cpp_prs/` (257 instances in eight repositories). Build
  Docker images sequentially by repository family; record each concrete CMake/make build/test
  command with its image artifact.
- Docker loopback begins with 94 GB free and must retain **>=40 GB free**. Check `df -h
  /var/lib/docker` before each family, use the disk-watchdog pattern while building, and prune
  dangling layers only between completed families. Stop and report immediately below the floor.
  No GPU, training, serving restart, or production-port change is authorized.
- Rust adapter v2 remains queued until after the Python verified-reward round.

## 6. If verified reward doesn't move the gate — ranked next levers (proposals, not authorizations)
1. num_gen 8 (needs per_device 2 × accum 4) — richer group signal.
2. More steps (300+) once curves prove stable.
3. Harness-level self-retry on empty patch (cheap, orthogonal, was optioned earlier).
4. Multi-turn RLVR (full rollouts with docker reward) — the heavy endgame.

## 7. Context docs
- `phaseH_eval/SESSION_STATE.md` (18)-(25): v7/v8 arc, plateau evidence, RL loop milestones.
- `phaseD_sft/V6_49K_RETROSPECTIVE.md`: the 49k attention saga.
- Task list #9 (RLVR lane) is yours; #8 (Rust baseline) fires only when GPU is idle and
  user re-engages; #4/#6 parked.
