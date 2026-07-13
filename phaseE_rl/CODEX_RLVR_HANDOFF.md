# CODEX HANDOFF — RLVR lane (Dr. GRPO vs GRPO comparison + eval)

Coordinator is out of token budget; you own this lane end-to-end now. User-authorized scope:
let Dr. GRPO finish → schedule the standard-GRPO comparison run → eval + compare both.
SESSION_STATE.md entries (24)-(25) have the full history.

## 0. Why this lane exists
SFT plateaued: v7/v8 mixtures both peaked at ~11-13/30 non-empty patches on the 30-case
SWE-Lite temp-0.7 smoke (gate ≥18/30), resolved 6-7/30, while hard30 semantics hit 25/30
(best-ever). The model READS but doesn't EDIT (perseveration). RLVR optimizes the edit
decision directly. User picked "why not DrGRPO" (arXiv:2503.20783) mid-run — its two fixes
(no length normalization: stops subsidizing rambling 0-reward completions; no σ-scaling:
stable advantages on our discrete rewards) map onto our observed pathologies.

## 1. Running now
- **Dr. GRPO s2**: `phaseE_rl/grpo_swe_edit_decision.py`, out `adapters/swe_drgrpo_v1_s2`,
  log `/tmp/drgrpo_s2.log`, 100 steps, save every 25, GPU1. Expect ~4h.
  Healthy signature: `completions/mean_length` VARIED (46-708, not pinned at cap),
  `reward_fn/mean` climbing off the 0.2 read-floor toward 0.6+.
- **Reward** (shaped, parse-only phase 1): 0.0 no tool call / 0.2 read / 0.6 edit /
  1.0 edit touching a file named in context. Regexes in the script; 8/8 unit tests
  (`phaseE_rl/tests/test_grpo_swe_reward.py`).
- **Dataset**: `data/rlvr_swe_decision_v1.jsonl` (2,000 stall-context prefixes ≤8k tokens;
  script also caps prompts at --max-prompt-tokens 2560 at load). Your v2 rebalanced build
  (per-source caps, no coder_repair) was directed earlier — finish it when idle; use for
  any run AFTER the comparison (don't change dataset mid-comparison).

## 2. Integration fixes already made — do NOT re-hit these
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

## 4. YOUR SEQUENCE (user-authorized)
1. **Watch Dr. GRPO s2 to completion** (`done ->` in /tmp/drgrpo_s2.log; checkpoints
   25/50/75/100 in `adapters/swe_drgrpo_v1_s2`). If it OOMs mid-run: latest checkpoint
   survives — note step count, don't relaunch, proceed with what saved.
2. **Schedule standard GRPO comparison** (user explicitly authorized): add a
   `--loss-type` CLI arg to the script (choices grpo|dr_grpo, plumb to
   `GRPOConfig(loss_type=...)`; for plain grpo also set `scale_rewards=True` i.e. drop
   the scale_rewards=False override). Then run IDENTICAL config except loss type:
   out `adapters/swe_grpo_v1_s2`, log `/tmp/grpo_cmp_s2.log`, same data/steps/caps/seed.
3. **Compare**: (a) training curves — reward_fn/mean trajectory, completion mean_length
   trajectory, frac_reward_zero_std (extract from both logs, table in the report);
   (b) behavioral eval BOTH final checkpoints:
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
4. **Report**: comparison table (GRPO vs Dr.GRPO: curves + smoke numbers) to the
   coordinator/user. NO further training or data rebuilds beyond this without approval.

## 5. If RL doesn't move the gate — ranked next levers (proposals, not authorizations)
1. Reward phase 2: docker apply-check for the 1.0 tier (patch actually applies) — kills
   reward-hacking headroom, slower loop.
2. v2 decision dataset (rebalanced, more smoke-derived stall contexts).
3. num_gen 8 (needs per_device 2 × accum 4) — richer group signal.
4. More steps (300+) once curves prove stable.
5. Harness-level self-retry on empty patch (cheap, orthogonal, was optioned earlier).
6. Multi-turn RLVR (full rollouts with docker reward) — the heavy endgame.

## 6. Context docs
- `phaseH_eval/SESSION_STATE.md` (18)-(25): v7/v8 arc, plateau evidence, RL loop milestones.
- `phaseD_sft/V6_49K_RETROSPECTIVE.md`: the 49k attention saga.
- Task list #9 (RLVR lane) is yours; #8 (Rust baseline) fires only when GPU is idle and
  user re-engages; #4/#6 parked.
