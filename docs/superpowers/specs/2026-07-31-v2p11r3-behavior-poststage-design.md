# v2.11r3 Transferable Empty/Loop Correction Design

## Goal

Produce the final v2.11 candidate by continuing the active reasoned-Fable r3
adapter with replay-proven recovery SFT and explicit empty/loop KTO penalties,
then evaluate only that final candidate against the frozen v2.10 full-300
control.

## Decision

The active r3 run remains useful as an intermediate reasoning-SFT stage. It
must not be promoted directly: its only new behavior signal is deterministic
reasoning inserted before admitted Fable tool calls, while its empty-only
reruns are controller-time and do not transfer to another serving harness.

The final candidate therefore uses this sequence:

1. Validate and freeze the completed r3 adapter and its 79-step evidence.
2. Continue r3 with the admitted 138-row portable recovery curriculum for
   105 optimizer steps at the full 32,768-token context limit.
3. Merge that recovery adapter into the frozen Gemma-4 base with the bounded
   streaming merger.
4. Run a one-step KTO canary against the merged recovery model.
5. Run 25 KTO optimizer steps using the coverage-balanced 50-row selection:
   25 desirable patches, eight empty-terminal negatives, eight repeated-read
   loop negatives, and nine wrong-nonempty-edit negatives.
6. Merge the KTO adapter into the recovery model, run portability, and execute
   the same full Lite300 and matched empty-only correction as v2.10.

This is one v2.11 candidate lineage. `r3` is an intermediate stage label, not
an additional comparison checkpoint.

## Inputs and gates

- Reasoning SFT: `data/teacher_train_mix_v2p11_fable1262_reasoned_v1`, already
  sealed at 1,262 rows with zero Lite overlap and 32,768-token admission.
- Recovery SFT: `data/v2p11_portable_recovery138_targeted`, exactly 138 rows,
  each with a portable source mutation and focused passing-test target.
- Behavior KTO: `data/v2p11_behavior_kto_v2.jsonl`, exactly 606 rows with
  replay-bound desirable and undesirable outcomes.
- Combined exclusions: `data/swe_all_eval_exclusions_v2.json`, covering all
  300 Lite IDs and the broader verified-evaluation exclusion set.
- Production identity is fixed by the existing posttrain contract hashes; no
  regenerated or substituted dataset is admissible.

## Runtime ownership

- GPU1 is the only permitted GPU. GPU0 must never be queried, waited on, or
  controlled by the new lane.
- All trainers bind `CUDA_VISIBLE_DEVICES=1`; service and log names include
  `gpu1` and the r3 behavior tag.
- Exact-PID RAM watchdogs retain the 12-GiB floor. No `pkill`, `pgrep`, broad
  kill, or host-wide process ownership inference is permitted.
- The currently active r3 trainer is not restarted. Its old CPU posttrain
  waiter stays disabled so direct-r3 evaluation cannot launch.

## Provenance and publication

Immutable phase markers bind the r3 completion, recovery inputs/output,
recovery merge, KTO inputs/output, final merge, and final model shards. The r3
completion provenance is extended with the poststage evidence and must rebuild
identically during final validation.

Promotion requires all of the following:

- the final model, not the intermediate r3 merge, passes portability;
- all 300 predictions and trajectories exist with zero pull failures;
- both first-pass and corrected scores are compared to v2.10's bound 157/300;
- first-pass empty/loop behavior remains separately reported from corrected
  composite results;
- the corrected score beats v2.10, no new empty IDs are introduced, and the
  goal audit covers every non-jointly-resolved task.

## Failure behavior

Every phase is restart-safe and fail-closed. Existing output without its bound
input marker is rejected. A failed KTO canary blocks the full KTO stage. A
losing or behavior-regressed full300 result is preserved as evidence but is not
published as a completed goal.

## Rejected alternatives

- Directly evaluating r3: does not include explicit transferable empty/loop
  penalties.
- Restarting reasoning SFT with all rows blended: discards a healthy active run
  and weakens separation between supervised recovery and negative preference
  evidence.
- Evaluating r3 before poststage: consumes a full 300-task run for a candidate
  already known to miss the requested training objective.
