# v2.11r3 Behavior Poststage Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Continue the completed v2.11r3 reasoning LoRA through recovery SFT and coverage-balanced KTO on GPU1, then evaluate only the final composed model against the frozen v2.10 full-300 control.

**Architecture:** Keep the current r3 completion as an immutable intermediate stage. A new GPU1-only launcher validates the production recovery/KTO inputs, publishes immutable phase markers, trains recovery and KTO adapters, and produces the final merged model. A distinct behavior-poststage provenance validator rebuilds the complete r3-to-final lineage; the existing full-300 evaluator dispatches to it and the goal audit requires the learned empty/loop mitigation evidence.

**Tech Stack:** Bash, Python 3.11, pytest, safetensors, Hugging Face/TRL LoRA training, systemd user units, immutable JSON/SHA-256 artifact bindings.

## Global Constraints

- GPU1 is the only permitted GPU; the new lane must never query, wait on, signal, or control GPU0.
- Use exact numeric child PIDs only; no `pkill`, `pgrep`, broad kill, or host-wide ownership inference.
- Preserve the active 79-step r3 trainer and its existing adapter, journal, watchdog, dataset, and context-audit identities.
- Require the production recovery-138 and behavior-KTO-606 hashes from `phaseH_eval/v2p11_posttrain_contract.py`; no regenerated or substituted data.
- Recovery SFT uses 105 optimizer steps and `max_seq=32768` from the r3 adapter.
- Full KTO uses exactly 25 optimizer steps and coverage counts `25/8/8/9`; a one-step canary must pass first.
- Evaluate exactly one v2.11 candidate: the final recovery-plus-KTO merged model.
- Keep v2.10 fixed at the verified complete 157/300 control; do not rerun base, v2, or v2.10.
- Do not stage or modify unrelated dirty-worktree files.

---

### Task 1: Preserve the fail-closed poststage core

**Files:**
- Add: `phaseD_sft/build_v2p11_recovery_curriculum.py`
- Add: `phaseD_sft/tests/test_build_v2p11_recovery_curriculum.py`
- Add: `phaseH_eval/v2p11_posttrain_contract.py`
- Add: `phaseH_eval/tests/test_v2p11_posttrain_contract.py`
- Add: `phaseH_eval/v2p11_poststage_phase_marker.py`
- Add: `phaseH_eval/tests/test_v2p11_poststage_phase_marker.py`

**Interfaces:**
- Consumes: existing production recovery/KTO artifacts on the remote host.
- Produces: `validate_posttrain_inputs(...)` and immutable `publish_or_verify(...)` contracts used by the launcher and provenance validator.

- [ ] **Step 1: Run the existing focused tests as a baseline**

Run:

```bash
.venv-eval/bin/python -m pytest -q \
  phaseD_sft/tests/test_build_v2p11_recovery_curriculum.py \
  phaseH_eval/tests/test_v2p11_posttrain_contract.py \
  phaseH_eval/tests/test_v2p11_poststage_phase_marker.py
```

Expected: all tests pass and prove mutation/test admission, immutable production hashes, and tamper rejection.

- [ ] **Step 2: Verify the dependency closure**

Run:

```bash
PYTHONPATH=$PWD .venv-eval/bin/python - <<'PY'
from phaseD_sft.build_v2p11_recovery_curriculum import PORTABLE_COMMAND_CLASSES
from phaseH_eval.v2p11_poststage_phase_marker import publish_or_verify
from phaseH_eval.v2p11_posttrain_contract import PRODUCTION_RECOVERY_ROWS
assert PORTABLE_COMMAND_CLASSES
assert callable(publish_or_verify)
assert PRODUCTION_RECOVERY_ROWS == 138
PY
```

Expected: exit 0 without importing the legacy GPU0 scripts or the optional dataset-regeneration closure.

- [ ] **Step 3: Commit only the six core files**

```bash
git add \
  phaseD_sft/build_v2p11_recovery_curriculum.py \
  phaseD_sft/tests/test_build_v2p11_recovery_curriculum.py \
  phaseH_eval/v2p11_posttrain_contract.py \
  phaseH_eval/tests/test_v2p11_posttrain_contract.py \
  phaseH_eval/v2p11_poststage_phase_marker.py \
  phaseH_eval/tests/test_v2p11_poststage_phase_marker.py
git commit -m "feat: preserve v2.11 behavior poststage contracts"
```

### Task 2: Add the GPU1-only r3 behavior launcher

**Files:**
- Create: `phaseH_eval/train_v2p11r3_behavior_gpu1.sh`
- Create: `phaseH_eval/tests/test_train_v2p11r3_behavior_gpu1.py`

**Interfaces:**
- Consumes: `runs/v2p11r3_v2p10init_fable_reasoned_training_completion.json`, the r3 adapter, production recovery/KTO artifacts, and the phase-marker helper.
- Produces: uniquely named recovery/KTO adapters, merged models, phase markers, and `runs/v2p11r3_behavior_posttrain_complete.json`.

- [ ] **Step 1: Write a failing launcher contract test**

The test must execute `bash -n`, run `PREFLIGHT_ONLY=1` against controlled fake paths where practical, and assert observable failures for a non-r3 marker or a non-GPU1 index. It must also require unique r3 behavior artifact names and reject source containing `CUDA_VISIBLE_DEVICES=0`, `GPU_INDEX=0`, `pkill`, or `pgrep`.

- [ ] **Step 2: Run the launcher test and verify RED**

Run:

```bash
.venv-eval/bin/python -m pytest -q phaseH_eval/tests/test_train_v2p11r3_behavior_gpu1.py
```

Expected: FAIL because `phaseH_eval/train_v2p11r3_behavior_gpu1.sh` does not exist.

- [ ] **Step 3: Implement the launcher**

Create an r3-specific launcher with these literal defaults:

```bash
GPU_INDEX="${GPU_INDEX:-1}"
STAGE_A_ADAPTER="${STAGE_A_ADAPTER:-adapters/teacher_sft_v2p11r3_v2p10init_fable_reasoned_bf16}"
STAGE_A_MARKER="${STAGE_A_MARKER:-runs/v2p11r3_v2p10init_fable_reasoned_training_completion.json}"
RECOVERY_ADAPTER="${RECOVERY_ADAPTER:-adapters/teacher_sft_v2p11r3_behavior_recovery_bf16}"
KTO_ADAPTER="${KTO_ADAPTER:-adapters/teacher_sft_v2p11r3_behavior_kto}"
FINAL_MERGED="${FINAL_MERGED:-/media/ironbcc/CrucialX10/models/merged/teacher_sft_v2p11r3_behavior_full}"
COMPLETION_MARKER="${COMPLETION_MARKER:-runs/v2p11r3_behavior_posttrain_complete.json}"
```

Validate `GPU_INDEX == 1`, the r3 marker fields `artifact_type=v2p11r3_training_completion`, `status=complete`, `optimizer_steps=max_steps=79`, and the nested `adapter.sha256`. Bind every training child with `CUDA_VISIBLE_DEVICES="$GPU_INDEX"`. Preserve restart-safe recovery, 12-GiB RAM watchdog, 105-step recovery, one-step KTO canary, coverage-balanced 25-step KTO, bounded merges, and immutable input/output markers.

- [ ] **Step 4: Run the launcher test and verify GREEN**

Run:

```bash
bash -n phaseH_eval/train_v2p11r3_behavior_gpu1.sh
.venv-eval/bin/python -m pytest -q phaseH_eval/tests/test_train_v2p11r3_behavior_gpu1.py
```

Expected: both commands pass.

- [ ] **Step 5: Commit the launcher and its test**

```bash
git add phaseH_eval/train_v2p11r3_behavior_gpu1.sh \
  phaseH_eval/tests/test_train_v2p11r3_behavior_gpu1.py
git commit -m "feat: add v2.11r3 GPU1 behavior poststage"
```

### Task 3: Bind the r3 behavior lineage in distinct provenance

**Files:**
- Create: `phaseH_eval/v2p11r3_behavior_completion_provenance.py`
- Create: `phaseH_eval/tests/test_v2p11r3_behavior_completion_provenance.py`
- Modify: `phaseH_eval/compare_v2p11_full300.py`
- Modify: `phaseH_eval/tests/test_compare_v2p11_full300.py`

**Interfaces:**
- Consumes: `_validate_dataset`, `_validate_training`, and final-model validation from `v2p11r3_completion_provenance.py`; `validate_posttrain_inputs(...)`; immutable phase markers; portability gate; v2.10 control.
- Produces: `artifact_type=v2p11r3_behavior_completion_provenance` and a dispatchable `validate_completion_provenance(...)` result containing `dataset`, `training`, `behavior_poststage`, `final_model`, and `portability`.

- [ ] **Step 1: Write failing lineage, tamper, and dispatch tests**

Use a real temporary lineage fixture. Require the stage marker to bind the nested r3 adapter, the recovery marker to bind that stage marker, the KTO evidence to report 606 source rows and coverage `25/8/8/9`, and the final marker to bind the final model. Mutating any bound marker, KTO coverage count, optimizer step, or final model shard must make validation fail.

- [ ] **Step 2: Run the provenance tests and verify RED**

Run:

```bash
.venv-eval/bin/python -m pytest -q \
  phaseH_eval/tests/test_v2p11r3_behavior_completion_provenance.py \
  phaseH_eval/tests/test_compare_v2p11_full300.py::test_completion_provenance_dispatches_r3_behavior_poststage_contract
```

Expected: FAIL because the behavior provenance module and dispatch branch do not exist.

- [ ] **Step 3: Implement the behavior provenance validator**

The rebuilt report must use this stable shape:

```python
{
    "artifact_type": "v2p11r3_behavior_completion_provenance",
    "lineage": {
        "kind": "r3_reasoning_then_recovery_sft_then_behavior_kto",
        "reasoned_fable_contract": True,
        "transferable_empty_loop_correction": True,
    },
    "behavior_poststage": {
        "recovery_rows": 138,
        "recovery_optimizer_steps": 105,
        "behavior_rows": 606,
        "kto_optimizer_steps": 25,
        "coverage_counts": {
            "desirable_correct_patch": 25,
            "empty_terminal": 8,
            "repeated_read_loop": 8,
            "wrong_nonempty_replay": 9,
        },
    },
}
```

Rebuild all bindings during validation. Add a dispatch branch in `compare_v2p11_full300.validate_completion_provenance` before the direct-r3 branch.

- [ ] **Step 4: Run the provenance tests and verify GREEN**

Run the two commands from Step 2 plus:

```bash
.venv-eval/bin/python -m pytest -q \
  phaseH_eval/tests/test_v2p11r3_completion_provenance.py \
  phaseH_eval/tests/test_v2p11_posttrain_lineage.py
```

Expected: behavior, direct-r3, and generic posttrain provenance tests all pass.

- [ ] **Step 5: Commit provenance and dispatch**

```bash
git add \
  phaseH_eval/v2p11r3_behavior_completion_provenance.py \
  phaseH_eval/tests/test_v2p11r3_behavior_completion_provenance.py \
  phaseH_eval/compare_v2p11_full300.py \
  phaseH_eval/tests/test_compare_v2p11_full300.py
git commit -m "feat: bind v2.11r3 behavior lineage"
```

### Task 4: Run only the final poststage model and audit learned mitigations

**Files:**
- Modify: `phaseH_eval/run_v2p11r3_posttrain_chain.sh`
- Modify: `phaseH_eval/tests/test_run_v2p11r3_posttrain_chain.py`
- Modify: `phaseH_eval/v2p11r3_goal_completion_audit.py`
- Modify: `phaseH_eval/tests/test_v2p11r3_goal_completion_audit.py`

**Interfaces:**
- Consumes: completed r3 evidence, the GPU1 behavior launcher, final poststage model/marker, behavior provenance, portability gate, and matched full-300 evaluator.
- Produces: one final-model portability run, one full-300 run, and a goal audit that proves the trained recovery/KTO mitigations.

- [ ] **Step 1: Write failing chain and goal-audit tests**

The chain test must require poststage invocation before portability, `LINEAGE_MODE=posttrain`, the final behavior model, the behavior provenance publisher, exactly one call to `eval_v2p11_full300_after_merge.sh`, and no intermediate r3 merge/eval. The goal-audit test must reject missing or changed recovery/KTO counts and coverage.

- [ ] **Step 2: Run the tests and verify RED**

Run:

```bash
.venv-eval/bin/python -m pytest -q \
  phaseH_eval/tests/test_run_v2p11r3_posttrain_chain.py \
  phaseH_eval/tests/test_v2p11r3_goal_completion_audit.py
```

Expected: FAIL because the current chain directly merges/evaluates r3 and the goal audit checks only reasoning SFT.

- [ ] **Step 3: Modify the chain and goal audit**

After immutable r3 capture/audit, invoke `train_v2p11r3_behavior_gpu1.sh`. Use only `/media/ironbcc/CrucialX10/models/merged/teacher_sft_v2p11r3_behavior_full`, its final merge audit, posttrain marker, and behavior provenance for portability and evaluation. Set `LINEAGE_MODE=posttrain`, `POSTTRAIN_MARKER` to the r3 behavior marker, `GPU_INDEX=1`, and `PORT=8013`. Extend `_validate_reasoned_lineage` or add `_validate_behavior_poststage` to require the exact production counts and coverage.

- [ ] **Step 4: Run the tests and verify GREEN**

Run:

```bash
bash -n phaseH_eval/run_v2p11r3_posttrain_chain.sh
.venv-eval/bin/python -m pytest -q \
  phaseH_eval/tests/test_run_v2p11r3_posttrain_chain.py \
  phaseH_eval/tests/test_v2p11r3_goal_completion_audit.py \
  phaseH_eval/tests/test_eval_v2p11_full300_after_merge.py
```

Expected: all pass and source contains exactly one final evaluator invocation.

- [ ] **Step 5: Commit final chain integration**

```bash
git add \
  phaseH_eval/run_v2p11r3_posttrain_chain.sh \
  phaseH_eval/tests/test_run_v2p11r3_posttrain_chain.py \
  phaseH_eval/v2p11r3_goal_completion_audit.py \
  phaseH_eval/tests/test_v2p11r3_goal_completion_audit.py
git commit -m "feat: evaluate final v2.11r3 behavior model"
```

### Task 5: Verify, sync, and restore the bounded continuation

**Files:**
- Verify: all files committed by Tasks 1-4
- Remote create: a CPU-only waiter unit invoking `phaseH_eval/run_v2p11r3_posttrain_chain.sh`

**Interfaces:**
- Consumes: the completed code changes and current remote r3 unit/artifacts.
- Produces: a checksum-matched remote checkout and exactly one bounded continuation from r3 completion through final full-300.

- [ ] **Step 1: Run focused and regression verification**

Run:

```bash
bash -n \
  phaseH_eval/train_v2p11r3_behavior_gpu1.sh \
  phaseH_eval/run_v2p11r3_posttrain_chain.sh
.venv-eval/bin/python -m pytest -q \
  phaseD_sft/tests/test_build_v2p11_recovery_curriculum.py \
  phaseH_eval/tests/test_v2p11_posttrain_contract.py \
  phaseH_eval/tests/test_v2p11_poststage_phase_marker.py \
  phaseH_eval/tests/test_train_v2p11r3_behavior_gpu1.py \
  phaseH_eval/tests/test_v2p11r3_behavior_completion_provenance.py \
  phaseH_eval/tests/test_v2p11r3_completion_provenance.py \
  phaseH_eval/tests/test_v2p11_posttrain_lineage.py \
  phaseH_eval/tests/test_run_v2p11r3_posttrain_chain.py \
  phaseH_eval/tests/test_v2p11r3_goal_completion_audit.py \
  phaseH_eval/tests/test_eval_v2p11_full300_after_merge.py \
  phaseH_eval/tests/test_compare_v2p11_full300.py
```

Expected: zero failures.

- [ ] **Step 2: Audit forbidden operations and commit scope**

Run:

```bash
rg -n 'CUDA_VISIBLE_DEVICES=0|GPU_INDEX=0|pkill|pgrep|nvidia-smi .*-i 0' \
  phaseH_eval/train_v2p11r3_behavior_gpu1.sh \
  phaseH_eval/run_v2p11r3_posttrain_chain.sh
git diff --check
git status --short
```

Expected: no forbidden-operation matches, no whitespace errors, and unrelated dirty files remain unstaged.

- [ ] **Step 3: Verify local/remote checksums and remote production inputs**

Sync only the scoped committed files. Compare SHA-256 for every launcher, validator, and test. Re-run `v2p11_posttrain_contract.py --require-production-identity` remotely and require 138 recovery rows, 606 behavior rows, 300 Lite IDs, zero overlap, and the pinned hashes.

- [ ] **Step 4: Restore a CPU-only waiter**

The waiter must wait for `v2p11r3-reasoned-train-gpu1.service` to reach a terminal successful state, then invoke the checksum-matched `run_v2p11r3_posttrain_chain.sh`. It must not query GPU0 and must not launch a direct-r3 evaluator.

- [ ] **Step 5: Verify live ownership and next ETA boundary**

Record exact unit, wrapper PID, trainer PID, optimizer step, GPU1 process identity, and artifact paths. Derive one PDT ETA covering remaining r3 training, recovery SFT, KTO, merges, portability, and full-300. Leave guards running and recheck only at the ETA boundary or on a real failure/resource exception.

### Task 6: Publish the matched verdict

**Files:**
- Verify: `runs/v2p11r3_behavior_vs_v2p10_full300.json`
- Verify: `runs/v2p11r3_behavior_full300_composite.json`
- Verify: `runs/v2p11r3_behavior_full300_official_score_binding.json`
- Verify: `runs/v2p11r3_behavior_goal_completion_audit.json`

**Interfaces:**
- Consumes: complete v2.10/v2.11 full-300 artifacts and behavior-poststage provenance.
- Produces: the final trustworthy score comparison and failure analysis.

- [ ] **Step 1: Revalidate all completion artifacts from current bytes**

Require 300 unique predictions, 300 trajectory files, zero pull failures, complete official score bindings, and a provenance model fingerprint matching the served final model.

- [ ] **Step 2: Verify promotion requirements**

Require v2.11 resolved count greater than 157, no new final empty IDs relative to v2.10, healthy first-pass empty/loop metrics, and failure analysis covering every non-jointly-resolved task.

- [ ] **Step 3: Publish the final report**

Report first-pass and corrected scores, resolved delta, empty/loop deltas, recovery/KTO coverage, Fable admission counts, and remaining failure categories. Keep the goal active if any promotion requirement fails.
