# v2.11r3 Promotion Proof Gates Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the v2.11r3 final verdict prove canonical v2.10 initialization and reject any regression in wrong non-empty outcomes.

**Architecture:** Reuse the existing immutable `v2p10_training_lineage` contract as the sole authority for the r3 init adapter, and bind that contract into the nested reasoning-to-behavior provenance. Derive wrong non-empty outcomes from the already verified full-300 partition as every unresolved ID that is neither empty nor resolved, including classified model failures, then require a non-positive paired count delta for trustworthy promotion.

**Tech Stack:** Python 3, pytest, Bash, immutable JSON bindings, SWE-bench Lite full-300 artifacts.

## Global Constraints

- Scope is only v2.10 and v2.11r3; do not launch base, v2, or r2 evaluation.
- GPU0 is unavailable and must not be queried, used, or controlled.
- GPU1 child commands remain directly bound and process cleanup remains exact-PID only.
- Preserve every unrelated tracked and untracked worktree change.
- Do not alter the running 79-step r3 training job or its immutable evidence.
- Final success still requires the same complete Lite300 population, a v2.11 score above the frozen v2.10 score, no empty regression, no behavior regression, and the final goal audit.

---

### Task 1: Bind r3 initialization to canonical v2.10 lineage

**Files:**
- Modify: `phaseH_eval/v2p11r3_completion_provenance.py`
- Modify: `phaseH_eval/v2p11r3_behavior_completion_provenance.py`
- Modify: `phaseH_eval/run_v2p11r3_posttrain_chain.sh`
- Test: `phaseH_eval/tests/test_v2p11r3_completion_provenance.py`
- Test: `phaseH_eval/tests/test_v2p11r3_behavior_completion_provenance.py`
- Test: `phaseH_eval/tests/test_run_v2p11r3_posttrain_chain.py`

**Interfaces:**
- Consumes: `validate_v2p10_training_lineage_contract(contract_path, composite_path=...)` and its `artifacts.adapter` binding.
- Produces: `_validate_v2p10_init_lineage(lineage_path, v2p10_composite_path, training) -> dict`, a `v2p10_training_lineage` binding in r3 provenance, and CLI option `--v2p10-lineage`.

- [x] **Step 1: Write the failing substituted-adapter test**

```python
with pytest.raises(ValueError, match="canonical v2.10 init adapter"):
    _validate_v2p10_init_lineage(
        lineage_path=lineage,
        v2p10_composite_path=composite,
        training={"init_adapter": binding(substituted)},
    )
```

- [x] **Step 2: Run the test and verify RED**

Run: `PYTHONPATH=$PWD .venv/bin/python -m pytest -q phaseH_eval/tests/test_v2p11r3_completion_provenance.py::test_reasoned_provenance_rejects_noncanonical_v2p10_init_adapter`

Expected: FAIL because `_validate_v2p10_init_lineage` is absent.

- [x] **Step 3: Implement canonical lineage validation**

Import the canonical validator, rebuild the lineage against `runs/v2p10_full300_composite.json`, and require exact binding equality:

```python
lineage = validate_v2p10_training_lineage_contract(
    lineage_path,
    composite_path=v2p10_composite_path,
)
if lineage["artifacts"]["adapter"] != training["init_adapter"]:
    raise ValueError("r3 training does not use the canonical v2.10 init adapter")
```

Store `_binding(lineage_path)` and the validated v2.10 training summary in the r3 report. Thread `v2p10_lineage_path` through publish, rebuild validation, the behavior wrapper, and both CLIs.

- [x] **Step 4: Make the posttrain chain create or revalidate the lineage before poststage**

Use the existing v2.10 marker, run manifest, dataset manifest/train JSONL, adapter, merge audit/model, full300 composite, and Stage-A manifest arguments already used by `run_v2p11_completion_chain.sh`. Pass `--v2p10-lineage "$V2P10_LINEAGE"` to behavior provenance publication.

- [x] **Step 5: Verify GREEN and commit**

Run the three focused provenance/chain test files, `bash -n phaseH_eval/run_v2p11r3_posttrain_chain.sh`, Python compilation, and `git diff --check`.

Commit only the Task 1 implementation, tests, and this plan with message `fix: bind v2.11r3 to canonical v2.10`.

### Task 2: Gate promotion on wrong non-empty nonregression

**Files:**
- Modify: `phaseH_eval/compare_v2p11_full300.py`
- Modify: `phaseH_eval/v2p11r2_goal_completion_audit.py`
- Modify: `phaseH_eval/v2p11r3_goal_completion_audit.py`
- Test: `phaseH_eval/tests/test_compare_v2p11_full300.py`
- Test: `phaseH_eval/tests/test_v2p11r2_goal_completion_audit.py`
- Test: `phaseH_eval/tests/test_v2p11r3_goal_completion_audit.py`

**Interfaces:**
- Consumes: the verified full ID set, resolved sets, empty sets, and official model-failure sets already calculated by `compare_full300`.
- Produces: `wrong_nonempty` with `v2p10`, `v2p11`, `introduced`, `eliminated`, `delta`, and `no_regression`; verdict fields `wrong_nonempty_delta` and `wrong_nonempty_no_regression`.

- [x] **Step 1: Write and run failing outcome tests**

The tests require a candidate that resolves one additional task but turns two control empties into wrong non-empty outcomes to fail trustworthy promotion. They also require the final r3 goal audit to record the gate.

Expected RED: the report lacks `wrong_nonempty`, the trustworthy validator accepts the regression, and the goal audit omits the requirement.

- [x] **Step 2: Derive immutable paired wrong non-empty evidence**

```python
control_wrong = full_set - v2p10_resolved - v2p10_empty
candidate_wrong = full_set - v2p11_resolved - v2p11_empty
delta = len(candidate_wrong) - len(control_wrong)
```

Publish the two sets plus introduced/eliminated IDs, `delta`, and `no_regression = delta <= 0`. Because model failures are unresolved and non-empty, they remain inside these sets while retaining their separate failure-analysis classification.

- [x] **Step 3: Enforce the gate at both promotion layers**

Require current, internally consistent wrong-nonempty sets and `wrong_nonempty_no_regression is True` in `_validate_trustworthy_verdict`. Include the same condition in `trustworthy_beats_v2p10`; surface it in Markdown and the r3 goal-audit requirements/score summary.

- [x] **Step 4: Verify GREEN and commit**

Run the three focused comparison/audit test files, then the complete v2.11 contract suite, Python compilation, `bash -n` for the r3 chain/evaluator, and `git diff --check`.

Commit Task 2 with message `fix: gate v2.11 on wrong nonempty outcomes`.

### Task 3: Remote verification and handoff

**Files:**
- Modify: `phaseH_eval/SESSION_STATE.md`

**Interfaces:**
- Consumes: the two reviewed commits and the existing CPU waiter.
- Produces: checksum-identical remote files, passing remote contract tests, and an updated handoff before the training ETA boundary.

- [ ] **Step 1: Request an independent read-only review**

Require exact file/line evidence for any lineage bypass, set-accounting error, backward-compatibility hole, or GPU/process-control regression.

- [ ] **Step 2: Sync only committed paths and verify checksums**

Use `rsync -ciR` followed by checksum dry-runs. Do not sync unrelated dirty-tree files.

- [ ] **Step 3: Run the remote full contract suite**

Use `.venv-eval/bin/python` for pytest, `.venv-train/bin/python` for runtime-helper compilation, and `bash -n` for shell launchers. Do not query GPU0 or poll training before the ETA boundary.

- [ ] **Step 4: Update the handoff**

Record commit IDs, test counts, lineage artifact path, wrong-nonempty gate semantics, unchanged waiter identity/restart policy, and the next PDT evidence boundary. Commit and checksum-sync only `phaseH_eval/SESSION_STATE.md`.
