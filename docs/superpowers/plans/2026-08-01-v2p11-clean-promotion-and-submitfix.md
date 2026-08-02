# v2.11 Clean Promotion and Submit-Fix Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Promote a trustworthy clean v2.11 model only if it beats canonical v2.10 on the identical SWE-bench Lite 300 evaluation without an empty-patch regression, otherwise train one evidence-selected raw-base successor and evaluate it under the same gate.

**Architecture:** Preserve the running clean training and fail-closed post-training chain as the primary experiment. The paired full300 comparator selects at most one successor data path: terminal submit repair for empty/format failures, diversity and measured source weight for a healthy tie, or replay-verified correction traces for loops and wrong edits. Every training candidate starts from the raw Gemma base and uses a single SFT stage; recovery SFT, KTO, interpolation, and controller-only promotion are excluded.

**Tech Stack:** Bash/systemd user units, Python 3.10+, pytest, Transformers Gemma chat templates, Unsloth/PEFT LoRA, vLLM, Mini-SWE-agent, SWE-bench Docker evaluation.

## Global Constraints

- GPU1 only; do not inspect, stop, or schedule work on GPU0.
- Canonical control remains v2.10 at 157/300 with two corrected empties.
- Candidate evaluation must contain exactly 300 predictions and 300 trajectory files for the identical ID set.
- Preserve the active `v2p11-clean-fable51-train-gpu1-v2.service` and `v2p11-clean-fable51-posttrain-chain-gpu1-v1.service` until a completion, safety, or failure boundary.
- Do not score or compose predictions after an owned serve or watchdog failure.
- Use exact numeric PIDs for process validation or control; never use pattern-based process killing.
- Admit only boolean-loss, mutation-bound, execution-verified, focused-test-passing Fable trajectories.
- Keep unrelated dirty-tree files unstaged and unmodified.

---

### Task 1: Close the current clean training boundary

**Files:**
- Inspect: `phaseH_eval/train_v2p11_clean_gpu1.sh`
- Inspect: `phaseH_eval/run_v2p11_clean_posttrain_chain.sh`
- Inspect: `runs/v2p11_clean_fable51_context_audit.json` on the authoritative remote checkout
- Inspect: `runs/v2p11_clean_fable51_gpu1_training_identity.json` on the authoritative remote checkout
- Modify after the boundary: `phaseH_eval/SESSION_STATE.md`

**Interfaces:**
- Consumes: systemd invocations `bf042e80eda24928a28c6c0784cbc0b2` and `96a45ce081d145cab517dd119df742b6`.
- Produces: immutable training completion, adapter audit, merge, portability, provenance, and full300 chain artifacts or one exact failure boundary.

- [ ] **Step 1: Wait for the evidence-based completion boundary**

Do not poll optimizer or batch boundaries. At the projected completion boundary, run only GPU1-safe checks:

```bash
ssh ironbccllm 'systemctl --user show v2p11-clean-fable51-train-gpu1-v2.service -p ActiveState -p SubState -p MainPID -p ExecMainStatus -p InvocationID --no-pager; systemctl --user show v2p11-clean-fable51-posttrain-chain-gpu1-v1.service -p ActiveState -p SubState -p MainPID -p ExecMainStatus -p InvocationID --no-pager'
```

Expected training boundary: inactive/success with invocation `bf042e80eda24928a28c6c0784cbc0b2`. The chain may remain active while it audits and evaluates.

- [ ] **Step 2: Verify immutable training evidence**

```bash
ssh ironbccllm 'cd /home/ironbcc/projects/gemma4-31B-Coder && test -s runs/v2p11_clean_fable51_training_completion.json && test -s runs/v2p11_clean_fable51_adapter_audit.json && python3 phaseH_eval/v2p11_clean_completion_provenance.py --help >/dev/null'
```

Read the artifacts and require 79/79 optimizer steps, raw-base lineage, `init_adapter=null`, 1,262 rows, rank/alpha 32/32, and the sealed dataset hashes. Stop on any mismatch.

- [ ] **Step 3: Run the existing focused verification suite locally**

```bash
pytest -q \
  phaseH_eval/tests/test_train_v2p11_clean_gpu1.py \
  phaseH_eval/tests/test_run_v2p11_clean_posttrain_chain.py \
  phaseH_eval/tests/test_v2p11_clean_completion_provenance.py \
  phaseH_eval/tests/test_v2p11_portability_gate.py \
  phaseH_eval/tests/test_compare_v2p11_full300.py
```

Expected: zero failures.

- [ ] **Step 4: Record the verified boundary**

Append one timestamped section to `phaseH_eval/SESSION_STATE.md` containing exact unit states, invocation IDs, artifact paths and hashes, adapter audit counts, and the next boundary. Do not copy transient batch status into the canonical result ledger.

- [ ] **Step 5: Commit only the session-state update**

```bash
git add phaseH_eval/SESSION_STATE.md
git diff --cached --check
git commit -m "docs: record clean v2.11 training boundary"
```

### Task 2: Verify portability and identical full300 output

**Files:**
- Inspect: `phaseH_eval/run_v2p11_portability_gate.sh`
- Inspect: `phaseH_eval/eval_v2p11_full300_after_merge.sh`
- Inspect: `phaseH_eval/compare_v2p11_full300.py`
- Inspect: `runs/v2p10_full300_composite.json` on the authoritative remote checkout
- Modify after the boundary: `phaseH_eval/SESSION_STATE.md`

**Interfaces:**
- Consumes: merged clean model plus immutable training and provenance artifacts from Task 1.
- Produces: portability verdict, 300 first-pass outcomes, empty-only correction, composite comparison JSON/Markdown, and a promotion decision.

- [ ] **Step 1: Verify the portability artifact before accepting full300**

Require stock-template serving, valid tool calls, owned serve/watchdog shutdown evidence, source-focused non-empty patches, and zero partial-output acceptance. If portability fails, skip directly to Task 4.

- [ ] **Step 2: Verify the harness contract against v2.10**

Use `runs/v2p10_full300_composite.json` as the trust root. Compare exact ID SHA, model/template fields, temperature, seed, step budget, environment class, worker ceiling, prediction schema, and empty-resampling policy. A field mismatch is a failed comparison, not a score.

- [ ] **Step 3: Verify full300 completeness**

```bash
ssh ironbccllm 'cd /home/ironbcc/projects/gemma4-31B-Coder && python3 - <<'"'"'PY'"'"'
import json
from pathlib import Path
p = Path("runs/v2p11_clean_fable51_full300_composite.json")
x = json.loads(p.read_text())
assert x["status"] == "complete"
assert x["preds"] == 300
assert x["traj_files"] == 300
assert x["pull_failed"] == 0
assert x["docker_failed"] == 0
print(x["resolved"], x["preds"], x["traj_files"])
PY'
```

If the chain uses a different sealed filename, take it only from the chain completion artifact; do not discover candidates by glob.

- [ ] **Step 4: Apply the promotion predicate**

Promote only if corrected resolved is greater than 157, corrected empties are at most two, and behavior health shows no material paired wrong-edit, loop, or tool-format regression. If it passes, continue to Task 3. Otherwise continue to Task 4.

### Task 3: Publish a passing clean v2.11

**Files:**
- Modify: `phaseH_eval/SESSION_STATE.md`
- Modify if required by the existing registry contract: `docs/ADAPTER_REGISTRY.md`
- Inspect: `phaseH_eval/train_merge_v2p11_after_full300.sh`

**Interfaces:**
- Consumes: passing Task 2 comparison and complete merged model provenance.
- Produces: canonical v2.11 registry entry, final score, checksums, and paired failure analysis.

- [ ] **Step 1: Re-run final provenance validation**

Run the existing clean completion and goal audit entry points against the exact final artifacts. Require zero validation errors and no changed hashes since Task 2.

- [ ] **Step 2: Publish the canonical result**

Record model path, adapter path, dataset hashes, training invocation, merged-model hashes, harness contract, first-pass and corrected scores, empty count, predictions, trajectory files, and paired movement against v2.10.

- [ ] **Step 3: Verify only intended documentation changes are staged**

```bash
git status --short
git diff --cached --check
git diff --cached --stat
```

- [ ] **Step 4: Commit the promoted result**

```bash
git add phaseH_eval/SESSION_STATE.md docs/ADAPTER_REGISTRY.md
git commit -m "docs: promote clean v2.11 full300 result"
```

After this commit, mark the active goal complete and report the final score. Do not execute Tasks 4-7.

### Task 4: Classify a non-promoting clean result

**Files:**
- Modify only if classification fields are missing: `phaseH_eval/compare_v2p11_full300.py`
- Test only if modified: `phaseH_eval/tests/test_compare_v2p11_full300.py`
- Create: `runs/v2p11_clean_fable51_failure_classification.json` on the authoritative remote checkout through the audited comparator

**Interfaces:**
- Consumes: paired v2.10 and clean-v2.11 300-instance outcomes.
- Produces: one selected successor mode: `submit_fix`, `signal_diversity`, or `verified_correction`.

- [ ] **Step 1: Write a failing comparator test only if a required class is absent**

Add a fixture with one instance for each classification:

```python
EXPECTED_CLASSES = {
    "both_resolved",
    "v2p10_only",
    "v2p11_only",
    "empty",
    "tool_format",
    "loop",
    "wrong_nonempty",
    "infrastructure",
}
```

Assert every one of the 300 IDs appears exactly once and infrastructure failures cannot be counted as model failures.

- [ ] **Step 2: Run the focused test and observe failure if code changed**

```bash
pytest -q phaseH_eval/tests/test_compare_v2p11_full300.py
```

- [ ] **Step 3: Implement the minimum missing classification logic**

Reuse existing prediction, trajectory, score, and behavior-health fields. Do not infer tool-format or loop failures from patch emptiness alone.

- [ ] **Step 4: Re-run the focused test**

```bash
pytest -q phaseH_eval/tests/test_compare_v2p11_full300.py
```

Expected: zero failures.

- [ ] **Step 5: Select exactly one successor**

- Select `submit_fix` when empty plus tool-format regressions explain the failed gate.
- Select `signal_diversity` when the result is a healthy statistical tie with no behavior regression.
- Select `verified_correction` when loops or wrong-nonempty paired losses dominate and raw evidence contains later verified corrections.

Store counts, instance IDs, input hashes, and the selection reason. Do not combine successor modes.

### Task 5: Build the evidence-selected successor dataset

**Files:**
- Create: `phaseD_sft/build_v2p11_submitfix_mix.py` only for `submit_fix`
- Create: `phaseD_sft/tests/test_build_v2p11_submitfix_mix.py` only for `submit_fix`
- Create: `phaseD_sft/build_v2p11_diverse_mix.py` only for `signal_diversity`
- Create: `phaseD_sft/tests/test_build_v2p11_diverse_mix.py` only for `signal_diversity`
- Reuse: `teacher_platform/success_trace_distill.py` for `verified_correction`
- Create: `phaseD_sft/tests/test_build_v2p11_verified_correction_mix.py` only for `verified_correction`

**Interfaces:**
- Consumes: exact 1,211-row v2.10 base JSONL, admitted Fable evidence, and Task 4 selection.
- Produces: one immutable raw-base successor dataset, manifest, format/loss report, context audit, and input hashes.

- [ ] **Step 1: Write the failing test for the selected path**

For `submit_fix`, define and test this interface:

```python
def replace_terminal_with_submit(row: dict, *, call_id: str) -> dict:
    """Return a copy ending in one supervised bash submission turn."""
```

The final message must equal this shape, with JSON-encoded arguments:

```python
{
    "role": "assistant",
    "content": "",
    "loss": True,
    "tool_calls": [{
        "id": call_id,
        "type": "function",
        "function": {
            "name": "bash",
            "arguments": json.dumps({
                "command": "git diff -- . > patch.txt && test -s patch.txt && echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT && cat patch.txt"
            }),
        },
    }],
}
```

Tests must reject a terminal prose target, empty diff, test-only diff, scratch artifact, duplicate ID, patch mismatch, non-boolean loss, and evaluation overlap.

For `signal_diversity`, define and test:

```python
def select_diverse_rows(rows: list[dict], *, per_repo_cap: int = 4) -> list[dict]:
    """Select deterministic, strictly admitted rows with at most four per repo."""
```

The test must prove deterministic ordering, no duplicate instances, repository cap four, and a measured admitted-Fable supervised-token share between 0.03 and 0.05.

For `verified_correction`, test that every wrong/loop/empty assistant action has `loss=False`, at least one later corrective mutation and terminal submit call have `loss=True`, and replay reproduces the admitted patch SHA.

- [ ] **Step 2: Run the selected failing test**

```bash
pytest -q phaseD_sft/tests/test_build_v2p11_submitfix_mix.py
# or the one selected alternative test file
```

Expected: fail because the selected builder does not yet exist.

- [ ] **Step 3: Implement only the selected builder**

Use immutable inputs and `Path.open("x")` or equivalent exclusive publication. The manifest must include source hashes, output hash, exact base-prefix hash, selection mode, per-repository counts, supervised-token report, context maximum, evaluation-overlap count, and `all_training_gates_complete=true` only after every validator passes.

- [ ] **Step 4: Run builder and validator tests**

```bash
pytest -q \
  phaseD_sft/tests/test_build_v2p11_submitfix_mix.py \
  phaseD_sft/tests/test_train_rust_lora.py \
  teacher_platform/tests/test_success_trace_distill.py
```

Use the selected alternative test filename when `submit_fix` is not selected. Expected: zero failures.

- [ ] **Step 5: Run the full real-data audit before any GPU launch**

Run the builder in the remote training environment with `CUDA_VISIBLE_DEVICES=""`. Then run `phaseD_sft/verify_gemma_format_loss.py` and the replay/mutation validator. Require exact prefix equality, unique IDs, zero evaluation overlap, no context over 32,768, no fallback spans, and all strict admission gates.

- [ ] **Step 6: Commit only the selected builder and tests**

```bash
# submit_fix
git add -- phaseD_sft/build_v2p11_submitfix_mix.py phaseD_sft/tests/test_build_v2p11_submitfix_mix.py
# signal_diversity
# git add -- phaseD_sft/build_v2p11_diverse_mix.py phaseD_sft/tests/test_build_v2p11_diverse_mix.py
# verified_correction
# git add -- phaseD_sft/build_v2p11_verified_correction_mix.py phaseD_sft/tests/test_build_v2p11_verified_correction_mix.py
git diff --cached --check
git commit -m "feat: build verified v2.11 successor mix"
```

Use exactly the uncommented pair for the mode selected in Task 4; never stage more than one builder path.

### Task 6: Train and merge one raw-base successor

**Files:**
- Create: `phaseH_eval/train_v2p11_successor_gpu1.sh`
- Create: `phaseH_eval/tests/test_train_v2p11_successor_gpu1.py`
- Create: `phaseH_eval/run_v2p11_successor_posttrain_chain.sh`
- Create: `phaseH_eval/tests/test_run_v2p11_successor_posttrain_chain.py`

**Interfaces:**
- Consumes: Task 5 immutable dataset and manifest.
- Produces: one raw-base LoRA, merged model, portability verdict, provenance, and full300 evaluation.

- [ ] **Step 1: Write failing launcher and chain tests**

Assert GPU index must be exactly `1`; base must be `/media/ironbcc/CrucialX10/models/google/gemma-4-31B-it`; `init_adapter` must be null; rank/alpha must be 32/32; maximum sequence length must be 32,768; recovery, KTO, DPO, and interpolation inputs must be absent; immutable dataset hashes and row counts must match; RAM floors and exact-PID watchdog ownership must be enabled.

- [ ] **Step 2: Run the failing tests**

```bash
pytest -q \
  phaseH_eval/tests/test_train_v2p11_successor_gpu1.py \
  phaseH_eval/tests/test_run_v2p11_successor_posttrain_chain.py
```

- [ ] **Step 3: Implement the minimum hardened launcher and chain**

Copy the proven safety and provenance structure from `train_v2p11_clean_gpu1.sh` and `run_v2p11_clean_posttrain_chain.sh`, replacing only sealed dataset/model names, hashes, row counts, and optimizer-step count. Preserve the owned process-group and watchdog semantics.

- [ ] **Step 4: Run focused and regression tests**

```bash
pytest -q \
  phaseH_eval/tests/test_train_v2p11_successor_gpu1.py \
  phaseH_eval/tests/test_run_v2p11_successor_posttrain_chain.py \
  phaseH_eval/tests/test_train_v2p11_clean_gpu1.py \
  phaseH_eval/tests/test_run_v2p11_clean_posttrain_chain.py \
  phaseH_eval/tests/test_compare_v2p11_full300.py
bash -n phaseH_eval/train_v2p11_successor_gpu1.sh phaseH_eval/run_v2p11_successor_posttrain_chain.sh
```

Expected: zero failures and valid Bash syntax.

- [ ] **Step 5: Sync and verify remote parity**

Copy only the scoped committed files to `/home/ironbcc/projects/gemma4-31B-Coder`, then compare SHA-256 for every local/remote file. Run the same focused tests remotely before launch.

- [ ] **Step 6: Launch exactly one GPU1 training unit and its CPU waiter**

Before launch, verify GPU1 has no foreign compute PID, host memory is above the start floor, Docker free space is above the admission floor, port 8013 is unowned, and no predecessor unit remains active. Record the exact unit names, invocation IDs, wrapper PID, trainer PID, watchdog PID, dataset/model hashes, and completion ETA.

- [ ] **Step 7: Commit launch contracts**

```bash
git add phaseH_eval/train_v2p11_successor_gpu1.sh phaseH_eval/run_v2p11_successor_posttrain_chain.sh phaseH_eval/tests/test_train_v2p11_successor_gpu1.py phaseH_eval/tests/test_run_v2p11_successor_posttrain_chain.py phaseH_eval/SESSION_STATE.md
git diff --cached --check
git commit -m "feat: launch verified v2.11 successor"
```

### Task 7: Evaluate and publish the successor verdict

**Files:**
- Reuse: `phaseH_eval/run_v2p11_portability_gate.sh`
- Reuse: `phaseH_eval/eval_v2p11_full300_after_merge.sh`
- Reuse: `phaseH_eval/compare_v2p11_full300.py`
- Modify: `phaseH_eval/SESSION_STATE.md`
- Modify only on promotion: `docs/ADAPTER_REGISTRY.md`

**Interfaces:**
- Consumes: Task 6 complete successor model and provenance.
- Produces: a promoted successor or a verified negative result retaining v2.10.

- [ ] **Step 1: Apply the same portability and full300 gates from Task 2**

Do not add a checkpoint-specific exception. Require identical IDs, harness, predictions, trajectories, correction policy, and infrastructure health.

- [ ] **Step 2: Apply the unchanged promotion predicate**

Require corrected resolved greater than 157, corrected empties at most two, and no material paired behavior regression.

- [ ] **Step 3: Publish the final result**

If passing, record and register the successor as canonical v2.11. If failing, retain v2.10 and publish the verified negative score plus per-instance failure classes. Do not weaken the gate or launch a second unplanned successor.

- [ ] **Step 4: Run final verification**

```bash
pytest -q \
  phaseH_eval/tests/test_compare_v2p11_full300.py \
  phaseH_eval/tests/test_v2p11_clean_completion_provenance.py \
  phaseH_eval/tests/test_v2p11_goal_completion_audit.py
git diff --check
git status --short
```

Expected: zero test failures, no whitespace errors, and only intended result documentation changes.

- [ ] **Step 5: Commit and close the goal**

```bash
git add phaseH_eval/SESSION_STATE.md docs/ADAPTER_REGISTRY.md
git commit -m "docs: publish final v2.11 verdict"
```

Mark the active goal complete only after the final score, predictions, trajectory counts, empty count, model path, and artifact hashes are verified.
