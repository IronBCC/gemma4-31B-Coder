# Fable Trace-Parity Admission Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Admit a Fable teacher result to the potential **v2.11** training mixture only when its raw stream, after normalization, reproduces the exact verified patch and records a focused passing FAIL_TO_PASS test.

**Architecture:** Move the raw-stream normalizer and a new `preflight_trainable_trace()` function into a small shared module. Both Claude/Codex collection in `teacher_trace_driver.py` and OpenRouter collection in `teacher_platform.py` will call the same preflight after strict patch verification and before setting `training_admitted=true`. The gate preserves raw artifacts and gives a deterministic rejection reason; it never fabricates a tool call, test result, or edit.

**Tech Stack:** Python 3.12, pytest, existing Docker-backed `success_trace_distill` replay contract, existing teacher trace JSONL schemas.

## Global Constraints

- Do not change the active canonical evaluation scripts, GPU0 production, GPU1 serve ownership, or remote processes.
- Do not merge, prepare, blend, or train these Fable rows into frozen v2.10; they are exclusively a possible v2.11 input after the canonical v2.10 verdict and empty-patch controller repair.
- Do not weaken `admission_is_exact`, synthesize verifier steps, or train from a final patch without a replayable raw trajectory.
- Preserve every raw `.stream.jsonl` and `.patch` artifact even when trace preflight rejects it.
- Use the exact numeric PID rule for any later process control; this change does not perform process control.
- Do not commit, stage, revert, or overwrite unrelated dirty-tree work.

---

### Task 1: Add a shared trace-preflight boundary

**Files:**
- Create: `teacher_platform/trace_gate.py`
- Modify: `teacher_platform/teacher_platform.py:1056-1238`
- Test: `teacher_platform/tests/test_trace_gate.py`

**Interfaces:**
- Consumes: `task: Mapping[str, Any]`, `backend: str`, `stream_path: Path`, `evidence: Mapping[str, Any]`.
- Produces: `TracePreflight` with `retained_steps: int` and `source_sha256: str`, or raises `TracePreflightError` with stable reason code `raw_patch_mismatch`, `missing_focused_test`, or `normalization_error`.
- `normalize_steps()` and `NormalizedTrace` remain importable from `teacher_platform.teacher_platform` as compatibility re-exports.

- [ ] **Step 1: Write failing preflight tests**

```python
def test_preflight_rejects_a_strict_patch_without_a_focused_f2p_test(tmp_path, monkeypatch):
    monkeypatch.setattr(trace_gate, "distill_success_path", lambda *_: (_ for _ in ()).throw(
        ValueError("distilled trace lacks a focused passing test from task contract")
    ))
    with pytest.raises(trace_gate.TracePreflightError, match="missing_focused_test"):
        trace_gate.preflight_trainable_trace(_task(), "claude", _stream(tmp_path), _evidence())


def test_preflight_rejects_a_trace_whose_edits_do_not_reproduce_the_admitted_patch(tmp_path, monkeypatch):
    monkeypatch.setattr(trace_gate, "distill_success_path", lambda *_: (_ for _ in ()).throw(
        ValueError("raw mutation subsequence does not reproduce admitted patch")
    ))
    with pytest.raises(trace_gate.TracePreflightError, match="raw_patch_mismatch"):
        trace_gate.preflight_trainable_trace(_task(), "claude", _stream(tmp_path), _evidence())
```

- [ ] **Step 2: Run the new tests and verify red**

Run: `PYTHONPATH=$PWD .venv-eval/bin/python -m pytest -q teacher_platform/tests/test_trace_gate.py`

Expected: FAIL because `teacher_platform.trace_gate` does not exist.

- [ ] **Step 3: Implement the minimal shared normalizer and preflight**

```python
class TracePreflightError(ValueError):
    def __init__(self, code: str, detail: str):
        super().__init__(f"{code}: {detail}")
        self.code = code


def preflight_trainable_trace(task, backend, stream_path, evidence):
    normalized = normalize_steps(backend, stream_path)
    trace = ReplayTrace(
        steps=tuple(ReplayStep(command=s["command"], observation=s["observation"],
                               returncode=s["returncode"], mutates_source=s["mutates_source"],
                               assistant=s.get("thought", ""), loss=True) for s in normalized),
        terminal_assistant=normalized.terminal_assistant,
    )
    try:
        distilled = distill_success_path(task, trace, evidence)
    except ValueError as exc:
        raise _classify_preflight_error(exc) from exc
    return TracePreflight(len(distilled.steps), distilled.source_sha256)
```

Move the existing `NormalizedTrace`, Docker-wrapper stripping, return-code parsing, and `normalize_steps()` implementation without behavior changes into `trace_gate.py`; import/re-export them from `teacher_platform.py`.

- [ ] **Step 4: Run trace-gate and existing normalization tests**

Run: `PYTHONPATH=$PWD .venv-eval/bin/python -m pytest -q teacher_platform/tests/test_trace_gate.py teacher_platform/tests/test_teacher_platform.py teacher_platform/tests/test_success_trace_distill.py`

Expected: PASS. The existing imports from `teacher_platform.teacher_platform` remain valid.

### Task 2: Require trace preflight in both collector paths

**Files:**
- Modify: `phaseD_sft/teacher_trace_driver.py:30-51,345-370`
- Modify: `teacher_platform/teacher_platform.py:562-586`
- Test: `phaseD_sft/tests/test_teacher_trace_driver.py`
- Test: `teacher_platform/tests/test_teacher_platform.py`

**Interfaces:**
- Consumes: strict `verify_candidate_patch()` evidence and the already fsynced raw stream.
- Produces: `training_admitted=false`, `resolved=false`, and `rejection_reasons=["trace_preflight:<code>"]` when raw trace preflight fails; retains `executed=true` and existing patch/control evidence.
- Success adds `trace_preflight_source_sha256` and `trace_preflight_retained_steps` to the ledger before marking it resolved.

- [ ] **Step 1: Write failing collector tests for both paths**

```python
monkeypatch.setattr(ttd, "preflight_trainable_trace", lambda *_: (_ for _ in ()).throw(
    TracePreflightError("missing_focused_test", "focused F2P command absent")
))
record = ttd.collect_one(task, outdir, max_turns=4, model="teacher", backend="claude")
assert record["training_admitted"] is False
assert record["resolved"] is False
assert record["rejection_reasons"] == ["trace_preflight:missing_focused_test"]
assert (outdir / f"{task['instance_id']}.stream.jsonl").is_file()
assert (outdir / f"{task['instance_id']}.patch").is_file()
```

Repeat the same assertion through `collect_one_openrouter()` with its OpenAI client and Docker calls mocked using the existing test fixture pattern.

- [ ] **Step 2: Run collector tests and verify red**

Run: `PYTHONPATH=$PWD .venv-eval/bin/python -m pytest -q phaseD_sft/tests/test_teacher_trace_driver.py teacher_platform/tests/test_teacher_platform.py`

Expected: FAIL because current collectors leave `training_admitted=true` after a strict patch/control pass regardless of raw trace trainability.

- [ ] **Step 3: Add fail-closed preflight after strict verification**

```python
try:
    preflight = preflight_trainable_trace(row, backend, raw_path, evidence)
except TracePreflightError as exc:
    rec.update(evidence)
    rec.update(training_admitted=False, resolved=False,
               rejection_reasons=[f"trace_preflight:{exc.code}"],
               trace_preflight_error=str(exc)[:300])
else:
    rec.update(evidence)
    rec.update(trace_preflight_source_sha256=preflight.source_sha256,
               trace_preflight_retained_steps=preflight.retained_steps,
               resolved=True)
```

Apply this exact behavior to the `teacher_trace_driver.collect_one()` and `collect_one_openrouter()` success branches. Preserve pre-existing strict-verification exception behavior.

- [ ] **Step 4: Run focused regression suite**

Run: `PYTHONPATH=$PWD .venv-eval/bin/python -m pytest -q phaseD_sft/tests/test_teacher_trace_driver.py teacher_platform/tests/test_teacher_platform.py teacher_platform/tests/test_generic_trace_replay.py teacher_platform/tests/test_success_trace_distill.py teacher_platform/tests/test_trace_gate.py`

Expected: PASS, including original strict-admission artifact-integrity tests.

### Task 3: Correctly normalize nested Docker teacher commands

**Files:**
- Modify: `teacher_platform/trace_gate.py`
- Test: `teacher_platform/tests/test_trace_gate.py`

**Interfaces:**
- Consumes: a recorded command such as `docker exec -i <container-id> python3 - <<'PY' ... PY`.
- Produces: a replay-safe direct command `python3 - <<'PY' ... PY`; it removes only the exact runtime container wrapper and never removes an arbitrary shell prefix.

- [ ] **Step 1: Write failing wrapper-normalization test**

```python
def test_strip_docker_exec_preserves_heredoc_body_and_removes_exact_container_wrapper():
    command = "docker exec -i e77a4f49a0e7 python3 - <<'PY'\\nopen('/testbed/x.py', 'w').write('x')\\nPY"
    assert trace_gate._strip_docker_exec(command).startswith("python3 - <<'PY'")
    assert "e77a4f49a0e7" not in trace_gate._strip_docker_exec(command)
```

- [ ] **Step 2: Run the test and verify red**

Run: `PYTHONPATH=$PWD .venv-eval/bin/python -m pytest -q teacher_platform/tests/test_trace_gate.py::test_strip_docker_exec_preserves_heredoc_body_and_removes_exact_container_wrapper`

Expected: FAIL because the current implementation only unwraps `docker exec <id> bash -c`.

- [ ] **Step 3: Extend wrapper parsing minimally**

```python
_DOCKER_EXEC_PREFIX_RE = re.compile(
    r"^\\s*docker\\s+exec(?:\\s+-[A-Za-z]+)*\\s+[0-9a-f]{12,64}\\s+(?P<inner>.+)$",
    re.DOTALL,
)

def _strip_docker_exec(command: str) -> str:
    command = command.strip()
    bash = _DOCKER_EXEC_RE.match(command)
    inner = bash.group(1) if bash else (_DOCKER_EXEC_PREFIX_RE.match(command).group("inner") if _DOCKER_EXEC_PREFIX_RE.match(command) else command)
    return re.sub(r"^\\s*cd\\s+/testbed\\s*&&\\s*", "", inner).strip()
```

Construct the match once in the real code; reject strings that do not contain a 12-64 lowercase-hex container ID so a non-teacher command is never rewritten.

- [ ] **Step 4: Run parser plus collector regression suite**

Run: `PYTHONPATH=$PWD .venv-eval/bin/python -m pytest -q teacher_platform/tests/test_trace_gate.py phaseD_sft/tests/test_teacher_trace_driver.py teacher_platform/tests/test_teacher_platform.py`

Expected: PASS. The previous `bash -c` form, direct commands, and nested `python - <<` form all retain command contents exactly.

### Task 4: Produce a fresh, trace-parity-gated Fable retry pool for v2.11 after the canonical verdict and controller repair

**Files:**
- Create: `data/fable5_trace_parity_retry19.jsonl`
- Create: `data/fable5_trace_parity_retry19.manifest.json`
- Create: `runs/teacher_fable5_trace_parity_retry19/`
- Test: remote collection ledger and post-merge/prepare manifest.

**Interfaces:**
- Consumes: the 19 source task IDs whose original exact records fail `preflight_trainable_trace()`.
- Produces: a retry manifest binding the source IDs and SHA256, then a new raw ledger whose `training_admitted=true` rows all carry `trace_preflight_source_sha256`.

- [ ] **Step 1: Create the retry source from evidence, not from a hand-written list**

```python
failed_ids = sorted(preflight_failures)
retry_rows = [task for task in source_rows if task["instance_id"] in failed_ids]
assert len(retry_rows) == 19
write_jsonl("data/fable5_trace_parity_retry19.jsonl", retry_rows)
write_manifest(source_sha256, failed_ids, "trace_preflight_v1")
```

- [ ] **Step 2: Verify the retry source is disjoint from the 24 renderable rows**

Run: `PYTHONPATH=$PWD .venv-eval/bin/python -c '<verify selected ids are exactly 19 and disjoint from the 24 rendered ids>'`

Expected: PASS with `retry=19`, `renderable=24`, `intersection=0`.

- [ ] **Step 3: Run bounded v2.11 collection only after GPU1 canonical work releases Docker pressure and the empty-patch controller repair has passed its focused tests**

Run: `.venv-train/bin/python teacher_platform/teacher_platform.py collect --backend claude --model claude-fable-5 --tasks data/fable5_trace_parity_retry19.jsonl --out-dir runs/teacher_fable5_trace_parity_retry19 --loop`

Expected: every raw result is preserved; only rows with both strict patch/control evidence and trace-preflight evidence become trainable.

- [ ] **Step 4: Merge selected exact rows and run prepare/format gates**

Run: `teacher_platform.py merge` with an explicit selected-ledger set, followed by `teacher_platform.py prepare` and the project format-loss gate.

Expected: a manifest records raw task total, strict patch/control total, trace-parity total, rejections by reason, source and artifact SHA256 values, and zero format-loss failures before any LoRA training.

## Self-Review

- Spec coverage: prevents the measured 19-row collector/prepare mismatch; retains raw evidence; covers Claude/Codex and OpenRouter paths; normalizes the observed nested Docker edit form; gates the follow-up retry and dataset build.
- Placeholder scan: no TODO/TBD items; each task names files, interfaces, tests, commands, and expected outcomes.
- Type consistency: both collectors consume `TracePreflight`/`TracePreflightError`; only `preflight_trainable_trace()` constructs the replay trace and returns the shared result.

## Execution Handoff

Plan complete and saved to `docs/superpowers/plans/2026-07-29-fable-trace-parity-admission.md`. The execution order is inline only after the frozen v2.10 canonical evaluation reaches its verdict and the controller repair passes; its output is a potential v2.11 input. The plan does not authorize a commit.
