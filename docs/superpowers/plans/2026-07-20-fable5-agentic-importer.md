# Fable 5 Agentic Importer Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a deterministic CPU-only importer that collapses pinned Fable cumulative rows into one structurally valid, edit-decisive Gemma-native trajectory per task.

**Architecture:** `teacher_platform/fable5_import.py` streams the pinned Hugging Face revision, retains only terminal training trajectories, translates the supported Claude-style tool set through the existing fail-closed shell/editor helpers, serializes multi-tool turns into one-bash turns, and applies behavior, decontamination, token, dedup, and manifest gates. Unit tests inject rows, token counts, metadata, and replay results; they never use the network, Docker, or a tokenizer model.

**Tech Stack:** Python 3.11+, `datasets` streaming, existing `teacher_platform.teacher_platform` translation helpers, `phaseD_sft.agentic_trace_filters`, `phaseD_sft.progress`, pytest.

## Global Constraints

- Dataset revision is exactly `aef8506515979988aa5c1a423f5b0fb3cee60382`; `traces.jsonl` LFS SHA-256 is exactly `ef86c61a8e3b69197d381e2e9b6fe1965005c604fa39ba35e0721457813306c3` and size is `730331947` bytes.
- Keep only row-level `split=train` and `assistant_step == assistant_steps`.
- Canonical languages are only `python` (`python|py`), `rust`, and `cpp` (`cpp|c++`).
- Reject unsupported tools, malformed or missing call/results, trailing user/tool turns, test/protected edits, path escapes, and unknown argument semantics.
- Convert to exactly one lowercase `bash` tool call per assistant turn; preserve original assistant content only once when serializing parallel calls.
- Keep one terminal row per task and mark every converted assistant message `loss=true`, so every source assistant turn is supervised once.
- First edit command <=10, maximum read streak <=5, no identical consecutive command, and a language-appropriate post-edit verification command are mandatory.
- Rendered length must be <=49152 tokens; tokenizer loads once in the main process and any parallelism uses threads only.
- Output is `data/fable5_agentic_pilot_v1/`; this plan does not train, serve, use GPU, pull images, or touch production ports.

---

### Task 1: Pinned source selection and structural validation

**Files:**
- Create: `teacher_platform/fable5_import.py`
- Create: `teacher_platform/tests/test_fable5_import.py`

**Interfaces:**
- Produces: `SourceContract`, `DropReason`, `canonical_language(value)`, `select_terminal_row(row)`, `validate_source_messages(messages)`.
- Consumes: injected `Iterable[dict[str, Any]]`; no network dependency in these functions.

- [ ] **Step 1: Write failing tests for the source contract and terminal selector**

Add fixtures that prove exact constants, alias normalization, validation-row rejection, nonterminal rejection, mismatched assistant counts, duplicate task rejection, malformed roles, missing tool IDs/results, duplicate results, and a required final assistant message:

```python
def test_selects_only_terminal_training_rows() -> None:
    row = fable_row(task="rs-one", split="train", assistant_step=4, assistant_steps=4)
    selected = select_terminal_row(row)
    assert selected.task == "rs-one"
    assert selected.language == "rust"


@pytest.mark.parametrize(
    ("updates", "reason"),
    [
        ({"split": "val"}, DropReason.VALIDATION),
        ({"assistant_step": 3}, DropReason.NONTERMINAL),
        ({"lang": "go"}, DropReason.LANGUAGE),
    ],
)
def test_rejects_ineligible_rows(updates: dict[str, object], reason: DropReason) -> None:
    row = fable_row(**updates)
    with pytest.raises(RowRejected, match=reason.value):
        select_terminal_row(row)
```

- [ ] **Step 2: Run the focused tests and verify RED**

Run:

```bash
PYTHONPATH=$PWD .venv/bin/pytest teacher_platform/tests/test_fable5_import.py -q
```

Expected: collection fails because `teacher_platform.fable5_import` does not exist.

- [ ] **Step 3: Implement the immutable source contract and row selector**

Implement frozen constants and immutable selected-row types:

```python
DATASET_ID = "greghavens/fable-5-coding-and-debugging-traces"
DATASET_REVISION = "aef8506515979988aa5c1a423f5b0fb3cee60382"
SOURCE_LFS_SHA256 = "ef86c61a8e3b69197d381e2e9b6fe1965005c604fa39ba35e0721457813306c3"
SOURCE_BYTES = 730_331_947
EXPECTED_ROWS = 12_408
EXPECTED_TERMINAL_TRAJECTORIES = 2_377


@dataclass(frozen=True)
class SelectedTrajectory:
    task: str
    language: Literal["python", "rust", "cpp"]
    category: str
    messages: tuple[dict[str, Any], ...]
    tools_json: str
```

`select_terminal_row` validates exact types rather than coercing values. `validate_source_messages` builds the ordered call-ID ledger and rejects an orphan, duplicate, mismatch, missing result, malformed argument JSON, unsupported role, or non-assistant terminal turn. It returns no partially repaired trajectory.

- [ ] **Step 4: Run focused tests and verify GREEN**

Run the command from Step 2. Expected: all Task 1 tests pass.

- [ ] **Step 5: Commit Task 1**

```bash
git add teacher_platform/fable5_import.py teacher_platform/tests/test_fable5_import.py
git commit -m "feat: select pinned Fable terminal traces"
```

---

### Task 2: Fail-closed Fable tool normalization

**Files:**
- Modify: `teacher_platform/fable5_import.py`
- Modify: `teacher_platform/tests/test_fable5_import.py`

**Interfaces:**
- Consumes: `SelectedTrajectory` from Task 1 and existing `UnsupportedTrajectoryTool`, `_normalize_trajectory_bash_command`, and `_quoted_python_editor` from `teacher_platform.teacher_platform`.
- Produces: `translate_fable_tool_call(tool_call, protected_paths) -> str` and `convert_trajectory(selected, protected_paths) -> dict[str, Any]`.

- [ ] **Step 1: Write failing translation tests**

Cover exact mappings for `Bash`, `Read`, `Write`, `Edit` exact-one and
`replace_all`, `Glob`, and bounded `Grep`; unknown tools/options, background
Bash, absolute/parent/NUL paths, test writes, malformed JSON, and symlink escape
must reject. Add an executable temporary-directory test for every mutating
translation.

```python
def test_parallel_reads_serialize_without_repeating_plan() -> None:
    source = terminal_with_parallel_reads(content="Inspect both files.")
    row = convert_trajectory(source, protected_paths=frozenset())
    assistants = [m for m in row["messages"] if m["role"] == "assistant"]
    assert [m["content"] for m in assistants[:2]] == ["Inspect both files.", ""]
    assert all(len(m["tool_calls"]) <= 1 for m in assistants)
    assert all(m["loss"] is True for m in assistants)


def test_edit_translation_requires_exact_one_match(tmp_path: Path) -> None:
    command = translate_fable_tool_call(edit_call("src/lib.rs", "old", "new"), frozenset())
    target = tmp_path / "src/lib.rs"
    target.parent.mkdir()
    target.write_text("old old", encoding="utf-8")
    result = subprocess.run(["bash", "-c", command.replace("/testbed", str(tmp_path))])
    assert result.returncode != 0
    assert target.read_text(encoding="utf-8") == "old old"
```

- [ ] **Step 2: Run translation tests and verify RED**

Run:

```bash
PYTHONPATH=$PWD .venv/bin/pytest teacher_platform/tests/test_fable5_import.py -q -k 'translation or serialize or pairing'
```

Expected: failures because translation and conversion functions are absent.

- [ ] **Step 3: Implement strict translation**

Parse `function.arguments` as an object and reject unknown keys that change
semantics. Permit Bash metadata only when it is inert (`description`, positive
`timeout`); reject `run_in_background`. Normalize paths to `/testbed/<relative>`
after `PurePosixPath` confinement. Use shell quoting for reads/globs/grep and
base64-backed Python heredocs for write/edit payloads. Do not interpolate file
contents into shell syntax.

`convert_trajectory` emits:

```python
{
    "instance_id": selected.task,
    "source_instance_id": selected.task,
    "repo": f"moonshiner/{selected.task}",
    "source": f"teacher:fable5:{DATASET_REVISION}:{selected.task}",
    "messages": [
        {"role": "system", "content": MINI_SWE_SYSTEM, "tool_calls": []},
        {"role": "user", "content": user_text, "tool_calls": []},
        # assistant messages each contain loss=True and zero or one bash call;
        # tool results become user OBSERVATION turns with loss absent.
    ],
}
```

Pair source calls to results by ID before serialization. Plain assistant text
turns remain assistant messages with `loss=true`. Require the final converted
message to be assistant; never synthesize a completion.

- [ ] **Step 4: Run translation tests and the existing translator suite**

Run:

```bash
PYTHONPATH=$PWD .venv/bin/pytest teacher_platform/tests/test_fable5_import.py teacher_platform/tests/test_teacher_platform.py -q
```

Expected: all tests pass and existing Open-SWE/teacher translation behavior is unchanged.

- [ ] **Step 5: Commit Task 2**

```bash
git add teacher_platform/fable5_import.py teacher_platform/tests/test_fable5_import.py
git commit -m "feat: normalize Fable tools to native bash"
```

---

### Task 3: Behavior, decontamination, token, and manifest pipeline

**Files:**
- Modify: `teacher_platform/fable5_import.py`
- Modify: `teacher_platform/tests/test_fable5_import.py`

**Interfaces:**
- Consumes: `convert_trajectory` from Task 2; injected `token_counter(messages)`, `source_metadata()`, and `replay(task, commands)` callbacks.
- Produces: `build_fable5_pilot(rows, config) -> BuildResult`, CLI `python -m teacher_platform.fable5_import`, `train.jsonl`, `manifest.json`, `rejected.jsonl`.

- [ ] **Step 1: Write failing pipeline tests**

Create mixed fixtures covering every arithmetic bucket, deterministic content
dedup, exact exclusion, 13-word-gram overlap rejection, first edit command 11,
read streak six, repeated commands, missing post-edit verification, token count
49152/49153, replay failure, stable output ordering, and source metadata drift.

```python
def test_manifest_arithmetic_and_determinism(tmp_path: Path) -> None:
    result = build_fable5_pilot(
        rows=mixed_rows(),
        config=fixture_config(tmp_path, token_counter=lambda messages: 4096),
    )
    assert result.manifest["streamed"] == sum(
        result.manifest[key] for key in result.manifest["arithmetic_buckets"]
    )
    assert result.manifest["output_sha256"] == sha256_file(tmp_path / "train.jsonl")
    assert all(row["source"].startswith(f"teacher:fable5:{DATASET_REVISION}:") for row in result.rows)


def test_token_budget_is_inclusive() -> None:
    assert token_gate(messages(), lambda _: 49_152) == 49_152
    with pytest.raises(RowRejected, match="token_budget"):
        token_gate(messages(), lambda _: 49_153)
```

- [ ] **Step 2: Run pipeline tests and verify RED**

Run:

```bash
PYTHONPATH=$PWD .venv/bin/pytest teacher_platform/tests/test_fable5_import.py -q -k 'manifest or dedup or overlap or behavior or token or metadata'
```

Expected: failures because pipeline APIs are absent.

- [ ] **Step 3: Implement the deterministic build pipeline**

Use `command_trace_quality_report(..., max_first_edit_index=10,
max_read_streak=5, reject_identical_consecutive_commands=True)` and then require
a language-specific verification regex after that edit. Canonicalize message
JSON with sorted keys and compact separators before SHA-256 deduplication.
Implement exact ID/content exclusions plus normalized 13-word-gram overlap;
maximum overlap >=0.8 fails closed and is reported.

The CLI accepts:

```text
--out data/fable5_agentic_pilot_v1
--dataset-revision aef8506515979988aa5c1a423f5b0fb3cee60382
--max-output 0
--max-tokens 49152
--workers 8
--tokenizer /media/ironbcc/CrucialX10/models/google/gemma-4-31B-it
--exclusion PATH   (repeatable)
--moonshiner-root PATH
--skip-replay      (structural smoke only; manifest marks replay_pending)
```

Resolve and validate the source LFS metadata before streaming. Load the
tokenizer once, then submit only message lists to a `ThreadPoolExecutor`. Write
to temporary files and atomically replace final outputs only after arithmetic
and hash checks pass. Emit progress at 100 terminal-trajectory boundaries with
elapsed time and ETA; do not poll externally.

- [ ] **Step 4: Run all importer tests and focused project regressions**

Run:

```bash
PYTHONPATH=$PWD .venv/bin/pytest \
  teacher_platform/tests/test_fable5_import.py \
  teacher_platform/tests/test_teacher_platform.py \
  phaseD_sft/tests/test_agentic_trace_filters.py \
  phaseD_sft/tests/test_verify_gemma_format_loss.py -q
```

Expected: all tests pass.

- [ ] **Step 5: Run a networked metadata-only dry run**

Run on the GPU box with no Docker/GPU use:

```bash
PYTHONPATH=$PWD .venv-eval/bin/python -m teacher_platform.fable5_import \
  --out /tmp/fable5_metadata_audit \
  --max-output 0 --skip-replay --metadata-only
```

Expected: exactly 12408 streamed rows, 2377 terminal trajectories, pinned LFS
hash/size match, and no training JSONL emitted.

- [ ] **Step 6: Commit Task 3**

```bash
git add teacher_platform/fable5_import.py teacher_platform/tests/test_fable5_import.py
git commit -m "feat: gate Fable agentic pilot imports"
```

---

### Task 4: Structural smoke and Gemma format gate

**Files:**
- Modify only if a failing test first demonstrates an importer defect:
  `teacher_platform/fable5_import.py`, `teacher_platform/tests/test_fable5_import.py`
- Create runtime artifacts: `data/fable5_agentic_pilot_v1_smoke/`

**Interfaces:**
- Consumes: Task 3 CLI with `--skip-replay`.
- Produces: 30 terminal trajectories plus manifest for the replay plan.

- [ ] **Step 1: Run the 30-trajectory structural smoke on the GPU box**

```bash
PYTHONPATH=$PWD .venv-eval/bin/python -m teacher_platform.fable5_import \
  --out data/fable5_agentic_pilot_v1_smoke \
  --max-output 30 --skip-replay --workers 8 \
  --max-tokens 49152 \
  --tokenizer /media/ironbcc/CrucialX10/models/google/gemma-4-31B-it \
  --moonshiner-root /home/ironbcc/src/moonshiner-pinned \
  --exclusion data/python_teacher_gap_probe60_manifest.json \
  --exclusion data/mswe_rust_prs_full239.jsonl
```

Expected: no malformed pairing, unsupported tool leak, validation row, or token overflow in output. Honest drops are allowed and counted.

- [ ] **Step 2: Run the mandatory format gate**

```bash
PYTHONPATH=$PWD .venv-train/bin/python phaseD_sft/verify_gemma_format_loss.py \
  --data data/fable5_agentic_pilot_v1_smoke/train.jsonl \
  --samples 30 --workers 8
```

Expected: process exits zero and `failure_count=0`.

- [ ] **Step 3: Audit the smoke manifest**

Verify arithmetic, row-level `split=train`, one source task per output row,
assistant `loss=true`, one call per assistant turn, first-edit and token
histograms, source revision/hash, and `replay_pending=true`. Stop rather than
scale if any invariant is missing.

- [ ] **Step 4: Commit only code fixes proven by new failing tests**

If the smoke required no code fix, create no commit. If it revealed a defect,
write the reproducer test first, verify RED, implement the minimal fix, rerun
Steps 1-3, and commit only the code/test files.
