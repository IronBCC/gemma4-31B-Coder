# Fable 5 Independent Replay Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Independently reconstruct and verify Fable candidates from pinned public Moonshiner seeds without executing transcript shell commands.

**Architecture:** `teacher_platform/fable5_replay.py` validates the pinned seed fixture, reconstructs only declarative `Write`/`Edit` operations into fresh workspaces, and runs the trusted seed `verify_cmd` through an injected restricted executor. The Linux implementation uses `bwrap`; unit tests use a local fake executor and never invoke Docker, bwrap, or the network. Fable-candidate, reference-positive, and corrupt-negative controls produce separate tainted evidence records.

**Tech Stack:** Python 3.11+, `pathlib`, `hashlib`, `subprocess`, `tempfile`, Linux `bwrap`, pytest.

## Global Constraints

- Moonshiner repository revision is exactly `436316e8f86eb136d5ce3ec95a1a6f48c1d7f940`; fixture root is `tasks/seeds/<task>/`.
- Dataset revision remains exactly `aef8506515979988aa5c1a423f5b0fb3cee60382`.
- Reconstruct candidates only from declarative `Write` and `Edit` operations. Never execute transcript Bash, scripts, interpreters, package managers, or command substitutions to create candidate state.
- Reject any Bash operation that is not proven read-only or a language-appropriate verification command; ambiguity fails closed.
- Protected `test_files` must remain byte-identical; paths are relative, NUL-free, non-symlink, and confined to the fixture workspace.
- Verification runs twice in fresh candidate states and must return zero with identical output hashes.
- Every smoke task also runs the pinned reference patch as a tainted positive control and a deterministic corrupt candidate as a tainted negative control; neither control may enter training output.
- Prefer `bwrap` on Linux. No automatic image pull, host execution of Fable commands, GPU use, production-port changes, broad prune, or broad process kill.

---

### Task 1: Seed contract and declarative candidate reconstruction

**Files:**
- Create: `teacher_platform/fable5_replay.py`
- Create: `teacher_platform/tests/test_fable5_replay.py`

**Interfaces:**
- Consumes: original terminal Fable `messages`, task ID, and pinned Moonshiner root.
- Produces: `SeedContract`, `MutationPlan`, `load_seed_contract(root, task)`, `reconstruct_candidate(contract, messages, destination)`.

- [ ] **Step 1: Write failing fixture and mutation tests**

Use temporary seed trees to cover exact required `task.json` types, missing
files, symlinks, escaping paths, duplicate/missing protected files, fixture
hashing, `Write`, exact-one `Edit`, `replace_all`, protected writes, arbitrary
Bash mutation, allowlisted read/test Bash, and deterministic candidate hashes.

```python
def test_reconstructs_write_and_edit_without_executing_bash(tmp_path: Path) -> None:
    contract = seed_fixture(tmp_path, files={"src/lib.rs": "old\n"})
    messages = fable_messages(
        write=("src/new.rs", "pub fn new() {}\n"),
        edit=("src/lib.rs", "old", "new"),
        bash="cargo test --offline",
    )
    out = tmp_path / "candidate"
    plan = reconstruct_candidate(contract, messages, out)
    assert (out / "src/lib.rs").read_text() == "new\n"
    assert (out / "src/new.rs").read_text() == "pub fn new() {}\n"
    assert plan.verify_commands == ("cargo test --offline",)


def test_rejects_shell_mutation_even_when_publisher_trace_passed(tmp_path: Path) -> None:
    contract = seed_fixture(tmp_path)
    with pytest.raises(ReplayRejected, match="bash_mutation"):
        reconstruct_candidate(
            contract,
            fable_messages(bash="python3 -c 'open(\"src/x.py\",\"w\").write(\"x\")'"),
            tmp_path / "candidate",
        )
```

- [ ] **Step 2: Run focused tests and verify RED**

```bash
PYTHONPATH=$PWD .venv/bin/pytest teacher_platform/tests/test_fable5_replay.py -q
```

Expected: collection fails because `teacher_platform.fable5_replay` does not exist.

- [ ] **Step 3: Implement seed validation and mutation planning**

`load_seed_contract` requires `id`, `lang`, `verify_cmd`, positive
`verify_timeout`, nonempty `test_files`, `files/`, and
`reference_fix.patch`. It computes canonical fixture and protected hashes and
binds the Moonshiner commit supplied by the CLI.

`reconstruct_candidate` copies the fixture without following symlinks, applies
only validated `Write` and `Edit` objects with Python file APIs, and records the
ordered operations. Bash is parsed only for classification; accepted read/test
commands are evidence and are not executed during reconstruction. Use the
existing robust shell tokenization helpers from
`teacher_platform.teacher_platform`; do not create a second permissive parser.

- [ ] **Step 4: Run focused tests and verify GREEN**

Run the command from Step 2. Expected: all Task 1 tests pass.

- [ ] **Step 5: Commit Task 1**

```bash
git add teacher_platform/fable5_replay.py teacher_platform/tests/test_fable5_replay.py
git commit -m "feat: reconstruct declarative Fable candidates"
```

---

### Task 2: Restricted double verification and controls

**Files:**
- Modify: `teacher_platform/fable5_replay.py`
- Modify: `teacher_platform/tests/test_fable5_replay.py`

**Interfaces:**
- Consumes: `SeedContract` and reconstructed candidate from Task 1.
- Produces: `RestrictedExecutor` protocol, `BwrapExecutor`, `verify_candidate(...) -> ReplayEvidence`, `run_control_triplet(...)`.

- [ ] **Step 1: Write failing executor and evidence tests**

Test exact bwrap argv construction, missing-bwrap admission, clean environment,
network/PID/temp isolation, only-workspace-writable policy, timeout, output byte
cap, two fresh workspaces, output-hash mismatch, protected drift, source-empty
diff, positive reference control, negative control, and cleanup on every failure.

```python
def test_double_verification_requires_identical_success(tmp_path: Path) -> None:
    executor = RecordingExecutor([
        CommandResult(0, b"18 passed\n"),
        CommandResult(0, b"18 passed\n"),
    ])
    evidence = verify_candidate(contract(), candidate(), executor)
    assert evidence.resolved is True
    assert evidence.output_sha256[0] == evidence.output_sha256[1]
    assert executor.workspaces[0] != executor.workspaces[1]


def test_control_artifacts_are_never_trainable() -> None:
    controls = run_control_triplet(contract(), candidate(), executor_factory())
    assert controls.fable.control_only is False
    assert controls.reference.control_only is True
    assert controls.corrupt.control_only is True
```

- [ ] **Step 2: Run executor tests and verify RED**

```bash
PYTHONPATH=$PWD .venv/bin/pytest teacher_platform/tests/test_fable5_replay.py -q -k 'verify or executor or control or cleanup'
```

Expected: failures because executor/evidence functions are absent.

- [ ] **Step 3: Implement the restricted executor and evidence schema**

Define:

```python
class RestrictedExecutor(Protocol):
    def run(self, workspace: Path, command: str, timeout_seconds: int) -> CommandResult: ...


@dataclass(frozen=True)
class ReplayEvidence:
    schema_version: int
    task: str
    candidate_sha256: str
    fixture_sha256: str
    protected_before: dict[str, str]
    protected_after: dict[str, str]
    return_codes: tuple[int, int]
    output_sha256: tuple[str, str]
    diff_sha256: str
    resolved: bool
    failure_class: str | None
    control_only: bool
```

`BwrapExecutor` uses a read-only `/`, `--unshare-all`, `--die-with-parent`,
empty environment plus deterministic `PATH/HOME/TMPDIR`, private `/tmp`, and a
single writable workspace bind. Stream output to a bounded file; 64 MiB or
timeout fails closed. Shell invocation is `bash -c` only for the trusted pinned
`verify_cmd`, never a transcript command.

Apply `reference_fix.patch` only inside the control workspace. Create the
negative by deterministically replacing the first byte of the first changed
nonprotected source file. Require Fable pass, reference pass, and corrupt fail
for a control triplet to be green.

- [ ] **Step 4: Run all replay tests and verify GREEN**

```bash
PYTHONPATH=$PWD .venv/bin/pytest \
  teacher_platform/tests/test_fable5_replay.py \
  teacher_platform/tests/test_fable5_import.py \
  teacher_platform/tests/test_teacher_platform.py -q
```

Expected: all tests pass.

- [ ] **Step 5: Commit Task 2**

```bash
git add teacher_platform/fable5_replay.py teacher_platform/tests/test_fable5_replay.py
git commit -m "feat: verify Fable candidates in restricted replay"
```

---

### Task 3: Replay CLI integration and five-task smoke

**Files:**
- Modify: `teacher_platform/fable5_replay.py`
- Modify: `teacher_platform/fable5_import.py`
- Modify: `teacher_platform/tests/test_fable5_replay.py`
- Modify: `teacher_platform/tests/test_fable5_import.py`

**Interfaces:**
- Consumes: structural importer output and its retained original terminal-row sidecar.
- Produces: CLI `python -m teacher_platform.fable5_replay`, `replay.jsonl`, control evidence under a separate directory, importer replay callback.

- [ ] **Step 1: Write failing integration tests**

Test atomic JSONL publication, resume by task/evidence hash, refusal to mix
control rows into training, importer drop on missing/failed evidence, exact
per-language smoke selection, and manifest replay arithmetic.

```python
def test_importer_keeps_only_noncontrol_resolved_replay(tmp_path: Path) -> None:
    evidence = replay_ledger(
        fable=resolved_evidence(control_only=False),
        reference=resolved_evidence(control_only=True),
        corrupt=failed_evidence(control_only=True),
    )
    rows, manifest = publish_replay_verified(structural_rows(), evidence, tmp_path)
    assert [row["instance_id"] for row in rows] == ["py-one"]
    assert manifest["controls_in_training"] == 0
```

- [ ] **Step 2: Run integration tests and verify RED**

```bash
PYTHONPATH=$PWD .venv/bin/pytest teacher_platform/tests/test_fable5_replay.py teacher_platform/tests/test_fable5_import.py -q -k 'ledger or publish or resume or smoke'
```

Expected: failures because CLI/ledger integration is absent.

- [ ] **Step 3: Implement CLI and importer replay join**

CLI inputs are the structural rejection-safe original-row sidecar, pinned
Moonshiner root/commit, output ledger, `--tasks` default 5, and
`--languages python:2,rust:2,cpp:1`. Publication uses temporary files plus
`os.replace`, fsyncs file and parent directory, and rejects a resume record
whose task/candidate/fixture hashes differ.

Controls live only under `runs/fable5_replay_controls/<run_id>/`; importer code
rejects `control_only=true` and any evidence path under the control namespace.

- [ ] **Step 4: Run focused tests and metadata inventory**

```bash
PYTHONPATH=$PWD .venv/bin/pytest teacher_platform/tests/test_fable5_replay.py teacher_platform/tests/test_fable5_import.py -q
ssh ironbccllm 'command -v bwrap || true; uname -m; free -g; df -BG /var/lib/docker'
```

Expected: tests pass; inventory proves `x86_64`, host RAM safety, Docker floor,
and either an available `bwrap` or a reported safe-environment blocker. Do not
pull an image.

- [ ] **Step 5: Run the five-task control smoke when admission is green**

```bash
PYTHONPATH=$PWD .venv-eval/bin/python -m teacher_platform.fable5_replay \
  --input data/fable5_agentic_pilot_v1_smoke/original_terminal_rows.jsonl \
  --moonshiner-root /home/ironbcc/src/moonshiner-pinned \
  --moonshiner-commit 436316e8f86eb136d5ce3ec95a1a6f48c1d7f940 \
  --out data/fable5_agentic_pilot_v1_smoke/replay.jsonl \
  --control-dir runs/fable5_replay_controls/smoke1 \
  --tasks 5 --languages python:2,rust:2,cpp:1
```

PASS: five reconstructed Fable candidates pass twice, five reference controls
pass twice, five corrupt controls fail, protected hashes stay unchanged, and
the restricted executor cleans every workspace. Any missing environment,
reconstruction ambiguity, or control failure stops scaling.

- [ ] **Step 6: Run the importer replay join and format gate**

```bash
PYTHONPATH=$PWD .venv-eval/bin/python -m teacher_platform.fable5_import \
  --out data/fable5_agentic_pilot_v1_smoke_verified \
  --replay-ledger data/fable5_agentic_pilot_v1_smoke/replay.jsonl \
  --max-output 5 --workers 8 --max-tokens 49152 \
  --tokenizer /media/ironbcc/CrucialX10/models/google/gemma-4-31B-it
PYTHONPATH=$PWD .venv-train/bin/python phaseD_sft/verify_gemma_format_loss.py \
  --data data/fable5_agentic_pilot_v1_smoke_verified/train.jsonl \
  --samples 5 --workers 5
```

PASS: exactly five noncontrol replay-verified rows and `failure_count=0`.

- [ ] **Step 7: Commit Task 3 code only**

```bash
git add teacher_platform/fable5_replay.py teacher_platform/fable5_import.py \
  teacher_platform/tests/test_fable5_replay.py teacher_platform/tests/test_fable5_import.py
git commit -m "feat: join verified Fable replay evidence"
```
