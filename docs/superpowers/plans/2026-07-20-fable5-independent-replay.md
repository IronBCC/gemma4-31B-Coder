# Fable 5 Independent Replay Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use
> `superpowers:subagent-driven-development` or `superpowers:executing-plans` and
> implement one reviewed task at a time.

**Goal:** Independently reconstruct and verify one deterministic Fable
trajectory per pinned Moonshiner task without executing transcript shell
commands or exposing host secrets.

**Architecture:** `teacher_platform/fable5_replay.py` materializes seed fixtures
from an exact Git object, consumes the importer's shared typed Fable operation
parser, applies only declarative `Write`/`Edit` operations, and executes the
pinned seed verifier in a functionally admitted immutable Docker image. Unit
tests inject a fake executor and never use Git, Docker, the network, or the GPU.

**Known admission state (2026-07-20):** the remote `bwrap --unshare-all` policy
fails with namespace permission errors, the pinned checkout is absent, and the
host lacks Cargo. `bwrap` is not an executor for this version. Runtime replay is
blocked until exact cached Docker digests and toolchains pass the complete
functional admission probe. No image pull is authorized.

## Global contracts

- Dataset revision:
  `aef8506515979988aa5c1a423f5b0fb3cee60382`.
- Moonshiner revision:
  `436316e8f86eb136d5ce3ec95a1a6f48c1d7f940`.
- Primary key:
  `trajectory_id = sha256(dataset_revision || NUL || task || NUL ||
  canonical_terminal_json)`. Bare task IDs are never replay/resume keys.
- One representative per task, selected before replay and after all Boolean
  gates by the ascending tuple `(first_edit_index, max_read_streak,
  normalized_command_count, rendered_token_count,
  canonical_terminal_sha256)`. A missing first edit is rejected before sorting.
- Reconstruct only canonical typed `WriteOp` and `EditOp` objects produced by
  `teacher_platform.fable5_import.parse_fable_tool_call`. Transcript Bash is
  evidence only and is never executed to create candidate state.
- Bash parsing has exactly two accepted types. `ReadOnlyBashOp` is statically
  confined and non-mutating. `VerifierEvidenceOp` equals the injected pinned
  seed `verify_cmd` byte-for-byte as a UTF-8 string; no trimming, line-ending,
  quoting, or whitespace normalization is permitted. It remains
  trusted build/test evidence but never affects reconstruction. All other Bash
  raises `ambiguous_bash_mutation`.
- Reject seed symlinks, hardlink anomalies, devices, FIFOs, sockets, submodules,
  non-regular protected files, escaping paths, protected aliases, and ambiguous
  source mutations.
- Missing seed `verify_timeout` uses policy default 300 seconds. Explicit values
  must be positive non-Boolean integers no greater than 1800. Record both source
  and effective values.
- Two independently materialized candidate runs must pass. Record each raw
  bounded output hash; raw hashes are not required to match.
- Required controls: untouched seed baseline fails; confined pinned reference
  patch passes. Optional deterministic corruption is diagnostic only. Controls
  are tainted and can never enter training output.
- No host verifier execution, host-root bind, host mount, Docker socket, network,
  image pull, GPU use, service change, broad process kill, broad prune, or
  production-port change.

## Canonical hashing

- Canonical JSON is UTF-8, NFC-normalized strings, sorted keys, compact
  separators, `allow_nan=false`, and a single trailing newline only for JSONL.
- Inventory paths are normalized POSIX relative paths sorted by UTF-8 bytes;
  reject undecodable/non-NFC names. Hash path, regular-file mode masked to
  `0o777`, byte length, and file bytes. Empty directories do not contribute.
- Candidate operation hash covers ordered typed operation JSON. Candidate tree
  and diff hashes exclude temporary roots and verifier-generated state and are
  computed before verification. Line endings and file bytes are never changed.

---

### Task 1: Immutable seed contract and declarative reconstruction

**Files:**
- Create: `teacher_platform/fable5_replay.py`
- Create: `teacher_platform/tests/test_fable5_replay.py`
- Modify only if a failing integration test proves a shared-parser defect:
  `teacher_platform/fable5_import.py`,
  `teacher_platform/tests/test_fable5_import.py`

**Interfaces:**
- `GitSeedSource(repo, commit)`
- `materialize_seed(task, destination) -> SeedContract`
- `preflight_reference_patch(contract) -> ReferencePatchContract`
- `canonical_mutation_plan(messages) -> MutationPlan`
- `reconstruct_candidate(contract, plan, destination) -> CandidateState`

- [ ] **Step 1: Write failing seed/materialization tests**

Cover exact Git commit enforcement with an injected archive reader, safe tar
member extraction, task ID/path confinement, required task fields, missing and
bounded timeouts, protected inventory, modes/hashes, symlinks, hardlinks,
special files, duplicate/missing protected files, canonical hashes, missing or
invalid reference patches, and deterministic `trajectory_id`.

Cover real-shape Fable native-object arguments, `Write`, exact-one `Edit`,
`replace_all`, parallel call order, protected aliases, arbitrary Bash mutation,
and candidates whose final state depends on Bash. Replay must consume the same
typed operation objects as the importer, not parse raw arguments independently.
With injected seed verifier strings, prove representative exact Python, Rust
(`cargo test --offline`), and C++ verifier calls become
`VerifierEvidenceOp`; prove `cargo fmt`, `make clean`, arbitrary scripts,
redirections, substitutions, and interpreter snippets reject.
Also prove leading/trailing spaces, CRLF/LF changes, repeated spaces, quoting
changes, and a trailing newline do not match the pinned verifier string; record
the exact verifier UTF-8 SHA-256 in the operation and run contracts.

```python
def test_missing_timeout_uses_recorded_policy_default(seed_repo: Path) -> None:
    contract = materialize_fixture(seed_repo, task_json={"verify_cmd": "pytest -q"})
    assert contract.source_verify_timeout is None
    assert contract.effective_verify_timeout == 300


def test_replay_and_import_share_canonical_edit_operation() -> None:
    call = real_fable_edit_call("src/lib.rs", "old", "new")
    operation = parse_fable_tool_call(call, protected_paths=frozenset())
    plan = canonical_mutation_plan(messages_with(call))
    assert plan.operations == (operation,)
```

- [ ] **Step 2: Run focused tests and verify RED**

```bash
PYTHONPATH=$PWD .venv/bin/pytest teacher_platform/tests/test_fable5_replay.py -q
```

Expected: collection fails because the replay module is absent.

- [ ] **Step 3: Implement immutable materialization and reconstruction**

Use `git archive --format=tar <exact-commit> -- tasks/seeds/<task>` through an
argv subprocess boundary or injected bytes in tests. Validate every tar member
before extraction; never trust or read the mutable worktree. Record Git commit,
task tree/object identity, and inventory hash.

Apply typed writes/edits with no-follow path checks, revalidate every ancestor
immediately before mutation, and compare canonical target identity against the
canonical protected set. Do not collect transcript verify commands; the only
executable verifier is the pinned seed `verify_cmd`. Pass that exact command to
the shared parser so matching transcript calls become `VerifierEvidenceOp`.

Reference-patch preflight rejects absolute/parent paths, binary patches,
symlink/submodule/rename/copy/mode-only changes, and protected paths, then runs
`git apply --check` only in a disposable fixture copy.

- [ ] **Step 4: Run focused tests and verify GREEN**

Run Step 2 plus `teacher_platform/tests/test_fable5_import.py`.

- [ ] **Step 5: Commit Task 1 code only**

```bash
git add teacher_platform/fable5_replay.py teacher_platform/tests/test_fable5_replay.py \
  teacher_platform/fable5_import.py teacher_platform/tests/test_fable5_import.py
git commit -m "feat: reconstruct pinned Fable candidates"
```

---

### Task 2: Restricted Docker verification, controls, and evidence

**Files:**
- Modify: `teacher_platform/fable5_replay.py`
- Modify: `teacher_platform/tests/test_fable5_replay.py`

**Interfaces:**
- `RestrictedExecutor` protocol
- `DockerPolicy` and `DockerExecutor`
- `functional_admission_probe(policy, language) -> AdmissionEvidence`
- `verify_candidate(...) -> ReplayEvidence`
- `run_control_set(...) -> ControlSetEvidence`

- [ ] **Step 1: Write failing executor/evidence tests**

Test exact `docker create/cp/start/exec/cp/stop/logs/inspect/rm` argv, immutable digest
requirement, no-pull behavior, no mounts, network/read-only/capability/security,
the narrowly scoped capability-free trusted root-wrapper exception, distinct
non-root verifier UID/GID, tmpfs/CPU/memory/PID/file-size bounds, output cap, wall timeout, exact
CID ownership, descendant cleanup, disk-floor admission, toolchain probe,
candidate copied rather than mounted, and cleanup on every failure.

Test the result-channel lifecycle: trusted wrapper and untrusted verifier use
different UIDs; result directory is wrapper-only; wrapper PID 1 stays alive
after writing a schema-bound bounded result; host polls an exact-CID done marker,
copies the result while the container is running, validates it, then stops and
removes only that CID. Cover wrapper/verifier rc separation, malformed/missing
result, verifier attempts to forge result, timeout, and no result reuse.

Test two fresh candidate materializations, stable pre-verifier candidate hashes,
independent successful output hashes, nondeterministic raw output acceptance,
protected drift, empty source diff, failing baseline, passing reference,
candidate pass/fail, control taint, and optional corrupt evidence.

- [ ] **Step 2: Run focused tests and verify RED**

```bash
PYTHONPATH=$PWD .venv/bin/pytest teacher_platform/tests/test_fable5_replay.py -q \
  -k 'executor or admission or verify or control or evidence or cleanup'
```

- [ ] **Step 3: Implement Docker-only execution and versioned evidence**

Docker admission requires an exact locally cached `image@sha256:...`, then runs
the complete production policy plus the required language executable. Create a
container with no host mounts; copy immutable fixture/candidate input into the
stopped container, run with read-only root and writable tmpfs work/scratch/result,
and use a fixed trusted PID-1 wrapper. The capability-free wrapper starts as the
container root only to create UID-separated directories, drops the exact pinned
verifier to a non-root UID, hashes post-run state itself, writes a schema-bound
bounded result to a wrapper-only directory, then remains alive. The host polls
and copies that result from the running exact CID before stop/removal. Verifier
stdout cannot serve as or overwrite the result record. Never execute a
transcript command.

Every execution is bounded by `--network none`, `--read-only`, `--cap-drop ALL`,
and `no-new-privileges`. The trusted PID-1 wrapper is the sole root exception and
may only set up UID-separated directories, drop the verifier to its pinned
non-root UID/GID, hash state, and publish the result. CPU, memory, PID,
tmpfs/file-size, and wall limits apply to the whole container. Capture output
incrementally to a mode-0600 bounded file. Kill/remove
only the exact CID; verify it and descendants are gone. Preserve the 40-GiB
Docker floor before and after.

`ReplayEvidence` schema v2 includes at minimum: trajectory/task/language;
dataset/Git/tree/inventory/terminal/operation/candidate/diff hashes;
protected hashes before/after per run; verifier text/hash; source/effective
timeout; executor policy version, image digest and runtime version;
run-contract hash; per-run rc, duration, termination, raw output/log hash,
pre/post candidate state; resolved/failure class/control identity; cleanup state;
resource peaks. Only `control_identity="candidate"` is trainable.

Fresh run 2 is rebuilt from the Git object plus operation plan, never copied
from run 1. Both must return zero and preserve protected files. Raw output hashes
may differ and are diagnostics only.

- [ ] **Step 4: Run all replay/importer tests and verify GREEN**

```bash
PYTHONPATH=$PWD .venv/bin/pytest \
  teacher_platform/tests/test_fable5_replay.py \
  teacher_platform/tests/test_fable5_import.py \
  teacher_platform/tests/test_teacher_platform.py -q
```

- [ ] **Step 5: Commit Task 2 code only**

```bash
git add teacher_platform/fable5_replay.py teacher_platform/tests/test_fable5_replay.py
git commit -m "feat: verify Fable candidates in restricted Docker"
```

---

### Task 3: Locked replay ledger, eligibility inventory, and smoke

**Files:**
- Modify: `teacher_platform/fable5_replay.py`
- Modify: `teacher_platform/fable5_import.py`
- Modify: `teacher_platform/tests/test_fable5_replay.py`
- Modify: `teacher_platform/tests/test_fable5_import.py`

**Interfaces:**
- CLI `python -m teacher_platform.fable5_replay`
- atomic `replay.jsonl`, control logs, and eligibility/smoke manifests
- importer join accepting only exact non-control resolved evidence

- [ ] **Step 1: Write failing ledger/integration tests**

Cover exclusive run lock, same-filesystem temporary files, mode 0600, atomic
log-before-ledger publication, full existing-ledger validation, duplicate
trajectory rejection, run-contract mismatch, resume, rejection/timeout attempt
records, control/train separation, importer evidence hash join, exact manifest
arithmetic, and refusal to publish any control row.

- [ ] **Step 2: Run integration tests and verify RED**

```bash
PYTHONPATH=$PWD .venv/bin/pytest \
  teacher_platform/tests/test_fable5_replay.py \
  teacher_platform/tests/test_fable5_import.py -q \
  -k 'ledger or lock or publish or resume or manifest or join'
```

- [ ] **Step 3: Implement CLI, locked ledger, and eligibility inventory**

Before runtime replay, stream the structural sidecar and publish a manifest for
the exact intersection of: supported canonical operations, decontamination,
valid Git seed, valid reference patch, language digest present, and functional
executor/toolchain admission. Include every exclusion reason and compute the
eligible ceiling.

Git-object admission is explicit. First require a local repository containing
the object and verify `git cat-file -e <commit>^{commit}`, the commit ID, and
the `tasks/seeds` tree ID. If the remote lacks it, create a bare Git bundle from
the locally verified repository, record its SHA-256, copy that bundle to a new
dedicated remote path, initialize/fetch only from the bundle, and repeat the
commit/tree checks. This one-time authorized source acquisition occurs before
container admission; task verification remains network-disabled. Any missing
or mismatched object stops before a container starts.

Select a deterministic smoke manifest of exactly two Python, two Rust, and one
C++ trajectory from this intersection. The smoke CLI accepts this explicit
manifest; it never discovers an arbitrary `2/2/1` at runtime.

- [ ] **Step 4: Run tests and remote admission inventory**

```bash
PYTHONPATH=$PWD .venv/bin/pytest \
  teacher_platform/tests/test_fable5_replay.py \
  teacher_platform/tests/test_fable5_import.py -q
ssh ironbccllm 'uname -m; free -g; df -BG /var/lib/docker; docker image ls --digests'
```

Inventory is read-only. Do not pull, prune, start a verifier container, or
materialize task data until exact digest policies and the Git bundle/object
admission record are reviewed.

- [ ] **Step 5: Functionally admit each required exact digest**

For each smoke language, run one harmless bounded container with the complete
policy and toolchain probe. Record the exact CID, digest, command, rc, output
hash, resource bounds, cleanup result, and final Docker free space. Any failure
stops before task verification.

- [ ] **Step 6: Materialize the pinned Git object and run five controls**

Materialize only the five manifest-pinned seed tasks from the exact commit.
For each, require baseline fail, reference pass, candidate pass twice, protected
hash stability, nonempty clean diff, and exact cleanup. Logs and ledgers are
0600. Any failure stops scaling.

- [ ] **Step 7: Join verified candidates and run format gate**

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

PASS: five candidate rows, zero controls, `failure_count=0`, and all hashes join
exactly.

- [ ] **Step 8: Commit Task 3 code only**

```bash
git add teacher_platform/fable5_replay.py teacher_platform/fable5_import.py \
  teacher_platform/tests/test_fable5_replay.py teacher_platform/tests/test_fable5_import.py
git commit -m "feat: join independently verified Fable evidence"
```

## Scale gate

After the five-task smoke, the full CPU build may be proposed only if the
pre-replay eligible ceiling can satisfy all immutable floors: 150 unique tasks,
including 100 Python, 30 Rust, and 12 C++, at least 70% edit by command ten,
median first edit at most eight, and zero format failures. Missing any floor
banks the converter and reports the honest yield; no safety, behavior,
contamination, or language floor is relaxed. No mixture or training launch is
authorized by this plan.
