"""Tests for immutable Fable seed reconstruction."""

from __future__ import annotations

import hashlib
import io
import json
import os
import stat
import subprocess
import sys
import tarfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fable5_import import (  # noqa: E402
    DATASET_REVISION,
    FableEditOp,
    FableWriteOp,
    UnsupportedTrajectoryTool,
    VerifierEvidenceOp,
    parse_fable_tool_call,
)
from fable5_replay import (  # noqa: E402
    MOONSHINER_REVISION,
    CandidateState,
    GitSeedSource,
    MutationPlan,
    ReplayContractError,
    canonical_mutation_plan,
    materialize_seed,
    preflight_reference_patch,
    reconstruct_candidate,
    trajectory_identity,
)


TASK = "py-safe"
VERIFY_CMD = "python3 -m pytest -q"


def _task_json(**updates: object) -> bytes:
    payload: dict[str, object] = {
        "id": TASK,
        "lang": "python",
        "verify_cmd": VERIFY_CMD,
        "test_files": ["test_src.py"],
    }
    payload.update(updates)
    return json.dumps(payload).encode()


def _patch(target: str = "src.py") -> bytes:
    return (
        f"diff --git a/{target} b/{target}\n"
        f"--- a/{target}\n"
        f"+++ b/{target}\n"
        "@@ -1 +1 @@\n"
        "-old\n"
        "+new\n"
    ).encode()


def _archive(
    *,
    task_json: bytes | None = None,
    patch: bytes | None = None,
    source: bytes = b"old\n",
    source_mode: int = 0o644,
    extra_members: tuple[tuple[str, bytes, int, bytes], ...] = (),
) -> bytes:
    """Build a Git-archive-shaped tar; tuple entries are name/data/type/linkname."""

    prefix = f"tasks/seeds/{TASK}"
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w") as archive:
        for directory in (prefix, f"{prefix}/files"):
            info = tarfile.TarInfo(directory)
            info.type = tarfile.DIRTYPE
            info.mode = 0o755
            archive.addfile(info)
        files = (
            (f"{prefix}/task.json", task_json or _task_json(), 0o644),
            (f"{prefix}/reference_fix.patch", patch or _patch(), 0o644),
            (f"{prefix}/files/src.py", source, source_mode),
            (f"{prefix}/files/test_src.py", b"assert True\n", 0o644),
        )
        for name, data, mode in files:
            info = tarfile.TarInfo(name)
            info.size = len(data)
            info.mode = mode
            archive.addfile(info, io.BytesIO(data))
        for name, data, kind, linkname in extra_members:
            info = tarfile.TarInfo(name)
            info.type = kind
            info.linkname = linkname.decode()
            info.size = len(data) if kind == tarfile.REGTYPE else 0
            archive.addfile(info, io.BytesIO(data) if data else None)
    return buffer.getvalue()


def _tree_entries(
    *,
    source_mode: int = 0o644,
    extra: tuple[tuple[str, str, str, str], ...] = (),
) -> bytes:
    prefix = f"tasks/seeds/{TASK}"
    rows = [
        ("100644", "blob", "1" * 40, f"{prefix}/task.json"),
        ("100644", "blob", "2" * 40, f"{prefix}/reference_fix.patch"),
        (f"100{source_mode:o}", "blob", "3" * 40, f"{prefix}/files/src.py"),
        ("100644", "blob", "4" * 40, f"{prefix}/files/test_src.py"),
        *extra,
    ]
    return b"".join(
        f"{mode} {kind} {oid}\t{path}".encode() + b"\0"
        for mode, kind, oid, path in rows
    )


class FakeGit:
    def __init__(self, archive: bytes, *, tree_entries: bytes | None = None) -> None:
        self.archive = archive
        self.tree_entries = tree_entries or _tree_entries()
        self.calls: list[tuple[str, ...]] = []

    def __call__(self, argv: tuple[str, ...]) -> bytes:
        self.calls.append(argv)
        tail = argv[3:]
        if tail == ("cat-file", "-e", f"{MOONSHINER_REVISION}^{{commit}}"):
            return b""
        if tail == ("rev-parse", f"{MOONSHINER_REVISION}^{{commit}}"):
            return f"{MOONSHINER_REVISION}\n".encode()
        if tail == ("rev-parse", f"{MOONSHINER_REVISION}^{{tree}}"):
            return ("a" * 40 + "\n").encode()
        if tail == (
            "rev-parse",
            f"{MOONSHINER_REVISION}:tasks/seeds/{TASK}",
        ):
            return ("b" * 40 + "\n").encode()
        if tail == (
            "ls-tree",
            "-rz",
            "-r",
            MOONSHINER_REVISION,
            "--",
            f"tasks/seeds/{TASK}",
        ):
            return self.tree_entries
        if tail == (
            "archive",
            "--format=tar",
            MOONSHINER_REVISION,
            "--",
            f"tasks/seeds/{TASK}",
        ):
            return self.archive
        raise AssertionError(f"unexpected Git command: {argv!r}")


def _source(archive: bytes | None = None, *, tree_entries: bytes | None = None) -> GitSeedSource:
    return GitSeedSource(
        Path("/immutable/moonshiner.git"),
        MOONSHINER_REVISION,
        runner=FakeGit(archive or _archive(), tree_entries=tree_entries),
    )


def _materialized(tmp_path: Path, archive: bytes | None = None):
    return materialize_seed(_source(archive), TASK, tmp_path / "seed")


def _call(name: str, arguments: object, call_id: str = "call-0") -> dict[str, object]:
    return {
        "id": call_id,
        "type": "function",
        "function": {"name": name, "arguments": arguments},
    }


def _messages(*calls: dict[str, object]) -> list[dict[str, object]]:
    return [
        {"role": "system", "content": "system"},
        {"role": "user", "content": "fix it"},
        {"role": "assistant", "content": "plan", "tool_calls": list(calls)},
    ]


def test_git_seed_source_requires_the_pinned_commit() -> None:
    with pytest.raises(ValueError, match="pinned Moonshiner commit"):
        GitSeedSource(Path("/repo"), "main")


def test_materialize_seed_admits_exact_commit_and_records_object_ids(tmp_path: Path) -> None:
    fake = FakeGit(_archive())
    source = GitSeedSource(Path("/immutable/moonshiner.git"), MOONSHINER_REVISION, fake)

    contract = materialize_seed(source, TASK, tmp_path / "seed")

    assert contract.task == TASK
    assert contract.language == "python"
    assert contract.source_commit_sha == MOONSHINER_REVISION
    assert contract.source_tree_sha == "a" * 40
    assert contract.task_tree_sha == "b" * 40
    assert contract.verify_cmd == VERIFY_CMD
    assert contract.verifier_sha256 == hashlib.sha256(VERIFY_CMD.encode()).hexdigest()
    assert contract.protected_paths == ("test_src.py",)
    assert contract.files_root == tmp_path / "seed" / "files"
    assert contract.fixture_sha256 == contract.inventory_sha256
    assert (contract.files_root / "src.py").read_bytes() == b"old\n"
    assert any(call[3] == "cat-file" for call in fake.calls)
    assert any(call[3] == "archive" for call in fake.calls)


def test_missing_timeout_uses_recorded_policy_default(tmp_path: Path) -> None:
    contract = _materialized(tmp_path)
    assert contract.source_verify_timeout is None
    assert contract.effective_verify_timeout == 300


@pytest.mark.parametrize("value", [True, False, 0, -1, 1801, 1.5, "300"])
def test_explicit_timeout_is_a_bounded_non_boolean_integer(
    tmp_path: Path, value: object
) -> None:
    with pytest.raises(ReplayContractError, match="verify_timeout"):
        _materialized(tmp_path, _archive(task_json=_task_json(verify_timeout=value)))


def test_explicit_timeout_records_source_and_effective_value(tmp_path: Path) -> None:
    contract = _materialized(
        tmp_path, _archive(task_json=_task_json(verify_timeout=1800))
    )
    assert contract.source_verify_timeout == 1800
    assert contract.effective_verify_timeout == 1800


@pytest.mark.parametrize(
    "task_json",
    [
        _task_json(id="wrong"),
        _task_json(lang="go"),
        _task_json(verify_cmd=""),
        _task_json(test_files=[]),
        _task_json(test_files=["../secret"]),
        _task_json(test_files=["missing.py"]),
        _task_json(test_files=["test_src.py", "test_src.py"]),
    ],
)
def test_seed_contract_rejects_invalid_required_fields(
    tmp_path: Path, task_json: bytes
) -> None:
    with pytest.raises(ReplayContractError):
        _materialized(tmp_path, _archive(task_json=task_json))


def test_materialization_preserves_file_mode_and_hashes_bytes(tmp_path: Path) -> None:
    first = materialize_seed(
        _source(
            _archive(source=b"\xef\xbb\xbfcrlf\r\n", source_mode=0o755),
            tree_entries=_tree_entries(source_mode=0o755),
        ),
        TASK,
        tmp_path / "seed",
    )
    assert stat.S_IMODE((first.files_root / "src.py").stat().st_mode) == 0o755
    assert (first.files_root / "src.py").read_bytes() == b"\xef\xbb\xbfcrlf\r\n"

    other_root = tmp_path / "other"
    second = materialize_seed(
        _source(
            _archive(source=b"\xef\xbb\xbfcrlf\n", source_mode=0o755),
            tree_entries=_tree_entries(source_mode=0o755),
        ),
        TASK,
        other_root,
    )
    assert first.inventory_sha256 != second.inventory_sha256


def test_inventory_hash_frames_sorted_path_mode_length_and_raw_bytes(tmp_path: Path) -> None:
    contract = _materialized(tmp_path)
    digest = hashlib.sha256()
    for path, mode, data in (
        ("src.py", 0o644, b"old\n"),
        ("test_src.py", 0o644, b"assert True\n"),
    ):
        encoded_path = path.encode("utf-8")
        digest.update(len(encoded_path).to_bytes(8, "big"))
        digest.update(encoded_path)
        digest.update(mode.to_bytes(4, "big"))
        digest.update(len(data).to_bytes(8, "big"))
        digest.update(data)
    assert contract.inventory_sha256 == digest.hexdigest()


@pytest.mark.parametrize(
    ("name", "kind", "linkname", "match"),
    [
        (f"tasks/seeds/{TASK}/files/link", tarfile.SYMTYPE, b"src.py", "symlink"),
        (f"tasks/seeds/{TASK}/files/hard", tarfile.LNKTYPE, b"src.py", "hardlink"),
        (f"tasks/seeds/{TASK}/files/fifo", tarfile.FIFOTYPE, b"", "special"),
        (f"tasks/seeds/{TASK}/files/../escape", tarfile.REGTYPE, b"", "parent"),
        (f"/tasks/seeds/{TASK}/files/abs", tarfile.REGTYPE, b"", "absolute"),
    ],
)
def test_archive_rejects_unsafe_members_before_extraction(
    tmp_path: Path, name: str, kind: bytes, linkname: bytes, match: str
) -> None:
    archive = _archive(extra_members=((name, b"x", kind, linkname),))
    with pytest.raises(ReplayContractError, match=match):
        _materialized(tmp_path, archive)
    assert not (tmp_path / "escape").exists()


def test_archive_rejects_duplicate_and_file_directory_collisions(tmp_path: Path) -> None:
    prefix = f"tasks/seeds/{TASK}/files"
    duplicate = _archive(
        extra_members=((f"{prefix}/src.py", b"again", tarfile.REGTYPE, b""),)
    )
    with pytest.raises(ReplayContractError, match="duplicate"):
        _materialized(tmp_path, duplicate)

    collision = _archive(
        extra_members=((f"{prefix}/src.py/child", b"x", tarfile.REGTYPE, b""),)
    )
    with pytest.raises(ReplayContractError, match="collision"):
        materialize_seed(_source(collision), TASK, tmp_path / "collision")


def test_git_tree_rejects_symlink_and_submodule_modes_before_archive(tmp_path: Path) -> None:
    prefix = f"tasks/seeds/{TASK}/files"
    for entry, match in (
        (("120000", "blob", "5" * 40, f"{prefix}/link"), "symlink"),
        (("160000", "commit", "6" * 40, f"{prefix}/sub"), "submodule"),
    ):
        with pytest.raises(ReplayContractError, match=match):
            materialize_seed(
                _source(_archive(), tree_entries=_tree_entries(extra=(entry,))),
                TASK,
                tmp_path / match,
            )


def test_archive_rejects_non_nfc_member_name(tmp_path: Path) -> None:
    decomposed = f"tasks/seeds/{TASK}/files/cafe\u0301.py"
    with pytest.raises(ReplayContractError, match="NFC"):
        _materialized(
            tmp_path,
            _archive(extra_members=((decomposed, b"x", tarfile.REGTYPE, b""),)),
        )


def test_archive_rejects_noncanonical_empty_path_components(tmp_path: Path) -> None:
    archive = _archive(
        extra_members=((
            f"tasks/seeds/{TASK}/files/pkg//extra.py",
            b"x",
            tarfile.REGTYPE,
            b"",
        ),)
    )
    tree = _tree_entries(
        extra=((
            "100644",
            "blob",
            "5" * 40,
            f"tasks/seeds/{TASK}/files/pkg/extra.py",
        ),)
    )
    with pytest.raises(ReplayContractError, match="canonical"):
        materialize_seed(
            _source(archive, tree_entries=tree), TASK, tmp_path / "seed"
        )


def test_materialize_refuses_existing_destination(tmp_path: Path) -> None:
    destination = tmp_path / "seed"
    destination.mkdir()
    with pytest.raises(ReplayContractError, match="destination already exists"):
        materialize_seed(_source(), TASK, destination)


def test_replay_and_import_share_canonical_edit_operation() -> None:
    call = _call(
        "Edit",
        {
            "file_path": "src.py",
            "old_string": "old",
            "new_string": "new",
            "replace_all": False,
        },
    )
    operation = parse_fable_tool_call(call, protected_paths=frozenset())
    plan = canonical_mutation_plan(_messages(call), frozenset(), VERIFY_CMD)
    assert plan.operations == (operation,)


def test_plan_accepts_native_object_and_json_arguments_in_parallel_order() -> None:
    write = _call(
        "Write", {"file_path": "a.py", "content": "a"}, "call-write"
    )
    edit = _call(
        "Edit",
        json.dumps(
            {"file_path": "src.py", "old_string": "old", "new_string": "new"}
        ),
        "call-edit",
    )
    plan = canonical_mutation_plan(_messages(write, edit), frozenset(), VERIFY_CMD)
    assert [type(op) for op in plan.operations] == [FableWriteOp, FableEditOp]
    assert [op.path for op in plan.operations] == ["/testbed/a.py", "/testbed/src.py"]
    assert len(plan.operation_sha256) == 64
    assert plan.verifier_sha256 == hashlib.sha256(VERIFY_CMD.encode()).hexdigest()


def test_plan_retains_only_declarative_mutations_and_exact_verifier_evidence() -> None:
    read = _call("Read", {"file_path": "src.py"}, "read")
    verifier = _call("Bash", {"command": VERIFY_CMD}, "verify")
    write = _call("Write", {"file_path": "new.py", "content": "x"}, "write")
    plan = canonical_mutation_plan(
        _messages(read, verifier, write), frozenset(), VERIFY_CMD
    )
    assert len(plan.operations) == 1
    assert isinstance(plan.operations[0], FableWriteOp)
    assert plan.verifier_evidence_count == 1
    parsed = parse_fable_tool_call(
        verifier,
        frozenset(),
        trusted_verifier_commands=frozenset({VERIFY_CMD}),
    )
    assert isinstance(parsed, VerifierEvidenceOp)
    assert parsed.command_sha256 == plan.verifier_sha256


@pytest.mark.parametrize(
    "command",
    [
        f" {VERIFY_CMD}",
        f"{VERIFY_CMD} ",
        f"{VERIFY_CMD}\n",
        VERIFY_CMD.replace(" ", "  ", 1),
        "python3 -m pytest '-q'",
        "cargo fmt",
        "make clean",
        "./script.sh",
        "echo x > src.py",
        "echo $(pwd)",
        "python3 -c 'open(\"src.py\", \"w\").write(\"x\")'",
    ],
)
def test_plan_rejects_verifier_whitespace_mismatch_and_ambiguous_bash(
    command: str,
) -> None:
    with pytest.raises(UnsupportedTrajectoryTool, match="ambiguous_bash_mutation"):
        canonical_mutation_plan(
            _messages(_call("Bash", {"command": command})),
            frozenset(),
            VERIFY_CMD,
        )


@pytest.mark.parametrize(
    "verify_cmd",
    [
        "python3 -m pytest -q",
        "cargo test --offline",
        "cmake --build build && ctest --test-dir build",
    ],
)
def test_exact_language_verifier_is_evidence_not_a_mutation(verify_cmd: str) -> None:
    plan = canonical_mutation_plan(
        _messages(_call("Bash", {"command": verify_cmd})),
        frozenset(),
        verify_cmd,
    )
    assert plan.operations == ()
    assert plan.verifier_evidence_count == 1


def test_plan_rejects_protected_mutation() -> None:
    with pytest.raises(UnsupportedTrajectoryTool, match="protected"):
        canonical_mutation_plan(
            _messages(_call("Write", {"file_path": "test_src.py", "content": "x"})),
            frozenset({"test_src.py"}),
            VERIFY_CMD,
        )


def test_trajectory_identity_uses_revision_task_and_canonical_terminal_json() -> None:
    row = {"task": TASK, "messages": [{"content": "é", "role": "assistant"}]}
    canonical = json.dumps(
        row,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode()
    expected = hashlib.sha256(
        DATASET_REVISION.encode() + b"\0" + TASK.encode() + b"\0" + canonical
    ).hexdigest()
    assert trajectory_identity(TASK, row) == expected
    assert trajectory_identity("other", row) != expected


def test_trajectory_identity_rejects_keys_that_collide_after_nfc() -> None:
    with pytest.raises(ReplayContractError, match="canonical JSON key"):
        trajectory_identity(TASK, {"e\u0301": 1, "é": 2})


def test_reconstruct_write_and_edit_are_byte_exact_and_preserve_mode(tmp_path: Path) -> None:
    contract = materialize_seed(
        _source(
            _archive(source=b"\xef\xbb\xbfold\r\nold", source_mode=0o755),
            tree_entries=_tree_entries(source_mode=0o755),
        ),
        TASK,
        tmp_path / "seed",
    )
    calls = (
        _call(
            "Edit",
            {
                "file_path": "src.py",
                "old_string": "old",
                "new_string": "néw",
                "replace_all": True,
            },
            "edit",
        ),
        _call("Write", {"file_path": "new.py", "content": "尾\r\n"}, "write"),
    )
    plan = canonical_mutation_plan(_messages(*calls), frozenset(), VERIFY_CMD)

    state = reconstruct_candidate(contract, plan, tmp_path / "candidate")

    assert isinstance(state, CandidateState)
    assert (state.root / "src.py").read_bytes() == "\ufeffnéw\r\nnéw".encode()
    assert (state.root / "new.py").read_bytes() == "尾\r\n".encode()
    assert stat.S_IMODE((state.root / "src.py").stat().st_mode) == 0o755
    assert state.changed_paths == ("new.py", "src.py")
    assert len(state.tree_sha256) == len(state.diff_sha256) == 64


def test_new_write_mode_is_deterministic_across_process_umask(tmp_path: Path) -> None:
    contract = _materialized(tmp_path)
    plan = canonical_mutation_plan(
        _messages(_call("Write", {"file_path": "new.py", "content": "x"})),
        frozenset(),
        VERIFY_CMD,
    )
    original_umask = os.umask(0o077)
    try:
        state = reconstruct_candidate(contract, plan, tmp_path / "candidate")
    finally:
        os.umask(original_umask)
    assert stat.S_IMODE((state.root / "new.py").stat().st_mode) == 0o644


@pytest.mark.parametrize("replace_all", [False, True])
def test_edit_validates_match_count_before_mutating(
    tmp_path: Path, replace_all: bool
) -> None:
    archive = _archive(source=b"old old\n") if not replace_all else _archive()
    contract = _materialized(tmp_path, archive)
    old = "missing" if replace_all else "old"
    plan = MutationPlan(
        operations=(
            FableEditOp("/testbed/src.py", old, "x", replace_all, ()),
        ),
        operation_sha256="0" * 64,
        verifier_sha256=contract.verifier_sha256,
        verifier_evidence_count=0,
    )
    destination = tmp_path / "candidate"
    with pytest.raises(ReplayContractError, match="match"):
        reconstruct_candidate(contract, plan, destination)
    assert (destination / "src.py").read_bytes() == (
        b"old old\n" if not replace_all else b"old\n"
    )


def test_reconstruction_requires_nonempty_source_diff(tmp_path: Path) -> None:
    contract = _materialized(tmp_path)
    no_change = canonical_mutation_plan(
        _messages(_call("Write", {"file_path": "src.py", "content": "old\n"})),
        frozenset(),
        VERIFY_CMD,
    )
    with pytest.raises(ReplayContractError, match="empty candidate diff"):
        reconstruct_candidate(contract, no_change, tmp_path / "candidate")


def test_reconstruction_detects_seed_drift_before_copy(tmp_path: Path) -> None:
    contract = _materialized(tmp_path)
    (contract.files_root / "src.py").write_text("tampered")
    plan = canonical_mutation_plan(
        _messages(_call("Write", {"file_path": "src.py", "content": "new"})),
        frozenset(),
        VERIFY_CMD,
    )
    with pytest.raises(ReplayContractError, match="immutable seed inventory changed"):
        reconstruct_candidate(contract, plan, tmp_path / "candidate")


def test_reconstruction_revalidates_symlinks_and_hardlinks_before_each_mutation(
    tmp_path: Path,
) -> None:
    contract = _materialized(tmp_path)
    first = FableWriteOp("/testbed/alias.py", "x", ())
    second = FableEditOp("/testbed/src.py", "old", "new", False, ())
    plan = MutationPlan(
        operations=(first, second),
        operation_sha256="0" * 64,
        verifier_sha256=contract.verifier_sha256,
        verifier_evidence_count=0,
    )
    destination = tmp_path / "candidate"

    def introduce_alias(operation_index: int, root: Path) -> None:
        if operation_index == 1:
            (root / "alias.py").unlink()
            os.link(root / "src.py", root / "alias.py")

    with pytest.raises(ReplayContractError, match="hardlink"):
        reconstruct_candidate(contract, plan, destination, before_operation=introduce_alias)


def test_reconstruction_rejects_manual_protected_or_escaping_plan(tmp_path: Path) -> None:
    contract = _materialized(tmp_path)
    for path, match in (("/testbed/test_src.py", "protected"), ("/etc/passwd", "outside")):
        plan = MutationPlan(
            operations=(FableWriteOp(path, "bad", ()),),
            operation_sha256="0" * 64,
            verifier_sha256=contract.verifier_sha256,
            verifier_evidence_count=0,
        )
        with pytest.raises(ReplayContractError, match=match):
            reconstruct_candidate(contract, plan, tmp_path / hashlib.sha256(path.encode()).hexdigest())


def test_reconstruction_hashes_are_deterministic_across_fresh_destinations(
    tmp_path: Path,
) -> None:
    contract = _materialized(tmp_path)
    plan = canonical_mutation_plan(
        _messages(_call("Edit", {"file_path": "src.py", "old_string": "old", "new_string": "new"})),
        frozenset(),
        VERIFY_CMD,
    )
    one = reconstruct_candidate(contract, plan, tmp_path / "one")
    two = reconstruct_candidate(contract, plan, tmp_path / "two")
    assert (one.tree_sha256, one.diff_sha256) == (two.tree_sha256, two.diff_sha256)


class PatchExecutor:
    def __init__(self, returncode: int = 0) -> None:
        self.returncode = returncode
        self.calls: list[tuple[tuple[str, ...], Path]] = []

    def __call__(self, argv: tuple[str, ...], cwd: Path) -> subprocess.CompletedProcess[bytes]:
        self.calls.append((argv, cwd))
        assert (cwd / "src.py").read_bytes() == b"old\n"
        assert (cwd / "reference_fix.patch").is_file()
        return subprocess.CompletedProcess(argv, self.returncode, b"", b"bad patch")


def test_reference_patch_preflight_parses_then_checks_disposable_copy(
    tmp_path: Path,
) -> None:
    contract = _materialized(tmp_path)
    executor = PatchExecutor()
    result = preflight_reference_patch(contract, executor=executor)
    assert result.eligible is True
    assert result.targets == ("src.py",)
    assert result.patch_sha256 == hashlib.sha256(_patch()).hexdigest()
    assert executor.calls[0][0] == ("git", "apply", "--check", "--", "reference_fix.patch")
    assert executor.calls[0][1] != contract.files_root
    assert (contract.files_root / "src.py").read_bytes() == b"old\n"


def test_missing_reference_patch_is_a_stable_exclusion_without_executor(
    tmp_path: Path,
) -> None:
    contract = _materialized(tmp_path)
    contract.reference_patch_path.unlink()
    executor = PatchExecutor()
    result = preflight_reference_patch(contract, executor=executor)
    assert result.eligible is False
    assert result.exclusion_reason == "missing_reference_patch"
    assert executor.calls == []


@pytest.mark.parametrize(
    ("patch", "reason"),
    [
        (b"GIT binary patch\n", "binary"),
        (b"diff --git a/../x b/../x\n--- a/../x\n+++ b/../x\n", "parent"),
        (b"diff --git a/x b//abs\n--- a/x\n+++ b//abs\n", "absolute"),
        (b"diff --git a/x b/y\nsimilarity index 100%\nrename from x\nrename to y\n", "rename"),
        (b"diff --git a/x b/x\nold mode 100644\nnew mode 100755\n", "mode"),
        (
            b"diff --git a/src.py b/src.py\n--- a/src.py\n+++ b/src.py\n"
            b"@@ -1 +1 @@\n-old\n+one\n"
            b"diff --git a/src.py b/src.py\n--- a/src.py\n+++ b/src.py\n"
            b"@@ -1 +1 @@\n-old\n+two\n",
            "duplicate",
        ),
        (
            b"diff --git a/test_src.py b/test_src.py\n--- a/test_src.py\n+++ b/test_src.py\n",
            "protected",
        ),
        (
            b"diff --git a/vendor b/vendor\nindex 1234567..7654321 160000\n"
            b"--- a/vendor\n+++ b/vendor\n@@ -1 +1 @@\n"
            b"-Subproject commit 1234567\n+Subproject commit 7654321\n",
            "submodule",
        ),
        (
            b"diff --git a/src.py b/other.py\n--- a/src.py\n+++ b/other.py\n"
            b"@@ -1 +1 @@\n-old\n+new\n",
            "rename",
        ),
    ],
)
def test_reference_patch_rejects_unsafe_content_before_executor(
    tmp_path: Path, patch: bytes, reason: str
) -> None:
    contract = _materialized(tmp_path, _archive(patch=patch))
    executor = PatchExecutor()
    result = preflight_reference_patch(contract, executor=executor)
    assert result.eligible is False
    assert reason in (result.exclusion_reason or "")
    assert executor.calls == []


def test_reference_patch_apply_check_failure_is_stable_exclusion(tmp_path: Path) -> None:
    contract = _materialized(tmp_path)
    result = preflight_reference_patch(contract, executor=PatchExecutor(returncode=1))
    assert result.eligible is False
    assert result.exclusion_reason == "reference_patch_apply_check_failed"
