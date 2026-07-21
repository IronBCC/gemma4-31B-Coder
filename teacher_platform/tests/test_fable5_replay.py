"""Tests for immutable Fable seed reconstruction."""

from __future__ import annotations

import dataclasses
import fcntl
import hashlib
import io
import json
import os
import stat
import subprocess
import sys
import tarfile
from collections import Counter
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import fable5_replay as replay  # noqa: E402
from fable5_import import (  # noqa: E402
    DATASET_REVISION,
    FableEditOp,
    FableWriteOp,
    UnsupportedTrajectoryTool,
    VerifierEvidenceOp,
    convert_trajectory,
    parse_fable_tool_call,
    select_terminal_row,
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


def _git_blob_oid(data: bytes) -> str:
    return hashlib.sha1(
        b"blob " + str(len(data)).encode() + b"\0" + data
    ).hexdigest()


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
            (f"{prefix}/task.json", task_json or _task_json(), 0o664),
            (f"{prefix}/reference_fix.patch", patch or _patch(), 0o664),
            (f"{prefix}/files/src.py", source, source_mode | 0o020),
            (f"{prefix}/files/test_src.py", b"assert True\n", 0o664),
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
            if kind == tarfile.REGTYPE:
                info.mode = 0o664
            info.size = len(data) if kind == tarfile.REGTYPE else 0
            archive.addfile(info, io.BytesIO(data) if data else None)
    return buffer.getvalue()


def _tree_entries(
    *,
    source_mode: int = 0o644,
    extra: tuple[tuple[str, str, str, str], ...] = (),
) -> bytes:
    prefix = f"tasks/seeds/{TASK}"
    task_json = _task_json()
    patch = _patch()

    rows = [
        ("100644", "blob", _git_blob_oid(task_json), f"{prefix}/task.json"),
        ("100644", "blob", _git_blob_oid(patch), f"{prefix}/reference_fix.patch"),
        (
            f"100{source_mode:o}",
            "blob",
            _git_blob_oid(b"old\n"),
            f"{prefix}/files/src.py",
        ),
        (
            "100644",
            "blob",
            _git_blob_oid(b"assert True\n"),
            f"{prefix}/files/test_src.py",
        ),
        *extra,
    ]
    return b"".join(
        f"{mode} {kind} {oid}\t{path}".encode() + b"\0"
        for mode, kind, oid, path in rows
    )


class FakeGit:
    def __init__(self, archive: bytes, *, tree_entries: bytes | None = None) -> None:
        self.archive = archive
        self.tree_entries = tree_entries or _tree_entries_from_archive(archive)
        self.calls: list[tuple[tuple[str, ...], dict[str, str]]] = []

    def __call__(self, argv: tuple[str, ...], env: dict[str, str]) -> bytes:
        self.calls.append((argv, dict(env)))
        tail = argv[4:]
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
            "-c",
            "tar.umask=0002",
            "archive",
            "--format=tar",
            MOONSHINER_REVISION,
            "--",
            f"tasks/seeds/{TASK}",
        ):
            return self.archive
        raise AssertionError(f"unexpected Git command: {argv!r}")


def _tree_entries_from_archive(archive_bytes: bytes) -> bytes:
    prefix = f"tasks/seeds/{TASK}/"
    rows: list[tuple[str, str, str, str]] = []
    with tarfile.open(fileobj=io.BytesIO(archive_bytes), mode="r:") as archive:
        for member in archive:
            if not member.isreg():
                continue
            handle = archive.extractfile(member)
            assert handle is not None
            data = handle.read()
            oid = _git_blob_oid(data)
            mode = "100755" if member.mode & 0o111 else "100644"
            rows.append((mode, "blob", oid, member.name))
    rows = [row for row in rows if row[3].startswith(prefix)]
    return b"".join(
        f"{mode} {kind} {oid}\t{path}".encode() + b"\0"
        for mode, kind, oid, path in rows
    )


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


def _plan_for(contract, *calls: dict[str, object]) -> MutationPlan:
    return canonical_mutation_plan(
        _messages(*calls),
        frozenset(contract.protected_paths),
        contract.verify_cmd,
    )


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
    assert any(argv[4] == "cat-file" for argv, _env in fake.calls)
    assert any("archive" in argv[4:] for argv, _env in fake.calls)


def test_git_boundary_disables_replacements_lazy_fetch_and_inherited_config(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("GIT_CONFIG_COUNT", "1")
    monkeypatch.setenv("GIT_CONFIG_KEY_0", "url.https://attacker.invalid/.insteadOf")
    monkeypatch.setenv("GIT_CONFIG_VALUE_0", "file://")
    monkeypatch.setenv("GIT_OBJECT_DIRECTORY", str(tmp_path / "attacker-objects"))
    monkeypatch.setenv("GIT_ALTERNATE_OBJECT_DIRECTORIES", str(tmp_path / "alternates"))
    inherited_path = os.environ["PATH"]
    monkeypatch.setenv("PATH", f"{tmp_path / 'attacker-bin'}:{inherited_path}")
    fake = FakeGit(_archive())

    materialize_seed(
        GitSeedSource(
            Path("/immutable/moonshiner.git"), MOONSHINER_REVISION, fake
        ),
        TASK,
        tmp_path / "seed",
    )

    assert fake.calls
    for argv, env in fake.calls:
        assert argv[:4] == (
            "git",
            "--no-replace-objects",
            "-C",
            "/immutable/moonshiner.git",
        )
        assert env["GIT_NO_REPLACE_OBJECTS"] == "1"
        assert env["GIT_NO_LAZY_FETCH"] == "1"
        assert env["GIT_CONFIG_NOSYSTEM"] == "1"
        assert env["GIT_CONFIG_GLOBAL"] == os.devnull
        assert env["GIT_CONFIG_COUNT"] == "0"
        assert env["GIT_TERMINAL_PROMPT"] == "0"
        assert env["GIT_ALLOW_PROTOCOL"] == "file"
        assert env["PATH"] == os.defpath
        assert "GIT_CONFIG_KEY_0" not in env
        assert "GIT_CONFIG_VALUE_0" not in env
        assert "GIT_OBJECT_DIRECTORY" not in env
        assert "GIT_ALTERNATE_OBJECT_DIRECTORIES" not in env


def test_real_git_replace_cannot_change_pinned_commit_tree_or_archive(
    tmp_path: Path,
) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()

    def git(*args: str) -> bytes:
        return subprocess.run(
            ("git", "-C", str(repo), *args),
            check=True,
            capture_output=True,
        ).stdout

    git("init")
    git("config", "user.name", "Replay Test")
    git("config", "user.email", "replay@example.invalid")
    tracked = repo / "value.txt"
    tracked.write_text("A\n")
    git("add", "value.txt")
    git("commit", "-m", "A")
    commit_a = git("rev-parse", "HEAD").decode().strip()
    tree_a = git("rev-parse", f"{commit_a}^{{tree}}").decode().strip()

    tracked.write_text("B\n")
    git("commit", "-am", "B")
    commit_b = git("rev-parse", "HEAD").decode().strip()
    git("replace", commit_a, commit_b)

    assert git("show", f"{commit_a}:value.txt") == b"B\n"

    def pinned(*args: str) -> bytes:
        return replay._run_git(
            (
                "git",
                "--no-replace-objects",
                "-C",
                str(repo),
                *args,
            ),
            replay._sanitized_git_environment(),
        )

    assert pinned("rev-parse", f"{commit_a}^{{commit}}") == (
        commit_a + "\n"
    ).encode()
    assert pinned("rev-parse", f"{commit_a}^{{tree}}") == (tree_a + "\n").encode()
    assert pinned("show", f"{commit_a}:value.txt") == b"A\n"
    archive_bytes = pinned(
        "archive", "--format=tar", commit_a, "--", "value.txt"
    )
    with tarfile.open(fileobj=io.BytesIO(archive_bytes), mode="r:") as archive:
        extracted = archive.extractfile("value.txt")
        assert extracted is not None
        assert extracted.read() == b"A\n"


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
        _source(_archive(source=b"\xef\xbb\xbfcrlf\r\n", source_mode=0o755)),
        TASK,
        tmp_path / "seed",
    )
    assert stat.S_IMODE((first.files_root / "src.py").stat().st_mode) == 0o755
    assert (first.files_root / "src.py").read_bytes() == b"\xef\xbb\xbfcrlf\r\n"

    other_root = tmp_path / "other"
    second = materialize_seed(
        _source(_archive(source=b"\xef\xbb\xbfcrlf\n", source_mode=0o755)),
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


def test_archive_bytes_must_match_pinned_git_blob_ids(tmp_path: Path) -> None:
    archive = _archive()
    entries = _tree_entries_from_archive(archive)
    entries = entries.replace(_git_blob_oid(b"old\n").encode(), b"0" * 40, 1)
    with pytest.raises(ReplayContractError, match="blob object ID"):
        materialize_seed(
            _source(archive, tree_entries=entries), TASK, tmp_path / "seed"
        )


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
            _git_blob_oid(b"x"),
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
        _source(_archive(source=b"\xef\xbb\xbfold\r\nold", source_mode=0o755)),
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
    plan = _plan_for(contract, *calls)

    state = reconstruct_candidate(contract, plan, tmp_path / "candidate")

    assert isinstance(state, CandidateState)
    assert (state.root / "src.py").read_bytes() == "\ufeffnéw\r\nnéw".encode()
    assert (state.root / "new.py").read_bytes() == "尾\r\n".encode()
    assert stat.S_IMODE((state.root / "src.py").stat().st_mode) == 0o755
    assert state.changed_paths == ("new.py", "src.py")
    assert len(state.tree_sha256) == len(state.diff_sha256) == 64


def test_new_write_mode_is_deterministic_across_process_umask(tmp_path: Path) -> None:
    contract = _materialized(tmp_path)
    plan = _plan_for(
        contract, _call("Write", {"file_path": "new.py", "content": "x"})
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
    plan = _plan_for(
        contract,
        _call(
            "Edit",
            {
                "file_path": "src.py",
                "old_string": old,
                "new_string": "x",
                "replace_all": replace_all,
            },
        ),
    )
    destination = tmp_path / "candidate"
    with pytest.raises(ReplayContractError, match="match"):
        reconstruct_candidate(contract, plan, destination)
    assert (destination / "src.py").read_bytes() == (
        b"old old\n" if not replace_all else b"old\n"
    )


def test_reconstruction_requires_nonempty_source_diff(tmp_path: Path) -> None:
    contract = _materialized(tmp_path)
    no_change = _plan_for(
        contract, _call("Write", {"file_path": "src.py", "content": "old\n"})
    )
    with pytest.raises(ReplayContractError, match="empty candidate diff"):
        reconstruct_candidate(contract, no_change, tmp_path / "candidate")


def test_reconstruction_detects_seed_drift_before_copy(tmp_path: Path) -> None:
    contract = _materialized(tmp_path)
    (contract.files_root / "src.py").write_text("tampered")
    plan = _plan_for(
        contract, _call("Write", {"file_path": "src.py", "content": "new"})
    )
    with pytest.raises(ReplayContractError, match="immutable seed inventory changed"):
        reconstruct_candidate(contract, plan, tmp_path / "candidate")


def test_reconstruction_revalidates_symlinks_and_hardlinks_before_each_mutation(
    tmp_path: Path,
) -> None:
    contract = _materialized(tmp_path)
    plan = _plan_for(
        contract,
        _call("Write", {"file_path": "alias.py", "content": "x"}, "write"),
        _call(
            "Edit",
            {"file_path": "src.py", "old_string": "old", "new_string": "new"},
            "edit",
        ),
    )
    destination = tmp_path / "candidate"

    def introduce_alias(operation_index: int, root: Path) -> None:
        if operation_index == 1:
            (root / "alias.py").unlink()
            os.link(root / "src.py", root / "alias.py")

    with pytest.raises(ReplayContractError, match="hardlink"):
        reconstruct_candidate(contract, plan, destination, before_operation=introduce_alias)


def test_descriptor_boundary_blocks_ancestor_swap_before_final_open(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    prefix = f"tasks/seeds/{TASK}/files"
    archive = _archive(
        extra_members=((f"{prefix}/pkg/src.py", b"old\n", tarfile.REGTYPE, b""),)
    )
    contract = _materialized(tmp_path, archive)
    plan = _plan_for(
        contract, _call("Write", {"file_path": "pkg/src.py", "content": "PWN"})
    )
    destination = tmp_path / "candidate"
    outside = tmp_path / "outside"
    outside.mkdir()
    sentinel = outside / "src.py"
    sentinel.write_bytes(b"SENTINEL")
    original_check = replay._check_protected_aliases
    swapped = False

    def swap_after_alias_check(*args, **kwargs):
        nonlocal swapped
        result = original_check(*args, **kwargs)
        if not swapped:
            swapped = True
            (destination / "pkg").rename(destination / "pkg-original")
            (destination / "pkg").symlink_to(outside, target_is_directory=True)
        return result

    monkeypatch.setattr(replay, "_check_protected_aliases", swap_after_alias_check)
    with pytest.raises(ReplayContractError):
        reconstruct_candidate(contract, plan, destination)
    assert swapped is True
    assert sentinel.read_bytes() == b"SENTINEL"


def test_reconstruction_rejects_operation_protected_set_mismatch_before_copy(
    tmp_path: Path,
) -> None:
    contract = _materialized(tmp_path)
    plan = canonical_mutation_plan(
        _messages(_call("Write", {"file_path": "src.py", "content": "bad"})),
        frozenset(),
        contract.verify_cmd,
    )
    destination = tmp_path / "candidate"
    with pytest.raises(ReplayContractError, match="protected_paths binding"):
        reconstruct_candidate(contract, plan, destination)
    assert not destination.exists()


def test_reconstruction_rejects_stale_operation_hash_before_copy(tmp_path: Path) -> None:
    contract = _materialized(tmp_path)
    honest = _plan_for(
        contract, _call("Write", {"file_path": "src.py", "content": "honest"})
    )
    altered_operation = dataclasses.replace(honest.operations[0], content="ALTERED")
    altered = dataclasses.replace(honest, operations=(altered_operation,))
    destination = tmp_path / "candidate"
    with pytest.raises(ReplayContractError, match="operation hash mismatch"):
        reconstruct_candidate(contract, altered, destination)
    assert not destination.exists()


def test_reconstruction_rejects_noncanonical_operation_type_before_copy(
    tmp_path: Path,
) -> None:
    contract = _materialized(tmp_path)
    honest = _plan_for(
        contract, _call("Write", {"file_path": "src.py", "content": "honest"})
    )
    altered = dataclasses.replace(honest, operations=(object(),))
    destination = tmp_path / "candidate"
    with pytest.raises(ReplayContractError, match="exact Write/Edit operation type"):
        reconstruct_candidate(contract, altered, destination)
    assert not destination.exists()


@pytest.mark.parametrize(
    "path",
    [
        "tests/helper.py",
        "fixtures/input.py",
        "benchmarks/speed.py",
        "generated/client.py",
        "vendor/library.py",
        "conftest.py",
    ],
)
def test_reconstruction_rejects_importer_mutation_path_families_before_copy(
    tmp_path: Path, path: str
) -> None:
    contract = _materialized(tmp_path)
    plan = _plan_for(
        contract, _call("Write", {"file_path": path, "content": "bad"})
    )
    destination = tmp_path / "candidate"
    with pytest.raises(ReplayContractError, match="rejected mutation path family"):
        reconstruct_candidate(contract, plan, destination)
    assert not destination.exists()


def test_reconstruction_hashes_are_deterministic_across_fresh_destinations(
    tmp_path: Path,
) -> None:
    contract = _materialized(tmp_path)
    plan = _plan_for(
        contract,
        _call("Edit", {"file_path": "src.py", "old_string": "old", "new_string": "new"}),
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


@pytest.mark.parametrize("repo_path", ["a/foo.py", "b/foo.py"])
def test_reference_patch_strips_transport_prefix_exactly_once(
    tmp_path: Path, repo_path: str
) -> None:
    contract = _materialized(tmp_path, _archive(patch=_patch(repo_path)))
    executor = PatchExecutor()
    result = preflight_reference_patch(contract, executor=executor)
    assert result.eligible is True
    assert result.targets == (repo_path,)
    assert len(executor.calls) == 1


IMAGE_DIGEST = "example.invalid/fable-python@sha256:" + "1" * 64
CID = "c" * 64
CAPTURED_MOBY_MASKED_PATHS = [
    "/proc/acpi",
    "/proc/asound",
    "/proc/kcore",
    "/proc/keys",
    "/proc/latency_stats",
    "/proc/timer_list",
    "/proc/timer_stats",
    "/proc/scsi",
    "/sys/firmware",
]
CAPTURED_MOBY_READONLY_PATHS = [
    "/proc/bus",
    "/proc/fs",
    "/proc/irq",
    "/proc/sys",
    "/proc/sysrq-trigger",
]


def _captured_moby_tmpfs_mounts(tmpfs: dict[str, str]) -> list[dict[str, object]]:
    """Top-level MountPoint shape emitted by Moby for running tmpfs mounts."""

    return [
        {
            "Type": "tmpfs",
            "Source": "",
            "Destination": destination,
            "Mode": "",
            "RW": True,
            "Propagation": "",
        }
        for destination in sorted(tmpfs)
    ]


def test_task2_interfaces_are_explicit_public_exports() -> None:
    assert {
        "AdmissionEvidence",
        "ControlSetEvidence",
        "DockerExecutor",
        "DockerPolicy",
        "ReplayEvidence",
        "RestrictedExecutor",
        "RunEvidence",
        "functional_admission_probe",
        "run_control_set",
        "verify_candidate",
    } <= set(replay.__all__)


def test_default_docker_client_environment_cannot_redirect_to_network_or_context(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("DOCKER_HOST", "tcp://attacker.invalid:2375")
    monkeypatch.setenv("DOCKER_CONTEXT", "remote")
    monkeypatch.setenv("DOCKER_CONFIG", "/attacker/config")
    monkeypatch.setenv("HTTP_PROXY", "http://attacker.invalid")
    monkeypatch.setenv("PATH", "/attacker/bin:" + os.environ["PATH"])

    env = replay._sanitized_docker_environment()

    assert env == {
        "PATH": os.defpath,
        "HOME": os.devnull,
        "XDG_CONFIG_HOME": os.devnull,
        "LANG": "C",
        "LC_ALL": "C",
    }


class FakeDockerRuntime:
    def __init__(
        self,
        *,
        verifier_rc: int = 0,
        fail_verb: str | None = None,
        result_updates: dict[str, object] | None = None,
        free_bytes: tuple[int, ...] = (80 << 30, 79 << 30),
        timed_out: bool = False,
        verifier_output: bytes = b"verifier output\n",
        output_truncated: bool = False,
        result_mode: int = 0o600,
        inspect_mutator: object | None = None,
        create_output: bytes | None = None,
    ) -> None:
        self.verifier_rc = verifier_rc
        self.fail_verb = fail_verb
        self.result_updates = result_updates or {}
        self.timed_out = timed_out
        self.verifier_output = verifier_output
        self.output_truncated = output_truncated
        self.result_mode = result_mode
        self.inspect_mutator = inspect_mutator
        self.create_output = create_output or (CID + "\n").encode()
        self.free_values = iter(free_bytes)
        self.calls: list[tuple[tuple[str, ...], int, int, Path | None]] = []
        self._failed = False
        self.protected_result: list[str] = []
        self.verifier_script = ""
        self.input_modes: dict[str, int] = {}
        self.stage_mode = 0
        self.container_cmd = ["/seed/wrapper.sh", "test_src.py"]
        self.started = False

    def disk_free_bytes(self) -> int:
        return next(self.free_values)

    def run(
        self,
        argv: tuple[str, ...],
        *,
        timeout_seconds: int,
        output_limit_bytes: int,
        output_path: Path | None = None,
    ):
        self.calls.append((argv, timeout_seconds, output_limit_bytes, output_path))
        verb = argv[1] if len(argv) > 1 else ""
        if self.fail_verb == verb and not self._failed:
            self._failed = True
            return replay.RuntimeCommandResult(1, b"forced failure", 0.01)
        if verb == "version":
            return replay.RuntimeCommandResult(0, b"27.5.1\n", 0.01)
        if verb == "image":
            return replay.RuntimeCommandResult(
                0, ("sha256:" + "1" * 64 + "\n").encode(), 0.01
            )
        if verb == "create":
            image_index = next(
                index for index, value in enumerate(argv) if "@sha256:" in value
            )
            self.container_cmd = list(argv[image_index + 1 :])
            return replay.RuntimeCommandResult(0, self.create_output, 0.01)
        if verb == "inspect":
            tmpfs = {
                "/work": "rw,nosuid,nodev,size=4294967296,uid=65532,gid=65532,mode=0700",
                "/scratch": "rw,nosuid,nodev,size=4294967296,uid=65532,gid=65532,mode=0700",
                "/control": "rw,nosuid,nodev,noexec,size=1048576,uid=65533,gid=65533,mode=0700",
                "/result": "rw,nosuid,nodev,noexec,size=1048576,uid=0,gid=0,mode=0700",
                "/status": "rw,nosuid,nodev,noexec,size=1048576,uid=0,gid=0,mode=0755",
            }
            payload = {
                "Id": CID,
                "Name": "/fable-replay-" + "a" * 32,
                "Image": "sha256:" + "1" * 64,
                "State": {
                    "Status": "running" if self.started else "created",
                    "Running": self.started,
                    "Pid": 1234 if self.started else 0,
                },
                "Config": {
                    "Labels": {"fable.replay.owner": "a" * 32},
                    "User": "0:0",
                    "Entrypoint": ["/bin/sh"],
                    "Cmd": self.container_cmd,
                    "Volumes": None,
                    "ExposedPorts": None,
                },
                "HostConfig": {
                    "NetworkMode": "none",
                    "ReadonlyRootfs": True,
                    "CapDrop": ["ALL"],
                    "SecurityOpt": ["no-new-privileges:true"],
                    "NanoCpus": 2_000_000_000,
                    "Memory": 4 * 1024**3,
                    "MemorySwap": 4 * 1024**3,
                    "PidsLimit": 256,
                    "Tmpfs": tmpfs,
                    "Ulimits": [
                        {"Name": "fsize", "Soft": 1_048_576, "Hard": 1_048_576}
                    ],
                    "Binds": None,
                    "Mounts": [],
                    "Privileged": False,
                    "CapAdd": None,
                    "Devices": None,
                    "DeviceRequests": None,
                    "DeviceCgroupRules": None,
                    "PidMode": "",
                    "IpcMode": "private",
                    "UTSMode": "",
                    "UsernsMode": "",
                    "PortBindings": {},
                    "PublishAllPorts": False,
                    "RestartPolicy": {"Name": "no", "MaximumRetryCount": 0},
                    "Runtime": "runc",
                    "CgroupnsMode": "private",
                    "Isolation": "",
                    "VolumesFrom": None,
                    "Links": None,
                    "GroupAdd": None,
                    "Sysctls": None,
                    "ExtraHosts": None,
                    "MaskedPaths": list(CAPTURED_MOBY_MASKED_PATHS),
                    "ReadonlyPaths": list(CAPTURED_MOBY_READONLY_PATHS),
                },
                "Mounts": _captured_moby_tmpfs_mounts(tmpfs) if self.started else [],
                "NetworkSettings": {"Ports": {}, "Networks": {"none": {}}},
            }
            if self.inspect_mutator is not None:
                self.inspect_mutator(payload)
            return replay.RuntimeCommandResult(
                0, (json.dumps(payload) + "\n").encode(), 0.01
            )
        if verb == "exec":
            if any(value.endswith("/verifier.sh") for value in argv):
                return replay.RuntimeCommandResult(
                    self.verifier_rc,
                    self.verifier_output,
                    0.25,
                    timed_out=self.timed_out,
                    truncated=self.output_truncated,
                )
            return replay.RuntimeCommandResult(0, b"", 0.01)
        if verb == "start":
            self.started = True
            return replay.RuntimeCommandResult(0, b"", 0.01)
        if verb == "stop":
            self.started = False
            return replay.RuntimeCommandResult(0, b"", 0.01)
        if verb == "cp" and argv[3] == f"{CID}:/seed":
            candidate = Path(argv[2]) / "candidate"
            self.stage_mode = stat.S_IMODE(Path(argv[2]).stat().st_mode)
            self.verifier_script = (Path(argv[2]) / "verifier.sh").read_text()
            self.input_modes = {
                path.relative_to(Path(argv[2])).as_posix(): stat.S_IMODE(path.stat().st_mode)
                for path in Path(argv[2]).rglob("*")
            }
            self.protected_result = [
                hashlib.sha256((candidate / path).read_bytes()).hexdigest()
                for path in self.container_cmd[1:]
            ]
            return replay.RuntimeCommandResult(0, b"", 0.01)
        if verb == "cp" and argv[2] == f"{CID}:/result/result.json":
            destination = Path(argv[3])
            result = {
                "schema_version": 1,
                "run_nonce": "a" * 32,
                "wrapper_rc": 0,
                "post_tree_sha256": "d" * 64,
                "protected_sha256": self.protected_result,
                "resource_peaks": {
                    "memory_bytes": 1024,
                    "pids": 3,
                    "cpu_usec": 4000,
                },
            }
            result.update(self.result_updates)
            destination.write_text(json.dumps(result), encoding="utf-8")
            os.chmod(destination, self.result_mode)
            return replay.RuntimeCommandResult(0, b"", 0.01)
        if verb == "logs":
            return replay.RuntimeCommandResult(0, b"wrapper log\n", 0.01)
        if verb == "ps":
            return replay.RuntimeCommandResult(0, b"", 0.01)
        return replay.RuntimeCommandResult(0, b"", 0.01)


def _docker_policy() -> object:
    return replay.DockerPolicy(images=(("python", IMAGE_DIGEST),))


def _docker_execute(
    runtime: FakeDockerRuntime,
    tmp_path: Path,
    *,
    effective_timeout: int = 300,
    clock: object | None = None,
    protected_path: str = "test_src.py",
):
    candidate = tmp_path / "candidate"
    candidate.mkdir()
    (candidate / "src.py").write_text("new\n")
    (candidate / protected_path).write_text("assert True\n")
    tree = replay._inventory_sha(replay._inventory_from_root(candidate))
    executor = replay.DockerExecutor(
        _docker_policy(),
        runner=runtime,
        disk_free_bytes=runtime.disk_free_bytes,
        token_factory=lambda: "a" * 32,
        **({"clock": clock} if clock is not None else {}),
    )
    return executor.execute(
        candidate_root=candidate,
        language="python",
        verifier_text="python3 -m pytest -q",
        effective_timeout=effective_timeout,
        control_identity="candidate",
        pre_candidate_tree_sha256=tree,
        pre_candidate_diff_sha256="e" * 64,
        protected_before=((protected_path, hashlib.sha256(b"assert True\n").hexdigest()),),
    )


@pytest.mark.parametrize(
    "image",
    [
        "example.invalid/fable-python:latest",
        "example.invalid/fable-python@sha256:short",
        "example.invalid/fable-python@sha256:" + "G" * 64,
    ],
)
def test_docker_policy_requires_an_exact_immutable_digest(image: str) -> None:
    with pytest.raises(ValueError, match="digest"):
        replay.DockerPolicy(images=(("python", image),))


def test_docker_policy_requires_deeply_immutable_image_mapping() -> None:
    with pytest.raises(ValueError, match="immutable"):
        replay.DockerPolicy(images=[["python", IMAGE_DIGEST]])

    policy = _docker_policy()
    with pytest.raises(dataclasses.FrozenInstanceError):
        policy.images = (("python", IMAGE_DIGEST),)


def test_run_contract_policy_payload_binds_docker_command_timeout() -> None:
    policy = _docker_policy()

    first = replay._policy_payload(policy, IMAGE_DIGEST)
    second = replay._policy_payload(
        dataclasses.replace(policy, command_timeout_seconds=17), IMAGE_DIGEST
    )

    assert first["command_timeout_seconds"] == 30
    assert second["command_timeout_seconds"] == 17
    assert replay._sha256(replay._canonical_json(first)) != replay._sha256(
        replay._canonical_json(second)
    )


@pytest.mark.parametrize(
    "updates",
    [
        {"cpus": "0"},
        {"cpus": "nan"},
        {"verifier_uid": 0},
        {"signal_uid": 65_532},
        {"observer_gid": 65_533},
        {"disk_floor_bytes": True},
    ],
)
def test_docker_policy_rejects_identity_or_resource_separation_overrides(
    updates: dict[str, object],
) -> None:
    with pytest.raises(ValueError):
        dataclasses.replace(_docker_policy(), **updates)


def test_docker_executor_uses_the_complete_restricted_state_machine(tmp_path: Path) -> None:
    runtime = FakeDockerRuntime()
    evidence = _docker_execute(runtime, tmp_path)
    argvs = [call[0] for call in runtime.calls]
    create = next(argv for argv in argvs if argv[1] == "create")

    assert "--pull=never" in create
    assert "--network=none" in create
    assert "--read-only" in create
    assert "--cap-drop=ALL" in create
    assert "--security-opt=no-new-privileges:true" in create
    assert "--ipc=private" in create
    assert "--cgroupns=private" in create
    assert "--runtime=runc" in create
    assert "--restart=no" in create
    assert "--entrypoint=/bin/sh" in create
    assert "--pids-limit=256" in create
    assert any(value.startswith("--memory=") for value in create)
    assert any(value.startswith("--memory-swap=") for value in create)
    assert any(value.startswith("--cpus=") for value in create)
    assert any(value.startswith("--ulimit=fsize=") for value in create)
    assert sum(value == "--tmpfs" for value in create) == 5
    assert not any(value in {"-v", "--volume", "--mount"} for value in create)
    assert not any("docker.sock" in value for value in create)
    assert IMAGE_DIGEST in create
    image_index = create.index(IMAGE_DIGEST)
    assert create[image_index + 1 :] == ("/seed/wrapper.sh", "test_src.py")

    verbs = [argv[1] for argv in argvs]
    assert verbs == [
        "version",
        "image",
        "create",
        "inspect",
        "cp",
        "start",
        "inspect",
        "exec",
        "exec",
        "exec",
        "exec",
        "inspect",
        "cp",
        "inspect",
        "inspect",
        "logs",
        "stop",
        "inspect",
        "rm",
        "ps",
        "ps",
    ]
    assert all("--no-trunc" in argv for argv in argvs if argv[1] == "ps")
    verifier = next(
        argv for argv in argvs if any(value.endswith("/verifier.sh") for value in argv)
    )
    signal = next(argv for argv in argvs if any("/control/go" in value for value in argv))
    poll = next(argv for argv in argvs if any("/status/done" in value for value in argv))
    assert ("--user", "65532:65532") == verifier[2:4]
    assert ("--user", "65533:65533") == signal[2:4]
    assert ("--user", "65534:65534") == poll[2:4]
    assert evidence.schema_version == 1
    assert evidence.resolved is True
    assert evidence.trainable is True
    assert evidence.cleanup_state == "verified_removed"
    assert evidence.protected_before == evidence.protected_after
    assert runtime.input_modes["candidate"] == 0o555
    assert runtime.input_modes["candidate/src.py"] == 0o444
    assert runtime.input_modes["candidate/test_src.py"] == 0o444
    assert runtime.input_modes["verifier.sh"] == 0o555
    assert runtime.input_modes["wrapper.sh"] == 0o500
    assert runtime.stage_mode == 0o555


def test_running_inspect_accepts_captured_real_moby_tmpfs_mount_shape(
    tmp_path: Path,
) -> None:
    captured = _captured_moby_tmpfs_mounts(
        replay._tmpfs_policy(_docker_policy())
    )
    assert all(
        set(mount)
        == {"Type", "Source", "Destination", "Mode", "RW", "Propagation"}
        and mount["Mode"] == ""
        for mount in captured
    )

    evidence = _docker_execute(FakeDockerRuntime(), tmp_path)

    assert evidence.resolved is True


def test_trusted_wrapper_quotes_hashes_as_json_and_never_embeds_verifier() -> None:
    script = replay._wrapper_script("a" * 32)

    assert '"protected_sha256":[' in script
    assert 'for path do' in script
    assert 'sha256sum -- "/work/$path"' in script
    assert "xargs -0 -r sha256sum --" in script
    assert "python3 -m pytest" not in script
    assert "/result/result.json" in script


def test_hostile_protected_path_is_argv_data_never_shell_source(tmp_path: Path) -> None:
    hostile = "odd' ; touch status-pwn ; echo '.py"
    runtime = FakeDockerRuntime()

    _docker_execute(runtime, tmp_path, protected_path=hostile)

    create = next(call[0] for call in runtime.calls if call[0][1] == "create")
    assert create[-2:] == ("/seed/wrapper.sh", hostile)
    assert hostile not in replay._wrapper_script("a" * 32)


def test_nonroot_verifier_copies_readonly_seed_to_private_writable_tmpfs() -> None:
    script = replay._verifier_script("python3 -m pytest -q", 1024)

    assert "cp -R /seed/candidate/. /work/" in script
    assert "chmod -R u+rwX /work" in script
    assert "cd /work" in script


@pytest.mark.parametrize("fail_verb", ["create", "inspect", "cp", "start"])
def test_docker_executor_cleans_only_its_exact_cid_on_every_failure(
    tmp_path: Path, fail_verb: str
) -> None:
    runtime = FakeDockerRuntime(fail_verb=fail_verb)
    with pytest.raises(ReplayContractError):
        _docker_execute(runtime, tmp_path)
    argvs = [call[0] for call in runtime.calls]
    if fail_verb == "create":
        assert all(CID not in argv for argv in argvs)
    elif fail_verb == "inspect":
        assert ("docker", "stop", "--time=2", CID) not in argvs
        assert ("docker", "rm", "--force", "--volumes", CID) not in argvs
    else:
        assert ("docker", "stop", "--time=2", CID) in argvs
        assert ("docker", "rm", "--force", "--volumes", CID) in argvs
    assert not any(argv[1] in {"pull", "prune"} for argv in argvs)


def test_invalid_create_stdout_never_becomes_a_cleanup_mutation_target(
    tmp_path: Path,
) -> None:
    victim = "victim-production"
    runtime = FakeDockerRuntime(create_output=(victim + "\n").encode())

    with pytest.raises(ReplayContractError, match="invalid exact CID"):
        _docker_execute(runtime, tmp_path)

    mutations = [
        argv
        for argv, *_rest in runtime.calls
        if argv[1] in {"stop", "rm"}
    ]
    assert all(victim not in argv for argv in mutations)


def test_invalid_create_stdout_recovers_only_owner_label_proven_exact_cid(
    tmp_path: Path,
) -> None:
    victim = "victim-production"

    class DiscoverRuntime(FakeDockerRuntime):
        def __init__(self) -> None:
            super().__init__(create_output=(victim + "\n").encode())
            self.ps_calls = 0

        def run(self, argv, **kwargs):
            if argv[1] == "ps":
                self.calls.append(
                    (
                        argv,
                        kwargs["timeout_seconds"],
                        kwargs["output_limit_bytes"],
                        kwargs.get("output_path"),
                    )
                )
                self.ps_calls += 1
                output = (CID + "\n").encode() if self.ps_calls == 1 else b""
                return replay.RuntimeCommandResult(0, output, 0.01)
            return super().run(argv, **kwargs)

    runtime = DiscoverRuntime()
    with pytest.raises(ReplayContractError, match="invalid exact CID"):
        _docker_execute(runtime, tmp_path)
    argvs = [call[0] for call in runtime.calls]

    assert ("docker", "stop", "--time=2", CID) in argvs
    assert ("docker", "rm", "--force", "--volumes", CID) in argvs
    assert all(victim not in argv for argv in argvs if argv[1] in {"stop", "rm"})


@pytest.mark.parametrize("drift", ["wrong_owner", "wrong_id"])
def test_failed_initial_ownership_proof_never_mutates_create_stdout(
    tmp_path: Path, drift: str
) -> None:
    def mutate(payload: dict[str, object]) -> None:
        if drift == "wrong_owner":
            payload["Config"]["Labels"]["fable.replay.owner"] = "b" * 32
        else:
            payload["Id"] = "d" * 64

    runtime = FakeDockerRuntime(inspect_mutator=mutate)
    with pytest.raises(ReplayContractError, match="ownership/policy"):
        _docker_execute(runtime, tmp_path)
    argvs = [call[0] for call in runtime.calls]

    assert ("docker", "stop", "--time=2", CID) not in argvs
    assert ("docker", "rm", "--force", "--volumes", CID) not in argvs


def test_malformed_initial_ownership_inspect_never_mutates_create_stdout(
    tmp_path: Path,
) -> None:
    class MalformedInspectRuntime(FakeDockerRuntime):
        def run(self, argv, **kwargs):
            if argv[1] == "inspect" and argv[2] == CID:
                self.calls.append(
                    (
                        argv,
                        kwargs["timeout_seconds"],
                        kwargs["output_limit_bytes"],
                        kwargs.get("output_path"),
                    )
                )
                return replay.RuntimeCommandResult(0, b"not-json", 0.01)
            return super().run(argv, **kwargs)

    runtime = MalformedInspectRuntime()
    with pytest.raises(ReplayContractError, match="schema"):
        _docker_execute(runtime, tmp_path)
    argvs = [call[0] for call in runtime.calls]

    assert ("docker", "stop", "--time=2", CID) not in argvs
    assert ("docker", "rm", "--force", "--volumes", CID) not in argvs


@pytest.mark.parametrize("drift", ["owner", "id"])
def test_cleanup_reproves_exact_identity_immediately_before_stop(
    tmp_path: Path, drift: str
) -> None:
    inspections = 0

    def mutate(payload: dict[str, object]) -> None:
        nonlocal inspections
        inspections += 1
        if inspections >= 5:
            if drift == "owner":
                payload["Config"]["Labels"]["fable.replay.owner"] = "b" * 32
            else:
                payload["Id"] = "d" * 64

    runtime = FakeDockerRuntime(inspect_mutator=mutate)
    with pytest.raises(ReplayContractError, match="cleanup"):
        _docker_execute(runtime, tmp_path)
    argvs = [call[0] for call in runtime.calls]

    assert ("docker", "stop", "--time=2", CID) not in argvs
    assert ("docker", "rm", "--force", "--volumes", CID) not in argvs


def test_cleanup_malformed_identity_proof_never_reaches_stop(tmp_path: Path) -> None:
    class MalformedCleanupInspectRuntime(FakeDockerRuntime):
        def __init__(self) -> None:
            super().__init__()
            self.inspect_count = 0

        def run(self, argv, **kwargs):
            if argv[1] == "inspect" and argv[2] == CID:
                self.inspect_count += 1
                if self.inspect_count == 5:
                    self.calls.append(
                        (
                            argv,
                            kwargs["timeout_seconds"],
                            kwargs["output_limit_bytes"],
                            kwargs.get("output_path"),
                        )
                    )
                    return replay.RuntimeCommandResult(0, b"not-json", 0.01)
            return super().run(argv, **kwargs)

    runtime = MalformedCleanupInspectRuntime()
    with pytest.raises(ReplayContractError, match="cleanup"):
        _docker_execute(runtime, tmp_path)
    argvs = [call[0] for call in runtime.calls]

    assert ("docker", "stop", "--time=2", CID) not in argvs
    assert ("docker", "rm", "--force", "--volumes", CID) not in argvs


def test_cleanup_reproves_owner_again_immediately_before_remove(tmp_path: Path) -> None:
    inspections = 0

    def mutate(payload: dict[str, object]) -> None:
        nonlocal inspections
        inspections += 1
        if inspections >= 6:
            payload["Config"]["Labels"]["fable.replay.owner"] = "b" * 32

    runtime = FakeDockerRuntime(inspect_mutator=mutate)
    with pytest.raises(ReplayContractError, match="cleanup"):
        _docker_execute(runtime, tmp_path)
    argvs = [call[0] for call in runtime.calls]

    assert ("docker", "stop", "--time=2", CID) in argvs
    assert ("docker", "rm", "--force", "--volumes", CID) not in argvs


def test_docker_executor_never_claims_cleanup_when_final_query_fails(
    tmp_path: Path,
) -> None:
    runtime = FakeDockerRuntime(fail_verb="ps")
    with pytest.raises(ReplayContractError, match="cleanup"):
        _docker_execute(runtime, tmp_path)


@pytest.mark.parametrize("verb", ["stop", "rm"])
def test_docker_executor_rejects_nonzero_exact_cid_cleanup_result(
    tmp_path: Path, verb: str
) -> None:
    runtime = FakeDockerRuntime(fail_verb=verb)
    with pytest.raises(ReplayContractError, match="cleanup"):
        _docker_execute(runtime, tmp_path)


def test_cleanup_failure_is_never_hidden_by_an_earlier_execution_error(
    tmp_path: Path,
) -> None:
    runtime = FakeDockerRuntime(
        fail_verb="stop", result_updates={"extra": "malformed"}
    )

    with pytest.raises(ReplayContractError) as raised:
        _docker_execute(runtime, tmp_path)

    message = str(raised.value)
    assert "cleanup failed" in message
    assert "preceding execution error" in message
    assert "wrapper result" in message


def test_docker_executor_inspects_and_removes_owned_descendants_by_exact_cid(
    tmp_path: Path,
) -> None:
    descendant = "d" * 64

    class DescendantRuntime(FakeDockerRuntime):
        def __init__(self) -> None:
            super().__init__()
            self.ps_calls = 0

        def run(self, argv, **kwargs):
            if argv[1] == "ps":
                self.calls.append(
                    (
                        argv,
                        kwargs["timeout_seconds"],
                        kwargs["output_limit_bytes"],
                        kwargs.get("output_path"),
                    )
                )
                self.ps_calls += 1
                output = (descendant + "\n").encode() if self.ps_calls == 1 else b""
                return replay.RuntimeCommandResult(0, output, 0.01)
            if argv[1] == "inspect" and argv[2] == descendant:
                self.calls.append(
                    (
                        argv,
                        kwargs["timeout_seconds"],
                        kwargs["output_limit_bytes"],
                        kwargs.get("output_path"),
                    )
                )
                return replay.RuntimeCommandResult(
                    0,
                    (json.dumps({"Id": descendant, "Config": {"Labels": {"fable.replay.owner": "a" * 32}}}) + "\n").encode(),
                    0.01,
                )
            return super().run(argv, **kwargs)

    runtime = DescendantRuntime()
    evidence = _docker_execute(runtime, tmp_path)
    argvs = [call[0] for call in runtime.calls]

    assert evidence.cleanup_state == "verified_removed"
    assert ("docker", "inspect", descendant) in argvs
    assert ("docker", "stop", "--time=2", descendant) in argvs
    assert ("docker", "rm", "--force", "--volumes", descendant) in argvs


def test_docker_executor_never_mutates_descendant_without_exact_owner_proof(
    tmp_path: Path,
) -> None:
    descendant = "d" * 64

    class WrongOwnerRuntime(FakeDockerRuntime):
        def run(self, argv, **kwargs):
            if argv[1] == "ps":
                self.calls.append(
                    (
                        argv,
                        kwargs["timeout_seconds"],
                        kwargs["output_limit_bytes"],
                        kwargs.get("output_path"),
                    )
                )
                return replay.RuntimeCommandResult(0, (descendant + "\n").encode(), 0.01)
            if argv[1] == "inspect" and argv[2] == descendant:
                self.calls.append(
                    (
                        argv,
                        kwargs["timeout_seconds"],
                        kwargs["output_limit_bytes"],
                        kwargs.get("output_path"),
                    )
                )
                payload = {
                    "Id": descendant,
                    "Config": {"Labels": {"fable.replay.owner": "b" * 32}},
                }
                return replay.RuntimeCommandResult(
                    0, (json.dumps(payload) + "\n").encode(), 0.01
                )
            return super().run(argv, **kwargs)

    runtime = WrongOwnerRuntime()
    with pytest.raises(ReplayContractError, match="cleanup"):
        _docker_execute(runtime, tmp_path)
    argvs = [call[0] for call in runtime.calls]
    assert ("docker", "stop", "--time=2", descendant) not in argvs
    assert ("docker", "rm", "--force", "--volumes", descendant) not in argvs


def test_docker_executor_records_verifier_failure_and_still_cleans(tmp_path: Path) -> None:
    runtime = FakeDockerRuntime(verifier_rc=1)
    evidence = _docker_execute(runtime, tmp_path)

    assert evidence.resolved is False
    assert evidence.failure_class == "verifier_failed"
    assert evidence.cleanup_state == "verified_removed"
    argvs = [call[0] for call in runtime.calls]
    assert ("docker", "stop", "--time=2", CID) in argvs
    assert ("docker", "rm", "--force", "--volumes", CID) in argvs


@pytest.mark.parametrize(
    "updates",
    [
        {"run_nonce": "b" * 32},
        {"schema_version": 2},
        {"protected_sha256": []},
        {"post_tree_sha256": "short"},
        {"extra": "forged"},
        {
            "resource_peaks": {
                "memory_bytes": 0,
                "pids": 0,
                "cpu_usec": 0,
            }
        },
    ],
)
def test_docker_executor_rejects_forged_or_malformed_wrapper_results(
    tmp_path: Path, updates: dict[str, object]
) -> None:
    runtime = FakeDockerRuntime(result_updates=updates)
    with pytest.raises(ReplayContractError, match="result"):
        _docker_execute(runtime, tmp_path)
    argvs = [call[0] for call in runtime.calls]
    assert ("docker", "stop", "--time=2", CID) in argvs
    assert ("docker", "rm", "--force", "--volumes", CID) in argvs


def test_docker_executor_requires_mode_0600_wrapper_result(tmp_path: Path) -> None:
    runtime = FakeDockerRuntime(result_mode=0o644)
    with pytest.raises(ReplayContractError, match="result"):
        _docker_execute(runtime, tmp_path)


def test_docker_executor_rejects_effective_container_policy_drift(tmp_path: Path) -> None:
    def mutate(payload: dict[str, object]) -> None:
        payload["Config"]["User"] = "65532:65532"

    runtime = FakeDockerRuntime(inspect_mutator=mutate)
    with pytest.raises(ReplayContractError, match="policy"):
        _docker_execute(runtime, tmp_path)

    argvs = [call[0] for call in runtime.calls]
    assert not any(argv[1] == "start" for argv in argvs)


@pytest.mark.parametrize(
    ("scope", "field", "unsafe"),
    [
        ("HostConfig", "Privileged", True),
        ("HostConfig", "CapAdd", ["SYS_ADMIN"]),
        ("HostConfig", "CapDrop", ["ALL", "NET_RAW"]),
        (
            "HostConfig",
            "SecurityOpt",
            ["no-new-privileges:true", "seccomp=unconfined"],
        ),
        (
            "HostConfig",
            "Mounts",
            [{"Type": "bind", "Source": "/", "Target": "/host"}],
        ),
        ("HostConfig", "Binds", ["/:/host:rw"]),
        (
            "HostConfig",
            "Devices",
            [{"PathOnHost": "/dev/sda", "PathInContainer": "/dev/sda"}],
        ),
        (
            "HostConfig",
            "DeviceRequests",
            [{"Driver": "nvidia", "Count": -1, "Capabilities": [["gpu"]]}],
        ),
        ("HostConfig", "DeviceCgroupRules", ["a *:* rwm"]),
        ("HostConfig", "PidMode", "host"),
        ("HostConfig", "IpcMode", "host"),
        ("HostConfig", "UTSMode", "host"),
        ("HostConfig", "UsernsMode", "host"),
        ("HostConfig", "PortBindings", {"8000/tcp": [{"HostPort": "8000"}]}),
        ("HostConfig", "PublishAllPorts", True),
        (
            "HostConfig",
            "RestartPolicy",
            {"Name": "always", "MaximumRetryCount": 0},
        ),
        ("HostConfig", "Runtime", "nvidia"),
        ("HostConfig", "CgroupnsMode", "host"),
        ("HostConfig", "Isolation", "hyperv"),
        ("HostConfig", "VolumesFrom", ["victim:rw"]),
        ("HostConfig", "Links", ["/victim:/target"]),
        ("HostConfig", "GroupAdd", ["0"]),
        ("HostConfig", "Sysctls", {"kernel.core_pattern": "/host/pwn"}),
        ("HostConfig", "ExtraHosts", ["metadata:169.254.169.254"]),
        ("HostConfig", "MaskedPaths", []),
        ("HostConfig", "ReadonlyPaths", []),
        (
            "HostConfig",
            "Ulimits",
            [
                {"Name": "fsize", "Soft": 1_048_576, "Hard": 1_048_576},
                {"Name": "nofile", "Soft": 1_024, "Hard": 1_024},
            ],
        ),
        ("HostConfig", "NetworkMode", "bridge"),
        ("Config", "ExposedPorts", {"8000/tcp": {}}),
        (
            "Mounts",
            "replace",
            [
                {
                    "Type": "bind",
                    "Source": "/",
                    "Destination": "/host",
                    "Mode": "rw",
                    "RW": True,
                    "Propagation": "rprivate",
                }
            ],
        ),
        ("NetworkSettings", "Ports", {"8000/tcp": [{"HostPort": "8000"}]}),
        ("NetworkSettings", "Networks", {"bridge": {}}),
    ],
    ids=lambda value: str(value)[:40],
)
def test_docker_executor_rejects_security_policy_drift_before_start(
    tmp_path: Path, scope: str, field: str, unsafe: object
) -> None:
    def mutate(payload: dict[str, object]) -> None:
        if scope == "Mounts":
            payload["Mounts"] = unsafe
        else:
            payload[scope][field] = unsafe

    runtime = FakeDockerRuntime(inspect_mutator=mutate)
    with pytest.raises(ReplayContractError, match="policy|mount|namespace"):
        _docker_execute(runtime, tmp_path)
    argvs = [call[0] for call in runtime.calls]

    assert not any(argv[1] == "start" for argv in argvs)


@pytest.mark.parametrize(
    ("field", "unsafe"),
    [
        ("MaskedPaths", CAPTURED_MOBY_MASKED_PATHS + ["relative/path"]),
        ("ReadonlyPaths", CAPTURED_MOBY_READONLY_PATHS + ["/proc/sys"]),
    ],
)
def test_moby_proc_restriction_invariant_rejects_relative_or_duplicate_paths(
    tmp_path: Path, field: str, unsafe: list[str]
) -> None:
    def mutate(payload: dict[str, object]) -> None:
        payload["HostConfig"][field] = unsafe

    runtime = FakeDockerRuntime(inspect_mutator=mutate)
    with pytest.raises(ReplayContractError, match="policy"):
        _docker_execute(runtime, tmp_path)

    assert not any(call[0][1] == "start" for call in runtime.calls)


def test_moby_proc_restriction_invariant_allows_additional_safe_masks(
    tmp_path: Path,
) -> None:
    def mutate(payload: dict[str, object]) -> None:
        payload["HostConfig"]["MaskedPaths"].append("/sys/devices/virtual/powercap")
        payload["HostConfig"]["ReadonlyPaths"].append("/proc/pressure")

    evidence = _docker_execute(
        FakeDockerRuntime(inspect_mutator=mutate), tmp_path
    )

    assert evidence.resolved is True


@pytest.mark.parametrize("drift", ["missing_mount", "wrong_mount_options", "zero_pid"])
def test_docker_executor_rejects_running_policy_drift_before_verifier(
    tmp_path: Path, drift: str
) -> None:
    def mutate(payload: dict[str, object]) -> None:
        if payload["State"]["Running"] is not True:
            return
        if drift == "missing_mount":
            payload["Mounts"].pop()
        elif drift == "wrong_mount_options":
            payload["Mounts"][0]["Mode"] = "rw,size=1"
        else:
            payload["State"]["Pid"] = 0

    runtime = FakeDockerRuntime(inspect_mutator=mutate)
    with pytest.raises(ReplayContractError, match="policy|mount|running"):
        _docker_execute(runtime, tmp_path)
    argvs = [call[0] for call in runtime.calls]

    assert not any(
        argv[1] == "exec" and any(value.endswith("/verifier.sh") for value in argv)
        for argv in argvs
    )


def test_docker_executor_timeout_is_bounded_and_cleaned(tmp_path: Path) -> None:
    runtime = FakeDockerRuntime(timed_out=True)
    evidence = _docker_execute(runtime, tmp_path)

    assert evidence.resolved is False
    assert evidence.termination == "wall_timeout"
    assert evidence.failure_class == "verifier_timeout"
    assert evidence.cleanup_state == "verified_removed"
    assert evidence.raw_output_bytes <= _docker_policy().output_limit_bytes


def test_docker_executor_applies_one_wall_budget_to_the_running_container(
    tmp_path: Path,
) -> None:
    class AdvancingRuntime(FakeDockerRuntime):
        def __init__(self) -> None:
            super().__init__()
            self.now = 100.0
            self.started = False

        def run(self, argv, **kwargs):
            result = super().run(argv, **kwargs)
            if argv[1] == "start":
                self.started = True
            if self.started:
                self.now += 1.0
            return result

    runtime = AdvancingRuntime()
    _docker_execute(
        runtime,
        tmp_path,
        effective_timeout=30,
        clock=lambda: runtime.now,
    )
    lifecycle_calls = []
    running = False
    for call in runtime.calls:
        if call[0][1] == "start":
            running = True
            continue
        if running and call[0][1] != "logs":
            lifecycle_calls.append(call)
        if call[0][1] == "logs":
            break

    timeouts = [call[1] for call in lifecycle_calls[:-1]]
    assert timeouts[0] < 30
    assert timeouts == sorted(timeouts, reverse=True)
    assert timeouts[-1] < timeouts[0]


def test_verifier_receives_the_full_remaining_effective_timeout(tmp_path: Path) -> None:
    runtime = FakeDockerRuntime()

    _docker_execute(runtime, tmp_path, effective_timeout=300, clock=lambda: 100.0)

    verifier_call = next(
        call
        for call in runtime.calls
        if any(value.endswith("/verifier.sh") for value in call[0])
    )
    assert verifier_call[1] == 300


def test_docker_executor_caps_output_and_records_truncation(tmp_path: Path) -> None:
    limit = _docker_policy().output_limit_bytes
    runtime = FakeDockerRuntime(
        verifier_output=b"x" * (limit + 100), output_truncated=True
    )
    evidence = _docker_execute(runtime, tmp_path)

    assert evidence.raw_output_bytes == limit
    assert evidence.output_truncated is True


def test_docker_executor_separates_wrapper_failure_from_verifier_rc(tmp_path: Path) -> None:
    runtime = FakeDockerRuntime(result_updates={"wrapper_rc": 1})
    evidence = _docker_execute(runtime, tmp_path)

    assert evidence.returncode == 0
    assert evidence.wrapper_returncode == 1
    assert evidence.resolved is False
    assert evidence.failure_class == "wrapper_failed"


def test_docker_executor_detects_protected_drift(tmp_path: Path) -> None:
    runtime = FakeDockerRuntime(
        result_updates={"protected_sha256": ["9" * 64]}
    )
    evidence = _docker_execute(runtime, tmp_path)

    assert evidence.resolved is False
    assert evidence.failure_class == "protected_drift"
    assert evidence.protected_before != evidence.protected_after


def test_docker_executor_quiesces_verifier_uid_before_hash_signal(tmp_path: Path) -> None:
    runtime = FakeDockerRuntime()
    _docker_execute(runtime, tmp_path)
    argvs = [call[0] for call in runtime.calls]
    verifier_index = next(
        index
        for index, argv in enumerate(argvs)
        if any(value.endswith("/verifier.sh") for value in argv)
    )
    quiesce = argvs[verifier_index + 1]
    signal = argvs[verifier_index + 2]

    assert quiesce[1:4] == ("exec", "--user", "65532:65532")
    assert any("/proc/[0-9]*" in value for value in quiesce)
    assert any("/control/go" in value for value in signal)


def test_docker_executor_rejects_pre_state_drift_for_tainted_controls(
    tmp_path: Path,
) -> None:
    candidate = tmp_path / "candidate"
    candidate.mkdir()
    (candidate / "src.py").write_text("old\n")
    runtime = FakeDockerRuntime()
    executor = replay.DockerExecutor(
        _docker_policy(),
        runner=runtime,
        disk_free_bytes=runtime.disk_free_bytes,
        token_factory=lambda: "a" * 32,
    )

    with pytest.raises(ReplayContractError, match="tree changed"):
        executor.execute(
            candidate_root=candidate,
            language="python",
            verifier_text="python3 -m pytest -q",
            effective_timeout=300,
            control_identity="baseline",
            pre_candidate_tree_sha256="b" * 64,
            pre_candidate_diff_sha256="e" * 64,
            protected_before=(),
        )

    assert runtime.calls == []


def test_docker_executor_enforces_the_40_gib_floor_before_create(tmp_path: Path) -> None:
    runtime = FakeDockerRuntime(free_bytes=(39 << 30,))
    with pytest.raises(ReplayContractError, match="disk floor"):
        _docker_execute(runtime, tmp_path)
    assert not runtime.calls


def test_docker_executor_enforces_disk_floor_after_exact_cleanup(tmp_path: Path) -> None:
    runtime = FakeDockerRuntime(free_bytes=(80 << 30, 39 << 30))
    with pytest.raises(ReplayContractError, match="disk floor"):
        _docker_execute(runtime, tmp_path)

    argvs = [call[0] for call in runtime.calls]
    assert ("docker", "stop", "--time=2", CID) in argvs
    assert ("docker", "rm", "--force", "--volumes", CID) in argvs


@pytest.mark.parametrize(
    ("language", "needle"),
    [("python", "python3"), ("rust", "rustc"), ("cpp", "c++")],
)
def test_functional_admission_probe_uses_full_policy_and_real_toolchain(
    tmp_path: Path, language: str, needle: str
) -> None:
    digest = f"example.invalid/fable-{language}@sha256:" + "1" * 64
    policy = replay.DockerPolicy(images=((language, digest),))
    runtime = FakeDockerRuntime()
    executor = replay.DockerExecutor(
        policy,
        runner=runtime,
        disk_free_bytes=runtime.disk_free_bytes,
        token_factory=lambda: "a" * 32,
    )

    evidence = replay.functional_admission_probe(
        policy, language, executor=executor, workspace=tmp_path / language
    )

    assert evidence.schema_version == 1
    assert evidence.admitted is True
    assert evidence.language == language
    assert needle in runtime.verifier_script
    assert all(call[0][1] != "pull" for call in runtime.calls)


def _run_evidence(
    *,
    identity: str,
    tree: str,
    diff: str,
    protected: tuple[tuple[str, str], ...],
    rc: int,
    output_hash: str,
    policy_version: str = "fable-docker-v1",
    image_digest: str = IMAGE_DIGEST,
    runtime_version: str = "27.5.1",
    run_contract_sha256: str = "f" * 64,
) -> object:
    return replay.RunEvidence(
        schema_version=1,
        run_id="a" * 32,
        control_identity=identity,
        trainable=identity == "candidate",
        returncode=rc,
        wrapper_returncode=0,
        duration_seconds=1.0,
        termination="exited",
        raw_output_sha256=output_hash,
        raw_output_bytes=10,
        output_truncated=False,
        pre_candidate_tree_sha256=tree,
        pre_candidate_diff_sha256=diff,
        post_candidate_tree_sha256="d" * 64,
        protected_before=protected,
        protected_after=protected,
        resolved=rc == 0,
        failure_class=None if rc == 0 else "verifier_failed",
        cleanup_state="verified_removed",
        policy_version=policy_version,
        image_digest=image_digest,
        runtime_version=runtime_version,
        run_contract_sha256=run_contract_sha256,
        resource_peaks=(("memory_bytes", 1024), ("pids", 3), ("cpu_usec", 4000)),
    )


def _verified_replay_evidence(
    trajectory_id: str = "1" * 64,
    *,
    task: str = "py-safe",
    language: str = "python",
) -> replay.ReplayEvidence:
    tree = "2" * 64
    diff = "3" * 64
    protected = (("test_src.py", "4" * 64),)
    run_contract = "5" * 64
    runs = tuple(
        dataclasses.replace(
            _run_evidence(
                identity="candidate",
                tree=tree,
                diff=diff,
                protected=protected,
                rc=0,
                output_hash=("6" if index == 0 else "7") * 64,
                run_contract_sha256=run_contract,
            ),
            run_id=("a" if index == 0 else "b") * 32,
        )
        for index in range(2)
    )
    outer_contract = replay._sha256(
        replay._canonical_json(
            {
                "schema_version": 2,
                "dataset_revision": DATASET_REVISION,
                "source_commit_sha": MOONSHINER_REVISION,
                "source_tree_sha": "8" * 64,
                "task_tree_sha": "9" * 64,
                "inventory_sha256": "c" * 64,
                "trajectory_id": trajectory_id,
                "source_terminal_sha256": "d" * 64,
                "operation_sha256": "e" * 64,
                "candidate_tree_sha256": tree,
                "candidate_diff_sha256": diff,
                "verifier_sha256": hashlib.sha256(VERIFY_CMD.encode()).hexdigest(),
                "source_verify_timeout": None,
                "effective_verify_timeout": 300,
                "policy_output_limit_bytes": 4 * 1024**2,
                "executor_runs": [run.run_contract_sha256 for run in runs],
            }
        )
    )
    return replay.ReplayEvidence(
        schema_version=2,
        trajectory_id=trajectory_id,
        task=task,
        language=language,
        dataset_revision=DATASET_REVISION,
        source_commit_sha=MOONSHINER_REVISION,
        source_tree_sha="8" * 64,
        task_tree_sha="9" * 64,
        inventory_sha256="c" * 64,
        source_terminal_sha256="d" * 64,
        operation_sha256="e" * 64,
        candidate_tree_sha256=tree,
        candidate_diff_sha256=diff,
        protected_sha256=protected,
        verifier_text=VERIFY_CMD,
        verifier_sha256=hashlib.sha256(VERIFY_CMD.encode()).hexdigest(),
        source_verify_timeout=None,
        effective_verify_timeout=300,
        policy_version="fable-docker-v1",
        image_digest=IMAGE_DIGEST,
        policy_output_limit_bytes=4 * 1024**2,
        runtime_version="27.5.1",
        run_contract_sha256=outer_contract,
        runs=runs,
        resolved=True,
        failure_class=None,
        control_identity="candidate",
        trainable=True,
    )


def test_locked_ledger_publishes_0600_log_before_atomic_record(tmp_path: Path) -> None:
    evidence = _verified_replay_evidence()
    ledger_path = tmp_path / "replay.jsonl"
    seen: list[tuple[bool, bool]] = []
    with replay.ReplayLedger(ledger_path, tmp_path / "logs") as ledger:
        ledger.publish_verified(
            evidence,
            source_content_sha256="f" * 64,
            fixture_sha256=evidence.inventory_sha256,
            log=b"strict verifier output\n",
            after_log_publish=lambda path: seen.append(
                (path.exists(), ledger_path.exists())
            ),
        )
    assert seen == [(True, False)]
    assert (
        stat.S_IMODE(
            (tmp_path / "logs" / f"{evidence.trajectory_id}.log").stat().st_mode
        )
        == 0o600
    )
    assert stat.S_IMODE(ledger_path.stat().st_mode) == 0o600
    assert not list(tmp_path.rglob("*.tmp"))


def test_ledger_lock_full_validation_resume_and_duplicate_rejection(tmp_path: Path) -> None:
    evidence = _verified_replay_evidence()
    path = tmp_path / "replay.jsonl"
    with replay.ReplayLedger(path, tmp_path / "logs") as first:
        with pytest.raises(ReplayContractError, match="locked"):
            with replay.ReplayLedger(path, tmp_path / "logs"):
                pass
        first.publish_verified(
            evidence,
            source_content_sha256="f" * 64,
            fixture_sha256=evidence.inventory_sha256,
            log=b"ok",
        )
    with replay.ReplayLedger(path, tmp_path / "logs") as resumed:
        assert resumed.completed_trajectory_ids == frozenset({evidence.trajectory_id})
        with pytest.raises(ReplayContractError, match="duplicate trajectory"):
            resumed.publish_verified(
                evidence,
                source_content_sha256="f" * 64,
                fixture_sha256=evidence.inventory_sha256,
                log=b"again",
            )
    os.chmod(path, 0o644)
    with pytest.raises(ReplayContractError, match="mode is not 0600"):
        with replay.ReplayLedger(path, tmp_path / "logs"):
            pass
    os.chmod(path, 0o600)
    payload = json.loads(path.read_text().splitlines()[0])
    payload["evidence"]["schema_version"] = 1
    path.write_text(json.dumps(payload) + "\n")
    with pytest.raises(ReplayContractError, match="schema_version"):
        with replay.ReplayLedger(path, tmp_path / "logs"):
            pass


def test_ledger_refuses_symlink_lock_and_substituted_log(tmp_path: Path) -> None:
    path = tmp_path / "replay.jsonl"
    lock = tmp_path / ".replay.jsonl.lock"
    outside = tmp_path / "outside"
    outside.write_text("do not chmod")
    lock.symlink_to(outside)
    with pytest.raises(ReplayContractError, match="lock"):
        with replay.ReplayLedger(path, tmp_path / "logs"):
            pass
    lock.unlink()

    evidence = _verified_replay_evidence()
    with replay.ReplayLedger(path, tmp_path / "logs") as ledger:
        ledger.publish_verified(
            evidence,
            source_content_sha256="f" * 64,
            fixture_sha256=evidence.inventory_sha256,
            log=b"ok",
        )
    log_path = tmp_path / "logs" / f"{evidence.trajectory_id}.log"
    log_path.unlink()
    log_path.symlink_to(outside)
    with pytest.raises(ReplayContractError, match="log"):
        with replay.ReplayLedger(path, tmp_path / "logs"):
            pass


def test_unlinked_lock_rejects_second_owner_and_blocks_first_publish(
    tmp_path: Path,
) -> None:
    path = tmp_path / "replay.jsonl"
    lock_path = tmp_path / ".replay.jsonl.lock"
    with replay.ReplayLedger(path, tmp_path / "logs") as first:
        lock_path.unlink()
        with pytest.raises(ReplayContractError, match="locked"):
            with replay.ReplayLedger(path, tmp_path / "logs"):
                pass
        with pytest.raises(ReplayContractError, match="lock identity"):
            first.publish_failure(
                trajectory_id="1" * 64,
                status="rejected",
                failure_class="unsupported_operation",
                log=b"bounded detail",
            )
    assert not path.exists()
    assert not (tmp_path / "logs" / f"{'1' * 64}.log").exists()


def test_external_lock_failure_releases_parent_and_log_descriptors(
    tmp_path: Path,
) -> None:
    path = tmp_path / "replay.jsonl"
    lock_path = tmp_path / ".replay.jsonl.lock"
    lock_fd = os.open(lock_path, os.O_RDWR | os.O_CREAT, 0o600)
    fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    try:
        with pytest.raises(ReplayContractError, match="already locked"):
            with replay.ReplayLedger(path, tmp_path / "logs"):
                pass
    finally:
        fcntl.flock(lock_fd, fcntl.LOCK_UN)
        os.close(lock_fd)

    with replay.ReplayLedger(path, tmp_path / "logs"):
        pass


def test_replaced_log_directory_blocks_ledger_publication(tmp_path: Path) -> None:
    path = tmp_path / "replay.jsonl"
    evidence = _verified_replay_evidence()

    def replace_logs(_path: Path) -> None:
        (tmp_path / "logs").rename(tmp_path / "moved")
        (tmp_path / "logs").mkdir(mode=0o700)

    with replay.ReplayLedger(path, tmp_path / "logs") as ledger:
        with pytest.raises(ReplayContractError, match="log directory identity"):
            ledger.publish_verified(
                evidence,
                source_content_sha256="f" * 64,
                fixture_sha256=evidence.inventory_sha256,
                log=b"orphaned but not published",
                after_log_publish=replace_logs,
            )
    assert not path.exists()
    assert not (tmp_path / "logs" / f"{evidence.trajectory_id}.log").exists()


def test_ledger_rejects_symlinked_parent_component(tmp_path: Path) -> None:
    real = tmp_path / "real"
    real.mkdir(mode=0o700)
    apparent = tmp_path / "apparent"
    apparent.symlink_to(real, target_is_directory=True)
    with pytest.raises(ReplayContractError, match="directory"):
        with replay.ReplayLedger(apparent / "replay.jsonl", apparent / "logs"):
            pass
    assert not (real / ".replay.jsonl.lock").exists()


def test_atomic_publish_detects_post_rename_swap_without_touching_victim(
    tmp_path: Path,
) -> None:
    victim = tmp_path / "victim"
    victim.write_bytes(b"victim")
    os.chmod(victim, 0o644)
    target = tmp_path / "target"

    def swap(_directory_fd: int, name: str) -> None:
        os.unlink(name, dir_fd=_directory_fd)
        os.symlink(victim, name, dir_fd=_directory_fd)

    with pytest.raises(ReplayContractError, match="publication identity"):
        replay._atomic_write_0600(target, b"published", after_rename=swap)
    assert victim.read_bytes() == b"victim"
    assert stat.S_IMODE(victim.stat().st_mode) == 0o644


def test_atomic_publish_rejects_replaced_parent_directory(tmp_path: Path) -> None:
    output = tmp_path / "out"
    output.mkdir(mode=0o700)
    target = output / "eligibility.json"

    def replace_parent(_directory_fd: int, _name: str) -> None:
        output.rename(tmp_path / "moved")
        output.mkdir(mode=0o700)

    with pytest.raises(ReplayContractError, match="parent directory identity"):
        replay._atomic_write_0600(target, b"published", after_rename=replace_parent)
    assert not target.exists()
    assert (tmp_path / "moved" / "eligibility.json").read_bytes() == b"published"


def test_ledger_records_rejection_timeout_and_rejects_contract_alias(tmp_path: Path) -> None:
    path = tmp_path / "replay.jsonl"
    with replay.ReplayLedger(path, tmp_path / "logs") as ledger:
        ledger.publish_failure(
            trajectory_id="1" * 64,
            status="rejected",
            failure_class="unsupported_operation",
            log=b"rejected",
        )
        ledger.publish_failure(
            trajectory_id="2" * 64,
            status="timeout",
            failure_class="verifier_timeout",
            log=b"timed out",
        )
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    assert [row["status"] for row in rows] == ["rejected", "timeout"]
    assert all(row["evidence"] is None for row in rows)

    bad = _verified_replay_evidence("3" * 64, task="other")
    bad = dataclasses.replace(bad, run_contract_sha256="0" * 64)
    with replay.ReplayLedger(path, tmp_path / "logs") as ledger:
        with pytest.raises(ReplayContractError, match="run contract"):
            ledger.publish_verified(
                bad,
                source_content_sha256="f" * 64,
                fixture_sha256=bad.inventory_sha256,
                log=b"bad",
            )


@pytest.mark.parametrize(
    "failure_class",
    ["unknown_reason", "contains\nnewline", "x" * 65, "secret=token"],
)
def test_failure_code_is_reviewed_and_validated_before_log_publish(
    tmp_path: Path, failure_class: str
) -> None:
    trajectory_id = "1" * 64
    with replay.ReplayLedger(tmp_path / "replay.jsonl", tmp_path / "logs") as ledger:
        with pytest.raises(ReplayContractError, match="failure code"):
            ledger.publish_failure(
                trajectory_id=trajectory_id,
                status="rejected",
                failure_class=failure_class,
                log=b"detail",
            )
    assert not (tmp_path / "logs" / f"{trajectory_id}.log").exists()


def test_v2_decoder_rejects_malformed_image_and_output_above_policy_limit() -> None:
    evidence = _verified_replay_evidence()
    malformed = replay.replay_evidence_payload(evidence)
    malformed["image_digest"] = "evil@sha256:nothex"
    for run in malformed["runs"]:
        run["image_digest"] = malformed["image_digest"]
    with pytest.raises(ReplayContractError, match="candidate namespace"):
        replay.validate_replay_evidence_payload(malformed)

    bounded = dataclasses.replace(evidence, policy_output_limit_bytes=10)
    oversized = replay.replay_evidence_payload(bounded)
    for run in oversized["runs"]:
        run["raw_output_bytes"] = 11
    with pytest.raises(ReplayContractError, match="strict candidate runs"):
        replay.validate_replay_evidence_payload(oversized)


def test_v2_output_limit_cannot_be_inflated_by_self_declared_evidence() -> None:
    evidence = _verified_replay_evidence()
    inflated_runs = tuple(
        dataclasses.replace(run, raw_output_bytes=10**18) for run in evidence.runs
    )
    inflated = dataclasses.replace(
        evidence,
        policy_output_limit_bytes=10**18,
        runs=inflated_runs,
    )
    inflated = dataclasses.replace(
        inflated,
        run_contract_sha256=replay._outer_replay_contract(inflated),
    )
    with pytest.raises(ReplayContractError, match="candidate namespace"):
        replay.validate_replay_evidence_payload(
            replay.replay_evidence_payload(inflated)
        )

    with pytest.raises(ValueError, match="output limit"):
        replay.DockerPolicy(
            images=(("python", "image@sha256:" + "a" * 64),),
            output_limit_bytes=10**18,
        )


def test_inventory_arithmetic_and_explicit_smoke_manifest_are_exhaustive() -> None:
    def candidate(
        trajectory_id: str, task: str, language: str, **updates: bool
    ) -> replay.EligibilityCandidate:
        gates = {
            "operations_supported": True,
            "decontaminated": True,
            "git_seed_valid": True,
            "reference_patch_valid": True,
            "language_digest_present": True,
            "admission_valid": True,
        }
        gates.update(updates)
        return replay.EligibilityCandidate(
            trajectory_id, task, language, **gates
        )

    rows = [
        candidate("1" * 64, "py-a", "python"),
        candidate("2" * 64, "py-b", "python"),
        candidate("3" * 64, "rs-a", "rust"),
        candidate("4" * 64, "rs-b", "rust"),
        candidate("5" * 64, "cpp-a", "cpp"),
        candidate(
            "6" * 64, "drop-op", "python", operations_supported=False
        ),
        candidate(
            "7" * 64, "drop-admit", "rust", admission_valid=False
        ),
        candidate("8" * 64, "drop-contam", "python", decontaminated=False),
        candidate("9" * 64, "drop-seed", "rust", git_seed_valid=False),
        candidate(
            "a" * 64, "drop-reference", "cpp", reference_patch_valid=False
        ),
        candidate(
            "b" * 64, "drop-image", "cpp", language_digest_present=False
        ),
    ]
    bindings = replay.EligibilityBindings(
        source_sha256="8" * 64,
        sidecar_sha256="9" * 64,
        policy_sha256="a" * 64,
        admission_sha256="b" * 64,
        policy_artifact_sha256="1" * 64,
        admission_artifact_sha256="2" * 64,
        smoke_manifest_sha256="3" * 64,
        exclusion_artifacts=(("evaluation.json", "e" * 64),),
        seed_evidence_sha256="d" * 64,
        seed_commit_sha=MOONSHINER_REVISION,
        seed_tree_sha="c" * 40,
    )
    inventory = replay.build_eligibility_inventory(rows, bindings)
    assert inventory["total"] == 11
    assert inventory["eligible_ceiling"] == 5
    assert set(inventory["exclusions"].values()) == {1}
    assert inventory["total"] == inventory["eligible_ceiling"] + sum(
        inventory["exclusions"].values()
    )
    smoke = replay.build_smoke_manifest(
        inventory,
        ["5" * 64, "4" * 64, "2" * 64, "3" * 64, "1" * 64],
    )
    assert [row["language"] for row in smoke["trajectories"]] == [
        "python", "python", "rust", "rust", "cpp"
    ]
    assert smoke["eligibility_sha256"] == inventory["inventory_sha256"]
    with pytest.raises(
        ReplayContractError, match=r"exactly 2 Python, 2 Rust, and 1 C\+\+"
    ):
        replay.build_smoke_manifest(inventory, ["1" * 64] * 5)


def _admission_for(policy: replay.DockerPolicy, language: str) -> replay.AdmissionEvidence:
    tree = replay._admission_tree_sha256()
    diff = hashlib.sha256(b"admission").hexdigest()
    image = policy.image_for(language)
    command = replay._ADMISSION_COMMANDS[language]
    contract = replay._executor_run_contract_sha256(
        policy,
        image,
        language=language,
        verifier_text=command,
        effective_timeout=60,
        control_identity="admission",
        pre_candidate_tree_sha256=tree,
        pre_candidate_diff_sha256=diff,
        protected_before=(),
    )
    run = dataclasses.replace(
        _run_evidence(
            identity="admission",
            tree=tree,
            diff=diff,
            protected=(),
            rc=0,
            output_hash="2" * 64,
            image_digest=image,
            policy_version=policy.policy_version,
            run_contract_sha256=contract,
        ),
        trainable=False,
    )
    return replay.AdmissionEvidence(
        1,
        language,
        True,
        policy.policy_version,
        image,
        policy.output_limit_bytes,
        run.runtime_version,
        run,
        None,
    )


def test_admission_artifact_is_bound_to_exact_policy_digest_and_hash() -> None:
    policy = dataclasses.replace(
        _docker_policy(),
        images=(
            ("python", "example.invalid/python@sha256:" + "1" * 64),
            ("rust", "example.invalid/rust@sha256:" + "2" * 64),
            ("cpp", "example.invalid/cpp@sha256:" + "3" * 64),
        ),
    )
    policy_document = replay.docker_policy_artifact(policy)
    admission = replay.admission_artifact(
        policy, [_admission_for(policy, language) for language in ("python", "rust", "cpp")]
    )
    validated = replay.validate_admission_artifact(policy_document, admission)
    assert set(validated) == {"python", "rust", "cpp"}

    changed_policy = dataclasses.replace(
        policy,
        images=tuple(
            (language, image[:-1] + ("0" if image[-1] != "0" else "1"))
            for language, image in policy.images
        ),
    )
    with pytest.raises(ReplayContractError, match="admission.*policy"):
        replay.validate_admission_artifact(
            replay.docker_policy_artifact(changed_policy), admission
        )
    tampered = json.loads(json.dumps(admission))
    tampered["records"][0]["evidence"]["runtime_version"] = "99.0.0"
    with pytest.raises(ReplayContractError, match="admission.*hash"):
        replay.validate_admission_artifact(policy_document, tampered)


def test_structural_sidecar_recomputes_source_identity_and_rejects_tampering() -> None:
    row = {
        "task": "py-bound",
        "lang": "python",
        "category": "debug",
        "split": "train",
        "assistant_step": 1,
        "assistant_steps": 1,
        "messages": [
            {"role": "system", "content": "work"},
            {"role": "user", "content": "fix"},
            {"role": "assistant", "content": "done"},
        ],
        "tools": "[]",
    }
    trajectory = replay.trajectory_identity("py-bound", row)
    terminal_sha = hashlib.sha256(replay._canonical_json(row)).hexdigest()
    sidecar = {
        "trajectory_id": trajectory,
        "source_instance_id": "py-bound",
        "source_terminal_sha256": terminal_sha,
        "row": row,
    }
    records = replay.validate_structural_sidecar([row], [sidecar])
    assert records[0]["trajectory_id"] == trajectory
    tampered = json.loads(json.dumps(sidecar))
    tampered["row"]["task"] = "other"
    with pytest.raises(ReplayContractError, match="sidecar"):
        replay.validate_structural_sidecar([row], [tampered])


def test_seed_repository_requires_pinned_commit_and_tasks_tree(tmp_path: Path) -> None:
    repository = tmp_path / "seed.git"
    repository.mkdir()

    def good_runner(argv: tuple[str, ...], _env: dict[str, str]) -> bytes:
        tail = argv[4:]
        if tail == ("cat-file", "-e", f"{MOONSHINER_REVISION}^{{commit}}"):
            return b""
        if tail == ("rev-parse", f"{MOONSHINER_REVISION}^{{commit}}"):
            return (MOONSHINER_REVISION + "\n").encode()
        if tail == ("rev-parse", f"{MOONSHINER_REVISION}:tasks/seeds"):
            return ("a" * 40 + "\n").encode()
        raise subprocess.CalledProcessError(1, argv)

    evidence = replay.validate_seed_repository(
        GitSeedSource(repository, MOONSHINER_REVISION, good_runner)
    )
    assert evidence["tasks_seed_tree_sha"] == "a" * 40

    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(ReplayContractError, match="pinned seed repository"):
        replay.validate_seed_repository(GitSeedSource(empty, MOONSHINER_REVISION))

    def wrong_runner(argv: tuple[str, ...], env: dict[str, str]) -> bytes:
        result = good_runner(argv, env)
        if argv[4:] == ("rev-parse", f"{MOONSHINER_REVISION}^{{commit}}"):
            return ("b" * 40 + "\n").encode()
        return result

    with pytest.raises(ReplayContractError, match="commit mismatch"):
        replay.validate_seed_repository(
            GitSeedSource(repository, MOONSHINER_REVISION, wrong_runner)
        )


def test_replay_cli_requires_every_pinned_inventory_input() -> None:
    with pytest.raises(SystemExit):
        replay.parse_args([])


def test_eligibility_candidate_requires_all_six_explicit_gates() -> None:
    with pytest.raises(TypeError):
        replay.EligibilityCandidate("1" * 64, "py", "python")


def test_eligibility_binds_canonical_ordered_exclusion_artifacts() -> None:
    candidate = replay.EligibilityCandidate(
        "1" * 64,
        "py",
        "python",
        True,
        True,
        True,
        True,
        True,
        True,
    )
    common = {
        "source_sha256": "8" * 64,
        "sidecar_sha256": "9" * 64,
        "policy_sha256": "a" * 64,
        "admission_sha256": "b" * 64,
        "policy_artifact_sha256": "1" * 64,
        "admission_artifact_sha256": "2" * 64,
        "smoke_manifest_sha256": "3" * 64,
        "seed_evidence_sha256": "d" * 64,
        "seed_commit_sha": MOONSHINER_REVISION,
        "seed_tree_sha": "c" * 40,
    }
    first = replay.build_eligibility_inventory(
        [candidate],
        replay.EligibilityBindings(
            **common,
            exclusion_artifacts=(("evaluation.json", "e" * 64),),
        ),
    )
    changed = replay.build_eligibility_inventory(
        [candidate],
        replay.EligibilityBindings(
            **common,
            exclusion_artifacts=(("evaluation.json", "f" * 64),),
        ),
    )
    assert first["inventory_sha256"] != changed["inventory_sha256"]
    changed_smoke_input = replay.build_eligibility_inventory(
        [candidate],
        replay.EligibilityBindings(
            **{
                **common,
                "smoke_manifest_sha256": "4" * 64,
            },
            exclusion_artifacts=(("evaluation.json", "e" * 64),),
        ),
    )
    assert first["inventory_sha256"] != changed_smoke_input["inventory_sha256"]
    with pytest.raises(ReplayContractError, match="exclusion artifact"):
        replay.build_eligibility_inventory(
            [candidate],
            replay.EligibilityBindings(
                **common,
                exclusion_artifacts=(
                    ("z.json", "e" * 64),
                    ("a.json", "f" * 64),
                ),
            ),
        )


@pytest.mark.parametrize(
    "digest",
    [
        "registry.example/python@sha256:",
        "registry.example/python@sha256:" + "A" * 64,
        "registry.example/python@sha256:" + "a" * 63,
        "registry.example/python@sha256:" + "a" * 64 + "junk",
        "registry.example/python:latest",
    ],
)
def test_inventory_cli_rejects_nonexact_policy_digest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, digest: str
) -> None:
    source = tmp_path / "source.jsonl"
    source.write_bytes(b"pinned fixture")
    monkeypatch.setattr(
        replay,
        "SOURCE_LFS_SHA256",
        hashlib.sha256(source.read_bytes()).hexdigest(),
    )
    monkeypatch.setattr(
        replay, "SOURCE_BYTES", source.stat().st_size, raising=False
    )
    sidecar = tmp_path / "sidecar.jsonl"
    sidecar.write_text("")
    seed_repo = tmp_path / "seed"
    seed_repo.mkdir()
    policy = tmp_path / "policy.json"
    policy_document = replay.docker_policy_artifact(_docker_policy())
    policy_document["policy"]["images"] = [["python", digest]]
    policy_document["policy_sha256"] = hashlib.sha256(
        replay._canonical_json(policy_document["policy"])
    ).hexdigest()
    policy.write_text(json.dumps(policy_document))
    admission = tmp_path / "admission.json"
    admission.write_text("{}")
    exclusion = tmp_path / "exclusion.json"
    exclusion.write_text("[]")
    smoke = tmp_path / "smoke.json"
    smoke.write_text("[]")
    with pytest.raises(ReplayContractError, match="policy artifact values"):
        replay.main(
            [
                "--source",
                str(source),
                "--sidecar",
                str(sidecar),
                "--seed-repo",
                str(seed_repo),
                "--policy",
                str(policy),
                "--admission",
                str(admission),
                "--smoke-manifest",
                str(smoke),
                "--exclusion",
                str(exclusion),
                "--ledger",
                str(tmp_path / "replay.jsonl"),
                "--logs",
                str(tmp_path / "logs"),
                "--out",
                str(tmp_path / "out"),
                "--inventory-only",
            ]
        )


def _inventory_cli_row(task: str, language: str) -> dict[str, object]:
    edit_id = f"edit-{task}"
    verify_id = f"verify-{task}"
    messages: list[dict[str, object]] = [
        {"role": "system", "content": "Work carefully."},
        {"role": "user", "content": f"Fix {task}."},
        {
            "role": "assistant",
            "content": "Apply the narrow fix.",
            "tool_calls": [
                {
                    "id": edit_id,
                    "type": "function",
                    "function": {
                        "name": "Edit",
                        "arguments": {
                            "file_path": "src.txt",
                            "old_string": "old",
                            "new_string": "new",
                            "replace_all": False,
                        },
                    },
                }
            ],
        },
        {"role": "tool", "tool_call_id": edit_id, "content": "updated"},
        {
            "role": "assistant",
            "content": "Verify the fix.",
            "tool_calls": [
                {
                    "id": verify_id,
                    "type": "function",
                    "function": {
                        "name": "Bash",
                        "arguments": {"command": "true"},
                    },
                }
            ],
        },
        {"role": "tool", "tool_call_id": verify_id, "content": "ok"},
        {"role": "assistant", "content": "Implemented and verified."},
    ]
    return {
        "task": task,
        "lang": language,
        "category": "debug",
        "split": "train",
        "assistant_step": 3,
        "assistant_steps": 3,
        "messages": messages,
        "tools": "[]",
    }


def _write_inventory_seed(
    repository: Path,
    task: str,
    language: str,
    *,
    include_patch: bool = True,
) -> None:
    seed = repository / "tasks" / "seeds" / task
    files = seed / "files"
    files.mkdir(parents=True)
    (seed / "task.json").write_text(
        json.dumps(
            {
                "id": task,
                "lang": language,
                "verify_cmd": "true",
                "test_files": ["test.txt"],
            }
        ),
        encoding="utf-8",
    )
    if include_patch:
        (seed / "reference_fix.patch").write_text(
            "diff --git a/src.txt b/src.txt\n"
            "--- a/src.txt\n"
            "+++ b/src.txt\n"
            "@@ -1 +1 @@\n"
            "-old\n"
            "+new\n",
            encoding="utf-8",
        )
    (files / "src.txt").write_text("old\n", encoding="utf-8")
    (files / "test.txt").write_text("ok\n", encoding="utf-8")


def _inventory_cli_fixture(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    include_extra: bool = False,
    missing_patch_task: str | None = None,
    invalid_task: tuple[str, str] | None = None,
    source_language_override: tuple[str, str] | None = None,
    exclusion_payload: object = (),
    out_name: str = "out",
) -> tuple[list[str], Path, Path]:
    tasks = [
        ("py-one", "python"),
        ("py-two", "python"),
        ("rs-one", "rust"),
        ("rs-two", "rust"),
        ("cpp-one", "cpp"),
    ]
    if include_extra:
        tasks.append(("py-extra", "python"))
    rows = [_inventory_cli_row(task, language) for task, language in tasks]
    if source_language_override is not None:
        task, language = source_language_override
        next(item for item in rows if item["task"] == task)["lang"] = language
    if invalid_task is not None:
        task, kind = invalid_task
        row = next(item for item in rows if item["task"] == task)
        if kind == "selection":
            del row["category"]
        elif kind == "conversion":
            calls = row["messages"][2]["tool_calls"]  # type: ignore[index]
            calls[0]["function"]["name"] = "Unsupported"  # type: ignore[index]
        else:  # pragma: no cover - fixture guard
            raise AssertionError(kind)

    repository = tmp_path / "seed-repo"
    repository.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=repository, check=True)
    subprocess.run(
        ["git", "config", "user.email", "test@example.com"],
        cwd=repository,
        check=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "Test"], cwd=repository, check=True
    )
    for task, language in tasks:
        _write_inventory_seed(
            repository,
            task,
            language,
            include_patch=task != missing_patch_task,
        )
    subprocess.run(["git", "add", "tasks"], cwd=repository, check=True)
    subprocess.run(["git", "commit", "-qm", "fixtures"], cwd=repository, check=True)
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    monkeypatch.setattr(replay, "MOONSHINER_REVISION", commit)

    source = tmp_path / "source.jsonl"
    source.write_text(
        "".join(json.dumps(row, separators=(",", ":")) + "\n" for row in rows),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        replay,
        "SOURCE_LFS_SHA256",
        hashlib.sha256(source.read_bytes()).hexdigest(),
    )
    monkeypatch.setattr(
        replay, "SOURCE_BYTES", source.stat().st_size, raising=False
    )
    sidecar = tmp_path / "sidecar.jsonl"
    sidecar.write_text(
        "".join(
            json.dumps(
                {
                    "trajectory_id": replay.trajectory_identity(row["task"], row),
                    "source_instance_id": row["task"],
                    "source_terminal_sha256": hashlib.sha256(
                        replay._canonical_json(row)
                    ).hexdigest(),
                    "row": row,
                },
                separators=(",", ":"),
            )
            + "\n"
            for row in rows
        ),
        encoding="utf-8",
    )
    policy = dataclasses.replace(
        _docker_policy(),
        images=(
            ("python", "example.invalid/python@sha256:" + "1" * 64),
            ("rust", "example.invalid/rust@sha256:" + "2" * 64),
            ("cpp", "example.invalid/cpp@sha256:" + "3" * 64),
        ),
    )
    policy_path = tmp_path / "policy.json"
    policy_path.write_text(json.dumps(replay.docker_policy_artifact(policy)))
    admission_path = tmp_path / "admission.json"
    admission_path.write_text(
        json.dumps(
            replay.admission_artifact(
                policy,
                [
                    _admission_for(policy, language)
                    for language in ("python", "rust", "cpp")
                ],
            )
        )
    )
    exclusion = tmp_path / "evaluation-exclusions.json"
    exclusion.write_text(json.dumps(exclusion_payload), encoding="utf-8")
    smoke_ids = [replay.trajectory_identity(row["task"], row) for row in rows[:5]]
    smoke = tmp_path / "smoke-input.json"
    smoke.write_text(json.dumps(smoke_ids), encoding="utf-8")
    out = tmp_path / out_name
    argv = [
        "--source",
        str(source),
        "--sidecar",
        str(sidecar),
        "--seed-repo",
        str(repository),
        "--policy",
        str(policy_path),
        "--admission",
        str(admission_path),
        "--smoke-manifest",
        str(smoke),
        "--exclusion",
        str(exclusion),
        "--ledger",
        str(tmp_path / "replay.jsonl"),
        "--logs",
        str(tmp_path / "logs"),
        "--out",
        str(out),
        "--inventory-only",
    ]
    return argv, out, exclusion


def test_inventory_cli_happy_path_publishes_bound_manifests(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    argv, out, _exclusion = _inventory_cli_fixture(tmp_path, monkeypatch)

    assert replay.main(argv) == 0

    inventory = json.loads((out / "eligibility.json").read_text())
    smoke = json.loads((out / "smoke.json").read_text())
    assert inventory["total"] == inventory["eligible_ceiling"] == 5
    assert inventory["exclusions"] == {
        "unsupported_operations": 0,
        "contamination": 0,
        "invalid_git_seed": 0,
        "invalid_reference_patch": 0,
        "missing_language_digest": 0,
        "functional_admission_failed": 0,
    }
    assert smoke["eligibility_sha256"] == inventory["inventory_sha256"]
    assert [row["language"] for row in smoke["trajectories"]] == [
        "python",
        "python",
        "rust",
        "rust",
        "cpp",
    ]
    assert stat.S_IMODE(out.stat().st_mode) == 0o700
    assert stat.S_IMODE((out / "eligibility.json").stat().st_mode) == 0o600
    assert stat.S_IMODE((out / "smoke.json").stat().st_mode) == 0o600


@pytest.mark.parametrize(
    ("kind", "expected_gate"),
    [
        ("selection", "unsupported_operations"),
        ("conversion", "unsupported_operations"),
    ],
)
def test_inventory_cli_conversion_failures_are_explicit_gate_failures(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    kind: str,
    expected_gate: str,
) -> None:
    argv, out, _exclusion = _inventory_cli_fixture(
        tmp_path,
        monkeypatch,
        include_extra=True,
        invalid_task=("py-extra", kind),
    )

    assert replay.main(argv) == 0

    inventory = json.loads((out / "eligibility.json").read_text())
    assert inventory["total"] == 6
    assert inventory["eligible_ceiling"] == 5
    assert inventory["exclusions"][expected_gate] == 1


def test_inventory_cli_missing_reference_patch_fails_reference_gate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    argv, out, _exclusion = _inventory_cli_fixture(
        tmp_path,
        monkeypatch,
        include_extra=True,
        missing_patch_task="py-extra",
    )

    assert replay.main(argv) == 0

    inventory = json.loads((out / "eligibility.json").read_text())
    assert inventory["eligible_ceiling"] == 5
    assert inventory["exclusions"]["invalid_reference_patch"] == 1


def test_inventory_cli_rejects_source_language_mismatched_to_pinned_seed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    argv, out, _exclusion = _inventory_cli_fixture(
        tmp_path,
        monkeypatch,
        include_extra=True,
        source_language_override=("py-extra", "rust"),
    )

    assert replay.main(argv) == 0

    inventory = json.loads((out / "eligibility.json").read_text())
    assert inventory["eligible_ceiling"] == 5
    assert inventory["exclusions"]["invalid_git_seed"] == 1
    assert all(row["task"] != "py-extra" for row in inventory["eligible"])


def test_inventory_cli_uses_converted_messages_content_hash_for_exclusion(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    row = _inventory_cli_row("py-extra", "python")
    selected = select_terminal_row(row)
    converted = convert_trajectory(
        selected,
        frozenset({"test.txt"}),
        trusted_verifier_commands=frozenset({"true"}),
    )
    content_sha = hashlib.sha256(
        replay._canonical_json(converted["messages"])
    ).hexdigest()
    argv, out, _exclusion = _inventory_cli_fixture(
        tmp_path,
        monkeypatch,
        include_extra=True,
        exclusion_payload=[{"content_sha256": content_sha}],
    )

    assert replay.main(argv) == 0

    inventory = json.loads((out / "eligibility.json").read_text())
    assert inventory["eligible_ceiling"] == 5
    assert inventory["exclusions"]["contamination"] == 1


def test_exclusion_bytes_are_bound_even_when_gate_outcomes_do_not_change(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    argv, out_a, exclusion = _inventory_cli_fixture(
        tmp_path,
        monkeypatch,
        exclusion_payload=[{"irrelevant": "a"}],
        out_name="out-a",
    )
    assert replay.main(argv) == 0
    first = json.loads((out_a / "eligibility.json").read_text())

    exclusion.write_text(json.dumps([{"irrelevant": "b"}]), encoding="utf-8")
    out_b = tmp_path / "out-b"
    argv[argv.index(str(out_a))] = str(out_b)
    assert replay.main(argv) == 0
    second = json.loads((out_b / "eligibility.json").read_text())

    assert first["eligible"] == second["eligible"]
    assert first["exclusions"] == second["exclusions"]
    assert first["bindings"]["exclusion_artifacts"] != second["bindings"][
        "exclusion_artifacts"
    ]
    assert first["inventory_sha256"] != second["inventory_sha256"]


def test_inventory_cli_rejects_missing_exclusion_artifact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    argv, _out, exclusion = _inventory_cli_fixture(tmp_path, monkeypatch)
    exclusion.unlink()

    with pytest.raises(ReplayContractError, match="exclusion.*missing"):
        replay.main(argv)


@pytest.mark.parametrize(
    "argument",
    [
        "--source",
        "--sidecar",
        "--exclusion",
        "--policy",
        "--admission",
        "--smoke-manifest",
    ],
)
def test_inventory_cli_rejects_input_replacement_before_publish(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    argument: str,
) -> None:
    argv, out, _exclusion = _inventory_cli_fixture(tmp_path, monkeypatch)
    target = Path(argv[argv.index(argument) + 1])
    original_build = replay.build_smoke_manifest

    def replace_after_all_inputs_are_parsed(
        inventory: dict[str, object], trajectory_ids: list[str]
    ) -> dict[str, object]:
        target.rename(target.with_name(target.name + ".original"))
        target.write_text("{}\n", encoding="utf-8")
        return original_build(inventory, trajectory_ids)

    monkeypatch.setattr(
        replay, "build_smoke_manifest", replace_after_all_inputs_are_parsed
    )
    with pytest.raises(ReplayContractError, match="input.*identity"):
        replay.main(argv)
    assert not out.exists()


def test_inventory_cli_requires_exact_pinned_source_size(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    argv, out, _exclusion = _inventory_cli_fixture(tmp_path, monkeypatch)
    monkeypatch.setattr(replay, "SOURCE_BYTES", replay.SOURCE_BYTES + 1)

    with pytest.raises(ReplayContractError, match="source.*size"):
        replay.main(argv)
    assert not out.exists()


@pytest.mark.parametrize("mode", [0o664, 0o755, 0o4644])
def test_inventory_cli_rejects_unsafe_input_mode(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mode: int
) -> None:
    argv, out, _exclusion = _inventory_cli_fixture(tmp_path, monkeypatch)
    policy = Path(argv[argv.index("--policy") + 1])
    policy.chmod(mode)

    with pytest.raises(ReplayContractError, match="input.*mode"):
        replay.main(argv)
    assert not out.exists()


def test_inventory_cli_opens_each_mutable_input_once_with_nofollow(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    argv, _out, _exclusion = _inventory_cli_fixture(tmp_path, monkeypatch)
    arguments = (
        "--source",
        "--sidecar",
        "--exclusion",
        "--policy",
        "--admission",
        "--smoke-manifest",
    )
    names = {Path(argv[argv.index(argument) + 1]).name for argument in arguments}
    opens: Counter[str] = Counter()
    original_open = os.open

    def recording_open(path: object, flags: int, *args: object, **kwargs: object) -> int:
        name = Path(os.fspath(path)).name
        if name in names and not flags & os.O_DIRECTORY:
            assert flags & os.O_NOFOLLOW
            opens[name] += 1
        return original_open(path, flags, *args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(replay.os, "open", recording_open)
    assert replay.main(argv) == 0
    assert opens == Counter({name: 1 for name in names})


def test_inventory_cli_mid_publication_input_replacement_leaves_no_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    argv, out, _exclusion = _inventory_cli_fixture(tmp_path, monkeypatch)
    source = Path(argv[argv.index("--source") + 1])
    original_write = replay._atomic_write_0600_at
    replaced = False

    def replace_after_first_manifest(
        directory_fd: int,
        name: str,
        data: bytes,
        **kwargs: object,
    ) -> None:
        nonlocal replaced
        original_write(directory_fd, name, data, **kwargs)
        if name == "eligibility.json" and not replaced:
            replaced = True
            source.rename(source.with_name(source.name + ".original"))
            source.write_text("{}\n", encoding="utf-8")

    monkeypatch.setattr(replay, "_atomic_write_0600_at", replace_after_first_manifest)
    with pytest.raises(ReplayContractError, match="source input identity"):
        replay.main(argv)
    assert not out.exists()


def test_inventory_cli_refuses_existing_output_without_modifying_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    argv, out, _exclusion = _inventory_cli_fixture(tmp_path, monkeypatch)
    out.mkdir(mode=0o700)
    marker = out / "owned-by-caller"
    marker.write_text("unchanged", encoding="utf-8")

    with pytest.raises(ReplayContractError, match="output.*already exists"):
        replay.main(argv)
    assert marker.read_text(encoding="utf-8") == "unchanged"
    assert sorted(path.name for path in out.iterdir()) == ["owned-by-caller"]


def test_inventory_cli_rename_race_preserves_victim_and_leaves_no_owned_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    argv, out, _exclusion = _inventory_cli_fixture(tmp_path, monkeypatch)
    victim = tmp_path / "victim"
    victim.mkdir(mode=0o700)
    marker = victim / "marker"
    marker.write_text("unchanged", encoding="utf-8")
    original_rename = replay._rename_directory_noreplace

    def create_target_then_rename(
        parent_fd: int, staging_name: str, output_name: str
    ) -> None:
        os.symlink(victim, output_name, dir_fd=parent_fd)
        original_rename(parent_fd, staging_name, output_name)

    monkeypatch.setattr(replay, "_rename_directory_noreplace", create_target_then_rename)
    with pytest.raises(ReplayContractError, match="output.*already exists"):
        replay.main(argv)
    assert marker.read_text(encoding="utf-8") == "unchanged"
    assert out.is_symlink()
    staging = [path for path in tmp_path.iterdir() if path.name.startswith(".out.")]
    assert len(staging) == 1
    assert stat.S_IMODE(staging[0].stat().st_mode) == 0o700
    assert sorted(path.name for path in staging[0].iterdir()) == [
        "eligibility.json",
        "smoke.json",
    ]


def test_inventory_cli_refuses_symlink_output_without_touching_victim(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    argv, out, _exclusion = _inventory_cli_fixture(tmp_path, monkeypatch)
    victim = tmp_path / "victim"
    victim.mkdir(mode=0o700)
    marker = victim / "marker"
    marker.write_text("unchanged", encoding="utf-8")
    out.symlink_to(victim, target_is_directory=True)

    with pytest.raises(ReplayContractError, match="output.*already exists"):
        replay.main(argv)
    assert marker.read_text(encoding="utf-8") == "unchanged"
    assert out.is_symlink()


def test_inventory_cli_never_rolls_back_committed_directory_after_path_swap(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    argv, out, _exclusion = _inventory_cli_fixture(tmp_path, monkeypatch)
    victim = tmp_path / "victim"
    victim.mkdir(mode=0o700)
    marker = victim / "marker"
    marker.write_text("unchanged", encoding="utf-8")
    moved = tmp_path / "committed-owned"
    original_rename = replay._rename_directory_noreplace

    def swap_output_after_commit(
        parent_fd: int, staging_name: str, output_name: str
    ) -> None:
        original_rename(parent_fd, staging_name, output_name)
        os.rename(output_name, moved.name, src_dir_fd=parent_fd, dst_dir_fd=parent_fd)
        os.symlink(victim, output_name, dir_fd=parent_fd)

    monkeypatch.setattr(replay, "_rename_directory_noreplace", swap_output_after_commit)
    assert replay.main(argv) == 0
    assert marker.read_text(encoding="utf-8") == "unchanged"
    assert out.is_symlink()
    assert sorted(path.name for path in moved.iterdir()) == [
        "eligibility.json",
        "smoke.json",
    ]
    assert all(stat.S_IMODE(path.stat().st_mode) == 0o600 for path in moved.iterdir())


class FakeRestrictedExecutor:
    def __init__(self, outcomes: dict[str, list[int]]) -> None:
        self.policy = _docker_policy()
        self.outcomes = {key: list(values) for key, values in outcomes.items()}
        self.calls: list[dict[str, object]] = []
        self.counter = 0

    def execute(self, **kwargs: object):
        self.calls.append(dict(kwargs))
        identity = str(kwargs["control_identity"])
        rc = self.outcomes[identity].pop(0)
        self.counter += 1
        image = self.policy.image_for(str(kwargs["language"]))
        run_contract = replay._sha256(
            replay._canonical_json(
                {
                    "policy": replay._policy_payload(self.policy, image),
                    "language": kwargs["language"],
                    "verifier_sha256": replay._sha256(
                        str(kwargs["verifier_text"]).encode()
                    ),
                    "effective_timeout": kwargs["effective_timeout"],
                    "control_identity": identity,
                    "pre_tree": kwargs["pre_candidate_tree_sha256"],
                    "pre_diff": kwargs["pre_candidate_diff_sha256"],
                    "protected": kwargs["protected_before"],
                }
            )
        )
        return _run_evidence(
            identity=identity,
            tree=str(kwargs["pre_candidate_tree_sha256"]),
            diff=str(kwargs["pre_candidate_diff_sha256"]),
            protected=tuple(kwargs["protected_before"]),
            rc=rc,
            output_hash=f"{self.counter:064x}",
            policy_version=self.policy.policy_version,
            image_digest=image,
            run_contract_sha256=run_contract,
        )


def test_verify_candidate_reconstructs_two_fresh_states_and_accepts_raw_output_variance(
    tmp_path: Path,
) -> None:
    contract = _materialized(tmp_path)
    plan = _plan_for(
        contract,
        _call("Write", {"file_path": "/testbed/src.py", "content": "new\n"}),
    )
    executor = FakeRestrictedExecutor({"candidate": [0, 0]})

    evidence = replay.verify_candidate(
        contract,
        plan,
        trajectory_id="1" * 64,
        source_terminal_sha256="2" * 64,
        executor=executor,
        workspace=tmp_path / "verify",
    )

    assert evidence.schema_version == 2
    assert evidence.resolved is True
    assert evidence.trainable is True
    assert len(evidence.runs) == 2
    assert evidence.runs[0].raw_output_sha256 != evidence.runs[1].raw_output_sha256
    roots = [Path(call["candidate_root"]) for call in executor.calls]
    assert roots[0] != roots[1]
    assert evidence.runs[0].pre_candidate_tree_sha256 == evidence.runs[1].pre_candidate_tree_sha256
    assert evidence.runs[0].pre_candidate_diff_sha256 == evidence.runs[1].pre_candidate_diff_sha256


def test_verify_candidate_rejects_protected_drift_or_single_run_failure(
    tmp_path: Path,
) -> None:
    contract = _materialized(tmp_path)
    plan = _plan_for(
        contract,
        _call("Write", {"file_path": "/testbed/src.py", "content": "new\n"}),
    )
    executor = FakeRestrictedExecutor({"candidate": [0, 1]})

    evidence = replay.verify_candidate(
        contract,
        plan,
        trajectory_id="1" * 64,
        source_terminal_sha256="2" * 64,
        executor=executor,
        workspace=tmp_path / "verify",
    )

    assert evidence.resolved is False
    assert evidence.trainable is False
    assert evidence.failure_class == "candidate_not_repeatable"


def test_replay_run_contract_binds_dataset_git_terminal_operation_and_policy(
    tmp_path: Path,
) -> None:
    contract = _materialized(tmp_path)
    plan = _plan_for(
        contract,
        _call("Write", {"file_path": "/testbed/src.py", "content": "new\n"}),
    )
    first = replay.verify_candidate(
        contract,
        plan,
        trajectory_id="1" * 64,
        source_terminal_sha256="2" * 64,
        executor=FakeRestrictedExecutor({"candidate": [0, 0]}),
        workspace=tmp_path / "first",
    )
    second = replay.verify_candidate(
        contract,
        plan,
        trajectory_id="1" * 64,
        source_terminal_sha256="3" * 64,
        executor=FakeRestrictedExecutor({"candidate": [0, 0]}),
        workspace=tmp_path / "second",
    )

    assert first.run_contract_sha256 != second.run_contract_sha256


def test_verify_candidate_rejects_executor_evidence_from_a_control_namespace(
    tmp_path: Path,
) -> None:
    contract = _materialized(tmp_path)
    plan = _plan_for(
        contract,
        _call("Write", {"file_path": "/testbed/src.py", "content": "new\n"}),
    )

    class WrongIdentityExecutor(FakeRestrictedExecutor):
        def execute(self, **kwargs: object):
            run = super().execute(**kwargs)
            return dataclasses.replace(
                run, control_identity="reference", trainable=False
            )

    evidence = replay.verify_candidate(
        contract,
        plan,
        trajectory_id="1" * 64,
        source_terminal_sha256="2" * 64,
        executor=WrongIdentityExecutor({"candidate": [0, 0]}),
        workspace=tmp_path / "verify",
    )

    assert evidence.resolved is False
    assert evidence.trainable is False
    assert evidence.failure_class == "candidate_not_repeatable"


@pytest.mark.parametrize(
    "updates",
    [
        {"returncode": 9},
        {"wrapper_returncode": 8},
        {"termination": "wall_timeout"},
        {"failure_class": "verifier_timeout"},
        {"post_candidate_tree_sha256": None},
        {"resource_peaks": ()},
        {
            "resource_peaks": (
                ("memory_bytes", 0),
                ("pids", 3),
                ("cpu_usec", 4000),
            )
        },
        {"resource_peaks": (("memory_bytes", 1, 2),)},
        {"cleanup_state": "cleanup_failed"},
        {"protected_after": ()},
        {"trainable": False},
        {"resolved": False},
        {"pre_candidate_tree_sha256": "9" * 64},
        {"pre_candidate_diff_sha256": "8" * 64},
        {"policy_version": "other-policy"},
        {"image_digest": "example.invalid/other@sha256:" + "2" * 64},
        {"runtime_version": "other-runtime"},
        {"run_contract_sha256": "7" * 64},
    ],
)
def test_verify_candidate_rejects_every_contradictory_positive_run_field(
    tmp_path: Path, updates: dict[str, object]
) -> None:
    contract = _materialized(tmp_path)
    plan = _plan_for(
        contract,
        _call("Write", {"file_path": "/testbed/src.py", "content": "new\n"}),
    )

    class ContradictoryExecutor(FakeRestrictedExecutor):
        def execute(self, **kwargs: object):
            run = super().execute(**kwargs)
            return dataclasses.replace(run, **updates)

    evidence = replay.verify_candidate(
        contract,
        plan,
        trajectory_id="1" * 64,
        source_terminal_sha256="2" * 64,
        executor=ContradictoryExecutor({"candidate": [0, 0]}),
        workspace=tmp_path / "verify",
    )

    assert evidence.resolved is False
    assert evidence.trainable is False
    assert evidence.failure_class == "candidate_not_repeatable"


def test_verify_candidate_requires_one_exact_runtime_identity_across_runs(
    tmp_path: Path,
) -> None:
    contract = _materialized(tmp_path)
    plan = _plan_for(
        contract,
        _call("Write", {"file_path": "/testbed/src.py", "content": "new\n"}),
    )

    class RuntimeDriftExecutor(FakeRestrictedExecutor):
        def execute(self, **kwargs: object):
            run = super().execute(**kwargs)
            if self.counter == 2:
                return dataclasses.replace(run, runtime_version="other-runtime")
            return run

    evidence = replay.verify_candidate(
        contract,
        plan,
        trajectory_id="1" * 64,
        source_terminal_sha256="2" * 64,
        executor=RuntimeDriftExecutor({"candidate": [0, 0]}),
        workspace=tmp_path / "verify",
    )

    assert evidence.resolved is False
    assert evidence.trainable is False


def test_control_set_requires_failing_baseline_passing_reference_and_two_candidates(
    tmp_path: Path,
) -> None:
    contract = _materialized(tmp_path)
    plan = _plan_for(
        contract,
        _call("Write", {"file_path": "/testbed/src.py", "content": "new\n"}),
    )
    executor = FakeRestrictedExecutor(
        {"baseline": [1], "reference": [0], "candidate": [0, 0]}
    )

    controls = replay.run_control_set(
        contract,
        plan,
        trajectory_id="1" * 64,
        source_terminal_sha256="2" * 64,
        executor=executor,
        workspace=tmp_path / "controls",
        reference_builder=lambda seed, destination: _copy_reference_fixture(
            seed, destination
        ),
    )

    assert controls.schema_version == 1
    assert controls.admitted is True
    assert controls.baseline.resolved is False
    assert controls.reference.resolved is True
    assert controls.candidate.resolved is True
    assert controls.baseline.trainable is False
    assert controls.reference.trainable is False
    assert controls.candidate.trainable is True
    assert [call["control_identity"] for call in executor.calls] == [
        "baseline",
        "reference",
        "candidate",
        "candidate",
    ]


def _copy_reference_fixture(contract: object, destination: Path) -> object:
    records = replay._inventory_from_root(contract.files_root)
    replay._write_records(destination, records)
    (destination / "src.py").write_text("new\n")
    return replay._candidate_state_for_control(contract, destination, "reference")


def test_control_set_is_tainted_when_control_expectations_do_not_hold(
    tmp_path: Path,
) -> None:
    contract = _materialized(tmp_path)
    plan = _plan_for(
        contract,
        _call("Write", {"file_path": "/testbed/src.py", "content": "new\n"}),
    )
    executor = FakeRestrictedExecutor(
        {"baseline": [0], "reference": [0], "candidate": [0, 0]}
    )

    controls = replay.run_control_set(
        contract,
        plan,
        trajectory_id="1" * 64,
        source_terminal_sha256="2" * 64,
        executor=executor,
        workspace=tmp_path / "controls",
        reference_builder=lambda seed, destination: _copy_reference_fixture(
            seed, destination
        ),
    )

    assert controls.admitted is False
    assert controls.failure_class == "baseline_unexpectedly_passed"
    assert controls.candidate.trainable is False
    assert [call["control_identity"] for call in executor.calls] == [
        "baseline",
        "reference",
        "candidate",
        "candidate",
    ]


def test_control_set_rejects_timeout_as_a_valid_failing_baseline(tmp_path: Path) -> None:
    contract = _materialized(tmp_path)
    plan = _plan_for(
        contract,
        _call("Write", {"file_path": "/testbed/src.py", "content": "new\n"}),
    )

    class DirtyBaselineExecutor(FakeRestrictedExecutor):
        def execute(self, **kwargs: object):
            run = super().execute(**kwargs)
            if kwargs["control_identity"] == "baseline":
                return dataclasses.replace(
                    run,
                    termination="wall_timeout",
                    wrapper_returncode=None,
                    protected_after=(),
                    failure_class="verifier_timeout",
                )
            return run

    controls = replay.run_control_set(
        contract,
        plan,
        trajectory_id="1" * 64,
        source_terminal_sha256="2" * 64,
        executor=DirtyBaselineExecutor(
            {"baseline": [1], "reference": [0], "candidate": [0, 0]}
        ),
        workspace=tmp_path / "controls",
        reference_builder=lambda seed, destination: _copy_reference_fixture(
            seed, destination
        ),
    )

    assert controls.admitted is False
    assert controls.failure_class == "baseline_invalid_failure"
    assert controls.candidate.trainable is False


@pytest.mark.parametrize(
    "updates",
    [
        {"resource_peaks": ()},
        {"cleanup_state": "cleanup_failed"},
        {"pre_candidate_tree_sha256": "9" * 64},
        {"pre_candidate_diff_sha256": "8" * 64},
        {"policy_version": "other-policy"},
        {"image_digest": "example.invalid/other@sha256:" + "2" * 64},
        {"run_contract_sha256": "7" * 64},
    ],
)
def test_control_set_requires_strict_clean_negative_baseline_evidence(
    tmp_path: Path, updates: dict[str, object]
) -> None:
    contract = _materialized(tmp_path)
    plan = _plan_for(
        contract,
        _call("Write", {"file_path": "/testbed/src.py", "content": "new\n"}),
    )

    class DirtyBaselineExecutor(FakeRestrictedExecutor):
        def execute(self, **kwargs: object):
            run = super().execute(**kwargs)
            if kwargs["control_identity"] == "baseline":
                return dataclasses.replace(run, **updates)
            return run

    controls = replay.run_control_set(
        contract,
        plan,
        trajectory_id="1" * 64,
        source_terminal_sha256="2" * 64,
        executor=DirtyBaselineExecutor(
            {"baseline": [1], "reference": [0], "candidate": [0, 0]}
        ),
        workspace=tmp_path / "controls",
        reference_builder=lambda seed, destination: _copy_reference_fixture(
            seed, destination
        ),
    )

    assert controls.admitted is False
    assert controls.failure_class == "baseline_invalid_failure"
    assert controls.candidate.trainable is False


@pytest.mark.parametrize(
    "updates",
    [
        {"control_identity": "candidate"},
        {"trainable": True},
        {"cleanup_state": "cleanup_failed"},
        {"termination": "wall_timeout"},
        {"returncode": 9},
        {"wrapper_returncode": None},
        {"failure_class": "verifier_timeout"},
        {"protected_after": ()},
        {"post_candidate_tree_sha256": None},
        {"resource_peaks": ()},
        {"pre_candidate_tree_sha256": "9" * 64},
        {"pre_candidate_diff_sha256": "8" * 64},
        {"policy_version": "other-policy"},
        {"image_digest": "example.invalid/other@sha256:" + "2" * 64},
        {"runtime_version": "other-runtime"},
        {"run_contract_sha256": "7" * 64},
        {"resolved": False},
    ],
)
def test_control_set_rejects_every_dirty_reference_field(
    tmp_path: Path, updates: dict[str, object]
) -> None:
    contract = _materialized(tmp_path)
    plan = _plan_for(
        contract,
        _call("Write", {"file_path": "/testbed/src.py", "content": "new\n"}),
    )

    class DirtyReferenceExecutor(FakeRestrictedExecutor):
        def execute(self, **kwargs: object):
            run = super().execute(**kwargs)
            if kwargs["control_identity"] == "reference":
                return dataclasses.replace(run, **updates)
            return run

    controls = replay.run_control_set(
        contract,
        plan,
        trajectory_id="1" * 64,
        source_terminal_sha256="2" * 64,
        executor=DirtyReferenceExecutor(
            {"baseline": [1], "reference": [0], "candidate": [0, 0]}
        ),
        workspace=tmp_path / "controls",
        reference_builder=lambda seed, destination: _copy_reference_fixture(
            seed, destination
        ),
    )

    assert controls.admitted is False
    assert controls.failure_class == "reference_invalid_evidence"
    assert controls.candidate.trainable is False


@pytest.mark.parametrize(
    "updates",
    [
        {"control_identity": "candidate", "trainable": True},
        {"cleanup_state": "cleanup_failed"},
        {"termination": "wall_timeout"},
        {"returncode": 9},
        {"wrapper_returncode": None},
        {"failure_class": "verifier_timeout"},
        {"post_candidate_tree_sha256": None},
        {"resource_peaks": ()},
        {"policy_version": "other-policy"},
        {"image_digest": "example.invalid/other@sha256:" + "2" * 64},
        {"run_contract_sha256": "7" * 64},
        {"resolved": False},
    ],
)
def test_functional_admission_rejects_every_dirty_positive_field(
    tmp_path: Path, updates: dict[str, object]
) -> None:
    policy = _docker_policy()

    class DirtyAdmissionExecutor(FakeRestrictedExecutor):
        def execute(self, **kwargs: object):
            run = super().execute(**kwargs)
            return dataclasses.replace(run, **updates)

    evidence = replay.functional_admission_probe(
        policy,
        "python",
        executor=DirtyAdmissionExecutor({"admission": [0]}),
        workspace=tmp_path / "admission",
    )

    assert evidence.admitted is False
    assert evidence.failure_class == "admission_invalid_evidence"
