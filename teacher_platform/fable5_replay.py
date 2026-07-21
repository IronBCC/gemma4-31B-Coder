"""Immutable, declarative reconstruction for pinned Fable trajectories.

This module deliberately stops before executing a verifier.  It admits a seed
from one exact Git object, parses source operations through the importer's
shared typed parser, and reconstructs only declarative Write/Edit mutations.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import os
import re
import stat
import subprocess
import tarfile
import tempfile
import unicodedata
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any, Final, Protocol

if __package__:  # Support both ``python -m teacher_platform...`` and local tests.
    from .fable5_import import (
        DATASET_REVISION,
        FableEditOp,
        FableWriteOp,
        ReadOnlyBashOp,
        VerifierEvidenceOp,
        parse_fable_tool_call,
    )
else:  # pragma: no cover - the branch is exercised by local tests.
    from fable5_import import (  # type: ignore[no-redef]
        DATASET_REVISION,
        FableEditOp,
        FableWriteOp,
        ReadOnlyBashOp,
        VerifierEvidenceOp,
        parse_fable_tool_call,
    )


MOONSHINER_REVISION: Final = "436316e8f86eb136d5ce3ec95a1a6f48c1d7f940"
DEFAULT_VERIFY_TIMEOUT: Final = 300
MAX_VERIFY_TIMEOUT: Final = 1_800
_TASK_RE: Final = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*\Z")
_OID_RE: Final = re.compile(r"[0-9a-f]{40}(?:[0-9a-f]{24})?\Z")


class ReplayContractError(ValueError):
    """A pinned source, operation, or candidate violated the replay contract."""


class GitRunner(Protocol):
    def __call__(self, argv: tuple[str, ...]) -> bytes: ...


class PatchExecutor(Protocol):
    def __call__(
        self, argv: tuple[str, ...], cwd: Path
    ) -> subprocess.CompletedProcess[bytes]: ...


def _run_git(argv: tuple[str, ...]) -> bytes:
    try:
        return subprocess.run(argv, check=True, capture_output=True).stdout
    except (OSError, subprocess.CalledProcessError) as exc:
        raise ReplayContractError(f"pinned Git command failed: {argv!r}") from exc


@dataclass(frozen=True)
class GitSeedSource:
    """A read-only admission boundary for the one reviewed Moonshiner commit."""

    repo: Path
    commit: str
    runner: GitRunner = field(default=_run_git, repr=False, compare=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "repo", Path(self.repo))
        if self.commit != MOONSHINER_REVISION:
            raise ValueError(
                f"expected pinned Moonshiner commit {MOONSHINER_REVISION}, "
                f"got {self.commit!r}"
            )

    def materialize_seed(self, task: str, destination: Path) -> SeedContract:
        return materialize_seed(self, task, destination)


@dataclass(frozen=True)
class SeedContract:
    task: str
    language: str
    verify_cmd: str
    verifier_sha256: str
    source_verify_timeout: int | None
    effective_verify_timeout: int
    protected_paths: tuple[str, ...]
    protected_sha256: tuple[tuple[str, str], ...]
    source_commit_sha: str
    source_tree_sha: str
    task_tree_sha: str
    inventory_sha256: str
    fixture_sha256: str
    seed_root: Path
    files_root: Path
    reference_patch_path: Path
    reference_patch_sha256: str | None


@dataclass(frozen=True)
class MutationPlan:
    operations: tuple[FableWriteOp | FableEditOp, ...]
    operation_sha256: str
    verifier_sha256: str
    verifier_evidence_count: int


@dataclass(frozen=True)
class CandidateState:
    root: Path
    tree_sha256: str
    diff_sha256: str
    changed_paths: tuple[str, ...]
    operation_sha256: str
    verifier_sha256: str
    protected_sha256: tuple[tuple[str, str], ...]


@dataclass(frozen=True)
class ReferencePatchContract:
    eligible: bool
    exclusion_reason: str | None
    patch_sha256: str | None
    targets: tuple[str, ...]


@dataclass(frozen=True)
class _InventoryFile:
    path: str
    mode: int
    data: bytes


def _nfc_string(value: str) -> str:
    return unicodedata.normalize("NFC", value)


def _canonical_value(value: Any) -> Any:
    if isinstance(value, str):
        return _nfc_string(value)
    if isinstance(value, Mapping):
        normalized: dict[str, Any] = {}
        for key, item in value.items():
            if type(key) is not str:
                raise ReplayContractError("canonical JSON keys must be strings")
            canonical_key = _nfc_string(key)
            if canonical_key in normalized:
                raise ReplayContractError("canonical JSON key collision after NFC")
            normalized[canonical_key] = _canonical_value(item)
        return normalized
    if isinstance(value, (list, tuple)):
        return [_canonical_value(item) for item in value]
    return value


def _canonical_json(value: Any) -> bytes:
    try:
        return json.dumps(
            _canonical_value(value),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except ReplayContractError:
        raise
    except (TypeError, ValueError, UnicodeEncodeError) as exc:
        raise ReplayContractError("value is not canonical UTF-8 JSON") from exc


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def trajectory_identity(
    task: str,
    canonical_terminal: Mapping[str, Any],
    *,
    dataset_revision: str = DATASET_REVISION,
) -> str:
    """Return the reviewed trajectory primary key."""

    if not _TASK_RE.fullmatch(task):
        raise ReplayContractError("task ID is not confined")
    terminal = _canonical_json(canonical_terminal)
    return _sha256(
        dataset_revision.encode("utf-8")
        + b"\0"
        + task.encode("utf-8")
        + b"\0"
        + terminal
    )


def _relative_path(value: object, *, field_name: str = "path") -> str:
    if type(value) is not str or not value:
        raise ReplayContractError(f"{field_name} must be a nonempty string")
    if value != unicodedata.normalize("NFC", value):
        raise ReplayContractError(f"{field_name} is not NFC")
    try:
        value.encode("utf-8")
    except UnicodeEncodeError as exc:
        raise ReplayContractError(f"{field_name} is not valid UTF-8") from exc
    if "\x00" in value or "\n" in value or "\r" in value:
        raise ReplayContractError(f"{field_name} contains a control character")
    if value.startswith("/"):
        raise ReplayContractError(f"{field_name} is absolute")
    if any(part in {"", "."} for part in value.split("/")):
        raise ReplayContractError(f"{field_name} is not canonical")
    path = PurePosixPath(value)
    if path.is_absolute():
        raise ReplayContractError(f"{field_name} is absolute")
    if any(part == ".." for part in path.parts):
        raise ReplayContractError(f"{field_name} contains parent traversal")
    if any(part in {"", "."} for part in path.parts):
        raise ReplayContractError(f"{field_name} is not canonical")
    return path.as_posix()


def _parse_ls_tree(raw: bytes, task: str) -> dict[str, tuple[int, str]]:
    prefix = f"tasks/seeds/{task}/"
    result: dict[str, tuple[int, str]] = {}
    for record in raw.split(b"\0"):
        if not record:
            continue
        try:
            header, path_bytes = record.split(b"\t", 1)
            mode_text, kind, object_id = header.decode("ascii").split(" ", 2)
            path = path_bytes.decode("utf-8")
        except (ValueError, UnicodeDecodeError) as exc:
            raise ReplayContractError("malformed pinned Git tree entry") from exc
        if not path.startswith(prefix):
            raise ReplayContractError("pinned Git tree entry escapes task prefix")
        relative = _relative_path(path.removeprefix(prefix), field_name="Git path")
        if relative in result:
            raise ReplayContractError("duplicate pinned Git tree path")
        if mode_text == "120000":
            raise ReplayContractError("pinned Git tree contains a symlink")
        if mode_text == "160000" or kind == "commit":
            raise ReplayContractError("pinned Git tree contains a submodule")
        if mode_text not in {"100644", "100755"} or kind != "blob":
            raise ReplayContractError("pinned Git tree contains a special object")
        if not _OID_RE.fullmatch(object_id):
            raise ReplayContractError("pinned Git tree has an invalid object ID")
        result[relative] = (int(mode_text[-3:], 8), object_id)
    if not result:
        raise ReplayContractError("pinned Git task tree is empty")
    return result


def _validate_archive(
    archive_bytes: bytes,
    task: str,
    git_entries: Mapping[str, tuple[int, str]],
) -> tuple[_InventoryFile, ...]:
    prefix = PurePosixPath("tasks", "seeds", task)
    records: dict[str, _InventoryFile] = {}
    seen: set[str] = set()
    directory_paths: set[str] = set()
    try:
        archive = tarfile.open(fileobj=__import__("io").BytesIO(archive_bytes), mode="r:")
    except tarfile.TarError as exc:
        raise ReplayContractError("Git archive is not a valid tar") from exc
    with archive:
        for member in archive:
            name = member.name
            if name.startswith("/"):
                raise ReplayContractError("archive member path is absolute")
            if name != unicodedata.normalize("NFC", name):
                raise ReplayContractError("archive member name is not NFC")
            try:
                name.encode("utf-8")
            except UnicodeEncodeError as exc:
                raise ReplayContractError("archive member name is undecodable") from exc
            if any(part in {"", "."} for part in name.split("/")):
                raise ReplayContractError("archive member path is not canonical")
            parts = PurePosixPath(name).parts
            if any(part == ".." for part in parts):
                raise ReplayContractError("archive member contains parent traversal")
            if tuple(parts[:3]) != prefix.parts:
                raise ReplayContractError("archive member escapes task prefix")
            relative_parts = parts[3:]
            if not relative_parts:
                relative = "."
            else:
                relative = _relative_path(
                    PurePosixPath(*relative_parts).as_posix(), field_name="archive path"
                )
            if relative in seen:
                raise ReplayContractError("archive contains duplicate paths")
            seen.add(relative)
            if member.issym():
                raise ReplayContractError("archive contains a symlink")
            if member.islnk():
                raise ReplayContractError("archive contains a hardlink")
            if member.isdir():
                directory_paths.add(relative)
                continue
            if not member.isreg():
                raise ReplayContractError("archive contains a special file")
            if relative == ".":
                raise ReplayContractError("archive root cannot be a file")
            handle = archive.extractfile(member)
            if handle is None:
                raise ReplayContractError("archive regular file has no content")
            data = handle.read()
            if len(data) != member.size:
                raise ReplayContractError("archive file was truncated")
            records[relative] = _InventoryFile(
                path=relative, mode=member.mode & 0o777, data=data
            )

    paths = set(records) | {path for path in directory_paths if path != "."}
    for path in paths:
        parts = PurePosixPath(path).parts
        for index in range(1, len(parts)):
            ancestor = PurePosixPath(*parts[:index]).as_posix()
            if ancestor in records:
                raise ReplayContractError("archive contains a file/directory collision")
    if set(records) != set(git_entries):
        raise ReplayContractError("archive file set does not match pinned Git tree")
    for path, record in records.items():
        git_mode, _ = git_entries[path]
        if record.mode != git_mode:
            raise ReplayContractError("archive mode does not match pinned Git tree")
    return tuple(records[path] for path in sorted(records, key=lambda p: p.encode()))


def _write_records(destination: Path, records: Sequence[_InventoryFile]) -> None:
    destination.mkdir(mode=0o700)
    for record in records:
        target = destination.joinpath(*PurePosixPath(record.path).parts)
        target.parent.mkdir(mode=0o755, parents=True, exist_ok=True)
        descriptor = os.open(
            target,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
            record.mode,
        )
        try:
            with os.fdopen(descriptor, "wb") as handle:
                descriptor = -1
                handle.write(record.data)
            os.chmod(target, record.mode, follow_symlinks=False)
        finally:
            if descriptor >= 0:
                os.close(descriptor)


def _inventory_sha(records: Sequence[_InventoryFile]) -> str:
    digest = hashlib.sha256()
    for record in sorted(records, key=lambda item: item.path.encode("utf-8")):
        path = record.path.encode("utf-8")
        digest.update(len(path).to_bytes(8, "big"))
        digest.update(path)
        digest.update(record.mode.to_bytes(4, "big"))
        digest.update(len(record.data).to_bytes(8, "big"))
        digest.update(record.data)
    return digest.hexdigest()


def _read_regular_no_follow(path: Path) -> tuple[bytes, int]:
    try:
        before = os.stat(path, follow_symlinks=False)
    except FileNotFoundError as exc:
        raise ReplayContractError(f"missing regular file: {path.name}") from exc
    if not stat.S_ISREG(before.st_mode):
        raise ReplayContractError(f"fixture path is a special file: {path.name}")
    if before.st_nlink != 1:
        raise ReplayContractError(f"fixture path has a hardlink alias: {path.name}")
    descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    try:
        opened = os.fstat(descriptor)
        if (
            not stat.S_ISREG(opened.st_mode)
            or opened.st_nlink != 1
            or (opened.st_dev, opened.st_ino) != (before.st_dev, before.st_ino)
        ):
            raise ReplayContractError("fixture identity changed before read")
        chunks: list[bytes] = []
        while chunk := os.read(descriptor, 1024 * 1024):
            chunks.append(chunk)
        return b"".join(chunks), stat.S_IMODE(opened.st_mode)
    finally:
        os.close(descriptor)


def _inventory_from_root(root: Path) -> tuple[_InventoryFile, ...]:
    try:
        root_stat = os.stat(root, follow_symlinks=False)
    except FileNotFoundError as exc:
        raise ReplayContractError("fixture root is missing") from exc
    if not stat.S_ISDIR(root_stat.st_mode) or root.is_symlink():
        raise ReplayContractError("fixture root is not a real directory")
    records: list[_InventoryFile] = []
    for current, dirnames, filenames in os.walk(root, followlinks=False):
        current_path = Path(current)
        for dirname in dirnames:
            child = current_path / dirname
            child_stat = os.stat(child, follow_symlinks=False)
            if not stat.S_ISDIR(child_stat.st_mode) or child.is_symlink():
                raise ReplayContractError("fixture tree contains a symlink or special path")
        for filename in filenames:
            child = current_path / filename
            relative = child.relative_to(root).as_posix()
            if relative != unicodedata.normalize("NFC", relative):
                raise ReplayContractError("fixture path is not NFC")
            data, mode = _read_regular_no_follow(child)
            records.append(_InventoryFile(relative, mode, data))
    return tuple(sorted(records, key=lambda item: item.path.encode("utf-8")))


def _git_output(source: GitSeedSource, *args: str) -> bytes:
    argv = ("git", "-C", str(source.repo), *args)
    try:
        return source.runner(argv)
    except ReplayContractError:
        raise
    except Exception as exc:
        raise ReplayContractError(f"pinned Git command failed: {args!r}") from exc


def _exact_oid(value: bytes, name: str) -> str:
    try:
        decoded = value.decode("ascii").removesuffix("\n")
    except UnicodeDecodeError as exc:
        raise ReplayContractError(f"{name} is not an ASCII object ID") from exc
    if not _OID_RE.fullmatch(decoded):
        raise ReplayContractError(f"{name} is not an exact Git object ID")
    return decoded


def materialize_seed(
    source: GitSeedSource, task: str, destination: Path
) -> SeedContract:
    """Materialize exactly one seed task from the pinned Git object."""

    if not _TASK_RE.fullmatch(task):
        raise ReplayContractError("task ID is not a confined seed name")
    destination = Path(destination)
    if destination.exists() or destination.is_symlink():
        raise ReplayContractError("destination already exists")
    _git_output(source, "cat-file", "-e", f"{source.commit}^{{commit}}")
    commit_sha = _exact_oid(
        _git_output(source, "rev-parse", f"{source.commit}^{{commit}}"),
        "commit",
    )
    if commit_sha != MOONSHINER_REVISION:
        raise ReplayContractError("resolved Git commit does not match pinned revision")
    source_tree_sha = _exact_oid(
        _git_output(source, "rev-parse", f"{source.commit}^{{tree}}"), "source tree"
    )
    task_tree_sha = _exact_oid(
        _git_output(
            source, "rev-parse", f"{source.commit}:tasks/seeds/{task}"
        ),
        "task tree",
    )
    git_entries = _parse_ls_tree(
        _git_output(
            source,
            "ls-tree",
            "-rz",
            "-r",
            source.commit,
            "--",
            f"tasks/seeds/{task}",
        ),
        task,
    )
    records = _validate_archive(
        _git_output(
            source,
            "archive",
            "--format=tar",
            source.commit,
            "--",
            f"tasks/seeds/{task}",
        ),
        task,
        git_entries,
    )
    record_map = {record.path: record for record in records}
    task_record = record_map.get("task.json")
    if task_record is None:
        raise ReplayContractError("pinned seed is missing task.json")
    try:
        payload = json.loads(task_record.data.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ReplayContractError("pinned task.json is invalid UTF-8 JSON") from exc
    if type(payload) is not dict:
        raise ReplayContractError("pinned task.json must be an object")
    if payload.get("id") != task:
        raise ReplayContractError("pinned task.json task ID mismatch")
    language = payload.get("lang")
    if language not in {"python", "rust", "cpp"}:
        raise ReplayContractError("pinned task.json has unsupported lang")
    verify_cmd = payload.get("verify_cmd")
    if type(verify_cmd) is not str or not verify_cmd:
        raise ReplayContractError("pinned task.json has no exact verify_cmd")
    try:
        verify_cmd.encode("utf-8")
    except UnicodeEncodeError as exc:
        raise ReplayContractError("verify_cmd is not UTF-8") from exc
    if "\x00" in verify_cmd or "\n" in verify_cmd or "\r" in verify_cmd:
        raise ReplayContractError("verify_cmd must be one exact line")
    source_timeout = payload.get("verify_timeout")
    if "verify_timeout" not in payload:
        source_timeout = None
        effective_timeout = DEFAULT_VERIFY_TIMEOUT
    elif (
        type(source_timeout) is not int
        or source_timeout <= 0
        or source_timeout > MAX_VERIFY_TIMEOUT
    ):
        raise ReplayContractError(
            f"verify_timeout must be a positive non-Boolean integer <= {MAX_VERIFY_TIMEOUT}"
        )
    else:
        effective_timeout = source_timeout
    test_files = payload.get("test_files")
    if (
        type(test_files) is not list
        or not test_files
        or any(type(path) is not str for path in test_files)
    ):
        raise ReplayContractError("test_files must be a nonempty string list")
    protected_paths = tuple(
        _relative_path(path, field_name="test_files path") for path in test_files
    )
    if len(set(protected_paths)) != len(protected_paths):
        raise ReplayContractError("test_files contains a duplicate path")
    fixture_records = tuple(
        _InventoryFile(
            path=record.path.removeprefix("files/"),
            mode=record.mode,
            data=record.data,
        )
        for record in records
        if record.path.startswith("files/")
    )
    if not fixture_records:
        raise ReplayContractError("pinned seed files tree is empty")
    fixture_map = {record.path: record for record in fixture_records}
    missing = [path for path in protected_paths if path not in fixture_map]
    if missing:
        raise ReplayContractError(f"missing protected test file: {missing[0]}")

    _write_records(destination, records)
    files_root = destination / "files"
    materialized = _inventory_from_root(files_root)
    inventory_sha = _inventory_sha(materialized)
    protected_sha = tuple(
        (path, _sha256(fixture_map[path].data)) for path in sorted(protected_paths)
    )
    patch = destination / "reference_fix.patch"
    patch_sha = (
        _sha256(record_map["reference_fix.patch"].data)
        if "reference_fix.patch" in record_map
        else None
    )
    return SeedContract(
        task=task,
        language=language,
        verify_cmd=verify_cmd,
        verifier_sha256=_sha256(verify_cmd.encode("utf-8")),
        source_verify_timeout=source_timeout,
        effective_verify_timeout=effective_timeout,
        protected_paths=tuple(sorted(protected_paths)),
        protected_sha256=protected_sha,
        source_commit_sha=commit_sha,
        source_tree_sha=source_tree_sha,
        task_tree_sha=task_tree_sha,
        inventory_sha256=inventory_sha,
        fixture_sha256=inventory_sha,
        seed_root=destination,
        files_root=files_root,
        reference_patch_path=patch,
        reference_patch_sha256=patch_sha,
    )


def _operation_payload(operation: FableWriteOp | FableEditOp) -> dict[str, Any]:
    return {"type": type(operation).__name__, **dataclasses.asdict(operation)}


def canonical_mutation_plan(
    messages: Sequence[Mapping[str, Any]],
    protected_paths: frozenset[str],
    verify_cmd: str,
) -> MutationPlan:
    """Parse source calls once and retain only canonical declarative mutations."""

    if type(verify_cmd) is not str or not verify_cmd:
        raise ReplayContractError("verify_cmd must be a nonempty string")
    verifier_sha = _sha256(verify_cmd.encode("utf-8"))
    operations: list[FableWriteOp | FableEditOp] = []
    verifier_count = 0
    for message in messages:
        if type(message) is not dict:
            raise ReplayContractError("trajectory message must be an object")
        if message.get("role") != "assistant":
            continue
        calls = message.get("tool_calls", [])
        if type(calls) is not list:
            raise ReplayContractError("assistant tool_calls must be a list")
        for call in calls:
            operation = parse_fable_tool_call(
                call,
                protected_paths,
                trusted_verifier_commands=frozenset({verify_cmd}),
            )
            if isinstance(operation, (FableWriteOp, FableEditOp)):
                operations.append(operation)
            elif isinstance(operation, VerifierEvidenceOp):
                if operation.command_sha256 != verifier_sha:
                    raise ReplayContractError("verifier evidence hash mismatch")
                verifier_count += 1
            elif isinstance(operation, ReadOnlyBashOp):
                continue
            else:
                # Read/Glob/Grep are observation-only and never affect reconstruction.
                continue
    payload = [_operation_payload(operation) for operation in operations]
    return MutationPlan(
        operations=tuple(operations),
        operation_sha256=_sha256(_canonical_json(payload)),
        verifier_sha256=verifier_sha,
        verifier_evidence_count=verifier_count,
    )


def _candidate_relative(path: str) -> str:
    prefix = "/testbed/"
    if not path.startswith(prefix):
        raise ReplayContractError("mutation target is outside /testbed")
    return _relative_path(path.removeprefix(prefix), field_name="mutation path")


def _revalidate_ancestors(root: Path, relative: str) -> Path:
    root_stat = os.stat(root, follow_symlinks=False)
    if not stat.S_ISDIR(root_stat.st_mode) or root.is_symlink():
        raise ReplayContractError("candidate root is not a real directory")
    cursor = root
    parts = PurePosixPath(relative).parts
    for part in parts[:-1]:
        cursor = cursor / part
        try:
            item = os.stat(cursor, follow_symlinks=False)
        except FileNotFoundError as exc:
            raise ReplayContractError("mutation parent does not exist") from exc
        if not stat.S_ISDIR(item.st_mode) or cursor.is_symlink():
            raise ReplayContractError("mutation ancestor is a symlink or special file")
    target = root.joinpath(*parts)
    try:
        target.relative_to(root)
    except ValueError as exc:
        raise ReplayContractError("mutation target escapes candidate root") from exc
    return target


def _target_stat(target: Path) -> os.stat_result | None:
    try:
        item = os.stat(target, follow_symlinks=False)
    except FileNotFoundError:
        return None
    if stat.S_ISLNK(item.st_mode):
        raise ReplayContractError("mutation target is a symlink")
    if not stat.S_ISREG(item.st_mode):
        raise ReplayContractError("mutation target is a special file")
    if item.st_nlink != 1:
        raise ReplayContractError("mutation target has a hardlink alias")
    return item


def _check_protected_aliases(root: Path, target: Path, contract: SeedContract) -> None:
    for protected in contract.protected_paths:
        protected_target = root.joinpath(*PurePosixPath(protected).parts)
        protected_stat = _target_stat(protected_target)
        target_stat = _target_stat(target)
        if target == protected_target:
            raise ReplayContractError("mutation targets a protected path")
        if (
            target_stat is not None
            and protected_stat is not None
            and (target_stat.st_dev, target_stat.st_ino)
            == (protected_stat.st_dev, protected_stat.st_ino)
        ):
            raise ReplayContractError("mutation targets a protected alias")


def _open_mutation_target(
    target: Path, before: os.stat_result | None, *, create: bool
) -> int:
    flags = os.O_RDWR | os.O_NOFOLLOW
    if create:
        flags |= os.O_CREAT | os.O_EXCL
    descriptor = os.open(target, flags, 0o666)
    opened = os.fstat(descriptor)
    if not stat.S_ISREG(opened.st_mode) or opened.st_nlink != 1:
        os.close(descriptor)
        raise ReplayContractError("opened mutation target is not a unique regular file")
    if before is not None and (
        opened.st_dev,
        opened.st_ino,
        opened.st_nlink,
    ) != (before.st_dev, before.st_ino, before.st_nlink):
        os.close(descriptor)
        raise ReplayContractError("mutation target identity changed before open")
    return descriptor


def _apply_operation(
    root: Path,
    contract: SeedContract,
    operation: FableWriteOp | FableEditOp,
) -> None:
    relative = _candidate_relative(operation.path)
    if relative in contract.protected_paths:
        raise ReplayContractError("mutation targets a protected path")
    target = _revalidate_ancestors(root, relative)
    _check_protected_aliases(root, target, contract)
    before = _target_stat(target)
    if isinstance(operation, FableEditOp) and before is None:
        raise ReplayContractError("edit target does not exist")
    descriptor = _open_mutation_target(
        target, before, create=isinstance(operation, FableWriteOp) and before is None
    )
    try:
        if isinstance(operation, FableWriteOp):
            if before is None:
                os.fchmod(descriptor, 0o644)
            data = operation.content.encode("utf-8")
        else:
            chunks: list[bytes] = []
            while chunk := os.read(descriptor, 1024 * 1024):
                chunks.append(chunk)
            data = b"".join(chunks)
            old = operation.old_string.encode("utf-8")
            new = operation.new_string.encode("utf-8")
            matches = data.count(old)
            if operation.replace_all and matches < 1:
                raise ReplayContractError("edit expected at least one match")
            if not operation.replace_all and matches != 1:
                raise ReplayContractError(
                    f"edit expected exactly one match, found {matches}"
                )
            data = data.replace(old, new)
        os.lseek(descriptor, 0, os.SEEK_SET)
        os.ftruncate(descriptor, 0)
        view = memoryview(data)
        while view:
            written = os.write(descriptor, view)
            view = view[written:]
    finally:
        os.close(descriptor)


def _copy_fixture(contract: SeedContract, destination: Path) -> tuple[_InventoryFile, ...]:
    current = _inventory_from_root(contract.files_root)
    if _inventory_sha(current) != contract.inventory_sha256:
        raise ReplayContractError("immutable seed inventory changed")
    if destination.exists() or destination.is_symlink():
        raise ReplayContractError("candidate destination already exists")
    _write_records(destination, current)
    return current


def _protected_hashes(root: Path, contract: SeedContract) -> tuple[tuple[str, str], ...]:
    return tuple(
        (
            path,
            _sha256(_read_regular_no_follow(root.joinpath(*PurePosixPath(path).parts))[0]),
        )
        for path in sorted(contract.protected_paths)
    )


def reconstruct_candidate(
    contract: SeedContract,
    plan: MutationPlan,
    destination: Path,
    *,
    before_operation: Callable[[int, Path], None] | None = None,
) -> CandidateState:
    """Apply only typed declarative mutations to a fresh seed copy."""

    if plan.verifier_sha256 != contract.verifier_sha256:
        raise ReplayContractError("mutation plan verifier does not match seed contract")
    destination = Path(destination)
    baseline = _copy_fixture(contract, destination)
    baseline_by_path = {record.path: record for record in baseline}
    for index, operation in enumerate(plan.operations):
        if before_operation is not None:
            before_operation(index, destination)
        _apply_operation(destination, contract, operation)
        # Reject aliases or special paths introduced between operations.
        _inventory_from_root(destination)
    final = _inventory_from_root(destination)
    if _protected_hashes(destination, contract) != contract.protected_sha256:
        raise ReplayContractError("protected file changed during reconstruction")
    final_by_path = {record.path: record for record in final}
    changed = tuple(
        sorted(
            {
                *(
                    path
                    for path, record in baseline_by_path.items()
                    if final_by_path.get(path) != record
                ),
                *(path for path in final_by_path if path not in baseline_by_path),
            },
            key=lambda path: path.encode("utf-8"),
        )
    )
    if not changed:
        raise ReplayContractError("empty candidate diff")
    if any(path in contract.protected_paths for path in changed):
        raise ReplayContractError("candidate diff contains a protected path")
    diff_payload = []
    for path in changed:
        before = baseline_by_path.get(path)
        after = final_by_path.get(path)
        diff_payload.append(
            {
                "path": path,
                "before": None
                if before is None
                else {"mode": before.mode, "length": len(before.data), "sha256": _sha256(before.data)},
                "after": None
                if after is None
                else {"mode": after.mode, "length": len(after.data), "sha256": _sha256(after.data)},
            }
        )
    return CandidateState(
        root=destination,
        tree_sha256=_inventory_sha(final),
        diff_sha256=_sha256(_canonical_json(diff_payload)),
        changed_paths=changed,
        operation_sha256=plan.operation_sha256,
        verifier_sha256=plan.verifier_sha256,
        protected_sha256=_protected_hashes(destination, contract),
    )


def _patch_path(value: str) -> str:
    if value == "/dev/null":
        return value
    if value.startswith("a/") or value.startswith("b/"):
        value = value[2:]
    return _relative_path(value, field_name="patch path")


def _parse_reference_patch(
    patch: bytes, protected_paths: frozenset[str]
) -> tuple[str, ...]:
    try:
        text = patch.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ReplayContractError("reference_patch_binary") from exc
    if not text:
        raise ReplayContractError("reference_patch_empty")
    lower = text.lower()
    if "git binary patch" in lower or "binary files " in lower:
        raise ReplayContractError("reference_patch_binary")
    forbidden = {
        "rename from ": "reference_patch_rename",
        "rename to ": "reference_patch_rename",
        "copy from ": "reference_patch_copy",
        "copy to ": "reference_patch_copy",
        "old mode ": "reference_patch_mode",
        "new mode ": "reference_patch_mode",
        "similarity index ": "reference_patch_rename",
    }
    for line in text.splitlines():
        for prefix, reason in forbidden.items():
            if line.startswith(prefix):
                raise ReplayContractError(reason)
        if line.startswith(("new file mode ", "deleted file mode ")):
            mode = line.rsplit(" ", 1)[-1]
            if mode not in {"100644", "100755"}:
                kind = "symlink" if mode == "120000" else "submodule"
                raise ReplayContractError(f"reference_patch_{kind}")
        if re.fullmatch(r"index [0-9a-f]+\.\.[0-9a-f]+ 160000", line):
            raise ReplayContractError("reference_patch_submodule")
        if line.startswith(("-Subproject commit ", "+Subproject commit ")):
            raise ReplayContractError("reference_patch_submodule")
    sections = re.split(r"(?=^diff --git )", text, flags=re.MULTILINE)
    targets: list[str] = []
    for section in sections:
        if not section.startswith("diff --git "):
            if section.strip():
                raise ReplayContractError("reference_patch_missing_diff_header")
            continue
        first = section.splitlines()[0]
        match = re.fullmatch(r"diff --git a/(\S+) b/(\S+)", first)
        if match is None:
            raise ReplayContractError("reference_patch_invalid_path_header")
        old_header, new_header = match.groups()
        old_path = _patch_path(old_header)
        new_path = _patch_path(new_header)
        markers = [line for line in section.splitlines() if line.startswith(("--- ", "+++ "))]
        if len(markers) < 2:
            raise ReplayContractError("reference_patch_mode_only")
        parsed_old = _patch_path(markers[0][4:].split("\t", 1)[0])
        parsed_new = _patch_path(markers[1][4:].split("\t", 1)[0])
        if parsed_old not in {old_path, "/dev/null"} or parsed_new not in {
            new_path,
            "/dev/null",
        }:
            raise ReplayContractError("reference_patch_header_mismatch")
        if (
            parsed_old != "/dev/null"
            and parsed_new != "/dev/null"
            and parsed_old != parsed_new
        ):
            raise ReplayContractError("reference_patch_rename")
        target = parsed_new if parsed_new != "/dev/null" else parsed_old
        if target in protected_paths:
            raise ReplayContractError("reference_patch_protected")
        if target in targets:
            raise ReplayContractError("reference_patch_duplicate_target")
        if not any(line.startswith("@@ ") for line in section.splitlines()):
            raise ReplayContractError("reference_patch_mode_only")
        targets.append(target)
    if not targets:
        raise ReplayContractError("reference_patch_has_no_targets")
    return tuple(targets)


def _default_patch_executor(
    argv: tuple[str, ...], cwd: Path
) -> subprocess.CompletedProcess[bytes]:
    try:
        return subprocess.run(argv, cwd=cwd, capture_output=True, check=False)
    except OSError as exc:
        raise ReplayContractError("git apply --check could not start") from exc


def preflight_reference_patch(
    contract: SeedContract,
    *,
    executor: PatchExecutor = _default_patch_executor,
) -> ReferencePatchContract:
    """Parse and check the pinned patch without ever applying it to a candidate."""

    patch_path = contract.reference_patch_path
    if not patch_path.is_file() or patch_path.is_symlink():
        return ReferencePatchContract(False, "missing_reference_patch", None, ())
    try:
        patch, _ = _read_regular_no_follow(patch_path)
    except ReplayContractError:
        return ReferencePatchContract(False, "invalid_reference_patch_file", None, ())
    patch_sha = _sha256(patch)
    if (
        contract.reference_patch_sha256 is not None
        and patch_sha != contract.reference_patch_sha256
    ):
        return ReferencePatchContract(False, "reference_patch_changed", patch_sha, ())
    try:
        targets = _parse_reference_patch(patch, frozenset(contract.protected_paths))
    except ReplayContractError as exc:
        return ReferencePatchContract(False, str(exc), patch_sha, ())
    with tempfile.TemporaryDirectory(prefix="fable-reference-check-") as temporary:
        root = Path(temporary)
        records = _inventory_from_root(contract.files_root)
        if _inventory_sha(records) != contract.inventory_sha256:
            return ReferencePatchContract(
                False, "immutable_seed_inventory_changed", patch_sha, targets
            )
        _write_records(root / "fixture", records)
        check_root = root / "fixture"
        patch_copy = check_root / "reference_fix.patch"
        descriptor = os.open(
            patch_copy,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
            0o600,
        )
        try:
            with os.fdopen(descriptor, "wb") as handle:
                descriptor = -1
                handle.write(patch)
        finally:
            if descriptor >= 0:
                os.close(descriptor)
        try:
            result = executor(
                ("git", "apply", "--check", "--", "reference_fix.patch"),
                check_root,
            )
        except Exception:
            return ReferencePatchContract(
                False, "reference_patch_apply_check_error", patch_sha, targets
            )
        if result.returncode != 0:
            return ReferencePatchContract(
                False, "reference_patch_apply_check_failed", patch_sha, targets
            )
    return ReferencePatchContract(True, None, patch_sha, targets)


__all__ = [
    "CandidateState",
    "GitSeedSource",
    "MOONSHINER_REVISION",
    "MutationPlan",
    "ReferencePatchContract",
    "ReplayContractError",
    "SeedContract",
    "canonical_mutation_plan",
    "materialize_seed",
    "preflight_reference_patch",
    "reconstruct_candidate",
    "trajectory_identity",
]
