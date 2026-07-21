"""Immutable, declarative reconstruction for pinned Fable trajectories.

This module deliberately stops before executing a verifier.  It admits a seed
from one exact Git object, parses source operations through the importer's
shared typed parser, and reconstructs only declarative Write/Edit mutations.
"""

from __future__ import annotations

import argparse
import dataclasses
import fcntl
import hashlib
import json
import math
import os
import re
import selectors
import secrets
import shlex
import shutil
import stat
import subprocess
import tarfile
import tempfile
import time
import unicodedata
from collections import Counter
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any, Final, Protocol

if __package__:  # Support both ``python -m teacher_platform...`` and local tests.
    from .fable5_import import (
        DATASET_REVISION,
        SOURCE_BYTES,
        SOURCE_LFS_SHA256,
        FableEditOp,
        FableWriteOp,
        ReadOnlyBashOp,
        VerifierEvidenceOp,
        _rejected_mutation_family,
        assess_converted_trajectory,
        parse_exclusion_artifact,
        parse_fable_tool_call,
        select_terminal_row,
    )
else:  # pragma: no cover - the branch is exercised by local tests.
    from fable5_import import (  # type: ignore[no-redef]
        DATASET_REVISION,
        SOURCE_BYTES,
        SOURCE_LFS_SHA256,
        FableEditOp,
        FableWriteOp,
        ReadOnlyBashOp,
        VerifierEvidenceOp,
        _rejected_mutation_family,
        assess_converted_trajectory,
        parse_exclusion_artifact,
        parse_fable_tool_call,
        select_terminal_row,
    )


MOONSHINER_REVISION: Final = "436316e8f86eb136d5ce3ec95a1a6f48c1d7f940"
DEFAULT_VERIFY_TIMEOUT: Final = 300
MAX_VERIFY_TIMEOUT: Final = 1_800
MAX_POLICY_OUTPUT_LIMIT_BYTES: Final = 4 * 1024**2
_TASK_RE: Final = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*\Z")
_OID_RE: Final = re.compile(r"[0-9a-f]{40}(?:[0-9a-f]{24})?\Z")


class ReplayContractError(ValueError):
    """A pinned source, operation, or candidate violated the replay contract."""


class GitRunner(Protocol):
    def __call__(self, argv: tuple[str, ...], env: Mapping[str, str]) -> bytes: ...


class PatchExecutor(Protocol):
    def __call__(
        self, argv: tuple[str, ...], cwd: Path
    ) -> subprocess.CompletedProcess[bytes]: ...


def _sanitized_git_environment() -> dict[str, str]:
    """Return a minimal, non-interactive environment for pinned object reads."""

    return {
        "PATH": os.defpath,
        "HOME": os.devnull,
        "XDG_CONFIG_HOME": os.devnull,
        "LANG": "C",
        "LC_ALL": "C",
        "GIT_ALLOW_PROTOCOL": "file",
        "GIT_CONFIG_COUNT": "0",
        "GIT_CONFIG_GLOBAL": os.devnull,
        "GIT_CONFIG_NOSYSTEM": "1",
        "GIT_NO_LAZY_FETCH": "1",
        "GIT_NO_REPLACE_OBJECTS": "1",
        "GIT_TERMINAL_PROMPT": "0",
        "GCM_INTERACTIVE": "never",
    }


def _run_git(argv: tuple[str, ...], env: Mapping[str, str]) -> bytes:
    try:
        return subprocess.run(
            argv,
            check=True,
            capture_output=True,
            env=dict(env),
        ).stdout
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
class DockerPolicy:
    """Immutable runtime policy keyed only by locally cached image digests."""

    images: tuple[tuple[str, str], ...]
    policy_version: str = "fable-docker-v1"
    docker_binary: str = "docker"
    cpus: str = "2.0"
    memory_bytes: int = 4 * 1024**3
    pids_limit: int = 256
    tmpfs_bytes: int = 4 * 1024**3
    file_limit_blocks: int = 1_048_576
    output_limit_bytes: int = 4 * 1024**2
    result_limit_bytes: int = 16 * 1024
    disk_floor_bytes: int = 40 * 1024**3
    command_timeout_seconds: int = 30
    verifier_uid: int = 65_532
    verifier_gid: int = 65_532
    signal_uid: int = 65_533
    signal_gid: int = 65_533
    observer_uid: int = 65_534
    observer_gid: int = 65_534

    def __post_init__(self) -> None:
        if type(self.images) is not tuple or any(
            type(item) is not tuple
            or len(item) != 2
            or any(type(value) is not str for value in item)
            for item in self.images
        ):
            raise ValueError("Docker policy image mapping must be deeply immutable")
        if not self.images:
            raise ValueError("Docker policy needs at least one exact image digest")
        seen: set[str] = set()
        for language, image in self.images:
            if language not in {"python", "rust", "cpp"} or language in seen:
                raise ValueError("Docker policy languages must be unique and supported")
            seen.add(language)
            if re.fullmatch(r"[^\s@]+@sha256:[0-9a-f]{64}", image) is None:
                raise ValueError("Docker image must use an exact immutable digest")
        if self.docker_binary != "docker" or not self.policy_version:
            raise ValueError("Docker policy command and version are fixed")
        integer_limits = (
            self.memory_bytes,
            self.pids_limit,
            self.tmpfs_bytes,
            self.file_limit_blocks,
            self.output_limit_bytes,
            self.result_limit_bytes,
            self.command_timeout_seconds,
        )
        if any(type(value) is not int or value <= 0 for value in integer_limits):
            raise ValueError("Docker policy resource limits must be positive integers")
        if self.output_limit_bytes > MAX_POLICY_OUTPUT_LIMIT_BYTES:
            raise ValueError(
                "Docker policy output limit exceeds the reviewed v2 maximum"
            )
        if (
            type(self.disk_floor_bytes) is not int
            or self.disk_floor_bytes < 40 * 1024**3
        ):
            raise ValueError("Docker policy disk floor must be at least 40 GiB")
        if (
            type(self.cpus) is not str
            or re.fullmatch(r"(?:0|[1-9][0-9]*)(?:\.[0-9]+)?", self.cpus) is None
            or float(self.cpus) <= 0
        ):
            raise ValueError("Docker policy CPU limit must be a positive decimal")
        identities = (
            (self.verifier_uid, self.verifier_gid),
            (self.signal_uid, self.signal_gid),
            (self.observer_uid, self.observer_gid),
        )
        if any(
            type(uid) is not int
            or type(gid) is not int
            or uid <= 0
            or gid <= 0
            or uid != gid
            for uid, gid in identities
        ) or len(set(identities)) != len(identities):
            raise ValueError("Docker policy identities must be distinct non-root UID/GIDs")

    def image_for(self, language: str) -> str:
        for candidate_language, image in self.images:
            if candidate_language == language:
                return image
        raise ReplayContractError(f"no exact cached image for language {language!r}")


@dataclass(frozen=True)
class RuntimeCommandResult:
    returncode: int
    output: bytes
    duration_seconds: float
    timed_out: bool = False
    truncated: bool = False


class DockerRuntime(Protocol):
    def run(
        self,
        argv: tuple[str, ...],
        *,
        timeout_seconds: int,
        output_limit_bytes: int,
        output_path: Path | None = None,
    ) -> RuntimeCommandResult: ...


@dataclass(frozen=True)
class RunEvidence:
    schema_version: int
    run_id: str
    control_identity: str
    trainable: bool
    returncode: int
    wrapper_returncode: int | None
    duration_seconds: float
    termination: str
    raw_output_sha256: str
    raw_output_bytes: int
    output_truncated: bool
    pre_candidate_tree_sha256: str
    pre_candidate_diff_sha256: str
    post_candidate_tree_sha256: str | None
    protected_before: tuple[tuple[str, str], ...]
    protected_after: tuple[tuple[str, str], ...]
    resolved: bool
    failure_class: str | None
    cleanup_state: str
    policy_version: str
    image_digest: str
    runtime_version: str
    run_contract_sha256: str
    resource_peaks: tuple[tuple[str, int], ...]


class RestrictedExecutor(Protocol):
    policy: DockerPolicy

    def execute(
        self,
        *,
        candidate_root: Path,
        language: str,
        verifier_text: str,
        effective_timeout: int,
        control_identity: str,
        pre_candidate_tree_sha256: str,
        pre_candidate_diff_sha256: str,
        protected_before: tuple[tuple[str, str], ...],
    ) -> RunEvidence: ...


@dataclass(frozen=True)
class AdmissionEvidence:
    schema_version: int
    language: str
    admitted: bool
    policy_version: str
    image_digest: str
    policy_output_limit_bytes: int
    runtime_version: str
    probe_run: RunEvidence
    failure_class: str | None


@dataclass(frozen=True)
class ReplayEvidence:
    schema_version: int
    trajectory_id: str
    task: str
    language: str
    dataset_revision: str
    source_commit_sha: str
    source_tree_sha: str
    task_tree_sha: str
    inventory_sha256: str
    source_terminal_sha256: str
    operation_sha256: str
    candidate_tree_sha256: str
    candidate_diff_sha256: str
    protected_sha256: tuple[tuple[str, str], ...]
    verifier_text: str
    verifier_sha256: str
    source_verify_timeout: int | None
    effective_verify_timeout: int
    policy_version: str
    image_digest: str
    policy_output_limit_bytes: int
    runtime_version: str
    run_contract_sha256: str
    runs: tuple[RunEvidence, RunEvidence]
    resolved: bool
    failure_class: str | None
    control_identity: str
    trainable: bool


@dataclass(frozen=True)
class ControlSetEvidence:
    schema_version: int
    trajectory_id: str
    baseline: RunEvidence
    reference: RunEvidence
    candidate: ReplayEvidence
    corrupt: RunEvidence | None
    admitted: bool
    failure_class: str | None


@dataclass(frozen=True)
class EligibilityCandidate:
    """One explicit structural candidate and its precomputed admission gates."""

    trajectory_id: str
    task: str
    language: str
    operations_supported: bool
    decontaminated: bool
    git_seed_valid: bool
    reference_patch_valid: bool
    language_digest_present: bool
    admission_valid: bool


@dataclass(frozen=True)
class EligibilityBindings:
    source_sha256: str
    sidecar_sha256: str
    policy_sha256: str
    admission_sha256: str
    policy_artifact_sha256: str
    admission_artifact_sha256: str
    smoke_manifest_sha256: str
    exclusion_artifacts: tuple[tuple[str, str], ...]
    seed_evidence_sha256: str
    seed_commit_sha: str
    seed_tree_sha: str


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


def _sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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
            if (
                len(parts) < len(prefix.parts)
                and tuple(parts) == prefix.parts[: len(parts)]
            ):
                if not member.isdir():
                    raise ReplayContractError(
                        "archive task ancestor must be a directory"
                    )
                continue
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
    normalized: dict[str, _InventoryFile] = {}
    for path, record in records.items():
        git_mode, git_object_id = git_entries[path]
        # Git's tar archive adds group-write to regular Git modes (0664/0775).
        # Compare that exact archive representation, then materialize/hash the
        # confined Git mode (0644/0755) used by the replay workspace.
        if record.mode != (git_mode | 0o020):
            raise ReplayContractError("archive mode does not match pinned Git tree")
        algorithm = hashlib.sha1 if len(git_object_id) == 40 else hashlib.sha256
        blob_header = b"blob " + str(len(record.data)).encode("ascii") + b"\0"
        if algorithm(blob_header + record.data).hexdigest() != git_object_id:
            raise ReplayContractError(
                "archive bytes do not match pinned Git blob object ID"
            )
        normalized[path] = _InventoryFile(path, git_mode, record.data)
    return tuple(
        normalized[path] for path in sorted(normalized, key=lambda p: p.encode())
    )


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
    argv = ("git", "--no-replace-objects", "-C", str(source.repo), *args)
    try:
        return source.runner(argv, _sanitized_git_environment())
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
            "-c",
            "tar.umask=0002",
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


def _operation_sha256(
    operations: Sequence[FableWriteOp | FableEditOp],
) -> str:
    return _sha256(
        _canonical_json([_operation_payload(operation) for operation in operations])
    )


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
    return MutationPlan(
        operations=tuple(operations),
        operation_sha256=_operation_sha256(operations),
        verifier_sha256=verifier_sha,
        verifier_evidence_count=verifier_count,
    )


def _candidate_relative(path: str) -> str:
    prefix = "/testbed/"
    if not path.startswith(prefix):
        raise ReplayContractError("mutation target is outside /testbed")
    return _relative_path(path.removeprefix(prefix), field_name="mutation path")


def _is_rejected_mutation_path(relative: str) -> bool:
    parts = PurePosixPath(relative).parts
    return any(
        _rejected_mutation_family(
            part,
            basename=index == len(parts) - 1,
        )
        for index, part in enumerate(parts)
    )


def _validate_mutation_plan(contract: SeedContract, plan: MutationPlan) -> None:
    if type(plan) is not MutationPlan:
        raise ReplayContractError("mutation plan must use the exact canonical type")
    if plan.verifier_sha256 != contract.verifier_sha256:
        raise ReplayContractError("mutation plan verifier does not match seed contract")
    expected_protected = tuple(
        sorted(f"/testbed/{path}" for path in contract.protected_paths)
    )
    for operation in plan.operations:
        if type(operation) not in {FableWriteOp, FableEditOp}:
            raise ReplayContractError(
                "mutation plan requires an exact Write/Edit operation type"
            )
        if operation.protected_paths != expected_protected:
            raise ReplayContractError("operation protected_paths binding mismatch")
        relative = _candidate_relative(operation.path)
        if _is_rejected_mutation_path(relative):
            raise ReplayContractError(
                f"rejected mutation path family: {relative}"
            )
    if _operation_sha256(plan.operations) != plan.operation_sha256:
        raise ReplayContractError("mutation plan operation hash mismatch")


_DIRECTORY_OPEN_FLAGS: Final = (
    os.O_RDONLY
    | os.O_DIRECTORY
    | os.O_NOFOLLOW
    | getattr(os, "O_CLOEXEC", 0)
)


def _identity(item: os.stat_result) -> tuple[int, int, int, int]:
    return (item.st_dev, item.st_ino, item.st_mode, item.st_nlink)


def _validate_directory_descriptor(descriptor: int, context: str) -> os.stat_result:
    item = os.fstat(descriptor)
    if not stat.S_ISDIR(item.st_mode) or item.st_nlink < 1:
        raise ReplayContractError(f"{context} is not a stable directory")
    return item


def _open_directory_path(path: Path, context: str) -> tuple[int, os.stat_result]:
    try:
        before = os.stat(path, follow_symlinks=False)
        if not stat.S_ISDIR(before.st_mode):
            raise ReplayContractError(f"{context} is a symlink or special path")
        descriptor = os.open(path, _DIRECTORY_OPEN_FLAGS)
    except ReplayContractError:
        raise
    except OSError as exc:
        raise ReplayContractError(f"cannot open confined {context}") from exc
    try:
        opened = _validate_directory_descriptor(descriptor, context)
        if _identity(opened) != _identity(before):
            raise ReplayContractError(f"{context} identity changed before open")
        return descriptor, opened
    except Exception:
        os.close(descriptor)
        raise


def _open_directory_at(
    parent_descriptor: int, component: str, context: str
) -> tuple[int, os.stat_result]:
    try:
        descriptor = os.open(
            component,
            _DIRECTORY_OPEN_FLAGS,
            dir_fd=parent_descriptor,
        )
    except OSError as exc:
        raise ReplayContractError(f"{context} is missing, linked, or not a directory") from exc
    try:
        return descriptor, _validate_directory_descriptor(descriptor, context)
    except Exception:
        os.close(descriptor)
        raise


def _walk_directories(
    root_descriptor: int,
    components: Sequence[str],
) -> tuple[int, tuple[tuple[str, tuple[int, int, int, int]], ...]]:
    current = os.dup(root_descriptor)
    identities: list[tuple[str, tuple[int, int, int, int]]] = []
    try:
        for component in components:
            child, child_stat = _open_directory_at(
                current, component, f"mutation ancestor {component!r}"
            )
            os.close(current)
            current = child
            identities.append((component, _identity(child_stat)))
        return current, tuple(identities)
    except Exception:
        os.close(current)
        raise


def _target_stat_at(
    parent_descriptor: int,
    basename: str,
    *,
    context: str = "mutation target",
) -> os.stat_result | None:
    try:
        item = os.stat(
            basename,
            dir_fd=parent_descriptor,
            follow_symlinks=False,
        )
    except FileNotFoundError:
        return None
    except OSError as exc:
        raise ReplayContractError(f"cannot inspect confined {context}") from exc
    if stat.S_ISLNK(item.st_mode):
        raise ReplayContractError(f"{context} is a symlink")
    if not stat.S_ISREG(item.st_mode):
        raise ReplayContractError(f"{context} is a special file")
    if item.st_nlink != 1:
        raise ReplayContractError(f"{context} has a hardlink alias")
    return item


@dataclass(frozen=True)
class _MutationBoundary:
    root: Path
    relative: str
    parent_components: tuple[str, ...]
    basename: str
    root_descriptor: int
    parent_descriptor: int
    root_identity: tuple[int, int, int, int]
    ancestor_identities: tuple[tuple[str, tuple[int, int, int, int]], ...]
    parent_identity: tuple[int, int, int, int]
    target_before: os.stat_result | None

    def close(self) -> None:
        os.close(self.parent_descriptor)
        os.close(self.root_descriptor)


def _open_mutation_boundary(root: Path, relative: str) -> _MutationBoundary:
    parts = PurePosixPath(relative).parts
    root_descriptor, root_stat = _open_directory_path(root, "candidate root")
    try:
        parent_descriptor, ancestor_identities = _walk_directories(
            root_descriptor, parts[:-1]
        )
        try:
            parent_stat = _validate_directory_descriptor(
                parent_descriptor, "mutation parent"
            )
            target_before = _target_stat_at(parent_descriptor, parts[-1])
            return _MutationBoundary(
                root=root,
                relative=relative,
                parent_components=tuple(parts[:-1]),
                basename=parts[-1],
                root_descriptor=root_descriptor,
                parent_descriptor=parent_descriptor,
                root_identity=_identity(root_stat),
                ancestor_identities=ancestor_identities,
                parent_identity=_identity(parent_stat),
                target_before=target_before,
            )
        except Exception:
            os.close(parent_descriptor)
            raise
    except Exception:
        os.close(root_descriptor)
        raise


def _stat_relative_under_root(
    root_descriptor: int, relative: str, *, context: str
) -> os.stat_result:
    parts = PurePosixPath(relative).parts
    parent_descriptor, _ = _walk_directories(root_descriptor, parts[:-1])
    try:
        item = _target_stat_at(parent_descriptor, parts[-1], context=context)
        if item is None:
            raise ReplayContractError(f"{context} is missing")
        return item
    finally:
        os.close(parent_descriptor)


def _check_protected_aliases(
    boundary: _MutationBoundary, contract: SeedContract
) -> None:
    if boundary.relative in contract.protected_paths:
        raise ReplayContractError("mutation targets a protected path")
    for protected in contract.protected_paths:
        protected_stat = _stat_relative_under_root(
            boundary.root_descriptor,
            protected,
            context=f"protected path {protected!r}",
        )
        if (
            boundary.target_before is not None
            and (boundary.target_before.st_dev, boundary.target_before.st_ino)
            == (protected_stat.st_dev, protected_stat.st_ino)
        ):
            raise ReplayContractError("mutation targets a protected alias")


def _confirm_mutation_boundary(boundary: _MutationBoundary) -> None:
    if _identity(os.fstat(boundary.root_descriptor)) != boundary.root_identity:
        raise ReplayContractError("candidate root descriptor identity changed")
    if _identity(os.fstat(boundary.parent_descriptor)) != boundary.parent_identity:
        raise ReplayContractError("mutation parent descriptor identity changed")

    fresh_root, fresh_root_stat = _open_directory_path(
        boundary.root, "candidate root"
    )
    try:
        if _identity(fresh_root_stat) != boundary.root_identity:
            raise ReplayContractError("candidate root path identity changed")
        fresh_parent, fresh_identities = _walk_directories(
            fresh_root, boundary.parent_components
        )
        try:
            if fresh_identities != boundary.ancestor_identities:
                raise ReplayContractError("mutation ancestor identity changed")
            if _identity(os.fstat(fresh_parent)) != boundary.parent_identity:
                raise ReplayContractError("mutation parent path identity changed")
        finally:
            os.close(fresh_parent)
    finally:
        os.close(fresh_root)

    target_now = _target_stat_at(
        boundary.parent_descriptor,
        boundary.basename,
    )
    if (target_now is None) != (boundary.target_before is None):
        raise ReplayContractError("mutation target existence changed before open")
    if target_now is not None and boundary.target_before is not None:
        if _identity(target_now) != _identity(boundary.target_before):
            raise ReplayContractError("mutation target identity changed before open")


def _open_mutation_target(boundary: _MutationBoundary, *, create: bool) -> int:
    _confirm_mutation_boundary(boundary)
    flags = os.O_RDWR | os.O_NOFOLLOW | getattr(os, "O_CLOEXEC", 0)
    if create:
        flags |= os.O_CREAT | os.O_EXCL
    try:
        descriptor = os.open(
            boundary.basename,
            flags,
            0o666,
            dir_fd=boundary.parent_descriptor,
        )
    except OSError as exc:
        raise ReplayContractError("cannot open confined mutation target") from exc
    try:
        opened = os.fstat(descriptor)
        if not stat.S_ISREG(opened.st_mode) or opened.st_nlink != 1:
            raise ReplayContractError(
                "opened mutation target is not a unique regular file"
            )
        if boundary.target_before is not None and _identity(opened) != _identity(
            boundary.target_before
        ):
            raise ReplayContractError("mutation target identity changed before open")
        return descriptor
    except Exception:
        os.close(descriptor)
        raise


def _apply_operation(
    root: Path,
    contract: SeedContract,
    operation: FableWriteOp | FableEditOp,
) -> None:
    relative = _candidate_relative(operation.path)
    if relative in contract.protected_paths:
        raise ReplayContractError("mutation targets a protected path")
    boundary = _open_mutation_boundary(root, relative)
    try:
        _check_protected_aliases(boundary, contract)
        before = boundary.target_before
        if isinstance(operation, FableEditOp) and before is None:
            raise ReplayContractError("edit target does not exist")
        descriptor = _open_mutation_target(
            boundary,
            create=isinstance(operation, FableWriteOp) and before is None,
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
    finally:
        boundary.close()


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

    _validate_mutation_plan(contract, plan)
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
    if any(_is_rejected_mutation_path(path) for path in changed):
        raise ReplayContractError("candidate diff contains a rejected path family")
    diff_payload = []
    for path in changed:
        before = baseline_by_path.get(path)
        after = final_by_path.get(path)
        diff_payload.append(
            {
                "path": path,
                "before": None
                if before is None
                else {
                    "mode": before.mode,
                    "length": len(before.data),
                    "sha256": _sha256(before.data),
                },
                "after": None
                if after is None
                else {
                    "mode": after.mode,
                    "length": len(after.data),
                    "sha256": _sha256(after.data),
                },
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


def _patch_path(value: str, *, strip_transport_prefix: bool) -> str:
    if value == "/dev/null":
        return value
    if strip_transport_prefix and (
        value.startswith("a/") or value.startswith("b/")
    ):
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
        old_path = _patch_path(old_header, strip_transport_prefix=False)
        new_path = _patch_path(new_header, strip_transport_prefix=False)
        markers = [line for line in section.splitlines() if line.startswith(("--- ", "+++ "))]
        if len(markers) < 2:
            raise ReplayContractError("reference_patch_mode_only")
        parsed_old = _patch_path(
            markers[0][4:].split("\t", 1)[0], strip_transport_prefix=True
        )
        parsed_new = _patch_path(
            markers[1][4:].split("\t", 1)[0], strip_transport_prefix=True
        )
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


def _sanitized_docker_environment() -> dict[str, str]:
    return {
        "PATH": os.defpath,
        "HOME": os.devnull,
        "XDG_CONFIG_HOME": os.devnull,
        "LANG": "C",
        "LC_ALL": "C",
    }


class _SubprocessDockerRuntime:
    """Run argv-only Docker clients with incremental, bounded output capture."""

    def run(
        self,
        argv: tuple[str, ...],
        *,
        timeout_seconds: int,
        output_limit_bytes: int,
        output_path: Path | None = None,
    ) -> RuntimeCommandResult:
        started = time.monotonic()
        descriptor = -1
        if output_path is not None:
            descriptor = os.open(
                output_path,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
                0o600,
            )
        process: subprocess.Popen[bytes] | None = None
        captured = bytearray()
        truncated = False
        timed_out = False
        try:
            process = subprocess.Popen(
                argv,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                close_fds=True,
                env=_sanitized_docker_environment(),
            )
            assert process.stdout is not None
            selector = selectors.DefaultSelector()
            selector.register(process.stdout, selectors.EVENT_READ)
            deadline = started + timeout_seconds
            while selector.get_map():
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    timed_out = True
                    process.terminate()
                    try:
                        process.wait(timeout=1)
                    except subprocess.TimeoutExpired:
                        process.kill()
                    break
                events = selector.select(min(remaining, 0.25))
                if not events and process.poll() is not None:
                    chunk = process.stdout.read()
                    if chunk:
                        events = [(None, None)]
                    else:
                        selector.unregister(process.stdout)
                        break
                else:
                    chunk = b""
                for key, _mask in events:
                    if key is not None:
                        chunk = os.read(key.fileobj.fileno(), 64 * 1024)
                    if not chunk:
                        if key is not None:
                            selector.unregister(key.fileobj)
                        continue
                    available = max(0, output_limit_bytes - len(captured))
                    accepted = chunk[:available]
                    captured.extend(accepted)
                    if descriptor >= 0 and accepted:
                        os.write(descriptor, accepted)
                    if len(accepted) != len(chunk):
                        truncated = True
            if process.poll() is None:
                process.wait(timeout=1)
            return RuntimeCommandResult(
                124 if timed_out else process.returncode,
                bytes(captured),
                time.monotonic() - started,
                timed_out=timed_out,
                truncated=truncated,
            )
        except OSError as exc:
            if process is not None and process.poll() is None:
                process.kill()
                process.wait()
            raise ReplayContractError(f"Docker client could not start: {argv!r}") from exc
        finally:
            if descriptor >= 0:
                os.close(descriptor)


def _default_docker_free_bytes() -> int:
    root = Path("/var/lib/docker")
    return shutil.disk_usage(root if root.exists() else Path("/")).free


def _validate_sha256(value: str, field_name: str) -> str:
    if re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise ReplayContractError(f"{field_name} must be a SHA-256 digest")
    return value


def _policy_payload(policy: DockerPolicy, image: str) -> dict[str, Any]:
    return {
        "policy_version": policy.policy_version,
        "image_digest": image,
        "cpus": policy.cpus,
        "memory_bytes": policy.memory_bytes,
        "pids_limit": policy.pids_limit,
        "tmpfs_bytes": policy.tmpfs_bytes,
        "file_limit_blocks": policy.file_limit_blocks,
        "output_limit_bytes": policy.output_limit_bytes,
        "result_limit_bytes": policy.result_limit_bytes,
        "command_timeout_seconds": policy.command_timeout_seconds,
        "disk_floor_bytes": policy.disk_floor_bytes,
        "verifier_uid": policy.verifier_uid,
        "verifier_gid": policy.verifier_gid,
        "signal_uid": policy.signal_uid,
        "signal_gid": policy.signal_gid,
        "observer_uid": policy.observer_uid,
        "observer_gid": policy.observer_gid,
    }


def _executor_run_contract_sha256(
    policy: DockerPolicy,
    image: str,
    *,
    language: str,
    verifier_text: str,
    effective_timeout: int,
    control_identity: str,
    pre_candidate_tree_sha256: str,
    pre_candidate_diff_sha256: str,
    protected_before: tuple[tuple[str, str], ...],
) -> str:
    return _sha256(
        _canonical_json(
            {
                "policy": _policy_payload(policy, image),
                "language": language,
                "verifier_sha256": _sha256(verifier_text.encode()),
                "effective_timeout": effective_timeout,
                "control_identity": control_identity,
                "pre_tree": pre_candidate_tree_sha256,
                "pre_diff": pre_candidate_diff_sha256,
                "protected": protected_before,
            }
        )
    )


def _is_sha256(value: object) -> bool:
    return type(value) is str and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def _is_runtime_version(value: object) -> bool:
    return (
        type(value) is str
        and re.fullmatch(r"[0-9]+(?:\.[0-9]+)+(?:[-+._A-Za-z0-9]*)?", value)
        is not None
        and len(value) <= 128
    )


def _has_positive_resource_peaks(run: RunEvidence) -> bool:
    expected = ("memory_bytes", "pids", "cpu_usec")
    return (
        type(run.resource_peaks) is tuple
        and len(run.resource_peaks) == len(expected)
        and all(
            type(item) is tuple
            and len(item) == 2
            and type(item[0]) is str
            and type(item[1]) is int
            and item[1] > 0
            for item in run.resource_peaks
        )
        and tuple(item[0] for item in run.resource_peaks) == expected
    )


def _common_run_evidence_is_valid(
    run: RunEvidence,
    *,
    control_identity: str,
    trainable: bool,
    pre_candidate_tree_sha256: str,
    pre_candidate_diff_sha256: str,
    protected_before: tuple[tuple[str, str], ...],
    policy_version: str,
    image_digest: str,
    run_contract_sha256: str,
    output_limit_bytes: int,
    effective_timeout: int,
    runtime_version: str | None = None,
) -> bool:
    return (
        type(run) is RunEvidence
        and type(run.schema_version) is int
        and run.schema_version == 1
        and type(run.run_id) is str
        and re.fullmatch(r"[0-9a-f]{32}", run.run_id) is not None
        and run.control_identity == control_identity
        and run.trainable is trainable
        and type(run.duration_seconds) in (int, float)
        and not isinstance(run.duration_seconds, bool)
        and math.isfinite(run.duration_seconds)
        and run.duration_seconds >= 0
        and run.duration_seconds <= effective_timeout
        and _is_sha256(run.raw_output_sha256)
        and type(run.raw_output_bytes) is int
        and 0 <= run.raw_output_bytes <= output_limit_bytes
        and type(run.output_truncated) is bool
        and run.pre_candidate_tree_sha256 == pre_candidate_tree_sha256
        and run.pre_candidate_diff_sha256 == pre_candidate_diff_sha256
        and run.protected_before == protected_before
        and run.policy_version == policy_version
        and run.image_digest == image_digest
        and _is_runtime_version(run.runtime_version)
        and (runtime_version is None or run.runtime_version == runtime_version)
        and run.run_contract_sha256 == run_contract_sha256
        and _is_sha256(run.run_contract_sha256)
        and run.cleanup_state == "verified_removed"
        and _has_positive_resource_peaks(run)
    )


def _strict_positive_run_evidence(
    run: RunEvidence,
    **expected: Any,
) -> bool:
    return (
        _common_run_evidence_is_valid(run, **expected)
        and type(run.returncode) is int
        and run.returncode == 0
        and type(run.wrapper_returncode) is int
        and run.wrapper_returncode == 0
        and run.termination == "exited"
        and _is_sha256(run.post_candidate_tree_sha256)
        and run.protected_after == run.protected_before
        and run.resolved is True
        and run.failure_class is None
    )


def _strict_clean_negative_run_evidence(
    run: RunEvidence,
    **expected: Any,
) -> bool:
    return (
        _common_run_evidence_is_valid(run, **expected)
        and type(run.returncode) is int
        and run.returncode != 0
        and type(run.wrapper_returncode) is int
        and run.wrapper_returncode == 0
        and run.termination == "exited"
        and _is_sha256(run.post_candidate_tree_sha256)
        and run.protected_after == run.protected_before
        and run.resolved is False
        and run.failure_class == "verifier_failed"
    )


def _wrapper_script(run_nonce: str) -> str:
    return f"""#!/bin/sh
set -eu
umask 077
while [ ! -f /control/go ]; do sleep 0.05; done
tree=$(find /work -xdev -type f -printf '%P\\0' | LC_ALL=C sort -z | xargs -0 -r sha256sum -- | sha256sum | awk '{{print $1}}')
memory=$(cat /sys/fs/cgroup/memory.peak)
pids=$(cat /sys/fs/cgroup/pids.peak)
cpu=$(awk '$1 == "usage_usec" {{print $2}}' /sys/fs/cgroup/cpu.stat)
for peak in "$memory" "$pids" "$cpu"; do
  case "$peak" in ''|0|*[!0-9]*) exit 1 ;; esac
done
tmp=/result/result.json.tmp
printf '{{"schema_version":1,"run_nonce":"{run_nonce}","wrapper_rc":0,"post_tree_sha256":"%s","protected_sha256":[' "$tree" > "$tmp"
separator=
for path do
  actual=$(sha256sum -- "/work/$path" | awk '{{print $1}}')
  printf '%s"%s"' "$separator" "$actual" >> "$tmp"
  separator=,
done
printf '],"resource_peaks":{{"memory_bytes":%s,"pids":%s,"cpu_usec":%s}}}}' "$memory" "$pids" "$cpu" >> "$tmp"
chmod 0600 "$tmp"
mv "$tmp" /result/result.json
: > /status/done
while :; do sleep 3600; done
"""


def _verifier_script(verifier_text: str, file_limit_blocks: int) -> str:
    return f"""#!/bin/sh
set -eu
umask 077
cp -R /seed/candidate/. /work/
chmod -R u+rwX /work
cd /work
ulimit -f {file_limit_blocks}
exec /bin/sh -c {shlex.quote(verifier_text)}
"""


def _make_staged_candidate_read_only(root: Path) -> None:
    for current, dirnames, filenames in os.walk(root, followlinks=False):
        current_path = Path(current)
        os.chmod(current_path, 0o555)
        for dirname in dirnames:
            os.chmod(current_path / dirname, 0o555)
        for filename in filenames:
            os.chmod(current_path / filename, 0o444)


def _strict_wrapper_result(
    path: Path,
    *,
    run_nonce: str,
    protected_before: tuple[tuple[str, str], ...],
    limit: int,
) -> tuple[int, str, tuple[tuple[str, str], ...], tuple[tuple[str, int], ...]]:
    try:
        item = os.stat(path, follow_symlinks=False)
    except OSError as exc:
        raise ReplayContractError("wrapper result is missing") from exc
    if (
        not stat.S_ISREG(item.st_mode)
        or item.st_nlink != 1
        or stat.S_IMODE(item.st_mode) != 0o600
        or item.st_size > limit
    ):
        raise ReplayContractError("wrapper result is not a bounded regular file")
    try:
        payload = json.loads(path.read_bytes())
    except (OSError, json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise ReplayContractError("wrapper result is malformed") from exc
    expected = {
        "schema_version",
        "run_nonce",
        "wrapper_rc",
        "post_tree_sha256",
        "protected_sha256",
        "resource_peaks",
    }
    if type(payload) is not dict or set(payload) != expected:
        raise ReplayContractError("wrapper result has the wrong exact schema")
    if payload["schema_version"] != 1 or payload["run_nonce"] != run_nonce:
        raise ReplayContractError("wrapper result binding mismatch")
    if type(payload["wrapper_rc"]) is not int:
        raise ReplayContractError("wrapper result has invalid wrapper rc")
    post_tree = payload["post_tree_sha256"]
    if type(post_tree) is not str:
        raise ReplayContractError("wrapper result has invalid state hash")
    _validate_sha256(post_tree, "wrapper result state hash")
    expected_paths = tuple(path for path, _digest in protected_before)
    protected = payload["protected_sha256"]
    if type(protected) is not list or len(protected) != len(expected_paths):
        raise ReplayContractError("wrapper result protected hash schema mismatch")
    protected_after = tuple(
        (path, _validate_sha256(digest, "wrapper result protected hash"))
        for path, digest in zip(expected_paths, protected, strict=True)
    )
    resources = payload["resource_peaks"]
    resource_keys = ("memory_bytes", "pids", "cpu_usec")
    if type(resources) is not dict or set(resources) != set(resource_keys):
        raise ReplayContractError("wrapper result resource schema mismatch")
    if any(type(resources[key]) is not int or resources[key] <= 0 for key in resource_keys):
        raise ReplayContractError("wrapper result resource values are invalid")
    return (
        payload["wrapper_rc"],
        post_tree,
        protected_after,
        tuple((key, resources[key]) for key in resource_keys),
    )


def _tmpfs_policy(policy: DockerPolicy) -> dict[str, str]:
    return {
        "/work": (
            f"rw,nosuid,nodev,size={policy.tmpfs_bytes},"
            f"uid={policy.verifier_uid},gid={policy.verifier_gid},mode=0700"
        ),
        "/scratch": (
            f"rw,nosuid,nodev,size={policy.tmpfs_bytes},"
            f"uid={policy.verifier_uid},gid={policy.verifier_gid},mode=0700"
        ),
        "/control": (
            "rw,nosuid,nodev,noexec,size=1048576,"
            f"uid={policy.signal_uid},gid={policy.signal_gid},mode=0700"
        ),
        "/result": "rw,nosuid,nodev,noexec,size=1048576,uid=0,gid=0,mode=0700",
        "/status": "rw,nosuid,nodev,noexec,size=1048576,uid=0,gid=0,mode=0755",
    }


_REQUIRED_MASKED_PATHS: Final = frozenset(
    {
        "/proc/acpi",
        "/proc/asound",
        "/proc/kcore",
        "/proc/keys",
        "/proc/latency_stats",
        "/proc/timer_list",
        "/proc/timer_stats",
        "/proc/scsi",
        "/sys/firmware",
    }
)
_REQUIRED_READONLY_PATHS: Final = frozenset(
    {
        "/proc/bus",
        "/proc/fs",
        "/proc/irq",
        "/proc/sys",
        "/proc/sysrq-trigger",
    }
)


def _safe_moby_restriction_paths(
    value: object, required: frozenset[str]
) -> bool:
    """Require stable Moby restrictions while allowing extra restrictive paths."""

    if type(value) is not list or any(type(path) is not str for path in value):
        return False
    if len(value) != len(set(value)):
        return False
    if any(
        re.fullmatch(r"/(?:[A-Za-z0-9._-]+)(?:/[A-Za-z0-9._-]+)*", path)
        is None
        or PurePosixPath(path).as_posix() != path
        for path in value
    ):
        return False
    return required <= set(value)


_QUIESCE_VERIFIER_SCRIPT: Final = """set -eu
self=$$
parent=$PPID
owned() {
  for status in /proc/[0-9]*/status; do
    [ -r "$status" ] || continue
    pid=${status#/proc/}; pid=${pid%/status}
    [ "$pid" = "$self" ] && continue
    [ "$pid" = "$parent" ] && continue
    uid=$(awk '$1 == "Uid:" {print $2}' "$status")
    [ "$uid" = "$(id -u)" ] && printf '%s\n' "$pid"
  done
}
for pid in $(owned); do kill -TERM "$pid" 2>/dev/null || :; done
sleep 0.1
for pid in $(owned); do kill -KILL "$pid" 2>/dev/null || :; done
sleep 0.1
[ -z "$(owned)" ]
"""


class DockerExecutor:
    """Exact-CID Docker verifier with no network, mounts, pulls, or root verifier."""

    def __init__(
        self,
        policy: DockerPolicy,
        *,
        runner: DockerRuntime | None = None,
        disk_free_bytes: Callable[[], int] = _default_docker_free_bytes,
        token_factory: Callable[[], str] = lambda: secrets.token_hex(16),
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.policy = policy
        self.runner = runner or _SubprocessDockerRuntime()
        self.disk_free_bytes = disk_free_bytes
        self.token_factory = token_factory
        self.clock = clock

    def _remaining_lifecycle_seconds(
        self, deadline: float, *, command_cap: bool = True
    ) -> int:
        remaining = int(deadline - self.clock())
        if remaining < 1:
            raise ReplayContractError("restricted Docker container wall timeout")
        if command_cap:
            return min(self.policy.command_timeout_seconds, remaining)
        return remaining

    def _run(
        self,
        argv: tuple[str, ...],
        *,
        timeout: int | None = None,
        output_limit: int | None = None,
        output_path: Path | None = None,
    ) -> RuntimeCommandResult:
        return self.runner.run(
            argv,
            timeout_seconds=(
                self.policy.command_timeout_seconds if timeout is None else timeout
            ),
            output_limit_bytes=(
                self.policy.output_limit_bytes
                if output_limit is None
                else output_limit
            ),
            output_path=output_path,
        )

    def _required(
        self,
        argv: tuple[str, ...],
        *,
        context: str,
        timeout: int | None = None,
        output_limit: int | None = None,
        output_path: Path | None = None,
    ) -> RuntimeCommandResult:
        result = self._run(
            argv,
            timeout=timeout,
            output_limit=output_limit,
            output_path=output_path,
        )
        if result.returncode != 0 or result.timed_out:
            raise ReplayContractError(f"restricted Docker {context} failed")
        return result

    def _inspect_owned(
        self,
        cid: str,
        nonce: str,
        image_id: str,
        expected_cmd: tuple[str, ...],
        *,
        require_running: bool,
        timeout: int | None = None,
    ) -> None:
        result = self._required(
            (self.policy.docker_binary, "inspect", cid),
            context="ownership inspect",
            timeout=timeout,
        )
        try:
            payload = json.loads(result.output)
            if type(payload) is list:
                if len(payload) != 1:
                    raise ValueError
                payload = payload[0]
            labels = payload["Config"]["Labels"]
            config = payload["Config"]
            host = payload["HostConfig"]
            state = payload["State"]
            network = payload["NetworkSettings"]
            if any(
                type(value) is not dict
                for value in (labels, config, host, state, network)
            ):
                raise ValueError
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise ReplayContractError("restricted Docker inspect schema mismatch") from exc
        if (
            payload.get("Id") != cid
            or payload.get("Name") != f"/fable-replay-{nonce}"
            or payload.get("Image") != image_id
            or labels.get("fable.replay.owner") != nonce
            or config.get("User") != "0:0"
            or config.get("Entrypoint") != ["/bin/sh"]
            or config.get("Cmd") != list(expected_cmd)
            or config.get("Volumes") not in (None, {})
            or config.get("ExposedPorts") not in (None, {})
            or host.get("NetworkMode") != "none"
            or host.get("ReadonlyRootfs") is not True
            or host.get("Privileged") is not False
            or host.get("CapAdd") not in (None, [], ())
            or host.get("CapDrop") != ["ALL"]
            or host.get("SecurityOpt") != ["no-new-privileges:true"]
            or host.get("NanoCpus") != int(float(self.policy.cpus) * 1_000_000_000)
            or host.get("Memory") != self.policy.memory_bytes
            or host.get("MemorySwap") != self.policy.memory_bytes
            or host.get("PidsLimit") != self.policy.pids_limit
            or host.get("Tmpfs") != _tmpfs_policy(self.policy)
            or host.get("Binds") not in (None, [], ())
            or host.get("Mounts") not in (None, [], ())
            or host.get("Devices") not in (None, [], ())
            or host.get("DeviceRequests") not in (None, [], ())
            or host.get("DeviceCgroupRules") not in (None, [], ())
            or host.get("VolumesFrom") not in (None, [], ())
            or host.get("Links") not in (None, [], ())
            or host.get("GroupAdd") not in (None, [], ())
            or host.get("Sysctls") not in (None, {})
            or host.get("ExtraHosts") not in (None, [], ())
            or not _safe_moby_restriction_paths(
                host.get("MaskedPaths"), _REQUIRED_MASKED_PATHS
            )
            or not _safe_moby_restriction_paths(
                host.get("ReadonlyPaths"), _REQUIRED_READONLY_PATHS
            )
            or host.get("PidMode") != ""
            or host.get("IpcMode") != "private"
            or host.get("UTSMode") != ""
            or host.get("UsernsMode") != ""
            or host.get("PortBindings") not in (None, {})
            or host.get("PublishAllPorts") is not False
            or host.get("RestartPolicy")
            != {"Name": "no", "MaximumRetryCount": 0}
            or host.get("Runtime") != "runc"
            or host.get("CgroupnsMode") != "private"
            or host.get("Isolation") != ""
            or network.get("Ports") not in (None, {})
            or type(network.get("Networks")) is not dict
            or set(network["Networks"]) != {"none"}
        ):
            raise ReplayContractError("restricted Docker ownership/policy mismatch")
        expected_ulimit = {
            "Name": "fsize",
            "Soft": self.policy.file_limit_blocks,
            "Hard": self.policy.file_limit_blocks,
        }
        if host.get("Ulimits") != [expected_ulimit]:
            raise ReplayContractError("restricted Docker resource policy mismatch")
        mounts = payload.get("Mounts")
        if type(mounts) is not list:
            raise ReplayContractError("restricted Docker mount policy mismatch")
        if require_running:
            expected_tmpfs = _tmpfs_policy(self.policy)
            if len(mounts) != len(expected_tmpfs):
                raise ReplayContractError("restricted Docker mount policy mismatch")
            seen_destinations: set[str] = set()
            for mount in mounts:
                if type(mount) is not dict or set(mount) != {
                    "Type",
                    "Source",
                    "Destination",
                    "Mode",
                    "RW",
                    "Propagation",
                }:
                    raise ReplayContractError("restricted Docker mount policy mismatch")
                destination = mount.get("Destination")
                if (
                    mount.get("Type") != "tmpfs"
                    or mount.get("Source") != ""
                    or type(destination) is not str
                    or destination not in expected_tmpfs
                    or mount.get("Mode") != ""
                    or mount.get("RW") is not True
                    or mount.get("Propagation") != ""
                    or destination in seen_destinations
                ):
                    raise ReplayContractError("restricted Docker mount policy mismatch")
                seen_destinations.add(destination)
            if (
                seen_destinations != set(expected_tmpfs)
                or state.get("Status") != "running"
                or state.get("Running") is not True
                or type(state.get("Pid")) is not int
                or state["Pid"] <= 0
            ):
                raise ReplayContractError("restricted Docker wrapper is not running")
        elif mounts:
            raise ReplayContractError("restricted Docker mount policy mismatch")
        elif (
            state.get("Status") != "created"
            or state.get("Running") is not False
            or state.get("Pid") != 0
        ):
            raise ReplayContractError("restricted Docker pre-start state mismatch")

    def _inspect_exact_owner(self, cid: str, nonce: str) -> None:
        if re.fullmatch(r"[0-9a-f]{64}", cid) is None:
            raise ReplayContractError("cleanup candidate is not an exact CID")
        inspected = self._required(
            (self.policy.docker_binary, "inspect", cid),
            context="cleanup ownership inspect",
        )
        try:
            payload = json.loads(inspected.output)
            if type(payload) is list:
                if len(payload) != 1:
                    raise ValueError
                payload = payload[0]
            owner = payload["Config"]["Labels"]["fable.replay.owner"]
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise ReplayContractError("cleanup ownership schema mismatch") from exc
        if payload.get("Id") != cid or owner != nonce:
            raise ReplayContractError("cleanup exact CID ownership mismatch")

    def _query_owned_cids(self, nonce: str) -> list[str]:
        queried = self._required(
            (
                self.policy.docker_binary,
                "ps",
                "-aq",
                "--no-trunc",
                "--filter",
                f"label=fable.replay.owner={nonce}",
            ),
            context="owned CID query",
        )
        try:
            ids = [
                line.strip()
                for line in queried.output.decode("ascii").splitlines()
                if line.strip()
            ]
        except UnicodeDecodeError as exc:
            raise ReplayContractError("owned CID query is not ASCII") from exc
        if len(ids) != len(set(ids)) or any(
            re.fullmatch(r"[0-9a-f]{64}", cid) is None for cid in ids
        ):
            raise ReplayContractError("owned CID query returned an invalid exact CID")
        return ids

    def _cleanup_one(self, cid: str, nonce: str, failures: list[str]) -> None:
        docker = self.policy.docker_binary
        try:
            self._inspect_exact_owner(cid, nonce)
        except ReplayContractError as exc:
            failures.append(str(exc))
            return
        self._run((docker, "logs", cid), output_limit=self.policy.output_limit_bytes)
        stopped = self._run((docker, "stop", "--time=2", cid))
        if stopped.returncode != 0 or stopped.timed_out:
            failures.append("owned_stop_failed")
        try:
            self._inspect_exact_owner(cid, nonce)
        except ReplayContractError as exc:
            failures.append(str(exc))
            return
        removed = self._run((docker, "rm", "--force", "--volumes", cid))
        if removed.returncode != 0 or removed.timed_out:
            failures.append("owned_remove_failed")

    def _cleanup(self, nonce: str, primary_cid: str | None) -> str:
        failures: list[str] = []
        if primary_cid is None:
            try:
                candidates = self._query_owned_cids(nonce)
            except ReplayContractError as exc:
                failures.append(str(exc))
                candidates = []
        else:
            candidates = [primary_cid]
        for candidate in candidates:
            self._cleanup_one(candidate, nonce, failures)
        try:
            descendants = self._query_owned_cids(nonce)
        except ReplayContractError as exc:
            failures.append(str(exc))
            descendants = []
        for descendant in descendants:
            if descendant not in candidates:
                self._cleanup_one(descendant, nonce, failures)
        try:
            if self._query_owned_cids(nonce):
                failures.append("owned descendants remain")
        except ReplayContractError as exc:
            failures.append(str(exc))
        if failures:
            raise ReplayContractError(
                "restricted Docker cleanup failed: " + ",".join(failures)
            )
        return "verified_removed"

    def execute(
        self,
        *,
        candidate_root: Path,
        language: str,
        verifier_text: str,
        effective_timeout: int,
        control_identity: str,
        pre_candidate_tree_sha256: str,
        pre_candidate_diff_sha256: str,
        protected_before: tuple[tuple[str, str], ...],
    ) -> RunEvidence:
        if control_identity not in {"candidate", "baseline", "reference", "corrupt", "admission"}:
            raise ReplayContractError("control identity is not explicit")
        if type(effective_timeout) is not int or not 0 < effective_timeout <= MAX_VERIFY_TIMEOUT:
            raise ReplayContractError("effective verifier timeout is invalid")
        _validate_sha256(pre_candidate_tree_sha256, "pre-candidate tree")
        _validate_sha256(pre_candidate_diff_sha256, "pre-candidate diff")
        for path, digest in protected_before:
            _relative_path(path, field_name="protected path")
            _validate_sha256(digest, "protected hash")
        image = self.policy.image_for(language)
        if self.disk_free_bytes() <= self.policy.disk_floor_bytes:
            raise ReplayContractError("Docker disk floor is not preserved before create")
        records = _inventory_from_root(Path(candidate_root))
        if _inventory_sha(records) != pre_candidate_tree_sha256:
            raise ReplayContractError("candidate tree changed before restricted execution")
        nonce = self.token_factory()
        if re.fullmatch(r"[0-9a-f]{32}", nonce) is None:
            raise ReplayContractError("run nonce is invalid")
        docker = self.policy.docker_binary
        cid: str | None = None
        create_succeeded = False
        cleanup_state = "not_created"
        evidence: RunEvidence | None = None
        pending_error: BaseException | None = None
        with tempfile.TemporaryDirectory(prefix="fable-docker-run-") as temporary:
            host = Path(temporary)
            stage = host / "stage"
            stage.mkdir(mode=0o700)
            _write_records(stage / "candidate", records)
            _make_staged_candidate_read_only(stage / "candidate")
            wrapper = stage / "wrapper.sh"
            verifier = stage / "verifier.sh"
            wrapper.write_text(
                _wrapper_script(nonce), encoding="utf-8"
            )
            verifier.write_text(
                _verifier_script(verifier_text, self.policy.file_limit_blocks),
                encoding="utf-8",
            )
            os.chmod(wrapper, 0o500)
            os.chmod(verifier, 0o555)
            os.chmod(stage, 0o555)
            raw_output = b""
            try:
                runtime = self._required(
                    (docker, "version", "--format={{.Server.Version}}"),
                    context="runtime version",
                ).output.decode("ascii").strip()
                if not runtime or len(runtime) > 128:
                    raise ReplayContractError("Docker runtime version is invalid")
                image_id = self._required(
                    (docker, "image", "inspect", "--format={{.Id}}", image),
                    context="cached image inspect",
                ).output.decode("ascii").strip()
                if re.fullmatch(r"sha256:[0-9a-f]{64}", image_id) is None:
                    raise ReplayContractError("cached image inspect returned an invalid ID")
                tmpfs = _tmpfs_policy(self.policy)
                container_cmd = (
                    "/seed/wrapper.sh",
                    *(path for path, _digest in protected_before),
                )
                create = (
                    docker,
                    "create",
                    f"--name=fable-replay-{nonce}",
                    "--pull=never",
                    "--network=none",
                    "--read-only",
                    "--cap-drop=ALL",
                    "--security-opt=no-new-privileges:true",
                    "--ipc=private",
                    "--cgroupns=private",
                    "--runtime=runc",
                    "--restart=no",
                    "--user=0:0",
                    f"--cpus={self.policy.cpus}",
                    f"--memory={self.policy.memory_bytes}",
                    f"--memory-swap={self.policy.memory_bytes}",
                    f"--pids-limit={self.policy.pids_limit}",
                    f"--ulimit=fsize={self.policy.file_limit_blocks}:{self.policy.file_limit_blocks}",
                    "--label",
                    f"fable.replay.owner={nonce}",
                    "--tmpfs",
                    f"/work:{tmpfs['/work']}",
                    "--tmpfs",
                    f"/scratch:{tmpfs['/scratch']}",
                    "--tmpfs",
                    f"/control:{tmpfs['/control']}",
                    "--tmpfs",
                    f"/result:{tmpfs['/result']}",
                    "--tmpfs",
                    f"/status:{tmpfs['/status']}",
                    "--entrypoint=/bin/sh",
                    image,
                    *container_cmd,
                )
                created = self._required(create, context="create")
                create_succeeded = True
                created_cid = created.output.decode("ascii").strip()
                if re.fullmatch(r"[0-9a-f]{64}", created_cid) is None:
                    raise ReplayContractError("Docker create returned an invalid exact CID")
                self._inspect_owned(
                    created_cid,
                    nonce,
                    image_id,
                    container_cmd,
                    require_running=False,
                )
                cid = created_cid
                self._required(
                    (docker, "cp", str(stage), f"{cid}:/seed"),
                    context="stopped-container input copy",
                )
                lifecycle_deadline = self.clock() + effective_timeout
                self._required(
                    (docker, "start", cid),
                    context="start",
                    timeout=self._remaining_lifecycle_seconds(lifecycle_deadline),
                )
                self._inspect_owned(
                    cid,
                    nonce,
                    image_id,
                    container_cmd,
                    require_running=True,
                    timeout=self._remaining_lifecycle_seconds(lifecycle_deadline),
                )
                output_path = host / "verifier-output.log"
                verifier_result = self._run(
                    (
                        docker,
                        "exec",
                        "--user",
                        f"{self.policy.verifier_uid}:{self.policy.verifier_gid}",
                        cid,
                        "/bin/sh",
                        "/seed/verifier.sh",
                    ),
                    timeout=self._remaining_lifecycle_seconds(
                        lifecycle_deadline, command_cap=False
                    ),
                    output_limit=self.policy.output_limit_bytes,
                    output_path=output_path,
                )
                raw_output = verifier_result.output[: self.policy.output_limit_bytes]
                run_contract = _executor_run_contract_sha256(
                    self.policy,
                    image,
                    language=language,
                    verifier_text=verifier_text,
                    effective_timeout=effective_timeout,
                    control_identity=control_identity,
                    pre_candidate_tree_sha256=pre_candidate_tree_sha256,
                    pre_candidate_diff_sha256=pre_candidate_diff_sha256,
                    protected_before=protected_before,
                )
                if verifier_result.timed_out:
                    evidence = RunEvidence(
                        1,
                        nonce,
                        control_identity,
                        False,
                        verifier_result.returncode,
                        None,
                        verifier_result.duration_seconds,
                        "wall_timeout",
                        _sha256(raw_output),
                        len(raw_output),
                        verifier_result.truncated,
                        pre_candidate_tree_sha256,
                        pre_candidate_diff_sha256,
                        None,
                        protected_before,
                        (),
                        False,
                        "verifier_timeout",
                        "cleanup_pending",
                        self.policy.policy_version,
                        image,
                        runtime,
                        run_contract,
                        (),
                    )
                else:
                    self._required(
                        (
                            docker,
                            "exec",
                            "--user",
                            f"{self.policy.verifier_uid}:{self.policy.verifier_gid}",
                            cid,
                            "/bin/sh",
                            "-ceu",
                            _QUIESCE_VERIFIER_SCRIPT,
                        ),
                        context="verifier process quiescence",
                        timeout=min(
                            30, self._remaining_lifecycle_seconds(lifecycle_deadline)
                        ),
                    )
                    self._required(
                        (
                            docker,
                            "exec",
                            "--user",
                            f"{self.policy.signal_uid}:{self.policy.signal_gid}",
                            cid,
                            "/bin/sh",
                            "-ceu",
                            "umask 077; : > /control/go",
                        ),
                        context="wrapper signal",
                        timeout=self._remaining_lifecycle_seconds(lifecycle_deadline),
                    )
                    self._required(
                        (
                            docker,
                            "exec",
                            "--user",
                            f"{self.policy.observer_uid}:{self.policy.observer_gid}",
                            cid,
                            "/bin/sh",
                            "-ceu",
                            "i=0; while [ ! -f /status/done ]; do i=$((i+1)); [ \"$i\" -lt 600 ] || exit 1; sleep 0.05; done",
                        ),
                        context="done marker",
                        timeout=min(
                            30, self._remaining_lifecycle_seconds(lifecycle_deadline)
                        ),
                    )
                    self._inspect_owned(
                        cid,
                        nonce,
                        image_id,
                        container_cmd,
                        require_running=True,
                        timeout=self._remaining_lifecycle_seconds(lifecycle_deadline),
                    )
                    result_path = host / "result.json"
                    self._required(
                        (docker, "cp", f"{cid}:/result/result.json", str(result_path)),
                        context="bounded result copy",
                        output_limit=self.policy.result_limit_bytes,
                        timeout=self._remaining_lifecycle_seconds(lifecycle_deadline),
                    )
                    self._inspect_owned(
                        cid,
                        nonce,
                        image_id,
                        container_cmd,
                        require_running=True,
                        timeout=self._remaining_lifecycle_seconds(lifecycle_deadline),
                    )
                    wrapper_rc, post_tree, protected_after, peaks = _strict_wrapper_result(
                        result_path,
                        run_nonce=nonce,
                        protected_before=protected_before,
                        limit=self.policy.result_limit_bytes,
                    )
                    protected_stable = protected_after == protected_before
                    resolved = (
                        verifier_result.returncode == 0
                        and wrapper_rc == 0
                        and protected_stable
                    )
                    failure = None
                    if verifier_result.returncode != 0:
                        failure = "verifier_failed"
                    elif wrapper_rc != 0:
                        failure = "wrapper_failed"
                    elif not protected_stable:
                        failure = "protected_drift"
                    evidence = RunEvidence(
                        1,
                        nonce,
                        control_identity,
                        control_identity == "candidate" and resolved,
                        verifier_result.returncode,
                        wrapper_rc,
                        verifier_result.duration_seconds,
                        "exited",
                        _sha256(raw_output),
                        len(raw_output),
                        verifier_result.truncated,
                        pre_candidate_tree_sha256,
                        pre_candidate_diff_sha256,
                        post_tree,
                        protected_before,
                        protected_after,
                        resolved,
                        failure,
                        "cleanup_pending",
                        self.policy.policy_version,
                        image,
                        runtime,
                        run_contract,
                        peaks,
                    )
            except BaseException as exc:
                pending_error = exc
            finally:
                if create_succeeded:
                    try:
                        cleanup_state = self._cleanup(nonce, cid)
                    except BaseException as cleanup_exc:
                        if pending_error is None:
                            pending_error = cleanup_exc
                        else:
                            combined = ReplayContractError(
                                f"{cleanup_exc}; preceding execution error: "
                                f"{pending_error}"
                            )
                            combined.__cause__ = pending_error
                            pending_error = combined
                        cleanup_state = "cleanup_failed"
                if self.disk_free_bytes() <= self.policy.disk_floor_bytes:
                    floor_error = ReplayContractError(
                        "Docker disk floor is not preserved after cleanup"
                    )
                    if pending_error is not None:
                        combined = ReplayContractError(
                            f"{floor_error}; preceding restricted execution error: "
                            f"{pending_error}"
                        )
                        combined.__cause__ = pending_error
                        floor_error = combined
                    pending_error = floor_error
            if pending_error is not None:
                if isinstance(pending_error, ReplayContractError):
                    raise pending_error
                raise ReplayContractError("restricted Docker execution failed") from pending_error
            if evidence is None:
                raise ReplayContractError("restricted Docker produced no run evidence")
            return dataclasses.replace(evidence, cleanup_state=cleanup_state)


_ADMISSION_COMMANDS: Final = {
    "python": "python3 -c 'print(\"fable-admission-v1\")'",
    "rust": "rustc --version",
    "cpp": "c++ --version",
}


def _admission_tree_sha256() -> str:
    return _inventory_sha(
        (_InventoryFile("README", 0o644, b"fable admission probe\n"),)
    )


def functional_admission_probe(
    policy: DockerPolicy,
    language: str,
    *,
    executor: RestrictedExecutor | None = None,
    workspace: Path | None = None,
) -> AdmissionEvidence:
    """Functionally probe one toolchain through the complete production policy."""

    if language not in _ADMISSION_COMMANDS:
        raise ReplayContractError("unsupported admission language")
    active = executor or DockerExecutor(policy)
    with tempfile.TemporaryDirectory(
        prefix="fable-admission-", dir=None if workspace is None else workspace.parent
    ) as temporary:
        root = Path(temporary) if workspace is None else Path(workspace)
        if workspace is not None:
            root.mkdir(parents=True, mode=0o700)
        probe = root / "probe"
        probe.mkdir(mode=0o700)
        readme = probe / "README"
        readme.write_bytes(b"fable admission probe\n")
        os.chmod(readme, 0o644)
        records = _inventory_from_root(probe)
        tree = _inventory_sha(records)
        if tree != _admission_tree_sha256():
            raise ReplayContractError("admission probe fixture identity mismatch")
        run = active.execute(
            candidate_root=probe,
            language=language,
            verifier_text=_ADMISSION_COMMANDS[language],
            effective_timeout=60,
            control_identity="admission",
            pre_candidate_tree_sha256=tree,
            pre_candidate_diff_sha256=_sha256(b"admission"),
            protected_before=(),
        )
    image = policy.image_for(language)
    admission_diff = _sha256(b"admission")
    expected_contract = _executor_run_contract_sha256(
        policy,
        image,
        language=language,
        verifier_text=_ADMISSION_COMMANDS[language],
        effective_timeout=60,
        control_identity="admission",
        pre_candidate_tree_sha256=tree,
        pre_candidate_diff_sha256=admission_diff,
        protected_before=(),
    )
    admitted = _strict_positive_run_evidence(
        run,
        control_identity="admission",
        trainable=False,
        pre_candidate_tree_sha256=tree,
        pre_candidate_diff_sha256=admission_diff,
        protected_before=(),
        policy_version=policy.policy_version,
        image_digest=image,
        run_contract_sha256=expected_contract,
        output_limit_bytes=policy.output_limit_bytes,
        effective_timeout=60,
    )
    return AdmissionEvidence(
        1,
        language,
        admitted,
        policy.policy_version,
        image,
        policy.output_limit_bytes,
        run.runtime_version,
        run,
        None if admitted else "admission_invalid_evidence",
    )


def _candidate_state_for_control(
    contract: SeedContract, root: Path, identity: str
) -> CandidateState:
    inventory = _inventory_from_root(root)
    tree = _inventory_sha(inventory)
    diff = _sha256(
        _canonical_json(
            {
                "identity": identity,
                "tree": tree,
                "baseline": contract.inventory_sha256,
            }
        )
    )
    return CandidateState(
        root=root,
        tree_sha256=tree,
        diff_sha256=diff,
        changed_paths=(),
        operation_sha256=_sha256(identity.encode()),
        verifier_sha256=contract.verifier_sha256,
        protected_sha256=_protected_hashes(root, contract),
    )


def _replay_contract_sha(
    contract: SeedContract,
    plan: MutationPlan,
    *,
    trajectory_id: str,
    source_terminal_sha256: str,
    candidate: CandidateState,
    runs: Sequence[RunEvidence],
    policy_output_limit_bytes: int,
) -> str:
    return _sha256(
        _canonical_json(
            {
                "schema_version": 2,
                "dataset_revision": DATASET_REVISION,
                "source_commit_sha": contract.source_commit_sha,
                "source_tree_sha": contract.source_tree_sha,
                "task_tree_sha": contract.task_tree_sha,
                "inventory_sha256": contract.inventory_sha256,
                "trajectory_id": trajectory_id,
                "source_terminal_sha256": source_terminal_sha256,
                "operation_sha256": plan.operation_sha256,
                "candidate_tree_sha256": candidate.tree_sha256,
                "candidate_diff_sha256": candidate.diff_sha256,
                "verifier_sha256": contract.verifier_sha256,
                "source_verify_timeout": contract.source_verify_timeout,
                "effective_verify_timeout": contract.effective_verify_timeout,
                "policy_output_limit_bytes": policy_output_limit_bytes,
                "executor_runs": [run.run_contract_sha256 for run in runs],
            }
        )
    )


def verify_candidate(
    contract: SeedContract,
    plan: MutationPlan,
    *,
    trajectory_id: str,
    source_terminal_sha256: str,
    executor: RestrictedExecutor,
    workspace: Path,
) -> ReplayEvidence:
    """Rebuild and verify two independent candidate states."""

    _validate_sha256(trajectory_id, "trajectory ID")
    _validate_sha256(source_terminal_sha256, "source terminal")
    workspace = Path(workspace)
    workspace.mkdir(parents=True, mode=0o700)
    states = (
        reconstruct_candidate(contract, plan, workspace / "candidate-run-1"),
        reconstruct_candidate(contract, plan, workspace / "candidate-run-2"),
    )
    same_state = (
        states[0].tree_sha256 == states[1].tree_sha256
        and states[0].diff_sha256 == states[1].diff_sha256
        and states[0].changed_paths == states[1].changed_paths
        and bool(states[0].changed_paths)
    )
    runs = tuple(
        executor.execute(
            candidate_root=state.root,
            language=contract.language,
            verifier_text=contract.verify_cmd,
            effective_timeout=contract.effective_verify_timeout,
            control_identity="candidate",
            pre_candidate_tree_sha256=state.tree_sha256,
            pre_candidate_diff_sha256=state.diff_sha256,
            protected_before=state.protected_sha256,
        )
        for state in states
    )
    assert len(runs) == 2
    policy = getattr(executor, "policy", None)
    if type(policy) is not DockerPolicy:
        raise ReplayContractError("restricted executor has no exact Docker policy")
    image = policy.image_for(contract.language)
    runtime_version = runs[0].runtime_version
    run_evidence_valid = all(
        _strict_positive_run_evidence(
            run,
            control_identity="candidate",
            trainable=True,
            pre_candidate_tree_sha256=state.tree_sha256,
            pre_candidate_diff_sha256=state.diff_sha256,
            protected_before=state.protected_sha256,
            policy_version=policy.policy_version,
            image_digest=image,
            run_contract_sha256=_executor_run_contract_sha256(
                policy,
                image,
                language=contract.language,
                verifier_text=contract.verify_cmd,
                effective_timeout=contract.effective_verify_timeout,
                control_identity="candidate",
                pre_candidate_tree_sha256=state.tree_sha256,
                pre_candidate_diff_sha256=state.diff_sha256,
                protected_before=state.protected_sha256,
            ),
            output_limit_bytes=policy.output_limit_bytes,
            effective_timeout=contract.effective_verify_timeout,
            runtime_version=runtime_version,
        )
        for run, state in zip(runs, states, strict=True)
    )
    repeatable = (
        same_state
        and run_evidence_valid
    )
    failure = None if repeatable else "candidate_not_repeatable"
    return ReplayEvidence(
        2,
        trajectory_id,
        contract.task,
        contract.language,
        DATASET_REVISION,
        contract.source_commit_sha,
        contract.source_tree_sha,
        contract.task_tree_sha,
        contract.inventory_sha256,
        source_terminal_sha256,
        plan.operation_sha256,
        states[0].tree_sha256,
        states[0].diff_sha256,
        states[0].protected_sha256,
        contract.verify_cmd,
        contract.verifier_sha256,
        contract.source_verify_timeout,
        contract.effective_verify_timeout,
        policy.policy_version,
        image,
        policy.output_limit_bytes,
        runtime_version,
        _replay_contract_sha(
            contract,
            plan,
            trajectory_id=trajectory_id,
            source_terminal_sha256=source_terminal_sha256,
            candidate=states[0],
            runs=runs,
            policy_output_limit_bytes=policy.output_limit_bytes,
        ),
        (runs[0], runs[1]),
        repeatable,
        failure,
        "candidate",
        repeatable,
    )


def _default_reference_builder(
    contract: SeedContract, destination: Path
) -> CandidateState:
    preflight = preflight_reference_patch(contract)
    if not preflight.eligible:
        raise ReplayContractError(
            preflight.exclusion_reason or "reference patch is ineligible"
        )
    records = _inventory_from_root(contract.files_root)
    _write_records(destination, records)
    patch_copy = destination / ".fable-reference.patch"
    patch, _mode = _read_regular_no_follow(contract.reference_patch_path)
    patch_copy.write_bytes(patch)
    result = _default_patch_executor(
        ("git", "apply", "--", patch_copy.name), destination
    )
    patch_copy.unlink()
    if result.returncode != 0:
        raise ReplayContractError("reference patch application failed")
    return _candidate_state_for_control(contract, destination, "reference")


def run_control_set(
    contract: SeedContract,
    plan: MutationPlan,
    *,
    trajectory_id: str,
    source_terminal_sha256: str,
    executor: RestrictedExecutor,
    workspace: Path,
    reference_builder: Callable[[SeedContract, Path], CandidateState] = _default_reference_builder,
    corrupt_builder: Callable[[SeedContract, Path], CandidateState] | None = None,
) -> ControlSetEvidence:
    """Run tainted controls and two-run candidate verification."""

    workspace = Path(workspace)
    workspace.mkdir(parents=True, mode=0o700)
    baseline_root = workspace / "baseline"
    _write_records(baseline_root, _inventory_from_root(contract.files_root))
    baseline_state = _candidate_state_for_control(contract, baseline_root, "baseline")
    baseline = executor.execute(
        candidate_root=baseline_state.root,
        language=contract.language,
        verifier_text=contract.verify_cmd,
        effective_timeout=contract.effective_verify_timeout,
        control_identity="baseline",
        pre_candidate_tree_sha256=baseline_state.tree_sha256,
        pre_candidate_diff_sha256=baseline_state.diff_sha256,
        protected_before=baseline_state.protected_sha256,
    )
    reference_state = reference_builder(contract, workspace / "reference")
    reference = executor.execute(
        candidate_root=reference_state.root,
        language=contract.language,
        verifier_text=contract.verify_cmd,
        effective_timeout=contract.effective_verify_timeout,
        control_identity="reference",
        pre_candidate_tree_sha256=reference_state.tree_sha256,
        pre_candidate_diff_sha256=reference_state.diff_sha256,
        protected_before=reference_state.protected_sha256,
    )
    corrupt = None
    if corrupt_builder is not None:
        corrupt_state = corrupt_builder(contract, workspace / "corrupt")
        corrupt = executor.execute(
            candidate_root=corrupt_state.root,
            language=contract.language,
            verifier_text=contract.verify_cmd,
            effective_timeout=contract.effective_verify_timeout,
            control_identity="corrupt",
            pre_candidate_tree_sha256=corrupt_state.tree_sha256,
            pre_candidate_diff_sha256=corrupt_state.diff_sha256,
            protected_before=corrupt_state.protected_sha256,
        )
        corrupt = dataclasses.replace(corrupt, trainable=False)
    candidate = verify_candidate(
        contract,
        plan,
        trajectory_id=trajectory_id,
        source_terminal_sha256=source_terminal_sha256,
        executor=executor,
        workspace=workspace / "candidate",
    )
    policy = getattr(executor, "policy", None)
    if type(policy) is not DockerPolicy:
        raise ReplayContractError("restricted executor has no exact Docker policy")
    image = policy.image_for(contract.language)

    def expected_run_contract(identity: str, state: CandidateState) -> str:
        return _executor_run_contract_sha256(
            policy,
            image,
            language=contract.language,
            verifier_text=contract.verify_cmd,
            effective_timeout=contract.effective_verify_timeout,
            control_identity=identity,
            pre_candidate_tree_sha256=state.tree_sha256,
            pre_candidate_diff_sha256=state.diff_sha256,
            protected_before=state.protected_sha256,
        )

    baseline_expected = {
        "control_identity": "baseline",
        "trainable": False,
        "pre_candidate_tree_sha256": baseline_state.tree_sha256,
        "pre_candidate_diff_sha256": baseline_state.diff_sha256,
        "protected_before": baseline_state.protected_sha256,
        "policy_version": policy.policy_version,
        "image_digest": image,
        "run_contract_sha256": expected_run_contract("baseline", baseline_state),
        "output_limit_bytes": policy.output_limit_bytes,
        "effective_timeout": contract.effective_verify_timeout,
    }
    baseline_is_clean_failure = _strict_clean_negative_run_evidence(
        baseline, **baseline_expected
    )
    baseline_is_clean_pass = _strict_positive_run_evidence(
        baseline, **baseline_expected
    )
    reference_is_clean_pass = _strict_positive_run_evidence(
        reference,
        control_identity="reference",
        trainable=False,
        pre_candidate_tree_sha256=reference_state.tree_sha256,
        pre_candidate_diff_sha256=reference_state.diff_sha256,
        protected_before=reference_state.protected_sha256,
        policy_version=policy.policy_version,
        image_digest=image,
        run_contract_sha256=expected_run_contract("reference", reference_state),
        output_limit_bytes=policy.output_limit_bytes,
        effective_timeout=contract.effective_verify_timeout,
        runtime_version=baseline.runtime_version,
    )
    candidate_matches_control_runtime = (
        candidate.policy_version == policy.policy_version
        and candidate.image_digest == image
        and candidate.runtime_version == baseline.runtime_version
    )
    if baseline_is_clean_pass:
        failure = "baseline_unexpectedly_passed"
    elif not baseline_is_clean_failure:
        failure = "baseline_invalid_failure"
    elif not reference_is_clean_pass:
        failure = "reference_invalid_evidence"
    elif not candidate.resolved or not candidate_matches_control_runtime:
        failure = candidate.failure_class or "candidate_failed"
    else:
        failure = None
    admitted = failure is None
    if not admitted:
        candidate = dataclasses.replace(candidate, trainable=False)
    return ControlSetEvidence(
        1,
        trajectory_id,
        dataclasses.replace(baseline, trainable=False),
        dataclasses.replace(reference, trainable=False),
        candidate,
        corrupt,
        admitted,
        failure,
    )


_RUN_EVIDENCE_FIELDS: Final = frozenset(
    field.name for field in dataclasses.fields(RunEvidence)
)
_REPLAY_EVIDENCE_FIELDS: Final = frozenset(
    field.name for field in dataclasses.fields(ReplayEvidence)
)
_ATTEMPT_FIELDS: Final = frozenset(
    {
        "schema_version",
        "trajectory_id",
        "status",
        "failure_class",
        "source_content_sha256",
        "fixture_sha256",
        "run_contract_sha256",
        "evidence_sha256",
        "evidence",
        "log",
    }
)
_LOG_FIELDS: Final = frozenset({"path", "sha256", "bytes"})
_ATTEMPT_FAILURE_CODES: Final = frozenset(
    {
        "admission_failed",
        "candidate_not_repeatable",
        "cleanup_failed",
        "invalid_reference_patch",
        "invalid_seed",
        "invalid_source",
        "missing_image_digest",
        "policy_mismatch",
        "protected_path_changed",
        "replay_rejected",
        "unsupported_operation",
        "verifier_failed",
        "verifier_timeout",
    }
)


def _validate_attempt_failure_code(value: object) -> str:
    if (
        type(value) is not str
        or value not in _ATTEMPT_FAILURE_CODES
        or len(value.encode("ascii", errors="ignore")) != len(value)
        or len(value) > 64
    ):
        raise ReplayContractError("attempt failure code is not reviewed")
    return value


def replay_evidence_payload(evidence: ReplayEvidence) -> dict[str, Any]:
    """Return the canonical JSON object for one schema-v2 replay result."""

    if type(evidence) is not ReplayEvidence:
        raise ReplayContractError("replay evidence must use the exact schema-v2 type")
    payload = dataclasses.asdict(evidence)
    return json.loads(_canonical_json(payload))


def _tuple_pairs(value: object, field_name: str) -> tuple[tuple[str, Any], ...]:
    if type(value) is not list:
        raise ReplayContractError(f"{field_name} must be an array")
    pairs: list[tuple[str, Any]] = []
    for item in value:
        if type(item) is not list or len(item) != 2 or type(item[0]) is not str:
            raise ReplayContractError(f"{field_name} contains an invalid pair")
        pairs.append((item[0], item[1]))
    return tuple(pairs)


def _run_evidence_from_payload(payload: object) -> RunEvidence:
    if type(payload) is not dict or set(payload) != _RUN_EVIDENCE_FIELDS:
        raise ReplayContractError("run evidence must have the exact schema")
    values = dict(payload)
    values["protected_before"] = _tuple_pairs(
        values["protected_before"], "protected_before"
    )
    values["protected_after"] = _tuple_pairs(
        values["protected_after"], "protected_after"
    )
    values["resource_peaks"] = _tuple_pairs(
        values["resource_peaks"], "resource_peaks"
    )
    try:
        return RunEvidence(**values)
    except TypeError as exc:  # pragma: no cover - exact fields make this defensive.
        raise ReplayContractError("run evidence cannot be decoded") from exc


def docker_policy_artifact(policy: DockerPolicy) -> dict[str, Any]:
    if type(policy) is not DockerPolicy:
        raise ReplayContractError("policy artifact requires the exact DockerPolicy type")
    policy_payload = json.loads(_canonical_json(dataclasses.asdict(policy)))
    return {
        "schema_version": 1,
        "policy": policy_payload,
        "policy_sha256": _sha256(_canonical_json(policy_payload)),
    }


def _policy_from_artifact(document: object) -> tuple[DockerPolicy, str]:
    if (
        type(document) is not dict
        or set(document) != {"schema_version", "policy", "policy_sha256"}
        or document["schema_version"] != 1
        or type(document["policy"]) is not dict
    ):
        raise ReplayContractError("policy artifact has an invalid exact schema")
    payload = dict(document["policy"])
    if set(payload) != {field.name for field in dataclasses.fields(DockerPolicy)}:
        raise ReplayContractError("policy artifact does not contain the full policy")
    images = payload.get("images")
    if type(images) is not list or any(
        type(item) is not list or len(item) != 2 for item in images
    ):
        raise ReplayContractError("policy artifact image map is invalid")
    payload["images"] = tuple(tuple(item) for item in images)
    try:
        policy = DockerPolicy(**payload)
    except (TypeError, ValueError) as exc:
        raise ReplayContractError("policy artifact values are invalid") from exc
    digest = _sha256(_canonical_json(document["policy"]))
    if document["policy_sha256"] != digest:
        raise ReplayContractError("policy artifact hash mismatch")
    return policy, digest


def _admission_evidence_payload(evidence: AdmissionEvidence) -> dict[str, Any]:
    if type(evidence) is not AdmissionEvidence:
        raise ReplayContractError("admission requires exact AdmissionEvidence")
    return json.loads(_canonical_json(dataclasses.asdict(evidence)))


def admission_artifact(
    policy: DockerPolicy, evidences: Sequence[AdmissionEvidence]
) -> dict[str, Any]:
    policy_document = docker_policy_artifact(policy)
    records = []
    for evidence in sorted(evidences, key=lambda item: item.language):
        payload = _admission_evidence_payload(evidence)
        records.append(
            {
                "evidence": payload,
                "evidence_sha256": _sha256(_canonical_json(payload)),
            }
        )
    body = {"policy_sha256": policy_document["policy_sha256"], "records": records}
    artifact = {
        "schema_version": 1,
        **body,
        "admission_sha256": _sha256(_canonical_json(body)),
    }
    validate_admission_artifact(policy_document, artifact)
    return artifact


def validate_admission_artifact(
    policy_document: object, admission_document: object
) -> dict[str, AdmissionEvidence]:
    policy, policy_sha = _policy_from_artifact(policy_document)
    if (
        type(admission_document) is not dict
        or set(admission_document)
        != {"schema_version", "policy_sha256", "records", "admission_sha256"}
        or admission_document["schema_version"] != 1
        or admission_document["policy_sha256"] != policy_sha
        or type(admission_document["records"]) is not list
    ):
        raise ReplayContractError("admission artifact is not bound to the exact policy")
    body = {
        "policy_sha256": admission_document["policy_sha256"],
        "records": admission_document["records"],
    }
    if admission_document["admission_sha256"] != _sha256(_canonical_json(body)):
        raise ReplayContractError("admission artifact hash mismatch")
    admitted: dict[str, AdmissionEvidence] = {}
    exact_fields = {field.name for field in dataclasses.fields(AdmissionEvidence)}
    for record in admission_document["records"]:
        if type(record) is not dict or set(record) != {"evidence", "evidence_sha256"}:
            raise ReplayContractError("admission record has an invalid exact schema")
        payload = record["evidence"]
        if (
            type(payload) is not dict
            or set(payload) != exact_fields
            or record["evidence_sha256"] != _sha256(_canonical_json(payload))
        ):
            raise ReplayContractError("admission evidence hash mismatch")
        values = dict(payload)
        values["probe_run"] = _run_evidence_from_payload(values["probe_run"])
        try:
            evidence = AdmissionEvidence(**values)
        except TypeError as exc:
            raise ReplayContractError("admission evidence cannot be decoded") from exc
        language = evidence.language
        if language in admitted or language not in _ADMISSION_COMMANDS:
            raise ReplayContractError("admission language is invalid or duplicate")
        image = policy.image_for(language)
        run = evidence.probe_run
        expected_contract = _executor_run_contract_sha256(
            policy,
            image,
            language=language,
            verifier_text=_ADMISSION_COMMANDS[language],
            effective_timeout=60,
            control_identity="admission",
            pre_candidate_tree_sha256=_admission_tree_sha256(),
            pre_candidate_diff_sha256=_sha256(b"admission"),
            protected_before=(),
        )
        valid = (
            evidence.schema_version == 1
            and evidence.admitted is True
            and evidence.failure_class is None
            and evidence.policy_version == policy.policy_version
            and evidence.image_digest == image
            and evidence.policy_output_limit_bytes == policy.output_limit_bytes
            and evidence.runtime_version == run.runtime_version
            and run.pre_candidate_tree_sha256 == _admission_tree_sha256()
            and run.raw_output_bytes <= policy.output_limit_bytes
            and _strict_positive_run_evidence(
                run,
                control_identity="admission",
                trainable=False,
                pre_candidate_tree_sha256=_admission_tree_sha256(),
                pre_candidate_diff_sha256=_sha256(b"admission"),
                protected_before=(),
                policy_version=policy.policy_version,
                image_digest=image,
                run_contract_sha256=expected_contract,
                output_limit_bytes=policy.output_limit_bytes,
                effective_timeout=60,
                runtime_version=evidence.runtime_version,
            )
        )
        if not valid:
            raise ReplayContractError("admission evidence is not a strict clean probe")
        admitted[language] = evidence
    return admitted


def validate_seed_repository(source: GitSeedSource) -> dict[str, Any]:
    """Bind the exact pinned commit and tasks/seeds tree through sanitized Git."""

    try:
        _git_output(source, "cat-file", "-e", f"{source.commit}^{{commit}}")
        commit = _exact_oid(
            _git_output(source, "rev-parse", f"{source.commit}^{{commit}}"),
            "seed commit",
        )
        tree = _exact_oid(
            _git_output(source, "rev-parse", f"{source.commit}:tasks/seeds"),
            "tasks/seeds tree",
        )
    except ReplayContractError as exc:
        raise ReplayContractError("pinned seed repository validation failed") from exc
    if commit != MOONSHINER_REVISION:
        raise ReplayContractError("pinned seed repository commit mismatch")
    body = {"source_commit_sha": commit, "tasks_seed_tree_sha": tree}
    return {**body, "seed_evidence_sha256": _sha256(_canonical_json(body))}


def validate_structural_sidecar(
    source_rows: Iterable[Mapping[str, Any]],
    sidecar_rows: Iterable[Mapping[str, Any]],
) -> tuple[dict[str, Any], ...]:
    """Recompute every sidecar source/trajectory binding against pinned rows."""

    required = {
        "trajectory_id",
        "source_instance_id",
        "source_terminal_sha256",
        "row",
    }
    pending: dict[str, dict[str, Any]] = {}
    for item in sidecar_rows:
        if type(item) is not dict or set(item) != required or type(item["row"]) is not dict:
            raise ReplayContractError("structural sidecar has an invalid exact schema")
        task = item["source_instance_id"]
        row = item["row"]
        if type(task) is not str or row.get("task") != task:
            raise ReplayContractError("structural sidecar task binding mismatch")
        terminal = _canonical_json(row)
        trajectory = trajectory_identity(task, row)
        terminal_sha = _sha256(terminal)
        language = {
            "python": "python",
            "py": "python",
            "rust": "rust",
            "cpp": "cpp",
            "c++": "cpp",
        }.get(row.get("lang"))
        if (
            item["trajectory_id"] != trajectory
            or item["source_terminal_sha256"] != terminal_sha
            or language is None
            or trajectory in pending
        ):
            raise ReplayContractError("structural sidecar identity mismatch")
        pending[trajectory] = {
            "trajectory_id": trajectory,
            "task": task,
            "language": language,
            "source_terminal_sha256": terminal_sha,
            "row": row,
        }
    matched: set[str] = set()
    for row in source_rows:
        if type(row) is not dict or type(row.get("task")) is not str:
            continue
        trajectory = trajectory_identity(row["task"], row)
        candidate = pending.get(trajectory)
        if candidate is not None and _canonical_json(candidate["row"]) == _canonical_json(row):
            matched.add(trajectory)
    if matched != set(pending):
        raise ReplayContractError("structural sidecar contains a row absent from source")
    ordered = sorted(
        pending,
        key=lambda key: (pending[key]["language"], pending[key]["task"], key),
    )
    return tuple(pending[key] for key in ordered)


def _outer_replay_contract(evidence: ReplayEvidence) -> str:
    return _sha256(
        _canonical_json(
            {
                "schema_version": 2,
                "dataset_revision": evidence.dataset_revision,
                "source_commit_sha": evidence.source_commit_sha,
                "source_tree_sha": evidence.source_tree_sha,
                "task_tree_sha": evidence.task_tree_sha,
                "inventory_sha256": evidence.inventory_sha256,
                "trajectory_id": evidence.trajectory_id,
                "source_terminal_sha256": evidence.source_terminal_sha256,
                "operation_sha256": evidence.operation_sha256,
                "candidate_tree_sha256": evidence.candidate_tree_sha256,
                "candidate_diff_sha256": evidence.candidate_diff_sha256,
                "verifier_sha256": evidence.verifier_sha256,
                "source_verify_timeout": evidence.source_verify_timeout,
                "effective_verify_timeout": evidence.effective_verify_timeout,
                "policy_output_limit_bytes": evidence.policy_output_limit_bytes,
                "executor_runs": [run.run_contract_sha256 for run in evidence.runs],
            }
        )
    )


def _validate_positive_candidate_run(run: RunEvidence, evidence: ReplayEvidence) -> None:
    valid = (
        type(run.schema_version) is int
        and run.schema_version == 1
        and type(run.run_id) is str
        and re.fullmatch(r"[0-9a-f]{32}", run.run_id) is not None
        and run.control_identity == "candidate"
        and run.trainable is True
        and type(run.returncode) is int
        and run.returncode == 0
        and type(run.wrapper_returncode) is int
        and run.wrapper_returncode == 0
        and type(run.duration_seconds) in (int, float)
        and not isinstance(run.duration_seconds, bool)
        and math.isfinite(run.duration_seconds)
        and 0 <= run.duration_seconds <= evidence.effective_verify_timeout
        and run.termination == "exited"
        and _is_sha256(run.raw_output_sha256)
        and type(run.raw_output_bytes) is int
        and 0 <= run.raw_output_bytes <= evidence.policy_output_limit_bytes
        and run.output_truncated is False
        and run.pre_candidate_tree_sha256 == evidence.candidate_tree_sha256
        and run.pre_candidate_diff_sha256 == evidence.candidate_diff_sha256
        and _is_sha256(run.post_candidate_tree_sha256)
        and run.protected_before == evidence.protected_sha256
        and run.protected_after == evidence.protected_sha256
        and run.resolved is True
        and run.failure_class is None
        and run.cleanup_state == "verified_removed"
        and run.policy_version == evidence.policy_version
        and run.image_digest == evidence.image_digest
        and run.runtime_version == evidence.runtime_version
        and _is_sha256(run.run_contract_sha256)
        and _has_positive_resource_peaks(run)
    )
    if not valid:
        raise ReplayContractError("replay evidence does not contain two strict candidate runs")


def validate_replay_evidence_payload(payload: object) -> ReplayEvidence:
    """Decode and fully validate the only evidence namespace importers may trust."""

    if type(payload) is not dict or set(payload) != _REPLAY_EVIDENCE_FIELDS:
        raise ReplayContractError("ReplayEvidence must have the exact schema")
    if payload.get("schema_version") != 2:
        raise ReplayContractError("ReplayEvidence schema_version must be 2")
    values = dict(payload)
    values["protected_sha256"] = _tuple_pairs(
        values["protected_sha256"], "protected_sha256"
    )
    raw_runs = values["runs"]
    if type(raw_runs) is not list or len(raw_runs) != 2:
        raise ReplayContractError("ReplayEvidence requires exactly two runs")
    values["runs"] = tuple(_run_evidence_from_payload(run) for run in raw_runs)
    try:
        evidence = ReplayEvidence(**values)
    except TypeError as exc:  # pragma: no cover
        raise ReplayContractError("ReplayEvidence cannot be decoded") from exc
    hashes = (
        evidence.trajectory_id,
        evidence.inventory_sha256,
        evidence.source_terminal_sha256,
        evidence.operation_sha256,
        evidence.candidate_tree_sha256,
        evidence.candidate_diff_sha256,
        evidence.verifier_sha256,
        evidence.run_contract_sha256,
    )
    valid = (
        all(_is_sha256(value) for value in hashes)
        and _TASK_RE.fullmatch(evidence.task) is not None
        and evidence.language in {"python", "rust", "cpp"}
        and evidence.dataset_revision == DATASET_REVISION
        and evidence.source_commit_sha == MOONSHINER_REVISION
        and _OID_RE.fullmatch(evidence.source_tree_sha) is not None
        and _OID_RE.fullmatch(evidence.task_tree_sha) is not None
        and type(evidence.verifier_text) is str
        and _sha256(evidence.verifier_text.encode("utf-8")) == evidence.verifier_sha256
        and (
            evidence.source_verify_timeout is None
            or (
                type(evidence.source_verify_timeout) is int
                and 0 < evidence.source_verify_timeout <= MAX_VERIFY_TIMEOUT
            )
        )
        and type(evidence.effective_verify_timeout) is int
        and 0 < evidence.effective_verify_timeout <= MAX_VERIFY_TIMEOUT
        and type(evidence.policy_version) is str
        and bool(evidence.policy_version)
        and type(evidence.image_digest) is str
        and re.fullmatch(r"[^\s@]+@sha256:[0-9a-f]{64}", evidence.image_digest)
        is not None
        and type(evidence.policy_output_limit_bytes) is int
        and 0 < evidence.policy_output_limit_bytes <= MAX_POLICY_OUTPUT_LIMIT_BYTES
        and _is_runtime_version(evidence.runtime_version)
        and evidence.resolved is True
        and evidence.failure_class is None
        and evidence.control_identity == "candidate"
        and evidence.trainable is True
        and len({run.run_id for run in evidence.runs}) == 2
    )
    if not valid:
        raise ReplayContractError("ReplayEvidence is not in the exact candidate namespace")
    protected_paths = tuple(pair[0] for pair in evidence.protected_sha256)
    if (
        not protected_paths
        or protected_paths != tuple(sorted(set(protected_paths)))
        or any(
            _relative_path(path, field_name="protected path") != path
            or type(digest) is not str
            or not _is_sha256(digest)
            for path, digest in evidence.protected_sha256
        )
    ):
        raise ReplayContractError("ReplayEvidence protected hashes are invalid")
    for run in evidence.runs:
        _validate_positive_candidate_run(run, evidence)
    if _outer_replay_contract(evidence) != evidence.run_contract_sha256:
        raise ReplayContractError("ReplayEvidence run contract mismatch")
    return evidence


def _open_real_input_directory(path: Path) -> int:
    """Open one user-owned, non-writable-by-others directory without symlinks."""

    absolute = Path(os.path.abspath(path))
    parts = absolute.parts
    descriptor = os.open(parts[0], os.O_RDONLY | os.O_DIRECTORY)
    try:
        for component in parts[1:]:
            try:
                child = os.open(
                    component,
                    os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
                    dir_fd=descriptor,
                )
            except OSError as exc:
                raise ReplayContractError(
                    "input parent has a symlink or non-directory component"
                ) from exc
            os.close(descriptor)
            descriptor = child
        status = os.fstat(descriptor)
        if (
            not stat.S_ISDIR(status.st_mode)
            or status.st_uid != os.getuid()
            or stat.S_IMODE(status.st_mode) & 0o022
        ):
            raise ReplayContractError(
                "input parent must be user-owned and not writable by others"
            )
        return descriptor
    except Exception:
        os.close(descriptor)
        raise


def _input_file_snapshot(status: os.stat_result) -> tuple[int, ...]:
    return (
        status.st_dev,
        status.st_ino,
        status.st_nlink,
        status.st_uid,
        stat.S_IMODE(status.st_mode),
        status.st_size,
        status.st_mtime_ns,
        status.st_ctime_ns,
    )


@dataclass
class _BoundInput:
    path: Path
    label: str
    parent_fd: int
    descriptor: int
    parent_identity: tuple[int, int]
    file_snapshot: tuple[int, ...]
    size: int
    max_bytes: int
    _consumed: bool = False
    _sha256: str | None = None

    @classmethod
    def open(
        cls,
        path: Path,
        *,
        label: str,
        max_bytes: int,
        exact_bytes: int | None = None,
    ) -> _BoundInput:
        absolute = Path(os.path.abspath(path))
        if not absolute.name or absolute.name in {".", ".."}:
            raise ReplayContractError(f"{label} input path is not confined")
        parent_fd = _open_real_input_directory(absolute.parent)
        descriptor = -1
        try:
            try:
                named = os.stat(
                    absolute.name,
                    dir_fd=parent_fd,
                    follow_symlinks=False,
                )
                descriptor = os.open(
                    absolute.name,
                    os.O_RDONLY
                    | os.O_NOFOLLOW
                    | getattr(os, "O_CLOEXEC", 0),
                    dir_fd=parent_fd,
                )
            except (FileNotFoundError, OSError) as exc:
                raise ReplayContractError(
                    f"{label} input is missing or unsafe"
                ) from exc
            opened = os.fstat(descriptor)
            mode = stat.S_IMODE(opened.st_mode)
            if (
                not stat.S_ISREG(named.st_mode)
                or not stat.S_ISREG(opened.st_mode)
                or named.st_nlink != 1
                or opened.st_nlink != 1
                or opened.st_uid != os.getuid()
                or (named.st_dev, named.st_ino) != (opened.st_dev, opened.st_ino)
            ):
                raise ReplayContractError(f"{label} input identity is unsafe")
            if not mode & 0o400 or mode & 0o7133:
                raise ReplayContractError(f"{label} input mode is unsafe")
            if type(max_bytes) is not int or max_bytes <= 0:
                raise ReplayContractError(f"{label} input size bound is invalid")
            if opened.st_size > max_bytes:
                raise ReplayContractError(f"{label} input size exceeds its bound")
            if exact_bytes is not None and opened.st_size != exact_bytes:
                raise ReplayContractError(f"{label} input size does not match its pin")
            parent_status = os.fstat(parent_fd)
            return cls(
                absolute,
                label,
                parent_fd,
                descriptor,
                (parent_status.st_dev, parent_status.st_ino),
                _input_file_snapshot(opened),
                opened.st_size,
                max_bytes,
            )
        except Exception:
            if descriptor >= 0:
                os.close(descriptor)
            os.close(parent_fd)
            raise

    def _finish_read(self, digest: Any, byte_count: int) -> None:
        if byte_count != self.size:
            raise ReplayContractError(f"{self.label} input changed while reading")
        if _input_file_snapshot(os.fstat(self.descriptor)) != self.file_snapshot:
            raise ReplayContractError(f"{self.label} input identity changed while reading")
        self._sha256 = digest.hexdigest()

    def read_bytes(self) -> bytes:
        if self._consumed:
            raise ReplayContractError(f"{self.label} input was consumed more than once")
        self._consumed = True
        digest = hashlib.sha256()
        chunks: list[bytes] = []
        byte_count = 0
        while chunk := os.read(self.descriptor, min(1024 * 1024, self.max_bytes + 1)):
            byte_count += len(chunk)
            if byte_count > self.max_bytes:
                raise ReplayContractError(f"{self.label} input size exceeds its bound")
            digest.update(chunk)
            chunks.append(chunk)
        self._finish_read(digest, byte_count)
        return b"".join(chunks)

    def iter_lines(self) -> Iterable[tuple[int, bytes]]:
        if self._consumed:
            raise ReplayContractError(f"{self.label} input was consumed more than once")
        self._consumed = True
        digest = hashlib.sha256()
        buffer = b""
        byte_count = 0
        line_number = 0
        while chunk := os.read(self.descriptor, 1024 * 1024):
            byte_count += len(chunk)
            if byte_count > self.max_bytes:
                raise ReplayContractError(f"{self.label} input size exceeds its bound")
            digest.update(chunk)
            buffer += chunk
            while b"\n" in buffer:
                line, buffer = buffer.split(b"\n", 1)
                line_number += 1
                yield line_number, line
            if len(buffer) > 64 * 1024**2:
                raise ReplayContractError(f"{self.label} input has an oversized line")
        if buffer:
            line_number += 1
            yield line_number, buffer
        self._finish_read(digest, byte_count)

    @property
    def sha256(self) -> str:
        if self._sha256 is None:
            raise ReplayContractError(f"{self.label} input was not fully consumed")
        return self._sha256

    def assert_identity(self) -> None:
        if _input_file_snapshot(os.fstat(self.descriptor)) != self.file_snapshot:
            raise ReplayContractError(f"{self.label} input identity changed")
        try:
            named = os.stat(
                self.path.name,
                dir_fd=self.parent_fd,
                follow_symlinks=False,
            )
        except (FileNotFoundError, OSError) as exc:
            raise ReplayContractError(f"{self.label} input identity changed") from exc
        if _input_file_snapshot(named) != self.file_snapshot:
            raise ReplayContractError(f"{self.label} input identity changed")
        reopened = _open_real_input_directory(self.path.parent)
        try:
            status = os.fstat(reopened)
            if (status.st_dev, status.st_ino) != self.parent_identity:
                raise ReplayContractError(f"{self.label} input identity changed")
        finally:
            os.close(reopened)

    def close(self) -> None:
        os.close(self.descriptor)
        os.close(self.parent_fd)


def _open_real_private_directory(path: Path, *, create: bool = False) -> int:
    """Open a real owner-private directory without following any component."""

    absolute = Path(os.path.abspath(path))
    parts = absolute.parts
    descriptor = os.open(parts[0], os.O_RDONLY | os.O_DIRECTORY)
    try:
        for index, component in enumerate(parts[1:], start=1):
            try:
                child = os.open(
                    component,
                    os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
                    dir_fd=descriptor,
                )
            except FileNotFoundError:
                if not create or index != len(parts) - 1:
                    raise ReplayContractError("confined directory is missing") from None
                os.mkdir(component, mode=0o700, dir_fd=descriptor)
                child = os.open(
                    component,
                    os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
                    dir_fd=descriptor,
                )
            except OSError as exc:
                raise ReplayContractError(
                    "confined directory has a symlink or non-directory component"
                ) from exc
            os.close(descriptor)
            descriptor = child
        status = os.fstat(descriptor)
        if (
            not stat.S_ISDIR(status.st_mode)
            or status.st_uid != os.getuid()
            or stat.S_IMODE(status.st_mode) & 0o077
        ):
            raise ReplayContractError("confined directory must be owner-private")
        return descriptor
    except Exception:
        os.close(descriptor)
        raise


def _open_private_subdirectory(parent_fd: int, name: str) -> int:
    if not name or name in {".", ".."} or "/" in name:
        raise ReplayContractError("log directory must be one confined component")
    try:
        descriptor = os.open(
            name,
            os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
            dir_fd=parent_fd,
        )
    except FileNotFoundError:
        os.mkdir(name, mode=0o700, dir_fd=parent_fd)
        descriptor = os.open(
            name,
            os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
            dir_fd=parent_fd,
        )
    except OSError as exc:
        raise ReplayContractError("log directory is not confined") from exc
    status = os.fstat(descriptor)
    if (
        not stat.S_ISDIR(status.st_mode)
        or status.st_uid != os.getuid()
        or stat.S_IMODE(status.st_mode) & 0o077
    ):
        os.close(descriptor)
        raise ReplayContractError("log directory must be owner-private")
    return descriptor


def _private_directory_identity(descriptor: int, *, label: str) -> tuple[int, int]:
    status = os.fstat(descriptor)
    if (
        not stat.S_ISDIR(status.st_mode)
        or status.st_nlink < 1
        or status.st_uid != os.getuid()
        or stat.S_IMODE(status.st_mode) & 0o077
    ):
        raise ReplayContractError(f"{label} is not an owner-private directory")
    return status.st_dev, status.st_ino


def _assert_path_directory_identity(
    path: Path,
    descriptor: int,
    identity: tuple[int, int],
    *,
    label: str,
) -> None:
    if _private_directory_identity(descriptor, label=label) != identity:
        raise ReplayContractError(f"{label} identity changed")
    reopened = _open_real_private_directory(path)
    try:
        if _private_directory_identity(reopened, label=label) != identity:
            raise ReplayContractError(f"{label} identity changed")
    finally:
        os.close(reopened)


def _assert_named_directory_identity(
    parent_fd: int,
    name: str,
    descriptor: int,
    identity: tuple[int, int],
    *,
    label: str,
) -> None:
    if _private_directory_identity(descriptor, label=label) != identity:
        raise ReplayContractError(f"{label} identity changed")
    try:
        named = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
    except (FileNotFoundError, OSError) as exc:
        raise ReplayContractError(f"{label} identity changed") from exc
    if (
        not stat.S_ISDIR(named.st_mode)
        or named.st_nlink < 1
        or named.st_uid != os.getuid()
        or stat.S_IMODE(named.st_mode) & 0o077
        or (named.st_dev, named.st_ino) != identity
    ):
        raise ReplayContractError(f"{label} identity changed")


def _read_bound_file_at(directory_fd: int, name: str, *, mode: int = 0o600) -> bytes:
    try:
        before = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
        descriptor = os.open(name, os.O_RDONLY | os.O_NOFOLLOW, dir_fd=directory_fd)
    except (FileNotFoundError, OSError) as exc:
        raise ReplayContractError("bound publication file is missing or unsafe") from exc
    try:
        opened = os.fstat(descriptor)
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_nlink != 1
            or stat.S_IMODE(before.st_mode) != mode
            or (opened.st_dev, opened.st_ino) != (before.st_dev, before.st_ino)
        ):
            raise ReplayContractError("bound publication identity is invalid")
        with os.fdopen(descriptor, "rb", closefd=True) as handle:
            descriptor = -1
            return handle.read()
    finally:
        if descriptor >= 0:
            os.close(descriptor)


def _atomic_write_0600_at(
    directory_fd: int,
    name: str,
    data: bytes,
    *,
    before_rename: Callable[[], None] | None = None,
    after_rename: Callable[[int, str], None] | None = None,
) -> None:
    if not name or name in {".", ".."} or "/" in name:
        raise ReplayContractError("publication name is not confined")
    temporary = f".{name}.{secrets.token_hex(16)}.tmp"
    descriptor = os.open(
        temporary,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
        0o600,
        dir_fd=directory_fd,
    )
    published = False
    try:
        os.fchmod(descriptor, 0o600)
        with os.fdopen(descriptor, "wb", closefd=True) as handle:
            descriptor = -1
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
            temporary_identity = os.fstat(handle.fileno())
        if before_rename is not None:
            before_rename()
        os.replace(
            temporary,
            name,
            src_dir_fd=directory_fd,
            dst_dir_fd=directory_fd,
        )
        published = True
        if after_rename is not None:
            after_rename(directory_fd, name)
        try:
            published_bytes = _read_bound_file_at(directory_fd, name)
        except ReplayContractError as exc:
            raise ReplayContractError(
                "publication identity changed after rename"
            ) from exc
        published_identity = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
        if (
            (published_identity.st_dev, published_identity.st_ino)
            != (temporary_identity.st_dev, temporary_identity.st_ino)
            or published_bytes != data
        ):
            raise ReplayContractError("publication identity changed after rename")
        os.fsync(directory_fd)
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        if not published:
            try:
                os.unlink(temporary, dir_fd=directory_fd)
            except FileNotFoundError:
                pass


def _atomic_write_0600(
    path: Path,
    data: bytes,
    *,
    after_rename: Callable[[int, str], None] | None = None,
) -> None:
    directory_fd = _open_real_private_directory(path.parent, create=True)
    identity = _private_directory_identity(
        directory_fd, label="publication parent directory"
    )

    def assert_parent_identity() -> None:
        _assert_path_directory_identity(
            path.parent,
            directory_fd,
            identity,
            label="publication parent directory",
        )

    def after_publish(bound_fd: int, name: str) -> None:
        if after_rename is not None:
            after_rename(bound_fd, name)
        assert_parent_identity()

    try:
        _atomic_write_0600_at(
            directory_fd,
            path.name,
            data,
            before_rename=assert_parent_identity,
            after_rename=after_publish,
        )
        assert_parent_identity()
    finally:
        os.close(directory_fd)


class ReplayLedger:
    """Exclusive, fully validated, atomically published replay attempt ledger."""

    def __init__(self, path: Path, log_dir: Path) -> None:
        self.path = Path(path)
        self.log_dir = Path(log_dir)
        self._parent_fd: int | None = None
        self._log_fd: int | None = None
        self._parent_identity: tuple[int, int] | None = None
        self._log_identity: tuple[int, int] | None = None
        self._lock_fd: int | None = None
        self._records: list[dict[str, Any]] = []

    def __enter__(self) -> ReplayLedger:
        if Path(os.path.abspath(self.log_dir.parent)) != Path(
            os.path.abspath(self.path.parent)
        ):
            raise ReplayContractError("ledger and logs must share one confined parent")
        self._parent_fd = _open_real_private_directory(self.path.parent, create=True)
        parent_status = os.fstat(self._parent_fd)
        self._parent_identity = (parent_status.st_dev, parent_status.st_ino)
        try:
            fcntl.flock(self._parent_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            os.close(self._parent_fd)
            self._parent_fd = None
            raise ReplayContractError("replay ledger parent is already locked") from exc
        try:
            self._log_fd = _open_private_subdirectory(
                self._parent_fd, self.log_dir.name
            )
            self._log_identity = _private_directory_identity(
                self._log_fd, label="replay log directory"
            )
        except Exception:
            self.__exit__(None, None, None)
            raise
        lock_name = f".{self.path.name}.lock"
        try:
            self._lock_fd = os.open(
                lock_name,
                os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW,
                0o600,
                dir_fd=self._parent_fd,
            )
        except OSError as exc:
            self.__exit__(None, None, None)
            raise ReplayContractError("replay lock is not a confined regular file") from exc
        lock_stat = os.fstat(self._lock_fd)
        if (
            not stat.S_ISREG(lock_stat.st_mode)
            or lock_stat.st_nlink != 1
            or lock_stat.st_dev != parent_status.st_dev
        ):
            self.__exit__(None, None, None)
            raise ReplayContractError("replay lock is not a same-filesystem single-link file")
        os.fchmod(self._lock_fd, 0o600)
        try:
            fcntl.flock(self._lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            self.__exit__(None, None, None)
            raise ReplayContractError("replay ledger is already locked") from exc
        try:
            self._assert_lock_identity()
            self._records = self._load_and_validate()
        except Exception:
            self.__exit__(None, None, None)
            raise
        return self

    def __exit__(self, *_exc: object) -> None:
        if self._lock_fd is not None:
            fcntl.flock(self._lock_fd, fcntl.LOCK_UN)
            os.close(self._lock_fd)
            self._lock_fd = None
        if self._log_fd is not None:
            os.close(self._log_fd)
            self._log_fd = None
        self._log_identity = None
        if self._parent_fd is not None:
            fcntl.flock(self._parent_fd, fcntl.LOCK_UN)
            os.close(self._parent_fd)
            self._parent_fd = None

    @property
    def completed_trajectory_ids(self) -> frozenset[str]:
        return frozenset(record["trajectory_id"] for record in self._records)

    @property
    def records(self) -> tuple[Mapping[str, Any], ...]:
        return tuple(self._records)

    def _require_locked(self) -> None:
        if self._lock_fd is None or self._parent_fd is None or self._log_fd is None:
            raise ReplayContractError("replay ledger is not locked")

    def _assert_lock_identity(self) -> None:
        self._require_locked()
        assert self._parent_fd is not None
        assert self._lock_fd is not None
        assert self._log_fd is not None
        assert self._log_identity is not None
        try:
            named = os.stat(
                f".{self.path.name}.lock",
                dir_fd=self._parent_fd,
                follow_symlinks=False,
            )
        except FileNotFoundError as exc:
            raise ReplayContractError("replay lock identity is missing") from exc
        held = os.fstat(self._lock_fd)
        if (
            not stat.S_ISREG(named.st_mode)
            or named.st_nlink != 1
            or held.st_nlink != 1
            or (named.st_dev, named.st_ino) != (held.st_dev, held.st_ino)
        ):
            raise ReplayContractError("replay lock identity changed")
        assert self._parent_identity is not None
        _assert_path_directory_identity(
            self.path.parent,
            self._parent_fd,
            self._parent_identity,
            label="replay parent directory",
        )
        _assert_named_directory_identity(
            self._parent_fd,
            self.log_dir.name,
            self._log_fd,
            self._log_identity,
            label="replay log directory",
        )

    def _validate_attempt(self, payload: object, line_number: int) -> dict[str, Any]:
        if type(payload) is not dict or set(payload) != _ATTEMPT_FIELDS:
            raise ReplayContractError(
                f"replay ledger line {line_number} is not a schema-v2 attempt ledger record"
            )
        if payload["schema_version"] != 2:
            raise ReplayContractError(
                f"replay ledger line {line_number} schema_version must be 2"
            )
        trajectory_id = payload["trajectory_id"]
        if not _is_sha256(trajectory_id):
            raise ReplayContractError(f"replay ledger line {line_number} has invalid trajectory")
        status = payload["status"]
        if status not in {"verified", "rejected", "timeout"}:
            raise ReplayContractError(f"replay ledger line {line_number} has invalid status")
        log = payload["log"]
        if type(log) is not dict or set(log) != _LOG_FIELDS:
            raise ReplayContractError(f"replay ledger line {line_number} has invalid log binding")
        if type(log["path"]) is not str or log["path"] != f"{trajectory_id}.log":
            raise ReplayContractError(f"replay ledger line {line_number} has invalid log path")
        if type(log["sha256"]) is not str or not _is_sha256(log["sha256"]):
            raise ReplayContractError(f"replay ledger line {line_number} has invalid log hash")
        if type(log["bytes"]) is not int or log["bytes"] < 0:
            raise ReplayContractError(f"replay ledger line {line_number} has invalid log size")
        try:
            assert self._log_fd is not None
            log_bytes = _read_bound_file_at(self._log_fd, log["path"])
        except ReplayContractError as exc:
            raise ReplayContractError(
                f"replay ledger line {line_number} log is missing or unsafe"
            ) from exc
        if (
            log["bytes"] != len(log_bytes)
            or log["sha256"] != _sha256(log_bytes)
        ):
            raise ReplayContractError(f"replay ledger line {line_number} log hash mismatch")
        if status == "verified":
            evidence = validate_replay_evidence_payload(payload["evidence"])
            evidence_payload = replay_evidence_payload(evidence)
            evidence_sha = _sha256(_canonical_json(evidence_payload))
            if (
                payload["failure_class"] is not None
                or payload["source_content_sha256"] is None
                or not _is_sha256(payload["source_content_sha256"])
                or payload["fixture_sha256"] != evidence.inventory_sha256
                or payload["run_contract_sha256"] != evidence.run_contract_sha256
                or payload["evidence_sha256"] != evidence_sha
            ):
                raise ReplayContractError(
                    f"replay ledger line {line_number} verified binding mismatch"
                )
        else:
            if (
                any(
                    payload[field] is not None
                    for field in (
                        "source_content_sha256",
                        "fixture_sha256",
                        "run_contract_sha256",
                        "evidence_sha256",
                        "evidence",
                    )
                )
            ):
                raise ReplayContractError(
                    f"replay ledger line {line_number} failure binding mismatch"
                )
            _validate_attempt_failure_code(payload["failure_class"])
        return dict(payload)

    def _load_and_validate(self) -> list[dict[str, Any]]:
        assert self._parent_fd is not None
        try:
            ledger_stat = os.stat(
                self.path.name,
                dir_fd=self._parent_fd,
                follow_symlinks=False,
            )
        except FileNotFoundError:
            return []
        if (
            not stat.S_ISREG(ledger_stat.st_mode)
            or ledger_stat.st_nlink != 1
            or ledger_stat.st_dev != os.fstat(self._parent_fd).st_dev
        ):
            raise ReplayContractError("replay ledger is not a confined single-link file")
        if stat.S_IMODE(ledger_stat.st_mode) != 0o600:
            raise ReplayContractError("replay ledger mode is not 0600")
        ledger_fd = os.open(
            self.path.name,
            os.O_RDONLY | os.O_NOFOLLOW,
            dir_fd=self._parent_fd,
        )
        try:
            opened = os.fstat(ledger_fd)
            if (opened.st_dev, opened.st_ino) != (
                ledger_stat.st_dev,
                ledger_stat.st_ino,
            ):
                raise ReplayContractError("replay ledger identity changed while opening")
            with os.fdopen(ledger_fd, "r", encoding="utf-8", closefd=True) as handle:
                ledger_fd = -1
                ledger_lines = handle.read().splitlines()
        finally:
            if ledger_fd >= 0:
                os.close(ledger_fd)
        records: list[dict[str, Any]] = []
        trajectories: set[str] = set()
        contracts: set[str] = set()
        for line_number, line in enumerate(ledger_lines, start=1):
            if not line:
                raise ReplayContractError(f"replay ledger line {line_number} is empty")
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ReplayContractError(
                    f"replay ledger line {line_number} is invalid JSON"
                ) from exc
            record = self._validate_attempt(payload, line_number)
            trajectory = record["trajectory_id"]
            if trajectory in trajectories:
                raise ReplayContractError(f"duplicate trajectory {trajectory}")
            trajectories.add(trajectory)
            contract = record["run_contract_sha256"]
            if contract is not None:
                if contract in contracts:
                    raise ReplayContractError("duplicate run contract")
                contracts.add(contract)
            records.append(record)
        return records

    def _publish(
        self,
        record: dict[str, Any],
        log: bytes,
        callback: Callable[[Path], None] | None,
    ) -> None:
        self._require_locked()
        self._assert_lock_identity()
        trajectory = record["trajectory_id"]
        if trajectory in self.completed_trajectory_ids:
            raise ReplayContractError(f"duplicate trajectory {trajectory}")
        contract = record["run_contract_sha256"]
        if contract is not None and any(
            existing["run_contract_sha256"] == contract for existing in self._records
        ):
            raise ReplayContractError("duplicate run contract")
        log_name = f"{trajectory}.log"
        assert self._log_fd is not None
        _atomic_write_0600_at(
            self._log_fd,
            log_name,
            log,
            before_rename=self._assert_lock_identity,
        )
        if callback is not None:
            callback(self.log_dir / log_name)
        self._assert_lock_identity()
        record["log"] = {
            "path": log_name,
            "sha256": _sha256(log),
            "bytes": len(log),
        }
        validated = self._validate_attempt(record, len(self._records) + 1)
        output = b"".join(
            _canonical_json(item) + b"\n" for item in (*self._records, validated)
        )
        assert self._parent_fd is not None
        _atomic_write_0600_at(
            self._parent_fd,
            self.path.name,
            output,
            before_rename=self._assert_lock_identity,
        )
        self._assert_lock_identity()
        self._records.append(validated)

    def publish_verified(
        self,
        evidence: ReplayEvidence,
        *,
        source_content_sha256: str,
        fixture_sha256: str,
        log: bytes,
        after_log_publish: Callable[[Path], None] | None = None,
    ) -> None:
        evidence_payload = replay_evidence_payload(evidence)
        validated = validate_replay_evidence_payload(evidence_payload)
        if not _is_sha256(source_content_sha256):
            raise ReplayContractError("source content hash is invalid")
        if fixture_sha256 != validated.inventory_sha256:
            raise ReplayContractError("fixture binding mismatch")
        self._publish(
            {
                "schema_version": 2,
                "trajectory_id": validated.trajectory_id,
                "status": "verified",
                "failure_class": None,
                "source_content_sha256": source_content_sha256,
                "fixture_sha256": fixture_sha256,
                "run_contract_sha256": validated.run_contract_sha256,
                "evidence_sha256": _sha256(_canonical_json(evidence_payload)),
                "evidence": evidence_payload,
                "log": {},
            },
            log,
            after_log_publish,
        )

    def publish_failure(
        self,
        *,
        trajectory_id: str,
        status: str,
        failure_class: str,
        log: bytes,
    ) -> None:
        if status not in {"rejected", "timeout"}:
            raise ReplayContractError("failure status must be rejected or timeout")
        if not _is_sha256(trajectory_id):
            raise ReplayContractError("trajectory ID is invalid")
        _validate_attempt_failure_code(failure_class)
        self._publish(
            {
                "schema_version": 2,
                "trajectory_id": trajectory_id,
                "status": status,
                "failure_class": failure_class,
                "source_content_sha256": None,
                "fixture_sha256": None,
                "run_contract_sha256": None,
                "evidence_sha256": None,
                "evidence": None,
                "log": {},
            },
            log,
            None,
        )


_ELIGIBILITY_GATES: Final = (
    ("operations_supported", "unsupported_operations"),
    ("decontaminated", "contamination"),
    ("git_seed_valid", "invalid_git_seed"),
    ("reference_patch_valid", "invalid_reference_patch"),
    ("language_digest_present", "missing_language_digest"),
    ("admission_valid", "functional_admission_failed"),
)


def build_eligibility_inventory(
    rows: Sequence[EligibilityCandidate],
    bindings: EligibilityBindings,
) -> dict[str, Any]:
    """Compute a deterministic, exclusive gate partition without discovery."""

    if type(bindings) is not EligibilityBindings:
        raise ReplayContractError("eligibility requires exact artifact bindings")
    for field_name in (
        "source_sha256",
        "sidecar_sha256",
        "policy_sha256",
        "admission_sha256",
        "policy_artifact_sha256",
        "admission_artifact_sha256",
        "smoke_manifest_sha256",
        "seed_evidence_sha256",
    ):
        if not _is_sha256(getattr(bindings, field_name)):
            raise ReplayContractError(f"eligibility {field_name} is invalid")
    if (
        bindings.seed_commit_sha != MOONSHINER_REVISION
        or _OID_RE.fullmatch(bindings.seed_tree_sha) is None
    ):
        raise ReplayContractError("eligibility seed binding is invalid")
    exclusion_artifacts = bindings.exclusion_artifacts
    if (
        type(exclusion_artifacts) is not tuple
        or not exclusion_artifacts
        or any(
            type(item) is not tuple
            or len(item) != 2
            or type(item[0]) is not str
            or _TASK_RE.fullmatch(item[0]) is None
            or type(item[1]) is not str
            or not _is_sha256(item[1])
            for item in exclusion_artifacts
        )
        or exclusion_artifacts
        != tuple(sorted(exclusion_artifacts, key=lambda item: item[0].encode("utf-8")))
        or len({label for label, _digest in exclusion_artifacts})
        != len(exclusion_artifacts)
    ):
        raise ReplayContractError("eligibility exclusion artifact binding is invalid")
    seen: set[str] = set()
    exclusions = {reason: 0 for _, reason in _ELIGIBILITY_GATES}
    eligible: list[dict[str, str]] = []
    gate_pass = {field: 0 for field, _ in _ELIGIBILITY_GATES}
    for row in rows:
        if type(row) is not EligibilityCandidate:
            raise ReplayContractError("eligibility rows must use the exact candidate type")
        if not _is_sha256(row.trajectory_id) or row.trajectory_id in seen:
            raise ReplayContractError("eligibility trajectory identity is invalid or duplicate")
        if not _TASK_RE.fullmatch(row.task) or row.language not in {"python", "rust", "cpp"}:
            raise ReplayContractError("eligibility task or language is invalid")
        seen.add(row.trajectory_id)
        dropped = False
        for field, reason in _ELIGIBILITY_GATES:
            value = getattr(row, field)
            if type(value) is not bool:
                raise ReplayContractError(f"eligibility gate {field} must be boolean")
            if not value:
                exclusions[reason] += 1
                dropped = True
                break
            gate_pass[field] += 1
        if not dropped:
            eligible.append(
                {
                    "trajectory_id": row.trajectory_id,
                    "task": row.task,
                    "language": row.language,
                }
            )
    eligible.sort(key=lambda row: (row["language"], row["task"], row["trajectory_id"]))
    total = len(rows)
    if total != len(eligible) + sum(exclusions.values()):
        raise ReplayContractError("eligibility manifest arithmetic mismatch")
    manifest = {
        "schema_version": 1,
        "bindings": dataclasses.asdict(bindings),
        "gate_order": [field for field, _ in _ELIGIBILITY_GATES],
        "total": total,
        "gate_pass": gate_pass,
        "exclusions": exclusions,
        "eligible_ceiling": len(eligible),
        "eligible_by_language": dict(
            sorted(Counter(row["language"] for row in eligible).items())
        ),
        "eligible": eligible,
    }
    manifest["inventory_sha256"] = _sha256(_canonical_json(manifest))
    return manifest


def build_smoke_manifest(
    inventory: Mapping[str, Any], trajectory_ids: Sequence[str]
) -> dict[str, Any]:
    """Validate an explicit smoke set; never select candidates implicitly."""

    eligible = inventory.get("eligible")
    inventory_sha = inventory.get("inventory_sha256")
    if type(inventory_sha) is not str or not _is_sha256(inventory_sha):
        raise ReplayContractError("eligibility inventory hash is missing")
    hash_input = dict(inventory)
    del hash_input["inventory_sha256"]
    if _sha256(_canonical_json(hash_input)) != inventory_sha:
        raise ReplayContractError("eligibility inventory hash mismatch")
    if type(eligible) is not list:
        raise ReplayContractError("eligibility inventory has no explicit candidates")
    by_id = {
        row["trajectory_id"]: row
        for row in eligible
        if type(row) is dict and type(row.get("trajectory_id")) is str
    }
    if len(trajectory_ids) != 5 or len(set(trajectory_ids)) != 5:
        raise ReplayContractError("smoke requires exactly 2 Python, 2 Rust, and 1 C++")
    try:
        selected = [by_id[trajectory_id] for trajectory_id in trajectory_ids]
    except KeyError as exc:
        raise ReplayContractError("smoke trajectory is outside the eligible intersection") from exc
    counts = Counter(row["language"] for row in selected)
    if counts != Counter({"python": 2, "rust": 2, "cpp": 1}):
        raise ReplayContractError("smoke requires exactly 2 Python, 2 Rust, and 1 C++")
    order = {"python": 0, "rust": 1, "cpp": 2}
    selected = sorted(
        selected,
        key=lambda row: (order[row["language"]], row["task"], row["trajectory_id"]),
    )
    return {
        "schema_version": 1,
        "counts": {"python": 2, "rust": 2, "cpp": 1},
        "eligibility_sha256": inventory_sha,
        "trajectories": selected,
        "selection_sha256": _sha256(_canonical_json(selected)),
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--sidecar", type=Path, required=True)
    parser.add_argument("--seed-repo", type=Path, required=True)
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--admission", type=Path, required=True)
    parser.add_argument("--smoke-manifest", type=Path, required=True)
    parser.add_argument("--exclusion", type=Path, action="append", required=True)
    parser.add_argument("--ledger", type=Path, required=True)
    parser.add_argument("--logs", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--inventory-only", action="store_true")
    return parser.parse_args(argv)


_MAX_JSON_INPUT_BYTES: Final = 64 * 1024**2
_MAX_EXCLUSION_INPUT_BYTES: Final = 256 * 1024**2
_MAX_SIDECAR_INPUT_BYTES: Final = 2 * 1024**3


def _bound_json_document(bound: _BoundInput) -> Any:
    try:
        return json.loads(bound.read_bytes())
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ReplayContractError(f"{bound.label} input is invalid JSON") from exc


def _bound_jsonl_rows(bound: _BoundInput) -> Iterable[Mapping[str, Any]]:
    for line_number, line in bound.iter_lines():
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ReplayContractError(
                f"{bound.label} input line {line_number} is invalid JSON"
            ) from exc
        if type(value) is not dict:
            raise ReplayContractError(
                f"{bound.label} input line {line_number} is not an object"
            )
        yield value


def _load_bound_exclusion_inputs(
    inputs: Sequence[tuple[str, _BoundInput]],
) -> tuple[tuple[Any, ...], tuple[tuple[str, str], ...]]:
    loaded: list[tuple[str, str, Any]] = []
    seen: set[str] = set()
    for label, bound in inputs:
        if not label or _TASK_RE.fullmatch(label) is None or label in seen:
            raise ReplayContractError(
                "exclusion artifacts require unique stable filename labels"
            )
        seen.add(label)
        try:
            raw = bound.read_bytes()
            record = parse_exclusion_artifact(raw)
        except (ValueError, json.JSONDecodeError) as exc:
            raise ReplayContractError(
                f"exclusion artifact {label!r} is invalid"
            ) from exc
        loaded.append((label, bound.sha256, record))
    loaded.sort(key=lambda item: item[0].encode("utf-8"))
    return (
        tuple(item[2] for item in loaded),
        tuple((item[0], item[1]) for item in loaded),
    )


def main(argv: Sequence[str] | None = None) -> int:
    """Publish only an explicit precomputed inventory in this code-only phase."""

    args = parse_args(argv)
    if not args.inventory_only:
        raise ReplayContractError("live replay is disabled; use --inventory-only")
    bound_inputs: list[_BoundInput] = []

    def bind(
        path: Path,
        *,
        label: str,
        max_bytes: int,
        exact_bytes: int | None = None,
    ) -> _BoundInput:
        bound = _BoundInput.open(
            path,
            label=label,
            max_bytes=max_bytes,
            exact_bytes=exact_bytes,
        )
        bound_inputs.append(bound)
        return bound

    try:
        source_input = bind(
            args.source,
            label="source",
            max_bytes=SOURCE_BYTES,
            exact_bytes=SOURCE_BYTES,
        )
        sidecar_input = bind(
            args.sidecar,
            label="sidecar",
            max_bytes=_MAX_SIDECAR_INPUT_BYTES,
        )
        policy_input = bind(
            args.policy,
            label="policy",
            max_bytes=_MAX_JSON_INPUT_BYTES,
        )
        admission_input = bind(
            args.admission,
            label="admission",
            max_bytes=_MAX_JSON_INPUT_BYTES,
        )
        smoke_input = bind(
            args.smoke_manifest,
            label="smoke manifest",
            max_bytes=_MAX_JSON_INPUT_BYTES,
        )
        exclusion_inputs: list[tuple[str, _BoundInput]] = []
        exclusion_labels: set[str] = set()
        for path in args.exclusion:
            label = Path(path).name
            if (
                not label
                or _TASK_RE.fullmatch(label) is None
                or label in exclusion_labels
            ):
                raise ReplayContractError(
                    "exclusion artifacts require unique stable filename labels"
                )
            exclusion_labels.add(label)
            exclusion_inputs.append(
                (
                    label,
                    bind(
                        path,
                        label=f"exclusion {label}",
                        max_bytes=_MAX_EXCLUSION_INPUT_BYTES,
                    ),
                )
            )

        policy_document = _bound_json_document(policy_input)
        admission_document = _bound_json_document(admission_input)
        smoke_payload = _bound_json_document(smoke_input)
        exclusions, exclusion_artifacts = _load_bound_exclusion_inputs(
            exclusion_inputs
        )
        policy, policy_sha = _policy_from_artifact(policy_document)
        admitted = validate_admission_artifact(
            policy_document, admission_document
        )
        sidecar_rows = list(_bound_jsonl_rows(sidecar_input))
        structural = validate_structural_sidecar(
            _bound_jsonl_rows(source_input), sidecar_rows
        )
        if source_input.sha256 != SOURCE_LFS_SHA256:
            raise ReplayContractError(
                "source input hash does not match its pin"
            )

        seed_source = GitSeedSource(args.seed_repo, MOONSHINER_REVISION)
        seed_evidence = validate_seed_repository(seed_source)
        candidates: list[EligibilityCandidate] = []
        with tempfile.TemporaryDirectory(prefix="fable-inventory-") as temporary:
            workspace = Path(temporary)
            for index, record in enumerate(structural):
                language = record["language"]
                selected = None
                operations_supported = True
                try:
                    selected = select_terminal_row(record["row"])
                except Exception:
                    operations_supported = False
                try:
                    contract = materialize_seed(
                        seed_source,
                        record["task"],
                        workspace / f"seed-{index}",
                    )
                except ReplayContractError:
                    candidates.append(
                        EligibilityCandidate(
                            trajectory_id=record["trajectory_id"],
                            task=record["task"],
                            language=language,
                            operations_supported=operations_supported,
                            decontaminated=True,
                            git_seed_valid=False,
                            reference_patch_valid=False,
                            language_digest_present=language in dict(policy.images),
                            admission_valid=language in admitted,
                        )
                    )
                    continue
                if (
                    selected is None
                    or selected.language != language
                    or contract.language != language
                ):
                    candidates.append(
                        EligibilityCandidate(
                            trajectory_id=record["trajectory_id"],
                            task=record["task"],
                            language=language,
                            operations_supported=operations_supported,
                            decontaminated=True,
                            git_seed_valid=False,
                            reference_patch_valid=False,
                            language_digest_present=language in dict(policy.images),
                            admission_valid=language in admitted,
                        )
                    )
                    continue
                try:
                    plan = canonical_mutation_plan(
                        selected.messages,
                        frozenset(contract.protected_paths),
                        contract.verify_cmd,
                    )
                    operations_supported = bool(plan.operations)
                except Exception:
                    operations_supported = False
                reference_ok = preflight_reference_patch(contract).eligible
                contaminated = None
                try:
                    assessment = assess_converted_trajectory(
                        selected,
                        protected_paths=contract.protected_paths,
                        verify_cmd=contract.verify_cmd,
                        exclusions=exclusions,
                    )
                    contaminated = assessment.decontamination_reason
                except Exception:
                    operations_supported = False
                candidates.append(
                    EligibilityCandidate(
                        trajectory_id=record["trajectory_id"],
                        task=record["task"],
                        language=language,
                        operations_supported=operations_supported,
                        decontaminated=contaminated is None,
                        git_seed_valid=True,
                        reference_patch_valid=reference_ok,
                        language_digest_present=language in dict(policy.images),
                        admission_valid=language in admitted,
                    )
                )

        bindings = EligibilityBindings(
            source_sha256=source_input.sha256,
            sidecar_sha256=sidecar_input.sha256,
            policy_sha256=policy_sha,
            admission_sha256=admission_document["admission_sha256"],
            policy_artifact_sha256=policy_input.sha256,
            admission_artifact_sha256=admission_input.sha256,
            smoke_manifest_sha256=smoke_input.sha256,
            exclusion_artifacts=exclusion_artifacts,
            seed_evidence_sha256=seed_evidence["seed_evidence_sha256"],
            seed_commit_sha=seed_evidence["source_commit_sha"],
            seed_tree_sha=seed_evidence["tasks_seed_tree_sha"],
        )
        inventory = build_eligibility_inventory(candidates, bindings)
        if type(smoke_payload) is not list or any(
            type(value) is not str for value in smoke_payload
        ):
            raise ReplayContractError(
                "smoke manifest input must be an explicit trajectory ID list"
            )
        smoke = build_smoke_manifest(inventory, smoke_payload)

        def assert_input_identities() -> None:
            for bound in bound_inputs:
                bound.assert_identity()

        assert_input_identities()
        output_fd = _open_real_private_directory(args.out, create=True)
        output_identity = _private_directory_identity(
            output_fd, label="output directory"
        )

        def assert_publication_boundaries() -> None:
            assert_input_identities()
            _assert_path_directory_identity(
                args.out,
                output_fd,
                output_identity,
                label="output directory",
            )

        try:
            for name, payload in (
                ("eligibility.json", inventory),
                ("smoke.json", smoke),
            ):
                _atomic_write_0600_at(
                    output_fd,
                    name,
                    _canonical_json(payload) + b"\n",
                    before_rename=assert_publication_boundaries,
                    after_rename=lambda _fd, _name: assert_publication_boundaries(),
                )
                assert_publication_boundaries()
        finally:
            os.close(output_fd)
        return 0
    finally:
        for bound in reversed(bound_inputs):
            bound.close()


__all__ = [
    "AdmissionEvidence",
    "CandidateState",
    "ControlSetEvidence",
    "DockerExecutor",
    "DockerPolicy",
    "EligibilityBindings",
    "EligibilityCandidate",
    "GitSeedSource",
    "MOONSHINER_REVISION",
    "MutationPlan",
    "ReferencePatchContract",
    "ReplayContractError",
    "ReplayEvidence",
    "ReplayLedger",
    "RestrictedExecutor",
    "RunEvidence",
    "SeedContract",
    "canonical_mutation_plan",
    "build_eligibility_inventory",
    "build_smoke_manifest",
    "admission_artifact",
    "docker_policy_artifact",
    "functional_admission_probe",
    "materialize_seed",
    "preflight_reference_patch",
    "reconstruct_candidate",
    "run_control_set",
    "replay_evidence_payload",
    "trajectory_identity",
    "verify_candidate",
    "validate_admission_artifact",
    "validate_replay_evidence_payload",
    "validate_seed_repository",
    "validate_structural_sidecar",
]


if __name__ == "__main__":
    raise SystemExit(main())
