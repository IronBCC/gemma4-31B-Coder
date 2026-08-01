"""Source-agnostic trusted mutation planning for pinned trace datasets.

The boundary in this module is intentionally narrow:

* source identity must exactly match an immutable, revision-pinned contract;
* source-specific adapters may return only typed declarative mutations;
* transcript shell/JavaScript is data and is never executed;
* static ``apply_patch`` payloads are parsed into typed Write/Edit operations.

Execution remains in :mod:`teacher_platform.fable5_replay`, which reconstructs
each candidate in a fresh pinned seed and runs the baseline/reference/candidate
control set twice.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import re
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import Final, Protocol

from .fable5_import import FableEditOp, FableWriteOp
from .fable5_replay import (
    MutationPlan,
    ReplayContractError,
    canonical_mutation_plan,
)

_HEX_40_OR_64: Final = re.compile(r"[0-9a-f]{40}(?:[0-9a-f]{24})?\Z")
_DATASET_ID: Final = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*/[A-Za-z0-9][A-Za-z0-9._-]*\Z")


@dataclass(frozen=True)
class PinnedSourceContract:
    """Immutable identity of one reviewed Hugging Face source artifact."""

    dataset_id: str
    revision: str
    source_sha256: str
    source_bytes: int

    def __post_init__(self) -> None:
        if _DATASET_ID.fullmatch(self.dataset_id) is None:
            raise ValueError("dataset_id must be an exact owner/name identifier")
        if _HEX_40_OR_64.fullmatch(self.revision) is None:
            raise ValueError("revision must be a lowercase 40- or 64-hex object ID")
        if re.fullmatch(r"[0-9a-f]{64}", self.source_sha256) is None:
            raise ValueError("source_sha256 must be lowercase 64-hex")
        if type(self.source_bytes) is not int or self.source_bytes <= 0:
            raise ValueError("source_bytes must be a positive integer")


@dataclass(frozen=True)
class SourceArtifactIdentity:
    """Observed source identity presented at the trusted replay boundary."""

    dataset_id: str
    revision: str
    source_sha256: str
    source_bytes: int


@dataclass(frozen=True)
class WriteMutation:
    """A repository-relative complete file write."""

    path: str
    content: str


@dataclass(frozen=True)
class EditMutation:
    """A repository-relative exact string replacement."""

    path: str
    old_string: str
    new_string: str
    replace_all: bool = False


@dataclass(frozen=True)
class StaticApplyPatch:
    """A literal OpenAI ``apply_patch`` payload; never a command to execute."""

    patch: str


TypedMutation = (
    WriteMutation
    | EditMutation
    | StaticApplyPatch
    | FableWriteOp
    | FableEditOp
)


@dataclass(frozen=True)
class MutationPlanInput:
    """Source adapter output before canonical protected-path binding."""

    operations: tuple[object, ...]
    verifier_evidence_count: int = 0

    def __init__(
        self,
        operations: Sequence[object],
        verifier_evidence_count: int = 0,
    ) -> None:
        object.__setattr__(self, "operations", tuple(operations))
        object.__setattr__(self, "verifier_evidence_count", verifier_evidence_count)
        if (
            type(verifier_evidence_count) is not int
            or verifier_evidence_count < 0
        ):
            raise ValueError("verifier_evidence_count must be a nonnegative integer")


class TypedMutationPlanAdapter(Protocol):
    """Convert one source row into declarations without executing its content."""

    def __call__(
        self, source_row: object, seed_contract: object
    ) -> MutationPlanInput: ...


def validate_source_identity(
    expected: PinnedSourceContract,
    observed: SourceArtifactIdentity,
) -> None:
    """Fail closed before parsing any row from a mismatched source artifact."""

    if type(expected) is not PinnedSourceContract:
        raise ReplayContractError("expected source contract has an invalid type")
    if type(observed) is not SourceArtifactIdentity:
        raise ReplayContractError("observed source identity has an invalid type")
    fields = (
        ("dataset ID", expected.dataset_id, observed.dataset_id),
        ("revision", expected.revision, observed.revision),
        ("source SHA256", expected.source_sha256, observed.source_sha256),
        ("source size", expected.source_bytes, observed.source_bytes),
    )
    for label, wanted, actual in fields:
        if actual != wanted:
            raise ReplayContractError(f"source {label} mismatch")


def source_artifact_identity(
    contract: PinnedSourceContract,
    payload: bytes,
) -> SourceArtifactIdentity:
    """Hash bytes into an observed identity bound to the pinned hub revision."""

    return SourceArtifactIdentity(
        dataset_id=contract.dataset_id,
        revision=contract.revision,
        source_sha256=hashlib.sha256(payload).hexdigest(),
        source_bytes=len(payload),
    )


def _safe_relative_path(value: object) -> str:
    if type(value) is not str or not value or "\x00" in value:
        raise ReplayContractError("typed mutation path must be nonempty text")
    if value.startswith("/testbed/"):
        value = value.removeprefix("/testbed/")
    path = PurePosixPath(value)
    if (
        path.is_absolute()
        or value in {".", ".."}
        or ".." in path.parts
        or "." in path.parts
    ):
        raise ReplayContractError("typed mutation path escapes the repository")
    return path.as_posix()


def _protected_paths(seed_contract: object) -> tuple[str, ...]:
    values = getattr(seed_contract, "protected_paths", None)
    if type(values) is not tuple or any(type(item) is not str for item in values):
        raise ReplayContractError("seed protected_paths binding is invalid")
    return tuple(sorted(f"/testbed/{_safe_relative_path(path)}" for path in values))


def _to_fable_write(
    operation: WriteMutation, protected_paths: tuple[str, ...]
) -> FableWriteOp:
    if type(operation.content) is not str or "\x00" in operation.content:
        raise ReplayContractError("typed write content must be NUL-free text")
    return FableWriteOp(
        path=f"/testbed/{_safe_relative_path(operation.path)}",
        content=operation.content,
        protected_paths=protected_paths,
    )


def _to_fable_edit(
    operation: EditMutation, protected_paths: tuple[str, ...]
) -> FableEditOp:
    if (
        type(operation.old_string) is not str
        or not operation.old_string
        or "\x00" in operation.old_string
        or type(operation.new_string) is not str
        or "\x00" in operation.new_string
        or type(operation.replace_all) is not bool
    ):
        raise ReplayContractError("typed edit strings or replace_all are invalid")
    return FableEditOp(
        path=f"/testbed/{_safe_relative_path(operation.path)}",
        old_string=operation.old_string,
        new_string=operation.new_string,
        replace_all=operation.replace_all,
        protected_paths=protected_paths,
    )


def _patch_lines(payload: str) -> list[str]:
    if type(payload) is not str or "\x00" in payload:
        raise ReplayContractError("static apply_patch payload must be NUL-free text")
    lines = payload.splitlines()
    if (
        len(lines) < 3
        or lines[0] != "*** Begin Patch"
        or lines[-1] != "*** End Patch"
    ):
        raise ReplayContractError("static apply_patch envelope is invalid")
    return lines[1:-1]


def _parse_update_hunks(
    path: str,
    lines: Sequence[str],
) -> tuple[EditMutation, ...]:
    operations: list[EditMutation] = []
    old: list[str] | None = None
    new: list[str] | None = None

    def finish() -> None:
        nonlocal old, new
        if old is None or new is None:
            return
        if not old:
            raise ReplayContractError("static update hunk has no original text")
        operations.append(
            EditMutation(
                path=path,
                old_string="".join(old),
                new_string="".join(new),
            )
        )
        old = None
        new = None

    for line in lines:
        if line.startswith("@@"):
            finish()
            old, new = [], []
            continue
        if old is None or new is None:
            raise ReplayContractError("static update content precedes its hunk header")
        if not line or line[0] not in {" ", "+", "-"}:
            raise ReplayContractError("static update hunk has an invalid line")
        content = line[1:] + "\n"
        if line[0] in {" ", "-"}:
            old.append(content)
        if line[0] in {" ", "+"}:
            new.append(content)
    finish()
    if not operations:
        raise ReplayContractError("static update has no hunks")
    return tuple(operations)


def parse_static_apply_patch(payload: str) -> tuple[WriteMutation | EditMutation, ...]:
    """Parse Add/Update actions without invoking ``apply_patch`` or a shell."""

    lines = _patch_lines(payload)
    actions: list[WriteMutation | EditMutation] = []
    index = 0
    while index < len(lines):
        header = lines[index]
        index += 1
        if header.startswith("*** Add File: "):
            path = _safe_relative_path(header.removeprefix("*** Add File: "))
            content: list[str] = []
            while index < len(lines) and not lines[index].startswith("*** "):
                line = lines[index]
                index += 1
                if not line.startswith("+"):
                    raise ReplayContractError(
                        "static Add File content must use plus-prefixed lines"
                    )
                content.append(line[1:] + "\n")
            actions.append(WriteMutation(path=path, content="".join(content)))
            continue
        if header.startswith("*** Update File: "):
            path = _safe_relative_path(header.removeprefix("*** Update File: "))
            body: list[str] = []
            while index < len(lines) and not lines[index].startswith("*** "):
                body.append(lines[index])
                index += 1
            actions.extend(_parse_update_hunks(path, body))
            continue
        raise ReplayContractError(
            "static apply_patch supports only Add File and Update File"
        )
    if not actions:
        raise ReplayContractError("static apply_patch contains no mutations")
    return tuple(actions)


_GIT_DIFF_HEADER: Final = re.compile(
    r"diff --git a/(?P<old>[^\s]+) b/(?P<new>[^\s]+)\n\Z"
)
_GIT_INDEX_HEADER: Final = re.compile(
    r"index [0-9a-f]+\.\.[0-9a-f]+(?: [0-7]{6})?\n\Z"
)
_GIT_HUNK_HEADER: Final = re.compile(
    r"@@ -[0-9]+(?:,(?P<old>[0-9]+))? "
    r"\+[0-9]+(?:,(?P<new>[0-9]+))? @@(?: .*)?\n\Z"
)


def parse_static_git_diff(observation: str) -> tuple[EditMutation, ...]:
    """Parse one complete tracked-file unified diff from inert tool output."""

    if type(observation) is not str or "\x00" in observation:
        raise ReplayContractError("static git diff observation must be NUL-free text")
    lines = observation.splitlines(keepends=True)
    try:
        index = next(
            offset
            for offset, line in enumerate(lines)
            if line.startswith("diff --git ")
        )
    except StopIteration as exc:
        raise ReplayContractError("static git diff observation has no diff") from exc

    operations: list[EditMutation] = []
    while index < len(lines) and lines[index].startswith("diff --git "):
        header = _GIT_DIFF_HEADER.fullmatch(lines[index])
        if header is None or header.group("old") != header.group("new"):
            raise ReplayContractError("static git diff has an unsafe file header")
        path = _safe_relative_path(header.group("old"))
        index += 1
        if index >= len(lines) or _GIT_INDEX_HEADER.fullmatch(lines[index]) is None:
            raise ReplayContractError("static git diff has no exact index header")
        index += 1
        if (
            index + 1 >= len(lines)
            or lines[index] != f"--- a/{path}\n"
            or lines[index + 1] != f"+++ b/{path}\n"
        ):
            raise ReplayContractError("static git diff old/new paths are inconsistent")
        index += 2
        file_hunks = 0
        while index < len(lines) and lines[index].startswith("@@ "):
            hunk = _GIT_HUNK_HEADER.fullmatch(lines[index])
            if hunk is None:
                raise ReplayContractError("static git diff hunk header is invalid")
            old_remaining = int(hunk.group("old") or "1")
            new_remaining = int(hunk.group("new") or "1")
            index += 1
            old: list[str] = []
            new: list[str] = []
            while old_remaining or new_remaining:
                if index >= len(lines):
                    raise ReplayContractError("static git diff hunk is truncated")
                line = lines[index]
                if line and not line.endswith("\n") and index == len(lines) - 1:
                    # Tool-result transports may strip the stream's terminal
                    # newline. Unified diff would emit a dedicated no-newline
                    # marker when the source file itself lacks one.
                    line += "\n"
                if not line.endswith("\n") or not line or line[0] not in {" ", "+", "-"}:
                    raise ReplayContractError("static git diff hunk is truncated")
                content = line[1:]
                if line[0] in {" ", "-"}:
                    if old_remaining == 0:
                        raise ReplayContractError("static git diff old hunk overflows")
                    old_remaining -= 1
                    old.append(content)
                if line[0] in {" ", "+"}:
                    if new_remaining == 0:
                        raise ReplayContractError("static git diff new hunk overflows")
                    new_remaining -= 1
                    new.append(content)
                index += 1
            if not old:
                raise ReplayContractError(
                    "static git diff may not synthesize an untracked file"
                )
            operations.append(
                EditMutation(
                    path=path,
                    old_string="".join(old),
                    new_string="".join(new),
                )
            )
            file_hunks += 1
        if file_hunks == 0:
            raise ReplayContractError("static git diff file has no complete hunks")
        if index < len(lines) and lines[index].startswith("diff --git "):
            continue
        if any(line.startswith("diff --git ") for line in lines[index:]):
            raise ReplayContractError("static git diff blocks are not contiguous")
        break
    if not operations:
        raise ReplayContractError("static git diff contains no typed edits")
    return tuple(operations)


def _operation_sha256(
    operations: Sequence[FableWriteOp | FableEditOp],
) -> str:
    payload = [
        {"type": type(operation).__name__, **dataclasses.asdict(operation)}
        for operation in operations
    ]
    canonical = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def fable_message_plan_adapter(
    source_row: object,
    seed_contract: object,
) -> MutationPlanInput:
    """Compatibility adapter for the existing Fable typed Write/Edit parser."""

    if type(source_row) is not dict or type(source_row.get("messages")) is not list:
        raise ReplayContractError("Fable source row has no messages list")
    verify_cmd = getattr(seed_contract, "verify_cmd", None)
    protected = getattr(seed_contract, "protected_paths", None)
    if type(verify_cmd) is not str or type(protected) is not tuple:
        raise ReplayContractError("seed verifier binding is invalid")
    plan = canonical_mutation_plan(
        source_row["messages"],
        frozenset(protected),
        verify_cmd,
    )
    return MutationPlanInput(
        plan.operations,
        verifier_evidence_count=plan.verifier_evidence_count,
    )


def build_trusted_mutation_plan(
    *,
    expected_source: PinnedSourceContract,
    observed_source: SourceArtifactIdentity,
    source_row: object,
    seed_contract: object,
    adapter: TypedMutationPlanAdapter,
) -> MutationPlan:
    """Build the only mutation type accepted by the trusted replay executor."""

    validate_source_identity(expected_source, observed_source)
    if not callable(adapter):
        raise ReplayContractError("typed mutation plan adapter is not callable")
    supplied = adapter(source_row, seed_contract)
    if type(supplied) is not MutationPlanInput:
        raise ReplayContractError("adapter must return the exact MutationPlanInput type")

    protected = _protected_paths(seed_contract)
    operations: list[FableWriteOp | FableEditOp] = []
    pending: list[object] = list(supplied.operations)
    while pending:
        operation = pending.pop(0)
        if type(operation) is WriteMutation:
            operations.append(_to_fable_write(operation, protected))
        elif type(operation) is EditMutation:
            operations.append(_to_fable_edit(operation, protected))
        elif type(operation) is StaticApplyPatch:
            pending[0:0] = list(parse_static_apply_patch(operation.patch))
        elif type(operation) in {FableWriteOp, FableEditOp}:
            if operation.protected_paths != protected:
                raise ReplayContractError(
                    "typed mutation protected_paths binding mismatch"
                )
            operations.append(operation)
        else:
            raise ReplayContractError(
                "raw transcript commands are not executable; "
                "adapter must return exact typed mutations"
            )

    verify_cmd = getattr(seed_contract, "verify_cmd", None)
    verifier_sha256 = getattr(seed_contract, "verifier_sha256", None)
    if type(verify_cmd) is not str or not verify_cmd:
        raise ReplayContractError("seed verify_cmd binding is invalid")
    expected_verifier_sha = hashlib.sha256(verify_cmd.encode("utf-8")).hexdigest()
    if verifier_sha256 != expected_verifier_sha:
        raise ReplayContractError("seed verifier SHA256 binding mismatch")
    return MutationPlan(
        operations=tuple(operations),
        operation_sha256=_operation_sha256(operations),
        verifier_sha256=expected_verifier_sha,
        verifier_evidence_count=supplied.verifier_evidence_count,
    )


__all__ = [
    "EditMutation",
    "MutationPlanInput",
    "PinnedSourceContract",
    "SourceArtifactIdentity",
    "StaticApplyPatch",
    "TypedMutationPlanAdapter",
    "WriteMutation",
    "build_trusted_mutation_plan",
    "fable_message_plan_adapter",
    "parse_static_apply_patch",
    "parse_static_git_diff",
    "source_artifact_identity",
    "validate_source_identity",
]
