from __future__ import annotations

import dataclasses
import hashlib
import json
from types import SimpleNamespace

import pytest

from teacher_platform.fable5_import import FableEditOp, FableWriteOp
from teacher_platform.fable5_replay import ReplayContractError
from teacher_platform.trace_replay import (
    MutationPlanInput,
    PinnedSourceContract,
    SourceArtifactIdentity,
    StaticApplyPatch,
    build_trusted_mutation_plan,
    fable_message_plan_adapter,
    parse_static_git_diff,
)

VERIFY_CMD = "python3 -m pytest -q"


def _source() -> PinnedSourceContract:
    return PinnedSourceContract(
        dataset_id="greghavens/fable-5-coding-and-debugging-traces",
        revision="a" * 40,
        source_sha256="b" * 64,
        source_bytes=123,
    )


def _identity(**updates: object) -> SourceArtifactIdentity:
    values: dict[str, object] = {
        "dataset_id": _source().dataset_id,
        "revision": _source().revision,
        "source_sha256": _source().source_sha256,
        "source_bytes": _source().source_bytes,
    }
    values.update(updates)
    return SourceArtifactIdentity(**values)  # type: ignore[arg-type]


def _seed() -> SimpleNamespace:
    return SimpleNamespace(
        protected_paths=("tests/test_locked.py",),
        verify_cmd=VERIFY_CMD,
        verifier_sha256=hashlib.sha256(VERIFY_CMD.encode()).hexdigest(),
    )


def test_fable_adapter_preserves_the_existing_typed_write_contract() -> None:
    row = {
        "messages": [
            {"role": "system", "content": "system"},
            {"role": "user", "content": "fix it"},
            {
                "role": "assistant",
                "content": "edit",
                "tool_calls": [
                    {
                        "id": "call-1",
                        "type": "function",
                        "function": {
                            "name": "Write",
                            "arguments": json.dumps(
                                {
                                    "file_path": "/testbed/src/fix.py",
                                    "content": "fixed = True\n",
                                }
                            ),
                        },
                    }
                ],
            },
        ]
    }

    plan = build_trusted_mutation_plan(
        expected_source=_source(),
        observed_source=_identity(),
        source_row=row,
        seed_contract=_seed(),
        adapter=fable_message_plan_adapter,
    )

    assert len(plan.operations) == 1
    assert plan.operations[0] == FableWriteOp(
        path="/testbed/src/fix.py",
        content="fixed = True\n",
        protected_paths=("/testbed/tests/test_locked.py",),
    )
    assert len(plan.operation_sha256) == 64


def test_source_revision_mismatch_is_rejected_before_adapter_runs() -> None:
    called = False

    def adapter(_row: object, _seed: object) -> MutationPlanInput:
        nonlocal called
        called = True
        return MutationPlanInput(())

    with pytest.raises(ReplayContractError, match="source revision mismatch"):
        build_trusted_mutation_plan(
            expected_source=_source(),
            observed_source=_identity(revision="c" * 40),
            source_row={},
            seed_contract=_seed(),
            adapter=adapter,
        )

    assert called is False


def test_raw_transcript_command_is_never_accepted_as_a_mutation() -> None:
    def unsafe_adapter(_row: object, _seed: object) -> MutationPlanInput:
        return MutationPlanInput(({"command": "rm -rf /"},))

    with pytest.raises(
        ReplayContractError, match="raw transcript commands are not executable"
    ):
        build_trusted_mutation_plan(
            expected_source=_source(),
            observed_source=_identity(),
            source_row={},
            seed_contract=_seed(),
            adapter=unsafe_adapter,
        )


def test_static_apply_patch_add_file_becomes_a_typed_write() -> None:
    patch = (
        "*** Begin Patch\n"
        "*** Add File: src/new.py\n"
        "+answer = 42\n"
        "*** End Patch"
    )

    def patch_adapter(_row: object, _seed: object) -> MutationPlanInput:
        return MutationPlanInput((StaticApplyPatch(patch),))

    plan = build_trusted_mutation_plan(
        expected_source=_source(),
        observed_source=_identity(),
        source_row={},
        seed_contract=_seed(),
        adapter=patch_adapter,
    )

    assert plan.operations == (
        FableWriteOp(
            path="/testbed/src/new.py",
            content="answer = 42\n",
            protected_paths=("/testbed/tests/test_locked.py",),
        ),
    )


def test_static_apply_patch_update_hunk_becomes_a_typed_edit() -> None:
    patch = (
        "*** Begin Patch\n"
        "*** Update File: src/current.py\n"
        "@@\n"
        " def solve():\n"
        "-    return 0\n"
        "+    return 42\n"
        "*** End Patch"
    )

    def patch_adapter(_row: object, _seed: object) -> MutationPlanInput:
        return MutationPlanInput((StaticApplyPatch(patch),))

    plan = build_trusted_mutation_plan(
        expected_source=_source(),
        observed_source=_identity(),
        source_row={},
        seed_contract=_seed(),
        adapter=patch_adapter,
    )

    assert plan.operations == (
        FableEditOp(
            path="/testbed/src/current.py",
            old_string="def solve():\n    return 0\n",
            new_string="def solve():\n    return 42\n",
            replace_all=False,
            protected_paths=("/testbed/tests/test_locked.py",),
        ),
    )


def test_static_git_diff_extracts_complete_multi_file_hunks_from_observation() -> None:
    observation = (
        "Script completed\nWall time 0.1 seconds\nOutput:\n\n"
        "diff --git a/one.py b/one.py\n"
        "index 1111111..2222222 100644\n"
        "--- a/one.py\n"
        "+++ b/one.py\n"
        "@@ -1,2 +1,2 @@\n"
        " keep\n"
        "-old\n"
        "+new\n"
        "diff --git a/src/two.rs b/src/two.rs\n"
        "index 3333333..4444444 100644\n"
        "--- a/src/two.rs\n"
        "+++ b/src/two.rs\n"
        "@@ -4 +4,2 @@\n"
        "-before\n"
        "+after\n"
        "+more\n"
        " M one.py\n"
    )

    operations = parse_static_git_diff(observation)

    assert [(item.path, item.old_string, item.new_string) for item in operations] == [
        ("one.py", "keep\nold\n", "keep\nnew\n"),
        ("src/two.rs", "before\n", "after\nmore\n"),
    ]


def test_static_git_diff_rejects_a_truncated_hunk() -> None:
    observation = (
        "diff --git a/one.py b/one.py\n"
        "index 1111111..2222222 100644\n"
        "--- a/one.py\n"
        "+++ b/one.py\n"
        "@@ -1,2 +1,2 @@\n"
        "-old\n"
        "+new\n"
    )

    with pytest.raises(ReplayContractError, match="truncated"):
        parse_static_git_diff(observation)


def test_source_contract_is_immutable() -> None:
    with pytest.raises(dataclasses.FrozenInstanceError):
        _source().revision = "c" * 40  # type: ignore[misc]
