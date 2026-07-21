"""Task 1 tests for pinned Fable source selection and structure validation."""

from __future__ import annotations

import json
from dataclasses import FrozenInstanceError

import pytest

from teacher_platform.fable5_import import (
    DATASET_ID,
    DATASET_REVISION,
    EXPECTED_ROWS,
    EXPECTED_TERMINAL_TRAJECTORIES,
    SOURCE_BYTES,
    SOURCE_LFS_SHA256,
    DropReason,
    RowRejected,
    SourceContract,
    canonical_language,
    select_terminal_row,
    validate_source_messages,
)


def tool_call(call_id: str, *, arguments: object | None = None) -> dict[str, object]:
    return {
        "id": call_id,
        "type": "function",
        "function": {
            "name": "Bash",
            "arguments": {"command": "true"} if arguments is None else arguments,
        },
    }


def source_messages(assistant_steps: int = 2) -> list[dict[str, object]]:
    messages: list[dict[str, object]] = [
        {"role": "system", "content": "Work carefully."},
        {"role": "user", "content": "Fix the implementation."},
    ]
    for step in range(1, assistant_steps):
        call_id = f"call-{step}"
        messages.extend(
            [
                {
                    "role": "assistant",
                    "content": "Inspect first.",
                    "tool_calls": [tool_call(call_id)],
                },
                {"role": "tool", "tool_call_id": call_id, "content": "ok"},
            ]
        )
    messages.append({"role": "assistant", "content": "Implemented and verified."})
    return messages


def fable_row(**updates: object) -> dict[str, object]:
    assistant_steps = updates.get("assistant_steps", 2)
    messages = updates.get("messages")
    if messages is None:
        steps_for_fixture = assistant_steps if type(assistant_steps) is int else 2
        messages = source_messages(steps_for_fixture)
    row: dict[str, object] = {
        "task": "rs-default",
        "lang": "rust",
        "category": "debug",
        "split": "train",
        "assistant_step": assistant_steps,
        "assistant_steps": assistant_steps,
        "messages": messages,
        "tools": json.dumps([]),
    }
    row.update(updates)
    return row


def test_source_contract_is_exact_and_frozen() -> None:
    assert (
        DATASET_ID,
        DATASET_REVISION,
        SOURCE_LFS_SHA256,
        SOURCE_BYTES,
        EXPECTED_ROWS,
        EXPECTED_TERMINAL_TRAJECTORIES,
    ) == (
        "greghavens/fable-5-coding-and-debugging-traces",
        "aef8506515979988aa5c1a423f5b0fb3cee60382",
        "ef86c61a8e3b69197d381e2e9b6fe1965005c604fa39ba35e0721457813306c3",
        730_331_947,
        12_408,
        2_377,
    )
    contract = SourceContract()
    assert (
        contract.dataset_id,
        contract.dataset_revision,
        contract.source_lfs_sha256,
        contract.source_bytes,
        contract.expected_rows,
        contract.expected_terminal_trajectories,
    ) == (
        DATASET_ID,
        DATASET_REVISION,
        SOURCE_LFS_SHA256,
        SOURCE_BYTES,
        EXPECTED_ROWS,
        EXPECTED_TERMINAL_TRAJECTORIES,
    )
    with pytest.raises(FrozenInstanceError):
        contract.expected_rows = 1  # type: ignore[misc]


@pytest.mark.parametrize(
    "override",
    [
        {"dataset_id": "other/source"},
        {"dataset_revision": "main"},
        {"source_lfs_sha256": "0" * 64},
        {"source_bytes": 1},
        {"expected_rows": 1},
        {"expected_terminal_trajectories": 1},
    ],
)
def test_source_contract_rejects_alternate_provenance(
    override: dict[str, object],
) -> None:
    with pytest.raises(TypeError):
        SourceContract(**override)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        ("python", "python"),
        ("py", "python"),
        ("rust", "rust"),
        ("cpp", "cpp"),
        ("c++", "cpp"),
        ("Python", None),
        (" python", None),
        ("go", None),
        (1, None),
    ],
)
def test_canonical_language_uses_only_exact_aliases(
    source: object, expected: str | None
) -> None:
    assert canonical_language(source) == expected


def test_selects_only_terminal_training_rows() -> None:
    row = fable_row(task="rs-one", split="train", assistant_step=4, assistant_steps=4)

    selected = select_terminal_row(row)

    assert selected.task == "rs-one"
    assert selected.language == "rust"
    assert selected.category == "debug"
    assert selected.messages == tuple(row["messages"])
    assert selected.tools_json == "[]"


@pytest.mark.parametrize(
    ("updates", "reason"),
    [
        ({"split": "val"}, DropReason.VALIDATION),
        ({"assistant_step": 3, "assistant_steps": 4}, DropReason.NONTERMINAL),
        ({"lang": "go"}, DropReason.LANGUAGE),
    ],
)
def test_rejects_ineligible_rows(
    updates: dict[str, object], reason: DropReason
) -> None:
    row = fable_row(**updates)
    with pytest.raises(RowRejected, match=reason.value):
        select_terminal_row(row)


@pytest.mark.parametrize(
    "updates",
    [
        {"task": 7},
        {"category": None},
        {"split": True},
        {"assistant_step": True},
        {"assistant_steps": "2"},
        {"messages": tuple(source_messages())},
        {"tools": []},
    ],
)
def test_row_fields_require_exact_source_types(updates: dict[str, object]) -> None:
    with pytest.raises(RowRejected, match=DropReason.MALFORMED_ROW.value):
        select_terminal_row(fable_row(**updates))


def test_rejects_mismatched_assistant_count() -> None:
    row = fable_row(assistant_steps=3, messages=source_messages(2))

    with pytest.raises(RowRejected, match=DropReason.ASSISTANT_COUNT.value):
        select_terminal_row(row)


def test_source_contract_rejects_duplicate_terminal_tasks() -> None:
    rows = [fable_row(task="rs-one"), fable_row(task="rs-one")]

    with pytest.raises(RowRejected, match=DropReason.DUPLICATE_TASK.value):
        SourceContract().select_terminal_rows(rows)


@pytest.mark.parametrize(
    "message",
    [
        {"role": "developer", "content": "unsupported"},
        {"content": "missing role"},
        {"role": 7, "content": "wrong type"},
    ],
)
def test_rejects_malformed_or_unsupported_roles(message: dict[str, object]) -> None:
    messages = source_messages()
    messages.insert(-1, message)

    with pytest.raises(RowRejected, match=DropReason.UNSUPPORTED_ROLE.value):
        validate_source_messages(messages)


def test_rejects_missing_tool_call_id() -> None:
    messages = source_messages()
    del messages[2]["tool_calls"][0]["id"]  # type: ignore[index]

    with pytest.raises(RowRejected, match=DropReason.MISSING_TOOL_CALL_ID.value):
        validate_source_messages(messages)


def test_rejects_duplicate_tool_call_id() -> None:
    messages = source_messages()
    messages[2]["tool_calls"].append(tool_call("call-1"))  # type: ignore[union-attr]

    with pytest.raises(RowRejected, match=DropReason.DUPLICATE_TOOL_CALL.value):
        validate_source_messages(messages)


def test_rejects_orphan_tool_result() -> None:
    messages = source_messages()
    messages.insert(2, {"role": "tool", "tool_call_id": "never-called", "content": "x"})

    with pytest.raises(RowRejected, match=DropReason.ORPHAN_TOOL_RESULT.value):
        validate_source_messages(messages)


def test_rejects_duplicate_tool_result() -> None:
    messages = source_messages()
    messages.insert(-1, dict(messages[3]))

    with pytest.raises(RowRejected, match=DropReason.DUPLICATE_TOOL_RESULT.value):
        validate_source_messages(messages)


def test_accepts_parallel_tool_results_in_any_order_without_reordering_calls() -> None:
    messages = source_messages()
    messages[2]["tool_calls"] = [tool_call("call-1"), tool_call("call-2")]
    messages[3]["tool_call_id"] = "call-2"
    messages.insert(4, {"role": "tool", "tool_call_id": "call-1", "content": "one"})

    validated = validate_source_messages(messages)

    assert validated == tuple(messages)
    assert [call["id"] for call in validated[2]["tool_calls"]] == [  # type: ignore[index]
        "call-1",
        "call-2",
    ]


def test_rejects_cross_group_interleaving_before_parallel_results_complete() -> None:
    messages = source_messages()
    messages[2]["tool_calls"] = [tool_call("call-1"), tool_call("call-2")]
    messages[3]["tool_call_id"] = "call-2"
    messages.insert(
        4,
        {
            "role": "assistant",
            "content": "Starting another group too early.",
            "tool_calls": [tool_call("call-3")],
        },
    )
    messages.insert(5, {"role": "tool", "tool_call_id": "call-3", "content": "three"})
    messages.insert(6, {"role": "tool", "tool_call_id": "call-1", "content": "one"})

    with pytest.raises(RowRejected, match=DropReason.MISSING_TOOL_RESULT.value):
        validate_source_messages(messages)


def test_rejects_missing_tool_result() -> None:
    messages = source_messages()
    del messages[3]

    with pytest.raises(RowRejected, match=DropReason.MISSING_TOOL_RESULT.value):
        validate_source_messages(messages)


def test_rejects_malformed_argument_json() -> None:
    messages = source_messages()
    messages[2]["tool_calls"][0]["function"]["arguments"] = "{"  # type: ignore[index]

    with pytest.raises(RowRejected, match=DropReason.MALFORMED_ARGUMENTS.value):
        validate_source_messages(messages)


@pytest.mark.parametrize("terminal_role", ["user", "tool"])
def test_requires_final_assistant_message(terminal_role: str) -> None:
    messages = source_messages()
    messages[-1] = {"role": terminal_role, "content": "trailing"}
    if terminal_role == "tool":
        messages[-1]["tool_call_id"] = "call-1"

    with pytest.raises(RowRejected, match=DropReason.NON_ASSISTANT_TERMINAL.value):
        validate_source_messages(messages)


def test_validates_native_and_json_encoded_argument_objects() -> None:
    messages = source_messages(3)
    messages[4]["tool_calls"][0]["function"]["arguments"] = json.dumps(  # type: ignore[index]
        {"file_path": "src/lib.rs"}
    )

    assert validate_source_messages(messages) == tuple(messages)


@pytest.mark.parametrize(
    "mutate",
    [
        lambda messages: messages[1].update(content=None),
        lambda messages: messages[2]["tool_calls"][0].update(type="custom"),
        lambda messages: messages[2]["tool_calls"][0]["function"].update(name=""),
    ],
)
def test_rejects_malformed_message_and_call_payloads(mutate) -> None:
    messages = source_messages()
    mutate(messages)

    with pytest.raises(RowRejected, match=DropReason.MALFORMED_ROW.value):
        validate_source_messages(messages)
