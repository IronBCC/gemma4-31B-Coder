"""Task 1 tests for pinned Fable source selection and structure validation."""

from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fable5_import import (  # noqa: E402
    DATASET_ID,
    DATASET_REVISION,
    EXPECTED_ROWS,
    EXPECTED_TERMINAL_TRAJECTORIES,
    SOURCE_BYTES,
    SOURCE_LFS_SHA256,
    DropReason,
    RowRejected,
    SelectedTrajectory,
    SourceContract,
    canonical_language,
    convert_trajectory,
    select_terminal_row,
    translate_fable_tool_call,
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


def fable_tool_call(
    name: str,
    arguments: object,
    *,
    call_id: str = "call-1",
) -> dict[str, object]:
    return {
        "id": call_id,
        "type": "function",
        "function": {"name": name, "arguments": arguments},
    }


def _run_in_testbed(command: str, testbed: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["bash", "-c", command.replace("/testbed", str(testbed))],
        text=True,
        capture_output=True,
        check=False,
    )


def test_bash_translation_accepts_native_and_json_arguments() -> None:
    native = fable_tool_call(
        "Bash",
        {"command": "cargo test", "description": "Run tests", "timeout": 30},
    )
    encoded = fable_tool_call("Bash", json.dumps({"command": "cargo test"}))

    assert translate_fable_tool_call(native, frozenset()) == "cargo test"
    assert translate_fable_tool_call(encoded, frozenset()) == "cargo test"


@pytest.mark.parametrize(
    ("arguments", "message"),
    [
        ({"command": "cargo test", "run_in_background": True}, "unsupported Bash arguments"),
        ({"command": "cargo test", "timeout": 0}, "timeout must be a positive integer"),
        ({"command": "cargo test", "extra": 1}, "unsupported Bash arguments"),
        ({"command": "cat ../secret"}, "unsafe workspace path"),
        ({"command": "cat /workspace/secret"}, "unsafe workspace path"),
        ({"command": "echo 'unterminated"}, "ambiguous shell quoting"),
        ({"command": "true &"}, "background execution"),
    ],
)
def test_bash_translation_rejects_unsafe_or_semantic_options(
    arguments: dict[str, object], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        translate_fable_tool_call(fable_tool_call("Bash", arguments), frozenset())


@pytest.mark.parametrize(
    "arguments",
    [
        [],
        "not-json",
        json.dumps([{"command": "true"}]),
    ],
)
def test_translation_rejects_malformed_argument_objects(arguments: object) -> None:
    with pytest.raises(ValueError, match="arguments"):
        translate_fable_tool_call(fable_tool_call("Bash", arguments), frozenset())


def test_translation_rejects_non_string_argument_keys_stably() -> None:
    with pytest.raises(ValueError, match="argument keys must be strings"):
        translate_fable_tool_call(
            fable_tool_call("Bash", {"command": "true", 7: "bad"}),
            frozenset(),
        )


def test_read_translation_is_bounded_and_requires_regular_file(tmp_path: Path) -> None:
    target = tmp_path / "src/lib.rs"
    target.parent.mkdir()
    target.write_text("one\ntwo\nthree\n", encoding="utf-8")
    command = translate_fable_tool_call(
        fable_tool_call(
            "Read", {"file_path": "src/lib.rs", "offset": 2, "limit": 1}
        ),
        frozenset(),
    )

    result = _run_in_testbed(command, tmp_path)

    assert result.returncode == 0
    assert result.stdout == "two\n"
    directory_result = _run_in_testbed(
        translate_fable_tool_call(
            fable_tool_call("Read", {"file_path": "src"}), frozenset()
        ),
        tmp_path,
    )
    assert directory_result.returncode != 0


def test_write_translation_uses_literal_payload_and_blocks_symlink_escape(
    tmp_path: Path,
) -> None:
    payload = "literal $(touch should-not-exist)\n'''\n"
    command = translate_fable_tool_call(
        fable_tool_call("Write", {"file_path": "src/lib.rs", "content": payload}),
        frozenset(),
    )
    (tmp_path / "src").mkdir()

    result = _run_in_testbed(command, tmp_path)

    assert result.returncode == 0
    assert (tmp_path / "src/lib.rs").read_text(encoding="utf-8") == payload
    assert not (tmp_path / "should-not-exist").exists()

    outside = tmp_path.parent / f"{tmp_path.name}-outside"
    outside.mkdir()
    (tmp_path / "escape").symlink_to(outside, target_is_directory=True)
    escaped = translate_fable_tool_call(
        fable_tool_call("Write", {"file_path": "escape/pwned", "content": "no"}),
        frozenset(),
    )
    escaped_result = _run_in_testbed(escaped, tmp_path)
    assert escaped_result.returncode != 0
    assert not (outside / "pwned").exists()


def test_write_translation_rejects_protected_path() -> None:
    with pytest.raises(ValueError, match="protected path"):
        translate_fable_tool_call(
            fable_tool_call(
                "Write", {"file_path": "tests/test_contract.py", "content": "bad"}
            ),
            frozenset({"tests/test_contract.py"}),
        )


def test_edit_translation_requires_exact_one_match_without_mutation(
    tmp_path: Path,
) -> None:
    target = tmp_path / "src/lib.rs"
    target.parent.mkdir()
    target.write_text("old old", encoding="utf-8")
    command = translate_fable_tool_call(
        fable_tool_call(
            "Edit",
            {
                "file_path": "src/lib.rs",
                "old_string": "old",
                "new_string": "new",
            },
        ),
        frozenset(),
    )

    result = _run_in_testbed(command, tmp_path)

    assert result.returncode != 0
    assert target.read_text(encoding="utf-8") == "old old"


def test_edit_translation_replace_all_is_explicit_and_executable(tmp_path: Path) -> None:
    target = tmp_path / "src/lib.rs"
    target.parent.mkdir()
    target.write_text("old old", encoding="utf-8")
    command = translate_fable_tool_call(
        fable_tool_call(
            "Edit",
            {
                "file_path": "src/lib.rs",
                "old_string": "old",
                "new_string": "new",
                "replace_all": True,
            },
        ),
        frozenset(),
    )

    result = _run_in_testbed(command, tmp_path)

    assert result.returncode == 0
    assert target.read_text(encoding="utf-8") == "new new"


def test_glob_and_grep_translation_are_bounded_and_shell_quoted(tmp_path: Path) -> None:
    source = tmp_path / "src/name with space.py"
    source.parent.mkdir()
    source.write_text("needle\nother\n", encoding="utf-8")
    glob_command = translate_fable_tool_call(
        fable_tool_call("Glob", {"pattern": "*.py", "path": "src"}),
        frozenset(),
    )
    grep_command = translate_fable_tool_call(
        fable_tool_call(
            "Grep",
            {
                "pattern": "needle",
                "path": "src",
                "glob": "*.py",
                "output_mode": "content",
                "head_limit": 5,
            },
        ),
        frozenset(),
    )

    glob_result = _run_in_testbed(glob_command, tmp_path)
    grep_result = _run_in_testbed(grep_command, tmp_path)

    assert glob_result.returncode == 0
    assert "name with space.py" in glob_result.stdout
    assert grep_result.returncode == 0
    assert "needle" in grep_result.stdout
    assert "head -n 5" in grep_command


@pytest.mark.parametrize(
    ("name", "arguments", "message"),
    [
        ("WebSearch", {"query": "x"}, "unsupported Fable tool"),
        ("Read", {"file_path": "/etc/passwd"}, "absolute path"),
        ("Read", {"file_path": "src/../secret"}, "parent traversal"),
        ("Read", {"file_path": "src//lib.rs"}, "empty path component"),
        ("Read", {"file_path": "src/nu\x00l"}, "NUL"),
        ("Glob", {"pattern": "*.py", "path": "src", "limit": 5}, "unsupported Glob arguments"),
        ("Grep", {"pattern": "x", "path": "src", "multiline": True}, "unsupported Grep arguments"),
        ("Grep", {"pattern": "x", "path": "src", "-n": False}, "Grep -n cannot be disabled"),
    ],
)
def test_translation_fails_closed_for_unknown_tools_paths_and_options(
    name: str, arguments: dict[str, object], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        translate_fable_tool_call(fable_tool_call(name, arguments), frozenset())


def test_conversion_serializes_parallel_calls_and_pairs_results_by_id() -> None:
    messages = [
        {"role": "system", "content": "Claude-specific instructions."},
        {"role": "user", "content": "Fix the parser."},
        {
            "role": "assistant",
            "content": "Inspect both files.",
            "tool_calls": [
                fable_tool_call("Read", {"file_path": "src/a.py"}, call_id="a"),
                fable_tool_call("Read", {"file_path": "src/b.py"}, call_id="b"),
            ],
        },
        {"role": "tool", "tool_call_id": "b", "content": "B bytes\n"},
        {"role": "tool", "tool_call_id": "a", "content": "A bytes\n"},
        {
            "role": "assistant",
            "content": "Write the fix.",
            "tool_calls": [
                fable_tool_call(
                    "Write",
                    {"file_path": "src/a.py", "content": "fixed = True\n"},
                    call_id="write",
                )
            ],
        },
        {"role": "tool", "tool_call_id": "write", "content": "wrote"},
        {
            "role": "assistant",
            "content": "Verify.",
            "tool_calls": [
                fable_tool_call("Bash", {"command": "pytest -q"}, call_id="test")
            ],
        },
        {"role": "tool", "tool_call_id": "test", "content": "1 passed"},
        {"role": "assistant", "content": "Implemented and verified."},
    ]
    selected = SelectedTrajectory(
        task="py-parser-fix",
        language="python",
        category="debug",
        messages=validate_source_messages(messages),
        tools_json="[]",
    )

    row = convert_trajectory(selected, frozenset())

    assert row["instance_id"] == "py-parser-fix"
    assert row["source_instance_id"] == "py-parser-fix"
    assert row["repo"] == "moonshiner/py-parser-fix"
    assert row["source"] == f"teacher:fable5:{DATASET_REVISION}:py-parser-fix"
    assert row["messages"][0]["role"] == "system"
    assert "practical software engineer" in row["messages"][0]["content"]
    assert "Claude-specific" not in row["messages"][0]["content"]
    assert row["messages"][1] == {
        "role": "user",
        "content": "<pr_description>\nFix the parser.\n</pr_description>",
        "tool_calls": [],
    }
    assistants = [m for m in row["messages"] if m["role"] == "assistant"]
    assert [m["content"] for m in assistants] == [
        "Inspect both files.",
        "",
        "Write the fix.",
        "Verify.",
        "Implemented and verified.",
    ]
    assert all(message["loss"] is True for message in assistants)
    assert all(len(message["tool_calls"]) <= 1 for message in assistants)
    call_ids = [
        message["tool_calls"][0]["id"]
        for message in assistants
        if message["tool_calls"]
    ]
    assert call_ids == ["fable-tool-0", "fable-tool-1", "fable-tool-2", "fable-tool-3"]
    observations = [
        message for message in row["messages"] if message["role"] == "user"
    ][1:]
    assert [message["content"] for message in observations] == [
        "OBSERVATION:\nA bytes\n",
        "OBSERVATION:\nB bytes\n",
        "OBSERVATION:\nwrote",
        "OBSERVATION:\n1 passed",
    ]
    assert all("loss" not in message for message in observations)
    assert row["messages"][-1] == {
        "role": "assistant",
        "content": "Implemented and verified.",
        "tool_calls": [],
        "loss": True,
    }
