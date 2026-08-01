from __future__ import annotations

import json
import shlex

import pytest

from phaseD_sft.build_verified_teacher_sft import (
    build_verified_teacher_rows,
    load_teacher_records,
    render_verified_teacher_sft,
    unwrap_testbed_command,
)


PATCH = """diff --git a/pkg/target.py b/pkg/target.py
index 1234567..89abcde 100644
--- a/pkg/target.py
+++ b/pkg/target.py
@@ -1,2 +1,2 @@
 def target():
-    return 'old'
+    return 'new'
"""


def _completed(item: dict[str, object]) -> str:
    return json.dumps({"type": "item.completed", "item": item})


def _command(command: str, output: str, *, exit_code: int = 0) -> str:
    payload = f"docker exec abc123def456 bash -c {shlex.quote(f'cd /testbed && {command}')}"
    wrapped = f"/bin/bash -lc {shlex.quote(payload)}"
    return _completed(
        {
            "type": "command_execution",
            "command": wrapped,
            "aggregated_output": output,
            "exit_code": exit_code,
        }
    )


def _record() -> dict[str, object]:
    return {
        "task": {
            "instance_id": "task-a",
            "problem_statement": "Fix target() in pkg/target.py.",
            "repo": "repo-a",
        },
        "result": {
            "instance_id": "task-a",
            "backend": "codex",
            "model": "gpt-5.6-terra",
            "resolved": True,
        },
        "patch": PATCH,
        "stream_lines": [
            _completed({"type": "agent_message", "text": "Codex prose must disappear."}),
            _command("ls", "README.md\npkg"),
            _command("sed -n '1,80p' pkg/target.py", "pkg/target.py\ndef target():\n    return 'old'"),
            _completed({"type": "agent_message", "text": "I know the fix now."}),
            _command("grep -n old pkg/target.py", "pkg/target.py:2:    return 'old'"),
            _command("sed -i \"s/'old'/'new'/\" pkg/target.py", ""),
            _command("python -m pytest -q tests/test_target.py", "1 passed"),
        ],
        "system_template": "You are a practical software engineer.",
        "instance_template": "<pr_description>\n{{task}}\n</pr_description>",
    }


def _assistant_commands(row: dict[str, object]) -> list[str]:
    commands: list[str] = []
    for message in row["messages"]:
        if message["role"] != "assistant":
            continue
        call = message["tool_calls"][0]
        commands.append(json.loads(call["function"]["arguments"])["command"])
    return commands


def test_unwraps_only_the_expected_testbed_container_command() -> None:
    wrapped = (
        "/bin/bash -lc \"docker exec abc123def456 bash -c "
        "\\\"cd /testbed && sed -n '1,20p' pkg/mod.py\\\"\""
    )

    assert unwrap_testbed_command(wrapped, expected_container="abc123def456") == "sed -n '1,20p' pkg/mod.py"
    assert unwrap_testbed_command(wrapped, expected_container="other") is None
    assert unwrap_testbed_command("docker exec abc123def456 bash -c 'cd /tmp && pwd'") is None


def test_unwrap_does_not_reparse_malformed_inner_shell_quoting() -> None:
    wrapped = (
        '/bin/bash -lc "docker exec abc123def456 bash -c '
        '\\"cd /testbed && rg -n \\\\\"def target|target\\\\(\\\" .\\""'
    )

    payload = (
        'docker exec abc123def456 bash -c '
        '"cd /testbed && rg -n \\"def target|target\\(" ."'
    )
    wrapped = f"/bin/bash -lc {shlex.quote(payload)}"
    assert unwrap_testbed_command(wrapped, expected_container="abc123def456") == (
        'rg -n \\"def target|target\\(" .'
    )


def test_verified_patch_renders_native_bash_calls_and_observation_pairs() -> None:
    row, report = render_verified_teacher_sft(_record(), keep_pre_edit_reads=3)

    assert report["kept"] is True
    assert row is not None
    assert row["source_instance_id"] == "task-a"
    assert row["messages"][-1]["role"] == "assistant"
    assert row["messages"][-1]["tool_calls"][0]["function"]["name"] == "bash"
    assert _assistant_commands(row) == [
        "sed -n '1,80p' pkg/target.py",
        "grep -n old pkg/target.py",
        "sed -i \"s/'old'/'new'/\" pkg/target.py",
    ]
    assistant_messages = [message for message in row["messages"] if message["role"] == "assistant"]
    assert [message["loss"] for message in assistant_messages] == [False, False, True]

    for index, message in enumerate(row["messages"][:-1]):
        if message["role"] == "assistant":
            observation = row["messages"][index + 1]
            assert observation["role"] == "user"
            assert observation["content"].startswith("OBSERVATION:\n")


def test_discards_codex_prose_and_stops_supervision_at_first_source_edit() -> None:
    row, report = render_verified_teacher_sft(_record())

    assert report["kept"] is True
    assert row is not None
    rendered = json.dumps(row)
    assert "Codex prose must disappear" not in rendered
    assert "I know the fix now" not in rendered
    assert "python -m pytest" not in rendered
    assert row["messages"][-1]["content"] == ""


def test_ignores_unparseable_commands_after_the_chosen_edit() -> None:
    record = _record()
    record["stream_lines"].append(
        _completed(
            {
                "type": "command_execution",
                "command": "host command after edit",
                "aggregated_output": "unused",
                "exit_code": 0,
            }
        )
    )

    row, report = render_verified_teacher_sft(record)

    assert report["kept"] is True
    assert row is not None


def test_uses_f2p_names_as_prompt_fallback_without_copying_gold_patch() -> None:
    record = _record()
    record["task"] = {
        **record["task"],
        "problem_statement": "",
        "FAIL_TO_PASS": ["tests/test_target.py::test_target"],
    }

    row, report = render_verified_teacher_sft(record)

    assert report["kept"] is True
    assert row is not None
    prompt = row["messages"][1]["content"]
    assert "tests/test_target.py::test_target" in prompt
    assert "return 'new'" not in prompt
    assert report["prompt_fallback_f2p"] is True


def test_recognizes_python_path_writer_as_source_edit() -> None:
    record = _record()
    record["stream_lines"] = [
        _command("sed -n '1,80p' pkg/target.py", "pkg/target.py\ndef target():"),
        _command(
            "python - <<'PY'\nfrom pathlib import Path\npath = Path('pkg/target.py')\n"
            "text = path.read_text()\npath.write_text(text.replace('old', 'new'))\nPY",
            "",
        ),
    ]

    row, report = render_verified_teacher_sft(record)

    assert report["kept"] is True
    assert row is not None
    assert _assistant_commands(row)[-1].startswith("python - <<'PY'")


def test_rejects_unresolved_or_test_modifying_teacher_results() -> None:
    unresolved = _record()
    unresolved["result"] = {**unresolved["result"], "resolved": False}
    row, report = render_verified_teacher_sft(unresolved)
    assert row is None
    assert report["reason"] == "not_f2p_resolved"

    test_patch = _record()
    test_patch["patch"] = PATCH.replace("pkg/target.py", "tests/test_target.py")
    row, report = render_verified_teacher_sft(test_patch)
    assert row is None
    assert report["reason"] == "non_source_patch"


def test_rejects_ungrounded_or_unbalanced_streams() -> None:
    ungrounded = _record()
    ungrounded["stream_lines"] = [
        _command("ls", "README.md"),
        _command("sed -i \"s/'old'/'new'/\" pkg/target.py", ""),
    ]
    row, report = render_verified_teacher_sft(ungrounded)
    assert row is None
    assert report["reason"] == "edited_file_not_seen"

    malformed = _record()
    malformed["stream_lines"] = [
        _command("sed -n '1,80p' pkg/target.py", "pkg/target.py"),
        _completed(
            {
                "type": "command_execution",
                "command": "echo host-command",
                "aggregated_output": "bad",
                "exit_code": 0,
            }
        ),
        _command("sed -i \"s/'old'/'new'/\" pkg/target.py", ""),
    ]
    row, report = render_verified_teacher_sft(malformed)
    assert row is None
    assert report["reason"] == "unparseable_command"


def test_requires_a_source_path_in_the_actual_edit_command() -> None:
    record = _record()
    record["stream_lines"] = [
        _command("sed -n '1,80p' pkg/target.py", "pkg/target.py"),
        _command("sed -i 's/old/new/' pkg/other.py", ""),
    ]

    row, report = render_verified_teacher_sft(record)

    assert row is None
    assert report["reason"] == "no_matching_source_edit"


def test_build_selects_one_best_attempt_per_task_and_filters_token_budget() -> None:
    early = _record()
    early["record_id"] = "early"
    late = _record()
    late["record_id"] = "late"
    late["stream_lines"] = [
        _command("ls", "pkg"),
        _command("pwd", "/testbed"),
        *late["stream_lines"],
    ]
    oversized = _record()
    oversized["record_id"] = "oversized"
    oversized["task"] = {
        "instance_id": "task-b",
        "problem_statement": "OVERSIZED Fix target() in pkg/target.py.",
    }
    oversized["result"] = {**oversized["result"], "instance_id": "task-b"}

    rows, manifest, rejected = build_verified_teacher_rows(
        [late, early, oversized],
        token_counter=lambda messages: 100 if "OVERSIZED" in json.dumps(messages) else 10,
        max_tokens=50,
    )

    assert [row["source_instance_id"] for row in rows] == ["task-a"]
    assert manifest["records_in"] == 3
    assert manifest["unique_tasks_selected"] == 2
    assert manifest["rows_kept"] == 1
    assert manifest["budget_rejected"] == 1
    assert manifest["superseded_verified_attempts"] == 1
    assert manifest["per_repo"] == {"repo-a": 1}
    assert manifest["prompt_fallback_f2p_rows"] == 0
    assert manifest["original_edit_command_index_histogram"] == {"4": 1}
    reasons = {item["record_id"]: item["reason"] for item in rejected}
    assert reasons == {
        "late": "superseded_verified_attempt",
        "oversized": "token_budget_exceeded",
    }


def test_build_deduplicates_identical_message_content() -> None:
    first = _record()
    first["record_id"] = "first"
    duplicate = _record()
    duplicate["record_id"] = "duplicate"
    duplicate["task"] = {**duplicate["task"], "instance_id": "task-b"}
    duplicate["result"] = {**duplicate["result"], "instance_id": "task-b"}

    rows, manifest, rejected = build_verified_teacher_rows(
        [first, duplicate],
        token_counter=lambda messages: 10,
    )

    assert len(rows) == 1
    assert manifest["content_duplicates"] == 1
    assert rejected[-1]["reason"] == "content_duplicate"


def test_loads_auditable_records_from_task_pools_and_run_ledgers(tmp_path) -> None:
    task_path = tmp_path / "tasks.jsonl"
    task_path.write_text(json.dumps(_record()["task"]) + "\n")
    run_dir = tmp_path / "teacher-run"
    run_dir.mkdir()
    run_dir.joinpath("results.jsonl").write_text(json.dumps(_record()["result"]) + "\n")
    run_dir.joinpath("task-a.patch").write_text(PATCH)
    run_dir.joinpath("task-a.stream.jsonl").write_text("\n".join(_record()["stream_lines"]) + "\n")

    records = load_teacher_records(
        [task_path],
        [run_dir],
        system_template="native system",
        instance_template="PR: {{task}}",
    )

    assert len(records) == 1
    assert records[0]["task"]["instance_id"] == "task-a"
    assert records[0]["result"]["resolved"] is True
    assert records[0]["patch"] == PATCH
    assert len(records[0]["stream_lines"]) == len(_record()["stream_lines"])
    assert records[0]["record_id"].endswith(":task-a")
    assert records[0]["provenance"]["run_dir"] == str(run_dir)


@pytest.mark.parametrize("bad_path", ["setup.py", "pyproject.toml", "docs/target.py"])
def test_rejects_non_runtime_python_paths(bad_path: str) -> None:
    record = _record()
    record["patch"] = PATCH.replace("pkg/target.py", bad_path)

    row, report = render_verified_teacher_sft(record)

    assert row is None
    assert report["reason"] == "non_source_patch"
