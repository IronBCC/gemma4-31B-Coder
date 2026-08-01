from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from phaseD_sft.build_v2p11_recovery_curriculum import (
    RECOVERY_STYLES,
    build_recovery_curriculum,
)


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")


def _assistant(command: str, *, call_id: str) -> dict:
    return {
        "role": "assistant",
        "content": "",
        "loss": True,
        "tool_calls": [
            {
                "id": call_id,
                "type": "function",
                "function": {
                    "name": "bash",
                    "arguments": json.dumps({"command": command}),
                },
            }
        ],
    }


def _tool(content: str, *, call_id: str) -> dict:
    return {
        "role": "tool",
        "tool_call_id": call_id,
        "content": content,
    }


def _source_dataset(tmp_path: Path, *, passing_test: bool = True) -> tuple[Path, Path]:
    exclusion = tmp_path / "exclusions.json"
    _write_json(
        exclusion,
        {
            "instance_ids": ["eval-a", "eval-b"],
            "repo_denylist": ["eval/repo"],
        },
    )
    source = tmp_path / "source"
    source.mkdir()
    row = {
        "instance_id": "train-task",
        "repo": "safe/repo",
        "source": "teacher:claude:claude-fable-5",
        "messages": [
            {
                "role": "system",
                "content": (
                    "Fix the bug and echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT."
                ),
                "tool_calls": [],
            },
            {
                "role": "user",
                "content": (
                    "<instructions>Inspect, edit, then run "
                    "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT.</instructions>"
                ),
                "tool_calls": [],
            },
            _assistant("sed -n '1,80p' src/mod.py", call_id="read"),
            _tool("<returncode>0</returncode>\n<output>old</output>", call_id="read"),
            _assistant(
                (
                    "python - <<'PY'\n"
                    "p='src/mod.py'\n"
                    "s=open(p).read()\n"
                    "open(p,'w').write(s.replace('old','new'))\n"
                    "PY"
                ),
                call_id="edit",
            ),
            _tool("<returncode>0</returncode>\n<output></output>", call_id="edit"),
            _assistant("python -m pytest tests/test_mod.py -q", call_id="test"),
            _tool(
                (
                    f"<returncode>{0 if passing_test else 1}</returncode>\n"
                    "<output>1 passed</output>"
                ),
                call_id="test",
            ),
            {
                "role": "assistant",
                "content": "The focused test passes and the source diff is ready.",
                "loss": True,
                "tool_calls": [],
            },
        ],
    }
    train = source / "train.jsonl"
    train.write_text(json.dumps(row, sort_keys=True) + "\n", encoding="utf-8")
    _write_json(
        source / "manifest.json",
        {
            "schema_version": 2,
            "kind": "teacher_blend",
            "rendered": 1,
            "training_admitted": 1,
            "all_training_gates_complete": True,
            "train_jsonl_sha256": hashlib.sha256(train.read_bytes()).hexdigest(),
            "standard_native_format_loss_gate": {
                "status": "passed",
                "samples": 1,
                "failure_count": 0,
            },
        },
    )
    return source, exclusion


def test_builds_three_portable_styles_with_only_edit_and_passing_test_supervised(
    tmp_path: Path,
) -> None:
    source, exclusion = _source_dataset(tmp_path)
    output = tmp_path / "output"

    result = build_recovery_curriculum(
        source=source,
        output=output,
        exclusion_path=exclusion,
    )

    rows = [
        json.loads(line)
        for line in (output / "train.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert result["rows"] == len(RECOVERY_STYLES) == 3
    assert {row["augmentation_style"] for row in rows} == set(RECOVERY_STYLES)
    assert len({row["instance_id"] for row in rows}) == 3
    assert {row["source_instance_id"] for row in rows} == {"train-task"}
    assert manifest["all_training_gates_complete"] is False
    assert manifest["training_admitted"] == 0
    assert manifest["source_rows"] == 1
    assert manifest["rendered"] == 3
    assert manifest["target_counts"] == {
        "rows_with_passing_test_target": 3,
        "rows_with_source_mutation_target": 3,
        "supervised_assistant_messages": 6,
    }
    for row in rows:
        serialized = json.dumps(row)
        assert "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT" not in serialized
        assistants = [
            message
            for message in row["messages"]
            if message.get("role") == "assistant"
        ]
        supervised = [
            message for message in assistants if message.get("loss") is True
        ]
        assert len(supervised) == 2
        commands = [
            json.loads(message["tool_calls"][0]["function"]["arguments"])["command"]
            for message in supervised
        ]
        assert "src/mod.py" in commands[0]
        assert "pytest" in commands[1]


def test_rejects_a_row_without_a_passing_focused_test(tmp_path: Path) -> None:
    source, exclusion = _source_dataset(tmp_path, passing_test=False)

    with pytest.raises(ValueError, match="passing focused test"):
        build_recovery_curriculum(
            source=source,
            output=tmp_path / "output",
            exclusion_path=exclusion,
        )


def test_accepts_legacy_user_observations_and_terminal_verified_git_apply(
    tmp_path: Path,
) -> None:
    source, exclusion = _source_dataset(tmp_path)
    row = json.loads((source / "train.jsonl").read_text(encoding="utf-8"))
    row["messages"][5] = {
        "role": "user",
        "content": (
            "OBSERVATION:\n<returncode>0</returncode>\n<output></output>"
        ),
        "tool_calls": [],
    }
    row["messages"][7] = {
        "role": "user",
        "content": (
            "OBSERVATION:\n<returncode>0</returncode>\n"
            "<output>1 passed</output>"
        ),
        "tool_calls": [],
    }
    row["messages"][-1] = _assistant(
        (
            "git apply - <<'PATCH'\n"
            "diff --git a/src/mod.py b/src/mod.py\n"
            "PATCH"
        ),
        call_id="verified-correction",
    )
    row["messages"].extend(
        [
            {
                "role": "user",
                "content": (
                    "OBSERVATION:\n<returncode>0</returncode>\n<output></output>"
                ),
                "tool_calls": [],
            },
            _assistant(
                "python -m pytest tests/test_mod.py -q",
                call_id="post-correction-test",
            ),
            {
                "role": "user",
                "content": (
                    "OBSERVATION:\n<returncode>0</returncode>\n"
                    "<output>1 passed</output>"
                ),
                "tool_calls": [],
            },
        ]
    )
    train = source / "train.jsonl"
    train.write_text(json.dumps(row, sort_keys=True) + "\n", encoding="utf-8")
    manifest = json.loads((source / "manifest.json").read_text(encoding="utf-8"))
    manifest["train_jsonl_sha256"] = hashlib.sha256(train.read_bytes()).hexdigest()
    _write_json(source / "manifest.json", manifest)

    result = build_recovery_curriculum(
        source=source,
        output=tmp_path / "output",
        exclusion_path=exclusion,
    )

    assert result["rows"] == 3


def test_rejects_passing_test_before_verifier_correction(
    tmp_path: Path,
) -> None:
    source, exclusion = _source_dataset(tmp_path)
    row = json.loads((source / "train.jsonl").read_text(encoding="utf-8"))
    row["messages"] = row["messages"][:-1]
    row["messages"].extend(
        [
            {
                "role": "user",
                "content": (
                    "VERIFIER FEEDBACK:\nThe legacy candidate does not repair "
                    "the bug-mutated task."
                ),
                "tool_calls": [],
            },
            _assistant(
                (
                    "git apply - <<'PATCH'\n"
                    "diff --git a/src/mod.py b/src/mod.py\n"
                    "PATCH"
                ),
                call_id="verified-correction",
            ),
        ]
    )
    train = source / "train.jsonl"
    train.write_text(json.dumps(row, sort_keys=True) + "\n", encoding="utf-8")
    manifest = json.loads((source / "manifest.json").read_text(encoding="utf-8"))
    manifest["train_jsonl_sha256"] = hashlib.sha256(train.read_bytes()).hexdigest()
    _write_json(source / "manifest.json", manifest)

    with pytest.raises(
        ValueError,
        match="passing focused test after source mutation",
    ):
        build_recovery_curriculum(
            source=source,
            output=tmp_path / "output",
            exclusion_path=exclusion,
        )


def test_rejects_passing_test_before_later_successful_source_mutation(
    tmp_path: Path,
) -> None:
    source, exclusion = _source_dataset(tmp_path)
    row = json.loads((source / "train.jsonl").read_text(encoding="utf-8"))
    row["messages"] = row["messages"][:-1]
    row["messages"].extend(
        [
            {
                "role": "user",
                "content": "VERIFIER FEEDBACK:\nApply the verified correction.",
                "tool_calls": [],
            },
            _assistant(
                (
                    "git apply - <<'PATCH'\n"
                    "diff --git a/src/mod.py b/src/mod.py\n"
                    "PATCH"
                ),
                call_id="verified-correction",
            ),
            _tool(
                "<returncode>0</returncode>\n<output></output>",
                call_id="verified-correction",
            ),
            _assistant(
                "python -m pytest tests/test_mod.py -q",
                call_id="post-correction-test",
            ),
            _tool(
                "<returncode>0</returncode>\n<output>1 passed</output>",
                call_id="post-correction-test",
            ),
            _assistant(
                "sed -i 's/new/newer/' src/mod.py",
                call_id="later-edit",
            ),
            _tool(
                "<returncode>0</returncode>\n<output></output>",
                call_id="later-edit",
            ),
        ]
    )
    train = source / "train.jsonl"
    train.write_text(json.dumps(row, sort_keys=True) + "\n", encoding="utf-8")
    manifest = json.loads((source / "manifest.json").read_text(encoding="utf-8"))
    manifest["train_jsonl_sha256"] = hashlib.sha256(train.read_bytes()).hexdigest()
    _write_json(source / "manifest.json", manifest)

    with pytest.raises(
        ValueError,
        match="passing focused test after source mutation",
    ):
        build_recovery_curriculum(
            source=source,
            output=tmp_path / "output",
            exclusion_path=exclusion,
        )


def test_accepts_a_passing_test_invoked_through_a_shell_python_variable(
    tmp_path: Path,
) -> None:
    source, exclusion = _source_dataset(tmp_path)
    row = json.loads((source / "train.jsonl").read_text(encoding="utf-8"))
    row["messages"][6]["tool_calls"][0]["function"]["arguments"] = json.dumps(
        {
            "command": (
                "P=/opt/miniconda3/envs/testbed/bin/python && "
                "$P -m pytest tests/test_mod.py -q"
            )
        }
    )
    row["messages"][7] = {
        "role": "user",
        "content": "OBSERVATION:\n1 passed in 0.01s",
        "tool_calls": [],
    }
    train = source / "train.jsonl"
    train.write_text(json.dumps(row, sort_keys=True) + "\n", encoding="utf-8")
    manifest = json.loads((source / "manifest.json").read_text(encoding="utf-8"))
    manifest["train_jsonl_sha256"] = hashlib.sha256(train.read_bytes()).hexdigest()
    _write_json(source / "manifest.json", manifest)

    result = build_recovery_curriculum(
        source=source,
        output=tmp_path / "output",
        exclusion_path=exclusion,
    )

    assert result["rows"] == 3
