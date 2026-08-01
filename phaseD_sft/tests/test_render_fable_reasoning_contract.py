from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from phaseD_sft import render_fable_reasoning_contract as renderer


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _bash(command: str) -> list[dict[str, object]]:
    return [{
        "id": "tool-1",
        "type": "function",
        "function": {"name": "bash", "arguments": json.dumps({"command": command})},
    }]


def _row(
    *,
    instance_id: str,
    source: str,
    first_content: str,
    command: str,
) -> dict[str, object]:
    return {
        "instance_id": instance_id,
        "repo": "example/repo",
        "source": source,
        "messages": [
            {"role": "system", "content": "system", "loss": False, "tool_calls": []},
            {
                "role": "assistant",
                "content": first_content,
                "loss": True,
                "tool_calls": _bash(command),
            },
        ],
    }


def _source_dataset(tmp_path: Path) -> tuple[Path, Path]:
    source = tmp_path / "source"
    rows = [
        _row(
            instance_id="fable__case-1",
            source="teacher:claude:claude-fable-5",
            first_content="",
            command="pytest tests/test_bug.py -q",
        ),
        _row(
            instance_id="base__case-2",
            source="teacher:open-swe:sweagent",
            first_content="",
            command="git diff -- src/module.py",
        ),
    ]
    train = source / "train.jsonl"
    _write_jsonl(train, rows)
    _write_json(
        source / "manifest.json",
        {
            "schema_version": 2,
            "complete": True,
            "dataset_variant": "teacher_train_mix_v2p11_fable_extension",
            "rendered": len(rows),
            "training_admitted": len(rows),
            "all_training_gates_complete": True,
            "train_jsonl_sha256": _sha(train),
            "canonical_rows_sha256": renderer.canonical_rows_sha256(rows),
            "evaluation_overlap": 0,
        },
    )
    lite_ids = tmp_path / "lite_ids.json"
    _write_json(lite_ids, ["django__django-999"])
    return source, lite_ids


def test_renders_blank_fable_tool_turn_with_command_matched_reasoning() -> None:
    row = _row(
        instance_id="fable__case-1",
        source="teacher:claude:claude-fable-5",
        first_content="",
        command="pytest tests/test_bug.py -q",
    )

    rendered, stats = renderer.render_fable_row(row)

    message = rendered["messages"][1]
    assert message["content"] == (
        "I will run the focused test to validate the current approach."
    )
    assert message["tool_calls"] == row["messages"][1]["tool_calls"]
    assert stats == {"canonical_bash_tool_turns": 1, "reasoned_tool_turns": 1}


def test_preserves_non_fable_row_byte_for_byte() -> None:
    row = _row(
        instance_id="base__case-2",
        source="teacher:open-swe:sweagent",
        first_content="",
        command="git diff -- src/module.py",
    )

    rendered, stats = renderer.render_fable_row(row)

    assert rendered == row
    assert stats == {"canonical_bash_tool_turns": 0, "reasoned_tool_turns": 0}


def test_rejects_fable_tool_call_that_is_not_canonical_bash() -> None:
    row = _row(
        instance_id="fable__case-1",
        source="teacher:claude:claude-fable-5",
        first_content="",
        command="pwd",
    )
    row["messages"][1]["tool_calls"][0]["function"]["name"] = "shell"

    with pytest.raises(ValueError, match="canonical bash"):
        renderer.render_fable_row(row)


def test_build_and_seal_bind_the_source_and_full_format_gate(tmp_path: Path) -> None:
    source, lite_ids = _source_dataset(tmp_path)
    output = tmp_path / "output"

    manifest = renderer.build_reasoned_mix(
        source=source,
        lite_ids=lite_ids,
        out=output,
    )

    assert manifest["all_training_gates_complete"] is False
    assert manifest["fable_rows"] == 1
    assert manifest["reasoned_tool_turns"] == 1
    rows = [json.loads(line) for line in (output / "train.jsonl").read_text().splitlines()]
    assert rows[0]["messages"][1]["content"].startswith("I will run the focused test")
    assert rows[1]["messages"][1]["content"] == ""

    report = tmp_path / "format-report.json"
    _write_json(report, {"data": str(output), "samples": 2, "failure_count": 0})

    sealed = renderer.seal_reasoned_mix(out=output, format_report=report)

    assert sealed["all_training_gates_complete"] is True
    assert sealed["format_loss_gate"]["failure_count"] == 0
    assert sealed["format_loss_gate"]["samples"] == 2
