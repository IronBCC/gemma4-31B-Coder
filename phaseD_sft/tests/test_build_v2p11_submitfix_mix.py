from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

import phaseD_sft.build_v2p11_submitfix_mix as submitfix


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(
            json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n"
            for row in rows
        )
    )


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _tool_turn(command: str) -> dict[str, object]:
    return {
        "role": "assistant",
        "content": "",
        "loss": True,
        "tool_calls": [
            {
                "id": "call-1",
                "type": "function",
                "function": {
                    "name": "bash",
                    "arguments": json.dumps({"command": command}),
                },
            }
        ],
    }


def _fixture(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[Path, list[dict[str, object]]]:
    source = tmp_path / "source"
    rows: list[dict[str, object]] = [
        {
            "instance_id": "base-one",
            "source": "v2p10",
            "messages": [
                {"role": "system", "content": "system", "loss": False},
                _tool_turn("sed -i 's/a/b/' src/base.py"),
            ],
        },
        {
            "instance_id": "base-two",
            "source": "v2p10",
            "messages": [
                {"role": "system", "content": "system", "loss": False},
                _tool_turn("git diff -- src/base.py"),
            ],
        },
        {
            "instance_id": "stage-one",
            "source": "teacher:claude:claude-fable-5",
            "messages": [
                {"role": "system", "content": "system", "loss": False},
                _tool_turn("sed -i 's/old/new/' src/module.py"),
                {"role": "user", "content": "1 passed", "loss": False},
                {
                    "role": "assistant",
                    "content": "The focused test passes. DONE",
                    "loss": True,
                    "tool_calls": [],
                },
            ],
        },
        {
            "instance_id": "late-one",
            "source": "fable5_verified_finalpatch",
            "messages": [
                {"role": "system", "content": "system", "loss": False},
                _tool_turn("git apply - <<'PATCH'\ndiff --git a/a.py b/a.py\nPATCH"),
            ],
        },
    ]
    train = source / "train.jsonl"
    _write_jsonl(train, rows)
    manifest = source / "manifest.json"
    _write_json(
        manifest,
        {
            "schema_version": 2,
            "complete": True,
            "dataset_variant": "teacher_train_mix_v2p11_fable_extension",
            "rendered": 4,
            "training_admitted": 4,
            "all_training_gates_complete": True,
            "evaluation_overlap": 0,
            "train_jsonl_sha256": _sha(train),
        },
    )
    monkeypatch.setattr(submitfix, "PRODUCTION_ROWS", 4)
    monkeypatch.setattr(submitfix, "PRODUCTION_BASE_ROWS", 2)
    monkeypatch.setattr(submitfix, "PRODUCTION_STAGE_ROWS", 1)
    monkeypatch.setattr(submitfix, "PRODUCTION_LATE_ROWS", 1)
    monkeypatch.setattr(
        submitfix,
        "PRODUCTION_SOURCE_MANIFEST_SHA256",
        _sha(manifest),
    )
    monkeypatch.setattr(
        submitfix,
        "PRODUCTION_SOURCE_TRAIN_SHA256",
        _sha(train),
    )
    return source, rows


def test_replace_terminal_with_submit_uses_harness_command() -> None:
    row = {
        "instance_id": "stage-one",
        "messages": [
            {"role": "user", "content": "1 passed", "loss": False},
            {
                "role": "assistant",
                "content": "Done",
                "loss": True,
                "tool_calls": [],
            },
        ],
    }

    repaired = submitfix.replace_terminal_with_submit(row, call_id="submit-1")

    assert row["messages"][-1]["content"] == "Done"
    terminal = repaired["messages"][-1]
    assert terminal["role"] == "assistant"
    assert terminal["content"] == ""
    assert terminal["loss"] is True
    assert terminal["tool_calls"][0]["id"] == "submit-1"
    arguments = json.loads(
        terminal["tool_calls"][0]["function"]["arguments"]
    )
    assert arguments == {"command": submitfix.SUBMIT_COMMAND}


def test_replace_terminal_rejects_non_prose_target() -> None:
    row = {
        "instance_id": "stage-one",
        "messages": [_tool_turn("git diff")],
    }

    with pytest.raises(ValueError, match="supervised prose"):
        submitfix.replace_terminal_with_submit(row, call_id="submit-1")


def test_build_repairs_only_stage_a_terminals(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source, source_rows = _fixture(tmp_path, monkeypatch)
    out = tmp_path / "submitfix"

    manifest = submitfix.build_submitfix_mix(source=source, out=out)

    output_rows = [
        json.loads(line)
        for line in (out / "train.jsonl").read_text().splitlines()
        if line
    ]
    assert manifest["dataset_variant"] == "teacher_train_mix_v2p11_submitfix_v1"
    assert manifest["rendered"] == 4
    assert manifest["terminal_repairs"] == 1
    assert manifest["unchanged_late_rows"] == 1
    assert manifest["training_admitted"] == 0
    assert manifest["all_training_gates_complete"] is False
    assert output_rows[:2] == source_rows[:2]
    assert output_rows[3] == source_rows[3]
    assert output_rows[2]["messages"][:-1] == source_rows[2]["messages"][:-1]
    assert output_rows[2]["messages"][-1]["tool_calls"]
    assert output_rows[2]["messages"][-1]["loss"] is True


def test_build_rejects_non_boolean_loss(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source, rows = _fixture(tmp_path, monkeypatch)
    rows[2]["messages"][0]["loss"] = 0
    _write_jsonl(source / "train.jsonl", rows)
    manifest_path = source / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["train_jsonl_sha256"] = _sha(source / "train.jsonl")
    _write_json(manifest_path, manifest)
    monkeypatch.setattr(
        submitfix,
        "PRODUCTION_SOURCE_TRAIN_SHA256",
        _sha(source / "train.jsonl"),
    )
    monkeypatch.setattr(
        submitfix,
        "PRODUCTION_SOURCE_MANIFEST_SHA256",
        _sha(manifest_path),
    )

    with pytest.raises(ValueError, match="boolean loss"):
        submitfix.build_submitfix_mix(
            source=source,
            out=tmp_path / "submitfix",
        )


def test_seal_requires_full_format_report(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source, _rows = _fixture(tmp_path, monkeypatch)
    out = tmp_path / "submitfix"
    submitfix.build_submitfix_mix(source=source, out=out)
    report = tmp_path / "format.json"
    _write_json(
        report,
        {
            "data": str(out.resolve()),
            "samples": 4,
            "failure_count": 0,
            "fallback_spans": 0,
        },
    )

    manifest = submitfix.seal_submitfix_mix(
        out=out,
        format_report=report,
    )

    assert manifest["training_admitted"] == 4
    assert manifest["all_training_gates_complete"] is True
    gate = manifest["standard_native_format_loss_gate"]
    assert gate["status"] == "passed"
    assert gate["failure_count"] == 0
    assert gate["samples"] == 4
    assert gate["report"]["sha256"] == _sha(
        out / gate["report"]["path"].split("/")[-1]
    )
