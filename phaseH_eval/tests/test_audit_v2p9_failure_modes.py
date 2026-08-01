from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from phaseH_eval.audit_v2p9_failure_modes import (
    artifact_binding,
    audit_rows,
    audit_run,
    extract_command_outcomes,
    validate_artifact_binding,
)


def _assistant(command: str, call_id: str) -> dict:
    return {
        "role": "assistant",
        "tool_calls": [{
            "id": call_id,
            "type": "function",
            "function": {
                "name": "bash",
                "arguments": json.dumps({"command": command}),
            },
        }],
    }


def _tool(returncode: int, call_id: str, output: str = "") -> dict:
    return {
        "role": "tool",
        "tool_call_id": call_id,
        "content": (
            f"<returncode>{returncode}</returncode>\n"
            f"<output>{output}</output>"
        ),
    }


def _trajectory(instance_id: str, messages: list[dict]) -> dict:
    return {
        "instance_id": instance_id,
        "messages": messages,
        "info": {"exit_status": "Submitted"},
    }


def test_audit_rows_partitions_outcomes_and_measures_wrong_edit_loops() -> None:
    predictions = {
        "resolved": {"instance_id": "resolved", "model_patch": "diff --git a/x.py b/x.py\n"},
        "empty": {"instance_id": "empty", "model_patch": ""},
        "wrong": {"instance_id": "wrong", "model_patch": "diff --git a/y.py b/y.py\n"},
    }
    repeated = "sed -i 's/old/new/' src/y.py"
    trajectories = {
        "resolved": _trajectory("resolved", []),
        "empty": _trajectory("empty", []),
        "wrong": _trajectory(
            "wrong",
            [
                _assistant(repeated, "one"),
                _tool(1, "one", "bad edit"),
                _assistant(repeated, "two"),
                _tool(1, "two", "bad edit"),
                {"role": "user", "content": "Tool call error: malformed arguments"},
            ],
        ),
    }

    report = audit_rows(
        predictions,
        resolved_ids={"resolved"},
        trajectories=trajectories,
        expected_ids=["resolved", "empty", "wrong", "missing"],
        scored_ids={"resolved", "empty", "wrong"},
    )

    assert report["training_eligible"] is False
    assert report["counts"] == {
        "expected": 4,
        "usable": 3,
        "resolved": 1,
        "empty": 1,
        "wrong_nonempty": 1,
        "missing_prediction": 1,
        "missing_trajectory": 0,
        "missing_score": 0,
    }
    wrong = next(row for row in report["rows"] if row["instance_id"] == "wrong")
    assert wrong["category"] == "wrong_nonempty"
    assert wrong["training_eligible"] is False
    assert wrong["failure_features"] == {
        "uses_sed_i": True,
        "standard_test_run": False,
        "nonzero_commands": 2,
        "max_exact_repeat": 2,
        "tool_format_errors": 1,
        "command_count": 2,
    }


def test_audit_rows_classifies_empty_prediction_before_absent_score() -> None:
    predictions = {
        "empty": {"instance_id": "empty", "model_patch": ""},
    }
    trajectories = {
        "empty": _trajectory("empty", []),
    }

    report = audit_rows(
        predictions,
        resolved_ids=set(),
        trajectories=trajectories,
        expected_ids=["empty"],
        scored_ids=set(),
    )

    assert report["counts"]["empty"] == 1
    assert report["counts"]["missing_score"] == 0
    assert report["counts"]["usable"] == 1
    assert report["rows"][0]["score_present"] is False


def test_extract_command_outcomes_requires_one_adjacent_matching_call() -> None:
    nonadjacent = [
        _assistant("python reproduce.py", "one"),
        {"role": "assistant", "content": "inspect"},
        _tool(1, "one", "failure"),
    ]
    multiple = _assistant("python one.py", "one")
    multiple["tool_calls"].extend(_assistant("python two.py", "two")["tool_calls"])

    assert extract_command_outcomes(nonadjacent) == []
    assert extract_command_outcomes([multiple, _tool(1, "one", "failure")]) == []


def test_artifact_binding_detects_changed_input(tmp_path: Path) -> None:
    trajectory = tmp_path / "one.traj.json"
    trajectory.write_text('{"instance_id":"one","messages":[]}\n')
    binding = artifact_binding([trajectory])

    trajectory.write_text('{"instance_id":"one","messages":[{}]}\n')

    with pytest.raises(ValueError, match="artifact changed"):
        validate_artifact_binding(binding)


def test_audit_run_binds_inputs_publishes_atomically_and_refuses_overwrite(
    tmp_path: Path,
) -> None:
    root = tmp_path / "repo"
    run = root / "runs" / "v2p9_full300"
    model_dir = run / "b00" / "teacher_sft_v2p9"
    report_dir = run / "b00" / "report"
    data_dir = root / "data"
    model_dir.mkdir(parents=True)
    report_dir.mkdir(parents=True)
    data_dir.mkdir()
    ids = ["resolved", "wrong", "missing"]
    ids_path = data_dir / "swebench_lite_test_ids.json"
    ids_path.write_text(json.dumps(ids) + "\n")
    ids_sha = hashlib.sha256(ids_path.read_bytes()).hexdigest()
    (run / "eval_manifest.json").write_text(json.dumps({
        "schema_version": 1,
        "run_id": "v2p9_full300",
        "served_name": "teacher_sft_v2p9",
        "ids_path": "data/swebench_lite_test_ids.json",
        "ids_sha256": ids_sha,
        "instances": 3,
    }))
    (run / "acceptance.json").write_text(json.dumps({
        "status": "incomplete",
        "expected": 3,
        "preds": 2,
        "traj_files": 2,
        "usable_outcomes": 2,
        "resolved": 1,
        "pull_failed": 1,
    }))
    (run / "summary.json").write_text(json.dumps({
        "instances": 3,
        "scored": 2,
        "resolved": ["resolved"],
        "empty": [],
        "pull_failed": ["fixture/image:latest"],
    }))
    (run / "preds_all.json").write_text(json.dumps({
        "resolved": {
            "instance_id": "resolved",
            "model_patch": "diff --git a/x.py b/x.py\n",
        },
        "wrong": {
            "instance_id": "wrong",
            "model_patch": "diff --git a/y.py b/y.py\n",
        },
    }))
    (report_dir / "official_report.json").write_text(json.dumps({
        "resolved_ids": ["resolved"],
        "unresolved_ids": ["wrong"],
        "error_ids": [],
    }))
    for instance_id in ("resolved", "wrong"):
        case = model_dir / instance_id
        case.mkdir()
        (case / f"{instance_id}.traj.json").write_text(
            json.dumps(_trajectory(instance_id, []))
        )
    output = root / "runs" / "v2p9_failure_audit"

    report = audit_run(run, output)

    assert report["counts"]["wrong_nonempty"] == 1
    assert report["counts"]["missing_prediction"] == 1
    assert report["source_run_status"] == "incomplete"
    assert (output / "audit.json").is_file()
    assert (output / "audit.md").is_file()
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["complete"] is True
    assert manifest["training_eligible"] is False
    bound_paths = {Path(row["path"]).name for row in manifest["input_artifacts"]}
    assert {
        "acceptance.json",
        "summary.json",
        "preds_all.json",
        "official_report.json",
        "resolved.traj.json",
        "wrong.traj.json",
        "swebench_lite_test_ids.json",
        "eval_manifest.json",
    } <= bound_paths
    assert not Path(str(output) + ".work").exists()

    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        audit_run(run, output)
