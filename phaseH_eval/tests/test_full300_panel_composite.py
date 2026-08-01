from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from phaseH_eval.full300_panel_composite import (
    PanelInput,
    join_disjoint_panels,
    verify_disjoint_panels,
)


MODEL_CONTRACT = {
    "served_name": "teacher_sft_v2p10",
    "model_path": "/models/teacher_sft_v2p10_full",
    "model_config_sha256": "a" * 64,
    "model_index_sha256": "b" * 64,
    "model_artifacts": [],
}
HARNESS_CONTRACT = {
    "batch": 20,
    "workers": 16,
    "pull_workers": 5,
    "config": "swebench_edit_first_selfretry_s120.yaml",
    "temperature": 0.7,
    "seed": 1,
    "step_limit": 120,
    "environment_class": "docker_selfretry.DockerSelfRetryEnv",
}


def _write_panel(
    root: Path,
    tag: str,
    ids: list[str],
    *,
    resolved: set[str],
    empty: set[str],
    model_contract: dict | None = None,
) -> PanelInput:
    ids_path = root / f"{tag}_ids.json"
    ids_path.write_text(json.dumps(ids) + "\n")
    preds = {
        instance_id: {
            "instance_id": instance_id,
            "model_name_or_path": "openai/teacher_sft_v2p10",
            "model_patch": (
                ""
                if instance_id in empty
                else f"diff --git a/{instance_id} b/{instance_id}\n"
            ),
        }
        for instance_id in ids
    }
    preds_path = root / f"{tag}_preds.json"
    preds_path.write_text(json.dumps(preds) + "\n")
    predictions_artifact = {
        "path": str(preds_path.resolve()),
        "sha256": hashlib.sha256(preds_path.read_bytes()).hexdigest(),
        "bytes": preds_path.stat().st_size,
    }
    empty_ids = sorted(empty)
    behavior = {
        instance_id: {
            "repeat_loops": 0,
            "tool_format_errors": 0,
            "assistant_responses": 3,
        }
        for instance_id in ids
    }
    composite = {
        "schema_version": 1,
        "artifact_type": "empty_patch_retry_composite",
        "status": "complete",
        "run_id": f"{tag}-source+{tag}-retry",
        "name": "teacher_sft_v2p10",
        "expected": len(ids),
        "preds": len(ids),
        "traj_files": len(ids),
        "unique_trajectories": len(ids),
        "usable_outcomes": len(ids),
        "pull_failed": 0,
        "docker_failed": 0,
        "resolved": len(resolved),
        "resolved_ids": sorted(resolved),
        "empty_split": {
            "model": empty_ids,
            "harness_forced": [],
            "docker_failed": [],
            "unknown_missing": [],
            "total": len(empty_ids),
        },
        "empty_resampling": {
            "attempted": 1,
            "became_nonempty": 0 if empty else 1,
            "became_resolved": 0 if empty else 1,
            "still_empty": len(empty_ids),
            "nonempty_rate": 0.0 if empty else 1.0,
            "resolved_rate": 0.0 if empty else 1.0,
        },
        "selected_attempts": {
            "source": len(ids) - 1,
            "empty_retry": 1,
        },
        "behavior_health_by_instance": behavior,
        "behavior_health": {
            "repeat_loops": 0,
            "tool_format_errors": 0,
            "assistant_responses": len(ids) * 3,
            "format_error_rate": 0.0,
        },
        "model_contract": model_contract or MODEL_CONTRACT,
        "harness_contract": HARNESS_CONTRACT,
        "predictions_artifact": predictions_artifact,
        "input_artifacts": [
            {
                "path": str(ids_path.resolve()),
                "sha256": hashlib.sha256(
                    ids_path.read_bytes()
                ).hexdigest(),
                "bytes": ids_path.stat().st_size,
            },
        ],
    }
    composite_path = root / f"{tag}_composite.json"
    composite_path.write_text(json.dumps(composite) + "\n")
    return PanelInput(
        tag=tag,
        ids_path=ids_path,
        composite_path=composite_path,
        predictions_path=preds_path,
    )


def test_join_disjoint_panels_preserves_full_id_order_and_contracts(
    tmp_path: Path,
) -> None:
    full_ids_path = tmp_path / "full_ids.json"
    full_ids_path.write_text(json.dumps(["b", "a", "d", "c"]) + "\n")
    fixed = _write_panel(
        tmp_path,
        "fixed",
        ["a", "b"],
        resolved={"a"},
        empty={"b"},
    )
    complement = _write_panel(
        tmp_path,
        "complement",
        ["c", "d"],
        resolved={"c", "d"},
        empty=set(),
    )
    output = tmp_path / "full_composite.json"
    predictions = tmp_path / "full_preds.json"

    composite = join_disjoint_panels(
        full_ids_path=full_ids_path,
        panels=[fixed, complement],
        output_path=output,
        predictions_path=predictions,
    )

    assert composite["status"] == "complete"
    assert composite["expected"] == 4
    assert composite["resolved_ids"] == ["a", "c", "d"]
    assert composite["empty_split"]["model"] == ["b"]
    assert composite["model_contract"] == MODEL_CONTRACT
    assert composite["harness_contract"] == HARNESS_CONTRACT
    assert composite["selected_attempts"] == {
        "fixed": {"source": 1, "empty_retry": 1},
        "complement": {"source": 1, "empty_retry": 1},
    }
    assert list(json.loads(predictions.read_text())) == [
        "b",
        "a",
        "d",
        "c",
    ]
    assert verify_disjoint_panels(
        full_ids_path=full_ids_path,
        panels=[fixed, complement],
        output_path=output,
        predictions_path=predictions,
    ) == composite


def test_join_disjoint_panels_rejects_model_contract_drift(
    tmp_path: Path,
) -> None:
    full_ids_path = tmp_path / "full_ids.json"
    full_ids_path.write_text(json.dumps(["a", "b"]) + "\n")
    first = _write_panel(
        tmp_path,
        "first",
        ["a"],
        resolved={"a"},
        empty=set(),
    )
    drifted = dict(MODEL_CONTRACT, model_config_sha256="f" * 64)
    second = _write_panel(
        tmp_path,
        "second",
        ["b"],
        resolved={"b"},
        empty=set(),
        model_contract=drifted,
    )

    with pytest.raises(ValueError, match="model contract"):
        join_disjoint_panels(
            full_ids_path=full_ids_path,
            panels=[first, second],
            output_path=tmp_path / "out.json",
            predictions_path=tmp_path / "preds.json",
        )


def test_join_disjoint_panels_rejects_incomplete_coverage(
    tmp_path: Path,
) -> None:
    full_ids_path = tmp_path / "full_ids.json"
    full_ids_path.write_text(json.dumps(["a", "b", "c"]) + "\n")
    first = _write_panel(
        tmp_path,
        "first",
        ["a", "b"],
        resolved={"a"},
        empty={"b"},
    )
    second = _write_panel(
        tmp_path,
        "second",
        ["b"],
        resolved={"b"},
        empty=set(),
    )

    with pytest.raises(ValueError, match="disjoint union"):
        join_disjoint_panels(
            full_ids_path=full_ids_path,
            panels=[first, second],
            output_path=tmp_path / "out.json",
            predictions_path=tmp_path / "preds.json",
        )


def test_join_disjoint_panels_rejects_changed_predictions_binding(
    tmp_path: Path,
) -> None:
    full_ids_path = tmp_path / "full_ids.json"
    full_ids_path.write_text(json.dumps(["a", "b"]) + "\n")
    first = _write_panel(
        tmp_path,
        "first",
        ["a"],
        resolved={"a"},
        empty=set(),
    )
    second = _write_panel(
        tmp_path,
        "second",
        ["b"],
        resolved={"b"},
        empty=set(),
    )
    predictions = json.loads(first.predictions_path.read_text())
    predictions["a"]["attempt_id"] = "mutated-after-composite"
    first.predictions_path.write_text(json.dumps(predictions) + "\n")

    with pytest.raises(ValueError, match="predictions artifact"):
        join_disjoint_panels(
            full_ids_path=full_ids_path,
            panels=[first, second],
            output_path=tmp_path / "out.json",
            predictions_path=tmp_path / "preds.json",
        )


def test_join_disjoint_panels_rejects_inconsistent_resampling(
    tmp_path: Path,
) -> None:
    full_ids_path = tmp_path / "full_ids.json"
    full_ids_path.write_text(json.dumps(["a", "b"]) + "\n")
    first = _write_panel(
        tmp_path,
        "first",
        ["a"],
        resolved=set(),
        empty={"a"},
    )
    second = _write_panel(
        tmp_path,
        "second",
        ["b"],
        resolved={"b"},
        empty=set(),
    )
    composite = json.loads(first.composite_path.read_text())
    composite["empty_resampling"]["became_nonempty"] = 1
    composite["empty_resampling"]["nonempty_rate"] = 1.0
    first.composite_path.write_text(json.dumps(composite) + "\n")

    with pytest.raises(ValueError, match="resampling"):
        join_disjoint_panels(
            full_ids_path=full_ids_path,
            panels=[first, second],
            output_path=tmp_path / "out.json",
            predictions_path=tmp_path / "preds.json",
        )
