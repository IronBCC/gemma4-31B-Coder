from __future__ import annotations

import hashlib
import importlib
import json
from pathlib import Path

import pytest

from phaseH_eval.tests.v2p11_lineage_fixture import (
    binding,
    create_complete_phase_lineage,
    write_json,
)


def _phase_sha(marker: Path, basename: str) -> str:
    report = json.loads(marker.read_text())
    matches = [
        artifact["sha256"]
        for artifact in report["artifacts"]
        if Path(artifact["path"]).name == basename
    ]
    assert len(matches) == 1
    return str(matches[0])


def _write_posttrain_fixture(
    tmp_path: Path,
) -> tuple[dict[str, Path], Path, Path]:
    stage_marker = tmp_path / "v2p11_training_complete.json"
    write_json(
        stage_marker,
        {
            "schema_version": 1,
            "complete": True,
            "adapter_sha256": "placeholder",
            "optimizer_steps": 78,
        },
    )
    final_model = tmp_path / "final_model"
    lineage = create_complete_phase_lineage(
        tmp_path / "lineage",
        final_model=final_model,
        stage_marker=stage_marker,
    )
    posttrain_marker = tmp_path / "v2p11_posttrain_complete.json"
    write_json(
        posttrain_marker,
        {
            "schema_version": 1,
            "complete": True,
            "gpu_uuid": "GPU-fixture",
            "stage_a_adapter_sha256": json.loads(
                stage_marker.read_text()
            )["adapter_sha256"],
            "recovery_adapter_sha256": _phase_sha(
                lineage["recovery"],
                "adapter_model.safetensors",
            ),
            "kto_adapter_sha256": _phase_sha(
                lineage["kto"],
                "adapter_model.safetensors",
            ),
            "recovery_manifest_sha256": binding(
                lineage["recovery_manifest"]
            )["sha256"],
            "behavior_manifest_sha256": binding(
                lineage["behavior_manifest"]
            )["sha256"],
            "final_merge_audit_sha256": binding(
                final_model / "v2p11_final_merge_audit.json"
            )["sha256"],
            "kto_training_evidence_sha256": binding(
                lineage["kto_evidence"]
            )["sha256"],
            "final_merge_marker_sha256": binding(
                lineage["final_merge"]
            )["sha256"],
            "recovery_input_marker_sha256": binding(
                lineage["recovery_input"]
            )["sha256"],
            "kto_input_marker_sha256": binding(
                lineage["kto_input"]
            )["sha256"],
            "recovery_optimizer_steps": 105,
            "kto_optimizer_steps": 25,
        },
    )
    return lineage, posttrain_marker, final_model


def _write_r3_behavior_posttrain_fixture(
    tmp_path: Path,
    *,
    coverage_override: dict[str, int] | None = None,
    selected_uids_override: list[str] | None = None,
    recovery_manifest_override: dict[str, object] | None = None,
    omit_recovery_audit_from_marker: bool = False,
) -> tuple[dict[str, Path], Path, Path]:
    stage_marker = tmp_path / "v2p11r3_v2p10init_fable_reasoned_training_completion.json"
    write_json(stage_marker, {})
    final_model = tmp_path / "r3_behavior_final_model"
    lineage = create_complete_phase_lineage(
        tmp_path / "r3_behavior_lineage",
        final_model=final_model,
        stage_marker=stage_marker,
        r3_stage=True,
        marker_prefix="v2p11r3_behavior",
        coverage_override=coverage_override,
        selected_uids_override=selected_uids_override,
        recovery_manifest_override=recovery_manifest_override,
        omit_recovery_audit_from_marker=omit_recovery_audit_from_marker,
    )
    posttrain_marker = tmp_path / "v2p11r3_behavior_posttrain_complete.json"
    write_json(
        posttrain_marker,
        {
            "schema_version": 1,
            "complete": True,
            "gpu_uuid": "GPU-fixture",
            "stage_a_adapter_sha256": json.loads(
                stage_marker.read_text()
            )["adapter"]["sha256"],
            "recovery_adapter_sha256": _phase_sha(
                lineage["recovery"], "adapter_model.safetensors"
            ),
            "kto_adapter_sha256": _phase_sha(
                lineage["kto"], "adapter_model.safetensors"
            ),
            "recovery_manifest_sha256": binding(
                lineage["recovery_manifest"]
            )["sha256"],
            "behavior_manifest_sha256": binding(
                lineage["behavior_manifest"]
            )["sha256"],
            "final_merge_audit_sha256": binding(
                final_model / "v2p11_final_merge_audit.json"
            )["sha256"],
            "kto_training_evidence_sha256": binding(
                lineage["kto_evidence"]
            )["sha256"],
            "final_merge_marker_sha256": binding(
                lineage["final_merge"]
            )["sha256"],
            "recovery_input_marker_sha256": binding(
                lineage["recovery_input"]
            )["sha256"],
            "kto_input_marker_sha256": binding(
                lineage["kto_input"]
            )["sha256"],
            "recovery_optimizer_steps": 105,
            "kto_optimizer_steps": 25,
        },
    )
    return lineage, posttrain_marker, final_model


def test_validates_complete_posttrain_lineage_before_portability(
    tmp_path: Path,
) -> None:
    module = importlib.import_module("phaseH_eval.v2p11_posttrain_lineage")
    lineage, posttrain_marker, final_model = _write_posttrain_fixture(tmp_path)

    report = module.validate_posttrain_lineage(
        posttrain_marker_path=posttrain_marker,
        recovery_marker_path=lineage["recovery"],
        kto_marker_path=lineage["kto"],
        final_merge_marker_path=lineage["final_merge"],
        final_model_path=final_model,
    )

    assert report["status"] == "complete"
    assert report["posttrain_marker_sha256"] == hashlib.sha256(
        posttrain_marker.read_bytes()
    ).hexdigest()
    assert report["final_model"]["model_path"] == str(final_model.resolve())


def test_rejects_tampered_phase_artifact_before_portability(
    tmp_path: Path,
) -> None:
    module = importlib.import_module("phaseH_eval.v2p11_posttrain_lineage")
    lineage, posttrain_marker, final_model = _write_posttrain_fixture(tmp_path)
    recovery_report = json.loads(lineage["recovery"].read_text())
    recovery_adapter = next(
        Path(artifact["path"])
        for artifact in recovery_report["artifacts"]
        if Path(artifact["path"]).name == "adapter_model.safetensors"
    )
    recovery_adapter.write_bytes(b"tampered")

    with pytest.raises(ValueError, match="binding changed"):
        module.validate_posttrain_lineage(
            posttrain_marker_path=posttrain_marker,
            recovery_marker_path=lineage["recovery"],
            kto_marker_path=lineage["kto"],
            final_merge_marker_path=lineage["final_merge"],
            final_model_path=final_model,
        )


def test_validates_r3_nested_stage_and_unique_marker_names(
    tmp_path: Path,
) -> None:
    module = importlib.import_module("phaseH_eval.v2p11_posttrain_lineage")
    lineage, posttrain_marker, final_model = (
        _write_r3_behavior_posttrain_fixture(tmp_path)
    )

    report = module.validate_posttrain_lineage(
        posttrain_marker_path=posttrain_marker,
        recovery_marker_path=lineage["recovery"],
        kto_marker_path=lineage["kto"],
        final_merge_marker_path=lineage["final_merge"],
        final_model_path=final_model,
    )

    assert report["status"] == "complete"
    assert report["final_model"]["model_path"] == str(final_model.resolve())


def test_rejects_r3_recovery_run_contract_drift(tmp_path: Path) -> None:
    module = importlib.import_module("phaseH_eval.v2p11_posttrain_lineage")
    lineage, posttrain_marker, final_model = (
        _write_r3_behavior_posttrain_fixture(
            tmp_path,
            recovery_manifest_override={"lr": 1.0},
        )
    )

    with pytest.raises(ValueError, match="recovery SFT"):
        module.validate_posttrain_lineage(
            posttrain_marker_path=posttrain_marker,
            recovery_marker_path=lineage["recovery"],
            kto_marker_path=lineage["kto"],
            final_merge_marker_path=lineage["final_merge"],
            final_model_path=final_model,
        )


def test_rejects_r3_recovery_merge_without_bound_audit(
    tmp_path: Path,
) -> None:
    module = importlib.import_module("phaseH_eval.v2p11_posttrain_lineage")
    lineage, posttrain_marker, final_model = (
        _write_r3_behavior_posttrain_fixture(
            tmp_path,
            omit_recovery_audit_from_marker=True,
        )
    )

    with pytest.raises(ValueError, match="recovery merge audit"):
        module.validate_posttrain_lineage(
            posttrain_marker_path=posttrain_marker,
            recovery_marker_path=lineage["recovery"],
            kto_marker_path=lineage["kto"],
            final_merge_marker_path=lineage["final_merge"],
            final_model_path=final_model,
        )
