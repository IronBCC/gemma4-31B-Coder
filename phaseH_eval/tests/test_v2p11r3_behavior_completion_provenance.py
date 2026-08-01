from __future__ import annotations

import importlib
import hashlib
import json
from pathlib import Path

import pytest

from phaseH_eval.tests.test_v2p11_posttrain_lineage import (
    _write_r3_behavior_posttrain_fixture,
)
from phaseH_eval.tests.v2p11_lineage_fixture import binding


EXPECTED_COVERAGE = {
    "desirable_correct_patch": 25,
    "empty_terminal": 8,
    "repeated_read_loop": 8,
    "wrong_nonempty_replay": 9,
}


def _production_contract(
    *,
    recovery_data: Path | None = None,
    behavior_data: Path | None = None,
    behavior_manifest: Path | None = None,
    exclusions: Path | None = None,
) -> dict[str, object]:
    def sha(path: Path | None, fallback: str) -> str:
        return (
            hashlib.sha256(path.read_bytes()).hexdigest()
            if path is not None
            else fallback
        )

    return {
        "recovery_rows": 138,
        "recovery_optimizer_steps": 105,
        "behavior_rows": 606,
        "behavior_negative_counts": {
            "empty_terminal": 33,
            "repeated_read_loop": 55,
            "wrong_nonempty_replay": 228,
        },
        "full_evaluation_ids": 300,
        "combined_exclusion_ids": 707,
        "excluded_repositories": 12,
        "evaluation_overlap": 0,
        "recovery_train_sha256": sha(
            recovery_data / "train.jsonl" if recovery_data else None,
            "recovery-train",
        ),
        "recovery_manifest_sha256": sha(
            recovery_data / "manifest.json" if recovery_data else None,
            "recovery-manifest",
        ),
        "behavior_data_sha256": sha(behavior_data, "behavior-data"),
        "behavior_manifest_sha256": sha(
            behavior_manifest, "behavior-manifest"
        ),
        "exclusion_sha256": sha(exclusions, "exclusions"),
    }


def test_behavior_poststage_validates_recovery_and_kto_coverage(
    tmp_path: Path,
) -> None:
    module = importlib.import_module(
        "phaseH_eval.v2p11r3_behavior_completion_provenance"
    )
    lineage, posttrain, final_model = _write_r3_behavior_posttrain_fixture(
        tmp_path
    )

    report = module._validate_behavior_poststage(
        posttrain_contract=_production_contract(),
        posttrain_marker_path=posttrain,
        recovery_marker_path=lineage["recovery"],
        kto_marker_path=lineage["kto"],
        final_merge_marker_path=lineage["final_merge"],
        final_model_path=final_model,
        final_audit_path=final_model / "v2p11_final_merge_audit.json",
    )

    assert report["recovery_rows"] == 138
    assert report["recovery_optimizer_steps"] == 105
    assert report["behavior_rows"] == 606
    assert report["kto_optimizer_steps"] == 25
    assert report["coverage_counts"] == EXPECTED_COVERAGE
    assert report["negative_counts"] == {
        "empty_terminal": 33,
        "repeated_read_loop": 55,
        "wrong_nonempty_replay": 228,
    }


def test_behavior_poststage_rejects_semantically_wrong_coverage(
    tmp_path: Path,
) -> None:
    module = importlib.import_module(
        "phaseH_eval.v2p11r3_behavior_completion_provenance"
    )
    wrong = {**EXPECTED_COVERAGE, "empty_terminal": 7}
    lineage, posttrain, final_model = _write_r3_behavior_posttrain_fixture(
        tmp_path,
        coverage_override=wrong,
    )

    with pytest.raises(ValueError, match="coverage"):
        module._validate_behavior_poststage(
            posttrain_contract=_production_contract(),
            posttrain_marker_path=posttrain,
            recovery_marker_path=lineage["recovery"],
            kto_marker_path=lineage["kto"],
            final_merge_marker_path=lineage["final_merge"],
            final_model_path=final_model,
            final_audit_path=final_model / "v2p11_final_merge_audit.json",
        )


def test_behavior_poststage_rejects_wrong_selected_training_rows(
    tmp_path: Path,
) -> None:
    module = importlib.import_module(
        "phaseH_eval.v2p11r3_behavior_completion_provenance"
    )
    lineage, posttrain, final_model = _write_r3_behavior_posttrain_fixture(
        tmp_path,
        selected_uids_override=[f"wrong-{index:02d}" for index in range(50)],
    )

    with pytest.raises(ValueError, match="selection"):
        module._validate_behavior_poststage(
            posttrain_contract=_production_contract(),
            posttrain_marker_path=posttrain,
            recovery_marker_path=lineage["recovery"],
            kto_marker_path=lineage["kto"],
            final_merge_marker_path=lineage["final_merge"],
            final_model_path=final_model,
            final_audit_path=final_model / "v2p11_final_merge_audit.json",
        )


def test_cross_stage_bindings_reject_spliced_reasoning_adapter(
    tmp_path: Path,
) -> None:
    module = importlib.import_module(
        "phaseH_eval.v2p11r3_behavior_completion_provenance"
    )
    lineage, posttrain, final_model = _write_r3_behavior_posttrain_fixture(
        tmp_path
    )
    recovery_data = lineage["recovery_manifest"].parent
    behavior_data = lineage["behavior_data"]
    behavior_manifest = lineage["behavior_manifest"]
    exclusions = tmp_path / "exclusions.json"
    exclusions.write_text("{}\n")
    contract = _production_contract(
        recovery_data=recovery_data,
        behavior_data=behavior_data,
        behavior_manifest=behavior_manifest,
        exclusions=exclusions,
    )
    poststage = module._validate_behavior_poststage(
        posttrain_contract=contract,
        posttrain_marker_path=posttrain,
        recovery_marker_path=lineage["recovery"],
        kto_marker_path=lineage["kto"],
        final_merge_marker_path=lineage["final_merge"],
        final_model_path=final_model,
        final_audit_path=final_model / "v2p11_final_merge_audit.json",
    )
    other_adapter = tmp_path / "other-adapter.safetensors"
    other_adapter.write_bytes(b"other")
    other_completion = tmp_path / "other-completion.json"
    other_completion.write_text("{}\n")

    with pytest.raises(ValueError, match="stage|lineage"):
        module._validate_cross_stage_bindings(
            training={
                "adapter": binding(other_adapter),
                "completion": binding(other_completion),
            },
            poststage=poststage,
            recovery_data_path=recovery_data,
            behavior_data_path=behavior_data,
            behavior_manifest_path=behavior_manifest,
            exclusions_path=exclusions,
            posttrain_contract=contract,
        )


def test_behavior_provenance_rebuilds_from_bound_inputs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = importlib.import_module(
        "phaseH_eval.v2p11r3_behavior_completion_provenance"
    )
    lineage, posttrain, final_model = _write_r3_behavior_posttrain_fixture(
        tmp_path
    )
    model_contract = {
        "model_path": str(final_model.resolve()),
        "served_name": "teacher_sft_v2p11r3_behavior",
    }
    base_report = {
        "schema_version": 1,
        "artifact_type": "v2p11r3_completion_provenance",
        "status": "complete",
        "candidate_name": "teacher_sft_v2p11r3_behavior",
        "lineage": {
            "kind": "direct_lora_from_v2p10",
            "reasoned_fable_contract": True,
        },
        "full_ids": {"path": str((tmp_path / "ids.json").resolve())},
        "v2p10_full300": {"path": str((tmp_path / "v2p10.json").resolve())},
        "dataset": {
            "path": str((tmp_path / "dataset").resolve()),
            "context_audit": {"path": str((tmp_path / "context.json").resolve())},
        },
        "training": {
            "adapter": binding(lineage["stage_weights"]),
            "init_adapter": {"path": str((tmp_path / "init" / "adapter_model.safetensors").resolve())},
            "completion": binding(lineage["stage_marker"]),
        },
        "final_model": model_contract,
        "merge_audit": {
            "path": str((final_model / "v2p11_final_merge_audit.json").resolve())
        },
        "portability": {
            "gate": {"path": str((tmp_path / "portability.json").resolve())}
        },
    }
    monkeypatch.setattr(module, "_build_r3_report", lambda **_: base_report)
    for path in (
        tmp_path / "ids.json",
        tmp_path / "v2p10.json",
        tmp_path / "context.json",
        tmp_path / "portability.json",
        tmp_path / "exclusions.json",
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}\n")
    (tmp_path / "dataset").mkdir()
    (tmp_path / "init").mkdir()
    (tmp_path / "init" / "adapter_model.safetensors").write_bytes(b"init")
    recovery_data = lineage["recovery_manifest"].parent
    behavior_data = lineage["behavior_data"]
    behavior_manifest = lineage["behavior_manifest"]
    exclusions = tmp_path / "exclusions.json"
    monkeypatch.setattr(
        module,
        "validate_posttrain_inputs",
        lambda **_: _production_contract(
            recovery_data=recovery_data,
            behavior_data=behavior_data,
            behavior_manifest=behavior_manifest,
            exclusions=exclusions,
        ),
    )

    report = module._build_report(
        candidate_name="teacher_sft_v2p11r3_behavior",
        dataset_path=tmp_path / "dataset",
        dataset_context_audit_path=tmp_path / "context.json",
        adapter_path=lineage["stage_weights"].parent,
        init_adapter_path=tmp_path / "init",
        training_completion_path=lineage["stage_marker"],
        recovery_data_path=recovery_data,
        behavior_data_path=behavior_data,
        behavior_manifest_path=behavior_manifest,
        exclusions_path=exclusions,
        posttrain_marker_path=posttrain,
        recovery_marker_path=lineage["recovery"],
        kto_marker_path=lineage["kto"],
        final_merge_marker_path=lineage["final_merge"],
        final_model_path=final_model,
        merge_audit_path=final_model / "v2p11_final_merge_audit.json",
        portability_gate_path=tmp_path / "portability.json",
        full_ids_path=tmp_path / "ids.json",
        v2p10_composite_path=tmp_path / "v2p10.json",
    )

    assert report["artifact_type"] == "v2p11r3_behavior_completion_provenance"
    assert report["lineage"] == {
        "kind": "r3_reasoning_then_recovery_sft_then_behavior_kto",
        "reasoned_fable_contract": True,
        "transferable_empty_loop_correction": True,
    }
    assert report["behavior_poststage"]["coverage_counts"] == EXPECTED_COVERAGE
    assert report["final_model"] == model_contract

    provenance = tmp_path / "behavior-provenance.json"
    provenance.write_text(json.dumps(report, sort_keys=True) + "\n")
    validated = module.validate_completion_provenance(
        provenance,
        full_ids_path=tmp_path / "ids.json",
        v2p10_composite_path=tmp_path / "v2p10.json",
        candidate_model_contract=model_contract,
        candidate_name="teacher_sft_v2p11r3_behavior",
    )

    assert validated["artifact"]["path"] == str(provenance.resolve())
    assert validated["behavior_poststage"]["coverage_counts"] == EXPECTED_COVERAGE
    assert validated["evaluation_exclusion"]["overlap"] == 0
