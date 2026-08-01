#!/usr/bin/env python3
"""Publish and revalidate the r3 reasoning-to-recovery-to-KTO lineage."""
from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from phaseH_eval.empty_retry_composite import (
    _binding,
    _publish_json_noreplace,
    _read_object,
)
from phaseH_eval.v2p11_posttrain_contract import validate_posttrain_inputs
from phaseH_eval.v2p11_posttrain_lineage import (
    _R3_COVERAGE_COUNTS,
    _R3_NEGATIVE_COUNTS,
    _validate_r3_behavior_posttrain_lineage,
)
from phaseH_eval.v2p11r3_completion_provenance import (
    _build_report as _build_r3_report,
)


ARTIFACT_TYPE = "v2p11r3_behavior_completion_provenance"
LINEAGE = {
    "kind": "r3_reasoning_then_recovery_sft_then_behavior_kto",
    "reasoned_fable_contract": True,
    "transferable_empty_loop_correction": True,
}


def _model_contracts_match(
    candidate: Mapping[str, Any],
    expected: Mapping[str, Any],
) -> bool:
    candidate_value = dict(candidate)
    expected_value = dict(expected)
    for key in ("model_index_sha256", "model_safetensors_sha256"):
        if key not in candidate_value and expected_value.get(key) is None:
            candidate_value[key] = None
        if key not in expected_value and candidate_value.get(key) is None:
            expected_value[key] = None
    return candidate_value == expected_value


def _validate_behavior_poststage(
    *,
    posttrain_contract: Mapping[str, Any],
    posttrain_marker_path: Path,
    recovery_marker_path: Path,
    kto_marker_path: Path,
    final_merge_marker_path: Path,
    final_model_path: Path,
    final_audit_path: Path,
) -> dict[str, Any]:
    lineage = _validate_r3_behavior_posttrain_lineage(
        posttrain_marker_path=posttrain_marker_path,
        recovery_marker_path=recovery_marker_path,
        kto_marker_path=kto_marker_path,
        final_merge_marker_path=final_merge_marker_path,
        final_model_path=final_model_path,
        final_audit_path=final_audit_path,
    )
    if (
        posttrain_contract.get("recovery_rows") != 138
        or posttrain_contract.get("recovery_optimizer_steps") != 105
        or posttrain_contract.get("behavior_rows") != 606
        or posttrain_contract.get("behavior_negative_counts")
        != _R3_NEGATIVE_COUNTS
        or posttrain_contract.get("full_evaluation_ids") != 300
        or posttrain_contract.get("combined_exclusion_ids") != 707
        or posttrain_contract.get("excluded_repositories") != 12
        or posttrain_contract.get("evaluation_overlap") != 0
        or lineage.get("coverage_counts") != _R3_COVERAGE_COUNTS
    ):
        raise ValueError("r3 behavior poststage coverage contract is incomplete")
    return {
        "recovery_rows": 138,
        "recovery_optimizer_steps": 105,
        "behavior_rows": 606,
        "kto_optimizer_steps": 25,
        "coverage_counts": dict(_R3_COVERAGE_COUNTS),
        "negative_counts": dict(_R3_NEGATIVE_COUNTS),
        "posttrain_contract": dict(posttrain_contract),
        "posttrain_marker": lineage["posttrain_marker"],
        "stage_marker": lineage["stage_marker"],
        "stage_adapter": lineage["stage_adapter"],
        "phase_markers": lineage["phase_markers"],
        "inputs": lineage["inputs"],
        "kto_training_evidence": lineage["kto_training_evidence"],
    }


def _validate_cross_stage_bindings(
    *,
    training: Mapping[str, Any],
    poststage: Mapping[str, Any],
    recovery_data_path: Path,
    behavior_data_path: Path,
    behavior_manifest_path: Path,
    exclusions_path: Path,
    posttrain_contract: Mapping[str, Any],
) -> None:
    recovery_data_path = Path(recovery_data_path).resolve()
    behavior_data_path = Path(behavior_data_path).resolve()
    behavior_manifest_path = Path(behavior_manifest_path).resolve()
    exclusions_path = Path(exclusions_path).resolve()
    inputs = poststage.get("inputs")
    if (
        not isinstance(inputs, Mapping)
        or training.get("completion") != poststage.get("stage_marker")
        or training.get("adapter") != poststage.get("stage_adapter")
    ):
        raise ValueError("reasoning and behavior stage lineage is inconsistent")
    current_inputs = {
        "recovery_train": _binding(recovery_data_path / "train.jsonl"),
        "recovery_manifest": _binding(recovery_data_path / "manifest.json"),
        "behavior_data": _binding(behavior_data_path),
        "behavior_manifest": _binding(behavior_manifest_path),
    }
    if any(inputs.get(key) != value for key, value in current_inputs.items()):
        raise ValueError("behavior stage input lineage is inconsistent")
    expected_hashes = {
        "recovery_train_sha256": current_inputs["recovery_train"]["sha256"],
        "recovery_manifest_sha256": current_inputs["recovery_manifest"][
            "sha256"
        ],
        "behavior_data_sha256": current_inputs["behavior_data"]["sha256"],
        "behavior_manifest_sha256": current_inputs["behavior_manifest"][
            "sha256"
        ],
        "exclusion_sha256": _binding(exclusions_path)["sha256"],
    }
    if any(
        posttrain_contract.get(key) != value
        for key, value in expected_hashes.items()
    ):
        raise ValueError("production posttrain input lineage is inconsistent")


def _build_report(
    *,
    candidate_name: str,
    dataset_path: Path,
    dataset_context_audit_path: Path,
    adapter_path: Path,
    init_adapter_path: Path,
    training_completion_path: Path,
    recovery_data_path: Path,
    behavior_data_path: Path,
    behavior_manifest_path: Path,
    exclusions_path: Path,
    posttrain_marker_path: Path,
    recovery_marker_path: Path,
    kto_marker_path: Path,
    final_merge_marker_path: Path,
    final_model_path: Path,
    merge_audit_path: Path,
    portability_gate_path: Path,
    full_ids_path: Path,
    v2p10_composite_path: Path,
    v2p10_lineage_path: Path,
) -> dict[str, Any]:
    base = _build_r3_report(
        candidate_name=candidate_name,
        dataset_path=dataset_path,
        dataset_context_audit_path=dataset_context_audit_path,
        adapter_path=adapter_path,
        init_adapter_path=init_adapter_path,
        training_completion_path=training_completion_path,
        final_model_path=final_model_path,
        merge_audit_path=merge_audit_path,
        portability_gate_path=portability_gate_path,
        full_ids_path=full_ids_path,
        v2p10_composite_path=v2p10_composite_path,
        v2p10_lineage_path=v2p10_lineage_path,
    )
    recovery_data_path = Path(recovery_data_path).resolve()
    behavior_data_path = Path(behavior_data_path).resolve()
    behavior_manifest_path = Path(behavior_manifest_path).resolve()
    exclusions_path = Path(exclusions_path).resolve()
    contract = validate_posttrain_inputs(
        recovery_data=recovery_data_path,
        behavior_data=behavior_data_path,
        behavior_manifest=behavior_manifest_path,
        exclusions=exclusions_path,
        full_ids=Path(full_ids_path).resolve(),
        require_production_identity=True,
    )
    poststage = _validate_behavior_poststage(
        posttrain_contract=contract,
        posttrain_marker_path=posttrain_marker_path,
        recovery_marker_path=recovery_marker_path,
        kto_marker_path=kto_marker_path,
        final_merge_marker_path=final_merge_marker_path,
        final_model_path=final_model_path,
        final_audit_path=merge_audit_path,
    )
    _validate_cross_stage_bindings(
        training=base["training"],
        poststage=poststage,
        recovery_data_path=recovery_data_path,
        behavior_data_path=behavior_data_path,
        behavior_manifest_path=behavior_manifest_path,
        exclusions_path=exclusions_path,
        posttrain_contract=contract,
    )
    poststage["contract_inputs"] = {
        "recovery_train": _binding(recovery_data_path / "train.jsonl"),
        "recovery_manifest": _binding(recovery_data_path / "manifest.json"),
        "behavior_data": _binding(behavior_data_path),
        "behavior_manifest": _binding(behavior_manifest_path),
        "exclusions": _binding(exclusions_path),
    }
    return {
        **base,
        "artifact_type": ARTIFACT_TYPE,
        "lineage": dict(LINEAGE),
        "behavior_poststage": poststage,
    }


def publish_completion_provenance(
    *,
    output_path: Path,
    **kwargs: Any,
) -> dict[str, Any]:
    report = _build_report(**kwargs)
    _publish_json_noreplace(Path(output_path).resolve(), report)
    return report


def validate_completion_provenance(
    provenance_path: Path,
    *,
    full_ids_path: Path,
    v2p10_composite_path: Path,
    v2p10_lineage_path: Path,
    candidate_model_contract: Mapping[str, Any],
    candidate_name: str,
) -> dict[str, Any]:
    provenance_path = Path(provenance_path).resolve()
    report = _read_object(provenance_path)
    dataset = report.get("dataset")
    training = report.get("training")
    final_model = report.get("final_model")
    portability = report.get("portability")
    v2p10_lineage = report.get("v2p10_training_lineage")
    poststage = report.get("behavior_poststage")
    if (
        report.get("schema_version") != 1
        or report.get("artifact_type") != ARTIFACT_TYPE
        or report.get("status") != "complete"
        or report.get("candidate_name") != candidate_name
        or report.get("lineage") != LINEAGE
        or not isinstance(dataset, Mapping)
        or not isinstance(training, Mapping)
        or not isinstance(final_model, Mapping)
        or not isinstance(portability, Mapping)
        or not isinstance(portability.get("gate"), Mapping)
        or not isinstance(v2p10_lineage, Mapping)
        or not isinstance(v2p10_lineage.get("contract"), Mapping)
        or not isinstance(poststage, Mapping)
        or not isinstance(poststage.get("phase_markers"), Mapping)
        or not isinstance(poststage.get("contract_inputs"), Mapping)
    ):
        raise ValueError("v2.11r3 behavior completion provenance is incomplete")
    phase_markers = poststage["phase_markers"]
    contract_inputs = poststage["contract_inputs"]
    expected = _build_report(
        candidate_name=candidate_name,
        dataset_path=Path(str(dataset.get("path", ""))),
        dataset_context_audit_path=Path(
            str(dataset.get("context_audit", {}).get("path", ""))
        ),
        adapter_path=Path(
            str(training.get("adapter", {}).get("path", ""))
        ).parent,
        init_adapter_path=Path(
            str(training.get("init_adapter", {}).get("path", ""))
        ).parent,
        training_completion_path=Path(
            str(training.get("completion", {}).get("path", ""))
        ),
        recovery_data_path=Path(
            str(contract_inputs.get("recovery_manifest", {}).get("path", ""))
        ).parent,
        behavior_data_path=Path(
            str(contract_inputs.get("behavior_data", {}).get("path", ""))
        ),
        behavior_manifest_path=Path(
            str(contract_inputs.get("behavior_manifest", {}).get("path", ""))
        ),
        exclusions_path=Path(
            str(contract_inputs.get("exclusions", {}).get("path", ""))
        ),
        posttrain_marker_path=Path(
            str(poststage.get("posttrain_marker", {}).get("path", ""))
        ),
        recovery_marker_path=Path(
            str(phase_markers.get("recovery", {}).get("path", ""))
        ),
        kto_marker_path=Path(
            str(phase_markers.get("kto", {}).get("path", ""))
        ),
        final_merge_marker_path=Path(
            str(phase_markers.get("final_merge", {}).get("path", ""))
        ),
        final_model_path=Path(str(final_model.get("model_path", ""))),
        merge_audit_path=Path(
            str(report.get("merge_audit", {}).get("path", ""))
        ),
        portability_gate_path=Path(
            str(portability.get("gate", {}).get("path", ""))
        ),
        full_ids_path=Path(full_ids_path),
        v2p10_composite_path=Path(v2p10_composite_path),
        v2p10_lineage_path=Path(v2p10_lineage_path).resolve(),
    )
    if report != expected:
        raise ValueError("v2.11r3 behavior completion provenance changed")
    if not _model_contracts_match(
        candidate_model_contract,
        expected["final_model"],
    ):
        raise ValueError(
            "v2.11r3 behavior provenance does not bind evaluated model"
        )
    source_fable = expected["dataset"].get("source_fable", {})
    source_ids = (
        source_fable.get("source_ids", [])
        if isinstance(source_fable, Mapping)
        else []
    )
    return {
        "artifact": _binding(provenance_path),
        "fable": {
            "stage_a": {
                "new_strict_rows": 36,
                "total_rows": 77,
            },
            "recent_strict_rows": len(source_ids),
            "recovery_unique_sources": 46,
            "targeted_recovery_rows": 138,
        },
        "evaluation_exclusion": {
            "full_ids": 300,
            "excluded_ids": 707,
            "excluded_repositories": 12,
            "overlap": 0,
        },
        "dataset": expected["dataset"],
        "training": expected["training"],
        "behavior_poststage": expected["behavior_poststage"],
        "final_model": expected["final_model"],
        "portability": expected["portability"],
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-name", required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--dataset-context-audit", type=Path, required=True)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--init-adapter", type=Path, required=True)
    parser.add_argument("--training-completion", type=Path, required=True)
    parser.add_argument("--recovery-data", type=Path, required=True)
    parser.add_argument("--behavior-data", type=Path, required=True)
    parser.add_argument("--behavior-manifest", type=Path, required=True)
    parser.add_argument("--exclusions", type=Path, required=True)
    parser.add_argument("--posttrain-marker", type=Path, required=True)
    parser.add_argument("--recovery-marker", type=Path, required=True)
    parser.add_argument("--kto-marker", type=Path, required=True)
    parser.add_argument("--final-merge-marker", type=Path, required=True)
    parser.add_argument("--final-model", type=Path, required=True)
    parser.add_argument("--merge-audit", type=Path, required=True)
    parser.add_argument("--portability-gate", type=Path, required=True)
    parser.add_argument("--full-ids", type=Path, required=True)
    parser.add_argument("--v2p10", type=Path, required=True)
    parser.add_argument("--v2p10-lineage", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    report = publish_completion_provenance(
        candidate_name=args.candidate_name,
        dataset_path=args.dataset,
        dataset_context_audit_path=args.dataset_context_audit,
        adapter_path=args.adapter,
        init_adapter_path=args.init_adapter,
        training_completion_path=args.training_completion,
        recovery_data_path=args.recovery_data,
        behavior_data_path=args.behavior_data,
        behavior_manifest_path=args.behavior_manifest,
        exclusions_path=args.exclusions,
        posttrain_marker_path=args.posttrain_marker,
        recovery_marker_path=args.recovery_marker,
        kto_marker_path=args.kto_marker,
        final_merge_marker_path=args.final_merge_marker,
        final_model_path=args.final_model,
        merge_audit_path=args.merge_audit,
        portability_gate_path=args.portability_gate,
        full_ids_path=args.full_ids,
        v2p10_composite_path=args.v2p10,
        v2p10_lineage_path=args.v2p10_lineage,
        output_path=args.out,
    )
    print(
        json.dumps(
            {
                "status": report["status"],
                "candidate_name": report["candidate_name"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
