#!/usr/bin/env python3
"""Build a checksum-bound paired v2.11 versus v2.10 full-300 verdict."""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

from phaseH_eval.empty_retry_composite import (
    _acceptance_empty_ids,
    _binding,
    _complete_run,
    _publish_bytes_noreplace,
    _read_ids,
    _read_object,
    _validate_composite_source,
)
from phaseH_eval.full300_panel_composite import (
    PanelInput,
    verify_disjoint_panels,
)
from phaseH_eval.full300_official_score_binding import (
    validate_official_score_binding,
)
from phaseH_eval.v2p11_completion_provenance import (
    _validate_bound_contract_semantics,
    _validate_phase_marker as _validate_completion_phase_marker,
)
from phaseH_eval.v2p10_training_lineage import (
    validate_v2p10_training_lineage_contract,
)


@dataclass(frozen=True)
class FirstPassPanel:
    tag: str
    ids_path: Path
    run_root: Path


_PORTABILITY_CRITERIA = {
    "candidate_empty_at_most_one": True,
    "candidate_empty_no_regression": True,
    "candidate_no_format_regression": True,
    "candidate_no_loop_regression": True,
    "candidate_resolution_floor": True,
}


def _model_fingerprint(contract: Mapping[str, Any]) -> dict[str, Any]:
    artifacts = contract.get("model_artifacts")
    if not isinstance(artifacts, list):
        raise ValueError("model contract is missing bound weight artifacts")
    normalized_artifacts = []
    for artifact in artifacts:
        if (
            not isinstance(artifact, Mapping)
            or not isinstance(artifact.get("sha256"), str)
            or type(artifact.get("bytes")) is not int
        ):
            raise ValueError("model weight artifact binding is incomplete")
        normalized_artifacts.append({
            "sha256": artifact["sha256"],
            "bytes": artifact["bytes"],
        })
    return {
        "served_name": contract.get("served_name"),
        "model_config_sha256": contract.get("model_config_sha256"),
        "model_index_sha256": contract.get("model_index_sha256"),
        "model_safetensors_sha256": contract.get(
            "model_safetensors_sha256"
        ),
        "model_artifacts": normalized_artifacts,
    }


def _official_model_failure_ids(
    report: Mapping[str, Any],
    *,
    full_ids: set[str],
    label: str,
) -> set[str]:
    final = report.get("final")
    direct = (
        final.get("model_failure_ids")
        if isinstance(final, Mapping)
        else None
    )
    if direct is not None:
        if (
            not isinstance(direct, list)
            or len(direct) != len(set(direct))
            or any(
                not isinstance(instance_id, str)
                for instance_id in direct
            )
            or not set(direct).issubset(full_ids)
        ):
            raise ValueError(
                f"{label} official model-failure IDs are invalid"
            )
        return set(direct)

    panels = report.get("panels")
    if not isinstance(panels, list):
        raise ValueError(
            f"{label} official model-failure evidence is missing"
        )
    selected: dict[str, bool] = {}
    for panel in panels:
        runs = panel.get("runs") if isinstance(panel, Mapping) else None
        if not isinstance(runs, list) or not runs:
            raise ValueError(
                f"{label} official model-failure runs are incomplete"
            )
        for run in runs:
            ids_binding = (
                run.get("ids") if isinstance(run, Mapping) else None
            )
            official = (
                run.get("official")
                if isinstance(run, Mapping)
                else None
            )
            failure_ids = (
                official.get("model_failure_ids")
                if isinstance(official, Mapping)
                else None
            )
            if (
                not isinstance(ids_binding, Mapping)
                or not isinstance(ids_binding.get("path"), str)
                or dict(ids_binding)
                != _binding(Path(ids_binding["path"]))
                or not isinstance(failure_ids, list)
                or len(failure_ids) != len(set(failure_ids))
                or any(
                    not isinstance(instance_id, str)
                    for instance_id in failure_ids
                )
            ):
                raise ValueError(
                    f"{label} official model-failure run is invalid"
                )
            run_ids = set(_read_ids(Path(ids_binding["path"])))
            if (
                not run_ids.issubset(full_ids)
                or not set(failure_ids).issubset(run_ids)
            ):
                raise ValueError(
                    f"{label} official model-failure IDs are out of scope"
                )
            failure_set = set(failure_ids)
            for instance_id in run_ids:
                selected[instance_id] = instance_id in failure_set
    return {
        instance_id
        for instance_id, is_failure in selected.items()
        if is_failure
    }


def _validate_current_bindings(
    value: object,
    *,
    expected_keys: set[str],
    label: str,
) -> dict[str, dict[str, Any]]:
    if not isinstance(value, Mapping) or set(value) != expected_keys:
        raise ValueError(f"completion provenance {label} are incomplete")
    result = {}
    for name, binding in value.items():
        if (
            not isinstance(binding, Mapping)
            or not isinstance(binding.get("path"), str)
        ):
            raise ValueError(
                f"completion provenance {label} binding is incomplete"
            )
        current = _binding(Path(binding["path"]))
        if dict(binding) != current:
            raise ValueError(
                f"completion provenance {label} binding changed"
            )
        result[str(name)] = current
    return result


def _validate_portability_provenance(
    value: object,
    *,
    candidate_model_contract: Mapping[str, Any],
    candidate_name: str = "teacher_sft_v2p11",
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError("completion provenance portability is incomplete")
    gate_binding = value.get("gate")
    if (
        not isinstance(gate_binding, Mapping)
        or not isinstance(gate_binding.get("path"), str)
    ):
        raise ValueError("completion provenance portability is incomplete")
    gate_path = Path(gate_binding["path"])
    if dict(gate_binding) != _binding(gate_path):
        raise ValueError("completion provenance portability binding changed")
    gate = _read_object(gate_path)
    control = gate.get("control")
    candidate = gate.get("candidate")
    candidate_gate_model = (
        candidate.get("model_contract")
        if isinstance(candidate, Mapping)
        else None
    )
    gate_fingerprint = None
    if isinstance(candidate_gate_model, Mapping):
        gate_fingerprint = _model_fingerprint(candidate_gate_model)
        gate_fingerprint.pop("served_name")
    expected_fingerprint = _model_fingerprint(candidate_model_contract)
    expected_fingerprint.pop("served_name")
    if (
        gate.get("schema_version") != 1
        or gate.get("artifact_type")
        != "v2p11_controller_free_portability_gate"
        or gate.get("status") != "complete"
        or gate.get("passed") is not True
        or gate.get("control_name") != "teacher_sft_v2p10"
        or gate.get("candidate_name") != candidate_name
        or gate.get("population") != 10
        or gate.get("evaluation_overlap") != 0
        or gate.get("criteria") != _PORTABILITY_CRITERIA
        or not isinstance(control, Mapping)
        or not isinstance(candidate_gate_model, Mapping)
        or gate_fingerprint != expected_fingerprint
    ):
        raise ValueError(
            "completion provenance portability gate did not pass"
        )
    expected = {
        "gate": dict(gate_binding),
        "passed": True,
        "population": 10,
        "criteria": dict(_PORTABILITY_CRITERIA),
    }
    if dict(value) != expected:
        raise ValueError("completion provenance portability is incomplete")
    return expected


def validate_completion_provenance(
    provenance_path: Path,
    *,
    full_ids_path: Path,
    v2p10_composite_path: Path,
    candidate_model_contract: Mapping[str, Any],
    candidate_name: str = "teacher_sft_v2p11",
) -> dict[str, Any]:
    provenance_path = Path(provenance_path).resolve()
    report = _read_object(provenance_path)
    if (
        report.get("artifact_type")
        == "v2p11r2_completion_provenance"
    ):
        from phaseH_eval.v2p11r2_completion_provenance import (
            validate_completion_provenance as validate_direct_lora,
        )

        return validate_direct_lora(
            provenance_path,
            full_ids_path=full_ids_path,
            v2p10_composite_path=v2p10_composite_path,
            candidate_model_contract=candidate_model_contract,
            candidate_name=candidate_name,
        )
    if (
        report.get("artifact_type")
        == "v2p11r3_behavior_completion_provenance"
    ):
        from phaseH_eval.v2p11r3_behavior_completion_provenance import (
            validate_completion_provenance as validate_behavior_poststage,
        )

        return validate_behavior_poststage(
            provenance_path,
            full_ids_path=full_ids_path,
            v2p10_composite_path=v2p10_composite_path,
            candidate_model_contract=candidate_model_contract,
            candidate_name=candidate_name,
        )
    if (
        report.get("artifact_type")
        == "v2p11r3_completion_provenance"
    ):
        from phaseH_eval.v2p11r3_completion_provenance import (
            validate_completion_provenance as validate_reasoned_direct_lora,
        )

        return validate_reasoned_direct_lora(
            provenance_path,
            full_ids_path=full_ids_path,
            v2p10_composite_path=v2p10_composite_path,
            candidate_model_contract=candidate_model_contract,
            candidate_name=candidate_name,
        )
    fable = report.get("fable")
    exclusion = report.get("evaluation_exclusion")
    final_model = report.get("final_model")
    contracts = _validate_current_bindings(
        report.get("contracts"),
        expected_keys={
            "training",
            "portable_recovery",
            "posttrain",
            "stage_data",
            "v2p10_training_lineage",
        },
        label="contracts",
    )
    markers = _validate_current_bindings(
        report.get("markers"),
        expected_keys={
            "stage_a",
            "recovery",
            "kto",
            "final_merge",
            "posttrain",
        },
        label="markers",
    )
    contract_values = _validate_bound_contract_semantics(
        training_contract_path=Path(contracts["training"]["path"]),
        recovery_contract_path=Path(
            contracts["portable_recovery"]["path"]
        ),
        posttrain_contract_path=Path(contracts["posttrain"]["path"]),
        stage_data_contract_path=Path(contracts["stage_data"]["path"]),
        stage_marker_path=Path(markers["stage_a"]["path"]),
        v2p10_composite_path=v2p10_composite_path,
    )
    v2p10_lineage = validate_v2p10_training_lineage_contract(
        Path(contracts["v2p10_training_lineage"]["path"]),
        composite_path=v2p10_composite_path,
    )
    phase_reports = {}
    for marker_name, phase in (
        ("recovery", "recovery_sft"),
        ("kto", "kto_full"),
        ("final_merge", "final_merge"),
    ):
        phase_reports[marker_name] = _validate_completion_phase_marker(
            Path(markers[marker_name]["path"]),
            expected_phase=phase,
        )
    recovery_artifacts = {
        Path(artifact["path"]).name: Path(artifact["path"]).resolve()
        for artifact in phase_reports["recovery"]["artifacts"]
    }
    recovery_input = _validate_completion_phase_marker(
        recovery_artifacts["v2p11_recovery_sft_inputs.json"],
        expected_phase="recovery_sft_inputs",
    )
    recovery_input_artifacts = {
        Path(artifact["path"]).name: Path(artifact["path"]).resolve()
        for artifact in recovery_input["artifacts"]
    }
    kto_artifacts = {
        Path(artifact["path"]).name: Path(artifact["path"]).resolve()
        for artifact in phase_reports["kto"]["artifacts"]
    }
    kto_input = _validate_completion_phase_marker(
        kto_artifacts["v2p11_kto_inputs.json"],
        expected_phase="kto_full_inputs",
    )
    kto_input_artifacts = {
        Path(artifact["path"]).name: Path(artifact["path"]).resolve()
        for artifact in kto_input["artifacts"]
    }
    recovery_merge = _validate_completion_phase_marker(
        kto_input_artifacts["v2p11_recovery_merge_complete.json"],
        expected_phase="recovery_merge",
    )
    recovery_merge_artifacts = {
        Path(artifact["path"]).name: Path(artifact["path"]).resolve()
        for artifact in recovery_merge["artifacts"]
    }
    final_merge_artifacts = {
        Path(artifact["path"]).name: Path(artifact["path"]).resolve()
        for artifact in phase_reports["final_merge"]["artifacts"]
    }
    if (
        recovery_input_artifacts["v2p11_training_complete.json"]
        != Path(markers["stage_a"]["path"]).resolve()
        or recovery_merge_artifacts[
            "v2p11_recovery_sft_complete.json"
        ]
        != Path(markers["recovery"]["path"]).resolve()
        or final_merge_artifacts[
            "v2p11_recovery_merge_complete.json"
        ]
        != kto_input_artifacts[
            "v2p11_recovery_merge_complete.json"
        ]
        or final_merge_artifacts["v2p11_kto_complete.json"]
        != Path(markers["kto"]["path"]).resolve()
    ):
        raise ValueError(
            "completion provenance phase lineage is inconsistent"
        )
    posttrain_contract = contract_values["posttrain"]
    stage_marker = _read_object(Path(markers["stage_a"]["path"]))
    posttrain_marker = _read_object(Path(markers["posttrain"]["path"]))
    recovery_manifest = recovery_input_artifacts["manifest.json"]
    behavior_manifest = kto_input_artifacts[
        "v2p11_behavior_kto_v2_manifest.json"
    ]
    kto_evidence = kto_artifacts["training_evidence.json"]
    final_merge_audit = final_merge_artifacts[
        "v2p11_final_merge_audit.json"
    ]
    expected_posttrain_marker = {
        "schema_version": 1,
        "complete": True,
        "stage_a_adapter_sha256": stage_marker.get("adapter_sha256"),
        "recovery_manifest_sha256": _binding(
            recovery_manifest
        )["sha256"],
        "behavior_manifest_sha256": _binding(
            behavior_manifest
        )["sha256"],
        "recovery_optimizer_steps": 105,
        "kto_optimizer_steps": 25,
        "recovery_input_marker_sha256": _binding(
            recovery_artifacts["v2p11_recovery_sft_inputs.json"]
        )["sha256"],
        "kto_input_marker_sha256": _binding(
            kto_artifacts["v2p11_kto_inputs.json"]
        )["sha256"],
        "kto_training_evidence_sha256": _binding(
            kto_evidence
        )["sha256"],
        "final_merge_audit_sha256": _binding(
            final_merge_audit
        )["sha256"],
        "final_merge_marker_sha256": _binding(
            Path(markers["final_merge"]["path"])
        )["sha256"],
    }
    if any(
        posttrain_marker.get(key) != value
        for key, value in expected_posttrain_marker.items()
    ):
        raise ValueError(
            "completion provenance posttrain marker is inconsistent"
        )
    if (
        posttrain_contract.get("recovery_manifest_sha256")
        != _binding(recovery_manifest)["sha256"]
        or posttrain_contract.get("behavior_manifest_sha256")
        != _binding(behavior_manifest)["sha256"]
    ):
        raise ValueError(
            "completion provenance posttrain manifests are inconsistent"
        )
    portability = _validate_portability_provenance(
        report.get("portability"),
        candidate_model_contract=candidate_model_contract,
        candidate_name=candidate_name,
    )
    if (
        report.get("schema_version") != 1
        or report.get("artifact_type")
        != "v2p11_completion_provenance"
        or report.get("status") != "complete"
        or report.get("full_ids") != _binding(full_ids_path)
        or report.get("v2p10_full300")
        != _binding(v2p10_composite_path)
        or fable
        != {
            "stage_a": {
                "inherited_pinned_rows": 31,
                "historical_verified_revision_rows": 10,
                "new_strict_rows": 36,
                "total_rows": 77,
            },
            "extracted_campaign": {
                "attempted": 60,
                "resolved": 47,
                "collection_rejected": 13,
                "strict_admitted": 36,
                "distillation_rejected": 11,
            },
            "recovery_unique_sources": 46,
            "targeted_recovery_rows": 138,
        }
        or not isinstance(exclusion, Mapping)
        or exclusion.get("full_ids") != 300
        or exclusion.get("excluded_ids") != 707
        or exclusion.get("excluded_repositories") != 12
        or exclusion.get("stage_a_overlap") != 0
        or exclusion.get("overlap") != 0
        or not isinstance(exclusion.get("sha256"), str)
        or not isinstance(final_model, Mapping)
    ):
        raise ValueError("v2.11 completion provenance is incomplete")
    candidate_model_path = candidate_model_contract.get("model_path")
    provenance_model_path = final_model.get("model_path")
    if (
        not isinstance(candidate_model_path, str)
        or not isinstance(provenance_model_path, str)
        or final_merge_artifacts["config.json"].parent.resolve()
        != Path(candidate_model_path).resolve()
        or Path(provenance_model_path).resolve()
        != Path(candidate_model_path).resolve()
    ):
        raise ValueError(
            "completion provenance does not bind evaluated candidate path"
        )
    expected_fingerprint = _model_fingerprint(candidate_model_contract)
    expected_fingerprint.pop("served_name")
    current_fingerprint = _model_fingerprint({
        **dict(final_model),
        "served_name": candidate_model_contract.get("served_name"),
    })
    current_fingerprint.pop("served_name")
    if current_fingerprint != expected_fingerprint:
        raise ValueError(
            "completion provenance does not bind evaluated candidate weights"
        )
    return {
        "artifact": _binding(provenance_path),
        "fable": dict(fable),
        "evaluation_exclusion": dict(exclusion),
        "training": report.get("training"),
        "behavior_kto": report.get("behavior_kto"),
        "final_model": dict(final_model),
        "contracts": contracts,
        "markers": markers,
        "portability": portability,
        "v2p10_training": dict(v2p10_lineage["training"]),
    }


def _behavior_regression(
    control: Mapping[str, Any],
    candidate: Mapping[str, Any],
) -> dict[str, Any]:
    control_health = control.get("behavior_health")
    candidate_health = candidate.get("behavior_health")
    required = (
        "repeat_loops",
        "tool_format_errors",
        "assistant_responses",
        "format_error_rate",
    )
    if (
        not isinstance(control_health, Mapping)
        or not isinstance(candidate_health, Mapping)
        or any(key not in control_health for key in required)
        or any(key not in candidate_health for key in required)
    ):
        raise ValueError("full-300 behavior health is incomplete")
    for health in (control_health, candidate_health):
        if (
            any(
                type(health.get(key)) is not int or health[key] < 0
                for key in required[:3]
            )
            or (
                health["format_error_rate"] is not None
                and not isinstance(health["format_error_rate"], (int, float))
            )
        ):
            raise ValueError("full-300 behavior health is invalid")
    control_rate = control_health["format_error_rate"]
    candidate_rate = candidate_health["format_error_rate"]
    no_rate_regression = (
        candidate_rate is None
        if control_rate is None
        else candidate_rate is not None and candidate_rate <= control_rate
    )
    no_loop_regression = (
        candidate_health["repeat_loops"]
        <= control_health["repeat_loops"]
    )
    no_format_count_regression = (
        candidate_health["tool_format_errors"]
        <= control_health["tool_format_errors"]
    )
    return {
        "v2p10": dict(control_health),
        "v2p11": dict(candidate_health),
        "repeat_loop_delta": (
            candidate_health["repeat_loops"]
            - control_health["repeat_loops"]
        ),
        "tool_format_error_delta": (
            candidate_health["tool_format_errors"]
            - control_health["tool_format_errors"]
        ),
        "format_error_rate_delta": (
            candidate_rate - control_rate
            if candidate_rate is not None and control_rate is not None
            else None
        ),
        "no_repeat_loop_regression": no_loop_regression,
        "no_tool_format_error_count_regression": (
            no_format_count_regression
        ),
        "no_format_error_rate_regression": no_rate_regression,
        "healthy": bool(
            no_loop_regression
            and no_format_count_regression
            and no_rate_regression
        ),
    }


def _behavior_failure_ids(
    composite: Mapping[str, Any],
    key: str,
) -> list[str]:
    by_instance = composite.get("behavior_health_by_instance")
    if not isinstance(by_instance, Mapping):
        raise ValueError("full-300 per-instance behavior health is missing")
    return sorted(
        instance_id
        for instance_id, health in by_instance.items()
        if isinstance(instance_id, str)
        and isinstance(health, Mapping)
        and isinstance(health.get(key), int)
        and health[key] > 0
    )


def _validate_panel_full300(
    *,
    full_ids_path: Path,
    composite_path: Path,
    expected_name: str,
    predictions_path: Path | None = None,
    validate_retry_policy: bool = False,
) -> dict[str, Any]:
    composite = _read_object(composite_path)
    panels = composite.get("panels")
    input_artifacts = composite.get("input_artifacts")
    predictions_artifact = composite.get("predictions_artifact")
    if (
        composite.get("schema_version") != 1
        or composite.get("artifact_type")
        != "disjoint_panel_full300_composite"
        or composite.get("status") != "complete"
        or composite.get("name") != expected_name
        or not isinstance(panels, list)
        or len(panels) != 2
        or not isinstance(input_artifacts, list)
        or len(input_artifacts) != 7
        or input_artifacts[0] != _binding(full_ids_path)
        or not isinstance(predictions_artifact, Mapping)
        or not isinstance(predictions_artifact.get("path"), str)
    ):
        raise ValueError(f"{expected_name} panel full-300 contract is incomplete")
    if predictions_path is not None and (
        Path(predictions_artifact["path"]).resolve()
        != predictions_path.resolve()
        or predictions_artifact != _binding(predictions_path)
    ):
        raise ValueError(
            f"{expected_name} full-300 predictions binding differs"
        )
    panel_inputs: list[PanelInput] = []
    panel_contracts: dict[str, dict[str, Any]] = {}
    for index, panel in enumerate(panels):
        if not isinstance(panel, Mapping) or not isinstance(
            panel.get("tag"),
            str,
        ):
            raise ValueError(
                f"{expected_name} panel metadata is incomplete"
            )
        ids_binding, composite_binding, predictions_binding = (
            input_artifacts[1 + index * 3 : 4 + index * 3]
        )
        if (
            not isinstance(ids_binding, Mapping)
            or not isinstance(composite_binding, Mapping)
            or not isinstance(predictions_binding, Mapping)
            or not all(
                isinstance(binding.get("path"), str)
                for binding in (
                    ids_binding,
                    composite_binding,
                    predictions_binding,
                )
            )
            or ids_binding.get("sha256") != panel.get("ids_sha256")
            or composite_binding.get("sha256")
            != panel.get("composite_sha256")
            or predictions_binding.get("sha256")
            != panel.get("predictions_sha256")
        ):
            raise ValueError(
                f"{expected_name} panel artifact binding changed"
            )
        panel_inputs.append(PanelInput(
            tag=panel["tag"],
            ids_path=Path(ids_binding["path"]),
            composite_path=Path(composite_binding["path"]),
            predictions_path=Path(predictions_binding["path"]),
        ))
        if validate_retry_policy:
            panel_contracts[panel["tag"]] = _validate_composite_source(
                ids_path=Path(ids_binding["path"]),
                composite_path=Path(composite_binding["path"]),
                predictions_path=Path(predictions_binding["path"]),
            )
    verified = verify_disjoint_panels(
        full_ids_path=full_ids_path,
        panels=panel_inputs,
        output_path=composite_path,
        predictions_path=Path(predictions_artifact["path"]),
    )
    if verified.get("name") != expected_name:
        raise ValueError(
            f"{expected_name} panel full-300 model identity differs"
        )
    return {
        "composite": verified,
        "predictions": _read_object(
            Path(predictions_artifact["path"])
        ),
        "panel_contracts": panel_contracts,
    }


def _validate_v2p10(
    *,
    full_ids_path: Path,
    composite_path: Path,
    predictions_path: Path | None = None,
    validate_retry_policy: bool = False,
) -> dict[str, Any]:
    return _validate_panel_full300(
        full_ids_path=full_ids_path,
        composite_path=composite_path,
        expected_name="teacher_sft_v2p10",
        predictions_path=predictions_path,
        validate_retry_policy=validate_retry_policy,
    )


def _empty_ids(composite: Mapping[str, Any]) -> set[str]:
    categories = _acceptance_empty_ids(composite)
    return {
        instance_id
        for values in categories.values()
        for instance_id in values
    }


def _validate_first_pass(
    *,
    full_ids_path: Path,
    panels: Sequence[FirstPassPanel],
    expected_name: str,
) -> dict[str, Any]:
    full_ids = _read_ids(full_ids_path)
    if (
        len(panels) != 2
        or {panel.tag for panel in panels}
        != {"fixed150", "complement150"}
    ):
        raise ValueError(
            f"{expected_name} first-pass panels must be fixed150 and "
            "complement150"
        )
    validated: list[dict[str, Any]] = []
    concatenated: list[str] = []
    for panel in panels:
        run = _complete_run(
            ids_path=panel.ids_path,
            run_root=panel.run_root,
            label=f"{expected_name}_{panel.tag}_first_pass",
        )
        if run["acceptance"].get("name") != expected_name:
            raise ValueError(
                f"{expected_name} first-pass model identity differs"
            )
        concatenated.extend(run["ids"])
        validated.append({
            "input": panel,
            "run": run,
        })
    if (
        len(concatenated) != len(set(concatenated))
        or set(concatenated) != set(full_ids)
    ):
        raise ValueError(
            f"{expected_name} first-pass panels are not an exact full-300 "
            "partition"
        )
    harness_contract = validated[0]["run"]["harness_contract"]
    model_contract = validated[0]["run"]["model_contract"]
    for row in validated[1:]:
        if row["run"]["harness_contract"] != harness_contract:
            raise ValueError(
                f"{expected_name} first-pass harness differs by panel"
            )
        if row["run"]["model_contract"] != model_contract:
            raise ValueError(
                f"{expected_name} first-pass model differs by panel"
            )
    resolved_ids = set().union(
        *(set(row["run"]["resolved_ids"]) for row in validated)
    )
    empty_ids = set().union(
        *(set(row["run"]["empty_ids"]) for row in validated)
    )
    return {
        "name": expected_name,
        "harness_contract": harness_contract,
        "model_contract": model_contract,
        "resolved_ids": resolved_ids,
        "empty_ids": empty_ids,
        "panels": [
            {
                "tag": row["input"].tag,
                "ids": _binding(row["input"].ids_path),
                "run_root": str(row["input"].run_root.resolve()),
                "run_manifest": _binding(
                    row["input"].run_root / "run_manifest.json"
                ),
                "eval_manifest": _binding(
                    row["input"].run_root / "eval_manifest.json"
                ),
                "acceptance": _binding(
                    row["input"].run_root / "acceptance.json"
                ),
                "predictions": _binding(
                    row["input"].run_root / "preds_all.json"
                ),
            }
            for row in validated
        ],
    }


def _compare_first_pass(
    *,
    full_ids_path: Path,
    v2p10_panels: Sequence[FirstPassPanel],
    v2p11_panels: Sequence[FirstPassPanel],
    candidate_name: str = "teacher_sft_v2p11",
) -> dict[str, Any]:
    control = _validate_first_pass(
        full_ids_path=full_ids_path,
        panels=v2p10_panels,
        expected_name="teacher_sft_v2p10",
    )
    candidate = _validate_first_pass(
        full_ids_path=full_ids_path,
        panels=v2p11_panels,
        expected_name=candidate_name,
    )
    if control["harness_contract"] != candidate["harness_contract"]:
        raise ValueError("first-pass harness contract mismatch")
    full_set = set(_read_ids(full_ids_path))
    control_resolved = control["resolved_ids"]
    candidate_resolved = candidate["resolved_ids"]
    return {
        "population": len(full_set),
        "harness_contract": candidate["harness_contract"],
        "v2p10": {
            "resolved": len(control_resolved),
            "empty": len(control["empty_ids"]),
            "panels": control["panels"],
            "model_fingerprint": _model_fingerprint(
                control["model_contract"]
            ),
        },
        "v2p11": {
            "resolved": len(candidate_resolved),
            "empty": len(candidate["empty_ids"]),
            "panels": candidate["panels"],
            "model_fingerprint": _model_fingerprint(
                candidate["model_contract"]
            ),
        },
        "paired": {
            "v2p11_only": sorted(
                candidate_resolved - control_resolved
            ),
            "v2p10_only": sorted(
                control_resolved - candidate_resolved
            ),
            "both_resolved": sorted(
                candidate_resolved & control_resolved
            ),
            "neither_resolved": sorted(
                full_set - candidate_resolved - control_resolved
            ),
        },
        "empty_patch": {
            "v2p10": sorted(control["empty_ids"]),
            "v2p11": sorted(candidate["empty_ids"]),
            "eliminated": sorted(
                control["empty_ids"] - candidate["empty_ids"]
            ),
            "introduced": sorted(
                candidate["empty_ids"] - control["empty_ids"]
            ),
        },
        "verdict": {
            "beats_v2p10": (
                len(candidate_resolved) > len(control_resolved)
            ),
            "resolved_delta": (
                len(candidate_resolved) - len(control_resolved)
            ),
            "empty_delta": (
                len(candidate["empty_ids"]) - len(control["empty_ids"])
            ),
        },
    }


def _matched_retry_policy(
    validated: Mapping[str, Any],
    *,
    label: str,
) -> dict[str, Any]:
    panel_contracts = validated.get("panel_contracts")
    if (
        not isinstance(panel_contracts, Mapping)
        or set(panel_contracts) != {"fixed150", "complement150"}
    ):
        raise ValueError(f"{label} corrected panel policy is incomplete")
    policy: dict[str, Any] = {}
    for tag, allowed_generation in (
        ("fixed150", 2),
        ("complement150", 1),
    ):
        panel = panel_contracts[tag]
        generation = panel["retry_generation"]
        remaining_empty = len(panel["empty_ids"])
        if generation > allowed_generation:
            raise ValueError(f"{label} {tag} exceeded retry policy")
        if (
            tag == "fixed150"
            and generation < allowed_generation
            and remaining_empty
        ):
            raise ValueError(
                f"{label} fixed150 stopped before retry generation 2"
            )
        if tag == "complement150" and generation != 1:
            raise ValueError(
                f"{label} complement150 retry depth differs"
            )
        policy[tag] = {
            "allowed_retry_generations": allowed_generation,
            "actual_retry_generation": generation,
            "remaining_empty": remaining_empty,
            "selected_attempts": panel["selected_attempts"],
        }
    return policy


def compare_full300(
    *,
    full_ids_path: Path,
    v2p10_composite_path: Path,
    v2p11_composite_path: Path,
    v2p11_predictions_path: Path,
    candidate_name: str = "teacher_sft_v2p11",
    candidate_model_path: Path | None = None,
    v2p10_predictions_path: Path | None = None,
    v2p10_score_binding_path: Path | None = None,
    v2p11_score_binding_path: Path | None = None,
    v2p10_first_pass_panels: Sequence[FirstPassPanel] | None = None,
    v2p11_first_pass_panels: Sequence[FirstPassPanel] | None = None,
    provenance_path: Path | None = None,
) -> dict[str, Any]:
    full_ids_path = Path(full_ids_path).resolve()
    v2p10_composite_path = Path(v2p10_composite_path).resolve()
    v2p11_composite_path = Path(v2p11_composite_path).resolve()
    v2p11_predictions_path = Path(v2p11_predictions_path).resolve()
    if candidate_model_path is not None:
        candidate_model_path = Path(candidate_model_path).resolve()
    if v2p10_predictions_path is not None:
        v2p10_predictions_path = Path(v2p10_predictions_path).resolve()
    full_ids = _read_ids(full_ids_path)
    if len(full_ids) != 300:
        raise ValueError("full-300 comparison requires exactly 300 IDs")
    require_matched_policy = bool(
        v2p10_first_pass_panels is not None
        or v2p11_first_pass_panels is not None
    )
    if require_matched_policy and provenance_path is None:
        raise ValueError(
            "matched comparison requires completion provenance"
        )
    if require_matched_policy and (
        v2p10_score_binding_path is None
        or v2p11_score_binding_path is None
    ):
        raise ValueError(
            "matched comparison requires official score bindings"
        )
    v2p10 = _validate_v2p10(
        full_ids_path=full_ids_path,
        composite_path=v2p10_composite_path,
        predictions_path=v2p10_predictions_path,
        validate_retry_policy=require_matched_policy,
    )
    candidate_artifact = _read_object(v2p11_composite_path)
    if (
        candidate_artifact.get("artifact_type")
        == "disjoint_panel_full300_composite"
    ):
        v2p11 = _validate_panel_full300(
            full_ids_path=full_ids_path,
            composite_path=v2p11_composite_path,
            expected_name=candidate_name,
            predictions_path=v2p11_predictions_path,
            validate_retry_policy=require_matched_policy,
        )
    else:
        legacy = _validate_composite_source(
            ids_path=full_ids_path,
            composite_path=v2p11_composite_path,
            predictions_path=v2p11_predictions_path,
        )
        v2p11 = {
            "composite": legacy["composite"],
            "predictions": legacy["predictions"],
            "panel_contracts": None,
        }
    control = v2p10["composite"]
    candidate = v2p11["composite"]
    evaluated_model_path = Path(
        str(candidate.get("model_contract", {}).get("model_path", ""))
    )
    if (
        candidate.get("name") != candidate_name
        or candidate.get("expected") != 300
        or candidate.get("model_contract", {}).get("served_name")
        != candidate_name
        or (
            candidate_model_path is None
            and not str(evaluated_model_path).endswith(
                "/teacher_sft_v2p11_full"
            )
        )
        or (
            candidate_model_path is not None
            and evaluated_model_path.resolve() != candidate_model_path
        )
    ):
        raise ValueError("v2.11 full-300 contract is incomplete")
    wrong_models = sorted(
        instance_id
        for instance_id, row in v2p11["predictions"].items()
        if row.get("model_name_or_path")
        not in {candidate_name, f"openai/{candidate_name}"}
    )
    if wrong_models:
        raise ValueError("v2.11 prediction model identity differs")
    if candidate.get("harness_contract") != control.get(
        "harness_contract"
    ):
        raise ValueError("harness contract mismatch")
    provenance = (
        validate_completion_provenance(
            provenance_path,
            full_ids_path=full_ids_path,
            v2p10_composite_path=v2p10_composite_path,
            candidate_model_contract=candidate["model_contract"],
            candidate_name=candidate_name,
        )
        if provenance_path is not None
        else None
    )
    official_scores = None
    official_model_failure_ids = {
        "v2p10": set(),
        "v2p11": set(),
    }
    if (
        v2p10_score_binding_path is not None
        and v2p11_score_binding_path is not None
    ):
        if v2p10_predictions_path is None:
            raise ValueError(
                "v2p10 official score binding requires predictions"
            )
        v2p10_score_binding_path = Path(
            v2p10_score_binding_path
        ).resolve()
        v2p11_score_binding_path = Path(
            v2p11_score_binding_path
        ).resolve()
        score_reports = {
            "v2p10": validate_official_score_binding(
                v2p10_score_binding_path,
                full_ids_path=full_ids_path,
                composite_path=v2p10_composite_path,
                predictions_path=v2p10_predictions_path,
            ),
            "v2p11": validate_official_score_binding(
                v2p11_score_binding_path,
                full_ids_path=full_ids_path,
                composite_path=v2p11_composite_path,
                predictions_path=v2p11_predictions_path,
            ),
        }
        for key, score_report, composite in (
            ("v2p10", score_reports["v2p10"], control),
            ("v2p11", score_reports["v2p11"], candidate),
        ):
            final = score_report.get("final")
            if (
                score_report.get("name") != composite["name"]
                or not isinstance(final, Mapping)
                or final.get("resolved_ids")
                != composite["resolved_ids"]
                or final.get("resolved") != composite["resolved"]
                or final.get("empty")
                != composite["empty_split"]["total"]
            ):
                raise ValueError(
                    f"{key} official score binding differs from composite"
                )
            official_model_failure_ids[key] = (
                _official_model_failure_ids(
                    score_report,
                    full_ids=set(full_ids),
                    label=key,
                )
            )
        official_scores = {
            "v2p10": _binding(v2p10_score_binding_path),
            "v2p11": _binding(v2p11_score_binding_path),
        }
    elif (
        v2p10_score_binding_path is not None
        or v2p11_score_binding_path is not None
    ):
        raise ValueError("official score bindings must be paired")
    behavior_regression = _behavior_regression(control, candidate)

    first_pass = None
    retry_policy = None
    if (
        v2p10_first_pass_panels is None
        and v2p11_first_pass_panels is None
    ):
        pass
    elif (
        v2p10_first_pass_panels is None
        or v2p11_first_pass_panels is None
    ):
        raise ValueError(
            "both v2p10 and v2p11 first-pass panels are required"
        )
    else:
        if v2p11["panel_contracts"] is None:
            raise ValueError(
                "matched comparison requires panelized v2p11 corrected data"
            )
        first_pass = _compare_first_pass(
            full_ids_path=full_ids_path,
            v2p10_panels=v2p10_first_pass_panels,
            v2p11_panels=v2p11_first_pass_panels,
            candidate_name=candidate_name,
        )
        retry_policy = {
            "v2p10": _matched_retry_policy(v2p10, label="v2p10"),
            "v2p11": _matched_retry_policy(v2p11, label="v2p11"),
        }
        if first_pass["v2p10"]["model_fingerprint"] != (
            _model_fingerprint(control["model_contract"])
        ):
            raise ValueError(
                "v2p10 first-pass and corrected model weights differ"
            )
        if first_pass["v2p11"]["model_fingerprint"] != (
            _model_fingerprint(candidate["model_contract"])
        ):
            raise ValueError(
                "v2p11 first-pass and corrected model weights differ"
            )

    full_set = set(full_ids)
    v2p10_resolved = set(control["resolved_ids"])
    v2p11_resolved = set(candidate["resolved_ids"])
    v2p10_empty = _empty_ids(control)
    v2p11_empty = _empty_ids(candidate)
    v2p11_only = v2p11_resolved - v2p10_resolved
    v2p10_only = v2p10_resolved - v2p11_resolved
    paired_win_delta = len(v2p11_only) - len(v2p10_only)
    failure_analysis = {
        "v2p10_nonempty_unresolved": sorted(
            full_set
            - v2p10_resolved
            - v2p10_empty
            - official_model_failure_ids["v2p10"]
        ),
        "v2p11_nonempty_unresolved": sorted(
            full_set
            - v2p11_resolved
            - v2p11_empty
            - official_model_failure_ids["v2p11"]
        ),
        "v2p10_model_failure_ids": sorted(
            official_model_failure_ids["v2p10"]
        ),
        "v2p11_model_failure_ids": sorted(
            official_model_failure_ids["v2p11"]
        ),
        "v2p10_repeat_loop_ids": _behavior_failure_ids(
            control,
            "repeat_loops",
        ),
        "v2p11_repeat_loop_ids": _behavior_failure_ids(
            candidate,
            "repeat_loops",
        ),
        "v2p10_tool_format_error_ids": _behavior_failure_ids(
            control,
            "tool_format_errors",
        ),
        "v2p11_tool_format_error_ids": _behavior_failure_ids(
            candidate,
            "tool_format_errors",
        ),
    }
    by_instance: dict[str, dict[str, list[str]]] = {}
    for instance_id in full_ids:
        model_reasons = {}
        for label, resolved, empty in (
            ("v2p10", v2p10_resolved, v2p10_empty),
            ("v2p11", v2p11_resolved, v2p11_empty),
        ):
            reasons = []
            if instance_id not in resolved:
                if instance_id in empty:
                    reasons.append("empty_patch")
                elif (
                    instance_id
                    in official_model_failure_ids[label]
                ):
                    reasons.append("model_failure")
                else:
                    reasons.append("nonempty_unresolved")
            if (
                instance_id
                in failure_analysis[f"{label}_repeat_loop_ids"]
            ):
                reasons.append("repeat_loop")
            if (
                instance_id
                in failure_analysis[
                    f"{label}_tool_format_error_ids"
                ]
            ):
                reasons.append("tool_format_error")
            if reasons:
                model_reasons[label] = reasons
        if model_reasons:
            by_instance[instance_id] = model_reasons
    failure_analysis["by_instance"] = by_instance
    report = {
        "schema_version": 1,
        "artifact_type": "v2p11_v2p10_full300_verdict",
        "status": "complete",
        "population": len(full_ids),
        "full_ids": _binding(full_ids_path),
        "harness_contract": candidate["harness_contract"],
        "v2p10": {
            "name": control["name"],
            "resolved": len(v2p10_resolved),
            "empty": len(v2p10_empty),
            "composite": _binding(v2p10_composite_path),
            "predictions": control["predictions_artifact"],
            "behavior_health": control["behavior_health"],
            "empty_resampling": control["empty_resampling"],
        },
        "v2p11": {
            "name": candidate["name"],
            "resolved": len(v2p11_resolved),
            "empty": len(v2p11_empty),
            "composite": _binding(v2p11_composite_path),
            "predictions": _binding(v2p11_predictions_path),
            "behavior_health": candidate["behavior_health"],
            "empty_resampling": candidate["empty_resampling"],
        },
        "paired": {
            "v2p11_only": sorted(v2p11_only),
            "v2p10_only": sorted(v2p10_only),
            "both_resolved": sorted(v2p11_resolved & v2p10_resolved),
            "neither_resolved": sorted(
                full_set - v2p11_resolved - v2p10_resolved
            ),
        },
        "empty_patch": {
            "v2p10": sorted(v2p10_empty),
            "v2p11": sorted(v2p11_empty),
            "eliminated": sorted(v2p10_empty - v2p11_empty),
            "introduced": sorted(v2p11_empty - v2p10_empty),
        },
        "failure_analysis": failure_analysis,
        "behavior_regression": behavior_regression,
        "verdict": {
            "beats_v2p10": len(v2p11_resolved) > len(v2p10_resolved),
            "resolved_delta": len(v2p11_resolved) - len(v2p10_resolved),
            "paired_win_delta": paired_win_delta,
            "behavior_healthy": behavior_regression["healthy"],
        },
    }
    if provenance is not None:
        report["provenance"] = provenance
    if official_scores is not None:
        report["official_scores"] = official_scores
    if first_pass is not None:
        report["first_pass"] = first_pass
        report["retry_policy"] = retry_policy
        report["verdict"]["first_pass_beats_v2p10"] = first_pass[
            "verdict"
        ]["beats_v2p10"]
        report["verdict"]["first_pass_resolved_delta"] = first_pass[
            "verdict"
        ]["resolved_delta"]
        report["verdict"]["first_pass_empty_delta"] = first_pass[
            "verdict"
        ]["empty_delta"]
        report["verdict"]["corrected_empty_delta"] = (
            len(v2p11_empty) - len(v2p10_empty)
        )
        report["verdict"]["first_pass_no_new_empty_ids"] = not bool(
            first_pass["empty_patch"]["introduced"]
        )
        report["verdict"]["corrected_no_new_empty_ids"] = not bool(
            report["empty_patch"]["introduced"]
        )
        report["verdict"]["trustworthy_beats_v2p10"] = bool(
            report["verdict"]["beats_v2p10"]
            and report["verdict"]["first_pass_beats_v2p10"]
            and report["verdict"]["first_pass_empty_delta"] <= 0
            and report["verdict"]["corrected_empty_delta"] <= 0
            and report["verdict"]["first_pass_no_new_empty_ids"]
            and report["verdict"]["corrected_no_new_empty_ids"]
            and report["verdict"]["behavior_healthy"]
            and provenance is not None
            and official_scores is not None
        )
    return report


def _markdown(report: Mapping[str, Any]) -> str:
    verdict = report["verdict"]
    paired = report["paired"]
    empty = report["empty_patch"]
    behavior = report["behavior_regression"]
    lines = [
        "# v2.11 vs v2.10 — SWE-bench Lite 300",
        "",
    ]
    first_pass = report.get("first_pass")
    if isinstance(first_pass, Mapping):
        lines.extend([
            "## First pass",
            "",
            "| Model | Resolved | Empty patches |",
            "|---|---:|---:|",
            (
                f"| v2.10 | {first_pass['v2p10']['resolved']}/300 | "
                f"{first_pass['v2p10']['empty']} |"
            ),
            (
                f"| v2.11 | {first_pass['v2p11']['resolved']}/300 | "
                f"{first_pass['v2p11']['empty']} |"
            ),
            "",
            (
                "First-pass resolved delta: "
                f"{verdict['first_pass_resolved_delta']:+d}. "
                "First-pass empty delta: "
                f"{verdict['first_pass_empty_delta']:+d}. "
                "First-pass beats v2.10: "
                f"{str(verdict['first_pass_beats_v2p10']).lower()}."
            ),
            "",
            "## Corrected with matched empty-retry policy",
            "",
        ])
    lines.extend([
        "| Model | Corrected resolved | Remaining empty patches |",
        "|---|---:|---:|",
        (
            f"| v2.10 | {report['v2p10']['resolved']}/300 | "
            f"{report['v2p10']['empty']} |"
        ),
        (
            f"| v2.11 | {report['v2p11']['resolved']}/300 | "
            f"{report['v2p11']['empty']} |"
        ),
        "",
        (
            f"Resolved delta: {verdict['resolved_delta']:+d}. "
            f"Paired win delta: {verdict['paired_win_delta']:+d}. "
            + (
                "Corrected empty delta: "
                f"{verdict['corrected_empty_delta']:+d}. "
                if "corrected_empty_delta" in verdict
                else ""
            )
            + f"Beats v2.10: {str(verdict['beats_v2p10']).lower()}."
        ),
        "",
        (
            f"v2.11-only wins: {len(paired['v2p11_only'])}; "
            f"v2.10-only wins: {len(paired['v2p10_only'])}; "
            f"both resolved: {len(paired['both_resolved'])}; "
            f"neither resolved: {len(paired['neither_resolved'])}."
        ),
        (
            f"Empty patches eliminated: {len(empty['eliminated'])}; "
            f"introduced: {len(empty['introduced'])}."
        ),
        (
            "Repeated-failure loop delta: "
            f"{behavior['repeat_loop_delta']:+d}; "
            "tool-format error delta: "
            f"{behavior['tool_format_error_delta']:+d}; "
            "behavior healthy: "
            f"{str(behavior['healthy']).lower()}."
        ),
        "",
    ])
    if "trustworthy_beats_v2p10" in verdict:
        lines.extend([
            (
                "Trustworthy promotion verdict: "
                f"{str(verdict['trustworthy_beats_v2p10']).lower()}."
            ),
            "",
        ])
    retry_policy = report.get("retry_policy")
    if isinstance(retry_policy, Mapping):
        lines.extend([
            "## Matched retry policy",
            "",
            "| Model | Panel | Allowed generations | Actual generation | "
            "Remaining empty |",
            "|---|---|---:|---:|---:|",
        ])
        for model in ("v2p10", "v2p11"):
            for panel in ("fixed150", "complement150"):
                policy = retry_policy[model][panel]
                lines.append(
                    f"| {model} | {panel} | "
                    f"{policy['allowed_retry_generations']} | "
                    f"{policy['actual_retry_generation']} | "
                    f"{policy['remaining_empty']} |"
                )
        lines.append("")
    failure = report["failure_analysis"]
    lines.extend([
        "## Failure analysis",
        "",
        "| Model | Non-empty unresolved | Model failures | "
        "Repeat-loop instances | "
        "Tool-format-error instances |",
        "|---|---:|---:|---:|---:|",
        (
            f"| v2.10 | {len(failure['v2p10_nonempty_unresolved'])} | "
            f"{len(failure['v2p10_model_failure_ids'])} | "
            f"{len(failure['v2p10_repeat_loop_ids'])} | "
            f"{len(failure['v2p10_tool_format_error_ids'])} |"
        ),
        (
            f"| v2.11 | {len(failure['v2p11_nonempty_unresolved'])} | "
            f"{len(failure['v2p11_model_failure_ids'])} | "
            f"{len(failure['v2p11_repeat_loop_ids'])} | "
            f"{len(failure['v2p11_tool_format_error_ids'])} |"
        ),
        "",
        "Per-instance failure reasons are recorded in the JSON verdict.",
        "",
    ])
    provenance = report.get("provenance")
    if isinstance(provenance, Mapping):
        fable = provenance["fable"]
        final_model = provenance["final_model"]
        official_scores = report.get("official_scores")
        lines.extend([
            "## Bound provenance",
            "",
            (
                f"Exact evaluation population: {report['population']} IDs; "
                f"IDs SHA-256 `{report['full_ids']['sha256']}`."
            ),
            (
                "Completion provenance SHA-256 "
                f"`{provenance['artifact']['sha256']}`."
            ),
            *(
                [
                    (
                        "Official score bindings SHA-256: v2.10 "
                        f"`{official_scores['v2p10']['sha256']}`; "
                        "v2.11 "
                        f"`{official_scores['v2p11']['sha256']}`."
                    )
                ]
                if isinstance(official_scores, Mapping)
                else []
            ),
            (
                "Final model config SHA-256 "
                f"`{final_model['model_config_sha256']}`; index SHA-256 "
                f"`{final_model['model_index_sha256']}`."
            ),
            (
                "Fable lineage: "
                f"{fable['stage_a']['total_rows']} Stage-A rows "
                f"({fable['stage_a']['new_strict_rows']} newly strict); "
                f"{fable['recovery_unique_sources']} recovery sources expanded "
                f"to {fable['targeted_recovery_rows']} targeted rows."
            ),
            (
                "Evaluation overlap: "
                f"{provenance['evaluation_exclusion']['overlap']}."
            ),
            "",
        ])
    return "\n".join(lines)


def publish_verdict_outputs(
    report: Mapping[str, Any],
    *,
    output_path: Path,
    markdown_output_path: Path,
) -> None:
    if os.path.lexists(output_path) or os.path.lexists(
        markdown_output_path
    ):
        raise FileExistsError(
            "refusing to overwrite paired verdict outputs"
        )
    json_payload = (
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    ).encode()
    markdown_payload = _markdown(report).encode()
    json_created = False
    try:
        _publish_bytes_noreplace(output_path, json_payload)
        json_created = True
        _publish_bytes_noreplace(
            markdown_output_path,
            markdown_payload,
        )
    except BaseException:
        if json_created:
            output_path.unlink(missing_ok=True)
        raise


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full-ids", type=Path, required=True)
    parser.add_argument("--v2p10", type=Path, required=True)
    parser.add_argument("--v2p10-preds", type=Path)
    parser.add_argument("--v2p10-score-binding", type=Path)
    parser.add_argument("--v2p11", type=Path, required=True)
    parser.add_argument("--v2p11-preds", type=Path, required=True)
    parser.add_argument("--v2p11-score-binding", type=Path)
    parser.add_argument(
        "--candidate-name",
        default="teacher_sft_v2p11",
    )
    parser.add_argument("--candidate-model-path", type=Path)
    parser.add_argument(
        "--v2p10-first-pass-panel",
        action="append",
        nargs=3,
        metavar=("TAG", "IDS", "RUN_ROOT"),
    )
    parser.add_argument(
        "--v2p11-first-pass-panel",
        action="append",
        nargs=3,
        metavar=("TAG", "IDS", "RUN_ROOT"),
    )
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--markdown-out", type=Path, required=True)
    parser.add_argument("--provenance", type=Path)
    parser.add_argument("--verify-existing", action="store_true")
    parser.add_argument("--require-trustworthy-win", action="store_true")
    args = parser.parse_args(argv)
    to_panels = lambda rows: (
        [
            FirstPassPanel(
                tag=tag,
                ids_path=Path(ids),
                run_root=Path(run_root),
            )
            for tag, ids, run_root in rows
        ]
        if rows is not None
        else None
    )
    report = compare_full300(
        full_ids_path=args.full_ids,
        v2p10_composite_path=args.v2p10,
        v2p10_predictions_path=args.v2p10_preds,
        v2p10_score_binding_path=args.v2p10_score_binding,
        v2p11_composite_path=args.v2p11,
        v2p11_predictions_path=args.v2p11_preds,
        candidate_name=args.candidate_name,
        candidate_model_path=args.candidate_model_path,
        v2p11_score_binding_path=args.v2p11_score_binding,
        v2p10_first_pass_panels=to_panels(
            args.v2p10_first_pass_panel
        ),
        v2p11_first_pass_panels=to_panels(
            args.v2p11_first_pass_panel
        ),
        provenance_path=args.provenance,
    )
    if args.verify_existing:
        if (
            _read_object(args.out) != report
            or args.markdown_out.read_text(encoding="utf-8")
            != _markdown(report)
        ):
            raise ValueError(
                "existing paired verdict differs from current inputs"
            )
    else:
        publish_verdict_outputs(
            report,
            output_path=args.out,
            markdown_output_path=args.markdown_out,
        )
    print(json.dumps(report["verdict"], sort_keys=True))
    if (
        args.require_trustworthy_win
        and report["verdict"].get("trustworthy_beats_v2p10") is not True
    ):
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
