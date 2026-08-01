#!/usr/bin/env python3
"""Validate the complete v2.11 post-training lineage before GPU evaluation."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from phaseE_rl.kto_swe_outcome import select_behavior_coverage_rows
from phaseH_eval.empty_retry_composite import _binding, _read_object
from phaseH_eval.v2p11_completion_provenance import (
    _model_contract,
    _phase_artifact_path,
    _phase_artifact_sha256,
    _phase_artifacts,
    _read_current_phase_marker,
    _require_fields,
    _validate_phase_marker,
)
from phaseH_eval.v2p11r2_completion_provenance import _MERGE_AUDIT


_R3_COVERAGE_COUNTS = {
    "desirable_correct_patch": 25,
    "empty_terminal": 8,
    "repeated_read_loop": 8,
    "wrong_nonempty_replay": 9,
}
_R3_NEGATIVE_COUNTS = {
    "empty_terminal": 33,
    "repeated_read_loop": 55,
    "wrong_nonempty_replay": 228,
}
_R3_RECOVERY_RUN_CONTRACT = {
    "data_len": 138,
    "rank": 32,
    "alpha": 32,
    "lr": 5e-6,
    "epochs": 3.0,
    "bsz": 1,
    "grad_accum": 4,
    "max_seq": 32768,
    "warmup_steps": 4,
    "load_4bit": False,
    "gradient_checkpointing": "bounded_unsloth",
    "selective_assistant_loss": True,
}


def _resolve_run_path(value: object) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError("poststage run path is incomplete")
    path = Path(value)
    return (path if path.is_absolute() else Path.cwd() / path).resolve()


def _phase_marker_artifact(
    report: Mapping[str, Any],
    *,
    expected_phase: str,
) -> Path:
    matches = []
    for path in _phase_artifacts(report).values():
        if path.suffix != ".json":
            continue
        try:
            value = _read_object(path)
        except (OSError, ValueError):
            continue
        if (
            value.get("artifact_type") == "v2p11_poststage_phase"
            and value.get("phase") == expected_phase
            and value.get("status") == "complete"
        ):
            matches.append(path.resolve())
    if len(matches) != 1:
        raise ValueError(
            f"poststage lineage requires one {expected_phase} marker"
        )
    return matches[0]


def _json_artifact(
    report: Mapping[str, Any],
    *,
    artifact_type: str,
) -> tuple[Path, dict[str, Any]]:
    matches = []
    for path in _phase_artifacts(report).values():
        if path.suffix != ".json":
            continue
        try:
            value = _read_object(path)
        except (OSError, ValueError):
            continue
        if value.get("artifact_type") == artifact_type:
            matches.append((path.resolve(), value))
    if len(matches) != 1:
        raise ValueError(
            f"poststage lineage requires one {artifact_type} artifact"
        )
    return matches[0]


def _validate_r3_behavior_posttrain_lineage(
    *,
    posttrain_marker_path: Path,
    recovery_marker_path: Path,
    kto_marker_path: Path,
    final_merge_marker_path: Path,
    final_model_path: Path,
    final_audit_path: Path | None = None,
) -> dict[str, Any]:
    posttrain_marker_path = Path(posttrain_marker_path).resolve()
    recovery_marker_path = Path(recovery_marker_path).resolve()
    kto_marker_path = Path(kto_marker_path).resolve()
    final_merge_marker_path = Path(final_merge_marker_path).resolve()
    final_model_path = Path(final_model_path).resolve()

    recovery = _read_current_phase_marker(
        recovery_marker_path,
        expected_phase="recovery_sft",
    )
    recovery_input_path = _phase_marker_artifact(
        recovery,
        expected_phase="recovery_sft_inputs",
    )
    recovery_input = _read_current_phase_marker(
        recovery_input_path,
        expected_phase="recovery_sft_inputs",
    )
    stage_marker_path, stage_marker = _json_artifact(
        recovery_input,
        artifact_type="v2p11r3_training_completion",
    )
    stage_adapter = stage_marker.get("adapter")
    recovery_inputs = _phase_artifacts(recovery_input)
    if (
        stage_marker.get("schema_version") != 1
        or stage_marker.get("status") != "complete"
        or stage_marker.get("optimizer_steps") != 79
        or stage_marker.get("max_steps") != 79
        or not isinstance(stage_adapter, Mapping)
        or dict(stage_adapter)
        != _binding(Path(str(stage_adapter.get("path", ""))))
        or Path(str(stage_adapter["path"])).resolve()
        not in {path.resolve() for path in recovery_inputs.values()}
    ):
        raise ValueError("r3 recovery inputs do not bind the completed adapter")
    stage_adapter_path = Path(str(stage_adapter["path"])).resolve()

    recovery_artifacts = _phase_artifacts(recovery)
    recovery_state = _read_object(recovery_artifacts["trainer_state.json"])
    recovery_manifest = _read_object(recovery_artifacts["run_manifest.json"])
    recovery_config = _read_object(recovery_artifacts["adapter_config.json"])
    recovery_weights = recovery_artifacts["adapter_model.safetensors"]
    recovery_data_manifest = recovery_inputs["manifest.json"]
    if (
        _phase_marker_artifact(recovery, expected_phase="recovery_sft_inputs")
        != recovery_input_path
        or recovery_state.get("global_step") != 105
        or recovery_state.get("max_steps") != 105
        or _resolve_run_path(recovery_manifest.get("data"))
        != recovery_data_manifest.parent.resolve()
        or _resolve_run_path(recovery_manifest.get("init_adapter"))
        != stage_adapter_path.parent
        or any(
            recovery_manifest.get(key) != value
            for key, value in _R3_RECOVERY_RUN_CONTRACT.items()
        )
        or recovery_config.get("r") != 32
        or recovery_config.get("lora_alpha") != 32
    ):
        raise ValueError("r3 recovery SFT phase evidence is incomplete")

    kto = _read_current_phase_marker(
        kto_marker_path,
        expected_phase="kto_full",
    )
    kto_input_path = _phase_marker_artifact(
        kto,
        expected_phase="kto_full_inputs",
    )
    kto_input = _read_current_phase_marker(
        kto_input_path,
        expected_phase="kto_full_inputs",
    )
    recovery_merge_path = _phase_marker_artifact(
        kto_input,
        expected_phase="recovery_merge",
    )
    recovery_merge = _read_current_phase_marker(
        recovery_merge_path,
        expected_phase="recovery_merge",
    )
    if (
        _phase_marker_artifact(recovery_merge, expected_phase="recovery_sft")
        != recovery_marker_path
    ):
        raise ValueError("r3 KTO lineage binds a different recovery adapter")
    recovery_merge_artifacts = _phase_artifacts(recovery_merge)
    recovery_model_path = recovery_merge_artifacts["config.json"].parent.resolve()
    recovery_merge_audit = recovery_merge_artifacts.get(
        "v2p11_recovery_merge_audit.json"
    )
    if (
        recovery_merge_audit is None
        or recovery_merge_audit.parent.resolve() != recovery_model_path
        or _read_object(recovery_merge_audit) != _MERGE_AUDIT
    ):
        raise ValueError("r3 recovery merge audit is incomplete")
    _model_contract(recovery_model_path)

    kto_artifacts = _phase_artifacts(kto)
    kto_state = _read_object(kto_artifacts["trainer_state.json"])
    kto_config = _read_object(kto_artifacts["adapter_config.json"])
    kto_evidence_path = kto_artifacts["training_evidence.json"]
    kto_evidence = _read_object(kto_evidence_path)
    kto_inputs = _phase_artifacts(kto_input)
    behavior_data = kto_inputs["v2p11_behavior_kto_v2.jsonl"]
    behavior_manifest = kto_inputs["v2p11_behavior_kto_v2_manifest.json"]
    behavior_rows = [
        json.loads(line)
        for line in behavior_data.read_text(encoding="utf-8").splitlines()
        if line
    ]
    selected_rows, selected_counts = select_behavior_coverage_rows(
        behavior_rows
    )
    selected_uids = [str(row["sample_uid"]) for row in selected_rows]
    if kto_evidence.get("selected_sample_uids") != selected_uids:
        raise ValueError("r3 behavior KTO training selection is inconsistent")
    if (
        kto_state.get("global_step") != 25
        or kto_state.get("max_steps") != 25
        or kto_config.get("r") != 32
        or kto_config.get("lora_alpha") != 32
        or kto_evidence.get("schema_version") != 1
        or kto_evidence.get("artifact_type")
        != "v2p11_kto_training_evidence"
        or kto_evidence.get("status") != "complete"
        or kto_evidence.get("base_model_path")
        != str(recovery_model_path)
        or kto_evidence.get("source_data_path")
        != str(behavior_data.resolve())
        or kto_evidence.get("source_data_sha256")
        != _binding(behavior_data)["sha256"]
        or kto_evidence.get("source_manifest_path")
        != str(behavior_manifest.resolve())
        or kto_evidence.get("source_manifest_sha256")
        != _binding(behavior_manifest)["sha256"]
        or kto_evidence.get("source_rows") != 606
        or kto_evidence.get("training_rows") != 50
        or kto_evidence.get("coverage_required") is not True
        or kto_evidence.get("coverage_counts") != selected_counts
        or selected_counts != _R3_COVERAGE_COUNTS
        or kto_evidence.get("per_device_train_batch_size") != 2
        or kto_evidence.get("gradient_accumulation_steps") != 1
        or kto_evidence.get("optimizer_steps") != 25
        or kto_evidence.get("seed") != 0
    ):
        raise ValueError("r3 behavior KTO coverage evidence is incomplete")

    final_merge = _read_current_phase_marker(
        final_merge_marker_path,
        expected_phase="final_merge",
    )
    if (
        _phase_marker_artifact(final_merge, expected_phase="recovery_merge")
        != recovery_merge_path
        or _phase_marker_artifact(final_merge, expected_phase="kto_full")
        != kto_marker_path
    ):
        raise ValueError("r3 final merge lineage binds different poststage inputs")
    final_artifacts = _phase_artifacts(final_merge)
    final_marker_model = final_artifacts["config.json"].parent.resolve()
    if final_marker_model != final_model_path:
        raise ValueError("r3 final merge lineage binds a different model")
    final_audit = final_artifacts["v2p11_final_merge_audit.json"].resolve()
    if (
        final_audit_path is not None
        and Path(final_audit_path).resolve() != final_audit
    ):
        raise ValueError("r3 final merge lineage binds a different audit")
    model = _model_contract(final_model_path)

    posttrain_marker = _read_object(posttrain_marker_path)
    expected_marker = {
        "schema_version": 1,
        "complete": True,
        "stage_a_adapter_sha256": _binding(stage_adapter_path)["sha256"],
        "recovery_adapter_sha256": _binding(recovery_weights)["sha256"],
        "kto_adapter_sha256": _binding(
            kto_artifacts["adapter_model.safetensors"]
        )["sha256"],
        "recovery_manifest_sha256": _binding(recovery_data_manifest)[
            "sha256"
        ],
        "behavior_manifest_sha256": _binding(behavior_manifest)["sha256"],
        "final_merge_audit_sha256": _binding(final_audit)["sha256"],
        "kto_training_evidence_sha256": _binding(kto_evidence_path)[
            "sha256"
        ],
        "final_merge_marker_sha256": _binding(final_merge_marker_path)[
            "sha256"
        ],
        "recovery_input_marker_sha256": _binding(recovery_input_path)[
            "sha256"
        ],
        "kto_input_marker_sha256": _binding(kto_input_path)["sha256"],
        "recovery_optimizer_steps": 105,
        "kto_optimizer_steps": 25,
    }
    if any(
        posttrain_marker.get(key) != value
        for key, value in expected_marker.items()
    ):
        raise ValueError("r3 behavior posttrain marker is inconsistent")
    gpu_uuid = posttrain_marker.get("gpu_uuid")
    if not isinstance(gpu_uuid, str) or not gpu_uuid.startswith("GPU-"):
        raise ValueError("r3 behavior posttrain GPU identity is incomplete")
    return {
        "schema_version": 1,
        "artifact_type": "v2p11r3_behavior_posttrain_lineage_validation",
        "status": "complete",
        "posttrain_marker": _binding(posttrain_marker_path),
        "stage_marker": _binding(stage_marker_path),
        "stage_adapter": _binding(stage_adapter_path),
        "phase_markers": {
            "recovery": _binding(recovery_marker_path),
            "recovery_merge": _binding(recovery_merge_path),
            "kto": _binding(kto_marker_path),
            "final_merge": _binding(final_merge_marker_path),
        },
        "inputs": {
            "recovery_train": _binding(recovery_inputs["train.jsonl"]),
            "recovery_manifest": _binding(recovery_data_manifest),
            "behavior_data": _binding(behavior_data),
            "behavior_manifest": _binding(behavior_manifest),
        },
        "kto_training_evidence": _binding(kto_evidence_path),
        "coverage_counts": dict(_R3_COVERAGE_COUNTS),
        "final_model": model,
    }


def _is_r3_behavior_lineage(recovery_marker_path: Path) -> bool:
    recovery = _read_current_phase_marker(
        Path(recovery_marker_path).resolve(),
        expected_phase="recovery_sft",
    )
    try:
        recovery_input = _read_current_phase_marker(
            _phase_marker_artifact(
                recovery,
                expected_phase="recovery_sft_inputs",
            ),
            expected_phase="recovery_sft_inputs",
        )
        _json_artifact(
            recovery_input,
            artifact_type="v2p11r3_training_completion",
        )
    except ValueError:
        return False
    return True


def validate_posttrain_lineage(
    *,
    posttrain_marker_path: Path,
    recovery_marker_path: Path,
    kto_marker_path: Path,
    final_merge_marker_path: Path,
    final_model_path: Path,
    final_audit_path: Path | None = None,
) -> dict[str, Any]:
    if _is_r3_behavior_lineage(recovery_marker_path):
        return _validate_r3_behavior_posttrain_lineage(
            posttrain_marker_path=posttrain_marker_path,
            recovery_marker_path=recovery_marker_path,
            kto_marker_path=kto_marker_path,
            final_merge_marker_path=final_merge_marker_path,
            final_model_path=final_model_path,
            final_audit_path=final_audit_path,
        )
    posttrain_marker_path = Path(posttrain_marker_path).resolve()
    recovery_marker_path = Path(recovery_marker_path).resolve()
    kto_marker_path = Path(kto_marker_path).resolve()
    final_merge_marker_path = Path(final_merge_marker_path).resolve()
    final_model_path = Path(final_model_path).resolve()
    recovery = _validate_phase_marker(
        recovery_marker_path,
        expected_phase="recovery_sft",
    )
    kto = _validate_phase_marker(
        kto_marker_path,
        expected_phase="kto_full",
    )
    final_merge = _validate_phase_marker(
        final_merge_marker_path,
        expected_phase="final_merge",
    )
    recovery_input_path = _phase_artifact_path(
        recovery,
        "v2p11_recovery_sft_inputs.json",
    )
    recovery_input = _validate_phase_marker(
        recovery_input_path,
        expected_phase="recovery_sft_inputs",
    )
    kto_input_path = _phase_artifact_path(kto, "v2p11_kto_inputs.json")
    kto_input = _validate_phase_marker(
        kto_input_path,
        expected_phase="kto_full_inputs",
    )
    recovery_merge_path = _phase_artifact_path(
        kto_input,
        "v2p11_recovery_merge_complete.json",
    )
    recovery_merge = _validate_phase_marker(
        recovery_merge_path,
        expected_phase="recovery_merge",
    )
    if (
        _phase_artifact_path(
            recovery_merge,
            "v2p11_recovery_sft_complete.json",
        )
        != recovery_marker_path
    ):
        raise ValueError(
            "KTO lineage does not bind the requested recovery marker"
        )
    final_artifacts = _phase_artifacts(final_merge)
    if (
        _phase_artifact_path(
            final_merge,
            "v2p11_recovery_merge_complete.json",
        )
        != recovery_merge_path
        or _phase_artifact_path(
            final_merge,
            "v2p11_kto_complete.json",
        )
        != kto_marker_path
    ):
        raise ValueError(
            "final merge lineage does not bind the requested poststage markers"
        )
    final_audit = _phase_artifact_path(
        final_merge,
        "v2p11_final_merge_audit.json",
    )
    if (
        final_audit_path is not None
        and Path(final_audit_path).resolve() != final_audit
    ):
        raise ValueError("final merge lineage binds a different audit")
    final_merge_model = final_artifacts["config.json"].parent.resolve()
    if final_merge_model != final_model_path:
        raise ValueError("final merge lineage binds a different model")
    stage_marker_path = _phase_artifact_path(
        recovery_input,
        "v2p11_training_complete.json",
    )
    stage_marker = _read_object(stage_marker_path)
    recovery_manifest = _phase_artifact_path(
        recovery_input,
        "manifest.json",
    )
    behavior_manifest = _phase_artifact_path(
        kto_input,
        "v2p11_behavior_kto_v2_manifest.json",
    )
    posttrain_marker = _read_object(posttrain_marker_path)
    _require_fields(
        posttrain_marker,
        {
            "schema_version": 1,
            "complete": True,
            "stage_a_adapter_sha256": stage_marker.get("adapter_sha256"),
            "recovery_adapter_sha256": _phase_artifact_sha256(
                recovery,
                "adapter_model.safetensors",
            ),
            "kto_adapter_sha256": _phase_artifact_sha256(
                kto,
                "adapter_model.safetensors",
            ),
            "recovery_manifest_sha256": _binding(recovery_manifest)[
                "sha256"
            ],
            "behavior_manifest_sha256": _binding(behavior_manifest)[
                "sha256"
            ],
            "final_merge_audit_sha256": _binding(final_audit)["sha256"],
            "kto_training_evidence_sha256": _phase_artifact_sha256(
                kto,
                "training_evidence.json",
            ),
            "final_merge_marker_sha256": _binding(
                final_merge_marker_path
            )["sha256"],
            "recovery_input_marker_sha256": _binding(
                recovery_input_path
            )["sha256"],
            "kto_input_marker_sha256": _binding(kto_input_path)["sha256"],
            "recovery_optimizer_steps": 105,
            "kto_optimizer_steps": 25,
        },
        label="posttrain marker",
    )
    gpu_uuid = posttrain_marker.get("gpu_uuid")
    if not isinstance(gpu_uuid, str) or not gpu_uuid.startswith("GPU-"):
        raise ValueError("posttrain marker GPU identity is incomplete")
    model = _model_contract(final_model_path)
    return {
        "schema_version": 1,
        "artifact_type": "v2p11_posttrain_lineage_validation",
        "status": "complete",
        "posttrain_marker_sha256": _binding(posttrain_marker_path)[
            "sha256"
        ],
        "phase_markers": {
            "recovery": _binding(recovery_marker_path),
            "kto": _binding(kto_marker_path),
            "final_merge": _binding(final_merge_marker_path),
        },
        "final_model": model,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--posttrain-marker", type=Path, required=True)
    parser.add_argument("--recovery-marker", type=Path, required=True)
    parser.add_argument("--kto-marker", type=Path, required=True)
    parser.add_argument("--final-merge-marker", type=Path, required=True)
    parser.add_argument("--final-model", type=Path, required=True)
    parser.add_argument("--final-audit", type=Path, required=True)
    args = parser.parse_args(argv)
    report = validate_posttrain_lineage(
        posttrain_marker_path=args.posttrain_marker,
        recovery_marker_path=args.recovery_marker,
        kto_marker_path=args.kto_marker,
        final_merge_marker_path=args.final_merge_marker,
        final_model_path=args.final_model,
        final_audit_path=args.final_audit,
    )
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
