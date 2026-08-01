#!/usr/bin/env python3
"""Validate the complete v2.11 post-training lineage before GPU evaluation."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from phaseH_eval.empty_retry_composite import _binding, _read_object
from phaseH_eval.v2p11_completion_provenance import (
    _model_contract,
    _phase_artifact_path,
    _phase_artifact_sha256,
    _phase_artifacts,
    _require_fields,
    _validate_phase_marker,
)


def validate_posttrain_lineage(
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
