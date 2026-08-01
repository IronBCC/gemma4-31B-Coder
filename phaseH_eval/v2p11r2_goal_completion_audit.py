#!/usr/bin/env python3
"""Publish a requirement-level audit for the direct-LoRA v2.11r2 run."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from phaseH_eval.compare_v2p11_full300 import (
    FirstPassPanel,
    compare_full300,
)
from phaseH_eval.empty_retry_composite import (
    _binding,
    _publish_json_noreplace,
    _read_ids,
    _read_object,
)
from phaseH_eval.v2p11_stage_data_contract import validate_stage_data


_CANONICAL_LITE300_SHA256 = (
    "b98fc2b1054dc8fdfcb94f083f43454fd568961a0b3dbf8c388c210b7b868e14"
)


def _require_current_bindings(
    bindings: Mapping[str, Mapping[str, Any]],
) -> None:
    for label, binding in bindings.items():
        path_value = binding.get("path")
        if not isinstance(path_value, str) or not path_value:
            raise ValueError(f"{label} binding is incomplete")
        if _binding(Path(path_value)) != dict(binding):
            raise ValueError(f"{label} binding changed")


def _validate_trustworthy_verdict(
    verdict: Mapping[str, Any],
    *,
    candidate_name: str,
    full_ids: Sequence[str] | None = None,
) -> None:
    control = verdict.get("v2p10")
    candidate = verdict.get("v2p11")
    decision = verdict.get("verdict")
    corrected_empty = verdict.get("empty_patch")
    first_pass = verdict.get("first_pass")
    first_pass_empty = (
        first_pass.get("empty_patch")
        if isinstance(first_pass, Mapping)
        else None
    )
    failure_analysis = verdict.get("failure_analysis")
    if (
        verdict.get("schema_version") != 1
        or verdict.get("artifact_type")
        != "v2p11_v2p10_full300_verdict"
        or verdict.get("status") != "complete"
        or verdict.get("population") != 300
        or not isinstance(control, Mapping)
        or control.get("name") != "teacher_sft_v2p10"
        or not isinstance(candidate, Mapping)
        or candidate.get("name") != candidate_name
        or not isinstance(decision, Mapping)
        or not isinstance(corrected_empty, Mapping)
        or not isinstance(first_pass_empty, Mapping)
        or not isinstance(failure_analysis, Mapping)
        or not isinstance(failure_analysis.get("by_instance"), Mapping)
        or type(control.get("resolved")) is not int
        or type(candidate.get("resolved")) is not int
        or type(control.get("empty")) is not int
        or type(candidate.get("empty")) is not int
        or candidate["resolved"] <= control["resolved"]
        or candidate["empty"] > control["empty"]
        or decision.get("beats_v2p10") is not True
        or decision.get("first_pass_beats_v2p10") is not True
        or decision.get("first_pass_empty_delta", 1) > 0
        or decision.get("corrected_empty_delta", 1) > 0
        or decision.get("first_pass_no_new_empty_ids") is not True
        or decision.get("corrected_no_new_empty_ids") is not True
        or corrected_empty.get("introduced") != []
        or first_pass_empty.get("introduced") != []
        or decision.get("behavior_healthy") is not True
        or decision.get("trustworthy_beats_v2p10") is not True
        or decision.get("resolved_delta")
        != candidate["resolved"] - control["resolved"]
    ):
        raise ValueError("final verdict is not a trustworthy win")
    if full_ids is None:
        return
    full_set = set(full_ids)
    paired = verdict.get("paired")
    required_failure_keys = {
        "v2p10_nonempty_unresolved",
        "v2p11_nonempty_unresolved",
        "v2p10_model_failure_ids",
        "v2p11_model_failure_ids",
        "v2p10_repeat_loop_ids",
        "v2p11_repeat_loop_ids",
        "v2p10_tool_format_error_ids",
        "v2p11_tool_format_error_ids",
    }
    if not isinstance(paired, Mapping):
        raise ValueError("failure analysis paired evidence is incomplete")
    paired_sets: dict[str, set[str]] = {}
    for key in (
        "v2p11_only",
        "v2p10_only",
        "both_resolved",
        "neither_resolved",
    ):
        values = paired.get(key)
        if (
            not isinstance(values, list)
            or len(values) != len(set(values))
            or any(not isinstance(value, str) for value in values)
        ):
            raise ValueError("failure analysis paired evidence is invalid")
        paired_sets[key] = set(values)
    union = set().union(*paired_sets.values())
    if (
        union != full_set
        or sum(len(values) for values in paired_sets.values()) != len(full_set)
        or len(paired_sets["v2p10_only"])
        + len(paired_sets["both_resolved"])
        != control["resolved"]
        or len(paired_sets["v2p11_only"])
        + len(paired_sets["both_resolved"])
        != candidate["resolved"]
        or not required_failure_keys.issubset(failure_analysis)
        or any(
            not isinstance(failure_analysis[key], list)
            or any(
                not isinstance(value, str) or value not in full_set
                for value in failure_analysis[key]
            )
            for key in required_failure_keys
        )
    ):
        raise ValueError("failure analysis is incomplete")
    by_instance = failure_analysis["by_instance"]
    assert isinstance(by_instance, Mapping)
    expected_ids = full_set - paired_sets["both_resolved"]
    if set(by_instance) != expected_ids:
        raise ValueError("failure analysis omits unresolved instances")
    for instance_id, reasons in by_instance.items():
        if (
            not isinstance(instance_id, str)
            or not isinstance(reasons, Mapping)
            or not reasons
            or any(
                label not in {"v2p10", "v2p11"}
                or not isinstance(values, list)
                or not values
                or any(not isinstance(value, str) for value in values)
                for label, values in reasons.items()
            )
        ):
            raise ValueError("failure analysis instance evidence is invalid")


def _build_report(
    *,
    full_ids: Mapping[str, Any],
    verdict: Mapping[str, Any],
    v2p10_score: Mapping[str, Any],
    v2p11_score: Mapping[str, Any],
    provenance: Mapping[str, Any],
    score: Mapping[str, Mapping[str, int]],
    stage_a: Mapping[str, int],
    direct_lora: Mapping[str, int],
) -> dict[str, Any]:
    control = score["v2p10"]
    candidate = score["v2p11"]
    return {
        "schema_version": 1,
        "artifact_type": "v2p11r2_goal_completion_audit",
        "status": "complete",
        "scope": ["v2p10", "v2p11"],
        "requirements": {
            "same_complete_lite300": True,
            "v2p11_beats_v2p10": True,
            "empty_patch_no_regression": True,
            "behavior_healthy": True,
            "stage_a_fable_rows": stage_a["new_fable_rows"],
            "recent_fable_strict_rows": direct_lora["strict_rows"],
            "full_context_window": direct_lora["max_seq"],
            "lora_training_and_merge_revalidated": True,
            "official_scores_bound": True,
            "failure_analysis_bound": True,
        },
        "score": {
            "v2p10": dict(control),
            "v2p11": dict(candidate),
            "resolved_delta": candidate["resolved"] - control["resolved"],
        },
        "training": {
            "stage_a": dict(stage_a),
            "direct_lora": dict(direct_lora),
        },
        "artifacts": {
            "full_ids": dict(full_ids),
            "verdict": dict(verdict),
            "v2p10_official_score": dict(v2p10_score),
            "v2p11_official_score": dict(v2p11_score),
            "v2p11r2_completion_provenance": dict(provenance),
        },
    }


def _publish_revalidated(
    *,
    verdict: Mapping[str, Any],
    full_ids: Mapping[str, Any],
    verdict_binding: Mapping[str, Any],
    v2p10_score: Mapping[str, Any],
    v2p11_score: Mapping[str, Any],
    provenance: Mapping[str, Any],
    stage_a: Mapping[str, int],
    direct_lora: Mapping[str, int],
    candidate_name: str,
    output_path: Path,
) -> dict[str, Any]:
    _validate_trustworthy_verdict(
        verdict,
        candidate_name=candidate_name,
    )
    if (
        dict(stage_a)
        != {"rows": 1247, "base_rows": 1211, "new_fable_rows": 36}
        or direct_lora.get("rows") != 1262
        or direct_lora.get("base_rows") != 1247
        or direct_lora.get("new_fable_rows") != 15
        or direct_lora.get("strict_rows") != 15
        or direct_lora.get("max_seq") != 32768
        or type(direct_lora.get("max_rendered_tokens")) is not int
        or not 1 <= direct_lora["max_rendered_tokens"] <= 32768
    ):
        raise ValueError("v2.11r2 training provenance is incomplete")
    control = verdict["v2p10"]
    candidate = verdict["v2p11"]
    assert isinstance(control, Mapping)
    assert isinstance(candidate, Mapping)
    report = _build_report(
        full_ids=full_ids,
        verdict=verdict_binding,
        v2p10_score=v2p10_score,
        v2p11_score=v2p11_score,
        provenance=provenance,
        score={
            "v2p10": {
                "resolved": control["resolved"],
                "empty": control["empty"],
            },
            "v2p11": {
                "resolved": candidate["resolved"],
                "empty": candidate["empty"],
            },
        },
        stage_a=stage_a,
        direct_lora=direct_lora,
    )
    output_path = Path(output_path).resolve()
    if output_path.exists():
        if _read_object(output_path) != report:
            raise ValueError("existing goal completion audit differs")
    else:
        _publish_json_noreplace(output_path, report)
    return report


def publish_goal_completion_audit(
    *,
    verdict_path: Path,
    full_ids_path: Path,
    v2p10_composite_path: Path,
    v2p10_predictions_path: Path,
    v2p10_score_path: Path,
    v2p11_composite_path: Path,
    v2p11_predictions_path: Path,
    v2p11_score_path: Path,
    candidate_name: str,
    candidate_model_path: Path,
    v2p10_fixed_ids_path: Path,
    v2p10_fixed_run: Path,
    v2p10_complement_ids_path: Path,
    v2p10_complement_run: Path,
    v2p11_fixed_ids_path: Path,
    v2p11_fixed_run: Path,
    v2p11_complement_ids_path: Path,
    v2p11_complement_run: Path,
    provenance_path: Path,
    stage_data_path: Path,
    stage_training_contract_path: Path,
    exclusions_path: Path,
    campaign_root: Path,
    output_path: Path,
) -> dict[str, Any]:
    full_ids_path = Path(full_ids_path).resolve()
    verdict_path = Path(verdict_path).resolve()
    v2p10_composite_path = Path(v2p10_composite_path).resolve()
    v2p10_predictions_path = Path(v2p10_predictions_path).resolve()
    v2p10_score_path = Path(v2p10_score_path).resolve()
    v2p11_composite_path = Path(v2p11_composite_path).resolve()
    v2p11_predictions_path = Path(v2p11_predictions_path).resolve()
    v2p11_score_path = Path(v2p11_score_path).resolve()
    provenance_path = Path(provenance_path).resolve()
    stage_data_path = Path(stage_data_path).resolve()
    stage_training_contract_path = Path(
        stage_training_contract_path
    ).resolve()
    exclusions_path = Path(exclusions_path).resolve()
    campaign_root = Path(campaign_root).resolve()
    evidence_bindings = {
        "full_ids": _binding(full_ids_path),
        "verdict": _binding(verdict_path),
        "v2p10_composite": _binding(v2p10_composite_path),
        "v2p10_predictions": _binding(v2p10_predictions_path),
        "v2p10_score": _binding(v2p10_score_path),
        "v2p11_composite": _binding(v2p11_composite_path),
        "v2p11_predictions": _binding(v2p11_predictions_path),
        "v2p11_score": _binding(v2p11_score_path),
        "provenance": _binding(provenance_path),
        "stage_manifest": _binding(stage_data_path / "manifest.json"),
        "stage_train": _binding(stage_data_path / "train.jsonl"),
        "stage_training_contract": _binding(
            stage_training_contract_path
        ),
        "exclusions": _binding(exclusions_path),
        "campaign_manifest": _binding(campaign_root / "manifest.json"),
        "campaign_results": _binding(campaign_root / "results.jsonl"),
        "campaign_resolved": _binding(campaign_root / "resolved.jsonl"),
        "campaign_rejected": _binding(campaign_root / "rejected.jsonl"),
    }
    full_ids = _read_ids(full_ids_path)
    full_ids_binding = evidence_bindings["full_ids"]
    if (
        len(full_ids) != 300
        or len(full_ids) != len(set(full_ids))
        or full_ids_binding["sha256"] != _CANONICAL_LITE300_SHA256
    ):
        raise ValueError("goal audit requires canonical Lite300 IDs")
    published = _read_object(verdict_path)
    if published.get("full_ids") != full_ids_binding:
        raise ValueError("published verdict binds different Lite300 IDs")
    recomputed = compare_full300(
        full_ids_path=full_ids_path,
        v2p10_composite_path=v2p10_composite_path,
        v2p10_predictions_path=v2p10_predictions_path,
        v2p10_score_binding_path=v2p10_score_path,
        v2p11_composite_path=v2p11_composite_path,
        v2p11_predictions_path=v2p11_predictions_path,
        v2p11_score_binding_path=v2p11_score_path,
        candidate_name=candidate_name,
        candidate_model_path=Path(candidate_model_path),
        v2p10_first_pass_panels=[
            FirstPassPanel(
                "fixed150",
                Path(v2p10_fixed_ids_path),
                Path(v2p10_fixed_run),
            ),
            FirstPassPanel(
                "complement150",
                Path(v2p10_complement_ids_path),
                Path(v2p10_complement_run),
            ),
        ],
        v2p11_first_pass_panels=[
            FirstPassPanel(
                "fixed150",
                Path(v2p11_fixed_ids_path),
                Path(v2p11_fixed_run),
            ),
            FirstPassPanel(
                "complement150",
                Path(v2p11_complement_ids_path),
                Path(v2p11_complement_run),
            ),
        ],
        provenance_path=provenance_path,
    )
    if published != recomputed:
        raise ValueError("published verdict differs from revalidated evidence")
    stage = validate_stage_data(
        stage_data=stage_data_path,
        training_contract_path=stage_training_contract_path,
        exclusions_path=exclusions_path,
        campaign_root=campaign_root,
        require_production_identity=True,
    )
    stage_summary = {
        key: stage.get(key)
        for key in ("rows", "base_rows", "new_fable_rows")
    }
    if stage_summary != {
        "rows": 1247,
        "base_rows": 1211,
        "new_fable_rows": 36,
    }:
        raise ValueError("Stage-A Fable evidence is incomplete")
    _validate_trustworthy_verdict(
        recomputed,
        candidate_name=candidate_name,
        full_ids=full_ids,
    )
    direct = recomputed.get("provenance")
    training = direct.get("training") if isinstance(direct, Mapping) else None
    dataset = direct.get("dataset") if isinstance(direct, Mapping) else None
    recent = direct.get("recent_fable") if isinstance(direct, Mapping) else None
    if (
        not isinstance(training, Mapping)
        or not isinstance(dataset, Mapping)
        or not isinstance(recent, Mapping)
    ):
        raise ValueError("direct-LoRA provenance is incomplete")
    direct_summary = {
        "rows": training.get("rows"),
        "base_rows": training.get("base_rows"),
        "new_fable_rows": training.get("new_fable_rows"),
        "max_seq": training.get("max_seq"),
        "max_rendered_tokens": dataset.get("max_rendered_tokens"),
        "strict_rows": recent.get("strict_rows"),
    }
    _require_current_bindings(evidence_bindings)
    return _publish_revalidated(
        verdict=recomputed,
        full_ids=full_ids_binding,
        verdict_binding=evidence_bindings["verdict"],
        v2p10_score=evidence_bindings["v2p10_score"],
        v2p11_score=evidence_bindings["v2p11_score"],
        provenance=evidence_bindings["provenance"],
        stage_a=stage_summary,
        direct_lora=direct_summary,
        candidate_name=candidate_name,
        output_path=output_path,
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verdict", type=Path, required=True)
    parser.add_argument("--full-ids", type=Path, required=True)
    parser.add_argument("--v2p10", type=Path, required=True)
    parser.add_argument("--v2p10-preds", type=Path, required=True)
    parser.add_argument("--v2p10-score", type=Path, required=True)
    parser.add_argument("--v2p11", type=Path, required=True)
    parser.add_argument("--v2p11-preds", type=Path, required=True)
    parser.add_argument("--v2p11-score", type=Path, required=True)
    parser.add_argument("--candidate-name", required=True)
    parser.add_argument("--candidate-model", type=Path, required=True)
    parser.add_argument("--v2p10-fixed-ids", type=Path, required=True)
    parser.add_argument("--v2p10-fixed-run", type=Path, required=True)
    parser.add_argument("--v2p10-complement-ids", type=Path, required=True)
    parser.add_argument("--v2p10-complement-run", type=Path, required=True)
    parser.add_argument("--v2p11-fixed-ids", type=Path, required=True)
    parser.add_argument("--v2p11-fixed-run", type=Path, required=True)
    parser.add_argument("--v2p11-complement-ids", type=Path, required=True)
    parser.add_argument("--v2p11-complement-run", type=Path, required=True)
    parser.add_argument("--provenance", type=Path, required=True)
    parser.add_argument("--stage-data", type=Path, required=True)
    parser.add_argument("--stage-training-contract", type=Path, required=True)
    parser.add_argument("--exclusions", type=Path, required=True)
    parser.add_argument("--campaign-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    report = publish_goal_completion_audit(
        verdict_path=args.verdict,
        full_ids_path=args.full_ids,
        v2p10_composite_path=args.v2p10,
        v2p10_predictions_path=args.v2p10_preds,
        v2p10_score_path=args.v2p10_score,
        v2p11_composite_path=args.v2p11,
        v2p11_predictions_path=args.v2p11_preds,
        v2p11_score_path=args.v2p11_score,
        candidate_name=args.candidate_name,
        candidate_model_path=args.candidate_model,
        v2p10_fixed_ids_path=args.v2p10_fixed_ids,
        v2p10_fixed_run=args.v2p10_fixed_run,
        v2p10_complement_ids_path=args.v2p10_complement_ids,
        v2p10_complement_run=args.v2p10_complement_run,
        v2p11_fixed_ids_path=args.v2p11_fixed_ids,
        v2p11_fixed_run=args.v2p11_fixed_run,
        v2p11_complement_ids_path=args.v2p11_complement_ids,
        v2p11_complement_run=args.v2p11_complement_run,
        provenance_path=args.provenance,
        stage_data_path=args.stage_data,
        stage_training_contract_path=args.stage_training_contract,
        exclusions_path=args.exclusions,
        campaign_root=args.campaign_root,
        output_path=args.out,
    )
    print(json.dumps({"status": report["status"], "score": report["score"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
