from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

import phaseH_eval.v2p11r2_goal_completion_audit as audit_module
from phaseH_eval.v2p11r2_goal_completion_audit import (
    _build_report,
    _require_current_bindings,
    _publish_revalidated,
    _validate_trustworthy_verdict,
    publish_goal_completion_audit,
)


def _binding(path: Path) -> dict[str, object]:
    return {
        "path": str(path.resolve()),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "bytes": path.stat().st_size,
    }


def test_build_report_exposes_direct_lora_and_both_fable_generations(
    tmp_path: Path,
) -> None:
    full_ids = tmp_path / "full_ids.json"
    full_ids.write_text(json.dumps(["case-a"]) + "\n")
    verdict = tmp_path / "verdict.json"
    verdict.write_text("{}\n")
    v2p10_score = tmp_path / "v2p10_score.json"
    v2p10_score.write_text("{}\n")
    v2p11_score = tmp_path / "v2p11_score.json"
    v2p11_score.write_text("{}\n")
    provenance = tmp_path / "provenance.json"
    provenance.write_text("{}\n")

    report = _build_report(
        full_ids=_binding(full_ids),
        verdict=_binding(verdict),
        v2p10_score=_binding(v2p10_score),
        v2p11_score=_binding(v2p11_score),
        provenance=_binding(provenance),
        score={"v2p10": {"resolved": 157, "empty": 2}, "v2p11": {"resolved": 160, "empty": 1}},
        stage_a={"rows": 1247, "base_rows": 1211, "new_fable_rows": 36},
        direct_lora={
            "rows": 1262,
            "base_rows": 1247,
            "new_fable_rows": 15,
            "max_seq": 32768,
            "max_rendered_tokens": 26594,
            "strict_rows": 15,
        },
    )

    assert report["status"] == "complete"
    assert report["requirements"] == {
        "same_complete_lite300": True,
        "v2p11_beats_v2p10": True,
        "empty_patch_no_regression": True,
        "behavior_healthy": True,
        "stage_a_fable_rows": 36,
        "recent_fable_strict_rows": 15,
        "full_context_window": 32768,
        "lora_training_and_merge_revalidated": True,
        "official_scores_bound": True,
        "failure_analysis_bound": True,
    }
    assert report["score"]["resolved_delta"] == 3


def test_rejects_verdict_without_a_first_pass_win() -> None:
    verdict = {
        "schema_version": 1,
        "artifact_type": "v2p11_v2p10_full300_verdict",
        "status": "complete",
        "population": 300,
        "v2p10": {"name": "teacher_sft_v2p10", "resolved": 157, "empty": 2},
        "v2p11": {"name": "teacher_sft_v2p11r2", "resolved": 160, "empty": 1},
        "empty_patch": {"introduced": []},
        "first_pass": {"empty_patch": {"introduced": []}},
        "failure_analysis": {"by_instance": {}},
        "verdict": {
            "beats_v2p10": True,
            "first_pass_beats_v2p10": False,
            "first_pass_empty_delta": -1,
            "corrected_empty_delta": -1,
            "first_pass_no_new_empty_ids": True,
            "corrected_no_new_empty_ids": True,
            "behavior_healthy": True,
            "trustworthy_beats_v2p10": True,
            "resolved_delta": 3,
        },
        "provenance": {
            "training": {
                "rows": 1262,
                "base_rows": 1247,
                "new_fable_rows": 15,
                "max_seq": 32768,
            },
            "recent_fable": {"strict_rows": 15},
            "dataset": {"max_rendered_tokens": 26594},
        },
    }

    try:
        _validate_trustworthy_verdict(
            verdict,
            candidate_name="teacher_sft_v2p11r2",
        )
    except ValueError as error:
        assert "trustworthy win" in str(error)
    else:
        raise AssertionError("a missing first-pass win was accepted")


def test_rejects_failure_analysis_missing_unresolved_instances() -> None:
    verdict = {
        "schema_version": 1,
        "artifact_type": "v2p11_v2p10_full300_verdict",
        "status": "complete",
        "population": 300,
        "v2p10": {"name": "teacher_sft_v2p10", "resolved": 1, "empty": 0},
        "v2p11": {"name": "teacher_sft_v2p11r2", "resolved": 2, "empty": 0},
        "paired": {
            "v2p11_only": ["case-b"],
            "v2p10_only": [],
            "both_resolved": ["case-a"],
            "neither_resolved": ["case-c"],
        },
        "empty_patch": {"introduced": []},
        "first_pass": {"empty_patch": {"introduced": []}},
        "failure_analysis": {
            "v2p10_nonempty_unresolved": [],
            "v2p11_nonempty_unresolved": [],
            "v2p10_model_failure_ids": [],
            "v2p11_model_failure_ids": [],
            "v2p10_repeat_loop_ids": [],
            "v2p11_repeat_loop_ids": [],
            "v2p10_tool_format_error_ids": [],
            "v2p11_tool_format_error_ids": [],
            "by_instance": {},
        },
        "verdict": {
            "beats_v2p10": True,
            "first_pass_beats_v2p10": True,
            "first_pass_empty_delta": 0,
            "corrected_empty_delta": 0,
            "first_pass_no_new_empty_ids": True,
            "corrected_no_new_empty_ids": True,
            "behavior_healthy": True,
            "trustworthy_beats_v2p10": True,
            "resolved_delta": 1,
        },
    }

    with pytest.raises(ValueError, match="failure analysis"):
        _validate_trustworthy_verdict(
            verdict,
            candidate_name="teacher_sft_v2p11r2",
            full_ids=["case-a", "case-b", "case-c"],
        )


def test_rejects_input_replaced_after_validation_snapshot(
    tmp_path: Path,
) -> None:
    evidence = tmp_path / "evidence.json"
    evidence.write_text('{"version": 1}\n')
    bindings = {"evidence": _binding(evidence)}
    evidence.write_text('{"version": 2}\n')

    with pytest.raises(ValueError, match="binding changed"):
        _require_current_bindings(bindings)


def test_publishes_only_the_revalidated_trustworthy_verdict(
    tmp_path: Path,
) -> None:
    full_ids = tmp_path / "full_ids.json"
    full_ids.write_text(json.dumps(["case-a"]) + "\n")
    verdict_path = tmp_path / "verdict.json"
    v2p10_score = tmp_path / "v2p10_score.json"
    v2p11_score = tmp_path / "v2p11_score.json"
    provenance = tmp_path / "provenance.json"
    for path in (v2p10_score, v2p11_score, provenance):
        path.write_text("{}\n")
    verdict = {
        "schema_version": 1,
        "artifact_type": "v2p11_v2p10_full300_verdict",
        "status": "complete",
        "population": 300,
        "v2p10": {"name": "teacher_sft_v2p10", "resolved": 157, "empty": 2},
        "v2p11": {"name": "teacher_sft_v2p11r2", "resolved": 160, "empty": 1},
        "empty_patch": {"introduced": []},
        "first_pass": {"empty_patch": {"introduced": []}},
        "failure_analysis": {"by_instance": {}},
        "verdict": {
            "beats_v2p10": True,
            "first_pass_beats_v2p10": True,
            "first_pass_empty_delta": -1,
            "corrected_empty_delta": -1,
            "first_pass_no_new_empty_ids": True,
            "corrected_no_new_empty_ids": True,
            "behavior_healthy": True,
            "trustworthy_beats_v2p10": True,
            "resolved_delta": 3,
        },
    }
    verdict_path.write_text(json.dumps(verdict) + "\n")
    output = tmp_path / "audit.json"

    report = _publish_revalidated(
        verdict=verdict,
        full_ids=_binding(full_ids),
        verdict_binding=_binding(verdict_path),
        v2p10_score=_binding(v2p10_score),
        v2p11_score=_binding(v2p11_score),
        provenance=_binding(provenance),
        stage_a={"rows": 1247, "base_rows": 1211, "new_fable_rows": 36},
        direct_lora={
            "rows": 1262,
            "base_rows": 1247,
            "new_fable_rows": 15,
            "max_seq": 32768,
            "max_rendered_tokens": 26594,
            "strict_rows": 15,
        },
        candidate_name="teacher_sft_v2p11r2",
        output_path=output,
    )

    assert report["score"]["resolved_delta"] == 3
    assert json.loads(output.read_text()) == report


def test_public_audit_rejects_stage_a_fable_count_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    full_ids = tmp_path / "full_ids.json"
    full_ids.write_text(
        json.dumps([f"case-{index:03d}" for index in range(300)]) + "\n"
    )
    verdict_path = tmp_path / "verdict.json"
    v2p10_score = tmp_path / "v2p10_score.json"
    v2p11_score = tmp_path / "v2p11_score.json"
    provenance = tmp_path / "provenance.json"
    for path in (v2p10_score, v2p11_score, provenance):
        path.write_text("{}\n")
    for name in (
        "v2p10.json",
        "v2p10_preds.json",
        "v2p11.json",
        "v2p11_preds.json",
        "stage_contract.json",
        "exclusions.json",
    ):
        (tmp_path / name).write_text("{}\n")
    stage_data = tmp_path / "stage_data"
    stage_data.mkdir()
    for name in ("manifest.json", "train.jsonl"):
        (stage_data / name).write_text("{}\n")
    campaign = tmp_path / "campaign"
    campaign.mkdir()
    for name in (
        "manifest.json",
        "results.jsonl",
        "resolved.jsonl",
        "rejected.jsonl",
    ):
        (campaign / name).write_text("{}\n")
    verdict = {
        "schema_version": 1,
        "artifact_type": "v2p11_v2p10_full300_verdict",
        "status": "complete",
        "population": 300,
        "full_ids": _binding(full_ids),
        "v2p10": {"name": "teacher_sft_v2p10", "resolved": 157, "empty": 2},
        "v2p11": {"name": "teacher_sft_v2p11r2", "resolved": 160, "empty": 1},
        "empty_patch": {"introduced": []},
        "first_pass": {"empty_patch": {"introduced": []}},
        "failure_analysis": {"by_instance": {}},
        "verdict": {
            "beats_v2p10": True,
            "first_pass_beats_v2p10": True,
            "first_pass_empty_delta": -1,
            "corrected_empty_delta": -1,
            "first_pass_no_new_empty_ids": True,
            "corrected_no_new_empty_ids": True,
            "behavior_healthy": True,
            "trustworthy_beats_v2p10": True,
            "resolved_delta": 3,
        },
        "provenance": {
            "training": {
                "rows": 1262,
                "base_rows": 1247,
                "new_fable_rows": 15,
                "max_seq": 32768,
            },
            "recent_fable": {"strict_rows": 15},
            "dataset": {"max_rendered_tokens": 26594},
        },
    }
    verdict_path.write_text(json.dumps(verdict) + "\n")
    monkeypatch.setattr(
        audit_module,
        "_CANONICAL_LITE300_SHA256",
        _binding(full_ids)["sha256"],
    )
    monkeypatch.setattr(
        audit_module,
        "compare_full300",
        lambda **_kwargs: verdict,
    )
    monkeypatch.setattr(
        audit_module,
        "validate_stage_data",
        lambda **_kwargs: {
            "rows": 1247,
            "base_rows": 1211,
            "new_fable_rows": 35,
        },
    )

    with pytest.raises(ValueError, match="Stage-A Fable"):
        publish_goal_completion_audit(
            verdict_path=verdict_path,
            full_ids_path=full_ids,
            v2p10_composite_path=tmp_path / "v2p10.json",
            v2p10_predictions_path=tmp_path / "v2p10_preds.json",
            v2p10_score_path=v2p10_score,
            v2p11_composite_path=tmp_path / "v2p11.json",
            v2p11_predictions_path=tmp_path / "v2p11_preds.json",
            v2p11_score_path=v2p11_score,
            candidate_name="teacher_sft_v2p11r2",
            candidate_model_path=tmp_path / "model",
            v2p10_fixed_ids_path=tmp_path / "v2p10_fixed_ids.json",
            v2p10_fixed_run=tmp_path / "v2p10_fixed",
            v2p10_complement_ids_path=tmp_path / "v2p10_complement_ids.json",
            v2p10_complement_run=tmp_path / "v2p10_complement",
            v2p11_fixed_ids_path=tmp_path / "v2p11_fixed_ids.json",
            v2p11_fixed_run=tmp_path / "v2p11_fixed",
            v2p11_complement_ids_path=tmp_path / "v2p11_complement_ids.json",
            v2p11_complement_run=tmp_path / "v2p11_complement",
            provenance_path=provenance,
            stage_data_path=stage_data,
            stage_training_contract_path=tmp_path / "stage_contract.json",
            exclusions_path=tmp_path / "exclusions.json",
            campaign_root=campaign,
            output_path=tmp_path / "audit.json",
        )
