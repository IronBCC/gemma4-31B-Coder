import json
from pathlib import Path

import pytest

import phaseH_eval.v2p11r3_goal_completion_audit as audit_module
from phaseH_eval.v2p11r3_goal_completion_audit import (
    _validate_behavior_poststage,
    _validate_reasoned_lineage,
    publish_goal_audit,
)


def _provenance() -> dict[str, object]:
    return {
        "training": {"rows": 1262, "max_seq": 32768, "optimizer_steps": 79},
        "dataset": {
            "max_rendered_tokens": 26594,
            "fable_rows": 92,
            "canonical_bash_tool_turns": 517,
            "reasoned_tool_turns": 468,
            "stage_a": {"summary": {"rows": 1247, "base_rows": 1211, "new_fable_rows": 36}},
            "source_fable": {"source_ids": [str(index) for index in range(15)]},
        },
        "behavior_poststage": {
            "recovery_rows": 138,
            "recovery_optimizer_steps": 105,
            "behavior_rows": 606,
            "kto_optimizer_steps": 25,
            "coverage_counts": {
                "desirable_correct_patch": 25,
                "empty_terminal": 8,
                "repeated_read_loop": 8,
                "wrong_nonempty_replay": 9,
            },
            "negative_counts": {
                "empty_terminal": 33,
                "repeated_read_loop": 55,
                "wrong_nonempty_replay": 228,
            },
        },
    }


def test_reasoned_lineage_requires_all_admitted_fable_evidence() -> None:
    _validate_reasoned_lineage(_provenance())

    broken = _provenance()
    broken["dataset"]["reasoned_tool_turns"] = 467
    with pytest.raises(ValueError, match="reasoning-contract"):
        _validate_reasoned_lineage(broken)


def test_behavior_poststage_requires_all_model_level_mitigations() -> None:
    _validate_behavior_poststage(_provenance())

    broken = _provenance()
    broken["behavior_poststage"]["coverage_counts"]["empty_terminal"] = 7
    with pytest.raises(ValueError, match="behavior poststage"):
        _validate_behavior_poststage(broken)


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value) + "\n")


def _goal_audit_fixture(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> dict[str, object]:
    ids = tmp_path / "ids.json"
    verdict = tmp_path / "verdict.json"
    v2p10_score = tmp_path / "v2p10_score.json"
    v2p11_score = tmp_path / "v2p11_score.json"
    provenance = tmp_path / "provenance.json"
    output = tmp_path / "goal_audit.json"
    full_ids = [f"repo__task-{index}" for index in range(300)]
    _write_json(ids, full_ids)
    _write_json(
        verdict,
        {
            "v2p10": {"resolved": 157, "empty": 2},
            "v2p11": {"resolved": 158, "empty": 1},
            "wrong_nonempty": {
                "v2p10": ["case-a", "case-b"],
                "v2p11": ["case-a"],
                "introduced": [],
                "eliminated": ["case-b"],
                "delta": -1,
                "no_regression": True,
            },
        },
    )
    for path in (v2p10_score, v2p11_score, provenance):
        _write_json(path, {})
    outcomes = {
        "v2p10": {
            "resolved": set(full_ids[:157]),
            "empty": {full_ids[157], full_ids[158]},
            "wrong_nonempty": set(full_ids[159:]),
        },
        "v2p11": {
            "resolved": set(full_ids[:158]),
            "empty": {full_ids[158]},
            "wrong_nonempty": set(full_ids[159:]),
        },
    }
    monkeypatch.setattr(
        audit_module,
        "_validate_trustworthy_verdict",
        lambda *_args, **_kwargs: outcomes,
    )

    def validate_score(path: Path, **_kwargs: object) -> dict[str, object]:
        label = "v2p10" if Path(path) == v2p10_score else "v2p11"
        outcome = outcomes[label]
        return {
            "final": {
                "resolved_ids": sorted(outcome["resolved"]),
                "empty_ids": sorted(outcome["empty"]),
                "resolved": len(outcome["resolved"]),
                "empty": len(outcome["empty"]),
            }
        }

    monkeypatch.setattr(
        audit_module,
        "validate_official_score_binding",
        validate_score,
    )
    monkeypatch.setattr(
        audit_module,
        "_model_contract",
        lambda *_args, **_kwargs: {},
    )
    monkeypatch.setattr(
        audit_module,
        "validate_completion_provenance",
        lambda *_args, **_kwargs: _provenance(),
    )
    return {
        "verdict_path": verdict,
        "full_ids_path": ids,
        "v2p10_composite_path": tmp_path / "v2p10.json",
        "v2p10_predictions_path": tmp_path / "v2p10_preds.json",
        "v2p10_score_path": v2p10_score,
        "v2p11_composite_path": tmp_path / "v2p11.json",
        "v2p11_predictions_path": tmp_path / "v2p11_preds.json",
        "v2p11_score_path": v2p11_score,
        "provenance_path": provenance,
        "candidate_model_path": tmp_path / "model",
        "candidate_name": "teacher_sft_v2p11r3_behavior",
        "output_path": output,
    }


def test_goal_audit_reuses_an_identical_existing_report(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    inputs = _goal_audit_fixture(tmp_path, monkeypatch)

    first = publish_goal_audit(**inputs)
    second = publish_goal_audit(**inputs)

    assert second == first


def test_goal_audit_records_wrong_nonempty_nonregression(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    inputs = _goal_audit_fixture(tmp_path, monkeypatch)

    report = publish_goal_audit(**inputs)

    assert report["requirements"]["wrong_nonempty_no_regression"] is True
    assert report["score"]["wrong_nonempty_delta"] == -1


def test_goal_audit_rejects_verdict_that_differs_from_official_scores(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    inputs = _goal_audit_fixture(tmp_path, monkeypatch)
    monkeypatch.setattr(
        audit_module,
        "validate_official_score_binding",
        lambda *_args, **_kwargs: {
            "final": {
                "resolved_ids": [],
                "empty_ids": [],
                "resolved": 0,
                "empty": 0,
            }
        },
    )

    with pytest.raises(ValueError, match="official score"):
        publish_goal_audit(**inputs)


def test_goal_audit_rejects_an_existing_report_when_inputs_change(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    inputs = _goal_audit_fixture(tmp_path, monkeypatch)
    publish_goal_audit(**inputs)
    verdict = Path(inputs["verdict_path"])
    value = json.loads(verdict.read_text())
    value["v2p11"]["resolved"] = 159
    _write_json(verdict, value)

    with pytest.raises(ValueError, match="official score"):
        publish_goal_audit(**inputs)
