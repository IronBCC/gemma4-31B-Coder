from __future__ import annotations

import json
from pathlib import Path

import pytest

import phaseH_eval.compare_v2p11_full300 as compare_module
import phaseH_eval.v2p11r3_goal_completion_audit as audit_module


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value) + "\n", encoding="utf-8")


def test_compare_dispatches_external_lineage_to_r3(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import phaseH_eval.v2p11r3_behavior_completion_provenance as behavior_module

    provenance = tmp_path / "provenance.json"
    _write_json(
        provenance,
        {"artifact_type": "v2p11r3_behavior_completion_provenance"},
    )
    lineage = tmp_path / "canonical-lineage.json"
    _write_json(lineage, {})
    captured: dict[str, object] = {}

    def fake_validate(path: Path, **kwargs: object) -> dict[str, object]:
        captured["path"] = path
        captured.update(kwargs)
        return {"status": "complete"}

    monkeypatch.setattr(
        behavior_module,
        "validate_completion_provenance",
        fake_validate,
    )

    compare_module.validate_completion_provenance(
        provenance,
        full_ids_path=tmp_path / "ids.json",
        v2p10_composite_path=tmp_path / "v2p10.json",
        v2p10_lineage_path=lineage,
        candidate_model_contract={"served_name": "candidate"},
        candidate_name="candidate",
    )

    assert captured["v2p10_lineage_path"] == lineage.resolve()


def test_goal_audit_dispatches_external_lineage_to_r3(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    full_ids = tmp_path / "ids.json"
    verdict = tmp_path / "verdict.json"
    v2p10_score = tmp_path / "v2p10-score.json"
    v2p11_score = tmp_path / "v2p11-score.json"
    provenance = tmp_path / "provenance.json"
    lineage = tmp_path / "canonical-lineage.json"
    _write_json(full_ids, [f"task-{index}" for index in range(300)])
    _write_json(
        verdict,
        {
            "v2p10": {"resolved": 157, "empty": 2},
            "v2p11": {"resolved": 158, "empty": 1},
            "wrong_nonempty": {
                "v2p10": ["task-a"],
                "v2p11": [],
                "introduced": [],
                "eliminated": ["task-a"],
                "delta": -1,
                "no_regression": True,
            },
        },
    )
    for path in (v2p10_score, v2p11_score, provenance, lineage):
        _write_json(path, {})
    captured: dict[str, object] = {}
    monkeypatch.setattr(
        audit_module,
        "_validate_trustworthy_verdict",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        audit_module,
        "validate_official_score_binding",
        lambda *_args, **_kwargs: {},
    )
    monkeypatch.setattr(audit_module, "_model_contract", lambda _path: {})

    def fake_validate(*_args: object, **kwargs: object) -> dict[str, object]:
        captured.update(kwargs)
        return {}

    monkeypatch.setattr(
        audit_module,
        "validate_completion_provenance",
        fake_validate,
    )
    monkeypatch.setattr(
        audit_module,
        "_validate_reasoned_lineage",
        lambda _value: None,
    )
    monkeypatch.setattr(
        audit_module,
        "_validate_behavior_poststage",
        lambda _value: None,
    )

    audit_module.publish_goal_audit(
        verdict_path=verdict,
        full_ids_path=full_ids,
        v2p10_composite_path=tmp_path / "v2p10.json",
        v2p10_predictions_path=tmp_path / "v2p10-preds.json",
        v2p10_score_path=v2p10_score,
        v2p11_composite_path=tmp_path / "v2p11.json",
        v2p11_predictions_path=tmp_path / "v2p11-preds.json",
        v2p11_score_path=v2p11_score,
        provenance_path=provenance,
        v2p10_lineage_path=lineage,
        candidate_model_path=tmp_path / "model",
        candidate_name="candidate",
        output_path=tmp_path / "audit.json",
    )

    assert captured["v2p10_lineage_path"] == lineage.resolve()
