from __future__ import annotations

import json
from pathlib import Path

import pytest

from phaseH_eval.full300_official_score_binding import (
    _validate_official_reports,
)


def _write_report(
    path: Path,
    *,
    full_ids: list[str],
    resolved: set[str],
    unresolved: set[str],
    empty: set[str],
    errors: set[str],
) -> Path:
    submitted = resolved | unresolved | empty | errors
    completed = resolved | unresolved
    value = {
        "schema_version": 2,
        "total_instances": len(full_ids),
        "submitted_instances": len(submitted),
        "submitted_ids": sorted(submitted),
        "completed_instances": len(completed),
        "completed_ids": sorted(completed),
        "resolved_instances": len(resolved),
        "resolved_ids": sorted(resolved),
        "unresolved_instances": len(unresolved),
        "unresolved_ids": sorted(unresolved),
        "empty_patch_instances": len(empty),
        "empty_patch_ids": sorted(empty),
        "error_instances": len(errors),
        "error_ids": sorted(errors),
        "incomplete_ids": sorted(set(full_ids) - submitted),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value) + "\n")
    return path


def test_validates_official_sets_and_model_failure_classification(
    tmp_path: Path,
) -> None:
    full_ids = ["a", "b", "c", "d"]
    report = _write_report(
        tmp_path / "official_report.json",
        full_ids=full_ids,
        resolved={"a"},
        unresolved={"b"},
        empty={"c"},
        errors=set(),
    )

    result = _validate_official_reports(
        full_ids=full_ids,
        run_ids={"a", "b", "c"},
        prediction_empty_ids={"c"},
        acceptance_resolved_ids={"a"},
        report_paths=[report],
        model_failure_ids=set(),
    )

    assert result == {
        "resolved_ids": ["a"],
        "unresolved_ids": ["b"],
        "empty_patch_ids": ["c"],
        "error_ids": [],
        "model_failure_ids": [],
        "submitted_ids": ["a", "b", "c"],
    }


def test_rejects_unclassified_official_score_error(
    tmp_path: Path,
) -> None:
    full_ids = ["a", "b"]
    report = _write_report(
        tmp_path / "official_report.json",
        full_ids=full_ids,
        resolved={"a"},
        unresolved=set(),
        empty=set(),
        errors={"b"},
    )

    with pytest.raises(ValueError, match="model-failure"):
        _validate_official_reports(
            full_ids=full_ids,
            run_ids={"a", "b"},
            prediction_empty_ids=set(),
            acceptance_resolved_ids={"a"},
            report_paths=[report],
            model_failure_ids=set(),
        )


def test_rejects_report_count_drift(tmp_path: Path) -> None:
    full_ids = ["a", "b"]
    report = _write_report(
        tmp_path / "official_report.json",
        full_ids=full_ids,
        resolved={"a"},
        unresolved={"b"},
        empty=set(),
        errors=set(),
    )
    value = json.loads(report.read_text())
    value["resolved_instances"] = 0
    report.write_text(json.dumps(value) + "\n")

    with pytest.raises(ValueError, match="counts"):
        _validate_official_reports(
            full_ids=full_ids,
            run_ids={"a", "b"},
            prediction_empty_ids=set(),
            acceptance_resolved_ids={"a"},
            report_paths=[report],
            model_failure_ids=set(),
        )
