from __future__ import annotations

import json

import pytest

from teacher_platform.fable5_verify import (
    build_verification_manifest,
    missing_admission_trajectories,
    preflight_candidate_plan,
    row_level_control_failure,
    select_representative_rows,
)
from teacher_platform.fable5_replay import ReplayContractError


def _sidecar(task: str, trajectory_id: str) -> dict[str, object]:
    return {
        "trajectory_id": trajectory_id,
        "source_instance_id": task,
        "source_terminal_sha256": task[0] * 64,
        "row": {"task": task},
    }


def _manifest(*representatives: tuple[str, str]) -> dict[str, object]:
    return {
        "source": {
            "dataset_revision": "aef8506515979988aa5c1a423f5b0fb3cee60382"
        },
        "eligible_unique_task_ceiling_before_replay": len(representatives),
        "representatives": [
            {"task": task, "trajectory_id": trajectory_id}
            for task, trajectory_id in representatives
        ],
    }


def test_select_representative_rows_preserves_manifest_order() -> None:
    first = "1" * 64
    second = "2" * 64
    selected = select_representative_rows(
        [_sidecar("py-two", second), _sidecar("py-one", first)],
        _manifest(("py-one", first), ("py-two", second)),
    )

    assert [row["source_instance_id"] for row in selected] == [
        "py-one",
        "py-two",
    ]


def test_select_representative_rows_accepts_validated_structural_task_key() -> None:
    trajectory_id = "3" * 64
    selected = select_representative_rows(
        [
            {
                "trajectory_id": trajectory_id,
                "task": "py-three",
                "source_terminal_sha256": "a" * 64,
                "row": {"task": "py-three"},
            }
        ],
        _manifest(("py-three", trajectory_id)),
    )

    assert selected[0]["task"] == "py-three"


def test_select_representative_rows_rejects_missing_sidecar_binding() -> None:
    with pytest.raises(ValueError, match="missing from structural sidecar"):
        select_representative_rows(
            [_sidecar("py-one", "1" * 64)],
            _manifest(("py-one", "1" * 64), ("py-two", "2" * 64)),
        )


def test_build_verification_manifest_counts_outcomes_and_hashes_ledger(
    tmp_path,
) -> None:
    ledger = tmp_path / "replay.jsonl"
    ledger.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "trajectory_id": "1" * 64,
                        "status": "verified",
                        "failure_class": None,
                    }
                ),
                json.dumps(
                    {
                        "trajectory_id": "2" * 64,
                        "status": "rejected",
                        "failure_class": "verifier_failed",
                    }
                ),
            ]
        )
        + "\n"
    )

    manifest = build_verification_manifest(
        selected_rows=[
            {
                "trajectory_id": "1" * 64,
                "source_instance_id": "py-one",
                "row": {"lang": "python"},
            },
            {
                "trajectory_id": "2" * 64,
                "source_instance_id": "py-two",
                "row": {"lang": "python"},
            },
        ],
        ledger_records=[
            {
                "trajectory_id": "1" * 64,
                "status": "verified",
                "failure_class": None,
            },
            {
                "trajectory_id": "2" * 64,
                "status": "rejected",
                "failure_class": "verifier_failed",
            },
        ],
        ledger_path=ledger,
        elapsed_seconds=12.5,
    )

    assert manifest["selected"] == 2
    assert manifest["verified"] == 1
    assert manifest["rejected"] == 1
    assert manifest["per_language"] == {
        "python": {"selected": 2, "verified": 1}
    }
    assert manifest["failure_counts"] == {"verifier_failed": 1}
    assert len(manifest["ledger_sha256"]) == 64


def test_missing_admission_is_row_scoped_not_a_global_block() -> None:
    rows = [
        {
            "trajectory_id": "1" * 64,
            "row": {"lang": "python"},
        },
        {
            "trajectory_id": "2" * 64,
            "row": {"lang": "cpp"},
        },
    ]

    assert missing_admission_trajectories(rows, {"python"}) == frozenset(
        {"2" * 64}
    )


def test_candidate_reconstruction_failure_is_a_row_level_rejection(tmp_path) -> None:
    def reject(_contract, _plan, _destination):
        raise ReplayContractError("mutation ancestor is missing")

    assert (
        preflight_candidate_plan(
            object(), object(), tmp_path / "candidate", reconstructor=reject
        )
        == "mutation ancestor is missing"
    )


def test_only_reviewed_quiescence_error_is_a_row_level_control_failure() -> None:
    assert (
        row_level_control_failure(
            ReplayContractError(
                "restricted Docker verifier process quiescence failed"
            )
        )
        == "verifier process quiescence failed"
    )
    assert (
        row_level_control_failure(
            ReplayContractError("restricted Docker cached image inspect failed")
        )
        is None
    )
