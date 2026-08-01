from __future__ import annotations

import json
from pathlib import Path

import pytest

from phaseH_eval.v2p11_poststage_phase_marker import publish_or_verify


def test_publishes_and_reopens_immutable_phase_marker(tmp_path: Path) -> None:
    first = tmp_path / "adapter.safetensors"
    second = tmp_path / "trainer_state.json"
    marker = tmp_path / "phase.json"
    first.write_bytes(b"adapter")
    second.write_text('{"global_step": 105}\n')

    report = publish_or_verify(
        marker_path=marker,
        phase="recovery_sft",
        artifact_paths=[first, second],
    )

    assert report["phase"] == "recovery_sft"
    assert report["status"] == "complete"
    assert [item["bytes"] for item in report["artifacts"]] == [7, 21]
    assert publish_or_verify(
        marker_path=marker,
        phase="recovery_sft",
        artifact_paths=[first, second],
    ) == report
    assert json.loads(marker.read_text()) == report


def test_rejects_tampered_phase_artifact(tmp_path: Path) -> None:
    artifact = tmp_path / "adapter.safetensors"
    marker = tmp_path / "phase.json"
    artifact.write_bytes(b"before")
    publish_or_verify(
        marker_path=marker,
        phase="kto_full",
        artifact_paths=[artifact],
    )
    artifact.write_bytes(b"after")

    with pytest.raises(ValueError, match="immutable phase marker changed"):
        publish_or_verify(
            marker_path=marker,
            phase="kto_full",
            artifact_paths=[artifact],
        )


def test_rejects_missing_phase_artifact(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="phase artifact is missing"):
        publish_or_verify(
            marker_path=tmp_path / "phase.json",
            phase="final_merge",
            artifact_paths=[tmp_path / "missing"],
        )
