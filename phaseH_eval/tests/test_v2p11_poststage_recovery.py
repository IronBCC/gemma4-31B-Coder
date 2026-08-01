from __future__ import annotations

import json
import importlib
from pathlib import Path
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[2]
HELPER = ROOT / "phaseH_eval" / "v2p11_poststage_recovery.py"


def _run(path: Path, *, mode: str, phase: str = "recovery-sft") -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            sys.executable,
            str(HELPER),
            "--path",
            str(path),
            "--phase",
            phase,
            "--mode",
            mode,
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
    )


def test_archives_checkpointless_recovery_output_without_deleting_evidence(
    tmp_path: Path,
) -> None:
    partial = tmp_path / "recovery"
    partial.mkdir()
    (partial / "run_manifest.json").write_text("partial\n")

    result = _run(partial, mode="checkpointless")

    assert result.returncode == 0, result.stderr
    report = json.loads(result.stdout)
    archive = Path(report["archive_path"])
    assert report["status"] == "archived"
    assert report["phase"] == "recovery-sft"
    assert not partial.exists()
    assert archive.parent == tmp_path
    assert archive.name.startswith("recovery.interrupted-")
    assert (archive / "run_manifest.json").read_text() == "partial\n"


def test_checkpointless_mode_preserves_a_resumable_checkpoint(
    tmp_path: Path,
) -> None:
    partial = tmp_path / "recovery"
    (partial / "checkpoint-5").mkdir(parents=True)

    result = _run(partial, mode="checkpointless")

    assert result.returncode == 0, result.stderr
    report = json.loads(result.stdout)
    assert report == {
        "archive_path": None,
        "phase": "recovery-sft",
        "status": "checkpointed",
    }
    assert partial.is_dir()


def test_always_mode_archives_a_kto_inprogress_directory_with_checkpoints(
    tmp_path: Path,
) -> None:
    partial = tmp_path / "kto.inprogress"
    (partial / "checkpoint-20").mkdir(parents=True)

    result = _run(partial, mode="always", phase="kto-full")

    assert result.returncode == 0, result.stderr
    report = json.loads(result.stdout)
    archive = Path(report["archive_path"])
    assert report["status"] == "archived"
    assert report["phase"] == "kto-full"
    assert not partial.exists()
    assert (archive / "checkpoint-20").is_dir()


def test_checkpointless_mode_does_not_follow_a_checkpoint_symlink(
    tmp_path: Path,
) -> None:
    external = tmp_path / "external-checkpoint"
    external.mkdir()
    partial = tmp_path / "recovery"
    partial.mkdir()
    (partial / "checkpoint-5").symlink_to(
        external,
        target_is_directory=True,
    )

    result = _run(partial, mode="checkpointless")

    assert result.returncode == 0, result.stderr
    report = json.loads(result.stdout)
    assert report["status"] == "archived"
    assert external.is_dir()
    assert not partial.exists()


def test_atomic_directory_archive_refuses_to_replace_a_collision(
    tmp_path: Path,
) -> None:
    module = importlib.import_module(
        "phaseH_eval.v2p11_poststage_recovery"
    )
    rename = getattr(module, "_rename_directory_noreplace")
    source = tmp_path / "source"
    destination = tmp_path / "destination"
    source.mkdir()
    destination.mkdir()
    (source / "source.txt").write_text("source\n")
    (destination / "destination.txt").write_text("destination\n")

    with pytest.raises(FileExistsError):
        rename(source, destination)

    assert (source / "source.txt").read_text() == "source\n"
    assert (destination / "destination.txt").read_text() == "destination\n"


def test_refuses_to_archive_a_symlink(
    tmp_path: Path,
) -> None:
    target = tmp_path / "target"
    target.mkdir()
    partial = tmp_path / "recovery"
    partial.symlink_to(target, target_is_directory=True)

    result = _run(partial, mode="always")

    assert result.returncode != 0
    assert "must be a real directory" in result.stderr
    assert partial.is_symlink()
    assert target.is_dir()
