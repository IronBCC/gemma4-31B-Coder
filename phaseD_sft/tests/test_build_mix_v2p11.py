from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil

from datasets import Dataset
import pytest

import phaseD_sft.build_mix_v2p11 as mix_module
from phaseD_sft.build_mix_v2p10 import canonical_rows_sha256
from phaseD_sft.tests.test_build_mix_v2p10 import _write_dataset


def _base(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    path = _write_dataset(
        tmp_path / "teacher_train_mix_v2p10",
        count=4,
        source="open-swe",
        variant="teacher_train_mix_v2p10",
    )
    rows = [
        json.loads(line)
        for line in (path / "train.jsonl").read_text().splitlines()
        if line.strip()
    ]
    for row in rows:
        row["messages"][0]["content"] = (
            "Frozen v2.10 base: " + row["messages"][0]["content"]
        )
    manifest = json.loads((path / "manifest.json").read_text())
    shutil.rmtree(path)
    Dataset.from_list(rows).save_to_disk(str(path))
    train = path / "train.jsonl"
    train.write_text(
        "".join(
            json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n"
            for row in rows
        )
    )
    manifest["train_jsonl_sha256"] = hashlib.sha256(
        train.read_bytes()
    ).hexdigest()
    manifest["canonical_rows_sha256"] = canonical_rows_sha256(rows)
    (path / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    monkeypatch.setattr(mix_module, "PRODUCTION_V2P10_ROWS", len(rows))
    monkeypatch.setattr(
        mix_module,
        "PRODUCTION_V2P10_CANONICAL_SHA256",
        canonical_rows_sha256(rows),
    )
    return path


def _fable(tmp_path: Path, *, count: int = 2) -> Path:
    return _write_dataset(
        tmp_path / "fable5_v2p11_strict",
        count=count,
        source="teacher:claude:claude-fable-5",
        variant="fable5_v2p11_strict",
    )


def test_v2p11_builder_adds_only_strict_fable_to_frozen_v2p10(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    base = _base(tmp_path, monkeypatch)
    fable = _fable(tmp_path)
    out = tmp_path / "teacher_train_mix_v2p11"

    manifest = mix_module.build_mix(base, fable, out)

    assert manifest["dataset_variant"] == "teacher_train_mix_v2p11"
    assert manifest["base_variant"] == "teacher_train_mix_v2p10"
    assert manifest["base_rows"] == 4
    assert manifest["fable_rows"] == 2
    assert manifest["rendered"] == 6
    assert manifest["training_admitted"] == 0
    assert manifest["all_training_gates_complete"] is False
    assert manifest["success_path_distilled"] is True
    assert len(manifest["artifact_bindings"]) == 2
    assert out.is_dir()


def test_v2p11_builder_rejects_fable_source_already_represented_in_v2p10(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    base = _base(tmp_path, monkeypatch)
    fable = _fable(tmp_path)
    delta_id = json.loads(
        (fable / "train.jsonl").read_text().splitlines()[0]
    )["instance_id"]
    manifest_path = base / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["artifact_bindings"] = [{
        "delta_kind": "fable_revision",
        "instance_id": f"fable5-revision::{delta_id}",
        "source_instance_id": delta_id,
    }]
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )

    with pytest.raises(ValueError, match="already represented in v2.10"):
        mix_module.build_mix(
            base,
            fable,
            tmp_path / "teacher_train_mix_v2p11",
        )


def test_v2p11_builder_rejects_incomplete_fable_gate(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    base = _base(tmp_path, monkeypatch)
    fable = _fable(tmp_path)
    manifest_path = fable / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["training_admitted"] = 0
    manifest["all_training_gates_complete"] = False
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )

    with pytest.raises(ValueError, match="gates incomplete"):
        mix_module.build_mix(
            base,
            fable,
            tmp_path / "teacher_train_mix_v2p11",
        )
