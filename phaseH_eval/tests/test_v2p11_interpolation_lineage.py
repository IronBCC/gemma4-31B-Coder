from __future__ import annotations

import hashlib
import importlib
import json
from pathlib import Path

import pytest


COPIED_FILES = [
    "chat_template.jinja",
    "config.json",
    "tokenizer_config.json",
]


def _binding(path: Path) -> dict[str, object]:
    path = path.resolve()
    return {
        "path": str(path),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "bytes": path.stat().st_size,
    }


def _write_model(root: Path, *, payload: bytes) -> None:
    root.mkdir()
    shard = root / "model-00001-of-00001.safetensors"
    shard.write_bytes(payload)
    (root / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "metadata": {"total_size": 12},
                "weight_map": {
                    "model.a": shard.name,
                    "model.b": shard.name,
                    "model.position_ids": shard.name,
                },
            }
        ),
        encoding="utf-8",
    )
    (root / "config.json").write_text(
        json.dumps({"architectures": ["TinyModel"]}),
        encoding="utf-8",
    )
    (root / "tokenizer_config.json").write_text(
        json.dumps({"model_max_length": 32768}),
        encoding="utf-8",
    )
    (root / "chat_template.jinja").write_text(
        "{{ messages }}\n",
        encoding="utf-8",
    )


def _contract(root: Path) -> dict[str, object]:
    index = root / "model.safetensors.index.json"
    shard = root / "model-00001-of-00001.safetensors"
    return {
        "model_path": str(root.resolve()),
        "model_config_sha256": hashlib.sha256(
            (root / "config.json").read_bytes()
        ).hexdigest(),
        "model_index_sha256": hashlib.sha256(index.read_bytes()).hexdigest(),
        "model_safetensors_sha256": None,
        "model_artifacts": [_binding(shard)],
        "copied_file_bindings": [
            _binding(root / name) for name in COPIED_FILES
        ],
    }


def _write_fixture(tmp_path: Path) -> dict[str, Path]:
    anchor = tmp_path / "v2p10"
    source = tmp_path / "v2p11r3"
    output = tmp_path / "v2p11r4"
    _write_model(anchor, payload=b"anchor")
    _write_model(source, payload=b"source")
    _write_model(output, payload=b"output")
    manifest = {
        "schema_version": 1,
        "artifact_type": "checkpoint_weight_interpolation",
        "status": "complete",
        "alpha": 0.25,
        "equation": "anchor + alpha * (candidate - anchor)",
        "anchor": _contract(anchor),
        "candidate": _contract(source),
        "output": _contract(output),
        "tensor_count": 3,
        "floating_tensors": 2,
        "copied_nonfloating_tensors": 1,
        "total_tensor_bytes": 12,
        "output_shards": 1,
        "copied_files": COPIED_FILES,
        "copied_file_bindings": _contract(output)[
            "copied_file_bindings"
        ],
        "peak_rss_bytes": 1234,
    }
    manifest_path = output / "interpolation_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return {
        "anchor": anchor,
        "source": source,
        "output": output,
        "manifest": manifest_path,
    }


def _validate(paths: dict[str, Path], *, name: str = "teacher_sft_v2p11r4_blend25"):
    module = importlib.import_module("phaseH_eval.v2p11_interpolation_lineage")
    return module.validate_interpolation_lineage(
        manifest_path=paths["manifest"],
        anchor_model_path=paths["anchor"],
        source_model_path=paths["source"],
        output_model_path=paths["output"],
        candidate_name=name,
    )


def test_validates_exact_interpolation_lineage(tmp_path: Path) -> None:
    paths = _write_fixture(tmp_path)

    report = _validate(paths)

    assert report["status"] == "complete"
    assert report["alpha"] == 0.25
    assert report["served_name"] == "teacher_sft_v2p11r4_blend25"
    assert report["manifest"] == _binding(paths["manifest"])
    assert report["output"] == _contract(paths["output"])


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("parent", "anchor model contract changed"),
        ("source_parent", "source model contract changed"),
        ("alpha", "interpolation alpha changed"),
        ("copied_file", "copied checkpoint files differ"),
        ("output", "output model contract changed"),
        ("output_path", "interpolation manifest path changed"),
        ("candidate_name", "candidate served name changed"),
    ],
)
def test_rejects_lineage_mutations_before_serving(
    tmp_path: Path,
    mutation: str,
    message: str,
) -> None:
    paths = _write_fixture(tmp_path)
    name = "teacher_sft_v2p11r4_blend25"
    if mutation == "parent":
        (paths["anchor"] / "model-00001-of-00001.safetensors").write_bytes(
            b"tampered"
        )
    elif mutation == "source_parent":
        (paths["source"] / "model-00001-of-00001.safetensors").write_bytes(
            b"tampered"
        )
    elif mutation == "alpha":
        manifest = json.loads(paths["manifest"].read_text(encoding="utf-8"))
        manifest["alpha"] = 0.5
        paths["manifest"].write_text(json.dumps(manifest), encoding="utf-8")
    elif mutation == "copied_file":
        (paths["source"] / "tokenizer_config.json").write_text(
            json.dumps({"model_max_length": 4096}),
            encoding="utf-8",
        )
    elif mutation == "output":
        (paths["output"] / "model-00001-of-00001.safetensors").write_bytes(
            b"tampered"
        )
    elif mutation == "output_path":
        wrong_output = tmp_path / "wrong-output"
        _write_model(wrong_output, payload=b"wrong")
        paths["output"] = wrong_output
    elif mutation == "candidate_name":
        name = "teacher_sft_v2p11r4_wrong"

    with pytest.raises(ValueError, match=message):
        _validate(paths, name=name)
