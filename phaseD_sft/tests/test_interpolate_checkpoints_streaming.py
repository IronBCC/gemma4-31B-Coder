import ast
import json
import subprocess
import sys
from pathlib import Path

import pytest


torch = pytest.importorskip("torch")
safetensors = pytest.importorskip("safetensors")
from safetensors import safe_open
from safetensors.torch import save_file

from phaseD_sft.atomic_checkpoint_publish import publish_after_audit


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "phaseD_sft" / "interpolate_checkpoints_streaming.py"


def _write_checkpoint(
    root: Path,
    *,
    tensors: dict[str, torch.Tensor],
    placement: dict[str, str],
) -> None:
    root.mkdir()
    for shard in sorted(set(placement.values())):
        save_file(
            {
                key: tensor
                for key, tensor in tensors.items()
                if placement[key] == shard
            },
            root / shard,
        )
    (root / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "metadata": {
                    "total_size": sum(
                        tensor.numel() * tensor.element_size()
                        for tensor in tensors.values()
                    )
                },
                "weight_map": placement,
            }
        ),
        encoding="utf-8",
    )
    (root / "config.json").write_text(
        json.dumps({"architectures": ["TinyModel"], "hidden_size": 2}),
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


def _run_interpolation(
    anchor: Path,
    candidate: Path,
    output: Path,
    *,
    group_gb: str = "0.000001",
):
    return subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--anchor",
            str(anchor),
            "--candidate",
            str(candidate),
            "--out",
            str(output),
            "--alpha",
            "0.25",
            "--group-gb",
            group_gb,
            "--max-rss-gb",
            "2",
        ],
        check=False,
        capture_output=True,
        text=True,
    )


def test_blending_path_streams_parent_slices():
    tree = ast.parse(SCRIPT.read_text(encoding="utf-8"))
    functions = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "_interpolate_tensor_streaming"
    ]

    assert len(functions) == 1
    calls = [
        node.func.attr
        for node in ast.walk(functions[0])
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
    ]
    assert "get_slice" in calls


def test_atomic_publish_does_not_replace_concurrent_empty_output(
    tmp_path: Path,
):
    staging = tmp_path / ".output.tmp-1"
    output = tmp_path / "output"
    staging.mkdir()
    (staging / "model").write_text("candidate", encoding="utf-8")

    def audit(_staging: Path) -> None:
        output.mkdir()

    with pytest.raises(FileExistsError):
        publish_after_audit(staging, output, audit)

    assert output.is_dir()
    assert not list(output.iterdir())
    assert (staging / "model").read_text(encoding="utf-8") == "candidate"


def test_interpolates_by_key_across_different_parent_shards(tmp_path: Path):
    anchor = tmp_path / "anchor"
    candidate = tmp_path / "candidate"
    output = tmp_path / "output"
    anchor_tensors = {
        "model.layer_a.weight": torch.tensor(
            [[1.0, 2.0], [3.0, 4.0]], dtype=torch.bfloat16
        ),
        "model.layer_b.weight": torch.tensor(
            [10.0, 20.0], dtype=torch.bfloat16
        ),
        "model.position_ids": torch.tensor([0, 1], dtype=torch.int64),
    }
    candidate_tensors = {
        "model.layer_a.weight": torch.tensor(
            [[5.0, 6.0], [7.0, 8.0]], dtype=torch.bfloat16
        ),
        "model.layer_b.weight": torch.tensor(
            [14.0, 28.0], dtype=torch.bfloat16
        ),
        "model.position_ids": torch.tensor([0, 1], dtype=torch.int64),
    }
    _write_checkpoint(
        anchor,
        tensors=anchor_tensors,
        placement={
            "model.layer_a.weight": "model-00001-of-00002.safetensors",
            "model.layer_b.weight": "model-00002-of-00002.safetensors",
            "model.position_ids": "model-00002-of-00002.safetensors",
        },
    )
    _write_checkpoint(
        candidate,
        tensors=candidate_tensors,
        placement={
            "model.layer_a.weight": "model-00002-of-00002.safetensors",
            "model.layer_b.weight": "model-00001-of-00002.safetensors",
            "model.position_ids": "model-00001-of-00002.safetensors",
        },
    )

    result = _run_interpolation(anchor, candidate, output)

    assert result.returncode == 0, result.stderr
    index = json.loads(
        (output / "model.safetensors.index.json").read_text(encoding="utf-8")
    )
    actual = {}
    for shard in sorted(set(index["weight_map"].values())):
        with safe_open(output / shard, framework="pt", device="cpu") as handle:
            actual.update({key: handle.get_tensor(key) for key in handle.keys()})
    expected_a = (
        anchor_tensors["model.layer_a.weight"].float()
        + 0.25
        * (
            candidate_tensors["model.layer_a.weight"].float()
            - anchor_tensors["model.layer_a.weight"].float()
        )
    ).to(torch.bfloat16)
    expected_b = (
        anchor_tensors["model.layer_b.weight"].float()
        + 0.25
        * (
            candidate_tensors["model.layer_b.weight"].float()
            - anchor_tensors["model.layer_b.weight"].float()
        )
    ).to(torch.bfloat16)
    assert torch.equal(actual["model.layer_a.weight"], expected_a)
    assert torch.equal(actual["model.layer_b.weight"], expected_b)
    assert torch.equal(
        actual["model.position_ids"], anchor_tensors["model.position_ids"]
    )
    manifest = json.loads(
        (output / "interpolation_manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["schema_version"] == 1
    assert manifest["artifact_type"] == "checkpoint_weight_interpolation"
    assert manifest["status"] == "complete"
    assert manifest["alpha"] == 0.25
    assert manifest["anchor"]["model_path"] == str(anchor.resolve())
    assert manifest["candidate"]["model_path"] == str(candidate.resolve())
    assert manifest["output"]["model_path"] == str(output.resolve())
    assert {
        Path(binding["path"]).name
        for binding in manifest["anchor"]["copied_file_bindings"]
    } == {"chat_template.jinja", "config.json", "tokenizer_config.json"}
    assert {
        Path(binding["path"]).name
        for binding in manifest["candidate"]["copied_file_bindings"]
    } == {"chat_template.jinja", "config.json", "tokenizer_config.json"}
    assert manifest["tensor_count"] == 3
    assert manifest["floating_tensors"] == 2
    assert manifest["copied_nonfloating_tensors"] == 1
    assert manifest["copied_files"] == [
        "chat_template.jinja",
        "config.json",
        "tokenizer_config.json",
    ]


def test_interpolates_float16_and_preserves_anchor_dtype(tmp_path: Path):
    anchor = tmp_path / "anchor"
    candidate = tmp_path / "candidate"
    output = tmp_path / "output"
    key = "model.layer.weight"
    _write_checkpoint(
        anchor,
        tensors={key: torch.tensor([1.0, 3.0], dtype=torch.float16)},
        placement={key: "model-00001-of-00001.safetensors"},
    )
    _write_checkpoint(
        candidate,
        tensors={key: torch.tensor([5.0, 7.0], dtype=torch.float16)},
        placement={key: "model-00001-of-00001.safetensors"},
    )

    result = _run_interpolation(anchor, candidate, output)

    assert result.returncode == 0, result.stderr
    index = json.loads(
        (output / "model.safetensors.index.json").read_text(encoding="utf-8")
    )
    with safe_open(
        output / index["weight_map"][key], framework="pt", device="cpu"
    ) as handle:
        actual = handle.get_tensor(key)
    assert actual.dtype == torch.float16
    assert torch.equal(actual, torch.tensor([2.0, 4.0], dtype=torch.float16))


def test_rejects_mismatched_copied_files_before_staging(tmp_path: Path):
    anchor = tmp_path / "anchor"
    candidate = tmp_path / "candidate"
    output = tmp_path / "output"
    key = "model.layer.weight"
    tensor = torch.ones(2, dtype=torch.bfloat16)
    placement = {key: "model-00001-of-00001.safetensors"}
    _write_checkpoint(anchor, tensors={key: tensor}, placement=placement)
    _write_checkpoint(candidate, tensors={key: tensor}, placement=placement)
    (candidate / "config.json").write_text(
        json.dumps({"architectures": ["OtherModel"], "hidden_size": 2}),
        encoding="utf-8",
    )

    result = _run_interpolation(anchor, candidate, output)

    assert result.returncode != 0
    assert "checkpoint copied-file mismatch: config.json" in result.stderr
    assert not output.exists()
    assert not list(tmp_path.glob(".output.tmp-*"))


def test_allows_only_tokenizer_padding_state_drift_and_copies_anchor(
    tmp_path: Path,
):
    anchor = tmp_path / "anchor"
    candidate = tmp_path / "candidate"
    output = tmp_path / "output"
    key = "model.layer.weight"
    tensor = torch.ones(2, dtype=torch.bfloat16)
    placement = {key: "model-00001-of-00001.safetensors"}
    _write_checkpoint(anchor, tensors={key: tensor}, placement=placement)
    _write_checkpoint(candidate, tensors={key: tensor}, placement=placement)
    anchor_tokenizer = {
        "version": "1.0",
        "padding": None,
        "model": {"type": "Unigram", "vocab": [["<pad>", 0.0]]},
    }
    candidate_tokenizer = {
        **anchor_tokenizer,
        "padding": {
            "strategy": "BatchLongest",
            "direction": "Left",
            "pad_id": 0,
            "pad_token": "<pad>",
        },
    }
    (anchor / "tokenizer.json").write_text(
        json.dumps(anchor_tokenizer), encoding="utf-8"
    )
    (candidate / "tokenizer.json").write_text(
        json.dumps(candidate_tokenizer), encoding="utf-8"
    )
    (anchor / "tokenizer_config.json").write_text(
        json.dumps({"model_max_length": 32768, "padding_side": "left"}),
        encoding="utf-8",
    )
    (candidate / "tokenizer_config.json").write_text(
        json.dumps({"model_max_length": 32768, "padding_side": "right"}),
        encoding="utf-8",
    )

    result = _run_interpolation(anchor, candidate, output)

    assert result.returncode == 0, result.stderr
    assert (output / "tokenizer.json").read_bytes() == (
        anchor / "tokenizer.json"
    ).read_bytes()
    assert (output / "tokenizer_config.json").read_bytes() == (
        anchor / "tokenizer_config.json"
    ).read_bytes()
    manifest = json.loads(
        (output / "interpolation_manifest.json").read_text(encoding="utf-8")
    )
    differences = manifest["controlled_copy_file_differences"]
    assert set(differences) == {"tokenizer.json", "tokenizer_config.json"}
    assert differences["tokenizer.json"]["ignored_top_level_fields"] == [
        "padding"
    ]
    assert differences["tokenizer_config.json"][
        "ignored_top_level_fields"
    ] == ["padding_side"]
    assert all(
        len(row["semantic_sha256"]) == 64 for row in differences.values()
    )
    assert all(
        row["output_source"] == "anchor" for row in differences.values()
    )


def test_rejects_tokenizer_vocabulary_drift_even_when_padding_differs(
    tmp_path: Path,
):
    anchor = tmp_path / "anchor"
    candidate = tmp_path / "candidate"
    output = tmp_path / "output"
    key = "model.layer.weight"
    tensor = torch.ones(2, dtype=torch.bfloat16)
    placement = {key: "model-00001-of-00001.safetensors"}
    _write_checkpoint(anchor, tensors={key: tensor}, placement=placement)
    _write_checkpoint(candidate, tensors={key: tensor}, placement=placement)
    (anchor / "tokenizer.json").write_text(
        json.dumps(
            {
                "padding": None,
                "model": {"vocab": [["<pad>", 0.0], ["a", -1.0]]},
            }
        ),
        encoding="utf-8",
    )
    (candidate / "tokenizer.json").write_text(
        json.dumps(
            {
                "padding": {"pad_id": 0},
                "model": {"vocab": [["<pad>", 0.0], ["b", -1.0]]},
            }
        ),
        encoding="utf-8",
    )

    result = _run_interpolation(anchor, candidate, output)

    assert result.returncode != 0
    assert "checkpoint copied-file mismatch: tokenizer.json" in result.stderr
    assert not output.exists()


def test_rejects_mismatched_tensor_keys(tmp_path: Path):
    anchor = tmp_path / "anchor"
    candidate = tmp_path / "candidate"
    output = tmp_path / "output"
    placement = {"model.a": "model-00001-of-00001.safetensors"}
    _write_checkpoint(
        anchor,
        tensors={"model.a": torch.ones(2, dtype=torch.bfloat16)},
        placement=placement,
    )
    _write_checkpoint(
        candidate,
        tensors={"model.b": torch.ones(2, dtype=torch.bfloat16)},
        placement={"model.b": "model-00001-of-00001.safetensors"},
    )

    result = _run_interpolation(anchor, candidate, output)

    assert result.returncode != 0
    assert "checkpoint tensor-key mismatch" in result.stderr
    assert not output.exists()


@pytest.mark.parametrize(
    "candidate_tensor",
    [
        pytest.param(
            torch.ones(3, dtype=torch.bfloat16), id="shape"
        ),
        pytest.param(torch.ones(2, dtype=torch.float16), id="dtype"),
    ],
)
def test_rejects_mismatched_tensor_specs(
    tmp_path: Path, candidate_tensor: torch.Tensor
):
    anchor = tmp_path / "anchor"
    candidate = tmp_path / "candidate"
    output = tmp_path / "output"
    key = "model.layer.weight"
    placement = {key: "model-00001-of-00001.safetensors"}
    _write_checkpoint(
        anchor,
        tensors={key: torch.ones(2, dtype=torch.bfloat16)},
        placement=placement,
    )
    _write_checkpoint(
        candidate, tensors={key: candidate_tensor}, placement=placement
    )

    result = _run_interpolation(anchor, candidate, output)

    assert result.returncode != 0
    assert "checkpoint tensor shape/dtype mismatch" in result.stderr
    assert not output.exists()


def test_rejects_nonfloating_mismatch_and_removes_staging(tmp_path: Path):
    anchor = tmp_path / "anchor"
    candidate = tmp_path / "candidate"
    output = tmp_path / "output"
    key = "model.position_ids"
    placement = {key: "model-00001-of-00001.safetensors"}
    _write_checkpoint(
        anchor,
        tensors={key: torch.tensor([0, 1], dtype=torch.int64)},
        placement=placement,
    )
    _write_checkpoint(
        candidate,
        tensors={key: torch.tensor([0, 2], dtype=torch.int64)},
        placement=placement,
    )

    result = _run_interpolation(anchor, candidate, output)

    assert result.returncode != 0
    assert "non-floating tensor mismatch" in result.stderr
    assert not output.exists()
    assert not list(tmp_path.glob(".output.tmp-*"))


def test_refuses_to_overwrite_existing_output(tmp_path: Path):
    anchor = tmp_path / "anchor"
    candidate = tmp_path / "candidate"
    output = tmp_path / "output"
    key = "model.layer.weight"
    tensor = torch.ones(2, dtype=torch.bfloat16)
    placement = {key: "model-00001-of-00001.safetensors"}
    _write_checkpoint(anchor, tensors={key: tensor}, placement=placement)
    _write_checkpoint(candidate, tensors={key: tensor}, placement=placement)
    output.mkdir()
    marker = output / "keep"
    marker.write_text("existing", encoding="utf-8")

    result = _run_interpolation(anchor, candidate, output)

    assert result.returncode != 0
    assert "output already exists" in result.stderr
    assert marker.read_text(encoding="utf-8") == "existing"


def test_rejects_tensor_larger_than_shard_budget(tmp_path: Path):
    anchor = tmp_path / "anchor"
    candidate = tmp_path / "candidate"
    output = tmp_path / "output"
    key = "model.layer.weight"
    tensor = torch.ones(2, dtype=torch.bfloat16)
    placement = {key: "model-00001-of-00001.safetensors"}
    _write_checkpoint(anchor, tensors={key: tensor}, placement=placement)
    _write_checkpoint(candidate, tensors={key: tensor}, placement=placement)

    result = _run_interpolation(
        anchor, candidate, output, group_gb="0.000000001"
    )

    assert result.returncode != 0
    assert "tensor exceeds --group-gb shard budget" in result.stderr
    assert not output.exists()
    assert not list(tmp_path.glob(".output.tmp-*"))


def test_failed_output_audit_removes_staging(tmp_path: Path):
    anchor = tmp_path / "anchor"
    candidate = tmp_path / "candidate"
    output = tmp_path / "output"
    key = "model.layer.weight"
    placement = {key: "model-00001-of-00001.safetensors"}
    _write_checkpoint(
        anchor,
        tensors={key: torch.ones(2, dtype=torch.bfloat16)},
        placement=placement,
    )
    _write_checkpoint(
        candidate,
        tensors={key: torch.tensor([float("inf"), 1.0], dtype=torch.bfloat16)},
        placement=placement,
    )

    result = _run_interpolation(anchor, candidate, output)

    assert result.returncode != 0
    assert "non-finite interpolated tensor" in result.stderr
    assert not output.exists()
    assert not list(tmp_path.glob(".output.tmp-*"))
