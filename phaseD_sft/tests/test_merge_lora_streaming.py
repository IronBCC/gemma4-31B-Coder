import ast
import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from phaseD_sft.atomic_checkpoint_publish import publish_after_audit


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "phaseD_sft" / "merge_lora_streaming.py"


class MergeLoraStreamingContractTests(unittest.TestCase):
    def test_failed_audit_does_not_publish_staging_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            staging = root / ".merged.tmp-1"
            output = root / "merged"
            staging.mkdir()
            (staging / "weights").write_text("candidate")

            def reject(candidate: Path) -> None:
                self.assertEqual(candidate, staging)
                self.assertFalse(output.exists())
                raise ValueError("bad checkpoint")

            with self.assertRaisesRegex(ValueError, "bad checkpoint"):
                publish_after_audit(staging, output, reject)

            self.assertFalse(output.exists())
            self.assertTrue(staging.exists())

    def test_adapter_loader_keeps_low_rank_pairs_without_materializing_deltas(self):
        source = SCRIPT.read_text(encoding="utf-8")
        tree = ast.parse(source)
        loaders = [
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef)
            and node.name == "load_adapter_pairs"
        ]

        self.assertEqual(len(loaders), 1)
        self.assertFalse(
            any(isinstance(node, ast.MatMult) for node in ast.walk(loaders[0])),
            "adapter loading must not materialize every full-rank LoRA delta",
        )

    def test_output_is_staged_atomically_and_peak_rss_is_fail_closed(self):
        source = SCRIPT.read_text(encoding="utf-8")

        self.assertIn("--max-rss-gb", source)
        self.assertIn('proc_memory_kib("VmHWM")', source)
        self.assertIn("assert_peak_rss", source)
        self.assertIn("publish_after_audit", source)
        self.assertNotIn("out.mkdir(parents=True", source)


@unittest.skipUnless(
    importlib.util.find_spec("torch") is not None
    and importlib.util.find_spec("safetensors") is not None,
    "torch and safetensors are required for the merge integration test",
)
class MergeLoraStreamingIntegrationTests(unittest.TestCase):
    def test_tiny_checkpoint_merges_exactly_and_publishes_only_complete_output(self):
        import torch
        from safetensors import safe_open
        from safetensors.torch import save_file

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            base = root / "base"
            adapter = root / "adapter"
            output = root / "merged"
            base.mkdir()
            adapter.mkdir()
            base_weight = torch.tensor(
                [[1.0, 2.0], [3.0, 4.0]],
                dtype=torch.bfloat16,
            )
            save_file(
                {
                    "model.layer.weight": base_weight,
                    "model.vision.weight": torch.ones(2, dtype=torch.bfloat16),
                },
                base / "model-00001-of-00001.safetensors",
            )
            (base / "model.safetensors.index.json").write_text(
                json.dumps(
                    {
                        "metadata": {"total_size": 12},
                        "weight_map": {
                            "model.layer.weight": "model-00001-of-00001.safetensors",
                            "model.vision.weight": "model-00001-of-00001.safetensors",
                        },
                    }
                ),
                encoding="utf-8",
            )
            (base / "config.json").write_text(
                json.dumps({"architectures": ["TinyModel"]}),
                encoding="utf-8",
            )
            lora_a = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
            lora_b = torch.tensor([[2.0, 0.0], [0.0, 3.0]])
            save_file(
                {
                    "base_model.model.model.layer.lora_A.default.weight": lora_a,
                    "base_model.model.model.layer.lora_B.default.weight": lora_b,
                },
                adapter / "adapter_model.safetensors",
            )
            (adapter / "adapter_config.json").write_text(
                json.dumps({"r": 2, "lora_alpha": 2}),
                encoding="utf-8",
            )

            subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT),
                    "--base",
                    str(base),
                    "--adapter",
                    str(adapter),
                    "--out",
                    str(output),
                    "--group-gb",
                    "0.000001",
                    "--max-rss-gb",
                    "2",
                    "--audit-architecture",
                    "TinyModel",
                    "--audit-tensors",
                    "2",
                    "--audit-vision",
                    "1",
                    "--audit-filename",
                    "merge_audit.json",
                ],
                check=True,
                capture_output=True,
                text=True,
            )

            index = json.loads(
                (output / "model.safetensors.index.json").read_text(encoding="utf-8")
            )
            shard = output / index["weight_map"]["model.layer.weight"]
            with safe_open(shard, framework="pt", device="cpu") as handle:
                merged = handle.get_tensor("model.layer.weight")
            expected = (
                base_weight.to(torch.float32) + lora_b @ lora_a
            ).to(torch.bfloat16)
            self.assertTrue(torch.equal(merged, expected))
            self.assertEqual(len(index["weight_map"]), 2)
            audit = json.loads(
                (output / "merge_audit.json").read_text(encoding="utf-8")
            )
            self.assertTrue(audit["complete"])
            self.assertEqual(audit["actual_tensors"], 2)
            self.assertEqual(audit["actual_vision"], 1)
            self.assertFalse(
                any(output.parent.glob(f".{output.name}.tmp-*")),
                "successful merge left a staging directory",
            )

    def test_failed_prepublish_audit_leaves_no_final_output(self):
        import torch
        from safetensors.torch import save_file

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            base = root / "base"
            adapter = root / "adapter"
            output = root / "merged"
            base.mkdir()
            adapter.mkdir()
            save_file(
                {"model.layer.weight": torch.ones((2, 2), dtype=torch.bfloat16)},
                base / "model-00001-of-00001.safetensors",
            )
            (base / "model.safetensors.index.json").write_text(
                json.dumps(
                    {
                        "metadata": {"total_size": 8},
                        "weight_map": {
                            "model.layer.weight": "model-00001-of-00001.safetensors",
                        },
                    }
                ),
                encoding="utf-8",
            )
            (base / "config.json").write_text(
                json.dumps({"architectures": ["TinyModel"]}),
                encoding="utf-8",
            )
            save_file(
                {
                    "base_model.model.model.layer.lora_A.default.weight": torch.eye(2),
                    "base_model.model.model.layer.lora_B.default.weight": torch.eye(2),
                },
                adapter / "adapter_model.safetensors",
            )
            (adapter / "adapter_config.json").write_text(
                json.dumps({"r": 2, "lora_alpha": 2}),
                encoding="utf-8",
            )

            result = subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT),
                    "--base",
                    str(base),
                    "--adapter",
                    str(adapter),
                    "--out",
                    str(output),
                    "--group-gb",
                    "0.000001",
                    "--max-rss-gb",
                    "2",
                    "--audit-architecture",
                    "TinyModel",
                    "--audit-tensors",
                    "2",
                    "--audit-vision",
                    "0",
                    "--audit-filename",
                    "merge_audit.json",
                ],
                check=False,
                capture_output=True,
                text=True,
            )

            self.assertNotEqual(result.returncode, 0)
            self.assertFalse(output.exists())
            self.assertFalse(
                any(output.parent.glob(f".{output.name}.tmp-*"))
            )


if __name__ == "__main__":
    unittest.main()
