import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from phaseE_rl.merge_swe_rlvr_for_eval import (
    build_arg_parser,
    lora_delta,
    lora_key_to_module_path,
)


class LoraKeyPathTests(unittest.TestCase):
    def test_merge_parser_accepts_bounded_shard_size(self):
        args = build_arg_parser().parse_args([
            "--sft-adapter", "sft", "--rl-adapter", "rl", "--out", "out",
            "--max-shard-size", "20GB",
        ])
        self.assertEqual(args.max_shard_size, "20GB")

    def test_remaps_multimodal_sft_key_to_grafted_text_path(self):
        key = "base_model.model.model.language_model.layers.3.mlp.down_proj.lora_A.weight"
        self.assertEqual(
            lora_key_to_module_path(key),
            "model.layers.3.mlp.down_proj",
        )

    def test_preserves_text_model_rl_key(self):
        key = "base_model.model.model.layers.3.self_attn.q_proj.lora_A.weight"
        self.assertEqual(
            lora_key_to_module_path(key),
            "model.layers.3.self_attn.q_proj",
        )

    def test_lora_delta_is_materialized_on_target_device(self):
        try:
            import torch
        except ModuleNotFoundError:
            self.skipTest("torch not installed in this test environment")

        if not torch.cuda.is_available():
            self.skipTest("CUDA required for device-transfer regression test")
        a = torch.ones((2, 3))
        b = torch.ones((4, 2))
        delta = lora_delta(b, a, 0.5, torch.device("cuda:0"))
        self.assertEqual(delta.device.type, "cuda")
        self.assertTrue(torch.equal(delta.cpu(), torch.full((4, 3), 1.0)))


if __name__ == "__main__":
    unittest.main()
