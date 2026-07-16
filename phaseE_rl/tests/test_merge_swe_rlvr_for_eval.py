import sys
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from phaseE_rl.merge_swe_rlvr_for_eval import (
    build_arg_parser,
    lora_delta,
    lora_key_to_module_path,
    merge_adapters,
    resolve_adapter_paths,
)


class LoraKeyPathTests(unittest.TestCase):
    def test_merge_parser_accepts_legacy_two_adapter_invocation(self):
        args = build_arg_parser().parse_args([
            "--sft-adapter", "sft", "--rl-adapter", "rl", "--out", "out",
            "--max-shard-size", "20GB",
        ])
        self.assertEqual(args.max_shard_size, "20GB")
        self.assertEqual(resolve_adapter_paths(args, build_arg_parser()), [
            Path("sft"), Path("rl"),
        ])

    def test_merge_parser_preserves_repeatable_adapter_order(self):
        parser = build_arg_parser()
        args = parser.parse_args([
            "--adapter", "sft", "--adapter", "grpo", "--adapter", "vgrpo",
            "--out", "out",
        ])
        self.assertEqual(resolve_adapter_paths(args, parser), [
            Path("sft"), Path("grpo"), Path("vgrpo"),
        ])

    def test_merge_adapters_applies_three_adapters_in_cli_order(self):
        ordered = [Path("sft"), Path("grpo"), Path("vgrpo")]
        with patch("phaseE_rl.merge_swe_rlvr_for_eval.merge_adapter",
                   side_effect=[410, 411, 412]) as merged:
            result = merge_adapters(object(), ordered)
        self.assertEqual([call.args[1] for call in merged.call_args_list], ordered)
        self.assertEqual(result, [(Path("sft"), 410), (Path("grpo"), 411),
                                  (Path("vgrpo"), 412)])

    def test_merge_adapters_rejects_an_undermerged_adapter(self):
        with patch("phaseE_rl.merge_swe_rlvr_for_eval.merge_adapter",
                   side_effect=[410, 100]):
            with self.assertRaisesRegex(RuntimeError, "only 100 matrices"):
                merge_adapters(object(), [Path("sft"), Path("broken")])

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
