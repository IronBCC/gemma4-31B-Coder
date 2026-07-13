import unittest
from pathlib import Path

from phaseD_sft.train_rust_lora import (
    configure_attention_implementation,
    label_tokens_for_spans,
    rendered_assistant_turn_spans,
)


class TrainRustLoraLossMaskTests(unittest.TestCase):
    def test_trainer_emits_microstep_step_and_eta_progress(self):
        source = (Path(__file__).resolve().parents[1] / "train_rust_lora.py").read_text(encoding="utf-8")

        self.assertIn("class TrainingProgressCallback(TrainerCallback):", source)
        self.assertIn("def on_substep_end", source)
        self.assertIn("def on_step_end", source)
        self.assertIn("event=optimizer_step", source)
        self.assertIn("eta_seconds=", source)
        self.assertIn("epoch_seconds=", source)
        self.assertIn("event=data_ready", source)

    def test_hf_flex_routing_debug_is_explicit_opt_in(self):
        source = (Path(__file__).resolve().parents[1] / "train_rust_lora.py").read_text(encoding="utf-8")

        self.assertIn("def install_hf_flex_routing_debug", source)
        self.assertIn("torch.compiler.disable(recursive=False)", source)
        self.assertIn("WrappedFlexAttention._is_flex_compiled", source)
        self.assertIn('"--debug-hf-flex-routing"', source)

    def test_configure_attention_implementation_updates_root_and_nested_configs(self):
        class Config:
            def __init__(self):
                self._attn_implementation = "sdpa"

        class Model:
            def __init__(self):
                self.config = Config()
                self.config.text_config = Config()
                self.base_model = type("Base", (), {"config": Config()})()

        model = Model()

        changed = configure_attention_implementation(model, "flex_attention")

        self.assertEqual(changed, 3)
        self.assertEqual(model.config._attn_implementation, "flex_attention")
        self.assertEqual(model.config.text_config._attn_implementation, "flex_attention")
        self.assertEqual(model.base_model.config._attn_implementation, "flex_attention")

    def test_configure_attention_implementation_is_noop_when_not_requested(self):
        model = type("Model", (), {"config": object()})()

        self.assertEqual(configure_attention_implementation(model, None), 0)

    def test_assistant_turn_span_stops_before_tool_observation_turn(self):
        messages = [
            {"role": "user", "content": "run tests"},
            {"role": "assistant", "content": "calling pytest"},
            {"role": "tool", "content": "FAILED tests/test_mod.py::test_case"},
            {"role": "user", "content": "fix it"},
        ]
        text = (
            "<bos><|turn>user\nrun tests"
            "<|turn>model\ncalling pytest<|tool_call>{}</tool_call>"
            "<|turn>tool\nFAILED tests/test_mod.py::test_case"
            "<|turn>user\nfix it"
        )

        spans, fallback_used = rendered_assistant_turn_spans(object(), messages, text)

        self.assertFalse(fallback_used)
        self.assertEqual(spans, [(len("<bos><|turn>user\nrun tests"), text.index("<|turn>tool\n"))])

    def test_label_tokens_for_spans_masks_tokens_outside_assistant_spans(self):
        input_ids = [10, 11, 12, 13]
        offsets = [(0, 4), (4, 8), (8, 12), (12, 16)]

        labels, supervised = label_tokens_for_spans(input_ids, offsets, [(4, 12)])

        self.assertEqual(labels, [-100, 11, 12, -100])
        self.assertEqual(supervised, 2)


if __name__ == "__main__":
    unittest.main()
