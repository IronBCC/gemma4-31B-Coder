import ast
from pathlib import Path
import unittest

from phaseD_sft import verify_gemma_format_loss as verifier


class _FixtureTokenizer:
    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt):
        assert not tokenize
        assert not add_generation_prompt
        rendered = "<bos>"
        for message in messages:
            role = "model" if message["role"] == "assistant" else message["role"]
            rendered += f"<|turn>{role}\\n{message.get('content', '')}"
        return rendered

    def __call__(self, *, text, add_special_tokens, return_offsets_mapping, truncation):
        assert not add_special_tokens
        assert return_offsets_mapping
        assert truncation
        return {
            "input_ids": list(range(len(text))),
            "offset_mapping": [(index, index + 1) for index in range(len(text))],
        }

    def decode(self, token_ids):
        return "<|turn>model\\nanswer"


class VerifyGemmaFormatLossTests(unittest.TestCase):
    def test_format_only_verifier_does_not_import_unsloth(self):
        source = Path(__file__).parents[1] / "verify_gemma_format_loss.py"
        tree = ast.parse(source.read_text(encoding="utf-8"))
        imported = {
            alias.name.split(".")[0]
            for node in ast.walk(tree)
            if isinstance(node, ast.Import)
            for alias in node.names
        }
        imported.update(
            node.module.split(".")[0]
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom) and node.module
        )

        self.assertNotIn("unsloth", imported)

    def test_ordered_parallel_map_matches_serial_order(self):
        inputs = [7, 2, 9, 1, 4]

        self.assertEqual(
            list(verifier.ordered_parallel_map(inputs, lambda value: value * value, workers=3)),
            [49, 4, 81, 1, 16],
        )

    def test_validate_one_exercises_render_and_label_helpers(self):
        counts, supervised, failures, _ = verifier.validate_one(
            _FixtureTokenizer(),
            17,
            [
                {"role": "user", "content": "question"},
                {"role": "assistant", "content": "answer"},
            ],
        )

        self.assertEqual(counts["assistant_messages"], 1)
        self.assertGreater(supervised, 0)
        self.assertEqual(failures, [])


if __name__ == "__main__":
    unittest.main()
