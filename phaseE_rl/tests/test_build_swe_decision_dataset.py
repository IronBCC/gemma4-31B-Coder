import json
from pathlib import Path
import tempfile
import unittest

from phaseE_rl.build_swe_decision_dataset import (
    _rendered_tokens,
    extract_decision_point,
    load_sft_rows,
)


def assistant(command: str) -> dict:
    return {
        "role": "assistant",
        "content": "",
        "tool_calls": [{"function": {"name": "bash", "arguments": json.dumps({"command": command})}}],
    }


def tool(content: str) -> dict:
    return {"role": "tool", "content": content}


class DecisionPointExtractionTests(unittest.TestCase):
    def test_load_sft_rows_accepts_jsonl_fallback(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "rows.jsonl"
            path.write_text('{"source":"x","messages":[]}\n', encoding="utf-8")
            self.assertEqual(load_sft_rows(path), [{"source": "x", "messages": []}])

    def test_rendered_tokens_counts_input_ids_not_encoding_mapping_keys(self):
        class Tokenizer:
            def apply_chat_template(self, messages, *, tokenize, add_generation_prompt):
                self.assertion = (tokenize, add_generation_prompt)
                return "rendered-prefix"

            def __call__(self, text, *, add_special_tokens):
                self.text = text
                self.add_special_tokens = add_special_tokens
                return {"input_ids": [1, 2, 3, 4]}

        tokenizer = Tokenizer()
        self.assertEqual(_rendered_tokens(tokenizer, [{"role": "user", "content": "x"}]), 4)
        self.assertEqual(tokenizer.assertion, (False, True))
        self.assertEqual(tokenizer.text, "rendered-prefix")
        self.assertFalse(tokenizer.add_special_tokens)

    def test_edit_prefix_excludes_the_first_edit_assistant_turn(self):
        messages = [
            {"role": "system", "content": "fix it"},
            {"role": "user", "content": "task"},
            assistant("sed -n '1,20p' pkg/mod.py"),
            tool("pkg/mod.py contains the target"),
            assistant("apply_patch <<'PATCH'\n*** End Patch\nPATCH"),
        ]

        point = extract_decision_point(messages)

        self.assertIsNotNone(point)
        assert point is not None
        self.assertTrue(point.had_edit)
        self.assertEqual(point.command_index, 2)
        self.assertEqual(point.messages, messages[:4])
        self.assertEqual(point.edit_files, ["pkg/mod.py"])

    def test_no_edit_prefix_stops_before_first_repeated_read(self):
        messages = [
            {"role": "user", "content": "task"},
            assistant("sed -n '1,20p' pkg/mod.py"),
            tool("pkg/mod.py"),
            assistant("sed -n '1,20p' pkg/mod.py"),
        ]

        point = extract_decision_point(messages)

        self.assertIsNotNone(point)
        assert point is not None
        self.assertFalse(point.had_edit)
        self.assertEqual(point.trigger, "repeated_command")
        self.assertEqual(point.messages, messages[:3])

    def test_no_edit_prefix_stops_before_fourth_read(self):
        messages = [{"role": "user", "content": "task"}]
        for command in ("cat a.py", "find . -name b.py", "grep x c.py", "rg y d.py"):
            messages.extend([assistant(command), tool("pkg/mod.py")])

        point = extract_decision_point(messages)

        self.assertIsNotNone(point)
        assert point is not None
        self.assertFalse(point.had_edit)
        self.assertEqual(point.trigger, "read_streak_4")
        self.assertEqual(point.messages, messages[:7])
