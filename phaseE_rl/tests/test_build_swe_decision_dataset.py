import json
from pathlib import Path
import tempfile
import unittest

from phaseE_rl.build_swe_decision_dataset import (
    Candidate,
    _rendered_tokens,
    extract_edit_adjacent_point,
    extract_decision_point,
    load_sft_rows,
    parse_source_caps,
    select_source_balanced,
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
    def test_parse_source_caps_accepts_all_and_rejects_invalid_values(self):
        self.assertEqual(
            parse_source_caps(["swe-smith=700", "smoke=all", "openswe=300"]),
            {"swe-smith": 700, "smoke": None, "openswe": 300},
        )
        with self.assertRaisesRegex(ValueError, "positive integer or 'all'"):
            parse_source_caps(["oracle=0"])

    def test_source_balanced_selection_enforces_caps_and_excludes_unknown_sources(self):
        def candidate(source: str, instance_id: str) -> Candidate:
            return Candidate(
                {"source": source, "instance_id": instance_id, "messages": []},
                "smoke" if source == "smoke:run" else "sft",
            )

        accepted = [
            (candidate("swe-smith", "long"), 20),
            (candidate("swe-smith", "short"), 10),
            (candidate("swe_train_oracle_edit_trace", "oracle"), 12),
            (candidate("open_swe_traces_qwen35", "open"), 11),
            (candidate("kwai_klear_miniswe", "kwai"), 13),
            (candidate("smoke:run", "smoke"), 30),
            (candidate("coder_repair_synthetic", "excluded"), 1),
        ]

        selected, counts = select_source_balanced(
            accepted,
            source_caps={"swe-smith": 1, "oracle": 1, "openswe": 1, "kwai": 1, "smoke": None},
            include_only_capped_sources=True,
        )

        self.assertEqual(
            [(item[0].row["source"], item[0].row["instance_id"]) for item in selected],
            [
                ("swe-smith", "short"),
                ("swe_train_oracle_edit_trace", "oracle"),
                ("open_swe_traces_qwen35", "open"),
                ("kwai_klear_miniswe", "kwai"),
                ("smoke:run", "smoke"),
            ],
        )
        self.assertEqual(counts, {"kwai": 1, "openswe": 1, "oracle": 1, "smoke": 1, "swe-smith": 1})

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

    def test_edit_adjacent_prefix_stops_before_command_preceding_first_edit(self):
        messages = [
            {"role": "system", "content": "fix it"},
            {"role": "user", "content": "task"},
            assistant("find . -name mod.py"),
            tool("./pkg/mod.py"),
            assistant("sed -n '1,20p' pkg/mod.py"),
            tool("pkg/mod.py contains the target"),
            assistant("apply_patch <<'PATCH'\n*** End Patch\nPATCH"),
        ]

        point = extract_edit_adjacent_point(messages)

        self.assertIsNotNone(point)
        assert point is not None
        self.assertTrue(point.had_edit)
        self.assertEqual(point.trigger, "one_command_before_first_edit")
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
