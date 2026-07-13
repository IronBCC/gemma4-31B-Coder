import json
import tempfile
import unittest
from pathlib import Path

from phaseD_sft.compact_observations import compact_jsonl, compact_observation_text, compact_row


class ObservationCompactionTests(unittest.TestCase):
    def test_compact_observation_text_keeps_head_important_tail(self):
        lines = ["OBSERVATION:"] + [f"line {idx}" for idx in range(30)]
        lines.insert(18, "ERROR: failed to compile target")
        text = "\n".join(lines)

        compacted, changed, stats = compact_observation_text(
            text,
            max_chars=120,
            head_lines=3,
            tail_lines=2,
            important_lines=5,
        )

        self.assertTrue(changed)
        self.assertIn("OBSERVATION:", compacted)
        self.assertIn("line 0", compacted)
        self.assertIn("ERROR: failed to compile target", compacted)
        self.assertIn("line 29", compacted)
        self.assertIn("[... omitted", compacted)
        self.assertGreater(stats["omitted_lines"], 0)

    def test_compact_observation_text_strips_base64(self):
        blob = "A" * 260
        text = f"OBSERVATION:\nimage/png;base64,{blob}\nfinished"

        compacted, changed, stats = compact_observation_text(text, max_chars=500)

        self.assertTrue(changed)
        self.assertIn("[base64 omitted:", compacted)
        self.assertNotIn(blob, compacted)
        self.assertEqual(stats["base64_replacements"], 1)

    def test_compact_row_preserves_assistant_and_dedups_repeated_observations(self):
        repeated = "OBSERVATION:\n" + "\n".join(f"same line {idx}" for idx in range(20))
        row = {
            "id": "trace-1",
            "messages": [
                {"role": "user", "content": "fix this bug"},
                {"role": "assistant", "content": "I will inspect the file."},
                {"role": "user", "content": repeated},
                {"role": "assistant", "content": "I will inspect it again."},
                {"role": "user", "content": repeated},
            ],
        }

        compacted, stats = compact_row(row, max_chars=120, head_lines=3, tail_lines=2)

        self.assertEqual(compacted["messages"][1]["content"], "I will inspect the file.")
        self.assertEqual(compacted["messages"][3]["content"], "I will inspect it again.")
        self.assertIn("same line 0", compacted["messages"][2]["content"])
        self.assertIn("[unchanged observation omitted;", compacted["messages"][4]["content"])
        self.assertEqual(stats["duplicate_observations"], 1)
        self.assertEqual(stats["assistant_messages_changed"], 0)

    def test_compact_jsonl_writes_manifest_counts(self):
        row = {
            "id": "trace-1",
            "messages": [
                {"role": "assistant", "content": "unchanged"},
                {"role": "user", "content": "OBSERVATION:\n" + "\n".join(f"line {idx}" for idx in range(40))},
            ],
        }

        with tempfile.TemporaryDirectory() as td:
            source = Path(td) / "source.jsonl"
            output = Path(td) / "compact.jsonl"
            source.write_text(json.dumps(row) + "\n{bad json\n", encoding="utf-8")

            manifest = compact_jsonl(source, output, max_chars=120, head_lines=3, tail_lines=2)

            written = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]

        self.assertEqual(len(written), 1)
        self.assertEqual(manifest["rows_read"], 1)
        self.assertEqual(manifest["malformed_rows"], 1)
        self.assertEqual(manifest["observation_messages_compacted"], 1)
        self.assertLess(manifest["compacted_chars"], manifest["original_chars"])


class CompactionHardCapTests(unittest.TestCase):
    def test_many_important_lines_observation_respects_max_chars(self):
        import re as _re

        from phaseD_sft.compact_observations import (
            compact_observation_text,
            _IMPORTANT_RE,
        )

        error_lines = [f"error line {i}: something failed here" for i in range(30)]
        text = "OBSERVATION:\n" + "\n".join(error_lines)

        compacted, changed, stats = compact_observation_text(
            text, max_chars=200, head_lines=4, tail_lines=2, important_lines=15
        )

        self.assertTrue(changed)
        self.assertLessEqual(len(compacted), 200)


if __name__ == "__main__":
    unittest.main()
