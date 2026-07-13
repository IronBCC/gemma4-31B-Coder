import json
import tempfile
import unittest
from pathlib import Path

from phaseD_sft.compaction_experiments import (
    assistant_fingerprint,
    split_long_rows,
    transform_observation_cap,
    validate_jsonl,
)


def row_with_long_observation() -> dict:
    return {
        "id": "trace-1",
        "messages": [
            {"role": "system", "content": "system prompt"},
            {"role": "user", "content": "fix this bug"},
            {"role": "assistant", "content": "I will inspect the file."},
            {"role": "user", "content": "OBSERVATION:\n" + "\n".join(f"line {idx}" for idx in range(80))},
            {"role": "assistant", "content": "I found the issue."},
        ],
    }


class CompactionExperimentTests(unittest.TestCase):
    def test_observation_cap_preserves_assistant_fingerprint(self):
        row = row_with_long_observation()

        compacted, stats = transform_observation_cap(row, max_chars=160)

        self.assertEqual(assistant_fingerprint([row]), assistant_fingerprint([compacted]))
        self.assertLess(len(compacted["messages"][3]["content"]), len(row["messages"][3]["content"]))
        self.assertEqual(stats["assistant_changed"], 0)

    def test_split_long_rows_keeps_preamble_and_assistant_text(self):
        row = {
            "id": "trace-1",
            "messages": [
                {"role": "system", "content": "system prompt"},
                {"role": "user", "content": "fix this bug"},
            ]
            + [
                message
                for idx in range(10)
                for message in (
                    {"role": "assistant", "content": f"assistant step {idx} " + ("x" * 120)},
                    {"role": "user", "content": "OBSERVATION:\n" + ("y" * 120)},
                )
            ],
        }

        rows, stats = split_long_rows(row, max_chars=700)

        self.assertGreater(len(rows), 1)
        self.assertTrue(all(split["messages"][0]["role"] == "system" for split in rows))
        self.assertTrue(all(split["messages"][1]["content"] == "fix this bug" for split in rows))
        joined_assistant = "\n".join(
            message["content"]
            for split in rows
            for message in split["messages"]
            if message.get("role") == "assistant"
        )
        self.assertIn("assistant step 0", joined_assistant)
        self.assertIn("assistant step 9", joined_assistant)
        self.assertEqual(stats["rows_split"], 1)

    def test_split_long_rows_does_not_duplicate_early_assistant(self):
        row = {
            "id": "trace-1",
            "messages": [
                {"role": "user", "content": "fix this bug"},
                {"role": "assistant", "content": "early assistant " + ("x" * 300)},
                {"role": "user", "content": "OBSERVATION:\n" + ("y" * 300)},
                {"role": "assistant", "content": "later assistant " + ("z" * 300)},
                {"role": "user", "content": "OBSERVATION:\n" + ("w" * 300)},
            ],
        }

        rows, stats = split_long_rows(row, max_chars=700)

        self.assertGreater(len(rows), 1)
        assistant_texts = [
            message["content"]
            for split in rows
            for message in split["messages"]
            if message.get("role") == "assistant"
        ]
        self.assertEqual(assistant_texts.count(row["messages"][1]["content"]), 1)
        self.assertEqual(assistant_texts.count(row["messages"][3]["content"]), 1)
        self.assertEqual(stats["rows_split"], 1)

    def test_split_long_rows_never_starts_chunk_with_observation(self):
        row = {
            "id": "trace-1",
            "messages": [
                {"role": "system", "content": "system prompt"},
                {"role": "user", "content": "fix this bug"},
            ]
            + [
                message
                for idx in range(6)
                for message in (
                    {"role": "assistant", "content": f"assistant step {idx} " + ("x" * 180)},
                    {"role": "user", "content": "OBSERVATION:\n" + ("y" * 180)},
                )
            ],
        }

        rows, _ = split_long_rows(row, max_chars=700)

        self.assertGreater(len(rows), 1)
        for split in rows:
            body = split["messages"][2:]
            self.assertEqual(body[0]["role"], "assistant")

    def test_validate_jsonl_reports_rows_and_assistant_fingerprint(self):
        row = row_with_long_observation()

        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "rows.jsonl"
            path.write_text(json.dumps(row) + "\n", encoding="utf-8")

            report = validate_jsonl(path)

        self.assertEqual(report["rows"], 1)
        self.assertEqual(report["malformed_rows"], 0)
        self.assertEqual(report["assistant_messages"], 2)
        self.assertTrue(report["assistant_sha256"])


if __name__ == "__main__":
    unittest.main()
