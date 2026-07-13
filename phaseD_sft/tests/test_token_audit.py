import json
import tempfile
import unittest
from pathlib import Path

from phaseD_sft.token_audit import audit_jsonl, message_text, percentile


def char_tokens(text: str) -> int:
    return len(text)


class TokenAuditTests(unittest.TestCase):
    def test_message_text_includes_tool_calls_as_assistant_tokens(self):
        message = {
            "role": "assistant",
            "content": "checking files",
            "tool_calls": [
                {
                    "type": "function",
                    "function": {
                        "name": "exec_command",
                        "arguments": {"cmd": "pytest -q"},
                    },
                }
            ],
        }

        text = message_text(message)

        self.assertIn("checking files", text)
        self.assertIn("exec_command", text)
        self.assertIn("pytest -q", text)

    def test_audit_jsonl_splits_tokens_by_role(self):
        rows = [
            {
                "id": "short",
                "messages": [
                    {"role": "system", "content": "ss"},
                    {"role": "user", "content": "uuuu"},
                    {"role": "assistant", "content": "aaaaaa"},
                    {"role": "tool", "content": "tttttttt"},
                ],
            },
            {
                "id": "long",
                "messages": [
                    {"role": "user", "content": "uuuuuu"},
                    {"role": "assistant", "content": "aaaaaaaaaa"},
                    {"role": "user", "content": "OBSERVATION:\ntttt"},
                ],
            },
        ]

        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "traces.jsonl"
            path.write_text("\n".join(json.dumps(row) for row in rows) + "\n{bad json\n", encoding="utf-8")

            report = audit_jsonl(path, char_tokens, thresholds=(10, 20))

        self.assertEqual(report["rows"], 2)
        self.assertEqual(report["malformed_rows"], 1)
        self.assertEqual(report["total_tokens"]["p50"], 20)
        self.assertEqual(report["total_tokens"]["p95"], 33)
        self.assertEqual(report["roles"]["assistant"]["total"], 16)
        self.assertEqual(report["roles"]["tool"]["total"], 25)
        self.assertEqual(report["roles"]["user"]["total"], 10)
        self.assertEqual(report["roles"]["system"]["total"], 2)
        self.assertEqual(report["thresholds"]["10"]["over"], 2)
        self.assertEqual(report["thresholds"]["20"]["over"], 1)

    def test_percentile_uses_nearest_rank(self):
        self.assertEqual(percentile([1, 10, 20, 100], 50), 10)
        self.assertEqual(percentile([1, 10, 20, 100], 95), 100)
        self.assertEqual(percentile([], 95), 0)


if __name__ == "__main__":
    unittest.main()
