import json
import tempfile
import unittest
from pathlib import Path

from phaseD_sft.prune_loops import classify_event, prune_jsonl, prune_row


def obs(text: str) -> dict:
    return {"role": "user", "content": "OBSERVATION:\n" + text}


class LoopPruningTests(unittest.TestCase):
    def test_prune_row_cuts_terminal_readonly_spam(self):
        messages = [
            {"role": "system", "content": "system"},
            {"role": "user", "content": "fix bug"},
            {"role": "assistant", "content": "I edited the implementation."},
            obs("The file /testbed/pkg/mod.py has been edited."),
        ]
        for _ in range(6):
            messages.extend(
                [
                    {"role": "assistant", "content": "Let's list the repository files again."},
                    obs("/testbed/pkg/mod.py\n/testbed/tests/test_mod.py"),
                ]
            )
        row = {"id": "spam", "messages": messages}

        pruned, stats = prune_row(row)

        self.assertEqual(len(pruned["messages"]), 4)
        self.assertEqual(pruned["messages"][-1]["content"], messages[3]["content"])
        self.assertEqual(stats["rows_pruned"], 1)
        self.assertEqual(stats["messages_pruned"], 12)
        self.assertEqual(stats["loop:no-mutation"], 1)

    def test_prune_row_preserves_failure_recovery_arc(self):
        messages = [
            {"role": "system", "content": "system"},
            {"role": "user", "content": "fix bug"},
        ]
        for _ in range(3):
            messages.extend(
                [
                    {"role": "assistant", "content": "Let's run pytest again."},
                    obs("FAILED tests/test_mod.py::test_case - AssertionError: bad value"),
                ]
            )
        messages.extend(
            [
                {"role": "assistant", "content": "I found the issue and edited the implementation."},
                obs("The file /testbed/pkg/mod.py has been edited."),
                {"role": "assistant", "content": "Let's run pytest after the fix."},
                obs("1 passed in 0.13s"),
            ]
        )
        row = {"id": "recovery", "messages": messages}

        pruned, stats = prune_row(row)

        self.assertEqual(pruned["messages"], messages)
        self.assertEqual(stats["rows_pruned"], 0)
        self.assertEqual(stats["messages_pruned"], 0)

    def test_classify_event_detects_edit_read_and_verify(self):
        self.assertEqual(
            classify_event("Let's inspect the file.", "OBSERVATION:\n/testbed/pkg/mod.py").tool_name,
            "read_file",
        )
        self.assertEqual(
            classify_event("I edited the file.", "OBSERVATION:\nThe file /testbed/pkg/mod.py has been edited.").tool_name,
            "edit",
        )
        verify = classify_event("Let's run pytest.", "OBSERVATION:\nFAILED tests/test_mod.py::test_case")
        self.assertEqual(verify.tool_name, "run_tests")
        self.assertTrue(verify.feedback_signature)

    def test_prune_jsonl_writes_manifest_counts(self):
        messages = [
            {"role": "system", "content": "system"},
            {"role": "user", "content": "fix bug"},
        ]
        for _ in range(6):
            messages.extend(
                [
                    {"role": "assistant", "content": "Let's cat the same file."},
                    obs("same file content"),
                ]
            )
        row = {"id": "spam", "messages": messages}

        with tempfile.TemporaryDirectory() as td:
            source = Path(td) / "source.jsonl"
            output = Path(td) / "pruned.jsonl"
            source.write_text(json.dumps(row) + "\n{bad json\n", encoding="utf-8")

            manifest = prune_jsonl(source, output)
            written = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]

        self.assertEqual(len(written), 1)
        self.assertEqual(manifest["rows_read"], 1)
        self.assertEqual(manifest["malformed_rows"], 1)
        self.assertEqual(manifest["rows_pruned"], 1)
        self.assertGreater(manifest["messages_pruned"], 0)


if __name__ == "__main__":
    unittest.main()
