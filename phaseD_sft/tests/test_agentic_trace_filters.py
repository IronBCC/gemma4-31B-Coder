import json
import unittest

from phaseD_sft.agentic_trace_filters import (
    command_trace_quality_report,
    trace_quality_report,
    trace_should_keep,
)


def obs(text: str) -> dict:
    return {"role": "tool", "content": "OBSERVATION:\n" + text}


def command(command: str) -> dict:
    return {
        "role": "assistant",
        "content": "",
        "tool_calls": [
            {
                "function": {
                    "name": "bash",
                    "arguments": json.dumps({"command": command}),
                }
            }
        ],
    }


class AgenticTraceFilterTests(unittest.TestCase):
    def test_keeps_edit_first_trace_with_verify_tail(self):
        messages = [
            {"role": "assistant", "content": "Let's inspect the file."},
            obs("/testbed/pkg/mod.py"),
            {"role": "assistant", "content": "I edited the implementation."},
            obs("The file /testbed/pkg/mod.py has been edited."),
            {"role": "assistant", "content": "Let's run pytest."},
            obs("1 passed in 0.13s"),
        ]

        report = trace_quality_report(messages)

        self.assertTrue(report.keep)
        self.assertTrue(trace_should_keep(messages))
        self.assertEqual(report.first_edit_index, 1)
        self.assertLessEqual(report.first_edit_ratio, 0.4)

    def test_rejects_traces_that_read_too_long_before_editing(self):
        messages = [{"role": "assistant", "content": "Let's inspect again."}, obs("/tmp/a.py")]
        for _ in range(5):
            messages.extend(
                [
                    {"role": "assistant", "content": "Let's cat the file again."},
                    obs("/tmp/a.py\n/tmp/b.py"),
                ]
            )
        messages.extend(
            [
                {"role": "assistant", "content": "I edited the implementation."},
                obs("The file /testbed/pkg/mod.py has been edited."),
                {"role": "assistant", "content": "Let's run pytest."},
                obs("1 passed in 0.13s"),
            ]
        )

        report = trace_quality_report(messages)

        self.assertFalse(report.keep)
        self.assertIn("late_first_edit", report.reasons)

    def test_rejects_traces_without_verify_tail(self):
        messages = [
            {"role": "assistant", "content": "Let's inspect the file."},
            obs("/testbed/pkg/mod.py"),
            {"role": "assistant", "content": "I edited the implementation."},
            obs("The file /testbed/pkg/mod.py has been edited."),
            {"role": "assistant", "content": "Let's inspect the file again."},
            obs("/testbed/pkg/mod.py"),
        ]

        report = trace_quality_report(messages)

        self.assertFalse(report.keep)
        self.assertIn("no_verify_tail", report.reasons)
        self.assertFalse(trace_should_keep(messages))

    def test_command_report_keeps_edit_by_tenth_command(self):
        report = command_trace_quality_report(
            [command("rg target src"), command("sed -i 's/old/new/' src/mod.py")],
            max_first_edit_index=10,
            max_read_streak=5,
            reject_identical_consecutive_commands=True,
        )

        self.assertTrue(report.keep)
        self.assertEqual(report.first_edit_index, 2)
        self.assertEqual(report.max_read_streak, 1)
        self.assertFalse(report.has_identical_consecutive_repeat)

    def test_command_report_rejects_edit_after_tenth_command(self):
        messages = [command(f"pytest -q tests/test_{index}.py") for index in range(10)]
        messages.append(command("apply_patch <<'PATCH'\n*** Begin Patch\n*** End Patch\nPATCH"))

        report = command_trace_quality_report(messages, max_first_edit_index=10)

        self.assertFalse(report.keep)
        self.assertEqual(report.first_edit_index, 11)
        self.assertIn("first_edit_after_limit", report.reasons)

    def test_command_report_rejects_read_streak_and_identical_repeat(self):
        messages = [command("rg target src")] * 2
        messages.extend(command(f"cat file_{index}.py") for index in range(4))
        messages.append(command("sed -i 's/old/new/' src/mod.py"))

        report = command_trace_quality_report(
            messages,
            max_first_edit_index=10,
            max_read_streak=5,
            reject_identical_consecutive_commands=True,
        )

        self.assertFalse(report.keep)
        self.assertEqual(report.max_read_streak, 6)
        self.assertTrue(report.has_identical_consecutive_repeat)
        self.assertIn("read_streak_exceeded", report.reasons)
        self.assertIn("identical_consecutive_command", report.reasons)


if __name__ == "__main__":
    unittest.main()
