import json
import unittest

from phaseD_sft.build_swe_trajectory_behavior_dataset import (
    LOOP_BREAK_COMMAND,
    extract_rows_from_trajectory,
)


def tool_call(command: str) -> dict:
    return {
        "id": "call_1",
        "type": "function",
        "function": {"name": "bash", "arguments": json.dumps({"command": command})},
    }


def assistant(command: str) -> dict:
    return {"role": "assistant", "content": "", "tool_calls": [tool_call(command)]}


def user(content: str) -> dict:
    return {"role": "user", "content": content}


def tool(content: str = "<returncode>0</returncode>\n<output></output>") -> dict:
    return {"role": "tool", "content": content, "tool_call_id": "call_1"}


class SweTrajectoryBehaviorDatasetTests(unittest.TestCase):
    def test_extracts_repeated_read_loop_to_diff_row(self):
        trajectory = {
            "instance_id": "pkg__pkg-1",
            "messages": [
                {"role": "system", "content": "fix"},
                user("Task"),
                *sum(([assistant("sed -n '1,20p' src/x.py"), tool()] for _ in range(5)), []),
                {"role": "exit", "content": "LimitsExceeded"},
            ],
        }

        rows = extract_rows_from_trajectory(trajectory, repeated_read_threshold=5)
        loop_rows = [row for row in rows if row.kind == "repeated_read_to_diff"]

        self.assertEqual(len(loop_rows), 1)
        self.assertEqual(loop_rows[0].messages[-1]["role"], "assistant")
        arguments = json.loads(loop_rows[0].messages[-1]["tool_calls"][0]["function"]["arguments"])
        self.assertEqual(arguments["command"], LOOP_BREAK_COMMAND)
        self.assertNotIn("tool", {message["role"] for message in loop_rows[0].messages})

    def test_extracts_success_trace_through_submit(self):
        trajectory = {
            "instance_id": "pkg__pkg-2",
            "messages": [
                {"role": "system", "content": "fix"},
                user("Task"),
                assistant("sed -i 's/a/b/' src/x.py"),
                tool("edited"),
                assistant("git diff -- src/x.py > patch.txt && cat patch.txt"),
                tool("diff --git a/src/x.py b/src/x.py"),
                assistant("echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT && cat patch.txt"),
                tool("diff --git a/src/x.py b/src/x.py"),
                {"role": "exit", "content": "Submitted"},
            ],
        }

        rows = extract_rows_from_trajectory(trajectory)
        success_rows = [row for row in rows if row.kind == "success_submit"]

        self.assertEqual(len(success_rows), 1)
        self.assertIn("COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT", json.dumps(success_rows[0].messages[-1]))
        self.assertEqual(success_rows[0].messages[-1]["role"], "assistant")

    def test_extracts_format_recovery_pair(self):
        trajectory = {
            "instance_id": "pkg__pkg-3",
            "messages": [
                user("Tool call error:\nNo tool calls found in the response."),
                assistant("find . -name '*.py'"),
            ],
        }

        rows = extract_rows_from_trajectory(trajectory)
        format_rows = [row for row in rows if row.kind == "format_recovery"]

        self.assertEqual(len(format_rows), 1)
        arguments = json.loads(format_rows[0].messages[-1]["tool_calls"][0]["function"]["arguments"])
        self.assertEqual(arguments["command"], "find . -name '*.py'")


if __name__ == "__main__":
    unittest.main()
