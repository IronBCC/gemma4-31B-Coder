import json
import unittest

from phaseD_sft.build_agentic_behavior_dataset import behavior_cases, row_for_case


class AgenticBehaviorDatasetTests(unittest.TestCase):
    def test_all_assistant_tool_calls_have_command_argument(self):
        for case in behavior_cases():
            row = row_for_case(case, 0)
            for message in row["messages"]:
                if message["role"] != "assistant":
                    continue
                for tool_call in message.get("tool_calls") or []:
                    self.assertEqual(tool_call["function"]["name"], "bash")
                    arguments = json.loads(tool_call["function"]["arguments"])
                    self.assertEqual(sorted(arguments), ["command"])
                    self.assertIsInstance(arguments["command"], str)
                    self.assertTrue(arguments["command"].strip())

    def test_synthetic_rows_do_not_use_tool_role(self):
        for case in behavior_cases():
            row = row_for_case(case, 0)
            self.assertNotIn("tool", {message["role"] for message in row["messages"]})

    def test_contains_loop_break_edit_case(self):
        cases = {case.name: case for case in behavior_cases()}
        row = row_for_case(cases["loop_warning_repeated_read_make_edit"], 0)
        last = row["messages"][-1]
        command = json.loads(last["tool_calls"][0]["function"]["arguments"])["command"]

        self.assertEqual(last["role"], "assistant")
        self.assertIn("Path(", command)
        self.assertIn("write_text", command)

    def test_contains_submit_case(self):
        cases = {case.name: case for case in behavior_cases()}
        row = row_for_case(cases["after_nonempty_diff_submit"], 0)
        command = json.loads(row["messages"][-1]["tool_calls"][0]["function"]["arguments"])["command"]

        self.assertIn("COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT", command)


if __name__ == "__main__":
    unittest.main()
