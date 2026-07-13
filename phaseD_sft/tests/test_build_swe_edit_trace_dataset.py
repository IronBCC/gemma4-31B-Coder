import json
import unittest

from phaseD_sft.build_swe_edit_trace_dataset import (
    build_rows,
    patch_paths,
    row_from_swe_instance,
    verify_command,
)


PATCH = """diff --git a/pkg/mod.py b/pkg/mod.py
--- a/pkg/mod.py
+++ b/pkg/mod.py
@@ -1,2 +1,2 @@
-old = 1
+new = 1
"""


class SweEditTraceDatasetTests(unittest.TestCase):
    def test_patch_paths_dedupes_diff_paths(self):
        self.assertEqual(patch_paths(PATCH), ["pkg/mod.py"])

    def test_verify_command_prefers_fail_to_pass_tests(self):
        row = {
            "FAIL_TO_PASS": json.dumps(["tests/test_mod.py::test_fix", "tests/test_mod.py::test_other"]),
            "PASS_TO_PASS": json.dumps(["tests/test_mod.py::test_existing"]),
            "patch": PATCH,
        }

        self.assertEqual(
            verify_command(row, max_tests=1),
            "python -m pytest -q tests/test_mod.py::test_fix",
        )

    def test_row_is_edit_first_agent_trace_without_tool_role(self):
        row = {
            "repo": "org/pkg",
            "instance_id": "org__pkg-1",
            "problem_statement": "Something is broken.",
            "patch": PATCH,
            "FAIL_TO_PASS": json.dumps(["tests/test_mod.py::test_fix"]),
            "PASS_TO_PASS": "[]",
        }

        built = row_from_swe_instance(
            row,
            repeat=0,
            max_problem_chars=1000,
            max_patch_chars=2000,
            max_tests=3,
        )

        self.assertIsNotNone(built)
        assert built is not None
        messages = built["messages"]
        self.assertNotIn("tool", {message["role"] for message in messages})
        commands = [
            json.loads(message["tool_calls"][0]["function"]["arguments"])["command"]
            for message in messages
            if message["role"] == "assistant"
        ]
        self.assertEqual(commands[0], "sed -n '1,220p' pkg/mod.py")
        self.assertTrue(commands[1].startswith("git apply <<'PATCH'"))
        self.assertEqual(commands[-1], "echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT && cat patch.txt")

    def test_build_rows_respects_limit_and_repeat(self):
        swe_rows = [
            {
                "repo": "org/pkg",
                "instance_id": f"org__pkg-{index}",
                "problem_statement": "Broken.",
                "patch": PATCH,
                "FAIL_TO_PASS": "[]",
                "PASS_TO_PASS": "[]",
            }
            for index in range(3)
        ]

        rows = build_rows(
            swe_rows,
            repeat=2,
            limit=2,
            max_problem_chars=1000,
            max_patch_chars=2000,
            max_tests=3,
        )

        self.assertEqual(len(rows), 4)


if __name__ == "__main__":
    unittest.main()
