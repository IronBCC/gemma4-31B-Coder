from __future__ import annotations

import io
import unittest
from contextlib import redirect_stdout

from phaseD_sft.scan_agentic_dataset import run_scan


class ScanAgenticDatasetTests(unittest.TestCase):
    def test_progress_reaches_total_when_row_is_invalid(self) -> None:
        output = io.StringIO()
        with redirect_stdout(output):
            report = run_scan([{"source": "broken", "messages": []}], "broken", progress_every=1)

        self.assertEqual(report["failure_count"], 1)
        self.assertIn("scan-agentic-dataset 1/1", output.getvalue())

    def test_empty_assistant_is_counted_as_a_failure(self) -> None:
        row = {"source": "bad", "messages": [{"role": "assistant", "content": "", "tool_calls": []}]}

        report = run_scan([row], "bad", progress_every=1)

        self.assertEqual(report["failure_count"], 1)
        self.assertIn("assistant empty", report["failures"][0])


if __name__ == "__main__":
    unittest.main()
