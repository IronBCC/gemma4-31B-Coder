from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from phaseH_eval.summarize_smoke import summarize_run


def _assistant(command: str) -> dict[str, object]:
    return {
        "role": "assistant",
        "tool_calls": [
            {
                "function": {
                    "name": "bash",
                    "arguments": json.dumps({"command": command}),
                }
            }
        ],
    }


class SummarizeSmokeTests(unittest.TestCase):
    def test_reports_patch_format_and_edit_metrics_from_raw_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp) / "model"
            (run / "case-edit").mkdir(parents=True)
            (run / "case-no-edit").mkdir()
            (run / "case-edit" / "case-edit.traj.json").write_text(
                json.dumps(
                    {
                        "instance_id": "case-edit",
                        "messages": [_assistant("rg target src"), _assistant("sed -i 's/a/b/' src/a.py")],
                        "info": {"exit_status": "Submitted", "submission": "diff --git a/src/a.py b/src/a.py\n"},
                    }
                )
            )
            (run / "case-no-edit" / "case-no-edit.traj.json").write_text(
                json.dumps(
                    {
                        "instance_id": "case-no-edit",
                        "messages": [_assistant("rg target src"), _assistant("sed -n '1,20p' src/a.py")],
                        "info": {"exit_status": "Submitted", "submission": ""},
                    }
                )
            )
            (run / "exit_statuses_test.yaml").write_text(
                "instances_by_exit_status:\n"
                "    RepeatedFormatError:\n"
                "    - case-format\n"
                "    Submitted:\n"
                "    - case-edit\n"
                "    - case-no-edit\n"
            )

            metrics = summarize_run(run)

        self.assertEqual(metrics["attempted"], 3)
        self.assertEqual(metrics["non_empty_patches"], 1)
        self.assertEqual(metrics["format_errors"], 1)
        self.assertEqual(metrics["edit_reach"], 1)
        self.assertEqual(metrics["first_edit_median"], 2)
        self.assertAlmostEqual(metrics["format_error_rate"], 1 / 3)

    def test_uses_predictions_as_the_patch_gate_source_when_present(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp) / "model"
            (run / "case").mkdir(parents=True)
            (run / "case" / "case.traj.json").write_text(
                json.dumps(
                    {
                        "instance_id": "case",
                        "messages": [],
                        "info": {"exit_status": "Submitted", "submission": "diff --git a/a b/a\n"},
                    }
                )
            )
            (run / "preds.json").write_text(
                json.dumps(
                    {
                        "case": {
                            "instance_id": "case",
                            "model_patch": "",
                        }
                    }
                )
            )

            metrics = summarize_run(run)

        self.assertEqual(metrics["non_empty_patches"], 0)

    def test_reads_resolved_count_from_harness_final_report(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp) / "model"
            run.mkdir()
            (Path(tmp) / "openai__model.smoke_model.json").write_text(
                json.dumps({"resolved_instances": 7})
            )

            metrics = summarize_run(run)

        self.assertEqual(metrics["resolved"], 7)

    def test_finds_report_written_at_repository_root(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "repo"
            run = root / "runs" / "smoke_model" / "model"
            run.mkdir(parents=True)
            (root / "openai__model.smoke_model.json").write_text(
                json.dumps({"resolved_instances": 8})
            )

            metrics = summarize_run(run)

        self.assertEqual(metrics["resolved"], 8)


if __name__ == "__main__":
    unittest.main()
