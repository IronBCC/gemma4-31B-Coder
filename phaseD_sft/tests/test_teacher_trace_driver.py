import unittest
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch

from phaseD_sft.teacher_trace_driver import (
    CREDIT_EXHAUSTED_ERROR,
    PROMPT,
    capture_worktree_patch,
    collect_one,
    credit_exhausted_in_stream,
    establish_mutation_baseline,
    next_credit_streak,
    remaining_task_rows,
    trailing_credit_streak,
)


class TeacherTraceCreditGuardTests(unittest.TestCase):
    def test_prompt_pins_the_mutated_head_as_the_task_baseline(self):
        self.assertIn(
            "bug-inducing mutation is already applied and committed as current HEAD",
            PROMPT,
        )
        self.assertIn("do not reset or check out its parent", PROMPT)

    def test_patch_capture_uses_temporary_index_and_includes_untracked_files(self):
        with patch(
            "phaseD_sft.teacher_trace_driver.sh",
            return_value=("diff --git a/new.py b/new.py\n", 0),
        ) as run:
            output = capture_worktree_patch("0123456789ab")

        self.assertIn("diff --git", output)
        command = run.call_args.args[0][-1]
        self.assertIn("GIT_INDEX_FILE", command)
        self.assertIn("git add -A -- .", command)
        self.assertIn("git diff --cached --binary --no-renames", command)

    def test_detects_credit_wall_even_when_cli_marks_result_success(self):
        stream = (
            '{"type":"result","subtype":"success",'
            '"result":"Error: out of usage credits. Try again later."}\n'
        )

        self.assertTrue(credit_exhausted_in_stream(stream))

    def test_credit_streak_resets_after_a_non_credit_result(self):
        self.assertEqual(next_credit_streak(2, CREDIT_EXHAUSTED_ERROR), 3)
        self.assertEqual(next_credit_streak(3, None), 0)

    def test_resume_skips_only_completed_instance_ids(self):
        rows = [{"instance_id": "a"}, {"instance_id": "b"}, {"instance_id": "c"}]
        existing = [{"instance_id": "a", "resolved": True}, {"instance_id": "c", "error": "x"}]

        self.assertEqual(remaining_task_rows(rows, existing), [{"instance_id": "b"}])

    def test_resume_restores_only_the_trailing_credit_streak(self):
        results = [
            {"error": CREDIT_EXHAUSTED_ERROR},
            {"error": None},
            {"error": CREDIT_EXHAUSTED_ERROR},
            {"error": CREDIT_EXHAUSTED_ERROR},
        ]

        self.assertEqual(trailing_credit_streak(results), 2)

    def test_establish_mutation_baseline_applies_reproduces_and_commits(self):
        row = {
            "instance_id": "org__repo.pr_1",
            "image_name": "fixture/image",
            "patch": "diff --git a/src/x.py b/src/x.py\n",
            "FAIL_TO_PASS": ["tests/test_fix.py::test_fix"],
            "PASS_TO_PASS": ["tests/test_fix.py::test_old"],
        }
        calls = []
        mutation_applied = False

        def fake_sh(command, timeout=300, input_text=None):
            nonlocal mutation_applied
            calls.append((command, timeout, input_text))
            if command[:4] == ["docker", "exec", "-i", "0123456789ab"]:
                if input_text == row["patch"]:
                    mutation_applied = True
                    return "", 0
                ids = json.loads(input_text)
                if ids == ["tests/test_fix.py::test_fix"] and mutation_applied:
                    return "F [100%]\n1 failed in 0.01s\n", 1
                self.assertIn(ids, (
                    ["tests/test_fix.py::test_fix"],
                    ["tests/test_fix.py::test_old"],
                ))
                return "1 passed in 0.01s\n", 0
            if command[:3] == ["docker", "exec", "0123456789ab"]:
                self.assertIn("commit --no-verify", command[-1])
                return "f" * 40 + "\n", 0
            raise AssertionError(command)

        with patch("phaseD_sft.teacher_trace_driver.sh", side_effect=fake_sh):
            evidence = establish_mutation_baseline("0123456789ab", row)

        self.assertTrue(evidence["mutation_f2p_reproduced"])
        self.assertTrue(evidence["reference_controls_passed"])
        self.assertTrue(evidence["mutation_p2p_passed"])
        self.assertEqual(evidence["mutation_baseline_commit"], "f" * 40)
        self.assertEqual(
            [call[2] for call in calls[:5]],
            [
                json.dumps(
                    ["tests/test_fix.py::test_fix"],
                    ensure_ascii=False,
                    separators=(",", ":"),
                ),
                json.dumps(
                    ["tests/test_fix.py::test_old"],
                    ensure_ascii=False,
                    separators=(",", ":"),
                ),
                row["patch"],
                json.dumps(
                    ["tests/test_fix.py::test_fix"],
                    ensure_ascii=False,
                    separators=(",", ":"),
                ),
                json.dumps(
                    ["tests/test_fix.py::test_old"],
                    ensure_ascii=False,
                    separators=(",", ":"),
                ),
            ],
        )

    def test_establish_mutation_baseline_rejects_clean_f2p(self):
        row = {
            "instance_id": "org__repo.pr_1",
            "image_name": "fixture/image",
            "patch": "diff --git a/src/x.py b/src/x.py\n",
            "FAIL_TO_PASS": ["tests/test_fix.py::test_fix"],
            "PASS_TO_PASS": ["tests/test_fix.py::test_old"],
        }

        def fake_sh(command, timeout=300, input_text=None):
            if input_text == row["patch"]:
                return "", 0
            if input_text is not None:
                return "1 passed in 0.01s\n", 0
            raise AssertionError("mutation baseline must not be committed")

        with (
            patch("phaseD_sft.teacher_trace_driver.sh", side_effect=fake_sh),
            self.assertRaisesRegex(ValueError, "did not reproduce"),
        ):
            establish_mutation_baseline("0123456789ab", row)

    def test_establish_mutation_baseline_rejects_invalid_clean_reference(self):
        row = {
            "instance_id": "org__repo.pr_1",
            "image_name": "fixture/image",
            "patch": "diff --git a/src/x.py b/src/x.py\n",
            "FAIL_TO_PASS": ["tests/test_fix.py::test_fix"],
            "PASS_TO_PASS": ["tests/test_fix.py::test_old"],
        }

        def fake_sh(_command, timeout=300, input_text=None):
            ids = json.loads(input_text)
            if ids == row["FAIL_TO_PASS"]:
                return "1 passed in 0.01s\n", 0
            return "F [100%]\n1 failed in 0.01s\n", 1

        with (
            patch("phaseD_sft.teacher_trace_driver.sh", side_effect=fake_sh),
            self.assertRaisesRegex(ValueError, "clean reference controls"),
        ):
            establish_mutation_baseline("0123456789ab", row)

    def test_collect_one_records_successful_execution_verification(self):
        row = {
            "instance_id": "org__repo.pr_1",
            "image_name": "fixture/image",
            "problem_statement": "Fix the bug.",
            "patch": "diff --git a/a.py b/a.py\n",
            "FAIL_TO_PASS": ["tests/test_fix.py::test_fix"],
            "PASS_TO_PASS": ["tests/test_fix.py::test_old"],
        }

        events = []

        def fake_run(*_args, stdout, **_kwargs):
            self.assertEqual(events, ["mutation_baseline"])
            events.append("teacher")
            stdout.write('{"type":"result","total_cost_usd":0}\n')
            return SimpleNamespace(returncode=0)

        def fake_baseline(_cid, _row):
            self.assertEqual(events, [])
            events.append("mutation_baseline")
            return {
                "mutation_f2p_reproduced": True,
                "mutation_baseline_commit": "f" * 40,
            }

        def fake_sh(command, *_args, **_kwargs):
            if command[:3] == ["docker", "run", "-d"]:
                return "0123456789abcdef\n", 0
            if command[:3] == ["docker", "exec", "0123456789ab"]:
                if "git diff --cached --binary --no-renames" in command[-1]:
                    return "diff --git a/a.py b/a.py\n", 0
                return "1 passed\n", 0
            if command[:3] == ["docker", "rm", "-f"]:
                return "", 0
            raise AssertionError(f"unexpected command: {command}")

        def assert_bound_preflight(_row, _backend, _stream, evidence):
            self.assertRegex(evidence["stream_sha256"], r"^[0-9a-f]{64}$")
            self.assertRegex(evidence["patch_sha256"], r"^[0-9a-f]{64}$")
            return SimpleNamespace(
                source_sha256="a" * 64,
                retained_steps=3,
            )

        with (
            TemporaryDirectory() as tmp,
            patch("phaseD_sft.teacher_trace_driver.subprocess.run", side_effect=fake_run),
            patch("phaseD_sft.teacher_trace_driver.sh", side_effect=fake_sh),
            patch(
                "phaseD_sft.teacher_trace_driver.verify_candidate_patch",
                return_value={
                    "training_admitted": True,
                    "candidate_passed_twice": True,
                    "baseline_failed": True,
                    "reference_passed": True,
                    "protected_stable": True,
                    "rejection_reasons": [],
                },
            ),
            patch(
                "phaseD_sft.teacher_trace_driver.establish_mutation_baseline",
                side_effect=fake_baseline,
            ),
            patch(
                "phaseD_sft.teacher_trace_driver.preflight_trainable_trace",
                side_effect=assert_bound_preflight,
            ),
        ):
            result = collect_one(row, Path(tmp), 40, "claude-opus-5", "claude")

        self.assertEqual(events, ["mutation_baseline", "teacher"])
        self.assertTrue(result["executed"])
        self.assertTrue(result["f2p_pass"])
        self.assertTrue(result["p2p_pass"])
        self.assertTrue(result["training_admitted"])
        self.assertTrue(result["resolved"])
        self.assertEqual(
            result["trace_preflight_source_sha256"],
            "a" * 64,
        )
        self.assertEqual(result["trace_preflight_retained_steps"], 3)

    def test_collect_one_rejects_strict_patch_when_trace_preflight_fails(self):
        row = {
            "instance_id": "org__repo.pr_1",
            "image_name": "fixture/image",
            "problem_statement": "Fix the bug.",
            "patch": "diff --git a/a.py b/a.py\n",
            "FAIL_TO_PASS": ["tests/test_fix.py::test_fix"],
            "PASS_TO_PASS": ["tests/test_fix.py::test_old"],
        }

        def fake_run(*_args, stdout, **_kwargs):
            stdout.write('{"type":"result","total_cost_usd":0}\n')
            return SimpleNamespace(returncode=0)

        def fake_sh(command, *_args, **_kwargs):
            if command[:3] == ["docker", "run", "-d"]:
                return "0123456789abcdef\n", 0
            if command[:3] == ["docker", "exec", "0123456789ab"]:
                if "git diff --cached --binary --no-renames" in command[-1]:
                    return "diff --git a/a.py b/a.py\n", 0
                return "1 passed\n", 0
            if command[:3] == ["docker", "rm", "-f"]:
                return "", 0
            raise AssertionError(f"unexpected command: {command}")

        def reject_preflight(*_args, **_kwargs):
            from teacher_platform.trace_gate import TracePreflightError

            raise TracePreflightError(
                "missing_focused_test",
                "focused F2P command absent",
            )

        with (
            TemporaryDirectory() as tmp,
            patch(
                "phaseD_sft.teacher_trace_driver.subprocess.run",
                side_effect=fake_run,
            ),
            patch(
                "phaseD_sft.teacher_trace_driver.sh",
                side_effect=fake_sh,
            ),
            patch(
                "phaseD_sft.teacher_trace_driver.verify_candidate_patch",
                return_value={
                    "training_admitted": True,
                    "candidate_passed_twice": True,
                    "baseline_failed": True,
                    "reference_passed": True,
                    "protected_stable": True,
                    "rejection_reasons": [],
                },
            ),
            patch(
                "phaseD_sft.teacher_trace_driver.establish_mutation_baseline",
                return_value={
                    "mutation_f2p_reproduced": True,
                    "mutation_baseline_commit": "f" * 40,
                },
            ),
            patch(
                "phaseD_sft.teacher_trace_driver.preflight_trainable_trace",
                side_effect=reject_preflight,
            ),
        ):
            outdir = Path(tmp)
            result = collect_one(
                row,
                outdir,
                40,
                "claude-fable-5",
                "claude",
            )
            self.assertTrue(
                (outdir / f"{row['instance_id']}.stream.jsonl").is_file()
            )
            self.assertTrue(
                (outdir / f"{row['instance_id']}.patch").is_file()
            )

        self.assertTrue(result["executed"])
        self.assertTrue(result["f2p_pass"])
        self.assertFalse(result["training_admitted"])
        self.assertFalse(result["resolved"])
        self.assertEqual(
            result["rejection_reasons"],
            ["trace_preflight:missing_focused_test"],
        )

    def test_collect_one_preserves_orphaned_raw_artifacts(self):
        row = {
            "instance_id": "org__repo.pr_1",
            "image_name": "fixture/image",
            "problem_statement": "Fix the bug.",
            "patch": "diff --git a/a.py b/a.py\n",
            "FAIL_TO_PASS": ["tests/test_fix.py::test_fix"],
            "PASS_TO_PASS": [],
        }
        with TemporaryDirectory() as tmp:
            outdir = Path(tmp)
            stream = outdir / "org__repo.pr_1.stream.jsonl"
            stream.write_text("preserve me\n")
            with patch(
                "phaseD_sft.teacher_trace_driver.sh",
                side_effect=AssertionError("must reject before starting Docker"),
            ):
                result = collect_one(
                    row,
                    outdir,
                    40,
                    "claude-fable-5",
                    "claude",
                )

            self.assertEqual(stream.read_text(), "preserve me\n")
            self.assertEqual(
                result["error"],
                "artifact_collision: raw stream or patch already exists",
            )
            self.assertFalse(result["resolved"])


if __name__ == "__main__":
    unittest.main()
