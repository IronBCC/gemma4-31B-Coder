from __future__ import annotations

import os
import signal
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
WATCHDOG = REPO_ROOT / "phaseD_sft" / "ram_watchdog.py"


class RamWatchdogCliTests(unittest.TestCase):
    def _start_sleeper(self, seconds: float = 30.0) -> subprocess.Popen[str]:
        return subprocess.Popen(
            [sys.executable, "-c", f"import time; time.sleep({seconds!r})"],
            text=True,
        )

    def _stop(self, process: subprocess.Popen[str]) -> None:
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=2)

    def _run_watchdog(
        self,
        pid: int,
        meminfo_path: Path,
        *extra_args: str,
    ) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [
                sys.executable,
                str(WATCHDOG),
                "--pid",
                str(pid),
                "--interval-seconds",
                "0.02",
                "--meminfo-path",
                str(meminfo_path),
                *extra_args,
            ],
            cwd=REPO_ROOT,
            text=True,
            capture_output=True,
            timeout=5,
            check=False,
        )

    def test_exits_zero_when_target_exits_naturally(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            temp_dir = Path(directory)
            meminfo_path = temp_dir / "meminfo"
            pid_path = temp_dir / "target.pid"
            meminfo_path.write_text("MemAvailable: 4194304 kB\n", encoding="utf-8")
            reaper = subprocess.Popen(
                [
                    sys.executable,
                    "-c",
                    (
                        "import pathlib, subprocess, sys, time; "
                        "target = subprocess.Popen([sys.executable, '-c', "
                        "'import time; time.sleep(0.5)']); "
                        "pathlib.Path(sys.argv[1]).write_text(str(target.pid)); "
                        "target.wait(); time.sleep(2)"
                    ),
                    str(pid_path),
                ],
                text=True,
            )
            try:
                deadline = time.monotonic() + 2
                while not pid_path.exists() and time.monotonic() < deadline:
                    time.sleep(0.01)
                self.assertTrue(pid_path.exists(), "reaper did not publish the target PID")

                result = self._run_watchdog(int(pid_path.read_text()), meminfo_path)

                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertIn("event=target_exited", result.stdout)
            finally:
                self._stop(reaper)

    def test_first_low_sample_terminates_only_exact_pid_and_logs_progress(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            temp_dir = Path(directory)
            meminfo_path = temp_dir / "meminfo"
            log_path = temp_dir / "watchdog.log"
            meminfo_path.write_text("MemAvailable: 1048576 kB\n", encoding="utf-8")
            target = self._start_sleeper()
            unrelated = self._start_sleeper()
            try:
                result = self._run_watchdog(
                    target.pid,
                    meminfo_path,
                    "--log-path",
                    str(log_path),
                )

                self.assertEqual(result.returncode, 3, result.stderr)
                self.assertEqual(target.wait(timeout=2), -signal.SIGTERM)
                self.assertIsNone(unrelated.poll())
                self.assertIn("available_gib=1.000000", result.stdout)
                self.assertIn("threshold_gib=2.000000", result.stdout)
                self.assertIn("timestamp=", result.stdout)
                self.assertIn("event=below_threshold", log_path.read_text(encoding="utf-8"))
            finally:
                self._stop(target)
                self._stop(unrelated)

    def test_missing_memavailable_fails_closed_without_killing_unrelated_pid(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            meminfo_path = Path(directory) / "meminfo"
            meminfo_path.write_text("MemTotal: 8388608 kB\n", encoding="utf-8")
            target = self._start_sleeper()
            unrelated = self._start_sleeper()
            try:
                result = self._run_watchdog(target.pid, meminfo_path)

                self.assertEqual(result.returncode, 4, result.stderr)
                self.assertEqual(target.wait(timeout=2), -signal.SIGTERM)
                self.assertIsNone(unrelated.poll())
                self.assertIn("event=meminfo_error", result.stdout)
            finally:
                self._stop(target)
                self._stop(unrelated)

    def test_sustained_health_failure_terminates_only_exact_pid(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            meminfo_path = Path(directory) / "meminfo"
            meminfo_path.write_text(
                "MemAvailable: 4194304 kB\n",
                encoding="utf-8",
            )
            target = self._start_sleeper()
            unrelated = self._start_sleeper()
            try:
                result = self._run_watchdog(
                    target.pid,
                    meminfo_path,
                    "--health-url",
                    "http://127.0.0.1:9/health",
                    "--health-failures",
                    "1",
                    "--health-timeout-seconds",
                    "0.1",
                )

                self.assertEqual(result.returncode, 6, result.stderr)
                self.assertEqual(target.wait(timeout=2), -signal.SIGTERM)
                self.assertIsNone(unrelated.poll())
                self.assertIn("event=health_failure", result.stdout)
                self.assertIn("consecutive_failures=1", result.stdout)
            finally:
                self._stop(target)
                self._stop(unrelated)

    def test_malformed_memavailable_fails_closed_without_killing_unrelated_pid(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            meminfo_path = Path(directory) / "meminfo"
            meminfo_path.write_text("MemAvailable: unknown kB\n", encoding="utf-8")
            target = self._start_sleeper()
            unrelated = self._start_sleeper()
            try:
                result = self._run_watchdog(target.pid, meminfo_path)

                self.assertEqual(result.returncode, 4, result.stderr)
                self.assertEqual(target.wait(timeout=2), -signal.SIGTERM)
                self.assertIsNone(unrelated.poll())
                self.assertIn("event=meminfo_error", result.stdout)
            finally:
                self._stop(target)
                self._stop(unrelated)

    def test_rejects_invalid_numeric_arguments(self) -> None:
        invalid_args = (
            ("--pid", "0", "must be a positive integer"),
            ("--pid", "-1", "must be a positive integer"),
            ("--interval-seconds", "0", "must be a positive number"),
            ("--interval-seconds", "-0.1", "must be a positive number"),
            ("--min-available-gib", "-1", "must be nonnegative"),
        )
        for option, value, expected_error in invalid_args:
            with self.subTest(option=option, value=value):
                command = [
                    sys.executable,
                    str(WATCHDOG),
                    "--pid",
                    str(os.getpid()),
                    option,
                    value,
                ]
                result = subprocess.run(
                    command,
                    cwd=REPO_ROOT,
                    text=True,
                    capture_output=True,
                    timeout=2,
                    check=False,
                )
                self.assertEqual(result.returncode, 2)
                self.assertIn(expected_error, result.stderr)

    def test_rejects_pid_that_is_not_alive(self) -> None:
        result = subprocess.run(
            [sys.executable, str(WATCHDOG), "--pid", "2147483647"],
            cwd=REPO_ROOT,
            text=True,
            capture_output=True,
            timeout=2,
            check=False,
        )

        self.assertEqual(result.returncode, 2)
        self.assertIn("not alive", result.stderr)


if __name__ == "__main__":
    unittest.main()
