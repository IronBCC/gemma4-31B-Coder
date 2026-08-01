from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time


ROOT = Path(__file__).resolve().parents[2]
SUPERVISOR = ROOT / "phaseH_eval" / "run_while_pids_alive.py"


def _sleeper(seconds: float = 30.0) -> subprocess.Popen[str]:
    return subprocess.Popen(
        [sys.executable, "-c", f"import time; time.sleep({seconds!r})"],
        text=True,
    )


def _stop(process: subprocess.Popen[str]) -> None:
    if process.poll() is None:
        process.terminate()
        try:
            process.wait(timeout=2)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=2)


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def test_stops_owned_command_group_when_watchdog_exits() -> None:
    serve = _sleeper()
    watchdog = _sleeper(0.4)
    try:
        with tempfile.TemporaryDirectory() as directory:
            pid_dir = Path(directory)
            leader_path = pid_dir / "leader.pid"
            child_path = pid_dir / "child.pid"
            command = (
                f"echo $$ > {leader_path}; "
                f"sleep 30 & echo $! > {child_path}; wait"
            )
            result = subprocess.run(
                [
                    sys.executable,
                    str(SUPERVISOR),
                    "--watch-pid",
                    f"serve={serve.pid}",
                    "--watch-pid",
                    f"watchdog={watchdog.pid}",
                    "--poll-seconds",
                    "0.02",
                    "--term-wait-seconds",
                    "1",
                    "--",
                    "bash",
                    "-c",
                    command,
                ],
                cwd=ROOT,
                text=True,
                capture_output=True,
                timeout=5,
                check=False,
            )

            assert result.returncode == 70, result.stderr
            assert f"label=watchdog pid={watchdog.pid}" in result.stderr
            assert leader_path.is_file()
            assert child_path.is_file()
            owned_pids = [
                int(leader_path.read_text()),
                int(child_path.read_text()),
            ]
            deadline = time.monotonic() + 2
            while any(_pid_alive(pid) for pid in owned_pids) and time.monotonic() < deadline:
                time.sleep(0.02)
            assert not any(_pid_alive(pid) for pid in owned_pids)
            assert serve.poll() is None
    finally:
        _stop(watchdog)
        _stop(serve)


def test_propagates_command_status_while_watched_pid_is_alive() -> None:
    serve = _sleeper()
    try:
        result = subprocess.run(
            [
                sys.executable,
                str(SUPERVISOR),
                "--watch-pid",
                f"serve={serve.pid}",
                "--poll-seconds",
                "0.02",
                "--",
                sys.executable,
                "-c",
                "raise SystemExit(17)",
            ],
            cwd=ROOT,
            text=True,
            capture_output=True,
            timeout=5,
            check=False,
        )

        assert result.returncode == 17, result.stderr
        assert serve.poll() is None
    finally:
        _stop(serve)


def test_stops_owned_descendant_after_command_leader_exits() -> None:
    serve = _sleeper()
    child_pid = 0
    try:
        with tempfile.TemporaryDirectory() as directory:
            child_path = Path(directory) / "child.pid"
            command = (
                "from pathlib import Path; import subprocess, sys; "
                "child = subprocess.Popen([sys.executable, '-c', "
                "'import time; time.sleep(30)']); "
                f"Path({str(child_path)!r}).write_text(str(child.pid))"
            )
            result = subprocess.run(
                [
                    sys.executable,
                    str(SUPERVISOR),
                    "--watch-pid",
                    f"serve={serve.pid}",
                    "--poll-seconds",
                    "0.02",
                    "--term-wait-seconds",
                    "1",
                    "--",
                    sys.executable,
                    "-c",
                    command,
                ],
                cwd=ROOT,
                text=True,
                capture_output=True,
                timeout=5,
                check=False,
            )

            assert result.returncode == 0, result.stderr
            assert child_path.is_file()
            child_pid = int(child_path.read_text())
            deadline = time.monotonic() + 2
            while _pid_alive(child_pid) and time.monotonic() < deadline:
                time.sleep(0.02)
            assert not _pid_alive(child_pid)
            assert serve.poll() is None
    finally:
        if child_pid and _pid_alive(child_pid):
            os.kill(child_pid, 9)
        _stop(serve)
