#!/usr/bin/env python3
"""Run one owned command group while required external PIDs remain alive."""

from __future__ import annotations

import argparse
import errno
import os
import re
import signal
import subprocess
import sys
import time
from pathlib import Path


EXIT_WATCHED_PID = 70
WATCH_SPEC = re.compile(r"^(?P<label>[A-Za-z][A-Za-z0-9_-]*)=(?P<pid>[1-9][0-9]*)$")


def positive_number(value: str) -> float:
    try:
        parsed = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be a positive number") from exc
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive number")
    return parsed


def watched_pid(value: str) -> tuple[str, int]:
    match = WATCH_SPEC.fullmatch(value)
    if match is None:
        raise argparse.ArgumentTypeError("must have the form label=positive_pid")
    return match.group("label"), int(match.group("pid"))


def pid_is_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError as exc:
        if exc.errno == errno.ESRCH:
            return False
        if exc.errno == errno.EPERM:
            return True
        raise
    stat_path = Path(f"/proc/{pid}/stat")
    if stat_path.is_file():
        try:
            stat = stat_path.read_text(encoding="utf-8")
        except OSError:
            return False
        closing_parenthesis = stat.rfind(")")
        if closing_parenthesis >= 0:
            fields = stat[closing_parenthesis + 1 :].split()
            return not fields or fields[0] != "Z"
        return True
    state = subprocess.run(
        ["ps", "-o", "stat=", "-p", str(pid)],
        capture_output=True,
        text=True,
        check=False,
    ).stdout.strip()
    return bool(state) and not state.startswith("Z")


def terminate_owned_group(process: subprocess.Popen[bytes], wait_seconds: float) -> None:
    group_pid = process.pid
    try:
        os.killpg(group_pid, signal.SIGTERM)
    except (ProcessLookupError, PermissionError):
        pass

    deadline = time.monotonic() + wait_seconds
    while time.monotonic() < deadline:
        process.poll()
        try:
            os.killpg(group_pid, 0)
        except (ProcessLookupError, PermissionError):
            break
        time.sleep(min(0.05, max(0.0, deadline - time.monotonic())))
    else:
        try:
            os.killpg(group_pid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            pass

    if process.poll() is None:
        try:
            process.wait(timeout=wait_seconds)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=wait_seconds)


def command_exit_status(returncode: int) -> int:
    if returncode < 0:
        return 128 + abs(returncode)
    return returncode


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--watch-pid",
        action="append",
        required=True,
        type=watched_pid,
        help="required external process as label=positive_pid; may be repeated",
    )
    parser.add_argument("--poll-seconds", type=positive_number, default=1.0)
    parser.add_argument("--term-wait-seconds", type=positive_number, default=10.0)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    return parser


def run(args: argparse.Namespace) -> int:
    command = list(args.command)
    if command[:1] == ["--"]:
        command = command[1:]
    if not command:
        raise ValueError("a command is required after --")
    for label, pid in args.watch_pid:
        if not pid_is_alive(pid):
            print(
                f"guard_target_exited label={label} pid={pid} before_launch=true",
                file=sys.stderr,
                flush=True,
            )
            return EXIT_WATCHED_PID

    process = subprocess.Popen(command, start_new_session=True)
    try:
        while True:
            returncode = process.poll()
            if returncode is not None:
                terminate_owned_group(process, args.term_wait_seconds)
                return command_exit_status(returncode)
            for label, pid in args.watch_pid:
                if not pid_is_alive(pid):
                    print(
                        f"guard_target_exited label={label} pid={pid} before_launch=false",
                        file=sys.stderr,
                        flush=True,
                    )
                    terminate_owned_group(process, args.term_wait_seconds)
                    return EXIT_WATCHED_PID
            time.sleep(args.poll_seconds)
    except BaseException:
        terminate_owned_group(process, args.term_wait_seconds)
        raise


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    try:
        return run(args)
    except ValueError as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    raise SystemExit(main())
