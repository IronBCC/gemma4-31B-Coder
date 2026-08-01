#!/usr/bin/env python3
"""Terminate one exact PID when host MemAvailable crosses a configured floor."""

from __future__ import annotations

import argparse
import errno
import math
import os
import re
import signal
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import TextIO
from urllib import error as urllib_error
from urllib import request as urllib_request


EXIT_BELOW_THRESHOLD = 3
EXIT_MEMINFO_ERROR = 4
EXIT_SIGNAL_ERROR = 5
EXIT_HEALTH_ERROR = 6
KIB_PER_GIB = 1024 * 1024
MEMAVAILABLE_PATTERN = re.compile(r"^MemAvailable:\s+(\d+)\s+kB\s*$")


class MeminfoError(RuntimeError):
    """Raised when MemAvailable cannot be read unambiguously."""


def positive_integer(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be a positive integer") from exc
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def positive_number(value: str) -> float:
    try:
        parsed = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be a positive number") from exc
    if not math.isfinite(parsed) or parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive number")
    return parsed


def nonnegative_number(value: str) -> float:
    try:
        parsed = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be nonnegative") from exc
    if not math.isfinite(parsed) or parsed < 0:
        raise argparse.ArgumentTypeError("must be nonnegative")
    return parsed


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
    return True


def read_available_gib(meminfo_path: Path) -> float:
    try:
        lines = meminfo_path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise MeminfoError(f"cannot read {meminfo_path}: {exc}") from exc

    matches = [MEMAVAILABLE_PATTERN.fullmatch(line) for line in lines]
    values = [int(match.group(1)) for match in matches if match is not None]
    if len(values) != 1:
        raise MeminfoError(f"expected exactly one valid MemAvailable line in {meminfo_path}")
    return values[0] / KIB_PER_GIB


def timestamp() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


class ProgressEmitter:
    def __init__(self, log_path: Path | None) -> None:
        self._log: TextIO | None = None
        if log_path is not None:
            self._log = log_path.open("a", encoding="utf-8", buffering=1)

    def emit(self, line: str) -> None:
        print(line, flush=True)
        if self._log is not None:
            self._log.write(line + "\n")
            self._log.flush()

    def close(self) -> None:
        if self._log is not None:
            self._log.close()


def terminate_exact_pid(pid: int, emitter: ProgressEmitter, reason: str) -> bool:
    try:
        os.kill(pid, signal.SIGTERM)
    except ProcessLookupError:
        emitter.emit(f"event=target_exited pid={pid} reason={reason} timestamp={timestamp()}")
        return False
    except OSError as exc:
        emitter.emit(
            f"event=signal_error pid={pid} reason={reason} errno={exc.errno} "
            f"timestamp={timestamp()}"
        )
        raise
    return True


def health_failure_detail(
    urls: list[str],
    *,
    timeout_seconds: float,
) -> str | None:
    for url in urls:
        try:
            with urllib_request.urlopen(
                url,
                timeout=timeout_seconds,
            ) as response:
                status = response.status
                response.read(1)
        except (
            OSError,
            TimeoutError,
            urllib_error.HTTPError,
            urllib_error.URLError,
        ) as exc:
            return f"{url}: {type(exc).__name__}: {exc}"
        if not 200 <= status < 300:
            return f"{url}: unexpected HTTP status {status}"
    return None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pid", required=True, type=positive_integer, help="exact target PID")
    parser.add_argument(
        "--min-available-gib",
        type=nonnegative_number,
        default=2.0,
        help="terminate below this MemAvailable value (default: 2)",
    )
    parser.add_argument(
        "--interval-seconds",
        type=positive_number,
        default=10.0,
        help="poll interval in seconds (default: 10)",
    )
    parser.add_argument(
        "--meminfo-path",
        type=Path,
        default=Path("/proc/meminfo"),
        help="meminfo source (default: /proc/meminfo)",
    )
    parser.add_argument("--log-path", type=Path, help="optional append-only progress log")
    parser.add_argument(
        "--health-url",
        action="append",
        default=[],
        help="HTTP health endpoint; may be repeated",
    )
    parser.add_argument(
        "--health-timeout-seconds",
        type=positive_number,
        default=5.0,
        help="per-endpoint HTTP timeout in seconds (default: 5)",
    )
    parser.add_argument(
        "--health-failures",
        type=positive_integer,
        default=3,
        help="consecutive failed health samples before termination (default: 3)",
    )
    return parser


def run(args: argparse.Namespace, emitter: ProgressEmitter) -> int:
    consecutive_health_failures = 0
    while True:
        if not pid_is_alive(args.pid):
            emitter.emit(f"event=target_exited pid={args.pid} timestamp={timestamp()}")
            return 0

        try:
            available_gib = read_available_gib(args.meminfo_path)
        except MeminfoError as exc:
            emitter.emit(
                f"event=meminfo_error pid={args.pid} available_gib=unknown "
                f"threshold_gib={args.min_available_gib:.6f} timestamp={timestamp()} "
                f"detail={str(exc)!r}"
            )
            try:
                terminated = terminate_exact_pid(args.pid, emitter, "meminfo_error")
            except OSError:
                return EXIT_SIGNAL_ERROR
            return EXIT_MEMINFO_ERROR if terminated else 0

        event = "below_threshold" if available_gib < args.min_available_gib else "sample"
        emitter.emit(
            f"event={event} pid={args.pid} available_gib={available_gib:.6f} "
            f"threshold_gib={args.min_available_gib:.6f} timestamp={timestamp()}"
        )
        if available_gib < args.min_available_gib:
            try:
                terminated = terminate_exact_pid(args.pid, emitter, "below_threshold")
            except OSError:
                return EXIT_SIGNAL_ERROR
            return EXIT_BELOW_THRESHOLD if terminated else 0

        health_detail = health_failure_detail(
            args.health_url,
            timeout_seconds=args.health_timeout_seconds,
        )
        if health_detail is None:
            consecutive_health_failures = 0
        else:
            consecutive_health_failures += 1
            emitter.emit(
                f"event=health_failure pid={args.pid} "
                f"consecutive_failures={consecutive_health_failures} "
                f"required_failures={args.health_failures} "
                f"urls={','.join(args.health_url)!r} "
                f"timestamp={timestamp()} detail={health_detail!r}"
            )
            if consecutive_health_failures >= args.health_failures:
                try:
                    terminated = terminate_exact_pid(
                        args.pid,
                        emitter,
                        "health_failure",
                    )
                except OSError:
                    return EXIT_SIGNAL_ERROR
                return EXIT_HEALTH_ERROR if terminated else 0

        time.sleep(args.interval_seconds)


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    if not pid_is_alive(args.pid):
        parser.error(f"PID {args.pid} is not alive")

    try:
        emitter = ProgressEmitter(args.log_path)
    except OSError as exc:
        parser.error(f"cannot open log path {args.log_path}: {exc}")
    try:
        return run(args, emitter)
    finally:
        emitter.close()


if __name__ == "__main__":
    raise SystemExit(main())
