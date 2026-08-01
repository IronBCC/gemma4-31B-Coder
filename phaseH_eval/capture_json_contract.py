#!/usr/bin/env python3
"""Run a contract command and atomically publish or revalidate its JSON."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
from typing import Any, Sequence

from phaseH_eval.empty_retry_composite import (
    _publish_json_noreplace,
    _read_object,
)


def capture_json_contract(
    *,
    output_path: Path,
    command: Sequence[str],
) -> dict[str, Any]:
    output_path = Path(output_path).resolve()
    if not command or any(not isinstance(value, str) or not value for value in command):
        raise ValueError("contract command must be a nonempty argv sequence")
    completed = subprocess.run(
        list(command),
        check=False,
        text=True,
        capture_output=True,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            "contract command failed "
            f"status={completed.returncode}: {completed.stderr.strip()}"
        )
    try:
        report = json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        raise ValueError("contract command did not emit one JSON value") from exc
    if not isinstance(report, dict):
        raise ValueError("contract command JSON is not an object")
    if os.path.lexists(output_path):
        if _read_object(output_path) != report:
            raise ValueError(
                "existing contract differs from current command"
            )
    else:
        _publish_json_noreplace(output_path, report)
    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    command = list(args.command)
    if command and command[0] == "--":
        command.pop(0)
    report = capture_json_contract(
        output_path=args.out,
        command=command,
    )
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
