#!/usr/bin/env python3
"""Preserve interrupted v2.11 poststage outputs before a safe restart."""
from __future__ import annotations

import argparse
import ctypes
from datetime import datetime, timezone
import errno
import json
import os
from pathlib import Path
import re
import sys
from typing import Any, Sequence


_CHECKPOINT_RE = re.compile(r"checkpoint-[1-9][0-9]*")
_PHASES = ("recovery-sft", "kto-canary", "kto-full")


def _rename_directory_noreplace(source: Path, destination: Path) -> None:
    """Atomically rename a directory without replacing any destination."""
    library = ctypes.CDLL(None, use_errno=True)
    source_bytes = os.fsencode(source)
    destination_bytes = os.fsencode(destination)
    ctypes.set_errno(0)
    if sys.platform == "darwin":
        operation = getattr(library, "renamex_np", None)
        if operation is None:
            raise RuntimeError("atomic no-replace rename is unavailable")
        operation.argtypes = [
            ctypes.c_char_p,
            ctypes.c_char_p,
            ctypes.c_uint,
        ]
        operation.restype = ctypes.c_int
        result = operation(
            source_bytes,
            destination_bytes,
            0x00000004,
        )
    elif sys.platform.startswith("linux"):
        operation = getattr(library, "renameat2", None)
        if operation is None:
            raise RuntimeError("atomic no-replace rename is unavailable")
        operation.argtypes = [
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_uint,
        ]
        operation.restype = ctypes.c_int
        result = operation(
            -100,
            source_bytes,
            -100,
            destination_bytes,
            0x00000001,
        )
    else:
        raise RuntimeError("atomic no-replace rename is unavailable")
    if result == 0:
        return
    error_number = ctypes.get_errno()
    if error_number in {errno.EEXIST, errno.ENOTEMPTY}:
        raise FileExistsError(
            error_number,
            os.strerror(error_number),
            os.fspath(destination),
        )
    raise OSError(
        error_number,
        os.strerror(error_number),
        os.fspath(destination),
    )


def recover_interrupted_output(
    path: Path,
    *,
    phase: str,
    mode: str,
) -> dict[str, Any]:
    if phase not in _PHASES:
        raise ValueError("poststage recovery phase is invalid")
    if mode not in {"checkpointless", "always"}:
        raise ValueError("poststage recovery mode is invalid")
    path = Path(os.path.abspath(path))
    if not os.path.lexists(path):
        return {
            "archive_path": None,
            "phase": phase,
            "status": "absent",
        }
    if path.is_symlink() or not path.is_dir():
        raise ValueError(f"interrupted output must be a real directory: {path}")
    checkpoints = sorted(
        child.name
        for child in path.iterdir()
        if (
            not child.is_symlink()
            and child.is_dir()
            and _CHECKPOINT_RE.fullmatch(child.name)
        )
    )
    if mode == "checkpointless" and checkpoints:
        return {
            "archive_path": None,
            "phase": phase,
            "status": "checkpointed",
        }
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    archive = path.with_name(
        f"{path.name}.interrupted-{stamp}-{os.getpid()}"
    )
    _rename_directory_noreplace(path, archive)
    directory_fd = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)
    return {
        "archive_path": str(archive),
        "phase": phase,
        "status": "archived",
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--path", type=Path, required=True)
    parser.add_argument("--phase", choices=_PHASES, required=True)
    parser.add_argument(
        "--mode",
        choices=("checkpointless", "always"),
        required=True,
    )
    args = parser.parse_args(argv)
    try:
        report = recover_interrupted_output(
            args.path,
            phase=args.phase,
            mode=args.mode,
        )
    except ValueError as error:
        raise SystemExit(str(error)) from error
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
