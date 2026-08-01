"""Publish a staged checkpoint only after its audit succeeds."""
from __future__ import annotations

from collections.abc import Callable
import ctypes
import errno
import os
from pathlib import Path
import sys


def _rename_noreplace(staging: Path, output: Path) -> None:
    source = os.fsencode(staging)
    destination = os.fsencode(output)
    libc = ctypes.CDLL(None, use_errno=True)
    if sys.platform.startswith("linux") and hasattr(libc, "renameat2"):
        renameat2 = libc.renameat2
        renameat2.argtypes = [
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_uint,
        ]
        renameat2.restype = ctypes.c_int
        result = renameat2(-100, source, -100, destination, 1)
    elif sys.platform == "darwin" and hasattr(libc, "renamex_np"):
        renamex_np = libc.renamex_np
        renamex_np.argtypes = [
            ctypes.c_char_p,
            ctypes.c_char_p,
            ctypes.c_uint,
        ]
        renamex_np.restype = ctypes.c_int
        result = renamex_np(source, destination, 0x00000004)
    else:
        raise RuntimeError("atomic no-replace directory rename is unsupported")
    if result == 0:
        return
    error = ctypes.get_errno()
    if error == errno.EEXIST:
        raise FileExistsError(error, os.strerror(error), output)
    raise OSError(error, os.strerror(error), output)


def publish_after_audit(
    staging: Path,
    output: Path,
    audit: Callable[[Path], None],
) -> None:
    audit(staging)
    _rename_noreplace(staging, output)
