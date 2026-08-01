"""Publish a staged checkpoint only after its audit succeeds."""
from __future__ import annotations

from collections.abc import Callable
import os
from pathlib import Path


def publish_after_audit(
    staging: Path,
    output: Path,
    audit: Callable[[Path], None],
) -> None:
    audit(staging)
    os.replace(staging, output)
