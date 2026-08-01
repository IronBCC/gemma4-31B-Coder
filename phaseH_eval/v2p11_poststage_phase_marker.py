#!/usr/bin/env python3
"""Publish or verify one immutable v2.11 post-training phase marker."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Sequence

from phaseH_eval.empty_retry_composite import (
    _binding,
    _publish_json_noreplace,
    _read_object,
)


def publish_or_verify(
    *,
    marker_path: Path,
    phase: str,
    artifact_paths: Sequence[Path],
) -> dict[str, Any]:
    marker_path = Path(marker_path).resolve()
    artifacts = [Path(path).resolve() for path in artifact_paths]
    if not phase or not artifacts:
        raise ValueError("phase name and artifacts are required")
    if any(not path.is_file() for path in artifacts):
        raise ValueError(f"{phase}: phase artifact is missing")
    report = {
        "schema_version": 1,
        "artifact_type": "v2p11_poststage_phase",
        "phase": phase,
        "status": "complete",
        "artifacts": [_binding(path) for path in artifacts],
    }
    if os.path.lexists(marker_path):
        if _read_object(marker_path) != report:
            raise ValueError(f"{phase}: immutable phase marker changed")
    else:
        _publish_json_noreplace(marker_path, report)
    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--marker", type=Path, required=True)
    parser.add_argument("--phase", required=True)
    parser.add_argument("artifacts", type=Path, nargs="+")
    args = parser.parse_args(argv)
    report = publish_or_verify(
        marker_path=args.marker,
        phase=args.phase,
        artifact_paths=args.artifacts,
    )
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
