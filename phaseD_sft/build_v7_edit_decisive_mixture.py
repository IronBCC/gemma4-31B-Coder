#!/usr/bin/env python3
"""Build the v7 edit-decisive mixture from the finalized v6 49k rows."""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
from typing import Any

from phaseD_sft.agentic_trace_filters import command_trace_quality_report
from phaseD_sft.progress import EtaProgress


EXTERNAL_SOURCES = frozenset({"kwai_klear_miniswe", "open_swe_traces_qwen35"})
ORACLE_SOURCE = "swe_train_oracle_edit_trace"
ORACLE_EDIT_UPWEIGHT_FACTOR = 2


def build_v7_mixture(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Filter only external rows, preserve anchors, and intentionally double oracle."""

    output: list[dict[str, Any]] = []
    external: dict[str, Counter[str]] = {source: Counter() for source in EXTERNAL_SOURCES}
    anchors: dict[str, Counter[str]] = {}

    for row in rows:
        source = str(row.get("source") or "MISSING")
        if source in EXTERNAL_SOURCES:
            stats = external[source]
            stats["in"] += 1
            report = command_trace_quality_report(
                row.get("messages") or [],
                max_first_edit_index=10,
                max_read_streak=5,
                reject_identical_consecutive_commands=True,
            )
            if report.keep:
                output.append(row)
                stats["kept"] += 1
            else:
                stats["dropped"] += 1
                for reason in report.reasons:
                    stats[f"reason:{reason}"] += 1
            continue

        anchor = anchors.setdefault(source, Counter())
        anchor["in"] += 1
        copies = ORACLE_EDIT_UPWEIGHT_FACTOR if source == ORACLE_SOURCE else 1
        output.extend(row.copy() for _ in range(copies))
        anchor["out"] += copies

    manifest = {
        "rows_in": len(rows),
        "rows_out_pre_dedup": len(output),
        "external_filter": {
            "max_first_edit_index": 10,
            "max_read_streak": 5,
            "reject_identical_consecutive_commands": True,
        },
        "external": {source: dict(external[source]) for source in sorted(EXTERNAL_SOURCES)},
        "anchors": {source: dict(stats) for source, stats in sorted(anchors.items())},
        "oracle_edit_source": ORACLE_SOURCE,
        "oracle_edit_upweight_factor": ORACLE_EDIT_UPWEIGHT_FACTOR,
        "oracle_edit_extra_rows": anchors.get(ORACLE_SOURCE, Counter()).get("in", 0),
        "dedup_passthrough_sources": sorted(anchors),
    }
    return output, manifest


def _load_rows(source: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in source.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if not isinstance(row, dict):
            raise ValueError(f"non-object JSONL row in {source}")
        rows.append(row)
    return rows


def _write_rows(rows: list[dict[str, Any]], output: Path, *, progress_every: int) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    progress = EtaProgress("write-v7-mixture", total=len(rows))
    with output.open("w", encoding="utf-8") as handle:
        for index, row in enumerate(rows, start=1):
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
            if index % progress_every == 0 or index == len(rows):
                print(progress.update(index), flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--in", dest="source", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--progress-every", type=int, default=500)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.progress_every <= 0:
        raise SystemExit("--progress-every must be positive")
    rows = _load_rows(args.source)
    output, manifest = build_v7_mixture(rows)
    _write_rows(output, args.out, progress_every=args.progress_every)
    manifest.update({"source": str(args.source), "output": str(args.out)})
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
