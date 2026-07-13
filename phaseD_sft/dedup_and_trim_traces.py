#!/usr/bin/env python3
"""Trim trailing non-assistant messages and deduplicate agentic traces by content hash."""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

from phaseD_sft.progress import EtaProgress

def _content_hash(messages: list[dict[str, Any]]) -> str:
    return hashlib.sha256(
        json.dumps(messages, sort_keys=True).encode("utf-8")
    ).hexdigest()


def trim_trace(row: dict[str, Any]) -> tuple[dict[str, Any] | None, Counter[str]]:
    """Drop trailing non-assistant messages; drop trace if no assistant or <2 total."""
    stats = Counter()
    row = json.loads(json.dumps(row))
    messages = row.get("messages")

    if not isinstance(messages, list):
        stats["trailing_trimmed_messages"] += 0
        stats["traces_dropped_no_assistant"] += 1
        return None, stats

    while len(messages) > 0 and messages[-1].get("role") != "assistant":
        messages.pop()
        stats["trailing_trimmed_messages"] += 1

    has_assistant = any(m.get("role") == "assistant" for m in messages)
    if not has_assistant or len(messages) < 2:
        stats["traces_dropped_no_assistant"] += 1
        return None, stats

    row["messages"] = messages
    return row, stats


def dedup_traces(
    rows: list[dict[str, Any]],
    passthrough_sources: frozenset[str] = frozenset(),
    *,
    progress_every: int | None = None,
) -> tuple[list[dict[str, Any]], Counter[str]]:
    """Deduplicate by content hash; keep FIRST occurrence. Never dedup by instance_id.

    Rows whose source is in passthrough_sources skip dedup entirely: sources with
    deliberate replication-based upweighting (e.g. synthetic anchors repeated x64)
    must keep their copies. Their duplicate counts are still reported.
    """
    seen: set[str] = set()
    seen_passthrough: set[str] = set()
    result: list[dict[str, Any]] = []
    stats: Counter[str] = Counter()
    progress = EtaProgress("dedup-traces", total=len(rows)) if progress_every else None

    for index, row in enumerate(rows, start=1):
        messages = row.get("messages")
        if not isinstance(messages, list):
            continue
        h = _content_hash(messages)
        if row.get("source") in passthrough_sources:
            if h in seen_passthrough:
                stats["intended_dups_kept"] += 1
            seen_passthrough.add(h)
            result.append(row)
        elif h in seen:
            stats["exact_dups_removed"] += 1
        else:
            seen.add(h)
            result.append(row)
        if progress and (index % progress_every == 0 or index == len(rows)):
            print(progress.update(index), flush=True)

    return result, stats


def load_rows_from_jsonl(source: Path) -> tuple[list[dict[str, Any]], Counter[str]]:
    rows: list[dict[str, Any]] = []
    stats: Counter[str] = Counter()
    with source.open(encoding="utf-8") as f:
        for line in f:
            stripped = line.strip()
            if not stripped:
                continue
            try:
                row = json.loads(stripped)
            except json.JSONDecodeError:
                stats["malformed_rows"] += 1
                continue
            if isinstance(row, dict):
                rows.append(row)
    return rows, stats


def load_rows_from_hf(source: Path) -> list[dict[str, Any]]:
    from datasets import load_from_disk

    ds = load_from_disk(str(source))
    return [dict(row) for row in ds]


def process(
    source: Path,
    output: Path,
    *,
    hf: bool = False,
    passthrough_sources: frozenset[str] = frozenset(),
    progress_every: int = 500,
) -> dict[str, Any]:
    if progress_every <= 0:
        raise ValueError("progress_every must be positive")
    totals: Counter[str] = Counter()

    if hf:
        raw_rows = load_rows_from_hf(source)
    else:
        raw_rows, parse_stats = load_rows_from_jsonl(source)
        totals.update(parse_stats)

    totals["rows_in"] += len(raw_rows)

    # Per-source in-counts (before any processing)
    per_source: dict[str, dict[str, int]] = {}
    for row in raw_rows:
        src = row.get("source", "MISSING")
        if src not in per_source:
            per_source[src] = {"in": 0, "out": 0}
        per_source[src]["in"] += 1

    # --- TRIM phase ---
    survived_trim: list[dict[str, Any]] = []
    trim_progress = EtaProgress("trim-traces", total=len(raw_rows))
    for index, row in enumerate(raw_rows, start=1):
        kept, trim_stats = trim_trace(row)
        totals.update(trim_stats)
        if kept is not None:
            survived_trim.append(kept)
        if index % progress_every == 0 or index == len(raw_rows):
            print(trim_progress.update(index), flush=True)

    # --- DEDUP phase (content hash only; never by instance_id) ---
    deduped, dedup_stats = dedup_traces(
        survived_trim,
        passthrough_sources,
        progress_every=progress_every,
    )
    totals.update(dedup_stats)
    totals["unique_content_count"] += len(deduped)

    # Update per-source out-counts
    for row in deduped:
        src = row.get("source", "MISSING")
        if src not in per_source:
            per_source[src] = {"in": 0, "out": 0}
        per_source[src]["out"] += 1

    # Write output
    output.parent.mkdir(parents=True, exist_ok=True)
    write_progress = EtaProgress("write-deduped-traces", total=len(deduped))
    with output.open("w", encoding="utf-8") as f:
        for index, row in enumerate(deduped, start=1):
            f.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
            if index % progress_every == 0 or index == len(deduped):
                print(write_progress.update(index), flush=True)

    manifest = dict(totals)
    manifest.update(
        {
            "rows_in": totals["rows_in"],
            "rows_out": len(deduped),
            "exact_dups_removed": totals.get("exact_dups_removed", 0),
            "trailing_trimmed_messages": totals.get("trailing_trimmed_messages", 0),
            "traces_dropped_no_assistant": totals.get(
                "traces_dropped_no_assistant", 0
            ),
            "unique_content_count": len(deduped),
            "intended_dups_kept": totals.get("intended_dups_kept", 0),
            "passthrough_sources": sorted(passthrough_sources),
            "per_source": per_source,
            "source": str(source),
            "output": str(output),
            "format": "hf" if hf else "jsonl",
        }
    )
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--in", dest="source", required=True, type=Path, help="Input JSONL file or HF dataset directory.")
    parser.add_argument("--out", dest="output", required=True, type=Path, help="Output deduplicated JSONL file.")
    parser.add_argument("--manifest", type=Path, help="Optional JSON manifest path.")
    parser.add_argument(
        "--hf",
        action="store_true",
        help="Treat --in as a HuggingFace dataset directory (load_from_disk).",
    )
    parser.add_argument(
        "--passthrough-sources",
        default="",
        help="Comma-separated source values exempt from dedup (deliberately replicated anchors).",
    )
    parser.add_argument("--progress-every", type=int, default=500)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.progress_every <= 0:
        raise SystemExit("--progress-every must be positive")
    passthrough = frozenset(s.strip() for s in args.passthrough_sources.split(",") if s.strip())
    manifest = process(
        args.source,
        args.output,
        hf=args.hf,
        passthrough_sources=passthrough,
        progress_every=args.progress_every,
    )
    if args.manifest:
        args.manifest.parent.mkdir(parents=True, exist_ok=True)
        args.manifest.write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
