#!/usr/bin/env python3
"""Remove zero-supervision assistant placeholders without changing real assistant turns."""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
from typing import Any

from phaseD_sft.progress import EtaProgress


def _is_empty_assistant(message: dict[str, Any]) -> bool:
    return (
        message.get("role") == "assistant"
        and not (message.get("content") or "").strip()
        and not message.get("tool_calls")
    )


def _merge_observations(previous: dict[str, Any], current: dict[str, Any]) -> bool:
    previous_text = previous.get("content")
    current_text = current.get("content")
    if not (
        previous.get("role") == current.get("role") == "user"
        and isinstance(previous_text, str)
        and isinstance(current_text, str)
        and previous_text.startswith("OBSERVATION:")
        and current_text.startswith("OBSERVATION:")
    ):
        return False

    prefix = "OBSERVATION:\n"
    suffix = current_text[len(prefix) :] if current_text.startswith(prefix) else current_text[len("OBSERVATION:") :].lstrip()
    previous["content"] = previous_text.rstrip() + "\n\n" + suffix.lstrip()
    return True


def sanitize_row(row: dict[str, Any]) -> tuple[dict[str, Any], Counter[str]]:
    """Remove empty assistant placeholders and merge only adjacent observation pairs."""
    sanitized = json.loads(json.dumps(row))
    stats: Counter[str] = Counter()
    messages = sanitized.get("messages")
    if not isinstance(messages, list):
        return sanitized, stats

    kept: list[dict[str, Any]] = []
    for message in messages:
        if not isinstance(message, dict):
            kept.append(message)
            continue
        if _is_empty_assistant(message):
            stats["empty_assistant_removed"] += 1
            continue
        if kept and isinstance(kept[-1], dict) and _merge_observations(kept[-1], message):
            stats["adjacent_observations_merged"] += 1
            continue
        kept.append(message)
    sanitized["messages"] = kept
    return sanitized, stats


def sanitize_jsonl(source: Path, output: Path, *, progress_every: int = 100) -> dict[str, Any]:
    if progress_every <= 0:
        raise ValueError("progress_every must be positive")
    output.parent.mkdir(parents=True, exist_ok=True)
    total_bytes = source.stat().st_size
    progress = EtaProgress("sanitize-empty-assistant", total=total_bytes)
    stats: Counter[str] = Counter()
    bytes_read = 0
    with source.open(encoding="utf-8") as src, output.open("w", encoding="utf-8") as dst:
        for line_number, line in enumerate(src, start=1):
            bytes_read += len(line.encode("utf-8"))
            if not line.strip():
                continue
            row = json.loads(line)
            sanitized, row_stats = sanitize_row(row)
            stats.update(row_stats)
            stats["rows_read"] += 1
            dst.write(json.dumps(sanitized, ensure_ascii=False, separators=(",", ":")) + "\n")
            if line_number % progress_every == 0:
                print(progress.update(min(bytes_read, total_bytes)), flush=True)
    print(progress.update(total_bytes), flush=True)
    return dict(stats)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--in", dest="source", required=True, type=Path)
    parser.add_argument("--out", dest="output", required=True, type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--progress-every", type=int, default=100)
    args = parser.parse_args()
    stats = sanitize_jsonl(args.source, args.output, progress_every=args.progress_every)
    manifest = {"source": str(args.source), "output": str(args.output), **stats}
    if args.manifest:
        args.manifest.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
