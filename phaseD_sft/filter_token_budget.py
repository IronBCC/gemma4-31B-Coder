#!/usr/bin/env python3
"""Split JSONL traces into accepted/rejected files by tokenizer budget."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Callable

from phaseD_sft.progress import EtaProgress
from phaseD_sft.token_audit import load_token_counter, message_text, row_messages

TokenCounter = Callable[[str], int]


def row_token_count(row: dict[str, Any], count_tokens: TokenCounter) -> int:
    return sum(count_tokens(message_text(message)) for message in row_messages(row))


def filter_jsonl_by_budget(
    source: Path,
    accepted: Path,
    rejected: Path,
    count_tokens: TokenCounter,
    *,
    max_tokens: int,
    progress_every: int = 100,
) -> dict[str, Any]:
    if progress_every <= 0:
        raise ValueError("progress_every must be positive")
    accepted.parent.mkdir(parents=True, exist_ok=True)
    rejected.parent.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, Any] = {
        "source": str(source),
        "accepted": str(accepted),
        "rejected": str(rejected),
        "max_tokens": max_tokens,
        "rows_read": 0,
        "rows_accepted": 0,
        "rows_rejected": 0,
        "malformed_rows": 0,
        "max_accepted_tokens": 0,
        "min_rejected_tokens": 0,
    }

    min_rejected: int | None = None
    total_bytes = source.stat().st_size
    progress = EtaProgress("filter-token-budget", total=total_bytes)
    bytes_read = 0
    with source.open(encoding="utf-8") as src, accepted.open("w", encoding="utf-8") as acc, rejected.open(
        "w", encoding="utf-8"
    ) as rej:
        for line_number, line in enumerate(src, start=1):
            bytes_read += len(line.encode("utf-8"))
            stripped = line.strip()
            if not stripped:
                continue
            try:
                row = json.loads(stripped)
            except json.JSONDecodeError:
                manifest["malformed_rows"] += 1
                continue
            if not isinstance(row, dict):
                manifest["malformed_rows"] += 1
                continue
            manifest["rows_read"] += 1
            tokens = row_token_count(row, count_tokens)
            if tokens <= max_tokens:
                acc.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
                manifest["rows_accepted"] += 1
                manifest["max_accepted_tokens"] = max(manifest["max_accepted_tokens"], tokens)
            else:
                rej.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
                manifest["rows_rejected"] += 1
                min_rejected = tokens if min_rejected is None else min(min_rejected, tokens)
            if line_number % progress_every == 0:
                print(progress.update(min(bytes_read, total_bytes)), flush=True)

    manifest["min_rejected_tokens"] = min_rejected or 0
    print(progress.update(total_bytes), flush=True)
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--in", dest="source", required=True, type=Path)
    parser.add_argument("--accepted", required=True, type=Path)
    parser.add_argument("--rejected", required=True, type=Path)
    parser.add_argument("--max-tokens", required=True, type=int)
    parser.add_argument("--tokenizer", help="HF tokenizer path or model id.")
    parser.add_argument("--approximate", action="store_true")
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--progress-every", type=int, default=100)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.progress_every <= 0:
        raise SystemExit("--progress-every must be positive")
    count_tokens = load_token_counter(args.tokenizer, args.approximate)
    manifest = filter_jsonl_by_budget(
        args.source,
        args.accepted,
        args.rejected,
        count_tokens,
        max_tokens=args.max_tokens,
        progress_every=args.progress_every,
    )
    if args.manifest:
        args.manifest.parent.mkdir(parents=True, exist_ok=True)
        args.manifest.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
