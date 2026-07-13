#!/usr/bin/env python3
"""Build the v8 edit-first mixture from budgeted external candidates and v6 anchors."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
from typing import Any, Callable, Iterable

from phaseD_sft.progress import EtaProgress
from phaseD_sft.token_audit import load_token_counter, message_text, row_messages


ANCHOR_SOURCES = (
    "swe-smith",
    "swe_train_oracle_edit_trace",
    "coder_repair_synthetic",
)
ORACLE_SOURCE = "swe_train_oracle_edit_trace"
TokenCounter = Callable[[str], int]


def instance_repo(row: dict[str, Any]) -> str:
    """Return the stable owner/repository prefix used by SWE-style instance ids."""
    instance_id = str(row.get("instance_id") or "")
    prefix = instance_id.split(".", 1)[0].strip()
    return prefix or f"missing:{instance_id}"


def row_token_count(row: dict[str, Any], count_tokens: TokenCounter) -> int:
    return sum(count_tokens(message_text(message)) for message in row_messages(row))


def _measure_rows(
    rows: Iterable[dict[str, Any]],
    count_tokens: TokenCounter,
    *,
    workers: int,
    progress_every: int,
) -> list[tuple[int, int, dict[str, Any]]]:
    indexed = list(enumerate(rows))
    progress = EtaProgress("v8-token-aware-select", total=len(indexed))

    def measure(item: tuple[int, dict[str, Any]]) -> tuple[int, int, dict[str, Any]]:
        index, row = item
        return index, row_token_count(row, count_tokens), row

    if workers == 1:
        results = map(measure, indexed)
    else:
        executor = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="v8-token-select")
        results = executor.map(measure, indexed)

    measured: list[tuple[int, int, dict[str, Any]]] = []
    try:
        for completed, item in enumerate(results, start=1):
            measured.append(item)
            if completed % progress_every == 0 or completed == len(indexed):
                print(progress.update(completed), flush=True)
    finally:
        if workers != 1:
            executor.shutdown(wait=True)
    return measured


def _token_stats(values: list[int]) -> dict[str, int]:
    if not values:
        return {"min": 0, "p50": 0, "max": 0}
    ordered = sorted(values)
    return {
        "min": ordered[0],
        "p50": ordered[(len(ordered) - 1) // 2],
        "max": ordered[-1],
    }


def select_externals(
    rows: list[dict[str, Any]],
    count_tokens: TokenCounter,
    *,
    external_cap: int,
    workers: int,
    progress_every: int = 100,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Select short rows while exhausting one trace per repository before repeats."""
    if external_cap <= 0:
        raise ValueError("external_cap must be positive")
    if workers <= 0:
        raise ValueError("workers must be positive")
    if progress_every <= 0:
        raise ValueError("progress_every must be positive")

    measured = _measure_rows(
        rows, count_tokens, workers=workers, progress_every=progress_every
    )
    by_repo: dict[str, list[tuple[int, int, dict[str, Any]]]] = defaultdict(list)
    for item in measured:
        by_repo[instance_repo(item[2])].append(item)
    for choices in by_repo.values():
        choices.sort(
            key=lambda item: (
                item[1],
                str(item[2].get("source") or ""),
                str(item[2].get("instance_id") or ""),
                item[0],
            )
        )

    selected: list[tuple[int, int, dict[str, Any]]] = []
    round_index = 0
    while len(selected) < min(external_cap, len(measured)):
        eligible = [choices[round_index] for choices in by_repo.values() if len(choices) > round_index]
        if not eligible:
            break
        eligible.sort(
            key=lambda item: (
                item[1],
                instance_repo(item[2]),
                str(item[2].get("source") or ""),
                str(item[2].get("instance_id") or ""),
                item[0],
            )
        )
        remaining = external_cap - len(selected)
        selected.extend(eligible[:remaining])
        round_index += 1

    rows_out = [item[2] for item in selected]
    selected_counts = Counter(str(row.get("source") or "MISSING") for row in rows_out)
    return rows_out, {
        "candidates": len(rows),
        "selected": len(rows_out),
        "external_cap": external_cap,
        "selection_policy": "one_per_instance_repo_before_repeats; shorter_token_count_first_within_round",
        "unique_repos_candidates": len(by_repo),
        "unique_repos_selected": len({instance_repo(row) for row in rows_out}),
        "selected_by_source": dict(sorted(selected_counts.items())),
        "candidate_token_stats": _token_stats([item[1] for item in measured]),
        "selected_token_stats": _token_stats([item[1] for item in selected]),
    }


def build_v8_mixture(
    externals: list[dict[str, Any]],
    anchors: list[dict[str, Any]],
    count_tokens: TokenCounter,
    *,
    external_cap: int = 2800,
    workers: int = 1,
    progress_every: int = 100,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Keep every v6 anchor once and add a token-aware diverse external selection."""
    unexpected = sorted({str(row.get("source") or "MISSING") for row in anchors} - set(ANCHOR_SOURCES))
    if unexpected:
        raise ValueError(f"anchor input contains unexpected sources: {unexpected}")
    anchor_counts = Counter(str(row.get("source") or "MISSING") for row in anchors)
    selected, selection = select_externals(
        externals,
        count_tokens,
        external_cap=external_cap,
        workers=workers,
        progress_every=progress_every,
    )
    mixture = list(anchors) + selected
    manifest = {
        "rows_in": {"anchors": len(anchors), "external_candidates": len(externals)},
        "rows_out_pre_dedup": len(mixture),
        "anchor_counts": dict(sorted(anchor_counts.items())),
        "oracle_edit_source": ORACLE_SOURCE,
        "oracle_edit_upweight_factor": 1,
        "dedup_passthrough_sources": list(ANCHOR_SOURCES),
        "external_selection": selection,
    }
    return mixture, manifest


def _load_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"non-object JSONL row at {path}:{line_number}")
            rows.append(row)
    return rows


def _write_rows(rows: list[dict[str, Any]], output: Path, *, progress_every: int) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    progress = EtaProgress("write-v8-mixture", total=len(rows))
    with output.open("w", encoding="utf-8") as handle:
        for index, row in enumerate(rows, start=1):
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
            if index % progress_every == 0 or index == len(rows):
                print(progress.update(index), flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--externals", required=True, type=Path)
    parser.add_argument("--anchors", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--external-cap", type=int, default=2800)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--progress-every", type=int, default=100)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    externals = _load_rows(args.externals)
    anchors = _load_rows(args.anchors)
    # The tokenizer is intentionally constructed once here; worker threads share it.
    count_tokens = load_token_counter(args.tokenizer, approximate=False)
    mixture, manifest = build_v8_mixture(
        externals,
        anchors,
        count_tokens,
        external_cap=args.external_cap,
        workers=args.workers,
        progress_every=args.progress_every,
    )
    _write_rows(mixture, args.out, progress_every=args.progress_every)
    manifest.update(
        {
            "externals": str(args.externals),
            "anchors": str(args.anchors),
            "output": str(args.out),
            "tokenizer": args.tokenizer,
            "workers": args.workers,
        }
    )
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
