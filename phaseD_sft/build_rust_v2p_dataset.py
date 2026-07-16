#!/usr/bin/env python3
"""Build the streaming, v1-novel Rust SFT v2p corpus under a Gemma token cap."""
from __future__ import annotations

import argparse
import hashlib
import heapq
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

from phaseD_sft.build_rust_sft import _norm, _synth_thought
from phaseD_sft.filter_token_budget import filter_jsonl_by_budget
from phaseD_sft.token_audit import load_token_counter


ASSERT_BUCKETS = ("1-4", "5-10", "11-20", "21+")
DEFAULT_QUOTAS = {"1-4": 750, "5-10": 1750, "11-20": 1750, "21+": 750}
INSTRUCTION = (
    "Write idiomatic, correct Rust to solve the following problem. "
    "Think step by step, then provide the implementation."
)


def assert_bucket(n_asserts: int) -> str:
    if n_asserts <= 4:
        return "1-4"
    if n_asserts <= 10:
        return "5-10"
    if n_asserts <= 20:
        return "11-20"
    return "21+"


def problem_key(row: dict[str, Any]) -> str:
    return _norm(str(row.get("problem") or ""))[:400]


def solution_key(row: dict[str, Any]) -> str:
    return _norm(str(row.get("solution") or ""))


def _rank(seed: int, row_id: str) -> int:
    digest = hashlib.sha256(f"{seed}\0{row_id}".encode()).digest()
    return int.from_bytes(digest, "big")


def _push_ranked(heap: list[tuple[int, str, dict[str, Any]]], item: tuple[int, str, dict[str, Any]], limit: int) -> None:
    """Keep the lowest deterministic ranks using a max-rank-at-root heap."""
    rank, row_id, row = item
    entry = (-rank, row_id, row)
    if len(heap) < limit:
        heapq.heappush(heap, entry)
    elif entry[0] > heap[0][0]:
        heapq.heapreplace(heap, entry)


def select_streaming_rows(
    rows: Iterable[dict[str, Any]],
    *,
    v1_ids: set[str],
    v1_problem_keys: set[str],
    v1_solution_keys: set[str],
    quotas: dict[str, int],
    seed: int,
    oversample: int = 2,
) -> tuple[list[dict[str, Any]], Counter[str]]:
    """Select deterministic, v1-novel rows without retaining the source bank."""
    if tuple(quotas) != ASSERT_BUCKETS:
        raise ValueError(f"quotas must use exactly {ASSERT_BUCKETS}")
    if oversample < 1:
        raise ValueError("oversample must be positive")

    stats: Counter[str] = Counter()
    heaps = {bucket: [] for bucket in ASSERT_BUCKETS}
    for row in rows:
        stats["rows_read"] += 1
        row_id = str(row.get("id") or "")
        pkey = problem_key(row)
        skey = solution_key(row)
        try:
            n_asserts = int(row.get("n_asserts"))
        except (TypeError, ValueError):
            n_asserts = 0
        if not row_id or not pkey or not skey or n_asserts <= 0:
            stats["invalid"] += 1
            continue
        if row_id in v1_ids:
            stats["excluded_v1_id"] += 1
            continue
        if pkey in v1_problem_keys:
            stats["excluded_v1_problem"] += 1
            continue
        if skey in v1_solution_keys:
            stats["excluded_v1_solution"] += 1
            continue
        bucket = assert_bucket(n_asserts)
        stats[f"eligible_{bucket}"] += 1
        _push_ranked(
            heaps[bucket],
            (_rank(seed, row_id), row_id, row),
            quotas[bucket] * oversample,
        )

    selected: list[dict[str, Any]] = []
    seen_problem: set[str] = set()
    seen_solution: set[str] = set()
    for bucket in ASSERT_BUCKETS:
        kept = 0
        for neg_rank, _, row in sorted(heaps[bucket], key=lambda item: -item[0]):
            pkey, skey = problem_key(row), solution_key(row)
            if pkey in seen_problem or skey in seen_solution:
                stats["dedup_selected"] += 1
                continue
            selected.append(row)
            seen_problem.add(pkey)
            seen_solution.add(skey)
            kept += 1
            if kept == quotas[bucket]:
                break
        stats[f"selected_{bucket}"] = kept
        if kept != quotas[bucket]:
            raise RuntimeError(
                f"only selected {kept}/{quotas[bucket]} rows for {bucket}; "
                "increase oversample or inspect source quality"
            )
    stats["selected"] = len(selected)
    return selected, stats


def _v1_problem_and_solution_keys(v1_dataset: Path) -> tuple[set[str], set[str]]:
    """Recover the keys rendered into rust_sft_v1 without assuming source IDs exist."""
    from datasets import load_from_disk

    problems: set[str] = set()
    solutions: set[str] = set()
    for row in load_from_disk(str(v1_dataset)):
        messages = row.get("messages") or []
        user = next((str(msg.get("content") or "") for msg in messages if msg.get("role") == "user"), "")
        assistant = next((str(msg.get("content") or "") for msg in messages if msg.get("role") == "assistant"), "")
        problem = user.rsplit("\n\n", 1)[-1]
        code = re.search(r"```rust\n(.*?)\n```", assistant, re.DOTALL)
        if problem:
            problems.add(_norm(problem)[:400])
        if code:
            solutions.add(_norm(code.group(1)))
    return problems, solutions


def _v1_source_ids(snapshot: Path) -> set[str]:
    ids: set[str] = set()
    with snapshot.open(encoding="utf-8") as source:
        for line in source:
            if line.strip():
                row = json.loads(line)
                row_id = str(row.get("id") or "")
                if row_id:
                    ids.add(row_id)
    return ids


def render_training_row(row: dict[str, Any]) -> dict[str, Any]:
    problem = str(row["problem"]).strip()
    solution = str(row["solution"]).strip()
    assistant = (
        f"<thought>\n{_synth_thought(problem, solution)}\n</thought>\n\n"
        f"Here is the Rust implementation:\n\n```rust\n{solution}\n```"
    )
    return {
        "messages": [
            {"role": "user", "content": f"{INSTRUCTION}\n\n{problem}"},
            {"role": "assistant", "content": assistant},
        ],
        "n_asserts": int(row["n_asserts"]),
    }


def _stream_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open(encoding="utf-8") as source:
        for line in source:
            if line.strip():
                yield json.loads(line)


def build(args: argparse.Namespace) -> dict[str, Any]:
    if args.out.exists():
        raise FileExistsError(f"refusing to overwrite dataset: {args.out}")
    if args.work_dir.exists():
        raise FileExistsError(f"refusing to overwrite work directory: {args.work_dir}")

    v1_ids = _v1_source_ids(args.v1_snapshot)
    v1_problem_keys, v1_solution_keys = _v1_problem_and_solution_keys(args.v1_dataset)
    selected, stats = select_streaming_rows(
        _stream_jsonl(args.source),
        v1_ids=v1_ids,
        v1_problem_keys=v1_problem_keys,
        v1_solution_keys=v1_solution_keys,
        quotas=DEFAULT_QUOTAS,
        seed=args.seed,
    )

    args.work_dir.mkdir(parents=True)
    selected_path = args.work_dir / "selected_unbudgeted.jsonl"
    accepted_path = args.work_dir / "accepted_49152.jsonl"
    rejected_path = args.work_dir / "rejected_49152.jsonl"
    with selected_path.open("w", encoding="utf-8") as target:
        for row in selected:
            target.write(json.dumps(render_training_row(row), ensure_ascii=False) + "\n")

    count_tokens = load_token_counter(args.tokenizer, approximate=False)
    budget = filter_jsonl_by_budget(
        selected_path,
        accepted_path,
        rejected_path,
        count_tokens,
        max_tokens=args.max_tokens,
        progress_every=args.progress_every,
    )
    if budget["rows_accepted"] == 0:
        raise RuntimeError("token budget rejected every selected row")

    from datasets import Dataset

    with accepted_path.open(encoding="utf-8") as source:
        final_rows = [json.loads(line) for line in source if line.strip()]
    Dataset.from_list(final_rows).save_to_disk(str(args.out))
    manifest = {
        "source": str(args.source),
        "v1_snapshot": str(args.v1_snapshot),
        "v1_dataset": str(args.v1_dataset),
        "seed": args.seed,
        "quotas": DEFAULT_QUOTAS,
        "selection": dict(stats),
        "v1_key_counts": {
            "ids": len(v1_ids),
            "normalized_problems": len(v1_problem_keys),
            "normalized_solutions": len(v1_solution_keys),
        },
        "mswe_rust_train_rows": 0,
        "repair_rows": 0,
        "budget": budget,
        "final_dataset": str(args.out),
        "final_rows": len(final_rows),
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("data/rlvr_rust_verified.jsonl"))
    parser.add_argument("--v1-snapshot", type=Path, default=Path("data/rlvr_rust_verified_snap1.jsonl"))
    parser.add_argument("--v1-dataset", type=Path, default=Path("data/rust_sft_v1"))
    parser.add_argument("--out", type=Path, default=Path("data/rust_sft_v2p_5k"))
    parser.add_argument("--work-dir", type=Path, default=Path("data/rust_v2p_5k_build"))
    parser.add_argument("--manifest", type=Path, default=Path("data/rust_v2p_5k_manifest.json"))
    parser.add_argument("--tokenizer", default="/media/ironbcc/CrucialX10/models/google/gemma-4-31B-it")
    parser.add_argument("--max-tokens", type=int, default=49152)
    parser.add_argument("--seed", type=int, default=20260714)
    parser.add_argument("--progress-every", type=int, default=250)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifest = build(args)
    print(json.dumps(manifest, indent=2, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
