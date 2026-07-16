#!/usr/bin/env python3
"""Build fixture-prioritized edit-adjacent prompts for verified-reward GRPO."""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import glob
import json
from pathlib import Path
import subprocess
from typing import Any

from phaseD_sft.progress import EtaProgress
from phaseE_rl.build_swe_decision_dataset import (
    Candidate,
    _content_hash,
    _rendered_tokens,
    candidates_from_sft_row,
    candidates_from_trajectory,
    extract_edit_adjacent_point,
    load_sft_rows,
    source_group,
)


FIXTURE_GROUPS = ("swe-smith", "kwai", "smoke")
EXTERNAL_GROUPS = ("oracle", "openswe")


def _sort_key(item: tuple[Candidate, int, dict[str, Any] | None]) -> tuple[Any, ...]:
    candidate, tokens, _ = item
    return tokens, candidate.row["source"], candidate.row["instance_id"]


def select_v3_rows(
    accepted: list[tuple[Candidate, int, dict[str, Any] | None]],
    *,
    target: int,
    fixture_caps: dict[str, int],
    external_caps: dict[str, int],
) -> list[tuple[Candidate, int, dict[str, Any] | None]]:
    """Prefer exact fixture-backed edits; reserve <=20% for diversity sources."""
    if target <= 0:
        raise ValueError("target must be positive")
    external_limit = target // 5
    if sum(external_caps.values()) > external_limit:
        raise ValueError("external caps exceed the 20% diversity ceiling")

    selected: list[tuple[Candidate, int, dict[str, Any] | None]] = []
    chosen: set[str] = set()

    def take(items: list[tuple[Candidate, int, dict[str, Any] | None]], limit: int) -> None:
        for item in sorted(items, key=_sort_key):
            digest = _content_hash(item[0].row["messages"])
            if len(selected) >= target or digest in chosen:
                continue
            selected.append(item)
            chosen.add(digest)
            if sum(1 for candidate, _, _ in selected if source_group(candidate) in EXTERNAL_GROUPS) >= external_limit:
                # Further external rows are blocked by the outer group selection.
                pass
            if len([1 for candidate, _, _ in selected if source_group(candidate) == source_group(item[0])]) >= limit:
                break

    for group in FIXTURE_GROUPS:
        fixture_backed = [
            item for item in accepted
            if source_group(item[0]) == group and bool(item[2] and item[2].get("verifiable"))
        ]
        take(fixture_backed, fixture_caps[group])

    # If one preferred source has fewer rows, fill from the other fixture-backed
    # sources before introducing oracle/OpenSWE prompts.
    fixture_backfill = [
        item for item in accepted
        if source_group(item[0]) in FIXTURE_GROUPS and bool(item[2] and item[2].get("verifiable"))
    ]
    take(fixture_backfill, target)

    for group in EXTERNAL_GROUPS:
        if len(selected) >= target:
            break
        external = [item for item in accepted if source_group(item[0]) == group]
        take(external, external_caps[group])

    return selected[:target]


def cap_external_share(
    selected: list[tuple[Candidate, int, dict[str, Any] | None]],
) -> list[tuple[Candidate, int, dict[str, Any] | None]]:
    """Keep external diversity at <=20% after an honest fixture shortfall."""
    fixture_rows = [item for item in selected if source_group(item[0]) not in EXTERNAL_GROUPS]
    external_rows = [item for item in selected if source_group(item[0]) in EXTERNAL_GROUPS]
    return fixture_rows + external_rows[: len(fixture_rows) // 4]


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _fixture_indexes(rows: list[dict[str, Any]]) -> tuple[dict[tuple[str, str], dict[str, Any]], dict[str, dict[str, Any]]]:
    exact = {(str(row["source"]), str(row["instance_id"])): row for row in rows}
    smoke: dict[str, dict[str, Any]] = {}
    for row in rows:
        if str(row["source"]).startswith("smoke:") and row.get("verifiable"):
            smoke.setdefault(str(row["instance_id"]), row)
    return exact, smoke


def _fixture_for(candidate: Candidate, exact: dict[tuple[str, str], dict[str, Any]], smoke: dict[str, dict[str, Any]]) -> dict[str, Any] | None:
    source = str(candidate.row["source"])
    instance_id = str(candidate.row["instance_id"])
    if source.startswith("smoke:"):
        return smoke.get(instance_id)
    return exact.get((source, instance_id))


def _local_image_names(cache_index: Path) -> set[str]:
    listed = subprocess.run(
        ["docker", "images", "-a", "--format", "{{.Repository}}:{{.Tag}}"],
        text=True,
        capture_output=True,
        check=True,
    ).stdout.splitlines()
    names = {name for name in listed if name and name != "<none>:<none>"}
    if cache_index.exists():
        index = json.loads(cache_index.read_text(encoding="utf-8"))
        names.update(str(meta["image_name"]) for meta in index.values() if meta.get("image_name"))
    return names


def _sidecar_record(decision: dict[str, Any], fixture: dict[str, Any] | None, local_images: set[str]) -> dict[str, Any]:
    source, instance_id = str(decision["source"]), str(decision["instance_id"])
    if fixture is None or not fixture.get("verifiable"):
        return {
            "source": source, "instance_id": instance_id, "verifiable": False, "unverified": True,
            "fixture_source": None, "image_name": None, "image_local": False, "f2p": [], "test_patch": "",
        }
    image_name = str(fixture["image_name"])
    return {
        **fixture,
        "source": source,
        "instance_id": instance_id,
        "image_local": image_name in local_images,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sft-data", type=Path, required=True)
    parser.add_argument("--trajectory-glob", action="append", required=True)
    parser.add_argument("--fixture-sidecar", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--fixtures-out", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--target", type=int, default=768)
    parser.add_argument("--max-tokens", type=int, default=3072)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--cache-index", type=Path, default=Path("runs/rlvr_state_cache/index.json"))
    parser.add_argument(
        "--allow-under-target",
        action="store_true",
        help="Write the verified selection when fixture eligibility is honestly below the requested target.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.target < 600 or args.target > 800 or args.max_tokens <= 0 or args.workers <= 0:
        raise SystemExit("target must be 600..800; max-tokens and workers must be positive")
    paths = sorted({Path(path) for pattern in args.trajectory_glob for path in glob.glob(pattern)})
    if not paths:
        raise SystemExit("no trajectory files matched")
    extraction: Counter[str] = Counter()
    candidates: list[Candidate] = []
    for path in paths:
        candidate, reason = candidates_from_trajectory(path, extract_edit_adjacent_point)
        extraction[f"smoke:{reason}"] += 1
        if candidate:
            candidates.append(candidate)
    for row in load_sft_rows(args.sft_data):
        candidate, reason = candidates_from_sft_row(row, extract_edit_adjacent_point)
        extraction[f"sft:{reason}"] += 1
        if candidate:
            candidates.append(candidate)

    deduped: list[Candidate] = []
    seen: set[str] = set()
    for candidate in candidates:
        digest = _content_hash(candidate.row["messages"])
        if digest not in seen:
            seen.add(digest)
            deduped.append(candidate)

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)

    def measure(candidate: Candidate) -> tuple[Candidate, int | None, str | None]:
        try:
            return candidate, _rendered_tokens(tokenizer, candidate.row["messages"]), None
        except Exception as exc:
            return candidate, None, type(exc).__name__

    progress = EtaProgress("rlvr-v3-budget", total=len(deduped))
    with ThreadPoolExecutor(max_workers=args.workers, thread_name_prefix="rlvr-v3") as executor:
        measured = list(executor.map(measure, deduped))
    accepted: list[tuple[Candidate, int]] = []
    budget_stats: Counter[str] = Counter()
    for completed, (candidate, tokens, error) in enumerate(measured, start=1):
        if error:
            budget_stats[f"render_error:{error}"] += 1
        elif tokens is not None and tokens <= args.max_tokens:
            accepted.append((candidate, tokens))
        else:
            budget_stats["over_budget"] += 1
        if completed % 100 == 0 or completed == len(measured):
            print(progress.update(completed), flush=True)

    exact, smoke = _fixture_indexes(_load_jsonl(args.fixture_sidecar))
    with_fixtures = [(candidate, tokens, _fixture_for(candidate, exact, smoke)) for candidate, tokens in accepted]
    selected = select_v3_rows(
        with_fixtures,
        target=args.target,
        fixture_caps={"swe-smith": 320, "kwai": 192, "smoke": 128},
        external_caps={"oracle": 64, "openswe": 64},
    )
    if args.allow_under_target and len(selected) < args.target:
        selected = cap_external_share(selected)
    local_images = _local_image_names(args.cache_index)
    decisions = [candidate.row for candidate, _, _ in selected]
    fixtures = [_sidecar_record(candidate.row, fixture, local_images) for candidate, _, fixture in selected]
    if len(decisions) != args.target and not args.allow_under_target:
        raise SystemExit(f"only selected {len(decisions)} rows; target was {args.target}")
    if sum(bool(row["verifiable"]) for row in fixtures) / len(fixtures) < 0.70:
        raise SystemExit("fixture-verifiable share is below required 70%")
    if sum(source_group(candidate) in EXTERNAL_GROUPS for candidate, _, _ in selected) > len(selected) // 5:
        raise SystemExit("external share exceeded the 20% cap")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    for path, rows in ((args.out, decisions), (args.fixtures_out, fixtures)):
        with path.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    tokens = [tokens for _, tokens, _ in selected]
    manifest = {
        "output": str(args.out), "fixtures_output": str(args.fixtures_out), "source_fixture_sidecar": str(args.fixture_sidecar),
        "sft_data": str(args.sft_data), "trajectory_globs": args.trajectory_glob, "trajectory_files": len(paths),
        "edit_cut": "before_assistant_command_immediately_preceding_first_source_edit", "tokenizer": args.tokenizer,
        "max_prompt_tokens": args.max_tokens, "workers": args.workers, "requested_target": args.target,
        "selected_shortfall": args.target - len(selected), "allow_under_target": args.allow_under_target,
        "candidates": len(candidates), "deduped": len(deduped), "duplicates_removed": len(candidates) - len(deduped),
        "budget_accepted": len(accepted), "budget_stats": dict(sorted(budget_stats.items())), "extraction": dict(sorted(extraction.items())),
        "selected": len(selected), "selected_by_group": dict(sorted(Counter(source_group(candidate) for candidate, _, _ in selected).items())),
        "selected_by_source": dict(sorted(Counter(candidate.row["source"] for candidate, _, _ in selected).items())),
        "fixture_verifiable": sum(bool(row["verifiable"]) for row in fixtures),
        "base_image_local": sum(bool(row["image_local"]) for row in fixtures),
        "prefix_token_stats": {"min": min(tokens), "p50": sorted(tokens)[(len(tokens)-1)//2], "max": max(tokens)},
        "selection_policy": "fixture_backed_swe-smith_kwai_smoke_first; oracle_openswe_diversity<=20%; shortest_prefix_with_deterministic_tiebreak",
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
