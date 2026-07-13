#!/usr/bin/env python3
"""Try and validate extra compaction variants for agentic SFT JSONL corpora."""
from __future__ import annotations

import argparse
from collections import Counter
import copy
import hashlib
import json
from pathlib import Path
import re
from typing import Any, Callable

from phaseD_sft.compact_observations import compact_row, is_observation_message

Transform = Callable[[dict[str, Any]], tuple[list[dict[str, Any]], Counter[str]]]

_BOILERPLATE_RE = re.compile(
    r"(?im)^\s*(?:great!?|now,?\s+|okay,?\s+|let'?s\s+(?:now\s+)?)"
    r"(?:look|inspect|check|run|create|modify|update|examine|test)\b[^\n]{0,160}\n+"
)


def assistant_fingerprint(rows: list[dict[str, Any]]) -> str:
    digest = hashlib.sha256()
    for row in rows:
        for message in row.get("messages", []):
            if isinstance(message, dict) and message.get("role") == "assistant":
                digest.update(str(message.get("content", "")).encode("utf-8", errors="replace"))
                digest.update(b"\0")
    return digest.hexdigest()


def validate_jsonl(path: Path) -> dict[str, Any]:
    digest = hashlib.sha256()
    rows = 0
    malformed = 0
    messages = 0
    assistant_messages = 0
    observation_messages = 0
    missing_preamble = 0

    with path.open(encoding="utf-8") as fh:
        for line in fh:
            stripped = line.strip()
            if not stripped:
                continue
            try:
                row = json.loads(stripped)
            except json.JSONDecodeError:
                malformed += 1
                continue
            if not isinstance(row, dict):
                malformed += 1
                continue
            rows += 1
            row_messages = row.get("messages")
            if not isinstance(row_messages, list) or len(row_messages) < 2:
                missing_preamble += 1
                continue
            if row_messages[0].get("role") != "system" or row_messages[1].get("role") != "user":
                missing_preamble += 1
            messages += len(row_messages)
            for message in row_messages:
                if not isinstance(message, dict):
                    continue
                if message.get("role") == "assistant":
                    assistant_messages += 1
                    digest.update(str(message.get("content", "")).encode("utf-8", errors="replace"))
                    digest.update(b"\0")
                if is_observation_message(message):
                    observation_messages += 1

    return {
        "path": str(path),
        "rows": rows,
        "malformed_rows": malformed,
        "messages": messages,
        "assistant_messages": assistant_messages,
        "observation_messages": observation_messages,
        "missing_preamble_rows": missing_preamble,
        "assistant_sha256": digest.hexdigest(),
    }


def transform_observation_cap(row: dict[str, Any], *, max_chars: int) -> tuple[dict[str, Any], Counter[str]]:
    before = assistant_fingerprint([row])
    compacted, stats = compact_row(row, max_chars=max_chars, head_lines=8, tail_lines=6, important_lines=24)
    stats["assistant_changed"] = int(before != assistant_fingerprint([compacted]))
    return compacted, stats


def transform_read_receipts(row: dict[str, Any]) -> tuple[dict[str, Any], Counter[str]]:
    compacted = copy.deepcopy(row)
    stats: Counter[str] = Counter()
    messages = compacted.get("messages")
    if not isinstance(messages, list):
        return compacted, stats
    for index, message in enumerate(messages):
        if not isinstance(message, dict) or not is_observation_message(message):
            continue
        content = message.get("content")
        if not isinstance(content, str):
            continue
        if _is_verify_or_edit_observation(content):
            continue
        if index + 1 < len(messages) and isinstance(messages[index + 1], dict) and messages[index + 1].get("role") == "assistant":
            stats["nonterminal_read_receipts"] += 1
        message["content"] = f"OBSERVATION:\n[read-only observation omitted: {len(content)} chars]"
        stats["read_observations_replaced"] += 1
    return compacted, stats


def transform_assistant_boilerplate(row: dict[str, Any]) -> tuple[dict[str, Any], Counter[str]]:
    compacted = copy.deepcopy(row)
    stats: Counter[str] = Counter()
    for message in compacted.get("messages", []):
        if not isinstance(message, dict) or message.get("role") != "assistant":
            continue
        content = message.get("content")
        if not isinstance(content, str):
            continue
        changed = _BOILERPLATE_RE.sub("", content)
        if changed != content:
            message["content"] = changed
            stats["assistant_messages_changed"] += 1
            stats["assistant_chars_removed"] += len(content) - len(changed)
    return compacted, stats


def split_long_rows(row: dict[str, Any], *, max_chars: int = 64_000) -> tuple[list[dict[str, Any]], Counter[str]]:
    messages = row.get("messages")
    stats: Counter[str] = Counter()
    if not isinstance(messages, list) or len(messages) <= 2:
        return [row], stats

    if _messages_chars(messages) <= max_chars:
        return [row], stats

    preamble_end = 0
    for message in messages:
        if isinstance(message, dict) and message.get("role") == "assistant":
            break
        preamble_end += 1
    preamble = messages[:preamble_end]
    body = messages[preamble_end:]
    if not body:
        return [row], stats
    chunks: list[list[dict[str, Any]]] = []
    current = copy.deepcopy(preamble)
    current_chars = _messages_chars(current)

    for group in _assistant_turn_groups(body):
        group_chars = _messages_chars(group)
        if len(current) > len(preamble) and current_chars + group_chars > max_chars:
            chunks.append(current)
            current = copy.deepcopy(preamble)
            current_chars = _messages_chars(current)
        current.extend(copy.deepcopy(group))
        current_chars += group_chars

    if len(current) > len(preamble):
        chunks.append(current)

    if len(chunks) <= 1:
        return [row], stats

    rows = []
    row_id = row.get("id") or row.get("instance_id") or "row"
    for index, chunk in enumerate(chunks, start=1):
        split_row = {key: copy.deepcopy(value) for key, value in row.items() if key != "messages"}
        split_row["messages"] = chunk
        split_row["split_parent_id"] = row_id
        split_row["split_index"] = index
        split_row["split_count"] = len(chunks)
        rows.append(split_row)

    stats["rows_split"] = 1
    stats["split_rows_created"] = len(rows)
    return rows, stats


def run_variant(source: Path, output: Path, variant: str, *, split_max_chars: int = 64_000) -> dict[str, Any]:
    def obs_cap(max_chars: int) -> Transform:
        def apply(row: dict[str, Any]) -> tuple[list[dict[str, Any]], Counter[str]]:
            compacted, stats = transform_observation_cap(row, max_chars=max_chars)
            return [compacted], stats

        return apply

    def single(fn: Callable[[dict[str, Any]], tuple[dict[str, Any], Counter[str]]]) -> Transform:
        def apply(row: dict[str, Any]) -> tuple[list[dict[str, Any]], Counter[str]]:
            transformed, stats = fn(row)
            return [transformed], stats

        return apply

    transforms: dict[str, Transform] = {
        "obs_cap400": obs_cap(400),
        "obs_cap200": obs_cap(200),
        "read_receipts": single(transform_read_receipts),
        "assistant_boilerplate": single(transform_assistant_boilerplate),
        "episode_split": lambda row: split_long_rows(row, max_chars=split_max_chars),
    }
    if variant not in transforms:
        raise ValueError(f"unknown variant: {variant}")

    before = validate_jsonl(source)
    stats: Counter[str] = Counter()
    output.parent.mkdir(parents=True, exist_ok=True)

    with source.open(encoding="utf-8") as src, output.open("w", encoding="utf-8") as out:
        for line in src:
            stripped = line.strip()
            if not stripped:
                continue
            try:
                row = json.loads(stripped)
            except json.JSONDecodeError:
                stats["malformed_rows"] += 1
                continue
            if not isinstance(row, dict):
                stats["malformed_rows"] += 1
                continue
            transformed_rows, row_stats = transforms[variant](row)
            stats.update(row_stats)
            stats["rows_read"] += 1
            for transformed in transformed_rows:
                out.write(json.dumps(transformed, ensure_ascii=False, separators=(",", ":")) + "\n")
                stats["rows_written"] += 1

    after = validate_jsonl(output)
    manifest = {
        "variant": variant,
        "source": str(source),
        "output": str(output),
        "stats": dict(stats),
        "before_validation": before,
        "after_validation": after,
        "assistant_preserved": before["assistant_sha256"] == after["assistant_sha256"],
        "preamble_not_worse": after["missing_preamble_rows"] <= before["missing_preamble_rows"],
        "jsonl_valid": after["malformed_rows"] == 0,
    }
    return manifest


def _messages_chars(messages: list[Any]) -> int:
    total = 0
    for message in messages:
        if isinstance(message, dict):
            total += len(str(message.get("content", "")))
        else:
            total += len(str(message))
    return total


def _assistant_turn_groups(messages: list[Any]) -> list[list[Any]]:
    groups: list[list[Any]] = []
    current: list[Any] = []
    for message in messages:
        if isinstance(message, dict) and message.get("role") == "assistant":
            if current:
                groups.append(current)
            current = [message]
        else:
            if not current:
                groups.append([message])
            else:
                current.append(message)
    if current:
        groups.append(current)
    return groups


def _is_verify_or_edit_observation(content: str) -> bool:
    lowered = content.lower()
    return (
        "traceback" in lowered
        or "failed" in lowered
        or "passed" in lowered
        or "error" in lowered
        or "has been edited" in lowered
        or "file created successfully" in lowered
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--in", dest="source", required=True, type=Path)
    parser.add_argument("--out", dest="output", required=True, type=Path)
    parser.add_argument("--variant", required=True)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--split-max-chars", type=int, default=64_000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = run_variant(args.source, args.output, args.variant, split_max_chars=args.split_max_chars)
    if args.manifest:
        args.manifest.parent.mkdir(parents=True, exist_ok=True)
        args.manifest.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
