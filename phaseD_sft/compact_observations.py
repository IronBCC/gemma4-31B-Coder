#!/usr/bin/env python3
"""Deterministically compact tool observations in agentic SFT traces."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
from typing import Any

from phaseD_sft.progress import EtaProgress
from phaseD_sft.token_audit import message_text, role_bucket

DEFAULT_MAX_OBSERVATION_CHARS = 2_400
DEFAULT_HEAD_LINES = 24
DEFAULT_TAIL_LINES = 16
DEFAULT_IMPORTANT_LINES = 48

_BASE64_RE = re.compile(r"(?<![A-Za-z0-9+/=])(?:[A-Za-z0-9+/]{160,}={0,2})(?![A-Za-z0-9+/=])")
_DATA_URI_RE = re.compile(r"data:[\w.+-]+/[\w.+-]+;base64,[A-Za-z0-9+/=\s]{160,}", re.IGNORECASE)
_IMPORTANT_RE = re.compile(
    r"error|exception|traceback|failed|failure|warning|warn|panic|assert|undefined|"
    r"not found|no such file|permission denied|denied|segmentation fault|oom|out of memory",
    re.IGNORECASE,
)


def is_observation_message(message: dict[str, Any]) -> bool:
    text = message_text(message)
    return role_bucket(message, text) == "tool"


def compact_jsonl(
    source: Path,
    output: Path,
    *,
    max_chars: int = DEFAULT_MAX_OBSERVATION_CHARS,
    head_lines: int = DEFAULT_HEAD_LINES,
    tail_lines: int = DEFAULT_TAIL_LINES,
    important_lines: int = DEFAULT_IMPORTANT_LINES,
    progress_every: int = 100,
) -> dict[str, Any]:
    if progress_every <= 0:
        raise ValueError("progress_every must be positive")
    totals: Counter[str] = Counter()
    output.parent.mkdir(parents=True, exist_ok=True)
    total_bytes = source.stat().st_size
    progress = EtaProgress("compact-observations", total=total_bytes)
    bytes_read = 0

    with source.open(encoding="utf-8") as src, output.open("w", encoding="utf-8") as out:
        for line_number, line in enumerate(src, start=1):
            bytes_read += len(line.encode("utf-8"))
            stripped = line.strip()
            if not stripped:
                continue
            try:
                row = json.loads(stripped)
            except json.JSONDecodeError:
                totals["malformed_rows"] += 1
                continue
            if not isinstance(row, dict):
                totals["malformed_rows"] += 1
                continue

            compacted, row_stats = compact_row(
                row,
                max_chars=max_chars,
                head_lines=head_lines,
                tail_lines=tail_lines,
                important_lines=important_lines,
            )
            totals.update(row_stats)
            totals["rows_read"] += 1
            out.write(json.dumps(compacted, ensure_ascii=False, separators=(",", ":")) + "\n")
            if line_number % progress_every == 0:
                print(progress.update(min(bytes_read, total_bytes)), flush=True)

    print(progress.update(total_bytes), flush=True)

    manifest = dict(totals)
    manifest.update(
        {
            "source": str(source),
            "output": str(output),
            "max_observation_chars": max_chars,
            "head_lines": head_lines,
            "tail_lines": tail_lines,
            "important_lines": important_lines,
            "char_reduction": totals["original_chars"] - totals["compacted_chars"],
            "char_reduction_share": round(
                (totals["original_chars"] - totals["compacted_chars"]) / totals["original_chars"], 4
            )
            if totals["original_chars"]
            else 0.0,
        }
    )
    return manifest


def compact_row(
    row: dict[str, Any],
    *,
    max_chars: int = DEFAULT_MAX_OBSERVATION_CHARS,
    head_lines: int = DEFAULT_HEAD_LINES,
    tail_lines: int = DEFAULT_TAIL_LINES,
    important_lines: int = DEFAULT_IMPORTANT_LINES,
) -> tuple[dict[str, Any], Counter[str]]:
    compacted = json.loads(json.dumps(row))
    stats: Counter[str] = Counter()
    seen_observations: dict[str, int] = {}

    messages = compacted.get("messages")
    if not isinstance(messages, list):
        return compacted, stats

    for index, message in enumerate(messages):
        if not isinstance(message, dict):
            continue
        content = message.get("content")
        if not isinstance(content, str) or not is_observation_message(message):
            if message.get("role") == "assistant":
                stats["assistant_messages_seen"] += 1
            continue

        stats["observation_messages_seen"] += 1
        stats["original_chars"] += len(content)
        content_hash = _hash_text(_normalize_newlines(content))
        if content_hash in seen_observations:
            first_index = seen_observations[content_hash]
            replacement = f"OBSERVATION:\n[unchanged observation omitted; first seen at message {first_index}]"
            message["content"] = replacement
            stats["duplicate_observations"] += 1
            stats["observation_messages_compacted"] += 1
            stats["compacted_chars"] += len(replacement)
            continue

        seen_observations[content_hash] = index
        new_content, changed, compact_stats = compact_observation_text(
            content,
            max_chars=max_chars,
            head_lines=head_lines,
            tail_lines=tail_lines,
            important_lines=important_lines,
        )
        message["content"] = new_content
        stats.update(compact_stats)
        if changed:
            stats["observation_messages_compacted"] += 1
        stats["compacted_chars"] += len(new_content)

    return compacted, stats


def compact_observation_text(
    text: str,
    *,
    max_chars: int = DEFAULT_MAX_OBSERVATION_CHARS,
    head_lines: int = DEFAULT_HEAD_LINES,
    tail_lines: int = DEFAULT_TAIL_LINES,
    important_lines: int = DEFAULT_IMPORTANT_LINES,
) -> tuple[str, bool, Counter[str]]:
    stats: Counter[str] = Counter()
    normalized = _normalize_newlines(text)
    redacted = _redact_base64(normalized, stats)
    redacted = _redact_binary(redacted, stats)
    changed = redacted != text

    if len(redacted) <= max_chars:
        return redacted, changed, stats

    compacted = _head_important_tail(
        redacted,
        max_chars=max_chars,
        head_lines=head_lines,
        tail_lines=tail_lines,
        important_lines=important_lines,
        stats=stats,
    )
    return compacted, True, stats


def _head_important_tail(
    text: str,
    *,
    max_chars: int,
    head_lines: int,
    tail_lines: int,
    important_lines: int,
    stats: Counter[str],
) -> str:
    lines = text.splitlines()
    head_indexes = set(range(min(head_lines, len(lines))))
    tail_start = max(0, len(lines) - tail_lines)
    tail_indexes = set(range(tail_start, len(lines)))
    important_indexes: set[int] = set()
    for index, line in enumerate(lines):
        if _IMPORTANT_RE.search(line):
            important_indexes.add(index)
            if len(important_indexes) >= important_lines:
                break

    keep = sorted(head_indexes | important_indexes | tail_indexes)
    chunks: list[str] = []
    omitted_lines = 0
    omitted_chars = 0
    previous = -1
    for index in keep:
        if index <= previous:
            continue
        if index > previous + 1:
            gap = lines[previous + 1 : index]
            omitted_lines += len(gap)
            omitted_chars += sum(len(line) + 1 for line in gap)
            chunks.append(f"[... omitted {len(gap)} lines / {sum(len(line) + 1 for line in gap)} chars ...]")
        chunks.append(lines[index])
        previous = index
    if previous < len(lines) - 1:
        gap = lines[previous + 1 :]
        omitted_lines += len(gap)
        omitted_chars += sum(len(line) + 1 for line in gap)
        chunks.append(f"[... omitted {len(gap)} lines / {sum(len(line) + 1 for line in gap)} chars ...]")

    compacted = "\n".join(chunks)
    if len(compacted) > max_chars:
        compacted = _trim_to_char_budget(compacted, max_chars)

    stats["long_observations_compacted"] += 1
    stats["omitted_lines"] += omitted_lines
    stats["omitted_chars"] += omitted_chars
    return compacted


def _trim_to_char_budget(text: str, max_chars: int) -> str:
    marker = "\n[... omitted by trim ...]\n"
    if max_chars <= len(marker) + 20:
        return text[:max_chars]

    lines = text.splitlines()
    important_lines = [line for line in lines if _IMPORTANT_RE.search(line)]
    if important_lines:
        head = lines[:4]
        tail = lines[-2:]
        while head:
            candidate = "\n".join(_dedup_lines([*head, marker.strip(), *important_lines, marker.strip(), *tail]))
            if len(candidate) <= max_chars:
                return candidate
            head.pop()

    head_budget = (max_chars - len(marker)) // 2
    tail_budget = max_chars - len(marker) - head_budget
    return text[:head_budget].rstrip() + marker + text[-tail_budget:].lstrip()


def _dedup_lines(lines: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for line in lines:
        if line in seen:
            continue
        seen.add(line)
        result.append(line)
    return result


def _redact_base64(text: str, stats: Counter[str]) -> str:
    def replace(match: re.Match[str]) -> str:
        stats["base64_replacements"] += 1
        return f"[base64 omitted: {len(match.group(0))} chars]"

    text = _DATA_URI_RE.sub(replace, text)
    return _BASE64_RE.sub(replace, text)


def _redact_binary(text: str, stats: Counter[str]) -> str:
    if "\x00" not in text:
        return text
    stats["binary_replacements"] += 1
    return text.replace("\x00", "[binary omitted]")


def _normalize_newlines(text: str) -> str:
    return text.replace("\r\n", "\n").replace("\r", "\n")


def _hash_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--in", dest="source", required=True, type=Path, help="Input JSONL trace file.")
    parser.add_argument("--out", dest="output", required=True, type=Path, help="Output compacted JSONL file.")
    parser.add_argument("--manifest", type=Path, help="Optional JSON manifest path.")
    parser.add_argument("--max-observation-chars", type=int, default=DEFAULT_MAX_OBSERVATION_CHARS)
    parser.add_argument("--head-lines", type=int, default=DEFAULT_HEAD_LINES)
    parser.add_argument("--tail-lines", type=int, default=DEFAULT_TAIL_LINES)
    parser.add_argument("--important-lines", type=int, default=DEFAULT_IMPORTANT_LINES)
    parser.add_argument("--progress-every", type=int, default=100)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.progress_every <= 0:
        raise SystemExit("--progress-every must be positive")
    manifest = compact_jsonl(
        args.source,
        args.output,
        max_chars=args.max_observation_chars,
        head_lines=args.head_lines,
        tail_lines=args.tail_lines,
        important_lines=args.important_lines,
        progress_every=args.progress_every,
    )
    if args.manifest:
        args.manifest.parent.mkdir(parents=True, exist_ok=True)
        args.manifest.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
