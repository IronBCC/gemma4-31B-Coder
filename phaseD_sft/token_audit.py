#!/usr/bin/env python3
"""Audit role-level token lengths for agentic SFT traces.

The audit is intentionally read-only for datasets. It measures how much of each
trace is assistant learning signal versus conditioning context such as user,
system, and tool-observation text.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
import math
from pathlib import Path
import re
from typing import Any, Callable, Iterable

TokenCounter = Callable[[str], int]

DEFAULT_THRESHOLDS = (8_192, 16_384, 24_576, 32_768, 65_536, 90_112)
ROLE_BUCKETS = ("assistant", "tool", "user", "system", "other")
_WORDLIKE = re.compile(r"\S+")
_OBSERVATION_PREFIX = re.compile(r"^\s*(observation|tool observation)\s*:", re.IGNORECASE)


def _stable_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _content_text(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                if isinstance(item.get("text"), str):
                    parts.append(item["text"])
                elif isinstance(item.get("content"), str):
                    parts.append(item["content"])
                else:
                    parts.append(_stable_json(item))
            else:
                parts.append(str(item))
        return "\n".join(part for part in parts if part)
    if isinstance(content, dict):
        return _stable_json(content)
    return str(content)


def message_text(message: dict[str, Any]) -> str:
    """Return countable text emitted by a single chat message."""

    chunks: list[str] = []
    reasoning = message.get("reasoning_content")
    if reasoning:
        chunks.append(_content_text(reasoning))
    content = _content_text(message.get("content"))
    if content:
        chunks.append(content)
    tool_calls = message.get("tool_calls")
    if tool_calls:
        chunks.append(_stable_json(tool_calls))
    function_call = message.get("function_call")
    if function_call:
        chunks.append(_stable_json(function_call))
    return "\n".join(chunks)


def normalize_role(role: Any) -> str:
    name = str(role or "other").lower()
    if name == "observation":
        return "tool"
    if name in ROLE_BUCKETS:
        return name
    return "other"


def role_bucket(message: dict[str, Any], text: str) -> str:
    role = normalize_role(message.get("role"))
    if role == "user" and _OBSERVATION_PREFIX.match(text):
        return "tool"
    return role


def row_messages(row: dict[str, Any]) -> Iterable[dict[str, Any]]:
    messages = row.get("messages")
    if isinstance(messages, list):
        for message in messages:
            if isinstance(message, dict):
                yield message
        return

    if isinstance(row.get("text"), str):
        yield {"role": "other", "content": row["text"]}
        return

    for key in ("prompt", "instruction", "problem", "input"):
        if isinstance(row.get(key), str):
            yield {"role": "user", "content": row[key]}
            return


def percentile(values: list[int], pct: int) -> int:
    if not values:
        return 0
    ordered = sorted(values)
    rank = max(1, math.ceil((pct / 100) * len(ordered)))
    return ordered[rank - 1]


def _series_stats(values: list[int]) -> dict[str, int | float]:
    if not values:
        return {"min": 0, "p50": 0, "p90": 0, "p95": 0, "p99": 0, "max": 0, "mean": 0.0}
    return {
        "min": min(values),
        "p50": percentile(values, 50),
        "p90": percentile(values, 90),
        "p95": percentile(values, 95),
        "p99": percentile(values, 99),
        "max": max(values),
        "mean": round(sum(values) / len(values), 2),
    }


def audit_jsonl(path: Path, count_tokens: TokenCounter, thresholds: Iterable[int] = DEFAULT_THRESHOLDS) -> dict[str, Any]:
    totals: list[int] = []
    role_totals: Counter[str] = Counter()
    role_series: dict[str, list[int]] = {role: [] for role in ROLE_BUCKETS}
    malformed_rows = 0
    empty_rows = 0
    longest: list[dict[str, Any]] = []

    with path.open(encoding="utf-8") as fh:
        for line_number, line in enumerate(fh, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                row = json.loads(stripped)
            except json.JSONDecodeError:
                malformed_rows += 1
                continue
            if not isinstance(row, dict):
                malformed_rows += 1
                continue

            row_counts = {role: 0 for role in ROLE_BUCKETS}
            for message in row_messages(row):
                text = message_text(message)
                if not text:
                    continue
                role = role_bucket(message, text)
                tokens = count_tokens(text)
                row_counts[role] += tokens

            total = sum(row_counts.values())
            if total == 0:
                empty_rows += 1
                continue

            totals.append(total)
            for role, tokens in row_counts.items():
                role_totals[role] += tokens
                role_series[role].append(tokens)

            trace_id = row.get("id") or row.get("trace_id") or row.get("uuid") or line_number
            longest.append({"id": trace_id, "line": line_number, "tokens": total, "roles": row_counts})
            longest = sorted(longest, key=lambda item: item["tokens"], reverse=True)[:20]

    grand_total = sum(totals)
    thresholds_report = {
        str(threshold): {
            "over": sum(1 for value in totals if value > threshold),
            "at_or_under": sum(1 for value in totals if value <= threshold),
        }
        for threshold in thresholds
    }

    roles = {}
    for role in ROLE_BUCKETS:
        total = role_totals[role]
        roles[role] = {
            "total": total,
            "share": round(total / grand_total, 4) if grand_total else 0.0,
            **_series_stats(role_series[role]),
        }

    return {
        "path": str(path),
        "rows": len(totals),
        "malformed_rows": malformed_rows,
        "empty_rows": empty_rows,
        "total_token_sum": grand_total,
        "total_tokens": _series_stats(totals),
        "roles": roles,
        "thresholds": thresholds_report,
        "longest": longest,
    }


def load_token_counter(tokenizer_name: str | None, approximate: bool) -> TokenCounter:
    if approximate:
        return lambda text: len(_WORDLIKE.findall(text))
    if not tokenizer_name:
        raise SystemExit("Pass --tokenizer for real counts, or --approximate for a smoke-only estimate.")
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=True)
    return lambda text: len(tokenizer.encode(text, add_special_tokens=False))


def format_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Agentic Trace Token Audit",
        "",
        f"- Path: `{report['path']}`",
        f"- Rows: {report['rows']}",
        f"- Malformed rows: {report['malformed_rows']}",
        f"- Empty rows: {report['empty_rows']}",
        f"- Total tokens: {report['total_token_sum']}",
        "",
        "## Total Lengths",
        "",
        "| min | p50 | p90 | p95 | p99 | max | mean |",
        "|---:|---:|---:|---:|---:|---:|---:|",
        _stats_row(report["total_tokens"]),
        "",
        "## Role Split",
        "",
        "| role | total | share | p50 | p95 | p99 | max |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for role in ROLE_BUCKETS:
        stats = report["roles"][role]
        lines.append(
            f"| {role} | {stats['total']} | {stats['share']:.2%} | {stats['p50']} | "
            f"{stats['p95']} | {stats['p99']} | {stats['max']} |"
        )

    lines.extend(["", "## Thresholds", "", "| threshold | over | at_or_under |", "|---:|---:|---:|"])
    for threshold, counts in report["thresholds"].items():
        lines.append(f"| {threshold} | {counts['over']} | {counts['at_or_under']} |")

    lines.extend(["", "## Longest Rows", "", "| rank | id | line | tokens | assistant | tool | user | system | other |"])
    lines.append("|---:|---|---:|---:|---:|---:|---:|---:|---:|")
    for index, item in enumerate(report["longest"][:10], start=1):
        roles = item["roles"]
        lines.append(
            f"| {index} | `{item['id']}` | {item['line']} | {item['tokens']} | "
            f"{roles['assistant']} | {roles['tool']} | {roles['user']} | {roles['system']} | {roles['other']} |"
        )
    return "\n".join(lines) + "\n"


def _stats_row(stats: dict[str, Any]) -> str:
    return (
        f"| {stats['min']} | {stats['p50']} | {stats['p90']} | {stats['p95']} | "
        f"{stats['p99']} | {stats['max']} | {stats['mean']} |"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, type=Path, help="JSONL trace file to audit.")
    parser.add_argument("--tokenizer", help="HF tokenizer path or model id for real token counts.")
    parser.add_argument("--approximate", action="store_true", help="Use whitespace counts for local smoke checks only.")
    parser.add_argument("--out-json", type=Path, help="Optional JSON report path.")
    parser.add_argument("--out-md", type=Path, help="Optional Markdown report path.")
    parser.add_argument(
        "--threshold",
        dest="thresholds",
        type=int,
        action="append",
        help="Context threshold to report. May be repeated. Defaults to common long-context cutoffs.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    count_tokens = load_token_counter(args.tokenizer, args.approximate)
    report = audit_jsonl(args.data, count_tokens, thresholds=args.thresholds or DEFAULT_THRESHOLDS)
    markdown = format_markdown(report)

    if args.out_json:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    if args.out_md:
        args.out_md.parent.mkdir(parents=True, exist_ok=True)
        args.out_md.write_text(markdown, encoding="utf-8")
    print(markdown)


if __name__ == "__main__":
    main()
