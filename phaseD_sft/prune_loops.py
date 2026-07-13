#!/usr/bin/env python3
"""Prune terminal low-value loops from compacted agentic SFT traces."""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
import sys
from typing import Any

from phaseD_sft.compact_observations import is_observation_message

_EDIT_RE = re.compile(r"\b(edited|created successfully|file created|apply_patch|search_replace|str_replace)\b", re.I)
_VERIFY_RE = re.compile(r"\b(pytest|unittest|run tests?|test suite|cargo test|make test|ctest|build)\b", re.I)
_VERIFY_OBS_RE = re.compile(r"\b(FAILED|passed|failed|Traceback|AssertionError|error:|FAILURES?|OK)\b")
_READ_RE = re.compile(r"\b(list|locate|search|grep|find|cat|read|inspect|examine|look at|view)\b", re.I)
_PASS_RE = re.compile(r"\b(\d+ passed|OK|all tests passed|no failures|success)\b", re.I)
_FAIL_LINE_RE = re.compile(r"(FAILED\s+\S+|FAIL:\s+\S+|ERROR:\s+\S+|AssertionError:?.*|Traceback.*|error:.*)", re.I)


@dataclass(frozen=True)
class TraceEvent:
    message_index: int
    tool_name: str
    args: str
    feedback_signature: str | None
    event_class: str
    verify_success: bool = False


def prune_jsonl(source: Path, output: Path) -> dict[str, Any]:
    totals: Counter[str] = Counter()
    output.parent.mkdir(parents=True, exist_ok=True)

    with source.open(encoding="utf-8") as src, output.open("w", encoding="utf-8") as out:
        for line in src:
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

            pruned, row_stats = prune_row(row)
            totals.update(row_stats)
            totals["rows_read"] += 1
            out.write(json.dumps(pruned, ensure_ascii=False, separators=(",", ":")) + "\n")

    manifest = dict(totals)
    manifest.update({"source": str(source), "output": str(output)})
    return manifest


def prune_row(row: dict[str, Any]) -> tuple[dict[str, Any], Counter[str]]:
    stats: Counter[str] = Counter()
    messages = row.get("messages")
    if not isinstance(messages, list):
        return row, stats

    events = extract_events(messages)
    if not events:
        return row, stats

    LoopAbortDetector = _load_loop_abort_detector()
    detector = LoopAbortDetector()
    candidate: tuple[int, str] | None = None
    recent_events: list[TraceEvent] = []

    for event in events:
        if _is_recovery_event(event):
            candidate = None
            detector = LoopAbortDetector()
            recent_events = []
            continue
        recent_events.append(event)
        detector.record(event.tool_name, event.args, feedback_signature=event.feedback_signature)
        abort, reason = detector.should_abort()
        if abort:
            start = _candidate_start_index(reason, recent_events, event)
            candidate = (start, reason)
            continue

    if not candidate:
        return row, stats

    cut_index, reason = candidate
    if cut_index < 2:
        return row, stats

    pruned_messages = messages[:cut_index]
    if len(pruned_messages) == len(messages):
        return row, stats

    compacted = dict(row)
    compacted["messages"] = pruned_messages
    stats["rows_pruned"] = 1
    stats["messages_pruned"] = len(messages) - len(pruned_messages)
    stats[reason] += 1
    return compacted, stats


def extract_events(messages: list[Any]) -> list[TraceEvent]:
    events: list[TraceEvent] = []
    last_assistant: tuple[int, str] | None = None
    for index, message in enumerate(messages):
        if not isinstance(message, dict):
            continue
        role = str(message.get("role") or "")
        content = message.get("content")
        if not isinstance(content, str):
            continue
        if role == "assistant":
            last_assistant = (index, content)
            continue
        if is_observation_message(message):
            assistant_index, assistant_text = last_assistant or (index, "")
            events.append(classify_event(assistant_text, content, message_index=assistant_index))
            last_assistant = None
    return events


def classify_event(assistant_text: str, observation_text: str, *, message_index: int = 0) -> TraceEvent:
    combined = f"{assistant_text}\n{observation_text}"
    if _EDIT_RE.search(combined):
        return TraceEvent(message_index, "edit", _short_args(combined), None, "mutate")

    is_verify = bool(
        _VERIFY_RE.search(assistant_text)
        or _VERIFY_OBS_RE.search(observation_text)
        or _PASS_RE.search(observation_text)
    )
    if is_verify:
        success = bool(_PASS_RE.search(observation_text)) and not _FAIL_LINE_RE.search(observation_text)
        signature = None if success else _failure_signature(observation_text)
        return TraceEvent(message_index, "run_tests", _short_args(assistant_text), signature, "verify", success)

    if _READ_RE.search(assistant_text):
        return TraceEvent(message_index, "read_file", _short_args(combined), None, "read")

    if _looks_like_file_listing(observation_text):
        return TraceEvent(message_index, "ls", _short_args(observation_text), None, "read")

    return TraceEvent(message_index, "read_file", _short_args(combined), None, "read")


def _candidate_start_index(reason: str, events: list[TraceEvent], current: TraceEvent) -> int:
    if reason == "loop:no-mutation":
        return events[-6].message_index if len(events) >= 6 else current.message_index
    if reason == "loop:no-fix-progress":
        verify_events = [event for event in events if event.feedback_signature][-3:]
        return verify_events[0].message_index if len(verify_events) == 3 else current.message_index
    if reason == "loop:exact-cycle":
        return events[-3].message_index if len(events) >= 3 else current.message_index
    return current.message_index


def _is_recovery_event(event: TraceEvent) -> bool:
    return event.event_class == "mutate" or event.verify_success


def _failure_signature(text: str) -> str:
    lines = []
    for line in text.splitlines():
        if _FAIL_LINE_RE.search(line):
            lines.append(re.sub(r"\s+", " ", line.strip())[:180])
    payload = "\n".join(lines[:20]) or re.sub(r"\s+", " ", text.strip())[:500]
    return hashlib.sha1(payload.encode("utf-8", errors="replace")).hexdigest()[:16]


def _short_args(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip())[:240]


def _looks_like_file_listing(text: str) -> bool:
    lines = [line.strip() for line in text.splitlines() if line.strip() and not line.startswith("OBSERVATION:")]
    if not lines:
        return False
    path_lines = sum(1 for line in lines[:20] if line.startswith("/") or "/" in line)
    return path_lines >= min(3, len(lines))


def _load_loop_abort_detector():
    if __package__ in (None, ""):
        sys.path.append(str(Path(__file__).resolve().parents[1]))
    try:
        from phaseA_scaffold.scaffold.verify.abort import LoopAbortDetector
    except ModuleNotFoundError:
        sys.path.append(str(Path(__file__).resolve().parents[1] / "phaseA_scaffold"))
        sys.path.append(str(Path.cwd() / "phaseA_scaffold"))
        from scaffold.verify.abort import LoopAbortDetector
    return LoopAbortDetector


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--in", dest="source", required=True, type=Path, help="Input JSONL trace file.")
    parser.add_argument("--out", dest="output", required=True, type=Path, help="Output pruned JSONL file.")
    parser.add_argument("--manifest", type=Path, help="Optional JSON manifest path.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = prune_jsonl(args.source, args.output)
    if args.manifest:
        args.manifest.parent.mkdir(parents=True, exist_ok=True)
        args.manifest.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
