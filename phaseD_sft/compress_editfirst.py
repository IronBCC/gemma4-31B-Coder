#!/usr/bin/env python3
"""Compress successful agent trajectories around their first source edit.

The compressor removes only pre-edit exploration.  It retains the initial task,
the last relevant read/observation pairs, and the complete edit-to-submission
suffix so assistant reasoning and post-edit behavior remain loss-supervised.
"""
from __future__ import annotations

import argparse
from collections import Counter
import copy
import json
from pathlib import Path
import re
import shlex
from statistics import median
from typing import Any, Iterable

from phaseD_sft.agentic_trace_filters import command_trace_quality_report


EDIT_COMMAND_RE = re.compile(
    r"\bsed\s+-i\b|\b(?:perl\s+-i|apply_patch|git\s+apply|tee\b)|"
    r"(?:^|[;\n])\s*cat\s+(?:>>|>)|\bopen\(\s*['\"][^'\"]+['\"]\s*,\s*['\"][wa]"
)
READ_COMMAND_RE = re.compile(
    r"^\s*(?:cat\s+|find\s+|grep\s+|rg\s+|sed\s+-n|head\s+|tail\s+|ls\s+|pwd\s*$|git\s+(?:status|diff|show|log))"
)
OPEN_WRITE_RE = re.compile(r"\bopen\(\s*['\"]([^'\"]+)['\"]\s*,\s*['\"][wa]")
PATH_WRITE_RE = re.compile(r"\bPath\(\s*['\"]([^'\"]+)['\"]\s*\)\.write_(?:text|bytes)")
PATH_ASSIGN_RE = re.compile(r"\b([A-Za-z_]\w*)\s*=\s*Path\(\s*['\"]([^'\"]+)['\"]\s*\)")
REDIRECT_RE = re.compile(r"(?:^|[;\n])\s*(?:cat|printf|echo)\b[^\n]*?(?:>>|>)\s*([^\s;|<&]+)")
DIFF_PATH_RE = re.compile(r"(?:diff --git a/|\*\*\* Update File: )([^\s]+)")


def _commands(message: dict[str, Any]) -> list[str]:
    commands: list[str] = []
    for call in message.get("tool_calls") or []:
        if not isinstance(call, dict):
            continue
        function = call.get("function") or {}
        if not isinstance(function, dict):
            continue
        arguments = function.get("arguments")
        try:
            decoded = json.loads(arguments) if isinstance(arguments, str) else arguments
        except (TypeError, json.JSONDecodeError):
            continue
        command = decoded.get("command") if isinstance(decoded, dict) else None
        if isinstance(command, str) and command:
            commands.append(command)
    return commands


def _clean_path(value: str) -> str:
    return value.strip().strip("'\"").removeprefix("a/").removeprefix("b/")


def edited_paths(command: str) -> list[str]:
    """Extract literal source-file paths from common shell and Python edit forms."""
    paths: list[str] = []
    # Tokenize shell control operators while respecting quoted semicolons in a
    # multi-expression ``sed`` program.  Regex-splitting the raw command loses
    # the target of commands such as ``sed -i 's/a/b/; s/c/d/' file.py``.
    try:
        lexer = shlex.shlex(command, posix=True, punctuation_chars=";&|\n")
        lexer.whitespace_split = True
        lexer.whitespace = " \t\r"
        lexer.commenters = ""
        shell_tokens = list(lexer)
    except ValueError:
        shell_tokens = []
    fragments: list[list[str]] = []
    fragment: list[str] = []
    for token in shell_tokens:
        if token and all(character in ";&|\n" for character in token):
            if fragment:
                fragments.append(fragment)
                fragment = []
            continue
        fragment.append(token)
    if fragment:
        fragments.append(fragment)
    for tokens in fragments:
        for executable in ("sed", "perl"):
            if executable not in tokens:
                continue
            position = tokens.index(executable)
            if position + 1 >= len(tokens) or not tokens[position + 1].startswith("-i"):
                continue
            candidate = _clean_path(tokens[-1])
            if candidate and not candidate.startswith("-") and candidate != executable:
                paths.append(candidate)
    for match in OPEN_WRITE_RE.finditer(command):
        paths.append(_clean_path(match.group(1)))
    for match in PATH_WRITE_RE.finditer(command):
        paths.append(_clean_path(match.group(1)))
    for match in PATH_ASSIGN_RE.finditer(command):
        variable, path = match.groups()
        if re.search(rf"\b{re.escape(variable)}\.write_(?:text|bytes)\s*\(", command):
            paths.append(_clean_path(path))
    for match in REDIRECT_RE.finditer(command):
        paths.append(_clean_path(match.group(1)))
    if re.search(r"\btee\b", command):
        try:
            tokens = shlex.split(command)
        except ValueError:
            tokens = []
        if "tee" in tokens:
            for token in tokens[tokens.index("tee") + 1 :]:
                if not token.startswith("-"):
                    paths.append(_clean_path(token))
                    break
    for match in DIFF_PATH_RE.finditer(command):
        paths.append(_clean_path(match.group(1)))
    return list(dict.fromkeys(path for path in paths if path and path not in {"/dev/null", "."}))


def _is_edit(message: dict[str, Any]) -> tuple[bool, list[str]]:
    paths: list[str] = []
    matched = False
    for command in _commands(message):
        if EDIT_COMMAND_RE.search(command):
            matched = True
            paths.extend(edited_paths(command))
    return matched, list(dict.fromkeys(paths))


def _is_observation(message: dict[str, Any]) -> bool:
    return message.get("role") == "user" and str(message.get("content") or "").startswith("OBSERVATION:")


def _pair_for_assistant(messages: list[dict[str, Any]], index: int) -> tuple[int, str] | None:
    next_index = index + 1
    if next_index < len(messages) and _is_observation(messages[next_index]):
        return next_index, str(messages[next_index].get("content") or "")
    return None


def _initial_context_indices(messages: list[dict[str, Any]], before: int) -> list[int]:
    system = next((index for index in range(before) if messages[index].get("role") == "system"), None)
    user = next(
        (index for index in range(before) if messages[index].get("role") == "user" and not _is_observation(messages[index])),
        None,
    )
    return sorted(index for index in (system, user) if index is not None)


def compress_editfirst_trace(
    row: dict[str, Any], *, keep_pre_edit_reads: int = 3
) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    """Return an edit-first trace or a reasoned drop record.

    A retained read is always emitted with its immediately following observation,
    so compaction cannot orphan conditioning from a supervised tool call.
    """
    if keep_pre_edit_reads < 1:
        raise ValueError("keep_pre_edit_reads must be positive")
    messages = row.get("messages")
    if not isinstance(messages, list):
        return None, {"kept": False, "reason": "missing_messages"}
    messages = [message for message in messages if isinstance(message, dict)]
    edit_index: int | None = None
    paths: list[str] = []
    for index, message in enumerate(messages):
        if message.get("role") != "assistant":
            continue
        is_edit, detected_paths = _is_edit(message)
        if is_edit:
            edit_index, paths = index, detected_paths
            break
    if edit_index is None:
        return None, {"kept": False, "reason": "no_source_edit"}
    if not paths:
        return None, {"kept": False, "reason": "edited_paths_unresolved"}

    reads: list[tuple[int, int, str, bool]] = []
    for index, message in enumerate(messages[:edit_index]):
        if message.get("role") != "assistant":
            continue
        commands = _commands(message)
        if not commands or not all(READ_COMMAND_RE.search(command) for command in commands):
            continue
        pair = _pair_for_assistant(messages, index)
        if pair is None:
            continue
        observation_index, observation = pair
        mentions_edited_file = any(
            path in observation or Path(path).name in observation
            for path in paths
        )
        reads.append((index, observation_index, observation, mentions_edited_file))

    matching_reads = [read for read in reads if read[3]]
    selected = (matching_reads if matching_reads else reads)[-keep_pre_edit_reads:]
    if not selected:
        return None, {"kept": False, "reason": "no_paired_pre_edit_reads", "edited_paths": paths}
    if not any(read[3] for read in selected):
        # The fallback could only be selected when no relevant observation was
        # found.  Refuse this trace rather than break the file-grounding guard.
        return None, {"kept": False, "reason": "edited_file_not_seen_in_pre_edit_observation", "edited_paths": paths}

    kept_indices = set(_initial_context_indices(messages, edit_index))
    for assistant_index, observation_index, _, _ in selected:
        kept_indices.update((assistant_index, observation_index))
    kept_indices.update(range(edit_index, len(messages)))
    compressed = copy.deepcopy(row)
    compressed["messages"] = [copy.deepcopy(message) for index, message in enumerate(messages) if index in kept_indices]
    quality = command_trace_quality_report(compressed["messages"])
    kept_observations = "\n".join(read[2] for read in selected)
    return compressed, {
        "kept": True,
        "edited_paths": paths,
        "first_edit_index": quality.first_edit_index,
        "kept_pre_edit_reads": len(selected),
        "kept_pre_edit_observation_text": kept_observations,
        "dropped_pre_edit_messages": edit_index - len(_initial_context_indices(messages, edit_index)) - 2 * len(selected),
    }


def _load_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def compress_jsonl(source: Path, output: Path, *, keep_pre_edit_reads: int = 3) -> dict[str, Any]:
    """Compress JSONL rows and emit an auditable first-edit distribution."""
    totals: Counter[str] = Counter()
    histogram: Counter[str] = Counter()
    first_edits: list[int] = []
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for row in _load_jsonl(source):
            totals["rows_in"] += 1
            compressed, report = compress_editfirst_trace(row, keep_pre_edit_reads=keep_pre_edit_reads)
            if compressed is None:
                totals[f"dropped_{report['reason']}"] += 1
                continue
            index = int(report["first_edit_index"])
            histogram[str(index)] += 1
            first_edits.append(index)
            handle.write(json.dumps(compressed, ensure_ascii=False, separators=(",", ":")) + "\n")
            totals["rows_kept"] += 1
    return {
        "schema_version": 1,
        "source": str(source),
        "output": str(output),
        "keep_pre_edit_reads": keep_pre_edit_reads,
        "rows_in": totals["rows_in"],
        "rows_kept": totals["rows_kept"],
        "rows_dropped": totals["rows_in"] - totals["rows_kept"],
        "drop_reasons": {key.removeprefix("dropped_"): value for key, value in sorted(totals.items()) if key.startswith("dropped_")},
        "first_edit_index_histogram": dict(sorted(histogram.items(), key=lambda item: int(item[0]))),
        "first_edit_index_median": median(first_edits) if first_edits else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--in", dest="source", type=Path, required=True)
    parser.add_argument("--out", dest="output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--keep-pre-edit-reads", type=int, default=3)
    args = parser.parse_args()
    manifest = compress_jsonl(args.source, args.output, keep_pre_edit_reads=args.keep_pre_edit_reads)
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(manifest, sort_keys=True))


if __name__ == "__main__":
    main()
