#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from phaseD_sft.agentic_trace_filters import trace_should_keep


LOOP_BREAK_COMMAND = "git diff -- . > patch.txt && cat patch.txt"
SUBMIT_COMMAND = "echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT && cat patch.txt"


@dataclass(frozen=True)
class TrajectoryRow:
    kind: str
    instance_id: str
    messages: list[dict[str, Any]]


def bash_call(command: str, call_id: str = "call_1") -> list[dict[str, Any]]:
    return [
        {
            "id": call_id,
            "type": "function",
            "function": {"name": "bash", "arguments": json.dumps({"command": command})},
        }
    ]


def assistant_bash(command: str, content: str = "") -> dict[str, Any]:
    return {"role": "assistant", "content": content, "tool_calls": bash_call(command)}


def user_msg(content: str) -> dict[str, Any]:
    return {"role": "user", "content": content, "tool_calls": []}


def normalize_message(message: dict[str, Any]) -> dict[str, Any] | None:
    role = message.get("role")
    if role == "exit":
        return None
    content = str(message.get("content") or "")
    if role == "tool":
        return user_msg(f"OBSERVATION:\n{content}")
    if role == "assistant":
        tool_calls: list[dict[str, Any]] = []
        for tool_call in message.get("tool_calls") or []:
            command = command_from_tool_call(tool_call)
            if command:
                tool_calls.extend(bash_call(command, str(tool_call.get("id") or f"call_{len(tool_calls) + 1}")))
        return {"role": "assistant", "content": content, "tool_calls": tool_calls}
    if role in {"system", "user"}:
        return {"role": role, "content": content, "tool_calls": []}
    return None


def command_from_tool_call(tool_call: Any) -> str | None:
    if not isinstance(tool_call, dict):
        return None
    function = tool_call.get("function") or {}
    if function.get("name") != "bash":
        return None
    arguments = function.get("arguments")
    if isinstance(arguments, dict):
        command = arguments.get("command")
        return command if isinstance(command, str) and command.strip() else None
    if not isinstance(arguments, str):
        return None
    try:
        decoded = json.loads(arguments)
    except json.JSONDecodeError:
        return None
    if not isinstance(decoded, dict):
        return None
    command = decoded.get("command")
    return command if isinstance(command, str) and command.strip() else None


def assistant_command(message: dict[str, Any]) -> str | None:
    if message.get("role") != "assistant":
        return None
    for tool_call in message.get("tool_calls") or []:
        command = command_from_tool_call(tool_call)
        if command:
            return command
    return None


def normalized_messages(raw_messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    messages: list[dict[str, Any]] = []
    for raw in raw_messages:
        if not isinstance(raw, dict):
            continue
        normalized = normalize_message(raw)
        if normalized is not None:
            messages.append(normalized)
    return messages


def extract_rows_from_trajectory(
    trajectory: dict[str, Any],
    *,
    repeated_read_threshold: int = 5,
    max_prefix_messages: int = 64,
    max_format_recovery_rows: int = 8,
) -> list[TrajectoryRow]:
    instance_id = str(trajectory.get("instance_id") or "unknown")
    raw_messages = [m for m in trajectory.get("messages") or [] if isinstance(m, dict)]
    messages = normalized_messages(raw_messages)
    rows: list[TrajectoryRow] = []

    success = success_submit_row(instance_id, messages)
    if success is not None:
        rows.append(success)

    loop = repeated_read_loop_row(
        instance_id,
        messages,
        repeated_read_threshold=repeated_read_threshold,
        max_prefix_messages=max_prefix_messages,
    )
    if loop is not None:
        rows.append(loop)

    rows.extend(format_recovery_rows(instance_id, messages, max_rows=max_format_recovery_rows))
    return rows


def success_submit_row(instance_id: str, messages: list[dict[str, Any]]) -> TrajectoryRow | None:
    for index, message in enumerate(messages):
        command = assistant_command(message)
        if command and "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT" in command:
            return TrajectoryRow("success_submit", instance_id, messages[: index + 1])
    return None


def repeated_read_loop_row(
    instance_id: str,
    messages: list[dict[str, Any]],
    *,
    repeated_read_threshold: int,
    max_prefix_messages: int,
) -> TrajectoryRow | None:
    last_command: str | None = None
    repeat_count = 0
    for index, message in enumerate(messages):
        command = assistant_command(message)
        if not command:
            continue
        if not is_read_command(command):
            last_command = command
            repeat_count = 1
            continue
        if command == last_command:
            repeat_count += 1
        else:
            last_command = command
            repeat_count = 1
        if repeat_count < repeated_read_threshold:
            continue
        prefix_end = index + 1
        if index + 1 < len(messages) and messages[index + 1].get("role") == "user":
            prefix_end = index + 2
        prefix = trim_prefix(messages[:prefix_end], max_prefix_messages)
        row_messages = [
            *prefix,
            user_msg(
                "Loop warning: you repeated the same read-only command. "
                "Do not run it again. Check the current patch and submit if it is non-empty."
            ),
            assistant_bash(LOOP_BREAK_COMMAND, "I will stop reading and inspect the current patch."),
        ]
        return TrajectoryRow("repeated_read_to_diff", instance_id, row_messages)
    return None


def format_recovery_rows(instance_id: str, messages: list[dict[str, Any]], *, max_rows: int) -> list[TrajectoryRow]:
    rows: list[TrajectoryRow] = []
    seen_commands: set[str] = set()
    for index, message in enumerate(messages[:-1]):
        if len(rows) >= max_rows:
            break
        if message.get("role") != "user":
            continue
        content = str(message.get("content") or "")
        if "Tool call error" not in content and "No tool calls found" not in content:
            continue
        next_message = messages[index + 1]
        command = assistant_command(next_message)
        if not command or command in seen_commands:
            continue
        seen_commands.add(command)
        rows.append(
            TrajectoryRow(
                "format_recovery",
                instance_id,
                [
                    user_msg(shorten(content, 1_200)),
                    assistant_bash(command, str(next_message.get("content") or "")),
                ],
            )
        )
    return rows


def is_read_command(command: str) -> bool:
    stripped = command.strip()
    read_prefixes = ("cat ", "find ", "grep ", "rg ", "sed -n ", "head ", "tail ", "ls ", "pwd")
    edit_markers = ("sed -i", "perl -pi", "write_text", "git apply", "apply_patch", "tee ")
    return stripped.startswith(read_prefixes) and not any(marker in stripped for marker in edit_markers)


def trim_prefix(messages: list[dict[str, Any]], max_messages: int) -> list[dict[str, Any]]:
    if len(messages) <= max_messages:
        return messages
    head = messages[:2]
    tail_budget = max(1, max_messages - len(head) - 1)
    return [
        *head,
        user_msg("[earlier repeated inspection messages omitted]"),
        *messages[-tail_budget:],
    ]


def shorten(text: str, max_chars: int) -> str:
    normalized = text.strip()
    if len(normalized) <= max_chars:
        return normalized
    marker = "\n[... omitted ...]\n"
    head = max_chars // 2
    tail = max_chars - head - len(marker)
    return normalized[:head].rstrip() + marker + normalized[-tail:].lstrip()


def row_to_dataset_dict(row: TrajectoryRow, repeat: int, ordinal: int) -> dict[str, Any]:
    return {
        "instance_id": f"swe-trajectory-{row.kind}-{row.instance_id}-{ordinal:04d}-{repeat:03d}",
        "source": f"swe_trajectory_{row.kind}",
        "messages": row.messages,
    }


def repeat_count_for_kind(row: TrajectoryRow, args: argparse.Namespace) -> int:
    if row.kind == "format_recovery":
        return args.repeat_format_recovery
    if row.kind == "repeated_read_to_diff":
        return args.repeat_repeated_read
    if row.kind == "success_submit":
        return args.repeat_success
    return args.repeat_real


def expand_paths(paths: list[str], globs: list[str]) -> list[Path]:
    expanded = [Path(path) for path in paths]
    for pattern in globs:
        expanded.extend(Path().glob(pattern))
    return sorted({path for path in expanded if path.is_file()})


def load_rows(
    paths: list[Path],
    *,
    repeated_read_threshold: int,
    max_prefix_messages: int,
    max_format_recovery_rows: int,
) -> list[TrajectoryRow]:
    rows: list[TrajectoryRow] = []
    for path in paths:
        trajectory = json.loads(path.read_text())
        rows.extend(
            extract_rows_from_trajectory(
                trajectory,
                repeated_read_threshold=repeated_read_threshold,
                max_prefix_messages=max_prefix_messages,
                max_format_recovery_rows=max_format_recovery_rows,
            )
        )
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", default="data/unsloth_agentic_24k_train_normalized_format_plus_coder_repair")
    parser.add_argument("--out", default="data/unsloth_agentic_24k_train_swe_trajectory_behavior_v3")
    parser.add_argument("--trajectory", action="append", default=[])
    parser.add_argument("--trajectory-glob", action="append", default=[])
    parser.add_argument("--repeat-real", type=int, default=32)
    parser.add_argument("--repeat-format-recovery", type=int, default=2)
    parser.add_argument("--repeat-repeated-read", type=int, default=24)
    parser.add_argument("--repeat-success", type=int, default=16)
    parser.add_argument("--base-limit", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--repeated-read-threshold", type=int, default=5)
    parser.add_argument("--max-prefix-messages", type=int, default=64)
    parser.add_argument("--max-format-recovery-rows", type=int, default=8)
    parser.add_argument("--filter-base-quality", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--max-first-edit-ratio", type=float, default=0.35)
    parser.add_argument("--max-read-streak", type=int, default=4)
    parser.add_argument("--require-verify-tail", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    from datasets import Dataset, concatenate_datasets, load_from_disk

    paths = expand_paths(args.trajectory, args.trajectory_glob)
    if not paths:
        raise SystemExit("no trajectory files matched")

    base = load_from_disk(args.base)
    if args.base_limit > 0 and args.base_limit < len(base):
        base = base.shuffle(seed=args.seed).select(range(args.base_limit))
    if args.filter_base_quality:
        before = len(base)

        def keep_quality(example: dict[str, Any]) -> bool:
            return trace_should_keep(
                example["messages"],
                max_first_edit_ratio=args.max_first_edit_ratio,
                max_read_streak=args.max_read_streak,
                require_verify_tail=args.require_verify_tail,
            )

        base = base.filter(keep_quality, desc="filter-base-quality")
        print(f"base_quality_rows={len(base)}/{before}", flush=True)

    trajectory_rows = load_rows(
        paths,
        repeated_read_threshold=args.repeated_read_threshold,
        max_prefix_messages=args.max_prefix_messages,
        max_format_recovery_rows=args.max_format_recovery_rows,
    )
    counts = Counter(row.kind for row in trajectory_rows)
    rows = []
    for ordinal, row in enumerate(trajectory_rows):
        for repeat in range(repeat_count_for_kind(row, args)):
            rows.append(row_to_dataset_dict(row, repeat, ordinal))
    trajectory_dataset = Dataset.from_list(rows, features=base.features)
    mixed = concatenate_datasets([base, trajectory_dataset])
    mixed.save_to_disk(args.out)

    print(f"trajectory_files={len(paths)}")
    print("trajectory_row_kinds=" + json.dumps(dict(sorted(counts.items())), sort_keys=True))
    print(f"base_rows={len(base)}")
    print(f"trajectory_rows={len(trajectory_dataset)}")
    print(f"total_rows={len(mixed)}")
    print(f"out={args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
