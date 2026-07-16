#!/usr/bin/env python3
"""Build RLVR prompts at the first edit decision or read-loop pressure point."""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import glob
import hashlib
import json
from pathlib import Path
import re
from typing import Any, Callable

from phaseD_sft.progress import EtaProgress


# Kept identical to /tmp/audit_edit_behavior.py so "edit" has one definition.
EDIT_COMMAND = re.compile(
    r"sed -i|perl -i|apply_patch|git apply|cat > |cat >> |tee |>>|"
    r"python3? - <<|str_replace|open\(.+[\"']w[\"']"
)
READ_COMMAND = re.compile(
    r"^\s*(cat |find |grep |rg |sed -n|head |tail |ls |pwd|git (status|diff|show|log))"
)
PATH_IN_OBSERVATION = re.compile(
    r"(?<![\w.-])((?:\.?/?[\w.-]+/)+[\w.-]+\.(?:py|pyi|rs|c|cc|cpp|h|hpp|toml|cfg|txt|rst|md))(?![\w.-])"
)


@dataclass(frozen=True)
class DecisionPoint:
    messages: list[dict[str, Any]]
    had_edit: bool
    trigger: str
    command_index: int
    edit_files: list[str]


@dataclass(frozen=True)
class Candidate:
    row: dict[str, Any]
    origin: str  # smoke is preserved first under the cap because it is on-policy.


def command_from_tool_call(tool_call: Any) -> str | None:
    if not isinstance(tool_call, dict):
        return None
    function = tool_call.get("function") or {}
    if function.get("name") != "bash":
        return None
    arguments = function.get("arguments")
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments)
        except json.JSONDecodeError:
            return None
    if not isinstance(arguments, dict):
        return None
    command = arguments.get("command")
    return command if isinstance(command, str) and command.strip() else None


def assistant_commands(message: dict[str, Any]) -> list[str]:
    if message.get("role") != "assistant":
        return []
    return [
        command
        for tool_call in message.get("tool_calls") or []
        if (command := command_from_tool_call(tool_call)) is not None
    ]


def files_in_prior_observations(messages: list[dict[str, Any]]) -> list[str]:
    files: set[str] = set()
    for message in messages:
        role = message.get("role")
        content = message.get("content")
        if role not in {"tool", "observation", "user"} or not isinstance(content, str):
            continue
        if role == "user" and not content.lstrip().lower().startswith(("observation:", "tool observation:")):
            continue
        files.update(match.group(1).lstrip("./") for match in PATH_IN_OBSERVATION.finditer(content))
    return sorted(files)


def extract_decision_point(messages: list[dict[str, Any]]) -> DecisionPoint | None:
    """Stop immediately before first edit, repeated command, or fourth read command."""
    command_index = 0
    previous_command: str | None = None
    read_streak = 0

    for message_index, message in enumerate(messages):
        for command in assistant_commands(message):
            command_index += 1
            prefix = messages[:message_index]
            if EDIT_COMMAND.search(command):
                return DecisionPoint(
                    messages=prefix,
                    had_edit=True,
                    trigger="first_edit",
                    command_index=command_index,
                    edit_files=files_in_prior_observations(prefix),
                )
            normalized = command.strip()
            if previous_command is not None and normalized == previous_command:
                return DecisionPoint(
                    messages=prefix,
                    had_edit=False,
                    trigger="repeated_command",
                    command_index=command_index,
                    edit_files=files_in_prior_observations(prefix),
                )
            read_streak = read_streak + 1 if READ_COMMAND.search(command) else 0
            if read_streak >= 4:
                return DecisionPoint(
                    messages=prefix,
                    had_edit=False,
                    trigger="read_streak_4",
                    command_index=command_index,
                    edit_files=files_in_prior_observations(prefix),
                )
            previous_command = normalized
    return None


def extract_edit_adjacent_point(messages: list[dict[str, Any]]) -> DecisionPoint | None:
    """Stop before the command immediately preceding the first source edit.

    The prompt therefore asks the policy to make the last read/inspection
    decision before it historically edited, rather than replaying the exact
    pre-edit state.  Multi-call assistant turns cannot be cut at a command
    boundary without fabricating a partial assistant message, so they are
    excluded deliberately.
    """
    commands: list[tuple[int, int, str]] = []
    command_index = 0
    for message_index, message in enumerate(messages):
        for command in assistant_commands(message):
            command_index += 1
            if EDIT_COMMAND.search(command):
                if not commands:
                    return None
                prior_message_index, prior_command_index, _ = commands[-1]
                if prior_message_index == message_index:
                    return None
                prefix = messages[:prior_message_index]
                if not prefix:
                    return None
                return DecisionPoint(
                    messages=prefix,
                    had_edit=True,
                    trigger="one_command_before_first_edit",
                    command_index=prior_command_index,
                    edit_files=files_in_prior_observations(prefix),
                )
            commands.append((message_index, command_index, command))
    return None


def _content_hash(messages: list[dict[str, Any]]) -> str:
    return hashlib.sha256(
        json.dumps(messages, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _make_row(source: str, instance_id: str, point: DecisionPoint) -> dict[str, Any]:
    return {
        "source": source,
        "instance_id": instance_id,
        "messages": point.messages,
        "target_hint": {"edit_files": point.edit_files, "had_edit": point.had_edit},
    }


def candidates_from_trajectory(
    path: Path,
    extractor: Callable[[list[dict[str, Any]]], DecisionPoint | None] = extract_decision_point,
) -> tuple[Candidate | None, str]:
    try:
        trajectory = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None, "invalid_trajectory"
    messages = trajectory.get("messages")
    if not isinstance(messages, list) or not all(isinstance(message, dict) for message in messages):
        return None, "invalid_messages"
    point = extractor(messages)
    if point is None or not point.messages:
        return None, "no_decision_point"
    instance_id = str(trajectory.get("instance_id") or path.parent.name)
    run_name = path.parents[2].name if len(path.parents) >= 3 else path.parent.name
    return Candidate(_make_row(f"smoke:{run_name}", instance_id, point), "smoke"), point.trigger


def candidates_from_sft_row(
    row: dict[str, Any],
    extractor: Callable[[list[dict[str, Any]]], DecisionPoint | None] = extract_decision_point,
) -> tuple[Candidate | None, str]:
    messages = row.get("messages")
    if not isinstance(messages, list) or not all(isinstance(message, dict) for message in messages):
        return None, "invalid_messages"
    point = extractor(messages)
    if point is None or not point.messages:
        return None, "no_decision_point"
    source = str(row.get("source") or "sft:MISSING")
    instance_id = str(row.get("instance_id") or "unknown")
    return Candidate(_make_row(source, instance_id, point), "sft"), point.trigger


def load_jsonl(path: Path) -> list[dict[str, Any]]:
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


def load_sft_rows(path: Path) -> list[dict[str, Any]]:
    """Read either the requested HuggingFace dataset directory or JSONL input."""
    if not path.is_dir():
        return load_jsonl(path)
    from datasets import load_from_disk

    return [dict(row) for row in load_from_disk(str(path))]


def _rendered_tokens(tokenizer, messages: list[dict[str, Any]]) -> int:
    rendered = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    input_ids = tokenizer(rendered, add_special_tokens=False)["input_ids"]
    if input_ids and isinstance(input_ids[0], list):
        input_ids = input_ids[0]
    return len(input_ids)


def source_group(candidate: Candidate) -> str:
    """Map concrete training/rollout sources to the v2 balancing groups."""
    if candidate.origin == "smoke":
        return "smoke"
    source = str(candidate.row["source"])
    if source == "swe-smith":
        return "swe-smith"
    if source == "swe_train_oracle_edit_trace":
        return "oracle"
    if source == "open_swe_traces_qwen35":
        return "openswe"
    if source == "kwai_klear_miniswe":
        return "kwai"
    return source


def select_source_balanced(
    accepted: list[tuple[Candidate, int]],
    *,
    source_caps: dict[str, int | None],
    include_only_capped_sources: bool,
) -> tuple[list[tuple[Candidate, int]], dict[str, int]]:
    """Take shortest deterministic prefixes within independently capped source groups."""
    buckets: dict[str, list[tuple[Candidate, int]]] = {
        source: [] for source in source_caps
    }
    for item in accepted:
        group = source_group(item[0])
        if group in buckets:
            buckets[group].append(item)
        elif not include_only_capped_sources:
            buckets.setdefault(group, []).append(item)

    selected: list[tuple[Candidate, int]] = []
    counts: dict[str, int] = {}
    for group, items in buckets.items():
        items.sort(key=lambda item: (item[1], item[0].row["source"], item[0].row["instance_id"]))
        cap = source_caps.get(group)
        chosen = items if cap is None else items[:cap]
        selected.extend(chosen)
        if chosen:
            counts[group] = len(chosen)
    return selected, dict(sorted(counts.items()))


def parse_source_caps(values: list[str] | None) -> dict[str, int | None] | None:
    if not values:
        return None
    caps: dict[str, int | None] = {}
    for value in values:
        group, separator, raw_cap = value.partition("=")
        group = group.strip()
        raw_cap = raw_cap.strip().lower()
        if not separator or not group or not raw_cap or group in caps:
            raise ValueError(f"invalid or duplicate --source-cap {value!r}; expected GROUP=COUNT|all")
        if raw_cap == "all":
            caps[group] = None
            continue
        try:
            cap = int(raw_cap)
        except ValueError as exc:
            raise ValueError(
                f"source cap for {group!r} must be a positive integer or 'all'"
            ) from exc
        if cap <= 0:
            raise ValueError(f"source cap for {group!r} must be a positive integer or 'all'")
        caps[group] = cap
    return caps


def dedup_budget_and_cap(
    candidates: list[Candidate],
    tokenizer,
    *,
    max_tokens: int,
    cap: int,
    workers: int,
    progress_every: int,
    source_caps: dict[str, int | None] | None = None,
    include_only_capped_sources: bool = False,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if workers <= 0 or cap <= 0 or max_tokens <= 0 or progress_every <= 0:
        raise ValueError("workers, cap, max_tokens, and progress_every must be positive")
    deduped: list[Candidate] = []
    seen: set[str] = set()
    duplicate_count = 0
    for candidate in candidates:
        digest = _content_hash(candidate.row["messages"])
        if digest in seen:
            duplicate_count += 1
            continue
        seen.add(digest)
        deduped.append(candidate)

    def measure(candidate: Candidate) -> tuple[Candidate, int | None, str | None]:
        try:
            return candidate, _rendered_tokens(tokenizer, candidate.row["messages"]), None
        except Exception as exc:  # Record template-incompatible prefixes; do not guess tokens.
            return candidate, None, type(exc).__name__

    progress = EtaProgress("rlvr-decision-budget", total=len(deduped))
    if workers == 1:
        results = map(measure, deduped)
    else:
        executor = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="rlvr-decision")
        results = executor.map(measure, deduped)

    accepted: list[tuple[Candidate, int]] = []
    budget_stats: Counter[str] = Counter()
    try:
        for completed, (candidate, tokens, error) in enumerate(results, start=1):
            if error:
                budget_stats[f"render_error:{error}"] += 1
            elif tokens is not None and tokens <= max_tokens:
                accepted.append((candidate, tokens))
            else:
                budget_stats["over_budget"] += 1
            if completed % progress_every == 0 or completed == len(deduped):
                print(progress.update(completed), flush=True)
    finally:
        if workers != 1:
            executor.shutdown(wait=True)

    selected_by_group: dict[str, int] | None = None
    if source_caps is not None:
        selected, selected_by_group = select_source_balanced(
            accepted,
            source_caps=source_caps,
            include_only_capped_sources=include_only_capped_sources,
        )
        if len(selected) > cap:
            raise ValueError(
                f"source-capped selection has {len(selected)} rows, exceeding global cap {cap}; "
                "lower a source cap rather than silently biasing a group"
            )
    else:
        # On-policy smoke decision points are retained first. Within each origin and source,
        # shorter valid prefixes are selected deterministically to maximize cap utility.
        accepted.sort(
            key=lambda item: (
                0 if item[0].origin == "smoke" else 1,
                item[1],
                item[0].row["source"],
                item[0].row["instance_id"],
            )
        )
        selected = accepted[:cap]
    source_counts = Counter(item[0].row["source"] for item in selected)
    origin_counts = Counter(item[0].origin for item in selected)
    tokens = [item[1] for item in selected]
    token_stats = {
        "min": min(tokens) if tokens else 0,
        "p50": sorted(tokens)[(len(tokens) - 1) // 2] if tokens else 0,
        "max": max(tokens) if tokens else 0,
    }
    return [item[0].row for item in selected], {
        "candidates_in": len(candidates),
        "deduped": len(deduped),
        "duplicates_removed": duplicate_count,
        "budget_accepted_pre_cap": len(accepted),
        "selected": len(selected),
        "cap": cap,
        "max_rendered_prefix_tokens": max_tokens,
        "budget_stats": dict(sorted(budget_stats.items())),
        "selected_by_source": dict(sorted(source_counts.items())),
        "selected_by_origin": dict(sorted(origin_counts.items())),
        "selected_by_selection_group": selected_by_group,
        "prefix_token_stats": token_stats,
        "selection_policy": (
            "source_capped_then_shorter_prefix; deterministic_source_instance_tiebreak"
            if source_caps is not None
            else "smoke_first_then_shorter_prefix; deterministic_source_instance_tiebreak"
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sft-data", type=Path, required=True)
    parser.add_argument("--trajectory-glob", action="append", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--max-tokens", type=int, default=8192)
    parser.add_argument("--cap", type=int, default=2000)
    parser.add_argument(
        "--source-cap",
        action="append",
        metavar="GROUP=COUNT|all",
        help="Independently select the shortest valid prefixes in each source group.",
    )
    parser.add_argument(
        "--only-capped-sources",
        action="store_true",
        help="Drop source groups without a --source-cap (used to exclude coder_repair in v2).",
    )
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--progress-every", type=int, default=100)
    parser.add_argument(
        "--edit-adjacent",
        action="store_true",
        help="Keep only edit trajectories and end each prompt before the command preceding its first edit.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        source_caps = parse_source_caps(args.source_cap)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    trajectory_paths = sorted({Path(path) for pattern in args.trajectory_glob for path in glob.glob(pattern)})
    if not trajectory_paths:
        raise SystemExit("no trajectory files matched")
    extraction_stats: Counter[str] = Counter()
    extractor = extract_edit_adjacent_point if args.edit_adjacent else extract_decision_point
    candidates: list[Candidate] = []
    for path in trajectory_paths:
        candidate, reason = candidates_from_trajectory(path, extractor)
        extraction_stats[f"smoke:{reason}"] += 1
        if candidate:
            candidates.append(candidate)
    for row in load_sft_rows(args.sft_data):
        candidate, reason = candidates_from_sft_row(row, extractor)
        extraction_stats[f"sft:{reason}"] += 1
        if candidate:
            candidates.append(candidate)

    from transformers import AutoTokenizer

    # Load exactly once in the main thread; ThreadPoolExecutor workers share this tokenizer.
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)
    rows, budget_manifest = dedup_budget_and_cap(
        candidates,
        tokenizer,
        max_tokens=args.max_tokens,
        cap=args.cap,
        workers=args.workers,
        progress_every=args.progress_every,
        source_caps=source_caps,
        include_only_capped_sources=args.only_capped_sources,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    manifest = {
        "sft_data": str(args.sft_data),
        "trajectory_globs": args.trajectory_glob,
        "trajectory_files": len(trajectory_paths),
        "extraction": dict(sorted(extraction_stats.items())),
        "tokenizer": args.tokenizer,
        "workers": args.workers,
        "edit_adjacent": args.edit_adjacent,
        "requested_source_caps": source_caps,
        "only_capped_sources": args.only_capped_sources,
        "output": str(args.out),
        **budget_manifest,
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
