#!/usr/bin/env python3
"""Convert F2P-verified Codex teacher runs into native mini-SWE SFT rows.

The converter deliberately supervises only the patch decision: a compact set
of grounded read/observation pairs followed by the teacher's first source edit.
Codex ``agent_message`` prose is never copied into the training conversation.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from collections import Counter
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import shlex
import sys
from typing import Any, Callable, Iterable

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from phaseD_sft.build_swe_edit_trace_dataset import parse_json_list, patch_paths
from phaseD_sft.compact_observations import compact_observation_text
from phaseD_sft.compress_editfirst import EDIT_COMMAND_RE, READ_COMMAND_RE, edited_paths


_TESTBED_PREFIX_RE = re.compile(r"^\s*cd\s+/testbed\s*&&\s*(?P<command>[\s\S]+?)\s*$")
_DOCKER_BASH_QUOTED_RE = re.compile(
    r"^docker\s+exec\s+(?P<container>[A-Za-z0-9_.-]+)\s+bash\s+-c\s+"
    r"(?P<quote>['\"])(?P<body>[\s\S]*)(?P=quote)$"
)
_NON_RUNTIME_DIRS = {
    ".github",
    "doc",
    "docs",
    "example",
    "examples",
    "test",
    "tests",
    "testing",
}
_NON_RUNTIME_FILES = {
    "setup.py",
    "conftest.py",
}


@dataclass(frozen=True)
class CommandEvent:
    command: str
    output: str
    exit_code: int
    container: str


def _unwrap_testbed_command(command: str) -> tuple[str, str] | None:
    """Return ``(container, inner_command)`` for the exact teacher contract."""
    try:
        tokens = shlex.split(command)
    except ValueError:
        return None
    payload = command
    if len(tokens) == 3 and tokens[0] in {"bash", "/bin/bash"} and tokens[1] in {"-c", "-lc"}:
        payload = tokens[2]
    try:
        tokens = shlex.split(payload)
    except ValueError:
        tokens = []
    if len(tokens) == 6 and tokens[0:2] == ["docker", "exec"] and tokens[3:5] == ["bash", "-c"]:
        container = tokens[2]
        if not re.fullmatch(r"[A-Za-z0-9_.-]+", container):
            return None
        match = _TESTBED_PREFIX_RE.fullmatch(tokens[5])
        if match is None or not match.group("command").strip():
            return None
        return container, match.group("command").strip()
    quoted = _DOCKER_BASH_QUOTED_RE.fullmatch(payload)
    if quoted is None:
        return None
    container = quoted.group("container")
    match = _TESTBED_PREFIX_RE.fullmatch(quoted.group("body"))
    if match is None or not match.group("command").strip():
        return None
    return container, match.group("command").strip()


def unwrap_testbed_command(command: str, *, expected_container: str | None = None) -> str | None:
    """Unwrap a teacher command only when it stays inside the expected container."""
    unwrapped = _unwrap_testbed_command(command)
    if unwrapped is None:
        return None
    container, inner = unwrapped
    if expected_container is not None and container != expected_container:
        return None
    return inner


def _is_matching_source_edit(command: str, paths: list[str]) -> bool:
    command_paths = edited_paths(command)
    if not any(path in command_paths for path in paths):
        return False
    return bool(
        EDIT_COMMAND_RE.search(command)
        or re.search(r"\.write_(?:text|bytes)\s*\(", command)
    )


def _extract_command_events(
    stream_lines: Iterable[str],
    *,
    stop_after: Callable[[str], bool] | None = None,
) -> tuple[list[CommandEvent] | None, str | None]:
    events: list[CommandEvent] = []
    expected_container: str | None = None
    for line in stream_lines:
        try:
            record = json.loads(line)
        except (TypeError, json.JSONDecodeError):
            continue
        if record.get("type") != "item.completed":
            continue
        item = record.get("item") or {}
        if not isinstance(item, dict) or item.get("type") != "command_execution":
            continue
        unwrapped = _unwrap_testbed_command(str(item.get("command") or ""))
        if unwrapped is None:
            return None, "unparseable_command"
        container, command = unwrapped
        if expected_container is None:
            expected_container = container
        elif container != expected_container:
            return None, "container_changed"
        try:
            exit_code = int(item.get("exit_code", 1))
        except (TypeError, ValueError):
            exit_code = 1
        event = CommandEvent(
            command=command,
            output=str(item.get("aggregated_output") or ""),
            exit_code=exit_code,
            container=container,
        )
        events.append(event)
        if stop_after is not None and stop_after(command):
            break
    if not events:
        return None, "no_command_events"
    return events, None


def _source_only_patch(paths: list[str]) -> bool:
    if not paths:
        return False
    for raw_path in paths:
        path = PurePosixPath(raw_path)
        lowered_parts = {part.lower() for part in path.parts}
        name = path.name.lower()
        if path.suffix.lower() != ".py":
            return False
        if lowered_parts & _NON_RUNTIME_DIRS:
            return False
        if name in _NON_RUNTIME_FILES or name.startswith("test_") or name.endswith("_test.py"):
            return False
        if name.startswith(("setup", "noxfile", "tox")):
            return False
    return True


def _assistant(command: str, index: int, *, loss: bool) -> dict[str, Any]:
    return {
        "role": "assistant",
        "content": "",
        "loss": loss,
        "tool_calls": [
            {
                "id": f"call_teacher_{index:03d}",
                "type": "function",
                "function": {
                    "name": "bash",
                    "arguments": json.dumps({"command": command}, ensure_ascii=False),
                },
            }
        ],
    }


def _observation(event: CommandEvent, *, max_chars: int) -> dict[str, Any]:
    content = (
        "OBSERVATION:\n"
        f"<returncode>{event.exit_code}</returncode>\n"
        f"<output>\n{event.output.rstrip()}\n</output>"
    )
    compacted, _, _ = compact_observation_text(content, max_chars=max_chars)
    return {"role": "user", "content": compacted, "tool_calls": []}


def _path_mentioned(text: str, paths: list[str]) -> bool:
    return any(path in text or PurePosixPath(path).name in text for path in paths)


def _task_problem(task: dict[str, Any]) -> str:
    problem = str(task.get("problem_statement") or "").strip()
    if problem:
        return problem
    failing_tests = parse_json_list(task.get("FAIL_TO_PASS"))
    if not failing_tests:
        return ""
    tests = "\n".join(f"- {test}" for test in failing_tests[:16])
    return (
        "A regression causes the following focused tests to fail. Diagnose the runtime source bug, "
        "make the smallest source-only fix, and verify it without modifying tests:\n"
        f"{tests}"
    )


def render_verified_teacher_sft(
    record: dict[str, Any],
    *,
    keep_pre_edit_reads: int = 3,
    max_observation_chars: int = 2_400,
) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    """Render one verified teacher record as a chosen-only patch decision."""
    if keep_pre_edit_reads < 1:
        raise ValueError("keep_pre_edit_reads must be positive")
    result = record.get("result") or {}
    task = record.get("task") or {}
    if not isinstance(result, dict) or not result.get("resolved"):
        return None, {"kept": False, "reason": "not_f2p_resolved"}
    instance_id = str(task.get("instance_id") or result.get("instance_id") or "")
    if not instance_id or str(result.get("instance_id") or instance_id) != instance_id:
        return None, {"kept": False, "reason": "instance_id_mismatch"}

    patch = str(record.get("patch") or "").strip()
    paths = patch_paths(patch)
    if not patch.startswith("diff --git ") or not _source_only_patch(paths):
        return None, {"kept": False, "reason": "non_source_patch", "patch_paths": paths}

    events, error = _extract_command_events(
        record.get("stream_lines") or [],
        stop_after=lambda command: _is_matching_source_edit(command, paths),
    )
    if events is None:
        return None, {"kept": False, "reason": error}

    edit_event_index: int | None = None
    original_edit_command_index: int | None = None
    for index, event in enumerate(events):
        if _is_matching_source_edit(event.command, paths):
            edit_event_index = index
            original_edit_command_index = index + 1
            break
    if edit_event_index is None:
        return None, {"kept": False, "reason": "no_matching_source_edit", "patch_paths": paths}

    reads = [
        event
        for event in events[:edit_event_index]
        if READ_COMMAND_RE.search(event.command)
    ]
    grounded = [event for event in reads if _path_mentioned(event.output, paths)]
    if not grounded:
        return None, {"kept": False, "reason": "edited_file_not_seen", "patch_paths": paths}
    selected = grounded[-keep_pre_edit_reads:]

    system_template = str(record.get("system_template") or "").strip()
    instance_template = str(record.get("instance_template") or "").strip()
    problem = _task_problem(task)
    if not system_template or "{{task}}" not in instance_template or not problem:
        return None, {"kept": False, "reason": "missing_native_template"}
    user_prompt = instance_template.replace("{{task}}", problem)
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": system_template, "tool_calls": []},
        {"role": "user", "content": user_prompt, "tool_calls": []},
    ]
    for call_index, event in enumerate(selected, start=1):
        messages.append(_assistant(event.command, call_index, loss=False))
        messages.append(_observation(event, max_chars=max_observation_chars))
    messages.append(_assistant(events[edit_event_index].command, len(selected) + 1, loss=True))

    row = {
        "instance_id": f"verified-teacher-{instance_id}",
        "source_instance_id": instance_id,
        "source": "python_verified_teacher_patch_decision",
        "messages": messages,
    }
    return row, {
        "kept": True,
        "patch_paths": paths,
        "original_edit_command_index": original_edit_command_index,
        "first_edit_index": len(selected) + 1,
        "kept_pre_edit_reads": len(selected),
        "teacher_model": result.get("model"),
        "teacher_backend": result.get("backend"),
        "prompt_fallback_f2p": not bool(str(task.get("problem_statement") or "").strip()),
    }


def _record_identity(record: dict[str, Any], index: int) -> str:
    return str(
        record.get("record_id")
        or (record.get("provenance") or {}).get("record_id")
        or f"record-{index:06d}"
    )


def build_verified_teacher_rows(
    records: Iterable[dict[str, Any]],
    *,
    token_counter: Callable[[list[dict[str, Any]]], int],
    max_tokens: int = 49_152,
    keep_pre_edit_reads: int = 3,
    max_observation_chars: int = 2_400,
) -> tuple[list[dict[str, Any]], dict[str, Any], list[dict[str, Any]]]:
    """Select, deduplicate, and budget one verified decision per source task."""
    if max_tokens < 1:
        raise ValueError("max_tokens must be positive")
    candidates: dict[str, list[tuple[dict[str, Any], dict[str, Any], dict[str, Any], str]]] = {}
    rejected: list[dict[str, Any]] = []
    drop_reasons: Counter[str] = Counter()
    records_in = 0
    for index, record in enumerate(records):
        records_in += 1
        record_id = _record_identity(record, index)
        row, report = render_verified_teacher_sft(
            record,
            keep_pre_edit_reads=keep_pre_edit_reads,
            max_observation_chars=max_observation_chars,
        )
        if row is None:
            reason = str(report.get("reason") or "unknown")
            drop_reasons[reason] += 1
            rejected.append(
                {
                    "record_id": record_id,
                    "instance_id": (record.get("result") or {}).get("instance_id"),
                    "reason": reason,
                    "report": report,
                }
            )
            continue
        candidates.setdefault(row["source_instance_id"], []).append((row, report, record, record_id))

    selected: list[tuple[dict[str, Any], dict[str, Any], dict[str, Any], str]] = []
    superseded = 0
    for instance_id in sorted(candidates):
        attempts = sorted(
            candidates[instance_id],
            key=lambda item: (
                int(item[1].get("original_edit_command_index") or 1_000_000),
                len(str(item[2].get("patch") or "")),
                item[3],
            ),
        )
        selected.append(attempts[0])
        for _, report, _, record_id in attempts[1:]:
            superseded += 1
            drop_reasons["superseded_verified_attempt"] += 1
            rejected.append(
                {
                    "record_id": record_id,
                    "instance_id": instance_id,
                    "reason": "superseded_verified_attempt",
                    "report": report,
                }
            )

    rows: list[dict[str, Any]] = []
    token_counts: list[int] = []
    per_repo: Counter[str] = Counter()
    original_edit_histogram: Counter[str] = Counter()
    prompt_fallback_f2p_rows = 0
    seen_content: dict[str, str] = {}
    content_duplicates = 0
    budget_rejected = 0
    for row, report, record, record_id in selected:
        content = json.dumps(row["messages"], ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        content_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()
        if content_hash in seen_content:
            content_duplicates += 1
            drop_reasons["content_duplicate"] += 1
            rejected.append(
                {
                    "record_id": record_id,
                    "instance_id": row["source_instance_id"],
                    "reason": "content_duplicate",
                    "duplicate_of": seen_content[content_hash],
                    "report": report,
                }
            )
            continue
        seen_content[content_hash] = row["source_instance_id"]
        token_count = int(token_counter(row["messages"]))
        if token_count > max_tokens:
            budget_rejected += 1
            drop_reasons["token_budget_exceeded"] += 1
            rejected.append(
                {
                    "record_id": record_id,
                    "instance_id": row["source_instance_id"],
                    "reason": "token_budget_exceeded",
                    "token_count": token_count,
                    "report": report,
                }
            )
            continue
        token_counts.append(token_count)
        rows.append(row)
        per_repo[str((record.get("task") or {}).get("repo") or "unknown")] += 1
        original_edit_histogram[str(report["original_edit_command_index"])] += 1
        prompt_fallback_f2p_rows += int(bool(report.get("prompt_fallback_f2p")))

    token_counts.sort()
    manifest = {
        "schema_version": 1,
        "records_in": records_in,
        "renderable_verified_attempts": sum(len(items) for items in candidates.values()),
        "unique_tasks_selected": len(selected),
        "superseded_verified_attempts": superseded,
        "content_duplicates": content_duplicates,
        "budget_rejected": budget_rejected,
        "rows_kept": len(rows),
        "max_tokens": max_tokens,
        "keep_pre_edit_reads": keep_pre_edit_reads,
        "max_observation_chars": max_observation_chars,
        "drop_reasons": dict(sorted(drop_reasons.items())),
        "per_repo": dict(sorted(per_repo.items())),
        "prompt_fallback_f2p_rows": prompt_fallback_f2p_rows,
        "original_edit_command_index_histogram": dict(
            sorted(original_edit_histogram.items(), key=lambda item: int(item[0]))
        ),
        "token_stats": {
            "min": token_counts[0] if token_counts else None,
            "median": token_counts[len(token_counts) // 2] if token_counts else None,
            "max": token_counts[-1] if token_counts else None,
        },
    }
    return rows, manifest, rejected


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: expected JSON object")
            records.append(value)
    return records


def load_teacher_records(
    task_paths: Iterable[Path],
    run_dirs: Iterable[Path],
    *,
    system_template: str,
    instance_template: str,
) -> list[dict[str, Any]]:
    """Join immutable task rows to raw streams, patches, and F2P ledgers."""
    tasks: dict[str, dict[str, Any]] = {}
    for task_path in task_paths:
        for row in _read_jsonl(Path(task_path)):
            instance_id = str(row.get("instance_id") or "")
            if not instance_id:
                raise ValueError(f"{task_path}: task missing instance_id")
            if instance_id in tasks and tasks[instance_id] != row:
                raise ValueError(f"conflicting task rows for {instance_id}")
            tasks[instance_id] = row

    records: list[dict[str, Any]] = []
    for raw_run_dir in run_dirs:
        run_dir = Path(raw_run_dir)
        ledger = run_dir / "results.jsonl"
        if not ledger.exists():
            raise FileNotFoundError(ledger)
        for result in _read_jsonl(ledger):
            instance_id = str(result.get("instance_id") or "")
            if instance_id not in tasks:
                raise ValueError(f"{ledger}: result task not found in supplied pools: {instance_id}")
            patch_path = run_dir / f"{instance_id}.patch"
            stream_path = run_dir / f"{instance_id}.stream.jsonl"
            records.append(
                {
                    "record_id": f"{run_dir}:{instance_id}",
                    "task": tasks[instance_id],
                    "result": result,
                    "patch": patch_path.read_text(errors="replace") if patch_path.exists() else "",
                    "stream_lines": stream_path.read_text(errors="replace").splitlines() if stream_path.exists() else [],
                    "system_template": system_template,
                    "instance_template": instance_template,
                    "provenance": {
                        "run_dir": str(run_dir),
                        "ledger": str(ledger),
                        "patch": str(patch_path),
                        "stream": str(stream_path),
                    },
                }
            )
    return records


def _token_counter(tokenizer: Any, messages: list[dict[str, Any]]) -> int:
    rendered = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    encoded = tokenizer(text=rendered, add_special_tokens=False, truncation=False)
    input_ids = encoded["input_ids"]
    if input_ids and isinstance(input_ids[0], list):
        input_ids = input_ids[0]
    return len(input_ids)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", type=Path, action="append", required=True)
    parser.add_argument("--run", dest="runs", type=Path, action="append", required=True)
    parser.add_argument("--config", type=Path, default=Path("phaseH_eval/swebench_edit_first_selfretry.yaml"))
    parser.add_argument("--base", default="/media/ironbcc/CrucialX10/models/google/gemma-4-31B-it")
    parser.add_argument("--out-jsonl", type=Path, required=True)
    parser.add_argument("--out-dataset", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--rejected-sidecar", type=Path, required=True)
    parser.add_argument("--max-tokens", type=int, default=49_152)
    parser.add_argument("--keep-pre-edit-reads", type=int, default=3)
    parser.add_argument("--max-observation-chars", type=int, default=2_400)
    args = parser.parse_args()

    import yaml
    from datasets import Dataset
    from transformers import AutoTokenizer

    config = yaml.safe_load(args.config.read_text())
    system_template = str(config["agent"]["system_template"])
    instance_template = str(config["agent"]["instance_template"])
    records = load_teacher_records(
        args.tasks,
        args.runs,
        system_template=system_template,
        instance_template=instance_template,
    )
    tokenizer = AutoTokenizer.from_pretrained(args.base)
    rows, manifest, rejected = build_verified_teacher_rows(
        records,
        token_counter=lambda messages: _token_counter(tokenizer, messages),
        max_tokens=args.max_tokens,
        keep_pre_edit_reads=args.keep_pre_edit_reads,
        max_observation_chars=args.max_observation_chars,
    )
    manifest.update(
        {
            "tasks": [str(path) for path in args.tasks],
            "runs": [str(path) for path in args.runs],
            "config": str(args.config),
            "base_tokenizer": args.base,
            "output_jsonl": str(args.out_jsonl),
            "output_dataset": str(args.out_dataset),
            "rejected_sidecar": str(args.rejected_sidecar),
            "rejected_records": len(rejected),
        }
    )
    for path in (args.out_jsonl, args.manifest, args.rejected_sidecar):
        path.parent.mkdir(parents=True, exist_ok=True)
    with args.out_jsonl.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    with args.rejected_sidecar.open("w", encoding="utf-8") as handle:
        for record in rejected:
            handle.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
    args.manifest.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    Dataset.from_list(rows).save_to_disk(str(args.out_dataset))
    print(json.dumps(manifest, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
