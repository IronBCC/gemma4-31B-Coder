#!/usr/bin/env python3
"""Build strict, portable final-patch SFT rows from verified teacher runs.

Unlike the first-edit corpus, this builder supervises the teacher's complete
submitted patch as one native mini-SWE bash call.  Every edited path must be
grounded by a pre-edit observation, large F2P contracts are excluded from the
strict replay lane, and an optional independent-credit ledger can restrict the
output to commands that earned tier 1.0 under isolated execution.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import sys
from typing import Any, Callable, Iterable, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from phaseD_sft.build_swe_edit_trace_dataset import parse_json_list, patch_paths
from phaseD_sft.build_verified_teacher_sft import (
    _assistant,
    _extract_command_events,
    _is_matching_source_edit,
    _observation,
    _record_identity,
    _source_only_patch,
    _task_problem,
    _token_counter,
    load_teacher_records,
)
from phaseD_sft.compress_editfirst import EDIT_COMMAND_RE, READ_COMMAND_RE


def portable_git_apply_command(patch: str) -> str:
    """Return a quoted-heredoc command that applies exactly ``patch``."""
    normalized = patch.rstrip("\n")
    if not normalized.strip():
        raise ValueError("empty patch")
    digest = hashlib.sha256(normalized.encode("utf-8")).hexdigest()
    for width in range(8, len(digest) + 1, 4):
        delimiter = f"PATCH_{digest[:width]}"
        if delimiter not in normalized.splitlines():
            return f"git apply - <<'{delimiter}'\n{normalized}\n{delimiter}"
    raise ValueError("could not construct a collision-free patch delimiter")


_UNSUPPORTED_PATCH_MARKERS = (
    "GIT binary patch",
    "Binary files ",
    "new file mode ",
    "deleted file mode ",
    "rename from ",
    "rename to ",
    "copy from ",
    "copy to ",
    "--- /dev/null",
    "+++ /dev/null",
)


def _supported_patch_shape(patch: str, paths: list[str]) -> bool:
    if any(marker in patch for marker in _UNSUPPORTED_PATCH_MARKERS):
        return False
    pairs = re.findall(r"^diff --git a/(.+) b/(.+)$", patch, flags=re.MULTILINE)
    old_headers = re.findall(r"^--- a/(.+)$", patch, flags=re.MULTILINE)
    new_headers = re.findall(r"^\+\+\+ b/(.+)$", patch, flags=re.MULTILINE)
    if not pairs or len(pairs) != len(paths):
        return False
    pair_paths: list[str] = []
    for old_path, new_path in pairs:
        if old_path != new_path:
            return False
        path = PurePosixPath(old_path)
        if path.is_absolute() or ".." in path.parts or not old_path.strip():
            return False
        pair_paths.append(old_path)
    return (
        sorted(pair_paths) == sorted(paths)
        and sorted(old_headers) == sorted(paths)
        and sorted(new_headers) == sorted(paths)
    )


def _safe_read_event(event: Any) -> bool:
    command = event.command
    if event.exit_code != 0 or not READ_COMMAND_RE.search(command):
        return False
    if EDIT_COMMAND_RE.search(command) or re.search(r"\.write_(?:text|bytes)\s*\(", command):
        return False
    if re.search(r"(?:^|[;&|]\s*)(?:git\s+apply|apply_patch|tee\s|sed\s+-i\b)", command):
        return False
    if re.search(r"(?<![0-9>])>(?![>&])|>>", command):
        return False
    return True


def _observation_mentions_exact_path(text: str, path: str) -> bool:
    return path in text


def _grounded_observation(event: Any, paths: list[str], *, max_chars: int) -> dict[str, Any]:
    covered = sorted(path for path in paths if _observation_mentions_exact_path(event.output, path))
    marker = f"<grounded_paths>{','.join(covered)}</grounded_paths>\n"
    compact_budget = max(80, max_chars - len(marker))
    observation = _observation(event, max_chars=compact_budget)
    content = str(observation["content"])
    if content.startswith("OBSERVATION:\n"):
        content = "OBSERVATION:\n" + marker + content.removeprefix("OBSERVATION:\n")
    else:
        content = "OBSERVATION:\n" + marker + content
    observation["content"] = content
    return observation


def _select_grounded_reads(
    events: Sequence[Any],
    *,
    edit_event_index: int,
    paths: list[str],
    max_grounded_reads: int,
) -> tuple[list[Any] | None, list[str]]:
    if max_grounded_reads < 1:
        raise ValueError("max_grounded_reads must be positive")
    if len(paths) > max_grounded_reads:
        return None, sorted(paths)

    reads = [
        (index, event)
        for index, event in enumerate(events[:edit_event_index])
        if _safe_read_event(event)
    ]
    selected_indices: set[int] = set()
    missing: list[str] = []
    for path in paths:
        matching = [
            index
            for index, event in reads
            if _observation_mentions_exact_path(event.output, path)
        ]
        if not matching:
            missing.append(path)
            continue
        selected_indices.add(matching[-1])
    if missing:
        return None, sorted(missing)

    grounded_indices = [
        index
        for index, event in reads
        if any(_observation_mentions_exact_path(event.output, path) for path in paths)
    ]
    for index in reversed(grounded_indices):
        if len(selected_indices) >= max_grounded_reads:
            break
        selected_indices.add(index)
    return [events[index] for index in sorted(selected_indices)], []


def render_verified_teacher_finalpatch_sft(
    record: dict[str, Any],
    *,
    max_grounded_reads: int = 5,
    max_observation_chars: int = 2_400,
    max_f2p_tests: int = 20,
) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    """Render one resolved teacher attempt as a strict final-patch decision."""
    if max_f2p_tests < 1:
        raise ValueError("max_f2p_tests must be positive")
    result = record.get("result") or {}
    task = record.get("task") or {}
    if not isinstance(result, dict) or not result.get("resolved"):
        return None, {"kept": False, "reason": "not_f2p_resolved"}
    instance_id = str(task.get("instance_id") or result.get("instance_id") or "")
    if not instance_id or str(result.get("instance_id") or instance_id) != instance_id:
        return None, {"kept": False, "reason": "instance_id_mismatch"}

    f2p = parse_json_list(task.get("FAIL_TO_PASS"))
    if not f2p:
        return None, {"kept": False, "reason": "no_f2p_tests"}
    if len(f2p) > max_f2p_tests:
        return None, {
            "kept": False,
            "reason": "f2p_over_cap",
            "f2p_count": len(f2p),
            "max_f2p_tests": max_f2p_tests,
        }

    patch = str(record.get("patch") or "").rstrip("\n")
    paths = patch_paths(patch)
    if not patch.startswith("diff --git ") or not _source_only_patch(paths):
        return None, {"kept": False, "reason": "non_source_patch", "patch_paths": paths}
    if not _supported_patch_shape(patch, paths):
        return None, {"kept": False, "reason": "unsupported_patch_shape", "patch_paths": paths}
    if len(paths) > max_grounded_reads:
        return None, {
            "kept": False,
            "reason": "too_many_edited_paths",
            "patch_paths": paths,
            "max_grounded_reads": max_grounded_reads,
        }

    events, error = _extract_command_events(
        record.get("stream_lines") or [],
        stop_after=lambda command: _is_matching_source_edit(command, paths),
    )
    if events is None:
        return None, {"kept": False, "reason": error}
    edit_event_index = next(
        (index for index, event in enumerate(events) if _is_matching_source_edit(event.command, paths)),
        None,
    )
    if edit_event_index is None:
        return None, {"kept": False, "reason": "no_matching_source_edit", "patch_paths": paths}

    selected, missing_paths = _select_grounded_reads(
        events,
        edit_event_index=edit_event_index,
        paths=paths,
        max_grounded_reads=max_grounded_reads,
    )
    if selected is None:
        return None, {
            "kept": False,
            "reason": "edited_path_not_seen",
            "patch_paths": paths,
            "missing_patch_paths": missing_paths,
        }

    system_template = str(record.get("system_template") or "").strip()
    instance_template = str(record.get("instance_template") or "").strip()
    problem = _task_problem(task)
    if not system_template or "{{task}}" not in instance_template or not problem:
        return None, {"kept": False, "reason": "missing_native_template"}

    messages: list[dict[str, Any]] = [
        {"role": "system", "content": system_template, "tool_calls": []},
        {
            "role": "user",
            "content": instance_template.replace("{{task}}", problem),
            "tool_calls": [],
        },
    ]
    compacted_observations: list[dict[str, Any]] = []
    for event in selected:
        compacted_observations.append(
            _grounded_observation(event, paths, max_chars=max_observation_chars)
        )
    missing_after_compaction = [
        path
        for path in paths
        if not any(_observation_mentions_exact_path(observation["content"], path) for observation in compacted_observations)
    ]
    if missing_after_compaction:
        return None, {
            "kept": False,
            "reason": "edited_path_lost_after_compaction",
            "patch_paths": paths,
            "missing_patch_paths": sorted(missing_after_compaction),
        }
    for call_index, (event, observation) in enumerate(zip(selected, compacted_observations), start=1):
        messages.append(_assistant(event.command, call_index, loss=False))
        messages.append(observation)
    command = portable_git_apply_command(patch)
    messages.append(_assistant(command, len(selected) + 1, loss=True))

    row = {
        "instance_id": f"verified-teacher-finalpatch-{instance_id}",
        "source_instance_id": instance_id,
        "source": "python_verified_teacher_final_patch",
        "messages": messages,
    }
    return row, {
        "kept": True,
        "patch_paths": paths,
        "covered_patch_paths": sorted(paths),
        "original_edit_command_index": edit_event_index + 1,
        "kept_pre_edit_reads": len(selected),
        "f2p_count": len(f2p),
        "command_sha256": hashlib.sha256(command.encode("utf-8")).hexdigest(),
        "teacher_model": result.get("model"),
        "teacher_backend": result.get("backend"),
        "prompt_fallback_f2p": not bool(str(task.get("problem_statement") or "").strip()),
    }


def _credit_record(
    credit_ledger: Mapping[str, dict[str, Any]] | None,
    instance_id: str,
    command_sha256: str,
) -> tuple[bool, str | None]:
    if credit_ledger is None:
        return True, None
    credit = credit_ledger.get(instance_id)
    if credit is None:
        return False, "missing_independent_credit"
    if str(credit.get("command_sha256") or "") != command_sha256:
        raise ValueError(f"stale credit ledger for {instance_id}: command hash mismatch")
    if float(credit.get("tier", 0.0)) < 1.0:
        return False, "independent_credit_below_1.0"
    return True, None


def build_verified_teacher_finalpatch_rows(
    records: Iterable[dict[str, Any]],
    *,
    token_counter: Callable[[list[dict[str, Any]]], int],
    max_tokens: int = 49_152,
    max_grounded_reads: int = 5,
    max_observation_chars: int = 2_400,
    max_f2p_tests: int = 20,
    credit_ledger: Mapping[str, dict[str, Any]] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any], list[dict[str, Any]]]:
    """Select, deduplicate, budget, and optionally credit-filter final patches."""
    if max_tokens < 1:
        raise ValueError("max_tokens must be positive")
    candidates: dict[str, list[tuple[dict[str, Any], dict[str, Any], dict[str, Any], str]]] = {}
    rejected: list[dict[str, Any]] = []
    drops: Counter[str] = Counter()
    records_in = 0
    for index, record in enumerate(records):
        records_in += 1
        record_id = _record_identity(record, index)
        row, report = render_verified_teacher_finalpatch_sft(
            record,
            max_grounded_reads=max_grounded_reads,
            max_observation_chars=max_observation_chars,
            max_f2p_tests=max_f2p_tests,
        )
        if row is None:
            reason = str(report.get("reason") or "unknown")
            drops[reason] += 1
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
                len(str(item[2].get("patch") or "")),
                int(item[1].get("original_edit_command_index") or 1_000_000),
                item[3],
            ),
        )
        selected.append(attempts[0])
        for _, report, _, record_id in attempts[1:]:
            superseded += 1
            drops["superseded_verified_attempt"] += 1
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
    f2p_histogram: Counter[str] = Counter()
    seen_content: dict[str, str] = {}
    content_duplicates = 0
    budget_rejected = 0
    tier_one_credit_rows = 0
    for row, report, record, record_id in selected:
        credit_ok, credit_reason = _credit_record(
            credit_ledger,
            row["source_instance_id"],
            str(report["command_sha256"]),
        )
        if not credit_ok:
            assert credit_reason is not None
            drops[credit_reason] += 1
            rejected.append(
                {
                    "record_id": record_id,
                    "instance_id": row["source_instance_id"],
                    "reason": credit_reason,
                    "report": report,
                }
            )
            continue
        if credit_ledger is not None:
            tier_one_credit_rows += 1

        content = json.dumps(row["messages"], ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        content_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()
        if content_hash in seen_content:
            content_duplicates += 1
            drops["content_duplicate"] += 1
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
            drops["token_budget_exceeded"] += 1
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
        rows.append(row)
        token_counts.append(token_count)
        per_repo[str((record.get("task") or {}).get("repo") or "unknown")] += 1
        f2p_histogram[str(report["f2p_count"])] += 1

    token_counts.sort()
    manifest = {
        "schema_version": 1,
        "target_mode": "portable_teacher_final_patch",
        "records_in": records_in,
        "renderable_verified_attempts": sum(len(items) for items in candidates.values()),
        "unique_tasks_selected": len(selected),
        "superseded_verified_attempts": superseded,
        "content_duplicates": content_duplicates,
        "budget_rejected": budget_rejected,
        "rows_kept": len(rows),
        "max_tokens": max_tokens,
        "max_grounded_reads": max_grounded_reads,
        "max_observation_chars": max_observation_chars,
        "max_f2p_tests": max_f2p_tests,
        "credit_ledger_required": credit_ledger is not None,
        "tier_one_credit_rows": tier_one_credit_rows,
        "drop_reasons": dict(sorted(drops.items())),
        "per_repo": dict(sorted(per_repo.items())),
        "f2p_count_histogram": dict(sorted(f2p_histogram.items(), key=lambda item: int(item[0]))),
        "token_stats": {
            "min": token_counts[0] if token_counts else None,
            "median": token_counts[len(token_counts) // 2] if token_counts else None,
            "max": token_counts[-1] if token_counts else None,
        },
    }
    return rows, manifest, rejected


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number}: expected JSON object")
            rows.append(row)
    return rows


def _load_credit_ledger(path: Path | None) -> dict[str, dict[str, Any]] | None:
    if path is None:
        return None
    ledger: dict[str, dict[str, Any]] = {}
    for row in _load_jsonl(path):
        instance_id = str(row.get("instance_id") or "").strip()
        if not instance_id:
            raise ValueError(f"{path}: credit row missing instance_id")
        if instance_id in ledger:
            raise ValueError(f"{path}: duplicate credit for {instance_id}")
        ledger[instance_id] = row
    return ledger


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", type=Path, action="append", required=True)
    parser.add_argument("--run", dest="runs", type=Path, action="append", required=True)
    parser.add_argument("--config", type=Path, default=Path("phaseH_eval/swebench_edit_first_selfretry.yaml"))
    parser.add_argument("--base", default="/media/ironbcc/CrucialX10/models/google/gemma-4-31B-it")
    parser.add_argument("--out-jsonl", type=Path, required=True)
    parser.add_argument("--out-dataset", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--rejected-sidecar", type=Path, required=True)
    parser.add_argument("--credit-ledger", type=Path)
    parser.add_argument("--max-tokens", type=int, default=49_152)
    parser.add_argument("--max-grounded-reads", type=int, default=5)
    parser.add_argument("--max-observation-chars", type=int, default=2_400)
    parser.add_argument("--max-f2p-tests", type=int, default=20)
    args = parser.parse_args(argv)

    import yaml
    from datasets import Dataset
    from transformers import AutoTokenizer

    config = yaml.safe_load(args.config.read_text())
    records = load_teacher_records(
        args.tasks,
        args.runs,
        system_template=str(config["agent"]["system_template"]),
        instance_template=str(config["agent"]["instance_template"]),
    )
    credit_ledger = _load_credit_ledger(args.credit_ledger)
    tokenizer = AutoTokenizer.from_pretrained(args.base)
    rows, manifest, rejected = build_verified_teacher_finalpatch_rows(
        records,
        token_counter=lambda messages: _token_counter(tokenizer, messages),
        max_tokens=args.max_tokens,
        max_grounded_reads=args.max_grounded_reads,
        max_observation_chars=args.max_observation_chars,
        max_f2p_tests=args.max_f2p_tests,
        credit_ledger=credit_ledger,
    )
    manifest.update(
        {
            "tasks": [str(path) for path in args.tasks],
            "runs": [str(path) for path in args.runs],
            "config": str(args.config),
            "base_tokenizer": args.base,
            "credit_ledger": str(args.credit_ledger) if args.credit_ledger else None,
            "output_jsonl": str(args.out_jsonl),
            "output_dataset": str(args.out_dataset),
            "rejected_sidecar": str(args.rejected_sidecar),
            "rejected_records": len(rejected),
        }
    )
    _write_jsonl(args.out_jsonl, rows)
    _write_jsonl(args.rejected_sidecar, rejected)
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    Dataset.from_list(rows).save_to_disk(str(args.out_dataset))
    print(json.dumps(manifest, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
