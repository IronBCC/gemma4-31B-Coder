#!/usr/bin/env python3
"""Independently replay-audit a published Rust v3.1 agentic dataset."""
from __future__ import annotations

import argparse
from collections import Counter, deque
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path, PurePosixPath
from typing import Any, Callable, Mapping, Sequence

from phaseD_sft.build_rust_v3_agentic_dataset import (
    _canonical_json,
    _forbidden_training_path,
    _literal_scratch_creator,
    _max_read_streak,
    _mutation_scope,
    _native_command,
    _native_message_pairs,
    _normalized_command,
    _observation_succeeded,
    _percentile,
    _repository_mutation_fragment_scratch_inputs,
    _trusted_rust_verification,
    canonical_id_sha256,
    rendered_token_count,
)


RenderedTokenCounter = Callable[[list[dict[str, Any]]], int]
_COMPLETE_COMMAND = "echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT"
MIN_ROWS = 60
MIN_REPOSITORIES = 35
MAX_MEDIAN_FIRST_EDIT = 4
MAX_TOKENS = 49_152
_METADATA_FIELDS = (
    "content_sha256",
    "first_edit_command_index",
    "final_edit_command_index",
    "verification_command_index",
    "max_read_streak",
)
_PER_CONFIG_COUNTER_FIELDS = frozenset(
    {
        "scanned",
        "resolved",
        "rust",
        "structurally_eligible",
        "excluded",
        "grounding_passed",
        "verification_passed",
        "pairing_passed",
        "repeat_read_streak_passed",
        "compressed_eligible",
        "task_selected",
        "content_dedup_kept",
        "token_budget_kept",
    }
)
_V3P1_BEHAVIOR_DROP_REASONS = frozenset(
    {
        "ambiguous_mutation_path",
        "first_edit_not_grounded",
        "forbidden_mutation_path",
        "forbidden_patch_path",
        "invalid_compressed_provenance",
        "invalid_native_pairing",
        "invalid_scratch_chain",
        "invalid_suffix_boundary",
        "missing_repository_mutation",
        "missing_trusted_final_verification",
        "nonallowlisted_repository_mutation",
        "repeated_edit_command",
        "retained_decisive_indices_invalid",
        "terminal_marker_not_last",
    }
)

ZERO_VIOLATIONS = {
    "heldout_overlap": 0,
    "duplicate_task_ids": 0,
    "duplicate_content_hashes": 0,
    "forbidden_mutations": 0,
    "nonallowlisted_mutations": 0,
    "ambiguous_mutations": 0,
    "repeated_edit_commands": 0,
    "missing_trusted_final_verification": 0,
    "unbalanced_pairs": 0,
    "terminal_not_last": 0,
    "token_mismatch": 0,
    "token_overflow": 0,
    "manifest_mismatch": 0,
}


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _valid_sequence_of_strings(value: object) -> bool:
    return (
        isinstance(value, Sequence)
        and not isinstance(value, (str, bytes, bytearray))
        and all(isinstance(item, str) for item in value)
    )


def _row_analysis(row: Mapping[str, Any]) -> tuple[dict[str, int], dict[str, Any] | None]:
    observed = {
        key: 0
        for key in (
            "forbidden_mutations",
            "nonallowlisted_mutations",
            "ambiguous_mutations",
            "repeated_edit_commands",
            "missing_trusted_final_verification",
            "unbalanced_pairs",
            "terminal_not_last",
        )
    }
    messages = row.get("messages")
    allowlist_value = row.get("textual_patch_allowlist")
    declared_root = row.get("declared_root")
    if (
        not isinstance(messages, list)
        or not all(isinstance(message, Mapping) for message in messages)
        or not _valid_sequence_of_strings(allowlist_value)
        or not isinstance(declared_root, str)
        or not declared_root
    ):
        observed["ambiguous_mutations"] += 1
        observed["unbalanced_pairs"] += 1
        return observed, None

    terminal_indices = [
        index
        for index, message in enumerate(messages)
        if message.get("role") == "assistant"
        and _native_command(message) == _COMPLETE_COMMAND
    ]
    if terminal_indices != [len(messages) - 1]:
        observed["terminal_not_last"] += 1

    pairing = _native_message_pairs(messages)
    if pairing is None:
        observed["unbalanced_pairs"] += 1
        return observed, None
    pairs, _terminal_index = pairing

    allowlist = set(allowlist_value)
    classified: list[tuple[int, int, int, str, Any]] = []
    repository_edits: list[tuple[int, int, int, str, Any]] = []
    scratch_edits: list[tuple[int, int, int, str, Any]] = []
    seen_edit_commands: set[str] = set()
    first_edit = -1
    final_edit = -1
    for command_index, (assistant_index, observation_index, command) in enumerate(
        pairs, start=1
    ):
        scope = _mutation_scope(command, declared_root)
        entry = (command_index, assistant_index, observation_index, command, scope)
        classified.append(entry)
        if scope.ambiguous:
            observed["ambiguous_mutations"] += 1
            continue
        if scope.repository_paths or scope.scratch_paths:
            normalized = _normalized_command(command)
            if normalized in seen_edit_commands:
                observed["repeated_edit_commands"] += 1
            seen_edit_commands.add(normalized)
        for path in scope.repository_paths:
            if _forbidden_training_path(path):
                observed["forbidden_mutations"] += 1
            if path not in allowlist:
                observed["nonallowlisted_mutations"] += 1
        if scope.repository_paths:
            repository_edits.append(entry)
            if first_edit < 0:
                first_edit = command_index
            final_edit = command_index
        if scope.scratch_paths:
            scratch_edits.append(entry)

    scratch_valid = True
    if repository_edits:
        final_repository_edit = repository_edits[-1]
        scratch_inputs = _repository_mutation_fragment_scratch_inputs(
            final_repository_edit[3], declared_root=declared_root
        )
        if scratch_edits:
            scratch_valid = (
                len(scratch_edits) == 1
                and len(scratch_edits[0][4].scratch_paths) == 1
                and not scratch_edits[0][4].repository_paths
                and scratch_edits[0][1] < final_repository_edit[1]
            )
            if scratch_valid:
                scratch_path = scratch_edits[0][4].scratch_paths[0]
                scratch_valid = (
                    _literal_scratch_creator(scratch_edits[0][3], scratch_path)
                    and _observation_succeeded(messages[scratch_edits[0][2]])
                    and scratch_inputs == {scratch_path}
                )
        elif scratch_inputs:
            scratch_valid = False
    elif scratch_edits:
        scratch_valid = False
    if not scratch_valid:
        observed["ambiguous_mutations"] += 1

    verification = -1
    if final_edit > 0:
        for command_index, _assistant, observation_index, command, _scope in classified:
            if command_index <= final_edit:
                continue
            if _trusted_rust_verification(command, messages[observation_index]):
                verification = command_index
                break
    if final_edit < 0 or verification < 0:
        observed["missing_trusted_final_verification"] += 1

    derived = None
    if first_edit > 0 and final_edit > 0 and verification > 0:
        derived = {
            "content_sha256": _sha256(_canonical_json(messages)),
            "first_edit_command_index": first_edit,
            "final_edit_command_index": final_edit,
            "verification_command_index": verification,
            "max_read_streak": _max_read_streak(messages),
        }
    return observed, derived


def _audit_row_messages(row: Mapping[str, Any]) -> dict[str, int]:
    """Replay message pairing, mutation scope, scratch, and final verification."""
    return _row_analysis(row)[0]


def _metadata_audit(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    mismatches = {field: 0 for field in _METADATA_FIELDS}
    for row in rows:
        _observed, derived = _row_analysis(row)
        if derived is None:
            for field in _METADATA_FIELDS:
                mismatches[field] += 1
            continue
        for field in _METADATA_FIELDS:
            expected = derived[field]
            actual = row.get(field)
            if type(actual) is not type(expected) or actual != expected:
                mismatches[field] += 1
    return {
        "rows_checked": len(rows),
        "mismatch_count": sum(mismatches.values()),
        "field_mismatches": mismatches,
    }


def _token_stats(rows: Sequence[Mapping[str, Any]]) -> dict[str, int | float]:
    try:
        values = [int(row["tokens"]) for row in rows]
    except (KeyError, TypeError, ValueError):
        return {}
    return {
        "min": min(values, default=0),
        "median": _percentile(values, 50),
        "p95": _percentile(values, 95),
        "max": max(values, default=0),
    }


def _histogram(rows: Sequence[Mapping[str, Any]], field: str) -> dict[str, int]:
    try:
        return dict(sorted(Counter(str(row[field]) for row in rows).items()))
    except KeyError:
        return {}


def _nonnegative_int(value: object) -> bool:
    return type(value) is int and value >= 0


def _count_mapping(value: object) -> dict[str, int] | None:
    if not isinstance(value, Mapping):
        return None
    counts: dict[str, int] = {}
    for key, count in value.items():
        if not isinstance(key, str) or not key or not _nonnegative_int(count):
            return None
        counts[key] = count
    return counts


def _telemetry_matches(manifest: Mapping[str, Any]) -> bool:
    arithmetic = manifest.get("arithmetic")
    per_config = manifest.get("per_config")
    global_drops = _count_mapping(manifest.get("drop_reasons"))
    behavior_drops = _count_mapping(manifest.get("behavior_drop_reasons"))
    boundary = manifest.get("input_boundary")
    if (
        not isinstance(arithmetic, Mapping)
        or not isinstance(per_config, Mapping)
        or not per_config
        or global_drops is None
        or behavior_drops is None
        or not isinstance(boundary, Mapping)
    ):
        return False

    rows_scanned = arithmetic.get("rows_scanned")
    rows_dropped = arithmetic.get("rows_dropped")
    rows_output = arithmetic.get("rows_output")
    if (
        not all(
            _nonnegative_int(value)
            for value in (rows_scanned, rows_dropped, rows_output)
        )
        or arithmetic.get("balanced") is not True
        or rows_scanned != rows_dropped + rows_output
    ):
        return False

    scanned_sum = 0
    output_sum = 0
    structurally_eligible_sum = 0
    per_reason: Counter[str] = Counter()
    for config_key, counters in per_config.items():
        if not isinstance(config_key, str) or not config_key:
            return False
        if not isinstance(counters, Mapping):
            return False
        if set(counters) != _PER_CONFIG_COUNTER_FIELDS | {"drop_reasons"}:
            return False
        if not all(
            _nonnegative_int(counters[field])
            for field in _PER_CONFIG_COUNTER_FIELDS
        ):
            return False
        scanned = counters.get("scanned")
        resolved = counters.get("resolved")
        rust = counters.get("rust")
        output = counters.get("token_budget_kept")
        structurally_eligible = counters.get("structurally_eligible")
        excluded = counters.get("excluded")
        pairing_passed = counters.get("pairing_passed")
        verification_passed = counters.get("verification_passed")
        grounding_passed = counters.get("grounding_passed")
        repeat_read_streak_passed = counters.get("repeat_read_streak_passed")
        compressed_eligible = counters.get("compressed_eligible")
        task_selected = counters.get("task_selected")
        content_dedup_kept = counters.get("content_dedup_kept")
        config_drops = _count_mapping(counters.get("drop_reasons"))
        if (
            config_drops is None
            or scanned != output + sum(config_drops.values())
            or resolved > scanned
            or rust > scanned
            or excluded > scanned
            or structurally_eligible > resolved
            or structurally_eligible > rust
            or not (
                structurally_eligible
                >= pairing_passed
                >= verification_passed
                >= grounding_passed
                >= repeat_read_streak_passed
                >= compressed_eligible
            )
            or excluded != config_drops.get("excluded_task_id", 0)
            or compressed_eligible
            != task_selected
            + config_drops.get("task_shorter_candidate", 0)
            + sum(
                count
                for reason, count in config_drops.items()
                if reason in _V3P1_BEHAVIOR_DROP_REASONS
            )
            or task_selected
            != content_dedup_kept + config_drops.get("content_duplicate", 0)
            or content_dedup_kept
            != output + config_drops.get("token_budget_exceeded", 0)
        ):
            return False
        scanned_sum += scanned
        output_sum += output
        structurally_eligible_sum += structurally_eligible
        per_reason.update(config_drops)

    expected_boundary = boundary.get("expected_pre_exclusion_eligible")
    actual_boundary = boundary.get("actual_pre_exclusion_eligible")
    expected_behavior_drops = {
        reason: count
        for reason, count in global_drops.items()
        if reason in _V3P1_BEHAVIOR_DROP_REASONS
    }
    return bool(
        scanned_sum == rows_scanned
        and output_sum == rows_output
        and dict(sorted(per_reason.items())) == dict(sorted(global_drops.items()))
        and sum(global_drops.values()) == rows_dropped
        and behavior_drops == expected_behavior_drops
        and _nonnegative_int(expected_boundary)
        and _nonnegative_int(actual_boundary)
        and expected_boundary == structurally_eligible_sum
        and actual_boundary == structurally_eligible_sum
    )


def _manifest_matches_rows(
    manifest: Mapping[str, Any], rows: Sequence[Mapping[str, Any]]
) -> bool:
    """Recompute row hashes, counts, histograms, metadata, and gate measurements."""
    try:
        row_count = len(rows)
        repository_count = len({str(row["repository"]) for row in rows})
        per_repo = dict(
            sorted(Counter(str(row["repository"]) for row in rows).items())
        )
        metadata = _metadata_audit(rows)
        row_invariants = Counter()
        lockfile_mutations = 0
        for row in rows:
            row_invariants.update(_audit_row_messages(row))
            messages = row.get("messages")
            declared_root = row.get("declared_root")
            if isinstance(messages, list) and isinstance(declared_root, str):
                pairing = _native_message_pairs(messages)
                if pairing is not None:
                    for _assistant, _observation, command in pairing[0]:
                        scope = _mutation_scope(command, declared_root)
                        lockfile_mutations += sum(
                            PurePosixPath(path).name.lower() == "cargo.lock"
                            or PurePosixPath(path).name.lower().endswith(".lock")
                            for path in scope.repository_paths
                        )
        expected_invariants = {
            "forbidden_mutations": (
                row_invariants["forbidden_mutations"] - lockfile_mutations
            ),
            "lockfile_mutations": lockfile_mutations,
            "nonallowlisted_mutations": row_invariants["nonallowlisted_mutations"],
            "ambiguous_mutations": row_invariants["ambiguous_mutations"],
            "repeated_edit_commands": row_invariants["repeated_edit_commands"],
            "missing_trusted_final_verification": row_invariants[
                "missing_trusted_final_verification"
            ],
            "unbalanced_pairs": row_invariants["unbalanced_pairs"],
            "terminal_not_last": row_invariants["terminal_not_last"],
        }
        arithmetic = manifest.get("arithmetic")
        if not isinstance(arithmetic, Mapping):
            return False
        arithmetic_matches = (
            arithmetic.get("rows_output") == row_count
            and _telemetry_matches(manifest)
        )
        rows_digest = _sha256(b"".join(_canonical_json(row) for row in rows))
        task_digest = canonical_id_sha256(str(row["task_id"]) for row in rows)
        content_digest = canonical_id_sha256(
            str(row["content_sha256"]) for row in rows
        )
        gates = manifest.get("publication_gates")
        if not isinstance(gates, Mapping):
            return False
        minimum_rows = gates.get("minimum_rows")
        minimum_repositories = gates.get("minimum_repositories")
        median_gate = gates.get("maximum_median_first_edit")
        invariant_gate = gates.get("post_build_invariants")
        metadata_gate = gates.get("metadata_binding")
        if not all(
            isinstance(gate, Mapping)
            for gate in (
                minimum_rows,
                minimum_repositories,
                median_gate,
                invariant_gate,
                metadata_gate,
            )
        ):
            return False
        median_first_edit = _percentile(
            [int(row["first_edit_command_index"]) for row in rows], 50
        )
        invariants_zero = all(
            type(value) is int and value == 0
            for value in expected_invariants.values()
        )
        metadata_zero = (
            metadata["mismatch_count"] == 0
            and all(
                type(value) is int and value == 0
                for value in metadata["field_mismatches"].values()
            )
        )
        manifest_invariants = manifest.get("post_build_invariants")
        manifest_metadata = manifest.get("post_build_metadata")
        manifest_invariants_typed = (
            isinstance(manifest_invariants, Mapping)
            and set(manifest_invariants) == set(expected_invariants)
            and all(
                type(value) is int and value == 0
                for value in manifest_invariants.values()
            )
        )
        manifest_metadata_typed = (
            isinstance(manifest_metadata, Mapping)
            and type(manifest_metadata.get("rows_checked")) is int
            and manifest_metadata.get("rows_checked") == row_count
            and type(manifest_metadata.get("mismatch_count")) is int
            and manifest_metadata.get("mismatch_count") == 0
            and isinstance(manifest_metadata.get("field_mismatches"), Mapping)
            and set(manifest_metadata["field_mismatches"]) == set(_METADATA_FIELDS)
            and all(
                type(value) is int and value == 0
                for value in manifest_metadata["field_mismatches"].values()
            )
        )
        gate_matches = (
            type(minimum_rows.get("threshold")) is int
            and minimum_rows.get("threshold") == MIN_ROWS
            and type(minimum_rows.get("measured")) is int
            and minimum_rows.get("measured") == row_count
            and minimum_rows.get("passed") is True
            and row_count >= MIN_ROWS
            and type(minimum_repositories.get("threshold")) is int
            and minimum_repositories.get("threshold") == MIN_REPOSITORIES
            and type(minimum_repositories.get("measured")) is int
            and minimum_repositories.get("measured") == repository_count
            and minimum_repositories.get("passed") is True
            and repository_count >= MIN_REPOSITORIES
            and type(median_gate.get("threshold")) is int
            and median_gate.get("threshold") == MAX_MEDIAN_FIRST_EDIT
            and type(median_gate.get("measured")) is int
            and median_gate.get("measured") == median_first_edit
            and median_gate.get("passed") is True
            and median_first_edit <= MAX_MEDIAN_FIRST_EDIT
            and type(invariant_gate.get("required")) is int
            and invariant_gate.get("required") == 0
            and invariant_gate.get("passed") is True
            and invariants_zero
            and type(metadata_gate.get("required_mismatches")) is int
            and metadata_gate.get("required_mismatches") == 0
            and type(metadata_gate.get("measured_mismatches")) is int
            and metadata_gate.get("measured_mismatches") == 0
            and metadata_gate.get("passed") is True
            and metadata_zero
            and manifest_invariants_typed
            and manifest_metadata_typed
        )
        return bool(
            manifest.get("schema_version") == 1
            and manifest.get("behavior_contract") == "v3p1"
            and type(manifest.get("max_tokens")) is int
            and manifest.get("max_tokens") == MAX_TOKENS
            and manifest.get("rows_output") == row_count
            and arithmetic_matches
            and manifest.get("output_rows_sha256") == rows_digest
            and manifest.get("output_task_ids_sha256") == task_digest
            and manifest.get("output_content_hashes_sha256") == content_digest
            and manifest.get("per_repo") == per_repo
            and manifest.get("token_stats") == _token_stats(rows)
            and manifest.get("first_edit_histogram")
            == _histogram(rows, "first_edit_command_index")
            and manifest.get("read_streak_histogram")
            == _histogram(rows, "max_read_streak")
            and manifest.get("final_edit_histogram")
            == _histogram(rows, "final_edit_command_index")
            and manifest.get("verification_histogram")
            == _histogram(rows, "verification_command_index")
            and manifest_invariants == expected_invariants
            and manifest_metadata == metadata
            and gate_matches
        )
    except (KeyError, TypeError, ValueError):
        return False


def audit_published_dataset(
    dataset_dir: Path,
    exclusions: set[str],
    count_rendered_tokens: RenderedTokenCounter,
) -> dict[str, Any]:
    """Audit only immutable publication artifacts, never builder source inputs."""
    dataset_dir = Path(dataset_dir)
    dataset_bytes = (dataset_dir / "dataset.jsonl").read_bytes()
    rows = [
        json.loads(line)
        for line in dataset_bytes.decode("utf-8").splitlines()
        if line.strip()
    ]
    manifest_bytes = (dataset_dir / "manifest.json").read_bytes()
    manifest = json.loads(manifest_bytes)
    violations = dict(ZERO_VIOLATIONS)
    task_counts = Counter(str(row.get("task_id")) for row in rows)
    content_counts = Counter(
        _sha256(_canonical_json(row.get("messages"))) for row in rows
    )
    violations["heldout_overlap"] = sum(
        str(row.get("task_id")) in exclusions for row in rows
    )
    violations["duplicate_task_ids"] = sum(
        count - 1 for count in task_counts.values() if count > 1
    )
    violations["duplicate_content_hashes"] = sum(
        count - 1 for count in content_counts.values() if count > 1
    )
    for row in rows:
        observed = _audit_row_messages(row)
        for key, value in observed.items():
            violations[key] += value
        try:
            actual_tokens = count_rendered_tokens(row["messages"])
        except (KeyError, TypeError, ValueError):
            actual_tokens = 0
        violations["token_mismatch"] += (
            type(actual_tokens) is not int or actual_tokens != row.get("tokens")
        )
        violations["token_overflow"] += (
            type(actual_tokens) is not int
            or not 0 < actual_tokens <= MAX_TOKENS
        )
    manifest_matches = (
        manifest.get("output_jsonl_sha256") == _sha256(dataset_bytes)
        and _manifest_matches_rows(manifest, rows)
    )
    exclusion_manifest = manifest.get("exclusion")
    manifest_matches = manifest_matches and isinstance(exclusion_manifest, Mapping)
    if isinstance(exclusion_manifest, Mapping):
        manifest_matches = manifest_matches and (
            exclusion_manifest.get("count") == len(exclusions)
            and exclusion_manifest.get("sorted_ids_sha256")
            == canonical_id_sha256(exclusions)
        )
    violations["manifest_mismatch"] += not manifest_matches
    status = "passed" if not any(violations.values()) else "failed"
    return {
        "status": status,
        "rows": len(rows),
        "repositories": len({str(row.get("repository")) for row in rows}),
        "exclusions": len(exclusions),
        "violations": violations,
        "manifest_sha256": _sha256(manifest_bytes),
        "behavior_contract": manifest.get("behavior_contract"),
    }


def _load_exclusions(path: Path) -> set[str]:
    exclusions: list[str] = []
    for line_number, line in enumerate(path.read_text().splitlines(), start=1):
        if not line.strip():
            continue
        value = json.loads(line)
        if isinstance(value, str):
            task_id = value
        elif isinstance(value, Mapping):
            task_id = value.get("instance_id") or value.get("task_id")
        else:
            task_id = None
        if not isinstance(task_id, str) or not task_id:
            raise ValueError(f"invalid exclusion ID at line {line_number}")
        exclusions.append(task_id)
    if len(exclusions) != len(set(exclusions)):
        raise ValueError("exclusion file contains duplicate IDs")
    return set(exclusions)


def _load_tokenizer(identity: str, revision: str) -> Any:
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(identity, revision=revision)


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--exclude", type=Path, required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--tokenizer-revision", required=True)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args(argv)
    if not 1 <= args.workers <= 32:
        parser.error("--workers must be between 1 and 32")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    tokenizer = _load_tokenizer(args.tokenizer, args.tokenizer_revision)
    dataset_lines = [
        line
        for line in (args.data / "dataset.jsonl").read_text().splitlines()
        if line.strip()
    ]
    messages = [json.loads(line)["messages"] for line in dataset_lines]

    def count_rendered(row_messages: list[dict[str, Any]]) -> int:
        rendered = tokenizer.apply_chat_template(
            row_messages, tokenize=True, add_generation_prompt=False
        )
        return rendered_token_count(rendered)

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        counted = deque(executor.map(count_rendered, messages))

    def next_count(_messages: list[dict[str, Any]]) -> int:
        return counted.popleft()

    report = audit_published_dataset(
        args.data,
        exclusions=_load_exclusions(args.exclude),
        count_rendered_tokens=next_count,
    )
    print(json.dumps(report, ensure_ascii=False, sort_keys=True, separators=(",", ":")))
    return 0 if report["status"] == "passed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
