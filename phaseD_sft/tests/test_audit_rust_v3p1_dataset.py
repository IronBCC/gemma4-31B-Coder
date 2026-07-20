from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path

import pytest

import phaseD_sft.audit_rust_v3p1_dataset as auditor


def _canonical(value: object) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode()


def _assistant(command: str) -> dict[str, object]:
    return {
        "role": "assistant",
        "content": "",
        "tool_calls": [
            {
                "id": f"call-{hashlib.sha256(command.encode()).hexdigest()[:8]}",
                "type": "function",
                "function": {
                    "name": "bash",
                    "arguments": json.dumps({"command": command}),
                },
            }
        ],
    }


def _observation(returncode: int = 0) -> dict[str, str]:
    return {
        "role": "user",
        "content": f"OBSERVATION: command completed\n<returncode>{returncode}</returncode>",
    }


def _safe_messages(index: int = 0) -> list[dict[str, object]]:
    return [
        {"role": "system", "content": "You are a coding agent."},
        {
            "role": "user",
            "content": (
                "<uploaded_files>\n/testbed\n</uploaded_files>\n"
                f"Fix the bug for published row {index}."
            ),
        },
        _assistant("cat src/lib.rs"),
        _observation(),
        _assistant("sed -i 's/old/new/' src/lib.rs"),
        _observation(),
        _assistant("cargo test"),
        _observation(),
        _assistant("echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT"),
    ]


def _safe_row(index: int = 0) -> dict[str, object]:
    messages = _safe_messages(index)
    return {
        "task_id": f"owner__repo-{index}",
        "repository": f"owner/repo-{index % 35}",
        "trajectory_id": f"trajectory-{index}",
        "source": "open-swe",
        "config": "cfg",
        "split": "train",
        "messages": messages,
        "tokens": 321,
        "declared_root": "/testbed",
        "textual_patch_allowlist": ["src/lib.rs"],
        "content_sha256": hashlib.sha256(_canonical(messages)).hexdigest(),
        "first_edit_command_index": 2,
        "final_edit_command_index": 2,
        "verification_command_index": 3,
        "max_read_streak": 1,
    }


def _canonical_ids(values: list[str]) -> str:
    payload = "".join(f"{value}\n" for value in sorted(set(values))).encode()
    return hashlib.sha256(payload).hexdigest()


def _manifest(
    rows: list[dict[str, object]], jsonl: bytes, exclusions: set[str]
) -> dict[str, object]:
    tokens = [int(row["tokens"]) for row in rows]
    per_repo = dict(sorted(Counter(str(row["repository"]) for row in rows).items()))
    first_edits = dict(
        sorted(Counter(str(row["first_edit_command_index"]) for row in rows).items())
    )
    final_edits = dict(
        sorted(Counter(str(row["final_edit_command_index"]) for row in rows).items())
    )
    verifications = dict(
        sorted(Counter(str(row["verification_command_index"]) for row in rows).items())
    )
    read_streaks = dict(
        sorted(Counter(str(row["max_read_streak"]) for row in rows).items())
    )
    invariants = {
        "forbidden_mutations": 0,
        "lockfile_mutations": 0,
        "nonallowlisted_mutations": 0,
        "ambiguous_mutations": 0,
        "repeated_edit_commands": 0,
        "missing_trusted_final_verification": 0,
        "unbalanced_pairs": 0,
        "terminal_not_last": 0,
    }
    metadata_fields = {
        "content_sha256": 0,
        "first_edit_command_index": 0,
        "final_edit_command_index": 0,
        "verification_command_index": 0,
        "max_read_streak": 0,
    }
    row_count = len(rows)
    repository_count = len(per_repo)
    median_first_edit = sorted(
        int(row["first_edit_command_index"]) for row in rows
    )[(row_count - 1) // 2] if rows else 0
    return {
        "schema_version": 1,
        "behavior_contract": "v3p1",
        "max_tokens": 49_152,
        "exclusion": {
            "count": len(exclusions),
            "sorted_ids_sha256": _canonical_ids(list(exclusions)),
        },
        "rows_output": row_count,
        "arithmetic": {
            "rows_scanned": row_count,
            "rows_dropped": 0,
            "rows_output": row_count,
            "balanced": True,
        },
        "per_config": {
            "open-swe/cfg/train": {
                "scanned": row_count,
                "resolved": row_count,
                "rust": row_count,
                "structurally_eligible": row_count,
                "excluded": 0,
                "grounding_passed": row_count,
                "verification_passed": row_count,
                "pairing_passed": row_count,
                "repeat_read_streak_passed": row_count,
                "compressed_eligible": row_count,
                "task_selected": row_count,
                "content_dedup_kept": row_count,
                "token_budget_kept": row_count,
                "drop_reasons": {},
            }
        },
        "drop_reasons": {},
        "behavior_drop_reasons": {},
        "input_boundary": {
            "expected_pre_exclusion_eligible": row_count,
            "actual_pre_exclusion_eligible": row_count,
        },
        "output_jsonl_sha256": hashlib.sha256(jsonl).hexdigest(),
        "output_rows_sha256": hashlib.sha256(
            b"".join(_canonical(row) for row in rows)
        ).hexdigest(),
        "output_task_ids_sha256": _canonical_ids(
            [str(row["task_id"]) for row in rows]
        ),
        "output_content_hashes_sha256": _canonical_ids(
            [str(row["content_sha256"]) for row in rows]
        ),
        "per_repo": per_repo,
        "token_stats": {
            "min": min(tokens, default=0),
            "median": sorted(tokens)[(len(tokens) - 1) // 2] if tokens else 0,
            "p95": max(tokens, default=0),
            "max": max(tokens, default=0),
        },
        "first_edit_histogram": first_edits,
        "read_streak_histogram": read_streaks,
        "final_edit_histogram": final_edits,
        "verification_histogram": verifications,
        "post_build_invariants": invariants,
        "post_build_metadata": {
            "rows_checked": row_count,
            "mismatch_count": 0,
            "field_mismatches": metadata_fields,
        },
        "publication_gates": {
            "minimum_rows": {
                "threshold": 60,
                "measured": row_count,
                "passed": True,
            },
            "minimum_repositories": {
                "threshold": 35,
                "measured": repository_count,
                "passed": True,
            },
            "maximum_median_first_edit": {
                "threshold": 4,
                "measured": median_first_edit,
                "passed": median_first_edit <= 4,
            },
            "post_build_invariants": {"required": 0, "passed": True},
            "metadata_binding": {
                "required_mismatches": 0,
                "measured_mismatches": 0,
                "passed": True,
            },
        },
    }


def _published_fixture(
    tmp_path: Path,
    *,
    rows: list[dict[str, object]] | None = None,
    extra_mutation: str | None = None,
    exclusions: set[str] | None = None,
) -> Path:
    dataset = tmp_path / "dataset"
    dataset.mkdir(parents=True)
    published_rows = rows or [_safe_row()]
    frozen_exclusions = exclusions or set()
    if extra_mutation is not None:
        messages = published_rows[0]["messages"]
        assert isinstance(messages, list)
        messages[4] = _assistant(f"sed -i 's/old/new/' {extra_mutation}")
        published_rows[0]["content_sha256"] = hashlib.sha256(
            _canonical(messages)
        ).hexdigest()
    jsonl = b"".join(_canonical(row) + b"\n" for row in published_rows)
    (dataset / "dataset.jsonl").write_bytes(jsonl)
    (dataset / "manifest.json").write_bytes(
        _canonical(_manifest(published_rows, jsonl, frozen_exclusions)) + b"\n"
    )
    return dataset


def _passing_rows(first: dict[str, object] | None = None) -> list[dict[str, object]]:
    rows = [_safe_row(index) for index in range(60)]
    if first is not None:
        rows[0] = first
    return rows


def _audit(dataset: Path, *, exclusions: set[str] | None = None, tokens: int = 321):
    return auditor.audit_published_dataset(
        dataset,
        exclusions=exclusions or set(),
        count_rendered_tokens=lambda _messages: tokens,
    )


def test_auditor_recomputes_hashes_pairing_mutations_and_token_counts(
    tmp_path: Path,
) -> None:
    exclusions = {"heldout__repo-1"}
    dataset = _published_fixture(
        tmp_path, rows=_passing_rows(), exclusions=exclusions
    )
    report = _audit(dataset, exclusions=exclusions)
    assert report["status"] == "passed"
    assert report["rows"] == 60
    assert report["repositories"] == 35
    assert report["violations"] == auditor.ZERO_VIOLATIONS


@pytest.mark.parametrize(
    "mutation",
    ["tests/case.rs", "examples/demo.rs", "Cargo.lock", "README_FIX.md"],
)
def test_auditor_fails_for_forbidden_or_nonallowlisted_mutation(
    tmp_path: Path, mutation: str
) -> None:
    dataset = _published_fixture(tmp_path, extra_mutation=mutation)
    report = _audit(dataset)
    assert report["status"] == "failed"
    assert sum(report["violations"].values()) > 0


def test_auditor_fails_for_heldout_overlap(tmp_path: Path) -> None:
    report = _audit(
        _published_fixture(tmp_path), exclusions={"owner__repo-0", *map(str, range(238))}
    )
    assert report["violations"]["heldout_overlap"] == 1


def test_auditor_binds_frozen_exclusion_set_to_manifest(tmp_path: Path) -> None:
    dataset = _published_fixture(tmp_path, exclusions={"heldout__repo-1"})
    report = _audit(dataset, exclusions={"different__repo-1"})
    assert report["violations"]["manifest_mismatch"] == 1


@pytest.mark.parametrize("duplicate", ["task_id", "content_sha256"])
def test_auditor_fails_for_duplicate_identity(tmp_path: Path, duplicate: str) -> None:
    first = _safe_row()
    second = _safe_row()
    second["repository"] = "owner/repo2"
    if duplicate == "content_sha256":
        second["task_id"] = "owner__repo-2"
        second["content_sha256"] = "f" * 64
    else:
        messages = second["messages"]
        assert isinstance(messages, list)
        messages[1]["content"] = "Fix another bug."
        second["content_sha256"] = hashlib.sha256(_canonical(messages)).hexdigest()
    report = _audit(_published_fixture(tmp_path, rows=[first, second]))
    key = "duplicate_task_ids" if duplicate == "task_id" else "duplicate_content_hashes"
    assert report["violations"][key] == 1


@pytest.mark.parametrize(
    ("actual_tokens", "key"),
    [(322, "token_mismatch"), (49_153, "token_overflow"), (0, "token_overflow")],
)
def test_auditor_fails_for_token_mismatch_or_overflow(
    tmp_path: Path, actual_tokens: int, key: str
) -> None:
    report = _audit(_published_fixture(tmp_path), tokens=actual_tokens)
    assert report["violations"][key] == 1


def test_auditor_requires_trusted_final_verification(tmp_path: Path) -> None:
    row = _safe_row()
    messages = row["messages"]
    assert isinstance(messages, list)
    messages[7] = _observation(returncode=1)
    row["content_sha256"] = hashlib.sha256(_canonical(messages)).hexdigest()
    report = _audit(_published_fixture(tmp_path, rows=[row]))
    assert report["violations"]["missing_trusted_final_verification"] == 1


def test_auditor_rejects_repeated_edit(tmp_path: Path) -> None:
    row = _safe_row()
    messages = row["messages"]
    assert isinstance(messages, list)
    messages[6:6] = [messages[4].copy(), _observation()]
    row.update(
        content_sha256=hashlib.sha256(_canonical(messages)).hexdigest(),
        final_edit_command_index=3,
        verification_command_index=4,
    )
    report = _audit(_published_fixture(tmp_path, rows=[row]))
    assert report["violations"]["repeated_edit_commands"] == 1


def test_auditor_rejects_terminal_not_last_and_unbalanced_pair(tmp_path: Path) -> None:
    row = _safe_row()
    messages = row["messages"]
    assert isinstance(messages, list)
    messages.append({"role": "user", "content": "after terminal"})
    row["content_sha256"] = hashlib.sha256(_canonical(messages)).hexdigest()
    report = _audit(_published_fixture(tmp_path, rows=[row]))
    assert report["violations"]["terminal_not_last"] == 1
    assert report["violations"]["unbalanced_pairs"] == 1


def test_auditor_validates_scratch_chain_semantics(tmp_path: Path) -> None:
    row = _safe_row()
    messages = row["messages"]
    assert isinstance(messages, list)
    messages[4:4] = [
        _assistant("cat > /tmp/change.sed <<'EOF'\ns/old/new/\nEOF"),
        _observation(),
    ]
    messages[6] = _assistant("sed -i -f /tmp/change.sed src/lib.rs")
    row.update(
        content_sha256=hashlib.sha256(_canonical(messages)).hexdigest(),
        first_edit_command_index=3,
        final_edit_command_index=3,
        verification_command_index=4,
        max_read_streak=2,
    )
    assert _audit(
        _published_fixture(tmp_path, rows=_passing_rows(row))
    )["status"] == "passed"

    messages[5] = _observation(returncode=1)
    row["content_sha256"] = hashlib.sha256(_canonical(messages)).hexdigest()
    report = _audit(_published_fixture(tmp_path / "bad", rows=[row]))
    assert report["violations"]["ambiguous_mutations"] > 0


@pytest.mark.parametrize(
    "corruption",
    ["output_jsonl_sha256", "arithmetic", "per_repo", "first_edit_histogram", "gate"],
)
def test_auditor_fails_for_manifest_hash_arithmetic_or_summary_mismatch(
    tmp_path: Path, corruption: str
) -> None:
    dataset = _published_fixture(tmp_path)
    manifest_path = dataset / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if corruption == "output_jsonl_sha256":
        manifest[corruption] = "0" * 64
    elif corruption == "arithmetic":
        manifest[corruption]["rows_output"] = 2
    elif corruption == "per_repo":
        manifest[corruption] = {"wrong/repo": 1}
    elif corruption == "first_edit_histogram":
        manifest[corruption] = {"99": 1}
    else:
        manifest["publication_gates"]["minimum_rows"]["measured"] = 2
    manifest_path.write_bytes(_canonical(manifest) + b"\n")
    assert _audit(dataset)["violations"]["manifest_mismatch"] == 1


def test_auditor_fails_for_metadata_mismatch(tmp_path: Path) -> None:
    row = _safe_row()
    row["verification_command_index"] = 99
    report = _audit(_published_fixture(tmp_path, rows=[row]))
    assert report["violations"]["manifest_mismatch"] == 1


@pytest.mark.parametrize(
    ("section", "threshold"),
    [
        ("minimum_rows", 1),
        ("minimum_repositories", 1),
        ("maximum_median_first_edit", 99),
    ],
)
def test_auditor_rejects_redefined_fixed_publication_thresholds(
    tmp_path: Path, section: str, threshold: int
) -> None:
    dataset = _published_fixture(tmp_path, rows=_passing_rows())
    manifest_path = dataset / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["publication_gates"][section]["threshold"] = threshold
    manifest_path.write_bytes(_canonical(manifest) + b"\n")
    assert _audit(dataset)["violations"]["manifest_mismatch"] == 1


def test_auditor_rejects_redefined_fixed_max_tokens(tmp_path: Path) -> None:
    dataset = _published_fixture(tmp_path, rows=_passing_rows())
    manifest_path = dataset / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["max_tokens"] = 49_151
    manifest_path.write_bytes(_canonical(manifest) + b"\n")
    assert _audit(dataset)["violations"]["manifest_mismatch"] == 1


@pytest.mark.parametrize(
    "gate",
    [
        "minimum_rows",
        "minimum_repositories",
        "maximum_median_first_edit",
        "post_build_invariants",
        "metadata_binding",
    ],
)
def test_auditor_requires_every_publication_gate_to_have_passed(
    tmp_path: Path, gate: str
) -> None:
    dataset = _published_fixture(tmp_path, rows=_passing_rows())
    manifest_path = dataset / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["publication_gates"][gate]["passed"] = False
    manifest_path.write_bytes(_canonical(manifest) + b"\n")
    assert _audit(dataset)["violations"]["manifest_mismatch"] == 1


def test_auditor_rejects_self_reported_metadata_failure(tmp_path: Path) -> None:
    row = _safe_row()
    row["verification_command_index"] = 99
    dataset = _published_fixture(tmp_path, rows=_passing_rows(row))
    manifest_path = dataset / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    metadata = manifest["post_build_metadata"]
    metadata["mismatch_count"] = 1
    metadata["field_mismatches"]["verification_command_index"] = 1
    gate = manifest["publication_gates"]["metadata_binding"]
    gate["measured_mismatches"] = 1
    gate["passed"] = False
    manifest_path.write_bytes(_canonical(manifest) + b"\n")
    assert _audit(dataset)["status"] == "failed"
    assert _audit(dataset)["violations"]["manifest_mismatch"] == 1


@pytest.mark.parametrize(
    "section",
    ["per_config", "drop_reasons", "behavior_drop_reasons", "input_boundary"],
)
def test_auditor_requires_manifest_telemetry_sections(
    tmp_path: Path, section: str
) -> None:
    dataset = _published_fixture(tmp_path, rows=_passing_rows())
    manifest_path = dataset / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    del manifest[section]
    manifest_path.write_bytes(_canonical(manifest) + b"\n")
    assert _audit(dataset)["violations"]["manifest_mismatch"] == 1


@pytest.mark.parametrize(
    "corruption",
    [
        "per_config_scanned",
        "per_config_output",
        "per_config_balance",
        "global_reason_sum",
        "behavior_subset",
        "input_boundary",
        "balanced_not_literal_true",
    ],
)
def test_auditor_rejects_inconsistent_manifest_telemetry_even_when_top_balances(
    tmp_path: Path, corruption: str
) -> None:
    dataset = _published_fixture(tmp_path, rows=_passing_rows())
    manifest_path = dataset / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    config = manifest["per_config"]["open-swe/cfg/train"]
    if corruption == "per_config_scanned":
        config["scanned"] = 61
    elif corruption == "per_config_output":
        config["token_budget_kept"] = 59
    elif corruption == "per_config_balance":
        config["drop_reasons"] = {"bad": 1}
    elif corruption == "global_reason_sum":
        manifest["drop_reasons"] = {"bad": 1}
    elif corruption == "behavior_subset":
        manifest["behavior_drop_reasons"] = {"bad": 1}
    elif corruption == "input_boundary":
        manifest["input_boundary"]["actual_pre_exclusion_eligible"] = 59
    else:
        manifest["arithmetic"]["balanced"] = 1
    manifest_path.write_bytes(_canonical(manifest) + b"\n")
    assert manifest["arithmetic"]["rows_scanned"] == (
        manifest["arithmetic"]["rows_dropped"]
        + manifest["arithmetic"]["rows_output"]
    )
    assert _audit(dataset)["violations"]["manifest_mismatch"] == 1


def test_auditor_reconciles_builder_emitted_excluded_counter(tmp_path: Path) -> None:
    dataset = _published_fixture(tmp_path, rows=_passing_rows())
    manifest_path = dataset / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["per_config"]["open-swe/cfg/train"]["excluded"] = 999
    manifest_path.write_bytes(_canonical(manifest) + b"\n")
    assert _audit(dataset)["violations"]["manifest_mismatch"] == 1


def test_auditor_requires_known_behavior_drop_reason_to_be_reported(
    tmp_path: Path,
) -> None:
    dataset = _published_fixture(tmp_path, rows=_passing_rows())
    manifest_path = dataset / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    config = manifest["per_config"]["open-swe/cfg/train"]
    config["scanned"] = 61
    config["structurally_eligible"] = 61
    config["resolved"] = 61
    config["rust"] = 61
    config["pairing_passed"] = 61
    config["verification_passed"] = 61
    config["grounding_passed"] = 61
    config["repeat_read_streak_passed"] = 61
    config["compressed_eligible"] = 61
    config["drop_reasons"] = {"missing_repository_mutation": 1}
    manifest["drop_reasons"] = {"missing_repository_mutation": 1}
    manifest["behavior_drop_reasons"] = {}
    manifest["input_boundary"] = {
        "expected_pre_exclusion_eligible": 61,
        "actual_pre_exclusion_eligible": 61,
    }
    manifest["arithmetic"] = {
        "rows_scanned": 61,
        "rows_dropped": 1,
        "rows_output": 60,
        "balanced": True,
    }
    manifest_path.write_bytes(_canonical(manifest) + b"\n")
    assert _audit(dataset)["violations"]["manifest_mismatch"] == 1


def test_cli_emits_json_and_returns_two_for_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    dataset = _published_fixture(tmp_path)
    exclusion = tmp_path / "exclude.jsonl"
    exclusion.write_text(json.dumps({"instance_id": "owner__repo-0"}) + "\n")
    loads: list[tuple[str, str]] = []

    class Tokenizer:
        def apply_chat_template(self, _messages, **_kwargs):
            return list(range(321))

    def load(identity: str, revision: str):
        loads.append((identity, revision))
        return Tokenizer()

    monkeypatch.setattr(auditor, "_load_tokenizer", load)
    status = auditor.main(
        [
            "--data", str(dataset),
            "--exclude", str(exclusion),
            "--tokenizer", "tokenizer",
            "--tokenizer-revision", "revision",
            "--workers", "2",
        ]
    )
    report = json.loads(capsys.readouterr().out)
    assert status == 2
    assert report["violations"]["heldout_overlap"] == 1
    assert loads == [("tokenizer", "revision")]
