from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from phaseH_eval.v2p11_recovery_contract import validate_recovery_data


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, sort_keys=True) + "\n")


def _assistant(*, command: str | None = None, content: str = "") -> dict:
    message = {
        "role": "assistant",
        "content": content,
        "loss": True,
        "tool_calls": [],
    }
    if command is not None:
        message["tool_calls"] = [{
            "type": "function",
            "function": {
                "name": "bash",
                "arguments": json.dumps({"command": command}),
            },
        }]
    return message


def _dataset(tmp_path: Path) -> tuple[Path, Path, Path]:
    full_ids = tmp_path / "full_ids.json"
    _write_json(full_ids, ["eval-a", "eval-b"])
    source_exclusions = tmp_path / "source-exclusions.json"
    _write_json(source_exclusions, {"instance_ids": ["verified-a"]})
    source_exclusion_sha = hashlib.sha256(
        source_exclusions.read_bytes()
    ).hexdigest()
    combined_exclusions = tmp_path / "combined-exclusions.json"
    _write_json(combined_exclusions, {
        "instance_ids": ["eval-a", "eval-b", "other"],
        "repo_denylist": ["blocked/repo"],
    })
    sources = []
    rows = [
        {
            "instance_id": "train-a",
            "messages": [
                {"role": "system", "content": "fix", "tool_calls": []},
                _assistant(command="git apply - <<'PATCH'\ndiff --git a/x b/x\nPATCH"),
            ],
            "source": "revision",
        },
        {
            "instance_id": "train-b",
            "messages": [
                {"role": "system", "content": "fix", "tool_calls": []},
                _assistant(command="sed -i 's/old/new/' y.py"),
            ],
            "source": "success",
        },
    ]
    for index, row in enumerate(rows):
        root = tmp_path / f"source-{index}"
        root.mkdir()
        _write_json(root / "train.jsonl", row)
        manifest = {
            "schema_version": 2,
            "rendered": 1,
            "training_admitted": 1,
            "all_training_gates_complete": True,
            "exclusions": {
                "artifacts": [{
                    "path": str(source_exclusions),
                    "sha256": source_exclusion_sha,
                }],
            },
        }
        _write_json(root / "manifest.json", manifest)
        sources.append({
            "path": str(root),
            "rows": 1,
            "selected": 1,
            "manifest_sha256": hashlib.sha256(
                (root / "manifest.json").read_bytes()
            ).hexdigest(),
        })
    data = tmp_path / "recovery"
    data.mkdir()
    blended = [
        {
            "instance_id": row["instance_id"],
            "messages": row["messages"],
            "repo": "",
            "source": row["source"],
        }
        for row in rows
    ]
    with (data / "train.jsonl").open("w") as handle:
        for row in blended:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    train_sha = hashlib.sha256((data / "train.jsonl").read_bytes()).hexdigest()
    _write_json(data / "manifest.json", {
        "schema_version": 2,
        "kind": "teacher_blend",
        "rendered": 2,
        "training_admitted": 2,
        "all_training_gates_complete": True,
        "train_jsonl_sha256": train_sha,
        "sources": sources,
        "per_source": {
            str(tmp_path / "source-0"): 1,
            str(tmp_path / "source-1"): 1,
        },
        "standard_native_format_loss_gate": {
            "status": "passed",
            "samples": 2,
            "failure_count": 0,
        },
    })
    return data, full_ids, combined_exclusions


def test_validates_bound_decontaminated_portable_recovery(tmp_path: Path) -> None:
    data, full_ids, combined_exclusions = _dataset(tmp_path)

    result = validate_recovery_data(
        data,
        full_ids_path=full_ids,
        exclusions_path=combined_exclusions,
        expected_rows=2,
        gradient_accumulation=1,
        epochs=2,
    )

    assert result["rows"] == 2
    assert result["evaluation_overlap"] == 0
    assert result["optimizer_steps"] == 4
    assert result["rows_with_literal_patch_target"] == 1
    assert result["rows_with_supervised_source_mutation"] == 2
    assert result["combined_exclusion_ids"] == 3
    assert result["excluded_repositories"] == 1


def test_rejects_an_evaluation_id_in_recovery_rows(tmp_path: Path) -> None:
    data, full_ids, combined_exclusions = _dataset(tmp_path)
    rows = [
        json.loads(line)
        for line in (data / "train.jsonl").read_text().splitlines()
    ]
    rows[0]["instance_id"] = "eval-a"
    with (data / "train.jsonl").open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    manifest = json.loads((data / "manifest.json").read_text())
    manifest["train_jsonl_sha256"] = hashlib.sha256(
        (data / "train.jsonl").read_bytes()
    ).hexdigest()
    _write_json(data / "manifest.json", manifest)

    with pytest.raises(ValueError, match="combined evaluation exclusion"):
        validate_recovery_data(
            data,
            full_ids_path=full_ids,
            exclusions_path=combined_exclusions,
            expected_rows=2,
        )


def test_rejects_changed_source_after_blend(tmp_path: Path) -> None:
    data, full_ids, combined_exclusions = _dataset(tmp_path)
    source = tmp_path / "source-0" / "train.jsonl"
    row = json.loads(source.read_text())
    row["messages"][0]["content"] = "changed after blend"
    _write_json(source, row)

    with pytest.raises(ValueError, match="blend differs"):
        validate_recovery_data(
            data,
            full_ids_path=full_ids,
            exclusions_path=combined_exclusions,
            expected_rows=2,
        )


def test_rejects_harness_specific_submission_target(tmp_path: Path) -> None:
    data, full_ids, combined_exclusions = _dataset(tmp_path)
    source = tmp_path / "source-0" / "train.jsonl"
    row = json.loads(source.read_text())
    row["messages"][-1]["tool_calls"][0]["function"]["arguments"] = json.dumps({
        "command": (
            "git apply - <<'PATCH'\ndiff --git a/x b/x\nPATCH\n"
            "echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT"
        ),
    })
    _write_json(source, row)
    output_rows = [
        json.loads(line)
        for line in (data / "train.jsonl").read_text().splitlines()
    ]
    output_rows[0]["messages"] = row["messages"]
    with (data / "train.jsonl").open("w") as handle:
        for output in output_rows:
            handle.write(json.dumps(output, sort_keys=True) + "\n")
    manifest = json.loads((data / "manifest.json").read_text())
    manifest["sources"][0]["manifest_sha256"] = hashlib.sha256(
        (tmp_path / "source-0" / "manifest.json").read_bytes()
    ).hexdigest()
    manifest["train_jsonl_sha256"] = hashlib.sha256(
        (data / "train.jsonl").read_bytes()
    ).hexdigest()
    _write_json(data / "manifest.json", manifest)

    with pytest.raises(ValueError, match="submission marker"):
        validate_recovery_data(
            data,
            full_ids_path=full_ids,
            exclusions_path=combined_exclusions,
            expected_rows=2,
        )


def test_rejects_supervised_legacy_edit_before_verifier_feedback(
    tmp_path: Path,
) -> None:
    data, full_ids, combined_exclusions = _dataset(tmp_path)
    source = tmp_path / "source-0" / "train.jsonl"
    row = json.loads(source.read_text())
    row["messages"] = [
        {"role": "system", "content": "fix", "tool_calls": []},
        _assistant(command="sed -i 's/right/wrong/' x.py"),
        {
            "role": "user",
            "content": (
                "VERIFIER FEEDBACK:\nThe legacy candidate does not repair "
                "the bug-mutated task."
            ),
            "tool_calls": [],
        },
        _assistant(
            command=(
                "git apply - <<'PATCH'\n"
                "diff --git a/x b/x\n"
                "PATCH"
            )
        ),
    ]
    _write_json(source, row)
    output_rows = [
        json.loads(line)
        for line in (data / "train.jsonl").read_text().splitlines()
    ]
    output_rows[0]["messages"] = row["messages"]
    with (data / "train.jsonl").open("w") as handle:
        for output in output_rows:
            handle.write(json.dumps(output, sort_keys=True) + "\n")
    manifest = json.loads((data / "manifest.json").read_text())
    manifest["train_jsonl_sha256"] = hashlib.sha256(
        (data / "train.jsonl").read_bytes()
    ).hexdigest()
    _write_json(data / "manifest.json", manifest)

    with pytest.raises(ValueError, match="before verifier feedback"):
        validate_recovery_data(
            data,
            full_ids_path=full_ids,
            exclusions_path=combined_exclusions,
            expected_rows=2,
        )


def test_rejects_prose_only_diff_without_source_mutation(
    tmp_path: Path,
) -> None:
    data, full_ids, combined_exclusions = _dataset(tmp_path)
    source = tmp_path / "source-0" / "train.jsonl"
    source_row = json.loads(source.read_text())
    source_row["messages"][-1] = _assistant(
        content="diff --git a/x b/x\n"
    )
    _write_json(source, source_row)
    output_rows = [
        json.loads(line)
        for line in (data / "train.jsonl").read_text().splitlines()
    ]
    output_rows[0]["messages"] = source_row["messages"]
    with (data / "train.jsonl").open("w") as handle:
        for output in output_rows:
            handle.write(json.dumps(output, sort_keys=True) + "\n")
    manifest = json.loads((data / "manifest.json").read_text())
    manifest["train_jsonl_sha256"] = hashlib.sha256(
        (data / "train.jsonl").read_bytes()
    ).hexdigest()
    _write_json(data / "manifest.json", manifest)

    with pytest.raises(ValueError, match="portable source mutation"):
        validate_recovery_data(
            data,
            full_ids_path=full_ids,
            exclusions_path=combined_exclusions,
            expected_rows=2,
        )


def test_rejects_combined_exclusion_missing_full_evaluation_id(
    tmp_path: Path,
) -> None:
    data, full_ids, combined_exclusions = _dataset(tmp_path)
    _write_json(combined_exclusions, {
        "instance_ids": ["eval-a"],
        "repo_denylist": [],
    })

    with pytest.raises(ValueError, match="combined evaluation exclusion"):
        validate_recovery_data(
            data,
            full_ids_path=full_ids,
            exclusions_path=combined_exclusions,
            expected_rows=2,
        )


def test_rejects_recovery_repo_in_combined_exclusion(tmp_path: Path) -> None:
    data, full_ids, combined_exclusions = _dataset(tmp_path)
    rows = [
        json.loads(line)
        for line in (data / "train.jsonl").read_text().splitlines()
    ]
    rows[0]["repo"] = "blocked/repo"
    with (data / "train.jsonl").open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    manifest = json.loads((data / "manifest.json").read_text())
    manifest["train_jsonl_sha256"] = hashlib.sha256(
        (data / "train.jsonl").read_bytes()
    ).hexdigest()
    _write_json(data / "manifest.json", manifest)

    with pytest.raises(ValueError, match="combined evaluation exclusion"):
        validate_recovery_data(
            data,
            full_ids_path=full_ids,
            exclusions_path=combined_exclusions,
            expected_rows=2,
        )
