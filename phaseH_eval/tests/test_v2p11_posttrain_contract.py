from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from phaseE_rl.kto_swe_outcome import BEHAVIOR_MANIFEST_SCHEMA
from phaseH_eval.v2p11_posttrain_contract import validate_posttrain_inputs


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")


def _assistant(command: str, *, loss: bool) -> dict:
    return {
        "role": "assistant",
        "content": "",
        "loss": loss,
        "tool_calls": [
            {
                "id": hashlib.sha256(command.encode()).hexdigest()[:12],
                "type": "function",
                "function": {
                    "name": "bash",
                    "arguments": json.dumps({"command": command}),
                },
            }
        ],
    }


def _fixture(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    full_ids = tmp_path / "full.json"
    _write_json(full_ids, ["eval-a", "eval-b"])
    exclusions = tmp_path / "exclusions.json"
    _write_json(
        exclusions,
        {
            "instance_ids": ["eval-a", "eval-b", "verified-a"],
            "repo_denylist": ["eval/repo"],
        },
    )

    recovery = tmp_path / "recovery"
    recovery.mkdir()
    recovery_rows = [
        {
            "instance_id": "safe-task::style-a",
            "source_instance_id": "safe-task",
            "repo": "safe/repo",
            "messages": [
                {"role": "system", "content": "fix", "tool_calls": []},
                _assistant("sed -n '1,20p' src/x.py", loss=False),
                _assistant("sed -i 's/old/new/' src/x.py", loss=True),
                _assistant("python -m pytest tests/test_x.py -q", loss=True),
            ],
        }
    ]
    recovery_train = recovery / "train.jsonl"
    recovery_train.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in recovery_rows),
        encoding="utf-8",
    )
    _write_json(
        recovery / "manifest.json",
        {
            "schema_version": 2,
            "kind": "v2p11_recovery_curriculum",
            "source_rows": 1,
            "rendered": 1,
            "training_admitted": 1,
            "all_training_gates_complete": True,
            "train_jsonl_sha256": hashlib.sha256(
                recovery_train.read_bytes()
            ).hexdigest(),
            "exclusion_sha256": hashlib.sha256(exclusions.read_bytes()).hexdigest(),
            "evaluation_overlap": 0,
            "private_submission_marker_hits": 0,
            "target_counts": {
                "rows_with_passing_test_target": 1,
                "rows_with_source_mutation_target": 1,
                "supervised_assistant_messages": 2,
            },
            "standard_native_format_loss_gate": {
                "status": "passed",
                "samples": 1,
                "failure_count": 0,
            },
        },
    )

    behavior_data = tmp_path / "behavior.jsonl"
    behavior_rows = []
    categories = []
    for index in range(500):
        if index < 150:
            category = "wrong_nonempty_replay"
            label = False
        else:
            category = "desirable_correct_patch"
            label = True
        prompt = f"prompt-{index}"
        completion = f"completion-{index}"
        behavior_rows.append(
            {
                "source_instance_id": f"safe__task-{index}",
                "prompt": prompt,
                "completion": completion,
                "label": label,
                "raw_token_count": 10,
                "token_count": 12,
                "provenance": {"punishment_category": category},
                "provenance_hashes": {
                    "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
                    "completion_sha256": hashlib.sha256(
                        completion.encode()
                    ).hexdigest(),
                    "stock_chat_template_sha256": "a" * 64,
                    "tool_schema_sha256": "b" * 64,
                },
            }
        )
        categories.append(category)
    behavior_data.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in behavior_rows),
        encoding="utf-8",
    )
    behavior_manifest = tmp_path / "behavior_manifest.json"
    _write_json(
        behavior_manifest,
        {
            "schema": BEHAVIOR_MANIFEST_SCHEMA,
            "row_count": 500,
            "desirable_count": 350,
            "undesirable_count": 150,
            "replay_verified_rows": 500,
            "structural_negative_rows": 0,
            "behavior_negative_counts": {
                "empty_terminal": 0,
                "repeated_read_loop": 0,
                "wrong_nonempty_replay": 150,
            },
            "immutable_max_tokens": 4096,
            "max_token_count": 12,
            "truncated_rows": 0,
            "render_failures": 0,
            "format_failures": 0,
            "duplicate_rows": 0,
            "evaluation_leakage_hits": 0,
            "format_loss_checked_rows": 500,
            "format_loss_failures": 0,
            "pre_rendered": True,
            "stock_native_template": True,
            "stock_chat_template_sha256": "a" * 64,
            "tool_schema_sha256": "b" * 64,
            "private_marker_positive_rows": 0,
            "full_evaluation_ids": 500,
            "full_evaluation_repositories": 12,
            "output_jsonl_sha256": hashlib.sha256(
                behavior_data.read_bytes()
            ).hexdigest(),
            "exclusion_sha256": hashlib.sha256(exclusions.read_bytes()).hexdigest(),
        },
    )
    return recovery, behavior_data, behavior_manifest, full_ids


def test_validates_recovery_and_preference_inputs_as_one_bound_contract(
    tmp_path: Path,
) -> None:
    recovery, behavior_data, behavior_manifest, full_ids = _fixture(tmp_path)

    result = validate_posttrain_inputs(
        recovery_data=recovery,
        behavior_data=behavior_data,
        behavior_manifest=behavior_manifest,
        exclusions=tmp_path / "exclusions.json",
        full_ids=full_ids,
        expected_recovery_rows=1,
        recovery_epochs=2,
        recovery_gradient_accumulation=1,
    )

    assert result["recovery_rows"] == 1
    assert result["recovery_optimizer_steps"] == 2
    assert result["behavior_rows"] == 500
    assert result["evaluation_overlap"] == 0


def test_rejects_a_recovery_source_id_from_the_evaluation_panel(
    tmp_path: Path,
) -> None:
    recovery, behavior_data, behavior_manifest, full_ids = _fixture(tmp_path)
    row = json.loads((recovery / "train.jsonl").read_text(encoding="utf-8"))
    row["source_instance_id"] = "eval-a"
    train = recovery / "train.jsonl"
    train.write_text(json.dumps(row, sort_keys=True) + "\n", encoding="utf-8")
    manifest = json.loads((recovery / "manifest.json").read_text(encoding="utf-8"))
    manifest["train_jsonl_sha256"] = hashlib.sha256(train.read_bytes()).hexdigest()
    _write_json(recovery / "manifest.json", manifest)

    with pytest.raises(ValueError, match="evaluation overlap"):
        validate_posttrain_inputs(
            recovery_data=recovery,
            behavior_data=behavior_data,
            behavior_manifest=behavior_manifest,
            exclusions=tmp_path / "exclusions.json",
            full_ids=full_ids,
            expected_recovery_rows=1,
        )
