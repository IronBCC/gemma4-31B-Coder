from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys

from datasets import Dataset, load_from_disk
import pytest

import phaseD_sft.build_mix_v2p10 as mix_module
from phaseD_sft.build_mix_v2p10 import build_mix, canonical_rows_sha256


def _messages(instance_id: str) -> list[dict]:
    return [
        {
            "role": "user",
            "content": f"Fix {instance_id}.",
            "tool_calls": [],
            "loss": False,
        },
        {
            "role": "assistant",
            "content": "Done.",
            "tool_calls": [],
            "loss": True,
        },
    ]


def _write_dataset(
    path: Path,
    *,
    count: int,
    source: str,
    schema_version: int = 2,
    training_admitted: bool = True,
    variant: str | None = None,
) -> Path:
    rows = [
        {
            "instance_id": f"{source.replace(':', '_')}__{index}",
            "messages": _messages(str(index)),
            "repo": f"fixture/repo-{index}",
            "source": source,
            **(
                {
                    "n_steps": 2,
                    "distilled_source_sha256": hashlib.sha256(
                        f"distilled-{index}".encode()
                    ).hexdigest(),
                }
                if source == "teacher:claude:claude-fable-5"
                else {}
            ),
        }
        for index in range(count)
    ]
    Dataset.from_list(rows).save_to_disk(str(path))
    train = path / "train.jsonl"
    train.write_text(
        "".join(
            json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n"
            for row in rows
        )
    )
    exclusion = path.parent / "fixture-exclusions.json"
    if not exclusion.exists():
        exclusion.write_text(json.dumps({
            "instance_ids": [],
            "repo_denylist": [],
        }))
    manifest = {
        "schema_version": schema_version,
        "complete": True,
        "dataset_variant": variant,
        "rendered": count,
        "training_admitted": count if training_admitted else 0,
        "all_training_gates_complete": training_admitted,
        "train_jsonl_sha256": hashlib.sha256(train.read_bytes()).hexdigest(),
        "canonical_rows_sha256": canonical_rows_sha256(rows),
        "success_path_distilled": True,
        "exclusions": {
            "instance_ids": 0,
            "repositories": 0,
            "artifacts": [{
                "path": str(exclusion),
                "sha256": hashlib.sha256(exclusion.read_bytes()).hexdigest(),
            }],
        },
        "artifact_bindings": [
            {
                "instance_id": row["instance_id"],
                "stream_sha256": "a" * 64,
                "patch_sha256": "b" * 64,
                "admission_evidence_sha256": "c" * 64,
                "task_contract_sha256": "d" * 64,
                "content_sha256": hashlib.sha256(
                    json.dumps(row["messages"], sort_keys=True).encode()
                ).hexdigest(),
                "distilled_source_sha256": hashlib.sha256(
                    row["instance_id"].encode()
                ).hexdigest(),
                "retained_commands": [
                    "sed -i 's/a/b/' src/x.py",
                    "python -m pytest -q tests/test_x.py::test_fix",
                ],
            }
            for row in rows
        ],
    }
    (path / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    return path


def _base(tmp_path: Path) -> Path:
    return _write_dataset(
        tmp_path / "teacher_train_mix_v2p8",
        count=4,
        source="open-swe",
        variant="teacher_train_mix_v2p8",
    )


def _write_gpt56sol_fixture(path: Path, *, count: int) -> tuple[Path, list[dict]]:
    rows = []
    for index in range(count):
        messages = _messages(str(index))
        messages[0].pop("loss")
        rows.append({
            "instance_id": f"gpt-task-{index}",
            "source_instance_id": f"gpt-task-{index}",
            "source_trajectory_id": f"gpt-task-{index}:trace",
            "repo": f"moonshiner/gpt-task-{index}",
            "source": (
                "teacher:gpt56sol:fixture-revision:"
                f"gpt-task-{index}:trace"
            ),
            "language": "python",
            "messages": messages,
            "content_sha256": hashlib.sha256(
                json.dumps(
                    messages,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode()
            ).hexdigest(),
            "source_terminal_sha256": "a" * 64,
            "replay_trajectory_id": "b" * 64,
            "replay_operation_sha256": "c" * 64,
            "estimated_tokens": 100,
            "n_steps": 1,
            "first_edit_cmd": 1,
            "max_read_streak": 1,
        })
    Dataset.from_list(rows).save_to_disk(str(path))
    train = path / "train.jsonl"
    train.write_text(
        "".join(
            json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n"
            for row in rows
        )
    )
    exclusion = path.parent / "gpt-exclusions.json"
    exclusion.write_text(json.dumps({
        "instance_ids": [],
        "repo_denylist": [],
    }))
    format_report = path / "format-gate.fixture.json"
    format_report.write_text("{}\n")
    manifest = {
        "schema_version": 1,
        "source": {
            "dataset_id": (
                "greghavens/gpt-5.6-sol-coding-and-debugging-traces"
            ),
            "dataset_revision": "fixture-revision",
        },
        "functionally_verified_input": count,
        "output_rows": count,
        "training_admitted": count,
        "all_training_gates_complete": True,
        "gates": {
            "input_rows": count,
            "output_rows": count,
            "behavior_rejected": 0,
            "decontamination_rejected": 0,
            "dedup_removed": 0,
            "format_failures": 0,
            "token_rejected": 0,
        },
        "replay_manifest_sha256": "d" * 64,
        "replay_results_artifact": {
            "generation": 1,
            "path": "results.fixture.jsonl",
            "rows": count,
            "sha256": "e" * 64,
        },
        "exclusions": {
            "instance_ids": 0,
            "repositories": 0,
            "artifacts": [{
                "path": str(exclusion),
                "sha256": hashlib.sha256(exclusion.read_bytes()).hexdigest(),
            }],
        },
        "standard_native_format_loss_gate": {
            "status": "passed",
            "samples": count,
            "failure_count": 0,
            "artifact": {
                "path": format_report.name,
                "sha256": hashlib.sha256(
                    format_report.read_bytes()
                ).hexdigest(),
                "bytes": format_report.stat().st_size,
            },
        },
        "train_jsonl_sha256": hashlib.sha256(train.read_bytes()).hexdigest(),
    }
    (path / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    return path, rows


def _write_fable_revision_fixture(path: Path, *, count: int) -> Path:
    source = "teacher:claude:claude-fable-5:verified-revision"
    path = _write_dataset(path, count=count, source=source)
    manifest_path = path / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    rows = [
        json.loads(line)
        for line in (path / "train.jsonl").read_text().splitlines()
        if line.strip()
    ]
    bindings = []
    for row in rows:
        row["source_instance_id"] = row["instance_id"]
        row["legacy_fable_conditioning"] = True
        row["oracle_repair_target"] = True
        row["content_sha256"] = hashlib.sha256(
            json.dumps(
                row["messages"],
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode()
        ).hexdigest()
        bindings.append({
            "instance_id": row["instance_id"],
            "source_instance_id": row["source_instance_id"],
            "content_sha256": row["content_sha256"],
            "legacy_stream_sha256": "a" * 64,
            "legacy_patch_sha256": "b" * 64,
            "repair_patch_sha256": "c" * 64,
            "repair_admission_evidence_sha256": "d" * 64,
            "task_contract_sha256": "e" * 64,
            "supervised_fable_inspection_turns": 1,
            "masked_legacy_turns": 1,
        })
    shutil.rmtree(path)
    Dataset.from_list(rows).save_to_disk(str(path))
    train = path / "train.jsonl"
    train.write_text(
        "".join(
            json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n"
            for row in rows
        )
    )
    manifest.update({
        "dataset_variant": "fable5_recent_verified_revision_v1",
        "source": source,
        "verified_revision_supervision": True,
        "success_path_distilled": False,
        "legacy_mutations_supervised": 0,
        "legacy_test_runs_supervised": 0,
        "repair_targets_exact_replay": True,
        "oracle_repair_targets": True,
        "artifact_bindings": bindings,
        "train_jsonl_sha256": hashlib.sha256(train.read_bytes()).hexdigest(),
        "canonical_rows_sha256": canonical_rows_sha256(rows),
    })
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return path


def test_v2p10_builder_rejects_f2p_only_legacy_rows(tmp_path: Path) -> None:
    base = _base(tmp_path)
    legacy = _write_dataset(
        tmp_path / "legacy",
        count=30,
        schema_version=1,
        training_admitted=True,
        source="teacher:claude:claude-fable-5",
    )

    with pytest.raises(ValueError, match="unregistered schema-1 teacher delta"):
        build_mix(base, legacy, tmp_path / "mix")

    assert not (tmp_path / "mix").exists()


def test_v2p10_builder_rejects_a_second_source(tmp_path: Path) -> None:
    base = _base(tmp_path)
    fable = _write_dataset(
        tmp_path / "fable",
        count=30,
        source="teacher:claude:claude-fable-5",
    )
    gpt = _write_dataset(
        tmp_path / "gpt",
        count=30,
        source="teacher:gpt56sol:pinned:one",
    )

    with pytest.raises(ValueError, match="Fable delta"):
        build_mix(base, fable, tmp_path / "mix", extra=(gpt,))


def test_v2p10_builder_publishes_pending_one_variable_mix_atomically(
    tmp_path: Path,
) -> None:
    base = _base(tmp_path)
    fable = _write_dataset(
        tmp_path / "fable",
        count=30,
        source="teacher:claude:claude-fable-5",
    )
    output = tmp_path / "mix"

    manifest = build_mix(base, fable, output)

    assert manifest["schema_version"] == 2
    assert manifest["dataset_variant"] == "teacher_train_mix_v2p10"
    assert manifest["base_rows"] == 4
    assert manifest["teacher_rows"] == 30
    assert manifest["teacher_delta_policy"] == "strict_fable_schema2"
    assert manifest["rendered"] == 34
    assert manifest["training_admitted"] == 0
    assert manifest["all_training_gates_complete"] is False
    assert len(load_from_disk(str(output))) == 34
    trainer_rows = [
        json.loads(line)
        for line in (output / "train.jsonl").read_text().splitlines()
        if line.strip()
    ]
    arrow_rows = [dict(row) for row in load_from_disk(str(output))]
    assert canonical_rows_sha256(trainer_rows) == canonical_rows_sha256(
        arrow_rows
    )
    assert all(
        set(row) == {"instance_id", "messages", "repo", "source"}
        for row in trainer_rows
    )
    assert all(
        type(message["loss"]) is bool
        for row in trainer_rows
        for message in row["messages"]
    )
    assert hashlib.sha256((output / "train.jsonl").read_bytes()).hexdigest() == (
        manifest["train_jsonl_sha256"]
    )
    assert not list(tmp_path.glob(".mix.*"))

    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        build_mix(base, fable, output)


def test_v2p10_builder_requires_explicit_override_for_exhausted_small_delta(
    tmp_path: Path,
) -> None:
    base = _base(tmp_path)
    fable = _write_dataset(
        tmp_path / "fable",
        count=2,
        source="teacher:claude:claude-fable-5",
    )

    with pytest.raises(ValueError, match="30-100"):
        build_mix(base, fable, tmp_path / "default")

    manifest = build_mix(
        base,
        fable,
        tmp_path / "exhausted",
        minimum_teacher_rows=1,
    )

    assert manifest["teacher_rows"] == 2
    assert manifest["minimum_teacher_rows"] == 1
    assert manifest["teacher_row_policy"] == "explicit_exhausted_collection_override"


@pytest.mark.parametrize("minimum", [0, 31, 101])
def test_v2p10_builder_rejects_invalid_teacher_minimum(
    tmp_path: Path,
    minimum: int,
) -> None:
    base = _base(tmp_path)
    fable = _write_dataset(
        tmp_path / f"fable-{minimum}",
        count=30,
        source="teacher:claude:claude-fable-5",
    )

    with pytest.raises(ValueError, match="minimum_teacher_rows"):
        build_mix(
            base,
            fable,
            tmp_path / f"mix-{minimum}",
            minimum_teacher_rows=minimum,
        )


def test_v2p10_builder_accepts_only_the_pinned_verified_gpt_fallback(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    base = _base(tmp_path)
    delta, rows = _write_gpt56sol_fixture(
        tmp_path / "gpt56sol_verified_v1",
        count=3,
    )
    monkeypatch.setattr(mix_module, "PRODUCTION_GPT56SOL_ROWS", 3)
    monkeypatch.setattr(
        mix_module,
        "PRODUCTION_GPT56SOL_MANIFEST_SHA256",
        hashlib.sha256((delta / "manifest.json").read_bytes()).hexdigest(),
    )
    monkeypatch.setattr(
        mix_module,
        "PRODUCTION_GPT56SOL_CANONICAL_SHA256",
        canonical_rows_sha256(rows),
    )

    manifest = build_mix(
        base,
        delta,
        tmp_path / "mix-gpt",
        minimum_teacher_rows=1,
    )

    assert manifest["teacher_rows"] == 3
    assert (
        manifest["teacher_delta_policy"]
        == "gpt56sol_verified_quota_fallback"
    )
    assert manifest["success_path_distilled"] is False
    assert manifest["functionally_verified_native_replay"] is True
    assert manifest["teacher"]["manifest"]["replay_manifest_sha256"] == "d" * 64


def test_v2p10_builder_quarantines_only_legacy_claude_fable_base_rows(
    tmp_path: Path,
) -> None:
    base = _base(tmp_path)
    manifest_path = base / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    rows = [dict(row) for row in load_from_disk(str(base))]
    rows[0]["source"] = "teacher:claude:claude-fable-5"
    shutil.rmtree(base)
    Dataset.from_list(rows).save_to_disk(str(base))
    train = base / "train.jsonl"
    train.write_text(
        "".join(
            json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n"
            for row in rows
        )
    )
    manifest["train_jsonl_sha256"] = hashlib.sha256(train.read_bytes()).hexdigest()
    manifest["canonical_rows_sha256"] = canonical_rows_sha256(rows)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    revision = _write_fable_revision_fixture(tmp_path / "revision", count=2)

    result = build_mix(
        base,
        revision,
        tmp_path / "mix-filtered",
        minimum_teacher_rows=1,
    )

    assert result["base_rows_input"] == 4
    assert result["base_rows_quarantined"] == 1
    assert result["base_rows"] == 3
    mixed = [
        json.loads(line)
        for line in (tmp_path / "mix-filtered" / "train.jsonl").read_text().splitlines()
        if line.strip()
    ]
    assert not any(
        row["source"] == "teacher:claude:claude-fable-5"
        for row in mixed
    )


def test_v2p10_builder_accepts_verified_fable_revision_plus_pinned_gpt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    base = _base(tmp_path)
    revision = _write_fable_revision_fixture(tmp_path / "revision", count=2)
    gpt, gpt_rows = _write_gpt56sol_fixture(
        tmp_path / "gpt56sol_verified_v1",
        count=3,
    )
    monkeypatch.setattr(mix_module, "PRODUCTION_GPT56SOL_ROWS", 3)
    monkeypatch.setattr(
        mix_module,
        "PRODUCTION_GPT56SOL_MANIFEST_SHA256",
        hashlib.sha256((gpt / "manifest.json").read_bytes()).hexdigest(),
    )
    monkeypatch.setattr(
        mix_module,
        "PRODUCTION_GPT56SOL_CANONICAL_SHA256",
        canonical_rows_sha256(gpt_rows),
    )

    result = build_mix(
        base,
        revision,
        tmp_path / "mix-combined",
        extra=(gpt,),
        minimum_teacher_rows=1,
    )

    assert result["teacher_rows"] == 5
    assert result["fable_revision_rows"] == 2
    assert result["gpt56sol_rows"] == 3
    assert (
        result["teacher_delta_policy"]
        == "verified_fable_revision_plus_gpt56sol"
    )


def test_v2p10_builder_direct_script_entrypoint_has_repo_imports() -> None:
    root = Path(__file__).resolve().parents[2]

    result = subprocess.run(
        [
            sys.executable,
            "phaseD_sft/build_mix_v2p10.py",
            "--help",
        ],
        cwd=root,
        text=True,
        capture_output=True,
    )

    assert result.returncode == 0, result.stderr
    assert "--teacher-delta" in result.stdout
    assert "--extra-teacher-delta" in result.stdout
