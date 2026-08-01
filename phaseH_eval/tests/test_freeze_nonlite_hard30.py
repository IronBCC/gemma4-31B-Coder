from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from phaseH_eval.freeze_nonlite_hard30 import freeze_nonlite_hard30


def _row(index: int) -> dict:
    mutation = (
        "diff --git a/src/x.py b/src/x.py\n"
        "--- a/src/x.py\n"
        "+++ b/src/x.py\n"
        "@@ -1 +1 @@\n"
        f"-before_{index}\n"
        f"+after_{index}\n"
    )
    return {
        "instance_id": f"fixture__repo.case_{index:02d}",
        "image_name": f"fixture/image-{index:02d}:latest",
        "problem_statement": f"Fix case {index}.",
        "patch": mutation,
        "FAIL_TO_PASS": ["tests/test_x.py::test_fix"],
        "PASS_TO_PASS": ["tests/test_x.py::test_old"],
        "repo": f"fixture/repo-{index % 5}",
        "source": "SWE-bench/SWE-smith",
    }


def _jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(
            json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n"
            for row in rows
        )
    )


def _preflight(
    rows: list[dict],
    *,
    invalid_ids: set[str] = frozenset(),
) -> dict[str, dict]:
    results = {}
    for row in rows:
        task = dict(row)
        mutation_sha256 = hashlib.sha256(task["patch"].encode()).hexdigest()
        task["task_patch_role"] = "bug_inducing_mutation"
        task["mutation_patch_sha256"] = mutation_sha256
        valid = row["instance_id"] not in invalid_ids
        results[row["instance_id"]] = {
            "schema_version": 2,
            "instance_id": row["instance_id"],
            "task_sha256": hashlib.sha256(
                json.dumps(
                    task,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                    allow_nan=False,
                ).encode()
            ).hexdigest(),
            "patch_sha256": hashlib.sha256(b"").hexdigest(),
            "mutation_patch_sha256": mutation_sha256,
            "image_id": "sha256:" + f"{rows.index(row):064x}",
            "status": "empty_patch" if valid else "invalid_task",
            "reference_f2p_pass": valid,
            "mutation_f2p_pass": False,
            "mutation_patch_applied": valid,
            "task_baseline_valid": valid,
            "infra_error": None if valid else "clean reference F2P control failed",
        }
    return results


def test_freeze_nonlite_hard30_is_bound_decontaminated_and_atomic(
    tmp_path: Path,
) -> None:
    source_rows = [_row(index) for index in range(40)]
    source = tmp_path / "pool.jsonl"
    _jsonl(source, source_rows)
    source_manifest = tmp_path / "pool_manifest.json"
    source_manifest.write_text('{"selection_frozen_before_eval":true}\n')
    exclusion = tmp_path / "teacher.jsonl"
    _jsonl(exclusion, [{"instance_id": source_rows[0]["instance_id"]}])
    output = tmp_path / "hard30"
    invalid_ids = {
        source_rows[2]["instance_id"],
        source_rows[3]["instance_id"],
        source_rows[4]["instance_id"],
    }

    manifest = freeze_nonlite_hard30(
        source=source,
        source_manifest=source_manifest,
        base_instance_ids={source_rows[1]["instance_id"]},
        exclusion_paths=(exclusion,),
        local_image_ids={
            row["image_name"]: "sha256:" + f"{index:064x}"
            for index, row in enumerate(source_rows)
        },
        baseline_preflight=_preflight(source_rows, invalid_ids=invalid_ids),
        output=output,
        expected_source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        expected_source_manifest_sha256=hashlib.sha256(
            source_manifest.read_bytes()
        ).hexdigest(),
    )

    tasks = [
        json.loads(line)
        for line in (output / "tasks.jsonl").read_text().splitlines()
    ]
    ids = json.loads((output / "ids.json").read_text())
    assert len(tasks) == len(ids) == 30
    assert ids == [row["instance_id"] for row in tasks]
    assert source_rows[0]["instance_id"] not in ids
    assert source_rows[1]["instance_id"] not in ids
    assert not (invalid_ids & set(ids))
    assert ids == [row["instance_id"] for row in source_rows[5:35]]
    assert manifest["training_eligible"] is False
    assert manifest["selection_frozen_before_eval"] is True
    assert manifest["tasks_sha256"] == hashlib.sha256(
        (output / "tasks.jsonl").read_bytes()
    ).hexdigest()
    assert manifest["ids_sha256"] == hashlib.sha256(
        (output / "ids.json").read_bytes()
    ).hexdigest()
    assert len(manifest["task_bindings"]) == 30
    assert manifest["schema_version"] == 3
    assert manifest["baseline_preflight"]["eligible_rows"] == 38
    assert manifest["baseline_preflight"]["valid_rows"] == 35
    assert manifest["baseline_preflight"]["invalid_rows"] == 3
    assert manifest["baseline_preflight"]["invalid_ids"] == sorted(invalid_ids)
    assert len(list((output / "baseline_preflight" / "results").glob("*.json"))) == 38
    for task, binding in zip(tasks, manifest["task_bindings"], strict=True):
        mutation_sha256 = hashlib.sha256(task["patch"].encode()).hexdigest()
        assert task["task_patch_role"] == "bug_inducing_mutation"
        assert task["mutation_patch_sha256"] == mutation_sha256
        assert binding["task_patch_role"] == "bug_inducing_mutation"
        assert binding["mutation_patch_sha256"] == mutation_sha256
    assert "task_patch_role" not in source_rows[2]
    assert not list(tmp_path.glob(".hard30.*"))

    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        freeze_nonlite_hard30(
            source=source,
            source_manifest=source_manifest,
            base_instance_ids=set(),
            exclusion_paths=(),
            local_image_ids={
                row["image_name"]: "sha256:" + "a" * 64
                for row in source_rows
            },
            baseline_preflight=_preflight(source_rows),
            output=output,
            expected_source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
            expected_source_manifest_sha256=hashlib.sha256(
                source_manifest.read_bytes()
            ).hexdigest(),
        )


def test_freeze_nonlite_hard30_rejects_missing_local_image(
    tmp_path: Path,
) -> None:
    rows = [_row(index) for index in range(40)]
    source = tmp_path / "pool.jsonl"
    _jsonl(source, rows)
    source_manifest = tmp_path / "pool_manifest.json"
    source_manifest.write_text("{}\n")

    with pytest.raises(ValueError, match="local images"):
        freeze_nonlite_hard30(
            source=source,
            source_manifest=source_manifest,
            base_instance_ids=set(),
            exclusion_paths=(),
            local_image_ids={
                row["image_name"]: "sha256:" + "a" * 64
                for row in rows[1:]
            },
            baseline_preflight=_preflight(rows),
            output=tmp_path / "hard30",
            expected_source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
            expected_source_manifest_sha256=hashlib.sha256(
                source_manifest.read_bytes()
            ).hexdigest(),
        )

    assert not (tmp_path / "hard30").exists()


@pytest.mark.parametrize("bad_patch", [None, "", "not a git patch\n"])
def test_freeze_nonlite_hard30_rejects_missing_or_malformed_mutation(
    tmp_path: Path,
    bad_patch: str | None,
) -> None:
    rows = [_row(index) for index in range(40)]
    if bad_patch is None:
        rows[0].pop("patch")
    else:
        rows[0]["patch"] = bad_patch
    source = tmp_path / "pool.jsonl"
    _jsonl(source, rows)
    source_manifest = tmp_path / "pool_manifest.json"
    source_manifest.write_text("{}\n")

    with pytest.raises(ValueError, match="mutation patch"):
        freeze_nonlite_hard30(
            source=source,
            source_manifest=source_manifest,
            base_instance_ids=set(),
            exclusion_paths=(),
            local_image_ids={
                row["image_name"]: "sha256:" + "a" * 64
                for row in rows
            },
            baseline_preflight={},
            output=tmp_path / "hard30",
            expected_source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
            expected_source_manifest_sha256=hashlib.sha256(
                source_manifest.read_bytes()
            ).hexdigest(),
        )

    assert not (tmp_path / "hard30").exists()
