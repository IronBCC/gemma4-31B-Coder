from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

import phaseH_eval.empty_retry_composite as composite_module
from phaseH_eval.empty_retry_composite import (
    combine_composite_retry,
    combine_empty_retry,
    freeze_composite_retry,
    freeze_empty_retry,
    promote_complete_run_without_retry,
    subset_empty_retry_composite,
    verify_composite_retry,
    verify_empty_retry_composite,
    verify_subset_composite,
)


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def _write_run(
    root: Path,
    *,
    name: str,
    patches: dict[str, str],
    resolved: set[str],
    ids_path: Path,
    seed: int = 1,
) -> None:
    model = ids_path.parent / f"{name}_full"
    _write_json(
        model / "config.json",
        {"architectures": ["Gemma4ForConditionalGeneration"]},
    )
    shard = model / "model-00001-of-00001.safetensors"
    shard.write_bytes(b"synthetic weights")
    weight_map = {
        (
            f"vision.layer.{index}"
            if index < 356
            else f"text.layer.{index}"
        ): shard.name
        for index in range(1188)
    }
    _write_json(
        model / "model.safetensors.index.json",
        {"weight_map": weight_map},
    )
    config_sha = hashlib.sha256((model / "config.json").read_bytes()).hexdigest()
    index_sha = hashlib.sha256(
        (model / "model.safetensors.index.json").read_bytes()
    ).hexdigest()
    model_artifacts = [
        {
            "path": str(shard.resolve()),
            "sha256": hashlib.sha256(shard.read_bytes()).hexdigest(),
            "bytes": shard.stat().st_size,
        }
    ]
    _write_json(
        root / "run_manifest.json",
        {
            "schema_version": 1,
            "name": name,
            "port": 8013,
            "ids_path": str(ids_path),
            "ids_sha256": hashlib.sha256(ids_path.read_bytes()).hexdigest(),
            "instances": len(patches),
            "batch": 20,
            "workers": 16,
            "pull_workers": 5,
            "config": "swebench_edit_first_selfretry_s120.yaml",
            "temperature": 0.7,
            "seed": seed,
            "step_limit": 120,
            "environment_class": "docker_selfretry.DockerSelfRetryEnv",
        },
    )
    _write_json(
        root / "eval_manifest.json",
        {
            "schema_version": 1,
            "run_id": root.name,
            "served_name": name,
            "model_path": str(model.resolve()),
            "model_config_sha256": config_sha,
            "model_index_sha256": index_sha,
            "model_artifacts": model_artifacts,
            "ids_path": str(ids_path),
            "ids_sha256": hashlib.sha256(ids_path.read_bytes()).hexdigest(),
            "instances": len(patches),
            "config": "swebench_edit_first_selfretry_s120.yaml",
            "environment_class": "docker_selfretry.DockerSelfRetryEnv",
            "temperature": 0.7,
            "seed": seed,
            "step_limit": 120,
            "generation_workers": 16,
            "scorer_workers": 16,
            "batch": 20,
            "pull_workers": 5,
        },
    )
    _write_json(
        root / "preds_all.json",
        {
            instance_id: {
                "instance_id": instance_id,
                "model_name_or_path": name,
                "model_patch": patch,
            }
            for instance_id, patch in patches.items()
        },
    )
    for instance_id in patches:
        _write_json(
            root / "b00" / instance_id / f"{instance_id}.traj.json",
            {"instance_id": instance_id, "messages": []},
        )
    empty = sorted(
        instance_id for instance_id, patch in patches.items() if not patch.strip()
    )
    _write_json(
        root / "acceptance.json",
        {
            "schema_version": 1,
            "run_id": root.name,
            "name": name,
            "status": "complete",
            "expected": len(patches),
            "preds": len(patches),
            "traj_files": len(patches),
            "unique_trajectories": len(patches),
            "usable_outcomes": len(patches),
            "resolved": len(resolved),
            "resolved_ids": sorted(resolved),
            "pull_failed": 0,
            "docker_failed": 0,
            "empty_split": {
                "total": len(empty),
                "model": empty,
                "harness_forced": [],
                "docker_failed": [],
                "unknown_missing": [],
            },
            "problems": [],
        },
    )


def test_freeze_preregisters_every_usable_empty_and_binds_source(
    tmp_path: Path,
) -> None:
    source_ids = tmp_path / "source_ids.json"
    source_run = tmp_path / "source"
    retry_ids = tmp_path / "retry_ids.json"
    plan_path = tmp_path / "retry_plan.json"
    _write_json(source_ids, ["case-a", "case-b", "case-c"])
    _write_run(
        source_run,
        name="candidate",
        patches={"case-a": "diff --git a/x b/x", "case-b": "", "case-c": " \n"},
        resolved={"case-a"},
        ids_path=source_ids,
    )

    plan = freeze_empty_retry(
        source_ids_path=source_ids,
        source_run_root=source_run,
        retry_ids_path=retry_ids,
        plan_path=plan_path,
    )

    assert json.loads(retry_ids.read_text()) == ["case-b", "case-c"]
    assert plan["status"] == "complete"
    assert plan["source_empty"] == 2
    assert plan["retry_ids_sha256"]
    assert {
        Path(row["path"]).name for row in plan["source_artifacts"]
    } == {
        "source_ids.json",
        "acceptance.json",
        "eval_manifest.json",
        "run_manifest.json",
        "preds_all.json",
        "model-00001-of-00001.safetensors",
        "case-a.traj.json",
        "case-b.traj.json",
        "case-c.traj.json",
    }
    plan_path.unlink()
    assert freeze_empty_retry(
        source_ids_path=source_ids,
        source_run_root=source_run,
        retry_ids_path=retry_ids,
        plan_path=plan_path,
    ) == plan
    retry_ids.unlink()
    assert freeze_empty_retry(
        source_ids_path=source_ids,
        source_run_root=source_run,
        retry_ids_path=retry_ids,
        plan_path=plan_path,
    ) == plan


def test_retry_plan_rejects_a_preregistered_nonempty_source_outcome(
    tmp_path: Path,
) -> None:
    source_ids = tmp_path / "source_ids.json"
    source_run = tmp_path / "source"
    retry_ids = tmp_path / "retry_ids.json"
    plan_path = tmp_path / "retry_plan.json"
    _write_json(source_ids, ["case-a", "case-b"])
    _write_run(
        source_run,
        name="candidate",
        patches={
            "case-a": "diff --git a/a.py b/a.py\n",
            "case-b": "",
        },
        resolved={"case-a"},
        ids_path=source_ids,
    )
    freeze_empty_retry(
        source_ids_path=source_ids,
        source_run_root=source_run,
        retry_ids_path=retry_ids,
        plan_path=plan_path,
    )
    _write_json(retry_ids, ["case-a", "case-b"])
    plan = json.loads(plan_path.read_text())
    plan["source_empty"] = 2
    plan["empty_ids"] = ["case-a", "case-b"]
    plan["retry_ids_sha256"] = hashlib.sha256(
        retry_ids.read_bytes()
    ).hexdigest()
    _write_json(plan_path, plan)

    with pytest.raises(ValueError, match="actual source empty"):
        composite_module.verify_retry_plan_empty_only(plan_path)


def test_promote_complete_nonempty_run_without_retry(
    tmp_path: Path,
) -> None:
    source_ids = tmp_path / "source_ids.json"
    source_run = tmp_path / "source"
    output = tmp_path / "composite.json"
    predictions_output = tmp_path / "predictions.json"
    _write_json(source_ids, ["case-a", "case-b"])
    _write_run(
        source_run,
        name="candidate",
        patches={
            "case-a": "diff --git a/x b/x",
            "case-b": "diff --git a/y b/y",
        },
        resolved={"case-a"},
        ids_path=source_ids,
    )

    composite = promote_complete_run_without_retry(
        source_ids_path=source_ids,
        source_run_root=source_run,
        output_path=output,
        predictions_output_path=predictions_output,
    )

    assert composite["status"] == "complete"
    assert composite["expected"] == 2
    assert composite["resolved"] == 1
    assert composite["empty_split"]["total"] == 0
    assert composite["empty_resampling"]["attempted"] == 0
    assert composite["selected_attempts"] == {
        "source": 2,
        "empty_retry": 0,
    }
    assert composite["retry_ids"] == []
    assert json.loads(predictions_output.read_text()) == json.loads(
        (source_run / "preds_all.json").read_text()
    )


def test_freeze_accepts_legacy_index_bound_manifest_and_binds_model_shards(
    tmp_path: Path,
) -> None:
    source_ids = tmp_path / "source_ids.json"
    source_run = tmp_path / "source"
    retry_ids = tmp_path / "retry_ids.json"
    retry_plan = tmp_path / "retry_plan.json"
    _write_json(source_ids, ["case-a"])
    _write_run(
        source_run,
        name="candidate",
        patches={"case-a": ""},
        resolved=set(),
        ids_path=source_ids,
    )
    eval_manifest_path = source_run / "eval_manifest.json"
    eval_manifest = json.loads(eval_manifest_path.read_text())
    eval_manifest.pop("model_artifacts")
    _write_json(eval_manifest_path, eval_manifest)

    plan = freeze_empty_retry(
        source_ids_path=source_ids,
        source_run_root=source_run,
        retry_ids_path=retry_ids,
        plan_path=retry_plan,
    )

    bound_names = {Path(row["path"]).name for row in plan["source_artifacts"]}
    assert "model-00001-of-00001.safetensors" in bound_names


def test_combine_uses_retry_for_every_frozen_id_without_cherry_picking(
    tmp_path: Path,
) -> None:
    source_ids = tmp_path / "source_ids.json"
    source_run = tmp_path / "source"
    retry_ids = tmp_path / "retry_ids.json"
    retry_plan = tmp_path / "retry_plan.json"
    retry_run = tmp_path / "retry"
    output = tmp_path / "composite.json"
    predictions_output = tmp_path / "preds.json"
    _write_json(source_ids, ["case-a", "case-b", "case-c"])
    _write_run(
        source_run,
        name="candidate",
        patches={
            "case-a": "diff --git a/a.py b/a.py\n",
            "case-b": "",
            "case-c": "",
        },
        resolved={"case-a"},
        ids_path=source_ids,
    )
    source_predictions = json.loads(
        (source_run / "preds_all.json").read_text()
    )
    source_predictions["case-c"]["attempt_id"] = "source-empty"
    _write_json(source_run / "preds_all.json", source_predictions)
    freeze_empty_retry(
        source_ids_path=source_ids,
        source_run_root=source_run,
        retry_ids_path=retry_ids,
        plan_path=retry_plan,
    )
    _write_run(
        retry_run,
        name="candidate",
        patches={"case-b": "diff --git a/b.py b/b.py\n", "case-c": ""},
        resolved={"case-b"},
        ids_path=retry_ids,
    )
    retry_predictions = json.loads(
        (retry_run / "preds_all.json").read_text()
    )
    retry_predictions["case-c"]["attempt_id"] = "retry-empty"
    _write_json(retry_run / "preds_all.json", retry_predictions)

    composite = combine_empty_retry(
        source_ids_path=source_ids,
        source_run_root=source_run,
        retry_ids_path=retry_ids,
        retry_run_root=retry_run,
        plan_path=retry_plan,
        output_path=output,
        predictions_output_path=predictions_output,
    )

    assert composite["status"] == "complete"
    assert composite["expected"] == 3
    assert composite["resolved_ids"] == ["case-a", "case-b"]
    assert composite["empty_split"]["model"] == ["case-c"]
    assert composite["empty_resampling"] == {
        "attempted": 2,
        "became_nonempty": 1,
        "became_resolved": 1,
        "still_empty": 1,
        "nonempty_rate": 0.5,
        "resolved_rate": 0.5,
    }
    assert composite["selected_attempts"] == {
        "source": 1,
        "empty_retry": 2,
    }
    assert composite["behavior_health"] == {
        "repeat_loops": 0,
        "tool_format_errors": 0,
        "assistant_responses": 0,
        "format_error_rate": None,
    }
    merged_predictions = json.loads(predictions_output.read_text())
    assert list(merged_predictions) == ["case-a", "case-b", "case-c"]
    assert merged_predictions["case-a"]["model_patch"] == (
        "diff --git a/a.py b/a.py\n"
    )
    assert merged_predictions["case-b"]["model_patch"] == (
        "diff --git a/b.py b/b.py\n"
    )
    assert merged_predictions["case-c"]["model_patch"] == ""
    assert merged_predictions["case-c"]["attempt_id"] == "retry-empty"
    assert composite["predictions_artifact"] == {
        "path": str(predictions_output.resolve()),
        "sha256": hashlib.sha256(
            predictions_output.read_bytes()
        ).hexdigest(),
        "bytes": predictions_output.stat().st_size,
    }
    assert verify_empty_retry_composite(
        source_ids_path=source_ids,
        source_run_root=source_run,
        retry_ids_path=retry_ids,
        retry_run_root=retry_run,
        plan_path=retry_plan,
        output_path=output,
    ) == composite
    subset_ids = tmp_path / "subset_ids.json"
    subset_output = tmp_path / "subset_composite.json"
    _write_json(subset_ids, ["case-a", "case-b"])
    subset = subset_empty_retry_composite(
        source_ids_path=source_ids,
        source_run_root=source_run,
        retry_ids_path=retry_ids,
        retry_run_root=retry_run,
        plan_path=retry_plan,
        full_composite_path=output,
        subset_ids_path=subset_ids,
        output_path=subset_output,
    )
    assert subset["panel_derivation"] == "fixed150_from_full300"
    assert subset["expected"] == 2
    assert subset["resolved_ids"] == ["case-a", "case-b"]
    assert subset["empty_resampling"]["attempted"] == 1
    assert subset["empty_resampling"]["became_nonempty"] == 1
    assert subset["behavior_health"]["repeat_loops"] == 0
    assert verify_subset_composite(
        source_ids_path=source_ids,
        source_run_root=source_run,
        retry_ids_path=retry_ids,
        retry_run_root=retry_run,
        plan_path=retry_plan,
        full_composite_path=output,
        subset_ids_path=subset_ids,
        output_path=subset_output,
    ) == subset
    stale = json.loads(output.read_text())
    stale["name"] = "wrong-model"
    _write_json(output, stale)
    with pytest.raises(ValueError, match="existing composite differs"):
        verify_empty_retry_composite(
            source_ids_path=source_ids,
            source_run_root=source_run,
            retry_ids_path=retry_ids,
            retry_run_root=retry_run,
            plan_path=retry_plan,
            output_path=output,
        )


def test_composite_retry_supersedes_only_current_empties_and_accumulates_attempts(
    tmp_path: Path,
) -> None:
    source_ids = tmp_path / "source_ids.json"
    source_run = tmp_path / "source"
    retry1_ids = tmp_path / "retry1_ids.json"
    retry1_plan = tmp_path / "retry1_plan.json"
    retry1_run = tmp_path / "retry1"
    generation1 = tmp_path / "generation1.json"
    generation1_predictions = tmp_path / "generation1_preds.json"
    retry2_ids = tmp_path / "retry2_ids.json"
    retry2_plan = tmp_path / "retry2_plan.json"
    retry2_run = tmp_path / "retry2"
    generation2 = tmp_path / "generation2.json"
    generation2_predictions = tmp_path / "generation2_preds.json"
    _write_json(source_ids, ["case-a", "case-b", "case-c"])
    _write_run(
        source_run,
        name="candidate",
        patches={
            "case-a": "diff --git a/a.py b/a.py\n",
            "case-b": "",
            "case-c": "",
        },
        resolved={"case-a"},
        ids_path=source_ids,
    )
    freeze_empty_retry(
        source_ids_path=source_ids,
        source_run_root=source_run,
        retry_ids_path=retry1_ids,
        plan_path=retry1_plan,
    )
    _write_run(
        retry1_run,
        name="candidate",
        patches={
            "case-b": "diff --git a/b.py b/b.py\n",
            "case-c": "",
        },
        resolved={"case-b"},
        ids_path=retry1_ids,
    )
    retry1_predictions = json.loads(
        (retry1_run / "preds_all.json").read_text()
    )
    retry1_predictions["case-c"]["attempt_id"] = "generation-1"
    _write_json(retry1_run / "preds_all.json", retry1_predictions)
    combine_empty_retry(
        source_ids_path=source_ids,
        source_run_root=source_run,
        retry_ids_path=retry1_ids,
        retry_run_root=retry1_run,
        plan_path=retry1_plan,
        output_path=generation1,
        predictions_output_path=generation1_predictions,
    )

    plan = freeze_composite_retry(
        source_ids_path=source_ids,
        source_composite_path=generation1,
        source_predictions_path=generation1_predictions,
        retry_ids_path=retry2_ids,
        plan_path=retry2_plan,
    )

    assert json.loads(retry2_ids.read_text()) == ["case-c"]
    assert plan["retry_generation"] == 2
    assert plan["source_empty"] == 1
    assert {
        Path(row["path"]).name for row in plan["source_artifacts"]
    } == {
        "source_ids.json",
        "generation1.json",
        "generation1_preds.json",
    }

    _write_run(
        retry2_run,
        name="candidate",
        patches={"case-c": "diff --git a/c.py b/c.py\n"},
        resolved={"case-c"},
        ids_path=retry2_ids,
        seed=2,
    )
    retry2_predictions = json.loads(
        (retry2_run / "preds_all.json").read_text()
    )
    retry2_predictions["case-c"]["attempt_id"] = "generation-2"
    _write_json(retry2_run / "preds_all.json", retry2_predictions)

    composite = combine_composite_retry(
        source_ids_path=source_ids,
        source_composite_path=generation1,
        source_predictions_path=generation1_predictions,
        retry_ids_path=retry2_ids,
        retry_run_root=retry2_run,
        plan_path=retry2_plan,
        output_path=generation2,
        predictions_output_path=generation2_predictions,
    )

    assert composite["retry_generation"] == 2
    assert composite["retry_harness_contract"]["seed"] == 2
    assert composite["resolved_ids"] == ["case-a", "case-b", "case-c"]
    assert composite["empty_split"]["total"] == 0
    assert composite["empty_resampling"] == {
        "attempted": 3,
        "became_nonempty": 2,
        "became_resolved": 2,
        "still_empty": 0,
        "nonempty_rate": 2 / 3,
        "resolved_rate": 2 / 3,
    }
    assert composite["selected_attempts"] == {
        "source": 1,
        "empty_retry": 1,
        "empty_retry_2": 1,
    }
    merged_predictions = json.loads(generation2_predictions.read_text())
    assert merged_predictions["case-c"]["attempt_id"] == "generation-2"
    assert verify_composite_retry(
        source_ids_path=source_ids,
        source_composite_path=generation1,
        source_predictions_path=generation1_predictions,
        retry_ids_path=retry2_ids,
        retry_run_root=retry2_run,
        plan_path=retry2_plan,
        output_path=generation2,
    ) == composite


def test_composite_retry_rejects_parent_predictions_changed_after_freeze(
    tmp_path: Path,
) -> None:
    source_ids = tmp_path / "source_ids.json"
    source_run = tmp_path / "source"
    retry1_ids = tmp_path / "retry1_ids.json"
    retry1_plan = tmp_path / "retry1_plan.json"
    retry1_run = tmp_path / "retry1"
    generation1 = tmp_path / "generation1.json"
    generation1_predictions = tmp_path / "generation1_preds.json"
    retry2_ids = tmp_path / "retry2_ids.json"
    retry2_plan = tmp_path / "retry2_plan.json"
    _write_json(source_ids, ["case-a"])
    _write_run(
        source_run,
        name="candidate",
        patches={"case-a": ""},
        resolved=set(),
        ids_path=source_ids,
    )
    freeze_empty_retry(
        source_ids_path=source_ids,
        source_run_root=source_run,
        retry_ids_path=retry1_ids,
        plan_path=retry1_plan,
    )
    _write_run(
        retry1_run,
        name="candidate",
        patches={"case-a": ""},
        resolved=set(),
        ids_path=retry1_ids,
    )
    combine_empty_retry(
        source_ids_path=source_ids,
        source_run_root=source_run,
        retry_ids_path=retry1_ids,
        retry_run_root=retry1_run,
        plan_path=retry1_plan,
        output_path=generation1,
        predictions_output_path=generation1_predictions,
    )
    freeze_composite_retry(
        source_ids_path=source_ids,
        source_composite_path=generation1,
        source_predictions_path=generation1_predictions,
        retry_ids_path=retry2_ids,
        plan_path=retry2_plan,
    )
    predictions = json.loads(generation1_predictions.read_text())
    predictions["case-a"]["attempt_id"] = "mutated-after-freeze"
    _write_json(generation1_predictions, predictions)

    with pytest.raises(ValueError, match="source artifact changed after freeze"):
        combine_composite_retry(
            source_ids_path=source_ids,
            source_composite_path=generation1,
            source_predictions_path=generation1_predictions,
            retry_ids_path=retry2_ids,
            retry_run_root=tmp_path / "unused-retry",
            plan_path=retry2_plan,
            output_path=tmp_path / "generation2.json",
        )


def test_combine_rejects_partial_retry_and_source_mutation(tmp_path: Path) -> None:
    source_ids = tmp_path / "source_ids.json"
    source_run = tmp_path / "source"
    retry_ids = tmp_path / "retry_ids.json"
    retry_plan = tmp_path / "retry_plan.json"
    retry_run = tmp_path / "retry"
    _write_json(source_ids, ["case-a", "case-b"])
    _write_run(
        source_run,
        name="candidate",
        patches={"case-a": "", "case-b": ""},
        resolved=set(),
        ids_path=source_ids,
    )
    freeze_empty_retry(
        source_ids_path=source_ids,
        source_run_root=source_run,
        retry_ids_path=retry_ids,
        plan_path=retry_plan,
    )
    _write_run(
        retry_run,
        name="candidate",
        patches={"case-a": "patch-a"},
        resolved={"case-a"},
        ids_path=retry_ids,
    )

    with pytest.raises(ValueError, match="retry run is incomplete"):
        combine_empty_retry(
            source_ids_path=source_ids,
            source_run_root=source_run,
            retry_ids_path=retry_ids,
            retry_run_root=retry_run,
            plan_path=retry_plan,
            output_path=tmp_path / "partial.json",
        )

    predictions = json.loads((source_run / "preds_all.json").read_text())
    predictions["case-a"]["model_patch"] = "mutated"
    _write_json(source_run / "preds_all.json", predictions)
    with pytest.raises(ValueError, match="source artifact changed after freeze"):
        combine_empty_retry(
            source_ids_path=source_ids,
            source_run_root=source_run,
            retry_ids_path=retry_ids,
            retry_run_root=retry_run,
            plan_path=retry_plan,
            output_path=tmp_path / "mutated.json",
        )


def test_combine_requires_a_distinct_retry_with_the_same_harness(
    tmp_path: Path,
) -> None:
    source_ids = tmp_path / "source_ids.json"
    source_run = tmp_path / "source"
    retry_ids = tmp_path / "retry_ids.json"
    retry_plan = tmp_path / "retry_plan.json"
    retry_run = tmp_path / "retry"
    _write_json(source_ids, ["case-a"])
    _write_run(
        source_run,
        name="candidate",
        patches={"case-a": ""},
        resolved=set(),
        ids_path=source_ids,
    )
    freeze_empty_retry(
        source_ids_path=source_ids,
        source_run_root=source_run,
        retry_ids_path=retry_ids,
        plan_path=retry_plan,
    )

    with pytest.raises(ValueError, match="distinct"):
        combine_empty_retry(
            source_ids_path=source_ids,
            source_run_root=source_run,
            retry_ids_path=retry_ids,
            retry_run_root=source_run,
            plan_path=retry_plan,
            output_path=tmp_path / "same-run.json",
        )

    _write_run(
        retry_run,
        name="candidate",
        patches={"case-a": "patch-a"},
        resolved={"case-a"},
        ids_path=retry_ids,
        seed=2,
    )
    with pytest.raises(ValueError, match="harness contract"):
        combine_empty_retry(
            source_ids_path=source_ids,
            source_run_root=source_run,
            retry_ids_path=retry_ids,
            retry_run_root=retry_run,
            plan_path=retry_plan,
            output_path=tmp_path / "wrong-harness.json",
        )


def test_run_identity_and_infrastructure_empty_categories_fail_closed(
    tmp_path: Path,
) -> None:
    source_ids = tmp_path / "source_ids.json"
    source_run = tmp_path / "source"
    retry_ids = tmp_path / "retry_ids.json"
    retry_plan = tmp_path / "retry_plan.json"
    retry_run = tmp_path / "retry"
    _write_json(source_ids, ["case-a"])
    _write_run(
        source_run,
        name="candidate",
        patches={"case-a": ""},
        resolved=set(),
        ids_path=source_ids,
    )
    predictions = json.loads((source_run / "preds_all.json").read_text())
    predictions["case-a"]["model_name_or_path"] = "different-model"
    _write_json(source_run / "preds_all.json", predictions)
    with pytest.raises(ValueError, match="prediction model identity"):
        freeze_empty_retry(
            source_ids_path=source_ids,
            source_run_root=source_run,
            retry_ids_path=retry_ids,
            plan_path=retry_plan,
        )

    predictions["case-a"]["model_name_or_path"] = "candidate"
    _write_json(source_run / "preds_all.json", predictions)
    freeze_empty_retry(
        source_ids_path=source_ids,
        source_run_root=source_run,
        retry_ids_path=retry_ids,
        plan_path=retry_plan,
    )
    _write_run(
        retry_run,
        name="candidate",
        patches={"case-a": ""},
        resolved=set(),
        ids_path=retry_ids,
    )
    acceptance = json.loads((retry_run / "acceptance.json").read_text())
    acceptance["empty_split"]["model"] = []
    acceptance["empty_split"]["docker_failed"] = ["case-a"]
    _write_json(retry_run / "acceptance.json", acceptance)
    with pytest.raises(ValueError, match="infrastructure empty"):
        combine_empty_retry(
            source_ids_path=source_ids,
            source_run_root=source_run,
            retry_ids_path=retry_ids,
            retry_run_root=retry_run,
            plan_path=retry_plan,
            output_path=tmp_path / "infra.json",
        )


def test_freeze_rejects_a_source_with_no_empty_patch(tmp_path: Path) -> None:
    source_ids = tmp_path / "source_ids.json"
    source_run = tmp_path / "source"
    retry_ids = tmp_path / "retry_ids.json"
    retry_plan = tmp_path / "retry_plan.json"
    _write_json(source_ids, ["case-a"])
    _write_run(
        source_run,
        name="candidate",
        patches={"case-a": "patch-a"},
        resolved={"case-a"},
        ids_path=source_ids,
    )

    with pytest.raises(ValueError, match="retry not required"):
        freeze_empty_retry(
            source_ids_path=source_ids,
            source_run_root=source_run,
            retry_ids_path=retry_ids,
            plan_path=retry_plan,
        )
    assert not retry_ids.exists()
    assert not retry_plan.exists()


def test_combine_rejects_a_model_shard_changed_after_freeze(
    tmp_path: Path,
) -> None:
    source_ids = tmp_path / "source_ids.json"
    source_run = tmp_path / "source"
    retry_ids = tmp_path / "retry_ids.json"
    retry_plan = tmp_path / "retry_plan.json"
    retry_run = tmp_path / "retry"
    _write_json(source_ids, ["case-a"])
    _write_run(
        source_run,
        name="candidate",
        patches={"case-a": ""},
        resolved=set(),
        ids_path=source_ids,
    )
    freeze_empty_retry(
        source_ids_path=source_ids,
        source_run_root=source_run,
        retry_ids_path=retry_ids,
        plan_path=retry_plan,
    )
    _write_run(
        retry_run,
        name="candidate",
        patches={"case-a": "patch-a"},
        resolved={"case-a"},
        ids_path=retry_ids,
    )
    (tmp_path / "candidate_full" / "model-00001-of-00001.safetensors").write_bytes(
        b"different weights"
    )

    with pytest.raises(ValueError, match="source artifact changed after freeze"):
        combine_empty_retry(
            source_ids_path=source_ids,
            source_run_root=source_run,
            retry_ids_path=retry_ids,
            retry_run_root=retry_run,
            plan_path=retry_plan,
            output_path=tmp_path / "composite.json",
        )
