from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

import phaseH_eval.v2p11_portability_gate as portability_module
from phaseH_eval.v2p11_portability_gate import evaluate_portability


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n")


def _write_run(
    root: Path,
    *,
    ids: list[str],
    name: str,
    empty: set[str],
    resolved: set[str],
    harness: dict[str, object],
) -> None:
    model = root.parent / f"{name}_model"
    _write_json(model / "config.json", {"architectures": ["Synthetic"]})
    shard = model / "model.safetensors"
    shard.write_bytes(name.encode())
    artifacts = [{
        "path": str(shard.resolve()),
        "sha256": hashlib.sha256(shard.read_bytes()).hexdigest(),
        "bytes": shard.stat().st_size,
    }]
    manifest_path = root / "eval_manifest.json"
    _write_json(
        manifest_path,
        {
            "schema_version": 1,
            "run_id": root.name,
            "served_name": name,
            "model_path": str(model.resolve()),
            "model_config_sha256": hashlib.sha256(
                (model / "config.json").read_bytes()
            ).hexdigest(),
            "model_safetensors_sha256": artifacts[0]["sha256"],
            "model_artifacts": artifacts,
            "ids_sha256": harness["ids_sha256"],
            "harness_contract": harness,
        },
    )
    _write_json(
        root / "tool_canary.json",
        {
            "schema_version": 1,
            "model": name,
            "tool_name": "bash",
            "command": "pwd",
            "finish_reason": "tool_calls",
            "eval_manifest_sha256": hashlib.sha256(
                manifest_path.read_bytes()
            ).hexdigest(),
        },
    )
    predictions_path = root / "preds.json"
    _write_json(
        predictions_path,
        {
            instance_id: {
                "instance_id": instance_id,
                "model_name_or_path": f"openai/{name}",
                "model_patch": (
                    ""
                    if instance_id in empty
                    else f"diff --git a/{instance_id} b/{instance_id}\n"
                ),
            }
            for instance_id in ids
        },
    )
    for instance_id in ids:
        _write_json(
            root / instance_id / f"{instance_id}.traj.json",
            {
                "instance_id": instance_id,
                "messages": [],
                "info": {
                    "submission": (
                        ""
                        if instance_id in empty
                        else f"diff --git a/{instance_id} b/{instance_id}\n"
                    )
                },
            },
        )
    report_path = root / "official_report.json"
    _write_json(
        report_path,
        {
            "schema_version": 2,
            "total_instances": 500,
            "submitted_instances": len(ids),
            "submitted_ids": ids,
            "resolved_instances": len(resolved),
            "resolved_ids": sorted(resolved),
            "empty_patch_instances": len(empty),
            "empty_patch_ids": sorted(empty),
            "error_instances": 0,
            "error_ids": [],
        },
    )
    _write_json(
        root / "official_report_binding.json",
        {
            "schema_version": 1,
            "eval_manifest_sha256": hashlib.sha256(
                manifest_path.read_bytes()
            ).hexdigest(),
            "predictions_sha256": hashlib.sha256(
                predictions_path.read_bytes()
            ).hexdigest(),
            "official_report_sha256": hashlib.sha256(
                report_path.read_bytes()
            ).hexdigest(),
        },
    )
    (root / "exit_statuses_test.yaml").write_text(
        "instances_by_exit_status:\n"
        "    Submitted:\n"
        + "".join(f"    - {instance_id}\n" for instance_id in ids)
    )


def _fixture(
    tmp_path: Path,
    *,
    candidate_name: str = "teacher_sft_v2p11",
) -> tuple[Path, Path, Path, Path, Path]:
    ids = [f"repo{index}__project-{index}" for index in range(10)]
    ids_path = tmp_path / "ids.json"
    verified_path = tmp_path / "verified.json"
    lite_path = tmp_path / "lite.json"
    _write_json(ids_path, ids)
    _write_json(
        verified_path,
        {"schema_version": 1, "instance_ids": [*ids, "other__case-1"]},
    )
    _write_json(lite_path, ["lite__case-1"])
    harness = {
        "mini_swe_agent_version": "2.4.1",
        "stock_config_sha256": "a" * 64,
        "environment_class": "docker",
        "model_class": (
            "minisweagent.models.litellm_model.LitellmModel"
        ),
        "temperature": 0.7,
        "seed": 1,
        "max_tokens": 4096,
        "workers": 4,
        "step_limit": 250,
        "subset": "verified",
        "ids_sha256": hashlib.sha256(ids_path.read_bytes()).hexdigest(),
    }
    control = tmp_path / "control"
    candidate = tmp_path / "candidate"
    _write_run(
        control,
        ids=ids,
        name="teacher_sft_v2p10",
        empty={ids[0], ids[1]},
        resolved={ids[2], ids[3], ids[4]},
        harness=harness,
    )
    _write_run(
        candidate,
        ids=ids,
        name=candidate_name,
        empty=set(),
        resolved={ids[2], ids[3], ids[4], ids[5]},
        harness=harness,
    )
    return ids_path, verified_path, lite_path, control, candidate


def test_portability_gate_passes_controller_free_empty_improvement(
    tmp_path: Path,
) -> None:
    ids, verified, lite, control, candidate = _fixture(tmp_path)

    report = evaluate_portability(
        ids_path=ids,
        verified_exclusions_path=verified,
        lite_ids_path=lite,
        control_root=control,
        candidate_root=candidate,
    )

    assert report["passed"] is True
    assert report["control"]["empty"] == 2
    assert report["candidate"]["empty"] == 0
    assert report["candidate"]["resolved"] == 4
    assert report["criteria"]["candidate_empty_at_most_one"] is True
    assert report["criteria"]["candidate_no_format_regression"] is True


def test_portability_gate_accepts_explicit_candidate_identity(
    tmp_path: Path,
) -> None:
    candidate_name = "teacher_sft_v2p11_v2p10init_fable"
    ids, verified, lite, control, candidate = _fixture(
        tmp_path,
        candidate_name=candidate_name,
    )

    report = evaluate_portability(
        ids_path=ids,
        verified_exclusions_path=verified,
        lite_ids_path=lite,
        control_root=control,
        candidate_root=candidate,
        candidate_name=candidate_name,
    )

    assert report["candidate_name"] == candidate_name
    assert report["candidate"]["name"] == candidate_name


def test_portability_cli_forwards_explicit_candidate_identity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    def fake_evaluate(**kwargs: object) -> dict[str, object]:
        captured.update(kwargs)
        return {
            "passed": True,
            "control": {"resolved": 3, "empty": 1},
            "candidate": {"resolved": 4, "empty": 0},
        }

    monkeypatch.setattr(
        portability_module,
        "evaluate_portability",
        fake_evaluate,
    )
    monkeypatch.setattr(
        portability_module,
        "_publish_json_noreplace",
        lambda *_args, **_kwargs: None,
    )
    candidate_name = "teacher_sft_v2p11_v2p10init_fable"

    status = portability_module.main([
        "--ids",
        str(tmp_path / "ids.json"),
        "--verified-exclusions",
        str(tmp_path / "verified.json"),
        "--lite-ids",
        str(tmp_path / "lite.json"),
        "--control-root",
        str(tmp_path / "control"),
        "--candidate-root",
        str(tmp_path / "candidate"),
        "--candidate-name",
        candidate_name,
        "--out",
        str(tmp_path / "gate.json"),
    ])

    assert status == 0
    assert captured["candidate_name"] == candidate_name


def test_portability_gate_rejects_harness_drift(
    tmp_path: Path,
) -> None:
    ids, verified, lite, control, candidate = _fixture(tmp_path)
    manifest_path = candidate / "eval_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["harness_contract"]["max_tokens"] = 8192
    _write_json(manifest_path, manifest)

    with pytest.raises(
        ValueError,
        match="harness|stock controller-free",
    ):
        evaluate_portability(
            ids_path=ids,
            verified_exclusions_path=verified,
            lite_ids_path=lite,
            control_root=control,
            candidate_root=candidate,
        )


def test_portability_gate_rejects_report_from_other_predictions(
    tmp_path: Path,
) -> None:
    ids, verified, lite, control, candidate = _fixture(tmp_path)
    predictions_path = candidate / "preds.json"
    predictions = json.loads(predictions_path.read_text())
    predictions[next(iter(predictions))]["model_patch"] += "# changed\n"
    _write_json(predictions_path, predictions)

    with pytest.raises(ValueError, match="official report binding"):
        evaluate_portability(
            ids_path=ids,
            verified_exclusions_path=verified,
            lite_ids_path=lite,
            control_root=control,
            candidate_root=candidate,
        )


def test_portability_gate_rejects_canary_from_other_manifest(
    tmp_path: Path,
) -> None:
    ids, verified, lite, control, candidate = _fixture(tmp_path)
    canary_path = candidate / "tool_canary.json"
    canary = json.loads(canary_path.read_text())
    canary["eval_manifest_sha256"] = "0" * 64
    _write_json(canary_path, canary)

    with pytest.raises(ValueError, match="tool-call canary"):
        evaluate_portability(
            ids_path=ids,
            verified_exclusions_path=verified,
            lite_ids_path=lite,
            control_root=control,
            candidate_root=candidate,
        )
