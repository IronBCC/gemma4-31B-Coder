from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

import phaseD_sft.build_v2p11_fable_extension as extension_module
from teacher_platform.generic_trace_replay import (
    assess_controls,
    build_task_contract,
    canonical_json_bytes,
    sha256_bytes,
)


def _json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True) + "\n")


def _jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows)
    )


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _exact_run(command: str, passed: bool) -> dict[str, object]:
    output = "1 passed in 0.01s\n" if passed else "1 failed in 0.01s\n"
    return {
        "command": command,
        "returncode": 0 if passed else 1,
        "strict_pass": passed,
        "output_sha256": sha256_bytes(output.encode()),
        "output_tail": output,
    }


def _exact_phase(
    name: str,
    contract: dict[str, object],
    candidate_patch_sha256: str,
    *,
    f2p: bool,
    p2p: bool,
) -> dict[str, object]:
    protected = "f" * 64
    phase: dict[str, object] = {
        "phase": name,
        "patch_applied": True,
        "f2p_pass": f2p,
        "p2p_pass": p2p,
        "f2p_runs": [
            _exact_run(command, f2p)
            for command in contract["f2p_commands"]
        ],
        "p2p_runs": [
            _exact_run(command, p2p)
            for command in contract["p2p_commands"]
        ],
        "protected_before": protected,
        "protected_after": protected,
        "protected_stable": True,
    }
    if contract["schema_version"] == 3:
        roles = {
            "baseline": [
                ("mutation", contract["mutation_patch_sha256"]),
            ],
            "reference": [],
            "candidate_1": [
                ("mutation", contract["mutation_patch_sha256"]),
                ("candidate", candidate_patch_sha256),
            ],
            "candidate_2": [
                ("mutation", contract["mutation_patch_sha256"]),
                ("candidate", candidate_patch_sha256),
            ],
        }[name]
        phase["patch_applications"] = [
            {
                "role": role,
                "patch_sha256": patch_sha256,
                "returncode": 0,
                "output_sha256": "0" * 64,
                "output_tail": "",
            }
            for role, patch_sha256 in roles
        ]
    else:
        phase["patch_apply"] = (
            None
            if name == "baseline"
            else {
                "returncode": 0,
                "output_sha256": "0" * 64,
                "output_tail": "",
            }
        )
    return phase


def _fixture(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[Path, Path, Path, Path, Path, Path]:
    base = tmp_path / "base"
    base_train = base / "train.jsonl"
    base_row = {
        "instance_id": "base__case-1",
        "messages": [
            {"role": "system", "content": "system", "loss": False},
            {"role": "assistant", "content": "", "loss": True,
             "tool_calls": [{
                 "type": "function",
                 "function": {
                     "name": "bash",
                     "arguments": json.dumps({"command": "pwd"}),
                 },
             }]},
        ],
        "repo": "base/repo",
        "source": "v2p10",
    }
    _jsonl(base_train, [base_row])
    _json(
        base / "manifest.json",
        {
            "schema_version": 2,
            "dataset_variant": "teacher_train_mix_v2p11",
            "rendered": 1,
            "training_admitted": 1,
            "all_training_gates_complete": True,
            "train_jsonl_sha256": _sha(base_train),
        },
    )
    monkeypatch.setattr(extension_module, "PRODUCTION_BASE_ROWS", 1)
    monkeypatch.setattr(
        extension_module,
        "PRODUCTION_BASE_MANIFEST_SHA256",
        _sha(base / "manifest.json"),
    )
    monkeypatch.setattr(
        extension_module,
        "PRODUCTION_BASE_TRAIN_SHA256",
        _sha(base_train),
    )

    source_id = "teacher__case-2"
    run = tmp_path / "run"
    run.mkdir()
    stream = run / f"{source_id}.stream.jsonl"
    patch = run / f"{source_id}.patch"
    stream.write_text('{"event":"assistant"}\n')
    patch.write_text("diff --git a/a.py b/a.py\n")
    candidate_patch_sha256 = _sha(patch)
    contract = build_task_contract({
        "instance_id": source_id,
        "image_name": "fixture/image:latest",
        "patch": "reference patch",
        "FAIL_TO_PASS": ["tests/test_x.py::test_fix"],
        "PASS_TO_PASS": ["tests/test_x.py::test_old"],
    })
    controls = {
        "baseline": _exact_phase(
            "baseline", contract, candidate_patch_sha256,
            f2p=False, p2p=True,
        ),
        "reference": _exact_phase(
            "reference", contract, candidate_patch_sha256,
            f2p=True, p2p=True,
        ),
        "candidate_1": _exact_phase(
            "candidate_1", contract, candidate_patch_sha256,
            f2p=True, p2p=True,
        ),
        "candidate_2": _exact_phase(
            "candidate_2", contract, candidate_patch_sha256,
            f2p=True, p2p=True,
        ),
    }
    assessment = assess_controls(controls, protected_patch_paths=())
    evidence = {
        "admission_schema_version": 2,
        "task_contract": contract,
        "task_contract_sha256": contract["contract_sha256"],
        "image_id": "sha256:" + "c" * 64,
        "candidate_patch_sha256": candidate_patch_sha256,
        "candidate_patch_paths": ["a.py"],
        "protected_patch_paths": [],
        "controls": controls,
        **assessment,
    }
    result = {
        "instance_id": source_id,
        **evidence,
        "admission_evidence_sha256": sha256_bytes(
            canonical_json_bytes(evidence)
        ),
        "stream_sha256": _sha(stream),
        "patch_sha256": candidate_patch_sha256,
        "executed": True,
        "f2p_pass": True,
        "p2p_pass": True,
        "resolved": True,
        "reference_controls_passed": True,
        "mutation_f2p_reproduced": True,
        "mutation_p2p_passed": True,
        "trace_preflight_source_sha256": "d" * 64,
        "trace_preflight_retained_steps": 2,
    }
    _jsonl(run / "results.jsonl", [result])

    delta = tmp_path / "delta.jsonl"
    apply_command = "git apply - <<'PATCH'\ndiff --git a/a.py b/a.py\nPATCH"
    delta_row = {
        "instance_id": f"verified-teacher-finalpatch-{source_id}",
        "source_instance_id": source_id,
        "source": "fable5_verified_finalpatch",
        "messages": [
            {"role": "system", "content": "system", "loss": False},
            {"role": "assistant", "content": "", "loss": True,
             "tool_calls": [{
                 "type": "function",
                 "function": {
                     "name": "bash",
                     "arguments": json.dumps({"command": apply_command}),
                 },
             }]},
        ],
    }
    _jsonl(delta, [delta_row])
    delta_manifest = tmp_path / "delta.manifest.json"
    _json(
        delta_manifest,
        {
            "rows": 1,
            "strict_admitted_selected": 1,
            "all_messages_have_boolean_loss": True,
            "terminal_assistant_loss_true": True,
            "source_runs": [str(run.resolve())],
            "decontam": {
                "assertions_pass": True,
                "rows": 1,
                "unique_source_ids": 1,
                "existing_overlap": [],
                "lite_overlap": [],
                "pool_manifest_lite_overlap": 0,
            },
        },
    )
    delta_report = tmp_path / "delta.report.json"
    _json(
        delta_report,
        {
            "strict_admitted": 1,
            "selected_per_instance_cost_usd": {source_id: 0.1},
        },
    )
    monkeypatch.setattr(extension_module, "PRODUCTION_DELTA_ROWS", 1)
    monkeypatch.setattr(
        extension_module,
        "PRODUCTION_DELTA_SHA256",
        _sha(delta),
    )
    monkeypatch.setattr(
        extension_module,
        "PRODUCTION_DELTA_MANIFEST_SHA256",
        _sha(delta_manifest),
    )
    monkeypatch.setattr(
        extension_module,
        "PRODUCTION_DELTA_REPORT_SHA256",
        _sha(delta_report),
    )
    lite = tmp_path / "lite.json"
    _json(lite, ["lite__case-1"])
    return base, delta, delta_manifest, delta_report, lite, run


def test_builds_extension_bound_to_raw_admission_evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    base, delta, delta_manifest, delta_report, lite, run = _fixture(
        tmp_path,
        monkeypatch,
    )
    output = tmp_path / "output"

    manifest = extension_module.build_extension(
        base=base,
        delta=delta,
        delta_manifest=delta_manifest,
        delta_report=delta_report,
        lite_ids=lite,
        out=output,
    )

    assert manifest["rendered"] == 2
    assert manifest["base_rows"] == 1
    assert manifest["new_fable_rows"] == 1
    assert manifest["evaluation_overlap"] == 0
    assert manifest["all_training_gates_complete"] is False
    evidence = manifest["new_fable_admission_evidence"]
    assert len(evidence) == 1
    assert evidence[0]["results"]["path"] == str(
        (run / "results.jsonl").resolve()
    )
    assert evidence[0]["stream"]["sha256"] == _sha(
        next(run.glob("*.stream.jsonl"))
    )
    assert evidence[0]["patch"]["sha256"] == _sha(
        next(run.glob("*.patch"))
    )
    output_rows = [
        json.loads(line)
        for line in (output / "train.jsonl").read_text().splitlines()
    ]
    assert output_rows[-1]["repo"] == "teacher/case"


def test_rejects_late_fable_target_without_the_admitted_patch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    base, delta, delta_manifest, delta_report, lite, _run = _fixture(
        tmp_path,
        monkeypatch,
    )
    row = json.loads(delta.read_text())
    row["messages"][-1]["tool_calls"][-1]["function"]["arguments"] = json.dumps({
        "command": "git diff",
    })
    _jsonl(delta, [row])
    monkeypatch.setattr(extension_module, "PRODUCTION_DELTA_SHA256", _sha(delta))

    with pytest.raises(ValueError, match="does not reproduce raw patch"):
        extension_module.build_extension(
            base=base,
            delta=delta,
            delta_manifest=delta_manifest,
            delta_report=delta_report,
            lite_ids=lite,
            out=tmp_path / "output",
        )


def test_rejects_raw_result_without_two_candidate_passes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    base, delta, delta_manifest, delta_report, lite, run = _fixture(
        tmp_path,
        monkeypatch,
    )
    result_path = run / "results.jsonl"
    result = json.loads(result_path.read_text())
    result["candidate_passed_twice"] = False
    _jsonl(result_path, [result])

    with pytest.raises(ValueError, match="raw admission"):
        extension_module.build_extension(
            base=base,
            delta=delta,
            delta_manifest=delta_manifest,
            delta_report=delta_report,
            lite_ids=lite,
            out=tmp_path / "output",
        )


def test_rejects_delta_overlapping_lite(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    base, delta, delta_manifest, delta_report, lite, _run = _fixture(
        tmp_path,
        monkeypatch,
    )
    row = json.loads(delta.read_text())
    _json(lite, [row["source_instance_id"]])

    with pytest.raises(ValueError, match="evaluation overlap"):
        extension_module.build_extension(
            base=base,
            delta=delta,
            delta_manifest=delta_manifest,
            delta_report=delta_report,
            lite_ids=lite,
            out=tmp_path / "output",
        )
