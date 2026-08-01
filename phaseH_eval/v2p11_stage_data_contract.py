#!/usr/bin/env python3
"""Bind the frozen v2.11 Stage-A data, Fable disposition, and exclusions."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PRODUCTION_STAGE_ROWS = 1_247
PRODUCTION_BASE_ROWS = 1_211
PRODUCTION_NEW_FABLE_ROWS = 36
PRODUCTION_STAGE_MANIFEST_SHA256 = (
    "23090fb8664134f3b8ec2c891b25fef1ca368abdba4bdc6fe2ab114b99f420a0"
)
PRODUCTION_STAGE_TRAIN_SHA256 = (
    "e671ca08b9cdefcea59ca767c6d6a5158101ccdcc3b2257dd797d9bdfceca8cd"
)
PRODUCTION_STRICT_MANIFEST_SHA256 = (
    "3708c56a6fa6b2baf68fc69b523c90e9ed30b70936ab67e8bd3e766ce68121e6"
)
PRODUCTION_CAMPAIGN_MANIFEST_SHA256 = (
    "ba291d069bef036680b05f6d5a73f7f9de662c575ea7ca83f17d4fbf1c81219b"
)
PRODUCTION_CAMPAIGN_RESULTS_SHA256 = (
    "afb74363aecfd4abba1258e383f996d262ecb791a92ff5f1f4b1295427e1a35f"
)
PRODUCTION_CAMPAIGN_RESOLVED_SHA256 = (
    "16dc6abc8f46bd44444856907ee00cc646b039d32fe7916ce8fdcbede53fccbd"
)
PRODUCTION_CAMPAIGN_REJECTED_SHA256 = (
    "bfe3cf1396839ba53781811fcf40c60809a74d21b1a75833fc365f852fddf687"
)
PRODUCTION_EXCLUSION_SHA256 = (
    "00171a1e7796fac2103317bcf4f05af42e46ff92db0041fd4a26ccbaaa7e4bd8"
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid JSON artifact: {path}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"JSON artifact is not an object: {path}")
    return value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise ValueError(f"invalid JSONL artifact: {path}") from exc
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(lines, start=1):
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"invalid JSONL artifact: {path}:{line_number}"
            ) from exc
        if not isinstance(row, dict):
            raise ValueError(
                f"JSONL row is not an object: {path}:{line_number}"
            )
        rows.append(row)
    return rows


def _resolve(value: str | Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def _ids(rows: Sequence[Mapping[str, Any]], *, label: str) -> list[str]:
    values = [row.get("instance_id") for row in rows]
    if (
        any(not isinstance(value, str) or not value for value in values)
        or len(values) != len(set(values))
    ):
        raise ValueError(f"{label} instance IDs are invalid")
    return [str(value) for value in values]


def _normalized_repo(value: object) -> str:
    if not isinstance(value, str):
        return ""
    return value.strip().casefold().replace("__", "/")


def _repo_from_instance_id(instance_id: str) -> str:
    if "__" not in instance_id:
        return ""
    owner, remainder = instance_id.split("__", 1)
    repository = remainder.split(".", 1)[0]
    return _normalized_repo(f"{owner}/{repository}")


def _validate_campaign(
    campaign_root: Path,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[str]]:
    manifest_path = campaign_root / "manifest.json"
    results_path = campaign_root / "results.jsonl"
    resolved_path = campaign_root / "resolved.jsonl"
    rejected_path = campaign_root / "rejected.jsonl"
    manifest = _read_json(manifest_path)
    bindings = {
        "results_sha256": results_path,
        "resolved_sha256": resolved_path,
        "rejected_sha256": rejected_path,
    }
    if any(
        manifest.get(key) != _sha256(path)
        for key, path in bindings.items()
    ):
        raise ValueError("Fable campaign artifact changed")
    results_ids = _ids(_read_jsonl(results_path), label="campaign results")
    resolved_rows = _read_jsonl(resolved_path)
    resolved_ids = _ids(resolved_rows, label="campaign resolved")
    rejected_ids = _ids(
        _read_jsonl(rejected_path),
        label="campaign rejected",
    )
    if (
        manifest.get("schema_version") != 2
        or manifest.get("complete") is not True
        or manifest.get("selection_excluded") != 0
        or manifest.get("attempted") != len(results_ids)
        or manifest.get("resolved") != len(resolved_ids)
        or manifest.get("rejected") != len(rejected_ids)
        or manifest.get("training_admitted") != len(resolved_ids)
        or set(resolved_ids).intersection(rejected_ids)
        or set(results_ids) != set(resolved_ids).union(rejected_ids)
    ):
        raise ValueError("Fable campaign disposition is incomplete")
    return manifest, resolved_rows, rejected_ids


def _validate_strict_fable(
    stage_manifest: Mapping[str, Any],
    *,
    campaign_root: Path,
    campaign_manifest: Mapping[str, Any],
    campaign_resolved_rows: Sequence[Mapping[str, Any]],
) -> tuple[Path, dict[str, Any], list[str], list[str]]:
    fable = stage_manifest.get("fable")
    if not isinstance(fable, Mapping) or not isinstance(
        fable.get("path"),
        str,
    ):
        raise ValueError("Stage-A Fable binding is incomplete")
    strict_root = _resolve(fable["path"])
    manifest_path = strict_root / "manifest.json"
    train_path = strict_root / "train.jsonl"
    manifest = _read_json(manifest_path)
    embedded = fable.get("manifest")
    exclusions = manifest.get("distillation_exclusions")
    admitted_ids = _ids(
        _read_jsonl(train_path),
        label="strict Fable",
    )
    campaign_resolved_ids = _ids(
        campaign_resolved_rows,
        label="campaign resolved",
    )
    if not isinstance(exclusions, Mapping):
        raise ValueError("Fable distillation exclusions are incomplete")
    excluded_bindings = exclusions.get("artifact_bindings")
    admitted_bindings = manifest.get("artifact_bindings")
    if (
        not isinstance(excluded_bindings, list)
        or not isinstance(admitted_bindings, list)
        or any(
            not isinstance(binding, Mapping)
            for binding in [*excluded_bindings, *admitted_bindings]
        )
    ):
        raise ValueError("Fable disposition bindings are incomplete")
    excluded_ids = _ids(
        [
            dict(binding)
            for binding in excluded_bindings
            if isinstance(binding, Mapping)
        ],
        label="Fable distillation exclusions",
    )
    admitted_by_id = {
        row["instance_id"]: row
        for row in campaign_resolved_rows
    }
    binding_by_id = {
        binding["instance_id"]: binding
        for binding in admitted_bindings
        if isinstance(binding, Mapping)
        and isinstance(binding.get("instance_id"), str)
    }
    required_true = (
        "resolved",
        "training_admitted",
        "executed",
        "f2p_pass",
        "p2p_pass",
        "reference_passed",
        "reference_controls_passed",
        "baseline_failed",
        "candidate_passed_twice",
        "mutation_f2p_reproduced",
        "mutation_p2p_passed",
        "protected_stable",
    )
    bound_hashes = (
        "admission_evidence_sha256",
        "patch_sha256",
        "stream_sha256",
        "task_contract_sha256",
    )
    for instance_id in admitted_ids:
        row = admitted_by_id.get(instance_id)
        binding = binding_by_id.get(instance_id)
        if (
            not isinstance(row, Mapping)
            or not isinstance(binding, Mapping)
            or row.get("admission_schema_version") != 2
            or any(row.get(key) is not True for key in required_true)
            or row.get("cli_rc") != 0
            or type(row.get("patch_len")) is not int
            or row["patch_len"] <= 0
            or row.get("rejection_reasons") != []
            or any(
                not isinstance(row.get(key), str)
                or len(row[key]) != 64
                or any(character not in "0123456789abcdef" for character in row[key])
                or binding.get(key) != row[key]
                for key in bound_hashes
            )
        ):
            raise ValueError(
                f"strict Fable admission evidence is incomplete: {instance_id}"
            )

    format_gate = manifest.get("standard_native_format_loss_gate")
    if not isinstance(format_gate, Mapping):
        raise ValueError("strict Fable render/loss evidence is incomplete")
    format_counts = format_gate.get("counts")
    format_artifact = format_gate.get("artifact")
    if (
        format_gate.get("status") != "passed"
        or format_gate.get("failure_count") != 0
        or format_gate.get("samples") != len(admitted_ids)
        or not isinstance(format_counts, Mapping)
        or format_counts.get("examples") != len(admitted_ids)
        or format_counts.get("fallback_spans") != 0
        or format_counts.get("supervised_fallback_spans") != 0
        or not isinstance(format_artifact, Mapping)
        or not isinstance(format_artifact.get("path"), str)
    ):
        raise ValueError("strict Fable render/loss evidence is incomplete")
    format_path = (strict_root / format_artifact["path"]).resolve()
    if (
        not format_path.is_relative_to(strict_root)
        or not format_path.is_file()
        or format_artifact.get("sha256") != _sha256(format_path)
        or format_artifact.get("bytes") != format_path.stat().st_size
    ):
        raise ValueError("strict Fable render/loss artifact changed")
    if (
        manifest != embedded
        or fable.get("manifest_sha256") != _sha256(manifest_path)
        or manifest.get("schema_version") != 2
        or manifest.get("complete") is not True
        or manifest.get("all_training_gates_complete") is not True
        or manifest.get("success_path_distilled") is not True
        or manifest.get("train_jsonl_sha256") != _sha256(train_path)
        or manifest.get("rendered") != len(admitted_ids)
        or manifest.get("training_admitted") != len(admitted_ids)
        or manifest.get("resolved_in") != len(campaign_resolved_ids)
        or manifest.get("merge_manifest_sha256")
        != _sha256(campaign_root / "manifest.json")
        or manifest.get("merge_resolved_sha256")
        != campaign_manifest.get("resolved_sha256")
        or exclusions.get("count") != len(excluded_ids)
        or len(admitted_bindings) != len(admitted_ids)
        or {
            binding.get("instance_id")
            for binding in admitted_bindings
            if isinstance(binding, Mapping)
        }
        != set(admitted_ids)
        or any(
            binding.get("stage") != "success_path_distillation"
            or not isinstance(binding.get("reason"), str)
            or not binding["reason"]
            for binding in excluded_bindings
            if isinstance(binding, Mapping)
        )
        or set(admitted_ids).intersection(excluded_ids)
        or set(campaign_resolved_ids)
        != set(admitted_ids).union(excluded_ids)
    ):
        raise ValueError("strict Fable disposition is incomplete")
    return strict_root, manifest, admitted_ids, excluded_ids


def validate_stage_data(
    *,
    stage_data: Path,
    training_contract_path: Path,
    exclusions_path: Path,
    campaign_root: Path,
    require_production_identity: bool = False,
) -> dict[str, Any]:
    stage_data = Path(stage_data).resolve()
    training_contract_path = Path(training_contract_path).resolve()
    exclusions_path = Path(exclusions_path).resolve()
    campaign_root = Path(campaign_root).resolve()
    manifest_path = stage_data / "manifest.json"
    train_path = stage_data / "train.jsonl"
    manifest = _read_json(manifest_path)
    rows = _read_jsonl(train_path)
    training = _read_json(training_contract_path)
    row_ids = _ids(rows, label="Stage-A")
    base_rows = training.get("base_rows")
    new_fable_rows = training.get("fable_rows")
    if (
        manifest.get("schema_version") != 2
        or manifest.get("complete") is not True
        or manifest.get("dataset_variant") != "teacher_train_mix_v2p11"
        or manifest.get("training_admitted") != len(rows)
        or manifest.get("all_training_gates_complete") is not True
        or manifest.get("rendered") != len(rows)
        or manifest.get("base_rows") != base_rows
        or manifest.get("fable_rows") != new_fable_rows
        or training.get("rows") != len(rows)
        or type(base_rows) is not int
        or type(new_fable_rows) is not int
        or base_rows + new_fable_rows != len(rows)
        or training.get("dataset_manifest_sha256") != _sha256(manifest_path)
        or training.get("train_jsonl_sha256") != _sha256(train_path)
        or manifest.get("train_jsonl_sha256") != _sha256(train_path)
    ):
        raise ValueError("Stage-A training artifact binding is incomplete")

    campaign_manifest, campaign_resolved_rows, campaign_rejected_ids = (
        _validate_campaign(campaign_root)
    )
    campaign_resolved_ids = _ids(
        campaign_resolved_rows,
        label="campaign resolved",
    )
    strict_root, _strict_manifest, strict_ids, distilled_ids = (
        _validate_strict_fable(
            manifest,
            campaign_root=campaign_root,
            campaign_manifest=campaign_manifest,
            campaign_resolved_rows=campaign_resolved_rows,
        )
    )
    fable_instance_ids = manifest.get("fable_instance_ids")
    if (
        not isinstance(fable_instance_ids, list)
        or fable_instance_ids != strict_ids
        or row_ids[-new_fable_rows:] != strict_ids
    ):
        raise ValueError("Stage-A new Fable row binding is incomplete")

    lineage = {
        "inherited_pinned_rows": 0,
        "historical_verified_revision_rows": 0,
        "new_strict_rows": 0,
    }
    for row in rows:
        source = row.get("source")
        source_text = source if isinstance(source, str) else ""
        if source_text.startswith("teacher:fable5:"):
            lineage["inherited_pinned_rows"] += 1
        elif source_text == "teacher:claude:claude-fable-5:verified-revision":
            lineage["historical_verified_revision_rows"] += 1
        elif source_text == "teacher:claude:claude-fable-5":
            lineage["new_strict_rows"] += 1
        elif "fable" in source_text.casefold():
            raise ValueError(f"unclassified Fable lineage: {source_text}")
    lineage["total_stage_a_fable_rows"] = sum(lineage.values())
    if lineage["new_strict_rows"] != new_fable_rows:
        raise ValueError("Stage-A new Fable lineage count changed")

    exclusions = _read_json(exclusions_path)
    excluded_ids_value = exclusions.get("instance_ids")
    excluded_repos_value = exclusions.get("repo_denylist")
    if (
        not isinstance(excluded_ids_value, list)
        or len(excluded_ids_value) != len(set(excluded_ids_value))
        or any(not isinstance(value, str) for value in excluded_ids_value)
        or not isinstance(excluded_repos_value, list)
        or len(excluded_repos_value) != len(set(excluded_repos_value))
        or any(not isinstance(value, str) for value in excluded_repos_value)
    ):
        raise ValueError("combined evaluation exclusion is incomplete")
    excluded_ids = set(excluded_ids_value)
    excluded_repos = {
        _normalized_repo(value)
        for value in excluded_repos_value
    }
    overlap_ids = sorted(excluded_ids.intersection(row_ids))
    overlap_repos = sorted({
        _normalized_repo(row.get("repo"))
        or _repo_from_instance_id(instance_id)
        for instance_id, row in zip(row_ids, rows)
    }.intersection(excluded_repos))
    if overlap_ids or overlap_repos:
        raise ValueError(
            "Stage-A row violates combined evaluation exclusion"
        )

    freeze = {
        "attempted": len(campaign_resolved_ids) + len(campaign_rejected_ids),
        "resolved": len(campaign_resolved_ids),
        "collection_rejected": len(campaign_rejected_ids),
        "strict_admitted": len(strict_ids),
        "distillation_rejected": len(distilled_ids),
    }
    exclusion_contract = {
        "ids": len(excluded_ids),
        "repositories": len(excluded_repos),
        "overlap": 0,
        "sha256": _sha256(exclusions_path),
    }
    if require_production_identity and (
        len(rows) != PRODUCTION_STAGE_ROWS
        or base_rows != PRODUCTION_BASE_ROWS
        or new_fable_rows != PRODUCTION_NEW_FABLE_ROWS
        or lineage
        != {
            "inherited_pinned_rows": 31,
            "historical_verified_revision_rows": 10,
            "new_strict_rows": 36,
            "total_stage_a_fable_rows": 77,
        }
        or freeze
        != {
            "attempted": 60,
            "resolved": 47,
            "collection_rejected": 13,
            "strict_admitted": 36,
            "distillation_rejected": 11,
        }
        or exclusion_contract["ids"] != 707
        or exclusion_contract["repositories"] != 12
        or _sha256(manifest_path) != PRODUCTION_STAGE_MANIFEST_SHA256
        or _sha256(train_path) != PRODUCTION_STAGE_TRAIN_SHA256
        or _sha256(strict_root / "manifest.json")
        != PRODUCTION_STRICT_MANIFEST_SHA256
        or _sha256(campaign_root / "manifest.json")
        != PRODUCTION_CAMPAIGN_MANIFEST_SHA256
        or campaign_manifest.get("results_sha256")
        != PRODUCTION_CAMPAIGN_RESULTS_SHA256
        or campaign_manifest.get("resolved_sha256")
        != PRODUCTION_CAMPAIGN_RESOLVED_SHA256
        or campaign_manifest.get("rejected_sha256")
        != PRODUCTION_CAMPAIGN_REJECTED_SHA256
        or exclusion_contract["sha256"] != PRODUCTION_EXCLUSION_SHA256
    ):
        raise ValueError("production Stage-A data identity changed")

    return {
        "stage_data": str(stage_data),
        "rows": len(rows),
        "base_rows": base_rows,
        "new_fable_rows": new_fable_rows,
        "dataset_manifest_sha256": _sha256(manifest_path),
        "train_jsonl_sha256": _sha256(train_path),
        "strict_fable_manifest_sha256": _sha256(
            strict_root / "manifest.json"
        ),
        "campaign_manifest_sha256": _sha256(
            campaign_root / "manifest.json"
        ),
        "fable_lineage": lineage,
        "fable_freeze": freeze,
        "evaluation_exclusion": exclusion_contract,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage-data", type=Path, required=True)
    parser.add_argument("--training-contract", type=Path, required=True)
    parser.add_argument("--exclusions", type=Path, required=True)
    parser.add_argument("--campaign-root", type=Path, required=True)
    parser.add_argument(
        "--require-production-identity",
        action="store_true",
    )
    args = parser.parse_args(argv)
    contract = validate_stage_data(
        stage_data=args.stage_data,
        training_contract_path=args.training_contract,
        exclusions_path=args.exclusions,
        campaign_root=args.campaign_root,
        require_production_identity=args.require_production_identity,
    )
    print(json.dumps(contract, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
