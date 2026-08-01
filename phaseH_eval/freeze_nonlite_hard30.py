#!/usr/bin/env python3
"""Freeze an exact, training-disjoint SWE-smith hard30 promotion slice."""
from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
from typing import Any

from phaseD_sft.build_mix_v2p10 import (
    PRODUCTION_V2P8_CANONICAL_SHA256,
    PRODUCTION_V2P8_ROWS,
    canonical_rows_sha256,
)
from teacher_platform.generic_trace_replay import (
    ReplayContractError,
    parse_patch_paths,
    protected_paths,
)


PRODUCTION_SOURCE_SHA256 = (
    "4876b22fe7e3c1ac86c867b89aa68e7021450ec60b5d6b163bbf0042ce8bcab8"
)
PRODUCTION_SOURCE_MANIFEST_SHA256 = (
    "a0c57122818ea82e717caad1b75fd03124e434a7d540ee154fd4cca728ae7c2e"
)
_SHA256_RE = re.compile(r"^(?:sha256:)?[0-9a-f]{64}$")


def _sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise ValueError(f"unreadable JSONL: {path}") from exc
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid JSONL: {path}:{line_number}") from exc
        if not isinstance(row, dict):
            raise ValueError(f"non-object JSONL row: {path}:{line_number}")
        rows.append(row)
    return rows


def _exclusion_ids(paths: Sequence[Path]) -> tuple[set[str], list[dict[str, Any]]]:
    instance_ids: set[str] = set()
    bindings = []
    for path in paths:
        try:
            text = path.read_text(encoding="utf-8")
        except OSError as exc:
            raise ValueError(f"unreadable exclusion artifact: {path}") from exc
        try:
            payload = json.loads(text)
        except json.JSONDecodeError:
            payload = [
                json.loads(line)
                for line in text.splitlines()
                if line.strip()
            ]
        values = payload if isinstance(payload, list) else [payload]
        for value in values:
            if not isinstance(value, Mapping):
                raise ValueError(f"invalid exclusion artifact: {path}")
            for key in ("instance_id", "source_instance_id"):
                if isinstance(value.get(key), str):
                    instance_ids.add(value[key])
            for key in ("instance_ids", "selected_ids", "lite_exclusion_ids"):
                listed = value.get(key, [])
                if not isinstance(listed, list) or any(
                    not isinstance(item, str) for item in listed
                ):
                    raise ValueError(f"invalid exclusion field {key}: {path}")
                instance_ids.update(listed)
        bindings.append({
            "path": str(path.resolve()),
            "sha256": _sha256_path(path),
            "bytes": path.stat().st_size,
        })
    return instance_ids, bindings


def _publish_directory_atomic(output: Path, build) -> None:
    from phaseD_sft.filter_edit_decisive_dataset import _rename_noreplace

    if os.path.lexists(output):
        raise FileExistsError(f"refusing to overwrite {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    stage_root = Path(
        tempfile.mkdtemp(prefix=f".{output.name}.", dir=output.parent)
    )
    payload = stage_root / "payload"
    try:
        build(payload)
        _rename_noreplace(payload, output)
        stage_root.rmdir()
    except BaseException:
        shutil.rmtree(stage_root, ignore_errors=True)
        raise


def freeze_nonlite_hard30(
    *,
    source: Path,
    source_manifest: Path,
    base_instance_ids: set[str],
    exclusion_paths: Sequence[Path],
    local_image_ids: Mapping[str, str],
    baseline_preflight: Mapping[str, Mapping[str, Any]],
    output: Path,
    expected_source_sha256: str = PRODUCTION_SOURCE_SHA256,
    expected_source_manifest_sha256: str = PRODUCTION_SOURCE_MANIFEST_SHA256,
    base_canonical_rows_sha256: str | None = None,
) -> dict[str, Any]:
    """Select the first 30 exact eligible source rows and publish without overwrite."""

    source = Path(source)
    source_manifest = Path(source_manifest)
    output = Path(output)
    if os.path.lexists(output):
        raise FileExistsError(f"refusing to overwrite {output}")
    if _sha256_path(source) != expected_source_sha256:
        raise ValueError("non-Lite source SHA-256 mismatch")
    if _sha256_path(source_manifest) != expected_source_manifest_sha256:
        raise ValueError("non-Lite source manifest SHA-256 mismatch")
    rows = _read_jsonl(source)
    if len(rows) != 40:
        raise ValueError(f"expected the registered 40-row source, got {len(rows)}")
    ids = [row.get("instance_id") for row in rows]
    if any(not isinstance(value, str) or not value for value in ids):
        raise ValueError("source contains a missing instance ID")
    if len(ids) != len(set(ids)):
        raise ValueError("source contains duplicate instance IDs")

    excluded, exclusion_bindings = _exclusion_ids(
        tuple(Path(path) for path in exclusion_paths)
    )
    excluded.update(base_instance_ids)
    eligible_source_rows = [
        row for row in rows if row["instance_id"] not in excluded
    ]
    if len(eligible_source_rows) < 30:
        raise ValueError(
            f"only {len(eligible_source_rows)} decontaminated source rows remain"
        )
    eligible = []
    for source_row in eligible_source_rows:
        mutation_patch = source_row.get("patch")
        try:
            mutation_paths = parse_patch_paths(mutation_patch)
        except (ReplayContractError, TypeError) as exc:
            raise ValueError(
                f"invalid mutation patch: {source_row.get('instance_id')}"
            ) from exc
        protected = protected_paths(mutation_paths)
        if protected:
            raise ValueError(
                f"mutation patch changes protected paths for "
                f"{source_row.get('instance_id')}: {protected}"
            )
        row = dict(source_row)
        row["task_patch_role"] = "bug_inducing_mutation"
        row["mutation_patch_sha256"] = hashlib.sha256(
            mutation_patch.encode("utf-8")
        ).hexdigest()
        eligible.append(row)
    missing_images = sorted({
        str(row.get("image_name") or "")
        for row in eligible
        if (
            not isinstance(row.get("image_name"), str)
            or not row["image_name"]
            or not _SHA256_RE.fullmatch(str(local_image_ids.get(row["image_name"], "")))
        )
    })
    if missing_images:
        raise ValueError(f"selected rows lack exact local images: {missing_images}")
    for row in eligible:
        if (
            not isinstance(row.get("problem_statement"), str)
            or not row["problem_statement"].strip()
            or not isinstance(row.get("FAIL_TO_PASS"), list)
            or not row["FAIL_TO_PASS"]
        ):
            raise ValueError(f"invalid selected task: {row.get('instance_id')}")

    missing_preflight = sorted(
        set(row["instance_id"] for row in eligible) - set(baseline_preflight)
    )
    if missing_preflight:
        raise ValueError(
            f"baseline preflight is missing eligible tasks: {missing_preflight}"
        )
    preflight_rows = []
    valid_rows = []
    invalid_ids = []
    for row in eligible:
        instance_id = row["instance_id"]
        result = dict(baseline_preflight[instance_id])
        expected_task_sha256 = hashlib.sha256(_canonical_json(row)).hexdigest()
        expected_image_id = local_image_ids[row["image_name"]]
        if (
            result.get("schema_version") != 2
            or result.get("instance_id") != instance_id
            or result.get("task_sha256") != expected_task_sha256
            or result.get("patch_sha256") != hashlib.sha256(b"").hexdigest()
            or result.get("mutation_patch_sha256")
            != row["mutation_patch_sha256"]
            or result.get("image_id") != expected_image_id
        ):
            raise ValueError(f"baseline preflight binding mismatch: {instance_id}")
        valid = bool(
            result.get("status") == "empty_patch"
            and result.get("reference_f2p_pass") is True
            and result.get("mutation_f2p_pass") is False
            and result.get("mutation_patch_applied") is True
            and result.get("task_baseline_valid") is True
            and result.get("infra_error") is None
        )
        preflight_rows.append(result)
        if valid:
            valid_rows.append(row)
        else:
            invalid_ids.append(instance_id)
    if len(valid_rows) < 30:
        raise ValueError(
            f"only {len(valid_rows)} baseline-valid decontaminated rows remain"
        )
    selected = valid_rows[:30]

    selected_ids = [row["instance_id"] for row in selected]
    preflight_bindings = [
        {
            "instance_id": result["instance_id"],
            "path": (
                "baseline_preflight/results/"
                + hashlib.sha256(result["instance_id"].encode()).hexdigest()
                + ".json"
            ),
            "sha256": hashlib.sha256(_canonical_json(result) + b"\n").hexdigest(),
        }
        for result in preflight_rows
    ]
    preflight_binding_map = {
        row["instance_id"]: row for row in preflight_bindings
    }
    preflight_summary = {
        "schema_version": 1,
        "contract": (
            "clean F2P passes; mutation applies and reproduces exact F2P failure"
        ),
        "eligible_rows": len(eligible),
        "valid_rows": len(valid_rows),
        "invalid_rows": len(invalid_ids),
        "invalid_ids": sorted(invalid_ids),
        "result_bindings": preflight_bindings,
        "results_sha256": hashlib.sha256(
            _canonical_json(preflight_bindings)
        ).hexdigest(),
    }
    holder: dict[str, Any] = {}

    def build(stage: Path) -> None:
        stage.mkdir(mode=0o700)
        preflight_root = stage / "baseline_preflight" / "results"
        preflight_root.mkdir(parents=True)
        for result, binding in zip(
            preflight_rows,
            preflight_bindings,
            strict=True,
        ):
            (stage / binding["path"]).write_bytes(_canonical_json(result) + b"\n")
        (stage / "baseline_preflight" / "summary.json").write_text(
            json.dumps(preflight_summary, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        tasks = stage / "tasks.jsonl"
        with tasks.open("w", encoding="utf-8") as handle:
            for row in selected:
                handle.write(_canonical_json(row).decode("utf-8") + "\n")
        ids_path = stage / "ids.json"
        ids_path.write_text(
            json.dumps(selected_ids, indent=2) + "\n",
            encoding="utf-8",
        )
        manifest = {
            "schema_version": 3,
            "name": "v2p10_nonlite_hard30",
            "complete": True,
            "training_eligible": False,
            "selection_frozen_before_eval": True,
            "selection": "source_order_after_exact_training_and_teacher_exclusions",
            "source_rows": len(rows),
            "eligible_rows": len(eligible),
            "selected_rows": len(selected),
            "baseline_preflight": preflight_summary,
            "source": {
                "path": str(source.resolve()),
                "sha256": _sha256_path(source),
                "bytes": source.stat().st_size,
            },
            "source_manifest": {
                "path": str(source_manifest.resolve()),
                "sha256": _sha256_path(source_manifest),
                "bytes": source_manifest.stat().st_size,
            },
            "base_instance_count": len(base_instance_ids),
            "base_canonical_rows_sha256": base_canonical_rows_sha256,
            "base_instance_ids_sha256": hashlib.sha256(
                "\n".join(sorted(base_instance_ids)).encode("utf-8")
            ).hexdigest(),
            "exclusion_artifacts": exclusion_bindings,
            "tasks_sha256": _sha256_path(tasks),
            "ids_sha256": _sha256_path(ids_path),
            "task_bindings": [
                {
                    "instance_id": row["instance_id"],
                    "task_sha256": hashlib.sha256(
                        _canonical_json(row)
                    ).hexdigest(),
                    "image_name": row["image_name"],
                    "image_id": local_image_ids[row["image_name"]],
                    "task_patch_role": row["task_patch_role"],
                    "mutation_patch_sha256": row["mutation_patch_sha256"],
                    "baseline_preflight_result_sha256": preflight_binding_map[
                        row["instance_id"]
                    ]["sha256"],
                }
                for row in selected
            ],
        }
        (stage / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        holder.update(manifest)

    _publish_directory_atomic(output, build)
    return holder


def _local_image_ids(names: Sequence[str]) -> dict[str, str]:
    if not names:
        return {}
    result = subprocess.run(
        [
            "docker",
            "image",
            "inspect",
            "--format",
            "{{.Id}}",
            *names,
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise ValueError(f"local image inspection failed: {result.stderr[-500:]}")
    identities = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    if len(identities) != len(names):
        raise ValueError("local image inspection returned an incomplete identity set")
    return dict(zip(names, identities, strict=True))


def _main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source",
        type=Path,
        default=Path("data/expert_iter2_pool40_v2.jsonl"),
    )
    parser.add_argument(
        "--source-manifest",
        type=Path,
        default=Path("data/expert_iter2_pool40_v2_manifest.json"),
    )
    parser.add_argument(
        "--base",
        type=Path,
        default=Path("data/teacher_train_mix_v2p8"),
    )
    parser.add_argument("--exclude", type=Path, action="append", default=[])
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("data/v2p10_nonlite_hard30"),
    )
    parser.add_argument("--preflight-workers", type=int, default=4)
    args = parser.parse_args()
    if args.preflight_workers < 1 or args.preflight_workers > 8:
        raise ValueError("preflight workers must be in [1,8]")

    from datasets import load_from_disk

    base_rows = [dict(row) for row in load_from_disk(str(args.base))]
    base_ids = {
        str(row["instance_id"])
        for row in base_rows
        if isinstance(row.get("instance_id"), str)
    }
    base_rows_sha256 = canonical_rows_sha256(base_rows)
    if (
        len(base_rows) != PRODUCTION_V2P8_ROWS
        or base_rows_sha256 != PRODUCTION_V2P8_CANONICAL_SHA256
    ):
        raise ValueError("registered v2.8 base identity mismatch")
    source_rows = _read_jsonl(args.source)
    image_names = [
        row["image_name"]
        for row in source_rows
        if isinstance(row.get("image_name"), str)
    ]
    local_image_ids = _local_image_ids(image_names)
    from phaseH_eval.score_nonlite_hard30 import score_prediction

    def preflight(source_row: Mapping[str, Any]) -> dict[str, Any]:
        task = dict(source_row)
        mutation_patch = task.get("patch")
        if isinstance(mutation_patch, str):
            task["task_patch_role"] = "bug_inducing_mutation"
            task["mutation_patch_sha256"] = hashlib.sha256(
                mutation_patch.encode("utf-8")
            ).hexdigest()
        instance_id = str(task.get("instance_id") or "")
        return score_prediction(
            task,
            {"instance_id": instance_id, "model_patch": ""},
            expected_image_id=str(local_image_ids.get(task.get("image_name"), "")),
        )

    with ThreadPoolExecutor(max_workers=args.preflight_workers) as pool:
        preflight_rows = list(pool.map(preflight, source_rows))
    baseline_preflight = {
        row["instance_id"]: row
        for row in preflight_rows
        if isinstance(row.get("instance_id"), str)
    }
    manifest = freeze_nonlite_hard30(
        source=args.source,
        source_manifest=args.source_manifest,
        base_instance_ids=base_ids,
        exclusion_paths=tuple(args.exclude),
        local_image_ids=local_image_ids,
        baseline_preflight=baseline_preflight,
        output=args.out,
        base_canonical_rows_sha256=base_rows_sha256,
    )
    print(json.dumps({
        "output": str(args.out),
        "selected_rows": manifest["selected_rows"],
        "tasks_sha256": manifest["tasks_sha256"],
        "ids_sha256": manifest["ids_sha256"],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
