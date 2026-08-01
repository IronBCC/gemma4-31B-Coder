#!/usr/bin/env python3
"""Build v2.11 from the frozen v2.10 mix plus strict Fable traces."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from phaseD_sft.build_mix_v2p10 import (
    FABLE_SOURCE,
    _bound_file_index,
    _canonical_json,
    _publish_directory_atomic,
    _sha256_path,
    _training_projection,
    _validate_fable_bindings,
    _validate_manifest_dataset,
    canonical_rows_sha256,
)


V2P10_VARIANT = "teacher_train_mix_v2p10"
V2P11_VARIANT = "teacher_train_mix_v2p11"
PRODUCTION_V2P10_ROWS = 1_211
PRODUCTION_V2P10_CANONICAL_SHA256 = (
    "7ab7b8c4aa262b7bba2cc6c19d3917e0e97b4add6b77bb0a949b88246dda415f"
)


def _represented_fable_source_ids(
    rows: list[dict[str, Any]],
    manifest: dict[str, Any],
) -> set[str]:
    represented = {
        value
        for row in rows
        for value in (row.get("source_instance_id"),)
        if isinstance(value, str) and value
    }
    bindings = manifest.get("artifact_bindings")
    if isinstance(bindings, list):
        represented.update(
            binding["source_instance_id"]
            for binding in bindings
            if isinstance(binding, dict)
            and binding.get("delta_kind") == "fable_revision"
            and isinstance(binding.get("source_instance_id"), str)
            and binding["source_instance_id"]
        )
    return represented


def build_mix(
    base: Path,
    fable_delta: Path,
    out: Path,
) -> dict[str, object]:
    """Publish a gated v2.11 candidate without mutating frozen v2.10."""

    base = Path(base).resolve()
    fable_delta = Path(fable_delta).resolve()
    out = Path(out).resolve()
    if os.path.lexists(out):
        raise FileExistsError(f"refusing to overwrite {out}")

    base_rows, base_manifest = _validate_manifest_dataset(
        base,
        expected_variant=V2P10_VARIANT,
    )
    if (
        len(base_rows) != PRODUCTION_V2P10_ROWS
        or canonical_rows_sha256(base_rows)
        != PRODUCTION_V2P10_CANONICAL_SHA256
    ):
        raise ValueError("frozen v2.10 base content hash mismatch")

    fable_rows, fable_manifest = _validate_manifest_dataset(fable_delta)
    _validate_fable_bindings(fable_rows, fable_manifest)
    if not fable_rows:
        raise ValueError("strict Fable delta is empty")
    if not all(row.get("source") == FABLE_SOURCE for row in fable_rows):
        raise ValueError("v2.11 delta must contain only strict Fable rows")

    projected_base = [_training_projection(row) for row in base_rows]
    projected_fable = [_training_projection(row) for row in fable_rows]
    base_ids = {row["instance_id"] for row in projected_base}
    fable_ids = [row["instance_id"] for row in projected_fable]
    if len(fable_ids) != len(set(fable_ids)) or base_ids & set(fable_ids):
        raise ValueError("duplicate instance IDs in v2.11 mix")

    represented_sources = _represented_fable_source_ids(
        base_rows,
        base_manifest,
    )
    overlapping_sources = sorted(represented_sources & set(fable_ids))
    if overlapping_sources:
        raise ValueError(
            "Fable source already represented in v2.10: "
            + ", ".join(overlapping_sources)
        )

    base_messages = {
        canonical_rows_sha256([{"messages": row["messages"]}])
        for row in projected_base
    }
    duplicate_content = [
        row["instance_id"]
        for row in projected_fable
        if canonical_rows_sha256([{"messages": row["messages"]}])
        in base_messages
    ]
    if duplicate_content:
        raise ValueError(
            "Fable message content already represented in v2.10: "
            + ", ".join(sorted(duplicate_content))
        )

    combined = [*projected_base, *projected_fable]
    manifest_holder: dict[str, object] = {}

    def publish(stage: Path) -> None:
        from datasets import Dataset

        Dataset.from_list(combined).save_to_disk(str(stage))
        train_path = stage / "train.jsonl"
        with train_path.open("w", encoding="utf-8") as handle:
            for row in combined:
                handle.write(
                    _canonical_json(row).decode("utf-8") + "\n"
                )
        manifest: dict[str, object] = {
            "schema_version": 2,
            "complete": True,
            "dataset_variant": V2P11_VARIANT,
            "base_variant": V2P10_VARIANT,
            "base_rows": len(projected_base),
            "fable_rows": len(projected_fable),
            "rendered": len(combined),
            "training_admitted": 0,
            "all_training_gates_complete": False,
            "canonical_rows_sha256": canonical_rows_sha256(combined),
            "train_jsonl_sha256": _sha256_path(train_path),
            "base": {
                "path": str(base),
                "manifest": base_manifest,
                "manifest_sha256": _sha256_path(
                    base / "manifest.json"
                ),
                "files": _bound_file_index(base),
            },
            "fable": {
                "path": str(fable_delta),
                "manifest": fable_manifest,
                "manifest_sha256": _sha256_path(
                    fable_delta / "manifest.json"
                ),
                "files": _bound_file_index(fable_delta),
            },
            "fable_instance_ids": fable_ids,
            "artifact_bindings": fable_manifest["artifact_bindings"],
            "exclusions": fable_manifest["exclusions"],
            "success_path_distilled": True,
            "standard_native_format_loss_gate": {
                "status": "pending_full_dataset_verification",
                "failure_count": None,
            },
        }
        (stage / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        manifest_holder.update(manifest)

    _publish_directory_atomic(out, publish)
    return manifest_holder


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--fable", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    manifest = build_mix(args.base, args.fable, args.out)
    print(json.dumps({
        "base_rows": manifest["base_rows"],
        "fable_rows": manifest["fable_rows"],
        "rendered": manifest["rendered"],
        "all_training_gates_complete": manifest[
            "all_training_gates_complete"
        ],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
