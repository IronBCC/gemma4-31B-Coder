"""Validate non-weight checkpoint files for safe model interpolation."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Sequence


_CONTROLLED_RUNTIME_FIELDS = {
    "tokenizer.json": ("padding",),
    "tokenizer_config.json": ("padding_side",),
}


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _json_object(value: bytes, *, path: Path) -> dict[str, Any]:
    try:
        parsed = json.loads(value)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"checkpoint copied-file mismatch: {path.name}") from exc
    if not isinstance(parsed, dict):
        raise ValueError(f"checkpoint copied-file mismatch: {path.name}")
    return parsed


def _field_value(value: dict[str, Any], field: str) -> dict[str, Any]:
    return {
        "present": field in value,
        "value": value.get(field),
    }


def compatible_copy_files(
    anchor: Path,
    candidate: Path,
    copy_files: Sequence[str],
) -> tuple[list[str], dict[str, dict[str, Any]]]:
    copied = []
    controlled_differences: dict[str, dict[str, Any]] = {}
    for name in copy_files:
        left = anchor / name
        right = candidate / name
        if left.is_file() != right.is_file():
            raise ValueError(
                f"checkpoint copied-file presence mismatch: {name}"
            )
        if not left.is_file():
            continue
        left_bytes = left.read_bytes()
        right_bytes = right.read_bytes()
        if left_bytes != right_bytes:
            allowed_fields = _CONTROLLED_RUNTIME_FIELDS.get(name)
            if allowed_fields is None:
                raise ValueError(f"checkpoint copied-file mismatch: {name}")
            left_value = _json_object(left_bytes, path=left)
            right_value = _json_object(right_bytes, path=right)
            anchor_values = {
                field: _field_value(left_value, field)
                for field in allowed_fields
            }
            candidate_values = {
                field: _field_value(right_value, field)
                for field in allowed_fields
            }
            changed_fields = [
                field
                for field in allowed_fields
                if anchor_values[field] != candidate_values[field]
            ]
            for field in allowed_fields:
                left_value.pop(field, None)
                right_value.pop(field, None)
            if not changed_fields or left_value != right_value:
                raise ValueError(f"checkpoint copied-file mismatch: {name}")
            semantic_bytes = json.dumps(
                left_value,
                ensure_ascii=False,
                separators=(",", ":"),
                sort_keys=True,
            ).encode("utf-8")
            controlled_differences[name] = {
                "ignored_top_level_fields": sorted(changed_fields),
                "anchor_values": {
                    field: anchor_values[field]
                    for field in changed_fields
                },
                "candidate_values": {
                    field: candidate_values[field]
                    for field in changed_fields
                },
                "anchor_sha256": _sha256_bytes(left_bytes),
                "candidate_sha256": _sha256_bytes(right_bytes),
                "semantic_sha256": _sha256_bytes(semantic_bytes),
                "output_source": "anchor",
            }
        copied.append(name)
    if "config.json" not in copied:
        raise ValueError("checkpoint config.json is missing")
    return sorted(copied), controlled_differences
