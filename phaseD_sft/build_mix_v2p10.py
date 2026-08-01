#!/usr/bin/env python3
"""Build v2.10 from vetted v2.8 rows plus verified teacher revisions."""
from __future__ import annotations

import argparse
from collections.abc import Iterable, Mapping, Sequence
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import sys
import tempfile
from typing import Any


if __package__ in {None, ""}:
    root = Path(__file__).resolve().parents[1]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))


PRODUCTION_V2P8_ROWS = 1_148
PRODUCTION_V2P8_CANONICAL_SHA256 = (
    "7c8f679502927c77914b4f7c4ddbeac65246eb6f443a29ed019b791129462ea1"
)
V2P8_VARIANT = "teacher_train_mix_v2p8"
V2P10_VARIANT = "teacher_train_mix_v2p10"
FABLE_SOURCE = "teacher:claude:claude-fable-5"
FABLE_REVISION_SOURCE = "teacher:claude:claude-fable-5:verified-revision"
FABLE_REVISION_VARIANT = "fable5_recent_verified_revision_v1"
GPT56SOL_DATASET_ID = "greghavens/gpt-5.6-sol-coding-and-debugging-traces"
GPT56SOL_DATASET_NAME = "gpt56sol_verified_v1"
PRODUCTION_GPT56SOL_ROWS = 59
PRODUCTION_GPT56SOL_MANIFEST_SHA256 = (
    "fefc7d83b921a258bc36155e082bda0f4bc9ea5d1c0a2db2b77d645455b07497"
)
PRODUCTION_GPT56SOL_CANONICAL_SHA256 = (
    "fa4b373d3c350363e4ff28c6217bbbf939efca50b8bcf7a5ae40b06f1c218202"
)
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def canonical_rows_sha256(rows: Iterable[Mapping[str, Any]]) -> str:
    """Hash ordered row values independently of Arrow file layout."""

    digest = hashlib.sha256()
    for row in rows:
        payload = _canonical_json(dict(row))
        digest.update(len(payload).to_bytes(8, "big"))
        digest.update(payload)
    return digest.hexdigest()


def _sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"unreadable manifest: {path}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"manifest is not an object: {path}")
    return value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise ValueError(f"unreadable trainer JSONL: {path}") from exc
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"invalid trainer JSONL at {path}:{line_number}"
            ) from exc
        if not isinstance(value, dict):
            raise ValueError(f"non-object trainer row at {path}:{line_number}")
        rows.append(value)
    return rows


def _load_arrow_rows(path: Path) -> list[dict[str, Any]]:
    from datasets import load_from_disk

    try:
        dataset = load_from_disk(str(path))
    except Exception as exc:  # noqa: BLE001
        raise ValueError(f"unreadable HF dataset: {path}") from exc
    return [dict(row) for row in dataset]


def _bound_file_index(path: Path) -> list[dict[str, object]]:
    files = sorted(value for value in path.iterdir() if value.is_file())
    if not files:
        raise ValueError(f"dataset directory has no files: {path}")
    return [
        {
            "path": str(value.resolve()),
            "bytes": value.stat().st_size,
            "sha256": _sha256_path(value),
        }
        for value in files
    ]


def _validate_manifest_dataset(
    path: Path,
    *,
    expected_variant: str | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    manifest_path = path / "manifest.json"
    manifest = _read_json(manifest_path)
    if manifest.get("schema_version") != 2:
        raise ValueError("schema-2 strict admission required")
    if (
        manifest.get("complete") is not True
        or manifest.get("all_training_gates_complete") is not True
    ):
        raise ValueError("schema-2 strict admission required: gates incomplete")
    if expected_variant and manifest.get("dataset_variant") != expected_variant:
        raise ValueError(f"dataset is not registered as {expected_variant}")

    train_path = path / "train.jsonl"
    if (
        not train_path.is_file()
        or manifest.get("train_jsonl_sha256") != _sha256_path(train_path)
    ):
        raise ValueError("schema-2 manifest does not bind trainer JSONL")
    train_rows = _read_jsonl(train_path)
    arrow_rows = _load_arrow_rows(path)
    if canonical_rows_sha256(train_rows) != canonical_rows_sha256(arrow_rows):
        raise ValueError("trainer JSONL and Arrow rows differ")
    row_hash = canonical_rows_sha256(train_rows)
    if manifest.get("canonical_rows_sha256") not in (None, row_hash):
        raise ValueError("manifest canonical row hash mismatch")
    if (
        manifest.get("rendered") != len(train_rows)
        or manifest.get("training_admitted") != len(train_rows)
    ):
        raise ValueError("schema-2 admission count mismatch")
    return train_rows, manifest


def _load_v2p8_base(path: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    manifest_path = path / "manifest.json"
    if manifest_path.is_file():
        rows, manifest = _validate_manifest_dataset(
            path,
            expected_variant=V2P8_VARIANT,
        )
        return rows, manifest

    if path.name != V2P8_VARIANT:
        raise ValueError("base must be the registered v2.8 dataset")
    rows = _load_arrow_rows(path)
    row_hash = canonical_rows_sha256(rows)
    if (
        len(rows) != PRODUCTION_V2P8_ROWS
        or row_hash != PRODUCTION_V2P8_CANONICAL_SHA256
    ):
        raise ValueError("legacy v2.8 base content hash mismatch")
    return rows, {
        "schema_version": 0,
        "dataset_variant": V2P8_VARIANT,
        "rendered": len(rows),
        "canonical_rows_sha256": row_hash,
        "legacy_registered_base": True,
    }


def _load_exclusions(manifest: Mapping[str, Any]) -> tuple[set[str], set[str]]:
    exclusions = manifest.get("exclusions")
    artifacts = (
        exclusions.get("artifacts")
        if isinstance(exclusions, Mapping)
        else None
    )
    if not isinstance(artifacts, list) or not artifacts:
        raise ValueError("strict Fable manifest lacks bound exclusions")
    instance_ids: set[str] = set()
    repositories: set[str] = set()
    for binding in artifacts:
        if not isinstance(binding, Mapping):
            raise ValueError("invalid exclusion artifact binding")
        path = Path(str(binding.get("path") or ""))
        digest = binding.get("sha256")
        if (
            not path.is_file()
            or not isinstance(digest, str)
            or _sha256_path(path) != digest
        ):
            raise ValueError(f"exclusion artifact binding mismatch: {path}")
        text = path.read_text(encoding="utf-8")
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
                raise ValueError("exclusion artifact contains a non-object row")
            for key in ("instance_ids", "selected_ids"):
                listed = value.get(key, [])
                if not isinstance(listed, list) or any(
                    not isinstance(item, str) for item in listed
                ):
                    raise ValueError(f"invalid exclusion field: {key}")
                instance_ids.update(listed)
            for key in ("instance_id", "source_instance_id"):
                if isinstance(value.get(key), str):
                    instance_ids.add(value[key])
            listed_repos = value.get("repo_denylist", [])
            if not isinstance(listed_repos, list) or any(
                not isinstance(item, str) for item in listed_repos
            ):
                raise ValueError("invalid repository exclusions")
            repositories.update(item.casefold() for item in listed_repos)
    return instance_ids, repositories


def _validate_fable_bindings(
    rows: Sequence[Mapping[str, Any]],
    manifest: Mapping[str, Any],
) -> None:
    if manifest.get("success_path_distilled") is not True:
        raise ValueError("strict Fable rows were not success-path distilled")
    if not all(row.get("source") == FABLE_SOURCE for row in rows):
        raise ValueError("exactly one Fable delta is required")

    bindings = manifest.get("artifact_bindings")
    if not isinstance(bindings, list):
        raise ValueError("strict Fable artifact bindings are missing")
    by_id: dict[str, Mapping[str, Any]] = {}
    for binding in bindings:
        if not isinstance(binding, Mapping):
            raise ValueError("invalid strict Fable artifact binding")
        instance_id = binding.get("instance_id")
        if not isinstance(instance_id, str) or instance_id in by_id:
            raise ValueError("duplicate strict Fable artifact identity")
        for key in (
            "stream_sha256",
            "patch_sha256",
            "admission_evidence_sha256",
            "task_contract_sha256",
            "content_sha256",
            "distilled_source_sha256",
        ):
            value = binding.get(key)
            if not isinstance(value, str) or not _SHA256_RE.fullmatch(value):
                raise ValueError(f"invalid strict Fable binding field: {key}")
        commands = binding.get("retained_commands")
        if not isinstance(commands, list) or not commands or any(
            not isinstance(command, str) or not command.strip()
            for command in commands
        ):
            raise ValueError("strict Fable retained command binding is invalid")
        by_id[instance_id] = binding
    row_ids = [row.get("instance_id") for row in rows]
    if any(not isinstance(instance_id, str) for instance_id in row_ids):
        raise ValueError("strict Fable row is missing instance_id")
    if set(row_ids) != set(by_id) or len(row_ids) != len(set(row_ids)):
        raise ValueError("strict Fable rows and artifact bindings differ")

    excluded_ids, excluded_repos = _load_exclusions(manifest)
    for row in rows:
        if (
            row["instance_id"] in excluded_ids
            or str(row.get("repo", "")).casefold() in excluded_repos
        ):
            raise ValueError(
                f"strict Fable row overlaps exclusions: {row['instance_id']}"
            )


def _validate_fable_revision_bindings(
    rows: Sequence[Mapping[str, Any]],
    manifest: Mapping[str, Any],
) -> None:
    if (
        manifest.get("dataset_variant") != FABLE_REVISION_VARIANT
        or manifest.get("source") != FABLE_REVISION_SOURCE
        or manifest.get("verified_revision_supervision") is not True
        or manifest.get("success_path_distilled") is not False
        or manifest.get("legacy_mutations_supervised") != 0
        or manifest.get("legacy_test_runs_supervised") != 0
        or manifest.get("repair_targets_exact_replay") is not True
        or manifest.get("oracle_repair_targets") is not True
    ):
        raise ValueError("verified Fable revision contract is invalid")
    bindings = manifest.get("artifact_bindings")
    if not isinstance(bindings, list) or len(bindings) != len(rows):
        raise ValueError("verified Fable revision bindings are missing")
    by_id: dict[str, Mapping[str, Any]] = {}
    for binding in bindings:
        if not isinstance(binding, Mapping):
            raise ValueError("invalid verified Fable revision binding")
        instance_id = binding.get("instance_id")
        if not isinstance(instance_id, str) or instance_id in by_id:
            raise ValueError("duplicate verified Fable revision identity")
        for key in (
            "content_sha256",
            "legacy_stream_sha256",
            "legacy_patch_sha256",
            "repair_patch_sha256",
            "repair_admission_evidence_sha256",
            "task_contract_sha256",
        ):
            if (
                not isinstance(binding.get(key), str)
                or not _SHA256_RE.fullmatch(str(binding[key]))
            ):
                raise ValueError(
                    f"invalid verified Fable revision binding field: {key}"
                )
        if (
            type(binding.get("supervised_fable_inspection_turns")) is not int
            or type(binding.get("masked_legacy_turns")) is not int
            or binding["supervised_fable_inspection_turns"] < 0
            or binding["masked_legacy_turns"] < 0
        ):
            raise ValueError("invalid verified Fable supervision counts")
        by_id[instance_id] = binding
    row_ids = []
    for row in rows:
        instance_id = row.get("instance_id")
        messages = row.get("messages")
        if (
            not isinstance(instance_id, str)
            or row.get("source") != FABLE_REVISION_SOURCE
            or row.get("legacy_fable_conditioning") is not True
            or row.get("oracle_repair_target") is not True
            or not isinstance(messages, list)
            or row.get("content_sha256")
            != sha256_bytes(_canonical_json(messages))
        ):
            raise ValueError("verified Fable revision row is invalid")
        row_ids.append(instance_id)
    if set(row_ids) != set(by_id) or len(row_ids) != len(set(row_ids)):
        raise ValueError("verified Fable rows and bindings differ")
    _load_exclusions(manifest)


def _load_gpt56sol_delta(
    path: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    manifest_path = path / "manifest.json"
    if (
        path.name != GPT56SOL_DATASET_NAME
        or not manifest_path.is_file()
        or _sha256_path(manifest_path)
        != PRODUCTION_GPT56SOL_MANIFEST_SHA256
    ):
        raise ValueError("unregistered schema-1 teacher delta")
    manifest = _read_json(manifest_path)
    source = manifest.get("source")
    replay = manifest.get("replay_results_artifact")
    gates = manifest.get("gates")
    format_gate = manifest.get("standard_native_format_loss_gate")
    if (
        manifest.get("schema_version") != 1
        or not isinstance(source, Mapping)
        or source.get("dataset_id") != GPT56SOL_DATASET_ID
        or not isinstance(source.get("dataset_revision"), str)
        or manifest.get("functionally_verified_input")
        != PRODUCTION_GPT56SOL_ROWS
        or manifest.get("output_rows") != PRODUCTION_GPT56SOL_ROWS
        or manifest.get("training_admitted") != PRODUCTION_GPT56SOL_ROWS
        or manifest.get("all_training_gates_complete") is not True
        or not isinstance(
            manifest.get("replay_manifest_sha256"),
            str,
        )
        or not _SHA256_RE.fullmatch(manifest["replay_manifest_sha256"])
        or not isinstance(replay, Mapping)
        or not isinstance(replay.get("sha256"), str)
        or not _SHA256_RE.fullmatch(replay["sha256"])
        or not isinstance(gates, Mapping)
        or gates.get("input_rows") != PRODUCTION_GPT56SOL_ROWS
        or gates.get("output_rows") != PRODUCTION_GPT56SOL_ROWS
        or any(
            gates.get(key) != 0
            for key in (
                "behavior_rejected",
                "decontamination_rejected",
                "dedup_removed",
                "format_failures",
                "token_rejected",
            )
        )
        or not isinstance(format_gate, Mapping)
        or format_gate.get("status") != "passed"
        or format_gate.get("samples") != PRODUCTION_GPT56SOL_ROWS
        or format_gate.get("failure_count") != 0
    ):
        raise ValueError("registered GPT-5.6 Sol admission manifest is invalid")

    train_path = path / "train.jsonl"
    if (
        not train_path.is_file()
        or manifest.get("train_jsonl_sha256") != _sha256_path(train_path)
    ):
        raise ValueError("registered GPT-5.6 Sol trainer JSONL is unbound")
    train_rows = _read_jsonl(train_path)
    arrow_rows = _load_arrow_rows(path)
    normalized_arrow_rows: list[dict[str, Any]] = []
    for raw_row in arrow_rows:
        row = dict(raw_row)
        messages = row.get("messages")
        if isinstance(messages, list):
            normalized_messages = []
            for raw_message in messages:
                message = dict(raw_message)
                if message.get("loss") is None:
                    message.pop("loss", None)
                normalized_messages.append(message)
            row["messages"] = normalized_messages
        normalized_arrow_rows.append(row)
    if canonical_rows_sha256(train_rows) != canonical_rows_sha256(
        normalized_arrow_rows
    ):
        raise ValueError("registered GPT-5.6 Sol Arrow and JSONL rows differ")
    if (
        len(train_rows) != PRODUCTION_GPT56SOL_ROWS
        or canonical_rows_sha256(train_rows)
        != PRODUCTION_GPT56SOL_CANONICAL_SHA256
    ):
        raise ValueError("registered GPT-5.6 Sol row identity mismatch")

    artifact = format_gate.get("artifact")
    if not isinstance(artifact, Mapping):
        raise ValueError("registered GPT-5.6 Sol format artifact is missing")
    artifact_name = artifact.get("path")
    if (
        not isinstance(artifact_name, str)
        or Path(artifact_name).name != artifact_name
    ):
        raise ValueError("registered GPT-5.6 Sol format artifact path is unsafe")
    artifact_path = path / artifact_name
    if (
        not artifact_path.is_file()
        or artifact.get("bytes") != artifact_path.stat().st_size
        or artifact.get("sha256") != _sha256_path(artifact_path)
    ):
        raise ValueError("registered GPT-5.6 Sol format artifact is unbound")

    revision = source["dataset_revision"]
    prefix = f"teacher:gpt56sol:{revision}:"
    for row in train_rows:
        messages = row.get("messages")
        if (
            not isinstance(row.get("instance_id"), str)
            or not isinstance(messages, list)
            or not isinstance(row.get("source"), str)
            or not row["source"].startswith(prefix)
            or row.get("content_sha256")
            != sha256_bytes(_canonical_json(messages))
        ):
            raise ValueError("registered GPT-5.6 Sol row content is unbound")
        for key in (
            "source_terminal_sha256",
            "replay_trajectory_id",
            "replay_operation_sha256",
        ):
            if (
                not isinstance(row.get(key), str)
                or not _SHA256_RE.fullmatch(row[key])
            ):
                raise ValueError(
                    f"registered GPT-5.6 Sol row lacks {key}"
                )
    _load_exclusions(manifest)
    return train_rows, manifest


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _validate_teacher_row_count(
    rows: Sequence[Mapping[str, Any]],
    *,
    minimum_teacher_rows: int,
) -> None:
    if (
        type(minimum_teacher_rows) is not int
        or not 1 <= minimum_teacher_rows <= 30
    ):
        raise ValueError("minimum_teacher_rows must be an integer in [1,30]")
    if not minimum_teacher_rows <= len(rows) <= 100:
        raise ValueError(
            "teacher delta must contain "
            f"{minimum_teacher_rows}-100 rows"
        )


def _training_projection(row: Mapping[str, Any]) -> dict[str, Any]:
    instance_id = row.get("instance_id")
    messages = row.get("messages")
    repo = row.get("repo")
    source = row.get("source")
    if (
        not isinstance(instance_id, str)
        or not instance_id
        or not isinstance(messages, list)
        or not isinstance(repo, str)
        or not isinstance(source, str)
    ):
        raise ValueError(f"invalid training row: {instance_id!r}")
    normalized_messages = []
    for raw_message in messages:
        if not isinstance(raw_message, Mapping):
            raise ValueError(
                f"invalid training message in row: {instance_id!r}"
            )
        message = dict(raw_message)
        loss = message.get("loss")
        if loss is None:
            message["loss"] = message.get("role") == "assistant"
        elif type(loss) is not bool:
            raise ValueError(
                f"invalid training loss flag in row: {instance_id!r}"
            )
        normalized_messages.append(message)
    return {
        "instance_id": instance_id,
        "messages": normalized_messages,
        "repo": repo,
        "source": source,
    }


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


def build_mix(
    base: Path,
    teacher_delta: Path,
    out: Path,
    extra: tuple[Path, ...] = (),
    *,
    minimum_teacher_rows: int = 30,
) -> dict[str, object]:
    """Build v2.10 from registered v2.8 plus approved teacher deltas."""

    base = Path(base).resolve()
    teacher_delta = Path(teacher_delta).resolve()
    out = Path(out).resolve()
    if len(extra) > 1:
        raise ValueError("at most two approved teacher deltas are supported")
    if os.path.lexists(out):
        raise FileExistsError(f"refusing to overwrite {out}")

    base_rows_input, base_manifest = _load_v2p8_base(base)
    quarantined_base_rows = [
        row for row in base_rows_input if row.get("source") == FABLE_SOURCE
    ]
    base_rows = [
        row for row in base_rows_input if row.get("source") != FABLE_SOURCE
    ]

    delta_records: list[dict[str, Any]] = []
    for delta_path in (teacher_delta, *extra):
        delta_path = Path(delta_path).resolve()
        manifest = _read_json(delta_path / "manifest.json")
        if manifest.get("schema_version") == 2:
            rows, manifest = _validate_manifest_dataset(delta_path)
            if manifest.get("dataset_variant") == FABLE_REVISION_VARIANT:
                _validate_fable_revision_bindings(rows, manifest)
                kind = "fable_revision"
                bindings = manifest["artifact_bindings"]
            else:
                _validate_fable_bindings(rows, manifest)
                kind = "strict_fable"
                bindings = manifest["artifact_bindings"]
        elif manifest.get("schema_version") == 1:
            rows, manifest = _load_gpt56sol_delta(delta_path)
            kind = "gpt56sol"
            bindings = [
                {
                    key: row[key]
                    for key in (
                        "instance_id",
                        "content_sha256",
                        "source_terminal_sha256",
                        "replay_trajectory_id",
                        "replay_operation_sha256",
                    )
                }
                for row in rows
            ]
        else:
            raise ValueError("teacher delta admission schema is unsupported")
        delta_records.append({
            "kind": kind,
            "path": delta_path,
            "rows": rows,
            "manifest": manifest,
            "bindings": bindings,
        })
    kinds = [record["kind"] for record in delta_records]
    if len(set(kinds)) != len(kinds):
        raise ValueError("duplicate teacher delta kind")
    allowed = (
        len(kinds) == 1
        or set(kinds) == {"fable_revision", "gpt56sol"}
    )
    if not allowed:
        raise ValueError("teacher delta combination is not approved")
    teacher_rows = [
        row
        for record in delta_records
        for row in record["rows"]
    ]
    if kinds == ["strict_fable"]:
        teacher_delta_policy = "strict_fable_schema2"
    elif kinds == ["gpt56sol"]:
        teacher_delta_policy = "gpt56sol_verified_quota_fallback"
    elif kinds == ["fable_revision"]:
        teacher_delta_policy = "verified_fable_revision"
    else:
        teacher_delta_policy = "verified_fable_revision_plus_gpt56sol"
    success_path_distilled = kinds == ["strict_fable"]
    functionally_verified_native_replay = all(
        kind in {"fable_revision", "gpt56sol"} for kind in kinds
    )
    teacher_bindings = [
        {"delta_kind": record["kind"], **binding}
        for record in delta_records
        for binding in record["bindings"]
    ]
    _validate_teacher_row_count(
        teacher_rows,
        minimum_teacher_rows=minimum_teacher_rows,
    )

    projected_base = [_training_projection(row) for row in base_rows]
    projected_teacher = [_training_projection(row) for row in teacher_rows]
    base_ids = [row["instance_id"] for row in projected_base]
    teacher_ids = [row["instance_id"] for row in projected_teacher]
    if (
        any(not isinstance(instance_id, str) for instance_id in base_ids)
        or len(base_ids) != len(set(base_ids))
        or set(base_ids) & set(teacher_ids)
    ):
        raise ValueError("duplicate instance IDs in v2.10 mix")
    combined = [*projected_base, *projected_teacher]
    manifest_holder: dict[str, object] = {}

    def publish(stage: Path) -> None:
        from datasets import Dataset

        Dataset.from_list(combined).save_to_disk(str(stage))
        train_path = stage / "train.jsonl"
        with train_path.open("w", encoding="utf-8") as handle:
            for row in combined:
                handle.write(_canonical_json(row).decode("utf-8") + "\n")
        manifest: dict[str, object] = {
            "schema_version": 2,
            "complete": True,
            "dataset_variant": V2P10_VARIANT,
            "base_variant": V2P8_VARIANT,
            "base_rows_input": len(base_rows_input),
            "base_rows_quarantined": len(quarantined_base_rows),
            "base_quarantined_instance_ids": [
                row["instance_id"] for row in quarantined_base_rows
            ],
            "base_rows": len(base_rows),
            "teacher_rows": len(teacher_rows),
            "fable_revision_rows": sum(
                len(record["rows"])
                for record in delta_records
                if record["kind"] == "fable_revision"
            ),
            "gpt56sol_rows": sum(
                len(record["rows"])
                for record in delta_records
                if record["kind"] == "gpt56sol"
            ),
            "minimum_teacher_rows": minimum_teacher_rows,
            "teacher_row_policy": (
                "standard_30_to_100"
                if minimum_teacher_rows == 30
                else "explicit_exhausted_collection_override"
            ),
            "teacher_delta_policy": teacher_delta_policy,
            "rendered": len(combined),
            "training_admitted": 0,
            "all_training_gates_complete": False,
            "canonical_rows_sha256": canonical_rows_sha256(combined),
            "train_jsonl_sha256": _sha256_path(train_path),
            "base": {
                "path": str(base),
                "manifest": base_manifest,
                "files": _bound_file_index(base),
            },
            "teacher": {
                "path": str(delta_records[0]["path"]),
                "manifest": delta_records[0]["manifest"],
                "manifest_sha256": _sha256_path(
                    delta_records[0]["path"] / "manifest.json"
                ),
                "train_jsonl_sha256": _sha256_path(
                    delta_records[0]["path"] / "train.jsonl"
                ),
                "canonical_rows_sha256": canonical_rows_sha256(
                    delta_records[0]["rows"]
                ),
                "files": _bound_file_index(delta_records[0]["path"]),
            },
            "teacher_sources": [
                {
                    "kind": record["kind"],
                    "path": str(record["path"]),
                    "manifest": record["manifest"],
                    "manifest_sha256": _sha256_path(
                        record["path"] / "manifest.json"
                    ),
                    "train_jsonl_sha256": _sha256_path(
                        record["path"] / "train.jsonl"
                    ),
                    "canonical_rows_sha256": canonical_rows_sha256(
                        record["rows"]
                    ),
                    "files": _bound_file_index(record["path"]),
                }
                for record in delta_records
            ],
            "delta_instance_ids": teacher_ids,
            "artifact_bindings": teacher_bindings,
            "exclusions": delta_records[0]["manifest"]["exclusions"],
            "success_path_distilled": success_path_distilled,
            "verified_revision_supervision": (
                "fable_revision" in kinds
            ),
            "functionally_verified_native_replay": (
                functionally_verified_native_replay
            ),
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
    parser.add_argument(
        "--teacher-delta",
        "--fable",
        dest="teacher_delta",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--extra-teacher-delta",
        dest="extra_teacher_delta",
        action="append",
        default=[],
        type=Path,
        help=(
            "Optional additional verified teacher dataset. May be supplied "
            "once; the only admitted two-source policy is recent verified "
            "Fable revision plus pinned GPT-5.6 Sol."
        ),
    )
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--minimum-teacher-rows",
        "--minimum-fable-rows",
        dest="minimum_teacher_rows",
        type=int,
        default=30,
        help=(
            "Explicit exhausted-collection floor in [1,30]; defaults to the "
            "standard 30-row gate."
        ),
    )
    args = parser.parse_args()
    manifest = build_mix(
        args.base,
        args.teacher_delta,
        args.out,
        extra=tuple(args.extra_teacher_delta),
        minimum_teacher_rows=args.minimum_teacher_rows,
    )
    print(json.dumps({
        "base_rows": manifest["base_rows"],
        "teacher_rows": manifest["teacher_rows"],
        "rendered": manifest["rendered"],
        "all_training_gates_complete": manifest[
            "all_training_gates_complete"
        ],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
