#!/usr/bin/env python3
"""Filter an HF trajectory dataset to strict, edit-decisive unique rows."""
from __future__ import annotations

import argparse
from collections import Counter
import ctypes
from dataclasses import dataclass
import errno
import hashlib
import json
import os
from pathlib import Path
import shutil
import stat
import sys
from typing import Any, Callable, Sequence
import uuid

from phaseD_sft.agentic_trace_filters import command_trace_quality_report


DatasetWriter = Callable[[list[dict[str, Any]], Path], None]
ArtifactPublisher = Callable[[Path, Path], None]
PathIdentity = tuple[int, int, int]


@dataclass(frozen=True)
class _OwnershipPin:
    descriptor: int
    relative_path: Path | None = None


@dataclass(frozen=True)
class _OwnedArtifact:
    path: Path
    pins: tuple[_OwnershipPin, ...]


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return parsed


def _nonnegative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be nonnegative")
    return parsed


def _validate_filter_limits(max_first_edit_index: int, max_read_streak: int) -> None:
    if type(max_first_edit_index) is not int or max_first_edit_index <= 0:
        raise ValueError("max_first_edit_index must be positive")
    if type(max_read_streak) is not int or max_read_streak < 0:
        raise ValueError("max_read_streak must be nonnegative")


def _lexists(path: Path) -> bool:
    return os.path.lexists(os.fspath(path))


def _absolute_path(path: Path) -> Path:
    return Path(os.path.abspath(os.fspath(path)))


def _paths_overlap(left: Path, right: Path) -> bool:
    return left == right or left in right.parents or right in left.parents


def _validate_artifact_topology(source: Path, output: Path, manifest_path: Path) -> None:
    paths = (source, output, manifest_path)
    representations = (
        tuple(_absolute_path(path) for path in paths),
        tuple(path.resolve(strict=False) for path in paths),
    )
    for represented in representations:
        for index, left in enumerate(represented):
            for right in represented[index + 1 :]:
                if _paths_overlap(left, right):
                    raise ValueError(
                        "source, output, and manifest paths must be distinct and non-nested"
                    )


def _rename_noreplace(source: Path, destination: Path) -> None:
    """Atomically rename ``source`` while refusing any destination collision."""
    libc = ctypes.CDLL(None, use_errno=True)
    encoded_source = os.fsencode(source)
    encoded_destination = os.fsencode(destination)
    ctypes.set_errno(0)
    if sys.platform == "darwin":
        rename = libc.renamex_np
        rename.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_uint]
        rename.restype = ctypes.c_int
        result = rename(encoded_source, encoded_destination, 0x00000004)
    elif sys.platform.startswith("linux") and hasattr(libc, "renameat2"):
        rename = libc.renameat2
        rename.argtypes = [
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_uint,
        ]
        rename.restype = ctypes.c_int
        result = rename(-100, encoded_source, -100, encoded_destination, 0x00000001)
    else:
        raise RuntimeError("atomic no-clobber rename is unsupported on this platform")
    if result == 0:
        return
    error_number = ctypes.get_errno()
    if error_number in (errno.EEXIST, errno.ENOTEMPTY):
        raise FileExistsError(
            error_number,
            os.strerror(error_number),
            os.fspath(destination),
        )
    raise OSError(error_number, os.strerror(error_number), os.fspath(destination))


def _path_status(path: Path) -> os.stat_result | None:
    try:
        return os.lstat(path)
    except FileNotFoundError:
        return None


def _path_identity(path: Path) -> PathIdentity | None:
    status = _path_status(path)
    if status is None:
        return None
    return status.st_dev, status.st_ino, stat.S_IFMT(status.st_mode)


def _remove_path(path: Path) -> None:
    identity = _path_identity(path)
    if identity is None:
        return
    if stat.S_ISDIR(identity[2]):
        shutil.rmtree(path)
    else:
        path.unlink()


def _descriptor_names_path(descriptor: int, path: Path) -> bool:
    """Return whether ``path`` still names the inode pinned by ``descriptor``."""
    try:
        held = os.fstat(descriptor)
        current = _path_status(path)
    except OSError:
        return False
    if current is None:
        return False
    return (
        held.st_nlink > 0
        and held.st_dev == current.st_dev
        and held.st_ino == current.st_ino
        and stat.S_IFMT(held.st_mode) == stat.S_IFMT(current.st_mode)
    )


def _artifact_names_root(artifact: _OwnedArtifact, root: Path) -> bool:
    return all(
        _descriptor_names_path(
            pin.descriptor,
            root if pin.relative_path is None else root / pin.relative_path,
        )
        for pin in artifact.pins
    )


def _open_pin(path: Path, *, directory: bool) -> int:
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    if directory:
        flags |= getattr(os, "O_DIRECTORY", 0)
    descriptor = os.open(path, flags)
    try:
        status = os.fstat(descriptor)
        expected_type = stat.S_IFDIR if directory else stat.S_IFREG
        if stat.S_IFMT(status.st_mode) != expected_type:
            raise ValueError(
                f"staged artifact is not a {'directory' if directory else 'regular file'}"
            )
        if not _descriptor_names_path(descriptor, path):
            raise RuntimeError("staged artifact changed while its ownership handle opened")
        return descriptor
    except BaseException:
        os.close(descriptor)
        raise


def _open_owned_artifact(staged: Path, final: Path, *, directory: bool) -> _OwnedArtifact:
    pins: list[_OwnershipPin] = []
    try:
        pins.append(_OwnershipPin(_open_pin(staged, directory=directory)))
        if directory:
            relative_anchor = Path("content.sha256")
            pins.append(
                _OwnershipPin(
                    _open_pin(staged / relative_anchor, directory=False),
                    relative_anchor,
                )
            )
        return _OwnedArtifact(path=final, pins=tuple(pins))
    except BaseException:
        for pin in pins:
            os.close(pin.descriptor)
        raise


def _remove_if_owned(artifact: _OwnedArtifact) -> None:
    if _artifact_names_root(artifact, artifact.path):
        _remove_path(artifact.path)


def _publish_owned(
    staged: Path,
    final: Path,
    publisher: ArtifactPublisher,
    artifact: _OwnedArtifact,
    owned: list[_OwnedArtifact],
) -> None:
    if artifact.path != final or not _artifact_names_root(artifact, staged):
        raise RuntimeError("ownership handle does not match the staged artifact")
    try:
        publisher(staged, final)
    except BaseException:
        if not _lexists(staged) and _artifact_names_root(artifact, final):
            owned.append(artifact)
        raise
    if _lexists(staged) or not _artifact_names_root(artifact, final):
        raise RuntimeError("publisher did not atomically move the staged artifact")
    owned.append(artifact)


def _canonical_json_bytes(value: Any) -> bytes:
    try:
        serialized = json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
    except (TypeError, ValueError) as error:
        raise ValueError("content must contain only canonical JSON values") from error
    return serialized.encode("utf-8")


def canonical_content_sha256(messages: Any) -> str:
    """Return a stable SHA-256 over canonical trajectory message content."""
    if not isinstance(messages, list):
        raise ValueError("messages must be a list")
    return hashlib.sha256(_canonical_json_bytes(messages)).hexdigest()


def filter_rows(
    rows: list[dict[str, Any]],
    max_first_edit_index: int = 10,
    max_read_streak: int = 5,
    reject_identical_consecutive_commands: bool = True,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Keep the first unique row that passes the strict command-level filter."""
    _validate_filter_limits(max_first_edit_index, max_read_streak)
    kept: list[dict[str, Any]] = []
    seen_content: set[str] = set()
    drop_reasons: Counter[str] = Counter()
    quality_dropped = 0
    content_duplicates = 0

    for row in rows:
        messages = row.get("messages")
        report = command_trace_quality_report(
            messages if isinstance(messages, list) else [],
            max_first_edit_index=max_first_edit_index,
            max_read_streak=max_read_streak,
            reject_identical_consecutive_commands=reject_identical_consecutive_commands,
        )
        if not report.keep:
            quality_dropped += 1
            drop_reasons.update(report.reasons)
            continue

        content_hash = canonical_content_sha256(messages)
        if content_hash in seen_content:
            content_duplicates += 1
            drop_reasons["duplicate_content"] += 1
            continue
        seen_content.add(content_hash)
        kept.append(row)

    manifest = {
        "schema_version": 1,
        "rows_in": len(rows),
        "rows_kept": len(kept),
        "rows_dropped": len(rows) - len(kept),
        "quality_dropped": quality_dropped,
        "content_duplicates": content_duplicates,
        "unique_content_hashes": len(seen_content),
        "filter": {
            "max_first_edit_index": max_first_edit_index,
            "max_read_streak": max_read_streak,
            "reject_identical_consecutive_commands": reject_identical_consecutive_commands,
        },
        "drop_reasons": dict(sorted(drop_reasons.items())),
    }
    return kept, manifest


def _canonical_rows_bytes(rows: list[dict[str, Any]]) -> bytes:
    return b"".join(_canonical_json_bytes(row) + b"\n" for row in rows)


def filter_hf_dataset(
    source: Path,
    output: Path,
    manifest_path: Path,
    *,
    max_first_edit_index: int = 10,
    max_read_streak: int = 5,
    reject_identical_consecutive_commands: bool = True,
    dataset_writer: DatasetWriter | None = None,
) -> dict[str, Any]:
    """Filter one HF directory and atomically publish its dataset and manifest."""
    source = Path(source)
    output = Path(output)
    manifest_path = Path(manifest_path)
    _validate_filter_limits(max_first_edit_index, max_read_streak)
    _validate_artifact_topology(source, output, manifest_path)
    existing = [path for path in (output, manifest_path) if _lexists(path)]
    if existing:
        raise FileExistsError(f"refusing to overwrite build artifact: {existing[0]}")

    from datasets import load_from_disk

    rows = [dict(row) for row in load_from_disk(str(source))]
    kept, manifest = filter_rows(
        rows,
        max_first_edit_index=max_first_edit_index,
        max_read_streak=max_read_streak,
        reject_identical_consecutive_commands=reject_identical_consecutive_commands,
    )
    if not kept:
        raise ValueError("filter produced no rows")
    expected_rows_bytes = _canonical_rows_bytes(kept)
    expected_content_sha256 = hashlib.sha256(expected_rows_bytes).hexdigest()
    output.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    if dataset_writer is None:
        from datasets import Dataset

        def dataset_writer(values: list[dict[str, Any]], path: Path) -> None:
            Dataset.from_list(values).save_to_disk(str(path))

    token = uuid.uuid4().hex
    staged_output = output.parent / f".{output.name}.{token}.tmp"
    staged_manifest = manifest_path.parent / f".{manifest_path.name}.{token}.tmp"
    if _lexists(staged_output) or _lexists(staged_manifest):
        raise FileExistsError("staging path collision")
    owned: list[_OwnedArtifact] = []
    opened: list[_OwnedArtifact] = []
    committed = False
    try:
        dataset_writer(kept, staged_output)
        staged_output_identity = _path_identity(staged_output)
        if staged_output_identity is None or not stat.S_ISDIR(staged_output_identity[2]):
            raise ValueError("dataset_writer did not create an HF dataset directory")

        try:
            stored_rows = [dict(row) for row in load_from_disk(str(staged_output))]
        except Exception as error:
            raise ValueError("staged HF dataset cannot be reloaded") from error

        stored_rows_bytes = _canonical_rows_bytes(stored_rows)
        stored_content_sha256 = hashlib.sha256(stored_rows_bytes).hexdigest()
        if (
            len(stored_rows) != len(kept)
            or stored_rows_bytes != expected_rows_bytes
            or stored_content_sha256 != expected_content_sha256
        ):
            raise ValueError("staged HF dataset differs from filtered rows")
        with (staged_output / "content.sha256").open("x", encoding="utf-8") as handle:
            handle.write(expected_content_sha256 + "\n")
        stored_manifest = {
            **manifest,
            "source": str(source),
            "output": str(output),
            "dataset_content_sha256": expected_content_sha256,
        }
        manifest_bytes = (
            json.dumps(stored_manifest, indent=2, sort_keys=True) + "\n"
        ).encode("utf-8")
        with staged_manifest.open("x", encoding="utf-8") as handle:
            handle.write(manifest_bytes.decode("utf-8"))

        dataset_artifact = _open_owned_artifact(
            staged_output,
            output,
            directory=True,
        )
        opened.append(dataset_artifact)
        manifest_artifact = _open_owned_artifact(
            staged_manifest,
            manifest_path,
            directory=False,
        )
        opened.append(manifest_artifact)
        _publish_owned(
            staged_output,
            output,
            _rename_noreplace,
            dataset_artifact,
            owned,
        )
        _publish_owned(
            staged_manifest,
            manifest_path,
            _rename_noreplace,
            manifest_artifact,
            owned,
        )
        committed = True
        return stored_manifest
    except BaseException:
        for artifact in reversed(owned):
            _remove_if_owned(artifact)
        raise
    finally:
        for artifact in opened:
            for pin in artifact.pins:
                try:
                    os.close(pin.descriptor)
                except OSError:
                    pass
        if not committed:
            _remove_path(staged_output)
            _remove_path(staged_manifest)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--in", dest="source", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--max-first-edit-index", type=_positive_int, default=10)
    parser.add_argument("--max-read-streak", type=_nonnegative_int, default=5)
    parser.add_argument(
        "--allow-identical-consecutive-commands",
        dest="reject_identical_consecutive_commands",
        action="store_false",
    )
    parser.set_defaults(reject_identical_consecutive_commands=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    manifest = filter_hf_dataset(
        args.source,
        args.out,
        args.manifest,
        max_first_edit_index=args.max_first_edit_index,
        max_read_streak=args.max_read_streak,
        reject_identical_consecutive_commands=args.reject_identical_consecutive_commands,
    )
    print(json.dumps(manifest, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
