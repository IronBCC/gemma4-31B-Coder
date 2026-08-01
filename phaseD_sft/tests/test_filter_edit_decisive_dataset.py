from __future__ import annotations

import copy
import json
import os
from pathlib import Path
import shutil
import sys
from typing import Any

import pytest

from phaseD_sft import filter_edit_decisive_dataset as filter_module
from phaseD_sft.filter_edit_decisive_dataset import (
    canonical_content_sha256,
    filter_hf_dataset,
    filter_rows,
    parse_args,
)

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "teacher_platform"))
from teacher_platform import translate_trajectory_tool_call  # noqa: E402


def _command(command: str) -> dict[str, Any]:
    return {
        "role": "assistant",
        "content": "",
        "tool_calls": [
            {
                "type": "function",
                "function": {
                    "name": "bash",
                    "arguments": json.dumps({"command": command}),
                },
            }
        ],
    }


def _row(instance_id: str, commands: list[str]) -> dict[str, Any]:
    return {
        "instance_id": instance_id,
        "source": "open_swe_traces_qwen35",
        "messages": [_command(command) for command in commands],
    }


def _tree_snapshot(root: Path) -> list[tuple[str, bytes]]:
    return [
        (str(path.relative_to(root)), path.read_bytes())
        for path in sorted(root.rglob("*"))
        if path.is_file()
    ]


def _translated_editor_command(command: str, **arguments: Any) -> str:
    translated = translate_trajectory_tool_call(
        {
            "function": {
                "name": "str_replace_editor",
                "arguments": json.dumps({"command": command, **arguments}),
            }
        }
    )
    assert isinstance(translated, str)
    return translated


def test_filter_rows_keeps_only_edit_decisive_traces_and_counts_exact_reasons() -> None:
    rows = [
        _row("keep", ["rg target src", "sed -i 's/old/new/' src/mod.py"]),
        _row(
            "late-edit",
            [*[f"pytest -q tests/test_{index}.py" for index in range(10)], "apply_patch <<'PATCH'\nPATCH"],
        ),
        _row(
            "read-loop-repeat",
            [
                "rg target src",
                "rg target src",
                *[f"cat file_{index}.py" for index in range(4)],
                "sed -i 's/old/new/' src/mod.py",
            ],
        ),
        _row("no-edit", ["pytest -q"]),
    ]

    kept, manifest = filter_rows(rows)

    assert [row["instance_id"] for row in kept] == ["keep"]
    assert manifest == {
        "schema_version": 1,
        "rows_in": 4,
        "rows_kept": 1,
        "rows_dropped": 3,
        "quality_dropped": 3,
        "content_duplicates": 0,
        "unique_content_hashes": 1,
        "filter": {
            "max_first_edit_index": 10,
            "max_read_streak": 5,
            "reject_identical_consecutive_commands": True,
        },
        "drop_reasons": {
            "first_edit_after_limit": 1,
            "identical_consecutive_command": 1,
            "no_edit": 1,
            "read_streak_exceeded": 1,
        },
    }


def test_filter_rows_deduplicates_canonical_message_content_and_keeps_first() -> None:
    first = _row("first", ["sed -i 's/old/new/' src/mod.py"])
    duplicate = copy.deepcopy(first)
    duplicate["instance_id"] = "duplicate-with-different-metadata"
    duplicate["extra"] = "metadata must not affect content deduplication"
    function = duplicate["messages"][0]["tool_calls"][0]["function"]
    duplicate["messages"][0]["tool_calls"][0]["function"] = {
        "arguments": function["arguments"],
        "name": function["name"],
    }

    kept, manifest = filter_rows([first, duplicate])

    assert kept == [first]
    assert canonical_content_sha256(first["messages"]) == canonical_content_sha256(
        duplicate["messages"]
    )
    assert manifest["rows_kept"] == 1
    assert manifest["content_duplicates"] == 1
    assert manifest["drop_reasons"] == {"duplicate_content": 1}


def test_filter_rows_counts_real_translated_views_toward_read_streak() -> None:
    views = [
        _translated_editor_command("view", path=f"/testbed/file_{index}.py")
        for index in range(6)
    ]
    edit = _translated_editor_command(
        "str_replace",
        path="/testbed/target.py",
        old_str="old",
        new_str="new",
    )
    rows = [
        _row("five-views-then-edit", [*views[:5], edit]),
        _row("six-views-then-edit", [*views, edit]),
    ]

    kept, manifest = filter_rows(rows)

    assert [row["instance_id"] for row in kept] == ["five-views-then-edit"]
    assert manifest["quality_dropped"] == 1
    assert manifest["drop_reasons"] == {"read_streak_exceeded": 1}


def test_filter_hf_dataset_round_trips_directory_and_publishes_manifest_last(
    tmp_path: Path,
) -> None:
    from datasets import Dataset, load_from_disk

    source = tmp_path / "source"
    output = tmp_path / "filtered"
    manifest_path = tmp_path / "filtered.manifest.json"
    rows = [
        _row("keep", ["sed -i 's/old/new/' src/mod.py"]),
        _row("drop", ["pytest -q"]),
    ]
    Dataset.from_list(rows).save_to_disk(str(source))

    manifest = filter_hf_dataset(source, output, manifest_path)

    assert [row["instance_id"] for row in load_from_disk(str(output))] == ["keep"]
    assert json.loads(manifest_path.read_text(encoding="utf-8")) == manifest
    assert manifest["source"] == str(source)
    assert manifest["output"] == str(output)
    assert manifest["rows_kept"] == 1
    assert "publication_id" not in manifest
    assert not (output / ".filter-edit-decisive-owner").exists()
    assert (output / "content.sha256").read_text(encoding="utf-8").strip() == manifest[
        "dataset_content_sha256"
    ]
    assert not list(tmp_path.glob(".*.tmp"))


def test_filter_hf_dataset_rebuild_is_byte_reproducible(tmp_path: Path) -> None:
    from datasets import Dataset

    source = tmp_path / "source"
    output = tmp_path / "filtered"
    manifest_path = tmp_path / "filtered.manifest.json"
    Dataset.from_list([
        _row("keep", ["sed -i 's/old/new/' src/mod.py"]),
        _row("drop", ["pytest -q"]),
    ]).save_to_disk(str(source))

    first_manifest = filter_hf_dataset(source, output, manifest_path)
    first_manifest_bytes = manifest_path.read_bytes()
    first_dataset_bytes = _tree_snapshot(output)
    shutil.rmtree(output)
    manifest_path.unlink()

    second_manifest = filter_hf_dataset(source, output, manifest_path)

    assert second_manifest == first_manifest
    assert manifest_path.read_bytes() == first_manifest_bytes
    assert _tree_snapshot(output) == first_dataset_bytes
    assert "publication_id" not in second_manifest
    assert not (output / ".filter-edit-decisive-owner").exists()


def test_manifest_publication_runs_no_post_commit_marker_cleanup(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from datasets import Dataset

    source = tmp_path / "source"
    output = tmp_path / "filtered"
    manifest_path = tmp_path / "filtered.manifest.json"
    Dataset.from_list([_row("keep", ["sed -i 's/old/new/' src/mod.py"])]).save_to_disk(
        str(source)
    )

    def forbidden_post_commit_cleanup(*_: object) -> None:
        raise RuntimeError("post-commit dataset mutation ran")

    monkeypatch.setattr(
        filter_module,
        "_remove_final_owner_marker",
        forbidden_post_commit_cleanup,
        raising=False,
    )

    manifest = filter_hf_dataset(source, output, manifest_path)

    assert output.is_dir()
    assert json.loads(manifest_path.read_text(encoding="utf-8")) == manifest
    assert not (output / ".filter-edit-decisive-owner").exists()


def test_filter_hf_dataset_failure_leaves_no_output_or_manifest(tmp_path: Path) -> None:
    from datasets import Dataset

    source = tmp_path / "source"
    output = tmp_path / "filtered"
    manifest_path = tmp_path / "filtered.manifest.json"
    Dataset.from_list([_row("keep", ["sed -i 's/old/new/' src/mod.py"])]).save_to_disk(
        str(source)
    )

    def failing_writer(_: list[dict[str, Any]], staged_output: Path) -> None:
        assert staged_output.parent == output.parent
        assert staged_output.name.startswith(f".{output.name}.")
        assert not output.exists()
        assert not manifest_path.exists()
        staged_output.mkdir()
        raise RuntimeError("synthetic dataset writer failure")

    with pytest.raises(RuntimeError, match="synthetic dataset writer failure"):
        filter_hf_dataset(
            source,
            output,
            manifest_path,
            dataset_writer=failing_writer,
        )

    assert not output.exists()
    assert not manifest_path.exists()
    assert not list(tmp_path.glob(".*.tmp"))


@pytest.mark.parametrize("corruption", ["drop", "reorder", "mutate"])
def test_filter_hf_dataset_rejects_readable_changed_writer_output(
    tmp_path: Path,
    corruption: str,
) -> None:
    from datasets import Dataset

    source = tmp_path / "source"
    output = tmp_path / "filtered"
    manifest_path = tmp_path / "filtered.manifest.json"
    rows = [
        _row("first", ["sed -i 's/old/new/' src/first.py"]),
        _row("second", ["sed -i 's/old/new/' src/second.py"]),
    ]
    Dataset.from_list(rows).save_to_disk(str(source))

    def changed_writer(values: list[dict[str, Any]], staged_output: Path) -> None:
        changed = copy.deepcopy(values)
        if corruption == "drop":
            changed.pop()
        elif corruption == "reorder":
            changed.reverse()
        else:
            changed[0]["instance_id"] = "mutated"
        Dataset.from_list(changed).save_to_disk(str(staged_output))

    with pytest.raises(ValueError, match="staged HF dataset differs from filtered rows"):
        filter_hf_dataset(
            source,
            output,
            manifest_path,
            dataset_writer=changed_writer,
        )

    assert not output.exists()
    assert not manifest_path.exists()
    assert not list(tmp_path.glob(".*.tmp"))


@pytest.mark.parametrize(
    "topology",
    [
        "source-equals-output",
        "source-equals-manifest",
        "output-equals-manifest",
        "output-under-source",
        "manifest-under-output",
        "output-under-manifest",
        "source-under-output",
    ],
)
def test_filter_hf_dataset_rejects_equal_or_nested_artifact_paths(
    tmp_path: Path,
    topology: str,
) -> None:
    from datasets import Dataset

    root = tmp_path / topology
    root.mkdir()
    source = root / "source"
    output = root / "output"
    manifest_path = root / "manifest.json"
    if topology == "source-under-output":
        output = root / "bundle"
        source = output / "source"
    Dataset.from_list([_row("keep", ["sed -i 's/old/new/' src/mod.py"])]).save_to_disk(
        str(source)
    )
    if topology == "source-equals-output":
        output = source
    elif topology == "source-equals-manifest":
        manifest_path = source
    elif topology == "output-equals-manifest":
        manifest_path = output
    elif topology == "output-under-source":
        output = source / "filtered"
    elif topology == "manifest-under-output":
        manifest_path = output / "manifest.json"
    elif topology == "output-under-manifest":
        manifest_path = root / "artifact"
        output = manifest_path / "dataset"

    with pytest.raises(ValueError, match="distinct and non-nested"):
        filter_hf_dataset(source, output, manifest_path)

    if topology == "manifest-under-output":
        assert not output.exists()


@pytest.mark.parametrize("collision_target", ["output", "manifest"])
def test_filter_hf_dataset_refuses_dangling_symlink_collision(
    tmp_path: Path,
    collision_target: str,
) -> None:
    from datasets import Dataset

    source = tmp_path / "source"
    output = tmp_path / "filtered"
    manifest_path = tmp_path / "filtered.manifest.json"
    Dataset.from_list([_row("keep", ["sed -i 's/old/new/' src/mod.py"])]).save_to_disk(
        str(source)
    )
    collision = output if collision_target == "output" else manifest_path
    collision.symlink_to(tmp_path / "missing-target")

    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        filter_hf_dataset(source, output, manifest_path)

    assert collision.is_symlink()
    assert os.path.lexists(collision)


@pytest.mark.parametrize("kind", ["directory", "file"])
def test_rename_noreplace_never_clobbers_existing_destination(
    tmp_path: Path,
    kind: str,
) -> None:
    source = tmp_path / "source"
    destination = tmp_path / "destination"
    if kind == "directory":
        source.mkdir()
        destination.mkdir()
        (destination / "foreign.txt").write_text("foreign", encoding="utf-8")
    else:
        source.write_text("owned", encoding="utf-8")
        destination.write_text("foreign", encoding="utf-8")

    with pytest.raises(FileExistsError):
        filter_module._rename_noreplace(source, destination)

    assert source.exists()
    if kind == "directory":
        assert (destination / "foreign.txt").read_text(encoding="utf-8") == "foreign"
    else:
        assert destination.read_text(encoding="utf-8") == "foreign"


def test_dataset_rename_collision_preserves_foreign_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from datasets import Dataset

    source = tmp_path / "source"
    output = tmp_path / "filtered"
    manifest_path = tmp_path / "filtered.manifest.json"
    Dataset.from_list([_row("keep", ["sed -i 's/old/new/' src/mod.py"])]).save_to_disk(
        str(source)
    )

    def collide(_: Path, destination: Path) -> None:
        assert destination == output
        output.mkdir()
        (output / "foreign.txt").write_text("foreign", encoding="utf-8")
        raise FileExistsError("injected dataset collision")

    monkeypatch.setattr(filter_module, "_rename_noreplace", collide)
    with pytest.raises(FileExistsError, match="injected dataset collision"):
        filter_hf_dataset(source, output, manifest_path)

    assert (output / "foreign.txt").read_text(encoding="utf-8") == "foreign"
    assert not manifest_path.exists()
    assert not list(tmp_path.glob(".*.tmp"))


def test_manifest_rename_collision_rolls_back_owned_dataset_only(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from datasets import Dataset

    source = tmp_path / "source"
    output = tmp_path / "filtered"
    manifest_path = tmp_path / "filtered.manifest.json"
    Dataset.from_list([_row("keep", ["sed -i 's/old/new/' src/mod.py"])]).save_to_disk(
        str(source)
    )

    def collide(staged: Path, destination: Path) -> None:
        if destination == output:
            staged.rename(destination)
            return
        manifest_path.write_text("foreign", encoding="utf-8")
        raise FileExistsError("injected manifest collision")

    monkeypatch.setattr(filter_module, "_rename_noreplace", collide)
    with pytest.raises(FileExistsError, match="injected manifest collision"):
        filter_hf_dataset(source, output, manifest_path)

    assert not output.exists()
    assert manifest_path.read_text(encoding="utf-8") == "foreign"
    assert not list(tmp_path.glob(".*.tmp"))


@pytest.mark.parametrize("failure_boundary", ["dataset", "manifest"])
def test_post_rename_failure_rolls_back_artifacts_owned_by_this_invocation(
    tmp_path: Path,
    failure_boundary: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from datasets import Dataset

    source = tmp_path / "source"
    output = tmp_path / "filtered"
    manifest_path = tmp_path / "filtered.manifest.json"
    Dataset.from_list([_row("keep", ["sed -i 's/old/new/' src/mod.py"])]).save_to_disk(
        str(source)
    )

    def fail_after_rename(staged: Path, destination: Path) -> None:
        staged.rename(destination)
        if destination == output and failure_boundary == "dataset":
            raise RuntimeError("failure after dataset rename")
        if destination == manifest_path and failure_boundary == "manifest":
            raise RuntimeError("failure after manifest rename")

    monkeypatch.setattr(filter_module, "_rename_noreplace", fail_after_rename)
    with pytest.raises(RuntimeError, match=f"failure after {failure_boundary} rename"):
        filter_hf_dataset(source, output, manifest_path)

    assert not output.exists()
    assert not manifest_path.exists()
    assert not list(tmp_path.glob(".*.tmp"))


def test_rollback_does_not_delete_exact_copy_output_under_forced_identity_reuse(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from datasets import Dataset

    source = tmp_path / "source"
    output = tmp_path / "filtered"
    manifest_path = tmp_path / "filtered.manifest.json"
    Dataset.from_list([_row("keep", ["sed -i 's/old/new/' src/mod.py"])]).save_to_disk(
        str(source)
    )
    real_path_status = filter_module._path_status
    forced_statuses: dict[Path, os.stat_result] = {}

    def status_with_reuse(path: Path) -> os.stat_result | None:
        return forced_statuses.get(path, real_path_status(path))

    monkeypatch.setattr(filter_module, "_path_status", status_with_reuse)

    def replace_before_manifest_collision(staged: Path, destination: Path) -> None:
        if destination == output:
            staged.rename(destination)
            return
        owned_status = real_path_status(output)
        assert owned_status is not None
        owned_anchor_status = real_path_status(output / "content.sha256")
        assert owned_anchor_status is not None
        exact_copy = tmp_path / "exact-dataset-copy"
        shutil.copytree(output, exact_copy)
        shutil.rmtree(output)
        exact_copy.rename(output)
        forced_statuses[output] = owned_status
        forced_statuses[output / "content.sha256"] = owned_anchor_status
        manifest_path.write_text("foreign manifest", encoding="utf-8")
        raise FileExistsError("manifest collision after output replacement")

    monkeypatch.setattr(
        filter_module,
        "_rename_noreplace",
        replace_before_manifest_collision,
    )
    with pytest.raises(FileExistsError, match="manifest collision after output replacement"):
        filter_hf_dataset(source, output, manifest_path)

    assert output.is_dir()
    assert (output / "content.sha256").exists()
    assert manifest_path.read_text(encoding="utf-8") == "foreign manifest"
    assert not list(tmp_path.glob(".*.tmp"))


def test_rollback_does_not_delete_exact_copy_manifest_under_forced_identity_reuse(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from datasets import Dataset

    source = tmp_path / "source"
    output = tmp_path / "filtered"
    manifest_path = tmp_path / "filtered.manifest.json"
    Dataset.from_list([_row("keep", ["sed -i 's/old/new/' src/mod.py"])]).save_to_disk(
        str(source)
    )
    real_path_status = filter_module._path_status
    forced_statuses: dict[Path, os.stat_result] = {}

    def status_with_reuse(path: Path) -> os.stat_result | None:
        return forced_statuses.get(path, real_path_status(path))

    monkeypatch.setattr(filter_module, "_path_status", status_with_reuse)

    def replace_manifest_after_rename(staged: Path, destination: Path) -> None:
        staged.rename(destination)
        if destination != manifest_path:
            return
        owned_status = real_path_status(manifest_path)
        assert owned_status is not None
        exact_bytes = manifest_path.read_bytes()
        manifest_path.unlink()
        manifest_path.write_bytes(exact_bytes)
        forced_statuses[manifest_path] = owned_status
        raise RuntimeError("failure after foreign manifest replacement")

    monkeypatch.setattr(
        filter_module,
        "_rename_noreplace",
        replace_manifest_after_rename,
    )
    with pytest.raises(RuntimeError, match="failure after foreign manifest replacement"):
        filter_hf_dataset(source, output, manifest_path)

    assert not output.exists()
    assert json.loads(manifest_path.read_text(encoding="utf-8"))["rows_kept"] == 1
    assert not list(tmp_path.glob(".*.tmp"))


def test_filter_hf_dataset_all_dropped_fails_before_writer(
    tmp_path: Path,
) -> None:
    from datasets import Dataset

    source = tmp_path / "source"
    output = tmp_path / "filtered"
    manifest_path = tmp_path / "filtered.manifest.json"
    Dataset.from_list([_row("drop", ["pytest -q"])]).save_to_disk(str(source))

    writer_called = False

    def writer(_: list[dict[str, Any]], __: Path) -> None:
        nonlocal writer_called
        writer_called = True

    with pytest.raises(ValueError, match="filter produced no rows"):
        filter_hf_dataset(source, output, manifest_path, dataset_writer=writer)

    assert not writer_called
    assert not output.exists()
    assert not manifest_path.exists()
    assert not list(tmp_path.glob(".*.tmp"))


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"max_first_edit_index": 0}, "max_first_edit_index must be positive"),
        ({"max_first_edit_index": -1}, "max_first_edit_index must be positive"),
        ({"max_read_streak": -1}, "max_read_streak must be nonnegative"),
    ],
)
def test_filter_rows_rejects_invalid_public_limits(
    kwargs: dict[str, int],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        filter_rows([_row("keep", ["sed -i 's/old/new/' src/mod.py"])], **kwargs)


@pytest.mark.parametrize(
    ("flag", "value"),
    [
        ("--max-first-edit-index", "0"),
        ("--max-first-edit-index", "-1"),
        ("--max-read-streak", "-1"),
    ],
)
def test_filter_cli_rejects_invalid_limits(flag: str, value: str) -> None:
    with pytest.raises(SystemExit) as error:
        parse_args([
            "--in", "source",
            "--out", "output",
            "--manifest", "manifest.json",
            flag, value,
        ])

    assert error.value.code == 2


def test_filter_limit_boundary_accepts_first_edit_one_and_zero_read_streak() -> None:
    rows, _ = filter_rows(
        [_row("keep", ["sed -i 's/old/new/' src/mod.py"])],
        max_first_edit_index=1,
        max_read_streak=0,
    )
    args = parse_args([
        "--in", "source",
        "--out", "output",
        "--manifest", "manifest.json",
        "--max-first-edit-index", "1",
        "--max-read-streak", "0",
    ])

    assert [row["instance_id"] for row in rows] == ["keep"]
    assert args.max_first_edit_index == 1
    assert args.max_read_streak == 0
