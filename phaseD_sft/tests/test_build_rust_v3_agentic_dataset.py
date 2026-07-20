from __future__ import annotations

import json
import hashlib
from pathlib import Path
import threading
import time

import pytest

import phaseD_sft.build_rust_v3_agentic_dataset as rust_v3_builder

from phaseD_sft.build_rust_v3_agentic_dataset import (
    AuditInputs,
    BuildInputs,
    SourceRowStream,
    analyze_source_row,
    audit_source_rows,
    audit_streaming_dataset,
    build_streaming_dataset,
    canonical_id_sha256,
    compress_tracked_source_row,
    convert_rich_trajectory,
    extract_tracked_rust_paths,
    rendered_token_count,
    tokenizer_artifact_sha256,
    _parse_args,
)


def _tool(name: str, arguments: dict[str, object]) -> dict[str, object]:
    return {
        "id": "call-1",
        "type": "function",
        "function": {"name": name, "arguments": json.dumps(arguments)},
    }


def _call(
    call_id: str, name: str, arguments: dict[str, object]
) -> dict[str, object]:
    call = _tool(name, arguments)
    call["id"] = call_id
    return call


def _assistant(
    call_id: str,
    name: str,
    arguments: dict[str, object],
    *,
    reasoning: str = "reasoning",
) -> dict[str, object]:
    return {
        "role": "assistant",
        "content": "",
        "reasoning_content": reasoning,
        "tool_calls": [_call(call_id, name, arguments)],
    }


def _result(call_id: str, content: str = "ok") -> dict[str, object]:
    return {"role": "tool", "tool_call_id": call_id, "content": content}


def _valid_rich_row(*, pre_reads: int = 1) -> dict[str, object]:
    row = _row(command="unused")
    trajectory: list[dict[str, object]] = [
        {"role": "system", "content": "system contract"},
        {"role": "user", "content": "Fix the Rust bug."},
    ]
    for index in range(pre_reads):
        call_id = f"read-{index}"
        trajectory.extend(
            [
                _assistant(
                    call_id,
                    "execute_bash",
                    {"command": f"cat src/lib.rs # read {index}"},
                    reasoning=f"inspect-{index}",
                ),
                _result(call_id, f"source-{index}"),
            ]
        )
    trajectory.extend(
        [
            _assistant(
                "edit-1",
                "str_replace_editor",
                {
                    "command": "str_replace",
                    "path": "/testbed/src/lib.rs",
                    "old_str": "old",
                    "new_str": "new",
                },
                reasoning="edit-reasoning-verbatim",
            ),
            _result("edit-1", "edited"),
            _assistant("verify-1", "execute_bash", {"command": "cargo test"}),
            _result("verify-1", "tests passed"),
            _assistant(
                "finish-1",
                "finish",
                {"message": "done"},
                reasoning="finish-reasoning",
            ),
        ]
    )
    row["trajectory"] = trajectory
    return row


def _row(
    *,
    task_id: str = "owner__crate-1",
    repo: str = "owner/crate",
    trajectory_id: str = "trajectory-1",
    patch_path: str = "src/lib.rs",
    command: str = "sed -i 's/old/new/' src/lib.rs",
) -> dict[str, object]:
    patch = (
        f"diff --git a/{patch_path} b/{patch_path}\n"
        f"--- a/{patch_path}\n"
        f"+++ b/{patch_path}\n"
        "@@ -1 +1 @@\n-old\n+new\n"
    )
    return {
        "instance_id": task_id,
        "repo": repo,
        "trajectory_id": trajectory_id,
        "language": "rust",
        "resolved": "1",
        "metadata": {
            "model_patch": {
                "patch": patch,
                "num_modified_files": 1,
                "num_modified_lines": 2,
            }
        },
        "trajectory": [
            {"role": "user", "content": "Fix the Rust bug."},
            {
                "role": "assistant",
                "content": "I will change the implementation.",
                "tool_calls": [_tool("execute_bash", {"command": command})],
            },
            {"role": "tool", "content": "ok"},
        ],
    }


def _patch(*sections: tuple[str, str]) -> str:
    return "".join(
        f"diff --git a/{path} b/{path}\n"
        f"--- a/{path}\n"
        f"+++ b/{path}\n"
        f"{body}\n"
        for path, body in sections
    )


def _command(message: dict[str, object]) -> str:
    calls = message["tool_calls"]
    assert isinstance(calls, list) and len(calls) == 1
    call = calls[0]
    assert isinstance(call, dict)
    function = call["function"]
    assert isinstance(function, dict)
    arguments = json.loads(str(function["arguments"]))
    return str(arguments["command"])


def _decisive_fixture(
    commands: list[tuple[str, int]],
    *,
    patch: str,
) -> tuple[
    rust_v3_builder.AnalyzedSourceRow,
    rust_v3_builder.CompressedAgenticRow,
]:
    """Build a rich trajectory whose paired observations expose the given rc values."""
    trajectory: list[dict[str, object]] = [
        {"role": "system", "content": "system contract"},
        {
            "role": "user",
            "content": (
                "<uploaded_files>\n/workspace/repo\n</uploaded_files>\n"
                "Fix the Rust bug."
            ),
        },
        _assistant(
            "ground-0",
            "execute_bash",
            {"command": "cat src/lib.rs"},
            reasoning="grounding-reasoning-verbatim",
        ),
        _result("ground-0", "old source\n<returncode>0</returncode>"),
    ]
    for index, (command, returncode) in enumerate(commands):
        call_id = f"command-{index}"
        trajectory.extend(
            [
                _assistant(
                    call_id,
                    "execute_bash",
                    {"command": command},
                    reasoning=f"reasoning-{index}-verbatim",
                ),
                _result(
                    call_id,
                    f"output-{index}-verbatim\n<returncode>{returncode}</returncode>",
                ),
            ]
        )
    trajectory.append(
        _assistant(
            "finish-decisive",
            "finish",
            {"message": "done"},
            reasoning="finish-reasoning-verbatim",
        )
    )
    row = _row(command="unused")
    row["metadata"] = {"model_patch": {"patch": patch}}
    row["trajectory"] = trajectory
    analyzed = _analyzed_rich(row)
    converted, conversion_report = convert_rich_trajectory(analyzed.trajectory)
    assert conversion_report == {"kept": True}
    assert converted is not None
    first_command_pair = next(
        pair for pair in converted.tool_pairs if pair.command == commands[0][0]
    )
    compressed = rust_v3_builder.CompressedAgenticRow(
        identity=analyzed.identity,
        messages=converted.messages,
        original_message_indices=tuple(range(len(converted.messages))),
        original_messages=converted.messages,
        original_suffix_start=first_command_pair.assistant_index,
        compressed_suffix_start=first_command_pair.assistant_index,
    )
    return analyzed, compressed


def _real_compressed_decisive_fixture(
    commands: list[tuple[str, int]],
    *,
    patch: str,
) -> tuple[
    rust_v3_builder.AnalyzedSourceRow,
    rust_v3_builder.CompressedAgenticRow,
]:
    analyzed, _native_fixture = _decisive_fixture(commands, patch=patch)
    compressed, report = compress_tracked_source_row(analyzed)
    assert report["kept"] is True
    assert compressed is not None
    return analyzed, compressed


def test_same_row_patch_is_never_cross_joined_between_equal_task_ids() -> None:
    first = _row(trajectory_id="t-a", patch_path="src/a.rs", command="sed -i s/x/y/ src/a.rs")
    second = _row(trajectory_id="t-b", patch_path="src/b.rs", command="sed -i s/x/y/ src/a.rs")

    accepted, report = analyze_source_row(first, config="cfg-a", split="train")
    rejected, rejected_report = analyze_source_row(second, config="cfg-b", split="validation")

    assert report == {"kept": True}
    assert accepted is not None
    assert accepted.identity.task_id == "owner__crate-1"
    assert accepted.identity.trajectory_id == "t-a"
    assert accepted.identity.config == "cfg-a"
    assert accepted.identity.split == "train"
    assert accepted.tracked_mutations[0].tracked_paths == ("src/a.rs",)
    assert rejected is None
    assert rejected_report == {"kept": False, "reason": "no_tracked_source_mutation"}


def test_exact_paths_not_basenames_classify_tracked_mutations() -> None:
    row = _row(patch_path="src/core/lib.rs", command="sed -i s/x/y/ tests/lib.rs")
    accepted, report = analyze_source_row(row, config="cfg", split="train")
    assert accepted is None
    assert report == {"kept": False, "reason": "no_tracked_source_mutation"}


@pytest.mark.parametrize(
    ("patch_path", "reason"),
    [
        ("../src/lib.rs", "unsafe_patch_path"),
        ("/tmp/src/lib.rs", "unsafe_patch_path"),
        ("src/../../lib.rs", "unsafe_patch_path"),
    ],
)
def test_patch_paths_fail_closed_on_traversal_or_absolute_paths(
    patch_path: str, reason: str
) -> None:
    paths, report = extract_tracked_rust_paths(
        _row(patch_path=patch_path)["metadata"]["model_patch"]["patch"]  # type: ignore[index]
    )
    assert paths is None
    assert report == {"kept": False, "reason": reason}


def test_rename_patch_tracks_both_safe_rust_names() -> None:
    patch = (
        "diff --git a/src/old.rs b/src/new.rs\n"
        "similarity index 98%\n"
        "rename from src/old.rs\n"
        "rename to src/new.rs\n"
        "--- a/src/old.rs\n"
        "+++ b/src/new.rs\n"
        "@@ -1 +1 @@\n-old\n+new\n"
    )
    paths, report = extract_tracked_rust_paths(patch)
    assert report == {"kept": True}
    assert paths is not None
    assert paths.tracked_paths == ("src/new.rs", "src/old.rs")
    assert paths.source_paths == ("src/new.rs", "src/old.rs")


def test_patch_parser_exposes_all_textual_paths_for_row_local_allowlist() -> None:
    patch = _patch(
        ("src/lib.rs", "@@ -1 +1 @@\n-old\n+new"),
        ("Cargo.toml", "@@ -1 +1 @@\n-old\n+new"),
        ("include/api.hpp", "@@ -1 +1 @@\n-old\n+new"),
    )
    paths, report = rust_v3_builder.extract_tracked_rust_paths(patch)
    assert report == {"kept": True}
    assert paths is not None
    assert paths.tracked_paths == ("src/lib.rs",)
    assert paths.source_paths == ("src/lib.rs",)
    assert paths.textual_paths == ("Cargo.toml", "include/api.hpp", "src/lib.rs")


def test_hunkless_or_binary_nonrust_section_is_not_allowlisted() -> None:
    patch = (
        "diff --git a/src/lib.rs b/src/lib.rs\n"
        "--- a/src/lib.rs\n+++ b/src/lib.rs\n@@ -1 +1 @@\n-old\n+new\n"
        "diff --git a/Cargo.toml b/Cargo.toml\n"
        "Binary files a/Cargo.toml and b/Cargo.toml differ\n"
    )
    paths, report = rust_v3_builder.extract_tracked_rust_paths(patch)
    assert report == {"kept": True}
    assert paths is not None
    assert paths.textual_paths == ("src/lib.rs",)


def test_multiple_trajectories_for_one_task_remain_distinct_rows() -> None:
    rows = [_row(trajectory_id="t-2"), _row(trajectory_id="t-1")]
    accepted, counters = audit_source_rows(rows, config="cfg", split="train")
    assert [row.identity.trajectory_id for row in accepted] == ["t-2", "t-1"]
    assert counters == {
        "rows_scanned": 2,
        "rows_eligible": 2,
        "unique_task_ids": 1,
        "repositories": 1,
    }


def test_test_or_repro_write_is_not_a_source_edit_unless_patch_tracks_it() -> None:
    row = _row(
        patch_path="src/lib.rs",
        command="cat > tests/repro.rs <<'EOF'\nfn main() {}\nEOF",
    )
    accepted, report = analyze_source_row(row, config="cfg", split="train")
    assert accepted is None
    assert report == {"kept": False, "reason": "no_tracked_source_mutation"}

    tracked_test_only = _row(
        patch_path="tests/repro.rs",
        command="cat > tests/repro.rs <<'EOF'\nfn main() {}\nEOF",
    )
    accepted, report = analyze_source_row(tracked_test_only, config="cfg", split="train")
    assert accepted is None
    assert report == {"kept": False, "reason": "test_only_patch"}


def test_tracked_test_write_in_mixed_patch_is_a_row_local_tracked_mutation() -> None:
    row = _row(command="cat > tests/regression.rs <<'EOF'\nfn regression() {}\nEOF")
    row["metadata"] = {
        "model_patch": {
            "patch": (
                "diff --git a/src/lib.rs b/src/lib.rs\n"
                "--- a/src/lib.rs\n+++ b/src/lib.rs\n@@ -1 +1 @@\n-old\n+new\n"
                "diff --git a/tests/regression.rs b/tests/regression.rs\n"
                "--- /dev/null\n+++ b/tests/regression.rs\n@@ -0,0 +1 @@\n+fn regression() {}\n"
            ),
            "num_modified_files": 2,
            "num_modified_lines": 4,
        }
    }

    accepted, report = analyze_source_row(row, config="cfg", split="train")

    assert report == {"kept": True}
    assert accepted is not None
    assert accepted.tracked_mutations[0].tracked_paths == ("tests/regression.rs",)


@pytest.mark.parametrize(
    ("field", "value", "reason"),
    [
        ("resolved", "0", "not_resolved"),
        ("language", "python", "not_rust"),
    ],
)
def test_source_eligibility_is_part_of_row_local_analysis(
    field: str, value: str, reason: str
) -> None:
    row = _row()
    row[field] = value
    accepted, report = analyze_source_row(row, config="cfg", split="train")
    assert accepted is None
    assert report == {"kept": False, "reason": reason}


@pytest.mark.parametrize(
    ("mutator", "reason"),
    [
        (lambda row: row.update(metadata={}), "missing_model_patch"),
        (lambda row: row.update(metadata={"model_patch": 123}), "malformed_model_patch"),
        (
            lambda row: row.update(metadata={"model_patch": "raw patch"}),
            "malformed_model_patch",
        ),
        (
            lambda row: row.update(
                metadata={
                    "model_patch": {
                        "patch": "diff --git a/src/logo.rs b/src/logo.rs\nGIT binary patch\n",
                        "num_modified_files": 1,
                        "num_modified_lines": 0,
                    }
                }
            ),
            "binary_only_patch",
        ),
        (
            lambda row: row.update(
                metadata={"model_patch": {"patch": "plain text only\n"}}
            ),
            "pathless_patch",
        ),
        (
            lambda row: row.update(
                metadata={
                    "model_patch": {
                        "patch": (
                            "diff --git a/README.md b/README.md\n"
                            "--- a/README.md\n+++ b/README.md\n"
                            "@@ -1 +1 @@\n-a\n+b\n"
                        )
                    }
                }
            ),
            "no_rust_paths",
        ),
        (lambda row: row.update(instance_id=" ../bad "), "invalid_task_id"),
        (lambda row: row.update(repo="owner/../crate"), "invalid_repository"),
        (lambda row: row.update(trajectory="not a trajectory"), "malformed_trajectory"),
    ],
)
def test_row_rejections_have_stable_explicit_reasons(mutator, reason: str) -> None:
    row = _row()
    mutator(row)
    accepted, report = analyze_source_row(row, config="cfg", split="train")
    assert accepted is None
    assert report == {"kept": False, "reason": reason}


def test_declared_workspace_root_is_normalized_for_editor_mutation() -> None:
    row = _row(command="unused")
    row["trajectory"] = [
        {
            "role": "user",
            "content": "<uploaded_files>\n/workspace/crate\n</uploaded_files>\nFix it.",
        },
        {
            "role": "assistant",
            "content": "edit",
            "tool_calls": [
                _tool(
                    "str_replace_editor",
                    {
                        "command": "str_replace",
                        "path": "/workspace/crate/src/lib.rs",
                        "old_str": "old",
                        "new_str": "new",
                    },
                )
            ],
        },
        {"role": "tool", "content": "ok"},
    ]
    accepted, report = analyze_source_row(row, config="cfg", split="train")
    assert report == {"kept": True}
    assert accepted is not None
    assert accepted.tracked_mutations[0].tracked_paths == ("src/lib.rs",)


def test_live_nested_model_patch_schema_is_unwrapped_strictly() -> None:
    accepted, report = analyze_source_row(_row(), config="cfg", split="train")
    assert report == {"kept": True}
    assert accepted is not None
    assert accepted.patch.startswith("diff --git a/src/lib.rs")


def test_diff_headers_must_belong_to_their_own_file_section() -> None:
    patch = (
        "diff --git a/src/a.rs b/src/a.rs\n"
        "--- a/src/other.rs\n"
        "+++ b/src/a.rs\n"
        "@@ -1 +1 @@\n-old\n+new\n"
    )
    paths, report = extract_tracked_rust_paths(patch)
    assert paths is None
    assert report == {"kept": False, "reason": "malformed_model_patch"}


def test_hunk_content_that_looks_like_a_file_header_is_not_reparsed() -> None:
    patch = (
        "diff --git a/src/a.rs b/src/a.rs\n"
        "--- a/src/a.rs\n"
        "+++ b/src/a.rs\n"
        "@@ -1 +1 @@\n"
        "--- this removed Rust string resembles a header\n"
        "+++ this added Rust string resembles a header\n"
    )
    paths, report = extract_tracked_rust_paths(patch)
    assert report == {"kept": True}
    assert paths is not None
    assert paths.tracked_paths == ("src/a.rs",)


def test_binary_rust_section_cannot_borrow_text_hunk_from_non_rust_section() -> None:
    patch = (
        "diff --git a/src/logo.rs b/src/logo.rs\n"
        "GIT binary patch\n"
        "literal 1\nA\n"
        "diff --git a/README.md b/README.md\n"
        "--- a/README.md\n+++ b/README.md\n"
        "@@ -1 +1 @@\n-old\n+new\n"
    )
    paths, report = extract_tracked_rust_paths(patch)
    assert paths is None
    assert report == {"kept": False, "reason": "binary_only_patch"}


def test_each_accepted_rust_section_requires_its_own_text_hunk() -> None:
    patch = (
        "diff --git a/src/lib.rs b/src/lib.rs\n"
        "old mode 100644\n"
        "new mode 100755\n"
    )
    paths, report = extract_tracked_rust_paths(patch)
    assert paths is None
    assert report == {"kept": False, "reason": "rust_path_without_text_hunk"}


def test_embedded_diff_redirected_to_repro_patch_is_not_a_source_mutation() -> None:
    command = (
        "cat > /tmp/repro.patch <<'PATCH'\n"
        "diff --git a/src/lib.rs b/src/lib.rs\n"
        "--- a/src/lib.rs\n+++ b/src/lib.rs\n@@ -1 +1 @@\n-old\n+new\n"
        "PATCH"
    )
    accepted, report = analyze_source_row(
        _row(command=command), config="cfg", split="train"
    )
    assert accepted is None
    assert report == {"kept": False, "reason": "no_tracked_source_mutation"}


def test_embedded_diff_counts_for_an_actual_apply_patch_operator() -> None:
    command = (
        "apply_patch <<'PATCH'\n"
        "diff --git a/src/lib.rs b/src/lib.rs\n"
        "--- a/src/lib.rs\n+++ b/src/lib.rs\n@@ -1 +1 @@\n-old\n+new\n"
        "PATCH"
    )
    accepted, report = analyze_source_row(
        _row(command=command), config="cfg", split="train"
    )
    assert report == {"kept": True}
    assert accepted is not None
    assert accepted.tracked_mutations[0].tracked_paths == ("src/lib.rs",)


def test_apply_patch_text_inside_repro_heredoc_is_not_an_operator() -> None:
    command = (
        "cat > /tmp/repro.sh <<'SCRIPT'\n"
        "apply_patch\n"
        "diff --git a/src/lib.rs b/src/lib.rs\n"
        "SCRIPT"
    )
    accepted, report = analyze_source_row(
        _row(command=command), config="cfg", split="train"
    )
    assert accepted is None
    assert report == {"kept": False, "reason": "no_tracked_source_mutation"}


@pytest.mark.parametrize(
    "command",
    [
        "printf 'diff --git a/src/lib.rs b/src/lib.rs' > /tmp/repro.patch",
        "echo '*** Update File: src/lib.rs' > /tmp/repro.patch",
    ],
)
def test_inline_inert_patch_markers_are_not_source_mutations(command: str) -> None:
    accepted, report = analyze_source_row(
        _row(command=command), config="cfg", split="train"
    )
    assert accepted is None
    assert report == {"kept": False, "reason": "no_tracked_source_mutation"}


@pytest.mark.parametrize(
    ("command", "repo", "scratch", "ambiguous"),
    [
        ("sed -i s/a/b/ src/lib.rs", ("src/lib.rs",), (), False),
        (
            "cd /workspace/repo && cat > Cargo.toml <<'EOF'\nX\nEOF",
            ("Cargo.toml",),
            (),
            False,
        ),
        (
            "cd /workspace && cat > repro.rs <<'EOF'\nfn main(){}\nEOF",
            (),
            ("/workspace/repro.rs",),
            False,
        ),
        ("cat > $TARGET <<'EOF'\nX\nEOF", (), (), True),
    ],
)
def test_mutation_scope_distinguishes_repo_scratch_and_ambiguous(
    command: str,
    repo: tuple[str, ...],
    scratch: tuple[str, ...],
    ambiguous: bool,
) -> None:
    scope = rust_v3_builder._mutation_scope(command, "/workspace/repo")
    assert scope.repository_paths == repo
    assert scope.scratch_paths == scratch
    assert scope.ambiguous is ambiguous


@pytest.mark.parametrize(
    "path",
    [
        "tests/case.rs",
        "examples/demo.rs",
        "fixtures/input.txt",
        "benches/bench.rs",
        "Cargo.lock",
        "supply-chain/imports.lock",
    ],
)
def test_forbidden_training_paths(path: str) -> None:
    assert rust_v3_builder._forbidden_training_path(path)


@pytest.mark.parametrize(
    "command",
    [
        "sed -i s/a/b/ ../src/lib.rs",
        "cat > ${TARGET} <<'EOF'\nX\nEOF",
        "cd /workspace/repo && sed -i s/a/b/ src/lib.rs; "
        "cd /workspace && cat > repro.rs <<'EOF'\nX\nEOF",
        "(cd /workspace && cat > repro.rs <<'EOF'\nX\nEOF)",
    ],
)
def test_mutation_scope_fails_closed_for_unresolvable_location(command: str) -> None:
    scope = rust_v3_builder._mutation_scope(command, "/workspace/repo")
    assert scope.repository_paths == ()
    assert scope.scratch_paths == ()
    assert scope.ambiguous is True


@pytest.mark.parametrize(
    ("command", "repo"),
    [
        (
            "printf x > src/lib.rs && printf y > tests/case.rs",
            ("src/lib.rs", "tests/case.rs"),
        ),
        (
            "cat > src/lib.rs </dev/null && cat > Cargo.lock </dev/null",
            ("src/lib.rs", "Cargo.lock"),
        ),
        (
            "cargo check > build.log && true > Cargo.lock",
            ("build.log", "Cargo.lock"),
        ),
    ],
)
def test_mutation_scope_classifies_every_compound_redirect(
    command: str, repo: tuple[str, ...]
) -> None:
    scope = rust_v3_builder._mutation_scope(command, "/workspace/repo")
    assert scope.repository_paths == repo
    assert scope.scratch_paths == ()
    assert scope.ambiguous is False


@pytest.mark.parametrize(
    "body",
    [
        "from pathlib import Path\nPath('src/lib.rs').write_text('x')",
        "open('Cargo.lock', 'w').write('x')",
        "sed -i s/a/b/ src/lib.rs",
    ],
)
def test_mutation_scope_ignores_inert_heredoc_body_mutations(body: str) -> None:
    command = f"cat > /tmp/repro.py <<'PY'\n{body}\nPY"
    scope = rust_v3_builder._mutation_scope(command, "/workspace/repo")
    assert scope.repository_paths == ()
    assert scope.scratch_paths == ("/tmp/repro.py",)
    assert scope.ambiguous is False


def test_mutation_scope_rejects_interpreter_heredoc_semantics() -> None:
    command = (
        "python3 - <<'PY'\n"
        "from pathlib import Path\n"
        "Path('src/lib.rs').write_text('x')\n"
        "PY"
    )
    scope = rust_v3_builder._mutation_scope(command, "/workspace/repo")
    assert scope.repository_paths == ()
    assert scope.scratch_paths == ()
    assert scope.ambiguous is True


@pytest.mark.parametrize(
    "command",
    [
        "command cd /workspace && sed -i s/a/b/ repro.rs",
        "builtin cd /workspace && sed -i s/a/b/ repro.rs",
        "{ cd /workspace; sed -i s/a/b/ repro.rs; }",
        "(cd /workspace && sed -i s/a/b/ repro.rs)",
        "pushd /workspace >/dev/null && sed -i s/a/b/ repro.rs",
    ],
)
def test_mutation_scope_rejects_unsupported_working_directory_changes(
    command: str,
) -> None:
    scope = rust_v3_builder._mutation_scope(command, "/workspace/repo")
    assert scope.repository_paths == ()
    assert scope.scratch_paths == ()
    assert scope.ambiguous is True


@pytest.mark.parametrize(
    "command",
    [
        "rm tests/case.rs",
        "rm -rf src",
        "touch Cargo.lock",
        "cp /tmp/fixed.rs src/lib.rs",
        "cp src/lib.rs tests/case.rs",
        "mv src/lib.rs tests/case.rs",
        "install /tmp/fixed.rs src/lib.rs",
        "truncate -s 0 src/lib.rs",
        "chmod +x src/lib.rs",
        "ln -sf /tmp/fixed.rs src/lib.rs",
        "git checkout -- src/lib.rs",
        "git restore src/lib.rs",
        "git reset --hard HEAD",
        "git clean -fd",
        "rm src/*.rs",
        "touch $TARGET",
        "cp /tmp/fixed.rs $TARGET",
        "git checkout -- .",
    ],
)
def test_mutation_scope_rejects_unclassified_worktree_mutators(
    command: str,
) -> None:
    scope = rust_v3_builder._mutation_scope(command, "/workspace/repo")
    assert scope.repository_paths == ()
    assert scope.scratch_paths == ()
    assert scope.ambiguous is True


@pytest.mark.parametrize(
    "command",
    [
        "command git reset --hard HEAD",
        "env git checkout -- src/lib.rs",
        "sudo git clean -fd",
        "xargs git reset --hard HEAD",
        "env FOO=1 git checkout -- src/lib.rs",
        "sudo -u root git clean -fd",
    ],
)
def test_mutation_scope_rejects_wrapped_git_worktree_mutators(
    command: str,
) -> None:
    scope = rust_v3_builder._mutation_scope(command, "/workspace/repo")
    assert scope.repository_paths == ()
    assert scope.scratch_paths == ()
    assert scope.ambiguous is True


@pytest.mark.parametrize(
    "command",
    [
        "sh -c 'git reset --hard HEAD'",
        "bash -c 'rm tests/case.rs'",
        "python3 -c \"import os; os.remove('tests/case.rs')\"",
    ],
)
def test_mutation_scope_rejects_interpreter_command_strings(command: str) -> None:
    scope = rust_v3_builder._mutation_scope(command, "/workspace/repo")
    assert scope.repository_paths == ()
    assert scope.scratch_paths == ()
    assert scope.ambiguous is True


@pytest.mark.parametrize(
    "command",
    [
        "bash -lc 'rm tests/case.rs'",
        "sh -ec 'git reset --hard HEAD'",
        "zsh -fc 'touch Cargo.lock'",
        "env bash -lc 'rm tests/case.rs'",
    ],
)
def test_mutation_scope_rejects_combined_command_string_flags(
    command: str,
) -> None:
    scope = rust_v3_builder._mutation_scope(command, "/workspace/repo")
    assert scope.repository_paths == ()
    assert scope.scratch_paths == ()
    assert scope.ambiguous is True


@pytest.mark.parametrize(
    "command",
    [
        "bash script-c",
        "bash --noprofile script-c",
        "bash --command script-c",
        "env bash script-c",
    ],
)
def test_mutation_scope_does_not_treat_long_options_or_operands_as_short_c(
    command: str,
) -> None:
    scope = rust_v3_builder._mutation_scope(command, "/workspace/repo")
    assert scope.repository_paths == ()
    assert scope.scratch_paths == ()
    assert scope.ambiguous is False


@pytest.mark.parametrize(
    "command",
    [
        "fish -C 'touch Cargo.lock'",
        "fish --command='rm tests/case.rs'",
        "fish --init-command='touch Cargo.lock'",
        "env fish -C 'git reset --hard HEAD'",
        "fish --command 'rm tests/case.rs'",
        "fish --init-command 'touch Cargo.lock'",
    ],
)
def test_mutation_scope_rejects_fish_command_string_options(command: str) -> None:
    scope = rust_v3_builder._mutation_scope(command, "/workspace/repo")
    assert scope.repository_paths == ()
    assert scope.scratch_paths == ()
    assert scope.ambiguous is True


def _observation(rc: int) -> dict[str, object]:
    return {
        "role": "user",
        "content": f"OBSERVATION:\n<returncode>{rc}</returncode>\noutput",
    }


@pytest.mark.parametrize(
    ("command", "rc", "expected"),
    [
        ("cargo test", 0, True),
        ("cargo check", 1, False),
        ("cargo test 2>&1 | tail -20", 0, False),
        ("set -o pipefail; cargo test 2>&1 | tail -20", 0, True),
        ("grep cargo Cargo.toml", 0, False),
    ],
)
def test_trusted_verification_requires_real_command_unmasked_zero_rc(
    command: str, rc: int, expected: bool
) -> None:
    assert rust_v3_builder._trusted_rust_verification(
        command, _observation(rc)
    ) is expected


@pytest.mark.parametrize(
    ("command", "content", "expected"),
    [
        ("cargo test", "OBSERVATION:\noutput only", False),
        (
            "cargo test",
            "OBSERVATION:\n<returncode>0</returncode>\n"
            "command finished with exit code 1",
            False,
        ),
        ("cargo test |& tail -20", "OBSERVATION:\n<returncode>0</returncode>", False),
        (
            "set -eo pipefail; cargo test | tail -20",
            "OBSERVATION:\ncommand completed with exit code 0",
            True,
        ),
    ],
)
def test_trusted_verification_fails_closed_on_masking_or_bad_returncode(
    command: str, content: str, expected: bool
) -> None:
    observation = {"role": "user", "content": content}
    assert rust_v3_builder._trusted_rust_verification(command, observation) is expected


@pytest.mark.parametrize(
    "command",
    [
        "cargo test || true",
        "cargo test; true",
        "cargo test; exit 0",
        "cargo test || echo ignored",
        "cargo test | tail -20; set -o pipefail",
        "set -o pipefail; set +o pipefail; cargo test | tail -20",
        "echo '; cargo test'",
    ],
)
def test_trusted_verification_rejects_masked_or_inert_rust_commands(
    command: str,
) -> None:
    assert rust_v3_builder._trusted_rust_verification(
        command, _observation(0)
    ) is False


def test_trusted_verification_uses_executable_tokens_not_quoted_operators() -> None:
    assert rust_v3_builder._trusted_rust_verification(
        "echo '|' && cargo test", _observation(0)
    ) is True


def test_synthetic_audit_counter_preserves_exact_736_boundary() -> None:
    rows = [
        _row(task_id=f"owner__crate-{index}", trajectory_id=f"t-{index}")
        for index in range(736)
    ]
    rows.extend(
        [
            _row(task_id="owner__drop-no-edit", command="cat src/lib.rs"),
            _row(task_id="owner__drop-test", patch_path="tests/repro.rs"),
        ]
    )

    accepted, counters = audit_source_rows(rows, config="fixture", split="train")

    assert len(accepted) == 736
    assert counters == {
        "rows_scanned": 738,
        "rows_eligible": 736,
        "unique_task_ids": 736,
        "repositories": 1,
        "dropped_no_tracked_source_mutation": 1,
        "dropped_test_only_patch": 1,
    }


def _analyzed_rich(row: dict[str, object]):
    analyzed, report = analyze_source_row(row, config="cfg", split="train")
    assert report == {"kept": True}
    assert analyzed is not None
    return analyzed


def test_rich_tools_render_to_one_native_bash_call_with_exact_pair_id() -> None:
    analyzed = _analyzed_rich(_valid_rich_row())
    converted, report = convert_rich_trajectory(analyzed.trajectory)
    assert report == {"kept": True}
    assert converted is not None
    edit_pair = next(pair for pair in converted.tool_pairs if pair.raw_assistant_index == 4)
    assistant = converted.messages[edit_pair.assistant_index]
    observation = converted.messages[edit_pair.observation_index]
    assert assistant["content"] == "edit-reasoning-verbatim"
    assert len(assistant["tool_calls"]) == 1
    assert assistant["tool_calls"][0]["id"] == "edit-1"
    assert assistant["tool_calls"][0]["function"]["name"] == "bash"
    assert observation == {
        "role": "user",
        "content": "OBSERVATION:\nedited",
        "tool_calls": [],
    }


def test_compression_keeps_last_three_relevant_reads_and_exact_suffix() -> None:
    analyzed = _analyzed_rich(_valid_rich_row(pre_reads=4))
    converted, convert_report = convert_rich_trajectory(analyzed.trajectory)
    assert convert_report == {"kept": True}
    assert converted is not None

    compressed, report = compress_tracked_source_row(analyzed)

    assert report["kept"] is True
    assert report["kept_pre_edit_reads"] == 3
    assert report["first_edit_command_index"] == 4
    assert compressed is not None
    kept_reasoning = [
        message["content"]
        for message in compressed.messages
        if message["role"] == "assistant"
    ]
    assert "inspect-0" not in kept_reasoning
    assert kept_reasoning[:3] == ["inspect-1", "inspect-2", "inspect-3"]
    assert compressed.messages[compressed.compressed_suffix_start :] == (
        converted.messages[compressed.original_suffix_start :]
    )
    assert json.dumps(
        compressed.messages[compressed.compressed_suffix_start :],
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode() == json.dumps(
        converted.messages[compressed.original_suffix_start :],
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode()
    assert compressed.messages[compressed.compressed_suffix_start] == (
        converted.messages[compressed.original_suffix_start]
    )


def test_decisive_suffix_keeps_gold_edits_through_first_trusted_success_verification():
    analyzed, compressed = _decisive_fixture(
        [
            ("sed -i s/old/new/ src/lib.rs", 0),
            ("cargo check", 1),
            ("sed -i s/new/fixed/ src/lib.rs", 0),
            ("cargo test", 0),
            ("cargo test", 0),
            ("cat IMPLEMENTATION_SUMMARY.md", 0),
        ],
        patch=_patch(("src/lib.rs", "@@ -1 +1 @@\n-old\n+fixed")),
    )
    result, report = rust_v3_builder._build_decisive_suffix(analyzed, compressed)
    assert report == {"kept": True}
    assert result is not None
    kept = [
        _command(message)
        for message in result.messages
        if message["role"] == "assistant"
    ]
    assert kept[-2:] == ["cargo test", rust_v3_builder._COMPLETE_COMMAND]
    assert result.final_edit_command_index == 4
    assert result.verification_command_index == 5


def test_decisive_suffix_rejects_later_valid_edit_without_later_verification():
    analyzed, compressed = _decisive_fixture(
        [
            ("sed -i s/old/new/ src/lib.rs", 0),
            ("cargo test", 0),
            ("sed -i s/new/fixed/ src/lib.rs", 0),
        ],
        patch=_patch(("src/lib.rs", "@@ -1 +1 @@\n-old\n+fixed")),
    )
    assert rust_v3_builder._build_decisive_suffix(analyzed, compressed)[1][
        "reason"
    ] == "missing_trusted_final_verification"


def test_decisive_suffix_rejects_later_forbidden_mutation_after_early_success():
    analyzed, compressed = _decisive_fixture(
        [
            ("sed -i s/a/b/ src/lib.rs", 0),
            ("cargo test", 0),
            ("cat > tests/case.rs <<'EOF'\nrepro\nEOF", 0),
        ],
        patch=_patch(("src/lib.rs", "@@ -1 +1 @@\n-a\n+b")),
    )
    assert rust_v3_builder._build_decisive_suffix(analyzed, compressed)[1][
        "reason"
    ] == "forbidden_mutation_path"


def test_decisive_suffix_rejects_later_duplicate_edit_after_early_success():
    edit = "sed -i s/a/b/ src/lib.rs"
    analyzed, compressed = _decisive_fixture(
        [(edit, 0), ("cargo test", 0), ("git status --short", 0), (edit, 0)],
        patch=_patch(("src/lib.rs", "@@ -1 +1 @@\n-a\n+b")),
    )
    assert rust_v3_builder._build_decisive_suffix(analyzed, compressed)[1][
        "reason"
    ] == "repeated_edit_command"


def test_decisive_suffix_uses_later_edit_and_its_following_success():
    analyzed, compressed = _decisive_fixture(
        [
            ("sed -i s/old/new/ src/lib.rs", 0),
            ("cargo test", 0),
            ("sed -i s/new/fixed/ src/lib.rs", 0),
            ("cargo test --all", 0),
        ],
        patch=_patch(("src/lib.rs", "@@ -1 +1 @@\n-old\n+fixed")),
    )
    result, report = rust_v3_builder._build_decisive_suffix(analyzed, compressed)
    assert report == {"kept": True}
    assert result is not None
    assert [
        _command(message)
        for message in result.messages
        if message["role"] == "assistant"
    ][-3:] == [
        "sed -i s/new/fixed/ src/lib.rs",
        "cargo test --all",
        rust_v3_builder._COMPLETE_COMMAND,
    ]
    assert result.final_edit_command_index == 4
    assert result.verification_command_index == 5


def test_decisive_suffix_rejects_repo_test_edit_even_when_final_patch_contains_it():
    analyzed, compressed = _decisive_fixture(
        [
            ("sed -i s/a/b/ src/lib.rs", 0),
            ("sed -i s/a/b/ tests/case.rs", 0),
            ("cargo test", 0),
        ],
        patch=_patch(
            ("src/lib.rs", "@@ -1 +1 @@\n-a\n+b"),
            ("tests/case.rs", "@@ -1 +1 @@\n-a\n+b"),
        ),
    )
    result, report = rust_v3_builder._build_decisive_suffix(analyzed, compressed)
    assert result is None
    assert report["reason"] == "forbidden_patch_path"


def test_decisive_suffix_rejects_forbidden_repository_mutation():
    analyzed, compressed = _decisive_fixture(
        [
            ("sed -i s/a/b/ src/lib.rs", 0),
            ("cat > tests/case.rs <<'EOF'\nrepro\nEOF", 0),
            ("cargo test", 0),
        ],
        patch=_patch(("src/lib.rs", "@@ -1 +1 @@\n-a\n+b")),
    )
    assert rust_v3_builder._build_decisive_suffix(analyzed, compressed)[1][
        "reason"
    ] == "forbidden_mutation_path"


def test_decisive_suffix_rejects_nonallowlisted_repo_mutation():
    analyzed, compressed = _decisive_fixture(
        [
            ("sed -i s/a/b/ src/lib.rs", 0),
            ("cat > README_FIX.md <<'EOF'\nsummary\nEOF", 0),
            ("cargo test", 0),
        ],
        patch=_patch(("src/lib.rs", "@@ -1 +1 @@\n-a\n+b")),
    )
    assert rust_v3_builder._build_decisive_suffix(analyzed, compressed)[1][
        "reason"
    ] == "nonallowlisted_repository_mutation"


def test_decisive_suffix_allows_final_patch_tracked_nonrust_source_file():
    analyzed, compressed = _decisive_fixture(
        [
            ("sed -i s/a/b/ src/lib.rs", 0),
            ("sed -i s/a/b/ include/api.hpp", 0),
            ("cargo test", 0),
        ],
        patch=_patch(
            ("src/lib.rs", "@@ -1 +1 @@\n-a\n+b"),
            ("include/api.hpp", "@@ -1 +1 @@\n-a\n+b"),
        ),
    )
    result, report = rust_v3_builder._build_decisive_suffix(analyzed, compressed)
    assert result is not None and report["kept"] is True


def test_decisive_suffix_rejects_nonconsecutive_identical_edit_commands():
    edit = "sed -i s/a/b/ src/lib.rs"
    analyzed, compressed = _decisive_fixture(
        [(edit, 0), ("sed -n 1,20p src/lib.rs", 0), (edit, 0), ("cargo test", 0)],
        patch=_patch(("src/lib.rs", "@@ -1 +1 @@\n-a\n+b")),
    )
    assert rust_v3_builder._build_decisive_suffix(analyzed, compressed)[1][
        "reason"
    ] == "repeated_edit_command"


def test_decisive_suffix_rejects_missing_trusted_verify_after_final_edit():
    analyzed, compressed = _decisive_fixture(
        [("sed -i s/a/b/ src/lib.rs", 0), ("cargo test 2>&1 | tail -20", 0)],
        patch=_patch(("src/lib.rs", "@@ -1 +1 @@\n-a\n+b")),
    )
    assert rust_v3_builder._build_decisive_suffix(analyzed, compressed)[1][
        "reason"
    ] == "missing_trusted_final_verification"


def test_decisive_suffix_retains_one_consumed_scratch_chain_before_final_edit():
    analyzed, compressed = _decisive_fixture(
        [
            ("cat > /tmp/fix.sed <<'EOF'\ns/a/b/\nEOF", 0),
            ("sed -i -f /tmp/fix.sed src/lib.rs", 0),
            ("cargo test", 0),
        ],
        patch=_patch(("src/lib.rs", "@@ -1 +1 @@\n-a\n+b")),
    )
    result, report = rust_v3_builder._build_decisive_suffix(analyzed, compressed)
    assert report == {"kept": True}
    assert result is not None
    assert [_command(message) for message in result.messages if message["role"] == "assistant"] == [
        "cat src/lib.rs",
        "cat > /tmp/fix.sed <<'EOF'\ns/a/b/\nEOF",
        "sed -i -f /tmp/fix.sed src/lib.rs",
        "cargo test",
        rust_v3_builder._COMPLETE_COMMAND,
    ]


def test_decisive_suffix_reinserts_real_compression_scratch_creator_in_original_order():
    analyzed, compressed = _real_compressed_decisive_fixture(
        [
            ("cat > /tmp/fix.sed <<'EOF'\ns/a/b/\nEOF", 0),
            ("sed -i -f /tmp/fix.sed src/lib.rs", 0),
            ("cargo test", 0),
        ],
        patch=_patch(("src/lib.rs", "@@ -1 +1 @@\n-a\n+b")),
    )
    assert "cat > /tmp/fix.sed <<'EOF'\ns/a/b/\nEOF" not in [
        _command(message)
        for message in compressed.messages
        if message["role"] == "assistant"
    ]

    result, report = rust_v3_builder._build_decisive_suffix(analyzed, compressed)

    assert report == {"kept": True}
    assert result is not None
    assert [
        _command(message)
        for message in result.messages
        if message["role"] == "assistant"
    ] == [
        "cat src/lib.rs",
        "cat > /tmp/fix.sed <<'EOF'\ns/a/b/\nEOF",
        "sed -i -f /tmp/fix.sed src/lib.rs",
        "cargo test",
        rust_v3_builder._COMPLETE_COMMAND,
    ]
    assert result.final_edit_command_index == 3
    assert result.verification_command_index == 4


def test_decisive_suffix_rejects_token_only_false_scratch_consume_after_real_compression():
    analyzed, compressed = _real_compressed_decisive_fixture(
        [
            ("cat > /tmp/fix.sed <<'EOF'\ns/a/b/\nEOF", 0),
            ("sed -i s/a/b/ src/lib.rs; cat /tmp/fix.sed", 0),
            ("cargo test", 0),
        ],
        patch=_patch(("src/lib.rs", "@@ -1 +1 @@\n-a\n+b")),
    )
    assert rust_v3_builder._build_decisive_suffix(analyzed, compressed)[1][
        "reason"
    ] == "invalid_scratch_chain"


def test_decisive_suffix_rejects_path_printed_into_repo_as_scratch_consume():
    analyzed, compressed = _real_compressed_decisive_fixture(
        [
            ("cat > /tmp/fix.sed <<'EOF'\ns/a/b/\nEOF", 0),
            ("printf %s /tmp/fix.sed > src/lib.rs", 0),
            ("cargo test", 0),
        ],
        patch=_patch(("src/lib.rs", "@@ -1 +1 @@\n-a\n+b")),
    )
    assert rust_v3_builder._build_decisive_suffix(analyzed, compressed)[1][
        "reason"
    ] == "invalid_scratch_chain"


def test_decisive_suffix_rejects_scratch_edit_that_requires_preexisting_state():
    analyzed, compressed = _real_compressed_decisive_fixture(
        [
            ("sed -i s/x/y/ /tmp/fix.sed", 0),
            ("sed -i -f /tmp/fix.sed src/lib.rs", 0),
            ("cargo test", 0),
        ],
        patch=_patch(("src/lib.rs", "@@ -1 +1 @@\n-a\n+b")),
    )
    assert rust_v3_builder._build_decisive_suffix(analyzed, compressed)[1][
        "reason"
    ] == "invalid_scratch_chain"


def test_decisive_suffix_rejects_failed_literal_scratch_creator():
    analyzed, compressed = _real_compressed_decisive_fixture(
        [
            ("cat > /tmp/fix.sed <<'EOF'\ns/a/b/\nEOF", 1),
            ("sed -i -f /tmp/fix.sed src/lib.rs", 0),
            ("cargo test", 0),
        ],
        patch=_patch(("src/lib.rs", "@@ -1 +1 @@\n-a\n+b")),
    )
    assert rust_v3_builder._build_decisive_suffix(analyzed, compressed)[1][
        "reason"
    ] == "invalid_scratch_chain"


@pytest.mark.parametrize(
    "edit",
    [
        "sed -i -f /tmp/fix.sed src/lib.rs",
        "sed -i -f/tmp/fix.sed src/lib.rs",
        "sed -i --file /tmp/fix.sed src/lib.rs",
        "sed -i --file=/tmp/fix.sed src/lib.rs",
    ],
)
def test_decisive_suffix_accepts_supported_sed_scratch_input_options(
    edit: str,
) -> None:
    analyzed, compressed = _real_compressed_decisive_fixture(
        [
            ("cat > /tmp/fix.sed <<'EOF'\ns/a/b/\nEOF", 0),
            (edit, 0),
            ("cargo test", 0),
        ],
        patch=_patch(("src/lib.rs", "@@ -1 +1 @@\n-a\n+b")),
    )
    result, report = rust_v3_builder._build_decisive_suffix(analyzed, compressed)
    assert report == {"kept": True}
    assert result is not None


def test_decisive_suffix_rejects_literal_scratch_input_without_original_creator():
    analyzed, compressed = _decisive_fixture(
        [
            ("sed -i -f /tmp/missing.sed src/lib.rs", 0),
            ("cargo test", 0),
        ],
        patch=_patch(("src/lib.rs", "@@ -1 +1 @@\n-a\n+b")),
    )
    assert rust_v3_builder._build_decisive_suffix(analyzed, compressed)[1][
        "reason"
    ] == "invalid_scratch_chain"


def test_decisive_suffix_scans_dropped_pre_edit_nonallowlisted_mutation():
    analyzed, compressed = _real_compressed_decisive_fixture(
        [
            ("cat > README_FIX.md <<'EOF'\nsummary\nEOF", 0),
            ("sed -i s/a/b/ src/lib.rs", 0),
            ("cargo test", 0),
        ],
        patch=_patch(("src/lib.rs", "@@ -1 +1 @@\n-a\n+b")),
    )
    assert rust_v3_builder._build_decisive_suffix(analyzed, compressed)[1][
        "reason"
    ] == "nonallowlisted_repository_mutation"


def test_decisive_suffix_rejects_safe_compressed_messages_for_unsafe_analyzed_row():
    safe_analyzed, safe_compressed = _real_compressed_decisive_fixture(
        [("sed -i s/a/b/ src/lib.rs", 0), ("cargo test", 0)],
        patch=_patch(("src/lib.rs", "@@ -1 +1 @@\n-a\n+b")),
    )
    unsafe_analyzed, _unsafe_compressed = _real_compressed_decisive_fixture(
        [
            ("cat > README_FIX.md <<'EOF'\nsummary\nEOF", 0),
            ("sed -i s/a/b/ src/lib.rs", 0),
            ("cargo test", 0),
        ],
        patch=_patch(("src/lib.rs", "@@ -1 +1 @@\n-a\n+b")),
    )
    assert safe_analyzed.identity == unsafe_analyzed.identity
    assert rust_v3_builder._build_decisive_suffix(
        unsafe_analyzed, safe_compressed
    )[1]["reason"] == "invalid_compressed_provenance"


def test_decisive_suffix_rejects_unsafe_compressed_messages_for_safe_analyzed_row():
    safe_analyzed, _safe_compressed = _real_compressed_decisive_fixture(
        [("sed -i s/a/b/ src/lib.rs", 0), ("cargo test", 0)],
        patch=_patch(("src/lib.rs", "@@ -1 +1 @@\n-a\n+b")),
    )
    _unsafe_analyzed, unsafe_compressed = _real_compressed_decisive_fixture(
        [
            ("cat > README_FIX.md <<'EOF'\nsummary\nEOF", 0),
            ("sed -i s/a/b/ src/lib.rs", 0),
            ("cargo test", 0),
        ],
        patch=_patch(("src/lib.rs", "@@ -1 +1 @@\n-a\n+b")),
    )
    assert rust_v3_builder._build_decisive_suffix(
        safe_analyzed, unsafe_compressed
    )[1]["reason"] == "invalid_compressed_provenance"


def test_decisive_suffix_rejects_compressed_identity_mismatch():
    analyzed, compressed = _real_compressed_decisive_fixture(
        [("sed -i s/a/b/ src/lib.rs", 0), ("cargo test", 0)],
        patch=_patch(("src/lib.rs", "@@ -1 +1 @@\n-a\n+b")),
    )
    mismatched = rust_v3_builder.CompressedAgenticRow(
        identity=rust_v3_builder.SourceIdentity(
            task_id=compressed.identity.task_id,
            repository=compressed.identity.repository,
            trajectory_id="different-trajectory",
            config=compressed.identity.config,
            split=compressed.identity.split,
        ),
        messages=compressed.messages,
        original_message_indices=compressed.original_message_indices,
        original_messages=compressed.original_messages,
        original_suffix_start=compressed.original_suffix_start,
        compressed_suffix_start=compressed.compressed_suffix_start,
    )
    assert rust_v3_builder._build_decisive_suffix(analyzed, mismatched)[1][
        "reason"
    ] == "invalid_compressed_provenance"


@pytest.mark.parametrize(
    "commands",
    [
        [
            ("cat > /tmp/fix.sed <<'EOF'\ns/a/b/\nEOF", 0),
            ("cat > /tmp/fix.sed <<'EOF'\ns/a/c/\nEOF", 0),
            ("sed -i -f /tmp/fix.sed src/lib.rs", 0),
            ("cargo test", 0),
        ],
        [
            ("cat > /tmp/unused.sed <<'EOF'\ns/a/b/\nEOF", 0),
            ("sed -i s/a/b/ src/lib.rs", 0),
            ("cargo test", 0),
        ],
        [
            ("sed -i s/a/b/ src/lib.rs", 0),
            ("cat > /tmp/late.sed <<'EOF'\ns/a/b/\nEOF", 0),
            ("cargo test", 0),
        ],
        [
            ("cat > /tmp/one.sed <<'EOF'\ns/a/b/\nEOF", 0),
            ("cat > /tmp/two.sed <<'EOF'\ns/b/c/\nEOF", 0),
            ("sed -i -f /tmp/one.sed -f /tmp/two.sed src/lib.rs", 0),
            ("cargo test", 0),
        ],
    ],
)
def test_decisive_suffix_rejects_invalid_scratch_chains(
    commands: list[tuple[str, int]],
) -> None:
    analyzed, compressed = _decisive_fixture(
        commands,
        patch=_patch(("src/lib.rs", "@@ -1 +1 @@\n-a\n+b")),
    )
    assert rust_v3_builder._build_decisive_suffix(analyzed, compressed)[1][
        "reason"
    ] == "invalid_scratch_chain"


def test_decisive_suffix_rejects_ambiguous_scratch_mutation():
    analyzed, compressed = _decisive_fixture(
        [
            ("cat > $SCRATCH <<'EOF'\ns/a/b/\nEOF", 0),
            ("sed -i s/a/b/ src/lib.rs", 0),
            ("cargo test", 0),
        ],
        patch=_patch(("src/lib.rs", "@@ -1 +1 @@\n-a\n+b")),
    )
    assert rust_v3_builder._build_decisive_suffix(analyzed, compressed)[1][
        "reason"
    ] == "ambiguous_mutation_path"


def test_decisive_suffix_preserves_exact_pairs_marker_and_one_following_git_read():
    analyzed, compressed = _decisive_fixture(
        [
            ("sed -i s/a/b/ src/lib.rs", 0),
            ("cargo test", 0),
            ("git diff -- src/lib.rs", 0),
            ("git status --short", 0),
        ],
        patch=_patch(("src/lib.rs", "@@ -1 +1 @@\n-a\n+b")),
    )
    result, report = rust_v3_builder._build_decisive_suffix(analyzed, compressed)
    assert report == {"kept": True}
    assert result is not None
    expected = compressed.messages[: compressed.compressed_suffix_start + 6] + (
        compressed.messages[-1],
    )
    assert result.messages == expected
    assert result.messages is not compressed.messages
    for actual, original in zip(result.messages, expected):
        assert actual == original
        assert actual is not original


def test_decisive_suffix_rejects_redirected_following_git_read_as_scratch_mutation():
    analyzed, compressed = _decisive_fixture(
        [
            ("sed -i s/a/b/ src/lib.rs", 0),
            ("cargo test", 0),
            ("git diff -- src/lib.rs > /tmp/final.diff", 0),
        ],
        patch=_patch(("src/lib.rs", "@@ -1 +1 @@\n-a\n+b")),
    )
    result, report = rust_v3_builder._build_decisive_suffix(analyzed, compressed)
    assert result is None
    assert report["reason"] == "invalid_scratch_chain"


def test_decisive_suffix_reruns_balanced_pairing_on_retained_messages():
    analyzed, compressed = _decisive_fixture(
        [("sed -i s/a/b/ src/lib.rs", 0), ("cargo test", 0)],
        patch=_patch(("src/lib.rs", "@@ -1 +1 @@\n-a\n+b")),
    )
    messages = list(compressed.messages)
    messages.insert(-3, {"role": "user", "content": "stray", "tool_calls": []})
    corrupted = rust_v3_builder.CompressedAgenticRow(
        identity=compressed.identity,
        messages=tuple(messages),
        original_message_indices=compressed.original_message_indices,
        original_messages=compressed.original_messages,
        original_suffix_start=compressed.original_suffix_start,
        compressed_suffix_start=compressed.compressed_suffix_start,
    )
    assert rust_v3_builder._build_decisive_suffix(analyzed, corrupted)[1][
        "reason"
    ] == "invalid_native_pairing"


def test_decisive_suffix_reruns_terminal_last_invariant():
    analyzed, compressed = _decisive_fixture(
        [("sed -i s/a/b/ src/lib.rs", 0), ("cargo test", 0)],
        patch=_patch(("src/lib.rs", "@@ -1 +1 @@\n-a\n+b")),
    )
    corrupted = rust_v3_builder.CompressedAgenticRow(
        identity=compressed.identity,
        messages=compressed.messages
        + ({"role": "user", "content": "late", "tool_calls": []},),
        original_message_indices=compressed.original_message_indices,
        original_messages=compressed.original_messages,
        original_suffix_start=compressed.original_suffix_start,
        compressed_suffix_start=compressed.compressed_suffix_start,
    )
    assert rust_v3_builder._build_decisive_suffix(analyzed, corrupted)[1][
        "reason"
    ] == "invalid_native_pairing"


def test_decisive_suffix_reruns_first_edit_grounding_invariant():
    analyzed, compressed = _decisive_fixture(
        [("sed -i s/a/b/ src/lib.rs", 0), ("cargo test", 0)],
        patch=_patch(("src/lib.rs", "@@ -1 +1 @@\n-a\n+b")),
    )
    messages = compressed.messages[:2] + compressed.messages[4:]
    corrupted = rust_v3_builder.CompressedAgenticRow(
        identity=compressed.identity,
        messages=messages,
        original_message_indices=(
            compressed.original_message_indices[:2]
            + compressed.original_message_indices[4:]
        ),
        original_messages=compressed.original_messages,
        original_suffix_start=compressed.original_suffix_start,
        compressed_suffix_start=2,
    )
    assert rust_v3_builder._build_decisive_suffix(analyzed, corrupted)[1][
        "reason"
    ] == "first_edit_not_grounded"


def test_first_edit_requires_every_tracked_edited_path_to_be_grounded() -> None:
    row = _valid_rich_row()
    row["metadata"] = {
        "model_patch": {
            "patch": (
                "diff --git a/src/lib.rs b/src/lib.rs\n"
                "--- a/src/lib.rs\n+++ b/src/lib.rs\n@@ -1 +1 @@\n-old\n+new\n"
                "diff --git a/src/other.rs b/src/other.rs\n"
                "--- a/src/other.rs\n+++ b/src/other.rs\n@@ -1 +1 @@\n-old\n+new\n"
            )
        }
    }
    trajectory = row["trajectory"]
    assert isinstance(trajectory, list)
    trajectory[4] = _assistant(
        "edit-1",
        "execute_bash",
        {
            "command": (
                "sed -i s/old/new/ src/lib.rs; "
                "sed -i s/old/new/ src/other.rs"
            )
        },
    )
    analyzed = _analyzed_rich(row)
    compressed, report = compress_tracked_source_row(analyzed)
    assert compressed is None
    assert report == {
        "kept": False,
        "reason": "edited_paths_not_grounded",
        "ungrounded_paths": ["src/other.rs"],
    }


def test_empty_read_observation_cannot_ground_an_edit() -> None:
    row = _valid_rich_row()
    trajectory = row["trajectory"]
    assert isinstance(trajectory, list)
    trajectory[3] = _result("read-0", "  ")
    compressed, report = compress_tracked_source_row(_analyzed_rich(row))
    assert compressed is None
    assert report["reason"] == "edited_paths_not_grounded"


def test_rg_pattern_that_looks_like_tracked_path_does_not_ground() -> None:
    row = _valid_rich_row()
    trajectory = row["trajectory"]
    assert isinstance(trajectory, list)
    trajectory[2] = _assistant(
        "read-0",
        "execute_bash",
        {"command": "rg 'src/lib.rs' Cargo.toml"},
    )
    compressed, report = compress_tracked_source_row(_analyzed_rich(row))
    assert compressed is None
    assert report == {
        "kept": False,
        "reason": "edited_paths_not_grounded",
        "ungrounded_paths": ["src/lib.rs"],
    }


def test_rg_file_operand_after_pattern_grounds_tracked_path() -> None:
    row = _valid_rich_row()
    trajectory = row["trajectory"]
    assert isinstance(trajectory, list)
    trajectory[2] = _assistant(
        "read-0",
        "execute_bash",
        {"command": "rg old src/lib.rs"},
    )
    compressed, report = compress_tracked_source_row(_analyzed_rich(row))
    assert compressed is not None
    assert report["kept"] is True


def test_no_tool_assistant_turn_is_rejected() -> None:
    row = _valid_rich_row()
    trajectory = row["trajectory"]
    assert isinstance(trajectory, list)
    trajectory.insert(4, {"role": "assistant", "content": "I will think more", "tool_calls": []})
    converted, report = convert_rich_trajectory(tuple(trajectory))
    assert converted is None
    assert report == {"kept": False, "reason": "no_tool_assistant"}


def _canonical_think_sequence(
    *, ack_role: str = "tool", ack_content: str = "Your thought has been logged."
) -> list[dict[str, object]]:
    ack: dict[str, object] = {
        "role": ack_role,
        "tool_call_id": "think-1",
        "content": ack_content,
    }
    return [
        {
            "role": "assistant",
            "content": "raw content after thought",
            "reasoning_content": "",
            "think": "",
            "tool_calls": [
                _call("think-1", "think", {"thought": "canonical thought text"})
            ],
        },
        ack,
        _assistant(
            "edit-1",
            "str_replace_editor",
            {
                "command": "str_replace",
                "path": "/testbed/src/lib.rs",
                "old_str": "old",
                "new_str": "new",
            },
            reasoning="next executable reasoning",
        ),
        _result("edit-1", "edited"),
    ]


@pytest.mark.parametrize(
    ("ack_role", "ack_content"),
    [
        ("tool", "Your thought has been logged."),
        ("user", "OBSERVATION:\nYour thought has been logged."),
    ],
)
def test_canonical_openhands_think_ack_merges_into_next_executable(
    ack_role: str, ack_content: str
) -> None:
    row = _valid_rich_row(pre_reads=1)
    trajectory = row["trajectory"]
    assert isinstance(trajectory, list)
    # Replace the original edit pair with the observed OpenHands four-event fixture.
    trajectory[4:6] = _canonical_think_sequence(
        ack_role=ack_role, ack_content=ack_content
    )

    converted, report = convert_rich_trajectory(tuple(trajectory))

    assert report == {"kept": True}
    assert converted is not None
    edit_pair = next(pair for pair in converted.tool_pairs if pair.command.startswith("python3"))
    assert edit_pair.raw_assistant_index == 6
    assistant = converted.messages[edit_pair.assistant_index]
    assert assistant["content"] == (
        "canonical thought text\n\nraw content after thought\n\n"
        "next executable reasoning"
    )
    assert assistant["tool_calls"][0]["id"] == "edit-1"
    assert converted.messages[edit_pair.observation_index]["content"] == (
        "OBSERVATION:\nedited"
    )
    assert all(
        message.get("tool_calls")
        for message in converted.messages
        if message["role"] == "assistant"
    )


def test_canonical_think_normalization_preserves_tracked_edit_and_compression() -> None:
    row = _valid_rich_row(pre_reads=1)
    trajectory = row["trajectory"]
    assert isinstance(trajectory, list)
    trajectory[4:6] = _canonical_think_sequence()

    analyzed, report = analyze_source_row(row, config="cfg", split="train")
    assert report == {"kept": True}
    assert analyzed is not None
    assert analyzed.tracked_mutations[0].message_index == 6
    compressed, compression_report = compress_tracked_source_row(analyzed)
    assert compression_report["kept"] is True
    assert compressed is not None
    edit = compressed.messages[compressed.compressed_suffix_start]
    assert edit["content"].startswith(
        "canonical thought text\n\nraw content after thought"
    )


def test_canonical_think_ack_may_omit_tool_call_id_like_normal_results() -> None:
    row = _valid_rich_row(pre_reads=1)
    trajectory = row["trajectory"]
    assert isinstance(trajectory, list)
    sequence = _canonical_think_sequence()
    sequence[1].pop("tool_call_id")
    trajectory[4:6] = sequence

    converted, report = convert_rich_trajectory(tuple(trajectory))

    assert report == {"kept": True}
    assert converted is not None


def test_canonical_think_ack_present_mismatched_tool_call_id_is_rejected() -> None:
    row = _valid_rich_row(pre_reads=1)
    trajectory = row["trajectory"]
    assert isinstance(trajectory, list)
    sequence = _canonical_think_sequence()
    sequence[1]["tool_call_id"] = "wrong-think-id"
    trajectory[4:6] = sequence

    converted, report = convert_rich_trajectory(tuple(trajectory))

    assert converted is None
    assert report == {"kept": False, "reason": "noncanonical_think_sequence"}


@pytest.mark.parametrize(
    "mutate",
    [
        lambda seq: seq[0].update(tool_calls=[]),
        lambda seq: seq[0]["tool_calls"][0].update(id=""),
        lambda seq: seq[0]["tool_calls"][0]["function"].update(
            arguments=json.dumps({"thought": ""})
        ),
        lambda seq: seq[0].update(reasoning_content="top reasoning forbidden"),
        lambda seq: seq[0].update(think="top think forbidden"),
        lambda seq: seq[1].update(tool_call_id="wrong-think-id"),
        lambda seq: seq[1].update(content="Thought logged, maybe."),
        lambda seq: seq[1].update(
            tool_calls=[_call("hidden", "execute_bash", {"command": "rm -rf src"})]
        ),
        lambda seq: seq[1].update(reasoning_content="hidden ack reasoning"),
        lambda seq: seq[1].update(extra_payload={"substantive": True}),
        lambda seq: seq.pop(1),
        lambda seq: seq[2].update(
            tool_calls=[_call("finish-next", "finish", {"message": "done"})]
        ),
        lambda seq: seq[2].update(
            tool_calls=[_call("think-next", "think", {"thought": "again"})]
        ),
        lambda seq: seq[2]["tool_calls"].append(
            _call("extra", "execute_bash", {"command": "git status"})
        ),
        lambda seq: seq[3].update(tool_call_id="wrong-edit-id"),
    ],
    ids=[
        "no-think-call",
        "missing-think-id",
        "empty-thought",
        "top-reasoning",
        "top-think",
        "ack-id-mismatch",
        "ack-text-mismatch",
        "ack-hidden-executable",
        "ack-hidden-reasoning",
        "ack-extra-substantive-field",
        "missing-ack",
        "next-terminal",
        "next-think",
        "next-mixed",
        "next-result-mismatch",
    ],
)
def test_noncanonical_openhands_think_variants_fail_closed(mutate) -> None:
    row = _valid_rich_row(pre_reads=1)
    trajectory = row["trajectory"]
    assert isinstance(trajectory, list)
    sequence = _canonical_think_sequence()
    mutate(sequence)
    trajectory[4:6] = sequence

    converted, report = convert_rich_trajectory(tuple(trajectory))

    assert converted is None
    assert report["kept"] is False


def test_canonical_think_at_trajectory_tail_fails_closed() -> None:
    sequence = _canonical_think_sequence()[:2]
    converted, report = convert_rich_trajectory(tuple(sequence))
    assert converted is None
    assert report == {"kept": False, "reason": "noncanonical_think_sequence"}


def test_orphan_tool_result_is_rejected() -> None:
    row = _valid_rich_row()
    trajectory = row["trajectory"]
    assert isinstance(trajectory, list)
    trajectory.insert(2, _result("ghost"))
    converted, report = convert_rich_trajectory(tuple(trajectory))
    assert converted is None
    assert report == {"kept": False, "reason": "orphan_tool_result"}


def test_mismatched_tool_result_is_rejected() -> None:
    row = _valid_rich_row()
    trajectory = row["trajectory"]
    assert isinstance(trajectory, list)
    trajectory[3] = _result("wrong")
    converted, report = convert_rich_trajectory(tuple(trajectory))
    assert converted is None
    assert report == {"kept": False, "reason": "mismatched_tool_result"}


def test_tool_call_without_immediate_result_is_rejected() -> None:
    row = _valid_rich_row()
    trajectory = row["trajectory"]
    assert isinstance(trajectory, list)
    del trajectory[3]
    converted, report = convert_rich_trajectory(tuple(trajectory))
    assert converted is None
    assert report == {"kept": False, "reason": "orphan_tool_call"}


def test_consecutive_identical_commands_are_rejected() -> None:
    row = _valid_rich_row(pre_reads=2)
    trajectory = row["trajectory"]
    assert isinstance(trajectory, list)
    trajectory[4] = _assistant("read-1", "execute_bash", {"command": "cat src/lib.rs # read 0"})
    compressed, report = compress_tracked_source_row(_analyzed_rich(row))
    assert compressed is None
    assert report == {"kept": False, "reason": "consecutive_identical_commands"}


def test_read_streak_above_five_is_rejected() -> None:
    row = _valid_rich_row()
    trajectory = row["trajectory"]
    assert isinstance(trajectory, list)
    post_edit_reads: list[dict[str, object]] = []
    for index in range(6):
        post_edit_reads.extend(
            [
                _assistant(
                    f"post-read-{index}",
                    "execute_bash",
                    {"command": f"cat src/lib.rs # post {index}"},
                ),
                _result(f"post-read-{index}"),
            ]
        )
    trajectory[6:6] = post_edit_reads
    compressed, report = compress_tracked_source_row(_analyzed_rich(row))
    assert compressed is None
    assert report == {"kept": False, "reason": "max_read_streak_exceeded"}


def test_long_pre_edit_read_streak_is_compressed_before_streak_gate() -> None:
    compressed, report = compress_tracked_source_row(
        _analyzed_rich(_valid_rich_row(pre_reads=6))
    )
    assert compressed is not None
    assert report["kept"] is True
    assert report["kept_pre_edit_reads"] == 3
    assert report["first_edit_command_index"] == 4


def test_post_edit_rust_verification_is_required() -> None:
    row = _valid_rich_row()
    trajectory = row["trajectory"]
    assert isinstance(trajectory, list)
    trajectory[6] = _assistant("verify-1", "execute_bash", {"command": "git diff"})
    compressed, report = compress_tracked_source_row(_analyzed_rich(row))
    assert compressed is None
    assert report == {"kept": False, "reason": "missing_post_edit_rust_verification"}


def test_terminal_finish_is_required() -> None:
    row = _valid_rich_row()
    trajectory = row["trajectory"]
    assert isinstance(trajectory, list)
    del trajectory[-1]
    compressed, report = compress_tracked_source_row(_analyzed_rich(row))
    assert compressed is None
    assert report == {"kept": False, "reason": "missing_terminal_finish"}


def test_finish_must_be_terminal() -> None:
    row = _valid_rich_row()
    trajectory = row["trajectory"]
    assert isinstance(trajectory, list)
    trajectory.extend(
        [
            _assistant("late", "execute_bash", {"command": "git status"}),
            _result("late"),
        ]
    )
    converted, report = convert_rich_trajectory(tuple(trajectory))
    assert converted is None
    assert report == {"kept": False, "reason": "finish_not_terminal"}


def test_corrected_submit_is_omitted_and_only_final_submit_renders_marker() -> None:
    row = _valid_rich_row()
    trajectory = row["trajectory"]
    assert isinstance(trajectory, list)
    trajectory[6:6] = [
        _assistant(
            "submit-rejected",
            "submit",
            {"message": "premature"},
            reasoning="premature-submit-reasoning",
        ),
        {
            "role": "tool",
            # Real OpenHands corrections can be adjacent without an ID.
            "content": "Submission rejected: marker must be the first line.",
        },
    ]

    analyzed = _analyzed_rich(row)
    converted, report = convert_rich_trajectory(analyzed.trajectory)

    assert report == {"kept": True}
    assert converted is not None
    marker_calls = [
        call
        for message in converted.messages
        for call in message.get("tool_calls", [])
        if json.loads(call["function"]["arguments"])["command"]
        == "echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT"
    ]
    assert len(marker_calls) == 1
    assert marker_calls[0]["id"] == "finish-1"
    assert all(
        "premature-submit-reasoning" not in str(message.get("content"))
        for message in converted.messages
    )

    compressed, compression_report = compress_tracked_source_row(analyzed)
    assert compression_report["kept"] is True
    assert compressed is not None
    assert compressed.messages[compressed.compressed_suffix_start :] == (
        converted.messages[compressed.original_suffix_start :]
    )


def test_final_finish_is_native_bash_marker_without_observation() -> None:
    converted, report = convert_rich_trajectory(
        _analyzed_rich(_valid_rich_row()).trajectory
    )
    assert report == {"kept": True}
    assert converted is not None
    final = converted.messages[-1]
    assert final["role"] == "assistant"
    assert final["content"] == "finish-reasoning"
    assert len(final["tool_calls"]) == 1
    assert json.loads(final["tool_calls"][0]["function"]["arguments"]) == {
        "command": "echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT"
    }
    assert converted.terminal_index == len(converted.messages) - 1


def test_multiple_executable_rich_tools_in_one_turn_are_rejected() -> None:
    row = _valid_rich_row()
    trajectory = row["trajectory"]
    assert isinstance(trajectory, list)
    assistant = trajectory[2]
    assert isinstance(assistant, dict)
    calls = assistant["tool_calls"]
    assert isinstance(calls, list)
    calls.append(_call("read-extra", "execute_bash", {"command": "git status"}))
    converted, report = convert_rich_trajectory(tuple(trajectory))
    assert converted is None
    assert report == {"kept": False, "reason": "multiple_executable_tool_calls"}


def _exclusion_file(path: Path, ids: list[str]) -> Path:
    path.write_text("".join(json.dumps({"instance_id": value}) + "\n" for value in ids))
    return path


def _build_inputs(exclusion: Path, ids: list[str], *, boundary: int) -> BuildInputs:
    template = "{{ messages | tojson }}"
    return BuildInputs(
        dataset_revision="revision-abc",
        tokenizer_identity="gemma-tokenizer@abc",
        tokenizer_revision="0123456789abcdef0123456789abcdef01234567",
        tokenizer_artifact_sha256="a" * 64,
        template_identity="gemma-native@abc",
        template_content=template,
        expected_template_sha256=hashlib.sha256(template.encode()).hexdigest(),
        exclusion_path=exclusion,
        expected_exclusion_count=len(ids),
        expected_exclusion_sha256=canonical_id_sha256(ids),
        expected_pre_exclusion_eligible=boundary,
        max_tokens=49_152,
        max_workers=2,
    )


def _fake_hf_publish(rows: list[dict[str, object]], path: Path) -> None:
    path.mkdir()
    (path / "rows.json").write_text(json.dumps(rows, sort_keys=True))


def test_streaming_build_validates_frozen_exclusion_before_consuming_rows(
    tmp_path: Path,
) -> None:
    consumed = False

    def rows():
        nonlocal consumed
        consumed = True
        yield _valid_rich_row()

    exclusion = _exclusion_file(tmp_path / "exclude.jsonl", ["held-out"])
    inputs = _build_inputs(exclusion, ["different-id"], boundary=1)

    with pytest.raises(ValueError, match="exclusion SHA256"):
        build_streaming_dataset(
            [SourceRowStream("open-swe", "cfg", "train", rows())],
            output_dir=tmp_path / "out",
            inputs=inputs,
            count_rendered_tokens=lambda messages: len(json.dumps(messages)),
            publish_hf=_fake_hf_publish,
        )

    assert consumed is False
    assert not (tmp_path / "out").exists()


def test_streaming_build_selects_shortest_per_task_then_content_dedups_and_budgets(
    tmp_path: Path,
) -> None:
    shortest = _valid_rich_row(pre_reads=1)
    shortest["trajectory_id"] = "z-short"
    longer = _valid_rich_row(pre_reads=3)
    longer["trajectory_id"] = "a-long"
    duplicate = _valid_rich_row(pre_reads=1)
    duplicate["instance_id"] = "owner__crate-duplicate"
    duplicate["trajectory_id"] = "duplicate"
    over_budget = _valid_rich_row(pre_reads=1)
    over_budget["instance_id"] = "owner__crate-huge"
    over_budget["trajectory_id"] = "huge"
    over_budget["trajectory"][1]["content"] = "X" * 60_000
    excluded = _valid_rich_row(pre_reads=1)
    excluded["instance_id"] = "held-out"
    exclusion = _exclusion_file(tmp_path / "exclude.jsonl", ["held-out"])

    manifest = build_streaming_dataset(
        [
            SourceRowStream(
                "open-swe",
                "cfg-a",
                "train",
                iter([longer, shortest, duplicate, over_budget, excluded]),
            )
        ],
        output_dir=tmp_path / "out",
        inputs=_build_inputs(exclusion, ["held-out"], boundary=4),
        count_rendered_tokens=lambda messages: len(json.dumps(messages)),
        publish_hf=_fake_hf_publish,
    )

    rows = [json.loads(line) for line in (tmp_path / "out/dataset.jsonl").read_text().splitlines()]
    assert [(row["task_id"], row["trajectory_id"]) for row in rows] == [
        ("owner__crate-1", "z-short")
    ]
    assert rows[0]["tokens"] <= 49_152
    assert rows[0]["content_sha256"] == hashlib.sha256(
        json.dumps(rows[0]["messages"], ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    assert manifest["rows_output"] == 1
    assert manifest["drop_reasons"] == {
        "content_duplicate": 1,
        "excluded_task_id": 1,
        "task_shorter_candidate": 1,
        "token_budget_exceeded": 1,
    }
    assert manifest["per_config"]["open-swe/cfg-a/train"]["scanned"] == 5
    assert manifest["per_config"]["open-swe/cfg-a/train"]["excluded"] == 1
    assert manifest["exclusion"]["count"] == 1
    assert manifest["input_boundary"]["actual_pre_exclusion_eligible"] == 4
    assert manifest["token_stats"]["max"] == rows[0]["tokens"]
    assert manifest["per_repo"] == {"owner/crate": 1}
    assert (tmp_path / "out/hf_dataset/rows.json").is_file()


def test_streaming_build_is_deterministic_across_stream_order(tmp_path: Path) -> None:
    exclusion = _exclusion_file(tmp_path / "exclude.jsonl", [])
    row_a = _valid_rich_row()
    row_a["trajectory_id"] = "trajectory-a"
    row_b = _valid_rich_row()
    row_b["trajectory_id"] = "trajectory-b"
    # Equal rendered lengths make source/config/split/content hash the tie-breaker.
    manifests = []
    outputs = []
    for index, rows in enumerate(([row_b, row_a], [row_a, row_b])):
        out = tmp_path / f"out-{index}"
        manifests.append(
            build_streaming_dataset(
                [SourceRowStream("open-swe", "cfg", "train", iter(rows))],
                output_dir=out,
                inputs=_build_inputs(exclusion, [], boundary=2),
                count_rendered_tokens=lambda messages: len(json.dumps(messages)),
                publish_hf=_fake_hf_publish,
            )
        )
        outputs.append((out / "dataset.jsonl").read_bytes())

    assert outputs[0] == outputs[1]
    assert manifests[0]["output_rows_sha256"] == manifests[1]["output_rows_sha256"]


def test_streaming_build_uses_bounded_shared_counter_and_fails_boundary_closed(
    tmp_path: Path,
) -> None:
    exclusion = _exclusion_file(tmp_path / "exclude.jsonl", [])
    lock = threading.Lock()
    active = 0
    peak = 0
    caller = threading.get_ident()
    seen_threads: set[int] = set()

    def counter(messages: list[dict[str, object]]) -> int:
        nonlocal active, peak
        with lock:
            active += 1
            peak = max(peak, active)
            seen_threads.add(threading.get_ident())
        time.sleep(0.005)
        with lock:
            active -= 1
        return len(json.dumps(messages))

    rows = []
    for index in range(8):
        row = _valid_rich_row()
        row["instance_id"] = f"owner__crate-{index}"
        row["trajectory_id"] = f"trajectory-{index}"
        rows.append(row)

    with pytest.raises(ValueError, match="pre-exclusion eligible boundary"):
        build_streaming_dataset(
            [SourceRowStream("open-swe", "cfg", "train", iter(rows))],
            output_dir=tmp_path / "out",
            inputs=_build_inputs(exclusion, [], boundary=7),
            count_rendered_tokens=counter,
            publish_hf=_fake_hf_publish,
        )

    # Boundary assertion happens before token work and publication.
    assert seen_threads == set()
    assert peak == 0
    assert caller not in seen_threads
    assert not (tmp_path / "out").exists()


def test_streaming_build_transaction_leaves_no_output_on_hf_failure(tmp_path: Path) -> None:
    exclusion = _exclusion_file(tmp_path / "exclude.jsonl", [])

    def fail_publish(rows: list[dict[str, object]], path: Path) -> None:
        path.mkdir()
        raise RuntimeError("publisher failed")

    with pytest.raises(RuntimeError, match="publisher failed"):
        build_streaming_dataset(
            [SourceRowStream("open-swe", "cfg", "train", iter([_valid_rich_row()]))],
            output_dir=tmp_path / "out",
            inputs=_build_inputs(exclusion, [], boundary=1),
            count_rendered_tokens=lambda messages: len(json.dumps(messages)),
            publish_hf=fail_publish,
        )

    assert not (tmp_path / "out").exists()
    assert not list(tmp_path.glob(".out.tmp-*"))


def test_streaming_manifest_round_trips_hash_arithmetic_and_bounded_workers(
    tmp_path: Path,
) -> None:
    exclusion = _exclusion_file(tmp_path / "exclude.jsonl", [])
    lock = threading.Lock()
    active = peak = 0

    def counter(messages: list[dict[str, object]]) -> int:
        nonlocal active, peak
        with lock:
            active += 1
            peak = max(peak, active)
        time.sleep(0.01)
        with lock:
            active -= 1
        return len(json.dumps(messages))

    rows = []
    for index in range(6):
        row = _valid_rich_row()
        row["instance_id"] = f"owner__crate-{index}"
        row["trajectory_id"] = f"trajectory-{index}"
        row["trajectory"][1]["content"] = f"Fix Rust bug {index}."
        rows.append(row)
    out = tmp_path / "out"
    manifest = build_streaming_dataset(
        [SourceRowStream("open-swe", "cfg", "train", iter(rows))],
        output_dir=out,
        inputs=_build_inputs(exclusion, [], boundary=6),
        count_rendered_tokens=counter,
        publish_hf=_fake_hf_publish,
    )

    on_disk = json.loads((out / "manifest.json").read_text())
    assert on_disk == manifest
    assert manifest["output_jsonl_sha256"] == hashlib.sha256(
        (out / "dataset.jsonl").read_bytes()
    ).hexdigest()
    assert manifest["arithmetic"] == {
        "rows_scanned": 6,
        "rows_dropped": 0,
        "rows_output": 6,
        "balanced": True,
    }
    assert 1 < peak <= 2

    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        build_streaming_dataset(
            [],
            output_dir=out,
            inputs=_build_inputs(exclusion, [], boundary=0),
            count_rendered_tokens=counter,
            publish_hf=_fake_hf_publish,
        )


def test_exclusion_precedes_structural_analysis_and_counts_malformed_heldout(
    tmp_path: Path,
) -> None:
    exclusion = _exclusion_file(tmp_path / "exclude.jsonl", ["held-out"])
    malformed = _valid_rich_row()
    malformed["instance_id"] = "held-out"
    malformed["metadata"] = {"model_patch": {"patch": "not a patch"}}

    manifest = build_streaming_dataset(
        [SourceRowStream("open-swe", "cfg", "train", iter([malformed]))],
        output_dir=tmp_path / "out",
        inputs=_build_inputs(exclusion, ["held-out"], boundary=0),
        count_rendered_tokens=lambda messages: len(json.dumps(messages)),
        publish_hf=_fake_hf_publish,
    )

    counters = manifest["per_config"]["open-swe/cfg/train"]
    assert counters["excluded"] == 1
    assert counters["structurally_eligible"] == 0
    assert manifest["drop_reasons"] == {"excluded_task_id": 1}
    assert manifest["input_boundary"]["actual_pre_exclusion_eligible"] == 0


def test_excluded_malformed_nonrust_unresolved_rows_cannot_satisfy_boundary(
    tmp_path: Path,
) -> None:
    held_ids = ["held-malformed", "held-nonrust", "held-unresolved"]
    exclusion = _exclusion_file(tmp_path / "exclude.jsonl", held_ids)
    malformed = _valid_rich_row()
    malformed["instance_id"] = held_ids[0]
    malformed["metadata"] = None
    nonrust = _valid_rich_row()
    nonrust["instance_id"] = held_ids[1]
    nonrust["language"] = "python"
    unresolved = _valid_rich_row()
    unresolved["instance_id"] = held_ids[2]
    unresolved["resolved"] = "0"
    valid = _valid_rich_row()
    valid["instance_id"] = "owner__valid"

    manifest = build_streaming_dataset(
        [
            SourceRowStream(
                "open-swe", "cfg", "train", iter([malformed, nonrust, unresolved, valid])
            )
        ],
        output_dir=tmp_path / "out",
        inputs=_build_inputs(exclusion, held_ids, boundary=1),
        count_rendered_tokens=lambda messages: len(json.dumps(messages)),
        publish_hf=_fake_hf_publish,
    )

    counters = manifest["per_config"]["open-swe/cfg/train"]
    assert counters["excluded"] == 3
    assert counters["structurally_eligible"] == 1
    assert manifest["input_boundary"] == {
        "expected_pre_exclusion_eligible": 1,
        "actual_pre_exclusion_eligible": 1,
        "definition": "structurally eligible rows after frozen task-ID exclusion",
    }

    with pytest.raises(ValueError, match="pre-exclusion eligible boundary"):
        build_streaming_dataset(
            [SourceRowStream("open-swe", "cfg", "train", iter([malformed]))],
            output_dir=tmp_path / "bad-boundary",
            inputs=_build_inputs(exclusion, held_ids, boundary=1),
            count_rendered_tokens=lambda messages: 0,
            publish_hf=_fake_hf_publish,
        )


def test_stage_counters_reflect_the_actual_compression_gate_reached(tmp_path: Path) -> None:
    exclusion = _exclusion_file(tmp_path / "exclude.jsonl", [])
    no_verify = _valid_rich_row()
    no_verify["instance_id"] = "owner__no-verify"
    no_verify["trajectory_id"] = "no-verify"
    no_verify["trajectory"][-3]["tool_calls"][0]["function"]["arguments"] = json.dumps(
        {"command": "git status"}
    )
    no_grounding = _valid_rich_row(pre_reads=0)
    no_grounding["instance_id"] = "owner__no-grounding"
    no_grounding["trajectory_id"] = "no-grounding"
    repeated = _valid_rich_row(pre_reads=2)
    repeated["instance_id"] = "owner__repeated"
    repeated["trajectory_id"] = "repeated"
    for message in (repeated["trajectory"][2], repeated["trajectory"][4]):
        message["tool_calls"][0]["function"]["arguments"] = json.dumps(
            {"command": "cat src/lib.rs"}
        )
    success = _valid_rich_row()
    success["instance_id"] = "owner__success"
    success["trajectory_id"] = "success"

    manifest = build_streaming_dataset(
        [
            SourceRowStream(
                "open-swe", "cfg", "train", iter([no_verify, no_grounding, repeated, success])
            )
        ],
        output_dir=tmp_path / "out",
        inputs=_build_inputs(exclusion, [], boundary=4),
        count_rendered_tokens=lambda messages: len(json.dumps(messages)),
        publish_hf=_fake_hf_publish,
    )

    counters = manifest["per_config"]["open-swe/cfg/train"]
    assert {
        key: counters[key]
        for key in (
            "pairing_passed",
            "verification_passed",
            "grounding_passed",
            "repeat_read_streak_passed",
            "compressed_eligible",
        )
    } == {
        "pairing_passed": 4,
        "verification_passed": 3,
        "grounding_passed": 2,
        "repeat_read_streak_passed": 1,
        "compressed_eligible": 1,
    }
    assert manifest["drop_reasons"] == {
        "consecutive_identical_commands": 1,
        "edited_paths_not_grounded": 1,
        "missing_post_edit_rust_verification": 1,
    }


@pytest.mark.parametrize("max_tokens", [0, 49_153])
def test_build_rejects_token_caps_outside_frozen_49k_lane(
    tmp_path: Path, max_tokens: int
) -> None:
    exclusion = _exclusion_file(tmp_path / f"exclude-{max_tokens}.jsonl", [])
    inputs = _build_inputs(exclusion, [], boundary=0)
    inputs = BuildInputs(**{**inputs.__dict__, "max_tokens": max_tokens})
    with pytest.raises(ValueError, match="max_tokens"):
        build_streaming_dataset(
            [],
            output_dir=tmp_path / f"out-{max_tokens}",
            inputs=inputs,
            count_rendered_tokens=lambda messages: 0,
            publish_hf=_fake_hf_publish,
        )


def test_publish_never_replaces_an_arbitrary_output_created_during_build(
    tmp_path: Path,
) -> None:
    exclusion = _exclusion_file(tmp_path / "exclude.jsonl", [])
    out = tmp_path / "out"
    victim = tmp_path / "victim"
    victim.mkdir()
    sentinel = victim / "sentinel"
    sentinel.write_text("owned elsewhere")

    def racing_publish(rows: list[dict[str, object]], path: Path) -> None:
        _fake_hf_publish(rows, path)
        out.symlink_to(victim, target_is_directory=True)

    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        build_streaming_dataset(
            [SourceRowStream("open-swe", "cfg", "train", iter([_valid_rich_row()]))],
            output_dir=out,
            inputs=_build_inputs(exclusion, [], boundary=1),
            count_rendered_tokens=lambda messages: len(json.dumps(messages)),
            publish_hf=racing_publish,
        )

    assert out.is_symlink()
    assert sentinel.read_text() == "owned elsewhere"


def test_publication_installs_only_a_complete_directory_atomically(tmp_path: Path) -> None:
    exclusion = _exclusion_file(tmp_path / "exclude.jsonl", [])
    out = tmp_path / "out"

    def observing_publish(rows: list[dict[str, object]], path: Path) -> None:
        assert not out.exists()
        _fake_hf_publish(rows, path)
        assert not out.exists()

    build_streaming_dataset(
        [SourceRowStream("open-swe", "cfg", "train", iter([_valid_rich_row()]))],
        output_dir=out,
        inputs=_build_inputs(exclusion, [], boundary=1),
        count_rendered_tokens=lambda messages: len(json.dumps(messages)),
        publish_hf=observing_publish,
    )

    assert sorted(path.name for path in out.iterdir()) == [
        "dataset.jsonl",
        "hf_dataset",
        "manifest.json",
    ]


def test_atomic_publication_fails_closed_on_unsupported_platform(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    exclusion = _exclusion_file(tmp_path / "exclude.jsonl", [])
    out = tmp_path / "out"
    monkeypatch.setattr(rust_v3_builder.sys, "platform", "unsupported-test-os")

    with pytest.raises(RuntimeError, match="atomic no-replace rename unsupported"):
        build_streaming_dataset(
            [SourceRowStream("open-swe", "cfg", "train", iter([_valid_rich_row()]))],
            output_dir=out,
            inputs=_build_inputs(exclusion, [], boundary=1),
            count_rendered_tokens=lambda messages: len(json.dumps(messages)),
            publish_hf=_fake_hf_publish,
        )

    assert not out.exists()
    assert not list(tmp_path.glob(".out.tmp-*"))


def test_template_hash_is_recomputed_and_tokenizer_artifact_is_bound(
    tmp_path: Path,
) -> None:
    exclusion = _exclusion_file(tmp_path / "exclude.jsonl", [])
    valid = _build_inputs(exclusion, [], boundary=0)
    invalid = BuildInputs(
        **{**valid.__dict__, "expected_template_sha256": "f" * 64}
    )
    with pytest.raises(ValueError, match="template SHA256"):
        build_streaming_dataset(
            [],
            output_dir=tmp_path / "invalid",
            inputs=invalid,
            count_rendered_tokens=lambda messages: 0,
            publish_hf=_fake_hf_publish,
        )

    manifest = build_streaming_dataset(
        [],
        output_dir=tmp_path / "valid",
        inputs=valid,
        count_rendered_tokens=lambda messages: 0,
        publish_hf=_fake_hf_publish,
    )
    assert manifest["tokenizer"] == {
        "identity": valid.tokenizer_identity,
        "revision": valid.tokenizer_revision,
        "artifact_sha256": valid.tokenizer_artifact_sha256,
    }
    assert manifest["template"] == {
        "identity": valid.template_identity,
        "sha256": valid.expected_template_sha256,
    }


def test_tokenizer_artifact_hash_comes_from_saved_files() -> None:
    class FakeTokenizer:
        def __init__(self, payload: str) -> None:
            self.payload = payload

        def save_pretrained(self, path: Path) -> None:
            (path / "tokenizer.json").write_text(self.payload)
            (path / "tokenizer_config.json").write_text('{"model_max_length":49152}')

    first = tokenizer_artifact_sha256(FakeTokenizer("one"))
    same = tokenizer_artifact_sha256(FakeTokenizer("one"))
    changed = tokenizer_artifact_sha256(FakeTokenizer("two"))

    assert first == same
    assert first != changed
    assert len(first) == 64


def test_streaming_progress_reports_cadence_and_partition_final(tmp_path: Path) -> None:
    exclusion = _exclusion_file(tmp_path / "exclude.jsonl", [])
    emitted: list[str] = []
    ticks = iter([10.0, 12.0, 14.0, 15.0])

    build_streaming_dataset(
        [
            SourceRowStream(
                "open-swe",
                "cfg",
                "train",
                iter({"instance_id": f"invalid..{index}"} for index in range(5)),
            )
        ],
        output_dir=tmp_path / "out",
        inputs=_build_inputs(exclusion, [], boundary=0),
        count_rendered_tokens=lambda messages: 0,
        publish_hf=_fake_hf_publish,
        progress_every=2,
        progress_sink=emitted.append,
        monotonic=lambda: next(ticks),
    )

    reports = [json.loads(line) for line in emitted]
    assert [report["scanned"] for report in reports] == [2, 4, 5]
    assert [report["final"] for report in reports] == [False, False, True]
    assert reports[-1] == {
        "config": "cfg",
        "elapsed_seconds": 5.0,
        "event": "rust_v3_stream_progress",
        "final": True,
        "rate_rows_per_second": 1.0,
        "scanned": 5,
        "source": "open-swe",
        "split": "train",
    }


def test_streaming_progress_zero_disables_all_reporting(tmp_path: Path) -> None:
    exclusion = _exclusion_file(tmp_path / "exclude.jsonl", [])

    def forbidden(_line: str) -> None:
        raise AssertionError("disabled progress emitted output")

    build_streaming_dataset(
        [SourceRowStream("open-swe", "cfg", "train", iter([]))],
        output_dir=tmp_path / "out",
        inputs=_build_inputs(exclusion, [], boundary=0),
        count_rendered_tokens=lambda messages: 0,
        publish_hf=_fake_hf_publish,
        progress_every=0,
        progress_sink=forbidden,
    )


@pytest.mark.parametrize(
    "rendered",
    [
        [11, 12, 13],
        {"input_ids": [11, 12, 13], "attention_mask": [1, 1, 1]},
        {"input_ids": [[11, 12, 13]]},
    ],
)
def test_rendered_token_count_accepts_one_flat_example(rendered) -> None:
    assert rendered_token_count(rendered) == 3


def test_rendered_token_count_accepts_batchencoding_like_mapping() -> None:
    class BatchEncodingLike(dict):
        pass

    rendered = BatchEncodingLike(input_ids=[[1, 2, 3, 4]], attention_mask=[[1] * 4])
    assert rendered_token_count(rendered) == 4


@pytest.mark.parametrize(
    ("shape", "payload", "expected"),
    [
        ((3,), [1, 2, 3], 3),
        ((1, 3), [[1, 2, 3]], 3),
    ],
)
def test_rendered_token_count_accepts_single_example_array_shapes(
    shape: tuple[int, ...], payload, expected: int
) -> None:
    class ArrayLike:
        def __init__(self, value_shape, value) -> None:
            self.shape = value_shape
            self.value = value

        def tolist(self):
            return self.value

    assert rendered_token_count(ArrayLike(shape, payload)) == expected


@pytest.mark.parametrize(
    "rendered",
    [
        [],
        {"input_ids": []},
        {"input_ids": [[1], [2]]},
        {"attention_mask": [1, 1]},
        {"input_ids": "123"},
        {"input_ids": [True, 2]},
        {"input_ids": [[[1, 2]]]},
        [1, [2]],
    ],
)
def test_rendered_token_count_rejects_zero_multi_batch_missing_and_ambiguous(
    rendered,
) -> None:
    with pytest.raises(ValueError, match="rendered token IDs"):
        rendered_token_count(rendered)


def test_rendered_token_count_rejects_scalar_multibatch_and_rank3_arrays() -> None:
    class ArrayLike:
        def __init__(self, shape, value) -> None:
            self.shape = shape
            self.value = value

        def tolist(self):
            return self.value

    for rendered in (
        ArrayLike((), 1),
        ArrayLike((2, 2), [[1, 2], [3, 4]]),
        ArrayLike((1, 1, 2), [[[1, 2]]]),
        ArrayLike((0,), []),
    ):
        with pytest.raises(ValueError, match="rendered token IDs"):
            rendered_token_count(rendered)


def test_build_rejects_zero_from_injected_rendered_counter(tmp_path: Path) -> None:
    exclusion = _exclusion_file(tmp_path / "exclude.jsonl", [])
    with pytest.raises(ValueError, match="positive integer"):
        build_streaming_dataset(
            [SourceRowStream("open-swe", "cfg", "train", iter([_valid_rich_row()]))],
            output_dir=tmp_path / "out",
            inputs=_build_inputs(exclusion, [], boundary=1),
            count_rendered_tokens=lambda _messages: 0,
            publish_hf=_fake_hf_publish,
            progress_every=0,
        )
    assert not (tmp_path / "out").exists()


def test_audit_only_emits_mismatch_evidence_without_tokenizer_or_publisher(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    exclusion = _exclusion_file(tmp_path / "exclude.jsonl", [])
    monkeypatch.setattr(
        rust_v3_builder,
        "tokenizer_artifact_sha256",
        lambda _tokenizer: (_ for _ in ()).throw(AssertionError("tokenizer loaded")),
    )
    monkeypatch.setattr(
        rust_v3_builder,
        "_default_hf_publisher",
        lambda _rows, _path: (_ for _ in ()).throw(AssertionError("published")),
    )
    valid = _valid_rich_row()
    invalid = _valid_rich_row()
    invalid["instance_id"] = "owner__invalid"
    invalid["metadata"] = None

    report = audit_streaming_dataset(
        [SourceRowStream("open-swe", "cfg", "train", iter([valid, invalid]))],
        inputs=AuditInputs(
            dataset_revision="revision-abc",
            exclusion_path=exclusion,
            expected_exclusion_count=0,
            expected_exclusion_sha256=canonical_id_sha256([]),
            expected_pre_exclusion_eligible=736,
        ),
        progress_every=0,
    )

    assert report["status"] == "boundary_mismatch"
    assert report["input_boundary"]["expected_pre_exclusion_eligible"] == 736
    assert report["input_boundary"]["actual_pre_exclusion_eligible"] == 1
    assert report["totals"]["scanned"] == 2
    assert report["totals"]["structurally_eligible"] == 1
    assert report["per_config"]["open-swe/cfg/train"]["drop_reasons"] == {
        "missing_model_patch": 1
    }
    assert report["unique_structural_task_ids"] == 1
    assert report["unique_structural_repositories"] == 1


def test_audit_cli_args_do_not_require_build_only_tokenizer_template_or_out(
    tmp_path: Path,
) -> None:
    args = _parse_args(
        [
            "--audit-only",
            "--revision",
            "revision-abc",
            "--config",
            "cfg:train",
            "--exclude",
            str(tmp_path / "exclude.jsonl"),
            "--expected-exclusion-count",
            "239",
            "--expected-exclusion-sha256",
            "a" * 64,
            "--expected-pre-exclusion-eligible",
            "736",
        ]
    )

    assert args.audit_only is True
    assert args.tokenizer is None
    assert args.template_identity is None
    assert args.out is None


def test_audit_cli_emits_final_json_before_nonzero_boundary_exit(
    tmp_path: Path,
) -> None:
    exclusion = _exclusion_file(tmp_path / "exclude.jsonl", [])
    emitted: list[str] = []
    status = rust_v3_builder.run_audit_cli(
        [SourceRowStream("open-swe", "cfg", "train", iter([_valid_rich_row()]))],
        inputs=AuditInputs(
            dataset_revision="revision-abc",
            exclusion_path=exclusion,
            expected_exclusion_count=0,
            expected_exclusion_sha256=canonical_id_sha256([]),
            expected_pre_exclusion_eligible=736,
        ),
        emit=emitted.append,
        progress_every=0,
    )

    assert status == 2
    assert len(emitted) == 1
    evidence = json.loads(emitted[0])
    assert evidence["event"] == "rust_v3_audit_final"
    assert evidence["status"] == "boundary_mismatch"
    assert evidence["input_boundary"]["actual_pre_exclusion_eligible"] == 1
