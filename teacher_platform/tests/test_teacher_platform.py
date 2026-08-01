"""Unit tests for teacher_platform pure logic (no docker / no network)."""
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from teacher_platform import teacher_platform as tp
from teacher_platform.generic_trace_replay import assess_controls
from teacher_platform.trace_gate import TracePreflightError


def _trajectory_tool_call(name, arguments):
    return {"function": {"name": name, "arguments": json.dumps(arguments)}}


def _nested_quoted_shell_heredoc(depth):
    command = "printf nested-ok\n"
    for level in range(depth):
        delimiter = f"EOF{level}"
        command = f"bash <<'{delimiter}'\n{command}{delimiter}\n"
    return command.rstrip()


def _strict_control_set(task):
    contract = tp.build_task_contract(task)

    def run(command, passed):
        output = (
            "1 passed in 0.01s\n"
            if passed
            else "1 failed in 0.01s\n"
        )
        return {
            "command": command,
            "returncode": 0 if passed else 1,
            "strict_pass": passed,
            "output_sha256": tp.sha256_bytes(output.encode()),
            "output_tail": output,
        }

    def phase(name, f2p, p2p):
        protected = "f" * 64
        result = {
            "phase": name,
            "patch_applied": True,
            "f2p_pass": f2p,
            "p2p_pass": p2p,
            "f2p_runs": [run(command, f2p) for command in contract["f2p_commands"]],
            "p2p_runs": [run(command, p2p) for command in contract["p2p_commands"]],
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
                    ("candidate", tp.sha256_bytes(task["patch"].encode())),
                ],
                "candidate_2": [
                    ("mutation", contract["mutation_patch_sha256"]),
                    ("candidate", tp.sha256_bytes(task["patch"].encode())),
                ],
            }[name]
            result["patch_applications"] = [
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
            result["patch_apply"] = (
                None
                if name == "baseline"
                else {
                    "returncode": 0,
                    "output_sha256": "0" * 64,
                    "output_tail": "",
                }
            )
        return result

    return contract, {
        "baseline": phase("baseline", False, True),
        "reference": phase("reference", True, True),
        "candidate_1": phase("candidate_1", True, True),
        "candidate_2": phase("candidate_2", True, True),
    }


def test_openrouter_collector_rejects_untrainable_strict_trace(
    tmp_path,
    monkeypatch,
):
    class FakeCompletions:
        @staticmethod
        def create(**_kwargs):
            message = SimpleNamespace(content="DONE", tool_calls=[])
            return SimpleNamespace(
                choices=[SimpleNamespace(message=message)]
            )

    class FakeOpenAI:
        def __init__(self, **_kwargs):
            self.chat = SimpleNamespace(completions=FakeCompletions())

    monkeypatch.setitem(
        sys.modules,
        "openai",
        SimpleNamespace(OpenAI=FakeOpenAI),
    )
    monkeypatch.setenv("OPENROUTER_API_KEY", "fixture")

    def fake_sh(command, *_args, **_kwargs):
        if command[:3] == ["docker", "run", "-d"]:
            return "0123456789abcdef\n", 0
        if command[:3] == ["docker", "inspect", "-f"]:
            return "true\n", 0
        if command[:3] == ["docker", "rm", "-f"]:
            return "", 0
        raise AssertionError(command)

    monkeypatch.setattr(tp.ttd, "sh", fake_sh)
    monkeypatch.setattr(
        tp.ttd,
        "establish_mutation_baseline",
        lambda *_args: {
            "mutation_f2p_reproduced": True,
            "mutation_baseline_commit": "f" * 40,
        },
    )
    patch_text = "diff --git a/src/x.py b/src/x.py\n"
    monkeypatch.setattr(
        tp.ttd,
        "capture_worktree_patch",
        lambda *_args: patch_text,
    )
    monkeypatch.setattr(
        tp,
        "verify_candidate_patch",
        lambda *_args, **_kwargs: {
            "training_admitted": True,
            "candidate_passed_twice": True,
            "candidate_patch_sha256": "a" * 64,
            "rejection_reasons": [],
        },
    )

    def reject_preflight(_row, _backend, _stream, evidence):
        assert tp._sha256_path(
            tmp_path / f"{row['instance_id']}.stream.jsonl"
        ) == evidence["stream_sha256"]
        assert tp._sha256_path(
            tmp_path / f"{row['instance_id']}.patch"
        ) == evidence["patch_sha256"]
        raise TracePreflightError(
            "raw_patch_mismatch",
            "raw mutation subsequence differs",
        )

    monkeypatch.setattr(
        tp,
        "preflight_trainable_trace",
        reject_preflight,
    )
    row = {
        "instance_id": "fixture__repo.pr_1",
        "image_name": "fixture/image:latest",
        "problem_statement": "Fix x.",
        "patch": patch_text,
        "FAIL_TO_PASS": ["tests/test_x.py::test_fix"],
        "PASS_TO_PASS": ["tests/test_x.py::test_old"],
    }

    result = tp.collect_one_openrouter(
        row,
        tmp_path,
        max_turns=1,
        model="fixture/model",
    )

    assert result["executed"] is True
    assert result["f2p_pass"] is True
    assert result["training_admitted"] is False
    assert result["resolved"] is False
    assert result["rejection_reasons"] == [
        "trace_preflight:raw_patch_mismatch"
    ]
    assert (tmp_path / f"{row['instance_id']}.stream.jsonl").is_file()
    assert (tmp_path / f"{row['instance_id']}.patch").is_file()


def _collector_preflight_bug_record(task, stream_bytes, candidate_patch):
    contract, controls = _strict_control_set(task)
    assessment = assess_controls(
        controls,
        protected_patch_paths=[],
    )
    evidence = {
        "admission_schema_version": 2,
        "task_contract": contract,
        "task_contract_sha256": contract["contract_sha256"],
        "image_id": "sha256:fixture",
        "candidate_patch_sha256": tp.sha256_bytes(
            candidate_patch.encode()
        ),
        "candidate_patch_paths": list(
            tp.parse_patch_paths(candidate_patch)
        ),
        "protected_patch_paths": [],
        "controls": controls,
        **assessment,
    }
    evidence["admission_evidence_sha256"] = tp.sha256_bytes(
        tp.canonical_json_bytes(evidence)
    )
    record = {
        "instance_id": task["instance_id"],
        "backend": "claude",
        "model": "claude-fable-5",
        "stream_sha256": tp.sha256_bytes(stream_bytes),
        "patch_sha256": tp.sha256_bytes(candidate_patch.encode()),
        "executed": True,
        "f2p_pass": True,
        "p2p_pass": True,
        "reference_controls_passed": True,
        "mutation_f2p_reproduced": True,
        "mutation_p2p_passed": True,
        **evidence,
    }
    assert tp.admission_is_exact(record)
    record.update(
        training_admitted=False,
        resolved=False,
        rejection_reasons=["trace_preflight:normalization_error"],
        trace_preflight_error=(
            "normalization_error: exact strict admission evidence is required"
        ),
    )
    return record


def test_cmd_repreflight_recovers_only_bound_collector_bug_artifacts(
    tmp_path,
    monkeypatch,
):
    task = {
        "instance_id": "fixture__repo.pr_1",
        "image_name": "fixture/image:latest",
        "problem_statement": "Fix x.",
        "patch": (
            "diff --git a/src/x.py b/src/x.py\n"
            "--- a/src/x.py\n"
            "+++ b/src/x.py\n"
            "@@ -1 +1 @@\n"
            "-bad\n"
            "+good\n"
        ),
        "FAIL_TO_PASS": ["tests/test_x.py::test_fix"],
        "PASS_TO_PASS": ["tests/test_x.py::test_old"],
    }
    stream_bytes = b'{"type":"result"}\n'
    candidate_patch = task["patch"]
    source = tmp_path / "source"
    source.mkdir()
    stream = source / f"{task['instance_id']}.stream.jsonl"
    patch = source / f"{task['instance_id']}.patch"
    stream.write_bytes(stream_bytes)
    patch.write_text(candidate_patch)
    record = _collector_preflight_bug_record(
        task,
        stream_bytes,
        candidate_patch,
    )
    ledger = source / "results.jsonl"
    ledger.write_text(json.dumps(record) + "\n")
    original_ledger = ledger.read_bytes()
    tasks = tmp_path / "tasks.jsonl"
    tasks.write_text(json.dumps(task) + "\n")
    out = tmp_path / "repreflighted"

    def successful_preflight(
        actual_task,
        backend,
        actual_stream,
        evidence,
    ):
        assert actual_task == task
        assert backend == "claude"
        assert actual_stream == stream
        assert tp.admission_is_exact(evidence)
        return SimpleNamespace(
            source_sha256="d" * 64,
            retained_steps=3,
        )

    monkeypatch.setattr(
        tp,
        "preflight_trainable_trace",
        successful_preflight,
    )

    assert tp.cmd_repreflight(SimpleNamespace(
        input=str(ledger),
        tasks=[str(tasks)],
        out_dir=str(out),
    )) == 0

    recovered = json.loads((out / "resolved.jsonl").read_text())
    manifest = json.loads((out / "manifest.json").read_text())
    assert recovered["training_admitted"] is True
    assert recovered["resolved"] is True
    assert recovered["rejection_reasons"] == []
    assert recovered["trace_preflight_source_sha256"] == "d" * 64
    assert recovered["trace_preflight_retained_steps"] == 3
    assert tp.admission_is_exact(recovered)
    assert manifest["artifact_type"] == "trace_preflight_recovery"
    assert manifest["selected"] == 1
    assert manifest["training_admitted"] == 1
    assert manifest["rejected"] == 0
    assert manifest["source_snapshot_sha256"] == tp._sha256_path(
        out / "source_results.jsonl"
    )
    assert (out / stream.name).read_bytes() == stream_bytes
    assert (out / patch.name).read_text() == candidate_patch
    assert ledger.read_bytes() == original_ledger


def _run_translated(command, tmp_path):
    lines = command.splitlines()
    for index, line in enumerate(lines):
        if line.startswith("path = Path(") and line.endswith(")"):
            source_path = json.loads(line[len("path = Path("):-1])
            local_path = tmp_path / Path(source_path).relative_to("/testbed")
            lines[index] = f"path = Path({json.dumps(str(local_path))})"
            local_command = "\n".join(lines)
            break
    else:
        # View translations contain only shell path tokens, never embedded payloads.
        local_command = command.replace("/testbed", str(tmp_path))
    return subprocess.run(["bash", "-c", local_command], capture_output=True, text=True)


@pytest.fixture
def run_ingest_cli(monkeypatch, tmp_path):
    """Run the real ingest CLI parser against an in-memory HF dataset."""
    import datasets

    def run(rows, output_name="ingested"):
        saved_rows = []

        class FakeSavedDataset:
            def save_to_disk(self, path):
                Path(path).mkdir()

        class FakeDataset:
            @staticmethod
            def from_list(output_rows):
                saved_rows.extend(output_rows)
                return FakeSavedDataset()

        def fake_load_dataset(dataset, config, streaming):
            assert (dataset, config, streaming) == ("fixture/open-swe", "fixture", True)
            return {"train": rows}

        monkeypatch.setattr(datasets, "Dataset", FakeDataset)
        monkeypatch.setattr(datasets, "load_dataset", fake_load_dataset)
        out = tmp_path / output_name
        monkeypatch.setattr(sys, "argv", [
            "teacher_platform.py",
            "ingest",
            "--out", str(out),
            "--dataset", "fixture/open-swe",
            "--configs", "fixture",
            "--max-steps", "0",
        ])
        return tp.main(), out, saved_rows

    return run


def test_dud_excluded_real_kept():
    assert tp._is_real_attempt({"n_assistant_events": 40, "patch_len": 500})
    assert tp._is_real_attempt({"n_assistant_events": 0, "patch_len": 900})
    assert not tp._is_real_attempt({"n_assistant_events": 1, "patch_len": 0})  # quota wall
    assert not tp._is_real_attempt({"n_assistant_events": 2, "patch_len": 0})
    assert not tp._is_real_attempt({
        "n_assistant_events": 67,
        "patch_len": 0,
        "error": tp.ttd.CREDIT_EXHAUSTED_ERROR,
    })


def test_hard_requires_exact_clean_base_sample_count(tmp_path):
    from types import SimpleNamespace

    labels = tmp_path / "labels.jsonl"
    pool = tmp_path / "pool.jsonl"
    output = tmp_path / "hard.jsonl"
    rows = []
    for instance_id, count in (("complete-fail", 4), ("short", 3)):
        rows.extend(
            {"instance_id": instance_id, "resolved": False}
            for _ in range(count)
        )
    rows.extend(
        {"instance_id": "solved", "resolved": index == 2}
        for index in range(4)
    )
    rows.extend([
        {"instance_id": "infra", "resolved": False},
        {"instance_id": "infra", "resolved": False},
        {"instance_id": "infra", "resolved": False, "pull_failed": True},
        {"instance_id": "infra", "resolved": False},
    ])
    labels.write_text("".join(json.dumps(row) + "\n" for row in rows))
    pool.write_text("".join(
        json.dumps({"instance_id": instance_id, "repo": "fixture/repo"}) + "\n"
        for instance_id in ("complete-fail", "short", "solved", "infra")
    ))

    assert tp.cmd_hard(SimpleNamespace(
        labels=str(labels),
        pool=str(pool),
        out=str(output),
        exclude=[],
        samples_per_instance=4,
    )) == 0

    assert tp._read_jsonl(output) == [{
        "instance_id": "complete-fail",
        "repo": "fixture/repo",
    }]
    manifest = json.loads(Path(str(output) + ".manifest.json").read_text())
    assert manifest["instances_with_complete_clean_samples"] == 2
    assert manifest["invalid_reasons"] == {
        "wrong_sample_count": 1,
        "invalid_or_infrastructure_result": 1,
    }


def test_strip_docker_exec_wrapper():
    raw = 'docker exec abc123 bash -c "cd /testbed && sed -i s/a/b/ x.py"'
    assert tp._strip_docker_exec(raw) == "sed -i s/a/b/ x.py"
    # already-bare command is unchanged
    assert tp._strip_docker_exec("grep -n foo x.py") == "grep -n foo x.py"
    # cd /testbed prefix without wrapper is stripped
    assert tp._strip_docker_exec("cd /testbed && pytest -x") == "pytest -x"


def test_render_sft_schema_uniform_and_toolcalls():
    row = {"instance_id": "r__x.pr_1", "problem_statement": "bug", "repo": "swesmith/r__x", "backend": "claude", "model": "fable"}
    steps = [{"thought": "look", "command": "grep -n foo x.py", "observation": "1: foo"},
             {"thought": "fix", "command": "sed -i s/foo/bar/ x.py", "observation": ""}]
    out = tp.render_sft(row, steps)
    msgs = out["messages"]
    # every message carries a tool_calls key (uniform arrow schema)
    assert all("tool_calls" in m and "role" in m and "content" in m for m in msgs)
    assert msgs[0]["role"] == "system" and msgs[1]["role"] == "user"
    # assistant turns carry exactly one bash tool_call with valid JSON args
    import json
    asg = [m for m in msgs if m["role"] == "assistant"]
    assert len(asg) == 2
    for m in asg:
        assert len(m["tool_calls"]) == 1
        tc = m["tool_calls"][0]
        assert tc["function"]["name"] == "bash"
        assert "command" in json.loads(tc["function"]["arguments"])
    # thought is preserved in assistant content
    assert asg[0]["content"] == "look"
    # observation rendered as a user turn
    assert any(m["role"] == "user" and m["content"].startswith("OBSERVATION") for m in msgs)


def test_render_sft_preserves_terminal_assistant_and_explicit_loss():
    row = {
        "instance_id": "r__x.pr_2",
        "problem_statement": "bug",
        "repo": "swesmith/r__x",
        "backend": "claude",
        "model": "fable",
    }
    trace = tp.NormalizedTrace(
        [{"thought": "fix", "command": "sed -i s/a/b/ x.py", "observation": "ok"}],
        terminal_assistant="Implemented and verified.\nDONE",
    )

    out = tp.render_sft(row, trace, require_terminal=True)
    messages = out["messages"]

    assert messages[-1] == {
        "role": "assistant",
        "content": "Implemented and verified.\nDONE",
        "tool_calls": [],
        "loss": True,
    }
    assert all(
        message["loss"] is (message["role"] == "assistant")
        for message in messages
    )


def test_render_sft_empty_steps_returns_none():
    assert tp.render_sft({"instance_id": "x", "problem_statement": "p"}, []) is None


def test_normalize_steps_openrouter(tmp_path):
    import json
    p = tmp_path / "s.stream.jsonl"
    p.write_text("\n".join(json.dumps(x) for x in [
        {"role": "assistant", "content": "read it", "tool_calls": ['{"command": "cat a.py"}']},
        {"role": "tool", "command": "cat a.py", "observation": "def a(): ..."},
        {"role": "assistant", "content": "patch it", "tool_calls": ['{"command": "sed -i s/a/b/ a.py"}']},
        {"role": "tool", "command": "sed -i s/a/b/ a.py", "observation": ""},
    ]))
    steps = tp.normalize_steps("openrouter", p)
    assert [s["command"] for s in steps] == ["cat a.py", "sed -i s/a/b/ a.py"]
    assert steps[0]["thought"] == "read it" and steps[0]["observation"].startswith("def a")


def test_steps_from_trajectory_openai_style():
    msgs = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "PR: fix it"},
        {"role": "assistant", "content": "explore",
         "tool_calls": [{"function": {"name": "bash", "arguments": '{"command": "ls /testbed"}'}}]},
        {"role": "tool", "content": "OBSERVATION:\na.py b.py"},
        {"role": "assistant", "content": "patch",
         "tool_calls": [{"function": {"name": "bash", "arguments": '{"command": "sed -i s/a/b/ a.py"}'}}]},
        {"role": "tool", "content": "done"},
    ]
    steps = tp.steps_from_trajectory(msgs)
    assert [s["command"] for s in steps] == ["ls /testbed", "sed -i s/a/b/ a.py"]
    assert steps[0]["thought"] == "explore" and steps[0]["observation"] == "a.py b.py"  # OBSERVATION: stripped


def test_steps_from_trajectory_preserves_reasoning_fallbacks():
    msgs = [
        {"role": "assistant", "content": "", "reasoning_content": "inspect carefully", "think": "ignored",
         "tool_calls": [_trajectory_tool_call("bash", {"command": "cat /testbed/a.py"})]},
        {"role": "tool", "content": "a"},
        {"role": "assistant", "content": "", "reasoning_content": "", "think": "verify now",
         "tool_calls": [_trajectory_tool_call("bash", {"command": "cd /testbed && pytest -q"})]},
        {"role": "tool", "content": "ok"},
    ]

    steps = tp.steps_from_trajectory(msgs)

    assert [step["thought"] for step in steps] == ["inspect carefully", "verify now"]


def test_translate_trajectory_bash_passthrough():
    call = _trajectory_tool_call("bash", {"command": "cd /testbed && pytest -q"})

    assert tp.translate_trajectory_tool_call(call) == "pytest -q"


def test_steps_from_trajectory_normalizes_declared_root_editor_paths():
    declared_root = "/workspace/demo-repo"
    msgs = [
        {
            "role": "user",
            "content": (
                "<uploaded_files>\n"
                f"{declared_root}\n"
                "</uploaded_files>\n"
                "Fix the failing behavior."
            ),
        },
        {
            "role": "assistant",
            "tool_calls": [_trajectory_tool_call(
                "str_replace_editor",
                {
                    "command": "view",
                    "path": f"{declared_root}/src/module.py",
                },
            )],
        },
        {"role": "tool", "content": "source"},
    ]

    command = tp.steps_from_trajectory(msgs)[0]["command"]

    assert "/testbed/src/module.py" in command
    assert declared_root not in command

    msgs[1]["tool_calls"] = [_trajectory_tool_call(
        "str_replace_editor",
        {
            "command": "view",
            "path": "/workspace/demo-repo-sibling/secret.py",
        },
    )]
    with pytest.raises(tp.UnsupportedTrajectoryTool, match="editor path"):
        tp.steps_from_trajectory(msgs)


@pytest.mark.parametrize("argument_name", ["command", "input"])
def test_steps_from_trajectory_normalizes_openhands_execute_bash(argument_name):
    declared_root = "/workspace/demo-repo"
    command = f"cd {declared_root} && pytest -q {declared_root}/tests"
    msgs = [
        {
            "role": "user",
            "content": f"<uploaded_files>\n{declared_root}\n</uploaded_files>\nFix it.",
        },
        {
            "role": "assistant",
            "tool_calls": [_trajectory_tool_call(
                "execute_bash", {argument_name: command}
            )],
        },
        {"role": "tool", "content": "passed"},
    ]

    assert tp.steps_from_trajectory(msgs)[0]["command"] == "pytest -q /testbed/tests"


@pytest.mark.parametrize(
    "command",
    [
        "cat /workspace/another-repo/secret.py",
        "cat /workspace/demo-repo/../secret.py",
        'cat "/workspace/demo-repo"/../secret.py',
        "cat /work'space'/another-repo/secret.py",
        (
            'docker exec container bash -c "cd /workspace/demo-repo && '
            'cat /workspace/another-repo/secret.py"'
        ),
    ],
)
def test_steps_from_trajectory_rejects_unsafe_execute_bash_workspace_paths(command):
    msgs = [
        {
            "role": "user",
            "content": "<uploaded_files>\n/workspace/demo-repo\n</uploaded_files>\nFix it.",
        },
        {
            "role": "assistant",
            "tool_calls": [_trajectory_tool_call("execute_bash", {"input": command})],
        },
        {"role": "tool", "content": "unsafe"},
    ]

    with pytest.raises(tp.UnsupportedTrajectoryTool, match="workspace path"):
        tp.steps_from_trajectory(msgs)


@pytest.mark.parametrize(
    "declared_root",
    [
        "/workspace",
        "/workspace/demo-repo/..",
    ],
)
def test_steps_from_trajectory_rejects_too_broad_declared_root(declared_root):
    msgs = [
        {
            "role": "user",
            "content": f"<uploaded_files>\n{declared_root}\n</uploaded_files>\nFix it.",
        },
        {
            "role": "assistant",
            "tool_calls": [_trajectory_tool_call(
                "execute_bash", {"input": "cat /workspace/another-repo/secret.py"}
            )],
        },
        {"role": "tool", "content": "unsafe"},
    ]

    with pytest.raises(tp.UnsupportedTrajectoryTool, match="too broad"):
        tp.steps_from_trajectory(msgs)


def test_steps_from_trajectory_rejects_double_leading_slash_declared_root():
    msgs = [
        {
            "role": "user",
            "content": "<uploaded_files>\n//workspace\n</uploaded_files>\nFix it.",
        },
        {
            "role": "assistant",
            "tool_calls": [_trajectory_tool_call("bash", {"command": "pwd"})],
        },
        {"role": "tool", "content": "/testbed"},
    ]

    with pytest.raises(tp.UnsupportedTrajectoryTool, match="double-leading"):
        tp.steps_from_trajectory(msgs)


@pytest.mark.parametrize(
    "command",
    [
        "cd /workspace/demo-repo && cat ../another-repo/secret.py",
        (
            'docker exec container bash -c "cd /workspace/demo-repo && '
            'cat ../another-repo/secret.py"'
        ),
    ],
)
def test_execute_bash_rejects_relative_escape_after_wrapper_cleanup(command):
    msgs = [
        {
            "role": "user",
            "content": (
                "<uploaded_files>\n/workspace/demo-repo\n</uploaded_files>\nFix it."
            ),
        },
        {
            "role": "assistant",
            "tool_calls": [_trajectory_tool_call("execute_bash", {"input": command})],
        },
        {"role": "tool", "content": "unsafe"},
    ]

    with pytest.raises(tp.UnsupportedTrajectoryTool, match="workspace path"):
        tp.steps_from_trajectory(msgs)


@pytest.mark.parametrize(
    "command",
    [
        "cat tests/workspace/fixture.py",
        "cat /tmp/workspace/fixture.py",
    ],
)
def test_translate_trajectory_preserves_nonroot_workspace_path_segments(command):
    call = _trajectory_tool_call("bash", {"command": command})

    assert tp.translate_trajectory_tool_call(call) == command


def test_translate_trajectory_preserves_valid_heredoc_body():
    command = (
        "cat <<'EOF' > /testbed/note.txt\n"
        "it's valid shell heredoc content\n"
        "EOF"
    )
    call = _trajectory_tool_call("bash", {"command": command})

    assert tp.translate_trajectory_tool_call(call) == command


def test_translate_trajectory_rejects_unquoted_heredoc_expansion_escape():
    call = _trajectory_tool_call(
        "bash",
        {
            "command": (
                "cat <<EOF\n"
                "$(cat /workspace/another-repo/secret.py)\n"
                "EOF"
            ),
        },
    )

    with pytest.raises(tp.UnsupportedTrajectoryTool, match="unquoted heredoc"):
        tp.translate_trajectory_tool_call(call)


def test_execute_bash_preserves_multiline_docker_wrapped_quoted_heredoc():
    command = (
        'docker exec container bash -c "cd /workspace/demo-repo && '
        "cat <<'EOF' > note.txt\n"
        "it's valid shell heredoc content\n"
        'EOF"'
    )
    msgs = [
        {
            "role": "user",
            "content": (
                "<uploaded_files>\n/workspace/demo-repo\n</uploaded_files>\nFix it."
            ),
        },
        {
            "role": "assistant",
            "tool_calls": [_trajectory_tool_call("execute_bash", {"input": command})],
        },
        {"role": "tool", "content": "created"},
    ]

    assert tp.steps_from_trajectory(msgs)[0]["command"] == (
        "cat <<'EOF' > note.txt\n"
        "it's valid shell heredoc content\n"
        "EOF"
    )


@pytest.mark.parametrize(
    "consumer",
    [
        "bash",
        "sh",
        "/bin/bash",
        "/bin/sh",
        "env bash",
        "/usr/bin/env bash",
    ],
)
def test_quoted_heredoc_shell_consumer_scans_executable_body(consumer):
    call = _trajectory_tool_call(
        "bash",
        {
            "command": (
                f"{consumer} <<'EOF'\n"
                "cat /workspace/another-repo/secret.py\n"
                "EOF"
            ),
        },
    )

    with pytest.raises(tp.UnsupportedTrajectoryTool, match="workspace path"):
        tp.translate_trajectory_tool_call(call)


def test_docker_wrapped_quoted_heredoc_shell_consumer_scans_executable_body():
    command = (
        'docker exec container bash -c "cd /workspace/demo-repo && '
        "bash <<'EOF'\n"
        "cat /workspace/another-repo/secret.py\n"
        'EOF"'
    )
    msgs = [
        {
            "role": "user",
            "content": (
                "<uploaded_files>\n/workspace/demo-repo\n</uploaded_files>\nFix it."
            ),
        },
        {
            "role": "assistant",
            "tool_calls": [_trajectory_tool_call("execute_bash", {"input": command})],
        },
        {"role": "tool", "content": "unsafe"},
    ]

    with pytest.raises(tp.UnsupportedTrajectoryTool, match="workspace path"):
        tp.steps_from_trajectory(msgs)


@pytest.mark.parametrize(
    "consumer",
    [
        "command bash",
        "exec bash",
        "source /dev/stdin",
        "2>/dev/null bash",
    ],
)
def test_quoted_heredoc_rejects_ambiguous_shell_consumer_wrappers(consumer):
    call = _trajectory_tool_call(
        "bash",
        {
            "command": (
                f"{consumer} <<'EOF'\n"
                "cat /workspace/another-repo/secret.py\n"
                "EOF"
            ),
        },
    )

    with pytest.raises(tp.UnsupportedTrajectoryTool, match="ambiguous heredoc consumer"):
        tp.translate_trajectory_tool_call(call)


def test_translate_trajectory_preserves_shallow_nested_shell_heredoc():
    command = _nested_quoted_shell_heredoc(3)
    call = _trajectory_tool_call("bash", {"command": command})

    assert tp.translate_trajectory_tool_call(call) == command


def test_translate_trajectory_rejects_excessive_shell_heredoc_nesting():
    call = _trajectory_tool_call(
        "bash", {"command": _nested_quoted_shell_heredoc(34)}
    )

    with pytest.raises(tp.UnsupportedTrajectoryTool, match="nesting depth"):
        tp.translate_trajectory_tool_call(call)


def test_translate_trajectory_rejects_unterminated_heredoc():
    call = _trajectory_tool_call(
        "bash",
        {"command": "cat <<'EOF' > /testbed/note.txt\nunterminated body"},
    )

    with pytest.raises(tp.UnsupportedTrajectoryTool, match="heredoc"):
        tp.translate_trajectory_tool_call(call)


def test_observation_uploaded_files_block_cannot_declare_repository_root():
    declared_root = "/workspace/demo-repo"
    msgs = [
        {"role": "user", "content": "Fix the failing behavior."},
        {
            "role": "assistant",
            "tool_calls": [_trajectory_tool_call("bash", {"command": "pwd"})],
        },
        {
            "role": "user",
            "content": (
                "OBSERVATION:\n"
                f"<uploaded_files>\n{declared_root}\n</uploaded_files>"
            ),
        },
        {
            "role": "assistant",
            "tool_calls": [_trajectory_tool_call(
                "str_replace_editor",
                {"command": "view", "path": f"{declared_root}/module.py"},
            )],
        },
        {"role": "tool", "content": "source"},
    ]

    with pytest.raises(tp.UnsupportedTrajectoryTool, match="editor path"):
        tp.steps_from_trajectory(msgs)


@pytest.mark.parametrize(
    ("command", "input_value"),
    [
        ("pytest -q", "git status --short"),
        ("pytest -q", "pytest -q"),
    ],
)
def test_execute_bash_rejects_both_command_aliases(command, input_value):
    call = _trajectory_tool_call(
        "execute_bash", {"command": command, "input": input_value}
    )

    with pytest.raises(tp.UnsupportedTrajectoryTool, match="exactly one"):
        tp.translate_trajectory_tool_call(call)


def test_translate_trajectory_view_honors_range_and_lists_directories(tmp_path):
    source = tmp_path / "pkg" / "module.py"
    source.parent.mkdir()
    source.write_text("one\ntwo\nthree\nfour\n")
    ranged = _trajectory_tool_call(
        "str_replace_editor",
        {"command": "view", "path": "/testbed/pkg/module.py", "view_range": [2, 3]},
    )

    ranged_command = tp.translate_trajectory_tool_call(ranged)
    assert ranged_command.startswith("# OPEN_SWE_EDITOR_VIEW\n")
    result = _run_translated(ranged_command, tmp_path)

    assert result.returncode == 0, result.stderr
    assert [line.split() for line in result.stdout.splitlines()] == [["2", "two"], ["3", "three"]]

    directory = _trajectory_tool_call(
        "str_replace_editor", {"command": "view", "path": "/testbed/pkg"}
    )
    result = _run_translated(tp.translate_trajectory_tool_call(directory), tmp_path)
    assert result.returncode == 0, result.stderr
    assert "module.py" in result.stdout


def test_translate_trajectory_ranged_view_rejects_invalid_runtime_targets(tmp_path):
    missing = _trajectory_tool_call(
        "str_replace_editor",
        {"command": "view", "path": "/testbed/missing.py", "view_range": [1, 2]},
    )
    result = _run_translated(tp.translate_trajectory_tool_call(missing), tmp_path)
    assert result.returncode != 0

    source = tmp_path / "short.py"
    source.write_text("one\ntwo\n")
    past_eof = _trajectory_tool_call(
        "str_replace_editor",
        {"command": "view", "path": "/testbed/short.py", "view_range": [3, 4]},
    )
    result = _run_translated(tp.translate_trajectory_tool_call(past_eof), tmp_path)
    assert result.returncode != 0

    directory = tmp_path / "pkg"
    directory.mkdir()
    ranged_directory = _trajectory_tool_call(
        "str_replace_editor",
        {"command": "view", "path": "/testbed/pkg", "view_range": [1, 2]},
    )
    result = _run_translated(tp.translate_trajectory_tool_call(ranged_directory), tmp_path)
    assert result.returncode != 0


def test_translate_trajectory_create_writes_exact_file_text(tmp_path):
    file_text = "alpha\n/testbed/source.py\n'''quoted'''\n$HOME\nOPEN_SWE_PY\n"
    call = _trajectory_tool_call(
        "str_replace_editor",
        {"command": "create", "path": "/testbed/new file.py", "file_text": file_text},
    )

    result = _run_translated(tp.translate_trajectory_tool_call(call), tmp_path)

    assert result.returncode == 0, result.stderr
    assert (tmp_path / "new file.py").read_text() == file_text


def test_translate_trajectory_create_refuses_existing_path_without_mutation(tmp_path):
    existing = tmp_path / "existing.py"
    existing.write_bytes(b"keep these bytes\n")
    call = _trajectory_tool_call(
        "str_replace_editor",
        {
            "command": "create",
            "path": "/testbed/existing.py",
            "file_text": "overwrite",
        },
    )

    result = _run_translated(tp.translate_trajectory_tool_call(call), tmp_path)

    assert result.returncode != 0
    assert existing.read_bytes() == b"keep these bytes\n"


def test_translate_trajectory_str_replace_requires_exactly_one_match(tmp_path):
    source = tmp_path / "module.py"
    source.write_text("old\nold\n")
    call = _trajectory_tool_call(
        "str_replace_editor",
        {
            "command": "str_replace",
            "path": "/testbed/module.py",
            "old_str": "old",
            "new_str": "new",
        },
    )
    command = tp.translate_trajectory_tool_call(call)

    result = _run_translated(command, tmp_path)

    assert result.returncode != 0
    assert source.read_text() == "old\nold\n"

    source.write_text("before old after\n")
    result = _run_translated(command, tmp_path)
    assert result.returncode == 0, result.stderr
    assert source.read_text() == "before new after\n"


def test_translate_trajectory_insert_adds_text_after_valid_line(tmp_path):
    source = tmp_path / "module.py"
    source.write_text("one\nthree\n")
    call = _trajectory_tool_call(
        "str_replace_editor",
        {
            "command": "insert",
            "path": "/testbed/module.py",
            "insert_line": 1,
            "new_str": "two",
        },
    )

    command = tp.translate_trajectory_tool_call(call)
    result = _run_translated(command, tmp_path)

    assert result.returncode == 0, result.stderr
    assert source.read_text() == "one\ntwo\nthree\n"

    source.write_text("one\n")
    invalid = _trajectory_tool_call(
        "str_replace_editor",
        {
            "command": "insert",
            "path": "/testbed/module.py",
            "insert_line": 3,
            "new_str": "never\n",
        },
    )
    result = _run_translated(tp.translate_trajectory_tool_call(invalid), tmp_path)
    assert result.returncode != 0
    assert source.read_text() == "one\n"


@pytest.mark.parametrize(
    ("new_str", "expected"),
    [
        ("two", "one\ntwo\nthree\n"),
        ("two\n", "one\ntwo\n\nthree\n"),
        ("two\nmiddle", "one\ntwo\nmiddle\nthree\n"),
        ("two\nmiddle\n", "one\ntwo\nmiddle\n\nthree\n"),
    ],
)
def test_translate_trajectory_insert_preserves_payload_line_semantics(tmp_path, new_str, expected):
    source = tmp_path / "module.py"
    source.write_text("one\nthree\n")
    call = _trajectory_tool_call(
        "str_replace_editor",
        {
            "command": "insert",
            "path": "/testbed/module.py",
            "insert_line": 1,
            "new_str": new_str,
        },
    )

    result = _run_translated(tp.translate_trajectory_tool_call(call), tmp_path)

    assert result.returncode == 0, result.stderr
    assert source.read_text() == expected


@pytest.mark.parametrize(
    ("initial", "insert_line"),
    [
        (b"", 1),
        (b"one\n", 2),
    ],
)
def test_translate_trajectory_insert_rejects_one_past_physical_eof(
    tmp_path, initial, insert_line
):
    source = tmp_path / "module.py"
    source.write_bytes(initial)
    call = _trajectory_tool_call(
        "str_replace_editor",
        {
            "command": "insert",
            "path": "/testbed/module.py",
            "insert_line": insert_line,
            "new_str": "never",
        },
    )

    result = _run_translated(tp.translate_trajectory_tool_call(call), tmp_path)

    assert result.returncode != 0
    assert source.read_bytes() == initial


def test_translate_trajectory_submit_is_omitted():
    assert tp.translate_trajectory_tool_call(_trajectory_tool_call("submit", {})) is None


@pytest.mark.parametrize(
    "intervening",
    [
        {"role": "user", "content": "new request"},
        {"role": "system", "content": "new system context"},
        {"role": "assistant", "content": "text-only answer", "tool_calls": []},
    ],
)
@pytest.mark.parametrize("observation_role", ["tool", "user"])
@pytest.mark.parametrize("with_ids", [False, True])
def test_steps_from_trajectory_rejects_delayed_submit_observation(
    intervening, observation_role, with_ids
):
    submit = _trajectory_tool_call("submit", {})
    observation = {
        "role": observation_role,
        "content": "late result" if observation_role == "tool" else "OBSERVATION:\nlate result",
    }
    if with_ids:
        submit["id"] = "call-submit"
        observation["tool_call_id"] = "call-submit"
    msgs = [
        {"role": "assistant", "tool_calls": [submit]},
        intervening,
        observation,
    ]

    with pytest.raises(tp.UnsupportedTrajectoryTool):
        tp.steps_from_trajectory(msgs)


@pytest.mark.parametrize("observation_role", ["tool", "user"])
@pytest.mark.parametrize("with_ids", [False, True])
def test_steps_from_trajectory_accepts_immediate_submit_observation(observation_role, with_ids):
    submit = _trajectory_tool_call("submit", {})
    observation = {
        "role": observation_role,
        "content": "result" if observation_role == "tool" else "OBSERVATION:\nresult",
    }
    if with_ids:
        submit["id"] = "call-submit"
        observation["tool_call_id"] = "call-submit"

    assert tp.steps_from_trajectory([
        {"role": "assistant", "tool_calls": [submit]},
        observation,
    ]) == []


@pytest.mark.parametrize(
    "call",
    [
        _trajectory_tool_call("str_replace_editor", {"command": "undo_edit", "path": "/testbed/a.py"}),
        _trajectory_tool_call("unknown", {}),
        {"function": {"name": "bash", "arguments": "{"}},
        {},
        _trajectory_tool_call("bash", {"command": 123}),
        _trajectory_tool_call("str_replace_editor", {"command": "view", "path": "/tmp/a.py"}),
        _trajectory_tool_call(
            "str_replace_editor", {"command": "view", "path": "/testbed/a.py", "view_range": [3, 2]}
        ),
        _trajectory_tool_call(
            "str_replace_editor", {"command": "create", "path": "/testbed/a.py", "file_text": 1}
        ),
        _trajectory_tool_call(
            "str_replace_editor",
            {"command": "str_replace", "path": "/testbed/a.py", "old_str": "x"},
        ),
        _trajectory_tool_call(
            "str_replace_editor",
            {"command": "insert", "path": "/testbed/a.py", "insert_line": True, "new_str": "x"},
        ),
    ],
)
def test_translate_trajectory_rejects_unsupported_or_malformed_calls(call):
    with pytest.raises(tp.UnsupportedTrajectoryTool):
        tp.translate_trajectory_tool_call(call)


def test_steps_from_trajectory_rejects_multiple_tool_calls():
    msgs = [
        {
            "role": "assistant",
            "content": "two at once",
            "tool_calls": [
                _trajectory_tool_call("bash", {"command": "pwd"}),
                _trajectory_tool_call("bash", {"command": "ls"}),
            ],
        },
        {"role": "tool", "content": "output"},
    ]

    with pytest.raises(tp.UnsupportedTrajectoryTool):
        tp.steps_from_trajectory(msgs)


@pytest.mark.parametrize(
    "msgs",
    [
        [
            {"role": "assistant", "tool_calls": [_trajectory_tool_call("bash", {"command": "pwd"})]},
        ],
        [{"role": "tool", "content": "orphaned"}],
        [
            {"role": "assistant", "tool_calls": [_trajectory_tool_call("bash", {"command": "pwd"})]},
            {"role": "assistant", "tool_calls": [_trajectory_tool_call("bash", {"command": "ls"})]},
            {"role": "tool", "content": "output"},
        ],
    ],
)
def test_steps_from_trajectory_rejects_unpaired_events(msgs):
    with pytest.raises(tp.UnsupportedTrajectoryTool):
        tp.steps_from_trajectory(msgs)


@pytest.mark.parametrize(
    "observation",
    [
        {"role": "user", "content": "This is a new request, not an observation."},
        {"role": "tool", "content": {"malformed": True}},
        {"role": "user", "content": 123},
    ],
)
def test_steps_from_trajectory_rejects_ambiguous_observations(observation):
    msgs = [
        {
            "role": "assistant",
            "tool_calls": [_trajectory_tool_call("bash", {"command": "pwd"})],
        },
        observation,
    ]

    with pytest.raises(tp.UnsupportedTrajectoryTool):
        tp.steps_from_trajectory(msgs)


def test_steps_from_trajectory_rejects_mismatched_tool_call_id():
    call = _trajectory_tool_call("bash", {"command": "pwd"})
    call["id"] = "call-a"
    msgs = [
        {"role": "assistant", "tool_calls": [call]},
        {"role": "tool", "tool_call_id": "call-b", "content": "wrong call"},
    ]

    with pytest.raises(tp.UnsupportedTrajectoryTool):
        tp.steps_from_trajectory(msgs)


def test_steps_from_trajectory_accepts_matching_tool_call_id():
    call = _trajectory_tool_call("bash", {"command": "pwd"})
    call["id"] = "call-a"
    msgs = [
        {"role": "assistant", "tool_calls": [call]},
        {"role": "tool", "tool_call_id": "call-a", "content": "/testbed"},
    ]

    assert tp.steps_from_trajectory(msgs)[0]["observation"] == "/testbed"


def test_steps_from_trajectory_accepts_immediate_result_without_tool_call_id():
    call = _trajectory_tool_call("bash", {"command": "pwd"})
    call["id"] = "call-a"
    msgs = [
        {"role": "assistant", "tool_calls": [call]},
        {"role": "tool", "content": "/testbed"},
    ]

    assert tp.steps_from_trajectory(msgs)[0]["observation"] == "/testbed"


def test_cmd_merge_writes_resolved_and_rejected_banks(tmp_path, monkeypatch):
    import hashlib
    from types import SimpleNamespace

    monkeypatch.setattr(
        tp,
        "admission_is_exact",
        lambda row: row.get("training_admitted") is True,
    )
    first = tmp_path / "batch_a"
    second = tmp_path / "batch_b"
    first.mkdir()
    second.mkdir()
    patch_text = "diff --git a/src/x.py b/src/x.py\n"
    stream_sha = hashlib.sha256(b"stream\n").hexdigest()
    patch_sha = hashlib.sha256(patch_text.encode()).hexdigest()
    (first / "results.jsonl").write_text("\n".join([
        json.dumps({"instance_id": "resolved", "n_assistant_events": 4,
                    "patch_len": 10, "resolved": True,
                    "training_admitted": True,
                    "stream_sha256": stream_sha,
                    "patch_sha256": patch_sha,
                    "candidate_patch_sha256": patch_sha,
                    "candidate_patch_paths": ["src/x.py"],
                    "admission_evidence_sha256": "a" * 64,
                    "task_contract_sha256": "b" * 64}),
        json.dumps({"instance_id": "rejected", "n_assistant_events": 4,
                    "patch_len": 10, "resolved": False}),
    ]) + "\n")
    for instance_id in ("resolved", "rejected"):
        (first / f"{instance_id}.stream.jsonl").write_text("stream\n")
        (first / f"{instance_id}.patch").write_text(patch_text)
    # The later batch wins for repeated instance IDs, as in retry collection.
    (second / "results.jsonl").write_text(json.dumps({
        "instance_id": "rejected", "n_assistant_events": 5,
        "patch_len": 11, "resolved": False,
    }) + "\n")
    (second / "rejected.stream.jsonl").write_text("latest stream\n")
    (second / "rejected.patch").write_text("latest patch\n")

    out = tmp_path / "merged"
    assert tp.cmd_merge(SimpleNamespace(glob=str(tmp_path / "batch_*"), out_dir=str(out))) == 0

    assert {row["instance_id"] for row in tp._read_jsonl(out / "resolved.jsonl")} == {"resolved"}
    assert {row["instance_id"] for row in tp._read_jsonl(out / "rejected.jsonl")} == {"rejected"}
    assert (out / "resolved.stream.jsonl").read_text() == "stream\n"
    assert (out / "rejected" / "rejected.stream.jsonl").read_text() == "latest stream\n"
    manifest = json.loads((out / "manifest.json").read_text())
    assert manifest["rejected"] == 1
    assert manifest["rejected_streams_copied"] == 2
    assert manifest["complete"] is True
    assert manifest["artifact_bindings"] == [{
        "instance_id": "resolved",
        "batch": "batch_a",
        "stream_sha256": stream_sha,
        "patch_sha256": patch_sha,
        "admission_evidence_sha256": "a" * 64,
        "task_contract_sha256": "b" * 64,
    }]


def test_cmd_merge_uses_exact_inputs_and_binds_selection_exclusions(
    tmp_path,
    monkeypatch,
):
    import hashlib

    monkeypatch.setattr(
        tp,
        "admission_is_exact",
        lambda row: row.get("training_admitted") is True,
    )
    batch = tmp_path / "canonical"
    batch.mkdir()
    patch_text = "diff --git a/src/x.py b/src/x.py\n"
    stream_sha = hashlib.sha256(b"stream\n").hexdigest()
    patch_sha = hashlib.sha256(patch_text.encode()).hexdigest()
    rows = []
    for instance_id in ("keep", "represented-in-v2p10"):
        rows.append({
            "instance_id": instance_id,
            "n_assistant_events": 4,
            "patch_len": len(patch_text),
            "resolved": True,
            "training_admitted": True,
            "stream_sha256": stream_sha,
            "patch_sha256": patch_sha,
            "candidate_patch_sha256": patch_sha,
            "candidate_patch_paths": ["src/x.py"],
            "admission_evidence_sha256": "a" * 64,
            "task_contract_sha256": "b" * 64,
        })
        (batch / f"{instance_id}.stream.jsonl").write_text("stream\n")
        (batch / f"{instance_id}.patch").write_text(patch_text)
    (batch / "results.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows)
    )
    exclusions = tmp_path / "v2p10-overlap.json"
    exclusions.write_text(
        json.dumps(["represented-in-v2p10"]) + "\n"
    )
    out = tmp_path / "merged"

    assert tp.cmd_merge(SimpleNamespace(
        glob=None,
        inputs=[str(batch)],
        exclude_instance_ids=[str(exclusions)],
        out_dir=str(out),
    )) == 0

    assert [
        row["instance_id"]
        for row in tp._read_jsonl(out / "resolved.jsonl")
    ] == ["keep"]
    manifest = json.loads((out / "manifest.json").read_text())
    assert manifest["selection_excluded"] == 1
    assert manifest["selection_excluded_instance_ids"] == [
        "represented-in-v2p10"
    ]
    assert manifest["selection_exclusion_artifacts"] == [{
        "path": str(exclusions.resolve()),
        "sha256": hashlib.sha256(exclusions.read_bytes()).hexdigest(),
    }]
    assert manifest["input_ledgers"] == [{
        "path": str((batch / "results.jsonl").resolve()),
        "sha256": hashlib.sha256(
            (batch / "results.jsonl").read_bytes()
        ).hexdigest(),
    }]


def test_cmd_merge_rejects_tampered_admitted_artifact_atomically(
    tmp_path, monkeypatch
):
    from types import SimpleNamespace

    monkeypatch.setattr(tp, "admission_is_exact", lambda _row: True)
    batch = tmp_path / "batch"
    batch.mkdir()
    (batch / "results.jsonl").write_text(json.dumps({
        "instance_id": "tampered",
        "n_assistant_events": 4,
        "patch_len": 10,
        "resolved": True,
        "stream_sha256": "0" * 64,
        "patch_sha256": "1" * 64,
        "candidate_patch_sha256": "1" * 64,
    }) + "\n")
    (batch / "tampered.stream.jsonl").write_text("stream\n")
    (batch / "tampered.patch").write_text("patch\n")
    out = tmp_path / "merged"

    with pytest.raises(tp.ReplayContractError, match="hash mismatch"):
        tp.cmd_merge(SimpleNamespace(glob=str(batch), out_dir=str(out)))

    assert out.exists() is False
    assert list(tmp_path.glob(".merged.*")) == []


def test_full_format_gate_is_hash_bound_before_teacher_training_admission(tmp_path):
    import hashlib

    dataset = tmp_path / "teacher"
    dataset.mkdir()
    train = dataset / "train.jsonl"
    train.write_text('{"instance_id":"one"}\n')
    (dataset / "manifest.json").write_text(json.dumps({
        "schema_version": 2,
        "rendered": 1,
        "training_admitted": 0,
        "all_training_gates_complete": False,
        "train_jsonl_sha256": hashlib.sha256(train.read_bytes()).hexdigest(),
        "standard_native_format_loss_gate": {
            "status": "pending_full_dataset_verification",
            "failure_count": None,
        },
    }))
    report = {
        "samples": 1,
        "failure_count": 0,
        "failures": [],
        "counts": {"examples": 1},
        "supervised_tokens": {"min": 10, "max": 10},
    }

    assert tp._attach_teacher_format_gate(dataset, report) is True

    manifest = json.loads((dataset / "manifest.json").read_text())
    assert manifest["training_admitted"] == 1
    assert manifest["all_training_gates_complete"] is True
    gate = manifest["standard_native_format_loss_gate"]
    assert gate["status"] == "passed"
    artifact = dataset / gate["artifact"]["path"]
    assert hashlib.sha256(artifact.read_bytes()).hexdigest() == gate["artifact"]["sha256"]


def test_render_sft_preserves_distilled_step_loss_mask() -> None:
    steps = tp.NormalizedTrace(
        [
            {
                "thought": "This failed and must remain context only.",
                "command": "python3 inspect.py",
                "observation": "failed",
                "returncode": 1,
                "mutates_source": False,
                "loss": False,
            },
            {
                "thought": "Apply the verified repair.",
                "command": "sed -i 's/a/b/' src/x.py",
                "observation": "ok",
                "returncode": 0,
                "mutates_source": True,
                "loss": True,
            },
        ],
        terminal_assistant="DONE",
    )

    row = tp.render_sft(
        {
            "instance_id": "fixture__repo.pr_1",
            "problem_statement": "Fix x.",
            "repo": "fixture/repo",
            "backend": "claude",
            "model": "claude-fable-5",
        },
        steps,
        require_terminal=True,
    )

    assistants = [
        message for message in row["messages"]
        if message["role"] == "assistant"
    ]
    assert [message["loss"] for message in assistants] == [False, True, True]


def test_cmd_prepare_publishes_bound_terminal_supervision_atomically(
    tmp_path, monkeypatch
):
    from types import SimpleNamespace

    from datasets import load_from_disk
    from teacher_platform.generic_trace_replay import assess_controls

    merged = tmp_path / "merged"
    merged.mkdir()
    instance_id = "fixture__repo.pr_1"
    reference_patch = (
        "diff --git a/src/x.py b/src/x.py\n"
        "--- a/src/x.py\n"
        "+++ b/src/x.py\n"
        "@@ -1 +1 @@\n"
        "-a\n"
        "+b\n"
    )
    task = {
        "instance_id": instance_id,
        "image_name": "fixture/image:latest",
        "problem_statement": "Fix x.",
        "repo": "fixture/repo",
        "patch": reference_patch,
        "FAIL_TO_PASS": ["tests/test_x.py::test_fix"],
        "PASS_TO_PASS": ["tests/test_x.py::test_old"],
    }
    tasks = tmp_path / "tasks.jsonl"
    rejected_instance_id = "fixture__repo.pr_distill_reject"
    rejected_task = dict(task, instance_id=rejected_instance_id)
    tasks.write_text(
        json.dumps(task) + "\n" + json.dumps(rejected_task) + "\n"
    )
    exclusion = tmp_path / "exclude.json"
    exclusion.write_text(json.dumps({"instance_ids": [], "repo_denylist": []}))
    stream = merged / f"{instance_id}.stream.jsonl"
    stream.write_text("\n".join(json.dumps(row) for row in [
        {"type": "assistant", "message": {"content": [
            {"type": "text", "text": "Make the narrow edit."},
            {"type": "tool_use", "id": "call-edit", "name": "Bash",
             "input": {"command": "sed -i s/a/b/ src/x.py"}},
        ]}},
        {"type": "user", "message": {"content": [
            {
                "type": "tool_result",
                "tool_use_id": "call-edit",
                "content": "ok",
                "is_error": False,
            },
        ]}},
        {"type": "assistant", "message": {"content": [
            {"type": "text", "text": "Run the focused contract test."},
            {"type": "tool_use", "id": "call-test", "name": "Bash",
             "input": {
                 "command": "python -m pytest -q tests/test_x.py::test_fix"
             }},
        ]}},
        {"type": "user", "message": {"content": [
            {
                "type": "tool_result",
                "tool_use_id": "call-test",
                "content": "1 passed",
                "is_error": False,
            },
        ]}},
        {"type": "assistant", "message": {"content": [
            {"type": "text", "text": "Implemented and verified.\nDONE"},
        ]}},
    ]) + "\n")
    patch = merged / f"{instance_id}.patch"
    patch.write_text(reference_patch)
    contract, controls = _strict_control_set(task)
    assessment = assess_controls(controls, protected_patch_paths=())
    evidence = {
        "admission_schema_version": 2,
        "task_contract": contract,
        "task_contract_sha256": contract["contract_sha256"],
        "image_id": "sha256:" + "1" * 64,
        "candidate_patch_sha256": tp._sha256_path(patch),
        "candidate_patch_paths": ["src/x.py"],
        "protected_patch_paths": [],
        "controls": controls,
        **assessment,
    }
    record = {
        "instance_id": instance_id,
        "backend": "claude",
        "model": "claude-fable-5",
        "resolved": True,
        "stream_sha256": tp._sha256_path(stream),
        "patch_sha256": tp._sha256_path(patch),
        **evidence,
        "admission_evidence_sha256": tp.sha256_bytes(
            tp.canonical_json_bytes(evidence)
        ),
    }
    rejected_stream = merged / f"{rejected_instance_id}.stream.jsonl"
    rejected_stream.write_bytes(stream.read_bytes())
    rejected_patch = merged / f"{rejected_instance_id}.patch"
    rejected_patch.write_bytes(patch.read_bytes())
    rejected_contract, rejected_controls = _strict_control_set(rejected_task)
    rejected_assessment = assess_controls(
        rejected_controls,
        protected_patch_paths=(),
    )
    rejected_evidence = {
        "admission_schema_version": 2,
        "task_contract": rejected_contract,
        "task_contract_sha256": rejected_contract["contract_sha256"],
        "image_id": "sha256:" + "2" * 64,
        "candidate_patch_sha256": tp._sha256_path(rejected_patch),
        "candidate_patch_paths": ["src/x.py"],
        "protected_patch_paths": [],
        "controls": rejected_controls,
        **rejected_assessment,
    }
    rejected_record = {
        "instance_id": rejected_instance_id,
        "backend": "claude",
        "model": "claude-fable-5",
        "resolved": True,
        "stream_sha256": tp._sha256_path(rejected_stream),
        "patch_sha256": tp._sha256_path(rejected_patch),
        **rejected_evidence,
        "admission_evidence_sha256": tp.sha256_bytes(
            tp.canonical_json_bytes(rejected_evidence)
        ),
    }
    resolved = merged / "resolved.jsonl"
    resolved.write_text(
        json.dumps(record) + "\n" + json.dumps(rejected_record) + "\n"
    )
    (merged / "manifest.json").write_text(json.dumps({
        "schema_version": 2,
        "complete": True,
        "resolved_sha256": tp._sha256_path(resolved),
        "artifact_bindings": [
            {
                "instance_id": instance_id,
                "stream_sha256": record["stream_sha256"],
                "patch_sha256": record["patch_sha256"],
            },
            {
                "instance_id": rejected_instance_id,
                "stream_sha256": rejected_record["stream_sha256"],
                "patch_sha256": rejected_record["patch_sha256"],
            },
        ],
    }))
    out = tmp_path / "prepared"

    def replay_mutations(task_row, steps):
        if task_row["instance_id"] == rejected_instance_id:
            raise subprocess.TimeoutExpired("docker mutation replay", 300)
        return record["candidate_patch_sha256"] if steps else "0" * 64

    monkeypatch.setattr(
        "teacher_platform.success_trace_distill.replay_mutation_subsequence",
        replay_mutations,
    )

    assert tp.cmd_prepare(SimpleNamespace(
        merged=str(merged),
        tasks=[str(tasks)],
        exclude=[str(exclusion)],
        out=str(out),
    )) == 0

    messages = load_from_disk(out)[0]["messages"]
    assert messages[-1]["role"] == "assistant"
    assert messages[-1]["content"] == "Implemented and verified.\nDONE"
    assert messages[-1]["loss"] is True
    assert all(
        message["loss"] is (message["role"] == "assistant")
        for message in messages
    )
    manifest = json.loads((out / "manifest.json").read_text())
    assert manifest["complete"] is True
    assert manifest["resolved_in"] == 2
    assert manifest["rendered"] == 1
    assert manifest["distillation_exclusions"]["count"] == 1
    rejected_binding = manifest["distillation_exclusions"]["artifact_bindings"][0]
    assert rejected_binding["instance_id"] == rejected_instance_id
    assert rejected_binding["reason_type"] == "TimeoutExpired"
    assert "docker mutation replay" in rejected_binding["reason"]
    assert len(rejected_binding["rejection_sha256"]) == 64
    assert rejected_binding["stream_sha256"] == rejected_record["stream_sha256"]
    assert rejected_binding["patch_sha256"] == rejected_record["patch_sha256"]
    assert (
        rejected_binding["admission_evidence_sha256"]
        == rejected_record["admission_evidence_sha256"]
    )
    assert manifest["training_admitted"] == 0
    assert manifest["all_training_gates_complete"] is False
    assert manifest["success_path_distilled"] is True
    assert manifest["artifact_bindings"][0]["retained_commands"] == [
        "sed -i s/a/b/ src/x.py",
        "python -m pytest -q tests/test_x.py::test_fix",
    ]
    assert len(
        manifest["artifact_bindings"][0]["distilled_source_sha256"]
    ) == 64
    assert manifest["standard_native_format_loss_gate"]["status"].startswith("pending")


def test_cmd_revalidate_publishes_new_evidence_without_mutating_source(
    tmp_path, monkeypatch
):
    from types import SimpleNamespace

    from teacher_platform.generic_trace_replay import assess_controls

    source = tmp_path / "raw"
    source.mkdir()
    instance_id = "fixture__repo.pr_2"
    candidate_patch = (
        "diff --git a/src/x.py b/src/x.py\n"
        "--- a/src/x.py\n"
        "+++ b/src/x.py\n"
        "@@ -1 +1 @@\n-a\n+b\n"
    )
    stream = source / f"{instance_id}.stream.jsonl"
    patch = source / f"{instance_id}.patch"
    stream.write_text('{"type":"result"}\n')
    patch.write_text(candidate_patch)
    (source / "results.jsonl").write_text("".join(
        json.dumps(row) + "\n"
        for row in (
            {
                "instance_id": "quota-wall",
                "backend": "claude",
                "model": "claude-fable-5",
                "n_assistant_events": 67,
                "patch_len": 0,
                "error": tp.ttd.CREDIT_EXHAUSTED_ERROR,
                "resolved": False,
            },
            {
                "instance_id": "empty-dud",
                "backend": "claude",
                "model": "claude-fable-5",
                "n_assistant_events": 2,
                "patch_len": 0,
                "resolved": False,
            },
            {
                "instance_id": instance_id,
                "backend": "claude",
                "model": "claude-fable-5",
                "n_assistant_events": 8,
                "patch_len": len(candidate_patch),
                "resolved": True,
            },
        )
    ))
    task = {
        "instance_id": instance_id,
        "image_name": "fixture/image:latest",
        "problem_statement": "Fix x.",
        "repo": "fixture/repo",
        "patch": candidate_patch,
        "FAIL_TO_PASS": ["tests/test_x.py::test_fix"],
        "PASS_TO_PASS": ["tests/test_x.py::test_old"],
    }
    tasks = tmp_path / "tasks.jsonl"
    tasks.write_text(json.dumps(task) + "\n")
    exclusion = tmp_path / "exclude.json"
    exclusion.write_text(json.dumps({"instance_ids": [], "repo_denylist": []}))
    contract, controls = _strict_control_set(task)
    assessment = assess_controls(controls, protected_patch_paths=())
    evidence = {
        "admission_schema_version": 2,
        "task_contract": contract,
        "task_contract_sha256": contract["contract_sha256"],
        "image_id": "sha256:" + "2" * 64,
        "candidate_patch_sha256": tp._sha256_path(patch),
        "candidate_patch_paths": ["src/x.py"],
        "protected_patch_paths": [],
        "controls": controls,
        **assessment,
    }
    evidence["admission_evidence_sha256"] = tp.sha256_bytes(
        tp.canonical_json_bytes(evidence)
    )
    monkeypatch.setattr(
        tp.ttd,
        "sh",
        lambda *_args, **_kwargs: (evidence["image_id"] + "\n", 0),
    )
    monkeypatch.setattr(tp, "verify_candidate_patch", lambda *_args, **_kwargs: evidence)
    out = tmp_path / "revalidated"

    assert tp.cmd_revalidate(SimpleNamespace(
        glob=str(source),
        tasks=[str(tasks)],
        exclude=[str(exclusion)],
        out_dir=str(out),
        test_timeout=30,
    )) == 0

    assert patch.read_text() == candidate_patch
    assert stream.read_text() == '{"type":"result"}\n'
    results = tp._read_jsonl(out / "results.jsonl")
    assert len(results) == 1
    assert tp.admission_is_exact(results[0])
    manifest = json.loads((out / "manifest.json").read_text())
    assert manifest["schema_version"] == 2
    assert manifest["complete"] is True
    assert manifest["training_admitted"] == 1
    assert manifest["selection_policy"] == "last_real_attempt_wins"
    assert manifest["real_attempts_seen"] == 1
    assert manifest["non_real_attempts_seen"] == 2
    assert manifest["non_real_attempt_reasons"] == {
        "credit_wall": 1,
        "empty_or_dud": 1,
    }
    assert manifest["superseded_attempts"] == []
    assert manifest["artifact_bindings"][0]["task_contract_sha256"] == (
        evidence["task_contract_sha256"]
    )
    assert Path(str(out) + ".work").exists() is False


def test_cmd_revalidate_rejects_a_derived_revalidation_ledger(tmp_path):
    from types import SimpleNamespace

    source = tmp_path / "derived"
    source.mkdir()
    (source / "results.jsonl").write_text(json.dumps({
        "instance_id": "fixture__repo.pr_derived",
        "revalidation_contract_sha256": "a" * 64,
        "training_admitted": False,
    }) + "\n")
    tasks = tmp_path / "tasks.jsonl"
    tasks.write_text(json.dumps({
        "instance_id": "fixture__repo.pr_derived",
        "image_name": "fixture/image:latest",
        "patch": "diff --git a/src/x.py b/src/x.py\n",
        "FAIL_TO_PASS": ["tests/test_x.py::test_fix"],
        "PASS_TO_PASS": ["tests/test_x.py::test_old"],
    }) + "\n")
    exclusion = tmp_path / "exclude.json"
    exclusion.write_text(json.dumps({"instance_ids": [], "repo_denylist": []}))

    with pytest.raises(tp.ReplayContractError, match="derived revalidation ledger"):
        tp.cmd_revalidate(SimpleNamespace(
            glob=str(source),
            tasks=[str(tasks)],
            exclude=[str(exclusion)],
            out_dir=str(tmp_path / "out"),
            test_timeout=30,
        ))


def test_revalidation_ledger_resolver_accepts_multiple_exact_inputs(tmp_path):
    first = tmp_path / "first"
    second = tmp_path / "second"
    first.mkdir()
    second.mkdir()
    for directory in (first, second):
        (directory / "results.jsonl").write_text("")

    resolved = tp._resolve_revalidation_ledgers([
        str(first / "results.jsonl"),
        str(second),
        str(first / "results.jsonl"),
    ])

    assert resolved == sorted({
        str(first / "results.jsonl"),
        str(second / "results.jsonl"),
    })


def test_cmd_revalidate_refuses_stale_resume_after_input_change(tmp_path, monkeypatch):
    from types import SimpleNamespace

    from teacher_platform.generic_trace_replay import assess_controls

    source = tmp_path / "raw"
    source.mkdir()
    instance_id = "fixture__repo.pr_stale"
    candidate_patch = (
        "diff --git a/src/x.py b/src/x.py\n"
        "--- a/src/x.py\n"
        "+++ b/src/x.py\n"
        "@@ -1 +1 @@\n-a\n+b\n"
    )
    (source / f"{instance_id}.stream.jsonl").write_text(
        '{"type":"result"}\n', encoding="utf-8"
    )
    patch = source / f"{instance_id}.patch"
    patch.write_text(candidate_patch, encoding="utf-8")
    (source / "results.jsonl").write_text(
        json.dumps({
            "instance_id": instance_id,
            "backend": "claude",
            "model": "claude-fable-5",
            "n_assistant_events": 8,
            "patch_len": len(candidate_patch),
            "resolved": True,
        }) + "\n",
        encoding="utf-8",
    )
    task = {
        "instance_id": instance_id,
        "image_name": "fixture/image:latest",
        "problem_statement": "Fix x.",
        "repo": "fixture/repo",
        "patch": candidate_patch,
        "FAIL_TO_PASS": ["tests/test_x.py::test_fix"],
        "PASS_TO_PASS": ["tests/test_x.py::test_old"],
    }
    tasks = tmp_path / "tasks.jsonl"
    tasks.write_text(json.dumps(task) + "\n", encoding="utf-8")
    exclusion = tmp_path / "exclude.json"
    exclusion.write_text(
        json.dumps({"instance_ids": [], "repo_denylist": []}), encoding="utf-8"
    )
    image_id = "sha256:" + "3" * 64
    contract, controls = _strict_control_set(task)
    assessment = assess_controls(controls, protected_patch_paths=())
    evidence = {
        "admission_schema_version": 2,
        "task_contract": contract,
        "task_contract_sha256": contract["contract_sha256"],
        "image_id": image_id,
        "candidate_patch_sha256": tp._sha256_path(patch),
        "candidate_patch_paths": ["src/x.py"],
        "protected_patch_paths": [],
        "controls": controls,
        **assessment,
    }
    evidence["admission_evidence_sha256"] = tp.sha256_bytes(
        tp.canonical_json_bytes(evidence)
    )
    verify_calls = []
    monkeypatch.setattr(tp.ttd, "sh", lambda *_args, **_kwargs: (image_id + "\n", 0))
    monkeypatch.setattr(
        tp,
        "verify_candidate_patch",
        lambda *_args, **_kwargs: verify_calls.append(True) or evidence,
    )
    monkeypatch.setattr(
        tp,
        "_publish_directory_atomic",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("stop before publish")),
    )
    out = tmp_path / "revalidated"
    args = SimpleNamespace(
        glob=str(source),
        tasks=[str(tasks)],
        exclude=[str(exclusion)],
        out_dir=str(out),
        test_timeout=30,
    )

    with pytest.raises(RuntimeError, match="stop before publish"):
        tp.cmd_revalidate(args)
    assert verify_calls == [True]

    task["problem_statement"] = "Changed task metadata."
    tasks.write_text(json.dumps(task) + "\n", encoding="utf-8")
    with pytest.raises(tp.ReplayContractError, match="inputs changed"):
        tp.cmd_revalidate(args)
    assert verify_calls == [True]


def test_collect_loop_stops_pass_and_backs_off_on_openrouter_429(monkeypatch, tmp_path):
    from types import SimpleNamespace

    tasks = tmp_path / "tasks.jsonl"
    tasks.write_text("\n".join(json.dumps({"instance_id": instance_id}) for instance_id in ("a", "b")) + "\n")
    calls = []
    sleeps = []
    probes = iter((True, True, False))

    def rate_limited(row, *_args):
        calls.append(row["instance_id"])
        return {"instance_id": row["instance_id"], "resolved": False,
                "error": "openrouter: Error code: 429 - rate limited upstream"}

    monkeypatch.setattr(tp, "collect_one_openrouter", rate_limited)
    monkeypatch.setattr(tp, "_probe_quota", lambda *_args: next(probes))
    monkeypatch.setattr(tp, "_docker_free_gib", lambda: 1000)
    monkeypatch.setattr(tp.subprocess, "run", lambda *_args, **_kwargs: SimpleNamespace())
    monkeypatch.setattr(tp.time, "sleep", sleeps.append)
    args = SimpleNamespace(
        tasks=str(tasks), out_dir=str(tmp_path / "run"), backend="openrouter",
        model="poolside/laguna-s-2.1", max_turns=1, loop=True,
        max_consecutive_credit_hits=3, max_walls=2, wall_sleep=99,
        rate_limit_sleep=7, floor_gib=40, exclude_runs=[],
    )

    assert tp.cmd_collect(args) == 0
    assert calls == ["a", "a"]
    assert sleeps == [7]


def test_collect_loop_counts_credit_passes_against_wall_budget(monkeypatch, tmp_path):
    from types import SimpleNamespace

    tasks = tmp_path / "tasks.jsonl"
    tasks.write_text(json.dumps({"instance_id": "a"}) + "\n")
    calls = []
    sleeps = []

    def credit_walled(row, *_args):
        calls.append(row["instance_id"])
        if len(calls) > 2:
            raise AssertionError("credit-walled passes must consume the wall budget")
        return {
            "instance_id": row["instance_id"],
            "n_assistant_events": 67,
            "patch_len": 0,
            "resolved": False,
            "error": tp.ttd.CREDIT_EXHAUSTED_ERROR,
        }

    monkeypatch.setattr(tp.ttd, "collect_one", credit_walled)
    monkeypatch.setattr(tp, "_probe_quota", lambda *_args: True)
    monkeypatch.setattr(tp, "_docker_free_gib", lambda: 1000)
    monkeypatch.setattr(
        tp.subprocess,
        "run",
        lambda *_args, **_kwargs: SimpleNamespace(),
    )
    monkeypatch.setattr(tp.time, "sleep", sleeps.append)
    args = SimpleNamespace(
        tasks=str(tasks),
        out_dir=str(tmp_path / "run"),
        backend="claude",
        model="claude-fable-5",
        max_turns=1,
        loop=True,
        max_consecutive_credit_hits=1,
        max_walls=2,
        wall_sleep=99,
        rate_limit_sleep=7,
        floor_gib=40,
        exclude_runs=[],
    )

    assert tp.cmd_collect(args) == 0
    assert calls == ["a", "a"]
    assert sleeps == [99]


def test_collect_cli_defaults_to_bounded_credit_wall_retries() -> None:
    args = tp.build_parser().parse_args([
        "collect",
        "--tasks",
        "tasks.jsonl",
        "--out-dir",
        "runs/teacher",
    ])

    assert args.max_walls == 3


def test_cross_run_exclusion(tmp_path):
    import json
    (tmp_path / "runA").mkdir()
    (tmp_path / "runB").mkdir()
    (tmp_path / "runA" / "results.jsonl").write_text(
        json.dumps({"instance_id": "x1", "n_assistant_events": 40, "patch_len": 10}) + "\n" +
        json.dumps({"instance_id": "dud", "n_assistant_events": 1, "patch_len": 0}) + "\n")
    (tmp_path / "runB" / "results.jsonl").write_text(
        json.dumps({"instance_id": "x2", "n_assistant_events": 0, "patch_len": 500}) + "\n")
    done = tp._attempted_from_run_globs(str(tmp_path / "run") )
    assert done == {"x1", "x2"}          # duds excluded, both real attempts caught across dirs


def test_collection_scheduler_consumes_mutation_baseline_rejection_once(tmp_path):
    tasks = tmp_path / "tasks.jsonl"
    tasks.write_text(
        "\n".join(
            json.dumps({"instance_id": instance_id})
            for instance_id in (
                "invalid-baseline",
                "orphaned-artifact",
                "credit-wall",
            )
        )
        + "\n"
    )
    run = tmp_path / "campaign_run1"
    run.mkdir()
    (run / "results.jsonl").write_text(
        json.dumps(
            {
                "instance_id": "invalid-baseline",
                "n_assistant_events": 0,
                "patch_len": 0,
                "error": (
                    "mutation_baseline: clean reference controls "
                    "did not pass exactly"
                ),
            }
        )
        + "\n"
        + json.dumps(
            {
                "instance_id": "orphaned-artifact",
                "n_assistant_events": 0,
                "patch_len": 0,
                "error": (
                    "artifact_collision: raw stream or patch already exists"
                ),
            }
        )
        + "\n"
        + json.dumps(
            {
                "instance_id": "credit-wall",
                "n_assistant_events": 67,
                "patch_len": 0,
                "error": tp.ttd.CREDIT_EXHAUSTED_ERROR,
            }
        )
        + "\n"
    )

    remaining = tp._remaining(
        str(tasks),
        str(tmp_path / "campaign"),
    )

    assert remaining == [{"instance_id": "credit-wall"}]


def test_collect_publishes_restart_ledger_atomically(monkeypatch, tmp_path):
    from types import SimpleNamespace

    tasks = tmp_path / "tasks.jsonl"
    tasks.write_text(json.dumps({"instance_id": "case"}) + "\n")
    out = tmp_path / "campaign"
    writes = []
    atomic_write = tp._replace_jsonl_atomic

    def record_atomic_write(path, rows):
        writes.append((path, list(rows)))
        atomic_write(path, rows)

    monkeypatch.setattr(
        tp.ttd,
        "collect_one",
        lambda *_args: {
            "instance_id": "case",
            "n_assistant_events": 3,
            "patch_len": 1,
            "resolved": False,
            "error": None,
        },
    )
    monkeypatch.setattr(tp, "_replace_jsonl_atomic", record_atomic_write)
    args = SimpleNamespace(
        tasks=str(tasks),
        out_dir=str(out),
        backend="claude",
        model="claude-fable-5",
        max_turns=1,
        loop=False,
        max_consecutive_credit_hits=3,
        max_walls=3,
        wall_sleep=1,
        rate_limit_sleep=1,
        floor_gib=40,
        exclude_runs=[],
    )

    assert tp.cmd_collect(args) == 0
    assert writes == [
        (
            out / "results.jsonl",
            [
                {
                    "instance_id": "case",
                    "n_assistant_events": 3,
                    "patch_len": 1,
                    "resolved": False,
                    "error": None,
                }
            ],
        )
    ]


def test_edit_first_trim_keeps_edit_spine():
    steps = [{"thought": "", "command": f"grep r{i}", "observation": ""} for i in range(20)]
    steps[15]["command"] = "sed -i s/a/b/ x.py"      # first edit late
    steps.append({"thought": "", "command": "pytest -x", "observation": ""})
    out = tp.edit_first_trim(steps, max_steps=10)
    cmds = [s["command"] for s in out]
    assert "sed -i s/a/b/ x.py" in cmds and "pytest -x" in cmds   # edit + verify retained
    assert len(out) <= 10
    # short trajectories pass through untouched
    short = steps[:5]
    assert tp.edit_first_trim(short, max_steps=10) == short


@pytest.mark.parametrize(
    "arguments",
    [
        {"command": "create", "path": "/testbed/new.py", "file_text": "x = 1\n"},
        {
            "command": "str_replace",
            "path": "/testbed/module.py",
            "old_str": "old",
            "new_str": "new",
        },
        {
            "command": "insert",
            "path": "/testbed/module.py",
            "insert_line": 1,
            "new_str": "new line",
        },
    ],
)
def test_edit_first_trim_retains_translated_editor_mutations(arguments):
    edit = tp.translate_trajectory_tool_call(_trajectory_tool_call("str_replace_editor", arguments))
    steps = [
        {"thought": "", "command": f"grep q{i}", "observation": ""}
        for i in range(30)
    ]
    steps[4]["command"] = edit

    out = tp.edit_first_trim(steps, max_steps=25)

    assert edit in [step["command"] for step in out]
    assert len(out) <= 25


def test_obs_truncation_head_tail():
    msgs = [{"role": "assistant", "content": "t",
             "tool_calls": [{"function": {"name": "bash", "arguments": '{"command": "cat big"}'}}]},
            {"role": "tool", "content": "OBSERVATION:\n" + "X" * 5000}]
    steps = tp.steps_from_trajectory(msgs, max_obs=400)
    assert len(steps[0]["observation"]) < 500 and "truncated" in steps[0]["observation"]


@pytest.mark.parametrize("max_obs", [0, -1])
def test_steps_from_trajectory_rejects_nonpositive_observation_cap(max_obs):
    with pytest.raises(ValueError, match="max_obs must be positive"):
        tp.steps_from_trajectory([], max_obs=max_obs)


def test_edit_first_trim_rejects_negative_step_cap_and_accepts_zero_disabled():
    steps = [{"thought": "", "command": "pwd", "observation": ""}]

    with pytest.raises(ValueError, match="max_steps must be nonnegative"):
        tp.edit_first_trim(steps, max_steps=-1)
    assert tp.edit_first_trim(steps, max_steps=0) == steps


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("limit", -1, "limit must be nonnegative"),
        ("max_steps", -1, "max_steps must be nonnegative"),
        ("max_pr_chars", -1, "max_pr_chars must be nonnegative"),
        ("max_obs_chars", 0, "max_obs_chars must be positive"),
        ("max_obs_chars", -1, "max_obs_chars must be positive"),
        ("progress_every", 0, "progress_every must be positive"),
        ("progress_every", -1, "progress_every must be positive"),
    ],
)
def test_cmd_ingest_defensively_rejects_invalid_limits(
    monkeypatch, tmp_path, field, value, message
):
    import datasets
    from types import SimpleNamespace

    def unexpected_load(*args, **kwargs):
        raise AssertionError("validation must run before dataset loading")

    monkeypatch.setattr(datasets, "load_dataset", unexpected_load)
    values = {
        "out": str(tmp_path / "out"),
        "dataset": "fixture/open-swe",
        "configs": ["fixture"],
        "language": "python",
        "resolved_only": True,
        "limit": 0,
        "max_obs_chars": 800,
        "max_steps": 25,
        "max_pr_chars": 6000,
        "progress_every": 10_000,
        "exclude": [],
    }
    values[field] = value

    with pytest.raises(ValueError, match=message):
        tp.cmd_ingest(SimpleNamespace(**values))


@pytest.mark.parametrize(
    ("flag", "value"),
    [
        ("--limit", "-1"),
        ("--max-steps", "-1"),
        ("--max-pr-chars", "-1"),
        ("--max-obs-chars", "0"),
        ("--max-obs-chars", "-1"),
    ],
)
def test_ingest_cli_rejects_invalid_limits(flag, value):
    with pytest.raises(SystemExit) as error:
        tp.build_parser().parse_args(["ingest", "--out", "out", flag, value])

    assert error.value.code == 2


def test_ingest_cli_accepts_documented_zero_disabled_boundaries():
    args = tp.build_parser().parse_args([
        "ingest",
        "--out", "out",
        "--limit", "0",
        "--max-steps", "0",
        "--max-pr-chars", "0",
        "--max-obs-chars", "1",
    ])

    assert args.limit == 0
    assert args.max_steps == 0
    assert args.max_pr_chars == 0
    assert args.max_obs_chars == 1


def test_ingest_cli_progress_interval_defaults_and_rejects_nonpositive_values():
    args = tp.build_parser().parse_args(["ingest", "--out", "out"])

    assert args.progress_every == 10_000

    for value in ("0", "-1"):
        with pytest.raises(SystemExit) as error:
            tp.build_parser().parse_args([
                "ingest", "--out", "out", "--progress-every", value,
            ])
        assert error.value.code == 2


def test_cmd_ingest_reports_exact_progress_from_metadata_without_reloading(
    monkeypatch, tmp_path, capsys
):
    import datasets
    from types import SimpleNamespace

    def row(index):
        return {
            "instance_id": f"row-{index}",
            "repo": "fixture/repo",
            "language": "python",
            "resolved": 1,
            "trajectory": [
                {"role": "user", "content": f"Fix row {index}."},
                {
                    "role": "assistant",
                    "content": "Inspect the repository.",
                    "tool_calls": [_trajectory_tool_call("bash", {"command": "pwd"})],
                },
                {"role": "tool", "content": "/testbed"},
            ],
        }

    rows_by_config = {
        "first": [row(1), row(2), row(3)],
        "second": [row(4), row(5)],
    }
    load_calls = []

    class StreamingSplit(list):
        def __init__(self, split_name, rows):
            super().__init__(rows)
            self.info = SimpleNamespace(splits={
                split_name: SimpleNamespace(num_examples=len(rows)),
            })

    def fake_load_dataset(dataset, config, streaming):
        assert (dataset, streaming) == ("fixture/open-swe", True)
        load_calls.append(config)
        return {"train": StreamingSplit("train", rows_by_config[config])}

    class FakeSavedDataset:
        def save_to_disk(self, path):
            Path(path).mkdir()

    class FakeDataset:
        @staticmethod
        def from_list(_rows):
            return FakeSavedDataset()

    clock = iter([100.0, 102.0, 104.0, 105.0])
    monkeypatch.setattr(datasets, "load_dataset", fake_load_dataset)
    monkeypatch.setattr(datasets, "Dataset", FakeDataset)
    monkeypatch.setattr(tp.time, "monotonic", lambda: next(clock))
    args = SimpleNamespace(
        out=str(tmp_path / "out"),
        dataset="fixture/open-swe",
        configs=["first", "second"],
        language="python",
        resolved_only=True,
        limit=0,
        max_obs_chars=800,
        max_steps=0,
        max_pr_chars=6000,
        progress_every=2,
        exclude=[],
    )

    assert tp.cmd_ingest(args) == 0

    progress = [
        json.loads(line)
        for line in capsys.readouterr().out.splitlines()
        if line.startswith("{")
    ]
    assert load_calls == ["first", "second"]
    assert progress == [
        {
            "scanned": 2,
            "total": 5,
            "percent": 40.0,
            "resolved_matched": 2,
            "ingested": 2,
            "elapsed_seconds": 2.0,
            "rows_per_second": 1.0,
            "eta_seconds": 3.0,
        },
        {
            "scanned": 4,
            "total": 5,
            "percent": 80.0,
            "resolved_matched": 4,
            "ingested": 4,
            "elapsed_seconds": 4.0,
            "rows_per_second": 1.0,
            "eta_seconds": 1.0,
        },
        {
            "scanned": 5,
            "total": 5,
            "percent": 100.0,
            "resolved_matched": 5,
            "ingested": 5,
            "elapsed_seconds": 5.0,
            "rows_per_second": 1.0,
            "eta_seconds": 0.0,
        },
    ]


def test_cmd_ingest_progress_handles_missing_metadata_and_dedupes_aligned_final(
    monkeypatch, tmp_path, capsys
):
    import datasets
    from types import SimpleNamespace

    rows = [
        {
            "instance_id": f"filtered-{index}",
            "language": "javascript",
            "resolved": 1,
        }
        for index in range(3)
    ]
    load_calls = []

    def fake_load_dataset(dataset, config, streaming):
        assert (dataset, config, streaming) == (
            "fixture/open-swe", "fixture", True,
        )
        load_calls.append(config)
        return {"train": rows}

    clock = iter([10.0, 13.0])
    monkeypatch.setattr(datasets, "load_dataset", fake_load_dataset)
    monkeypatch.setattr(tp.time, "monotonic", lambda: next(clock))
    args = SimpleNamespace(
        out=str(tmp_path / "out"),
        dataset="fixture/open-swe",
        configs=["fixture"],
        language="python",
        resolved_only=True,
        limit=0,
        max_obs_chars=800,
        max_steps=0,
        max_pr_chars=6000,
        progress_every=3,
        exclude=[],
    )

    assert tp.cmd_ingest(args) == 1

    progress = [
        json.loads(line)
        for line in capsys.readouterr().out.splitlines()
        if line.startswith("{")
    ]
    assert load_calls == ["fixture"]
    assert progress == [{
        "scanned": 3,
        "total": None,
        "percent": None,
        "resolved_matched": 0,
        "ingested": 0,
        "elapsed_seconds": 3.0,
        "rows_per_second": 1.0,
        "eta_seconds": None,
    }]


def test_ingest_cli_retains_valid_rich_row_and_audits_malformed_drop(run_ingest_cli):
    valid_trajectory = [
        {"role": "user", "content": "Create the missing module."},
        {
            "role": "assistant",
            "reasoning_content": "Write the requested implementation.",
            "tool_calls": [_trajectory_tool_call(
                "str_replace_editor",
                {
                    "command": "create",
                    "path": "/testbed/module.py",
                    "file_text": "answer = 42\n",
                },
            )],
        },
        {"role": "tool", "content": "created"},
    ]
    malformed_trajectory = [
        {"role": "user", "content": "Inspect the module."},
        {
            "role": "assistant",
            "tool_calls": [{
                "function": {"name": "str_replace_editor", "arguments": "{"},
            }],
        },
        {"role": "tool", "content": "never reached"},
    ]
    rows = [
        {
            "instance_id": "valid-rich",
            "repo": "fixture/repo",
            "language": "python",
            "resolved": 1,
            "trajectory": valid_trajectory,
        },
        {
            "instance_id": "malformed-rich",
            "repo": "fixture/repo",
            "language": "python",
            "resolved": 1,
            "trajectory": malformed_trajectory,
        },
    ]

    result, out, saved_rows = run_ingest_cli(rows)

    assert result == 0
    assert [row["instance_id"] for row in saved_rows] == ["valid-rich"]
    manifest = json.loads(Path(str(out) + ".manifest.json").read_text())
    assert {
        "scanned": manifest["scanned"],
        "resolved_matched": manifest["resolved_matched"],
        "ingested": manifest["ingested"],
        "dropped_tool_conversion": manifest["dropped_tool_conversion"],
        "tool_conversion_drop_reasons": manifest["tool_conversion_drop_reasons"],
        "per_source": manifest["per_source"],
    } == {
        "scanned": 2,
        "resolved_matched": 2,
        "ingested": 1,
        "dropped_tool_conversion": 1,
        "tool_conversion_drop_reasons": {"malformed tool arguments": 1},
        "per_source": {"fixture/train": 1},
    }
    assistant = next(
        message for message in saved_rows[0]["messages"] if message["role"] == "assistant"
    )
    rendered_command = json.loads(
        assistant["tool_calls"][0]["function"]["arguments"]
    )["command"]
    assert "# OPEN_SWE_EDITOR_MUTATION" in rendered_command


def test_ingest_cli_converts_raw_openhands_fixture_without_pseudo_tools(run_ingest_cli):
    declared_root = "/workspace/demo-repo"
    rows = [{
        "instance_id": "raw-openhands",
        "repo": "fixture/repo",
        "language": "python",
        "resolved": 1,
        "trajectory": [
            {
                "role": "user",
                "content": (
                    f"<uploaded_files>\n{declared_root}\n</uploaded_files>\n"
                    "Fix the failing behavior."
                ),
            },
            {
                "role": "assistant",
                "reasoning_content": "Run a focused check.",
                "tool_calls": [{
                    "id": "call-exec",
                    "type": "function",
                    "function": {
                        "name": "execute_bash",
                        "arguments": json.dumps({
                            "input": f"cd {declared_root} && printf raw-ok",
                        }),
                    },
                }],
            },
            {"role": "tool", "content": "raw-ok"},
            {
                "role": "assistant",
                "reasoning_content": "Inspect the target file.",
                "tool_calls": [_trajectory_tool_call(
                    "str_replace_editor",
                    {
                        "command": "view",
                        "path": f"{declared_root}/src/module.py",
                    },
                )],
            },
            {"role": "tool", "content": "1\tvalue = 1"},
        ],
    }]

    result, _, saved_rows = run_ingest_cli(rows, output_name="raw-openhands")

    assert result == 0
    messages = saved_rows[0]["messages"]
    assistant_calls = [
        call
        for message in messages
        if message["role"] == "assistant"
        for call in message["tool_calls"]
    ]
    assert assistant_calls
    assert {call["function"]["name"] for call in assistant_calls} == {"bash"}
    rendered_commands = [
        json.loads(call["function"]["arguments"])["command"]
        for call in assistant_calls
    ]
    assert rendered_commands[0] == "printf raw-ok"
    assert "/testbed/src/module.py" in rendered_commands[1]
    assert all(declared_root not in command for command in rendered_commands)
    execution = subprocess.run(
        ["bash", "-c", rendered_commands[0]], capture_output=True, text=True
    )
    assert execution.returncode == 0
    assert execution.stdout == "raw-ok"
    observations = [
        message["content"] for message in messages if message["role"] == "user"
    ][1:]
    assert observations == ["OBSERVATION:\nraw-ok", "OBSERVATION:\n1\tvalue = 1"]
    serialized = json.dumps(messages)
    assert "execute_bash" not in serialized
    assert "str_replace_editor" not in serialized


def test_ingest_cli_writes_audit_manifest_when_every_eligible_row_drops(
    run_ingest_cli,
):
    rows = [{
        "instance_id": "malformed-only",
        "repo": "fixture/repo",
        "language": "python",
        "resolved": 1,
        "trajectory": [
            {"role": "user", "content": "Inspect the module."},
            {
                "role": "assistant",
                "tool_calls": [{
                    "function": {"name": "str_replace_editor", "arguments": "{"},
                }],
            },
            {"role": "tool", "content": "never reached"},
        ],
    }]

    result, out, saved_rows = run_ingest_cli(rows, output_name="all-dropped")

    assert result == 1
    manifest_path = Path(str(out) + ".manifest.json")
    assert manifest_path.exists()
    manifest = json.loads(manifest_path.read_text())
    assert {
        "scanned": manifest["scanned"],
        "resolved_matched": manifest["resolved_matched"],
        "ingested": manifest["ingested"],
        "dropped_tool_conversion": manifest["dropped_tool_conversion"],
        "tool_conversion_drop_reasons": manifest["tool_conversion_drop_reasons"],
        "per_source": manifest["per_source"],
    } == {
        "scanned": 1,
        "resolved_matched": 1,
        "ingested": 0,
        "dropped_tool_conversion": 1,
        "tool_conversion_drop_reasons": {"malformed tool arguments": 1},
        "per_source": {},
    }
    assert saved_rows == []
    assert not out.exists()
    assert not Path(str(out) + ".jsonl").exists()


def test_ingest_cli_does_not_hide_unexpected_converter_errors(
    run_ingest_cli, monkeypatch
):
    rows = [{
        "instance_id": "unexpected",
        "repo": "fixture/repo",
        "language": "python",
        "resolved": 1,
        "trajectory": [],
    }]

    def raise_unexpected(*args, **kwargs):
        raise RuntimeError("unexpected converter failure")

    monkeypatch.setattr(tp, "steps_from_trajectory", raise_unexpected)

    with pytest.raises(RuntimeError, match="unexpected converter failure"):
        run_ingest_cli(rows, output_name="unexpected")


def test_blend_dedup_earlier_source_wins(tmp_path):
    import json
    a = tmp_path / "a.jsonl"
    b = tmp_path / "b.jsonl"
    a.write_text(json.dumps({"instance_id": "i1", "messages": [1], "source": "A"}) + "\n")
    b.write_text(json.dumps({"instance_id": "i1", "messages": [2], "source": "B"}) + "\n" +
                 json.dumps({"instance_id": "i2", "messages": [3], "source": "B"}) + "\n")
    from types import SimpleNamespace
    out = tmp_path / "blended"
    tp.cmd_blend(SimpleNamespace(sources=[str(a), str(b)], out=str(out), cap=0))
    rows = tp._read_jsonl(out / "train.jsonl")
    by = {r["instance_id"]: r for r in rows}
    assert set(by) == {"i1", "i2"}
    assert by["i1"]["source"] == "A"      # earlier source wins on dedup
    manifest = json.loads((out / "manifest.json").read_text())
    assert manifest["training_admitted"] == 0
    assert manifest["all_training_gates_complete"] is False


def test_blend_rejects_pending_schema_v2_teacher_source(tmp_path):
    from types import SimpleNamespace

    source = tmp_path / "pending"
    source.mkdir()
    (source / "manifest.json").write_text(json.dumps({
        "schema_version": 2,
        "all_training_gates_complete": False,
    }))

    with pytest.raises(tp.ReplayContractError, match="incomplete"):
        tp.cmd_blend(SimpleNamespace(
            sources=[str(source)],
            out=str(tmp_path / "blend"),
            cap=0,
        ))


def test_normalize_steps_claude(tmp_path):
    import json
    p = tmp_path / "c.stream.jsonl"
    p.write_text("\n".join(json.dumps(x) for x in [
        {"type": "assistant", "message": {"content": [
            {"type": "text", "text": "inspect"},
            {"type": "tool_use", "name": "Bash",
             "input": {"command": 'docker exec cid bash -c "cd /testbed && ls"'}}]}},
        {"type": "user", "message": {"content": [
            {
                "type": "tool_result",
                "content": "a.py b.py",
                "is_error": False,
            }]}},
    ]))
    steps = tp.normalize_steps("claude", p)
    assert len(steps) == 1
    assert steps[0]["command"] == "ls"          # docker-exec wrapper stripped
    assert steps[0]["thought"] == "inspect"
    assert steps[0]["observation"] == "a.py b.py"
    assert steps[0]["returncode"] == 0
    assert steps[0]["mutates_source"] is False


def test_normalize_steps_claude_preserves_failed_mutation_outcome(tmp_path):
    p = tmp_path / "failed-mutation.stream.jsonl"
    p.write_text("\n".join(json.dumps(x) for x in [
        {"type": "assistant", "message": {"content": [
            {"type": "text", "text": "edit"},
            {
                "type": "tool_use",
                "id": "call-edit",
                "name": "Bash",
                "input": {"command": "sed -i 's/a/b/' src/x.py"},
            },
        ]}},
        {"type": "user", "message": {"content": [{
            "type": "tool_result",
            "tool_use_id": "call-edit",
            "content": "sed: no match",
            "is_error": True,
        }]}},
    ]) + "\n")

    steps = tp.normalize_steps("claude", p)

    assert steps[0]["returncode"] == 1
    assert steps[0]["mutates_source"] is True


def test_normalize_steps_claude_pairs_parallel_results_and_keeps_terminal(tmp_path):
    p = tmp_path / "parallel.stream.jsonl"
    p.write_text("\n".join(json.dumps(x) for x in [
        {"type": "assistant", "message": {"content": [
            {"type": "text", "text": "inspect and test"},
            {"type": "tool_use", "id": "call-a", "name": "Bash",
             "input": {"command": "cat a.py"}},
            {"type": "tool_use", "id": "call-b", "name": "Bash",
             "input": {"command": "pytest -q"}},
        ]}},
        {"type": "user", "message": {"content": [
            {"type": "tool_result", "tool_use_id": "call-b", "content": "1 passed"},
            {"type": "tool_result", "tool_use_id": "call-a", "content": "SOURCE"},
        ]}},
        {"type": "assistant", "message": {"content": [
            {"type": "text", "text": "Implemented and verified.\nDONE"},
        ]}},
    ]) + "\n")

    trace = tp.normalize_steps("claude", p)

    assert [step["command"] for step in trace] == ["cat a.py", "pytest -q"]
    assert [step["observation"] for step in trace] == ["SOURCE", "1 passed"]
    assert trace.terminal_assistant == "Implemented and verified.\nDONE"


@pytest.mark.parametrize(
    "result_blocks,match",
    [
        (
            [{"type": "tool_result", "tool_use_id": "unknown", "content": "x"}],
            "orphan",
        ),
        (
            [
                {"type": "tool_result", "tool_use_id": "call-a", "content": "x"},
                {"type": "tool_result", "tool_use_id": "call-a", "content": "y"},
            ],
            "duplicate",
        ),
        ([], "missing"),
    ],
)
def test_normalize_steps_claude_rejects_invalid_result_ledger(
    tmp_path, result_blocks, match
):
    p = tmp_path / f"{match}.stream.jsonl"
    p.write_text("\n".join(json.dumps(x) for x in [
        {"type": "assistant", "message": {"content": [
            {"type": "tool_use", "id": "call-a", "name": "Bash",
             "input": {"command": "cat a.py"}},
        ]}},
        {"type": "user", "message": {"content": result_blocks}},
        {"type": "assistant", "message": {"content": [
            {"type": "text", "text": "DONE"},
        ]}},
    ]) + "\n")

    with pytest.raises(tp.TraceNormalizationError, match=match):
        tp.normalize_steps("claude", p)


def test_direct_script_cli_entrypoint_avoids_package_shadowing():
    root = Path(__file__).resolve().parents[2]
    result = subprocess.run(
        [sys.executable, "teacher_platform/teacher_platform.py", "--help"],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "Teacher-trace platform" in result.stdout


def test_smoke_micro_train_uses_the_current_trainer_cli(monkeypatch) -> None:
    commands = []
    monkeypatch.setattr(
        tp,
        "_load_rows_any",
        lambda _path: [{
            "messages": [
                {"role": "system", "content": "system"},
                {
                    "role": "assistant",
                    "content": "run",
                    "tool_calls": [{"function": {"name": "bash"}}],
                },
            ],
        }],
    )

    def run(command, **_kwargs):
        commands.append(command)
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(tp.subprocess, "run", run)
    args = SimpleNamespace(
        data="not-a-real-dataset",
        skip_format=True,
        train_python=sys.executable,
        micro_train=True,
        micro_out="adapters/_fixture_micro",
        micro_max_seq=16384,
    )

    assert tp.cmd_smoke_train(args) == 0

    command = commands[-1]
    assert command[command.index("--max-steps") + 1] == "2"
    assert command[command.index("--rank") + 1] == "16"
    assert command[command.index("--max-seq") + 1] == "16384"
    assert "--train-steps" not in command
    assert "--lora-r" not in command
