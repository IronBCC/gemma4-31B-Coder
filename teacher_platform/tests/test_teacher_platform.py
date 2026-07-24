"""Unit tests for teacher_platform pure logic (no docker / no network)."""
import json
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import teacher_platform as tp  # noqa: E402


def _trajectory_tool_call(name, arguments):
    return {"function": {"name": name, "arguments": json.dumps(arguments)}}


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


def test_cmd_merge_writes_resolved_and_rejected_banks(tmp_path):
    from types import SimpleNamespace

    first = tmp_path / "batch_a"
    second = tmp_path / "batch_b"
    first.mkdir()
    second.mkdir()
    (first / "results.jsonl").write_text("\n".join([
        json.dumps({"instance_id": "resolved", "n_assistant_events": 4,
                    "patch_len": 10, "resolved": True}),
        json.dumps({"instance_id": "rejected", "n_assistant_events": 4,
                    "patch_len": 10, "resolved": False}),
    ]) + "\n")
    for instance_id in ("resolved", "rejected"):
        (first / f"{instance_id}.stream.jsonl").write_text("stream\n")
        (first / f"{instance_id}.patch").write_text("patch\n")
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
    assert sleeps == [7, 7]


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
    rows = tp._read_jsonl(str(out) + ".jsonl")
    by = {r["instance_id"]: r for r in rows}
    assert set(by) == {"i1", "i2"}
    assert by["i1"]["source"] == "A"      # earlier source wins on dedup


def test_normalize_steps_claude(tmp_path):
    import json
    p = tmp_path / "c.stream.jsonl"
    p.write_text("\n".join(json.dumps(x) for x in [
        {"type": "assistant", "message": {"content": [
            {"type": "text", "text": "inspect"},
            {"type": "tool_use", "name": "Bash",
             "input": {"command": 'docker exec cid bash -c "cd /testbed && ls"'}}]}},
        {"type": "user", "message": {"content": [
            {"type": "tool_result", "content": "a.py b.py"}]}},
    ]))
    steps = tp.normalize_steps("claude", p)
    assert len(steps) == 1
    assert steps[0]["command"] == "ls"          # docker-exec wrapper stripped
    assert steps[0]["thought"] == "inspect"
    assert steps[0]["observation"] == "a.py b.py"
