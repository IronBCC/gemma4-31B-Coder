"""Unit tests for teacher_platform pure logic (no docker / no network)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import teacher_platform as tp  # noqa: E402


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
