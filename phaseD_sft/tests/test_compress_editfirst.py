from __future__ import annotations

import json

from phaseD_sft.agentic_trace_filters import command_trace_quality_report
from phaseD_sft.compress_editfirst import compress_editfirst_trace, edited_paths


def _assistant(command: str, reasoning: str) -> dict[str, object]:
    return {
        "role": "assistant",
        "content": reasoning,
        "tool_calls": [{
            "id": f"call_{command[:4]}",
            "type": "function",
            "function": {"name": "bash", "arguments": json.dumps({"command": command})},
        }],
    }


def _observation(content: str) -> dict[str, object]:
    return {"role": "user", "content": f"OBSERVATION:\n{content}", "tool_calls": []}


def _trace() -> dict[str, object]:
    return {
        "instance_id": "example",
        "messages": [
            {"role": "system", "content": "Solve the issue.", "tool_calls": []},
            {"role": "user", "content": "PR: fix pkg/target.py", "tool_calls": []},
            _assistant("ls", "First exploratory thought."),
            _observation("README.md\npkg/other.py"),
            _assistant("sed -n '1,200p' pkg/target.py", "Read target one."),
            _observation("pkg/target.py\ndef target():\n    return 'old'"),
            _assistant("rg 'target' pkg/target.py", "Read target two."),
            _observation("pkg/target.py:1:def target():"),
            _assistant("git status", "Read unrelated state."),
            _observation("On branch main"),
            _assistant("grep -n old pkg/target.py", "Read target three."),
            _observation("pkg/target.py:2:    return 'old'"),
            _assistant("sed -i \"s/'old'/'new'/\" pkg/target.py", "Apply the minimal edit verbatim."),
            _observation("changed pkg/target.py"),
            _assistant("pytest -q tests/test_target.py", "Run the focused test verbatim."),
            _observation("1 passed"),
            _assistant("git diff -- pkg/target.py", "Inspect and submit the diff verbatim."),
            _observation("diff --git a/pkg/target.py b/pkg/target.py"),
        ],
    }


def test_compression_drops_exploration_and_moves_first_edit_to_fourth_command() -> None:
    compressed, report = compress_editfirst_trace(_trace(), keep_pre_edit_reads=3)

    assert report["kept"] is True
    assert compressed is not None
    commands = [
        json.loads(call["function"]["arguments"])["command"]
        for message in compressed["messages"]
        if message["role"] == "assistant"
        for call in message["tool_calls"]
    ]
    assert commands == [
        "sed -n '1,200p' pkg/target.py",
        "rg 'target' pkg/target.py",
        "grep -n old pkg/target.py",
        "sed -i \"s/'old'/'new'/\" pkg/target.py",
        "pytest -q tests/test_target.py",
        "git diff -- pkg/target.py",
    ]
    assert command_trace_quality_report(compressed["messages"]).first_edit_index == 4
    assert "First exploratory thought." not in json.dumps(compressed)


def test_compression_preserves_edit_and_all_post_edit_reasoning_verbatim() -> None:
    original = _trace()
    compressed, report = compress_editfirst_trace(original, keep_pre_edit_reads=3)

    assert report["kept"] is True
    assert compressed is not None
    original_messages = original["messages"]
    edit_index = next(
        index for index, message in enumerate(original_messages)
        if message["role"] == "assistant" and "sed -i" in message["tool_calls"][0]["function"]["arguments"]
    )
    assert compressed["messages"][-6:] == original_messages[edit_index:]


def test_compression_keeps_observation_that_names_edited_file_and_balances_pairs() -> None:
    compressed, report = compress_editfirst_trace(_trace(), keep_pre_edit_reads=1)

    assert report["kept"] is True
    assert compressed is not None
    assert "pkg/target.py" in report["kept_pre_edit_observation_text"]
    messages = compressed["messages"]
    for index, message in enumerate(messages[:-1]):
        if message["role"] == "assistant" and message["tool_calls"]:
            assert messages[index + 1]["role"] == "user"
            assert messages[index + 1]["content"].startswith("OBSERVATION:")


def test_edited_paths_uses_the_sed_target_not_a_later_diff_redirect() -> None:
    command = (
        "sed -i 's/old/new/' src/pkg/target.py\n"
        "git diff -- src/pkg/target.py > patch.txt\n"
        "cat patch.txt"
    )

    assert edited_paths(command) == ["src/pkg/target.py"]


def test_edited_paths_handles_quoted_multi_expression_sed() -> None:
    command = (
        "sed -i 's/old_a/new_a/; s/old_b/new_b/; s/old_c/new_c/' "
        "src/pkg/target.py"
    )

    assert edited_paths(command) == ["src/pkg/target.py"]


def test_edited_paths_handles_path_variable_writer() -> None:
    command = """python - <<'PY'
from pathlib import Path
path = Path('src/pkg/target.py')
text = path.read_text()
path.write_text(text.replace('old', 'new'))
PY"""

    assert edited_paths(command) == ["src/pkg/target.py"]
