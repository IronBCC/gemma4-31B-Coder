import json
import tempfile
import unittest
from pathlib import Path

try:
    import pyarrow as pa
    import pyarrow.parquet as pq
except ImportError:  # pragma: no cover - optional dep
    pa = None
    pq = None

from phaseD_sft.convert_external_traces import (
    _BASH_FENCE_RE,
    convert_kwai,
    convert_openswe,
    write_jsonl,
    _extract_bash_fence,
    _extract_think_text,
    _parse_tc_args,
    _reset_call_ids,
    _translate_sre_command,
)


def _write_parquet(rows: list[dict], path: Path) -> None:
    if pa is None or pq is None:
        raise unittest.SkipTest("pyarrow not installed")
    if not rows:
        table = pa.table({"instance_id": [], "messages": []})
        pq.write_table(table, str(path))
        return

    columns = {}
    for key in rows[0]:
        columns[key] = [r.get(key) for r in rows]
    table = pa.table(columns)
    pq.write_table(table, str(path))


class BashFenceExtractionTests(unittest.TestCase):
    def setUp(self) -> None:
        _reset_call_ids()

    def test_extract_bash_fence_from_thought_block(self):
        content = "THOUGHT: Let me check the file.\n```bash\necho hello\n```"
        plain, tool_calls = _extract_bash_fence(content)

        self.assertIn("Let me check the file", plain)
        self.assertNotIn("THOUGHT:", plain)
        self.assertIsNotNone(tool_calls)
        self.assertEqual(len(tool_calls), 1)
        self.assertEqual(tool_calls[0]["function"]["name"], "bash")
        args = json.loads(tool_calls[0]["function"]["arguments"])
        self.assertEqual(args["command"], "echo hello")

    def test_extract_last_bash_fence_when_multiple(self):
        content = (
            "First attempt:\n```bash\necho old\n```\n"
            "Second attempt:\nTHOUGHT: Trying again.\n```bash\ncat file.txt\n```"
        )
        plain, tool_calls = _extract_bash_fence(content)

        self.assertIsNotNone(tool_calls)
        args = json.loads(tool_calls[0]["function"]["arguments"])
        self.assertEqual(args["command"], "cat file.txt")
        self.assertIn("First attempt", plain)

    def test_no_fence_returns_original_content(self):
        content = "THOUGHT: Just a summary with no code.\nThis is the final answer."
        plain, tool_calls = _extract_bash_fence(content)

        self.assertIsNone(tool_calls)
        self.assertIn("Just a summary", plain)


class StrReplaceEditorTranslationTests(unittest.TestCase):
    """Unit tests for the str_replace_editor translation helper functions."""

    def test_view_with_range_produces_sed_command(self):
        args = {"command": "view", "path": "/src/main.py", "view_range": [10, 20]}
        cmd, reason = _translate_sre_command(args)
        self.assertIsNone(reason)
        self.assertEqual(cmd, "sed -n '10,20p' /src/main.py")

    def test_view_without_range_produces_cat_n(self):
        args = {"command": "view", "path": "/src/util.go"}
        cmd, reason = _translate_sre_command(args)
        self.assertIsNone(reason)
        self.assertEqual(cmd, "cat -n /src/util.go")

    def test_create_produces_heredoc(self):
        args = {
            "command": "create",
            "path": "/tmp/hello.py",
            "file_text": 'print("hello world")\n',
        }
        cmd, reason = _translate_sre_command(args)
        self.assertIsNone(reason)
        self.assertTrue(cmd.startswith("cat > /tmp/hello.py <<'EOF'"))
        self.assertIn('print("hello world")', cmd)
        self.assertIn("\nEOF", cmd.rstrip())

    def test_str_replace_produces_python3_heredoc(self):
        args = {
            "command": "str_replace",
            "path": "/src/main.py",
            "old_str": 'def old_func():',
            "new_str": 'def new_func():',
        }
        cmd, reason = _translate_sre_command(args)
        self.assertIsNone(reason)
        self.assertIn("python3 - <<'PYEOF'", cmd)
        self.assertIn("pathlib.Path('/src/main.py')", cmd)
        self.assertIn("s.replace('''def old_func():''', '''def new_func():''', 1)", cmd)

    def test_str_replace_drops_on_triple_quotes_in_old_str(self):
        args = {
            "command": "str_replace",
            "path": "/src/main.py",
            "old_str": 'contains''' + "'''" + ' triple quote',
            "new_str": "replacement",
        }
        cmd, reason = _translate_sre_command(args)
        self.assertIsNone(cmd)
        self.assertEqual(reason, "triple_quote_embed")

    def test_str_replace_drops_on_triple_quotes_in_new_str(self):
        args = {
            "command": "str_replace",
            "path": "/src/main.py",
            "old_str": "original",
            "new_str": 'has''' + "'''" + ' triple quote',
        }
        cmd, reason = _translate_sre_command(args)
        self.assertIsNone(cmd)
        self.assertEqual(reason, "triple_quote_embed")

    def test_insert_produces_python3_heredoc(self):
        args = {
            "command": "insert",
            "path": "/src/main.py",
            "insert_line": 5,
            "file_text": "# new line inserted here",
        }
        cmd, reason = _translate_sre_command(args)
        self.assertIsNone(reason)
        self.assertIn("python3 - <<'PYEOF'", cmd)
        self.assertIn("lines.insert(5,", cmd)
        self.assertIn("'# new line inserted here'", cmd)

    def test_insert_drops_on_triple_quotes_in_file_text(self):
        args = {
            "command": "insert",
            "path": "/src/main.py",
            "insert_line": 3,
            "file_text": "has '''triple quotes''' inside",
        }
        cmd, reason = _translate_sre_command(args)
        self.assertIsNone(cmd)
        self.assertEqual(reason, "triple_quote_embed")

    def test_undo_edit_returns_drop_reason(self):
        args = {"command": "undo_edit", "path": "/src/main.py"}
        cmd, reason = _translate_sre_command(args)
        self.assertIsNone(cmd)
        self.assertEqual(reason, "undo_edit")


class ThinkToolExtractionTests(unittest.TestCase):
    def setUp(self) -> None:
        _reset_call_ids()

    def test_extract_think_text_from_tool_call(self):
        tc = {
            "id": "tc-think",
            "type": "function",
            "function": {
                "name": "think",
                "arguments": json.dumps({"thought": "I should verify the file first."}),
            },
        }
        text = _extract_think_text(tc)
        self.assertEqual(text, "I should verify the file first.")

    def test_extract_think_returns_none_when_empty(self):
        tc = {
            "id": "tc-think",
            "type": "function",
            "function": {"name": "think", "arguments": json.dumps({"thought": ""})},
        }
        self.assertIsNone(_extract_think_text(tc))

    def test_parse_tc_args_handles_malformed_json(self):
        tc = {
            "id": "tc-bad",
            "type": "function",
            "function": {"name": "bash", "arguments": "{not valid json"},
        }
        self.assertIsNone(_parse_tc_args(tc))


class KwaiKlearConversionTests(unittest.TestCase):
    def setUp(self) -> None:
        _reset_call_ids()

    def test_basic_conversion_with_bash_fence(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "kwai.parquet"
            rows = [
                {
                    "instance_id": "kw-001",
                    "messages": [
                        {"role": "system", "content": "You are a coding assistant."},
                        {"role": "user", "content": "Fix the bug in main.py"},
                        {
                            "role": "assistant",
                            "content": "THOUGHT: I'll fix it.\n```bash\npython -m pytest\n```",
                        },
                        {"role": "user", "content": "/tmp/output.txt:\nall tests passed"},
                    ],
                }
            ]
            _write_parquet(rows, path)

            output_rows, stats = convert_kwai(path)

        self.assertEqual(len(output_rows), 1)
        row = output_rows[0]
        self.assertEqual(row["instance_id"], "kw-001")
        self.assertEqual(row["source"], "kwai_klear_miniswe")
        msgs = row["messages"]
        self.assertEqual(msgs[0]["role"], "system")
        self.assertEqual(msgs[1]["role"], "user")
        self.assertEqual(msgs[1]["content"], "Fix the bug in main.py")
        asst = msgs[2]
        self.assertEqual(asst["role"], "assistant")
        self.assertIsNotNone(asst.get("tool_calls"))
        args_parsed = json.loads(asst["tool_calls"][0]["function"]["arguments"])
        self.assertEqual(args_parsed["command"], "python -m pytest")
        obs_msg = msgs[3]
        self.assertEqual(obs_msg["role"], "user")
        self.assertTrue(obs_msg["content"].startswith("OBSERVATION:\n"))

    def test_assistant_without_bash_fence_kept_plain(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "kwai.parquet"
            rows = [
                {
                    "instance_id": "kw-002",
                    "messages": [
                        {"role": "user", "content": "What is 2+2?"},
                        {"role": "assistant", "content": "THOUGHT: Simple math.\n4"},
                    ],
                }
            ]
            _write_parquet(rows, path)

            output_rows, stats = convert_kwai(path)

        row = output_rows[0]
        asst = row["messages"][1]
        self.assertEqual(asst["role"], "assistant")
        self.assertIsInstance(asst.get("tool_calls"), list)
        self.assertEqual(len(asst["tool_calls"]), 0)

    def test_thought_label_stripped_from_content(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "kwai.parquet"
            rows = [
                {
                    "instance_id": "kw-003",
                    "messages": [
                        {"role": "user", "content": "run this"},
                        {
                            "role": "assistant",
                            "content": (
                                "THOUGHT: I'll run it now.\n"
                                "```bash\nls -la\n```\nDone."
                            ),
                        },
                    ],
                }
            ]
            _write_parquet(rows, path)

            output_rows, stats = convert_kwai(path)

        row = output_rows[0]
        asst_content = row["messages"][1]["content"]
        self.assertNotIn("THOUGHT:", asst_content)
        self.assertIn("I'll run it now", asst_content)

    def test_schema_matches_target(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "kwai.parquet"
            rows = [
                {
                    "instance_id": "kw-010",
                    "messages": [
                        {"role": "user", "content": "task"},
                        {
                            "role": "assistant",
                            "content": "THOUGHT: ok\n```bash\necho hi\n```",
                        },
                    ],
                }
            ]
            _write_parquet(rows, path)

            output_rows, stats = convert_kwai(path)

        row = output_rows[0]
        self.assertIn("instance_id", row)
        self.assertIn("source", row)
        self.assertIn("messages", row)
        asst_msg = row["messages"][1]
        tc = asst_msg["tool_calls"][0]
        self.assertEqual(tc["type"], "function")
        self.assertIn("id", tc)
        self.assertEqual(tc["function"]["name"], "bash")
        args = json.loads(tc["function"]["arguments"])
        self.assertIn("command", args)


class OpenSWETracesConversionTests(unittest.TestCase):
    def setUp(self) -> None:
        _reset_call_ids()

    def test_execute_bash_mapped_to_bash_tool_calls(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "openswe.parquet"
            rows = [
                {
                    "instance_id": "os-001",
                    "repo": "example/repo",
                    "license": "mit",
                    "language": "python",
                    "trajectory_id": "traj-1",
                    "trajectory": [
                        {"role": "user", "content": "Fix the bug"},
                        {
                            "role": "assistant",
                            "content": "",
                            "reasoning_content": "I will run tests.",
                            "think": "",
                            "tool_calls": [
                                {
                                    "id": "tc-1",
                                    "type": "function",
                                    "function": {
                                        "name": "execute_bash",
                                        "arguments": json.dumps({"command": "pytest"}),
                                    },
                                }
                            ],
                        },
                        {"role": "tool", "content": "PASSED"},
                        {"role": "assistant", "content": "Tests passed."},
                    ],
                    "tools": ["execute_bash"],
                    "resolved": "1",
                }
            ]
            _write_parquet(rows, path)

            output_rows, stats = convert_openswe(path)

        self.assertEqual(len(output_rows), 1)
        row = output_rows[0]
        self.assertEqual(row["source"], "open_swe_traces_qwen35")
        msgs = row["messages"]
        asst = msgs[1]
        self.assertEqual(asst["role"], "assistant")
        self.assertIn("I will run tests.", asst["content"])
        self.assertIsNotNone(asst.get("tool_calls"))
        self.assertEqual(len(asst["tool_calls"]), 1)
        tc = asst["tool_calls"][0]
        self.assertEqual(tc["function"]["name"], "bash")
        args = json.loads(tc["function"]["arguments"])
        self.assertEqual(args["command"], "pytest")
        obs_msg = msgs[2]
        self.assertEqual(obs_msg["role"], "user")
        self.assertTrue(obs_msg["content"].startswith("OBSERVATION:\n"))

    def test_browser_tool_drops_trajectory_and_counts_by_name(self):
        """Truly forbidden tools (browser) drop the trajectory; counter keyed by tool name."""
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "openswe.parquet"
            rows = [
                {
                    "instance_id": "os-100",
                    "repo": "example/repo",
                    "license": "mit",
                    "language": "python",
                    "trajectory_id": "traj-browser",
                    "trajectory": [
                        {"role": "user", "content": "Browse the web"},
                        {
                            "role": "assistant",
                            "content": "",
                            "reasoning_content": "",
                            "think": "",
                            "tool_calls": [
                                {
                                    "id": "tc-browser",
                                    "type": "function",
                                    "function": {
                                        "name": "browser",
                                        "arguments": json.dumps({"url": "https://example.com"}),
                                    },
                                }
                            ],
                        },
                    ],
                    "tools": ["browser"],
                    "resolved": "1",
                }
            ]
            _write_parquet(rows, path)

            output_rows, stats = convert_openswe(path)

        self.assertEqual(len(output_rows), 0)
        self.assertEqual(stats["dropped_tool:browser"], 1)

    def test_str_replace_editor_view_translated_to_bash(self):
        """str_replace_editor view -> bash sed/cat tool_call; trajectory preserved."""
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "openswe.parquet"
            rows = [
                {
                    "instance_id": "os-010",
                    "repo": "example/repo",
                    "license": "mit",
                    "language": "python",
                    "trajectory_id": "traj-view",
                    "trajectory": [
                        {"role": "user", "content": "Inspect the file"},
                        {
                            "role": "assistant",
                            "content": "",
                            "reasoning_content": "",
                            "think": "",
                            "tool_calls": [
                                {
                                    "id": "tc-view",
                                    "type": "function",
                                    "function": {
                                        "name": "str_replace_editor",
                                        "arguments": json.dumps({
                                            "command": "view",
                                            "path": "/src/main.py",
                                            "view_range": [1, 5],
                                        }),
                                    },
                                }
                            ],
                        },
                    ],
                    "tools": ["str_replace_editor"],
                    "resolved": "1",
                }
            ]
            _write_parquet(rows, path)

            output_rows, stats = convert_openswe(path)

        self.assertEqual(len(output_rows), 1)
        asst = output_rows[0]["messages"][1]
        self.assertIsNotNone(asst.get("tool_calls"))
        tc = asst["tool_calls"][0]
        self.assertEqual(tc["function"]["name"], "bash")
        cmd_args = json.loads(tc["function"]["arguments"])
        self.assertIn("command", cmd_args)
        self.assertTrue(cmd_args["command"].startswith("sed -n '1,5p'"))
        self.assertEqual(stats["translated_str_replace_editor"], 1)

    def test_str_replace_editor_create_translated_to_heredoc(self):
        """str_replace_editor create -> bash heredoc cat > file; trajectory preserved."""
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "openswe.parquet"
            rows = [
                {
                    "instance_id": "os-011",
                    "repo": "example/repo",
                    "license": "mit",
                    "language": "python",
                    "trajectory_id": "traj-create",
                    "trajectory": [
                        {"role": "user", "content": "Create a new file"},
                        {
                            "role": "assistant",
                            "content": "",
                            "reasoning_content": "",
                            "think": "",
                            "tool_calls": [
                                {
                                    "id": "tc-create",
                                    "type": "function",
                                    "function": {
                                        "name": "str_replace_editor",
                                        "arguments": json.dumps({
                                            "command": "create",
                                            "path": "/src/new.py",
                                            "file_text": 'print("hello")',
                                        }),
                                    },
                                }
                            ],
                        },
                    ],
                    "tools": ["str_replace_editor"],
                    "resolved": "1",
                }
            ]
            _write_parquet(rows, path)

            output_rows, stats = convert_openswe(path)

        self.assertEqual(len(output_rows), 1)
        asst = output_rows[0]["messages"][1]
        tc = asst["tool_calls"][0]
        cmd_args = json.loads(tc["function"]["arguments"])
        self.assertIn("command", cmd_args)
        cmd_text = cmd_args["command"]
        self.assertTrue(cmd_text.startswith("cat > /src/new.py <<'EOF'"))
        self.assertIn('print("hello")', cmd_text)

    def test_str_replace_editor_replace_translated_to_python3_heredoc(self):
        """str_replace_editor str_replace -> python3 heredoc; trajectory preserved."""
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "openswe.parquet"
            rows = [
                {
                    "instance_id": "os-012",
                    "repo": "example/repo",
                    "license": "mit",
                    "language": "python",
                    "trajectory_id": "traj-replace",
                    "trajectory": [
                        {"role": "user", "content": "Replace a function"},
                        {
                            "role": "assistant",
                            "content": "",
                            "reasoning_content": "",
                            "think": "",
                            "tool_calls": [
                                {
                                    "id": "tc-replace",
                                    "type": "function",
                                    "function": {
                                        "name": "str_replace_editor",
                                        "arguments": json.dumps({
                                            "command": "str_replace",
                                            "path": "/src/main.py",
                                            "old_str": 'def old_func():',
                                            "new_str": 'def new_func():',
                                        }),
                                    },
                                }
                            ],
                        },
                    ],
                    "tools": ["str_replace_editor"],
                    "resolved": "1",
                }
            ]
            _write_parquet(rows, path)

            output_rows, stats = convert_openswe(path)

        self.assertEqual(len(output_rows), 1)
        asst = output_rows[0]["messages"][1]
        tc = asst["tool_calls"][0]
        cmd_args = json.loads(tc["function"]["arguments"])
        cmd_text = cmd_args["command"]
        self.assertIn("python3 - <<'PYEOF'", cmd_text)
        self.assertIn("'def old_func():'", cmd_text)
        self.assertIn("'def new_func():'", cmd_text)

    def test_str_replace_editor_insert_translated(self):
        """str_replace_editor insert -> python3 heredoc with lines.insert; trajectory preserved."""
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "openswe.parquet"
            rows = [
                {
                    "instance_id": "os-013",
                    "repo": "example/repo",
                    "license": "mit",
                    "language": "python",
                    "trajectory_id": "traj-insert",
                    "trajectory": [
                        {"role": "user", "content": "Insert a line"},
                        {
                            "role": "assistant",
                            "content": "",
                            "reasoning_content": "",
                            "think": "",
                            "tool_calls": [
                                {
                                    "id": "tc-insert",
                                    "type": "function",
                                    "function": {
                                        "name": "str_replace_editor",
                                        "arguments": json.dumps({
                                            "command": "insert",
                                            "path": "/src/main.py",
                                            "insert_line": 5,
                                            "file_text": "# inserted line",
                                        }),
                                    },
                                }
                            ],
                        },
                    ],
                    "tools": ["str_replace_editor"],
                    "resolved": "1",
                }
            ]
            _write_parquet(rows, path)

            output_rows, stats = convert_openswe(path)

        self.assertEqual(len(output_rows), 1)
        asst = output_rows[0]["messages"][1]
        tc = asst["tool_calls"][0]
        cmd_args = json.loads(tc["function"]["arguments"])
        self.assertIn("python3 - <<'PYEOF'", cmd_args["command"])
        self.assertIn("lines.insert(5,", cmd_args["command"])

    def test_undo_edit_drops_trajectory_and_counts(self):
        """str_replace_editor undo_edit -> drop trajectory; dropped_undo_edit counted."""
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "openswe.parquet"
            rows = [
                {
                    "instance_id": "os-014",
                    "repo": "example/repo",
                    "license": "mit",
                    "language": "python",
                    "trajectory_id": "traj-undo",
                    "trajectory": [
                        {"role": "user", "content": "Undo the edit"},
                        {
                            "role": "assistant",
                            "content": "",
                            "reasoning_content": "",
                            "think": "",
                            "tool_calls": [
                                {
                                    "id": "tc-undo",
                                    "type": "function",
                                    "function": {
                                        "name": "str_replace_editor",
                                        "arguments": json.dumps({
                                            "command": "undo_edit",
                                            "path": "/src/main.py",
                                        }),
                                    },
                                }
                            ],
                        },
                    ],
                    "tools": ["str_replace_editor"],
                    "resolved": "1",
                }
            ]
            _write_parquet(rows, path)

            output_rows, stats = convert_openswe(path)

        self.assertEqual(len(output_rows), 0)
        self.assertEqual(stats["dropped_undo_edit"], 1)

    def test_triple_quote_embed_drops_trajectory_and_counts(self):
        """str_replace_editor with ''' in old_str -> drop trajectory; dropped_triple_quote_embed counted."""
        triple = "'''"
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "openswe.parquet"
            rows = [
                {
                    "instance_id": "os-015",
                    "repo": "example/repo",
                    "license": "mit",
                    "language": "python",
                    "trajectory_id": "traj-triple",
                    "trajectory": [
                        {"role": "user", "content": "Replace with triple quote"},
                        {
                            "role": "assistant",
                            "content": "",
                            "reasoning_content": "",
                            "think": "",
                            "tool_calls": [
                                {
                                    "id": "tc-triple",
                                    "type": "function",
                                    "function": {
                                        "name": "str_replace_editor",
                                        "arguments": json.dumps({
                                            "command": "str_replace",
                                            "path": "/src/main.py",
                                            "old_str": f"before{triple}after",
                                            "new_str": "replacement",
                                        }),
                                    },
                                }
                            ],
                        },
                    ],
                    "tools": ["str_replace_editor"],
                    "resolved": "1",
                }
            ]
            _write_parquet(rows, path)

            output_rows, stats = convert_openswe(path)

        self.assertEqual(len(output_rows), 0)
        self.assertEqual(stats["dropped_triple_quote_embed"], 1)

    def test_think_tool_folds_into_assistant_content(self):
        """openhands 'think' tool_call -> thought prepended as plain text, no tool_call emitted."""
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "openswe.parquet"
            rows = [
                {
                    "instance_id": "os-020",
                    "repo": "example/repo",
                    "license": "mit",
                    "language": "python",
                    "trajectory_id": "traj-think",
                    "trajectory": [
                        {"role": "user", "content": "Think about this"},
                        {
                            "role": "assistant",
                            "content": "I have an idea.",
                            "reasoning_content": "",
                            "think": "",
                            "tool_calls": [
                                {
                                    "id": "tc-think",
                                    "type": "function",
                                    "function": {
                                        "name": "think",
                                        "arguments": json.dumps({
                                            "thought": "Let me analyze the problem step by step.",
                                        }),
                                    },
                                }
                            ],
                        },
                    ],
                    "tools": ["think"],
                    "resolved": "1",
                }
            ]
            _write_parquet(rows, path)

            output_rows, stats = convert_openswe(path)

        self.assertEqual(len(output_rows), 1)
        asst = output_rows[0]["messages"][1]
        # Thought should be prepended to content, separated by blank line.
        self.assertIn("Let me analyze the problem step by step.", asst["content"])
        self.assertIn("I have an idea.", asst["content"])
        # No tool_calls emitted for think-only message.
        self.assertEqual(len(asst.get("tool_calls", [])), 0)
        self.assertEqual(stats["translated_think"], 1)

    def test_think_with_other_bash_emits_both(self):
        """think + bash in same assistant turn: thought prepended, bash emitted as tool_call."""
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "openswe.parquet"
            rows = [
                {
                    "instance_id": "os-021",
                    "repo": "example/repo",
                    "license": "mit",
                    "language": "python",
                    "trajectory_id": "traj-think-bash",
                    "trajectory": [
                        {"role": "user", "content": "Think and run"},
                        {
                            "role": "assistant",
                            "content": "",
                            "reasoning_content": "",
                            "think": "",
                            "tool_calls": [
                                {
                                    "id": "tc-think2",
                                    "type": "function",
                                    "function": {
                                        "name": "think",
                                        "arguments": json.dumps({
                                            "thought": "I'll run the tests now.",
                                        }),
                                    },
                                },
                                {
                                    "id": "tc-bash2",
                                    "type": "function",
                                    "function": {
                                        "name": "bash",
                                        "arguments": json.dumps({"command": "pytest -q"}),
                                    },
                                },
                            ],
                        },
                    ],
                    "tools": ["think", "bash"],
                    "resolved": "1",
                }
            ]
            _write_parquet(rows, path)

            output_rows, stats = convert_openswe(path)

        self.assertEqual(len(output_rows), 1)
        asst = output_rows[0]["messages"][1]
        self.assertIn("I'll run the tests now.", asst["content"])
        # Bash tool_call should still be present.
        tc_names = [tc["function"]["name"] for tc in asst.get("tool_calls", [])]
        self.assertIn("bash", tc_names)

    def test_resolved_filter_drops_unresolved(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "openswe.parquet"
            rows = [
                {
                    "instance_id": "os-003",
                    "repo": "example/repo",
                    "license": "mit",
                    "language": "python",
                    "trajectory_id": "traj-unresolved",
                    "trajectory": [
                        {"role": "user", "content": "task"},
                        {"role": "assistant", "content": "ok"},
                    ],
                    "tools": [],
                    "resolved": "0",
                }
            ]
            _write_parquet(rows, path)

            output_rows, stats = convert_openswe(path)

        self.assertEqual(len(output_rows), 0)
        self.assertEqual(stats["dropped_not_resolved"], 1)

    def test_language_filter_drops_disallowed(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "openswe.parquet"
            rows = [
                {
                    "instance_id": "os-004",
                    "repo": "example/repo",
                    "license": "mit",
                    "language": "javascript",
                    "trajectory_id": "traj-js",
                    "trajectory": [
                        {"role": "user", "content": "task"},
                        {"role": "assistant", "content": "ok"},
                    ],
                    "tools": [],
                    "resolved": "1",
                }
            ]
            _write_parquet(rows, path)

            output_rows, stats = convert_openswe(path, languages=("python",))

        self.assertEqual(len(output_rows), 0)
        self.assertIn("dropped_language", stats)

    def test_finish_tool_converted_to_plain_content(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "openswe.parquet"
            rows = [
                {
                    "instance_id": "os-005",
                    "repo": "example/repo",
                    "license": "mit",
                    "language": "python",
                    "trajectory_id": "traj-finish",
                    "trajectory": [
                        {"role": "user", "content": "Fix the bug"},
                        {
                            "role": "assistant",
                            "content": "",
                            "reasoning_content": "Done.",
                            "think": "",
                            "tool_calls": [
                                {
                                    "id": "tc-finish",
                                    "type": "function",
                                    "function": {
                                        "name": "finish",
                                        "arguments": json.dumps({"message": "All fixed!"}),
                                    },
                                }
                            ],
                        },
                    ],
                    "tools": ["finish"],
                    "resolved": "1",
                }
            ]
            _write_parquet(rows, path)

            output_rows, stats = convert_openswe(path)

        self.assertEqual(len(output_rows), 1)
        asst = output_rows[0]["messages"][1]
        self.assertIsInstance(asst.get("tool_calls"), list)
        self.assertEqual(len(asst["tool_calls"]), 0)
        self.assertIn("Done.", asst["content"])
        self.assertIn("[Tool: finish]", asst["content"])
        self.assertIn("All fixed!", asst["content"])

    def test_multiple_bash_tool_calls_in_one_assistant_turn(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "openswe.parquet"
            rows = [
                {
                    "instance_id": "os-006",
                    "repo": "example/repo",
                    "license": "mit",
                    "language": "rust",
                    "trajectory_id": "traj-multi",
                    "trajectory": [
                        {"role": "user", "content": "Run the build"},
                        {
                            "role": "assistant",
                            "content": "",
                            "reasoning_content": "I'll run two commands.",
                            "think": "",
                            "tool_calls": [
                                {
                                    "id": "tc-1",
                                    "type": "function",
                                    "function": {
                                        "name": "bash",
                                        "arguments": json.dumps({"command": "cargo build"}),
                                    },
                                },
                                {
                                    "id": "tc-2",
                                    "type": "function",
                                    "function": {
                                        "name": "execute_bash",
                                        "arguments": json.dumps({"command": "cargo test"}),
                                    },
                                },
                            ],
                        },
                    ],
                    "tools": ["bash"],
                    "resolved": "1",
                }
            ]
            _write_parquet(rows, path)

            output_rows, stats = convert_openswe(path)

        self.assertEqual(len(output_rows), 1)
        asst = output_rows[0]["messages"][1]
        self.assertEqual(len(asst["tool_calls"]), 2)
        commands = [json.loads(tc["function"]["arguments"])["command"] for tc in asst["tool_calls"]]
        self.assertIn("cargo build", commands)
        self.assertIn("cargo test", commands)


class ManifestCounterTests(unittest.TestCase):
    """Verify manifest contains all required Counter-style counters."""

    def setUp(self) -> None:
        _reset_call_ids()

    def test_manifest_has_required_counters_after_conversion(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "openswe.parquet"
            # Mix of resolved/unresolved, allowed/disallowed language, and a valid row.
            rows = [
                {
                    "instance_id": "os-m1",
                    "repo": "r", "license": "mit", "language": "python",
                    "trajectory_id": "t1",
                    "trajectory": [
                        {"role": "user", "content": "task"},
                        {"role": "assistant", "content": "ok", "reasoning_content": "", "think": "", "tool_calls": []},
                    ],
                    "tools": [], "resolved": "1",
                },
                {
                    "instance_id": "os-m2",
                    "repo": "r", "license": "mit", "language": "python",
                    "trajectory_id": "t2",
                    "trajectory": [{"role": "user", "content": "task"}],
                    "tools": [], "resolved": "0",
                },
                {
                    "instance_id": "os-m3",
                    "repo": "r", "license": "mit", "language": "javascript",
                    "trajectory_id": "t3",
                    "trajectory": [{"role": "user", "content": "task"}],
                    "tools": [], "resolved": "1",
                },
            ]
            _write_parquet(rows, path)

            output_rows, stats = convert_openswe(path)

        # Required top-level counters must be present.
        required_keys = [
            "rows_read", "rows_written", "dropped_not_resolved", "dropped_language",
        ]
        for key in required_keys:
            self.assertIn(key, stats, msg=f"Missing manifest counter: {key}")

        # Numeric sanity.
        self.assertEqual(stats["rows_read"], 3)
        self.assertEqual(stats["rows_written"], 1)
        self.assertEqual(stats["dropped_not_resolved"], 1)
        self.assertEqual(stats["dropped_language"], 1)

    def test_manifest_has_per_language_written_counter(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "openswe.parquet"
            rows = [
                {
                    "instance_id": "os-pl1",
                    "repo": "r", "license": "mit", "language": "python",
                    "trajectory_id": "t1",
                    "trajectory": [
                        {"role": "user", "content": "task"},
                        {"role": "assistant", "content": "ok", "reasoning_content": "", "think": "", "tool_calls": []},
                    ],
                    "tools": [], "resolved": "1",
                },
                {
                    "instance_id": "os-pl2",
                    "repo": "r", "license": "mit", "language": "rust",
                    "trajectory_id": "t2",
                    "trajectory": [
                        {"role": "user", "content": "task"},
                        {"role": "assistant", "content": "ok", "reasoning_content": "", "think": "", "tool_calls": []},
                    ],
                    "tools": [], "resolved": "1",
                },
            ]
            _write_parquet(rows, path)

            _, stats = convert_openswe(path)

        self.assertIn("lang_written:python", stats)
        self.assertIn("lang_written:rust", stats)
        self.assertEqual(stats["lang_written:python"], 1)
        self.assertEqual(stats["lang_written:rust"], 1)

    def test_manifest_counters_for_translation_and_drops(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "openswe.parquet"
            rows = [
                # Row with str_replace_editor translate
                {
                    "instance_id": "os-mt1",
                    "repo": "r", "license": "mit", "language": "python",
                    "trajectory_id": "t1",
                    "trajectory": [
                        {"role": "user", "content": "task"},
                        {
                            "role": "assistant", "content": "", "reasoning_content": "", "think": "",
                            "tool_calls": [{
                                "id": "tc-sre", "type": "function",
                                "function": {"name": "str_replace_editor", "arguments": json.dumps({
                                    "command": "view", "path": "/x.py", "view_range": [1, 3],
                                })},
                            }],
                        },
                    ],
                    "tools": ["str_replace_editor"], "resolved": "1",
                },
                # Row with think fold
                {
                    "instance_id": "os-mt2",
                    "repo": "r", "license": "mit", "language": "python",
                    "trajectory_id": "t2",
                    "trajectory": [
                        {"role": "user", "content": "task"},
                        {
                            "role": "assistant", "content": "", "reasoning_content": "", "think": "",
                            "tool_calls": [{
                                "id": "tc-think", "type": "function",
                                "function": {"name": "think", "arguments": json.dumps({"thought": "hmm"})},
                            }],
                        },
                    ],
                    "tools": ["think"], "resolved": "1",
                },
                # Row with undo_edit -> dropped
                {
                    "instance_id": "os-mt3",
                    "repo": "r", "license": "mit", "language": "python",
                    "trajectory_id": "t3",
                    "trajectory": [
                        {"role": "user", "content": "task"},
                        {
                            "role": "assistant", "content": "", "reasoning_content": "", "think": "",
                            "tool_calls": [{
                                "id": "tc-undo", "type": "function",
                                "function": {"name": "str_replace_editor", "arguments": json.dumps({
                                    "command": "undo_edit", "path": "/x.py",
                                })},
                            }],
                        },
                    ],
                    "tools": ["str_replace_editor"], "resolved": "1",
                },
            ]
            _write_parquet(rows, path)

            output_rows, stats = convert_openswe(path)

        self.assertEqual(stats["translated_str_replace_editor"], 1)
        self.assertEqual(stats["translated_think"], 1)
        self.assertEqual(stats["dropped_undo_edit"], 1)
        # rows_written counts only the valid rows that survived.
        self.assertEqual(stats["rows_written"], 2)


class WriteJsonlTests(unittest.TestCase):
    def test_write_jsonl_creates_file_and_manifest(self):
        with tempfile.TemporaryDirectory() as td:
            out_path = Path(td) / "out.jsonl"
            manifest_path = Path(td) / "manifest.json"
            rows = [
                {
                    "instance_id": "t1",
                    "source": "kwai_klear_miniswe",
                    "messages": [{"role": "user", "content": "hi", "tool_calls": []}],
                }
            ]

            manifest = write_jsonl(rows, out_path, manifest_path=manifest_path)

            self.assertTrue(out_path.exists())
            lines = out_path.read_text(encoding="utf-8").splitlines()
            self.assertEqual(len(lines), 1)
            parsed = json.loads(lines[0])
            self.assertEqual(parsed["instance_id"], "t1")
            self.assertTrue(manifest_path.exists())
            mdata = json.loads(manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(mdata["rows_written"], 1)


if __name__ == "__main__":
    unittest.main()
