"""Smoke tests for the dependency-free Phase A components."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scaffold import edit_tool, localizer, test_loop  # noqa: E402


def test_search_replace_unique(tmp_path):
    f = tmp_path / "m.py"
    f.write_text("x = 1\ny = 2\n")
    r = edit_tool.search_replace(str(f), "x = 1", "x = 42")
    assert r.ok and r.n_replacements == 1
    assert "x = 42" in f.read_text()


def test_search_replace_ambiguous_blocks(tmp_path):
    f = tmp_path / "m.py"
    f.write_text("a = 0\na = 0\n")
    r = edit_tool.search_replace(str(f), "a = 0", "a = 1")
    assert not r.ok and "ambiguous" in r.message
    assert f.read_text() == "a = 0\na = 0\n"  # unchanged


def test_ast_edit_replaces_method(tmp_path):
    f = tmp_path / "c.py"
    f.write_text(
        "class Foo:\n"
        "    def bar(self):\n"
        "        return 1\n"
        "\n"
        "    def baz(self):\n"
        "        return 2\n"
    )
    r = edit_tool.ast_edit(str(f), "Foo.bar", "def bar(self):\n    return 99")
    assert r.ok, r.message
    src = f.read_text()
    assert "return 99" in src and "return 2" in src
    import ast
    ast.parse(src)  # still valid


def test_ast_edit_rejects_bad_syntax(tmp_path):
    f = tmp_path / "c.py"
    f.write_text("def f():\n    return 1\n")
    r = edit_tool.ast_edit(str(f), "f", "def f(:\n    return 1")
    assert not r.ok
    assert f.read_text() == "def f():\n    return 1\n"  # unchanged


def test_localizer_ranks_relevant_file(tmp_path):
    (tmp_path / "pkg").mkdir()
    (tmp_path / "pkg" / "parser.py").write_text(
        "def parse_config(path):\n    return tokenize_config(path)\n"
    )
    (tmp_path / "pkg" / "unrelated.py").write_text("def render_html(x):\n    return str(x)\n")
    cands = localizer.localize(str(tmp_path), "parse_config raises on empty config file", top_k=5)
    assert cands, "expected at least one candidate"
    assert cands[0].path.endswith("parser.py")


def test_test_loop_detects_pass_and_fail(tmp_path):
    ok = test_loop.run_tests(str(tmp_path), "true", timeout_s=10)
    assert ok.passed
    bad = test_loop.run_tests(str(tmp_path), "echo 'FAILED tests/test_x.py::test_y'; false", timeout_s=10)
    assert not bad.passed
    assert "tests/test_x.py::test_y" in bad.failed_tests


def test_test_loop_timeout(tmp_path):
    tr = test_loop.run_tests(str(tmp_path), "sleep 5", timeout_s=1)
    assert tr.timed_out and not tr.passed
