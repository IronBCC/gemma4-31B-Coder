import sys
from pathlib import Path
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scaffold import edit_tool  # noqa: E402
from scaffold import treesitter_edit as ts  # noqa: E402


def _has(suffix):
    try:
        ts.language_for(suffix)
        return True
    except ts.GrammarUnavailable:
        return False


@pytest.mark.skipif(not _has(".java"), reason="tree-sitter-java not installed")
def test_java_method_replace(tmp_path):
    f = tmp_path / "Foo.java"
    f.write_text("class Foo {\n  int bar() { return 1; }\n  int baz() { return 2; }\n}\n")
    r = edit_tool.ast_edit(str(f), "Foo.bar", "int bar() { return 99; }")
    assert r.ok, r.message
    src = f.read_text()
    assert "return 99" in src and "return 2" in src


@pytest.mark.skipif(not _has(".rs"), reason="tree-sitter-rust not installed")
def test_rust_fn_replace(tmp_path):
    f = tmp_path / "m.rs"
    f.write_text("fn add(a: i32) -> i32 {\n    a + 1\n}\n\nfn sub(a: i32) -> i32 {\n    a - 1\n}\n")
    r = edit_tool.ast_edit(str(f), "add", "fn add(a: i32) -> i32 {\n    a + 100\n}")
    assert r.ok, r.message
    assert "a + 100" in f.read_text() and "a - 1" in f.read_text()


@pytest.mark.skipif(not _has(".go"), reason="tree-sitter-go not installed")
def test_go_func_replace(tmp_path):
    f = tmp_path / "m.go"
    f.write_text("package main\n\nfunc Add(a int) int {\n\treturn a + 1\n}\n")
    r = edit_tool.ast_edit(str(f), "Add", "func Add(a int) int {\n\treturn a + 7\n}")
    assert r.ok, r.message
    assert "a + 7" in f.read_text()


def test_missing_grammar_is_graceful(tmp_path):
    f = tmp_path / "m.swift"  # no mapping -> actionable message, no crash
    f.write_text("func foo() {}\n")
    r = edit_tool.ast_edit(str(f), "foo", "func foo() {}")
    assert not r.ok and "search_replace" in r.message
