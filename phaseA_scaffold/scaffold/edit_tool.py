"""Fine-grained, AST-aware editing.

Blind unified-diff application is brittle: it silently misapplies when context
drifts. The documented ~2.6-pt edge comes from (a) exact, *unique-match*
search/replace and (b) AST-aware edits keyed on a symbol rather than a text
offset. Both are implemented here. `ast_edit` uses the stdlib for Python and
delegates other languages to a tree-sitter backend (TODO / optional dependency).
"""
from __future__ import annotations

import ast
import difflib
from dataclasses import dataclass
from pathlib import Path


@dataclass
class EditResult:
    ok: bool
    path: str
    message: str = ""
    n_replacements: int = 0
    diff: str = ""


def _read(path: str) -> str:
    return Path(path).read_text(encoding="utf-8")


def _write(path: str, text: str) -> None:
    Path(path).write_text(text, encoding="utf-8")


def _unified_diff(before: str, after: str, path: str) -> str:
    return "".join(
        difflib.unified_diff(
            before.splitlines(keepends=True),
            after.splitlines(keepends=True),
            fromfile=f"a/{path}",
            tofile=f"b/{path}",
        )
    )


def search_replace(
    path: str,
    find: str,
    replace: str,
    *,
    expected_count: int = 1,
    dry_run: bool = False,
) -> EditResult:
    """Exact-match search/replace with a uniqueness guard.

    Fails (no write) unless `find` occurs exactly `expected_count` times. This
    stops the model from silently editing the wrong site when its anchor text is
    ambiguous — a common cause of bad patches. Pass `expected_count=N` to allow
    a known number of edits.
    """
    p = Path(path)
    if not p.is_file():
        return EditResult(False, path, f"no such file: {path}")
    if not find:
        return EditResult(False, path, "empty `find` string")

    before = _read(path)
    occurrences = before.count(find)
    if occurrences == 0:
        return EditResult(False, path, "`find` text not present; re-localize the anchor")
    if occurrences != expected_count:
        return EditResult(
            False,
            path,
            f"ambiguous edit: `find` occurs {occurrences}x but expected_count={expected_count}; "
            f"widen the anchor or set expected_count",
        )

    after = before.replace(find, replace)
    diff = _unified_diff(before, after, path)
    if not dry_run:
        _write(path, after)
    return EditResult(True, path, "ok", n_replacements=occurrences, diff=diff)


# --------------------------------------------------------------------------- #
# AST-aware edits
# --------------------------------------------------------------------------- #
@dataclass
class _SymbolSpan:
    name: str
    start_line: int  # 1-based, inclusive (includes decorators)
    end_line: int  # 1-based, inclusive


def _python_symbol_span(source: str, target_symbol: str) -> _SymbolSpan | None:
    """Find a top-level or dotted (Class.method) function/class span in Python.

    `target_symbol` may be 'func', 'Class', or 'Class.method'.
    """
    tree = ast.parse(source)
    parts = target_symbol.split(".")

    def walk(nodes, prefix: str):
        for node in nodes:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                qualified = f"{prefix}{node.name}"
                if qualified == target_symbol:
                    start = node.decorator_list[0].lineno if node.decorator_list else node.lineno
                    end = node.end_lineno or node.lineno
                    return _SymbolSpan(qualified, start, end)
                if isinstance(node, ast.ClassDef) and len(parts) > 1:
                    found = walk(node.body, f"{qualified}.")
                    if found:
                        return found
        return None

    return walk(tree.body, "")


def _detect_indent(lines: list[str], start_idx: int) -> str:
    line = lines[start_idx]
    return line[: len(line) - len(line.lstrip())]


def ast_edit(path: str, target_symbol: str, new_source: str, *, dry_run: bool = False) -> EditResult:
    """Replace a whole function/class definition by symbol name.

    Robust to whitespace/line drift because it locates the node structurally,
    not by text offset. `new_source` is the full replacement definition
    (e.g. ``def foo(...):\\n    ...``); it is re-indented to match the original.
    """
    p = Path(path)
    if not p.is_file():
        return EditResult(False, path, f"no such file: {path}")

    before = _read(path)
    suffix = p.suffix.lower()

    if suffix != ".py":
        return EditResult(
            False,
            path,
            f"ast_edit currently implements Python only ({suffix} unsupported); "
            f"TODO: tree-sitter backend for java/kotlin/rust/cpp — fall back to search_replace",
        )

    try:
        span = _python_symbol_span(before, target_symbol)
    except SyntaxError as e:
        return EditResult(False, path, f"cannot parse {path}: {e}")
    if span is None:
        return EditResult(False, path, f"symbol not found: {target_symbol}")

    lines = before.splitlines(keepends=True)
    indent = _detect_indent(lines, span.start_line - 1)

    # Re-indent the replacement to the original definition's column.
    new_lines = new_source.splitlines()
    if new_lines:
        base = len(new_lines[0]) - len(new_lines[0].lstrip())
        reindented = []
        for ln in new_lines:
            stripped = ln[base:] if ln[:base].strip() == "" else ln.lstrip()
            reindented.append((indent + stripped).rstrip() if stripped.strip() else "")
        new_block = "\n".join(reindented) + "\n"
    else:
        new_block = "\n"

    after = "".join(lines[: span.start_line - 1]) + new_block + "".join(lines[span.end_line :])

    # Validate the result still parses before writing.
    try:
        ast.parse(after)
    except SyntaxError as e:
        return EditResult(False, path, f"edit would break syntax at line {e.lineno}: {e.msg}")

    diff = _unified_diff(before, after, path)
    if not dry_run:
        _write(path, after)
    return EditResult(True, path, "ok", n_replacements=1, diff=diff)


# Tool schema exposed to the model (OpenAI tool-calling format). Keep names/args
# identical between collection and serving.
TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "search_replace",
            "description": (
                "Exact-match edit. Replaces `find` with `replace` in `path`. Fails if `find` is not "
                "unique (set expected_count to allow N edits). Prefer this for small, targeted changes."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "find": {"type": "string", "description": "exact text to locate (use enough context to be unique)"},
                    "replace": {"type": "string"},
                    "expected_count": {"type": "integer", "default": 1},
                },
                "required": ["path", "find", "replace"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "ast_edit",
            "description": (
                "Replace an entire function/class by symbol name (e.g. 'Class.method'). Python only for now; "
                "robust to whitespace drift and validates syntax before writing."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "target_symbol": {"type": "string"},
                    "new_source": {"type": "string", "description": "full replacement definition"},
                },
                "required": ["path", "target_symbol", "new_source"],
            },
        },
    },
]
