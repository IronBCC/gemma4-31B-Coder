"""Fine-grained, AST-aware editing (the documented ~2.6-pt edge over blind diffs)."""
from dataclasses import dataclass


@dataclass
class EditResult:
    ok: bool
    path: str
    message: str = ""


def search_replace(path: str, find: str, replace: str, *, count: int = 1) -> EditResult:
    """Structured, exact-match search/replace. Prefer over blind unified-diff apply.

    TODO: enforce unique-match (fail if `find` is ambiguous) so the model can't
    silently edit the wrong site — a common cause of bad patches.
    """
    raise NotImplementedError


def ast_edit(path: str, target_symbol: str, new_body: str) -> EditResult:
    """AST-aware edit: replace a function/class body by symbol, not by text offset.

    TODO: use tree-sitter (multi-lang: Python/Java/Kotlin/Rust/C++) to locate the
    node, then splice. Keeps edits robust to whitespace/line drift.
    """
    raise NotImplementedError
