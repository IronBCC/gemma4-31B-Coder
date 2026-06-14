"""Tree-sitter backend for `ast_edit` — multi-language symbol-span lookup.

Python is handled by the stdlib `ast` in `edit_tool`. For Java/Kotlin/Rust/C++/
Go/TypeScript/JavaScript this module locates a function/class/method *node*
structurally and returns its 1-based inclusive line span, so `ast_edit` can
splice a replacement that's robust to whitespace/line drift.

Grammars are optional, per-language pip packages (e.g. `tree-sitter-java`),
loaded lazily. A missing grammar yields a clear, actionable message rather than
an import error; the caller then falls back to `search_replace`.
"""
from __future__ import annotations

from dataclasses import dataclass

# ext -> (pip package, import module, optional sub-language attr for the grammar fn)
_LANGS: dict[str, tuple[str, str, str | None]] = {
    ".java": ("tree-sitter-java", "tree_sitter_java", None),
    ".kt": ("tree-sitter-kotlin", "tree_sitter_kotlin", None),
    ".kts": ("tree-sitter-kotlin", "tree_sitter_kotlin", None),
    ".rs": ("tree-sitter-rust", "tree_sitter_rust", None),
    ".go": ("tree-sitter-go", "tree_sitter_go", None),
    ".cc": ("tree-sitter-cpp", "tree_sitter_cpp", None),
    ".cpp": ("tree-sitter-cpp", "tree_sitter_cpp", None),
    ".cxx": ("tree-sitter-cpp", "tree_sitter_cpp", None),
    ".hpp": ("tree-sitter-cpp", "tree_sitter_cpp", None),
    ".h": ("tree-sitter-cpp", "tree_sitter_cpp", None),
    ".ts": ("tree-sitter-typescript", "tree_sitter_typescript", "language_typescript"),
    ".tsx": ("tree-sitter-typescript", "tree_sitter_typescript", "language_tsx"),
    ".js": ("tree-sitter-javascript", "tree_sitter_javascript", None),
    ".jsx": ("tree-sitter-javascript", "tree_sitter_javascript", None),
}

# Declaration node types that carry a name and can be replaced wholesale.
_DECL_TYPES = {
    "function_declaration", "function_definition", "function_item",
    "method_declaration", "method_definition", "constructor_declaration",
    "class_declaration", "class_specifier", "class_definition",
    "struct_item", "impl_item", "object_declaration", "type_declaration",
    "interface_declaration", "enum_declaration",
}
# Container node types whose name qualifies nested members (Class.method).
_CONTAINER_TYPES = {
    "class_declaration", "class_specifier", "class_definition", "impl_item",
    "object_declaration", "interface_declaration", "enum_declaration",
}


class GrammarUnavailable(RuntimeError):
    pass


@dataclass
class Span:
    name: str
    start_line: int  # 1-based inclusive
    end_line: int  # 1-based inclusive


def language_for(suffix: str):
    """Return a compiled tree_sitter Language for the file suffix, or raise."""
    suffix = suffix.lower()
    if suffix not in _LANGS:
        raise GrammarUnavailable(f"no tree-sitter mapping for '{suffix}'")
    pkg, mod_name, sub = _LANGS[suffix]
    try:
        from tree_sitter import Language

        mod = __import__(mod_name)
        grammar_fn = getattr(mod, sub) if sub else mod.language
        return Language(grammar_fn())
    except ImportError as e:
        raise GrammarUnavailable(
            f"grammar not installed for '{suffix}': pip install {pkg}"
        ) from e


def _node_name(node, source: bytes) -> str | None:
    """Best-effort name extraction across grammars."""
    nm = node.child_by_field_name("name")
    if nm is not None:
        return source[nm.start_byte : nm.end_byte].decode("utf-8", "ignore")
    # C++ function_definition: name lives inside the declarator subtree.
    decl = node.child_by_field_name("declarator")
    if decl is not None:
        ident = _first_identifier(decl, source)
        if ident:
            return ident
    return _first_identifier(node, source)


def _first_identifier(node, source: bytes) -> str | None:
    for child in node.children:
        if child.type in ("identifier", "field_identifier", "type_identifier"):
            return source[child.start_byte : child.end_byte].decode("utf-8", "ignore")
        nested = _first_identifier(child, source)
        if nested:
            return nested
    return None


def find_span(source: str, target_symbol: str, suffix: str) -> Span | None:
    """Locate a (possibly dotted) symbol and return its line span, or None.

    `target_symbol` may be 'name' or 'Container.name'.
    """
    lang = language_for(suffix)  # raises GrammarUnavailable
    from tree_sitter import Parser

    src_bytes = source.encode("utf-8")
    tree = Parser(lang).parse(src_bytes)

    matches: list[Span] = []

    def walk(node, prefix: str):
        for child in node.children:
            qualified = prefix
            if child.type in _DECL_TYPES:
                name = _node_name(child, src_bytes)
                if name:
                    qualified = f"{prefix}{name}"
                    if qualified == target_symbol:
                        matches.append(
                            Span(qualified, child.start_point[0] + 1, child.end_point[0] + 1)
                        )
                    if child.type in _CONTAINER_TYPES:
                        walk(child, f"{qualified}.")
                        continue
            walk(child, prefix)

    walk(tree.root_node, "")
    if not matches:
        return None
    # Prefer an exact full match; if several, take the first (outermost-first by walk order).
    return matches[0]
