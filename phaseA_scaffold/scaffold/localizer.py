"""Localization pass: retrieve candidate files/functions before editing (PLAN §3).

Dependency-free BM25 over source files, plus a light boost for identifiers in the
problem statement that appear as defined symbols (def/class/func/fn). This is the
cheap first pass; an optional embedding rerank or LLM rerank can sit on top.
"""
from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

CODE_EXTS = {".py", ".java", ".kt", ".rs", ".cc", ".cpp", ".cxx", ".h", ".hpp", ".go", ".ts", ".js"}
_SKIP_DIRS = {".git", "node_modules", "venv", ".venv", "__pycache__", "build", "dist", ".tox"}
_TOKEN = re.compile(r"[A-Za-z_][A-Za-z0-9_]+")
# definition sites across several languages
_DEF = re.compile(
    r"\b(?:def|class|func|fn|function)\s+([A-Za-z_][A-Za-z0-9_]*)|"
    r"\b(?:public|private|protected|static|final|\s)+\s+([A-Za-z_][A-Za-z0-9_]*)\s*\(",
)


@dataclass
class Candidate:
    path: str
    symbol: str | None
    score: float


def _tokenize(text: str) -> list[str]:
    return [t.lower() for t in _TOKEN.findall(text)]


def _iter_files(repo_dir: str):
    root = Path(repo_dir)
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in CODE_EXTS:
            if any(part in _SKIP_DIRS for part in p.parts):
                continue
            try:
                if p.stat().st_size > 1_000_000:  # skip huge/generated files
                    continue
            except OSError:
                continue
            yield p


def localize(repo_dir: str, problem_statement: str, *, top_k: int = 10) -> list[Candidate]:
    """Rank repo files relevant to the issue. Returns up to `top_k` candidates."""
    query = _tokenize(problem_statement)
    if not query:
        return []
    q_terms = set(query)
    # identifiers explicitly named in the issue get a definition-match boost
    named = {t for t in _TOKEN.findall(problem_statement) if len(t) > 2}

    docs: list[tuple[Path, list[str], set[str]]] = []  # (path, tokens, defined_symbols)
    df: Counter[str] = Counter()
    for p in _iter_files(repo_dir):
        try:
            text = p.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        toks = _tokenize(text)
        if not toks:
            continue
        defs = {m.group(1) or m.group(2) for m in _DEF.finditer(text)}
        defs.discard(None)
        docs.append((p, toks, defs))
        for term in set(toks) & q_terms:
            df[term] += 1

    if not docs:
        return []

    N = len(docs)
    avgdl = sum(len(toks) for _, toks, _ in docs) / N
    k1, b = 1.5, 0.75

    scored: list[Candidate] = []
    for p, toks, defs in docs:
        tf = Counter(toks)
        dl = len(toks)
        score = 0.0
        for term in q_terms:
            n_term = df.get(term, 0)
            if n_term == 0:
                continue
            idf = math.log(1 + (N - n_term + 0.5) / (n_term + 0.5))
            f = tf.get(term, 0)
            score += idf * (f * (k1 + 1)) / (f + k1 * (1 - b + b * dl / avgdl))
        # boost: a symbol the issue names is actually *defined* in this file
        defined_named = defs & named
        if defined_named:
            score *= 1.0 + 0.5 * len(defined_named)
        if score > 0:
            best_sym = next(iter(defined_named)) if defined_named else None
            scored.append(Candidate(str(p.relative_to(repo_dir)), best_sym, round(score, 4)))

    scored.sort(key=lambda c: c.score, reverse=True)
    return scored[:top_k]
