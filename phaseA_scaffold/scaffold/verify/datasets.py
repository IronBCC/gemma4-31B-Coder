"""T3 — dataset loaders normalizing differing SWE schemas into one record.

The Rust set (r1v3r/multi_SWE_Bench_Rust) uses native SWE-bench field names
(FAIL_TO_PASS/PASS_TO_PASS as JSON-string lists). The C++ set (ByteDance
Multi-SWE-bench) uses f2p_tests/p2p_tests + org/repo/number. Both normalize to a
common Instance the runner consumes, so the agent/verify loop is schema-agnostic.

Pure parsing (no network/docker) so it is unit-testable on dict fixtures.
"""
from __future__ import annotations

import ast
import json
from dataclasses import dataclass, field


@dataclass
class Instance:
    instance_id: str
    lang: str
    repo: str
    f2p: list[str] = field(default_factory=list)   # FAIL_TO_PASS test ids
    p2p: list[str] = field(default_factory=list)   # PASS_TO_PASS test ids
    base_commit: str = ""
    problem_statement: str = ""
    test_patch: str = ""
    gold_patch: str = ""
    image_name: str = ""        # the prebuilt docker image; MUST be set or the runner phantoms
    build_cmd: str = ""         # per-repo override (C++ cmake vs make); "" = use LangSpec default
    test_cmd: str = ""


def _as_list(v) -> list[str]:
    """FAIL_TO_PASS may be a real list, a JSON string, or a python-repr string."""
    if isinstance(v, list):
        return [str(x) for x in v]
    if not v:
        return []
    s = str(v).strip()
    for parse in (json.loads, ast.literal_eval):
        try:
            out = parse(s)
            if isinstance(out, (list, tuple)):
                return [str(x) for x in out]
        except (ValueError, SyntaxError):
            continue
    return [s]


def from_rust_row(row: dict) -> Instance:
    """r1v3r/multi_SWE_Bench_Rust — native SWE-bench field names, all repo=sharkdp/bat etc."""
    return Instance(
        instance_id=row.get("instance_id") or f"{row.get('repo','?')}-{row.get('pull_number','?')}",
        lang="rust",
        repo=row.get("repo", ""),
        f2p=_as_list(row.get("FAIL_TO_PASS")),
        p2p=_as_list(row.get("PASS_TO_PASS")),
        base_commit=row.get("base_commit", ""),
        problem_statement=row.get("problem_statement", ""),
        test_patch=row.get("test_patch", ""),
        gold_patch=row.get("patch", ""),
        image_name=row.get("image_name", ""),   # injected by the rust_dataset_adapter
    )


def from_cpp_row(row: dict) -> Instance:
    """ByteDance Multi-SWE-bench C++ — org/repo/number + f2p_tests/p2p_tests."""
    org, repo, num = row.get("org", ""), row.get("repo", ""), row.get("number", row.get("pull_number", ""))
    iid = row.get("instance_id") or (f"{org}__{repo}-{num}" if org else f"{repo}-{num}")
    return Instance(
        instance_id=iid,
        lang="cpp",
        repo=f"{org}/{repo}" if org else repo,
        f2p=_as_list(row.get("f2p_tests") or row.get("FAIL_TO_PASS")),
        p2p=_as_list(row.get("p2p_tests") or row.get("PASS_TO_PASS")),
        base_commit=row.get("base", row.get("base_commit", "")),
        problem_statement=row.get("body") or row.get("problem_statement", ""),
        test_patch=row.get("test_patch", ""),
        gold_patch=row.get("fix_patch") or row.get("patch", ""),
        image_name=row.get("image_name", ""),
        build_cmd=row.get("build_cmd", ""),   # from cpp_build_matrix per-repo (cmake vs make)
    )


_LOADERS = {"rust": from_rust_row, "cpp": from_cpp_row}


def normalize(lang: str, row: dict) -> Instance:
    try:
        return _LOADERS[lang](row)
    except KeyError:
        raise ValueError(f"no dataset loader for lang={lang!r}; known: {sorted(_LOADERS)}")
