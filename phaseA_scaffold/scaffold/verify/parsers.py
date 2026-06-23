"""T1 — structured-output parsers, one per language, all returning ParsedTests.

Each parser is a PURE function (no subprocess) so it is unit-testable on captured
fixtures. They never raise on malformed input: a truncated/garbage artifact yields
build_ok=False with empty id sets (the gate then treats it as BUILD_FAILED, which
is the safe direction — never a silent pass).

Test-id normalization matters: the dataset's FAIL_TO_PASS / PASS_TO_PASS ids must
match what the runner extracts. We keep the parser's ids verbatim from the tool and
leave id-set reconciliation to the gate (which intersects against the task's lists).
"""
from __future__ import annotations

import json
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field


@dataclass
class ParsedTests:
    passed_ids: set[str] = field(default_factory=set)
    failed_ids: set[str] = field(default_factory=set)
    build_ok: bool = True
    build_diag: str = ""


# ---------------------------------------------------------------- python (pytest)
# `-rA` short-test-summary lines: "PASSED path::test", "FAILED path::test - msg".
_PYTEST_SUMMARY = re.compile(r"^(PASSED|FAILED|ERROR)\s+(\S+)", re.MULTILINE)


def parse_pytest(stdout: str, stderr: str = "", artifact_path: str | None = None) -> ParsedTests:
    """Prefer the pytest-json-report artifact; fall back to the -rA summary lines."""
    pt = ParsedTests()
    if artifact_path:
        try:
            with open(artifact_path) as fh:
                data = json.load(fh)
            for t in data.get("tests", []):
                nid, ot = t.get("nodeid"), t.get("outcome")
                if not nid:
                    continue
                if ot == "passed":
                    pt.passed_ids.add(nid)
                elif ot in ("failed", "error"):
                    pt.failed_ids.add(nid)
            # a collection error => build/collection problem
            if data.get("collectors"):
                if any(c.get("outcome") == "error" for c in data["collectors"]):
                    pt.build_ok = False
                    pt.build_diag = "pytest collection error"
            return pt
        except (OSError, ValueError, KeyError):
            pass  # fall through to text
    text = f"{stdout}\n{stderr}"
    # pytest returncode 5 / no tests collected is a collection failure, not a pass
    if "no tests ran" in text.lower() or "errors during collection" in text.lower():
        pt.build_ok = False
        pt.build_diag = "pytest collected no tests / collection error"
    for m in _PYTEST_SUMMARY.finditer(text):
        kind, nid = m.group(1), m.group(2)
        if kind == "PASSED":
            pt.passed_ids.add(nid)
        else:
            pt.failed_ids.add(nid)
    return pt


# ------------------------------------------------------------------- rust (cargo)
def parse_cargo_ndjson(stdout: str, stderr: str = "", artifact_path: str | None = None) -> ParsedTests:
    """Parse `cargo ... --message-format=json` NDJSON.

    Two record kinds matter:
      {"reason":"compiler-message","message":{"level":"error",...}}  -> build error
      {"type":"test","name":"...","event":"ok"|"failed"}             -> libtest result
    libtest json is emitted by the test BINARY (not cargo's reason-tagged stream),
    so both shapes can interleave on stdout; handle either.
    """
    pt = ParsedTests()
    saw_build_error = False
    for line in stdout.splitlines():
        line = line.strip()
        if not line or not line.startswith("{"):
            continue
        try:
            obj = json.loads(line)
        except ValueError:
            continue  # non-JSON chatter; skip, don't fail
        # cargo compiler stream
        reason = obj.get("reason")
        if reason == "compiler-message":
            msg = obj.get("message", {})
            if msg.get("level") == "error":
                saw_build_error = True
                if not pt.build_diag:
                    pt.build_diag = (msg.get("rendered") or msg.get("message") or "")[:2000]
            continue
        if reason == "build-finished":
            if obj.get("success") is False:
                saw_build_error = True
            continue
        # libtest stream
        if obj.get("type") == "test":
            name, event = obj.get("name"), obj.get("event")
            if not name:
                continue
            if event == "ok":
                pt.passed_ids.add(name)
            elif event == "failed":
                pt.failed_ids.add(name)
    pt.build_ok = not saw_build_error
    return pt


# -------------------------------------------------------------------- cpp (ctest)
def parse_ctest_junit(stdout: str, stderr: str = "", artifact_path: str | None = None) -> ParsedTests:
    """Parse a ctest `--output-junit` XML file (artifact_path) into id sets.

    A testcase with a child <failure>/<error> is failed; otherwise passed. If the
    XML is missing/unparseable we treat it as build_ok=False (the build or ctest
    invocation itself broke) rather than silently reporting zero tests.
    """
    pt = ParsedTests()
    if not artifact_path:
        pt.build_ok = False
        pt.build_diag = "no ctest junit artifact produced (build/configure likely failed)"
        return pt
    try:
        tree = ET.parse(artifact_path)
    except (OSError, ET.ParseError):
        pt.build_ok = False
        pt.build_diag = "ctest junit XML missing or malformed"
        return pt
    root = tree.getroot()
    for tc in root.iter("testcase"):
        name = tc.get("name") or tc.get("classname") or ""
        if not name:
            continue
        failed = any(child.tag in ("failure", "error") for child in tc)
        # ctest also marks skips/disabled via status attr; treat non-fail as pass
        status = (tc.get("status") or "").lower()
        if failed or status in ("fail", "failed"):
            pt.failed_ids.add(name)
        else:
            pt.passed_ids.add(name)
    return pt


_DISPATCH = {"python": parse_pytest, "rust": parse_cargo_ndjson, "cpp": parse_ctest_junit}


def parse(lang: str, stdout: str, stderr: str = "", artifact_path: str | None = None) -> ParsedTests:
    try:
        fn = _DISPATCH[lang]
    except KeyError:
        raise ValueError(f"no parser for lang={lang!r}; known: {sorted(_DISPATCH)}")
    return fn(stdout, stderr, artifact_path)
