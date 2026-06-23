"""T5 — generic, language-agnostic loop-abort detector.

The baseline exposed Qwen's failure mode: burning the whole turn budget repeating
low-information actions (cd x204, ls x184, echo x197) with empty observations,
never converging to an edit. This detector kills such trajectories early so they
never enter training and never waste the turn budget at inference.

Operates on the ACTION STREAM, not on language — it keys on tool CLASS (read-only
vs mutating) and on repetition, so it works identically for Rust/C++/Python.
Pure/stateful, O(window) per turn, no I/O, no model call.
"""
from __future__ import annotations

import hashlib
import re
from collections import deque
from dataclasses import dataclass, field

# tool classes — anything not mutating/verifying is "read-only navigation"
_MUTATING = {"search_replace", "edit", "edit_file", "apply_patch", "write", "str_replace", "create"}
_VERIFYING = {"run_tests", "run_verify", "build", "cargo_test", "ctest", "pytest"}
_READONLY = {"cd", "ls", "pwd", "echo", "cat", "read_file", "localize", "grep", "find", "head", "tail"}


def canonicalize(tool_name: str, args: str) -> str:
    """Collapse volatile bits so near-identical actions hash the same.

    Strips line-ranges/timestamps/whitespace; for shell, reduces to the sequence
    of leading argv verbs (so `cd src && ls` and `cd src && ls -la` both -> 'cd;ls').
    """
    a = (args or "").strip().lower()
    if tool_name in ("bash", "shell", "run", "sh"):
        verbs = []
        for part in re.split(r"&&|\||;", a):
            tok = part.strip().split()
            if tok:
                verbs.append(tok[0])
        return "sh:" + ";".join(verbs)
    a = re.sub(r"\b\d+\b", "N", a)          # line numbers/offsets -> N
    a = re.sub(r"\s+", " ", a)
    return f"{tool_name}:{a[:120]}"


def _tool_class(tool_name: str, canon: str) -> str:
    if tool_name in _MUTATING:
        return "mutate"
    if tool_name in _VERIFYING:
        return "verify"
    if tool_name in _READONLY:
        return "read"
    # shell: classify by the verbs it ran
    if canon.startswith("sh:"):
        verbs = set(canon[3:].split(";"))
        if verbs & {"sed", "tee", "patch"} or any(">" in v for v in verbs):
            return "mutate"
        if verbs & {"cargo", "ctest", "pytest", "make", "ctest", "go"}:
            return "verify"
        if verbs <= (_READONLY | {""}):
            return "read"
    return "other"


@dataclass
class LoopAbortDetector:
    window: int = 8
    readonly_run: int = 6        # N consecutive read-only actions -> no-mutation abort
    cycle_repeats: int = 3       # same canonical action >=3x in window -> exact-cycle
    stagnant_runs: int = 3       # same test-failure signature N times -> no-fix-progress
    _actions: deque = field(default_factory=lambda: deque(maxlen=64))
    _classes: deque = field(default_factory=lambda: deque(maxlen=64))
    _sigs: deque = field(default_factory=lambda: deque(maxlen=16))

    def record(self, tool_name: str, args: str = "", feedback_signature: str | None = None) -> None:
        canon = canonicalize(tool_name, args)
        self._actions.append(canon)
        self._classes.append(_tool_class(tool_name, canon))
        if feedback_signature is not None:
            self._sigs.append(feedback_signature)

    def should_abort(self) -> tuple[bool, str]:
        acts = list(self._actions)[-self.window:]
        cls = list(self._classes)[-self.window:]

        # (1) READ_ONLY_SPAM: a run of read-only/navigation with zero mutate/verify
        recent_classes = list(self._classes)[-self.readonly_run:]
        if len(recent_classes) >= self.readonly_run and all(c == "read" for c in recent_classes):
            return True, "loop:no-mutation"

        # (2) EXACT_CYCLE: one canonical action repeated, or a short A,B,A,B cycle
        if acts:
            from collections import Counter
            top, cnt = Counter(acts).most_common(1)[0]
            if cnt >= self.cycle_repeats and not any(c in ("mutate",) for c in cls):
                return True, "loop:exact-cycle"

        # (3) STAGNANT_FEEDBACK: last K verify signatures identical (re-running tests, no real change)
        sigs = list(self._sigs)[-self.stagnant_runs:]
        if len(sigs) >= self.stagnant_runs and len(set(sigs)) == 1 and sigs[0]:
            return True, "loop:no-fix-progress"

        return False, ""


def failure_signature(build_ok: bool, failed_ids) -> str:
    """Stable hash of a verify outcome, for STAGNANT_FEEDBACK detection."""
    payload = f"{int(build_ok)}|" + ",".join(sorted(failed_ids or []))
    return hashlib.sha1(payload.encode()).hexdigest()[:16]
