"""Test-execution feedback loop — where most resolves come from (PLAN §3).

Runs the task's test command with a hard timeout, parses failing test ids from
pytest/unittest output, and produces a compact feedback string to splice back
into the agent context (reason -> action -> feedback). The 5-minute default cap
matches the RL execution-reward time cap so collection and RL agree.
"""
from __future__ import annotations

import re
import subprocess
from dataclasses import dataclass, field


@dataclass
class TestRun:
    passed: bool
    failed_tests: list[str] = field(default_factory=list)
    returncode: int = 0
    timed_out: bool = False
    stdout: str = ""
    stderr: str = ""

    def feedback(self, max_chars: int = 4000) -> str:
        """Compact, model-facing summary to feed back into the next turn."""
        if self.timed_out:
            return f"TESTS TIMED OUT after the cap.\n{_tail(self.stdout, 1500)}"
        if self.passed:
            return "ALL TESTS PASSED."
        head = f"TESTS FAILED ({len(self.failed_tests)} failing):\n"
        names = "\n".join(f"  - {t}" for t in self.failed_tests[:25])
        tail = _tail(self.stdout + "\n" + self.stderr, max_chars - len(head) - len(names))
        return f"{head}{names}\n\n--- output tail ---\n{tail}"


def _tail(text: str, n: int) -> str:
    n = max(n, 200)
    return text[-n:] if len(text) > n else text


# pytest:  "FAILED tests/test_x.py::test_y - AssertionError"
_PYTEST_FAIL = re.compile(r"^(?:FAILED|ERROR)\s+(\S+)", re.MULTILINE)
# pytest summary line: "path::name FAILED"
_PYTEST_INLINE = re.compile(r"^(\S+::\S+)\s+(?:FAILED|ERROR)", re.MULTILINE)
# unittest: "FAIL: test_y (module.Class)"
_UNITTEST_FAIL = re.compile(r"^(?:FAIL|ERROR):\s+(\S+)", re.MULTILINE)


def parse_failures(output: str) -> list[str]:
    found: list[str] = []
    for rx in (_PYTEST_FAIL, _PYTEST_INLINE, _UNITTEST_FAIL):
        found.extend(m.group(1) for m in rx.finditer(output))
    # dedupe, preserve order
    seen: set[str] = set()
    out: list[str] = []
    for t in found:
        if t not in seen:
            seen.add(t)
            out.append(t)
    return out


def run_tests(
    repo_dir: str,
    test_cmd: str,
    *,
    timeout_s: int = 300,
    env: dict[str, str] | None = None,
    docker_container: str | None = None,
) -> TestRun:
    """Run `test_cmd` in `repo_dir` (or inside a Docker container) and capture results.

    For SWE-bench-style tasks, set `docker_container` to the task's running container
    name so tests run in the task environment; otherwise it runs locally in `repo_dir`.
    """
    if docker_container:
        cmd = ["docker", "exec", "-w", repo_dir, docker_container, "bash", "-lc", test_cmd]
        cwd = None
    else:
        cmd = ["bash", "-lc", test_cmd]
        cwd = repo_dir

    try:
        proc = subprocess.run(
            cmd,
            cwd=cwd,
            env=env,
            capture_output=True,
            text=True,
            timeout=timeout_s,
        )
    except subprocess.TimeoutExpired as e:
        return TestRun(
            passed=False,
            timed_out=True,
            returncode=124,
            stdout=(e.stdout or "") if isinstance(e.stdout, str) else "",
            stderr=(e.stderr or "") if isinstance(e.stderr, str) else "",
        )

    combined = proc.stdout + "\n" + proc.stderr
    failures = parse_failures(combined)
    passed = proc.returncode == 0 and not failures
    return TestRun(
        passed=passed,
        failed_tests=failures,
        returncode=proc.returncode,
        stdout=proc.stdout,
        stderr=proc.stderr,
    )
