"""Test-execution feedback loop — where most resolves come from (PLAN §3)."""
from dataclasses import dataclass


@dataclass
class TestRun:
    passed: bool
    failed_tests: list[str]
    stdout: str
    stderr: str


def run_tests(repo_dir: str, test_cmd: str, *, timeout_s: int = 300) -> TestRun:
    """Run tests in the task's Docker env, capture failures to feed back to the model.

    TODO: stream failing test names + tracebacks back into the agent context so the
    next turn is reason -> action -> feedback. Cap runtime (5 min cap matches RL reward).
    """
    raise NotImplementedError
