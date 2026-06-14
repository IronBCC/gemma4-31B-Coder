"""Multi-turn agent loop with crash/retry recovery (the 90%-Verified run used this)."""
from .localizer import localize
from .test_loop import run_tests


def solve_task(
    repo_dir: str,
    problem_statement: str,
    test_cmd: str,
    *,
    max_turns: int = 40,
    max_retries: int = 3,
) -> dict:
    """Drive the model: localize -> propose edits -> run tests -> iterate until pass/budget.

    Returns a trajectory dict (messages, tool calls, tool results, reasoning) — the same
    schema collected for Phase-D SFT. PRESERVE reasoning_content across turns.

    TODO:
      - wire to the vLLM server with the locked Gemma-4 chat/tool template + EOS;
      - implement crash/retry recovery around tool calls and server errors;
      - enforce tool-call format (root-cause the vLLM gemma4 parser bug here).
    """
    _ = (localize, run_tests)  # used in the real loop
    raise NotImplementedError
