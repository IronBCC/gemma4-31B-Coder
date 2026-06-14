"""Localization pass: retrieve candidate files/functions before editing (PLAN §3)."""
from dataclasses import dataclass


@dataclass
class Candidate:
    path: str
    symbol: str | None
    score: float


def localize(repo_dir: str, problem_statement: str, *, top_k: int = 10) -> list[Candidate]:
    """Rank repo locations relevant to the issue before the agent starts editing.

    TODO: hybrid retrieval — BM25/grep over symbols + embedding similarity on
    chunked files; optionally a cheap LLM rerank of the top candidates.
    """
    raise NotImplementedError
