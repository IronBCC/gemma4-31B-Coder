"""Multi-language verification harness (keystone for Phase-C).

Replaces the Phase-0 `returncode==0 && !failures` heuristic with an explicit
id-level gate (FAIL_TO_PASS must flip fail->pass AND PASS_TO_PASS must stay pass),
parsed from STRUCTURED tool output per language (pytest -rA/json / cargo NDJSON /
ctest JUnit). See gemma4-coder-keystone-baseline-runbook.

Layers (built bottom-up, each independently unit-testable):
  outcome.py   T0  Outcome enum + VerifyResult data contract
  langspec.py  T0  per-language build/test/timeout specs
  parsers.py   T1  structured-output parsers -> normalized ParsedTests
  gate.py      T2  decide() -> VerifyResult from parsed tests + f2p/p2p ids
"""
from __future__ import annotations

from .outcome import Outcome, VerifyResult
from .langspec import LANG_SPECS, LangSpec

__all__ = ["Outcome", "VerifyResult", "LangSpec", "LANG_SPECS"]
