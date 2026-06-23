"""T3 — the runner: drive one instance's build+test inside its docker container,
parse structured output, apply the id-level gate -> VerifyResult.

Two modes: 'turn' (incremental, fast, agent feedback) vs 'final' (clean build,
authoritative score). INFRA_FLAKE (exit 125/137, ld-killed, OOM, no-space) is
classified and surfaced for retry, never scored as a code failure.

The docker exec is injectable (`exec_fn`) so the wiring is unit-testable without a
container: a fake exec_fn returns canned (stdout, stderr, returncode).
"""
from __future__ import annotations

import re
import subprocess
from dataclasses import dataclass

from . import parsers, gate
from .outcome import Outcome, VerifyResult
from .langspec import LANG_SPECS
from .datasets import Instance

_INFRA_RC = {125, 137}
_INFRA_PAT = re.compile(r"no space left|cannot allocate memory|ld(?:: error)?: .*killed|killed signal|oom-kill", re.I)


@dataclass
class ExecResult:
    stdout: str
    stderr: str
    returncode: int


def _docker_exec(container: str, workdir: str, cmd: str, timeout_s: int) -> ExecResult:
    try:
        p = subprocess.run(
            ["docker", "exec", "-w", workdir, container, "sh", "-c", cmd],
            capture_output=True, text=True, timeout=timeout_s,
        )
        return ExecResult(p.stdout, p.stderr, p.returncode)
    except subprocess.TimeoutExpired as e:
        return ExecResult((e.stdout or "") if isinstance(e.stdout, str) else "",
                          (e.stderr or "") if isinstance(e.stderr, str) else "", 124)


def _is_infra(rc: int, stderr: str) -> bool:
    return rc in _INFRA_RC or bool(_INFRA_PAT.search(stderr or ""))


def run_verify(inst: Instance, *, container: str, workdir: str = "/testbed",
               mode: str = "turn", exec_fn=_docker_exec, junit_path: str = "/tmp/ctest.xml") -> VerifyResult:
    spec = LANG_SPECS[inst.lang]
    timeout = spec.per_turn_timeout_s

    # clean build at the authoritative final gate (defeat incremental-cache fake-pass)
    if mode == "final" and spec.clean_cmd:
        exec_fn(container, workdir, spec.clean_cmd, timeout)

    # 1) build step (separate, so BUILD_FAILED is distinguishable)
    build_ok = True
    build_diag = ""
    build_cmd = inst.build_cmd or spec.build_cmd
    if build_cmd:
        b = exec_fn(container, workdir, build_cmd, timeout)
        if _is_infra(b.returncode, b.stderr):
            return gate.decide(parsers.ParsedTests(build_ok=False), inst.f2p, inst.p2p,
                               lang=inst.lang, infra=True, returncode=b.returncode,
                               stdout=b.stdout, stderr=b.stderr)
        if inst.lang == "rust":
            pb = parsers.parse_cargo_ndjson(b.stdout, b.stderr)
            build_ok = pb.build_ok
            build_diag = pb.build_diag
        else:
            build_ok = b.returncode == 0
            build_diag = "" if build_ok else parsers._tail(b.stderr, 1500) if hasattr(parsers, "_tail") else b.stderr[-1500:]
        if not build_ok:
            pt = parsers.ParsedTests(build_ok=False, build_diag=build_diag)
            return gate.decide(pt, inst.f2p, inst.p2p, lang=inst.lang,
                               returncode=b.returncode, stdout=b.stdout, stderr=b.stderr)

    # 2) test step
    ids = " ".join(inst.f2p + inst.p2p) if inst.lang == "python" else ""
    test_cmd = inst.test_cmd or spec.test_cmd_template.format(ids=ids, junit=junit_path)
    t = exec_fn(container, workdir, test_cmd, timeout)
    if _is_infra(t.returncode, t.stderr):
        return gate.decide(parsers.ParsedTests(build_ok=True), inst.f2p, inst.p2p,
                           lang=inst.lang, infra=True, returncode=t.returncode,
                           stdout=t.stdout, stderr=t.stderr)
    timed_out = t.returncode == 124

    # 3) parse the structured test output
    artifact = junit_path if inst.lang == "cpp" else None
    parsed = parsers.parse(inst.lang, t.stdout, t.stderr, artifact)
    # build already proven ok above (or no build step) -> force build_ok True for the gate
    parsed.build_ok = parsed.build_ok and True if build_cmd else parsed.build_ok

    return gate.decide(parsed, inst.f2p, inst.p2p, lang=inst.lang,
                       timed_out=timed_out, returncode=t.returncode,
                       stdout=t.stdout, stderr=t.stderr)
