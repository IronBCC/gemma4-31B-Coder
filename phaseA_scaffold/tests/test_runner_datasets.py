"""T3 unit proofs (no docker): dataset normalization + runner via injected fake exec."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scaffold.verify import datasets, runner  # noqa: E402
from scaffold.verify.outcome import Outcome  # noqa: E402

FIX = Path(__file__).resolve().parent / "fixtures"


# ---- dataset normalization ----
def test_rust_row_normalizes_json_string_lists():
    row = {"instance_id": "bat-1", "repo": "sharkdp/bat", "base_commit": "abc",
           "FAIL_TO_PASS": '["tests::a"]', "PASS_TO_PASS": '["tests::b", "tests::c"]',
           "patch": "diff", "image_name": "img:1"}
    inst = datasets.normalize("rust", row)
    assert inst.lang == "rust" and inst.f2p == ["tests::a"] and inst.p2p == ["tests::b", "tests::c"]
    assert inst.image_name == "img:1"


def test_cpp_row_normalizes_org_repo_and_f2p_tests():
    row = {"org": "simdjson", "repo": "simdjson", "number": 99,
           "f2p_tests": ["parse_x"], "p2p_tests": ["parse_y"], "fix_patch": "diff", "build_cmd": "cmake --build build"}
    inst = datasets.normalize("cpp", row)
    assert inst.instance_id == "simdjson__simdjson-99" and inst.lang == "cpp"
    assert inst.f2p == ["parse_x"] and inst.p2p == ["parse_y"] and inst.build_cmd == "cmake --build build"


def test_unknown_lang_raises():
    try:
        datasets.normalize("go", {})
    except ValueError as e:
        assert "go" in str(e)
    else:
        raise AssertionError("expected ValueError")


# ---- runner via fake exec_fn ----
def _fake_exec(scripted):
    """scripted: list of ExecResult returned in call order."""
    calls = {"i": 0}
    def fn(container, workdir, cmd, timeout):
        r = scripted[min(calls["i"], len(scripted) - 1)]
        calls["i"] += 1
        return r
    return fn


def test_runner_rust_resolved():
    inst = datasets.normalize("rust", {"instance_id": "x", "repo": "r",
                                       "FAIL_TO_PASS": '["tests::test_highlight_regression"]',
                                       "PASS_TO_PASS": '["tests::test_decode_ok"]', "image_name": "i"})
    ndjson = (FIX / "cargo_1pass_1fail.ndjson").read_text()
    # make the f2p test pass: reuse fixture but treat the failing one as target that now passes
    ndjson_pass = ndjson.replace('"event":"failed"', '"event":"ok"')
    ef = _fake_exec([
        runner.ExecResult("", "", 0),            # build ok (cargo build)
        runner.ExecResult(ndjson_pass, "", 0),   # cargo test: both ok
    ])
    r = runner.run_verify(inst, container="c", exec_fn=ef)
    assert r.outcome is Outcome.RESOLVED


def test_runner_rust_build_failed_over_rc0():
    inst = datasets.normalize("rust", {"instance_id": "x", "repo": "r",
                                       "FAIL_TO_PASS": '["tests::a"]', "image_name": "i"})
    berr = (FIX / "cargo_build_error.ndjson").read_text()
    ef = _fake_exec([runner.ExecResult(berr, "", 0)])  # build emits compiler error even at rc 0
    r = runner.run_verify(inst, container="c", exec_fn=ef)
    assert r.outcome is Outcome.BUILD_FAILED


def test_runner_infra_flake_on_125():
    inst = datasets.normalize("rust", {"instance_id": "x", "repo": "r",
                                       "FAIL_TO_PASS": '["tests::a"]', "image_name": "i"})
    ef = _fake_exec([runner.ExecResult("", "docker: Error response", 125)])
    r = runner.run_verify(inst, container="c", exec_fn=ef)
    assert r.outcome is Outcome.INFRA_FLAKE


def test_runner_cpp_resolved_via_junit(tmp_path):
    # cpp path: build ok (rc0), then ctest writes junit we point at
    junit = tmp_path / "ct.xml"
    junit.write_text('<testsuite><testcase name="parse_x"/><testcase name="parse_y"/></testsuite>')
    inst = datasets.normalize("cpp", {"org": "s", "repo": "s", "number": 1,
                                      "f2p_tests": ["parse_x"], "p2p_tests": ["parse_y"], "build_cmd": "cmake --build build"})
    ef = _fake_exec([runner.ExecResult("", "", 0), runner.ExecResult("", "", 0)])
    r = runner.run_verify(inst, container="c", mode="turn", exec_fn=ef, junit_path=str(junit))
    assert r.outcome is Outcome.RESOLVED
