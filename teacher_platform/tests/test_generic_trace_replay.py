from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shlex
import subprocess
import sys

import pytest

from teacher_platform.generic_trace_replay import (
    _PROTECTED_TREE_COMMAND,
    ReplayContractError,
    _run_phase,
    _test_invocations_v1,
    _run_suite,
    admission_is_exact,
    assess_controls,
    build_task_contract,
    canonical_json_bytes,
    sha256_bytes,
    strict_test_run_failed,
    strict_test_run_passed,
    task_contract_is_exact,
    test_invocations as build_test_invocations,
    verify_candidate_patch,
)


def _phase(*, f2p: bool, p2p: bool, protected: bool = True) -> dict:
    return {
        "f2p_pass": f2p,
        "p2p_pass": p2p,
        "protected_stable": protected,
        "patch_applied": True,
    }


def _exact_run(command: str, *, passed: bool) -> dict:
    if " pytest " in command:
        output = "1 passed in 0.01s\n" if passed else "1 failed in 0.01s\n"
    else:
        output = "Ran 1 test in 0.001s\n\nOK\n" if passed else "FAILED (failures=1)\n"
    return {
        "command": command,
        "returncode": 0 if passed else 1,
        "strict_pass": passed,
        "output_sha256": sha256_bytes(output.encode()),
        "output_tail": output,
    }


def _exact_phase(
    name: str,
    contract: dict,
    *,
    f2p: bool,
    p2p: bool,
) -> dict:
    protected = "f" * 64
    phase = {
        "phase": name,
        "patch_applied": True,
        "f2p_pass": f2p,
        "p2p_pass": p2p,
        "f2p_runs": [
            _exact_run(command, passed=f2p)
            for command in contract["f2p_commands"]
        ],
        "p2p_runs": [
            _exact_run(command, passed=p2p)
            for command in contract["p2p_commands"]
        ],
        "protected_before": protected,
        "protected_after": protected,
        "protected_stable": True,
    }
    if contract["schema_version"] == 3:
        roles = {
            "baseline": [
                ("mutation", contract["mutation_patch_sha256"]),
            ],
            "reference": [],
            "candidate_1": [
                ("mutation", contract["mutation_patch_sha256"]),
                ("candidate", "a" * 64),
            ],
            "candidate_2": [
                ("mutation", contract["mutation_patch_sha256"]),
                ("candidate", "a" * 64),
            ],
        }[name]
        phase["patch_applications"] = [
            {
                "role": role,
                "patch_sha256": patch_sha256,
                "returncode": 0,
                "output_sha256": "0" * 64,
                "output_tail": "",
            }
            for role, patch_sha256 in roles
        ]
    else:
        phase["patch_apply"] = (
            None
            if name == "baseline"
            else {
                "returncode": 0,
                "output_sha256": "0" * 64,
                "output_tail": "",
            }
        )
    return phase


def test_test_invocations_covers_every_pytest_and_unittest_id() -> None:
    pytest_ids = [f"tests/test_x.py::test_{index}" for index in range(23)]
    values = [
        "test_old (pkg.tests.Case)",
        *pytest_ids,
    ]

    commands = build_test_invocations(values)

    assert commands[0] == "python -m unittest -q pkg.tests.Case.test_old"
    assert len(commands) == 2
    assert sum(command.count("tests/test_x.py::test_") for command in commands) == 23


def test_test_invocations_keeps_large_p2p_in_one_logical_command() -> None:
    values = [
        f"tests/test_long_module_{index:05d}.py::test_behavior_{'x' * 70}"
        for index in range(15_147)
    ]

    commands = build_test_invocations(values)

    assert len(commands) == 1
    assert shlex.split(commands[0])[5:] == values


def test_test_invocations_preserves_unicode_beyond_single_argument_limit() -> None:
    value = "tests/test_unicode.py::test_" + "é" * 30_000

    commands = build_test_invocations([value])

    assert shlex.split(commands[0])[5:] == [value]


@pytest.mark.parametrize(
    "command,output,returncode,expected",
    [
        (
            "python -m pytest -x -q tests/test_x.py::test_fix",
            ". [100%]\n1 passed in 0.02s\n",
            0,
            True,
        ),
        (
            "python -m pytest -x -q tests/test_x.py::test_fix",
            "\x1b[32m.\x1b[0m [100%]\n\x1b[32m\x1b[1m1 passed"
            "\x1b[0m\x1b[32m in 0.02s\x1b[0m\n",
            0,
            True,
        ),
        (
            "python -m pytest -x -q tests/test_x.py::test_fix",
            "s [100%]\n1 skipped in 0.02s\n",
            0,
            False,
        ),
        (
            "python -m unittest -q pkg.tests.Case.test_old",
            "Ran 1 test in 0.001s\n\nOK\n",
            0,
            True,
        ),
        (
            "python -m unittest -q pkg.tests.Case.test_old",
            "Ran 1 test in 0.001s\n\nOK (skipped=1)\n",
            0,
            False,
        ),
    ],
)
def test_strict_test_result_requires_passes_not_only_zero_exit(
    command: str, output: str, returncode: int, expected: bool
) -> None:
    assert strict_test_run_passed(command, output, returncode) is expected


@pytest.mark.parametrize(
    "command,output,returncode,expected",
    [
        (
            "python -m pytest -x -q tests/test_x.py::test_fix",
            "F [100%]\n1 failed in 0.02s\n",
            1,
            True,
        ),
        (
            "python -m pytest -x -q tests/test_x.py::test_fix",
            "\x1b[31mF\x1b[0m [100%]\n\x1b[31m\x1b[1m1 failed"
            "\x1b[0m\x1b[31m in 0.02s\x1b[0m\n",
            1,
            True,
        ),
        (
            "python -m pytest -x -q tests/test_x.py::test_fix",
            "ERROR tests/test_x.py\n1 error in 0.02s\n",
            1,
            False,
        ),
        (
            "python -m unittest -q pkg.tests.Case.test_old",
            "Ran 1 test\n\nFAILED (failures=1)\n",
            1,
            True,
        ),
        (
            "python -m unittest -q pkg.tests.Case.test_old",
            "Ran 1 test\n\nFAILED (errors=1)\n",
            1,
            False,
        ),
    ],
)
def test_strict_failure_requires_a_real_assertion_failure(
    command: str,
    output: str,
    returncode: int,
    expected: bool,
) -> None:
    assert strict_test_run_failed(command, output, returncode) is expected


def test_strict_suite_sends_exact_ids_on_stdin_with_constant_argv() -> None:
    calls = []
    container_id = "a" * 12
    ids = [
        "tests/test_x.py::test_fix",
        "tests/test_unicode.py::test_é[quote ' space;$(false)]",
    ]
    command = build_test_invocations(ids)[0]

    def run(argv, timeout, stdin):
        calls.append((argv, timeout, stdin))
        return "2 passed in 0.01s\n", 0

    result = _run_suite(
        container_id,
        [command],
        run=run,
        timeout=900,
        env_bootstrap="source /opt/env.sh && ",
    )

    assert result["passed"] is True
    assert len(calls) == 1
    argv, timeout, stdin = calls[0]
    assert argv[:5] == ["docker", "exec", "-i", container_id, "bash"]
    assert timeout == 900
    assert json.loads(stdin) == ids
    assert all(test_id not in "\n".join(argv) for test_id in ids)
    assert "pytest.main" in argv[-1]


def test_schema1_contract_remains_exact_after_stdin_schema2() -> None:
    ids = [f"tests/test_x.py::test_{index}" for index in range(1_001)]
    unsigned = {
        "schema_version": 1,
        "instance_id": "fixture",
        "image_name": "fixture/image:latest",
        "reference_patch_sha256": "a" * 64,
        "f2p": ids,
        "p2p": ["tests/test_x.py::test_old"],
        "f2p_commands": _test_invocations_v1(ids),
        "p2p_commands": _test_invocations_v1(["tests/test_x.py::test_old"]),
    }
    contract = {
        **unsigned,
        "contract_sha256": sha256_bytes(canonical_json_bytes(unsigned)),
    }

    assert len(contract["f2p_commands"]) == 2
    assert task_contract_is_exact(contract) is True


def test_schema3_contract_hash_changes_for_dropped_reordered_or_duplicate_id() -> None:
    row = {
        "instance_id": "fixture",
        "image_name": "fixture/image:latest",
        "patch": "reference",
        "FAIL_TO_PASS": [
            "tests/test_x.py::test_a",
            "tests/test_x.py::test_b",
        ],
        "PASS_TO_PASS": ["tests/test_x.py::test_old"],
    }
    original = build_task_contract(row)

    variants = []
    for ids in (
        row["FAIL_TO_PASS"][:1],
        list(reversed(row["FAIL_TO_PASS"])),
        [*row["FAIL_TO_PASS"], row["FAIL_TO_PASS"][0]],
    ):
        variants.append(build_task_contract({**row, "FAIL_TO_PASS": ids}))

    assert original["schema_version"] == 3
    assert all(task_contract_is_exact(contract) for contract in variants)
    assert all(
        contract["contract_sha256"] != original["contract_sha256"]
        for contract in variants
    )


@pytest.mark.parametrize(
    "output,returncode,expected",
    [
        ("15147 passed in 1.00s\n", 0, True),
        ("15146 passed in 1.00s\n", 0, False),
        ("15147 passed, 1 skipped in 1.00s\n", 0, False),
        ("15147 passed, 1 deselected in 1.00s\n", 0, False),
        ("15147 passed in 1.00s\n", 1, False),
    ],
)
def test_large_logical_suite_requires_every_test_to_pass(
    output: str,
    returncode: int,
    expected: bool,
) -> None:
    command = build_test_invocations(
        [f"tests/test_x.py::test_{index}" for index in range(15_147)]
    )[0]

    assert strict_test_run_passed(command, output, returncode) is expected


def test_phase_timeout_still_removes_the_exact_container() -> None:
    calls = []
    container_id = "a" * 12
    contract = build_task_contract({
        "instance_id": "fixture",
        "image_name": "fixture/image:latest",
        "patch": "reference",
        "FAIL_TO_PASS": ["tests/test_x.py::test_fix"],
        "PASS_TO_PASS": ["tests/test_x.py::test_old"],
    })

    def run(argv, timeout, stdin):
        calls.append((argv, timeout, stdin))
        if argv[:3] == ["docker", "run", "-d"]:
            return container_id + "\n", 0
        if argv[:3] == ["docker", "exec", container_id]:
            return "f" * 64 + "\n", 0
        if argv[:4] == ["docker", "exec", "-i", container_id]:
            raise subprocess.TimeoutExpired(argv, timeout)
        if argv[:3] == ["docker", "rm", "-f"]:
            return "", 0
        raise AssertionError(argv)

    with pytest.raises(subprocess.TimeoutExpired):
        _run_phase(
            image_name="fixture/image:latest",
            patches=(),
            phase="baseline",
            contract=contract,
            run=run,
            timeout=900,
            env_bootstrap="",
        )

    assert calls[-1][0] == ["docker", "rm", "-f", container_id]


@pytest.mark.parametrize("field", ["FAIL_TO_PASS", "PASS_TO_PASS", "patch"])
def test_build_task_contract_fails_closed_on_missing_verifier_inputs(field: str) -> None:
    row = {
        "instance_id": "demo",
        "image_name": "fixture/image:latest",
        "FAIL_TO_PASS": ["tests/test_x.py::test_fix"],
        "PASS_TO_PASS": ["tests/test_x.py::test_old"],
        "patch": "diff --git a/x.py b/x.py\n",
    }
    row[field] = [] if field != "patch" else ""

    with pytest.raises(ReplayContractError):
        build_task_contract(row)


def test_task_contract_identifies_swesmith_patch_as_bug_mutation() -> None:
    contract = build_task_contract({
        "instance_id": "demo",
        "image_name": "fixture/image:latest",
        "FAIL_TO_PASS": ["tests/test_x.py::test_fix"],
        "PASS_TO_PASS": ["tests/test_x.py::test_old"],
        "patch": "diff --git a/src/x.py b/src/x.py\n",
    })

    assert contract["schema_version"] == 3
    assert contract["task_patch_role"] == "bug_inducing_mutation"
    assert contract["mutation_patch_sha256"] == sha256_bytes(
        b"diff --git a/src/x.py b/src/x.py\n"
    )
    assert "reference_patch_sha256" not in contract


def test_verify_layers_mutation_before_candidate_and_keeps_reference_clean(
    monkeypatch,
) -> None:
    mutation = "diff --git a/src/x.py b/src/x.py\nmutation\n"
    candidate = "diff --git a/src/x.py b/src/x.py\ncandidate\n"
    task = {
        "instance_id": "demo",
        "image_name": "fixture/image:latest",
        "FAIL_TO_PASS": ["tests/test_x.py::test_fix"],
        "PASS_TO_PASS": ["tests/test_x.py::test_old"],
        "patch": mutation,
    }
    calls = {}

    def phase(**kwargs):
        name = kwargs["phase"]
        calls[name] = kwargs["patches"]
        if name == "baseline":
            return _phase(f2p=False, p2p=True)
        return _phase(f2p=True, p2p=True)

    monkeypatch.setattr(
        "teacher_platform.generic_trace_replay._run_phase",
        phase,
    )

    evidence = verify_candidate_patch(
        task,
        candidate,
        run=lambda *_args, **_kwargs: ("sha256:" + "b" * 64 + "\n", 0),
        env_bootstrap="",
    )

    assert calls == {
        "baseline": (("mutation", mutation),),
        "reference": (),
        "candidate_1": (("mutation", mutation), ("candidate", candidate)),
        "candidate_2": (("mutation", mutation), ("candidate", candidate)),
    }
    assert evidence["training_admitted"] is True
    assert evidence["admission_schema_version"] == 2


def test_verify_rejects_failed_candidate_before_any_p2p_control(
    monkeypatch,
) -> None:
    mutation = "diff --git a/src/x.py b/src/x.py\nmutation\n"
    candidate = "diff --git a/src/x.py b/src/x.py\ncandidate\n"
    task = {
        "instance_id": "demo",
        "image_name": "fixture/image:latest",
        "FAIL_TO_PASS": ["tests/test_x.py::test_fix"],
        "PASS_TO_PASS": ["tests/test_x.py::test_old"],
        "patch": mutation,
    }
    calls = []

    def phase(**kwargs):
        calls.append((kwargs["phase"], kwargs.get("run_p2p")))
        name = kwargs["phase"]
        return _phase(
            f2p=name not in {"baseline", "candidate_1"},
            p2p=False,
        )

    monkeypatch.setattr(
        "teacher_platform.generic_trace_replay._run_phase",
        phase,
    )

    evidence = verify_candidate_patch(
        task,
        candidate,
        run=lambda *_args, **_kwargs: ("sha256:" + "b" * 64 + "\n", 0),
        env_bootstrap="",
    )

    assert calls == [
        ("baseline", False),
        ("reference", False),
        ("candidate_1", False),
    ]
    assert evidence["training_admitted"] is False
    assert evidence["rejection_reasons"] == ["candidate_did_not_pass_twice"]
    assert evidence["preflight_only"] is True
    assert "controls" not in evidence


def test_assess_controls_requires_clean_baseline_reference_and_repeatable_candidate() -> None:
    controls = {
        "baseline": _phase(f2p=False, p2p=True),
        "reference": _phase(f2p=True, p2p=True),
        "candidate_1": _phase(f2p=True, p2p=True),
        "candidate_2": _phase(f2p=True, p2p=True),
    }

    admitted = assess_controls(controls, protected_patch_paths=())

    assert admitted["training_admitted"] is True
    assert admitted["baseline_failed"] is True
    assert admitted["reference_passed"] is True
    assert admitted["candidate_passed_twice"] is True
    assert admitted["rejection_reasons"] == []
    assert len(admitted["controls_sha256"]) == hashlib.sha256().digest_size * 2

    controls["baseline"] = _phase(f2p=True, p2p=True)
    assert assess_controls(controls, protected_patch_paths=())[
        "rejection_reasons"
    ] == ["baseline_did_not_fail"]

    controls["baseline"] = _phase(f2p=False, p2p=True)
    controls["candidate_2"] = _phase(f2p=True, p2p=False)
    assert assess_controls(controls, protected_patch_paths=())[
        "rejection_reasons"
    ] == ["candidate_did_not_pass_twice"]


def test_assess_controls_rejects_protected_patch_or_hash_drift() -> None:
    controls = {
        "baseline": _phase(f2p=False, p2p=True),
        "reference": _phase(f2p=True, p2p=True),
        "candidate_1": _phase(f2p=True, p2p=True, protected=False),
        "candidate_2": _phase(f2p=True, p2p=True),
    }

    result = assess_controls(
        controls,
        protected_patch_paths=("tests/test_x.py",),
    )

    assert result["training_admitted"] is False
    assert result["rejection_reasons"] == [
        "candidate_did_not_pass_twice",
        "protected_test_or_harness_mutation",
        "protected_hash_drift",
    ]


@pytest.mark.parametrize(
    "path",
    [
        "conftest.py",
        "pytest.ini",
        "pyproject.toml",
        "setup.cfg",
        "setup.py",
        "noxfile.py",
        "tests/fixtures/input.json",
    ],
)
def test_harness_and_test_configuration_paths_are_protected(path: str) -> None:
    from teacher_platform.generic_trace_replay import protected_paths

    assert protected_paths([path]) == (path,)


def test_protected_tree_hash_handles_a_tracked_gitlink_directory(
    tmp_path: Path,
) -> None:
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    tree = subprocess.run(
        ["git", "mktree"],
        cwd=tmp_path,
        input="",
        text=True,
        capture_output=True,
        check=True,
    ).stdout.strip()
    commit = subprocess.run(
        ["git", "hash-object", "-t", "commit", "-w", "--stdin"],
        cwd=tmp_path,
        input=(
            f"tree {tree}\n"
            "author Fixture <fixture@example.invalid> 0 +0000\n"
            "committer Fixture <fixture@example.invalid> 0 +0000\n"
            "\nfixture\n"
        ),
        text=True,
        capture_output=True,
        check=True,
    ).stdout.strip()
    subprocess.run(
        [
            "git",
            "update-index",
            "--add",
            "--cacheinfo",
            f"160000,{commit},tests/latest",
        ],
        cwd=tmp_path,
        check=True,
    )
    (tmp_path / "tests" / "latest").mkdir(parents=True)
    command = _PROTECTED_TREE_COMMAND.replace(
        "cd /testbed",
        f"cd {shlex.quote(str(tmp_path))}",
        1,
    ).replace(
        "&& python -",
        f"&& {shlex.quote(sys.executable)} -",
        1,
    )

    result = subprocess.run(
        ["bash", "-c", command],
        text=True,
        capture_output=True,
    )

    assert result.returncode == 0, result.stderr
    assert len(result.stdout.strip()) == 64


def test_exact_admission_binds_evidence_stream_and_patch_identities() -> None:
    patch_sha = "a" * 64
    contract = build_task_contract({
        "instance_id": "demo",
        "image_name": "fixture/image:latest",
        "patch": "reference",
        "FAIL_TO_PASS": ["tests/test_x.py::test_fix"],
        "PASS_TO_PASS": ["tests/test_x.py::test_old"],
    })
    controls = {
        "baseline": _exact_phase("baseline", contract, f2p=False, p2p=True),
        "reference": _exact_phase("reference", contract, f2p=True, p2p=True),
        "candidate_1": _exact_phase("candidate_1", contract, f2p=True, p2p=True),
        "candidate_2": _exact_phase("candidate_2", contract, f2p=True, p2p=True),
    }
    assessment = assess_controls(controls, protected_patch_paths=())
    evidence = {
        "admission_schema_version": 2,
        "task_contract": contract,
        "task_contract_sha256": contract["contract_sha256"],
        "image_id": "sha256:" + "c" * 64,
        "candidate_patch_sha256": patch_sha,
        "candidate_patch_paths": ["src/x.py"],
        "protected_patch_paths": [],
        "controls": controls,
        **assessment,
    }
    record = {
        **evidence,
        "admission_evidence_sha256": sha256_bytes(canonical_json_bytes(evidence)),
        "stream_sha256": "e" * 64,
        "patch_sha256": patch_sha,
    }

    assert admission_is_exact(record) is True

    record["stream_sha256"] = "tampered"
    assert admission_is_exact(record) is False
