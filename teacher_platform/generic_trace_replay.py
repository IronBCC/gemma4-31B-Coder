#!/usr/bin/env python3
"""Fail-closed execution admission for locally collected teacher patches."""

from __future__ import annotations

import hashlib
import json
import re
import shlex
from collections.abc import Callable, Mapping, Sequence
from pathlib import PurePosixPath
from typing import Any


class ReplayContractError(ValueError):
    """A teacher row lacks exact, executable admission evidence."""


RunCommand = Callable[[list[str], int, str | None], tuple[str, int]]
_UNITTEST_ID = re.compile(r"^(?P<method>[\w\[\]-]+)\s+\((?P<path>[\w.]+)\)$")
_PROTECTED_COMPONENTS = {
    "test",
    "tests",
    "testing",
    "fixture",
    "fixtures",
    "benchmark",
    "benchmarks",
}
_PROTECTED_BASENAMES = {
    "conftest.py",
    "run_tests.sh",
    "run_tests.py",
    "runtests.py",
    "eval.sh",
    "evaluation.sh",
    "noxfile.py",
    "pytest.ini",
    "pyproject.toml",
    "setup.cfg",
    "setup.py",
    "test-requirements.txt",
    "tox.ini",
}
_PROTECTED_TREE_COMMAND = r"""cd /testbed && python - <<'PY'
import hashlib
import os
import subprocess

index_entries = {}
for raw in subprocess.check_output(
    ["git", "ls-files", "--stage", "-z"]
).split(b"\0"):
    if not raw:
        continue
    header, path_raw = raw.split(b"\t", 1)
    mode, object_id, _stage = header.split(b" ", 2)
    index_entries[os.fsdecode(path_raw)] = (
        mode.decode("ascii"),
        object_id.decode("ascii"),
    )
selected = []
for path in index_entries:
    parts = [part.casefold() for part in path.split("/")]
    name = parts[-1]
    if (
        any(part in {"test", "tests", "testing", "fixture", "fixtures", "benchmark", "benchmarks"} for part in parts)
        or name.startswith("test_")
        or name.endswith(("_test.py", "_tests.py"))
        or name in {
            "conftest.py", "run_tests.sh", "run_tests.py", "runtests.py",
            "eval.sh", "evaluation.sh", "noxfile.py", "pytest.ini",
            "pyproject.toml", "setup.cfg", "setup.py",
            "test-requirements.txt", "tox.ini"
        }
    ):
        selected.append(path)
digest = hashlib.sha256()
for path in sorted(selected):
    mode, object_id = index_entries[path]
    digest.update(path.encode("utf-8", "surrogateescape"))
    digest.update(b"\0")
    digest.update(mode.encode("ascii"))
    digest.update(b"\0")
    if os.path.islink(path):
        digest.update(b"symlink\0")
        digest.update(os.fsencode(os.readlink(path)))
    elif os.path.isfile(path):
        digest.update(b"file\0")
        with open(path, "rb") as handle:
            digest.update(hashlib.sha256(handle.read()).digest())
    elif os.path.isdir(path) and mode == "160000":
        digest.update(b"gitlink\0")
        digest.update(object_id.encode("ascii"))
    elif os.path.isdir(path):
        digest.update(b"directory\0")
    else:
        digest.update(b"missing\0")
print(digest.hexdigest())
PY"""


def canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _bounded_invocations(
    prefix: str,
    values: Sequence[str],
    *,
    chunk_size: int,
    max_command_bytes: int,
) -> list[str]:
    commands: list[str] = []
    group: list[str] = []
    prefix_bytes = len(prefix.encode("utf-8"))
    command_bytes = prefix_bytes
    for value in values:
        quoted = shlex.quote(value)
        added_bytes = 1 + len(quoted.encode("utf-8"))
        if prefix_bytes + added_bytes > max_command_bytes:
            raise ReplayContractError("one test ID exceeds the command-size limit")
        if group and (
            len(group) >= chunk_size
            or command_bytes + added_bytes > max_command_bytes
        ):
            commands.append(prefix + " " + " ".join(group))
            group = []
            command_bytes = prefix_bytes
        group.append(quoted)
        command_bytes += added_bytes
    if group:
        commands.append(prefix + " " + " ".join(group))
    return commands


def _partition_test_ids(
    values: Sequence[str],
) -> tuple[list[str], list[str]]:
    if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
        raise ReplayContractError("test contract must be a sequence")
    unittest_ids: list[str] = []
    pytest_ids: list[str] = []
    for raw in values:
        if not isinstance(raw, str) or not raw.strip():
            raise ReplayContractError("test contract contains an empty ID")
        entry = raw.strip()
        match = _UNITTEST_ID.fullmatch(entry)
        if match:
            unittest_ids.append(f"{match.group('path')}.{match.group('method')}")
        elif "::" in entry and not any(char in entry for char in "\n\r\x00"):
            pytest_ids.append(entry)
        else:
            raise ReplayContractError(f"unsupported test ID: {entry[:120]}")
    if not unittest_ids and not pytest_ids:
        raise ReplayContractError("test contract has no executable IDs")
    return unittest_ids, pytest_ids


def _test_invocations_v1(
    values: Sequence[str],
    *,
    chunk_size: int = 1_000,
    max_command_bytes: int = 60_000,
) -> list[str]:
    """Reproduce schema-1 bounded commands for stored-evidence verification."""

    if (
        isinstance(chunk_size, bool)
        or not isinstance(chunk_size, int)
        or chunk_size < 1
        or isinstance(max_command_bytes, bool)
        or not isinstance(max_command_bytes, int)
        or max_command_bytes < 1
    ):
        raise ReplayContractError("test command bounds must be positive integers")
    unittest_ids, pytest_ids = _partition_test_ids(values)
    commands = [
        *_bounded_invocations(
            "python -m unittest -q",
            unittest_ids,
            chunk_size=chunk_size,
            max_command_bytes=max_command_bytes,
        ),
        *_bounded_invocations(
            "python -m pytest -x -q",
            pytest_ids,
            chunk_size=chunk_size,
            max_command_bytes=max_command_bytes,
        ),
    ]
    return commands


def test_invocations(values: Sequence[str]) -> list[str]:
    """Render one logical command per test runner for schema-2 stdin execution."""

    unittest_ids, pytest_ids = _partition_test_ids(values)
    commands: list[str] = []
    if unittest_ids:
        commands.append(
            "python -m unittest -q "
            + " ".join(shlex.quote(value) for value in unittest_ids)
        )
    if pytest_ids:
        commands.append(
            "python -m pytest -x -q "
            + " ".join(shlex.quote(value) for value in pytest_ids)
        )
    return commands


def build_task_contract(row: Mapping[str, Any]) -> dict[str, Any]:
    required_strings = ("instance_id", "image_name", "patch")
    for field in required_strings:
        if not isinstance(row.get(field), str) or not str(row[field]).strip():
            raise ReplayContractError(f"task is missing {field}")
    f2p = row.get("FAIL_TO_PASS")
    p2p = row.get("PASS_TO_PASS")
    if not isinstance(f2p, list) or not f2p:
        raise ReplayContractError("task is missing FAIL_TO_PASS")
    if not isinstance(p2p, list) or not p2p:
        raise ReplayContractError("task is missing PASS_TO_PASS")
    contract = {
        "schema_version": 3,
        "instance_id": row["instance_id"],
        "image_name": row["image_name"],
        "task_patch_role": "bug_inducing_mutation",
        "mutation_patch_sha256": sha256_bytes(row["patch"].encode("utf-8")),
        "f2p": list(f2p),
        "p2p": list(p2p),
        "f2p_commands": test_invocations(f2p),
        "p2p_commands": test_invocations(p2p),
    }
    contract["contract_sha256"] = sha256_bytes(canonical_json_bytes(contract))
    return contract


def task_contract_is_exact(contract: object) -> bool:
    if not isinstance(contract, Mapping):
        return False
    common = {
        "schema_version",
        "instance_id",
        "image_name",
        "f2p",
        "p2p",
        "f2p_commands",
        "p2p_commands",
        "contract_sha256",
    }
    schema_version = contract.get("schema_version")
    if schema_version in {1, 2}:
        required = common | {"reference_patch_sha256"}
        patch_hash_field = "reference_patch_sha256"
    elif schema_version == 3:
        required = common | {"task_patch_role", "mutation_patch_sha256"}
        patch_hash_field = "mutation_patch_sha256"
    else:
        return False
    if set(contract) != required:
        return False
    unsigned = {key: value for key, value in contract.items() if key != "contract_sha256"}
    try:
        invocation_builder = {
            1: _test_invocations_v1,
            2: test_invocations,
            3: test_invocations,
        }.get(schema_version)
        return bool(
            invocation_builder is not None
            and (
                schema_version != 3
                or contract.get("task_patch_role") == "bug_inducing_mutation"
            )
            and isinstance(contract.get("instance_id"), str)
            and bool(contract["instance_id"])
            and isinstance(contract.get("image_name"), str)
            and bool(contract["image_name"])
            and re.fullmatch(
                r"[0-9a-f]{64}", str(contract.get(patch_hash_field))
            )
            and contract.get("f2p_commands")
            == invocation_builder(contract.get("f2p"))
            and contract.get("p2p_commands")
            == invocation_builder(contract.get("p2p"))
            and contract.get("contract_sha256")
            == sha256_bytes(canonical_json_bytes(unsigned))
        )
    except (ReplayContractError, TypeError):
        return False


def _normalized_patch_path(raw: str) -> str | None:
    if raw in {"/dev/null", "dev/null", "a/dev/null", "b/dev/null"}:
        return None
    if raw.startswith(("a/", "b/")):
        raw = raw[2:]
    path = PurePosixPath(raw)
    if (
        not raw
        or path.is_absolute()
        or ".." in path.parts
        or "." in path.parts
        or "\x00" in raw
    ):
        raise ReplayContractError("candidate patch contains an unsafe path")
    return str(path)


def parse_patch_paths(patch: str) -> tuple[str, ...]:
    if not isinstance(patch, str) or not patch.strip():
        raise ReplayContractError("candidate patch is empty")
    paths: set[str] = set()
    for line in patch.splitlines():
        if not line.startswith("diff --git "):
            continue
        try:
            pieces = shlex.split(line)
        except ValueError as exc:
            raise ReplayContractError("candidate patch has a malformed header") from exc
        if len(pieces) != 4 or pieces[:2] != ["diff", "--git"]:
            raise ReplayContractError("candidate patch has a malformed header")
        old = _normalized_patch_path(pieces[2])
        new = _normalized_patch_path(pieces[3])
        if old is not None and new is not None and old != new:
            raise ReplayContractError("candidate patch renames a path")
        selected = new if new is not None else old
        if selected is None:
            raise ReplayContractError("candidate patch header has no path")
        paths.add(selected)
    if not paths:
        raise ReplayContractError("candidate patch has no git diff headers")
    return tuple(sorted(paths))


def protected_paths(paths: Sequence[str]) -> tuple[str, ...]:
    protected: list[str] = []
    for raw in paths:
        path = PurePosixPath(raw)
        components = tuple(part.casefold() for part in path.parts)
        basename = path.name.casefold()
        if (
            any(component in _PROTECTED_COMPONENTS for component in components)
            or basename in _PROTECTED_BASENAMES
            or basename.startswith("test_")
            or basename.endswith(("_test.py", "_tests.py"))
        ):
            protected.append(raw)
    return tuple(protected)


def assess_controls(
    controls: Mapping[str, Mapping[str, Any]],
    *,
    protected_patch_paths: Sequence[str],
) -> dict[str, Any]:
    required = ("baseline", "reference", "candidate_1", "candidate_2")
    if any(name not in controls for name in required):
        raise ReplayContractError("control set is incomplete")
    baseline = controls["baseline"]
    reference = controls["reference"]
    candidates = (controls["candidate_1"], controls["candidate_2"])
    baseline_failed = (
        baseline.get("f2p_pass") is False
        and baseline.get("p2p_pass") is True
        and baseline.get("protected_stable") is True
    )
    reference_passed = (
        reference.get("patch_applied") is True
        and reference.get("f2p_pass") is True
        and reference.get("p2p_pass") is True
        and reference.get("protected_stable") is True
    )
    candidate_passed_twice = all(
        candidate.get("patch_applied") is True
        and candidate.get("f2p_pass") is True
        and candidate.get("p2p_pass") is True
        and candidate.get("protected_stable") is True
        for candidate in candidates
    )
    protected_stable = all(
        control.get("protected_stable") is True for control in controls.values()
    )
    reasons: list[str] = []
    if not baseline_failed:
        reasons.append("baseline_did_not_fail")
    if not reference_passed:
        reasons.append("reference_did_not_pass")
    if not candidate_passed_twice:
        reasons.append("candidate_did_not_pass_twice")
    if protected_patch_paths:
        reasons.append("protected_test_or_harness_mutation")
    if not protected_stable:
        reasons.append("protected_hash_drift")
    return {
        "baseline_failed": baseline_failed,
        "reference_passed": reference_passed,
        "candidate_passed_twice": candidate_passed_twice,
        "protected_stable": protected_stable,
        "training_admitted": not reasons,
        "rejection_reasons": reasons,
        "controls_sha256": sha256_bytes(canonical_json_bytes(controls)),
    }


_ANSI_CSI_RE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")


def _strip_ansi_csi(value: str) -> str:
    return _ANSI_CSI_RE.sub("", value)


def strict_test_run_passed(command: str, output: str, returncode: int) -> bool:
    """Require explicit passed-test evidence; rc=0 with skips is not success."""

    if returncode != 0:
        return False
    tokens = shlex.split(command)
    output = _strip_ansi_csi(output)
    lowered = output.casefold()
    if tokens[:4] == ["python", "-m", "unittest", "-q"]:
        expected = len(tokens[4:])
        matches = re.findall(r"\bRan\s+(\d+)\s+tests?\b", output)
        return bool(
            expected
            and matches
            and int(matches[-1]) >= expected
            and re.search(r"(?m)^OK(?:\s|$)", output)
            and not re.search(
                r"\b(?:skipped|expected failures|unexpected successes)\s*=",
                lowered,
            )
        )
    if tokens[:3] == ["python", "-m", "pytest"]:
        try:
            arguments = tokens[tokens.index("-q") + 1 :]
        except ValueError:
            return False
        expected = len(arguments)
        passed = [int(value) for value in re.findall(r"\b(\d+)\s+passed\b", lowered)]
        nonpasses = re.search(
            r"\b\d+\s+(?:failed|error|errors|skipped|xfailed|xpassed|deselected)\b",
            lowered,
        )
        return bool(expected and passed and passed[-1] >= expected and not nonpasses)
    return False


def strict_test_run_failed(command: str, output: str, returncode: int) -> bool:
    """Accept only an assertion/test failure, never collection or infrastructure errors."""

    if returncode != 1:
        return False
    tokens = shlex.split(command)
    output = _strip_ansi_csi(output)
    lowered = output.casefold()
    if tokens[:4] == ["python", "-m", "unittest", "-q"]:
        return bool(
            tokens[4:]
            and re.search(r"\bRan\s+\d+\s+tests?\b", output)
            and re.search(r"\bFAILED\s*\([^)]*\bfailures\s*=\s*[1-9]\d*", output)
            and not re.search(
                r"\b(?:errors|skipped|unexpected successes)\s*=\s*[1-9]\d*",
                lowered,
            )
        )
    if tokens[:5] == ["python", "-m", "pytest", "-x", "-q"]:
        return bool(
            tokens[5:]
            and re.search(r"\b[1-9]\d*\s+failed\b", lowered)
            and not re.search(
                r"\b[1-9]\d*\s+(?:error|errors|skipped|xfailed|xpassed|deselected)\b",
                lowered,
            )
            and "no tests ran" not in lowered
        )
    return False


def _suite_evidence_is_exact(
    runs: object,
    commands: object,
) -> bool:
    if (
        not isinstance(runs, list)
        or not isinstance(commands, list)
        or len(runs) != len(commands)
    ):
        return False
    for run, expected_command in zip(runs, commands, strict=True):
        if (
            not isinstance(run, Mapping)
            or run.get("command") != expected_command
            or type(run.get("returncode")) is not int
            or type(run.get("strict_pass")) is not bool
            or not re.fullmatch(r"[0-9a-f]{64}", str(run.get("output_sha256")))
            or not isinstance(run.get("output_tail"), str)
            or run["strict_pass"]
            is not strict_test_run_passed(
                expected_command,
                run["output_tail"],
                run["returncode"],
            )
        ):
            return False
    return bool(runs)


def _phase_evidence_is_exact(
    name: str,
    phase: object,
    contract: Mapping[str, Any],
    *,
    candidate_patch_sha256: object = None,
) -> bool:
    if not isinstance(phase, Mapping) or phase.get("phase") != name:
        return False
    before = phase.get("protected_before")
    after = phase.get("protected_after")
    if (
        not re.fullmatch(r"[0-9a-f]{64}", str(before))
        or not re.fullmatch(r"[0-9a-f]{64}", str(after))
        or phase.get("protected_stable") is not (before == after)
        or type(phase.get("patch_applied")) is not bool
        or not _suite_evidence_is_exact(
            phase.get("f2p_runs"),
            contract.get("f2p_commands"),
        )
        or not _suite_evidence_is_exact(
            phase.get("p2p_runs"),
            contract.get("p2p_commands"),
        )
    ):
        return False
    f2p_pass = all(run["strict_pass"] is True for run in phase["f2p_runs"])
    p2p_pass = all(run["strict_pass"] is True for run in phase["p2p_runs"])
    if phase.get("f2p_pass") is not f2p_pass or phase.get("p2p_pass") is not p2p_pass:
        return False
    if contract.get("schema_version") == 3:
        expected = {
            "baseline": [
                ("mutation", contract.get("mutation_patch_sha256")),
            ],
            "reference": [],
            "candidate_1": [
                ("mutation", contract.get("mutation_patch_sha256")),
                ("candidate", candidate_patch_sha256),
            ],
            "candidate_2": [
                ("mutation", contract.get("mutation_patch_sha256")),
                ("candidate", candidate_patch_sha256),
            ],
        }[name]
        applications = phase.get("patch_applications")
        if not isinstance(applications, list) or len(applications) != len(expected):
            return False
        for application, (role, patch_sha256) in zip(
            applications,
            expected,
            strict=True,
        ):
            if (
                not isinstance(application, Mapping)
                or set(application) != {
                    "role",
                    "patch_sha256",
                    "returncode",
                    "output_sha256",
                    "output_tail",
                }
                or application.get("role") != role
                or application.get("patch_sha256") != patch_sha256
                or application.get("returncode") != 0
                or not re.fullmatch(
                    r"[0-9a-f]{64}",
                    str(application.get("output_sha256")),
                )
                or not isinstance(application.get("output_tail"), str)
            ):
                return False
        return phase["patch_applied"] is True
    patch_apply = phase.get("patch_apply")
    if name == "baseline":
        return phase["patch_applied"] is True and patch_apply is None
    return bool(
        phase["patch_applied"] is True
        and isinstance(patch_apply, Mapping)
        and patch_apply.get("returncode") == 0
        and re.fullmatch(r"[0-9a-f]{64}", str(patch_apply.get("output_sha256")))
        and isinstance(patch_apply.get("output_tail"), str)
    )


def _run_suite(
    cid: str,
    commands: Sequence[str],
    *,
    run: RunCommand,
    timeout: int,
    env_bootstrap: str,
) -> dict[str, Any]:
    results: list[dict[str, Any]] = []
    for command in commands:
        tokens = shlex.split(command)
        if tokens[:4] == ["python", "-m", "unittest", "-q"]:
            test_ids = tokens[4:]
            launcher = (
                "import json,runpy,sys\n"
                "ids=json.load(sys.stdin)\n"
                "valid=isinstance(ids,list) and bool(ids) and "
                "all(isinstance(value,str) and bool(value) for value in ids)\n"
                "valid or sys.exit(4)\n"
                "sys.argv=['unittest','-q',*ids]\n"
                "runpy.run_module('unittest',run_name='__main__')\n"
            )
        elif tokens[:5] == ["python", "-m", "pytest", "-x", "-q"]:
            test_ids = tokens[5:]
            launcher = (
                "import json,sys\n"
                "import pytest\n"
                "ids=json.load(sys.stdin)\n"
                "valid=isinstance(ids,list) and bool(ids) and "
                "all(isinstance(value,str) and bool(value) for value in ids)\n"
                "valid or sys.exit(4)\n"
                "sys.exit(pytest.main(['-x','-q',*ids]))\n"
            )
        else:
            raise ReplayContractError("unsupported logical test command")
        if not test_ids:
            raise ReplayContractError("logical test command has no IDs")
        output, returncode = run(
            [
                "docker",
                "exec",
                "-i",
                cid,
                "bash",
                "-c",
                f"cd /testbed && {env_bootstrap}python -c {shlex.quote(launcher)}",
            ],
            timeout,
            json.dumps(test_ids, ensure_ascii=False, separators=(",", ":")),
        )
        results.append({
            "command": command,
            "returncode": returncode,
            "strict_pass": strict_test_run_passed(command, output, returncode),
            "output_sha256": sha256_bytes(output.encode("utf-8", errors="replace")),
            "output_tail": output[-2000:],
        })
    return {
        "passed": bool(results) and all(row["strict_pass"] is True for row in results),
        "runs": results,
    }


def _run_phase(
    *,
    image_name: str,
    patches: Sequence[tuple[str, str]],
    phase: str,
    contract: Mapping[str, Any],
    run: RunCommand,
    timeout: int,
    env_bootstrap: str,
    run_p2p: bool = True,
) -> dict[str, Any]:
    output, returncode = run(
        ["docker", "run", "-d", "--network", "none", image_name, "sleep", "infinity"],
        180,
        None,
    )
    if returncode != 0:
        raise ReplayContractError(f"{phase} container start failed: {output[-200:]}")
    cid = output.strip().splitlines()[-1][:12]
    if not re.fullmatch(r"[0-9a-f]{12}", cid):
        raise ReplayContractError(f"{phase} container ID is invalid")
    try:
        before, before_rc = run(
            ["docker", "exec", cid, "bash", "-c", _PROTECTED_TREE_COMMAND],
            180,
            None,
        )
        if before_rc != 0 or not re.fullmatch(r"[0-9a-f]{64}", before.strip()):
            raise ReplayContractError(f"{phase} protected-tree prehash failed")
        patch_applied = True
        patch_applications: list[dict[str, Any]] = []
        for role, patch in patches:
            if role not in {"mutation", "candidate"}:
                raise ReplayContractError(f"{phase} patch role is invalid")
            if not isinstance(patch, str) or not patch.strip():
                raise ReplayContractError(f"{phase} {role} patch is empty")
            apply_output, apply_rc = run(
                [
                    "docker",
                    "exec",
                    "-i",
                    cid,
                    "bash",
                    "-c",
                    "cd /testbed && git apply --whitespace=nowarn -",
                ],
                180,
                patch,
            )
            patch_applications.append({
                "role": role,
                "patch_sha256": sha256_bytes(patch.encode("utf-8")),
                "returncode": apply_rc,
                "output_sha256": sha256_bytes(apply_output.encode("utf-8", errors="replace")),
                "output_tail": apply_output[-2000:],
            })
            if apply_rc != 0:
                patch_applied = False
                break
        if patch_applied:
            f2p = _run_suite(
                cid,
                contract["f2p_commands"],
                run=run,
                timeout=timeout,
                env_bootstrap=env_bootstrap,
            )
            p2p = (
                _run_suite(
                    cid,
                    contract["p2p_commands"],
                    run=run,
                    timeout=timeout,
                    env_bootstrap=env_bootstrap,
                )
                if run_p2p
                else {"passed": False, "runs": []}
            )
        else:
            f2p = {"passed": False, "runs": []}
            p2p = {"passed": False, "runs": []}
        after, after_rc = run(
            ["docker", "exec", cid, "bash", "-c", _PROTECTED_TREE_COMMAND],
            180,
            None,
        )
        if after_rc != 0 or not re.fullmatch(r"[0-9a-f]{64}", after.strip()):
            raise ReplayContractError(f"{phase} protected-tree posthash failed")
        return {
            "phase": phase,
            "patch_applied": patch_applied,
            "patch_applications": patch_applications,
            "f2p_pass": f2p["passed"],
            "p2p_pass": p2p["passed"],
            "f2p_runs": f2p["runs"],
            "p2p_runs": p2p["runs"],
            "protected_before": before.strip(),
            "protected_after": after.strip(),
            "protected_stable": before.strip() == after.strip(),
        }
    finally:
        run(["docker", "rm", "-f", cid], 60, None)


def verify_candidate_patch(
    row: Mapping[str, Any],
    candidate_patch: str,
    *,
    run: RunCommand,
    env_bootstrap: str,
    timeout: int = 900,
) -> dict[str, Any]:
    """Run fresh baseline/reference/candidate controls and bind their evidence."""

    contract = build_task_contract(row)
    paths = parse_patch_paths(candidate_patch)
    protected = protected_paths(paths)
    image_output, image_rc = run(
        ["docker", "image", "inspect", "--format", "{{.Id}}", contract["image_name"]],
        60,
        None,
    )
    image_id = image_output.strip()
    if image_rc != 0 or not image_id:
        raise ReplayContractError("image identity is unavailable")
    mutation_patch = str(row["patch"])
    mutation_paths = parse_patch_paths(mutation_patch)
    mutation_protected = protected_paths(mutation_paths)
    if mutation_protected:
        raise ReplayContractError("task mutation touches protected test or harness paths")

    common_evidence = {
        "admission_schema_version": 2,
        "task_contract": contract,
        "task_contract_sha256": contract["contract_sha256"],
        "image_id": image_id,
        "candidate_patch_sha256": sha256_bytes(candidate_patch.encode("utf-8")),
        "candidate_patch_paths": list(paths),
        "protected_patch_paths": list(protected),
    }

    preflight_controls: dict[str, dict[str, Any]] = {}

    def run_preflight(
        name: str,
        patches: Sequence[tuple[str, str]],
    ) -> dict[str, Any]:
        phase = _run_phase(
            image_name=contract["image_name"],
            patches=patches,
            phase=name,
            contract=contract,
            run=run,
            timeout=timeout,
            env_bootstrap=env_bootstrap,
            run_p2p=False,
        )
        preflight_controls[name] = phase
        return phase

    def reject_preflight(reason: str) -> dict[str, Any]:
        evidence = {
            **common_evidence,
            "preflight_only": True,
            "preflight_controls": preflight_controls,
            "preflight_controls_sha256": sha256_bytes(
                canonical_json_bytes(preflight_controls)
            ),
            "baseline_failed": bool(
                preflight_controls.get("baseline", {}).get("patch_applied")
                and not preflight_controls.get("baseline", {}).get("f2p_pass")
            ),
            "reference_passed": bool(
                preflight_controls.get("reference", {}).get("f2p_pass")
            ),
            "candidate_passed_twice": False,
            "protected_stable": all(
                phase.get("protected_stable") is True
                for phase in preflight_controls.values()
            ),
            "training_admitted": False,
            "rejection_reasons": [reason],
        }
        evidence["admission_evidence_sha256"] = sha256_bytes(
            canonical_json_bytes(evidence)
        )
        return evidence

    baseline_preflight = run_preflight(
        "baseline",
        (("mutation", mutation_patch),),
    )
    if (
        baseline_preflight.get("patch_applied") is not True
        or baseline_preflight.get("f2p_pass") is not False
        or baseline_preflight.get("protected_stable") is not True
    ):
        return reject_preflight("baseline_did_not_fail")
    reference_preflight = run_preflight("reference", ())
    if (
        reference_preflight.get("patch_applied") is not True
        or reference_preflight.get("f2p_pass") is not True
        or reference_preflight.get("protected_stable") is not True
    ):
        return reject_preflight("reference_did_not_pass")
    for repeat in (1, 2):
        candidate_preflight = run_preflight(
            f"candidate_{repeat}",
            (
                ("mutation", mutation_patch),
                ("candidate", candidate_patch),
            ),
        )
        if (
            candidate_preflight.get("patch_applied") is not True
            or candidate_preflight.get("f2p_pass") is not True
            or candidate_preflight.get("protected_stable") is not True
        ):
            return reject_preflight("candidate_did_not_pass_twice")

    controls = {
        "baseline": _run_phase(
            image_name=contract["image_name"],
            patches=(("mutation", mutation_patch),),
            phase="baseline",
            contract=contract,
            run=run,
            timeout=timeout,
            env_bootstrap=env_bootstrap,
        ),
        "reference": _run_phase(
            image_name=contract["image_name"],
            patches=(),
            phase="reference",
            contract=contract,
            run=run,
            timeout=timeout,
            env_bootstrap=env_bootstrap,
        ),
    }
    for repeat in (1, 2):
        controls[f"candidate_{repeat}"] = _run_phase(
            image_name=contract["image_name"],
            patches=(
                ("mutation", mutation_patch),
                ("candidate", candidate_patch),
            ),
            phase=f"candidate_{repeat}",
            contract=contract,
            run=run,
            timeout=timeout,
            env_bootstrap=env_bootstrap,
        )
    assessment = assess_controls(controls, protected_patch_paths=protected)
    evidence = {
        **common_evidence,
        "controls": controls,
        **assessment,
    }
    evidence["admission_evidence_sha256"] = sha256_bytes(canonical_json_bytes(evidence))
    return evidence


def admission_is_exact(record: Mapping[str, Any]) -> bool:
    controls = record.get("controls")
    contract = record.get("task_contract")
    patch_paths = record.get("candidate_patch_paths")
    protected_patch_paths = record.get("protected_patch_paths")
    if (
        not isinstance(controls, Mapping)
        or not task_contract_is_exact(contract)
        or not isinstance(patch_paths, list)
        or any(not isinstance(path, str) for path in patch_paths)
        or not isinstance(protected_patch_paths, list)
        or any(not isinstance(path, str) for path in protected_patch_paths)
        or list(protected_paths(patch_paths)) != protected_patch_paths
    ):
        return False
    if any(
        not _phase_evidence_is_exact(
            name,
            controls.get(name),
            contract,
            candidate_patch_sha256=record.get("candidate_patch_sha256"),
        )
        for name in ("baseline", "reference", "candidate_1", "candidate_2")
    ):
        return False
    try:
        assessment = assess_controls(
            controls,
            protected_patch_paths=protected_patch_paths,
        )
    except ReplayContractError:
        return False
    evidence = {
        key: value
        for key, value in record.items()
        if key
        in {
            "admission_schema_version",
            "task_contract",
            "task_contract_sha256",
            "image_id",
            "candidate_patch_sha256",
            "candidate_patch_paths",
            "protected_patch_paths",
            "controls",
            "baseline_failed",
            "reference_passed",
            "candidate_passed_twice",
            "protected_stable",
            "training_admitted",
            "rejection_reasons",
            "controls_sha256",
        }
    }
    expected_admission_schema = (
        2 if contract.get("schema_version") == 3 else 1
    )
    return bool(
        record.get("admission_schema_version") == expected_admission_schema
        and record.get("task_contract_sha256") == contract["contract_sha256"]
        and all(
            record.get(key) == assessment[key]
            for key in (
                "training_admitted",
                "baseline_failed",
                "reference_passed",
                "candidate_passed_twice",
                "protected_stable",
                "rejection_reasons",
                "controls_sha256",
            )
        )
        and assessment["training_admitted"] is True
        and re.fullmatch(r"[0-9a-f]{64}", str(record.get("task_contract_sha256")))
        and isinstance(record.get("image_id"), str)
        and bool(record.get("image_id"))
        and re.fullmatch(r"[0-9a-f]{64}", str(record.get("candidate_patch_sha256")))
        and isinstance(record.get("stream_sha256"), str)
        and re.fullmatch(r"[0-9a-f]{64}", str(record.get("stream_sha256")))
        and record.get("candidate_patch_sha256") == record.get("patch_sha256")
        and record.get("admission_evidence_sha256")
        == sha256_bytes(canonical_json_bytes(evidence))
    )
