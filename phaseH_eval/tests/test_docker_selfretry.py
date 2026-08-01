"""Unit tests for the submission self-retry docker environment (no docker)."""
from __future__ import annotations

import hashlib
import os
import subprocess
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

_MINISWE_MODULES = (
    "minisweagent",
    "minisweagent.environments",
    "minisweagent.environments.docker",
    "minisweagent.exceptions",
)
_saved_miniswe_modules = {
    name: sys.modules.get(name)
    for name in _MINISWE_MODULES
}
_using_miniswe_stub = False
try:
    from minisweagent.exceptions import Submitted  # type: ignore[import-not-found]  # noqa: E402
    from minisweagent.environments.docker import DockerEnvironment  # type: ignore[import-not-found]  # noqa: E402,F401
except ModuleNotFoundError:
    _using_miniswe_stub = True

    class Submitted(Exception):
        pass

    class DockerEnvironment:
        def _check_finished(self, output):
            text = output.get("output", "")
            lines = text.lstrip().splitlines(keepends=True)
            if (
                lines
                and lines[0].strip() == "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT"
                and output.get("returncode") == 0
            ):
                raise Submitted("".join(lines[1:]))

    minisweagent = types.ModuleType("minisweagent")
    environments = types.ModuleType("minisweagent.environments")
    docker = types.ModuleType("minisweagent.environments.docker")
    exceptions = types.ModuleType("minisweagent.exceptions")
    docker.DockerEnvironment = DockerEnvironment
    exceptions.Submitted = Submitted
    sys.modules["minisweagent"] = minisweagent
    sys.modules["minisweagent.environments"] = environments
    sys.modules["minisweagent.environments.docker"] = docker
    sys.modules["minisweagent.exceptions"] = exceptions

from docker_selfretry import (  # noqa: E402
    DockerSelfRetryEnv,
    MARKER,
    validate_submission_patch,
)

if _using_miniswe_stub:
    for _name, _module in _saved_miniswe_modules.items():
        if _module is None:
            sys.modules.pop(_name, None)
        else:
            sys.modules[_name] = _module


def make_env(retries=2):
    env = DockerSelfRetryEnv.__new__(DockerSelfRetryEnv)
    env._submit_retries_left = retries
    env._marker_nudges_left = 1
    env._task_mutation_evidence = None
    return env


def test_top_level_harness_import_resolves_source_only_diff_command(
    tmp_path: Path,
) -> None:
    package = tmp_path / "minisweagent" / "environments"
    package.mkdir(parents=True)
    (tmp_path / "minisweagent" / "__init__.py").write_text("")
    (package / "__init__.py").write_text("")
    (package / "docker.py").write_text("class DockerEnvironment:\n    pass\n")
    environment = dict(os.environ)
    environment["PYTHONPATH"] = os.pathsep.join(
        [str(Path(__file__).resolve().parents[1]), str(tmp_path)]
    )

    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import docker_selfretry; "
                "assert docker_selfretry.SOURCE_ONLY_DIFF_COMMAND.startswith("
                "'git add -N')"
            ),
        ],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr


def test_task_baseline_applies_commits_and_binds_exact_mutation(monkeypatch):
    env = make_env()
    env.container_id = "a" * 12
    env.config = SimpleNamespace(executable="docker")
    mutation = (
        "diff --git a/src/x.py b/src/x.py\n"
        "--- a/src/x.py\n"
        "+++ b/src/x.py\n"
        "@@ -1 +1 @@\n-a\n+b\n"
    )
    calls = []

    def run(argv, **kwargs):
        calls.append((argv, kwargs))
        if "-i" in argv:
            return SimpleNamespace(stdout="", returncode=0)
        return SimpleNamespace(stdout="f" * 40 + "\n", returncode=0)

    monkeypatch.setattr("docker_selfretry.subprocess.run", run)

    evidence = env.establish_task_baseline("fixture__repo.case", mutation)

    assert calls[0][0][:5] == ["docker", "exec", "-i", "a" * 12, "bash"]
    assert calls[0][1]["input"] == mutation
    assert "git apply --whitespace" in calls[0][0][-1]
    assert "git add -A" in calls[0][0][-1]
    assert "git status --porcelain" in calls[0][0][-1]
    assert "commit --no-verify" in calls[1][0][-1]
    assert "git status --porcelain" in calls[1][0][-1]
    assert evidence == env._task_mutation_evidence
    assert evidence["instance_id"] == "fixture__repo.case"
    assert evidence["mutation_patch_sha256"] == hashlib.sha256(
        mutation.encode()
    ).hexdigest()
    assert evidence["mutation_baseline_commit"] == "f" * 40


def test_task_baseline_fails_closed_and_cleans_no_evidence(monkeypatch):
    env = make_env()
    env.container_id = "a" * 12
    env.config = SimpleNamespace(executable="docker")
    mutation = "diff --git a/src/x.py b/src/x.py\n--- a/src/x.py\n+++ b/src/x.py\n"

    monkeypatch.setattr(
        "docker_selfretry.subprocess.run",
        lambda *_args, **_kwargs: SimpleNamespace(
            stdout="patch does not apply",
            returncode=1,
        ),
    )

    with pytest.raises(RuntimeError, match="mutation apply failed"):
        env.establish_task_baseline("fixture__repo.case", mutation)

    assert env._task_mutation_evidence is None


def test_task_mutation_evidence_is_serialized_into_trajectory(monkeypatch):
    env = make_env()
    evidence = {
        "schema_version": 1,
        "instance_id": "fixture",
        "task_patch_role": "bug_inducing_mutation",
        "mutation_patch_sha256": "a" * 64,
        "mutation_baseline_commit": "b" * 40,
    }
    env._task_mutation_evidence = evidence
    monkeypatch.setattr(
        DockerEnvironment,
        "serialize",
        lambda _self: {"info": {"config": {"environment": {}}}},
        raising=False,
    )

    serialized = env.serialize()

    assert serialized["info"]["task_mutation"] == evidence


def test_valid_submission_still_raises():
    env = make_env()
    out = {"output": f"{MARKER}\ndiff --git a/x b/x\n+fix\n", "returncode": 0}
    with pytest.raises(Submitted):
        env._check_finished(out)


def test_empty_submission_rejected_with_recovery_message():
    env = make_env()
    out = {"output": f"{MARKER}\n\n", "returncode": 0}
    env._check_finished(out)  # no raise
    assert out["returncode"] == 1
    assert "SUBMISSION REJECTED" in out["output"]
    assert "git diff HEAD" in out["output"]
    assert "git add -A" not in out["output"]


def test_empty_submission_retries_exhausted_still_rejects():
    env = make_env(retries=1)
    out1 = {"output": f"{MARKER}\n", "returncode": 0}
    env._check_finished(out1)  # consumes the retry
    out2 = {"output": f"{MARKER}\n", "returncode": 0}
    env._check_finished(out2)
    assert out2["returncode"] == 1
    assert "SUBMISSION REJECTED" in out2["output"]


@pytest.mark.parametrize(
    "path",
    ["tests/test_x.py", "tox.ini", "setup.py", "pyproject.toml"],
)
def test_submission_rejects_protected_or_packaging_changes(path):
    patch = f"diff --git a/{path} b/{path}\n--- a/{path}\n+++ b/{path}\n"
    accepted, reason = validate_submission_patch(patch)
    assert accepted is False
    assert reason


def test_submission_rejects_renames_and_traversal():
    rename = "diff --git a/src/old.py b/src/new.py\n--- a/src/old.py\n+++ b/src/new.py\n"
    traversal = "diff --git a/../outside.py b/../outside.py\n--- a/../outside.py\n+++ b/../outside.py\n"

    assert validate_submission_patch(rename)[0] is False
    assert validate_submission_patch(traversal)[0] is False


def test_submission_accepts_regular_source_change():
    patch = "diff --git a/src/x.py b/src/x.py\n--- a/src/x.py\n+++ b/src/x.py\n"
    assert validate_submission_patch(patch) == (True, None)


def test_unsafe_submission_retries_exhausted_still_rejects():
    env = make_env(retries=0)
    patch = "diff --git a/tests/test_x.py b/tests/test_x.py\n--- a/tests/test_x.py\n+++ b/tests/test_x.py\n"
    out = {"output": f"{MARKER}\n{patch}", "returncode": 0}

    env._check_finished(out)

    assert out["returncode"] == 1
    assert "SUBMISSION REJECTED" in out["output"]
    assert "tests/test_x.py" in out["output"]


def test_marker_not_first_gets_one_nudge():
    env = make_env()
    out = {"output": f"diff --git a/x b/x\n{MARKER}\nmore\n", "returncode": 0}
    env._check_finished(out)
    assert "NOT as the first output line" in out["output"]
    out2 = {"output": f"noise\n{MARKER}\n", "returncode": 0}
    env._check_finished(out2)
    assert "NOT as the first output line" not in out2["output"]  # nudge spent


def test_standalone_marker_after_noise_accepts_following_diff():
    env = make_env()
    patch = "diff --git a/src/x.py b/src/x.py\n--- a/src/x.py\n+++ b/src/x.py\n"
    out = {
        "output": f"diagnostic noise\n{MARKER}\n{patch}",
        "returncode": 0,
    }

    with pytest.raises(Submitted):
        env._check_finished(out)


def test_marker_after_noise_rejects_empty_suffix():
    env = make_env()
    out = {"output": f"diagnostic noise\n{MARKER}\n", "returncode": 0}

    env._check_finished(out)

    assert out["returncode"] == 1
    assert "SUBMISSION REJECTED" in out["output"]


def test_plain_output_untouched():
    env = make_env()
    out = {"output": "just some ls output\n", "returncode": 0}
    env._check_finished(out)
    assert out["output"] == "just some ls output\n"
    assert out["returncode"] == 0


def test_nonzero_rc_marker_not_submission():
    env = make_env()
    out = {"output": f"{MARKER}\ndiff\n", "returncode": 1}
    env._check_finished(out)  # parent ignores rc!=0; no raise, no mutation
    assert "SUBMISSION REJECTED" not in out["output"]
