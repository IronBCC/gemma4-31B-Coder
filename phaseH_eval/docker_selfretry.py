"""Docker environment with fail-closed submission recovery.

Root cause (2026-07-15, cp300 smoke investigation): minisweagent's docker env
accepts a submission whenever the FIRST output line is the marker and rc==0 —
even when the printed patch is EMPTY — and does not react when the marker
appears mid-output. Real completed work was lost both ways (8/30 cases had
tracked hunks mid-trajectory but scored empty).

This subclass:
  1. Rejects EMPTY and structurally unsafe submissions, returning a corrective
     observation instead of raising ``Submitted``.
  2. Nudges once when the marker appears in the output but not as the first
     line (compound-command fumble), instead of silently ignoring it.

Wire via ``environment_class: docker_selfretry.DockerSelfRetryEnv`` with
PYTHONPATH=phaseH_eval (already set by smoke_single.sh).
"""
from __future__ import annotations

import hashlib
from pathlib import PurePosixPath
import re
import shlex
import subprocess

from minisweagent.environments.docker import DockerEnvironment
try:
    from phaseH_eval.model_clamps import SOURCE_ONLY_DIFF_COMMAND
except ModuleNotFoundError as exc:
    if exc.name != "phaseH_eval":
        raise
    from model_clamps import SOURCE_ONLY_DIFF_COMMAND

MARKER = "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT"

EMPTY_PATCH_MSG = (
    "<returncode>1</returncode>\n"
    "SUBMISSION REJECTED: your printed patch was EMPTY, so nothing would be "
    "submitted. Your file edits may still exist in the repository. Recover in "
    "two commands:\n"
    f"1. {SOURCE_ONLY_DIFF_COMMAND}   "
    "(verify it shows your changes)\n"
    f"2. echo {MARKER} && cat patch.txt\n"
    f"The line `{MARKER}` must be the FIRST line of output of your final command."
)

MARKER_NOT_FIRST_MSG = (
    "\n<warning>Submission marker detected but NOT as the first output line, "
    "so no submission happened. To submit, run exactly:\n"
    f"echo {MARKER} && cat patch.txt\n"
    "with nothing printed before the marker.</warning>"
)

_PROTECTED_COMPONENTS = {
    "test",
    "tests",
    "testing",
    "fixture",
    "fixtures",
    "benchmark",
    "benchmarks",
    "harness",
    "harnesses",
}
_PROTECTED_BASENAMES = {
    "conftest.py",
    "eval.sh",
    "evaluation.sh",
    "noxfile.py",
    "pyproject.toml",
    "pytest.ini",
    "run_tests.py",
    "run_tests.sh",
    "runtests.py",
    "setup.cfg",
    "setup.py",
    "test-requirements.txt",
    "tox.ini",
}


def validate_submission_patch(patch: str) -> tuple[bool, str | None]:
    """Validate patch structure and path scope without judging semantics."""

    if not isinstance(patch, str) or not patch.strip():
        return False, "patch is empty"
    if any(
        line.startswith(("rename from ", "rename to ", "copy from ", "copy to "))
        for line in patch.splitlines()
    ):
        return False, "renames and copies are not allowed"

    paths: set[str] = set()
    for line in patch.splitlines():
        if not line.startswith("diff --git "):
            continue
        try:
            pieces = shlex.split(line)
        except ValueError:
            return False, "patch has a malformed git diff header"
        if len(pieces) != 4 or pieces[:2] != ["diff", "--git"]:
            return False, "patch has a malformed git diff header"
        try:
            old_path = _normalized_patch_path(pieces[2])
            new_path = _normalized_patch_path(pieces[3])
        except ValueError as exc:
            return False, str(exc)
        if old_path != new_path:
            return False, f"patch renames a path: {old_path} -> {new_path}"
        paths.add(old_path)

    if not paths:
        return False, "patch has no git diff headers"
    protected = [path for path in sorted(paths) if _is_protected_path(path)]
    if protected:
        return False, f"protected path is not allowed: {', '.join(protected)}"
    return True, None


def _normalized_patch_path(raw: str) -> str:
    if raw.startswith(("a/", "b/")):
        raw = raw[2:]
    raw_parts = raw.split("/")
    path = PurePosixPath(raw)
    if (
        not raw
        or raw.startswith("/")
        or "\\" in raw
        or "\x00" in raw
        or any(part in {"", ".", ".."} for part in raw_parts)
        or path.is_absolute()
    ):
        raise ValueError(f"patch contains an unsafe path: {raw!r}")
    return str(path)


def _is_protected_path(raw: str) -> bool:
    path = PurePosixPath(raw)
    components = tuple(part.casefold() for part in path.parts)
    basename = path.name.casefold()
    return (
        any(component in _PROTECTED_COMPONENTS for component in components)
        or basename in _PROTECTED_BASENAMES
        or basename.startswith("test_")
        or basename.endswith(("_test.py", "_tests.py"))
    )


def _reject_submission(output: dict, reason: str, retries_left: int) -> None:
    output["output"] = (
        "<returncode>1</returncode>\n"
        f"SUBMISSION REJECTED: {reason}. "
        "Inspect `git diff --check` and `git status --short`, repair or revert "
        "the invalid hunk, run a focused behavior check, then submit a non-empty "
        "source-only patch."
        f" Recovery retries remaining: {max(0, retries_left)}."
    )
    output["returncode"] = 1


class DockerSelfRetryEnv(DockerEnvironment):
    def __init__(self, *args, max_submit_retries: int = 2, **kwargs):
        super().__init__(*args, **kwargs)
        self._submit_retries_left = max_submit_retries
        self._marker_nudges_left = 1
        self._task_mutation_evidence = None

    def establish_task_baseline(
        self,
        instance_id: str,
        mutation_patch: str,
    ) -> dict[str, str | int]:
        """Apply and commit a bound SWE-smith mutation before the agent runs."""

        if not isinstance(instance_id, str) or not instance_id.strip():
            raise ValueError("instance_id is required for task mutation")
        accepted, reason = validate_submission_patch(mutation_patch)
        if not accepted:
            raise ValueError(f"invalid task mutation: {reason}")

        container_id = getattr(self, "container_id", "")
        if not isinstance(container_id, str) or not re.fullmatch(
            r"[0-9a-f]{12}(?:[0-9a-f]{52})?",
            container_id,
        ):
            raise RuntimeError("task mutation requires a valid container id")
        executable = getattr(getattr(self, "config", None), "executable", "docker")
        if not isinstance(executable, str) or not executable:
            raise RuntimeError("task mutation requires a docker executable")

        self._task_mutation_evidence = None
        applied = subprocess.run(
            [
                executable,
                "exec",
                "-i",
                container_id,
                "bash",
                "-c",
                (
                    "set -euo pipefail; cd /testbed; "
                    'test -z "$(git status --porcelain=v1 '
                    '--untracked-files=all)"; '
                    "git apply --whitespace=nowarn -; "
                    "git add -A"
                ),
            ],
            input=mutation_patch,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=180,
            check=False,
        )
        if applied.returncode != 0:
            detail = applied.stdout.strip()[-1000:]
            raise RuntimeError(f"mutation apply failed: {detail}")

        committed = subprocess.run(
            [
                executable,
                "exec",
                container_id,
                "bash",
                "-c",
                (
                    "set -euo pipefail; cd /testbed; "
                    "git -c user.name='SWE-smith Task' "
                    "-c user.email='swe-smith-task@invalid' "
                    "-c commit.gpgsign=false commit --no-verify "
                    "-m swe-smith-task-mutation >/dev/null; "
                    'test -z "$(git status --porcelain=v1 '
                    '--untracked-files=all)"; '
                    "git rev-parse HEAD"
                ),
            ],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=180,
            check=False,
        )
        commit = committed.stdout.strip().splitlines()[-1] if committed.stdout.strip() else ""
        if committed.returncode != 0 or not re.fullmatch(r"[0-9a-f]{40,64}", commit):
            detail = committed.stdout.strip()[-1000:]
            raise RuntimeError(f"mutation commit failed: {detail}")

        evidence: dict[str, str | int] = {
            "schema_version": 1,
            "instance_id": instance_id,
            "task_patch_role": "bug_inducing_mutation",
            "mutation_patch_sha256": hashlib.sha256(
                mutation_patch.encode("utf-8")
            ).hexdigest(),
            "mutation_baseline_commit": commit,
        }
        self._task_mutation_evidence = evidence
        return evidence

    def serialize(self) -> dict:
        serialized = super().serialize()
        if self._task_mutation_evidence is not None:
            serialized.setdefault("info", {})["task_mutation"] = dict(
                self._task_mutation_evidence
            )
        return serialized

    def _check_finished(self, output: dict):
        text = output.get("output", "")
        lines = text.lstrip().splitlines(keepends=True)
        marker_lines = [
            index for index, line in enumerate(lines) if line.strip() == MARKER
        ]
        marker_index = marker_lines[0] if len(marker_lines) == 1 else None
        if marker_index is not None and output["returncode"] == 0:
            submission = "".join(lines[marker_index + 1:])
            if not submission.strip():
                if self._submit_retries_left > 0:
                    self._submit_retries_left -= 1
                output["output"] = EMPTY_PATCH_MSG
                output["returncode"] = 1
                return
            if not submission.lstrip().startswith("diff --git "):
                if marker_index > 0 and self._marker_nudges_left > 0:
                    self._marker_nudges_left -= 1
                    output["output"] = text + MARKER_NOT_FIRST_MSG
                    return
                if self._submit_retries_left > 0:
                    self._submit_retries_left -= 1
                _reject_submission(
                    output,
                    "submission marker must be followed by a git diff",
                    self._submit_retries_left,
                )
                return
            accepted, reason = validate_submission_patch(submission)
            if not accepted:
                if self._submit_retries_left > 0:
                    self._submit_retries_left -= 1
                _reject_submission(
                    output,
                    reason or "patch failed structural validation",
                    self._submit_retries_left,
                )
                return
            output["output"] = f"{MARKER}\n{submission.lstrip()}"
            return super()._check_finished(output)
        if (marker_index is None and MARKER in text and output["returncode"] == 0
                and self._marker_nudges_left > 0):
            self._marker_nudges_left -= 1
            output["output"] = text + MARKER_NOT_FIRST_MSG
