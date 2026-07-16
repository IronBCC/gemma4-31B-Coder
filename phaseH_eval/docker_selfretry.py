"""Docker environment with submission self-retry (empty-patch recovery).

Root cause (2026-07-15, cp300 smoke investigation): minisweagent's docker env
accepts a submission whenever the FIRST output line is the marker and rc==0 —
even when the printed patch is EMPTY — and does not react when the marker
appears mid-output. Real completed work was lost both ways (8/30 cases had
tracked hunks mid-trajectory but scored empty).

This subclass:
  1. Rejects an EMPTY submission up to ``max_submit_retries`` times, returning
     a corrective observation instead of raising ``Submitted``.
  2. Nudges once when the marker appears in the output but not as the first
     line (compound-command fumble), instead of silently ignoring it.

Wire via ``environment_class: docker_selfretry.DockerSelfRetryEnv`` with
PYTHONPATH=phaseH_eval (already set by smoke_single.sh).
"""
from __future__ import annotations

from minisweagent.environments.docker import DockerEnvironment

MARKER = "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT"

EMPTY_PATCH_MSG = (
    "<returncode>1</returncode>\n"
    "SUBMISSION REJECTED: your printed patch was EMPTY, so nothing would be "
    "submitted. Your file edits may still exist in the repository. Recover in "
    "two commands:\n"
    "1. git add -A && git diff --cached > patch.txt && cat patch.txt   "
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


class DockerSelfRetryEnv(DockerEnvironment):
    def __init__(self, *args, max_submit_retries: int = 2, **kwargs):
        super().__init__(*args, **kwargs)
        self._submit_retries_left = max_submit_retries
        self._marker_nudges_left = 1

    def _check_finished(self, output: dict):
        text = output.get("output", "")
        lines = text.lstrip().splitlines(keepends=True)
        marker_first = bool(lines) and lines[0].strip() == MARKER
        if marker_first and output["returncode"] == 0:
            submission = "".join(lines[1:])
            if not submission.strip() and self._submit_retries_left > 0:
                self._submit_retries_left -= 1
                output["output"] = EMPTY_PATCH_MSG
                output["returncode"] = 1
                return  # no Submitted: agent gets one recovery turn
            return super()._check_finished(output)
        if (not marker_first and MARKER in text and output["returncode"] == 0
                and self._marker_nudges_left > 0):
            self._marker_nudges_left -= 1
            output["output"] = text + MARKER_NOT_FIRST_MSG
