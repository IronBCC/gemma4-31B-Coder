"""Reproducibly summarize edit-first SWE-Lite smoke artifacts."""

from __future__ import annotations

import argparse
import json
import re
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any


EDIT_COMMAND = re.compile(
    r"sed -i|perl -i|apply_patch|git apply|cat > |cat >> |tee |>>|python3? - <<|str_replace|open\(.+[\"']w[\"']"
)


def _command(tool_call: dict[str, Any]) -> str:
    function = tool_call.get("function") or {}
    arguments = function.get("arguments", {})
    try:
        arguments = json.loads(arguments) if isinstance(arguments, str) else arguments
    except json.JSONDecodeError:
        return ""
    return arguments.get("command", "") if isinstance(arguments, dict) else ""


def _first_edit_command(messages: list[dict[str, Any]]) -> int | None:
    commands: list[str] = []
    for message in messages:
        if message.get("role") != "assistant":
            continue
        commands.extend(_command(call) for call in message.get("tool_calls") or [])
    for index, command in enumerate((command for command in commands if command), start=1):
        if EDIT_COMMAND.search(command):
            return index
    return None


def _exit_statuses(run_dir: Path) -> dict[str, set[str]]:
    """Read mini-SWE's small exit-status YAML without adding a PyYAML dependency."""
    statuses: dict[str, set[str]] = defaultdict(set)
    for path in run_dir.glob("exit_statuses_*.yaml"):
        status: str | None = None
        for raw in path.read_text().splitlines():
            if raw.startswith("    - ") and status:
                statuses[status].add(raw.removeprefix("    - ").strip())
            elif raw.startswith("    ") and raw.rstrip().endswith(":"):
                status = raw.strip().removesuffix(":")
    return statuses


def _non_empty_prediction_count(run_dir: Path) -> int | None:
    """Return the harness-facing patch count when `preds.json` is available."""
    path = run_dir / "preds.json"
    if not path.exists():
        return None
    predictions = json.loads(path.read_text())
    values = predictions.values() if isinstance(predictions, dict) else predictions
    return sum(
        bool(str(prediction.get("model_patch") or "").strip())
        for prediction in values
        if isinstance(prediction, dict)
    )


def _resolved_count(run_dir: Path) -> int | None:
    """Find the SWE-bench final report written beside the repository root."""
    pattern = f"*smoke_{run_dir.name}.json"
    for directory in (run_dir, run_dir.parent, run_dir.parent.parent, run_dir.parent.parent.parent):
        for path in sorted(directory.glob(pattern), key=lambda item: item.stat().st_mtime, reverse=True):
            try:
                report = json.loads(path.read_text())
            except json.JSONDecodeError:
                continue
            value = report.get("resolved_instances")
            if isinstance(value, int):
                return value
    return None


def summarize_run(run_dir: Path) -> dict[str, int | float | None]:
    trajectories = sorted(run_dir.glob("**/*.traj.json"))
    statuses = _exit_statuses(run_dir)
    attempted_ids = set().union(*statuses.values()) if statuses else set()
    first_edits: list[int] = []
    non_empty_patches = 0

    for path in trajectories:
        trace = json.loads(path.read_text())
        instance_id = trace.get("instance_id")
        if instance_id:
            attempted_ids.add(instance_id)
        if str((trace.get("info") or {}).get("submission") or "").strip():
            non_empty_patches += 1
        first_edit = _first_edit_command(trace.get("messages") or [])
        if first_edit is not None:
            first_edits.append(first_edit)

    attempted = len(attempted_ids) if attempted_ids else len(trajectories)
    format_errors = sum(
        len(instance_ids)
        for status, instance_ids in statuses.items()
        if "format" in status.lower()
    )
    prediction_patches = _non_empty_prediction_count(run_dir)
    return {
        "attempted": attempted,
        "trajectories": len(trajectories),
        "non_empty_patches": (
            prediction_patches if prediction_patches is not None else non_empty_patches
        ),
        "format_errors": format_errors,
        "format_error_rate": format_errors / attempted if attempted else 0.0,
        "edit_reach": len(first_edits),
        "first_edit_median": statistics.median(first_edits) if first_edits else None,
        "first_edit_by_command_10": sum(index <= 10 for index in first_edits),
        "resolved": _resolved_count(run_dir),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path, help="model-specific smoke output directory")
    args = parser.parse_args()
    print(json.dumps(summarize_run(args.run_dir), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
