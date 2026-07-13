#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shlex
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from phaseD_sft.agentic_trace_filters import trace_should_keep


def bash_call(command: str, call_id: str = "call_1") -> list[dict[str, Any]]:
    return [
        {
            "id": call_id,
            "type": "function",
            "function": {"name": "bash", "arguments": json.dumps({"command": command})},
        }
    ]


def assistant_bash(command: str, content: str = "") -> dict[str, Any]:
    return {"role": "assistant", "content": content, "tool_calls": bash_call(command)}


def user_msg(content: str) -> dict[str, Any]:
    return {"role": "user", "content": content, "tool_calls": []}


def observation(content: str) -> dict[str, Any]:
    return user_msg(f"OBSERVATION:\n{content}")


def shorten(text: str, max_chars: int) -> str:
    normalized = str(text or "").strip()
    if len(normalized) <= max_chars:
        return normalized
    marker = "\n[... omitted ...]\n"
    head = max_chars // 2
    tail = max_chars - head - len(marker)
    return normalized[:head].rstrip() + marker + normalized[-tail:].lstrip()


def parse_json_list(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item) for item in value if str(item).strip()]
    if not isinstance(value, str) or not value.strip():
        return []
    try:
        decoded = json.loads(value)
    except json.JSONDecodeError:
        return []
    if not isinstance(decoded, list):
        return []
    return [str(item) for item in decoded if str(item).strip()]


def patch_paths(patch: str) -> list[str]:
    paths: list[str] = []
    for line in patch.splitlines():
        if line.startswith("diff --git "):
            parts = line.split()
            if len(parts) >= 4:
                candidate = parts[2]
                if candidate.startswith("a/"):
                    paths.append(candidate[2:])
        elif line.startswith("+++ b/"):
            paths.append(line.removeprefix("+++ b/").strip())
    deduped: list[str] = []
    for path in paths:
        if path != "/dev/null" and path not in deduped:
            deduped.append(path)
    return deduped


def primary_path(paths: list[str]) -> str:
    for path in paths:
        if path.endswith(".py"):
            return path
    return paths[0] if paths else "."


def inspect_command(path: str) -> str:
    quoted = shlex.quote(path)
    return f"sed -n '1,220p' {quoted}"


def apply_patch_command(patch: str) -> str:
    return "git apply <<'PATCH'\n" + patch.rstrip() + "\nPATCH"


def verify_command(row: dict[str, Any], *, max_tests: int) -> str:
    tests = parse_json_list(row.get("FAIL_TO_PASS"))
    if not tests:
        tests = parse_json_list(row.get("PASS_TO_PASS"))
    if tests:
        quoted = " ".join(shlex.quote(test) for test in tests[:max_tests])
        return f"python -m pytest -q {quoted}"
    paths = patch_paths(str(row.get("patch") or ""))
    test_paths = [path for path in paths if "/tests/" in path or path.startswith("tests/")]
    if test_paths:
        quoted = " ".join(shlex.quote(path) for path in test_paths[:max_tests])
        return f"python -m pytest -q {quoted}"
    return "python -m pytest -q"


def diff_command(paths: list[str]) -> str:
    if paths:
        quoted = " ".join(shlex.quote(path) for path in paths[:8])
        return f"git diff -- {quoted} > patch.txt && cat patch.txt"
    return "git diff -- . > patch.txt && cat patch.txt"


def row_prompt(row: dict[str, Any], *, max_problem_chars: int) -> str:
    problem = shorten(str(row.get("problem_statement") or ""), max_problem_chars)
    repo = row.get("repo") or "unknown"
    instance_id = row.get("instance_id") or "unknown"
    return (
        "You are fixing a SWE-bench issue. Work edit-first: inspect only the most relevant file, "
        "make the minimal source edit, run a focused verification command, then submit a non-empty diff.\n\n"
        f"repo: {repo}\n"
        f"instance_id: {instance_id}\n\n"
        f"issue:\n{problem}"
    )


def row_from_swe_instance(
    row: dict[str, Any],
    *,
    repeat: int,
    max_problem_chars: int,
    max_patch_chars: int,
    max_tests: int,
) -> dict[str, Any] | None:
    patch = str(row.get("patch") or "").strip()
    if not patch.startswith("diff --git ") or len(patch) > max_patch_chars:
        return None
    paths = patch_paths(patch)
    if not paths:
        return None
    path = primary_path(paths)
    instance_id = str(row.get("instance_id") or "unknown")
    messages = [
        user_msg(row_prompt(row, max_problem_chars=max_problem_chars)),
        assistant_bash(inspect_command(path), f"I will inspect the likely target file {path}."),
        observation(
            "<returncode>0</returncode>\n"
            f"<output>Focused inspection complete for {path}. The issue requires a source edit.</output>"
        ),
        assistant_bash(apply_patch_command(patch), "I have enough context; I will apply the minimal source patch now."),
        observation("<returncode>0</returncode>\n<output>Patch applied cleanly.</output>"),
        assistant_bash(verify_command(row, max_tests=max_tests), "I will run focused verification before submitting."),
        observation("<returncode>0</returncode>\n<output>Focused verification passed.</output>"),
        assistant_bash(diff_command(paths), "I will inspect the non-empty source diff."),
        observation("<returncode>0</returncode>\n<output>" + shorten(patch, max_patch_chars) + "</output>"),
        assistant_bash("echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT && cat patch.txt", "I will submit the verified non-empty patch."),
    ]
    return {
        "instance_id": f"swe-edit-trace-{instance_id}-{repeat:03d}",
        "source": "swe_train_oracle_edit_trace",
        "messages": messages,
    }


def build_rows(
    swe_rows: list[dict[str, Any]],
    *,
    repeat: int,
    limit: int,
    max_problem_chars: int,
    max_patch_chars: int,
    max_tests: int,
) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    for row in swe_rows:
        probe = row_from_swe_instance(
            row,
            repeat=0,
            max_problem_chars=max_problem_chars,
            max_patch_chars=max_patch_chars,
            max_tests=max_tests,
        )
        if probe is None:
            continue
        selected.append(row)
        if limit > 0 and len(selected) >= limit:
            break

    rows: list[dict[str, Any]] = []
    for repeat_index in range(repeat):
        for row in selected:
            built = row_from_swe_instance(
                row,
                repeat=repeat_index,
                max_problem_chars=max_problem_chars,
                max_patch_chars=max_patch_chars,
                max_tests=max_tests,
            )
            if built is not None:
                rows.append(built)
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", default="data/unsloth_agentic_24k_train_normalized_format_plus_coder_repair")
    parser.add_argument("--out", default="data/unsloth_agentic_24k_train_swe_edit_trace_v4")
    parser.add_argument("--swe-dataset", default="princeton-nlp/SWE-bench")
    parser.add_argument("--swe-split", default="train")
    parser.add_argument("--limit", type=int, default=256)
    parser.add_argument("--repeat", type=int, default=4)
    parser.add_argument("--max-problem-chars", type=int, default=3000)
    parser.add_argument("--max-patch-chars", type=int, default=6000)
    parser.add_argument("--max-tests", type=int, default=3)
    parser.add_argument("--base-limit", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--filter-base-quality", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--max-first-edit-ratio", type=float, default=0.35)
    parser.add_argument("--max-read-streak", type=int, default=4)
    parser.add_argument("--require-verify-tail", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    from datasets import Dataset, concatenate_datasets, load_dataset, load_from_disk

    base = load_from_disk(args.base)
    if args.base_limit > 0 and args.base_limit < len(base):
        base = base.shuffle(seed=args.seed).select(range(args.base_limit))
    if args.filter_base_quality:
        before = len(base)

        def keep_quality(example: dict[str, Any]) -> bool:
            return trace_should_keep(
                example["messages"],
                max_first_edit_ratio=args.max_first_edit_ratio,
                max_read_streak=args.max_read_streak,
                require_verify_tail=args.require_verify_tail,
            )

        base = base.filter(keep_quality, desc="filter-base-quality")
        print(f"base_quality_rows={len(base)}/{before}", flush=True)

    swe = load_dataset(args.swe_dataset, split=args.swe_split)
    if args.seed:
        swe = swe.shuffle(seed=args.seed)
    rows = build_rows(
        [dict(row) for row in swe],
        repeat=args.repeat,
        limit=args.limit,
        max_problem_chars=args.max_problem_chars,
        max_patch_chars=args.max_patch_chars,
        max_tests=args.max_tests,
    )
    if not rows:
        raise SystemExit("no SWE edit trace rows built")

    edit_traces = Dataset.from_list(rows, features=base.features)
    mixed = concatenate_datasets([base, edit_traces])
    mixed.save_to_disk(args.out)

    print(f"swe_dataset={args.swe_dataset}")
    print(f"swe_split={args.swe_split}")
    print(f"base_rows={len(base)}")
    print(f"edit_trace_rows={len(edit_traces)}")
    print(f"total_rows={len(mixed)}")
    print(f"out={args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
