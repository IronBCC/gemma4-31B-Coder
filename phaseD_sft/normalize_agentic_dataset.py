#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any

from datasets import load_from_disk

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from phaseD_sft.agent_smoke_eval import CASES
from phaseD_sft.agentic_trace_filters import trace_should_keep
from phaseD_sft.build_format_correction_dataset import ANSWERS


FENCE_RE = re.compile(r"^\s*```(?:diff|patch|python|bash)?\s*\n(?P<body>.*?)\n```\s*$", re.DOTALL)


def text_content(message: dict[str, Any]) -> str:
    content = message.get("content") or ""
    return content if isinstance(content, str) else str(content)


def patch_like(text: str) -> bool:
    stripped = text.lstrip()
    return (
        stripped.startswith("--- ")
        or stripped.startswith("diff --git ")
        or ("@@ " in stripped and ("\n-" in stripped or "\n+" in stripped))
    )


def normalize_patch_text(text: str, counts: Counter[str]) -> str:
    stripped = text.strip()

    match = FENCE_RE.match(stripped)
    if match:
        body = match.group("body").strip()
        if patch_like(body):
            stripped = body
            counts["stripped_fence"] += 1

    first_diff = stripped.find("--- ")
    if first_diff > 0:
        prefix = stripped[:first_diff].strip()
        if prefix and all(re.fullmatch(r"(?:a/|b/)?[^\s]+", part) for part in prefix.split()):
            stripped = stripped[first_diff:].lstrip()
            counts["dropped_path_preamble"] += 1

    lines = stripped.splitlines()
    if lines and re.fullmatch(r"a/[^\s]+(?:\s+b/[^\s]+)?", lines[0]) and any(line.startswith("@@") for line in lines[1:]):
        path = lines[0].split()[0]
        rest = lines[1:]
        stripped = "\n".join([f"--- {path}", f"+++ b/{path[2:]}", *rest])
        counts["synthesized_diff_header"] += 1

    return stripped


def normalize_message_content(content: str, counts: Counter[str]) -> str:
    normalized = normalize_patch_text(content, counts)
    if normalized != content:
        counts["changed_patch_messages"] += 1
        return normalized
    fixed = content.replace("Focused test", "focused test")
    if fixed != content:
        counts["fixed_focused_case"] += 1
    return fixed


def clone_message(message: dict[str, Any]) -> dict[str, Any]:
    cloned = dict(message)
    if "tool_calls" not in cloned:
        cloned["tool_calls"] = []
    return cloned


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/unsloth_agentic_24k_train")
    parser.add_argument("--out", default="data/unsloth_agentic_24k_train_normalized_format")
    parser.add_argument("--append-format-corrections", action="store_true")
    parser.add_argument(
        "--filter-quality",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Drop traces that read too long, edit too late, or never verify after the first edit.",
    )
    parser.add_argument("--max-first-edit-ratio", type=float, default=0.4)
    parser.add_argument("--max-read-streak", type=int, default=6)
    parser.add_argument(
        "--require-verify-tail",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Require a verify event after the first edit before keeping a trace.",
    )
    args = parser.parse_args()

    ds = load_from_disk(args.data)
    counts: Counter[str] = Counter()

    def normalize_example(example: dict[str, Any]) -> dict[str, Any]:
        messages = []
        for message in example["messages"]:
            cloned = clone_message(message)
            if cloned.get("role") == "assistant":
                before = text_content(cloned)
                after = normalize_message_content(before, counts)
                if after != before:
                    cloned["content"] = after
                    counts["changed_assistant_messages"] += 1
            messages.append(cloned)
        out = dict(example)
        out["messages"] = messages
        return out

    normalized = ds.map(normalize_example, desc="normalize-agentic-format")

    if args.append_format_corrections:
        from datasets import Dataset, concatenate_datasets

        rows = []
        for index, case in enumerate(CASES):
            for repeat in range(64):
                rows.append(
                    {
                        "instance_id": f"format-hard10-{index:02d}-{repeat:02d}",
                        "source": "format_hard10_correction",
                        "messages": [
                            {"role": "user", "content": case.prompt, "tool_calls": []},
                            {"role": "assistant", "content": ANSWERS[case.name], "tool_calls": []},
                        ],
                    }
                )
        correction = Dataset.from_list(rows, features=normalized.features)
        normalized = concatenate_datasets([normalized, correction])
        counts["appended_format_rows"] = len(correction)

    if args.filter_quality:
        before = len(normalized)

        def keep_quality(example: dict[str, Any]) -> bool:
            return trace_should_keep(
                example["messages"],
                max_first_edit_ratio=args.max_first_edit_ratio,
                max_read_streak=args.max_read_streak,
                require_verify_tail=args.require_verify_tail,
            )

        normalized = normalized.filter(keep_quality, desc="filter-agentic-quality")
        counts["quality_rows_kept"] = len(normalized)
        counts["quality_rows_dropped"] = before - len(normalized)
        print(
            f"[quality] kept {len(normalized)}/{before} rows "
            f"(max_first_edit_ratio={args.max_first_edit_ratio}, "
            f"max_read_streak={args.max_read_streak}, "
            f"require_verify_tail={args.require_verify_tail})",
            flush=True,
        )

    normalized.save_to_disk(args.out)
    print(f"[done] saved {len(normalized)} rows -> {args.out}")
    for key, value in sorted(counts.items()):
        print(f"{key}: {value}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
