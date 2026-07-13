#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from phaseD_sft.agentic_trace_filters import trace_should_keep


@dataclass(frozen=True)
class BehaviorCase:
    name: str
    messages: tuple[dict[str, Any], ...]


def bash_call(command: str, call_id: str = "call_1") -> list[dict[str, Any]]:
    return [
        {
            "id": call_id,
            "type": "function",
            "function": {"name": "bash", "arguments": json.dumps({"command": command})},
        }
    ]


def assistant_bash(content: str, command: str, call_id: str = "call_1") -> dict[str, Any]:
    return {"role": "assistant", "content": content, "tool_calls": bash_call(command, call_id)}


def msg(role: str, content: str) -> dict[str, Any]:
    return {"role": role, "content": content, "tool_calls": []}


def tool_obs(content: str) -> dict[str, Any]:
    # Gemma's native chat template in this training stack expects a specific
    # tool-message shape. The source traces represent observations as user
    # messages, so synthetic traces follow that safer convention.
    return {"role": "user", "content": f"OBSERVATION:\n{content}", "tool_calls": []}


def malformed_tool_cases() -> list[BehaviorCase]:
    return [
        BehaviorCase(
            name="format_error_nested_parameters_to_one_bash",
            messages=(
                msg(
                    "user",
                    "Tool call error: Missing 'command' argument in bash tool call. "
                    "Your previous arguments were nested under parameters. Reply with exactly one bash tool call.",
                ),
                assistant_bash(
                    "I will use one valid bash command with the required command argument.",
                    "find . -maxdepth 3 -type f -name '*.py' | head -n 50",
                ),
            ),
        ),
        BehaviorCase(
            name="format_error_commands_array_to_one_bash",
            messages=(
                msg(
                    "user",
                    "Tool call error: bash arguments used a commands list. Reply with exactly one bash tool call "
                    "whose JSON has only a command string.",
                ),
                assistant_bash(
                    "I will run one concise inspection command.",
                    "grep -R \"def separability_matrix\" -n . | head -n 20",
                ),
            ),
        ),
        BehaviorCase(
            name="format_error_text_pseudo_tool_to_real_tool",
            messages=(
                msg(
                    "user",
                    "No tool calls found. Do not describe tool calls in prose. Reply with exactly one real bash tool call.",
                ),
                assistant_bash(
                    "I will call bash directly instead of describing the call.",
                    "sed -n '1,220p' astropy/modeling/separable.py",
                ),
            ),
        ),
    ]


def read_loop_cases() -> list[BehaviorCase]:
    repeated = 'grep -n "def separable" astropy/modeling/core.py | grep -A 100 "class CompoundModel" | head -n 100'
    edit_command = (
        "python - <<'PY'\n"
        "from pathlib import Path\n"
        "p = Path('astropy/modeling/separable.py')\n"
        "s = p.read_text()\n"
        "old = '        cright = _coord_matrix(right, \"right\", noutp)\\n'\n"
        "new = '        cright = _separable(right)\\n'\n"
        "if old not in s:\n"
        "    raise SystemExit('target line not found')\n"
        "p.write_text(s.replace(old, new, 1))\n"
        "PY"
    )
    return [
        BehaviorCase(
            name="loop_warning_repeated_read_make_edit",
            messages=(
                msg(
                    "user",
                    "Fix nested CompoundModel separability. You have already inspected "
                    "astropy/modeling/separable.py and astropy/modeling/core.py.",
                ),
                assistant_bash("I will inspect the separable implementation.", "sed -n '200,320p' astropy/modeling/separable.py"),
                tool_obs("<returncode>0</returncode>\n<output>def _cstack(left, right): ...</output>"),
                assistant_bash("I will inspect CompoundModel.separable.", repeated),
                tool_obs("<returncode>0</returncode>\n<output></output>"),
                assistant_bash("I will try the same lookup once more.", repeated),
                tool_obs("<returncode>0</returncode>\n<output></output>"),
                msg(
                    "user",
                    "Loop warning: you repeated the same read-only command. Do not run it again; make a minimal edit now.",
                ),
                assistant_bash("I have enough context; I will make the smallest source edit now.", edit_command),
            ),
        ),
        BehaviorCase(
            name="after_edit_run_diff_not_more_reads",
            messages=(
                msg("user", "You edited the relevant source file. Stop reading and submit the patch."),
                assistant_bash("I will inspect the current source diff.", "git diff -- astropy/modeling/separable.py > patch.txt && cat patch.txt"),
            ),
        ),
        BehaviorCase(
            name="after_nonempty_diff_submit",
            messages=(
                msg("user", "The current patch is non-empty. Submit it now."),
                assistant_bash("I will submit the non-empty patch.", "echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT && cat patch.txt"),
            ),
        ),
    ]


def behavior_cases() -> list[BehaviorCase]:
    return [*malformed_tool_cases(), *read_loop_cases()]


def row_for_case(case: BehaviorCase, repeat: int) -> dict[str, Any]:
    return {
        "instance_id": f"agentic-behavior-{case.name}-{repeat:03d}",
        "source": "agentic_behavior_synthetic",
        "messages": [dict(message) for message in case.messages],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", default="data/unsloth_agentic_24k_train_normalized_format_plus_coder_repair")
    parser.add_argument("--out", default="data/unsloth_agentic_24k_train_agentic_behavior_v2")
    parser.add_argument("--repeat", type=int, default=128)
    parser.add_argument("--base-limit", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--filter-base-quality", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--max-first-edit-ratio", type=float, default=0.35)
    parser.add_argument("--max-read-streak", type=int, default=4)
    parser.add_argument("--require-verify-tail", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    from datasets import Dataset, concatenate_datasets, load_from_disk

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

    rows = [row_for_case(case, repeat) for repeat in range(args.repeat) for case in behavior_cases()]
    behavior = Dataset.from_list(rows, features=base.features)
    mixed = concatenate_datasets([base, behavior])
    mixed.save_to_disk(args.out)
    print(f"base_rows={len(base)}")
    print(f"behavior_rows={len(behavior)}")
    print(f"total_rows={len(mixed)}")
    print(f"out={args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
