#!/usr/bin/env python3
"""Structural scan of an agentic dataset: schema, roles, tool calls, CoT markers."""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

from phaseD_sft.progress import EtaProgress


DEFAULT_DATA_DIR = "data/unsloth_agentic_24k_train_swe_edit_trace_v5_budget18432"

THINK_TAG_RE = re.compile(r"<(thought|thinking|think)>", re.IGNORECASE)
TOOL_CODE_RE = re.compile(r"```tool_code")
OMIT_RE = re.compile(r"\[\.\.\. omitted")


def scan_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            stripped = line.strip()
            if not stripped:
                continue
            try:
                row = json.loads(stripped)
            except json.JSONDecodeError:
                continue
            if isinstance(row, dict):
                rows.append(row)
    return rows


def scan_hf(dataset_path: str) -> list[dict]:
    from datasets import load_from_disk

    ds = load_from_disk(dataset_path)
    return [dict(row) for row in ds]


def run_scan(rows: list[dict], source_label: str, *, progress_every: int = 500) -> dict:
    if progress_every <= 0:
        raise ValueError("progress_every must be positive")
    per_source: dict[str, Counter] = defaultdict(Counter)
    glob_counter: Counter[str] = Counter()
    instance_ids: Counter = Counter()
    src_of_id: dict = defaultdict(set)
    obs_len_max: dict[str, int] = {}
    failures: list[str] = []
    progress = EtaProgress("scan-agentic-dataset", total=len(rows))

    for i, row in enumerate(rows):
        src = row.get("source", "MISSING")
        c = per_source[src]
        c["rows"] += 1

        iid = row.get("instance_id")
        if iid is not None:
            instance_ids[iid] += 1
            src_of_id[iid].add(src)

        msgs = row.get("messages")
        if not isinstance(msgs, list) or not msgs:
            msg = f"row {i} [{src}]: messages missing/empty"
            failures.append(msg)
            glob_counter["failures"] += 1
            completed = i + 1
            if completed % progress_every == 0 or completed == len(rows):
                print(progress.update(completed), flush=True)
            continue

        roles = [m.get("role") for m in msgs]
        c[f"first_role:{roles[0]}"] += 1
        c[f"last_role:{roles[-1]}"] += 1
        for r in set(roles):
            c[f"has_role:{r}"] += 1

        # Role alternation check (user/assistant strict)
        ok_alt = all(
            (roles[j] == "user" and (j == 0 or roles[j - 1] == "assistant"))
            or (roles[j] == "assistant" and j > 0 and roles[j - 1] == "user")
            or roles[j] not in ("user", "assistant")
            for j in range(len(roles))
        )
        if not ok_alt:
            c["bad_alternation"] += 1

        n_user = 0
        for j, m in enumerate(msgs):
            role = m.get("role")
            content = m.get("content")
            tcs = m.get("tool_calls")
            if not isinstance(content, str):
                msg = f"row {i} [{src}] msg {j}: non-string content type={type(content).__name__}"
                failures.append(msg)
                c["nonstring_content"] += 1
                continue

            if role == "user":
                n_user += 1
                if content.startswith("OBSERVATION:"):
                    c["obs_msgs"] += 1
                    L = len(content)
                    obs_len_max[src] = max(obs_len_max.get(src, 0), L)
                    if L > 2600:
                        c["obs_over_2600"] += 1
                    if OMIT_RE.search(content):
                        c["obs_with_omit_marker"] += 1
                    if "[unchanged observation omitted" in content:
                        c["obs_dedup_marker"] += 1
                else:
                    c["user_task_msgs"] += 1
                if tcs:
                    c["user_with_tool_calls"] += 1
                    msg = f"row {i} [{src}] msg {j}: user has tool_calls"
                    failures.append(msg)

            elif role == "assistant":
                c["asst_msgs"] += 1
                if THINK_TAG_RE.search(content):
                    c["asst_think_tag"] += 1
                if TOOL_CODE_RE.search(content):
                    c["asst_tool_code_fence"] += 1
                if isinstance(tcs, list) and tcs:
                    c["asst_with_tool_calls"] += 1
                    for tc in tcs or []:
                        fn = (tc or {}).get("function") or {}
                        name = fn.get("name")
                        c[f"tool:{name}"] += 1
                        if tc.get("type") != "function":
                            msg = f"row {i} [{src}] msg {j}: tool_call type={tc.get('type')}"
                            failures.append(msg)
                            c["bad_tc_type"] += 1
                        try:
                            json.loads(fn.get("arguments") or "")
                        except Exception:
                            msg = f"row {i} [{src}] msg {j}: tool_call args not JSON"
                            failures.append(msg)
                            c["bad_tc_args"] += 1
                else:
                    c["asst_no_tool_calls"] += 1
                    if not content.strip():
                        msg = f"row {i} [{src}] msg {j}: assistant empty content and no tool_calls"
                        failures.append(msg)
                        c["asst_empty"] += 1

            else:
                c[f"other_role:{role}"] += 1

        completed = i + 1
        if completed % progress_every == 0 or completed == len(rows):
            print(progress.update(completed), flush=True)

    dups = {k: v for k, v in instance_ids.items() if v > 1}
    cross = {k: sorted(v) for k, v in src_of_id.items() if len(v) > 1}

    return {
        "rows": len(rows),
        "per_source": {k: dict(v) for k, v in per_source.items()},
        "obs_len_max": obs_len_max,
        "dup_instance_ids": {"count": len(dups), "examples": dict(list(dups.items())[:10])},
        "cross_source_ids": {"count": len(cross), "examples": dict(list(cross.items())[:10])},
        "failure_count": len(failures),
        "failures": failures[:40],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset", default=DEFAULT_DATA_DIR, help="Dataset path (HF directory or JSONL file)."
    )
    parser.add_argument(
        "--hf", action="store_true", help="Treat --dataset as an HF dataset directory."
    )
    parser.add_argument(
        "--jsonl", action="store_true", help="Treat --dataset as a JSONL file (default if not --hf)."
    )
    parser.add_argument("--progress-every", type=int, default=500)
    args = parser.parse_args()

    path = Path(args.dataset)
    use_hf = args.hf or (not args.jsonl and path.is_dir())

    if use_hf:
        rows = scan_hf(str(path))
    else:
        rows = scan_jsonl(path)

    report = run_scan(rows, source_label=path.name, progress_every=args.progress_every)
    print(json.dumps(report, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
