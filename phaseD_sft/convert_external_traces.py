#!/usr/bin/env python3
"""Convert external HF trace datasets (Kwai-Klear, Open-SWE-Traces) to training schema."""
from __future__ import annotations

import argparse
import glob as _glob
import json
from collections import Counter
from pathlib import Path
import re
from typing import Any

try:
    import pyarrow.parquet as pq
except ImportError:  # pragma: no cover - optional dep
    pq = None  # type: ignore[assignment]


DEFAULT_OPENSWP_LANGUAGES = ("python", "rust", "cpp", "c", "c++")
_CALL_ID_COUNTER = 0

# Tools that are truly forbidden (no translation possible) and cause trajectory drop.
_FORBIDDEN_TOOL_NAMES = frozenset({"browser"})


def _next_call_id() -> str:
    global _CALL_ID_COUNTER  # noqa: PLW0603 - intentional counter across conversions
    _CALL_ID_COUNTER += 1
    return f"call_{_CALL_ID_COUNTER}"


def _reset_call_ids() -> None:
    """Reset the call ID counter (for testing determinism)."""
    global _CALL_ID_COUNTER  # noqa: PLW0603
    _CALL_ID_COUNTER = 0


_BASH_FENCE_RE = re.compile(r"```bash\s*\n(.*?)```", re.DOTALL)
_THOUGHT_LABEL_RE = re.compile(r"^THOUGHT:\s*", re.IGNORECASE)


def _extract_bash_fence(content: str) -> tuple[str, list[dict] | None]:
    """Extract the LAST ```bash fenced block from assistant content.

    Returns (plain_text_content, tool_calls_or_None). If no fence found returns
    (original_content, None).
    """
    matches = list(_BASH_FENCE_RE.finditer(content))
    if not matches:
        return content, None

    last = matches[-1]
    command = last.group(1).strip()
    call_id = _next_call_id()
    tool_calls = [
        {
            "id": call_id,
            "type": "function",
            "function": {
                "name": "bash",
                "arguments": json.dumps({"command": command}),
            },
        }
    ]

    # Strip the last fence and any leading THOUGHT: label from remaining text
    before = content[: last.start()].rstrip()
    after_fence_start = last.end()
    if after_fence_start < len(content):
        before += "\n" + content[after_fence_start:].strip()
    before = _THOUGHT_LABEL_RE.sub("", before).strip()

    return before, tool_calls


def _kwai_convert_messages(
    messages: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], Counter[str]]:
    """Convert Kwai-Klear message list to target schema."""
    stats = Counter()
    converted: list[dict[str, Any]] = []
    seen_user = False

    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "") or ""

        if role == "assistant":
            plain, tool_calls = _extract_bash_fence(content)
            entry: dict[str, Any] = {"role": "assistant", "content": plain}
            if tool_calls is not None:
                entry["tool_calls"] = tool_calls
                stats["tool_calls_extracted"] += 1
            else:
                entry["tool_calls"] = []
            seen_user = True
            converted.append(entry)

        elif role == "system":
            converted.append({"role": "system", "content": content, "tool_calls": []})

        elif role == "user":
            if not seen_user:
                # First user message is the task prompt
                converted.append({"role": "user", "content": content, "tool_calls": []})
                seen_user = True
            else:
                # Subsequent user/env messages are observations
                obs_content = f"OBSERVATION:\n{content}" if not content.startswith("OBSERVATION:") else content
                converted.append({"role": "user", "content": obs_content, "tool_calls": []})
                stats["observations_prefixed"] += 1

        else:
            # Unknown role -> treat as user observation
            obs_content = f"OBSERVATION:\n{content}" if not content.startswith("OBSERVATION:") else content
            converted.append({"role": "user", "content": obs_content, "tool_calls": []})
            stats["observations_prefixed"] += 1

    return converted, stats


def _load_parquet_rows(source: Path) -> list[dict[str, Any]]:
    """Load rows from a parquet file (or glob of files)."""
    if pq is None:
        raise ImportError("pyarrow.parquet is required; install with `pip install pyarrow`")

    paths = sorted(_glob.glob(str(source)))
    if not paths:
        return []

    all_rows: list[dict[str, Any]] = []
    for path in paths:
        table = pq.read_table(path)
        for batch in table.to_batches():
            cols = {name: batch.column(name).to_pylist() for name in batch.column_names}
            nrows = len(cols[next(iter(cols))]) if cols else 0
            for i in range(nrows):
                row = {k: cols[k][i] for k in cols}
                all_rows.append(row)

    return all_rows


def convert_kwai(
    source: Path,
    *,
    limit: int | None = None,
) -> tuple[list[dict[str, Any]], Counter[str]]:
    """Convert Kwai-Klear parquet to target schema."""
    stats = Counter()
    rows_in = 0

    raw_rows = _load_parquet_rows(source)
    if limit is not None:
        raw_rows = raw_rows[:limit]

    output: list[dict[str, Any]] = []
    for row in raw_rows:
        rows_in += 1
        instance_id = str(row.get("instance_id", f"kwai_{rows_in}"))
        messages = row.get("messages") or []
        if not isinstance(messages, list):
            stats["malformed_rows"] += 1
            continue

        converted_msgs, row_stats = _kwai_convert_messages(messages)
        stats.update(row_stats)

        output.append(
            {
                "instance_id": instance_id,
                "source": "kwai_klear_miniswe",
                "messages": converted_msgs,
            }
        )

    stats["rows_in"] += rows_in
    stats["rows_out"] += len(output)
    return output, stats


def _parse_tc_args(tc: dict[str, Any]) -> dict[str, Any] | None:
    """Parse the arguments JSON from a tool call. Returns None on failure."""
    fn = (tc or {}).get("function") or {}
    args_str = fn.get("arguments") or ""
    if not isinstance(args_str, str):
        return None
    try:
        parsed = json.loads(args_str)
    except (json.JSONDecodeError, TypeError):
        return None
    if not isinstance(parsed, dict):
        return None
    return parsed


def _translate_sre_command(args: dict[str, Any]) -> tuple[str | None, str | None]:
    """Translate a single str_replace_editor sub-command into a bash command.

    Returns (bash_command_or_None, drop_reason_or_None). If drop_reason is set, the
    trajectory should be dropped and that reason counted.
    """
    command = args.get("command", "") or ""
    path = str(args.get("path", ""))

    if command == "view":
        view_range = args.get("view_range")
        if isinstance(view_range, list) and len(view_range) == 2:
            try:
                a, b = int(view_range[0]), int(view_range[1])
                cmd = f"sed -n '{a},{b}p' {path}"
            except (TypeError, ValueError):
                cmd = f"cat -n {path}"
        else:
            cmd = f"cat -n {path}"
        return cmd, None

    if command == "create":
        file_text = args.get("file_text") or ""
        if not isinstance(file_text, str):
            file_text = str(file_text)
        # Shell heredoc with single-quoted delimiter is safe for any content.
        cmd = f"cat > {path} <<'EOF'\n{file_text}\nEOF"
        return cmd, None

    if command == "str_replace":
        old_str = args.get("old_str") or ""
        new_str = args.get("new_str") or ""
        if not isinstance(old_str, str):
            old_str = str(old_str)
        if not isinstance(new_str, str):
            new_str = str(new_str)

        # Triple-quoted safe embedding: drop trajectory if the original string
        # contains ''' which would break our python3 heredoc literal.
        if "'''" in old_str or "'''" in new_str:
            return None, "triple_quote_embed"

        cmd = (
            f"python3 - <<'PYEOF'\n"
            f"import pathlib\n"
            f"p=pathlib.Path('{path}')\n"
            f"s=p.read_text()\n"
            f"s=s.replace('''{old_str}''', '''{new_str}''', 1)\n"
            f"p.write_text(s)\n"
            f"PYEOF"
        )
        return cmd, None

    if command == "insert":
        insert_line = args.get("insert_line") or 0
        try:
            insert_line = int(insert_line)
        except (TypeError, ValueError):
            insert_line = 0
        file_text = args.get("file_text") or ""
        if not isinstance(file_text, str):
            file_text = str(file_text)

        # Triple-quote safety check for the inserted content.
        if "'''" in file_text:
            return None, "triple_quote_embed"

        cmd = (
            f"python3 - <<'PYEOF'\n"
            f"import pathlib\n"
            f"p=pathlib.Path('{path}')\n"
            f"s=p.read_text()\n"
            f"lines=s.split('\\n')\n"
            f"lines.insert({insert_line}, '''{file_text}''')\n"
            f"p.write_text('\\n'.join(lines))\n"
            f"PYEOF"
        )
        return cmd, None

    if command == "undo_edit":
        # Per spec: drop trajectory for undo_edit.
        return None, "undo_edit"

    # Unknown sub-command - caller should treat as forbidden tool.
    return None, "unknown_command"


def _extract_think_text(tc: dict[str, Any]) -> str | None:
    """Extract thought text from an openhands 'think' tool call."""
    args = _parse_tc_args(tc)
    if args is None:
        return None
    thought = args.get("thought") or ""
    if isinstance(thought, str) and thought.strip():
        return thought
    return None


def _is_forbidden_tool(tc: dict[str, Any]) -> str | None:
    """Return the tool name if this is a forbidden tool (trajectory must be dropped), else None."""
    fn = (tc or {}).get("function") or {}
    name = fn.get("name", "")
    if name in _FORBIDDEN_TOOL_NAMES:
        return name
    # bash, execute_bash, str_replace_editor, think, finish, submit are all handled during conversion.
    return None


def convert_openswe(
    source: Path,
    *,
    languages: tuple[str, ...] = DEFAULT_OPENSWP_LANGUAGES,
    limit: int | None = None,
) -> tuple[list[dict[str, Any]], Counter[str]]:
    """Convert Open-SWE-Traces parquet to target schema."""
    stats = Counter()

    raw_rows = _load_parquet_rows(source)
    if limit is not None:
        raw_rows = raw_rows[:limit]

    output: list[dict[str, Any]] = []
    rows_in = 0

    for row in raw_rows:
        rows_in += 1
        resolved = str(row.get("resolved", ""))
        language = (row.get("language") or "").lower().strip()

        if resolved != "1":
            stats["dropped_not_resolved"] += 1
            continue

        if language not in languages:
            stats["dropped_language"] += 1
            continue

        trajectory = row.get("trajectory") or []
        instance_id = str(row.get("instance_id", f"openswe_{rows_in}"))

        if not isinstance(trajectory, list):
            stats["malformed_rows"] += 1
            continue

        # Validation phase: drop only on truly forbidden tools (browser etc.).
        # str_replace_editor and think are translated during conversion.
        drop_reason: str | None = None
        for msg in trajectory:
            if not isinstance(msg, dict) or msg.get("role") != "assistant":
                continue
            tcs = msg.get("tool_calls")
            if not isinstance(tcs, list):
                continue
            for tc in tcs:
                forbidden_name = _is_forbidden_tool(tc)
                if forbidden_name is not None:
                    drop_reason = f"dropped_tool:{forbidden_name}"
                    break
            if drop_reason is not None:
                break

        if drop_reason is not None:
            stats[drop_reason] += 1
            continue

        # Conversion phase: translate each tool call individually.
        converted_msgs: list[dict[str, Any]] = []
        seen_user = False
        row_dropped: str | None = None
        row_stats = Counter()

        for msg in trajectory:
            if not isinstance(msg, dict):
                continue

            role = msg.get("role", "user")
            content = msg.get("content") or ""
            reasoning_content = msg.get("reasoning_content") or ""
            think_field = msg.get("think") or ""

            if role == "assistant":
                # Build plain-text prefix from reasoning/think fields.
                prefix_parts: list[str] = []
                if isinstance(reasoning_content, str) and reasoning_content.strip():
                    prefix_parts.append(reasoning_content.strip())
                elif isinstance(think_field, str) and think_field.strip():
                    prefix_parts.append(think_field.strip())

                tcs_raw = msg.get("tool_calls") or []
                if not isinstance(tcs_raw, list):
                    tcs_raw = []

                assistant_tool_calls: list[dict[str, Any]] = []
                has_any_real_tool_call = False

                for tc in tcs_raw:
                    fn = (tc or {}).get("function") or {}
                    tool_name = fn.get("name", "")

                    # --- bash / execute_bash -> translated to bash tool_call ---
                    if tool_name in ("bash", "execute_bash"):
                        cmd = _extract_bash_command_from_tool(tc)
                        if cmd is not None:
                            call_id = _next_call_id()
                            assistant_tool_calls.append(
                                {
                                    "id": call_id,
                                    "type": "function",
                                    "function": {"name": "bash", "arguments": json.dumps({"command": cmd})},
                                }
                            )
                            has_any_real_tool_call = True

                    # --- str_replace_editor -> translated to bash tool_call (or drop) ---
                    elif tool_name == "str_replace_editor":
                        sre_args = _parse_tc_args(tc) or {}
                        sub_cmd = sre_args.get("command", "") or ""
                        if sub_cmd == "undo_edit":
                            row_dropped = "dropped_undo_edit"
                            break
                        translated, reason = _translate_sre_command(sre_args)
                        if reason is not None:
                            row_dropped = f"dropped_{reason}"
                            break
                        if translated is not None:
                            call_id = _next_call_id()
                            assistant_tool_calls.append(
                                {
                                    "id": call_id,
                                    "type": "function",
                                    "function": {"name": "bash", "arguments": json.dumps({"command": translated})},
                                }
                            )
                            has_any_real_tool_call = True
                        row_stats["translated_str_replace_editor"] += 1

                    # --- think -> fold thought into plain text, no tool_call emitted ---
                    elif tool_name == "think":
                        thought_text = _extract_think_text(tc)
                        if thought_text is not None:
                            prefix_parts.append(thought_text)
                            row_stats["translated_think"] += 1

                    # --- finish / submit -> converted to plain content ---
                    elif tool_name in ("finish", "submit"):
                        args_parsed = _parse_tc_args(tc) or {}
                        msg_arg = ""
                        if isinstance(args_parsed, dict):
                            msg_arg = args_parsed.get("message") or args_parsed.get("output") or ""
                        if isinstance(msg_arg, str) and msg_arg.strip():
                            prefix_parts.append(f"[Tool: {tool_name}]\n{msg_arg}")

                    # --- anything else should not appear (already validated) but handle gracefully ---
                    else:
                        pass

                if row_dropped is not None:
                    break

                # Assemble plain content: always include original content, prepend any
                # reasoning/think/finish-text as a prefix separated by blank lines.
                all_text_parts = list(prefix_parts)
                if content.strip():
                    all_text_parts.append(content)
                plain_content = "\n\n".join(all_text_parts) if all_text_parts else content

                entry: dict[str, Any] = {
                    "role": "assistant",
                    "content": plain_content,
                    "tool_calls": assistant_tool_calls if assistant_tool_calls else [],
                }
                if assistant_tool_calls:
                    row_stats["tool_calls_extracted"] += 1
                seen_user = True
                converted_msgs.append(entry)

            elif role == "user" or role == "system" or role == "tool":
                if role == "tool":
                    obs_content = f"OBSERVATION:\n{content}" if not content.startswith("OBSERVATION:") else content
                    converted_msgs.append({"role": "user", "content": obs_content, "tool_calls": []})
                    row_stats["observations_prefixed"] += 1
                elif role == "system":
                    converted_msgs.append({"role": "system", "content": content, "tool_calls": []})
                else:
                    if not seen_user:
                        converted_msgs.append({"role": "user", "content": content, "tool_calls": []})
                        seen_user = True
                    else:
                        obs_content = f"OBSERVATION:\n{content}" if not content.startswith("OBSERVATION:") else content
                        converted_msgs.append({"role": "user", "content": obs_content, "tool_calls": []})
                        row_stats["observations_prefixed"] += 1

        if row_dropped is not None:
            stats[row_dropped] += 1
            continue

        stats.update(row_stats)

        output.append(
            {
                "instance_id": instance_id,
                "source": "open_swe_traces_qwen35",
                "messages": converted_msgs,
            }
        )
        stats["rows_written"] += 1
        stats[f"lang_written:{language}"] += 1

    stats["rows_read"] += rows_in
    return output, stats


def _extract_bash_command_from_tool(tool_call: dict[str, Any]) -> str | None:
    """Extract command string from a bash or execute_bash tool call."""
    fn = (tool_call or {}).get("function") or {}
    name = fn.get("name", "")

    if name == "bash":
        args_str = fn.get("arguments") or ""
        try:
            args = json.loads(args_str) if isinstance(args_str, str) else args_str
        except (json.JSONDecodeError, TypeError):
            return None
        if isinstance(args, dict):
            return args.get("command")
        return None

    if name == "execute_bash":
        # OpenHands execute_bash: command may be in 'input' or 'arguments.command'
        args_str = fn.get("arguments") or ""
        try:
            args = json.loads(args_str) if isinstance(args_str, str) else (args_str or {})
        except (json.JSONDecodeError, TypeError):
            args = {}
        if isinstance(args, dict):
            return args.get("command") or args.get("input")
        return None

    return None


def write_jsonl(
    rows: list[dict[str, Any]],
    output: Path,
    *,
    manifest_path: Path | None = None,
    source_label: str = "unknown",
    stats: dict[str, int] | None = None,
) -> dict[str, Any]:
    """Write converted rows to JSONL and return a manifest."""
    output.parent.mkdir(parents=True, exist_ok=True)

    with output.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")

    manifest = {
        "source": source_label,
        "output": str(output),
        "rows_written": len(rows),
        "stats": dict(stats or {}),
    }

    if manifest_path is not None:
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )

    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--format", required=True, choices=["kwai", "openswe"], help="Source dataset format.")
    parser.add_argument("--in", dest="source", required=True, type=Path, help="Input parquet file or glob pattern.")
    parser.add_argument("--out", dest="output", required=True, type=Path, help="Output JSONL file.")
    parser.add_argument("--manifest", type=Path, default=None, help="Optional manifest JSON path.")
    parser.add_argument(
        "--languages",
        type=str,
        default=",".join(DEFAULT_OPENSWP_LANGUAGES),
        help="Comma-separated list of allowed languages for openswe (default: python,rust,cpp,c,c++).",
    )
    parser.add_argument("--limit", type=int, default=None, help="Optional row limit.")
    args = parser.parse_args()

    _reset_call_ids()

    if args.format == "kwai":
        rows, stats = convert_kwai(args.source, limit=args.limit)
        source_label = "kwai_klear_miniswe"
    else:
        lang_list = tuple(l.strip().lower() for l in args.languages.split(",") if l.strip())
        rows, stats = convert_openswe(args.source, languages=lang_list, limit=args.limit)
        source_label = "open_swe_traces_qwen35"

    manifest = write_jsonl(
        rows, args.output, manifest_path=args.manifest, source_label=source_label, stats=dict(stats)
    )
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
