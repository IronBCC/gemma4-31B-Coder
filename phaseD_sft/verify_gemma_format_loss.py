#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import tempfile
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from phaseD_sft.progress import EtaProgress
from phaseD_sft.train_rust_lora import (
    label_tokens_for_spans,
    load_training_dataset,
    rendered_assistant_turn_spans,
    supervised_assistant_turn_spans,
)


def content_text(message: dict) -> str:
    content = message.get("content") or ""
    if isinstance(content, str):
        return content
    return json.dumps(content, ensure_ascii=False, sort_keys=True)


def is_subspan(inner: tuple[int, int], spans: list[tuple[int, int]]) -> bool:
    return any(start <= inner[0] and inner[1] <= end for start, end in spans)


def ordered_parallel_map(items, function, *, workers: int):
    """Apply ``function`` concurrently while yielding results in input order."""
    if workers == 1:
        yield from map(function, items)
        return
    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="format-loss") as executor:
        yield from executor.map(function, items)


def write_report_atomic(path: Path, report: dict) -> None:
    """Publish a machine-readable verifier report without partial JSON."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2, ensure_ascii=False, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def validate_one(tokenizer, idx: int, messages: list[dict]):
    """Render, label, and validate one sample using an already-loaded tokenizer."""
    counts = Counter()
    failures: list[str] = []
    thinking_examples: list[dict] = []
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    spans, fallback_used = rendered_assistant_turn_spans(tokenizer, messages, text)
    supervised_spans, supervised_fallback_used = supervised_assistant_turn_spans(
        tokenizer, messages, text
    )

    counts["examples"] += 1
    counts["assistant_messages"] += sum(1 for msg in messages if msg.get("role") == "assistant")
    counts["assistant_messages_masked_by_loss_flag"] += sum(
        1
        for msg in messages
        if msg.get("role") == "assistant" and msg.get("loss", True) is False
    )
    counts["assistant_messages_supervised"] += sum(
        1
        for msg in messages
        if msg.get("role") == "assistant" and msg.get("loss", True) is not False
    )
    counts["fallback_spans"] += int(fallback_used)
    counts["supervised_fallback_spans"] += int(supervised_fallback_used)
    counts["gemma_turns"] += text.count("<|turn>")
    counts["assistant_turn_markers"] += sum(text[start:end].startswith("<|turn>model\n") for start, end in spans)
    counts["tool_call_open"] += sum("<|tool_call>" in text[start:end] for start, end in spans)
    counts["tool_call_close"] += sum("<tool_call|>" in text[start:end] for start, end in spans)
    counts["tool_response"] += sum("<|tool_response>" in text[start:end] for start, end in spans)

    if len(spans) != sum(1 for msg in messages if msg.get("role") == "assistant"):
        failures.append(f"{idx}: assistant span count mismatch")

    if not text.startswith("<bos><|turn>") and not text.startswith("<|begin_of_text|><|turn>"):
        failures.append(f"{idx}: rendered text does not start with Gemma turn format")

    for i in range(1, len(messages) + 1):
        prefix = tokenizer.apply_chat_template(messages[:i], tokenize=False, add_generation_prompt=False)
        if not text.startswith(prefix):
            failures.append(f"{idx}: render prefix not stable at message {i}")
            break

    cursor = 0
    for msg_i, msg in enumerate(messages):
        content = content_text(msg)
        pos = text.find(content, cursor) if content else cursor
        if pos >= 0:
            content_span = (pos, pos + len(content))
            if msg.get("role") == "assistant" and content and not is_subspan(content_span, spans):
                failures.append(f"{idx}: assistant content outside supervised turn at message {msg_i}")
            if msg.get("role") != "assistant" and content and is_subspan(content_span, spans):
                failures.append(f"{idx}: non-assistant content inside supervised turn at message {msg_i}")
            cursor = content_span[1]

        lowered = content.lower()
        if any(word in lowered for word in ("thinking", "thought", "reasoning", "analysis")):
            counts[f"thinking_role:{msg.get('role')}"] += 1
            thinking_examples.append({
                "idx": int(idx),
                "role": msg.get("role"),
                "in_supervised_assistant_turn": bool(
                    content and pos >= 0 and is_subspan((pos, pos + len(content)), supervised_spans)
                ),
                "preview": content[:240].replace("\n", "\\n"),
            })

    enc = tokenizer(
        text=text,
        add_special_tokens=False,
        return_offsets_mapping=True,
        truncation=True,
    )
    input_ids = enc["input_ids"]
    offsets = enc["offset_mapping"]
    if input_ids and isinstance(input_ids[0], list):
        input_ids = input_ids[0]
    if offsets and isinstance(offsets[0], list) and offsets[0] and isinstance(offsets[0][0], (list, tuple)):
        offsets = offsets[0]
    labels, supervised = label_tokens_for_spans(input_ids, offsets, supervised_spans)
    if supervised <= 0:
        failures.append(f"{idx}: no supervised tokens")
    decoded = tokenizer.decode([tok for tok, lab in zip(input_ids, labels) if lab != -100])
    if "<|turn>model" not in decoded:
        failures.append(f"{idx}: decoded labels missing model turn marker")

    return counts, supervised, failures, thinking_examples


def main() -> int:
    parser = argparse.ArgumentParser(
        epilog="Invoke with PYTHONPATH=<repo-root> so phaseD_sft imports resolve consistently."
    )
    parser.add_argument("--base", default="/media/ironbcc/CrucialX10/models/google/gemma-4-31B-it")
    parser.add_argument("--data", default="data/unsloth_agentic_24k_train")
    parser.add_argument("--samples", type=int, default=500)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--report-out",
        type=Path,
        default=None,
        help="atomically write the final JSON report for manifest binding",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="CPU validation threads; tokenizer loads once in the main thread (default: 1).",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=10,
        help="Emit completed/total, it/s, elapsed, ETA, and total estimate every N samples.",
    )
    args = parser.parse_args()
    if args.progress_every <= 0:
        parser.error("--progress-every must be positive")
    if args.workers <= 0:
        parser.error("--workers must be positive")

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.base)
    print("[template] using base tokenizer Gemma chat template", flush=True)
    print(f"[workers] {args.workers} (one shared tokenizer; ThreadPoolExecutor)", flush=True)

    ds = load_training_dataset(args.data)
    rng = random.Random(args.seed)
    indices = list(range(len(ds)))
    rng.shuffle(indices)
    indices = indices[: min(args.samples, len(indices))]
    progress = EtaProgress("format-loss", total=len(indices))

    counts = Counter()
    supervised_tokens = []
    failures: list[str] = []
    thinking_examples: list[dict] = []

    samples = ((int(idx), ds[int(idx)]["messages"]) for idx in indices)
    for completed, result in enumerate(
        ordered_parallel_map(
            samples,
            lambda sample: validate_one(tokenizer, *sample),
            workers=args.workers,
        ),
        start=1,
    ):
        sample_counts, supervised, sample_failures, sample_thinking_examples = result
        counts.update(sample_counts)
        supervised_tokens.append(supervised)
        failures.extend(sample_failures)
        for example in sample_thinking_examples:
            if len(thinking_examples) < 8:
                thinking_examples.append(example)

        if completed % args.progress_every == 0 or completed == len(indices):
            print(progress.update(completed), flush=True)

    supervised_tokens.sort()
    def pct(p: float) -> int:
        if not supervised_tokens:
            return 0
        return supervised_tokens[min(len(supervised_tokens) - 1, int(round((len(supervised_tokens) - 1) * p)))]

    report = {
        "data": args.data,
        "samples": len(indices),
        "counts": dict(counts),
        "supervised_tokens": {
            "min": supervised_tokens[0] if supervised_tokens else 0,
            "p50": pct(0.50),
            "p90": pct(0.90),
            "max": supervised_tokens[-1] if supervised_tokens else 0,
        },
        "thinking_examples": thinking_examples,
        "failures": failures[:20],
        "failure_count": len(failures),
    }
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if args.report_out is not None:
        write_report_atomic(args.report_out, report)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
