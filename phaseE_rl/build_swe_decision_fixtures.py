#!/usr/bin/env python3
"""Join RLVR decision prefixes to Docker-verifiable SWE fixtures.

The output is a sidecar, deliberately separate from the prompt dataset: reward
code can look up a row by ``instance_id`` without changing prompts or silently
claiming execution evidence for sources that have no trustworthy fixture.
"""
from __future__ import annotations

import argparse
import ast
from collections import Counter, defaultdict
import json
import subprocess
from pathlib import Path
from typing import Any, Iterable


def as_list(value: Any) -> list[str]:
    """Normalize native and serialized SWE FAIL_TO_PASS fields."""
    if isinstance(value, list):
        return [str(item) for item in value]
    if not value:
        return []
    text = str(value).strip()
    for parser in (json.loads, ast.literal_eval):
        try:
            parsed = parser(text)
        except (ValueError, SyntaxError, json.JSONDecodeError):
            continue
        if isinstance(parsed, (list, tuple)):
            return [str(item) for item in parsed]
    return [text]


def lite_image_name(instance_id: str) -> str:
    """Return SWE-bench's standard local image name for a Lite instance."""
    return f"swebench/sweb.eval.x86_64.{instance_id.replace('__', '_1776_')}:latest"


def _fixture(row: dict[str, Any], *, fixture_source: str) -> dict[str, Any]:
    instance_id = str(row["instance_id"])
    image_name = str(row.get("image_name") or "")
    if fixture_source == "swe-lite" and not image_name:
        image_name = lite_image_name(instance_id)
    return {
        "fixture_source": fixture_source,
        "image_name": image_name,
        "f2p": as_list(row.get("FAIL_TO_PASS")),
        "test_patch": str(row.get("test_patch") or ""),
    }


def build_fixture_rows(
    decisions: Iterable[dict[str, Any]],
    swe_smith: Iterable[dict[str, Any]],
    swe_lite: Iterable[dict[str, Any]],
    *,
    local_images: set[str],
) -> list[dict[str, Any]]:
    """Return one fixture record per decision, exact-ID joined only.

    SWE-smith and Kwai share the SWE-smith identifier space. Smoke rows use
    SWE-bench Lite. Oracle/OpenSWE remain deliberately unverified unless an
    explicit fixture mapping is supplied in a future data migration.
    """
    smith_index = {
        str(row["instance_id"]): _fixture(row, fixture_source="swe-smith")
        for row in swe_smith
    }
    lite_index = {
        str(row["instance_id"]): _fixture(row, fixture_source="swe-lite")
        for row in swe_lite
    }
    output: list[dict[str, Any]] = []
    for decision in decisions:
        source = str(decision["source"])
        instance_id = str(decision["instance_id"])
        fixture: dict[str, Any] | None = None
        if source.startswith("smoke:"):
            fixture = lite_index.get(instance_id)
        elif source in {"swe-smith", "kwai_klear_miniswe"}:
            fixture = smith_index.get(instance_id)

        record: dict[str, Any] = {"instance_id": instance_id, "source": source}
        if fixture is None or not fixture["image_name"] or not fixture["f2p"]:
            record.update(
                {
                    "verifiable": False,
                    "unverified": True,
                    "fixture_source": None,
                    "image_name": None,
                    "image_local": False,
                    "f2p": [],
                    "test_patch": "",
                }
            )
        else:
            record.update(fixture)
            record["verifiable"] = True
            record["unverified"] = False
            record["image_local"] = fixture["image_name"] in local_images
        output.append(record)
    return output


def summarize_coverage(rows: Iterable[dict[str, Any]]) -> dict[str, Any]:
    rows = list(rows)
    by_source: dict[str, dict[str, int]] = defaultdict(lambda: Counter())
    for row in rows:
        counters = by_source[str(row["source"])]
        counters["rows"] += 1
        counters["verifiable_rows" if row["verifiable"] else "unverified_rows"] += 1
        if row["image_local"]:
            counters["local_image_rows"] += 1
    return {
        "rows": len(rows),
        "verifiable_rows": sum(bool(row["verifiable"]) for row in rows),
        "unverified_rows": sum(not bool(row["verifiable"]) for row in rows),
        "local_image_rows": sum(bool(row["image_local"]) for row in rows),
        "by_source": {source: dict(sorted(counts.items())) for source, counts in sorted(by_source.items())},
    }


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def local_docker_images() -> set[str]:
    completed = subprocess.run(
        ["docker", "images", "--format", "{{.Repository}}:{{.Tag}}"],
        check=True,
        text=True,
        capture_output=True,
    )
    return set(completed.stdout.splitlines())


def load_hf_rows(dataset_name: str, split: str) -> list[dict[str, Any]]:
    from datasets import load_dataset

    return list(load_dataset(dataset_name, split=split))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--decisions", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--coverage-out", type=Path, required=True)
    parser.add_argument("--swe-smith", default="SWE-bench/SWE-smith")
    parser.add_argument("--swe-smith-split", default="train")
    parser.add_argument("--swe-lite", default="princeton-nlp/SWE-bench_Lite")
    parser.add_argument("--swe-lite-split", default="test")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = build_fixture_rows(
        load_jsonl(args.decisions),
        load_hf_rows(args.swe_smith, args.swe_smith_split),
        load_hf_rows(args.swe_lite, args.swe_lite_split),
        local_images=local_docker_images(),
    )
    coverage = summarize_coverage(rows)
    for path, payload in ((args.out, rows), (args.coverage_out, coverage)):
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            if path == args.out:
                for row in payload:
                    handle.write(json.dumps(row, sort_keys=True) + "\n")
            else:
                json.dump(payload, handle, indent=2, sort_keys=True)
                handle.write("\n")
    print(json.dumps(coverage, sort_keys=True))


if __name__ == "__main__":
    main()
