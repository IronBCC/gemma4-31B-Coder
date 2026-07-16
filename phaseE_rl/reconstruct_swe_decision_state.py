#!/usr/bin/env python3
"""Create bounded Docker state-cache images for RLVR decision prefixes.

Only fixture-backed rows can enter the cache. Prefixes are expected to end
before the first source edit; the replay guard therefore permits a deliberately
small read-only shell subset. Every container is isolated from the network and
removed by its exact id. Cache eviction only ever deletes images marked with
the ``rlvr.state=1`` label.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import shlex
import subprocess
import time
from pathlib import Path
from typing import Any, Iterable


READ_COMMANDS = {
    "awk", "basename", "cat", "cd", "dirname", "echo", "file", "find", "git", "grep",
    "head", "ls", "pwd", "rg", "sed", "stat", "tail", "tree", "wc", "which",
}
GIT_READ_SUBCOMMANDS = {"branch", "diff", "grep", "log", "ls-files", "show", "status"}
_UNSAFE_SHELL = re.compile(r"(?:`|\$\(|[<>]|\b(?:curl|wget|apt|apt-get|pip|conda|npm|yarn)\b|\b(?:rm|mv|cp|touch|mkdir|tee|chmod|chown)\b|\bsed\s+-[^\n;|]*i\b|\b(?:git\s+(?:apply|checkout|clean|commit|reset|restore))\b)")


def prompt_hash(row: dict[str, Any]) -> str:
    payload = json.dumps(row.get("messages", []), ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()


def extract_bash_commands(row: dict[str, Any]) -> list[str]:
    commands: list[str] = []
    for message in row.get("messages", []):
        if message.get("role") != "assistant":
            continue
        for call in message.get("tool_calls") or []:
            function = call.get("function") or {}
            if function.get("name") != "bash":
                continue
            try:
                arguments = json.loads(function.get("arguments") or "{}")
            except json.JSONDecodeError:
                continue
            command = arguments.get("command")
            if isinstance(command, str) and command.strip():
                commands.append(command.strip())
    return commands


def is_replay_safe(command: str) -> bool:
    """Allow only read-only prefix commands; reject shell writes and network setup."""
    if _UNSAFE_SHELL.search(command):
        return False
    for segment in re.split(r"&&|\|\||;|\|", command):
        segment = segment.strip()
        if not segment:
            continue
        try:
            words = shlex.split(segment)
        except ValueError:
            return False
        if not words:
            continue
        executable = words[0]
        if executable not in READ_COMMANDS:
            return False
        if executable == "find" and "-exec" in words:
            return False
        if executable == "git":
            subcommand = next((word for word in words[1:] if not word.startswith("-")), "")
            if subcommand not in GIT_READ_SUBCOMMANDS:
                return False
    return True


def replay_plan(decision: dict[str, Any], fixture: dict[str, Any]) -> dict[str, Any] | None:
    if not fixture.get("verifiable") or fixture.get("instance_id") != decision.get("instance_id"):
        return None
    commands = extract_bash_commands(decision)
    return {
        "instance_id": str(decision["instance_id"]),
        "source": str(decision["source"]),
        "prompt_hash": prompt_hash(decision),
        "image_name": str(fixture["image_name"]),
        "f2p": list(fixture.get("f2p") or []),
        "commands": commands,
        "replay_safe": all(is_replay_safe(command) for command in commands),
    }


def select_plans(
    decisions: Iterable[dict[str, Any]],
    fixtures: dict[tuple[str, str], dict[str, Any]],
    *,
    allowed_images: set[str] | None = None,
    include_local: bool = True,
) -> tuple[list[dict[str, Any]], int]:
    """Select local fixtures plus an optional explicit pulled-image window."""
    plans: list[dict[str, Any]] = []
    skipped_unsafe = 0
    for decision in decisions:
        fixture = fixtures.get((decision["source"], decision["instance_id"]))
        if fixture is None:
            continue
        image_name = fixture.get("image_name")
        selected = (include_local and bool(fixture.get("image_local"))) or (
            allowed_images is not None and image_name in allowed_images
        )
        if not selected:
            continue
        plan = replay_plan(decision, fixture)
        if plan is None:
            continue
        if plan["replay_safe"]:
            plans.append(plan)
        else:
            skipped_unsafe += 1
    return plans, skipped_unsafe


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def run(cmd: list[str], *, timeout: int = 120) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, text=True, capture_output=True, timeout=timeout, check=False)


def docker_available_bytes() -> int:
    result = run(["df", "-PB1", "/var/lib/docker"], timeout=30)
    if result.returncode:
        raise RuntimeError(result.stderr.strip() or "df /var/lib/docker failed")
    return int(result.stdout.splitlines()[1].split()[3])


def image_exists(name: str) -> bool:
    return run(["docker", "image", "inspect", name], timeout=30).returncode == 0


def state_tag(prompt_digest: str) -> str:
    return f"rlvr-state:{prompt_digest}"


def _load_index(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _save_index(path: Path, index: dict[str, dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(index, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def evict_lru(index_path: Path, *, max_cache_bytes: int, floor_bytes: int) -> list[str]:
    """Evict only labelled RLVR state images until cache budget and disk floor hold."""
    index = _load_index(index_path)
    records: list[tuple[str, int, float]] = []
    for tag, meta in list(index.items()):
        inspected = run(["docker", "image", "inspect", "--format", "{{.Size}}", tag], timeout=30)
        if inspected.returncode:
            index.pop(tag, None)
            continue
        records.append((tag, int(inspected.stdout.strip()), float(meta.get("last_used", 0))))
    total = sum(size for _, size, _ in records)
    evicted: list[str] = []
    for tag, size, _ in sorted(records, key=lambda item: item[2]):
        if total <= max_cache_bytes and docker_available_bytes() >= floor_bytes:
            break
        removed = run(["docker", "image", "rm", tag], timeout=90)
        if removed.returncode:
            raise RuntimeError(f"failed LRU eviction {tag}: {removed.stderr.strip()}")
        total -= size
        index.pop(tag, None)
        evicted.append(tag)
    _save_index(index_path, index)
    if docker_available_bytes() < floor_bytes:
        raise RuntimeError("Docker loopback is below the required floor after RLVR-only LRU eviction")
    return evicted


def reconstruct(plan: dict[str, Any], *, index_path: Path, floor_bytes: int, max_cache_bytes: int) -> dict[str, Any]:
    tag = state_tag(plan["prompt_hash"])
    index = _load_index(index_path)
    if image_exists(tag):
        index.setdefault(tag, {})["last_used"] = time.time()
        _save_index(index_path, index)
        return {**plan, "state_tag": tag, "cache_hit": True, "status": "cached"}
    if docker_available_bytes() < floor_bytes:
        raise RuntimeError("Docker loopback is below the required floor before reconstruction")
    created = run(
        ["docker", "run", "-d", "--network", "none", "--pids-limit", "512", plan["image_name"], "sleep", "infinity"],
        timeout=180,
    )
    if created.returncode:
        return {**plan, "state_tag": tag, "cache_hit": False, "status": "container_start_failed", "detail": created.stderr[-500:]}
    cid = created.stdout.strip()
    command_results: list[dict[str, Any]] = []
    try:
        for command in plan["commands"]:
            result = run(["docker", "exec", cid, "bash", "-c", f"cd /testbed && {command}"], timeout=45)
            command_results.append({"command": command, "returncode": result.returncode})
        if docker_available_bytes() < floor_bytes:
            raise RuntimeError("Docker loopback fell below floor before cache commit")
        committed = run(
            ["docker", "commit", "--change", "LABEL rlvr.state=1", "--change", f"LABEL rlvr.prompt_hash={plan['prompt_hash']}", cid, tag],
            timeout=180,
        )
        if committed.returncode:
            return {**plan, "state_tag": tag, "cache_hit": False, "status": "commit_failed", "detail": committed.stderr[-500:], "commands": command_results}
    finally:
        run(["docker", "rm", "-f", cid], timeout=60)
    index[tag] = {"last_used": time.time(), "prompt_hash": plan["prompt_hash"], "image_name": plan["image_name"]}
    _save_index(index_path, index)
    evicted = evict_lru(index_path, max_cache_bytes=max_cache_bytes, floor_bytes=floor_bytes)
    return {**plan, "state_tag": tag, "cache_hit": False, "status": "created", "commands": command_results, "evicted": evicted}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--decisions", type=Path, required=True)
    parser.add_argument("--fixtures", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--cache-index", type=Path, default=Path("runs/rlvr_state_cache/index.json"))
    parser.add_argument(
        "--allowed-images-file",
        type=Path,
        help="newline-delimited pulled images; local fixture images are always included",
    )
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--local-only", action="store_true")
    parser.add_argument(
        "--no-local-images",
        action="store_true",
        help="with --allowed-images-file, process only newly pulled fixture images",
    )
    parser.add_argument("--floor-gib", type=int, default=40)
    parser.add_argument("--max-cache-gib", type=int, default=20)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.limit < 0 or args.floor_gib <= 0 or args.max_cache_gib <= 0:
        raise SystemExit("limits must be positive")
    fixtures = {(row["source"], row["instance_id"]): row for row in load_jsonl(args.fixtures)}
    if args.local_only and args.allowed_images_file:
        raise SystemExit("--local-only and --allowed-images-file are mutually exclusive")
    if args.local_only and args.no_local_images:
        raise SystemExit("--local-only and --no-local-images are mutually exclusive")
    allowed_images = None
    if args.allowed_images_file:
        allowed_images = {
            line.strip() for line in args.allowed_images_file.read_text(encoding="utf-8").splitlines()
            if line.strip()
        }
    if args.local_only:
        allowed_images = set()
    plans, skipped_unsafe = select_plans(
        load_jsonl(args.decisions), fixtures, allowed_images=allowed_images,
        include_local=not args.no_local_images,
    )
    if args.limit:
        plans = plans[:args.limit]
    floor_bytes = args.floor_gib * 1024**3
    max_cache_bytes = args.max_cache_gib * 1024**3
    evict_lru(args.cache_index, max_cache_bytes=max_cache_bytes, floor_bytes=floor_bytes)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as handle:
        for plan in plans:
            result = reconstruct(plan, index_path=args.cache_index, floor_bytes=floor_bytes, max_cache_bytes=max_cache_bytes)
            handle.write(json.dumps(result, sort_keys=True) + "\n")
            handle.flush()
            print(json.dumps({key: result.get(key) for key in ("instance_id", "source", "status", "cache_hit", "state_tag")}), flush=True)
    print(json.dumps({"selected_plans": len(plans), "replay_unsafe_skipped": skipped_unsafe}), flush=True)


if __name__ == "__main__":
    main()
