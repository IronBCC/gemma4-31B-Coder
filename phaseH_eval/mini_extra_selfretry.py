#!/usr/bin/env python3
"""Launch Mini-SWE batch eval with per-instance images and Docker self-retry.

Mini-SWE 2.4.1 injects a SWE-bench instance's image only when the configured
environment-class string is exactly ``docker``. For a self-retry YAML, keep that
runner-visible alias while resolving the alias to our Docker subclass.
"""
from __future__ import annotations

from collections.abc import Callable, MutableMapping, Sequence
import hashlib
from pathlib import Path
import re
import sys
from typing import Any

import yaml


SELFRETRY_CLASS = "docker_selfretry.DockerSelfRetryEnv"
TASK_PATCH_ROLE = "bug_inducing_mutation"
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


def _config_specs(args: Sequence[str]) -> list[str]:
    specs: list[str] = []
    index = 0
    while index < len(args):
        arg = args[index]
        if arg in {"-c", "--config"}:
            if index + 1 >= len(args):
                raise ValueError(f"{arg} is missing its config value")
            specs.append(args[index + 1])
            index += 2
            continue
        if arg.startswith("--config="):
            specs.append(arg.split("=", 1)[1])
        index += 1
    return specs


def _environment_class_from_specs(specs: Sequence[str]) -> str | None:
    environment_class: str | None = None
    for spec in specs:
        path = Path(spec)
        if path.is_file():
            loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                environment = loaded.get("environment")
                if isinstance(environment, dict) and isinstance(
                    environment.get("environment_class"), str
                ):
                    environment_class = environment["environment_class"]
            continue
        prefix = "environment.environment_class="
        if spec.startswith(prefix):
            environment_class = spec.removeprefix(prefix)
    return environment_class


def prepare_swebench_args(args: Sequence[str]) -> tuple[list[str], bool]:
    """Return Mini-SWE args plus whether the docker alias needs remapping."""

    prepared = list(args)
    use_selfretry = (
        _environment_class_from_specs(_config_specs(prepared)) == SELFRETRY_CLASS
    )
    if not use_selfretry:
        return prepared, False
    if "--environment-class" in prepared or any(
        arg.startswith("--environment-class=") for arg in prepared
    ):
        raise ValueError(
            "self-retry config cannot be combined with an explicit "
            "--environment-class override"
        )
    prepared.extend(["--environment-class", "docker"])
    return prepared, True


def install_selfretry_alias(mapping: MutableMapping[str, Any]) -> None:
    """Resolve Mini-SWE's docker alias to the self-retrying subclass."""

    mapping["docker"] = SELFRETRY_CLASS


def local_jsonl_loader(
    original: Callable[..., Any],
    tasks_path: Path,
) -> Callable[..., Any]:
    """Wrap datasets.load_dataset so Mini-SWE can consume one frozen JSONL."""

    exact = Path(tasks_path).resolve()
    if exact.suffix != ".jsonl" or not exact.is_file():
        raise ValueError(f"local task dataset must be an existing JSONL: {exact}")

    def load(path: str, *args: Any, **kwargs: Any) -> Any:
        try:
            candidate = Path(path).resolve()
        except TypeError:
            candidate = Path()
        if candidate == exact:
            return original(
                "json",
                *args,
                data_files=str(exact),
                **kwargs,
            )
        return original(path, *args, **kwargs)

    return load


def mutation_layered_environment(
    original: Callable[..., Any],
) -> Callable[..., Any]:
    """Apply an explicitly bound SWE-smith mutation to a new environment."""

    def get_environment(config: Any, instance: Any) -> Any:
        environment = original(config, instance)
        role = instance.get("task_patch_role") if isinstance(instance, MutableMapping) else None
        if role is None:
            return environment
        try:
            if role != TASK_PATCH_ROLE:
                raise ValueError(f"unsupported task patch role: {role!r}")
            instance_id = instance.get("instance_id")
            mutation_patch = instance.get("patch")
            expected_sha256 = instance.get("mutation_patch_sha256")
            if not isinstance(instance_id, str) or not instance_id:
                raise ValueError("mutation-layered task is missing instance_id")
            if not isinstance(mutation_patch, str) or not mutation_patch.strip():
                raise ValueError("mutation-layered task is missing patch")
            if (
                not isinstance(expected_sha256, str)
                or not _SHA256_RE.fullmatch(expected_sha256)
                or hashlib.sha256(mutation_patch.encode("utf-8")).hexdigest()
                != expected_sha256
            ):
                raise ValueError("mutation patch SHA-256 binding mismatch")
            establish = getattr(environment, "establish_task_baseline", None)
            if not callable(establish):
                raise TypeError(
                    "mutation-layered task requires DockerSelfRetryEnv"
                )
            evidence = establish(instance_id, mutation_patch)
            if (
                not isinstance(evidence, dict)
                or evidence.get("mutation_patch_sha256") != expected_sha256
            ):
                raise RuntimeError("environment returned mismatched mutation evidence")
            return environment
        except BaseException:
            cleanup = getattr(environment, "cleanup", None)
            if callable(cleanup):
                cleanup()
            raise

    return get_environment


def _subset_from_args(args: Sequence[str]) -> str | None:
    for index, arg in enumerate(args):
        if arg == "--subset" and index + 1 < len(args):
            return args[index + 1]
        if arg.startswith("--subset="):
            return arg.split("=", 1)[1]
    return None


def main() -> None:
    args = sys.argv[1:]
    if not args or args[0] != "swebench":
        raise SystemExit("mini_extra_selfretry.py only supports the swebench command")
    prepared, use_selfretry = prepare_swebench_args(args[1:])
    if use_selfretry:
        import minisweagent.environments as environments

        install_selfretry_alias(environments._ENVIRONMENT_MAPPING)
    subset = _subset_from_args(prepared)
    if subset and Path(subset).suffix == ".jsonl":
        import datasets

        datasets.load_dataset = local_jsonl_loader(
            datasets.load_dataset,
            Path(subset),
        )

    from minisweagent.run.benchmarks import swebench as swebench_module

    if use_selfretry:
        swebench_module.get_sb_environment = mutation_layered_environment(
            swebench_module.get_sb_environment
        )

    sys.argv = [sys.argv[0], *prepared]
    swebench_module.app()


if __name__ == "__main__":
    main()
