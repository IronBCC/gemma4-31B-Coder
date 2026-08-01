from __future__ import annotations

import importlib
import hashlib
from pathlib import Path

import pytest


def test_selfretry_config_uses_docker_alias_for_swebench_image_injection(
    tmp_path: Path,
) -> None:
    module = importlib.import_module("phaseH_eval.mini_extra_selfretry")
    config = tmp_path / "selfretry.yaml"
    config.write_text(
        "environment:\n"
        "  environment_class: docker_selfretry.DockerSelfRetryEnv\n"
    )

    args, use_selfretry = module.prepare_swebench_args(
        ["-c", str(config), "-c", "model.model_kwargs.temperature=0.7"]
    )

    assert use_selfretry is True
    assert args[-2:] == ["--environment-class", "docker"]
    assert args[:2] == ["-c", str(config)]


def test_plain_docker_config_is_not_remapped(tmp_path: Path) -> None:
    module = importlib.import_module("phaseH_eval.mini_extra_selfretry")
    config = tmp_path / "plain.yaml"
    config.write_text("environment:\n  environment_class: docker\n")
    original = ["-c", str(config), "--workers", "16"]

    args, use_selfretry = module.prepare_swebench_args(original)

    assert use_selfretry is False
    assert args == original


def test_selfretry_launcher_rejects_an_explicit_environment_override(
    tmp_path: Path,
) -> None:
    module = importlib.import_module("phaseH_eval.mini_extra_selfretry")
    config = tmp_path / "selfretry.yaml"
    config.write_text(
        "environment:\n"
        "  environment_class: docker_selfretry.DockerSelfRetryEnv\n"
    )

    with pytest.raises(ValueError, match="environment-class"):
        module.prepare_swebench_args(
            [
                "-c",
                str(config),
                "--environment-class",
                "singularity",
            ]
        )


def test_alias_installation_is_explicit_and_narrow() -> None:
    module = importlib.import_module("phaseH_eval.mini_extra_selfretry")
    mapping = {"docker": "original", "local": "local"}

    module.install_selfretry_alias(mapping)

    assert mapping == {
        "docker": "docker_selfretry.DockerSelfRetryEnv",
        "local": "local",
    }


def test_local_jsonl_subset_is_loaded_as_an_exact_json_dataset(
    tmp_path: Path,
) -> None:
    module = importlib.import_module("phaseH_eval.mini_extra_selfretry")
    tasks = tmp_path / "hard30.jsonl"
    tasks.write_text('{"instance_id":"one"}\n')
    calls = []

    def original(path, *args, **kwargs):
        calls.append((path, args, kwargs))
        return "dataset"

    loader = module.local_jsonl_loader(original, tasks)

    assert loader(str(tasks), split="train") == "dataset"
    assert calls == [(
        "json",
        (),
        {"data_files": str(tasks.resolve()), "split": "train"},
    )]
    assert loader("SWE-bench/SWE-smith", split="train") == "dataset"
    assert calls[-1][0] == "SWE-bench/SWE-smith"


def test_mutation_layer_is_established_before_environment_is_returned() -> None:
    module = importlib.import_module("phaseH_eval.mini_extra_selfretry")
    mutation = (
        "diff --git a/src/x.py b/src/x.py\n"
        "--- a/src/x.py\n"
        "+++ b/src/x.py\n"
    )
    calls = []

    class Env:
        def establish_task_baseline(self, instance_id, patch):
            calls.append((instance_id, patch))
            return {"mutation_patch_sha256": hashlib.sha256(patch.encode()).hexdigest()}

        def cleanup(self):
            calls.append("cleanup")

    env = Env()
    wrapped = module.mutation_layered_environment(lambda _config, _row: env)
    row = {
        "instance_id": "fixture__repo.case",
        "task_patch_role": "bug_inducing_mutation",
        "patch": mutation,
        "mutation_patch_sha256": hashlib.sha256(mutation.encode()).hexdigest(),
    }

    assert wrapped(object(), row) is env
    assert calls == [("fixture__repo.case", mutation)]


def test_mutation_layer_leaves_ordinary_swebench_instance_unchanged() -> None:
    module = importlib.import_module("phaseH_eval.mini_extra_selfretry")
    env = object()
    wrapped = module.mutation_layered_environment(lambda _config, _row: env)

    assert wrapped(object(), {"instance_id": "plain"}) is env


def test_mutation_layer_cleans_environment_and_fails_closed() -> None:
    module = importlib.import_module("phaseH_eval.mini_extra_selfretry")
    calls = []

    class Env:
        def establish_task_baseline(self, _instance_id, _patch):
            raise RuntimeError("apply failed")

        def cleanup(self):
            calls.append("cleanup")

    wrapped = module.mutation_layered_environment(lambda _config, _row: Env())
    mutation = "diff --git a/x.py b/x.py\n"

    with pytest.raises(RuntimeError, match="apply failed"):
        wrapped(
            object(),
            {
                "instance_id": "fixture",
                "task_patch_role": "bug_inducing_mutation",
                "patch": mutation,
                "mutation_patch_sha256": hashlib.sha256(
                    mutation.encode()
                ).hexdigest(),
            },
        )

    assert calls == ["cleanup"]


def test_mutation_layer_rejects_mismatched_binding_and_cleans() -> None:
    module = importlib.import_module("phaseH_eval.mini_extra_selfretry")
    calls = []

    class Env:
        def cleanup(self):
            calls.append("cleanup")

    wrapped = module.mutation_layered_environment(lambda _config, _row: Env())

    with pytest.raises(ValueError, match="SHA-256"):
        wrapped(
            object(),
            {
                "instance_id": "fixture",
                "task_patch_role": "bug_inducing_mutation",
                "patch": "diff --git a/x.py b/x.py\n",
                "mutation_patch_sha256": "0" * 64,
            },
        )

    assert calls == ["cleanup"]
