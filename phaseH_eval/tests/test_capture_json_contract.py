from __future__ import annotations

import importlib
import json
from pathlib import Path
import sys

import pytest


def test_captures_and_revalidates_one_json_contract(
    tmp_path: Path,
) -> None:
    module = importlib.import_module(
        "phaseH_eval.capture_json_contract"
    )
    output = tmp_path / "contract.json"
    command = [
        sys.executable,
        "-c",
        "import json; print(json.dumps({'status': 'complete', 'rows': 36}))",
    ]

    report = module.capture_json_contract(
        output_path=output,
        command=command,
    )

    assert report == {"status": "complete", "rows": 36}
    assert json.loads(output.read_text()) == report
    assert module.capture_json_contract(
        output_path=output,
        command=command,
    ) == report
    with pytest.raises(ValueError, match="differs from current command"):
        module.capture_json_contract(
            output_path=output,
            command=[
                sys.executable,
                "-c",
                "print('{\"status\":\"complete\",\"rows\":35}')",
            ],
        )
