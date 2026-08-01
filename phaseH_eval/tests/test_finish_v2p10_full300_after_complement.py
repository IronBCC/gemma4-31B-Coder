from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = (
    ROOT / "phaseH_eval" / "finish_v2p10_full300_after_complement.sh"
)


def test_v2p10_finish_binds_official_scores_before_completion() -> None:
    text = SCRIPT.read_text()

    assert "full300_official_score_binding.py" in text
    assert "v2p10_full300_official_score_binding.json" in text
    assert text.index("full300_panel_composite.py") < text.index(
        "full300_official_score_binding.py"
    )
    assert '"$SCORE_BINDING"' in text


def test_v2p10_finish_is_valid_bash() -> None:
    result = subprocess.run(
        ["bash", "-n", str(SCRIPT)],
        text=True,
        capture_output=True,
    )
    assert result.returncode == 0, result.stderr
