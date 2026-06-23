"""T0 — per-language build/test/timeout specs.

The turn CAP is language-independent (~90; measured: all Phase-0 resolves landed
by turn ~94). The per-turn TIMEOUT scales with compile cost (a slow compiler eats
seconds per turn, not more turns). warm_build runs once at container start so the
first real compile is incremental and doesn't false-timeout; clean_cmd runs at the
FINAL scoring gate so a stale incremental cache can't fake a pass.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class LangSpec:
    lang: str
    build_cmd: str            # "" for python (no separate build step)
    test_cmd_template: str    # {ids} substituted with the targeted test ids when supported
    structured_flag: str      # the flag that yields machine-parseable output
    per_turn_timeout_s: int   # raised for slow compilers (Phase-0 default 300 false-timed-out cold Rust/C++)
    warm_build: bool          # warm once before turn 1 so incremental rebuilds are fast
    clean_cmd: str            # run before the FINAL scoring build (defeats incremental-cache fake-pass)


LANG_SPECS: dict[str, LangSpec] = {
    "python": LangSpec(
        lang="python",
        build_cmd="",
        test_cmd_template="pytest -p no:cacheprovider -rA --no-header -q {ids}",
        structured_flag="-rA",
        per_turn_timeout_s=300,
        warm_build=False,
        clean_cmd="find . -name '__pycache__' -type d -prune -exec rm -rf {} +",
    ),
    "rust": LangSpec(
        lang="rust",
        build_cmd="cargo build --offline --message-format=json",
        test_cmd_template="cargo test --no-fail-fast --message-format=json {ids}",
        structured_flag="--message-format=json",
        per_turn_timeout_s=900,
        warm_build=True,
        clean_cmd="cargo clean",
    ),
    "cpp": LangSpec(
        lang="cpp",
        # build/test are per-repo (CMake vs make) — resolved from cpp_build_matrix.json at
        # runtime by the runner; these are the CMake-default templates as a safe starting point.
        build_cmd="cmake --build build -j$(nproc)",
        test_cmd_template="ctest --test-dir build --output-on-failure --output-junit {junit}",
        structured_flag="--output-junit",
        per_turn_timeout_s=1200,
        warm_build=True,
        clean_cmd="rm -rf build && cmake -S . -B build -DCMAKE_BUILD_TYPE=Debug -DCMAKE_EXPORT_COMPILE_COMMANDS=ON",
    ),
}

# Language-independent agent turn budget (round-trips), per measured resolve ceiling.
TURN_CAP = 90

# Wall-clock backstop per trajectory (seconds) so a pathological slow-compile loop
# can't run forever even under the turn cap.
WALL_CLOCK_CEILING_S: dict[str, int] = {"python": 2700, "rust": 5400, "cpp": 7200}
