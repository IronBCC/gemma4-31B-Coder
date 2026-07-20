# Rust v3.1 Gold-Path Decisive-Suffix Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an explicit Rust-v3.1 dataset contract that retains only row-local gold-path repository mutations and ends supervision after the first trusted successful verification following the final valid edit.

**Architecture:** Extend the existing streaming Rust-v3 builder rather than introducing a second source scanner. Preserve `v3` as the default behavior contract, add `v3p1` as an opt-in transformation before per-task selection, and reuse the existing strict patch parser, native tool conversion, grounding, tokenizer, manifest, and atomic publication machinery. Every unsafe or ambiguous row fails closed with a stable drop reason.

**Tech Stack:** Python 3.11+, dataclasses, standard-library parsing/hash/atomic I/O, Hugging Face streaming datasets/tokenizer, pytest, existing Gemma format/loss verifier.

## Global Constraints

- Work only in `phaseD_sft/build_rust_v3_agentic_dataset.py`, its focused tests, the v3.1 output directory, and the canonical session/progress docs.
- Keep `AuditInputs.behavior_contract="v3"` backward compatible; only explicit `v3p1` enables the new gates.
- Never rewrite assistant reasoning, commands, observations, return codes, or tool-call IDs.
- Apply the frozen 239-task exclusion before conversion; zero overlap is mandatory.
- Repository test/fixture/example/bench, lockfile, non-allowlisted, or ambiguous mutations are hard row rejections.
- The final retained repository mutation must be followed by a trusted verification with explicit return code zero; a pipeline is trusted only with `pipefail`.
- Output is atomically published to `data/rust_sft_v3p1_agentic_49k`; never overwrite v3 or an existing v3.1 directory.
- Hard token ceiling is 49,152; tokenizer loads once in the main thread; only bounded threads count tokens.
- Build success requires at least 60 rows across at least 35 repositories, first-edit median at most 4, format/loss `failure_count=0`, and every behavioral invariant at zero violations.
- This plan is CPU-only. Do not touch GPU, Docker, serving processes, or production ports.

---

### Task 1: Generalize the row-local patch parser to expose every textual path

**Files:**
- Modify: `phaseD_sft/build_rust_v3_agentic_dataset.py:85-333`
- Test: `phaseD_sft/tests/test_build_rust_v3_agentic_dataset.py:152-453`

**Interfaces:**
- Produces: `TrackedRustPaths.textual_paths: tuple[str, ...]`, containing all normalized paths whose own diff section contains a text hunk.
- Preserves: `tracked_paths` remains textual `.rs` paths; `source_paths` remains non-test textual `.rs` paths.

- [ ] **Step 1: Write failing parser tests**

Add tests proving all textual paths are row-local, hunk-backed, normalized, and independent of Rust-source eligibility:

```python
def test_patch_parser_exposes_all_textual_paths_for_row_local_allowlist() -> None:
    patch = _patch(
        ("src/lib.rs", "@@ -1 +1 @@\n-old\n+new"),
        ("Cargo.toml", "@@ -1 +1 @@\n-old\n+new"),
        ("include/api.hpp", "@@ -1 +1 @@\n-old\n+new"),
    )
    paths, report = builder.extract_tracked_rust_paths(patch)
    assert report == {"kept": True}
    assert paths is not None
    assert paths.tracked_paths == ("src/lib.rs",)
    assert paths.source_paths == ("src/lib.rs",)
    assert paths.textual_paths == ("Cargo.toml", "include/api.hpp", "src/lib.rs")


def test_hunkless_or_binary_nonrust_section_is_not_allowlisted() -> None:
    patch = (
        "diff --git a/src/lib.rs b/src/lib.rs\n"
        "--- a/src/lib.rs\n+++ b/src/lib.rs\n@@ -1 +1 @@\n-old\n+new\n"
        "diff --git a/Cargo.toml b/Cargo.toml\n"
        "Binary files a/Cargo.toml and b/Cargo.toml differ\n"
    )
    paths, report = builder.extract_tracked_rust_paths(patch)
    assert report == {"kept": True}
    assert paths is not None
    assert paths.textual_paths == ("src/lib.rs",)
```

- [ ] **Step 2: Run the tests against unchanged production code and verify RED**

Sync only the focused test file to the host, then run:

```bash
cd /home/ironbcc/projects/gemma4-31B-Coder
.venv-eval/bin/python -m pytest \
  phaseD_sft/tests/test_build_rust_v3_agentic_dataset.py \
  -q -k 'exposes_all_textual_paths or hunkless_or_binary_nonrust'
```

Expected: failures because `TrackedRustPaths` has no `textual_paths` field.

- [ ] **Step 3: Refactor the existing parser without weakening Rust gates**

Change the dataclass and the final section loop so only per-section text-hunk paths enter the all-path allowlist:

```python
@dataclass(frozen=True)
class TrackedRustPaths:
    tracked_paths: tuple[str, ...]
    source_paths: tuple[str, ...]
    textual_paths: tuple[str, ...]


textual_paths: set[str] = set()
textual_rust_paths: set[str] = set()
for section in sections:
    # retain all existing rename/header/hunk consistency checks
    header_paths = {section.old_path, section.new_path}
    if section.saw_text_hunk:
        # retain the existing ---/+++ validation before adding paths
        textual_paths.update(header_paths)
        textual_rust_paths.update(
            path for path in header_paths if path.lower().endswith(".rs")
        )

rust_paths = tuple(sorted(textual_rust_paths))
source_paths = tuple(path for path in rust_paths if not _is_test_path(path))
return (
    TrackedRustPaths(rust_paths, source_paths, tuple(sorted(textual_paths))),
    {"kept": True},
)
```

Do not accept a malformed non-Rust section merely because a Rust section is valid; every parsed section retains the current header and rename consistency checks.

- [ ] **Step 4: Run focused and parser regression tests**

```bash
.venv-eval/bin/python -m pytest \
  phaseD_sft/tests/test_build_rust_v3_agentic_dataset.py \
  -q -k 'patch or path or rename or binary or hunk'
```

Expected: all selected tests pass.

- [ ] **Step 5: Commit only Task 1 files**

```bash
git commit --only \
  phaseD_sft/build_rust_v3_agentic_dataset.py \
  phaseD_sft/tests/test_build_rust_v3_agentic_dataset.py \
  -m "feat: expose row-local textual patch paths"
```

---

### Task 2: Add fail-closed mutation-scope and trusted-verification primitives

**Files:**
- Modify: `phaseD_sft/build_rust_v3_agentic_dataset.py:390-790`
- Test: `phaseD_sft/tests/test_build_rust_v3_agentic_dataset.py`

**Interfaces:**
- Produces: `MutationScope(repository_paths, scratch_paths, ambiguous)`.
- Produces: `_mutation_scope(command: str, declared_root: str) -> MutationScope`.
- Produces: `_trusted_rust_verification(command: str, observation: Mapping[str, Any]) -> bool`.
- Produces: `_forbidden_training_path(path: str) -> bool`.

- [ ] **Step 1: Write failing scope tests**

```python
@pytest.mark.parametrize(
    ("command", "repo", "scratch", "ambiguous"),
    [
        ("sed -i s/a/b/ src/lib.rs", ("src/lib.rs",), (), False),
        ("cd /workspace/repo && cat > Cargo.toml <<'EOF'\nX\nEOF", ("Cargo.toml",), (), False),
        ("cd /workspace && cat > repro.rs <<'EOF'\nfn main(){}\nEOF", (), ("/workspace/repro.rs",), False),
        ("cat > $TARGET <<'EOF'\nX\nEOF", (), (), True),
    ],
)
def test_mutation_scope_distinguishes_repo_scratch_and_ambiguous(
    command, repo, scratch, ambiguous
) -> None:
    scope = builder._mutation_scope(command, "/workspace/repo")
    assert scope.repository_paths == repo
    assert scope.scratch_paths == scratch
    assert scope.ambiguous is ambiguous
```

Also add hard-path tests:

```python
@pytest.mark.parametrize(
    "path",
    ["tests/case.rs", "examples/demo.rs", "fixtures/input.txt", "benches/bench.rs",
     "Cargo.lock", "supply-chain/imports.lock"],
)
def test_forbidden_training_paths(path: str) -> None:
    assert builder._forbidden_training_path(path)
```

- [ ] **Step 2: Write failing trusted-verification tests**

```python
def _observation(rc: int) -> dict[str, object]:
    return {"role": "user", "content": f"OBSERVATION:\n<returncode>{rc}</returncode>\noutput"}


@pytest.mark.parametrize(
    ("command", "rc", "expected"),
    [
        ("cargo test", 0, True),
        ("cargo check", 1, False),
        ("cargo test 2>&1 | tail -20", 0, False),
        ("set -o pipefail; cargo test 2>&1 | tail -20", 0, True),
        ("grep cargo Cargo.toml", 0, False),
    ],
)
def test_trusted_verification_requires_real_command_unmasked_zero_rc(
    command, rc, expected
) -> None:
    assert builder._trusted_rust_verification(command, _observation(rc)) is expected
```

- [ ] **Step 3: Verify RED on the host**

Run:

```bash
.venv-eval/bin/python -m pytest \
  phaseD_sft/tests/test_build_rust_v3_agentic_dataset.py -q \
  -k 'mutation_scope or forbidden_training_paths or trusted_verification'
```

Expected: missing helper/dataclass failures, not fixture or import errors.

- [ ] **Step 4: Implement minimal pure helpers**

Add:

```python
@dataclass(frozen=True)
class MutationScope:
    repository_paths: tuple[str, ...]
    scratch_paths: tuple[str, ...]
    ambiguous: bool


_RETURN_CODE_RE = re.compile(
    r"(?:<returncode>|command (?:completed|finished) with exit code )(?P<rc>[0-9]+)",
    re.IGNORECASE,
)
_SINGLE_PIPE_RE = re.compile(r"(?<!\|)\|(?!\|)")


def _forbidden_training_path(path: str) -> bool:
    name = PurePosixPath(path).name.lower()
    return _is_test_path(path) or name == "cargo.lock" or name.endswith(".lock")


def _trusted_rust_verification(command: str, observation: Mapping[str, Any]) -> bool:
    visible = _shell_visible_text(command)
    if _RUST_VERIFY_RE.search(visible) is None:
        return False
    if _SINGLE_PIPE_RE.search(visible) and not re.search(
        r"(?:set\s+-o\s+pipefail|set\s+-[^;\n]*o[^;\n]*pipefail)", visible
    ):
        return False
    matches = list(_RETURN_CODE_RE.finditer(_observation_body(observation)))
    return bool(matches) and int(matches[-1].group("rc")) == 0
```

Implement `_mutation_scope` by reusing `edited_paths`, parsing an initial/compound `cd`, resolving relative operands against that directory, and rejecting variables, traversal, multiple incompatible working directories, or unnormalizable operands as ambiguous. Absolute paths inside the declared root become repository-relative; safe absolute paths outside it remain scratch paths.

- [ ] **Step 5: Run focused tests and the existing mutation/parser suite**

Expected: new tests plus existing `tracked`, `mutation`, `embedded`, `operator`, and `path` tests pass.

- [ ] **Step 6: Commit only Task 2 files**

```bash
git commit --only phaseD_sft/build_rust_v3_agentic_dataset.py \
  phaseD_sft/tests/test_build_rust_v3_agentic_dataset.py \
  -m "feat: classify rust training mutation scope"
```

---

### Task 3: Build the decisive suffix without fabricating trajectory state

**Files:**
- Modify: `phaseD_sft/build_rust_v3_agentic_dataset.py:1053-1185`
- Test: `phaseD_sft/tests/test_build_rust_v3_agentic_dataset.py`

**Interfaces:**
- Produces: `DecisiveSuffix(messages, final_edit_command_index, verification_command_index)`.
- Produces: `_build_decisive_suffix(analyzed: AnalyzedSourceRow, compressed: CompressedAgenticRow) -> tuple[DecisiveSuffix | None, dict[str, object]]`.

- [ ] **Step 1: Add one fixture builder that creates a native compressed row**

The fixture must expose exact assistant/tool/observation pairs and a final marker; it must not bypass `convert_rich_trajectory`. Give it this interface so each behavior test uses real conversion:

```python
def _decisive_fixture(
    commands: list[tuple[str, int]],
    *,
    patch: str,
) -> tuple[builder.AnalyzedSourceRow, builder.CompressedAgenticRow]:
    """Build a rich trajectory whose paired observations expose the given rc values."""
```

- [ ] **Step 2: Write failing gold-path/suffix tests**

Cover these separate behaviors:

```python
def test_decisive_suffix_keeps_gold_edits_through_first_trusted_success_verification():
    analyzed, compressed = _decisive_fixture(
        [
            ("sed -i s/old/new/ src/lib.rs", 0),
            ("cargo check", 1),
            ("sed -i s/new/fixed/ src/lib.rs", 0),
            ("cargo test", 0),
            ("cargo test", 0),
            ("cat > IMPLEMENTATION_SUMMARY.md <<'EOF'\nsummary\nEOF", 0),
        ],
        patch=_patch(("src/lib.rs", "@@ -1 +1 @@\n-old\n+fixed")),
    )
    result, report = builder._build_decisive_suffix(analyzed, compressed)
    assert report["kept"] is True
    assert result is not None
    kept = [_command(message) for message in result.messages if message["role"] == "assistant"]
    assert kept[-2:] == ["cargo test", builder._COMPLETE_COMMAND]


def test_decisive_suffix_rejects_repo_test_edit_even_when_final_patch_contains_it():
    analyzed, compressed = _decisive_fixture(
        [("sed -i s/a/b/ src/lib.rs", 0), ("sed -i s/a/b/ tests/case.rs", 0),
         ("cargo test", 0)],
        patch=_patch(("src/lib.rs", "@@ -1 +1 @@\n-a\n+b"),
                     ("tests/case.rs", "@@ -1 +1 @@\n-a\n+b")),
    )
    result, report = builder._build_decisive_suffix(analyzed, compressed)
    assert result is None
    assert report["reason"] == "forbidden_patch_path"


def test_decisive_suffix_rejects_nonallowlisted_repo_mutation():
    analyzed, compressed = _decisive_fixture(
        [("sed -i s/a/b/ src/lib.rs", 0),
         ("cat > README_FIX.md <<'EOF'\nsummary\nEOF", 0), ("cargo test", 0)],
        patch=_patch(("src/lib.rs", "@@ -1 +1 @@\n-a\n+b")),
    )
    assert builder._build_decisive_suffix(analyzed, compressed)[1]["reason"] == (
        "nonallowlisted_repository_mutation"
    )


def test_decisive_suffix_allows_final_patch_tracked_nonrust_source_file():
    analyzed, compressed = _decisive_fixture(
        [("sed -i s/a/b/ src/lib.rs", 0), ("sed -i s/a/b/ include/api.hpp", 0),
         ("cargo test", 0)],
        patch=_patch(("src/lib.rs", "@@ -1 +1 @@\n-a\n+b"),
                     ("include/api.hpp", "@@ -1 +1 @@\n-a\n+b")),
    )
    result, report = builder._build_decisive_suffix(analyzed, compressed)
    assert result is not None and report["kept"] is True


def test_decisive_suffix_rejects_nonconsecutive_identical_edit_commands():
    edit = "sed -i s/a/b/ src/lib.rs"
    analyzed, compressed = _decisive_fixture(
        [(edit, 0), ("sed -n 1,20p src/lib.rs", 0), (edit, 0), ("cargo test", 0)],
        patch=_patch(("src/lib.rs", "@@ -1 +1 @@\n-a\n+b")),
    )
    assert builder._build_decisive_suffix(analyzed, compressed)[1]["reason"] == (
        "repeated_edit_command"
    )


def test_decisive_suffix_rejects_missing_trusted_verify_after_final_edit():
    analyzed, compressed = _decisive_fixture(
        [("sed -i s/a/b/ src/lib.rs", 0), ("cargo test 2>&1 | tail -20", 0)],
        patch=_patch(("src/lib.rs", "@@ -1 +1 @@\n-a\n+b")),
    )
    assert builder._build_decisive_suffix(analyzed, compressed)[1]["reason"] == (
        "missing_trusted_final_verification"
    )
```

Add scratch tests proving exactly one create/consume chain before the final repository edit is retained, while repeated, unconsumed, post-final-edit, or ambiguous scratch mutations reject the row.

- [ ] **Step 3: Run the new tests and verify RED**

Expected: `_build_decisive_suffix`/`DecisiveSuffix` missing.

- [ ] **Step 4: Implement the transformation**

```python
@dataclass(frozen=True)
class DecisiveSuffix:
    messages: tuple[dict[str, Any], ...]
    final_edit_command_index: int
    verification_command_index: int
```

The implementation must:

1. derive `allowlist = set(analyzed.patch_paths.textual_paths)`;
2. reject any allowlisted forbidden path;
3. iterate real native pairs from `compressed.messages` and classify every mutation;
4. reject ambiguous, forbidden, or non-allowlisted repository mutations;
5. track normalized edit commands globally and reject any duplicate;
6. validate at most one consumed scratch chain before the final repository mutation;
7. locate the first trusted successful verification after the final repository mutation;
8. retain the existing prefix, every original pair through that verification, at most one following read-only git diff/status pair, and the exact original terminal marker;
9. deep-copy retained messages without changing their values; and
10. rerun balanced pairing, terminal-last, and first-edit grounding checks on the result.

Return stable reasons such as `forbidden_patch_path`, `ambiguous_mutation_path`, `nonallowlisted_repository_mutation`, `invalid_scratch_chain`, `repeated_edit_command`, and `missing_trusted_final_verification`.

- [ ] **Step 5: Run focused tests, then the complete builder test file**

```bash
.venv-eval/bin/python -m pytest \
  phaseD_sft/tests/test_build_rust_v3_agentic_dataset.py -q
```

Expected: all focused builder tests pass.

- [ ] **Step 6: Commit only Task 3 files**

```bash
git commit --only phaseD_sft/build_rust_v3_agentic_dataset.py \
  phaseD_sft/tests/test_build_rust_v3_agentic_dataset.py \
  -m "feat: construct decisive rust verification suffixes"
```

---

### Task 4: Integrate v3.1 before selection and bind every behavior gate in the manifest

**Files:**
- Modify: `phaseD_sft/build_rust_v3_agentic_dataset.py:1350-2005`
- Test: `phaseD_sft/tests/test_build_rust_v3_agentic_dataset.py:1055-1795`

**Interfaces:**
- Extends: `AuditInputs.behavior_contract: str = "v3"`.
- CLI: `--behavior-contract {v3,v3p1}`, default `v3`.
- Candidate fields for v3.1: `final_edit_command_index`, `verification_command_index`.
- Manifest: behavior contract, behavior drop reasons, final-edit/verification histograms, and post-build invariant counts.

- [ ] **Step 1: Write failing compatibility and ordering tests**

Add tests proving:

- default `v3` produces the exact existing fixture output and counters;
- unknown behavior contracts fail before consuming streams;
- `v3p1` calls `_build_decisive_suffix` after compression but before task selection;
- an unsafe shorter trajectory is dropped and a safe longer trajectory for the same task wins;
- behavior drop reasons balance manifest arithmetic;
- v3.1 candidate/manifest fields round-trip and are absent from v3 output;
- CLI build and audit modes accept `--behavior-contract v3p1`.

- [ ] **Step 2: Verify RED on the host**

Expected: missing dataclass field/CLI option and wrong winner-selection behavior.

- [ ] **Step 3: Implement opt-in integration**

```python
@dataclass(frozen=True)
class AuditInputs:
    dataset_revision: str
    exclusion_path: Path
    expected_exclusion_count: int
    expected_exclusion_sha256: str
    expected_pre_exclusion_eligible: int
    behavior_contract: str = "v3"
```

Validate the contract before `_load_frozen_exclusions`. In `_scan_streams`, after the existing compression gate:

```python
messages = [copy.deepcopy(message) for message in compressed.messages]
behavior_fields: dict[str, int] = {}
if inputs.behavior_contract == "v3p1":
    decisive, decisive_report = _build_decisive_suffix(analyzed, compressed)
    if decisive is None:
        reason = str(decisive_report["reason"])
        drop_reasons[reason] += 1
        config_drops[reason] += 1
        continue
    messages = [copy.deepcopy(message) for message in decisive.messages]
    behavior_fields = {
        "final_edit_command_index": decisive.final_edit_command_index,
        "verification_command_index": decisive.verification_command_index,
    }
```

Add `behavior_contract` to the manifest and include final-edit and verification histograms only for v3.1. Recompute `content_sha256`, first-edit index, and max read streak from the decisive messages rather than copying stale pre-transform values.

- [ ] **Step 4: Add post-selection invariant enforcement**

Before publication, audit every final v3.1 row and raise without publishing unless:

```python
invariants == {
    "forbidden_mutations": 0,
    "lockfile_mutations": 0,
    "nonallowlisted_mutations": 0,
    "ambiguous_mutations": 0,
    "repeated_edit_commands": 0,
    "missing_trusted_final_verification": 0,
    "unbalanced_pairs": 0,
    "terminal_not_last": 0,
}
```

Also fail publication when `len(rows) < 60`, unique repositories `< 35`, or median first edit `> 4`. Record the exact thresholds and measured values in the manifest.

- [ ] **Step 5: Run full Phase-D tests**

```bash
PYTHONPATH=$PWD .venv-eval/bin/python -m pytest phaseD_sft/tests -q
```

Expected: zero failures; existing v3 behavior remains unchanged.

- [ ] **Step 6: Commit only Task 4 files**

```bash
git commit --only phaseD_sft/build_rust_v3_agentic_dataset.py \
  phaseD_sft/tests/test_build_rust_v3_agentic_dataset.py \
  -m "feat: gate rust v3p1 before task selection"
```

---

### Task 5: Run the CPU-only audit, build, and independent gates

**Files:**
- Create: `phaseD_sft/audit_rust_v3p1_dataset.py`
- Create: `phaseD_sft/tests/test_audit_rust_v3p1_dataset.py`
- Create: `data/rust_sft_v3p1_agentic_49k/` (only on passing publication)
- Create: `data/rust_sft_v3p1_agentic_49k.audit.log`
- Create: `data/rust_sft_v3p1_agentic_49k.validation.json`
- Create: `data/rust_sft_v3p1_agentic_49k.format_gate.log`
- Modify: `phaseH_eval/SESSION_STATE.md`

**Interfaces:**
- Consumes: the exact CLI and manifest contract from Task 4.
- Produces: immutable dataset/manifest plus independent validation and format artifacts.
- Produces: `audit_published_dataset(dataset_dir: Path, exclusions: set[str], count_rendered_tokens: RenderedTokenCounter) -> dict[str, Any]`.
- Internal pure helpers: `_audit_row_messages(row: Mapping[str, Any]) -> dict[str, int]` and `_manifest_matches_rows(manifest: Mapping[str, Any], rows: Sequence[Mapping[str, Any]]) -> bool`.

- [ ] **Step 1: Write failing independent-auditor tests**

Create fixtures for one safe published row and one violation at a time. Tests must prove:

```python
def test_auditor_recomputes_hashes_pairing_mutations_and_token_counts(tmp_path):
    report = auditor.audit_published_dataset(
        tmp_path / "dataset",
        exclusions={"heldout__repo-1"},
        count_rendered_tokens=lambda _messages: 321,
    )
    assert report["status"] == "passed"
    assert report["rows"] == 1
    assert report["violations"] == auditor.ZERO_VIOLATIONS


@pytest.mark.parametrize(
    "mutation",
    ["tests/case.rs", "examples/demo.rs", "Cargo.lock", "README_FIX.md"],
)
def test_auditor_fails_for_forbidden_or_nonallowlisted_mutation(tmp_path, mutation):
    dataset = _published_fixture(tmp_path, extra_mutation=mutation)
    report = auditor.audit_published_dataset(
        dataset, exclusions=set(), count_rendered_tokens=lambda _messages: 321
    )
    assert report["status"] == "failed"
    assert sum(report["violations"].values()) > 0
```

Also test 239 leakage, duplicate task/content hashes, token mismatch/overflow, missing trusted verification, repeated edit, terminal-not-last, manifest hash/arithmetic mismatch, and CLI exit 2 after emitting JSON.

- [ ] **Step 2: Run auditor tests and verify RED**

```bash
PYTHONPATH=$PWD .venv-eval/bin/python -m pytest \
  phaseD_sft/tests/test_audit_rust_v3p1_dataset.py -q
```

Expected: import failure because the auditor module does not exist.

- [ ] **Step 3: Implement the independent auditor and CLI**

The module must read only `dataset.jsonl`, `manifest.json`, and the frozen exclusion JSONL. It may import pure parsing primitives from the builder, but must not call the builder's scan, selection, manifest, or publication functions.

```python
ZERO_VIOLATIONS = {
    "heldout_overlap": 0,
    "duplicate_task_ids": 0,
    "duplicate_content_hashes": 0,
    "forbidden_mutations": 0,
    "nonallowlisted_mutations": 0,
    "ambiguous_mutations": 0,
    "repeated_edit_commands": 0,
    "missing_trusted_final_verification": 0,
    "unbalanced_pairs": 0,
    "terminal_not_last": 0,
    "token_mismatch": 0,
    "token_overflow": 0,
    "manifest_mismatch": 0,
}


def audit_published_dataset(
    dataset_dir: Path,
    exclusions: set[str],
    count_rendered_tokens: RenderedTokenCounter,
) -> dict[str, Any]:
    rows = [
        json.loads(line)
        for line in (dataset_dir / "dataset.jsonl").read_text().splitlines()
        if line.strip()
    ]
    manifest = json.loads((dataset_dir / "manifest.json").read_text())
    violations = dict(ZERO_VIOLATIONS)
    task_counts = Counter(str(row["task_id"]) for row in rows)
    content_counts = Counter(str(row["content_sha256"]) for row in rows)
    violations["heldout_overlap"] = sum(
        str(row["task_id"]) in exclusions for row in rows
    )
    violations["duplicate_task_ids"] = sum(
        count - 1 for count in task_counts.values() if count > 1
    )
    violations["duplicate_content_hashes"] = sum(
        count - 1 for count in content_counts.values() if count > 1
    )
    for row in rows:
        observed = _audit_row_messages(row)
        for key, value in observed.items():
            violations[key] += value
        actual_tokens = count_rendered_tokens(row["messages"])
        violations["token_mismatch"] += actual_tokens != row["tokens"]
        violations["token_overflow"] += not 0 < actual_tokens <= 49_152
    violations["manifest_mismatch"] += not _manifest_matches_rows(manifest, rows)
    status = "passed" if not any(violations.values()) else "failed"
    return {
        "status": status,
        "rows": len(rows),
        "repositories": len({row["repository"] for row in rows}),
        "violations": violations,
        "manifest_sha256": hashlib.sha256(
            (dataset_dir / "manifest.json").read_bytes()
        ).hexdigest(),
        "behavior_contract": manifest.get("behavior_contract"),
    }
```

The CLI accepts `--data`, `--exclude`, `--tokenizer`, `--tokenizer-revision`, and `--workers`; loads the tokenizer once; counts with one bounded `ThreadPoolExecutor`; prints one canonical JSON report; and exits 0 only when every violation is zero and manifest thresholds/hashes agree.

- [ ] **Step 4: Run focused auditor tests and full Phase-D tests**

Expected: auditor tests pass and the existing suite remains green.

- [ ] **Step 5: Commit only the auditor files**

```bash
git commit --only phaseD_sft/audit_rust_v3p1_dataset.py \
  phaseD_sft/tests/test_audit_rust_v3p1_dataset.py \
  -m "feat: audit published rust v3p1 datasets"
```

- [ ] **Step 6: Preflight CPU/RAM and confirm no final output exists**

```bash
cd /home/ironbcc/projects/gemma4-31B-Coder
free -g
test ! -e data/rust_sft_v3p1_agentic_49k
test ! -e .data/rust_sft_v3p1_agentic_49k.publish.lock
```

Require at least 15 GiB available RAM. Do not inspect or alter GPUs/services for this CPU-only build.

- [ ] **Step 7: Run the structural/behavior audit first**

```bash
PYTHONPATH=$PWD .venv-train/bin/python \
  phaseD_sft/build_rust_v3_agentic_dataset.py \
  --audit-only \
  --behavior-contract v3p1 \
  --dataset nvidia/Open-SWE-Traces \
  --revision 9c0e4579a4ee0effa3e5f7a552494a045f29377d \
  --config openhands:qwen35_122b \
  --exclude data/mswe_rust_prs_full239.jsonl \
  --expected-exclusion-count 239 \
  --expected-exclusion-sha256 5893e74d6e45183fc4e922dbe5bbe1169c1c26c8fbfa31129fb84a0e0c43fa8b \
  --expected-pre-exclusion-eligible 825 \
  --progress-every 1000 \
  > data/rust_sft_v3p1_agentic_49k.audit.log 2>&1
```

Expected: exit 0, input boundary 825, and explicit behavior drop counts. If the boundary drifts, stop without building.

- [ ] **Step 8: Run the v3.1 build with one tokenizer and eight threads**

```bash
PYTHONPATH=$PWD .venv-train/bin/python \
  phaseD_sft/build_rust_v3_agentic_dataset.py \
  --behavior-contract v3p1 \
  --dataset nvidia/Open-SWE-Traces \
  --revision 9c0e4579a4ee0effa3e5f7a552494a045f29377d \
  --config openhands:qwen35_122b \
  --exclude data/mswe_rust_prs_full239.jsonl \
  --expected-exclusion-count 239 \
  --expected-exclusion-sha256 5893e74d6e45183fc4e922dbe5bbe1169c1c26c8fbfa31129fb84a0e0c43fa8b \
  --expected-pre-exclusion-eligible 825 \
  --tokenizer /media/ironbcc/CrucialX10/models/google/gemma-4-31B-it \
  --tokenizer-revision local-gemma4-31b-it-20260614 \
  --template-identity gemma4-native-chat-template \
  --template-sha256 36e3a42e5cf14cd0020e72d92e1fdd9970f59b82170e421f0cbe1bb42bead3f0 \
  --out data/rust_sft_v3p1_agentic_49k \
  --max-tokens 49152 \
  --workers 8 \
  --progress-every 1000 \
  > /tmp/rust_v3p1_build.log 2>&1
```

Expected: atomic publication only when row/repository/behavior thresholds pass. If yield is below 60 rows or 35 repositories, exit nonzero and publish nothing.

- [ ] **Step 9: Independently validate every output row**

```bash
PYTHONPATH=$PWD .venv-train/bin/python \
  phaseD_sft/audit_rust_v3p1_dataset.py \
  --data data/rust_sft_v3p1_agentic_49k \
  --exclude data/mswe_rust_prs_full239.jsonl \
  --tokenizer /media/ironbcc/CrucialX10/models/google/gemma-4-31B-it \
  --tokenizer-revision local-gemma4-31b-it-20260614 \
  --workers 8 \
  > data/rust_sft_v3p1_agentic_49k.validation.json
```

Expected: exit 0, `status="passed"`, every `ZERO_VIOLATIONS` value zero, and independently recomputed task/content uniqueness, 239 overlap, pairing, terminal-last, edit/verify indices, mutation scope, token counts, per-repo counts, and hashes agree with the manifest.

- [ ] **Step 10: Run the Gemma format/loss gate over all rows**

```bash
PYTHONPATH=$PWD .venv-train/bin/python \
  phaseD_sft/verify_gemma_format_loss.py \
  --base /media/ironbcc/CrucialX10/models/google/gemma-4-31B-it \
  --data data/rust_sft_v3p1_agentic_49k \
  --samples 1000 \
  --workers 8 \
  --progress-every 60 \
  > data/rust_sft_v3p1_agentic_49k.format_gate.log 2>&1
```

Expected: exit 0, `failure_count=0`, no fallback spans, and all output rows sampled.

- [ ] **Step 11: Run fresh full tests and artifact hash verification**

```bash
PYTHONPATH=$PWD .venv-eval/bin/python -m pytest phaseD_sft/tests -q
sha256sum \
  data/rust_sft_v3p1_agentic_49k/dataset.jsonl \
  data/rust_sft_v3p1_agentic_49k/manifest.json \
  data/rust_sft_v3p1_agentic_49k.validation.json \
  data/rust_sft_v3p1_agentic_49k.format_gate.log
```

Expected: tests pass and hashes match the manifest/validation evidence.

- [ ] **Step 12: Record the result and stop before training**

Append the exact row/repository yield, drop table, token/first-edit/final-edit/verify histograms, violation table, format counts, hashes, and elapsed wall to `phaseH_eval/SESSION_STATE.md`. Explicitly state that training remains gated on the honest 239 raw-base baseline.

---

## Plan completion criteria

- Existing v3 behavior and tests remain unchanged by default.
- v3.1 publishes only if every design invariant and minimum diversity threshold passes.
- All output rows are independently replay-audited and Gemma-format valid.
- No GPU, Docker, serving, or production state is touched.
- No Rust adapter training is launched.
