# Rust Literal Python-Heredoc Mutation Classifier Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Classify the exact literal Python patch-writer shape used by OpenHands Rust traces so v3.1 can distinguish allowlisted source edits from ambiguous code without weakening any contamination gate.

**Architecture:** Add one pure, exact-shape AST extractor beside the existing shell mutation parser and route only proven `python3 - <<'PYEOF'` commands through it. All unsupported shell or Python forms remain ambiguous and continue through the existing full-trajectory decisive-suffix gates unchanged.

**Tech Stack:** Python 3.11 standard-library `ast`, `re`, `PurePosixPath`, pytest, existing Rust-v3 streaming builder and independent auditor.

## Global Constraints

- The accepted shell and AST grammars are exactly those frozen in `docs/superpowers/specs/2026-07-20-rust-python-heredoc-classifier-design.md`.
- Commands, reasoning, observations, return codes, tool-call IDs, and message order are never rewritten.
- Dynamic paths, unknown calls, additional statements, unquoted heredocs, wrappers, pipes, and malformed programs remain ambiguous.
- Whole-original-trajectory classification remains active.
- Test/fixture/example/bench, lockfile, non-allowlisted, duplicate-edit, grounding, trusted-verification, pairing, and terminal gates remain unchanged.
- The frozen source revision, 239-task exclusion, canonical ID-set hash, and pre-exclusion boundary remain unchanged.
- Live work is CPU-only. Do not touch GPU, Docker, serving, or production.
- Do not publish unless all existing v3.1 row/repository/behavior/token/format gates pass.

---

### Task 1: Add the exact Python-heredoc classifier and integrate mutation scope

**Files:**
- Modify: `phaseD_sft/build_rust_v3_agentic_dataset.py:1-35,780-1165`
- Modify: `phaseD_sft/tests/test_build_rust_v3_agentic_dataset.py:635-875,1180-1590`

**Interfaces:**
- Produces: `_literal_python_heredoc_path(command: str) -> str | None`.
- Integrates with: `_mutation_scope(command: str, declared_root: str) -> MutationScope`.
- Preserves: every existing caller and public dataset/manifest schema.

- [ ] **Step 1: Add focused failing tests for the accepted exact shape**

Add the following helper and assertions near the existing mutation-scope tests:

```python
def _python_replace(path: str, *, path_name: str = "p", text_name: str = "s") -> str:
    return (
        "python3 - <<'PYEOF'\n"
        "import pathlib\n"
        f"{path_name} = pathlib.Path({path!r})\n"
        f"{text_name} = {path_name}.read_text()\n"
        f"{text_name} = {text_name}.replace('old', 'new', 1)\n"
        f"{path_name}.write_text({text_name})\n"
        "PYEOF"
    )


@pytest.mark.parametrize(
    ("command", "repo", "scratch"),
    [
        (_python_replace("/workspace/repo/src/lib.rs"), ("src/lib.rs",), ()),
        (
            "cd /workspace/repo && " + _python_replace("src/lib.rs"),
            ("src/lib.rs",),
            (),
        ),
        (_python_replace("/tmp/repro.rs"), (), ("/tmp/repro.rs",)),
        (
            _python_replace("src/lib.rs", path_name="target", text_name="body"),
            ("src/lib.rs",),
            (),
        ),
    ],
)
def test_mutation_scope_accepts_exact_literal_python_replace(
    command: str, repo: tuple[str, ...], scratch: tuple[str, ...]
) -> None:
    scope = rust_v3_builder._mutation_scope(command, "/workspace/repo")
    assert scope == rust_v3_builder.MutationScope(repo, scratch, False)
```

- [ ] **Step 2: Add fail-closed tests for shell-envelope deviations**

Parameterize the exact deviations below. Each must return empty paths with
`ambiguous is True`:

```python
@pytest.mark.parametrize(
    "command",
    [
        _python_replace("src/lib.rs").replace("<<'PYEOF'", "<<PYEOF", 1),
        _python_replace("src/lib.rs").replace("python3 -", "python3 -u -", 1),
        "env " + _python_replace("src/lib.rs"),
        _python_replace("src/lib.rs") + "\necho done",
        _python_replace("src/lib.rs").replace("PYEOF", "PYEOF2", 1),
        _python_replace("src/lib.rs") + " | cat",
    ],
)
def test_literal_python_heredoc_shell_deviations_remain_ambiguous(
    command: str,
) -> None:
    scope = rust_v3_builder._mutation_scope(command, "/workspace/repo")
    assert scope == rust_v3_builder.MutationScope((), (), True)
```

Use explicit command construction rather than a replacement if a replacement
would change both the opener and terminator.

- [ ] **Step 3: Add fail-closed tests for AST deviations**

Build each body from the valid five statements and change exactly one property:

- `from pathlib import Path` or `import pathlib as p`;
- dynamic path expression;
- missing or extra statement;
- `read_bytes`/`write_bytes` or built-in `open`;
- omitted/nonliteral/boolean replacement count;
- different receiver or output variable;
- extra import, `os.system`, subprocess, loop, conditional, exception handler,
  lambda, comprehension, `eval`, or `exec`.

Every case must assert `MutationScope((), (), True)`. Retain the existing
`test_mutation_scope_rejects_interpreter_heredoc_semantics`: its
`from pathlib import Path` form must remain rejected.

- [ ] **Step 4: Add decisive-suffix composition tests**

Use `_decisive_fixture` to prove the valid Python writer reaches the same
allowlist gates as a `sed` edit:

```python
def test_decisive_suffix_accepts_allowlisted_literal_python_writer() -> None:
    analyzed, compressed = _decisive_fixture(
        [(_python_replace("/workspace/repo/src/lib.rs"), 0), ("cargo test", 0)],
        patch=_patch(("src/lib.rs", "@@ -1 +1 @@\n-old\n+new")),
    )
    result, report = rust_v3_builder._build_decisive_suffix(analyzed, compressed)
    assert result is not None
    assert report == {"kept": True}


@pytest.mark.parametrize("path", ["tests/case.rs", "Cargo.lock", "README.md"])
def test_decisive_suffix_keeps_python_writer_path_gates(path: str) -> None:
    analyzed, compressed = _decisive_fixture(
        [(_python_replace(f"/workspace/repo/{path}"), 0), ("cargo test", 0)],
        patch=_patch(("src/lib.rs", "@@ -1 +1 @@\n-old\n+new")),
    )
    result, report = rust_v3_builder._build_decisive_suffix(analyzed, compressed)
    assert result is None
    assert report["reason"] in {
        "forbidden_mutation_path",
        "nonallowlisted_repository_mutation",
    }
```

- [ ] **Step 5: Run the new tests and verify RED**

Sync only the focused test file to the CPU host, then run:

```bash
cd /home/ironbcc/projects/gemma4-31B-Coder
PYTHONPATH=$PWD .venv-eval/bin/python -m pytest \
  phaseD_sft/tests/test_build_rust_v3_agentic_dataset.py -q \
  -k 'literal_python or python_writer'
```

Expected: accepted-shape and composition tests fail because the current
classifier returns `ambiguous=True`; rejection tests remain green.

- [ ] **Step 6: Implement the exact AST extractor**

Add `import ast`. Implement `_literal_python_heredoc_path` using these exact
checks:

```python
def _literal_python_heredoc_path(command: str) -> str | None:
    lines = command.splitlines()
    if not lines:
        return None
    start = _HEREDOC_START_RE.search(lines[0])
    if (
        start is None
        or start.group("strip")
        or start.group("quote") != "'"
        or lines[-1] != start.group("delimiter")
    ):
        return None

    fragments = _shell_executable_fragments(command)
    if not fragments:
        return None
    executable_fragments = fragments
    if PurePosixPath(fragments[0][0][0]).name == "cd":
        executable_fragments = fragments[1:]
    if len(executable_fragments) != 1:
        return None
    tokens, following = executable_fragments[0]
    if (
        following is not None
        or len(tokens) != 4
        or PurePosixPath(tokens[0]).name != "python3"
        or tokens[1] != "-"
        or tokens[2] != "<<"
        or tokens[3] != start.group("delimiter")
    ):
        return None

    try:
        module = ast.parse("\n".join(lines[1:-1]))
    except (SyntaxError, ValueError):
        return None

    if len(module.body) != 5:
        return None
    import_stmt, path_stmt, read_stmt, replace_stmt, write_stmt = module.body
    if not (
        isinstance(import_stmt, ast.Import)
        and len(import_stmt.names) == 1
        and import_stmt.names[0].name == "pathlib"
        and import_stmt.names[0].asname is None
    ):
        return None
    if not (
        isinstance(path_stmt, ast.Assign)
        and path_stmt.type_comment is None
        and len(path_stmt.targets) == 1
        and isinstance(path_stmt.targets[0], ast.Name)
        and isinstance(path_stmt.value, ast.Call)
        and isinstance(path_stmt.value.func, ast.Attribute)
        and isinstance(path_stmt.value.func.value, ast.Name)
        and path_stmt.value.func.value.id == "pathlib"
        and path_stmt.value.func.attr == "Path"
        and len(path_stmt.value.args) == 1
        and not path_stmt.value.keywords
        and isinstance(path_stmt.value.args[0], ast.Constant)
        and type(path_stmt.value.args[0].value) is str
    ):
        return None
    path_name = path_stmt.targets[0].id
    literal_path = path_stmt.value.args[0].value
    if not (
        isinstance(read_stmt, ast.Assign)
        and read_stmt.type_comment is None
        and len(read_stmt.targets) == 1
        and isinstance(read_stmt.targets[0], ast.Name)
        and isinstance(read_stmt.value, ast.Call)
        and isinstance(read_stmt.value.func, ast.Attribute)
        and isinstance(read_stmt.value.func.value, ast.Name)
        and read_stmt.value.func.value.id == path_name
        and read_stmt.value.func.attr == "read_text"
        and not read_stmt.value.args
        and not read_stmt.value.keywords
    ):
        return None
    text_name = read_stmt.targets[0].id
    if text_name == path_name:
        return None
    if not (
        isinstance(replace_stmt, ast.Assign)
        and replace_stmt.type_comment is None
        and len(replace_stmt.targets) == 1
        and isinstance(replace_stmt.targets[0], ast.Name)
        and replace_stmt.targets[0].id == text_name
        and isinstance(replace_stmt.value, ast.Call)
        and isinstance(replace_stmt.value.func, ast.Attribute)
        and isinstance(replace_stmt.value.func.value, ast.Name)
        and replace_stmt.value.func.value.id == text_name
        and replace_stmt.value.func.attr == "replace"
        and len(replace_stmt.value.args) == 3
        and not replace_stmt.value.keywords
        and all(
            isinstance(value, ast.Constant)
            for value in replace_stmt.value.args
        )
        and type(replace_stmt.value.args[0].value) is str
        and type(replace_stmt.value.args[1].value) is str
        and type(replace_stmt.value.args[2].value) is int
        and replace_stmt.value.args[2].value == 1
    ):
        return None
    if not (
        isinstance(write_stmt, ast.Expr)
        and isinstance(write_stmt.value, ast.Call)
        and isinstance(write_stmt.value.func, ast.Attribute)
        and isinstance(write_stmt.value.func.value, ast.Name)
        and write_stmt.value.func.value.id == path_name
        and write_stmt.value.func.attr == "write_text"
        and len(write_stmt.value.args) == 1
        and isinstance(write_stmt.value.args[0], ast.Name)
        and write_stmt.value.args[0].id == text_name
        and not write_stmt.value.keywords
    ):
        return None
    return literal_path
```

Require `type(replace_count.value) is int` and value `1`, so `True` is not
accepted. Return only the literal path string.

- [ ] **Step 7: Integrate the extractor into `_mutation_scope`**

Compute the proven path once from the full command. In the fragment loop,
replace the blanket non-`cat`/`tee` heredoc rejection with:

```python
literal_python_path = _literal_python_heredoc_path(command)

# inside the fragment loop
raw_paths: tuple[str, ...]
if has_heredoc and executable == "python3":
    if literal_python_path is None:
        return MutationScope((), (), True)
    raw_paths = (literal_python_path,)
elif has_heredoc and executable not in {"cat", "tee"}:
    return MutationScope((), (), True)
else:
    raw_paths = _fragment_mutation_paths(tokens)

for raw_path in raw_paths:
    # existing resolution and repo/scratch classification, unchanged
```

The helper's shell-envelope check must prevent the same proven path from being
applied to more than one executable fragment.

- [ ] **Step 8: Run focused and full tests**

```bash
PYTHONPATH=$PWD .venv-eval/bin/python -m pytest \
  phaseD_sft/tests/test_build_rust_v3_agentic_dataset.py -q
PYTHONPATH=$PWD .venv-eval/bin/python -m pytest phaseD_sft/tests -q
python -m py_compile \
  phaseD_sft/build_rust_v3_agentic_dataset.py \
  phaseD_sft/tests/test_build_rust_v3_agentic_dataset.py
git diff --check -- \
  phaseD_sft/build_rust_v3_agentic_dataset.py \
  phaseD_sft/tests/test_build_rust_v3_agentic_dataset.py
```

Expected: all selected tests and the complete Phase-D suite pass.

- [ ] **Step 9: Commit only the two Task 1 files**

```bash
git commit --only \
  phaseD_sft/build_rust_v3_agentic_dataset.py \
  phaseD_sft/tests/test_build_rust_v3_agentic_dataset.py \
  -m "feat: classify literal python rust edits"
```

---

### Task 2: Independently review and remeasure live v3.1 yield

**Files:**
- Review: Task 1 commit and the frozen design/spec.
- Create remotely: `data/rust_sft_v3p1_agentic_49k.audit.log`.
- Modify only after verified results: `.superpowers/sdd/progress.md`, `phaseH_eval/SESSION_STATE.md`.

**Interfaces:**
- Consumes: `_literal_python_heredoc_path` and `_mutation_scope` from Task 1.
- Produces: independent spec/quality verdict and authoritative audit-only yield.

- [ ] **Step 1: Run an independent task review**

The reviewer must inspect every accepted and rejected AST/shell shape, confirm
the existing full-trajectory and path gates remain unchanged, run the focused
builder tests, and issue explicit SPEC/QUALITY verdicts. Any Critical or
Important finding returns to Task 1 for a test-first fix and re-review.

- [ ] **Step 2: Sync reviewed files and verify hashes**

Sync only the builder and focused test file from the local repository of record
to `/home/ironbcc/projects/gemma4-31B-Coder`, then compare SHA256 values.

- [ ] **Step 3: Run the CPU/RAM preflight**

```bash
cd /home/ironbcc/projects/gemma4-31B-Coder
test "$(awk '/MemAvailable:/ {print $2}' /proc/meminfo)" -ge 15728640
test ! -e data/rust_sft_v3p1_agentic_49k
test ! -e data/.rust_sft_v3p1_agentic_49k.publish.lock
```

- [ ] **Step 4: Rerun audit-only with the corrected exclusion hash contract**

```bash
PYTHONPATH=$PWD .venv-train/bin/python \
  phaseD_sft/build_rust_v3_agentic_dataset.py \
  --audit-only --behavior-contract v3p1 \
  --dataset nvidia/Open-SWE-Traces \
  --revision 9c0e4579a4ee0effa3e5f7a552494a045f29377d \
  --config openhands:qwen35_122b \
  --exclude data/mswe_rust_prs_full239.jsonl \
  --expected-exclusion-count 239 \
  --expected-exclusion-sha256 bc0a6b0994d437af5f00323fbe87846e0f424c6a88b484f3f4c5592f1f1dc645 \
  --expected-pre-exclusion-eligible 825 \
  --progress-every 1000 \
  > data/rust_sft_v3p1_agentic_49k.audit.log 2>&1
```

Expected boundary: 825. Report compressed eligible, every behavior drop count,
candidate task/repository counts, and elapsed time.

- [ ] **Step 5: Apply the build decision gate**

If audit survivors cannot plausibly produce at least 60 rows across 35
repositories, stop without building and measure alternate revision-pinned
Open-SWE configs. Do not relax gates.

If the threshold is plausible, execute the already-reviewed Task 5 build,
independent auditor, format/loss gate, and full Phase-D commands in
`docs/superpowers/plans/2026-07-20-rust-v3p1-decisive-suffix.md`, using the
canonical exclusion ID-set hash
`bc0a6b0994d437af5f00323fbe87846e0f424c6a88b484f3f4c5592f1f1dc645`.

- [ ] **Step 6: Record artifacts and keep training gated**

Record exact counts, hashes, commands, and pass/fail evidence in the progress
ledger and `phaseH_eval/SESSION_STATE.md`. Do not start training. The full-239
raw-base baseline and its production-health/>6-hour gates remain prerequisites.
