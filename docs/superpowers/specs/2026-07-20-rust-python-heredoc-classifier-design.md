# Rust Literal Python-Heredoc Mutation Classifier Design

## Purpose

Restore meaningful Rust-v3.1 yield without weakening its contamination gates.
The revision-pinned OpenHands source produced 294 compressed-eligible Rust
trajectories, but v3.1 rejected all 294 before selection. The immediate cause
is classifier coverage: every published v3 row uses literal Python heredoc
patch writers, while `_mutation_scope` treats every non-`cat`/`tee` heredoc as
ambiguous.

This change classifies one proven Python edit shape. It does not relax
test/fixture/example/bench, lockfile, non-allowlisted, duplicate-edit,
verification, grounding, pairing, or whole-trajectory gates.

## Evidence and frozen source shape

The authoritative audit is
`data/rust_sft_v3p1_agentic_49k.audit.log` on the training host:

- 55,488 rows scanned;
- 825 structurally eligible after the frozen 239-task exclusion;
- 294 compressed eligible;
- 109 `forbidden_mutation_path`;
- 85 `ambiguous_mutation_path`;
- 62 `forbidden_patch_path`;
- 38 `nonallowlisted_repository_mutation`;
- zero candidates reach selection.

An independent scan of the existing 140-row v3 dataset found 845 Python
heredoc commands. All 845 use the exact shell header
`python3 - <<'PYEOF'`. Of them, 838 parse into the same five-statement AST:

```python
import pathlib
p = pathlib.Path("/literal/absolute/or/relative/path")
s = p.read_text()
s = s.replace("literal old", "literal new", 1)
p.write_text(s)
```

The remaining seven fail parsing and remain ambiguous.

## Chosen approach

Add a strict, pure classifier for the exact five-statement AST above. This is
preferred over either classifying arbitrary Python side effects or relaxing
eligibility to compressed messages only. It recovers the dominant source edit
form while preserving whole-trajectory purity and fail-closed behavior.

Alternative source configurations remain a fallback only after the corrected
classifier's measured audit yield is known.

## Accepted shell envelope

A Python heredoc is classifiable only when all of these are true:

1. The command contains exactly one heredoc.
2. The delimiter is single-quoted, so the shell cannot expand the body.
3. The executable basename is exactly `python3`.
4. Its only argument is `-`.
5. Apart from one already-supported leading literal `cd` fragment, the Python
   invocation is the only executable fragment.
6. The terminator is exact and no executable text follows it.

Any unquoted delimiter, multiple heredoc, extra flag, wrapper, pipe, trailing
command, malformed terminator, or unsupported working-directory change remains
ambiguous.

## Accepted Python AST

The module body must contain exactly five statements in this order:

1. `import pathlib`, with no alias and no additional import;
2. assignment of `pathlib.Path(<literal str>)` to one simple name;
3. assignment of `<path-name>.read_text()` to a second simple name;
4. reassignment of that same text name using
   `<text-name>.replace(<literal str>, <literal str>, 1)`;
5. expression `<path-name>.write_text(<text-name>)`.

The path is resolved through the existing `_scope_path` logic and classified
as repository-relative or external scratch against the declared repository
root. The returned `MutationScope` contains that one path and is not ambiguous.

Exact variable names and literal contents may vary. Every other AST shape is
ambiguous, including dynamic paths, aliases, multiple paths, bytes operations,
`open`, `os`, subprocesses, comprehensions, loops, conditions, exception
handling, additional calls, omitted replacement count, nonliteral replacement
arguments, or any sixth statement.

## Integration

Introduce two pure helpers in
`phaseD_sft/build_rust_v3_agentic_dataset.py`:

```python
def _literal_python_heredoc_path(command: str) -> str | None:
    """Return the one proven literal write path, else None."""

def _literal_python_heredoc_scope(
    command: str, *, declared_root: str
) -> MutationScope | None:
    """Return a proven scope, or None when the command is unsupported."""
```

`_mutation_scope` invokes the helper only for the exact Python-heredoc shell
envelope. A returned scope participates in the existing full-trajectory
allowlist, forbidden-path, duplicate-command, scratch-chain, and verification
logic. `None` preserves the current ambiguous result. Commands and messages
are never rewritten.

## Tests

Focused tests must first fail against current production code and then prove:

- the exact source shape maps an absolute allowlisted Rust path into
  `repository_paths`;
- an equivalent relative path after a supported leading `cd` maps correctly;
- a literal outside-root path maps into `scratch_paths`;
- exact alternate variable names remain accepted;
- every shell-envelope deviation fails closed;
- every AST deviation listed above fails closed;
- test/example/fixture/lockfile paths are still rejected by the existing
  decisive-suffix gate;
- non-allowlisted paths are still rejected;
- a real compressed fixture containing the exact Python writer reaches the
  same decisive-suffix gates as an equivalent `sed` edit;
- all existing builder and Phase-D tests remain green.

## Live measurement and publication gate

After independent review, rerun audit-only against the frozen source revision,
the exact 239-task exclusion, and boundary 825. Report the new per-reason yield
before building.

Only run the atomic build if audit yield can plausibly satisfy all existing
publication gates:

- at least 60 output rows;
- at least 35 repositories;
- median first edit at most 4;
- maximum 49,152 rendered tokens;
- all eight behavior invariants zero;
- format/loss `failure_count=0`.

If the corrected audit still cannot satisfy row or repository thresholds, do
not publish and do not relax safety rules. Measure alternative revision-pinned
Open-SWE configurations next.

## Operational constraints

- CPU only; no GPU, Docker, serving, or production changes.
- The tokenizer is loaded once only if a passing audit advances to build.
- Token counting uses the existing bounded thread pool, never multiprocessing.
- Existing v3 and failed v3.1 paths are never overwritten.
- All manifests remain independently auditable and fail closed.
