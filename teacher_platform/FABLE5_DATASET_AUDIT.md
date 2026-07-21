# External coding-trace dataset audit (2026-07-20)

## Decision

- Do not bulk-import either audited dataset.
- Stop the Fable lane before full replay or training. The strict evidence is useful only as a tiny
  Python auxiliary candidate, subject to a separately approved A/B experiment.
- Use the SupraLabs corpus only as an index back to revision-pinned upstream sources; reconstruct
  and execution-verify any selected examples before they enter a Gemma training mixture.

## Fable-5 coding and debugging traces

### Pinned provenance

| Item | Value |
| --- | --- |
| Dataset revision | `aef8506515979988aa5c1a423f5b0fb3cee60382` |
| Moonshiner commit | `436316e8f86eb136d5ce3ec95a1a6f48c1d7f940` |
| Source JSONL bytes | `730331947` |
| Source JSONL SHA256 | `ef86c61a8e3b69197d381e2e9b6fe1965005c604fa39ba35e0721457813306c3` |
| License | CC BY 4.0 |

The streaming audit read 12,408 cumulative rows and 2,377 terminal trajectories. Metadata gates
left 243 terminal candidates. Hard exclusion inputs covered the Python Lite 30 slice, hard30,
Multi-SWE Rust 239, and all eight local Multi-SWE C++ family datasets.

### Structural yield

Strict native-Gemma conversion left 46 unique rows: 45 Python, one C++, and zero Rust. The 30-row
format pilot rendered to 1,950-15,572 tokens, had a first-edit median of three, and passed the
Gemma format/loss gate with zero failures. An integration bug found during the run counted
`BatchEncoding` keys instead of `input_ids`; the invalid two-token artifact was preserved and the
counter now has regression coverage.

The zero Rust yield is intentional fail-closed behavior. The terminal Rust traces use shell
pipelines, compound commands, loops, redirection, or other operations outside the deliberately
narrow replay grammar. These must not be admitted by broadly relaxing the parser.

### Replay evidence

The requested two-Python/two-Rust/one-C++ smoke was impossible without weakening the Rust gate. The
explicit fallback was four Python plus the sole C++ row:

| Language | Attempted | Admitted | Finding |
| --- | ---: | ---: | --- |
| Python | 4 | 4 | Baseline failed; gold and candidate passed repeatably |
| Rust | 0 | 0 | No structural survivor |
| C++ | 1 | 0 | Gold/reference also failed `make test` with rc 2 |

Replay artifact: `data/fable5_build/replay_smoke_py4_cpp1_v1`.

The Docker executor was validated live with digest-pinned cached images, no network, a read-only
root filesystem, all capabilities dropped, non-root identities, bounded cgroups, private streamed
inputs/results, protected-file hashes, repeatability runs, and exact cleanup. Final host regression:
755 tests passed.

### Recommendation

Do not replay all 46 or launch training. If a later Python-only experiment is approved, first fix
or remove the invalid C++ control, replay the Python ceiling, cap the admitted rows around one
percent of the training mixture, and require an identical-harness A/B promotion gate. The current
evidence provides no Rust or C++ training value.

## SupraLabs reasoning corpus

The published rows are flattened single-turn distillation records with fields such as `repo_id`,
`tok_len`, `user`, `thought_trace`, `assistant`, and `ChatML`. They do not bind task IDs, commits,
tool calls to observations, executable tests, container images, or outcome rewards. The aggregation
also includes benchmark-derived upstream sources, so direct use creates provenance, deduplication,
and evaluation-contamination risk.

Do not use this corpus directly for agentic SFT, DPO, or RLVR. A safe use is source discovery:
return to the named upstream dataset, pin its revision, apply the project exclusion sets, reconstruct
native Gemma edit-decision messages, and retain only examples whose patches pass an exact executable
verifier. Any resulting auxiliary set should remain small and pass the usual format/loss,
49,152-token, behavior, hard30-floor, and SWE-Lite promotion gates.
