# v2.11 Clean Promotion and Submit-Fix Design

## Goal

Deliver a trustworthy Gemma-4 teacher-SFT LoRA that beats canonical v2.10 on the same complete SWE-bench Lite 300 population without increasing corrected empty-patch failures. Preserve model-level behavior across serving installations; controller-only recovery cannot establish promotion.

## Current Evidence

- Canonical v2.10 is 157/300 after empty-only correction, with two corrected empty outputs.
- v2.11r3 is rejected at 124/300. It combined a v2.10-initialized reasoning continuation, a 138-row recovery SFT, KTO, and a final merge. The 50-task ablation shows that the post-training lineage materially reduced correctness.
- v2.11r4 is diagnostic-only because its serve was terminated by the memory watchdog and the surviving evaluator manufactured empty outputs.
- The clean candidate uses the raw Gemma base, the exact frozen 1,211-row v2.10 training prefix, and 51 admitted Fable-5 rows. It rejects v2.10 initialization, recovery SFT, KTO, and interpolation.
- The 51 new rows are 4.04% of examples but only about 1.2% of supervised assistant turns and supervised-content size. Thirty-six terminate in supervised prose; fifteen terminate in a supervised tool call.

## Non-Negotiable Constraints

- Use GPU1 only. Never inspect, stop, or schedule work on GPU0.
- Keep the active clean training and post-training chain unchanged until a real completion, safety, or failure boundary.
- Preserve exact dataset, model, invocation, harness, prediction, trajectory, and score provenance.
- Evaluate the candidate on the identical 300 SWE-bench Lite IDs and the same serving/harness contract as v2.10.
- Do not reuse partial output after a serve or watchdog failure.
- Admit Fable rows only with boolean loss flags, supervised assistant behavior, mutation identity, focused passing-test evidence, and replay or equivalent lineage evidence.
- Do not promote on aggregate score alone if corrected empties or paired wrong-edit behavior regress.

## Decision Flow

### Stage 1: Complete the Clean Candidate

The existing automatic chain waits for the 79-step training unit, captures immutable completion evidence, audits the adapter, merges it, runs controller-free portability, validates provenance, and dispatches the identical full300 plus empty-only correction.

The chain must stop on any of these conditions:

- incomplete or mismatched training evidence;
- adapter tensor, rank, or lineage mismatch;
- serving model/template mismatch;
- watchdog or owned-process failure;
- partial predictions or trajectories;
- harness-contract drift from canonical v2.10.

### Stage 2: Promotion Gate

Promote the clean candidate only when all conditions hold:

- portability passes with stock model behavior;
- full300 has 300 predictions and 300 trajectory files over the identical IDs;
- corrected resolved count is greater than 157;
- corrected empty count is at most two;
- no material paired increase in wrong-nonempty edits, tool-format failures, or repeated loops;
- all final artifacts and hashes are complete.

If the candidate passes, publish it as v2.11 and stop. Do not create a successor merely to chase a larger number.

### Stage 3: Failure Classification

If the candidate does not promote, classify every paired movement against v2.10 as one of:

- resolved by both;
- v2.10-only resolved;
- v2.11-only resolved;
- empty or no submission;
- tool-format failure;
- repeated-read or repeated-edit loop;
- wrong non-empty patch;
- harness, image, or infrastructure failure.

The classification chooses exactly one successor path. Do not combine paths before measuring the first correction.

## Successor Paths

### Path A: Submit-Fixed SFT

Use this path when empties, prose finalization, or tool-format failures regress.

- Re-render the 36 Stage-A Fable rows so the final supervised assistant turn is the exact harness-compatible bash submission action.
- Retain failed attempts and explanatory prose only as conditioning with `loss=false`.
- Require the final supervised submission to expose the exact admitted source-only diff.
- Reject scratch files, reproduction scripts, databases, logs, test edits, empty diffs, and patch-hash drift.
- Train from the raw Gemma base with the unchanged 1,211-row v2.10 prefix. Do not add recovery SFT, KTO, or interpolation.

### Path B: Signal-Strength and Diversity SFT

Use this path only when the clean candidate ties v2.10 and behavior does not regress, indicating insufficient rather than harmful Fable signal.

- Add newly available strict Fable traces from repositories underrepresented in the current 51.
- Cap each repository at four added rows.
- Target 3-5% of supervised-token mass from the admitted Fable delta.
- Prefer new verified instances over duplicate-row oversampling.
- Keep the raw-base, one-epoch, single-stage SFT construction.

### Path C: Verified Correction Traces

Use this path when paired failures show loops or wrong edits that are followed by a successful correction in raw evidence.

- Keep wrong, empty, and looping actions as `loss=false` conditioning.
- Supervise only the first verified corrective mutation, focused passing test, final diff, and submission.
- Reject any row whose supervised mutation sequence does not replay to the admitted patch.
- Do not introduce a preference objective until this SFT-only candidate independently clears correctness and portability gates.

## Explicitly Rejected Approaches

- Repeating the full v2.10 corpus from a v2.10-initialized adapter.
- Legacy recovery traces that supervise a wrong edit before verifier feedback.
- KTO or another preference stage before a clean SFT candidate proves correctness.
- Blending a rejected adapter back into v2.10.
- Controller-only retries presented as a model-level improvement.
- Full300 evaluation after a failed serve, watchdog, or incomplete prediction boundary.

## Verification Strategy

- Unit tests cover terminal-turn rendering, loss masks, source-only patch filtering, replay binding, repository caps, supervised-token accounting, and fail-closed manifests.
- A dataset audit proves exact v2.10 prefix equality, unique IDs, zero evaluation overlap, all context lengths within 32,768, and immutable hashes.
- A controller-free portability gate checks valid tool calls, non-empty diffs, loops, and source-file focus before full300.
- The final comparator reports first-pass and corrected scores, predictions, trajectory files, empties, paired movement, behavior classes, and artifact hashes.

## Completion Boundary

The work is complete only when either:

1. clean v2.11 passes the promotion gate and the final model, score, and failure analysis are published; or
2. a single evidence-selected successor passes the same gate and is published.

If neither candidate passes, publish the verified negative result and retain v2.10 as canonical rather than weakening the gate.
