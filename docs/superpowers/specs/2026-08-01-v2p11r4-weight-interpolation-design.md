# v2.11r4 Weight-Interpolation Design

## Objective

Produce one conservative model-level successor that can retain the verified empty/loop and admitted Fable-5 influence from v2.11r3 without repeating its 33-task full300 correctness regression. Promotion still requires beating canonical v2.10 on the same SWE-bench Lite 300 after the same bounded empty-only correction.

## Evidence and choice

Canonical v2.10 resolves 157/300. Final v2.11r3 resolves 124/300 with the same two corrected empties. On the exact 50-task first-pass decision set, v2.10 resolves 38, the pre-KTO recovery checkpoint resolves 23, and final r3 resolves 12. Recovery SFT therefore caused most of the regression and KTO worsened it.

Three successor paths were considered:

1. Stream a merged-weight interpolation between v2.10 and final r3. This is recommended because it retains every realized r3 update direction at reduced strength and costs about 8-12 minutes of CPU/I/O.
2. Retrain a low-rank, low-LR adapter from v2.10. This takes about seven hours before evaluation and can recreate the same drift.
3. Interpolate LoRA A/B factors. This is rejected because factor interpolation creates cross terms and is not equivalent to interpolating effective model weights.

The sole initial candidate is:

`W_v2p11r4 = 0.75 * W_v2p10 + 0.25 * W_v2p11r3`

There is no alpha grid. If this candidate fails the pre-full300 gate, stop this interpolation lane and return to a new low-rank/low-LR training revision.

## Checkpoint construction

A bounded-memory Python utility streams tensors by key from both sharded checkpoints. It must:

- require exact equality for architecture-bearing configuration and tokenizer/template files;
- require identical tensor-key sets and independently resolve each key through each parent's shard index;
- require equal shape and dtype for every paired tensor;
- compute floating-point tensors in FP32 as `anchor + 0.25 * (candidate - anchor)`, then cast to the anchor dtype;
- copy non-floating tensors from v2.10 only after verifying exact equality;
- enforce a peak-RSS ceiling;
- write bounded output shards into a staging directory;
- audit the complete staged checkpoint before atomic publication;
- refuse to overwrite an existing output;
- publish a manifest binding alpha, both resolved parent paths, parent model contracts, copied-file hashes, tensor count, total bytes, and output index/config hashes.

The output model is named `teacher_sft_v2p11r4_blend25` and lives at `/media/ironbcc/CrucialX10/models/merged/teacher_sft_v2p11r4_blend25`.

## Lineage and evaluation

The portability runner gains an `interpolation` lineage mode. It accepts the candidate only when the interpolation manifest reconstructs the current model contract and binds the immutable v2.10 and final-r3 parents. It remains fixed to GPU1 and port 8013 and must reject any GPU0 selection.

Evaluation is controller-free on first pass:

1. Run portability10 against a fresh v2.10 control.
2. Run the immutable fixed150 panel under the same serve and harness contract.
3. Authorize full300 only when candidate resolved is at least v2.10, paired wins are at least losses, wrong-nonempty does not increase, empty and repeat-loop counts do not regress, and at least one of empty or repeat-loop count improves strictly.
4. If authorized, run the same full300 and the existing bounded empty-only correction, then publish the checksum-bound v2.11-versus-v2.10 verdict.

No base, v2, v2.6, v2.7, or other checkpoint is rerun. GPU0 is never queried or touched.

## Failure handling

Any parent mismatch, missing tensor, dtype/shape mismatch, copied-file mismatch, RSS breach, partial output, lineage mismatch, portability regression, or fixed150 regression fails closed. A failed candidate is preserved as diagnostic evidence but is not promoted and does not receive full300.

