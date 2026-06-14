# Decontamination (mandatory before any training source is used)

Hard gate (PLAN §13): any overlap with SWE-bench Pro's 41 repos or Verified voids the headline number.

Procedure:
1. Drop any task whose source repo ∈ {Pro 41 ∪ Verified}.
2. 13-gram problem-statement overlap check — reuse **open-r1 `decontaminate.py`** (don't reinvent):
   https://github.com/huggingface/open-r1
3. Write the dropped-count manifest to `data/manifests/decontam_<source>_<date>.json`
   (fields: source, n_in, n_dropped_repo, n_dropped_ngram, n_out, pro_verified_snapshot).
4. Commit the manifest (raw data stays git-ignored).
