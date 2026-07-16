#!/usr/bin/env bash
# Pull a ranked, bounded image window for RLVR state reconstruction.
# Usage: pull_rlvr_cache_window.sh RANKED_TSV OUTDIR [LIMIT [FLOOR_GIB [PROJECT_GIB [INITIAL_AVG_BYTES]]]]
set -euo pipefail

ranked_tsv=${1:?ranked TSV required}
outdir=${2:?output directory required}
limit=${3:-20}
floor_gib=${4:-45}
project_gib=${5:-50}
initial_average_bytes=${6:-0}

mkdir -p "$outdir"
pulls_tsv="$outdir/pulls.tsv"
selected_images="$outdir/selected_images.txt"
log="$outdir/pull.log"
: >"$pulls_tsv"
: >"$selected_images"

available_bytes() {
  df -PB1 /var/lib/docker | tail -n 1 | tr -s ' ' | cut -d ' ' -f4
}

floor_bytes=$((floor_gib * 1024 * 1024 * 1024))
project_bytes=$((project_gib * 1024 * 1024 * 1024))
initial_available=$(available_bytes)
attempted=0
succeeded=0
covered_rows=0
actual_bytes=0
estimated_next_bytes=$initial_average_bytes
stop_reason=complete

while IFS=$'\t' read -r rows image; do
  [[ -n "$rows" && -n "$image" ]] || continue
  (( attempted < limit )) || break
  before=$(available_bytes)
  if (( before < floor_bytes )); then
    stop_reason=pre_pull_floor
    break
  fi
  if (( estimated_next_bytes > 0 && before - estimated_next_bytes <= floor_bytes )); then
    stop_reason=pre_pull_projected_floor
    break
  fi
  attempted=$((attempted + 1))
  set +e
  docker pull "$image" >>"$log" 2>&1
  rc=$?
  set -e
  after=$(available_bytes)
  delta=$((before - after))
  (( delta >= 0 )) || delta=0
  printf '%s\t%s\t%s\t%s\t%s\t%s\n' "$attempted" "$rows" "$image" "$rc" "$delta" "$after" >>"$pulls_tsv"
  if (( rc == 0 )); then
    succeeded=$((succeeded + 1))
    covered_rows=$((covered_rows + rows))
    actual_bytes=$((actual_bytes + delta))
    estimated_next_bytes=$((actual_bytes / succeeded))
    printf '%s\n' "$image" >>"$selected_images"
  fi
  if (( after < floor_bytes )); then
    stop_reason=post_pull_floor
    break
  fi
  if (( attempted == 5 && succeeded > 0 )); then
    projected=$((actual_bytes * limit / succeeded))
    if (( projected > project_bytes )); then
      stop_reason=projection_exceeds_limit
      break
    fi
  fi
done <"$ranked_tsv"

final_available=$(available_bytes)
printf 'stop_reason=%s attempted=%s succeeded=%s covered_rows=%s actual_bytes=%s initial_available=%s final_available=%s\n' \
  "$stop_reason" "$attempted" "$succeeded" "$covered_rows" "$actual_bytes" "$initial_available" "$final_available" \
  | tee "$outdir/summary.txt"
