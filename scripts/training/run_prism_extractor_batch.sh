#!/usr/bin/env bash
# Batch driver for prism_only_feature_extractor_v2.py across all local TWIN targets.
# Runs MAX_PARALLEL extractions concurrently. Each target needs its companion JSONs
# (binding_sites, kcc_visualization, topology.prism_therm) — they live next to the .arrow.
#
# Usage: scripts/training/run_prism_extractor_batch.sh [MAX_PARALLEL=2]

set -u

MAX_PARALLEL="${1:-2}"
OUT_DIR="/home/diddy/prism4d_training/prism_only_features"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXTRACTOR="$SCRIPT_DIR/prism_only_feature_extractor_v3_gpu.py"
LOG_DIR="/home/diddy/prism4d_training/extractor_logs"

mkdir -p "$OUT_DIR" "$LOG_DIR"

# Search roots for TWIN-density arrow files
SEARCH_ROOTS=(
    "/mnt/storage/prism-outputs/hect-family"
    "/mnt/storage/diddy_recovery_archive/m1_strict_dcc"
    "/mnt/storage/prism-outputs/m1-strict-dcc-panel"
    "/mnt/storage/prism-outputs/runs"
    "/mnt/storage/prism-outputs/blind_validation"
)

TARGETS=()
for root in "${SEARCH_ROOTS[@]}"; do
    [ -d "$root" ] || continue
    while IFS= read -r arrow_path; do
        [ -z "$arrow_path" ] && continue
        TARGETS+=("$arrow_path")
    done < <(find "$root" -name "*.topology.spike_events.arrow" -size +1G 2>/dev/null)
done

echo "found ${#TARGETS[@]} TWIN-density arrow files"
echo "output: $OUT_DIR"
echo "logs:   $LOG_DIR"
echo "max parallel: $MAX_PARALLEL"
echo ""

extract_one() {
    local arrow="$1"
    local dir
    dir="$(dirname "$arrow")"
    local base
    base="$(basename "$arrow" .topology.spike_events.arrow)"
    local parent
    parent="$(basename "$(dirname "$(dirname "$arrow")")")"
    local key="${base}_${parent}"
    local out="$OUT_DIR/${key}.parquet"
    local log="$LOG_DIR/${key}.log"

    if [ -f "$out" ]; then
        echo "[skip] $key (already exists)"
        return 0
    fi

    local bsites="${dir}/${base}.binding_sites.json"
    local kcc="${dir}/${base}.kcc_visualization.json"
    local therm="${dir}/${base}.topology.prism_therm.json"
    local asc="${dir}/${base}.topology.asc_consensus.json"
    local gcpid="${dir}/${base}.topology.gcpid_synergy.json"
    local dpdb="${dir}/${base}.topology.druggability.pdb"

    local args=("--arrow" "$arrow" "--output" "$out")
    [ -f "$bsites" ] && args+=("--binding-sites" "$bsites")
    [ -f "$kcc" ] && args+=("--kcc" "$kcc")
    [ -f "$therm" ] && args+=("--therm" "$therm")
    [ -f "$asc" ] && args+=("--asc-consensus" "$asc")
    [ -f "$gcpid" ] && args+=("--gcpid" "$gcpid")
    [ -f "$dpdb" ] && args+=("--druggability-pdb" "$dpdb")

    echo "[start] $key  ($(date +%H:%M:%S))"
    python3 "$EXTRACTOR" "${args[@]}" > "$log" 2>&1
    local rc=$?
    if [ $rc -eq 0 ]; then
        echo "[done ] $key  ($(date +%H:%M:%S)) — $(du -h "$out" 2>/dev/null | cut -f1)"
    else
        echo "[FAIL ] $key  rc=$rc  (see $log)"
    fi
    return $rc
}

export -f extract_one
export OUT_DIR LOG_DIR EXTRACTOR

# Drive parallelism with a simple background-pool
running=0
fail_count=0
done_count=0
for arrow in "${TARGETS[@]}"; do
    while [ "$running" -ge "$MAX_PARALLEL" ]; do
        wait -n
        running=$((running - 1))
    done
    extract_one "$arrow" &
    running=$((running + 1))
done
wait

echo ""
echo "=== summary ==="
ls -la "$OUT_DIR" 2>/dev/null | tail -25
