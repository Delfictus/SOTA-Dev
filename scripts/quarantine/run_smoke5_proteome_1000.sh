#!/usr/bin/env bash
# QUARANTINED 2026-04-21 — uses pre-lockdown canonical.
# For current canonical, see CLAUDE.md §B or scripts/prism-validate-and-run.sh header.
# This script is preserved for historical reproduction only.
# 5-target smoke test for the proteome_1000 corpus generation campaign.
#
# Validates the full fetch → run → push → cleanup loop on a small set
# before launching the full 1098-target sweep.
#
# Engine: nhs_rt_full --multi-differential --multi-stream 8 --emit-spike-json false
# (Arrow IPC writer produces spike_events.arrow — no JSON needed)

set -uo pipefail

WORK_DIR=/mnt/storage/prism-outputs/10k-runs
MANIFEST=/tmp/smoke4.txt
SUMMARY_LOG=/tmp/smoke4_summary.log
R2_OUT=r2:prism-archive/10k-runs

mkdir -p "$WORK_DIR"
: > "$SUMMARY_LOG"

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
WRAPPER="$PROJECT_DIR/scripts/prism-validate-and-run.sh"

if [ ! -x "$WRAPPER" ]; then
    echo "FATAL: $WRAPPER not found or not executable" >&2
    exit 1
fi

if [ ! -f "$MANIFEST" ]; then
    echo "FATAL: manifest $MANIFEST not found" >&2
    exit 1
fi

TOTAL=$(wc -l < "$MANIFEST")
echo "================================================================"
echo "PROTEOME_1000 5-TARGET SMOKE TEST"
echo "Targets: $TOTAL"
echo "Workdir: $WORK_DIR"
echo "R2 out:  $R2_OUT/"
echo "Started: $(date -Iseconds)"
echo "================================================================"

i=0
declare -a RESULTS=()

while IFS= read -r target || [ -n "$target" ]; do
    [ -z "$target" ] && continue
    i=$((i+1))
    pdb="${target%%_*}"

    OUTDIR="$WORK_DIR/$target"
    mkdir -p "$OUTDIR"

    echo
    echo "── [$i/$TOTAL] $target ────────────────────────────────────────"
    T0=$(date +%s)

    # 1. Fetch prepped inputs from R2 (small, ~1-2 MB total)
    echo "  [1/4] Fetching topology from r2:prism-archive/proteome_1000/$pdb/"
    rclone copy "r2:prism-archive/proteome_1000/$pdb/" "$OUTDIR/" \
        --include "${target}.topology.json" \
        --include "${target}.residue_map.json" \
        --include "${target}_clean.pdb" 2>&1 | tail -3 || true

    if [ ! -f "$OUTDIR/${target}.topology.json" ]; then
        echo "  SKIP: topology not present after rclone copy"
        RESULTS+=("$target SKIP_NO_TOPO -")
        rm -rf "$OUTDIR"
        continue
    fi
    INPUT_BYTES=$(stat -c%s "$OUTDIR/${target}.topology.json")
    echo "  topology size: $((INPUT_BYTES/1024)) KB"

    # 2. Run engine — multi-differential, no spike JSON (Arrow IPC instead)
    echo "  [2/4] Running engine (--multi-differential --multi-stream 8 --emit-spike-json false)"
    T_ENGINE_START=$(date +%s)
    timeout 30m "$WRAPPER" \
        -t "$OUTDIR/${target}.topology.json" \
        -o "$OUTDIR" \
        --fast --hysteresis \
        --multi-stream 8 \
        --multi-differential \
        --spike-percentile 95 \
        --prism-therm \
        --fused-steps 4 \
        --hmr \
        --adaptive-dt \
        --emit-spike-json false \
        --replica-seed 42 \
        -v \
        > "$OUTDIR/run.log" 2>&1
    rc=$?
    T_ENGINE_END=$(date +%s)
    ENGINE_SECS=$((T_ENGINE_END - T_ENGINE_START))

    if [ $rc -ne 0 ]; then
        echo "  ENGINE FAILED rc=$rc after ${ENGINE_SECS}s"
        echo "  ── tail of run.log ──"
        tail -15 "$OUTDIR/run.log" | sed 's/^/  /'
        echo "  ── end log tail ──"
        rclone copy "$OUTDIR/run.log" "$R2_OUT/$target/" 2>/dev/null || true
        RESULTS+=("$target FAILED_rc${rc} ${ENGINE_SECS}s")
        rm -rf "$OUTDIR"
        continue
    fi
    echo "  engine OK in ${ENGINE_SECS}s"

    # 3. Push everything to R2
    echo "  [3/4] Uploading outputs to $R2_OUT/$target/"
    T_UP_START=$(date +%s)
    rclone copy "$OUTDIR" "$R2_OUT/$target/" \
        --transfers 32 --s3-chunk-size 128M 2>&1 | tail -3 || true
    T_UP_END=$(date +%s)
    UPLOAD_SECS=$((T_UP_END - T_UP_START))

    # 4. Verify critical artifacts on R2 — CORRECTED for multi-stream multi-diff path
    #    The --multi-stream 8 --multi-differential code path does NOT produce
    #    multi_differential_result.json (that's only the 4x1 dispatch). Instead
    #    it writes modular per-output files. Verify the three that matter:
    #      *.spike_events.arrow   ← Stage 1B-1, the foundation-model substrate
    #      *.gcpid_synergy.json   ← Stage 1B-3, per-residue synergy_fraction
    #      *.binding_sites.json   ← detected sites + classifications
    echo "  [4/4] Verifying R2 has critical artifacts..."
    R2_FILES=$(rclone lsf "$R2_OUT/$target/" 2>/dev/null)
    HAS_ARROW=$(echo "$R2_FILES" | grep -c "spike_events.arrow" || true)
    HAS_GCPID=$(echo "$R2_FILES" | grep -c "gcpid_synergy.json" || true)
    HAS_SITES=$(echo "$R2_FILES" | grep -c "binding_sites.json" || true)
    HAS_THERM=$(echo "$R2_FILES" | grep -c "prism_therm" || true)
    HAS_ASC=$(echo "$R2_FILES" | grep -c "asc_consensus" || true)

    echo "  R2 has: arrow=$HAS_ARROW gcpid=$HAS_GCPID sites=$HAS_SITES therm=$HAS_THERM asc=$HAS_ASC"

    if [ "$HAS_ARROW" -ge 1 ] && [ "$HAS_GCPID" -ge 1 ] && [ "$HAS_SITES" -ge 1 ]; then
        ARROW_BYTES=$(rclone size "$R2_OUT/$target/" --json 2>/dev/null | python3 -c "import sys,json; print(json.load(sys.stdin).get('bytes',0))" 2>/dev/null || echo 0)
        # NOTE: smoke phase keeps local copies for inspection. For full corpus run, uncomment:
        # rm -rf "$OUTDIR"
        echo "  KEEPING local for inspection (smoke phase) — R2 total $((ARROW_BYTES/1024/1024)) MB"
        TOTAL_SECS=$(($(date +%s) - T0))
        RESULTS+=("$target OK engine=${ENGINE_SECS}s upload=${UPLOAD_SECS}s total=${TOTAL_SECS}s r2_mb=$((ARROW_BYTES/1024/1024))")
    else
        echo "  KEEPING LOCAL — R2 verify failed (arrow=$HAS_ARROW gcpid=$HAS_GCPID sites=$HAS_SITES)"
        TOTAL_SECS=$(($(date +%s) - T0))
        RESULTS+=("$target VERIFY_FAILED engine=${ENGINE_SECS}s upload=${UPLOAD_SECS}s total=${TOTAL_SECS}s")
    fi

done < "$MANIFEST"

echo
echo "================================================================"
echo "SMOKE TEST COMPLETE — $(date -Iseconds)"
echo "================================================================"
for r in "${RESULTS[@]}"; do
    echo "$r"
done | tee -a "$SUMMARY_LOG"
echo "Summary written to $SUMMARY_LOG"
