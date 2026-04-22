#!/usr/bin/env bash
# Batch Arrow IPC → Parquet+zstd conversion for R2 corpus targets.
#
# Downloads each Arrow file from R2, converts to Parquet+zstd via
# arrow_to_parquet.py, uploads Parquet back, verifies, then deletes
# the Arrow original from R2.
#
# Runs independently of the corpus runner. Does NOT touch targets
# currently being processed by the runner.

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONVERTER="$SCRIPT_DIR/arrow_to_parquet.py"
WORK_DIR=/mnt/storage/parquet-batch
LOG_FILE="$WORK_DIR/conversion_log.txt"
R2_PREFIX="r2:prism-archive/10k-runs"

mkdir -p "$WORK_DIR"
: > "$LOG_FILE"

echo "================================================================"
echo "BATCH ARROW → PARQUET+ZSTD CONVERSION"
echo "  Work dir:   $WORK_DIR"
echo "  R2 prefix:  $R2_PREFIX"
echo "  Converter:  $CONVERTER"
echo "  Started:    $(date -Iseconds)"
echo "================================================================"

# Get list of all Arrow files on R2
ARROW_LIST="$WORK_DIR/arrow_inventory.txt"
rclone ls "$R2_PREFIX/" --include "*.spike_events.arrow" 2>/dev/null | awk '{print $2}' | sort > "$ARROW_LIST"
TOTAL=$(wc -l < "$ARROW_LIST")
echo "  Found $TOTAL Arrow files on R2"
echo

n_ok=0
n_fail=0
bytes_arrow_total=0
bytes_parquet_total=0

i=0
while IFS= read -r rel_path; do
    [ -z "$rel_path" ] && continue
    i=$((i+1))

    # Parse target name from path: 10dc_chainA/10dc_chainA.topology.spike_events.arrow
    target=$(echo "$rel_path" | cut -d/ -f1)
    filename=$(basename "$rel_path")
    parquet_filename="${filename%.arrow}.parquet"

    echo "── [$i/$TOTAL] $target ──────────────────────────────────"

    TARGET_DIR="$WORK_DIR/$target"
    rm -rf "$TARGET_DIR"
    mkdir -p "$TARGET_DIR"

    # 1. Download Arrow from R2
    echo "  [1/6] Downloading $rel_path from R2..."
    T0=$(date +%s)
    rclone copy "$R2_PREFIX/$target/$filename" "$TARGET_DIR/" 2>&1 | tail -2 || true

    if [ ! -f "$TARGET_DIR/$filename" ]; then
        echo "  FAILED: download failed"
        echo "$target DOWNLOAD_FAILED" >> "$LOG_FILE"
        n_fail=$((n_fail+1))
        rm -rf "$TARGET_DIR"
        continue
    fi

    ARROW_BYTES=$(stat -c%s "$TARGET_DIR/$filename")

    # 2. Convert Arrow → Parquet+zstd
    echo "  [2/6] Converting (timeout 5m)..."
    timeout 5m python3 "$CONVERTER" "$TARGET_DIR/$filename" > "$TARGET_DIR/convert.log" 2>&1
    rc=$?

    if [ $rc -ne 0 ]; then
        echo "  FAILED: conversion failed (rc=$rc)"
        tail -5 "$TARGET_DIR/convert.log" | sed 's/^/    /'
        echo "$target CONVERT_FAILED rc=$rc arrow_bytes=$ARROW_BYTES" >> "$LOG_FILE"
        n_fail=$((n_fail+1))
        rm -rf "$TARGET_DIR"
        continue
    fi

    if [ ! -f "$TARGET_DIR/$parquet_filename" ]; then
        echo "  FAILED: Parquet file not produced"
        echo "$target PARQUET_MISSING arrow_bytes=$ARROW_BYTES" >> "$LOG_FILE"
        n_fail=$((n_fail+1))
        rm -rf "$TARGET_DIR"
        continue
    fi

    PARQUET_BYTES=$(stat -c%s "$TARGET_DIR/$parquet_filename")
    RATIO=$(python3 -c "print(f'{$ARROW_BYTES/$PARQUET_BYTES:.2f}')")
    echo "  [3/6] Conversion OK: $((ARROW_BYTES/1024/1024)) MB → $((PARQUET_BYTES/1024/1024)) MB (${RATIO}×)"

    # 3. Upload Parquet to R2
    echo "  [4/6] Uploading Parquet to $R2_PREFIX/$target/"
    rclone copy "$TARGET_DIR/$parquet_filename" "$R2_PREFIX/$target/" --transfers 16 --s3-chunk-size 128M 2>&1 | tail -2 || true

    # 4. Verify Parquet on R2
    R2_CHECK=$(rclone ls "$R2_PREFIX/$target/" --include "$parquet_filename" 2>/dev/null)
    if [ -z "$R2_CHECK" ]; then
        echo "  FAILED: Parquet not found on R2 after upload"
        echo "$target UPLOAD_FAILED arrow_bytes=$ARROW_BYTES parquet_bytes=$PARQUET_BYTES" >> "$LOG_FILE"
        n_fail=$((n_fail+1))
        rm -rf "$TARGET_DIR"
        continue
    fi
    echo "  [5/6] Parquet verified on R2 ✓"

    # 5. Delete Arrow from R2
    rclone deletefile "$R2_PREFIX/$target/$filename" 2>&1 || true

    # 6. Verify Arrow deleted
    ARROW_GONE=$(rclone ls "$R2_PREFIX/$target/" --include "$filename" 2>/dev/null)
    if [ -n "$ARROW_GONE" ]; then
        echo "  WARNING: Arrow file still on R2 after deletefile"
        echo "$target ARROW_DELETE_FAILED arrow_bytes=$ARROW_BYTES parquet_bytes=$PARQUET_BYTES ratio=$RATIO" >> "$LOG_FILE"
    else
        echo "  [6/6] Arrow deleted from R2 ✓"
    fi

    T1=$(date +%s)
    ELAPSED=$((T1 - T0))
    echo "  OK in ${ELAPSED}s — saved $((( ARROW_BYTES - PARQUET_BYTES ) / 1024 / 1024)) MB"
    echo "$target OK arrow_bytes=$ARROW_BYTES parquet_bytes=$PARQUET_BYTES ratio=$RATIO elapsed=${ELAPSED}s" >> "$LOG_FILE"

    bytes_arrow_total=$((bytes_arrow_total + ARROW_BYTES))
    bytes_parquet_total=$((bytes_parquet_total + PARQUET_BYTES))
    n_ok=$((n_ok+1))

    # Clean up local working dir for this target
    rm -rf "$TARGET_DIR"

done < "$ARROW_LIST"

echo
echo "================================================================"
echo "BATCH CONVERSION COMPLETE — $(date -Iseconds)"
echo "================================================================"
echo "  Converted: $n_ok / $TOTAL"
echo "  Failed:    $n_fail"
echo "  Arrow total:   $((bytes_arrow_total / 1024 / 1024)) MB"
echo "  Parquet total: $((bytes_parquet_total / 1024 / 1024)) MB"
if [ $bytes_parquet_total -gt 0 ]; then
    echo "  Overall ratio: $(python3 -c "print(f'{$bytes_arrow_total/$bytes_parquet_total:.2f}')") ×"
fi
echo "  Saved:     $(( (bytes_arrow_total - bytes_parquet_total) / 1024 / 1024 )) MB"
echo "  Log:       $LOG_FILE"

# Final verification
echo
echo "=== R2 FINAL STATE ==="
echo "Parquet files:"
rclone ls "$R2_PREFIX/" --include "*.spike_events.parquet" 2>/dev/null | wc -l
echo "Arrow files (should be 0):"
rclone ls "$R2_PREFIX/" --include "*.spike_events.arrow" 2>/dev/null | wc -l
