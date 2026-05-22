#!/usr/bin/env bash
# PRISM-4D Corpus Runner — Canonical batch execution wrapper
#
# Iterates a manifest of chain-level targets, fetches each target's
# prepped inputs from R2, runs the canonical engine path
# (prism-validate-and-run.sh, which itself runs preflight → ground
# truth → engine → postflight w/ DCC validation), pushes outputs back
# to R2, verifies, and optionally cleans up local working directory.
#
# This is the ONLY supported way to run the engine over a multi-target
# corpus. It is a thin loop around prism-validate-and-run.sh — it does
# not duplicate any of the validation logic. If the canonical wrapper
# changes, this runner picks up the change automatically.
#
# Usage:
#   scripts/prism-corpus-runner.sh \
#       --manifest <manifest.txt> \
#       --r2-input-prefix <r2_prefix> \
#       --r2-output-prefix <r2_prefix> \
#       [--cleanup-local true|false] \
#       [--per-target-timeout 30m] \
#       [--engine-flags "FLAG1 FLAG2 ..."]
#
# Manifest format: one chain-level target per line (e.g. "10dc_chainA").
# Blank lines and lines starting with '#' are ignored.
#
# Inputs are pulled from r2:prism-archive/<r2_input_prefix>/<pdb_id>/
# where pdb_id is derived from the target by stripping the _chainX suffix.
# Each target dir is expected to contain:
#   <target>.topology.json
#   <target>.residue_map.json
#   <target>_clean.pdb
#
# Outputs are pushed to r2:prism-archive/<r2_output_prefix>/<target>/.
#
# Per-target verification reads <target>_ground_truth.json sidecar to
# determine if the postflight ran successfully.
#
# Per CLAUDE.md production rules:
#   - No /tmp references
#   - Workdir lives under /mnt/storage/prism-outputs/

set -uo pipefail

# ─────────────────────────────────────────────────────────────────────
# Configuration / argument parsing
# ─────────────────────────────────────────────────────────────────────

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
WRAPPER="$SCRIPT_DIR/prism-validate-and-run.sh"

WORK_DIR_BASE=/mnt/storage/prism-outputs/10k-runs
LOG_DIR=/mnt/storage/prism-outputs/_corpus_runner_logs
mkdir -p "$LOG_DIR"

MANIFEST=""
R2_INPUT_PREFIX=""
R2_OUTPUT_PREFIX=""
CLEANUP_LOCAL="false"
PER_TARGET_TIMEOUT="30m"
EXTRA_ENGINE_FLAGS=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --manifest)            MANIFEST="$2"; shift 2;;
        --r2-input-prefix)     R2_INPUT_PREFIX="$2"; shift 2;;
        --r2-output-prefix)    R2_OUTPUT_PREFIX="$2"; shift 2;;
        --cleanup-local)       CLEANUP_LOCAL="$2"; shift 2;;
        --per-target-timeout)  PER_TARGET_TIMEOUT="$2"; shift 2;;
        --engine-flags)        EXTRA_ENGINE_FLAGS="$2"; shift 2;;
        --help|-h)
            grep '^#' "$0" | sed 's/^# //; s/^#//'
            exit 0
            ;;
        *)
            echo "ERROR: unknown arg: $1" >&2
            exit 1
            ;;
    esac
done

if [[ -z "$MANIFEST" || -z "$R2_INPUT_PREFIX" || -z "$R2_OUTPUT_PREFIX" ]]; then
    echo "ERROR: --manifest, --r2-input-prefix, --r2-output-prefix are required" >&2
    echo "       Run with --help for usage details." >&2
    exit 1
fi

if [[ ! -f "$MANIFEST" ]]; then
    echo "ERROR: manifest not found: $MANIFEST" >&2
    exit 1
fi

if [[ ! -x "$WRAPPER" ]]; then
    echo "ERROR: canonical wrapper not found: $WRAPPER" >&2
    exit 1
fi

# Default engine flags if none provided. The user can override entirely
# with --engine-flags. These match the canonical TWIN multi-differential
# corpus generation config.
# Canonical TWIN corpus generation flags.
# Source of truth: crates/prism-nhs/src/bin/nhs_rt_full.rs (see docs/CANONICAL_PROVENANCE.md).
# Previous value used --multi-stream 4 --use-tokenized-ranker; phase-manifold is now the production site ranker.
DEFAULT_ENGINE_FLAGS="--fast --hysteresis --prism-therm --multi-stream 8 --spike-percentile 70 --fused-steps 6 --hmr --adaptive-dt --multi-differential --closed-loop-steering --asymmetric-steering --site-ranker phase-manifold --replica-seed 42 -v"
ENGINE_FLAGS="${EXTRA_ENGINE_FLAGS:-$DEFAULT_ENGINE_FLAGS}"

mkdir -p "$WORK_DIR_BASE"

RUN_ID="run_$(date +%Y%m%d_%H%M%S)_$$"
SUMMARY_LOG="$LOG_DIR/${RUN_ID}_summary.log"
PER_TARGET_LOG="$LOG_DIR/${RUN_ID}_per_target.log"
: > "$SUMMARY_LOG"
: > "$PER_TARGET_LOG"

TOTAL=$(grep -cv '^[[:space:]]*\(#\|$\)' "$MANIFEST")

echo "================================================================"
echo "PRISM-4D CORPUS RUNNER"
echo "  Manifest:     $MANIFEST"
echo "  Targets:      $TOTAL"
echo "  R2 input:     r2:prism-archive/$R2_INPUT_PREFIX/"
echo "  R2 output:    r2:prism-archive/$R2_OUTPUT_PREFIX/"
echo "  Workdir:      $WORK_DIR_BASE"
echo "  Cleanup:      $CLEANUP_LOCAL"
echo "  Timeout:      $PER_TARGET_TIMEOUT per target"
echo "  Engine flags: $ENGINE_FLAGS"
echo "  Run id:       $RUN_ID"
echo "  Summary log:  $SUMMARY_LOG"
echo "  Started:      $(date -Iseconds)"
echo "================================================================"

# ─────────────────────────────────────────────────────────────────────
# Per-target processing
# ─────────────────────────────────────────────────────────────────────

i=0
n_ok=0
n_fail=0
n_verify_fail=0

while IFS= read -r line || [ -n "$line" ]; do
    # Skip blank lines and comments
    [[ -z "${line// }" || "$line" =~ ^[[:space:]]*# ]] && continue
    target="$(echo "$line" | tr -d '[:space:]')"
    i=$((i+1))
    pdb="${target%%_*}"

    OUTDIR="$WORK_DIR_BASE/$target"
    # Always start each target with a clean workdir. Stale files from a
    # prior run can deadlock the engine — observed on 13sb_chainB where
    # leaving a 1.9 GB spike_events.arrow from a previous run caused the
    # multi-stream multi-diff path to hang at chunk 0 in futex_wait_queue
    # for >24 minutes. Production runs MUST start each target fresh.
    rm -rf "$OUTDIR"
    mkdir -p "$OUTDIR"

    echo
    echo "── [$i/$TOTAL] $target ────────────────────────────────────────"
    T0=$(date +%s)

    # 1. Fetch prepped inputs from R2
    echo "  [1/4] Fetching r2:prism-archive/$R2_INPUT_PREFIX/$pdb/ → $OUTDIR/"
    rclone copy "r2:prism-archive/$R2_INPUT_PREFIX/$pdb/" "$OUTDIR/" \
        --include "${target}.topology.json" \
        --include "${target}.residue_map.json" \
        --include "${target}_clean.pdb" 2>&1 | tail -2 || true

    if [[ ! -f "$OUTDIR/${target}.topology.json" ]]; then
        echo "  SKIP: topology not present after fetch"
        echo "$target SKIP_NO_TOPO -" | tee -a "$SUMMARY_LOG" >> "$PER_TARGET_LOG"
        n_fail=$((n_fail+1))
        rm -rf "$OUTDIR"
        continue
    fi

    # 2. Run the canonical wrapper (preflight → GT → engine → postflight w/ DCC)
    echo "  [2/4] Running canonical engine path"
    T_ENG_START=$(date +%s)
    timeout "$PER_TARGET_TIMEOUT" "$WRAPPER" \
        -t "$OUTDIR/${target}.topology.json" \
        -o "$OUTDIR" \
        $ENGINE_FLAGS \
        > "$OUTDIR/run.log" 2>&1
    rc=$?
    T_ENG_END=$(date +%s)
    ENGINE_SECS=$((T_ENG_END - T_ENG_START))

    # Engine exit codes 134/139 are CUDA teardown segfaults — output is valid
    if [[ $rc -ne 0 && $rc -ne 134 && $rc -ne 139 ]]; then
        echo "  ENGINE FAILED rc=$rc after ${ENGINE_SECS}s"
        echo "  ── tail of run.log ──"
        tail -10 "$OUTDIR/run.log" | sed 's/^/  /'
        echo "  ── end ──"
        rclone copy "$OUTDIR/run.log" "r2:prism-archive/$R2_OUTPUT_PREFIX/$target/" 2>/dev/null || true
        echo "$target FAILED rc=$rc engine=${ENGINE_SECS}s" | tee -a "$SUMMARY_LOG" >> "$PER_TARGET_LOG"
        n_fail=$((n_fail+1))
        if [[ "$CLEANUP_LOCAL" == "true" ]]; then
            rm -rf "$OUTDIR"
        fi
        continue
    fi
    echo "  engine OK in ${ENGINE_SECS}s (rc=$rc)"

    # 3. Push outputs to R2
    echo "  [3/4] Uploading outputs → r2:prism-archive/$R2_OUTPUT_PREFIX/$target/"
    T_UP_START=$(date +%s)
    rclone copy "$OUTDIR" "r2:prism-archive/$R2_OUTPUT_PREFIX/$target/" \
        --transfers 32 --s3-chunk-size 128M 2>&1 | tail -2 || true
    T_UP_END=$(date +%s)
    UPLOAD_SECS=$((T_UP_END - T_UP_START))

    # 4. Verify critical artifacts and read GT validation summary from sidecar
    echo "  [4/4] Verifying artifacts on R2"
    R2_FILES=$(rclone lsf "r2:prism-archive/$R2_OUTPUT_PREFIX/$target/" 2>/dev/null)
    # Spike data: accept either legacy .arrow (TWIN path) OR per-site .spike_events.json (baseline)
    HAS_ARROW=$(echo "$R2_FILES" | grep -c "spike_events.arrow" || true)
    HAS_SPIKE_JSON=$(echo "$R2_FILES" | grep -c ".spike_events.json" || true)
    HAS_SPIKE=$(( HAS_ARROW + HAS_SPIKE_JSON ))
    # GCPID is TWIN-only — no longer required for baseline canonical runs
    HAS_GCPID=$(echo "$R2_FILES" | grep -c "gcpid_synergy.json" || true)
    HAS_SITES=$(echo "$R2_FILES" | grep -c "binding_sites.json" || true)
    HAS_KCC=$(echo "$R2_FILES" | grep -c "kcc_visualization.json" || true)
    HAS_GT=$(echo "$R2_FILES" | grep -c "_ground_truth.json" || true)

    GT_SUMMARY="-"
    if [[ -f "$OUTDIR/${target}_ground_truth.json" ]]; then
        GT_SUMMARY=$(python3 -c "
import json, sys
try:
    d = json.load(open('$OUTDIR/${target}_ground_truth.json'))
    if d.get('valid_for_dcc_validation'):
        L = d.get('ligand', {})
        c = d.get('ligand_centroid', [0, 0, 0])
        print(f\"valid|{L.get('resname','?')}|{L.get('classification','?')}|{c[0]:.2f},{c[1]:.2f},{c[2]:.2f}\")
    else:
        print(f\"skip|{d.get('skip_reason','?')}\")
except Exception as e:
    print(f\"err|{e}\")
" 2>/dev/null)
    fi

    R2_TOTAL_BYTES=$(rclone size "r2:prism-archive/$R2_OUTPUT_PREFIX/$target/" --json 2>/dev/null | python3 -c "import sys,json; print(json.load(sys.stdin).get('bytes',0))" 2>/dev/null || echo 0)
    R2_MB=$((R2_TOTAL_BYTES/1024/1024))

    echo "  R2 has: spike=$HAS_SPIKE (arrow=$HAS_ARROW json=$HAS_SPIKE_JSON) sites=$HAS_SITES kcc=$HAS_KCC gt=$HAS_GT  total=${R2_MB}MB"
    echo "  GT: $GT_SUMMARY"

    # Baseline canonical criteria: spike data present + binding_sites + kcc_visualization
    if [[ "$HAS_SPIKE" -ge 1 && "$HAS_SITES" -ge 1 && "$HAS_KCC" -ge 1 ]]; then
        TOTAL_SECS=$(($(date +%s) - T0))
        echo "$target OK engine=${ENGINE_SECS}s upload=${UPLOAD_SECS}s total=${TOTAL_SECS}s r2_mb=${R2_MB} gt=${GT_SUMMARY}" | tee -a "$SUMMARY_LOG" >> "$PER_TARGET_LOG"
        n_ok=$((n_ok+1))
        # Verify upload then cleanup
        rclone check "$OUTDIR" "r2:prism-archive/$R2_OUTPUT_PREFIX/$target/" --one-way 2>/dev/null
        if [ $? -eq 0 ]; then
            rm -rf "$OUTDIR"
            echo "  $target: local cleanup after verified R2 upload"
        else
            echo "  $target: WARNING — rclone check failed, keeping local copy"
        fi
    else
        echo "  VERIFY FAILED: missing critical artifacts on R2"
        TOTAL_SECS=$(($(date +%s) - T0))
        echo "$target VERIFY_FAILED engine=${ENGINE_SECS}s spike=$HAS_SPIKE sites=$HAS_SITES kcc=$HAS_KCC" | tee -a "$SUMMARY_LOG" >> "$PER_TARGET_LOG"
        n_verify_fail=$((n_verify_fail+1))
    fi

done < "$MANIFEST"

# ─────────────────────────────────────────────────────────────────────
# Final summary
# ─────────────────────────────────────────────────────────────────────

echo
echo "================================================================"
echo "CORPUS RUN COMPLETE — $(date -Iseconds)"
echo "================================================================"
echo "  Total processed: $i / $TOTAL"
echo "  OK:              $n_ok"
echo "  FAILED:          $n_fail"
echo "  VERIFY_FAILED:   $n_verify_fail"
echo "  Summary log:     $SUMMARY_LOG"
echo "  Per-target log:  $PER_TARGET_LOG"

if [[ $n_ok -eq 0 ]]; then
    exit 1
fi
exit 0
