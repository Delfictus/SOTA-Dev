#!/usr/bin/env bash
# [ORCHESTRATOR - TWIN-10 PATENT RUN]
#
# Drives the 6-stage provenance-wrapped pipeline for the 10 TWIN targets.
#
# First-target checkpoint: runs target 1 (KRAS G12D by default) through
# all stages, then invokes twin10_audit.py. If audit overall = PASS,
# proceeds with targets 2-10. If audit = WARN or FAIL, halts for review.
#
# Usage:
#   ./twin10_orchestrator.sh                        # Run target 1 + checkpoint
#   ./twin10_orchestrator.sh --continue-after-checkpoint  # Run all 10
#   ./twin10_orchestrator.sh --only kras_g12d_apo   # Single target
#   ./twin10_orchestrator.sh --no-cupti             # Skip Nsight trace
#   ./twin10_orchestrator.sh --resume <target> --stage 5_engine
#
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
OUTPUT_BASE="${TWIN10_OUTPUT:-/mnt/storage/prism-outputs/twin-10-patent}"
TARGET_CONFIG="${TWIN10_TARGETS:-$SCRIPT_DIR/twin10_targets.json}"

# Flags
CHECKPOINT_ONLY=true
ONLY_TARGET=""
NO_CUPTI=false
RESUME_TARGET=""
RESUME_STAGE=""
DRY_RUN=false

while [[ $# -gt 0 ]]; do
    case "$1" in
        --continue-after-checkpoint) CHECKPOINT_ONLY=false; shift ;;
        --only) ONLY_TARGET="$2"; shift 2 ;;
        --no-cupti) NO_CUPTI=true; shift ;;
        --resume) RESUME_TARGET="$2"; shift 2 ;;
        --stage) RESUME_STAGE="$2"; shift 2 ;;
        --dry-run) DRY_RUN=true; shift ;;
        --output-base) OUTPUT_BASE="$2"; shift 2 ;;
        --targets) TARGET_CONFIG="$2"; shift 2 ;;
        -h|--help)
            grep '^#' "$0" | head -30
            exit 0 ;;
        *)
            echo "Unknown option: $1" >&2
            exit 2 ;;
    esac
done

if [[ ! -f "$TARGET_CONFIG" ]]; then
    echo "FATAL: target config not found: $TARGET_CONFIG" >&2
    echo "Run with --targets <path> or create twin10_targets.json first." >&2
    exit 2
fi

CUPTI_FLAG=""
$NO_CUPTI && CUPTI_FLAG="--no-cupti"

mkdir -p "$OUTPUT_BASE"

BATCH_LOG="$OUTPUT_BASE/batch_$(date +%Y%m%d_%H%M%S).log"
BATCH_SUMMARY="$OUTPUT_BASE/batch_summary.json"

# ─────────────────────────────────────────────────────────────
# Helper: run one target through all 6 stages
# ─────────────────────────────────────────────────────────────
run_one_target() {
    local target_name="$1"
    local target_config_json="$2"
    local target_dir="$OUTPUT_BASE/$target_name"

    echo
    echo "╔══════════════════════════════════════════════════════════════════╗"
    echo "║ TARGET: $target_name"
    echo "║ DIR:    $target_dir"
    echo "╚══════════════════════════════════════════════════════════════════╝"
    mkdir -p "$target_dir"

    local tc_file="$target_dir/target_config.json"
    echo "$target_config_json" > "$tc_file"

    if $DRY_RUN; then
        echo "  [DRY RUN] Would run all 6 stages for $target_name"
        return 0
    fi

    local stages=(1_download 2_clean 3_prep 4_ground_truth 5_engine 6_dcc)

    for stage in "${stages[@]}"; do
        echo
        echo "── stage: $stage"
        python3 "$SCRIPT_DIR/run_stages.py" \
            --target-config "$tc_file" \
            --stage "$stage" \
            --target-dir "$target_dir" \
            $CUPTI_FLAG \
            2>&1 | tee -a "$BATCH_LOG"
        local rc=${PIPESTATUS[0]}
        if [[ $rc -ne 0 ]]; then
            echo "  STAGE $stage FAILED (exit=$rc) — target $target_name BLOCKED"
            echo "{\"target\":\"$target_name\",\"blocked_at\":\"$stage\",\"exit\":$rc}" \
                >> "$BATCH_SUMMARY.fragments"
            return $rc
        fi
    done

    echo "  All 6 stages completed for $target_name"
    return 0
}

# ─────────────────────────────────────────────────────────────
# Helper: run audit on one target
# ─────────────────────────────────────────────────────────────
audit_one_target() {
    local target_name="$1"
    local target_config_json="$2"
    local target_dir="$OUTPUT_BASE/$target_name"
    local audit_out="$target_dir/audit_report.json"

    local known_res=$(echo "$target_config_json" | python3 -c "
import json, sys
t = json.load(sys.stdin)
r = t.get('known_binding_residues') or []
print(','.join(r))
")

    local args=(
        --target-dir "$target_dir"
        --out "$audit_out"
    )
    if [[ -n "$known_res" ]]; then
        args+=(--known-binding-residues "$known_res")
    fi

    python3 "$SCRIPT_DIR/twin10_audit.py" "${args[@]}"
    return $?
}

# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────
echo "TWIN-10 orchestrator"
echo "  repo_root:       $REPO_ROOT"
echo "  target_config:   $TARGET_CONFIG"
echo "  output_base:     $OUTPUT_BASE"
echo "  checkpoint_only: $CHECKPOINT_ONLY"
echo "  no_cupti:        $NO_CUPTI"
echo "  dry_run:         $DRY_RUN"
echo "  batch_log:       $BATCH_LOG"

# Read target list as JSON array
if ! python3 -c "import json; json.load(open('$TARGET_CONFIG'))['targets']" >/dev/null 2>&1; then
    echo "FATAL: target config must be JSON with 'targets' array" >&2
    exit 2
fi

# Iterate targets
TARGET_NAMES=$(python3 -c "
import json
with open('$TARGET_CONFIG') as f:
    cfg = json.load(f)
for t in cfg['targets']:
    print(t['target'])
")

first=true
for target_name in $TARGET_NAMES; do
    if [[ -n "$ONLY_TARGET" && "$target_name" != "$ONLY_TARGET" ]]; then
        continue
    fi

    # Extract this target's config as JSON string
    target_json=$(python3 -c "
import json
with open('$TARGET_CONFIG') as f:
    cfg = json.load(f)
for t in cfg['targets']:
    if t['target'] == '$target_name':
        print(json.dumps(t))
        break
")

    run_one_target "$target_name" "$target_json"
    run_rc=$?

    if [[ $run_rc -eq 0 ]]; then
        echo
        echo "── audit: $target_name"
        audit_one_target "$target_name" "$target_json"
        audit_rc=$?

        if $first && $CHECKPOINT_ONLY; then
            echo
            echo "╔══════════════════════════════════════════════════════════════════╗"
            echo "║ CHECKPOINT-1 AUDIT: target=$target_name audit_rc=$audit_rc"
            echo "║ (0=PASS, 1=FAIL, 2=WARN)"
            echo "║ Stopping per --checkpoint-only. Review audit_report.json."
            echo "║ To continue with targets 2-10, rerun with --continue-after-checkpoint"
            echo "╚══════════════════════════════════════════════════════════════════╝"
            exit $audit_rc
        fi
    fi

    first=false
done

echo
echo "All eligible targets processed. Summary: $BATCH_SUMMARY"
