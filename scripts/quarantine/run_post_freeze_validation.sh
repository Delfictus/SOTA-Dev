#!/usr/bin/env bash
# run_post_freeze_validation.sh — Post-freeze PRISM4D blind validation scoring
# Run AFTER freeze_blind_validation.sh completes.
# Requires: prism_pub_baseline_validator.py in scripts/quarantine/
set -u

BLIND_BASE=/mnt/storage/prism-outputs/blind_validation
REPO=/home/diddy/Desktop/Prism4D-bio
VALIDATOR="${REPO}/scripts/quarantine/prism_pub_baseline_validator.py"
POST_FREEZE="${REPO}/docs/blind_validation/post_freeze_validation"
mkdir -p "$POST_FREEZE"

PRISM_VALIDATED=1  # validator is read-only analysis

score_target() {
    local SLOT=$1
    local TARGET_NAME=$2   # must match TARGET_HOLOS key in validator
    local CLEAN_PDB=$3
    
    local RUN_DIR="${BLIND_BASE}/${SLOT}/run"
    local PREP_DIR="${BLIND_BASE}/${SLOT}/prep"
    local OUTDIR="${POST_FREEZE}/${SLOT}"
    
    echo "[$(date -Iseconds)] SCORE: $SLOT ($TARGET_NAME)"
    
    if ! ls "${RUN_DIR}"/*.binding_sites.json >/dev/null 2>&1; then
        echo "  SKIP: no binding_sites.json for $SLOT"
        return 1
    fi
    
    mkdir -p "$OUTDIR"
    
    python3 "$VALIDATOR" \
        --run-dir   "$RUN_DIR" \
        --query-pdb "${PREP_DIR}/${CLEAN_PDB}" \
        --target    "$TARGET_NAME" \
        --outdir    "$OUTDIR" \
        --max-rank  10 \
        >> "${OUTDIR}/validator.log" 2>&1
    
    RC=$?
    if [ $RC -eq 0 ]; then
        echo "  PASS: $SLOT validation complete → $OUTDIR"
        grep -E "SR@|PRISM SR@" "${OUTDIR}/validator.log" | tail -5
    else
        echo "  FAIL: $SLOT validator exit=$RC"
        tail -5 "${OUTDIR}/validator.log"
    fi
}

echo "[$(date -Iseconds)] === POST-FREEZE BLIND VALIDATION SCORING ==="

# Score each target
score_target B01_HRAS_Q61H         HRAS_Q61H          4L9S_clean.pdb
score_target B02_CDK2_allosteric   CDK2_allosteric    1HCL_clean.pdb
score_target B03_Kv1.2             Kv1.2              3LUT_clean.pdb
score_target B04_MDM2              MDM2               1YCR_clean.pdb
score_target B05_TP53_R175H        TP53_apo           2OCJ_clean.pdb
score_target B06_cGAS              cGAS               4KM5_clean.pdb
score_target B07_TEAD1             TEAD1              3KYS_clean.pdb
score_target B08_CRBN              CRBN               4TZ4_chainC_clean.pdb
score_target B09_Thrombin_exosite  Thrombin_exosite   1PPB_clean.pdb
score_target B10_ADRB2             ADRB2              2RH1_clean.pdb

echo ""
echo "[$(date -Iseconds)] === ALL SCORES COMPLETE ==="
echo "Results in: $POST_FREEZE"
echo "Aggregate CSV files will be in each target subdirectory"
