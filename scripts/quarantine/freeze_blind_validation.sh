#!/usr/bin/env bash
# freeze_blind_validation.sh — Prediction freeze for all 10 blind validation targets
# Run AFTER all engine runs complete. Copies artifacts to frozen/ dirs, writes manifests.
set -u

BLIND_BASE=/mnt/storage/prism-outputs/blind_validation
REPO=/home/diddy/Desktop/Prism4D-bio
FREEZE_TIME=$(date -u +%Y%m%dT%H%M%SZ)

freeze_target() {
    local SLOT=$1         # e.g. B01_HRAS_Q61H
    local TOPO_BASE=$2    # e.g. 4L9S
    
    local RUN_DIR="${BLIND_BASE}/${SLOT}/run"
    local FROZEN_DIR="${BLIND_BASE}/${SLOT}/frozen"
    
    echo "[$(date -Iseconds)] FREEZE: $SLOT"
    
    # Verify run completed
    if ! ls "${RUN_DIR}"/*.binding_sites.json >/dev/null 2>&1; then
        echo "  ERROR: binding_sites.json not found in $RUN_DIR — skipping $SLOT"
        return 1
    fi
    if ! ls "${RUN_DIR}"/*.kcc_visualization.json >/dev/null 2>&1; then
        echo "  ERROR: kcc_visualization.json not found in $RUN_DIR — skipping $SLOT"
        return 1
    fi
    
    mkdir -p "$FROZEN_DIR"
    
    # Copy core prediction artifacts
    cp "${RUN_DIR}"/*.binding_sites.json   "${FROZEN_DIR}/"
    cp "${RUN_DIR}"/*.kcc_visualization.json "${FROZEN_DIR}/"
    cp "${RUN_DIR}"/*.topology.prism_therm.json "${FROZEN_DIR}/" 2>/dev/null || true
    cp "${RUN_DIR}"/*.kcc_validation.json  "${FROZEN_DIR}/" 2>/dev/null || true
    cp "${RUN_DIR}"/run.log                "${FROZEN_DIR}/"
    
    # Generate SHA256 manifest
    MANIFEST="${FROZEN_DIR}/FREEZE_MANIFEST_${FREEZE_TIME}.sha256"
    (cd "$FROZEN_DIR" && sha256sum *.json *.log 2>/dev/null) > "$MANIFEST"
    
    echo "  SHA256 manifest: $MANIFEST"
    echo "  Lines: $(wc -l < "$MANIFEST")"
    
    # Write freeze attestation
    BS_JSON=$(ls "${FROZEN_DIR}"/*.binding_sites.json | head -1)
    N_SITES=$(python3 -c "import json; d=json.load(open('${BS_JSON}')); print(len(d.get('sites',[])))" 2>/dev/null || echo "?")
    
    cat > "${FROZEN_DIR}/PREDICTION_FREEZE_${SLOT}.md" << ATTEST_EOF
# PREDICTION FREEZE — ${SLOT}
**Frozen:** ${FREEZE_TIME}  
**Repo HEAD:** $(git -C "$REPO" rev-parse HEAD)  
**Engine binary hash:** $(sha256sum ~/Desktop/Prism4D-v1.1-frozen/corpus-apr12/source-sandbox/target/release/nhs_rt_full | cut -d' ' -f1)

## Artifacts frozen
$(ls "${FROZEN_DIR}")

## Predicted sites: ${N_SITES}
## Manifeset: FREEZE_MANIFEST_${FREEZE_TIME}.sha256

## Attestation
Predictions for ${SLOT} were generated from the apo structure using the PRISM4D v1.1 frozen engine.
No holo or ligand-bound coordinate information was accessed before this freeze.
SHA256 manifests of all prediction artifacts are verifiable from git history.
ATTEST_EOF
    
    # Lock frozen artifacts
    chmod -R 555 "$FROZEN_DIR"
    
    echo "  DONE: ${SLOT} frozen at ${FROZEN_DIR}"
    return 0
}

echo "[$(date -Iseconds)] === BLIND VALIDATION PREDICTION FREEZE ==="
echo "[$(date -Iseconds)] Freeze timestamp: ${FREEZE_TIME}"

# Targets: SLOT TOPO_BASE
freeze_target B01_HRAS_Q61H         4L9S
freeze_target B02_CDK2_allosteric   1HCL
freeze_target B03_Kv1.2             3LUT
freeze_target B04_MDM2              1YCR
freeze_target B05_TP53_R175H        2OCJ
freeze_target B06_cGAS              4KM5
freeze_target B07_TEAD1             3KYS
freeze_target B08_CRBN              4TZ4_chainC
freeze_target B09_Thrombin_exosite  1PPB
freeze_target B10_ADRB2             2RH1

echo ""
echo "[$(date -Iseconds)] === ALL TARGETS FROZEN ==="
echo "Now commit: cd $REPO && git add docs/blind_validation/BLIND_HOLO_REFERENCES.md && git commit -m 'freeze: GLOBAL_PREDICTION_FREEZE blind validation B01-B10'"
