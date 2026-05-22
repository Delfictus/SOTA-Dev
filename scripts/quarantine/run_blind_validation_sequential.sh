#!/usr/bin/env bash
# Sequential blind validation runner — runs 2 at a time to avoid GPU OOM
set -u

BLIND_BASE=/mnt/storage/prism-outputs/blind_validation
PRISM_FROZEN=~/Desktop/Prism4D-v1.1-frozen/corpus-apr12/source-sandbox

run_target() {
    local SLOT=$1
    local TOPO_BASE=$2
    local NMA_BASE=$3
    local MSTREAM=$4
    
    local TOPO="${BLIND_BASE}/${SLOT}/topologies/${TOPO_BASE}.topology.json"
    local NMA="${BLIND_BASE}/${SLOT}/topologies/${NMA_BASE}_nma_modes.json"
    local OUTDIR="${BLIND_BASE}/${SLOT}/run"
    
    echo "[$(date -Iseconds)] START: $SLOT (streams=$MSTREAM)"
    
    cd ${PRISM_FROZEN}
    RUST_LOG=info ./scripts/prism-validate-and-run.sh \
        -t "$TOPO" \
        -o "$OUTDIR" \
        --fast --hysteresis --prism-therm \
        --multi-stream ${MSTREAM} \
        --multi-scale \
        --spike-percentile 50 \
        --hmr --adaptive-dt \
        --fused-steps 6 \
        --nma-perturb "$NMA" \
        --nma-amplification 3.0 \
        --replica-seed 42 \
        -v >> "${OUTDIR}/run.log" 2>&1
    
    local RC=$?
    echo "[$(date -Iseconds)] DONE: $SLOT exit=$RC"
}

echo "[$(date -Iseconds)] Sequential blind validation runner started"

# Run 2 at a time (pairs)
# Pair 1
run_target B02_CDK2_allosteric 1HCL 1HCL 8 &
run_target B04_MDM2 1YCR 1YCR 8 &
wait
echo "[$(date -Iseconds)] Pair 1 done"

# Pair 2
run_target B03_Kv1.2 3LUT 3LUT 8 &
run_target B05_TP53_R175H 2OCJ 2OCJ 8 &
wait
echo "[$(date -Iseconds)] Pair 2 done"

# Pair 3
run_target B06_cGAS 4KM5 4KM5 8 &
run_target B07_TEAD1 3KYS 3KYS 8 &
wait
echo "[$(date -Iseconds)] Pair 3 done"

# Pair 4
run_target B08_CRBN 4TZ4_chainC 4TZ4_chainC 8 &
run_target B09_Thrombin_exosite 1PPB 1PPB 8 &
wait
echo "[$(date -Iseconds)] Pair 4 done"

# Single: B10 ADRB2 (20 streams, most VRAM)
run_target B10_ADRB2 2RH1 2RH1 20
echo "[$(date -Iseconds)] B10 done"

echo "[$(date -Iseconds)] ALL RUNS COMPLETE"
