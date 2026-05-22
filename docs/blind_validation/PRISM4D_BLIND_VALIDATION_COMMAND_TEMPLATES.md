# BLIND VALIDATION COMMAND TEMPLATES
**Version:** 1.0  
**Locked:** 2026-05-13 UTC  
**Repo HEAD:** f8f368f6b83e118126e691626a823866e49906f5

---

## Template variables

```bash
PRISM_SRC=~/Desktop/Prism4D-bio
PRISM_FROZEN=~/Desktop/Prism4D-v1.1-frozen/corpus-apr12/source-sandbox
BLIND_BASE=/mnt/storage/prism-outputs/blind_validation
TARGET=B01_HRAS_Q61H            # slot_name
PDBID=4L9S                      # apo PDB ID (lowercase for files)
CHAIN=A
```

---

## Phase 0: Setup

```bash
mkdir -p ${BLIND_BASE}/${TARGET}/{prep,topologies,run,fpocket,p2rank,frozen}
```

---

## Phase 1: Download + Clean

```bash
cd ${BLIND_BASE}/${TARGET}/prep

# Download apo
curl -s "https://files.rcsb.org/download/${PDBID}.pdb" -o ${PDBID}_raw.pdb

# Clean (strip altconfs, keep chain A, validate residue diversity ≥15)
python3 ${PRISM_SRC}/scripts/prism-clean.py \
    ${PDBID}_raw.pdb \
    ${PDBID}_clean.pdb \
    ${CHAIN}
```

---

## Phase 2: Topology + NMA

```bash
cd ${BLIND_BASE}/${TARGET}/topologies

# Generate topology + NMA modes
${PRISM_FROZEN}/scripts/prism-prep \
    ${BLIND_BASE}/${TARGET}/prep/${PDBID}_clean.pdb \
    ${PDBID}_clean.topology.json

# NMA modes should auto-generate at: ${PDBID}_nma_modes.json
# If not present, generate separately:
# ${PRISM_FROZEN}/scripts/prism-prep \
#     ${BLIND_BASE}/${TARGET}/prep/${PDBID}_clean.pdb \
#     ${PDBID}_clean.topology.json \
#     --nma
```

---

## Phase 3: PRISM4D Engine Run (LOCKED FLAGS)

**Standard (≤400 residues):**
```bash
RUST_LOG=info ${PRISM_FROZEN}/scripts/prism-validate-and-run.sh \
    -t ${BLIND_BASE}/${TARGET}/topologies/${PDBID}_clean.topology.json \
    -o ${BLIND_BASE}/${TARGET}/run \
    --fast \
    --hysteresis \
    --prism-therm \
    --multi-stream 8 \
    --multi-scale \
    --spike-percentile 50 \
    --hmr \
    --adaptive-dt \
    --fused-steps 6 \
    --nma-perturb ${BLIND_BASE}/${TARGET}/topologies/${PDBID}_nma_modes.json \
    --nma-amplification 3.0 \
    --replica-seed 42 \
    -v 2>&1 | tee ${BLIND_BASE}/${TARGET}/run/run.log
```

**Large (>400 residues, e.g., Kv1.2, ADRB2):**
```bash
# Same as above but --multi-stream 20
```

---

## Phase 4: fpocket Baseline

```bash
cd ${BLIND_BASE}/${TARGET}/fpocket
fpocket -f ${BLIND_BASE}/${TARGET}/prep/${PDBID}_clean.pdb 2>&1 | tee fpocket.log
```

---

## Phase 5: P2Rank Baseline

```bash
/opt/p2rank/prank predict \
    -f ${BLIND_BASE}/${TARGET}/prep/${PDBID}_clean.pdb \
    -o ${BLIND_BASE}/${TARGET}/p2rank \
    -threads 4 \
    2>&1 | tee ${BLIND_BASE}/${TARGET}/p2rank/p2rank.log
```

---

## Phase 6: Freeze

```bash
# (See PREDICTION_FREEZE_PROTOCOL.md for full procedure)
FREEZE_DIR=${BLIND_BASE}/${TARGET}/frozen
cp ${BLIND_BASE}/${TARGET}/run/*.binding_sites.json      ${FREEZE_DIR}/
cp ${BLIND_BASE}/${TARGET}/run/*.kcc_visualization.json  ${FREEZE_DIR}/
cp ${BLIND_BASE}/${TARGET}/run/*.kcc_validation.json     ${FREEZE_DIR}/
cp ${BLIND_BASE}/${TARGET}/run/*.topology.prism_therm.json ${FREEZE_DIR}/
cp ${BLIND_BASE}/${TARGET}/run/run.log                   ${FREEZE_DIR}/
cd ${FREEZE_DIR}
find . -type f | sort | xargs sha256sum > FREEZE_MANIFEST_${TARGET}.sha256
chmod -R 555 ${FREEZE_DIR}
```

---

## Phase 7: Post-Freeze Validation (after GLOBAL freeze commit)

```bash
python3 ${PRISM_SRC}/scripts/quarantine/prism_pub_baseline_validator.py \
    --run-dir ${BLIND_BASE}/${TARGET}/run \
    --query-pdb ${BLIND_BASE}/${TARGET}/prep/${PDBID}_clean.pdb \
    --target ${TARGET_NAME_SHORT} \
    --outdir ${BLIND_BASE}/post_freeze_validation/${TARGET} \
    2>&1 | tee ${BLIND_BASE}/post_freeze_validation/${TARGET}/validator.log
```

---

## Null controls (if script available)

```bash
python3 ${PRISM_SRC}/scripts/quarantine/null_controls/pair_breaking_null.py \
    --prism-csv ${BLIND_BASE}/post_freeze_validation/AGGREGATE_PRISM_VS_HOLO.csv \
    --n-iters 1000 \
    --out ${BLIND_BASE}/post_freeze_validation/NULL_CONTROL_RESULTS.csv
```

---

## Per-target sizing override table

| Target | Slot | PDBID | CHAIN | multi_stream | HMR | Notes |
|--------|------|-------|-------|-------------|-----|-------|
| HRAS_Q61H | B01 | 4L9S | A | 8 | YES | standard |
| CDK2_allosteric | B02 | 1HCL | A | 8 | YES | standard |
| Kv1.2 | B03 | 3LUT | A | 20 | YES | >400 res |
| MDM2 | B04 | 1Z1M | A | 8 | YES | standard |
| TP53_R175H | B05 | 2OCJ | A | 8 | YES | standard |
| cGAS | B06 | 4KM5 | A | 8 | YES | standard |
| TEAD1 | B07 | 3KYS | A | 8 | YES | standard |
| CRBN | B08 | 4TZ4 | A | 8 | YES | standard |
| Thrombin_exosite | B09 | 1HAH | A | 8 | YES | standard |
| ADRB2 | B10 | 2RH1 | A | 20 | YES | >400 res; hard neg |
