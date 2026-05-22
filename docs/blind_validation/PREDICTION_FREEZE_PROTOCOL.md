# PREDICTION FREEZE PROTOCOL
**Version:** 1.0  
**Locked:** 2026-05-13 UTC  
**Repo HEAD:** f8f368f6b83e118126e691626a823866e49906f5

---

## Purpose

Establishes a cryptographically-attested point-in-time freeze of all PRISM4D predictions before any holo validation coordinate is accessed. Post-freeze, predictions cannot be modified without audit trail invalidation.

---

## Freeze trigger

Freeze is triggered when ALL of the following are true for a target:
1. PRISM4D engine run completed (run.log exits 0)
2. binding_sites.json is non-empty (≥1 site)
3. kcc_visualization.json is present
4. fpocket baseline run completed
5. P2Rank baseline run completed
6. No holo coordinate has been accessed since prep step

---

## Per-target freeze procedure

For each target B01–B10, after engine + baselines complete:

```bash
TARGET=B01_HRAS_Q61H
OUTDIR=/mnt/storage/prism-outputs/blind_validation/${TARGET}
FREEZE_DIR=${OUTDIR}/frozen

mkdir -p ${FREEZE_DIR}

# 1. Copy prediction artifacts to freeze dir (read-only copy)
cp ${OUTDIR}/*.binding_sites.json        ${FREEZE_DIR}/
cp ${OUTDIR}/*.kcc_visualization.json    ${FREEZE_DIR}/
cp ${OUTDIR}/*.kcc_validation.json       ${FREEZE_DIR}/
cp ${OUTDIR}/*.topology.prism_therm.json ${FREEZE_DIR}/
cp ${OUTDIR}/run.log                     ${FREEZE_DIR}/
cp ${OUTDIR}/fpocket/                    ${FREEZE_DIR}/fpocket/ -r 2>/dev/null || true
cp ${OUTDIR}/p2rank/                     ${FREEZE_DIR}/p2rank/  -r 2>/dev/null || true

# 2. Generate SHA256 manifest
cd ${FREEZE_DIR}
find . -type f | sort | xargs sha256sum > FREEZE_MANIFEST_${TARGET}.sha256

# 3. Record freeze timestamp
echo "FREEZE_TIMESTAMP: $(date -u +%Y-%m-%dT%H:%M:%SZ)" > FREEZE_METADATA_${TARGET}.txt
echo "TARGET: ${TARGET}"                                 >> FREEZE_METADATA_${TARGET}.txt
echo "OPERATOR: $(whoami)"                               >> FREEZE_METADATA_${TARGET}.txt
echo "ENGINE_COMMIT: $(git -C ~/Desktop/Prism4D-bio rev-parse HEAD)" >> FREEZE_METADATA_${TARGET}.txt
echo "HOLO_ACCESS: NONE_BEFORE_FREEZE"                  >> FREEZE_METADATA_${TARGET}.txt

# 4. Lock freeze dir (no further writes)
chmod -R 555 ${FREEZE_DIR}

# 5. Write per-target freeze doc to repo
cat > ~/Desktop/Prism4D-bio/docs/blind_validation/frozen_predictions/PREDICTION_FREEZE_${TARGET}.md << EOF
# Prediction Freeze: ${TARGET}
Frozen: $(date -u +%Y-%m-%dT%H:%M:%SZ)
Engine commit: $(git -C ~/Desktop/Prism4D-bio rev-parse HEAD)
Frozen artifacts: ${FREEZE_DIR}
Manifest: ${FREEZE_DIR}/FREEZE_MANIFEST_${TARGET}.sha256
Holo access before freeze: NONE
EOF
```

---

## Global freeze procedure

After all 10 per-target freezes are complete:

```bash
# Generate global freeze manifest
cat docs/blind_validation/frozen_predictions/PREDICTION_FREEZE_B*.md \
  > docs/blind_validation/GLOBAL_PREDICTION_FREEZE.md

echo "---" >> docs/blind_validation/GLOBAL_PREDICTION_FREEZE.md
echo "Global freeze complete: $(date -u +%Y-%m-%dT%H:%M:%SZ)" >> docs/blind_validation/GLOBAL_PREDICTION_FREEZE.md

# Commit the freeze state (this is the irrevocable timestamp in git history)
git add docs/blind_validation/frozen_predictions/
git add docs/blind_validation/GLOBAL_PREDICTION_FREEZE.md
git commit -m "blind_validation: freeze predictions B01-B10 (pre-holo)"
```

The git commit hash becomes the forensic timestamp for the freeze.

---

## Post-freeze rules (MANDATORY)

1. PRISM4D binding_sites.json output files in frozen dirs are READ-ONLY — chmod 444 enforced.
2. No re-running the engine on any frozen target without opening a new freeze slot (B11+).
3. Holo coordinate access (any download of validation PDBs, any `prism_pub_baseline_validator.py` run with holos) is only permitted after the git freeze commit is in history.
4. If a post-freeze discrepancy is found (hash mismatch), investigate immediately — it is a leakage incident.
5. Freeze covers: binding_sites.json, kcc_visualization.json, kcc_validation.json, prism_therm.json, fpocket output, P2Rank output.

---

## Freeze artifacts per target (expected)

```
frozen_predictions/
  PREDICTION_FREEZE_B01_HRAS_Q61H.md
  PREDICTION_FREEZE_B02_CDK2_allosteric.md
  PREDICTION_FREEZE_B03_Kv1.2.md
  PREDICTION_FREEZE_B04_MDM2.md
  PREDICTION_FREEZE_B05_TP53_R175H.md
  PREDICTION_FREEZE_B06_cGAS.md
  PREDICTION_FREEZE_B07_TEAD1.md
  PREDICTION_FREEZE_B08_CRBN.md
  PREDICTION_FREEZE_B09_Thrombin_exosite.md
  PREDICTION_FREEZE_B10_ADRB2.md
GLOBAL_PREDICTION_FREEZE.md
```
