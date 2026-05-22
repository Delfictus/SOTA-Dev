# POST-FREEZE VALIDATION PROTOCOL
**Version:** 1.0  
**Locked:** 2026-05-13 UTC  
**Repo HEAD:** f8f368f6b83e118126e691626a823866e49906f5

---

## Trigger condition

This protocol runs ONLY after `GLOBAL_PREDICTION_FREEZE.md` is committed to git and all 10 per-target freeze SHA256 manifests are verified intact.

```bash
# Verify all freeze manifests before opening any holo
for T in B01 B02 B03 B04 B05 B06 B07 B08 B09 B10; do
    FREEZE_DIR=/mnt/storage/prism-outputs/blind_validation/${T}*/frozen
    cd $FREEZE_DIR
    sha256sum -c FREEZE_MANIFEST_${T}*.sha256 || echo "INTEGRITY FAILURE: $T"
done
```

All 10 must pass before proceeding.

---

## Step 1: Run post-freeze baseline validator (all targets)

```bash
FROZEN_PREDICTIONS=/mnt/storage/prism-outputs/blind_validation
POSTFREEZE_DIR=/mnt/storage/prism-outputs/blind_validation/post_freeze_validation
mkdir -p ${POSTFREEZE_DIR}

# Per target — repeat for each B01..B10
TARGET_NAME=HRAS_Q61H
APO_PDB=/mnt/storage/prism-outputs/blind_validation/B01_HRAS_Q61H/prep/hras_q61h_clean.pdb
RUN_DIR=/mnt/storage/prism-outputs/blind_validation/B01_HRAS_Q61H

python3 ~/Desktop/Prism4D-bio/scripts/quarantine/prism_pub_baseline_validator.py \
    --run-dir ${RUN_DIR} \
    --query-pdb ${APO_PDB} \
    --target ${TARGET_NAME} \
    --outdir ${POSTFREEZE_DIR}/${TARGET_NAME} \
    2>&1 | tee ${POSTFREEZE_DIR}/${TARGET_NAME}/validator.log
```

Outputs per target:
- `prism_vs_holo.csv` — PRISM4D site vs holo shell scoring
- `baseline_vs_holo.csv` — fpocket/P2Rank vs holo shell scoring
- `loro_results.csv` — LORO cross-validation
- `family_results.csv` — family collapse
- `validation_report.md` — human-readable summary
- `pymol_overlay.pml` — structural overlay

---

## Step 2: Aggregate results

```bash
python3 - << 'EOF'
import glob, pandas as pd, os

outbase = "/mnt/storage/prism-outputs/blind_validation/post_freeze_validation"
targets = ["HRAS_Q61H", "CDK2_allosteric", "Kv1.2", "MDM2", "TP53_R175H",
           "cGAS", "TEAD1", "CRBN", "Thrombin_exosite", "ADRB2"]

prism_rows, base_rows = [], []
for t in targets:
    p = f"{outbase}/{t}/prism_vs_holo.csv"
    b = f"{outbase}/{t}/baseline_vs_holo.csv"
    if os.path.exists(p):
        df = pd.read_csv(p); df["target"] = t; prism_rows.append(df)
    if os.path.exists(b):
        df = pd.read_csv(b); df["target"] = t; base_rows.append(df)

pd.concat(prism_rows).to_csv(f"{outbase}/AGGREGATE_PRISM_VS_HOLO.csv", index=False)
pd.concat(base_rows).to_csv(f"{outbase}/AGGREGATE_BASELINE_VS_HOLO.csv", index=False)
print("Aggregated")
EOF
```

---

## Step 3: SR@k computation

Compute success rate at top-k (SR@1, SR@3, SR@5) for PRISM4D vs fpocket vs P2Rank:
- SR@k = fraction of targets where ≥1 holo shell hit is in top-k ranked sites
- Shell thresholds: 4 Å, 6 Å, 8 Å (report all three)
- Method: `scripts/quarantine/prism_pub_baseline_validator.py` computes these internally

---

## Step 4: LORO results

LORO is embedded in the validator. Confirm:
- `loro_results.csv` generated for each target
- Withhold-one reconstruction rate (RR@LORO) computed per target

---

## Step 5: Family collapse

Family collapse embedded in validator (union-find on co-validated ligand instances):
- `family_results.csv` generated per target
- Family coverage = fraction of distinct ligand classes co-validated by ≥1 PRISM site

---

## Step 6: Null controls (if implemented)

If null control scripts are available:
```bash
python3 scripts/quarantine/null_controls/pair_breaking_null.py \
    --prism-csv ${POSTFREEZE_DIR}/AGGREGATE_PRISM_VS_HOLO.csv \
    --n-iters 1000 \
    --out ${POSTFREEZE_DIR}/NULL_CONTROL_RESULTS.csv
```
Report: empirical p-value for SR@k vs null distribution.

---

## Step 7: Generate final validation report

```bash
python3 scripts/quarantine/generate_blind_validation_report.py \
    --aggregate-prism ${POSTFREEZE_DIR}/AGGREGATE_PRISM_VS_HOLO.csv \
    --aggregate-baseline ${POSTFREEZE_DIR}/AGGREGATE_BASELINE_VS_HOLO.csv \
    --loro-glob "${POSTFREEZE_DIR}/*/loro_results.csv" \
    --family-glob "${POSTFREEZE_DIR}/*/family_results.csv" \
    --out ${POSTFREEZE_DIR}/BLIND_VALIDATION_FINAL_REPORT.md
```

---

## Acceptance criteria

| Metric | Accept threshold | Notes |
|--------|-----------------|-------|
| SR@5@8Å (PRISM4D) | ≥ 0.70 | 7/10 targets with ≥1 holo hit in top 5 |
| SR@1@8Å (PRISM4D) | ≥ 0.50 | 5/10 targets top-1 hit |
| SR@5@8Å (fpocket) | Report as-is | Comparative, no fixed gate |
| SR@5@8Å (P2Rank) | Report as-is | Comparative, no fixed gate |
| Hard negative pass (ADRB2) | SR@8Å < 0.20 | |
| LORO reconstruction rate | ≥ 0.50 | Over all targets |
| Null p-value (if available) | < 0.05 | |
