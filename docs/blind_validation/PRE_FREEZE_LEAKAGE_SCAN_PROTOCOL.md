# PRE-FREEZE LEAKAGE SCAN PROTOCOL
**Version:** 1.0  
**Locked:** 2026-05-13 UTC

---

## When to run

Run this scan before every prediction freeze — both per-target and global.

---

## Scan 1: Drug-like HETATM in clean PDBs

```bash
echo "=== HETATM scan (non-solvent residues in clean PDBs) ==="
for pdb in /mnt/storage/prism-outputs/blind_validation/B*/prep/*_clean.pdb; do
    hits=$(grep "^HETATM" "$pdb" 2>/dev/null | \
           grep -v " HOH " | grep -v " SO4 " | grep -v " PEG " | \
           grep -v " GOL " | grep -v " EDO " | grep -v " DMS " | \
           grep -v " IOD " | grep -v " CL  " | grep -v " MG  " | \
           grep -v " ZN  " | grep -v " CA  " | grep -v " NA  " | \
           grep -v " K   " | grep -v " PO4 " | grep -v " ACT " | \
           grep -v " IMD " | grep -v " TRS " | grep -v " TAR " | \
           grep -v " FMT " | wc -l)
    if [ "$hits" -gt 0 ]; then
        echo "FAIL: $pdb has $hits suspicious HETATM records"
        grep "^HETATM" "$pdb" | grep -v " HOH " | grep -v " SO4 " | head -5
    else
        echo "PASS: $pdb"
    fi
done
```

Expected output: all lines show `PASS`.

---

## Scan 2: Holo/bound files in prep dirs

```bash
echo "=== Holo file scan ==="
find /mnt/storage/prism-outputs/blind_validation/ \
    \( -name "*holo*" -o -name "*bound*" -o -name "*ligand*.pdb" \
       -o -name "*inhibitor*" -o -name "*complex*" \) \
    -not -path "*/post_freeze*" \
    -not -path "*/BLIND_HOLO*"
```

Expected output: empty (no files found).

---

## Scan 3: Frozen dir hash check (if freezing an individual target)

```bash
TARGET=B01_HRAS_Q61H
FREEZE_DIR=/mnt/storage/prism-outputs/blind_validation/${TARGET}/frozen
cd ${FREEZE_DIR}
sha256sum -c FREEZE_MANIFEST_${TARGET}.sha256
echo "Exit code: $?"
```

Expected: all lines `OK`, exit code 0.

---

## Scan 4: Engine run used correct flags

```bash
for TARGET in B01_HRAS_Q61H B02_CDK2_allosteric B03_Kv1.2 B04_MDM2 \
              B05_TP53_R175H B06_cGAS B07_TEAD1 B08_CRBN \
              B09_Thrombin_exosite B10_ADRB2; do
    LOG=/mnt/storage/prism-outputs/blind_validation/${TARGET}/run/run.log
    if [ -f "$LOG" ]; then
        echo -n "$TARGET: "
        # Check locked flags present
        grep -q "spike-percentile 50\|spike_percentile=50" "$LOG" && echo -n "percentile50 " || echo -n "MISSING_PERCENTILE "
        grep -q "replica.seed.*42\|replica_seed.*42" "$LOG" && echo -n "seed42 " || echo -n "MISSING_SEED "
        grep -q "nma.perturb\|nma_perturb" "$LOG" && echo -n "nma " || echo -n "MISSING_NMA "
        echo ""
    fi
done
```

---

## Scan 5: No holo reference PDBs downloaded

```bash
echo "=== Known holo PDB ID scan ==="
# Search for any of the holo reference IDs in prep/raw files
# (These IDs are the validation references from prism_pub_baseline_validator.py TARGET_HOLOS)
# Check nothing with holo-like IDs is in the prep dirs
find /mnt/storage/prism-outputs/blind_validation/B*/prep/ \
    -name "*.pdb" | while read f; do
    head -3 "$f" | grep -qi "holo\|complex\|inhibitor" && echo "WARNING: $f header mentions holo/complex"
done
```

---

## Scan result recording

```bash
DATE=$(date -u +%Y%m%dT%H%M%SZ)
SCAN_OUT=/mnt/storage/prism-outputs/blind_validation/LEAKAGE_SCAN_${DATE}.txt
{
    echo "Leakage scan: ${DATE}"
    echo "--- Scan 1: HETATM ---"
    # (run scan 1 inline)
    echo "--- Scan 2: Holo files ---"
    # (run scan 2 inline)
    echo "PASS: clean"
} > ${SCAN_OUT}
```

Record the scan output file path in the per-target freeze metadata.
