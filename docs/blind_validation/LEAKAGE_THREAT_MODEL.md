# LEAKAGE THREAT MODEL
**Version:** 1.0  
**Locked:** 2026-05-13 UTC

---

## Threat taxonomy

### T1 — Direct holo coordinate access before freeze
**Vector:** Operator downloads validation holo PDB or mmCIF before prediction freeze.  
**Severity:** CRITICAL — invalidates blind validation completely.  
**Prevention:** Holo PDB IDs are in BLIND_HOLO_REFERENCES.md (not opened until post-freeze). prism-clean.py and prism-prep operate on apo only. Engine sees only apo topology.  
**Detection:** `find /mnt/storage/prism-outputs/blind_validation/B*/prep -name "*holo*" -o -name "*_bound*"` — must return empty pre-freeze.

### T2 — Ligand coordinates embedded in "apo" PDB
**Vector:** Selected apo PDB contains ligand/co-crystal at target site.  
**Severity:** HIGH — steers detection to known binding site.  
**Prevention:** prism-clean.py strips HETATM records. Manual review: `grep HETATM *_clean.pdb | grep -v HOH | grep -v SO4 | grep -v PEG` — all drug-like ligands must be absent.  
**Detection:** Run pre-freeze leakage scan per `PRE_FREEZE_LEAKAGE_SCAN_PROTOCOL.md`.

### T3 — NMA modes derived from holo structure
**Vector:** NMA mode generation uses holo coordinates instead of apo.  
**Severity:** MEDIUM — NMA perturbation biased toward holo conformation.  
**Prevention:** prism-prep called exclusively on apo clean PDB. NMA modes file name includes apo PDB ID as prefix.

### T4 — Publication target reuse
**Vector:** One of B01–B10 is identical to or overlapping with a publication target (same protein, same PDB chain, same binding site).  
**Severity:** HIGH — engine has effectively been benchmarked on this before.  
**Prevention:** TARGET_SELECTION_PROTOCOL.md requirement: zero overlap with KRAS_G12C/Kv3.1/p53_Y220C/AKT1/TEAD3/TRPV1/GLP1R/MCL1/STING/M4R.

### T5 — Post-freeze prediction modification
**Vector:** Operator modifies binding_sites.json after seeing holo structures.  
**Severity:** CRITICAL — invalidates all scoring.  
**Prevention:** chmod 555 on frozen dir. SHA256 manifest verified at post-freeze phase start.  
**Detection:** Hash mismatch in `sha256sum -c FREEZE_MANIFEST_*.sha256`.

### T6 — Scoring parameter tuning after holo access
**Vector:** Shell cutoffs, alignment gates, or minimum ligand atom threshold modified after examining holo structures.  
**Severity:** HIGH — p-hacking the scoring framework.  
**Prevention:** All scoring parameters locked in LAST10_METHOD_LOCK.md §4 before any holo access. prism_pub_baseline_validator.py used verbatim with no parameter edits.

### T7 — Transcript / memory leakage
**Vector:** Prior session knowledge of holo binding sites for B01–B10 targets influences target selection or run interpretation.  
**Severity:** MEDIUM — soft bias toward known sites.  
**Prevention:** Target selection documented in this session starting from apo PDB IDs. No holo references examined in this session.

---

## Pre-freeze leakage scan (automated)

Run before any freeze:
```bash
# Check for holo-like files in prep dirs
find /mnt/storage/prism-outputs/blind_validation/ \
    \( -name "*holo*" -o -name "*bound*" -o -name "*ligand*" \) \
    -not -path "*/post_freeze*" \
    -not -path "*/BLIND_HOLO*"

# Check for drug-like HETATM in clean PDBs
for pdb in /mnt/storage/prism-outputs/blind_validation/*/prep/*_clean.pdb; do
    echo "=== $pdb ==="; 
    grep "^HETATM" $pdb | grep -v " HOH " | grep -v " SO4 " | grep -v " PEG " | grep -v " GOL " | grep -v " EDO " | grep -v " DMS " | grep -v " IOD " | grep -v " CL  " | grep -v " MG  " | grep -v " ZN  " | grep -v " CA  "
done
```

Both must return empty.
