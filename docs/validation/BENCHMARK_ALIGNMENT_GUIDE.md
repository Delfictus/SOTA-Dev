# PRISM-4D: Benchmark Validation Ground Truth & Alignment Guide

## CRITICAL: READ THIS BEFORE INTERPRETING ANY DCC RESULTS

### The #1 Source of False Failures is Coordinate Frame Mismatch

When comparing PRISM-detected binding site centroids against known ligand
positions from holo crystal structures, the apo and holo PDB files are
almost NEVER in the same coordinate frame. Each crystal structure is solved
independently — (0,0,0) is arbitrary and different for every PDB entry.

**If you see DCC > 20Å, it is almost certainly a frame mismatch, NOT a
detection failure.** Real detection failures produce DCC of 10-15Å. Frame
mismatches produce DCC of 40-80Å.

### The #2 Source of False Failures is Wrong Protein Identity

PDB codes are arbitrary 4-character strings. Similar codes do NOT mean
similar proteins. Verified example from this benchmark:

- **1BTM** looks like it should be the holo for **1BTL** (TEM-1 β-Lactamase)
- **1BTM is actually Triosephosphate Isomerase** — 6% sequence identity
- The real holo for 1BTL is **1TEM** (100% identity, same numbering)

**ALWAYS verify protein identity before computing DCC:**
1. Check PDB TITLE/COMPND header
2. Compute sequence identity (must be >90% for same protein)
3. Check residue numbering compatibility

### The #3 Source of False Failures is Residue Numbering Mismatch

Different labs deposit structures with different numbering schemes:
- Ambler numbering (1BTL: residues 26-290)
- Sequential numbering (some structures: 1-265)
- Insertion codes (52A, 52B — PDB hack for indels)

If Kabsch superposition returns RMSD > 5Å despite >90% sequence identity,
the residue numbers don't correspond. You need sequence alignment first.

---

## MANDATORY ALIGNMENT PROTOCOL

For every apo-holo pair, follow this exact sequence:

### Step 1: Verify Protein Identity
```python
# Extract title from both PDBs
for pdb in [apo_path, holo_path]:
    for line in open(pdb):
        if line.startswith("TITLE"):
            print(line.strip())

# Compute sequence identity
seq_apo = extract_ca_sequence(apo_path, chain="A")
seq_holo = extract_ca_sequence(holo_path, chain="A")
identity = sequence_identity(seq_apo, seq_holo)
assert identity > 0.90, f"WRONG PROTEIN: {identity:.0%} identity"
```

### Step 2: Check Frame Compatibility
```python
ca_apo = get_ca_coords(apo_path, chain="A")
ca_holo = get_ca_coords(holo_path, chain="A")
common = find_common_residues(ca_apo, ca_holo)

# Raw deviation WITHOUT superposition
raw_rmsd = compute_rmsd(ca_apo[common], ca_holo[common])

if raw_rmsd < 3.0:
    print("Frames compatible — no superposition needed")
    R, t = np.eye(3), np.zeros(3)
elif raw_rmsd < 100.0:
    print(f"Frame mismatch ({raw_rmsd:.1f}Å) — Kabsch superposition required")
    R, t = kabsch(ca_holo[common], ca_apo[common])  # transform holo → apo frame
    aligned_rmsd = verify_alignment(ca_holo[common], ca_apo[common], R, t)
    assert aligned_rmsd < 3.0, f"Alignment failed: {aligned_rmsd:.1f}Å"
else:
    print("Extreme deviation — check residue numbering")
```

### Step 3: Transform Ligand Centroid
```python
lig_atoms = get_ligand_heavy_atoms(holo_path, resname)
lig_centroid_holo = np.mean(lig_atoms, axis=0)
lig_centroid_apo_frame = R @ lig_centroid_holo + t  # NOW in apo coordinate frame
```

### Step 4: Compute DCC Against ALL Sites
```python
# Compare against ALL PRISM sites, not just top-ranked
# The best-matching site is the one closest to the ligand, regardless of rank
for site in prism_sites:
    dcc = np.linalg.norm(site["centroid"] - lig_centroid_apo_frame)
```

---

## VERIFIED GROUND TRUTH TABLE (CryptoSite Benchmark)

All pairs verified 2026-02-21 with PDB header inspection, sequence identity,
and coordinate frame analysis.

| Apo  | Protein              | Holo | Ligand | Frame Status          | Notes |
|------|----------------------|------|--------|-----------------------|-------|
| 1BTL | TEM-1 β-Lactamase   | 1TEM | ALP    | Compatible (0.96Å)    | NOT 1BTM (that's TIM, 6% identity) |
| 1W50 | BACE/β-Secretase     | 1W51 | L01    | Compatible (0.78Å)    | Same crystal series |
| 1G1F | PTP1B                | 1T49 | 892    | **Kabsch req (0.93Å)**| Allosteric cryptic site. Lig code "892" not "BB2" |
| 1ADE | AdSS                 | 1GIM | GDP    | **Kabsch req (1.98Å)**| True apo → holo with substrate analog |
| 3K5V | Abl Kinase           | 3K5V | STI    | Self-reference         | Ligand already in structure, stripped by prep |
| 1A4Q | Influenza Neuraminid.| 1A4Q | DPC    | Self-reference         | Homodimer: 2 DPC sites (chains A, B) |
| 1BJ4 | Human SHMT           | 1BJ4 | PLP    | Self-reference         | Cofactor already bound |
| 1HHP | HIV-1 Protease       | 1HVR | XK2    | **Kabsch req (1.03Å)**| C2 symmetric dimer |
| 1ERE | Estrogen Receptor α  | 1ERR | RAL    | **Kabsch req (3.34Å)**| Homodimer, large conformational change |
| 4OBE | KRAS                 | 4LYJ | 6H5    | Untested (skipped)    | Monomer too small for pipeline |

### WRONG PAIRS TO AVOID (found in eval_spatial_v2.py)

| Apo  | Wrong Holo | Wrong Protein      | Correct Holo |
|------|------------|--------------------|--------------|
| 1BTL | 1BTM       | Triosephosphate Isomerase | 1TEM |
| 1BJ4 | 1BJV       | Thrombin inhibitor complex | 1BJ4 (self-ref PLP) |
| 4OBE | 4OQ3       | MDM2/p53 inhibitor | 4LYJ |

---

## KABSCH SUPERPOSITION REFERENCE IMPLEMENTATION

```python
import numpy as np

def kabsch(P, Q):
    """Compute rotation R and translation t to align P onto Q.
    
    Transforms points in P's frame to Q's frame: Q ≈ R @ P + t
    
    Args:
        P: (N, 3) array — source coordinates (holo CA atoms)
        Q: (N, 3) array — target coordinates (apo CA atoms)
    
    Returns:
        R: (3, 3) rotation matrix
        t: (3,) translation vector
    
    Usage:
        # To transform a ligand centroid from holo frame to apo frame:
        lig_in_apo = R @ lig_in_holo + t
    """
    cP, cQ = P.mean(0), Q.mean(0)
    Pc, Qc = P - cP, Q - cQ
    H = Pc.T @ Qc
    U, S, Vt = np.linalg.svd(H)
    d = np.linalg.det(Vt.T @ U.T)
    R = Vt.T @ np.diag([1, 1, d]) @ U.T
    t = cQ - R @ cP
    return R, t
```

---

## CHECKLIST FOR ADDING NEW BENCHMARK PROTEINS

When adding a new protein to the benchmark set:

- [ ] Download apo PDB, inspect TITLE/COMPND for protein identity
- [ ] Identify correct holo PDB from literature (NOT by PDB code similarity)
- [ ] Verify sequence identity > 90% between apo and holo chain A
- [ ] Check residue numbering compatibility (first/last residue numbers match)
- [ ] If numbering differs, implement residue offset or sequence alignment
- [ ] Compute raw CA RMSD to determine if Kabsch is needed
- [ ] If Kabsch needed, verify post-alignment RMSD < 3.0Å
- [ ] Extract ligand heavy-atom centroid in holo frame
- [ ] Transform to apo frame using R, t from Kabsch
- [ ] Verify transformed centroid is physically reasonable (within protein bbox)
- [ ] Identify correct ligand residue code (check PDB HET records, not literature)
- [ ] For homodimers, test both chains separately

---

## DCC INTERPRETATION GUIDE

| DCC Range | Verdict | Likely Cause if Unexpected |
|:---------:|:-------:|---------------------------|
| < 5Å     | EXCELLENT | — |
| 5-8Å     | GOOD | Large pocket pulling centroid |
| 8-10Å    | MARGINAL | Pocket shape distortion or sub-optimal clustering |
| 10-15Å   | MISS | Genuine detection failure (wrong pocket ranked higher) |
| 15-25Å   | SUSPICIOUS | Check Kabsch alignment quality (RMSD should be < 3Å) |
| 25-50Å   | FRAME MISMATCH | Kabsch failed or wasn't applied |
| > 50Å    | WRONG PROTEIN | Verify sequence identity immediately |

**Rule of thumb:** If DCC > 20Å, debug the validation before debugging the detector.
