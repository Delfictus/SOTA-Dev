# TARGET SELECTION PROTOCOL — BLIND VALIDATION
**Version:** 1.0  
**Locked:** 2026-05-13 UTC  
**Repo HEAD:** f8f368f6b83e118126e691626a823866e49906f5

---

## Principle

Targets are selected blind: the selector (PRISM4D operator) has not examined holo structures for the chosen PDB IDs. Apo structure identity is public (necessary for simulation); ligand-bound validation structures are withheld until post-freeze.

---

## Selection criteria

| Criterion | Requirement |
|-----------|-------------|
| Apo structure | Crystal structure, resolution ≤ 3.0 Å, no ligand in target pocket |
| Holo validation | ≥2 independent holo structures with small-molecule ligand at known cryptic/allosteric/binding site |
| Ligand MW | ≥ 150 Da (drug-like; excludes cofactor-only pockets) |
| Chain length | ≥ 100 residues (avoids trivially small binding proteins) |
| Publication target overlap | ZERO — no target from the 10-target pub run (KRAS_G12C, Kv3.1, p53_Y220C, AKT1, TEAD3, TRPV1, GLP1R, MCL1, STING, M4R) |
| Diversity | Must span ≥5 distinct protein classes (see required classes below) |

---

## Required target classes (10 total)

| Slot | Class | Rationale |
|------|-------|-----------|
| 1 | RAS-family | Tests allosteric switch pocket detection in same gene family as KRAS pub run |
| 2 | Kinase (non-AKT) | Tests DFG-out / C-helix cryptic detection |
| 3 | Ion channel (non-Kv3.1) | Tests transmembrane hydrophobic pocket detection |
| 4 | PPI interface | Tests shallow/flat site detection |
| 5 | Mutant TP53 (non-Y220C) | Tests thermodynamic rescue pocket detection |
| 6 | cGAS or STING-adjacent | Tests nucleotide analog/allosteric site detection |
| 7 | TEAD or nuclear receptor (non-TEAD3) | Tests palmitate/coactivator site |
| 8 | E3 ligase/adapter | Tests buried substrate-binding site |
| 9 | Protease exosite | Tests secondary binding site detection |
| 10 | GPCR hard negative | Expected: no cryptic pocket detectable in inactive state apo form |

---

## Selected blind targets (locked 2026-05-13)

| # | Target | Class | Apo PDB | Chain | Expected_residues | multi_stream | Hard negative? | Notes |
|---|--------|-------|---------|-------|-------------------|-------------|----------------|-------|
| B01 | HRAS_Q61H | RAS-family | 4L9S | A | 166 | 8 | NO | GDP/MG stripped; switch II pocket target |
| B02 | CDK2_allosteric | Kinase | 1HCL | A | 298 | 8 | NO | ATP-site apo; C-helix allosteric target |
| B03 | Kv1.2 | Ion channel | 3LUT | A | ~320 | 20 | NO | Paddle chimera; use chain A; NAP stripped |
| B04 | MDM2 | PPI | 1YCR | A | ~109 | 8 | NO | Chain A only; chain B (p53 peptide) stripped |
| B05 | TP53_apo | WT p53 allosteric | 2OCJ | A | ~219 | 8 | NO | WT p53 core domain apo; L1/H2 allosteric target; no cancer mutant apo crystal found |
| B06 | cGAS | STING-adjacent | 4KM5 | A | 212 | 8 | NO | ZN structural; apo nucleotide-binding site |
| B07 | TEAD1 | TEAD/nuclear | 3KYS | A | ~177 | 8 | NO | Chain A only; palmitate (P1L) stripped — exact site target |
| B08 | CRBN | E3 ligase | 4TZ4 | A | ~317 | 8 | NO | Chain A only; LVY (imide drug) stripped; tri-Trp cavity target |
| B09 | Thrombin_exosite | Protease exosite | 1PPB | H | ~232 | 8 | NO | Chain H only; PPACK (0G6) stripped; exosite I free |
| B10 | ADRB2 | GPCR hard neg | 2RH1 | A | ~365 | 20 | YES | Carazolol/cholesterol stripped; orthosteric cavity preformed |

---

## Apo selection rules

1. Apo = no small-molecule ligand within 12 Å of any detected pocket centroid in the apo PDB.
2. Crystal contacts, cryo-protectants (PEG, glycerol), and ions are acceptable in the apo PDB if they are not within the target pocket.
3. If the apo PDB contains a peptide or disulfide-linked fragment at the binding site, it is NOT acceptable as apo — select an alternate apo PDB.
4. For mutant targets, the mutation must be present in the apo structure (do not simulate WT if the target is a mutant).

---

## GPCR hard negative protocol (B10)

ADRB2 (2RH1) is the designated hard negative:
- Inactive state, inverse agonist removed (T4L fusion; use only chain A of protein)
- No known cryptic small-molecule pocket in inactive-state apo
- Expected PRISM4D output: 0–2 sites, no shell overlap with orthosteric holo references
- Scoring cutoff: SR@8Å < 0.20 considered a true negative pass
- Report alongside GLP1R pub hard negative result

---

## Blinding attestation

By running this pipeline, the operator confirms:
- No holo structures for B01–B10 have been examined
- Site predictions will be generated from apo structures only
- Freeze happens before any holo coordinate is accessed
- Scoring is performed only after freeze SHA256 manifest is committed

---

## Holo references (WITHHELD until post-freeze)

Holo reference PDB IDs are recorded in `BLIND_HOLO_REFERENCES.md` (file kept separate, not opened until post-freeze scoring phase). The validator will download and align these automatically.
