# STATISTICAL ANALYSIS PLAN — BLIND VALIDATION
**Version:** 1.0  
**Locked:** 2026-05-13 UTC  
**Repo HEAD:** f8f368f6b83e118126e691626a823866e49906f5

---

## Primary endpoint

**Success Rate at k (SR@k):** fraction of N blind targets where PRISM4D places ≥1 correct site (shell overlap ≥ threshold) within its top-k ranked predictions.

Primary: SR@5@8Å  
Secondary: SR@1@8Å, SR@3@8Å, SR@5@6Å, SR@5@4Å

---

## Scoring framework (locked, matched to publication run)

### Shell overlap
- Holo ligand heavy atoms (≥6) after Kabsch alignment to apo (RMSD ≤ 5.0 Å, seqid ≥ 0.50)
- Shell: all protein residues with any atom within cutoff (4.0, 6.0, 8.0 Å)
- Hit = Jaccard(PRISM_manifold_residues ∩ holo_shell_residues) / Jaccard ≥ defined threshold

### Ranking
Lexicographic: persistence → pass_fraction → stability → quality (frozen at prediction time)

---

## Comparative analysis

All three methods (PRISM4D, fpocket, P2Rank) scored under identical framework:
- Same apo PDB input
- Same holo validation set
- Same shell cutoffs
- Same alignment gates

Report pairwise comparison table. Fisher's exact test on 2×2 contingency (hit/miss × method) for each cutoff.

---

## LORO analysis

Leave-one-reference-out: for each target with ≥2 holo references, withhold one, build support from rest, test if withheld is recovered. Report RR@LORO = mean reconstruction rate. Expected: ≥ 0.50 indicates consistent site detection across binding conformations.

---

## Family recurrence analysis

Union-find collapse: sites that co-validate same ligand instances are collapsed. Report:
- Total sites detected
- Sites after family collapse
- Ligand-class coverage per target

---

## Null controls (pair-breaking null)

If null control script is available:
1. For each target, take PRISM4D ranked site list
2. Randomly permute site ranks (10,000 iterations)
3. Compute SR@k for each permutation → null distribution
4. Empirical p-value = fraction of null draws ≥ observed SR@k
5. Bonferroni correction for k=5 shell cutoffs × 3 rank depths = 15 comparisons

If n_iterations < 10,000 (reduced power), document explicitly.

---

## Causal driver enrichment

For sites with driver_residue_id in kcc_visualization.json:
- Translate topology ID → PDB author residue number
- Check if driver residue is in holo shell at 4/6/8 Å
- Report driver enrichment rate = fraction of correctly-predicted sites where driver is in shell

---

## Hard negative pass criterion

ADRB2 (B10): PRISM4D must not falsely predict a cryptic pocket at the orthosteric site of the inactive-state receptor. Pass if SR@5@8Å for ADRB2 orthosteric holo references = 0. Report top-5 detected sites for manual inspection.

---

## Report table structure (final)

| Method | SR@1@4Å | SR@1@6Å | SR@1@8Å | SR@5@4Å | SR@5@6Å | SR@5@8Å | LORO_RR | p-value |
|--------|---------|---------|---------|---------|---------|---------|---------|---------|
| PRISM4D | — | — | — | — | — | — | — | — |
| fpocket | — | — | — | — | — | — | N/A | N/A |
| P2Rank | — | — | — | — | — | — | N/A | N/A |

---

## Pre-registration note

This SAP was written and committed to git history before any holo structure was accessed. The SAP commit hash serves as pre-registration timestamp.
