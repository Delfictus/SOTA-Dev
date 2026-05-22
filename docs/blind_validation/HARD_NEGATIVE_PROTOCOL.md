# HARD NEGATIVE PROTOCOL — ADRB2 (B10)
**Version:** 1.0  
**Locked:** 2026-05-13 UTC

---

## Rationale

One of 10 blind targets must be a hard negative — a protein where PRISM4D is EXPECTED TO FAIL to detect a cryptic drug-binding pocket. This mirrors the GLP1R hard negative in the publication run. A credible validation requires demonstrating the method is not universally over-optimistic.

---

## Selected hard negative

**Target:** Beta-2 adrenergic receptor (ADRB2)  
**Apo PDB:** 2RH1 (inactive state, inverse agonist carazolol removed from structure; T4L fusion present)  
**Chain:** A  
**Class:** GPCR, Class A  

---

## Why ADRB2 is a valid hard negative

1. **Orthosteric site is preformed:** The GPCR orthosteric binding cleft is a large, open, well-formed pocket even in the apo/inactive state. PRISM4D's cryptic detection (CryoUV hysteresis) targets pockets that open transiently; a preformed pocket is not a cryptic pocket and should not be flagged as one.
2. **No cryptic allosteric small-molecule pocket documented:** The ADRB2 bitopic allosteric site (extracellular vestibule) is preformed. No buried cryptic allosteric pocket has been documented in the inactive state.
3. **Different from GLP1R:** GLP1R (pub run hard negative) is a GPCR Class B with a different mechanism of "failure." ADRB2 (Class A) tests a distinct GPCR topology.
4. **T4L fusion in 2RH1:** The T4L insertion replaces the ICL3 loop, which could create a false positive near the fusion junction. This must be excluded from site reporting.

---

## Expected PRISM4D output (hard negative pass criteria)

| Criterion | Pass |
|-----------|------|
| No site with shell overlap ≥ threshold at orthosteric holo references | SR@5@8Å = 0 |
| Any detected sites are at T4L fusion junction (excluded from scoring) | Note and exclude |
| Sites near known allosteric ECV sites are weak (pass_fraction < 0.3) | Report |

Pass: SR@5@8Å for ADRB2 orthosteric validation holos = 0.

---

## T4L handling

2RH1 includes a T4L domain fused into ICL3 (residues approximately 231–262 replaced). When running prism-clean.py:
- Keep chain A (GPCR portion)
- prism-clean.py will keep chain A including T4L residues
- Post-detection: any site whose centroid is within the T4L domain should be flagged and excluded from hard-negative scoring

Topology offset note: verify with prism-lookup-residue.py that detected sites attributed to GPCR residues are not from the T4L insertion region.

---

## Holo references for ADRB2 scoring

Used by validator post-freeze:
- 2RH1: apo/inactive (reference structure itself — no ligand at orthosteric site)
- Active-state holos: 3SN6, 4LDE, 4LDO, 3PDS (all have agonist/ligand)
- Hard negative pass: PRISM4D should NOT overlap with the agonist/ligand shells in active-state holos

---

## Reporting language

"ADRB2 (Class A GPCR) served as the hard negative case. PRISM4D detected [N] sites in the inactive-state apo receptor, none of which overlapped with the agonist-bound shell in active-state validation structures (SR@5@8Å = 0), consistent with the expected absence of cryptic pocket opening in this receptor class. This mirrors the GLP1R hard negative finding in the publication validation."

If PRISM4D incorrectly flags orthosteric sites:
"PRISM4D detected [N] sites in ADRB2, [M] of which overlapped with the orthosteric binding site. This represents a false positive for cryptic pocket detection and is discussed as a limitation of the perturbation-based hysteresis approach for preformed-pocket GPCRs."
