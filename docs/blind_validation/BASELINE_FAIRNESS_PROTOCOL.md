# BASELINE FAIRNESS PROTOCOL
**Version:** 1.0  
**Locked:** 2026-05-13 UTC

---

## Purpose

Ensures fpocket and P2Rank are given identical conditions to PRISM4D so that performance differences are attributable to method, not input.

---

## Equal-input rule

All three methods receive the same clean apo PDB:
- Same file: `${PDBID}_clean.pdb` (output of prism-clean.py)
- Same chain
- Same protonation / altconf state
- No ligand coordinates provided to any method

PRISM4D additionally receives the topology JSON and NMA modes — these are derived from the same clean PDB and are not holo-informed.

---

## Baseline invocation (locked)

**fpocket:**
```bash
fpocket -f ${PDBID}_clean.pdb
```
No custom parameters. Default algorithm. Version 4.2.3 (snap).

**P2Rank:**
```bash
/opt/p2rank/prank predict -f ${PDBID}_clean.pdb -o p2rank/ -threads 4
```
No custom parameters. Default model. Version 2.4.2.

---

## Scoring framework (identical for all methods)

Embedded in `prism_pub_baseline_validator.py`:
- Shell cutoffs: 4.0, 6.0, 8.0 Å from holo ligand heavy atoms (≥6 atoms)
- Alignment gate: RMSD ≤ 5.0 Å, seqid ≥ 0.50
- MIN_LIGAND_HEAVY = 6
- Scoring: Jaccard intersection of predicted residues vs holo shell residues

fpocket top sites sourced from `*_out/*_info.txt` (pocket rank = fpocket druggability score rank).  
P2Rank top sites sourced from `*_predictions.csv` (pocket rank = score column rank).  
PRISM4D sites ranked lexicographically (persistence → pass_fraction → stability → quality).

All three capped at top-5 for SR@k computation.

---

## What NOT to do

- Do NOT tune fpocket or P2Rank parameters to improve their scores.
- Do NOT apply PRISM4D-derived topology information to guide fpocket site selection.
- Do NOT use holo structure information to select which fpocket pockets to include.
- Do NOT re-rank any method's output post-hoc.

---

## Documented asymmetries (unavoidable)

| Asymmetry | Description | Impact |
|-----------|-------------|--------|
| PRISM4D uses MD + NMA perturbation | fpocket/P2Rank are geometry-only | PRISM4D has additional information about conformational flexibility |
| PRISM4D uses CUDA simulation | fpocket/P2Rank use CPU geometry | Computational cost is NOT a fairness issue — only input fairness |
| PRISM4D output includes thermodynamic class | fpocket/P2Rank do not | PRISM4D therm_class not used for shell scoring (blind scoring only) |

These asymmetries are the point of the comparison — they represent the value of the perturbation-driven approach vs geometry-only methods. Report them explicitly.

---

## Reporting

State verbatim: "fpocket (v4.2.3) and P2Rank (v2.4.2) were applied to the same clean apo structures under default parameters and scored under identical ligand-shell criteria."
