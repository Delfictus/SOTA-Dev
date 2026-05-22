# FIGURE GENERATION PLAN — BLIND VALIDATION
**Version:** 1.0  
**Locked:** 2026-05-13 UTC

---

## Figure list

| Fig | Type | Contents | Priority |
|-----|------|----------|---------|
| BV-F1 | Summary table | SR@k (1/3/5 × 4/6/8Å) for PRISM4D vs fpocket vs P2Rank | REQUIRED |
| BV-F2 | Bar chart | Per-target SR@5@8Å for all 10 targets | REQUIRED |
| BV-F3 | Structural overlay | Representative hit target (best-scoring, e.g., B01 HRAS) | REQUIRED |
| BV-F4 | Structural overlay | Second representative hit | REQUIRED |
| BV-F5 | Hard negative | ADRB2 — sites detected, no orthosteric overlap | REQUIRED |
| BV-F6 | LORO summary | Reconstruction rate per target | SUPPLEMENTAL |
| BV-F7 | Family collapse | Family tree per target with ligand class coverage | SUPPLEMENTAL |
| BV-F8 | Null control | Observed SR@k vs null distribution (if available) | SUPPLEMENTAL |
| BV-F9 | Causal driver | Representative driver residue enrichment panel | SUPPLEMENTAL |

---

## BV-F1: SR@k comparison table

Generate after post-freeze scoring:
```python
# From AGGREGATE_PRISM_VS_HOLO.csv + AGGREGATE_BASELINE_VS_HOLO.csv
# Compute SR@k for each method × each (k, shell_cutoff) combo
# Format as publication-grade LaTeX or markdown table
```

---

## BV-F2: Per-target bar chart

X-axis: B01–B10 targets  
Y-axis: SR@5@8Å  
Color: PRISM4D (blue), fpocket (gray), P2Rank (orange)  
Annotate hard negative (B10) bar separately.

---

## BV-F3/F4: Structural overlays

Use validator `pymol_overlay.pml` + publication renderer.  
Select targets with highest shell overlap (best visual story).  
Panels: apo cartoon + PRISM manifold + holo ligand.

---

## BV-F5: Hard negative ADRB2

Show apo ADRB2 with PRISM4D detected sites highlighted.  
Overlay orthosteric holo (active-state ligand position from validation holo).  
Demonstrate no overlap between PRISM sites and orthosteric holo shell.

---

## Data pipeline for figures

```
post_freeze_validation/
  AGGREGATE_PRISM_VS_HOLO.csv
  AGGREGATE_BASELINE_VS_HOLO.csv
       ↓
  generate_blind_validation_figures.py
       ↓
  reports/figures/
    BV_F1_SR_table.pdf
    BV_F2_per_target_bar.pdf
    BV_F5_hard_negative_ADRB2.png
```

Figure generation script: `scripts/quarantine/generate_blind_validation_figures.py` (to be written).

---

## Figure resolution specs

- PNG panels: 300 DPI minimum
- PyMOL: `ray 2400, 2000` for publication quality
- PDF figures: vector (matplotlib savefig with backend='pdf')

---

## Caption templates

**BV-F1:** "Blind validation success rates (SR@k) for PRISM4D, fpocket, and P2Rank across 10 new targets. Shell cutoff: [X] Å from holo ligand heavy atoms. SR@k = fraction of targets with ≥1 holo-validated site in top-k predictions."

**BV-F5:** "ADRB2 hard negative case. PRISM4D detected [N] candidate sites in the inactive-state apo receptor; none overlapped with the orthosteric binding site from active-state validation structures (SR@5@8Å = 0)."
