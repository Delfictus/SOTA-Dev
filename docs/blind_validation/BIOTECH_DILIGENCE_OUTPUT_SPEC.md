# BIOTECH DILIGENCE OUTPUT SPECIFICATION
**Version:** 1.0  
**Locked:** 2026-05-13 UTC

---

## Purpose

Defines the package of materials producible from the blind validation for biotech/pharma due diligence review.

---

## Tier 1: Public-ready materials (no IP restriction)

| Document | Source | Contains |
|----------|--------|---------|
| BLIND_VALIDATION_FINAL_REPORT.md | docs/blind_validation/reports/ | SR@k table, LORO, family collapse, null controls, per-target summaries |
| SR@k comparison table (BV-F1) | Generated post-freeze | PRISM4D vs fpocket vs P2Rank |
| Per-target validation cards | One per B01–B09 | Site count, top site shell overlap, therm_class, causal driver |
| Hard negative documentation | B10 ADRB2 | Demonstrates specificity |
| Method lock document (LAST10_METHOD_LOCK.md) | docs/blind_validation/ | Exact protocol reproducibility |

---

## Tier 2: Due diligence deep dive (NDA-appropriate)

| Document | Source | Contains |
|----------|--------|---------|
| binding_sites.json per target (frozen) | frozen/ dirs | Full site predictions with coordinates, rankings, features |
| kcc_visualization.json per target | frozen/ dirs | Manifold residues, causal driver, centroids |
| prism_therm.json per target | frozen/ dirs | Thermodynamic classifications |
| Design briefs (if generated) | prism_canonical.py output | Anchor points, growth vectors, pocket profiles |
| PyMOL sessions (.pse) | Structural visualization | Interactive 3D exploration |
| run.log per target | run/ dirs | Full execution record |

---

## Tier 3: Source code / IP (patent-protected, access by permission only)

- PRISM4D engine source (Rust/CUDA)
- fused_engine.rs (CryoUV protocol implementation)
- nhs_rt_full.rs (engine binary)
- Feature registry (14-feature canonical)

Not included in standard diligence package. Reference to patent filings.

---

## Diligence package assembly

```bash
DILIGENCE_DIR=/mnt/storage/prism-outputs/blind_validation/diligence_package_$(date +%Y%m%d)
mkdir -p ${DILIGENCE_DIR}/{tier1,tier2}

# Tier 1
cp docs/blind_validation/reports/BLIND_VALIDATION_FINAL_REPORT.md ${DILIGENCE_DIR}/tier1/
cp docs/blind_validation/LAST10_METHOD_LOCK.md ${DILIGENCE_DIR}/tier1/

# Tier 2
for T in B01 B02 B03 B04 B05 B06 B07 B08 B09 B10; do
    mkdir -p ${DILIGENCE_DIR}/tier2/${T}
    cp /mnt/storage/prism-outputs/blind_validation/${T}*/frozen/*.binding_sites.json \
       ${DILIGENCE_DIR}/tier2/${T}/
    cp /mnt/storage/prism-outputs/blind_validation/${T}*/frozen/*.kcc_visualization.json \
       ${DILIGENCE_DIR}/tier2/${T}/
done
```

---

## Claims supportable in diligence conversations

From blind validation alone:
- SR@k performance metrics (quantitative, with methodology)
- Comparison to standard geometry-only methods
- Prospective prediction capability (freeze timestamp evidence)
- Cross-target generalization (10 diverse target classes)
- Hard negative specificity (ADRB2)
- Reproducibility (--replica-seed 42 set; methodology locked)

Not supportable from computational data alone:
- In vitro binding affinity at predicted sites
- Cellular target engagement
- In vivo target modulation
