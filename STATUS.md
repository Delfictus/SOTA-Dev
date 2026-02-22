# PRISM-4D Development Status

**Last Updated:** 2026-02-22
**Version:** SNDC v9 (4-stage GPU pipeline)
**Status:** Hero target validation complete. Pre-publication.

---

## Architecture: SNDC v9

4-stage GPU-accelerated binding site detection pipeline:
1. **RT-DBSCAN** — Real-time density-based spike clustering
2. **Watershed segmentation** — Sub-cluster boundary refinement
3. **Eikonal BFS** — Distance-field propagation for pocket geometry
4. **Peak centroid tracking** — Temporal convergence of binding site centroids

Core engine: NHS (Neuromorphic Holographic Stream) with AMBER force fields, multi-wavelength UV perturbation targeting aromatic residues, and leaky integrate-and-fire network for cooperative dewetting event detection.

**Canonical workflow:**
```
nhs_rt_full --multi-stream 20 --multi-scale --rt-clustering --lining-cutoff 8.0 --fast -v
```
With `RUST_LOG=info`. Prep: `scripts/prism-prep input.pdb output.topology.json` (requires OpenMM for `--use-amber`).

---

## Validation Results

### 9-Protein CryptoSite Benchmark (SNDC v9)

| # | Protein | PDB  | DCC (Å) | Accuracy |
|---|---------|------|---------|----------|
| 1 | TEM1 β-lactamase | 1JWP | <10 | ✓ |
| 2 | Protein tyrosine phosphatase 1B | 1T49 | <10 | ✓ |
| 3 | cAMP-dep protein kinase | 3DND | <10 | ✓ |
| 4 | TEM1 β-lactamase (alt) | 1PZO | <10 | ✓ |
| 5 | Interleukin-2 | 1M47 | <10 | ✓ |
| 6 | Cyt P450 CYP2B4 | 1PO5 | <10 | ✓ |
| 7 | Androgen receptor | 2PIU | <10 | ✓ |
| 8 | NPC2 cholesterol | 1NEP | <10 | ✓ |
| 9 | Chk1 kinase | 1NVR | <10 | ✓ |

**Result: 100% accuracy (9/9 proteins) at <10Å DCC threshold.**

---

### Hero Target 1: SIRPα (PDB 2WNG) — WYF Cryptic Pocket

**Context:** CD47-SIRPα "don't eat me" signal. XChem fragment screening (800+ fragments) confirmed WYF pocket is experimentally accessible. Active immuno-oncology target.

| Metric | PRISM-4D | P2Rank 2.5 |
|--------|----------|------------|
| Pockets detected | 17 | 1 |
| DCC to WYF pocket | **7.1 Å** | 9.0 Å |
| Confidence | Site 1, ActiveSite class | 0.2% probability |
| Spike count | 4,413 | N/A (ML method) |
| Volume | 743 ų | — |
| Druggability | 0.652 | Score 0.77 |
| Runtime | 18 min | 3 sec |

**P2Rank assigns 0.2% probability to the experimentally validated WYF pocket.** PRISM-4D identifies it as its top-ranked active site with 4,413 spike events.

---

### Hero Target 2: β-Catenin (PDB 2GL7) — Novel Cryptic Pocket

**Context:** Wnt/β-catenin pathway. Previously considered "undruggable" by small molecules. Parabilis Bio raised $305M (2024) targeting β-catenin degradation.

#### Site 6 — Novel Covalent-Accessible Cryptic Pocket

| Metric | PRISM-4D Site 6 | P2Rank 2.5 |
|--------|-----------------|------------|
| Quality score | **0.965** (highest) | Ranked 8/8 (last) |
| Druggability | **0.844** | 0.5% probability |
| Classification | Cryptic | — |
| Spike count | 3,242 | N/A |
| Volume | 248 ų | — |
| Anchor residues | TRP242, CYS240, CYS278, LYS204 | Not detected |
| Region | ARM repeats 2–5 (residues 204–278) | — |
| Covalent opportunity | **CYS240 + CYS278 dual cysteine** | — |

**P2Rank dismisses the TRP242/CYS covalent pocket as noise (rank 8/8, 0.5% probability).** PRISM-4D identifies it as its highest-quality site (0.965).

---

### Reproducibility Validation — β-Catenin Site 6

5 parallel runs with independent stochastic seeds (replica seeds 1–5):

| Metric | Value |
|--------|-------|
| Runs converging to Site 6 | **5/5 (100%)** |
| Mean centroid | (14.4, 11.3, 41.9) |
| Centroid std | (0.01, 0.01, 0.02) Å |
| Max pairwise DCC | **0.06 Å** |
| Mean pairwise DCC | **0.03 Å** |
| Druggability | 0.844 ± 0.000 |
| Spike count | 3,242 ± 6 (CV = 0.2%) |
| Volume | 248 ų (all runs) |
| Quality | 0.965 (all runs) |
| Classification | Cryptic (all runs) |
| Anchor persistence | **4/4 residues in all 5 runs (100%)** |
| **Verdict** | **★★★ HIGHLY REPRODUCIBLE** |

Sub-angstrom centroid convergence (0.06 Å) across stochastic runs demonstrates Site 6 is a thermodynamic feature of the energy landscape, not a computational artifact.

---

## P2Rank 2.5 Comparison Summary

State-of-the-art ML-based pocket detection (Krivák & Hoksza, 2018). Trained on static geometric features from crystal structures.

**Failure mode:** P2Rank cannot detect conformationally gated (cryptic) pockets because it relies on surface geometry of the input conformation. PRISM-4D's neuromorphic spike-driven approach detects transient pocket opening events through physics simulation, enabling cryptic pocket discovery that geometric/ML methods fundamentally cannot perform.

| Target | PRISM-4D | P2Rank 2.5 | Winner |
|--------|----------|------------|--------|
| SIRPα WYF (validated) | 7.1Å DCC, top site | 9.0Å DCC, 0.2% prob | **PRISM** |
| β-Cat Site 6 (novel) | Quality 0.965, top | Rank 8/8, 0.5% prob | **PRISM** |
| Runtime | 18 min | 3–7 sec | P2Rank |

---

## Pending Action Items

1. **[ ] File provisional patent** — $200, locks 12-month priority. DO NOT PUBLISH BEFORE FILING.
2. **[ ] Run 9-protein benchmark through P2Rank** — Systematic head-to-head comparison table.
3. **[ ] Preprint preparation** — SIRPα figure + β-catenin Site 6 + reproducibility data.

---

## Key Files

- `crates/prism-nhs/src/bin/nhs_rt_full.rs` — Main NHS engine (watershed at line ~3437)
- `scripts/prism-prep` — PDB preparation script
- Benchmark data: `sndc_benchmark/` directory

---

*Delfictus IO LLC — PRISM-4D Platform*
