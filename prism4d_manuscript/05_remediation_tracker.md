# PRISM-4D Live Remediation & Action Pathway Tracker

**Document**: PRISM4D_Preprint_v3-3.pdf -> Target: v4.0 submission-ready
**Created**: 2026-02-22
**Status Legend**: BLOCKED | TODO | IN_PROGRESS | DONE | DEFERRED

---

## PRIORITY 1: SUBMISSION BLOCKERS (Must fix before any journal submission)

| ID | Task | Section | Status | Impact | Notes |
|----|------|---------|--------|--------|-------|
| P1.1 | Strip ALL marketing language (see Reframing Ledger) | All | TODO | Prevents desk rejection | ~20 substitutions + ~10 deletions |
| P1.2 | Remove title page branding (SNDC v9 Engine, SAM.gov, UEI/CAGE) | Title | TODO | Prevents desk rejection | Replace with standard academic header |
| P1.3 | Number ALL equations (minimum 4, likely 8-10 after expansion) | Methods | TODO | Required by all target journals | Inline equations -> display equations |
| P1.4 | Define quality score and druggability score formulas | Methods | TODO | Currently undefined black boxes | Extract from persistent_engine.rs |
| P1.5 | Expand references from 13 to 40+ | Refs | TODO | 13 refs = auto-reject | See missing citations list |
| P1.6 | Add error bars (mean +/- std) to ALL DCC values in Table 1 | Results | TODO | Statistical reporting requirement | Run from existing 8-stream data |
| P1.7 | Report false positive rate (FPR) | Results | TODO | Basic benchmark requirement | Sites detected per protein minus true positives |
| P1.8 | Add negative controls (1-2 proteins with no cryptic pockets) | Results | BLOCKED | Requires GPU runs | Candidates: 1UBQ (ubiquitin), 1LYZ (lysozyme) |
| P1.9 | Fix Figure 4 (remove PyMOL GUI from screenshot) | Figures | TODO | Publication-quality requirement | Re-render with `ray 2400, 1800` |
| P1.10 | Complete force field specification in Methods | Methods | TODO | Reproducibility requirement | Timestep, cutoff, water model, thermostat coupling |

---

## PRIORITY 2: SCIENTIFIC STRENGTHENING (Significantly improves review outcome)

| ID | Task | Section | Status | Impact | Notes |
|----|------|---------|--------|--------|-------|
| P2.1 | Run CryptoSite benchmark (~55 proteins) | Results | DEFERRED | Gold-standard benchmark comparison | Major computational effort |
| P2.2 | Run PocketMiner comparison on same 12 proteins | Results | TODO | Most relevant ML competitor | PocketMiner is open-source |
| P2.3 | Run FTMap on same 12 proteins | Results | TODO | Most relevant physics competitor | FTMap has web server |
| P2.4 | Add bootstrap 95% CI on detection rates | Results | TODO | Statistical rigor | e.g., 4/11 at <5A -> CI [11%, 69%] |
| P2.5 | Expand Limitations section to full subsections | Discussion | TODO | Currently 1 paragraph -> need 4-5 | See list of missing limitations |
| P2.6 | Add sensitivity analysis for key parameters | Methods/Results | TODO | Reviewer will ask | tau, threshold, burst energy, epsilon |
| P2.7 | Reproduce Table 5 analysis for ALL 12 targets | Results | TODO | Currently only beta-catenin | Mean/std DCC per target across seeds |
| P2.8 | Qualify "neuromorphic" terminology or change title | Title/Throughout | TODO | Reviewers in comp neuro will object | Consider "spike-event-driven" |
| P2.9 | Add formal algorithm boxes for core methods | Methods | TODO | Standard for computational papers | SNDC, RT-DBSCAN, watershed |
| P2.10 | Reclassify "11 innovations" -> honest breakdown | Discussion | TODO | Currently overclaimed | ~5 novel + ~3 novel application + ~3 engineering |

---

## PRIORITY 3: POLISH (Improves quality but not blocking)

| ID | Task | Section | Status | Impact | Notes |
|----|------|---------|--------|--------|-------|
| P3.1 | Define all acronyms at first use in abstract | Abstract | TODO | Clarity | LIF, DCC, SNDC, BVH, EFP |
| P3.2 | Replace color-coded table text with symbols | Tables 1,6,8 | TODO | Grayscale compatibility | Use symbols/bold instead of color |
| P3.3 | Add residue labels to Figure 5 (CYS240, CYS278, TRP242) | Figures | TODO | Reader comprehension | |
| P3.4 | Replace Figure 10 infographic with standard figure | Discussion | TODO | Scientific figure standards | Innovation stack pyramid -> comparison table |
| P3.5 | Add N/C terminal labels and scale bar to Figure 6 | Figures | TODO | Standard structural figure elements | |
| P3.6 | Complete Reference [13] with proper author names/title | Refs | TODO | Currently incomplete | SIRPa XChem bioRxiv preprint |
| P3.7 | Specify Kabsch superposition details (which atoms) | Methods | TODO | Reproducibility | Backbone CA? All heavy atoms? |
| P3.8 | Remove "Hero Target" -> "Case Study" | Results | TODO | See Reframing Ledger | |
| P3.9 | Add per-target runtime table | Results | TODO | Currently only "3-18 min" range | |
| P3.10 | State units for LIF threshold (0.5 of what?) | Methods | TODO | Currently dimensionless without context | |

---

## PRIORITY 4: STRETCH GOALS (Maximal impact but high effort)

| ID | Task | Section | Status | Impact | Notes |
|----|------|---------|--------|--------|-------|
| P4.1 | Open-source code or deposit under embargo | Data Avail. | BLOCKED | Required by Nature Methods | Patent filing status unclear |
| P4.2 | Deposit benchmark data to Zenodo/Figshare | Data Avail. | TODO | "Upon request" insufficient | Topologies, JSONs, run logs |
| P4.3 | Add ROC/AUC analysis (DCC threshold vs detection rate curve) | Results | TODO | Standard benchmark visualization | |
| P4.4 | Validate 85-90% water density claim | Methods | BLOCKED | Requires explicit solvent comparison | Need reference dataset |
| P4.5 | Cross-validate with leave-one-out analysis | Results | DEFERRED | Statistial validation | N=12 is small for LOO |
| P4.6 | Add mixed-solvent MD comparison (MDmix/MixMD) | Results | DEFERRED | Directly comparable approach | Significant computational effort |

---

## DEPENDENCY GRAPH

```
P1.8 (negative controls) -> BLOCKED on GPU compute time
P1.6 (error bars) -> needs per-stream DCC extraction from existing JSONs
P2.1 (CryptoSite) -> BLOCKED on significant compute + topology prep for 55 proteins
P4.1 (open-source) -> BLOCKED on patent/IP decision
P4.4 (water validation) -> BLOCKED on explicit solvent reference runs
```

---

## MISSING REFERENCES (Priority for P1.5)

### Must-add (all mentioned in text but uncited):
1. Maier, J.A. et al. (2015). ff14SB: Improving the Accuracy of Protein Side Chain... JCTC.
2. Halgren, T.A. (2009). Identifying and Characterizing Binding Sites... JCTC (SiteMap).
3. Jimenez, J. et al. (2017). DeepSite: protein-binding site predictor. Bioinformatics.
4. Hendlich, M. et al. (1997). LIGSITE: automatic and efficient detection of potential... Proteins.
5. Ester, M. et al. (1996). A density-based algorithm for discovering clusters... KDD (DBSCAN).

### Should-add (methodological foundations):
6. Parker, S.G. et al. (2010). OptiX: A General Purpose Ray Tracing Engine. SIGGRAPH.
7. Vincent, L. & Soille, P. (1991). Watersheds in digital spaces. IEEE TPAMI.
8. Gerstner, W. & Kistler, W. (2002). Spiking Neuron Models. Cambridge UP (LIF model).
9. Izaguirre, J.A. et al. (2001). Langevin stabilization of molecular dynamics. JCP.
10. Ryckaert, J.P. et al. (1977). Numerical integration of Cartesian equations of motion... JCP (SHAKE).

### Should-add (field context):
11. Bowman, G.R. et al. (2015). Quantitative comparison of alternative methods... JCTC.
12. Porter, J.R. et al. (2019). Enspara: Modeling molecular ensembles... JCTC.
13. Lexa, K.W. & Carlson, H.A. (2012). Protein flexibility in docking and SBDD. JCTC.
14. Schmidtke, P. et al. (2011). MDpocket: open-source cavity detection on MD trajectories. Bioinformatics.
15. Tan, Y.S. et al. (2012). Using ligand-mapping simulations to design a ligand... JACS.
16. Beglov, D. et al. (2018). Exploring the structural origins of cryptic sites on proteins. PNAS.

### Should-add (validation context):
17. Kuhn, B. et al. (2017). Prospective evaluation of free energy calculations for... JCTC.
18. Young, T. et al. (2007). Motifs for molecular recognition exploiting hydrophobic enclosure... PNAS (GIST/SSTMap).
19. Goodford, P.J. (1985). A computational procedure for determining... JMC (GRID).
20. Ngan, C.H. et al. (2012). FTSite: high accuracy detection... NAR.

---

## MISSING LIMITATIONS (Priority for P2.5)

Each should be 2-3 sentences in expanded Limitations section:

1. **Benchmark size**: N=12 is modest; CryptoSite (N~55) and PocketMiner benchmarks are standard
2. **No membrane protein validation**: Paper claims membrane protein support but provides no evidence
3. **No IDP validation**: Intrinsically disordered proteins/regions not tested
4. **GPU vendor lock-in**: NVIDIA CUDA only; no AMD/Intel GPU support; "accessible" claim requires qualifying
5. **Physically unrealistic temperatures**: 50K cold phase may introduce crystallization artifacts absent in biological systems
6. **Parameter sensitivity unexplored**: LIF tau, threshold, UV burst energy, epsilon range -- no sensitivity analysis
7. **False positive rate**: Not reported; 17 sites on SIRPa (only 1 is the target) suggests FPR needs quantification
8. **No comparison with most relevant methods**: PocketMiner (cryptic pocket ML), MDpocket (MD-based), mixed-solvent MD
9. **Single-chain bias**: Most targets are single-chain; multi-chain complexes show marginal performance (ERa 9.5A)
10. **Runtime tradeoff**: 3-18 min vs 3 sec for geometric methods is 60-360x slower; acceptable for discovery but limits screening

---

## REMEDIATION EFFORT ESTIMATE

| Priority | Items | Effort | Timeline |
|----------|-------|--------|----------|
| P1 (Blockers) | 10 | ~3-5 days writing + 1-2 days compute | Week 1 |
| P2 (Strengthening) | 10 | ~5-7 days writing + 3-5 days compute | Weeks 2-3 |
| P3 (Polish) | 10 | ~2-3 days | Week 3 |
| P4 (Stretch) | 6 | ~2-4 weeks (CryptoSite benchmark dominates) | Weeks 4-8 |
| **Total to submission-ready** | | | **~3-4 weeks (P1+P2+P3)** |
