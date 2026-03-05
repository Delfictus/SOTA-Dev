# PRISM-4D Preprint v3.3 -- Document Quality Grading Matrix

**Audit Date**: 2026-02-22
**Source**: `PRISM4D_Preprint_v3-3.pdf` (18 pages, Serfaty 2026)
**Grading Standard**: Nature Methods / Bioinformatics / JCTC submission readiness
**Grading Scale**: 1-10 (1=unpublishable, 5=major revision, 7=minor revision, 9+=accept)

---

## A. STRUCTURAL INTEGRITY

| # | Dimension | Score | Status | Key Issues |
|---|-----------|-------|--------|------------|
| A1 | Title accuracy | 5/10 | REVISE | "Neuromorphic" is non-standard usage; term implies hardware (Loihi/SpiNNaker) not LIF threshold detectors. Suggest "spike-event-driven" |
| A2 | Abstract: structured | 7/10 | OK | Background/Methods/Results/Conclusions present and organized |
| A3 | Abstract: claims substantiated | 3/10 | FAIL | "100% accuracy" at 10A is inflated; $999 price in abstract is marketing; "democratizes" is advocacy |
| A4 | Introduction: gap established | 7/10 | OK | Cryptic pocket vs static-structure gap clearly articulated |
| A5 | Introduction: prior work fairness | 4/10 | REVISE | fpocket/P2Rank are free/open-source CPU tools, not "$100M+" methods; cost barrier applies only to MD approaches |
| A6 | Introduction: contribution statement | 5/10 | REVISE | "Eleven techniques without precedent" conflates engineering (Rust, RT cores) with scientific innovation |
| A7 | Methods: reproducibility from text | 3/10 | FAIL | No equation numbers; undefined F, eta; missing timestep, cutoff, water model, thermostat coupling |
| A8 | Methods: parameter justification | 4/10 | REVISE | 42 kcal/mol burst energy underivedl; LIF threshold=0.5 units unspecified; quality/druggability scores undefined |
| A9 | Results: claims supported by data | 5/10 | REVISE | N=12 is tiny; no error bars on DCC; beta-catenin "novel" is unvalidated prediction |
| A10 | Results: statistics correctly reported | 3/10 | FAIL | No CI on 100% rate; reproducibility only for 1 target; no p-values anywhere |
| A11 | Discussion: contextualization | 4/10 | REVISE | Reads as advocacy not analysis; Sec 5.1 presents interpretation as established fact |
| A12 | Discussion: limitations | 3/10 | FAIL | One paragraph; missing: membrane proteins, IDP, GPU vendor lock-in, 50K artifacts, parameter sensitivity, FPR |
| A13 | References: sufficiency | 2/10 | FAIL | 13 refs for 18 pages claiming 11 innovations. Need 40+ minimum |
| A14 | References: completeness | 2/10 | FAIL | Missing: AMBER ff14SB (Maier 2015), DBSCAN (Ester 1996), OptiX, watershed, LIF neuron, SiteMap, DeepSite, LIGSITE, SSTMap, PocketMiner comparison |
| A15 | References: formatting | 5/10 | REVISE | Ref [13] incomplete (no named authors, just "bioRxiv preprint") |

**Section A Mean: 4.1/10**

---

## B. SCIENTIFIC RIGOR

| # | Dimension | Score | Status | Key Issues |
|---|-----------|-------|--------|------------|
| B1 | Error bars / CIs | 2/10 | FAIL | Zero error bars on any DCC value; reproducibility shown for 1/12 targets only |
| B2 | Negative controls | 2/10 | FAIL | No true negatives (proteins known to have zero cryptic pockets) |
| B3 | False positive rate | 1/10 | FAIL | Never reported. SIRPa has 17 detected sites -- what are the other 16? |
| B4 | Benchmark size | 3/10 | FAIL | N=12 vs CryptoSite N=55, PocketMiner N=~100. Far below community standard |
| B5 | Benchmark fairness | 3/10 | FAIL | Only P2Rank compared head-to-head; FTMap/fpocket/PocketMiner/DeepSite absent from benchmark |
| B6 | Comparison fairness | 3/10 | FAIL | 18min GPU MD vs 3sec CPU geometry compared as equivalent; Table 8 compares pocket detection vs general MD hardware |
| B7 | Reproducibility: code | 1/10 | FAIL | Private repo, no release timeline, no embargo deposit |
| B8 | Reproducibility: data | 3/10 | FAIL | "Available upon request" violates open science policy of all target journals |
| B9 | Reproducibility: protocol | 4/10 | REVISE | Command line given but prep script unavailable; force field details incomplete |
| B10 | Statistical significance | 1/10 | FAIL | No hypothesis tests; no p-values; no bootstrap CIs on detection rates |

**Section B Mean: 2.3/10**

---

## C. WRITING QUALITY

| # | Dimension | Score | Status | Key Issues |
|---|-----------|-------|--------|------------|
| C1 | Jargon defined at first use | 5/10 | REVISE | LIF, BVH, CCNS, DCC, EFP used before definition; SNDC never fully expanded in abstract |
| C2 | Tone: scientific appropriateness | 2/10 | FAIL | "democratize" 5x, "Hero Target", "$305M opportunity", "the future is neuromorphic", product branding on title page |
| C3 | Tone: advocacy vs reporting | 2/10 | FAIL | Sec 5.4 is 2-page advocacy piece; Conclusion reads as manifesto; Table 8 formatted as advertisement |
| C4 | Figure: publication quality | 4/10 | REVISE | Fig 4 has PyMOL GUI visible; Fig 10 is infographic not figure; Fig 5 lacks atom labels |
| C5 | Figure: captions adequate | 6/10 | OK | Most captions acceptable; Fig 4-6 captions could be more detailed |
| C6 | Table: formatting | 5/10 | REVISE | Color-coded text in Tables 1, 6, 8 fails grayscale; green highlighting biases toward PRISM-4D |
| C7 | Equations numbered | 1/10 | FAIL | Zero equations numbered; at least 4 key equations present inline without numbering |
| C8 | Acronym consistency | 6/10 | OK | Generally consistent after first use, but first uses are sometimes in wrong location |

**Section C Mean: 3.9/10**

---

## D. NOVELTY CLAIM AUDIT

| # | Claimed Innovation (Table 7) | Verdict | Rationale |
|---|------------------------------|---------|-----------|
| D1 | SNDC v9 4-Stage GPU Pipeline | VALID | Novel integration of RT-DBSCAN + watershed + eikonal + peak tracking for spike data |
| D2 | In Silico UV Pump-Probe Spectroscopy | VALID | No prior computational analogue of wavelength-specific UV perturbation for pocket detection |
| D3 | 3-Channel Neuromorphic Spike Detection | VALID | UV/LIF/EFP temporal isolation is genuinely novel in pocket detection |
| D4 | UV-Activated Benzene Cosolvent Probing | VALID | Combining cosolvent probes with active UV energy deposition is novel |
| D5 | 5-Phase Cryo-Thermal Hysteresis | PARTIALLY VALID | Cryo-thermal cycling exists in experiment; computational implementation is novel but heating to 300K from 50K in ~6K steps is physically aggressive |
| D6 | Hydrophobic Exclusion Water Inference | PARTIALLY VALID | Concept is valid; "85-90% accuracy" claim is unsubstantiated (no validation data in paper) |
| D7 | RT-Core Accelerated Spatial Clustering | ENGINEERING | Using RT cores for BVH traversal is performance optimization, not scientific innovation |
| D8 | Intensity^2-Weighted Centroid | MINOR | Weighted centroid is standard; I^2 weighting is a reasonable but incremental choice |
| D9 | Electrostatic Flux Probe (EFP) | VALID | First electrostatic event channel in pocket detection |
| D10 | Covalent Warhead Residue ID | TRIVIAL | Flagging Cys/Lys/Ser/His near a pocket is a database lookup, not innovation |
| D11 | Watershed + Eikonal BFS | PARTIALLY VALID | Watershed segmentation well-known; application to spike density field is modestly novel |

**Genuinely Novel: 5/11 | Partially Valid: 3/11 | Engineering/Trivial: 3/11**

---

## E. TITLE PAGE ISSUES

| Issue | Severity | Recommendation |
|-------|----------|----------------|
| "SNDC v9 Engine" product branding | HIGH | Remove. Use "PRISM-4D" only |
| "Commit b7a36179" for private repo | MEDIUM | Remove until code is public |
| "SAM.gov Registered" | HIGH | Remove. Government contractor credentials have no place in scientific preprint |
| "Active UEI & CAGE" | HIGH | Remove. Same as above |
| Subtitle format "A neuromorphic..." | MEDIUM | Remove or move to abstract |

---

## COMPOSITE SCORES

| Section | Score | Weight | Weighted |
|---------|-------|--------|----------|
| A. Structural Integrity | 4.1/10 | 1.0 | 4.1 |
| B. Scientific Rigor | 2.3/10 | 1.5 | 3.5 |
| C. Writing Quality | 3.9/10 | 1.0 | 3.9 |
| D. Novelty Claims | 5.5/10 | 0.5 | 2.8 |
| **COMPOSITE** | | | **3.7/10** |

**Verdict: MAJOR REVISION required before submission to any peer-reviewed journal.**

The paper contains genuine scientific contributions (UV pump-probe, 3-channel spike detection, SIRPa detection, RT-core clustering) buried under promotional language, insufficient benchmarking, and incomplete methods. The science is potentially publishable; the manuscript in its current form is not.
