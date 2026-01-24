# PRISM Cryptic Detection v2 - v2-lowest-thresh Report
Generated: 2026-01-10 23:26:48 UTC
Git Commit: 5d2c75b

## Executive Summary

| Metric | v1 Baseline | This Run | Delta | Target | Status |
|--------|-------------|----------|-------|--------|--------|
| ROC AUC | 0.455 | 0.459 | +0.004 | >0.90 | ❌ |
| PR AUC | 0.083 | 0.083 | +0.000 | >0.20 | ❌ |
| Success Rate | 84.8% | 76.1% | -8.713% | >90% | ❌ |
| Best F1 | 0.545 | 0.489 | -0.056 | >0.70 | ❌ |

## Configuration Used

- ANM: n_modes=30, n_conformations=100, amplitude_scale=5.0
- EFE: prior=0.070, epistemic_weight=0.4, pragmatic_weight=0.6
- Threshold: adaptive=true, z=0.0, floor=0.20
- Clustering: graph=true, distance=8.0Å, min_size=2
- HMC: enabled=true
- PRISM-ZrO: enabled=true

## Per-Structure Results (Top 10 by F1)

| PDB | Recall | Precision | F1 | Clusters | Ground Truth | Status |
|-----|--------|-----------|----|---------:|-------------:|--------|
| 1kx9B | 71.9% | 37.1% | 0.489 | 3 | 32 | ✅ |
| 1tvqA | 65.7% | 35.9% | 0.465 | 5 | 35 | ✅ |
| 5h9aA | 63.6% | 18.2% | 0.283 | 7 | 22 | ✅ |
| 2w9tA | 37.5% | 19.7% | 0.259 | 5 | 40 | ✅ |
| 6e5dA | 56.5% | 16.5% | 0.255 | 6 | 23 | ✅ |
| 3rwvA | 52.9% | 16.7% | 0.254 | 10 | 34 | ✅ |
| 2oy4A | 50.0% | 16.9% | 0.252 | 6 | 26 | ✅ |
| 3qxwB | 50.0% | 14.5% | 0.225 | 5 | 20 | ✅ |
| 3fvjA | 47.6% | 14.7% | 0.225 | 6 | 21 | ✅ |
| 3ugkA | 50.0% | 13.6% | 0.214 | 13 | 42 | ✅ |

## Failure Analysis (Recall < 30%)

| PDB | Recall | Issue | Hypothesis |
|-----|--------|-------|------------|
| 4r72A | 29.2% | Very low precision | Large cryptic region may need more conformations |
| 5za4A | 18.2% | Very low precision | Large cryptic region may need more conformations |
| 5niaA | 29.6% | Very low precision | Large cryptic region may need more conformations |
| 3p53A | 25.0% | Very low precision | May need HMC refinement for better sampling |
| 2fd7A | 0.0% | Very low precision | Threshold too high for this structure |
| 4tqlA | 0.0% | Very low precision | Threshold too high for this structure |
| 2alpA | 0.0% | Very low precision | May need HMC refinement for better sampling |
| 1ammA | 0.0% | Very low precision | Threshold too high for this structure |
| 4hjkA | 0.0% | Very low precision | Threshold too high for this structure |
| 1igdA | 0.0% | Very low precision | Threshold too high for this structure |
| 1hcl | 0.0% | Very low precision | Threshold too high for this structure |

## Score Distribution Analysis

- Mean cryptic score: 0.467
- Std cryptic score: 0.242
- Median: 0.335, Q25: 0.258, Q75: 0.718
- Adaptive threshold range: [0.353, 0.856]

## Timing Breakdown

| Phase | Mean (ms) | Std (ms) | % Total |
|-------|-----------|----------|--------|
| ANM Generation | 0.0 | 0.0 | 0.0% |
| Feature Extraction | 11.0 | 0.0 | 0.8% |
| Scoring | 6.6 | 0.0 | 0.5% |
| Clustering | 4.4 | 0.0 | 0.3% |
| HMC Refinement | 1103.8 | 0.0 | 80.3% |
| PRISM-ZrO Scoring | 258.0 | 0.0 | 18.8% |
| **Total** | **1374.1** | **0.0** | **100%** |

## Memory Usage

- Peak GPU memory: 0.0 MB
- Peak CPU memory: 0.0 MB

## Recommendations for Next Phase

- [ ] ROC AUC still low - consider enabling HMC refinement
- [ ] PR AUC below target - improve precision with stricter clustering

