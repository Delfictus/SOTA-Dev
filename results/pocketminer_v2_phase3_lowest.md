# PRISM Cryptic Detection v2 - v2-phase3-lowest Report
Generated: 2026-01-11 00:45:50 UTC
Git Commit: 5d2c75b

## Executive Summary

| Metric | v1 Baseline | This Run | Delta | Target | Status |
|--------|-------------|----------|-------|--------|--------|
| ROC AUC | 0.455 | 0.462 | +0.007 | >0.90 | ❌ |
| PR AUC | 0.083 | 0.084 | +0.001 | >0.20 | ❌ |
| Success Rate | 84.8% | 84.8% | -0.017% | >90% | ❌ |
| Best F1 | 0.545 | 0.505 | -0.040 | >0.70 | ❌ |

## Configuration Used

- ANM: n_modes=30, n_conformations=100, amplitude_scale=5.0
- EFE: prior=0.070, epistemic_weight=0.4, pragmatic_weight=0.6
- Threshold: adaptive=true, z=0.0, floor=0.15
- Clustering: graph=true, distance=8.0Å, min_size=2
- HMC: enabled=true
- PRISM-ZrO: enabled=true

## Per-Structure Results (Top 10 by F1)

| PDB | Recall | Precision | F1 | Clusters | Ground Truth | Status |
|-----|--------|-----------|----|---------:|-------------:|--------|
| 1kx9B | 81.2% | 36.6% | 0.505 | 3 | 32 | ✅ |
| 1tvqA | 71.4% | 29.1% | 0.413 | 5 | 35 | ✅ |
| 2w9tA | 62.5% | 24.5% | 0.352 | 5 | 40 | ✅ |
| 3qxwB | 85.0% | 18.5% | 0.304 | 2 | 20 | ✅ |
| 5h9aA | 77.3% | 17.9% | 0.291 | 4 | 22 | ✅ |
| 3fvjA | 66.7% | 16.9% | 0.269 | 3 | 21 | ✅ |
| 5uxaA | 76.7% | 15.9% | 0.264 | 10 | 43 | ✅ |
| 2oy4A | 69.2% | 16.2% | 0.263 | 5 | 26 | ✅ |
| 6e5dA | 82.6% | 15.3% | 0.259 | 6 | 23 | ✅ |
| 2hq8B | 72.0% | 14.9% | 0.247 | 4 | 25 | ✅ |

## Failure Analysis (Recall < 30%)

| PDB | Recall | Issue | Hypothesis |
|-----|--------|-------|------------|
| 2fd7A | 0.0% | Very low precision | Threshold too high for this structure |
| 4tqlA | 0.0% | Very low precision | Threshold too high for this structure |
| 2alpA | 0.0% | Very low precision | Threshold too high for this structure |
| 1ammA | 0.0% | Very low precision | Threshold too high for this structure |
| 4hjkA | 0.0% | Very low precision | Threshold too high for this structure |
| 1igdA | 0.0% | Very low precision | Threshold too high for this structure |
| 1hcl | 0.0% | Very low precision | Threshold too high for this structure |

## Score Distribution Analysis

- Mean cryptic score: 0.609
- Std cryptic score: 0.178
- Median: 0.669, Q25: 0.567, Q75: 0.724
- Adaptive threshold range: [0.543, 0.754]

## Timing Breakdown

| Phase | Mean (ms) | Std (ms) | % Total |
|-------|-----------|----------|--------|
| ANM Generation | 0.0 | 0.0 | 0.0% |
| Feature Extraction | 10.5 | 0.0 | 0.8% |
| Scoring | 6.3 | 0.0 | 0.5% |
| Clustering | 4.2 | 0.0 | 0.3% |
| HMC Refinement | 1040.3 | 0.0 | 79.4% |
| PRISM-ZrO Scoring | 258.5 | 0.0 | 19.7% |
| **Total** | **1310.5** | **0.0** | **100%** |

## Memory Usage

- Peak GPU memory: 0.0 MB
- Peak CPU memory: 0.0 MB

## Recommendations for Next Phase

- [ ] ROC AUC still low - consider enabling HMC refinement
- [ ] PR AUC below target - improve precision with stricter clustering

