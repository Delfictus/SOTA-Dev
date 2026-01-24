# PRISM Cryptic Detection v2 - v2-fixed-threshold Report
Generated: 2026-01-10 23:33:55 UTC
Git Commit: 5d2c75b

## Executive Summary

| Metric | v1 Baseline | This Run | Delta | Target | Status |
|--------|-------------|----------|-------|--------|--------|
| ROC AUC | 0.455 | 0.466 | +0.011 | >0.90 | ❌ |
| PR AUC | 0.083 | 0.082 | -0.001 | >0.20 | ❌ |
| Success Rate | 84.8% | 82.6% | -2.191% | >90% | ❌ |
| Best F1 | 0.545 | 0.471 | -0.074 | >0.70 | ❌ |

## Configuration Used

- ANM: n_modes=30, n_conformations=100, amplitude_scale=5.0
- EFE: prior=0.070, epistemic_weight=0.4, pragmatic_weight=0.6
- Threshold: adaptive=true, z=0.0, floor=0.30
- Clustering: graph=true, distance=8.0Å, min_size=2
- HMC: enabled=true
- PRISM-ZrO: enabled=true

## Per-Structure Results (Top 10 by F1)

| PDB | Recall | Precision | F1 | Clusters | Ground Truth | Status |
|-----|--------|-----------|----|---------:|-------------:|--------|
| 1kx9B | 75.0% | 34.3% | 0.471 | 4 | 32 | ✅ |
| 1tvqA | 62.9% | 28.2% | 0.389 | 2 | 35 | ✅ |
| 2w9tA | 52.5% | 20.0% | 0.290 | 5 | 40 | ✅ |
| 3rwvA | 70.6% | 16.4% | 0.267 | 6 | 34 | ✅ |
| 5h9aA | 72.7% | 15.5% | 0.256 | 4 | 22 | ✅ |
| 2hq8B | 76.0% | 14.8% | 0.248 | 5 | 25 | ✅ |
| 2oy4A | 65.4% | 15.3% | 0.248 | 5 | 26 | ✅ |
| 6e5dA | 78.3% | 14.5% | 0.245 | 5 | 23 | ✅ |
| 2fjyA | 73.7% | 14.1% | 0.237 | 5 | 19 | ✅ |
| 3fvjA | 57.1% | 14.5% | 0.231 | 4 | 21 | ✅ |

## Failure Analysis (Recall < 30%)

| PDB | Recall | Issue | Hypothesis |
|-----|--------|-------|------------|
| 2laoA | 26.7% | Very low precision | Threshold too high for this structure |
| 2fd7A | 0.0% | Very low precision | Threshold too high for this structure |
| 4tqlA | 0.0% | Very low precision | Threshold too high for this structure |
| 2alpA | 0.0% | Very low precision | Threshold too high for this structure |
| 1ammA | 0.0% | Very low precision | Threshold too high for this structure |
| 4hjkA | 0.0% | Very low precision | Threshold too high for this structure |
| 1igdA | 0.0% | Very low precision | Threshold too high for this structure |
| 1hcl | 0.0% | Very low precision | Threshold too high for this structure |

## Score Distribution Analysis

- Mean cryptic score: 0.589
- Std cryptic score: 0.193
- Median: 0.659, Q25: 0.426, Q75: 0.724
- Adaptive threshold range: [0.492, 0.822]

## Timing Breakdown

| Phase | Mean (ms) | Std (ms) | % Total |
|-------|-----------|----------|--------|
| ANM Generation | 0.0 | 0.0 | 0.0% |
| Feature Extraction | 10.5 | 0.0 | 0.8% |
| Scoring | 6.3 | 0.0 | 0.5% |
| Clustering | 4.2 | 0.0 | 0.3% |
| HMC Refinement | 1002.7 | 0.0 | 78.8% |
| PRISM-ZrO Scoring | 257.8 | 0.0 | 20.3% |
| **Total** | **1272.1** | **0.0** | **100%** |

## Memory Usage

- Peak GPU memory: 0.0 MB
- Peak CPU memory: 0.0 MB

## Recommendations for Next Phase

- [ ] ROC AUC still low - consider enabling HMC refinement
- [ ] PR AUC below target - improve precision with stricter clustering

