# PRISM Cryptic Detection v2 - v2-phase3-low-thresh Report
Generated: 2026-01-11 00:44:36 UTC
Git Commit: 5d2c75b

## Executive Summary

| Metric | v1 Baseline | This Run | Delta | Target | Status |
|--------|-------------|----------|-------|--------|--------|
| ROC AUC | 0.455 | 0.481 | +0.026 | >0.90 | ❌ |
| PR AUC | 0.083 | 0.079 | -0.004 | >0.20 | ❌ |
| Success Rate | 84.8% | 84.8% | -0.017% | >90% | ❌ |
| Best F1 | 0.545 | 0.433 | -0.112 | >0.70 | ❌ |

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
| 1kx9B | 65.6% | 32.3% | 0.433 | 2 | 32 | ✅ |
| 2w9tA | 70.0% | 27.5% | 0.394 | 7 | 40 | ✅ |
| 1tvqA | 65.7% | 26.1% | 0.374 | 4 | 35 | ✅ |
| 2oy4A | 73.1% | 18.4% | 0.295 | 7 | 26 | ✅ |
| 3fvjA | 61.9% | 17.1% | 0.268 | 3 | 21 | ✅ |
| 3rwvA | 64.7% | 16.3% | 0.260 | 8 | 34 | ✅ |
| 5uxaA | 72.1% | 15.7% | 0.257 | 11 | 43 | ✅ |
| 5h9aA | 63.6% | 16.1% | 0.257 | 4 | 22 | ✅ |
| 3qxwB | 60.0% | 15.0% | 0.240 | 3 | 20 | ✅ |
| 2hq8B | 68.0% | 14.3% | 0.236 | 8 | 25 | ✅ |

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

- Mean cryptic score: 0.564
- Std cryptic score: 0.196
- Median: 0.637, Q25: 0.347, Q75: 0.709
- Adaptive threshold range: [0.504, 0.913]

## Timing Breakdown

| Phase | Mean (ms) | Std (ms) | % Total |
|-------|-----------|----------|--------|
| ANM Generation | 0.0 | 0.0 | 0.0% |
| Feature Extraction | 10.7 | 0.0 | 0.8% |
| Scoring | 6.4 | 0.0 | 0.5% |
| Clustering | 4.3 | 0.0 | 0.3% |
| HMC Refinement | 1061.4 | 0.0 | 79.7% |
| PRISM-ZrO Scoring | 258.2 | 0.0 | 19.4% |
| **Total** | **1331.7** | **0.0** | **100%** |

## Memory Usage

- Peak GPU memory: 0.0 MB
- Peak CPU memory: 0.0 MB

## Recommendations for Next Phase

- [ ] ROC AUC still low - consider enabling HMC refinement
- [ ] PR AUC below target - improve precision with stricter clustering

