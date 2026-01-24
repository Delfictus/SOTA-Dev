# PRISM Cryptic Detection v2 - v2-best Report
Generated: 2026-01-10 23:24:24 UTC
Git Commit: 5d2c75b

## Executive Summary

| Metric | v1 Baseline | This Run | Delta | Target | Status |
|--------|-------------|----------|-------|--------|--------|
| ROC AUC | 0.455 | 0.468 | +0.013 | >0.90 | ❌ |
| PR AUC | 0.083 | 0.083 | -0.000 | >0.20 | ❌ |
| Success Rate | 84.8% | 58.7% | -26.104% | >90% | ❌ |
| Best F1 | 0.545 | 0.343 | -0.202 | >0.70 | ❌ |

## Configuration Used

- ANM: n_modes=30, n_conformations=100, amplitude_scale=5.0
- EFE: prior=0.070, epistemic_weight=0.4, pragmatic_weight=0.6
- Threshold: adaptive=true, z=0.1, floor=0.28
- Clustering: graph=true, distance=8.0Å, min_size=2
- HMC: enabled=true
- PRISM-ZrO: enabled=true

## Per-Structure Results (Top 10 by F1)

| PDB | Recall | Precision | F1 | Clusters | Ground Truth | Status |
|-----|--------|-----------|----|---------:|-------------:|--------|
| 1kx9B | 37.5% | 31.6% | 0.343 | 5 | 32 | ✅ |
| 5h9aA | 59.1% | 22.8% | 0.329 | 4 | 22 | ✅ |
| 1tvqA | 37.1% | 28.9% | 0.325 | 5 | 35 | ✅ |
| 2w9tA | 35.0% | 20.3% | 0.257 | 7 | 40 | ✅ |
| 3rwvA | 35.3% | 19.0% | 0.247 | 9 | 34 | ✅ |
| 2oy4A | 38.5% | 16.4% | 0.230 | 5 | 26 | ✅ |
| 3ugkA | 42.9% | 14.2% | 0.213 | 12 | 42 | ✅ |
| 6e5dA | 39.1% | 13.8% | 0.205 | 7 | 23 | ✅ |
| 2fjyA | 31.6% | 14.3% | 0.197 | 6 | 19 | ✅ |
| 5uxaA | 34.9% | 13.6% | 0.196 | 12 | 43 | ✅ |

## Failure Analysis (Recall < 30%)

| PDB | Recall | Issue | Hypothesis |
|-----|--------|-------|------------|
| 1urpA | 23.1% | Very low precision | May need HMC refinement for better sampling |
| 6ypkA | 16.7% | Very low precision | Large cryptic region may need more conformations |
| 5nzmA | 29.6% | Very low precision | Large cryptic region may need more conformations |
| 2hq8B | 24.0% | Very low precision | Threshold too high for this structure |
| 4r72A | 25.0% | Very low precision | Large cryptic region may need more conformations |
| 5za4A | 22.7% | Very low precision | Large cryptic region may need more conformations |
| 5niaA | 25.9% | Very low precision | Large cryptic region may need more conformations |
| 2zkuB | 29.0% | Very low precision | Large cryptic region may need more conformations |
| 3nx1A | 27.3% | Very low precision | May need HMC refinement for better sampling |
| 3qxwB | 25.0% | Low recall (25.0%) | Threshold too high for this structure |
| 1ezmA | 23.8% | Very low precision | Large cryptic region may need more conformations |
| 3fvjA | 28.6% | Low recall (28.6%) | Threshold too high for this structure |
| 2fd7A | 0.0% | Very low precision | Threshold too high for this structure |
| 4tqlA | 0.0% | Very low precision | Threshold too high for this structure |
| 2alpA | 0.0% | Very low precision | May need HMC refinement for better sampling |
| 1ammA | 0.0% | Very low precision | May need HMC refinement for better sampling |
| 4hjkA | 0.0% | Very low precision | Threshold too high for this structure |
| 1igdA | 0.0% | Very low precision | Threshold too high for this structure |
| 1hcl | 0.0% | Very low precision | May need HMC refinement for better sampling |

## Score Distribution Analysis

- Mean cryptic score: 0.436
- Std cryptic score: 0.202
- Median: 0.347, Q25: 0.277, Q75: 0.644
- Adaptive threshold range: [0.382, 0.734]

## Timing Breakdown

| Phase | Mean (ms) | Std (ms) | % Total |
|-------|-----------|----------|--------|
| ANM Generation | 0.0 | 0.0 | 0.0% |
| Feature Extraction | 10.7 | 0.0 | 0.8% |
| Scoring | 6.4 | 0.0 | 0.5% |
| Clustering | 4.3 | 0.0 | 0.3% |
| HMC Refinement | 1078.1 | 0.0 | 80.0% |
| PRISM-ZrO Scoring | 257.7 | 0.0 | 19.1% |
| **Total** | **1347.7** | **0.0** | **100%** |

## Memory Usage

- Peak GPU memory: 0.0 MB
- Peak CPU memory: 0.0 MB

## Recommendations for Next Phase

- [ ] ROC AUC still low - consider enabling HMC refinement
- [ ] PR AUC below target - improve precision with stricter clustering
- [ ] 19 structures failed - analyze common patterns

