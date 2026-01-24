# PRISM Cryptic Detection v2 - v2-low-thresh Report
Generated: 2026-01-10 23:25:33 UTC
Git Commit: 5d2c75b

## Executive Summary

| Metric | v1 Baseline | This Run | Delta | Target | Status |
|--------|-------------|----------|-------|--------|--------|
| ROC AUC | 0.455 | 0.490 | +0.035 | >0.90 | ❌ |
| PR AUC | 0.083 | 0.076 | -0.007 | >0.20 | ❌ |
| Success Rate | 84.8% | 71.7% | -13.061% | >90% | ❌ |
| Best F1 | 0.545 | 0.294 | -0.251 | >0.70 | ❌ |

## Configuration Used

- ANM: n_modes=30, n_conformations=100, amplitude_scale=5.0
- EFE: prior=0.070, epistemic_weight=0.4, pragmatic_weight=0.6
- Threshold: adaptive=true, z=0.0, floor=0.25
- Clustering: graph=true, distance=8.0Å, min_size=2
- HMC: enabled=true
- PRISM-ZrO: enabled=true

## Per-Structure Results (Top 10 by F1)

| PDB | Recall | Precision | F1 | Clusters | Ground Truth | Status |
|-----|--------|-----------|----|---------:|-------------:|--------|
| 1kx9B | 31.2% | 27.8% | 0.294 | 4 | 32 | ✅ |
| 1tvqA | 31.4% | 26.2% | 0.286 | 6 | 35 | ✅ |
| 2w9tA | 32.5% | 22.8% | 0.268 | 6 | 40 | ✅ |
| 5h9aA | 40.9% | 16.7% | 0.237 | 5 | 22 | ✅ |
| 2oy4A | 34.6% | 15.8% | 0.217 | 5 | 26 | ✅ |
| 5uxaA | 39.5% | 13.8% | 0.205 | 11 | 43 | ✅ |
| 3ugkA | 50.0% | 11.3% | 0.184 | 10 | 42 | ✅ |
| 3fvjA | 28.6% | 13.3% | 0.182 | 6 | 21 | ❌ |
| 6e5dA | 30.4% | 12.3% | 0.175 | 7 | 23 | ✅ |
| 4p0iB | 52.4% | 10.3% | 0.172 | 11 | 21 | ✅ |

## Failure Analysis (Recall < 30%)

| PDB | Recall | Issue | Hypothesis |
|-----|--------|-------|------------|
| 2hq8B | 28.0% | Very low precision | Threshold too high for this structure |
| 2fjyA | 26.3% | Low recall (26.3%) | May need HMC refinement for better sampling |
| 1y1aA | 14.3% | Very low precision | May need HMC refinement for better sampling |
| 3fvjA | 28.6% | Low recall (28.6%) | Large cryptic region may need more conformations |
| 3rwvA | 23.5% | Low recall (23.5%) | Large cryptic region may need more conformations |
| 4w51A | 23.5% | Very low precision | Threshold too high for this structure |
| 2fd7A | 0.0% | Very low precision | Threshold too high for this structure |
| 4tqlA | 0.0% | Very low precision | Threshold too high for this structure |
| 2alpA | 0.0% | Very low precision | May need HMC refinement for better sampling |
| 1ammA | 0.0% | Very low precision | May need HMC refinement for better sampling |
| 4hjkA | 0.0% | Very low precision | Threshold too high for this structure |
| 1igdA | 0.0% | Very low precision | May need HMC refinement for better sampling |
| 1hcl | 0.0% | Very low precision | May need HMC refinement for better sampling |

## Score Distribution Analysis

- Mean cryptic score: 0.462
- Std cryptic score: 0.221
- Median: 0.366, Q25: 0.272, Q75: 0.687
- Adaptive threshold range: [0.411, 0.621]

## Timing Breakdown

| Phase | Mean (ms) | Std (ms) | % Total |
|-------|-----------|----------|--------|
| ANM Generation | 0.0 | 0.0 | 0.0% |
| Feature Extraction | 10.8 | 0.0 | 0.8% |
| Scoring | 6.5 | 0.0 | 0.5% |
| Clustering | 4.3 | 0.0 | 0.3% |
| HMC Refinement | 1071.2 | 0.0 | 79.9% |
| PRISM-ZrO Scoring | 257.7 | 0.0 | 19.2% |
| **Total** | **1340.9** | **0.0** | **100%** |

## Memory Usage

- Peak GPU memory: 0.0 MB
- Peak CPU memory: 0.0 MB

## Recommendations for Next Phase

- [ ] ROC AUC still low - consider enabling HMC refinement
- [ ] PR AUC below target - improve precision with stricter clustering
- [ ] 13 structures failed - analyze common patterns

