# PRISM Cryptic Detection v2 - v2-tuned Report
Generated: 2026-01-10 23:23:13 UTC
Git Commit: 5d2c75b

## Executive Summary

| Metric | v1 Baseline | This Run | Delta | Target | Status |
|--------|-------------|----------|-------|--------|--------|
| ROC AUC | 0.455 | 0.482 | +0.027 | >0.90 | ❌ |
| PR AUC | 0.083 | 0.081 | -0.002 | >0.20 | ❌ |
| Success Rate | 84.8% | 71.7% | -13.061% | >90% | ❌ |
| Best F1 | 0.545 | 0.394 | -0.151 | >0.70 | ❌ |

## Configuration Used

- ANM: n_modes=30, n_conformations=100, amplitude_scale=5.0
- EFE: prior=0.070, epistemic_weight=0.4, pragmatic_weight=0.6
- Threshold: adaptive=true, z=0.2, floor=0.25
- Clustering: graph=true, distance=8.0Å, min_size=2
- HMC: enabled=true
- PRISM-ZrO: enabled=true

## Per-Structure Results (Top 10 by F1)

| PDB | Recall | Precision | F1 | Clusters | Ground Truth | Status |
|-----|--------|-----------|----|---------:|-------------:|--------|
| 1kx9B | 40.6% | 38.2% | 0.394 | 3 | 32 | ✅ |
| 1tvqA | 40.0% | 32.6% | 0.359 | 6 | 35 | ✅ |
| 2w9tA | 40.0% | 25.8% | 0.314 | 5 | 40 | ✅ |
| 6e5dA | 43.5% | 19.6% | 0.270 | 9 | 23 | ✅ |
| 3qxwB | 40.0% | 20.0% | 0.267 | 5 | 20 | ✅ |
| 3fvjA | 33.3% | 21.2% | 0.259 | 4 | 21 | ✅ |
| 4w51A | 52.9% | 15.8% | 0.243 | 4 | 17 | ✅ |
| 5uxaA | 53.5% | 15.5% | 0.241 | 11 | 43 | ✅ |
| 2hq8B | 36.0% | 17.3% | 0.234 | 4 | 25 | ✅ |
| 2oy4A | 30.8% | 14.5% | 0.198 | 5 | 26 | ✅ |

## Failure Analysis (Recall < 30%)

| PDB | Recall | Issue | Hypothesis |
|-----|--------|-------|------------|
| 4r72A | 29.2% | Very low precision | Large cryptic region may need more conformations |
| 4i92A | 26.3% | Very low precision | Threshold too high for this structure |
| 5h9aA | 22.7% | Low recall (22.7%) | Threshold too high for this structure |
| 1s2oA | 14.3% | Very low precision | Threshold too high for this structure |
| 1j8fC | 23.1% | Very low precision | Threshold too high for this structure |
| 3rwvA | 26.5% | Low recall (26.5%) | Threshold too high for this structure |
| 2fd7A | 0.0% | Very low precision | Threshold too high for this structure |
| 4tqlA | 0.0% | Very low precision | Threshold too high for this structure |
| 2alpA | 0.0% | Very low precision | May need HMC refinement for better sampling |
| 1ammA | 0.0% | Very low precision | Threshold too high for this structure |
| 4hjkA | 0.0% | Very low precision | Threshold too high for this structure |
| 1igdA | 0.0% | Very low precision | Threshold too high for this structure |
| 1hcl | 0.0% | Very low precision | Threshold too high for this structure |

## Score Distribution Analysis

- Mean cryptic score: 0.460
- Std cryptic score: 0.220
- Median: 0.358, Q25: 0.277, Q75: 0.680
- Adaptive threshold range: [0.473, 0.712]

## Timing Breakdown

| Phase | Mean (ms) | Std (ms) | % Total |
|-------|-----------|----------|--------|
| ANM Generation | 0.0 | 0.0 | 0.0% |
| Feature Extraction | 10.6 | 0.0 | 0.8% |
| Scoring | 6.4 | 0.0 | 0.5% |
| Clustering | 4.3 | 0.0 | 0.3% |
| HMC Refinement | 1072.2 | 0.0 | 79.9% |
| PRISM-ZrO Scoring | 257.9 | 0.0 | 19.2% |
| **Total** | **1342.0** | **0.0** | **100%** |

## Memory Usage

- Peak GPU memory: 0.0 MB
- Peak CPU memory: 0.0 MB

## Recommendations for Next Phase

- [ ] ROC AUC still low - consider enabling HMC refinement
- [ ] PR AUC below target - improve precision with stricter clustering
- [ ] 13 structures failed - analyze common patterns

