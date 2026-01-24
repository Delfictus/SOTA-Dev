# PRISM Cryptic Detection v2 - v2-gpu-full Report
Generated: 2026-01-10 23:07:03 UTC
Git Commit: 5d2c75b

## Executive Summary

| Metric | v1 Baseline | This Run | Delta | Target | Status |
|--------|-------------|----------|-------|--------|--------|
| ROC AUC | 0.455 | 0.463 | +0.008 | >0.90 | ❌ |
| PR AUC | 0.083 | 0.083 | -0.000 | >0.20 | ❌ |
| Success Rate | 84.8% | 41.3% | -43.496% | >90% | ❌ |
| Best F1 | 0.545 | 0.407 | -0.138 | >0.70 | ❌ |

## Configuration Used

- ANM: n_modes=30, n_conformations=100, amplitude_scale=5.0
- EFE: prior=0.070, epistemic_weight=0.4, pragmatic_weight=0.6
- Threshold: adaptive=true, z=0.5, floor=0.20
- Clustering: graph=true, distance=8.0Å, min_size=2
- HMC: enabled=true
- PRISM-ZrO: enabled=true

## Per-Structure Results (Top 10 by F1)

| PDB | Recall | Precision | F1 | Clusters | Ground Truth | Status |
|-----|--------|-----------|----|---------:|-------------:|--------|
| 1kx9B | 37.5% | 44.4% | 0.407 | 3 | 32 | ✅ |
| 1tvqA | 40.0% | 35.9% | 0.378 | 4 | 35 | ✅ |
| 6e5dA | 47.8% | 26.2% | 0.338 | 7 | 23 | ✅ |
| 3ugkA | 61.9% | 22.4% | 0.329 | 10 | 42 | ✅ |
| 5h9aA | 40.9% | 24.3% | 0.305 | 6 | 22 | ✅ |
| 2hq8B | 36.0% | 19.6% | 0.254 | 4 | 25 | ✅ |
| 2w9tA | 25.0% | 20.4% | 0.225 | 5 | 40 | ❌ |
| 2oy4A | 30.8% | 15.7% | 0.208 | 5 | 26 | ✅ |
| 1s2oA | 71.4% | 11.4% | 0.196 | 4 | 14 | ✅ |
| 4p0iB | 42.9% | 11.7% | 0.184 | 8 | 21 | ✅ |

## Failure Analysis (Recall < 30%)

| PDB | Recall | Issue | Hypothesis |
|-----|--------|-------|------------|
| 6hb0A | 26.7% | Very low precision | May need HMC refinement for better sampling |
| 2laoA | 20.0% | Very low precision | Threshold too high for this structure |
| 1urpA | 23.1% | Very low precision | Threshold too high for this structure |
| 6ypkA | 12.5% | Very low precision | Threshold too high for this structure |
| 5g1mA | 26.3% | Very low precision | Threshold too high for this structure |
| 5nzmA | 29.6% | Very low precision | Large cryptic region may need more conformations |
| 2w9tA | 25.0% | Low recall (25.0%) | Threshold too high for this structure |
| 5niaA | 29.6% | Very low precision | Large cryptic region may need more conformations |
| 2fjyA | 21.1% | Low recall (21.1%) | Threshold too high for this structure |
| 4i92A | 21.1% | Very low precision | Threshold too high for this structure |
| 3qxwB | 15.0% | Very low precision | Threshold too high for this structure |
| 3ppnA | 25.0% | Very low precision | Threshold too high for this structure |
| 1ezmA | 23.8% | Very low precision | Threshold too high for this structure |
| 3kjeA | 23.8% | Very low precision | Threshold too high for this structure |
| 1y1aA | 14.3% | Very low precision | Threshold too high for this structure |
| 5uxaA | 23.3% | Low recall (23.3%) | Threshold too high for this structure |
| 6rvmC | 26.9% | Very low precision | Threshold too high for this structure |
| 3fvjA | 23.8% | Low recall (23.8%) | Threshold too high for this structure |
| 3rwvA | 17.6% | Low recall (17.6%) | Threshold too high for this structure |
| 4w51A | 23.5% | Very low precision | Threshold too high for this structure |
| 2fd7A | 0.0% | Very low precision | Threshold too high for this structure |
| 4tqlA | 0.0% | Very low precision | Threshold too high for this structure |
| 2alpA | 0.0% | Very low precision | May need HMC refinement for better sampling |
| 1ammA | 0.0% | Very low precision | Threshold too high for this structure |
| 4hjkA | 0.0% | Very low precision | Threshold too high for this structure |
| 1igdA | 0.0% | Very low precision | Threshold too high for this structure |
| 1hcl | 0.0% | Very low precision | Threshold too high for this structure |

## Score Distribution Analysis

- Mean cryptic score: 0.481
- Std cryptic score: 0.235
- Median: 0.381, Q25: 0.267, Q75: 0.709
- Adaptive threshold range: [0.458, 0.924]

## Timing Breakdown

| Phase | Mean (ms) | Std (ms) | % Total |
|-------|-----------|----------|--------|
| ANM Generation | 0.0 | 0.0 | 0.0% |
| Feature Extraction | 10.5 | 0.0 | 0.8% |
| Scoring | 6.3 | 0.0 | 0.5% |
| Clustering | 4.2 | 0.0 | 0.3% |
| HMC Refinement | 1065.9 | 0.0 | 79.8% |
| PRISM-ZrO Scoring | 257.5 | 0.0 | 19.3% |
| **Total** | **1335.1** | **0.0** | **100%** |

## Memory Usage

- Peak GPU memory: 0.0 MB
- Peak CPU memory: 0.0 MB

## Recommendations for Next Phase

- [ ] ROC AUC still low - consider enabling HMC refinement
- [ ] PR AUC below target - improve precision with stricter clustering
- [ ] 27 structures failed - analyze common patterns

