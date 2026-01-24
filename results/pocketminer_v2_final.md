# PRISM Cryptic Detection v2 - v2-final-gpu Report
Generated: 2026-01-10 23:19:28 UTC
Git Commit: 5d2c75b

## Executive Summary

| Metric | v1 Baseline | This Run | Delta | Target | Status |
|--------|-------------|----------|-------|--------|--------|
| ROC AUC | 0.455 | 0.505 | +0.050 | >0.90 | ❌ |
| PR AUC | 0.083 | 0.073 | -0.010 | >0.20 | ❌ |
| Success Rate | 84.8% | 47.8% | -36.974% | >90% | ❌ |
| Best F1 | 0.545 | 0.297 | -0.248 | >0.70 | ❌ |

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
| 1tvqA | 31.4% | 28.2% | 0.297 | 6 | 35 | ✅ |
| 3rwvA | 35.3% | 23.1% | 0.279 | 9 | 34 | ✅ |
| 1kx9B | 21.9% | 35.0% | 0.269 | 4 | 32 | ❌ |
| 2w9tA | 27.5% | 25.0% | 0.262 | 6 | 40 | ❌ |
| 3qxwB | 35.0% | 20.0% | 0.255 | 3 | 20 | ✅ |
| 3fvjA | 28.6% | 22.2% | 0.250 | 3 | 21 | ❌ |
| 5uxaA | 46.5% | 15.2% | 0.229 | 11 | 43 | ✅ |
| 2oy4A | 26.9% | 15.6% | 0.197 | 8 | 26 | ❌ |
| 6rvmC | 50.0% | 9.6% | 0.161 | 7 | 26 | ✅ |
| 3ugkA | 33.3% | 10.2% | 0.156 | 13 | 42 | ✅ |

## Failure Analysis (Recall < 30%)

| PDB | Recall | Issue | Hypothesis |
|-----|--------|-------|------------|
| 2laoA | 20.0% | Very low precision | Threshold too high for this structure |
| 5g1mA | 26.3% | Very low precision | Threshold too high for this structure |
| 2w9tA | 27.5% | Low recall (27.5%) | Threshold too high for this structure |
| 2hq8B | 20.0% | Low recall (20.0%) | Threshold too high for this structure |
| 2fjyA | 21.1% | Low recall (21.1%) | Threshold too high for this structure |
| 1kx9B | 21.9% | Low recall (21.9%) | Threshold too high for this structure |
| 3nx1A | 18.2% | Very low precision | Threshold too high for this structure |
| 5h9aA | 22.7% | Low recall (22.7%) | Threshold too high for this structure |
| 1s2oA | 21.4% | Very low precision | Threshold too high for this structure |
| 3ppnA | 8.3% | Very low precision | Threshold too high for this structure |
| 1ezmA | 28.6% | Very low precision | Threshold too high for this structure |
| 1j8fC | 26.9% | Very low precision | Threshold too high for this structure |
| 1y1aA | 14.3% | Very low precision | Threshold too high for this structure |
| 3fvjA | 28.6% | Low recall (28.6%) | Threshold too high for this structure |
| 2oy4A | 26.9% | Low recall (26.9%) | Threshold too high for this structure |
| 6e5dA | 17.4% | Very low precision | Threshold too high for this structure |
| 4w51A | 17.6% | Very low precision | Threshold too high for this structure |
| 2fd7A | 0.0% | Very low precision | Threshold too high for this structure |
| 4tqlA | 0.0% | Very low precision | Threshold too high for this structure |
| 2alpA | 0.0% | Very low precision | Threshold too high for this structure |
| 1ammA | 0.0% | Very low precision | Threshold too high for this structure |
| 4hjkA | 0.0% | Very low precision | Threshold too high for this structure |
| 1igdA | 0.0% | Very low precision | Threshold too high for this structure |
| 1hcl | 0.0% | Very low precision | Threshold too high for this structure |

## Score Distribution Analysis

- Mean cryptic score: 0.460
- Std cryptic score: 0.173
- Median: 0.399, Q25: 0.311, Q75: 0.636
- Adaptive threshold range: [0.509, 0.637]

## Timing Breakdown

| Phase | Mean (ms) | Std (ms) | % Total |
|-------|-----------|----------|--------|
| ANM Generation | 0.0 | 0.0 | 0.0% |
| Feature Extraction | 10.2 | 0.0 | 0.8% |
| Scoring | 6.1 | 0.0 | 0.5% |
| Clustering | 4.1 | 0.0 | 0.3% |
| HMC Refinement | 1058.0 | 0.0 | 79.7% |
| PRISM-ZrO Scoring | 257.7 | 0.0 | 19.4% |
| **Total** | **1327.1** | **0.0** | **100%** |

## Memory Usage

- Peak GPU memory: 0.0 MB
- Peak CPU memory: 0.0 MB

## Recommendations for Next Phase

- [ ] ROC AUC still low - consider enabling HMC refinement
- [ ] PR AUC below target - improve precision with stricter clustering
- [ ] 24 structures failed - analyze common patterns

