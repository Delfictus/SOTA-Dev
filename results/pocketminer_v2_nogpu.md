# PRISM Cryptic Detection v2 - v2-no-gpu Report
Generated: 2026-01-10 23:12:31 UTC
Git Commit: 5d2c75b

## Executive Summary

| Metric | v1 Baseline | This Run | Delta | Target | Status |
|--------|-------------|----------|-------|--------|--------|
| ROC AUC | 0.455 | 0.445 | -0.010 | >0.90 | ❌ |
| PR AUC | 0.083 | 0.086 | +0.003 | >0.20 | ❌ |
| Success Rate | 84.8% | 45.7% | -39.148% | >90% | ❌ |
| Best F1 | 0.545 | 0.382 | -0.163 | >0.70 | ❌ |

## Configuration Used

- ANM: n_modes=30, n_conformations=100, amplitude_scale=5.0
- EFE: prior=0.070, epistemic_weight=0.4, pragmatic_weight=0.6
- Threshold: adaptive=true, z=0.5, floor=0.20
- Clustering: graph=true, distance=8.0Å, min_size=2
- HMC: enabled=false
- PRISM-ZrO: enabled=false

## Per-Structure Results (Top 10 by F1)

| PDB | Recall | Precision | F1 | Clusters | Ground Truth | Status |
|-----|--------|-----------|----|---------:|-------------:|--------|
| 6e5dA | 56.5% | 28.9% | 0.382 | 6 | 23 | ✅ |
| 1kx9B | 34.4% | 39.3% | 0.367 | 3 | 32 | ✅ |
| 5h9aA | 50.0% | 28.2% | 0.361 | 6 | 22 | ✅ |
| 1tvqA | 34.3% | 30.0% | 0.320 | 4 | 35 | ✅ |
| 3ugkA | 54.8% | 21.7% | 0.311 | 9 | 42 | ✅ |
| 3fvjA | 33.3% | 17.9% | 0.233 | 3 | 21 | ✅ |
| 2oy4A | 30.8% | 17.4% | 0.222 | 5 | 26 | ✅ |
| 3rwvA | 29.4% | 16.4% | 0.211 | 5 | 34 | ❌ |
| 1s2oA | 71.4% | 12.2% | 0.208 | 5 | 14 | ✅ |
| 2hq8B | 32.0% | 15.4% | 0.208 | 5 | 25 | ✅ |

## Failure Analysis (Recall < 30%)

| PDB | Recall | Issue | Hypothesis |
|-----|--------|-------|------------|
| 6hb0A | 26.7% | Very low precision | May need HMC refinement for better sampling |
| 2laoA | 13.3% | Very low precision | Threshold too high for this structure |
| 1urpA | 23.1% | Very low precision | Threshold too high for this structure |
| 6ypkA | 16.7% | Very low precision | Threshold too high for this structure |
| 5g1mA | 21.1% | Very low precision | Threshold too high for this structure |
| 2w9tA | 22.5% | Low recall (22.5%) | Threshold too high for this structure |
| 5za4A | 22.7% | Very low precision | Large cryptic region may need more conformations |
| 2fjyA | 26.3% | Low recall (26.3%) | Threshold too high for this structure |
| 2zkuB | 16.1% | Very low precision | Threshold too high for this structure |
| 4i92A | 26.3% | Very low precision | Threshold too high for this structure |
| 3qxwB | 20.0% | Low recall (20.0%) | Threshold too high for this structure |
| 1ezmA | 28.6% | Very low precision | Threshold too high for this structure |
| 3kjeA | 23.8% | Very low precision | Threshold too high for this structure |
| 1y1aA | 14.3% | Very low precision | Threshold too high for this structure |
| 5uxaA | 23.3% | Low recall (23.3%) | Large cryptic region may need more conformations |
| 6rvmC | 23.1% | Very low precision | Large cryptic region may need more conformations |
| 3rwvA | 29.4% | Low recall (29.4%) | Threshold too high for this structure |
| 4w51A | 17.6% | Very low precision | Threshold too high for this structure |
| 2fd7A | 0.0% | Very low precision | Threshold too high for this structure |
| 4tqlA | 0.0% | Very low precision | Threshold too high for this structure |
| 2alpA | 0.0% | Very low precision | Threshold too high for this structure |
| 1ammA | 0.0% | Very low precision | Threshold too high for this structure |
| 4hjkA | 0.0% | Very low precision | Threshold too high for this structure |
| 1igdA | 0.0% | Very low precision | Threshold too high for this structure |
| 1hcl | 0.0% | Very low precision | Threshold too high for this structure |

## Score Distribution Analysis

- Mean cryptic score: 0.483
- Std cryptic score: 0.129
- Median: 0.479, Q25: 0.400, Q75: 0.557
- Adaptive threshold range: [0.417, 0.935]

## Timing Breakdown

| Phase | Mean (ms) | Std (ms) | % Total |
|-------|-----------|----------|--------|
| ANM Generation | 840.7 | 0.0 | 98.7% |
| Feature Extraction | 9.9 | 0.0 | 1.2% |
| Scoring | 5.9 | 0.0 | 0.7% |
| Clustering | 4.0 | 0.0 | 0.5% |
| **Total** | **851.5** | **0.0** | **100%** |

## Memory Usage

- Peak GPU memory: 0.0 MB
- Peak CPU memory: 0.0 MB

## Recommendations for Next Phase

- [ ] ROC AUC still low - consider enabling HMC refinement
- [ ] PR AUC below target - improve precision with stricter clustering
- [ ] 25 structures failed - analyze common patterns

