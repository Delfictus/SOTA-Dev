# PRISM Cryptic Detection v2 - v2-gpu-redetect Report
Generated: 2026-01-10 23:11:36 UTC
Git Commit: 5d2c75b

## Executive Summary

| Metric | v1 Baseline | This Run | Delta | Target | Status |
|--------|-------------|----------|-------|--------|--------|
| ROC AUC | 0.455 | 0.462 | +0.007 | >0.90 | ❌ |
| PR AUC | 0.083 | 0.082 | -0.001 | >0.20 | ❌ |
| Success Rate | 84.8% | 56.5% | -28.278% | >90% | ❌ |
| Best F1 | 0.545 | 0.421 | -0.124 | >0.70 | ❌ |

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
| 1kx9B | 50.0% | 36.4% | 0.421 | 3 | 32 | ✅ |
| 3rwvA | 52.9% | 24.3% | 0.333 | 6 | 34 | ✅ |
| 1tvqA | 37.1% | 26.5% | 0.310 | 5 | 35 | ✅ |
| 2w9tA | 37.5% | 25.9% | 0.306 | 7 | 40 | ✅ |
| 4w51A | 52.9% | 16.1% | 0.247 | 7 | 17 | ✅ |
| 5h9aA | 40.9% | 17.0% | 0.240 | 5 | 22 | ✅ |
| 3ugkA | 47.6% | 14.9% | 0.227 | 12 | 42 | ✅ |
| 2oy4A | 34.6% | 14.8% | 0.207 | 6 | 26 | ✅ |
| 3qxwB | 35.0% | 13.7% | 0.197 | 6 | 20 | ✅ |
| 1ezmA | 52.4% | 12.1% | 0.196 | 9 | 21 | ✅ |

## Failure Analysis (Recall < 30%)

| PDB | Recall | Issue | Hypothesis |
|-----|--------|-------|------------|
| 6hb0A | 26.7% | Very low precision | Threshold too high for this structure |
| 2laoA | 13.3% | Very low precision | Threshold too high for this structure |
| 1urpA | 15.4% | Very low precision | Threshold too high for this structure |
| 6ypkA | 16.7% | Very low precision | Threshold too high for this structure |
| 2hq8B | 28.0% | Low recall (28.0%) | Threshold too high for this structure |
| 1kmoA | 22.7% | Very low precision | Large cryptic region may need more conformations |
| 5niaA | 18.5% | Very low precision | Large cryptic region may need more conformations |
| 4i92A | 26.3% | Very low precision | Threshold too high for this structure |
| 3kjeA | 28.6% | Very low precision | Threshold too high for this structure |
| 1y1aA | 28.6% | Very low precision | Threshold too high for this structure |
| 4p0iB | 28.6% | Very low precision | Threshold too high for this structure |
| 5uxaA | 23.3% | Low recall (23.3%) | Large cryptic region may need more conformations |
| 3fvjA | 28.6% | Low recall (28.6%) | Threshold too high for this structure |
| 2fd7A | 0.0% | Very low precision | Threshold too high for this structure |
| 4tqlA | 0.0% | Very low precision | Threshold too high for this structure |
| 2alpA | 0.0% | Very low precision | May need HMC refinement for better sampling |
| 1ammA | 0.0% | Very low precision | Threshold too high for this structure |
| 4hjkA | 0.0% | Very low precision | Threshold too high for this structure |
| 1igdA | 0.0% | Very low precision | Threshold too high for this structure |
| 1hcl | 0.0% | Very low precision | Threshold too high for this structure |

## Score Distribution Analysis

- Mean cryptic score: 0.441
- Std cryptic score: 0.196
- Median: 0.358, Q25: 0.283, Q75: 0.636
- Adaptive threshold range: [0.445, 0.829]

## Timing Breakdown

| Phase | Mean (ms) | Std (ms) | % Total |
|-------|-----------|----------|--------|
| ANM Generation | 0.0 | 0.0 | 0.0% |
| Feature Extraction | 10.1 | 0.0 | 0.8% |
| Scoring | 6.0 | 0.0 | 0.5% |
| Clustering | 4.0 | 0.0 | 0.3% |
| HMC Refinement | 1039.3 | 0.0 | 79.4% |
| PRISM-ZrO Scoring | 257.7 | 0.0 | 19.7% |
| **Total** | **1308.2** | **0.0** | **100%** |

## Memory Usage

- Peak GPU memory: 0.0 MB
- Peak CPU memory: 0.0 MB

## Recommendations for Next Phase

- [ ] ROC AUC still low - consider enabling HMC refinement
- [ ] PR AUC below target - improve precision with stricter clustering
- [ ] 20 structures failed - analyze common patterns

