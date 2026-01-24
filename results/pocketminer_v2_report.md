# PRISM Cryptic Detection v2 - v2 Report
Generated: 2026-01-10 22:43:35 UTC
Git Commit: 5d2c75b

## Executive Summary

| Metric | v1 Baseline | This Run | Delta | Target | Status |
|--------|-------------|----------|-------|--------|--------|
| ROC AUC | 0.455 | 0.445 | -0.010 | >0.90 | ❌ |
| PR AUC | 0.083 | 0.085 | +0.002 | >0.20 | ❌ |
| Success Rate | 84.8% | 0.0% | -84.800% | >90% | ❌ |
| Best F1 | 0.545 | 0.233 | -0.312 | >0.70 | ❌ |

## Configuration Used

- ANM: n_modes=30, n_conformations=100, amplitude_scale=5.0
- EFE: prior=0.070, epistemic_weight=0.4, pragmatic_weight=0.6
- Threshold: adaptive=true, z=1.5, floor=0.25
- Clustering: graph=true, distance=8.0Å, min_size=2
- HMC: enabled=false
- PRISM-ZrO: enabled=false

## Per-Structure Results (Top 10 by F1)

| PDB | Recall | Precision | F1 | Clusters | Ground Truth | Status |
|-----|--------|-----------|----|---------:|-------------:|--------|
| 1tvqA | 14.3% | 62.5% | 0.233 | 2 | 35 | ❌ |
| 3ugkA | 14.3% | 37.5% | 0.207 | 4 | 42 | ❌ |
| 6e5dA | 13.0% | 27.3% | 0.176 | 3 | 23 | ❌ |
| 2ceyA | 11.1% | 22.2% | 0.148 | 2 | 18 | ❌ |
| 1kx9B | 9.4% | 33.3% | 0.146 | 3 | 32 | ❌ |
| 4i92A | 15.8% | 13.0% | 0.143 | 8 | 19 | ❌ |
| 4r72A | 8.3% | 16.7% | 0.111 | 5 | 24 | ❌ |
| 1j8fC | 7.7% | 11.8% | 0.093 | 7 | 26 | ❌ |
| 1s2oA | 7.1% | 12.5% | 0.091 | 1 | 14 | ❌ |
| 4v38A | 8.7% | 8.7% | 0.087 | 4 | 23 | ❌ |

## Failure Analysis (Recall < 30%)

| PDB | Recall | Issue | Hypothesis |
|-----|--------|-------|------------|
| 6hb0A | 6.7% | Low recall (6.7%) | Threshold too high for this structure |
| 2laoA | 0.0% | Very low precision | Threshold too high for this structure |
| 1urpA | 0.0% | Very low precision | Threshold too high for this structure |
| 6ypkA | 4.2% | Very low precision | Threshold too high for this structure |
| 3ugkA | 14.3% | Low recall (14.3%) | Threshold too high for this structure |
| 5g1mA | 0.0% | Very low precision | Threshold too high for this structure |
| 5nzmA | 0.0% | Very low precision | Threshold too high for this structure |
| 4v38A | 8.7% | Very low precision | Threshold too high for this structure |
| 2w9tA | 0.0% | Very low precision | Threshold too high for this structure |
| 2hq8B | 0.0% | Very low precision | Threshold too high for this structure |
| 4r72A | 8.3% | Low recall (8.3%) | Threshold too high for this structure |
| 5za4A | 9.1% | Very low precision | Threshold too high for this structure |
| 2ceyA | 11.1% | Low recall (11.1%) | Threshold too high for this structure |
| 1kmoA | 4.5% | Very low precision | Threshold too high for this structure |
| 5niaA | 0.0% | Very low precision | Threshold too high for this structure |
| 2fjyA | 0.0% | Very low precision | Threshold too high for this structure |
| 3p53A | 6.2% | Very low precision | Threshold too high for this structure |
| 1kx9B | 9.4% | Low recall (9.4%) | Threshold too high for this structure |
| 1tvqA | 14.3% | Low recall (14.3%) | Threshold too high for this structure |
| 2zkuB | 0.0% | Very low precision | Threshold too high for this structure |
| 3nx1A | 0.0% | Very low precision | Threshold too high for this structure |
| 4i92A | 15.8% | Low recall (15.8%) | Threshold too high for this structure |
| 3qxwB | 0.0% | Very low precision | Threshold too high for this structure |
| 5h9aA | 0.0% | Very low precision | Threshold too high for this structure |
| 1s2oA | 7.1% | Low recall (7.1%) | Threshold too high for this structure |
| 3ppnA | 0.0% | Very low precision | Threshold too high for this structure |
| 1ezmA | 0.0% | Very low precision | Threshold too high for this structure |
| 1j8fC | 7.7% | Low recall (7.7%) | Threshold too high for this structure |
| 3kjeA | 0.0% | Very low precision | Threshold too high for this structure |
| 1y1aA | 0.0% | Very low precision | Threshold too high for this structure |
| 4p0iB | 0.0% | Very low precision | Threshold too high for this structure |
| 5uxaA | 0.0% | Very low precision | Threshold too high for this structure |
| 6rvmC | 3.8% | Low recall (3.8%) | Threshold too high for this structure |
| 3fvjA | 0.0% | Very low precision | Threshold too high for this structure |
| 2oy4A | 0.0% | Very low precision | Threshold too high for this structure |
| 3rwvA | 0.0% | Very low precision | Threshold too high for this structure |
| 6e5dA | 13.0% | Low recall (13.0%) | Threshold too high for this structure |
| 4ic4A | 0.0% | Very low precision | Threshold too high for this structure |
| 4w51A | 0.0% | Very low precision | Threshold too high for this structure |
| 2fd7A | 0.0% | No predictions made | Threshold too high for this structure |
| 4tqlA | 0.0% | Very low precision | Threshold too high for this structure |
| 2alpA | 0.0% | Very low precision | Threshold too high for this structure |
| 1ammA | 0.0% | Very low precision | Threshold too high for this structure |
| 4hjkA | 0.0% | Very low precision | Threshold too high for this structure |
| 1igdA | 0.0% | Very low precision | Threshold too high for this structure |
| 1hcl | 0.0% | Very low precision | Threshold too high for this structure |

## Score Distribution Analysis

- Mean cryptic score: 0.482
- Std cryptic score: 0.130
- Median: 0.479, Q25: 0.399, Q75: 0.556
- Adaptive threshold range: [0.532, 1.072]

## Timing Breakdown

| Phase | Mean (ms) | Std (ms) | % Total |
|-------|-----------|----------|--------|
| ANM Generation | 838.1 | 0.0 | 98.8% |
| Feature Extraction | 9.8 | 0.0 | 1.2% |
| Scoring | 5.9 | 0.0 | 0.7% |
| Clustering | 3.9 | 0.0 | 0.5% |
| **Total** | **848.7** | **0.0** | **100%** |

## Memory Usage

- Peak GPU memory: 0.0 MB
- Peak CPU memory: 0.0 MB

## Recommendations for Next Phase

- [ ] ROC AUC still low - consider enabling HMC refinement
- [ ] PR AUC below target - improve precision with stricter clustering
- [ ] 46 structures failed - analyze common patterns

