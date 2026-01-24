# PRISM Cryptic Detection v2 - v2 Report
Generated: 2026-01-10 22:48:39 UTC
Git Commit: 5d2c75b

## Executive Summary

| Metric | v1 Baseline | This Run | Delta | Target | Status |
|--------|-------------|----------|-------|--------|--------|
| ROC AUC | 0.455 | 0.444 | -0.011 | >0.90 | ❌ |
| PR AUC | 0.083 | 0.086 | +0.003 | >0.20 | ❌ |
| Success Rate | 84.8% | 34.8% | -50.017% | >90% | ❌ |
| Best F1 | 0.545 | 0.406 | -0.139 | >0.70 | ❌ |

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
| 6e5dA | 60.9% | 30.4% | 0.406 | 6 | 23 | ✅ |
| 1kx9B | 34.4% | 45.8% | 0.393 | 3 | 32 | ✅ |
| 1tvqA | 37.1% | 35.1% | 0.361 | 4 | 35 | ✅ |
| 5h9aA | 50.0% | 26.2% | 0.344 | 6 | 22 | ✅ |
| 3ugkA | 52.4% | 20.0% | 0.289 | 10 | 42 | ✅ |
| 3fvjA | 28.6% | 17.1% | 0.214 | 4 | 21 | ❌ |
| 2hq8B | 32.0% | 15.1% | 0.205 | 5 | 25 | ✅ |
| 2fjyA | 26.3% | 12.8% | 0.172 | 5 | 19 | ❌ |
| 2w9tA | 17.5% | 16.3% | 0.169 | 5 | 40 | ❌ |
| 1s2oA | 57.1% | 9.8% | 0.167 | 6 | 14 | ✅ |

## Failure Analysis (Recall < 30%)

| PDB | Recall | Issue | Hypothesis |
|-----|--------|-------|------------|
| 6hb0A | 26.7% | Very low precision | Threshold too high for this structure |
| 2laoA | 20.0% | Very low precision | Threshold too high for this structure |
| 6ypkA | 20.8% | Very low precision | Threshold too high for this structure |
| 5g1mA | 26.3% | Very low precision | Threshold too high for this structure |
| 5nzmA | 29.6% | Very low precision | Large cryptic region may need more conformations |
| 4v38A | 26.1% | Very low precision | Large cryptic region may need more conformations |
| 2w9tA | 17.5% | Low recall (17.5%) | Threshold too high for this structure |
| 5za4A | 27.3% | Very low precision | Large cryptic region may need more conformations |
| 2ceyA | 27.8% | Very low precision | Threshold too high for this structure |
| 2fjyA | 26.3% | Low recall (26.3%) | Threshold too high for this structure |
| 2zkuB | 16.1% | Very low precision | Threshold too high for this structure |
| 3nx1A | 27.3% | Very low precision | Threshold too high for this structure |
| 4i92A | 26.3% | Very low precision | Threshold too high for this structure |
| 3qxwB | 15.0% | Very low precision | Threshold too high for this structure |
| 3ppnA | 25.0% | Very low precision | Threshold too high for this structure |
| 3kjeA | 23.8% | Very low precision | Threshold too high for this structure |
| 1y1aA | 14.3% | Very low precision | Threshold too high for this structure |
| 5uxaA | 20.9% | Low recall (20.9%) | Large cryptic region may need more conformations |
| 6rvmC | 26.9% | Very low precision | Large cryptic region may need more conformations |
| 3fvjA | 28.6% | Low recall (28.6%) | Threshold too high for this structure |
| 2oy4A | 19.2% | Low recall (19.2%) | Threshold too high for this structure |
| 3rwvA | 20.6% | Low recall (20.6%) | Threshold too high for this structure |
| 4w51A | 29.4% | Very low precision | Threshold too high for this structure |
| 2fd7A | 0.0% | Very low precision | Threshold too high for this structure |
| 4tqlA | 0.0% | Very low precision | Threshold too high for this structure |
| 2alpA | 0.0% | Very low precision | May need HMC refinement for better sampling |
| 1ammA | 0.0% | Very low precision | Threshold too high for this structure |
| 4hjkA | 0.0% | Very low precision | Threshold too high for this structure |
| 1igdA | 0.0% | Very low precision | Threshold too high for this structure |
| 1hcl | 0.0% | Very low precision | Threshold too high for this structure |

## Score Distribution Analysis

- Mean cryptic score: 0.483
- Std cryptic score: 0.129
- Median: 0.480, Q25: 0.400, Q75: 0.559
- Adaptive threshold range: [0.420, 0.940]

## Timing Breakdown

| Phase | Mean (ms) | Std (ms) | % Total |
|-------|-----------|----------|--------|
| ANM Generation | 802.8 | 0.0 | 98.8% |
| Feature Extraction | 9.4 | 0.0 | 1.2% |
| Scoring | 5.6 | 0.0 | 0.7% |
| Clustering | 3.8 | 0.0 | 0.5% |
| **Total** | **813.0** | **0.0** | **100%** |

## Memory Usage

- Peak GPU memory: 0.0 MB
- Peak CPU memory: 0.0 MB

## Recommendations for Next Phase

- [ ] ROC AUC still low - consider enabling HMC refinement
- [ ] PR AUC below target - improve precision with stricter clustering
- [ ] 30 structures failed - analyze common patterns

