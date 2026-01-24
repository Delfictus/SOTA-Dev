# PRISM Cryptic Detection v2 - v2-phase3-enhanced Report
Generated: 2026-01-11 00:42:46 UTC
Git Commit: 5d2c75b

## Executive Summary

| Metric | v1 Baseline | This Run | Delta | Target | Status |
|--------|-------------|----------|-------|--------|--------|
| ROC AUC | 0.455 | 0.479 | +0.024 | >0.90 | ❌ |
| PR AUC | 0.083 | 0.077 | -0.006 | >0.20 | ❌ |
| Success Rate | 84.8% | 45.7% | -39.148% | >90% | ❌ |
| Best F1 | 0.545 | 0.342 | -0.203 | >0.70 | ❌ |

## Configuration Used

- ANM: n_modes=30, n_conformations=100, amplitude_scale=5.0
- EFE: prior=0.070, epistemic_weight=0.4, pragmatic_weight=0.6
- Threshold: adaptive=true, z=0.0, floor=0.30
- Clustering: graph=true, distance=8.0Å, min_size=2
- HMC: enabled=true
- PRISM-ZrO: enabled=true

## Per-Structure Results (Top 10 by F1)

| PDB | Recall | Precision | F1 | Clusters | Ground Truth | Status |
|-----|--------|-----------|----|---------:|-------------:|--------|
| 1tvqA | 37.1% | 31.7% | 0.342 | 6 | 35 | ✅ |
| 5h9aA | 50.0% | 23.9% | 0.324 | 7 | 22 | ✅ |
| 1kx9B | 31.2% | 30.3% | 0.308 | 5 | 32 | ✅ |
| 6e5dA | 47.8% | 20.4% | 0.286 | 9 | 23 | ✅ |
| 3qxwB | 35.0% | 20.6% | 0.259 | 5 | 20 | ✅ |
| 5uxaA | 34.9% | 16.3% | 0.222 | 7 | 43 | ✅ |
| 3ugkA | 35.7% | 15.8% | 0.219 | 13 | 42 | ✅ |
| 3fvjA | 33.3% | 14.6% | 0.203 | 4 | 21 | ✅ |
| 2fjyA | 31.6% | 14.0% | 0.194 | 6 | 19 | ✅ |
| 2oy4A | 26.9% | 14.6% | 0.189 | 6 | 26 | ❌ |

## Failure Analysis (Recall < 30%)

| PDB | Recall | Issue | Hypothesis |
|-----|--------|-------|------------|
| 1urpA | 23.1% | Very low precision | May need HMC refinement for better sampling |
| 6ypkA | 20.8% | Very low precision | Large cryptic region may need more conformations |
| 2w9tA | 20.0% | Low recall (20.0%) | Large cryptic region may need more conformations |
| 2hq8B | 28.0% | Low recall (28.0%) | Large cryptic region may need more conformations |
| 4r72A | 20.8% | Very low precision | Large cryptic region may need more conformations |
| 5za4A | 27.3% | Very low precision | Large cryptic region may need more conformations |
| 5niaA | 22.2% | Very low precision | Large cryptic region may need more conformations |
| 3p53A | 25.0% | Very low precision | May need HMC refinement for better sampling |
| 4i92A | 15.8% | Very low precision | May need HMC refinement for better sampling |
| 1ezmA | 19.0% | Very low precision | Large cryptic region may need more conformations |
| 3kjeA | 23.8% | Very low precision | Large cryptic region may need more conformations |
| 1y1aA | 14.3% | Very low precision | May need HMC refinement for better sampling |
| 4p0iB | 19.0% | Very low precision | Large cryptic region may need more conformations |
| 6rvmC | 26.9% | Very low precision | Large cryptic region may need more conformations |
| 2oy4A | 26.9% | Low recall (26.9%) | Large cryptic region may need more conformations |
| 3rwvA | 20.6% | Low recall (20.6%) | Large cryptic region may need more conformations |
| 4ic4A | 23.3% | Very low precision | Large cryptic region may need more conformations |
| 4w51A | 5.9% | Very low precision | May need HMC refinement for better sampling |
| 2fd7A | 0.0% | Very low precision | Threshold too high for this structure |
| 4tqlA | 0.0% | Very low precision | Threshold too high for this structure |
| 2alpA | 0.0% | Very low precision | May need HMC refinement for better sampling |
| 1ammA | 0.0% | Very low precision | May need HMC refinement for better sampling |
| 4hjkA | 0.0% | Very low precision | Threshold too high for this structure |
| 1igdA | 0.0% | Very low precision | May need HMC refinement for better sampling |
| 1hcl | 0.0% | Very low precision | May need HMC refinement for better sampling |

## Score Distribution Analysis

- Mean cryptic score: 0.408
- Std cryptic score: 0.201
- Median: 0.324, Q25: 0.261, Q75: 0.611
- Adaptive threshold range: [0.332, 0.663]

## Timing Breakdown

| Phase | Mean (ms) | Std (ms) | % Total |
|-------|-----------|----------|--------|
| ANM Generation | 0.0 | 0.0 | 0.0% |
| Feature Extraction | 23.4 | 0.0 | 0.8% |
| Scoring | 14.0 | 0.0 | 0.5% |
| Clustering | 9.4 | 0.0 | 0.3% |
| HMC Refinement | 2792.6 | 0.0 | 90.8% |
| PRISM-ZrO Scoring | 258.6 | 0.0 | 8.4% |
| **Total** | **3075.8** | **0.0** | **100%** |

## Memory Usage

- Peak GPU memory: 0.0 MB
- Peak CPU memory: 0.0 MB

## Recommendations for Next Phase

- [ ] ROC AUC still low - consider enabling HMC refinement
- [ ] PR AUC below target - improve precision with stricter clustering
- [ ] 25 structures failed - analyze common patterns

