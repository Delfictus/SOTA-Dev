# PRISM Cryptic Detection v2 - v2-fixed-residue-ids Report
Generated: 2026-01-10 23:16:26 UTC
Git Commit: 5d2c75b

## Executive Summary

| Metric | v1 Baseline | This Run | Delta | Target | Status |
|--------|-------------|----------|-------|--------|--------|
| ROC AUC | 0.455 | 0.535 | +0.080 | >0.90 | ❌ |
| PR AUC | 0.083 | 0.063 | -0.020 | >0.20 | ❌ |
| Success Rate | 84.8% | 41.3% | -43.496% | >90% | ❌ |
| Best F1 | 0.545 | 0.272 | -0.273 | >0.70 | ❌ |

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
| 6e5dA | 47.8% | 19.0% | 0.272 | 4 | 23 | ✅ |
| 3rwvA | 38.2% | 21.0% | 0.271 | 9 | 34 | ✅ |
| 1tvqA | 28.6% | 23.8% | 0.260 | 4 | 35 | ❌ |
| 2w9tA | 27.5% | 22.0% | 0.244 | 8 | 40 | ❌ |
| 3qxwB | 35.0% | 17.5% | 0.233 | 4 | 20 | ✅ |
| 1kx9B | 21.9% | 23.3% | 0.226 | 3 | 32 | ❌ |
| 2hq8B | 32.0% | 15.1% | 0.205 | 6 | 25 | ✅ |
| 5h9aA | 31.8% | 14.9% | 0.203 | 6 | 22 | ✅ |
| 3ugkA | 38.1% | 13.4% | 0.199 | 14 | 42 | ✅ |
| 3kjeA | 47.6% | 10.6% | 0.174 | 6 | 21 | ✅ |

## Failure Analysis (Recall < 30%)

| PDB | Recall | Issue | Hypothesis |
|-----|--------|-------|------------|
| 6hb0A | 6.7% | Very low precision | Threshold too high for this structure |
| 1urpA | 23.1% | Very low precision | Threshold too high for this structure |
| 5nzmA | 25.9% | Very low precision | Large cryptic region may need more conformations |
| 2w9tA | 27.5% | Low recall (27.5%) | Threshold too high for this structure |
| 1kmoA | 18.2% | Very low precision | Large cryptic region may need more conformations |
| 2fjyA | 26.3% | Low recall (26.3%) | Threshold too high for this structure |
| 3p53A | 25.0% | Very low precision | May need HMC refinement for better sampling |
| 1kx9B | 21.9% | Low recall (21.9%) | Threshold too high for this structure |
| 1tvqA | 28.6% | Low recall (28.6%) | Threshold too high for this structure |
| 3nx1A | 18.2% | Very low precision | Threshold too high for this structure |
| 1s2oA | 28.6% | Very low precision | Threshold too high for this structure |
| 3ppnA | 25.0% | Very low precision | Threshold too high for this structure |
| 1ezmA | 23.8% | Very low precision | Threshold too high for this structure |
| 1j8fC | 0.0% | Very low precision | Threshold too high for this structure |
| 1y1aA | 28.6% | Very low precision | Threshold too high for this structure |
| 5uxaA | 18.6% | Very low precision | Large cryptic region may need more conformations |
| 6rvmC | 19.2% | Very low precision | Large cryptic region may need more conformations |
| 3fvjA | 23.8% | Low recall (23.8%) | Threshold too high for this structure |
| 2oy4A | 26.9% | Low recall (26.9%) | Threshold too high for this structure |
| 4ic4A | 0.0% | Very low precision | Large cryptic region may need more conformations |
| 2fd7A | 0.0% | Very low precision | Threshold too high for this structure |
| 4tqlA | 0.0% | Very low precision | Threshold too high for this structure |
| 2alpA | 0.0% | Very low precision | May need HMC refinement for better sampling |
| 1ammA | 0.0% | Very low precision | Threshold too high for this structure |
| 4hjkA | 0.0% | Very low precision | Threshold too high for this structure |
| 1igdA | 0.0% | Very low precision | Threshold too high for this structure |
| 1hcl | 0.0% | Very low precision | Threshold too high for this structure |

## Score Distribution Analysis

- Mean cryptic score: 0.429
- Std cryptic score: 0.200
- Median: 0.339, Q25: 0.275, Q75: 0.622
- Adaptive threshold range: [0.447, 0.826]

## Timing Breakdown

| Phase | Mean (ms) | Std (ms) | % Total |
|-------|-----------|----------|--------|
| ANM Generation | 0.0 | 0.0 | 0.0% |
| Feature Extraction | 10.5 | 0.0 | 0.8% |
| Scoring | 6.3 | 0.0 | 0.5% |
| Clustering | 4.2 | 0.0 | 0.3% |
| HMC Refinement | 1061.1 | 0.0 | 79.8% |
| PRISM-ZrO Scoring | 256.7 | 0.0 | 19.3% |
| **Total** | **1329.3** | **0.0** | **100%** |

## Memory Usage

- Peak GPU memory: 0.0 MB
- Peak CPU memory: 0.0 MB

## Recommendations for Next Phase

- [ ] ROC AUC still low - consider enabling HMC refinement
- [ ] PR AUC below target - improve precision with stricter clustering
- [ ] 27 structures failed - analyze common patterns

