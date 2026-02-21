# PRISM-4D v8 DCC Validation Report
## Intensity²-Weighted Centroids, Dynamic LIGSITE

**Date:** 2026-02-21
**Build:** sota-dev/main (weighted centroid patch, commit 2983a8ff)
**Command:** `nhs_rt_full --fast --hysteresis --multi-stream 8 --spike-percentile 95 -v`
**Metric:** DCC (Distance Centroid-to-Centroid) — standard pocket detection benchmark

### Ground Truth Verification

All apo-holo pairs verified by:
1. PDB header/TITLE inspection (protein identity)
2. Sequence identity check (>95% for same protein)
3. Coordinate frame alignment (raw CA deviation; Kabsch superposition where needed)

**Critical finding:** Previous eval_spatial_v2.py contained fatal errors:
- 1BTM mapped to 1BTL → actually Triosephosphate Isomerase (6% seq identity), not β-Lactamase
- 1BJV mapped to 1BJ4 → actually Thrombin, not SHMT
- 4OQ3 mapped to 4OBE → actually MDM2, not KRAS

### Results

| Apo   | Protein           | Holo/Lig   | Alignment    | Site | DCC    | Verdict      |
|-------|-------------------|------------|--------------|:----:|:------:|:------------:|
| 1BTL  | TEM-1 β-Lactamase | 1TEM/ALP   | Compatible   |   8  | 2.1 Å | ✓ EXCELLENT  |
| 1W50  | BACE              | 1W51/L01   | Compatible   |   3  | 3.6 Å | ✓ EXCELLENT  |
| 1G1F  | PTP1B (cryptic)   | 1T49/892   | Kabsch 0.93Å |   8  | 4.8 Å | ✓ EXCELLENT  |
| 1ADE  | AdSS              | 1GIM/GDP   | Kabsch 1.98Å |  12  | 5.9 Å | ✓ GOOD       |
| 3K5V  | Abl Kinase        | 3K5V/STI   | Self-ref     |   3  | 6.2 Å | ✓ GOOD       |
| 1A4Q  | Neuraminidase     | 1A4Q/DPC   | Self-ref     |   5  | 6.3 Å | ✓ GOOD       |
| 1BJ4  | SHMT              | 1BJ4/PLP   | Self-ref     |   0  | 9.8 Å | ~ MARGINAL   |
| 1HHP  | HIV-1 Protease    | 1HVR/XK2   | Kabsch 1.03Å |   1  | 9.8 Å | ~ MARGINAL   |
| 1ERE  | ER-α              | 1ERR/RAL   | Kabsch 3.34Å |   9  | 20.3Å | ✗ MISS       |

**4OBE (KRAS):** Skipped — monomer too small for current pipeline.

### Accuracy Summary (n=9)

| Threshold | Count | Accuracy |
|:---------:|:-----:|:--------:|
| < 5 Å    | 3/9   | 33%      |
| < 8 Å    | 6/9   | 67%      |
| < 10 Å   | 8/9   | 89%      |

### Weighted vs Geometric Centroid (verified subset)

| Protein | Geometric DCC | Weighted DCC | Δ      |
|---------|:------------:|:------------:|:------:|
| 1W50    | 5.4 Å       | 3.6 Å       | -1.8 Å |
| 1G1F    | 10.6 Å      | 4.8 Å       | -5.7 Å |

Intensity² weighting converted PTP1B cryptic site from MISS to EXCELLENT.

### Failure Analysis

**1ERE (20.3Å):** Pocket 4 = 18,037 Å³ — pathological mega-pocket that subsumes the
estrogen binding cavity. Weighted centroid cannot fix this because the pocket itself
is misdelineated. Root fix: sub-pocket decomposition or spike-native clustering.

**1BJ4, 1HHP (9.8Å):** Borderline marginal. Sites detected in correct region but
centroid offset by pocket shape irregularity.
