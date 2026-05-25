# WT Projection Parity Report

Generated: `2026-05-25T05:48:24.834574+00:00`

Status: `CALIBRATED_PARITY_CONFIRMED`

## Summary

- Native authority: `rust_survivor_lookup_oracle`
- Raw projection status: `WT_PROJECTION_COLLAPSE`
- Repaired method: `native_wt_reward_plus_relative_field_liability_delta_v1`
- Stored/Rust native ratio mean: `1.000000`
- Projection/native ratio mean: `1.415939e-09`
- Calibrated WT self-parity ratio: `1.000000`

## Candidate Comparison

| idx | stored WT | Rust WT | projected WT | projection/native | WT liability |
|---:|---:|---:|---:|---:|---:|
| 0 | 8.7843 | 8.7843 | 1.0000e-08 | 1.1384e-09 | 66.0000 |
| 1 | 7.6608 | 7.6608 | 1.0000e-08 | 1.3054e-09 | 63.5000 |
| 2 | 8.0283 | 8.0283 | 1.0000e-08 | 1.2456e-09 | 60.0000 |
| 3 | 7.9059 | 7.9059 | 1.0000e-08 | 1.2649e-09 | 70.0000 |
| 4 | 5.7779 | 5.7779 | 1.0000e-08 | 1.7307e-09 | 67.0000 |
| 5 | 10.2519 | 10.2519 | 1.0000e-08 | 9.7543e-10 | 65.5000 |
| 6 | 9.6564 | 9.6564 | 1.0000e-08 | 1.0356e-09 | 64.0000 |
| 7 | 5.6141 | 5.6141 | 1.0000e-08 | 1.7812e-09 | 62.0000 |
| 8 | 10.1049 | 10.1049 | 1.0000e-08 | 9.8962e-10 | 63.0000 |
| 9 | 3.7139 | 3.7139 | 1.0000e-08 | 2.6926e-09 | 66.5000 |

## Determination

Absolute coordinate-field projection is not the native Rust reward path and remains preserved as negative evidence.
PGx resilience must use native WT reward as denominator and N80 fields as relative variant liability deltas.
