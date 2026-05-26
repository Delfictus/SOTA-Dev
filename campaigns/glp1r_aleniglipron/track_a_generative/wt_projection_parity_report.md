# WT Projection Parity Report

Generated: `2026-05-26T13:44:50.241748+00:00`

Status: `VERIFIED_RAW_WT_PARITY`

## Summary

- Native authority: `rust_survivor_lookup_oracle`
- Raw projection status: `WT_NATIVE_RAW_PARITY_CONFIRMED`
- Raw WT/native ratio mean: `1.000000`
- Coordinate projection status: `WT_COORDINATE_PROJECTION_COLLAPSE`
- Repaired method: `native_wt_reward_plus_relative_field_liability_delta_v1`
- Stored/Rust diagnostic ratio mean: `1.844245`
- Stored native/Rust ratio mean: `1.000000`
- Stored reward status: `STORED_REWARD_INCLUDES_NON_WT_BONUSES`
- Coordinate projection/native ratio mean: `1.407075e-09`
- Calibrated WT self-parity ratio: `1.000000`

## Candidate Comparison

| idx | stored reward | stored native WT | native Rust WT | raw WT/native | coordinate WT | coordinate/native | WT liability |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 16.6299 | 10.6299 | 10.6299 | 1.0000 | 1.0000e-08 | 9.4074e-10 | 60.0000 |
| 1 | 14.6097 | 8.6097 | 8.6097 | 1.0000 | 1.0000e-08 | 1.1615e-09 | 62.0000 |
| 2 | 16.6088 | 10.6088 | 10.6088 | 1.0000 | 1.0000e-08 | 9.4261e-10 | 58.5000 |
| 3 | 14.1083 | 8.1083 | 8.1083 | 1.0000 | 1.0000e-08 | 1.2333e-09 | 64.5000 |
| 4 | 10.0793 | 4.0793 | 4.0793 | 1.0000 | 1.0000e-08 | 2.4514e-09 | 63.5000 |
| 5 | 12.0072 | 6.0072 | 6.0072 | 1.0000 | 1.0000e-08 | 1.6647e-09 | 65.5000 |
| 6 | 15.6299 | 9.6299 | 9.6299 | 1.0000 | 1.0000e-08 | 1.0384e-09 | 62.5000 |
| 7 | 15.6044 | 9.6044 | 9.6044 | 1.0000 | 1.0000e-08 | 1.0412e-09 | 63.5000 |
| 8 | 11.5833 | 5.5833 | 5.5833 | 1.0000 | 1.0000e-08 | 1.7911e-09 | 57.5000 |
| 9 | 11.5374 | 5.5374 | 5.5374 | 1.0000 | 1.0000e-08 | 1.8059e-09 | 58.5000 |

## Determination

The raw WT PGx path is the native Rust oracle reward invoked with an explicit survivor corpus.
Coordinate-field projection is not the native Rust reward path and remains preserved as relative liability evidence.
Candidate stored reward may include downstream consensus bonuses; it is reported as a diagnostic, not the WT denominator.
