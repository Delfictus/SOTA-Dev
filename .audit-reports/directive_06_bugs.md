# Directive 06 Bug Hunt

Commit audited after fix: `58c35e20` (`debt(06): derive shear principal directions`)

Verdict: PASS — no CRITICAL or HIGH defects remain.

## Adversarial Probes

- Empty input: covered by `test_empty_shear_frame_returns_empty_direction_columns`.
- Singleton/thin partitions: covered by `test_singleton_partition_gets_zero_direction`.
- Sparse constant grids: covered by `test_sparse_constant_grid_does_not_invent_gradients`.
- Huge sparse indices: covered by `test_malformed_huge_sparse_grid_is_rejected`.
- Mixed L5/L3 row validity: covered by `test_mixed_l5_direction_validity_is_handled_per_row`.
- NaN/inf shear stress: covered by `test_nonfinite_shear_values_are_rejected`.
- Archive reproducibility: clean `git archive` gate passed.

## Resolved Findings

BUG_1:
  severity: HIGH
  location: `scripts/derive_shear_principal_directions.py:71`
  description: The initial adversarial probe showed that `NaN` or `inf` values in `shear_stress` could propagate into non-finite principal direction columns.
  reproduction: Construct a three-row shear frame with `shear_stress=[1.0, NaN, inf]` and run `derive_principal_directions`.
  impact: Malformed shear parquet could corrupt `principal_x/y/z` and downstream field-stack metadata.
  fix: The derivation now rejects non-finite shear values with `ValueError("shear_stress contains non-finite values")`.
  validation: The adversarial probe now reports `nan_inf_shear expected_rejection ValueError`; focused tests pass.

BUG_2:
  severity: LOW
  location: `src/prism_dstw/scoring/product_fiber_lookup.py:303`
  description: Direction metadata is loaded into `VoxelThermodynamicProfile` and `ThermodynamicFieldStack.grid`, but it is not exposed as extra lanes in the `[5, 12]` field-stack tensor.
  reproduction: Inspect `lookup_full_fiber`; the twelve lanes remain spike statistics, class one-hot, consensus, shear, hysteresis, pathway, and reversibility.
  impact: D06 satisfies the directive gate and runtime metadata requirement, but directional lanes are not yet part of the model tensor.
  suggested_fix: Add explicit direction lanes only in a future feature-dimension directive so tensor shape changes remain coordinated with policy dimensions and tests.

## Remaining Status

No CRITICAL or HIGH findings remain. LOW BUG_2 is documented as a scope boundary: D06 required materialized direction columns and field-stack consumption/provenance, not a new policy tensor dimension.
