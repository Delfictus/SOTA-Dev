# Directive 06 Enforcement Audit

Commit audited: `58c35e20` (`debt(06): derive shear principal directions`)

Verdict: PASS — Directive 06 is VERIFIED_RUNTIME.

## Requirements Checked

- `shear_stress_field.parquet` has `principal_x`, `principal_y`, `principal_z`.
- Principal direction columns are nonzero and varying.
- Direction provenance is present as `L3_FINITE_DIFFERENCE` because the campaign shear parquet did not contain L5 warp-matrix principal vectors.
- `ThermodynamicFieldStack` loads the principal direction metadata.
- The committed root `shear_stress_field.parquet` symlink is reproducible from a clean `git archive`.

## Runtime Evidence

- Current repository gate:
  - `D06_GATE_PASS 7077888 0.2354128509759903 L3_FINITE_DIFFERENCE`
  - all principal columns finite.

- Clean archive gate:
  - `ARCHIVE_D06_GATE_PASS 7077888 0.2354128509759903 L3_FINITE_DIFFERENCE`
  - root symlink and symlink target are both tracked in the commit.

- Focused tests:
  - `PYTHONPATH=src python3 -m pytest tests/test_shear_principal_directions.py tests/test_product_fiber_lookup.py -q`
  - result: `19 passed`

- Full regression:
  - `bash scripts/regression_gate.sh "directive-06-nonfinite-hardening"`
  - result: PASS
  - root pytest result inside gate: `1027 passed, 14 skipped`

## File Evidence

- `scripts/derive_shear_principal_directions.py`
  - derives finite-difference principal vectors from observed neighboring voxels.
  - preserves valid row-level L5 vectors when present.
  - rejects malformed sparse grids above `MAX_DENSE_FINITE_DIFFERENCE_CELLS`.
  - rejects non-finite `shear_stress` values before writing principal vectors.

- `src/prism_dstw/scoring/product_fiber_lookup.py`
  - `VoxelThermodynamicProfile` stores `shear_principal_direction` and `shear_direction_provenance`.
  - `_merge_shear()` consumes `principal_x/y/z` and provenance.
  - `ThermodynamicFieldStack.grid` mirrors direction metadata for downstream lookup.

- `tests/test_shear_principal_directions.py`
  - covers finite-difference derivation, empty input, singleton input, sparse constant grids, huge sparse rejection, non-finite shear rejection, L5 preservation, mixed row validity, and field-stack loading.

Final status: PASS.
