# Directive 01 Bug Hunter

Audited committed state: `debt-resolution-clean` at `1ab253e0`.

Verdict: no CRITICAL or HIGH bugs remain.

Focused probes passed:

- `numeric_value()` rejects `NaN`, `inf`, `-inf`, and string equivalents.
- Optional non-finite fields do not poison reward tensors.
- Canonical aliases are consumed: `shear_stress`, `hysteresis_score`, `reversibility`, `am1bcc_charge`.
- Single-row action tables produce finite tensors with expected shapes.
- Empty action tables raise clear `ValueError`.
- All-invalid action masks raise clear `ValueError`.
- Empty batches raise clear `ValueError`.
- `v1_base` and unknown reward versions sanitize non-finite raw rewards.

Findings:

- CRITICAL: none
- HIGH: none
- MEDIUM: none found in focused D01 probes
- LOW: none found in focused D01 probes

Validation probe exit code: `0`.
