# Directive 11 Bug Hunt

Verdict: PASS after amended bugfix

Committed state audited: `358785ea17535aabbe2a5c410ebec90c8212a47f`

## Prior High Findings

The first D11 bug hunt rejected the implementation because:

- `raw_wt_projection_ratio_mean` could be tautological if computed as Rust/Rust.
- Mixed candidate `training_survivor_corpus` values could fall back to stale default survivors.
- Missing candidate-recorded survivor corpora were accepted too late.
- Duplicate survivor `canonical_smiles` could provide arbitrary fallback coordinates.
- Survivor oracle reward rows could be consumed by row position without validating canonical order.

Those findings were fixed before commit `f09fa73314b4d985ea8d80aeb829ba76beb3a916`.

## Amended Bugfix Finding

Additional adversarial probes found HIGH severity bonus-removal corruption in `scripts/repair_wt_projection_parity.py`:

- Negative consensus bonus values were accepted and inflated native WT reward.
- `NaN` or infinite consensus bonus values were treated as absent.
- Multiple nonzero, distinct consensus-bonus columns were accepted by first-column precedence.

Fix applied in commit `358785ea17535aabbe2a5c410ebec90c8212a47f`:

- `required_finite_number()` parses parity numeric fields strictly.
- `candidate_consensus_bonus()` rejects negative, non-finite, and conflicting consensus-bonus fields.
- Duplicate equal bonus aliases remain accepted because the D10 candidate corpus records both `scaffold_consensus_bonus=3.0` and `consensus_complement_bonus=3.0`.

## Probe Evidence

```text
negative_bonus RAISED RuntimeError scaffold_consensus_bonus must be non-negative: -1.0
ambiguous_bonus_columns RAISED RuntimeError ambiguous consensus bonus columns for WT parity: scaffold_consensus_bonus=1.0, consensus_complement_bonus=5.0
nan_bonus_only RAISED RuntimeError scaffold_consensus_bonus is not finite: nan
duplicate_equal_bonus_columns VALUE 10.0
mismatch_status WT_NATIVE_RAW_PARITY_FAILED 0.925
```

Other probes:

- Mixed candidate survivor corpora raise and require explicit `--survivors`.
- Missing candidate-recorded survivor corpus raises immediately.
- Malformed candidate coordinates raise instead of silently falling back to survivor coordinates.
- Identical duplicate survivor fallback coordinates are accepted and conflicting duplicates are rejected.
- `generate_m3_dossier.py` parity context consumes the v2 parity report fields without error.

## Validation

```text
PYTHONPATH=src python3 -m pytest tests/test_wt_projection_parity.py tests/test_oracle_contract.py -q
34 passed in 0.75s
```

```text
PYTHONPATH=src python3 -m mypy --strict scripts/repair_wt_projection_parity.py tests/test_wt_projection_parity.py
Success: no issues found in 2 source files
```

```text
bash scripts/regression_gate.sh directive-11-bugfix
=== REGRESSION GATE PASSED for directive-11-bugfix ===
```

## Remaining Bugs

No CRITICAL or HIGH findings remain for Directive 11.

LOW observation: malformed candidate coordinate JSON currently surfaces as `JSONDecodeError` from the projection scorer. It is fail-closed and does not affect the D11 parity result, but a future CLI ergonomics patch could wrap it in a friendlier `RuntimeError`.
