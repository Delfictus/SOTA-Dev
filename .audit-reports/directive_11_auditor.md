# Directive 11 Enforcement Audit

Verdict: PASS

Committed state audited: `358785ea17535aabbe2a5c410ebec90c8212a47f`

Requirement: Fix the WT -> WT PGx raw projection path so the raw native WT parity ratio is approximately 1.0. The Rust survivor lookup command must carry the explicit `--survivors` corpus argument, and stale default survivor corpora must not silently override candidate training provenance.

## Runtime Evidence

- `src/prism_dstw/orchestration/rust_reward_oracle.py` validates survivor oracle rows by exact `canonical_smiles` order and rejects duplicates before consuming rewards.
- `scripts/repair_wt_projection_parity.py` resolves the candidate-recorded `training_survivor_corpus` when the CLI uses the stale default.
- `scripts/repair_wt_projection_parity.py` computes `raw_wt_projection_ratio_mean` from stored candidate native WT reward after strict consensus-bonus removal against the Rust WT rescore, not Rust/Rust.
- `campaigns/glp1r_aleniglipron/track_a_generative/wt_projection_parity_report.json` reports:
  - `schema_version=PRISM.wt_projection_parity.v2`
  - `wt_parity_status=VERIFIED_RAW_WT_PARITY`
  - `raw_wt_projection_ratio_mean=0.9999999860729115`
  - `stored_native_vs_rust_ratio_mean=0.9999999860729115`
  - `raw_projection_status=WT_NATIVE_RAW_PARITY_CONFIRMED`
  - `coordinate_projection_status=WT_COORDINATE_PROJECTION_COLLAPSE`
  - `survivor_corpus_source=candidate_training_survivor_corpus`

## Commands Run

```text
PYTHONPATH=src python3 -m pytest tests/test_wt_projection_parity.py tests/test_oracle_contract.py -q
34 passed in 0.75s
```

```text
PYTHONPATH=src python3 -m mypy --strict scripts/repair_wt_projection_parity.py tests/test_wt_projection_parity.py
Success: no issues found in 2 source files
```

```text
PYTHONPATH=src python3 scripts/repair_wt_projection_parity.py --candidates campaigns/glp1r_aleniglipron/track_a_generative/gflownet_top100_cross_scaffold.parquet --n 10 --report campaigns/glp1r_aleniglipron/track_a_generative/wt_projection_parity_report.json --markdown campaigns/glp1r_aleniglipron/track_a_generative/wt_projection_parity_report.md
wt_parity_CONFIRMED method=parity_calibrated_liability_delta_v1 raw_wt_projection_ratio_mean=1.000000 raw_projection_status=WT_NATIVE_RAW_PARITY_CONFIRMED coordinate_projection_status=WT_COORDINATE_PROJECTION_COLLAPSE coordinate_projection_native_ratio_mean=1.407075e-09 survivor_corpus_source=candidate_training_survivor_corpus report=campaigns/glp1r_aleniglipron/track_a_generative/wt_projection_parity_report.json
```

```text
bash scripts/regression_gate.sh directive-11-bugfix
=== REGRESSION GATE PASSED for directive-11-bugfix ===
Root pytest: 1166 passed, 14 skipped, 8 warnings
```

## Determination

Directive 11 is `VERIFIED_RUNTIME`. The raw WT path now compares the candidate WT-native stored reward against a Rust survivor lookup rescore using the correct candidate training survivor corpus, and corrupt bonus-removal inputs fail closed.
