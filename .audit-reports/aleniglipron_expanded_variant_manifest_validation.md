# Aleniglipron Expanded Variant Manifest Validation

## Runtime Outputs

- Manifest root: `campaigns/glp1r_aleniglipron/track_b_chronological/expanded_variant_run`
- Expanded variants: 1,464
- Baseline genealogical panel variants: 96
- Queued topology/trajectory jobs: 1,441
- Observed-grid extension rows: 3
- Receptor target/avoid entries: 63
- Validation verdict: PASS

## Generated Artifacts

- `aleniglipron_expanded_variant_run_manifest.json`
- `aleniglipron_expanded_variant_run_manifest.parquet`
- `aleniglipron_receptor_target_avoidance_matrix.parquet`
- `aleniglipron_receptor_target_avoidance_report.json`
- `true_perturbed_trajectory_run_plan.json`
- `variant_topology_materialization_manifest.json`
- `TRUE_PERTURBED_TRAJECTORY_RUNBOOK.md`
- Per-priority trajectory batch manifests

## Hashes

```text
1e4567fed8a5b09eece5ec14ede9a2f350a849071e1f933df8af4bd71df46cd2 aleniglipron_expanded_variant_run_manifest.json
ad233159b04762401221cf18a3ae021c2cc2b8127b62dc734d85da57c49b53bd aleniglipron_expanded_variant_run_manifest.parquet
ca192802b8aba311d9fd973c232a7a2f1aaa700d33f672873d0bf78199b50861 aleniglipron_receptor_target_avoidance_matrix.parquet
47e8fa030ad8baed1ba5538a7714a2c6d28c8b55e2d4e5c2000d2d1053e8c74d true_perturbed_trajectory_run_plan.json
23d471ae617cb11646be8f21c00094730d597476c9f30eecf47beb949193ea07 variant_topology_materialization_manifest.json
```

## Validation Commands

```bash
PYTHONPATH=src python3 scripts/build_aleniglipron_expanded_variant_run_manifest.py
PYTHONPATH=src python3 -m mypy --strict src/prism_dstw/calibration/aleniglipron_variant_manifest.py scripts/build_aleniglipron_expanded_variant_run_manifest.py tests/test_aleniglipron_expanded_variant_run_manifest.py
PYTHONPATH=src python3 -m pytest tests/test_aleniglipron_expanded_variant_run_manifest.py -q
bash scripts/regression_gate.sh aleniglipron-expanded-variant-manifest
```

## Gate Evidence

- Manifest builder: `variants=1464 baseline=96 queued=1401 observed_grid_extensions=3 targets=63 validation=PASS`
- Focused pytest: `4 passed`
- Focused mypy: `Success: no issues found in 3 source files`
- Regression gate: `=== REGRESSION GATE PASSED for aleniglipron-expanded-variant-manifest ===`
- Root pytest inside regression gate: `1215 passed, 14 skipped, 9 warnings`

## Claim Boundary

The manifest queues true perturbed PRISM trajectories. It does not label queued
variants as observed trajectory data until the referenced PRISM engine output
directories contain runtime artifacts.
