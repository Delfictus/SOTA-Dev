# PRISM-DSTW Final Production Run Report

Generated: 2026-05-22

## Status

**COMPLETE for the physically sound production path.**

`warp_jacobian` was quarantined because the available `warp_matrix.bin` payload contains `GpuWarpEntry` atom index/weight records, not physical displacement vectors. The DAG no longer registers `warp_jacobian`, and both proxy outputs were deleted:

- `shear_stress_field.parquet`: deleted
- `shear_stress_field.propagation.jsonl`: deleted

## Full-Scale Rust Execution

The remaining Rust extractors were run against the full 1,600-stream n80 source set.

```text
aromatic_kinematics          real 3.84s
mechanical_load              real 46.83s
signal_grid_differential     real 663.78s
total Rust extractor wall    714.45s
```

## Production Tensor Row Counts

```text
stream_snr_masks                         9,600 rows
spike_events_snr_masked                 75,390 rows
bocpd_survival_regimes                   3,200 rows
kinetic_strain_events                   23,920 rows
autonomous_steering_tensor               6,126 rows
aromatic_reorganization_tensor           2,336 rows
mechanical_load_network             10,689,600 rows
signal_grid_variance_channel         7,077,888 rows
```

## Omni-Tensor Evaluator

The legacy posthoc evaluator path was replaced with:

- `scripts/evaluate-glp1r-posthoc.py`

The evaluator uses lazy Polars scans and fuses these seven tensor classes:

1. `spike_events_snr_masked.parquet`
2. `signal_grid_variance_channel.parquet`
3. `mechanical_load_network.parquet`
4. `bocpd_survival_regimes.parquet`
5. `kinetic_strain_events.parquet`
6. `autonomous_steering_tensor.parquet`
7. `aromatic_reorganization_tensor.parquet`

KCC residue fields are used as the base receptor-interface surface.

Evaluator runtime:

```text
evaluate-glp1r-posthoc.py     real 1.05s
```

Final outputs:

```text
receptor_durability_risk_map.parquet             3,263 rows
receptor_durability_channel_summary.parquet         17 rows
receptor_durability_evaluation_summary.json
```

Validation:

```text
durability_risk_score_raw NaN count: 0
durability_risk_score_raw null count: 0
```

Durability class distribution:

```text
critical_durability_risk       164
elevated_durability_risk       652
mechanically_pruned            267
moderate_durability_risk     2,180
```

## Output Locations

```text
campaigns/glp1r_aleniglipron/integrated_spike_events/n80_full_scale/receptor_durability_risk_map.parquet
campaigns/glp1r_aleniglipron/integrated_spike_events/n80_full_scale/receptor_durability_channel_summary.parquet
campaigns/glp1r_aleniglipron/integrated_spike_events/n80_full_scale/receptor_durability_evaluation_summary.json
```
