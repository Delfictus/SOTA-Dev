# DuckDB Purge Manifest

## Current Code Violations

- No active `import duckdb`, `from duckdb`, or DuckDB dependency violations found by AST/dependency scan.

## Remediated Active Producers

- `scripts/prism_interface_timestamp_miner.py` -> Polars lazy producer; regenerated `interface_time_bins.parquet`, `interface_support_intervals.parquet`, `interface_state_transitions.parquet`.
- `scripts/prism_dynamic_aligned_voxel_export.py` -> Polars dynamic event-bin join; regenerated `dynamic_voxel_event_time_bins.parquet` and Arrow sidecars with provenance metadata.
- `scripts/prism_hydration_event_statistics.py` -> Polars lazy producer; regenerated hydration ontology parquets.
- `scripts/prism_path_sampling_launcher.py` -> Polars lazy/eager bounded launcher; regenerated ranked windows and launch queue.
- `scripts/prism_voxel_variance_proxy_classifier.py` -> strict banned-producer rejection by default; no permissive override flag.
