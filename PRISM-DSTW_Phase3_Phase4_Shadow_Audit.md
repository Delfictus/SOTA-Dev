# PRISM-DSTW Phase 3/4 Shadow Audit

Generated: 2026-05-22

## 1. EXECUTIVE SUMMARY

Status: **[PENDING FULL-SCALE EXECUTION]**

Phase 3 Rust provenance parity and Phase 4 Rust-native extractors are implemented, compile cleanly, emit Parquet outputs, and pass the requested shadow audits after remediation. The full 1,600-stream heavy binary campaign was **not** claimed as complete in this dossier. Execution scope was:

- `aromatic_kinematics`: all 1,600 selected aromatic centroid streams, because payload is lightweight.
- `mechanical_load`: bounded 80-stream validation run.
- `signal_grid_differential`: bounded 80-stream validation run.
- `warp_jacobian`: bounded 2-stream validation run because each `warp_matrix.bin` is about 120 MB and full n80 warp hashing/processing is roughly 192 GiB.

No OOM occurred in the bounded runs. Full-scale Phase 5 execution remains pending as an explicit operational run.

## 2. IMPLEMENTED ARTIFACTS

Rust provenance bridge:

- `crates/prism-nhs/src/io/provenance.rs`
- `crates/prism-nhs/src/io/mod.rs`
- `crates/prism-nhs/src/lib.rs`

Rust-native extractors:

- `crates/prism-nhs/src/bin/aromatic_kinematics.rs`
- `crates/prism-nhs/src/bin/mechanical_load.rs`
- `crates/prism-nhs/src/bin/signal_grid_differential.rs`
- `crates/prism-nhs/src/bin/warp_jacobian.rs`

DAG registration:

- `scripts/scratch/topo_regenerate.py`

Protocol-state bridge update:

- `scripts/prism_protocol_state_extractor.py` now emits `protocol_state_summary.parquet` for Rust cold/warm phase grouping.

## 3. COMPILE GATE STATUS

All four Rust extractors pass `cargo check`:

```text
cargo check -p prism-nhs --bin aromatic_kinematics --no-default-features
Finished `dev` profile [unoptimized + debuginfo] target(s)

cargo check -p prism-nhs --bin mechanical_load --no-default-features
Finished `dev` profile [unoptimized + debuginfo] target(s)

cargo check -p prism-nhs --bin signal_grid_differential --no-default-features
Finished `dev` profile [unoptimized + debuginfo] target(s)

cargo check -p prism-nhs --bin warp_jacobian --no-default-features
Finished `dev` profile [unoptimized + debuginfo] target(s)
```

Existing workspace warnings remain in upstream `prism-nhs` library modules; the new bins compile.

## 4. OUTPUT STATUS

Current outputs in `campaigns/glp1r_aleniglipron/integrated_spike_events/n80_full_scale/`:

```text
aromatic_reorganization_tensor.parquet  rows=2336    row_groups=1   selected_input_count=1600
mechanical_load_network.parquet         rows=558400  row_groups=80  selected_stream_pair_count=80
signal_grid_variance_channel.parquet    rows=884736  row_groups=14  selected_input_count=80
shear_stress_field.parquet              rows=884736  row_groups=14  selected_input_count=2
```

The voxel outputs are physically chunked into 65,536-row Parquet batches after remediation. `ProvenanceParquetWriter::write()` now flushes each batch so ArrowWriter materializes separate row groups.

## 5. RUST PROVENANCE PARITY

`provenance.rs` implements:

- SHA-256 hashing of raw inputs with a streamed 1 MiB buffer.
- Arrow/Parquet key-value metadata: `created_by`, `generator_script`, `generator_hash`, `schema_version`, `pipeline_stage`, `partition_keys`, `source_parquets`, `raw_input_checksums`.
- Append-only `.propagation.jsonl` sidecars via `OpenOptions::append(true)`.
- Shared newtypes and typed math wrappers: `StreamId`, `ReplicaId`, `VoxelIdx`, `AtomIdx`, `RingIdx`, `HitCount`, `HitMean`, `MechanicalLoad`, `ShearStress`, `ForceVector`, `AscVector`.

Representative Parquet metadata:

```text
mechanical_load_network.parquet
created_by=prism-nhs-rust-provenance/0.1.0
schema_version=prism.mechanical_load_network.v1
pipeline_stage=phase4_mechanical_load
generator_hash=66b0e84489cd650df4d1d41211088eb5fff12f1051464b1210630c4791e9ff8b
```

Representative `.propagation.jsonl` raw checksum entry:

```json
{
  "module": "phase4_warp_jacobian",
  "operation": "raw_input_checksum_capture",
  "input_checksums": {
    "raw://media/diddy/PRISM-LBS/prism-glp1r-aleniglipron-workspace/20260518T031002Z/05_RESULTS/glp1r_aleniglipron_risk_map/glp1r_5VEX_WT/replica_0/glp1r_5VEX_WT_stream0_warp_matrix.bin": "8d41edc3c488fd3cda7d320b3532a902d24fc046d82b49f8057651a78c8a4741"
  },
  "parameters": {
    "grid_reconstruction": "3D cubic grid_dim reconstructed from raw warp_matrix byte length / 136-byte GpuWarpEntry records",
    "jacobian": "central finite differences over x/y/z spatial strides; boundaries use one-sided differences",
    "selected_input_count": 2,
    "voxel_output_row_group_size": 65536,
    "warp_vector_source": "GpuWarpEntry atom_indices/atom_weights proxy because raw warp_matrix.bin has no explicit displacement-vector header"
  }
}
```

## 6. MATHEMATICAL IMPLEMENTATION

`aromatic_kinematics`:

- Streams `aromatic_centroids_final.bin` as 12-byte XYZ triples.
- Uses `protocol_state_summary.parquet` to classify streams as `Cold_Phase` or `Warm_Phase`.
- Aggregates per `(condition_id, RingIdx)` without retaining all centroid rows.
- Computes warm-cold displacement and pooled displacement standard deviation.

`mechanical_load`:

- Pairs `forces_final.bin` and `asc_vectors.bin` by condition/replica/stream.
- Wraps triples as `ForceVector` and `AscVector`.
- Computes `MechanicalLoad = force dot asc` through `mechanical_dot()`.
- Writes in bounded 8-stream-pair batches.

`signal_grid_differential`:

- Parses `PRSGD001` signal grids into `HitCount`.
- Uses protocol-state thermal class to aggregate cold/warm hit means per voxel.
- Classifies voxels into `stable_occupied`, `thermally_destabilized`, `thermally_activated`, or `void`.
- Writes 65,536-row voxel chunks.

`warp_jacobian`:

- Reconstructs `grid_dim` from `warp_matrix.bin` byte length divided by 136-byte `GpuWarpEntry` records.
- Rejects non-cubic grids and `grid_dim < 2`.
- Reads z-slices by true 3D byte stride, keeping only previous/current/next slices.
- Computes x/y/z finite differences through typed `DerivativeScale` and `WarpDerivative`.
- Computes shear from symmetric off-diagonal Jacobian terms and emits `ShearStress`.

Known math caveat: the observed `warp_matrix.bin` format contains `GpuWarpEntry` atom indices/weights, not an explicit displacement-vector header. The current warp vector is therefore a documented proxy derived from atom indices/weights. Derivatives are in grid-index units because no physical voxel spacing is present in the raw file header.

## 7. DAG DEPENDENCY REPORT

Topological order:

```text
1. noise_floor_extractor -> scripts/prism_noise_floor_extractor.py -> stream_snr_masks.parquet
2. bocpd_survival_extractor -> scripts/prism_bocpd_extractor.py -> bocpd_survival_regimes.parquet
3. protocol_state_extractor -> scripts/prism_protocol_state_extractor.py -> autonomous_steering_tensor.parquet
4. kcc_decoder -> scripts/prism_kcc_decoder.py -> kcc_residue_fields.parquet
5. mechanical_load -> crates/prism-nhs/src/bin/mechanical_load.rs -> mechanical_load_network.parquet
6. warp_jacobian -> crates/prism-nhs/src/bin/warp_jacobian.rs -> shear_stress_field.parquet
7. spike_event_integrator -> scripts/prism_spike_event_integrator.py -> spike_events_snr_masked.parquet
8. adaptive_dt_extractor -> scripts/prism_adaptive_dt_extractor.py -> kinetic_strain_events.parquet
9. aromatic_kinematics -> crates/prism-nhs/src/bin/aromatic_kinematics.rs -> aromatic_reorganization_tensor.parquet
10. signal_grid_differential -> crates/prism-nhs/src/bin/signal_grid_differential.rs -> signal_grid_variance_channel.parquet
```

Dependency edges:

```text
noise_floor_extractor -> spike_event_integrator
bocpd_survival_extractor -> adaptive_dt_extractor
protocol_state_extractor -> aromatic_kinematics
protocol_state_extractor -> signal_grid_differential
```

Track 0 isolation check on the Phase 3/4 Rust files passed. The only Track 0 mentions found by repository scan are unrelated legacy/proxy scripts outside these extractors.

## 8. SHADOW AUDIT REPORT

Sub-Agent 1, Memory: **PASS with bounded remaining risks**

- No global 1,600-file raw-byte load found.
- `aromatic_kinematics` streams centroid triples and uses worker-local `try_fold`.
- `mechanical_load` is bounded to 8 stream pairs per chunk, but each pair still loads the force and ASC f32 files fully at the pair boundary.
- `signal_grid_differential` and `warp_jacobian` use worker-local reductions and 65,536-row output batches.
- Remaining bounded risk: full-grid accumulator arrays are still retained per worker/condition for signal and warp.

Sub-Agent 2, Ontology: **PASS**

- `warp_jacobian` now uses `DerivativeScale`, `WarpDerivative`, `ShearStress`, `VoxelIdx`, and `AtomIdx` at math boundaries.
- Raw scalar parsing is limited to binary boundary conversion.

Sub-Agent 3, Math: **PASS**

- `warp_jacobian` reconstructs the cubic 3D grid, computes x/y/z stride-aware derivatives, and does not use a naive `[i] - [i - 1]` gradient.
- Caveats: grid-index units, one-sided boundary derivatives, and documented proxy warp vector source.

Sub-Agent 4, Ledger: **PASS**

- Parquet metadata keys are present.
- Sidecars are append-only.
- Representative raw SHA-256 values were recomputed and matched.

## 9. FULL-SCALE EXECUTION GATE

Required before claiming complete Phase 5 execution:

```bash
cargo run -p prism-nhs --bin mechanical_load --no-default-features
cargo run -p prism-nhs --bin signal_grid_differential --no-default-features
cargo run -p prism-nhs --bin warp_jacobian --no-default-features
```

Expected resource notes:

- `mechanical_load`: bounded by 8 paired streams per chunk.
- `signal_grid_differential`: accumulates full-grid cold/warm means per worker/condition and writes chunked row groups.
- `warp_jacobian`: reads each warp file by z-slice, but full n80 will still read/hash roughly 192 GiB of warp payload.

No rescue-mode decision was made by this extraction pass.
