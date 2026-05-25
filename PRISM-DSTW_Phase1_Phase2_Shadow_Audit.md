# PRISM-DSTW Phase 1/2 Shadow Audit

## 1. EXECUTIVE SUMMARY

Status: **[PASS]**

Scope executed: Phase 1 legacy purge and Phase 2 temporal/high-order Polars extractors only. Phase 3 kinematic extractors and Phase 4 Rust voxel/Jacobian engines were not started.

Raw 12-class manifest confirmed by `scripts/scratch/audit_raw_classes.py`:

1. `*_adaptive_dt.bin`
2. `*_aromatic_centroids_final.bin`
3. `*_asc_vectors.bin`
4. `*_bocpd.jsonl`
5. `*_forces_final.bin`
6. `*_kcc_v2full.bin`
7. `*_noise_floor.json`
8. `prism_v2_*.bin`
9. `*_protocol_state.json`
10. `*_signal_grid.bin`
11. `*_spikes.bin`
12. `*_warp_matrix.bin`

Completed Phase 1/2 outputs in `campaigns/glp1r_aleniglipron/integrated_spike_events/n80_full_scale/`:

- `stream_snr_masks.parquet`: 9,600 rows
- `autonomous_steering_tensor.parquet`: 6,126 rows
- `bocpd_survival_regimes.parquet`: 3,200 rows
- `kinetic_strain_events.parquet`: 23,920 rows
- `kcc_residue_fields.parquet`: 652,600 rows
- `spike_events_snr_masked.parquet`: 75,390 rows from bounded smoke window, 100,000 selected raw spike records from `glp1r_6XOX_WT`, replica 0, stream 0

Gate status:

```text
$ PYTHONPATH=src:. mypy --strict scripts/prism_spike_event_integrator.py scripts/prism_kcc_decoder.py scripts/prism_noise_floor_extractor.py scripts/prism_protocol_state_extractor.py scripts/prism_bocpd_extractor.py scripts/prism_adaptive_dt_extractor.py scripts/prism_n80_extraction_common.py scripts/scratch/topo_regenerate.py
Success: no issues found in 8 source files

$ python3 scripts/ci/ban_check.py
BANNED IMPLEMENTATION CHECK PASSED
```

Shadow sub-agent verdicts:

- Sub-agent 1, memory/lazy execution: PASS
- Sub-agent 2, ontology/type enforcement: PASS after remediation
- Sub-agent 3, thermodynamic math: PASS
- Sub-agent 4, cryptographic ledger: PASS after superseding historical deficient entry
- Sub-agent 5, DAG/dependency and Track 0 isolation: PASS

## 2. SUB-AGENT 1: MEMORY & LAZY EXECUTION REPORT

No eager Polars readers were found in the target extraction scripts: no `pl.read_parquet()`, `pl.read_ipc()`, `pl.read_csv()`, or `pl.read_ndjson()`.

No `.collect()` calls were found in the target extraction scripts. Final materialization uses `write_provenance_parquet()` and routes `pl.LazyFrame` to `LazyFrame.sink_parquet()` in `src/prism_dstw/io.py`, so there is no final `.collect(streaming=True)` call to configure. Streaming/out-of-core writing is provided by the Polars sink.

Spike integrator memory behavior:

- `scripts/prism_spike_event_integrator.py` reads `spikes.bin` in fixed 65,536-record batches.
- It applies raw SNR prefiltering before Polars DataFrame construction.
- It then scans `stream_snr_masks.parquet` lazily and applies the persisted SNR predicate before output.

Exact bounded spike query plan:

```text
simple π 25/25 ["campaign_id", "condition_id", ... 23 other columns]
   WITH_COLUMNS:
   [[(col("__POLARS_CSER_0xbf46dd3ba077297c")) >= (col("snr5_threshold"))].alias("causal_anchor"), [(col("__POLARS_CSER_0xbf46dd3ba077297c")) / (col("snr3_threshold"))].alias("snr3_ratio")]
     WITH_COLUMNS:
     [col("intensity").cast(Float64).alias("__POLARS_CSER_0xbf46dd3ba077297c")]
      FILTER [(col("intensity").cast(Float64)) >= (col("snr3_threshold"))]
      FROM
        INNER JOIN:
        LEFT PLAN ON: [col("condition_id"), col("replica_id"), col("stream_id")]
           WITH_COLUMNS:
           [col("stream_id").strict_cast(UInt8), col("replica_id").strict_cast(UInt8), col("timestep"), col("voxel_idx"), col("primary_residue_idx")]
            DF ["campaign_id", "condition_id", "replica_id", "stream_id", ...]; PROJECT */21 COLUMNS
        RIGHT PLAN ON: [col("condition_id"), col("replica_id"), col("stream_id")]
          simple π 5/5 ["condition_id", "replica_id", ... 3 other columns]
            Parquet SCAN [campaigns/glp1r_aleniglipron/integrated_spike_events/n80_full_scale/stream_snr_masks.parquet]
            PROJECT 6/10 COLUMNS
            SELECTION: [([([([(col("sh_channel")) == (0)]) & ([(col("replica_id")) == (0)])]) & ([(col("condition_id")) == ("glp1r_6XOX_WT")])]) & ([(col("stream_id")) == (0)])]
            ESTIMATED ROWS: 9600
        END INNER JOIN
```

KCC decoder remediation:

- Prior row-dict accumulation was replaced with column-oriented arrays.
- `residue_indices: list[ResidueIdx]` is materialized before primitive conversion for Polars.

Adaptive DT remediation:

- No `path.read_bytes()` remains.
- The extractor validates 32-byte record alignment and reads one fixed-size binary record at a time.

## 3. SUB-AGENT 2: ONTOLOGY & MYPY REPORT

Exact strict type output:

```text
Success: no issues found in 8 source files
```

Escape-hatch scan result:

- No `typing.Any` in the Phase 1/2 extraction/common target files.
- No `typing.cast()` in the Phase 1/2 extraction/common target files.
- No `# type: ignore` in the Phase 1/2 extraction/common target files.

NewType wrappers used at extraction boundaries:

- `StreamId`: `scripts/prism_n80_extraction_common.py`, `scripts/prism_bocpd_extractor.py`, `scripts/prism_protocol_state_extractor.py`, `scripts/prism_kcc_decoder.py`
- `TimeStep`: `scripts/prism_spike_event_integrator.py`, `scripts/prism_protocol_state_extractor.py`, `scripts/prism_bocpd_extractor.py`
- `Picosecond`: `scripts/prism_spike_event_integrator.py`, `scripts/prism_bocpd_extractor.py`, `scripts/prism_adaptive_dt_extractor.py`
- `RunLength`: `scripts/prism_bocpd_extractor.py`
- `ResidueIdx`: `scripts/prism_protocol_state_extractor.py`, `scripts/prism_kcc_decoder.py`, `scripts/prism_spike_event_integrator.py`
- `VoxelIdx`: `scripts/prism_spike_event_integrator.py`

Ontology remediation that closed the final shadow finding:

```python
@dataclass(frozen=True)
class BocpdRegimeBoundary:
    frame_idx: TimeStep
    map_run_length: RunLength
    dt_ps: Picosecond
    survival_time_ps: Picosecond

def first_regime_boundary(item: StreamFile, dt_ps: Picosecond) -> BocpdRegimeBoundary:
    ...
    run_length = RunLength(json_int(record.get("map_run_length")))
    return BocpdRegimeBoundary(
        frame_idx=TimeStep(json_int(record.get("frame_idx"))),
        map_run_length=run_length,
        dt_ps=dt_ps,
        survival_time_ps=survival_time_ps(run_length, dt_ps),
    )
```

```python
residue_indices: list[ResidueIdx] = [
    ResidueIdx(residue)
    for residue in range(header.record_count)
]
"residue_idx": [int(residue) for residue in residue_indices],
```

## 4. SUB-AGENT 3: THERMODYNAMIC MATH REPORT

BOCPD physical survival time code:

```python
def survival_time_ps(map_run_length: RunLength, dt_ps: Picosecond) -> Picosecond:
    return Picosecond(float(map_run_length) * float(dt_ps))

joined = (
    base.join(ghost, on=("condition_id", "replica_id", "stream_id"), how="left")
    .join(protocol, on=("condition_id", "replica_id", "stream_id"), how="left")
    .with_columns(
        (pl.col("frame_idx").cast(pl.Float64) * pl.col("dt_ps")).alias("time_ps"),
        (pl.col("map_run_length").cast(pl.Float64) * pl.col("dt_ps")).alias("survival_time_ps"),
    )
)
```

Dynamic SNR threshold code:

```python
NoiseFloorRow(
    mu=json_float(mu_raw[channel]),
    sigma=json_float(sigma_raw[channel]),
    snr3_threshold=json_float(mu_raw[channel]) + snr_multiplier * json_float(sigma_raw[channel]),
    snr5_threshold=json_float(mu_raw[channel]) + anchor_multiplier * json_float(sigma_raw[channel]),
)
```

Spike SNR integration code:

```python
mask = (
    pl.scan_parquet(snr_mask_parquet)
    .filter(
        (pl.col("condition_id") == item.condition_id)
        & (pl.col("replica_id") == item.replica_id)
        & (pl.col("stream_id") == int(item.stream_id))
        & (pl.col("sh_channel") == 0)
    )
    .select("condition_id", "replica_id", "stream_id", "snr3_threshold", "snr5_threshold")
)

lazy = (
    frame.lazy()
    .join(mask, on=("condition_id", "replica_id", "stream_id"), how="inner")
    .filter(pl.col("intensity") >= pl.col("snr3_threshold"))
    .with_columns(
        (pl.col("intensity") >= pl.col("snr5_threshold")).alias("causal_anchor"),
        (pl.col("intensity") / pl.col("snr3_threshold")).alias("snr3_ratio"),
    )
)
```

Kinetic strain / BOCPD cross-reference code:

```python
regimes = pl.scan_parquet(bocpd_survival_parquet).select(
    "condition_id",
    "replica_id",
    "stream_id",
    "chunk_idx",
    "regime_id",
    "thermal_phase",
    "survival_time_ps",
    "temperature_K",
    "posterior_max",
    "reset_probability",
)

return (
    dt_lf.join(regimes, on=("condition_id", "replica_id", "stream_id", "chunk_idx"), how="left")
    .with_columns(
        (
            pl.col("dt_reduction_event")
            & pl.col("regime_id").is_not_null()
        ).alias("dt_drop_coincident_with_regime_change")
    )
)
```

Protocol steering aggregation is implemented by exploding `steering_focus_residues`, grouping by `condition_id`, `residue_idx`, and `thermal_phase`, then emitting steering sum, mean, max, observation count, stream support, replica support, and mean temperature.

## 5. SUB-AGENT 4: CRYPTOGRAPHIC LEDGER REPORT

Noise-floor raw checksum ledger excerpt from `stream_snr_masks.propagation.jsonl`. The full entry contains all 1,600 raw noise-floor JSON files; this excerpt preserves the raw JSON structure and shows the physical parameters plus representative SHA-256 input checksums.

```json
{
  "entry_id": "9c8ba114-e4c5-453f-9ebc-e013d05582c5",
  "gate_status": {
    "external_raw_uri": true,
    "raw_sha256": true,
    "write_provenance_parquet_used": true
  },
  "input_checksums_sample": {
    "raw://media/diddy/PRISM-LBS/prism-glp1r-aleniglipron-workspace/20260518T031002Z/05_RESULTS/glp1r_aleniglipron_risk_map/glp1r_5VEX_WT/replica_0/glp1r_5VEX_WT_stream00_noise_floor.json": "2f3ae6e0433d049e0e3429606c9a4f999072eea6269ce5e1879a9b14635d6bc5",
    "raw://media/diddy/PRISM-LBS/prism-glp1r-aleniglipron-workspace/20260518T031002Z/05_RESULTS/glp1r_aleniglipron_risk_map/glp1r_5VEX_WT/replica_0/glp1r_5VEX_WT_stream01_noise_floor.json": "ece72e66803a53f4359f9115b2014b25675762394f4754776f19c3d4b9f3971c",
    "raw://media/diddy/PRISM-LBS/prism-glp1r-aleniglipron-workspace/20260518T031002Z/05_RESULTS/glp1r_aleniglipron_risk_map/glp1r_5VEX_WT/replica_0/glp1r_5VEX_WT_stream02_noise_floor.json": "4b68a38ebfe4363b3a77f5e6a0571e88f7d95c75d7583ae2ffc76b78e09318aa"
  },
  "module": "phase2_noise_floor_extractor",
  "operation": "raw_input_checksum_capture",
  "output_value": {
    "output_path": "campaigns/glp1r_aleniglipron/integrated_spike_events/n80_full_scale/stream_snr_masks.parquet",
    "row_count": 9600
  },
  "parameters": {
    "causal_anchor_multiplier": 5.0,
    "channel_count": 6,
    "snr_multiplier": 3.0
  }
}
```

Ledger enforcement status:

- Every Phase 1/2 output is written through `write_provenance_parquet()`.
- Raw input SHA-256 sidecars are appended in `.propagation.jsonl`.
- The ledger writer uses append mode only.
- A historical spike smoke-test ledger entry that omitted the noise-floor raw file was not rewritten. It was superseded by appended entry `43afce9e-5094-49b0-bfc0-0070c58a8a8e`, with `supersedes: 7fd8477b-0d6a-4f4a-8f3f-027e3dc6a69f`.

Superseding spike correction includes SHA-256 for:

- `ghost_time_map.json`: `bf702c0d9b1fa779304b72c2f2d8cf01ac533701885d0f493780a24cdc21e8b3`
- `glp1r_6XOX_WT_stream00_noise_floor.json`: `530be9adb70c10753e3f3974ed5c0dd542709137c7d1ef534c07601760a29039`
- `glp1r_6XOX_WT_stream0_spikes.bin`: `f71fdaed62b8e2524bc08c1e2d994b8305ce6b1c8ce51b9eef923de4129e6eea`

## 6. SUB-AGENT 5: DAG DEPENDENCY REPORT

Exact topological sorter output:

```text
TOPOLOGICAL_ORDER
1. noise_floor_extractor -> scripts/prism_noise_floor_extractor.py -> stream_snr_masks.parquet
2. bocpd_survival_extractor -> scripts/prism_bocpd_extractor.py -> bocpd_survival_regimes.parquet
3. protocol_state_extractor -> scripts/prism_protocol_state_extractor.py -> autonomous_steering_tensor.parquet
4. kcc_decoder -> scripts/prism_kcc_decoder.py -> kcc_residue_fields.parquet
5. spike_event_integrator -> scripts/prism_spike_event_integrator.py -> spike_events_snr_masked.parquet
6. adaptive_dt_extractor -> scripts/prism_adaptive_dt_extractor.py -> kinetic_strain_events.parquet
DEPENDENCY_EDGES
noise_floor_extractor -> spike_event_integrator
bocpd_survival_extractor -> adaptive_dt_extractor
```

Dependency compliance:

- Noise floor runs before spike integration.
- BOCPD survival regimes run before adaptive DT kinetic strain extraction.
- Protocol state is registered as a Phase 2 extractor and is available before downstream cold/warm spatial classification work in Phase 4.

Track 0 isolation:

- No references found to `Track_0_Interference_Workbook.csv`.
- No references found to `track_0_manual_emulation`.
- No references found to `Track 0`.

Final compliance state: **[PASS]** for Phase 1 and Phase 2 implementation, validation gates, shadow audit, and append-only lineage.
