#!/usr/bin/env python3
"""Chunk-decode PRSPK001 spike streams and apply n80 SNR masking."""

from __future__ import annotations

import argparse
import struct
import time
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq

from prism_dstw.io import provenance_metadata
from prism_dstw.ontology import Picosecond, StreamId
from prism_dstw.propagation_ledger import append_propagation_entry, build_entry

from scripts.prism_n80_extraction_common import (
    CAMPAIGN_ID,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_RAW_ROOT,
    StreamFile,
    discover_stream_files,
    emit,
    filter_streams,
    json_float,
    json_int,
    json_list,
    json_object,
    parse_stream_selector,
    read_json_object,
)


SPIKE_DTYPE = np.dtype(
    [
        ("timestep", "<i4"),
        ("voxel_idx", "<i4"),
        ("position", "<f4", (3,)),
        ("intensity", "<f4"),
        ("nearby_residues", "<i4", (8,)),
        ("n_residues", "<i4"),
        ("spike_source", "<i4"),
        ("wavelength_nm", "<f4"),
        ("aromatic_type", "<i4"),
        ("aromatic_residue_id", "<i4"),
        ("water_density", "<f4"),
        ("vibrational_energy", "<f4"),
        ("n_nearby_excited", "<i4"),
        ("wd_change", "<f4"),
        ("phase_bits", "<u4"),
    ],
    align=False,
)
PHASE_LABELS = np.array(
    [
        "phase_cold_hold",
        "phase_ramp_up",
        "phase_warm_hold",
        "phase_ramp_down",
        "phase_cold_return",
    ],
    dtype=object,
)
PHASE_COLUMNS = (
    "spikes_cold_hold",
    "spikes_ramp_up",
    "spikes_warm_hold",
    "spikes_ramp_down",
    "spikes_cold_return",
)


@dataclass(frozen=True)
class SpikeHeader:
    stream_id: StreamId
    run_id: str
    stem: str
    record_count: int
    byte_stride: int
    payload_size: int
    payload_offset: int


def _read_u64(handle: object) -> int:
    if not hasattr(handle, "read"):
        raise TypeError("binary handle does not expose read()")
    raw = handle.read(8)
    if not isinstance(raw, bytes) or len(raw) != 8:
        raise ValueError("unexpected EOF while reading u64")
    return int(struct.unpack("<Q", raw)[0])


def parse_header(path: Path) -> SpikeHeader:
    with path.open("rb") as handle:
        magic = handle.read(8).decode("ascii", errors="replace")
        if magic != "PRSPK001":
            raise ValueError(f"{path}: expected PRSPK001, got {magic}")
        schema_version, endian_marker, stream_id = struct.unpack("<III", handle.read(12))
        if schema_version != 1:
            raise ValueError(f"{path}: unsupported schema_version {schema_version}")
        if endian_marker != 0x01020304:
            raise ValueError(f"{path}: unsupported endian marker {endian_marker}")
        run_len = _read_u64(handle)
        run_id = handle.read(run_len).decode("utf-8", errors="replace")
        stem_len = _read_u64(handle)
        stem = handle.read(stem_len).decode("utf-8", errors="replace")
        record_count, byte_stride, payload_size = struct.unpack("<QQQ", handle.read(24))
        payload_offset = handle.tell()
    if int(byte_stride) != SPIKE_DTYPE.itemsize:
        raise ValueError(f"{path}: expected {SPIKE_DTYPE.itemsize} byte stride, got {byte_stride}")
    return SpikeHeader(
        stream_id=StreamId(stream_id),
        run_id=run_id,
        stem=stem,
        record_count=int(record_count),
        byte_stride=int(byte_stride),
        payload_size=int(payload_size),
        payload_offset=int(payload_offset),
    )


@dataclass(frozen=True)
class SpikeThresholds:
    snr3_threshold: float
    snr5_threshold: float


@dataclass(frozen=True)
class ProtocolPhaseBoundaries:
    cold_hold_end: int
    ramp_end: int
    warm_hold_end: int
    ramp_down_end: int
    graph_capture_multiplier: float


@dataclass
class SawtoothUnrollState:
    current_offset: int = 0
    current_chunk_max: int = 0
    previous_timestep: int | None = None


def dt_ps_from_ghost_time(item: StreamFile) -> Picosecond:
    from scripts.prism_n80_extraction_common import json_float, read_json_object

    data = read_json_object(item.path.parent / "ghost_time_map.json")
    streams = json_list(data.get("streams"))
    if not streams:
        return Picosecond(0.004)
    for row in streams:
        stream_object = json_object(row)
        if stream_object and json_int(stream_object.get("stream_id"), -1) == int(item.stream_id):
            return Picosecond(json_float(stream_object.get("dt_fs"), 4.0) / 1000.0)
    return Picosecond(0.004)


@dataclass(frozen=True)
class IntegrationStats:
    raw_records: int
    kept_records: int
    streams_processed: int
    elapsed_seconds: float


def load_snr_thresholds(path: Path) -> dict[tuple[str, int, int], SpikeThresholds]:
    """Load the all-stream SNR mask once and use it as the filtering authority."""

    mask = (
        pl.scan_parquet(path)
        .filter(pl.col("sh_channel") == 0)
        .select(
            "condition_id",
            "replica_id",
            "stream_id",
            "snr3_threshold",
            "snr5_threshold",
        )
        .collect()
    )
    if mask.is_empty():
        raise ValueError(f"{path}: no sh_channel=0 SNR mask rows found")

    thresholds: dict[tuple[str, int, int], SpikeThresholds] = {}
    for row in mask.iter_rows(named=True):
        key = (str(row["condition_id"]), int(row["replica_id"]), int(row["stream_id"]))
        if key in thresholds:
            raise ValueError(f"{path}: duplicate SNR mask key {key}")
        thresholds[key] = SpikeThresholds(
            snr3_threshold=float(row["snr3_threshold"]),
            snr5_threshold=float(row["snr5_threshold"]),
        )
    return thresholds


def load_protocol_boundaries(path: Path) -> dict[tuple[str, int, int], ProtocolPhaseBoundaries]:
    """Load five-phase protocol schedule boundaries for each stream."""

    required = {
        "condition_id",
        "replica_id",
        "stream_id",
        "cold_hold_end",
        "ramp_end",
        "warm_hold_end",
        "ramp_down_end",
    }
    schema_names = set(pl.scan_parquet(path).collect_schema().names())
    missing = sorted(required - schema_names)
    if missing:
        raise ValueError(f"{path}: missing protocol boundary column(s): {missing}")
    optional_columns = [
        name
        for name in ("total_steps", "observed_timestep_max", "graph_capture_multiplier")
        if name in schema_names
    ]

    frame = (
        pl.scan_parquet(path)
        .select(
            [
                "condition_id",
                "replica_id",
                "stream_id",
                "cold_hold_end",
                "ramp_end",
                "warm_hold_end",
                "ramp_down_end",
                *optional_columns,
            ]
        )
        .unique()
        .collect()
    )
    boundaries: dict[tuple[str, int, int], ProtocolPhaseBoundaries] = {}
    for row in frame.iter_rows(named=True):
        key = (str(row["condition_id"]), int(row["replica_id"]), int(row["stream_id"]))
        if key in boundaries:
            raise ValueError(f"{path}: duplicate protocol boundary key {key}")
        boundaries[key] = ProtocolPhaseBoundaries(
            cold_hold_end=int(row["cold_hold_end"]),
            ramp_end=int(row["ramp_end"]),
            warm_hold_end=int(row["warm_hold_end"]),
            ramp_down_end=int(row["ramp_down_end"]),
            graph_capture_multiplier=(
                float(row["graph_capture_multiplier"])
                if "graph_capture_multiplier" in row and row["graph_capture_multiplier"] is not None
                else (
                    float(row["total_steps"]) / float(row["observed_timestep_max"])
                    if "total_steps" in row
                    and "observed_timestep_max" in row
                    and row["observed_timestep_max"] is not None
                    and float(row["observed_timestep_max"]) != 0.0
                    else 1.0
                )
            ),
        )
    return boundaries


def spike_output_schema(metadata: Mapping[str, str]) -> pa.Schema:
    return pa.schema(
        [
            pa.field("campaign_id", pa.large_string()),
            pa.field("condition_id", pa.large_string()),
            pa.field("replica_id", pa.uint8()),
            pa.field("stream_id", pa.uint8()),
            pa.field("run_id", pa.large_string()),
            pa.field("event_index_in_stream", pa.int64()),
            pa.field("timestep", pa.int32()),
            pa.field("absolute_timestep", pa.int64()),
            pa.field("true_md_step", pa.float64()),
            pa.field("time_ps", pa.float64()),
            pa.field("voxel_idx", pa.int32()),
            pa.field("x", pa.float32()),
            pa.field("y", pa.float32()),
            pa.field("z", pa.float32()),
            pa.field("intensity", pa.float32()),
            pa.field("primary_residue_idx", pa.int32()),
            pa.field("n_residues", pa.int32()),
            pa.field("spike_source", pa.int32()),
            pa.field("wavelength_nm", pa.float32()),
            pa.field("water_density", pa.float32()),
            pa.field("wd_change", pa.float32()),
            pa.field("vibrational_energy", pa.float32()),
            pa.field("phase_bits", pa.uint32()),
            pa.field("thermal_phase_code", pa.uint8()),
            pa.field("thermal_phase", pa.large_string()),
            pa.field("snr3_threshold", pa.float64()),
            pa.field("snr5_threshold", pa.float64()),
            pa.field("causal_anchor", pa.bool_()),
            pa.field("snr3_ratio", pa.float64()),
        ]
    ).with_metadata({key.encode("utf-8"): value.encode("utf-8") for key, value in metadata.items()})


def spike_batch_frame(
    *,
    item: StreamFile,
    header: SpikeHeader,
    records: np.ndarray[tuple[int], np.dtype[np.void]],
    keep_mask: np.ndarray[tuple[int], np.dtype[np.bool_]],
    absolute_record: int,
    dt_ps: Picosecond,
    thresholds: SpikeThresholds,
) -> pl.DataFrame | None:
    kept = int(np.count_nonzero(keep_mask))
    if kept == 0:
        return None

    timestep = records["timestep"].astype(np.int32, copy=False)[keep_mask]
    voxel_idx = records["voxel_idx"].astype(np.int32, copy=False)[keep_mask]
    position = records["position"][keep_mask]
    intensity = records["intensity"].astype(np.float32, copy=False)[keep_mask]
    primary_residue = records["nearby_residues"][:, 0].astype(np.int32, copy=False)[keep_mask]
    filtered = records[keep_mask]
    event_index = np.arange(absolute_record, absolute_record + len(records), dtype=np.int64)[keep_mask]
    snr3 = np.full(kept, thresholds.snr3_threshold, dtype=np.float64)
    snr5 = np.full(kept, thresholds.snr5_threshold, dtype=np.float64)

    return pl.DataFrame(
        {
            "campaign_id": [CAMPAIGN_ID] * kept,
            "condition_id": [item.condition_id] * kept,
            "replica_id": np.full(kept, item.replica_id, dtype=np.uint8),
            "stream_id": np.full(kept, int(header.stream_id), dtype=np.uint8),
            "run_id": [header.run_id] * kept,
            "event_index_in_stream": event_index,
            "timestep": timestep,
            "time_ps": timestep.astype(np.float64, copy=False) * float(dt_ps),
            "voxel_idx": voxel_idx,
            "x": position[:, 0].astype(np.float32, copy=False),
            "y": position[:, 1].astype(np.float32, copy=False),
            "z": position[:, 2].astype(np.float32, copy=False),
            "intensity": intensity,
            "primary_residue_idx": primary_residue,
            "n_residues": filtered["n_residues"].astype(np.int32, copy=False),
            "spike_source": filtered["spike_source"].astype(np.int32, copy=False),
            "wavelength_nm": filtered["wavelength_nm"].astype(np.float32, copy=False),
            "water_density": filtered["water_density"].astype(np.float32, copy=False),
            "wd_change": filtered["wd_change"].astype(np.float32, copy=False),
            "vibrational_energy": filtered["vibrational_energy"].astype(np.float32, copy=False),
            "phase_bits": filtered["phase_bits"].astype(np.uint32, copy=False),
            "snr3_threshold": snr3,
            "snr5_threshold": snr5,
            "causal_anchor": intensity >= thresholds.snr5_threshold,
            "snr3_ratio": intensity.astype(np.float64, copy=False) / thresholds.snr3_threshold,
        }
    )


def unroll_and_bin_spike_frame(
    frame: pl.DataFrame,
    *,
    phase_boundaries: ProtocolPhaseBoundaries,
    dt_ps: Picosecond,
    state: SawtoothUnrollState,
) -> pl.DataFrame:
    first_timestep = int(frame.select(pl.col("timestep").first()).item())
    first_row_resets = state.previous_timestep is not None and first_timestep < state.previous_timestep
    annotated = (
        frame.with_row_index("_row_nr")
        .with_columns(
            pl.when(pl.col("_row_nr") == 0)
            .then(pl.lit(first_row_resets))
            .otherwise((pl.col("timestep").diff() < 0).fill_null(False))
            .alias("_is_reset")
        )
        .with_columns(pl.col("_is_reset").cast(pl.UInt32).cum_sum().alias("_chunk_idx"))
    )
    chunks = (
        annotated.group_by("_chunk_idx")
        .agg(pl.col("timestep").max().alias("_chunk_size"))
        .sort("_chunk_idx")
    )
    first_chunk_id = int(chunks.select(pl.col("_chunk_idx").first()).item())
    initial_offset = state.current_offset + (state.current_chunk_max if first_row_resets else 0)
    first_effective_prior = 0 if first_row_resets else state.current_chunk_max
    chunks = chunks.with_columns(
        pl.when(pl.col("_chunk_idx") == first_chunk_id)
        .then(pl.max_horizontal(pl.col("_chunk_size"), pl.lit(first_effective_prior)))
        .otherwise(pl.col("_chunk_size"))
        .alias("_effective_chunk_size")
    ).with_columns(
        (pl.lit(initial_offset) + pl.col("_effective_chunk_size").shift(1).fill_null(0).cum_sum()).alias("_offset")
    )
    out = annotated.join(chunks.select(["_chunk_idx", "_offset"]), on="_chunk_idx", how="left").with_columns(
        (pl.col("timestep").cast(pl.Int64) + pl.col("_offset").cast(pl.Int64)).alias("absolute_timestep")
    )
    absolute = pl.col("absolute_timestep")
    out = out.with_columns(
        (absolute.cast(pl.Float64) * float(phase_boundaries.graph_capture_multiplier)).alias("true_md_step")
    )
    phase_step = pl.col("true_md_step")
    out = out.with_columns(
        [
            (phase_step * float(dt_ps)).alias("time_ps"),
            pl.when(phase_step <= phase_boundaries.cold_hold_end)
            .then(pl.lit(0, dtype=pl.UInt8))
            .when((phase_step > phase_boundaries.cold_hold_end) & (phase_step <= phase_boundaries.ramp_end))
            .then(pl.lit(1, dtype=pl.UInt8))
            .when((phase_step > phase_boundaries.ramp_end) & (phase_step <= phase_boundaries.warm_hold_end))
            .then(pl.lit(2, dtype=pl.UInt8))
            .when((phase_step > phase_boundaries.warm_hold_end) & (phase_step <= phase_boundaries.ramp_down_end))
            .then(pl.lit(3, dtype=pl.UInt8))
            .otherwise(pl.lit(4, dtype=pl.UInt8))
            .alias("thermal_phase_code"),
            pl.when(phase_step <= phase_boundaries.cold_hold_end)
            .then(pl.lit("phase_cold_hold"))
            .when((phase_step > phase_boundaries.cold_hold_end) & (phase_step <= phase_boundaries.ramp_end))
            .then(pl.lit("phase_ramp_up"))
            .when((phase_step > phase_boundaries.ramp_end) & (phase_step <= phase_boundaries.warm_hold_end))
            .then(pl.lit("phase_warm_hold"))
            .when((phase_step > phase_boundaries.warm_hold_end) & (phase_step <= phase_boundaries.ramp_down_end))
            .then(pl.lit("phase_ramp_down"))
            .otherwise(pl.lit("phase_cold_return"))
            .alias("thermal_phase"),
        ]
    ).drop(["_row_nr", "_is_reset", "_chunk_idx", "_offset"])

    last_timestep = int(frame.select(pl.col("timestep").last()).item())
    last_chunk_id = int(chunks.select(pl.col("_chunk_idx").last()).item())
    last_offset = int(chunks.select(pl.col("_offset").last()).item())
    last_chunk_size = int(chunks.select(pl.col("_chunk_size").last()).item())
    if last_chunk_id == first_chunk_id and not first_row_resets:
        state.current_chunk_max = max(state.current_chunk_max, last_chunk_size)
    else:
        state.current_chunk_max = last_chunk_size
    state.current_offset = last_offset
    state.previous_timestep = last_timestep
    return out.select(
        [
            "campaign_id",
            "condition_id",
            "replica_id",
            "stream_id",
            "run_id",
            "event_index_in_stream",
            "timestep",
            "absolute_timestep",
            "true_md_step",
            "time_ps",
            "voxel_idx",
            "x",
            "y",
            "z",
            "intensity",
            "primary_residue_idx",
            "n_residues",
            "spike_source",
            "wavelength_nm",
            "water_density",
            "wd_change",
            "vibrational_energy",
            "phase_bits",
            "thermal_phase_code",
            "thermal_phase",
            "snr3_threshold",
            "snr5_threshold",
            "causal_anchor",
            "snr3_ratio",
        ]
    )


def iter_spike_batches(
    item: StreamFile,
    *,
    thresholds_by_stream: Mapping[tuple[str, int, int], SpikeThresholds],
    protocol_boundaries_by_stream: Mapping[tuple[str, int, int], ProtocolPhaseBoundaries],
    max_records: int | None,
    start_record: int,
    chunk_records: int,
) -> Iterator[tuple[pl.DataFrame | None, int, int]]:
    header = parse_header(item.path)
    key = (item.condition_id, item.replica_id, int(header.stream_id))
    thresholds = thresholds_by_stream.get(key)
    if thresholds is None:
        raise ValueError(f"missing SNR mask row for {key}")
    phase_boundaries = protocol_boundaries_by_stream.get(key)
    if phase_boundaries is None:
        raise ValueError(f"missing protocol boundary row for {key}")
    dt_ps = dt_ps_from_ghost_time(item)
    available = max(header.record_count - start_record, 0)
    requested = available if max_records is None else min(max_records, available)
    if requested <= 0:
        raise ValueError(f"{item.path}: no records selected")
    remaining = requested
    absolute_record = start_record
    unroll_state = SawtoothUnrollState()
    with item.path.open("rb") as handle:
        handle.seek(header.payload_offset + (start_record * header.byte_stride))
        while remaining > 0:
            batch_records = min(chunk_records, remaining)
            raw = handle.read(batch_records * header.byte_stride)
            if len(raw) != batch_records * header.byte_stride:
                raise ValueError(f"{item.path}: truncated spike batch")
            records = np.frombuffer(raw, dtype=SPIKE_DTYPE, count=batch_records)
            intensity = records["intensity"].astype(np.float32, copy=False)
            keep_mask = intensity >= thresholds.snr3_threshold
            frame = spike_batch_frame(
                item=item,
                header=header,
                records=records,
                keep_mask=keep_mask,
                absolute_record=absolute_record,
                dt_ps=dt_ps,
                thresholds=thresholds,
            )
            if frame is not None:
                frame = unroll_and_bin_spike_frame(
                    frame,
                    phase_boundaries=phase_boundaries,
                    dt_ps=dt_ps,
                    state=unroll_state,
                )
            if frame is not None:
                yield frame, batch_records, frame.height
            else:
                yield None, batch_records, 0
            remaining -= batch_records
            absolute_record += batch_records


def aggregate_phase_batch(frame: pl.DataFrame) -> pl.DataFrame:
    break_phase = pl.col("thermal_phase").is_in(["phase_ramp_up", "phase_warm_hold"])
    return (
        frame.group_by(["condition_id", "replica_id", "stream_id", "primary_residue_idx"])
        .agg(
            [
                pl.len().cast(pl.UInt64).alias("masked_spike_count"),
                (pl.col("thermal_phase") == "phase_cold_hold").cast(pl.UInt64).sum().alias("spikes_cold_hold"),
                (pl.col("thermal_phase") == "phase_ramp_up").cast(pl.UInt64).sum().alias("spikes_ramp_up"),
                (pl.col("thermal_phase") == "phase_warm_hold").cast(pl.UInt64).sum().alias("spikes_warm_hold"),
                (pl.col("thermal_phase") == "phase_ramp_down").cast(pl.UInt64).sum().alias("spikes_ramp_down"),
                (pl.col("thermal_phase") == "phase_cold_return").cast(pl.UInt64).sum().alias("spikes_cold_return"),
                pl.col("time_ps").filter(break_phase).min().alias("first_break_time_ps"),
                pl.col("time_ps").filter(break_phase).sum().alias("break_time_ps_sum"),
                break_phase.cast(pl.UInt64).sum().alias("break_spike_count"),
            ]
        )
        .rename({"primary_residue_idx": "residue_idx"})
    )


def reduce_phase_partials(partials: list[pl.DataFrame]) -> pl.DataFrame:
    combined = pl.concat(partials, how="vertical")
    return combined.group_by(["condition_id", "replica_id", "stream_id", "residue_idx"]).agg(
        [
            pl.col("masked_spike_count").sum().alias("masked_spike_count"),
            *[pl.col(name).sum().alias(name) for name in PHASE_COLUMNS],
            pl.col("first_break_time_ps").min().alias("first_break_time_ps"),
            pl.col("break_time_ps_sum").sum().alias("break_time_ps_sum"),
            pl.col("break_spike_count").sum().alias("break_spike_count"),
        ]
    )


def finalize_phase_aggregate(frame: pl.DataFrame) -> pl.DataFrame:
    return frame.with_columns(
        [
            (pl.col("spikes_cold_return").cast(pl.Int64) - pl.col("spikes_cold_hold").cast(pl.Int64)).alias(
                "hysteresis_delta"
            ),
            pl.when(pl.col("break_spike_count") > 0)
            .then(pl.col("break_time_ps_sum") / pl.col("break_spike_count"))
            .otherwise(None)
            .alias("mean_break_time_ps"),
        ]
    ).select(
        [
            "condition_id",
            "replica_id",
            "stream_id",
            "residue_idx",
            "masked_spike_count",
            *PHASE_COLUMNS,
            "hysteresis_delta",
            "first_break_time_ps",
            "mean_break_time_ps",
            "break_spike_count",
        ]
    )


def write_full_spike_parquet(
    *,
    selected: list[StreamFile],
    output: Path,
    snr_mask_parquet: Path,
    protocol_state_summary: Path,
    start_record: int,
    max_records_per_stream: int | None,
    chunk_records: int,
) -> IntegrationStats:
    repo_root = Path.cwd().resolve()
    raw_bytes = sum(item.path.stat().st_size for item in selected)
    thresholds_by_stream = load_snr_thresholds(snr_mask_parquet)
    protocol_boundaries_by_stream = load_protocol_boundaries(protocol_state_summary)
    missing = [
        (item.condition_id, item.replica_id, int(item.stream_id))
        for item in selected
        if (item.condition_id, item.replica_id, int(item.stream_id)) not in thresholds_by_stream
    ]
    if missing:
        raise ValueError(f"missing SNR mask rows for {len(missing)} selected streams; first={missing[0]}")
    missing_protocol = [
        (item.condition_id, item.replica_id, int(item.stream_id))
        for item in selected
        if (item.condition_id, item.replica_id, int(item.stream_id)) not in protocol_boundaries_by_stream
    ]
    if missing_protocol:
        raise ValueError(
            f"missing protocol boundary rows for {len(missing_protocol)} selected streams; "
            f"first={missing_protocol[0]}"
        )

    metadata = provenance_metadata(
        producer_script=Path(__file__),
        source_parquets=[snr_mask_parquet, protocol_state_summary],
        schema_version="prism.spike_events_snr_masked.v5",
        pipeline_stage="phase1_spike_event_integrator",
        partition_keys=("condition_id", "replica_id", "stream_id", "timestep"),
        extra={
            "campaign_id": CAMPAIGN_ID,
            "raw_input_count": len(selected),
            "raw_bytes_declared": raw_bytes,
            "snr_mask_channel": 0,
            "writer": "pyarrow_streaming_parquet_writer",
            "thermal_phase_binning": "five_phase_ccns_protocol",
            "time_unroll": "absolute_timestep * graph_capture_multiplier",
        },
        repo_root=repo_root,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        output.unlink()
    phase_output = output.parent / "spike_phase_residue_fields.parquet"
    if phase_output.exists():
        phase_output.unlink()

    schema = spike_output_schema(metadata)
    started = time.monotonic()
    raw_records = 0
    kept_records = 0
    streams_processed = 0
    phase_partials: list[pl.DataFrame] = []
    phase_reduced: pl.DataFrame | None = None
    with pq.ParquetWriter(output, schema=schema, compression="zstd", write_statistics=True) as writer:
        for index, item in enumerate(selected, start=1):
            stream_raw = 0
            stream_kept = 0
            for frame, batch_raw, batch_kept in iter_spike_batches(
                item,
                thresholds_by_stream=thresholds_by_stream,
                protocol_boundaries_by_stream=protocol_boundaries_by_stream,
                max_records=max_records_per_stream,
                start_record=start_record,
                chunk_records=chunk_records,
            ):
                stream_raw += batch_raw
                stream_kept += batch_kept
                if frame is not None and batch_kept > 0:
                    phase_partials.append(aggregate_phase_batch(frame))
                    if len(phase_partials) >= 128:
                        phase_reduced = reduce_phase_partials(
                            [phase_reduced, *phase_partials] if phase_reduced is not None else phase_partials
                        )
                        phase_partials = []
                    writer.write_table(frame.to_arrow(), row_group_size=100_000)
            raw_records += stream_raw
            kept_records += stream_kept
            streams_processed += 1
            emit(
                "PROGRESS "
                f"{index}/{len(selected)} streams "
                f"raw={raw_records} kept={kept_records} "
                f"last={item.condition_id}/replica_{item.replica_id}/stream_{int(item.stream_id)}"
            )
    if phase_partials or phase_reduced is not None:
        final_phase = finalize_phase_aggregate(
            reduce_phase_partials([phase_reduced, *phase_partials] if phase_reduced is not None else phase_partials)
        )
        final_phase.write_parquet(phase_output, compression="zstd", statistics=True, row_group_size=100_000)
    elapsed = time.monotonic() - started
    stats = IntegrationStats(
        raw_records=raw_records,
        kept_records=kept_records,
        streams_processed=streams_processed,
        elapsed_seconds=elapsed,
    )
    append_propagation_entry(
        output.with_suffix(".propagation.jsonl"),
        build_entry(
            module="phase1_spike_event_integrator",
            operation="write_streaming_prspk001_parquet",
            inputs={"snr_mask_parquet": snr_mask_parquet},
            parameters={
                "decoder": "PRSPK001",
                "schema_version": "prism.spike_events_snr_masked.v5",
                "snr_multiplier": 3.0,
                "causal_anchor_multiplier": 5.0,
                "snr_mask_channel": 0,
                "protocol_state_summary": protocol_state_summary,
                "thermal_phase_binning": {
                    "phase_cold_hold": "true_md_step <= cold_hold_end",
                    "phase_ramp_up": "true_md_step > cold_hold_end and true_md_step <= ramp_end",
                    "phase_warm_hold": "true_md_step > ramp_end and true_md_step <= warm_hold_end",
                    "phase_ramp_down": "true_md_step > warm_hold_end and true_md_step <= ramp_down_end",
                    "phase_cold_return": "true_md_step > ramp_down_end",
                },
                "true_md_step": "absolute_timestep * graph_capture_multiplier",
                "hysteresis_delta": "spikes_cold_return - spikes_cold_hold",
                "first_break_time_ps": "first spike in phase_ramp_up or phase_warm_hold",
                "start_record": start_record,
                "max_records_per_stream": max_records_per_stream,
                "chunk_records": chunk_records,
                "selected_streams": len(selected),
                "raw_bytes_declared": raw_bytes,
                "streaming_binary_reader": True,
            },
            output_value={
                "output_path": output,
                "phase_residue_output_path": phase_output,
                "row_count": kept_records,
                "raw_record_count": raw_records,
                "elapsed_seconds": elapsed,
            },
            output_uncertainty=None,
            gate_status={
                "bounded_default_removed": max_records_per_stream is None,
                "all_selected_streams_processed": streams_processed == len(selected),
                "snr_mask_applied_before_write": True,
                "protocol_phase_boundaries_applied": True,
                "five_phase_hysteresis_aggregate_written": phase_output.exists(),
                "streaming_pyarrow_writer": True,
                "append_only_ledger": True,
            },
            repo_root=repo_root,
        ),
        repo_root=repo_root,
    )
    return stats


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", type=Path, default=DEFAULT_RAW_ROOT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--condition-id")
    parser.add_argument("--replica-id", type=int)
    parser.add_argument("--streams")
    parser.add_argument("--max-streams", type=int)
    parser.add_argument("--start-record", type=int, default=0)
    parser.add_argument("--max-records-per-stream", type=int)
    parser.add_argument("--chunk-records", type=int, default=1_000_000)
    parser.add_argument("--snr-mask-parquet", type=Path, default=DEFAULT_OUTPUT_DIR / "stream_snr_masks.parquet")
    parser.add_argument(
        "--protocol-state-summary",
        type=Path,
        default=DEFAULT_OUTPUT_DIR / "protocol_state_summary.parquet",
    )
    args = parser.parse_args()

    selected = filter_streams(
        discover_stream_files(args.raw_root, "spikes.bin"),
        condition_id=args.condition_id,
        replica_id=args.replica_id,
        stream_ids=parse_stream_selector(args.streams),
        max_streams=args.max_streams,
    )
    output = args.out_dir / "spike_events_snr_masked.parquet"
    stats = write_full_spike_parquet(
        selected=selected,
        output=output,
        snr_mask_parquet=args.snr_mask_parquet,
        protocol_state_summary=args.protocol_state_summary,
        start_record=args.start_record,
        max_records_per_stream=args.max_records_per_stream,
        chunk_records=args.chunk_records,
    )
    emit(
        f"WROTE {output} rows={stats.kept_records} raw_records={stats.raw_records} "
        f"streams={stats.streams_processed} elapsed_seconds={stats.elapsed_seconds:.3f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
