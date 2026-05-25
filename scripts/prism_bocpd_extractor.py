#!/usr/bin/env python3
"""Derive physical BOCPD survival regimes from chunk posterior streams."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import polars as pl

from prism_dstw.ontology import Picosecond, RunLength, StreamId, TimeStep

from scripts.prism_n80_extraction_common import (
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
    json_value,
    parse_stream_selector,
    read_json_object,
    replica_json_for,
    write_n80_parquet,
)


@dataclass(frozen=True)
class GhostTimeRow:
    condition_id: str
    replica_id: int
    stream_id: StreamId
    dt_ps: Picosecond
    timestep_min: TimeStep
    timestep_max: TimeStep


@dataclass(frozen=True)
class ProtocolScheduleRow:
    condition_id: str
    replica_id: int
    stream_id: StreamId
    start_temp_K: float
    end_temp_K: float
    cold_hold_end: TimeStep
    ramp_end: TimeStep
    warm_hold_end: TimeStep
    ramp_down_end: TimeStep


@dataclass(frozen=True)
class BocpdRegimeBoundary:
    frame_idx: TimeStep
    map_run_length: RunLength
    dt_ps: Picosecond
    survival_time_ps: Picosecond


def ghost_rows_for_replica(sample: StreamFile) -> list[GhostTimeRow]:
    data = read_json_object(replica_json_for(sample, "ghost_time_map.json"))
    streams = json_list(data.get("streams"))
    if not streams:
        raise ValueError(f"{sample.path.parent}: ghost_time_map streams missing")
    rows: list[GhostTimeRow] = []
    for row in streams:
        stream_object = json_object(row)
        if not stream_object:
            continue
        rows.append(
            GhostTimeRow(
                condition_id=sample.condition_id,
                replica_id=sample.replica_id,
                stream_id=StreamId(json_int(stream_object.get("stream_id"), -1)),
                dt_ps=Picosecond(json_float(stream_object.get("dt_fs"), 4.0) / 1000.0),
                timestep_min=TimeStep(json_int(stream_object.get("timestep_min"))),
                timestep_max=TimeStep(json_int(stream_object.get("timestep_max"))),
            )
        )
    return rows


def protocol_row_for_stream(item: StreamFile) -> ProtocolScheduleRow:
    data = read_json_object(
        item.path.parent / f"{item.condition_id}_stream{item.stream_id}_protocol_state.json"
    )
    return ProtocolScheduleRow(
        condition_id=item.condition_id,
        replica_id=item.replica_id,
        stream_id=StreamId(item.stream_id),
        start_temp_K=json_float(data.get("start_temp_K")),
        end_temp_K=json_float(data.get("end_temp_K")),
        cold_hold_end=TimeStep(json_int(data.get("cold_hold_end"))),
        ramp_end=TimeStep(json_int(data.get("ramp_end"))),
        warm_hold_end=TimeStep(json_int(data.get("warm_hold_end"))),
        ramp_down_end=TimeStep(json_int(data.get("ramp_down_end"))),
    )


def metadata_frames(selected: list[StreamFile]) -> tuple[pl.LazyFrame, pl.LazyFrame, list[Path]]:
    replica_seen: set[tuple[str, int]] = set()
    ghost_rows: list[GhostTimeRow] = []
    protocol_rows: list[ProtocolScheduleRow] = []
    raw_inputs: list[Path] = []
    for item in selected:
        protocol_rows.append(protocol_row_for_stream(item))
        raw_inputs.append(item.path.parent / f"{item.condition_id}_stream{item.stream_id}_protocol_state.json")
        key = (item.condition_id, item.replica_id)
        if key in replica_seen:
            continue
        replica_seen.add(key)
        ghost_rows.extend(ghost_rows_for_replica(item))
        raw_inputs.append(replica_json_for(item, "ghost_time_map.json"))
    ghost = pl.DataFrame(
        {
            "condition_id": [row.condition_id for row in ghost_rows],
            "replica_id": [row.replica_id for row in ghost_rows],
            "stream_id": [int(row.stream_id) for row in ghost_rows],
            "dt_ps": [float(row.dt_ps) for row in ghost_rows],
            "timestep_min": [int(row.timestep_min) for row in ghost_rows],
            "timestep_max": [int(row.timestep_max) for row in ghost_rows],
        }
    ).lazy()
    protocol = pl.DataFrame(
        {
            "condition_id": [row.condition_id for row in protocol_rows],
            "replica_id": [row.replica_id for row in protocol_rows],
            "stream_id": [int(row.stream_id) for row in protocol_rows],
            "start_temp_K": [row.start_temp_K for row in protocol_rows],
            "end_temp_K": [row.end_temp_K for row in protocol_rows],
            "cold_hold_end": [int(row.cold_hold_end) for row in protocol_rows],
            "ramp_end": [int(row.ramp_end) for row in protocol_rows],
            "warm_hold_end": [int(row.warm_hold_end) for row in protocol_rows],
            "ramp_down_end": [int(row.ramp_down_end) for row in protocol_rows],
        }
    ).lazy()
    return ghost, protocol, raw_inputs


def bocpd_lazy_frame(selected: list[StreamFile]) -> pl.LazyFrame:
    scans: list[pl.LazyFrame] = []
    for item in selected:
        scans.append(
            pl.scan_ndjson(item.path).with_columns(
                pl.lit(item.condition_id).alias("condition_id"),
                pl.lit(item.replica_id).cast(pl.UInt8).alias("replica_id"),
                pl.lit(item.stream_id).cast(pl.UInt8).alias("stream_id"),
            )
        )
    if not scans:
        raise ValueError("no BOCPD JSONL files selected")
    return pl.concat(scans, how="vertical")


def survival_time_ps(map_run_length: RunLength, dt_ps: Picosecond) -> Picosecond:
    return Picosecond(float(map_run_length) * float(dt_ps))


def dt_ps_for_stream(item: StreamFile) -> Picosecond:
    for row in ghost_rows_for_replica(item):
        if row.stream_id == item.stream_id:
            return row.dt_ps
    raise ValueError(f"{item.path.parent}: ghost_time_map missing stream {item.stream_id}")


def first_regime_boundary(item: StreamFile, dt_ps: Picosecond) -> BocpdRegimeBoundary:
    with item.path.open("r", encoding="utf-8") as handle:
        for line in handle:
            loaded: object = json.loads(line)
            record = json_object(json_value(loaded))
            if record and bool(record.get("chunk_close_signal")):
                run_length = RunLength(json_int(record.get("map_run_length")))
                return BocpdRegimeBoundary(
                    frame_idx=TimeStep(json_int(record.get("frame_idx"))),
                    map_run_length=run_length,
                    dt_ps=dt_ps,
                    survival_time_ps=survival_time_ps(run_length, dt_ps),
                )
    raise ValueError(f"{item.path}: no BOCPD regime boundary rows found")


def build_survival_frame(selected: list[StreamFile]) -> tuple[pl.LazyFrame, list[Path]]:
    ghost, protocol, metadata_inputs = metadata_frames(selected)
    boundary_samples = [
        first_regime_boundary(item, dt_ps_for_stream(item))
        for item in selected
    ]
    if any(float(sample.survival_time_ps) < 0.0 for sample in boundary_samples):
        raise ValueError("BOCPD survival time cannot be negative")
    base = bocpd_lazy_frame(selected)
    joined = (
        base.join(ghost, on=("condition_id", "replica_id", "stream_id"), how="left")
        .join(protocol, on=("condition_id", "replica_id", "stream_id"), how="left")
        .with_columns(
            (pl.col("frame_idx").cast(pl.Float64) * pl.col("dt_ps")).alias("time_ps"),
            (pl.col("map_run_length").cast(pl.Float64) * pl.col("dt_ps")).alias("survival_time_ps"),
            pl.when(pl.col("frame_idx") <= pl.col("cold_hold_end"))
            .then(pl.lit("Cold_Hold"))
            .when(pl.col("frame_idx") <= pl.col("ramp_end"))
            .then(pl.lit("Ramp_Up"))
            .when(pl.col("frame_idx") <= pl.col("warm_hold_end"))
            .then(pl.lit("Warm_Hold"))
            .when(pl.col("frame_idx") <= pl.col("ramp_down_end"))
            .then(pl.lit("Ramp_Down"))
            .otherwise(pl.lit("Cold_Return"))
            .alias("thermal_phase"),
            pl.when(pl.col("frame_idx") <= pl.col("cold_hold_end"))
            .then(pl.col("start_temp_K"))
            .when(pl.col("frame_idx") <= pl.col("ramp_end"))
            .then(
                pl.col("start_temp_K")
                + (pl.col("end_temp_K") - pl.col("start_temp_K"))
                * (pl.col("frame_idx") - pl.col("cold_hold_end"))
                / (pl.col("ramp_end") - pl.col("cold_hold_end"))
            )
            .when(pl.col("frame_idx") <= pl.col("warm_hold_end"))
            .then(pl.col("end_temp_K"))
            .when(pl.col("frame_idx") <= pl.col("ramp_down_end"))
            .then(
                pl.col("end_temp_K")
                - (pl.col("end_temp_K") - pl.col("start_temp_K"))
                * (pl.col("frame_idx") - pl.col("warm_hold_end"))
                / (pl.col("ramp_down_end") - pl.col("warm_hold_end"))
            )
            .otherwise(pl.col("start_temp_K"))
            .alias("temperature_K"),
        )
        .with_columns(
            pl.col("chunk_close_signal").cast(pl.Boolean),
            pl.col("chunk_close_signal").cast(pl.UInt32).cum_sum().over(
                "condition_id", "replica_id", "stream_id"
            ).alias("regime_id"),
        )
        .filter(pl.col("chunk_close_signal"))
    )
    return joined, [item.path for item in selected] + metadata_inputs


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", type=Path, default=DEFAULT_RAW_ROOT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--condition-id")
    parser.add_argument("--replica-id", type=int)
    parser.add_argument("--streams")
    parser.add_argument("--max-streams", type=int)
    args = parser.parse_args()

    selected = filter_streams(
        discover_stream_files(args.raw_root, "bocpd.jsonl"),
        condition_id=args.condition_id,
        replica_id=args.replica_id,
        stream_ids=parse_stream_selector(args.streams),
        max_streams=args.max_streams,
    )
    lazy, raw_inputs = build_survival_frame(selected)
    output = args.out_dir / "bocpd_survival_regimes.parquet"
    write_n80_parquet(
        lazy,
        output,
        producer_script=Path(__file__),
        pipeline_stage="phase2_bocpd_survival_extractor",
        schema_version="prism.bocpd_survival_regimes.v1",
        partition_keys=("condition_id", "replica_id", "stream_id", "chunk_idx"),
        raw_inputs=raw_inputs,
        ledger_parameters={"survival_equation": "survival_time_ps = map_run_length * dt_ps"},
        row_count=None,
    )
    emit(f"WROTE {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
