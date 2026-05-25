#!/usr/bin/env python3
"""Extract autonomous steering focus tensors from protocol_state sidecars."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import polars as pl

from prism_dstw.ontology import ResidueIdx, StreamId, TimeStep

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
    parse_stream_selector,
    read_json_object,
    write_n80_parquet,
)


@dataclass(frozen=True)
class SteeringRow:
    condition_id: str
    replica_id: int
    stream_id: StreamId
    residue_idx: ResidueIdx
    steering_weight: float
    thermal_phase: str
    current_step: TimeStep
    current_temperature_K: float


@dataclass(frozen=True)
class ProtocolSummaryRow:
    condition_id: str
    replica_id: int
    stream_id: StreamId
    thermal_phase: str
    thermal_class: str
    current_step: TimeStep
    current_temperature_K: float
    start_temp_K: float
    end_temp_K: float
    cold_hold_end: TimeStep
    ramp_end: TimeStep
    warm_hold_end: TimeStep
    ramp_down_end: TimeStep
    total_steps: TimeStep
    observed_timestep_max: TimeStep
    graph_capture_multiplier: float


def thermal_phase_for_step(
    step: TimeStep,
    *,
    cold_hold_end: TimeStep,
    ramp_end: TimeStep,
    warm_hold_end: TimeStep,
    ramp_down_end: TimeStep,
) -> str:
    step_i = int(step)
    if step_i <= int(cold_hold_end):
        return "Cold_Hold"
    if step_i <= int(ramp_end):
        return "Ramp_Up"
    if step_i <= int(warm_hold_end):
        return "Warm_Hold"
    if step_i <= int(ramp_down_end):
        return "Ramp_Down"
    return "Cold_Return"


def thermal_class_for_temperature(
    *,
    current_temperature_K: float,
    start_temp_K: float,
    end_temp_K: float,
) -> str:
    midpoint = (start_temp_K + end_temp_K) / 2.0
    return "Cold_Phase" if current_temperature_K < midpoint else "Warm_Phase"


def observed_timestep_max_for_file(item: StreamFile, default: TimeStep) -> TimeStep:
    ghost_path = item.path.parent / "ghost_time_map.json"
    if not ghost_path.exists():
        return default
    data = read_json_object(ghost_path)
    for row in json_list(data.get("streams")):
        stream_object = json_object(row)
        if stream_object and json_int(stream_object.get("stream_id"), -1) == int(item.stream_id):
            value = json_int(stream_object.get("timestep_max"), json_int(stream_object.get("frame_idx_at_teardown")))
            return TimeStep(value if value > 0 else int(default))
    return default


def summary_for_file(item: StreamFile) -> ProtocolSummaryRow:
    data = read_json_object(item.path)
    current_step = TimeStep(json_int(data.get("current_step")))
    total_steps = TimeStep(json_int(data.get("total_steps"), int(current_step)))
    observed_timestep_max = observed_timestep_max_for_file(item, current_step)
    graph_capture_multiplier = float(total_steps) / float(observed_timestep_max) if int(observed_timestep_max) else 1.0
    start_temp = json_float(data.get("start_temp_K"))
    end_temp = json_float(data.get("end_temp_K"))
    current_temp = json_float(data.get("current_temperature_K"))
    cold_hold_end = TimeStep(json_int(data.get("cold_hold_end")))
    ramp_end = TimeStep(json_int(data.get("ramp_end")))
    warm_hold_end = TimeStep(json_int(data.get("warm_hold_end")))
    ramp_down_end = TimeStep(json_int(data.get("ramp_down_end")))
    phase = thermal_phase_for_step(
        current_step,
        cold_hold_end=cold_hold_end,
        ramp_end=ramp_end,
        warm_hold_end=warm_hold_end,
        ramp_down_end=ramp_down_end,
    )
    return ProtocolSummaryRow(
        condition_id=item.condition_id,
        replica_id=item.replica_id,
        stream_id=StreamId(item.stream_id),
        thermal_phase=phase,
        thermal_class=thermal_class_for_temperature(
            current_temperature_K=current_temp,
            start_temp_K=start_temp,
            end_temp_K=end_temp,
        ),
        current_step=current_step,
        current_temperature_K=current_temp,
        start_temp_K=start_temp,
        end_temp_K=end_temp,
        cold_hold_end=cold_hold_end,
        ramp_end=ramp_end,
        warm_hold_end=warm_hold_end,
        ramp_down_end=ramp_down_end,
        total_steps=total_steps,
        observed_timestep_max=observed_timestep_max,
        graph_capture_multiplier=graph_capture_multiplier,
    )


def rows_for_file(item: StreamFile) -> list[SteeringRow]:
    data = read_json_object(item.path)
    focus_rows = json_list(data.get("steering_focus_residues"))
    if not focus_rows:
        return []
    current_step = TimeStep(json_int(data.get("current_step")))
    phase = thermal_phase_for_step(
        current_step,
        cold_hold_end=TimeStep(json_int(data.get("cold_hold_end"))),
        ramp_end=TimeStep(json_int(data.get("ramp_end"))),
        warm_hold_end=TimeStep(json_int(data.get("warm_hold_end"))),
        ramp_down_end=TimeStep(json_int(data.get("ramp_down_end"))),
    )
    rows: list[SteeringRow] = []
    for focus in focus_rows:
        focus_object = json_object(focus)
        if not focus_object:
            continue
        residue_value = json_int(focus_object.get("residue_id"), -1)
        if residue_value < 0:
            continue
        rows.append(
            SteeringRow(
                condition_id=item.condition_id,
                replica_id=item.replica_id,
                stream_id=StreamId(item.stream_id),
                residue_idx=ResidueIdx(residue_value),
                steering_weight=json_float(focus_object.get("weight")),
                thermal_phase=phase,
                current_step=current_step,
                current_temperature_K=json_float(data.get("current_temperature_K")),
            )
        )
    return rows


def summary_from_rows(rows: list[ProtocolSummaryRow]) -> pl.LazyFrame:
    frame = pl.DataFrame(
        {
            "condition_id": [row.condition_id for row in rows],
            "replica_id": [row.replica_id for row in rows],
            "stream_id": [int(row.stream_id) for row in rows],
            "thermal_phase": [row.thermal_phase for row in rows],
            "thermal_class": [row.thermal_class for row in rows],
            "current_step": [int(row.current_step) for row in rows],
            "current_temperature_K": [row.current_temperature_K for row in rows],
            "start_temp_K": [row.start_temp_K for row in rows],
            "end_temp_K": [row.end_temp_K for row in rows],
            "cold_hold_end": [int(row.cold_hold_end) for row in rows],
            "ramp_end": [int(row.ramp_end) for row in rows],
            "warm_hold_end": [int(row.warm_hold_end) for row in rows],
            "ramp_down_end": [int(row.ramp_down_end) for row in rows],
            "total_steps": [int(row.total_steps) for row in rows],
            "observed_timestep_max": [int(row.observed_timestep_max) for row in rows],
            "graph_capture_multiplier": [row.graph_capture_multiplier for row in rows],
        }
    )
    return frame.lazy().with_columns(
        pl.col("replica_id").cast(pl.UInt8),
        pl.col("stream_id").cast(pl.UInt8),
        pl.col("current_step").cast(pl.Int32),
        pl.col("cold_hold_end").cast(pl.Int32),
        pl.col("ramp_end").cast(pl.Int32),
        pl.col("warm_hold_end").cast(pl.Int32),
        pl.col("ramp_down_end").cast(pl.Int32),
        pl.col("total_steps").cast(pl.Int32),
        pl.col("observed_timestep_max").cast(pl.Int32),
        pl.col("graph_capture_multiplier").cast(pl.Float64),
    )


def tensor_from_rows(rows: list[SteeringRow]) -> pl.LazyFrame:
    frame = pl.DataFrame(
        {
            "condition_id": [row.condition_id for row in rows],
            "replica_id": [row.replica_id for row in rows],
            "stream_id": [int(row.stream_id) for row in rows],
            "residue_idx": [int(row.residue_idx) for row in rows],
            "thermal_phase": [row.thermal_phase for row in rows],
            "steering_weight": [row.steering_weight for row in rows],
            "current_step": [int(row.current_step) for row in rows],
            "current_temperature_K": [row.current_temperature_K for row in rows],
        }
    )
    return (
        frame.lazy()
        .with_columns(
            pl.col("replica_id").cast(pl.UInt8),
            pl.col("stream_id").cast(pl.UInt8),
            pl.col("residue_idx").cast(pl.Int32),
            pl.col("current_step").cast(pl.Int32),
        )
        .group_by("condition_id", "residue_idx", "thermal_phase")
        .agg(
            pl.col("steering_weight").sum().alias("steering_weight_sum"),
            pl.col("steering_weight").mean().alias("steering_weight_mean"),
            pl.col("steering_weight").max().alias("steering_weight_max"),
            pl.len().alias("steering_focus_observations"),
            pl.col("stream_id").n_unique().alias("supporting_stream_count"),
            pl.col("replica_id").n_unique().alias("supporting_replica_count"),
            pl.col("current_temperature_K").mean().alias("mean_temperature_K"),
        )
    )


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
        discover_stream_files(args.raw_root, "protocol_state.json"),
        condition_id=args.condition_id,
        replica_id=args.replica_id,
        stream_ids=parse_stream_selector(args.streams),
        max_streams=args.max_streams,
    )
    rows: list[SteeringRow] = []
    summary_rows: list[ProtocolSummaryRow] = []
    for item in selected:
        summary_rows.append(summary_for_file(item))
        rows.extend(rows_for_file(item))
    lazy = tensor_from_rows(rows)
    output = args.out_dir / "autonomous_steering_tensor.parquet"
    write_n80_parquet(
        lazy,
        output,
        producer_script=Path(__file__),
        pipeline_stage="phase2_protocol_state_extractor",
        schema_version="prism.autonomous_steering_tensor.v1",
        partition_keys=("condition_id", "residue_idx", "thermal_phase"),
        raw_inputs=[item.path for item in selected],
        ledger_parameters={"aggregation": "sum_mean_max_by_residue_and_thermal_phase"},
        row_count=None,
    )
    emit(f"WROTE {output}")
    summary = summary_from_rows(summary_rows)
    summary_output = args.out_dir / "protocol_state_summary.parquet"
    write_n80_parquet(
        summary,
        summary_output,
        producer_script=Path(__file__),
        pipeline_stage="phase2_protocol_state_summary",
        schema_version="prism.protocol_state_summary.v3",
        partition_keys=("condition_id", "replica_id", "stream_id"),
        raw_inputs=[item.path for item in selected],
        ledger_parameters={
            "thermal_class_rule": "Cold_Phase if current_temperature_K < midpoint(start_temp_K,end_temp_K), else Warm_Phase",
            "schedule_boundaries": ["cold_hold_end", "ramp_end", "warm_hold_end", "ramp_down_end"],
            "graph_capture_multiplier": "total_steps / observed_timestep_max from ghost_time_map.json",
        },
        row_count=len(summary_rows),
    )
    emit(f"WROTE {summary_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
