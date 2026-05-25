#!/usr/bin/env python3
"""Generate Phase 2C high-resolution snapshot trigger windows from mined tensors."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import TypeAlias, cast

import polars as pl


REPO_ROOT = Path(__file__).resolve().parents[1]
CAMPAIGN_ID = "glp1r_aleniglipron"
N80_DIR = REPO_ROOT / "campaigns/glp1r_aleniglipron/integrated_spike_events/n80_full_scale"
DEFAULT_TEMPORAL_CASCADE = N80_DIR / "temporal_cascade.parquet"
DEFAULT_PROTOCOL_STATE = N80_DIR / "protocol_state_summary.parquet"
DEFAULT_STREAM_COUNTS = N80_DIR / "stream_level_phase_counts.parquet"
DEFAULT_EDGE_VALIDATION = N80_DIR / "phase_manifold_edge_validation.parquet"
DEFAULT_OUTPUT = REPO_ROOT / "campaigns/glp1r_aleniglipron/phase_2c_snapshot_triggers.json"
TARGET_CONDITIONS = ("glp1r_6XOX_WT", "glp1r_6LN2_A316T")
EQUILIBRIUM_GROUP_ALIASES = ("B_Equilibrium", "C_Equilibrium")
HYSTERESIS_GROUP = "D_Hysteresis"
DOWNSTREAM_LOCK_EDGE = "downstream_lock"
RUPTURE_PRE_STEPS = 10
RUPTURE_POST_STEPS = 5
STREAMS_PER_GROUP = 2

JsonObject: TypeAlias = dict[str, object]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--temporal-cascade", type=Path, default=DEFAULT_TEMPORAL_CASCADE)
    parser.add_argument("--protocol-state-summary", type=Path, default=DEFAULT_PROTOCOL_STATE)
    parser.add_argument("--stream-level-phase-counts", type=Path, default=DEFAULT_STREAM_COUNTS)
    parser.add_argument("--edge-validation", type=Path, default=DEFAULT_EDGE_VALIDATION)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def emit(message: str) -> None:
    sys.stdout.write(message + "\n")


def require_file(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(path)


def json_dumps(payload: JsonObject) -> str:
    return json.dumps(payload, indent=2, sort_keys=False) + "\n"


def as_int(value: object, label: str) -> int:
    if isinstance(value, bool):
        raise TypeError(f"{label} must be an integer, got bool")
    if isinstance(value, int | float | str):
        return int(value)
    raise TypeError(f"{label} must be an integer")


def as_float(value: object, label: str) -> float:
    if isinstance(value, bool):
        raise TypeError(f"{label} must be numeric, got bool")
    if isinstance(value, int | float | str):
        return float(value)
    raise TypeError(f"{label} must be numeric")


def as_str(value: object, label: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{label} must be a string")
    return value


def group_names(path: Path) -> set[str]:
    groups = pl.scan_parquet(path).select(pl.col("protocol_group").unique()).collect().to_series().to_list()
    return {str(item) for item in groups}


def resolved_equilibrium_group(stream_counts: Path) -> str:
    available = group_names(stream_counts)
    for group in EQUILIBRIUM_GROUP_ALIASES:
        if group in available:
            return group
    raise ValueError(f"no equilibrium protocol group found in {stream_counts}: {sorted(available)}")


def selected_streams(stream_counts: Path, protocol_state: Path) -> list[dict[str, object]]:
    equilibrium_group = resolved_equilibrium_group(stream_counts)
    target_groups = [equilibrium_group, HYSTERESIS_GROUP]
    stream_stats = (
        pl.scan_parquet(stream_counts)
        .filter(pl.col("condition_id").is_in(TARGET_CONDITIONS))
        .filter(pl.col("protocol_group").is_in(target_groups))
        .group_by(["condition_id", "protocol_group", "stream_id"])
        .agg(
            [
                pl.col("spike_count").sum().alias("stream_spikes"),
                pl.col("primary_residue_idx").n_unique().alias("active_residues"),
                pl.col("thermal_phase").n_unique().alias("active_phases"),
            ]
        )
        .with_columns(pl.col("stream_spikes").median().over(["condition_id", "protocol_group"]).alias("median_spikes"))
        .with_columns(
            (
                (pl.col("stream_spikes").cast(pl.Float64) - pl.col("median_spikes")).abs()
                / pl.max_horizontal(pl.col("median_spikes"), pl.lit(1.0))
            ).alias("representativeness_score")
        )
        .sort(
            ["condition_id", "protocol_group", "representativeness_score", "active_phases", "active_residues", "stream_id"],
            descending=[False, False, False, True, True, False],
        )
        .group_by(["condition_id", "protocol_group"], maintain_order=True)
        .head(STREAMS_PER_GROUP)
    )
    protocol_rows = (
        pl.scan_parquet(protocol_state)
        .filter(pl.col("condition_id").is_in(TARGET_CONDITIONS))
        .join(stream_stats, on=["condition_id", "stream_id"], how="inner")
        .with_columns(
            pl.col("graph_capture_multiplier")
            .median()
            .over(["condition_id", "protocol_group", "stream_id"])
            .alias("median_capture_multiplier")
        )
        .with_columns(
            (pl.col("graph_capture_multiplier") - pl.col("median_capture_multiplier"))
            .abs()
            .alias("replica_representativeness_score")
        )
        .sort(
            [
                "condition_id",
                "protocol_group",
                "stream_id",
                "replica_representativeness_score",
                "replica_id",
            ],
            descending=[False, False, False, False, False],
        )
        .group_by(["condition_id", "protocol_group", "stream_id"], maintain_order=True)
        .head(1)
        .sort(["condition_id", "protocol_group", "representativeness_score", "stream_id"])
        .collect()
    )
    return cast(list[dict[str, object]], protocol_rows.to_dicts())


def downstream_lock_edges(edge_validation: Path) -> dict[str, dict[str, object]]:
    edges = (
        pl.scan_parquet(edge_validation)
        .filter(pl.col("condition_id").is_in(TARGET_CONDITIONS))
        .filter(pl.col("edge_class") == DOWNSTREAM_LOCK_EDGE)
        .with_columns(
            pl.when(pl.col("edge_label") == "ARG421 -> TRP417")
            .then(pl.lit(0))
            .otherwise(pl.lit(1))
            .alias("edge_priority")
        )
        .sort(
            ["condition_id", "edge_priority", "durability_risk_score_raw", "edge_coherence_score"],
            descending=[False, False, True, True],
        )
        .group_by("condition_id", maintain_order=True)
        .head(1)
        .collect()
    )
    rows = cast(list[dict[str, object]], edges.to_dicts())
    return {as_str(row["condition_id"], "condition_id"): row for row in rows}


def rupture_steps(temporal_cascade: Path, stream_counts: Path, edge_validation: Path) -> dict[tuple[str, str], int]:
    edges = downstream_lock_edges(edge_validation)
    records: list[dict[str, object]] = []
    for condition_id in TARGET_CONDITIONS:
        edge = edges.get(condition_id)
        if edge is None:
            continue
        from_residue = as_int(edge["edge_from_residue"], "edge_from_residue")
        to_residue = as_int(edge["edge_to_residue"], "edge_to_residue")
        records.append(
            {
                "condition_id": condition_id,
                "edge_label": as_str(edge["edge_label"], "edge_label"),
                "edge_from_residue": from_residue,
                "edge_to_residue": to_residue,
            }
        )
    if not records:
        raise ValueError("no downstream lock edges available for target conditions")
    edge_lf = pl.DataFrame(records).lazy()
    cascade = (
        pl.scan_parquet(temporal_cascade)
        .filter(pl.col("condition_id").is_in(TARGET_CONDITIONS))
        .filter(pl.col("protocol_group").is_in([resolved_equilibrium_group(stream_counts), HYSTERESIS_GROUP]))
        .join(edge_lf, on="condition_id", how="inner")
        .filter(
            (pl.col("primary_residue_idx") == pl.col("edge_from_residue"))
            | (pl.col("primary_residue_idx") == pl.col("edge_to_residue"))
        )
        .group_by(["condition_id", "protocol_group", "edge_label"])
        .agg(
            [
                pl.col("first_ramp_md_step").max().alias("rupture_step_float"),
                pl.col("primary_residue_idx").n_unique().alias("observed_edge_residue_count"),
            ]
        )
        .with_columns(pl.col("rupture_step_float").round(0).cast(pl.Int32).alias("rupture_step"))
        .collect()
    )
    rows = cast(list[dict[str, object]], cascade.to_dicts())
    steps: dict[tuple[str, str], int] = {}
    for row in rows:
        steps[(as_str(row["condition_id"], "condition_id"), as_str(row["protocol_group"], "protocol_group"))] = as_int(
            row["rupture_step"], "rupture_step"
        )
    return steps


def single_step_window(step: int, rationale: str) -> JsonObject:
    return {
        "start_step": step,
        "end_step": step,
        "rationale": rationale,
    }


def bounded_window(start_step: int, end_step: int, total_steps: int, rationale: str) -> JsonObject:
    return {
        "start_step": max(0, min(start_step, total_steps)),
        "end_step": max(0, min(end_step, total_steps)),
        "rationale": rationale,
    }


def trigger_row(row: dict[str, object], rupture_step: int) -> JsonObject:
    condition_id = as_str(row["condition_id"], "condition_id")
    protocol_group = as_str(row["protocol_group"], "protocol_group")
    cold_hold_end = as_int(row["cold_hold_end"], "cold_hold_end")
    warm_hold_end = as_int(row["warm_hold_end"], "warm_hold_end")
    total_steps = as_int(row["total_steps"], "total_steps")
    windows = [
        single_step_window(cold_hold_end, "Apo Baseline (Cold Hold End)"),
        bounded_window(
            rupture_step - RUPTURE_PRE_STEPS,
            rupture_step + RUPTURE_POST_STEPS,
            total_steps,
            f"High-Resolution Rupture Window ({protocol_group} downstream-lock first-ramp capture)",
        ),
        single_step_window(warm_hold_end, "Desensitized State (Warm Hold End)"),
        single_step_window(total_steps, "Hysteresis State (Cold Return End)"),
    ]
    return {
        "condition_id": condition_id,
        "replica_id": as_int(row["replica_id"], "replica_id"),
        "stream_id": as_int(row["stream_id"], "stream_id"),
        "windows": windows,
    }


def build_manifest(args: argparse.Namespace) -> JsonObject:
    for path in (args.temporal_cascade, args.protocol_state_summary, args.stream_level_phase_counts, args.edge_validation):
        require_file(path)
    rows = selected_streams(args.stream_level_phase_counts, args.protocol_state_summary)
    steps = rupture_steps(args.temporal_cascade, args.stream_level_phase_counts, args.edge_validation)
    triggers: list[JsonObject] = []
    for row in rows:
        condition_id = as_str(row["condition_id"], "condition_id")
        protocol_group = as_str(row["protocol_group"], "protocol_group")
        rupture_step = steps.get((condition_id, protocol_group))
        if rupture_step is None:
            continue
        triggers.append(trigger_row(row, rupture_step))
    if not triggers:
        raise ValueError("no snapshot triggers could be generated")
    return {
        "campaign_id": CAMPAIGN_ID,
        "target_conditions": list(TARGET_CONDITIONS),
        "capture_mode": "targeted_window_reintegration",
        "stride": 1,
        "triggers": triggers,
    }


def main() -> int:
    args = parse_args()
    manifest = build_manifest(args)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json_dumps(manifest), encoding="utf-8")
    triggers = cast(list[JsonObject], manifest["triggers"])
    trigger_count = len(triggers)
    window_count = sum(len(cast(list[JsonObject], trigger["windows"])) for trigger in triggers)
    emit(f"wrote {output.relative_to(REPO_ROOT)} triggers={trigger_count} windows={window_count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
