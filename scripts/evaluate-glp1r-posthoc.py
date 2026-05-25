#!/usr/bin/env python3
"""Fuse n80 PRISM-DSTW tensors into an edge-level GLP-1R durability risk map."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Sequence, cast

import polars as pl
import pyarrow.parquet as pq


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from prism_n80_extraction_common import CAMPAIGN_ID, DEFAULT_OUTPUT_DIR, write_n80_parquet  # type: ignore[import-not-found]


DEFAULT_TOPOLOGY_DIR = Path("campaigns/glp1r_aleniglipron/inputs/topologies")
DEFAULT_TOPOLOGY_REGISTER = Path(
    "campaigns/glp1r_aleniglipron/integrated_spike_events/full/"
    "sar_steric_interface_catalog.parquet"
)
DEFAULT_PHASE_EDGE_VALIDATION = DEFAULT_OUTPUT_DIR / "phase_manifold_edge_validation.parquet"
ROW_GROUP_SIZE = 65_536
SPIKE_AGGREGATE_BATCH_SIZE = 1_000_000
LOGGER = logging.getLogger("evaluate_glp1r_posthoc")

STREAM_KEY = ["condition_id", "replica_id", "stream_id"]
EDGE_KEY = ["condition_id", "replica_id", "stream_id", "edge_from_residue", "edge_to_residue"]
EDGE_OUTPUT_KEY = ["condition_id", "edge_from_residue", "edge_to_residue", "edge_class"]
SPIKE_RESIDUE_AGGREGATE = "spike_residue_fields.parquet"


def read_atom_map(path: Path) -> list[int]:
    loaded = json.loads(path.read_text(encoding="utf-8"))
    values = loaded.get("atom_to_residue")
    if not isinstance(values, list):
        raise ValueError(f"{path} does not contain atom_to_residue list")
    return [int(item) for item in values]


def topology_for_condition(condition_id: str) -> str:
    return "5vex" if "5VEX" in condition_id else "6x1a"


def atom_residue_lf(mechanical_path: Path, topology_dir: Path) -> pl.LazyFrame:
    conditions = (
        pl.scan_parquet(mechanical_path)
        .select(pl.col("condition_id").unique().sort())
        .collect()
        .get_column("condition_id")
        .to_list()
    )
    maps = {
        "5vex": read_atom_map(topology_dir / "5vex_clean.atom_to_residue.json"),
        "6x1a": read_atom_map(topology_dir / "6x1a_clean.atom_to_residue.json"),
    }
    rows: list[dict[str, int | str]] = []
    for condition in conditions:
        values = maps[topology_for_condition(str(condition))]
        rows.extend(
            {
                "condition_id": str(condition),
                "atom_idx": atom_idx,
                "residue_idx": residue_idx,
            }
            for atom_idx, residue_idx in enumerate(values)
        )
    return pl.DataFrame(rows).lazy()


def condition_signal_lf(signal: pl.LazyFrame) -> pl.LazyFrame:
    return signal.group_by("condition_id").agg(
        [
            pl.len().alias("condition_voxel_count"),
            (pl.col("variance_class") == "thermally_destabilized")
            .cast(pl.Float64)
            .mean()
            .alias("condition_destabilized_voxel_fraction"),
            (pl.col("variance_class") == "thermally_activated")
            .cast(pl.Float64)
            .mean()
            .alias("condition_activated_voxel_fraction"),
            pl.col("hit_count_cold_mean").mean().alias("condition_cold_hit_mean"),
            pl.col("hit_count_warm_mean").mean().alias("condition_warm_hit_mean"),
        ]
    )


def edge_stream_lf(kcc: pl.LazyFrame, topology_register: Path) -> pl.LazyFrame:
    streams = kcc.select(STREAM_KEY).unique()
    edges = (
        pl.scan_parquet(topology_register)
        .select(
            [
                "condition_id",
                pl.col("edge_from_residue").cast(pl.UInt32),
                pl.col("edge_to_residue").cast(pl.UInt32),
                "edge_class",
                "minimum_heavy_atom_distance_angstrom",
                "contact_cutoff_angstrom",
            ]
        )
        .unique()
    )
    return streams.join(edges, on="condition_id", how="inner")


def residue_kcc_lf(kcc: pl.LazyFrame) -> pl.LazyFrame:
    return (
        kcc.with_columns(pl.col("residue_idx").cast(pl.UInt32))
        .group_by([*STREAM_KEY, "residue_idx"])
        .agg(
            [
                pl.col("temporal_corr").mean().alias("temporal_corr"),
                pl.col("direction_score").mean().alias("direction_score"),
                pl.col("motion_efficiency").mean().alias("motion_efficiency"),
                pl.col("lag_corr_peak").mean().alias("lag_corr_peak"),
                pl.col("local_cov").mean().alias("local_cov"),
                pl.col("active_causal").sum().alias("active_causal_sum"),
                pl.len().alias("kcc_observations"),
            ]
        )
        .with_columns(
            [
                pl.col("temporal_corr").fill_nan(0.0),
                pl.col("direction_score").fill_nan(0.0),
                pl.col("motion_efficiency").fill_nan(0.0),
                pl.col("lag_corr_peak").fill_nan(0.0),
                pl.col("local_cov").fill_nan(0.0),
            ]
        )
    )


def residue_spike_lf(
    spikes: pl.LazyFrame,
    signal: pl.LazyFrame,
    bocpd: pl.LazyFrame,
    kinetic: pl.LazyFrame,
) -> pl.LazyFrame:
    if "masked_spike_count" in spikes.collect_schema().names():
        return spikes

    signal_lookup = signal.select(["condition_id", "voxel_idx", "variance_class"])
    protocol_lookup = (
        bocpd.select([*STREAM_KEY, "cold_hold_end"])
        .group_by(STREAM_KEY)
        .agg(pl.col("cold_hold_end").min().alias("cold_hold_end"))
    )
    bocpd_lookup = (
        bocpd.select(
            [
                "condition_id",
                "replica_id",
                "stream_id",
                pl.col("chunk_idx").cast(pl.UInt32),
                "survival_time_ps",
                "temperature_K",
                "regime_id",
                "reset_probability",
            ]
        )
        .group_by([*STREAM_KEY, "chunk_idx"])
        .agg(
            [
                pl.col("survival_time_ps").mean().alias("stream_chunk_survival_time_ps"),
                pl.col("survival_time_ps").min().alias("stream_chunk_min_survival_time_ps"),
                pl.col("reset_probability").max().alias("stream_chunk_reset_probability"),
                pl.col("temperature_K").mean().alias("stream_chunk_temperature_K"),
                pl.col("regime_id").max().alias("stream_chunk_regime_id"),
            ]
        )
    )
    kinetic_lookup = (
        kinetic.select(
            [
                "condition_id",
                "replica_id",
                "stream_id",
                pl.col("chunk_idx").cast(pl.UInt32),
                "dt_ps",
                "dt_reduction_event",
                "dt_drop_coincident_with_regime_change",
            ]
        )
        .group_by([*STREAM_KEY, "chunk_idx"])
        .agg(
            [
                pl.col("dt_reduction_event").cast(pl.UInt32).sum().alias("dt_drop_count"),
                pl.col("dt_drop_coincident_with_regime_change")
                .cast(pl.UInt32)
                .sum()
                .alias("violent_dt_drop_count"),
                pl.col("dt_ps").min().alias("min_dt_ps"),
            ]
        )
    )
    enriched = (
        spikes.with_columns(
            [
                pl.col("voxel_idx").cast(pl.UInt64),
                pl.col("primary_residue_idx").cast(pl.UInt32).alias("residue_idx"),
                (pl.col("timestep") // pl.lit(500)).cast(pl.UInt32).alias("chunk_idx"),
            ]
        )
        .join(protocol_lookup, on=STREAM_KEY, how="left")
        .with_columns((pl.col("timestep") > pl.col("cold_hold_end")).fill_null(False).alias("post_cold_hold_spike"))
        .join(signal_lookup, on=["condition_id", "voxel_idx"], how="left")
        .join(bocpd_lookup, on=[*STREAM_KEY, "chunk_idx"], how="left")
        .join(kinetic_lookup, on=[*STREAM_KEY, "chunk_idx"], how="left")
    )
    return enriched.group_by([*STREAM_KEY, "residue_idx"]).agg(
        [
            pl.len().alias("masked_spike_count"),
            pl.col("causal_anchor").cast(pl.UInt32).sum().alias("causal_anchor_count"),
            pl.col("snr3_ratio").mean().alias("mean_snr3_ratio"),
            pl.col("snr3_ratio").max().alias("max_snr3_ratio"),
            pl.col("intensity").mean().alias("mean_spike_intensity"),
            pl.col("time_ps").filter(pl.col("post_cold_hold_spike")).min().alias("first_break_time_ps"),
            pl.col("time_ps").filter(pl.col("post_cold_hold_spike")).mean().alias("mean_break_time_ps"),
            (pl.col("variance_class") == "thermally_destabilized")
            .cast(pl.UInt32)
            .sum()
            .alias("thermally_destabilized_spike_count"),
            pl.col("post_cold_hold_spike")
            .cast(pl.UInt32)
            .sum()
            .alias("thermally_activated_spike_count"),
            (pl.col("variance_class") == "stable_occupied")
            .cast(pl.UInt32)
            .sum()
            .alias("stable_occupied_spike_count"),
            pl.col("stream_chunk_survival_time_ps").mean().alias("mean_survival_time_ps"),
            (pl.col("stream_chunk_survival_time_ps") <= 0.008)
            .cast(pl.UInt32)
            .sum()
            .alias("short_lived_regime_break_count"),
            pl.col("dt_drop_count").fill_null(0).sum().alias("dt_drop_count"),
            pl.col("violent_dt_drop_count").fill_null(0).sum().alias("violent_dt_drop_count"),
            pl.col("min_dt_ps").min().alias("min_dt_ps"),
        ]
    )


def spike_bocpd_lookup_df(bocpd_path: Path) -> pl.DataFrame:
    return (
        pl.scan_parquet(bocpd_path)
        .select(
            [
                "condition_id",
                "replica_id",
                "stream_id",
                pl.col("chunk_idx").cast(pl.UInt32),
                "survival_time_ps",
                "temperature_K",
                "regime_id",
                "reset_probability",
            ]
        )
        .group_by([*STREAM_KEY, "chunk_idx"])
        .agg(
            [
                pl.col("survival_time_ps").mean().alias("stream_chunk_survival_time_ps"),
                pl.col("survival_time_ps").min().alias("stream_chunk_min_survival_time_ps"),
                pl.col("reset_probability").max().alias("stream_chunk_reset_probability"),
                pl.col("temperature_K").mean().alias("stream_chunk_temperature_K"),
                pl.col("regime_id").max().alias("stream_chunk_regime_id"),
            ]
        )
        .collect()
    )


def spike_kinetic_lookup_df(kinetic_path: Path) -> pl.DataFrame:
    return (
        pl.scan_parquet(kinetic_path)
        .select(
            [
                "condition_id",
                "replica_id",
                "stream_id",
                pl.col("chunk_idx").cast(pl.UInt32),
                "dt_ps",
                "dt_reduction_event",
                "dt_drop_coincident_with_regime_change",
            ]
        )
        .group_by([*STREAM_KEY, "chunk_idx"])
        .agg(
            [
                pl.col("dt_reduction_event").cast(pl.UInt64).sum().alias("dt_drop_count"),
                pl.col("dt_drop_coincident_with_regime_change")
                .cast(pl.UInt64)
                .sum()
                .alias("violent_dt_drop_count"),
                pl.col("dt_ps").min().alias("min_dt_ps"),
            ]
        )
        .collect()
    )


def spike_protocol_lookup_df(bocpd_path: Path) -> pl.DataFrame:
    return (
        pl.scan_parquet(bocpd_path)
        .select([*STREAM_KEY, "cold_hold_end"])
        .group_by(STREAM_KEY)
        .agg(pl.col("cold_hold_end").min().alias("cold_hold_end"))
        .collect()
    )


def aggregate_spike_batch(
    batch: object,
    *,
    signal_lookup: pl.DataFrame,
    bocpd_lookup: pl.DataFrame,
    kinetic_lookup: pl.DataFrame,
    protocol_lookup: pl.DataFrame,
) -> pl.DataFrame:
    frame = cast(pl.DataFrame, pl.from_arrow(batch)).with_columns(
        [
            pl.col("voxel_idx").cast(pl.UInt64),
            pl.col("replica_id").cast(pl.UInt8),
            pl.col("stream_id").cast(pl.UInt8),
            pl.col("primary_residue_idx").cast(pl.UInt32).alias("residue_idx"),
            (pl.col("timestep") // pl.lit(500)).cast(pl.UInt32).alias("chunk_idx"),
        ]
    )
    enriched = (
        frame.join(protocol_lookup, on=STREAM_KEY, how="left")
        .with_columns((pl.col("timestep") > pl.col("cold_hold_end")).fill_null(False).alias("post_cold_hold_spike"))
        .join(signal_lookup, on=["condition_id", "voxel_idx"], how="left")
        .join(bocpd_lookup, on=[*STREAM_KEY, "chunk_idx"], how="left")
        .join(kinetic_lookup, on=[*STREAM_KEY, "chunk_idx"], how="left")
    )
    return enriched.group_by([*STREAM_KEY, "residue_idx"]).agg(
        [
            pl.len().cast(pl.UInt64).alias("masked_spike_count"),
            pl.col("causal_anchor").cast(pl.UInt64).sum().alias("causal_anchor_count"),
            pl.col("snr3_ratio").sum().alias("snr3_ratio_sum"),
            pl.col("snr3_ratio").max().alias("max_snr3_ratio"),
            pl.col("intensity").sum().alias("spike_intensity_sum"),
            pl.col("time_ps").filter(pl.col("post_cold_hold_spike")).min().alias("first_break_time_ps"),
            pl.col("time_ps").filter(pl.col("post_cold_hold_spike")).sum().alias("break_time_ps_sum"),
            (pl.col("variance_class") == "thermally_destabilized")
            .fill_null(False)
            .cast(pl.UInt64)
            .sum()
            .alias("thermally_destabilized_spike_count"),
            pl.col("post_cold_hold_spike")
            .cast(pl.UInt64)
            .sum()
            .alias("thermally_activated_spike_count"),
            (pl.col("variance_class") == "stable_occupied")
            .fill_null(False)
            .cast(pl.UInt64)
            .sum()
            .alias("stable_occupied_spike_count"),
            pl.col("stream_chunk_survival_time_ps").fill_null(0.0).sum().alias("survival_time_ps_sum"),
            pl.col("stream_chunk_survival_time_ps").is_not_null().cast(pl.UInt64).sum().alias("survival_time_ps_count"),
            (pl.col("stream_chunk_survival_time_ps") <= 0.008)
            .fill_null(False)
            .cast(pl.UInt64)
            .sum()
            .alias("short_lived_regime_break_count"),
            pl.col("dt_drop_count").fill_null(0).sum().alias("dt_drop_count"),
            pl.col("violent_dt_drop_count").fill_null(0).sum().alias("violent_dt_drop_count"),
            pl.col("min_dt_ps").min().alias("min_dt_ps"),
        ]
    )


def reduce_spike_partials(partials: Sequence[pl.DataFrame]) -> pl.DataFrame:
    combined = pl.concat(list(partials), how="vertical")
    return combined.group_by([*STREAM_KEY, "residue_idx"]).agg(
        [
            pl.col("masked_spike_count").sum().alias("masked_spike_count"),
            pl.col("causal_anchor_count").sum().alias("causal_anchor_count"),
            pl.col("snr3_ratio_sum").sum().alias("snr3_ratio_sum"),
            pl.col("max_snr3_ratio").max().alias("max_snr3_ratio"),
            pl.col("spike_intensity_sum").sum().alias("spike_intensity_sum"),
            pl.col("first_break_time_ps").min().alias("first_break_time_ps"),
            pl.col("break_time_ps_sum").sum().alias("break_time_ps_sum"),
            pl.col("thermally_destabilized_spike_count").sum().alias("thermally_destabilized_spike_count"),
            pl.col("thermally_activated_spike_count").sum().alias("thermally_activated_spike_count"),
            pl.col("stable_occupied_spike_count").sum().alias("stable_occupied_spike_count"),
            pl.col("survival_time_ps_sum").sum().alias("survival_time_ps_sum"),
            pl.col("survival_time_ps_count").sum().alias("survival_time_ps_count"),
            pl.col("short_lived_regime_break_count").sum().alias("short_lived_regime_break_count"),
            pl.col("dt_drop_count").sum().alias("dt_drop_count"),
            pl.col("violent_dt_drop_count").sum().alias("violent_dt_drop_count"),
            pl.col("min_dt_ps").min().alias("min_dt_ps"),
        ]
    )


def finalize_spike_residue_aggregate(frame: pl.DataFrame) -> pl.DataFrame:
    return frame.with_columns(
        [
            (pl.col("snr3_ratio_sum") / pl.col("masked_spike_count")).alias("mean_snr3_ratio"),
            (pl.col("spike_intensity_sum") / pl.col("masked_spike_count")).alias("mean_spike_intensity"),
            pl.when(pl.col("thermally_activated_spike_count") > 0)
            .then(pl.col("break_time_ps_sum") / pl.col("thermally_activated_spike_count"))
            .otherwise(None)
            .alias("mean_break_time_ps"),
            pl.when(pl.col("survival_time_ps_count") > 0)
            .then(pl.col("survival_time_ps_sum") / pl.col("survival_time_ps_count"))
            .otherwise(None)
            .alias("mean_survival_time_ps"),
        ]
    ).select(
        [
            *STREAM_KEY,
            "residue_idx",
            "masked_spike_count",
            "causal_anchor_count",
            "mean_snr3_ratio",
            "max_snr3_ratio",
            "mean_spike_intensity",
            "first_break_time_ps",
            "mean_break_time_ps",
            "thermally_destabilized_spike_count",
            "thermally_activated_spike_count",
            "stable_occupied_spike_count",
            "mean_survival_time_ps",
            "short_lived_regime_break_count",
            "dt_drop_count",
            "violent_dt_drop_count",
            "min_dt_ps",
        ]
    )


def ensure_spike_residue_aggregate(
    *,
    base_dir: Path,
    spike_path: Path,
    signal_path: Path,
    bocpd_path: Path,
    kinetic_path: Path,
) -> Path:
    output_path = base_dir / SPIKE_RESIDUE_AGGREGATE
    dependencies = [spike_path, signal_path, bocpd_path, kinetic_path]
    if output_path.exists() and output_path.stat().st_mtime >= max(path.stat().st_mtime for path in dependencies):
        return output_path

    LOGGER.info("building_spike_residue_aggregate path=%s", output_path)
    signal_lookup = (
        pl.scan_parquet(signal_path)
        .select(["condition_id", pl.col("voxel_idx").cast(pl.UInt64), "variance_class"])
        .unique()
        .collect()
    )
    bocpd_lookup = spike_bocpd_lookup_df(bocpd_path)
    kinetic_lookup = spike_kinetic_lookup_df(kinetic_path)
    protocol_lookup = spike_protocol_lookup_df(bocpd_path)

    partials: list[pl.DataFrame] = []
    reduced: pl.DataFrame | None = None
    rows_seen = 0
    next_report = 250_000_000
    parquet_file = pq.ParquetFile(spike_path)
    columns = [
        "condition_id",
        "replica_id",
        "stream_id",
        "timestep",
        "time_ps",
        "voxel_idx",
        "intensity",
        "primary_residue_idx",
        "causal_anchor",
        "snr3_ratio",
    ]
    for batch in parquet_file.iter_batches(batch_size=SPIKE_AGGREGATE_BATCH_SIZE, columns=columns):
        partials.append(
            aggregate_spike_batch(
                batch,
                signal_lookup=signal_lookup,
                bocpd_lookup=bocpd_lookup,
                kinetic_lookup=kinetic_lookup,
                protocol_lookup=protocol_lookup,
            )
        )
        rows_seen += batch.num_rows
        if len(partials) >= 128:
            reduced = reduce_spike_partials([reduced, *partials] if reduced is not None else partials)
            partials = []
        if rows_seen >= next_report:
            LOGGER.info("spike_aggregate_progress rows=%s", rows_seen)
            next_report += 250_000_000

    final_reduced = reduce_spike_partials([reduced, *partials] if reduced is not None else partials)
    final = finalize_spike_residue_aggregate(final_reduced)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_n80_parquet(
        final,
        output_path,
        producer_script=Path("scripts/evaluate-glp1r-posthoc.py"),
        pipeline_stage="posthoc_spike_residue_aggregate",
        schema_version="prism.spike_residue_fields.v2",
        partition_keys=STREAM_KEY,
        raw_inputs=[],
        source_parquets=[spike_path, signal_path, bocpd_path, kinetic_path],
        ledger_parameters={"row_group_size": ROW_GROUP_SIZE, "source_spike_rows": rows_seen},
        row_count=final.height,
    )
    LOGGER.info("wrote_spike_residue_aggregate path=%s rows=%s source_spike_rows=%s", output_path, final.height, rows_seen)
    return output_path


def residue_mechanical_lf(mechanical: pl.LazyFrame, atom_map: pl.LazyFrame) -> pl.LazyFrame:
    return (
        mechanical.join(atom_map, on=["condition_id", "atom_idx"], how="inner")
        .with_columns(pl.col("residue_idx").cast(pl.UInt32))
        .group_by([*STREAM_KEY, "residue_idx"])
        .agg(
            [
                pl.col("mechanical_load").abs().mean().alias("mean_abs_mechanical_load"),
                pl.col("mechanical_load").abs().max().alias("max_abs_mechanical_load"),
                (pl.col("mechanical_load").abs() > 1.0)
                .cast(pl.Float64)
                .mean()
                .alias("active_load_fraction"),
                pl.len().alias("mechanical_atom_observations"),
            ]
        )
    )


def residue_steering_lf(steering: pl.LazyFrame) -> pl.LazyFrame:
    return (
        steering.with_columns(pl.col("residue_idx").cast(pl.UInt32))
        .group_by(["condition_id", "residue_idx"])
        .agg(
            [
                pl.col("steering_weight_sum").sum().alias("steering_weight_sum"),
                pl.col("steering_weight_mean").mean().alias("steering_weight_mean"),
                pl.col("steering_weight_max").max().alias("steering_weight_max"),
                pl.col("steering_focus_observations").sum().alias("steering_focus_observations"),
            ]
        )
    )


def stream_survival_lf(bocpd: pl.LazyFrame) -> pl.LazyFrame:
    return bocpd.group_by(STREAM_KEY).agg(
        [
            pl.col("survival_time_ps").mean().alias("stream_mean_survival_time_ps"),
            pl.col("survival_time_ps").max().alias("stream_max_survival_time_ps"),
            (pl.col("survival_time_ps") <= 0.008)
            .cast(pl.Float64)
            .mean()
            .alias("stream_short_lived_regime_fraction"),
            pl.col("reset_probability").mean().alias("stream_reset_probability_mean"),
        ]
    )


def stream_kinetic_lf(kinetic: pl.LazyFrame) -> pl.LazyFrame:
    return kinetic.group_by(STREAM_KEY).agg(
        [
            pl.col("dt_reduction_event").cast(pl.UInt32).sum().alias("stream_dt_drop_count"),
            pl.col("dt_drop_coincident_with_regime_change")
            .cast(pl.UInt32)
            .sum()
            .alias("stream_violent_dt_drop_count"),
            pl.col("dt_ps").min().alias("stream_min_dt_ps"),
        ]
    )


def condition_aromatic_lf(aromatic: pl.LazyFrame) -> pl.LazyFrame:
    return aromatic.group_by("condition_id").agg(
        [
            pl.col("centroid_displacement_std").mean().alias("aromatic_mean_displacement_std"),
            pl.col("centroid_displacement_std").max().alias("aromatic_max_displacement_std"),
            pl.col("centroid_displacement_std").quantile(0.90).alias("aromatic_p90_displacement_std"),
        ]
    )


def side_lookup(frame: pl.LazyFrame, side: str, residue_column: str) -> pl.LazyFrame:
    return frame.rename(
        {
            name: f"{side}_{name}"
            for name in frame.collect_schema().names()
            if name not in STREAM_KEY and name != residue_column
        }
    ).rename({residue_column: f"edge_{side}_residue"})


def build_edge_stream_features(
    *,
    edge_stream: pl.LazyFrame,
    kcc: pl.LazyFrame,
    spikes: pl.LazyFrame,
    signal: pl.LazyFrame,
    mechanical: pl.LazyFrame,
    bocpd: pl.LazyFrame,
    kinetic: pl.LazyFrame,
    steering: pl.LazyFrame,
    aromatic: pl.LazyFrame,
    mechanical_path: Path,
    topology_dir: Path,
) -> pl.LazyFrame:
    kcc_residue = residue_kcc_lf(kcc)
    spike_residue = residue_spike_lf(spikes, signal, bocpd, kinetic)
    mechanical_residue = residue_mechanical_lf(
        mechanical,
        atom_residue_lf(mechanical_path, topology_dir),
    )
    steering_residue = residue_steering_lf(steering)

    joined = (
        edge_stream.join(side_lookup(kcc_residue, "from", "residue_idx"), on=[*STREAM_KEY, "edge_from_residue"], how="left")
        .join(side_lookup(kcc_residue, "to", "residue_idx"), on=[*STREAM_KEY, "edge_to_residue"], how="left")
        .join(side_lookup(spike_residue, "from", "residue_idx"), on=[*STREAM_KEY, "edge_from_residue"], how="left")
        .join(side_lookup(spike_residue, "to", "residue_idx"), on=[*STREAM_KEY, "edge_to_residue"], how="left")
        .join(side_lookup(mechanical_residue, "from", "residue_idx"), on=[*STREAM_KEY, "edge_from_residue"], how="left")
        .join(side_lookup(mechanical_residue, "to", "residue_idx"), on=[*STREAM_KEY, "edge_to_residue"], how="left")
        .join(side_lookup(steering_residue, "from", "residue_idx"), on=["condition_id", "edge_from_residue"], how="left")
        .join(side_lookup(steering_residue, "to", "residue_idx"), on=["condition_id", "edge_to_residue"], how="left")
        .join(stream_survival_lf(bocpd), on=STREAM_KEY, how="left")
        .join(stream_kinetic_lf(kinetic), on=STREAM_KEY, how="left")
        .join(condition_signal_lf(signal), on="condition_id", how="left")
        .join(condition_aromatic_lf(aromatic), on="condition_id", how="left")
    )
    numeric_fills = [
        "from_temporal_corr",
        "from_direction_score",
        "from_motion_efficiency",
        "from_lag_corr_peak",
        "from_local_cov",
        "from_active_causal_sum",
        "from_kcc_observations",
        "to_temporal_corr",
        "to_direction_score",
        "to_motion_efficiency",
        "to_lag_corr_peak",
        "to_local_cov",
        "to_active_causal_sum",
        "to_kcc_observations",
        "from_masked_spike_count",
        "from_causal_anchor_count",
        "from_mean_snr3_ratio",
        "from_max_snr3_ratio",
        "from_mean_spike_intensity",
        "from_thermally_destabilized_spike_count",
        "from_thermally_activated_spike_count",
        "from_short_lived_regime_break_count",
        "from_dt_drop_count",
        "from_violent_dt_drop_count",
        "to_masked_spike_count",
        "to_causal_anchor_count",
        "to_mean_snr3_ratio",
        "to_max_snr3_ratio",
        "to_mean_spike_intensity",
        "to_thermally_destabilized_spike_count",
        "to_thermally_activated_spike_count",
        "to_short_lived_regime_break_count",
        "to_dt_drop_count",
        "to_violent_dt_drop_count",
        "from_mean_abs_mechanical_load",
        "from_max_abs_mechanical_load",
        "from_active_load_fraction",
        "from_mechanical_atom_observations",
        "to_mean_abs_mechanical_load",
        "to_max_abs_mechanical_load",
        "to_active_load_fraction",
        "to_mechanical_atom_observations",
        "from_steering_weight_sum",
        "from_steering_weight_mean",
        "from_steering_weight_max",
        "from_steering_focus_observations",
        "to_steering_weight_sum",
        "to_steering_weight_mean",
        "to_steering_weight_max",
        "to_steering_focus_observations",
        "stream_dt_drop_count",
        "stream_violent_dt_drop_count",
        "stream_short_lived_regime_fraction",
        "condition_destabilized_voxel_fraction",
        "condition_activated_voxel_fraction",
        "aromatic_mean_displacement_std",
        "aromatic_max_displacement_std",
        "aromatic_p90_displacement_std",
    ]
    return joined.with_columns([pl.col(name).fill_null(0.0) for name in numeric_fills]).with_columns(
        [
            pl.col("from_first_break_time_ps").fill_null(pl.col("to_first_break_time_ps")).alias("from_first_break_time_ps"),
            pl.col("to_first_break_time_ps").fill_null(pl.col("from_first_break_time_ps")).alias("to_first_break_time_ps"),
            pl.col("from_mean_break_time_ps").fill_null(pl.col("to_mean_break_time_ps")).alias("from_mean_break_time_ps"),
            pl.col("to_mean_break_time_ps").fill_null(pl.col("from_mean_break_time_ps")).alias("to_mean_break_time_ps"),
            pl.col("from_mean_survival_time_ps").fill_null(pl.col("stream_mean_survival_time_ps")).alias("from_mean_survival_time_ps"),
            pl.col("to_mean_survival_time_ps").fill_null(pl.col("stream_mean_survival_time_ps")).alias("to_mean_survival_time_ps"),
        ]
    )


def edge_metrics_lf(edge_features: pl.LazyFrame) -> pl.LazyFrame:
    with_edge_channels = edge_features.with_columns(
        [
            (
                (
                    pl.col("from_direction_score") * pl.col("to_temporal_corr")
                    - pl.col("to_direction_score") * pl.col("from_temporal_corr")
                )
                * (-pl.col("minimum_heavy_atom_distance_angstrom") / pl.lit(8.0)).exp()
            ).alias("signed_te_stream"),
            (
                (
                    pl.col("from_temporal_corr").abs()
                    + pl.col("to_temporal_corr").abs()
                    + pl.col("from_direction_score").abs()
                    + pl.col("to_direction_score").abs()
                )
                * pl.lit(0.25)
            ).alias("edge_kcc_coherence_stream"),
            (
                pl.col("from_masked_spike_count").cast(pl.Float64)
                + pl.col("to_masked_spike_count").cast(pl.Float64)
            ).alias("edge_masked_spike_count_stream"),
            (
                pl.col("from_causal_anchor_count").cast(pl.Float64)
                + pl.col("to_causal_anchor_count").cast(pl.Float64)
            ).alias("edge_causal_anchor_count_stream"),
            pl.max_horizontal("from_max_snr3_ratio", "to_max_snr3_ratio").alias("edge_max_snr3_ratio_stream"),
            (
                pl.col("from_thermally_destabilized_spike_count").cast(pl.Float64)
                + pl.col("to_thermally_destabilized_spike_count").cast(pl.Float64)
            ).alias("edge_thermally_destabilized_spike_count_stream"),
            (
                pl.col("from_thermally_activated_spike_count").cast(pl.Float64)
                + pl.col("to_thermally_activated_spike_count").cast(pl.Float64)
            ).alias("edge_thermally_activated_spike_count_stream"),
            (
                pl.col("from_short_lived_regime_break_count").cast(pl.Float64)
                + pl.col("to_short_lived_regime_break_count").cast(pl.Float64)
            ).alias("edge_short_lived_break_count_stream"),
            (
                (
                    (pl.col("from_mean_abs_mechanical_load") + pl.lit(1.0))
                    * (pl.col("to_mean_abs_mechanical_load") + pl.lit(1.0))
                ).sqrt()
                * (-pl.col("minimum_heavy_atom_distance_angstrom") / pl.lit(12.0)).exp()
            ).alias("edge_abs_mechanical_load_stream"),
            (
                (pl.col("from_active_load_fraction") + pl.col("to_active_load_fraction"))
                * pl.lit(0.5)
            ).alias("edge_active_load_fraction_stream"),
            (
                pl.col("from_dt_drop_count").cast(pl.Float64)
                + pl.col("to_dt_drop_count").cast(pl.Float64)
                + pl.col("stream_dt_drop_count").cast(pl.Float64) * pl.lit(0.05)
            ).alias("edge_dt_drop_count_stream"),
            (
                pl.col("from_violent_dt_drop_count").cast(pl.Float64)
                + pl.col("to_violent_dt_drop_count").cast(pl.Float64)
                + pl.col("stream_violent_dt_drop_count").cast(pl.Float64) * pl.lit(0.05)
            ).alias("edge_violent_dt_drop_count_stream"),
            (
                pl.col("from_steering_weight_sum").cast(pl.Float64)
                + pl.col("to_steering_weight_sum").cast(pl.Float64)
            ).alias("edge_steering_weight_sum_stream"),
            (
                pl.col("from_steering_focus_observations").cast(pl.Float64)
                + pl.col("to_steering_focus_observations").cast(pl.Float64)
            ).alias("edge_steering_focus_observations_stream"),
            pl.mean_horizontal("from_mean_survival_time_ps", "to_mean_survival_time_ps", "stream_mean_survival_time_ps")
            .alias("edge_survival_time_ps_stream"),
            pl.min_horizontal("from_first_break_time_ps", "to_first_break_time_ps").alias("edge_first_break_time_ps_stream"),
            pl.mean_horizontal("from_mean_break_time_ps", "to_mean_break_time_ps").alias("edge_mean_break_time_ps_stream"),
        ]
    )
    aggregated = with_edge_channels.group_by(
        [
            *EDGE_OUTPUT_KEY,
            "minimum_heavy_atom_distance_angstrom",
            "contact_cutoff_angstrom",
        ]
    ).agg(
        [
            pl.col("signed_te_stream").mean().alias("signed_te_mean"),
            pl.col("signed_te_stream").std().fill_null(0.0).alias("signed_te_std"),
            pl.col("edge_kcc_coherence_stream").mean().alias("edge_kcc_coherence"),
            pl.col("edge_masked_spike_count_stream").sum().alias("masked_spike_count"),
            pl.col("edge_causal_anchor_count_stream").sum().alias("causal_anchor_count"),
            pl.col("edge_max_snr3_ratio_stream").max().alias("max_snr3_ratio"),
            pl.col("edge_thermally_destabilized_spike_count_stream").sum().alias("thermally_destabilized_spike_count"),
            pl.col("edge_thermally_activated_spike_count_stream").sum().alias("thermally_activated_spike_count"),
            pl.col("edge_short_lived_break_count_stream").sum().alias("short_lived_regime_break_count"),
            pl.col("edge_abs_mechanical_load_stream").mean().alias("mean_abs_mechanical_load"),
            pl.col("edge_abs_mechanical_load_stream").max().alias("max_abs_mechanical_load"),
            pl.col("edge_active_load_fraction_stream").mean().alias("active_load_fraction"),
            pl.col("edge_dt_drop_count_stream").sum().alias("dt_drop_count"),
            pl.col("edge_violent_dt_drop_count_stream").sum().alias("violent_dt_drop_count"),
            pl.col("edge_steering_weight_sum_stream").mean().alias("steering_weight_sum"),
            pl.col("edge_steering_focus_observations_stream").mean().alias("steering_focus_observations"),
            pl.col("edge_survival_time_ps_stream").mean().alias("mean_survival_time_ps"),
            pl.col("edge_first_break_time_ps_stream").min().alias("first_break_time_ps"),
            pl.col("edge_mean_break_time_ps_stream").mean().alias("mean_break_time_ps"),
            pl.col("condition_destabilized_voxel_fraction").mean().alias("condition_destabilized_voxel_fraction"),
            pl.col("condition_activated_voxel_fraction").mean().alias("condition_activated_voxel_fraction"),
            pl.col("stream_short_lived_regime_fraction").mean().alias("stream_short_lived_regime_fraction"),
            pl.col("aromatic_p90_displacement_std").mean().alias("aromatic_p90_displacement_std"),
            pl.len().alias("edge_stream_observations"),
        ]
    )
    filled = aggregated.with_columns(
        [
            pl.col("masked_spike_count").fill_null(0.0),
            pl.col("causal_anchor_count").fill_null(0.0),
            pl.col("max_snr3_ratio").fill_null(0.0),
            pl.col("mean_abs_mechanical_load").fill_null(0.0),
            pl.col("active_load_fraction").fill_null(0.0),
            pl.col("mean_survival_time_ps").fill_null(0.0),
            pl.col("aromatic_p90_displacement_std").fill_null(0.0),
            pl.col("condition_destabilized_voxel_fraction").fill_null(0.0),
            pl.col("condition_activated_voxel_fraction").fill_null(0.0),
            pl.col("stream_short_lived_regime_fraction").fill_null(0.0),
        ]
    )
    with_ratios = filled.with_columns(
        [
            pl.when(pl.col("masked_spike_count") > 0)
            .then(pl.col("causal_anchor_count") / pl.col("masked_spike_count"))
            .otherwise(0.0)
            .alias("causal_anchor_fraction"),
            pl.when(pl.col("masked_spike_count") > 0)
            .then(pl.col("thermally_destabilized_spike_count") / pl.col("masked_spike_count"))
            .otherwise(pl.col("condition_destabilized_voxel_fraction"))
            .alias("thermally_destabilized_fraction"),
            pl.when(pl.col("masked_spike_count") > 0)
            .then(pl.col("thermally_activated_spike_count") / pl.col("masked_spike_count"))
            .otherwise(pl.col("condition_activated_voxel_fraction"))
            .alias("thermally_activated_fraction"),
            pl.when(pl.col("masked_spike_count") > 0)
            .then(pl.col("short_lived_regime_break_count") / pl.col("masked_spike_count"))
            .otherwise(pl.col("stream_short_lived_regime_fraction"))
            .alias("short_lived_regime_fraction"),
            (
                (pl.col("mean_abs_mechanical_load") > 1e-6)
                & (pl.col("active_load_fraction") >= 0.05)
            ).alias("mechanical_validated"),
        ]
    )
    return with_ratios.with_columns(
        [
            (
                pl.col("signed_te_mean").abs() * 0.45
                + pl.col("edge_kcc_coherence") * 0.25
                + pl.col("signed_te_std") * 0.10
                + (pl.lit(1.0) / (pl.lit(1.0) + pl.col("minimum_heavy_atom_distance_angstrom"))) * 0.20
            ).alias("base_interface_score"),
            (
                pl.col("masked_spike_count").log1p() * 0.18
                + pl.col("causal_anchor_fraction") * 0.75
                + pl.col("max_snr3_ratio").log1p() * 0.08
            ).alias("snr_masked_spike_risk"),
            (
                pl.lit(1.20) * pl.col("thermally_destabilized_fraction")
                + pl.lit(0.45) * pl.col("thermally_activated_fraction")
            )
            .exp()
            .alias("variance_risk_penalty"),
            (pl.lit(1.0) - (-(pl.col("mean_abs_mechanical_load") / pl.lit(50.0))).exp())
            .alias("mechanical_coupling_score"),
            (
                pl.col("short_lived_regime_fraction") * 0.80
                + (
                    pl.lit(1.0)
                    / (pl.lit(1.0) + pl.col("mean_survival_time_ps").fill_null(0.0) * 100.0)
                )
                * 0.20
            ).alias("temporal_durability_risk"),
            (
                pl.col("dt_drop_count").log1p() * 0.12
                + (pl.col("violent_dt_drop_count") > 0).cast(pl.Float64) * 1.25
            ).alias("kinetic_rupture_violence_risk"),
            (
                pl.lit(1.0)
                + pl.col("steering_weight_sum").log1p()
                * pl.lit(0.15)
                * (-pl.col("minimum_heavy_atom_distance_angstrom") / pl.lit(20.0)).exp()
            ).alias("autonomous_steering_prior_multiplier"),
            (pl.col("aromatic_p90_displacement_std") * 5.0).exp().alias("aromatic_uv_penalty"),
        ]
    ).with_columns(
        [
            pl.when(pl.col("mechanical_validated"))
            .then(pl.col("mechanical_coupling_score"))
            .otherwise(0.0)
            .alias("mechanical_validation_multiplier")
        ]
    ).with_columns(
        [
            (
                (
                    pl.col("base_interface_score")
                    + pl.col("snr_masked_spike_risk")
                    + pl.col("temporal_durability_risk")
                    + pl.col("kinetic_rupture_violence_risk")
                    + pl.col("condition_destabilized_voxel_fraction") * 0.20
                )
                * pl.col("variance_risk_penalty")
                * pl.col("mechanical_validation_multiplier")
                * pl.col("autonomous_steering_prior_multiplier")
                * pl.col("aromatic_uv_penalty")
            ).alias("durability_risk_score_raw")
        ]
    )


def phase_edge_validation_lf(path: Path) -> pl.LazyFrame:
    return pl.scan_parquet(path).select(
        [
            "condition_id",
            pl.col("edge_from_residue").cast(pl.UInt32),
            pl.col("edge_to_residue").cast(pl.UInt32),
            "edge_class",
            "validation_status",
            "edge_coherence_score",
            pl.col("from_coherence_class").alias("phase_from_coherence_class"),
            pl.col("to_coherence_class").alias("phase_to_coherence_class"),
            pl.col("from_concordance_ratio").alias("phase_from_concordance_ratio"),
            pl.col("to_concordance_ratio").alias("phase_to_concordance_ratio"),
            pl.col("from_n_concordant_channels").alias("phase_from_n_concordant_channels"),
            pl.col("to_n_concordant_channels").alias("phase_to_n_concordant_channels"),
        ]
    )


def build_risk_map(
    base_dir: Path,
    topology_dir: Path,
    topology_register: Path,
    phase_edge_validation_path: Path,
) -> pl.LazyFrame:
    kcc_path = base_dir / "kcc_residue_fields.parquet"
    spike_path = base_dir / "spike_events_snr_masked.parquet"
    signal_path = base_dir / "signal_grid_variance_channel.parquet"
    mechanical_path = base_dir / "mechanical_load_network.parquet"
    bocpd_path = base_dir / "bocpd_survival_regimes.parquet"
    kinetic_path = base_dir / "kinetic_strain_events.parquet"
    steering_path = base_dir / "autonomous_steering_tensor.parquet"
    aromatic_path = base_dir / "aromatic_reorganization_tensor.parquet"
    spike_residue_path = ensure_spike_residue_aggregate(
        base_dir=base_dir,
        spike_path=spike_path,
        signal_path=signal_path,
        bocpd_path=bocpd_path,
        kinetic_path=kinetic_path,
    )

    kcc = pl.scan_parquet(kcc_path)
    edge_stream = edge_stream_lf(kcc, topology_register)
    edge_features = build_edge_stream_features(
        edge_stream=edge_stream,
        kcc=kcc,
        spikes=pl.scan_parquet(spike_residue_path),
        signal=pl.scan_parquet(signal_path),
        mechanical=pl.scan_parquet(mechanical_path),
        bocpd=pl.scan_parquet(bocpd_path),
        kinetic=pl.scan_parquet(kinetic_path),
        steering=pl.scan_parquet(steering_path),
        aromatic=pl.scan_parquet(aromatic_path),
        mechanical_path=mechanical_path,
        topology_dir=topology_dir,
    )
    scored = edge_metrics_lf(edge_features).join(
        phase_edge_validation_lf(phase_edge_validation_path),
        on=EDGE_OUTPUT_KEY,
        how="left",
    ).with_columns(
        [
            (pl.col("durability_risk_score_raw").rank("average") / pl.len()).alias(
                "durability_risk_percentile"
            ),
            (pl.col("violent_dt_drop_count") > 0).alias("violent_rupture_event"),
            (~pl.col("mechanical_validated")).alias("mechanically_pruned"),
            pl.col("validation_status").fill_null("phase_validation_missing"),
            pl.col("edge_coherence_score").fill_null(0.0),
            pl.col("phase_from_concordance_ratio").fill_null(0.0),
            pl.col("phase_to_concordance_ratio").fill_null(0.0),
            pl.col("phase_from_n_concordant_channels").fill_null(0),
            pl.col("phase_to_n_concordant_channels").fill_null(0),
        ]
    )
    return scored.with_columns(
        [
            pl.lit(CAMPAIGN_ID).alias("campaign_id"),
            pl.col("edge_from_residue").alias("residue_idx"),
            pl.when(pl.col("mechanically_pruned"))
            .then(pl.lit("mechanically_pruned"))
            .when(pl.col("violent_rupture_event"))
            .then(pl.lit("violent_rupture"))
            .when(pl.col("durability_risk_percentile") >= 0.95)
            .then(pl.lit("critical_durability_risk"))
            .when(pl.col("durability_risk_percentile") >= 0.75)
            .then(pl.lit("elevated_durability_risk"))
            .when(
                (pl.col("durability_risk_percentile") <= 0.25)
                & (pl.col("short_lived_regime_fraction") <= 0.25)
            )
            .then(pl.lit("quiet_thermal_lock"))
            .otherwise(pl.lit("moderate_durability_risk"))
            .alias("durability_class"),
        ]
    ).select(
        [
            "campaign_id",
            "condition_id",
            "edge_from_residue",
            "edge_to_residue",
            "edge_class",
            "residue_idx",
            "minimum_heavy_atom_distance_angstrom",
            "contact_cutoff_angstrom",
            "durability_class",
            "validation_status",
            "edge_coherence_score",
            "phase_from_coherence_class",
            "phase_to_coherence_class",
            "phase_from_concordance_ratio",
            "phase_to_concordance_ratio",
            "phase_from_n_concordant_channels",
            "phase_to_n_concordant_channels",
            "durability_risk_score_raw",
            "durability_risk_percentile",
            "signed_te_mean",
            "signed_te_std",
            "base_interface_score",
            "snr_masked_spike_risk",
            "variance_risk_penalty",
            "mechanical_validation_multiplier",
            "temporal_durability_risk",
            "kinetic_rupture_violence_risk",
            "autonomous_steering_prior_multiplier",
            "aromatic_uv_penalty",
            "masked_spike_count",
            "causal_anchor_count",
            "causal_anchor_fraction",
            "thermally_destabilized_fraction",
            "thermally_activated_fraction",
            "mean_abs_mechanical_load",
            "active_load_fraction",
            "mechanical_validated",
            "mechanically_pruned",
            "mean_survival_time_ps",
            "short_lived_regime_fraction",
            "dt_drop_count",
            "violent_dt_drop_count",
            "violent_rupture_event",
            "steering_weight_sum",
            "steering_focus_observations",
            "aromatic_p90_displacement_std",
            "first_break_time_ps",
            "mean_break_time_ps",
            "edge_stream_observations",
        ]
    )


def source_parquets(base_dir: Path, topology_register: Path, phase_edge_validation: Path) -> list[Path]:
    names = [
        "kcc_residue_fields.parquet",
        "spike_events_snr_masked.parquet",
        "stream_level_phase_counts.parquet",
        "signal_grid_variance_channel.parquet",
        "mechanical_load_network.parquet",
        "bocpd_survival_regimes.parquet",
        "kinetic_strain_events.parquet",
        "autonomous_steering_tensor.parquet",
        "aromatic_reorganization_tensor.parquet",
        "phase_manifold_coherence.parquet",
        "shear_stress_field.parquet",
    ]
    return [base_dir / name for name in names] + [phase_edge_validation, topology_register]


def write_outputs(risk_map: pl.LazyFrame, output_dir: Path, sources: Sequence[Path]) -> dict[str, int | str]:
    risk_path = output_dir / "receptor_durability_risk_map.parquet"
    summary_path = output_dir / "receptor_durability_channel_summary.parquet"
    parameters = {
        "fused_tensor_count": 10,
        "risk_granularity": "edge",
        "composite_join_key": "condition_id,replica_id,stream_id,edge_from_residue,edge_to_residue",
        "edge_validation_join_key": "condition_id,edge_from_residue,edge_to_residue,edge_class",
        "warp_jacobian_quarantined": False,
        "phase_manifold_coherence_fused": True,
        "snr_source": "spike_events_snr_masked.parquet",
        "topology_register": "sar_steric_interface_catalog.parquet",
        "variance_penalty": "exp(1.20 * thermally_destabilized_fraction + 0.45 * thermally_activated_fraction)",
        "mechanical_prune_rule": "edge mean_abs_mechanical_load <= 1e-6 or active_load_fraction < 0.05",
        "row_group_size": ROW_GROUP_SIZE,
    }
    write_n80_parquet(
        risk_map,
        risk_path,
        producer_script=Path("scripts/evaluate-glp1r-posthoc.py"),
        pipeline_stage="posthoc_edge_omni_tensor_fusion",
        schema_version="prism.receptor_durability_edge_risk_map.v2",
        partition_keys=["condition_id", "edge_from_residue", "edge_to_residue", "edge_class"],
        raw_inputs=[],
        source_parquets=sources,
        ledger_parameters=parameters,
        row_count=None,
    )
    summary = (
        pl.scan_parquet(risk_path)
        .group_by(["condition_id", "durability_class"])
        .agg(
            [
                pl.len().alias("edge_count"),
                pl.col("durability_risk_score_raw").mean().alias("mean_risk_score"),
                pl.col("durability_risk_score_raw").max().alias("max_risk_score"),
                pl.col("masked_spike_count").sum().alias("masked_spike_count"),
                pl.col("violent_rupture_event").cast(pl.UInt32).sum().alias("violent_rupture_edges"),
                pl.col("mechanically_pruned").cast(pl.UInt32).sum().alias("mechanically_pruned_edges"),
                (pl.col("validation_status") == "validated_constitutive")
                .cast(pl.UInt32)
                .sum()
                .alias("validated_constitutive_edges"),
                (pl.col("validation_status") == "divergent_artifact_warning")
                .cast(pl.UInt32)
                .sum()
                .alias("divergent_artifact_warning_edges"),
                pl.col("thermally_destabilized_fraction").mean().alias("mean_destabilized_fraction"),
                pl.col("aromatic_p90_displacement_std").mean().alias("mean_aromatic_p90_displacement_std"),
            ]
        )
        .sort(["condition_id", "durability_class"])
    )
    write_n80_parquet(
        summary,
        summary_path,
        producer_script=Path("scripts/evaluate-glp1r-posthoc.py"),
        pipeline_stage="posthoc_edge_omni_tensor_fusion_summary",
        schema_version="prism.receptor_durability_edge_channel_summary.v2",
        partition_keys=["condition_id", "durability_class"],
        raw_inputs=[],
        source_parquets=[risk_path, *sources],
        ledger_parameters=parameters,
        row_count=None,
    )
    risk_rows = pl.scan_parquet(risk_path).select(pl.len().alias("n")).collect().item()
    summary_rows = pl.scan_parquet(summary_path).select(pl.len().alias("n")).collect().item()
    top = (
        pl.scan_parquet(risk_path)
        .sort("durability_risk_score_raw", descending=True)
        .select(
            [
                "condition_id",
                "edge_from_residue",
                "edge_to_residue",
                "edge_class",
                "durability_class",
                "validation_status",
                "edge_coherence_score",
                "durability_risk_score_raw",
            ]
        )
        .head(10)
        .collect()
    )
    report = {
        "campaign_id": CAMPAIGN_ID,
        "risk_map_path": risk_path.as_posix(),
        "summary_path": summary_path.as_posix(),
        "risk_map_rows": int(risk_rows),
        "summary_rows": int(summary_rows),
        "risk_granularity": "edge",
        "composite_join_key": EDGE_KEY,
        "fused_tensors": [
            "spike_events_snr_masked",
            "signal_grid_variance_channel",
            "mechanical_load_network",
            "bocpd_survival_regimes",
            "kinetic_strain_events",
            "autonomous_steering_tensor",
            "aromatic_reorganization_tensor",
            "phase_manifold_coherence",
            "phase_manifold_edge_validation",
            "shear_stress_field",
        ],
        "warp_jacobian_quarantined": False,
        "top10": top.to_dicts(),
    }
    summary_json = output_dir / "receptor_durability_evaluation_summary.json"
    summary_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return {
        "risk_map_rows": int(risk_rows),
        "summary_rows": int(summary_rows),
        "risk_map_path": risk_path.as_posix(),
        "summary_path": summary_path.as_posix(),
        "summary_json": summary_json.as_posix(),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--topology-dir", type=Path, default=DEFAULT_TOPOLOGY_DIR)
    parser.add_argument("--topology-register", type=Path, default=DEFAULT_TOPOLOGY_REGISTER)
    parser.add_argument("--phase-edge-validation", type=Path, default=DEFAULT_PHASE_EDGE_VALIDATION)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    sources = source_parquets(args.input_dir, args.topology_register, args.phase_edge_validation)
    missing = [path for path in sources if not path.exists()]
    if missing:
        raise FileNotFoundError(f"missing required source parquet(s): {missing}")
    risk_map = build_risk_map(
        args.input_dir,
        args.topology_dir,
        args.topology_register,
        args.phase_edge_validation,
    )
    result = write_outputs(risk_map, args.output_dir, sources)
    LOGGER.info("evaluation_complete result=%s", json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
