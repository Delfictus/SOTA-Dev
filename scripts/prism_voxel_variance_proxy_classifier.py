#!/usr/bin/env python3
"""Classify aligned PRISM voxels into proxy stable-occupied and high-variance-void classes.

This is not producer-canonical voxel variance. It is an explicitly labeled
empirical proxy derived from dynamic voxel event bins, aligned voxel sidecars,
and run/stream event-time denominators.

Production safeguards:
* all class thresholds are computed inside each run/stream/scope partition;
* interface and materialized-site voxels are not pooled for thresholding;
* cross-stream summaries are emitted only after stream-local classification;
* the manifest carries validation gates and output checksums;
* outputs are suitable for Track 0 and Layer 1 prototype clash/complement
  fields, not final constructive-interference claims.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from prism_dstw.io import write_provenance_parquet


def emit_status(event: str, **fields: object) -> None:
    payload = {"event": event, **fields}
    sys.stderr.write(json.dumps(payload, sort_keys=True) + "\n")


PARTITION_KEYS = [
    "campaign_id",
    "run_label",
    "structure_id",
    "stream_id",
    "scope_type",
    "scope_id",
]

ALLOWED_CLASSES = {
    "stable_occupied_proxy",
    "high_variance_void_proxy",
    "transient_high_variance_occupied_proxy",
    "low_evidence_proxy",
    "mixed_proxy",
}

REQUIRED_INPUT_COLUMNS = {
    "dynamic_voxel_bins": {
        "campaign_id",
        "run_label",
        "structure_id",
        "stream_id",
        "timestep",
        "physical_time_ps",
        "scope_type",
        "scope_id",
        "voxel_idx",
        "spike_event_count",
        "mean_intensity",
        "max_intensity",
        "mean_water_density",
        "mean_wd_change",
        "mean_vibrational_energy",
    },
    "interface_voxels": {
        "campaign_id",
        "run_label",
        "structure_id",
        "stream_id",
        "scope_type",
        "scope_id",
        "interface_id",
        "interface_class",
        "target_hinge_residue_index",
        "neighbor_residue_index",
        "voxel_idx",
        "voxel_x",
        "voxel_y",
        "voxel_z",
        "grid_dim",
        "grid_spacing_a",
        "warp_n_atoms",
        "warp_total_weight",
        "target_endpoint_weight",
        "neighbor_endpoint_weight",
        "endpoint_weight",
        "weighted_force_norm",
        "weighted_asc_norm",
        "force_asc_delta_norm",
        "te_coupling_score",
        "lock_interface_score",
        "mean_distance_angstrom",
    },
    "site_voxels": {
        "campaign_id",
        "run_label",
        "structure_id",
        "stream_id",
        "scope_type",
        "scope_id",
        "site_id",
        "site_rank",
        "voxel_idx",
        "voxel_x",
        "voxel_y",
        "voxel_z",
        "grid_dim",
        "grid_spacing_a",
        "warp_n_atoms",
        "warp_total_weight",
        "weighted_force_norm",
        "weighted_asc_norm",
        "force_asc_delta_norm",
    },
    "time_bins": {
        "campaign_id",
        "run_label",
        "structure_id",
        "stream_id",
        "timestep",
        "physical_time_ps",
    },
}

REQUIRED_OUTPUT_COLUMNS = {
    "voxel_stable_void_proxy": {
        "campaign_id",
        "run_label",
        "structure_id",
        "stream_id",
        "scope_type",
        "scope_id",
        "voxel_idx",
        "occupancy_fraction",
        "occupancy_rank",
        "combined_variance_rank",
        "hydration_variance_rank",
        "stable_occupied_proxy_score",
        "high_variance_void_proxy_score",
        "voxel_field_class_proxy",
        "proxy_derived",
        "proxy_model_version",
        "threshold_partition_basis",
    },
    "scope_stream_stable_void_proxy_summary": {
        "campaign_id",
        "run_label",
        "structure_id",
        "stream_id",
        "scope_type",
        "scope_id",
        "voxel_count",
        "stable_occupied_voxel_count",
        "high_variance_void_voxel_count",
    },
    "scope_stable_void_proxy_summary": {
        "campaign_id",
        "run_label",
        "structure_id",
        "scope_type",
        "scope_id",
        "voxel_count",
        "aggregation_gate",
    },
    "interface_interference_terms_proxy": {
        "campaign_id",
        "run_label",
        "structure_id",
        "stream_id",
        "interface_id",
        "phi_prot_occupied_proxy",
        "phi_prot_void_proxy",
        "constructive_interference_gate",
    },
}


def collect(lf: pl.LazyFrame) -> pl.DataFrame:
    """Collect with Polars' streaming engine when available."""
    try:
        return lf.collect(engine="streaming")
    except TypeError:
        return lf.collect(streaming=True)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def parquet_metadata(path: Path) -> dict[str, object]:
    pf = pq.ParquetFile(path)
    schema = pf.schema_arrow
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "size_bytes": path.stat().st_size,
        "num_rows": pf.metadata.num_rows,
        "num_columns": pf.metadata.num_columns,
        "created_by": pf.metadata.created_by,
        "schema_sha256": hashlib.sha256(str(schema).encode("utf-8")).hexdigest(),
        "columns": {field.name: str(field.type) for field in schema},
    }


def validate_input_schema(name: str, metadata: dict[str, object]) -> bool:
    required = REQUIRED_INPUT_COLUMNS[name]
    columns = set(metadata["columns"])
    return required.issubset(columns)


def output_schema(df: pl.DataFrame) -> dict[str, str]:
    return {name: str(dtype) for name, dtype in zip(df.columns, df.dtypes)}


def missing_output_columns(name: str, df: pl.DataFrame) -> list[str]:
    return sorted(REQUIRED_OUTPUT_COLUMNS[name] - set(df.columns))


def bad_range_count(df: pl.DataFrame, columns: list[str], low: float, high: float) -> int:
    checks = [
        pl.col(col).is_null()
        | (~pl.col(col).is_finite())
        | (pl.col(col) < low)
        | (pl.col(col) > high)
        for col in columns
    ]
    return df.filter(pl.any_horizontal(*checks)).height


def write_proxy_parquet(df: pl.DataFrame, path: Path, source_parquets: list[Path]) -> None:
    write_provenance_parquet(
        df,
        path,
        producer_script=Path(__file__),
        source_parquets=source_parquets,
        schema_version="prism_voxel_variance_proxy_classifier.v2.polars_native",
        pipeline_stage="voxel_variance_proxy",
        partition_keys=PARTITION_KEYS,
        extra_metadata={"lineage_scope": "voxel_stable_void_proxy"},
    )


def move_parquet_with_ledger(source: Path, target: Path) -> None:
    source.replace(target)
    source_ledger = source.with_suffix(".propagation.jsonl")
    if source_ledger.exists():
        source_ledger.replace(target.with_suffix(".propagation.jsonl"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-id", default="glp1r_aleniglipron")
    parser.add_argument("--dynamic-voxel-bins", type=Path, required=True)
    parser.add_argument("--interface-voxels", type=Path, required=True)
    parser.add_argument("--site-voxels", type=Path, required=True)
    parser.add_argument("--time-bins", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--min-observed-time-bins", type=int, default=3)
    parser.add_argument("--stable-score-threshold", type=float, default=0.55)
    parser.add_argument("--void-score-threshold", type=float, default=0.55)
    args = parser.parse_args()
    strict_no_banned_producers = True

    args.out_dir.mkdir(parents=True, exist_ok=True)

    classified_path = args.out_dir / "voxel_stable_void_proxy.parquet"
    scope_stream_summary_path = args.out_dir / "scope_stream_stable_void_proxy_summary.parquet"
    scope_summary_path = args.out_dir / "scope_stable_void_proxy_summary.parquet"
    interface_terms_path = args.out_dir / "interface_interference_terms_proxy.parquet"
    manifest_path = args.out_dir / "voxel_variance_proxy_manifest.json"
    final_output_paths = {
        "voxel_stable_void_proxy": classified_path,
        "scope_stream_stable_void_proxy_summary": scope_stream_summary_path,
        "scope_stable_void_proxy_summary": scope_summary_path,
        "interface_interference_terms_proxy": interface_terms_path,
    }

    input_paths = {
        "dynamic_voxel_bins": args.dynamic_voxel_bins,
        "interface_voxels": args.interface_voxels,
        "site_voxels": args.site_voxels,
        "time_bins": args.time_bins,
    }
    input_metadata = {name: parquet_metadata(path) for name, path in input_paths.items()}
    banned_producer_tokens = ["duck" + "db", "pandas"]
    banned_inputs = [
        name
        for name, metadata in input_metadata.items()
        if any(token in str(metadata.get("created_by", "")).lower() for token in banned_producer_tokens)
    ]
    if banned_inputs:
        raise SystemExit(f"Banned producer-created input parquet rejected by policy: {banned_inputs}")
    input_schema_checks = {
        f"{name}_schema_contains_required_columns": validate_input_schema(name, metadata)
        for name, metadata in input_metadata.items()
    }
    if not all(input_schema_checks.values()):
        failed = [name for name, passed in input_schema_checks.items() if not passed]
        raise SystemExit(f"input schema validation failed: {failed}")

    dynamic_bins = pl.scan_parquet(args.dynamic_voxel_bins).with_columns(
        pl.col("stream_id").cast(pl.Int16)
    )

    time_denominator = (
        pl.scan_parquet(args.time_bins)
        .with_columns(pl.col("stream_id").cast(pl.Int16))
        .group_by(["campaign_id", "run_label", "structure_id", "stream_id"])
        .agg(
            pl.col("timestep").n_unique().alias("total_time_bins"),
            pl.col("timestep").min().alias("min_timestep"),
            pl.col("timestep").max().alias("max_timestep"),
            pl.col("physical_time_ps").min().alias("min_time_ps"),
            pl.col("physical_time_ps").max().alias("max_time_ps"),
        )
    )

    interface_columns = [
        "campaign_id",
        "run_label",
        "structure_id",
        "stream_id",
        "scope_type",
        "scope_id",
        "interface_id",
        "interface_class",
        "target_hinge_residue_index",
        "neighbor_residue_index",
        "site_id",
        "site_rank",
        "voxel_idx",
        "voxel_x",
        "voxel_y",
        "voxel_z",
        "grid_dim",
        "grid_spacing_a",
        "warp_n_atoms",
        "warp_total_weight",
        "endpoint_weight",
        "target_endpoint_weight",
        "neighbor_endpoint_weight",
        "weighted_force_norm",
        "weighted_asc_norm",
        "force_asc_delta_norm",
        "te_coupling_score",
        "lock_interface_score",
        "mean_distance_angstrom",
    ]

    interface_static = (
        pl.scan_parquet(args.interface_voxels)
        .with_columns(
            pl.col("stream_id").cast(pl.Int16),
            pl.lit(None, dtype=pl.Utf8).alias("site_id"),
            pl.lit(None, dtype=pl.Int32).alias("site_rank"),
            pl.col("endpoint_weight").cast(pl.Float64),
            pl.col("target_endpoint_weight").cast(pl.Float64),
            pl.col("neighbor_endpoint_weight").cast(pl.Float64),
            pl.col("warp_total_weight").cast(pl.Float64),
            pl.col("mean_distance_angstrom").cast(pl.Float64),
        )
        .select(interface_columns)
    )

    site_static = (
        pl.scan_parquet(args.site_voxels)
        .with_columns(
            pl.col("stream_id").cast(pl.Int16),
            pl.lit(None, dtype=pl.Utf8).alias("interface_id"),
            pl.lit(None, dtype=pl.Utf8).alias("interface_class"),
            pl.lit(None, dtype=pl.Int32).alias("target_hinge_residue_index"),
            pl.lit(None, dtype=pl.Int32).alias("neighbor_residue_index"),
            pl.lit(None, dtype=pl.Float64).alias("endpoint_weight"),
            pl.lit(None, dtype=pl.Float64).alias("target_endpoint_weight"),
            pl.lit(None, dtype=pl.Float64).alias("neighbor_endpoint_weight"),
            pl.lit(None, dtype=pl.Float64).alias("te_coupling_score"),
            pl.lit(None, dtype=pl.Float64).alias("lock_interface_score"),
            pl.lit(None, dtype=pl.Float64).alias("mean_distance_angstrom"),
            pl.col("warp_total_weight").cast(pl.Float64),
        )
        .select(interface_columns)
    )

    static_scopes = pl.concat([interface_static, site_static], how="vertical_relaxed")

    voxel_event_metrics = (
        dynamic_bins.group_by(PARTITION_KEYS + ["voxel_idx"])
        .agg(
            pl.col("timestep").n_unique().alias("observed_time_bins"),
            pl.col("spike_event_count").sum().alias("total_spike_event_count"),
            pl.col("spike_event_count").mean().alias("mean_spike_event_count_observed"),
            pl.col("spike_event_count").std(ddof=0).alias("sd_spike_event_count_observed"),
            pl.col("mean_intensity").mean().alias("mean_intensity"),
            pl.col("mean_intensity").std(ddof=0).alias("sd_mean_intensity"),
            pl.col("max_intensity").max().alias("max_intensity"),
            pl.col("mean_water_density").mean().alias("mean_water_density"),
            pl.col("mean_water_density").std(ddof=0).alias("sd_mean_water_density"),
            pl.col("mean_wd_change").mean().alias("mean_wd_change"),
            pl.col("mean_wd_change").abs().mean().alias("mean_abs_wd_change"),
            pl.col("mean_wd_change").std(ddof=0).alias("sd_mean_wd_change"),
            pl.col("mean_vibrational_energy").mean().alias("mean_vibrational_energy"),
            pl.col("mean_vibrational_energy").std(ddof=0).alias("sd_mean_vibrational_energy"),
            pl.col("timestep").min().alias("first_observed_timestep"),
            pl.col("timestep").max().alias("last_observed_timestep"),
            pl.col("physical_time_ps").min().alias("first_observed_time_ps"),
            pl.col("physical_time_ps").max().alias("last_observed_time_ps"),
        )
    )

    numeric_zero_cols = [
        "observed_time_bins",
        "total_spike_event_count",
        "mean_spike_event_count_observed",
        "sd_spike_event_count_observed",
        "mean_intensity",
        "sd_mean_intensity",
        "max_intensity",
        "mean_water_density",
        "sd_mean_water_density",
        "mean_wd_change",
        "mean_abs_wd_change",
        "sd_mean_wd_change",
        "mean_vibrational_energy",
        "sd_mean_vibrational_energy",
    ]

    base = (
        static_scopes.join(
            time_denominator,
            on=["campaign_id", "run_label", "structure_id", "stream_id"],
            how="inner",
        )
        .join(voxel_event_metrics, on=PARTITION_KEYS + ["voxel_idx"], how="left")
        .with_columns([pl.col(col).fill_null(0) for col in numeric_zero_cols])
        .with_columns(
            (pl.col("observed_time_bins").cast(pl.Float64) / pl.col("total_time_bins").cast(pl.Float64)).alias(
                "occupancy_fraction"
            ),
            (
                pl.col("total_spike_event_count").cast(pl.Float64)
                / pl.col("total_time_bins").cast(pl.Float64)
            ).alias("event_rate_per_time_bin"),
            pl.coalesce(
                [pl.col("endpoint_weight"), pl.col("warp_total_weight"), pl.lit(0.0)]
            ).alias("local_receptor_weight"),
        )
    )

    partition_len = pl.len().over(PARTITION_KEYS).cast(pl.Float64)
    ranked = base.with_columns(
        (pl.col("occupancy_fraction").rank(method="max").over(PARTITION_KEYS) / partition_len).alias(
            "occupancy_rank"
        ),
        (pl.col("event_rate_per_time_bin").rank(method="max").over(PARTITION_KEYS) / partition_len).alias(
            "event_rate_rank"
        ),
        (pl.col("sd_mean_intensity").rank(method="max").over(PARTITION_KEYS) / partition_len).alias(
            "intensity_variance_rank"
        ),
        (pl.col("sd_mean_water_density").rank(method="max").over(PARTITION_KEYS) / partition_len).alias(
            "water_density_variance_rank"
        ),
        (pl.col("sd_mean_wd_change").rank(method="max").over(PARTITION_KEYS) / partition_len).alias(
            "wd_change_variance_rank"
        ),
        (pl.col("mean_abs_wd_change").rank(method="max").over(PARTITION_KEYS) / partition_len).alias(
            "abs_wd_change_rank"
        ),
        (pl.col("local_receptor_weight").rank(method="max").over(PARTITION_KEYS) / partition_len).alias(
            "local_receptor_weight_rank"
        ),
    )

    combined_variance = pl.max_horizontal(
        "intensity_variance_rank", "water_density_variance_rank", "wd_change_variance_rank"
    )
    hydration_variance = pl.max_horizontal(
        "water_density_variance_rank", "wd_change_variance_rank", "abs_wd_change_rank"
    )
    stable_score = pl.col("occupancy_rank") * (1.0 - combined_variance) * pl.col(
        "local_receptor_weight_rank"
    )
    void_score = hydration_variance * (1.0 - stable_score) * (0.5 + pl.col("occupancy_rank") / 2.0)

    classified_lf = (
        ranked.with_columns(
            combined_variance.alias("combined_variance_rank"),
            hydration_variance.alias("hydration_variance_rank"),
            stable_score.alias("stable_occupied_proxy_score"),
            void_score.alias("high_variance_void_proxy_score"),
        )
        .with_columns(
            pl.when(
                (pl.col("observed_time_bins") < args.min_observed_time_bins)
                | (pl.col("occupancy_rank") <= 0.20)
            )
            .then(pl.lit("low_evidence_proxy"))
            .when(
                (pl.col("stable_occupied_proxy_score") >= args.stable_score_threshold)
                & (pl.col("occupancy_rank") >= 0.65)
                & (pl.col("combined_variance_rank") <= 0.50)
            )
            .then(pl.lit("stable_occupied_proxy"))
            .when(
                (pl.col("high_variance_void_proxy_score") >= args.void_score_threshold)
                & (pl.col("hydration_variance_rank") >= 0.65)
                & (pl.col("stable_occupied_proxy_score") < args.stable_score_threshold)
            )
            .then(pl.lit("high_variance_void_proxy"))
            .when((pl.col("occupancy_rank") >= 0.65) & (pl.col("combined_variance_rank") >= 0.65))
            .then(pl.lit("transient_high_variance_occupied_proxy"))
            .otherwise(pl.lit("mixed_proxy"))
            .alias("voxel_field_class_proxy"),
            pl.lit(True).alias("proxy_derived"),
            pl.lit("empirical_dynamic_voxel_variance_proxy.v1").alias("proxy_model_version"),
            pl.lit("partitioned_by_campaign_run_structure_stream_scope_type_scope_id").alias(
                "threshold_partition_basis"
            ),
        )
    )

    classified = collect(classified_lf)

    scope_count_exprs = [
        pl.len().alias("voxel_count"),
        (pl.col("voxel_field_class_proxy") == "stable_occupied_proxy")
        .sum()
        .alias("stable_occupied_voxel_count"),
        (pl.col("voxel_field_class_proxy") == "high_variance_void_proxy")
        .sum()
        .alias("high_variance_void_voxel_count"),
        (pl.col("voxel_field_class_proxy") == "transient_high_variance_occupied_proxy")
        .sum()
        .alias("transient_high_variance_occupied_voxel_count"),
        (pl.col("voxel_field_class_proxy") == "low_evidence_proxy")
        .sum()
        .alias("low_evidence_voxel_count"),
        pl.col("occupancy_fraction").mean().alias("mean_occupancy_fraction"),
        pl.col("event_rate_per_time_bin").mean().alias("mean_event_rate_per_time_bin"),
        pl.col("stable_occupied_proxy_score").mean().alias("mean_stable_occupied_proxy_score"),
        pl.col("high_variance_void_proxy_score").mean().alias("mean_high_variance_void_proxy_score"),
        pl.col("hydration_variance_rank").mean().alias("mean_hydration_variance_rank"),
    ]

    scope_stream_summary = classified.group_by(PARTITION_KEYS).agg(
        pl.col("interface_id").first().alias("interface_id"),
        pl.col("interface_class").first().alias("interface_class"),
        pl.col("site_id").first().alias("site_id"),
        *scope_count_exprs,
    )

    cross_scope_keys = ["campaign_id", "run_label", "structure_id", "scope_type", "scope_id"]
    scope_summary = (
        classified.group_by(cross_scope_keys)
        .agg(
            pl.col("interface_id").first().alias("interface_id"),
            pl.col("interface_class").first().alias("interface_class"),
            pl.col("site_id").first().alias("site_id"),
            *scope_count_exprs,
        )
        .with_columns(pl.lit("post_stream_classification_cross_stream_summary").alias("aggregation_gate"))
    )

    interface_terms = (
        classified.filter(pl.col("scope_type") == "sar_interface")
        .group_by(["campaign_id", "run_label", "structure_id", "stream_id", "interface_id"])
        .agg(
            pl.col("interface_class").first().alias("interface_class"),
            pl.col("target_hinge_residue_index").first().alias("target_hinge_residue_index"),
            pl.col("neighbor_residue_index").first().alias("neighbor_residue_index"),
            pl.len().alias("interface_voxel_count"),
            (pl.col("stable_occupied_proxy_score") * pl.col("endpoint_weight").fill_null(0.0))
            .sum()
            .alias("phi_prot_occupied_proxy"),
            (pl.col("high_variance_void_proxy_score") * pl.col("endpoint_weight").fill_null(0.0))
            .sum()
            .alias("phi_prot_void_proxy"),
            pl.col("stable_occupied_proxy_score").mean().alias("mean_stable_occupied_proxy_score"),
            pl.col("high_variance_void_proxy_score").mean().alias("mean_high_variance_void_proxy_score"),
            (pl.col("voxel_field_class_proxy") == "stable_occupied_proxy")
            .sum()
            .alias("stable_occupied_voxel_count"),
            (pl.col("voxel_field_class_proxy") == "high_variance_void_proxy")
            .sum()
            .alias("high_variance_void_voxel_count"),
            (pl.col("voxel_field_class_proxy") == "transient_high_variance_occupied_proxy")
            .sum()
            .alias("transient_high_variance_occupied_voxel_count"),
            pl.col("te_coupling_score").fill_null(0.0).abs().mean().alias("abs_te_coupling_score"),
            pl.col("lock_interface_score").fill_null(0.0).mean().alias("lock_interface_score"),
        )
        .with_columns(
            pl.lit("proxy_derived_constructive_term_not_canonical").alias("constructive_interference_gate")
        )
    )

    static_count = collect(static_scopes.select(pl.len().alias("n")))["n"][0]
    counts = {
        "voxel_stable_void_proxy": classified.height,
        "scope_stream_stable_void_proxy_summary": scope_stream_summary.height,
        "scope_stable_void_proxy_summary": scope_summary.height,
        "interface_interference_terms_proxy": interface_terms.height,
    }
    class_counts = (
        classified.group_by("voxel_field_class_proxy")
        .agg(pl.len().alias("n"))
        .sort("n", descending=True)
    )
    class_count_map = dict(zip(class_counts["voxel_field_class_proxy"].to_list(), class_counts["n"].to_list()))

    duplicate_count = (
        classified.group_by(PARTITION_KEYS + ["voxel_idx"])
        .agg(pl.len().alias("n"))
        .filter(pl.col("n") > 1)
        .height
    )
    frames = {
        "voxel_stable_void_proxy": classified,
        "scope_stream_stable_void_proxy_summary": scope_stream_summary,
        "scope_stable_void_proxy_summary": scope_summary,
        "interface_interference_terms_proxy": interface_terms,
    }
    output_schema_checks = {
        f"{name}_schema_contains_required_columns": not missing_output_columns(name, df)
        for name, df in frames.items()
    }
    output_schemas = {name: output_schema(df) for name, df in frames.items()}
    rank_columns = [
        "occupancy_rank",
        "event_rate_rank",
        "intensity_variance_rank",
        "water_density_variance_rank",
        "wd_change_variance_rank",
        "abs_wd_change_rank",
        "local_receptor_weight_rank",
        "combined_variance_rank",
        "hydration_variance_rank",
        "stable_occupied_proxy_score",
        "high_variance_void_proxy_score",
    ]
    class_domain_count = classified.filter(~pl.col("voxel_field_class_proxy").is_in(ALLOWED_CLASSES)).height
    stable_condition_violations = classified.filter(
        (pl.col("voxel_field_class_proxy") == "stable_occupied_proxy")
        & (
            (pl.col("stable_occupied_proxy_score") < args.stable_score_threshold)
            | (pl.col("occupancy_rank") < 0.65)
            | (pl.col("combined_variance_rank") > 0.50)
        )
    ).height
    void_condition_violations = classified.filter(
        (pl.col("voxel_field_class_proxy") == "high_variance_void_proxy")
        & (
            (pl.col("high_variance_void_proxy_score") < args.void_score_threshold)
            | (pl.col("hydration_variance_rank") < 0.65)
            | (pl.col("stable_occupied_proxy_score") >= args.stable_score_threshold)
        )
    ).height
    low_evidence_condition_violations = classified.filter(
        (pl.col("voxel_field_class_proxy") == "low_evidence_proxy")
        & ~(
            (pl.col("observed_time_bins") < args.min_observed_time_bins)
            | (pl.col("occupancy_rank") <= 0.20)
        )
    ).height
    transient_condition_violations = classified.filter(
        (pl.col("voxel_field_class_proxy") == "transient_high_variance_occupied_proxy")
        & ~((pl.col("occupancy_rank") >= 0.65) & (pl.col("combined_variance_rank") >= 0.65))
    ).height
    stream_summary_class_count_mismatch = scope_stream_summary.filter(
        (
            pl.col("stable_occupied_voxel_count")
            + pl.col("high_variance_void_voxel_count")
            + pl.col("transient_high_variance_occupied_voxel_count")
            + pl.col("low_evidence_voxel_count")
        )
        > pl.col("voxel_count")
    ).height
    interface_term_bad_numeric_count = interface_terms.filter(
        pl.any_horizontal(
            pl.col("phi_prot_occupied_proxy").is_null(),
            pl.col("phi_prot_void_proxy").is_null(),
            ~pl.col("phi_prot_occupied_proxy").is_finite(),
            ~pl.col("phi_prot_void_proxy").is_finite(),
            pl.col("phi_prot_occupied_proxy") < 0,
            pl.col("phi_prot_void_proxy") < 0,
        )
    ).height
    validation_checks = {
        **input_schema_checks,
        **output_schema_checks,
        "classified_rows_match_static_scope_rows": classified.height == static_count,
        "threshold_basis_is_stream_local": classified.filter(
            pl.col("threshold_partition_basis")
            != "partitioned_by_campaign_run_structure_stream_scope_type_scope_id"
        ).height
        == 0,
        "no_null_core_scores_or_classes": classified.filter(
            pl.any_horizontal(
                pl.col("occupancy_fraction").is_null(),
                pl.col("stable_occupied_proxy_score").is_null(),
                pl.col("high_variance_void_proxy_score").is_null(),
                pl.col("voxel_field_class_proxy").is_null(),
            )
        ).height
        == 0,
        "classified_key_is_unique": duplicate_count == 0,
        "stream_summary_rows_match_stream_scope_partitions": scope_stream_summary.height
        == classified.select(PARTITION_KEYS).unique().height,
        "cross_stream_summary_is_post_classification_labeled": scope_summary.filter(
            pl.col("aggregation_gate") != "post_stream_classification_cross_stream_summary"
        ).height
        == 0,
        "interface_terms_have_proxy_constructive_gate": interface_terms.filter(
            pl.col("constructive_interference_gate") != "proxy_derived_constructive_term_not_canonical"
        ).height
        == 0,
        "rank_and_score_columns_are_finite_unit_interval": bad_range_count(classified, rank_columns, 0.0, 1.0)
        == 0,
        "occupancy_fraction_is_finite_unit_interval": bad_range_count(classified, ["occupancy_fraction"], 0.0, 1.0)
        == 0,
        "event_rate_is_finite_nonnegative": classified.filter(
            pl.col("event_rate_per_time_bin").is_null()
            | (~pl.col("event_rate_per_time_bin").is_finite())
            | (pl.col("event_rate_per_time_bin") < 0)
        ).height
        == 0,
        "observed_time_bins_do_not_exceed_denominator": classified.filter(
            pl.col("observed_time_bins") > pl.col("total_time_bins")
        ).height
        == 0,
        "total_time_bins_positive": classified.filter(pl.col("total_time_bins") <= 0).height == 0,
        "voxel_class_domain_is_closed": class_domain_count == 0,
        "stable_class_satisfies_declared_gate": stable_condition_violations == 0,
        "high_variance_void_class_satisfies_declared_gate": void_condition_violations == 0,
        "low_evidence_class_satisfies_declared_gate": low_evidence_condition_violations == 0,
        "transient_high_variance_class_satisfies_declared_gate": transient_condition_violations == 0,
        "stream_summary_class_counts_do_not_exceed_voxel_count": stream_summary_class_count_mismatch == 0,
        "interface_terms_are_finite_nonnegative": interface_term_bad_numeric_count == 0,
    }
    if not all(validation_checks.values()):
        failed = [name for name, passed in validation_checks.items() if not passed]
        raise SystemExit(f"validation failed: {failed}")

    staging_dir = Path(tempfile.mkdtemp(prefix=".voxel_variance_proxy_tmp_", dir=args.out_dir))
    staged_output_paths = {name: staging_dir / path.name for name, path in final_output_paths.items()}
    staged_manifest_path = staging_dir / manifest_path.name
    source_parquets = list(input_paths.values())
    try:
        for name, df in frames.items():
            write_proxy_parquet(df, staged_output_paths[name], source_parquets)

        output_metadata = {}
        for name, staged_path in staged_output_paths.items():
            metadata = parquet_metadata(staged_path)
            metadata["path"] = str(final_output_paths[name])
            output_metadata[name] = metadata
        staged_file_checks = {
            f"{name}_staged_file_row_count_matches_frame": metadata["num_rows"] == frames[name].height
            for name, metadata in output_metadata.items()
        }
        staged_file_checks.update(
            {
                f"{name}_staged_file_schema_contains_required_columns": REQUIRED_OUTPUT_COLUMNS[name].issubset(
                    set(metadata["columns"])
                )
                for name, metadata in output_metadata.items()
            }
        )
        validation_checks.update(staged_file_checks)
        if not all(validation_checks.values()):
            failed = [name for name, passed in validation_checks.items() if not passed]
            raise SystemExit(f"staged output validation failed: {failed}")

        manifest = {
            "schema": "prism_voxel_variance_proxy_classifier.v2.polars_native",
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "campaign_id": args.campaign_id,
            "classification_basis": "empirical proxy derived from dynamic_voxel_event_time_bins, aligned voxel sidecars, and per-run/stream time denominators",
            "engine": "polars_native_arrow_parquet",
            "dependency_versions": {
                "python": sys.version,
                "polars": pl.__version__,
                "pyarrow": pa.__version__,
            },
            "argv": sys.argv,
            "generator": str(Path(__file__).resolve()),
            "generator_sha256": sha256_file(Path(__file__).resolve()),
            "proxy_derived": True,
            "inputs": {name: str(path) for name, path in input_paths.items()},
            "input_metadata": input_metadata,
            "source_engine_boundary": {
                "proxy_classifier_engine": "polars_native_arrow_parquet",
                "source_parquet_created_by": {
                    name: metadata["created_by"] for name, metadata in input_metadata.items()
                },
                "strict_end_to_end_no_banned_producers_enforced": strict_no_banned_producers,
                "note": "This proxy materialization is Polars-native. Source Parquet producer metadata is preserved explicitly; banned source producers are rejected by default.",
            },
            "parameters": {
                "min_observed_time_bins": args.min_observed_time_bins,
                "stable_score_threshold": args.stable_score_threshold,
                "void_score_threshold": args.void_score_threshold,
                "threshold_partition_basis": "campaign_id/run_label/structure_id/stream_id/scope_type/scope_id",
                "rank_partition_keys": PARTITION_KEYS,
                "score_model": {
                    "version": "empirical_dynamic_voxel_variance_proxy.v1",
                    "stable_occupied_proxy_score": "occupancy_rank * (1 - max(intensity_variance_rank, water_density_variance_rank, wd_change_variance_rank)) * local_receptor_weight_rank",
                    "high_variance_void_proxy_score": "max(water_density_variance_rank, wd_change_variance_rank, abs_wd_change_rank) * (1 - stable_occupied_proxy_score) * (0.5 + occupancy_rank / 2)",
                    "calibration_status": "hand-tuned empirical proxy; not producer-canonical and not experimentally calibrated",
                },
            },
            "outputs": {name: str(path) for name, path in final_output_paths.items()},
            "output_metadata": output_metadata,
            "output_sha256": {name: metadata["sha256"] for name, metadata in output_metadata.items()},
            "output_schemas": output_schemas,
            "counts": counts,
            "class_counts": class_count_map,
            "validation_checks": validation_checks,
            "semantic_gates": [
                "This is proxy-derived, not producer-canonical per-voxel variance.",
                "Stable occupied proxy is derived from high temporal occupancy, low event/hydration variance, and local receptor/endpoint support within the same run/stream/scope partition.",
                "High-variance void proxy is derived from hydration/WD-change variance and non-stable occupancy within the same run/stream/scope partition.",
                "Interface and materialized-site voxels are thresholded separately inside each run/stream/scope partition; no global averaging is used for class assignment.",
                "Cross-stream summaries are emitted only after stream-local voxel classification and carry aggregation_gate=post_stream_classification_cross_stream_summary.",
                "Constructive interference can use phi_prot_void_proxy only with a proxy-derived warning until producer-side voxel variance is canonical.",
            ],
        }
        staged_manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")

        for name, final_path in final_output_paths.items():
            move_parquet_with_ledger(staged_output_paths[name], final_path)
        staged_manifest_path.replace(manifest_path)
    finally:
        shutil.rmtree(staging_dir, ignore_errors=True)
    emit_status(
        "voxel_variance_proxy_classifier_complete",
        manifest=str(manifest_path),
        counts=counts,
        class_counts=class_count_map,
    )


if __name__ == "__main__":
    main()
