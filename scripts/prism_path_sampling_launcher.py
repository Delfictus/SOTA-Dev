#!/usr/bin/env python3
"""Build localized path-sampling launch windows from PRISM transition evidence.

The output is a launch queue, not a claim that an interface has broken. Each row
is a bounded, reproducible sampling target derived from event-support
transitions, SAR interface metadata, optional KCC deltas, and optional aligned
voxel evidence.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import polars as pl
import pyarrow.parquet as pq

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from prism_dstw.io import write_provenance_parquet


PARTITION_KEYS = ["campaign_id", "run_label", "structure_id", "stream_id", "interface_id"]


def emit_status(event: str, **fields: object) -> None:
    payload = {"event": event, **fields}
    sys.stderr.write(json.dumps(payload, sort_keys=True) + "\n")


def parquet_rows(path: Path) -> int:
    return int(pq.read_metadata(path).num_rows)


def validate_paths(paths: list[Path]) -> None:
    missing = [str(path) for path in paths if path and not path.exists()]
    if missing:
        raise SystemExit(f"Missing required path-sampling inputs: {missing}")


def interface_catalog(path: Path) -> pl.LazyFrame:
    return (
        pl.scan_parquet(path)
        .with_columns(
            pl.concat_str([pl.col("target_hinge_label"), pl.lit("__"), pl.col("neighbor_label")]).alias(
                "interface_id"
            )
        )
        .group_by(["interface_id", "target_hinge_residue_index", "neighbor_residue_index"])
        .agg(
            [
                pl.col("target_hinge_label").drop_nulls().first().alias("target_hinge_label"),
                pl.col("neighbor_label").drop_nulls().first().alias("neighbor_label"),
                pl.col("pocket_accessibility_class").drop_nulls().first().alias("interface_class"),
                pl.col("lock_interface_score").drop_nulls().first().fill_null(0.0).alias("lock_interface_score"),
                pl.col("te_coupling_score").drop_nulls().first().fill_null(0.0).alias("te_coupling_score"),
                pl.col("interface_te_asymmetry").drop_nulls().first().alias("interface_te_asymmetry"),
                pl.col("mean_distance_angstrom").drop_nulls().first().alias("mean_distance_angstrom"),
                pl.col("nearest_materialized_pocket_site_id").drop_nulls().first().alias(
                    "nearest_materialized_pocket_site_id"
                ),
                pl.col("nearest_materialized_pocket_rank").drop_nulls().first().alias(
                    "nearest_materialized_pocket_rank"
                ),
            ]
        )
    )


def kcc_context(paths: list[str]) -> pl.LazyFrame:
    if not paths:
        return pl.LazyFrame(
            {
                "run_label": pl.Series([], dtype=pl.Utf8),
                "interface_id": pl.Series([], dtype=pl.Utf8),
                "endpoint_delta_norm": pl.Series([], dtype=pl.Float64),
                "force_delta_norm": pl.Series([], dtype=pl.Float64),
                "asc_delta_norm": pl.Series([], dtype=pl.Float64),
            }
        )
    lf = pl.scan_parquet(paths)
    endpoint_terms = [
        pl.col("delta_target_minus_neighbor_net_dx").fill_null(0.0).pow(2),
        pl.col("delta_target_minus_neighbor_net_dy").fill_null(0.0).pow(2),
        pl.col("delta_target_minus_neighbor_net_dz").fill_null(0.0).pow(2),
    ]
    force_terms = [
        pl.col("delta_target_minus_neighbor_temporal_corr").fill_null(0.0).pow(2),
        pl.col("delta_target_minus_neighbor_direction_score").fill_null(0.0).pow(2),
        pl.col("delta_target_minus_neighbor_motion_efficiency").fill_null(0.0).pow(2),
        pl.col("delta_target_minus_neighbor_burst_motion").fill_null(0.0).pow(2),
        pl.col("delta_target_minus_neighbor_phase_shift").fill_null(0.0).pow(2),
        pl.col("delta_target_minus_neighbor_local_cov").fill_null(0.0).pow(2),
        pl.col("delta_target_minus_neighbor_sum_m").fill_null(0.0).pow(2),
    ]
    return (
        lf.with_columns(
            [
                sum(endpoint_terms).sqrt().alias("_endpoint_delta_norm"),
                sum(force_terms).sqrt().alias("_force_delta_norm"),
            ]
        )
        .group_by(["run_label", "interface_id"])
        .agg(
            [
                pl.col("_endpoint_delta_norm").mean().alias("endpoint_delta_norm"),
                pl.col("_force_delta_norm").mean().alias("force_delta_norm"),
                pl.lit(None, dtype=pl.Float64).alias("asc_delta_norm"),
            ]
        )
    )


def voxel_event_context(path: Path | None) -> pl.LazyFrame:
    if path is None or not path.exists():
        return pl.LazyFrame(
            {
                "run_label": pl.Series([], dtype=pl.Utf8),
                "stream_id": pl.Series([], dtype=pl.Int16),
                "scope_id": pl.Series([], dtype=pl.Utf8),
                "voxel_event_rows": pl.Series([], dtype=pl.UInt64),
                "voxel_spike_event_count": pl.Series([], dtype=pl.UInt64),
                "voxel_mean_intensity": pl.Series([], dtype=pl.Float64),
                "voxel_mean_water_density": pl.Series([], dtype=pl.Float64),
                "voxel_mean_wd_change": pl.Series([], dtype=pl.Float64),
            }
        )
    return (
        pl.scan_parquet(path)
        .with_columns(pl.col("stream_id").cast(pl.Int16))
        .filter(pl.col("scope_type") == "sar_interface")
        .group_by(["run_label", "stream_id", "scope_id"])
        .agg(
            [
                pl.len().cast(pl.UInt64).alias("voxel_event_rows"),
                pl.col("spike_event_count").sum().cast(pl.UInt64).alias("voxel_spike_event_count"),
                pl.col("mean_intensity").mean().alias("voxel_mean_intensity"),
                pl.col("mean_water_density").mean().alias("voxel_mean_water_density"),
                pl.col("mean_wd_change").mean().alias("voxel_mean_wd_change"),
            ]
        )
    )


def interface_voxel_context(path: Path | None) -> pl.LazyFrame:
    if path is None or not path.exists():
        return pl.LazyFrame(
            {
                "run_label": pl.Series([], dtype=pl.Utf8),
                "stream_id": pl.Series([], dtype=pl.Int16),
                "interface_id": pl.Series([], dtype=pl.Utf8),
                "aligned_voxel_count": pl.Series([], dtype=pl.UInt64),
                "mean_endpoint_weight": pl.Series([], dtype=pl.Float64),
                "mean_weighted_force_norm": pl.Series([], dtype=pl.Float64),
                "mean_weighted_asc_norm": pl.Series([], dtype=pl.Float64),
            }
        )
    return (
        pl.scan_parquet(path)
        .with_columns(pl.col("stream_id").cast(pl.Int16))
        .group_by(["run_label", "stream_id", "interface_id"])
        .agg(
            [
                pl.col("voxel_idx").n_unique().cast(pl.UInt64).alias("aligned_voxel_count"),
                pl.col("endpoint_weight").mean().alias("mean_endpoint_weight"),
                pl.col("weighted_force_norm").mean().alias("mean_weighted_force_norm"),
                pl.col("weighted_asc_norm").mean().alias("mean_weighted_asc_norm"),
            ]
        )
    )


def support_context(base: pl.LazyFrame, time_bins: pl.LazyFrame, *, window_steps: int, side: str) -> pl.LazyFrame:
    tb = time_bins.select(
        [
            *PARTITION_KEYS,
            pl.col("timestep").alias("tb_timestep"),
            pl.col("support_fraction").alias("tb_support_fraction"),
            pl.col("hit_count").alias("tb_hit_count"),
        ]
    )
    joined = base.join(tb, on=PARTITION_KEYS, how="inner")
    if side == "pre":
        filtered = joined.filter(
            (pl.col("tb_timestep") >= pl.max_horizontal(pl.lit(0), pl.col("transition_to_timestep") - window_steps))
            & (pl.col("tb_timestep") <= pl.col("transition_to_timestep"))
        )
        prefix = "pre"
    elif side == "post":
        filtered = joined.filter(
            (pl.col("tb_timestep") >= pl.col("transition_to_timestep"))
            & (pl.col("tb_timestep") <= pl.col("transition_to_timestep") + window_steps)
        )
        prefix = "post"
    else:
        raise ValueError(f"unsupported support side: {side}")
    return filtered.group_by("transition_row_id").agg(
        [
            pl.col("tb_support_fraction").mean().alias(f"{prefix}_mean_support_fraction"),
            pl.col("tb_hit_count").mean().alias(f"{prefix}_mean_hit_count"),
            pl.col("tb_hit_count").max().alias(f"{prefix}_max_hit_count"),
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-id", default="glp1r_aleniglipron")
    parser.add_argument("--transitions", type=Path, required=True)
    parser.add_argument("--time-bins", type=Path, required=True)
    parser.add_argument("--interfaces", type=Path, required=True)
    parser.add_argument("--kcc-pair-deltas", action="append", default=[])
    parser.add_argument("--dynamic-voxel-bins", type=Path)
    parser.add_argument("--interface-voxels", type=Path)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--window-steps", type=int, default=50)
    parser.add_argument("--cluster-width-steps", type=int, default=25)
    parser.add_argument("--top-per-interface-class", type=int, default=24)
    args = parser.parse_args()

    required = [args.transitions, args.time_bins, args.interfaces] + [Path(p) for p in args.kcc_pair_deltas]
    if args.dynamic_voxel_bins:
        required.append(args.dynamic_voxel_bins)
    if args.interface_voxels:
        required.append(args.interface_voxels)
    validate_paths(required)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    ranked_path = args.out_dir / "localized_path_sampling_ranked_windows.parquet"
    queue_path = args.out_dir / "localized_path_sampling_launch_queue.parquet"
    jsonl_path = args.out_dir / "localized_path_sampling_launch_queue.jsonl"
    shell_path = args.out_dir / "localized_path_sampling_commands.sh"

    transitions = (
        pl.scan_parquet(args.transitions)
        .with_columns(pl.col("stream_id").cast(pl.Int16))
        .filter(pl.col("transition_class").is_in(["formation_candidate", "break_candidate"]))
        .filter(pl.col("transition_to_timestep").is_not_null())
        .with_row_index("transition_row_id")
        .with_columns(
            [
                (pl.col("transition_to_timestep") // args.cluster_width_steps).cast(pl.Int64).alias(
                    "transition_cluster_bin"
                ),
                pl.max_horizontal(pl.lit(0), pl.col("transition_to_timestep") - args.window_steps)
                .cast(pl.Int64)
                .alias("window_start_timestep"),
                (pl.col("transition_to_timestep") + args.window_steps).cast(pl.Int64).alias("window_end_timestep"),
            ]
        )
    )
    time_bins = pl.scan_parquet(args.time_bins).with_columns(pl.col("stream_id").cast(pl.Int16))

    clusters = transitions.group_by(["run_label", "interface_id", "transition_class", "transition_cluster_bin"]).agg(
        [
            pl.len().cast(pl.UInt64).alias("cluster_transition_count"),
            pl.col("stream_id").n_unique().cast(pl.UInt64).alias("cluster_stream_count"),
            pl.col("transition_to_timestep").min().alias("cluster_min_timestep"),
            pl.col("transition_to_timestep").max().alias("cluster_max_timestep"),
        ]
    )
    pre = support_context(transitions, time_bins, window_steps=args.window_steps, side="pre")
    post = support_context(transitions, time_bins, window_steps=args.window_steps, side="post")
    ic = interface_catalog(args.interfaces)
    kcc = kcc_context(args.kcc_pair_deltas)
    iv = interface_voxel_context(args.interface_voxels)
    ve = voxel_event_context(args.dynamic_voxel_bins)

    transition_context = (
        transitions.join(clusters, on=["run_label", "interface_id", "transition_class", "transition_cluster_bin"])
        .join(ic, on="interface_id", how="left", suffix="_catalog")
        .join(pre, on="transition_row_id", how="left")
        .join(post, on="transition_row_id", how="left")
        .join(kcc, on=["run_label", "interface_id"], how="left")
        .join(iv, on=["run_label", "stream_id", "interface_id"], how="left")
        .join(ve, left_on=["run_label", "stream_id", "interface_id"], right_on=["run_label", "stream_id", "scope_id"], how="left")
        .with_columns(
            [
                pl.coalesce([pl.col("interface_class_catalog"), pl.col("interface_class")]).alias("interface_class"),
                pl.col("te_coupling_score").fill_null(0.0),
                pl.col("lock_interface_score").fill_null(0.0),
                pl.col("mean_endpoint_weight").fill_null(0.0),
                pl.col("voxel_spike_event_count").fill_null(0).cast(pl.UInt64),
            ]
        )
        .with_columns(
            [
                pl.col("te_coupling_score").abs().alias("abs_te_coupling_score"),
                (
                    10.0 * (pl.lit(1.0) + pl.col("cluster_stream_count").cast(pl.Float64)).log()
                    + 2.0 * (pl.lit(1.0) + pl.col("cluster_transition_count").cast(pl.Float64)).log()
                    + 25.0 * pl.col("te_coupling_score").abs()
                    + 100.0 * pl.col("lock_interface_score")
                    + 5.0 * pl.col("mean_endpoint_weight")
                    + 2.0 * (pl.lit(1.0) + pl.col("voxel_spike_event_count").cast(pl.Float64)).log()
                    + pl.when(pl.col("transition_class") == "break_candidate").then(3.0).otherwise(1.0)
                ).alias("path_sampling_priority_score"),
                pl.when(pl.col("transition_class") == "break_candidate")
                .then(pl.lit("interface_breaking_path_sample"))
                .when(pl.col("transition_class") == "formation_candidate")
                .then(pl.lit("interface_forming_path_sample"))
                .otherwise(pl.lit("transition_path_sample"))
                .alias("sampling_objective"),
                pl.lit("launch_plan_only_no_md_restart_executed").alias("execution_status"),
            ]
        )
        .select(
            [
                "campaign_id",
                "run_label",
                "structure_id",
                "stream_id",
                "interface_id",
                "interface_class",
                "target_hinge_residue_index",
                "neighbor_residue_index",
                "target_hinge_label",
                "neighbor_label",
                "transition_class",
                "transition_from_timestep",
                "transition_to_timestep",
                "transition_from_time_ps",
                "transition_to_time_ps",
                "window_start_timestep",
                "window_end_timestep",
                "transition_cluster_bin",
                "cluster_transition_count",
                "cluster_stream_count",
                "cluster_min_timestep",
                "cluster_max_timestep",
                "pre_mean_support_fraction",
                "post_mean_support_fraction",
                "pre_mean_hit_count",
                "post_mean_hit_count",
                "pre_max_hit_count",
                "post_max_hit_count",
                "abs_te_coupling_score",
                "te_coupling_score",
                "lock_interface_score",
                "interface_te_asymmetry",
                "mean_distance_angstrom",
                "nearest_materialized_pocket_site_id",
                "nearest_materialized_pocket_rank",
                "endpoint_delta_norm",
                "force_delta_norm",
                "asc_delta_norm",
                "aligned_voxel_count",
                "mean_endpoint_weight",
                "mean_weighted_force_norm",
                "mean_weighted_asc_norm",
                "voxel_event_rows",
                "voxel_spike_event_count",
                "voxel_mean_intensity",
                "voxel_mean_water_density",
                "voxel_mean_wd_change",
                "path_sampling_priority_score",
                "sampling_objective",
                "execution_status",
            ]
        )
        .sort(
            ["path_sampling_priority_score", "cluster_stream_count", "transition_to_timestep"],
            descending=[True, True, False],
        )
    )

    write_provenance_parquet(
        transition_context,
        ranked_path,
        producer_script=Path(__file__),
        source_parquets=required,
        schema_version="prism_localized_path_sampling_launcher.v2.polars_native",
        pipeline_stage="localized_path_sampling_launch",
        partition_keys=PARTITION_KEYS,
        extra_metadata={"lineage_scope": "localized_path_sampling_launcher"},
    )
    ranked = pl.scan_parquet(ranked_path).collect(engine="streaming").sort(
        ["path_sampling_priority_score", "cluster_stream_count", "transition_to_timestep"],
        descending=[True, True, False],
    )
    queue = (
        ranked.with_columns(
            pl.int_range(1, pl.len() + 1)
            .over(["run_label", "interface_id", "transition_class"])
            .alias("rank_within_interface_transition")
        )
        .filter(pl.col("rank_within_interface_transition") <= args.top_per_interface_class)
        .sort(
            ["path_sampling_priority_score", "cluster_stream_count", "transition_to_timestep"],
            descending=[True, True, False],
        )
    )
    write_provenance_parquet(
        queue,
        queue_path,
        producer_script=Path(__file__),
        source_parquets=required,
        schema_version="prism_localized_path_sampling_launcher.v2.polars_native",
        pipeline_stage="localized_path_sampling_launch",
        partition_keys=PARTITION_KEYS,
        extra_metadata={"lineage_scope": "localized_path_sampling_launcher"},
    )

    with jsonl_path.open("w", encoding="utf-8") as fh:
        for row in queue.to_dicts():
            fh.write(json.dumps(row, sort_keys=True, default=str) + "\n")

    with shell_path.open("w", encoding="utf-8") as fh:
        fh.write("#!/usr/bin/env bash\n")
        fh.write("set -euo pipefail\n\n")
        fh.write("# Launch queue generated from PRISM event-support transitions.\n")
        fh.write("# Commands are templates because no MD restart/path-sampling executable was invoked by this script.\n\n")
        for row in queue.head(100).to_dicts():
            fh.write(
                "# prism-local-path-sample "
                f"--campaign-id {args.campaign_id} "
                f"--run-label {row['run_label']} "
                f"--structure-id {row['structure_id']} "
                f"--stream-id {int(row['stream_id'])} "
                f"--interface-id {row['interface_id']} "
                f"--objective {row['sampling_objective']} "
                f"--start-step {int(row['window_start_timestep'])} "
                f"--end-step {int(row['window_end_timestep'])}\n"
            )

    counts = {"ranked_windows": parquet_rows(ranked_path), "launch_queue": parquet_rows(queue_path)}
    manifest = {
        "schema": "prism_localized_path_sampling_launcher.v2.polars_native",
        "engine": "polars_native_arrow_parquet",
        "campaign_id": args.campaign_id,
        "inputs": {
            "transitions": str(args.transitions),
            "time_bins": str(args.time_bins),
            "interfaces": str(args.interfaces),
            "kcc_pair_deltas": args.kcc_pair_deltas,
            "dynamic_voxel_bins": str(args.dynamic_voxel_bins) if args.dynamic_voxel_bins else None,
            "interface_voxels": str(args.interface_voxels) if args.interface_voxels else None,
        },
        "parameters": {
            "window_steps": args.window_steps,
            "cluster_width_steps": args.cluster_width_steps,
            "top_per_interface_class": args.top_per_interface_class,
        },
        "outputs": {
            "ranked_windows": str(ranked_path),
            "launch_queue": str(queue_path),
            "launch_queue_jsonl": str(jsonl_path),
            "command_templates": str(shell_path),
        },
        "counts": counts,
        "semantic_gates": [
            "Rows are localized path-sampling targets, not completed MD path-sampling results.",
            "transition_class comes from event-support threshold crossings, not biological interface truth.",
            "priority scoring is for queue triage only and must not be interpreted as a probability.",
            "Any actual interface-breaking timestamp requires executing restart/path-sampling and validating contact geometry over the sampled paths.",
            "The launch JSONL is orchestration output only; analytical data are the provenance-stamped parquet files.",
        ],
    }
    manifest_path = args.out_dir / "localized_path_sampling_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    emit_status("localized_path_sampling_launch_complete", manifest=str(manifest_path), counts=counts)


if __name__ == "__main__":
    main()
