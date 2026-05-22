#!/usr/bin/env python3
"""Mine PRISM spike-event/interface-hit Parquets into temporal support intervals.

Inputs are produced by `prism_spike_event_integrator.py`.

This producer is Arrow/Polars-native. It deliberately separates:
* observed event-time bins, derived from all spike events;
* interface-hit support, derived only from interface-hit rows;
* support intervals and transitions, derived from explicit threshold rules.

It does not claim that a support-loss transition is a biological interface
break. It is a timestamp candidate that must be interpreted with state controls,
path-sampling, and assay context.
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import polars as pl
import pyarrow.parquet as pq

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from prism_dstw.io import write_provenance_parquet


GROUP_KEYS = ["campaign_id", "run_label", "structure_id", "stream_id", "timestep"]
INTERFACE_KEYS = GROUP_KEYS + ["interface_id"]
PARTITION_KEYS = ["campaign_id", "run_label", "structure_id", "stream_id", "interface_id"]


def expand_patterns(patterns: list[str]) -> list[str]:
    out: list[str] = []
    for pattern in patterns:
        matches = sorted(glob.glob(pattern))
        if matches:
            out.extend(matches)
        else:
            out.append(pattern)
    return sorted(dict.fromkeys(out))


def sha256_path(path: Path, chunk_size: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while True:
            chunk = fh.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def parquet_rows(path: Path) -> int:
    return int(pq.read_metadata(path).num_rows)


def parquet_created_by(path: Path) -> str | None:
    metadata = pq.read_metadata(path)
    return metadata.created_by


def emit_status(event: str, **fields: Any) -> None:
    payload = {"event": event, **fields}
    sys.stderr.write(json.dumps(payload, sort_keys=True) + "\n")


def collect_streaming(lf: pl.LazyFrame) -> pl.DataFrame:
    return lf.collect(engine="streaming")


def write_lazy_parquet(lf: pl.LazyFrame, path: Path, source_parquets: list[Path]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    write_provenance_parquet(
        lf,
        path,
        producer_script=Path(__file__),
        source_parquets=source_parquets,
        schema_version="prism_interface_timestamp_mining.v2.polars_native",
        pipeline_stage="interface_timestamp_mining",
        partition_keys=PARTITION_KEYS,
        extra_metadata={"lineage_scope": "interface_timestamp_mining"},
    )


def move_parquet_with_ledger(source: Path, target: Path) -> None:
    source.replace(target)
    source_ledger = source.with_suffix(".propagation.jsonl")
    if source_ledger.exists():
        source_ledger.replace(target.with_suffix(".propagation.jsonl"))


def source_inventory(paths: list[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for raw in paths:
        path = Path(raw)
        rows.append(
            {
                "path": str(path),
                "rows": parquet_rows(path),
                "created_by": parquet_created_by(path),
                "sha256": sha256_path(path),
            }
        )
    return rows


def validate_inputs(event_paths: list[str], hit_paths: list[str], interface_catalog: Path) -> None:
    if not event_paths:
        raise SystemExit("No event Parquet paths resolved.")
    if not hit_paths:
        raise SystemExit("No interface-hit Parquet paths resolved.")
    missing = [p for p in event_paths + hit_paths + [str(interface_catalog)] if not Path(p).exists()]
    if missing:
        raise SystemExit(f"Missing required Parquet inputs: {missing}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--event-parquet", action="append", required=True, help="Path or glob. Repeatable.")
    parser.add_argument("--hit-parquet", action="append", required=True, help="Path or glob. Repeatable.")
    parser.add_argument("--interface-catalog", type=Path, required=True, help="Interface catalog parquet.")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--support-threshold", type=float, default=0.0)
    parser.add_argument("--min-hit-count", type=int, default=1)
    args = parser.parse_args()

    event_paths = expand_patterns(args.event_parquet)
    hit_paths = expand_patterns(args.hit_parquet)
    validate_inputs(event_paths, hit_paths, args.interface_catalog)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    staging = args.out_dir / ".staging_polars_interface_timestamp_mining"
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)

    bins_stage = staging / "interface_time_bins.parquet"
    intervals_stage = staging / "interface_support_intervals.parquet"
    transitions_stage = staging / "interface_state_transitions.parquet"
    source_parquets = [Path(p) for p in event_paths + hit_paths] + [args.interface_catalog]

    events = (
        pl.scan_parquet(event_paths)
        .select(
            [
                pl.col("campaign_id").cast(pl.Utf8),
                pl.col("run_label").cast(pl.Utf8),
                pl.col("structure_id").cast(pl.Utf8),
                pl.col("stream_id").cast(pl.Int16),
                pl.col("timestep").cast(pl.Int64),
                pl.col("physical_time_ps").cast(pl.Float64),
                pl.col("intensity").cast(pl.Float64),
                pl.col("water_density").cast(pl.Float64),
                pl.col("wd_change").cast(pl.Float64),
            ]
        )
    )
    hits = (
        pl.scan_parquet(hit_paths)
        .select(
            [
                pl.col("campaign_id").cast(pl.Utf8),
                pl.col("run_label").cast(pl.Utf8),
                pl.col("structure_id").cast(pl.Utf8),
                pl.col("stream_id").cast(pl.Int16),
                pl.col("timestep").cast(pl.Int64),
                pl.col("physical_time_ps").cast(pl.Float64),
                pl.col("interface_id").cast(pl.Utf8),
                pl.col("interface_class").cast(pl.Utf8),
                pl.col("target_hinge_residue_index").cast(pl.Int64),
                pl.col("neighbor_residue_index").cast(pl.Int64),
                pl.col("match_basis").cast(pl.Utf8),
                pl.col("intensity").cast(pl.Float64),
                pl.col("water_density").cast(pl.Float64),
                pl.col("wd_change").cast(pl.Float64),
            ]
        )
    )
    interface_catalog = (
        pl.scan_parquet(args.interface_catalog)
        .select(
            [
                pl.col("interface_id").cast(pl.Utf8),
                pl.col("interface_class").cast(pl.Utf8),
                pl.col("target_hinge_residue_index").cast(pl.Int64),
                pl.col("neighbor_residue_index").cast(pl.Int64),
            ]
        )
        .unique()
    )

    event_bins = events.group_by(GROUP_KEYS).agg(
        [
            pl.col("physical_time_ps").min().alias("physical_time_ps"),
            pl.len().cast(pl.UInt64).alias("total_event_count"),
            pl.col("intensity").mean().alias("event_mean_intensity"),
            pl.col("water_density").mean().alias("event_mean_water_density"),
            pl.col("wd_change").mean().alias("event_mean_wd_change"),
        ]
    )
    hit_bins = hits.group_by(INTERFACE_KEYS).agg(
        [
            pl.len().cast(pl.UInt64).alias("hit_count"),
            (pl.col("match_basis") == "residue_pair").sum().cast(pl.UInt64).alias("residue_pair_hit_count"),
            (pl.col("match_basis") == "residue_single").sum().cast(pl.UInt64).alias("residue_single_hit_count"),
            (pl.col("match_basis") == "site_shell").sum().cast(pl.UInt64).alias("site_shell_hit_count"),
            pl.col("intensity").mean().alias("hit_mean_intensity"),
            pl.col("water_density").mean().alias("hit_mean_water_density"),
            pl.col("wd_change").mean().alias("hit_mean_wd_change"),
            pl.col("physical_time_ps").min().alias("hit_first_time_ps"),
            pl.col("physical_time_ps").max().alias("hit_last_time_ps"),
        ]
    )

    interface_time_bins = (
        event_bins.join(interface_catalog, how="cross")
        .join(hit_bins, on=INTERFACE_KEYS, how="left")
        .with_columns(
            [
                pl.col("hit_count").fill_null(0).cast(pl.UInt64),
                pl.col("residue_pair_hit_count").fill_null(0).cast(pl.UInt64),
                pl.col("residue_single_hit_count").fill_null(0).cast(pl.UInt64),
                pl.col("site_shell_hit_count").fill_null(0).cast(pl.UInt64),
            ]
        )
        .with_columns(
            [
                (pl.col("hit_count").cast(pl.Float64) / pl.col("total_event_count").cast(pl.Float64)).alias(
                    "support_fraction"
                ),
                (
                    pl.col("residue_pair_hit_count").cast(pl.Float64)
                    / pl.col("total_event_count").cast(pl.Float64)
                ).alias("residue_pair_support_fraction"),
                (
                    pl.col("residue_single_hit_count").cast(pl.Float64)
                    / pl.col("total_event_count").cast(pl.Float64)
                ).alias("residue_single_support_fraction"),
                (
                    pl.col("site_shell_hit_count").cast(pl.Float64)
                    / pl.col("total_event_count").cast(pl.Float64)
                ).alias("site_shell_support_fraction"),
            ]
        )
        .with_columns(
            (
                (pl.col("hit_count") >= int(args.min_hit_count))
                & (pl.col("support_fraction") > float(args.support_threshold))
            ).alias("is_supported")
        )
        .select(
            [
                "campaign_id",
                "run_label",
                "structure_id",
                "stream_id",
                "timestep",
                "physical_time_ps",
                "interface_id",
                "interface_class",
                "target_hinge_residue_index",
                "neighbor_residue_index",
                "total_event_count",
                "hit_count",
                "residue_pair_hit_count",
                "residue_single_hit_count",
                "site_shell_hit_count",
                "support_fraction",
                "residue_pair_support_fraction",
                "residue_single_support_fraction",
                "site_shell_support_fraction",
                "event_mean_intensity",
                "event_mean_water_density",
                "event_mean_wd_change",
                "hit_mean_intensity",
                "hit_mean_water_density",
                "hit_mean_wd_change",
                "is_supported",
            ]
        )
        .sort(["run_label", "structure_id", "stream_id", "timestep", "interface_id"])
    )
    write_lazy_parquet(interface_time_bins, bins_stage, source_parquets)

    marked = (
        pl.scan_parquet(bins_stage)
        .sort(PARTITION_KEYS + ["timestep"])
        .with_columns(
            [
                pl.col("is_supported").shift(1).over(PARTITION_KEYS).alias("prev_supported"),
                pl.col("timestep").shift(1).over(PARTITION_KEYS).alias("prev_timestep"),
                pl.col("physical_time_ps").shift(1).over(PARTITION_KEYS).alias("prev_time_ps"),
            ]
        )
        .with_columns(
            (
                pl.col("is_supported")
                & (~pl.col("prev_supported").fill_null(False))
            )
            .cast(pl.Int64)
            .alias("interval_start")
        )
        .with_columns(pl.col("interval_start").cum_sum().over(PARTITION_KEYS).alias("interval_group"))
    )

    support_intervals = (
        marked.filter(pl.col("is_supported"))
        .group_by(PARTITION_KEYS + ["interval_group"])
        .agg(
            [
                pl.col("interface_class").drop_nulls().first().alias("interface_class"),
                pl.col("target_hinge_residue_index").drop_nulls().first().alias("target_hinge_residue_index"),
                pl.col("neighbor_residue_index").drop_nulls().first().alias("neighbor_residue_index"),
                pl.col("timestep").min().alias("start_timestep"),
                pl.col("timestep").max().alias("end_timestep"),
                pl.col("physical_time_ps").min().alias("start_time_ps"),
                pl.col("physical_time_ps").max().alias("end_time_ps"),
                pl.len().cast(pl.UInt64).alias("supported_bin_count"),
                pl.col("hit_count").sum().alias("total_hit_count"),
                pl.col("residue_pair_hit_count").sum().alias("total_residue_pair_hit_count"),
                pl.col("residue_single_hit_count").sum().alias("total_residue_single_hit_count"),
                pl.col("site_shell_hit_count").sum().alias("total_site_shell_hit_count"),
                pl.col("support_fraction").mean().alias("mean_support_fraction"),
                pl.col("support_fraction").max().alias("max_support_fraction"),
                pl.col("hit_mean_intensity").mean().alias("mean_hit_intensity"),
                pl.col("hit_mean_water_density").mean().alias("mean_hit_water_density"),
                pl.col("hit_mean_wd_change").mean().alias("mean_hit_wd_change"),
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
                "start_timestep",
                "end_timestep",
                "start_time_ps",
                "end_time_ps",
                "supported_bin_count",
                "total_hit_count",
                "total_residue_pair_hit_count",
                "total_residue_single_hit_count",
                "total_site_shell_hit_count",
                "mean_support_fraction",
                "max_support_fraction",
                "mean_hit_intensity",
                "mean_hit_water_density",
                "mean_hit_wd_change",
            ]
        )
        .sort(["run_label", "structure_id", "stream_id", "interface_id", "start_timestep"])
    )
    write_lazy_parquet(support_intervals, intervals_stage, source_parquets)

    state_transitions = (
        marked.filter(
            (pl.col("is_supported") & (~pl.col("prev_supported").fill_null(False)))
            | ((~pl.col("is_supported")) & pl.col("prev_supported").fill_null(False))
        )
        .with_columns(
            [
                pl.when(pl.col("is_supported") & (~pl.col("prev_supported").fill_null(False)))
                .then(pl.lit("formation_candidate"))
                .when((~pl.col("is_supported")) & pl.col("prev_supported").fill_null(False))
                .then(pl.lit("break_candidate"))
                .otherwise(pl.lit("none"))
                .alias("transition_class"),
                pl.when(pl.col("is_supported") & (~pl.col("prev_supported").fill_null(False)))
                .then(pl.col("timestep"))
                .when((~pl.col("is_supported")) & pl.col("prev_supported").fill_null(False))
                .then(pl.col("prev_timestep"))
                .otherwise(None)
                .alias("transition_from_timestep"),
                pl.when(pl.col("is_supported") & (~pl.col("prev_supported").fill_null(False)))
                .then(pl.col("timestep"))
                .when((~pl.col("is_supported")) & pl.col("prev_supported").fill_null(False))
                .then(pl.col("timestep"))
                .otherwise(None)
                .alias("transition_to_timestep"),
                pl.when(pl.col("is_supported") & (~pl.col("prev_supported").fill_null(False)))
                .then(pl.col("physical_time_ps"))
                .when((~pl.col("is_supported")) & pl.col("prev_supported").fill_null(False))
                .then(pl.col("prev_time_ps"))
                .otherwise(None)
                .alias("transition_from_time_ps"),
                pl.when(pl.col("is_supported") & (~pl.col("prev_supported").fill_null(False)))
                .then(pl.col("physical_time_ps"))
                .when((~pl.col("is_supported")) & pl.col("prev_supported").fill_null(False))
                .then(pl.col("physical_time_ps"))
                .otherwise(None)
                .alias("transition_to_time_ps"),
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
                "transition_class",
                "transition_from_timestep",
                "transition_to_timestep",
                "transition_from_time_ps",
                "transition_to_time_ps",
                "hit_count",
                "support_fraction",
                "residue_pair_hit_count",
                "residue_single_hit_count",
                "site_shell_hit_count",
            ]
        )
        .sort(["run_label", "structure_id", "stream_id", "interface_id", "transition_to_timestep"])
    )
    write_lazy_parquet(state_transitions, transitions_stage, source_parquets)

    event_time_bins = int(collect_streaming(event_bins.select(pl.len().alias("n"))).item())
    interface_catalog_rows = int(collect_streaming(interface_catalog.select(pl.len().alias("n"))).item())
    counts = {
        "event_time_bins": event_time_bins,
        "interface_catalog_rows": interface_catalog_rows,
        "interface_time_bins": parquet_rows(bins_stage),
        "support_intervals": parquet_rows(intervals_stage),
        "state_transitions": parquet_rows(transitions_stage),
    }

    bins_path = args.out_dir / "interface_time_bins.parquet"
    intervals_path = args.out_dir / "interface_support_intervals.parquet"
    transitions_path = args.out_dir / "interface_state_transitions.parquet"
    for source, target in [
        (bins_stage, bins_path),
        (intervals_stage, intervals_path),
        (transitions_stage, transitions_path),
    ]:
        move_parquet_with_ledger(source, target)

    summary = {
        "schema": "prism_interface_timestamp_mining.v2.polars_native",
        "engine": "polars_native_arrow_parquet",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "event_parquets": event_paths,
        "hit_parquets": hit_paths,
        "interface_catalog": str(args.interface_catalog),
        "support_threshold": float(args.support_threshold),
        "min_hit_count": int(args.min_hit_count),
        "outputs": {
            "interface_time_bins": str(bins_path),
            "interface_support_intervals": str(intervals_path),
            "interface_state_transitions": str(transitions_path),
        },
        "counts": counts,
        "output_sha256": {
            "interface_time_bins": sha256_path(bins_path),
            "interface_support_intervals": sha256_path(intervals_path),
            "interface_state_transitions": sha256_path(transitions_path),
        },
        "source_inventory": {
            "events": source_inventory(event_paths),
            "hits": source_inventory(hit_paths),
            "interface_catalog": source_inventory([str(args.interface_catalog)]),
        },
        "semantic_warnings": [
            "formation_candidate and break_candidate are event-support transitions, not biological claims.",
            "A break candidate means observed support fell below the explicit threshold in the event-time grid.",
            "Absence must be interpreted with event density, stream support, state controls, and path-sampling design.",
            "This materialization is Polars-native and writes Arrow Parquet; no SQL engine is used.",
        ],
    }
    summary_path = args.out_dir / "interface_timestamp_mining_manifest.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    shutil.rmtree(staging)
    emit_status("interface_timestamp_mining_complete", manifest=str(summary_path), counts=counts)


if __name__ == "__main__":
    main()
