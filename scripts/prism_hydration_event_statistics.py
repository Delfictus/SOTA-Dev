#!/usr/bin/env python3
"""Compute ontology-specific hydration statistics from integrated spike events.

This stage uses raw spike-event `water_density` and `wd_change` fields. It does
not reuse or reinterpret DSTW `sigma_hydration_sq`, which is a separate
Path-B centroid-spread proxy in the current campaign.
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

import polars as pl
import pyarrow.parquet as pq

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from prism_dstw.io import write_provenance_parquet


def emit_status(event: str, **fields: object) -> None:
    payload = {"event": event, **fields}
    sys.stderr.write(json.dumps(payload, sort_keys=True) + "\n")


def expand_patterns(patterns: list[str]) -> list[str]:
    out: list[str] = []
    for pattern in patterns:
        matches = sorted(glob.glob(pattern))
        out.extend(matches or [pattern])
    return sorted(dict.fromkeys(out))


def validate_paths(paths: list[str]) -> None:
    missing = [p for p in paths if not Path(p).exists()]
    if missing:
        raise SystemExit(f"Missing required Parquet inputs: {missing}")


def base_event_scan(paths: list[str]) -> pl.LazyFrame:
    return pl.scan_parquet(paths).select(
        [
            pl.col("campaign_id").cast(pl.Utf8),
            pl.col("run_label").cast(pl.Utf8),
            pl.col("structure_id").cast(pl.Utf8),
            pl.col("stream_id").cast(pl.Int16),
            pl.col("timestep").cast(pl.Int64),
            pl.col("physical_time_ps").cast(pl.Float64),
            pl.col("nearest_site_id").cast(pl.Utf8),
            pl.col("inside_nearest_site_radius").cast(pl.Boolean),
            pl.col("primary_uniprot_residue").cast(pl.Int64),
            pl.col("water_density").cast(pl.Float64),
            pl.col("wd_change").cast(pl.Float64),
            pl.col("intensity").cast(pl.Float64),
        ]
    )


def base_hit_scan(paths: list[str]) -> pl.LazyFrame:
    return pl.scan_parquet(paths).select(
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
            pl.col("water_density").cast(pl.Float64),
            pl.col("wd_change").cast(pl.Float64),
            pl.col("intensity").cast(pl.Float64),
        ]
    )


def hydration_aggs(count_name: str) -> list[pl.Expr]:
    return [
        pl.len().cast(pl.UInt64).alias(count_name),
        pl.col("water_density").mean().alias("mean_water_density"),
        pl.col("water_density").std(ddof=0).alias("std_water_density"),
        pl.col("water_density").min().alias("min_water_density"),
        pl.col("water_density").max().alias("max_water_density"),
        pl.col("wd_change").mean().alias("mean_wd_change"),
        pl.col("wd_change").std(ddof=0).alias("std_wd_change"),
        pl.col("wd_change").min().alias("min_wd_change"),
        pl.col("wd_change").max().alias("max_wd_change"),
        pl.col("intensity").mean().alias("mean_intensity"),
        pl.col("intensity").std(ddof=0).alias("std_intensity"),
    ]


def parquet_rows(path: Path) -> int:
    return int(pq.read_metadata(path).num_rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--event-parquet", action="append", required=True, help="Path or glob. Repeatable.")
    parser.add_argument("--hit-parquet", action="append", required=True, help="Path or glob. Repeatable.")
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()

    event_paths = expand_patterns(args.event_parquet)
    hit_paths = expand_patterns(args.hit_parquet)
    validate_paths(event_paths + hit_paths)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    source_parquets = [Path(p) for p in event_paths + hit_paths]

    events = base_event_scan(event_paths)
    hits = base_hit_scan(hit_paths)

    event_time_path = args.out_dir / "event_time_hydration_bins.parquet"
    site_time_path = args.out_dir / "site_time_hydration_bins.parquet"
    residue_time_path = args.out_dir / "residue_time_hydration_bins.parquet"
    interface_time_path = args.out_dir / "interface_time_hydration_bins.parquet"
    summary_path = args.out_dir / "hydration_event_statistics_manifest.json"

    event_time = (
        events.group_by(["campaign_id", "run_label", "structure_id", "stream_id", "timestep"])
        .agg([pl.col("physical_time_ps").min().alias("physical_time_ps"), *hydration_aggs("event_count")])
        .sort(["run_label", "stream_id", "timestep"])
    )
    site_time = (
        events.group_by(
            [
                "campaign_id",
                "run_label",
                "structure_id",
                "stream_id",
                "timestep",
                "nearest_site_id",
                "inside_nearest_site_radius",
            ]
        )
        .agg([pl.col("physical_time_ps").min().alias("physical_time_ps"), *hydration_aggs("event_count")])
        .sort(["run_label", "stream_id", "timestep", "nearest_site_id", "inside_nearest_site_radius"])
    )
    residue_time = (
        events.filter(pl.col("primary_uniprot_residue") >= 0)
        .group_by(
            [
                "campaign_id",
                "run_label",
                "structure_id",
                "stream_id",
                "timestep",
                "primary_uniprot_residue",
            ]
        )
        .agg([pl.col("physical_time_ps").min().alias("physical_time_ps"), *hydration_aggs("event_count")])
        .sort(["run_label", "stream_id", "timestep", "primary_uniprot_residue"])
    )
    interface_time = (
        hits.group_by(["campaign_id", "run_label", "structure_id", "stream_id", "timestep", "interface_id"])
        .agg(
            [
                pl.col("physical_time_ps").min().alias("physical_time_ps"),
                pl.col("interface_class").drop_nulls().first().alias("interface_class"),
                pl.col("target_hinge_residue_index").drop_nulls().first().alias("target_hinge_residue_index"),
                pl.col("neighbor_residue_index").drop_nulls().first().alias("neighbor_residue_index"),
                pl.len().cast(pl.UInt64).alias("hit_count"),
                (pl.col("match_basis") == "residue_pair").sum().cast(pl.UInt64).alias("residue_pair_hit_count"),
                (pl.col("match_basis") == "residue_single").sum().cast(pl.UInt64).alias("residue_single_hit_count"),
                (pl.col("match_basis") == "site_shell").sum().cast(pl.UInt64).alias("site_shell_hit_count"),
                *hydration_aggs("event_count")[1:],
            ]
        )
        .sort(["run_label", "stream_id", "timestep", "interface_id"])
    )

    for lf, path in [
        (event_time, event_time_path),
        (site_time, site_time_path),
        (residue_time, residue_time_path),
        (interface_time, interface_time_path),
    ]:
        write_provenance_parquet(
            lf,
            path,
            producer_script=Path(__file__),
            source_parquets=source_parquets,
            schema_version="prism_hydration_event_statistics.v2.polars_native",
            pipeline_stage="hydration_event_statistics",
            partition_keys=["campaign_id", "run_label", "structure_id", "stream_id", "timestep"],
            extra_metadata={"lineage_scope": "hydration_event_statistics"},
        )

    counts = {
        "event_time_hydration_bins": parquet_rows(event_time_path),
        "site_time_hydration_bins": parquet_rows(site_time_path),
        "residue_time_hydration_bins": parquet_rows(residue_time_path),
        "interface_time_hydration_bins": parquet_rows(interface_time_path),
    }

    manifest = {
        "schema": "prism_hydration_event_statistics.v2.polars_native",
        "engine": "polars_native_arrow_parquet",
        "event_parquets": event_paths,
        "hit_parquets": hit_paths,
        "outputs": {
            "event_time_hydration_bins": str(event_time_path),
            "site_time_hydration_bins": str(site_time_path),
            "residue_time_hydration_bins": str(residue_time_path),
            "interface_time_hydration_bins": str(interface_time_path),
        },
        "counts": counts,
        "semantic_warnings": [
            "These statistics are computed from raw spike-event water_density and wd_change fields.",
            "They must not be averaged together with DSTW sigma_hydration_sq without an explicit projection model.",
            "Interface hydration rows summarize interface-hit events, not all events in the interface neighborhood.",
            "Each output is a separate ontology class and must not be pooled without an explicit projection model.",
        ],
    }
    summary_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    emit_status("hydration_event_statistics_complete", manifest=str(summary_path), counts=counts)


if __name__ == "__main__":
    main()
