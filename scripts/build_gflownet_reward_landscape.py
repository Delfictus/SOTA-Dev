#!/usr/bin/env python3
"""Build Track A TSO and bridge-anchor boundaries for GFlowNet rewards."""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import polars as pl


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from prism_dstw.io import sha256_path, write_provenance_parquet
from prism_dstw.ontology import EpistemicClass, epistemic_metadata


CAMPAIGN_DIR = REPO_ROOT / "campaigns/glp1r_aleniglipron"
N80_DIR = CAMPAIGN_DIR / "integrated_spike_events/n80_full_scale"
TRACK0_DIR = CAMPAIGN_DIR / "track_0_manual_emulation"
TRACK_A_DIR = CAMPAIGN_DIR / "track_a_generative"
DEFAULT_SIGNAL_GRID = N80_DIR / "signal_grid_variance_channel.parquet"
DEFAULT_PATHWAY_NODES = N80_DIR / "translation_pathway_nodes.parquet"
DEFAULT_SHEAR_FIELD = N80_DIR / "shear_stress_field.parquet"
DEFAULT_GRID_MAPPING = TRACK0_DIR / "grid_coordinate_mapping.json"
DEFAULT_OUTPUT = TRACK_A_DIR / "gflownet_tso_bridge_boundaries.parquet"
NO_FLY_RADIUS_A = 4.0
EPSILON = 1.0e-9


@dataclass(frozen=True)
class GridGeometry:
    condition_id: str
    grid_dim: int
    origin_x: float
    origin_y: float
    origin_z: float
    spacing: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--signal-grid", type=Path, default=DEFAULT_SIGNAL_GRID)
    parser.add_argument("--pathway-nodes", type=Path, default=DEFAULT_PATHWAY_NODES)
    parser.add_argument("--shear-field", type=Path, default=DEFAULT_SHEAR_FIELD)
    parser.add_argument("--grid-mapping", type=Path, default=DEFAULT_GRID_MAPPING)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--no-fly-radius-A", type=float, default=NO_FLY_RADIUS_A)
    return parser.parse_args()


def as_float(value: object, label: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be numeric, got bool")
    if isinstance(value, int | float | str):
        return float(value)
    raise ValueError(f"{label} must be numeric")


def as_int(value: object, label: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be an integer, got bool")
    if isinstance(value, int | float | str):
        return int(value)
    raise ValueError(f"{label} must be an integer")


def load_json(path: Path) -> dict[str, object]:
    loaded = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        raise ValueError(f"{path} did not decode to a JSON object")
    return cast(dict[str, object], loaded)


def json_object(value: object, label: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object")
    return cast(dict[str, object], value)


def load_geometries(path: Path) -> dict[str, GridGeometry]:
    payload = load_json(path)
    conditions = json_object(payload.get("conditions"), "conditions")
    geometries: dict[str, GridGeometry] = {}
    for condition_id, raw_geometry in conditions.items():
        geometry = json_object(raw_geometry, f"conditions[{condition_id}]")
        origin = geometry.get("origin_xyz_angstrom")
        if not isinstance(origin, list) or len(origin) != 3:
            raise ValueError(f"conditions[{condition_id}].origin_xyz_angstrom must have three values")
        geometries[str(condition_id)] = GridGeometry(
            condition_id=str(condition_id),
            grid_dim=as_int(geometry["grid_dim"], "grid_dim"),
            origin_x=as_float(origin[0], "origin_x"),
            origin_y=as_float(origin[1], "origin_y"),
            origin_z=as_float(origin[2], "origin_z"),
            spacing=as_float(geometry["spacing_angstrom"], "spacing_angstrom"),
        )
    return geometries


def voxel_idx(x_idx: int, y_idx: int, z_idx: int, grid_dim: int) -> int:
    return z_idx * grid_dim * grid_dim + y_idx * grid_dim + x_idx


def bridge_anchor_rows(pathway_nodes: Path, geometries: dict[str, GridGeometry], radius_a: float) -> pl.DataFrame:
    nodes = (
        pl.scan_parquet(pathway_nodes)
        .sort(["pathway_rank", "wire_score"], descending=[False, True])
        .head(10)
        .collect()
    )
    rows: list[dict[str, object]] = []
    for row in nodes.iter_rows(named=True):
        condition_id = str(row["condition_id"])
        geometry = geometries.get(condition_id)
        if geometry is None:
            continue
        center_x = as_int(row["x_idx"], "x_idx")
        center_y = as_int(row["y_idx"], "y_idx")
        center_z = as_int(row["z_idx"], "z_idx")
        radius_idx = max(int(math.ceil(radius_a / geometry.spacing)), 1)
        for dz in range(-radius_idx, radius_idx + 1):
            for dy in range(-radius_idx, radius_idx + 1):
                for dx in range(-radius_idx, radius_idx + 1):
                    x_idx = center_x + dx
                    y_idx = center_y + dy
                    z_idx = center_z + dz
                    if (
                        x_idx < 0
                        or y_idx < 0
                        or z_idx < 0
                        or x_idx >= geometry.grid_dim
                        or y_idx >= geometry.grid_dim
                        or z_idx >= geometry.grid_dim
                    ):
                        continue
                    distance_a = geometry.spacing * math.sqrt(float(dx * dx + dy * dy + dz * dz))
                    if distance_a > radius_a:
                        continue
                    rows.append(
                        {
                            "campaign_id": "glp1r_aleniglipron",
                            "condition_id": condition_id,
                            "voxel_idx": voxel_idx(x_idx, y_idx, z_idx, geometry.grid_dim),
                            "x_idx": x_idx,
                            "y_idx": y_idx,
                            "z_idx": z_idx,
                            "boundary_class": "bridge_anchor_no_fly_zone",
                            "anchor_residue_idx": as_int(row["residue_idx"], "residue_idx"),
                            "anchor_residue_name": str(row["residue_name"]),
                            "anchor_pathway_rank": as_int(row["pathway_rank"], "pathway_rank"),
                            "distance_to_bridge_anchor_A": distance_a,
                        }
                    )
    if not rows:
        return pl.DataFrame(
            schema={
                "campaign_id": pl.String,
                "condition_id": pl.String,
                "voxel_idx": pl.Int64,
                "x_idx": pl.Int64,
                "y_idx": pl.Int64,
                "z_idx": pl.Int64,
                "boundary_class": pl.String,
                "anchor_residue_idx": pl.Int64,
                "anchor_residue_name": pl.String,
                "anchor_pathway_rank": pl.Int64,
                "distance_to_bridge_anchor_A": pl.Float64,
            }
        )
    return pl.DataFrame(rows)


def normalization_constants(signal_grid: Path) -> tuple[float, float]:
    signal = pl.scan_parquet(signal_grid).with_columns(
        (pl.col("hit_count_warm_mean") - pl.col("hit_count_cold_mean")).alias("hit_count_delta")
    )
    row = (
        signal.filter(pl.col("variance_class") != "void")
        .select(
            [
                pl.col("hit_count_cold_mean").quantile(0.95).alias("cold_q95"),
                pl.col("hit_count_delta").abs().quantile(0.95).alias("delta_q95"),
            ]
        )
        .collect()
        .row(0, named=True)
    )
    return max(as_float(row["cold_q95"], "cold_q95"), 1.0), max(as_float(row["delta_q95"], "delta_q95"), 1.0)


def geometry_frame(geometries: dict[str, GridGeometry]) -> pl.DataFrame:
    return pl.DataFrame(
        [
            {
                "condition_id": item.condition_id,
                "origin_x": item.origin_x,
                "origin_y": item.origin_y,
                "origin_z": item.origin_z,
                "spacing_A": item.spacing,
            }
            for item in geometries.values()
        ]
    )


def shear_frame(shear_field: Path) -> pl.LazyFrame:
    shear = (
        pl.scan_parquet(shear_field)
        .select(
            [
                "condition_id",
                pl.col("voxel_idx").cast(pl.UInt64),
                pl.col("shear_stress").abs().alias("shear_abs"),
            ]
        )
    )
    thresholds = shear.group_by("condition_id").agg(pl.col("shear_abs").quantile(0.90).alias("shear_p90"))
    return shear.join(thresholds, on="condition_id", how="left").with_columns(
        (pl.col("shear_abs") / pl.max_horizontal(pl.col("shear_p90"), pl.lit(EPSILON))).alias("sigma_shear")
    )


def boundary_frame(
    signal_grid: Path,
    shear_field: Path,
    pathway_nodes: Path,
    grid_mapping: Path,
    radius_a: float,
) -> pl.LazyFrame:
    geometries = load_geometries(grid_mapping)
    cold_q95, delta_q95 = normalization_constants(signal_grid)
    geometry_lf = geometry_frame(geometries).lazy()
    shear_lf = shear_frame(shear_field)
    signal = (
        pl.scan_parquet(signal_grid)
        .with_columns(
            [
                pl.col("voxel_idx").cast(pl.UInt64),
                (pl.col("hit_count_warm_mean") - pl.col("hit_count_cold_mean")).alias("hit_count_delta"),
            ]
        )
        .join(geometry_lf, on="condition_id", how="left")
        .join(shear_lf, on=["condition_id", "voxel_idx"], how="left")
        .with_columns(
            [
                (pl.col("origin_x") + (pl.col("x_idx").cast(pl.Float64) + 0.5) * pl.col("spacing_A")).alias("center_x_A"),
                (pl.col("origin_y") + (pl.col("y_idx").cast(pl.Float64) + 0.5) * pl.col("spacing_A")).alias("center_y_A"),
                (pl.col("origin_z") + (pl.col("z_idx").cast(pl.Float64) + 0.5) * pl.col("spacing_A")).alias("center_z_A"),
                pl.col("sigma_shear").fill_null(0.0),
            ]
        )
    )
    tso = (
        signal.filter(pl.col("variance_class").is_in(["thermally_activated", "stable_occupied"]))
        .with_columns(
            [
                pl.when(pl.col("variance_class") == "thermally_activated")
                .then(pl.lit("reward_tso"))
                .otherwise(pl.lit("penalty_tso"))
                .alias("boundary_class"),
                pl.when(pl.col("variance_class") == "thermally_activated")
                .then(pl.col("hit_count_delta").clip(0.0, None) / pl.lit(delta_q95))
                .otherwise(pl.lit(0.0))
                .alias("pi_complement"),
                pl.when(pl.col("variance_class") == "stable_occupied")
                .then(pl.col("hit_count_cold_mean") / pl.lit(cold_q95))
                .otherwise(pl.lit(0.0))
                .alias("pi_clash"),
                pl.lit(None, dtype=pl.Int64).alias("anchor_residue_idx"),
                pl.lit(None, dtype=pl.String).alias("anchor_residue_name"),
                pl.lit(None, dtype=pl.Int64).alias("anchor_pathway_rank"),
                pl.lit(None, dtype=pl.Float64).alias("distance_to_bridge_anchor_A"),
            ]
        )
    )

    no_fly_base = bridge_anchor_rows(pathway_nodes, geometries, radius_a).lazy().with_columns(
        pl.col("voxel_idx").cast(pl.UInt64)
    )
    no_fly = (
        no_fly_base.join(
            signal.select(
                [
                    "condition_id",
                    "voxel_idx",
                    "variance_class",
                    "hit_count_cold_mean",
                    "hit_count_warm_mean",
                    "hit_count_delta",
                    "cold_stream_count",
                    "warm_stream_count",
                    "center_x_A",
                    "center_y_A",
                    "center_z_A",
                    "sigma_shear",
                ]
            ),
            on=["condition_id", "voxel_idx"],
            how="left",
        )
        .with_columns(
            [
                pl.lit(0.0).alias("pi_complement"),
                (pl.col("hit_count_cold_mean").fill_null(0.0) / pl.lit(cold_q95)).alias("pi_clash"),
                pl.col("sigma_shear").fill_null(0.0),
            ]
        )
    )
    selected = [
        "campaign_id",
        "condition_id",
        "voxel_idx",
        "x_idx",
        "y_idx",
        "z_idx",
        "center_x_A",
        "center_y_A",
        "center_z_A",
        "boundary_class",
        "variance_class",
        "hit_count_cold_mean",
        "hit_count_warm_mean",
        "hit_count_delta",
        "pi_complement",
        "pi_clash",
        "sigma_shear",
        "anchor_residue_idx",
        "anchor_residue_name",
        "anchor_pathway_rank",
        "distance_to_bridge_anchor_A",
    ]
    return pl.concat([tso.select(selected), no_fly.select(selected)], how="diagonal_relaxed").with_columns(
        [
            pl.lit(EpistemicClass.INFERRED.value).alias("Epistemic_Class"),
            pl.lit(EpistemicClass.INFERRED.value).alias("epistemic_class"),
        ]
    )


def main() -> int:
    args = parse_args()
    frame = boundary_frame(
        signal_grid=args.signal_grid,
        shear_field=args.shear_field,
        pathway_nodes=args.pathway_nodes,
        grid_mapping=args.grid_mapping,
        radius_a=args.no_fly_radius_A,
    )
    written = write_provenance_parquet(
        frame,
        args.output,
        producer_script=Path(__file__),
        source_parquets=[args.signal_grid, args.pathway_nodes, args.shear_field],
        schema_version="gflownet_tso_bridge_boundaries.v1",
        pipeline_stage="track_a_gflownet_reward_landscape",
        partition_keys=["condition_id", "boundary_class", "voxel_idx"],
        extra_metadata=epistemic_metadata(EpistemicClass.INFERRED),
        ledger_parameters={
            "grid_mapping_sha256": sha256_path(args.grid_mapping),
            "no_fly_radius_A": args.no_fly_radius_A,
            "reward_tso_variance_class": "thermally_activated",
            "penalty_tso_variance_class": "stable_occupied",
        },
    )
    rows = pl.scan_parquet(written).select(pl.len()).collect().item()
    sys.stdout.write(f"wrote {written.relative_to(REPO_ROOT)} rows={rows}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
