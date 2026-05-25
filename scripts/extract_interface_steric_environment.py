#!/usr/bin/env python3
"""Extract receptor-side steric shells around critical GLP-1R interfaces."""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import TypeAlias, cast

import polars as pl
from scipy.spatial import cKDTree  # type: ignore[import-untyped]


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from prism_dstw.io import write_provenance_parquet


Coordinate: TypeAlias = tuple[float, float, float]

TRACK0_DIR = REPO_ROOT / "campaigns/glp1r_aleniglipron/track_0_manual_emulation"
N80_DIR = REPO_ROOT / "campaigns/glp1r_aleniglipron/integrated_spike_events/n80_full_scale"
DEFAULT_GRID_MAPPING = TRACK0_DIR / "grid_coordinate_mapping.json"
DEFAULT_BINDING_SITE = TRACK0_DIR / "binding_site_reference.json"
DEFAULT_SIGNAL_GRID = N80_DIR / "signal_grid_variance_channel.parquet"
DEFAULT_RISK_MAP = N80_DIR / "receptor_durability_risk_map.parquet"
DEFAULT_MAPPING = REPO_ROOT / "campaigns/glp1r_aleniglipron/topology/residue_index_mapping_matrix.parquet"
DEFAULT_ENVIRONMENT = TRACK0_DIR / "interface_steric_environment.parquet"
DEFAULT_SUMMARY = TRACK0_DIR / "interface_steric_summary.parquet"

SHELL_RADIUS_ANGSTROM = 6.0
VARIANCE_CLASSES = (
    "stable_occupied",
    "thermally_destabilized",
    "thermally_activated",
    "void",
)


@dataclass(frozen=True)
class GridGeometry:
    condition_id: str
    grid_dim: int
    origin: Coordinate
    spacing: float


@dataclass(frozen=True)
class CriticalEdge:
    edge_id: str
    condition_id: str
    edge_label: str
    edge_class: str
    edge_from_residue: int
    edge_to_residue: int
    durability_risk_score_raw: float
    atom_coordinates: tuple[Coordinate, ...]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--grid-mapping", type=Path, default=DEFAULT_GRID_MAPPING)
    parser.add_argument("--binding-site", type=Path, default=DEFAULT_BINDING_SITE)
    parser.add_argument("--signal-grid", type=Path, default=DEFAULT_SIGNAL_GRID)
    parser.add_argument("--risk-map", type=Path, default=DEFAULT_RISK_MAP)
    parser.add_argument("--mapping", type=Path, default=DEFAULT_MAPPING)
    parser.add_argument("--environment-out", type=Path, default=DEFAULT_ENVIRONMENT)
    parser.add_argument("--summary-out", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--shell-radius", type=float, default=SHELL_RADIUS_ANGSTROM)
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def load_json_object(path: Path) -> dict[str, object]:
    loaded = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        raise ValueError(f"{path} did not decode to a JSON object")
    return loaded


def json_object(value: object, label: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise ValueError(f"{label} is not an object")
    return cast(dict[str, object], value)


def json_list(value: object, label: str) -> list[object]:
    if not isinstance(value, list):
        raise ValueError(f"{label} is not a list")
    return value


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


def coordinate_from_json(value: object, label: str) -> Coordinate:
    raw = json_list(value, label)
    if len(raw) != 3:
        raise ValueError(f"{label} must have exactly three coordinates")
    return (
        as_float(raw[0], f"{label}[0]"),
        as_float(raw[1], f"{label}[1]"),
        as_float(raw[2], f"{label}[2]"),
    )


def load_geometries(path: Path) -> dict[str, GridGeometry]:
    payload = load_json_object(path)
    raw_conditions = json_object(payload.get("conditions"), f"{path}:conditions")
    geometries: dict[str, GridGeometry] = {}
    for condition_id, raw_geometry in raw_conditions.items():
        geometry = json_object(raw_geometry, f"{path}:conditions[{condition_id}]")
        origin = coordinate_from_json(
            geometry.get("origin_xyz_angstrom"),
            f"{path}:conditions[{condition_id}].origin_xyz_angstrom",
        )
        grid_dim = as_int(geometry["grid_dim"], "grid_dim")
        geometries[condition_id] = GridGeometry(
            condition_id=condition_id,
            grid_dim=grid_dim,
            origin=origin,
            spacing=as_float(geometry["spacing_angstrom"], "spacing_angstrom"),
        )
    return geometries


def load_edges(path: Path) -> list[CriticalEdge]:
    payload = load_json_object(path)
    edges: list[CriticalEdge] = []
    for raw_edge in json_list(payload.get("critical_edges"), f"{path}:critical_edges"):
        edge = json_object(raw_edge, f"{path}:critical_edges[]")
        atom_coords: list[Coordinate] = []
        for side in ("from_atom_coordinates", "to_atom_coordinates"):
            for raw_atom in json_list(edge.get(side), f"{path}:{side}"):
                atom = json_object(raw_atom, f"{path}:{side}[]")
                atom_coords.append(coordinate_from_json(atom.get("xyz_angstrom"), f"{path}:{side}.xyz"))
        if not atom_coords:
            raise ValueError(f"{edge['edge_id']} has no atom coordinates")
        edges.append(
            CriticalEdge(
                edge_id=str(edge["edge_id"]),
                condition_id=str(edge["condition_id"]),
                edge_label=str(edge["edge_label"]),
                edge_class=str(edge["edge_class"]),
                edge_from_residue=as_int(edge["edge_from_residue"], "edge_from_residue"),
                edge_to_residue=as_int(edge["edge_to_residue"], "edge_to_residue"),
                durability_risk_score_raw=as_float(edge["durability_risk_score_raw"], "durability_risk_score_raw"),
                atom_coordinates=tuple(atom_coords),
            )
        )
    return edges


def center_coordinate(index: int, origin: float, spacing: float) -> float:
    return origin + (float(index) + 0.5) * spacing


def voxel_idx(ix: int, iy: int, iz: int, grid_dim: int) -> int:
    return iz * grid_dim * grid_dim + iy * grid_dim + ix


def clipped_index_range(low: float, high: float, origin: float, spacing: float, grid_dim: int) -> range:
    start = max(0, math.ceil((low - origin) / spacing - 0.5))
    stop = min(grid_dim - 1, math.floor((high - origin) / spacing - 0.5))
    if stop < start:
        return range(0)
    return range(start, stop + 1)


def candidate_rows_for_edge(edge: CriticalEdge, geometry: GridGeometry, shell_radius: float) -> list[dict[str, int | float | str]]:
    xs = [coord[0] for coord in edge.atom_coordinates]
    ys = [coord[1] for coord in edge.atom_coordinates]
    zs = [coord[2] for coord in edge.atom_coordinates]
    x_range = clipped_index_range(min(xs) - shell_radius, max(xs) + shell_radius, geometry.origin[0], geometry.spacing, geometry.grid_dim)
    y_range = clipped_index_range(min(ys) - shell_radius, max(ys) + shell_radius, geometry.origin[1], geometry.spacing, geometry.grid_dim)
    z_range = clipped_index_range(min(zs) - shell_radius, max(zs) + shell_radius, geometry.origin[2], geometry.spacing, geometry.grid_dim)
    centers: list[Coordinate] = []
    index_triplets: list[tuple[int, int, int]] = []
    for iz in z_range:
        center_z = center_coordinate(iz, geometry.origin[2], geometry.spacing)
        for iy in y_range:
            center_y = center_coordinate(iy, geometry.origin[1], geometry.spacing)
            for ix in x_range:
                centers.append(
                    (
                        center_coordinate(ix, geometry.origin[0], geometry.spacing),
                        center_y,
                        center_z,
                    )
                )
                index_triplets.append((ix, iy, iz))
    if not centers:
        return []
    tree = cKDTree(edge.atom_coordinates)
    distances, _ = tree.query(centers, k=1, distance_upper_bound=shell_radius)
    rows: list[dict[str, int | float | str]] = []
    for coord, indices, distance in zip(centers, index_triplets, distances, strict=True):
        if not math.isfinite(float(distance)) or float(distance) > shell_radius:
            continue
        ix, iy, iz = indices
        rows.append(
            {
                "edge_id": edge.edge_id,
                "condition_id": edge.condition_id,
                "edge_label": edge.edge_label,
                "edge_class": edge.edge_class,
                "edge_from_residue": edge.edge_from_residue,
                "edge_to_residue": edge.edge_to_residue,
                "durability_risk_score_raw": edge.durability_risk_score_raw,
                "voxel_idx": voxel_idx(ix, iy, iz, geometry.grid_dim),
                "x_idx": ix,
                "y_idx": iy,
                "z_idx": iz,
                "center_x_angstrom": coord[0],
                "center_y_angstrom": coord[1],
                "center_z_angstrom": coord[2],
                "distance_to_nearest_edge_atom_angstrom": float(distance),
            }
        )
    return rows


def build_candidate_shells(
    edges: list[CriticalEdge],
    geometries: dict[str, GridGeometry],
    shell_radius: float,
) -> pl.DataFrame:
    rows: list[dict[str, int | float | str]] = []
    for edge in edges:
        geometry = geometries.get(edge.condition_id)
        if geometry is None:
            raise ValueError(f"missing grid geometry for {edge.condition_id}")
        rows.extend(candidate_rows_for_edge(edge, geometry, shell_radius))
    if not rows:
        raise ValueError("no steric shell voxels were generated")
    return pl.DataFrame(rows).with_columns(
        [
            pl.col("edge_from_residue").cast(pl.UInt32),
            pl.col("edge_to_residue").cast(pl.UInt32),
            pl.col("voxel_idx").cast(pl.UInt64),
            pl.col("x_idx").cast(pl.UInt32),
            pl.col("y_idx").cast(pl.UInt32),
            pl.col("z_idx").cast(pl.UInt32),
        ]
    )


def enrich_environment(candidates: pl.DataFrame, signal_grid: Path) -> pl.DataFrame:
    signal = pl.scan_parquet(signal_grid).select(
        [
            "condition_id",
            pl.col("voxel_idx").cast(pl.UInt64),
            "variance_class",
            "hit_count_cold_mean",
            "hit_count_warm_mean",
            "cold_stream_count",
            "warm_stream_count",
        ]
    )
    return (
        candidates.lazy()
        .join(signal, on=["condition_id", "voxel_idx"], how="left")
        .with_columns(
            [
                pl.col("variance_class").fill_null("void"),
                pl.col("hit_count_cold_mean").fill_null(0.0),
                pl.col("hit_count_warm_mean").fill_null(0.0),
                pl.col("cold_stream_count").fill_null(0).cast(pl.UInt64),
                pl.col("warm_stream_count").fill_null(0).cast(pl.UInt64),
                (pl.col("hit_count_warm_mean") - pl.col("hit_count_cold_mean")).alias(
                    "hit_count_delta"
                ),
            ]
        )
        .sort(["edge_id", "voxel_idx"])
        .collect()
    )


def summary_for_environment(environment: pl.DataFrame) -> pl.DataFrame:
    metadata = (
        environment.select(
            [
                "edge_id",
                "condition_id",
                "edge_label",
                "edge_class",
                "edge_from_residue",
                "edge_to_residue",
                "durability_risk_score_raw",
            ]
        )
        .unique()
        .sort("edge_id")
    )
    rows: list[dict[str, int | float | str | None]] = []
    for edge_row in metadata.to_dicts():
        edge_id = str(edge_row["edge_id"])
        edge_frame = environment.filter(pl.col("edge_id") == edge_id)
        row: dict[str, int | float | str | None] = {
            "edge_id": edge_id,
            "condition_id": str(edge_row["condition_id"]),
            "edge_label": str(edge_row["edge_label"]),
            "edge_class": str(edge_row["edge_class"]),
            "edge_from_residue": as_int(edge_row["edge_from_residue"], "edge_from_residue"),
            "edge_to_residue": as_int(edge_row["edge_to_residue"], "edge_to_residue"),
            "durability_risk_score_raw": as_float(edge_row["durability_risk_score_raw"], "durability_risk_score_raw"),
            "n_voxels_total": edge_frame.height,
        }
        for variance_class in VARIANCE_CLASSES:
            zone = edge_frame.filter(pl.col("variance_class") == variance_class)
            prefix = variance_class
            row[f"n_voxels_{prefix}"] = zone.height
            row[f"{prefix}_centroid_x_angstrom"] = None if zone.is_empty() else as_float(zone["center_x_angstrom"].mean(), "center_x_mean")
            row[f"{prefix}_centroid_y_angstrom"] = None if zone.is_empty() else as_float(zone["center_y_angstrom"].mean(), "center_y_mean")
            row[f"{prefix}_centroid_z_angstrom"] = None if zone.is_empty() else as_float(zone["center_z_angstrom"].mean(), "center_z_mean")
            row[f"{prefix}_mean_hit_count_delta"] = None if zone.is_empty() else as_float(zone["hit_count_delta"].mean(), "hit_count_delta_mean")
        rows.append(row)
    return pl.DataFrame(rows).with_columns(
        [
            pl.col("edge_from_residue").cast(pl.UInt32),
            pl.col("edge_to_residue").cast(pl.UInt32),
            pl.col("n_voxels_total").cast(pl.UInt32),
            *[pl.col(f"n_voxels_{variance_class}").cast(pl.UInt32) for variance_class in VARIANCE_CLASSES],
        ]
    )


def main() -> int:
    args = parse_args()
    logging.basicConfig(level=str(args.log_level).upper(), format="%(levelname)s %(message)s")
    geometries = load_geometries(args.grid_mapping)
    edges = load_edges(args.binding_site)
    candidates = build_candidate_shells(edges, geometries, float(args.shell_radius))
    environment = enrich_environment(candidates, args.signal_grid)
    summary = summary_for_environment(environment)
    source_parquets = [args.signal_grid, args.risk_map, args.mapping]
    write_provenance_parquet(
        environment,
        args.environment_out,
        producer_script=Path(__file__),
        source_parquets=source_parquets,
        schema_version="interface_steric_environment.v1",
        pipeline_stage="track0_receptor_side_steric_environment",
        partition_keys=["condition_id", "edge_id"],
        ledger_parameters={
            "shell_radius_angstrom": float(args.shell_radius),
            "grid_mapping_json": args.grid_mapping.as_posix(),
            "binding_site_reference_json": args.binding_site.as_posix(),
            "kd_tree_filter": "scipy.spatial.cKDTree nearest-edge-atom distance <= shell_radius",
        },
        ledger_output_value={"rows": environment.height, "output_path": args.environment_out.as_posix()},
        repo_root=REPO_ROOT,
    )
    write_provenance_parquet(
        summary,
        args.summary_out,
        producer_script=Path(__file__),
        source_parquets=source_parquets,
        schema_version="interface_steric_summary.v1",
        pipeline_stage="track0_receptor_side_steric_summary",
        partition_keys=["condition_id", "edge_id"],
        ledger_parameters={
            "shell_radius_angstrom": float(args.shell_radius),
            "summary_variance_classes": list(VARIANCE_CLASSES),
        },
        ledger_output_value={"rows": summary.height, "output_path": args.summary_out.as_posix()},
        repo_root=REPO_ROOT,
    )
    logging.info("wrote %s rows to %s", environment.height, args.environment_out)
    logging.info("wrote %s rows to %s", summary.height, args.summary_out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
