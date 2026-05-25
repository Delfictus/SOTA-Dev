#!/usr/bin/env python3
"""Trace triple-validated allosteric translation pathway residues.

Derived formulas:
    load_directionality = abs(mean_signed_load) / max(mean_abs_load, epsilon).
    triple_evidence = top 50 percent mechanical load, top 50 percent absolute
        KCC temporal correlation, and coherence_class in {fiber_invariant, partially_coherent}.
    betweenness = dot(CA - pocket_centroid, lock_centroid - pocket_centroid)
        / ||lock_centroid - pocket_centroid||^2.
    wire_score = load_normalized * abs_kcc_normalized * coherence_score.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, TypeAlias, cast

import polars as pl
import structlog


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from prism_dstw.io import sha256_path, write_provenance_parquet
from prism_dstw.ontology import ResidueIdx


JsonObject: TypeAlias = dict[str, object]
Coordinate: TypeAlias = tuple[float, float, float]

N80_DIR = REPO_ROOT / "campaigns/glp1r_aleniglipron/integrated_spike_events/n80_full_scale"
TRACK0_DIR = REPO_ROOT / "campaigns/glp1r_aleniglipron/track_0_manual_emulation"
DEFAULT_MECHANICAL = N80_DIR / "mechanical_load_network.parquet"
DEFAULT_KCC = N80_DIR / "kcc_residue_fields.parquet"
DEFAULT_SHEAR_STRESS = N80_DIR / "shear_stress_field.parquet"
DEFAULT_RISK_MAP = N80_DIR / "receptor_durability_risk_map.parquet"
DEFAULT_COHERENCE = N80_DIR / "phase_manifold_coherence.parquet"
DEFAULT_MAPPING = REPO_ROOT / "campaigns/glp1r_aleniglipron/topology/residue_index_mapping_matrix.parquet"
DEFAULT_GRID_MAPPING = TRACK0_DIR / "grid_coordinate_mapping.json"
DEFAULT_OUTPUT = N80_DIR / "translation_pathway_nodes.parquet"
EPSILON = 1.0e-6


@dataclass(frozen=True)
class TopologyProducts:
    atom_to_residue: pl.DataFrame
    residue_coordinates: pl.DataFrame
    topology_hashes: dict[str, str]
    topology_files: dict[str, str]


@dataclass(frozen=True)
class GridGeometry:
    condition_id: str
    grid_dim: int
    origin: Coordinate
    spacing: float


class BoundLogger(Protocol):
    def info(self, event: str, **kwargs: object) -> None:
        """Emit a structured info event."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mechanical-load", type=Path, default=DEFAULT_MECHANICAL)
    parser.add_argument("--kcc", type=Path, default=DEFAULT_KCC)
    parser.add_argument("--shear-stress", type=Path, default=DEFAULT_SHEAR_STRESS)
    parser.add_argument("--risk-map", type=Path, default=DEFAULT_RISK_MAP)
    parser.add_argument("--coherence", type=Path, default=DEFAULT_COHERENCE)
    parser.add_argument("--mapping", type=Path, default=DEFAULT_MAPPING)
    parser.add_argument("--grid-mapping", type=Path, default=DEFAULT_GRID_MAPPING)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--final-nodes-per-condition", type=int, default=10)
    parser.add_argument("--min-betweenness", type=float, default=0.1)
    parser.add_argument("--max-betweenness", type=float, default=0.9)
    parser.add_argument("--max-lateral-A", type=float, default=15.0)
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def configure_logging(level: str) -> BoundLogger:
    structlog.configure(
        wrapper_class=structlog.make_filtering_bound_logger(getattr(logging, level.upper(), logging.INFO)),
        processors=[
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.KeyValueRenderer(),
        ],
    )
    return cast(BoundLogger, structlog.get_logger("generate_translation_pathway"))


def collect_streaming(lazy_frame: pl.LazyFrame) -> pl.DataFrame:
    return lazy_frame.collect(engine="streaming")


def require_columns(path: Path, required: set[str]) -> None:
    schema_names = set(pl.scan_parquet(path).collect_schema().names())
    missing = sorted(required - schema_names)
    if missing:
        raise ValueError(f"{path} is missing required columns: {missing}")


def first_existing_column(path: Path, options: tuple[str, ...]) -> str:
    schema_names = set(pl.scan_parquet(path).collect_schema().names())
    for option in options:
        if option in schema_names:
            return option
    raise ValueError(f"{path} has none of the expected columns: {options}")


def load_json_object(path: Path) -> JsonObject:
    loaded = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        raise ValueError(f"{path} did not decode to an object")
    return cast(JsonObject, loaded)


def json_object(value: object, label: str) -> JsonObject:
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be an object")
    return cast(JsonObject, value)


def json_list(value: object, label: str) -> list[object]:
    if not isinstance(value, list):
        raise ValueError(f"{label} must be a list")
    return value


def as_int(value: object, label: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be an integer, got bool")
    if isinstance(value, int | float | str):
        return int(value)
    raise ValueError(f"{label} must be an integer")


def as_float(value: object, label: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be numeric, got bool")
    if isinstance(value, int | float | str):
        return float(value)
    raise ValueError(f"{label} must be numeric")


def residue_value(value: object, label: str) -> int:
    return int(ResidueIdx(as_int(value, label)))


def float_list(value: object, label: str) -> list[float]:
    return [as_float(item, f"{label}[]") for item in json_list(value, label)]


def int_list(value: object, label: str) -> list[int]:
    return [as_int(item, f"{label}[]") for item in json_list(value, label)]


def topology_paths_from_grid_mapping(path: Path) -> dict[str, Path]:
    payload = load_json_object(path)
    raw_conditions = json_object(payload.get("conditions"), f"{path}:conditions")
    out: dict[str, Path] = {}
    for condition_id, raw_geometry in raw_conditions.items():
        geometry = json_object(raw_geometry, f"{path}:conditions[{condition_id}]")
        topology_path = Path(str(geometry["topology_path"]))
        if not topology_path.exists():
            raise FileNotFoundError(topology_path)
        out[str(condition_id)] = topology_path
    return out


def coordinate_at(positions: list[float], atom_idx: int) -> Coordinate:
    base = atom_idx * 3
    return (positions[base], positions[base + 1], positions[base + 2])


def coordinate_from_json(value: object, label: str) -> Coordinate:
    raw = json_list(value, label)
    if len(raw) != 3:
        raise ValueError(f"{label} must contain three coordinates")
    return (
        as_float(raw[0], f"{label}[0]"),
        as_float(raw[1], f"{label}[1]"),
        as_float(raw[2], f"{label}[2]"),
    )


def grid_geometries(path: Path) -> dict[str, GridGeometry]:
    payload = load_json_object(path)
    raw_conditions = json_object(payload.get("conditions"), f"{path}:conditions")
    out: dict[str, GridGeometry] = {}
    for condition_id, raw_geometry in raw_conditions.items():
        geometry = json_object(raw_geometry, f"{path}:conditions[{condition_id}]")
        out[str(condition_id)] = GridGeometry(
            condition_id=str(condition_id),
            grid_dim=as_int(geometry["grid_dim"], "grid_dim"),
            origin=coordinate_from_json(
                geometry.get("origin_xyz_angstrom"),
                f"{path}:conditions[{condition_id}].origin_xyz_angstrom",
            ),
            spacing=as_float(geometry["spacing_angstrom"], "spacing_angstrom"),
        )
    return out


def coordinate_to_voxel(coord: Coordinate, geometry: GridGeometry) -> tuple[int, int, int, int] | None:
    ix = math.trunc((coord[0] - geometry.origin[0]) / geometry.spacing)
    iy = math.trunc((coord[1] - geometry.origin[1]) / geometry.spacing)
    iz = math.trunc((coord[2] - geometry.origin[2]) / geometry.spacing)
    if ix < 0 or iy < 0 or iz < 0:
        return None
    if ix >= geometry.grid_dim or iy >= geometry.grid_dim or iz >= geometry.grid_dim:
        return None
    voxel_idx = iz * geometry.grid_dim * geometry.grid_dim + iy * geometry.grid_dim + ix
    return ix, iy, iz, voxel_idx


def topology_products(grid_mapping_path: Path) -> TopologyProducts:
    atom_rows: list[dict[str, int | str]] = []
    coord_rows: list[dict[str, int | float | str]] = []
    topology_hashes: dict[str, str] = {}
    topology_files: dict[str, str] = {}
    for condition_id, topology_path in topology_paths_from_grid_mapping(grid_mapping_path).items():
        topology = load_json_object(topology_path)
        residue_ids = int_list(topology.get("residue_ids"), f"{topology_path.name}:residue_ids")
        ca_indices = int_list(topology.get("ca_indices"), f"{topology_path.name}:ca_indices")
        positions = float_list(topology.get("positions"), f"{topology_path.name}:positions")
        topology_hashes[condition_id] = sha256_path(topology_path)
        topology_files[condition_id] = topology_path.name
        for atom_idx, residue_idx in enumerate(residue_ids):
            atom_rows.append(
                {
                    "condition_id": condition_id,
                    "atom_idx": atom_idx,
                    "residue_idx": residue_value(residue_idx, "residue_idx"),
                }
            )
        for residue_idx, ca_atom_idx in enumerate(ca_indices):
            ca = coordinate_at(positions, ca_atom_idx)
            coord_rows.append(
                {
                    "condition_id": condition_id,
                    "residue_idx": residue_value(residue_idx, "residue_idx"),
                    "ca_x": ca[0],
                    "ca_y": ca[1],
                    "ca_z": ca[2],
                }
            )
    return TopologyProducts(
        atom_to_residue=pl.DataFrame(atom_rows),
        residue_coordinates=pl.DataFrame(coord_rows),
        topology_hashes=topology_hashes,
        topology_files=topology_files,
    )


def residue_mapping_frame(mapping_path: Path) -> pl.LazyFrame:
    require_columns(mapping_path, {"condition_id", "residue_idx", "canonical_residue_label"})
    return pl.scan_parquet(mapping_path).select(
        [
            "condition_id",
            pl.col("residue_idx").cast(pl.Int32),
            pl.col("canonical_residue_label").alias("residue_name"),
        ]
    )


def endpoint_sets(risk_map_path: Path) -> pl.DataFrame:
    require_columns(risk_map_path, {"condition_id", "edge_from_residue", "edge_to_residue", "edge_class"})
    edges = pl.scan_parquet(risk_map_path).filter(pl.col("edge_class").is_in(["pocket_vector", "downstream_lock"]))
    return pl.concat(
        [
            edges.select(
                [
                    "condition_id",
                    pl.col("edge_class").alias("endpoint_class"),
                    pl.col("edge_from_residue").cast(pl.Int32).alias("residue_idx"),
                ]
            ),
            edges.select(
                [
                    "condition_id",
                    pl.col("edge_class").alias("endpoint_class"),
                    pl.col("edge_to_residue").cast(pl.Int32).alias("residue_idx"),
                ]
            ),
        ]
    ).unique().collect()


def residue_load_frame(mechanical_path: Path, atom_map: pl.DataFrame) -> pl.DataFrame:
    load_col = first_existing_column(mechanical_path, ("load_dot_product", "mechanical_load", "mean_load_dot_product"))
    atom_col = first_existing_column(mechanical_path, ("atom_index", "atom_idx"))
    require_columns(mechanical_path, {"condition_id", load_col, atom_col})
    return collect_streaming(
        pl.scan_parquet(mechanical_path)
        .select(
            [
                "condition_id",
                pl.col(atom_col).cast(pl.Int32).alias("atom_idx"),
                pl.col(load_col).cast(pl.Float64).alias("load_dot_product"),
            ]
        )
        .join(atom_map.lazy(), on=["condition_id", "atom_idx"], how="inner")
        .group_by(["condition_id", "residue_idx"])
        .agg(
            [
                pl.col("load_dot_product").abs().mean().alias("mean_abs_load"),
                pl.col("load_dot_product").mean().alias("mean_signed_load"),
                pl.col("load_dot_product").std().fill_null(0.0).alias("load_std"),
            ]
        )
        .with_columns(
            (
                pl.col("mean_signed_load").abs()
                / pl.max_horizontal([pl.col("mean_abs_load"), pl.lit(EPSILON)])
            )
            .clip(0.0, 1.0)
            .alias("load_directionality")
        )
    )


def kcc_frame(kcc_path: Path) -> pl.DataFrame:
    temporal_col = first_existing_column(kcc_path, ("temporal_correlation", "temporal_corr"))
    require_columns(kcc_path, {"condition_id", "residue_idx", temporal_col, "direction_score", "burst_motion"})
    return collect_streaming(
        pl.scan_parquet(kcc_path)
        .select(
            [
                "condition_id",
                pl.col("residue_idx").cast(pl.Int32),
                pl.col(temporal_col).cast(pl.Float64).alias("temporal_correlation"),
                pl.col("direction_score").cast(pl.Float64),
                pl.col("burst_motion").cast(pl.Float64),
            ]
        )
        .group_by(["condition_id", "residue_idx"])
        .agg(
            [
                pl.col("temporal_correlation").mean(),
                pl.col("direction_score").mean(),
                pl.col("burst_motion").mean(),
                pl.col("burst_motion").max().alias("max_burst_motion"),
            ]
        )
    )


def residue_shear_frame(shear_path: Path, grid_mapping_path: Path, coordinates: pl.DataFrame) -> pl.DataFrame:
    require_columns(
        shear_path,
        {"condition_id", "voxel_idx", "shear_stress", "shear_stress_max", "stream_count"},
    )
    geometries = grid_geometries(grid_mapping_path)
    rows: list[dict[str, int | float | str | bool]] = []
    for row in coordinates.iter_rows(named=True):
        condition_id = str(row["condition_id"])
        geometry = geometries.get(condition_id)
        if geometry is None:
            continue
        coord = (
            as_float(row["ca_x"], "ca_x"),
            as_float(row["ca_y"], "ca_y"),
            as_float(row["ca_z"], "ca_z"),
        )
        mapped = coordinate_to_voxel(coord, geometry)
        if mapped is None:
            rows.append(
                {
                    "condition_id": condition_id,
                    "residue_idx": residue_value(row["residue_idx"], "residue_idx"),
                    "voxel_idx": -1,
                    "x_idx": -1,
                    "y_idx": -1,
                    "z_idx": -1,
                    "voxel_in_bounds": False,
                }
            )
            continue
        ix, iy, iz, voxel_idx = mapped
        rows.append(
            {
                "condition_id": condition_id,
                "residue_idx": residue_value(row["residue_idx"], "residue_idx"),
                "voxel_idx": voxel_idx,
                "x_idx": ix,
                "y_idx": iy,
                "z_idx": iz,
                "voxel_in_bounds": True,
            }
        )
    residue_voxels = pl.DataFrame(rows)
    thresholds = (
        pl.scan_parquet(shear_path)
        .with_columns(pl.col("shear_stress").cast(pl.Float64).abs().alias("shear_stress_abs"))
        .group_by("condition_id")
        .agg(pl.col("shear_stress_abs").quantile(0.90).alias("shear_stress_abs_p90"))
    )
    shear = pl.scan_parquet(shear_path).select(
        [
            "condition_id",
            pl.col("voxel_idx").cast(pl.Int64),
            pl.col("shear_stress").cast(pl.Float64),
            pl.col("shear_stress").cast(pl.Float64).abs().alias("shear_stress_abs"),
            pl.col("shear_stress_max").cast(pl.Float64),
            pl.col("stream_count").cast(pl.UInt32).alias("shear_stream_count"),
        ]
    )
    return (
        residue_voxels.lazy()
        .with_columns(pl.col("voxel_idx").cast(pl.Int64))
        .join(shear, on=["condition_id", "voxel_idx"], how="left")
        .join(thresholds, on="condition_id", how="left")
        .with_columns(
            (
                pl.col("voxel_in_bounds")
                & (pl.col("shear_stress_abs") >= pl.col("shear_stress_abs_p90"))
            )
            .fill_null(False)
            .alias("structural_fault_line")
        )
        .collect()
    )


def centroid_frame(endpoints: pl.DataFrame, coordinates: pl.DataFrame) -> pl.DataFrame:
    return (
        endpoints.lazy()
        .join(coordinates.lazy(), on=["condition_id", "residue_idx"], how="inner")
        .group_by(["condition_id", "endpoint_class"])
        .agg(
            [
                pl.col("ca_x").mean().alias("centroid_x"),
                pl.col("ca_y").mean().alias("centroid_y"),
                pl.col("ca_z").mean().alias("centroid_z"),
            ]
        )
        .collect()
    )


def pathway_frame(
    residue_load: pl.DataFrame,
    kcc: pl.DataFrame,
    shear: pl.DataFrame,
    coherence_path: Path,
    endpoints: pl.DataFrame,
    coordinates: pl.DataFrame,
    mapping_path: Path,
    final_nodes_per_condition: int,
    min_betweenness: float,
    max_betweenness: float,
    max_lateral_a: float,
) -> pl.DataFrame:
    require_columns(
        coherence_path,
        {"condition_id", "residue_idx", "coherence_score", "coherence_class", "n_concordant_channels", "ch_kcc_coherence"},
    )
    endpoint_union = endpoints.select(["condition_id", "residue_idx"]).unique()
    centroids = centroid_frame(endpoints, coordinates)
    pocket_centroids = centroids.filter(pl.col("endpoint_class") == "pocket_vector").select(
        [
            "condition_id",
            pl.col("centroid_x").alias("pocket_x"),
            pl.col("centroid_y").alias("pocket_y"),
            pl.col("centroid_z").alias("pocket_z"),
        ]
    )
    lock_centroids = centroids.filter(pl.col("endpoint_class") == "downstream_lock").select(
        [
            "condition_id",
            pl.col("centroid_x").alias("lock_x"),
            pl.col("centroid_y").alias("lock_y"),
            pl.col("centroid_z").alias("lock_z"),
        ]
    )
    coherence = pl.scan_parquet(coherence_path).select(
        [
            "condition_id",
            pl.col("residue_idx").cast(pl.Int32),
            "coherence_score",
            "coherence_class",
            "n_concordant_channels",
            pl.col("ch_kcc_coherence").alias("kcc_coherence"),
        ]
    )
    return (
        residue_load.lazy()
        .join(endpoint_union.lazy(), on=["condition_id", "residue_idx"], how="anti")
        .join(kcc.lazy(), on=["condition_id", "residue_idx"], how="inner")
        .join(shear.lazy(), on=["condition_id", "residue_idx"], how="left")
        .join(coherence, on=["condition_id", "residue_idx"], how="inner")
        .join(coordinates.lazy(), on=["condition_id", "residue_idx"], how="inner")
        .join(pocket_centroids.lazy(), on="condition_id", how="inner")
        .join(lock_centroids.lazy(), on="condition_id", how="inner")
        .with_columns(
            [
                pl.col("mean_abs_load").median().over("condition_id").alias("load_threshold"),
                pl.col("temporal_correlation").abs().median().over("condition_id").alias("kcc_threshold"),
                pl.col("mean_abs_load").max().over("condition_id").alias("load_max"),
                pl.col("temporal_correlation").abs().max().over("condition_id").alias("kcc_abs_max"),
            ]
        )
        .with_columns(
            [
                (pl.col("mean_abs_load") / pl.max_horizontal([pl.col("load_max"), pl.lit(EPSILON)])).alias(
                    "load_normalized"
                ),
                (
                    pl.col("temporal_correlation").abs()
                    / pl.max_horizontal([pl.col("kcc_abs_max"), pl.lit(EPSILON)])
                ).alias("kcc_normalized"),
            ]
        )
        .with_columns(
                (
                    (pl.col("mean_abs_load") >= pl.col("load_threshold"))
                & (pl.col("temporal_correlation").abs() >= pl.col("kcc_threshold"))
                & pl.col("coherence_class").is_in(["fiber_invariant", "partially_coherent"])
            ).alias("triple_evidence")
        )
        .with_columns(
            [
                pl.col("structural_fault_line").fill_null(False),
                (pl.col("max_burst_motion").fill_null(0.0) > 1000.0).alias("violent_kinetic_node"),
            ]
        )
        .with_columns(
            [
                (pl.col("lock_x") - pl.col("pocket_x")).alias("axis_x"),
                (pl.col("lock_y") - pl.col("pocket_y")).alias("axis_y"),
                (pl.col("lock_z") - pl.col("pocket_z")).alias("axis_z"),
                (pl.col("ca_x") - pl.col("pocket_x")).alias("rel_x"),
                (pl.col("ca_y") - pl.col("pocket_y")).alias("rel_y"),
                (pl.col("ca_z") - pl.col("pocket_z")).alias("rel_z"),
            ]
        )
        .with_columns(
            [
                (
                    pl.col("axis_x") * pl.col("axis_x")
                    + pl.col("axis_y") * pl.col("axis_y")
                    + pl.col("axis_z") * pl.col("axis_z")
                ).alias("axis_norm_sq"),
                (
                    pl.col("rel_x") * pl.col("axis_x")
                    + pl.col("rel_y") * pl.col("axis_y")
                    + pl.col("rel_z") * pl.col("axis_z")
                ).alias("axis_dot"),
            ]
        )
        .with_columns(
            [
                pl.col("axis_norm_sq").sqrt().alias("pocket_to_lock_dist_A"),
                (pl.col("axis_dot") / pl.col("axis_norm_sq")).alias("betweenness"),
                (
                    (
                        (pl.col("ca_x") - pl.col("pocket_x")) * (pl.col("ca_x") - pl.col("pocket_x"))
                        + (pl.col("ca_y") - pl.col("pocket_y")) * (pl.col("ca_y") - pl.col("pocket_y"))
                        + (pl.col("ca_z") - pl.col("pocket_z")) * (pl.col("ca_z") - pl.col("pocket_z"))
                    )
                    .sqrt()
                    .alias("dist_to_pocket_centroid_A")
                ),
                (
                    (
                        (pl.col("ca_x") - pl.col("lock_x")) * (pl.col("ca_x") - pl.col("lock_x"))
                        + (pl.col("ca_y") - pl.col("lock_y")) * (pl.col("ca_y") - pl.col("lock_y"))
                        + (pl.col("ca_z") - pl.col("lock_z")) * (pl.col("ca_z") - pl.col("lock_z"))
                    )
                    .sqrt()
                    .alias("dist_to_lock_centroid_A")
                ),
            ]
        )
        .with_columns((pl.col("axis_dot") / pl.col("pocket_to_lock_dist_A")).alias("axial_projection_A"))
        .with_columns(
            (
                (
                    pl.col("dist_to_pocket_centroid_A") * pl.col("dist_to_pocket_centroid_A")
                    - pl.col("axial_projection_A") * pl.col("axial_projection_A")
                )
                .clip(0.0, None)
                .sqrt()
            ).alias("lateral_displacement_A")
        )
        .filter(
            pl.col("triple_evidence")
            & (pl.col("betweenness") >= min_betweenness)
            & (pl.col("betweenness") <= max_betweenness)
            & (pl.col("lateral_displacement_A") < max_lateral_a)
        )
        .with_columns((pl.col("load_normalized") * pl.col("kcc_normalized") * pl.col("coherence_score")).alias("wire_score"))
        .with_columns(pl.col("wire_score").rank("ordinal", descending=True).over("condition_id").alias("pathway_rank"))
        .filter(pl.col("pathway_rank") <= final_nodes_per_condition)
        .with_columns(pl.lit("triple_validated").alias("evidence_class"))
        .join(residue_mapping_frame(mapping_path), on=["condition_id", "residue_idx"], how="left")
        .with_columns(
            pl.coalesce(
                [
                    pl.col("residue_name"),
                    pl.concat_str([pl.lit("Residue"), pl.col("residue_idx").cast(pl.String)]),
                ]
            ).alias("residue_name")
        )
        .select(
            [
                "condition_id",
                pl.col("residue_idx").cast(pl.Int32),
                "residue_name",
                "mean_abs_load",
                "mean_signed_load",
                "load_std",
                "load_directionality",
                "temporal_correlation",
                "direction_score",
                "burst_motion",
                "max_burst_motion",
                "kcc_coherence",
                "coherence_score",
                "coherence_class",
                pl.col("n_concordant_channels").cast(pl.UInt8),
                "voxel_idx",
                "x_idx",
                "y_idx",
                "z_idx",
                "voxel_in_bounds",
                "shear_stress",
                "shear_stress_abs",
                "shear_stress_abs_p90",
                "shear_stress_max",
                "shear_stream_count",
                "structural_fault_line",
                "violent_kinetic_node",
                "dist_to_pocket_centroid_A",
                "dist_to_lock_centroid_A",
                "betweenness",
                "lateral_displacement_A",
                "wire_score",
                pl.col("pathway_rank").cast(pl.UInt8),
                "evidence_class",
                "load_normalized",
                "kcc_normalized",
            ]
        )
        .sort(["condition_id", "pathway_rank"])
        .collect()
    )


def main() -> int:
    args = parse_args()
    log = configure_logging(str(args.log_level))
    topologies = topology_products(cast(Path, args.grid_mapping))
    residue_load = residue_load_frame(cast(Path, args.mechanical_load), topologies.atom_to_residue)
    kcc = kcc_frame(cast(Path, args.kcc))
    shear = residue_shear_frame(cast(Path, args.shear_stress), cast(Path, args.grid_mapping), topologies.residue_coordinates)
    endpoints = endpoint_sets(cast(Path, args.risk_map))
    pathway = pathway_frame(
        residue_load,
        kcc,
        shear,
        cast(Path, args.coherence),
        endpoints,
        topologies.residue_coordinates,
        cast(Path, args.mapping),
        int(args.final_nodes_per_condition),
        float(args.min_betweenness),
        float(args.max_betweenness),
        float(args.max_lateral_A),
    )
    source_parquets = [
        cast(Path, args.mechanical_load),
        cast(Path, args.kcc),
        cast(Path, args.shear_stress),
        cast(Path, args.risk_map),
        cast(Path, args.coherence),
        cast(Path, args.mapping),
    ]
    ledger_parameters: JsonObject = {
        "input_sha256": {path.name: sha256_path(path) for path in source_parquets},
        "topology_sha256_by_condition": topologies.topology_hashes,
        "topology_files_by_condition": topologies.topology_files,
        "triple_evidence": {
            "mechanical_load": "mean_abs_load >= condition median among non-endpoint candidates",
            "kcc": "temporal_correlation >= condition median among non-endpoint candidates",
            "kcc_directionality": "ranking uses absolute KCC magnitude while preserving signed temporal_correlation",
            "coherence": "coherence_class in {fiber_invariant, partially_coherent}",
        },
        "fault_line_flags": {
            "structural_fault_line": "C-alpha voxel shear_stress_abs >= condition p90 from shear_stress_field.parquet",
            "violent_kinetic_node": "max burst_motion across KCC stream rows > 1000",
            "voxel_key": "condition_id,voxel_idx mapped from residue C-alpha coordinates",
        },
        "spatial_filter": {
            "min_betweenness": float(args.min_betweenness),
            "max_betweenness": float(args.max_betweenness),
            "max_lateral_A": float(args.max_lateral_A),
        },
        "wire_score": "load_normalized * abs_kcc_normalized * coherence_score",
    }
    write_provenance_parquet(
        pathway,
        cast(Path, args.output),
        producer_script=Path(__file__),
        source_parquets=source_parquets,
        schema_version="translation_pathway_nodes.v1",
        pipeline_stage="translation_pathway_nodes",
        partition_keys=["condition_id"],
        extra_metadata={"wire_score": "load_normalized*abs_kcc_normalized*coherence_score"},
        ledger_parameters=ledger_parameters,
        ledger_output_value={"rows": pathway.height, "output_path": cast(Path, args.output)},
        repo_root=REPO_ROOT,
    )
    log.info("translation_pathway_complete", rows=pathway.height, output=str(args.output))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
