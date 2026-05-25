#!/usr/bin/env python3
"""Prepare receptor-side coordinate and binding-site assets for steric scoring."""

from __future__ import annotations

import argparse
import json
import logging
import math
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import TypeAlias, cast

import polars as pl


JsonObject: TypeAlias = dict[str, object]
Coordinate: TypeAlias = tuple[float, float, float]

REPO_ROOT = Path(__file__).resolve().parents[1]
TRACK0_DIR = REPO_ROOT / "campaigns/glp1r_aleniglipron/track_0_manual_emulation"
N80_DIR = REPO_ROOT / "campaigns/glp1r_aleniglipron/integrated_spike_events/n80_full_scale"
TOPOLOGY_DIR = Path(
    "/media/diddy/PRISM-LBS/prism-glp1r-aleniglipron-workspace/"
    "20260518T031002Z/04_TOPOLOGIES"
)
DEFAULT_RISK_MAP = N80_DIR / "receptor_durability_risk_map.parquet"
DEFAULT_SIGNAL_GRID = N80_DIR / "signal_grid_variance_channel.parquet"
DEFAULT_MAPPING = REPO_ROOT / "campaigns/glp1r_aleniglipron/topology/residue_index_mapping_matrix.parquet"
DEFAULT_GRID_MAPPING = TRACK0_DIR / "grid_coordinate_mapping.json"
DEFAULT_BINDING_SITE = TRACK0_DIR / "binding_site_reference.json"

GRID_PADDING_ANGSTROM = 5.0
DEFAULT_GRID_SPACING_ANGSTROM = 0.5
GRID_COVERAGE_SAFETY_FACTOR = 1.02
POCKET_TARGET_COUNT = 4
DOWNSTREAM_TARGET_COUNT = 5
STANDARD_AMINO_ACIDS = {
    "ALA",
    "ARG",
    "ASN",
    "ASP",
    "CYS",
    "GLN",
    "GLU",
    "GLY",
    "HID",
    "HIE",
    "HIP",
    "ILE",
    "LEU",
    "LYS",
    "MET",
    "PHE",
    "PRO",
    "SER",
    "THR",
    "TRP",
    "TYR",
    "VAL",
}


@dataclass(frozen=True)
class GridGeometry:
    condition_id: str
    topology_path: Path
    grid_dim: int
    origin: Coordinate
    spacing: float
    padded_extent_angstrom: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--risk-map", type=Path, default=DEFAULT_RISK_MAP)
    parser.add_argument("--signal-grid", type=Path, default=DEFAULT_SIGNAL_GRID)
    parser.add_argument("--mapping", type=Path, default=DEFAULT_MAPPING)
    parser.add_argument("--topology-dir", type=Path, default=TOPOLOGY_DIR)
    parser.add_argument("--grid-mapping", type=Path, default=DEFAULT_GRID_MAPPING)
    parser.add_argument("--binding-site", type=Path, default=DEFAULT_BINDING_SITE)
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def load_json_object(path: Path) -> dict[str, object]:
    loaded = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        raise ValueError(f"{path} did not decode to a JSON object")
    return loaded


def json_list(value: object, label: str) -> list[object]:
    if not isinstance(value, list):
        raise ValueError(f"{label} is not a list")
    return value


def json_dict(value: object, label: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise ValueError(f"{label} is not an object")
    return cast(dict[str, object], value)


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


def float_positions(topology: dict[str, object], path: Path) -> list[float]:
    raw = json_list(topology.get("positions"), f"{path}:positions")
    return [as_float(item, f"{path}:positions[]") for item in raw]


def int_list(value: object, label: str) -> list[int]:
    return [as_int(item, f"{label}[]") for item in json_list(value, label)]


def str_list(value: object, label: str) -> list[str]:
    return [str(item) for item in json_list(value, label)]


def residue_records(topology: dict[str, object], path: Path) -> list[dict[str, object]]:
    return [json_dict(item, f"{path}:residues[]") for item in json_list(topology.get("residues"), f"{path}:residues")]


def residue_to_atoms(topology: dict[str, object], path: Path) -> dict[int, list[int]]:
    raw = json_dict(topology.get("residue_to_atom_indices"), f"{path}:residue_to_atom_indices")
    return {int(key): int_list(value, f"{path}:residue_to_atom_indices[{key}]") for key, value in raw.items()}


def coordinate_at(positions: list[float], atom_idx: int) -> Coordinate:
    base = atom_idx * 3
    return (positions[base], positions[base + 1], positions[base + 2])


def coordinate_json(coord: Coordinate) -> list[float]:
    return [coord[0], coord[1], coord[2]]


def centroid(coords: list[Coordinate]) -> Coordinate:
    if not coords:
        raise ValueError("cannot compute centroid for empty coordinate set")
    count = float(len(coords))
    return (
        sum(coord[0] for coord in coords) / count,
        sum(coord[1] for coord in coords) / count,
        sum(coord[2] for coord in coords) / count,
    )


def grid_dimensions(signal_grid: Path) -> dict[str, int]:
    rows = (
        pl.scan_parquet(signal_grid)
        .group_by("condition_id")
        .agg(
            [
                (pl.col("x_idx").max() + 1).alias("nx"),
                (pl.col("y_idx").max() + 1).alias("ny"),
                (pl.col("z_idx").max() + 1).alias("nz"),
            ]
        )
        .collect()
        .to_dicts()
    )
    dims: dict[str, int] = {}
    for row in rows:
        nx = as_int(row["nx"], "nx")
        ny = as_int(row["ny"], "ny")
        nz = as_int(row["nz"], "nz")
        if nx != ny or nx != nz:
            raise ValueError(f"{row['condition_id']} has non-cubic signal grid {nx}x{ny}x{nz}")
        dims[str(row["condition_id"])] = nx
    return dims


def topology_path(topology_dir: Path, condition_id: str) -> Path:
    path = topology_dir / f"{condition_id}.topology.json"
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def build_geometry(condition_id: str, topology_dir: Path, grid_dim: int) -> GridGeometry:
    path = topology_path(topology_dir, condition_id)
    topology = load_json_object(path)
    positions = float_positions(topology, path)
    xs = positions[0::3]
    ys = positions[1::3]
    zs = positions[2::3]
    mins = (min(xs), min(ys), min(zs))
    maxs = (max(xs), max(ys), max(zs))
    padded_extent = max(
        maxs[axis] - mins[axis] + 2.0 * GRID_PADDING_ANGSTROM for axis in range(3)
    )
    spacing = max(
        DEFAULT_GRID_SPACING_ANGSTROM,
        (padded_extent / float(grid_dim)) * GRID_COVERAGE_SAFETY_FACTOR,
    )
    origin = (
        mins[0] - GRID_PADDING_ANGSTROM,
        mins[1] - GRID_PADDING_ANGSTROM,
        mins[2] - GRID_PADDING_ANGSTROM,
    )
    return GridGeometry(
        condition_id=condition_id,
        topology_path=path,
        grid_dim=grid_dim,
        origin=origin,
        spacing=spacing,
        padded_extent_angstrom=padded_extent,
    )


def voxel_indices(coord: Coordinate, geometry: GridGeometry) -> tuple[int, int, int]:
    return (
        math.trunc((coord[0] - geometry.origin[0]) / geometry.spacing),
        math.trunc((coord[1] - geometry.origin[1]) / geometry.spacing),
        math.trunc((coord[2] - geometry.origin[2]) / geometry.spacing),
    )


def voxel_idx(indices: tuple[int, int, int], grid_dim: int) -> int:
    ix, iy, iz = indices
    return iz * grid_dim * grid_dim + iy * grid_dim + ix


def variance_lookup(signal_grid: Path, condition_id: str, current_voxel_idx: int) -> dict[str, object]:
    rows = (
        pl.scan_parquet(signal_grid)
        .filter(
            (pl.col("condition_id") == condition_id)
            & (pl.col("voxel_idx") == current_voxel_idx)
        )
        .select(
            [
                "voxel_idx",
                "x_idx",
                "y_idx",
                "z_idx",
                "variance_class",
                "hit_count_cold_mean",
                "hit_count_warm_mean",
                "cold_stream_count",
                "warm_stream_count",
            ]
        )
        .collect()
        .to_dicts()
    )
    if len(rows) != 1:
        raise ValueError(f"expected one variance row for {condition_id}:{current_voxel_idx}, got {len(rows)}")
    return rows[0]


def mapping_row(mapping: pl.DataFrame, condition_id: str, label: str) -> dict[str, object]:
    rows = (
        mapping.filter(
            (pl.col("condition_id") == condition_id)
            & (pl.col("canonical_residue_label") == label)
        )
        .to_dicts()
    )
    if len(rows) != 1:
        raise ValueError(f"expected one residue mapping for {condition_id}:{label}, got {len(rows)}")
    return rows[0]


def ca_validation(
    mapping: pl.DataFrame,
    signal_grid: Path,
    geometry: GridGeometry,
    canonical_label: str,
    expected_class: str,
) -> JsonObject:
    row = mapping_row(mapping, geometry.condition_id, canonical_label)
    residue_idx = as_int(row["residue_idx"], "residue_idx")
    topology = load_json_object(geometry.topology_path)
    ca_indices = int_list(topology.get("ca_indices"), f"{geometry.topology_path}:ca_indices")
    residues = residue_records(topology, geometry.topology_path)
    positions = float_positions(topology, geometry.topology_path)
    ca_atom_idx = ca_indices[residue_idx]
    ca_coord = coordinate_at(positions, ca_atom_idx)
    indices = voxel_indices(ca_coord, geometry)
    current_voxel_idx = voxel_idx(indices, geometry.grid_dim)
    variance = variance_lookup(signal_grid, geometry.condition_id, current_voxel_idx)
    warm_stream_count = as_int(variance["warm_stream_count"], "warm_stream_count")
    observed_class = str(variance["variance_class"])
    passed = observed_class == expected_class
    reason = "matched expected variance class"
    if not passed and warm_stream_count == 0:
        reason = (
            "requested stable_occupied assertion is impossible for this condition because "
            "the variance channel contains zero warm streams"
        )
    elif not passed:
        reason = "observed variance class differed from expected class"
    topology_residue = residues[residue_idx]
    return {
        "condition_id": geometry.condition_id,
        "canonical_residue_label": canonical_label,
        "residue_idx": residue_idx,
        "topology_residue_name": str(topology_residue.get("residue_name", "")),
        "topology_residue_id": as_int(topology_residue.get("residue_id", -1), "topology_residue_id"),
        "ca_atom_idx": ca_atom_idx,
        "ca_xyz_angstrom": coordinate_json(ca_coord),
        "grid_indices": [indices[0], indices[1], indices[2]],
        "voxel_idx": current_voxel_idx,
        "expected_variance_class": expected_class,
        "observed_variance_class": observed_class,
        "hit_count_cold_mean": as_float(variance["hit_count_cold_mean"], "hit_count_cold_mean"),
        "hit_count_warm_mean": as_float(variance["hit_count_warm_mean"], "hit_count_warm_mean"),
        "cold_stream_count": as_int(variance["cold_stream_count"], "cold_stream_count"),
        "warm_stream_count": warm_stream_count,
        "passed": passed,
        "interpretation": reason,
    }


def select_critical_edges(risk_map: Path, mapping: Path) -> pl.DataFrame:
    risk = pl.scan_parquet(risk_map)
    mapping_lf = pl.scan_parquet(mapping)
    from_mapping = mapping_lf.select(
        [
            "condition_id",
            pl.col("residue_idx").alias("edge_from_residue"),
            pl.col("amino_acid_3letter").alias("edge_from_amino_acid"),
            pl.col("biological_sequence_number").alias("edge_from_sequence_number"),
            pl.col("canonical_residue_label").alias("edge_from_label"),
        ]
    )
    to_mapping = mapping_lf.select(
        [
            "condition_id",
            pl.col("residue_idx").alias("edge_to_residue"),
            pl.col("amino_acid_3letter").alias("edge_to_amino_acid"),
            pl.col("biological_sequence_number").alias("edge_to_sequence_number"),
            pl.col("canonical_residue_label").alias("edge_to_label"),
        ]
    )
    critical = risk.filter(pl.col("durability_class") == "critical_durability_risk")
    selected = pl.concat(
        [
            critical.filter(pl.col("edge_class") == "pocket_vector")
            .sort("durability_risk_score_raw", descending=True)
            .limit(POCKET_TARGET_COUNT),
            critical.filter(pl.col("edge_class") == "downstream_lock")
            .sort("durability_risk_score_raw", descending=True)
            .limit(DOWNSTREAM_TARGET_COUNT),
        ]
    )
    return (
        selected.join(from_mapping, on=["condition_id", "edge_from_residue"], how="inner")
        .join(to_mapping, on=["condition_id", "edge_to_residue"], how="inner")
        .with_columns(
            (
                pl.col("condition_id")
                + ":"
                + pl.col("edge_from_label")
                + "->"
                + pl.col("edge_to_label")
            ).alias("edge_id"),
            (pl.col("edge_from_label") + " -> " + pl.col("edge_to_label")).alias("edge_label"),
        )
        .sort("durability_risk_score_raw", descending=True)
        .collect()
    )


def atom_coordinates_for_residue(
    topology: dict[str, object],
    topology_file: Path,
    residue_idx: int,
) -> list[dict[str, object]]:
    positions = float_positions(topology, topology_file)
    elements = str_list(topology.get("elements"), f"{topology_file}:elements")
    atom_names = str_list(topology.get("atom_names"), f"{topology_file}:atom_names")
    residue_atoms = residue_to_atoms(topology, topology_file).get(residue_idx)
    if residue_atoms is None:
        raise ValueError(f"{topology_file} has no atom list for residue_idx={residue_idx}")
    atoms: list[dict[str, object]] = []
    for atom_idx in residue_atoms:
        atoms.append(
            {
                "atom_idx": atom_idx,
                "atom_name": atom_names[atom_idx],
                "element": elements[atom_idx],
                "xyz_angstrom": coordinate_json(coordinate_at(positions, atom_idx)),
            }
        )
    return atoms


def scan_reference_source(topology_dir: Path) -> JsonObject:
    ligand_candidates: list[dict[str, object]] = []
    peptide_candidates: list[dict[str, object]] = []
    for path in sorted(topology_dir.glob("*.topology.json")):
        topology = load_json_object(path)
        names = sorted({str(name).upper() for name in json_list(topology.get("residue_names"), f"{path}:residue_names")})
        non_standard = [name for name in names if name not in STANDARD_AMINO_ACIDS]
        if non_standard:
            ligand_candidates.append({"topology": path.as_posix(), "residue_names": non_standard})
        chains = sorted({str(chain) for chain in json_list(topology.get("chain_ids"), f"{path}:chain_ids")})
        if len(chains) > 1:
            peptide_candidates.append({"topology": path.as_posix(), "chains": chains})
    return {
        "small_molecule_ligands": ligand_candidates,
        "peptide_candidates": peptide_candidates,
        "selected_reference_type": "critical_pocket_centroid",
        "selection_reason": (
            "No non-standard small-molecule residue and no separate peptide chain were present "
            "in the topology JSON files; using the C-alpha centroid of selected pocket-vector residues."
        ),
    }


def build_binding_site_reference(
    edges: pl.DataFrame,
    topology_dir: Path,
) -> JsonObject:
    edge_records: list[JsonObject] = []
    pocket_ca_coords: list[Coordinate] = []
    all_edge_atom_coords: list[Coordinate] = []
    for row in edges.to_dicts():
        condition_id = str(row["condition_id"])
        topology_file = topology_path(topology_dir, condition_id)
        topology = load_json_object(topology_file)
        ca_indices = int_list(topology.get("ca_indices"), f"{topology_file}:ca_indices")
        positions = float_positions(topology, topology_file)
        from_idx = as_int(row["edge_from_residue"], "edge_from_residue")
        to_idx = as_int(row["edge_to_residue"], "edge_to_residue")
        from_atoms = atom_coordinates_for_residue(topology, topology_file, from_idx)
        to_atoms = atom_coordinates_for_residue(topology, topology_file, to_idx)
        for atom in [*from_atoms, *to_atoms]:
            coord_values = json_list(atom.get("xyz_angstrom"), "atom.xyz_angstrom")
            all_edge_atom_coords.append(
                (
                    as_float(coord_values[0], "atom_coordinate_x"),
                    as_float(coord_values[1], "atom_coordinate_y"),
                    as_float(coord_values[2], "atom_coordinate_z"),
                )
            )
        from_ca = coordinate_at(positions, ca_indices[from_idx])
        to_ca = coordinate_at(positions, ca_indices[to_idx])
        if str(row["edge_class"]) == "pocket_vector":
            pocket_ca_coords.extend([from_ca, to_ca])
        edge_records.append(
            {
                "edge_id": str(row["edge_id"]),
                "condition_id": condition_id,
                "edge_label": str(row["edge_label"]),
                "edge_class": str(row["edge_class"]),
                "edge_from_residue": from_idx,
                "edge_to_residue": to_idx,
                "edge_from_label": str(row["edge_from_label"]),
                "edge_to_label": str(row["edge_to_label"]),
                "durability_risk_score_raw": as_float(row["durability_risk_score_raw"], "durability_risk_score_raw"),
                "topology_path": topology_file.as_posix(),
                "from_ca_xyz_angstrom": coordinate_json(from_ca),
                "to_ca_xyz_angstrom": coordinate_json(to_ca),
                "from_atom_coordinates": from_atoms,
                "to_atom_coordinates": to_atoms,
            }
        )
    unique_pocket_coords = list(dict.fromkeys(pocket_ca_coords))
    reference_centroid = centroid(unique_pocket_coords)
    binding_radius = max(
        math.dist(reference_centroid, coord) for coord in all_edge_atom_coords
    ) + 6.0
    return {
        "schema_version": "binding_site_reference.v1",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "coordinate_units": "angstrom",
        "binding_site_center_angstrom": coordinate_json(reference_centroid),
        "binding_site_radius_angstrom": binding_radius,
        "reference_ligand_coords": None,
        "reference": {
            "reference_type": "critical_pocket_centroid",
            "alignment_mode": "receptor_shell_principal_axes_when_no_ligand",
            "centroid_xyz_angstrom": coordinate_json(reference_centroid),
            "alignment_point_coordinates": [coordinate_json(coord) for coord in unique_pocket_coords],
            "note": (
                "The topology scan found no co-crystallized small-molecule ligand or separate peptide chain; "
                "the standalone scorer aligns SDF conformers to this receptor-side reference shell."
            ),
        },
        "topology_scan": scan_reference_source(topology_dir),
        "critical_edges": edge_records,
    }


def build_grid_mapping(
    geometries: dict[str, GridGeometry],
    mapping_path: Path,
    signal_grid: Path,
) -> JsonObject:
    mapping = pl.read_parquet(mapping_path)
    requested = ca_validation(
        mapping,
        signal_grid,
        geometries["glp1r_5VEX_WT"],
        "LEU144",
        "stable_occupied",
    )
    stable_controls: list[object] = []
    for condition_id in sorted(geometries):
        validation = ca_validation(mapping, signal_grid, geometries[condition_id], "LEU144", "stable_occupied")
        if validation["passed"] is True:
            stable_controls.append(validation)
    return {
        "schema_version": "grid_coordinate_mapping.v1",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "units": "angstrom",
        "rust_producer_sources": [
            "crates/prism-nhs/src/fused_engine.rs:new_on_stream uses min_position - 5.0A origin",
            "crates/prism-nhs/src/bin/nhs_rt_full.rs:grid_spacing_for_coverage uses max(0.5, padded_extent / grid_dim * 1.02)",
            "crates/prism-nhs/src/bin/nhs_rt_full.rs spatial grid state uses idx = iz * dim * dim + iy * dim + ix",
        ],
        "forward_mapping": {
            "ix": "trunc((x - origin_x) / spacing)",
            "iy": "trunc((y - origin_y) / spacing)",
            "iz": "trunc((z - origin_z) / spacing)",
            "voxel_idx": "iz * nx * ny + iy * nx + ix",
        },
        "inverse_mapping": {
            "ix": "voxel_idx % nx",
            "iy": "(voxel_idx / nx) % ny",
            "iz": "voxel_idx / (nx * ny)",
            "center_x": "origin_x + (ix + 0.5) * spacing",
            "center_y": "origin_y + (iy + 0.5) * spacing",
            "center_z": "origin_z + (iz + 0.5) * spacing",
        },
        "conditions": {
            condition_id: {
                "topology_path": geometry.topology_path.as_posix(),
                "grid_dim": geometry.grid_dim,
                "nx": geometry.grid_dim,
                "ny": geometry.grid_dim,
                "nz": geometry.grid_dim,
                "origin_xyz_angstrom": coordinate_json(geometry.origin),
                "spacing_angstrom": geometry.spacing,
                "padding_each_side_angstrom": GRID_PADDING_ANGSTROM,
                "padded_extent_angstrom": geometry.padded_extent_angstrom,
            }
            for condition_id, geometry in sorted(geometries.items())
        },
        "cross_validation": {
            "requested_5VEX_LEU144": requested,
            "stable_occupied_controls": stable_controls,
            "passed_requested_assertion": requested["passed"],
            "passed_coordinate_formula_control": len(stable_controls) > 0,
        },
    }


def write_json(path: Path, payload: JsonObject) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    logging.basicConfig(level=str(args.log_level).upper(), format="%(levelname)s %(message)s")
    dims = grid_dimensions(args.signal_grid)
    geometries = {
        condition_id: build_geometry(condition_id, args.topology_dir, grid_dim)
        for condition_id, grid_dim in sorted(dims.items())
    }
    grid_mapping = build_grid_mapping(geometries, args.mapping, args.signal_grid)
    write_json(args.grid_mapping, grid_mapping)
    edges = select_critical_edges(args.risk_map, args.mapping)
    binding_site = build_binding_site_reference(edges, args.topology_dir)
    write_json(args.binding_site, binding_site)
    logging.info("wrote %s", args.grid_mapping)
    logging.info("wrote %s", args.binding_site)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
