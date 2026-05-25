#!/usr/bin/env python3
"""Generate a residue-based intracellular lock voxel mask."""

from __future__ import annotations

import argparse
import json
import math
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, TypeAlias, cast

import polars as pl


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_GRID_MAPPING = REPO_ROOT / "campaigns/glp1r_aleniglipron/track_0_manual_emulation/grid_coordinate_mapping.json"
DEFAULT_SIGNAL_GRID = (
    REPO_ROOT / "campaigns/glp1r_aleniglipron/integrated_spike_events/n80_full_scale/signal_grid_variance_channel.parquet"
)
DEFAULT_OUTPUT = REPO_ROOT / "campaigns/glp1r_aleniglipron/track_a_generative/lock_region_mask.json"
DEFAULT_CONDITION = "glp1r_6XOX_WT"
DEFAULT_LOCK_RESIDUES = "245-260,301-340,390-410"

JsonObject: TypeAlias = dict[str, Any]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pdb", type=Path, default=None, help="Optional PDB. Topology fallback is used if absent.")
    parser.add_argument("--topology", type=Path, default=None)
    parser.add_argument("--grid-config", type=Path, default=DEFAULT_GRID_MAPPING)
    parser.add_argument("--signal-grid", type=Path, default=DEFAULT_SIGNAL_GRID)
    parser.add_argument("--condition-id", type=str, default=DEFAULT_CONDITION)
    parser.add_argument("--lock-residues", type=str, default=DEFAULT_LOCK_RESIDUES)
    parser.add_argument("--radius", type=float, default=6.0)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def parse_ranges(value: str) -> list[tuple[int, int]]:
    ranges: list[tuple[int, int]] = []
    for part in value.split(","):
        if "-" in part:
            start, end = part.split("-", 1)
            ranges.append((int(start), int(end)))
        else:
            residue = int(part)
            ranges.append((residue, residue))
    return ranges


def load_grid_mapping(path: Path, condition_id: str) -> JsonObject:
    decoded = json.loads(path.read_text(encoding="utf-8"))
    conditions = decoded.get("conditions")
    if not isinstance(conditions, dict):
        raise ValueError(f"{path} missing conditions")
    condition = conditions.get(condition_id)
    if not isinstance(condition, dict):
        raise ValueError(f"{path} missing condition {condition_id}")
    return cast(JsonObject, condition)


def topology_path_from_grid(grid: JsonObject) -> Path:
    value = grid.get("topology_path")
    if not isinstance(value, str):
        raise ValueError("grid condition missing topology_path")
    return Path(value)


def ca_coordinates_from_topology(path: Path, ranges: list[tuple[int, int]]) -> list[tuple[int, tuple[float, float, float]]]:
    topology = json.loads(path.read_text(encoding="utf-8"))
    residues = cast(list[JsonObject], topology["residues"])
    residue_id_to_idx = {int(row["residue_id"]): int(row["residue_idx"]) for row in residues}
    ca_indices = cast(list[Any], topology["ca_indices"])
    positions = cast(list[Any], topology["positions"])
    coordinates: list[tuple[int, tuple[float, float, float]]] = []
    for start, end in ranges:
        for residue_id in range(start, end + 1):
            residue_idx = residue_id_to_idx.get(residue_id)
            if residue_idx is None or residue_idx >= len(ca_indices):
                continue
            atom_idx = int(ca_indices[residue_idx])
            coordinates.append(
                (
                    residue_id,
                    (
                        float(positions[3 * atom_idx]),
                        float(positions[3 * atom_idx + 1]),
                        float(positions[3 * atom_idx + 2]),
                    ),
                )
            )
    if not coordinates:
        raise RuntimeError(f"no CA coordinates found in {path} for residue ranges {ranges}")
    return coordinates


def voxel_center(ix: int, iy: int, iz: int, origin: tuple[float, float, float], spacing: float) -> tuple[float, float, float]:
    return (
        origin[0] + (float(ix) + 0.5) * spacing,
        origin[1] + (float(iy) + 0.5) * spacing,
        origin[2] + (float(iz) + 0.5) * spacing,
    )


def build_mask(
    grid: JsonObject,
    ca_coordinates: list[tuple[int, tuple[float, float, float]]],
    radius: float,
) -> set[int]:
    nx = int(grid["nx"])
    ny = int(grid["ny"])
    nz = int(grid["nz"])
    origin_raw = cast(list[Any], grid["origin_xyz_angstrom"])
    origin = (float(origin_raw[0]), float(origin_raw[1]), float(origin_raw[2]))
    spacing = float(grid["spacing_angstrom"])
    lock_voxels: set[int] = set()
    radius_sq = radius * radius
    shell = int(math.ceil(radius / spacing)) + 1
    for _residue_id, xyz in ca_coordinates:
        center_ix = math.floor((xyz[0] - origin[0]) / spacing)
        center_iy = math.floor((xyz[1] - origin[1]) / spacing)
        center_iz = math.floor((xyz[2] - origin[2]) / spacing)
        for ix in range(max(0, center_ix - shell), min(nx, center_ix + shell + 1)):
            for iy in range(max(0, center_iy - shell), min(ny, center_iy + shell + 1)):
                for iz in range(max(0, center_iz - shell), min(nz, center_iz + shell + 1)):
                    center = voxel_center(ix, iy, iz, origin, spacing)
                    dist_sq = sum((center[axis] - xyz[axis]) ** 2 for axis in range(3))
                    if dist_sq <= radius_sq:
                        lock_voxels.add(int(iz * nx * ny + iy * nx + ix))
    return lock_voxels


def nonvoid_voxels(signal_grid: Path, condition_id: str) -> set[int]:
    frame = (
        pl.scan_parquet(signal_grid)
        .filter((pl.col("condition_id") == condition_id) & (pl.col("variance_class") != "void"))
        .select("voxel_idx")
        .collect()
    )
    return {int(value) for value in frame.get_column("voxel_idx").to_list()}


def z_proxy_voxels(grid: JsonObject, voxel_indices: set[int]) -> set[int]:
    nx = int(grid["nx"])
    ny = int(grid["ny"])
    nz = int(grid["nz"])
    if not voxel_indices:
        return set()
    z_values = [voxel_idx // (nx * ny) for voxel_idx in voxel_indices]
    z_min = min(z_values)
    z_max = max(z_values)
    cutoff = z_min + int(math.floor(0.20 * max(z_max - z_min, 1)))
    return {voxel_idx for voxel_idx in voxel_indices if (voxel_idx // (nx * ny)) <= cutoff and (voxel_idx // (nx * ny)) < nz}


def atomic_write_json(path: Path, payload: JsonObject) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    tmp_path.replace(path)


def main() -> int:
    args = parse_args()
    grid = load_grid_mapping(Path(args.grid_config), str(args.condition_id))
    topology = Path(args.topology) if args.topology is not None else topology_path_from_grid(grid)
    ranges = parse_ranges(str(args.lock_residues))
    ca_coords = ca_coordinates_from_topology(topology, ranges)
    mask = build_mask(grid, ca_coords, float(args.radius))
    nonvoid = nonvoid_voxels(Path(args.signal_grid), str(args.condition_id))
    nonvoid_lock = mask & nonvoid
    z_proxy = z_proxy_voxels(grid, nonvoid)
    union = nonvoid_lock | z_proxy
    agreement = (len(nonvoid_lock & z_proxy) / float(len(union))) if union else 0.0
    grid_payload = {
        "condition_id": str(args.condition_id),
        "nx": int(grid["nx"]),
        "ny": int(grid["ny"]),
        "nz": int(grid["nz"]),
        "origin_xyz_angstrom": grid["origin_xyz_angstrom"],
        "spacing_angstrom": float(grid["spacing_angstrom"]),
    }
    payload: JsonObject = {
        "schema_version": "PRISM.lock_region_mask.v1",
        "epistemic_class": "DERIVED",
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "source_topology": topology.as_posix(),
        "source_pdb": Path(args.pdb).as_posix() if args.pdb is not None else None,
        "condition_id": str(args.condition_id),
        "lock_residue_ranges": str(args.lock_residues),
        "radius_A": float(args.radius),
        "residues_used": [residue_id for residue_id, _xyz in ca_coords],
        "n_lock_voxels": len(mask),
        "n_lock_nonvoid_voxels": len(nonvoid_lock),
        "n_nonvoid_voxels": len(nonvoid),
        "lock_fraction_nonvoid": len(nonvoid_lock) / float(max(len(nonvoid), 1)),
        "z_proxy_agreement_jaccard": agreement,
        "grid": grid_payload,
        "lock_voxel_indices": sorted(mask),
        "notes": [
            "Generated from topology C-alpha coordinates because no local 6XOX PDB was required for this run.",
            "Agreement is Jaccard overlap with the prior bottom-Z proxy over non-void voxels.",
        ],
    }
    atomic_write_json(Path(args.output), payload)
    print(
        "lock_mask_generated "
        f"condition_id={args.condition_id} n_lock_voxels={len(mask)} "
        f"n_lock_nonvoid={len(nonvoid_lock)} n_nonvoid={len(nonvoid)} "
        f"lock_fraction_nonvoid={payload['lock_fraction_nonvoid']:.4f} "
        f"z_proxy_agreement={agreement:.4f} output={args.output}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
