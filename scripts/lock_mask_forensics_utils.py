#!/usr/bin/env python3
"""Shared geometry utilities for Track A lock-mask forensics."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, TypeAlias, cast

from rdkit import Chem


Coordinate3D: TypeAlias = tuple[float, float, float]
JsonObject: TypeAlias = dict[str, Any]


@dataclass(frozen=True)
class GridSpec:
    condition_id: str
    nx: int
    ny: int
    nz: int
    origin: Coordinate3D
    spacing: float
    topology_path: Path


@dataclass(frozen=True)
class BBox:
    min_xyz: Coordinate3D
    max_xyz: Coordinate3D

    @property
    def centroid(self) -> Coordinate3D:
        return tuple(
            (self.min_xyz[axis] + self.max_xyz[axis]) / 2.0 for axis in range(3)
        )  # type: ignore[return-value]

    def contains(self, point: Coordinate3D) -> bool:
        return all(
            self.min_xyz[axis] <= point[axis] <= self.max_xyz[axis] for axis in range(3)
        )

    def fully_contains(self, other: "BBox") -> bool:
        return all(
            self.min_xyz[axis] <= other.min_xyz[axis]
            and other.max_xyz[axis] <= self.max_xyz[axis]
            for axis in range(3)
        )


@dataclass(frozen=True)
class ResiduePoint:
    residue_id: int
    residue_name: str
    xyz: Coordinate3D


@dataclass(frozen=True)
class LockMask:
    lock_voxel_indices: frozenset[int]
    grid: GridSpec


def parse_ranges(value: str) -> list[tuple[int, int]]:
    ranges: list[tuple[int, int]] = []
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start_text, end_text = part.split("-", 1)
            ranges.append((int(start_text), int(end_text)))
        else:
            residue_id = int(part)
            ranges.append((residue_id, residue_id))
    if not ranges:
        raise ValueError("expected at least one residue range")
    return ranges


def grid_bbox(spec: GridSpec) -> BBox:
    return BBox(
        min_xyz=spec.origin,
        max_xyz=(
            spec.origin[0] + spec.nx * spec.spacing,
            spec.origin[1] + spec.ny * spec.spacing,
            spec.origin[2] + spec.nz * spec.spacing,
        ),
    )


def bbox_for_points(points: Iterable[Coordinate3D]) -> BBox:
    point_list = list(points)
    if not point_list:
        raise ValueError("cannot compute bbox for empty point set")
    min_xyz: Coordinate3D = (
        min(point[0] for point in point_list),
        min(point[1] for point in point_list),
        min(point[2] for point in point_list),
    )
    max_xyz: Coordinate3D = (
        max(point[0] for point in point_list),
        max(point[1] for point in point_list),
        max(point[2] for point in point_list),
    )
    return BBox(
        min_xyz=min_xyz,
        max_xyz=max_xyz,
    )


def centroid_for_points(points: Iterable[Coordinate3D]) -> Coordinate3D:
    point_list = list(points)
    if not point_list:
        raise ValueError("cannot compute centroid for empty point set")
    return tuple(
        sum(point[axis] for point in point_list) / float(len(point_list)) for axis in range(3)
    )  # type: ignore[return-value]


def load_grid_spec(path: Path, condition_id: str) -> GridSpec:
    decoded = json.loads(path.read_text(encoding="utf-8"))
    conditions = decoded.get("conditions")
    if not isinstance(conditions, dict):
        raise ValueError(f"{path} missing conditions mapping")
    condition = conditions.get(condition_id)
    if not isinstance(condition, dict):
        raise ValueError(f"{path} missing condition_id={condition_id}")
    origin_raw = condition.get("origin_xyz_angstrom")
    if not isinstance(origin_raw, list) or len(origin_raw) != 3:
        raise ValueError(f"{path} condition {condition_id} missing origin_xyz_angstrom")
    topology_value = condition.get("topology_path")
    if not isinstance(topology_value, str):
        raise ValueError(f"{path} condition {condition_id} missing topology_path")
    return GridSpec(
        condition_id=condition_id,
        nx=int(condition["nx"]),
        ny=int(condition["ny"]),
        nz=int(condition["nz"]),
        origin=(
            float(origin_raw[0]),
            float(origin_raw[1]),
            float(origin_raw[2]),
        ),
        spacing=float(condition["spacing_angstrom"]),
        topology_path=Path(topology_value),
    )


def load_lock_mask(path: Path) -> LockMask:
    decoded = json.loads(path.read_text(encoding="utf-8"))
    grid_object = decoded.get("grid")
    if not isinstance(grid_object, dict):
        raise ValueError(f"{path} missing grid object")
    topology_path = decoded.get("source_topology")
    if not isinstance(topology_path, str):
        topology_path = str(Path("."))
    origin_raw = grid_object.get("origin_xyz_angstrom")
    if not isinstance(origin_raw, list) or len(origin_raw) != 3:
        raise ValueError(f"{path} lock mask missing origin_xyz_angstrom")
    lock_values = decoded.get("lock_voxel_indices")
    if not isinstance(lock_values, list):
        raise ValueError(f"{path} lock mask missing lock_voxel_indices")
    return LockMask(
        lock_voxel_indices=frozenset(int(value) for value in lock_values),
        grid=GridSpec(
            condition_id=str(decoded.get("condition_id", "unknown")),
            nx=int(grid_object["nx"]),
            ny=int(grid_object["ny"]),
            nz=int(grid_object["nz"]),
            origin=(
                float(origin_raw[0]),
                float(origin_raw[1]),
                float(origin_raw[2]),
            ),
            spacing=float(grid_object["spacing_angstrom"]),
            topology_path=Path(topology_path),
        ),
    )


def load_topology_residues(path: Path) -> list[ResiduePoint]:
    decoded = json.loads(path.read_text(encoding="utf-8"))
    residues = decoded.get("residues")
    ca_indices = decoded.get("ca_indices")
    positions = decoded.get("positions")
    if not isinstance(residues, list) or not isinstance(ca_indices, list) or not isinstance(positions, list):
        raise ValueError(f"{path} missing topology arrays")
    typed_residues = cast(list[JsonObject], residues)
    typed_positions = [float(value) for value in positions]
    result: list[ResiduePoint] = []
    for residue in typed_residues:
        residue_idx = int(residue["residue_idx"])
        atom_idx = int(ca_indices[residue_idx])
        result.append(
            ResiduePoint(
                residue_id=int(residue["residue_id"]),
                residue_name=str(residue["residue_name"]),
                xyz=(
                    typed_positions[3 * atom_idx],
                    typed_positions[3 * atom_idx + 1],
                    typed_positions[3 * atom_idx + 2],
                ),
            )
        )
    return result


def load_pdb_atom_points(path: Path) -> list[Coordinate3D]:
    points: list[Coordinate3D] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.startswith(("ATOM  ", "HETATM")):
            continue
        points.append(
            (
                float(line[30:38]),
                float(line[38:46]),
                float(line[46:54]),
            )
        )
    if not points:
        raise ValueError(f"{path} contains no ATOM/HETATM coordinates")
    return points


def load_pdb_residues(path: Path) -> list[ResiduePoint]:
    residues: list[ResiduePoint] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.startswith("ATOM  "):
            continue
        if line[12:16].strip() != "CA":
            continue
        residues.append(
            ResiduePoint(
                residue_id=int(line[22:26]),
                residue_name=line[17:20].strip(),
                xyz=(
                    float(line[30:38]),
                    float(line[38:46]),
                    float(line[46:54]),
                ),
            )
        )
    if not residues:
        raise ValueError(f"{path} contains no C-alpha residues")
    return residues


def load_sdf_coordinates(path: Path) -> list[Coordinate3D]:
    supplier = Chem.SDMolSupplier(str(path), removeHs=False)
    mol = supplier[0]
    if mol is None:
        raise ValueError(f"failed to parse SDF {path}")
    conf = mol.GetConformer()
    return [
        (
            float(conf.GetAtomPosition(atom_idx).x),
            float(conf.GetAtomPosition(atom_idx).y),
            float(conf.GetAtomPosition(atom_idx).z),
        )
        for atom_idx in range(mol.GetNumAtoms())
    ]


def residue_subset(
    residues: Iterable[ResiduePoint], ranges: list[tuple[int, int]]
) -> list[ResiduePoint]:
    selected: list[ResiduePoint] = []
    for residue in residues:
        if any(start <= residue.residue_id <= end for start, end in ranges):
            selected.append(residue)
    if not selected:
        raise ValueError("no residues matched requested ranges")
    return selected


def mean_coordinate_delta(
    lhs: Iterable[Coordinate3D], rhs: Iterable[Coordinate3D]
) -> Coordinate3D:
    lhs_list = list(lhs)
    rhs_list = list(rhs)
    if len(lhs_list) != len(rhs_list):
        raise ValueError("delta calculation requires equal-length coordinate sets")
    if not lhs_list:
        raise ValueError("delta calculation requires at least one point")
    return tuple(
        sum(rhs_list[idx][axis] - lhs_list[idx][axis] for idx in range(len(lhs_list)))
        / float(len(lhs_list))
        for axis in range(3)
    )  # type: ignore[return-value]


def rms_distance(lhs: Iterable[Coordinate3D], rhs: Iterable[Coordinate3D]) -> float:
    lhs_list = list(lhs)
    rhs_list = list(rhs)
    if len(lhs_list) != len(rhs_list):
        raise ValueError("RMS distance requires equal-length coordinate sets")
    if not lhs_list:
        raise ValueError("RMS distance requires at least one point")
    return math.sqrt(
        sum(
            sum((lhs_list[idx][axis] - rhs_list[idx][axis]) ** 2 for axis in range(3))
            for idx in range(len(lhs_list))
        )
        / float(len(lhs_list))
    )


def common_residue_pairs(
    lhs: Iterable[ResiduePoint], rhs: Iterable[ResiduePoint]
) -> list[tuple[ResiduePoint, ResiduePoint]]:
    lhs_map = {residue.residue_id: residue for residue in lhs}
    rhs_map = {residue.residue_id: residue for residue in rhs}
    common = sorted(set(lhs_map).intersection(rhs_map))
    return [(lhs_map[residue_id], rhs_map[residue_id]) for residue_id in common]


def z_proxy_atom_indices(coords: list[Coordinate3D], fraction: float = 0.20) -> list[int]:
    if not coords:
        return []
    z_values = [coord[2] for coord in coords]
    z_min = min(z_values)
    z_max = max(z_values)
    cutoff = z_min + (fraction * (z_max - z_min))
    return [idx for idx, coord in enumerate(coords) if coord[2] <= cutoff]


def voxel_index_for_coordinate(coord: Coordinate3D, grid: GridSpec) -> int | None:
    ix = math.floor((coord[0] - grid.origin[0]) / grid.spacing)
    iy = math.floor((coord[1] - grid.origin[1]) / grid.spacing)
    iz = math.floor((coord[2] - grid.origin[2]) / grid.spacing)
    if ix < 0 or iy < 0 or iz < 0 or ix >= grid.nx or iy >= grid.ny or iz >= grid.nz:
        return None
    return int(iz * (grid.nx * grid.ny) + iy * grid.nx + ix)


def voxel_center(voxel_idx: int, grid: GridSpec) -> Coordinate3D:
    plane = grid.nx * grid.ny
    iz = voxel_idx // plane
    rem = voxel_idx % plane
    iy = rem // grid.nx
    ix = rem % grid.nx
    return (
        grid.origin[0] + (float(ix) + 0.5) * grid.spacing,
        grid.origin[1] + (float(iy) + 0.5) * grid.spacing,
        grid.origin[2] + (float(iz) + 0.5) * grid.spacing,
    )


def atom_indices_in_lock_mask(coords: list[Coordinate3D], mask: LockMask) -> list[int]:
    indices: list[int] = []
    for atom_idx, coord in enumerate(coords):
        voxel_idx = voxel_index_for_coordinate(coord, mask.grid)
        if voxel_idx is not None and voxel_idx in mask.lock_voxel_indices:
            indices.append(atom_idx)
    return indices


def voxel_indices_for_atoms(
    coords: list[Coordinate3D], grid: GridSpec, atom_indices: Iterable[int]
) -> list[int]:
    voxels: list[int] = []
    for atom_idx in atom_indices:
        voxel_idx = voxel_index_for_coordinate(coords[atom_idx], grid)
        if voxel_idx is not None:
            voxels.append(voxel_idx)
    return voxels


def nearest_residue(point: Coordinate3D, residues: Iterable[ResiduePoint]) -> tuple[ResiduePoint, float]:
    residue_list = list(residues)
    if not residue_list:
        raise ValueError("nearest residue requires at least one residue")
    best = residue_list[0]
    best_distance = euclidean_distance(point, best.xyz)
    for residue in residue_list[1:]:
        distance = euclidean_distance(point, residue.xyz)
        if distance < best_distance:
            best = residue
            best_distance = distance
    return best, best_distance


def euclidean_distance(lhs: Coordinate3D, rhs: Coordinate3D) -> float:
    return math.sqrt(sum((lhs[axis] - rhs[axis]) ** 2 for axis in range(3)))
