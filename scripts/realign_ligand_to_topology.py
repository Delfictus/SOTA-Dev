#!/usr/bin/env python3
"""Rigidly superimpose Aleniglipron into the 6XOX frame and relieve steric clashes."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from dataclasses import dataclass
from importlib import import_module
from pathlib import Path
from typing import Any, TypeAlias, cast

import numpy as np
import polars as pl
from numpy.typing import NDArray


REPO_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_TOPOLOGIES = Path("/media/diddy/PRISM-LBS/prism-glp1r-aleniglipron-workspace/20260518T031002Z/04_TOPOLOGIES")
DEFAULT_INPUT = (
    REPO_ROOT / "campaigns/glp1r_aleniglipron/track_0_manual_emulation/conformers/ALENI-PARENT_am1bcc.sdf"
)
DEFAULT_OUTPUT_KABSCH = (
    REPO_ROOT
    / "campaigns/glp1r_aleniglipron/track_a_generative/ALENI-PARENT_6XOX_frame_kabsch.sdf"
)
DEFAULT_OUTPUT_MINIMIZED = (
    REPO_ROOT
    / "campaigns/glp1r_aleniglipron/track_a_generative/ALENI-PARENT_6XOX_frame_minimized.sdf"
)
DEFAULT_OUTPUT_O3A = (
    REPO_ROOT / "campaigns/glp1r_aleniglipron/track_a_generative/ALENI-PARENT_6XOX_frame_o3a.sdf"
)
DEFAULT_OUTPUT_O3A_BEST = (
    REPO_ROOT / "campaigns/glp1r_aleniglipron/track_a_generative/ALENI-PARENT_6XOX_frame_o3a_best.sdf"
)
DEFAULT_OUTPUT_O3A_RELAXED = (
    REPO_ROOT
    / "campaigns/glp1r_aleniglipron/track_a_generative/ALENI-PARENT_6XOX_frame_o3a_relaxed.sdf"
)
DEFAULT_MANIFEST = (
    REPO_ROOT
    / "campaigns/glp1r_aleniglipron/track_a_generative/ALENI-PARENT_6XOX_alignment_manifest.json"
)
DEFAULT_POSE_MANIFEST = (
    REPO_ROOT
    / "campaigns/glp1r_aleniglipron/track_a_generative/ALENI-PARENT_6XOX_pose_reconciliation_manifest.json"
)
DEFAULT_PSEUDOLIGAND = (
    REPO_ROOT
    / "campaigns/glp1r_aleniglipron/track_a_generative/6XOX_PRISM_negative_image_pseudoligand.sdf"
)
DEFAULT_PSEUDOLIGAND_MANIFEST = (
    REPO_ROOT
    / "campaigns/glp1r_aleniglipron/track_a_generative/6XOX_PRISM_negative_image_pseudoligand_manifest.json"
)
DEFAULT_COMPACT_PSEUDOLIGAND = (
    REPO_ROOT
    / "campaigns/glp1r_aleniglipron/track_a_generative/6XOX_PRISM_compact_negative_image_pseudoligand.sdf"
)
DEFAULT_COMPACT_PSEUDOLIGAND_MANIFEST = (
    REPO_ROOT
    / "campaigns/glp1r_aleniglipron/track_a_generative/6XOX_PRISM_compact_negative_image_pseudoligand_manifest.json"
)
DEFAULT_O3A_FAILURE_REPORT = (
    REPO_ROOT / "campaigns/glp1r_aleniglipron/track_a_generative/o3a_failure_report.json"
)
DEFAULT_RISK_MAP = REPO_ROOT / "campaigns/glp1r_aleniglipron/integrated_spike_events/n80_full_scale/receptor_durability_risk_map.parquet"
DEFAULT_GRID_MAPPING = REPO_ROOT / "campaigns/glp1r_aleniglipron/track_0_manual_emulation/grid_coordinate_mapping.json"
DEFAULT_SIGNAL_GRID = (
    REPO_ROOT / "campaigns/glp1r_aleniglipron/integrated_spike_events/n80_full_scale/signal_grid_variance_channel.parquet"
)
DEFAULT_VOXEL_THRESHOLDS = REPO_ROOT / "campaigns/glp1r_aleniglipron/track_a_generative/voxel_thresholds.json"
DEFAULT_BOUNDARIES = REPO_ROOT / "campaigns/glp1r_aleniglipron/track_a_generative/gflownet_tso_bridge_boundaries.parquet"
DEFAULT_SOURCE_TOPOLOGY = WORKSPACE_TOPOLOGIES / "glp1r_5VEX_WT.topology.json"
DEFAULT_TARGET_TOPOLOGY = WORKSPACE_TOPOLOGIES / "glp1r_6XOX_WT.topology.json"
DEFAULT_SOURCE_CANDIDATES = (
    "glp1r_5VEX_WT",
    "glp1r_6LN2_WT",
    "glp1r_6LN2_A316T",
)
DEFAULT_SOURCE_CONDITION = "glp1r_5VEX_WT"
TARGET_CONDITION = "glp1r_6XOX_WT"
MIN_MATCHED_CA = 8
MINIMIZATION_SHELL_A = 8.0
POSITION_CONSTRAINT_MAX_DISPL_A = 1.5
POSITION_CONSTRAINT_FORCE = 5.0
RECEPTOR_CLASH_CUTOFF_A = 2.5
RECEPTOR_CONSTRAINT_MIN_A = 1.65
RECEPTOR_CONSTRAINT_MAX_A = 100.0
RECEPTOR_CONSTRAINT_FORCE = 150.0
O3A_CONFORMER_COUNT = 50
O3A_GUARD_CLASH_TOLERANCE = 0.25

JsonObject: TypeAlias = dict[str, Any]
FloatArray: TypeAlias = NDArray[np.float64]


Chem = cast(Any, import_module("rdkit.Chem"))
AllChem = cast(Any, import_module("rdkit.Chem.AllChem"))
rdMolAlign = cast(Any, import_module("rdkit.Chem.rdMolAlign"))
KDTree = cast(Any, import_module("scipy.spatial")).cKDTree


@dataclass(frozen=True)
class AlignmentCandidate:
    source_condition: str
    source_topology: Path
    residue_ids: tuple[int, ...]
    rotation: FloatArray
    translation: FloatArray
    pocket_rmsd_a: float
    pre_min_distance_a: float
    transformed_centroid: tuple[float, float, float]


@dataclass(frozen=True)
class FieldPoint:
    voxel_idx: int
    xyz: tuple[float, float, float]
    complement: float
    cryptic_bonus: float
    rigid_core: bool
    no_fly: bool


@dataclass(frozen=True)
class FieldModel:
    complement_points: tuple[FieldPoint, ...]
    cryptic_points: tuple[FieldPoint, ...]
    rigid_points: tuple[FieldPoint, ...]
    no_fly_points: tuple[FieldPoint, ...]


@dataclass(frozen=True)
class PoseMetrics:
    reference_type: str
    reference_atom_count: int
    probe_atom_count: int
    o3a_score: float | None
    o3a_rmsd: float | None
    ligand_centroid_after_o3a: tuple[float, float, float]
    grid_inside_status: bool
    min_receptor_heavy_distance_A: float
    rigid_core_clash_score: float
    no_fly_min_distance_A: float
    complement_overlap_score: float
    cryptic_overlap_score: float
    ligand_strain_penalty: float
    grid_boundary_penalty: float
    guard_level_passed: str | None
    repairable_guard_passed: bool
    reason_failed_guard: str | None
    selection_score: float
    method: str
    conformer_id: int | None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-kabsch", type=Path, default=DEFAULT_OUTPUT_KABSCH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_MINIMIZED)
    parser.add_argument("--output-o3a", type=Path, default=DEFAULT_OUTPUT_O3A)
    parser.add_argument("--output-o3a-best", type=Path, default=DEFAULT_OUTPUT_O3A_BEST)
    parser.add_argument("--output-o3a-relaxed", type=Path, default=DEFAULT_OUTPUT_O3A_RELAXED)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--pose-manifest", type=Path, default=DEFAULT_POSE_MANIFEST)
    parser.add_argument("--reference-ligand", type=Path, default=None)
    parser.add_argument("--pseudoligand", type=Path, default=DEFAULT_PSEUDOLIGAND)
    parser.add_argument("--pseudoligand-manifest", type=Path, default=DEFAULT_PSEUDOLIGAND_MANIFEST)
    parser.add_argument("--compact-pseudoligand", type=Path, default=DEFAULT_COMPACT_PSEUDOLIGAND)
    parser.add_argument("--compact-pseudoligand-manifest", type=Path, default=DEFAULT_COMPACT_PSEUDOLIGAND_MANIFEST)
    parser.add_argument("--o3a-failure-report", type=Path, default=DEFAULT_O3A_FAILURE_REPORT)
    parser.add_argument("--risk-map", type=Path, default=DEFAULT_RISK_MAP)
    parser.add_argument("--grid-mapping", type=Path, default=DEFAULT_GRID_MAPPING)
    parser.add_argument("--signal-grid", type=Path, default=DEFAULT_SIGNAL_GRID)
    parser.add_argument("--voxel-thresholds", type=Path, default=DEFAULT_VOXEL_THRESHOLDS)
    parser.add_argument("--boundaries", type=Path, default=DEFAULT_BOUNDARIES)
    parser.add_argument("--source-topology", type=Path, default=DEFAULT_SOURCE_TOPOLOGY)
    parser.add_argument("--target-topology", type=Path, default=DEFAULT_TARGET_TOPOLOGY)
    parser.add_argument("--source-condition", action="append", default=None)
    parser.add_argument("--topology-dir", type=Path, default=WORKSPACE_TOPOLOGIES)
    parser.add_argument("--pocket-residues", type=int, default=24)
    parser.add_argument("--minimize-steps", type=int, default=500)
    parser.add_argument("--o3a-conformers", type=int, default=O3A_CONFORMER_COUNT)
    return parser.parse_args()


def emit(message: str) -> None:
    sys.stdout.write(message + "\n")


def read_json_object(path: Path) -> JsonObject:
    decoded = json.loads(path.read_text())
    if not isinstance(decoded, dict):
        raise ValueError(f"{path} did not contain a JSON object")
    return cast(JsonObject, decoded)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def vector_to_list(values: FloatArray) -> list[float]:
    return [float(value) for value in values.tolist()]


def load_single_mol(path: Path) -> Any:
    supplier = Chem.SDMolSupplier(str(path), removeHs=False)
    mol = supplier[0] if len(supplier) else None
    if mol is None:
        raise ValueError(f"failed to parse SDF: {path}")
    if int(mol.GetNumConformers()) != 1:
        raise ValueError(f"expected one conformer in {path}")
    return mol


def topology_points(topology: JsonObject) -> list[tuple[float, float, float]]:
    raw = topology["positions"]
    if not isinstance(raw, list) or len(raw) % 3 != 0:
        raise ValueError("topology positions must be a flat xyz array")
    values = [float(value) for value in raw]
    return [(values[index], values[index + 1], values[index + 2]) for index in range(0, len(values), 3)]


def residue_index_by_residue_id(topology: JsonObject) -> dict[int, int]:
    residues = topology["residues"]
    if not isinstance(residues, list):
        raise ValueError("topology residues must be a list")
    mapping: dict[int, int] = {}
    for residue in residues:
        if not isinstance(residue, dict):
            continue
        mapping[int(residue["residue_id"])] = int(residue["residue_idx"])
    return mapping


def ca_coordinate(topology: JsonObject, residue_idx: int) -> FloatArray:
    ca_indices = cast(list[Any], topology["ca_indices"])
    positions = cast(list[Any], topology["positions"])
    atom_idx = int(ca_indices[residue_idx])
    return np.array(
        [float(positions[3 * atom_idx]), float(positions[3 * atom_idx + 1]), float(positions[3 * atom_idx + 2])],
        dtype=np.float64,
    )


def pocket_residue_ids(
    risk_map: Path,
    source_topology: JsonObject,
    target_topology: JsonObject,
    condition_id: str,
    limit: int,
) -> tuple[int, ...]:
    source_residue_ids = residue_index_by_residue_id(source_topology)
    target_residue_ids = residue_index_by_residue_id(target_topology)
    source_rows = cast(list[dict[str, object]], source_topology["residues"])
    rows = (
        pl.scan_parquet(risk_map)
        .filter((pl.col("condition_id") == condition_id) & (pl.col("edge_class") == "pocket_vector"))
        .sort("durability_risk_score_raw", descending=True)
        .select(["edge_from_residue", "edge_to_residue"])
        .collect()
        .to_dicts()
    )
    residue_ids: list[int] = []
    for row in rows:
        for column in ("edge_from_residue", "edge_to_residue"):
            residue_idx = int(row[column])
            residue_value = source_rows[residue_idx]["residue_id"]
            if not isinstance(residue_value, int | float | str):
                raise TypeError(f"invalid residue_id value: {residue_value!r}")
            residue_id = int(residue_value)
            if residue_id in source_residue_ids and residue_id in target_residue_ids and residue_id not in residue_ids:
                residue_ids.append(residue_id)
        if len(residue_ids) >= limit:
            break
    if len(residue_ids) < MIN_MATCHED_CA:
        raise ValueError(
            f"not enough shared pocket residues for {condition_id}: "
            f"{len(residue_ids)} found, {MIN_MATCHED_CA} required"
        )
    return tuple(residue_ids)


def kabsch(source: FloatArray, target: FloatArray) -> tuple[FloatArray, FloatArray, float]:
    source_centroid = source.mean(axis=0)
    target_centroid = target.mean(axis=0)
    centered_source = source - source_centroid
    centered_target = target - target_centroid
    covariance = centered_source.T @ centered_target
    u_mat, _singular_values, vt_mat = np.linalg.svd(covariance)
    rotation = vt_mat.T @ u_mat.T
    if float(np.linalg.det(rotation)) < 0.0:
        vt_mat[-1, :] *= -1.0
        rotation = vt_mat.T @ u_mat.T
    translation = target_centroid - rotation @ source_centroid
    aligned = (source @ rotation.T) + translation
    rmsd = float(np.sqrt(np.mean(np.sum((aligned - target) ** 2, axis=1))))
    return rotation.astype(np.float64), translation.astype(np.float64), rmsd


def ligand_heavy_coordinates(mol: Any, rotation: FloatArray | None = None, translation: FloatArray | None = None) -> FloatArray:
    conformer = mol.GetConformer(0)
    points: list[list[float]] = []
    for atom in mol.GetAtoms():
        if int(atom.GetAtomicNum()) <= 1:
            continue
        pos = conformer.GetAtomPosition(int(atom.GetIdx()))
        point = np.array([float(pos.x), float(pos.y), float(pos.z)], dtype=np.float64)
        if rotation is not None and translation is not None:
            point = rotation @ point + translation
        points.append([float(point[0]), float(point[1]), float(point[2])])
    return np.array(points, dtype=np.float64)


def ligand_atom_coordinates(mol: Any) -> FloatArray:
    conformer = mol.GetConformer(0)
    points: list[list[float]] = []
    for atom_idx in range(int(mol.GetNumAtoms())):
        pos = conformer.GetAtomPosition(atom_idx)
        points.append([float(pos.x), float(pos.y), float(pos.z)])
    return np.array(points, dtype=np.float64)


def ligand_heavy_atom_indices(mol: Any) -> list[int]:
    return [int(atom.GetIdx()) for atom in mol.GetAtoms() if int(atom.GetAtomicNum()) > 1]


def ligand_centroid(mol: Any) -> tuple[float, float, float]:
    points = ligand_heavy_coordinates(mol)
    centroid = points.mean(axis=0)
    return (float(centroid[0]), float(centroid[1]), float(centroid[2]))


def grid_bounds_6xox(grid_mapping: Path) -> dict[str, list[float] | list[int] | float]:
    payload = read_json_object(grid_mapping)
    conditions = payload.get("conditions")
    if not isinstance(conditions, dict) or TARGET_CONDITION not in conditions:
        raise ValueError(f"{grid_mapping} does not contain {TARGET_CONDITION}")
    condition = conditions[TARGET_CONDITION]
    if not isinstance(condition, dict):
        raise ValueError(f"invalid {TARGET_CONDITION} grid mapping")
    origin_raw = condition["origin_xyz_angstrom"]
    if not isinstance(origin_raw, list) or len(origin_raw) != 3:
        raise ValueError(f"{TARGET_CONDITION} origin_xyz_angstrom must have three values")
    origin = [float(value) for value in origin_raw]
    spacing = float(condition["spacing_angstrom"])
    nx = int(condition.get("nx", condition.get("grid_dim", 96)))
    ny = int(condition.get("ny", condition.get("grid_dim", 96)))
    nz = int(condition.get("nz", condition.get("grid_dim", 96)))
    max_xyz = [origin[0] + nx * spacing, origin[1] + ny * spacing, origin[2] + nz * spacing]
    return {
        "min_xyz": origin,
        "max_xyz": max_xyz,
        "grid_shape": [nx, ny, nz],
        "spacing_A": spacing,
    }


def ligand_inside_grid(mol: Any, bounds: dict[str, list[float] | list[int] | float]) -> bool:
    min_xyz = cast(list[float], bounds["min_xyz"])
    max_xyz = cast(list[float], bounds["max_xyz"])
    for point in ligand_heavy_coordinates(mol):
        if any(float(point[axis]) < min_xyz[axis] or float(point[axis]) > max_xyz[axis] for axis in range(3)):
            return False
    return True


def receptor_heavy_coordinates(topology: JsonObject) -> FloatArray:
    points = topology_points(topology)
    elements = [str(value) for value in cast(list[Any], topology["elements"])]
    heavy = [points[index] for index, element in enumerate(elements) if element.upper() not in {"H", "D"}]
    return np.array(heavy, dtype=np.float64)


def min_pairwise_distance(left: FloatArray, right: FloatArray) -> float:
    best = math.inf
    for point in left:
        distances = np.linalg.norm(right - point, axis=1)
        best = min(best, float(distances.min()))
    return best


def build_candidate(
    *,
    source_condition: str,
    source_topology_path: Path,
    target_topology_path: Path,
    risk_map: Path,
    mol: Any,
    pocket_limit: int,
) -> AlignmentCandidate:
    source_topology = read_json_object(source_topology_path)
    target_topology = read_json_object(target_topology_path)
    residue_ids = pocket_residue_ids(risk_map, source_topology, target_topology, source_condition, pocket_limit)
    source_by_id = residue_index_by_residue_id(source_topology)
    target_by_id = residue_index_by_residue_id(target_topology)
    source_points = np.vstack([ca_coordinate(source_topology, source_by_id[residue_id]) for residue_id in residue_ids])
    target_points = np.vstack([ca_coordinate(target_topology, target_by_id[residue_id]) for residue_id in residue_ids])
    rotation, translation, rmsd = kabsch(source_points, target_points)
    ligand_points = ligand_heavy_coordinates(mol, rotation, translation)
    receptor_points = receptor_heavy_coordinates(target_topology)
    pre_min = min_pairwise_distance(ligand_points, receptor_points)
    centroid = ligand_points.mean(axis=0)
    return AlignmentCandidate(
        source_condition=source_condition,
        source_topology=source_topology_path,
        residue_ids=residue_ids,
        rotation=rotation,
        translation=translation,
        pocket_rmsd_a=rmsd,
        pre_min_distance_a=pre_min,
        transformed_centroid=(float(centroid[0]), float(centroid[1]), float(centroid[2])),
    )


def select_candidate(candidates: list[AlignmentCandidate]) -> AlignmentCandidate:
    plausible = [candidate for candidate in candidates if candidate.pre_min_distance_a < 8.0]
    if plausible:
        return sorted(plausible, key=lambda item: (abs(item.pre_min_distance_a - 2.2), item.pocket_rmsd_a))[0]
    return sorted(candidates, key=lambda item: item.pre_min_distance_a)[0]


def apply_transform(mol: Any, rotation: FloatArray, translation: FloatArray) -> None:
    conformer = mol.GetConformer(0)
    for atom_idx in range(int(mol.GetNumAtoms())):
        pos = conformer.GetAtomPosition(atom_idx)
        point = rotation @ np.array([float(pos.x), float(pos.y), float(pos.z)], dtype=np.float64) + translation
        conformer.SetAtomPosition(atom_idx, (float(point[0]), float(point[1]), float(point[2])))


def build_shell_complex(ligand: Any, receptor_topology: JsonObject) -> tuple[Any, list[int], list[int]]:
    ligand_coords = ligand_heavy_coordinates(ligand)
    receptor_points_all = topology_points(receptor_topology)
    elements = [str(value) for value in cast(list[Any], receptor_topology["elements"])]
    selected_receptor_indices: list[int] = []
    for index, (element, point) in enumerate(zip(elements, receptor_points_all, strict=True)):
        if element.upper() in {"H", "D"}:
            continue
        point_array = np.array(point, dtype=np.float64)
        if float(np.linalg.norm(ligand_coords - point_array, axis=1).min()) <= MINIMIZATION_SHELL_A:
            selected_receptor_indices.append(index)
    rw_mol = Chem.RWMol()
    receptor_complex_indices: list[int] = []
    for atom_index in selected_receptor_indices:
        atom = Chem.Atom(elements[atom_index])
        receptor_complex_indices.append(int(rw_mol.AddAtom(atom)))
    ligand_complex_indices: list[int] = []
    ligand_offset = int(rw_mol.GetNumAtoms())
    for atom in ligand.GetAtoms():
        new_atom = Chem.Atom(int(atom.GetAtomicNum()))
        new_atom.SetFormalCharge(int(atom.GetFormalCharge()))
        ligand_complex_indices.append(int(rw_mol.AddAtom(new_atom)))
    for bond in ligand.GetBonds():
        rw_mol.AddBond(
            ligand_offset + int(bond.GetBeginAtomIdx()),
            ligand_offset + int(bond.GetEndAtomIdx()),
            bond.GetBondType(),
        )
    complex_mol = rw_mol.GetMol()
    conformer = Chem.Conformer(int(complex_mol.GetNumAtoms()))
    for complex_idx, topo_idx in enumerate(selected_receptor_indices):
        x_coord, y_coord, z_coord = receptor_points_all[topo_idx]
        conformer.SetAtomPosition(complex_idx, (x_coord, y_coord, z_coord))
    ligand_conformer = ligand.GetConformer(0)
    for atom_idx in range(int(ligand.GetNumAtoms())):
        pos = ligand_conformer.GetAtomPosition(atom_idx)
        conformer.SetAtomPosition(ligand_offset + atom_idx, (float(pos.x), float(pos.y), float(pos.z)))
    complex_mol.AddConformer(conformer, assignId=True)
    complex_mol.UpdatePropertyCache(strict=False)
    return complex_mol, receptor_complex_indices, ligand_complex_indices


def minimize_ligand_in_shell(ligand: Any, receptor_topology: JsonObject, steps: int) -> float:
    ligand_positions = ligand_atom_coordinates(ligand)
    receptor_points = receptor_heavy_coordinates(receptor_topology)
    props = None
    force_field = None
    use_mmff = bool(AllChem.MMFFHasAllMoleculeParams(ligand))
    if use_mmff:
        props = AllChem.MMFFGetMoleculeProperties(ligand, mmffVariant="MMFF94s")
        force_field = AllChem.MMFFGetMoleculeForceField(ligand, props, confId=0)
    if force_field is None:
        force_field = AllChem.UFFGetMoleculeForceField(ligand, confId=0)
    if force_field is None:
        raise ValueError("ligand force field could not be constructed")

    for atom_idx in range(int(ligand.GetNumAtoms())):
        if use_mmff:
            force_field.MMFFAddPositionConstraint(
                atom_idx,
                POSITION_CONSTRAINT_MAX_DISPL_A,
                POSITION_CONSTRAINT_FORCE,
            )
        else:
            force_field.UFFAddPositionConstraint(
                atom_idx,
                POSITION_CONSTRAINT_MAX_DISPL_A,
                POSITION_CONSTRAINT_FORCE,
            )

    ligand_heavy = ligand_heavy_atom_indices(ligand)
    receptor_extra_points: dict[int, int] = {}
    constraint_count = 0
    for ligand_atom_idx in ligand_heavy:
        ligand_point = ligand_positions[ligand_atom_idx]
        distances = np.linalg.norm(receptor_points - ligand_point, axis=1)
        close_indices = np.where(distances < RECEPTOR_CLASH_CUTOFF_A)[0]
        for receptor_index in [int(value) for value in close_indices.tolist()]:
            if receptor_index not in receptor_extra_points:
                receptor_point = receptor_points[receptor_index]
                force_field.AddExtraPoint(
                    float(receptor_point[0]),
                    float(receptor_point[1]),
                    float(receptor_point[2]),
                    True,
                )
                receptor_extra_points[receptor_index] = int(force_field.NumPoints()) - 1
            force_field.AddDistanceConstraint(
                ligand_atom_idx,
                receptor_extra_points[receptor_index],
                RECEPTOR_CONSTRAINT_MIN_A,
                RECEPTOR_CONSTRAINT_MAX_A,
                RECEPTOR_CONSTRAINT_FORCE,
            )
            constraint_count += 1
    if constraint_count == 0:
        return 0.0
    force_field.Initialize()
    force_field.Minimize(maxIts=steps)
    rigid_clash_relief(ligand, receptor_topology, RECEPTOR_CONSTRAINT_MIN_A)
    windowed_rigid_contact_relief(ligand, receptor_topology, RECEPTOR_CONSTRAINT_MIN_A, 3.5)
    return float(force_field.CalcEnergy())


def nearest_contact_vector(ligand: Any, receptor_topology: JsonObject) -> tuple[float, FloatArray]:
    receptor_points = receptor_heavy_coordinates(receptor_topology)
    conformer = ligand.GetConformer(0)
    best_distance = math.inf
    best_vector = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    for atom in ligand.GetAtoms():
        if int(atom.GetAtomicNum()) <= 1:
            continue
        pos = conformer.GetAtomPosition(int(atom.GetIdx()))
        ligand_point = np.array([float(pos.x), float(pos.y), float(pos.z)], dtype=np.float64)
        distances = np.linalg.norm(receptor_points - ligand_point, axis=1)
        nearest_index = int(np.argmin(distances))
        current = float(distances[nearest_index])
        if current < best_distance:
            best_distance = current
            vector = ligand_point - receptor_points[nearest_index]
            norm_value = float(np.linalg.norm(vector))
            best_vector = vector / norm_value if norm_value > 1.0e-9 else best_vector
    return best_distance, best_vector


def translate_ligand(ligand: Any, displacement: FloatArray) -> None:
    conformer = ligand.GetConformer(0)
    for atom_idx in range(int(ligand.GetNumAtoms())):
        pos = conformer.GetAtomPosition(atom_idx)
        conformer.SetAtomPosition(
            atom_idx,
            (
                float(pos.x) + float(displacement[0]),
                float(pos.y) + float(displacement[1]),
                float(pos.z) + float(displacement[2]),
            ),
        )


def rigid_clash_relief(ligand: Any, receptor_topology: JsonObject, target_min_distance: float) -> None:
    total_displacement = np.array([0.0, 0.0, 0.0], dtype=np.float64)
    for _step in range(20):
        min_distance, direction = nearest_contact_vector(ligand, receptor_topology)
        if min_distance >= target_min_distance:
            return
        step = min(target_min_distance - min_distance + 0.05, 0.25)
        displacement = direction * step
        total_displacement = total_displacement + displacement
        if float(np.linalg.norm(total_displacement)) > POSITION_CONSTRAINT_MAX_DISPL_A:
            return
        translate_ligand(ligand, displacement)


def fibonacci_directions(count: int) -> FloatArray:
    directions: list[list[float]] = [
        [1.0, 0.0, 0.0],
        [-1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, -1.0, 0.0],
        [0.0, 0.0, 1.0],
        [0.0, 0.0, -1.0],
    ]
    golden_angle = math.pi * (3.0 - math.sqrt(5.0))
    for index in range(count):
        y_coord = 1.0 - (2.0 * (index + 0.5) / count)
        radius = math.sqrt(max(0.0, 1.0 - y_coord * y_coord))
        theta = golden_angle * index
        directions.append([math.cos(theta) * radius, y_coord, math.sin(theta) * radius])
    return np.array(directions, dtype=np.float64)


def min_distance_with_tree(points: FloatArray, tree: Any) -> float:
    distances = tree.query(points, k=1)[0]
    return float(np.min(distances))


def windowed_rigid_contact_relief(
    ligand: Any,
    receptor_topology: JsonObject,
    min_target: float,
    max_target: float,
) -> None:
    receptor_points = receptor_heavy_coordinates(receptor_topology)
    ligand_points = ligand_heavy_coordinates(ligand)
    tree = KDTree(receptor_points)
    current = min_distance_with_tree(ligand_points, tree)
    if min_target <= current <= max_target:
        return
    directions = fibonacci_directions(256)
    best_distance = current
    best_displacement = np.array([0.0, 0.0, 0.0], dtype=np.float64)
    for radius in np.linspace(0.05, 8.0, 160):
        for direction in directions:
            displacement = direction * float(radius)
            candidate_distance = min_distance_with_tree(ligand_points + displacement, tree)
            if min_target <= candidate_distance <= max_target:
                translate_ligand(ligand, displacement.astype(np.float64))
                return
            if candidate_distance > best_distance and candidate_distance <= max_target:
                best_distance = candidate_distance
                best_displacement = displacement.astype(np.float64)
    if best_distance > current:
        translate_ligand(ligand, best_displacement)


def copy_mol(mol: Any) -> Any:
    copied = Chem.Mol(mol)
    if int(copied.GetNumConformers()) != 1:
        raise ValueError("copied ligand lost its conformer")
    return copied


def write_plain_mol(mol: Any, output: Path, properties: dict[str, str] | None = None) -> None:
    if properties:
        for key, value in properties.items():
            mol.SetProp(key, value)
    output.parent.mkdir(parents=True, exist_ok=True)
    writer = Chem.SDWriter(str(output))
    writer.write(mol)
    writer.close()


def voxel_center_from_indices(
    x_idx: int,
    y_idx: int,
    z_idx: int,
    bounds: dict[str, list[float] | list[int] | float],
) -> tuple[float, float, float]:
    min_xyz = cast(list[float], bounds["min_xyz"])
    spacing = float(bounds["spacing_A"])
    return (
        min_xyz[0] + (x_idx + 0.5) * spacing,
        min_xyz[1] + (y_idx + 0.5) * spacing,
        min_xyz[2] + (z_idx + 0.5) * spacing,
    )


def coordinate_to_voxel_idx(point: FloatArray, bounds: dict[str, list[float] | list[int] | float]) -> int | None:
    min_xyz = cast(list[float], bounds["min_xyz"])
    shape = cast(list[int], bounds["grid_shape"])
    spacing = float(bounds["spacing_A"])
    indices: list[int] = []
    for axis in range(3):
        raw = math.floor((float(point[axis]) - min_xyz[axis]) / spacing)
        if raw < 0 or raw >= shape[axis]:
            return None
        indices.append(raw)
    return indices[2] * shape[0] * shape[1] + indices[1] * shape[0] + indices[0]


def load_field_model(
    *,
    signal_grid: Path,
    voxel_thresholds: Path,
    boundaries: Path,
    bounds: dict[str, list[float] | list[int] | float],
) -> FieldModel:
    thresholds = read_json_object(voxel_thresholds)
    cold_p80 = float(thresholds["cold_p80"])
    warm_p80 = float(thresholds["warm_p80"])
    release_p90 = float(thresholds["release_p90"])
    release_p95 = float(thresholds["release_p95"])
    rows = (
        pl.scan_parquet(signal_grid)
        .filter(pl.col("condition_id") == TARGET_CONDITION)
        .with_columns((pl.col("hit_count_cold_mean") - pl.col("hit_count_warm_mean")).alias("release_delta"))
        .filter(
            (pl.col("hit_count_cold_mean") >= cold_p80)
            | (pl.col("hit_count_warm_mean") >= max(warm_p80, 1.0e-9))
            | (pl.col("release_delta") >= release_p90)
        )
        .select(["voxel_idx", "x_idx", "y_idx", "z_idx", "hit_count_cold_mean", "hit_count_warm_mean", "release_delta"])
        .collect()
        .to_dicts()
    )
    no_fly_voxels: set[int] = set()
    boundary_no_fly_points: list[FieldPoint] = []
    if boundaries.exists():
        boundary_rows = (
            pl.scan_parquet(boundaries)
            .filter(
                (pl.col("condition_id") == TARGET_CONDITION)
                & (pl.col("boundary_class").cast(pl.Utf8).str.contains("no_fly"))
            )
            .select(["voxel_idx", "center_x_A", "center_y_A", "center_z_A"])
            .collect()
            .to_dicts()
        )
        no_fly_voxels = set(int(row["voxel_idx"]) for row in boundary_rows)
        boundary_no_fly_points = [
            FieldPoint(
                voxel_idx=int(row["voxel_idx"]),
                xyz=(float(row["center_x_A"]), float(row["center_y_A"]), float(row["center_z_A"])),
                complement=0.0,
                cryptic_bonus=0.0,
                rigid_core=False,
                no_fly=True,
            )
            for row in boundary_rows
        ]
    complement: list[FieldPoint] = []
    cryptic: list[FieldPoint] = []
    rigid: list[FieldPoint] = []
    no_fly: list[FieldPoint] = []
    for row in rows:
        voxel_idx = int(row["voxel_idx"])
        cold = float(row["hit_count_cold_mean"])
        warm = float(row["hit_count_warm_mean"])
        release = float(row["release_delta"])
        stable = cold >= cold_p80 and warm >= warm_p80
        released = release >= release_p90
        cryptic_bonus = 2.0 if release >= release_p95 else 0.0
        center = voxel_center_from_indices(int(row["x_idx"]), int(row["y_idx"]), int(row["z_idx"]), bounds)
        point = FieldPoint(
            voxel_idx=voxel_idx,
            xyz=center,
            complement=1.0 if released else 0.0,
            cryptic_bonus=cryptic_bonus,
            rigid_core=stable and not released,
            no_fly=voxel_idx in no_fly_voxels,
        )
        if point.complement > 0.0:
            complement.append(point)
        if point.cryptic_bonus > 0.0:
            cryptic.append(point)
        if point.rigid_core:
            rigid.append(point)
        if point.no_fly:
            no_fly.append(point)
    known_no_fly = {point.voxel_idx for point in no_fly}
    no_fly.extend(point for point in boundary_no_fly_points if point.voxel_idx not in known_no_fly)
    return FieldModel(
        complement_points=tuple(complement),
        cryptic_points=tuple(cryptic),
        rigid_points=tuple(rigid),
        no_fly_points=tuple(no_fly),
    )


def field_point_array(points: tuple[FieldPoint, ...]) -> FloatArray:
    if not points:
        return np.zeros((0, 3), dtype=np.float64)
    return np.array([point.xyz for point in points], dtype=np.float64)


def count_points_within(points: FloatArray, targets: FloatArray, radius: float) -> int:
    if points.size == 0 or targets.size == 0:
        return 0
    tree = KDTree(targets)
    distances = tree.query(points, k=1)[0]
    return int(np.sum(distances <= radius))


def min_distance_to_targets(points: FloatArray, targets: FloatArray) -> float:
    if points.size == 0 or targets.size == 0:
        return math.inf
    return min_distance_with_tree(points, KDTree(targets))


def build_pseudoligand_mol(points: list[tuple[float, float, float]]) -> Any:
    if not points:
        raise ValueError("cannot build pseudo-ligand without points")
    rw_mol = Chem.RWMol()
    for _point in points:
        rw_mol.AddAtom(Chem.Atom("C"))
    mol = rw_mol.GetMol()
    conformer = Chem.Conformer(int(mol.GetNumAtoms()))
    for atom_idx, point in enumerate(points):
        conformer.SetAtomPosition(atom_idx, point)
    mol.AddConformer(conformer, assignId=True)
    mol.SetProp("pseudo_ligand_source", "PRISM_negative_image_release_field")
    mol.UpdatePropertyCache(strict=False)
    Chem.SanitizeMol(mol, catchErrors=True)
    Chem.GetSymmSSSR(mol)
    return mol


def generate_prism_negative_image_pseudoligand(
    *,
    signal_grid: Path,
    voxel_thresholds: Path,
    bounds: dict[str, list[float] | list[int] | float],
    output: Path,
    manifest: Path,
    placed_ligand: Any,
) -> Any:
    thresholds = read_json_object(voxel_thresholds) if voxel_thresholds.exists() else {}
    release_threshold = float(thresholds.get("release_p90", 0.0))
    ligand_points = ligand_heavy_coordinates(placed_ligand)
    ligand_centroid_array = ligand_points.mean(axis=0)
    selected = (
        pl.scan_parquet(signal_grid)
        .filter(pl.col("condition_id") == TARGET_CONDITION)
        .with_columns((pl.col("hit_count_cold_mean") - pl.col("hit_count_warm_mean")).alias("release_delta"))
        .filter(pl.col("release_delta") >= release_threshold)
        .sort("release_delta", descending=True)
        .select(["voxel_idx", "x_idx", "y_idx", "z_idx", "release_delta", "variance_class"])
        .head(2000)
        .collect()
    )
    rows = selected.to_dicts()
    candidate_points: list[tuple[float, float, float, float, str]] = []
    for row in rows:
        center = voxel_center_from_indices(int(row["x_idx"]), int(row["y_idx"]), int(row["z_idx"]), bounds)
        distance_to_ligand = float(np.linalg.norm(np.array(center, dtype=np.float64) - ligand_centroid_array))
        if distance_to_ligand > 18.0:
            continue
        candidate_points.append(
            (
                center[0],
                center[1],
                center[2],
                float(row["release_delta"]),
                str(row.get("variance_class", "")),
            )
        )
    if not candidate_points:
        for row in rows[:32]:
            center = voxel_center_from_indices(int(row["x_idx"]), int(row["y_idx"]), int(row["z_idx"]), bounds)
            candidate_points.append(
                (
                    center[0],
                    center[1],
                    center[2],
                    float(row["release_delta"]),
                    str(row.get("variance_class", "")),
                )
            )
    representative_points: list[tuple[float, float, float]] = []
    representative_rows: list[dict[str, object]] = []
    for x_coord, y_coord, z_coord, release_delta, variance_class in candidate_points:
        point_array = np.array([x_coord, y_coord, z_coord], dtype=np.float64)
        if any(
            float(np.linalg.norm(point_array - np.array(existing, dtype=np.float64))) < 2.5
            for existing in representative_points
        ):
            continue
        representative_points.append((x_coord, y_coord, z_coord))
        representative_rows.append(
            {
                "x": x_coord,
                "y": y_coord,
                "z": z_coord,
                "release_delta": release_delta,
                "variance_class": variance_class,
            }
        )
        if len(representative_points) >= 16:
            break
    pseudo = build_pseudoligand_mol(representative_points)
    write_plain_mol(
        pseudo,
        output,
        {
            "selected_alignment_reference": "PRISM_negative_image_pseudoligand",
            "representative_pseudo_atom_count": str(len(representative_points)),
        },
    )
    manifest_payload = {
        "source_signal_grid": signal_grid.as_posix(),
        "source_voxel_thresholds": voxel_thresholds.as_posix(),
        "target_condition": TARGET_CONDITION,
        "release_delta_threshold": release_threshold,
        "pseudo_atom_count": len(representative_points),
        "representative_points": representative_rows,
        "sha256": {
            "source_signal_grid": sha256_file(signal_grid),
            "source_voxel_thresholds": sha256_file(voxel_thresholds) if voxel_thresholds.exists() else None,
            "output_pseudoligand": sha256_file(output),
        },
    }
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(json.dumps(manifest_payload, indent=2, sort_keys=True) + "\n")
    return pseudo


def generate_compact_negative_image_pseudoligand(
    *,
    field_model: FieldModel,
    placed_ligand: Any,
    output: Path,
    manifest: Path,
) -> Any:
    ligand_points = ligand_heavy_coordinates(placed_ligand)
    ligand_centroid_array = ligand_points.mean(axis=0)
    complement_points = [point for point in field_model.complement_points if not point.no_fly and not point.rigid_core]
    source_voxel_count = len(complement_points)
    selected: list[tuple[FieldPoint, float, float]] = []
    no_fly_array = field_point_array(field_model.no_fly_points)
    rigid_array = field_point_array(field_model.rigid_points)
    no_fly_excluded = 0
    rigid_excluded = 0
    for point in complement_points:
        xyz = np.array(point.xyz, dtype=np.float64)
        if float(np.linalg.norm(xyz - ligand_centroid_array)) > 8.0:
            continue
        if min_distance_to_targets(xyz.reshape(1, 3), no_fly_array) < 1.0:
            no_fly_excluded += 1
            continue
        if min_distance_to_targets(xyz.reshape(1, 3), rigid_array) < 0.75:
            rigid_excluded += 1
            continue
        nearest_ligand = float(np.linalg.norm(ligand_points - xyz, axis=1).min())
        weight = point.cryptic_bonus * 4.0 + point.complement * 2.0 + max(0.0, 8.0 - nearest_ligand)
        selected.append((point, weight, nearest_ligand))
    selected.sort(key=lambda row: row[1], reverse=True)
    representative: list[tuple[FieldPoint, float, float]] = []
    for point, weight, nearest_ligand in selected:
        xyz = np.array(point.xyz, dtype=np.float64)
        if any(float(np.linalg.norm(xyz - np.array(existing[0].xyz, dtype=np.float64))) < 1.75 for existing in representative):
            continue
        representative.append((point, weight, nearest_ligand))
        if len(representative) >= 24:
            break
    if len(representative) < 12:
        for point, weight, nearest_ligand in selected:
            if (point, weight, nearest_ligand) not in representative:
                representative.append((point, weight, nearest_ligand))
            if len(representative) >= 12:
                break
    if not representative:
        raise ValueError("compact pseudo-ligand selection produced zero pseudo-atoms")
    pseudo_points = [row[0].xyz for row in representative]
    pseudo = build_pseudoligand_mol(pseudo_points)
    write_plain_mol(
        pseudo,
        output,
        {
            "selected_alignment_reference": "PRISM_compact_negative_image_pseudoligand",
            "representative_pseudo_atom_count": str(len(pseudo_points)),
        },
    )
    pseudo_array = np.array(pseudo_points, dtype=np.float64)
    bbox_min = pseudo_array.min(axis=0)
    bbox_max = pseudo_array.max(axis=0)
    distances_to_ligand = [row[2] for row in representative]
    manifest_payload = {
        "target_condition": TARGET_CONDITION,
        "source_voxel_count": source_voxel_count,
        "selected_voxel_count": len(selected),
        "pseudo_atom_count": len(pseudo_points),
        "centroid_xyz": vector_to_list(pseudo_array.mean(axis=0)),
        "bbox_xyz": {"min": vector_to_list(bbox_min), "max": vector_to_list(bbox_max)},
        "min_distance_to_kabsch_ligand_A": min(distances_to_ligand),
        "max_distance_to_kabsch_ligand_A": max(distances_to_ligand),
        "no_fly_excluded_count": no_fly_excluded,
        "rigid_core_excluded_count": rigid_excluded,
        "representative_points": [
            {
                "voxel_idx": row[0].voxel_idx,
                "x": row[0].xyz[0],
                "y": row[0].xyz[1],
                "z": row[0].xyz[2],
                "complement": row[0].complement,
                "cryptic_bonus": row[0].cryptic_bonus,
                "selection_weight": row[1],
                "distance_to_kabsch_ligand_A": row[2],
            }
            for row in representative
        ],
        "sha256": {"output_pseudoligand": sha256_file(output)},
    }
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(json.dumps(manifest_payload, indent=2, sort_keys=True) + "\n")
    return pseudo


def prepare_o3a_mol(mol: Any) -> Any:
    prepared = copy_mol(mol)
    prepared.UpdatePropertyCache(strict=False)
    Chem.SanitizeMol(prepared, catchErrors=True)
    Chem.GetSymmSSSR(prepared)
    return prepared


def try_o3a_alignment(probe: Any, reference: Any) -> tuple[Any | None, float | None, float | None, str | None]:
    aligned = copy_mol(probe)
    aligned = prepare_o3a_mol(aligned)
    reference = prepare_o3a_mol(reference)
    try:
        o3a = rdMolAlign.GetO3A(aligned, reference)
        score = float(o3a.Score())
        rmsd = float(o3a.Align())
        return aligned, score, rmsd, None
    except Exception as exc:
        try:
            o3a = rdMolAlign.GetCrippenO3A(aligned, reference)
            score = float(o3a.Score())
            rmsd = float(o3a.Align())
            return aligned, score, rmsd, None
        except Exception as fallback_exc:
            return None, None, None, f"O3A failed: {exc}; CrippenO3A failed: {fallback_exc}"


def forcefield_energy(mol: Any) -> float:
    try:
        if AllChem.MMFFHasAllMoleculeParams(mol):
            props = AllChem.MMFFGetMoleculeProperties(mol, mmffVariant="MMFF94s")
            ff = AllChem.MMFFGetMoleculeForceField(mol, props, confId=0)
        else:
            ff = AllChem.UFFGetMoleculeForceField(mol, confId=0)
        if ff is None:
            return 0.0
        return float(ff.CalcEnergy())
    except Exception:
        return 0.0


def evaluate_pose(
    *,
    mol: Any,
    reference_type: str,
    reference_atom_count: int,
    o3a_score: float | None,
    o3a_rmsd: float | None,
    target_topology: JsonObject,
    bounds: dict[str, list[float] | list[int] | float],
    field_model: FieldModel,
    ligand_strain_penalty: float,
    method: str,
    conformer_id: int | None,
    pre_min_distance: float | None = None,
) -> PoseMetrics:
    ligand_points = ligand_heavy_coordinates(mol)
    receptor_distance = min_pairwise_distance(ligand_points, receptor_heavy_coordinates(target_topology))
    inside = ligand_inside_grid(mol, bounds)
    grid_penalty = 0.0 if inside else 100.0
    complement_array = field_point_array(field_model.complement_points)
    cryptic_array = field_point_array(field_model.cryptic_points)
    rigid_array = field_point_array(field_model.rigid_points)
    no_fly_array = field_point_array(field_model.no_fly_points)
    complement_overlap = float(count_points_within(ligand_points, complement_array, 1.2))
    cryptic_overlap = float(count_points_within(ligand_points, cryptic_array, 1.2))
    rigid_clash = float(count_points_within(ligand_points, rigid_array, 1.2))
    no_fly_min = min_distance_to_targets(ligand_points, no_fly_array)
    no_fly_intrusion = max(0.0, 4.0 - no_fly_min)
    repairable = (
        pre_min_distance is not None
        and pre_min_distance >= 1.25
        and receptor_distance > pre_min_distance
        and receptor_distance >= 1.25
    )
    guard_level: str | None = None
    guard_failures: list[str] = []
    if inside:
        if receptor_distance >= 1.70:
            guard_level = "strict"
        elif receptor_distance >= 1.50:
            guard_level = "standard"
        elif repairable:
            guard_level = "repairable"
    else:
        guard_failures.append("ligand_outside_grid")
    if no_fly_min < 4.0:
        guard_failures.append(f"no_fly_min_distance_A={no_fly_min:.3f}<4.0")
    if guard_level is None and inside:
        guard_failures.append(f"min_receptor_heavy_distance_A={receptor_distance:.3f}<1.50_and_not_repairable")
    selection_score = (
        complement_overlap
        + cryptic_overlap
        - rigid_clash
        - no_fly_intrusion * 10.0
        - ligand_strain_penalty
        - grid_penalty
    )
    return PoseMetrics(
        reference_type=reference_type,
        reference_atom_count=reference_atom_count,
        probe_atom_count=int(mol.GetNumAtoms()),
        o3a_score=o3a_score,
        o3a_rmsd=o3a_rmsd,
        ligand_centroid_after_o3a=ligand_centroid(mol),
        grid_inside_status=inside,
        min_receptor_heavy_distance_A=receptor_distance,
        rigid_core_clash_score=rigid_clash,
        no_fly_min_distance_A=no_fly_min,
        complement_overlap_score=complement_overlap,
        cryptic_overlap_score=cryptic_overlap,
        ligand_strain_penalty=ligand_strain_penalty,
        grid_boundary_penalty=grid_penalty,
        guard_level_passed=guard_level,
        repairable_guard_passed=guard_level == "repairable",
        reason_failed_guard="; ".join(guard_failures) if guard_failures else None,
        selection_score=selection_score,
        method=method,
        conformer_id=conformer_id,
    )


def pose_metrics_to_json(metrics: PoseMetrics) -> JsonObject:
    return {
        "reference_type": metrics.reference_type,
        "reference_atom_count": metrics.reference_atom_count,
        "probe_atom_count": metrics.probe_atom_count,
        "o3a_score": metrics.o3a_score,
        "o3a_rmsd": metrics.o3a_rmsd,
        "ligand_centroid_after_o3a": list(metrics.ligand_centroid_after_o3a),
        "grid_inside_status": metrics.grid_inside_status,
        "min_receptor_heavy_distance_A": metrics.min_receptor_heavy_distance_A,
        "rigid_core_clash_score": metrics.rigid_core_clash_score,
        "no_fly_min_distance_A": metrics.no_fly_min_distance_A,
        "complement_overlap_score": metrics.complement_overlap_score,
        "cryptic_overlap_score": metrics.cryptic_overlap_score,
        "ligand_strain_penalty": metrics.ligand_strain_penalty,
        "grid_boundary_penalty": metrics.grid_boundary_penalty,
        "guard_level_passed": metrics.guard_level_passed,
        "repairable_guard_passed": metrics.repairable_guard_passed,
        "reason_failed_guard": metrics.reason_failed_guard,
        "selection_score": metrics.selection_score,
        "method": metrics.method,
        "conformer_id": metrics.conformer_id,
    }


def generate_probe_conformers(mol: Any, count: int) -> list[Any]:
    base = Chem.Mol(mol)
    base.RemoveAllConformers()
    params = AllChem.ETKDGv3()
    params.randomSeed = 61453
    params.pruneRmsThresh = 0.25
    conf_ids = list(AllChem.EmbedMultipleConfs(base, numConfs=count, params=params))
    if not conf_ids:
        return [copy_mol(mol)]
    conformers: list[Any] = []
    for conf_id in conf_ids:
        conformer_mol = Chem.Mol(base)
        conf = base.GetConformer(conf_id)
        conformer_mol.RemoveAllConformers()
        conformer_mol.AddConformer(Chem.Conformer(conf), assignId=True)
        try:
            if AllChem.MMFFHasAllMoleculeParams(conformer_mol):
                AllChem.MMFFOptimizeMolecule(conformer_mol, mmffVariant="MMFF94s", maxIters=100)
            else:
                AllChem.UFFOptimizeMolecule(conformer_mol, maxIters=100)
        except Exception:
            pass
        conformers.append(conformer_mol)
    return conformers


def write_o3a_failure_report(path: Path, attempts: list[PoseMetrics]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps([pose_metrics_to_json(item) for item in attempts], indent=2, sort_keys=True) + "\n")


def attempt_reference_alignment(
    *,
    probes: list[Any],
    reference: Any,
    reference_type: str,
    target_topology: JsonObject,
    bounds: dict[str, list[float] | list[int] | float],
    field_model: FieldModel,
    minimize_steps: int,
    baseline_energy: float,
) -> list[tuple[Any, PoseMetrics, Any]]:
    attempts: list[tuple[Any, PoseMetrics, Any]] = []
    reference_atom_count = int(reference.GetNumAtoms())
    for conf_id, probe in enumerate(probes):
        aligned, o3a_score, o3a_rmsd, error = try_o3a_alignment(probe, reference)
        if aligned is None:
            metrics = PoseMetrics(
                reference_type=reference_type,
                reference_atom_count=reference_atom_count,
                probe_atom_count=int(probe.GetNumAtoms()),
                o3a_score=o3a_score,
                o3a_rmsd=o3a_rmsd,
                ligand_centroid_after_o3a=ligand_centroid(probe),
                grid_inside_status=False,
                min_receptor_heavy_distance_A=math.inf,
                rigid_core_clash_score=math.inf,
                no_fly_min_distance_A=0.0,
                complement_overlap_score=0.0,
                cryptic_overlap_score=0.0,
                ligand_strain_penalty=0.0,
                grid_boundary_penalty=100.0,
                guard_level_passed=None,
                repairable_guard_passed=False,
                reason_failed_guard=error,
                selection_score=-math.inf,
                method=f"{reference_type}_o3a",
                conformer_id=conf_id,
            )
            attempts.append((copy_mol(probe), metrics, copy_mol(probe)))
            continue
        pre_min = min_pairwise_distance(ligand_heavy_coordinates(aligned), receptor_heavy_coordinates(target_topology))
        relaxed = copy_mol(aligned)
        try:
            minimize_ligand_in_shell(relaxed, target_topology, minimize_steps)
        except Exception:
            pass
        strain_penalty = max(0.0, forcefield_energy(relaxed) - baseline_energy) * 0.001
        metrics = evaluate_pose(
            mol=relaxed,
            reference_type=reference_type,
            reference_atom_count=reference_atom_count,
            o3a_score=o3a_score,
            o3a_rmsd=o3a_rmsd,
            target_topology=target_topology,
            bounds=bounds,
            field_model=field_model,
            ligand_strain_penalty=strain_penalty,
            method=f"{reference_type}_o3a",
            conformer_id=conf_id,
            pre_min_distance=pre_min,
        )
        attempts.append((aligned, metrics, relaxed))
    return attempts


def reconcile_pose_with_o3a_or_fallback(
    *,
    kabsch_ligand: Any,
    target_topology: JsonObject,
    bounds: dict[str, list[float] | list[int] | float],
    reference_ligand: Path | None,
    signal_grid: Path,
    voxel_thresholds: Path,
    boundaries: Path,
    pseudoligand: Path,
    pseudoligand_manifest: Path,
    compact_pseudoligand: Path,
    compact_pseudoligand_manifest: Path,
    output_o3a: Path,
    output_o3a_best: Path,
    output_o3a_relaxed: Path,
    failure_report: Path,
    minimize_steps: int,
    conformer_count: int,
) -> tuple[Any, str, float | None, str | None, Path | None, float, float, bool, float, JsonObject]:
    pseudo_manifest_path: Path | None = None
    field_model = load_field_model(
        signal_grid=signal_grid,
        voxel_thresholds=voxel_thresholds,
        boundaries=boundaries,
        bounds=bounds,
    )
    baseline_energy = forcefield_energy(kabsch_ligand)
    kabsch_metrics = evaluate_pose(
        mol=kabsch_ligand,
        reference_type="kabsch_fallback",
        reference_atom_count=0,
        o3a_score=None,
        o3a_rmsd=None,
        target_topology=target_topology,
        bounds=bounds,
        field_model=field_model,
        ligand_strain_penalty=0.0,
        method="kabsch_fallback",
        conformer_id=None,
    )

    attempts: list[tuple[Any, PoseMetrics, Any]] = []
    diffuse_reference = generate_prism_negative_image_pseudoligand(
        signal_grid=signal_grid,
        voxel_thresholds=voxel_thresholds,
        bounds=bounds,
        output=pseudoligand,
        manifest=pseudoligand_manifest,
        placed_ligand=kabsch_ligand,
    )
    pseudo_manifest_path = pseudoligand_manifest
    attempts.extend(
        attempt_reference_alignment(
            probes=[kabsch_ligand],
            reference=diffuse_reference,
            reference_type="diffuse_pseudo_ligand",
            target_topology=target_topology,
            bounds=bounds,
            field_model=field_model,
            minimize_steps=minimize_steps,
            baseline_energy=baseline_energy,
        )
    )

    compact_reference = generate_compact_negative_image_pseudoligand(
        field_model=field_model,
        placed_ligand=kabsch_ligand,
        output=compact_pseudoligand,
        manifest=compact_pseudoligand_manifest,
    )
    probes = [kabsch_ligand] + generate_probe_conformers(kabsch_ligand, conformer_count)
    compact_attempts = attempt_reference_alignment(
        probes=probes,
        reference=compact_reference,
        reference_type="compact_pseudo_ligand",
        target_topology=target_topology,
        bounds=bounds,
        field_model=field_model,
        minimize_steps=minimize_steps,
        baseline_energy=baseline_energy,
    )
    attempts.extend(compact_attempts)

    def accepted(metrics: PoseMetrics) -> bool:
        return (
            metrics.guard_level_passed is not None
            and metrics.grid_inside_status
            and metrics.no_fly_min_distance_A >= 4.0
            and metrics.complement_overlap_score > kabsch_metrics.complement_overlap_score
            and metrics.rigid_core_clash_score
            <= kabsch_metrics.rigid_core_clash_score + O3A_GUARD_CLASH_TOLERANCE
            and (metrics.min_receptor_heavy_distance_A >= 1.5 or metrics.repairable_guard_passed)
        )

    accepted_attempts = [attempt for attempt in attempts if accepted(attempt[1])]
    if accepted_attempts:
        _aligned, selected_metrics, relaxed = max(accepted_attempts, key=lambda item: item[1].selection_score)
        selected_method = selected_metrics.method
        o3a_score = selected_metrics.o3a_score
        o3a_error = None
        selected = relaxed
    else:
        selected_metrics = kabsch_metrics
        selected_method = "kabsch_fallback"
        o3a_score = None
        best_failed = max(attempts, key=lambda item: item[1].selection_score)[1] if attempts else None
        o3a_error = best_failed.reason_failed_guard if best_failed else "no O3A attempts were generated"
        selected = copy_mol(kabsch_ligand)
    write_o3a_failure_report(failure_report, [attempt[1] for attempt in attempts])
    write_plain_mol(
        selected,
        output_o3a,
        {
            "selected_alignment_method": selected_method,
            "o3a_score": "" if o3a_score is None else f"{o3a_score:.6f}",
            "o3a_error": o3a_error or "",
        },
    )
    write_plain_mol(
        selected,
        output_o3a_best,
        {
            "selected_alignment_method": selected_method,
            "o3a_score": "" if o3a_score is None else f"{o3a_score:.6f}",
            "o3a_error": o3a_error or "",
        },
    )
    relaxed = copy_mol(selected)
    relaxation_status = "already_relaxed_by_o3a_ladder" if selected_method != "kabsch_fallback" else "kabsch_fallback_relaxed"
    post_distance = selected_metrics.min_receptor_heavy_distance_A
    write_plain_mol(
        relaxed,
        output_o3a_relaxed,
        {
            "selected_alignment_method": selected_method,
            "torsion_relaxation_status": relaxation_status,
            "o3a_score": "" if o3a_score is None else f"{o3a_score:.6f}",
            "o3a_error": o3a_error or "",
        },
    )
    compact_best = max(compact_attempts, key=lambda item: item[1].selection_score)[1] if compact_attempts else None
    diffuse_best = (
        max([attempt for attempt in attempts if attempt[1].reference_type == "diffuse_pseudo_ligand"], key=lambda item: item[1].selection_score)[1]
        if any(attempt[1].reference_type == "diffuse_pseudo_ligand" for attempt in attempts)
        else None
    )
    comparison: JsonObject = {
        "kabsch_fallback": pose_metrics_to_json(kabsch_metrics),
        "diffuse_pseudo_ligand_o3a": pose_metrics_to_json(diffuse_best) if diffuse_best else None,
        "compact_pseudo_ligand_o3a": pose_metrics_to_json(compact_best) if compact_best else None,
        "selected_final_method": selected_method,
        "compact_pseudoligand_manifest_path": compact_pseudoligand_manifest.as_posix(),
        "o3a_failure_report_path": failure_report.as_posix(),
        "validation": {
            "selected_alignment_method_not_kabsch": selected_method != "kabsch_fallback",
            "ligand_inside_grid": selected_metrics.grid_inside_status,
            "no_fly_min_distance_A": selected_metrics.no_fly_min_distance_A,
            "post_min_heavy_distance_A": selected_metrics.min_receptor_heavy_distance_A,
            "repairable_guard_passed": selected_metrics.repairable_guard_passed,
            "complement_overlap_after_o3a": selected_metrics.complement_overlap_score,
            "complement_overlap_after_kabsch": kabsch_metrics.complement_overlap_score,
            "rigid_core_clash_after_o3a": selected_metrics.rigid_core_clash_score,
            "rigid_core_clash_after_kabsch": kabsch_metrics.rigid_core_clash_score,
            "guard_level_passed": selected_metrics.guard_level_passed,
        },
    }
    return (
        relaxed,
        selected_method,
        o3a_score,
        o3a_error,
        pseudo_manifest_path,
        kabsch_metrics.min_receptor_heavy_distance_A,
        post_distance,
        ligand_inside_grid(relaxed, bounds),
        selected_metrics.no_fly_min_distance_A,
        comparison,
    )


def write_ligand(mol: Any, output: Path, candidate: AlignmentCandidate, energy: float, final_min_distance: float) -> None:
    mol.SetProp("alignment_method", "kabsch_pocket_superposition_plus_constrained_uff")
    mol.SetProp("selected_source_condition", candidate.source_condition)
    mol.SetProp("selected_source_topology", candidate.source_topology.as_posix())
    mol.SetProp("target_condition", TARGET_CONDITION)
    mol.SetProp("kabsch_residue_ids", json.dumps(list(candidate.residue_ids)))
    mol.SetProp("kabsch_rotation_matrix", json.dumps(candidate.rotation.tolist()))
    mol.SetProp("kabsch_translation_vector", json.dumps(candidate.translation.tolist()))
    mol.SetProp("kabsch_pocket_rmsd_A", f"{candidate.pocket_rmsd_a:.6f}")
    mol.SetProp("pre_min_heavy_distance_A", f"{candidate.pre_min_distance_a:.6f}")
    mol.SetProp("post_min_heavy_distance_A", f"{final_min_distance:.6f}")
    mol.SetProp("constrained_uff_final_energy", f"{energy:.6f}")
    output.parent.mkdir(parents=True, exist_ok=True)
    writer = Chem.SDWriter(str(output))
    writer.write(mol)
    writer.close()


def write_kabsch_ligand(mol: Any, output: Path, candidate: AlignmentCandidate) -> None:
    mol.SetProp("alignment_method", "kabsch_pocket_superposition")
    mol.SetProp("selected_source_condition", candidate.source_condition)
    mol.SetProp("selected_source_topology", candidate.source_topology.as_posix())
    mol.SetProp("target_condition", TARGET_CONDITION)
    mol.SetProp("kabsch_residue_ids", json.dumps(list(candidate.residue_ids)))
    mol.SetProp("kabsch_rotation_matrix", json.dumps(candidate.rotation.tolist()))
    mol.SetProp("kabsch_translation_vector", json.dumps(candidate.translation.tolist()))
    mol.SetProp("kabsch_pocket_rmsd_A", f"{candidate.pocket_rmsd_a:.6f}")
    mol.SetProp("pre_min_heavy_distance_A", f"{candidate.pre_min_distance_a:.6f}")
    output.parent.mkdir(parents=True, exist_ok=True)
    writer = Chem.SDWriter(str(output))
    writer.write(mol)
    writer.close()


def write_manifest(
    *,
    manifest_path: Path,
    source_topology_path: Path,
    target_topology_path: Path,
    source_ligand_sdf: Path,
    output_kabsch_sdf: Path,
    output_minimized_sdf: Path,
    candidate: AlignmentCandidate,
    pre_centroid: tuple[float, float, float],
    post_min_centroid: tuple[float, float, float],
    post_min_heavy_distance_a: float,
    grid_bounds: dict[str, list[float] | list[int] | float],
    ligand_inside: bool,
) -> None:
    paths = {
        "source_topology": source_topology_path,
        "target_topology": target_topology_path,
        "source_ligand_sdf": source_ligand_sdf,
        "output_kabsch_sdf": output_kabsch_sdf,
        "output_minimized_sdf": output_minimized_sdf,
    }
    manifest = {
        "source_topology_path": source_topology_path.as_posix(),
        "target_topology_path": target_topology_path.as_posix(),
        "source_ligand_sdf": source_ligand_sdf.as_posix(),
        "output_kabsch_sdf": output_kabsch_sdf.as_posix(),
        "output_minimized_sdf": output_minimized_sdf.as_posix(),
        "selected_source_condition": candidate.source_condition,
        "matched_ca_count": len(candidate.residue_ids),
        "matched_residue_ids": list(candidate.residue_ids),
        "kabsch_rmsd_A": candidate.pocket_rmsd_a,
        "kabsch_rotation_matrix": candidate.rotation.tolist(),
        "kabsch_translation_vector": vector_to_list(candidate.translation),
        "pre_min_heavy_distance_A": candidate.pre_min_distance_a,
        "post_min_heavy_distance_A": post_min_heavy_distance_a,
        "ligand_centroid_before": list(pre_centroid),
        "ligand_centroid_after_kabsch": list(candidate.transformed_centroid),
        "ligand_centroid_after_min": list(post_min_centroid),
        "grid_bounds_6XOX": grid_bounds,
        "ligand_inside_grid_after_alignment": ligand_inside,
        "sha256": {name: sha256_file(path) for name, path in paths.items()},
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")


def write_pose_reconciliation_manifest(
    *,
    manifest_path: Path,
    selected_alignment_method: str,
    kabsch_used_only_as_initializer: bool,
    o3a_score: float | None,
    o3a_error: str | None,
    pseudo_manifest_path: Path | None,
    source_ligand_sdf: Path,
    output_o3a_sdf: Path,
    output_o3a_relaxed_sdf: Path,
    ligand_centroid_before: tuple[float, float, float],
    ligand_centroid_after: tuple[float, float, float],
    ligand_inside: bool,
    rigid_core_clash_before: float,
    rigid_core_clash_after: float,
    no_fly_min_distance_before: float,
    no_fly_min_distance_after: float,
    torsion_relaxation_status: str,
    comparison: JsonObject,
) -> None:
    paths: dict[str, Path] = {
        "source_ligand_sdf": source_ligand_sdf,
        "output_o3a_sdf": output_o3a_sdf,
        "output_o3a_relaxed_sdf": output_o3a_relaxed_sdf,
    }
    if pseudo_manifest_path is not None:
        paths["pseudo_ligand_manifest"] = pseudo_manifest_path
    manifest = {
        "selected_alignment_method": selected_alignment_method,
        "kabsch_used_only_as_initializer": kabsch_used_only_as_initializer,
        "o3a_score": o3a_score,
        "o3a_error": o3a_error,
        "pseudo_ligand_manifest_path": pseudo_manifest_path.as_posix() if pseudo_manifest_path else None,
        "ligand_centroid_before": list(ligand_centroid_before),
        "ligand_centroid_after": list(ligand_centroid_after),
        "ligand_inside_grid": ligand_inside,
        "rigid_core_clash_before": rigid_core_clash_before,
        "rigid_core_clash_after": rigid_core_clash_after,
        "complement_overlap_before": None,
        "complement_overlap_after": None,
        "no_fly_min_distance_before": no_fly_min_distance_before,
        "no_fly_min_distance_after": no_fly_min_distance_after,
        "torsion_relaxation_status": torsion_relaxation_status,
        "pose_comparison": comparison,
        "sha256": {name: sha256_file(path) for name, path in paths.items() if path.exists()},
    }
    if selected_alignment_method == "kabsch_fallback" and o3a_score is not None:
        raise ValueError("kabsch_fallback cannot report O3A success")
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")


def main() -> int:
    args = parse_args()
    input_path = Path(args.input)
    output_kabsch_path = Path(args.output_kabsch)
    output_path = Path(args.output)
    output_o3a_path = Path(args.output_o3a)
    output_o3a_best_path = Path(args.output_o3a_best)
    output_o3a_relaxed_path = Path(args.output_o3a_relaxed)
    manifest_path = Path(args.manifest)
    pose_manifest_path = Path(args.pose_manifest)
    source_topology_path = Path(args.source_topology)
    target_topology_path = Path(args.target_topology)
    topology_dir = Path(args.topology_dir)
    grid_mapping = Path(args.grid_mapping)
    risk_map = Path(args.risk_map)
    signal_grid = Path(args.signal_grid)
    voxel_thresholds = Path(args.voxel_thresholds)
    reference_ligand = cast(Path | None, args.reference_ligand)
    pseudoligand = Path(args.pseudoligand)
    pseudoligand_manifest = Path(args.pseudoligand_manifest)
    compact_pseudoligand = Path(args.compact_pseudoligand)
    compact_pseudoligand_manifest = Path(args.compact_pseudoligand_manifest)
    failure_report = Path(args.o3a_failure_report)
    boundaries = Path(args.boundaries)
    for path in (input_path, source_topology_path, target_topology_path, grid_mapping, risk_map, signal_grid, boundaries):
        if not path.exists():
            raise FileNotFoundError(path)
    mol = load_single_mol(input_path)
    centroid_before = ligand_centroid(mol)
    requested_sources = cast(list[str] | None, args.source_condition)
    if requested_sources is None:
        candidates = [
            build_candidate(
                source_condition=DEFAULT_SOURCE_CONDITION,
                source_topology_path=source_topology_path,
                target_topology_path=target_topology_path,
                risk_map=risk_map,
                mol=mol,
                pocket_limit=int(args.pocket_residues),
            )
        ]
    else:
        candidates = [
            build_candidate(
            source_condition=source_condition,
            source_topology_path=topology_dir / f"{source_condition}.topology.json",
            target_topology_path=target_topology_path,
            risk_map=risk_map,
            mol=mol,
            pocket_limit=int(args.pocket_residues),
        )
        for source_condition in requested_sources
        if (topology_dir / f"{source_condition}.topology.json").exists()
        ]
    if not candidates:
        raise ValueError("no source topology candidates were available")
    candidate = select_candidate(candidates)
    if len(candidate.residue_ids) < MIN_MATCHED_CA:
        raise ValueError(f"matched C-alpha count {len(candidate.residue_ids)} is below {MIN_MATCHED_CA}")
    apply_transform(mol, candidate.rotation, candidate.translation)
    bounds = grid_bounds_6xox(grid_mapping)
    inside_after_alignment = ligand_inside_grid(mol, bounds)
    if not inside_after_alignment:
        raise ValueError("ligand is outside 6XOX grid after Kabsch alignment")
    write_kabsch_ligand(mol, output_kabsch_path, candidate)
    target_topology = read_json_object(target_topology_path)
    energy = minimize_ligand_in_shell(mol, target_topology, int(args.minimize_steps))
    final_min_distance = min_pairwise_distance(ligand_heavy_coordinates(mol), receptor_heavy_coordinates(target_topology))
    if final_min_distance < 1.0:
        raise ValueError(f"post-minimization heavy-atom distance {final_min_distance:.3f} A is below 1.0 A")
    write_ligand(mol, output_path, candidate, energy, final_min_distance)
    (
        reconciled_mol,
        selected_alignment_method,
        o3a_score,
        o3a_error,
        pseudo_manifest_path,
        o3a_pre_distance,
        o3a_post_distance,
        o3a_inside_grid,
        no_fly_min_after,
        pose_comparison,
    ) = reconcile_pose_with_o3a_or_fallback(
        kabsch_ligand=mol,
        target_topology=target_topology,
        bounds=bounds,
        reference_ligand=reference_ligand,
        signal_grid=signal_grid,
        voxel_thresholds=voxel_thresholds,
        boundaries=boundaries,
        pseudoligand=pseudoligand,
        pseudoligand_manifest=pseudoligand_manifest,
        compact_pseudoligand=compact_pseudoligand,
        compact_pseudoligand_manifest=compact_pseudoligand_manifest,
        output_o3a=output_o3a_path,
        output_o3a_best=output_o3a_best_path,
        output_o3a_relaxed=output_o3a_relaxed_path,
        failure_report=failure_report,
        minimize_steps=int(args.minimize_steps),
        conformer_count=int(args.o3a_conformers),
    )
    if not o3a_inside_grid:
        raise ValueError("final reconciled pose is outside 6XOX grid")
    write_manifest(
        manifest_path=manifest_path,
        source_topology_path=candidate.source_topology,
        target_topology_path=target_topology_path,
        source_ligand_sdf=input_path,
        output_kabsch_sdf=output_kabsch_path,
        output_minimized_sdf=output_path,
        candidate=candidate,
        pre_centroid=centroid_before,
        post_min_centroid=ligand_centroid(mol),
        post_min_heavy_distance_a=final_min_distance,
        grid_bounds=bounds,
        ligand_inside=inside_after_alignment,
    )
    write_pose_reconciliation_manifest(
        manifest_path=pose_manifest_path,
        selected_alignment_method=selected_alignment_method,
        kabsch_used_only_as_initializer=selected_alignment_method != "kabsch_fallback",
        o3a_score=o3a_score,
        o3a_error=o3a_error,
        pseudo_manifest_path=pseudo_manifest_path,
        source_ligand_sdf=input_path,
        output_o3a_sdf=output_o3a_path,
        output_o3a_relaxed_sdf=output_o3a_relaxed_path,
        ligand_centroid_before=centroid_before,
        ligand_centroid_after=ligand_centroid(reconciled_mol),
        ligand_inside=o3a_inside_grid,
        rigid_core_clash_before=o3a_pre_distance,
        rigid_core_clash_after=o3a_post_distance,
        no_fly_min_distance_before=float(
            cast(JsonObject, pose_comparison["kabsch_fallback"])["no_fly_min_distance_A"]
        ),
        no_fly_min_distance_after=no_fly_min_after,
        torsion_relaxation_status=reconciled_mol.GetProp("torsion_relaxation_status")
        if reconciled_mol.HasProp("torsion_relaxation_status")
        else "constrained_relaxation_complete",
        comparison=pose_comparison,
    )
    emit(
        f"wrote {output_kabsch_path}, {output_path}, and {output_o3a_relaxed_path} "
        f"manifest={manifest_path} pose_manifest={pose_manifest_path} "
        f"matched_ca_count={len(candidate.residue_ids)} selected_source={candidate.source_condition} "
        f"selected_alignment_method={selected_alignment_method} "
        f"pocket_rmsd_A={candidate.pocket_rmsd_a:.3f} "
        f"pre_min_heavy_distance_A={candidate.pre_min_distance_a:.3f} "
        f"post_min_heavy_distance_A={final_min_distance:.3f} "
        f"o3a_score={o3a_score if o3a_score is not None else 'NA'}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
