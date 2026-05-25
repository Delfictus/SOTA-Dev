"""Typed Track A action-space and reward primitives for GFlowNet training."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from enum import Enum
from importlib import import_module
from pathlib import Path
from typing import Any, Mapping, Sequence, cast

import polars as pl

from prism_dstw.ontology import EpistemicClass


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_VSPACE_SURVIVORS = (
    REPO_ROOT / "campaigns/glp1r_aleniglipron/track_a_generative/vspace_survivors.parquet"
)
Coordinate3D = tuple[float, float, float]
Matrix3D = tuple[Coordinate3D, Coordinate3D, Coordinate3D]


class BoundaryClass(str, Enum):
    REWARD_TSO = "reward_tso"
    PENALTY_TSO = "penalty_tso"
    BRIDGE_ANCHOR_NO_FLY_ZONE = "bridge_anchor_no_fly_zone"


@dataclass(frozen=True)
class RewardWeights:
    complement: float = 1.0
    clash: float = 1.0
    bias_lock: float = 1.0
    shear: float = 1.0
    gamma: float = 2.0
    uncertainty_lambda: float = 1.0
    sa_lambda: float = 1.0
    oral_lambda: float = 1.0


@dataclass(frozen=True)
class OralBioavailabilityDescriptor:
    tpsa_A2: float
    rotatable_bonds: int
    hydrogen_bond_donors: int
    violation_score: float


@dataclass(frozen=True)
class ChemicalAnchor:
    anchor_id: str
    canonical_smiles: str
    steric_volume_a3: float
    formal_charge: int
    epistemic_class: EpistemicClass = EpistemicClass.HYPOTHESIZED


@dataclass(frozen=True)
class AlignmentPoint:
    residue_idx: int
    residue_name: str
    ca_xyz: Coordinate3D


@dataclass(frozen=True)
class DynamicAlignmentReference:
    condition_id: str
    md_step: int
    pocket_points: tuple[AlignmentPoint, ...]


@dataclass(frozen=True)
class ScaffoldAlignedFragmentPose:
    scaffold_exit_atom_idx: int
    fragment_attachment_atom_idx: int
    fragment_dummy_atom_idx: int | None
    exit_vector: Coordinate3D
    attachment_bond_length_A: float
    conformer_atoms_json: str
    aligned_mol: Any


@dataclass(frozen=True)
class GridGeometry:
    condition_id: str
    grid_dim: int
    origin_xyz: Coordinate3D
    spacing_A: float


@dataclass(frozen=True)
class FragmentVoxelScore:
    mapped_atom_count: int
    pi_complement: float
    pi_clash: float
    sigma_shear: float
    reward: float
    oral_violation_score: float = 0.0
    oral_penalty: float = 1.0


@dataclass(frozen=True)
class ExitVector:
    vector_id: str
    atom_idx: int
    dx: float
    dy: float
    dz: float
    u_pose: float

    def unit(self) -> "ExitVector":
        norm = math.sqrt(self.dx * self.dx + self.dy * self.dy + self.dz * self.dz)
        if norm <= 0.0:
            raise ValueError(f"exit vector {self.vector_id} has zero length")
        return ExitVector(
            vector_id=self.vector_id,
            atom_idx=self.atom_idx,
            dx=self.dx / norm,
            dy=self.dy / norm,
            dz=self.dz / norm,
            u_pose=self.u_pose,
        )


@dataclass(frozen=True)
class ExitVectorPhaseContext:
    """Phase-resolved local field context used for steric action masking."""

    stable_occupied: bool = False
    thermally_destabilized: bool = False
    thermally_activated: bool = False
    cold_normalized: float = 0.0
    out_of_grid: bool = False


@dataclass(frozen=True)
class BoundaryVoxel:
    condition_id: str
    voxel_idx: int
    boundary_class: BoundaryClass
    pi_complement: float
    pi_clash: float
    sigma_shear: float


@dataclass(frozen=True)
class GFlowNetState:
    scaffold_id: str
    canonical_smiles: str
    occupied_voxels: tuple[int, ...]
    epistemic_class: EpistemicClass = EpistemicClass.HYPOTHESIZED


@dataclass(frozen=True)
class GrowthAction:
    action_id: str
    anchor: ChemicalAnchor
    exit_vector: ExitVector
    target_condition_id: str
    target_voxel_idx: int


@dataclass(frozen=True)
class ScoredAction:
    action: GrowthAction
    reward: float
    pi_complement: float
    pi_clash: float
    sigma_shear: float
    attenuation: float
    oral_violation_score: float = 0.0
    oral_penalty: float = 1.0


@dataclass(frozen=True)
class VSpaceSurvivor:
    product_id: str
    canonical_smiles: str
    synthon_a_id: str
    synthon_b_id: str
    score: float
    pi_complement: float
    pi_clash: float
    coordinates_json: str
    epistemic_class: EpistemicClass = EpistemicClass.HYPOTHESIZED


@dataclass(frozen=True)
class VSpaceGrowthAction:
    action_id: str
    survivor: VSpaceSurvivor
    exit_vector: ExitVector | None = None


@dataclass(frozen=True)
class ScoredVSpaceAction:
    action: VSpaceGrowthAction
    reward: float
    pi_complement: float
    pi_clash: float
    attenuation: float
    oral_violation_score: float = 0.0
    oral_penalty: float = 1.0


def _json_object(value: object, label: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object")
    return cast(dict[str, object], value)


def _json_list(value: object, label: str) -> list[object]:
    if not isinstance(value, list):
        raise ValueError(f"{label} must be a JSON array")
    return value


def _as_float(value: object, label: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be numeric, got bool")
    if isinstance(value, int | float | str):
        return float(value)
    raise ValueError(f"{label} must be numeric")


def _as_int(value: object, label: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be an integer, got bool")
    if isinstance(value, int | float | str):
        return int(value)
    raise ValueError(f"{label} must be an integer")


def _as_str(value: object, label: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{label} must be a string")
    return value


def load_dynamic_alignment_reference(path: Path) -> DynamicAlignmentReference:
    payload = _json_object(json.loads(path.read_text(encoding="utf-8")), "dynamic_alignment_reference")
    raw_points = _json_list(payload.get("pocket_residue_coordinates"), "pocket_residue_coordinates")
    points: list[AlignmentPoint] = []
    for raw_point in raw_points:
        point = _json_object(raw_point, "pocket_residue_coordinate")
        raw_xyz = _json_list(point.get("ca_xyz"), "ca_xyz")
        if len(raw_xyz) != 3:
            raise ValueError("ca_xyz must contain exactly three coordinates")
        points.append(
            AlignmentPoint(
                residue_idx=_as_int(point.get("residue_idx"), "residue_idx"),
                residue_name=_as_str(point.get("residue_name", "UNK"), "residue_name"),
                ca_xyz=(
                    _as_float(raw_xyz[0], "x"),
                    _as_float(raw_xyz[1], "y"),
                    _as_float(raw_xyz[2], "z"),
                ),
            )
        )
    if len(points) < 3:
        raise ValueError("dynamic alignment reference requires at least three C-alpha points")
    return DynamicAlignmentReference(
        condition_id=_as_str(payload.get("condition_id"), "condition_id"),
        md_step=_as_int(payload.get("md_step"), "md_step"),
        pocket_points=tuple(points),
    )


def _rdkit_modules() -> tuple[Any, Any]:
    chem = cast(Any, import_module("rdkit.Chem"))
    all_chem = cast(Any, import_module("rdkit.Chem.AllChem"))
    return chem, all_chem


def _rdkit_descriptor_modules() -> tuple[Any, Any]:
    rd_mol_descriptors = cast(Any, import_module("rdkit.Chem.rdMolDescriptors"))
    lipinski = cast(Any, import_module("rdkit.Chem.Lipinski"))
    return rd_mol_descriptors, lipinski


def _mol_from_reward_smiles(smiles: str) -> Any | None:
    chem, _ = _rdkit_modules()
    mol = chem.MolFromSmiles(smiles)
    if mol is not None:
        return mol
    left_hand_side = smiles.split(">>", maxsplit=1)[0]
    candidates: list[Any] = []
    for component in left_hand_side.split("."):
        parsed = chem.MolFromSmiles(component)
        if parsed is not None:
            candidates.append(parsed)
    if not candidates:
        return None
    return max(candidates, key=lambda item: int(item.GetNumHeavyAtoms()))


def oral_bioavailability_descriptors(mol: Any) -> OralBioavailabilityDescriptor:
    """Compute oral small-molecule descriptor gates for Track A rewards."""

    rd_mol_descriptors, lipinski = _rdkit_descriptor_modules()
    tpsa = float(rd_mol_descriptors.CalcTPSA(mol))
    rotatable_bonds = int(lipinski.NumRotatableBonds(mol))
    hydrogen_bond_donors = int(lipinski.NumHDonors(mol))
    tpsa_excess = max(0.0, (tpsa - 140.0) / 20.0)
    rotatable_excess = max(0.0, (float(rotatable_bonds) - 12.0) / 2.0)
    donor_excess = max(0.0, float(hydrogen_bond_donors) - 5.0)
    return OralBioavailabilityDescriptor(
        tpsa_A2=tpsa,
        rotatable_bonds=rotatable_bonds,
        hydrogen_bond_donors=hydrogen_bond_donors,
        violation_score=tpsa_excess + rotatable_excess + donor_excess,
    )


def oral_violation_score_for_smiles(smiles: str) -> float:
    mol = _mol_from_reward_smiles(smiles)
    if mol is None:
        return 0.0
    return oral_bioavailability_descriptors(mol).violation_score


def synthetic_accessibility_score_for_mol(mol: Any) -> float:
    try:
        sascorer = cast(Any, import_module("rdkit.Contrib.SA_Score.sascorer"))
        return float(sascorer.calculateScore(mol))
    except Exception:
        heavy_atoms = sum(1 for atom in mol.GetAtoms() if int(atom.GetAtomicNum()) > 1)
        rings = int(mol.GetRingInfo().NumRings())
        return max(1.0, min(10.0, 1.0 + 0.12 * float(heavy_atoms) + 0.35 * float(rings)))


def synthetic_accessibility_score_for_smiles(smiles: str) -> float:
    mol = _mol_from_reward_smiles(smiles)
    if mol is None:
        return 0.0
    return synthetic_accessibility_score_for_mol(mol)


def _embed_anchor_molecule(smiles: str) -> Any:
    chem, all_chem = _rdkit_modules()
    mol = chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError("anchor canonical_smiles could not be parsed by RDKit")
    mol_h = chem.AddHs(mol)
    params = all_chem.ETKDGv3()
    params.randomSeed = 91_721
    params.useRandomCoords = True
    status = int(all_chem.EmbedMolecule(mol_h, params))
    if status != 0:
        raise ValueError(f"RDKit ETKDGv3 embedding failed with code {status}")
    if bool(all_chem.MMFFHasAllMoleculeParams(mol_h)):
        all_chem.MMFFOptimizeMolecule(mol_h, mmffVariant="MMFF94s", maxIters=200)
    else:
        all_chem.UFFOptimizeMolecule(mol_h, maxIters=200)
    return mol_h


def _heavy_atom_indices(mol: Any) -> list[int]:
    return [int(atom.GetIdx()) for atom in mol.GetAtoms() if int(atom.GetAtomicNum()) > 1]


def _conformer_atoms_json(mol: Any) -> str:
    conformer = mol.GetConformer()
    rows: list[dict[str, object]] = []
    for atom in mol.GetAtoms():
        if int(atom.GetAtomicNum()) <= 1:
            continue
        atom_idx = int(atom.GetIdx())
        pos = conformer.GetAtomPosition(atom_idx)
        rows.append(
            {
                "atom_idx": atom_idx,
                "atomic_num": int(atom.GetAtomicNum()),
                "symbol": str(atom.GetSymbol()),
                "x": float(pos.x),
                "y": float(pos.y),
                "z": float(pos.z),
            }
        )
    return json.dumps(rows, sort_keys=True, separators=(",", ":"))


def _point3d(mol: Any, atom_idx: int) -> Coordinate3D:
    pos = mol.GetConformer().GetAtomPosition(atom_idx)
    return (float(pos.x), float(pos.y), float(pos.z))


def _set_point3d(mol: Any, atom_idx: int, xyz: Coordinate3D) -> None:
    geometry = cast(Any, import_module("rdkit.Geometry"))
    x_coord, y_coord, z_coord = xyz
    mol.GetConformer().SetAtomPosition(atom_idx, geometry.Point3D(x_coord, y_coord, z_coord))


def _vec_add(lhs: Coordinate3D, rhs: Coordinate3D) -> Coordinate3D:
    return (lhs[0] + rhs[0], lhs[1] + rhs[1], lhs[2] + rhs[2])


def _vec_sub(lhs: Coordinate3D, rhs: Coordinate3D) -> Coordinate3D:
    return (lhs[0] - rhs[0], lhs[1] - rhs[1], lhs[2] - rhs[2])


def _vec_scale(value: Coordinate3D, scalar: float) -> Coordinate3D:
    return (value[0] * scalar, value[1] * scalar, value[2] * scalar)


def _dot(lhs: Coordinate3D, rhs: Coordinate3D) -> float:
    return lhs[0] * rhs[0] + lhs[1] * rhs[1] + lhs[2] * rhs[2]


def _cross(lhs: Coordinate3D, rhs: Coordinate3D) -> Coordinate3D:
    return (
        lhs[1] * rhs[2] - lhs[2] * rhs[1],
        lhs[2] * rhs[0] - lhs[0] * rhs[2],
        lhs[0] * rhs[1] - lhs[1] * rhs[0],
    )


def _norm(value: Coordinate3D) -> float:
    return math.sqrt(_dot(value, value))


def _unit(value: Coordinate3D, label: str) -> Coordinate3D:
    length = _norm(value)
    if length <= 1.0e-12:
        raise ValueError(f"{label} must be non-zero")
    return _vec_scale(value, 1.0 / length)


def _centroid(points: Sequence[Coordinate3D]) -> Coordinate3D:
    if not points:
        raise ValueError("cannot compute centroid of an empty point set")
    scale = 1.0 / float(len(points))
    return (
        sum(point[0] for point in points) * scale,
        sum(point[1] for point in points) * scale,
        sum(point[2] for point in points) * scale,
    )


def _identity_matrix() -> Matrix3D:
    return ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))


def _axis_angle_matrix(axis: Coordinate3D, angle_rad: float) -> Matrix3D:
    axis_x, axis_y, axis_z = _unit(axis, "rotation axis")
    cos_theta = math.cos(angle_rad)
    sin_theta = math.sin(angle_rad)
    one_minus_cos = 1.0 - cos_theta
    return (
        (
            cos_theta + axis_x * axis_x * one_minus_cos,
            axis_x * axis_y * one_minus_cos - axis_z * sin_theta,
            axis_x * axis_z * one_minus_cos + axis_y * sin_theta,
        ),
        (
            axis_y * axis_x * one_minus_cos + axis_z * sin_theta,
            cos_theta + axis_y * axis_y * one_minus_cos,
            axis_y * axis_z * one_minus_cos - axis_x * sin_theta,
        ),
        (
            axis_z * axis_x * one_minus_cos - axis_y * sin_theta,
            axis_z * axis_y * one_minus_cos + axis_x * sin_theta,
            cos_theta + axis_z * axis_z * one_minus_cos,
        ),
    )


def _orthogonal_axis(vector: Coordinate3D) -> Coordinate3D:
    basis: Coordinate3D = (1.0, 0.0, 0.0) if abs(vector[0]) < 0.9 else (0.0, 1.0, 0.0)
    return _unit(_cross(vector, basis), "orthogonal rotation axis")


def _rotation_matrix_from_vectors(source: Coordinate3D, target: Coordinate3D) -> Matrix3D:
    source_unit = _unit(source, "source attachment vector")
    target_unit = _unit(target, "target scaffold exit vector")
    dot_value = max(-1.0, min(1.0, _dot(source_unit, target_unit)))
    if dot_value > 1.0 - 1.0e-10:
        return _identity_matrix()
    if dot_value < -1.0 + 1.0e-10:
        return _axis_angle_matrix(_orthogonal_axis(source_unit), math.pi)
    return _axis_angle_matrix(_cross(source_unit, target_unit), math.acos(dot_value))


def _mat_vec_mul(matrix: Matrix3D, vector: Coordinate3D) -> Coordinate3D:
    return (
        _dot(matrix[0], vector),
        _dot(matrix[1], vector),
        _dot(matrix[2], vector),
    )


def _copy_mol_with_conformer(mol: Any) -> Any:
    chem, all_chem = _rdkit_modules()
    copied = chem.Mol(mol)
    if int(copied.GetNumConformers()) > 0:
        return copied
    params = all_chem.ETKDGv3()
    params.randomSeed = 91_723
    params.useRandomCoords = True
    status = int(all_chem.EmbedMolecule(copied, params))
    if status != 0:
        raise ValueError(f"RDKit ETKDGv3 embedding failed with code {status}")
    if _mol_has_dummy_atom(copied):
        return copied
    if bool(all_chem.MMFFHasAllMoleculeParams(copied)):
        all_chem.MMFFOptimizeMolecule(copied, mmffVariant="MMFF94s", maxIters=200)
    else:
        all_chem.UFFOptimizeMolecule(copied, maxIters=200)
    return copied


def _mol_has_dummy_atom(mol: Any) -> bool:
    return any(int(atom.GetAtomicNum()) == 0 for atom in mol.GetAtoms())


def _remove_dummy_atoms(mol: Any) -> tuple[Any, dict[int, int]]:
    chem, _ = _rdkit_modules()
    dummy_indices = [int(atom.GetIdx()) for atom in mol.GetAtoms() if int(atom.GetAtomicNum()) == 0]
    if not dummy_indices:
        return (mol, {int(atom.GetIdx()): int(atom.GetIdx()) for atom in mol.GetAtoms()})

    removed = set(dummy_indices)
    index_map: dict[int, int] = {}
    next_idx = 0
    for atom in mol.GetAtoms():
        atom_idx = int(atom.GetIdx())
        if atom_idx in removed:
            continue
        index_map[atom_idx] = next_idx
        next_idx += 1

    editable = chem.RWMol(mol)
    for atom_idx in sorted(dummy_indices, reverse=True):
        editable.RemoveAtom(atom_idx)
    return (editable.GetMol(), index_map)


def _atom_neighbor_indices(mol: Any, atom_idx: int, *, include_dummy: bool) -> list[int]:
    atom = mol.GetAtomWithIdx(atom_idx)
    indices: list[int] = []
    for neighbor in atom.GetNeighbors():
        atomic_num = int(neighbor.GetAtomicNum())
        if atomic_num > 1 or (include_dummy and atomic_num == 0):
            indices.append(int(neighbor.GetIdx()))
    return indices


def _scaffold_exit_direction(
    scaffold_mol: Any,
    exit_atom_idx: int,
    explicit_exit_vector: Coordinate3D | None,
) -> Coordinate3D:
    if explicit_exit_vector is not None:
        return _unit(explicit_exit_vector, "explicit scaffold exit vector")
    exit_position = _point3d(scaffold_mol, exit_atom_idx)
    dummy_neighbors = [
        int(neighbor.GetIdx())
        for neighbor in scaffold_mol.GetAtomWithIdx(exit_atom_idx).GetNeighbors()
        if int(neighbor.GetAtomicNum()) == 0
    ]
    if dummy_neighbors:
        return _unit(_vec_sub(_point3d(scaffold_mol, dummy_neighbors[0]), exit_position), "scaffold dummy exit vector")
    neighbor_points = [_point3d(scaffold_mol, idx) for idx in _atom_neighbor_indices(scaffold_mol, exit_atom_idx, include_dummy=False)]
    if not neighbor_points:
        raise ValueError("scaffold exit atom must have a dummy neighbor, heavy neighbor, or explicit exit vector")
    return _unit(_vec_sub(exit_position, _centroid(neighbor_points)), "scaffold local outward vector")


def _fragment_dummy_atom_idx(fragment_mol: Any, requested_dummy_atom_idx: int | None) -> int | None:
    if requested_dummy_atom_idx is not None:
        atom = fragment_mol.GetAtomWithIdx(int(requested_dummy_atom_idx))
        if int(atom.GetAtomicNum()) != 0:
            raise ValueError("requested fragment_dummy_atom_idx does not point to a dummy atom")
        return int(requested_dummy_atom_idx)
    dummies = [int(atom.GetIdx()) for atom in fragment_mol.GetAtoms() if int(atom.GetAtomicNum()) == 0]
    if len(dummies) == 1:
        return dummies[0]
    if len(dummies) > 1:
        raise ValueError("fragment contains multiple dummy atoms; provide fragment_dummy_atom_idx")
    return None


def _fragment_attachment_geometry(
    fragment_mol: Any,
    *,
    fragment_attachment_atom_idx: int | None,
    fragment_dummy_atom_idx: int | None,
    fallback_source_length_A: float,
) -> tuple[int | None, int, Coordinate3D, Coordinate3D, bool]:
    dummy_idx = _fragment_dummy_atom_idx(fragment_mol, fragment_dummy_atom_idx)
    if dummy_idx is not None:
        neighbors = _atom_neighbor_indices(fragment_mol, dummy_idx, include_dummy=False)
        if len(neighbors) != 1:
            raise ValueError("fragment dummy atom must have exactly one heavy attachment neighbor")
        attachment_idx = neighbors[0]
        dummy_position = _point3d(fragment_mol, dummy_idx)
        attachment_position = _point3d(fragment_mol, attachment_idx)
        source_vector = _vec_sub(attachment_position, dummy_position)
        return (dummy_idx, attachment_idx, attachment_position, source_vector, True)
    if fragment_attachment_atom_idx is None:
        raise ValueError("fragment must contain one dummy atom or provide fragment_attachment_atom_idx")
    attachment_idx = int(fragment_attachment_atom_idx)
    attachment_position = _point3d(fragment_mol, attachment_idx)
    neighbor_points = [_point3d(fragment_mol, idx) for idx in _atom_neighbor_indices(fragment_mol, attachment_idx, include_dummy=False)]
    if not neighbor_points:
        raise ValueError("fragment attachment atom must have a heavy neighbor when no dummy atom is present")
    source_vector = _vec_sub(attachment_position, _centroid(neighbor_points))
    if _norm(source_vector) <= 1.0e-12:
        source_vector = (fallback_source_length_A, 0.0, 0.0)
    return (None, attachment_idx, attachment_position, source_vector, False)


def align_fragment_to_scaffold_exit_vector(
    fragment_mol: Any,
    scaffold_mol: Any,
    exit_atom_idx: int,
    *,
    fragment_attachment_atom_idx: int | None = None,
    fragment_dummy_atom_idx: int | None = None,
    scaffold_exit_vector: Coordinate3D | None = None,
    attachment_bond_length_A: float = 1.50,
) -> ScaffoldAlignedFragmentPose:
    """Rigidly attach a fragment to a parent scaffold exit vector using 3D vector math."""

    fragment = _copy_mol_with_conformer(fragment_mol)
    scaffold = _copy_mol_with_conformer(scaffold_mol)
    exit_position = _point3d(scaffold, int(exit_atom_idx))
    exit_direction = _scaffold_exit_direction(scaffold, int(exit_atom_idx), scaffold_exit_vector)
    dummy_idx, attachment_idx, attachment_origin, fragment_vector, uses_dummy = _fragment_attachment_geometry(
        fragment,
        fragment_attachment_atom_idx=fragment_attachment_atom_idx,
        fragment_dummy_atom_idx=fragment_dummy_atom_idx,
        fallback_source_length_A=attachment_bond_length_A,
    )
    target_vector = _vec_scale(exit_direction, -1.0) if uses_dummy else exit_direction
    rotation = _rotation_matrix_from_vectors(fragment_vector, target_vector)
    target_attachment = _vec_add(exit_position, _vec_scale(target_vector, attachment_bond_length_A))
    for atom in fragment.GetAtoms():
        atom_idx = int(atom.GetIdx())
        relative = _vec_sub(_point3d(fragment, atom_idx), attachment_origin)
        rotated = _mat_vec_mul(rotation, relative)
        _set_point3d(fragment, atom_idx, _vec_add(target_attachment, rotated))

    aligned_fragment, index_map = _remove_dummy_atoms(fragment)
    if attachment_idx not in index_map:
        raise ValueError("fragment attachment atom was removed while purging dummy atoms")
    clean_attachment_idx = index_map[attachment_idx]
    return ScaffoldAlignedFragmentPose(
        scaffold_exit_atom_idx=int(exit_atom_idx),
        fragment_attachment_atom_idx=clean_attachment_idx,
        fragment_dummy_atom_idx=None,
        exit_vector=exit_direction,
        attachment_bond_length_A=attachment_bond_length_A,
        conformer_atoms_json=_conformer_atoms_json(aligned_fragment),
        aligned_mol=aligned_fragment,
    )


def voxel_idx_for_xyz(xyz: Coordinate3D, geometry: GridGeometry) -> int | None:
    x_idx = math.floor((xyz[0] - geometry.origin_xyz[0]) / geometry.spacing_A)
    y_idx = math.floor((xyz[1] - geometry.origin_xyz[1]) / geometry.spacing_A)
    z_idx = math.floor((xyz[2] - geometry.origin_xyz[2]) / geometry.spacing_A)
    if (
        x_idx < 0
        or y_idx < 0
        or z_idx < 0
        or x_idx >= geometry.grid_dim
        or y_idx >= geometry.grid_dim
        or z_idx >= geometry.grid_dim
    ):
        return None
    return z_idx * geometry.grid_dim * geometry.grid_dim + y_idx * geometry.grid_dim + x_idx


def score_aligned_fragment_against_tso(
    pose: ScaffoldAlignedFragmentPose,
    *,
    condition_id: str,
    geometry: GridGeometry,
    boundaries: Mapping[tuple[str, int], BoundaryVoxel],
    weights: RewardWeights,
    u_pose: float,
) -> FragmentVoxelScore:
    atom_rows = _json_list(json.loads(pose.conformer_atoms_json), "conformer_atoms_json")
    pi_complement = 0.0
    pi_clash = 0.0
    sigma_shear = 0.0
    mapped_atom_count = 0
    for raw_atom in atom_rows:
        atom = _json_object(raw_atom, "conformer_atom")
        xyz = (_as_float(atom.get("x"), "x"), _as_float(atom.get("y"), "y"), _as_float(atom.get("z"), "z"))
        voxel_idx = voxel_idx_for_xyz(xyz, geometry)
        if voxel_idx is None:
            continue
        boundary = boundaries.get((condition_id, voxel_idx))
        if boundary is None:
            continue
        mapped_atom_count += 1
        pi_complement += boundary.pi_complement
        pi_clash += boundary.pi_clash
        sigma_shear += boundary.sigma_shear
    reward = saturated_reward(
        pi_complement=pi_complement,
        pi_clash=pi_clash,
        sigma_shear=sigma_shear,
        u_pose=u_pose,
        sa_score=synthetic_accessibility_score_for_mol(pose.aligned_mol),
        oral_violation_score=oral_bioavailability_descriptors(pose.aligned_mol).violation_score,
        weights=weights,
    )
    oral_violation_score = oral_bioavailability_descriptors(pose.aligned_mol).violation_score
    return FragmentVoxelScore(
        mapped_atom_count=mapped_atom_count,
        pi_complement=pi_complement,
        pi_clash=pi_clash,
        sigma_shear=sigma_shear,
        reward=reward,
        oral_violation_score=oral_violation_score,
        oral_penalty=math.exp(-weights.oral_lambda * oral_violation_score),
    )


def saturated_reward(
    *,
    pi_complement: float,
    pi_clash: float,
    sigma_shear: float,
    u_pose: float,
    weights: RewardWeights,
    sa_score: float = 0.0,
    oral_violation_score: float = 0.0,
    pi_clash_lock: float = 0.0,
) -> float:
    """Compute the non-linear Track A reward for one candidate growth vector."""

    if weights.gamma <= 1.0:
        raise ValueError("gamma must be greater than 1.0 for saturated clash penalties")
    if weights.uncertainty_lambda < 0.0:
        raise ValueError("uncertainty_lambda must be non-negative")
    if weights.sa_lambda < 0.0 or weights.oral_lambda < 0.0:
        raise ValueError("reward attenuation lambdas must be non-negative")
    if min(pi_complement, pi_clash, pi_clash_lock, sigma_shear, u_pose, sa_score, oral_violation_score) < 0.0:
        raise ValueError("reward inputs must be non-negative")

    raw_score = (
        weights.complement * pi_complement
        - weights.clash * math.pow(pi_clash, weights.gamma)
        + weights.bias_lock * pi_clash_lock
        - weights.shear * math.log1p(sigma_shear)
    )
    attenuation = (
        math.exp(-weights.uncertainty_lambda * u_pose)
        * math.exp(-weights.sa_lambda * sa_score)
        * math.exp(-weights.oral_lambda * oral_violation_score)
    )
    return raw_score * attenuation


class GFlowNetActionSpace:
    """Small, framework-neutral environment facade used by PyTorch trainers."""

    def __init__(
        self,
        anchors: Sequence[ChemicalAnchor],
        boundaries: Mapping[tuple[str, int], BoundaryVoxel],
        weights: RewardWeights | None = None,
    ) -> None:
        self._anchors = tuple(anchors)
        self._boundaries = dict(boundaries)
        self._weights = weights or RewardWeights()

    @property
    def anchors(self) -> tuple[ChemicalAnchor, ...]:
        return self._anchors

    @property
    def weights(self) -> RewardWeights:
        return self._weights

    def legal_actions(
        self,
        state: GFlowNetState,
        exit_vectors: Sequence[ExitVector],
        candidate_voxels: Sequence[tuple[str, int]],
    ) -> tuple[GrowthAction, ...]:
        actions: list[GrowthAction] = []
        occupied = set(state.occupied_voxels)
        for condition_id, voxel_idx in candidate_voxels:
            boundary = self._boundaries.get((condition_id, voxel_idx))
            if boundary is None:
                continue
            if boundary.boundary_class is BoundaryClass.BRIDGE_ANCHOR_NO_FLY_ZONE:
                continue
            if voxel_idx in occupied:
                continue
            for anchor in self._anchors:
                for vector in exit_vectors:
                    unit_vector = vector.unit()
                    actions.append(
                        GrowthAction(
                            action_id=f"{state.scaffold_id}:{anchor.anchor_id}:{unit_vector.vector_id}:{condition_id}:{voxel_idx}",
                            anchor=anchor,
                            exit_vector=unit_vector,
                            target_condition_id=condition_id,
                            target_voxel_idx=voxel_idx,
                        )
                    )
        return tuple(actions)

    def score_action(self, action: GrowthAction) -> ScoredAction:
        boundary = self._boundaries[(action.target_condition_id, action.target_voxel_idx)]
        reward = saturated_reward(
            pi_complement=boundary.pi_complement,
            pi_clash=boundary.pi_clash,
            sigma_shear=boundary.sigma_shear,
            u_pose=action.exit_vector.u_pose,
            sa_score=synthetic_accessibility_score_for_smiles(action.anchor.canonical_smiles),
            oral_violation_score=oral_violation_score_for_smiles(action.anchor.canonical_smiles),
            weights=self._weights,
        )
        oral_violation_score = oral_violation_score_for_smiles(action.anchor.canonical_smiles)
        return ScoredAction(
            action=action,
            reward=reward,
            pi_complement=boundary.pi_complement,
            pi_clash=boundary.pi_clash,
            sigma_shear=boundary.sigma_shear,
            attenuation=math.exp(-self._weights.uncertainty_lambda * action.exit_vector.u_pose),
            oral_violation_score=oral_violation_score,
            oral_penalty=math.exp(-self._weights.oral_lambda * oral_violation_score),
        )

    def top_k_actions(self, actions: Sequence[GrowthAction], k: int) -> tuple[ScoredAction, ...]:
        if k <= 0:
            return ()
        scored = [self.score_action(action) for action in actions]
        return tuple(sorted(scored, key=lambda item: item.reward, reverse=True)[:k])

    def score_aligned_fragment_pose(
        self,
        pose: ScaffoldAlignedFragmentPose,
        *,
        condition_id: str,
        geometry: GridGeometry,
        u_pose: float,
    ) -> FragmentVoxelScore:
        return score_aligned_fragment_against_tso(
            pose,
            condition_id=condition_id,
            geometry=geometry,
            boundaries=self._boundaries,
            weights=self._weights,
            u_pose=u_pose,
        )


def compute_exit_vector_mask(exit_vector_contexts: Sequence[Sequence[ExitVectorPhaseContext]]) -> tuple[float, ...]:
    """Compute additive action-logit penalties from five-phase local field context."""

    penalties: list[float] = []
    for phases in exit_vector_contexts:
        if not phases:
            penalties.append(-2.0)
            continue
        stable_count = sum(1 for phase in phases if phase.stable_occupied)
        if stable_count == len(phases):
            penalties.append(float("-inf"))
            continue
        if stable_count >= 3:
            penalties.append(-50.0)
            continue
        if any(phase.thermally_activated for phase in phases):
            penalties.append(0.0)
            continue
        if any(phase.thermally_destabilized for phase in phases):
            cold_normalized = max(phase.cold_normalized for phase in phases)
            penalties.append(-5.0 * max(0.0, cold_normalized))
            continue
        penalties.append(-2.0)
    return tuple(penalties)


def load_vspace_survivors(
    path: Path = DEFAULT_VSPACE_SURVIVORS,
    *,
    limit: int | None = None,
) -> tuple[VSpaceSurvivor, ...]:
    """Load thermodynamically pruned V-Space survivors for GFlowNet actions."""

    query = pl.scan_parquet(path).select(
        [
            "product_id",
            "smiles",
            "synthon_a_id",
            "synthon_b_id",
            "score",
            "pi_complement",
            "pi_clash",
            "coordinates_json",
        ]
    )
    if limit is not None:
        if limit <= 0:
            raise ValueError("limit must be positive when provided")
        query = query.head(limit)
    frame = query.collect()
    survivors: list[VSpaceSurvivor] = []
    for row in frame.iter_rows(named=True):
        row_map = cast(Mapping[str, object], row)
        survivors.append(
            VSpaceSurvivor(
                product_id=_as_str(row_map.get("product_id"), "product_id"),
                canonical_smiles=_as_str(row_map.get("smiles"), "smiles"),
                synthon_a_id=_as_str(row_map.get("synthon_a_id"), "synthon_a_id"),
                synthon_b_id=_as_str(row_map.get("synthon_b_id"), "synthon_b_id"),
                score=_as_float(row_map.get("score"), "score"),
                pi_complement=_as_float(row_map.get("pi_complement"), "pi_complement"),
                pi_clash=_as_float(row_map.get("pi_clash"), "pi_clash"),
                coordinates_json=_as_str(row_map.get("coordinates_json"), "coordinates_json"),
            )
        )
    return tuple(survivors)


class EnrichedVSpaceActionSpace:
    """GFlowNet action space backed by pruned Enamine REAL V-Space survivors."""

    def __init__(self, survivors: Sequence[VSpaceSurvivor], weights: RewardWeights | None = None) -> None:
        if not survivors:
            raise ValueError("EnrichedVSpaceActionSpace requires at least one survivor")
        self._survivors = tuple(survivors)
        self._weights = weights or RewardWeights()

    @classmethod
    def from_parquet(
        cls,
        path: Path = DEFAULT_VSPACE_SURVIVORS,
        *,
        limit: int | None = None,
        weights: RewardWeights | None = None,
    ) -> "EnrichedVSpaceActionSpace":
        return cls(load_vspace_survivors(path, limit=limit), weights=weights)

    @property
    def survivors(self) -> tuple[VSpaceSurvivor, ...]:
        return self._survivors

    @property
    def weights(self) -> RewardWeights:
        return self._weights

    def legal_actions(
        self,
        state: GFlowNetState,
        exit_vectors: Sequence[ExitVector] | None = None,
    ) -> tuple[VSpaceGrowthAction, ...]:
        vectors = tuple(exit_vectors or ())
        if not vectors:
            return tuple(
                VSpaceGrowthAction(action_id=f"{state.scaffold_id}:{survivor.product_id}", survivor=survivor)
                for survivor in self._survivors
            )
        return tuple(
            VSpaceGrowthAction(
                action_id=f"{state.scaffold_id}:{survivor.product_id}:{vector.vector_id}",
                survivor=survivor,
                exit_vector=vector.unit(),
            )
            for survivor in self._survivors
            for vector in vectors
        )

    def score_action(self, action: VSpaceGrowthAction) -> ScoredVSpaceAction:
        survivor = action.survivor
        u_pose = action.exit_vector.u_pose if action.exit_vector is not None else 0.0
        oral_violation_score = oral_violation_score_for_smiles(survivor.canonical_smiles)
        attenuation = (
            math.exp(-self._weights.uncertainty_lambda * u_pose)
            * math.exp(-self._weights.sa_lambda * synthetic_accessibility_score_for_smiles(survivor.canonical_smiles))
            * math.exp(-self._weights.oral_lambda * oral_violation_score)
        )
        reward = (
            survivor.score
            + self._weights.complement * survivor.pi_complement
            - self._weights.clash * math.pow(max(survivor.pi_clash, 0.0), self._weights.gamma)
        ) * attenuation
        return ScoredVSpaceAction(
            action=action,
            reward=reward,
            pi_complement=survivor.pi_complement,
            pi_clash=survivor.pi_clash,
            attenuation=attenuation,
            oral_violation_score=oral_violation_score,
            oral_penalty=math.exp(-self._weights.oral_lambda * oral_violation_score),
        )

    def top_k_actions(self, actions: Sequence[VSpaceGrowthAction], k: int) -> tuple[ScoredVSpaceAction, ...]:
        if k <= 0:
            return ()
        scored = [self.score_action(action) for action in actions]
        return tuple(sorted(scored, key=lambda item: item.reward, reverse=True)[:k])


def load_enriched_vspace_action_space(
    path: Path = DEFAULT_VSPACE_SURVIVORS,
    *,
    limit: int | None = None,
    weights: RewardWeights | None = None,
) -> EnrichedVSpaceActionSpace:
    return EnrichedVSpaceActionSpace.from_parquet(path, limit=limit, weights=weights)
