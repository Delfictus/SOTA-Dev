#!/usr/bin/env python3
"""Generate zero-shot thermodynamic replacement motifs for the FRAG-A liability."""

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

import polars as pl
from rdkit import Chem


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from prism_dstw.io import write_provenance_parquet
from prism_dstw.orchestration.gflownet_action_space import align_fragment_to_scaffold_exit_vector


Coordinate: TypeAlias = tuple[float, float, float]

CAMPAIGN_DIR = REPO_ROOT / "campaigns/glp1r_aleniglipron"
TRACK0_DIR = CAMPAIGN_DIR / "track_0_manual_emulation"
TRACKA_DIR = CAMPAIGN_DIR / "track_a_generative"
N80_DIR = CAMPAIGN_DIR / "integrated_spike_events/n80_full_scale"

DEFAULT_PARENT_SDF = TRACK0_DIR / "conformers/ALENI-PARENT_whole_molecule_aligned.sdf"
DEFAULT_FRAGMENT_REGISTRY = TRACK0_DIR / "aleniglipron_brics_fragment_registry.json"
DEFAULT_GRID_MAPPING = TRACK0_DIR / "grid_coordinate_mapping.json"
DEFAULT_STERIC_ENV = TRACK0_DIR / "interface_steric_environment.parquet"
DEFAULT_SIGNAL_GRID = N80_DIR / "signal_grid_variance_channel.parquet"
DEFAULT_RISK_MAP = N80_DIR / "receptor_durability_risk_map.parquet"
DEFAULT_ANCHORS = TRACKA_DIR / "calibration_anchors_3d.parquet"
DEFAULT_FRAG_A_EDGE_INTERFERENCE = TRACK0_DIR / "layer2_fragments/FRAG-A/per_edge_interference.parquet"
DEFAULT_OUTPUT = TRACK0_DIR / "teaser_solutions.parquet"
LIABILITY_EDGE_ID = "glp1r_5VEX_WT:PHE143->TYR148"
LIABILITY_EDGE_LABEL = "PHE143 -> TYR148"
TARGET_CONDITION = "glp1r_5VEX_WT"
TARGET_BOND_LENGTH_A = 1.50
SA_MAX = 3.5
CLASH_MAX = 0.1
EPSILON = 1.0e-12


@dataclass(frozen=True)
class GridGeometry:
    condition_id: str
    grid_dim: int
    origin: Coordinate
    spacing: float


@dataclass(frozen=True)
class BoundaryBond:
    scaffold_atom_idx: int
    original_scaffold_atom_idx: int
    removed_fragment_atom_idx: int
    exit_vector: Coordinate
    exit_xyz: Coordinate


@dataclass(frozen=True)
class VoxelField:
    voxel_idx: int
    variance_class: str
    hit_count_cold_mean: float
    hit_count_warm_mean: float
    hit_count_delta: float
    pi_clash: float
    pi_complement: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--anchors", type=Path, default=DEFAULT_ANCHORS)
    parser.add_argument("--parent-sdf", type=Path, default=DEFAULT_PARENT_SDF)
    parser.add_argument("--fragment-registry", type=Path, default=DEFAULT_FRAGMENT_REGISTRY)
    parser.add_argument("--grid-mapping", type=Path, default=DEFAULT_GRID_MAPPING)
    parser.add_argument("--steric-env", type=Path, default=DEFAULT_STERIC_ENV)
    parser.add_argument("--signal-grid", type=Path, default=DEFAULT_SIGNAL_GRID)
    parser.add_argument("--risk-map", type=Path, default=DEFAULT_RISK_MAP)
    parser.add_argument("--frag-a-edge-interference", type=Path, default=DEFAULT_FRAG_A_EDGE_INTERFERENCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--top-k", type=int, default=10)
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


def sha256_path(path: Path, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def json_object(value: object, label: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object")
    return cast(dict[str, object], value)


def json_list(value: object, label: str) -> list[object]:
    if not isinstance(value, list):
        raise ValueError(f"{label} must be a JSON list")
    return value


def coordinate_from_json(value: object, label: str) -> Coordinate:
    raw = json_list(value, label)
    if len(raw) != 3:
        raise ValueError(f"{label} must contain exactly three coordinates")
    return (as_float(raw[0], f"{label}[0]"), as_float(raw[1], f"{label}[1]"), as_float(raw[2], f"{label}[2]"))


def load_json(path: Path) -> dict[str, object]:
    return json_object(json.loads(path.read_text(encoding="utf-8")), str(path))


def vector_sub(lhs: Coordinate, rhs: Coordinate) -> Coordinate:
    return (lhs[0] - rhs[0], lhs[1] - rhs[1], lhs[2] - rhs[2])


def vector_norm(value: Coordinate) -> float:
    return math.sqrt(value[0] * value[0] + value[1] * value[1] + value[2] * value[2])


def unit_vector(value: Coordinate, label: str) -> Coordinate:
    norm = vector_norm(value)
    if norm <= EPSILON:
        raise ValueError(f"{label} vector is zero length")
    return (value[0] / norm, value[1] / norm, value[2] / norm)


def atom_xyz(mol: Any, atom_idx: int) -> Coordinate:
    pos = mol.GetConformer().GetAtomPosition(atom_idx)
    return (float(pos.x), float(pos.y), float(pos.z))


def load_parent_mol(path: Path) -> Any:
    supplier = Chem.SDMolSupplier(str(path), removeHs=False)
    mol = supplier[0] if len(supplier) else None
    if mol is None:
        raise ValueError(f"{path} did not contain a valid parent conformer")
    return mol


def frag_a_parent_indices(registry_path: Path) -> set[int]:
    payload = load_json(registry_path)
    fragments = json_list(payload.get("fragments"), "fragments")
    for raw_fragment in fragments:
        fragment = json_object(raw_fragment, "fragment")
        if fragment.get("fragment_id") == "FRAG-A":
            return {as_int(value, "FRAG-A parent atom index") for value in json_list(fragment.get("parent_atom_indices"), "parent_atom_indices")}
    raise ValueError("FRAG-A was not present in the BRICS registry")


def remove_fragment_atoms(parent: Any, removed_atom_indices: set[int]) -> tuple[Any, dict[int, int]]:
    editable = Chem.RWMol(parent)
    index_map: dict[int, int] = {}
    next_idx = 0
    for atom in parent.GetAtoms():
        atom_idx = int(atom.GetIdx())
        if atom_idx in removed_atom_indices:
            continue
        index_map[atom_idx] = next_idx
        next_idx += 1
    for atom_idx in sorted(removed_atom_indices, reverse=True):
        editable.RemoveAtom(atom_idx)
    core = editable.GetMol()
    Chem.SanitizeMol(core, catchErrors=True)
    return core, index_map


def boundary_bonds(parent: Any, core_index_map: dict[int, int], removed_atom_indices: set[int]) -> list[BoundaryBond]:
    bonds: list[BoundaryBond] = []
    for bond in parent.GetBonds():
        begin = int(bond.GetBeginAtomIdx())
        end = int(bond.GetEndAtomIdx())
        begin_is_removed = begin in removed_atom_indices
        end_is_removed = end in removed_atom_indices
        if begin_is_removed == end_is_removed:
            continue
        removed_idx = begin if begin_is_removed else end
        scaffold_original_idx = end if begin_is_removed else begin
        if scaffold_original_idx not in core_index_map:
            continue
        if int(parent.GetAtomWithIdx(scaffold_original_idx).GetAtomicNum()) <= 1:
            continue
        if int(parent.GetAtomWithIdx(removed_idx).GetAtomicNum()) <= 1:
            continue
        scaffold_xyz = atom_xyz(parent, scaffold_original_idx)
        removed_xyz = atom_xyz(parent, removed_idx)
        bonds.append(
            BoundaryBond(
                scaffold_atom_idx=core_index_map[scaffold_original_idx],
                original_scaffold_atom_idx=scaffold_original_idx,
                removed_fragment_atom_idx=removed_idx,
                exit_vector=unit_vector(vector_sub(removed_xyz, scaffold_xyz), "FRAG-A severed bond"),
                exit_xyz=scaffold_xyz,
            )
        )
    if not bonds:
        raise ValueError("no scaffold/FRAG-A boundary bonds were found")
    return bonds


def load_grid_geometry(path: Path, condition_id: str) -> GridGeometry:
    payload = load_json(path)
    conditions = json_object(payload.get("conditions"), "conditions")
    geometry = json_object(conditions.get(condition_id), condition_id)
    return GridGeometry(
        condition_id=condition_id,
        grid_dim=as_int(geometry["grid_dim"], "grid_dim"),
        origin=coordinate_from_json(geometry.get("origin_xyz_angstrom"), "origin_xyz_angstrom"),
        spacing=as_float(geometry["spacing_angstrom"], "spacing_angstrom"),
    )


def coordinate_to_voxel(coord: Coordinate, geometry: GridGeometry) -> int | None:
    x_idx = math.floor((coord[0] - geometry.origin[0]) / geometry.spacing)
    y_idx = math.floor((coord[1] - geometry.origin[1]) / geometry.spacing)
    z_idx = math.floor((coord[2] - geometry.origin[2]) / geometry.spacing)
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


def normalization_constants(steric_env: Path) -> tuple[float, float]:
    base = pl.scan_parquet(steric_env).filter(pl.col("variance_class") != "void")
    cold = base.select(pl.col("hit_count_cold_mean").quantile(0.95).alias("q")).collect().item()
    delta = base.select(pl.col("hit_count_delta").abs().quantile(0.95).alias("q")).collect().item()
    return (max(as_float(cold, "cold_q95"), 1.0), max(as_float(delta, "delta_q95"), 1.0))


def liability_signed_te(risk_map: Path) -> float:
    row = (
        pl.scan_parquet(risk_map)
        .filter(pl.col("condition_id") == TARGET_CONDITION)
        .filter(pl.col("edge_from_residue") == 7)
        .filter(pl.col("edge_to_residue") == 12)
        .select("signed_te_mean")
        .head(1)
        .collect()
    )
    if row.height != 1:
        return 1.0
    return as_float(row.item(), "signed_te_mean")


def liability_beta_calibration(path: Path) -> tuple[float, float]:
    row = (
        pl.scan_parquet(path)
        .filter(pl.col("edge_id") == LIABILITY_EDGE_ID)
        .select(["beta_f", "beta_s"])
        .head(1)
        .collect()
    )
    if row.height != 1:
        raise ValueError(f"{path} did not contain beta calibration for {LIABILITY_EDGE_ID}")
    values = row.to_dicts()[0]
    return (as_float(values["beta_f"], "beta_f"), as_float(values["beta_s"], "beta_s"))


def score_atom_interference(
    *,
    variance_class: str,
    hit_count_cold_mean: float,
    hit_count_delta: float,
    signed_te_mean: float,
    cold_q95: float,
    delta_q95: float,
) -> tuple[float, float]:
    cold_norm = hit_count_cold_mean / cold_q95
    delta_norm = abs(hit_count_delta) / delta_q95
    if variance_class == "stable_occupied":
        return (cold_norm, 0.0)
    if variance_class == "thermally_activated":
        return (0.0, delta_norm)
    if variance_class == "thermally_destabilized":
        if signed_te_mean >= 0.0:
            return (0.0, 0.5 * delta_norm)
        return (0.5 * delta_norm, 0.0)
    return (0.0, 0.0)


def load_edge_fields(steric_env: Path, risk_map: Path) -> dict[int, VoxelField]:
    cold_q95, delta_q95 = normalization_constants(steric_env)
    signed_te = liability_signed_te(risk_map)
    rows = (
        pl.scan_parquet(steric_env)
        .filter(pl.col("edge_id") == LIABILITY_EDGE_ID)
        .select(
            [
                "voxel_idx",
                "variance_class",
                "hit_count_cold_mean",
                "hit_count_warm_mean",
                "hit_count_delta",
            ]
        )
        .collect()
    )
    if rows.height == 0:
        raise ValueError(f"no steric environment rows found for {LIABILITY_EDGE_ID}")
    lookup: dict[int, VoxelField] = {}
    for row in rows.iter_rows(named=True):
        variance_class = str(row["variance_class"])
        clash, complement = score_atom_interference(
            variance_class=variance_class,
            hit_count_cold_mean=as_float(row["hit_count_cold_mean"], "hit_count_cold_mean"),
            hit_count_delta=as_float(row["hit_count_delta"], "hit_count_delta"),
            signed_te_mean=signed_te,
            cold_q95=cold_q95,
            delta_q95=delta_q95,
        )
        voxel_idx = as_int(row["voxel_idx"], "voxel_idx")
        lookup[voxel_idx] = VoxelField(
            voxel_idx=voxel_idx,
            variance_class=variance_class,
            hit_count_cold_mean=as_float(row["hit_count_cold_mean"], "hit_count_cold_mean"),
            hit_count_warm_mean=as_float(row["hit_count_warm_mean"], "hit_count_warm_mean"),
            hit_count_delta=as_float(row["hit_count_delta"], "hit_count_delta"),
            pi_clash=clash,
            pi_complement=complement,
        )
    return lookup


def signal_voxel_lookup(signal_grid: Path) -> set[int]:
    frame = (
        pl.scan_parquet(signal_grid)
        .filter(pl.col("condition_id") == TARGET_CONDITION)
        .filter(pl.col("variance_class") != "void")
        .select("voxel_idx")
        .collect()
    )
    return {as_int(value, "voxel_idx") for value in frame.get_column("voxel_idx").to_list()}


def synthetic_accessibility_score(smiles: str) -> float:
    try:
        sascorer = cast(Any, import_module("rdkit.Contrib.SA_Score.sascorer"))
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return 10.0
        return as_float(sascorer.calculateScore(mol), "SA score")
    except Exception:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return 10.0
        mol_any = cast(Any, mol)
        heavy_atoms = sum(1 for atom in mol_any.GetAtoms() if int(atom.GetAtomicNum()) > 1)
        rings = int(mol_any.GetRingInfo().NumRings())
        return max(1.0, min(10.0, 1.0 + 0.12 * float(heavy_atoms) + 0.35 * float(rings)))


def mol_from_anchor_row(row: dict[str, object]) -> Any:
    smiles = str(row["canonical_smiles"])
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"could not parse anchor SMILES: {smiles}")
    atom_rows = json_list(json.loads(str(row["conformer_atoms_json"])), "conformer_atoms_json")
    conformer = Chem.Conformer(mol.GetNumAtoms())
    assigned: set[int] = set()
    for raw_atom in atom_rows:
        atom = json_object(raw_atom, "conformer_atom")
        atom_idx = as_int(atom["atom_idx"], "atom_idx")
        if atom_idx < 0 or atom_idx >= mol.GetNumAtoms():
            continue
        conformer.SetAtomPosition(
            atom_idx,
            (as_float(atom["x"], "x"), as_float(atom["y"], "y"), as_float(atom["z"], "z")),
        )
        assigned.add(atom_idx)
    if len(assigned) != mol.GetNumAtoms():
        raise ValueError(f"stored conformer did not cover all atoms for {smiles}")
    mol.AddConformer(conformer)
    return mol


def heavy_atom_indices(mol: Any) -> list[int]:
    return [int(atom.GetIdx()) for atom in mol.GetAtoms() if int(atom.GetAtomicNum()) > 1]


def score_pose(
    pose_mol: Any,
    geometry: GridGeometry,
    signal_voxels: set[int],
    edge_fields: dict[int, VoxelField],
) -> dict[str, object]:
    pi_clash = 0.0
    pi_complement = 0.0
    mapped_voxels: list[int] = []
    edge_voxels: list[int] = []
    variance_classes: list[str] = []
    signal_mapped = 0
    for atom_idx in heavy_atom_indices(pose_mol):
        voxel_idx = coordinate_to_voxel(atom_xyz(pose_mol, atom_idx), geometry)
        if voxel_idx is None:
            continue
        mapped_voxels.append(voxel_idx)
        if voxel_idx in signal_voxels:
            signal_mapped += 1
        field = edge_fields.get(voxel_idx)
        if field is None:
            continue
        edge_voxels.append(voxel_idx)
        variance_classes.append(field.variance_class)
        pi_clash += field.pi_clash
        pi_complement += field.pi_complement
    return {
        "pi_clash": pi_clash,
        "pi_complement": pi_complement,
        "mapped_atom_count": len(mapped_voxels),
        "signal_mapped_atom_count": signal_mapped,
        "edge_matched_atom_count": len(edge_voxels),
        "mapped_voxel_indices_json": json.dumps(mapped_voxels, separators=(",", ":")),
        "edge_voxel_indices_json": json.dumps(edge_voxels, separators=(",", ":")),
        "thermally_activated_voxel_count": sum(1 for item in variance_classes if item == "thermally_activated"),
        "stable_occupied_voxel_count": sum(1 for item in variance_classes if item == "stable_occupied"),
        "thermally_destabilized_voxel_count": sum(1 for item in variance_classes if item == "thermally_destabilized"),
    }


def screen_anchors(args: argparse.Namespace) -> pl.DataFrame:
    parent = load_parent_mol(args.parent_sdf)
    removed = frag_a_parent_indices(args.fragment_registry)
    core, core_index_map = remove_fragment_atoms(parent, removed)
    exits = boundary_bonds(parent, core_index_map, removed)
    geometry = load_grid_geometry(args.grid_mapping, TARGET_CONDITION)
    signal_voxels = signal_voxel_lookup(args.signal_grid)
    edge_fields = load_edge_fields(args.steric_env, args.risk_map)
    beta_f, beta_s = liability_beta_calibration(args.frag_a_edge_interference)

    anchors = (
        pl.scan_parquet(args.anchors)
        .filter(pl.col("generation_status") == "ok")
        .select(
            [
                "anchor_idx",
                "anchor_id",
                "source_anchor_id",
                "source_kind",
                "canonical_smiles",
                "n_heavy_atoms",
                "molecular_weight",
                "steric_volume_A3",
                "force_field",
                "conformer_atoms_json",
                "epistemic_class",
            ]
        )
        .collect()
    )
    rows: list[dict[str, object]] = []
    for row in anchors.iter_rows(named=True):
        smiles = str(row["canonical_smiles"])
        sa_score = synthetic_accessibility_score(smiles)
        if sa_score > SA_MAX:
            continue
        try:
            fragment = mol_from_anchor_row(row)
        except ValueError:
            continue
        best_row: dict[str, object] | None = None
        for exit_bond in exits:
            for attachment_idx in heavy_atom_indices(fragment):
                pose = align_fragment_to_scaffold_exit_vector(
                    fragment,
                    core,
                    exit_bond.scaffold_atom_idx,
                    fragment_attachment_atom_idx=attachment_idx,
                    scaffold_exit_vector=exit_bond.exit_vector,
                    attachment_bond_length_A=TARGET_BOND_LENGTH_A,
                )
                score = score_pose(pose.aligned_mol, geometry, signal_voxels, edge_fields)
                pi_clash = as_float(score["pi_clash"], "pi_clash")
                pi_complement = as_float(score["pi_complement"], "pi_complement")
                te_multiplier = math.exp(beta_f * pi_clash) * math.exp(-beta_s * pi_complement)
                candidate = {
                    "anchor_idx": as_int(row["anchor_idx"], "anchor_idx"),
                    "anchor_id": str(row["anchor_id"]),
                    "source_anchor_id": str(row["source_anchor_id"]),
                    "source_kind": str(row["source_kind"]),
                    "canonical_smiles": smiles,
                    "n_heavy_atoms": as_int(row["n_heavy_atoms"], "n_heavy_atoms"),
                    "molecular_weight": as_float(row["molecular_weight"], "molecular_weight"),
                    "steric_volume_A3": as_float(row["steric_volume_A3"], "steric_volume_A3"),
                    "force_field": str(row["force_field"]),
                    "sa_score": sa_score,
                    "pi_clash": pi_clash,
                    "pi_complement": pi_complement,
                    "beta_f": beta_f,
                    "beta_s": beta_s,
                    "te_multiplier": te_multiplier,
                    "projected_durability_improvement": max(0.0, 1.0 - te_multiplier),
                    "mapped_atom_count": as_int(score["mapped_atom_count"], "mapped_atom_count"),
                    "signal_mapped_atom_count": as_int(score["signal_mapped_atom_count"], "signal_mapped_atom_count"),
                    "edge_matched_atom_count": as_int(score["edge_matched_atom_count"], "edge_matched_atom_count"),
                    "mapped_voxel_indices_json": str(score["mapped_voxel_indices_json"]),
                    "edge_voxel_indices_json": str(score["edge_voxel_indices_json"]),
                    "thermally_activated_voxel_count": as_int(
                        score["thermally_activated_voxel_count"], "thermally_activated_voxel_count"
                    ),
                    "stable_occupied_voxel_count": as_int(score["stable_occupied_voxel_count"], "stable_occupied_voxel_count"),
                    "thermally_destabilized_voxel_count": as_int(
                        score["thermally_destabilized_voxel_count"], "thermally_destabilized_voxel_count"
                    ),
                    "scaffold_exit_atom_idx": exit_bond.scaffold_atom_idx,
                    "original_scaffold_exit_atom_idx": exit_bond.original_scaffold_atom_idx,
                    "removed_frag_a_atom_idx": exit_bond.removed_fragment_atom_idx,
                    "fragment_attachment_atom_idx": pose.fragment_attachment_atom_idx,
                    "attachment_bond_length_A": pose.attachment_bond_length_A,
                    "exit_vector_json": json.dumps(list(exit_bond.exit_vector), separators=(",", ":")),
                    "scaffold_exit_xyz_json": json.dumps(list(exit_bond.exit_xyz), separators=(",", ":")),
                    "aligned_conformer_atoms_json": pose.conformer_atoms_json,
                    "liability_edge_id": LIABILITY_EDGE_ID,
                    "liability_edge_label": LIABILITY_EDGE_LABEL,
                    "condition_id": TARGET_CONDITION,
                    "selection_status": "passes_zero_shot_filter",
                    "anchor_epistemic_class": str(row["epistemic_class"]),
                    "solution_epistemic_class": "PROJECTED",
                }
                if best_row is None:
                    best_row = candidate
                    continue
                if (
                    as_float(candidate["pi_complement"], "candidate_pi_complement"),
                    -as_float(candidate["pi_clash"], "candidate_pi_clash"),
                    -as_float(candidate["sa_score"], "candidate_sa_score"),
                ) > (
                    as_float(best_row["pi_complement"], "best_pi_complement"),
                    -as_float(best_row["pi_clash"], "best_pi_clash"),
                    -as_float(best_row["sa_score"], "best_sa_score"),
                ):
                    best_row = candidate
        if best_row is not None and as_float(best_row["pi_clash"], "pi_clash") <= CLASH_MAX:
            rows.append(best_row)

    if len(rows) < args.top_k:
        raise ValueError(f"only {len(rows)} anchors passed the clash and SA filters; requested top_k={args.top_k}")
    frame = pl.DataFrame(rows).sort(["pi_complement", "pi_clash", "sa_score"], descending=[True, False, False]).head(args.top_k)
    return frame.with_row_index("solution_rank", offset=1)


def main() -> int:
    args = parse_args()
    frame = screen_anchors(args)
    output = write_provenance_parquet(
        frame,
        args.output,
        producer_script=Path(__file__),
        source_parquets=[args.anchors, args.signal_grid, args.steric_env, args.risk_map, args.frag_a_edge_interference],
        schema_version="teaser_solutions.v1",
        pipeline_stage="zero_shot_teaser_solutions",
        partition_keys=["condition_id", "liability_edge_id"],
        extra_metadata={
            "parent_sdf_sha256": sha256_path(args.parent_sdf),
            "fragment_registry_sha256": sha256_path(args.fragment_registry),
            "grid_mapping_sha256": sha256_path(args.grid_mapping),
            "liability_edge_id": LIABILITY_EDGE_ID,
            "frag_a_beta_source_sha256": sha256_path(args.frag_a_edge_interference),
            "selection_filter": {"pi_clash_max": CLASH_MAX, "sa_score_max": SA_MAX, "sort": "pi_complement_desc"},
        },
        ledger_parameters={
            "top_k": args.top_k,
            "target_bond_length_A": TARGET_BOND_LENGTH_A,
            "frag_a_removed": True,
        },
        ledger_output_value={"row_count": frame.height, "output_path": args.output.as_posix()},
    )
    print(f"wrote {output.relative_to(REPO_ROOT)} rows={frame.height}")
    print(frame.select(["solution_rank", "anchor_id", "source_anchor_id", "pi_complement", "pi_clash", "sa_score"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
