#!/usr/bin/env python3
"""Train the Track A Fiber-Bundle GFlowNet policy against the Rust oracle."""

from __future__ import annotations

import argparse
import asyncio
import csv
import hashlib
import json
import math
import random
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence, cast

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import polars as pl
import torch
from rdkit import Chem
from torch import Tensor
from torch.nn import functional as F
from torch_geometric.data import Batch, Data  # type: ignore[import-untyped]

from prism_dstw.hierarchical_bayes.gflownet_policy import FiberBundleGFlowNetPolicy
from prism_dstw.hierarchical_bayes.trajectory_balance import TrajectoryBalanceLoss
from prism_dstw.orchestration.multi_scaffold_entropy_router import (
    MultiScaffoldEntropyRouter,
    lock_phase_maps_from_rows,
    phase_occupancy_from_fiber_bundle,
)
from prism_dstw.orchestration.rust_reward_oracle import (
    BatchedRustOracle,
    OracleProposal,
    telemetry_to_dict,
)
from prism_dstw.scoring.tripartite_bias_scorer import compute_tripartite_bias


REPO_ROOT = Path(__file__).resolve().parents[1]
TRACK_A_DIR = REPO_ROOT / "campaigns/glp1r_aleniglipron/track_a_generative"
N80_DIR = REPO_ROOT / "campaigns/glp1r_aleniglipron/integrated_spike_events/n80_full_scale"
DEFAULT_LIGAND_SDF = TRACK_A_DIR / "ALENI-PARENT_6XOX_frame_o3a_relaxed.sdf"
DEFAULT_ORFOR_SDF = TRACK_A_DIR / "conformers/ORFOR-PARENT_6XOX_o3a.sdf"
DEFAULT_DANU_SDF = TRACK_A_DIR / "conformers/DANU-PARENT_6XOX_o3a.sdf"
DEFAULT_SCAFFOLD_POOL = (DEFAULT_LIGAND_SDF, DEFAULT_ORFOR_SDF, DEFAULT_DANU_SDF)
DEFAULT_ANCHORS = TRACK_A_DIR / "enamine_115k_synthons_3d.parquet"
DEFAULT_SURVIVORS = TRACK_A_DIR / "vspace_survivors_full_scale.parquet"
DEFAULT_RESIDUE_PHASE = N80_DIR / "residue_phase_tensor.parquet"
DEFAULT_INTERFEROMETRIC = N80_DIR / "interferometric_differential.parquet"
DEFAULT_TOPOLOGY = REPO_ROOT / "04_TOPOLOGIES/glp1r_6XOX_HOLO_ALENI.topology.json"
DEFAULT_FRAGMENT_REGISTRY = (
    REPO_ROOT / "campaigns/glp1r_aleniglipron/track_0_manual_emulation/aleniglipron_brics_fragment_registry.json"
)
DEFAULT_PHASE3_PGX_MANIFEST = REPO_ROOT / "campaigns/glp1r_aleniglipron/Phase3_PGx_Exclusion_Manifest.json"
DEFAULT_OUTPUT_DIR = TRACK_A_DIR
PHASES = ("cold_hold", "ramp_up", "warm_hold", "ramp_down", "cold_return")
PHASE_LABELS = ("Cold Hold", "Ramp Up", "Warm Hold", "Ramp Down", "Cold Return")


@dataclass(frozen=True)
class TrainingPaths:
    ligand_sdf: Path
    anchors: Path
    survivors: Path
    residue_phase: Path
    interferometric: Path
    topology: Path
    fragment_registry: Path
    output_dir: Path


@dataclass(frozen=True)
class ScaffoldGraph:
    scaffold_id: str
    ligand_sdf: Path
    data: Data
    exit_node_idx: int
    base_feature_dim: int
    phase_feature_dim: int
    edge_feature_dim: int
    retained_atom_count: int


@dataclass(frozen=True)
class ActionSpace:
    table: pl.DataFrame
    anchor_embeddings: Tensor
    valid_mask: Tensor
    reward_targets: Tensor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--warm-start-epochs", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=20260524)
    parser.add_argument("--hidden-dim", type=int, default=96)
    parser.add_argument("--embedding-dim", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1.0e-3)
    parser.add_argument("--log-z-learning-rate", type=float, default=1.0e-2)
    parser.add_argument("--learning-rate-floor", type=float, default=1.0e-5)
    parser.add_argument("--log-z-learning-rate-floor", type=float, default=1.0e-4)
    parser.add_argument("--policy-warmup-epochs", type=int, default=20)
    parser.add_argument("--log-z-warmup-epochs", type=int, default=10)
    parser.add_argument("--reward-scale", type=float, default=10.0)
    parser.add_argument("--policy-aux-weight", type=float, default=0.0)
    parser.add_argument("--diversity-beta", type=float, default=0.5)
    parser.add_argument("--max-trajectory-steps", type=int, default=5)
    parser.add_argument("--reward-version", type=str, default="v2_tripartite")
    parser.add_argument("--lock-directional-bias-alpha", type=float, default=2.0)
    parser.add_argument("--lock-reaching-synthon-boost", type=float, default=2.0)
    parser.add_argument("--torch-threads", type=int, default=4)
    parser.add_argument("--entropy-router", action="store_true", default=True)
    parser.add_argument("--disable-entropy-router", action="store_true", default=False)
    parser.add_argument("--dstw-aligned", action="store_true", default=True)
    parser.add_argument("--router-telemetry-interval", type=int, default=10)
    parser.add_argument("--active-variant", type=str, default="WT")
    parser.add_argument("--phase3-pgx-manifest", type=Path, default=DEFAULT_PHASE3_PGX_MANIFEST)
    parser.add_argument("--ligand-sdf", type=Path, default=DEFAULT_LIGAND_SDF)
    parser.add_argument(
        "--scaffold-pool",
        type=Path,
        action="append",
        default=None,
        help="Repeatable SDF path for multi-scaffold trajectory initialization.",
    )
    parser.add_argument("--anchors", type=Path, default=DEFAULT_ANCHORS)
    parser.add_argument("--survivors", type=Path, default=DEFAULT_SURVIVORS)
    parser.add_argument("--residue-phase", type=Path, default=DEFAULT_RESIDUE_PHASE)
    parser.add_argument("--interferometric", type=Path, default=DEFAULT_INTERFEROMETRIC)
    parser.add_argument("--topology", type=Path, default=DEFAULT_TOPOLOGY)
    parser.add_argument("--fragment-registry", type=Path, default=DEFAULT_FRAGMENT_REGISTRY)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(False)


def read_first_mol(path: Path) -> Chem.Mol:
    supplier = Chem.SDMolSupplier(str(path), removeHs=False)
    if not supplier or supplier[0] is None:
        raise RuntimeError(f"could not read ligand SDF: {path}")
    mol = supplier[0]
    if mol.GetNumConformers() == 0:
        raise RuntimeError(f"ligand SDF has no conformer: {path}")
    return mol


def mol_bonds(mol: Chem.Mol) -> list[Chem.Bond]:
    return list(mol.GetBonds())  # type: ignore[no-untyped-call]


def mol_atoms(mol: Chem.Mol) -> list[Chem.Atom]:
    return list(mol.GetAtoms())  # type: ignore[no-untyped-call]


def numeric_value(value: object, default: float = 0.0) -> float:
    if value is None:
        return default
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return float(str(value))
    except ValueError:
        return default


def removed_fragment_atoms(mol: Chem.Mol, registry_path: Path, fragment_id: str = "FRAG-A") -> set[int]:
    registry = json.loads(registry_path.read_text())
    fragment_atoms: set[int] = set()
    for fragment in registry.get("fragments", []):
        if fragment.get("fragment_id") == fragment_id:
            fragment_atoms = {int(idx) for idx in fragment["parent_atom_indices"]}
            break
    if not fragment_atoms:
        raise RuntimeError(f"{fragment_id} not found in {registry_path}")
    removed = set(fragment_atoms)
    for atom_idx in fragment_atoms:
        atom = mol.GetAtomWithIdx(atom_idx)
        for neighbor in atom.GetNeighbors():
            if neighbor.GetAtomicNum() == 1:
                removed.add(int(neighbor.GetIdx()))
    return removed


def boundary_exit_atom(mol: Chem.Mol, removed_atoms: set[int]) -> int:
    fragment_heavy = {idx for idx in removed_atoms if mol.GetAtomWithIdx(idx).GetAtomicNum() > 1}
    candidates: list[int] = []
    for bond in mol_bonds(mol):
        lhs = int(bond.GetBeginAtomIdx())
        rhs = int(bond.GetEndAtomIdx())
        lhs_removed = lhs in fragment_heavy
        rhs_removed = rhs in fragment_heavy
        if lhs_removed == rhs_removed:
            continue
        scaffold_idx = rhs if lhs_removed else lhs
        if scaffold_idx not in removed_atoms and mol.GetAtomWithIdx(scaffold_idx).GetAtomicNum() > 1:
            candidates.append(scaffold_idx)
    if not candidates:
        raise RuntimeError("could not identify scaffold exit atom")
    return min(candidates)


def inferred_competitor_exit_atom(mol: Chem.Mol) -> int:
    """Select a deterministic exposed heavy atom when no FRAG-A registry exists."""

    conformer = mol.GetConformer()
    heavy_atoms = [int(atom.GetIdx()) for atom in mol_atoms(mol) if atom.GetAtomicNum() > 1]
    if not heavy_atoms:
        raise RuntimeError("competitor scaffold has no heavy atoms")
    centroid = torch.zeros(3, dtype=torch.float32)
    for atom_idx in heavy_atoms:
        centroid += torch.tensor(list(conformer.GetAtomPosition(atom_idx)), dtype=torch.float32)
    centroid /= float(len(heavy_atoms))
    terminal_heavy_atoms: list[int] = []
    for atom_idx in heavy_atoms:
        atom = mol.GetAtomWithIdx(atom_idx)
        heavy_degree = sum(1 for neighbor in atom.GetNeighbors() if neighbor.GetAtomicNum() > 1)
        if heavy_degree <= 1:
            terminal_heavy_atoms.append(atom_idx)
    candidate_atoms = terminal_heavy_atoms or heavy_atoms

    def distance_from_centroid(atom_idx: int) -> float:
        xyz = torch.tensor(list(conformer.GetAtomPosition(atom_idx)), dtype=torch.float32)
        return float(torch.linalg.vector_norm(xyz - centroid).item())

    return max(candidate_atoms, key=distance_from_centroid)


def scaffold_paths_from_args(raw_scaffold_pool: Sequence[Path] | None, ligand_sdf: Path) -> tuple[Path, ...]:
    candidates = tuple(Path(path) for path in raw_scaffold_pool) if raw_scaffold_pool else DEFAULT_SCAFFOLD_POOL
    if ligand_sdf not in candidates:
        candidates = (ligand_sdf, *candidates)
    existing: list[Path] = []
    for path in candidates:
        if not path.is_file():
            continue
        if path not in existing:
            existing.append(path)
    if not existing:
        raise RuntimeError("no scaffold SDFs were available for training initialization")
    return tuple(existing)


def topology_ca_coordinates(path: Path) -> list[tuple[int, list[float]]]:
    topology = json.loads(path.read_text())
    positions = topology["positions"]
    atom_names = topology["atom_names"]
    residue_ids = topology["residue_ids"]
    ca_indices = topology.get("ca_indices") or [
        idx for idx, atom_name in enumerate(atom_names) if str(atom_name).strip().upper() == "CA"
    ]
    coordinates: list[tuple[int, list[float]]] = []
    for atom_idx in ca_indices:
        residue_id = int(residue_ids[atom_idx])
        if residue_id >= int(topology["n_residues"]) - 1:
            continue
        offset = int(atom_idx) * 3
        coordinates.append(
            (
                residue_id,
                [float(positions[offset]), float(positions[offset + 1]), float(positions[offset + 2])],
            )
        )
    if len(coordinates) < 8:
        raise RuntimeError(f"insufficient CA coordinates in topology: {path}")
    return coordinates


def nearest_residue_idx(xyz: Sequence[float], ca_coordinates: Sequence[tuple[int, Sequence[float]]]) -> int:
    best_residue = ca_coordinates[0][0]
    best_dist = float("inf")
    for residue_idx, ca_xyz in ca_coordinates:
        dist = sum((float(xyz[axis]) - float(ca_xyz[axis])) ** 2 for axis in range(3))
        if dist < best_dist:
            best_dist = dist
            best_residue = residue_idx
    return best_residue


def load_phase_feature_maps(paths: TrainingPaths, condition_id: str = "glp1r_6XOX_WT") -> dict[int, list[list[float]]]:
    residue_phase = (
        pl.scan_parquet(paths.residue_phase)
        .filter(pl.col("condition_id") == condition_id)
        .collect()
    )
    interferometric = (
        pl.scan_parquet(paths.interferometric)
        .filter(pl.col("condition_id") == condition_id)
        .collect()
    )
    if residue_phase.height == 0:
        raise RuntimeError(f"no residue phase rows for {condition_id}")
    max_count = 1.0
    count_columns = [f"{group}_{phase}_spikes" for group in ("A", "B", "C", "D") for phase in PHASES]
    for column in count_columns:
        if column in residue_phase.columns:
            max_count = max(max_count, numeric_value(residue_phase.get_column(column).max(), 1.0))
    phase_rows = {int(row["primary_residue_idx"]): row for row in residue_phase.to_dicts()}
    interferometric_rows = {int(row["primary_residue_idx"]): row for row in interferometric.to_dicts()}
    feature_map: dict[int, list[list[float]]] = {}
    for residue_idx, row in phase_rows.items():
        diff_row = interferometric_rows.get(residue_idx, {})
        matrix: list[list[float]] = []
        for phase_index, phase in enumerate(PHASES):
            counts = [
                math.log1p(numeric_value(row.get(f"{group}_{phase}_spikes"), 0.0)) / math.log1p(max_count)
                for group in ("A", "B", "C", "D")
            ]
            ratios = [
                min(numeric_value(diff_row.get(f"{group}_over_A_{phase}_ratio"), 0.0), 5.0) / 5.0
                for group in ("B", "C", "D")
            ]
            matrix.append([*counts, *ratios, phase_index / float(len(PHASES) - 1)])
        feature_map[residue_idx] = matrix
    return feature_map


def atom_base_features(atom: Chem.Atom, partial_charge: float, is_exit: bool) -> list[float]:
    hybridization = atom.GetHybridization()
    return [
        atom.GetAtomicNum() / 100.0,
        atom.GetTotalDegree() / 6.0,
        atom.GetFormalCharge() / 4.0,
        1.0 if atom.GetIsAromatic() else 0.0,
        1.0 if atom.IsInRing() else 0.0,
        1.0 if hybridization == Chem.HybridizationType.SP else 0.0,
        1.0 if hybridization == Chem.HybridizationType.SP2 else 0.0,
        1.0 if hybridization == Chem.HybridizationType.SP3 else 0.0,
        atom.GetMass() / 250.0,
        partial_charge,
        1.0 if is_exit else 0.0,
        1.0,
    ]


def atom_partial_charge(mol: Chem.Mol, atom_idx: int) -> float:
    atom = mol.GetAtomWithIdx(atom_idx)
    for prop in ("AM1BCCCharge", "PartialCharge", "_GasteigerCharge"):
        if atom.HasProp(prop):
            try:
                return float(atom.GetProp(prop))
            except ValueError:
                return 0.0
    return 0.0


def build_edges(
    mol: Chem.Mol,
    retained_original_indices: Sequence[int],
    old_to_new: Mapping[int, int],
    exit_original_idx: int,
) -> tuple[Tensor, Tensor, Tensor]:
    conf = mol.GetConformer()
    coords = {
        old_idx: torch.tensor(list(conf.GetAtomPosition(old_idx)), dtype=torch.float32)
        for old_idx in retained_original_indices
    }
    edge_pairs: list[tuple[int, int]] = []
    edge_attr: list[list[float]] = []
    active_mask: list[bool] = []
    bonded_pairs: set[tuple[int, int]] = set()
    exit_xyz = coords[exit_original_idx]

    def add_edge(lhs_old: int, rhs_old: int, bond_order: float, aromatic: bool, conjugated: bool, same_ring: bool) -> None:
        lhs = old_to_new[lhs_old]
        rhs = old_to_new[rhs_old]
        vector = coords[rhs_old] - coords[lhs_old]
        distance = float(torch.linalg.vector_norm(vector).item())
        exit_vector = coords[lhs_old] - exit_xyz
        denom = float(torch.linalg.vector_norm(vector).item() * torch.linalg.vector_norm(exit_vector).item())
        exit_alignment = float(torch.dot(vector, exit_vector).item() / denom) if denom > 1.0e-6 else 0.0
        edge_pairs.append((lhs, rhs))
        edge_attr.append(
            [
                distance / 5.0,
                bond_order / 3.0,
                1.0 if aromatic else 0.0,
                1.0 if conjugated else 0.0,
                1.0 if same_ring else 0.0,
                exit_alignment,
            ]
        )
        active_mask.append(bool(aromatic or conjugated))

    for bond in mol_bonds(mol):
        lhs = int(bond.GetBeginAtomIdx())
        rhs = int(bond.GetEndAtomIdx())
        if lhs not in old_to_new or rhs not in old_to_new:
            continue
        bond_order = float(bond.GetBondTypeAsDouble())
        aromatic = bool(bond.GetIsAromatic())
        conjugated = bool(bond.GetIsConjugated())
        same_ring = bool(mol.GetRingInfo().NumBondRings(bond.GetIdx()) > 0)
        bonded_pairs.add((min(lhs, rhs), max(lhs, rhs)))
        add_edge(lhs, rhs, bond_order, aromatic, conjugated, same_ring)
        add_edge(rhs, lhs, bond_order, aromatic, conjugated, same_ring)

    retained = list(retained_original_indices)
    for i, lhs in enumerate(retained):
        if mol.GetAtomWithIdx(lhs).GetAtomicNum() <= 1:
            continue
        for rhs in retained[i + 1 :]:
            if mol.GetAtomWithIdx(rhs).GetAtomicNum() <= 1:
                continue
            if (min(lhs, rhs), max(lhs, rhs)) in bonded_pairs:
                continue
            distance = float(torch.linalg.vector_norm(coords[rhs] - coords[lhs]).item())
            if distance <= 4.25:
                add_edge(lhs, rhs, 0.0, False, False, False)
                add_edge(rhs, lhs, 0.0, False, False, False)
    if not edge_pairs:
        raise RuntimeError("scaffold graph has no sparse edges")
    edge_index = torch.tensor(edge_pairs, dtype=torch.long).t().contiguous()
    return edge_index, torch.tensor(edge_attr, dtype=torch.float32), torch.tensor(active_mask, dtype=torch.bool)


def build_scaffold_graph(paths: TrainingPaths, ligand_sdf: Path | None = None) -> ScaffoldGraph:
    scaffold_sdf = Path(ligand_sdf) if ligand_sdf is not None else paths.ligand_sdf
    mol = read_first_mol(scaffold_sdf)
    if scaffold_sdf.name.startswith("ALENI-"):
        removed = removed_fragment_atoms(mol, paths.fragment_registry)
        exit_original_idx = boundary_exit_atom(mol, removed)
    else:
        removed = set()
        exit_original_idx = inferred_competitor_exit_atom(mol)
    retained_original_indices = [idx for idx in range(mol.GetNumAtoms()) if idx not in removed]
    old_to_new = {old_idx: new_idx for new_idx, old_idx in enumerate(retained_original_indices)}
    exit_node_idx = old_to_new[exit_original_idx]
    ca_coordinates = topology_ca_coordinates(paths.topology)
    phase_maps = load_phase_feature_maps(paths)
    conf = mol.GetConformer()
    x_base: list[list[float]] = []
    x_phase: list[list[list[float]]] = []
    zero_phase = [[0.0] * 8 for _ in PHASES]
    for old_idx in retained_original_indices:
        atom = mol.GetAtomWithIdx(old_idx)
        xyz = list(conf.GetAtomPosition(old_idx))
        residue_idx = nearest_residue_idx(xyz, ca_coordinates)
        x_base.append(atom_base_features(atom, atom_partial_charge(mol, old_idx), old_idx == exit_original_idx))
        x_phase.append(phase_maps.get(residue_idx, zero_phase))
    edge_index, edge_attr, active_mask = build_edges(mol, retained_original_indices, old_to_new, exit_original_idx)
    data = Data(
        x_base=torch.tensor(x_base, dtype=torch.float32),
        x_phase=torch.tensor(x_phase, dtype=torch.float32),
        edge_index=edge_index,
        edge_attr=edge_attr,
        active_dendrite_mask=active_mask,
        num_nodes=len(x_base),
    )
    return ScaffoldGraph(
        scaffold_id=scaffold_sdf.stem,
        ligand_sdf=scaffold_sdf,
        data=data,
        exit_node_idx=exit_node_idx,
        base_feature_dim=len(x_base[0]),
        phase_feature_dim=len(x_phase[0][0]),
        edge_feature_dim=int(edge_attr.shape[1]),
        retained_atom_count=len(retained_original_indices),
    )


def build_scaffold_pool(paths: TrainingPaths, scaffold_sdfs: Sequence[Path]) -> tuple[ScaffoldGraph, ...]:
    graphs = tuple(build_scaffold_graph(paths, path) for path in scaffold_sdfs)
    if not graphs:
        raise RuntimeError("empty scaffold pool")
    reference = graphs[0]
    for graph in graphs[1:]:
        if (
            graph.base_feature_dim != reference.base_feature_dim
            or graph.phase_feature_dim != reference.phase_feature_dim
            or graph.edge_feature_dim != reference.edge_feature_dim
        ):
            raise RuntimeError(
                "scaffold graph feature dimensions do not match: "
                f"{graph.scaffold_id} vs {reference.scaffold_id}"
            )
    print(
        "scaffold_pool_loaded "
        f"count={len(graphs)} scaffolds={','.join(graph.scaffold_id for graph in graphs)}",
        flush=True,
    )
    return graphs


def load_action_space(paths: TrainingPaths, embedding_dim: int) -> ActionSpace:
    anchors = normalize_anchor_table(pl.read_parquet(paths.anchors))
    survivors = (
        pl.read_parquet(paths.survivors)
        .sort("score", descending=True)
        .group_by("anchor_id")
        .agg(
            pl.col("score").first().alias("oracle_reward"),
            pl.col("canonical_smiles").first().alias("survivor_smiles"),
            pl.col("synthon_b_id").first().alias("synthon_b_id"),
            pl.col("product_id").first().alias("product_id"),
            pl.col("selected_dihedral_deg").first().alias("selected_dihedral_deg"),
            pl.col("survival_tier").first().alias("survival_tier"),
        )
    )
    table = anchors.join(survivors, on="anchor_id", how="left").with_columns(
        (
            (pl.col("generation_status") == "ok")
            & pl.col("oracle_reward").is_not_null()
            & pl.col("canonical_smiles").is_not_null()
        ).alias("action_valid")
    )
    table = (
        table.filter(pl.col("action_valid"))
        .sort("oracle_reward", descending=True)
        .unique(subset=["survivor_smiles"], keep="first")
    )
    valid_count = int(table.get_column("action_valid").sum())
    if valid_count < 100:
        raise RuntimeError(f"too few valid oracle-scored actions: {valid_count}")
    embeddings = anchor_embeddings_from_table(table, embedding_dim)
    valid_mask = torch.tensor(table.get_column("action_valid").to_list(), dtype=torch.bool)
    reward_values = torch.tensor(
        [float(value or 0.0) for value in table.get_column("oracle_reward").to_list()],
        dtype=torch.float32,
    )
    reward_logits = reward_values.clone()
    reward_logits = reward_logits.masked_fill(~valid_mask, torch.finfo(torch.float32).min)
    reward_targets = torch.softmax(reward_logits / 0.08, dim=0)
    return ActionSpace(table=table, anchor_embeddings=embeddings, valid_mask=valid_mask, reward_targets=reward_targets)


def normalize_anchor_table(anchors: pl.DataFrame) -> pl.DataFrame:
    """Normalize calibration and full Enamine synthon schemas for policy use."""

    table = anchors
    if "anchor_id" not in table.columns and "synthon_id" in table.columns:
        table = table.rename({"synthon_id": "anchor_id"})
    if "generation_status" not in table.columns:
        if "ingest_status" in table.columns:
            table = table.with_columns(pl.col("ingest_status").alias("generation_status"))
        else:
            table = table.with_columns(pl.lit("ok").alias("generation_status"))
    if "n_heavy_atoms" not in table.columns and "heavy_atom_count" in table.columns:
        table = table.with_columns(pl.col("heavy_atom_count").alias("n_heavy_atoms"))
    for column in ("steric_volume_A3", "bbox_x_A", "bbox_y_A", "bbox_z_A", "mmff_energy_kcal_mol"):
        if column not in table.columns:
            table = table.with_columns(pl.lit(0.0).alias(column))
    if "formal_charge" not in table.columns:
        table = table.with_columns(pl.lit(0.0).alias("formal_charge"))
    if "molecular_weight" not in table.columns:
        table = table.with_columns(pl.lit(0.0).alias("molecular_weight"))
    required = {"anchor_id", "canonical_smiles", "generation_status"}
    missing = required.difference(table.columns)
    if missing:
        raise RuntimeError(f"anchor table missing required columns: {sorted(missing)}")
    return table


def anchor_embeddings_from_table(table: pl.DataFrame, embedding_dim: int) -> Tensor:
    numeric_columns = [
        "n_heavy_atoms",
        "molecular_weight",
        "formal_charge",
        "steric_volume_A3",
        "bbox_x_A",
        "bbox_y_A",
        "bbox_z_A",
        "mmff_energy_kcal_mol",
    ]
    features: list[list[float]] = []
    for row in table.to_dicts():
        smiles = str(row["canonical_smiles"])
        mol = Chem.MolFromSmiles(smiles)
        descriptor_values = [numeric_value(row.get(column), 0.0) for column in numeric_columns]
        atoms = mol_atoms(mol) if mol is not None else []
        descriptor_values.extend(
            [
                float(mol.GetNumAtoms()) if mol is not None else 0.0,
                float(len([atom for atom in atoms if atom.GetIsAromatic()])),
                float(len([atom for atom in atoms if atom.IsInRing()])),
            ]
        )
        features.append(descriptor_values)
    raw = torch.tensor(features, dtype=torch.float32)
    mean = raw.mean(dim=0, keepdim=True)
    std = raw.std(dim=0, keepdim=True).clamp_min(1.0e-6)
    normalized = (raw - mean) / std
    repeats = math.ceil(embedding_dim / int(normalized.shape[1]))
    expanded = normalized.repeat(1, repeats)[:, :embedding_dim]
    return expanded


def clone_graph_batch(
    graphs: ScaffoldGraph | Sequence[ScaffoldGraph],
    batch_size: int,
    selected_graphs: Sequence[ScaffoldGraph] | None = None,
) -> tuple[Batch, Tensor]:
    pool = (graphs,) if isinstance(graphs, ScaffoldGraph) else tuple(graphs)
    if not pool:
        raise RuntimeError("cannot clone an empty scaffold pool")
    if selected_graphs is None:
        if len(pool) == 1:
            selected = [pool[0] for _ in range(batch_size)]
        else:
            indices = torch.randint(low=0, high=len(pool), size=(batch_size,), dtype=torch.long)
            selected = [pool[int(index)] for index in indices.tolist()]
    else:
        if len(selected_graphs) != batch_size:
            raise ValueError("selected_graphs length must equal batch_size")
        selected = list(selected_graphs)
    data_list = [graph.data.clone() for graph in selected]
    batch = Batch.from_data_list(data_list)
    ptr = cast(Tensor, batch.ptr)
    exit_offsets = torch.tensor([graph.exit_node_idx for graph in selected], dtype=torch.long)
    exit_indices = ptr[:-1].to(dtype=torch.long) + exit_offsets
    return batch, exit_indices.to(dtype=torch.long)


def forward_policy(
    model: FiberBundleGFlowNetPolicy,
    graph: ScaffoldGraph | Sequence[ScaffoldGraph],
    action_space: ActionSpace,
    batch_size: int,
    selected_graphs: Sequence[ScaffoldGraph] | None = None,
) -> tuple[Any, Batch, Tensor]:
    batch, exit_indices = clone_graph_batch(graph, batch_size, selected_graphs)
    forward_mask = action_space.valid_mask.unsqueeze(0).expand(batch_size, -1)
    backward_mask = torch.ones((batch_size, 1), dtype=torch.bool)
    output = model(
        x_base=cast(Tensor, batch.x_base),
        x_phase=cast(Tensor, batch.x_phase),
        edge_index=cast(Tensor, batch.edge_index),
        edge_attr=cast(Tensor, batch.edge_attr),
        active_dendrite_mask=cast(Tensor, batch.active_dendrite_mask),
        batch_index=cast(Tensor, batch.batch),
        exit_node_indices=exit_indices,
        forward_action_mask=forward_mask,
        backward_action_mask=backward_mask,
    )
    return output, batch, exit_indices


def proposals_for_actions(action_space: ActionSpace, actions: Sequence[int]) -> list[OracleProposal]:
    table_rows = action_space.table.to_dicts()
    proposals: list[OracleProposal] = []
    for batch_idx, action_idx in enumerate(actions):
        row = table_rows[int(action_idx)]
        survivor_smiles = row.get("survivor_smiles")
        canonical_smiles = survivor_smiles if isinstance(survivor_smiles, str) else row["canonical_smiles"]
        proposals.append(
            OracleProposal(
                anchor_id=str(row["anchor_id"]),
                canonical_smiles=str(canonical_smiles),
                trajectory_id=f"tb-{batch_idx:06d}",
            )
        )
    return proposals


def reaction_classes_for_actions(action_space: ActionSpace, actions: Sequence[int]) -> list[str]:
    table_rows = action_space.table.to_dicts()
    reaction_classes: list[str] = []
    for action_idx in actions:
        row = table_rows[int(action_idx)]
        for column in ("reaction_class", "reaction_id", "survival_tier", "assembly_mode"):
            value = row.get(column)
            if isinstance(value, str) and value:
                reaction_classes.append(value)
                break
        else:
            reaction_classes.append("unknown_reaction_class")
    return reaction_classes


def select_scaffold_graphs(
    scaffold_pool: Sequence[ScaffoldGraph],
    *,
    batch_size: int,
    router: MultiScaffoldEntropyRouter | None,
    active_variant: str,
) -> list[ScaffoldGraph]:
    if not scaffold_pool:
        raise RuntimeError("empty scaffold pool")
    by_name = {graph.scaffold_id: graph for graph in scaffold_pool}
    if router is None:
        if len(scaffold_pool) == 1:
            return [scaffold_pool[0] for _ in range(batch_size)]
        indices = torch.randint(low=0, high=len(scaffold_pool), size=(batch_size,), dtype=torch.long)
        return [scaffold_pool[int(index)] for index in indices.tolist()]
    names = router.sample_batch(batch_size, active_variant=active_variant)
    return [by_name[name] for name in names]


def graph_channel_counts(graphs: Sequence[ScaffoldGraph]) -> tuple[list[int], list[int]]:
    channel_a: list[int] = []
    channel_b: list[int] = []
    for graph in graphs:
        mask = cast(Tensor, graph.data.active_dendrite_mask)
        channel_a.append(int(mask.sum().item()))
        channel_b.append(max(int(mask.numel()), 1))
    return channel_a, channel_b


def update_router_from_oracle(
    router: MultiScaffoldEntropyRouter | None,
    *,
    selected_graphs: Sequence[ScaffoldGraph],
    proposals: Sequence[OracleProposal],
    rewards: Tensor,
    reaction_classes: Sequence[str],
    oracle_rows: pl.DataFrame,
    fiber_bundle: Tensor,
) -> None:
    if router is None:
        return
    row_dicts = cast(list[dict[str, object]], oracle_rows.to_dicts())
    lock_maps = lock_phase_maps_from_rows(row_dicts)
    phase_maps = phase_occupancy_from_fiber_bundle(fiber_bundle)
    channel_a, channel_b = graph_channel_counts(selected_graphs)
    by_scaffold: dict[str, list[int]] = {}
    for index, graph in enumerate(selected_graphs):
        by_scaffold.setdefault(graph.scaffold_id, []).append(index)
    for scaffold_id, indices in by_scaffold.items():
        router.update(
            scaffold_id,
            [proposals[index].canonical_smiles for index in indices],
            [float(rewards[index].item()) for index in indices],
            [reaction_classes[index] for index in indices],
            [numeric_value(row_dicts[index].get("pi_clash_lock", 0.0), 0.0) for index in indices],
            phase_occupancy_batch=[phase_maps[index] for index in indices],
            lock_clash_phase_batch=[lock_maps[index] for index in indices],
            channel_a_activations=[channel_a[index] for index in indices],
            channel_b_activations=[channel_b[index] for index in indices],
        )


def write_csv(path: Path, rows: Iterable[dict[str, Any]], fieldnames: Sequence[str]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def plot_csv(path: Path, x_key: str, y_key: str, output: Path, ylabel: str) -> None:
    df = pl.read_csv(path)
    plt.figure(figsize=(7, 4))
    plt.plot(df.get_column(x_key), df.get_column(y_key), marker="o", linewidth=1.5)
    plt.xlabel(x_key)
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(output, dpi=160)
    plt.close()


def cosine_warmup_lr(epoch: int, *, warmup_epochs: int, total_epochs: int, peak_lr: float, floor_lr: float) -> float:
    """Cosine decay with a linear warmup, using one-indexed epoch values."""

    if total_epochs <= 0:
        raise ValueError("total_epochs must be positive")
    if peak_lr <= 0.0 or floor_lr < 0.0:
        raise ValueError("learning rates must be non-negative and peak_lr must be positive")
    warmup = max(1, warmup_epochs)
    if epoch <= warmup:
        return floor_lr + (peak_lr - floor_lr) * (float(epoch) / float(warmup))
    decay_steps = max(1, total_epochs - warmup)
    progress = min(1.0, max(0.0, float(epoch - warmup) / float(decay_steps)))
    return floor_lr + 0.5 * (peak_lr - floor_lr) * (1.0 + math.cos(math.pi * progress))


def reward_entropy(rewards: Tensor, temperature: float) -> float:
    scaled = rewards.detach().float() / max(temperature, 1.0e-6)
    probabilities = torch.softmax(scaled, dim=0)
    entropy = -(probabilities * probabilities.clamp_min(1.0e-12).log()).sum()
    return float(entropy.item())


def compute_adaptive_temperature(rewards: Tensor, target_entropy: float = 2.0) -> float:
    """Choose a bounded temperature whose reward softmax entropy is near target."""

    if rewards.numel() < 2 or float(rewards.std(unbiased=False).item()) < 1.0e-8:
        return 1.0
    low = 0.1
    high = 20.0
    for _ in range(24):
        mid = 0.5 * (low + high)
        if reward_entropy(rewards, mid) < target_entropy:
            low = mid
        else:
            high = mid
    return max(0.1, min(20.0, high))


def compute_diversity_bonus(smiles: Sequence[str], seen_smiles_counts: Mapping[str, int], beta: float = 0.5) -> Tensor:
    values = [float(beta) / math.sqrt(float(seen_smiles_counts.get(value, 0) + 1)) for value in smiles]
    return torch.tensor(values, dtype=torch.float32)


def parameter_grad_norm(parameters: Sequence[torch.nn.Parameter]) -> float:
    total = 0.0
    for parameter in parameters:
        if parameter.grad is None:
            continue
        value = float(parameter.grad.detach().norm(2.0).item())
        total += value * value
    return math.sqrt(total)


def save_training_checkpoint(
    path: Path,
    *,
    model: FiberBundleGFlowNetPolicy,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    config: Mapping[str, object],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "config": dict(config),
        },
        tmp_path,
    )
    tmp_path.replace(path)


async def train() -> None:
    args = parse_args()
    torch.set_num_threads(max(1, int(args.torch_threads)))
    torch.set_num_interop_threads(max(1, min(4, int(args.torch_threads))))
    set_seed(int(args.seed))
    paths = TrainingPaths(
        ligand_sdf=Path(args.ligand_sdf),
        anchors=Path(args.anchors),
        survivors=Path(args.survivors),
        residue_phase=Path(args.residue_phase),
        interferometric=Path(args.interferometric),
        topology=Path(args.topology),
        fragment_registry=Path(args.fragment_registry),
        output_dir=Path(args.output_dir),
    )
    paths.output_dir.mkdir(parents=True, exist_ok=True)
    scaffold_sdfs = scaffold_paths_from_args(cast(Sequence[Path] | None, args.scaffold_pool), paths.ligand_sdf)
    scaffold_pool = build_scaffold_pool(paths, scaffold_sdfs)
    reference_graph = scaffold_pool[0]
    entropy_router_enabled = bool(args.entropy_router) and not bool(args.disable_entropy_router) and len(scaffold_pool) > 1
    entropy_router = (
        MultiScaffoldEntropyRouter(
            [graph.scaffold_id for graph in scaffold_pool],
            pgx_exclusions=MultiScaffoldEntropyRouter.load_pgx_exclusions(Path(args.phase3_pgx_manifest)),
        )
        if entropy_router_enabled
        else None
    )
    if entropy_router is not None:
        print(
            "scaffold_router_initialized "
            f"dstw_aligned={bool(args.dstw_aligned)} active_variant={args.active_variant} "
            f"scaffolds={','.join(graph.scaffold_id for graph in scaffold_pool)}",
            flush=True,
        )
    action_space = load_action_space(paths, int(args.embedding_dim))
    model = FiberBundleGFlowNetPolicy(
        base_feature_dim=reference_graph.base_feature_dim,
        phase_feature_dim=reference_graph.phase_feature_dim,
        edge_feature_dim=reference_graph.edge_feature_dim,
        anchor_embeddings=action_space.anchor_embeddings,
        hidden_dim=int(args.hidden_dim),
        embedding_dim=int(args.embedding_dim),
        learn_anchor_embeddings=True,
    )
    policy_parameters: list[torch.nn.Parameter] = []
    log_z_parameters: list[torch.nn.Parameter] = []
    for name, parameter in model.named_parameters():
        if name == "log_z":
            log_z_parameters.append(parameter)
        else:
            policy_parameters.append(parameter)
    optimizer = torch.optim.AdamW(
        [
            {"params": policy_parameters, "lr": float(args.learning_rate)},
            {"params": log_z_parameters, "lr": float(args.log_z_learning_rate)},
        ]
    )
    tb_loss = TrajectoryBalanceLoss()
    oracle = BatchedRustOracle(survivor_corpus=paths.survivors, max_batch_size=int(args.batch_size))

    config = {
        "architecture": "FiberBundleGFlowNetPolicy",
        "phase_labels": list(PHASE_LABELS),
        "x_phase_shape": [reference_graph.retained_atom_count, 5, reference_graph.phase_feature_dim],
        "base_feature_dim": reference_graph.base_feature_dim,
        "phase_feature_dim": reference_graph.phase_feature_dim,
        "edge_feature_dim": reference_graph.edge_feature_dim,
        "scaffold_pool_strategy": "dstw_entropy_router" if entropy_router is not None else "uniform_per_trajectory",
        "entropy_router_enabled": entropy_router is not None,
        "entropy_router_type": "DSTW_ALIGNED" if entropy_router is not None else "disabled",
        "active_variant": str(args.active_variant),
        "scaffold_pool": [
            {
                "scaffold_id": graph.scaffold_id,
                "ligand_sdf": graph.ligand_sdf.as_posix(),
                "retained_atom_count": graph.retained_atom_count,
                "exit_node_idx": graph.exit_node_idx,
            }
            for graph in scaffold_pool
        ],
        "valid_action_count": int(action_space.valid_mask.sum().item()),
        "batch_size": int(args.batch_size),
        "warm_start_epochs": int(args.warm_start_epochs),
        "tb_epochs": int(args.epochs),
        "learning_rate": float(args.learning_rate),
        "log_z_learning_rate": float(args.log_z_learning_rate),
        "learning_rate_floor": float(args.learning_rate_floor),
        "log_z_learning_rate_floor": float(args.log_z_learning_rate_floor),
        "policy_warmup_epochs": int(args.policy_warmup_epochs),
        "log_z_warmup_epochs": int(args.log_z_warmup_epochs),
        "reward_scale": float(args.reward_scale),
        "policy_aux_weight": float(args.policy_aux_weight),
        "diversity_beta": float(args.diversity_beta),
        "max_trajectory_steps": int(args.max_trajectory_steps),
        "reward_version": str(args.reward_version),
        "lock_directional_bias_alpha": float(args.lock_directional_bias_alpha),
        "lock_reaching_synthon_boost": float(args.lock_reaching_synthon_boost),
        "rust_oracle_reward_authority": True,
        "uses_pyg_batch": True,
        "dense_adjacency": False,
        "learn_anchor_embeddings": True,
        "input_sha256": {
            "ligand_sdf": sha256_file(paths.ligand_sdf),
            "scaffold_pool": {path.as_posix(): sha256_file(path) for path in scaffold_sdfs},
            "anchors": sha256_file(paths.anchors),
            "survivors": sha256_file(paths.survivors),
            "residue_phase": sha256_file(paths.residue_phase),
            "interferometric": sha256_file(paths.interferometric),
        },
        "generated_at_utc": datetime.now(UTC).isoformat(),
    }
    (paths.output_dir / "gflownet_training_config.json").write_text(json.dumps(config, indent=2))
    (paths.output_dir / "fiber_gnn_architecture_summary.md").write_text(
        "# Fiber-Bundle GFlowNet Architecture\n\n"
        "- Base space: PyG-packed scaffold atom graph with sparse `edge_index`.\n"
        "- Fiber space: explicit `[N_atoms, 5, D]` CCNS phase tensor.\n"
        "- Phase routing: GRU plus 1D convolution over Cold Hold, Ramp Up, Warm Hold, Ramp Down, Cold Return.\n"
        "- Orthogonal routing: base graph messages and within-atom fiber messages remain separate before gated fusion.\n"
        "- Action policy: exit-vector-conditioned dot-product attention over calibration anchor embeddings.\n"
        "- Trajectory horizon: max 5 synthetic steps with early termination retained.\n"
        "- Bias scoring: tripartite observed/derived/projected fields from the Rust oracle.\n"
        "- Scaffold routing: DSTW entropy router balances phase coverage, hysteresis, reaction entropy, novelty, and lock signal when multiple scaffolds are available.\n"
        "- Reward authority: Batched Rust Oracle; Python never computes terminal rewards.\n"
    )

    warm_rows: list[dict[str, Any]] = []
    for epoch in range(1, int(args.warm_start_epochs) + 1):
        model.train()
        selected_graphs = select_scaffold_graphs(
            scaffold_pool,
            batch_size=int(args.batch_size),
            router=entropy_router,
            active_variant=str(args.active_variant),
        )
        output, _, _ = forward_policy(
            model,
            scaffold_pool,
            action_space,
            int(args.batch_size),
            selected_graphs=selected_graphs,
        )
        log_probs = output.forward_log_probs
        target = action_space.reward_targets.unsqueeze(0).expand_as(log_probs)
        loss = -(target * log_probs).sum(dim=1).mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()
        warm_rows.append({"epoch": epoch, "warm_start_kl_loss": float(loss.item())})

    loss_rows: list[dict[str, Any]] = []
    reward_rows: list[dict[str, Any]] = []
    entropy_rows: list[dict[str, Any]] = []
    latency_rows: list[dict[str, Any]] = []
    generated_smiles: set[str] = set()
    seen_smiles_counts: dict[str, int] = {}
    sampled_anchor_counts: dict[str, int] = {}
    unique_growth_history: list[int] = []
    reward_std_history: list[float] = []
    plateau_logged = False
    top_reward_seen = 0.0
    final_epoch = 0
    checkpoint_dir = REPO_ROOT / ".scratch/checkpoints"
    for epoch in range(1, int(args.epochs) + 1):
        final_epoch = epoch
        lr_policy = cosine_warmup_lr(
            epoch,
            warmup_epochs=int(args.policy_warmup_epochs),
            total_epochs=int(args.epochs),
            peak_lr=float(args.learning_rate),
            floor_lr=float(args.learning_rate_floor),
        )
        lr_log_z = cosine_warmup_lr(
            epoch,
            warmup_epochs=int(args.log_z_warmup_epochs),
            total_epochs=int(args.epochs),
            peak_lr=float(args.log_z_learning_rate),
            floor_lr=float(args.log_z_learning_rate_floor),
        )
        optimizer.param_groups[0]["lr"] = lr_policy
        optimizer.param_groups[1]["lr"] = lr_log_z
        print(f"tb_epoch_start epoch={epoch}/{int(args.epochs)}", flush=True)
        model.train()
        selected_graphs = select_scaffold_graphs(
            scaffold_pool,
            batch_size=int(args.batch_size),
            router=entropy_router,
            active_variant=str(args.active_variant),
        )
        output, _, _ = forward_policy(
            model,
            scaffold_pool,
            action_space,
            int(args.batch_size),
            selected_graphs=selected_graphs,
        )
        progress = 0.0 if int(args.epochs) <= 1 else float(epoch - 1) / float(int(args.epochs) - 1)
        sampling_temperature = 5.0 - (4.90 * progress)
        sampling_logits = output.forward_log_probs[0].detach().clone()
        sampling_logits = sampling_logits.masked_fill(~action_space.valid_mask, -torch.inf)
        sampling_probs = torch.softmax(sampling_logits / max(sampling_temperature, 1.0e-3), dim=0)
        if (not bool(torch.isfinite(sampling_probs).all())) or float(sampling_probs.sum().item()) <= 0.0:
            sampling_probs = action_space.valid_mask.to(dtype=torch.float32)
            sampling_probs = sampling_probs / sampling_probs.sum().clamp_min(1.0e-12)
        actions_tensor = torch.multinomial(sampling_probs, int(args.batch_size), replacement=False)
        actions = [int(value) for value in actions_tensor.tolist()]
        proposals = proposals_for_actions(action_space, actions)
        reaction_classes = reaction_classes_for_actions(action_space, actions)
        oracle_result = await oracle.score_batch(proposals)
        rewards = oracle_result.rewards
        oracle_row_dicts = cast(list[dict[str, object]], oracle_result.rows.to_dicts())
        tripartite_scores = [compute_tripartite_bias(row) for row in oracle_row_dicts]
        lock_geo_positive = sum(1 for score in tripartite_scores if score.lock_geometry_score > 0.0)
        lock_proj_gt_05 = sum(1 for score in tripartite_scores if score.bias_projection_score > 0.5)
        update_router_from_oracle(
            entropy_router,
            selected_graphs=selected_graphs,
            proposals=proposals,
            rewards=rewards,
            reaction_classes=reaction_classes,
            oracle_rows=oracle_result.rows,
            fiber_bundle=oracle_result.fiber_bundle,
        )
        scaled_rewards = rewards * float(args.reward_scale)
        current_temperature = compute_adaptive_temperature(scaled_rewards, target_entropy=2.0)
        tempered_rewards = (scaled_rewards / current_temperature).clamp_min(1.0e-8)
        proposal_smiles = [proposal.canonical_smiles for proposal in proposals]
        diversity_bonus = compute_diversity_bonus(
            proposal_smiles,
            seen_smiles_counts,
            beta=float(args.diversity_beta),
        )
        effective_rewards = (tempered_rewards + diversity_bonus).clamp_min(1.0e-8)
        row_indices = torch.arange(int(args.batch_size), dtype=torch.long)
        forward_selected = output.forward_log_probs[row_indices, actions_tensor].unsqueeze(1)
        backward_selected = output.backward_log_probs[:, 0].unsqueeze(1)
        tb = tb_loss(
            output.log_z,
            forward_selected,
            backward_selected,
            effective_rewards,
        )
        target = action_space.reward_targets.unsqueeze(0).expand_as(output.forward_log_probs)
        policy_aux_loss = -(target * output.forward_log_probs).sum(dim=1).mean()
        total_loss = tb.loss + float(args.policy_aux_weight) * policy_aux_loss
        optimizer.zero_grad(set_to_none=True)
        total_loss.backward()
        gradient_norm_policy = parameter_grad_norm(policy_parameters)
        gradient_norm_log_z = parameter_grad_norm(log_z_parameters)
        gradient_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0).item())
        if not math.isfinite(gradient_norm):
            raise RuntimeError("non-finite gradient norm")
        optimizer.step()

        reward_mean = float(rewards.mean().item())
        reward_std = float(rewards.std(unbiased=False).item())
        reward_p95 = float(torch.quantile(rewards, 0.95).item())
        reward_max = float(rewards.max().item())
        scaled_reward_mean = float(scaled_rewards.mean().item())
        top_reward_seen = max(top_reward_seen, reward_max)
        previous_unique_count = len(generated_smiles)
        for proposal in proposals:
            generated_smiles.add(proposal.canonical_smiles)
            seen_smiles_counts[proposal.canonical_smiles] = seen_smiles_counts.get(proposal.canonical_smiles, 0) + 1
            sampled_anchor_counts[proposal.anchor_id] = sampled_anchor_counts.get(proposal.anchor_id, 0) + 1
        n_new_unique = len(generated_smiles) - previous_unique_count
        unique_growth_history.append(n_new_unique)
        reward_std_history.append(reward_std)
        dot_count = sum(1 for proposal in proposals if "." in proposal.canonical_smiles)
        if dot_count > 0:
            save_training_checkpoint(
                checkpoint_dir / f"gflownet_epoch_{epoch:04d}_topology_regression.pt",
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                config=config,
            )
            print(f"training_convergence type=topology_regression epoch={epoch}", flush=True)
            raise RuntimeError("TOPOLOGY_REGRESSION_ABORT: dot-disconnected SMILES generated")
        top_index = int(torch.argmax(rewards).item())
        top1_smiles = proposals[top_index].canonical_smiles
        top1_reward = float(rewards[top_index].item())
        entropy = float((-(output.forward_probs[0] * output.forward_log_probs[0]).sum()).item())
        loss_rows.append(
            {
                "epoch": epoch,
                "tb_loss": float(tb.loss.item()),
                "gradient_norm": gradient_norm,
                "gradient_norm_policy": gradient_norm_policy,
                "gradient_norm_logZ": gradient_norm_log_z,
            }
        )
        reward_rows.append(
            {
                "epoch": epoch,
                "reward_mean": reward_mean,
                "reward_std": reward_std,
                "reward_max": reward_max,
                "scaled_reward_mean": scaled_reward_mean,
                "reward_p95": reward_p95,
                "top_reward_seen": top_reward_seen,
                "unique_smiles_generated": len(generated_smiles),
                "diversity_bonus_mean": float(diversity_bonus.mean().item()),
                "temperature": current_temperature,
                "lock_geo_positive": lock_geo_positive,
                "lock_proj_gt_05": lock_proj_gt_05,
            }
        )
        entropy_rows.append({"epoch": epoch, "trajectory_entropy": entropy})
        latency = telemetry_to_dict(oracle_result.telemetry)
        latency_rows.append({"epoch": epoch, **latency})
        if not math.isfinite(float(tb.loss.item())):
            raise RuntimeError("NaN/Inf TB loss")
        telemetry = {
            "epoch": epoch,
            "tb_loss": f"{float(tb.loss.item()):.6f}",
            "reward_mean": f"{reward_mean:.6f}",
            "reward_std": f"{reward_std:.6f}",
            "reward_max": f"{reward_max:.6f}",
            "scaled_reward_mean": f"{scaled_reward_mean:.6f}",
            "temperature": f"{current_temperature:.6f}",
            "lr_policy": f"{lr_policy:.8f}",
            "lr_logZ": f"{lr_log_z:.8f}",
            "unique_smiles_cumulative": len(generated_smiles),
            "unique_smiles_this_epoch": n_new_unique,
            "diversity_bonus_mean": f"{float(diversity_bonus.mean().item()):.6f}",
            "dot_smiles_count": dot_count,
            "logZ": f"{float(output.log_z.item()):.6f}",
            "top1_smiles": top1_smiles,
            "top1_reward": f"{top1_reward:.6f}",
            "gradient_norm_policy": f"{gradient_norm_policy:.6f}",
            "gradient_norm_logZ": f"{gradient_norm_log_z:.6f}",
            "oracle_latency_ms": f"{oracle_result.telemetry.oracle_latency_ms:.3f}",
            "lock_geo_positive": f"{lock_geo_positive}/{int(args.batch_size)}",
            "lock_proj_gt_05": f"{lock_proj_gt_05}/{int(args.batch_size)}",
        }
        print("tb_epoch_complete " + " ".join(f"{key}={value}" for key, value in telemetry.items()), flush=True)
        print(
            "lock_positive_rate "
            f"epoch={epoch} lock_geo_positive={lock_geo_positive}/{int(args.batch_size)} "
            f"lock_proj_gt_05={lock_proj_gt_05}/{int(args.batch_size)}",
            flush=True,
        )
        interval = max(1, int(args.router_telemetry_interval))
        if entropy_router is not None and epoch % interval == 0:
            for line in entropy_router.telemetry_lines(active_variant=str(args.active_variant)):
                print(line, flush=True)
        if epoch % 50 == 0:
            save_training_checkpoint(
                checkpoint_dir / f"gflownet_epoch_{epoch:04d}.pt",
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                config=config,
            )
        if epoch > 100 and len(loss_rows) >= 51:
            current_loss = float(loss_rows[-1]["tb_loss"])
            previous_loss = float(loss_rows[-51]["tb_loss"])
            improvement = (previous_loss - current_loss) / max(abs(previous_loss), 1.0e-8)
            if improvement <= 0.01 and not plateau_logged:
                save_training_checkpoint(
                    checkpoint_dir / f"gflownet_epoch_{epoch:04d}_plateau.pt",
                    model=model,
                    optimizer=optimizer,
                    epoch=epoch,
                    config=config,
                )
                print(f"training_convergence type=plateau epoch={epoch}", flush=True)
                plateau_logged = True
            if sum(unique_growth_history[-50:]) < 100:
                save_training_checkpoint(
                    checkpoint_dir / f"gflownet_epoch_{epoch:04d}_terminal.pt",
                    model=model,
                    optimizer=optimizer,
                    epoch=epoch,
                    config=config,
                )
                print(f"training_convergence type=terminal epoch={epoch} reason=unique_exhausted", flush=True)
                break
        if len(reward_std_history) >= 20 and all(value < 0.01 for value in reward_std_history[-20:]):
            save_training_checkpoint(
                checkpoint_dir / f"gflownet_epoch_{epoch:04d}_terminal.pt",
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                config=config,
            )
            print(f"training_convergence type=terminal epoch={epoch} reason=reward_collapse", flush=True)
            break

    model_path = paths.output_dir / "gflownet_policy_v1.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": config,
            "anchor_table": action_space.table.select(["anchor_id", "canonical_smiles", "action_valid"]).to_dicts(),
        },
        model_path,
    )
    write_csv(paths.output_dir / "warm_start_loss.csv", warm_rows, ["epoch", "warm_start_kl_loss"])
    write_csv(
        paths.output_dir / "epoch_loss.csv",
        loss_rows,
        ["epoch", "tb_loss", "gradient_norm", "gradient_norm_policy", "gradient_norm_logZ"],
    )
    write_csv(
        paths.output_dir / "reward_progression.csv",
        reward_rows,
        [
            "epoch",
            "reward_mean",
            "reward_std",
            "reward_max",
            "scaled_reward_mean",
            "reward_p95",
            "top_reward_seen",
            "unique_smiles_generated",
            "diversity_bonus_mean",
            "temperature",
            "lock_geo_positive",
            "lock_proj_gt_05",
        ],
    )
    write_csv(paths.output_dir / "trajectory_entropy.csv", entropy_rows, ["epoch", "trajectory_entropy"])
    write_csv(
        paths.output_dir / "oracle_latency.csv",
        latency_rows,
        [
            "epoch",
            "oracle_batch_size",
            "oracle_latency_ms",
            "rust_scoring_time_ms",
            "parquet_write_ms",
            "parquet_read_ms",
            "reward_mean",
            "reward_std",
            "invalid_reward_count",
            "duplicate_smiles_count",
        ],
    )
    plot_csv(paths.output_dir / "epoch_loss.csv", "epoch", "tb_loss", paths.output_dir / "epoch_loss.png", "TB loss")
    plot_csv(
        paths.output_dir / "reward_progression.csv",
        "epoch",
        "reward_mean",
        paths.output_dir / "reward_progression.png",
        "reward mean",
    )
    plot_csv(
        paths.output_dir / "trajectory_entropy.csv",
        "epoch",
        "trajectory_entropy",
        paths.output_dir / "trajectory_entropy.png",
        "trajectory entropy",
    )

    first_loss = float(loss_rows[0]["tb_loss"])
    final_loss = float(loss_rows[-1]["tb_loss"])
    first_reward = float(reward_rows[0]["reward_mean"])
    final_reward = float(reward_rows[-1]["reward_mean"])
    final_entropy = float(entropy_rows[-1]["trajectory_entropy"])
    total_samples = sum(sampled_anchor_counts.values())
    top_anchor_share = max(sampled_anchor_counts.values()) / float(total_samples) if total_samples else 1.0
    dot_smiles_total = sum(1 for smiles in generated_smiles if "." in smiles)
    validation = {
        "validation_status": "PASS"
        if final_loss < first_loss
        and final_reward >= first_reward
        and final_entropy > 0.1
        and len(generated_smiles) > 2000
        and dot_smiles_total == 0
        and top_anchor_share < 0.05
        else "WARN_REVIEW",
        "epoch_1_tb_loss": first_loss,
        f"epoch_{final_epoch}_tb_loss": final_loss,
        "reward_mean_epoch_1": first_reward,
        f"reward_mean_epoch_{final_epoch}": final_reward,
        "top_reward_seen": top_reward_seen,
        "unique_smiles_generated": len(generated_smiles),
        "dot_smiles_count": dot_smiles_total,
        "mode_collapse_metric_top_anchor_share": top_anchor_share,
        "completed_epochs": final_epoch,
        "rust_oracle_reward_authority": True,
        "pyg_batch_used": True,
        "x_phase_preserved_shape": [
            reference_graph.retained_atom_count,
            5,
            reference_graph.phase_feature_dim,
        ],
        "scaffold_pool_count": len(scaffold_pool),
        "scaffold_pool_ids": [graph.scaffold_id for graph in scaffold_pool],
        "entropy_router_enabled": entropy_router is not None,
        "entropy_router_telemetry": entropy_router.get_telemetry(str(args.active_variant)) if entropy_router else {},
        "model_path": str(model_path),
    }
    (paths.output_dir / "gflownet_learning_validation.json").write_text(json.dumps(validation, indent=2))
    (paths.output_dir / "gflownet_learning_validation.md").write_text(
        "# GFlowNet Learning Validation\n\n"
        f"- Validation status: `{validation['validation_status']}`\n"
        f"- Epoch 1 TB loss: `{first_loss:.6f}`\n"
        f"- Epoch {final_epoch} TB loss: `{final_loss:.6f}`\n"
        f"- Reward mean progression: `{first_reward:.6f}` -> `{final_reward:.6f}`\n"
        f"- Top reward seen: `{top_reward_seen:.6f}`\n"
        f"- Unique generated SMILES: `{len(generated_smiles)}`\n"
        f"- Dot-disconnected SMILES count: `{dot_smiles_total}`\n"
        f"- Top anchor share: `{top_anchor_share:.4f}`\n"
        "- Reward authority: Rust `oracle_scorer` via `BatchedRustOracle`.\n"
    )
    print(
        "gflownet_training_complete "
        f"epoch1_tb_loss={first_loss:.6f} epoch{final_epoch}_tb_loss={final_loss:.6f} "
        f"reward_mean={first_reward:.6f}->{final_reward:.6f} "
        f"top_reward_seen={top_reward_seen:.6f} unique_smiles={len(generated_smiles)} "
        f"dot_smiles_count={dot_smiles_total} top_anchor_share={top_anchor_share:.6f} "
        f"validation_status={validation['validation_status']} model={model_path}"
    )


if __name__ == "__main__":
    asyncio.run(train())
