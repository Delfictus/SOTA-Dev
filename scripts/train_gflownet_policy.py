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
import subprocess
import sys
import tempfile
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence, TextIO, cast

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import torch
from rdkit import Chem
from torch import Tensor
from torch.nn import functional as F
from torch_geometric.data import Batch, Data  # type: ignore[import-untyped]

from prism_dstw.hierarchical_bayes.gflownet_policy import (
    FiberBundleGFlowNetPolicy,
    FieldConditionedDualChannelGFlowNetPolicy,
)
from prism_dstw.hierarchical_bayes.trajectory_balance import TrajectoryBalanceLoss
from prism_dstw.orchestration.multi_scaffold_entropy_router import (
    MultiScaffoldEntropyRouter,
    lock_phase_maps_from_rows,
    phase_occupancy_from_fiber_bundle,
)
from prism_dstw.orchestration.field_conditioner import FieldConditioner
from prism_dstw.orchestration.rust_reward_oracle import (
    LiveSignalGridOracle,
    OracleProposal,
    SurvivorCorpusOracle,
    telemetry_to_dict,
)
from prism_dstw.scoring.exit_atom_ray_cast import ExitAtomRayCaster
from prism_dstw.scoring.product_fiber_lookup import SignalGridFiberLookup, ThermodynamicFieldStack
from prism_dstw.scoring.tripartite_bias_scorer import (
    TripartiteBiasScore,
    compute_reward_v2,
    compute_tripartite_bias,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
TRACK_A_DIR = REPO_ROOT / "campaigns/glp1r_aleniglipron/track_a_generative"
N80_DIR = REPO_ROOT / "campaigns/glp1r_aleniglipron/integrated_spike_events/n80_full_scale"
DEFAULT_LIGAND_SDF = TRACK_A_DIR / "ALENI-PARENT_6XOX_frame_o3a_relaxed.sdf"
DEFAULT_ORFOR_SDF = TRACK_A_DIR / "conformers/ORFOR-PARENT_6XOX_o3a.sdf"
DEFAULT_DANU_SDF = TRACK_A_DIR / "conformers/DANU-PARENT_6XOX_o3a.sdf"
DEFAULT_SCAFFOLD_POOL = (DEFAULT_LIGAND_SDF, DEFAULT_ORFOR_SDF, DEFAULT_DANU_SDF)
DEFAULT_ANCHORS = TRACK_A_DIR / "enamine_115k_synthons_3d.parquet"
DEFAULT_SURVIVORS = TRACK_A_DIR / "vspace_survivors_full_scale.parquet"
DEFAULT_POPULATION_CONSENSUS_SURVIVORS = TRACK_A_DIR / "vspace_survivors_population_consensus_action_corpus.parquet"
DEFAULT_RESIDUE_PHASE = N80_DIR / "residue_phase_tensor.parquet"
DEFAULT_INTERFEROMETRIC = N80_DIR / "interferometric_differential.parquet"
DEFAULT_TOPOLOGY = REPO_ROOT / "04_TOPOLOGIES/glp1r_6XOX_HOLO_ALENI.topology.json"
DEFAULT_FRAGMENT_REGISTRY = (
    REPO_ROOT / "campaigns/glp1r_aleniglipron/track_0_manual_emulation/aleniglipron_brics_fragment_registry.json"
)
DEFAULT_PHASE3_PGX_MANIFEST = REPO_ROOT / "campaigns/glp1r_aleniglipron/Phase3_PGx_Exclusion_Manifest.json"
DEFAULT_GRID_MAPPING = REPO_ROOT / "campaigns/glp1r_aleniglipron/track_0_manual_emulation/grid_coordinate_mapping.json"
DEFAULT_POPULATION_CONSENSUS_GRID = TRACK_A_DIR / "signal_grid_population_consensus.parquet"
DEFAULT_SHEAR_STRESS = N80_DIR / "shear_stress_field.parquet"
DEFAULT_HYSTERESIS_TENSOR = N80_DIR / "hysteresis_tensor.parquet"
DEFAULT_TRANSLATION_PATHWAY = N80_DIR / "translation_pathway_nodes.parquet"
DEFAULT_CROSS_SPECIES = TRACK_A_DIR / "population_pgx/source/GLP1R_cross_species_conservation.csv"
DEFAULT_OUTPUT_DIR = TRACK_A_DIR
PHASES = ("cold_hold", "ramp_up", "warm_hold", "ramp_down", "cold_return")
PHASE_LABELS = ("Cold Hold", "Ramp Up", "Warm Hold", "Ramp Down", "Cold Return")
FieldLookup = SignalGridFiberLookup | ThermodynamicFieldStack


def resolve_survivor_corpus_for_reward(
    *,
    requested_survivors: Path,
    reward_version: str,
    signal_grid: Path | None,
) -> Path:
    """Select the survivor corpus that matches the requested reward semantics.

    The WT/full-scale corpus is still the default for legacy v1/v2 runs. For
    population-consensus and full-field training, silently using that default
    erases the consensus component even when the signal grid is the consensus
    grid. When the caller leaves ``--survivors`` at the default, switch to the
    consensus action corpus if it is present.
    """

    requested = Path(requested_survivors)
    version = reward_version.strip().lower()
    signal_grid_name = "" if signal_grid is None else Path(signal_grid).name.lower()
    default_like = (
        requested == DEFAULT_SURVIVORS
        or requested.name == DEFAULT_SURVIVORS.name
    )
    consensus_reward = (
        version.startswith("v3")
        or version.startswith("v4")
        or "consensus" in signal_grid_name
    )
    if default_like and consensus_reward and DEFAULT_POPULATION_CONSENSUS_SURVIVORS.exists():
        return DEFAULT_POPULATION_CONSENSUS_SURVIVORS
    return requested


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
    exit_ray_adjustments: Tensor | None = None
    action_phase_features: Tensor | None = None
    action_base_features: Tensor | None = None
    action_atom_features: Tensor | None = None
    action_atom_mask: Tensor | None = None
    action_rows: tuple[dict[str, object], ...] = tuple()


@dataclass(frozen=True)
class AssemblyHistoryStep:
    """One real growth operation in a variable-length trajectory."""

    step: int
    synthon_id: str
    exit_atom_idx: int
    action_idx: int
    node_count: int


@dataclass(frozen=True)
class AssembledState:
    """Current product graph state fed back into the policy at the next step."""

    scaffold: ScaffoldGraph
    data: Data
    exit_node_idx: int
    history: tuple[AssemblyHistoryStep, ...]
    last_growth_action_idx: int | None
    canonical_smiles: str
    anchor_id: str
    coordinates_json: str
    score_atom_offset: int
    survivor_lookup_smiles: str | None


@dataclass(frozen=True)
class RustAssemblyProduct:
    """Product topology and coordinates returned by the Rust Z-matrix assembler."""

    fragment_coordinates: Tensor
    product_coordinates: Tensor
    product_bonds: tuple[tuple[int, int], ...]
    assembly_mode: str


@dataclass(frozen=True)
class MultiStepRollout:
    """Policy samples and product states for a multi-step GFlowNet trajectory."""

    forward_outputs: tuple[Any, ...]
    terminal_states: tuple[AssembledState, ...]
    terminal_actions: Tensor
    history_actions: Tensor
    forward_log_probs: Tensor
    backward_log_probs: Tensor
    trajectory_lengths: Tensor
    trajectory_mask: Tensor
    assembly_histories: tuple[tuple[AssemblyHistoryStep, ...], ...]
    assembled_state_node_count_mean: float
    node_count_events: tuple[tuple[int, int, int, int], ...]
    field_conditioner_call_counts: tuple[int, ...]
    field_conditioner_shape_events: tuple[tuple[int, int, int, int], ...]


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
    parser.add_argument("--dual-channel", action="store_true", default=False)
    parser.add_argument(
        "--rf-mode",
        choices=("gain", "hard_zero", "soft"),
        default="gain",
        help="Resonate-and-fire edge weighting for the single-route fallback policy.",
    )
    parser.add_argument(
        "--oracle-mode",
        choices=("survivor_lookup", "live_signal_grid"),
        default="survivor_lookup",
        help="Explicit oracle contract. Use live_signal_grid to score proposal coordinates against the signal grid.",
    )
    parser.add_argument(
        "--live-scoring",
        action="store_true",
        default=False,
        help="Shortcut for --oracle-mode live_signal_grid.",
    )
    parser.add_argument("--lock-directional-bias-alpha", type=float, default=2.0)
    parser.add_argument("--lock-reaching-synthon-boost", type=float, default=2.0)
    parser.add_argument("--lock-geo-intrinsic-bonus", type=float, default=0.0)
    parser.add_argument(
        "--consensus-bonus-weight",
        type=float,
        default=0.0,
        help="Recorded weight for population-consensus survivor-corpus shaping.",
    )
    parser.add_argument("--lock-mask", type=Path, default=TRACK_A_DIR / "lock_region_mask.json")
    parser.add_argument("--synthon-parquet", type=Path, default=None)
    parser.add_argument("--signal-grid", type=Path, default=None)
    parser.add_argument("--grid-coordinate-mapping", type=Path, default=DEFAULT_GRID_MAPPING)
    parser.add_argument(
        "--field-stack-version",
        choices=("v1", "v2"),
        default="v1",
        help="v1 uses [5,8]/12D features; v2 uses the full [5,12]/13D thermodynamic field stack.",
    )
    parser.add_argument("--disable-exit-ray-masks", action="store_true", default=False)
    parser.add_argument("--shear-stress", type=Path, default=None)
    parser.add_argument("--hysteresis-tensor", type=Path, default=None)
    parser.add_argument("--translation-pathway", type=Path, default=None)
    parser.add_argument("--cross-species", type=Path, default=None)
    parser.add_argument("--checkpoint-dir", type=Path, default=REPO_ROOT / ".scratch/checkpoints")
    parser.add_argument("--checkpoint-interval", type=int, default=50)
    parser.add_argument("--output-policy", type=Path, default=TRACK_A_DIR / "gflownet_policy_v1.pt")
    parser.add_argument("--telemetry-log", type=Path, default=None)
    parser.add_argument("--resume", type=Path, default=None)
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
    parsed: float
    if isinstance(value, (int, float)):
        parsed = float(value)
    else:
        try:
            parsed = float(str(value))
        except ValueError:
            return default
    return parsed if math.isfinite(parsed) else default


def sample_actions_per_row(
    log_probs: Tensor,
    valid_mask: Tensor,
    temperature: float,
    *,
    require_unique: bool = True,
    generator: torch.Generator | None = None,
) -> Tensor:
    """Sample one action from each row's own action distribution."""

    if log_probs.ndim != 2:
        raise ValueError("log_probs must have shape [batch_size, num_actions]")
    batch_size, num_actions = int(log_probs.shape[0]), int(log_probs.shape[1])
    if batch_size == 0:
        raise ValueError("log_probs must contain at least one batch row")
    if valid_mask.ndim == 1:
        if int(valid_mask.shape[0]) != num_actions:
            raise ValueError("valid_mask length must match num_actions")
        mask = valid_mask.unsqueeze(0).expand(batch_size, num_actions)
    elif valid_mask.ndim == 2:
        if tuple(valid_mask.shape) != tuple(log_probs.shape):
            raise ValueError("2D valid_mask must match log_probs shape")
        mask = valid_mask
    else:
        raise ValueError("valid_mask must be 1D or 2D")
    if bool((mask.sum(dim=1) == 0).any()):
        raise ValueError("valid_mask leaves at least one batch row with zero valid actions")

    masked_logits = log_probs.detach().clone()
    masked_logits = masked_logits.masked_fill(~mask, -torch.inf)
    probs = torch.softmax(masked_logits / max(temperature, 1.0e-3), dim=1)
    invalid_rows = (~torch.isfinite(probs).all(dim=1)) | (probs.sum(dim=1) <= 0.0)
    if bool(invalid_rows.any()):
        fallback = mask.to(dtype=torch.float32)
        fallback = fallback / fallback.sum(dim=1, keepdim=True).clamp_min(1.0e-12)
        probs[invalid_rows] = fallback[invalid_rows]
    actions = torch.multinomial(probs, num_samples=1, replacement=True, generator=generator).squeeze(1)
    if not require_unique:
        return actions

    used: set[int] = set()
    repaired: list[int] = []
    for row_idx, raw_action in enumerate(actions.tolist()):
        action = int(raw_action)
        if action in used:
            row_probs = probs[row_idx].clone()
            for used_action in used:
                row_probs[used_action] = 0.0
            row_sum = row_probs.sum()
            if float(row_sum.item()) > 0.0:
                row_probs = row_probs / row_sum
                action = int(torch.multinomial(row_probs, num_samples=1, replacement=True, generator=generator).item())
        repaired.append(action)
        used.add(action)
    return torch.tensor(repaired, dtype=torch.long, device=log_probs.device)


def selected_forward_log_probs(log_probs: Tensor, actions: Tensor) -> Tensor:
    """Gather log-probs from the same row that produced each action."""

    if log_probs.ndim != 2:
        raise ValueError("log_probs must have shape [batch_size, num_actions]")
    if actions.ndim != 1 or int(actions.shape[0]) != int(log_probs.shape[0]):
        raise ValueError("actions must have shape [batch_size]")
    return log_probs.gather(1, actions.unsqueeze(1)).squeeze(1)


def reward_targets_for_policy_output(action_space: ActionSpace, log_probs: Tensor) -> Tensor:
    """Expand supervised warm-start targets, appending a zero-probability STOP lane when present."""

    if log_probs.ndim != 2:
        raise ValueError("log_probs must have shape [batch_size, num_actions]")
    target = action_space.reward_targets
    action_count = int(action_space.valid_mask.shape[0])
    if int(log_probs.shape[1]) == action_count:
        return target.unsqueeze(0).expand_as(log_probs)
    if int(log_probs.shape[1]) == action_count + 1:
        target_with_stop = torch.cat([target, target.new_zeros(1)], dim=0)
        return target_with_stop.unsqueeze(0).expand_as(log_probs)
    raise ValueError("policy output action dimension does not match action space")


def compute_effective_reward_tensor(
    *,
    reward_version: str,
    oracle_rows: Sequence[Mapping[str, object]],
    raw_rewards: Tensor,
    tripartite_scores: Sequence[TripartiteBiasScore],
    consensus_bonus_weight: float,
) -> tuple[Tensor, dict[str, float]]:
    """Compute the terminal reward tensor selected by the reward-version flag."""

    if len(oracle_rows) != int(raw_rewards.shape[0]):
        raise ValueError("oracle_rows length must match raw_rewards")
    if len(tripartite_scores) != len(oracle_rows):
        raise ValueError("tripartite_scores length must match oracle_rows")

    version = reward_version.strip().lower()
    effective_values: list[float] = []
    base_values: list[float] = []
    consensus_values: list[float] = []
    complement_values: list[float] = []
    clash_values: list[float] = []
    shear_values: list[float] = []
    hysteresis_values: list[float] = []
    reversibility_values: list[float] = []
    pathway_direct_values: list[float] = []
    pathway_neighborhood_values: list[float] = []
    pathway_score_values: list[float] = []
    pose_values: list[float] = []
    charge_values: list[float] = []

    for index, row in enumerate(oracle_rows):
        pi_complement = numeric_value(row.get("pi_complement"), 0.0)
        pi_clash_pocket = numeric_value(row.get("pi_clash_pocket"), 0.0)
        lock_geometry = numeric_value(row.get("lock_geometry_score", row.get("pi_clash_lock", 0.0)), 0.0)
        sigma_shear = numeric_value(row.get("sigma_shear_mean", row.get("sigma_shear", row.get("shear_stress"))), 0.0)
        hysteresis_mean = numeric_value(row.get("hysteresis_mean", row.get("hysteresis_score")), 0.0)
        reversibility_mean = numeric_value(row.get("reversibility_mean", row.get("reversibility")), 1.0)
        pathway_voxels = numeric_value(row.get("pathway_voxels_occupied"), 0.0)
        pathway_neighborhood = numeric_value(row.get("pathway_neighborhood_contacts"), 0.0)
        pathway_score = numeric_value(row.get("pathway_score_mean"), 0.0)
        u_pose = numeric_value(row.get("u_pose"), 0.0)
        consensus_bonus = numeric_value(
            row.get("consensus_complement_bonus", row.get("population_consensus_bonus")),
            0.0,
        )
        charge_feature = numeric_value(row.get("charge_feature_mean", row.get("am1bcc_charge")), 0.0)
        complement_values.append(pi_complement)
        clash_values.append(pi_clash_pocket)
        consensus_values.append(consensus_bonus)
        shear_values.append(sigma_shear)
        hysteresis_values.append(hysteresis_mean)
        reversibility_values.append(reversibility_mean)
        pathway_direct_values.append(pathway_voxels)
        pathway_neighborhood_values.append(pathway_neighborhood)
        pathway_score_values.append(pathway_score)
        pose_values.append(u_pose)
        charge_values.append(charge_feature)

        if version == "v1_base":
            base_reward = numeric_value(raw_rewards[index].item(), 0.0)
            effective = base_reward
        elif version == "v2_tripartite":
            base_reward = compute_reward_v2(row, tripartite_scores[index])
            effective = base_reward
        elif version in {"v3_consensus", "v3_consensus_grid", "v3_population_consensus"}:
            base_reward = (
                pi_complement
                - pi_clash_pocket
                + lock_geometry
                - math.log1p(max(sigma_shear, 0.0))
            )
            effective = base_reward + consensus_bonus_weight * consensus_bonus
        elif version in {"v4_full_field", "v4_thermodynamic_field", "v4_full_stack"}:
            base_reward = (
                pi_complement
                - pi_clash_pocket
                + lock_geometry
                + tripartite_scores[index].bias_projection_score
            )
            shear_penalty = math.log1p(max(sigma_shear, 0.0))
            hysteresis_penalty = max(0.0, 1.0 - max(0.0, min(1.0, reversibility_mean)))
            pathway_bonus = (
                0.5 * max(pathway_voxels, 0.0)
                + 0.1 * max(pathway_neighborhood, 0.0)
                + max(pathway_score, 0.0)
            )
            effective = (
                base_reward
                + consensus_bonus_weight * consensus_bonus
                - shear_penalty
                - hysteresis_penalty
                + pathway_bonus
                - max(u_pose, 0.0)
            )
        else:
            base_reward = numeric_value(raw_rewards[index].item(), 0.0)
            effective = base_reward

        base_values.append(base_reward)
        effective_values.append(effective)

    if version in {"v4_full_field", "v4_thermodynamic_field", "v4_full_stack"} and effective_values:
        min_effective = min(effective_values)
        if min_effective <= 0.0:
            effective_values = [value - min_effective + 1.0e-6 for value in effective_values]
    else:
        effective_values = [max(value, 1.0e-8) for value in effective_values]

    effective_rewards = torch.tensor(
        effective_values,
        dtype=raw_rewards.dtype,
        device=raw_rewards.device,
    )
    metrics = {
        "effective_reward_mean": _mean(effective_values),
        "effective_reward_std": _std(effective_values),
        "reward_base_mean": _mean(base_values),
        "consensus_bonus_mean": _mean(consensus_values),
        "consensus_bonus_std": _std(consensus_values),
        "pi_complement_std": _std(complement_values),
        "pi_clash_pocket_std": _std(clash_values),
        "shear_mean": _mean(shear_values),
        "hysteresis_mean": _mean(hysteresis_values),
        "reversibility_mean": _mean(reversibility_values),
        "pathway_voxels_occupied": _mean(pathway_direct_values),
        "pathway_neighborhood_contacts": _mean(pathway_neighborhood_values),
        "pathway_score_mean": _mean(pathway_score_values),
        "u_pose_mean": _mean(pose_values),
        "charge_feature_mean": _mean(charge_values),
    }
    return effective_rewards, metrics


def _mean(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    return float(sum(values) / float(len(values)))


def _std(values: Sequence[float]) -> float:
    if len(values) < 2:
        return 0.0
    mean = _mean(values)
    variance = sum((value - mean) ** 2 for value in values) / float(len(values))
    return math.sqrt(variance)


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
        expanded = sorted(path.glob("*.sdf")) if path.is_dir() else [path]
        for candidate in expanded:
            if not candidate.is_file():
                continue
            if candidate not in existing:
                existing.append(candidate)
    if not existing:
        raise RuntimeError("no scaffold SDFs were available for training initialization")
    return tuple(existing)


def resolve_track_a_path(path: Path | None, fallback: Path) -> Path:
    """Resolve directive-style relative paths against cwd and Track A."""

    if path is None:
        return fallback
    candidate = Path(path)
    if candidate.is_absolute() or candidate.exists():
        return candidate
    track_candidate = TRACK_A_DIR / candidate
    if track_candidate.exists():
        return track_candidate
    n80_candidate = N80_DIR / candidate
    if n80_candidate.exists():
        return n80_candidate
    return candidate


class TelemetryTee:
    """Write stdout/stderr to both terminal and a telemetry file."""

    def __init__(self, stream: TextIO, handle: TextIO) -> None:
        self._stream = stream
        self._handle = handle

    def write(self, data: str) -> int:
        written = self._stream.write(data)
        self._handle.write(data)
        self._handle.flush()
        return int(written or len(data))

    def flush(self) -> None:
        self._stream.flush()
        self._handle.flush()


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


def atom_base_features(atom: Chem.Atom, partial_charge: float, is_exit: bool, *, field_stack_version: str = "v1") -> list[float]:
    hybridization = atom.GetHybridization()
    features = [
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
    if field_stack_version == "v2":
        # Dedicated AM1-BCC/NAGL charge lane. The legacy charge lane at index 9
        # is retained for checkpoint compatibility; v2 exposes the explicit
        # electronic feature at x_base[:, 12] as required by the field stack.
        features.append(partial_charge)
    return features


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


def extend_phase_matrix(
    phase_matrix: Sequence[Sequence[float]],
    *,
    target_dim: int,
    field_fiber: np.ndarray | None = None,
) -> list[list[float]]:
    """Resize residue phase features and append field-stack lanes when available."""

    extended: list[list[float]] = []
    for phase_index, row in enumerate(phase_matrix):
        values = [float(value) for value in row[:target_dim]]
        if len(values) < target_dim:
            field_tail = (
                [float(value) for value in field_fiber[phase_index, 8:12]]
                if field_fiber is not None and field_fiber.shape == (5, 12)
                else [0.0, 0.0, 0.0, 1.0]
            )
            values.extend(field_tail[: max(0, target_dim - len(values))])
        extended.append(values[:target_dim])
    return extended


def build_scaffold_graph(
    paths: TrainingPaths,
    ligand_sdf: Path | None = None,
    *,
    field_stack_version: str = "v1",
    fiber_lookup: FieldLookup | None = None,
) -> ScaffoldGraph:
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
    xyz_rows: list[list[float]] = []
    target_phase_dim = 12 if field_stack_version == "v2" else 8
    zero_phase = [[0.0] * target_phase_dim for _ in PHASES]
    if target_phase_dim >= 12:
        for row in zero_phase:
            row[11] = 1.0
    for old_idx in retained_original_indices:
        atom = mol.GetAtomWithIdx(old_idx)
        xyz = list(conf.GetAtomPosition(old_idx))
        residue_idx = nearest_residue_idx(xyz, ca_coordinates)
        partial_charge = atom_partial_charge(mol, old_idx)
        x_base.append(
            atom_base_features(
                atom,
                partial_charge,
                old_idx == exit_original_idx,
                field_stack_version=field_stack_version,
            )
        )
        field_fiber = (
            fiber_lookup.lookup_atom_fiber(np.array(xyz, dtype=np.float32))
            if field_stack_version == "v2" and fiber_lookup is not None
            else None
        )
        x_phase.append(
            extend_phase_matrix(
                phase_maps.get(residue_idx, zero_phase),
                target_dim=target_phase_dim,
                field_fiber=field_fiber,
            )
        )
        xyz_rows.append([float(value) for value in xyz])
    edge_index, edge_attr, active_mask = build_edges(mol, retained_original_indices, old_to_new, exit_original_idx)
    data = Data(
        x_base=torch.tensor(x_base, dtype=torch.float32),
        x_phase=torch.tensor(x_phase, dtype=torch.float32),
        xyz=torch.tensor(xyz_rows, dtype=torch.float32),
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


def build_scaffold_pool(
    paths: TrainingPaths,
    scaffold_sdfs: Sequence[Path],
    *,
    field_stack_version: str = "v1",
    fiber_lookup: FieldLookup | None = None,
) -> tuple[ScaffoldGraph, ...]:
    graphs = tuple(
        build_scaffold_graph(
            paths,
            path,
            field_stack_version=field_stack_version,
            fiber_lookup=fiber_lookup,
        )
        for path in scaffold_sdfs
    )
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


def load_action_space(paths: TrainingPaths, embedding_dim: int, *, base_feature_dim: int = 13) -> ActionSpace:
    anchors = normalize_anchor_table(pl.read_parquet(paths.anchors))
    survivor_source = pl.read_parquet(paths.survivors)
    if "coordinates_json" not in survivor_source.columns:
        survivor_source = survivor_source.with_columns(pl.lit("").alias("coordinates_json"))
    aggregations = [
        pl.col("score").first().alias("oracle_reward"),
        pl.col("canonical_smiles").first().alias("survivor_smiles"),
        pl.col("synthon_b_id").first().alias("synthon_b_id"),
        pl.col("product_id").first().alias("product_id"),
        pl.col("coordinates_json").first().alias("coordinates_json"),
        pl.col("selected_dihedral_deg").first().alias("selected_dihedral_deg"),
        pl.col("survival_tier").first().alias("survival_tier"),
    ]
    for column in ("rotamers_evaluated", "best_rotamer_rank", "n_surviving_rotamers"):
        if column in survivor_source.columns:
            aggregations.append(pl.col(column).first().alias(column))
    if "population_consensus_atoms_scored" in survivor_source.columns:
        aggregations.append(pl.col("population_consensus_atoms_scored").first().alias("population_consensus_atoms_scored"))
    for column in (
        "consensus_complement_bonus",
        "population_consensus_bonus",
        "population_consensus_bonus_scaled",
    ):
        if column in survivor_source.columns:
            aggregations.append(pl.col(column).first().alias(column))
    survivors = (
        survivor_source
        .sort("score", descending=True)
        .group_by("anchor_id")
        .agg(aggregations)
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
    table = table.with_columns(
        pl.struct(table.columns).map_elements(pose_penalty_from_row, return_dtype=pl.Float64).alias("u_pose"),
        pl.struct(table.columns).map_elements(pose_penalty_source_from_row, return_dtype=pl.String).alias("u_pose_source"),
        pl.col("partial_charge_mean_abs").fill_null(0.0).alias("charge_feature_mean"),
    )
    pose_source_counts = {
        str(row["u_pose_source"]): int(row["count"])
        for row in table.get_column("u_pose_source").value_counts().to_dicts()
    }
    print(f"u_pose_provenance counts={json.dumps(pose_source_counts, sort_keys=True)}", flush=True)
    valid_count = int(table.get_column("action_valid").sum())
    if valid_count < 100:
        raise RuntimeError(f"too few valid oracle-scored actions: {valid_count}")
    embeddings = anchor_embeddings_from_table(table, embedding_dim)
    action_base_features = action_base_features_from_table(table, base_feature_dim=base_feature_dim)
    charge_nonzero = int((action_base_features[:, -1].abs() > 0.0).sum().item()) if action_base_features.numel() else 0
    action_atom_features, action_atom_mask = action_atom_features_from_table(table, base_feature_dim=base_feature_dim)
    atom_charge_nonzero = (
        int(((action_atom_features[:, :, -1].abs() > 0.0) & action_atom_mask).sum().item())
        if action_atom_features.numel()
        else 0
    )
    atom_count = int(action_atom_mask.sum().item()) if action_atom_mask.numel() else 0
    print(
        "action_product_base_features_initialized "
        f"shape={list(action_base_features.shape)} charge_nonzero={charge_nonzero}/{action_base_features.shape[0]} "
        "charge_source=am1bcc_partial_charges_json",
        flush=True,
    )
    print(
        "action_product_atom_features_initialized "
        f"shape={list(action_atom_features.shape)} atom_mask_count={atom_count} "
        f"charge_nonzero={atom_charge_nonzero}/{atom_count} "
        "charge_source=am1bcc_partial_charges_json",
        flush=True,
    )
    valid_mask = torch.tensor(table.get_column("action_valid").to_list(), dtype=torch.bool)
    reward_values = torch.tensor(
        [float(value or 0.0) for value in table.get_column("oracle_reward").to_list()],
        dtype=torch.float32,
    )
    reward_logits = reward_values.clone()
    reward_logits = reward_logits.masked_fill(~valid_mask, torch.finfo(torch.float32).min)
    reward_targets = torch.softmax(reward_logits / 0.08, dim=0)
    return ActionSpace(
        table=table,
        anchor_embeddings=embeddings,
        valid_mask=valid_mask,
        reward_targets=reward_targets,
        action_base_features=action_base_features,
        action_atom_features=action_atom_features,
        action_atom_mask=action_atom_mask,
        action_rows=tuple(table.to_dicts()),
    )


def pose_penalty_from_row(row: Mapping[str, object]) -> float:
    """Compute u_pose from rotamer survival count, or a rank proxy when absent."""

    n_surviving = numeric_value(row.get("n_surviving_rotamers"), 0.0)
    if n_surviving > 0.0:
        surviving = max(1.0, min(6.0, n_surviving))
        return float(-math.log(surviving / 6.0))
    best_rank = numeric_value(row.get("best_rotamer_rank"), 0.0)
    if best_rank > 0.0:
        # The full survivor schema lacks n_surviving_rotamers. A late-winning
        # best rotamer means pose sensitivity, so convert rank 1..6 to a
        # conservative survivor-count proxy 6..1 with explicit source tagging.
        surviving_proxy = max(1.0, min(6.0, 7.0 - best_rank))
        return float(-math.log(surviving_proxy / 6.0))
    evaluated = numeric_value(row.get("rotamers_evaluated"), 6.0)
    return float(-math.log(max(1.0, min(6.0, evaluated)) / 6.0))


def pose_penalty_source_from_row(row: Mapping[str, object]) -> str:
    if numeric_value(row.get("n_surviving_rotamers"), 0.0) > 0.0:
        return "n_surviving_rotamers"
    if numeric_value(row.get("best_rotamer_rank"), 0.0) > 0.0:
        return "best_rotamer_rank_proxy"
    return "rotamers_evaluated_proxy"


def build_signal_fiber_lookup(signal_grid: Path | None, grid_mapping: Path) -> SignalGridFiberLookup | None:
    """Create the direct signal-grid lookup used for product fibers and exit masks."""

    if signal_grid is None:
        return None
    resolved_grid = resolve_track_a_path(signal_grid, N80_DIR / "signal_grid_variance_channel.parquet")
    resolved_mapping = resolve_track_a_path(grid_mapping, DEFAULT_GRID_MAPPING)
    if not resolved_grid.is_file() or not resolved_mapping.is_file():
        print(
            "product_fiber_lookup_unavailable "
            f"signal_grid={resolved_grid} grid_mapping={resolved_mapping}",
            flush=True,
        )
        return None
    lookup = SignalGridFiberLookup(resolved_grid, resolved_mapping)
    print(
        "product_fiber_lookup_initialized "
        f"method=direct_signal_grid_lookup signal_grid={resolved_grid} grid_mapping={resolved_mapping}",
        flush=True,
    )
    return lookup


def build_field_lookup(
    *,
    field_stack_version: str,
    signal_grid: Path | None,
    grid_mapping: Path,
    shear_stress: Path | None,
    hysteresis_tensor: Path | None,
    translation_pathway: Path | None,
    cross_species: Path | None,
) -> FieldLookup | None:
    """Create either the legacy signal lookup or the full thermodynamic stack."""

    version = field_stack_version.strip().lower()
    if version == "v1":
        return build_signal_fiber_lookup(signal_grid, grid_mapping)
    resolved_grid = resolve_track_a_path(signal_grid, DEFAULT_POPULATION_CONSENSUS_GRID)
    resolved_mapping = resolve_track_a_path(grid_mapping, DEFAULT_GRID_MAPPING)
    resolved_shear = resolve_track_a_path(shear_stress, DEFAULT_SHEAR_STRESS)
    resolved_hysteresis = resolve_track_a_path(hysteresis_tensor, DEFAULT_HYSTERESIS_TENSOR)
    resolved_pathway = resolve_track_a_path(translation_pathway, DEFAULT_TRANSLATION_PATHWAY)
    resolved_cross_species = resolve_track_a_path(cross_species, DEFAULT_CROSS_SPECIES) if cross_species is not None else DEFAULT_CROSS_SPECIES
    required = {
        "signal_grid": resolved_grid,
        "grid_mapping": resolved_mapping,
        "shear_stress": resolved_shear,
        "hysteresis_tensor": resolved_hysteresis,
        "translation_pathway": resolved_pathway,
    }
    missing = {name: path for name, path in required.items() if not path.is_file()}
    if missing:
        raise RuntimeError(f"field-stack v2 missing inputs: {missing}")
    lookup = ThermodynamicFieldStack(
        resolved_grid,
        resolved_mapping,
        shear_stress_path=resolved_shear,
        hysteresis_tensor_path=resolved_hysteresis,
        translation_pathway_path=resolved_pathway,
        cross_species_path=resolved_cross_species if resolved_cross_species.is_file() else None,
    )
    print(
        "thermodynamic_field_stack_initialized "
        f"version=v2 signal_grid={resolved_grid} shear={resolved_shear} "
        f"hysteresis={resolved_hysteresis} pathway={resolved_pathway} "
        f"voxels={len(lookup.field)} pathway_voxels={len(lookup.pathway_voxels)}",
        flush=True,
    )
    return lookup


def attach_exit_ray_adjustments(
    action_space: ActionSpace,
    *,
    reference_graph: ScaffoldGraph,
    fiber_lookup: FieldLookup | None,
    enabled: bool,
) -> ActionSpace:
    """Attach action-specific steric logit adjustments from product coordinates."""

    if fiber_lookup is None or "coordinates_json" not in action_space.table.columns:
        return action_space
    ray_caster = ExitAtomRayCaster(fiber_lookup)
    n_scaffold = reference_graph.retained_atom_count
    adjustments: list[float] = []
    action_phase_features = torch.zeros(
        (action_space.table.height, 5, reference_graph.phase_feature_dim),
        dtype=torch.float32,
    )
    sigma_shear_values: list[float] = []
    hysteresis_values: list[float] = []
    reversibility_values: list[float] = []
    pathway_counts: list[float] = []
    pathway_neighborhood_counts: list[float] = []
    pathway_neighborhood_scores: list[float] = []
    pathway_scores: list[float] = []
    species_conservation_scores: list[float] = []
    field_consensus_values: list[float] = []
    scored = 0
    conditioned = 0
    for row in action_space.table.iter_rows(named=True):
        coords = _coordinates_from_row(row)
        if coords is None or coords.shape[0] <= n_scaffold:
            adjustments.append(0.0)
            stats = _zero_action_field_stats()
            sigma_shear_values.append(stats["sigma_shear_mean"])
            hysteresis_values.append(stats["hysteresis_mean"])
            reversibility_values.append(stats["reversibility_mean"])
            pathway_counts.append(stats["pathway_voxels_occupied"])
            pathway_neighborhood_counts.append(stats["pathway_neighborhood_contacts"])
            pathway_neighborhood_scores.append(stats["pathway_neighborhood_score_mean"])
            pathway_scores.append(stats["pathway_score_mean"])
            species_conservation_scores.append(stats["species_conservation_score_mean"])
            field_consensus_values.append(stats["field_consensus_complement_bonus"])
            continue
        scaffold_xyz = coords[:n_scaffold]
        synthon_xyz = coords[n_scaffold:]
        centroid = scaffold_xyz.mean(axis=0)
        distances = np.linalg.norm(synthon_xyz - centroid, axis=1)
        distal = synthon_xyz[int(np.argmax(distances))]
        mask_value = (
            float(ray_caster.compute_exit_masks(np.array([distal], dtype=np.float32), centroid)[0].item())
            if enabled
            else 0.0
        )
        adjustments.append(mask_value)
        product_fiber = fiber_lookup.lookup_product_fiber(
            cast(Tensor, reference_graph.data.x_phase),
            coords,
            n_scaffold=n_scaffold,
        )
        action_phase_features[len(adjustments) - 1] = product_fiber[n_scaffold:].mean(dim=0)
        stats = fiber_lookup.field_stats_for_coordinates(coords, n_scaffold=n_scaffold)
        sigma_shear_values.append(stats["sigma_shear_mean"])
        hysteresis_values.append(stats["hysteresis_mean"])
        reversibility_values.append(stats["reversibility_mean"])
        pathway_counts.append(stats["pathway_voxels_occupied"])
        pathway_neighborhood_counts.append(stats["pathway_neighborhood_contacts"])
        pathway_neighborhood_scores.append(stats["pathway_neighborhood_score_mean"])
        pathway_scores.append(stats["pathway_score_mean"])
        species_conservation_scores.append(stats["species_conservation_score_mean"])
        field_consensus_values.append(stats["consensus_complement_bonus"])
        scored += 1
        conditioned += 1
    tensor = torch.tensor(adjustments, dtype=torch.float32)
    finite_values = tensor[torch.isfinite(tensor)]
    mean_value = float(finite_values.mean().item()) if finite_values.numel() else float("-inf")
    print(
        "exit_ray_masks_initialized "
        f"actions_scored={scored}/{len(adjustments)} blocked={int(torch.isneginf(tensor).sum().item())} "
        f"mean_adjustment={mean_value:.6f}",
        flush=True,
    )
    print(
        "action_product_fiber_initialized "
        f"actions_conditioned={conditioned}/{len(adjustments)} method=direct_signal_grid_lookup "
        f"shape={list(action_phase_features.shape)}",
        flush=True,
    )
    enriched_table = action_space.table.with_columns(
        pl.Series("sigma_shear_mean", sigma_shear_values),
        pl.Series("hysteresis_mean", hysteresis_values),
        pl.Series("reversibility_mean", reversibility_values),
        pl.Series("pathway_voxels_occupied", pathway_counts),
        pl.Series("pathway_neighborhood_contacts", pathway_neighborhood_counts),
        pl.Series("pathway_neighborhood_score_mean", pathway_neighborhood_scores),
        pl.Series("pathway_score_mean", pathway_scores),
        pl.Series("species_conservation_score_mean", species_conservation_scores),
        pl.Series("field_consensus_complement_bonus", field_consensus_values),
    )
    return ActionSpace(
        table=enriched_table,
        anchor_embeddings=action_space.anchor_embeddings,
        valid_mask=action_space.valid_mask,
        reward_targets=action_space.reward_targets,
        exit_ray_adjustments=tensor,
        action_phase_features=action_phase_features,
        action_base_features=action_space.action_base_features,
        action_atom_features=action_space.action_atom_features,
        action_atom_mask=action_space.action_atom_mask,
        action_rows=tuple(enriched_table.to_dicts()),
    )


def _zero_action_field_stats() -> dict[str, float]:
    return {
        "sigma_shear_mean": 0.0,
        "hysteresis_mean": 0.0,
        "reversibility_mean": 1.0,
        "pathway_voxels_occupied": 0.0,
        "pathway_neighborhood_contacts": 0.0,
        "pathway_neighborhood_score_mean": 0.0,
        "pathway_score_mean": 0.0,
        "species_conservation_score_mean": 1.0,
        "field_consensus_complement_bonus": 0.0,
    }


def _coordinates_from_row(row: Mapping[str, object]) -> np.ndarray | None:
    raw = row.get("coordinates_json")
    if not isinstance(raw, str) or not raw:
        return None
    try:
        coords = np.array(json.loads(raw), dtype=np.float32)
    except (TypeError, ValueError, json.JSONDecodeError):
        return None
    if coords.ndim != 2 or coords.shape[1] != 3:
        return None
    if not bool(np.isfinite(coords).all()):
        return None
    return coords


def _tensor_to_coordinates_json(xyz: Tensor) -> str:
    rows = xyz.detach().cpu().to(dtype=torch.float32).tolist()
    return json.dumps([[float(value) for value in row] for row in rows], separators=(",", ":"))


def action_row(action_space: ActionSpace, action_idx: int) -> Mapping[str, object]:
    if action_space.action_rows:
        if action_idx < 0 or action_idx >= len(action_space.action_rows):
            raise ValueError("action_idx must reference a real growth action")
        return action_space.action_rows[int(action_idx)]
    if action_idx < 0 or action_idx >= action_space.table.height:
        raise ValueError("action_idx must reference a real growth action")
    return cast(dict[str, object], action_space.table.row(int(action_idx), named=True))


def _state_node_count(state: AssembledState) -> int:
    return int(cast(Tensor, state.data.x_base).shape[0])


def initial_assembled_state(graph: ScaffoldGraph) -> AssembledState:
    data = graph.data.clone()
    coordinates = _tensor_to_coordinates_json(cast(Tensor, data.xyz))
    return AssembledState(
        scaffold=graph,
        data=data,
        exit_node_idx=graph.exit_node_idx,
        history=tuple(),
        last_growth_action_idx=None,
        canonical_smiles=graph.scaffold_id,
        anchor_id=graph.scaffold_id,
        coordinates_json=coordinates,
        score_atom_offset=graph.retained_atom_count,
        survivor_lookup_smiles=None,
    )


def _fragment_offset_from_row(row: Mapping[str, object], coords: np.ndarray | None, fallback_scaffold_atoms: int) -> int:
    if coords is None or int(coords.shape[0]) == 0:
        return 0
    total_atoms = int(coords.shape[0])
    atoms_scored = numeric_value(row.get("n_heavy_atoms"), 0.0)
    if atoms_scored <= 0.0:
        atoms_scored = numeric_value(row.get("population_consensus_atoms_scored"), 0.0)
    if atoms_scored > 0.0:
        return max(0, min(total_atoms - 1, total_atoms - int(atoms_scored)))
    return max(0, min(total_atoms - 1, fallback_scaffold_atoms))


def _resize_feature_rows(features: Tensor, *, row_count: int, feature_dim: int) -> Tensor:
    if features.ndim != 2:
        raise ValueError("features must have shape [num_atoms, feature_dim]")
    if int(features.shape[1]) < feature_dim:
        pad = features.new_zeros((int(features.shape[0]), feature_dim - int(features.shape[1])))
        features = torch.cat([features, pad], dim=1)
    elif int(features.shape[1]) > feature_dim:
        features = features[:, :feature_dim]
    if int(features.shape[0]) == row_count:
        return features
    if int(features.shape[0]) > row_count:
        return features[:row_count]
    pad_rows = features.new_zeros((row_count - int(features.shape[0]), feature_dim))
    return torch.cat([features, pad_rows], dim=0)


def _action_fragment_features(action_space: ActionSpace, action_idx: int, *, feature_dim: int) -> Tensor:
    features: Tensor
    if action_space.action_atom_features is not None and action_space.action_atom_mask is not None:
        atom_features = action_space.action_atom_features[int(action_idx)]
        atom_mask = action_space.action_atom_mask[int(action_idx)]
        selected = atom_features[atom_mask]
        features = selected.clone() if int(selected.shape[0]) > 0 else atom_features[:1].clone()
    elif action_space.action_base_features is not None:
        features = action_space.action_base_features[int(action_idx)].unsqueeze(0).clone()
    else:
        features = torch.zeros((1, feature_dim), dtype=torch.float32)
    return _resize_feature_rows(features.to(dtype=torch.float32), row_count=max(1, int(features.shape[0])), feature_dim=feature_dim)


def _action_phase_features(action_space: ActionSpace, action_idx: int, *, atom_count: int, phase_dim: int) -> Tensor:
    if action_space.action_phase_features is not None:
        phase = action_space.action_phase_features[int(action_idx)].to(dtype=torch.float32)
        if int(phase.shape[1]) < phase_dim:
            phase = torch.cat([phase, phase.new_zeros((5, phase_dim - int(phase.shape[1])))], dim=1)
        elif int(phase.shape[1]) > phase_dim:
            phase = phase[:, :phase_dim]
    else:
        phase = torch.zeros((5, phase_dim), dtype=torch.float32)
        if phase_dim >= 1:
            phase[:, -1] = 1.0
    return phase.unsqueeze(0).expand(atom_count, -1, -1).clone()


def _fragment_coordinates(
    *,
    row: Mapping[str, object],
    current_xyz: Tensor,
    exit_node_idx: int,
    fragment_atoms: int,
    fallback_scaffold_atoms: int,
) -> Tensor:
    coords = _coordinates_from_row(row)
    if coords is not None:
        offset = _fragment_offset_from_row(row, coords, fallback_scaffold_atoms)
        fragment = torch.from_numpy(coords[offset:].astype(np.float32, copy=False))
        if int(fragment.shape[0]) >= fragment_atoms:
            return fragment[:fragment_atoms].clone()
    exit_xyz = current_xyz[exit_node_idx].to(dtype=torch.float32)
    rows: list[Tensor] = []
    for atom_idx in range(fragment_atoms):
        rows.append(exit_xyz + torch.tensor([1.45 * float(atom_idx + 1), 0.35 * float(atom_idx % 3), 0.0]))
    return torch.stack(rows, dim=0)


def _finite_xyz_rows(tensor: Tensor, *, label: str) -> list[list[float]]:
    values = tensor.detach().cpu().to(dtype=torch.float32)
    if values.ndim != 2 or int(values.shape[1]) != 3:
        raise ValueError(f"{label} must have shape [N, 3]")
    if int(values.shape[0]) == 0:
        raise ValueError(f"{label} must contain at least one atom")
    if not bool(torch.isfinite(values).all().item()):
        raise ValueError(f"{label} contains non-finite coordinates")
    return [[float(component) for component in row] for row in values.tolist()]


def _perpendicular_unit(axis: Tensor) -> Tensor:
    axis = _unit_vector(axis.to(dtype=torch.float32), torch.tensor([1.0, 0.0, 0.0], dtype=torch.float32))
    candidate = torch.tensor([1.0, 0.0, 0.0], dtype=torch.float32)
    if float(torch.abs(torch.dot(axis, candidate)).item()) > 0.85:
        candidate = torch.tensor([0.0, 1.0, 0.0], dtype=torch.float32)
    perpendicular = torch.cross(axis, candidate, dim=0)
    return _unit_vector(perpendicular, torch.tensor([0.0, 0.0, 1.0], dtype=torch.float32))


def _scaffold_zmatrix_reference_points(current_xyz: Tensor, exit_node_idx: int) -> tuple[Tensor, Tensor, Tensor]:
    product_xyz = current_xyz.detach().to(dtype=torch.float32)
    if product_xyz.ndim != 2 or int(product_xyz.shape[1]) != 3:
        raise ValueError("current_xyz must have shape [N, 3]")
    if not bool(torch.isfinite(product_xyz).all().item()):
        raise ValueError("current_xyz contains non-finite coordinates")
    exit_idx = int(exit_node_idx)
    if exit_idx < 0 or exit_idx >= int(product_xyz.shape[0]):
        raise ValueError("exit_node_idx is outside current_xyz")
    exit_xyz = product_xyz[exit_idx]
    other_indices = [idx for idx in range(int(product_xyz.shape[0])) if idx != exit_idx]
    if other_indices:
        distances = torch.linalg.vector_norm(product_xyz[other_indices] - exit_xyz.unsqueeze(0), dim=1)
        nearest = int(other_indices[int(torch.argmin(distances).item())])
        ref1 = product_xyz[nearest]
    else:
        ref1 = exit_xyz - torch.tensor([1.0, 0.0, 0.0], dtype=torch.float32)
    axis = exit_xyz - ref1
    perpendicular = _perpendicular_unit(axis)
    ref2 = ref1 + perpendicular
    return exit_xyz, ref1, ref2


def _undirected_bonds_from_edge_index(edge_index: Tensor, edge_attr: Tensor) -> list[list[int]]:
    edges = edge_index.detach().cpu().to(dtype=torch.long)
    attrs = edge_attr.detach().cpu().to(dtype=torch.float32)
    bonds: set[tuple[int, int]] = set()
    if edges.ndim != 2 or int(edges.shape[0]) != 2:
        raise ValueError("edge_index must have shape [2, E]")
    if attrs.ndim != 2 or int(attrs.shape[0]) != int(edges.shape[1]):
        raise ValueError("edge_attr must have one row per edge")
    for edge_idx in range(int(edges.shape[1])):
        bond_order = float(attrs[edge_idx, 1].item()) * 3.0 if int(attrs.shape[1]) > 1 else 1.0
        if bond_order <= 0.25:
            continue
        lhs = int(edges[0, edge_idx].item())
        rhs = int(edges[1, edge_idx].item())
        if lhs == rhs:
            continue
        bonds.add((min(lhs, rhs), max(lhs, rhs)))
    return [[lhs, rhs] for lhs, rhs in sorted(bonds)]


def _ensure_connected_fragment_bonds(fragment_bonds: set[tuple[int, int]], fragment_xyz: Tensor) -> None:
    atom_count = int(fragment_xyz.shape[0])
    if atom_count <= 1:
        return
    connected: set[int] = {0}
    remaining: set[int] = set(range(1, atom_count))
    while remaining:
        best_pair: tuple[int, int] | None = None
        best_distance = float("inf")
        for lhs in connected:
            lhs_xyz = fragment_xyz[lhs]
            for rhs in remaining:
                distance = float(torch.linalg.vector_norm(fragment_xyz[rhs] - lhs_xyz).item())
                if distance < best_distance:
                    best_distance = distance
                    best_pair = (lhs, rhs)
        if best_pair is None:
            break
        lhs, rhs = best_pair
        fragment_bonds.add((min(lhs, rhs), max(lhs, rhs)))
        connected.add(rhs)
        remaining.remove(rhs)


def _fragment_bonds_from_geometry(fragment_xyz: Tensor, fragment_features: Tensor) -> list[list[int]]:
    """Infer covalent fragment topology from conformer geometry and atom features."""

    xyz = fragment_xyz.detach().to(dtype=torch.float32)
    features = fragment_features.detach().to(dtype=torch.float32)
    if xyz.ndim != 2 or int(xyz.shape[1]) != 3:
        raise ValueError("fragment_xyz must have shape [N, 3]")
    if features.ndim != 2 or int(features.shape[0]) != int(xyz.shape[0]):
        raise ValueError("fragment_features must have one row per fragment atom")
    periodic_table = Chem.GetPeriodicTable()
    bonds: set[tuple[int, int]] = set()
    atom_count = int(xyz.shape[0])
    for lhs in range(atom_count):
        lhs_atomic = _atomic_num_from_features(features[lhs])
        lhs_radius = float(periodic_table.GetRcovalent(lhs_atomic))
        for rhs in range(lhs + 1, atom_count):
            rhs_atomic = _atomic_num_from_features(features[rhs])
            rhs_radius = float(periodic_table.GetRcovalent(rhs_atomic))
            distance = float(torch.linalg.vector_norm(xyz[rhs] - xyz[lhs]).item())
            threshold = max(0.75, 1.25 * (lhs_radius + rhs_radius) + 0.20)
            if distance <= threshold:
                bonds.add((lhs, rhs))
    _ensure_connected_fragment_bonds(bonds, xyz)
    return [[lhs, rhs] for lhs, rhs in sorted(bonds)]


def ensure_kinematic_assemble_binary() -> Path:
    binary = REPO_ROOT / "target/release/kinematic_assemble"
    if binary.is_file():
        return binary
    print(
        "rust_kinematic_assembly_preflight status=building binary=kinematic_assemble",
        flush=True,
    )
    completed = subprocess.run(
        ["cargo", "build", "--release", "-p", "prism-forge", "--bin", "kinematic_assemble"],
        cwd=REPO_ROOT,
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            "failed to build Rust kinematic assembly binary "
            f"returncode={completed.returncode} stderr={completed.stderr.strip()}"
        )
    if not binary.is_file():
        raise RuntimeError(f"Rust kinematic assembly build completed but binary is missing: {binary}")
    print(
        "rust_kinematic_assembly_preflight status=ready binary=kinematic_assemble",
        flush=True,
    )
    return binary


def rust_zmatrix_assemble_product(
    fragment_xyz: Tensor,
    *,
    fragment_bonds: Sequence[Sequence[int]],
    current_xyz: Tensor,
    current_edge_index: Tensor,
    current_edge_attr: Tensor,
    exit_node_idx: int,
) -> RustAssemblyProduct:
    """Assemble a product graph through the Rust Z-matrix subprocess."""

    binary = ensure_kinematic_assemble_binary()
    fragment = fragment_xyz.detach().to(dtype=torch.float32)
    _finite_xyz_rows(fragment, label="fragment_xyz")
    exit_xyz, ref1, ref2 = _scaffold_zmatrix_reference_points(current_xyz, int(exit_node_idx))
    scaffold_coordinates = _finite_xyz_rows(current_xyz, label="current_xyz")
    request = {
        "requests": [
            {
                "trajectory_id": "growth-000000",
                "scaffold_coordinates": scaffold_coordinates,
                "scaffold_bonds": _undirected_bonds_from_edge_index(current_edge_index, current_edge_attr),
                "scaffold_exit_atom_index": int(exit_node_idx),
                "scaffold_exit_atom": [float(value) for value in exit_xyz.tolist()],
                "scaffold_reference_atom_1": [float(value) for value in ref1.tolist()],
                "scaffold_reference_atom_2": [float(value) for value in ref2.tolist()],
                "fragment_coordinates": _finite_xyz_rows(fragment, label="fragment_xyz"),
                "fragment_bonds": [[int(lhs), int(rhs)] for lhs, rhs in fragment_bonds],
                "bond_length_a": 1.45,
                "bond_angle_deg": 109.5,
                "dihedral_deg": 180.0,
                "hybridization_model": "sp3_default",
            }
        ]
    }
    scratch = REPO_ROOT / ".scratch/kinematic_assembly"
    scratch.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="zmatrix_", dir=scratch) as tmpdir:
        request_path = Path(tmpdir) / "request.json"
        response_path = Path(tmpdir) / "response.json"
        request_path.write_text(json.dumps(request, separators=(",", ":")), encoding="utf-8")
        completed = subprocess.run(
            [str(binary), "--input", str(request_path), "--output", str(response_path)],
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        if completed.returncode != 0:
            raise RuntimeError(
                "Rust kinematic assembly failed "
                f"returncode={completed.returncode} stderr={completed.stderr.strip()}"
            )
        decoded = json.loads(response_path.read_text(encoding="utf-8"))
    responses = decoded.get("responses") if isinstance(decoded, dict) else None
    if not isinstance(responses, list) or len(responses) != 1:
        raise RuntimeError("Rust kinematic assembly emitted invalid response envelope")
    response = responses[0]
    if not isinstance(response, dict) or response.get("assembly_mode") != "rust_zmatrix_subprocess":
        raise RuntimeError("Rust kinematic assembly did not report rust_zmatrix_subprocess mode")
    transformed = torch.tensor(response.get("coordinates"), dtype=torch.float32)
    product_coordinates = torch.tensor(response.get("product_coordinates"), dtype=torch.float32)
    raw_bonds = response.get("product_bonds")
    if transformed.ndim != 2 or int(transformed.shape[1]) != 3 or int(transformed.shape[0]) != int(fragment.shape[0]):
        raise RuntimeError("Rust kinematic assembly emitted coordinates with an unexpected shape")
    if product_coordinates.ndim != 2 or int(product_coordinates.shape[1]) != 3:
        raise RuntimeError("Rust kinematic assembly emitted product coordinates with an unexpected shape")
    if int(product_coordinates.shape[0]) != len(scaffold_coordinates) + int(fragment.shape[0]):
        raise RuntimeError("Rust kinematic assembly product atom count does not match scaffold + fragment")
    if not bool(torch.isfinite(transformed).all().item()) or not bool(torch.isfinite(product_coordinates).all().item()):
        raise RuntimeError("Rust kinematic assembly emitted non-finite coordinates")
    if not isinstance(raw_bonds, list):
        raise RuntimeError("Rust kinematic assembly emitted invalid product bond list")
    product_bonds: list[tuple[int, int]] = []
    for item in raw_bonds:
        if not isinstance(item, list) or len(item) != 2:
            raise RuntimeError("Rust kinematic assembly emitted malformed product bond")
        product_bonds.append((int(item[0]), int(item[1])))
    print(
        "rust_kinematic_assembly_applied "
        f"mode=rust_zmatrix_subprocess atoms={int(transformed.shape[0])} "
        f"product_atoms={int(product_coordinates.shape[0])} product_bonds={len(product_bonds)} "
        "z_matrix_active=true",
        flush=True,
    )
    return RustAssemblyProduct(
        fragment_coordinates=transformed,
        product_coordinates=product_coordinates,
        product_bonds=tuple(product_bonds),
        assembly_mode="rust_zmatrix_subprocess",
    )


def rust_zmatrix_attach_fragment(fragment_xyz: Tensor, *, current_xyz: Tensor, exit_node_idx: int) -> Tensor:
    """Compatibility wrapper returning only Rust-assembled fragment coordinates."""

    product = rust_zmatrix_assemble_product(
        fragment_xyz,
        fragment_bonds=_fragment_bonds_from_geometry(
            fragment_xyz,
            torch.zeros((int(fragment_xyz.shape[0]), 12), dtype=torch.float32),
        ),
        current_xyz=current_xyz,
        current_edge_index=torch.empty((2, 0), dtype=torch.long),
        current_edge_attr=torch.empty((0, 6), dtype=torch.float32),
        exit_node_idx=exit_node_idx,
    )
    return product.fragment_coordinates


def _unit_vector(vector: Tensor, fallback: Tensor) -> Tensor:
    norm = torch.linalg.vector_norm(vector)
    if float(norm.item()) <= 1.0e-6:
        fallback_norm = torch.linalg.vector_norm(fallback).clamp_min(1.0e-6)
        return cast(Tensor, fallback / fallback_norm)
    return cast(Tensor, vector / norm)


def _rotation_between_vectors(source: Tensor, target: Tensor) -> Tensor:
    source_u = _unit_vector(source, torch.tensor([1.0, 0.0, 0.0], dtype=torch.float32))
    target_u = _unit_vector(target, torch.tensor([1.0, 0.0, 0.0], dtype=torch.float32))
    dot = torch.dot(source_u, target_u).clamp(-1.0, 1.0)
    if float(dot.item()) > 1.0 - 1.0e-6:
        return torch.eye(3, dtype=torch.float32)
    if float(dot.item()) < -1.0 + 1.0e-6:
        axis = torch.cross(source_u, torch.tensor([1.0, 0.0, 0.0], dtype=torch.float32), dim=0)
        if float(torch.linalg.vector_norm(axis).item()) <= 1.0e-6:
            axis = torch.cross(source_u, torch.tensor([0.0, 1.0, 0.0], dtype=torch.float32), dim=0)
        axis = _unit_vector(axis, torch.tensor([0.0, 0.0, 1.0], dtype=torch.float32))
        return -torch.eye(3, dtype=torch.float32) + 2.0 * torch.outer(axis, axis)
    axis = torch.cross(source_u, target_u, dim=0)
    skew = torch.tensor(
        [
            [0.0, -float(axis[2].item()), float(axis[1].item())],
            [float(axis[2].item()), 0.0, -float(axis[0].item())],
            [-float(axis[1].item()), float(axis[0].item()), 0.0],
        ],
        dtype=torch.float32,
    )
    return torch.eye(3, dtype=torch.float32) + skew + skew.matmul(skew) * (1.0 / (1.0 + float(dot.item())))


def transform_fragment_to_exit(fragment_xyz: Tensor, *, current_xyz: Tensor, exit_node_idx: int) -> Tensor:
    """Rigidly place a fragment at the current product exit vector for the next growth step."""

    if fragment_xyz.ndim != 2 or int(fragment_xyz.shape[1]) != 3:
        raise ValueError("fragment_xyz must have shape [N_fragment, 3]")
    if int(fragment_xyz.shape[0]) == 0:
        raise ValueError("fragment_xyz must contain at least one atom")
    product_xyz = current_xyz.detach().to(dtype=torch.float32)
    fragment = fragment_xyz.detach().to(dtype=torch.float32)
    exit_xyz = product_xyz[int(exit_node_idx)]
    centroid = product_xyz.mean(dim=0)
    target_dir = _unit_vector(exit_xyz - centroid, torch.tensor([1.0, 0.0, 0.0], dtype=torch.float32))
    if int(fragment.shape[0]) > 1:
        source_dir = fragment[-1] - fragment[0]
        bond_length = float(torch.linalg.vector_norm(fragment[1] - fragment[0]).clamp(1.1, 1.8).item())
    else:
        source_dir = torch.tensor([1.0, 0.0, 0.0], dtype=torch.float32)
        bond_length = 1.45
    rotation = _rotation_between_vectors(source_dir, target_dir)
    centered = fragment - fragment[0].unsqueeze(0)
    transformed = centered.matmul(rotation.transpose(0, 1))
    return transformed + (exit_xyz + target_dir * bond_length).unsqueeze(0)


def _edge_attr_for_pair(xyz: Tensor, lhs: int, rhs: int, exit_node_idx: int, edge_dim: int) -> Tensor:
    lhs_xyz = xyz[lhs]
    rhs_xyz = xyz[rhs]
    vector = rhs_xyz - lhs_xyz
    distance = torch.linalg.vector_norm(vector).clamp_min(1.0e-6)
    exit_vector = lhs_xyz - xyz[exit_node_idx]
    denom = distance * torch.linalg.vector_norm(exit_vector).clamp_min(1.0e-6)
    alignment = torch.dot(vector, exit_vector) / denom
    values = [float(distance.item()) / 5.0, 1.0 / 3.0, 0.0, 0.0, 0.0, float(alignment.item())]
    if len(values) < edge_dim:
        values.extend([0.0] * (edge_dim - len(values)))
    return torch.tensor(values[:edge_dim], dtype=torch.float32)


def _append_growth_edges(
    *,
    current_edge_index: Tensor,
    current_edge_attr: Tensor,
    current_active_mask: Tensor,
    xyz: Tensor,
    old_exit_idx: int,
    first_new_idx: int,
    new_atom_count: int,
) -> tuple[Tensor, Tensor, Tensor]:
    edge_pairs: list[tuple[int, int]] = [(old_exit_idx, first_new_idx), (first_new_idx, old_exit_idx)]
    for offset in range(new_atom_count - 1):
        lhs = first_new_idx + offset
        rhs = lhs + 1
        edge_pairs.extend([(lhs, rhs), (rhs, lhs)])
    edge_dim = int(current_edge_attr.shape[1])
    new_edge_index = torch.tensor(edge_pairs, dtype=torch.long).t().contiguous()
    new_edge_attr = torch.stack(
        [_edge_attr_for_pair(xyz, lhs, rhs, old_exit_idx, edge_dim) for lhs, rhs in edge_pairs],
        dim=0,
    )
    new_active = torch.zeros((len(edge_pairs),), dtype=torch.bool)
    return (
        torch.cat([current_edge_index, new_edge_index], dim=1),
        torch.cat([current_edge_attr, new_edge_attr], dim=0),
        torch.cat([current_active_mask, new_active], dim=0),
    )


def _edge_tensors_from_rust_product_bonds(
    *,
    product_bonds: Sequence[tuple[int, int]],
    current_edge_index: Tensor,
    current_edge_attr: Tensor,
    current_active_mask: Tensor,
    xyz: Tensor,
    old_exit_idx: int,
) -> tuple[Tensor, Tensor, Tensor]:
    old_edges = current_edge_index.detach().to(dtype=torch.long)
    old_attrs = current_edge_attr.detach().to(dtype=torch.float32)
    old_mask = current_active_mask.detach().to(dtype=torch.bool)
    attr_by_direction: dict[tuple[int, int], tuple[Tensor, bool]] = {}
    for edge_idx in range(int(old_edges.shape[1])):
        lhs = int(old_edges[0, edge_idx].item())
        rhs = int(old_edges[1, edge_idx].item())
        attr_by_direction[(lhs, rhs)] = (old_attrs[edge_idx].clone(), bool(old_mask[edge_idx].item()))

    edge_pairs: list[tuple[int, int]] = []
    edge_attrs: list[Tensor] = []
    active_values: list[bool] = []
    edge_dim = int(current_edge_attr.shape[1])
    atom_count = int(xyz.shape[0])
    seen: set[tuple[int, int]] = set()
    for lhs_raw, rhs_raw in product_bonds:
        lhs = int(lhs_raw)
        rhs = int(rhs_raw)
        if lhs == rhs or lhs < 0 or rhs < 0 or lhs >= atom_count or rhs >= atom_count:
            continue
        for src, dst in ((lhs, rhs), (rhs, lhs)):
            if (src, dst) in seen:
                continue
            seen.add((src, dst))
            stored = attr_by_direction.get((src, dst))
            if stored is not None:
                attr, active = stored
            else:
                attr = _edge_attr_for_pair(xyz, src, dst, old_exit_idx, edge_dim)
                active = False
            edge_pairs.append((src, dst))
            edge_attrs.append(attr)
            active_values.append(active)

    if not edge_pairs:
        return (
            torch.empty((2, 0), dtype=torch.long),
            torch.empty((0, edge_dim), dtype=torch.float32),
            torch.empty((0,), dtype=torch.bool),
        )
    return (
        torch.tensor(edge_pairs, dtype=torch.long).t().contiguous(),
        torch.stack(edge_attrs, dim=0),
        torch.tensor(active_values, dtype=torch.bool),
    )


def fallback_history_chain_smiles(atom_count: int, history: Sequence[AssemblyHistoryStep]) -> str:
    rw_mol = Chem.RWMol()
    count = max(1, min(256, int(atom_count)))
    anchors_by_node = {min(count - 1, step.node_count - 1): step for step in history}
    for atom_idx in range(count):
        atom = Chem.Atom(6)
        matching_step = anchors_by_node.get(atom_idx)
        if matching_step is not None:
            atom.SetAtomMapNum(int(matching_step.action_idx) + 1)
        rw_mol.AddAtom(atom)
        if atom_idx > 0:
            rw_mol.AddBond(atom_idx - 1, atom_idx, Chem.BondType.SINGLE)
    return str(Chem.MolToSmiles(rw_mol.GetMol(), canonical=True))


def _atomic_num_from_features(features: Tensor) -> int:
    if int(features.shape[0]) == 0:
        return 6
    raw_atomic_num = int(round(float(features[0].item()) * 100.0))
    return raw_atomic_num if 1 <= raw_atomic_num <= 118 else 6


def canonical_smiles_from_product_graph(data: Data, history: Sequence[AssemblyHistoryStep]) -> str:
    """Create a connected RDKit-canonical identity for the current assembled graph."""

    x_base = cast(Tensor, data.x_base).detach().cpu()
    edge_index = cast(Tensor, data.edge_index).detach().cpu()
    edge_attr = cast(Tensor, data.edge_attr).detach().cpu()
    rw_mol = Chem.RWMol()
    anchors_by_node = {min(int(x_base.shape[0]) - 1, step.node_count - 1): step for step in history}
    for row_idx in range(int(x_base.shape[0])):
        atom = Chem.Atom(_atomic_num_from_features(x_base[row_idx]))
        matching_step = anchors_by_node.get(row_idx)
        if matching_step is not None:
            atom.SetAtomMapNum(int(matching_step.action_idx) + 1)
        rw_mol.AddAtom(atom)
    added: set[tuple[int, int]] = set()
    for edge_idx in range(int(edge_index.shape[1])):
        lhs = int(edge_index[0, edge_idx].item())
        rhs = int(edge_index[1, edge_idx].item())
        if lhs == rhs:
            continue
        pair = (min(lhs, rhs), max(lhs, rhs))
        if pair in added:
            continue
        bond_order = float(edge_attr[edge_idx, 1].item()) * 3.0 if int(edge_attr.shape[1]) > 1 else 1.0
        if bond_order <= 0.25:
            continue
        bond_type = Chem.BondType.SINGLE
        if bond_order > 2.5:
            bond_type = Chem.BondType.TRIPLE
        elif bond_order > 1.5:
            bond_type = Chem.BondType.DOUBLE
        try:
            rw_mol.AddBond(pair[0], pair[1], bond_type)
        except RuntimeError:
            continue
        added.add(pair)
    mol = rw_mol.GetMol()
    try:
        smiles = Chem.MolToSmiles(mol, canonical=True)
        if smiles and "." not in smiles:
            return str(smiles)
    except (ValueError, RuntimeError):
        pass
    return fallback_history_chain_smiles(int(x_base.shape[0]), history)


def grow_state_with_action(
    state: AssembledState,
    *,
    action_space: ActionSpace,
    action_idx: int,
    step: int,
) -> AssembledState:
    row = action_row(action_space, int(action_idx))
    current_x_base = cast(Tensor, state.data.x_base).detach().clone()
    current_x_phase = cast(Tensor, state.data.x_phase).detach().clone()
    current_xyz = cast(Tensor, state.data.xyz).detach().clone()
    base_dim = int(current_x_base.shape[1])
    phase_dim = int(current_x_phase.shape[2])
    fragment_features = _action_fragment_features(action_space, int(action_idx), feature_dim=base_dim)
    fragment_atoms = max(1, int(fragment_features.shape[0]))
    raw_fragment_xyz = _fragment_coordinates(
        row=row,
        current_xyz=current_xyz,
        exit_node_idx=state.exit_node_idx,
        fragment_atoms=fragment_atoms,
        fallback_scaffold_atoms=state.scaffold.retained_atom_count,
    )
    rust_product = rust_zmatrix_assemble_product(
        raw_fragment_xyz,
        fragment_bonds=_fragment_bonds_from_geometry(raw_fragment_xyz, fragment_features),
        current_xyz=current_xyz,
        current_edge_index=cast(Tensor, state.data.edge_index).detach().clone(),
        current_edge_attr=cast(Tensor, state.data.edge_attr).detach().clone(),
        exit_node_idx=state.exit_node_idx,
    )
    fragment_xyz = rust_product.fragment_coordinates
    fragment_features = _resize_feature_rows(fragment_features, row_count=int(fragment_xyz.shape[0]), feature_dim=base_dim)
    if base_dim > 10:
        current_x_base[:, 10] = 0.0
        fragment_features[:, 10] = 0.0
        fragment_features[-1, 10] = 1.0
    fragment_phase = _action_phase_features(
        action_space,
        int(action_idx),
        atom_count=int(fragment_features.shape[0]),
        phase_dim=phase_dim,
    )
    first_new_idx = int(current_x_base.shape[0])
    xyz = rust_product.product_coordinates
    edge_index, edge_attr, active_mask = _edge_tensors_from_rust_product_bonds(
        product_bonds=rust_product.product_bonds,
        current_edge_index=cast(Tensor, state.data.edge_index).detach().clone(),
        current_edge_attr=cast(Tensor, state.data.edge_attr).detach().clone(),
        current_active_mask=cast(Tensor, state.data.active_dendrite_mask).detach().clone(),
        xyz=xyz,
        old_exit_idx=state.exit_node_idx,
    )
    x_base = torch.cat([current_x_base, fragment_features], dim=0)
    x_phase = torch.cat([current_x_phase, fragment_phase], dim=0)
    data = Data(
        x_base=x_base,
        x_phase=x_phase,
        xyz=xyz,
        edge_index=edge_index,
        edge_attr=edge_attr,
        active_dendrite_mask=active_mask,
        num_nodes=int(x_base.shape[0]),
    )
    survivor_smiles = row.get("survivor_smiles")
    survivor_lookup_smiles = str(survivor_smiles if isinstance(survivor_smiles, str) else row.get("canonical_smiles") or "")
    anchor_id = str(row.get("anchor_id") or survivor_lookup_smiles or state.anchor_id)
    history_step = AssemblyHistoryStep(
        step=int(step),
        synthon_id=anchor_id,
        exit_atom_idx=state.exit_node_idx,
        action_idx=int(action_idx),
        node_count=int(x_base.shape[0]),
    )
    product_history = (*state.history, history_step)
    canonical_smiles = canonical_smiles_from_product_graph(data, product_history)
    return AssembledState(
        scaffold=state.scaffold,
        data=data,
        exit_node_idx=int(x_base.shape[0]) - 1,
        history=product_history,
        last_growth_action_idx=int(action_idx),
        canonical_smiles=canonical_smiles,
        anchor_id=anchor_id,
        coordinates_json=_tensor_to_coordinates_json(xyz),
        score_atom_offset=state.scaffold.retained_atom_count,
        survivor_lookup_smiles=survivor_lookup_smiles,
    )


def rollout_forward_action_mask(
    action_space: ActionSpace,
    *,
    active_rows: Tensor,
    step: int,
    max_steps: int,
) -> Tensor:
    if active_rows.ndim != 1 or active_rows.dtype != torch.bool:
        raise ValueError("active_rows must be a bool tensor with shape [batch_size]")
    if not bool(action_space.valid_mask.any().item()):
        raise ValueError("rollout requires at least one valid growth action")
    batch_size = int(active_rows.shape[0])
    action_count = int(action_space.valid_mask.shape[0])
    action_mask = action_space.valid_mask.unsqueeze(0).expand(batch_size, -1).clone()
    stop_mask = torch.zeros((batch_size, 1), dtype=torch.bool)
    if max_steps > 1 and step > 0:
        stop_mask[active_rows] = True
    if max_steps > 1 and step == max_steps - 1:
        action_mask[active_rows] = False
        stop_mask[active_rows] = True
    action_mask[~active_rows] = False
    stop_mask[~active_rows] = True
    mask = torch.cat([action_mask, stop_mask], dim=1)
    dead_rows = ~mask.any(dim=1)
    if bool(dead_rows.any().item()):
        mask[dead_rows, action_count] = True
    return mask


def apply_stop_horizon_prior(output: Any, forward_mask: Tensor, *, action_count: int, step: int, max_steps: int) -> Any:
    """Make the learned STOP head selectable under large action spaces without replacing it."""

    if max_steps <= 1 or step <= 0 or step >= max_steps - 1:
        return output
    if int(output.forward_logits.shape[1]) != action_count + 1:
        return output
    stop_valid = forward_mask[:, action_count]
    if not bool(stop_valid.any().item()):
        return output
    adjusted_logits = output.forward_logits.clone()
    # With tens of thousands of synthons, an untrained one-logit STOP head has
    # vanishing sampling mass. A horizon prior keeps termination exploratory
    # while the STOP logit remains the learned row-specific component.
    adjusted_logits[stop_valid, action_count] += math.log(float(max(2, action_count)))
    floor = torch.finfo(adjusted_logits.dtype).min
    adjusted_log_probs = torch.log_softmax(adjusted_logits.masked_fill(~forward_mask, floor), dim=1)
    return replace(
        output,
        forward_logits=adjusted_logits,
        forward_log_probs=adjusted_log_probs,
        forward_probs=adjusted_log_probs.exp(),
    )


def product_fiber_batch_telemetry(
    *,
    action_space: ActionSpace,
    actions: Sequence[int],
    selected_graphs: Sequence[ScaffoldGraph],
    fiber_lookup: FieldLookup | None,
) -> dict[str, int | str | float]:
    """Compute direct product fiber tensors for selected terminal products."""

    if fiber_lookup is None:
        return {
            "product_fiber_method": "unavailable",
            "product_fiber_products": 0,
            "product_fiber_synthon_atoms": 0,
            "shear_mean": 0.0,
            "hysteresis_mean": 0.0,
            "reversibility_mean": 1.0,
            "pathway_voxels_occupied": 0.0,
            "pathway_neighborhood_contacts": 0.0,
            "consensus_complement_bonus": 0.0,
        }
    rows = action_space.table.to_dicts()
    synthon_atoms = 0
    estimated = 0
    stats_rows: list[dict[str, float]] = []
    for action_idx, graph in zip(actions, selected_graphs, strict=True):
        coords = _coordinates_from_row(rows[int(action_idx)])
        if coords is None:
            continue
        n_scaffold = min(graph.retained_atom_count, coords.shape[0])
        fiber_lookup.lookup_product_fiber(
            cast(Tensor, graph.data.x_phase),
            coords,
            n_scaffold=n_scaffold,
        )
        stats_rows.append(fiber_lookup.field_stats_for_coordinates(coords, n_scaffold=n_scaffold))
        synthon_atoms += max(0, int(coords.shape[0]) - n_scaffold)
        estimated += 1
    return {
        "product_fiber_method": "full_thermodynamic_field_stack" if isinstance(fiber_lookup, ThermodynamicFieldStack) else "direct_signal_grid_lookup",
        "product_fiber_products": estimated,
        "product_fiber_synthon_atoms": synthon_atoms,
        "shear_mean": _mean([row["sigma_shear_mean"] for row in stats_rows]),
        "hysteresis_mean": _mean([row["hysteresis_mean"] for row in stats_rows]),
        "pathway_voxels_occupied": _mean([row["pathway_voxels_occupied"] for row in stats_rows]),
        "pathway_neighborhood_contacts": _mean([row["pathway_neighborhood_contacts"] for row in stats_rows]),
    }


def product_fiber_state_telemetry(
    *,
    states: Sequence[AssembledState],
    fiber_lookup: FieldLookup | None,
) -> dict[str, int | str | float]:
    """Compute product-fiber telemetry from assembled multi-step product coordinates."""

    if fiber_lookup is None:
        return {
            "product_fiber_method": "unavailable",
            "product_fiber_products": 0,
            "product_fiber_synthon_atoms": 0,
            "shear_mean": 0.0,
            "hysteresis_mean": 0.0,
            "reversibility_mean": 1.0,
            "pathway_voxels_occupied": 0.0,
            "pathway_neighborhood_contacts": 0.0,
            "consensus_complement_bonus": 0.0,
        }
    synthon_atoms = 0
    estimated = 0
    stats_rows: list[dict[str, float]] = []
    for state in states:
        coords = _coordinates_from_row({"coordinates_json": state.coordinates_json})
        if coords is None:
            continue
        n_scaffold = min(state.scaffold.retained_atom_count, coords.shape[0])
        fiber_lookup.lookup_product_fiber(
            cast(Tensor, state.scaffold.data.x_phase),
            coords,
            n_scaffold=n_scaffold,
        )
        stats_rows.append(fiber_lookup.field_stats_for_coordinates(coords, n_scaffold=n_scaffold))
        synthon_atoms += max(0, int(coords.shape[0]) - n_scaffold)
        estimated += 1
    return {
        "product_fiber_method": "full_thermodynamic_field_stack" if isinstance(fiber_lookup, ThermodynamicFieldStack) else "direct_signal_grid_lookup",
        "product_fiber_products": estimated,
        "product_fiber_synthon_atoms": synthon_atoms,
        "shear_mean": _mean([row["sigma_shear_mean"] for row in stats_rows]),
        "hysteresis_mean": _mean([row["hysteresis_mean"] for row in stats_rows]),
        "pathway_voxels_occupied": _mean([row["pathway_voxels_occupied"] for row in stats_rows]),
        "pathway_neighborhood_contacts": _mean([row["pathway_neighborhood_contacts"] for row in stats_rows]),
    }


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
    if "partial_charges_json" in table.columns and "partial_charge_mean_abs" not in table.columns:
        charge_stats = [_partial_charge_stats(raw) for raw in table.get_column("partial_charges_json").to_list()]
        table = table.with_columns(
            pl.Series("partial_charge_mean_abs", [stats["mean_abs"] for stats in charge_stats]),
            pl.Series("partial_charge_sum", [stats["sum"] for stats in charge_stats]),
            pl.Series("partial_charge_span", [stats["span"] for stats in charge_stats]),
        )
    for column in ("partial_charge_mean_abs", "partial_charge_sum", "partial_charge_span"):
        if column not in table.columns:
            table = table.with_columns(pl.lit(0.0).alias(column))
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
        "partial_charge_mean_abs",
        "partial_charge_sum",
        "partial_charge_span",
        "steric_volume_A3",
        "bbox_x_A",
        "bbox_y_A",
        "bbox_z_A",
        "mmff_energy_kcal_mol",
    ]
    if table.height == 0:
        raise ValueError("action table must contain at least one row")
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
    std = raw.std(dim=0, keepdim=True, unbiased=False).clamp_min(1.0e-6)
    normalized = (raw - mean) / std
    repeats = math.ceil(embedding_dim / int(normalized.shape[1]))
    expanded = normalized.repeat(1, repeats)[:, :embedding_dim]
    return expanded


def action_base_features_from_table(table: pl.DataFrame, base_feature_dim: int = 13) -> Tensor:
    """Build action-conditioned product atom features aligned to ``x_base``.

    Synthon atoms do not exist in the current scaffold graph until an action is
    selected. This per-action tensor carries the candidate product's synthon
    atom summary, including AM1-BCC/NAGL charge statistics in the same final
    slot used by v2 node features, into anchor attention before sampling.
    """

    if table.height == 0:
        raise ValueError("action table must contain at least one row")
    rows: list[list[float]] = []
    for row in table.to_dicts():
        features = [0.0] * base_feature_dim
        features[0] = numeric_value(row.get("n_heavy_atoms"), 0.0)
        if base_feature_dim > 1:
            features[1] = numeric_value(row.get("formal_charge"), 0.0)
        if base_feature_dim > 2:
            features[2] = numeric_value(row.get("partial_charge_sum"), 0.0)
        if base_feature_dim > 3:
            features[3] = numeric_value(row.get("partial_charge_span"), 0.0)
        features[-1] = numeric_value(row.get("partial_charge_mean_abs", row.get("charge_feature_mean")), 0.0)
        rows.append(features)
    tensor = torch.tensor(rows, dtype=torch.float32)
    if tensor.numel() == 0:
        return tensor
    if base_feature_dim > 1:
        scaled = tensor[:, :-1]
        mean = scaled.mean(dim=0, keepdim=True)
        std = scaled.std(dim=0, keepdim=True, unbiased=False).clamp_min(1.0e-6)
        tensor[:, :-1] = (scaled - mean) / std
    return tensor


def action_atom_features_from_table(table: pl.DataFrame, base_feature_dim: int = 13) -> tuple[Tensor, Tensor]:
    """Build per-action atom features from synthon conformer atoms and AM1-BCC charges."""

    parsed_rows: list[list[dict[str, object]]] = []
    max_atoms = 1
    for row in table.to_dicts():
        parsed_atoms: list[dict[str, object]] = []
        parsed_charges: list[object] = []
        raw_charges = row.get("partial_charges_json")
        if isinstance(raw_charges, str) and raw_charges.strip():
            try:
                raw_parsed_charges = json.loads(raw_charges)
            except json.JSONDecodeError:
                raw_parsed_charges = []
            if isinstance(raw_parsed_charges, list):
                parsed_charges = raw_parsed_charges
        raw_atoms = row.get("conformer_atoms_json")
        if isinstance(raw_atoms, str) and raw_atoms.strip():
            try:
                raw_parsed = json.loads(raw_atoms)
            except json.JSONDecodeError:
                raw_parsed = []
            if isinstance(raw_parsed, list):
                parsed_atoms = [item for item in raw_parsed if isinstance(item, dict)]
        if not parsed_atoms:
            parsed_atoms = [
                {"atomic_num": 0, "partial_charge": value}
                for value in parsed_charges
            ]
        else:
            for atom_index, atom in enumerate(parsed_atoms):
                if "partial_charge" not in atom and atom_index < len(parsed_charges):
                    atom["partial_charge"] = parsed_charges[atom_index]
        parsed_rows.append(parsed_atoms)
        max_atoms = max(max_atoms, len(parsed_atoms))

    features = torch.zeros((len(parsed_rows), max_atoms, base_feature_dim), dtype=torch.float32)
    mask = torch.zeros((len(parsed_rows), max_atoms), dtype=torch.bool)
    periodic_table = Chem.GetPeriodicTable()
    for row_idx, atoms in enumerate(parsed_rows):
        for atom_idx, atom in enumerate(atoms):
            if atom_idx >= max_atoms:
                break
            atomic_num = int(numeric_value(atom.get("atomic_num"), 0.0))
            partial_charge = numeric_value(atom.get("partial_charge"), 0.0)
            if atomic_num > 0:
                features[row_idx, atom_idx, 0] = atomic_num / 100.0
                features[row_idx, atom_idx, 8] = float(periodic_table.GetAtomicWeight(atomic_num)) / 250.0
            features[row_idx, atom_idx, 9] = partial_charge
            if base_feature_dim > 10:
                features[row_idx, atom_idx, 10] = 0.0
            if base_feature_dim > 11:
                features[row_idx, atom_idx, 11] = 1.0
            features[row_idx, atom_idx, -1] = partial_charge
            mask[row_idx, atom_idx] = True
    return features, mask


def _partial_charge_stats(raw: object) -> dict[str, float]:
    if not isinstance(raw, str) or not raw:
        return {"mean_abs": 0.0, "sum": 0.0, "span": 0.0}
    try:
        parsed = json.loads(raw)
    except (TypeError, ValueError, json.JSONDecodeError):
        return {"mean_abs": 0.0, "sum": 0.0, "span": 0.0}
    if not isinstance(parsed, list):
        return {"mean_abs": 0.0, "sum": 0.0, "span": 0.0}
    values: list[float] = []
    for value in parsed:
        try:
            number = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(number):
            values.append(number)
    if not values:
        return {"mean_abs": 0.0, "sum": 0.0, "span": 0.0}
    return {
        "mean_abs": sum(abs(value) for value in values) / len(values),
        "sum": sum(values),
        "span": max(values) - min(values),
    }


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
    model: torch.nn.Module,
    graph: ScaffoldGraph | Sequence[ScaffoldGraph],
    action_space: ActionSpace,
    batch_size: int,
    selected_graphs: Sequence[ScaffoldGraph] | None = None,
    state_graphs: Sequence[AssembledState] | None = None,
    forward_action_mask: Tensor | None = None,
) -> tuple[Any, Batch, Tensor]:
    if state_graphs is None:
        batch, exit_indices = clone_graph_batch(graph, batch_size, selected_graphs)
    else:
        if len(state_graphs) != batch_size:
            raise ValueError("state_graphs length must equal batch_size")
        batch = Batch.from_data_list([state.data.clone() for state in state_graphs])
        ptr = cast(Tensor, batch.ptr)
        exit_offsets = torch.tensor([state.exit_node_idx for state in state_graphs], dtype=torch.long)
        exit_indices = ptr[:-1].to(dtype=torch.long) + exit_offsets
    base_forward_mask = action_space.valid_mask.unsqueeze(0).expand(batch_size, -1)
    if forward_action_mask is None:
        forward_mask = base_forward_mask
    else:
        action_count = int(action_space.valid_mask.shape[0])
        if forward_action_mask.ndim != 2 or int(forward_action_mask.shape[0]) != batch_size:
            raise ValueError("forward_action_mask must have shape [batch_size, num_actions]")
        if forward_action_mask.dtype != torch.bool:
            raise TypeError("forward_action_mask must have dtype torch.bool")
        if int(forward_action_mask.shape[1]) == action_count:
            forward_mask = base_forward_mask & forward_action_mask
        elif int(forward_action_mask.shape[1]) == action_count + 1:
            stop_column = forward_action_mask[:, action_count:]
            forward_mask = torch.cat([base_forward_mask & forward_action_mask[:, :action_count], stop_column], dim=1)
        else:
            raise ValueError("forward_action_mask width must match action count or action count + STOP")
    if action_space.exit_ray_adjustments is not None:
        ray_base = torch.isfinite(action_space.exit_ray_adjustments).unsqueeze(0).expand(batch_size, -1)
        if int(forward_mask.shape[1]) == int(ray_base.shape[1]) + 1:
            ray_finite = torch.cat([ray_base, torch.ones((batch_size, 1), dtype=torch.bool)], dim=1)
        else:
            ray_finite = ray_base.expand_as(forward_mask)
        row_mask = forward_mask & ray_finite
        dead_rows = ~row_mask.any(dim=1)
        if bool(dead_rows.any().item()):
            row_mask[dead_rows] = forward_mask[dead_rows]
        forward_mask = row_mask
    backward_mask = torch.ones((batch_size, 1), dtype=torch.bool)
    policy_kwargs: dict[str, Tensor] = {
        "x_base": cast(Tensor, batch.x_base),
        "x_phase": cast(Tensor, batch.x_phase),
        "edge_index": cast(Tensor, batch.edge_index),
        "edge_attr": cast(Tensor, batch.edge_attr),
        "active_dendrite_mask": cast(Tensor, batch.active_dendrite_mask),
        "batch_index": cast(Tensor, batch.batch),
        "exit_node_indices": exit_indices,
        "forward_action_mask": forward_mask,
        "backward_action_mask": backward_mask,
    }
    if isinstance(model, FieldConditionedDualChannelGFlowNetPolicy) and action_space.action_phase_features is not None:
        policy_kwargs["action_phase_features"] = action_space.action_phase_features
    if isinstance(model, FieldConditionedDualChannelGFlowNetPolicy) and action_space.action_base_features is not None:
        policy_kwargs["action_base_features"] = action_space.action_base_features
    if isinstance(model, FieldConditionedDualChannelGFlowNetPolicy) and action_space.action_atom_features is not None:
        policy_kwargs["action_atom_features"] = action_space.action_atom_features
    if isinstance(model, FieldConditionedDualChannelGFlowNetPolicy) and action_space.action_atom_mask is not None:
        policy_kwargs["action_atom_mask"] = action_space.action_atom_mask
    output = model(**policy_kwargs)
    if action_space.exit_ray_adjustments is not None:
        base_adjustments = action_space.exit_ray_adjustments.to(dtype=output.forward_logits.dtype).unsqueeze(0).expand(
            batch_size, -1
        )
        if int(output.forward_logits.shape[1]) == int(base_adjustments.shape[1]) + 1:
            adjustments = torch.cat([base_adjustments, base_adjustments.new_zeros((batch_size, 1))], dim=1)
        else:
            adjustments = base_adjustments.expand_as(output.forward_logits)
        adjusted_logits = output.forward_logits + adjustments
        floor = torch.finfo(adjusted_logits.dtype).min
        adjusted_log_probs = torch.log_softmax(adjusted_logits.masked_fill(~forward_mask, floor), dim=1)
        output = replace(
            output,
            forward_logits=adjusted_logits,
            forward_log_probs=adjusted_log_probs,
            forward_probs=adjusted_log_probs.exp(),
        )
    return output, batch, exit_indices


def state_with_conditioned_phase(state: AssembledState, conditioned_phase: Tensor) -> AssembledState:
    """Return a product state whose phase tensor was produced by step conditioning."""

    phase = conditioned_phase.detach().to(dtype=torch.float32)
    node_count = int(cast(Tensor, state.data.x_base).shape[0])
    expected_dim = int(cast(Tensor, state.data.x_phase).shape[2])
    if phase.ndim != 3 or int(phase.shape[0]) != node_count or int(phase.shape[1]) != 5:
        raise ValueError(
            "conditioned phase tensor must have shape [N_state_atoms, 5, D]; "
            f"got {list(phase.shape)} for N_state_atoms={node_count}"
        )
    if int(phase.shape[2]) != expected_dim:
        raise ValueError(
            "conditioned phase feature dimension does not match current state; "
            f"got {int(phase.shape[2])}, expected {expected_dim}"
        )
    data = state.data.clone()
    data.x_phase = phase
    data.num_nodes = node_count
    return replace(state, data=data)


def condition_states_for_step(
    states: Sequence[AssembledState],
    conditioners: Sequence[FieldConditioner],
    *,
    step: int,
    active_rows: Tensor | None = None,
) -> tuple[list[AssembledState], int, tuple[tuple[int, int, int, int], ...]]:
    """Apply step-aware field conditioning to each current product state."""

    if len(states) != len(conditioners):
        raise ValueError("states and conditioners must have matching length")
    if step < 0:
        raise ValueError("step must be non-negative")
    if active_rows is not None:
        if active_rows.ndim != 1 or active_rows.dtype != torch.bool:
            raise ValueError("active_rows must be a 1D bool tensor")
        if int(active_rows.shape[0]) != len(states):
            raise ValueError("active_rows length must match states")
    conditioned_states: list[AssembledState] = []
    shape_events: list[tuple[int, int, int, int]] = []
    call_count = 0
    for row_idx, (state, conditioner) in enumerate(zip(states, conditioners, strict=True)):
        if active_rows is not None and not bool(active_rows[row_idx].item()):
            conditioned_states.append(state)
            continue
        if step <= 0:
            conditioned = conditioner.condition_at_step(0)
        else:
            conditioned = conditioner.condition_at_step(
                int(step),
                current_product_xyz=cast(Tensor, state.data.xyz),
                n_scaffold=state.scaffold.retained_atom_count,
            )
        conditioned_states.append(state_with_conditioned_phase(state, conditioned))
        shape_events.append((row_idx, int(step), int(conditioned.shape[0]), int(conditioned.shape[2])))
        call_count += 1
    return conditioned_states, call_count, tuple(shape_events)


def sample_multistep_rollout(
    *,
    model: torch.nn.Module,
    scaffold_pool: Sequence[ScaffoldGraph],
    action_space: ActionSpace,
    selected_graphs: Sequence[ScaffoldGraph],
    batch_size: int,
    max_steps: int,
    sampling_temperature: float,
    fiber_lookup: FieldLookup | None = None,
) -> MultiStepRollout:
    """Sample a variable-length trajectory where each step consumes the prior product graph."""

    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    if len(selected_graphs) != batch_size:
        raise ValueError("selected_graphs length must equal batch_size")
    step_count = max(1, int(max_steps))
    stop_action_idx = int(action_space.valid_mask.shape[0])
    states = [initial_assembled_state(graph) for graph in selected_graphs]
    conditioners = [
        FieldConditioner(
            cast(Tensor, graph.data.x_phase),
            cast(Tensor, graph.data.xyz),
            fiber_lookup=fiber_lookup,
        )
        for graph in selected_graphs
    ]
    active = torch.ones((batch_size,), dtype=torch.bool)
    trajectory_lengths = torch.zeros((batch_size,), dtype=torch.long)
    history_actions = torch.full((batch_size, step_count), stop_action_idx, dtype=torch.long)
    forward_outputs: list[Any] = []
    forward_steps: list[Tensor] = []
    backward_steps: list[Tensor] = []
    mask_steps: list[Tensor] = []
    node_count_events: list[tuple[int, int, int, int]] = []
    conditioner_call_counts = [0 for _ in range(step_count)]
    conditioner_shape_events: list[tuple[int, int, int, int]] = []

    for step in range(step_count):
        states, call_count, shape_events = condition_states_for_step(
            states,
            conditioners,
            step=step,
            active_rows=active,
        )
        conditioner_call_counts[step] += call_count
        conditioner_shape_events.extend(shape_events)
        step_active = active.clone()
        forward_mask = rollout_forward_action_mask(
            action_space,
            active_rows=step_active,
            step=step,
            max_steps=step_count,
        )
        output, _, _ = forward_policy(
            model,
            scaffold_pool,
            action_space,
            batch_size,
            selected_graphs=selected_graphs,
            state_graphs=states,
            forward_action_mask=forward_mask,
        )
        output = apply_stop_horizon_prior(
            output,
            forward_mask,
            action_count=stop_action_idx,
            step=step,
            max_steps=step_count,
        )
        actions_step = sample_actions_per_row(
            output.forward_log_probs,
            forward_mask,
            min(sampling_temperature, 1.0) if bool(forward_mask[:, stop_action_idx].any().item()) else sampling_temperature,
            require_unique=False,
        )
        history_actions[:, step] = actions_step
        forward_selected = selected_forward_log_probs(output.forward_log_probs, actions_step)
        backward_selected = output.backward_log_probs[:, 0]
        forward_steps.append(torch.where(step_active, forward_selected, torch.zeros_like(forward_selected)))
        backward_steps.append(torch.where(step_active, backward_selected, torch.zeros_like(backward_selected)))
        mask_steps.append(step_active)
        forward_outputs.append(output)

        next_states = list(states)
        for row_idx, action_idx_value in enumerate(actions_step.tolist()):
            if not bool(step_active[row_idx].item()):
                continue
            action_idx = int(action_idx_value)
            if action_idx == stop_action_idx:
                active[row_idx] = False
                if int(trajectory_lengths[row_idx].item()) == 0:
                    trajectory_lengths[row_idx] = step + 1
                continue
            before_count = _state_node_count(states[row_idx])
            grown_state = grow_state_with_action(
                states[row_idx],
                action_space=action_space,
                action_idx=action_idx,
                step=step,
            )
            after_count = _state_node_count(grown_state)
            node_count_events.append((row_idx, step, before_count, after_count))
            next_states[row_idx] = grown_state
        states = next_states

    still_active = trajectory_lengths == 0
    if bool(still_active.any().item()):
        trajectory_lengths[still_active] = step_count
    trajectory_mask = torch.stack(mask_steps, dim=1)
    terminal_actions = torch.tensor(
        [
            stop_action_idx if state.last_growth_action_idx is None else int(state.last_growth_action_idx)
            for state in states
        ],
        dtype=torch.long,
    )
    node_counts = [_state_node_count(state) for state in states]
    return MultiStepRollout(
        forward_outputs=tuple(forward_outputs),
        terminal_states=tuple(states),
        terminal_actions=terminal_actions,
        history_actions=history_actions,
        forward_log_probs=torch.stack(forward_steps, dim=1),
        backward_log_probs=torch.stack(backward_steps, dim=1),
        trajectory_lengths=trajectory_lengths,
        trajectory_mask=trajectory_mask,
        assembly_histories=tuple(state.history for state in states),
        assembled_state_node_count_mean=float(sum(node_counts) / len(node_counts)) if node_counts else 0.0,
        node_count_events=tuple(node_count_events),
        field_conditioner_call_counts=tuple(conditioner_call_counts),
        field_conditioner_shape_events=tuple(conditioner_shape_events),
    )


def proposals_for_actions(action_space: ActionSpace, actions: Sequence[int]) -> list[OracleProposal]:
    table_rows = action_space.table.to_dicts()
    proposals: list[OracleProposal] = []
    for batch_idx, action_idx in enumerate(actions):
        row = table_rows[int(action_idx)]
        survivor_smiles = row.get("survivor_smiles")
        canonical_smiles = survivor_smiles if isinstance(survivor_smiles, str) else row["canonical_smiles"]
        coordinates = str(row.get("coordinates_json") or "")
        score_atom_offset = 0
        try:
            total_atoms = len(json.loads(coordinates)) if coordinates else 0
        except (TypeError, ValueError, json.JSONDecodeError):
            total_atoms = 0
        atoms_scored = numeric_value(row.get("n_heavy_atoms"), 0.0)
        if atoms_scored <= 0.0:
            atoms_scored = numeric_value(row.get("population_consensus_atoms_scored"), 0.0)
        if total_atoms > 0 and atoms_scored > 0.0:
            score_atom_offset = max(0, total_atoms - int(atoms_scored))
        proposals.append(
            OracleProposal(
                anchor_id=str(row["anchor_id"]),
                canonical_smiles=str(canonical_smiles),
                trajectory_id=f"tb-{batch_idx:06d}",
                coordinates_json=coordinates,
                score_atom_offset=score_atom_offset,
            )
        )
    return proposals


def proposals_for_terminal_states(states: Sequence[AssembledState], *, oracle_mode: str) -> list[OracleProposal]:
    proposals: list[OracleProposal] = []
    for batch_idx, state in enumerate(states):
        if state.last_growth_action_idx is None:
            raise RuntimeError("terminal state has no growth action before STOP")
        canonical_smiles = state.canonical_smiles
        score_atom_offset = state.score_atom_offset
        if oracle_mode == "survivor_lookup":
            canonical_smiles = state.survivor_lookup_smiles or canonical_smiles
        elif oracle_mode == "live_signal_grid":
            score_atom_offset = 0
        proposals.append(
            OracleProposal(
                anchor_id=state.anchor_id,
                canonical_smiles=canonical_smiles,
                trajectory_id=f"tb-{batch_idx:06d}",
                coordinates_json=state.coordinates_json,
                score_atom_offset=score_atom_offset,
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


def enrich_oracle_rows_with_action_fields(
    action_space: ActionSpace,
    actions: Sequence[int],
    oracle_rows: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    """Merge selected action-level field-stack metrics into oracle result rows."""

    table_rows = action_space.table.to_dicts()
    enriched: list[dict[str, object]] = []
    passthrough_columns = (
        "sigma_shear_mean",
        "hysteresis_mean",
        "reversibility_mean",
        "pathway_voxels_occupied",
        "pathway_neighborhood_contacts",
        "pathway_neighborhood_score_mean",
        "pathway_score_mean",
        "species_conservation_score_mean",
        "field_consensus_complement_bonus",
        "charge_feature_mean",
        "u_pose",
        "u_pose_source",
    )
    for action_idx, row in zip(actions, oracle_rows, strict=True):
        merged = dict(row)
        action_row = table_rows[int(action_idx)]
        for column in passthrough_columns:
            if column in action_row:
                merged[column] = action_row[column]
        if "consensus_complement_bonus" not in merged or numeric_value(merged.get("consensus_complement_bonus"), 0.0) == 0.0:
            merged["consensus_complement_bonus"] = (
                action_row.get("consensus_complement_bonus")
                or action_row.get("population_consensus_bonus")
                or action_row.get("population_consensus_bonus_scaled")
                or action_row.get("field_consensus_complement_bonus")
                or merged.get("consensus_complement_bonus", 0.0)
            )
        if "sigma_shear" not in merged:
            merged["sigma_shear"] = action_row.get("sigma_shear_mean", 0.0)
        enriched.append(merged)
    return enriched


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
    model: torch.nn.Module,
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
    if bool(args.live_scoring):
        args.oracle_mode = "live_signal_grid"
    if int(args.max_trajectory_steps) > 1 and str(args.oracle_mode) == "survivor_lookup":
        args.oracle_mode = "live_signal_grid"
        print(
            "oracle_mode_auto_selected "
            "reason=multi_step_rollout_requires_live_scoring selected=live_signal_grid",
            flush=True,
        )
    telemetry_handle = None
    original_stdout = sys.stdout
    original_stderr = sys.stderr
    if args.telemetry_log is not None:
        telemetry_path = Path(args.telemetry_log)
        if not telemetry_path.is_absolute():
            telemetry_path = REPO_ROOT / telemetry_path
        telemetry_path.parent.mkdir(parents=True, exist_ok=True)
        telemetry_handle = telemetry_path.open("a", encoding="utf-8")
        sys.stdout = TelemetryTee(sys.stdout, telemetry_handle)
        sys.stderr = TelemetryTee(sys.stderr, telemetry_handle)
    torch.set_num_threads(max(1, int(args.torch_threads)))
    torch.set_num_interop_threads(max(1, min(4, int(args.torch_threads))))
    set_seed(int(args.seed))
    anchors_path = resolve_track_a_path(cast(Path | None, args.synthon_parquet), Path(args.anchors))
    survivors_path = resolve_survivor_corpus_for_reward(
        requested_survivors=Path(args.survivors),
        reward_version=str(args.reward_version),
        signal_grid=cast(Path | None, args.signal_grid),
    )
    if survivors_path != Path(args.survivors):
        print(
            "survivor_corpus_auto_selected "
            f"requested={Path(args.survivors).as_posix()} selected={survivors_path.as_posix()} "
            f"reward_version={args.reward_version}",
            flush=True,
        )
    paths = TrainingPaths(
        ligand_sdf=Path(args.ligand_sdf),
        anchors=anchors_path,
        survivors=survivors_path,
        residue_phase=Path(args.residue_phase),
        interferometric=Path(args.interferometric),
        topology=Path(args.topology),
        fragment_registry=Path(args.fragment_registry),
        output_dir=Path(args.output_dir),
    )
    paths.output_dir.mkdir(parents=True, exist_ok=True)
    fiber_lookup = build_field_lookup(
        field_stack_version=str(args.field_stack_version),
        signal_grid=cast(Path | None, args.signal_grid),
        grid_mapping=Path(args.grid_coordinate_mapping),
        shear_stress=cast(Path | None, args.shear_stress),
        hysteresis_tensor=cast(Path | None, args.hysteresis_tensor),
        translation_pathway=cast(Path | None, args.translation_pathway),
        cross_species=cast(Path | None, args.cross_species),
    )
    scaffold_sdfs = scaffold_paths_from_args(cast(Sequence[Path] | None, args.scaffold_pool), paths.ligand_sdf)
    scaffold_pool = build_scaffold_pool(
        paths,
        scaffold_sdfs,
        field_stack_version=str(args.field_stack_version),
        fiber_lookup=fiber_lookup,
    )
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
    action_space = load_action_space(
        paths,
        int(args.embedding_dim),
        base_feature_dim=reference_graph.base_feature_dim,
    )
    action_space = attach_exit_ray_adjustments(
        action_space,
        reference_graph=reference_graph,
        fiber_lookup=fiber_lookup,
        enabled=not bool(args.disable_exit_ray_masks),
    )
    if str(args.oracle_mode) not in {"survivor_lookup", "live_signal_grid"}:
        raise RuntimeError(f"unsupported oracle_mode={args.oracle_mode}")
    model: torch.nn.Module
    if bool(args.dual_channel):
        model = FieldConditionedDualChannelGFlowNetPolicy(
            base_feature_dim=reference_graph.base_feature_dim,
            phase_feature_dim=reference_graph.phase_feature_dim,
            edge_feature_dim=reference_graph.edge_feature_dim,
            anchor_embeddings=action_space.anchor_embeddings,
            hidden_dim=int(args.hidden_dim),
            embedding_dim=int(args.embedding_dim),
            learn_anchor_embeddings=True,
        )
        architecture_name = "FieldConditionedDualChannelGFlowNetPolicy"
    else:
        model = FiberBundleGFlowNetPolicy(
            base_feature_dim=reference_graph.base_feature_dim,
            phase_feature_dim=reference_graph.phase_feature_dim,
            edge_feature_dim=reference_graph.edge_feature_dim,
            anchor_embeddings=action_space.anchor_embeddings,
            hidden_dim=int(args.hidden_dim),
            embedding_dim=int(args.embedding_dim),
            learn_anchor_embeddings=True,
            rf_mode=str(args.rf_mode),
        )
        architecture_name = "FiberBundleGFlowNetPolicy"
    print(
        f"policy_instantiated type={architecture_name} dual_channel={bool(args.dual_channel)} "
        f"rf_mode={args.rf_mode} oracle_mode={args.oracle_mode}",
        flush=True,
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
    oracle_args: list[str] = []
    lock_mask = resolve_track_a_path(cast(Path | None, args.lock_mask), TRACK_A_DIR / "lock_region_mask.json")
    if lock_mask.is_file():
        oracle_args.extend(["--lock-mask", str(lock_mask)])
    oracle: LiveSignalGridOracle | SurvivorCorpusOracle
    if str(args.oracle_mode) == "live_signal_grid":
        oracle = LiveSignalGridOracle(
            survivor_corpus=paths.survivors,
            max_batch_size=int(args.batch_size),
            signal_grid=resolve_track_a_path(cast(Path | None, args.signal_grid), DEFAULT_POPULATION_CONSENSUS_GRID),
            grid_config=resolve_track_a_path(Path(args.grid_coordinate_mapping), DEFAULT_GRID_MAPPING),
            shear_stress=resolve_track_a_path(cast(Path | None, args.shear_stress), N80_DIR / "shear_stress_field.parquet"),
            lock_mask=lock_mask if lock_mask.is_file() else None,
        )
    else:
        oracle = SurvivorCorpusOracle(
            survivor_corpus=paths.survivors,
            max_batch_size=int(args.batch_size),
            extra_args=tuple(oracle_args),
        )

    config = {
        "architecture": architecture_name,
        "dual_channel": bool(args.dual_channel),
        "rf_mode": str(args.rf_mode),
        "oracle_mode": str(args.oracle_mode),
        "grid_coordinate_mapping": str(args.grid_coordinate_mapping),
        "exit_ray_masks_enabled": action_space.exit_ray_adjustments is not None,
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
        "field_stack_version": str(args.field_stack_version),
        "lock_directional_bias_alpha": float(args.lock_directional_bias_alpha),
        "lock_reaching_synthon_boost": float(args.lock_reaching_synthon_boost),
        "lock_geo_intrinsic_bonus": float(args.lock_geo_intrinsic_bonus),
        "consensus_bonus_weight": float(args.consensus_bonus_weight),
        "lock_mask": lock_mask.as_posix(),
        "signal_grid": str(args.signal_grid) if args.signal_grid is not None else "",
        "shear_stress": str(args.shear_stress) if args.shear_stress is not None else "",
        "hysteresis_tensor": str(args.hysteresis_tensor) if args.hysteresis_tensor is not None else "",
        "translation_pathway": str(args.translation_pathway) if args.translation_pathway is not None else "",
        "cross_species": str(args.cross_species) if args.cross_species is not None else "",
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
        "- Reward authority: Rust `oracle_scorer` survivor-corpus lookup; Python only selects versioned reward components.\n"
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
        target = reward_targets_for_policy_output(action_space, log_probs)
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
    checkpoint_dir = Path(args.checkpoint_dir)
    if not checkpoint_dir.is_absolute():
        checkpoint_dir = REPO_ROOT / checkpoint_dir
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    if args.resume is not None:
        resume_path = Path(args.resume)
        checkpoint = torch.load(resume_path, map_location="cpu", weights_only=False)
        if not isinstance(checkpoint, dict):
            raise RuntimeError(f"resume checkpoint must contain a dict: {resume_path}")
        state_dict = checkpoint.get("model_state_dict")
        if not isinstance(state_dict, dict):
            raise RuntimeError(f"resume checkpoint missing model_state_dict: {resume_path}")
        model.load_state_dict(state_dict)
        optimizer_state = checkpoint.get("optimizer_state_dict")
        if isinstance(optimizer_state, dict):
            optimizer.load_state_dict(optimizer_state)
        print(f"training_resume checkpoint={resume_path} prior_epoch={checkpoint.get('epoch', 0)}", flush=True)
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
        progress = 0.0 if int(args.epochs) <= 1 else float(epoch - 1) / float(int(args.epochs) - 1)
        sampling_temperature = 5.0 - (4.90 * progress)
        rollout = sample_multistep_rollout(
            model=model,
            scaffold_pool=scaffold_pool,
            action_space=action_space,
            selected_graphs=selected_graphs,
            batch_size=int(args.batch_size),
            max_steps=int(args.max_trajectory_steps),
            sampling_temperature=sampling_temperature,
            fiber_lookup=fiber_lookup,
        )
        output = rollout.forward_outputs[-1]
        actions_tensor = rollout.terminal_actions
        actions = [int(value) for value in actions_tensor.tolist()]
        product_fiber_metrics = product_fiber_state_telemetry(
            states=rollout.terminal_states,
            fiber_lookup=fiber_lookup,
        )
        proposals = proposals_for_terminal_states(rollout.terminal_states, oracle_mode=str(args.oracle_mode))
        reaction_classes = reaction_classes_for_actions(action_space, actions)
        oracle_result = await oracle.score_batch(proposals)
        rewards = oracle_result.rewards
        oracle_row_dicts = enrich_oracle_rows_with_action_fields(
            action_space,
            actions,
            cast(list[dict[str, object]], oracle_result.rows.to_dicts()),
        )
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
        versioned_rewards, reward_component_metrics = compute_effective_reward_tensor(
            reward_version=str(args.reward_version),
            oracle_rows=oracle_row_dicts,
            raw_rewards=rewards,
            tripartite_scores=tripartite_scores,
            consensus_bonus_weight=float(args.consensus_bonus_weight),
        )
        scaled_rewards = versioned_rewards * float(args.reward_scale)
        current_temperature = compute_adaptive_temperature(scaled_rewards, target_entropy=2.0)
        tempered_rewards = (scaled_rewards / current_temperature).clamp_min(1.0e-8)
        proposal_smiles = [proposal.canonical_smiles for proposal in proposals]
        diversity_bonus = compute_diversity_bonus(
            proposal_smiles,
            seen_smiles_counts,
            beta=float(args.diversity_beta),
        )
        lock_intrinsic = torch.tensor(
            [numeric_value(row.get("lock_geometry_score", row.get("pi_clash_lock", 0.0)), 0.0) for row in oracle_row_dicts],
            dtype=torch.float32,
        )
        effective_rewards = (
            tempered_rewards
            + diversity_bonus
            + float(args.lock_geo_intrinsic_bonus) * lock_intrinsic
        ).clamp_min(1.0e-8)
        tb = tb_loss(
            output.log_z,
            rollout.forward_log_probs,
            rollout.backward_log_probs,
            effective_rewards,
            trajectory_mask=rollout.trajectory_mask,
        )
        policy_aux_terms = []
        for step_output in rollout.forward_outputs:
            target = reward_targets_for_policy_output(action_space, step_output.forward_log_probs)
            policy_aux_terms.append(-(target * step_output.forward_log_probs).sum(dim=1).mean())
        policy_aux_loss = torch.stack(policy_aux_terms).mean()
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
        effective_reward_mean = float(versioned_rewards.mean().item())
        effective_reward_std = float(versioned_rewards.std(unbiased=False).item())
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
        trajectory_length_mean = float(rollout.trajectory_lengths.float().mean().item())
        trajectory_length_max = int(rollout.trajectory_lengths.max().item())
        trajectories_ge2 = int((rollout.trajectory_lengths >= 2).sum().item())
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
                "effective_reward_mean": effective_reward_mean,
                "effective_reward_std": effective_reward_std,
                "consensus_bonus_mean": reward_component_metrics["consensus_bonus_mean"],
                "consensus_bonus_std": reward_component_metrics["consensus_bonus_std"],
                "pi_complement_std": reward_component_metrics["pi_complement_std"],
                "pi_clash_pocket_std": reward_component_metrics["pi_clash_pocket_std"],
                "shear_mean": reward_component_metrics["shear_mean"],
                "hysteresis_mean": reward_component_metrics["hysteresis_mean"],
                "pathway_voxels_occupied": reward_component_metrics["pathway_voxels_occupied"],
                "pathway_neighborhood_contacts": reward_component_metrics["pathway_neighborhood_contacts"],
                "pathway_score_mean": reward_component_metrics["pathway_score_mean"],
                "charge_feature_mean": reward_component_metrics["charge_feature_mean"],
                "u_pose_mean": reward_component_metrics["u_pose_mean"],
                "top_reward_seen": top_reward_seen,
                "unique_smiles_generated": len(generated_smiles),
                "diversity_bonus_mean": float(diversity_bonus.mean().item()),
                "temperature": current_temperature,
                "lock_geo_positive": lock_geo_positive,
                "lock_proj_gt_05": lock_proj_gt_05,
                "trajectory_length_mean": trajectory_length_mean,
                "trajectory_length_max": trajectory_length_max,
                "trajectories_ge2": trajectories_ge2,
                "assembled_state_node_count_mean": rollout.assembled_state_node_count_mean,
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
            "effective_reward_mean": f"{effective_reward_mean:.6f}",
            "effective_reward_std": f"{effective_reward_std:.6f}",
            "consensus_bonus_mean": f"{reward_component_metrics['consensus_bonus_mean']:.6f}",
            "shear_mean": f"{reward_component_metrics['shear_mean']:.6f}",
            "hysteresis_mean": f"{reward_component_metrics['hysteresis_mean']:.6f}",
            "pathway_voxels_occupied": f"{reward_component_metrics['pathway_voxels_occupied']:.6f}",
            "pathway_neighborhood_contacts": f"{reward_component_metrics['pathway_neighborhood_contacts']:.6f}",
            "pathway_score_mean": f"{reward_component_metrics['pathway_score_mean']:.6f}",
            "charge_feature_mean": f"{reward_component_metrics['charge_feature_mean']:.6f}",
            "u_pose_mean": f"{reward_component_metrics['u_pose_mean']:.6f}",
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
            "product_fiber_method": product_fiber_metrics["product_fiber_method"],
            "product_fiber_synthon_atoms": product_fiber_metrics["product_fiber_synthon_atoms"],
            "product_fiber_shear_mean": f"{float(product_fiber_metrics['shear_mean']):.6f}",
            "product_fiber_hysteresis_mean": f"{float(product_fiber_metrics['hysteresis_mean']):.6f}",
            "product_fiber_pathway_voxels": f"{float(product_fiber_metrics['pathway_voxels_occupied']):.6f}",
            "product_fiber_pathway_neighborhood_contacts": f"{float(product_fiber_metrics['pathway_neighborhood_contacts']):.6f}",
            "trajectory_length_mean": f"{trajectory_length_mean:.2f}",
            "trajectory_length_max": trajectory_length_max,
            "trajectories_ge2": f"{trajectories_ge2}/{int(args.batch_size)}",
            "assembled_state_node_count_mean": f"{rollout.assembled_state_node_count_mean:.2f}",
        }
        print("tb_epoch_complete " + " ".join(f"{key}={value}" for key, value in telemetry.items()), flush=True)
        for row_idx, step_idx, before_count, after_count in rollout.node_count_events:
            print(
                "assembled_state_node_count "
                f"epoch={epoch} row={row_idx} step={step_idx} before={before_count} count={after_count}",
                flush=True,
            )
        call_parts = [
            f"field_conditioner_calls_step{step_idx}={call_count}"
            for step_idx, call_count in enumerate(rollout.field_conditioner_call_counts)
        ]
        print(
            "field_conditioner_calls "
            f"epoch={epoch} "
            + " ".join(call_parts),
            flush=True,
        )
        for row_idx, step_idx, atoms, phase_dim in rollout.field_conditioner_shape_events:
            print(
                "field_conditioner_product_fiber_shape "
                f"epoch={epoch} row={row_idx} step={step_idx} shape=[{atoms},5,{phase_dim}] "
                f"atoms={atoms} phases=5 dim={phase_dim}",
                flush=True,
            )
        print(
            "product_fiber_estimated "
            f"epoch={epoch} products={product_fiber_metrics.get('product_fiber_products', 0)} "
            f"n_synthon_atoms={product_fiber_metrics['product_fiber_synthon_atoms']} "
            f"method={product_fiber_metrics['product_fiber_method']}",
            flush=True,
        )
        print(
            "reward_component_variance "
            f"epoch={epoch} "
            f"pi_complement_std={reward_component_metrics['pi_complement_std']:.6f} "
            f"pi_clash_pocket_std={reward_component_metrics['pi_clash_pocket_std']:.6f} "
            f"consensus_bonus_std={reward_component_metrics['consensus_bonus_std']:.6f}",
            flush=True,
        )
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
        if epoch % max(1, int(args.checkpoint_interval)) == 0:
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

    model_path = Path(args.output_policy)
    if not model_path.is_absolute():
        model_path = paths.output_dir / model_path if model_path.parent == Path(".") else REPO_ROOT / model_path
    model_path.parent.mkdir(parents=True, exist_ok=True)
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
            "effective_reward_mean",
            "effective_reward_std",
            "consensus_bonus_mean",
            "consensus_bonus_std",
            "pi_complement_std",
            "pi_clash_pocket_std",
            "shear_mean",
            "hysteresis_mean",
            "pathway_voxels_occupied",
            "pathway_neighborhood_contacts",
            "pathway_score_mean",
            "charge_feature_mean",
            "u_pose_mean",
            "top_reward_seen",
            "unique_smiles_generated",
            "diversity_bonus_mean",
            "temperature",
            "lock_geo_positive",
            "lock_proj_gt_05",
            "trajectory_length_mean",
            "trajectory_length_max",
            "trajectories_ge2",
            "assembled_state_node_count_mean",
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
        "- Reward authority: Rust `oracle_scorer` via `SurvivorCorpusOracle` survivor lookup.\n"
    )
    print(
        "gflownet_training_complete "
        f"epoch1_tb_loss={first_loss:.6f} epoch{final_epoch}_tb_loss={final_loss:.6f} "
        f"reward_mean={first_reward:.6f}->{final_reward:.6f} "
        f"top_reward_seen={top_reward_seen:.6f} unique_smiles={len(generated_smiles)} "
        f"dot_smiles_count={dot_smiles_total} top_anchor_share={top_anchor_share:.6f} "
        f"validation_status={validation['validation_status']} model={model_path}"
    )
    if telemetry_handle is not None:
        sys.stdout = original_stdout
        sys.stderr = original_stderr
        telemetry_handle.close()


if __name__ == "__main__":
    asyncio.run(train())
