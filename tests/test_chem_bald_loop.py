from __future__ import annotations

import json
import importlib.util
import math
from dataclasses import dataclass
from importlib import import_module
from pathlib import Path
from typing import Any, Mapping, Sequence, cast

import polars as pl
import torch
from torch import Tensor

from prism_dstw.orchestration.gflownet_action_space import (
    BoundaryClass,
    RewardWeights,
    align_fragment_to_scaffold_exit_vector,
    load_dynamic_alignment_reference,
    saturated_reward,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
TRACK_A_DIR = REPO_ROOT / "campaigns/glp1r_aleniglipron/track_a_generative"
CALIBRATION_ANCHORS = TRACK_A_DIR / "calibration_anchors_3d.parquet"
REWARD_BOUNDARIES = TRACK_A_DIR / "gflownet_tso_bridge_boundaries.parquet"
DYNAMIC_ALIGNMENT_REFERENCE = TRACK_A_DIR / "dynamic_alignment_reference.json"
LATENT_DIM = 8
NEIGHBOR_COUNT = 10
TOP_BATCH_SIZE = 5


@dataclass(frozen=True)
class Candidate:
    candidate_id: str
    anchor_id: str
    canonical_smiles: str
    latent_z: Tensor
    boundary_class: BoundaryClass
    variance_class: str
    pi_complement: float
    pi_clash: float
    sigma_shear: float
    u_pose: float
    sa_score: float
    crosses_bridge_no_fly_zone: bool


@dataclass(frozen=True)
class NeighborBatch:
    indices: Tensor
    distances: Tensor
    durability_scores: Tensor


@dataclass(frozen=True)
class Posterior:
    mean: Tensor
    variance: Tensor


class PyTorchExactStore:
    def __init__(self, dim: int, backend_label: str = "torch_exact") -> None:
        self._dim = dim
        self._backend_label = backend_label
        self._vectors: Tensor | None = None
        self._durability_scores: Tensor | None = None

    @property
    def backend(self) -> str:
        return self._backend_label

    def add(self, vectors: Tensor, durability_scores: Tensor) -> None:
        if vectors.ndim != 2 or int(vectors.shape[1]) != self._dim:
            raise ValueError(f"vectors must have shape [N, {self._dim}]")
        if durability_scores.ndim != 1 or int(durability_scores.shape[0]) != int(vectors.shape[0]):
            raise ValueError("durability_scores must have one value per vector")
        self._vectors = vectors.detach().clone().to(dtype=torch.float32)
        self._durability_scores = durability_scores.detach().clone().to(dtype=torch.float32)

    def query(self, vector: Tensor, k: int) -> NeighborBatch:
        vectors = self._vectors
        durability_scores = self._durability_scores
        if vectors is None or durability_scores is None:
            raise RuntimeError("vector store has not been populated")
        if vector.ndim != 1 or int(vector.shape[0]) != self._dim:
            raise ValueError(f"query vector must have shape [{self._dim}]")
        distances = torch.linalg.vector_norm(vectors - vector.reshape(1, self._dim), dim=1)
        nearest = torch.topk(distances, k=min(k, int(vectors.shape[0])), largest=False)
        return NeighborBatch(
            indices=nearest.indices,
            distances=nearest.values,
            durability_scores=durability_scores.index_select(0, nearest.indices),
        )


class LocalHNSWStore:
    """Deterministic local store facade with a PyTorch exact fallback."""

    def __init__(self, dim: int) -> None:
        backend_label = (
            "torch_exact_fallback_hnswlib_unavailable"
            if importlib.util.find_spec("hnswlib") is None
            else "torch_exact_deterministic_test"
        )
        self._store = PyTorchExactStore(dim=dim, backend_label=backend_label)

    @property
    def backend(self) -> str:
        return self._store.backend

    def add(self, vectors: Tensor, durability_scores: Tensor) -> None:
        self._store.add(vectors, durability_scores)

    def query(self, vector: Tensor, k: int) -> NeighborBatch:
        return self._store.query(vector, k)


class ForwardOnlyChemBALDOrchestrator:
    def __init__(self, store: LocalHNSWStore, neighbor_count: int) -> None:
        self._store = store
        self._neighbor_count = neighbor_count
        self.last_neighbor_counts: list[int] = []

    def posterior(self, candidates: Sequence[Candidate]) -> Posterior:
        means: list[Tensor] = []
        variances: list[Tensor] = []
        self.last_neighbor_counts = []
        for candidate in candidates:
            neighbors = self._store.query(candidate.latent_z, self._neighbor_count)
            self.last_neighbor_counts.append(int(neighbors.indices.numel()))
            weights = torch.softmax(-neighbors.distances, dim=0)
            mean = (weights * neighbors.durability_scores).sum()
            centered = neighbors.durability_scores - mean
            local_variance = (weights * centered.square()).sum()
            distance_uncertainty = torch.min(neighbors.distances).clamp_min(0.0) * 0.015
            pose_uncertainty = torch.tensor(candidate.u_pose * 0.04, dtype=torch.float32)
            variances.append((local_variance + distance_uncertainty + pose_uncertainty + 1.0e-4).clamp_min(1.0e-6))
            means.append(mean)
        return Posterior(mean=torch.stack(means), variance=torch.stack(variances))


def string_value(row: Mapping[str, object], key: str) -> str:
    value = row[key]
    if not isinstance(value, str):
        raise TypeError(f"{key} must be a string")
    return value


def float_value(row: Mapping[str, object], key: str) -> float:
    value = row[key]
    if isinstance(value, bool):
        raise TypeError(f"{key} must be numeric")
    if isinstance(value, int | float):
        return float(value)
    raise TypeError(f"{key} must be numeric")


def calibration_anchor_rows(count: int) -> list[dict[str, object]]:
    frame = (
        pl.scan_parquet(CALIBRATION_ANCHORS)
        .filter(pl.col("generation_status") == "ok")
        .select(["anchor_id", "canonical_smiles"])
        .head(count)
        .collect()
    )
    if frame.height < count:
        raise AssertionError(f"expected at least {count} calibration anchors, found {frame.height}")
    return cast(list[dict[str, object]], frame.to_dicts())


def no_fly_reference_row() -> dict[str, object]:
    frame = (
        pl.scan_parquet(REWARD_BOUNDARIES)
        .filter(pl.col("boundary_class") == BoundaryClass.BRIDGE_ANCHOR_NO_FLY_ZONE.value)
        .select(["condition_id", "voxel_idx", "sigma_shear", "distance_to_bridge_anchor_A"])
        .head(1)
        .collect()
    )
    if frame.height != 1:
        raise AssertionError("expected at least one bridge-anchor no-fly row in the reward landscape")
    return cast(dict[str, object], frame.row(0, named=True))


def dynamic_reference_heavy_atom_count() -> int:
    assert DYNAMIC_ALIGNMENT_REFERENCE.exists()
    reference = load_dynamic_alignment_reference(DYNAMIC_ALIGNMENT_REFERENCE)
    assert len(reference.pocket_points) >= 3
    payload = json.loads(DYNAMIC_ALIGNMENT_REFERENCE.read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    heavy_atoms = payload.get("pocket_heavy_atom_coordinates")
    assert isinstance(heavy_atoms, list)
    assert len(heavy_atoms) > len(reference.pocket_points)
    return len(heavy_atoms)


def assert_scaffold_exit_vector_alignment() -> float:
    chem = cast(Any, import_module("rdkit.Chem"))
    all_chem = cast(Any, import_module("rdkit.Chem.AllChem"))
    geometry = cast(Any, import_module("rdkit.Geometry"))
    scaffold = chem.AddHs(chem.MolFromSmiles("CC"))
    fragment = chem.MolFromSmiles("[*]C")
    assert fragment is not None
    params = all_chem.ETKDGv3()
    params.randomSeed = 917
    status = int(all_chem.EmbedMolecule(scaffold, params))
    assert status == 0
    # The fragment still contains the attachment dummy here; force fields must only
    # see the dummy-free molecule returned by the action-space aligner.
    fragment_conformer = chem.Conformer(fragment.GetNumAtoms())
    fragment_conformer.SetAtomPosition(0, geometry.Point3D(0.0, 0.0, 0.0))
    fragment_conformer.SetAtomPosition(1, geometry.Point3D(0.86, 0.0, 0.0))
    fragment.AddConformer(fragment_conformer)
    all_chem.UFFOptimizeMolecule(scaffold, maxIters=100)
    pose = align_fragment_to_scaffold_exit_vector(
        fragment,
        scaffold,
        1,
        scaffold_exit_vector=(1.0, 0.0, 0.0),
        attachment_bond_length_A=1.50,
    )
    assert all(int(atom.GetAtomicNum()) != 0 for atom in pose.aligned_mol.GetAtoms())
    scaffold_exit = scaffold.GetConformer().GetAtomPosition(1)
    fragment_attachment = pose.aligned_mol.GetConformer().GetAtomPosition(pose.fragment_attachment_atom_idx)
    dx = float(fragment_attachment.x - scaffold_exit.x)
    dy = float(fragment_attachment.y - scaffold_exit.y)
    dz = float(fragment_attachment.z - scaffold_exit.z)
    norm = math.sqrt(dx * dx + dy * dy + dz * dz)
    assert norm > 0.0
    assert math.isclose(norm, 1.50, rel_tol=0.0, abs_tol=1.0e-6)
    assert dx / norm < -0.999
    assert abs(dy) < 1.0e-6
    assert abs(dz) < 1.0e-6
    return norm


def lowest_penalty_row() -> dict[str, object]:
    frame = (
        pl.scan_parquet(REWARD_BOUNDARIES)
        .filter(pl.col("boundary_class") == BoundaryClass.PENALTY_TSO.value)
        .select(["condition_id", "voxel_idx", "pi_clash", "sigma_shear"])
        .sort(["pi_clash", "sigma_shear"])
        .head(1)
        .collect()
    )
    if frame.height != 1:
        raise AssertionError("expected at least one penalty TSO row in the reward landscape")
    return cast(dict[str, object], frame.row(0, named=True))


def make_candidate_pool() -> list[Candidate]:
    torch.manual_seed(32019)
    anchors = calibration_anchor_rows(20)
    penalty_row = lowest_penalty_row()
    no_fly_row = no_fly_reference_row()
    historical_latents = historical_vectors()
    golden_latent = historical_latents[-1] + 0.001
    poison_latent = historical_latents[-2] + 0.001
    uncertain_control_latent = historical_latents[-1] + 0.002

    golden_anchor = anchors[0]
    poison_anchor = anchors[1]
    candidates = [
        Candidate(
            candidate_id="GOLDEN_CANDIDATE",
            anchor_id=string_value(golden_anchor, "anchor_id"),
            canonical_smiles=string_value(golden_anchor, "canonical_smiles"),
            latent_z=golden_latent,
            boundary_class=BoundaryClass.REWARD_TSO,
            variance_class="thermally_activated",
            pi_complement=3.60,
            pi_clash=0.00,
            sigma_shear=max(float_value(penalty_row, "sigma_shear"), 0.01),
            u_pose=0.05,
            sa_score=1.0,
            crosses_bridge_no_fly_zone=False,
        ),
        Candidate(
            candidate_id="POISON_CANDIDATE",
            anchor_id=string_value(poison_anchor, "anchor_id"),
            canonical_smiles=string_value(poison_anchor, "canonical_smiles"),
            latent_z=poison_latent,
            boundary_class=BoundaryClass.REWARD_TSO,
            variance_class="thermally_activated",
            pi_complement=3.60,
            pi_clash=0.00,
            sigma_shear=max(float_value(no_fly_row, "sigma_shear"), 0.01),
            u_pose=0.05,
            sa_score=6.0,
            crosses_bridge_no_fly_zone=True,
        ),
        Candidate(
            candidate_id="HIGH_UPOSE_CONTROL",
            anchor_id=string_value(anchors[2], "anchor_id"),
            canonical_smiles=string_value(anchors[2], "canonical_smiles"),
            latent_z=uncertain_control_latent,
            boundary_class=BoundaryClass.REWARD_TSO,
            variance_class="thermally_activated",
            pi_complement=2.40,
            pi_clash=0.00,
            sigma_shear=max(float_value(penalty_row, "sigma_shear"), 0.01),
            u_pose=3.0,
            sa_score=1.5,
            crosses_bridge_no_fly_zone=False,
        ),
    ]
    for idx, anchor in enumerate(anchors[3:], start=3):
        latent = torch.randn(LATENT_DIM, dtype=torch.float32)
        is_no_fly = idx % 7 == 0
        candidates.append(
            Candidate(
                candidate_id=f"MOCK_CANDIDATE_{idx:02d}",
                anchor_id=string_value(anchor, "anchor_id"),
                canonical_smiles=string_value(anchor, "canonical_smiles"),
                latent_z=latent,
                boundary_class=BoundaryClass.BRIDGE_ANCHOR_NO_FLY_ZONE if is_no_fly else BoundaryClass.PENALTY_TSO,
                variance_class="stable_occupied",
                pi_complement=0.20 + 0.03 * float(idx % 5),
                pi_clash=0.45 + 0.04 * float(idx % 4),
                sigma_shear=0.20 + 0.05 * float(idx % 6),
                u_pose=0.35 + 0.04 * float(idx % 5),
                sa_score=2.0 + 0.25 * float(idx % 6),
                crosses_bridge_no_fly_zone=is_no_fly,
            )
        )
    if len(candidates) != 20:
        raise AssertionError(f"expected 20 candidates, built {len(candidates)}")
    return candidates


def historical_vectors() -> Tensor:
    generator = torch.Generator().manual_seed(9173)
    return torch.randn((50, LATENT_DIM), generator=generator, dtype=torch.float32)


def historical_durability_scores() -> Tensor:
    return torch.linspace(0.20, 1.35, 50, dtype=torch.float32)


def normal_pdf(value: Tensor) -> Tensor:
    return torch.exp(-0.5 * value.square()) / math.sqrt(2.0 * math.pi)


def normal_cdf(value: Tensor) -> Tensor:
    return 0.5 * (1.0 + torch.erf(value / math.sqrt(2.0)))


def q_expected_improvement(mean: Tensor, variance: Tensor, best_observed: float, u_pose: Tensor) -> Tensor:
    std = torch.sqrt(variance.clamp_min(1.0e-9))
    improvement = mean - best_observed
    z_score = improvement / std
    expected_improvement = improvement * normal_cdf(z_score) + std * normal_pdf(z_score)
    return expected_improvement.clamp_min(0.0) * torch.exp(-0.65 * u_pose)


def physics_reward(candidate: Candidate) -> float:
    return saturated_reward(
        pi_complement=candidate.pi_complement,
        pi_clash=candidate.pi_clash,
        sigma_shear=candidate.sigma_shear,
        u_pose=candidate.u_pose,
        weights=RewardWeights(complement=1.0, clash=1.0, shear=1.0, gamma=2.0, uncertainty_lambda=1.0),
    )


def final_reward(candidate: Candidate) -> float:
    base_reward = max(physics_reward(candidate), 0.0)
    sa_penalized = base_reward * math.exp(-candidate.sa_score)
    if candidate.crosses_bridge_no_fly_zone or candidate.boundary_class is BoundaryClass.BRIDGE_ANCHOR_NO_FLY_ZONE:
        return 0.0
    return sa_penalized


def test_chem_bald_gflownet_forward_pass_ranks_and_rejects_edge_cases() -> None:
    assert CALIBRATION_ANCHORS.exists()
    assert REWARD_BOUNDARIES.exists()
    heavy_atom_count = dynamic_reference_heavy_atom_count()
    aligned_bond_length = assert_scaffold_exit_vector_alignment()

    candidates = make_candidate_pool()
    store = LocalHNSWStore(dim=LATENT_DIM)
    store.add(historical_vectors(), historical_durability_scores())
    orchestrator = ForwardOnlyChemBALDOrchestrator(store=store, neighbor_count=NEIGHBOR_COUNT)
    posterior = orchestrator.posterior(candidates)

    assert orchestrator.last_neighbor_counts == [NEIGHBOR_COUNT] * len(candidates)
    assert posterior.mean.shape == (20,)
    assert posterior.variance.shape == (20,)
    assert bool(torch.isfinite(posterior.mean).all().item())
    assert bool(torch.isfinite(posterior.variance).all().item())

    u_pose = torch.tensor([candidate.u_pose for candidate in candidates], dtype=torch.float32)
    acquisition = q_expected_improvement(
        posterior.mean,
        posterior.variance,
        best_observed=0.85,
        u_pose=u_pose,
    )
    assert acquisition.shape == (20,)
    assert bool(torch.isfinite(acquisition).all().item())

    final_rewards = torch.tensor([final_reward(candidate) for candidate in candidates], dtype=torch.float32)
    selection_scores = final_rewards * acquisition
    top_batch_indices = torch.topk(selection_scores, k=TOP_BATCH_SIZE, largest=True).indices.tolist()
    top_batch = [candidates[int(index)].candidate_id for index in top_batch_indices]

    golden_idx = 0
    poison_idx = 1
    high_upose_idx = 2
    golden_reward = float(final_rewards[golden_idx].item())
    poison_reward = float(final_rewards[poison_idx].item())
    poison_sa_only_reward = max(physics_reward(candidates[poison_idx]), 0.0) * math.exp(-candidates[poison_idx].sa_score)

    print(f"store_backend={store.backend} neighbor_count={orchestrator.last_neighbor_counts[0]}")
    print(f"dynamic_reference_heavy_atoms={heavy_atom_count} aligned_fragment_bond_A={aligned_bond_length:.6f}")
    print(
        "golden_final_reward="
        f"{golden_reward:.8f} poison_final_reward={poison_reward:.8f} "
        f"poison_sa_only_reward={poison_sa_only_reward:.8f}"
    )
    print(
        "golden_qei="
        f"{float(acquisition[golden_idx].item()):.8f} "
        f"poison_qei={float(acquisition[poison_idx].item()):.8f} "
        f"high_upose_control_qei={float(acquisition[high_upose_idx].item()):.8f}"
    )
    print(f"top_batch={top_batch}")

    assert int(torch.argmax(final_rewards).item()) == golden_idx
    assert "GOLDEN_CANDIDATE" in top_batch
    assert "POISON_CANDIDATE" not in top_batch
    assert poison_reward <= 1.0e-8
    assert poison_sa_only_reward < golden_reward * 0.01
    assert float(acquisition[golden_idx].item()) > float(acquisition[high_upose_idx].item())
