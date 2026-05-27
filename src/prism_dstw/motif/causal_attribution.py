"""Causal attribution motif extraction via integrated gradients."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Protocol, Sequence, cast

import numpy as np
import torch
from rdkit import Chem
from torch import Tensor, nn

from prism_dstw.motif.functional_groups import get_k_hop_neighborhood, subgraph_to_smarts


class FiberBatch(Protocol):
    """Minimal batch protocol required for CAME."""

    x_phase: Tensor
    mol: Chem.Mol
    trajectory_step: int
    scaffold_id: str
    reaction_class: str


@dataclass(frozen=True)
class CausalAttributionMotif:
    """Motif produced by integrated-gradient causal attribution."""

    smarts: str
    atom_indices: list[int]
    attribution_score_mean: float
    attribution_score_std: float
    trajectory_step: int
    scaffold_conditional: str
    reaction_class: str
    frequency: int
    causal_direction: str


def compute_causal_attribution(
    policy: nn.Module,
    batch: FiberBatch,
    target_action_idx: int,
    *,
    n_steps: int = 50,
    checkpoint_paths: Sequence[str] | None = None,
    forward_for_action: Callable[[nn.Module, FiberBatch, Tensor, int], Tensor] | None = None,
) -> dict[int, tuple[float, float]]:
    """Compute per-atom integrated-gradient attribution.

    This is a direct integrated-gradient implementation. It intentionally does
    not read attention weights.
    """

    if n_steps < 2:
        raise ValueError("n_steps must be >= 2")
    forward = forward_for_action or policy_forward_for_action
    original_state = {
        name: tensor.detach().clone()
        for name, tensor in policy.state_dict().items()
    }
    policy_states: list[dict[str, Tensor] | None] = [None]
    if checkpoint_paths:
        for checkpoint_path in checkpoint_paths[1:]:
            loaded = torch.load(Path(checkpoint_path), map_location=batch.x_phase.device, weights_only=False)
            if isinstance(loaded, dict) and "model_state_dict" in loaded and isinstance(loaded["model_state_dict"], dict):
                policy_states.append(cast(dict[str, Tensor], loaded["model_state_dict"]))
            elif isinstance(loaded, dict):
                policy_states.append(cast(dict[str, Tensor], loaded))

    all_signed_scores: list[Tensor] = []
    for candidate_state in policy_states:
        if candidate_state is not None:
            policy.load_state_dict(candidate_state)
        policy.eval()
        input_x = batch.x_phase.detach()
        baseline = torch.zeros_like(input_x)
        delta = input_x - baseline
        accumulated = torch.zeros_like(input_x)
        for alpha in torch.linspace(0.0, 1.0, steps=n_steps, device=input_x.device, dtype=input_x.dtype):
            interpolated = (baseline + alpha * delta).detach().requires_grad_(True)
            score = forward(policy, batch, interpolated, target_action_idx)
            scalar = score.sum()
            grad = torch.autograd.grad(scalar, interpolated, retain_graph=False, create_graph=False)[0]
            accumulated = accumulated + grad
        avg_grad = accumulated / float(n_steps)
        attribution = delta * avg_grad
        reduce_dims = tuple(range(1, attribution.dim()))
        all_signed_scores.append(attribution.sum(dim=reduce_dims).detach())
    policy.load_state_dict(original_state)

    stacked = torch.stack(all_signed_scores)
    mean = stacked.mean(dim=0)
    std = stacked.std(dim=0, unbiased=False) if stacked.shape[0] > 1 else torch.zeros_like(mean)
    return {
        int(atom_idx): (float(mean[atom_idx].item()), float(std[atom_idx].item()))
        for atom_idx in range(int(mean.shape[0]))
    }


def policy_forward_for_action(policy: nn.Module, batch: FiberBatch, x_phase: Tensor, target_action_idx: int) -> Tensor:
    """Default policy adapter returning a target action logit."""

    output = policy(x_phase)
    if isinstance(output, dict):
        logits = cast(Tensor, output["forward_logits"])
    else:
        logits = cast(Tensor, output)
    if logits.dim() == 1:
        return logits[target_action_idx].reshape(1)
    return logits[:, target_action_idx]


def extract_causal_motifs(
    policy: nn.Module,
    candidates: Sequence[FiberBatch],
    chosen_actions: Sequence[int],
    *,
    threshold_sigma: float = 1.0,
    k_hop: int = 2,
    n_steps: int = 50,
    forward_for_action: Callable[[nn.Module, FiberBatch, Tensor, int], Tensor] | None = None,
) -> list[CausalAttributionMotif]:
    """Extract hotspot-neighborhood motifs from integrated gradients."""

    if len(candidates) != len(chosen_actions):
        raise ValueError("candidates and chosen_actions length mismatch")
    motifs: list[CausalAttributionMotif] = []
    for batch, action in zip(candidates, chosen_actions):
        attr_map = compute_causal_attribution(
            policy,
            batch,
            int(action),
            n_steps=n_steps,
            forward_for_action=forward_for_action,
        )
        scores = np.array([abs(value[0]) for value in attr_map.values()], dtype=np.float64)
        if scores.size == 0:
            continue
        threshold = float(scores.mean() + threshold_sigma * (scores.std() + 1.0e-8))
        hotspot_atoms = [idx for idx, (score, _) in attr_map.items() if abs(score) > threshold]
        if not hotspot_atoms:
            continue
        neighborhood: set[int] = set()
        for atom_idx in hotspot_atoms:
            neighborhood |= get_k_hop_neighborhood(batch.mol, atom_idx, k_hop)
        smarts = subgraph_to_smarts(batch.mol, neighborhood)
        motif_scores = [abs(attr_map[idx][0]) for idx in neighborhood]
        motif_stds = [attr_map[idx][1] for idx in neighborhood]
        motifs.append(
            CausalAttributionMotif(
                smarts=smarts,
                atom_indices=sorted(neighborhood),
                attribution_score_mean=float(np.mean(motif_scores)),
                attribution_score_std=float(np.mean(motif_stds)),
                trajectory_step=int(batch.trajectory_step),
                scaffold_conditional=str(batch.scaffold_id),
                reaction_class=str(batch.reaction_class),
                frequency=1,
                causal_direction=classify_direction(attr_map, hotspot_atoms),
            )
        )
    return deduplicate_causal_motifs(motifs)


def classify_direction(attr_map: dict[int, tuple[float, float]], hotspot_atoms: Sequence[int]) -> str:
    """Classify causal direction from attribution signs."""

    values = [attr_map[idx][0] for idx in hotspot_atoms]
    if not values:
        return "MIXED"
    positive = sum(1 for value in values if value > 0.0)
    if positive == len(values):
        return "PROMOTES_LOCK"
    if positive == 0:
        return "PROMOTES_COMPLEMENT"
    return "MIXED"


def deduplicate_causal_motifs(motifs: Sequence[CausalAttributionMotif]) -> list[CausalAttributionMotif]:
    """Merge identical SMARTS motifs."""

    merged: dict[str, CausalAttributionMotif] = {}
    for motif in motifs:
        prior = merged.get(motif.smarts)
        if prior is None:
            merged[motif.smarts] = motif
        else:
            merged[motif.smarts] = CausalAttributionMotif(
                smarts=motif.smarts,
                atom_indices=sorted(set(prior.atom_indices) | set(motif.atom_indices)),
                attribution_score_mean=(prior.attribution_score_mean * prior.frequency + motif.attribution_score_mean)
                / float(prior.frequency + 1),
                attribution_score_std=max(prior.attribution_score_std, motif.attribution_score_std),
                trajectory_step=min(prior.trajectory_step, motif.trajectory_step),
                scaffold_conditional=prior.scaffold_conditional,
                reaction_class=prior.reaction_class,
                frequency=prior.frequency + 1,
                causal_direction=prior.causal_direction if prior.causal_direction == motif.causal_direction else "MIXED",
            )
    return list(merged.values())
