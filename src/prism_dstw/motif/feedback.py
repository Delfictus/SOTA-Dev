"""Motif-conditioned generative feedback functions."""

from __future__ import annotations

import math
from typing import Sequence

import numpy as np
from rdkit import Chem

from prism_dstw.motif.registry import MotifRegistry


def compute_motif_bonus(
    product_mol: Chem.Mol,
    motif_registry: MotifRegistry,
    *,
    bonus_weight: float = 1.0,
    decay_lambda: float = 0.05,
) -> float:
    """Return frequency-decayed reward bonus for high-value motifs."""

    bonus = 0.0
    for entry in motif_registry.query_lock_enriched(min_enrichment=1.5):
        pattern = Chem.MolFromSmarts(entry.canonical_smarts)
        if pattern is None or not product_mol.HasSubstructMatch(pattern):
            continue
        frequency_decay = math.exp(-decay_lambda * float(entry.n_occurrences_top100))
        thermodynamic_weight = (
            float(entry.lock_geometry_contribution or 0.0) * 0.4
            + float(entry.consensus_resilience or 0.0) * 0.3
            + float(entry.hysteresis_score or 0.0) * 0.2
            + frequency_decay * 0.1
        )
        bonus += bonus_weight * thermodynamic_weight * frequency_decay
    return bonus


def compute_motif_diversity_penalty(batch_mols: Sequence[Chem.Mol], motif_registry: MotifRegistry) -> float:
    """Penalize batches with collapsed motif sets."""

    batch_sets: list[set[str]] = []
    lock_entries = motif_registry.query_by_role("LOCK_WEDGE")
    for mol in batch_mols:
        motif_ids: set[str] = set()
        for entry in lock_entries:
            pattern = Chem.MolFromSmarts(entry.canonical_smarts)
            if pattern is not None and mol.HasSubstructMatch(pattern):
                motif_ids.add(entry.motif_id)
        batch_sets.append(motif_ids)
    if len(batch_sets) < 2:
        return 0.0
    total = 0.0
    pairs = 0
    for i, left in enumerate(batch_sets):
        for right in batch_sets[i + 1 :]:
            union = left | right
            if not union:
                continue
            total += 1.0 - len(left & right) / float(len(union))
            pairs += 1
    if pairs == 0:
        return 0.0
    mean_diversity = total / max(float(pairs), 1.0)
    if mean_diversity < 0.3:
        return -2.0 * (0.3 - mean_diversity)
    return 0.0


def compute_motif_action_bias(
    current_product: Chem.Mol,
    current_product_xyz: np.ndarray,
    scaffold_id: str,
    trajectory_step: int,
    exit_vector_idx: int,
    exit_vector_xyz: np.ndarray,
    lock_region_centroid: np.ndarray,
    available_synthons: Sequence[str],
    motif_registry: MotifRegistry,
) -> np.ndarray:
    """Compute exit-vector-conditioned synthon motif bias."""

    del current_product, scaffold_id, trajectory_step
    biases = np.zeros(len(available_synthons), dtype=np.float64)
    if current_product_xyz.size == 0:
        return biases
    product_centroid = current_product_xyz.mean(axis=0)
    exit_direction = _unit(exit_vector_xyz - product_centroid)
    lock_direction = _unit(lock_region_centroid - exit_vector_xyz)
    directional_alignment = float(np.dot(exit_direction, lock_direction))
    if directional_alignment <= 0.0:
        return biases
    enriched = motif_registry.query_lock_enriched(min_enrichment=1.5)
    patterns = [
        (Chem.MolFromSmarts(entry.canonical_smarts), entry)
        for entry in enriched
        if entry.canonical_smarts
    ]
    for idx, synthon_smi in enumerate(available_synthons):
        synthon_mol = Chem.MolFromSmiles(synthon_smi)
        if synthon_mol is None:
            continue
        for pattern, entry in patterns:
            if pattern is None or not synthon_mol.HasSubstructMatch(pattern):
                continue
            base_bias = math.log(max(float(entry.enrichment_ratio or 1.01), 1.01))
            ev_pref = 0.5
            if entry.exit_vector_preference is not None:
                ev_pref = float(entry.exit_vector_preference.get(exit_vector_idx, 0.5))
            freq_decay = math.exp(-0.05 * float(entry.n_occurrences_top100))
            biases[idx] += base_bias * max(directional_alignment, 0.0) * ev_pref * freq_decay
            break
    return biases


def _unit(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm <= 1.0e-8:
        return np.zeros_like(vector, dtype=np.float64)
    return np.asarray(vector, dtype=np.float64) / norm
