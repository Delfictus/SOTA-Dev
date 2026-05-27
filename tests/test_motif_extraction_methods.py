from __future__ import annotations

import numpy as np
import polars as pl
import torch
from rdkit import Chem
from torch import Tensor, nn

from prism_dstw.motif.causal_attribution import FiberBatch, compute_causal_attribution, extract_causal_motifs
from prism_dstw.motif.functional_groups import AtomThermoAnnotation, classify_atom_roles, extract_tfg_with_neighborhood
from prism_dstw.motif.phase_resolved_mcs import extract_phase_resolved_mcs
from prism_dstw.motif.synthon_ancestry import compute_synthon_ancestry


class SimpleBatch:
    def __init__(self, mol: Chem.Mol, x_phase: Tensor) -> None:
        self.mol = mol
        self.x_phase = x_phase
        self.trajectory_step = 1
        self.scaffold_id = "mock_scaffold"
        self.reaction_class = "mock_reaction"


class AtomZeroPolicy(nn.Module):
    def forward(self, x_phase: Tensor) -> Tensor:
        return torch.stack([x_phase[0].sum().reshape(1), x_phase[1:].sum().reshape(1)], dim=1)


class NegativeAtomZeroPolicy(nn.Module):
    def forward(self, x_phase: Tensor) -> Tensor:
        return torch.stack([-x_phase[0].sum().reshape(1), x_phase[1:].sum().reshape(1)], dim=1)


def test_tfgd_k_hop_expansion_includes_neutral_bridge_atoms() -> None:
    mol = Chem.MolFromSmiles("CCCCC")
    assert mol is not None
    annotations = {
        idx: AtomThermoAnnotation(lock_geometry=1.0 if idx in {0, 4} else 0.0)
        for idx in range(mol.GetNumAtoms())
    }
    roles = classify_atom_roles(annotations)
    motifs = extract_tfg_with_neighborhood(mol, roles, annotations, k_hops=2)

    assert len(motifs) == 1
    assert motifs[0].role == "LOCK_WEDGE"
    assert motifs[0].bridge_atom_count >= 3
    assert motifs[0].atom_indices == [0, 1, 2, 3, 4]


def test_came_integrated_gradients_identifies_causal_atom_without_attention() -> None:
    mol = Chem.MolFromSmiles("CCO")
    assert mol is not None
    x_phase = torch.zeros((3, 5, 8), dtype=torch.float32)
    x_phase[0, :, :] = 1.0
    batch: FiberBatch = SimpleBatch(mol, x_phase)
    policy = AtomZeroPolicy()

    attribution = compute_causal_attribution(policy, batch, 0, n_steps=8)
    assert attribution[0][0] > 0.0
    assert attribution[0][0] > attribution[1][0]

    motifs = extract_causal_motifs(policy, [batch], [0], threshold_sigma=0.0, n_steps=8)
    assert motifs
    assert 0 in motifs[0].atom_indices
    assert motifs[0].causal_direction == "PROMOTES_LOCK"


def test_came_preserves_signed_direction_for_negative_attribution() -> None:
    mol = Chem.MolFromSmiles("CCO")
    assert mol is not None
    x_phase = torch.zeros((3, 5, 8), dtype=torch.float32)
    x_phase[0, :, :] = 1.0
    batch: FiberBatch = SimpleBatch(mol, x_phase)
    policy = NegativeAtomZeroPolicy()

    attribution = compute_causal_attribution(policy, batch, 0, n_steps=8)
    assert attribution[0][0] < 0.0

    motifs = extract_causal_motifs(policy, [batch], [0], threshold_sigma=0.0, n_steps=8)
    assert motifs
    assert 0 in motifs[0].atom_indices
    assert motifs[0].causal_direction == "PROMOTES_COMPLEMENT"


def test_phase_resolved_mcs_uses_prefilter_and_timeout() -> None:
    mols = [Chem.MolFromSmiles("CCOC"), Chem.MolFromSmiles("CCOCC"), Chem.MolFromSmiles("c1ccccc1")]
    assert all(mol is not None for mol in mols)
    profiles = [
        np.array([1.0, 1.2, 1.4, 1.1, 1.0]),
        np.array([1.1, 1.3, 1.5, 1.2, 1.1]),
        np.array([0.0, 0.0, 0.0, 0.0, 0.0]),
    ]
    motifs = extract_phase_resolved_mcs(
        [mol for mol in mols if mol is not None],
        profiles,
        tanimoto_threshold=0.2,
        mcs_timeout_seconds=1,
        butina_cutoff=0.8,
    )
    assert motifs
    assert motifs[0].phase_profile_centroid.shape == (5,)
    assert motifs[0].smarts


def test_synthon_ancestry_fisher_exact_enrichment() -> None:
    frame = pl.DataFrame(
        {
            "synthon_smiles": ["CCO", "CCO", "CCO", "CCO", "CCC", "CCC", "CCC", "CCC"],
            "enamine_id": ["EN-A"] * 4 + ["EN-B"] * 4,
            "lock_geometry_score": [1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0],
            "reward": [5.0, 5.2, 5.1, 5.3, 1.0, 1.1, 1.2, 1.3],
            "reaction_class": ["amide"] * 8,
            "exit_vector_idx": [2] * 8,
        }
    )

    ancestry = compute_synthon_ancestry(frame, min_occurrences=2)
    top = ancestry[0]
    assert top.synthon_smiles == "CCO"
    assert top.enrichment_ratio > 1.0
    assert top.p_value <= 1.0
    assert top.exit_vector_preference == {2: 1.0}
