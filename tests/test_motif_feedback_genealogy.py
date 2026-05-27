from __future__ import annotations

import numpy as np
from rdkit import Chem

from prism_dstw.motif.feedback import (
    compute_motif_action_bias,
    compute_motif_bonus,
    compute_motif_diversity_penalty,
)
from prism_dstw.motif.genealogy import compute_motif_receptor_heatmap, is_parent_motif
from prism_dstw.motif.registry import MotifEntry, MotifRegistry, motif_id_for_smarts


def _entry(smarts: str, *, occurrences: int = 1, lock: float = 1.0, enrichment: float = 2.0) -> MotifEntry:
    return MotifEntry(
        motif_id=motif_id_for_smarts(smarts),
        canonical_smarts=smarts,
        discovery_method="SAD",
        thermodynamic_role="LOCK_WEDGE",
        lock_geometry_contribution=lock,
        consensus_resilience=0.5,
        hysteresis_score=0.1,
        n_occurrences_top100=occurrences,
        enrichment_ratio=enrichment,
        exit_vector_preference={1: 1.0},
    ).with_completeness()


def test_motif_bonus_decays_below_ten_percent_at_high_frequency(tmp_path) -> None:
    mol = Chem.MolFromSmiles("CCO")
    assert mol is not None
    low = MotifRegistry(tmp_path / "low.parquet")
    high = MotifRegistry(tmp_path / "high.parquet")
    low.register(_entry("[#6]-[#8]", occurrences=1))
    high.register(_entry("[#6]-[#8]", occurrences=50))

    low_bonus = compute_motif_bonus(mol, low, bonus_weight=1.0, decay_lambda=0.05)
    high_bonus = compute_motif_bonus(mol, high, bonus_weight=1.0, decay_lambda=0.05)
    assert high_bonus < low_bonus * 0.10


def test_motif_action_bias_requires_exit_vector_toward_lock(tmp_path) -> None:
    registry = MotifRegistry(tmp_path / "motifs.parquet")
    registry.register(_entry("[#6]-[#8]", occurrences=1))
    mol = Chem.MolFromSmiles("CCO")
    assert mol is not None
    xyz = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]])

    toward = compute_motif_action_bias(
        mol,
        xyz,
        "scaffold",
        1,
        1,
        np.array([2.0, 0.0, 0.0]),
        np.array([4.0, 0.0, 0.0]),
        ["CCO"],
        registry,
    )
    away = compute_motif_action_bias(
        mol,
        xyz,
        "scaffold",
        1,
        1,
        np.array([2.0, 0.0, 0.0]),
        np.array([-4.0, 0.0, 0.0]),
        ["CCO"],
        registry,
    )
    assert toward[0] > 0.0
    assert away[0] == 0.0


def test_diversity_penalty_activates_on_collapsed_motif_sets(tmp_path) -> None:
    registry = MotifRegistry(tmp_path / "motifs.parquet")
    registry.register(_entry("[#6]-[#8]", occurrences=1))
    mols = [Chem.MolFromSmiles("CCO"), Chem.MolFromSmiles("CCO")]
    assert all(mol is not None for mol in mols)
    penalty = compute_motif_diversity_penalty([mol for mol in mols if mol is not None], registry)
    assert penalty < 0.0


def test_genealogy_parent_requires_overlap_and_higher_lock() -> None:
    parent = _entry("[#6]-[#6]", lock=0.2)
    child = _entry("[#6]-[#6]-[#6]", lock=0.4)
    lateral = _entry("[#6]-[#6]-[#6]", lock=0.1)

    assert is_parent_motif(parent, child)
    assert not is_parent_motif(parent, lateral)


def test_receptor_heatmap_spatial_consistency_for_repeated_voxel() -> None:
    motif = _entry("[#6]-[#8]", lock=0.3)
    mol = Chem.MolFromSmiles("CCO")
    assert mol is not None
    coords = np.array([[9.0, 9.0, 9.0], [9.0, 9.0, 9.0], [1.2, 1.2, 1.2]])
    counts, consistency = compute_motif_receptor_heatmap(
        motif,
        [mol, mol],
        [coords, coords],
        {"origin": [0.0, 0.0, 0.0], "spacing": 1.0, "dims": [3, 3, 3]},
    )
    assert counts
    assert consistency == 0.5
