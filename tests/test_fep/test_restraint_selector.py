"""Tests for restraint_selector.py — Boresch restraint selection."""
from __future__ import annotations

import math

import pytest

from scripts.fep.restraint_selector import (
    AtomInfo,
    BoreshRestraint,
    _angle,
    _collinearity_penalty,
    _dist,
    score_restraint,
    select_boresch_restraint,
    select_ligand_atoms,
    select_protein_atoms,
)


class TestGeometryHelpers:
    def test_dist_zero(self):
        assert _dist((0, 0, 0), (0, 0, 0)) == 0.0

    def test_dist_unit(self):
        assert abs(_dist((0, 0, 0), (1, 0, 0)) - 1.0) < 1e-9

    def test_dist_3d(self):
        d = _dist((1, 2, 3), (4, 6, 3))
        assert abs(d - 5.0) < 1e-9

    def test_angle_90_degrees(self):
        angle = _angle((1, 0, 0), (0, 0, 0), (0, 1, 0))
        assert abs(angle - math.pi / 2) < 1e-9

    def test_angle_180_degrees(self):
        angle = _angle((-1, 0, 0), (0, 0, 0), (1, 0, 0))
        assert abs(angle - math.pi) < 1e-9

    def test_collinearity_penalty_ideal(self):
        assert _collinearity_penalty(math.pi / 2) == 0.0

    def test_collinearity_penalty_near_zero(self):
        p = _collinearity_penalty(0.1)
        assert p > 0

    def test_collinearity_penalty_near_pi(self):
        p = _collinearity_penalty(3.0)
        assert p > 0


class TestSelectProteinAtoms:
    def test_basic_selection(self, mock_protein_atoms):
        selected = select_protein_atoms(mock_protein_atoms, [100, 101, 102, 103, 104], n_candidates=3)
        assert len(selected) == 3
        # Should be sorted by B-factor (lowest first)
        assert selected[0].b_factor <= selected[1].b_factor

    def test_fewer_lining_residues(self, mock_protein_atoms):
        # Only 2 lining residues — should fall back to all CA atoms
        selected = select_protein_atoms(mock_protein_atoms, [100, 101], n_candidates=3)
        assert len(selected) == 3

    def test_insufficient_atoms_raises(self):
        atoms = [AtomInfo(0, "CA", "ALA", 100, (0, 0, 0))]
        with pytest.raises(ValueError, match="Need at least 3"):
            select_protein_atoms(atoms, [100], n_candidates=3)


class TestSelectLigandAtoms:
    def test_basic_selection(self, mock_ligand_atoms, spike_positions):
        selected = select_ligand_atoms(mock_ligand_atoms, spike_positions, n_candidates=3)
        assert len(selected) == 3
        # First selected should be nearest to first spike
        first_spike = spike_positions[0]
        first_dist = _dist(selected[0].position, first_spike)
        # It should be close to the first spike position
        assert first_dist < 3.0

    def test_insufficient_atoms_raises(self):
        atoms = [AtomInfo(0, "C1", "LIG", 1, (0, 0, 0))]
        with pytest.raises(ValueError, match="Need at least 3"):
            select_ligand_atoms(atoms, [(0, 0, 0)], n_candidates=3)


class TestScoreRestraint:
    def test_returns_boresch_restraint(self, mock_protein_atoms, mock_ligand_atoms):
        p = mock_protein_atoms[:3]
        l = mock_ligand_atoms[:3]
        result = score_restraint(p, l)
        assert isinstance(result, BoreshRestraint)
        assert result.distance_p2_l0 > 0
        assert result.score >= 0

    def test_to_dict(self, mock_protein_atoms, mock_ligand_atoms):
        result = score_restraint(mock_protein_atoms[:3], mock_ligand_atoms[:3])
        d = result.to_dict()
        assert "protein_atoms" in d
        assert "ligand_atoms" in d
        assert "score" in d
        assert len(d["protein_atoms"]) == 3
        assert len(d["ligand_atoms"]) == 3


class TestSelectBoreshRestraint:
    def test_full_selection(self, mock_protein_atoms, mock_ligand_atoms, spike_positions):
        restraint = select_boresch_restraint(
            mock_protein_atoms,
            mock_ligand_atoms,
            [100, 101, 102, 103, 104],
            spike_positions,
            n_protein_candidates=4,
            n_ligand_candidates=4,
        )
        assert isinstance(restraint, BoreshRestraint)
        assert len(restraint.protein_atoms) == 3
        assert len(restraint.ligand_atoms) == 3
        assert restraint.score >= 0

    def test_indices_property(self, mock_protein_atoms, mock_ligand_atoms, spike_positions):
        restraint = select_boresch_restraint(
            mock_protein_atoms,
            mock_ligand_atoms,
            [100, 101, 102],
            spike_positions,
            n_protein_candidates=3,
            n_ligand_candidates=3,
        )
        assert len(restraint.protein_indices) == 3
        assert len(restraint.ligand_indices) == 3
