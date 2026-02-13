"""Tests for EnsembleMMGBSA and InteractionEntropy (WT-9 V2)."""
from __future__ import annotations

import json
import pickle

import pytest

from scripts.interfaces.ensemble_score import EnsembleMMGBSA, InteractionEntropy


class TestEnsembleMMGBSA:
    """Tests for EnsembleMMGBSA dataclass."""

    def test_construction(self, sample_ensemble_mmgbsa):
        mg = sample_ensemble_mmgbsa
        assert mg.compound_id == "cmpd-001"
        assert mg.delta_g_mean == -8.2
        assert mg.delta_g_std == 1.4
        assert mg.delta_g_sem == 0.14
        assert mg.n_snapshots == 100
        assert mg.snapshot_interval_ps == 100.0
        assert mg.decomposition == {"vdw": -12.0, "elec": -5.0, "gb": 8.0, "sa": -1.2}
        assert mg.per_residue_contributions[142] == -2.5
        assert mg.method == "MMGBSA_ensemble"

    def test_dict_round_trip(self, sample_ensemble_mmgbsa):
        d = sample_ensemble_mmgbsa.to_dict()
        mg2 = EnsembleMMGBSA.from_dict(d)
        assert mg2.to_dict() == d

    def test_json_round_trip(self, sample_ensemble_mmgbsa):
        j = sample_ensemble_mmgbsa.to_json()
        mg2 = EnsembleMMGBSA.from_json(j)
        assert mg2.compound_id == "cmpd-001"
        assert mg2.delta_g_mean == -8.2
        assert mg2.per_residue_contributions[142] == -2.5

    def test_pickle_round_trip(self, sample_ensemble_mmgbsa):
        data = sample_ensemble_mmgbsa.to_pickle()
        mg2 = EnsembleMMGBSA.from_pickle(data)
        assert mg2.compound_id == "cmpd-001"
        assert mg2.decomposition["vdw"] == -12.0

    def test_pickle_type_check(self):
        bad = pickle.dumps(99)
        with pytest.raises(TypeError, match="Expected EnsembleMMGBSA"):
            EnsembleMMGBSA.from_pickle(bad)

    def test_int_keys_in_per_residue(self, sample_ensemble_mmgbsa):
        """per_residue_contributions int keys survive JSON round-trip."""
        j = sample_ensemble_mmgbsa.to_json()
        parsed = json.loads(j)
        # JSON keys are always strings
        assert "142" in parsed["per_residue_contributions"]
        # Round-trip restores int keys
        mg2 = EnsembleMMGBSA.from_json(j)
        assert isinstance(list(mg2.per_residue_contributions.keys())[0], int)

    def test_decomposition_keys(self, sample_ensemble_mmgbsa):
        j = sample_ensemble_mmgbsa.to_json()
        parsed = json.loads(j)
        assert set(parsed["decomposition"].keys()) == {"vdw", "elec", "gb", "sa"}

    def test_mmpbsa_method(self):
        mg = EnsembleMMGBSA(
            compound_id="cmpd-002", delta_g_mean=-6.5, delta_g_std=1.8,
            delta_g_sem=0.18, n_snapshots=100, snapshot_interval_ps=100.0,
            decomposition={"vdw": -10.0, "elec": -3.0, "gb": 6.0, "sa": -0.5},
            per_residue_contributions={100: -1.5},
            method="MMPBSA_ensemble",
        )
        j = mg.to_json()
        mg2 = EnsembleMMGBSA.from_json(j)
        assert mg2.method == "MMPBSA_ensemble"


class TestInteractionEntropy:
    """Tests for InteractionEntropy dataclass."""

    def test_construction(self, sample_interaction_entropy):
        ie = sample_interaction_entropy
        assert ie.compound_id == "cmpd-001"
        assert ie.minus_t_delta_s == -3.2
        assert ie.delta_h == -5.0
        assert ie.delta_g_ie == -8.2
        assert ie.n_frames == 100
        assert ie.convergence_block_std == 0.35

    def test_dict_round_trip(self, sample_interaction_entropy):
        d = sample_interaction_entropy.to_dict()
        ie2 = InteractionEntropy.from_dict(d)
        assert ie2.to_dict() == d

    def test_json_round_trip(self, sample_interaction_entropy):
        j = sample_interaction_entropy.to_json()
        ie2 = InteractionEntropy.from_json(j)
        assert ie2.compound_id == "cmpd-001"
        assert abs(ie2.delta_g_ie - (-8.2)) < 1e-6

    def test_pickle_round_trip(self, sample_interaction_entropy):
        data = sample_interaction_entropy.to_pickle()
        ie2 = InteractionEntropy.from_pickle(data)
        assert ie2.minus_t_delta_s == -3.2

    def test_pickle_type_check(self):
        bad = pickle.dumps("wrong")
        with pytest.raises(TypeError, match="Expected InteractionEntropy"):
            InteractionEntropy.from_pickle(bad)

    def test_delta_g_identity(self, sample_interaction_entropy):
        """delta_g_ie should equal delta_h + minus_t_delta_s."""
        ie = sample_interaction_entropy
        assert abs(ie.delta_g_ie - (ie.delta_h + ie.minus_t_delta_s)) < 1e-6
