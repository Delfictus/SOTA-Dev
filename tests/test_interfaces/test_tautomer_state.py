"""Tests for TautomerState and TautomerEnsemble (WT-9 V2)."""
from __future__ import annotations

import json
import pickle

import pytest

from scripts.interfaces.tautomer_state import TautomerState, TautomerEnsemble


class TestTautomerState:
    """Tests for TautomerState dataclass."""

    def test_construction(self, sample_tautomer_anion):
        ts = sample_tautomer_anion
        assert ts.smiles == "CC(=O)[O-]"
        assert ts.parent_smiles == "CC(=O)O"
        assert ts.protonation_ph == 7.4
        assert ts.charge == -1
        assert ts.pka_shifts == [(3, 4.76)]
        assert ts.population_fraction == 0.99
        assert ts.source_tool == "dimorphite_dl"

    def test_dict_round_trip(self, sample_tautomer_anion):
        d = sample_tautomer_anion.to_dict()
        ts2 = TautomerState.from_dict(d)
        assert ts2.to_dict() == d

    def test_json_round_trip(self, sample_tautomer_anion):
        j = sample_tautomer_anion.to_json()
        ts2 = TautomerState.from_json(j)
        assert ts2.smiles == "CC(=O)[O-]"
        assert ts2.charge == -1
        assert ts2.pka_shifts == [(3, 4.76)]

    def test_pickle_round_trip(self, sample_tautomer_anion):
        data = sample_tautomer_anion.to_pickle()
        ts2 = TautomerState.from_pickle(data)
        assert ts2.smiles == sample_tautomer_anion.smiles
        assert ts2.population_fraction == 0.99

    def test_pickle_type_check(self):
        bad = pickle.dumps({"fake": True})
        with pytest.raises(TypeError, match="Expected TautomerState"):
            TautomerState.from_pickle(bad)

    def test_pka_shifts_tuple_to_list_in_json(self, sample_tautomer_anion):
        j = sample_tautomer_anion.to_json()
        parsed = json.loads(j)
        # JSON converts tuples to lists
        assert isinstance(parsed["pka_shifts"][0], list)
        # Round-trip restores tuples
        ts2 = TautomerState.from_json(j)
        assert isinstance(ts2.pka_shifts[0], tuple)

    def test_multiple_pka_shifts(self):
        ts = TautomerState(
            smiles="NC(=O)c1ccc(O)cc1",
            parent_smiles="NC(=O)c1ccc(O)cc1",
            protonation_ph=7.4,
            charge=0,
            pka_shifts=[(0, 9.2), (7, 10.1)],
            population_fraction=0.65,
            source_tool="pkasolver",
        )
        j = ts.to_json()
        ts2 = TautomerState.from_json(j)
        assert len(ts2.pka_shifts) == 2
        assert ts2.pka_shifts[1] == (7, 10.1)

    def test_zero_charge(self, sample_tautomer_neutral):
        assert sample_tautomer_neutral.charge == 0
        j = sample_tautomer_neutral.to_json()
        ts2 = TautomerState.from_json(j)
        assert ts2.charge == 0

    def test_empty_pka_shifts(self):
        ts = TautomerState(
            smiles="C", parent_smiles="C", protonation_ph=7.4,
            charge=0, pka_shifts=[], population_fraction=1.0,
            source_tool="dimorphite_dl",
        )
        d = ts.to_dict()
        ts2 = TautomerState.from_dict(d)
        assert ts2.pka_shifts == []


class TestTautomerEnsemble:
    """Tests for TautomerEnsemble dataclass."""

    def test_construction(self, sample_tautomer_ensemble):
        te = sample_tautomer_ensemble
        assert te.parent_smiles == "CC(=O)O"
        assert len(te.states) == 2
        assert te.dominant_state.smiles == "CC(=O)[O-]"
        assert te.target_ph == 7.4
        assert te.enumeration_method == "dimorphite_dl_rdk_mstandardize"

    def test_dict_round_trip(self, sample_tautomer_ensemble):
        d = sample_tautomer_ensemble.to_dict()
        te2 = TautomerEnsemble.from_dict(d)
        assert te2.to_dict() == d

    def test_json_round_trip(self, sample_tautomer_ensemble):
        j = sample_tautomer_ensemble.to_json()
        te2 = TautomerEnsemble.from_json(j)
        assert te2.parent_smiles == "CC(=O)O"
        assert len(te2.states) == 2
        assert isinstance(te2.states[0], TautomerState)
        assert isinstance(te2.dominant_state, TautomerState)
        assert te2.dominant_state.charge == -1

    def test_pickle_round_trip(self, sample_tautomer_ensemble):
        data = sample_tautomer_ensemble.to_pickle()
        te2 = TautomerEnsemble.from_pickle(data)
        assert te2.parent_smiles == sample_tautomer_ensemble.parent_smiles
        assert len(te2.states) == 2

    def test_pickle_type_check(self):
        bad = pickle.dumps({"fake": True})
        with pytest.raises(TypeError, match="Expected TautomerEnsemble"):
            TautomerEnsemble.from_pickle(bad)

    def test_nested_pka_shifts_preserved(self, sample_tautomer_ensemble):
        j = sample_tautomer_ensemble.to_json()
        parsed = json.loads(j)
        # Nested tuples become lists in JSON
        assert isinstance(parsed["states"][0]["pka_shifts"][0], list)
        assert isinstance(parsed["dominant_state"]["pka_shifts"][0], list)
        # Round-trip restores tuples
        te2 = TautomerEnsemble.from_json(j)
        assert isinstance(te2.states[0].pka_shifts[0], tuple)
        assert isinstance(te2.dominant_state.pka_shifts[0], tuple)

    def test_single_state_ensemble(self, sample_tautomer_neutral):
        te = TautomerEnsemble(
            parent_smiles="C",
            states=[sample_tautomer_neutral],
            dominant_state=sample_tautomer_neutral,
            target_ph=7.4,
            enumeration_method="dimorphite_dl",
        )
        j = te.to_json()
        te2 = TautomerEnsemble.from_json(j)
        assert len(te2.states) == 1
        assert te2.dominant_state.smiles == sample_tautomer_neutral.smiles
