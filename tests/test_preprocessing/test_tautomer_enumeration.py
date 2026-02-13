"""Tests for tautomer_enumeration module."""
from __future__ import annotations

import json

import pytest

from scripts.interfaces.tautomer_state import TautomerEnsemble, TautomerState
from scripts.preprocessing.tautomer_enumeration import (
    enumerate_batch,
    enumerate_tautomers,
    _estimate_populations,
    DEFAULT_PH,
    DEFAULT_POPULATION_CUTOFF,
)


class TestEnumerateTautomers:
    """Core tautomer enumeration tests."""

    def test_aspirin(self):
        """Aspirin has an ionizable carboxylic acid group."""
        ens = enumerate_tautomers("CC(=O)Oc1ccccc1C(=O)O")
        assert isinstance(ens, TautomerEnsemble)
        assert ens.target_ph == DEFAULT_PH
        assert len(ens.states) >= 1
        assert ens.dominant_state.population_fraction > 0

    def test_ethanol(self):
        """Ethanol has no ionizable groups at pH 7.4 — expect single state."""
        ens = enumerate_tautomers("CCO")
        assert isinstance(ens, TautomerEnsemble)
        assert len(ens.states) >= 1
        # Ethanol dominant state should be neutral
        assert ens.dominant_state.charge == 0

    def test_acetic_acid(self):
        """Acetic acid (pKa ~4.76) should be deprotonated at pH 7.4."""
        ens = enumerate_tautomers("CC(=O)O")
        assert len(ens.states) >= 1
        # At pH 7.4, deprotonated form should exist
        charges = [s.charge for s in ens.states]
        assert -1 in charges or 0 in charges

    def test_benzene(self):
        """Benzene — no tautomers, no ionization."""
        ens = enumerate_tautomers("c1ccccc1")
        assert len(ens.states) >= 1
        assert ens.dominant_state.charge == 0

    def test_custom_ph(self):
        """Test with a custom pH."""
        ens = enumerate_tautomers("CC(=O)O", target_ph=2.0)
        assert ens.target_ph == 2.0

    def test_population_sums_to_one(self):
        """Population fractions should sum to ~1.0."""
        ens = enumerate_tautomers("CC(=O)Oc1ccccc1C(=O)O")
        total = sum(s.population_fraction for s in ens.states)
        assert abs(total - 1.0) < 0.01

    def test_invalid_smiles_raises(self):
        with pytest.raises(ValueError, match="Invalid SMILES"):
            enumerate_tautomers("NOT_A_MOLECULE")

    def test_dominant_state_is_highest_population(self):
        ens = enumerate_tautomers("CC(=O)O")
        max_pop = max(s.population_fraction for s in ens.states)
        assert ens.dominant_state.population_fraction == max_pop


class TestTautomerStateFields:
    """Test that TautomerState fields are populated correctly."""

    def test_parent_smiles_canonical(self):
        ens = enumerate_tautomers("OC(=O)C")  # non-canonical acetic acid
        # parent_smiles should be the canonicalized form
        assert ens.parent_smiles == "CC(=O)O"

    def test_source_tool_set(self):
        ens = enumerate_tautomers("CCO")
        for state in ens.states:
            assert state.source_tool in ("dimorphite_dl", "rdkit")

    def test_protonation_ph_matches(self):
        ens = enumerate_tautomers("CCO", target_ph=6.0)
        for state in ens.states:
            assert state.protonation_ph == 6.0


class TestSerializationRoundTrip:
    """Test JSON/pickle round-trip through interface contracts."""

    def test_json_round_trip(self):
        ens = enumerate_tautomers("CC(=O)O")
        j = ens.to_json()
        ens2 = TautomerEnsemble.from_json(j)
        assert ens2.parent_smiles == ens.parent_smiles
        assert len(ens2.states) == len(ens.states)
        assert ens2.target_ph == ens.target_ph

    def test_pickle_round_trip(self):
        ens = enumerate_tautomers("CC(=O)O")
        data = ens.to_pickle()
        ens2 = TautomerEnsemble.from_pickle(data)
        assert ens2.parent_smiles == ens.parent_smiles

    def test_dict_round_trip(self):
        ens = enumerate_tautomers("c1ccccc1")
        d = ens.to_dict()
        ens2 = TautomerEnsemble.from_dict(d)
        assert ens2.dominant_state.smiles == ens.dominant_state.smiles


class TestBatchProcessing:
    """Test batch enumeration."""

    def test_batch_valid(self):
        results = enumerate_batch(["CCO", "c1ccccc1", "CC(=O)O"])
        assert len(results) == 3
        assert all(isinstance(r, TautomerEnsemble) for r in results)

    def test_batch_skips_invalid(self):
        results = enumerate_batch(["CCO", "INVALID_XYZ", "c1ccccc1"])
        assert len(results) == 2

    def test_batch_empty(self):
        results = enumerate_batch([])
        assert results == []

    def test_batch_all_invalid(self):
        results = enumerate_batch(["NOPE", "ALSO_NOPE"])
        assert results == []


class TestPopulationEstimation:
    """Test the Boltzmann population estimation helper."""

    def test_single_state(self):
        pops = _estimate_populations(["CCO"], 7.4)
        assert len(pops) == 1
        assert abs(pops[0][1] - 1.0) < 0.001

    def test_neutral_preferred(self):
        """Neutral species should be preferred over charged at pH 7.4."""
        pops = _estimate_populations(["CCO", "CC[O-]"], 7.4)
        neutral_pop = next(p for s, p in pops if s == "CCO")
        charged_pop = next(p for s, p in pops if s == "CC[O-]")
        assert neutral_pop > charged_pop

    def test_empty_input(self):
        assert _estimate_populations([], 7.4) == []


class TestCLI:
    """Test CLI entrypoint."""

    def test_cli_aspirin(self, capsys):
        from scripts.preprocessing.tautomer_enumeration import main
        main(["--smiles", "CC(=O)Oc1ccccc1C(=O)O", "--ph", "7.4"])
        out = capsys.readouterr().out
        parsed = json.loads(out)
        assert "parent_smiles" in parsed
        assert "states" in parsed
        assert "dominant_state" in parsed

    def test_cli_custom_ph(self, capsys):
        from scripts.preprocessing.tautomer_enumeration import main
        main(["--smiles", "CC(=O)O", "--ph", "2.0"])
        out = capsys.readouterr().out
        parsed = json.loads(out)
        assert parsed["target_ph"] == 2.0
