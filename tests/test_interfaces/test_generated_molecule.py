"""Tests for GeneratedMolecule interface."""
from __future__ import annotations

import json
import pickle

import pytest

from scripts.interfaces.generated_molecule import GeneratedMolecule


class TestGeneratedMolecule:
    def test_construction(self, sample_molecule):
        assert sample_molecule.smiles == "c1ccc(CC(=O)O)cc1"
        assert sample_molecule.source == "phoregen"
        assert sample_molecule.pharmacophore_match_score == 0.80
        assert sample_molecule.matched_features == ["AR", "NI"]

    def test_json_round_trip(self, sample_molecule):
        j = sample_molecule.to_json()
        gm2 = GeneratedMolecule.from_json(j)
        assert gm2.smiles == sample_molecule.smiles
        assert gm2.source == sample_molecule.source
        assert gm2.matched_features == sample_molecule.matched_features
        assert gm2.generation_batch_id == sample_molecule.generation_batch_id

    def test_dict_round_trip(self, sample_molecule):
        d = sample_molecule.to_dict()
        gm2 = GeneratedMolecule.from_dict(d)
        assert gm2.to_dict() == d

    def test_pickle_round_trip(self, sample_molecule):
        data = sample_molecule.to_pickle()
        gm2 = GeneratedMolecule.from_pickle(data)
        assert gm2.smiles == sample_molecule.smiles

    def test_pickle_type_check(self):
        bad = pickle.dumps(42)
        with pytest.raises(TypeError, match="Expected GeneratedMolecule"):
            GeneratedMolecule.from_pickle(bad)

    def test_mol_block_preserved(self, sample_molecule):
        j = sample_molecule.to_json()
        gm2 = GeneratedMolecule.from_json(j)
        assert gm2.mol_block == sample_molecule.mol_block
        assert "V2000" in gm2.mol_block

    def test_default_timestamp(self):
        gm = GeneratedMolecule(
            smiles="C", mol_block="", source="pgmg",
            pharmacophore_match_score=0.5, matched_features=[],
            generation_batch_id="test",
        )
        assert "T" in gm.generation_timestamp

    def test_pgmg_source(self):
        gm = GeneratedMolecule(
            smiles="CC", mol_block="", source="pgmg",
            pharmacophore_match_score=0.6, matched_features=["HY"],
            generation_batch_id="pgmg-001",
        )
        assert gm.source == "pgmg"
