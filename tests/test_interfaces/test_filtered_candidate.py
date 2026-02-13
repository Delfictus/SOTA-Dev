"""Tests for FilteredCandidate interface."""
from __future__ import annotations

import json
import pickle

import pytest

from scripts.interfaces.filtered_candidate import FilteredCandidate
from scripts.interfaces.generated_molecule import GeneratedMolecule


class TestFilteredCandidate:
    def test_construction_pass(self, sample_candidate):
        assert sample_candidate.passed_all_filters is True
        assert sample_candidate.rejection_reason is None
        assert sample_candidate.qed_score == 0.72
        assert sample_candidate.lipinski_violations == 0
        assert sample_candidate.pains_alerts == []

    def test_construction_reject(self, sample_rejected_candidate):
        rc = sample_rejected_candidate
        assert rc.passed_all_filters is False
        assert rc.rejection_reason is not None
        assert "QED" in rc.rejection_reason
        assert rc.lipinski_violations == 3
        assert len(rc.pains_alerts) == 1

    def test_json_round_trip(self, sample_candidate):
        j = sample_candidate.to_json()
        fc2 = FilteredCandidate.from_json(j)
        assert fc2.molecule.smiles == sample_candidate.molecule.smiles
        assert fc2.qed_score == sample_candidate.qed_score
        assert fc2.cluster_id == sample_candidate.cluster_id
        assert fc2.passed_all_filters is True

    def test_nested_molecule_preserved(self, sample_candidate):
        j = sample_candidate.to_json()
        parsed = json.loads(j)
        assert "molecule" in parsed
        assert parsed["molecule"]["smiles"] == "c1ccc(CC(=O)O)cc1"
        assert parsed["molecule"]["source"] == "phoregen"

    def test_dict_round_trip(self, sample_candidate):
        d = sample_candidate.to_dict()
        fc2 = FilteredCandidate.from_dict(d)
        assert fc2.to_dict() == d

    def test_pickle_round_trip(self, sample_candidate):
        data = sample_candidate.to_pickle()
        fc2 = FilteredCandidate.from_pickle(data)
        assert fc2.molecule.smiles == sample_candidate.molecule.smiles

    def test_pickle_type_check(self):
        bad = pickle.dumps([1, 2, 3])
        with pytest.raises(TypeError, match="Expected FilteredCandidate"):
            FilteredCandidate.from_pickle(bad)

    def test_rejected_json_round_trip(self, sample_rejected_candidate):
        j = sample_rejected_candidate.to_json()
        fc2 = FilteredCandidate.from_json(j)
        assert fc2.passed_all_filters is False
        assert fc2.rejection_reason == sample_rejected_candidate.rejection_reason
        assert fc2.pains_alerts == ["catechol_A(92)"]
