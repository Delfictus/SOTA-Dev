"""Tests for ResidueMapping interface."""
from __future__ import annotations

import json
import pickle

import pytest

from scripts.interfaces.residue_mapping import ResidueEntry, ResidueMapping


class TestResidueEntry:
    def test_construction(self):
        e = ResidueEntry(
            topology_id=5, pdb_resid=10, pdb_chain="B",
            pdb_insertion_code="", uniprot_position=15,
            residue_name="ARG",
        )
        assert e.topology_id == 5
        assert e.pdb_chain == "B"

    def test_pdb_label_no_insertion(self):
        e = ResidueEntry(
            topology_id=0, pdb_resid=142, pdb_chain="A",
            pdb_insertion_code="", uniprot_position=142,
            residue_name="TYR",
        )
        assert e.pdb_label == "A:TYR142"

    def test_pdb_label_with_insertion(self):
        e = ResidueEntry(
            topology_id=3, pdb_resid=27, pdb_chain="A",
            pdb_insertion_code="A", uniprot_position=27,
            residue_name="ALA",
        )
        assert e.pdb_label == "A:ALA27A"


class TestResidueMapping:
    def test_construction(self, sample_residue_mapping):
        rm = sample_residue_mapping
        assert rm.pdb_id == "6GJ8"
        assert rm.uniprot_id == "P01116"
        assert len(rm.entries) == 4
        assert rm.mapping_source == "SIFTS"

    def test_topology_to_pdb(self, sample_residue_mapping):
        e = sample_residue_mapping.topology_to_pdb(0)
        assert e is not None
        assert e.pdb_resid == 1
        assert e.residue_name == "MET"

    def test_topology_to_pdb_missing(self, sample_residue_mapping):
        e = sample_residue_mapping.topology_to_pdb(999)
        assert e is None

    def test_pdb_to_topology(self, sample_residue_mapping):
        e = sample_residue_mapping.pdb_to_topology(2, "A")
        assert e is not None
        assert e.topology_id == 1
        assert e.residue_name == "THR"

    def test_pdb_to_topology_with_insertion(self, sample_residue_mapping):
        e = sample_residue_mapping.pdb_to_topology(27, "A", "A")
        assert e is not None
        assert e.topology_id == 3
        assert e.residue_name == "ALA"

    def test_pdb_to_topology_missing(self, sample_residue_mapping):
        e = sample_residue_mapping.pdb_to_topology(999, "A")
        assert e is None

    def test_uniprot_to_entries(self, sample_residue_mapping):
        entries = sample_residue_mapping.uniprot_to_entries(1)
        assert len(entries) == 1
        assert entries[0].residue_name == "MET"

    def test_uniprot_to_entries_missing(self, sample_residue_mapping):
        entries = sample_residue_mapping.uniprot_to_entries(999)
        assert entries == []

    def test_topology_ids_to_pdb_labels(self, sample_residue_mapping):
        labels = sample_residue_mapping.topology_ids_to_pdb_labels([0, 1, 999])
        assert labels == ["A:MET1", "A:THR2", "?:999"]

    def test_json_round_trip(self, sample_residue_mapping):
        j = sample_residue_mapping.to_json()
        rm2 = ResidueMapping.from_json(j)
        assert rm2.pdb_id == "6GJ8"
        assert len(rm2.entries) == 4
        assert rm2.entries[3].pdb_label == "A:ALA27A"

    def test_dict_round_trip(self, sample_residue_mapping):
        d = sample_residue_mapping.to_dict()
        rm2 = ResidueMapping.from_dict(d)
        assert rm2.to_dict() == d

    def test_pickle_round_trip(self, sample_residue_mapping):
        data = sample_residue_mapping.to_pickle()
        rm2 = ResidueMapping.from_pickle(data)
        assert rm2.uniprot_id == "P01116"

    def test_pickle_type_check(self):
        bad = pickle.dumps(set())
        with pytest.raises(TypeError, match="Expected ResidueMapping"):
            ResidueMapping.from_pickle(bad)

    def test_multi_chain(self):
        """Test with entries spanning multiple chains."""
        entries = [
            ResidueEntry(0, 1, "A", "", 1, "MET"),
            ResidueEntry(1, 2, "A", "", 2, "ALA"),
            ResidueEntry(2, 1, "B", "", None, "MET"),
            ResidueEntry(3, 2, "B", "", None, "GLY"),
        ]
        rm = ResidueMapping(
            pdb_id="1ABC", uniprot_id="P00001", chains=["A", "B"],
            entries=entries, topology_residue_count=4, pdb_residue_count=4,
        )
        a1 = rm.pdb_to_topology(1, "A")
        b1 = rm.pdb_to_topology(1, "B")
        assert a1.topology_id == 0
        assert b1.topology_id == 2
        assert a1.uniprot_position == 1
        assert b1.uniprot_position is None
