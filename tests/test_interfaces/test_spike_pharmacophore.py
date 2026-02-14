"""Tests for SpikePharmacophore interface."""
from __future__ import annotations

import json
import pickle

import pytest

from scripts.interfaces.spike_pharmacophore import (
    ExclusionSphere,
    PharmacophoreFeature,
    SpikePharmacophore,
    SPIKE_TYPE_TO_FEATURE,
)


class TestPharmacophoreFeature:
    def test_construction(self, sample_feature):
        assert sample_feature.feature_type == "AR"
        assert sample_feature.x == 12.5
        assert sample_feature.intensity == 0.85

    def test_distance_to_self(self, sample_feature):
        assert sample_feature.distance_to(sample_feature) == 0.0

    def test_distance_to_other(self, sample_feature, sample_feature_2):
        d = sample_feature.distance_to(sample_feature_2)
        # sqrt((15.5-12.5)^2 + (-1.2-(-3.2))^2 + (10.7-8.7)^2)
        # = sqrt(9 + 4 + 4) = sqrt(17) ≈ 4.123
        assert abs(d - 4.123) < 0.01

    def test_as_tuple(self, sample_feature):
        assert sample_feature.as_tuple() == (12.5, -3.2, 8.7)


class TestExclusionSphere:
    def test_construction(self, sample_exclusion):
        assert sample_exclusion.radius == 2.0
        assert sample_exclusion.source_atom == "CA:ALA145"

    def test_as_tuple(self, sample_exclusion):
        assert sample_exclusion.as_tuple() == (14.0, -2.0, 9.0)


class TestSpikePharmacophore:
    def test_construction(self, sample_pharmacophore):
        sp = sample_pharmacophore
        assert sp.target_name == "KRAS_G12C"
        assert sp.pdb_id == "6GJ8"
        assert sp.pocket_id == 0
        assert len(sp.features) == 2
        assert len(sp.exclusion_spheres) == 1
        assert sp.pocket_centroid == (13.5, -2.0, 9.5)
        assert len(sp.pocket_lining_residues) == 5

    def test_json_round_trip(self, sample_pharmacophore):
        j = sample_pharmacophore.to_json()
        parsed = json.loads(j)
        assert parsed["target_name"] == "KRAS_G12C"
        assert isinstance(parsed["pocket_centroid"], list)

        sp2 = SpikePharmacophore.from_json(j)
        assert sp2.target_name == sample_pharmacophore.target_name
        assert sp2.features[0].x == sample_pharmacophore.features[0].x
        assert sp2.pocket_centroid == sample_pharmacophore.pocket_centroid
        assert sp2.pocket_lining_residues == sample_pharmacophore.pocket_lining_residues

    def test_dict_round_trip(self, sample_pharmacophore):
        d = sample_pharmacophore.to_dict()
        sp2 = SpikePharmacophore.from_dict(d)
        assert sp2.to_dict() == d

    def test_pickle_round_trip(self, sample_pharmacophore):
        data = sample_pharmacophore.to_pickle()
        sp2 = SpikePharmacophore.from_pickle(data)
        assert sp2.target_name == "KRAS_G12C"
        assert sp2.features[0].feature_type == "AR"

    def test_pickle_type_check(self, sample_pharmacophore):
        bad = pickle.dumps("not a pharmacophore")
        with pytest.raises(TypeError, match="Expected SpikePharmacophore"):
            SpikePharmacophore.from_pickle(bad)

    def test_to_phoregen_json(self, sample_pharmacophore):
        pj = sample_pharmacophore.to_phoregen_json()
        assert pj["target"] == "KRAS_G12C"
        assert pj["pdb_id"] == "6GJ8"
        assert pj["pocket_id"] == 0
        assert len(pj["features"]) == 2
        assert pj["features"][0]["type"] == "AR"
        assert pj["features"][0]["radius"] == 1.5
        # High-intensity feature → not optional
        assert pj["features"][0]["optional"] is False
        assert len(pj["exclusions"]) == 1
        assert "bounds" in pj

    def test_to_pgmg_posp(self, sample_pharmacophore):
        posp = sample_pharmacophore.to_pgmg_posp()
        lines = posp.strip().split("\n")
        # Header comments
        assert lines[0].startswith("# PGMG")
        # Feature lines
        feature_lines = [l for l in lines if l.startswith("FEATURE")]
        assert len(feature_lines) == 2
        assert "AR" in feature_lines[0]
        assert "NI" in feature_lines[1]
        # Exclusion
        exc_lines = [l for l in lines if l.startswith("EXCLUSION")]
        assert len(exc_lines) == 1
        # Centroid
        centroid_lines = [l for l in lines if l.startswith("CENTROID")]
        assert len(centroid_lines) == 1
        assert "13.500" in centroid_lines[0]

    def test_to_docking_box_with_features(self, sample_pharmacophore):
        box = sample_pharmacophore.to_docking_box(padding=4.0)
        assert "center_x" in box
        assert "center_y" in box
        assert "center_z" in box
        assert "size_x" in box
        assert "size_y" in box
        assert "size_z" in box
        # Size should be >= 20 (minimum)
        assert box["size_x"] >= 20.0
        assert box["size_y"] >= 20.0
        assert box["size_z"] >= 20.0
        # Size should be <= 40 (Vina max)
        assert box["size_x"] <= 40.0

    def test_to_docking_box_no_features(self):
        sp = SpikePharmacophore(
            target_name="empty", pdb_id="XXXX", pocket_id=0,
            features=[], exclusion_spheres=[],
            pocket_centroid=(10.0, 20.0, 30.0),
            pocket_lining_residues=[],
            prism_run_hash="000",
        )
        box = sp.to_docking_box()
        assert box["center_x"] == 10.0
        assert box["center_y"] == 20.0
        assert box["center_z"] == 30.0
        assert box["size_x"] == 20.0

    def test_default_timestamp(self):
        sp = SpikePharmacophore(
            target_name="test", pdb_id="1ABC", pocket_id=0,
            features=[], exclusion_spheres=[],
            pocket_centroid=(0.0, 0.0, 0.0),
            pocket_lining_residues=[],
            prism_run_hash="xxx",
        )
        assert "T" in sp.creation_timestamp
        assert "+" in sp.creation_timestamp or "Z" in sp.creation_timestamp


class TestSpikeTypeMapping:
    def test_all_known_types_mapped(self):
        expected = {"BNZ", "PHE", "TYR", "TRP", "CATION", "ANION", "UNK", "SS"}
        assert set(SPIKE_TYPE_TO_FEATURE.keys()) == expected

    def test_mapping_values_are_valid(self):
        valid_types = {"AR", "PI", "NI", "HBD", "HBA", "HY"}
        for v in SPIKE_TYPE_TO_FEATURE.values():
            assert v in valid_types
