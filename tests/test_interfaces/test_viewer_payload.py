"""Tests for ViewerPayload (WT-9 V2)."""
from __future__ import annotations

import json
import pickle

import pytest

from scripts.interfaces.viewer_payload import ViewerPayload


class TestViewerPayload:
    """Tests for ViewerPayload dataclass."""

    def test_construction(self, sample_viewer_payload):
        vp = sample_viewer_payload
        assert vp.target_name == "KRAS_G12C"
        assert "ATOM" in vp.pdb_structure
        assert len(vp.pocket_surfaces) == 1
        assert len(vp.spike_positions) == 1
        assert len(vp.water_map_sites) == 1
        assert len(vp.ligand_poses) == 1
        assert vp.lining_residues == [140, 141, 142, 145, 148]
        assert vp.p_open == 0.72
        assert vp.metadata["qa_score"] == 0.92

    def test_dict_round_trip(self, sample_viewer_payload):
        d = sample_viewer_payload.to_dict()
        vp2 = ViewerPayload.from_dict(d)
        assert vp2.to_dict() == d

    def test_json_round_trip(self, sample_viewer_payload):
        j = sample_viewer_payload.to_json()
        vp2 = ViewerPayload.from_json(j)
        assert vp2.target_name == "KRAS_G12C"
        assert vp2.p_open == 0.72
        assert len(vp2.pocket_surfaces) == 1
        assert vp2.spike_positions[0]["type"] == "BNZ"
        assert vp2.water_map_sites[0]["classification"] == "CONSERVED_UNHAPPY"
        assert vp2.ligand_poses[0]["dg_kcal"] == -8.2
        assert vp2.metadata["n_displaceable_waters"] == 1

    def test_pickle_round_trip(self, sample_viewer_payload):
        data = sample_viewer_payload.to_pickle()
        vp2 = ViewerPayload.from_pickle(data)
        assert vp2.target_name == "KRAS_G12C"
        assert vp2.p_open == 0.72

    def test_pickle_type_check(self):
        bad = pickle.dumps(set())
        with pytest.raises(TypeError, match="Expected ViewerPayload"):
            ViewerPayload.from_pickle(bad)

    def test_defaults(self):
        vp = ViewerPayload(
            target_name="TEST",
            pdb_structure="",
            pocket_surfaces=[],
            spike_positions=[],
            water_map_sites=[],
            ligand_poses=[],
            lining_residues=[],
        )
        assert vp.p_open is None
        assert vp.metadata == {}

    def test_p_open_none_round_trip(self):
        vp = ViewerPayload(
            target_name="TEST",
            pdb_structure="ATOM ...\n",
            pocket_surfaces=[],
            spike_positions=[],
            water_map_sites=[],
            ligand_poses=[],
            lining_residues=[],
        )
        j = vp.to_json()
        vp2 = ViewerPayload.from_json(j)
        assert vp2.p_open is None
        assert vp2.metadata == {}

    def test_multiple_ligand_poses(self):
        vp = ViewerPayload(
            target_name="TEAD2",
            pdb_structure="ATOM ...\n",
            pocket_surfaces=[],
            spike_positions=[],
            water_map_sites=[],
            ligand_poses=[
                {"smiles": "CCO", "mol_block": "sdf1", "dg_kcal": -7.5},
                {"smiles": "CCCO", "mol_block": "sdf2", "dg_kcal": -6.8},
                {"smiles": "CCCCO", "mol_block": "sdf3", "dg_kcal": -5.2},
            ],
            lining_residues=[10, 20, 30],
            p_open=0.45,
        )
        j = vp.to_json()
        vp2 = ViewerPayload.from_json(j)
        assert len(vp2.ligand_poses) == 3
        assert vp2.ligand_poses[2]["dg_kcal"] == -5.2

    def test_pocket_surface_structure(self, sample_viewer_payload):
        j = sample_viewer_payload.to_json()
        parsed = json.loads(j)
        surf = parsed["pocket_surfaces"][0]
        assert "vertices" in surf
        assert "triangles" in surf
        assert "color" in surf

    def test_metadata_extensible(self):
        vp = ViewerPayload(
            target_name="TEST",
            pdb_structure="",
            pocket_surfaces=[],
            spike_positions=[],
            water_map_sites=[],
            ligand_poses=[],
            lining_residues=[],
            metadata={
                "custom_field": "value",
                "nested": {"a": 1, "b": [2, 3]},
                "score": 0.95,
            },
        )
        j = vp.to_json()
        vp2 = ViewerPayload.from_json(j)
        assert vp2.metadata["custom_field"] == "value"
        assert vp2.metadata["nested"]["a"] == 1
        assert vp2.metadata["score"] == 0.95

    def test_lining_residues_preserved(self, sample_viewer_payload):
        j = sample_viewer_payload.to_json()
        parsed = json.loads(j)
        assert parsed["lining_residues"] == [140, 141, 142, 145, 148]
