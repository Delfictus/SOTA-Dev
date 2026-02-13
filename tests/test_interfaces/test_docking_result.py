"""Tests for DockingResult interface."""
from __future__ import annotations

import json
import pickle

import pytest

from scripts.interfaces.docking_result import DockingPose, DockingResult


class TestDockingPose:
    def test_construction(self, sample_pose):
        assert sample_pose.pose_rank == 1
        assert sample_pose.vina_score == -8.5
        assert sample_pose.cnn_score == 0.92
        assert sample_pose.cnn_affinity == -9.3

    def test_defaults(self):
        p = DockingPose(pose_rank=1, mol_block="test", vina_score=-5.0)
        assert p.cnn_score == 0.0
        assert p.cnn_affinity == 0.0
        assert p.rmsd_lb == 0.0
        assert p.rmsd_ub == 0.0


class TestDockingResult:
    def test_construction(self, sample_docking_result):
        dr = sample_docking_result
        assert dr.compound_id == "cmpd-001"
        assert len(dr.poses) == 2
        assert dr.best_vina_score == -8.5
        assert dr.docking_engine == "unidock+gnina"
        assert dr.box_center == (13.5, -2.0, 9.5)

    def test_json_round_trip(self, sample_docking_result):
        j = sample_docking_result.to_json()
        dr2 = DockingResult.from_json(j)
        assert dr2.compound_id == "cmpd-001"
        assert dr2.box_center == (13.5, -2.0, 9.5)
        assert dr2.box_size == (25.0, 25.0, 25.0)
        assert len(dr2.poses) == 2
        assert dr2.poses[0].vina_score == -8.5
        assert dr2.poses[1].cnn_score == 0.85

    def test_tuple_to_list_in_json(self, sample_docking_result):
        j = sample_docking_result.to_json()
        parsed = json.loads(j)
        assert isinstance(parsed["box_center"], list)
        assert isinstance(parsed["box_size"], list)

    def test_dict_round_trip(self, sample_docking_result):
        d = sample_docking_result.to_dict()
        dr2 = DockingResult.from_dict(d)
        assert dr2.to_dict() == d

    def test_pickle_round_trip(self, sample_docking_result):
        data = sample_docking_result.to_pickle()
        dr2 = DockingResult.from_pickle(data)
        assert dr2.compound_id == "cmpd-001"

    def test_pickle_type_check(self):
        bad = pickle.dumps({"fake": True})
        with pytest.raises(TypeError, match="Expected DockingResult"):
            DockingResult.from_pickle(bad)

    def test_default_exhaustiveness(self):
        dr = DockingResult(
            compound_id="x", smiles="C", site_id=0,
            receptor_pdb="r.pdb", poses=[], best_vina_score=0.0,
            best_cnn_affinity=0.0, docking_engine="gnina",
            box_center=(0.0, 0.0, 0.0), box_size=(20.0, 20.0, 20.0),
        )
        assert dr.exhaustiveness == 32
