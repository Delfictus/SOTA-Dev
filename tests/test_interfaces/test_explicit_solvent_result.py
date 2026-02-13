"""Tests for ExplicitSolventResult (WT-9 V2)."""
from __future__ import annotations

import json
import pickle

import pytest

from scripts.interfaces.explicit_solvent_result import ExplicitSolventResult
from scripts.interfaces.water_map import HydrationSite, WaterMap


class TestExplicitSolventResult:
    """Tests for ExplicitSolventResult dataclass."""

    def test_construction_stable(self, sample_explicit_solvent_stable):
        esr = sample_explicit_solvent_stable
        assert esr.pocket_id == 0
        assert esr.simulation_time_ns == 10.0
        assert esr.water_model == "TIP3P"
        assert esr.force_field == "ff19SB"
        assert esr.pocket_stable is True
        assert esr.pocket_rmsd_mean == 1.2
        assert esr.n_structural_waters == 4
        assert len(esr.snapshot_frames) == 10
        assert esr.water_map is not None

    def test_construction_collapsed(self, sample_explicit_solvent_collapsed):
        esr = sample_explicit_solvent_collapsed
        assert esr.pocket_stable is False
        assert esr.water_model == "OPC"
        assert esr.snapshot_frames == []
        assert esr.water_map is None

    def test_dict_round_trip_with_water_map(self, sample_explicit_solvent_stable):
        d = sample_explicit_solvent_stable.to_dict()
        esr2 = ExplicitSolventResult.from_dict(d)
        assert esr2.to_dict() == d

    def test_dict_round_trip_without_water_map(self, sample_explicit_solvent_collapsed):
        d = sample_explicit_solvent_collapsed.to_dict()
        esr2 = ExplicitSolventResult.from_dict(d)
        assert esr2.to_dict() == d

    def test_json_round_trip_with_water_map(self, sample_explicit_solvent_stable):
        j = sample_explicit_solvent_stable.to_json()
        esr2 = ExplicitSolventResult.from_json(j)
        assert esr2.pocket_stable is True
        assert isinstance(esr2.water_map, WaterMap)
        assert isinstance(esr2.water_map.hydration_sites[0], HydrationSite)
        assert esr2.water_map.hydration_sites[0].classification == "CONSERVED_HAPPY"

    def test_json_round_trip_without_water_map(self, sample_explicit_solvent_collapsed):
        j = sample_explicit_solvent_collapsed.to_json()
        esr2 = ExplicitSolventResult.from_json(j)
        assert esr2.water_map is None
        assert esr2.pocket_stable is False

    def test_pickle_round_trip(self, sample_explicit_solvent_stable):
        data = sample_explicit_solvent_stable.to_pickle()
        esr2 = ExplicitSolventResult.from_pickle(data)
        assert esr2.pocket_id == 0
        assert esr2.water_map is not None

    def test_pickle_type_check(self):
        bad = pickle.dumps([1, 2, 3])
        with pytest.raises(TypeError, match="Expected ExplicitSolventResult"):
            ExplicitSolventResult.from_pickle(bad)

    def test_nested_water_map_in_json(self, sample_explicit_solvent_stable):
        j = sample_explicit_solvent_stable.to_json()
        parsed = json.loads(j)
        assert "water_map" in parsed
        assert parsed["water_map"] is not None
        assert len(parsed["water_map"]["hydration_sites"]) == 2

    def test_water_map_none_in_json(self, sample_explicit_solvent_collapsed):
        j = sample_explicit_solvent_collapsed.to_json()
        parsed = json.loads(j)
        assert parsed["water_map"] is None

    def test_snapshot_frames_preserved(self, sample_explicit_solvent_stable):
        j = sample_explicit_solvent_stable.to_json()
        esr2 = ExplicitSolventResult.from_json(j)
        assert esr2.snapshot_frames == [0, 100, 200, 300, 400, 500, 600, 700, 800, 900]
