"""Tests for HydrationSite and WaterMap (WT-9 V2)."""
from __future__ import annotations

import json
import pickle

import pytest

from scripts.interfaces.water_map import HydrationSite, WaterMap


class TestHydrationSite:
    """Tests for HydrationSite dataclass."""

    def test_construction(self, sample_happy_water):
        hs = sample_happy_water
        assert hs.x == 10.0
        assert hs.y == 12.5
        assert hs.z == 8.3
        assert hs.occupancy == 0.92
        assert hs.delta_g_transfer == -2.1
        assert hs.classification == "CONSERVED_HAPPY"
        assert hs.displaceable is False

    def test_dict_round_trip(self, sample_happy_water):
        d = sample_happy_water.to_dict()
        hs2 = HydrationSite.from_dict(d)
        assert hs2.to_dict() == d

    def test_json_round_trip(self, sample_unhappy_water):
        j = sample_unhappy_water.to_json()
        hs2 = HydrationSite.from_json(j)
        assert hs2.delta_g_transfer == 1.8
        assert hs2.classification == "CONSERVED_UNHAPPY"
        assert hs2.displaceable is True

    def test_pickle_round_trip(self, sample_happy_water):
        data = sample_happy_water.to_pickle()
        hs2 = HydrationSite.from_pickle(data)
        assert hs2.x == 10.0
        assert hs2.n_hbonds_mean == 2.8

    def test_pickle_type_check(self):
        bad = pickle.dumps("not a HydrationSite")
        with pytest.raises(TypeError, match="Expected HydrationSite"):
            HydrationSite.from_pickle(bad)

    def test_bulk_water(self):
        hs = HydrationSite(
            x=20.0, y=20.0, z=20.0,
            occupancy=0.15,
            delta_g_transfer=0.2,
            entropy_contribution=-0.1,
            enthalpy_contribution=0.3,
            n_hbonds_mean=0.5,
            classification="BULK",
            displaceable=False,
        )
        d = hs.to_dict()
        hs2 = HydrationSite.from_dict(d)
        assert hs2.classification == "BULK"
        assert hs2.displaceable is False


class TestWaterMap:
    """Tests for WaterMap dataclass."""

    def test_construction(self, sample_water_map):
        wm = sample_water_map
        assert wm.pocket_id == 0
        assert len(wm.hydration_sites) == 2
        assert wm.n_displaceable == 1
        assert wm.max_displacement_energy == 1.8
        assert wm.total_displacement_energy == 1.8
        assert wm.grid_resolution == 0.5
        assert wm.analysis_frames == 1000

    def test_dict_round_trip(self, sample_water_map):
        d = sample_water_map.to_dict()
        wm2 = WaterMap.from_dict(d)
        assert wm2.to_dict() == d

    def test_json_round_trip(self, sample_water_map):
        j = sample_water_map.to_json()
        wm2 = WaterMap.from_json(j)
        assert len(wm2.hydration_sites) == 2
        assert isinstance(wm2.hydration_sites[0], HydrationSite)
        assert wm2.hydration_sites[0].classification == "CONSERVED_HAPPY"
        assert wm2.hydration_sites[1].displaceable is True

    def test_pickle_round_trip(self, sample_water_map):
        data = sample_water_map.to_pickle()
        wm2 = WaterMap.from_pickle(data)
        assert wm2.pocket_id == 0
        assert len(wm2.hydration_sites) == 2

    def test_pickle_type_check(self):
        bad = pickle.dumps(42)
        with pytest.raises(TypeError, match="Expected WaterMap"):
            WaterMap.from_pickle(bad)

    def test_nested_hydration_site_in_json(self, sample_water_map):
        j = sample_water_map.to_json()
        parsed = json.loads(j)
        assert "hydration_sites" in parsed
        assert len(parsed["hydration_sites"]) == 2
        assert parsed["hydration_sites"][0]["classification"] == "CONSERVED_HAPPY"

    def test_empty_hydration_sites(self):
        wm = WaterMap(
            pocket_id=3,
            hydration_sites=[],
            n_displaceable=0,
            max_displacement_energy=0.0,
            total_displacement_energy=0.0,
            grid_resolution=0.5,
            analysis_frames=500,
        )
        j = wm.to_json()
        wm2 = WaterMap.from_json(j)
        assert len(wm2.hydration_sites) == 0
        assert wm2.n_displaceable == 0
