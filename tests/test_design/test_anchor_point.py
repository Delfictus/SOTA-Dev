"""Tests for AnchorPoint dataclass and AnchorPointMapper."""
import json
import math
import random

import pytest

from scripts.interfaces.anchor_point import (
    IDEAL_DISTANCE,
    SPIKE_TYPE_TO_INTERACTION,
    AnchorPoint,
    AnchorPointMap,
)
from scripts.anchor_point_map import AnchorPointMapper, AnchorPointConfig, _geometric_alignment


# ---------------------------------------------------------------------------
# Mock data helpers
# ---------------------------------------------------------------------------
def _make_spike(
    x=10.0, y=10.0, z=10.0, intensity=5.0,
    spike_type="BNZ", frame_index=0, phase="warm_hold",
):
    return {
        "x": x, "y": y, "z": z,
        "intensity": intensity,
        "type": spike_type,
        "ccns_phase": phase,
        "frame_index": frame_index,
        "spike_source": "UV",
        "vibrational_energy": 0.5,
        "water_density": 0.01,
        "wavelength_nm": 280.0,
        "timestep": frame_index * 100,
        "n_nearby_excited": 5,
        "aromatic_residue_id": -1,
    }


def _make_site(
    site_id=0, centroid=(10.0, 10.0, 10.0), volume=500.0, n_lining=8,
):
    lining = []
    resnames = ["TYR", "PHE", "ALA", "LEU", "ASP", "SER", "TRP", "HIS"]
    for i in range(n_lining):
        lining.append({
            "resid": 100 + i,
            "resname": resnames[i % len(resnames)],
            "chain": "A",
            "min_distance": 4.0 + i * 0.5,
            "n_atoms": 8,
            "is_catalytic": i == 0,
        })
    return {
        "id": site_id,
        "centroid": list(centroid),
        "volume": volume,
        "lining_residues": lining,
    }


def _make_spikes_for_site(n=50, centroid=(10.0, 10.0, 10.0)):
    random.seed(42)
    spikes = []
    types = ["BNZ", "PHE", "TYR", "TRP"]
    for i in range(n):
        spikes.append(_make_spike(
            x=centroid[0] + random.gauss(0, 2.0),
            y=centroid[1] + random.gauss(0, 2.0),
            z=centroid[2] + random.gauss(0, 2.0),
            intensity=random.uniform(1.0, 12.0),
            spike_type=random.choice(types),
            frame_index=i % 15,
        ))
    return spikes


# ---------------------------------------------------------------------------
# Dataclass tests
# ---------------------------------------------------------------------------
class TestAnchorPointDataclass:
    def test_json_round_trip(self):
        ap = AnchorPoint(
            residue_name="TYR", residue_id=142, chain="A",
            atom_label="TYR142_BNZ", interaction_type="PI_STACK",
            x=10.0, y=11.0, z=12.0, distance_to_centroid=3.5,
            spike_intensity=8.0, temporal_persistence=0.8,
            geometric_alignment=0.9, stability_stddev=0.5,
            confidence=1.23,
        )
        j = ap.to_json()
        ap2 = AnchorPoint.from_json(j)
        assert ap2.residue_name == "TYR"
        assert ap2.confidence == 1.23

    def test_anchor_map_json_round_trip(self):
        am = AnchorPointMap(
            site_id=0,
            pocket_centroid=(10.0, 10.0, 10.0),
            anchors=[],
            n_anchors=0,
            anchor_density=0.0,
        )
        j = am.to_json()
        am2 = AnchorPointMap.from_json(j)
        assert am2.site_id == 0
        assert am2.pocket_centroid == (10.0, 10.0, 10.0)

    def test_pickle_round_trip(self):
        ap = AnchorPoint(
            residue_name="PHE", residue_id=55, chain="B",
            atom_label="PHE55_PHE", interaction_type="HYDROPHOBIC",
            x=5.0, y=6.0, z=7.0, distance_to_centroid=5.0,
            spike_intensity=3.0, temporal_persistence=0.5,
            geometric_alignment=0.7, stability_stddev=1.0,
            confidence=0.5,
        )
        data = ap.to_pickle()
        ap2 = AnchorPoint.from_pickle(data)
        assert ap2.residue_id == 55


# ---------------------------------------------------------------------------
# Geometry tests
# ---------------------------------------------------------------------------
class TestGeometry:
    def test_alignment_in_range(self):
        """Distance within ideal range → alignment = 1.0."""
        assert _geometric_alignment(4.0, "PI_STACK") == 1.0

    def test_alignment_below_range(self):
        """Distance below ideal range → alignment < 1.0."""
        a = _geometric_alignment(1.0, "PI_STACK")  # ideal 3.5-5.5
        assert 0.0 < a < 1.0

    def test_alignment_above_range(self):
        a = _geometric_alignment(8.0, "PI_STACK")  # ideal 3.5-5.5
        assert 0.0 <= a < 1.0

    def test_interaction_mapping_complete(self):
        """All spike types have a mapping."""
        for spike_type in ["BNZ", "PHE", "TYR", "TRP", "CATION", "ANION", "UNK", "SS"]:
            assert spike_type in SPIKE_TYPE_TO_INTERACTION


# ---------------------------------------------------------------------------
# Mapper tests
# ---------------------------------------------------------------------------
class TestAnchorPointMapper:
    def test_compute_with_spikes(self):
        site = _make_site(0)
        spikes = _make_spikes_for_site(80)
        mapper = AnchorPointMapper()
        am = mapper.compute(site, spikes)

        assert am.site_id == 0
        assert am.n_anchors > 0
        assert am.anchor_density > 0
        assert all(isinstance(a, AnchorPoint) for a in am.anchors)
        # Sorted by confidence descending
        for i in range(len(am.anchors) - 1):
            assert am.anchors[i].confidence >= am.anchors[i + 1].confidence

    def test_compute_no_spikes(self):
        site = _make_site(1)
        mapper = AnchorPointMapper()
        am = mapper.compute(site, [])
        assert am.n_anchors == 0
        assert am.anchor_density == 0.0

    def test_compute_no_lining(self):
        site = {"id": 2, "centroid": [0, 0, 0], "volume": 100, "lining_residues": []}
        spikes = _make_spikes_for_site(50)
        mapper = AnchorPointMapper()
        am = mapper.compute(site, spikes)
        assert am.n_anchors == 0

    def test_deduplication(self):
        """One anchor per residue (highest confidence wins)."""
        site = _make_site(3, n_lining=3)
        spikes = _make_spikes_for_site(100)
        mapper = AnchorPointMapper()
        am = mapper.compute(site, spikes)
        residue_keys = [f"{a.chain}:{a.residue_id}" for a in am.anchors]
        assert len(residue_keys) == len(set(residue_keys))

    def test_compute_all(self):
        sites = [_make_site(0), _make_site(1)]
        sites[0]["spikes"] = _make_spikes_for_site(50)
        sites[1]["spikes"] = _make_spikes_for_site(30)
        mapper = AnchorPointMapper()
        results = mapper.compute_all(sites)
        assert 0 in results
        assert 1 in results

    def test_custom_config(self):
        site = _make_site(0)
        spikes = _make_spikes_for_site(80)
        config = AnchorPointConfig(
            min_spike_intensity=10.0,  # high threshold
            min_temporal_persistence=0.5,
        )
        mapper = AnchorPointMapper(config)
        am = mapper.compute(site, spikes)
        # Fewer anchors with high threshold
        for a in am.anchors:
            assert a.spike_intensity >= 10.0
