"""Tests for GrowthVector dataclass and GrowthVectorMapper."""
import json
import math

import pytest

from scripts.interfaces.anchor_point import AnchorPoint, AnchorPointMap
from scripts.interfaces.growth_vector import (
    GrowthVector,
    GrowthVectorMap,
    SubPocket,
)
from scripts.growth_vector_map import GrowthVectorMapper, GrowthVectorConfig


# ---------------------------------------------------------------------------
# Mock data
# ---------------------------------------------------------------------------
def _make_anchor(
    x=10.0, y=10.0, z=10.0, label="TYR100_BNZ", itype="PI_STACK",
):
    return AnchorPoint(
        residue_name="TYR", residue_id=100, chain="A",
        atom_label=label, interaction_type=itype,
        x=x, y=y, z=z, distance_to_centroid=3.0,
        spike_intensity=8.0, temporal_persistence=0.7,
        geometric_alignment=0.9, stability_stddev=0.5,
        confidence=1.5,
    )


def _make_anchor_map(n_anchors=3):
    anchors = [
        _make_anchor(10.0, 10.0, 10.0, "TYR100_BNZ"),
        _make_anchor(13.0, 10.0, 10.0, "PHE105_PHE", "HYDROPHOBIC"),
        _make_anchor(10.0, 13.0, 10.0, "ASP110_ANION", "SALT_BRIDGE"),
    ][:n_anchors]
    return AnchorPointMap(
        site_id=0,
        pocket_centroid=(10.0, 10.0, 10.0),
        anchors=anchors,
        n_anchors=len(anchors),
        anchor_density=len(anchors) / 8.0,
    )


def _make_site(site_id=0, n_lining=8):
    lining = []
    for i in range(n_lining):
        lining.append({
            "resid": 100 + i,
            "resname": "ALA",
            "chain": "A",
            "min_distance": 4.0 + i,
        })
    return {
        "id": site_id,
        "centroid": [10.0, 10.0, 10.0],
        "volume": 500.0,
        "lining_residues": lining,
    }


# ---------------------------------------------------------------------------
# Dataclass tests
# ---------------------------------------------------------------------------
class TestGrowthVectorDataclass:
    def test_json_round_trip(self):
        gv = GrowthVector(
            origin=(10.0, 10.0, 10.0),
            direction=(1.0, 0.0, 0.0),
            free_length=5.0,
            contact_density=0.3,
            expansion_stability=0.8,
            exits_to_solvent=False,
            vector_score=12.0,
            source_anchor_label="TYR100_BNZ",
        )
        j = gv.to_json()
        gv2 = GrowthVector.from_json(j)
        assert gv2.free_length == 5.0
        assert gv2.origin == (10.0, 10.0, 10.0)

    def test_subpocket_json_round_trip(self):
        sp = SubPocket(
            sub_pocket_id=0,
            centroid=(10.0, 11.0, 12.0),
            volume=150.0,
            feature_types=["PI_STACK", "HYDROPHOBIC"],
            n_features=2,
            dominant_interaction="PI_STACK",
        )
        j = sp.to_json()
        sp2 = SubPocket.from_json(j)
        assert sp2.dominant_interaction == "PI_STACK"
        assert sp2.centroid == (10.0, 11.0, 12.0)

    def test_gvm_json_round_trip(self):
        gvm = GrowthVectorMap(
            site_id=0,
            vectors=[],
            sub_pockets=[],
            n_vectors=0,
            n_sub_pockets=0,
        )
        j = gvm.to_json()
        gvm2 = GrowthVectorMap.from_json(j)
        assert gvm2.site_id == 0


# ---------------------------------------------------------------------------
# Mapper tests
# ---------------------------------------------------------------------------
class TestGrowthVectorMapper:
    def test_compute_with_anchors(self):
        site = _make_site(0)
        am = _make_anchor_map(3)
        mapper = GrowthVectorMapper()
        gvm = mapper.compute(site, am)

        assert gvm.site_id == 0
        assert gvm.n_vectors >= 0
        assert gvm.n_sub_pockets >= 1  # at least 1 cluster from 3 anchors
        for v in gvm.vectors:
            assert not v.exits_to_solvent  # filtered out
            assert v.free_length >= 2.0     # min free length

    def test_compute_no_anchors(self):
        site = _make_site(1)
        am = AnchorPointMap(
            site_id=1,
            pocket_centroid=(10, 10, 10),
            anchors=[],
            n_anchors=0,
            anchor_density=0.0,
        )
        mapper = GrowthVectorMapper()
        gvm = mapper.compute(site, am)
        assert gvm.n_vectors == 0
        assert gvm.n_sub_pockets == 0

    def test_subpocket_segmentation(self):
        """Anchors close together → 1 subpocket. Far apart → 2+."""
        am_close = AnchorPointMap(
            site_id=0,
            pocket_centroid=(10, 10, 10),
            anchors=[
                _make_anchor(10.0, 10.0, 10.0, "A"),
                _make_anchor(11.0, 10.0, 10.0, "B"),
            ],
            n_anchors=2,
            anchor_density=0.25,
        )
        am_far = AnchorPointMap(
            site_id=0,
            pocket_centroid=(10, 10, 10),
            anchors=[
                _make_anchor(10.0, 10.0, 10.0, "A"),
                _make_anchor(20.0, 20.0, 20.0, "B"),
            ],
            n_anchors=2,
            anchor_density=0.25,
        )
        mapper = GrowthVectorMapper()
        gvm_close = mapper.compute(_make_site(), am_close)
        gvm_far = mapper.compute(_make_site(), am_far)

        assert gvm_close.n_sub_pockets == 1  # within 5A cluster radius
        assert gvm_far.n_sub_pockets == 2    # >5A apart

    def test_compute_all(self):
        sites = [_make_site(0), _make_site(1)]
        ams = {0: _make_anchor_map(2), 1: _make_anchor_map(1)}
        mapper = GrowthVectorMapper()
        results = mapper.compute_all(sites, ams)
        assert 0 in results
        assert 1 in results

    def test_vectors_per_anchor_limit(self):
        """Max 3 vectors per anchor."""
        site = _make_site(0, n_lining=2)
        am = _make_anchor_map(1)
        mapper = GrowthVectorMapper()
        gvm = mapper.compute(site, am)

        label_counts = {}
        for v in gvm.vectors:
            label_counts[v.source_anchor_label] = (
                label_counts.get(v.source_anchor_label, 0) + 1
            )
        for count in label_counts.values():
            assert count <= 3
