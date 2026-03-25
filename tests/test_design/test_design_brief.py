"""Tests for DesignBrief dataclass and DesignBriefBuilder."""
import json
import os
import tempfile

import pytest

from scripts.interfaces.anchor_point import AnchorPoint, AnchorPointMap
from scripts.interfaces.design_brief import DesignBrief
from scripts.interfaces.growth_vector import (
    GrowthVector,
    GrowthVectorMap,
    SubPocket,
)
from scripts.interfaces.pocket_profile import PocketProfile
from scripts.interfaces.site_ranking import RankedSite, SiteRanking
from scripts.design_brief_builder import DesignBriefBuilder


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
def _anchor():
    return AnchorPoint(
        residue_name="TYR", residue_id=142, chain="A",
        atom_label="TYR142_BNZ", interaction_type="PI_STACK",
        x=10.0, y=11.0, z=12.0, distance_to_centroid=3.5,
        spike_intensity=8.0, temporal_persistence=0.8,
        geometric_alignment=0.9, stability_stddev=0.5,
        confidence=1.23,
    )


def _anchor_map():
    return AnchorPointMap(
        site_id=0,
        pocket_centroid=(10.0, 10.0, 10.0),
        anchors=[_anchor()],
        n_anchors=1,
        anchor_density=0.125,
    )


def _growth_vector():
    return GrowthVector(
        origin=(10.0, 11.0, 12.0),
        direction=(1.0, 0.0, 0.0),
        free_length=5.0,
        contact_density=0.3,
        expansion_stability=0.8,
        exits_to_solvent=False,
        vector_score=12.0,
        source_anchor_label="TYR142_BNZ",
    )


def _subpocket():
    return SubPocket(
        sub_pocket_id=0,
        centroid=(10.0, 11.0, 12.0),
        volume=150.0,
        feature_types=["PI_STACK"],
        n_features=1,
        dominant_interaction="PI_STACK",
    )


def _growth_map():
    return GrowthVectorMap(
        site_id=0,
        vectors=[_growth_vector()],
        sub_pockets=[_subpocket()],
        n_vectors=1,
        n_sub_pockets=1,
    )


def _pocket_profile():
    return PocketProfile(
        site_id=0,
        aromatic_fraction=0.3,
        polar_fraction=0.4,
        hydrophobic_fraction=0.3,
        charged_positive_fraction=0.1,
        charged_negative_fraction=0.1,
        charge_bias=0.0,
        volume=500.0,
        enclosure=0.4,
        n_lining_residues=8,
        feature_coupling=0.8,
        mw_class="lead",
        polarity_class="mixed",
        water_displacement_energy=3.5,
    )


def _ranked_site():
    return RankedSite(
        site_id=0, rank=1,
        engine_chem=2.388, engine_vcs=0.5,
        contact_reorg_strength=0.12,
        anchor_density=0.125,
        water_displacement=3.5,
    )


def _design_brief():
    return DesignBrief(
        target_name="1btl",
        pdb_id="1BTL",
        site_id=0,
        ranked_site=_ranked_site(),
        anchor_map=_anchor_map(),
        growth_map=_growth_map(),
        pocket_profile=_pocket_profile(),
        water_sites=[
            {"x": 10.0, "y": 10.5, "z": 11.0, "delta_g_transfer": 2.1,
             "classification": "CONSERVED_UNHAPPY", "displaceable": True},
        ],
    )


# ---------------------------------------------------------------------------
# Dataclass tests
# ---------------------------------------------------------------------------
class TestDesignBriefDataclass:
    def test_json_round_trip(self):
        db = _design_brief()
        j = db.to_json()
        db2 = DesignBrief.from_json(j)

        assert db2.target_name == "1btl"
        assert db2.pdb_id == "1BTL"
        assert db2.site_id == 0
        assert db2.ranked_site.rank == 1
        assert db2.anchor_map.n_anchors == 1
        assert db2.growth_map.n_vectors == 1
        assert db2.pocket_profile.mw_class == "lead"
        assert len(db2.water_sites) == 1

    def test_pickle_round_trip(self):
        db = _design_brief()
        data = db.to_pickle()
        db2 = DesignBrief.from_pickle(data)
        assert db2.target_name == "1btl"

    def test_no_water_sites(self):
        db = DesignBrief(
            target_name="test",
            pdb_id="XXXX",
            site_id=0,
            ranked_site=_ranked_site(),
            anchor_map=_anchor_map(),
            growth_map=_growth_map(),
            pocket_profile=_pocket_profile(),
        )
        assert db.water_sites == []
        j = db.to_json()
        db2 = DesignBrief.from_json(j)
        assert db2.water_sites == []


# ---------------------------------------------------------------------------
# Builder tests
# ---------------------------------------------------------------------------
class TestDesignBriefBuilder:
    def test_build_single(self):
        builder = DesignBriefBuilder()
        db = builder.build(
            target_name="1btl",
            pdb_id="1BTL",
            site_id=0,
            ranked_site=_ranked_site(),
            anchor_map=_anchor_map(),
            growth_map=_growth_map(),
            pocket_profile=_pocket_profile(),
        )
        assert db.target_name == "1btl"

    def test_build_all(self):
        ranking = SiteRanking(
            target_name="1btl",
            ranked_sites=[_ranked_site()],
            n_ranked=1,
        )
        builder = DesignBriefBuilder()
        briefs = builder.build_all(
            target_name="1btl",
            pdb_id="1BTL",
            ranking=ranking,
            anchor_maps={0: _anchor_map()},
            growth_maps={0: _growth_map()},
            profiles={0: _pocket_profile()},
        )
        assert len(briefs) == 1

    def test_write_json(self, tmp_path):
        db = _design_brief()
        builder = DesignBriefBuilder()
        path = str(tmp_path / "brief.json")
        builder.write_json(db, path)

        with open(path) as f:
            data = json.load(f)
        assert data["target_name"] == "1btl"
        assert "anchor_map" in data

    def test_write_pymol(self, tmp_path):
        db = _design_brief()
        builder = DesignBriefBuilder()
        path = str(tmp_path / "brief.pml")
        builder.write_pymol(db, path, pdb_path="1btl.pdb")

        content = open(path).read()
        assert "load 1btl.pdb" in content
        assert "pseudoatom centroid_0" in content
        assert "anchor_0_0" in content
        assert "PI_STACK" in content

    def test_write_html(self, tmp_path):
        db = _design_brief()
        builder = DesignBriefBuilder()
        path = str(tmp_path / "brief.html")
        builder.write_html(db, path)

        content = open(path).read()
        assert "PRISM4D DesignBrief" in content
        assert "1btl" in content
        assert "PI_STACK" in content
        assert "lead" in content
        assert "CONSERVED_UNHAPPY" in content
        # No recommendations or confidence statements
        assert "recommend" not in content.lower()
        assert "confidence" not in content.lower()

    def test_write_all(self, tmp_path):
        db = _design_brief()
        builder = DesignBriefBuilder()
        builder.write_all([db], str(tmp_path))

        assert (tmp_path / "1btl_site0.json").exists()
        assert (tmp_path / "1btl_site0.pml").exists()
        assert (tmp_path / "1btl_site0.html").exists()
