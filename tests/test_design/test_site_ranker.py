"""Tests for SiteRanker and ranking dataclasses."""
import json

import pytest

from scripts.interfaces.anchor_point import AnchorPointMap
from scripts.interfaces.contact_reorg_result import ContactReorgResult
from scripts.interfaces.gating_result import GatingResult, SiteGateDecision
from scripts.interfaces.response_profile import ResponseProfile
from scripts.interfaces.site_ranking import RankedSite, SiteRanking
from scripts.site_ranker import SiteRanker


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _decision(
    site_id, overall=True, cr_lr=0.1, rs_sharp=1.0,
):
    return SiteGateDecision(
        site_id=site_id,
        therm_pass=True,
        coherence_pass=True,
        localization_pass=True,
        contact_reorg_pass=True,
        response_selectivity_pass=True,
        overall_pass=overall,
        blocked_by=None if overall else "therm",
        contact_reorg=ContactReorgResult(
            site_id=site_id,
            contact_change_density=2.0,
            localization_ratio=cr_lr,
            persistence=0.5,
            boundary_growth=0.1,
            n_frames_analyzed=8,
            gate_pass=True,
            gate_reason="pass",
        ),
        response_profile=ResponseProfile(
            site_id=site_id,
            sharpness=rs_sharp,
            temporal_asymmetry=0.3,
            energy_density=0.01,
            contact_coupling=float("nan"),
            n_spikes_analyzed=100,
            gate_pass=True,
            gate_reason="pass",
        ),
    )


def _anchor_map(site_id, density=0.5):
    return AnchorPointMap(
        site_id=site_id,
        pocket_centroid=(10, 10, 10),
        anchors=[],
        n_anchors=int(density * 10),
        anchor_density=density,
    )


# ---------------------------------------------------------------------------
# Dataclass tests
# ---------------------------------------------------------------------------
class TestRankedSiteDataclass:
    def test_json_round_trip(self):
        rs = RankedSite(
            site_id=3, rank=1,
            engine_chem=2.5, engine_vcs=0.5,
            contact_reorg_strength=0.12,
            anchor_density=0.4,
            water_displacement=2.5,
        )
        j = rs.to_json()
        rs2 = RankedSite.from_json(j)
        assert rs2.site_id == 3
        assert rs2.rank == 1

    def test_site_ranking_json_round_trip(self):
        sr = SiteRanking(
            target_name="test",
            ranked_sites=[
                RankedSite(0, 1, 2.0, 0.5, 0.1, 0.5, 3.0),
                RankedSite(1, 2, 1.5, 0.3, 0.05, 0.3, 1.0),
            ],
            n_ranked=2,
        )
        j = sr.to_json()
        sr2 = SiteRanking.from_json(j)
        assert sr2.n_ranked == 2
        assert sr2.ranked_sites[0].rank == 1


# ---------------------------------------------------------------------------
# Ranker tests
# ---------------------------------------------------------------------------
class TestSiteRanker:
    def test_basic_ranking(self):
        """Sites ranked by chem+vcs rank fusion."""
        gr = GatingResult(
            target_name="test",
            n_sites_input=3,
            n_sites_passed=2,
            decisions=[
                _decision(0, overall=True, cr_lr=0.05),
                _decision(1, overall=False),
                _decision(2, overall=True, cr_lr=0.15),
            ],
            passed_site_ids=[0, 2],
        )
        # Site 2 has higher chem+vcs
        sites = [
            {"id": 0, "engine_chem": 1.0, "engine_vcs": 0.1},
            {"id": 2, "engine_chem": 2.5, "engine_vcs": 0.5},
        ]
        ams = {0: _anchor_map(0, 0.3), 2: _anchor_map(2, 0.5)}

        ranker = SiteRanker()
        ranking = ranker.rank(gr, sites, ams)

        assert ranking.n_ranked == 2
        assert ranking.ranked_sites[0].site_id == 2  # higher chem+vcs
        assert ranking.ranked_sites[0].rank == 1

    def test_tie_breaker_contact_reorg(self):
        """Same chem+vcs fusion sum → rank by contact_reorg_strength."""
        gr = GatingResult(
            target_name="test",
            n_sites_input=3,
            n_sites_passed=3,
            decisions=[
                _decision(0, overall=True, cr_lr=0.05),
                _decision(1, overall=True, cr_lr=0.20),
                _decision(2, overall=True, cr_lr=0.01),
            ],
            passed_site_ids=[0, 1, 2],
        )
        # Sites 0 and 1 tie on fusion (both get chem_rank+vcs_rank = 3+3=6
        # if we make them swap ranks: 0 has chem=high,vcs=low; 1 has chem=low,vcs=high)
        sites = [
            {"id": 0, "engine_chem": 3.0, "engine_vcs": 0.1},  # cR=1, vR=3 → sum=4
            {"id": 1, "engine_chem": 1.0, "engine_vcs": 0.5},  # cR=3, vR=1 → sum=4
            {"id": 2, "engine_chem": 2.0, "engine_vcs": 0.3},  # cR=2, vR=2 → sum=4
        ]
        ranker = SiteRanker()
        ranking = ranker.rank(gr, sites)
        # All three tie on fusion=4, so cr tiebreaker: site 1 (cr=0.20) wins
        assert ranking.ranked_sites[0].site_id == 1

    def test_water_displacement_tie_breaker(self):
        """Same fusion+cr+density → rank by water_displacement."""
        gr = GatingResult(
            target_name="test",
            n_sites_input=3,
            n_sites_passed=3,
            decisions=[
                _decision(0, overall=True, cr_lr=0.10),
                _decision(1, overall=True, cr_lr=0.10),
                _decision(2, overall=True, cr_lr=0.10),
            ],
            passed_site_ids=[0, 1, 2],
        )
        sites = [
            {"id": 0, "engine_chem": 3.0, "engine_vcs": 0.1},
            {"id": 1, "engine_chem": 1.0, "engine_vcs": 0.5},
            {"id": 2, "engine_chem": 2.0, "engine_vcs": 0.3},
        ]
        we = {0: 1.0, 1: 5.0, 2: 2.0}

        ranker = SiteRanker()
        ranking = ranker.rank(gr, sites, water_energies=we)
        # All tie on fusion=4, cr=0.10, density=0 → wd tiebreaker: site 1 wins
        assert ranking.ranked_sites[0].site_id == 1

    def test_no_passed_sites(self):
        gr = GatingResult(
            target_name="test",
            n_sites_input=1,
            n_sites_passed=0,
            decisions=[_decision(0, overall=False)],
            passed_site_ids=[],
        )
        ranker = SiteRanker()
        ranking = ranker.rank(gr)
        assert ranking.n_ranked == 0

    def test_no_site_data(self):
        """Ranking works without site data (chem/vcs default to 0)."""
        gr = GatingResult(
            target_name="test",
            n_sites_input=1,
            n_sites_passed=1,
            decisions=[_decision(0, overall=True, cr_lr=0.1)],
            passed_site_ids=[0],
        )
        ranker = SiteRanker()
        ranking = ranker.rank(gr)
        assert ranking.n_ranked == 1
        assert ranking.ranked_sites[0].engine_chem == 0.0
