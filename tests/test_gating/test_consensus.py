"""Tests for cross-run metastable pocket consensus."""
import json
import math

import pytest

from scripts.interfaces.consensus_site import (
    MemberSite, ConsensusSite, ConsensusResult,
)
from scripts.consensus import (
    ConsensusBuilder,
    cluster_sites,
    build_consensus_site,
    _jaccard,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _member(run_id=0, site_id=0, centroid=(10.0, 10.0, 10.0),
            qs=0.5, passed=True, lining=None, anchors=None):
    return MemberSite(
        run_id=run_id, site_id=site_id,
        centroid=centroid, quality_score=qs,
        volume=500.0, gate_passed=passed,
        blocked_by=None if passed else "response_selectivity",
        contact_reorg_strength=0.1,
        response_sharpness=2.0,
        response_energy_density=0.01,
        anchor_residue_ids=anchors or [100, 101, 102],
        n_anchors=3,
        lining_residue_ids=lining or [100, 101, 102, 103, 104],
    )


# ---------------------------------------------------------------------------
# Dataclass tests
# ---------------------------------------------------------------------------
class TestConsensusSiteSerialization:
    def test_json_round_trip(self):
        cs = ConsensusSite(
            cluster_id=0,
            member_sites=[_member()],
            n_runs_total=5,
            persistence=0.8,
            pass_fraction=1.0,
            centroid_mean=(10.0, 10.0, 10.0),
            centroid_variance=1.5,
            mean_quality_score=0.5,
            mean_contact_reorg=0.1,
            mean_response_sharpness=2.0,
            anchor_consistency=0.8,
            lining_consistency=0.9,
            gate_failure_reasons={},
        )
        j = cs.to_json()
        cs2 = ConsensusSite.from_json(j)
        assert cs2.persistence == 0.8
        assert cs2.centroid_mean == (10.0, 10.0, 10.0)
        assert len(cs2.member_sites) == 1

    def test_consensus_result_round_trip(self):
        cr = ConsensusResult(
            target_name="test", n_replicates=5,
            consensus_sites=[], n_consensus=0,
        )
        j = cr.to_json()
        cr2 = ConsensusResult.from_json(j)
        assert cr2.target_name == "test"


# ---------------------------------------------------------------------------
# Jaccard tests
# ---------------------------------------------------------------------------
class TestJaccard:
    def test_identical(self):
        assert _jaccard({1, 2, 3}, {1, 2, 3}) == 1.0

    def test_disjoint(self):
        assert _jaccard({1, 2}, {3, 4}) == 0.0

    def test_partial(self):
        assert _jaccard({1, 2, 3}, {2, 3, 4}) == pytest.approx(0.5)

    def test_empty(self):
        assert _jaccard(set(), set()) == 1.0


# ---------------------------------------------------------------------------
# Clustering tests
# ---------------------------------------------------------------------------
class TestClustering:
    def test_same_location_clusters_together(self):
        """Sites at same centroid from different runs → one cluster."""
        members = [
            _member(run_id=0, centroid=(10.0, 10.0, 10.0)),
            _member(run_id=1, centroid=(10.5, 10.5, 10.5)),
            _member(run_id=2, centroid=(11.0, 10.0, 10.0)),
        ]
        clusters = cluster_sites(members, centroid_threshold=5.0)
        assert len(clusters) == 1
        assert len(clusters[0]) == 3

    def test_distant_sites_separate_clusters(self):
        """Sites >threshold apart → different clusters."""
        members = [
            _member(run_id=0, centroid=(10.0, 10.0, 10.0)),
            _member(run_id=1, centroid=(50.0, 50.0, 50.0)),
        ]
        clusters = cluster_sites(members, centroid_threshold=5.0)
        assert len(clusters) == 2

    def test_lining_overlap_required(self):
        """Close centroids but different lining → separate clusters."""
        members = [
            _member(run_id=0, centroid=(10.0, 10.0, 10.0),
                    lining=[100, 101, 102, 103]),
            _member(run_id=1, centroid=(11.0, 10.0, 10.0),
                    lining=[200, 201, 202, 203]),  # completely different
        ]
        clusters = cluster_sites(members, centroid_threshold=5.0,
                                 lining_overlap_min=0.2)
        assert len(clusters) == 2  # lining Jaccard = 0

    def test_persistence_calculation(self):
        """Persistence = fraction of runs where site appears."""
        members = [
            _member(run_id=0), _member(run_id=1), _member(run_id=2),
        ]
        cs = build_consensus_site(0, members, n_runs=5)
        assert cs.persistence == pytest.approx(3.0 / 5.0)

    def test_pass_fraction(self):
        """Pass fraction tracks gate survival rate."""
        members = [
            _member(run_id=0, passed=True),
            _member(run_id=1, passed=True),
            _member(run_id=2, passed=False),
        ]
        cs = build_consensus_site(0, members, n_runs=3)
        assert cs.pass_fraction == pytest.approx(2.0 / 3.0, abs=1e-3)

    def test_centroid_variance(self):
        """Tight centroids → low variance."""
        members = [
            _member(run_id=0, centroid=(10.0, 10.0, 10.0)),
            _member(run_id=1, centroid=(10.0, 10.0, 10.0)),
        ]
        cs = build_consensus_site(0, members, n_runs=2)
        assert cs.centroid_variance == 0.0

    def test_anchor_consistency(self):
        """Same anchors across runs → high consistency."""
        members = [
            _member(run_id=0, anchors=[100, 101, 102]),
            _member(run_id=1, anchors=[100, 101, 102]),
        ]
        cs = build_consensus_site(0, members, n_runs=2)
        assert cs.anchor_consistency == 1.0

    def test_anchor_inconsistency(self):
        """Different anchors → low consistency."""
        members = [
            _member(run_id=0, anchors=[100, 101]),
            _member(run_id=1, anchors=[200, 201]),
        ]
        cs = build_consensus_site(0, members, n_runs=2)
        assert cs.anchor_consistency == 0.0

    def test_gate_failure_tracking(self):
        """Failed sites record why they were blocked."""
        members = [
            _member(run_id=0, passed=False),
            _member(run_id=1, passed=False),
            _member(run_id=2, passed=True),
        ]
        cs = build_consensus_site(0, members, n_runs=3)
        assert "response_selectivity" in cs.gate_failure_reasons
        assert cs.gate_failure_reasons["response_selectivity"] == 2


# ---------------------------------------------------------------------------
# Ranking tests
# ---------------------------------------------------------------------------
class TestConsensusRanking:
    def test_persistence_is_primary_key(self):
        """Higher persistence → higher rank."""
        builder = ConsensusBuilder()
        # Manually create a result and verify ordering
        cs_high = ConsensusSite(
            cluster_id=0, member_sites=[], n_runs_total=5,
            persistence=1.0, pass_fraction=0.5,
            centroid_mean=(0, 0, 0), centroid_variance=2.0,
            mean_quality_score=0.3, mean_contact_reorg=0.1,
            mean_response_sharpness=1.0,
            anchor_consistency=0.5, lining_consistency=0.5,
            gate_failure_reasons={},
        )
        cs_low = ConsensusSite(
            cluster_id=1, member_sites=[], n_runs_total=5,
            persistence=0.4, pass_fraction=1.0,
            centroid_mean=(0, 0, 0), centroid_variance=0.5,
            mean_quality_score=0.9, mean_contact_reorg=0.5,
            mean_response_sharpness=5.0,
            anchor_consistency=1.0, lining_consistency=1.0,
            gate_failure_reasons={},
        )
        # Sort by consensus ranking keys
        sites = [cs_low, cs_high]
        sites.sort(key=lambda cs: (
            -cs.persistence, -cs.pass_fraction,
            cs.centroid_variance, -cs.mean_quality_score,
        ))
        assert sites[0].persistence == 1.0  # high persistence wins
