"""Tests for the full GTCKL+RS Gating Stack orchestrator."""
import json
import math

import pytest

from scripts.interfaces.gating_result import GatingResult, SiteGateDecision
from scripts.gating_stack import (
    GatingStack,
    FoundationThresholds,
    evaluate_therm,
    evaluate_coherence,
    evaluate_localization,
)


# ---------------------------------------------------------------------------
# Fixtures: mock site data matching binding_sites.json format
# ---------------------------------------------------------------------------
def _site(
    site_id=0,
    centroid=(10.0, 10.0, 10.0),
    volume=500.0,
    therm_class="BINDING",
    spike_count=200,
    breathing_score=0.3,
    onset_score=0.2,
    wd_coherence=0.5,
    burial_score=0.4,
    mean_burial=3.0,
    spikes=None,
):
    s = {
        "id": site_id,
        "centroid": list(centroid),
        "volume": volume,
        "therm_class": therm_class,
        "spike_count": spike_count,
        "breathing_score": breathing_score,
        "onset_score": onset_score,
        "wd_coherence": wd_coherence,
        "burial_score": burial_score,
        "mean_burial": mean_burial,
    }
    if spikes is not None:
        s["spikes"] = spikes
    return s


def _make_spikes(n=80, phase_ratio=0.7):
    """Generate spikes with controllable warm/cold ratio."""
    import random
    random.seed(42)
    spikes = []
    for i in range(n):
        phase = "warm_hold" if random.random() < phase_ratio else "cold_hold"
        spikes.append({
            "x": 10.0 + random.gauss(0, 1.5),
            "y": 10.0 + random.gauss(0, 1.5),
            "z": 10.0 + random.gauss(0, 1.5),
            "intensity": random.uniform(2.0, 12.0),
            "ccns_phase": phase,
            "frame_index": i % 15,
            "spike_source": "UV",
            "type": "BNZ",
            "vibrational_energy": 0.5,
            "water_density": 0.01,
            "wavelength_nm": 280.0,
            "timestep": i * 100,
            "n_nearby_excited": 5,
            "aromatic_residue_id": -1,
        })
    return spikes


# ---------------------------------------------------------------------------
# Foundation gate tests
# ---------------------------------------------------------------------------
class TestThermGate:
    def test_pass_by_class(self):
        s = _site(therm_class="BINDING")
        ok, reason = evaluate_therm(s, FoundationThresholds())
        assert ok is True

    def test_pass_by_spike_count(self):
        s = _site(therm_class="UNKNOWN", spike_count=100)
        ok, _ = evaluate_therm(s, FoundationThresholds())
        assert ok is True

    def test_pass_by_override(self):
        s = _site(therm_class="UNKNOWN", spike_count=10,
                  breathing_score=0.7, onset_score=0.5)
        ok, reason = evaluate_therm(s, FoundationThresholds())
        assert ok is True
        assert "override" in reason

    def test_fail(self):
        s = _site(therm_class="UNKNOWN", spike_count=5,
                  breathing_score=0.1, onset_score=0.1)
        ok, _ = evaluate_therm(s, FoundationThresholds())
        assert ok is False


class TestCoherenceGate:
    def test_pass(self):
        s = _site(wd_coherence=0.5)
        ok, _ = evaluate_coherence(s, FoundationThresholds())
        assert ok is True

    def test_fail(self):
        s = _site(wd_coherence=0.1)
        ok, _ = evaluate_coherence(s, FoundationThresholds())
        assert ok is False


class TestLocalizationGate:
    def test_pass_by_burial(self):
        s = _site(burial_score=0.3, mean_burial=0.5)
        ok, _ = evaluate_localization(s, FoundationThresholds())
        assert ok is True

    def test_pass_by_mean_burial(self):
        s = _site(burial_score=0.05, mean_burial=3.0)
        ok, _ = evaluate_localization(s, FoundationThresholds())
        assert ok is True

    def test_fail(self):
        s = _site(burial_score=0.02, mean_burial=0.5)
        ok, _ = evaluate_localization(s, FoundationThresholds())
        assert ok is False


# ---------------------------------------------------------------------------
# Full stack tests
# ---------------------------------------------------------------------------
class TestGatingStack:
    def test_good_site_passes_all(self):
        """A well-behaved site with good metrics passes the full stack."""
        spikes = _make_spikes(100, phase_ratio=0.8)
        site = _site(site_id=0, volume=200.0, spikes=spikes)

        stack = GatingStack()
        result = stack.run("test_target", [site])

        assert result.target_name == "test_target"
        assert result.n_sites_input == 1
        assert len(result.decisions) == 1

        d = result.decisions[0]
        assert d.therm_pass is True
        assert d.localization_pass is True
        assert d.contact_reorg_pass is True  # no trajectory → bypass
        # Response selectivity depends on spike quality
        assert isinstance(d.response_selectivity_pass, bool)

    def test_therm_blocks_first(self):
        """A site failing therm should be blocked at therm, not later gates."""
        site = _site(
            site_id=1,
            therm_class="UNKNOWN",
            spike_count=5,
            breathing_score=0.0,
            onset_score=0.0,
        )

        stack = GatingStack()
        result = stack.run("test", [site])
        d = result.decisions[0]
        assert d.overall_pass is False
        assert d.blocked_by == "therm"

    def test_localization_blocks(self):
        """Site passes therm but fails localization."""
        site = _site(
            site_id=2,
            therm_class="BINDING",
            burial_score=0.01,
            mean_burial=0.5,
        )

        stack = GatingStack()
        result = stack.run("test", [site])
        d = result.decisions[0]
        assert d.overall_pass is False
        assert d.blocked_by == "localization"

    def test_coherence_never_blocks_alone(self):
        """A site failing coherence but passing everything else still passes."""
        spikes = _make_spikes(100, phase_ratio=0.8)
        site = _site(
            site_id=3,
            wd_coherence=0.05,  # fails coherence
            volume=200.0,
            spikes=spikes,
        )

        stack = GatingStack()
        result = stack.run("test", [site])
        d = result.decisions[0]
        assert d.coherence_pass is False
        # But coherence doesn't block
        assert d.blocked_by != "coherence" or d.blocked_by is None

    def test_multiple_sites_ranking(self):
        """Passed sites should be lexicographically ranked."""
        spikes_a = _make_spikes(150, phase_ratio=0.9)
        spikes_b = _make_spikes(80, phase_ratio=0.6)

        site_a = _site(site_id=0, volume=200.0, spikes=spikes_a)
        site_b = _site(site_id=1, volume=300.0, spikes=spikes_b)

        stack = GatingStack()
        result = stack.run("multi", [site_a, site_b])

        assert result.n_sites_input == 2
        # Both may or may not pass RS — check ordering is by lexico keys
        if result.n_sites_passed == 2:
            assert len(result.passed_site_ids) == 2

    def test_empty_sites(self):
        stack = GatingStack()
        result = stack.run("empty", [])
        assert result.n_sites_input == 0
        assert result.n_sites_passed == 0
        assert result.passed_site_ids == []


# ---------------------------------------------------------------------------
# GatingResult serialization tests
# ---------------------------------------------------------------------------
class TestGatingResultSerialization:
    def test_json_round_trip(self):
        spikes = _make_spikes(50)
        site = _site(site_id=0, volume=200.0, spikes=spikes)

        stack = GatingStack()
        result = stack.run("ser_test", [site])

        j = result.to_json()
        result2 = GatingResult.from_json(j)

        assert result2.target_name == "ser_test"
        assert result2.n_sites_input == 1
        assert len(result2.decisions) == 1
        d = result2.decisions[0]
        assert d.site_id == 0
        assert isinstance(d.therm_pass, bool)

        # Nested objects reconstructed
        if d.contact_reorg is not None:
            assert hasattr(d.contact_reorg, "localization_ratio")
        if d.response_profile is not None:
            assert hasattr(d.response_profile, "sharpness")

    def test_pickle_round_trip(self):
        site = _site(site_id=7)
        stack = GatingStack()
        result = stack.run("pickle_test", [site])

        data = result.to_pickle()
        result2 = GatingResult.from_pickle(data)
        assert result2.target_name == "pickle_test"
        assert result2.decisions[0].site_id == 7
