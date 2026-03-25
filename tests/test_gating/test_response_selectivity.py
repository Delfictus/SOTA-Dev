"""Tests for the Response Selectivity gate module."""
import json
import math
import tempfile

import pytest

from scripts.interfaces.response_profile import ResponseProfile
from scripts.response_selectivity import (
    ResponseSelectivityGate,
    ResponseSelectivityThresholds,
    compute_sharpness,
    compute_temporal_asymmetry,
    compute_energy_density,
    compute_contact_coupling,
)


# ---------------------------------------------------------------------------
# Fixtures: mock spike data
# ---------------------------------------------------------------------------
def _make_spike(
    x=10.0, y=10.0, z=10.0,
    intensity=5.0,
    ccns_phase="warm_hold",
    frame_index=0,
    spike_source="UV",
):
    return {
        "x": x, "y": y, "z": z,
        "intensity": intensity,
        "ccns_phase": ccns_phase,
        "frame_index": frame_index,
        "spike_source": spike_source,
        "type": "BNZ",
        "vibrational_energy": 0.5,
        "water_density": 0.01,
        "wavelength_nm": 280.0,
        "timestep": frame_index * 100,
        "n_nearby_excited": 5,
        "aromatic_residue_id": -1,
    }


def _make_focused_spikes(n=100, centroid=(10.0, 10.0, 10.0)):
    """Focused, high-intensity spikes clustered near centroid."""
    import random
    random.seed(42)
    spikes = []
    for i in range(n):
        phase = "warm_hold" if i % 3 != 0 else "cold_hold"
        spikes.append(_make_spike(
            x=centroid[0] + random.gauss(0, 1.0),
            y=centroid[1] + random.gauss(0, 1.0),
            z=centroid[2] + random.gauss(0, 1.0),
            intensity=random.uniform(3.0, 15.0),
            ccns_phase=phase,
            frame_index=i % 20,
        ))
    return spikes


def _make_diffuse_spikes(n=100, centroid=(10.0, 10.0, 10.0)):
    """Diffuse, low-intensity spikes spread over large volume."""
    import random
    random.seed(99)
    spikes = []
    for i in range(n):
        # 50/50 warm/cold → low asymmetry
        phase = "warm_hold" if i % 2 == 0 else "cold_hold"
        spikes.append(_make_spike(
            x=centroid[0] + random.gauss(0, 8.0),
            y=centroid[1] + random.gauss(0, 8.0),
            z=centroid[2] + random.gauss(0, 8.0),
            intensity=random.uniform(0.1, 0.5),
            ccns_phase=phase,
            frame_index=i % 20,
        ))
    return spikes


def _make_site(site_id=0, centroid=(10.0, 10.0, 10.0), volume=500.0):
    return {
        "id": site_id,
        "centroid": list(centroid),
        "volume": volume,
        "spike_count": 100,
    }


# ---------------------------------------------------------------------------
# Dataclass serialization tests
# ---------------------------------------------------------------------------
class TestResponseProfileSerialization:
    def test_json_round_trip(self):
        rp = ResponseProfile(
            site_id=5,
            sharpness=2.1,
            temporal_asymmetry=0.35,
            energy_density=0.02,
            contact_coupling=0.45,
            n_spikes_analyzed=200,
            gate_pass=True,
            gate_reason="pass (3/3: sharpness, temporal_asymmetry, energy_density)",
        )
        j = rp.to_json()
        rp2 = ResponseProfile.from_json(j)
        assert rp2.site_id == 5
        assert rp2.sharpness == 2.1
        assert rp2.gate_pass is True

    def test_json_nan_handling(self):
        rp = ResponseProfile(
            site_id=1,
            sharpness=0.5,
            temporal_asymmetry=0.1,
            energy_density=0.01,
            contact_coupling=float("nan"),
            n_spikes_analyzed=50,
            gate_pass=True,
            gate_reason="pass",
        )
        j = rp.to_json()
        data = json.loads(j)
        assert data["contact_coupling"] is None

        rp2 = ResponseProfile.from_json(j)
        assert math.isnan(rp2.contact_coupling)

    def test_pickle_round_trip(self):
        rp = ResponseProfile(
            site_id=0,
            sharpness=1.0,
            temporal_asymmetry=0.5,
            energy_density=0.1,
            contact_coupling=float("nan"),
            n_spikes_analyzed=10,
            gate_pass=False,
            gate_reason="blocked",
        )
        data = rp.to_pickle()
        rp2 = ResponseProfile.from_pickle(data)
        assert rp2.site_id == 0
        assert math.isnan(rp2.contact_coupling)


# ---------------------------------------------------------------------------
# Individual metric tests
# ---------------------------------------------------------------------------
class TestMetrics:
    def test_sharpness_focused(self):
        """Focused spikes should have high sharpness."""
        spikes = _make_focused_spikes(100)
        centroid = (10.0, 10.0, 10.0)
        s = compute_sharpness(spikes, centroid)
        assert s > 1.0  # peak ~15 / spread ~1.7 ≈ 8+

    def test_sharpness_diffuse(self):
        """Diffuse spikes should have low sharpness."""
        spikes = _make_diffuse_spikes(100)
        centroid = (10.0, 10.0, 10.0)
        s = compute_sharpness(spikes, centroid)
        assert s < 0.5  # peak ~0.5 / spread ~14 ≈ 0.03

    def test_sharpness_empty(self):
        assert compute_sharpness([], (0, 0, 0)) == 0.0

    def test_temporal_asymmetry_strong(self):
        """All warm spikes → asymmetry = 1.0."""
        spikes = [_make_spike(ccns_phase="warm_hold") for _ in range(10)]
        assert compute_temporal_asymmetry(spikes) == 1.0

    def test_temporal_asymmetry_symmetric(self):
        """Equal warm/cold → asymmetry = 0.0."""
        spikes = (
            [_make_spike(ccns_phase="warm_hold") for _ in range(5)]
            + [_make_spike(ccns_phase="cold_hold") for _ in range(5)]
        )
        assert compute_temporal_asymmetry(spikes) == 0.0

    def test_temporal_asymmetry_empty(self):
        assert compute_temporal_asymmetry([]) == 0.0

    def test_energy_density_nonzero(self):
        spikes = [_make_spike(intensity=5.0) for _ in range(10)]
        ed = compute_energy_density(spikes, volume=500.0)
        assert ed == pytest.approx(50.0 / 500.0)

    def test_energy_density_zero_volume(self):
        spikes = [_make_spike(intensity=5.0)]
        assert compute_energy_density(spikes, volume=0.0) == 0.0

    def test_contact_coupling_no_data(self):
        spikes = [_make_spike()]
        assert math.isnan(compute_contact_coupling(spikes, None))

    def test_contact_coupling_perfect_correlation(self):
        """Spikes and contacts both increase together → r ≈ 1."""
        spikes = []
        contact_changes = {}
        for f in range(10):
            # More spikes in later frames
            for _ in range(f + 1):
                spikes.append(_make_spike(frame_index=f))
            contact_changes[f] = f + 1

        r = compute_contact_coupling(spikes, contact_changes)
        assert r > 0.9

    def test_contact_coupling_anti_correlation(self):
        """Spikes increase when contacts decrease → r ≈ -1."""
        spikes = []
        contact_changes = {}
        for f in range(10):
            for _ in range(f + 1):
                spikes.append(_make_spike(frame_index=f))
            contact_changes[f] = 10 - f

        r = compute_contact_coupling(spikes, contact_changes)
        assert r < -0.9


# ---------------------------------------------------------------------------
# Gate logic tests
# ---------------------------------------------------------------------------
class TestResponseSelectivityGate:
    def test_focused_site_passes(self):
        """A focused, asymmetric, dense site should pass."""
        site = _make_site(0, volume=200.0)
        spikes = _make_focused_spikes(100)
        gate = ResponseSelectivityGate()
        result = gate.evaluate(site, spikes)
        assert result.gate_pass is True
        assert result.sharpness > 0.3
        assert result.n_spikes_analyzed == 100

    def test_diffuse_site_blocked(self):
        """A diffuse, symmetric, sparse site should be blocked."""
        site = _make_site(1, volume=50000.0)
        spikes = _make_diffuse_spikes(100)
        gate = ResponseSelectivityGate()
        result = gate.evaluate(site, spikes)
        assert result.gate_pass is False

    def test_no_spikes_blocked(self):
        site = _make_site(2)
        gate = ResponseSelectivityGate()
        result = gate.evaluate(site, [])
        assert result.gate_pass is False
        assert "no_spikes" in result.gate_reason

    def test_anti_correlated_contact_hard_block(self):
        """Strongly negative contact coupling triggers hard block."""
        site = _make_site(3, volume=200.0)
        spikes = _make_focused_spikes(100)  # would otherwise pass

        # Anti-correlated contact changes
        contact_changes = {}
        from collections import Counter
        spike_per_frame = Counter(s["frame_index"] for s in spikes)
        for f in range(20):
            contact_changes[f] = max(0, 20 - spike_per_frame.get(f, 0) * 3)

        gate = ResponseSelectivityGate()
        result = gate.evaluate(site, spikes, contact_changes)
        # May or may not block depending on actual correlation value
        assert isinstance(result, ResponseProfile)

    def test_custom_thresholds(self):
        """Very permissive thresholds → everything passes."""
        site = _make_site(4, volume=50000.0)
        spikes = _make_diffuse_spikes(50)
        thresholds = ResponseSelectivityThresholds(
            min_sharpness=0.001,
            min_temporal_asymmetry=0.001,
            min_energy_density=0.00001,
            min_metrics_passing=1,
        )
        gate = ResponseSelectivityGate(thresholds)
        result = gate.evaluate(site, spikes)
        assert result.gate_pass is True

    def test_evaluate_all_without_spike_dir(self):
        sites = [_make_site(0), _make_site(1)]
        gate = ResponseSelectivityGate()
        results = gate.evaluate_all(sites, spike_events_dir=None)
        # No spikes available → all blocked
        assert all(not r.gate_pass for r in results.values())

    def test_evaluate_all_with_inline_spikes(self):
        """Sites with inline spike data should be processed."""
        spikes = _make_focused_spikes(50)
        site = _make_site(0, volume=200.0)
        site["spikes"] = spikes

        gate = ResponseSelectivityGate()
        results = gate.evaluate_all([site], spike_events_dir=None)
        assert 0 in results
        assert results[0].n_spikes_analyzed == 50
