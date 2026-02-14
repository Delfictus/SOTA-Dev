"""Tests for pocket_popen.py — P_open from multi-stream trajectories."""
from __future__ import annotations

import numpy as np
import pytest

from scripts.interfaces import PocketDynamics
from scripts.ensemble.pocket_popen import (
    CLASSIFICATION_RARE_EVENT,
    CLASSIFICATION_STABLE_OPEN,
    CLASSIFICATION_TRANSIENT,
    DEFAULT_VOLUME_THRESHOLD_A3,
    POPEN_STABLE_OPEN_THRESHOLD,
    POPEN_TRANSIENT_THRESHOLD,
    bootstrap_popen,
    classify_druggability,
    compute_binary_trajectory,
    compute_lifetimes,
    compute_popen,
    compute_volume_autocorrelation,
    popen_from_trajectory_files,
)


# ═══════════════════════════════════════════════════════════════════════════
#  Classification
# ═══════════════════════════════════════════════════════════════════════════

class TestClassifyDruggability:
    def test_stable_open(self):
        assert classify_druggability(0.8) == CLASSIFICATION_STABLE_OPEN

    def test_stable_open_boundary(self):
        """P_open = 0.51 → STABLE_OPEN."""
        assert classify_druggability(0.51) == CLASSIFICATION_STABLE_OPEN

    def test_stable_open_at_threshold(self):
        """P_open = 0.5 exactly → TRANSIENT (not > 0.5)."""
        assert classify_druggability(0.5) == CLASSIFICATION_TRANSIENT

    def test_transient(self):
        assert classify_druggability(0.3) == CLASSIFICATION_TRANSIENT

    def test_transient_boundary(self):
        """P_open = 0.11 → TRANSIENT."""
        assert classify_druggability(0.11) == CLASSIFICATION_TRANSIENT

    def test_transient_at_threshold(self):
        """P_open = 0.1 exactly → RARE_EVENT (not > 0.1)."""
        assert classify_druggability(0.1) == CLASSIFICATION_RARE_EVENT

    def test_rare_event(self):
        assert classify_druggability(0.05) == CLASSIFICATION_RARE_EVENT

    def test_zero(self):
        assert classify_druggability(0.0) == CLASSIFICATION_RARE_EVENT

    def test_one(self):
        assert classify_druggability(1.0) == CLASSIFICATION_STABLE_OPEN


# ═══════════════════════════════════════════════════════════════════════════
#  Binary trajectory
# ═══════════════════════════════════════════════════════════════════════════

class TestBinaryTrajectory:
    def test_basic(self):
        vols = np.array([300, 100, 250, 50, 400])
        binary = compute_binary_trajectory(vols, threshold=200.0)
        assert list(binary) == [True, False, True, False, True]

    def test_all_open(self):
        vols = np.array([500, 600, 700])
        binary = compute_binary_trajectory(vols, threshold=200.0)
        assert all(binary)

    def test_all_closed(self):
        vols = np.array([50, 100, 150])
        binary = compute_binary_trajectory(vols, threshold=200.0)
        assert not any(binary)

    def test_at_threshold_closed(self):
        """Volume exactly at threshold → closed (> threshold required)."""
        vols = np.array([200.0])
        binary = compute_binary_trajectory(vols, threshold=200.0)
        assert not binary[0]

    def test_custom_threshold(self):
        vols = np.array([100, 200, 300])
        binary = compute_binary_trajectory(vols, threshold=150.0)
        assert list(binary) == [False, True, True]


# ═══════════════════════════════════════════════════════════════════════════
#  Lifetimes
# ═══════════════════════════════════════════════════════════════════════════

class TestLifetimes:
    def test_alternating(self):
        binary = np.array([True, False, True, False, True])
        open_lt, closed_lt = compute_lifetimes(binary, dt_ns=0.1)
        assert len(open_lt) == 3
        assert len(closed_lt) == 2
        assert all(abs(lt - 0.1) < 1e-10 for lt in open_lt)
        assert all(abs(lt - 0.1) < 1e-10 for lt in closed_lt)

    def test_all_open(self):
        binary = np.array([True, True, True, True])
        open_lt, closed_lt = compute_lifetimes(binary, dt_ns=0.1)
        assert len(open_lt) == 1
        assert abs(open_lt[0] - 0.4) < 1e-10
        assert len(closed_lt) == 0

    def test_all_closed(self):
        binary = np.array([False, False, False])
        open_lt, closed_lt = compute_lifetimes(binary, dt_ns=0.1)
        assert len(open_lt) == 0
        assert len(closed_lt) == 1
        assert abs(closed_lt[0] - 0.3) < 1e-10

    def test_empty(self):
        open_lt, closed_lt = compute_lifetimes(np.array([]), dt_ns=0.1)
        assert open_lt == []
        assert closed_lt == []

    def test_long_runs(self):
        binary = np.array([True]*10 + [False]*5 + [True]*3)
        open_lt, closed_lt = compute_lifetimes(binary, dt_ns=0.01)
        assert len(open_lt) == 2
        assert len(closed_lt) == 1
        assert abs(open_lt[0] - 0.10) < 1e-10
        assert abs(closed_lt[0] - 0.05) < 1e-10
        assert abs(open_lt[1] - 0.03) < 1e-10


# ═══════════════════════════════════════════════════════════════════════════
#  Bootstrap P_open
# ═══════════════════════════════════════════════════════════════════════════

class TestBootstrapPopen:
    def test_basic(self):
        fractions = np.array([0.6, 0.7, 0.5, 0.65, 0.72])
        p_open, error = bootstrap_popen(fractions)
        assert 0.5 < p_open < 0.8
        assert error > 0

    def test_single_stream(self):
        p_open, error = bootstrap_popen(np.array([0.6]))
        assert p_open == 0.6
        assert error == 0.0

    def test_empty(self):
        p_open, error = bootstrap_popen(np.array([]))
        assert p_open == 0.0
        assert error == 0.0

    def test_identical_fractions(self):
        fractions = np.full(10, 0.5)
        p_open, error = bootstrap_popen(fractions)
        assert abs(p_open - 0.5) < 1e-10
        assert error < 1e-10

    def test_reproducible(self):
        fractions = np.array([0.6, 0.7, 0.5, 0.65, 0.72])
        p1, e1 = bootstrap_popen(fractions, seed=42)
        p2, e2 = bootstrap_popen(fractions, seed=42)
        assert p1 == p2
        assert e1 == e2

    def test_error_decreases_with_streams(self):
        """More streams → smaller error."""
        rng = np.random.default_rng(42)
        small = rng.normal(0.6, 0.05, 5)
        large = rng.normal(0.6, 0.05, 50)
        _, error_small = bootstrap_popen(small)
        _, error_large = bootstrap_popen(large)
        assert error_large < error_small


# ═══════════════════════════════════════════════════════════════════════════
#  Volume autocorrelation
# ═══════════════════════════════════════════════════════════════════════════

class TestVolumeAutocorrelation:
    def test_uncorrelated(self):
        rng = np.random.default_rng(42)
        vols = rng.normal(300, 50, 1000)
        tau = compute_volume_autocorrelation(vols, dt_ns=0.001)
        # Uncorrelated → short autocorrelation
        assert tau < 0.01

    def test_constant_volumes(self):
        vols = np.full(100, 300.0)
        tau = compute_volume_autocorrelation(vols, dt_ns=0.001)
        assert tau == 0.0

    def test_positive(self):
        rng = np.random.default_rng(42)
        vols = rng.normal(300, 50, 100)
        tau = compute_volume_autocorrelation(vols, dt_ns=0.001)
        assert tau >= 0.0


# ═══════════════════════════════════════════════════════════════════════════
#  Full P_open computation
# ═══════════════════════════════════════════════════════════════════════════

class TestComputePopen:
    def test_stable_open(self, open_pocket_streams):
        result = compute_popen(open_pocket_streams, pocket_id=0)
        assert isinstance(result, PocketDynamics)
        assert result.pocket_id == 0
        assert result.p_open > POPEN_STABLE_OPEN_THRESHOLD
        assert result.druggability_classification == CLASSIFICATION_STABLE_OPEN
        assert result.p_open_error > 0
        assert result.n_opening_events > 0

    def test_transient(self, transient_pocket_streams):
        result = compute_popen(transient_pocket_streams, pocket_id=1)
        assert result.druggability_classification == CLASSIFICATION_TRANSIENT

    def test_rare_event(self, rare_event_streams):
        result = compute_popen(rare_event_streams, pocket_id=2)
        assert result.druggability_classification == CLASSIFICATION_RARE_EVENT

    def test_empty_streams_raises(self):
        with pytest.raises(ValueError, match="No volume streams"):
            compute_popen([])

    def test_all_empty_streams_raises(self):
        with pytest.raises(ValueError, match="All streams were empty"):
            compute_popen([np.array([]), np.array([])])

    def test_lifetimes_populated(self, open_pocket_streams):
        result = compute_popen(open_pocket_streams)
        assert result.mean_open_lifetime_ns > 0
        assert result.mean_closed_lifetime_ns >= 0

    def test_msm_weights_none(self, open_pocket_streams):
        result = compute_popen(open_pocket_streams)
        assert result.msm_state_weights is None

    def test_custom_threshold(self, rng):
        """Very low threshold → everything is open."""
        streams = [rng.normal(300, 50, 100) for _ in range(5)]
        result = compute_popen(streams, volume_threshold=10.0)
        assert result.p_open > 0.99

    def test_very_high_threshold(self, rng):
        """Very high threshold → almost nothing is open."""
        streams = [rng.normal(300, 50, 100) for _ in range(5)]
        result = compute_popen(streams, volume_threshold=10000.0)
        assert result.p_open < 0.01


# ═══════════════════════════════════════════════════════════════════════════
#  Dry run
# ═══════════════════════════════════════════════════════════════════════════

class TestDryRun:
    def test_dry_run_returns_result(self):
        result = popen_from_trajectory_files(
            ["stream1.xtc", "stream2.xtc"],
            pocket_lining_residues=[10, 20, 30],
            dry_run=True,
        )
        assert isinstance(result, PocketDynamics)
        assert result.p_open > 0
        assert result.druggability_classification in {
            CLASSIFICATION_STABLE_OPEN,
            CLASSIFICATION_TRANSIENT,
            CLASSIFICATION_RARE_EVENT,
        }


# ═══════════════════════════════════════════════════════════════════════════
#  Serialisation
# ═══════════════════════════════════════════════════════════════════════════

class TestSerialisation:
    def test_json_roundtrip(self, sample_pocket_dynamics):
        j = sample_pocket_dynamics.to_json()
        loaded = PocketDynamics.from_json(j)
        assert loaded.pocket_id == 0
        assert abs(loaded.p_open - 0.72) < 1e-10
        assert loaded.druggability_classification == "STABLE_OPEN"

    def test_pickle_roundtrip(self, sample_pocket_dynamics):
        data = sample_pocket_dynamics.to_pickle()
        loaded = PocketDynamics.from_pickle(data)
        assert loaded.pocket_id == 0
        assert abs(loaded.p_open_error - 0.05) < 1e-10

    def test_with_msm_weights(self):
        pd = PocketDynamics(
            pocket_id=0, p_open=0.5, p_open_error=0.1,
            mean_open_lifetime_ns=1.0, mean_closed_lifetime_ns=1.0,
            n_opening_events=5, druggability_classification="TRANSIENT",
            volume_autocorrelation_ns=0.1,
            msm_state_weights={0: 0.3, 1: 0.5, 2: 0.2},
        )
        j = pd.to_json()
        loaded = PocketDynamics.from_json(j)
        assert loaded.msm_state_weights == {0: 0.3, 1: 0.5, 2: 0.2}


# ═══════════════════════════════════════════════════════════════════════════
#  Threshold constants
# ═══════════════════════════════════════════════════════════════════════════

class TestConstants:
    def test_threshold_values(self):
        assert POPEN_STABLE_OPEN_THRESHOLD == 0.5
        assert POPEN_TRANSIENT_THRESHOLD == 0.1
        assert DEFAULT_VOLUME_THRESHOLD_A3 == 200.0
