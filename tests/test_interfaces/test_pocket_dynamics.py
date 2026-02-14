"""Tests for PocketDynamics (WT-9 V2)."""
from __future__ import annotations

import json
import pickle

import pytest

from scripts.interfaces.pocket_dynamics import PocketDynamics


class TestPocketDynamics:
    """Tests for PocketDynamics dataclass."""

    def test_construction_stable(self, sample_pocket_dynamics_stable):
        pd = sample_pocket_dynamics_stable
        assert pd.pocket_id == 0
        assert pd.p_open == 0.72
        assert pd.p_open_error == 0.06
        assert pd.mean_open_lifetime_ns == 3.5
        assert pd.mean_closed_lifetime_ns == 1.2
        assert pd.n_opening_events == 48
        assert pd.druggability_classification == "STABLE_OPEN"
        assert pd.volume_autocorrelation_ns == 0.8
        assert pd.msm_state_weights == {0: 0.6, 1: 0.25, 2: 0.15}

    def test_construction_rare(self, sample_pocket_dynamics_rare):
        pd = sample_pocket_dynamics_rare
        assert pd.p_open == 0.05
        assert pd.druggability_classification == "RARE_EVENT"
        assert pd.msm_state_weights is None

    def test_dict_round_trip_with_msm(self, sample_pocket_dynamics_stable):
        d = sample_pocket_dynamics_stable.to_dict()
        pd2 = PocketDynamics.from_dict(d)
        assert pd2.to_dict() == d

    def test_dict_round_trip_without_msm(self, sample_pocket_dynamics_rare):
        d = sample_pocket_dynamics_rare.to_dict()
        pd2 = PocketDynamics.from_dict(d)
        assert pd2.to_dict() == d

    def test_json_round_trip_with_msm(self, sample_pocket_dynamics_stable):
        j = sample_pocket_dynamics_stable.to_json()
        pd2 = PocketDynamics.from_json(j)
        assert pd2.p_open == 0.72
        assert pd2.msm_state_weights[0] == 0.6
        assert isinstance(list(pd2.msm_state_weights.keys())[0], int)

    def test_json_round_trip_without_msm(self, sample_pocket_dynamics_rare):
        j = sample_pocket_dynamics_rare.to_json()
        pd2 = PocketDynamics.from_json(j)
        assert pd2.msm_state_weights is None
        assert pd2.druggability_classification == "RARE_EVENT"

    def test_pickle_round_trip(self, sample_pocket_dynamics_stable):
        data = sample_pocket_dynamics_stable.to_pickle()
        pd2 = PocketDynamics.from_pickle(data)
        assert pd2.pocket_id == 0
        assert pd2.msm_state_weights is not None

    def test_pickle_type_check(self):
        bad = pickle.dumps({"fake": True})
        with pytest.raises(TypeError, match="Expected PocketDynamics"):
            PocketDynamics.from_pickle(bad)

    def test_msm_int_keys_in_json(self, sample_pocket_dynamics_stable):
        """MSM state int keys survive JSON round-trip."""
        j = sample_pocket_dynamics_stable.to_json()
        parsed = json.loads(j)
        # JSON keys are strings
        assert "0" in parsed["msm_state_weights"]
        # Round-trip restores int keys
        pd2 = PocketDynamics.from_json(j)
        assert 0 in pd2.msm_state_weights

    def test_msm_none_in_json(self, sample_pocket_dynamics_rare):
        j = sample_pocket_dynamics_rare.to_json()
        parsed = json.loads(j)
        assert parsed["msm_state_weights"] is None

    def test_transient_classification(self):
        pd = PocketDynamics(
            pocket_id=1, p_open=0.3, p_open_error=0.05,
            mean_open_lifetime_ns=1.0, mean_closed_lifetime_ns=2.5,
            n_opening_events=20,
            druggability_classification="TRANSIENT",
            volume_autocorrelation_ns=1.5,
        )
        j = pd.to_json()
        pd2 = PocketDynamics.from_json(j)
        assert pd2.druggability_classification == "TRANSIENT"
        assert pd2.msm_state_weights is None

    def test_boundary_p_open_zero(self):
        pd = PocketDynamics(
            pocket_id=5, p_open=0.0, p_open_error=0.0,
            mean_open_lifetime_ns=0.0, mean_closed_lifetime_ns=10.0,
            n_opening_events=0,
            druggability_classification="RARE_EVENT",
            volume_autocorrelation_ns=8.0,
        )
        d = pd.to_dict()
        pd2 = PocketDynamics.from_dict(d)
        assert pd2.p_open == 0.0
        assert pd2.n_opening_events == 0

    def test_boundary_p_open_one(self):
        pd = PocketDynamics(
            pocket_id=6, p_open=1.0, p_open_error=0.0,
            mean_open_lifetime_ns=100.0, mean_closed_lifetime_ns=0.0,
            n_opening_events=0,
            druggability_classification="STABLE_OPEN",
            volume_autocorrelation_ns=0.1,
        )
        j = pd.to_json()
        pd2 = PocketDynamics.from_json(j)
        assert pd2.p_open == 1.0
