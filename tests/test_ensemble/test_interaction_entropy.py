"""Tests for interaction_entropy.py — IE method (Duan et al., JACS 2016)."""
from __future__ import annotations

import numpy as np
import pytest

from scripts.interfaces import InteractionEntropy
from scripts.ensemble.interaction_entropy import (
    KB_KCAL,
    DEFAULT_TEMPERATURE_K,
    compute_ie,
    compute_interaction_energy,
    compute_interaction_entropy,
    ie_from_mmgbsa_components,
)


# ═══════════════════════════════════════════════════════════════════════════
#  Interaction energy computation
# ═══════════════════════════════════════════════════════════════════════════

class TestComputeInteractionEnergy:
    def test_basic_subtraction(self):
        e_c = np.array([-100.0, -95.0])
        e_r = np.array([-60.0, -58.0])
        e_l = np.array([-10.0, -9.0])
        result = compute_interaction_energy(e_c, e_r, e_l)
        np.testing.assert_allclose(result, [-30.0, -28.0])

    def test_length_mismatch_raises(self):
        with pytest.raises(ValueError, match="mismatch"):
            compute_interaction_energy(
                np.array([1.0, 2.0]), np.array([1.0]), np.array([1.0, 2.0])
            )

    def test_zeros(self):
        e = np.zeros(5)
        result = compute_interaction_energy(e, e, e)
        np.testing.assert_allclose(result, np.zeros(5))


# ═══════════════════════════════════════════════════════════════════════════
#  Core IE computation
# ═══════════════════════════════════════════════════════════════════════════

class TestComputeIE:
    def test_constant_energy_zero_entropy(self):
        """If all interaction energies are identical, -TdS should be 0."""
        e_int = np.full(100, -40.0)
        minus_tds, mean_e = compute_ie(e_int)
        assert abs(minus_tds) < 1e-10
        assert abs(mean_e - (-40.0)) < 1e-10

    def test_positive_entropy_penalty(self, interaction_energies):
        """-TdS should be positive (unfavorable entropy) for fluctuating energies."""
        minus_tds, mean_e = compute_ie(interaction_energies)
        assert minus_tds > 0

    def test_larger_fluctuations_larger_entropy(self):
        """More fluctuation → larger -TdS."""
        rng = np.random.default_rng(42)
        e_small_fluct = rng.normal(-40.0, 1.0, 200)
        e_large_fluct = rng.normal(-40.0, 10.0, 200)

        tds_small, _ = compute_ie(e_small_fluct)
        tds_large, _ = compute_ie(e_large_fluct)
        assert tds_large > tds_small

    def test_temperature_dependence(self, interaction_energies):
        """Higher temperature → smaller -TdS (≈ σ²/(2kT) for Gaussian)."""
        tds_300, _ = compute_ie(interaction_energies, 300.0)
        tds_400, _ = compute_ie(interaction_energies, 400.0)
        assert tds_400 < tds_300

    def test_empty_array_raises(self):
        with pytest.raises(ValueError, match="Empty"):
            compute_ie(np.array([]))

    def test_single_value(self):
        """Single frame → -TdS = 0 (no fluctuation)."""
        minus_tds, mean_e = compute_ie(np.array([-40.0]))
        assert abs(minus_tds) < 1e-10

    def test_numerical_stability(self):
        """Large energy fluctuations should not cause overflow."""
        rng = np.random.default_rng(42)
        e_int = rng.normal(-40.0, 50.0, 500)
        minus_tds, mean_e = compute_ie(e_int)
        assert np.isfinite(minus_tds)
        assert np.isfinite(mean_e)


# ═══════════════════════════════════════════════════════════════════════════
#  Full InteractionEntropy interface
# ═══════════════════════════════════════════════════════════════════════════

class TestComputeInteractionEntropy:
    def test_returns_interface(self, interaction_energies):
        result = compute_interaction_entropy(
            interaction_energies, delta_h=-8.0, compound_id="CMPD001"
        )
        assert isinstance(result, InteractionEntropy)
        assert result.compound_id == "CMPD001"
        assert result.n_frames == 200

    def test_delta_g_ie_sum(self, interaction_energies):
        """dG_IE should equal dH + (-TdS)."""
        result = compute_interaction_entropy(
            interaction_energies, delta_h=-8.0, compound_id="CMPD001"
        )
        expected_dg = result.delta_h + result.minus_t_delta_s
        assert abs(result.delta_g_ie - expected_dg) < 1e-10

    def test_default_delta_h_uses_mean(self, interaction_energies):
        """If delta_h is None, use mean interaction energy."""
        result = compute_interaction_entropy(interaction_energies, compound_id="X")
        expected_mean = float(np.mean(interaction_energies))
        assert abs(result.delta_h - expected_mean) < 1e-10

    def test_convergence_block_std(self, interaction_energies):
        """Block SEM should be positive."""
        result = compute_interaction_entropy(interaction_energies, compound_id="X")
        assert result.convergence_block_std > 0

    def test_too_few_frames_raises(self):
        with pytest.raises(ValueError, match="at least 5"):
            compute_interaction_entropy(np.array([1.0, 2.0, 3.0]))


# ═══════════════════════════════════════════════════════════════════════════
#  Convenience: IE from MM-GBSA components
# ═══════════════════════════════════════════════════════════════════════════

class TestIEFromMMGBSA:
    def test_basic(self):
        rng = np.random.default_rng(42)
        vdw = rng.normal(-25.0, 3.0, 100)
        elec = rng.normal(-15.0, 5.0, 100)
        result = ie_from_mmgbsa_components(vdw, elec, -8.0, "CMPD002")
        assert isinstance(result, InteractionEntropy)
        assert result.delta_h == -8.0
        assert result.n_frames == 100

    def test_length_mismatch_raises(self):
        with pytest.raises(ValueError, match="mismatch"):
            ie_from_mmgbsa_components(
                np.array([1.0]), np.array([1.0, 2.0]), -8.0
            )


# ═══════════════════════════════════════════════════════════════════════════
#  Serialisation
# ═══════════════════════════════════════════════════════════════════════════

class TestSerialisation:
    def test_json_roundtrip(self, sample_interaction_entropy):
        j = sample_interaction_entropy.to_json()
        loaded = InteractionEntropy.from_json(j)
        assert loaded.compound_id == "TEST001"
        assert abs(loaded.delta_g_ie - 3.3) < 1e-10
        assert loaded.n_frames == 200

    def test_pickle_roundtrip(self, sample_interaction_entropy):
        data = sample_interaction_entropy.to_pickle()
        loaded = InteractionEntropy.from_pickle(data)
        assert loaded.compound_id == "TEST001"
        assert abs(loaded.minus_t_delta_s - 11.5) < 1e-10
