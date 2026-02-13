"""Tests for analyze_fep.py — result analysis, corrections, classification."""
from __future__ import annotations

import math

import pytest

from scripts.fep.analyze_fep import (
    classify_result,
    compute_boresch_correction,
    compute_charge_correction,
)


class TestBoreshCorrection:
    def test_typical_values(self):
        corr = compute_boresch_correction(
            distance_a=7.0,
            angle_a_rad=math.pi / 2,
            angle_b_rad=math.pi / 2,
        )
        # Should be a negative correction, typically -2 to -6 kcal/mol
        assert corr < 0
        assert corr > -10.0

    def test_short_distance(self):
        corr_short = compute_boresch_correction(3.0, math.pi / 2, math.pi / 2)
        corr_long = compute_boresch_correction(10.0, math.pi / 2, math.pi / 2)
        # Shorter distance → more negative correction (stronger restraint effect)
        assert corr_short < corr_long

    def test_degenerate_angle(self):
        corr = compute_boresch_correction(7.0, 0.0, 0.0)
        # Should handle gracefully
        assert isinstance(corr, float)

    def test_temperature_dependence(self):
        corr_298 = compute_boresch_correction(7.0, math.pi / 2, math.pi / 2, temperature_k=298.15)
        corr_310 = compute_boresch_correction(7.0, math.pi / 2, math.pi / 2, temperature_k=310.0)
        # Higher temperature → larger magnitude correction
        assert abs(corr_310) > abs(corr_298)


class TestChargeCorrection:
    def test_neutral_ligand(self):
        corr = compute_charge_correction(net_charge=0)
        assert corr == 0.0

    def test_charged_ligand(self):
        corr = compute_charge_correction(net_charge=1, box_length_a=40.0)
        assert corr != 0.0
        assert isinstance(corr, float)

    def test_double_charge(self):
        corr_1 = compute_charge_correction(net_charge=1, box_length_a=40.0)
        corr_2 = compute_charge_correction(net_charge=2, box_length_a=40.0)
        # Correction scales with charge squared
        assert abs(corr_2) > abs(corr_1)

    def test_larger_box(self):
        corr_small = compute_charge_correction(net_charge=1, box_length_a=30.0)
        corr_large = compute_charge_correction(net_charge=1, box_length_a=60.0)
        # Larger box → smaller correction
        assert abs(corr_large) < abs(corr_small)


class TestClassifyResult:
    def test_novel_hit(self):
        assert classify_result(-8.0, True) == "NOVEL_HIT"

    def test_novel_hit_threshold(self):
        assert classify_result(-6.0, True) == "NOVEL_HIT"

    def test_recapitulated(self):
        assert classify_result(-8.0, True, known_binder=True) == "RECAPITULATED"

    def test_weak_binder(self):
        assert classify_result(-4.0, True) == "WEAK_BINDER"

    def test_weak_binder_positive(self):
        assert classify_result(0.0, True) == "WEAK_BINDER"

    def test_failed_qc(self):
        assert classify_result(-8.0, False) == "FAILED_QC"

    def test_failed_qc_overrides_strong_binding(self):
        # Even with excellent dG, failed QC means FAILED_QC
        assert classify_result(-15.0, False) == "FAILED_QC"
