"""Tests for FEPResult interface."""
from __future__ import annotations

import json
import pickle

import pytest

from scripts.interfaces.fep_result import FEPResult


class TestFEPResult:
    def test_construction(self, sample_fep_pass):
        fep = sample_fep_pass
        assert fep.compound_id == "cmpd-001"
        assert fep.delta_g_bind == -7.2
        assert fep.method == "ABFE"
        assert fep.classification == "NOVEL_HIT"

    def test_corrected_delta_g(self, sample_fep_pass):
        # -7.2 + (-0.4) + 0.1 = -7.5
        assert abs(sample_fep_pass.corrected_delta_g - (-7.5)) < 1e-6

    def test_passed_qc_true(self, sample_fep_pass):
        assert sample_fep_pass.passed_qc is True

    def test_passed_qc_false(self, sample_fep_fail):
        assert sample_fep_fail.passed_qc is False

    def test_qc_failure_convergence(self, sample_fep_pass):
        """Verify each QC gate independently."""
        import copy
        # Convergence failure
        d = sample_fep_pass.to_dict()
        d["convergence_passed"] = False
        fep = FEPResult.from_dict(d)
        assert fep.passed_qc is False

    def test_qc_failure_hysteresis(self, sample_fep_pass):
        d = sample_fep_pass.to_dict()
        d["hysteresis_kcal"] = 1.5
        fep = FEPResult.from_dict(d)
        assert fep.passed_qc is False

    def test_qc_failure_overlap(self, sample_fep_pass):
        d = sample_fep_pass.to_dict()
        d["overlap_minimum"] = 0.02
        fep = FEPResult.from_dict(d)
        assert fep.passed_qc is False

    def test_qc_failure_rmsd(self, sample_fep_pass):
        d = sample_fep_pass.to_dict()
        d["max_protein_rmsd"] = 3.5
        fep = FEPResult.from_dict(d)
        assert fep.passed_qc is False

    def test_json_round_trip(self, sample_fep_pass):
        j = sample_fep_pass.to_json()
        parsed = json.loads(j)
        # Computed properties included in JSON
        assert "corrected_delta_g" in parsed
        assert "passed_qc" in parsed
        assert parsed["passed_qc"] is True

        fep2 = FEPResult.from_json(j)
        assert fep2.compound_id == "cmpd-001"
        assert fep2.classification == "NOVEL_HIT"

    def test_from_dict_strips_computed(self, sample_fep_pass):
        """from_dict should handle dicts that include computed properties."""
        d = sample_fep_pass.to_dict()
        assert "corrected_delta_g" in d
        fep2 = FEPResult.from_dict(d)
        assert abs(fep2.corrected_delta_g - (-7.5)) < 1e-6

    def test_pickle_round_trip(self, sample_fep_pass):
        data = sample_fep_pass.to_pickle()
        fep2 = FEPResult.from_pickle(data)
        assert fep2.delta_g_bind == -7.2

    def test_pickle_type_check(self):
        bad = pickle.dumps(3.14)
        with pytest.raises(TypeError, match="Expected FEPResult"):
            FEPResult.from_pickle(bad)

    def test_vina_deprecated_none(self):
        fep = FEPResult(
            compound_id="x", delta_g_bind=-5.0, delta_g_error=0.5,
            method="RBFE", n_repeats=3, convergence_passed=True,
            hysteresis_kcal=0.2, overlap_minimum=0.1,
            max_protein_rmsd=1.0, restraint_correction=0.0,
            charge_correction=0.0, vina_score_deprecated=None,
            spike_pharmacophore_match="3/5",
            classification="WEAK_BINDER", raw_data_path="/tmp/x",
        )
        assert fep.vina_score_deprecated is None
        j = fep.to_json()
        fep2 = FEPResult.from_json(j)
        assert fep2.vina_score_deprecated is None
