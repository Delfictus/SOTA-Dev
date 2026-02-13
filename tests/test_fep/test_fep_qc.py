"""Tests for fep_qc.py — QC gate implementations."""
from __future__ import annotations

import pytest

from scripts.fep.fep_qc import (
    FEPQCReport,
    QCGateResult,
    check_hysteresis,
    check_ligand_in_pocket,
    check_overlap,
    check_protein_rmsd,
    check_repeat_convergence,
    run_all_qc_gates,
)


class TestCheckOverlap:
    def test_pass(self):
        result = check_overlap([0.05, 0.08, 0.12])
        assert result.passed is True
        assert result.measured_value == 0.05

    def test_fail_below_threshold(self):
        result = check_overlap([0.05, 0.02, 0.08])
        assert result.passed is False
        assert result.measured_value == 0.02

    def test_exactly_at_threshold(self):
        result = check_overlap([0.03])
        assert result.passed is True

    def test_empty_list(self):
        result = check_overlap([])
        assert result.passed is False

    def test_custom_threshold(self):
        result = check_overlap([0.04], threshold=0.05)
        assert result.passed is False


class TestCheckHysteresis:
    def test_pass_small_hysteresis(self):
        result = check_hysteresis(-7.5, -7.2)
        assert result.passed is True
        assert abs(result.measured_value - 0.3) < 1e-9

    def test_fail_large_hysteresis(self):
        result = check_hysteresis(-7.5, -5.0)
        assert result.passed is False
        assert abs(result.measured_value - 2.5) < 1e-9

    def test_exactly_at_threshold(self):
        result = check_hysteresis(-7.5, -6.0)
        assert result.passed is True  # 1.5 <= 1.5

    def test_negative_direction(self):
        result = check_hysteresis(-5.0, -7.5)
        assert result.passed is False


class TestCheckProteinRMSD:
    def test_pass(self):
        result = check_protein_rmsd(2.5)
        assert result.passed is True

    def test_fail(self):
        result = check_protein_rmsd(5.0)
        assert result.passed is False

    def test_exactly_at_threshold(self):
        result = check_protein_rmsd(4.0)
        assert result.passed is True


class TestCheckLigandInPocket:
    def test_pass(self):
        result = check_ligand_in_pocket(2.0)
        assert result.passed is True

    def test_fail(self):
        result = check_ligand_in_pocket(6.0)
        assert result.passed is False


class TestCheckRepeatConvergence:
    def test_all_converge(self):
        result = check_repeat_convergence([-7.5, -7.2, -7.8])
        assert result.passed is True
        assert result.measured_value == 1.0

    def test_two_of_three_converge(self):
        result = check_repeat_convergence([-7.5, -2.0, -7.8])
        assert result.passed is True  # 2/3 >= 2/3

    def test_one_of_three_fails(self):
        # median of [-10.0, -2.0, -1.0] is -2.0; only -2.0 and -1.0 are
        # within 1.0 of median → 2/3 converge (passes).
        # Use values where only 1/3 converges:
        result = check_repeat_convergence([-10.0, 0.0, 5.0])
        assert result.passed is False

    def test_single_repeat(self):
        result = check_repeat_convergence([-7.5])
        assert result.passed is True

    def test_empty(self):
        result = check_repeat_convergence([])
        assert result.passed is False

    def test_tight_threshold(self):
        result = check_repeat_convergence([-7.5, -7.2, -7.8], convergence_kcal=0.1)
        assert result.passed is False


class TestRunAllQCGates:
    def test_all_pass(self):
        report = run_all_qc_gates(
            compound_id="test_pass",
            overlap_matrix=[0.05, 0.08],
            forward_dg=-7.5,
            reverse_dg=-7.2,
            max_protein_rmsd=2.0,
            max_ligand_com_drift=1.5,
            repeat_dgs=[-7.5, -7.2, -7.8],
        )
        assert report.all_passed is True
        assert len(report.gate_results) == 5
        assert report.failed_gates == []

    def test_multiple_fail(self):
        report = run_all_qc_gates(
            compound_id="test_fail",
            overlap_matrix=[0.01],
            forward_dg=-7.5,
            reverse_dg=-5.0,
            max_protein_rmsd=5.0,
            max_ligand_com_drift=6.0,
            repeat_dgs=[-7.5, -2.0, -1.0],
        )
        assert report.all_passed is False
        assert "lambda_overlap" in report.failed_gates
        assert "hysteresis" in report.failed_gates
        assert "protein_rmsd" in report.failed_gates
        assert "ligand_in_pocket" in report.failed_gates

    def test_summary_format(self):
        report = run_all_qc_gates(
            compound_id="test_summary",
            overlap_matrix=[0.05],
            forward_dg=-7.5,
            reverse_dg=-7.5,
            max_protein_rmsd=1.0,
            max_ligand_com_drift=1.0,
            repeat_dgs=[-7.5],
        )
        summary = report.summary()
        assert "PASS" in summary
        assert "test_summary" in summary


class TestQCGateResult:
    def test_dataclass(self):
        r = QCGateResult(
            gate_name="test_gate",
            passed=True,
            measured_value=0.5,
            threshold=0.3,
            message="OK",
        )
        assert r.gate_name == "test_gate"
        assert r.passed is True


class TestFEPQCReport:
    def test_empty_report(self):
        report = FEPQCReport(compound_id="empty")
        assert report.all_passed is True
        assert report.failed_gates == []
