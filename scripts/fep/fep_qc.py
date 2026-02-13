"""FEP Quality-Control gates — automatic pass/fail checks for ABFE/RBFE runs.

QC Gates (blueprint spec):
    1. Lambda-window overlap >= 0.03
    2. Forward/reverse hysteresis <= 1.5 kcal/mol
    3. Protein Ca RMSD <= 4.0 A
    4. Ligand stays in pocket (COM < 5 A from initial)
    5. >= 2/3 repeats converge within 1.0 kcal/mol
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

logger = logging.getLogger(__name__)

# ── Default thresholds (overridable) ──────────────────────────────────────

OVERLAP_MIN: float = 0.03
HYSTERESIS_MAX_KCAL: float = 1.5
PROTEIN_RMSD_MAX_A: float = 4.0
LIGAND_COM_MAX_A: float = 5.0
REPEAT_CONVERGENCE_KCAL: float = 1.0
MIN_CONVERGED_FRACTION: float = 2 / 3


@dataclass
class QCGateResult:
    """Result of a single QC gate evaluation."""
    gate_name: str
    passed: bool
    measured_value: float
    threshold: float
    message: str = ""


@dataclass
class FEPQCReport:
    """Aggregated QC report for a single compound's FEP calculation."""
    compound_id: str
    gate_results: List[QCGateResult] = field(default_factory=list)

    @property
    def all_passed(self) -> bool:
        return all(g.passed for g in self.gate_results)

    @property
    def failed_gates(self) -> List[str]:
        return [g.gate_name for g in self.gate_results if not g.passed]

    def summary(self) -> str:
        status = "PASS" if self.all_passed else "FAIL"
        lines = [f"QC Report [{status}] — {self.compound_id}"]
        for g in self.gate_results:
            mark = "OK" if g.passed else "FAIL"
            lines.append(
                f"  [{mark}] {g.gate_name}: "
                f"{g.measured_value:.4f} (threshold: {g.threshold:.4f})"
            )
            if g.message:
                lines.append(f"        {g.message}")
        return "\n".join(lines)


# ── Individual gate functions ─────────────────────────────────────────────

def check_overlap(
    overlap_matrix: Sequence[float],
    threshold: float = OVERLAP_MIN,
) -> QCGateResult:
    """Gate 1: minimum lambda-window overlap across all adjacent pairs.

    Args:
        overlap_matrix: Overlap values between adjacent lambda windows.
            Typically from MBAR overlap matrix diagonal ± 1.
        threshold: Minimum acceptable overlap (default 0.03).
    """
    if not overlap_matrix:
        return QCGateResult(
            gate_name="lambda_overlap",
            passed=False,
            measured_value=0.0,
            threshold=threshold,
            message="No overlap data provided",
        )
    min_overlap = min(overlap_matrix)
    return QCGateResult(
        gate_name="lambda_overlap",
        passed=min_overlap >= threshold,
        measured_value=min_overlap,
        threshold=threshold,
    )


def check_hysteresis(
    forward_dg: float,
    reverse_dg: float,
    threshold: float = HYSTERESIS_MAX_KCAL,
) -> QCGateResult:
    """Gate 2: forward/reverse free-energy hysteresis.

    Args:
        forward_dg: Free energy from forward (lambda 0→1) integration.
        reverse_dg: Free energy from reverse (lambda 1→0) integration.
        threshold: Maximum acceptable |forward - reverse| in kcal/mol.
    """
    hysteresis = abs(forward_dg - reverse_dg)
    return QCGateResult(
        gate_name="hysteresis",
        passed=hysteresis <= threshold,
        measured_value=hysteresis,
        threshold=threshold,
    )


def check_protein_rmsd(
    max_rmsd: float,
    threshold: float = PROTEIN_RMSD_MAX_A,
) -> QCGateResult:
    """Gate 3: maximum protein backbone RMSD during simulation.

    Args:
        max_rmsd: Maximum Ca RMSD observed across all frames (Angstrom).
        threshold: Maximum acceptable RMSD (default 4.0 A).
    """
    return QCGateResult(
        gate_name="protein_rmsd",
        passed=max_rmsd <= threshold,
        measured_value=max_rmsd,
        threshold=threshold,
    )


def check_ligand_in_pocket(
    max_com_drift: float,
    threshold: float = LIGAND_COM_MAX_A,
) -> QCGateResult:
    """Gate 4: ligand center-of-mass stays within pocket.

    Args:
        max_com_drift: Max distance of ligand COM from initial position (A).
        threshold: Maximum acceptable COM drift (default 5.0 A).
    """
    return QCGateResult(
        gate_name="ligand_in_pocket",
        passed=max_com_drift <= threshold,
        measured_value=max_com_drift,
        threshold=threshold,
    )


def check_repeat_convergence(
    repeat_dgs: Sequence[float],
    convergence_kcal: float = REPEAT_CONVERGENCE_KCAL,
    min_fraction: float = MIN_CONVERGED_FRACTION,
) -> QCGateResult:
    """Gate 5: sufficient repeat convergence.

    At least ``min_fraction`` of repeats must agree within
    ``convergence_kcal`` of the median value.

    Args:
        repeat_dgs: dG values from each independent repeat.
        convergence_kcal: Max deviation from median for a repeat to
            count as converged (default 1.0 kcal/mol).
        min_fraction: Minimum fraction of repeats that must converge
            (default 2/3).
    """
    if len(repeat_dgs) < 2:
        return QCGateResult(
            gate_name="repeat_convergence",
            passed=len(repeat_dgs) == 1,
            measured_value=1.0 if repeat_dgs else 0.0,
            threshold=min_fraction,
            message="Single repeat — convergence trivially satisfied"
            if repeat_dgs else "No repeats provided",
        )

    sorted_dgs = sorted(repeat_dgs)
    n = len(sorted_dgs)
    median = sorted_dgs[n // 2] if n % 2 else (sorted_dgs[n // 2 - 1] + sorted_dgs[n // 2]) / 2.0

    converged = sum(1 for dg in repeat_dgs if abs(dg - median) <= convergence_kcal)
    fraction = converged / len(repeat_dgs)

    return QCGateResult(
        gate_name="repeat_convergence",
        passed=fraction >= min_fraction,
        measured_value=fraction,
        threshold=min_fraction,
        message=f"{converged}/{len(repeat_dgs)} repeats within "
                f"{convergence_kcal} kcal/mol of median ({median:.2f})",
    )


# ── Aggregate runner ──────────────────────────────────────────────────────

def run_all_qc_gates(
    compound_id: str,
    overlap_matrix: Sequence[float],
    forward_dg: float,
    reverse_dg: float,
    max_protein_rmsd: float,
    max_ligand_com_drift: float,
    repeat_dgs: Sequence[float],
    *,
    overlap_threshold: float = OVERLAP_MIN,
    hysteresis_threshold: float = HYSTERESIS_MAX_KCAL,
    rmsd_threshold: float = PROTEIN_RMSD_MAX_A,
    com_threshold: float = LIGAND_COM_MAX_A,
    convergence_kcal: float = REPEAT_CONVERGENCE_KCAL,
    min_converged_fraction: float = MIN_CONVERGED_FRACTION,
) -> FEPQCReport:
    """Run all 5 QC gates and return an aggregated report.

    Returns:
        FEPQCReport with all gate results.
    """
    report = FEPQCReport(compound_id=compound_id)

    report.gate_results.append(
        check_overlap(overlap_matrix, overlap_threshold)
    )
    report.gate_results.append(
        check_hysteresis(forward_dg, reverse_dg, hysteresis_threshold)
    )
    report.gate_results.append(
        check_protein_rmsd(max_protein_rmsd, rmsd_threshold)
    )
    report.gate_results.append(
        check_ligand_in_pocket(max_ligand_com_drift, com_threshold)
    )
    report.gate_results.append(
        check_repeat_convergence(repeat_dgs, convergence_kcal, min_converged_fraction)
    )

    logger.info("QC %s: %s", "PASS" if report.all_passed else "FAIL", compound_id)
    return report
