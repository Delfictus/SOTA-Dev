"""Stage 2 — Drug-likeness filter: Lipinski, QED, SA score.

Computes quantitative drug-likeness metrics and rejects molecules that
fall outside acceptable thresholds.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Tuple

from rdkit import Chem
from rdkit.Chem import Descriptors, QED as QEDModule
from rdkit.Contrib.SA_Score import sascorer

from scripts.interfaces import GeneratedMolecule, FilterConfig

logger = logging.getLogger(__name__)


@dataclass
class DrugLikenessResult:
    """Computed drug-likeness metrics for a single molecule."""
    qed: float
    sa_score: float
    lipinski_violations: int
    mw: float
    logp: float
    hbd: int
    hba: int


def compute_lipinski_violations(mol: Chem.Mol) -> int:
    """Count Lipinski Rule-of-Five violations."""
    violations = 0
    if Descriptors.MolWt(mol) > 500:
        violations += 1
    if Descriptors.MolLogP(mol) > 5:
        violations += 1
    if Descriptors.NumHDonors(mol) > 5:
        violations += 1
    if Descriptors.NumHAcceptors(mol) > 10:
        violations += 1
    return violations


def compute_metrics(mol: Chem.Mol) -> DrugLikenessResult:
    """Compute all drug-likeness metrics for an RDKit Mol."""
    return DrugLikenessResult(
        qed=QEDModule.qed(mol),
        sa_score=sascorer.calculateScore(mol),
        lipinski_violations=compute_lipinski_violations(mol),
        mw=Descriptors.MolWt(mol),
        logp=Descriptors.MolLogP(mol),
        hbd=Descriptors.NumHDonors(mol),
        hba=Descriptors.NumHAcceptors(mol),
    )


def run(
    molecules: List[GeneratedMolecule],
    config: FilterConfig,
) -> Tuple[
    List[Tuple[GeneratedMolecule, DrugLikenessResult]],
    List[Tuple[GeneratedMolecule, str]],
]:
    """Filter molecules by drug-likeness thresholds.

    Returns:
        (passed, rejected) — passed includes computed metrics for downstream use.
    """
    passed: List[Tuple[GeneratedMolecule, DrugLikenessResult]] = []
    rejected: List[Tuple[GeneratedMolecule, str]] = []

    for mol_obj in molecules:
        mol = Chem.MolFromSmiles(mol_obj.smiles)
        if mol is None:
            rejected.append((mol_obj, "SMILES parse failed in stage2"))
            continue

        metrics = compute_metrics(mol)

        reasons: List[str] = []
        if metrics.qed < config.qed_min:
            reasons.append(f"QED {metrics.qed:.3f} < {config.qed_min}")
        if metrics.sa_score > config.sa_max:
            reasons.append(f"SA {metrics.sa_score:.2f} > {config.sa_max}")
        if metrics.lipinski_violations > config.lipinski_max_violations:
            reasons.append(
                f"Lipinski violations {metrics.lipinski_violations} > "
                f"{config.lipinski_max_violations}"
            )

        if reasons:
            rejected.append((mol_obj, "; ".join(reasons)))
        else:
            passed.append((mol_obj, metrics))

    logger.info(
        "Stage 2 (drug-likeness): %d in → %d passed, %d rejected",
        len(molecules), len(passed), len(rejected),
    )
    return passed, rejected
