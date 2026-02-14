"""Stage 3 — PAINS substructure alert filter.

Uses RDKit's built-in FilterCatalog (480 PAINS patterns) to detect
Pan-Assay Interference compoundS.  Returns alert names for flagged
compounds so they can be recorded in FilteredCandidate.pains_alerts.
"""
from __future__ import annotations

import logging
from typing import List, Tuple

from rdkit import Chem
from rdkit.Chem import FilterCatalog

from scripts.interfaces import GeneratedMolecule
from scripts.filters.stage2_druglikeness import DrugLikenessResult

logger = logging.getLogger(__name__)

# Build the PAINS catalog once at import time
_PAINS_PARAMS = FilterCatalog.FilterCatalogParams()
_PAINS_PARAMS.AddCatalog(FilterCatalog.FilterCatalogParams.FilterCatalogs.PAINS)
PAINS_CATALOG = FilterCatalog.FilterCatalog(_PAINS_PARAMS)


def get_pains_alerts(mol: Chem.Mol) -> List[str]:
    """Return list of PAINS alert names for a molecule."""
    alerts: List[str] = []
    matches = PAINS_CATALOG.GetMatches(mol)
    for match in matches:
        alerts.append(match.GetDescription())
    return alerts


def run(
    molecules: List[Tuple[GeneratedMolecule, DrugLikenessResult]],
    reject_pains: bool = True,
) -> Tuple[
    List[Tuple[GeneratedMolecule, DrugLikenessResult, List[str]]],
    List[Tuple[GeneratedMolecule, str]],
]:
    """Screen molecules for PAINS alerts.

    Args:
        molecules: (molecule, metrics) tuples from stage 2.
        reject_pains: If True, molecules with any PAINS alert are rejected.

    Returns:
        (passed, rejected) — passed includes PAINS alert list (empty if clean).
    """
    passed: List[Tuple[GeneratedMolecule, DrugLikenessResult, List[str]]] = []
    rejected: List[Tuple[GeneratedMolecule, str]] = []

    for mol_obj, metrics in molecules:
        mol = Chem.MolFromSmiles(mol_obj.smiles)
        if mol is None:
            rejected.append((mol_obj, "SMILES parse failed in stage3"))
            continue

        alerts = get_pains_alerts(mol)

        if alerts and reject_pains:
            rejected.append(
                (mol_obj, f"PAINS alerts: {', '.join(alerts)}")
            )
        else:
            passed.append((mol_obj, metrics, alerts))

    logger.info(
        "Stage 3 (PAINS): %d in → %d passed, %d rejected",
        len(molecules), len(passed), len(rejected),
    )
    return passed, rejected
