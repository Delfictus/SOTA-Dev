"""Stage 1 — Validity filter: RDKit sanitization.

Rejects molecules whose SMILES cannot be parsed or sanitized by RDKit.
This is the first and cheapest gate in the cascade.
"""
from __future__ import annotations

import logging
from typing import List, Tuple

from rdkit import Chem

from scripts.interfaces import GeneratedMolecule

logger = logging.getLogger(__name__)


def run(
    molecules: List[GeneratedMolecule],
) -> Tuple[List[GeneratedMolecule], List[Tuple[GeneratedMolecule, str]]]:
    """Filter molecules by RDKit parseability.

    Returns:
        (passed, rejected) where rejected is list of (molecule, reason) tuples.
    """
    passed: List[GeneratedMolecule] = []
    rejected: List[Tuple[GeneratedMolecule, str]] = []

    for mol_obj in molecules:
        smi = mol_obj.smiles.strip()
        if not smi:
            rejected.append((mol_obj, "empty SMILES"))
            continue

        mol = Chem.MolFromSmiles(smi, sanitize=False)
        if mol is None:
            rejected.append((mol_obj, f"unparseable SMILES: {smi!r}"))
            continue

        try:
            Chem.SanitizeMol(mol)
        except Exception as exc:
            rejected.append((mol_obj, f"sanitization failed: {exc}"))
            continue

        passed.append(mol_obj)

    logger.info(
        "Stage 1 (validity): %d in → %d passed, %d rejected",
        len(molecules), len(passed), len(rejected),
    )
    return passed, rejected
