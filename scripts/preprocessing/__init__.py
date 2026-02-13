"""WT-5 Pre-Processing — tautomer enumeration, membrane embedding, PDB fixing.

Modules
-------
target_classifier
    Auto-detect membrane vs soluble protein (OPM → UniProt → hydrophobicity belt).
protein_fixer
    Repair PDB files: missing residues/atoms, non-standard residues, alt conformations.
tautomer_enumeration
    Enumerate tautomers & protonation states at physiological pH (Dimorphite-DL + RDKit).
membrane_builder
    Automated lipid-bilayer embedding for membrane targets (packmol-memgen + OPM).
"""

from .target_classifier import ClassificationResult, classify_target
from .protein_fixer import FixerResult, fix_protein
from .tautomer_enumeration import (
    enumerate_batch,
    enumerate_tautomers,
)
from .membrane_builder import build_membrane

__all__ = [
    "ClassificationResult",
    "classify_target",
    "FixerResult",
    "fix_protein",
    "enumerate_tautomers",
    "enumerate_batch",
    "build_membrane",
]
