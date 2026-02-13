"""Stage 4 — Pharmacophore re-validation.

For each molecule with a 3D conformer, identify pharmacophore-relevant
functional groups (AR, HBD, HBA, HY, PI, NI), compute their 3D
positions, and match against the SpikePharmacophore features.

A molecule passes if >= min_matches features align within
distance_tolerance Angstrom of the corresponding spike feature.
"""
from __future__ import annotations

import logging
import math
from typing import Dict, List, Optional, Tuple

from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, rdMolDescriptors

from scripts.interfaces import GeneratedMolecule, PharmacophoreFeature, SpikePharmacophore
from scripts.filters.stage2_druglikeness import DrugLikenessResult

logger = logging.getLogger(__name__)

# SMARTS definitions for pharmacophore feature types
_FEATURE_SMARTS: Dict[str, List[str]] = {
    "AR": ["a1aaaaa1", "a1aaaa1"],                                    # aromatic rings
    "HBD": ["[#7H,$([#7H2])]", "[OH]", "[NH]"],                      # H-bond donors
    "HBA": ["[#7;!$([nH]);!$([#7H2])]", "[O;!$([OH])]", "[#8H0]"],  # H-bond acceptors
    "HY": ["[CH3]", "[CH2;X2]", "[$([cH0]);!$(c=O)]"],              # hydrophobic
    "PI": ["[NH3+]", "[nH+]", "[NH2+]", "[$([#7;+;!$([N]~[O])])]"], # positive ionizable
    "NI": ["[C;$(C(=O)[OH])]", "[S;$(S(=O)(=O)[OH])]", "[#15;$(P(=O)([OH]))]"], # negative ionizable
}

# Compiled SMARTS patterns (built once)
_COMPILED_SMARTS: Dict[str, List[Chem.Mol]] = {}
for _ftype, _smarts_list in _FEATURE_SMARTS.items():
    _COMPILED_SMARTS[_ftype] = []
    for _sma in _smarts_list:
        _pat = Chem.MolFromSmarts(_sma)
        if _pat is not None:
            _COMPILED_SMARTS[_ftype].append(_pat)


def _get_conformer_coords(mol: Chem.Mol) -> Optional[List[Tuple[float, float, float]]]:
    """Extract 3D coordinates from the first conformer, or None."""
    if mol.GetNumConformers() == 0:
        return None
    conf = mol.GetConformer(0)
    return [
        (conf.GetAtomPosition(i).x, conf.GetAtomPosition(i).y, conf.GetAtomPosition(i).z)
        for i in range(mol.GetNumAtoms())
    ]


def _centroid(coords: List[Tuple[float, float, float]], atom_indices: Tuple[int, ...]) -> Tuple[float, float, float]:
    """Compute centroid of selected atoms."""
    xs = [coords[i][0] for i in atom_indices]
    ys = [coords[i][1] for i in atom_indices]
    zs = [coords[i][2] for i in atom_indices]
    n = len(atom_indices)
    return (sum(xs) / n, sum(ys) / n, sum(zs) / n)


def _distance(a: Tuple[float, float, float], b: Tuple[float, float, float]) -> float:
    return math.sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2 + (a[2] - b[2])**2)


def extract_molecule_features(
    mol: Chem.Mol,
) -> List[Tuple[str, Tuple[float, float, float]]]:
    """Extract pharmacophore features with 3D positions from a molecule.

    Returns list of (feature_type, (x, y, z)) tuples.
    Requires the molecule to have a 3D conformer.
    """
    coords = _get_conformer_coords(mol)
    if coords is None:
        return []

    features: List[Tuple[str, Tuple[float, float, float]]] = []
    seen_atoms: set = set()  # avoid duplicate features from overlapping SMARTS

    for ftype, patterns in _COMPILED_SMARTS.items():
        for pat in patterns:
            for match in mol.GetSubstructMatches(pat):
                key = (ftype, match)
                if key not in seen_atoms:
                    seen_atoms.add(key)
                    pos = _centroid(coords, match)
                    features.append((ftype, pos))

    return features


def match_pharmacophore(
    mol_features: List[Tuple[str, Tuple[float, float, float]]],
    pharmacophore: SpikePharmacophore,
    distance_tolerance: float = 1.5,
) -> Tuple[int, List[str]]:
    """Match molecule features against spike pharmacophore.

    For each pharmacophore feature, find the closest molecule feature of
    the same type within distance_tolerance.  Uses greedy matching (each
    molecule feature can match at most one pharmacophore feature).

    Returns:
        (n_matched, matched_types) — number of matched features and their types.
    """
    matched_types: List[str] = []
    used_mol_features: set = set()

    for pharm_feat in pharmacophore.features:
        best_dist = float("inf")
        best_idx = -1

        for i, (ftype, pos) in enumerate(mol_features):
            if i in used_mol_features:
                continue
            if ftype != pharm_feat.feature_type:
                continue

            dist = _distance(pos, (pharm_feat.x, pharm_feat.y, pharm_feat.z))
            if dist < best_dist:
                best_dist = dist
                best_idx = i

        if best_idx >= 0 and best_dist <= distance_tolerance:
            used_mol_features.add(best_idx)
            matched_types.append(pharm_feat.feature_type)

    return len(matched_types), matched_types


def _ensure_3d(mol: Chem.Mol) -> Optional[Chem.Mol]:
    """Ensure the molecule has a 3D conformer. Embed if needed."""
    if mol.GetNumConformers() > 0:
        return mol

    mol_h = Chem.AddHs(mol)
    params = AllChem.ETKDGv3()
    params.randomSeed = 42
    status = AllChem.EmbedMolecule(mol_h, params)
    if status == -1:
        # Fallback: try without distance geometry constraints
        params.useRandomCoords = True
        status = AllChem.EmbedMolecule(mol_h, params)
        if status == -1:
            return None

    AllChem.MMFFOptimizeMolecule(mol_h, maxIters=200)
    return Chem.RemoveHs(mol_h)


def run(
    molecules: List[Tuple[GeneratedMolecule, DrugLikenessResult, List[str]]],
    pharmacophore: SpikePharmacophore,
    min_matches: int = 3,
    distance_tolerance: float = 1.5,
) -> Tuple[
    List[Tuple[GeneratedMolecule, DrugLikenessResult, List[str], int, List[str]]],
    List[Tuple[GeneratedMolecule, str]],
]:
    """Re-validate 3D pharmacophore overlap.

    Args:
        molecules: (molecule, metrics, pains_alerts) from stage 3.
        pharmacophore: Reference SpikePharmacophore.
        min_matches: Minimum feature matches to pass (default 3).
        distance_tolerance: Max distance in Angstrom for a match (default 1.5).

    Returns:
        (passed, rejected) — passed adds (n_matched, matched_types).
    """
    passed: List[Tuple[GeneratedMolecule, DrugLikenessResult, List[str], int, List[str]]] = []
    rejected: List[Tuple[GeneratedMolecule, str]] = []

    n_features = len(pharmacophore.features)

    for mol_obj, metrics, pains_alerts in molecules:
        # Try mol_block first (already 3D), fall back to SMILES + embed
        mol = None
        if mol_obj.mol_block.strip():
            mol = Chem.MolFromMolBlock(mol_obj.mol_block, removeHs=True)

        if mol is None:
            mol = Chem.MolFromSmiles(mol_obj.smiles)
            if mol is None:
                rejected.append((mol_obj, "SMILES parse failed in stage4"))
                continue

        mol = _ensure_3d(mol)
        if mol is None:
            rejected.append((mol_obj, "3D embedding failed"))
            continue

        mol_features = extract_molecule_features(mol)
        n_matched, matched_types = match_pharmacophore(
            mol_features, pharmacophore, distance_tolerance
        )

        if n_matched < min_matches:
            rejected.append(
                (mol_obj, f"pharmacophore match {n_matched}/{n_features} < {min_matches}")
            )
        else:
            passed.append((mol_obj, metrics, pains_alerts, n_matched, matched_types))

    logger.info(
        "Stage 4 (pharmacophore): %d in → %d passed, %d rejected",
        len(molecules), len(passed), len(rejected),
    )
    return passed, rejected
