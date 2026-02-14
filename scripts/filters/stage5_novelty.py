"""Stage 5 — Novelty filter: Tanimoto similarity vs known compounds.

Uses Morgan fingerprints (radius=2, 2048 bits) and Tanimoto coefficient.
Rejects molecules with similarity > threshold to any reference compound.
"""
from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem

from scripts.interfaces import GeneratedMolecule
from scripts.filters.stage2_druglikeness import DrugLikenessResult

logger = logging.getLogger(__name__)

# Default reference fingerprint directory
_DEFAULT_REF_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "reference_fingerprints"

# Morgan FP parameters matching blueprint spec
_MORGAN_RADIUS = 2
_MORGAN_NBITS = 2048


def compute_morgan_fp(mol: Chem.Mol) -> DataStructs.ExplicitBitVect:
    """Compute Morgan fingerprint (ECFP4 equivalent)."""
    return AllChem.GetMorganFingerprintAsBitVect(mol, _MORGAN_RADIUS, nBits=_MORGAN_NBITS)


def load_reference_fingerprints(
    ref_dir: Optional[Path] = None,
) -> List[Tuple[str, DataStructs.ExplicitBitVect]]:
    """Load reference compound fingerprints from JSON files.

    Each JSON file should contain a list of objects with 'cid' and 'smiles' fields.
    Returns list of (cid, fingerprint) tuples.
    """
    ref_dir = ref_dir or _DEFAULT_REF_DIR
    ref_fps: List[Tuple[str, DataStructs.ExplicitBitVect]] = []

    if not ref_dir.exists():
        logger.warning("Reference fingerprint directory %s not found; novelty check will pass all", ref_dir)
        return ref_fps

    for fp_file in sorted(ref_dir.glob("*.json")):
        try:
            with open(fp_file) as f:
                entries = json.load(f)
            for entry in entries:
                mol = Chem.MolFromSmiles(entry["smiles"])
                if mol is not None:
                    fp = compute_morgan_fp(mol)
                    ref_fps.append((str(entry.get("cid", "")), fp))
        except Exception as exc:
            logger.warning("Failed to load reference file %s: %s", fp_file, exc)

    logger.info("Loaded %d reference fingerprints", len(ref_fps))
    return ref_fps


def compute_max_tanimoto(
    query_fp: DataStructs.ExplicitBitVect,
    reference_fps: List[Tuple[str, DataStructs.ExplicitBitVect]],
) -> Tuple[float, str]:
    """Find the maximum Tanimoto similarity to any reference compound.

    Returns:
        (max_similarity, nearest_cid) — 0.0 and "" if no references.
    """
    if not reference_fps:
        return 0.0, ""

    max_sim = 0.0
    nearest_cid = ""

    for cid, ref_fp in reference_fps:
        sim = DataStructs.TanimotoSimilarity(query_fp, ref_fp)
        if sim > max_sim:
            max_sim = sim
            nearest_cid = cid

    return max_sim, nearest_cid


def run(
    molecules: List[Tuple[GeneratedMolecule, DrugLikenessResult, List[str], int, List[str]]],
    tanimoto_max: float = 0.85,
    reference_fps: Optional[List[Tuple[str, DataStructs.ExplicitBitVect]]] = None,
    ref_dir: Optional[Path] = None,
) -> Tuple[
    List[Tuple[GeneratedMolecule, DrugLikenessResult, List[str], int, List[str], float, str]],
    List[Tuple[GeneratedMolecule, str]],
]:
    """Filter molecules by novelty (Tanimoto similarity).

    Args:
        molecules: Tuples from stage 4.
        tanimoto_max: Maximum allowed Tanimoto similarity (default 0.85).
        reference_fps: Pre-loaded reference fingerprints (avoids re-loading).
        ref_dir: Directory containing reference fingerprint JSON files.

    Returns:
        (passed, rejected) — passed adds (tanimoto_score, nearest_cid).
    """
    if reference_fps is None:
        reference_fps = load_reference_fingerprints(ref_dir)

    passed: List[Tuple[GeneratedMolecule, DrugLikenessResult, List[str], int, List[str], float, str]] = []
    rejected: List[Tuple[GeneratedMolecule, str]] = []

    for mol_obj, metrics, pains, n_matched, matched_types in molecules:
        mol = Chem.MolFromSmiles(mol_obj.smiles)
        if mol is None:
            rejected.append((mol_obj, "SMILES parse failed in stage5"))
            continue

        query_fp = compute_morgan_fp(mol)
        max_sim, nearest_cid = compute_max_tanimoto(query_fp, reference_fps)

        if max_sim > tanimoto_max:
            rejected.append(
                (mol_obj, f"Tanimoto {max_sim:.3f} > {tanimoto_max} (nearest CID: {nearest_cid})")
            )
        else:
            passed.append((mol_obj, metrics, pains, n_matched, matched_types, max_sim, nearest_cid))

    logger.info(
        "Stage 5 (novelty): %d in → %d passed, %d rejected",
        len(molecules), len(passed), len(rejected),
    )
    return passed, rejected
