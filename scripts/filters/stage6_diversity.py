"""Stage 6 — Diversity filter: Butina clustering + representative selection.

Clusters molecules by Morgan fingerprint Tanimoto distance using the
Butina algorithm.  Selects the best representative from each cluster
(by pharmacophore match score).  Ensures sufficient structural diversity
in the final set.
"""
from __future__ import annotations

import logging
import math
from typing import List, Tuple

from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem
from rdkit.ML.Cluster import Butina

from scripts.interfaces import GeneratedMolecule
from scripts.filters.stage2_druglikeness import DrugLikenessResult
from scripts.filters.stage5_novelty import compute_morgan_fp

logger = logging.getLogger(__name__)


def _compute_distance_matrix(
    fps: List[DataStructs.ExplicitBitVect],
) -> List[float]:
    """Compute condensed distance matrix (1 - Tanimoto) for Butina clustering.

    Returns flat lower-triangular distance list as expected by Butina.
    """
    n = len(fps)
    dists: List[float] = []
    for i in range(1, n):
        for j in range(i):
            sim = DataStructs.TanimotoSimilarity(fps[i], fps[j])
            dists.append(1.0 - sim)
    return dists


def cluster_molecules(
    fps: List[DataStructs.ExplicitBitVect],
    cutoff: float = 0.4,
) -> List[List[int]]:
    """Butina clustering of fingerprints.

    Args:
        fps: List of fingerprint bit vectors.
        cutoff: Tanimoto distance cutoff (default 0.4 = 60% similarity).

    Returns:
        List of clusters, each a list of molecule indices.
        Clusters are sorted largest-first.
    """
    if len(fps) == 0:
        return []
    if len(fps) == 1:
        return [[0]]

    dists = _compute_distance_matrix(fps)
    clusters_raw = Butina.ClusterData(dists, len(fps), cutoff, isDistData=True)

    # Convert tuples to lists and sort by size (largest first)
    clusters = [list(c) for c in clusters_raw]
    clusters.sort(key=len, reverse=True)
    return clusters


def run(
    molecules: List[Tuple[GeneratedMolecule, DrugLikenessResult, List[str], int, List[str], float, str]],
    diversity_cutoff: float = 0.4,
    top_n: int = 5,
) -> Tuple[
    List[Tuple[GeneratedMolecule, DrugLikenessResult, List[str], int, List[str], float, str, int]],
    List[Tuple[GeneratedMolecule, str]],
]:
    """Select diverse representatives via Butina clustering.

    Strategy:
    1. Cluster all molecules by Tanimoto distance.
    2. From each cluster, pick the molecule with the best pharmacophore
       match score.
    3. Take top_n representatives, ensuring >= ceil(top_n/2) distinct
       clusters when possible.

    Args:
        molecules: Tuples from stage 5.
        diversity_cutoff: Butina distance cutoff (default 0.4).
        top_n: Maximum number of molecules to keep.

    Returns:
        (passed, rejected) — passed adds cluster_id.
    """
    if not molecules:
        return [], []

    # Compute fingerprints
    fps: List[DataStructs.ExplicitBitVect] = []
    valid_indices: List[int] = []
    for i, (mol_obj, *_rest) in enumerate(molecules):
        mol = Chem.MolFromSmiles(mol_obj.smiles)
        if mol is not None:
            fps.append(compute_morgan_fp(mol))
            valid_indices.append(i)

    # Cluster
    clusters = cluster_molecules(fps, cutoff=diversity_cutoff)

    # Build index → cluster_id mapping
    idx_to_cluster: dict[int, int] = {}
    for cluster_id, members in enumerate(clusters):
        for member_idx in members:
            # member_idx is index into fps/valid_indices
            orig_idx = valid_indices[member_idx]
            idx_to_cluster[orig_idx] = cluster_id

    # Select best representative per cluster (by pharmacophore match score)
    cluster_best: dict[int, int] = {}  # cluster_id → best original index
    for orig_idx, cluster_id in idx_to_cluster.items():
        mol_obj = molecules[orig_idx][0]
        score = mol_obj.pharmacophore_match_score
        if cluster_id not in cluster_best:
            cluster_best[cluster_id] = orig_idx
        else:
            current_best = molecules[cluster_best[cluster_id]][0].pharmacophore_match_score
            if score > current_best:
                cluster_best[cluster_id] = orig_idx

    # Ensure diversity: take one from each cluster first, then fill
    min_clusters = math.ceil(top_n / 2)
    selected_indices: List[int] = []

    # Phase 1: one best per cluster
    for cluster_id in range(len(clusters)):
        if cluster_id in cluster_best:
            selected_indices.append(cluster_best[cluster_id])
        if len(selected_indices) >= top_n:
            break

    # Phase 2: if we have room, add remaining molecules ranked by pharm score
    if len(selected_indices) < top_n:
        selected_set = set(selected_indices)
        remaining = [
            (i, molecules[i][0].pharmacophore_match_score)
            for i in idx_to_cluster
            if i not in selected_set
        ]
        remaining.sort(key=lambda x: x[1], reverse=True)
        for idx, _ in remaining:
            if len(selected_indices) >= top_n:
                break
            selected_indices.append(idx)

    selected_set = set(selected_indices)

    passed: List[Tuple[GeneratedMolecule, DrugLikenessResult, List[str], int, List[str], float, str, int]] = []
    rejected: List[Tuple[GeneratedMolecule, str]] = []

    for i, entry in enumerate(molecules):
        mol_obj = entry[0]
        if i in selected_set:
            cluster_id = idx_to_cluster.get(i, -1)
            passed.append((*entry, cluster_id))
        else:
            cluster_id = idx_to_cluster.get(i, -1)
            rejected.append((mol_obj, f"diversity selection: not in top-{top_n} (cluster {cluster_id})"))

    logger.info(
        "Stage 6 (diversity): %d in → %d passed (%d clusters), %d rejected",
        len(molecules), len(passed), len(set(e[-1] for e in passed)), len(rejected),
    )
    return passed, rejected
