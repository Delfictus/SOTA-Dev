"""Multi-objective Pareto ranking for filtered candidates.

Ranks molecules by combining:
  - QED (higher is better)
  - SA score (lower is better)
  - Pharmacophore match score (higher is better)
  - Novelty / low Tanimoto to known (lower is better = more novel)

Uses Pareto dominance to find non-dominated fronts, then ranks within
each front by a weighted composite score.
"""
from __future__ import annotations

import logging
from typing import List, Tuple

from scripts.interfaces import FilteredCandidate, GeneratedMolecule
from scripts.filters.stage2_druglikeness import DrugLikenessResult

logger = logging.getLogger(__name__)

# Objective weights for composite scoring (within a Pareto front)
_WEIGHTS = {
    "qed": 0.30,
    "sa": 0.20,           # inverted: 1 - sa/10
    "pharm_match": 0.35,
    "novelty": 0.15,      # inverted: 1 - tanimoto
}


def _objectives(
    mol_obj: GeneratedMolecule,
    metrics: DrugLikenessResult,
    tanimoto: float,
) -> Tuple[float, float, float, float]:
    """Compute normalised objectives (all higher = better)."""
    return (
        metrics.qed,                              # 0-1, higher better
        1.0 - (metrics.sa_score / 10.0),          # 0-1, higher better
        mol_obj.pharmacophore_match_score,         # 0-1, higher better
        1.0 - tanimoto,                            # 0-1, higher better (more novel)
    )


def _dominates(a: Tuple[float, ...], b: Tuple[float, ...]) -> bool:
    """True if a Pareto-dominates b (at least as good in all, strictly better in one)."""
    at_least_as_good = all(ai >= bi for ai, bi in zip(a, b))
    strictly_better = any(ai > bi for ai, bi in zip(a, b))
    return at_least_as_good and strictly_better


def pareto_fronts(
    objectives: List[Tuple[float, ...]],
) -> List[List[int]]:
    """Compute Pareto fronts by successive non-dominated sorting.

    Returns list of fronts, each a list of indices into the input.
    Front 0 = first Pareto front (best).
    """
    n = len(objectives)
    remaining = set(range(n))
    fronts: List[List[int]] = []

    while remaining:
        front: List[int] = []
        for i in remaining:
            dominated = False
            for j in remaining:
                if i != j and _dominates(objectives[j], objectives[i]):
                    dominated = True
                    break
            if not dominated:
                front.append(i)
        fronts.append(sorted(front))
        remaining -= set(front)

    return fronts


def composite_score(obj: Tuple[float, ...]) -> float:
    """Weighted composite score for ranking within a Pareto front."""
    weights = (_WEIGHTS["qed"], _WEIGHTS["sa"], _WEIGHTS["pharm_match"], _WEIGHTS["novelty"])
    return sum(w * v for w, v in zip(weights, obj))


def rank_candidates(
    molecules: List[Tuple[GeneratedMolecule, DrugLikenessResult, List[str], int, List[str], float, str, int]],
    top_n: int = 5,
) -> List[FilteredCandidate]:
    """Rank molecules and return top-N as FilteredCandidate objects.

    Args:
        molecules: Tuples from stage 6 (mol, metrics, pains, n_matched,
                   matched_types, tanimoto, nearest_cid, cluster_id).
        top_n: Number of candidates to return.

    Returns:
        Ranked list of FilteredCandidate objects.
    """
    if not molecules:
        return []

    # Compute objectives
    objs = []
    for mol_obj, metrics, _, _, _, tanimoto, _, _ in molecules:
        objs.append(_objectives(mol_obj, metrics, tanimoto))

    # Pareto fronts
    fronts = pareto_fronts(objs)

    # Rank: front order, then composite score within front
    ranked_indices: List[int] = []
    for front in fronts:
        front_scored = [(idx, composite_score(objs[idx])) for idx in front]
        front_scored.sort(key=lambda x: x[1], reverse=True)
        ranked_indices.extend(idx for idx, _ in front_scored)

    # Build FilteredCandidate objects for top-N
    result: List[FilteredCandidate] = []
    for rank, idx in enumerate(ranked_indices[:top_n]):
        mol_obj, metrics, pains, n_matched, matched_types, tanimoto, nearest_cid, cluster_id = molecules[idx]
        candidate = FilteredCandidate(
            molecule=mol_obj,
            qed_score=metrics.qed,
            sa_score=metrics.sa_score,
            lipinski_violations=metrics.lipinski_violations,
            pains_alerts=pains,
            tanimoto_to_nearest_known=tanimoto,
            nearest_known_cid=nearest_cid,
            cluster_id=cluster_id,
            passed_all_filters=True,
            rejection_reason=None,
        )
        result.append(candidate)

    logger.info(
        "Ranking: %d molecules → %d fronts → top-%d selected",
        len(molecules), len(fronts), len(result),
    )
    return result
