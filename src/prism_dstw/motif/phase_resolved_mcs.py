"""Phase-resolved maximum common substructure extraction."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence, cast

import numpy as np
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem, rdFMCS
from rdkit.DataStructs.cDataStructs import ExplicitBitVect
from rdkit.ML.Cluster import Butina


@dataclass(frozen=True)
class PhaseResolvedMCS:
    """MCS motif with PRISM phase annotations."""

    smarts: str
    n_molecules: int
    cluster_id: int
    phase_profile_centroid: np.ndarray
    phase_profile_variance: np.ndarray
    hysteresis_score: float
    variant_resilience_mean: float
    variant_resilience_worst: float
    is_evolutionary_invariant: bool
    tanimoto_cohesion: float


def extract_phase_resolved_mcs(
    candidates: Sequence[Chem.Mol],
    candidate_phase_profiles: Sequence[np.ndarray],
    *,
    tanimoto_threshold: float = 0.3,
    mcs_timeout_seconds: int = 10,
    butina_cutoff: float = 0.6,
) -> list[PhaseResolvedMCS]:
    """Extract phase-resolved MCS motifs with clustering and timeouts."""

    if len(candidates) != len(candidate_phase_profiles):
        raise ValueError("candidates and phase profiles length mismatch")
    if not candidates:
        return []
    fps = [_fingerprint(mol) for mol in candidates]
    clusters = _butina_clusters(fps, butina_cutoff)
    motifs: list[PhaseResolvedMCS] = []
    seen: set[str] = set()
    for cluster_id, cluster_indices in enumerate(clusters):
        if len(cluster_indices) < 2:
            continue
        for i, idx_i in enumerate(cluster_indices):
            for idx_j in cluster_indices[i + 1 :]:
                sim = float(DataStructs.TanimotoSimilarity(fps[idx_i], fps[idx_j]))
                if sim < tanimoto_threshold:
                    continue
                result = rdFMCS.FindMCS(
                    [candidates[idx_i], candidates[idx_j]],
                    timeout=int(mcs_timeout_seconds),
                    matchValences=True,
                    ringMatchesRingOnly=True,
                )
                if result.canceled or result.numAtoms < 4 or not result.smartsString:
                    continue
                key = f"{cluster_id}:{result.smartsString}"
                if key in seen:
                    continue
                seen.add(key)
                profiles = np.stack(
                    [
                        _safe_phase(candidate_phase_profiles[idx_i]),
                        _safe_phase(candidate_phase_profiles[idx_j]),
                    ]
                )
                centroid = profiles.mean(axis=0)
                variance = profiles.var(axis=0)
                hysteresis = abs(float(centroid[4] - centroid[0])) / max(abs(float(centroid[1])), 1.0e-8)
                motifs.append(
                    PhaseResolvedMCS(
                        smarts=str(result.smartsString),
                        n_molecules=2,
                        cluster_id=cluster_id,
                        phase_profile_centroid=centroid,
                        phase_profile_variance=variance,
                        hysteresis_score=hysteresis,
                        variant_resilience_mean=float(max(0.0, 1.0 - hysteresis)),
                        variant_resilience_worst=float(max(0.0, 1.0 - hysteresis - float(variance.mean()))),
                        is_evolutionary_invariant=hysteresis < 0.25,
                        tanimoto_cohesion=sim,
                    )
                )
    return motifs


def _fingerprint(mol: Chem.Mol) -> ExplicitBitVect:
    generator_fn = getattr(AllChem, "GetMorganGenerator")
    generator = generator_fn(radius=2, fpSize=2048)
    return cast(ExplicitBitVect, generator.GetFingerprint(mol))


def _butina_clusters(fps: Sequence[ExplicitBitVect], cutoff: float) -> list[tuple[int, ...]]:
    dists: list[float] = []
    n = len(fps)
    for i in range(1, n):
        for j in range(i):
            sim = float(DataStructs.TanimotoSimilarity(fps[i], fps[j]))
            dists.append(1.0 - sim)
    raw_clusters = cast(Sequence[Sequence[int]], Butina.ClusterData(dists, n, cutoff, isDistData=True))  # type: ignore[no-untyped-call]
    return [tuple(int(idx) for idx in cluster) for cluster in raw_clusters]


def _safe_phase(profile: np.ndarray) -> np.ndarray:
    arr = np.asarray(profile, dtype=np.float64)
    if arr.shape != (5,):
        raise ValueError("candidate phase profile must have shape [5]")
    return arr
