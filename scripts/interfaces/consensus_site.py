"""Cross-run consensus site interface.

Dataclasses
-----------
MemberSite
    A single site from one replicate run that belongs to a consensus cluster.
ConsensusSite
    A metastable pocket manifold identified across multiple stochastic runs.
ConsensusResult
    Complete consensus output for a target.
"""
from __future__ import annotations

import copy
import json
import pickle
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class MemberSite:
    """A site from one replicate that maps to a consensus cluster."""

    run_id: int
    site_id: int
    centroid: Tuple[float, float, float]
    quality_score: float
    volume: float
    enclosure: float
    therm_class: str
    gate_passed: bool
    blocked_by: Optional[str]
    contact_reorg_strength: float
    response_sharpness: float
    response_energy_density: float
    mean_localization: float
    anchor_residue_ids: List[int]
    n_anchors: int
    lining_residue_ids: List[int]
    growth_vector_directions: List[Tuple[float, float, float]]

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["centroid"] = list(d["centroid"])
        d["growth_vector_directions"] = [list(v) for v in d["growth_vector_directions"]]
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> MemberSite:
        data = copy.deepcopy(d)
        data["centroid"] = tuple(data["centroid"])
        data["growth_vector_directions"] = [
            tuple(v) for v in data.get("growth_vector_directions", [])
        ]
        return cls(**data)


@dataclass
class ConsensusSite:
    """A metastable pocket manifold across stochastic runs.

    Attributes:
        cluster_id:              Consensus cluster index.
        member_sites:            All member sites across runs.
        n_runs_total:            Total number of replicates.
        persistence:             Fraction of runs where site appears (0-1).
        pass_fraction:           Fraction of members that passed all gates.
        centroid_mean:           Mean centroid across members.
        centroid_variance:       RMS spread of centroids (Angstrom).
        mean_quality_score:      Mean GTCKL quality_score.
        mean_contact_reorg:      Mean contact_reorg localization_ratio.
        mean_response_sharpness: Mean response selectivity sharpness.
        anchor_consistency:      Jaccard similarity of anchor residue sets.
        growth_vector_consistency: Cosine similarity of growth vector directions.
        lining_consistency:      Jaccard similarity of lining residue sets.
        mean_localization:       Mean localization score across members.
        gate_failure_reasons:    Why failed members were blocked.
    """

    cluster_id: int
    member_sites: List[MemberSite]
    n_runs_total: int
    persistence: float
    pass_fraction: float
    centroid_mean: Tuple[float, float, float]
    centroid_variance: float
    mean_quality_score: float
    mean_contact_reorg: float
    mean_response_sharpness: float
    mean_localization: float
    anchor_consistency: float
    growth_vector_consistency: float
    lining_consistency: float
    gate_failure_reasons: Dict[str, int]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "cluster_id": self.cluster_id,
            "member_sites": [m.to_dict() for m in self.member_sites],
            "n_runs_total": self.n_runs_total,
            "persistence": self.persistence,
            "pass_fraction": self.pass_fraction,
            "centroid_mean": list(self.centroid_mean),
            "centroid_variance": self.centroid_variance,
            "mean_quality_score": self.mean_quality_score,
            "mean_contact_reorg": self.mean_contact_reorg,
            "mean_response_sharpness": self.mean_response_sharpness,
            "mean_localization": self.mean_localization,
            "anchor_consistency": self.anchor_consistency,
            "growth_vector_consistency": self.growth_vector_consistency,
            "lining_consistency": self.lining_consistency,
            "gate_failure_reasons": dict(self.gate_failure_reasons),
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> ConsensusSite:
        data = copy.deepcopy(d)
        data["member_sites"] = [MemberSite.from_dict(m) for m in data["member_sites"]]
        data["centroid_mean"] = tuple(data["centroid_mean"])
        return cls(**data)

    @classmethod
    def from_json(cls, s: str) -> ConsensusSite:
        return cls.from_dict(json.loads(s))

    def to_pickle(self) -> bytes:
        return pickle.dumps(self)

    @classmethod
    def from_pickle(cls, data: bytes) -> ConsensusSite:
        obj = pickle.loads(data)  # noqa: S301
        if not isinstance(obj, cls):
            raise TypeError(f"Expected {cls.__name__}, got {type(obj).__name__}")
        return obj


@dataclass
class ConsensusResult:
    """Complete consensus output for a target.

    Attributes:
        target_name:      Target identifier.
        n_replicates:     Number of replicate runs.
        consensus_sites:  Consensus sites ranked by persistence.
        n_consensus:      Number of consensus clusters.
    """

    target_name: str
    n_replicates: int
    consensus_sites: List[ConsensusSite]
    n_consensus: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "target_name": self.target_name,
            "n_replicates": self.n_replicates,
            "consensus_sites": [cs.to_dict() for cs in self.consensus_sites],
            "n_consensus": self.n_consensus,
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> ConsensusResult:
        data = copy.deepcopy(d)
        data["consensus_sites"] = [
            ConsensusSite.from_dict(cs) for cs in data["consensus_sites"]
        ]
        return cls(**data)

    @classmethod
    def from_json(cls, s: str) -> ConsensusResult:
        return cls.from_dict(json.loads(s))

    def to_pickle(self) -> bytes:
        return pickle.dumps(self)

    @classmethod
    def from_pickle(cls, data: bytes) -> ConsensusResult:
        obj = pickle.loads(data)  # noqa: S301
        if not isinstance(obj, cls):
            raise TypeError(f"Expected {cls.__name__}, got {type(obj).__name__}")
        return obj
