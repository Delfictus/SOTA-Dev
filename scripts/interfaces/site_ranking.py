"""Site ranking interfaces — lexicographic ordering of gated sites.

Dataclasses
-----------
RankedSite
    A single site with its ranking keys exposed.
SiteRanking
    Ordered list of sites that passed the full gating stack.
"""
from __future__ import annotations

import copy
import json
import pickle
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional


@dataclass
class RankedSite:
    """A site that passed gating, with its lexicographic ranking keys.

    Keys are exposed individually — never blended.

    Attributes:
        site_id:                 Zero-based site index.
        rank:                    1-based rank (1 = top).
        engine_chem:             Engine chemistry score (from Rust).
        engine_vcs:              Voxel contact score (from Rust).
        contact_reorg_strength:  localization_ratio from ContactReorgResult.
        anchor_density:          anchors per lining residue.
        water_displacement:      Total positive dG from water map (kcal/mol).
                                 Used as tie-breaker only.
    """

    site_id: int
    rank: int
    engine_chem: float
    engine_vcs: float
    contact_reorg_strength: float
    anchor_density: float
    water_displacement: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> RankedSite:
        return cls(**copy.deepcopy(d))

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)

    @classmethod
    def from_json(cls, s: str) -> RankedSite:
        return cls.from_dict(json.loads(s))

    def to_pickle(self) -> bytes:
        return pickle.dumps(self)

    @classmethod
    def from_pickle(cls, data: bytes) -> RankedSite:
        obj = pickle.loads(data)  # noqa: S301
        if not isinstance(obj, cls):
            raise TypeError(f"Expected {cls.__name__}, got {type(obj).__name__}")
        return obj


@dataclass
class SiteRanking:
    """Lexicographic ranking of all sites that passed the gating stack.

    Attributes:
        target_name: Target identifier.
        ranked_sites: List of RankedSite in rank order (1 = best).
        n_ranked: Number of ranked sites.
    """

    target_name: str
    ranked_sites: List[RankedSite]
    n_ranked: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "target_name": self.target_name,
            "ranked_sites": [rs.to_dict() for rs in self.ranked_sites],
            "n_ranked": self.n_ranked,
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> SiteRanking:
        data = copy.deepcopy(d)
        data["ranked_sites"] = [
            RankedSite.from_dict(rs) for rs in data["ranked_sites"]
        ]
        return cls(**data)

    @classmethod
    def from_json(cls, s: str) -> SiteRanking:
        return cls.from_dict(json.loads(s))

    def to_pickle(self) -> bytes:
        return pickle.dumps(self)

    @classmethod
    def from_pickle(cls, data: bytes) -> SiteRanking:
        obj = pickle.loads(data)  # noqa: S301
        if not isinstance(obj, cls):
            raise TypeError(f"Expected {cls.__name__}, got {type(obj).__name__}")
        return obj
