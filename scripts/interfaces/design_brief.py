"""DesignBrief interface — complete design output for a binding site.

Dataclasses
-----------
DesignBrief
    Aggregates all design layer outputs for one site that passed the full
    gating stack.  Used to generate reports (JSON, PyMOL, HTML).
"""
from __future__ import annotations

import copy
import json
import pickle
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

from .anchor_point import AnchorPointMap
from .growth_vector import GrowthVectorMap
from .pocket_profile import PocketProfile
from .site_ranking import RankedSite


@dataclass
class DesignBrief:
    """Complete design output for a single gated binding site.

    Pure projection of computed data — no recommendations, no confidence
    statements, no executive summaries.

    Attributes:
        target_name:    Target identifier (e.g. "1btl").
        pdb_id:         PDB accession code.
        site_id:        Zero-based site index.
        ranked_site:    Ranking keys and position.
        anchor_map:     Anchor point map.
        growth_map:     Growth vector and subpocket map.
        pocket_profile: Descriptive pocket chemistry.
        water_sites:    Hydration sites summary (from WaterMap, if available).
    """

    target_name: str
    pdb_id: str
    site_id: int
    ranked_site: RankedSite
    anchor_map: AnchorPointMap
    growth_map: GrowthVectorMap
    pocket_profile: PocketProfile
    water_sites: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "target_name": self.target_name,
            "pdb_id": self.pdb_id,
            "site_id": self.site_id,
            "ranked_site": self.ranked_site.to_dict(),
            "anchor_map": self.anchor_map.to_dict(),
            "growth_map": self.growth_map.to_dict(),
            "pocket_profile": self.pocket_profile.to_dict(),
            "water_sites": list(self.water_sites),
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> DesignBrief:
        data = copy.deepcopy(d)
        data["ranked_site"] = RankedSite.from_dict(data["ranked_site"])
        data["anchor_map"] = AnchorPointMap.from_dict(data["anchor_map"])
        data["growth_map"] = GrowthVectorMap.from_dict(data["growth_map"])
        data["pocket_profile"] = PocketProfile.from_dict(data["pocket_profile"])
        return cls(**data)

    @classmethod
    def from_json(cls, s: str) -> DesignBrief:
        return cls.from_dict(json.loads(s))

    def to_pickle(self) -> bytes:
        return pickle.dumps(self)

    @classmethod
    def from_pickle(cls, data: bytes) -> DesignBrief:
        obj = pickle.loads(data)  # noqa: S301
        if not isinstance(obj, cls):
            raise TypeError(f"Expected {cls.__name__}, got {type(obj).__name__}")
        return obj
