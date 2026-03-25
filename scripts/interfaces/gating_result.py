"""Gating stack result interfaces.

Dataclasses
-----------
SiteGateDecision
    Per-site gate pass/fail decisions across the full GTCKL+RS stack.
GatingResult
    Complete gating outcome for all sites of a target.
"""
from __future__ import annotations

import copy
import json
import pickle
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

from .contact_reorg_result import ContactReorgResult
from .response_profile import ResponseProfile


@dataclass
class SiteGateDecision:
    """Gate decision for a single site across the full GTCKL+RS stack.

    Gates are evaluated in strict order.  The first gate that blocks
    determines ``blocked_by``; subsequent gates are not evaluated.

    Attributes:
        site_id:                    Zero-based site index.
        therm_pass:                 True if thermodynamic gate passed.
        coherence_pass:             True if coherence gate passed (soft gate).
        localization_pass:          True if localization gate passed.
        contact_reorg_pass:         True if contact reorganization gate passed.
        response_selectivity_pass:  True if response selectivity gate passed.
        overall_pass:               True if ALL hard gates passed.
        blocked_by:                 Name of first blocking gate, or None.
        contact_reorg:              Full ContactReorgResult, if computed.
        response_profile:           Full ResponseProfile, if computed.
    """

    site_id: int
    therm_pass: bool
    coherence_pass: bool
    localization_pass: bool
    contact_reorg_pass: bool
    response_selectivity_pass: bool
    overall_pass: bool
    blocked_by: Optional[str]
    contact_reorg: Optional[ContactReorgResult] = None
    response_profile: Optional[ResponseProfile] = None

    # -- Serialization --------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {
            "site_id": self.site_id,
            "therm_pass": self.therm_pass,
            "coherence_pass": self.coherence_pass,
            "localization_pass": self.localization_pass,
            "contact_reorg_pass": self.contact_reorg_pass,
            "response_selectivity_pass": self.response_selectivity_pass,
            "overall_pass": self.overall_pass,
            "blocked_by": self.blocked_by,
            "contact_reorg": (
                self.contact_reorg.to_dict() if self.contact_reorg else None
            ),
            "response_profile": (
                self.response_profile.to_dict() if self.response_profile else None
            ),
        }
        return d

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> SiteGateDecision:
        data = copy.deepcopy(d)
        if data.get("contact_reorg") is not None:
            data["contact_reorg"] = ContactReorgResult.from_dict(
                data["contact_reorg"]
            )
        if data.get("response_profile") is not None:
            data["response_profile"] = ResponseProfile.from_dict(
                data["response_profile"]
            )
        return cls(**data)

    @classmethod
    def from_json(cls, s: str) -> SiteGateDecision:
        return cls.from_dict(json.loads(s))

    def to_pickle(self) -> bytes:
        return pickle.dumps(self)

    @classmethod
    def from_pickle(cls, data: bytes) -> SiteGateDecision:
        obj = pickle.loads(data)  # noqa: S301
        if not isinstance(obj, cls):
            raise TypeError(f"Expected {cls.__name__}, got {type(obj).__name__}")
        return obj


@dataclass
class GatingResult:
    """Complete gating outcome for all sites of a target.

    Attributes:
        target_name:     Target identifier (e.g. "1btl").
        n_sites_input:   Total sites before gating.
        n_sites_passed:  Sites that passed all hard gates.
        decisions:       Per-site gate decisions (all sites, not just passed).
        passed_site_ids: IDs of passed sites in lexicographic rank order.
    """

    target_name: str
    n_sites_input: int
    n_sites_passed: int
    decisions: List[SiteGateDecision]
    passed_site_ids: List[int]

    # -- Serialization --------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        return {
            "target_name": self.target_name,
            "n_sites_input": self.n_sites_input,
            "n_sites_passed": self.n_sites_passed,
            "decisions": [d.to_dict() for d in self.decisions],
            "passed_site_ids": list(self.passed_site_ids),
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> GatingResult:
        data = copy.deepcopy(d)
        data["decisions"] = [
            SiteGateDecision.from_dict(dec) for dec in data["decisions"]
        ]
        return cls(**data)

    @classmethod
    def from_json(cls, s: str) -> GatingResult:
        return cls.from_dict(json.loads(s))

    def to_pickle(self) -> bytes:
        return pickle.dumps(self)

    @classmethod
    def from_pickle(cls, data: bytes) -> GatingResult:
        obj = pickle.loads(data)  # noqa: S301
        if not isinstance(obj, cls):
            raise TypeError(f"Expected {cls.__name__}, got {type(obj).__name__}")
        return obj
