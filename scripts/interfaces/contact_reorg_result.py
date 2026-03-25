"""Contact reorganization result interface.

Dataclasses
-----------
ContactReorgResult
    Per-site contact reorganization metrics from trajectory analysis.
    Used by the Contact Reorg gate in the GTCKL+RS gating stack.
"""
from __future__ import annotations

import copy
import json
import pickle
from dataclasses import asdict, dataclass
from typing import Any, Dict


@dataclass
class ContactReorgResult:
    """Contact reorganization metrics for a single binding site.

    Computed by comparing CA-CA contact maps across trajectory frames
    within a local radius of the site centroid.  Measures whether
    localized structural rearrangement occurs near the pocket.

    Attributes:
        site_id:                 Zero-based site index from PRISM detection.
        contact_change_density:  Mean local contacts formed + broken per frame.
        localization_ratio:      Fraction of total contact change that is local
                                 to this site (0-1).
        persistence:             Fraction of early-formed local contacts that
                                 persist to late frames (0-1).
        boundary_growth:         Relative change in local contact count from
                                 first to last frame (positive = pocket wall forming).
        n_frames_analyzed:       Number of trajectory frames used.
        gate_pass:               True if site passes the contact reorg gate.
        gate_reason:             Human-readable reason for pass/fail.
    """

    site_id: int
    contact_change_density: float
    localization_ratio: float
    persistence: float
    boundary_growth: float
    n_frames_analyzed: int
    gate_pass: bool
    gate_reason: str

    # -- Serialization --------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> ContactReorgResult:
        return cls(**copy.deepcopy(d))

    @classmethod
    def from_json(cls, s: str) -> ContactReorgResult:
        return cls.from_dict(json.loads(s))

    def to_pickle(self) -> bytes:
        return pickle.dumps(self)

    @classmethod
    def from_pickle(cls, data: bytes) -> ContactReorgResult:
        obj = pickle.loads(data)  # noqa: S301
        if not isinstance(obj, cls):
            raise TypeError(f"Expected {cls.__name__}, got {type(obj).__name__}")
        return obj
