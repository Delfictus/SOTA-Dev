"""Hydration-site and water-map interfaces for WT-6 explicit-solvent analysis.

Dataclasses
-----------
HydrationSite
    A single discrete hydration site with thermodynamic decomposition.
WaterMap
    Collection of hydration sites for one pocket, with displacement summary.
"""
from __future__ import annotations

import copy
import json
import pickle
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List


# ---------------------------------------------------------------------------
# HydrationSite
# ---------------------------------------------------------------------------

@dataclass
class HydrationSite:
    """A single hydration site identified by IST / SSTMap analysis."""

    x: float                                # Angstrom
    y: float
    z: float
    occupancy: float                        # Fraction of simulation occupied (0–1)
    delta_g_transfer: float                 # kcal/mol (neg=happy, pos=unhappy)
    entropy_contribution: float             # -TdS (kcal/mol)
    enthalpy_contribution: float            # dH (kcal/mol)
    n_hbonds_mean: float                    # Mean H-bonds to protein
    classification: str                     # CONSERVED_HAPPY | CONSERVED_UNHAPPY | BULK
    displaceable: bool                      # True if dG_transfer > +1.0 kcal/mol

    # ── Serialization ────────────────────────────────────────────────

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> HydrationSite:
        data = copy.deepcopy(d)
        return cls(**data)

    @classmethod
    def from_json(cls, s: str) -> HydrationSite:
        return cls.from_dict(json.loads(s))

    def to_pickle(self) -> bytes:
        return pickle.dumps(self)

    @classmethod
    def from_pickle(cls, data: bytes) -> HydrationSite:
        obj = pickle.loads(data)  # noqa: S301
        if not isinstance(obj, cls):
            raise TypeError(f"Expected {cls.__name__}, got {type(obj).__name__}")
        return obj


# ---------------------------------------------------------------------------
# WaterMap
# ---------------------------------------------------------------------------

@dataclass
class WaterMap:
    """Hydration-site map for a single pocket."""

    pocket_id: int
    hydration_sites: List[HydrationSite]
    n_displaceable: int                     # Count of displaceable sites
    max_displacement_energy: float          # Highest single-site dG (kcal/mol)
    total_displacement_energy: float        # Sum of positive dG values
    grid_resolution: float                  # Angstrom (typically 0.5)
    analysis_frames: int                    # Trajectory frames analysed

    # ── Serialization ────────────────────────────────────────────────

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> WaterMap:
        data = copy.deepcopy(d)
        data["hydration_sites"] = [
            HydrationSite(**s) for s in data["hydration_sites"]
        ]
        return cls(**data)

    @classmethod
    def from_json(cls, s: str) -> WaterMap:
        return cls.from_dict(json.loads(s))

    def to_pickle(self) -> bytes:
        return pickle.dumps(self)

    @classmethod
    def from_pickle(cls, data: bytes) -> WaterMap:
        obj = pickle.loads(data)  # noqa: S301
        if not isinstance(obj, cls):
            raise TypeError(f"Expected {cls.__name__}, got {type(obj).__name__}")
        return obj
