"""Explicit-solvent refinement result interface for WT-6.

Dataclasses
-----------
ExplicitSolventResult
    Pocket stability metrics after explicit-water MD simulation.
    Optionally references a :class:`~scripts.interfaces.water_map.WaterMap`.
"""
from __future__ import annotations

import copy
import json
import pickle
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

from .water_map import HydrationSite, WaterMap


# ---------------------------------------------------------------------------
# ExplicitSolventResult
# ---------------------------------------------------------------------------

@dataclass
class ExplicitSolventResult:
    """Result of an explicit-solvent MD pocket-stability simulation."""

    pocket_id: int
    simulation_time_ns: float
    water_model: str                        # "TIP3P" | "OPC" | "TIP4P-Ew"
    force_field: str                        # "ff19SB" | "ff14SB" | "CHARMM36m"
    pocket_stable: bool                     # QC gate: stable → proceed
    pocket_rmsd_mean: float                 # Angstrom
    pocket_rmsd_std: float
    pocket_volume_mean: float               # Angstrom^3
    pocket_volume_std: float
    n_structural_waters: int                # Conserved waters (>80% occupancy)
    trajectory_path: str
    snapshot_frames: List[int]              # Frame indices for ensemble scoring
    water_map: Optional[WaterMap] = None

    # ── Serialization ────────────────────────────────────────────────

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        # asdict converts the Optional WaterMap recursively; keep as-is
        return d

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> ExplicitSolventResult:
        data = copy.deepcopy(d)
        if data.get("water_map") is not None:
            wm = data["water_map"]
            wm["hydration_sites"] = [
                HydrationSite(**s) for s in wm["hydration_sites"]
            ]
            data["water_map"] = WaterMap(**wm)
        return cls(**data)

    @classmethod
    def from_json(cls, s: str) -> ExplicitSolventResult:
        return cls.from_dict(json.loads(s))

    def to_pickle(self) -> bytes:
        return pickle.dumps(self)

    @classmethod
    def from_pickle(cls, data: bytes) -> ExplicitSolventResult:
        obj = pickle.loads(data)  # noqa: S301
        if not isinstance(obj, cls):
            raise TypeError(f"Expected {cls.__name__}, got {type(obj).__name__}")
        return obj
