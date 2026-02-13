"""Membrane-system interface for WT-5 preprocessing.

Dataclasses
-----------
MembraneSystem
    Lipid-bilayer embedding parameters for membrane-protein targets
    (GPCRs, ion channels, transporters).
"""
from __future__ import annotations

import copy
import json
import pickle
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Tuple


# ---------------------------------------------------------------------------
# MembraneSystem
# ---------------------------------------------------------------------------

@dataclass
class MembraneSystem:
    """Lipid bilayer system built around a membrane protein."""

    lipid_composition: Dict[str, float]     # {"POPC": 0.7, "CHOL": 0.3}
    bilayer_method: str                     # "packmol_memgen" | "charmm_gui" | …
    n_lipids: int
    membrane_thickness: float               # Angstrom
    protein_orientation: str                # "OPM" | "manual" | "PPM"
    opm_tilt_angle: float                   # Degrees
    system_size: Tuple[float, float, float] # Box dimensions (x, y, z) in Angstrom
    total_atoms: int
    equilibration_protocol: str             # e.g. "CHARMM_GUI_6step"

    # ── Serialization ────────────────────────────────────────────────

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["system_size"] = list(d["system_size"])
        return d

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> MembraneSystem:
        data = copy.deepcopy(d)
        data["system_size"] = tuple(data["system_size"])
        return cls(**data)

    @classmethod
    def from_json(cls, s: str) -> MembraneSystem:
        return cls.from_dict(json.loads(s))

    def to_pickle(self) -> bytes:
        return pickle.dumps(self)

    @classmethod
    def from_pickle(cls, data: bytes) -> MembraneSystem:
        obj = pickle.loads(data)  # noqa: S301
        if not isinstance(obj, cls):
            raise TypeError(f"Expected {cls.__name__}, got {type(obj).__name__}")
        return obj
