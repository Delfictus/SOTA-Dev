"""Viewer-payload interface for WT-8 interactive HTML5/WebGL deliverable.

Dataclasses
-----------
ViewerPayload
    Aggregated data structure that WT-4 builds and WT-8 renders as a
    single-file interactive HTML5 viewer.
"""
from __future__ import annotations

import copy
import json
import pickle
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------------
# ViewerPayload
# ---------------------------------------------------------------------------

@dataclass
class ViewerPayload:
    """All data required to render the interactive HTML5 viewer."""

    target_name: str                        # e.g. "KRAS_G12C"
    pdb_structure: str                      # Full PDB text
    pocket_surfaces: List[dict]             # [{vertices, triangles, color}, …]
    spike_positions: List[dict]             # [{position, type, intensity, residue}, …]
    water_map_sites: List[dict]             # [{position, occupancy, delta_g, …}, …]
    ligand_poses: List[dict]                # [{smiles, mol_block, dg_kcal, …}, …]
    lining_residues: List[int]              # PDB residue numbers
    p_open: Optional[float] = None          # From PocketDynamics (0–1)
    metadata: Dict[str, Any] = field(default_factory=dict)

    # ── Serialization ────────────────────────────────────────────────

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> ViewerPayload:
        data = copy.deepcopy(d)
        return cls(**data)

    @classmethod
    def from_json(cls, s: str) -> ViewerPayload:
        return cls.from_dict(json.loads(s))

    def to_pickle(self) -> bytes:
        return pickle.dumps(self)

    @classmethod
    def from_pickle(cls, data: bytes) -> ViewerPayload:
        obj = pickle.loads(data)  # noqa: S301
        if not isinstance(obj, cls):
            raise TypeError(f"Expected {cls.__name__}, got {type(obj).__name__}")
        return obj
