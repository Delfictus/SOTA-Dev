"""DockingResult interface — output of GPU docking pipeline (UniDock / GNINA).

Extends the existing dict-based result format from gpu_dock.py into a
typed dataclass.  Consumed by WT-2 (FEP), WT-3 (post-dock filtering),
WT-7 (ensemble rescoring), and WT-4 (orchestrator).
"""
from __future__ import annotations

import copy
import json
import pickle
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class DockingPose:
    """A single docked pose within a DockingResult.

    Attributes:
        pose_rank:      1-based rank by primary scoring function.
        mol_block:      SDF mol-block of the docked 3-D pose.
        vina_score:     AutoDock Vina score (kcal/mol, more negative = better).
        cnn_score:      GNINA CNN pose-quality score (0–1, higher = better).
        cnn_affinity:   GNINA CNN predicted binding affinity (kcal/mol).
        rmsd_lb:        RMSD lower-bound to best mode (Angstrom).
        rmsd_ub:        RMSD upper-bound to best mode (Angstrom).
    """
    pose_rank: int
    mol_block: str
    vina_score: float
    cnn_score: float = 0.0
    cnn_affinity: float = 0.0
    rmsd_lb: float = 0.0
    rmsd_ub: float = 0.0


@dataclass
class DockingResult:
    """Docking result for one ligand against one binding site.

    Attributes:
        compound_id:        Unique compound identifier (name or SMILES hash).
        smiles:             Canonical SMILES of the docked ligand.
        site_id:            PRISM pocket index that was targeted.
        receptor_pdb:       Path to the receptor PDB used for docking.
        poses:              Ranked list of DockingPose objects.
        best_vina_score:    Best (most negative) Vina score across poses.
        best_cnn_affinity:  Best (most negative) CNN affinity across poses.
        docking_engine:     Engine used ("unidock", "gnina", "unidock+gnina").
        box_center:         (x, y, z) centre of the docking box (Angstrom).
        box_size:           (sx, sy, sz) dimensions of the docking box (Angstrom).
        exhaustiveness:     Search exhaustiveness parameter.
        docking_timestamp:  ISO-8601 UTC timestamp of the docking run.
    """
    compound_id: str
    smiles: str
    site_id: int
    receptor_pdb: str
    poses: List[DockingPose]
    best_vina_score: float
    best_cnn_affinity: float
    docking_engine: str
    box_center: Tuple[float, float, float]
    box_size: Tuple[float, float, float]
    exhaustiveness: int = 32
    docking_timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    # ── Serialization ────────────────────────────────────────────────────

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["box_center"] = list(d["box_center"])
        d["box_size"] = list(d["box_size"])
        return d

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> DockingResult:
        data = copy.deepcopy(d)
        data["poses"] = [DockingPose(**p) for p in data["poses"]]
        data["box_center"] = tuple(data["box_center"])
        data["box_size"] = tuple(data["box_size"])
        return cls(**data)

    @classmethod
    def from_json(cls, s: str) -> DockingResult:
        return cls.from_dict(json.loads(s))

    def to_pickle(self) -> bytes:
        return pickle.dumps(self)

    @classmethod
    def from_pickle(cls, data: bytes) -> DockingResult:
        obj = pickle.loads(data)  # noqa: S301
        if not isinstance(obj, cls):
            raise TypeError(f"Expected {cls.__name__}, got {type(obj).__name__}")
        return obj
