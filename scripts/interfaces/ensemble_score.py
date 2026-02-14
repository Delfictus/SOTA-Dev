"""Ensemble scoring interfaces for WT-7.

Dataclasses
-----------
EnsembleMMGBSA
    Ensemble-averaged MM-GBSA (or MM-PBSA) free-energy estimate.
InteractionEntropy
    Interaction-entropy estimate of -TdS and dG (Duan et al., JACS 2016).
"""
from __future__ import annotations

import copy
import json
import pickle
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List


# ---------------------------------------------------------------------------
# EnsembleMMGBSA
# ---------------------------------------------------------------------------

@dataclass
class EnsembleMMGBSA:
    """Ensemble-averaged MM-GBSA free energy for a compound."""

    compound_id: str
    delta_g_mean: float                     # kcal/mol (ensemble-averaged)
    delta_g_std: float
    delta_g_sem: float                      # Standard error of mean
    n_snapshots: int
    snapshot_interval_ps: float             # Time between snapshots (ps)
    decomposition: Dict[str, float]         # {"vdw": …, "elec": …, "gb": …, "sa": …}
    per_residue_contributions: Dict[int, float]  # topology_resid → kcal/mol
    method: str                             # "MMGBSA_ensemble" | "MMPBSA_ensemble"

    # ── Serialization ────────────────────────────────────────────────

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        # asdict converts int keys in per_residue_contributions to int,
        # but JSON keys must be strings; convert explicitly.
        d["per_residue_contributions"] = {
            str(k): v for k, v in d["per_residue_contributions"].items()
        }
        return d

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> EnsembleMMGBSA:
        data = copy.deepcopy(d)
        # Restore int keys from JSON string keys
        data["per_residue_contributions"] = {
            int(k): v for k, v in data["per_residue_contributions"].items()
        }
        return cls(**data)

    @classmethod
    def from_json(cls, s: str) -> EnsembleMMGBSA:
        return cls.from_dict(json.loads(s))

    def to_pickle(self) -> bytes:
        return pickle.dumps(self)

    @classmethod
    def from_pickle(cls, data: bytes) -> EnsembleMMGBSA:
        obj = pickle.loads(data)  # noqa: S301
        if not isinstance(obj, cls):
            raise TypeError(f"Expected {cls.__name__}, got {type(obj).__name__}")
        return obj


# ---------------------------------------------------------------------------
# InteractionEntropy
# ---------------------------------------------------------------------------

@dataclass
class InteractionEntropy:
    """Interaction-entropy free-energy decomposition (replaces NMA)."""

    compound_id: str
    minus_t_delta_s: float                  # -TdS (kcal/mol)
    delta_h: float                          # dH (kcal/mol)
    delta_g_ie: float                       # dH + (-TdS)
    n_frames: int
    convergence_block_std: float            # Block-average convergence (kcal/mol)

    # ── Serialization ────────────────────────────────────────────────

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> InteractionEntropy:
        data = copy.deepcopy(d)
        return cls(**data)

    @classmethod
    def from_json(cls, s: str) -> InteractionEntropy:
        return cls.from_dict(json.loads(s))

    def to_pickle(self) -> bytes:
        return pickle.dumps(self)

    @classmethod
    def from_pickle(cls, data: bytes) -> InteractionEntropy:
        obj = pickle.loads(data)  # noqa: S301
        if not isinstance(obj, cls):
            raise TypeError(f"Expected {cls.__name__}, got {type(obj).__name__}")
        return obj
