"""Tautomer / protonation-state interfaces for WT-5 preprocessing.

Dataclasses
-----------
TautomerState
    A single protonation / tautomeric form at a given pH.
TautomerEnsemble
    The complete Boltzmann-weighted set of tautomeric states for one
    parent molecule at a target pH (default 7.4).
"""
from __future__ import annotations

import copy
import json
import pickle
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Tuple


# ---------------------------------------------------------------------------
# TautomerState
# ---------------------------------------------------------------------------

@dataclass
class TautomerState:
    """A single tautomeric / protonation state."""

    smiles: str                             # Canonical SMILES of this tautomer
    parent_smiles: str                      # Original input SMILES
    protonation_ph: float                   # pH at which this state was generated
    charge: int                             # Net formal charge
    pka_shifts: List[Tuple[int, float]]     # (atom_idx, predicted_pKa)
    population_fraction: float              # Boltzmann weight at target pH
    source_tool: str                        # "dimorphite_dl" | "openeye" | "pkasolver"

    # ── Serialization ────────────────────────────────────────────────

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        # Tuple pairs → lists for JSON compatibility
        d["pka_shifts"] = [list(pair) for pair in d["pka_shifts"]]
        return d

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> TautomerState:
        data = copy.deepcopy(d)
        data["pka_shifts"] = [tuple(pair) for pair in data["pka_shifts"]]
        return cls(**data)

    @classmethod
    def from_json(cls, s: str) -> TautomerState:
        return cls.from_dict(json.loads(s))

    def to_pickle(self) -> bytes:
        return pickle.dumps(self)

    @classmethod
    def from_pickle(cls, data: bytes) -> TautomerState:
        obj = pickle.loads(data)  # noqa: S301
        if not isinstance(obj, cls):
            raise TypeError(f"Expected {cls.__name__}, got {type(obj).__name__}")
        return obj


# ---------------------------------------------------------------------------
# TautomerEnsemble
# ---------------------------------------------------------------------------

@dataclass
class TautomerEnsemble:
    """Boltzmann-weighted set of tautomeric states for one molecule."""

    parent_smiles: str
    states: List[TautomerState]
    dominant_state: TautomerState           # Highest population at target pH
    target_ph: float                        # Default 7.4
    enumeration_method: str                 # e.g. "dimorphite_dl_rdk_mstandardize"

    # ── Serialization ────────────────────────────────────────────────

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        # Tuple pairs inside nested states → lists
        for state in d["states"]:
            state["pka_shifts"] = [list(p) for p in state["pka_shifts"]]
        d["dominant_state"]["pka_shifts"] = [
            list(p) for p in d["dominant_state"]["pka_shifts"]
        ]
        return d

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> TautomerEnsemble:
        data = copy.deepcopy(d)
        data["states"] = [TautomerState.from_dict(s) for s in data["states"]]
        data["dominant_state"] = TautomerState.from_dict(data["dominant_state"])
        return cls(**data)

    @classmethod
    def from_json(cls, s: str) -> TautomerEnsemble:
        return cls.from_dict(json.loads(s))

    def to_pickle(self) -> bytes:
        return pickle.dumps(self)

    @classmethod
    def from_pickle(cls, data: bytes) -> TautomerEnsemble:
        obj = pickle.loads(data)  # noqa: S301
        if not isinstance(obj, cls):
            raise TypeError(f"Expected {cls.__name__}, got {type(obj).__name__}")
        return obj
