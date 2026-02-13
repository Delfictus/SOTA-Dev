"""Pocket-dynamics interface for WT-7 cryptic-pocket analysis.

Dataclasses
-----------
PocketDynamics
    P_open, open/closed lifetimes, druggability classification, and
    optional Markov State Model weights.
"""
from __future__ import annotations

import copy
import json
import pickle
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------------
# PocketDynamics
# ---------------------------------------------------------------------------

@dataclass
class PocketDynamics:
    """Cryptic-pocket dynamics summary from multi-stream trajectories."""

    pocket_id: int
    p_open: float                           # 0–1
    p_open_error: float                     # Bootstrap error
    mean_open_lifetime_ns: float
    mean_closed_lifetime_ns: float
    n_opening_events: int
    druggability_classification: str        # STABLE_OPEN | TRANSIENT | RARE_EVENT
    volume_autocorrelation_ns: float
    msm_state_weights: Optional[Dict[int, float]] = None

    # ── Serialization ────────────────────────────────────────────────

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        # Convert int keys to str for JSON compatibility
        if d["msm_state_weights"] is not None:
            d["msm_state_weights"] = {
                str(k): v for k, v in d["msm_state_weights"].items()
            }
        return d

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> PocketDynamics:
        data = copy.deepcopy(d)
        if data.get("msm_state_weights") is not None:
            data["msm_state_weights"] = {
                int(k): v for k, v in data["msm_state_weights"].items()
            }
        return cls(**data)

    @classmethod
    def from_json(cls, s: str) -> PocketDynamics:
        return cls.from_dict(json.loads(s))

    def to_pickle(self) -> bytes:
        return pickle.dumps(self)

    @classmethod
    def from_pickle(cls, data: bytes) -> PocketDynamics:
        obj = pickle.loads(data)  # noqa: S301
        if not isinstance(obj, cls):
            raise TypeError(f"Expected {cls.__name__}, got {type(obj).__name__}")
        return obj
