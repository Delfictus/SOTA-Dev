"""Response selectivity profile interface.

Dataclasses
-----------
ResponseProfile
    Per-site response selectivity metrics computed from spike dynamics.
    Used by the Response Selectivity gate in the GTCKL+RS gating stack.
"""
from __future__ import annotations

import copy
import json
import math
import pickle
from dataclasses import asdict, dataclass
from typing import Any, Dict


@dataclass
class ResponseProfile:
    """Response selectivity metrics for a single binding site.

    Measures whether a site *behaves* like a binding region under perturbation,
    rather than merely *looking* like one geometrically.  Computed from spike
    event dynamics (intensity distribution, spatial focus, thermal phase
    response).

    Attributes:
        site_id:              Zero-based site index from PRISM detection.
        sharpness:            spike_intensity_peak / intensity-weighted spatial
                              spread.  High = focused response.
        temporal_asymmetry:   |n_warm - n_cold| / (n_warm + n_cold).  High =
                              directional thermal response.
        energy_density:       total_spike_intensity / pocket_volume (A^-3).
                              High = concentrated energy.
        contact_coupling:     Pearson r between per-frame spike counts and
                              local contact changes.  NaN if trajectory
                              unavailable.
        n_spikes_analyzed:    Number of spikes used in computation.
        gate_pass:            True if site passes the response selectivity gate.
        gate_reason:          Human-readable reason for pass/fail.
    """

    site_id: int
    sharpness: float
    temporal_asymmetry: float
    energy_density: float
    contact_coupling: float
    n_spikes_analyzed: int
    gate_pass: bool
    gate_reason: str

    # -- Serialization --------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        # NaN is not valid JSON — serialize as null
        if math.isnan(d["contact_coupling"]):
            d["contact_coupling"] = None
        return d

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> ResponseProfile:
        data = copy.deepcopy(d)
        if data.get("contact_coupling") is None:
            data["contact_coupling"] = float("nan")
        return cls(**data)

    @classmethod
    def from_json(cls, s: str) -> ResponseProfile:
        return cls.from_dict(json.loads(s))

    def to_pickle(self) -> bytes:
        return pickle.dumps(self)

    @classmethod
    def from_pickle(cls, data: bytes) -> ResponseProfile:
        obj = pickle.loads(data)  # noqa: S301
        if not isinstance(obj, cls):
            raise TypeError(f"Expected {cls.__name__}, got {type(obj).__name__}")
        return obj
