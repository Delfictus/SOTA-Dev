"""Battery interphase reward surface for PRISM materials-track smoke tests."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from math import exp, isfinite, log1p
from typing import Any

import numpy as np
from numpy.typing import NDArray


CCNS_TO_BATTERY_PHASE: dict[str, str] = {
    "cold_hold": "rest_discharged",
    "ramp_up": "fast_charge",
    "warm_hold": "fully_charged",
    "ramp_down": "discharge",
    "cold_return": "rest_post_cycle",
}


def _float_array(value: object) -> NDArray[np.float64]:
    if value is None:
        return np.asarray([], dtype=np.float64)
    return np.asarray(value, dtype=np.float64).reshape(-1)


@dataclass(frozen=True)
class BatteryInterphaseReward:
    """SEI additive score from electronic, mechanical, ion, and pose terms.

    The adapter is intentionally small and deterministic: it is the runtime
    contract that materials-track consumers can instantiate without an xTB
    binary, while production scoring can substitute measured inputs.
    """

    w_elec: float = 1.0
    w_mech: float = 1.0
    w_ion: float = 1.0
    w_pose: float = 0.5
    target_homo_lumo_gap_ev: float = 4.5

    def compute(self, mol_data: Mapping[str, Any]) -> float:
        gap = float(mol_data.get("homo_lumo_gap", self.target_homo_lumo_gap_ev))
        shear = _float_array(mol_data.get("shear_stress"))
        hysteresis = _float_array(mol_data.get("hysteresis_tensor"))
        activated_voxels = float(mol_data.get("thermally_activated_voxels", 0.0))

        electronic_reward = exp(-abs(gap - self.target_homo_lumo_gap_ev))
        mechanical_penalty = log1p(float(np.mean(np.abs(shear))) if shear.size else 0.0)
        ion_reward = log1p(max(activated_voxels, 0.0))
        pose_uncertainty = float(np.std(hysteresis)) if hysteresis.size else 0.0

        score = (
            self.w_elec * electronic_reward
            - self.w_mech * mechanical_penalty
            + self.w_ion * ion_reward
            - self.w_pose * pose_uncertainty
        )
        if not isfinite(score):
            raise ValueError("battery interphase reward produced non-finite score")
        return float(score)


def coulombic_inefficiency_proxy(phase_spikes: Mapping[str, float]) -> float:
    """Return cycle-return inefficiency using CCNS battery phase semantics."""

    cold_return = float(phase_spikes.get("cold_return", 0.0))
    cold_hold = float(phase_spikes.get("cold_hold", 0.0))
    ramp_up = max(float(phase_spikes.get("ramp_up", 0.0)), 1.0)
    value = abs(cold_return - cold_hold) / ramp_up
    if not isfinite(value):
        raise ValueError("non-finite coulombic inefficiency proxy")
    return float(value)


def phase_vector_from_sequence(values: Sequence[float]) -> dict[str, float]:
    """Map a five-element CCNS vector to named battery-cycle phases."""

    if len(values) != len(CCNS_TO_BATTERY_PHASE):
        raise ValueError("expected exactly five CCNS phase values")
    return dict(zip(CCNS_TO_BATTERY_PHASE.keys(), (float(v) for v in values), strict=True))
