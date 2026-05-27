from __future__ import annotations

import math
import subprocess
from pathlib import Path

import numpy as np

from prism_dstw.adapters.materials.battery_interphase_reward import BatteryInterphaseReward, CCNS_TO_BATTERY_PHASE
from prism_dstw.adapters.materials.universal_materials_action_space import UniversalMaterialsActionSpace
from prism_dstw.adapters.materials.xtb_reward_adapter import XTBRewardAdapter


ROOT = Path(__file__).resolve().parents[2]


def test_battery_reward_function_computes() -> None:
    reward_fn = BatteryInterphaseReward(w_elec=1.0, w_mech=1.0, w_ion=1.0, w_pose=0.5)
    score = reward_fn.compute(
        {
            "smiles": "C1=CC=C(C=C1)F",
            "coords_3d": np.random.default_rng(7).normal(size=(12, 3)),
            "homo_lumo_gap": 4.5,
            "shear_stress": np.linspace(0.05, 1.2, 12),
            "thermally_activated_voxels": 5,
            "hysteresis_tensor": np.linspace(0.1, 0.8, 5),
        }
    )
    assert isinstance(score, float)
    assert math.isfinite(score)
    assert score != 0.0


def test_battery_ccns_mapping() -> None:
    assert CCNS_TO_BATTERY_PHASE == {
        "cold_hold": "rest_discharged",
        "ramp_up": "fast_charge",
        "warm_hold": "fully_charged",
        "ramp_down": "discharge",
        "cold_return": "rest_post_cycle",
    }


def test_universal_materials_action_space() -> None:
    space = UniversalMaterialsActionSpace()
    assert len(space.action_types) >= 9
    for action_type in space.action_types:
        action = space.instantiate(action_type)
        updated = action.apply({"candidate_id": "mat-smoke"})
        assert updated["action_history"][-1] == action_type


def test_xtb_adapter_interface_and_parsers() -> None:
    adapter = XTBRewardAdapter.__new__(XTBRewardAdapter)
    assert hasattr(adapter, "compute_homo_lumo_gap")
    assert hasattr(adapter, "compute_electron_affinity")
    assert XTBRewardAdapter.parse_homo_lumo_gap("HOMO-LUMO GAP       4.250") == 4.25
    assert XTBRewardAdapter.parse_electron_affinity("electron affinity 1.75") == 1.75


def test_prism_nhs_materials_crate_compiles() -> None:
    result = subprocess.run(
        ["cargo", "check", "-p", "prism-nhs", "--bin", "warp_jacobian"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=180,
        check=False,
    )
    assert result.returncode == 0, result.stderr[-2000:]
