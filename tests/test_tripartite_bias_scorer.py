from __future__ import annotations

from prism_dstw.orchestration.gflownet_action_space import compute_lock_directional_bias
from prism_dstw.scoring.tripartite_bias_scorer import compute_reward_v2, compute_tripartite_bias


def test_tripartite_bias_confidence_promotes_geometry_and_persistence() -> None:
    row = {
        "pi_complement": 10.0,
        "pi_clash_pocket": 1.0,
        "lock_geometry_score": 6.0,
        "lock_geometry_atom_count": 3,
        "lock_voxel_indices_json": "[1,2,3]",
        "lock_occupancy_cold_hold": 5.0,
        "lock_occupancy_ramp_up": 5.0,
        "lock_occupancy_warm_hold": 5.0,
        "lock_occupancy_ramp_down": 5.0,
        "lock_occupancy_cold_return": 4.5,
        "intracellular_penetration_depth_angstrom": 4.0,
        "lock_steric_volume_angstrom3": 120.0,
    }
    score = compute_tripartite_bias(row)
    assert score.epistemic_confidence == "L3"
    assert score.lock_geometry_voxels == [1, 2, 3]
    assert score.lock_persistence_score > 0.8
    assert compute_reward_v2(row, score) > 0.0


def test_tripartite_bias_keeps_projection_low_confidence_without_geometry() -> None:
    score = compute_tripartite_bias({"pi_complement": 1.0, "pi_clash_pocket": 0.1})
    assert score.epistemic_confidence == "L1"
    assert score.lock_geometry_score == 0.0
    assert score.lock_persistence_score == 0.0


def test_lock_directional_bias_cosine() -> None:
    assert compute_lock_directional_bias((1.0, 0.0, 0.0), (2.0, 0.0, 0.0), (0.0, 0.0, 0.0)) == 1.0
    assert compute_lock_directional_bias((-1.0, 0.0, 0.0), (2.0, 0.0, 0.0), (0.0, 0.0, 0.0)) == -1.0
