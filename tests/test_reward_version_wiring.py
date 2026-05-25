from __future__ import annotations

import torch

from prism_dstw.scoring.tripartite_bias_scorer import compute_tripartite_bias
from scripts.train_gflownet_policy import (
    DEFAULT_POPULATION_CONSENSUS_SURVIVORS,
    DEFAULT_SURVIVORS,
    action_atom_features_from_table,
    compute_effective_reward_tensor,
    pose_penalty_from_row,
    pose_penalty_source_from_row,
    resolve_survivor_corpus_for_reward,
)


def test_v3_consensus_uses_consensus_bonus() -> None:
    """Changing consensus_bonus_weight must change the effective reward."""

    mock_row = {
        "pi_complement": 10.0,
        "pi_clash_pocket": 2.0,
        "lock_geometry_score": 0.5,
        "sigma_shear": 0.1,
        "consensus_complement_bonus": 5.0,
    }
    raw_rewards = torch.tensor([1.0], dtype=torch.float32)
    tripartite_scores = [compute_tripartite_bias(mock_row)]

    reward_w0, metrics_w0 = compute_effective_reward_tensor(
        reward_version="v3_population_consensus",
        oracle_rows=[mock_row],
        raw_rewards=raw_rewards,
        tripartite_scores=tripartite_scores,
        consensus_bonus_weight=0.0,
    )
    reward_w2, metrics_w2 = compute_effective_reward_tensor(
        reward_version="v3_population_consensus",
        oracle_rows=[mock_row],
        raw_rewards=raw_rewards,
        tripartite_scores=tripartite_scores,
        consensus_bonus_weight=2.0,
    )

    difference = float((reward_w2 - reward_w0).item())
    assert float(reward_w2.item()) > float(reward_w0.item())
    assert abs(difference - 10.0) < 0.01
    assert metrics_w0["consensus_bonus_mean"] == 5.0
    assert metrics_w2["consensus_bonus_mean"] == 5.0


def test_v1_base_uses_raw_oracle_reward() -> None:
    """The v1 reward path preserves the raw Rust oracle reward."""

    mock_row = {"pi_complement": 100.0, "consensus_complement_bonus": 100.0}
    raw_rewards = torch.tensor([7.25], dtype=torch.float32)
    reward, _ = compute_effective_reward_tensor(
        reward_version="v1_base",
        oracle_rows=[mock_row],
        raw_rewards=raw_rewards,
        tripartite_scores=[compute_tripartite_bias(mock_row)],
        consensus_bonus_weight=10.0,
    )

    assert torch.allclose(reward, raw_rewards)


def test_v4_full_field_penalizes_shear_hysteresis_and_pose() -> None:
    base_row = {
        "pi_complement": 10.0,
        "pi_clash_pocket": 1.0,
        "lock_geometry_score": 0.5,
        "bias_projection_score": 0.5,
        "consensus_complement_bonus": 2.0,
        "pathway_voxels_occupied": 1.0,
        "pathway_score_mean": 0.5,
        "reversibility_mean": 1.0,
        "sigma_shear_mean": 0.0,
        "u_pose": 0.0,
    }
    stressed = {
        **base_row,
        "sigma_shear_mean": 10.0,
        "reversibility_mean": 0.4,
        "u_pose": 1.5,
        "pathway_voxels_occupied": 0.0,
        "pathway_score_mean": 0.0,
    }
    raw_rewards = torch.tensor([1.0, 1.0], dtype=torch.float32)
    rewards, metrics = compute_effective_reward_tensor(
        reward_version="v4_full_field",
        oracle_rows=[base_row, stressed],
        raw_rewards=raw_rewards,
        tripartite_scores=[compute_tripartite_bias(base_row), compute_tripartite_bias(stressed)],
        consensus_bonus_weight=2.0,
    )
    assert float(rewards[0].item()) > float(rewards[1].item())
    assert metrics["shear_mean"] == 5.0
    assert metrics["pathway_voxels_occupied"] == 0.5
    assert metrics["u_pose_mean"] == 0.75


def test_consensus_reward_auto_selects_population_survivor_corpus() -> None:
    """v3/v4 runs must not silently use the WT survivor corpus when consensus data exists."""

    selected = resolve_survivor_corpus_for_reward(
        requested_survivors=DEFAULT_SURVIVORS,
        reward_version="v4_full_field",
        signal_grid=DEFAULT_POPULATION_CONSENSUS_SURVIVORS.with_name("signal_grid_population_consensus.parquet"),
    )

    assert selected == DEFAULT_POPULATION_CONSENSUS_SURVIVORS


def test_u_pose_uses_surviving_rotamers_when_present() -> None:
    assert pose_penalty_source_from_row({"n_surviving_rotamers": 6}) == "n_surviving_rotamers"
    assert abs(pose_penalty_from_row({"n_surviving_rotamers": 6}) - 0.0) < 1.0e-8
    assert pose_penalty_from_row({"n_surviving_rotamers": 1}) > 1.5


def test_u_pose_proxy_source_is_explicit_when_survivor_count_absent() -> None:
    row = {"best_rotamer_rank": 6}
    assert pose_penalty_source_from_row(row) == "best_rotamer_rank_proxy"
    assert pose_penalty_from_row(row) > 1.5


def test_action_atom_features_preserve_per_atom_charge_lane() -> None:
    import json
    import polars as pl

    table = pl.DataFrame(
        {
            "conformer_atoms_json": [
                json.dumps(
                    [
                        {"atomic_num": 6, "partial_charge": -0.25},
                        {"atomic_num": 8, "partial_charge": 0.42},
                    ]
                )
            ]
        }
    )
    features, mask = action_atom_features_from_table(table, base_feature_dim=13)

    assert tuple(features.shape) == (1, 2, 13)
    assert tuple(mask.shape) == (1, 2)
    assert bool(mask.all().item())
    assert torch.allclose(features[0, :, -1], torch.tensor([-0.25, 0.42]))
