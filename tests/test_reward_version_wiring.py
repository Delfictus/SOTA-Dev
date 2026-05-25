from __future__ import annotations

import torch

from prism_dstw.scoring.tripartite_bias_scorer import compute_tripartite_bias
from scripts.train_gflownet_policy import compute_effective_reward_tensor


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
