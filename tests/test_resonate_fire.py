from __future__ import annotations

import torch

from prism_dstw.hierarchical_bayes.gflownet_policy import FiberBundleGFlowNetPolicy


def test_hard_zero_kills_aliphatic_message() -> None:
    mask = torch.tensor([True, False, True, False])
    messages = torch.randn(4, 16)
    masked = messages * mask.unsqueeze(-1).float()
    assert float(masked[1].abs().sum().item()) == 0.0
    assert float(masked[3].abs().sum().item()) == 0.0
    assert float(masked[0].abs().sum().item()) > 0.0


def test_single_route_rf_mode_hard_zero_is_supported() -> None:
    policy = FiberBundleGFlowNetPolicy(
        base_feature_dim=4,
        phase_feature_dim=8,
        edge_feature_dim=1,
        anchor_embeddings=torch.randn(3, 8),
        hidden_dim=16,
        embedding_dim=8,
        rf_mode="hard_zero",
    )
    assert policy.orthogonal_mp.rf_mode == "hard_zero"
