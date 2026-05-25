from __future__ import annotations

from pathlib import Path
from typing import cast

import polars as pl
import torch
from torch import Tensor

from prism_dstw.hierarchical_bayes.gflownet_policy import AnchorAttentionGFlowNetPolicy, PolicyOutput
from prism_dstw.hierarchical_bayes.trajectory_balance import TrajectoryBalanceLoss, TrajectoryBalanceOutput


REPO_ROOT = Path(__file__).resolve().parents[1]
CALIBRATION_ANCHORS = REPO_ROOT / "campaigns/glp1r_aleniglipron/track_a_generative/calibration_anchors_3d.parquet"
ACTION_EMBEDDING_DIM = 8
BATCH_SIZE = 32
TRAJECTORY_LENGTH = 3
NODE_COUNT = 6


def load_calibration_anchor_embeddings(limit: int = 512) -> Tensor:
    frame = (
        pl.scan_parquet(CALIBRATION_ANCHORS)
        .filter(pl.col("generation_status") == "ok")
        .select(
            [
                "n_heavy_atoms",
                "molecular_weight",
                "formal_charge",
                "steric_volume_A3",
                "bbox_x_A",
                "bbox_y_A",
                "bbox_z_A",
                "mmff_energy_kcal_mol",
            ]
        )
        .head(limit)
        .collect()
    )
    if frame.height < limit:
        raise AssertionError(f"expected {limit} calibration anchors, found {frame.height}")
    tensor = torch.tensor(frame.to_numpy(), dtype=torch.float32)
    mean = tensor.mean(dim=0, keepdim=True)
    std = tensor.std(dim=0, keepdim=True).clamp_min(1.0e-6)
    return (tensor - mean) / std


def mock_trajectory_batch(anchor_embeddings: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    torch.manual_seed(77123)
    flat_batch = BATCH_SIZE * TRAJECTORY_LENGTH
    chosen_actions = (torch.arange(flat_batch, dtype=torch.long) * 17 + 3) % int(anchor_embeddings.shape[0])
    chosen_anchor_features = anchor_embeddings.index_select(0, chosen_actions)
    repeated = chosen_anchor_features[:, None, :].repeat(1, NODE_COUNT, 1)
    node_noise = 0.08 * torch.randn(flat_batch, NODE_COUNT, ACTION_EMBEDDING_DIM)
    node_offsets = torch.linspace(-0.15, 0.15, steps=NODE_COUNT).view(1, NODE_COUNT, 1)
    node_features = repeated + node_noise + node_offsets
    node_mask = torch.ones(flat_batch, NODE_COUNT, dtype=torch.bool)
    forward_mask = torch.ones(flat_batch, int(anchor_embeddings.shape[0]), dtype=torch.bool)
    invalid_columns = (torch.arange(int(anchor_embeddings.shape[0])) % 29) == 0
    forward_mask[:, invalid_columns] = False
    forward_mask[torch.arange(flat_batch), chosen_actions] = True
    backward_mask = torch.ones(flat_batch, TRAJECTORY_LENGTH, dtype=torch.bool)
    chosen_parents = torch.arange(flat_batch, dtype=torch.long) % TRAJECTORY_LENGTH
    trajectory_action_features = chosen_anchor_features.view(BATCH_SIZE, TRAJECTORY_LENGTH, ACTION_EMBEDDING_DIM)
    reward_signal = trajectory_action_features.mean(dim=(1, 2))
    terminal_rewards = torch.exp(0.65 + reward_signal.clamp(min=-0.5, max=0.8))
    return node_features, node_mask, forward_mask, backward_mask, chosen_actions, chosen_parents, terminal_rewards


def trajectory_log_probs(
    output: PolicyOutput,
    chosen_actions: Tensor,
    chosen_parents: Tensor,
) -> tuple[Tensor, Tensor]:
    forward = output.forward_log_probs.gather(1, chosen_actions.view(-1, 1)).view(BATCH_SIZE, TRAJECTORY_LENGTH)
    backward = output.backward_log_probs.gather(1, chosen_parents.view(-1, 1)).view(BATCH_SIZE, TRAJECTORY_LENGTH)
    return forward, backward


def grad_norm(tensor: Tensor | None) -> float:
    if tensor is None:
        return 0.0
    return float(tensor.detach().abs().sum().item())


def compute_loss(
    policy: AnchorAttentionGFlowNetPolicy,
    tb_loss: TrajectoryBalanceLoss,
    node_features: Tensor,
    node_mask: Tensor,
    forward_mask: Tensor,
    backward_mask: Tensor,
    chosen_actions: Tensor,
    chosen_parents: Tensor,
    terminal_rewards: Tensor,
) -> Tensor:
    output = cast(PolicyOutput, policy(node_features, node_mask, forward_mask, backward_mask))
    forward_log_probs, backward_log_probs = trajectory_log_probs(output, chosen_actions, chosen_parents)
    tb_output = cast(
        TrajectoryBalanceOutput,
        tb_loss(output.log_z, forward_log_probs, backward_log_probs, terminal_rewards),
    )
    return tb_output.loss


def test_gflownet_trajectory_balance_loss_converges(capsys: object) -> None:
    torch.manual_seed(80219)
    anchor_embeddings = load_calibration_anchor_embeddings()
    policy = AnchorAttentionGFlowNetPolicy(
        node_feature_dim=ACTION_EMBEDDING_DIM,
        anchor_embeddings=anchor_embeddings,
        hidden_dim=48,
        max_backward_actions=TRAJECTORY_LENGTH,
    )
    tb_loss = TrajectoryBalanceLoss()
    (
        node_features,
        node_mask,
        forward_mask,
        backward_mask,
        chosen_actions,
        chosen_parents,
        terminal_rewards,
    ) = mock_trajectory_batch(anchor_embeddings)

    initial_loss = compute_loss(
        policy,
        tb_loss,
        node_features,
        node_mask,
        forward_mask,
        backward_mask,
        chosen_actions,
        chosen_parents,
        terminal_rewards,
    )
    assert bool(torch.isfinite(initial_loss).item())
    initial_loss.backward()  # type: ignore[no-untyped-call]
    assert policy.log_z.grad is not None
    assert bool(torch.isfinite(policy.log_z.grad).all().item())
    assert grad_norm(policy.log_z.grad) > 0.0
    first_weight = cast(torch.nn.Linear, policy.node_encoder[0]).weight
    assert first_weight.grad is not None
    assert bool(torch.isfinite(first_weight.grad).all().item())
    assert grad_norm(first_weight.grad) > 0.0

    optimizer = torch.optim.AdamW(policy.parameters(), lr=3.0e-3, weight_decay=1.0e-4)
    initial_scalar = float(initial_loss.detach().item())
    final_loss = initial_loss.detach()
    for _ in range(50):
        optimizer.zero_grad(set_to_none=True)
        loss = compute_loss(
            policy,
            tb_loss,
            node_features,
            node_mask,
            forward_mask,
            backward_mask,
            chosen_actions,
            chosen_parents,
            terminal_rewards,
        )
        loss.backward()  # type: ignore[no-untyped-call]
        optimizer.step()
        final_loss = loss.detach()

    final_scalar = float(final_loss.item())
    print(f"initial_L_TB={initial_scalar:.6f} final_L_TB={final_scalar:.6f}")
    assert bool(torch.isfinite(final_loss).item())
    assert final_scalar < initial_scalar
