"""Trajectory Balance loss for PRISM-DSTW GFlowNets."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn


@dataclass(frozen=True)
class TrajectoryBalanceOutput:
    """Batch Trajectory Balance diagnostics."""

    loss: Tensor
    residual: Tensor
    sum_forward_log_prob: Tensor
    sum_backward_log_prob: Tensor
    log_reward: Tensor


class TrajectoryBalanceLoss(nn.Module):
    """Vectorized GFlowNet Trajectory Balance objective.

    For each trajectory:
    ``(logZ + sum_t log P_F - log R(s_n) - sum_t log P_B)^2``.
    """

    def __init__(self, reward_epsilon: float = 1.0e-8) -> None:
        super().__init__()
        if reward_epsilon <= 0.0:
            raise ValueError("reward_epsilon must be positive")
        self.reward_epsilon = reward_epsilon

    def forward(
        self,
        log_z: Tensor,
        forward_log_probs: Tensor,
        backward_log_probs: Tensor,
        terminal_rewards: Tensor,
        trajectory_mask: Tensor | None = None,
    ) -> TrajectoryBalanceOutput:
        if forward_log_probs.shape != backward_log_probs.shape:
            raise ValueError("forward_log_probs and backward_log_probs must have identical shape")
        if forward_log_probs.ndim != 2:
            raise ValueError("trajectory log probabilities must have shape [batch, trajectory_length]")
        if terminal_rewards.shape != forward_log_probs.shape[:1]:
            raise ValueError("terminal_rewards must have shape [batch]")
        if bool((terminal_rewards <= 0.0).any().item()):
            raise ValueError("terminal rewards must be strictly positive")
        if trajectory_mask is None:
            weights = torch.ones_like(forward_log_probs)
        else:
            if trajectory_mask.shape != forward_log_probs.shape:
                raise ValueError("trajectory_mask must match trajectory log-probability shape")
            if trajectory_mask.dtype != torch.bool:
                raise TypeError("trajectory_mask must have dtype torch.bool")
            weights = trajectory_mask.to(dtype=forward_log_probs.dtype)
        sum_forward = (forward_log_probs * weights).sum(dim=1)
        sum_backward = (backward_log_probs * weights).sum(dim=1)
        log_reward = terminal_rewards.clamp_min(self.reward_epsilon).log()
        if log_z.ndim == 0:
            log_z_batch = log_z.expand_as(log_reward)
        elif log_z.shape == log_reward.shape:
            log_z_batch = log_z
        else:
            raise ValueError("log_z must be scalar or have shape [batch]")
        residual = log_z_batch + sum_forward - log_reward - sum_backward
        return TrajectoryBalanceOutput(
            loss=residual.square().mean(),
            residual=residual,
            sum_forward_log_prob=sum_forward,
            sum_backward_log_prob=sum_backward,
            log_reward=log_reward,
        )


def trajectory_balance_loss_from_action_tables(
    *,
    log_z: Tensor,
    forward_log_prob_tables: Tensor,
    backward_log_prob_tables: Tensor,
    forward_action_indices: Tensor,
    backward_action_indices: Tensor,
    terminal_rewards: Tensor,
    trajectory_mask: Tensor | None = None,
    reward_epsilon: float = 1.0e-8,
) -> TrajectoryBalanceOutput:
    """Vectorized TB loss from batched action log-probability tables.

    ``forward_log_prob_tables`` and ``backward_log_prob_tables`` have shape
    ``[batch, trajectory_length, action_count]``. The selected actions are
    extracted with ``torch.gather`` and reduced with ``torch.bmm``.
    """

    if reward_epsilon <= 0.0:
        raise ValueError("reward_epsilon must be positive")
    if forward_log_prob_tables.ndim != 3 or backward_log_prob_tables.ndim != 3:
        raise ValueError("log probability tables must have shape [batch, trajectory_length, action_count]")
    if forward_log_prob_tables.shape[:2] != backward_log_prob_tables.shape[:2]:
        raise ValueError("forward/backward tables must agree on batch and trajectory dimensions")
    if forward_action_indices.shape != forward_log_prob_tables.shape[:2]:
        raise ValueError("forward_action_indices must have shape [batch, trajectory_length]")
    if backward_action_indices.shape != backward_log_prob_tables.shape[:2]:
        raise ValueError("backward_action_indices must have shape [batch, trajectory_length]")
    if forward_action_indices.dtype != torch.long or backward_action_indices.dtype != torch.long:
        raise TypeError("action index tensors must have dtype torch.long")
    if terminal_rewards.shape != forward_log_prob_tables.shape[:1]:
        raise ValueError("terminal_rewards must have shape [batch]")
    if bool((terminal_rewards <= 0.0).any().item()):
        raise ValueError("terminal rewards must be strictly positive")
    forward_selected = torch.gather(forward_log_prob_tables, 2, forward_action_indices.unsqueeze(-1)).squeeze(-1)
    backward_selected = torch.gather(backward_log_prob_tables, 2, backward_action_indices.unsqueeze(-1)).squeeze(-1)
    if trajectory_mask is None:
        weights = torch.ones_like(forward_selected)
    else:
        if trajectory_mask.shape != forward_selected.shape:
            raise ValueError("trajectory_mask must match selected log-probability shape")
        if trajectory_mask.dtype != torch.bool:
            raise TypeError("trajectory_mask must have dtype torch.bool")
        weights = trajectory_mask.to(dtype=forward_selected.dtype)
    weight_row = weights.unsqueeze(1)
    sum_forward = torch.bmm(weight_row, forward_selected.unsqueeze(-1)).squeeze(-1).squeeze(-1)
    sum_backward = torch.bmm(weight_row, backward_selected.unsqueeze(-1)).squeeze(-1).squeeze(-1)
    log_reward = terminal_rewards.clamp_min(reward_epsilon).log()
    if log_z.ndim == 0:
        log_z_batch = log_z.expand_as(log_reward)
    elif log_z.shape == log_reward.shape:
        log_z_batch = log_z
    else:
        raise ValueError("log_z must be scalar or have shape [batch]")
    residual = log_z_batch + sum_forward - log_reward - sum_backward
    return TrajectoryBalanceOutput(
        loss=residual.square().mean(),
        residual=residual,
        sum_forward_log_prob=sum_forward,
        sum_backward_log_prob=sum_backward,
        log_reward=log_reward,
    )


__all__ = [
    "TrajectoryBalanceLoss",
    "TrajectoryBalanceOutput",
    "trajectory_balance_loss_from_action_tables",
]
