"""Dot-product attention policies for PRISM-DSTW GFlowNets."""

from __future__ import annotations

import math
from dataclasses import dataclass
from importlib import import_module
from typing import Any, cast

import torch
from torch import Tensor, nn
from torch.nn import functional as F

_PyGGCNConv: Any = None
try:
    _PyGGCNConv = getattr(import_module("torch_geometric.nn"), "GCNConv")
except ImportError:
    _PyGGCNConv = None


@dataclass(frozen=True)
class PolicyOutput:
    """Forward and backward policy tensors for one scaffold-state batch."""

    state_embedding: Tensor
    forward_logits: Tensor
    forward_log_probs: Tensor
    forward_probs: Tensor
    backward_logits: Tensor
    backward_log_probs: Tensor
    backward_probs: Tensor
    log_z: Tensor


@dataclass(frozen=True)
class TrilinearPolicyOutput:
    """Forward/backward tensors for atom x synthon x reaction-rule actions."""

    node_embeddings: Tensor
    graph_embeddings: Tensor
    forward_logits: Tensor
    forward_log_probs: Tensor
    forward_probs: Tensor
    backward_logits: Tensor
    backward_log_probs: Tensor
    backward_probs: Tensor
    log_z: Tensor


@dataclass(frozen=True)
class DualChannelPolicyOutput:
    """Forward/backward tensors for field-conditioned packed graph states."""

    node_embeddings: Tensor
    electronic_embeddings: Tensor
    steric_embeddings: Tensor
    graph_embeddings: Tensor
    phase_global_embeddings: Tensor
    exit_embeddings: Tensor
    action_context_embeddings: Tensor
    action_field_embeddings: Tensor | None
    action_base_embeddings: Tensor | None
    action_atom_embeddings: Tensor | None
    channel_gate: Tensor
    forward_logits: Tensor
    forward_log_probs: Tensor
    forward_probs: Tensor
    backward_logits: Tensor
    backward_log_probs: Tensor
    backward_probs: Tensor
    log_z: Tensor


@dataclass(frozen=True)
class PhaseResolvedFiberOutput:
    """Phase-axis embeddings that preserve the five CCNS states."""

    h_fiber_per_phase: Tensor
    h_fiber_summary: Tensor
    hysteresis_embedding: Tensor
    activation_gradient_embedding: Tensor


@dataclass(frozen=True)
class OrthogonalMessageOutput:
    """Separated base-space and fiber-space message passing outputs."""

    h_base: Tensor
    h_fiber: Tensor


@dataclass(frozen=True)
class FiberBundlePolicyOutput:
    """Forward/backward tensors for the Fiber-Bundle GFlowNet policy."""

    phase_output: PhaseResolvedFiberOutput
    orthogonal_output: OrthogonalMessageOutput
    node_embeddings: Tensor
    graph_embeddings: Tensor
    phase_global_embeddings: Tensor
    exit_embeddings: Tensor
    query_embeddings: Tensor
    fusion_gate: Tensor
    forward_logits: Tensor
    forward_log_probs: Tensor
    forward_probs: Tensor
    backward_logits: Tensor
    backward_log_probs: Tensor
    backward_probs: Tensor
    log_z: Tensor


def _validate_mask(name: str, mask: Tensor, expected_shape: torch.Size) -> None:
    if mask.shape != expected_shape:
        raise ValueError(f"{name} must have shape {tuple(expected_shape)}, got {tuple(mask.shape)}")
    if mask.dtype != torch.bool:
        raise TypeError(f"{name} must have dtype torch.bool")
    if not bool(mask.any(dim=-1).all().item()):
        raise ValueError(f"{name} must leave at least one valid option per batch row")


def masked_log_softmax(logits: Tensor, mask: Tensor) -> Tensor:
    """Apply a numerically stable masked log-softmax over the final dimension."""

    _validate_mask("mask", mask, logits.shape)
    floor = torch.finfo(logits.dtype).min
    return torch.log_softmax(logits.masked_fill(~mask, floor), dim=-1)


def _scatter_mean(values: Tensor, batch_index: Tensor, batch_size: int) -> Tensor:
    if batch_index.ndim != 1 or batch_index.shape[0] != values.shape[0]:
        raise ValueError("batch_index must have shape [num_nodes]")
    if batch_index.dtype != torch.long:
        raise TypeError("batch_index must have dtype torch.long")
    sums = values.new_zeros((batch_size, values.shape[1]))
    counts = values.new_zeros((batch_size, 1))
    sums.index_add_(0, batch_index, values)
    counts.index_add_(0, batch_index, torch.ones((values.shape[0], 1), dtype=values.dtype, device=values.device))
    if bool((counts <= 0.0).any().item()):
        raise ValueError("each graph in the batch must contain at least one node")
    return sums / counts.clamp_min(1.0)


def _masked_graph_log_softmax(
    logits: Tensor,
    mask: Tensor,
    batch_index: Tensor,
    batch_size: int,
) -> Tensor:
    if logits.shape != mask.shape:
        raise ValueError("forward_action_mask must match trilinear logits shape")
    if logits.ndim != 3:
        raise ValueError("trilinear logits must have shape [num_nodes, num_synthons, num_rules]")
    if mask.dtype != torch.bool:
        raise TypeError("forward_action_mask must have dtype torch.bool")
    if batch_index.dtype != torch.long:
        raise TypeError("batch_index must have dtype torch.long")
    actions_per_node = int(logits.shape[1] * logits.shape[2])
    flat_batch = batch_index.view(-1, 1).expand(logits.shape[0], actions_per_node).reshape(-1)
    flat_mask = mask.reshape(-1)
    valid_counts = torch.zeros(batch_size, dtype=torch.long, device=logits.device)
    valid_counts.scatter_add_(0, flat_batch, flat_mask.to(dtype=torch.long))
    if not bool((valid_counts > 0).all().item()):
        raise ValueError("forward_action_mask must leave at least one valid action per graph")

    floor = torch.finfo(logits.dtype).min
    values = logits.masked_fill(~mask, floor).reshape(-1)
    max_per_graph = torch.full((batch_size,), floor, dtype=logits.dtype, device=logits.device)
    max_per_graph.scatter_reduce_(0, flat_batch, values, reduce="amax", include_self=True)
    exp_values = torch.exp(values - max_per_graph.index_select(0, flat_batch)) * flat_mask.to(dtype=logits.dtype)
    sum_exp = torch.zeros(batch_size, dtype=logits.dtype, device=logits.device)
    sum_exp.scatter_add_(0, flat_batch, exp_values)
    log_norm = max_per_graph + sum_exp.clamp_min(torch.finfo(logits.dtype).tiny).log()
    log_probs = values - log_norm.index_select(0, flat_batch)
    return log_probs.reshape_as(logits).masked_fill(~mask, floor)


class AnchorAttentionGFlowNetPolicy(nn.Module):
    """GFlowNet forward/backward policy using anchor dot-product attention.

    The forward policy scales to large action spaces by comparing each scaffold
    state embedding against the pre-computed chemical-anchor embedding table:
    ``logits = h @ E_anchor.T / sqrt(d)``.
    """

    def __init__(
        self,
        *,
        node_feature_dim: int,
        anchor_embeddings: Tensor,
        hidden_dim: int = 64,
        state_embedding_dim: int | None = None,
        max_backward_actions: int = 8,
        learn_anchor_embeddings: bool = False,
    ) -> None:
        super().__init__()
        if anchor_embeddings.ndim != 2:
            raise ValueError("anchor_embeddings must have shape [num_anchors, embedding_dim]")
        if max_backward_actions < 1:
            raise ValueError("max_backward_actions must be positive")
        embedding_dim = int(anchor_embeddings.shape[1])
        if state_embedding_dim is not None and state_embedding_dim != embedding_dim:
            raise ValueError("state_embedding_dim must match anchor embedding dimension for dot-product attention")
        self.anchor_embedding_dim = embedding_dim
        self.num_anchors = int(anchor_embeddings.shape[0])
        self.max_backward_actions = max_backward_actions
        self.node_encoder = nn.Sequential(
            nn.Linear(node_feature_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        )
        self.state_projection = nn.Linear(hidden_dim, embedding_dim)
        self.backward_head = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, max_backward_actions),
        )
        self.log_z = nn.Parameter(torch.zeros(()))
        prepared_embeddings = anchor_embeddings.detach().clone().to(dtype=torch.float32)
        if learn_anchor_embeddings:
            self.anchor_embeddings = nn.Parameter(prepared_embeddings)
        else:
            self.register_buffer("anchor_embeddings", prepared_embeddings)

    def embed_state(self, node_features: Tensor, node_mask: Tensor) -> Tensor:
        """Embed a padded scaffold graph batch by masked node pooling."""

        if node_features.ndim != 3:
            raise ValueError("node_features must have shape [batch, nodes, features]")
        if node_mask.shape != node_features.shape[:2]:
            raise ValueError("node_mask must have shape [batch, nodes]")
        if node_mask.dtype != torch.bool:
            raise TypeError("node_mask must have dtype torch.bool")
        if not bool(node_mask.any(dim=1).all().item()):
            raise ValueError("each scaffold graph must contain at least one valid node")
        hidden = self.node_encoder(node_features)
        weights = node_mask.to(dtype=hidden.dtype).unsqueeze(-1)
        pooled = (hidden * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)
        return cast(Tensor, self.state_projection(pooled))

    def forward(
        self,
        node_features: Tensor,
        node_mask: Tensor,
        forward_action_mask: Tensor,
        backward_action_mask: Tensor,
    ) -> PolicyOutput:
        state_embedding = self.embed_state(node_features, node_mask)
        forward_logits = state_embedding.matmul(self.anchor_embeddings.transpose(0, 1)) / math.sqrt(
            float(self.anchor_embedding_dim)
        )
        _validate_mask("forward_action_mask", forward_action_mask, forward_logits.shape)
        if int(backward_action_mask.shape[1]) > self.max_backward_actions:
            raise ValueError("backward_action_mask exceeds max_backward_actions")
        raw_backward_logits = self.backward_head(state_embedding)
        backward_logits = raw_backward_logits[:, : int(backward_action_mask.shape[1])]
        _validate_mask("backward_action_mask", backward_action_mask, backward_logits.shape)
        forward_log_probs = masked_log_softmax(forward_logits, forward_action_mask)
        backward_log_probs = masked_log_softmax(backward_logits, backward_action_mask)
        return PolicyOutput(
            state_embedding=state_embedding,
            forward_logits=forward_logits,
            forward_log_probs=forward_log_probs,
            forward_probs=forward_log_probs.exp(),
            backward_logits=backward_logits,
            backward_log_probs=backward_log_probs,
            backward_probs=backward_log_probs.exp(),
            log_z=self.log_z,
        )


class PhaseResolvedFiberEmbedder(nn.Module):
    """Embed `[N_atoms, 5, D_features]` CCNS phase fibers.

    The phase axis is processed before any projection to atom-level summaries,
    preserving cold-hold, ramp-up, warm-hold, ramp-down, and cold-return order.
    """

    expected_phase_count = 5

    def __init__(self, *, phase_feature_dim: int, hidden_dim: int, embedding_dim: int) -> None:
        super().__init__()
        if phase_feature_dim < 1:
            raise ValueError("phase_feature_dim must be positive")
        if hidden_dim < 1 or embedding_dim < 1:
            raise ValueError("hidden_dim and embedding_dim must be positive")
        self.phase_feature_dim = phase_feature_dim
        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim
        self.phase_input = nn.Sequential(
            nn.Linear(phase_feature_dim, hidden_dim),
            nn.SiLU(),
        )
        self.phase_gru = nn.GRU(
            input_size=hidden_dim,
            hidden_size=embedding_dim,
            batch_first=True,
            bidirectional=False,
        )
        self.summary_projection = nn.Sequential(
            nn.Linear(embedding_dim * 2, embedding_dim),
            nn.SiLU(),
            nn.Linear(embedding_dim, embedding_dim),
        )
        self.hysteresis_projection = nn.Sequential(
            nn.Linear(embedding_dim * 2, embedding_dim),
            nn.SiLU(),
            nn.Linear(embedding_dim, embedding_dim),
        )
        self.activation_projection = nn.Sequential(
            nn.Linear(embedding_dim * 2, embedding_dim),
            nn.SiLU(),
            nn.Linear(embedding_dim, embedding_dim),
        )

    def forward(self, x_phase: Tensor) -> PhaseResolvedFiberOutput:
        if x_phase.ndim != 3:
            raise ValueError("x_phase must have shape [num_atoms, 5, phase_feature_dim]")
        if int(x_phase.shape[1]) != self.expected_phase_count:
            raise ValueError("x_phase must preserve exactly five CCNS phases")
        if int(x_phase.shape[2]) != self.phase_feature_dim:
            raise ValueError("x_phase feature dimension mismatch")
        phase_hidden = self.phase_input(x_phase)
        h_fiber_per_phase, final_hidden = self.phase_gru(phase_hidden)
        final_state = final_hidden.squeeze(0)
        phase_mean = h_fiber_per_phase.mean(dim=1)
        h_fiber_summary = self.summary_projection(torch.cat([phase_mean, final_state], dim=1))

        cold_hold = h_fiber_per_phase[:, 0, :]
        ramp_up = h_fiber_per_phase[:, 1, :]
        warm_hold = h_fiber_per_phase[:, 2, :]
        cold_return = h_fiber_per_phase[:, 4, :]
        hysteresis_embedding = self.hysteresis_projection(
            torch.cat([warm_hold - cold_return, cold_hold - cold_return], dim=1)
        )
        activation_gradient_embedding = self.activation_projection(
            torch.cat([warm_hold - cold_hold, ramp_up - cold_hold], dim=1)
        )
        return PhaseResolvedFiberOutput(
            h_fiber_per_phase=h_fiber_per_phase,
            h_fiber_summary=h_fiber_summary,
            hysteresis_embedding=hysteresis_embedding,
            activation_gradient_embedding=activation_gradient_embedding,
        )


class OrthogonalMessagePassing(nn.Module):
    """Separate base-space graph routing and within-atom fiber routing."""

    def __init__(
        self,
        *,
        base_feature_dim: int,
        fiber_embedding_dim: int,
        edge_feature_dim: int,
        hidden_dim: int,
        output_dim: int,
        rf_mode: str = "gain",
    ) -> None:
        super().__init__()
        if min(base_feature_dim, fiber_embedding_dim, edge_feature_dim, hidden_dim, output_dim) < 1:
            raise ValueError("all dimensions must be positive")
        if rf_mode not in {"gain", "hard_zero", "soft"}:
            raise ValueError("rf_mode must be one of: gain, hard_zero, soft")
        self.base_feature_dim = base_feature_dim
        self.fiber_embedding_dim = fiber_embedding_dim
        self.edge_feature_dim = edge_feature_dim
        self.rf_mode = rf_mode
        self.base_input = nn.Sequential(
            nn.Linear(base_feature_dim, hidden_dim),
            nn.SiLU(),
        )
        self.base_message = nn.Sequential(
            nn.Linear(hidden_dim + edge_feature_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.base_update = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, output_dim),
        )
        self.phase_conv = nn.Conv1d(fiber_embedding_dim, fiber_embedding_dim, kernel_size=3, padding=1)
        self.fiber_update = nn.Sequential(
            nn.Linear(fiber_embedding_dim * 4, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def route_base_space(
        self,
        *,
        x_base: Tensor,
        edge_index: Tensor,
        edge_attr: Tensor,
        active_dendrite_mask: Tensor,
    ) -> Tensor:
        if x_base.ndim != 2 or int(x_base.shape[1]) != self.base_feature_dim:
            raise ValueError("x_base must have shape [num_atoms, base_feature_dim]")
        if edge_index.ndim != 2 or int(edge_index.shape[0]) != 2:
            raise ValueError("edge_index must have shape [2, num_edges]")
        if edge_index.dtype != torch.long:
            raise TypeError("edge_index must have dtype long")
        if edge_attr.ndim != 2 or int(edge_attr.shape[1]) != self.edge_feature_dim:
            raise ValueError("edge_attr must have shape [num_edges, edge_feature_dim]")
        if int(edge_attr.shape[0]) != int(edge_index.shape[1]):
            raise ValueError("edge_attr row count must match edge_index")
        if active_dendrite_mask.shape != edge_attr.shape[:1]:
            raise ValueError("active_dendrite_mask must have shape [num_edges]")
        if active_dendrite_mask.dtype != torch.bool:
            raise TypeError("active_dendrite_mask must have dtype bool")

        hidden = self.base_input(x_base)
        if edge_index.numel() == 0:
            aggregate = hidden.new_zeros(hidden.shape)
        else:
            src = edge_index[0]
            dst = edge_index[1]
            message_input = torch.cat([hidden.index_select(0, src), edge_attr.to(dtype=hidden.dtype)], dim=1)
            messages = self.base_message(message_input)
            if self.rf_mode == "hard_zero":
                edge_weight = active_dendrite_mask.to(dtype=hidden.dtype).unsqueeze(-1)
            elif self.rf_mode == "soft":
                edge_weight = 0.1 + 0.9 * active_dendrite_mask.to(dtype=hidden.dtype).unsqueeze(-1)
            else:
                edge_weight = 1.0 + active_dendrite_mask.to(dtype=hidden.dtype).unsqueeze(-1)
            messages = messages * edge_weight
            aggregate = hidden.new_zeros(hidden.shape)
            aggregate.index_add_(0, dst, messages)
            counts = hidden.new_zeros((hidden.shape[0], 1))
            counts.index_add_(0, dst, edge_weight)
            aggregate = aggregate / counts.clamp_min(1.0)
        return cast(Tensor, self.base_update(torch.cat([hidden, aggregate], dim=1)))

    def route_fiber_space(self, phase_output: PhaseResolvedFiberOutput) -> Tensor:
        phase_tensor = phase_output.h_fiber_per_phase
        if phase_tensor.ndim != 3 or int(phase_tensor.shape[1]) != 5:
            raise ValueError("phase tensor must have shape [num_atoms, 5, fiber_embedding_dim]")
        conv_input = phase_tensor.transpose(1, 2)
        phase_routed = F.silu(self.phase_conv(conv_input)).transpose(1, 2)
        phase_summary = phase_routed.mean(dim=1)
        return cast(
            Tensor,
            self.fiber_update(
                torch.cat(
                    [
                        phase_summary,
                        phase_output.h_fiber_summary,
                        phase_output.hysteresis_embedding,
                        phase_output.activation_gradient_embedding,
                    ],
                    dim=1,
                )
            ),
        )

    def forward(
        self,
        *,
        x_base: Tensor,
        phase_output: PhaseResolvedFiberOutput,
        edge_index: Tensor,
        edge_attr: Tensor,
        active_dendrite_mask: Tensor,
    ) -> OrthogonalMessageOutput:
        h_base = self.route_base_space(
            x_base=x_base,
            edge_index=edge_index,
            edge_attr=edge_attr,
            active_dendrite_mask=active_dendrite_mask,
        )
        h_fiber = self.route_fiber_space(phase_output)
        return OrthogonalMessageOutput(h_base=h_base, h_fiber=h_fiber)


class FiberBundleGFlowNetPolicy(nn.Module):
    """Fiber-Bundle GNN policy with exit-vector-conditioned anchor attention."""

    def __init__(
        self,
        *,
        base_feature_dim: int,
        phase_feature_dim: int,
        edge_feature_dim: int,
        anchor_embeddings: Tensor,
        hidden_dim: int = 96,
        embedding_dim: int = 64,
        max_backward_actions: int = 8,
        learn_anchor_embeddings: bool = False,
        rf_mode: str = "gain",
    ) -> None:
        super().__init__()
        if anchor_embeddings.ndim != 2:
            raise ValueError("anchor_embeddings must have shape [num_anchors, embedding_dim]")
        if int(anchor_embeddings.shape[1]) != embedding_dim:
            raise ValueError("anchor embedding width must match embedding_dim")
        if max_backward_actions < 1:
            raise ValueError("max_backward_actions must be positive")
        self.embedding_dim = embedding_dim
        self.num_anchors = int(anchor_embeddings.shape[0])
        self.max_backward_actions = max_backward_actions
        self.phase_embedder = PhaseResolvedFiberEmbedder(
            phase_feature_dim=phase_feature_dim,
            hidden_dim=hidden_dim,
            embedding_dim=embedding_dim,
        )
        self.orthogonal_mp = OrthogonalMessagePassing(
            base_feature_dim=base_feature_dim,
            fiber_embedding_dim=embedding_dim,
            edge_feature_dim=edge_feature_dim,
            hidden_dim=hidden_dim,
            output_dim=embedding_dim,
            rf_mode=rf_mode,
        )
        self.fusion_gate = nn.Sequential(
            nn.Linear(embedding_dim * 2, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, embedding_dim),
            nn.Sigmoid(),
        )
        self.query_mlp = nn.Sequential(
            nn.Linear(embedding_dim * 3, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, embedding_dim),
        )
        self.backward_head = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, max_backward_actions),
        )
        self.stop_mlp = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )
        self.log_z = nn.Parameter(torch.zeros(()))
        prepared_embeddings = anchor_embeddings.detach().clone().to(dtype=torch.float32)
        if learn_anchor_embeddings:
            self.anchor_embeddings = nn.Parameter(prepared_embeddings)
        else:
            self.register_buffer("anchor_embeddings", prepared_embeddings)

    def forward(
        self,
        *,
        x_base: Tensor,
        x_phase: Tensor,
        edge_index: Tensor,
        edge_attr: Tensor,
        active_dendrite_mask: Tensor,
        batch_index: Tensor,
        exit_node_indices: Tensor,
        forward_action_mask: Tensor,
        backward_action_mask: Tensor,
    ) -> FiberBundlePolicyOutput:
        if batch_index.ndim != 1 or batch_index.shape[0] != x_base.shape[0]:
            raise ValueError("batch_index must have shape [num_atoms]")
        if batch_index.dtype != torch.long:
            raise TypeError("batch_index must have dtype long")
        batch_size = int(batch_index.max().item()) + 1 if batch_index.numel() else 0
        if batch_size < 1:
            raise ValueError("batch_index must describe at least one graph")
        if exit_node_indices.shape != (batch_size,):
            raise ValueError("exit_node_indices must have shape [batch]")
        phase_output = self.phase_embedder(x_phase)
        orthogonal_output = self.orthogonal_mp(
            x_base=x_base,
            phase_output=phase_output,
            edge_index=edge_index,
            edge_attr=edge_attr,
            active_dendrite_mask=active_dendrite_mask,
        )
        fusion_gate = self.fusion_gate(torch.cat([orthogonal_output.h_base, orthogonal_output.h_fiber], dim=1))
        node_embeddings = fusion_gate * orthogonal_output.h_base + (1.0 - fusion_gate) * orthogonal_output.h_fiber
        graph_embeddings = _scatter_mean(node_embeddings, batch_index, batch_size)
        phase_global_embeddings = _scatter_mean(orthogonal_output.h_fiber, batch_index, batch_size)
        exit_embeddings = node_embeddings.index_select(0, exit_node_indices)
        query_embeddings = self.query_mlp(torch.cat([exit_embeddings, graph_embeddings, phase_global_embeddings], dim=1))
        anchor_logits = query_embeddings.matmul(self.anchor_embeddings.transpose(0, 1)) / math.sqrt(
            float(self.embedding_dim)
        )
        if int(forward_action_mask.shape[1]) == self.num_anchors + 1:
            forward_logits = torch.cat([anchor_logits, self.stop_mlp(query_embeddings)], dim=1)
        else:
            forward_logits = anchor_logits
        _validate_mask("forward_action_mask", forward_action_mask, forward_logits.shape)
        if int(backward_action_mask.shape[1]) > self.max_backward_actions:
            raise ValueError("backward_action_mask exceeds max_backward_actions")
        backward_logits = self.backward_head(query_embeddings)[:, : int(backward_action_mask.shape[1])]
        _validate_mask("backward_action_mask", backward_action_mask, backward_logits.shape)
        forward_log_probs = masked_log_softmax(forward_logits, forward_action_mask)
        backward_log_probs = masked_log_softmax(backward_logits, backward_action_mask)
        return FiberBundlePolicyOutput(
            phase_output=phase_output,
            orthogonal_output=orthogonal_output,
            node_embeddings=node_embeddings,
            graph_embeddings=graph_embeddings,
            phase_global_embeddings=phase_global_embeddings,
            exit_embeddings=exit_embeddings,
            query_embeddings=query_embeddings,
            fusion_gate=fusion_gate,
            forward_logits=forward_logits,
            forward_log_probs=forward_log_probs,
            forward_probs=forward_log_probs.exp(),
            backward_logits=backward_logits,
            backward_log_probs=backward_log_probs,
            backward_probs=backward_log_probs.exp(),
            log_z=self.log_z,
        )


class FieldConditionedDualChannelGFlowNetPolicy(nn.Module):
    """Dual-channel policy over PyG-packed scaffold states.

    The electronic channel only propagates over RDKit-derived conjugated or
    aromatic bonds. The steric channel propagates over all graph edges with an
    inverse-distance weighting so aliphatic bulk remains visible to the policy.
    Node features are expected to include atom descriptors plus local PRISM
    field context such as Pi_clash, Pi_complement, and shear stress.
    """

    def __init__(
        self,
        *,
        node_feature_dim: int | None = None,
        base_feature_dim: int | None = None,
        phase_feature_dim: int | None = None,
        edge_feature_dim: int | None = None,
        anchor_embeddings: Tensor,
        hidden_dim: int = 96,
        embedding_dim: int | None = None,
        max_backward_actions: int = 8,
        distance_decay_a: float = 2.5,
        learn_anchor_embeddings: bool = False,
    ) -> None:
        super().__init__()
        if anchor_embeddings.ndim != 2:
            raise ValueError("anchor_embeddings must have shape [num_anchors, embedding_dim]")
        if max_backward_actions < 1:
            raise ValueError("max_backward_actions must be positive")
        if distance_decay_a <= 0.0:
            raise ValueError("distance_decay_a must be positive")
        resolved_node_feature_dim = node_feature_dim if node_feature_dim is not None else base_feature_dim
        if resolved_node_feature_dim is None or resolved_node_feature_dim < 1:
            raise ValueError("node_feature_dim or base_feature_dim must be positive")
        if edge_feature_dim is not None and edge_feature_dim < 1:
            raise ValueError("edge_feature_dim must be positive when provided")
        resolved_embedding_dim = int(embedding_dim or anchor_embeddings.shape[1])
        if int(anchor_embeddings.shape[1]) != resolved_embedding_dim:
            raise ValueError("anchor_embeddings width must match embedding_dim")
        self.embedding_dim = resolved_embedding_dim
        self.num_anchors = int(anchor_embeddings.shape[0])
        self.max_backward_actions = max_backward_actions
        self.distance_decay_a = float(distance_decay_a)
        self.phase_embedder = (
            PhaseResolvedFiberEmbedder(
                phase_feature_dim=int(phase_feature_dim),
                hidden_dim=hidden_dim,
                embedding_dim=resolved_embedding_dim,
            )
            if phase_feature_dim is not None
            else None
        )
        self.node_input = nn.Sequential(
            nn.Linear(int(resolved_node_feature_dim), hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        )
        self.electronic_message = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.electronic_update = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, resolved_embedding_dim),
        )
        self.steric_message = nn.Sequential(
            nn.Linear(hidden_dim + 1, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.steric_update = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, resolved_embedding_dim),
        )
        self.channel_gate = nn.Sequential(
            nn.Linear(resolved_embedding_dim * 2, resolved_embedding_dim),
            nn.Sigmoid(),
        )
        self.action_context = nn.Sequential(
            nn.Linear(resolved_embedding_dim * 3, resolved_embedding_dim),
            nn.SiLU(),
            nn.Linear(resolved_embedding_dim, resolved_embedding_dim),
        )
        self.action_field_projection = nn.Sequential(
            nn.Linear(resolved_embedding_dim * 3, resolved_embedding_dim),
            nn.SiLU(),
            nn.Linear(resolved_embedding_dim, resolved_embedding_dim),
        )
        self.action_base_projection = nn.Sequential(
            nn.Linear(int(resolved_node_feature_dim), hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, resolved_embedding_dim),
        )
        self.action_atom_projection = nn.Sequential(
            nn.Linear(int(resolved_node_feature_dim), hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, resolved_embedding_dim),
        )
        self.backward_head = nn.Sequential(
            nn.Linear(resolved_embedding_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, max_backward_actions),
        )
        self.stop_mlp = nn.Sequential(
            nn.Linear(resolved_embedding_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )
        self.log_z = nn.Parameter(torch.zeros(()))
        prepared_embeddings = anchor_embeddings.detach().clone().to(dtype=torch.float32)
        if learn_anchor_embeddings:
            self.anchor_embeddings = nn.Parameter(prepared_embeddings)
        else:
            self.register_buffer("anchor_embeddings", prepared_embeddings)

    def _aggregate_messages(
        self,
        *,
        node_hidden: Tensor,
        edge_index: Tensor,
        edge_distance_a: Tensor,
        active_dendrite_mask: Tensor,
    ) -> tuple[Tensor, Tensor]:
        if edge_index.ndim != 2 or edge_index.shape[0] != 2:
            raise ValueError("edge_index must have shape [2, num_edges]")
        if edge_distance_a.ndim != 1 or edge_distance_a.shape[0] != edge_index.shape[1]:
            raise ValueError("edge_distance_a must have shape [num_edges]")
        if active_dendrite_mask.shape != edge_distance_a.shape:
            raise ValueError("active_dendrite_mask must have shape [num_edges]")
        if active_dendrite_mask.dtype != torch.bool:
            raise TypeError("active_dendrite_mask must have dtype bool")
        if edge_index.dtype != torch.long:
            raise TypeError("edge_index must have dtype long")
        src = edge_index[0]
        dst = edge_index[1]
        num_nodes = int(node_hidden.shape[0])

        electronic_messages = self.electronic_message(node_hidden.index_select(0, src))
        electronic_messages = electronic_messages * active_dendrite_mask.to(dtype=node_hidden.dtype).unsqueeze(-1)
        electronic_agg = node_hidden.new_zeros((num_nodes, electronic_messages.shape[1]))
        electronic_agg.index_add_(0, dst, electronic_messages)
        electronic_counts = node_hidden.new_zeros((num_nodes, 1))
        electronic_counts.index_add_(
            0,
            dst,
            active_dendrite_mask.to(dtype=node_hidden.dtype).unsqueeze(-1),
        )
        electronic_agg = electronic_agg / electronic_counts.clamp_min(1.0)

        distance_feature = edge_distance_a.to(dtype=node_hidden.dtype).clamp_min(0.0).unsqueeze(-1)
        steric_input = torch.cat([node_hidden.index_select(0, src), distance_feature], dim=1)
        steric_messages = self.steric_message(steric_input)
        steric_weights = torch.exp(-distance_feature / self.distance_decay_a)
        steric_messages = steric_messages * steric_weights
        steric_agg = node_hidden.new_zeros((num_nodes, steric_messages.shape[1]))
        steric_agg.index_add_(0, dst, steric_messages)
        steric_weight_sums = node_hidden.new_zeros((num_nodes, 1))
        steric_weight_sums.index_add_(0, dst, steric_weights)
        steric_agg = steric_agg / steric_weight_sums.clamp_min(1.0e-6)
        return electronic_agg, steric_agg

    def embed_graph(
        self,
        *,
        node_features: Tensor,
        edge_index: Tensor,
        edge_distance_a: Tensor,
        active_dendrite_mask: Tensor,
        batch_index: Tensor,
        x_phase: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        """Embed a PyG-style disconnected super-graph without dense padding."""

        if node_features.ndim != 2:
            raise ValueError("node_features must have shape [num_nodes, node_feature_dim]")
        if batch_index.ndim != 1 or batch_index.shape[0] != node_features.shape[0]:
            raise ValueError("batch_index must have shape [num_nodes]")
        if batch_index.dtype != torch.long:
            raise TypeError("batch_index must have dtype long")
        batch_size = int(batch_index.max().item()) + 1 if batch_index.numel() else 0
        if batch_size < 1:
            raise ValueError("batch_index must describe at least one graph")
        node_hidden = self.node_input(node_features)
        electronic_agg, steric_agg = self._aggregate_messages(
            node_hidden=node_hidden,
            edge_index=edge_index,
            edge_distance_a=edge_distance_a,
            active_dendrite_mask=active_dendrite_mask,
        )
        electronic_embeddings = self.electronic_update(torch.cat([node_hidden, electronic_agg], dim=1))
        steric_embeddings = self.steric_update(torch.cat([node_hidden, steric_agg], dim=1))
        channel_gate = self.channel_gate(torch.cat([electronic_embeddings, steric_embeddings], dim=1))
        node_embeddings = channel_gate * electronic_embeddings + (1.0 - channel_gate) * steric_embeddings
        phase_global_embeddings = node_embeddings.new_zeros((batch_size, self.embedding_dim))
        if x_phase is not None:
            if self.phase_embedder is None:
                raise ValueError("x_phase was provided but this policy was built without phase_feature_dim")
            phase_output = self.phase_embedder(x_phase)
            node_embeddings = node_embeddings + phase_output.h_fiber_summary
            phase_global_embeddings = _scatter_mean(phase_output.h_fiber_summary, batch_index, batch_size)
        graph_embeddings = _scatter_mean(node_embeddings, batch_index, batch_size)
        return (
            node_embeddings,
            electronic_embeddings,
            steric_embeddings,
            graph_embeddings,
            phase_global_embeddings,
            channel_gate,
        )

    def forward(
        self,
        *,
        node_features: Tensor | None = None,
        x_base: Tensor | None = None,
        x_phase: Tensor | None = None,
        edge_index: Tensor,
        edge_distance_a: Tensor | None = None,
        edge_attr: Tensor | None = None,
        active_dendrite_mask: Tensor,
        batch_index: Tensor,
        exit_node_indices: Tensor,
        forward_action_mask: Tensor,
        backward_action_mask: Tensor,
        action_phase_features: Tensor | None = None,
        action_base_features: Tensor | None = None,
        action_atom_features: Tensor | None = None,
        action_atom_mask: Tensor | None = None,
    ) -> DualChannelPolicyOutput:
        if node_features is None:
            if x_base is None:
                raise ValueError("node_features or x_base must be provided")
            node_features = x_base
        if edge_distance_a is None:
            if edge_attr is None:
                raise ValueError("edge_distance_a or edge_attr must be provided")
            if edge_attr.ndim != 2 or int(edge_attr.shape[1]) < 1:
                raise ValueError("edge_attr must have shape [num_edges, edge_features]")
            edge_distance_a = edge_attr[:, 0].to(dtype=node_features.dtype) * 5.0
        (
            node_embeddings,
            electronic_embeddings,
            steric_embeddings,
            graph_embeddings,
            phase_global_embeddings,
            channel_gate,
        ) = self.embed_graph(
            node_features=node_features,
            edge_index=edge_index,
            edge_distance_a=edge_distance_a,
            active_dendrite_mask=active_dendrite_mask,
            batch_index=batch_index,
            x_phase=x_phase,
        )
        batch_size = int(graph_embeddings.shape[0])
        if exit_node_indices.shape != (batch_size,):
            raise ValueError("exit_node_indices must have shape [batch]")
        if exit_node_indices.dtype != torch.long:
            raise TypeError("exit_node_indices must have dtype long")
        exit_embeddings = node_embeddings.index_select(0, exit_node_indices)
        action_context_embeddings = self.action_context(
            torch.cat([exit_embeddings, graph_embeddings, phase_global_embeddings], dim=1)
        )
        action_embeddings: Tensor = self.anchor_embeddings
        action_field_embeddings: Tensor | None = None
        action_base_embeddings: Tensor | None = None
        action_atom_embeddings: Tensor | None = None
        if action_base_features is not None:
            if action_base_features.ndim != 2:
                raise ValueError("action_base_features must have shape [num_anchors, node_feature_dim]")
            if int(action_base_features.shape[0]) != self.num_anchors:
                raise ValueError("action_base_features row count must match num_anchors")
            if int(action_base_features.shape[1]) != int(node_features.shape[1]):
                raise ValueError("action_base_features width must match node feature dimension")
            action_base_embeddings = self.action_base_projection(action_base_features.to(dtype=node_features.dtype))
            action_embeddings = action_embeddings + action_base_embeddings
        if action_atom_features is not None:
            if action_atom_mask is None:
                raise ValueError("action_atom_mask is required when action_atom_features are provided")
            if action_atom_features.ndim != 3:
                raise ValueError("action_atom_features must have shape [num_anchors, max_atoms, node_feature_dim]")
            if int(action_atom_features.shape[0]) != self.num_anchors:
                raise ValueError("action_atom_features row count must match num_anchors")
            if int(action_atom_features.shape[2]) != int(node_features.shape[1]):
                raise ValueError("action_atom_features width must match node feature dimension")
            if action_atom_mask.shape != action_atom_features.shape[:2]:
                raise ValueError("action_atom_mask must have shape [num_anchors, max_atoms]")
            if action_atom_mask.dtype != torch.bool:
                raise TypeError("action_atom_mask must have dtype bool")
            atom_features = action_atom_features.to(dtype=node_features.dtype)
            flat_atom_embeddings = self.action_atom_projection(atom_features.reshape(-1, int(atom_features.shape[-1])))
            atom_embeddings = flat_atom_embeddings.reshape(
                int(atom_features.shape[0]),
                int(atom_features.shape[1]),
                self.embedding_dim,
            )
            atom_weights = action_atom_mask.to(dtype=node_features.dtype).unsqueeze(-1)
            action_atom_embeddings = (atom_embeddings * atom_weights).sum(dim=1) / atom_weights.sum(dim=1).clamp_min(1.0)
            action_embeddings = action_embeddings + action_atom_embeddings
        if action_phase_features is not None:
            if self.phase_embedder is None:
                raise ValueError("action_phase_features require phase_feature_dim")
            if action_phase_features.ndim != 3 or int(action_phase_features.shape[1]) != 5:
                raise ValueError("action_phase_features must have shape [num_anchors, 5, phase_feature_dim]")
            if int(action_phase_features.shape[0]) != self.num_anchors:
                raise ValueError("action_phase_features row count must match num_anchors")
            action_phase_output = self.phase_embedder(action_phase_features)
            action_field_embeddings = self.action_field_projection(
                torch.cat(
                    [
                        action_phase_output.h_fiber_summary,
                        action_phase_output.hysteresis_embedding,
                        action_phase_output.activation_gradient_embedding,
                    ],
                    dim=1,
                )
            )
            action_embeddings = action_embeddings + action_field_embeddings
        anchor_logits = action_context_embeddings.matmul(action_embeddings.transpose(0, 1)) / math.sqrt(
            float(self.embedding_dim)
        )
        if int(forward_action_mask.shape[1]) == self.num_anchors + 1:
            forward_logits = torch.cat([anchor_logits, self.stop_mlp(action_context_embeddings)], dim=1)
        else:
            forward_logits = anchor_logits
        _validate_mask("forward_action_mask", forward_action_mask, forward_logits.shape)
        if int(backward_action_mask.shape[1]) > self.max_backward_actions:
            raise ValueError("backward_action_mask exceeds max_backward_actions")
        raw_backward_logits = self.backward_head(action_context_embeddings)
        backward_logits = raw_backward_logits[:, : int(backward_action_mask.shape[1])]
        _validate_mask("backward_action_mask", backward_action_mask, backward_logits.shape)
        forward_log_probs = masked_log_softmax(forward_logits, forward_action_mask)
        backward_log_probs = masked_log_softmax(backward_logits, backward_action_mask)
        return DualChannelPolicyOutput(
            node_embeddings=node_embeddings,
            electronic_embeddings=electronic_embeddings,
            steric_embeddings=steric_embeddings,
            graph_embeddings=graph_embeddings,
            phase_global_embeddings=phase_global_embeddings,
            exit_embeddings=exit_embeddings,
            action_context_embeddings=action_context_embeddings,
            action_field_embeddings=action_field_embeddings,
            action_base_embeddings=action_base_embeddings,
            action_atom_embeddings=action_atom_embeddings,
            channel_gate=channel_gate,
            forward_logits=forward_logits,
            forward_log_probs=forward_log_probs,
            forward_probs=forward_log_probs.exp(),
            backward_logits=backward_logits,
            backward_log_probs=backward_log_probs,
            backward_probs=backward_log_probs.exp(),
            log_z=self.log_z,
        )


class TrilinearAttentionGFlowNetPolicy(nn.Module):
    """PyG-backed policy over scaffold atoms, synthons, and reaction rules."""

    def __init__(
        self,
        *,
        node_feature_dim: int,
        synthon_embeddings: Tensor,
        num_reaction_rules: int,
        hidden_dim: int = 64,
        embedding_dim: int | None = None,
        max_backward_actions: int = 8,
    ) -> None:
        super().__init__()
        if synthon_embeddings.ndim != 2:
            raise ValueError("synthon_embeddings must have shape [num_synthons, embedding_dim]")
        if num_reaction_rules < 1:
            raise ValueError("num_reaction_rules must be positive")
        if max_backward_actions < 1:
            raise ValueError("max_backward_actions must be positive")
        resolved_embedding_dim = int(embedding_dim or synthon_embeddings.shape[1])
        if int(synthon_embeddings.shape[1]) != resolved_embedding_dim:
            raise ValueError("synthon_embeddings width must match embedding_dim")
        self.embedding_dim = resolved_embedding_dim
        self.num_synthons = int(synthon_embeddings.shape[0])
        self.num_reaction_rules = num_reaction_rules
        self.max_backward_actions = max_backward_actions
        self.node_input = nn.Sequential(
            nn.Linear(node_feature_dim, hidden_dim),
            nn.SiLU(),
        )
        self._uses_pyg = _PyGGCNConv is not None
        self.message_passing = _PyGGCNConv(hidden_dim, hidden_dim) if self._uses_pyg else nn.Linear(hidden_dim, hidden_dim)
        self.node_projection = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_dim, resolved_embedding_dim),
        )
        self.rule_embeddings = nn.Embedding(num_reaction_rules, resolved_embedding_dim)
        self.backward_head = nn.Sequential(
            nn.Linear(resolved_embedding_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, max_backward_actions),
        )
        self.log_z = nn.Parameter(torch.zeros(()))
        self.register_buffer("synthon_embeddings", synthon_embeddings.detach().clone().to(dtype=torch.float32))

    def embed_nodes(self, node_features: Tensor, edge_index: Tensor) -> Tensor:
        if node_features.ndim != 2:
            raise ValueError("node_features must have shape [num_nodes, node_feature_dim]")
        hidden = self.node_input(node_features)
        if self._uses_pyg:
            hidden = cast(Tensor, self.message_passing(hidden, edge_index))
        else:
            hidden = cast(Tensor, self.message_passing(hidden))
        return cast(Tensor, self.node_projection(hidden))

    def forward(
        self,
        node_features: Tensor,
        edge_index: Tensor,
        batch_index: Tensor,
        forward_action_mask: Tensor,
        backward_action_mask: Tensor,
    ) -> TrilinearPolicyOutput:
        node_embeddings = self.embed_nodes(node_features, edge_index)
        batch_size = int(batch_index.max().item()) + 1 if batch_index.numel() else 0
        if batch_size < 1:
            raise ValueError("batch_index must describe at least one graph")
        graph_embeddings = _scatter_mean(node_embeddings, batch_index, batch_size)
        rule_embeddings = self.rule_embeddings.weight
        node_rule = node_embeddings[:, None, :] * rule_embeddings[None, :, :]
        forward_logits = torch.einsum("nrd,jd->njr", node_rule, self.synthon_embeddings) / math.sqrt(
            float(self.embedding_dim)
        )
        forward_log_probs = _masked_graph_log_softmax(forward_logits, forward_action_mask, batch_index, batch_size)

        if int(backward_action_mask.shape[1]) > self.max_backward_actions:
            raise ValueError("backward_action_mask exceeds max_backward_actions")
        raw_backward_logits = self.backward_head(graph_embeddings)
        backward_logits = raw_backward_logits[:, : int(backward_action_mask.shape[1])]
        _validate_mask("backward_action_mask", backward_action_mask, backward_logits.shape)
        backward_log_probs = masked_log_softmax(backward_logits, backward_action_mask)
        return TrilinearPolicyOutput(
            node_embeddings=node_embeddings,
            graph_embeddings=graph_embeddings,
            forward_logits=forward_logits,
            forward_log_probs=forward_log_probs,
            forward_probs=forward_log_probs.exp(),
            backward_logits=backward_logits,
            backward_log_probs=backward_log_probs,
            backward_probs=backward_log_probs.exp(),
            log_z=self.log_z,
        )


__all__ = [
    "AnchorAttentionGFlowNetPolicy",
    "DualChannelPolicyOutput",
    "FieldConditionedDualChannelGFlowNetPolicy",
    "FiberBundleGFlowNetPolicy",
    "FiberBundlePolicyOutput",
    "OrthogonalMessageOutput",
    "OrthogonalMessagePassing",
    "PhaseResolvedFiberEmbedder",
    "PhaseResolvedFiberOutput",
    "PolicyOutput",
    "TrilinearAttentionGFlowNetPolicy",
    "TrilinearPolicyOutput",
    "masked_log_softmax",
]
