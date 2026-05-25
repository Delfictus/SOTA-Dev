"""Errors-in-variables MPNN embeddings for Chem-Perturbed DTSG graphs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import torch
from torch import Tensor, nn
from torch.distributions import Normal
from torch_geometric.nn import MessagePassing  # type: ignore[import-untyped]


@dataclass(frozen=True)
class DTSGGraphTensors:
    """Typed tensors required for an EiV DTSG forward pass."""

    node_features: Tensor
    edge_index: Tensor
    signed_te_mean: Tensor
    u_pose: Tensor
    edge_tensors: Tensor | None = None


class EiVSignedTEConv(MessagePassing):  # type: ignore[misc]
    """Message passing with SignedTE edge weights sampled under pose uncertainty."""

    def __init__(self, hidden_channels: int) -> None:
        super().__init__(aggr="add")
        self.message_projection = nn.Linear(hidden_channels, hidden_channels, bias=False)
        self.self_projection = nn.Linear(hidden_channels, hidden_channels)
        self.last_sampled_edge_weight: Tensor | None = None

    def sample_edge_weight(self, signed_te_mean: Tensor, u_pose: Tensor) -> Tensor:
        scale = u_pose.clamp_min(1.0e-6)
        sampled = Normal(signed_te_mean, scale).rsample()
        precision_gate = torch.exp(-scale)
        edge_weight = sampled * precision_gate
        self.last_sampled_edge_weight = edge_weight
        return edge_weight

    def forward(
        self,
        x: Tensor,
        edge_index: Tensor,
        signed_te_mean: Tensor,
        u_pose: Tensor,
    ) -> Tensor:
        edge_weight = self.sample_edge_weight(signed_te_mean, u_pose)
        propagated = cast(Tensor, self.propagate(edge_index, x=x, edge_weight=edge_weight))
        self_term = cast(Tensor, self.self_projection(x))
        return self_term + propagated

    def message(self, x_j: Tensor, edge_weight: Tensor) -> Tensor:
        projected = cast(Tensor, self.message_projection(x_j))
        return projected * edge_weight.view(-1, 1)


class EiVMPNNEmbedding(nn.Module):
    """Small PyG MPNN that gates causal messages by sampled SignedTE."""

    def __init__(
        self,
        input_channels: int,
        hidden_channels: int = 16,
        output_channels: int = 4,
    ) -> None:
        super().__init__()
        self.input_projection = nn.Linear(input_channels, hidden_channels)
        self.convolution = EiVSignedTEConv(hidden_channels)
        self.activation = nn.SiLU()
        self.output_projection = nn.Linear(hidden_channels, output_channels)

    def forward(
        self,
        node_features: Tensor,
        edge_index: Tensor,
        signed_te_mean: Tensor,
        u_pose: Tensor,
    ) -> Tensor:
        hidden = cast(Tensor, self.input_projection(node_features))
        hidden = cast(Tensor, self.activation(hidden))
        convolved = self.convolution(
                hidden,
                edge_index,
                signed_te_mean,
                u_pose,
        )
        hidden = cast(Tensor, self.activation(convolved))
        return cast(Tensor, self.output_projection(hidden))

    def embed(self, graph: DTSGGraphTensors) -> Tensor:
        return self.forward(
            graph.node_features,
            graph.edge_index,
            graph.signed_te_mean,
            graph.u_pose,
        )
