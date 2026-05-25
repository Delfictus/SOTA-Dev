"""Sinkhorn graph-distance regularization for DTSG deep kernel learning."""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch
from torch import Tensor

from prism_dstw.hierarchical_bayes.mpnn_embedding import DTSGGraphTensors


def _uniform_log_weights(size: int, *, dtype: torch.dtype, device: torch.device) -> Tensor:
    return torch.full(
        (size,),
        -math.log(float(size)),
        dtype=dtype,
        device=device,
    )


def sinkhorn_transport_cost(
    source: Tensor,
    target: Tensor,
    *,
    epsilon: float = 0.08,
    iterations: int = 24,
) -> Tensor:
    """Compute entropy-regularized OT cost between two empirical measures."""

    if source.ndim != 2 or target.ndim != 2:
        raise ValueError("source and target must be rank-2 tensors")
    if source.shape[1] != target.shape[1]:
        raise ValueError("source and target feature dimensions must match")
    if source.shape[0] == 0 or target.shape[0] == 0:
        raise ValueError("source and target must contain at least one support point")
    if epsilon <= 0.0:
        raise ValueError("epsilon must be positive")
    if iterations <= 0:
        raise ValueError("iterations must be positive")

    cost = torch.cdist(source, target, p=2.0)
    log_kernel = -cost / epsilon
    log_source_mass = _uniform_log_weights(source.shape[0], dtype=source.dtype, device=source.device)
    log_target_mass = _uniform_log_weights(target.shape[0], dtype=target.dtype, device=target.device)
    log_u = torch.zeros_like(log_source_mass)
    log_v = torch.zeros_like(log_target_mass)

    for _ in range(iterations):
        log_u = log_source_mass - torch.logsumexp(log_kernel + log_v.view(1, -1), dim=1)
        log_v = log_target_mass - torch.logsumexp(log_kernel.transpose(0, 1) + log_u.view(1, -1), dim=1)

    transport_plan = torch.exp(log_u.view(-1, 1) + log_kernel + log_v.view(1, -1))
    return (transport_plan * cost).sum()


def sinkhorn_divergence(
    source: Tensor,
    target: Tensor,
    *,
    epsilon: float = 0.08,
    iterations: int = 24,
) -> Tensor:
    """Compute Sinkhorn divergence outside the GP covariance path."""

    cross_cost = sinkhorn_transport_cost(
        source,
        target,
        epsilon=epsilon,
        iterations=iterations,
    )
    source_self_cost = sinkhorn_transport_cost(
        source,
        source,
        epsilon=epsilon,
        iterations=iterations,
    )
    target_self_cost = sinkhorn_transport_cost(
        target,
        target,
        epsilon=epsilon,
        iterations=iterations,
    )
    divergence = cross_cost - (0.5 * source_self_cost) - (0.5 * target_self_cost)
    return divergence.clamp_min(0.0)


def graph_measure(graph: DTSGGraphTensors) -> Tensor:
    """Return the physical graph support used for Sinkhorn regularization."""

    if graph.edge_tensors is not None:
        return graph.edge_tensors
    return graph.node_features


def pairwise_sinkhorn_divergence(
    graphs: Sequence[DTSGGraphTensors],
    *,
    epsilon: float = 0.08,
    iterations: int = 24,
) -> Tensor:
    """Build a differentiable pairwise graph-distance matrix for a graph batch."""

    if len(graphs) < 2:
        raise ValueError("at least two graphs are required for pairwise regularization")

    measures = [graph_measure(graph) for graph in graphs]
    first_measure = measures[0]
    distance_matrix = first_measure.new_zeros((len(measures), len(measures)))

    for left_index, left_measure in enumerate(measures):
        for right_index in range(left_index + 1, len(measures)):
            right_measure = measures[right_index]
            divergence = sinkhorn_divergence(
                left_measure,
                right_measure,
                epsilon=epsilon,
                iterations=iterations,
            )
            distance_matrix[left_index, right_index] = divergence
            distance_matrix[right_index, left_index] = divergence

    return distance_matrix


def wasserstein_metric_regularization(
    latent_z: Tensor,
    graphs: Sequence[DTSGGraphTensors],
    *,
    regularization_lambda: float = 0.05,
    epsilon: float = 0.08,
    iterations: int = 24,
) -> Tensor:
    """Penalize latent graph distances that disagree with physical graph distances."""

    if latent_z.ndim != 2:
        raise ValueError("latent_z must be a rank-2 tensor")
    if latent_z.shape[0] != len(graphs):
        raise ValueError("latent_z rows must match the number of graphs")
    if regularization_lambda < 0.0:
        raise ValueError("regularization_lambda must be non-negative")

    physical_distances = pairwise_sinkhorn_divergence(
        graphs,
        epsilon=epsilon,
        iterations=iterations,
    )
    latent_distances = torch.cdist(latent_z, latent_z, p=2.0)
    pair_mask = torch.triu(
        torch.ones(
            (latent_z.shape[0], latent_z.shape[0]),
            dtype=torch.bool,
            device=latent_z.device,
        ),
        diagonal=1,
    )
    residual = latent_distances[pair_mask] - physical_distances[pair_mask]
    return latent_z.new_tensor(regularization_lambda) * residual.square().mean()
