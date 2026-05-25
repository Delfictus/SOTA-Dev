"""Wasserstein-regularized deep kernel learning for DTSG assay response."""

from __future__ import annotations

from collections.abc import Sequence
from typing import cast

import gpytorch  # type: ignore[import-untyped]
import torch
from gpytorch.distributions import MultivariateNormal  # type: ignore[import-untyped]
from torch import Tensor

from prism_dstw.hierarchical_bayes.dkl_wasserstein import wasserstein_metric_regularization
from prism_dstw.hierarchical_bayes.mpnn_embedding import DTSGGraphTensors, EiVMPNNEmbedding


class DTSGDeepKernelGP(gpytorch.models.ApproximateGP):  # type: ignore[misc]
    """Approximate GP over EiV-MPNN graph embeddings."""

    def __init__(
        self,
        *,
        node_feature_dim: int,
        latent_dim: int = 4,
        hidden_channels: int = 16,
        num_inducing: int = 6,
    ) -> None:
        if num_inducing < 2:
            raise ValueError("num_inducing must be at least 2")
        inducing_points = torch.linspace(
            0.0,
            1.0,
            steps=num_inducing,
            dtype=torch.float32,
        ).view(num_inducing, 1).repeat(1, latent_dim)
        variational_distribution = gpytorch.variational.CholeskyVariationalDistribution(
            num_inducing_points=num_inducing,
        )
        variational_strategy = gpytorch.variational.VariationalStrategy(
            self,
            inducing_points,
            variational_distribution,
            learn_inducing_locations=True,
        )
        super().__init__(variational_strategy)
        self.feature_extractor = EiVMPNNEmbedding(
            input_channels=node_feature_dim,
            hidden_channels=hidden_channels,
            output_channels=latent_dim,
        )
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())

    def embed_graph(self, graph: DTSGGraphTensors) -> Tensor:
        node_embeddings = self.feature_extractor.embed(graph)
        return node_embeddings.mean(dim=0)

    def embed_graphs(self, graphs: Sequence[DTSGGraphTensors]) -> Tensor:
        if len(graphs) == 0:
            raise ValueError("at least one graph is required")
        return torch.stack([self.embed_graph(graph) for graph in graphs], dim=0)

    def forward(self, latent_z: Tensor) -> MultivariateNormal:
        mean = cast(Tensor, self.mean_module(latent_z))
        covariance = self.covar_module(latent_z)
        return MultivariateNormal(mean, covariance)

    def forward_graphs(self, graphs: Sequence[DTSGGraphTensors]) -> MultivariateNormal:
        return cast(MultivariateNormal, self(self.embed_graphs(graphs)))

    def wasserstein_regularization(
        self,
        graphs: Sequence[DTSGGraphTensors],
        latent_z: Tensor,
        *,
        regularization_lambda: float = 0.05,
        epsilon: float = 0.08,
        iterations: int = 24,
    ) -> Tensor:
        return wasserstein_metric_regularization(
            latent_z,
            graphs,
            regularization_lambda=regularization_lambda,
            epsilon=epsilon,
            iterations=iterations,
        )
