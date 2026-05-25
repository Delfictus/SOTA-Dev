"""Stable Tobit likelihood for left-censored assay observations."""

from __future__ import annotations

import math
from typing import Any

import gpytorch  # type: ignore[import-untyped]
import torch
from gpytorch.distributions import MultivariateNormal  # type: ignore[import-untyped]
from torch import Tensor, nn
from torch.distributions import Normal
from torch.nn import functional as F


def inverse_softplus(value: float) -> float:
    if value <= 0.0:
        raise ValueError("value must be positive")
    return math.log(math.expm1(value))


class TobitLikelihood(gpytorch.likelihoods.Likelihood):  # type: ignore[misc]
    """Gaussian observation model with left-censoring at the assay floor."""

    def __init__(self, *, assay_floor: float, initial_noise: float = 0.02) -> None:
        super().__init__()
        if initial_noise <= 0.0:
            raise ValueError("initial_noise must be positive")
        self.assay_floor = float(assay_floor)
        self.raw_noise = nn.Parameter(torch.tensor(inverse_softplus(initial_noise), dtype=torch.float32))

    @property
    def noise(self) -> Tensor:
        return F.softplus(self.raw_noise).clamp_min(1.0e-8)

    def forward(self, function_samples: Tensor, *args: Any, **kwargs: Any) -> Normal:
        del args, kwargs
        observation_std = torch.sqrt(self.noise).expand_as(function_samples)
        return Normal(function_samples, observation_std)

    def tobit_log_prob(self, observations: Tensor, mean: Tensor, variance: Tensor) -> Tensor:
        marginal_variance = variance.clamp_min(1.0e-8) + self.noise
        sigma = torch.sqrt(marginal_variance)
        floor = torch.as_tensor(self.assay_floor, dtype=mean.dtype, device=mean.device)

        standardized = (observations - mean) / sigma
        uncensored_log_prob = (
            -0.5 * standardized.square()
            - torch.log(sigma)
            - (0.5 * math.log(2.0 * math.pi))
        )
        censored_z = (floor - mean) / sigma
        censored_log_prob = torch.special.log_ndtr(censored_z)
        censored = observations <= floor
        return torch.where(censored, censored_log_prob, uncensored_log_prob)

    def expected_log_prob(
        self,
        observations: Tensor,
        function_dist: MultivariateNormal,
        *args: Any,
        **kwargs: Any,
    ) -> Tensor:
        del args, kwargs
        return self.tobit_log_prob(observations, function_dist.mean, function_dist.variance)
