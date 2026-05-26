"""Step-aware field conditioning facade for GFlowNet trajectories."""

from __future__ import annotations

import numpy as np
import torch
from torch import Tensor

from prism_dstw.scoring.product_fiber_lookup import SignalGridFiberLookup, ThermodynamicFieldStack


class FieldConditioner:
    """Return scaffold or product field tensors for the current rollout step."""

    def __init__(
        self,
        scaffold_phase: Tensor,
        scaffold_xyz: Tensor,
        *,
        fiber_lookup: SignalGridFiberLookup | ThermodynamicFieldStack | None = None,
    ) -> None:
        if scaffold_phase.ndim != 3 or int(scaffold_phase.shape[1]) != 5:
            raise ValueError("scaffold_phase must have shape [N_scaffold, 5, D]")
        if scaffold_xyz.ndim != 2 or int(scaffold_xyz.shape[1]) != 3:
            raise ValueError("scaffold_xyz must have shape [N_scaffold, 3]")
        if int(scaffold_phase.shape[0]) != int(scaffold_xyz.shape[0]):
            raise ValueError("scaffold_phase and scaffold_xyz must have matching atom counts")
        if not bool(torch.isfinite(scaffold_phase).all().item()):
            raise ValueError("scaffold_phase contains non-finite values")
        if not bool(torch.isfinite(scaffold_xyz).all().item()):
            raise ValueError("scaffold_xyz contains non-finite coordinates")
        self.scaffold_phase = scaffold_phase.detach().clone()
        self.scaffold_xyz = scaffold_xyz.detach().clone()
        self.fiber_lookup = fiber_lookup

    def condition_at_step(
        self,
        step: int,
        current_product_xyz: Tensor | None = None,
        n_scaffold: int | None = None,
    ) -> Tensor:
        """Return field conditioning for a trajectory step."""

        if step < 0:
            raise ValueError("step must be non-negative")
        if step == 0:
            return self.scaffold_phase.clone()
        if current_product_xyz is None:
            raise ValueError("current_product_xyz is required for step-aware product conditioning")
        scaffold_count = int(self.scaffold_phase.shape[0]) if n_scaffold is None else int(n_scaffold)
        if current_product_xyz.ndim != 2 or int(current_product_xyz.shape[1]) != 3:
            raise ValueError("current_product_xyz must have shape [N_product, 3]")
        if not bool(torch.isfinite(current_product_xyz).all().item()):
            raise ValueError("current_product_xyz contains non-finite coordinates")
        if self.fiber_lookup is None:
            return estimate_product_fiber(
                self.scaffold_phase,
                current_product_xyz,
                n_scaffold=scaffold_count,
            )
        xyz = current_product_xyz.detach().cpu().numpy().astype(np.float32)
        return self.fiber_lookup.lookup_product_fiber(
            self.scaffold_phase,
            xyz,
            n_scaffold=scaffold_count,
        )


def estimate_product_fiber(scaffold_phase: Tensor, product_xyz: Tensor, *, n_scaffold: int) -> Tensor:
    """Shape-preserving fallback used when no signal grid lookup is configured."""

    if product_xyz.ndim != 2 or int(product_xyz.shape[1]) != 3:
        raise ValueError("product_xyz must have shape [N_product, 3]")
    if not bool(torch.isfinite(product_xyz).all().item()):
        raise ValueError("product_xyz contains non-finite coordinates")
    if n_scaffold < 0 or n_scaffold > int(product_xyz.shape[0]):
        raise ValueError("n_scaffold must be between 0 and the product atom count")
    if n_scaffold > int(scaffold_phase.shape[0]):
        raise ValueError("n_scaffold cannot exceed scaffold_phase atom count")
    feature_dim = int(scaffold_phase.shape[2])
    product_fiber = torch.zeros((int(product_xyz.shape[0]), 5, feature_dim), dtype=torch.float32)
    product_fiber[:n_scaffold] = scaffold_phase[:n_scaffold].detach().to(dtype=torch.float32)
    if int(product_xyz.shape[0]) > n_scaffold:
        if feature_dim >= 7:
            product_fiber[n_scaffold:, :, 6] = 1.0
        if feature_dim >= 12:
            product_fiber[n_scaffold:, :, 11] = 1.0
    return product_fiber


__all__ = ["FieldConditioner", "estimate_product_fiber"]
