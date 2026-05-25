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
        if scaffold_phase.ndim != 3:
            raise ValueError("scaffold_phase must have shape [N_scaffold, 5, D]")
        if scaffold_xyz.ndim != 2 or int(scaffold_xyz.shape[1]) != 3:
            raise ValueError("scaffold_xyz must have shape [N_scaffold, 3]")
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

        if step <= 0 or current_product_xyz is None:
            return self.scaffold_phase
        if self.fiber_lookup is None:
            return estimate_product_fiber(
                self.scaffold_phase,
                current_product_xyz,
                n_scaffold=n_scaffold or int(self.scaffold_phase.shape[0]),
            )
        xyz = current_product_xyz.detach().cpu().numpy().astype(np.float32)
        return self.fiber_lookup.lookup_product_fiber(
            self.scaffold_phase,
            xyz,
            n_scaffold=n_scaffold or int(self.scaffold_phase.shape[0]),
        )


def estimate_product_fiber(scaffold_phase: Tensor, product_xyz: Tensor, *, n_scaffold: int) -> Tensor:
    """Shape-preserving fallback used when no signal grid lookup is configured."""

    if product_xyz.ndim != 2 or int(product_xyz.shape[1]) != 3:
        raise ValueError("product_xyz must have shape [N_product, 3]")
    product_fiber = torch.zeros((int(product_xyz.shape[0]), 5, int(scaffold_phase.shape[2])), dtype=torch.float32)
    product_fiber[:n_scaffold] = scaffold_phase[:n_scaffold].detach().to(dtype=torch.float32)
    if int(product_xyz.shape[0]) > n_scaffold:
        product_fiber[n_scaffold:, :, -2] = 1.0
    return product_fiber


__all__ = ["FieldConditioner", "estimate_product_fiber"]
