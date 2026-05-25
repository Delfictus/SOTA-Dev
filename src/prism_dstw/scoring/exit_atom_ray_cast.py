"""Per-exit-atom steric ray casting against a thermodynamic signal grid."""

from __future__ import annotations

from typing import Protocol

import numpy as np
import torch
from torch import Tensor


class GridLookupProtocol(Protocol):
    grid: dict[int, dict[str, object]]

    def xyz_to_voxel(self, xyz: np.ndarray) -> int | None: ...


class ExitAtomRayCaster:
    """Compute logit adjustments for exit atoms by ray-casting into the grid."""

    def __init__(
        self,
        fiber_lookup: GridLookupProtocol,
        *,
        ray_distances: tuple[float, ...] = (1.5, 2.5, 4.0),
    ) -> None:
        self.fiber_lookup = fiber_lookup
        self.ray_distances = ray_distances

    def compute_exit_masks(
        self,
        exit_atom_positions: np.ndarray,
        molecule_centroid: np.ndarray,
    ) -> Tensor:
        """Return one logit adjustment per exit atom."""

        if exit_atom_positions.ndim != 2 or int(exit_atom_positions.shape[1]) != 3:
            raise ValueError("exit_atom_positions must have shape [N_exits, 3]")
        masks = torch.zeros(int(exit_atom_positions.shape[0]), dtype=torch.float32)
        for index, exit_xyz in enumerate(exit_atom_positions):
            direction = exit_xyz - molecule_centroid
            norm = float(np.linalg.norm(direction))
            if norm < 1.0e-6:
                masks[index] = -2.0
                continue
            direction = direction / norm
            blocked = 0.0
            void = 0
            for distance in self.ray_distances:
                point = exit_xyz + direction * distance
                voxel_idx = self.fiber_lookup.xyz_to_voxel(point)
                if voxel_idx is None or voxel_idx not in self.fiber_lookup.grid:
                    void += 1
                    continue
                classification = str(self.fiber_lookup.grid[voxel_idx].get("classification", "void"))
                if classification == "stable_occupied":
                    blocked += 1.0
                elif classification == "thermally_destabilized":
                    blocked += 0.5
            if blocked >= 2.0:
                masks[index] = -torch.inf
            elif blocked >= 1.0:
                masks[index] = -10.0 * blocked / float(len(self.ray_distances))
            elif void == len(self.ray_distances):
                masks[index] = -2.0
            else:
                masks[index] = 0.0
        return masks


__all__ = ["ExitAtomRayCaster"]
