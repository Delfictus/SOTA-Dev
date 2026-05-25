"""Direct signal-grid lookup for product atom fiber tensors."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import polars as pl
import torch
from torch import Tensor


PHASE_WEIGHTS = np.array(
    [
        [1.2, 0.8, 0.5, 1.0, 1.0, 1.0, 1.0, 1.0],
        [0.8, 1.0, 1.2, 1.0, 1.0, 1.0, 1.0, 1.0],
        [0.6, 1.2, 0.8, 1.0, 1.0, 1.0, 1.0, 1.0],
        [0.8, 1.0, 1.2, 1.0, 1.0, 1.0, 1.0, 1.0],
        [1.2, 0.8, 0.5, 1.0, 1.0, 1.0, 1.0, 1.0],
    ],
    dtype=np.float32,
)


@dataclass(frozen=True)
class GridSpec:
    """Cartesian mapping for one thermodynamic signal grid."""

    origin: np.ndarray
    spacing: float
    dims: tuple[int, int, int]


class SignalGridFiberLookup:
    """Map atom coordinates directly to `[5, 8]` signal-grid fiber vectors.

    Scaffold atoms may keep their pre-computed residue phase tensors. New product
    atoms are mapped to the thermodynamic signal grid by voxel index, so the
    fallback is explicit `L0_VOID` rather than a nearest-scaffold broadcast.
    """

    def __init__(
        self,
        signal_grid_path: str | Path,
        grid_config_path: str | Path,
        *,
        condition_id: str = "glp1r_6XOX_WT",
    ) -> None:
        self.signal_grid_path = Path(signal_grid_path)
        self.grid_config_path = Path(grid_config_path)
        self.condition_id = condition_id
        self.spec = self._load_grid_spec(self.grid_config_path, condition_id)
        grid_df = pl.read_parquet(self.signal_grid_path)
        self.grid = self._load_grid_rows(grid_df)

    @staticmethod
    def _load_grid_spec(path: Path, condition_id: str) -> GridSpec:
        payload = json.loads(path.read_text())
        config: Mapping[str, Any]
        if isinstance(payload, dict) and "conditions" in payload:
            conditions = payload.get("conditions")
            if not isinstance(conditions, dict) or condition_id not in conditions:
                raise ValueError(f"grid mapping missing condition_id={condition_id}")
            config = conditions[condition_id]
        else:
            config = payload
        origin_raw = config.get("origin_xyz_angstrom", config.get("origin", config.get("origin_xyz")))
        if origin_raw is None:
            origin_raw = [config["Ox"], config["Oy"], config["Oz"]]
        origin = np.array(origin_raw, dtype=np.float32)
        spacing = float(config.get("spacing_angstrom", config.get("spacing_A", config.get("spacing"))))
        nx = int(config.get("nx", config.get("grid_dim")))
        ny = int(config.get("ny", config.get("grid_dim")))
        nz = int(config.get("nz", config.get("grid_dim")))
        return GridSpec(origin=origin, spacing=spacing, dims=(nx, ny, nz))

    @staticmethod
    def _load_grid_rows(grid_df: pl.DataFrame) -> dict[int, dict[str, object]]:
        rows: dict[int, dict[str, object]] = {}
        for row in grid_df.iter_rows(named=True):
            voxel_idx = int(row["voxel_idx"])
            cold = _float(row.get("hit_count_cold_mean"))
            warm = _float(row.get("hit_count_warm_mean"))
            delta = _float(row.get("hit_count_delta"), warm - cold)
            classification = str(
                row.get(
                    "variance_classification",
                    row.get("variance_class", row.get("consensus_raw_variance_class", "void")),
                )
            )
            rows[voxel_idx] = {
                "cold_mean": cold,
                "warm_mean": warm,
                "delta": delta,
                "classification": classification,
                "consensus_bonus": _float(row.get("consensus_complement_bonus")),
            }
        return rows

    def xyz_to_voxel(self, xyz: np.ndarray) -> int | None:
        """Map Cartesian coordinates to a signal-grid voxel index."""

        ijk = np.floor((xyz.astype(np.float32) - self.spec.origin) / self.spec.spacing).astype(int)
        nx, ny, nz = self.spec.dims
        if not (0 <= ijk[0] < nx and 0 <= ijk[1] < ny and 0 <= ijk[2] < nz):
            return None
        return int(ijk[0] + ijk[1] * nx + ijk[2] * nx * ny)

    def lookup_atom_fiber(self, xyz: np.ndarray) -> np.ndarray:
        """Return a `[5, 8]` fiber vector for one atom coordinate."""

        voxel_idx = self.xyz_to_voxel(xyz)
        if voxel_idx is None or voxel_idx not in self.grid:
            return _void_fiber()
        voxel = self.grid[voxel_idx]
        class_vec = _classification_one_hot(str(voxel["classification"]))
        base_vec = np.array(
            [
                _float(voxel["cold_mean"]),
                _float(voxel["warm_mean"]),
                _float(voxel["delta"]),
                *class_vec,
                _float(voxel["consensus_bonus"]),
            ],
            dtype=np.float32,
        )
        return np.tile(base_vec, (5, 1)) * PHASE_WEIGHTS

    def lookup_product_fiber(
        self,
        scaffold_phase: Tensor,
        product_xyz: np.ndarray,
        *,
        n_scaffold: int,
    ) -> Tensor:
        """Return `[N_product, 5, 8]` fiber data for scaffold plus product atoms."""

        if scaffold_phase.ndim != 3 or int(scaffold_phase.shape[1]) != 5:
            raise ValueError("scaffold_phase must have shape [N_scaffold, 5, D]")
        if int(scaffold_phase.shape[2]) != 8:
            raise ValueError("scaffold_phase feature dimension must be 8")
        if product_xyz.ndim != 2 or int(product_xyz.shape[1]) != 3:
            raise ValueError("product_xyz must have shape [N_product, 3]")
        if n_scaffold < 0 or n_scaffold > int(product_xyz.shape[0]):
            raise ValueError("n_scaffold must be within product atom count")
        if int(scaffold_phase.shape[0]) < n_scaffold:
            raise ValueError("scaffold_phase does not contain n_scaffold rows")

        product_fiber = torch.zeros((int(product_xyz.shape[0]), 5, 8), dtype=torch.float32)
        product_fiber[:n_scaffold] = scaffold_phase[:n_scaffold].detach().to(dtype=torch.float32)
        for atom_index in range(n_scaffold, int(product_xyz.shape[0])):
            product_fiber[atom_index] = torch.from_numpy(self.lookup_atom_fiber(product_xyz[atom_index]))
        return product_fiber


def _classification_one_hot(classification: str) -> list[float]:
    normalized = classification.lower()
    if normalized == "stable_occupied":
        return [1.0, 0.0, 0.0, 0.0]
    if normalized == "thermally_destabilized":
        return [0.0, 1.0, 0.0, 0.0]
    if normalized == "thermally_activated":
        return [0.0, 0.0, 1.0, 0.0]
    if normalized == "variant_disputed":
        return [0.5, 0.5, 0.0, 0.0]
    return [0.0, 0.0, 0.0, 1.0]


def _void_fiber() -> np.ndarray:
    fiber = np.zeros((5, 8), dtype=np.float32)
    fiber[:, 6] = 1.0
    return fiber


def _float(value: object, default: float = 0.0) -> float:
    if value is None or isinstance(value, bool):
        return default
    if isinstance(value, int | float | str):
        parsed = float(value)
        return parsed if np.isfinite(parsed) else default
    return default


__all__ = ["GridSpec", "SignalGridFiberLookup"]
