from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import polars as pl
import torch

from prism_dstw.scoring.product_fiber_lookup import SignalGridFiberLookup


def _lookup(tmp_path: Path) -> SignalGridFiberLookup:
    grid = tmp_path / "signal_grid.parquet"
    mapping = tmp_path / "grid_mapping.json"
    pl.DataFrame(
        {
            "voxel_idx": [21],
            "hit_count_cold_mean": [2.0],
            "hit_count_warm_mean": [3.0],
            "hit_count_delta": [1.0],
            "variance_classification": ["thermally_activated"],
            "consensus_complement_bonus": [0.5],
        }
    ).write_parquet(grid)
    mapping.write_text(json.dumps({"origin": [0.0, 0.0, 0.0], "spacing_A": 1.0, "nx": 4, "ny": 4, "nz": 4}))
    return SignalGridFiberLookup(grid, mapping)


def test_scaffold_atoms_unchanged(tmp_path: Path) -> None:
    lookup = _lookup(tmp_path)
    scaffold_phase = torch.randn(2, 5, 8)
    product_xyz = np.array([[0.1, 0.1, 0.1], [0.2, 0.2, 0.2], [1.2, 1.2, 1.2]], dtype=np.float32)
    result = lookup.lookup_product_fiber(scaffold_phase, product_xyz, n_scaffold=2)
    assert torch.allclose(result[:2], scaffold_phase)


def test_synthon_in_grid_gets_real_data(tmp_path: Path) -> None:
    lookup = _lookup(tmp_path)
    fiber = lookup.lookup_atom_fiber(np.array([1.2, 1.2, 1.2], dtype=np.float32))
    assert float(fiber.sum()) > 0.0
    assert fiber.shape == (5, 8)


def test_synthon_outside_grid_gets_void(tmp_path: Path) -> None:
    lookup = _lookup(tmp_path)
    fiber = lookup.lookup_atom_fiber(np.array([99.0, 99.0, 99.0], dtype=np.float32))
    assert float(fiber[:, 6].sum()) == 5.0
