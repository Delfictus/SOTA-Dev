from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import polars as pl
import torch

from prism_dstw.scoring.product_fiber_lookup import SignalGridFiberLookup, ThermodynamicFieldStack


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


def _field_stack(tmp_path: Path) -> ThermodynamicFieldStack:
    signal = tmp_path / "signal_grid.parquet"
    shear = tmp_path / "shear.parquet"
    hyst = tmp_path / "hysteresis.parquet"
    pathway = tmp_path / "pathway.parquet"
    mapping = tmp_path / "grid_mapping.json"
    pl.DataFrame(
        {
            "voxel_idx": [21, 22],
            "hit_count_cold_mean": [2.0, 2.0],
            "hit_count_warm_mean": [4.0, 4.0],
            "variance_class": ["thermally_activated", "stable_occupied"],
            "consensus_complement_bonus": [0.25, 0.0],
        }
    ).write_parquet(signal)
    pl.DataFrame(
        {
            "condition_id": ["glp1r_6XOX_WT", "glp1r_6XOX_WT"],
            "voxel_idx": [21, 22],
            "shear_stress": [3.0, 0.1],
            "shear_stress_max": [4.0, 0.2],
        }
    ).write_parquet(shear)
    pl.DataFrame(
        {
            "condition_id": ["glp1r_6XOX_WT"],
            "primary_residue_idx": [182],
            "cold_hold_spikes": [10],
            "ramp_up_spikes": [20],
            "warm_hold_spikes": [18],
            "ramp_down_spikes": [12],
            "cold_return_spikes": [30],
            "thermal_irreversibility": [0.5],
        }
    ).write_parquet(hyst)
    pl.DataFrame(
        {
            "condition_id": ["glp1r_6XOX_WT"],
            "residue_idx": [182],
            "residue_name": ["ASN182"],
            "voxel_idx": [21],
            "wire_score": [2.0],
            "load_normalized": [0.5],
            "kcc_normalized": [0.25],
        }
    ).write_parquet(pathway)
    mapping.write_text(json.dumps({"origin": [0.0, 0.0, 0.0], "spacing_A": 1.0, "nx": 4, "ny": 4, "nz": 4}))
    return ThermodynamicFieldStack(
        signal,
        mapping,
        shear_stress_path=shear,
        hysteresis_tensor_path=hyst,
        translation_pathway_path=pathway,
    )


def test_full_field_stack_returns_5_by_12_fiber(tmp_path: Path) -> None:
    lookup = _field_stack(tmp_path)
    fiber = lookup.lookup_atom_fiber(np.array([1.2, 1.2, 1.2], dtype=np.float32))
    assert fiber.shape == (5, 12)
    assert float(fiber[:, 8].mean()) > 0.0
    assert float(fiber[:, 9].mean()) > 0.0
    assert float(fiber[:, 10].mean()) > 0.0
    assert float(fiber[:, 11].mean()) < 1.0


def test_full_field_product_preserves_scaffold_and_adds_synthon(tmp_path: Path) -> None:
    lookup = _field_stack(tmp_path)
    scaffold_phase = torch.randn(2, 5, 12)
    product_xyz = np.array([[0.1, 0.1, 0.1], [0.2, 0.2, 0.2], [1.2, 1.2, 1.2]], dtype=np.float32)
    result = lookup.lookup_product_fiber(scaffold_phase, product_xyz, n_scaffold=2)
    assert result.shape == (3, 5, 12)
    assert torch.allclose(result[:2], scaffold_phase)
    stats = lookup.field_stats_for_coordinates(product_xyz, n_scaffold=2)
    assert stats["sigma_shear_mean"] > 0.0
    assert stats["pathway_voxels_occupied"] == 1.0
