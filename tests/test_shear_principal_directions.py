from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl
import pytest

from prism_dstw.scoring.product_fiber_lookup import ThermodynamicFieldStack
from scripts.derive_shear_principal_directions import derive_principal_directions, materialize_shear_principal_directions
from tests.test_product_fiber_lookup import _field_stack


def _shear_frame() -> pl.DataFrame:
    rows: list[dict[str, object]] = []
    for x_idx in range(4):
        for y_idx in range(4):
            for z_idx in range(4):
                rows.append(
                    {
                        "campaign_id": "test",
                        "condition_id": "glp1r_6XOX_WT",
                        "voxel_idx": x_idx + y_idx * 4 + z_idx * 16,
                        "x_idx": x_idx,
                        "y_idx": y_idx,
                        "z_idx": z_idx,
                        "shear_stress": float(x_idx * x_idx + 2 * y_idx + 3 * z_idx),
                        "shear_stress_max": float(x_idx * x_idx + 2 * y_idx + 3 * z_idx),
                    }
                )
    return pl.DataFrame(rows)


def test_derive_principal_directions_from_finite_differences() -> None:
    enriched = derive_principal_directions(_shear_frame())

    assert {"principal_x", "principal_y", "principal_z", "shear_direction_provenance"}.issubset(enriched.columns)
    assert float(enriched["principal_x"].std() or 0.0) > 0.0
    assert enriched["shear_direction_provenance"].unique().to_list() == ["L3_FINITE_DIFFERENCE"]
    norms = enriched.select(
        (
            pl.col("principal_x") ** 2
            + pl.col("principal_y") ** 2
            + pl.col("principal_z") ** 2
        ).sqrt().alias("norm")
    )
    assert float(norms["norm"].max()) <= 1.0001


def test_materialize_shear_principal_directions_writes_parquet(tmp_path: Path) -> None:
    source = tmp_path / "shear.parquet"
    output = tmp_path / "shear_with_principal.parquet"
    _shear_frame().write_parquet(source)

    materialize_shear_principal_directions(source, output)
    enriched = pl.read_parquet(output)

    assert "principal_x" in enriched.columns
    assert float(enriched["principal_x"].std() or 0.0) > 0.0


def test_singleton_partition_gets_zero_direction() -> None:
    frame = pl.DataFrame(
        {
            "condition_id": ["one"],
            "voxel_idx": [0],
            "x_idx": [0],
            "y_idx": [0],
            "z_idx": [0],
            "shear_stress": [10.0],
        }
    )

    enriched = derive_principal_directions(frame)

    assert float(enriched["principal_x"][0]) == 0.0
    assert float(enriched["principal_y"][0]) == 0.0
    assert float(enriched["principal_z"][0]) == 0.0


def test_sparse_constant_grid_does_not_invent_gradients() -> None:
    frame = pl.DataFrame(
        {
            "condition_id": ["sparse", "sparse"],
            "voxel_idx": [0, 26],
            "x_idx": [0, 2],
            "y_idx": [0, 2],
            "z_idx": [0, 2],
            "shear_stress": [1.0, 1.0],
        }
    )

    enriched = derive_principal_directions(frame)

    assert float(enriched["principal_x"].abs().sum()) == 0.0
    assert float(enriched["principal_y"].abs().sum()) == 0.0
    assert float(enriched["principal_z"].abs().sum()) == 0.0


def test_malformed_huge_sparse_grid_is_rejected() -> None:
    frame = pl.DataFrame(
        {
            "condition_id": ["huge", "huge"],
            "voxel_idx": [0, 1],
            "x_idx": [0, 1_000_000],
            "y_idx": [0, 1_000_000],
            "z_idx": [0, 1_000_000],
            "shear_stress": [1.0, 2.0],
        }
    )

    with pytest.raises(ValueError, match="too large for dense finite differences"):
        derive_principal_directions(frame)


def test_nonfinite_shear_values_are_rejected() -> None:
    frame = pl.DataFrame(
        {
            "condition_id": ["bad", "bad", "bad"],
            "voxel_idx": [0, 1, 2],
            "x_idx": [0, 1, 2],
            "y_idx": [0, 0, 0],
            "z_idx": [0, 0, 0],
            "shear_stress": [1.0, float("nan"), float("inf")],
        }
    )

    with pytest.raises(ValueError, match="non-finite"):
        derive_principal_directions(frame)


def test_existing_constant_l5_direction_is_preserved() -> None:
    frame = _shear_frame().with_columns(
        pl.lit(1.0).alias("principal_x"),
        pl.lit(0.0).alias("principal_y"),
        pl.lit(0.0).alias("principal_z"),
        pl.lit("L5_WARP_MATRIX").alias("shear_direction_provenance"),
    )

    enriched = derive_principal_directions(frame)

    assert enriched["shear_direction_provenance"].unique().to_list() == ["L5_WARP_MATRIX"]
    assert float(enriched["principal_x"].min()) == 1.0
    assert float(enriched["principal_x"].max()) == 1.0


def test_mixed_l5_direction_validity_is_handled_per_row() -> None:
    frame = pl.DataFrame(
        {
            "condition_id": ["mixed", "mixed", "mixed"],
            "voxel_idx": [0, 1, 2],
            "x_idx": [0, 1, 2],
            "y_idx": [0, 0, 0],
            "z_idx": [0, 0, 0],
            "shear_stress": [1.0, 2.0, 4.0],
            "principal_x": [1.0, float("nan"), 0.0],
            "principal_y": [0.0, 0.0, 0.0],
            "principal_z": [0.0, 0.0, 0.0],
            "shear_direction_provenance": ["L5_WARP_MATRIX", "L5_WARP_MATRIX", "L5_WARP_MATRIX"],
        }
    )

    enriched = derive_principal_directions(frame)

    assert float(enriched["principal_x"][0]) == 1.0
    assert enriched["shear_direction_provenance"][0] == "L5_WARP_MATRIX"
    assert np.isfinite(float(enriched["principal_x"][1]))
    assert enriched["shear_direction_provenance"][1] == "L3_FINITE_DIFFERENCE"
    assert enriched["shear_direction_provenance"][2] == "L3_FINITE_DIFFERENCE"


def test_empty_shear_frame_returns_empty_direction_columns() -> None:
    frame = pl.DataFrame(
        {
            "condition_id": [],
            "voxel_idx": [],
            "x_idx": [],
            "y_idx": [],
            "z_idx": [],
            "shear_stress": [],
        },
        schema={
            "condition_id": pl.String,
            "voxel_idx": pl.Int64,
            "x_idx": pl.Int64,
            "y_idx": pl.Int64,
            "z_idx": pl.Int64,
            "shear_stress": pl.Float64,
        },
    )

    enriched = derive_principal_directions(frame)

    assert len(enriched) == 0
    assert {"principal_x", "principal_y", "principal_z", "shear_direction_provenance"}.issubset(enriched.columns)


def test_field_stack_loads_shear_principal_directions(tmp_path: Path) -> None:
    lookup: ThermodynamicFieldStack = _field_stack(tmp_path)
    shear_with_directions = pl.read_parquet(lookup.shear_stress_path).with_columns(
        pl.lit(0.5).alias("principal_x"),
        pl.lit(0.0).alias("principal_y"),
        pl.lit(0.8660254).alias("principal_z"),
        pl.lit("L3_FINITE_DIFFERENCE").alias("shear_direction_provenance"),
    )
    shear_with_directions.write_parquet(lookup.shear_stress_path)
    lookup = ThermodynamicFieldStack(
        lookup.signal_grid_path,
        lookup.grid_config_path,
        shear_stress_path=lookup.shear_stress_path,
        hysteresis_tensor_path=lookup.hysteresis_tensor_path,
        translation_pathway_path=lookup.translation_pathway_path,
    )

    profile = lookup.lookup_atom_profile(lookup.voxel_to_xyz(21))

    assert profile is not None
    assert profile.shear_principal_direction is not None
    assert profile.shear_direction_provenance == "L3_FINITE_DIFFERENCE"
