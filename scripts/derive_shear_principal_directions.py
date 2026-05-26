#!/usr/bin/env python3
"""Materialize shear principal directions from the shear stress field."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import polars as pl


PRINCIPAL_COLUMNS = ("principal_x", "principal_y", "principal_z")
PROVENANCE_COLUMN = "shear_direction_provenance"
MAX_DENSE_FINITE_DIFFERENCE_CELLS = 16_000_000


def _principal_valid_mask(frame: pl.DataFrame) -> np.ndarray:
    """Return row-level validity for existing principal direction columns."""

    if not all(column in frame.columns for column in PRINCIPAL_COLUMNS):
        return np.zeros(len(frame), dtype=bool)
    vectors = frame.select([pl.col(column).cast(pl.Float64) for column in PRINCIPAL_COLUMNS])
    principal_x = vectors["principal_x"].to_numpy()
    principal_y = vectors["principal_y"].to_numpy()
    principal_z = vectors["principal_z"].to_numpy()
    norm_sq = principal_x * principal_x + principal_y * principal_y + principal_z * principal_z
    valid: np.ndarray = np.isfinite(principal_x) & np.isfinite(principal_y) & np.isfinite(principal_z) & (norm_sq > 1.0e-24)
    return valid


def _existing_provenance(frame: pl.DataFrame, valid_mask: np.ndarray) -> np.ndarray:
    if PROVENANCE_COLUMN in frame.columns:
        raw = frame[PROVENANCE_COLUMN].to_list()
        return np.array(
            [
                str(value) if valid_mask[index] and value is not None else "L3_FINITE_DIFFERENCE"
                for index, value in enumerate(raw)
            ],
            dtype=object,
        )
    return np.where(valid_mask, "L5_WARP_MATRIX", "L3_FINITE_DIFFERENCE")


def _axis_gradient(dense: np.ndarray, observed: np.ndarray, axis: int) -> np.ndarray:
    forward_values = np.roll(dense, -1, axis=axis)
    backward_values = np.roll(dense, 1, axis=axis)
    forward_seen = np.roll(observed, -1, axis=axis)
    backward_seen = np.roll(observed, 1, axis=axis)

    index = [slice(None), slice(None), slice(None)]
    index[axis] = -1
    forward_seen[tuple(index)] = False
    index[axis] = 0
    backward_seen[tuple(index)] = False

    both = forward_seen & backward_seen
    forward_only = forward_seen & ~backward_seen
    backward_only = backward_seen & ~forward_seen
    gradient = np.zeros_like(dense, dtype=np.float32)
    gradient[both] = (forward_values[both] - backward_values[both]) * 0.5
    gradient[forward_only] = forward_values[forward_only] - dense[forward_only]
    gradient[backward_only] = dense[backward_only] - backward_values[backward_only]
    return gradient


def _normalized_gradients_for_partition(partition: pl.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x_idx = partition["x_idx"].to_numpy().astype(np.int64)
    y_idx = partition["y_idx"].to_numpy().astype(np.int64)
    z_idx = partition["z_idx"].to_numpy().astype(np.int64)
    shear_raw = partition["shear_stress"].to_numpy().astype(np.float32)
    if not bool(np.isfinite(shear_raw).all()):
        raise ValueError("shear_stress contains non-finite values")
    shear = np.abs(shear_raw)
    dims = (int(x_idx.max()) + 1, int(y_idx.max()) + 1, int(z_idx.max()) + 1)
    dense_cell_count = dims[0] * dims[1] * dims[2]
    if dense_cell_count > MAX_DENSE_FINITE_DIFFERENCE_CELLS:
        raise ValueError(
            "shear partition grid too large for dense finite differences: "
            f"dims={dims} cells={dense_cell_count} "
            f"limit={MAX_DENSE_FINITE_DIFFERENCE_CELLS}"
        )
    dense = np.zeros(dims, dtype=np.float32)
    observed = np.zeros(dims, dtype=bool)
    dense[x_idx, y_idx, z_idx] = shear
    observed[x_idx, y_idx, z_idx] = True

    grad_x = _axis_gradient(dense, observed, axis=0) if dims[0] >= 2 else np.zeros_like(dense)
    grad_y = _axis_gradient(dense, observed, axis=1) if dims[1] >= 2 else np.zeros_like(dense)
    grad_z = _axis_gradient(dense, observed, axis=2) if dims[2] >= 2 else np.zeros_like(dense)
    dir_x = grad_x[x_idx, y_idx, z_idx].astype(np.float32)
    dir_y = grad_y[x_idx, y_idx, z_idx].astype(np.float32)
    dir_z = grad_z[x_idx, y_idx, z_idx].astype(np.float32)
    norm = np.sqrt(dir_x * dir_x + dir_y * dir_y + dir_z * dir_z)
    nonzero = norm > 1.0e-12
    dir_x[nonzero] /= norm[nonzero]
    dir_y[nonzero] /= norm[nonzero]
    dir_z[nonzero] /= norm[nonzero]
    dir_x[~nonzero] = 0.0
    dir_y[~nonzero] = 0.0
    dir_z[~nonzero] = 0.0
    return dir_x, dir_y, dir_z


def _derive_fallback_principal_arrays(frame: pl.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    indexed = frame.with_row_index("_row_nr")
    principal_x = np.zeros(len(indexed), dtype=np.float32)
    principal_y = np.zeros(len(indexed), dtype=np.float32)
    principal_z = np.zeros(len(indexed), dtype=np.float32)
    partition_keys = ["condition_id"] if "condition_id" in indexed.columns else None
    partitions = indexed.partition_by(partition_keys, maintain_order=True) if partition_keys else [indexed]

    for partition in partitions:
        row_indices = partition["_row_nr"].to_numpy().astype(np.int64)
        dir_x, dir_y, dir_z = _normalized_gradients_for_partition(partition)
        principal_x[row_indices] = dir_x
        principal_y[row_indices] = dir_y
        principal_z[row_indices] = dir_z
    return principal_x, principal_y, principal_z


def derive_principal_directions(frame: pl.DataFrame) -> pl.DataFrame:
    """Return a shear frame with principal direction and provenance columns."""

    required = {"voxel_idx", "x_idx", "y_idx", "z_idx", "shear_stress"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"shear frame missing required columns: {sorted(missing)}")
    if len(frame) == 0:
        return frame.with_columns(
            pl.Series("principal_x", [], dtype=pl.Float32),
            pl.Series("principal_y", [], dtype=pl.Float32),
            pl.Series("principal_z", [], dtype=pl.Float32),
            pl.Series(PROVENANCE_COLUMN, [], dtype=pl.String),
        )

    valid_existing = _principal_valid_mask(frame)
    if bool(valid_existing.all()) and len(valid_existing) > 0:
        provenance = (
            pl.col(PROVENANCE_COLUMN)
            if PROVENANCE_COLUMN in frame.columns
            else pl.lit("L5_WARP_MATRIX")
        )
        return frame.with_columns(provenance.alias(PROVENANCE_COLUMN))

    fallback_x, fallback_y, fallback_z = _derive_fallback_principal_arrays(frame)
    if all(column in frame.columns for column in PRINCIPAL_COLUMNS):
        existing = frame.select([pl.col(column).cast(pl.Float64) for column in PRINCIPAL_COLUMNS])
        fallback_x = np.where(valid_existing, existing["principal_x"].to_numpy().astype(np.float32), fallback_x)
        fallback_y = np.where(valid_existing, existing["principal_y"].to_numpy().astype(np.float32), fallback_y)
        fallback_z = np.where(valid_existing, existing["principal_z"].to_numpy().astype(np.float32), fallback_z)
    provenance_values = _existing_provenance(frame, valid_existing)

    return frame.with_columns(
        pl.Series("principal_x", fallback_x),
        pl.Series("principal_y", fallback_y),
        pl.Series("principal_z", fallback_z),
        pl.Series(PROVENANCE_COLUMN, provenance_values),
    )


def materialize_shear_principal_directions(input_path: Path, output_path: Path) -> pl.DataFrame:
    """Read a shear parquet, write principal direction columns, and return the frame."""

    frame = pl.read_parquet(input_path)
    enriched = derive_principal_directions(frame)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    enriched.write_parquet(output_path)
    return enriched


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    enriched = materialize_shear_principal_directions(args.input, args.output)
    stats = enriched.select(
        pl.len().alias("rows"),
        pl.col("principal_x").std().alias("principal_x_std"),
        pl.col("principal_y").std().alias("principal_y_std"),
        pl.col("principal_z").std().alias("principal_z_std"),
    ).to_dicts()[0]
    print(
        "shear_principal_directions_materialized "
        f"rows={stats['rows']} "
        f"principal_x_std={float(stats['principal_x_std'] or 0.0):.6f} "
        f"principal_y_std={float(stats['principal_y_std'] or 0.0):.6f} "
        f"principal_z_std={float(stats['principal_z_std'] or 0.0):.6f} "
        f"provenance={enriched[PROVENANCE_COLUMN][0]}"
    )


if __name__ == "__main__":
    main()
