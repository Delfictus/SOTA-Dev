#!/usr/bin/env python3
"""Compute data-derived voxel thresholds for V-space pruning."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import polars as pl


DEFAULT_INPUT = Path(
    "campaigns/glp1r_aleniglipron/integrated_spike_events/n80_full_scale/"
    "signal_grid_variance_channel.parquet"
)
DEFAULT_OUTPUT = Path("campaigns/glp1r_aleniglipron/track_a_generative/voxel_thresholds.json")
DEFAULT_CONDITION = "glp1r_6XOX_WT"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--condition-id", default=DEFAULT_CONDITION)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def scalar_float(frame: pl.LazyFrame, expression: pl.Expr) -> float:
    value = frame.select(expression).collect().item()
    if value is None:
        return 0.0
    return float(value)


def positive_quantile_or_max(frame: pl.LazyFrame, column: str, quantile: float) -> float:
    positive = frame.filter(pl.col(column) > 0)
    count = int(positive.select(pl.len()).collect().item())
    if count > 0:
        return scalar_float(positive, pl.col(column).quantile(quantile, interpolation="nearest"))
    return scalar_float(frame, pl.col(column).max())


def main() -> int:
    args = parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)
    condition_id = str(args.condition_id)
    if not input_path.exists():
        raise FileNotFoundError(input_path)

    nonvoid = (
        pl.scan_parquet(input_path)
        .filter(pl.col("condition_id") == condition_id)
        .filter((pl.col("hit_count_cold_mean") > 0) | (pl.col("hit_count_warm_mean") > 0))
        .with_columns(
            gain_delta=pl.col("hit_count_warm_mean") - pl.col("hit_count_cold_mean"),
            release_delta=pl.col("hit_count_cold_mean") - pl.col("hit_count_warm_mean"),
        )
    )

    nonvoid_count = int(nonvoid.select(pl.len()).collect().item())
    if nonvoid_count == 0:
        raise ValueError(f"no non-void voxels found for {condition_id}")

    thresholds: dict[str, Any] = {
        "condition_id": condition_id,
        "nonvoid_voxel_count": nonvoid_count,
        "cold_p80": scalar_float(nonvoid, pl.col("hit_count_cold_mean").quantile(0.80, interpolation="nearest")),
        "warm_p80": scalar_float(nonvoid, pl.col("hit_count_warm_mean").quantile(0.80, interpolation="nearest")),
        "gain_p90": positive_quantile_or_max(nonvoid, "gain_delta", 0.90),
        "release_p90": positive_quantile_or_max(nonvoid, "release_delta", 0.90),
        "cold_p95": scalar_float(nonvoid, pl.col("hit_count_cold_mean").quantile(0.95, interpolation="nearest")),
        "warm_p95": scalar_float(nonvoid, pl.col("hit_count_warm_mean").quantile(0.95, interpolation="nearest")),
        "release_p95": positive_quantile_or_max(nonvoid, "release_delta", 0.95),
        "source_sha256": sha256_file(input_path),
        "generated_at_utc": datetime.now(UTC).isoformat(),
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(thresholds, indent=2, sort_keys=True) + "\n")
    print(
        "wrote "
        f"{output_path} condition_id={condition_id} nonvoid_voxel_count={nonvoid_count} "
        f"cold_p80={thresholds['cold_p80']:.6f} warm_p80={thresholds['warm_p80']:.6f} "
        f"gain_p90={thresholds['gain_p90']:.6f} release_p90={thresholds['release_p90']:.6f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
