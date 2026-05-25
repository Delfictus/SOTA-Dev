#!/usr/bin/env python3
"""Validate materialized observed variant signal grids against the WT N80 condition."""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import polars as pl


REPO_ROOT = Path(__file__).resolve().parents[1]
CAMPAIGN_DIR = REPO_ROOT / "campaigns/glp1r_aleniglipron"
TRACK_A = CAMPAIGN_DIR / "track_a_generative"
N80_DIR = CAMPAIGN_DIR / "integrated_spike_events/n80_full_scale"
DEFAULT_WT_GRID = N80_DIR / "signal_grid_variance_channel.parquet"
DEFAULT_REPORT = TRACK_A / "variant_grid_validation_report.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wt-grid", type=Path, default=DEFAULT_WT_GRID)
    parser.add_argument("--wt-condition", type=str, default="glp1r_6XOX_WT")
    parser.add_argument("--variant-grid", type=Path, action="append", default=None)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    variant_grids = args.variant_grid or [
        N80_DIR / "signal_grid_variance_channel_A316T.parquet",
        N80_DIR / "signal_grid_variance_channel_T149M.parquet",
    ]
    wt_frame = load_condition_frame(Path(args.wt_grid), str(args.wt_condition))
    wt_schema = {name: str(dtype) for name, dtype in zip(wt_frame.columns, wt_frame.dtypes, strict=True)}
    reports: list[dict[str, Any]] = []
    for raw_path in variant_grids:
        path = Path(raw_path)
        frame = load_condition_frame(path, infer_condition_id(path))
        schema = {name: str(dtype) for name, dtype in zip(frame.columns, frame.dtypes, strict=True)}
        joined = wt_frame.select(["voxel_idx", "variance_class"]).join(
            frame.select(["voxel_idx", "variance_class"]),
            on="voxel_idx",
            how="inner",
            suffix="_variant",
        )
        diff_rows = joined.filter(pl.col("variance_class") != pl.col("variance_class_variant")).height
        nonvoid = frame.filter(pl.col("variance_class") != "void").height if "variance_class" in frame.columns else 0
        report = {
            "variant_grid": path.as_posix(),
            "condition_id": infer_condition_id(path),
            "rows": frame.height,
            "wt_rows": wt_frame.height,
            "schema_matches_wt": schema == wt_schema,
            "voxel_min": int_scalar(frame.get_column("voxel_idx").min()),
            "voxel_max": int_scalar(frame.get_column("voxel_idx").max()),
            "nonvoid": nonvoid,
            "differs_from_wt_rows": diff_rows,
            "differs_from_wt_fraction": diff_rows / max(joined.height, 1),
            "validation_status": (
                "PASS"
                if frame.height == wt_frame.height
                and schema == wt_schema
                and nonvoid > 0
                and diff_rows > 0
                else "FAIL"
            ),
            "provenance": "OBSERVED_EMBEDDED_N80_SIGNAL_GRID",
        }
        reports.append(report)
        print(
            "variant_grid_validated "
            f"path={path} status={report['validation_status']} rows={frame.height} "
            f"nonvoid={nonvoid} differs_from_wt={diff_rows}"
        )
    payload = {
        "schema_version": "PRISM.variant_grid_validation.v1",
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "wt_grid": Path(args.wt_grid).as_posix(),
        "wt_condition": str(args.wt_condition),
        "variants": reports,
        "overall_status": "PASS" if all(row["validation_status"] == "PASS" for row in reports) else "FAIL",
    }
    atomic_write_json(Path(args.report), payload)
    if payload["overall_status"] != "PASS":
        return 1
    return 0


def infer_condition_id(path: Path) -> str:
    schema = pl.scan_parquet(path).collect_schema()
    if "condition_id" not in schema.names():
        stem = path.stem.removeprefix("signal_grid_variance_channel_")
        return f"glp1r_6XOX_{stem}"
    values = (
        pl.scan_parquet(path)
        .select(pl.col("condition_id").unique())
        .collect()
        .get_column("condition_id")
        .to_list()
    )
    if len(values) != 1:
        raise RuntimeError(f"{path} should contain exactly one condition_id, found {values}")
    return str(values[0])


def load_condition_frame(path: Path, condition_id: str) -> pl.DataFrame:
    scan = pl.scan_parquet(path)
    schema = scan.collect_schema()
    if "condition_id" in schema.names():
        scan = scan.filter(pl.col("condition_id") == condition_id)
    frame = scan.collect()
    if frame.height == 0:
        raise RuntimeError(f"{path} has no rows for condition_id={condition_id}")
    return frame


def int_scalar(value: object) -> int:
    if isinstance(value, bool) or value is None:
        raise ValueError(f"expected integer scalar, got {value!r}")
    if isinstance(value, int | float | str):
        return int(value)
    raise ValueError(f"expected integer scalar, got {value!r}")


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp_path.replace(path)


if __name__ == "__main__":
    raise SystemExit(main())
