#!/usr/bin/env python3
"""Diagnose and adapt PGx variant grid scoring inputs for tripartite screening."""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import polars as pl


REPO_ROOT = Path(__file__).resolve().parents[1]
TRACK_A = REPO_ROOT / "campaigns/glp1r_aleniglipron/track_a_generative"
GRID_DIR = REPO_ROOT / "campaigns/glp1r_aleniglipron/integrated_spike_events/n80_full_scale"
DEFAULT_WT = GRID_DIR / "signal_grid_variance_channel.parquet"
DEFAULT_A316T = GRID_DIR / "signal_grid_variance_channel_A316T.parquet"
DEFAULT_T149M = GRID_DIR / "signal_grid_variance_channel_T149M.parquet"
DEFAULT_REPORT = TRACK_A / "pgx_scoring_path_diagnosis_v2.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wt-grid", type=Path, default=DEFAULT_WT)
    parser.add_argument("--variant-grid", type=Path, action="append", default=[DEFAULT_A316T, DEFAULT_T149M])
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    wt = inspect_grid(Path(args.wt_grid))
    variants = [inspect_grid(Path(path)) for path in args.variant_grid]
    report = {
        "schema_version": "PRISM.pgx_scoring_path_diagnosis.v2",
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "wt_grid": wt,
        "variant_grids": variants,
        "determination": determine(wt, variants),
        "next_action": (
            "Use corrected lock mask and variant-specific normalization only after the variant grid "
            "coordinate projection returns nonzero WT reference scores."
        ),
    }
    atomic_write_json(Path(args.report), report)
    print(f"pgx_scoring_path_diagnosis_written report={args.report} determination={report['determination']}")
    return 0


def inspect_grid(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {"path": str(path), "exists": False}
    frame = pl.read_parquet(path)
    required = {
        "voxel_idx",
        "variance_classification",
        "hit_count_cold_mean",
        "hit_count_warm_mean",
        "hit_count_delta",
    }
    nonvoid = 0
    activated = 0
    if "variance_classification" in frame.columns:
        nonvoid = frame.filter(pl.col("variance_classification") != "void").height
        activated = frame.filter(pl.col("variance_classification") == "thermally_activated").height
    return {
        "path": str(path),
        "exists": True,
        "rows": frame.height,
        "columns": frame.columns,
        "missing_required": sorted(required.difference(frame.columns)),
        "voxel_idx_min": optional_int(frame.get_column("voxel_idx").min()) if "voxel_idx" in frame.columns else None,
        "voxel_idx_max": optional_int(frame.get_column("voxel_idx").max()) if "voxel_idx" in frame.columns else None,
        "nonvoid": nonvoid,
        "thermally_activated": activated,
    }


def determine(wt: dict[str, Any], variants: list[dict[str, Any]]) -> str:
    if not bool(wt.get("exists")):
        return "blocked_missing_wt_grid"
    missing = [grid["path"] for grid in variants if not bool(grid.get("exists"))]
    if missing:
        return "blocked_missing_variant_grids:" + ",".join(str(path) for path in missing)
    schema_mismatch = [grid["path"] for grid in variants if grid.get("columns") != wt.get("columns")]
    if schema_mismatch:
        return "schema_adapter_required:" + ",".join(str(path) for path in schema_mismatch)
    empty = [grid["path"] for grid in variants if int(grid.get("nonvoid", 0)) == 0]
    if empty:
        return "variant_grid_empty:" + ",".join(str(path) for path in empty)
    return "schemas_match_projection_validation_required"


def optional_int(value: object) -> int | None:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float | str):
        return int(float(value))
    return None


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp_path.replace(path)


if __name__ == "__main__":
    raise SystemExit(main())
