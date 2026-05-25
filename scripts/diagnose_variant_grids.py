#!/usr/bin/env python3
"""Diagnose variant grid availability, schema, and projection collapse."""

from __future__ import annotations

import argparse
import json
import math
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, TypeAlias

import polars as pl

from lock_mask_forensics_utils import GridSpec, load_grid_spec, voxel_index_for_coordinate


REPO_ROOT = Path(__file__).resolve().parents[1]
TRACK_A = REPO_ROOT / "campaigns/glp1r_aleniglipron/track_a_generative"
DEFAULT_GRID_MAPPING = REPO_ROOT / "campaigns/glp1r_aleniglipron/track_0_manual_emulation/grid_coordinate_mapping.json"
DEFAULT_SIGNAL_GRID = (
    REPO_ROOT / "campaigns/glp1r_aleniglipron/integrated_spike_events/n80_full_scale/signal_grid_variance_channel.parquet"
)
DEFAULT_CANDIDATES = TRACK_A / "gflownet_top_100_candidates.parquet"
DEFAULT_SURVIVORS = TRACK_A / "vspace_survivors_full_scale.parquet"
DEFAULT_WT_CONDITION = "glp1r_6XOX_WT"
DEFAULT_VARIANTS = ("A316T:glp1r_6XOX_A316T", "T149M:glp1r_6XOX_T149M")
DEFAULT_OUTPUT = TRACK_A / "variant_grid_diagnosis.json"

JsonObject: TypeAlias = dict[str, Any]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--grid-mapping", type=Path, default=DEFAULT_GRID_MAPPING)
    parser.add_argument("--signal-grid", type=Path, default=DEFAULT_SIGNAL_GRID)
    parser.add_argument("--candidates", type=Path, default=DEFAULT_CANDIDATES)
    parser.add_argument("--survivors", type=Path, default=DEFAULT_SURVIVORS)
    parser.add_argument("--wt-condition", type=str, default=DEFAULT_WT_CONDITION)
    parser.add_argument("--variant", action="append", default=None, help="NAME:condition_id")
    parser.add_argument("--sample-size", type=int, default=25)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def atomic_write_json(path: Path, payload: JsonObject) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    tmp_path.replace(path)


def parse_variants(values: list[str] | None) -> dict[str, str]:
    selected = values or list(DEFAULT_VARIANTS)
    parsed: dict[str, str] = {}
    for value in selected:
        if ":" not in value:
            raise ValueError(f"variant must be NAME:condition_id, got {value!r}")
        name, condition_id = value.split(":", 1)
        parsed[name] = condition_id
    return parsed


def load_field(path: Path, condition_id: str) -> dict[int, tuple[str, float, float]]:
    frame = (
        pl.scan_parquet(path)
        .filter(pl.col("condition_id") == condition_id)
        .select(["voxel_idx", "variance_class", "hit_count_cold_mean", "hit_count_warm_mean"])
        .collect()
    )
    return {
        int(row["voxel_idx"]): (
            str(row["variance_class"]),
            float(row["hit_count_cold_mean"]),
            float(row["hit_count_warm_mean"]),
        )
        for row in frame.to_dicts()
    }


def int_stat(value: object, label: str) -> int:
    if isinstance(value, bool) or value is None:
        raise ValueError(f"{label} must be an integer-compatible value")
    if isinstance(value, int | float | str):
        return int(value)
    raise ValueError(f"{label} must be an integer-compatible value")


def score_coordinates(
    coordinates_json: str, grid: GridSpec, field: dict[int, tuple[str, float, float]]
) -> tuple[float, int, int]:
    coords_raw = json.loads(coordinates_json)
    complement = 0.0
    clash = 0.0
    scored = 0
    out_of_bounds = 0
    for item in coords_raw:
        if not isinstance(item, list) or len(item) < 3:
            continue
        coord = (float(item[0]), float(item[1]), float(item[2]))
        voxel_idx = voxel_index_for_coordinate(coord, grid)
        if voxel_idx is None:
            out_of_bounds += 1
            continue
        voxel = field.get(voxel_idx)
        if voxel is None:
            continue
        scored += 1
        variance_class, cold_mean, warm_mean = voxel
        if variance_class == "stable_occupied":
            clash += 1.0
        elif variance_class == "thermally_destabilized" or warm_mean > cold_mean:
            clash += 0.5
        elif variance_class == "thermally_activated" or cold_mean > warm_mean:
            complement += 1.0
    return max(complement - clash, 1.0e-8), scored, out_of_bounds


def condition_summary(signal_grid: Path, condition_id: str) -> JsonObject:
    frame = (
        pl.scan_parquet(signal_grid)
        .filter(pl.col("condition_id") == condition_id)
        .select(["voxel_idx", "variance_class"])
        .collect()
    )
    if frame.height == 0:
        return {"exists": False, "rows": 0, "nonvoid": 0, "activated": 0}
    nonvoid = frame.filter(pl.col("variance_class") != "void").height
    activated = frame.filter(pl.col("variance_class") == "thermally_activated").height
    return {
        "exists": True,
        "rows": frame.height,
        "nonvoid": nonvoid,
        "activated": activated,
        "voxel_min": int_stat(frame.get_column("voxel_idx").min(), "voxel_min"),
        "voxel_max": int_stat(frame.get_column("voxel_idx").max(), "voxel_max"),
    }


def main() -> int:
    args = parse_args()
    variants = parse_variants(args.variant)
    condition_ids = [str(args.wt_condition), *variants.values()]
    grids = {
        condition_id: load_grid_spec(Path(args.grid_mapping), condition_id) for condition_id in condition_ids
    }
    summaries = {
        condition_id: condition_summary(Path(args.signal_grid), condition_id) for condition_id in condition_ids
    }
    fields = {
        condition_id: load_field(Path(args.signal_grid), condition_id)
        for condition_id, summary in summaries.items()
        if bool(summary["exists"])
    }

    candidates = (
        pl.scan_parquet(Path(args.candidates))
        .select(["canonical_smiles"])
        .head(args.sample_size)
    )
    survivors = (
        pl.scan_parquet(Path(args.survivors))
        .select(["canonical_smiles", "coordinates_json"])
        .unique(subset=["canonical_smiles"], keep="first")
    )
    joined = candidates.join(survivors, on="canonical_smiles", how="left").collect()
    missing_coords = joined.filter(pl.col("coordinates_json").is_null()).height
    if missing_coords > 0:
        raise RuntimeError(f"{missing_coords} sampled candidates are missing coordinates_json")

    projection_report: JsonObject = {}
    wt_scores: list[float] = []
    wt_scored_atoms: list[int] = []
    for row in joined.iter_rows(named=True):
        coords_json = row["coordinates_json"]
        if not isinstance(coords_json, str):
            raise TypeError("coordinates_json must be a string")
        reward, scored, _out_of_bounds = score_coordinates(
            coords_json,
            grids[str(args.wt_condition)],
            fields[str(args.wt_condition)],
        )
        wt_scores.append(reward)
        wt_scored_atoms.append(scored)
    projection_report["WT"] = {
        "sample_size": len(wt_scores),
        "reward_mean": sum(wt_scores) / float(len(wt_scores)) if wt_scores else 0.0,
        "reward_max": max(wt_scores) if wt_scores else 0.0,
        "reward_nonzero": sum(1 for value in wt_scores if value > 0.01),
        "atoms_scored_mean": sum(wt_scored_atoms) / float(len(wt_scored_atoms)) if wt_scored_atoms else 0.0,
    }
    for variant_name, condition_id in variants.items():
        variant_scores: list[float] = []
        variant_scored_atoms: list[int] = []
        for row in joined.iter_rows(named=True):
            coords_json = row["coordinates_json"]
            if not isinstance(coords_json, str):
                raise TypeError("coordinates_json must be a string")
            reward, scored, _out_of_bounds = score_coordinates(
                coords_json,
                grids[condition_id],
                fields[condition_id],
            )
            variant_scores.append(reward)
            variant_scored_atoms.append(scored)
        projection_report[variant_name] = {
            "condition_id": condition_id,
            "reward_mean": sum(variant_scores) / float(len(variant_scores)) if variant_scores else 0.0,
            "reward_max": max(variant_scores) if variant_scores else 0.0,
            "reward_nonzero": sum(1 for value in variant_scores if value > 0.01),
            "atoms_scored_mean": sum(variant_scored_atoms) / float(len(variant_scored_atoms))
            if variant_scored_atoms
            else 0.0,
        }

    collapse = projection_report["WT"]["reward_nonzero"] == 0
    payload: JsonObject = {
        "schema_version": "PRISM.variant_grid_diagnosis.v1",
        "epistemic_class": "DERIVED",
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "wt_condition": str(args.wt_condition),
        "variants": variants,
        "condition_summaries": summaries,
        "grid_metadata": {
            condition_id: {
                "origin_xyz_angstrom": list(grid.origin),
                "spacing_angstrom": grid.spacing,
                "dims_xyz": [grid.nx, grid.ny, grid.nz],
            }
            for condition_id, grid in grids.items()
        },
        "projection_report": projection_report,
        "determination": (
            "variant_grids_present_but_projection_collapsed"
            if collapse
            else "variant_grids_present_and_projection_nonzero"
        ),
        "notes": [
            "Variant conditions exist inside the main signal-grid parquet.",
            "If WT reward_nonzero is zero under this projector, PGx resilience cannot be interpreted on the current scale.",
        ],
    }
    atomic_write_json(Path(args.output), payload)
    for variant_name, condition_id in variants.items():
        summary = summaries[condition_id]
        print(
            "variant_grid_check "
            f"variant={variant_name} exists={summary['exists']} rows={summary['rows']} "
            f"nonvoid={summary['nonvoid']} activated={summary['activated']}"
        )
    print(
        "variant_projection_diagnosis "
        f"wt_reward_nonzero={projection_report['WT']['reward_nonzero']} "
        f"wt_reward_mean={projection_report['WT']['reward_mean']:.6f} "
        f"collapse={collapse} output={args.output}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
