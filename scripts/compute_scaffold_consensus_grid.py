#!/usr/bin/env python3
"""Compute a scaffold-consensus thermodynamic grid from scaffold-bound grids."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import polars as pl


REPO_ROOT = Path(__file__).resolve().parents[1]
TRACK_A = REPO_ROOT / "campaigns/glp1r_aleniglipron/track_a_generative"
DEFAULT_MANIFEST = TRACK_A / "scaffold_bound/scaffold_bound_grid_manifest.json"
DEFAULT_OUTPUT = TRACK_A / "signal_grid_scaffold_consensus.parquet"
DEFAULT_REPORT = TRACK_A / "scaffold_bound/scaffold_consensus_grid_report.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    return parser.parse_args()


def resolve_path(path: Path) -> Path:
    if path.is_absolute() or path.exists():
        return path
    return REPO_ROOT / path


def load_manifest(path: Path) -> dict[str, Path]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    scaffolds = payload.get("scaffolds", [])
    if not isinstance(scaffolds, list):
        raise ValueError(f"{path} missing scaffolds list")
    grids: dict[str, Path] = {}
    for item in scaffolds:
        if not isinstance(item, dict):
            raise ValueError(f"{path} contains a non-object scaffold entry")
        scaffold = str(item["scaffold_id"]).upper()
        if scaffold in grids:
            raise ValueError(f"duplicate scaffold_id in manifest: {scaffold}")
        grids[scaffold] = resolve_path(Path(str(item["grid_path"])))
    if len(grids) < 2:
        raise ValueError("scaffold consensus requires at least two scaffold grids")
    return grids


def class_column(frame: pl.DataFrame) -> str:
    if "variance_classification" in frame.columns:
        return "variance_classification"
    if "variance_class" in frame.columns:
        return "variance_class"
    raise ValueError("scaffold grid missing variance class column")


def consensus_for(classes: list[str]) -> tuple[str, str, int, float, float]:
    counts = Counter(classes)
    class_name, support = counts.most_common(1)[0]
    if len(counts) == 1:
        consensus_type = "SCAFFOLD_INVARIANT"
    elif support > len(classes) / 2:
        consensus_type = "SCAFFOLD_MAJORITY"
    else:
        consensus_type = "SCAFFOLD_DISPUTED"
        class_name = "scaffold_disputed"
    strength = support / max(len(classes), 1)
    bonus = 0.0
    if class_name == "thermally_activated":
        bonus = 3.0 if consensus_type == "SCAFFOLD_INVARIANT" else 1.5 if consensus_type == "SCAFFOLD_MAJORITY" else 0.0
    return class_name, consensus_type, support, strength, bonus


def require_finite_column(frame: pl.DataFrame, column: str, path: Path) -> None:
    invalid = frame.filter(~pl.col(column).is_finite()).height
    if invalid > 0:
        raise ValueError(f"{path} column {column} contains {invalid} non-finite values")
    nulls = int(frame.get_column(column).null_count())
    if nulls > 0:
        raise ValueError(f"{path} column {column} contains {nulls} null values")


def main() -> int:
    args = parse_args()
    grid_paths = load_manifest(resolve_path(Path(args.manifest)))
    scaffold_frames: dict[str, pl.DataFrame] = {}
    for scaffold, path in grid_paths.items():
        frame = pl.read_parquet(path)
        class_col = class_column(frame)
        required = {
            "voxel_idx",
            "x_idx",
            "y_idx",
            "z_idx",
            "hit_count_cold_mean",
            "hit_count_warm_mean",
            class_col,
        }
        missing = sorted(required.difference(frame.columns))
        if missing:
            raise ValueError(f"{path} missing required columns: {missing}")
        if frame.height == 0:
            raise ValueError(f"{path} contains zero rows")
        if frame.get_column("voxel_idx").n_unique() != frame.height:
            raise ValueError(f"{path} contains duplicate voxel_idx rows")
        require_finite_column(frame, "hit_count_cold_mean", path)
        require_finite_column(frame, "hit_count_warm_mean", path)
        scaffold_frames[scaffold] = frame.select(
            "voxel_idx",
            "x_idx",
            "y_idx",
            "z_idx",
            "hit_count_cold_mean",
            "hit_count_warm_mean",
            pl.col(class_col).alias(f"class_{scaffold.lower()}"),
        )

    scaffolds = sorted(scaffold_frames)
    base = scaffold_frames[scaffolds[0]]
    base_voxels = set(base.get_column("voxel_idx").to_list())
    for scaffold in scaffolds[1:]:
        scaffold_voxels = set(scaffold_frames[scaffold].get_column("voxel_idx").to_list())
        if scaffold_voxels != base_voxels:
            missing = len(base_voxels.difference(scaffold_voxels))
            extra = len(scaffold_voxels.difference(base_voxels))
            raise ValueError(
                f"scaffold grid voxel_idx mismatch for {scaffold}: "
                f"missing={missing} extra={extra}"
            )
        base = base.join(
            scaffold_frames[scaffold].select("voxel_idx", pl.col(f"class_{scaffold.lower()}")),
            on="voxel_idx",
            how="inner",
        )
    if base.is_empty():
        raise RuntimeError("scaffold consensus join produced zero rows")

    class_cols = [f"class_{scaffold.lower()}" for scaffold in scaffolds]
    consensus_classes: list[str] = []
    consensus_types: list[str] = []
    support_counts: list[int] = []
    strengths: list[float] = []
    bonuses: list[float] = []
    for row in base.select(class_cols).iter_rows(named=True):
        cls, ctype, support, strength, bonus = consensus_for([str(row[col]) for col in class_cols])
        consensus_classes.append(cls)
        consensus_types.append(ctype)
        support_counts.append(support)
        strengths.append(strength)
        bonuses.append(bonus)

    output = base.with_columns(
        pl.lit("glp1r_aleniglipron").alias("campaign_id"),
        pl.lit("glp1r_6XOX_SCAFFOLD_CONSENSUS").alias("condition_id"),
        pl.Series("variance_class", consensus_classes),
        pl.Series("variance_classification", consensus_classes),
        pl.lit("SCAFFOLD_CONSENSUS").alias("scaffold_id"),
        pl.lit("L2_PROJECTED_THERMODYNAMIC_CONSENSUS").alias("scaffold_bound_provenance"),
        pl.Series("consensus_type", consensus_types),
        pl.Series("scaffold_consensus_type", consensus_types),
        pl.Series("scaffold_support_count", support_counts),
        pl.Series("scaffold_consensus_strength", strengths),
        pl.Series("scaffold_consensus_bonus", bonuses),
        pl.Series("consensus_complement_bonus", bonuses),
    )
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = output_path.with_suffix(output_path.suffix + ".tmp")
    output.write_parquet(tmp)
    tmp.replace(output_path)

    type_counts = dict(Counter(consensus_types))
    class_counts = dict(Counter(consensus_classes))
    report: dict[str, Any] = {
        "schema_version": "PRISM.scaffold_consensus_grid.v1",
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "status": "PASS",
        "output": str(output_path),
        "manifest": str(args.manifest),
        "scaffold_grids": {scaffold: str(path) for scaffold, path in grid_paths.items()},
        "metrics": {
            "scaffold_count": len(scaffolds),
            "scaffolds": scaffolds,
            "row_count": output.height,
            "consensus_type_counts": type_counts,
            "consensus_class_counts": class_counts,
            "scaffold_invariant_activated_voxels": sum(
                1 for cls, ctype in zip(consensus_classes, consensus_types, strict=True)
                if cls == "thermally_activated" and ctype == "SCAFFOLD_INVARIANT"
            ),
            "scaffold_majority_activated_voxels": sum(
                1 for cls, ctype in zip(consensus_classes, consensus_types, strict=True)
                if cls == "thermally_activated" and ctype == "SCAFFOLD_MAJORITY"
            ),
            "bonus_positive_voxels": sum(1 for bonus in bonuses if bonus > 0.0),
            "mean_consensus_strength": sum(strengths) / len(strengths),
        },
        "bonus_policy": {
            "SCAFFOLD_INVARIANT thermally_activated": 3.0,
            "SCAFFOLD_MAJORITY thermally_activated": 1.5,
            "SCAFFOLD_DISPUTED": 0.0,
        },
    }
    atomic_write_json(Path(args.report), report)
    print(
        "scaffold_consensus_grid_computed "
        f"rows={output.height} invariant_activated={report['metrics']['scaffold_invariant_activated_voxels']} "
        f"output={output_path}"
    )
    return 0


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(path)


if __name__ == "__main__":
    raise SystemExit(main())
