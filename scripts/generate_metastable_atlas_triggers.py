#!/usr/bin/env python3
"""Generate Phase 2C metastable atlas trigger windows from probabilistic clusters."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import TypeAlias, cast

import polars as pl


REPO_ROOT = Path(__file__).resolve().parents[1]
CAMPAIGN_ID = "glp1r_aleniglipron"
N80_DIR = REPO_ROOT / "campaigns/glp1r_aleniglipron/integrated_spike_events/n80_full_scale"
DEFAULT_CLUSTERS = N80_DIR / "probabilistic_break_clusters.parquet"
DEFAULT_OUTPUT = REPO_ROOT / "campaigns/glp1r_aleniglipron/phase_2c_metastable_atlas_triggers.json"
WINDOW_RADIUS_STEPS = 2

JsonObject: TypeAlias = dict[str, object]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clusters", type=Path, default=DEFAULT_CLUSTERS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--condition",
        action="append",
        dest="conditions",
        help="Optional condition_id filter. May be passed multiple times. Defaults to all conditions in the parquet.",
    )
    return parser.parse_args()


def emit(message: str) -> None:
    sys.stdout.write(message + "\n")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def as_int(value: object, label: str) -> int:
    if isinstance(value, bool):
        raise TypeError(f"{label} must be an integer, got bool")
    if isinstance(value, int | float | str):
        return int(value)
    raise TypeError(f"{label} must be an integer")


def as_float(value: object, label: str) -> float:
    if isinstance(value, bool):
        raise TypeError(f"{label} must be numeric, got bool")
    if isinstance(value, int | float | str):
        return float(value)
    raise TypeError(f"{label} must be numeric")


def as_str(value: object, label: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{label} must be a string")
    return value


def cluster_rows(clusters: Path, conditions: list[str] | None) -> list[dict[str, object]]:
    require_columns = {
        "condition_id",
        "protocol_group",
        "cluster_id",
        "first_ramp_md_step",
        "temporal_overlap_entropy",
        "cluster_confidence",
        "cluster_probability",
        "primary_residue_idx",
        "inter_replicate_stability",
    }
    schema = pl.scan_parquet(clusters).collect_schema()
    missing = sorted(require_columns.difference(schema.names()))
    if missing:
        raise ValueError(f"{clusters} is missing required columns: {missing}")

    frame = pl.scan_parquet(clusters).filter(pl.col("first_ramp_md_step").is_not_null())
    if conditions:
        frame = frame.filter(pl.col("condition_id").is_in(conditions))

    grouped = (
        frame.group_by(["condition_id", "protocol_group", "cluster_id"])
        .agg(
            [
                pl.col("first_ramp_md_step").mean().alias("centroid_md_step"),
                pl.col("temporal_overlap_entropy").mean().alias("temporal_overlap_entropy"),
                pl.col("cluster_confidence").mean().alias("cluster_confidence"),
                pl.col("cluster_probability").mean().alias("mean_cluster_probability"),
                pl.col("primary_residue_idx").n_unique().alias("unique_residue_count"),
                pl.len().alias("cluster_member_count"),
                pl.col("inter_replicate_stability").mean().alias("inter_replicate_stability"),
            ]
        )
        .with_columns(pl.col("centroid_md_step").round(0).cast(pl.Int64).alias("centroid_step"))
        .with_columns(
            [
                (pl.col("centroid_step") - WINDOW_RADIUS_STEPS).clip(0, None).alias("start_step"),
                (pl.col("centroid_step") + WINDOW_RADIUS_STEPS).alias("end_step"),
            ]
        )
        .sort(["condition_id", "protocol_group", "centroid_step", "cluster_id"])
        .collect()
    )
    return cast(list[dict[str, object]], grouped.to_dicts())


def cooperative_cluster_id(condition_id: str, protocol_group: str, cluster_id: int) -> str:
    return f"{condition_id}:{protocol_group}:C{cluster_id:02d}"


def trigger_from_row(row: dict[str, object]) -> JsonObject:
    condition_id = as_str(row["condition_id"], "condition_id")
    protocol_group = as_str(row["protocol_group"], "protocol_group")
    cluster_id = as_int(row["cluster_id"], "cluster_id")
    centroid_step = as_int(row["centroid_step"], "centroid_step")
    start_step = as_int(row["start_step"], "start_step")
    end_step = as_int(row["end_step"], "end_step")
    cooperative_id = cooperative_cluster_id(condition_id, protocol_group, cluster_id)
    entropy = as_float(row["temporal_overlap_entropy"], "temporal_overlap_entropy")
    return {
        "trigger_id": cooperative_id,
        "condition_id": condition_id,
        "protocol_group": protocol_group,
        "Cooperative_Cluster_ID": cooperative_id,
        "cluster_id": cluster_id,
        "centroid_md_step": as_float(row["centroid_md_step"], "centroid_md_step"),
        "centroid_step": centroid_step,
        "temporal_overlap_entropy": entropy,
        "cluster_confidence": as_float(row["cluster_confidence"], "cluster_confidence"),
        "mean_cluster_probability": as_float(row["mean_cluster_probability"], "mean_cluster_probability"),
        "inter_replicate_stability": as_float(row["inter_replicate_stability"], "inter_replicate_stability"),
        "unique_residue_count": as_int(row["unique_residue_count"], "unique_residue_count"),
        "cluster_member_count": as_int(row["cluster_member_count"], "cluster_member_count"),
        "capture_selection": {
            "mode": "cluster_centroid_high_resolution_window",
            "stride": 1,
            "window_radius_steps": WINDOW_RADIUS_STEPS,
            "epistemic_fuzziness_metric": "temporal_overlap_entropy",
        },
        "windows": [
            {
                "start_step": start_step,
                "end_step": end_step,
                "stride": 1,
                "rationale": (
                    f"Metastable atlas capture centered on cooperative break cluster {cooperative_id}; "
                    f"temporal_overlap_entropy={entropy:.6f} documents cluster boundary uncertainty."
                ),
            }
        ],
    }


def build_manifest(clusters: Path, rows: list[dict[str, object]]) -> JsonObject:
    triggers = [trigger_from_row(row) for row in rows]
    return {
        "campaign_id": CAMPAIGN_ID,
        "phase": "phase_2c_metastable_atlas",
        "capture_mode": "targeted_window_reintegration_or_representative_capture",
        "stride": 1,
        "source_parquet": clusters.as_posix(),
        "source_parquet_sha256": sha256_file(clusters),
        "cluster_centroid_definition": "mean(first_ramp_md_step) grouped by condition_id, protocol_group, and cluster_id",
        "window_definition": "[round(centroid) - 2, round(centroid) + 2]",
        "trigger_count": len(triggers),
        "triggers": triggers,
    }


def main() -> int:
    args = parse_args()
    clusters = Path(args.clusters)
    output = Path(args.output)
    if not clusters.exists():
        raise FileNotFoundError(clusters)
    conditions = cast(list[str] | None, args.conditions)
    rows = cluster_rows(clusters, conditions)
    if not rows:
        raise ValueError(f"no cooperative break clusters found in {clusters}")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(build_manifest(clusters, rows), indent=2, sort_keys=False) + "\n")
    emit(f"wrote {output} triggers={len(rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
