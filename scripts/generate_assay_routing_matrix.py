#!/usr/bin/env python3
"""Route tensor-supported residue findings to wet-lab assay recommendations."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import TypeAlias

import polars as pl


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from prism_dstw.io import write_provenance_parquet


CAMPAIGN_DIR = REPO_ROOT / "campaigns/glp1r_aleniglipron"
N80_DIR = CAMPAIGN_DIR / "integrated_spike_events/n80_full_scale"
DEFAULT_OUTPUT = N80_DIR / "assay_routing_recommendations.parquet"
JsonObject: TypeAlias = dict[str, object]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pathway", type=Path, default=N80_DIR / "translation_pathway_nodes.parquet")
    parser.add_argument("--hysteresis", type=Path, default=N80_DIR / "hysteresis_tensor.parquet")
    parser.add_argument("--mapping", type=Path, default=CAMPAIGN_DIR / "topology/residue_index_mapping_matrix.parquet")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def collect_streaming(lazy_frame: pl.LazyFrame) -> pl.DataFrame:
    return lazy_frame.collect(engine="streaming")


def pathway_routes(pathway: Path) -> pl.LazyFrame:
    base = pl.scan_parquet(pathway).select(
        [
            "condition_id",
            "residue_idx",
            "residue_name",
            "shear_stress",
            "shear_stress_abs_p90",
            "max_burst_motion",
            "wire_score",
        ]
    )
    shear = (
        base.filter(pl.col("shear_stress").abs() > pl.col("shear_stress_abs_p90"))
        .with_columns(
            [
                pl.lit("HDX-MS").alias("recommended_assay"),
                pl.lit("High spatial gradient of structural deformation indicating backbone solvent exposure").alias(
                    "rationale"
                ),
                pl.col("shear_stress").abs().alias("trigger_value"),
                pl.col("shear_stress_abs_p90").alias("trigger_threshold"),
                pl.lit("shear_stress_abs_gt_p90").alias("trigger_rule"),
            ]
        )
    )
    burst = (
        base.filter(pl.col("max_burst_motion") > 1000.0)
        .with_columns(
            [
                pl.lit("BRET_Kinetics").alias("recommended_assay"),
                pl.lit("High kinetic burst region indicating rapid conformational transition").alias("rationale"),
                pl.col("max_burst_motion").alias("trigger_value"),
                pl.lit(1000.0).alias("trigger_threshold"),
                pl.lit("burst_motion_gt_1000").alias("trigger_rule"),
            ]
        )
    )
    return pl.concat([shear, burst], how="diagonal")


def hysteresis_routes(hysteresis: Path, mapping: Path) -> pl.LazyFrame:
    labels = pl.scan_parquet(mapping).select(
        [
            "condition_id",
            pl.col("residue_idx").cast(pl.Int32).alias("primary_residue_idx"),
            pl.col("canonical_residue_label").alias("residue_name"),
        ]
    )
    return (
        pl.scan_parquet(hysteresis)
        .select(
            [
                "condition_id",
                "primary_residue_idx",
                "protocol_group",
                pl.col("cold_hold_spikes").cast(pl.Float64),
                pl.col("cold_return_spikes").cast(pl.Float64),
            ]
        )
        .with_columns(
            pl.when(pl.col("cold_hold_spikes") > 0.0)
            .then(pl.col("cold_return_spikes") / pl.col("cold_hold_spikes"))
            .otherwise(0.0)
            .alias("hysteresis_ratio")
        )
        .filter(pl.col("hysteresis_ratio") < 0.5)
        .join(labels, on=["condition_id", "primary_residue_idx"], how="left")
        .rename({"primary_residue_idx": "residue_idx"})
        .with_columns(
            [
                pl.lit("Washout_Recovery_Assay").alias("recommended_assay"),
                pl.lit("Persistent recovery impairment signature consistent with receptor-state trapping").alias(
                    "rationale"
                ),
                pl.col("hysteresis_ratio").alias("trigger_value"),
                pl.lit(0.5).alias("trigger_threshold"),
                pl.lit("hysteresis_ratio_lt_0_5").alias("trigger_rule"),
                pl.lit(None, dtype=pl.Float64).alias("shear_stress"),
                pl.lit(None, dtype=pl.Float64).alias("shear_stress_abs_p90"),
                pl.lit(None, dtype=pl.Float64).alias("max_burst_motion"),
                pl.lit(None, dtype=pl.Float64).alias("wire_score"),
            ]
        )
    )


def build_matrix(pathway: Path, hysteresis: Path, mapping: Path) -> pl.LazyFrame:
    columns = [
        "condition_id",
        "residue_idx",
        "residue_name",
        "recommended_assay",
        "rationale",
        "trigger_rule",
        "trigger_value",
        "trigger_threshold",
        "shear_stress",
        "shear_stress_abs_p90",
        "max_burst_motion",
        "wire_score",
    ]
    return (
        pl.concat([pathway_routes(pathway), hysteresis_routes(hysteresis, mapping)], how="diagonal")
        .select(columns)
        .sort(["recommended_assay", "condition_id", "residue_idx"])
    )


def main() -> int:
    args = parse_args()
    output = Path(args.output)
    frame = build_matrix(Path(args.pathway), Path(args.hysteresis), Path(args.mapping))
    row_count = collect_streaming(frame.select(pl.len().alias("n"))).item()
    write_provenance_parquet(
        frame,
        output,
        producer_script=Path(__file__),
        source_parquets=[Path(args.pathway), Path(args.hysteresis), Path(args.mapping)],
        schema_version="assay_routing_recommendations.v1",
        pipeline_stage="assay_routing_recommendations",
        partition_keys=["recommended_assay", "condition_id"],
        ledger_parameters={
            "routing_rules": {
                "HDX-MS": "shear_stress_abs > p90",
                "BRET_Kinetics": "burst_motion > 1000",
                "Washout_Recovery_Assay": "hysteresis_ratio < 0.5",
            }
        },
        ledger_output_value={"rows": int(row_count), "output_path": output.as_posix()},
        repo_root=REPO_ROOT,
    )
    sys.stdout.write(f"wrote {output} rows={row_count}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
