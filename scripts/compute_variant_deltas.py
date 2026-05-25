#!/usr/bin/env python3
"""Compute mutation-conditioned propagation deltas against matched WT edges."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import polars as pl


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from prism_dstw.io import write_provenance_parquet


CAMPAIGN_DIR = REPO_ROOT / "campaigns/glp1r_aleniglipron"
N80_DIR = CAMPAIGN_DIR / "integrated_spike_events/n80_full_scale"
DEFAULT_OUTPUT = N80_DIR / "variant_propagation_deltas.parquet"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--risk-map", type=Path, default=N80_DIR / "receptor_durability_risk_map.parquet")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def family_expr() -> pl.Expr:
    return (
        pl.when(pl.col("condition_id").str.contains("6LN2"))
        .then(pl.lit("glp1r_6LN2"))
        .when(pl.col("condition_id").str.contains("6XOX"))
        .then(pl.lit("glp1r_6XOX"))
        .when(pl.col("condition_id").str.contains("5VEX"))
        .then(pl.lit("glp1r_5VEX"))
        .when(pl.col("condition_id").str.contains("6X1A"))
        .then(pl.lit("glp1r_6X1A"))
        .otherwise(pl.col("condition_id"))
    )


def mutation_expr() -> pl.Expr:
    return (
        pl.when(pl.col("condition_id").str.ends_with("_WT"))
        .then(pl.lit("WT"))
        .otherwise(pl.col("condition_id").str.extract(r"_([A-Z][0-9]+[A-Z])$", 1))
    )


def deltas_frame(risk_map: Path) -> pl.LazyFrame:
    base = (
        pl.scan_parquet(risk_map)
        .select(
            [
                "condition_id",
                "edge_from_residue",
                "edge_to_residue",
                "edge_class",
                "durability_risk_score_raw",
                "signed_te_mean",
                "validation_status",
                "mechanically_pruned",
            ]
        )
        .with_columns(
            [
                family_expr().alias("condition_family"),
                mutation_expr().alias("mutation_label"),
                pl.concat_str(
                    [
                        pl.col("edge_class"),
                        pl.lit(":"),
                        pl.col("edge_from_residue").cast(pl.String),
                        pl.lit("->"),
                        pl.col("edge_to_residue").cast(pl.String),
                    ]
                ).alias("edge_id"),
            ]
        )
    )
    wt = base.filter(pl.col("mutation_label") == "WT").rename(
        {
            "condition_id": "wt_condition_id",
            "durability_risk_score_raw": "wt_risk",
            "signed_te_mean": "wt_signed_te",
            "validation_status": "wt_validation_status",
            "mechanically_pruned": "wt_mechanically_pruned",
        }
    )
    mutant = base.filter(pl.col("mutation_label") != "WT").rename(
        {
            "condition_id": "mutant_condition_id",
            "durability_risk_score_raw": "mutant_risk",
            "signed_te_mean": "mutant_signed_te",
            "validation_status": "mutant_validation_status",
            "mechanically_pruned": "mutant_mechanically_pruned",
        }
    )
    keys = ["condition_family", "edge_id", "edge_class", "edge_from_residue", "edge_to_residue"]
    return (
        mutant.join(wt, on=keys, how="inner", suffix="_wt")
        .with_columns(
            [
                (pl.col("mutant_risk") - pl.col("wt_risk")).alias("propagation_delta_risk"),
                (pl.col("mutant_signed_te") - pl.col("wt_signed_te")).alias("resilience_shift_signed_te"),
            ]
        )
        .select(
            [
                "condition_family",
                "mutation_label",
                "mutant_condition_id",
                "wt_condition_id",
                "edge_id",
                "edge_class",
                "edge_from_residue",
                "edge_to_residue",
                "mutant_risk",
                "wt_risk",
                "propagation_delta_risk",
                "mutant_signed_te",
                "wt_signed_te",
                "resilience_shift_signed_te",
                "mutant_validation_status",
                "wt_validation_status",
                "mutant_mechanically_pruned",
                "wt_mechanically_pruned",
            ]
        )
        .sort(["condition_family", "mutation_label", "propagation_delta_risk"], descending=[False, False, True])
    )


def main() -> int:
    args = parse_args()
    output = Path(args.output)
    frame = deltas_frame(Path(args.risk_map))
    rows = frame.select(pl.len().alias("n")).collect(engine="streaming").item()
    write_provenance_parquet(
        frame,
        output,
        producer_script=Path(__file__),
        source_parquets=[Path(args.risk_map)],
        schema_version="variant_propagation_deltas.v1",
        pipeline_stage="variant_propagation_deltas",
        partition_keys=["condition_family", "mutation_label"],
        ledger_parameters={"join_key": "condition_family,edge_class,edge_from_residue,edge_to_residue"},
        ledger_output_value={"rows": int(rows), "output_path": output.as_posix()},
        repo_root=REPO_ROOT,
    )
    sys.stdout.write(f"wrote {output} rows={rows}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
