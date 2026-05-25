#!/usr/bin/env python3
"""Generate a falsification-framed CRO wet-lab handoff from tensor assay routing."""

from __future__ import annotations

import argparse
from pathlib import Path

import polars as pl


CAMPAIGN_DIR = Path("campaigns/glp1r_aleniglipron")
DEFAULT_INPUT = CAMPAIGN_DIR / "integrated_spike_events/n80_full_scale/assay_routing_recommendations.parquet"
DEFAULT_OUTPUT = CAMPAIGN_DIR / "CRO_WetLab_Action_Plan.parquet"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assay-routing", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def residue_label_expr() -> pl.Expr:
    return pl.concat_str(
        [
            pl.col("residue_name").cast(pl.Utf8),
            pl.lit(":"),
            pl.col("residue_idx").cast(pl.Utf8),
        ]
    )


def hdx_rows(source: pl.LazyFrame) -> pl.LazyFrame:
    peptide_start = pl.max_horizontal(pl.col("residue_idx").cast(pl.Int64) - 5, pl.lit(1))
    peptide_end = pl.col("residue_idx").cast(pl.Int64) + 5
    peptide_window = pl.concat_str([pl.lit("["), peptide_start.cast(pl.Utf8), pl.lit(", "), peptide_end.cast(pl.Utf8), pl.lit("]")])
    return (
        source.filter(pl.col("recommended_assay") == "HDX-MS")
        .filter(pl.col("shear_stress").abs() > pl.col("shear_stress_abs_p90"))
        .select(
            [
                pl.concat_str([pl.lit("HDX-MS:"), pl.col("condition_id"), pl.lit(":"), pl.col("residue_idx").cast(pl.Utf8)]).alias("action_id"),
                pl.lit("PROJECTED").alias("epistemic_class"),
                pl.lit("HDX-MS").alias("assay_category"),
                pl.col("condition_id"),
                pl.col("residue_idx").cast(pl.Int64),
                pl.col("residue_name").cast(pl.Utf8),
                peptide_start.alias("peptide_start_residue_idx"),
                peptide_end.alias("peptide_end_residue_idx"),
                pl.lit("WT-normalized apo and holo GLP-1R peptide-mapping panel.").alias("construct"),
                pl.lit("shear_stress > condition p90").alias("source_trigger_rule"),
                pl.col("shear_stress").abs().cast(pl.Float64).alias("source_trigger_value"),
                pl.col("shear_stress_abs_p90").cast(pl.Float64).alias("source_trigger_threshold"),
                pl.lit("Directional uptake asymmetry between Apo and Holo states.").alias("measurement_readout"),
                pl.concat_str(
                    [
                        pl.lit("Run HDX-MS as a falsification gate for the predicted shear-associated exposure asymmetry; target peptide window "),
                        peptide_window,
                        pl.lit("."),
                    ]
                ).alias("execution_instruction"),
                pl.lit("Falsification Condition: Lack of WT-normalized uptake asymmetry in the targeted peptide window.").alias("falsification_condition"),
                pl.concat_str([pl.lit("Predicted shear-associated exposure asymmetry at "), residue_label_expr()]).alias("claim_at_risk"),
                pl.col("shear_stress").abs().cast(pl.Float64).alias("priority_score"),
            ]
        )
    )


def bret_rows(source: pl.LazyFrame) -> pl.LazyFrame:
    return (
        source.filter(pl.col("max_burst_motion") > 1000.0)
        .select(
            [
                pl.concat_str([pl.lit("BRET:"), pl.col("condition_id"), pl.lit(":"), pl.col("residue_idx").cast(pl.Utf8)]).alias("action_id"),
                pl.lit("PROJECTED").alias("epistemic_class"),
                pl.lit("BRET_Kinetics").alias("assay_category"),
                pl.col("condition_id"),
                pl.col("residue_idx").cast(pl.Int64),
                pl.col("residue_name").cast(pl.Utf8),
                pl.lit(None).cast(pl.Int64).alias("peptide_start_residue_idx"),
                pl.lit(None).cast(pl.Int64).alias("peptide_end_residue_idx"),
                pl.lit("N-terminal SNAP-tag and C-terminal HaloTag GLP-1R constructs with matched donor/acceptor labeling controls.").alias("construct"),
                pl.lit("burst_motion > 1000").alias("source_trigger_rule"),
                pl.col("max_burst_motion").cast(pl.Float64).alias("source_trigger_value"),
                pl.lit(1000.0).alias("source_trigger_threshold"),
                pl.lit("Relative kinetic ordering versus initiation-wave residue controls.").alias("measurement_readout"),
                pl.lit("Run BRET kinetics as a falsification gate for high kinetic burst regions; do not interpret the readout as absolute sub-picosecond timing.").alias("execution_instruction"),
                pl.lit("Falsification Condition: Lack of predicted early-transition delay relative to initiation-wave residues.").alias("falsification_condition"),
                pl.concat_str([pl.lit("Predicted high kinetic burst region at "), residue_label_expr()]).alias("claim_at_risk"),
                pl.col("max_burst_motion").cast(pl.Float64).alias("priority_score"),
            ]
        )
    )


def washout_rows(source: pl.LazyFrame) -> pl.LazyFrame:
    return (
        source.filter(pl.col("recommended_assay") == "Washout_Recovery_Assay")
        .filter(pl.col("condition_id") == "glp1r_6LN2_A316T")
        .group_by("condition_id")
        .agg(
            [
                pl.max("trigger_value").cast(pl.Float64).alias("max_trigger_value"),
                pl.max("trigger_threshold").cast(pl.Float64).alias("max_trigger_threshold"),
            ]
        )
        .select(
            [
                pl.concat_str([pl.lit("WASHOUT:"), pl.col("condition_id")]).alias("action_id"),
                pl.lit("PROJECTED").alias("epistemic_class"),
                pl.lit("Washout_Recovery_Assay").alias("assay_category"),
                pl.col("condition_id"),
                pl.lit(None).cast(pl.Int64).alias("residue_idx"),
                pl.lit("variant_level").alias("residue_name"),
                pl.lit(None).cast(pl.Int64).alias("peptide_start_residue_idx"),
                pl.lit(None).cast(pl.Int64).alias("peptide_end_residue_idx"),
                pl.lit("6LN2_A316T holo and apo matched receptor-state recovery construct panel.").alias("construct"),
                pl.lit("hysteresis_ratio < 0.5").alias("source_trigger_rule"),
                pl.col("max_trigger_value").alias("source_trigger_value"),
                pl.col("max_trigger_threshold").alias("source_trigger_threshold"),
                pl.lit("WT-normalized recovery curve after 4-hour washout.").alias("measurement_readout"),
                pl.lit("Run washout recovery as a falsification gate for a recovery impairment signature consistent with persistent thermodynamic trapping.").alias("execution_instruction"),
                pl.lit("Falsification Condition: No WT-normalized recovery impairment after 4-hour washout relative to matched WT control.").alias("falsification_condition"),
                pl.lit("Predicted persistent recovery impairment signature in 6LN2_A316T.").alias("claim_at_risk"),
                pl.col("max_trigger_value").alias("priority_score"),
            ]
        )
    )


def build_handoff(input_path: Path, output_path: Path) -> pl.DataFrame:
    if not input_path.exists():
        raise FileNotFoundError(f"assay routing parquet not found: {input_path}")
    source = pl.scan_parquet(input_path)
    combined = pl.concat([hdx_rows(source), bret_rows(source), washout_rows(source)], how="vertical_relaxed").sort(
        ["assay_category", "priority_score"],
        descending=[False, True],
        nulls_last=True,
    )
    frame = combined.collect(engine="streaming")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame.write_parquet(output_path)
    return frame


def main() -> None:
    args = parse_args()
    frame = build_handoff(args.assay_routing, args.output)
    counts_frame = frame.group_by("assay_category").len()
    counts = {str(row["assay_category"]): int(row["len"]) for row in counts_frame.to_dicts()}
    print(f"wrote {args.output} rows={frame.height} assay_counts={counts}")


if __name__ == "__main__":
    main()
