#!/usr/bin/env python3
"""Audit V-space survivor Pi scores for voxel-mapping false positives."""

from __future__ import annotations

from pathlib import Path

import polars as pl


SURVIVORS = Path("campaigns/glp1r_aleniglipron/track_a_generative/vspace_survivors.parquet")


def pct(column: str) -> pl.Expr:
    return (100.0 * pl.col(column).cast(pl.Float64) / pl.col("total").cast(pl.Float64)).round(4)


def main() -> None:
    if not SURVIVORS.exists():
        raise FileNotFoundError(f"missing survivor parquet: {SURVIVORS}")

    lf = pl.scan_parquet(SURVIVORS).with_columns(
        [
            ((pl.col("pi_clash") == 0.0) & (pl.col("pi_complement") == 0.0)).alias("zero_zero"),
            (pl.col("pi_clash") == 0.0).alias("zero_clash"),
            (pl.col("pi_complement") == 0.0).alias("zero_complement"),
            (pl.col("pi_complement") > 0.0).alias("positive_complement"),
            (pl.col("pi_clash") > 0.0).alias("positive_clash"),
        ]
    )

    summary = lf.select(
        [
            pl.len().alias("total"),
            pl.col("pi_clash").min().alias("pi_clash_min"),
            pl.col("pi_clash").quantile(0.25).alias("pi_clash_p25"),
            pl.col("pi_clash").median().alias("pi_clash_median"),
            pl.col("pi_clash").quantile(0.75).alias("pi_clash_p75"),
            pl.col("pi_clash").quantile(0.95).alias("pi_clash_p95"),
            pl.col("pi_clash").max().alias("pi_clash_max"),
            pl.col("pi_complement").min().alias("pi_complement_min"),
            pl.col("pi_complement").quantile(0.25).alias("pi_complement_p25"),
            pl.col("pi_complement").median().alias("pi_complement_median"),
            pl.col("pi_complement").quantile(0.75).alias("pi_complement_p75"),
            pl.col("pi_complement").quantile(0.95).alias("pi_complement_p95"),
            pl.col("pi_complement").max().alias("pi_complement_max"),
            pl.col("score").min().alias("score_min"),
            pl.col("score").median().alias("score_median"),
            pl.col("score").max().alias("score_max"),
            pl.col("zero_zero").sum().alias("zero_zero_count"),
            pl.col("zero_clash").sum().alias("zero_clash_count"),
            pl.col("zero_complement").sum().alias("zero_complement_count"),
            pl.col("positive_complement").sum().alias("positive_complement_count"),
            pl.col("positive_clash").sum().alias("positive_clash_count"),
        ]
    ).with_columns(
        [
            pct("zero_zero_count").alias("zero_zero_pct"),
            pct("zero_clash_count").alias("zero_clash_pct"),
            pct("zero_complement_count").alias("zero_complement_pct"),
            pct("positive_complement_count").alias("positive_complement_pct"),
            pct("positive_clash_count").alias("positive_clash_pct"),
        ]
    )

    by_pi_state = (
        lf.group_by(
            [
                pl.when(pl.col("pi_clash") == 0.0)
                .then(pl.lit("zero_clash"))
                .otherwise(pl.lit("positive_clash"))
                .alias("clash_state"),
                pl.when(pl.col("pi_complement") == 0.0)
                .then(pl.lit("zero_complement"))
                .otherwise(pl.lit("positive_complement"))
                .alias("complement_state"),
            ]
        )
        .agg(
            [
                pl.len().alias("count"),
                pl.col("score").mean().alias("mean_score"),
                pl.col("pi_clash").mean().alias("mean_pi_clash"),
                pl.col("pi_complement").mean().alias("mean_pi_complement"),
            ]
        )
        .sort("count", descending=True)
    )

    print("=== V-space survivor Pi summary ===")
    print(summary.collect())
    print("\n=== Survivor Pi state counts ===")
    print(by_pi_state.collect())
    print("\n=== Lowest-complement survivors ===")
    print(
        lf.select(["product_id", "synthon_a_id", "score", "pi_clash", "pi_complement"])
        .sort(["pi_complement", "pi_clash", "score"])
        .head(10)
        .collect()
    )


if __name__ == "__main__":
    main()
