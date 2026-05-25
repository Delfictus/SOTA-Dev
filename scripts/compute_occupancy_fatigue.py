#!/usr/bin/env python3
"""Compute bounded occupancy fatigue from hysteresis, persistence, and burst motion."""

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
DEFAULT_OUTPUT = N80_DIR / "occupancy_fatigue_risk.parquet"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hysteresis", type=Path, default=N80_DIR / "hysteresis_tensor.parquet")
    parser.add_argument("--bocpd", type=Path, default=N80_DIR / "bocpd_survival_regimes.parquet")
    parser.add_argument("--kcc", type=Path, default=N80_DIR / "kcc_residue_fields.parquet")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def fatigue_frame(hysteresis: Path, bocpd: Path, kcc: Path) -> pl.LazyFrame:
    stream_key = ["condition_id", "replica_id", "stream_id"]
    persistence = (
        pl.scan_parquet(bocpd)
        .select([*stream_key, pl.col("survival_time_ps").cast(pl.Float64).alias("regime_duration_ps")])
        .group_by(stream_key)
        .agg(pl.col("regime_duration_ps").mean())
    )
    kcc_persistence = (
        pl.scan_parquet(kcc)
        .select(
            [
                *stream_key,
                pl.col("residue_idx").cast(pl.Int32).alias("primary_residue_idx"),
                pl.col("burst_motion").cast(pl.Float64),
            ]
        )
        .join(persistence, on=stream_key, how="left")
        .group_by(["condition_id", "primary_residue_idx"])
        .agg(
            [
                pl.col("burst_motion").mean().alias("mean_burst_motion"),
                pl.col("burst_motion").std().fill_null(0.0).alias("burst_motion_std"),
                pl.col("burst_motion").max().alias("max_burst_motion"),
                pl.col("regime_duration_ps").mean().alias("tau_persist"),
                pl.col("regime_duration_ps").std().fill_null(0.0).alias("tau_persist_std"),
            ]
        )
    )
    hysteresis_base = (
        pl.scan_parquet(hysteresis)
        .select(
            [
                "condition_id",
                "primary_residue_idx",
                "protocol_group",
                pl.col("cold_hold_spikes").cast(pl.Float64),
                pl.col("cold_return_spikes").cast(pl.Float64),
                "thermal_irreversibility",
            ]
        )
        .with_columns(
            pl.when(pl.col("cold_hold_spikes") > 0.0)
            .then(pl.col("cold_return_spikes") / pl.col("cold_hold_spikes"))
            .otherwise(0.0)
            .alias("hysteresis_ratio")
        )
    )
    joined = hysteresis_base.join(kcc_persistence, on=["condition_id", "primary_residue_idx"], how="left")
    return (
        joined.with_columns(
            [
                pl.col("mean_burst_motion").max().over("condition_id").alias("condition_max_burst_motion"),
                pl.col("tau_persist").fill_null(0.0),
                pl.col("mean_burst_motion").fill_null(0.0),
                pl.col("max_burst_motion").fill_null(0.0),
            ]
        )
        .with_columns(
            (
                pl.col("mean_burst_motion")
                / pl.max_horizontal([pl.col("condition_max_burst_motion"), pl.lit(1.0)])
            )
            .clip(0.0, 1.0)
            .alias("sigma_burst")
        )
        .with_columns(pl.col("hysteresis_ratio").clip(0.0, 1.0).alias("hysteresis_ratio_bounded"))
        .with_columns(
            (
                (1.0 - pl.col("hysteresis_ratio_bounded"))
                * (1.0 + pl.col("tau_persist")).log()
                * (1.0 + pl.col("sigma_burst"))
            ).alias("occupancy_fatigue_index")
        )
        .with_columns(
            pl.when(pl.col("occupancy_fatigue_index") >= pl.col("occupancy_fatigue_index").quantile(0.9).over("condition_id"))
            .then(pl.lit("high_occupancy_fatigue"))
            .when(pl.col("occupancy_fatigue_index") >= pl.col("occupancy_fatigue_index").quantile(0.5).over("condition_id"))
            .then(pl.lit("moderate_occupancy_fatigue"))
            .otherwise(pl.lit("low_occupancy_fatigue"))
            .alias("fatigue_class")
        )
        .drop("condition_max_burst_motion")
        .sort(["condition_id", "protocol_group", "occupancy_fatigue_index"], descending=[False, False, True])
    )


def main() -> int:
    args = parse_args()
    output = Path(args.output)
    frame = fatigue_frame(Path(args.hysteresis), Path(args.bocpd), Path(args.kcc))
    rows = frame.select(pl.len().alias("n")).collect(engine="streaming").item()
    write_provenance_parquet(
        frame,
        output,
        producer_script=Path(__file__),
        source_parquets=[Path(args.hysteresis), Path(args.bocpd), Path(args.kcc)],
        schema_version="occupancy_fatigue_risk.v1",
        pipeline_stage="occupancy_fatigue_risk",
        partition_keys=["condition_id", "protocol_group"],
        ledger_parameters={
            "formula": "F_occ=(1-H_r_bounded)*log(1+tau_persist)*(1+sigma_burst)",
            "duration_stability": "logarithmic duration transform prevents reciprocal-duration instability",
        },
        ledger_output_value={"rows": int(rows), "output_path": output.as_posix()},
        repo_root=REPO_ROOT,
    )
    sys.stdout.write(f"wrote {output} rows={rows}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
