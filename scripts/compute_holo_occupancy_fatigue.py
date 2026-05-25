#!/usr/bin/env python3
"""Compute holo-minus-apo occupancy fatigue deltas once holo tensors exist."""

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
APO_DIR = CAMPAIGN_DIR / "integrated_spike_events/n80_full_scale"
HOLO_DIR = CAMPAIGN_DIR / "integrated_spike_events/holo_n80_full_scale"
DEFAULT_OUTPUT = HOLO_DIR / "holo_occupancy_fatigue_delta.parquet"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apo-hysteresis", type=Path, default=APO_DIR / "hysteresis_tensor.parquet")
    parser.add_argument("--holo-hysteresis", type=Path, default=HOLO_DIR / "holo_hysteresis_tensor.parquet")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--apo-condition", default="glp1r_6XOX_WT")
    parser.add_argument("--holo-condition", default="glp1r_6XOX_HOLO_ALENI")
    parser.add_argument(
        "--fatigue-threshold",
        type=float,
        default=-0.10,
        help="Delta hysteresis ratio at or below this value is classified as drug-induced fatigue positive.",
    )
    return parser.parse_args()


def reversibility_expr(path: Path) -> pl.Expr:
    schema = pl.scan_parquet(path).collect_schema()
    if "hysteresis_ratio" in schema.names():
        return pl.col("hysteresis_ratio").cast(pl.Float64).clip(0.0, 1.0)
    if "thermal_irreversibility" in schema.names():
        return (1.0 - pl.col("thermal_irreversibility").cast(pl.Float64)).clip(0.0, 1.0)
    raise ValueError(f"{path} must contain hysteresis_ratio or thermal_irreversibility")


def state_frame(path: Path, condition_id: str, prefix: str) -> pl.LazyFrame:
    return (
        pl.scan_parquet(path)
        .filter(pl.col("condition_id") == condition_id)
        .select(
            [
                pl.col("primary_residue_idx").cast(pl.Int32),
                pl.col("protocol_group").cast(pl.Utf8),
                reversibility_expr(path).alias(f"{prefix}_hysteresis_ratio"),
                pl.col("thermal_irreversibility").cast(pl.Float64).alias(f"{prefix}_thermal_irreversibility"),
                pl.col("hysteresis_delta").cast(pl.Float64).alias(f"{prefix}_hysteresis_delta"),
            ]
        )
    )


def delta_frame(
    *,
    apo_hysteresis: Path,
    holo_hysteresis: Path,
    apo_condition: str,
    holo_condition: str,
    fatigue_threshold: float,
) -> pl.LazyFrame:
    apo = state_frame(apo_hysteresis, apo_condition, "apo")
    holo = state_frame(holo_hysteresis, holo_condition, "holo")
    return (
        holo.join(apo, on=["primary_residue_idx", "protocol_group"], how="inner")
        .with_columns(
            [
                (pl.col("holo_hysteresis_ratio") - pl.col("apo_hysteresis_ratio")).alias(
                    "delta_hysteresis_ratio"
                ),
                (pl.col("holo_thermal_irreversibility") - pl.col("apo_thermal_irreversibility")).alias(
                    "delta_thermal_irreversibility"
                ),
                (pl.col("holo_hysteresis_delta") - pl.col("apo_hysteresis_delta")).alias(
                    "delta_hysteresis_spikes"
                ),
            ]
        )
        .with_columns(
            pl.when(pl.col("delta_hysteresis_ratio") <= fatigue_threshold)
            .then(pl.lit("drug_induced_fatigue_positive"))
            .when(pl.col("delta_hysteresis_ratio") >= abs(fatigue_threshold))
            .then(pl.lit("holo_recovery_improved"))
            .otherwise(pl.lit("holo_apo_neutral"))
            .alias("occupancy_fatigue_delta_class")
        )
        .with_columns(
            [
                pl.lit(apo_condition).alias("apo_condition_id"),
                pl.lit(holo_condition).alias("holo_condition_id"),
                pl.lit(fatigue_threshold).alias("fatigue_threshold"),
            ]
        )
        .sort(["occupancy_fatigue_delta_class", "delta_hysteresis_ratio"])
    )


def main() -> int:
    args = parse_args()
    apo_hysteresis = Path(args.apo_hysteresis)
    holo_hysteresis = Path(args.holo_hysteresis)
    output = Path(args.output)
    if not apo_hysteresis.exists():
        raise FileNotFoundError(apo_hysteresis)
    if not holo_hysteresis.exists():
        raise FileNotFoundError(holo_hysteresis)

    frame = delta_frame(
        apo_hysteresis=apo_hysteresis,
        holo_hysteresis=holo_hysteresis,
        apo_condition=str(args.apo_condition),
        holo_condition=str(args.holo_condition),
        fatigue_threshold=float(args.fatigue_threshold),
    )
    rows = frame.select(pl.len().alias("n")).collect(engine="streaming").item()
    write_provenance_parquet(
        frame,
        output,
        producer_script=Path(__file__),
        source_parquets=[apo_hysteresis, holo_hysteresis],
        schema_version="holo_occupancy_fatigue_delta.v1",
        pipeline_stage="holo_occupancy_fatigue_delta",
        partition_keys=["holo_condition_id", "protocol_group"],
        ledger_parameters={
            "delta_formula": "Delta H_r = H_r(holo) - H_r(apo)",
            "classification": "Delta H_r <= fatigue_threshold flags drug_induced_fatigue_positive",
            "fatigue_threshold": float(args.fatigue_threshold),
        },
        ledger_output_value={"rows": int(rows), "output_path": output.as_posix()},
        repo_root=REPO_ROOT,
    )
    sys.stdout.write(f"wrote {output} rows={rows}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
