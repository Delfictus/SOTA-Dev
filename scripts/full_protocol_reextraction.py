#!/usr/bin/env python3
"""Full-protocol spike re-extraction with captured-graph timestep unrolling."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import polars as pl
import pyarrow.parquet as pq


N80_DIR = Path("campaigns/glp1r_aleniglipron/integrated_spike_events/n80_full_scale")
PHASES = (
    "phase_cold_hold",
    "phase_ramp_up",
    "phase_warm_hold",
    "phase_ramp_down",
    "phase_cold_return",
)
GROUPS = (
    "A_ThermalShock",
    "B_UVAromatic",
    "C_Equilibrium",
    "D_Hysteresis",
)
GROUP_STREAMS = {
    "A_ThermalShock": (0, 1, 8, 9, 16, 17),
    "B_UVAromatic": (4, 5, 12, 13),
    "C_Equilibrium": (2, 3, 10, 11, 18, 19),
    "D_Hysteresis": (6, 7, 14, 15),
}
OUTPUT_NAMES = (
    "stream_level_phase_counts.parquet",
    "residue_phase_tensor.parquet",
    "temporal_cascade.parquet",
    "hysteresis_tensor.parquet",
    "interferometric_differential.parquet",
)
SCHEMA_VERSION = "full_protocol_reextraction.v2"
PHASE_MODEL = "captured_graph_multiplier_5phase.v1"
SNR_FILTER_TAG = "sh_channel_0_intensity_gte_snr3_threshold"
CONDITION_RESIDUE_KEYS = ["condition_id", "primary_residue_idx"]
TAG_COLUMNS = {
    "condition_residue_key",
    "output_tensor",
    "aggregation_level",
    "schema_version",
    "phase_model",
    "snr_filter",
    "capture_multiplier_applied",
    "sample_fraction",
    "merge_key_columns",
    "generated_by",
}
RESIDUE_ACCOUNTING_COLUMNS = {
    "source_event_count",
    "src1_spike_count",
    "wavelength_event_count",
    "dominant_wavelength_count",
    "entropy_event_count",
    "source_minus_phase_spikes",
    "wavelength_minus_source_events",
    "entropy_minus_source_events",
    "accounting_valid",
}


def log(message: str) -> None:
    print(message, flush=True)


def stream_group_expr() -> pl.Expr:
    expr: pl.Expr | None = None
    for group, streams in GROUP_STREAMS.items():
        clause = pl.col("stream_id").is_in(list(streams))
        expr = pl.when(clause).then(pl.lit(group)) if expr is None else expr.when(clause).then(pl.lit(group))
    if expr is None:
        raise AssertionError("GROUP_STREAMS is empty")
    return expr.otherwise(pl.lit("unknown")).alias("protocol_group")


def protocol_lf(path: Path) -> pl.LazyFrame:
    required = {
        "condition_id",
        "replica_id",
        "stream_id",
        "current_step",
        "cold_hold_end",
        "ramp_end",
        "warm_hold_end",
        "ramp_down_end",
        "total_steps",
    }
    schema = set(pl.scan_parquet(path).collect_schema().names())
    missing = sorted(required - schema)
    if missing:
        raise ValueError(
            f"{path} is missing {missing}. Re-run scripts/prism_protocol_state_extractor.py "
            "after the capture-multiplier patch."
        )

    return (
        pl.scan_parquet(path)
        .select(
            [
                pl.col("condition_id"),
                pl.col("replica_id").cast(pl.UInt8),
                pl.col("stream_id").cast(pl.UInt8),
                pl.col("current_step").cast(pl.Float64),
                pl.col("total_steps").cast(pl.Float64),
                pl.col("cold_hold_end").cast(pl.Float64),
                pl.col("ramp_end").cast(pl.Float64),
                pl.col("warm_hold_end").cast(pl.Float64),
                pl.col("ramp_down_end").cast(pl.Float64),
            ]
        )
        .with_columns(
            [
                (pl.col("total_steps") / pl.col("current_step")).alias("capture_multiplier"),
                stream_group_expr(),
            ]
        )
        .unique(subset=["condition_id", "replica_id", "stream_id"])
    )


def snr_lf(path: Path) -> pl.LazyFrame:
    return (
        pl.scan_parquet(path)
        .filter(pl.col("sh_channel") == 0)
        .select(
            [
                pl.col("condition_id"),
                pl.col("replica_id").cast(pl.UInt8),
                pl.col("stream_id").cast(pl.UInt8),
                pl.col("snr3_threshold").cast(pl.Float64),
            ]
        )
        .unique(subset=["condition_id", "replica_id", "stream_id"])
    )


def annotated_spikes(
    *,
    spike_parquet: Path,
    protocol_state: Path,
    snr_masks: Path,
    sample_fraction: float,
) -> pl.LazyFrame:
    if sample_fraction <= 0.0 or sample_fraction > 1.0:
        raise ValueError("--sample-fraction must be in (0, 1]")

    spikes = pl.scan_parquet(spike_parquet).select(
        [
            "condition_id",
            pl.col("replica_id").cast(pl.UInt8),
            pl.col("stream_id").cast(pl.UInt8),
            "event_index_in_stream",
            pl.col("primary_residue_idx").cast(pl.Int32),
            pl.col("timestep").cast(pl.Float64),
            pl.col("intensity").cast(pl.Float64),
            pl.col("spike_source").cast(pl.Int32),
            pl.col("wavelength_nm").cast(pl.Float64),
            pl.col("phase_bits").cast(pl.UInt32),
        ]
    )
    if sample_fraction < 1.0:
        stride = max(1, round(1.0 / sample_fraction))
        spikes = spikes.filter((pl.col("event_index_in_stream") % stride) == 0)

    joined = (
        spikes.join(
            snr_lf(snr_masks),
            on=["condition_id", "replica_id", "stream_id"],
            how="inner",
        )
        .filter(pl.col("intensity") >= pl.col("snr3_threshold"))
        .join(
            protocol_lf(protocol_state),
            on=["condition_id", "replica_id", "stream_id"],
            how="inner",
        )
        .with_columns((pl.col("timestep") * pl.col("capture_multiplier")).alias("true_md_step"))
    )
    step = pl.col("true_md_step")
    return joined.with_columns(
        pl.when(step <= pl.col("cold_hold_end"))
        .then(pl.lit("phase_cold_hold"))
        .when((step > pl.col("cold_hold_end")) & (step <= pl.col("ramp_end")))
        .then(pl.lit("phase_ramp_up"))
        .when((step > pl.col("ramp_end")) & (step <= pl.col("warm_hold_end")))
        .then(pl.lit("phase_warm_hold"))
        .when((step > pl.col("warm_hold_end")) & (step <= pl.col("ramp_down_end")))
        .then(pl.lit("phase_ramp_down"))
        .otherwise(pl.lit("phase_cold_return"))
        .alias("thermal_phase")
    )


def stream_level_phase_counts(events: pl.LazyFrame) -> pl.LazyFrame:
    keys = ["condition_id", "primary_residue_idx", "stream_id", "thermal_phase"]
    return events.group_by(keys).agg(
        [
            pl.col("protocol_group").first().alias("protocol_group"),
            pl.len().cast(pl.UInt64).alias("spike_count"),
            pl.col("true_md_step").min().alias("first_md_step"),
            pl.col("true_md_step").max().alias("last_md_step"),
        ]
    )


def conditional_count(*, group: str | None = None, phase: str | None = None) -> pl.Expr:
    mask = pl.lit(True)
    if group is not None:
        mask = mask & (pl.col("protocol_group") == group)
    if phase is not None:
        mask = mask & (pl.col("thermal_phase") == phase)
    return pl.when(mask).then(pl.col("spike_count")).otherwise(0).sum().cast(pl.UInt64)


def residue_phase_counts(stream_counts: pl.LazyFrame) -> pl.LazyFrame:
    aggs: list[pl.Expr] = [
        pl.col("spike_count").sum().cast(pl.UInt64).alias("total_spikes"),
        pl.len().cast(pl.UInt32).alias("stream_phase_bucket_count"),
        pl.col("stream_id").n_unique().cast(pl.UInt8).alias("supporting_stream_count"),
    ]
    for group in GROUPS:
        group_key = group.split("_", 1)[0]
        for phase in PHASES:
            phase_key = phase.removeprefix("phase_")
            aggs.append(conditional_count(group=group, phase=phase).alias(f"{group_key}_{phase_key}_spikes"))
    return stream_counts.group_by(CONDITION_RESIDUE_KEYS).agg(aggs)


def residue_source_stats(events: pl.LazyFrame) -> pl.LazyFrame:
    return events.group_by(CONDITION_RESIDUE_KEYS).agg(
        [
            pl.len().cast(pl.UInt64).alias("source_event_count"),
            pl.when(pl.col("spike_source") == 1)
            .then(1)
            .otherwise(0)
            .sum()
            .cast(pl.UInt64)
            .alias("src1_spike_count"),
            (
                pl.when(pl.col("spike_source") == 1).then(1).otherwise(0).sum().cast(pl.Float64)
                / pl.len().cast(pl.Float64)
            ).alias("src1_fraction")
        ]
    )


def residue_dominant_wavelength(events: pl.LazyFrame) -> pl.LazyFrame:
    keys = CONDITION_RESIDUE_KEYS
    counts = events.group_by(keys + ["wavelength_nm"]).agg(pl.len().cast(pl.UInt64).alias("wavelength_count"))
    return (
        counts.with_columns(pl.col("wavelength_count").sum().over(keys).alias("wavelength_event_count"))
        .sort(keys + ["wavelength_count", "wavelength_nm"], descending=[False, False, True, False])
        .group_by(keys)
        .agg(
            [
                pl.col("wavelength_nm").first().alias("dominant_wavelength_nm"),
                pl.col("wavelength_count").first().alias("dominant_wavelength_count"),
                pl.col("wavelength_event_count").first().alias("wavelength_event_count"),
            ]
        )
    )


def residue_phase_bits_entropy(events: pl.LazyFrame) -> pl.LazyFrame:
    keys = CONDITION_RESIDUE_KEYS
    counts = events.group_by(keys + ["phase_bits"]).agg(pl.len().cast(pl.Float64).alias("phase_bit_count"))
    return (
        counts.with_columns(pl.col("phase_bit_count").sum().over(keys).alias("phase_bit_total"))
        .with_columns((pl.col("phase_bit_count") / pl.col("phase_bit_total")).alias("_p"))
        .group_by(keys)
        .agg(
            [
                pl.col("phase_bit_count").sum().cast(pl.UInt64).alias("entropy_event_count"),
                (-(pl.col("_p") * pl.col("_p").log(2))).sum().alias("phase_bits_entropy"),
            ]
        )
    )


def residue_phase_tensor_from_parts(
    phase_counts: pl.LazyFrame,
    source_stats: pl.LazyFrame,
    wavelength_stats: pl.LazyFrame,
    entropy_stats: pl.LazyFrame,
    *,
    sample_fraction: float,
) -> pl.LazyFrame:
    keys = CONDITION_RESIDUE_KEYS
    return (
        phase_counts.join(source_stats, on=keys, how="left")
        .join(wavelength_stats, on=keys, how="left")
        .join(entropy_stats, on=keys, how="left")
        .with_columns(
            [
                (
                    pl.col("source_event_count").cast(pl.Int64) - pl.col("total_spikes").cast(pl.Int64)
                ).alias("source_minus_phase_spikes"),
                (
                    pl.col("wavelength_event_count").cast(pl.Int64) - pl.col("source_event_count").cast(pl.Int64)
                ).alias("wavelength_minus_source_events"),
                (
                    pl.col("entropy_event_count").cast(pl.Int64) - pl.col("source_event_count").cast(pl.Int64)
                ).alias("entropy_minus_source_events"),
            ]
        )
        .with_columns(
            (
                (pl.col("source_minus_phase_spikes") == 0)
                & (pl.col("wavelength_minus_source_events") == 0)
                & (pl.col("entropy_minus_source_events") == 0)
            ).alias("accounting_valid")
        )
        .pipe(with_output_tags, "residue_phase_tensor.parquet", "condition_residue", sample_fraction)
    )


def temporal_cascade(stream_counts: pl.LazyFrame) -> pl.LazyFrame:
    return (
        stream_counts.filter(pl.col("thermal_phase") == "phase_ramp_up")
        .group_by(["condition_id", "protocol_group", "primary_residue_idx"])
        .agg(
            [
                pl.col("first_md_step").min().alias("first_ramp_md_step"),
                pl.col("spike_count").sum().cast(pl.UInt64).alias("ramp_up_spikes"),
                pl.col("stream_id").n_unique().cast(pl.UInt8).alias("supporting_stream_count"),
            ]
        )
    )


def hysteresis_tensor(stream_counts: pl.LazyFrame) -> pl.LazyFrame:
    return (
        stream_counts.group_by(["condition_id", "primary_residue_idx", "protocol_group"])
        .agg(
            [
                conditional_count(phase="phase_cold_hold").alias("cold_hold_spikes"),
                conditional_count(phase="phase_ramp_up").alias("ramp_up_spikes"),
                conditional_count(phase="phase_warm_hold").alias("warm_hold_spikes"),
                conditional_count(phase="phase_ramp_down").alias("ramp_down_spikes"),
                conditional_count(phase="phase_cold_return").alias("cold_return_spikes"),
            ]
        )
        .with_columns(
            [
                (pl.col("cold_return_spikes").cast(pl.Int64) - pl.col("cold_hold_spikes").cast(pl.Int64)).alias(
                    "hysteresis_delta"
                ),
                (
                    (pl.col("cold_return_spikes").cast(pl.Float64) - pl.col("cold_hold_spikes").cast(pl.Float64)).abs()
                    / (pl.col("cold_hold_spikes") + pl.col("cold_return_spikes")).cast(pl.Float64)
                )
                .fill_nan(None)
                .alias("thermal_irreversibility"),
            ]
        )
    )


def interferometric_differential(stream_counts: pl.LazyFrame) -> pl.LazyFrame:
    aggs = [conditional_count(group="A_ThermalShock", phase=phase).alias(f"A_{phase.removeprefix('phase_')}") for phase in PHASES]
    for group in GROUPS[1:]:
        group_key = group.split("_", 1)[0]
        for phase in PHASES:
            phase_key = phase.removeprefix("phase_")
            aggs.append(conditional_count(group=group, phase=phase).alias(f"{group_key}_{phase_key}"))
    base = stream_counts.group_by(["condition_id", "primary_residue_idx"]).agg(aggs)
    derived: list[pl.Expr] = []
    for group in GROUPS[1:]:
        group_key = group.split("_", 1)[0]
        for phase in PHASES:
            phase_key = phase.removeprefix("phase_")
            a = pl.col(f"A_{phase_key}").cast(pl.Int64)
            g = pl.col(f"{group_key}_{phase_key}").cast(pl.Int64)
            derived.extend(
                [
                    (g - a).alias(f"{group_key}_over_A_{phase_key}_delta"),
                    (g.cast(pl.Float64) / a.cast(pl.Float64)).fill_nan(None).alias(
                        f"{group_key}_over_A_{phase_key}_ratio"
                    ),
                ]
            )
    return base.with_columns(derived)


def with_output_tags(lf: pl.LazyFrame, output_name: str, aggregation_level: str, sample_fraction: float) -> pl.LazyFrame:
    return lf.with_columns(
        [
            pl.concat_str(
                [pl.col("condition_id"), pl.col("primary_residue_idx").cast(pl.Utf8)],
                separator=":",
            ).alias("condition_residue_key"),
            pl.lit(output_name.removesuffix(".parquet")).alias("output_tensor"),
            pl.lit(aggregation_level).alias("aggregation_level"),
            pl.lit(SCHEMA_VERSION).alias("schema_version"),
            pl.lit(PHASE_MODEL).alias("phase_model"),
            pl.lit(SNR_FILTER_TAG).alias("snr_filter"),
            pl.lit(True).alias("capture_multiplier_applied"),
            pl.lit(sample_fraction).alias("sample_fraction"),
            pl.lit(",".join(CONDITION_RESIDUE_KEYS)).alias("merge_key_columns"),
            pl.lit("scripts/full_protocol_reextraction.py").alias("generated_by"),
        ]
    )


def sink(lf: pl.LazyFrame, output: Path) -> float:
    if output.exists():
        output.unlink()
    started = time.monotonic()
    lf.sink_parquet(
        output,
        compression="zstd",
        statistics=True,
        row_group_size=100_000,
        mkdir=True,
        engine="streaming",
    )
    return time.monotonic() - started


def parquet_rows(path: Path) -> int:
    return pq.ParquetFile(path).metadata.num_rows


def parquet_columns(path: Path) -> set[str]:
    return set(pq.ParquetFile(path).schema_arrow.names)


def valid_parquet_rows(path: Path) -> int | None:
    if not path.exists() or path.stat().st_size == 0:
        return None
    try:
        return parquet_rows(path)
    except Exception:
        return None


def record_existing(
    path: Path,
    outputs: dict[str, dict[str, float | int | str | bool]],
    *,
    required_columns: set[str] | None = None,
) -> bool:
    rows = valid_parquet_rows(path)
    if rows is None:
        return False
    if required_columns is not None and not required_columns <= parquet_columns(path):
        missing = sorted(required_columns - parquet_columns(path))
        log(f"REBUILD {path} missing_required_columns={missing}")
        return False
    outputs[path.name] = {"path": str(path), "rows": rows, "seconds": 0.0, "reused": True}
    log(f"REUSE {path} rows={rows}")
    return True


def run_sink(
    name: str,
    lf: pl.LazyFrame,
    path: Path,
    outputs: dict[str, dict[str, float | int | str | bool]],
) -> None:
    log(f"RUN {name} -> {path}")
    elapsed = sink(lf, path)
    outputs[path.name] = {"path": str(path), "rows": parquet_rows(path), "seconds": elapsed, "reused": False}
    log(f"WROTE {path} rows={outputs[path.name]['rows']} elapsed_seconds={elapsed:.3f}")


def ensure_tagged_existing(
    path: Path,
    *,
    output_name: str,
    aggregation_level: str,
    sample_fraction: float,
    outputs: dict[str, dict[str, float | int | str | bool]],
) -> bool:
    rows = valid_parquet_rows(path)
    if rows is None:
        return False
    if TAG_COLUMNS <= parquet_columns(path):
        outputs[path.name] = {"path": str(path), "rows": rows, "seconds": 0.0, "reused": True}
        log(f"REUSE {path} rows={rows}")
        return True

    tagged_path = path.with_name(f".{path.stem}.tagged.tmp.parquet")
    log(f"RETAG {path} -> {tagged_path}")
    elapsed = sink(with_output_tags(pl.scan_parquet(path), output_name, aggregation_level, sample_fraction), tagged_path)
    tagged_path.replace(path)
    rows = parquet_rows(path)
    outputs[path.name] = {"path": str(path), "rows": rows, "seconds": elapsed, "reused": True, "retagged": True}
    log(f"RETAGGED {path} rows={rows} elapsed_seconds={elapsed:.3f}")
    return True


def residue_tensor_accounting(residue_path: Path, stream_counts_path: Path) -> dict[str, int]:
    residue = pl.scan_parquet(residue_path)
    stream_counts = pl.scan_parquet(stream_counts_path)
    metrics = residue.select(
        [
            pl.len().cast(pl.UInt64).alias("residue_rows"),
            pl.col("total_spikes").sum().cast(pl.UInt64).alias("residue_total_spikes"),
            pl.col("source_event_count").sum().cast(pl.UInt64).alias("source_event_count"),
            pl.col("wavelength_event_count").sum().cast(pl.UInt64).alias("wavelength_event_count"),
            pl.col("entropy_event_count").sum().cast(pl.UInt64).alias("entropy_event_count"),
            (~pl.col("accounting_valid")).sum().cast(pl.UInt64).alias("invalid_accounting_rows"),
        ]
    ).collect().to_dicts()[0]
    duplicate_keys = (
        residue.group_by(CONDITION_RESIDUE_KEYS)
        .agg(pl.len().alias("_key_count"))
        .filter(pl.col("_key_count") > 1)
        .select(pl.len().cast(pl.UInt64).alias("duplicate_key_rows"))
        .collect()
        .item()
    )
    stream_total = (
        stream_counts.select(pl.col("spike_count").sum().cast(pl.UInt64).alias("stream_total_spikes")).collect().item()
    )
    accounting = {
        "residue_rows": int(metrics["residue_rows"]),
        "residue_total_spikes": int(metrics["residue_total_spikes"]),
        "source_event_count": int(metrics["source_event_count"]),
        "wavelength_event_count": int(metrics["wavelength_event_count"]),
        "entropy_event_count": int(metrics["entropy_event_count"]),
        "invalid_accounting_rows": int(metrics["invalid_accounting_rows"]),
        "duplicate_key_rows": int(duplicate_keys),
        "stream_total_spikes": int(stream_total),
    }
    if accounting["invalid_accounting_rows"] or accounting["duplicate_key_rows"]:
        raise ValueError(f"residue_phase_tensor accounting failed: {accounting}")
    if accounting["residue_total_spikes"] != accounting["stream_total_spikes"]:
        raise ValueError(f"residue/stream spike total mismatch: {accounting}")
    return accounting


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spike-parquet", type=Path, default=N80_DIR / "spike_events_snr_masked.parquet")
    parser.add_argument("--protocol-state-summary", type=Path, default=N80_DIR / "protocol_state_summary.parquet")
    parser.add_argument("--snr-masks", type=Path, default=N80_DIR / "stream_snr_masks.parquet")
    parser.add_argument("--out-dir", type=Path, default=N80_DIR)
    parser.add_argument("--sample-fraction", type=float, default=1.0)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    started_all = time.monotonic()
    events = annotated_spikes(
        spike_parquet=args.spike_parquet,
        protocol_state=args.protocol_state_summary,
        snr_masks=args.snr_masks,
        sample_fraction=args.sample_fraction,
    )

    outputs: dict[str, dict[str, float | int | str | bool]] = {}

    stream_counts_path = args.out_dir / "stream_level_phase_counts.parquet"
    if not ensure_tagged_existing(
        stream_counts_path,
        output_name="stream_level_phase_counts.parquet",
        aggregation_level="condition_residue_stream_phase",
        sample_fraction=args.sample_fraction,
        outputs=outputs,
    ):
        run_sink(
            "stream_level_phase_counts",
            with_output_tags(
                stream_level_phase_counts(events),
                "stream_level_phase_counts.parquet",
                "condition_residue_stream_phase",
                args.sample_fraction,
            ),
            stream_counts_path,
            outputs,
        )

    stream_counts_lf = pl.scan_parquet(stream_counts_path)

    residue_path = args.out_dir / "residue_phase_tensor.parquet"
    if record_existing(residue_path, outputs, required_columns=TAG_COLUMNS | RESIDUE_ACCOUNTING_COLUMNS):
        accounting = residue_tensor_accounting(residue_path, stream_counts_path)
        outputs[residue_path.name]["accounting"] = json.dumps(accounting, sort_keys=True)
        log(f"VALIDATED {residue_path} accounting={json.dumps(accounting, sort_keys=True)}")
    else:
        tmp_phase_counts = args.out_dir / ".full_protocol_residue_phase_counts.tmp.parquet"
        tmp_source_stats = args.out_dir / ".full_protocol_residue_source_stats.tmp.parquet"
        tmp_wavelength_stats = args.out_dir / ".full_protocol_residue_wavelength_stats.tmp.parquet"
        tmp_entropy_stats = args.out_dir / ".full_protocol_residue_phase_bits_entropy.tmp.parquet"

        log(f"RUN residue_phase_counts_tmp -> {tmp_phase_counts}")
        elapsed = sink(residue_phase_counts(stream_counts_lf), tmp_phase_counts)
        log(f"WROTE {tmp_phase_counts} rows={parquet_rows(tmp_phase_counts)} elapsed_seconds={elapsed:.3f}")

        log(f"RUN residue_source_stats_tmp -> {tmp_source_stats}")
        elapsed = sink(residue_source_stats(events), tmp_source_stats)
        log(f"WROTE {tmp_source_stats} rows={parquet_rows(tmp_source_stats)} elapsed_seconds={elapsed:.3f}")

        log(f"RUN residue_dominant_wavelength_tmp -> {tmp_wavelength_stats}")
        elapsed = sink(residue_dominant_wavelength(events), tmp_wavelength_stats)
        log(f"WROTE {tmp_wavelength_stats} rows={parquet_rows(tmp_wavelength_stats)} elapsed_seconds={elapsed:.3f}")

        log(f"RUN residue_phase_bits_entropy_tmp -> {tmp_entropy_stats}")
        elapsed = sink(residue_phase_bits_entropy(events), tmp_entropy_stats)
        log(f"WROTE {tmp_entropy_stats} rows={parquet_rows(tmp_entropy_stats)} elapsed_seconds={elapsed:.3f}")

        run_sink(
            "residue_phase_tensor",
            residue_phase_tensor_from_parts(
                pl.scan_parquet(tmp_phase_counts),
                pl.scan_parquet(tmp_source_stats),
                pl.scan_parquet(tmp_wavelength_stats),
                pl.scan_parquet(tmp_entropy_stats),
                sample_fraction=args.sample_fraction,
            ),
            residue_path,
            outputs,
        )
        accounting = residue_tensor_accounting(residue_path, stream_counts_path)
        outputs[residue_path.name]["accounting"] = json.dumps(accounting, sort_keys=True)
        log(f"VALIDATED {residue_path} accounting={json.dumps(accounting, sort_keys=True)}")

    temporal_path = args.out_dir / "temporal_cascade.parquet"
    if not record_existing(temporal_path, outputs, required_columns=TAG_COLUMNS):
        run_sink(
            "temporal_cascade",
            with_output_tags(
                temporal_cascade(stream_counts_lf),
                "temporal_cascade.parquet",
                "condition_protocol_group_residue",
                args.sample_fraction,
            ),
            temporal_path,
            outputs,
        )

    hysteresis_path = args.out_dir / "hysteresis_tensor.parquet"
    if not record_existing(hysteresis_path, outputs, required_columns=TAG_COLUMNS):
        run_sink(
            "hysteresis_tensor",
            with_output_tags(
                hysteresis_tensor(stream_counts_lf),
                "hysteresis_tensor.parquet",
                "condition_protocol_group_residue",
                args.sample_fraction,
            ),
            hysteresis_path,
            outputs,
        )

    differential_path = args.out_dir / "interferometric_differential.parquet"
    if not record_existing(differential_path, outputs, required_columns=TAG_COLUMNS):
        run_sink(
            "interferometric_differential",
            with_output_tags(
                interferometric_differential(stream_counts_lf),
                "interferometric_differential.parquet",
                "condition_residue",
                args.sample_fraction,
            ),
            differential_path,
            outputs,
        )

    summary_path = args.out_dir / "full_protocol_reextraction_summary.json"
    summary = {
        "sample_fraction": args.sample_fraction,
        "total_seconds": time.monotonic() - started_all,
        "spike_parquet": str(args.spike_parquet),
        "protocol_state_summary": str(args.protocol_state_summary),
        "snr_masks": str(args.snr_masks),
        "outputs": outputs,
    }
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    log(f"WROTE {summary_path} elapsed_seconds={summary['total_seconds']:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
