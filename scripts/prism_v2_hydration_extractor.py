#!/usr/bin/env python3
"""Extract V2 hydration statistics from SNR-masked spike events.

This extractor is intentionally condition-partitioned. The source parquet can
be large enough that a global group-by over voxel_idx is unsafe, so each
condition_id is aggregated and appended as its own output row group.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Iterable

import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq


REQUIRED_COLUMNS = {
    "condition_id",
    "primary_residue_idx",
    "voxel_idx",
    "water_density",
    "wd_change",
    "intensity",
}

OUTPUT_COLUMNS = [
    "condition_id",
    "residue_id",
    "voxel_idx",
    "hydration_tunnel_id",
    "sigma_hyd",
    "solvent_wire_importance",
    "occlusion_risk",
]

OUTPUT_ARROW_SCHEMA = pa.schema(
    [
        pa.field("condition_id", pa.large_string()),
        pa.field("residue_id", pa.int64()),
        pa.field("voxel_idx", pa.int64()),
        pa.field("hydration_tunnel_id", pa.large_string()),
        pa.field("sigma_hyd", pa.float64()),
        pa.field("solvent_wire_importance", pa.float64()),
        pa.field("occlusion_risk", pa.float64()),
    ]
)


def emit_status(event: str, **fields: object) -> None:
    payload = {
        "event": event,
        "timestamp": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        **fields,
    }
    sys.stderr.write(json.dumps(payload, sort_keys=True) + "\n")
    sys.stderr.flush()


def collect_streaming(lazy_frame: pl.LazyFrame) -> pl.DataFrame:
    """Collect with the streaming engine, supporting older Polars call forms."""

    try:
        return lazy_frame.collect(engine="streaming")
    except TypeError:
        return lazy_frame.collect(streaming=True)


def validate_input_schema(event_parquet: Path) -> None:
    if not event_parquet.is_file():
        raise SystemExit(f"--event-parquet does not exist or is not a file: {event_parquet}")
    schema_names = set(pl.scan_parquet(str(event_parquet)).collect_schema().names())
    missing = sorted(REQUIRED_COLUMNS - schema_names)
    if missing:
        raise SystemExit(f"{event_parquet}: missing required V2 column(s): {missing}")


def base_event_scan(event_parquet: Path) -> pl.LazyFrame:
    return pl.scan_parquet(str(event_parquet)).select(
        [
            pl.col("condition_id").cast(pl.Utf8),
            pl.col("primary_residue_idx").cast(pl.Int64),
            pl.col("voxel_idx").cast(pl.Int64),
            pl.col("water_density").cast(pl.Float64),
            pl.col("wd_change").cast(pl.Float64),
            pl.col("intensity").cast(pl.Float64),
        ]
    )


def discover_condition_ids(event_parquet: Path) -> list[str]:
    frame = collect_streaming(
        pl.scan_parquet(str(event_parquet))
        .select(pl.col("condition_id").cast(pl.Utf8))
        .filter(pl.col("condition_id").is_not_null())
        .unique()
        .sort("condition_id")
    )
    return [str(value) for value in frame.get_column("condition_id").to_list()]


def aggregate_condition(event_parquet: Path, condition_id: str) -> pl.DataFrame:
    event_count = pl.len().cast(pl.UInt64).alias("_event_count")
    lazy_frame = (
        base_event_scan(event_parquet)
        .filter(pl.col("condition_id") == condition_id)
        .filter(pl.col("water_density").is_not_null() & (pl.col("water_density") != 0.0))
        .group_by(["condition_id", "primary_residue_idx", "voxel_idx"])
        .agg(
            [
                pl.col("water_density").std(ddof=0).fill_null(0.0).alias("sigma_hyd"),
                (pl.col("wd_change").abs() * pl.col("intensity")).mean().alias("solvent_wire_importance"),
                (pl.col("water_density").max() - pl.col("water_density").min()).alias("occlusion_risk"),
                event_count,
            ]
        )
        .rename({"primary_residue_idx": "residue_id"})
        .with_columns(
            [
                pl.concat_str(
                    [
                        pl.col("condition_id"),
                        pl.lit("_res"),
                        pl.col("residue_id").cast(pl.Utf8),
                        pl.lit("_vox"),
                        pl.col("voxel_idx").cast(pl.Utf8),
                    ]
                ).alias("hydration_tunnel_id"),
                pl.col("sigma_hyd").cast(pl.Float64),
                pl.col("solvent_wire_importance").cast(pl.Float64),
                pl.col("occlusion_risk").cast(pl.Float64),
            ]
        )
        .select([*OUTPUT_COLUMNS, "_event_count"])
    )
    return collect_streaming(lazy_frame)


def write_hydration_statistics(event_parquet: Path, out_dir: Path, conditions: Iterable[str]) -> tuple[Path, int, int]:
    out_dir.mkdir(parents=True, exist_ok=True)
    output_path = out_dir / "hydration_statistics.parquet"
    tmp_path = out_dir / f".hydration_statistics.parquet.tmp-{os.getpid()}"
    tmp_path.unlink(missing_ok=True)

    total_groups = 0
    total_events = 0

    try:
        with pq.ParquetWriter(
            tmp_path,
            schema=OUTPUT_ARROW_SCHEMA,
            compression="zstd",
            write_statistics=True,
        ) as writer:
            for index, condition_id in enumerate(conditions, start=1):
                started = time.monotonic()
                emit_status("condition_chunk_start", condition_index=index, condition_id=condition_id)
                chunk = aggregate_condition(event_parquet, condition_id)
                event_rows = int(chunk.get_column("_event_count").sum()) if chunk.height else 0
                output_chunk = chunk.drop("_event_count").select(OUTPUT_COLUMNS)
                group_rows = output_chunk.height
                if group_rows:
                    table = output_chunk.to_arrow().cast(OUTPUT_ARROW_SCHEMA)
                    writer.write_table(table, row_group_size=100_000)
                total_groups += group_rows
                total_events += event_rows
                emit_status(
                    "condition_chunk_written",
                    condition_index=index,
                    condition_id=condition_id,
                    input_event_rows=event_rows,
                    output_group_rows=group_rows,
                    elapsed_seconds=round(time.monotonic() - started, 3),
                )
            if total_groups == 0:
                writer.write_table(pa.Table.from_batches([], schema=OUTPUT_ARROW_SCHEMA))
        tmp_path.replace(output_path)
    except Exception:
        tmp_path.unlink(missing_ok=True)
        raise

    return output_path, total_events, total_groups


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--event-parquet", required=True, help="Path to spike_events_snr_masked.parquet")
    parser.add_argument("--out-dir", required=True, help="Directory to write hydration_statistics.parquet")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    event_parquet = Path(args.event_parquet)
    out_dir = Path(args.out_dir)

    started = time.monotonic()
    validate_input_schema(event_parquet)
    emit_status("hydration_extractor_start", event_parquet=str(event_parquet), out_dir=str(out_dir))
    conditions = discover_condition_ids(event_parquet)
    emit_status("conditions_discovered", condition_count=len(conditions), condition_ids=conditions)
    output_path, total_events, total_groups = write_hydration_statistics(event_parquet, out_dir, conditions)
    emit_status(
        "hydration_extractor_complete",
        output_path=str(output_path),
        condition_count=len(conditions),
        input_event_rows=total_events,
        output_group_rows=total_groups,
        elapsed_seconds=round(time.monotonic() - started, 3),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
