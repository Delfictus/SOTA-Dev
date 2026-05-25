#!/usr/bin/env python3
"""Cross-reference adaptive timestep records with BOCPD survival regimes."""

from __future__ import annotations

import argparse
import struct
from dataclasses import dataclass
from pathlib import Path

import polars as pl

from prism_dstw.ontology import Picosecond, StreamId, TimeStep

from scripts.prism_n80_extraction_common import (
    DEFAULT_OUTPUT_DIR,
    DEFAULT_RAW_ROOT,
    StreamFile,
    discover_stream_files,
    emit,
    filter_streams,
    parse_stream_selector,
    write_n80_parquet,
)


RECORD_BYTES = 32


@dataclass(frozen=True)
class AdaptiveDtRow:
    condition_id: str
    replica_id: int
    stream_id: StreamId
    chunk_idx: int
    frame_idx: TimeStep
    dt_ps: Picosecond
    reason_code: int
    dt_reduction_event: bool


def parse_adaptive_dt(item: StreamFile, *, default_dt_ps: Picosecond) -> list[AdaptiveDtRow]:
    file_size = item.path.stat().st_size
    if file_size % RECORD_BYTES != 0:
        raise ValueError(f"{item.path}: byte length is not a multiple of {RECORD_BYTES}")
    rows: list[AdaptiveDtRow] = []
    with item.path.open("rb") as handle:
        while True:
            raw_record = handle.read(RECORD_BYTES)
            if not raw_record:
                break
            if len(raw_record) != RECORD_BYTES:
                raise ValueError(f"{item.path}: truncated adaptive-dt record")
            chunk_idx, frame_idx, dt_ps_raw, reason_code, _pad = struct.unpack(
                "<QQdII", raw_record
            )
            dt_ps = Picosecond(float(dt_ps_raw))
            rows.append(
                AdaptiveDtRow(
                    condition_id=item.condition_id,
                    replica_id=item.replica_id,
                    stream_id=StreamId(item.stream_id),
                    chunk_idx=int(chunk_idx),
                    frame_idx=TimeStep(int(frame_idx)),
                    dt_ps=dt_ps,
                    reason_code=int(reason_code),
                    dt_reduction_event=float(dt_ps) < float(default_dt_ps),
                )
            )
    return rows


def frame_from_rows(rows: list[AdaptiveDtRow]) -> pl.LazyFrame:
    return pl.DataFrame(
        {
            "condition_id": [row.condition_id for row in rows],
            "replica_id": [row.replica_id for row in rows],
            "stream_id": [int(row.stream_id) for row in rows],
            "chunk_idx": [row.chunk_idx for row in rows],
            "frame_idx": [int(row.frame_idx) for row in rows],
            "dt_ps": [float(row.dt_ps) for row in rows],
            "reason_code": [row.reason_code for row in rows],
            "dt_reduction_event": [row.dt_reduction_event for row in rows],
        }
    ).lazy().with_columns(
        pl.col("replica_id").cast(pl.UInt8),
        pl.col("stream_id").cast(pl.UInt8),
        pl.col("chunk_idx").cast(pl.UInt32),
        pl.col("frame_idx").cast(pl.Int32),
    )


def build_kinetic_strain_frame(
    rows: list[AdaptiveDtRow],
    *,
    bocpd_survival_parquet: Path,
) -> pl.LazyFrame:
    dt_lf = frame_from_rows(rows)
    regimes = pl.scan_parquet(bocpd_survival_parquet).select(
        "condition_id",
        "replica_id",
        "stream_id",
        "chunk_idx",
        "regime_id",
        "thermal_phase",
        "survival_time_ps",
        "temperature_K",
        "posterior_max",
        "reset_probability",
    )
    return (
        dt_lf.join(regimes, on=("condition_id", "replica_id", "stream_id", "chunk_idx"), how="left")
        .with_columns(
            (
                pl.col("dt_reduction_event")
                & pl.col("regime_id").is_not_null()
            ).alias("dt_drop_coincident_with_regime_change")
        )
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", type=Path, default=DEFAULT_RAW_ROOT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--bocpd-survival-parquet", type=Path, default=DEFAULT_OUTPUT_DIR / "bocpd_survival_regimes.parquet")
    parser.add_argument("--condition-id")
    parser.add_argument("--replica-id", type=int)
    parser.add_argument("--streams")
    parser.add_argument("--max-streams", type=int)
    parser.add_argument("--default-dt-ps", type=float, default=0.004)
    args = parser.parse_args()

    selected = filter_streams(
        discover_stream_files(args.raw_root, "adaptive_dt.bin"),
        condition_id=args.condition_id,
        replica_id=args.replica_id,
        stream_ids=parse_stream_selector(args.streams),
        max_streams=args.max_streams,
    )
    rows: list[AdaptiveDtRow] = []
    default_dt = Picosecond(args.default_dt_ps)
    for item in selected:
        rows.extend(parse_adaptive_dt(item, default_dt_ps=default_dt))
    lazy = build_kinetic_strain_frame(rows, bocpd_survival_parquet=args.bocpd_survival_parquet)
    output = args.out_dir / "kinetic_strain_events.parquet"
    write_n80_parquet(
        lazy,
        output,
        producer_script=Path(__file__),
        pipeline_stage="phase2_adaptive_dt_extractor",
        schema_version="prism.kinetic_strain_events.v1",
        partition_keys=("condition_id", "replica_id", "stream_id", "chunk_idx"),
        raw_inputs=[item.path for item in selected],
        source_parquets=[args.bocpd_survival_parquet],
        ledger_parameters={
            "default_dt_ps": args.default_dt_ps,
            "dt_reduction_predicate": "dt_ps < default_dt_ps",
            "bocpd_join_key": "condition_id,replica_id,stream_id,chunk_idx",
        },
        row_count=len(rows),
    )
    emit(f"WROTE {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
