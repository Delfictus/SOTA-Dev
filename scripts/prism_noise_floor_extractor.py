#!/usr/bin/env python3
"""Extract six-channel noise-floor masks for n80 spike significance filtering."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import polars as pl

from prism_dstw.ontology import StreamId

from scripts.prism_n80_extraction_common import (
    DEFAULT_OUTPUT_DIR,
    DEFAULT_RAW_ROOT,
    StreamFile,
    discover_stream_files,
    emit,
    filter_streams,
    json_float,
    json_int,
    json_str,
    parse_stream_selector,
    read_json_object,
    write_n80_parquet,
)


@dataclass(frozen=True)
class NoiseFloorRow:
    condition_id: str
    replica_id: int
    stream_id: StreamId
    sh_channel: int
    mu: float
    sigma: float
    snr3_threshold: float
    snr5_threshold: float
    n_samples: int
    phase: str


def rows_for_file(item: StreamFile, *, snr_multiplier: float, anchor_multiplier: float) -> list[NoiseFloorRow]:
    data = read_json_object(item.path)
    mu_raw = data.get("mu")
    sigma_raw = data.get("sigma")
    if not isinstance(mu_raw, list) or not isinstance(sigma_raw, list):
        raise ValueError(f"{item.path}: expected mu/sigma arrays")
    if len(mu_raw) != 6 or len(sigma_raw) != 6:
        raise ValueError(f"{item.path}: expected six SH-band channels")
    phase_value = data.get("phase")
    n_samples_value = data.get("n_samples")
    return [
        NoiseFloorRow(
            condition_id=item.condition_id,
            replica_id=item.replica_id,
            stream_id=StreamId(item.stream_id),
            sh_channel=channel,
            mu=json_float(mu_raw[channel]),
            sigma=json_float(sigma_raw[channel]),
            snr3_threshold=json_float(mu_raw[channel]) + snr_multiplier * json_float(sigma_raw[channel]),
            snr5_threshold=json_float(mu_raw[channel]) + anchor_multiplier * json_float(sigma_raw[channel]),
            n_samples=json_int(n_samples_value),
            phase=json_str(phase_value),
        )
        for channel in range(6)
    ]


def frame_from_rows(rows: list[NoiseFloorRow]) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "condition_id": [row.condition_id for row in rows],
            "replica_id": [row.replica_id for row in rows],
            "stream_id": [int(row.stream_id) for row in rows],
            "sh_channel": [row.sh_channel for row in rows],
            "mu": [row.mu for row in rows],
            "sigma": [row.sigma for row in rows],
            "snr3_threshold": [row.snr3_threshold for row in rows],
            "snr5_threshold": [row.snr5_threshold for row in rows],
            "n_samples": [row.n_samples for row in rows],
            "phase": [row.phase for row in rows],
        }
    ).with_columns(
        pl.col("replica_id").cast(pl.UInt8),
        pl.col("stream_id").cast(pl.UInt8),
        pl.col("sh_channel").cast(pl.UInt8),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", type=Path, default=DEFAULT_RAW_ROOT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--condition-id")
    parser.add_argument("--replica-id", type=int)
    parser.add_argument("--streams")
    parser.add_argument("--max-streams", type=int)
    parser.add_argument("--snr-multiplier", type=float, default=3.0)
    parser.add_argument("--causal-anchor-multiplier", type=float, default=5.0)
    args = parser.parse_args()

    selected = filter_streams(
        discover_stream_files(args.raw_root, "noise_floor.json"),
        condition_id=args.condition_id,
        replica_id=args.replica_id,
        stream_ids=parse_stream_selector(args.streams),
        max_streams=args.max_streams,
    )
    rows: list[NoiseFloorRow] = []
    for item in selected:
        rows.extend(
            rows_for_file(
                item,
                snr_multiplier=args.snr_multiplier,
                anchor_multiplier=args.causal_anchor_multiplier,
            )
        )
    frame = frame_from_rows(rows)
    output = args.out_dir / "stream_snr_masks.parquet"
    write_n80_parquet(
        frame.lazy(),
        output,
        producer_script=Path(__file__),
        pipeline_stage="phase2_noise_floor_extractor",
        schema_version="prism.stream_snr_masks.v1",
        partition_keys=("condition_id", "replica_id", "stream_id", "sh_channel"),
        raw_inputs=[item.path for item in selected],
        ledger_parameters={
            "snr_multiplier": args.snr_multiplier,
            "causal_anchor_multiplier": args.causal_anchor_multiplier,
            "channel_count": 6,
        },
        row_count=frame.height,
    )
    emit(f"WROTE {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
