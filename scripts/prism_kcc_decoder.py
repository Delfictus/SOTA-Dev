#!/usr/bin/env python3
"""Decode PRKCC001 causal/kinematic fields for the n80 GLP-1R campaign."""

from __future__ import annotations

import argparse
import struct
import sys
from dataclasses import dataclass
from pathlib import Path

import polars as pl

from prism_dstw.ontology import ResidueIdx, StreamId

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


KCC_F32_FIELDS = (
    "temporal_corr",
    "direction_score",
    "motion_efficiency",
    "burst_motion",
    "phase_shift",
    "causal_lag",
    "lag_corr_peak",
    "local_cov",
    "net_dx",
    "net_dy",
    "net_dz",
    "sum_m",
)
KCC_U32_FIELDS = ("residue_count", "active_causal")


@dataclass(frozen=True)
class KccHeader:
    stream_id: StreamId
    run_id: str
    stem: str
    record_count: int
    byte_stride: int
    payload_size: int
    payload_offset: int


@dataclass(frozen=True)
class KccResidueRow:
    campaign_id: str
    condition_id: str
    replica_id: int
    stream_id: StreamId
    run_id: str
    stem: str
    residue_idx: ResidueIdx


def _read_u64(handle: object) -> int:
    if not hasattr(handle, "read"):
        raise TypeError("binary handle does not expose read()")
    raw = handle.read(8)
    if not isinstance(raw, bytes) or len(raw) != 8:
        raise ValueError("unexpected EOF while reading u64")
    return int(struct.unpack("<Q", raw)[0])


def parse_header(path: Path) -> KccHeader:
    with path.open("rb") as handle:
        magic = handle.read(8).decode("ascii", errors="replace")
        if magic != "PRKCC001":
            raise ValueError(f"{path}: expected PRKCC001, got {magic}")
        schema_version, endian_marker, stream_id = struct.unpack("<III", handle.read(12))
        if schema_version != 1:
            raise ValueError(f"{path}: unsupported schema_version {schema_version}")
        if endian_marker != 0x01020304:
            raise ValueError(f"{path}: unsupported endian marker {endian_marker}")
        run_len = _read_u64(handle)
        run_id = handle.read(run_len).decode("utf-8", errors="replace")
        stem_len = _read_u64(handle)
        stem = handle.read(stem_len).decode("utf-8", errors="replace")
        record_count, byte_stride, payload_size = struct.unpack("<QQQ", handle.read(24))
        payload_offset = handle.tell()
    return KccHeader(
        stream_id=StreamId(stream_id),
        run_id=run_id,
        stem=stem,
        record_count=int(record_count),
        byte_stride=int(byte_stride),
        payload_size=int(payload_size),
        payload_offset=int(payload_offset),
    )


def decode_kcc_stream(item: StreamFile) -> pl.DataFrame:
    header = parse_header(item.path)
    with item.path.open("rb") as handle:
        handle.seek(header.payload_offset)
        n_residues, field_count = struct.unpack("<QQ", handle.read(16))
        if int(n_residues) != header.record_count:
            raise ValueError(f"{item.path}: n_residues does not match record_count")
        fields_f32: dict[str, list[float]] = {}
        fields_u32: dict[str, list[int]] = {}
        for _ in range(int(field_count)):
            name_len = _read_u64(handle)
            name = handle.read(name_len).decode("utf-8", errors="replace")
            dtype_code = struct.unpack("<B", handle.read(1))[0]
            section_size = _read_u64(handle)
            raw = handle.read(section_size)
            if dtype_code == 1:
                values = struct.unpack("<" + "f" * (section_size // 4), raw)
                fields_f32[name] = [float(value) for value in values]
            elif dtype_code == 2:
                values_i = struct.unpack("<" + "I" * (section_size // 4), raw)
                fields_u32[name] = [int(value) for value in values_i]
            else:
                raise ValueError(f"{item.path}: unknown KCC dtype code {dtype_code}")
    for field in KCC_F32_FIELDS:
        if field not in fields_f32:
            raise ValueError(f"{item.path}: missing required f32 KCC field {field!r}")
        if len(fields_f32[field]) != header.record_count:
            raise ValueError(
                f"{item.path}: f32 KCC field {field!r} has {len(fields_f32[field])} "
                f"values, expected {header.record_count}"
            )
    for field in KCC_U32_FIELDS:
        if field not in fields_u32:
            raise ValueError(f"{item.path}: missing required u32 KCC field {field!r}")
        if len(fields_u32[field]) != header.record_count:
            raise ValueError(
                f"{item.path}: u32 KCC field {field!r} has {len(fields_u32[field])} "
                f"values, expected {header.record_count}"
            )
    boundary = KccResidueRow(
        campaign_id="glp1r_aleniglipron",
        condition_id=item.condition_id,
        replica_id=item.replica_id,
        stream_id=header.stream_id,
        run_id=header.run_id,
        stem=header.stem,
        residue_idx=ResidueIdx(0),
    )
    residue_indices: list[ResidueIdx] = [
        ResidueIdx(residue)
        for residue in range(header.record_count)
    ]
    columns: dict[str, list[int] | list[float] | list[str]] = {
        "campaign_id": [boundary.campaign_id] * header.record_count,
        "condition_id": [boundary.condition_id] * header.record_count,
        "replica_id": [boundary.replica_id] * header.record_count,
        "stream_id": [int(boundary.stream_id)] * header.record_count,
        "run_id": [boundary.run_id] * header.record_count,
        "stem": [boundary.stem] * header.record_count,
        "residue_idx": [int(residue) for residue in residue_indices],
    }
    for field in KCC_F32_FIELDS:
        columns[field] = fields_f32[field]
    for field in KCC_U32_FIELDS:
        columns[field] = fields_u32[field]
    return pl.DataFrame(columns)


def build_lazy_frame(frames: list[pl.DataFrame]) -> pl.LazyFrame:
    if not frames:
        raise ValueError("no KCC frames decoded")
    return pl.concat(frames, how="vertical").lazy().with_columns(
        pl.col("residue_idx").cast(pl.Int32),
        pl.col("stream_id").cast(pl.UInt8),
        pl.col("replica_id").cast(pl.UInt8),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", type=Path, default=DEFAULT_RAW_ROOT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--condition-id")
    parser.add_argument("--replica-id", type=int)
    parser.add_argument("--streams")
    parser.add_argument("--max-streams", type=int)
    args = parser.parse_args()

    selected = filter_streams(
        discover_stream_files(args.raw_root, "kcc_v2full.bin"),
        condition_id=args.condition_id,
        replica_id=args.replica_id,
        stream_ids=parse_stream_selector(args.streams),
        max_streams=args.max_streams,
    )
    frames = [decode_kcc_stream(item) for item in selected]
    lazy = build_lazy_frame(frames)
    output = args.out_dir / "kcc_residue_fields.parquet"
    write_n80_parquet(
        lazy,
        output,
        producer_script=Path(__file__),
        pipeline_stage="phase1_kcc_decoder",
        schema_version="prism.kcc_residue_fields.v2",
        partition_keys=("condition_id", "replica_id", "stream_id", "residue_idx"),
        raw_inputs=[item.path for item in selected],
        ledger_parameters={"decoder": "PRKCC001", "field_count": len(KCC_F32_FIELDS) + len(KCC_U32_FIELDS)},
        row_count=sum(frame.height for frame in frames),
    )
    emit(f"WROTE {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
