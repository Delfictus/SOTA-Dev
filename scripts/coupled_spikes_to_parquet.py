#!/usr/bin/env python3
"""
Convert coupled_spikes.json (streamed, multi-GB) to a compressed Parquet file.

Format produced by the PRISM-TWIN engine:
    {"n_spikes_a": N, "n_spikes_b": M, "spikes": [ {stream_id, timestep, x, y, z,
     intensity, vib_energy, water_density, spike_source, n_nearby_excited,
     wavelength_nm, ...}, ... ]}

Output Parquet schema:
    stream_id   u8   0=A (scout, thermal)   1=B (observer, thermal+NMA)
    timestep    u32
    x,y,z       f32
    intensity   f32
    vib_energy  f32
    water_density f32
    spike_source  i8
    n_nearby_excited u8
    wavelength_nm f32

Usage:
    python3 scripts/coupled_spikes_to_parquet.py <target_dir>
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import ijson
import pyarrow as pa
import pyarrow.parquet as pq

CHUNK_ROWS = 2_000_000

SCHEMA = pa.schema([
    ("stream_id", pa.uint8()),
    ("timestep", pa.uint32()),
    ("x", pa.float32()),
    ("y", pa.float32()),
    ("z", pa.float32()),
    ("intensity", pa.float32()),
    ("vib_energy", pa.float32()),
    ("water_density", pa.float32()),
    ("spike_source", pa.int8()),
    ("n_nearby_excited", pa.uint8()),
    ("wavelength_nm", pa.float32()),
])


def convert(src: Path, dst: Path) -> int:
    """Stream JSON → Parquet. Returns row count."""
    cols = {f.name: [] for f in SCHEMA}
    writer = pq.ParquetWriter(dst, SCHEMA, compression="zstd", compression_level=5)

    n = 0
    t0 = time.time()
    with open(src, "rb") as f:
        # use_float=True makes ijson return Python floats (not Decimal)
        for rec in ijson.items(f, "spikes.item", use_float=True):
            cols["stream_id"].append(int(rec.get("stream_id", 0)))
            cols["timestep"].append(int(rec.get("timestep", 0)))
            cols["x"].append(float(rec.get("x", 0.0)))
            cols["y"].append(float(rec.get("y", 0.0)))
            cols["z"].append(float(rec.get("z", 0.0)))
            cols["intensity"].append(float(rec.get("intensity", 0.0)))
            cols["vib_energy"].append(float(rec.get("vib_energy", 0.0)))
            cols["water_density"].append(float(rec.get("water_density", 0.0)))
            cols["spike_source"].append(int(rec.get("spike_source", 0)))
            cols["n_nearby_excited"].append(int(rec.get("n_nearby_excited", 0)))
            cols["wavelength_nm"].append(float(rec.get("wavelength_nm", 0.0)))
            n += 1
            if len(cols["stream_id"]) >= CHUNK_ROWS:
                writer.write_table(pa.table(cols, schema=SCHEMA))
                cols = {f.name: [] for f in SCHEMA}
                if n % 20_000_000 == 0:
                    rate = n / (time.time() - t0 + 1e-6)
                    print(f"  {n/1e6:.0f}M rows ({rate/1e6:.2f}M/s)", flush=True)

    if cols["stream_id"]:
        writer.write_table(pa.table(cols, schema=SCHEMA))
    writer.close()
    return n


def main() -> int:
    if len(sys.argv) < 2:
        print("Usage: coupled_spikes_to_parquet.py <target_dir>", file=sys.stderr)
        return 2

    target_dir = Path(sys.argv[1])
    src = target_dir / "coupled_spikes.json"
    dst = target_dir / "coupled_spikes.parquet"

    if not src.exists():
        print(f"No {src} found", file=sys.stderr)
        return 1
    if dst.exists() and dst.stat().st_size > 1024:
        print(f"{dst.name} already exists ({dst.stat().st_size/1e6:.1f} MB) — skipping")
        return 0

    print(f"Converting {src.name} ({src.stat().st_size/1e9:.2f} GB) → {dst.name}")
    t0 = time.time()
    n = convert(src, dst)
    dt = time.time() - t0
    print(f"Done: {n:,} rows in {dt:.1f}s")
    print(f"  JSON:    {src.stat().st_size/1e9:.3f} GB")
    print(f"  Parquet: {dst.stat().st_size/1e9:.3f} GB")
    print(f"  Compression ratio: {src.stat().st_size / max(dst.stat().st_size, 1):.1f}x")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
