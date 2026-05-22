#!/usr/bin/env python3
"""GHOST_NATIVE_SPATIAL_MAPPING_WIRE — probe.

Reads emitted GhostTileFrame v2 records on disk and verifies that the native
aabb_min / aabb_max / centroid_xyz spatial fields are populated (nonzero +
finite + Bit 4 of field_completeness_flags set).

Emits ghost_v2_native_spatial_probe.json with the report fields the directive
requires.

Usage:
    python3 ghost_v2_native_spatial_probe.py <run_dir> [--output <path>]

GhostTileFrame v2 schema (4096 B per record, sector-aligned counter at offset 0):
  offset    field
  0         frame_idx u64
  8         site_id u32
  12        chain_id u8
  13        adjudication_code u8
  14        telemetry_flags u16
  16        kl_divergence f32
  20..116   power_spectrum [f32; 24]
  116..124  thermo_flux [f32; 2]
  124       causal_lead_residue u32
  128       schema_version u32      (= 2 for v2 records)
  132       observation_pass u8
  133       discovery_pass u8
  134       perturbation_chan u8
  135       _pad8 u8
  136       uv_wavelength_nm u16
  138       field_completeness_flags u16  (Bit 4 = SPATIAL_NATIVE_AABB_MIDPOINT)
  140       gear_id u32
  144       dt_fs f32
  148..152  _pad32_for_step_align (M1.2.24)
  152       step_idx u64
  160       aabb_min [f32; 3]   <-- NATIVE
  172       aabb_max [f32; 3]   <-- NATIVE
  184       centroid_xyz [f32; 3]   <-- NATIVE (AABB midpoint alias)
  196..256  _v2_reserved [u32; 15]
  256..4096 _slack [u8; 3840]
"""
from __future__ import annotations

import argparse
import json
import os
import struct
import sys
from glob import glob
from pathlib import Path

PRISM_GHOST_COUNTER_SECTOR = 4096
GHOST_RECORD_BYTES = 4096

# v2 schema offsets
OFF_FRAME_IDX = 0
OFF_SITE_ID = 8
OFF_KL = 16
OFF_SCHEMA_VERSION = 128
OFF_FIELD_COMPLETENESS_FLAGS = 138
OFF_GEAR_ID = 140
OFF_DT_FS = 144
OFF_STEP_IDX = 152
OFF_AABB_MIN = 160
OFF_AABB_MAX = 172
OFF_CENTROID = 184

GHOST_FCF_BIT_SPATIAL_NATIVE_AABB_MIDPOINT = 0x0010


def parse_record(buf: bytes, byte_offset: int) -> dict:
    frame_idx = struct.unpack_from("<Q", buf, OFF_FRAME_IDX)[0]
    site_id = struct.unpack_from("<I", buf, OFF_SITE_ID)[0]
    kl = struct.unpack_from("<f", buf, OFF_KL)[0]
    schema_v = struct.unpack_from("<I", buf, OFF_SCHEMA_VERSION)[0]
    fcf = struct.unpack_from("<H", buf, OFF_FIELD_COMPLETENESS_FLAGS)[0]
    gear_id = struct.unpack_from("<I", buf, OFF_GEAR_ID)[0]
    dt_fs = struct.unpack_from("<f", buf, OFF_DT_FS)[0]
    step_idx = struct.unpack_from("<Q", buf, OFF_STEP_IDX)[0]
    aabb_min = list(struct.unpack_from("<3f", buf, OFF_AABB_MIN))
    aabb_max = list(struct.unpack_from("<3f", buf, OFF_AABB_MAX))
    centroid = list(struct.unpack_from("<3f", buf, OFF_CENTROID))
    aabb_min_nonzero = any(x != 0.0 for x in aabb_min)
    aabb_max_nonzero = any(x != 0.0 for x in aabb_max)
    centroid_nonzero = any(x != 0.0 for x in centroid)
    return {
        "byte_offset": byte_offset,
        "frame_idx": frame_idx,
        "site_id": site_id,
        "kl_divergence": kl,
        "schema_version": schema_v,
        "field_completeness_flags": fcf,
        "gear_id": gear_id,
        "dt_fs": dt_fs,
        "step_idx": step_idx,
        "aabb_min": aabb_min,
        "aabb_max": aabb_max,
        "centroid": centroid,
        "aabb_min_nonzero": aabb_min_nonzero,
        "aabb_max_nonzero": aabb_max_nonzero,
        "centroid_nonzero": centroid_nonzero,
        "spatial_native_bit_set": bool(fcf & GHOST_FCF_BIT_SPATIAL_NATIVE_AABB_MIDPOINT),
    }


def probe_run(run_dir: Path, max_records_per_file: int = 5000) -> dict:
    ghost_files = sorted(glob(str(run_dir / "*_ghost_tiles.bin")))
    schema_v_dist: dict[int, int] = {}
    n_records_checked = 0
    n_records_with_nonzero_aabb = 0
    n_records_with_nonzero_centroid = 0
    n_records_spatial_bit_set = 0
    first_nonzero_record: dict | None = None
    first_nonzero_stream_id: int | None = None
    per_stream = []

    for path in ghost_files:
        sz = os.path.getsize(path)
        # Stream id from filename pattern: ..._stream{NN}_ghost_tiles.bin
        try:
            stream_id = int(Path(path).stem.split("_stream")[1].split("_")[0])
        except Exception:
            stream_id = -1
        n_slots_total = (sz - PRISM_GHOST_COUNTER_SECTOR) // GHOST_RECORD_BYTES
        n_slots_to_check = min(n_slots_total, max_records_per_file)
        stream_summary = {
            "path": str(path),
            "stream_id": stream_id,
            "size_bytes": sz,
            "n_slots_total": n_slots_total,
            "n_slots_checked": n_slots_to_check,
            "n_with_nonzero_aabb": 0,
            "n_with_nonzero_centroid": 0,
            "n_with_spatial_native_bit": 0,
            "schema_version_seen": set(),
        }
        with open(path, "rb") as f:
            for s in range(n_slots_to_check):
                rec_off = PRISM_GHOST_COUNTER_SECTOR + s * GHOST_RECORD_BYTES
                f.seek(rec_off)
                buf = f.read(GHOST_RECORD_BYTES)
                if len(buf) < 256:
                    break
                r = parse_record(buf, rec_off)
                schema_v_dist[r["schema_version"]] = schema_v_dist.get(r["schema_version"], 0) + 1
                stream_summary["schema_version_seen"].add(r["schema_version"])
                if r["aabb_min_nonzero"] or r["aabb_max_nonzero"]:
                    n_records_with_nonzero_aabb += 1
                    stream_summary["n_with_nonzero_aabb"] += 1
                if r["centroid_nonzero"]:
                    n_records_with_nonzero_centroid += 1
                    stream_summary["n_with_nonzero_centroid"] += 1
                if r["spatial_native_bit_set"]:
                    n_records_spatial_bit_set += 1
                    stream_summary["n_with_spatial_native_bit"] += 1
                if first_nonzero_record is None and (r["aabb_min_nonzero"] or r["aabb_max_nonzero"]):
                    first_nonzero_record = dict(r)
                    first_nonzero_stream_id = stream_id
                n_records_checked += 1
        stream_summary["schema_version_seen"] = sorted(stream_summary["schema_version_seen"])
        per_stream.append(stream_summary)

    final_status = "GHOST_NATIVE_SPATIAL_MAPPING_PASS"
    if not ghost_files:
        final_status = "GHOST_NATIVE_SPATIAL_MAPPING_BLOCKED_BY_MISSING_CONTACTSHELL_GEOMETRY"
    elif n_records_with_nonzero_aabb == 0:
        final_status = "GHOST_NATIVE_SPATIAL_MAPPING_BLOCKED_BY_MISSING_CONTACTSHELL_GEOMETRY"

    out = {
        "schema_version": 1,
        "schema_kind": "ghost_v2_native_spatial_probe",
        "run_dir": str(run_dir),
        "n_ghost_files": len(ghost_files),
        "n_records_checked": n_records_checked,
        "schema_version_distribution": {str(k): v for k, v in schema_v_dist.items()},
        "n_records_with_nonzero_aabb": n_records_with_nonzero_aabb,
        "n_records_with_nonzero_centroid_alias": n_records_with_nonzero_centroid,
        "n_records_spatial_native_bit_set": n_records_spatial_bit_set,
        "first_nonzero_record": first_nonzero_record,
        "first_nonzero_stream_id": first_nonzero_stream_id,
        "selected_centroid_view": "aabb_midpoint_native_contact_shell_tile",
        "centroid_is_phase_manifold_complete": False,
        "missing_phase_manifold_views": [
            "spike_density",
            "kcc_driver",
            "phasor_coherent",
            "thermo_weighted",
            "ghost_zstr_event_weighted",
        ],
        "centroid_xyz_alias_of": "aabb_midpoint_native_contact_shell_tile",
        "per_stream": per_stream,
        "final_status": final_status,
        "field_offsets_used": {
            "schema_version": OFF_SCHEMA_VERSION,
            "field_completeness_flags": OFF_FIELD_COMPLETENESS_FLAGS,
            "aabb_min": OFF_AABB_MIN,
            "aabb_max": OFF_AABB_MAX,
            "centroid_xyz": OFF_CENTROID,
        },
        "spatial_native_bit_value": GHOST_FCF_BIT_SPATIAL_NATIVE_AABB_MIDPOINT,
    }
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("run_dir", help="Run directory containing *_ghost_tiles.bin")
    p.add_argument("--output", default=None, help="Output JSON path")
    p.add_argument("--max-records-per-file", type=int, default=5000)
    args = p.parse_args()
    rd = Path(args.run_dir)
    out = probe_run(rd, max_records_per_file=args.max_records_per_file)
    out_path = Path(args.output) if args.output else rd / "ghost_v2_native_spatial_probe.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"wrote {out_path}", file=sys.stderr)
    print(json.dumps({k: out.get(k) for k in (
        "n_ghost_files",
        "n_records_checked",
        "schema_version_distribution",
        "n_records_with_nonzero_aabb",
        "n_records_with_nonzero_centroid_alias",
        "n_records_spatial_native_bit_set",
        "final_status",
    )}, indent=2))


if __name__ == "__main__":
    main()
