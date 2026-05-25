#!/usr/bin/env python3
"""Audit GLP-1R stream protocol groups and captured-graph timestep scaling."""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

REPO_ROOT = Path(__file__).resolve().parents[2]
for import_root in (REPO_ROOT, REPO_ROOT / "src"):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from scripts.prism_n80_extraction_common import (
    DEFAULT_OUTPUT_DIR,
    DEFAULT_RAW_ROOT,
    StreamFile,
    discover_stream_files,
    filter_streams,
    json_float,
    json_int,
    json_list,
    json_object,
    parse_stream_selector,
    read_json_object,
)


CAPTURE_KEY_FRAGMENTS = (
    "steps_per_graph_launch",
    "step_per_graph",
    "graph_capture",
    "capture_multiplier",
    "chunk_size",
    "save_interval",
)
REPRESENTATIVE_GROUPS = {0: "A", 5: "B", 10: "C", 15: "D"}


def recursive_capture_keys(value: Any, *, prefix: str = "") -> list[tuple[str, Any]]:
    rows: list[tuple[str, Any]] = []
    if isinstance(value, dict):
        for key, item in value.items():
            path = f"{prefix}.{key}" if prefix else str(key)
            lowered = str(key).lower()
            if any(fragment in lowered for fragment in CAPTURE_KEY_FRAGMENTS):
                rows.append((path, item))
            rows.extend(recursive_capture_keys(item, prefix=path))
    elif isinstance(value, list):
        for idx, item in enumerate(value):
            rows.extend(recursive_capture_keys(item, prefix=f"{prefix}[{idx}]"))
    return rows


def stream_ghost_rows(replica_dir: Path) -> dict[int, dict[str, Any]]:
    ghost = read_json_object(replica_dir / "ghost_time_map.json")
    rows: dict[int, dict[str, Any]] = {}
    for row in json_list(ghost.get("streams")):
        item = json_object(row)
        if item:
            rows[json_int(item.get("stream_id"), -1)] = item
    return rows


def protocol_signature(protocol: dict[str, Any]) -> tuple[Any, ...]:
    return (
        json_float(protocol.get("dt_ps")),
        json_int(protocol.get("total_steps")),
        json_float(protocol.get("start_temp_K")),
        json_float(protocol.get("end_temp_K")),
        json_int(protocol.get("cold_hold_end")),
        json_int(protocol.get("ramp_end")),
        json_int(protocol.get("warm_hold_end")),
        json_int(protocol.get("ramp_down_end")),
        json_int(protocol.get("uv_burst_interval_steps")),
        json_int(protocol.get("uv_burst_duration_steps")),
        json_int(protocol.get("wavelength_dwell_steps")),
    )


def representative_group_map(protocols: dict[int, dict[str, Any]]) -> dict[tuple[Any, ...], str]:
    out: dict[tuple[Any, ...], str] = {}
    for stream_id, group in REPRESENTATIVE_GROUPS.items():
        protocol = protocols.get(stream_id)
        if protocol is not None:
            out[protocol_signature(protocol)] = group
    return out


def parquet_max_timesteps(path: Path, *, condition_id: str, replica_id: int, stream_ids: set[int]) -> dict[int, int]:
    pf = pq.ParquetFile(path)
    names = pf.schema_arrow.names
    idx = {name: names.index(name) for name in ("condition_id", "replica_id", "stream_id", "timestep")}
    max_by_stream: dict[int, int] = {}
    for row_group_idx in range(pf.metadata.num_row_groups):
        row_group = pf.metadata.row_group(row_group_idx)
        condition_stats = row_group.column(idx["condition_id"]).statistics
        replica_stats = row_group.column(idx["replica_id"]).statistics
        stream_stats = row_group.column(idx["stream_id"]).statistics
        timestep_stats = row_group.column(idx["timestep"]).statistics
        if not all((condition_stats, replica_stats, stream_stats, timestep_stats)):
            continue
        if condition_stats.min != condition_id or condition_stats.max != condition_id:
            continue
        if int(replica_stats.min) != replica_id or int(replica_stats.max) != replica_id:
            continue
        if int(stream_stats.min) != int(stream_stats.max):
            continue
        stream_id = int(stream_stats.min)
        if stream_id not in stream_ids:
            continue
        max_by_stream[stream_id] = max(max_by_stream.get(stream_id, -1), int(timestep_stats.max))
    missing = sorted(stream_ids - set(max_by_stream))
    if missing:
        raise ValueError(f"{path}: no exact row-group timestep stats for stream(s) {missing}")
    return max_by_stream


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", type=Path, default=DEFAULT_RAW_ROOT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--condition-id", default="glp1r_5VEX_WT")
    parser.add_argument("--replica-id", type=int, default=0)
    parser.add_argument("--streams", default="0,5,10,15")
    parser.add_argument("--spike-parquet", type=Path, default=DEFAULT_OUTPUT_DIR / "spike_events_snr_masked.parquet")
    args = parser.parse_args()

    selected = filter_streams(
        discover_stream_files(args.raw_root, "protocol_state.json"),
        condition_id=args.condition_id,
        replica_id=args.replica_id,
        stream_ids=parse_stream_selector(args.streams),
        max_streams=None,
    )
    if not selected:
        raise ValueError("no protocol_state.json files selected")
    replica_dir = selected[0].path.parent
    all_protocol_files = filter_streams(
        discover_stream_files(args.raw_root, "protocol_state.json"),
        condition_id=args.condition_id,
        replica_id=args.replica_id,
        stream_ids=None,
        max_streams=None,
    )
    ghost_rows = stream_ghost_rows(replica_dir)
    protocols: dict[int, dict[str, Any]] = {int(item.stream_id): read_json_object(item.path) for item in all_protocol_files}
    groups = representative_group_map(protocols)
    streams_by_group: dict[str, list[int]] = defaultdict(list)
    for stream_id, protocol in sorted(protocols.items()):
        group = groups.get(protocol_signature(protocol), "?")
        streams_by_group[group].append(stream_id)

    requested_streams = {int(item.stream_id) for item in selected}
    parquet_max = parquet_max_timesteps(
        args.spike_parquet,
        condition_id=args.condition_id,
        replica_id=args.replica_id,
        stream_ids=requested_streams,
    )

    capture_hits: list[tuple[int, str, Any]] = []
    for item in all_protocol_files:
        stream_id = int(item.stream_id)
        for key, value in recursive_capture_keys(protocols[stream_id]):
            capture_hits.append((stream_id, f"protocol_state.json:{key}", value))
    ghost = read_json_object(replica_dir / "ghost_time_map.json")
    for key, value in recursive_capture_keys(ghost):
        capture_hits.append((-1, f"ghost_time_map.json:{key}", value))

    print("GROUP_SIGNATURES source=protocol_signature_representative_map")
    for group in ("A", "B", "C", "D", "?"):
        if group in streams_by_group:
            print(f"Group {group}: streams={streams_by_group[group]}")
    print("CAPTURE_KEY_SEARCH")
    if capture_hits:
        for stream_id, key, value in capture_hits:
            stream_label = "all" if stream_id < 0 else str(stream_id)
            print(f"stream={stream_label} {key}={value}")
    else:
        print("explicit_capture_key=NOT_FOUND")
    print("STREAM_GROUP_AUDIT")
    for item in sorted(selected, key=lambda file: int(file.stream_id)):
        stream_id = int(item.stream_id)
        protocol = protocols[stream_id]
        ghost_row = ghost_rows.get(stream_id, {})
        observed_max = json_int(ghost_row.get("timestep_max"), json_int(protocol.get("current_step")))
        total_steps = json_int(protocol.get("total_steps"))
        multiplier = total_steps / observed_max if observed_max else 1.0
        group = groups.get(protocol_signature(protocol), "?")
        print(
            f"stream={stream_id} group={group} dt_ps={json_float(protocol.get('dt_ps')):.12g} "
            f"total_steps={total_steps} protocol_current_step={json_int(protocol.get('current_step'))} "
            f"ghost_timestep_max={observed_max} derived_capture_multiplier={multiplier:.12g}"
        )
    print("PARQUET_COMPLETION_MATH")
    for stream_id in sorted(requested_streams):
        protocol = protocols[stream_id]
        ghost_row = ghost_rows.get(stream_id, {})
        observed_max = json_int(ghost_row.get("timestep_max"), json_int(protocol.get("current_step")))
        total_steps = json_int(protocol.get("total_steps"))
        multiplier = total_steps / observed_max if observed_max else 1.0
        parquet_max_step = parquet_max[stream_id]
        scaled = parquet_max_step * multiplier
        pct = (scaled / total_steps * 100.0) if total_steps else 0.0
        print(
            f"Stream {stream_id}: max_timestep {parquet_max_step} * "
            f"{multiplier:.12g} steps/timestep = {scaled:.3f} MD steps; "
            f"total_steps={total_steps}; {pct:.6f}% masked-parquet coverage"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
