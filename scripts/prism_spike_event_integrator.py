#!/usr/bin/env python3
"""Decode PRISM PRSPK001 spike streams into typed Parquet event layers.

The integrator is intentionally ontology-preserving:

* raw spike event fields stay raw event fields;
* residue mapping is added as an annotation layer;
* materialized-site assignment is added as a projection layer;
* SAR interface hits are emitted as an optional long table.

This is the first durable bridge from Prism4D MD evidence into a temporal
mechanistic event surface suitable for downstream interface timestamp mining.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


SPIKE_DTYPE = np.dtype(
    [
        ("timestep", "<i4"),
        ("voxel_idx", "<i4"),
        ("position", "<f4", (3,)),
        ("intensity", "<f4"),
        ("nearby_residues", "<i4", (8,)),
        ("n_residues", "<i4"),
        ("spike_source", "<i4"),
        ("wavelength_nm", "<f4"),
        ("aromatic_type", "<i4"),
        ("aromatic_residue_id", "<i4"),
        ("water_density", "<f4"),
        ("vibrational_energy", "<f4"),
        ("n_nearby_excited", "<i4"),
        ("wd_change", "<f4"),
        ("phase_bits", "<u4"),
    ],
    align=False,
)

if SPIKE_DTYPE.itemsize != 96:
    raise RuntimeError(f"GpuSpikeEvent dtype must be 96 bytes, got {SPIKE_DTYPE.itemsize}")


EVENT_SCHEMA = pa.schema(
    [
        ("campaign_id", pa.string()),
        ("run_label", pa.string()),
        ("run_id", pa.string()),
        ("stem", pa.string()),
        ("structure_id", pa.string()),
        ("stream_id", pa.uint8()),
        ("event_index_in_stream", pa.int64()),
        ("spike_id", pa.string()),
        ("timestep", pa.int32()),
        ("physical_time_fs", pa.float64()),
        ("physical_time_ps", pa.float64()),
        ("voxel_idx", pa.int32()),
        ("x", pa.float32()),
        ("y", pa.float32()),
        ("z", pa.float32()),
        ("intensity", pa.float32()),
        ("spike_source", pa.int32()),
        ("mechanism_tag", pa.string()),
        ("wavelength_nm", pa.float32()),
        ("aromatic_type", pa.int32()),
        ("aromatic_residue_id", pa.int32()),
        ("water_density", pa.float32()),
        ("wd_change", pa.float32()),
        ("vibrational_energy", pa.float32()),
        ("n_nearby_excited", pa.int32()),
        ("phase_bits", pa.uint32()),
        ("n_residues", pa.int32()),
        *[(f"nearby_topology_residue_{i}", pa.int32()) for i in range(8)],
        *[(f"nearby_uniprot_residue_{i}", pa.int32()) for i in range(8)],
        ("primary_topology_residue", pa.int32()),
        ("primary_uniprot_residue", pa.int32()),
        ("nearest_site_id", pa.string()),
        ("nearest_site_rank", pa.int32()),
        ("nearest_site_distance_a", pa.float32()),
        ("nearest_site_radius_a", pa.float32()),
        ("inside_nearest_site_radius", pa.bool_()),
        ("best_interface_id", pa.string()),
        ("best_interface_class", pa.string()),
        ("best_interface_match_basis", pa.string()),
        ("best_interface_te_coupling_score", pa.float64()),
        ("best_interface_score", pa.float64()),
    ]
)

INTERFACE_HIT_SCHEMA = pa.schema(
    [
        ("campaign_id", pa.string()),
        ("run_label", pa.string()),
        ("run_id", pa.string()),
        ("structure_id", pa.string()),
        ("stream_id", pa.uint8()),
        ("event_index_in_stream", pa.int64()),
        ("spike_id", pa.string()),
        ("timestep", pa.int32()),
        ("physical_time_ps", pa.float64()),
        ("interface_id", pa.string()),
        ("interface_class", pa.string()),
        ("match_basis", pa.string()),
        ("target_hinge_residue_index", pa.int32()),
        ("neighbor_residue_index", pa.int32()),
        ("nearest_materialized_pocket_site_id", pa.string()),
        ("te_coupling_score", pa.float64()),
        ("lock_interface_score", pa.float64()),
        ("x", pa.float32()),
        ("y", pa.float32()),
        ("z", pa.float32()),
        ("intensity", pa.float32()),
        ("water_density", pa.float32()),
        ("wd_change", pa.float32()),
        ("phase_bits", pa.uint32()),
    ]
)


@dataclass(frozen=True)
class EnvelopeHeader:
    path: Path
    magic: str
    schema_version: int
    endian_marker: int
    stream_id: int
    run_id: str
    stem: str
    record_count: int
    byte_stride: int
    payload_size: int
    payload_offset: int


@dataclass(frozen=True)
class MaterializedSite:
    site_id: str
    rank: int
    centroid: tuple[float, float, float]
    radius_a: float
    lining_uniprot_residues: tuple[int, ...]
    driver_uniprot_residues: tuple[int, ...]


@dataclass(frozen=True)
class InterfaceRow:
    interface_id: str
    interface_class: str
    target_hinge_residue_index: int
    neighbor_residue_index: int
    nearest_materialized_pocket_site_id: str
    te_coupling_score: float
    lock_interface_score: float


def parse_envelope(path: Path) -> EnvelopeHeader:
    with path.open("rb") as fh:
        magic = fh.read(8).decode("ascii", errors="replace")
        if magic != "PRSPK001":
            raise ValueError(f"{path}: expected PRSPK001, got {magic!r}")
        schema_version, endian_marker, stream_id = struct.unpack("<III", fh.read(12))
        run_len = struct.unpack("<Q", fh.read(8))[0]
        run_id = fh.read(run_len).decode("utf-8", errors="replace")
        stem_len = struct.unpack("<Q", fh.read(8))[0]
        stem = fh.read(stem_len).decode("utf-8", errors="replace")
        record_count, byte_stride, payload_size = struct.unpack("<QQQ", fh.read(24))
        payload_offset = fh.tell()
    if endian_marker != 0x01020304:
        raise ValueError(f"{path}: unsupported endian marker {hex(endian_marker)}")
    if byte_stride != 96:
        raise ValueError(f"{path}: expected 96-byte spike stride, got {byte_stride}")
    return EnvelopeHeader(
        path=path,
        magic=magic,
        schema_version=schema_version,
        endian_marker=endian_marker,
        stream_id=stream_id,
        run_id=run_id,
        stem=stem,
        record_count=record_count,
        byte_stride=byte_stride,
        payload_size=payload_size,
        payload_offset=payload_offset,
    )


def read_json(path: Path) -> Any:
    return json.loads(path.read_text())


def load_dt_fs_by_stream(path: Path) -> dict[int, float]:
    data = read_json(path)
    out: dict[int, float] = {}
    for row in data.get("streams", []):
        sid = int(row.get("stream_id", row.get("stream", 0)))
        out[sid] = float(row.get("dt_fs", 4.0))
    return out


def load_residue_maps(
    residue_map_path: Path,
    *,
    structure_id: str,
    mapping_parquet: Path | None,
) -> tuple[np.ndarray, dict[int, dict[str, Any]]]:
    data = read_json(residue_map_path)
    rows = data.get("residues", [])
    if not rows:
        raise ValueError(f"{residue_map_path}: no residues[] rows found")

    pdb_chain_res_to_uniprot: dict[tuple[str, str], int] = {}
    if mapping_parquet and mapping_parquet.exists():
        table = pq.read_table(
            mapping_parquet,
            columns=[
                "pdb_id",
                "auth_asym_id",
                "auth_seq_id",
                "uniprot_residue_index",
                "is_target_uniprot",
            ],
        )
        cols = table.to_pydict()
        for pdb_id, chain, auth_seq, uniprot, is_target in zip(
            cols["pdb_id"],
            cols["auth_asym_id"],
            cols["auth_seq_id"],
            cols["uniprot_residue_index"],
            cols["is_target_uniprot"],
        ):
            if str(pdb_id).upper() != structure_id.upper() or not bool(is_target):
                continue
            try:
                pdb_chain_res_to_uniprot[(str(chain), str(auth_seq))] = int(uniprot)
            except (TypeError, ValueError):
                continue

    max_topo = max(int(r["topology_index"]) for r in rows)
    topo_to_uniprot = np.full(max_topo + 1, -1, dtype=np.int32)
    meta: dict[int, dict[str, Any]] = {}
    for row in rows:
        topo = int(row["topology_index"])
        chain = str(row.get("chain", ""))
        pdb_resid = int(row["pdb_resid"])
        uniprot = pdb_chain_res_to_uniprot.get((chain, str(pdb_resid)), pdb_resid)
        topo_to_uniprot[topo] = int(uniprot)
        meta[topo] = {
            "chain": chain,
            "pdb_resid": pdb_resid,
            "uniprot_residue_index": int(uniprot),
            "resname": row.get("resname"),
        }
    return topo_to_uniprot, meta


def load_sites(path: Path, topo_to_uniprot: np.ndarray) -> list[MaterializedSite]:
    data = read_json(path)
    sites: list[MaterializedSite] = []
    for idx, site in enumerate(data.get("binding_sites", [])):
        centroid = site.get("centroid_xyz") or [math.nan, math.nan, math.nan]
        rank = int(site.get("new_prism_rank", site.get("rank", idx)))
        radius = float(site.get("region_radius_a") or site.get("centroid_spread_a") or 0.0)
        radius = max(radius, float(site.get("centroid_spread_a") or 0.0), 0.0)
        lining = topology_residues_to_uniprot(site.get("lining_residues") or [], topo_to_uniprot)
        drivers = topology_residues_to_uniprot(site.get("driver_residues") or [], topo_to_uniprot)
        sites.append(
            MaterializedSite(
                site_id=str(site.get("site_id", f"site_{idx:03d}")),
                rank=rank,
                centroid=(float(centroid[0]), float(centroid[1]), float(centroid[2])),
                radius_a=radius,
                lining_uniprot_residues=tuple(lining),
                driver_uniprot_residues=tuple(drivers),
            )
        )
    if not sites:
        raise ValueError(f"{path}: no materialized binding sites")
    return sites


def topology_residues_to_uniprot(values: Iterable[Any], topo_to_uniprot: np.ndarray) -> list[int]:
    out: list[int] = []
    for value in values:
        try:
            topo = int(value)
        except (TypeError, ValueError):
            continue
        if 0 <= topo < len(topo_to_uniprot):
            mapped = int(topo_to_uniprot[topo])
            if mapped >= 0:
                out.append(mapped)
    return out


def load_interfaces(path: Path) -> list[InterfaceRow]:
    table = pq.read_table(path)
    rows = table.to_pylist()
    out: list[InterfaceRow] = []
    for row in rows:
        target = int(row["target_hinge_residue_index"])
        neighbor = int(row["neighbor_residue_index"])
        interface_id = f"{row['target_hinge_label']}__{row['neighbor_label']}"
        out.append(
            InterfaceRow(
                interface_id=interface_id,
                interface_class=str(row.get("pocket_accessibility_class") or ""),
                target_hinge_residue_index=target,
                neighbor_residue_index=neighbor,
                nearest_materialized_pocket_site_id=str(
                    row.get("nearest_materialized_pocket_site_id") or ""
                ),
                te_coupling_score=float(row.get("te_coupling_score") or 0.0),
                lock_interface_score=float(row.get("lock_interface_score") or 0.0),
            )
        )
    return out


def mechanism_tags(codes: np.ndarray, aromatic_type: np.ndarray) -> np.ndarray:
    out = np.empty(codes.shape[0], dtype=object)
    out[:] = "LIF_LOCAL_INTENSITY"
    out[codes == 1] = "UV_AROMATIC_PERTURBATION"
    out[codes == 3] = "EFP_ELECTROSTATIC_FIELD"
    out[codes == 4] = "LADD_ATOM_DEPARTURE"
    out[codes == 5] = "COFIRE_COHERENCE"
    out[(codes != 1) & (codes != 3) & (codes != 4) & (codes != 5) & (aromatic_type < 0)] = (
        "LIF_THERMAL_SHAPE"
    )
    return out


def map_nearby_residues(events: np.ndarray, topo_to_uniprot: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    nearby = events["nearby_residues"].astype(np.int32, copy=True)
    n_res = np.clip(events["n_residues"].astype(np.int32), 0, 8)
    valid_slots = np.arange(8, dtype=np.int32)[None, :] < n_res[:, None]
    topo = np.where(valid_slots, nearby, -1).astype(np.int32)
    uniprot = np.full(topo.shape, -1, dtype=np.int32)
    valid = (topo >= 0) & (topo < len(topo_to_uniprot))
    uniprot[valid] = topo_to_uniprot[topo[valid]]
    return topo, uniprot


def annotate_sites(
    positions: np.ndarray, sites: list[MaterializedSite]
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    centroids = np.array([s.centroid for s in sites], dtype=np.float32)
    radii = np.array([s.radius_a for s in sites], dtype=np.float32)
    diff = positions[:, None, :].astype(np.float32) - centroids[None, :, :]
    dist = np.sqrt(np.sum(diff * diff, axis=2, dtype=np.float32))
    nearest = np.argmin(dist, axis=1).astype(np.int32)
    nearest_dist = dist[np.arange(len(positions)), nearest].astype(np.float32)
    nearest_radius = radii[nearest].astype(np.float32)
    inside = nearest_dist <= nearest_radius
    nearest_rank = np.array([sites[i].rank for i in nearest], dtype=np.int32)
    nearest_site_id = np.array([sites[i].site_id for i in nearest], dtype=object)
    return nearest_site_id, nearest_rank, nearest_dist, nearest_radius, inside


def annotate_best_interfaces(
    nearby_uniprot: np.ndarray,
    nearest_site_id: np.ndarray,
    interfaces: list[InterfaceRow],
) -> dict[str, np.ndarray]:
    n = nearby_uniprot.shape[0]
    best_priority = np.zeros(n, dtype=np.int8)
    best_abs_score = np.zeros(n, dtype=np.float64)
    best_interface_id = np.full(n, "", dtype=object)
    best_class = np.full(n, "", dtype=object)
    best_basis = np.full(n, "", dtype=object)
    best_te = np.full(n, np.nan, dtype=np.float64)
    best_score = np.full(n, np.nan, dtype=np.float64)

    for iface in interfaces:
        target_present = np.any(nearby_uniprot == iface.target_hinge_residue_index, axis=1)
        neighbor_present = np.any(nearby_uniprot == iface.neighbor_residue_index, axis=1)
        pair = target_present & neighbor_present
        single = target_present ^ neighbor_present
        site = (
            nearest_site_id == iface.nearest_materialized_pocket_site_id
            if iface.nearest_materialized_pocket_site_id
            else np.zeros(n, dtype=bool)
        )
        priority = np.zeros(n, dtype=np.int8)
        basis = np.full(n, "", dtype=object)
        priority[site] = 1
        basis[site] = "site_shell"
        priority[single] = 2
        basis[single] = "residue_single"
        priority[pair] = 3
        basis[pair] = "residue_pair"
        abs_score = abs(iface.te_coupling_score)
        update = (priority > best_priority) | (
            (priority == best_priority) & (priority > 0) & (abs_score > best_abs_score)
        )
        best_priority[update] = priority[update]
        best_abs_score[update] = abs_score
        best_interface_id[update] = iface.interface_id
        best_class[update] = iface.interface_class
        best_basis[update] = basis[update]
        best_te[update] = iface.te_coupling_score
        best_score[update] = iface.lock_interface_score

    return {
        "best_interface_id": best_interface_id,
        "best_interface_class": best_class,
        "best_interface_match_basis": best_basis,
        "best_interface_te_coupling_score": best_te,
        "best_interface_score": best_score,
    }


def interface_hit_table(
    *,
    campaign_id: str,
    run_label: str,
    run_id: str,
    structure_id: str,
    stream_id: int,
    event_indices: np.ndarray,
    spike_ids: np.ndarray,
    events: np.ndarray,
    physical_time_ps: np.ndarray,
    positions: np.ndarray,
    nearby_uniprot: np.ndarray,
    nearest_site_id: np.ndarray,
    interfaces: list[InterfaceRow],
) -> pa.Table | None:
    chunks: list[dict[str, Any]] = []
    for iface in interfaces:
        target_present = np.any(nearby_uniprot == iface.target_hinge_residue_index, axis=1)
        neighbor_present = np.any(nearby_uniprot == iface.neighbor_residue_index, axis=1)
        pair = target_present & neighbor_present
        single = target_present ^ neighbor_present
        site = (
            nearest_site_id == iface.nearest_materialized_pocket_site_id
            if iface.nearest_materialized_pocket_site_id
            else np.zeros(len(events), dtype=bool)
        )
        mask = pair | single | site
        if not np.any(mask):
            continue
        basis = np.full(len(events), "site_shell", dtype=object)
        basis[single] = "residue_single"
        basis[pair] = "residue_pair"
        idx = np.where(mask)[0]
        chunks.append(
            {
                "campaign_id": np.full(len(idx), campaign_id, dtype=object),
                "run_label": np.full(len(idx), run_label, dtype=object),
                "run_id": np.full(len(idx), run_id, dtype=object),
                "structure_id": np.full(len(idx), structure_id, dtype=object),
                "stream_id": np.full(len(idx), stream_id, dtype=np.uint8),
                "event_index_in_stream": event_indices[idx].astype(np.int64),
                "spike_id": spike_ids[idx],
                "timestep": events["timestep"][idx].astype(np.int32),
                "physical_time_ps": physical_time_ps[idx].astype(np.float64),
                "interface_id": np.full(len(idx), iface.interface_id, dtype=object),
                "interface_class": np.full(len(idx), iface.interface_class, dtype=object),
                "match_basis": basis[idx],
                "target_hinge_residue_index": np.full(
                    len(idx), iface.target_hinge_residue_index, dtype=np.int32
                ),
                "neighbor_residue_index": np.full(
                    len(idx), iface.neighbor_residue_index, dtype=np.int32
                ),
                "nearest_materialized_pocket_site_id": np.full(
                    len(idx), iface.nearest_materialized_pocket_site_id, dtype=object
                ),
                "te_coupling_score": np.full(len(idx), iface.te_coupling_score, dtype=np.float64),
                "lock_interface_score": np.full(
                    len(idx), iface.lock_interface_score, dtype=np.float64
                ),
                "x": positions[idx, 0].astype(np.float32),
                "y": positions[idx, 1].astype(np.float32),
                "z": positions[idx, 2].astype(np.float32),
                "intensity": events["intensity"][idx].astype(np.float32),
                "water_density": events["water_density"][idx].astype(np.float32),
                "wd_change": events["wd_change"][idx].astype(np.float32),
                "phase_bits": events["phase_bits"][idx].astype(np.uint32),
            }
        )
    if not chunks:
        return None
    arrays: dict[str, list[np.ndarray]] = {name: [] for name in INTERFACE_HIT_SCHEMA.names}
    for chunk in chunks:
        for name in INTERFACE_HIT_SCHEMA.names:
            arrays[name].append(chunk[name])
    merged = {name: np.concatenate(parts) for name, parts in arrays.items()}
    return pa.Table.from_pydict(merged, schema=INTERFACE_HIT_SCHEMA)


def event_table(
    *,
    campaign_id: str,
    run_label: str,
    header: EnvelopeHeader,
    structure_id: str,
    event_indices: np.ndarray,
    events: np.ndarray,
    dt_fs: float,
    nearby_topo: np.ndarray,
    nearby_uniprot: np.ndarray,
    nearest_site_id: np.ndarray,
    nearest_site_rank: np.ndarray,
    nearest_site_distance: np.ndarray,
    nearest_site_radius: np.ndarray,
    inside_site: np.ndarray,
    best_interfaces: dict[str, np.ndarray],
) -> tuple[pa.Table, np.ndarray, np.ndarray, np.ndarray]:
    n = len(events)
    positions = events["position"].astype(np.float32)
    physical_time_fs = events["timestep"].astype(np.float64) * float(dt_fs)
    physical_time_ps = physical_time_fs / 1000.0
    spike_ids = np.array(
        [f"{header.run_id}:s{header.stream_id}:e{int(i)}" for i in event_indices], dtype=object
    )
    n_res = np.clip(events["n_residues"].astype(np.int32), 0, 8)
    primary_topo = nearby_topo[:, 0].astype(np.int32)
    primary_uniprot = nearby_uniprot[:, 0].astype(np.int32)

    data: dict[str, Any] = {
        "campaign_id": np.full(n, campaign_id, dtype=object),
        "run_label": np.full(n, run_label, dtype=object),
        "run_id": np.full(n, header.run_id, dtype=object),
        "stem": np.full(n, header.stem, dtype=object),
        "structure_id": np.full(n, structure_id, dtype=object),
        "stream_id": np.full(n, header.stream_id, dtype=np.uint8),
        "event_index_in_stream": event_indices.astype(np.int64),
        "spike_id": spike_ids,
        "timestep": events["timestep"].astype(np.int32),
        "physical_time_fs": physical_time_fs.astype(np.float64),
        "physical_time_ps": physical_time_ps.astype(np.float64),
        "voxel_idx": events["voxel_idx"].astype(np.int32),
        "x": positions[:, 0].astype(np.float32),
        "y": positions[:, 1].astype(np.float32),
        "z": positions[:, 2].astype(np.float32),
        "intensity": events["intensity"].astype(np.float32),
        "spike_source": events["spike_source"].astype(np.int32),
        "mechanism_tag": mechanism_tags(events["spike_source"], events["aromatic_type"]),
        "wavelength_nm": events["wavelength_nm"].astype(np.float32),
        "aromatic_type": events["aromatic_type"].astype(np.int32),
        "aromatic_residue_id": events["aromatic_residue_id"].astype(np.int32),
        "water_density": events["water_density"].astype(np.float32),
        "wd_change": events["wd_change"].astype(np.float32),
        "vibrational_energy": events["vibrational_energy"].astype(np.float32),
        "n_nearby_excited": events["n_nearby_excited"].astype(np.int32),
        "phase_bits": events["phase_bits"].astype(np.uint32),
        "n_residues": n_res,
        "primary_topology_residue": primary_topo,
        "primary_uniprot_residue": primary_uniprot,
        "nearest_site_id": nearest_site_id,
        "nearest_site_rank": nearest_site_rank,
        "nearest_site_distance_a": nearest_site_distance.astype(np.float32),
        "nearest_site_radius_a": nearest_site_radius.astype(np.float32),
        "inside_nearest_site_radius": inside_site.astype(bool),
        **best_interfaces,
    }
    for i in range(8):
        data[f"nearby_topology_residue_{i}"] = nearby_topo[:, i].astype(np.int32)
        data[f"nearby_uniprot_residue_{i}"] = nearby_uniprot[:, i].astype(np.int32)

    return pa.Table.from_pydict(data, schema=EVENT_SCHEMA), spike_ids, physical_time_ps, positions


def open_writer(path: Path, schema: pa.Schema) -> pq.ParquetWriter:
    path.parent.mkdir(parents=True, exist_ok=True)
    return pq.ParquetWriter(
        path,
        schema,
        compression="zstd",
        compression_level=6,
        use_dictionary=True,
    )


def integrate_stream(
    *,
    spike_path: Path,
    out_dir: Path,
    campaign_id: str,
    run_label: str,
    structure_id: str,
    topo_to_uniprot: np.ndarray,
    sites: list[MaterializedSite],
    interfaces: list[InterfaceRow],
    dt_fs: float,
    chunk_rows: int,
    start_record: int,
    max_records: int | None,
    emit_interface_hits: bool,
) -> dict[str, Any]:
    header = parse_envelope(spike_path)
    if start_record < 0 or start_record >= header.record_count:
        raise ValueError(
            f"{spike_path}: start_record={start_record} outside record_count={header.record_count}"
        )
    available = header.record_count - start_record
    limit = min(available, max_records) if max_records else available
    event_path = out_dir / f"{run_label}_stream{header.stream_id:02d}_spike_events.parquet"
    hit_path = out_dir / f"{run_label}_stream{header.stream_id:02d}_interface_hits.parquet"
    event_writer = open_writer(event_path, EVENT_SCHEMA)
    hit_writer = open_writer(hit_path, INTERFACE_HIT_SCHEMA) if emit_interface_hits else None

    written = 0
    interface_hits = 0
    best_interface_counts: dict[str, int] = {}
    site_counts: dict[str, int] = {}
    basis_counts: dict[str, int] = {}
    try:
        with spike_path.open("rb") as fh:
            fh.seek(header.payload_offset + start_record * header.byte_stride)
            while written < limit:
                n = min(chunk_rows, limit - written)
                raw = fh.read(n * header.byte_stride)
                if len(raw) != n * header.byte_stride:
                    raise EOFError(
                        f"{spike_path}: short read at event {written}, expected {n * header.byte_stride}, got {len(raw)}"
                    )
                events = np.frombuffer(raw, dtype=SPIKE_DTYPE, count=n)
                event_indices = np.arange(
                    start_record + written, start_record + written + n, dtype=np.int64
                )
                nearby_topo, nearby_uniprot = map_nearby_residues(events, topo_to_uniprot)
                nearest_site_id, nearest_site_rank, nearest_dist, nearest_radius, inside = (
                    annotate_sites(events["position"], sites)
                )
                best = annotate_best_interfaces(nearby_uniprot, nearest_site_id, interfaces)
                table, spike_ids, physical_time_ps, positions = event_table(
                    campaign_id=campaign_id,
                    run_label=run_label,
                    header=header,
                    structure_id=structure_id,
                    event_indices=event_indices,
                    events=events,
                    dt_fs=dt_fs,
                    nearby_topo=nearby_topo,
                    nearby_uniprot=nearby_uniprot,
                    nearest_site_id=nearest_site_id,
                    nearest_site_rank=nearest_site_rank,
                    nearest_site_distance=nearest_dist,
                    nearest_site_radius=nearest_radius,
                    inside_site=inside,
                    best_interfaces=best,
                )
                event_writer.write_table(table)

                for value, count in zip(*np.unique(best["best_interface_id"], return_counts=True)):
                    if value:
                        best_interface_counts[str(value)] = best_interface_counts.get(str(value), 0) + int(count)
                for value, count in zip(*np.unique(nearest_site_id, return_counts=True)):
                    site_counts[str(value)] = site_counts.get(str(value), 0) + int(count)
                for value, count in zip(*np.unique(best["best_interface_match_basis"], return_counts=True)):
                    if value:
                        basis_counts[str(value)] = basis_counts.get(str(value), 0) + int(count)

                if hit_writer is not None:
                    hit_table = interface_hit_table(
                        campaign_id=campaign_id,
                        run_label=run_label,
                        run_id=header.run_id,
                        structure_id=structure_id,
                        stream_id=header.stream_id,
                        event_indices=event_indices,
                        spike_ids=spike_ids,
                        events=events,
                        physical_time_ps=physical_time_ps,
                        positions=positions,
                        nearby_uniprot=nearby_uniprot,
                        nearest_site_id=nearest_site_id,
                        interfaces=interfaces,
                    )
                    if hit_table is not None and hit_table.num_rows:
                        hit_writer.write_table(hit_table)
                        interface_hits += hit_table.num_rows
                written += n
    finally:
        event_writer.close()
        if hit_writer is not None:
            hit_writer.close()

    return {
        "spike_path": str(spike_path),
        "event_path": str(event_path),
        "interface_hit_path": str(hit_path) if emit_interface_hits else None,
        "run_id": header.run_id,
        "stem": header.stem,
        "stream_id": header.stream_id,
        "source_record_count": header.record_count,
        "source_start_record": start_record,
        "records_written": written,
        "dt_fs": dt_fs,
        "interface_hit_rows": interface_hits if emit_interface_hits else None,
        "best_interface_counts": best_interface_counts,
        "nearest_site_counts": site_counts,
        "best_interface_match_basis_counts": basis_counts,
    }


def discover_spike_paths(run_dir: Path, streams: list[int] | None) -> list[Path]:
    all_paths = sorted(run_dir.glob("*_stream*_spikes.bin"))
    if streams is None:
        return all_paths
    wanted = {int(s) for s in streams}
    out = []
    for path in all_paths:
        match = re.search(r"_stream(\d+)_spikes\.bin$", path.name)
        if match and int(match.group(1)) in wanted:
            out.append(path)
    return out


def parse_streams(value: str | None) -> list[int] | None:
    if not value:
        return None
    return [int(x.strip()) for x in value.split(",") if x.strip()]


def write_site_catalog(out_dir: Path, run_label: str, sites: list[MaterializedSite]) -> Path:
    path = out_dir / f"{run_label}_materialized_site_catalog.parquet"
    table = pa.Table.from_pylist(
        [
            {
                "site_id": s.site_id,
                "rank": s.rank,
                "centroid_x": s.centroid[0],
                "centroid_y": s.centroid[1],
                "centroid_z": s.centroid[2],
                "radius_a": s.radius_a,
                "lining_uniprot_residues": list(s.lining_uniprot_residues),
                "driver_uniprot_residues": list(s.driver_uniprot_residues),
            }
            for s in sites
        ]
    )
    pq.write_table(table, path, compression="zstd", compression_level=6)
    return path


def write_interface_catalog(out_dir: Path, interfaces: list[InterfaceRow]) -> Path:
    path = out_dir / "sar_steric_interface_catalog.parquet"
    table = pa.Table.from_pylist([iface.__dict__ for iface in interfaces])
    pq.write_table(table, path, compression="zstd", compression_level=6)
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-id", required=True)
    parser.add_argument("--run-label", required=True)
    parser.add_argument("--structure-id", required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--residue-map", type=Path, required=True)
    parser.add_argument("--residue-mapping-parquet", type=Path)
    parser.add_argument("--binding-sites", type=Path, required=True)
    parser.add_argument("--interfaces", type=Path, required=True)
    parser.add_argument("--ghost-time-map", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--streams", help="Comma-separated stream ids. Default: all streams.")
    parser.add_argument("--chunk-rows", type=int, default=100_000)
    parser.add_argument(
        "--start-record",
        type=int,
        default=0,
        help="Zero-based event offset inside each stream. Useful for temporal smoke windows.",
    )
    parser.add_argument("--max-records-per-stream", type=int)
    parser.add_argument("--emit-interface-hits", action="store_true")
    args = parser.parse_args()

    streams = parse_streams(args.streams)
    spike_paths = discover_spike_paths(args.run_dir, streams)
    if not spike_paths:
        raise SystemExit(f"No spike streams found in {args.run_dir} for streams={streams}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    topo_to_uniprot, residue_meta = load_residue_maps(
        args.residue_map,
        structure_id=args.structure_id,
        mapping_parquet=args.residue_mapping_parquet,
    )
    sites = load_sites(args.binding_sites, topo_to_uniprot)
    interfaces = load_interfaces(args.interfaces)
    dt_by_stream = load_dt_fs_by_stream(args.ghost_time_map)
    site_catalog = write_site_catalog(args.out_dir, args.run_label, sites)
    interface_catalog = write_interface_catalog(args.out_dir, interfaces)

    results = []
    for spike_path in spike_paths:
        header = parse_envelope(spike_path)
        dt_fs = dt_by_stream.get(header.stream_id, 4.0)
        result = integrate_stream(
            spike_path=spike_path,
            out_dir=args.out_dir,
            campaign_id=args.campaign_id,
            run_label=args.run_label,
            structure_id=args.structure_id,
            topo_to_uniprot=topo_to_uniprot,
            sites=sites,
            interfaces=interfaces,
            dt_fs=dt_fs,
            chunk_rows=args.chunk_rows,
            start_record=args.start_record,
            max_records=args.max_records_per_stream,
            emit_interface_hits=args.emit_interface_hits,
        )
        results.append(result)

    manifest = {
        "schema": "prism_spike_event_temporal_integration.v1",
        "campaign_id": args.campaign_id,
        "run_label": args.run_label,
        "structure_id": args.structure_id,
        "run_dir": str(args.run_dir),
        "residue_map": str(args.residue_map),
        "residue_mapping_parquet": str(args.residue_mapping_parquet)
        if args.residue_mapping_parquet
        else None,
        "binding_sites": str(args.binding_sites),
        "interfaces": str(args.interfaces),
        "ghost_time_map": str(args.ghost_time_map),
        "site_catalog": str(site_catalog),
        "interface_catalog": str(interface_catalog),
        "chunk_rows": args.chunk_rows,
        "start_record": args.start_record,
        "max_records_per_stream": args.max_records_per_stream,
        "emit_interface_hits": args.emit_interface_hits,
        "residue_index_annotation": {
            "nearby_topology_residue_*": "Raw topology residue ids from GpuSpikeEvent; padded slots are -1.",
            "nearby_uniprot_residue_*": "Mapped through residue_map plus residue_index_mapping_matrix when available; padded/unmapped slots are -1.",
        },
        "semantic_warnings": [
            "This table is a raw-event integration layer, not a chronic durability endpoint.",
            "Materialized-site assignment is nearest-centroid/radius projection, not proof of ligand occupancy.",
            "Interface matches are event annotations for timestamp mining; interface breakage must be inferred by subsequent temporal models over absence/presence windows.",
            "warp_matrix, asc_vectors, forces_final, and adaptive_dt are not parsed by this decoder.",
        ],
        "streams": results,
    }
    manifest_path = args.out_dir / f"{args.run_label}_integration_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(f"Wrote {manifest_path}")
    for result in results:
        print(
            f"stream {result['stream_id']}: {result['records_written']} rows -> {result['event_path']}"
        )
        if result.get("interface_hit_path"):
            print(
                f"stream {result['stream_id']}: {result['interface_hit_rows']} interface-hit rows -> {result['interface_hit_path']}"
            )


if __name__ == "__main__":
    main()
