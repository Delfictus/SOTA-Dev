#!/usr/bin/env python3
"""Export PRISM raw sidecars into an aligned dynamic voxel evidence layer.

This decoder is deliberately conservative:

* `warp_matrix.bin` is parsed using the producer's packed `GpuWarpEntry`
  layout: i32 voxel_idx, i32 atom_indices[16], f32 atom_weights[16], i32 n_atoms.
* `forces_final.bin` and `asc_vectors.bin` are parsed as final per-atom f32
  vectors. They are final/snapshot vector fields, not a per-frame trajectory.
* `adaptive_dt.bin` is parsed as per-chunk records:
  u64 chunk_idx, u64 steps_run, f64 dt_ps, u32 reason_code, u32 pad.
* Dynamic time enters through spike-event Parquets joined onto aligned voxel
  candidates. The script does not invent per-frame warp matrices that were not
  written by the producer.
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import math
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from prism_dstw.io import write_provenance_parquet


WARP_DTYPE = np.dtype(
    [
        ("voxel_idx", "<i4"),
        ("atom_indices", "<i4", (16,)),
        ("atom_weights", "<f4", (16,)),
        ("n_atoms", "<i4"),
    ],
    align=False,
)

ADAPTIVE_DT_DTYPE = np.dtype(
    [
        ("chunk_idx", "<u8"),
        ("steps_run", "<u8"),
        ("dt_ps", "<f8"),
        ("reason_code", "<u4"),
        ("pad", "<u4"),
    ],
    align=False,
)

if WARP_DTYPE.itemsize != 136:
    raise RuntimeError(f"GpuWarpEntry layout expected 136 bytes, got {WARP_DTYPE.itemsize}")

if ADAPTIVE_DT_DTYPE.itemsize != 32:
    raise RuntimeError(f"adaptive_dt record layout expected 32 bytes, got {ADAPTIVE_DT_DTYPE.itemsize}")


@dataclass(frozen=True)
class RunSpec:
    run_label: str
    structure_id: str
    run_dir: Path
    topology_json: Path
    residue_map_json: Path
    atom_to_residue_json: Path
    binding_sites_json: Path
    event_glob: str


def read_json(path: Path) -> Any:
    return json.loads(path.read_text())


def emit_status(event: str, **fields: Any) -> None:
    payload = {"event": event, **fields}
    sys.stderr.write(json.dumps(payload, sort_keys=True) + "\n")


def sha256_path(path: Path, chunk_size: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while True:
            chunk = fh.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def source_file_checksums(paths: list[Path]) -> dict[str, str]:
    return {str(path): sha256_path(path) for path in sorted(set(paths), key=lambda p: str(p))}


def cube_dim(n: int) -> int:
    dim = round(n ** (1.0 / 3.0))
    if dim**3 != n:
        raise ValueError(f"voxel record count {n} is not a perfect cube")
    return dim


def stream_id_from_name(path: Path) -> int:
    match = re.search(r"_stream0?([0-9]+)_", path.name)
    if not match:
        raise ValueError(f"cannot parse stream id from {path}")
    return int(match.group(1))


def find_required(run_dir: Path, suffix: str, stream_id: int, *, two_digit: bool = False) -> Path:
    sid = f"{stream_id:02d}" if two_digit else str(stream_id)
    matches = sorted(run_dir.glob(f"*_stream{sid}_{suffix}"))
    if not matches:
        raise FileNotFoundError(f"missing stream {stream_id} {suffix} in {run_dir}")
    if len(matches) > 1:
        raise ValueError(f"ambiguous stream {stream_id} {suffix}: {matches}")
    return matches[0]


def load_topology(
    topology_json: Path,
    residue_map_json: Path,
    atom_to_residue_json: Path,
    mapping_parquet: Path | None,
    structure_id: str,
) -> dict[str, Any]:
    topology = read_json(topology_json)
    residue_map = read_json(residue_map_json)
    atom_to_residue = np.asarray(read_json(atom_to_residue_json)["atom_to_residue"], dtype=np.int32)
    positions = np.asarray(topology["positions"], dtype=np.float32).reshape((-1, 3))
    if len(atom_to_residue) != int(topology["n_atoms"]):
        raise ValueError(f"{atom_to_residue_json}: atom_to_residue length does not match topology n_atoms")

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
        ).to_pydict()
        for pdb_id, chain, auth_seq, uniprot, is_target in zip(
            table["pdb_id"],
            table["auth_asym_id"],
            table["auth_seq_id"],
            table["uniprot_residue_index"],
            table["is_target_uniprot"],
        ):
            if str(pdb_id).upper() != structure_id.upper() or not bool(is_target):
                continue
            try:
                pdb_chain_res_to_uniprot[(str(chain), str(auth_seq))] = int(uniprot)
            except (TypeError, ValueError):
                continue

    residues = residue_map.get("residues", [])
    max_topo = max(int(r["topology_index"]) for r in residues)
    topo_to_uniprot = np.full(max_topo + 1, -1, dtype=np.int32)
    topo_to_name = np.full(max_topo + 1, "", dtype=object)
    topo_to_pdb_resid = np.full(max_topo + 1, -1, dtype=np.int32)
    for row in residues:
        topo_idx = int(row["topology_index"])
        chain = str(row.get("chain", ""))
        pdb_resid = int(row["pdb_resid"])
        topo_to_uniprot[topo_idx] = pdb_chain_res_to_uniprot.get((chain, str(pdb_resid)), pdb_resid)
        topo_to_name[topo_idx] = str(row.get("resname", ""))
        topo_to_pdb_resid[topo_idx] = pdb_resid

    atom_uniprot = np.full(len(atom_to_residue), -1, dtype=np.int32)
    valid = (atom_to_residue >= 0) & (atom_to_residue < len(topo_to_uniprot))
    atom_uniprot[valid] = topo_to_uniprot[atom_to_residue[valid]]

    min_pos = positions.min(axis=0)
    max_pos = positions.max(axis=0)
    origin = min_pos - np.float32(5.0)

    return {
        "topology": topology,
        "positions": positions,
        "atom_names": np.asarray(topology.get("atom_names", [""] * len(positions)), dtype=object),
        "atom_to_residue": atom_to_residue,
        "atom_uniprot": atom_uniprot,
        "topo_to_uniprot": topo_to_uniprot,
        "topo_to_name": topo_to_name,
        "topo_to_pdb_resid": topo_to_pdb_resid,
        "min_pos": min_pos,
        "max_pos": max_pos,
        "origin": origin,
        "n_atoms": int(topology["n_atoms"]),
        "n_residues": int(topology["n_residues"]),
    }


def grid_spacing_for(topology_info: dict[str, Any], grid_dim: int, floor_spacing: float = 0.75) -> tuple[float, float]:
    extent = np.asarray(topology_info["max_pos"]) - np.asarray(topology_info["min_pos"]) + np.float32(10.0)
    padded_extent = float(np.nanmax(extent))
    required = (padded_extent / float(grid_dim)) * 1.02 if grid_dim else floor_spacing
    return max(floor_spacing, required), padded_extent


def load_interfaces(path: Path) -> list[dict[str, Any]]:
    table = pq.read_table(path).to_pylist()
    out: list[dict[str, Any]] = []
    seen: set[str] = set()
    for row in table:
        target = int(row["target_hinge_residue_index"])
        neighbor = int(row["neighbor_residue_index"])
        target_label = str(row.get("target_hinge_label") or f"{target}")
        neighbor_label = str(row.get("neighbor_label") or f"{neighbor}")
        interface_id = f"{target_label}__{neighbor_label}"
        if interface_id in seen:
            continue
        seen.add(interface_id)
        out.append(
            {
                "interface_id": interface_id,
                "target_hinge_residue_index": target,
                "neighbor_residue_index": neighbor,
                "target_hinge_label": target_label,
                "neighbor_label": neighbor_label,
                "interface_class": str(row.get("pocket_accessibility_class") or row.get("interface_class") or ""),
                "lock_interface_score": float(row.get("lock_interface_score") or 0.0),
                "te_coupling_score": float(row.get("te_coupling_score") or 0.0),
                "interface_te_asymmetry": (
                    None if row.get("interface_te_asymmetry") is None else float(row.get("interface_te_asymmetry"))
                ),
                "mean_distance_angstrom": float(row.get("mean_distance_angstrom") or 0.0),
                "nearest_materialized_pocket_site_id": str(row.get("nearest_materialized_pocket_site_id") or ""),
            }
        )
    return out


def load_sites(path: Path, limit: int) -> list[dict[str, Any]]:
    data = read_json(path)
    sites = data.get("binding_sites") or []
    sites = sorted(sites, key=lambda r: int(r.get("new_prism_rank", r.get("rank", 10**9))))
    out: list[dict[str, Any]] = []
    for site in sites[:limit]:
        centroid = site.get("centroid_xyz") or [math.nan, math.nan, math.nan]
        out.append(
            {
                "site_id": str(site.get("site_id")),
                "site_rank": int(site.get("new_prism_rank", site.get("rank", -1))),
                "centroid_x": float(centroid[0]),
                "centroid_y": float(centroid[1]),
                "centroid_z": float(centroid[2]),
                "region_radius_a": float(site.get("region_radius_a") or site.get("centroid_spread_a") or 0.0),
                "n_spikes": int(site.get("n_spikes") or 0),
                "materialization_level": str(site.get("materialization_level") or ""),
            }
        )
    return out


def np_norm(v: np.ndarray) -> np.ndarray:
    return np.sqrt(np.sum(v.astype(np.float64) * v.astype(np.float64), axis=-1))


def arrow_table_from_arrays(columns: dict[str, Any]) -> pa.Table:
    return pa.Table.from_pydict(columns)


def stream_paths(run: RunSpec, stream_id: int) -> dict[str, Path]:
    return {
        "warp_matrix": find_required(run.run_dir, "warp_matrix.bin", stream_id),
        "forces_final": find_required(run.run_dir, "forces_final.bin", stream_id),
        "adaptive_dt": find_required(run.run_dir, "adaptive_dt.bin", stream_id),
        "asc_vectors": find_required(run.run_dir, "asc_vectors.bin", stream_id, two_digit=True),
    }


def vector_table_for_stream(
    *,
    campaign_id: str,
    run: RunSpec,
    stream_id: int,
    topology_info: dict[str, Any],
    forces: np.ndarray,
    asc: np.ndarray,
) -> pa.Table:
    atom_idx = np.arange(topology_info["n_atoms"], dtype=np.int32)
    topo_res = topology_info["atom_to_residue"]
    uniprot = topology_info["atom_uniprot"]
    residue_names = np.asarray(
        [
            topology_info["topo_to_name"][r] if 0 <= r < len(topology_info["topo_to_name"]) else ""
            for r in topo_res
        ],
        dtype=object,
    )
    positions = topology_info["positions"]
    force_norm = np_norm(forces)
    asc_norm = np_norm(asc)
    delta = forces - asc
    delta_norm = np_norm(delta)
    dot = np.sum(forces.astype(np.float64) * asc.astype(np.float64), axis=1)
    cos = dot / np.maximum(force_norm * asc_norm, 1e-12)
    return arrow_table_from_arrays(
        {
            "campaign_id": [campaign_id] * len(atom_idx),
            "run_label": [run.run_label] * len(atom_idx),
            "structure_id": [run.structure_id] * len(atom_idx),
            "stream_id": np.full(len(atom_idx), stream_id, dtype=np.int16),
            "atom_index": atom_idx,
            "topology_residue_index": topo_res.astype(np.int32),
            "uniprot_residue_index": uniprot.astype(np.int32),
            "residue_name": residue_names.tolist(),
            "atom_name": topology_info["atom_names"].astype(object).tolist(),
            "x": positions[:, 0].astype(np.float32),
            "y": positions[:, 1].astype(np.float32),
            "z": positions[:, 2].astype(np.float32),
            "force_x": forces[:, 0].astype(np.float32),
            "force_y": forces[:, 1].astype(np.float32),
            "force_z": forces[:, 2].astype(np.float32),
            "force_norm": force_norm.astype(np.float32),
            "asc_x": asc[:, 0].astype(np.float32),
            "asc_y": asc[:, 1].astype(np.float32),
            "asc_z": asc[:, 2].astype(np.float32),
            "asc_norm": asc_norm.astype(np.float32),
            "force_minus_asc_norm": delta_norm.astype(np.float32),
            "force_asc_cosine": cos.astype(np.float32),
        }
    )


def adaptive_dt_table(
    *,
    campaign_id: str,
    run: RunSpec,
    stream_id: int,
    path: Path,
) -> pa.Table:
    records = np.fromfile(path, dtype=ADAPTIVE_DT_DTYPE)
    reason = np.asarray(
        [
            {0: "unknown_or_disabled", 1: "hold_phase_upscale", 2: "ramp_or_base_dt", 3: "gearbox_owned"}.get(
                int(x), f"unknown_{int(x)}"
            )
            for x in records["reason_code"]
        ],
        dtype=object,
    )
    return arrow_table_from_arrays(
        {
            "campaign_id": [campaign_id] * len(records),
            "run_label": [run.run_label] * len(records),
            "structure_id": [run.structure_id] * len(records),
            "stream_id": np.full(len(records), stream_id, dtype=np.int16),
            "chunk_idx": records["chunk_idx"].astype(np.int64),
            "steps_run": records["steps_run"].astype(np.int64),
            "dt_ps": records["dt_ps"].astype(np.float64),
            "reason_code": records["reason_code"].astype(np.int32),
            "reason_label": reason.tolist(),
            "nominal_time_ps_from_steps": records["steps_run"].astype(np.float64) * 0.004,
            "source_path": [str(path)] * len(records),
        }
    )


def weighted_vectors(atom_indices: np.ndarray, weights: np.ndarray, vectors: np.ndarray) -> np.ndarray:
    safe = np.where(atom_indices >= 0, atom_indices, 0)
    gathered = vectors[safe]
    gathered = np.where((atom_indices >= 0)[..., None], gathered, 0.0)
    return np.sum(gathered * weights[..., None], axis=1)


def voxel_coords(voxel_idx: np.ndarray, grid_dim: int, origin: np.ndarray, spacing: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    idx = voxel_idx.astype(np.int64)
    vz = idx // (grid_dim * grid_dim)
    vy = (idx // grid_dim) % grid_dim
    vx = idx % grid_dim
    x = origin[0] + (vx.astype(np.float32) + 0.5) * spacing
    y = origin[1] + (vy.astype(np.float32) + 0.5) * spacing
    z = origin[2] + (vz.astype(np.float32) + 0.5) * spacing
    return x.astype(np.float32), y.astype(np.float32), z.astype(np.float32)


def interface_voxel_table(
    *,
    campaign_id: str,
    run: RunSpec,
    stream_id: int,
    interface: dict[str, Any],
    warp: np.memmap,
    grid_dim: int,
    origin: np.ndarray,
    spacing: float,
    topology_info: dict[str, Any],
    forces: np.ndarray,
    asc: np.ndarray,
) -> pa.Table | None:
    atom_uniprot = topology_info["atom_uniprot"]
    target_atoms = np.flatnonzero(atom_uniprot == int(interface["target_hinge_residue_index"])).astype(np.int32)
    neighbor_atoms = np.flatnonzero(atom_uniprot == int(interface["neighbor_residue_index"])).astype(np.int32)
    if len(target_atoms) == 0 and len(neighbor_atoms) == 0:
        return None

    atom_idx = warp["atom_indices"]
    weights = warp["atom_weights"]
    target_mask = np.isin(atom_idx, target_atoms) if len(target_atoms) else np.zeros(atom_idx.shape, dtype=bool)
    neighbor_mask = np.isin(atom_idx, neighbor_atoms) if len(neighbor_atoms) else np.zeros(atom_idx.shape, dtype=bool)
    selected = target_mask.any(axis=1) | neighbor_mask.any(axis=1)
    if not bool(selected.any()):
        return None

    rows = np.flatnonzero(selected)
    selected_atoms = atom_idx[rows]
    selected_weights = weights[rows]
    voxel_idx = warp["voxel_idx"][rows]
    n_atoms = warp["n_atoms"][rows]
    target_weight = np.sum(np.where(target_mask[rows], selected_weights, 0.0), axis=1)
    neighbor_weight = np.sum(np.where(neighbor_mask[rows], selected_weights, 0.0), axis=1)
    total_weight = np.sum(np.where(selected_atoms >= 0, selected_weights, 0.0), axis=1)
    force_vec = weighted_vectors(selected_atoms, selected_weights, forces)
    asc_vec = weighted_vectors(selected_atoms, selected_weights, asc)
    x, y, z = voxel_coords(voxel_idx, grid_dim, origin, spacing)
    return arrow_table_from_arrays(
        {
            "campaign_id": [campaign_id] * len(rows),
            "run_label": [run.run_label] * len(rows),
            "structure_id": [run.structure_id] * len(rows),
            "stream_id": np.full(len(rows), stream_id, dtype=np.int16),
            "scope_type": ["sar_interface"] * len(rows),
            "scope_id": [interface["interface_id"]] * len(rows),
            "interface_id": [interface["interface_id"]] * len(rows),
            "interface_class": [interface["interface_class"]] * len(rows),
            "target_hinge_residue_index": np.full(len(rows), int(interface["target_hinge_residue_index"]), dtype=np.int32),
            "neighbor_residue_index": np.full(len(rows), int(interface["neighbor_residue_index"]), dtype=np.int32),
            "voxel_idx": voxel_idx.astype(np.int32),
            "voxel_x": x,
            "voxel_y": y,
            "voxel_z": z,
            "grid_dim": np.full(len(rows), grid_dim, dtype=np.int16),
            "grid_spacing_a": np.full(len(rows), spacing, dtype=np.float32),
            "warp_n_atoms": n_atoms.astype(np.int16),
            "warp_total_weight": total_weight.astype(np.float32),
            "target_endpoint_weight": target_weight.astype(np.float32),
            "neighbor_endpoint_weight": neighbor_weight.astype(np.float32),
            "endpoint_weight": (target_weight + neighbor_weight).astype(np.float32),
            "weighted_force_x": force_vec[:, 0].astype(np.float32),
            "weighted_force_y": force_vec[:, 1].astype(np.float32),
            "weighted_force_z": force_vec[:, 2].astype(np.float32),
            "weighted_force_norm": np_norm(force_vec).astype(np.float32),
            "weighted_asc_x": asc_vec[:, 0].astype(np.float32),
            "weighted_asc_y": asc_vec[:, 1].astype(np.float32),
            "weighted_asc_z": asc_vec[:, 2].astype(np.float32),
            "weighted_asc_norm": np_norm(asc_vec).astype(np.float32),
            "force_asc_delta_norm": np_norm(force_vec - asc_vec).astype(np.float32),
            "te_coupling_score": np.full(len(rows), float(interface["te_coupling_score"]), dtype=np.float64),
            "lock_interface_score": np.full(len(rows), float(interface["lock_interface_score"]), dtype=np.float64),
            "mean_distance_angstrom": np.full(len(rows), float(interface["mean_distance_angstrom"]), dtype=np.float32),
        }
    )


def site_voxel_table(
    *,
    campaign_id: str,
    run: RunSpec,
    stream_id: int,
    site: dict[str, Any],
    warp: np.memmap,
    grid_dim: int,
    origin: np.ndarray,
    spacing: float,
    forces: np.ndarray,
    asc: np.ndarray,
) -> pa.Table | None:
    radius = max(float(site["region_radius_a"]), spacing)
    cx, cy, cz = float(site["centroid_x"]), float(site["centroid_y"]), float(site["centroid_z"])
    if not all(math.isfinite(v) for v in [cx, cy, cz, radius]):
        return None

    vx0 = max(0, int(math.floor((cx - radius - origin[0]) / spacing)))
    vx1 = min(grid_dim - 1, int(math.ceil((cx + radius - origin[0]) / spacing)))
    vy0 = max(0, int(math.floor((cy - radius - origin[1]) / spacing)))
    vy1 = min(grid_dim - 1, int(math.ceil((cy + radius - origin[1]) / spacing)))
    vz0 = max(0, int(math.floor((cz - radius - origin[2]) / spacing)))
    vz1 = min(grid_dim - 1, int(math.ceil((cz + radius - origin[2]) / spacing)))
    if vx1 < vx0 or vy1 < vy0 or vz1 < vz0:
        return None

    xs, ys, zs = np.meshgrid(
        np.arange(vx0, vx1 + 1, dtype=np.int32),
        np.arange(vy0, vy1 + 1, dtype=np.int32),
        np.arange(vz0, vz1 + 1, dtype=np.int32),
        indexing="xy",
    )
    voxel_idx = (zs.astype(np.int64) * grid_dim * grid_dim + ys.astype(np.int64) * grid_dim + xs.astype(np.int64)).ravel()
    x = origin[0] + (xs.ravel().astype(np.float32) + 0.5) * spacing
    y = origin[1] + (ys.ravel().astype(np.float32) + 0.5) * spacing
    z = origin[2] + (zs.ravel().astype(np.float32) + 0.5) * spacing
    dist = np.sqrt((x - cx) ** 2 + (y - cy) ** 2 + (z - cz) ** 2)
    keep = dist <= radius
    if not bool(keep.any()):
        return None
    voxel_idx = voxel_idx[keep].astype(np.int32)
    x, y, z, dist = x[keep], y[keep], z[keep], dist[keep]
    atom_idx = warp["atom_indices"][voxel_idx]
    weights = warp["atom_weights"][voxel_idx]
    n_atoms = warp["n_atoms"][voxel_idx]
    total_weight = np.sum(np.where(atom_idx >= 0, weights, 0.0), axis=1)
    force_vec = weighted_vectors(atom_idx, weights, forces)
    asc_vec = weighted_vectors(atom_idx, weights, asc)
    return arrow_table_from_arrays(
        {
            "campaign_id": [campaign_id] * len(voxel_idx),
            "run_label": [run.run_label] * len(voxel_idx),
            "structure_id": [run.structure_id] * len(voxel_idx),
            "stream_id": np.full(len(voxel_idx), stream_id, dtype=np.int16),
            "scope_type": ["materialized_site"] * len(voxel_idx),
            "scope_id": [site["site_id"]] * len(voxel_idx),
            "site_id": [site["site_id"]] * len(voxel_idx),
            "site_rank": np.full(len(voxel_idx), int(site["site_rank"]), dtype=np.int32),
            "voxel_idx": voxel_idx,
            "voxel_x": x.astype(np.float32),
            "voxel_y": y.astype(np.float32),
            "voxel_z": z.astype(np.float32),
            "distance_to_site_centroid_a": dist.astype(np.float32),
            "site_radius_a": np.full(len(voxel_idx), radius, dtype=np.float32),
            "grid_dim": np.full(len(voxel_idx), grid_dim, dtype=np.int16),
            "grid_spacing_a": np.full(len(voxel_idx), spacing, dtype=np.float32),
            "warp_n_atoms": n_atoms.astype(np.int16),
            "warp_total_weight": total_weight.astype(np.float32),
            "weighted_force_x": force_vec[:, 0].astype(np.float32),
            "weighted_force_y": force_vec[:, 1].astype(np.float32),
            "weighted_force_z": force_vec[:, 2].astype(np.float32),
            "weighted_force_norm": np_norm(force_vec).astype(np.float32),
            "weighted_asc_x": asc_vec[:, 0].astype(np.float32),
            "weighted_asc_y": asc_vec[:, 1].astype(np.float32),
            "weighted_asc_z": asc_vec[:, 2].astype(np.float32),
            "weighted_asc_norm": np_norm(asc_vec).astype(np.float32),
            "force_asc_delta_norm": np_norm(force_vec - asc_vec).astype(np.float32),
            "site_n_spikes": np.full(len(voxel_idx), int(site["n_spikes"]), dtype=np.int64),
            "materialization_level": [site["materialization_level"]] * len(voxel_idx),
        }
    )


def stream_summary(
    *,
    campaign_id: str,
    run: RunSpec,
    stream_id: int,
    paths: dict[str, Path],
    topology_info: dict[str, Any],
    warp: np.memmap,
    grid_dim: int,
    spacing: float,
    padded_extent: float,
    forces: np.ndarray,
    asc: np.ndarray,
    adaptive_dt: np.ndarray,
) -> dict[str, Any]:
    n_atoms = warp["n_atoms"].astype(np.int16)
    occupied = n_atoms > 0
    weights = warp["atom_weights"]
    valid_weights = np.where(warp["atom_indices"] >= 0, weights, 0.0)
    weight_sums = valid_weights.sum(axis=1)
    force_norm = np_norm(forces)
    asc_norm = np_norm(asc)
    delta_norm = np_norm(forces - asc)
    return {
        "campaign_id": campaign_id,
        "run_label": run.run_label,
        "structure_id": run.structure_id,
        "stream_id": stream_id,
        "grid_dim": grid_dim,
        "total_voxels": int(len(warp)),
        "grid_spacing_a": spacing,
        "padded_extent_a": padded_extent,
        "origin_x": float(topology_info["origin"][0]),
        "origin_y": float(topology_info["origin"][1]),
        "origin_z": float(topology_info["origin"][2]),
        "topology_n_atoms": int(topology_info["n_atoms"]),
        "topology_n_residues": int(topology_info["n_residues"]),
        "warp_bytes": int(paths["warp_matrix"].stat().st_size),
        "warp_record_size": WARP_DTYPE.itemsize,
        "warp_voxels_with_atoms": int(occupied.sum()),
        "warp_occupied_fraction": float(occupied.mean()),
        "warp_mean_atoms_per_voxel": float(n_atoms.mean()),
        "warp_mean_atoms_per_occupied_voxel": float(n_atoms[occupied].mean()) if occupied.any() else 0.0,
        "warp_max_atoms_per_voxel": int(n_atoms.max()),
        "warp_mean_weight_sum_occupied": float(weight_sums[occupied].mean()) if occupied.any() else 0.0,
        "force_mean_norm": float(force_norm.mean()),
        "force_max_norm": float(force_norm.max()),
        "asc_mean_norm": float(asc_norm.mean()),
        "asc_max_norm": float(asc_norm.max()),
        "force_minus_asc_mean_norm": float(delta_norm.mean()),
        "force_minus_asc_max_norm": float(delta_norm.max()),
        "adaptive_dt_record_count": int(len(adaptive_dt)),
        "adaptive_dt_min_ps": float(np.min(adaptive_dt["dt_ps"])) if len(adaptive_dt) else math.nan,
        "adaptive_dt_max_ps": float(np.max(adaptive_dt["dt_ps"])) if len(adaptive_dt) else math.nan,
        "adaptive_dt_mean_ps": float(np.mean(adaptive_dt["dt_ps"])) if len(adaptive_dt) else math.nan,
        "warp_sha256": sha256_path(paths["warp_matrix"]),
        "forces_sha256": sha256_path(paths["forces_final"]),
        "asc_sha256": sha256_path(paths["asc_vectors"]),
        "adaptive_dt_sha256": sha256_path(paths["adaptive_dt"]),
    }


def expand_globs(patterns: list[str]) -> list[str]:
    out: list[str] = []
    for pattern in patterns:
        matches = sorted(glob.glob(pattern))
        if matches:
            out.extend(matches)
        else:
            out.append(pattern)
    return sorted(dict.fromkeys(out))


def collect_streaming(lf: pl.LazyFrame) -> pl.DataFrame:
    return lf.collect(engine="streaming")


def write_lazy_parquet(lf: pl.LazyFrame, path: Path, source_parquets: list[Path]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    write_provenance_parquet(
        lf,
        path,
        producer_script=Path(__file__),
        source_parquets=source_parquets,
        schema_version="prism_dynamic_aligned_voxel_export.v1.dynamic_event_bins",
        pipeline_stage="dynamic_voxel_event_bins",
        partition_keys=["campaign_id", "run_label", "structure_id", "stream_id", "scope_type", "scope_id"],
        extra_metadata={"lineage_scope": "dynamic_voxel_event_time_bins"},
    )


def write_dynamic_event_bins(
    *,
    event_patterns: list[str],
    interface_voxels: Path,
    site_voxels: Path,
    out_path: Path,
    manifest: dict[str, Any],
) -> None:
    event_paths = expand_globs(event_patterns)
    if not event_paths:
        raise ValueError("no event Parquet paths resolved for dynamic voxel event join")
    missing = [p for p in event_paths + [str(interface_voxels), str(site_voxels)] if not Path(p).exists()]
    if missing:
        raise FileNotFoundError(f"missing dynamic voxel event-bin inputs: {missing}")

    events = (
        pl.scan_parquet(event_paths)
        .select(
            [
                pl.col("campaign_id").cast(pl.Utf8),
                pl.col("run_label").cast(pl.Utf8),
                pl.col("structure_id").cast(pl.Utf8),
                pl.col("stream_id").cast(pl.Int16),
                pl.col("timestep").cast(pl.Int64),
                pl.col("physical_time_ps").cast(pl.Float64),
                pl.col("voxel_idx").cast(pl.Int64),
                pl.col("intensity").cast(pl.Float64),
                pl.col("water_density").cast(pl.Float64),
                pl.col("wd_change").cast(pl.Float64),
                pl.col("vibrational_energy").cast(pl.Float64),
            ]
        )
    )
    event_bins = events.group_by(
        ["campaign_id", "run_label", "structure_id", "stream_id", "timestep", "voxel_idx"]
    ).agg(
        [
            pl.col("physical_time_ps").min().alias("physical_time_ps"),
            pl.len().cast(pl.UInt64).alias("spike_event_count"),
            pl.col("intensity").mean().alias("mean_intensity"),
            pl.col("intensity").max().alias("max_intensity"),
            pl.col("water_density").mean().alias("mean_water_density"),
            pl.col("wd_change").mean().alias("mean_wd_change"),
            pl.col("vibrational_energy").mean().alias("mean_vibrational_energy"),
        ]
    )
    scope_columns = [
        pl.col("campaign_id").cast(pl.Utf8),
        pl.col("run_label").cast(pl.Utf8),
        pl.col("structure_id").cast(pl.Utf8),
        pl.col("stream_id").cast(pl.Int16),
        pl.col("scope_type").cast(pl.Utf8),
        pl.col("scope_id").cast(pl.Utf8),
        pl.col("voxel_idx").cast(pl.Int64),
    ]
    scopes = (
        pl.concat(
            [
                pl.scan_parquet(interface_voxels).select(scope_columns),
                pl.scan_parquet(site_voxels).select(scope_columns),
            ],
            how="vertical_relaxed",
        )
        .unique()
    )
    dynamic_bins = (
        event_bins.join(
            scopes,
            on=["campaign_id", "run_label", "structure_id", "stream_id", "voxel_idx"],
            how="inner",
        )
        .select(
            [
                "campaign_id",
                "run_label",
                "structure_id",
                "stream_id",
                "timestep",
                "physical_time_ps",
                "scope_type",
                "scope_id",
                "voxel_idx",
                "spike_event_count",
                "mean_intensity",
                "max_intensity",
                "mean_water_density",
                "mean_wd_change",
                "mean_vibrational_energy",
            ]
        )
        .sort(["run_label", "structure_id", "stream_id", "scope_type", "scope_id", "timestep", "voxel_idx"])
    )
    write_lazy_parquet(dynamic_bins, out_path, [Path(p) for p in event_paths] + [interface_voxels, site_voxels])
    count = pq.read_metadata(out_path).num_rows
    manifest["outputs"]["dynamic_voxel_event_time_bins"] = str(out_path)
    manifest["counts"]["dynamic_voxel_event_time_bins"] = int(count)
    manifest["event_parquets"] = event_paths
    manifest["dynamic_event_bin_engine"] = "polars_native_arrow_parquet"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-id", default="glp1r_aleniglipron")
    parser.add_argument("--inactive-run-dir", type=Path, required=True)
    parser.add_argument("--active-run-dir", type=Path, required=True)
    parser.add_argument("--inactive-topology", type=Path, required=True)
    parser.add_argument("--active-topology", type=Path, required=True)
    parser.add_argument("--inactive-residue-map", type=Path, required=True)
    parser.add_argument("--active-residue-map", type=Path, required=True)
    parser.add_argument("--inactive-atom-to-residue", type=Path, required=True)
    parser.add_argument("--active-atom-to-residue", type=Path, required=True)
    parser.add_argument("--residue-mapping-parquet", type=Path, required=True)
    parser.add_argument("--interfaces", type=Path, required=True)
    parser.add_argument("--event-parquet", action="append", default=[])
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--site-limit", type=int, default=10)
    parser.add_argument("--skip-event-join", action="store_true")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    runs = [
        RunSpec(
            "inactive_5vex",
            "5VEX",
            args.inactive_run_dir,
            args.inactive_topology,
            args.inactive_residue_map,
            args.inactive_atom_to_residue,
            args.inactive_run_dir / "binding_sites.materialized.json",
            "",
        ),
        RunSpec(
            "active_6x1a",
            "6X1A",
            args.active_run_dir,
            args.active_topology,
            args.active_residue_map,
            args.active_atom_to_residue,
            args.active_run_dir / "binding_sites.materialized.json",
            "",
        ),
    ]
    interfaces = load_interfaces(args.interfaces)

    atom_tables: list[pa.Table] = []
    dt_tables: list[pa.Table] = []
    interface_tables: list[pa.Table] = []
    site_tables: list[pa.Table] = []
    summaries: list[dict[str, Any]] = []
    coverage_gaps: list[dict[str, Any]] = []

    atom_path = args.out_dir / "aligned_atom_vector_fields.parquet"
    dt_path = args.out_dir / "adaptive_dt_history.parquet"
    interface_path = args.out_dir / "interface_aligned_voxel_fields.parquet"
    site_path = args.out_dir / "site_aligned_voxel_fields.parquet"
    sidecar_source_parquets = [
        args.residue_mapping_parquet,
        args.interfaces,
    ]
    sidecar_raw_sources = [
        args.inactive_topology,
        args.active_topology,
        args.inactive_residue_map,
        args.active_residue_map,
        args.inactive_atom_to_residue,
        args.active_atom_to_residue,
        runs[0].binding_sites_json,
        runs[1].binding_sites_json,
    ]

    for run in runs:
        topology_info = load_topology(
            run.topology_json,
            run.residue_map_json,
            run.atom_to_residue_json,
            args.residue_mapping_parquet,
            run.structure_id,
        )
        sites = load_sites(run.binding_sites_json, args.site_limit)
        for stream_id in range(8):
            paths = stream_paths(run, stream_id)
            sidecar_raw_sources.extend(paths.values())
            warp_records = paths["warp_matrix"].stat().st_size // WARP_DTYPE.itemsize
            grid_dim = cube_dim(warp_records)
            spacing, padded_extent = grid_spacing_for(topology_info, grid_dim)
            warp = np.memmap(paths["warp_matrix"], dtype=WARP_DTYPE, mode="r")
            forces = np.fromfile(paths["forces_final"], dtype="<f4").reshape((-1, 3))
            asc = np.fromfile(paths["asc_vectors"], dtype="<f4").reshape((-1, 3))
            adaptive_dt = np.fromfile(paths["adaptive_dt"], dtype=ADAPTIVE_DT_DTYPE)
            if len(forces) != topology_info["n_atoms"] or len(asc) != topology_info["n_atoms"]:
                raise ValueError(f"{run.run_label} stream {stream_id}: vector sidecar atom count mismatch")

            atom_tables.append(
                vector_table_for_stream(
                    campaign_id=args.campaign_id,
                    run=run,
                    stream_id=stream_id,
                    topology_info=topology_info,
                    forces=forces,
                    asc=asc,
                )
            )
            dt_tables.append(
                adaptive_dt_table(campaign_id=args.campaign_id, run=run, stream_id=stream_id, path=paths["adaptive_dt"])
            )

            for interface in interfaces:
                table = interface_voxel_table(
                    campaign_id=args.campaign_id,
                    run=run,
                    stream_id=stream_id,
                    interface=interface,
                    warp=warp,
                    grid_dim=grid_dim,
                    origin=topology_info["origin"],
                    spacing=spacing,
                    topology_info=topology_info,
                    forces=forces,
                    asc=asc,
                )
                if table is None or table.num_rows == 0:
                    coverage_gaps.append(
                        {
                            "run_label": run.run_label,
                            "structure_id": run.structure_id,
                            "stream_id": stream_id,
                            "interface_id": interface["interface_id"],
                            "reason": "no_endpoint_atoms_or_no_warp_voxels",
                        }
                    )
                else:
                    interface_tables.append(table)

            for site in sites:
                table = site_voxel_table(
                    campaign_id=args.campaign_id,
                    run=run,
                    stream_id=stream_id,
                    site=site,
                    warp=warp,
                    grid_dim=grid_dim,
                    origin=topology_info["origin"],
                    spacing=spacing,
                    forces=forces,
                    asc=asc,
                )
                if table is not None and table.num_rows > 0:
                    site_tables.append(table)

            summaries.append(
                stream_summary(
                    campaign_id=args.campaign_id,
                    run=run,
                    stream_id=stream_id,
                    paths=paths,
                    topology_info=topology_info,
                    warp=warp,
                    grid_dim=grid_dim,
                    spacing=spacing,
                    padded_extent=padded_extent,
                    forces=forces,
                    asc=asc,
                    adaptive_dt=adaptive_dt,
                )
            )
            emit_status("dynamic_stream_processed", run_label=run.run_label, stream_id=int(stream_id))

    summary_path = args.out_dir / "aligned_voxel_stream_summary.parquet"
    write_provenance_parquet(
        pa.concat_tables(atom_tables),
        atom_path,
        producer_script=Path(__file__),
        source_parquets=sidecar_source_parquets,
        schema_version="prism_dynamic_aligned_voxel_export.v1",
        pipeline_stage="dynamic_aligned_voxel_sidecars",
        partition_keys=["campaign_id", "run_label", "structure_id", "stream_id"],
        extra_metadata={
            "lineage_scope": "aligned_atom_vector_fields",
            "source_files": source_file_checksums(sidecar_raw_sources),
        },
    )
    write_provenance_parquet(
        pa.concat_tables(dt_tables),
        dt_path,
        producer_script=Path(__file__),
        source_parquets=sidecar_source_parquets,
        schema_version="prism_dynamic_aligned_voxel_export.v1",
        pipeline_stage="dynamic_aligned_voxel_sidecars",
        partition_keys=["campaign_id", "run_label", "structure_id", "stream_id"],
        extra_metadata={
            "lineage_scope": "adaptive_dt_history",
            "source_files": source_file_checksums(sidecar_raw_sources),
        },
    )
    write_provenance_parquet(
        pa.concat_tables(interface_tables),
        interface_path,
        producer_script=Path(__file__),
        source_parquets=sidecar_source_parquets,
        schema_version="prism_dynamic_aligned_voxel_export.v1",
        pipeline_stage="dynamic_aligned_voxel_sidecars",
        partition_keys=["campaign_id", "run_label", "structure_id", "stream_id", "scope_type", "scope_id"],
        extra_metadata={
            "lineage_scope": "interface_aligned_voxel_fields",
            "source_files": source_file_checksums(sidecar_raw_sources),
        },
    )
    write_provenance_parquet(
        pa.concat_tables(site_tables),
        site_path,
        producer_script=Path(__file__),
        source_parquets=sidecar_source_parquets,
        schema_version="prism_dynamic_aligned_voxel_export.v1",
        pipeline_stage="dynamic_aligned_voxel_sidecars",
        partition_keys=["campaign_id", "run_label", "structure_id", "stream_id", "scope_type", "scope_id"],
        extra_metadata={
            "lineage_scope": "site_aligned_voxel_fields",
            "source_files": source_file_checksums(sidecar_raw_sources),
        },
    )
    write_provenance_parquet(
        pa.Table.from_pylist(summaries),
        summary_path,
        producer_script=Path(__file__),
        source_parquets=sidecar_source_parquets,
        schema_version="prism_dynamic_aligned_voxel_export.v1",
        pipeline_stage="dynamic_aligned_voxel_sidecars",
        partition_keys=["campaign_id", "run_label", "structure_id", "stream_id"],
        extra_metadata={
            "lineage_scope": "aligned_voxel_stream_summary",
            "source_files": source_file_checksums(sidecar_raw_sources),
        },
    )

    manifest = {
        "schema": "prism_dynamic_aligned_voxel_export.v1",
        "engine": "arrow_parquet_sidecars_plus_polars_dynamic_event_bins",
        "campaign_id": args.campaign_id,
        "producer_layouts": {
            "warp_matrix": "GpuWarpEntry repr(C, packed): i32 voxel_idx, i32 atom_indices[16], f32 atom_weights[16], i32 n_atoms; 136 bytes",
            "forces_final": "f32[n_atoms,3] final per-atom force vector snapshot",
            "asc_vectors": "f32[n_atoms,3] ASC force-vector snapshot",
            "adaptive_dt": "u64 chunk_idx, u64 steps_run, f64 dt_ps, u32 reason_code, u32 pad; 32 bytes",
        },
        "outputs": {
            "aligned_atom_vector_fields": str(atom_path),
            "adaptive_dt_history": str(dt_path),
            "interface_aligned_voxel_fields": str(interface_path),
            "site_aligned_voxel_fields": str(site_path),
            "aligned_voxel_stream_summary": str(summary_path),
        },
        "counts": {
            "aligned_atom_vector_fields": pq.read_metadata(atom_path).num_rows,
            "adaptive_dt_history": pq.read_metadata(dt_path).num_rows,
            "interface_aligned_voxel_fields": pq.read_metadata(interface_path).num_rows,
            "site_aligned_voxel_fields": pq.read_metadata(site_path).num_rows,
            "aligned_voxel_stream_summary": pq.read_metadata(summary_path).num_rows,
        },
        "vector_channel_equivalence": [
            {
                "run_label": row["run_label"],
                "structure_id": row["structure_id"],
                "stream_id": row["stream_id"],
                "forces_sha256_equals_asc_sha256": row["forces_sha256"] == row["asc_sha256"],
                "force_minus_asc_mean_norm": row["force_minus_asc_mean_norm"],
                "force_minus_asc_max_norm": row["force_minus_asc_max_norm"],
            }
            for row in summaries
        ],
        "coverage_gaps": coverage_gaps,
        "semantic_gates": [
            "warp_matrix is final aligned voxel support, not a per-frame warp trajectory.",
            "forces_final and asc_vectors are final per-atom vector snapshots.",
            "When forces_final and asc_vectors have identical checksums, they are duplicate final force-vector evidence for that stream and must not be treated as independent channels.",
            "dynamic time is admitted only through spike-event joins and adaptive_dt chunk history.",
            "site voxel rows are centroid/radius projections and do not prove ligand occupancy.",
            "interface voxel rows are endpoint-warp support fields and do not prove biological interface breaking.",
        ],
    }

    if args.event_parquet and not args.skip_event_join:
        write_dynamic_event_bins(
            event_patterns=args.event_parquet,
            interface_voxels=interface_path,
            site_voxels=site_path,
            out_path=args.out_dir / "dynamic_voxel_event_time_bins.parquet",
            manifest=manifest,
        )

    manifest_path = args.out_dir / "dynamic_aligned_voxel_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    emit_status("dynamic_aligned_voxel_export_complete", manifest=str(manifest_path), counts=manifest["counts"])


if __name__ == "__main__":
    main()
