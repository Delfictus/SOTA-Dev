#!/usr/bin/env python3
"""
prism-aggregate-sites.py — Null-Manifest post-hoc site reconstructor.

Reconstruct site objects from preserved per-residue ASC + KCC + spatial-grid
+ ghost-firehose telemetry, for runs in which the V2 12-sigma adjudicator
gate did not trigger and `kcc_visualization.sites` therefore landed empty.

This is a quarantine-tier diagnostic / forensic tool. It does not replace
the engine's V2 adjudicator path; it salvages information from runs that
failed to ignite the per-frame SITE construction path.

Inputs (resolved by glob inside --run-dir):
    REQUIRED-ish (skipped gracefully if missing):
        <stem>.kcc_visualization.json
        <stem>.binding_sites.json
        <stem>.spatial_grid_state.json   (or <stem>_spatial_grid_state.json)
    OPTIONAL (defensive — lanes still landing):
        <stem>_ghost_tiles.bin                       (legacy single-stream)
        <stem>_ghost_tiles_stream*.bin               (firehose lane)
        <stem>_stream*_bocpd.jsonl                   (telemetry lane)
        <stem>_stream*_asc_trajectory.bin            (telemetry lane)
        <stem>_stream*_warp_matrix.bin               (vram-teardown lane)
        <stem>_stream*_protocol_state.json           (vram-teardown lane)
        <stem>.aromatic_centroids_map.json
        <stem>.prism_therm_telemetry.json

CLAUDE.md immutable rules respected:
    - NO composite scores in ranking. Lexicographic only:
        (recurrence_density desc,
         kcc_summary.mean_max_kl desc,
         lining_residue_count desc).
    - Ranking is the only ordering authority.

Usage (do NOT execute on real data without separate operator authorization):
    python3 scripts/quarantine/prism-aggregate-sites.py \
        --run-dir /mnt/storage/phase_A_monomer_smoke \
        --topology data/targets/mpro_monomer.topology.json \
        --output cryptic_sites.json \
        -v
"""

from __future__ import annotations

import argparse
import datetime as _dt
import glob
import json
import logging
import os
import struct
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

# ---------------------------------------------------------------------------
# Constants pinned to crates/prism-nhs/src/ghost_tile.rs (Amendment 3.14 v9D')
# ---------------------------------------------------------------------------

GHOST_RECORD_BYTES = 4096
GHOST_COUNTER_SECTOR = 4096

# Field offsets inside one GhostTileFrame record (0..256 used; rest is slack).
GHOST_OFF_FRAME_IDX           = 0    # u64
GHOST_OFF_SITE_ID             = 8    # u32
GHOST_OFF_CHAIN_ID            = 12   # u8
GHOST_OFF_ADJ_CODE            = 13   # u8
GHOST_OFF_TELEMETRY_FLAGS     = 14   # u16
GHOST_OFF_KL                  = 16   # f32
GHOST_OFF_POWER_SPECTRUM      = 20   # f32 * 24
GHOST_OFF_THERMO_FLUX         = 116  # f32 * 2
GHOST_OFF_CAUSAL_LEAD_RESIDUE = 124  # u32

GHOST_TELEMETRY_CLASS_TAINTED = 0x0001

# Schema version emitted in the output JSON.
OUTPUT_SCHEMA_VERSION = "prism-aggregate-sites/1.0"

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

log = logging.getLogger("prism-aggregate-sites")


def _setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    fmt = "%(asctime)s [%(levelname)s] %(message)s"
    logging.basicConfig(level=level, format=fmt, stream=sys.stderr, force=True)


# ---------------------------------------------------------------------------
# Defensive JSON load
# ---------------------------------------------------------------------------


def _load_json(path: Path) -> Optional[Any]:
    try:
        with path.open("r") as fh:
            return json.load(fh)
    except FileNotFoundError:
        log.debug("missing JSON: %s", path)
        return None
    except json.JSONDecodeError as exc:
        log.warning("malformed JSON %s: %s", path, exc)
        return None
    except OSError as exc:
        log.warning("could not read %s: %s", path, exc)
        return None


# ---------------------------------------------------------------------------
# Topology loader (residue_id -> (residue_name, chain_id))
# ---------------------------------------------------------------------------


@dataclass
class Topology:
    residue_id_to_name: Dict[int, str] = field(default_factory=dict)
    residue_id_to_chain: Dict[int, str] = field(default_factory=dict)
    n_residues: int = 0
    source_pdb: Optional[str] = None


def load_topology(path: Path) -> Topology:
    raw = _load_json(path)
    if raw is None:
        log.warning("topology %s could not be loaded — residue names will be UNK", path)
        return Topology()

    topo = Topology(source_pdb=raw.get("source_pdb"))

    # Preferred path: per-residue records.
    residues = raw.get("residues")
    if isinstance(residues, list) and residues:
        for entry in residues:
            try:
                rid = int(entry["residue_id"])
            except (KeyError, TypeError, ValueError):
                continue
            topo.residue_id_to_name[rid] = entry.get("residue_name", "UNK")

    # Fallback: parallel arrays at atom level — collapse to first-seen residue_name.
    if not topo.residue_id_to_name:
        rids = raw.get("residue_ids") or []
        rnames = raw.get("residue_names") or []
        for rid, rn in zip(rids, rnames):
            try:
                rid_i = int(rid)
            except (TypeError, ValueError):
                continue
            topo.residue_id_to_name.setdefault(rid_i, str(rn))

    # Chain mapping (atom-level chain_ids array, collapse to first-seen).
    rids = raw.get("residue_ids") or []
    chain_ids = raw.get("chain_ids") or []
    for rid, ch in zip(rids, chain_ids):
        try:
            rid_i = int(rid)
        except (TypeError, ValueError):
            continue
        if rid_i not in topo.residue_id_to_chain:
            topo.residue_id_to_chain[rid_i] = str(ch) if ch is not None else "?"

    topo.n_residues = int(raw.get("n_residues", len(topo.residue_id_to_name)))
    log.info(
        "topology: %d residues, %d chains, source=%s",
        topo.n_residues,
        len(set(topo.residue_id_to_chain.values())),
        topo.source_pdb,
    )
    return topo


# ---------------------------------------------------------------------------
# Ghost-tile firehose parser (binary)
# ---------------------------------------------------------------------------


@dataclass
class GhostRecord:
    frame_idx: int
    site_id: int
    chain_id: int
    adjudication_code: int
    telemetry_flags: int
    kl_divergence: float
    power_spectrum: List[float]
    thermo_flux: List[float]
    causal_lead_residue: int
    source_stream: int


def _parse_ghost_record(buf: bytes, stream_idx: int) -> Optional[GhostRecord]:
    if len(buf) < 256:
        return None
    try:
        frame_idx = struct.unpack_from("<Q", buf, GHOST_OFF_FRAME_IDX)[0]
        site_id = struct.unpack_from("<I", buf, GHOST_OFF_SITE_ID)[0]
        chain_id = struct.unpack_from("<B", buf, GHOST_OFF_CHAIN_ID)[0]
        adj_code = struct.unpack_from("<B", buf, GHOST_OFF_ADJ_CODE)[0]
        tflags = struct.unpack_from("<H", buf, GHOST_OFF_TELEMETRY_FLAGS)[0]
        kl = struct.unpack_from("<f", buf, GHOST_OFF_KL)[0]
        power = list(
            struct.unpack_from("<24f", buf, GHOST_OFF_POWER_SPECTRUM)
        )
        thermo = list(
            struct.unpack_from("<2f", buf, GHOST_OFF_THERMO_FLUX)
        )
        causal = struct.unpack_from("<I", buf, GHOST_OFF_CAUSAL_LEAD_RESIDUE)[0]
    except struct.error as exc:
        log.debug("struct.unpack error stream=%d: %s", stream_idx, exc)
        return None
    return GhostRecord(
        frame_idx=frame_idx,
        site_id=site_id,
        chain_id=chain_id,
        adjudication_code=adj_code,
        telemetry_flags=tflags,
        kl_divergence=kl,
        power_spectrum=power,
        thermo_flux=thermo,
        causal_lead_residue=causal,
        source_stream=stream_idx,
    )


def parse_ghost_bin(path: Path, stream_idx: int) -> List[GhostRecord]:
    """Read a Channel-B firehose file. Returns [] if absent / empty."""
    try:
        size = path.stat().st_size
    except FileNotFoundError:
        log.debug("ghost bin missing: %s", path)
        return []
    if size == 0:
        log.info("ghost bin %s is empty (0 bytes)", path)
        return []
    if size < GHOST_COUNTER_SECTOR:
        log.warning(
            "ghost bin %s smaller than counter sector (%d bytes) — skip",
            path,
            size,
        )
        return []
    records: List[GhostRecord] = []
    try:
        with path.open("rb") as fh:
            counter_sector = fh.read(GHOST_COUNTER_SECTOR)
            n_written = struct.unpack_from("<I", counter_sector, 0)[0]
            max_records_in_file = (size - GHOST_COUNTER_SECTOR) // GHOST_RECORD_BYTES
            n = min(int(n_written), int(max_records_in_file))
            log.info(
                "ghost bin %s: counter=%d, file capacity=%d, parsing %d",
                path.name,
                n_written,
                max_records_in_file,
                n,
            )
            for _ in range(n):
                buf = fh.read(GHOST_RECORD_BYTES)
                if len(buf) < GHOST_RECORD_BYTES:
                    break
                rec = _parse_ghost_record(buf, stream_idx)
                if rec is not None:
                    records.append(rec)
    except OSError as exc:
        log.warning("could not read ghost bin %s: %s", path, exc)
        return []
    return records


def discover_ghost_bins(run_dir: Path, stem: str) -> List[Tuple[Path, int]]:
    """Find all ghost bins. Returns (path, stream_idx)."""
    out: List[Tuple[Path, int]] = []
    # Firehose lane: <stem>_ghost_tiles_streamN.bin
    pat = str(run_dir / f"{stem}_ghost_tiles_stream*.bin")
    for p in sorted(glob.glob(pat)):
        path = Path(p)
        # extract N from "..._streamN.bin"
        try:
            base = path.stem  # "<stem>_ghost_tiles_streamN"
            stream_idx = int(base.split("_stream")[-1])
        except (ValueError, IndexError):
            stream_idx = -1
        out.append((path, stream_idx))
    # Legacy single-stream file (current production).
    legacy = run_dir / f"{stem}_ghost_tiles.bin"
    if legacy.exists():
        out.append((legacy, 0))
    return out


# ---------------------------------------------------------------------------
# Optional telemetry / vram-teardown sidecar discovery
# ---------------------------------------------------------------------------


def discover_optional_sidecars(run_dir: Path, stem: str) -> Dict[str, List[Path]]:
    patterns = {
        "bocpd_jsonl":         f"{stem}_stream*_bocpd.jsonl",
        "asc_trajectory_bin":  f"{stem}_stream*_asc_trajectory.bin",
        "adj_flags_bin":       f"{stem}_stream*_adj_flags.bin",
        "adaptive_dt_bin":     f"{stem}_stream*_adaptive_dt.bin",
        "warp_matrix_bin":     f"{stem}_stream*_warp_matrix.bin",
        "forces_final_bin":    f"{stem}_stream*_forces_final.bin",
        "aromatic_final_bin":  f"{stem}_stream*_aromatic_centroids_final.bin",
        "protocol_state_json": f"{stem}_stream*_protocol_state.json",
    }
    out: Dict[str, List[Path]] = {}
    for key, pat in patterns.items():
        matches = sorted(Path(p) for p in glob.glob(str(run_dir / pat)))
        out[key] = matches
        if matches:
            log.info("sidecar %-22s: %d file(s)", key, len(matches))
    return out


def parse_bocpd_changepoints(paths: Sequence[Path]) -> Dict[int, int]:
    """Count BOCPD changepoints per cluster_id across all provided JSONL files.

    Each JSONL line is expected to be {"frame": int, "cluster_id": int,
    "changepoint": bool, ...}. Schema is GUESSED until the telemetry lane lands;
    we degrade gracefully on any KeyError.
    """
    counts: Dict[int, int] = defaultdict(int)
    for p in paths:
        try:
            with p.open("r") as fh:
                for raw_line in fh:
                    raw_line = raw_line.strip()
                    if not raw_line:
                        continue
                    try:
                        rec = json.loads(raw_line)
                    except json.JSONDecodeError:
                        continue
                    cid = rec.get("cluster_id", rec.get("site_id"))
                    cp = rec.get("changepoint", rec.get("is_changepoint"))
                    if cid is None or cp is None:
                        continue
                    if cp:
                        try:
                            counts[int(cid)] += 1
                        except (TypeError, ValueError):
                            continue
        except OSError as exc:
            log.warning("could not read bocpd jsonl %s: %s", p, exc)
    return dict(counts)


# ---------------------------------------------------------------------------
# Spatial-grid voxel parsing -> per-residue coupled_voxel_count
# ---------------------------------------------------------------------------


def build_residue_voxel_index(
    spatial_grid: Optional[Dict[str, Any]],
    min_coupled: int,
) -> Tuple[Dict[int, int], Dict[int, List[Tuple[int, int, int]]]]:
    """Walk voxels[]; for each primary_residue_id, count voxels that meet the
    coupled-spike threshold and remember their (ix,iy,iz) cells.

    Returns:
        (residue_id -> n_qualifying_voxels,
         residue_id -> [(ix,iy,iz), ...])
    """
    counts: Dict[int, int] = defaultdict(int)
    cells: Dict[int, List[Tuple[int, int, int]]] = defaultdict(list)
    if not spatial_grid:
        return dict(counts), dict(cells)
    voxels = spatial_grid.get("voxels") or []
    for vx in voxels:
        try:
            csc = int(vx.get("coupled_spike_count", 0))
        except (TypeError, ValueError):
            csc = 0
        if csc < min_coupled:
            continue
        try:
            rid = int(vx["primary_residue_id"])
        except (KeyError, TypeError, ValueError):
            continue
        counts[rid] += 1
        try:
            cells[rid].append(
                (int(vx["ix"]), int(vx["iy"]), int(vx["iz"]))
            )
        except (KeyError, TypeError, ValueError):
            continue
    return dict(counts), dict(cells)


# ---------------------------------------------------------------------------
# Cluster aggregation across streams
# ---------------------------------------------------------------------------


@dataclass
class ClusterAccum:
    cluster_id: int
    streams: set = field(default_factory=set)
    frames: set = field(default_factory=set)
    chain_codes: List[int] = field(default_factory=list)
    kls: List[float] = field(default_factory=list)
    causal_leads: List[int] = field(default_factory=list)
    plane_l_means: List[List[float]] = field(default_factory=list)  # per-record plane sums
    construct_count: int = 0
    violation_count: int = 0
    tainted_count: int = 0
    record_count: int = 0
    # The set of residues that appeared as causal_lead across the cluster's
    # firehose history. Used as a seed lining residue set; augmented by
    # spatial-grid coverage post-cluster.
    lining_residue_ids: set = field(default_factory=set)
    lining_frame_counts: Dict[int, int] = field(default_factory=lambda: defaultdict(int))

    def absorb(self, rec: GhostRecord) -> None:
        self.streams.add(rec.source_stream)
        self.frames.add(rec.frame_idx)
        self.chain_codes.append(rec.chain_id)
        self.kls.append(rec.kl_divergence)
        self.plane_l_means.append(list(rec.power_spectrum))
        self.record_count += 1
        if rec.adjudication_code == 1:
            self.construct_count += 1
        elif rec.adjudication_code == 2:
            self.violation_count += 1
        if rec.telemetry_flags & GHOST_TELEMETRY_CLASS_TAINTED:
            self.tainted_count += 1
        if rec.causal_lead_residue != 0xFFFFFFFF:
            self.causal_leads.append(int(rec.causal_lead_residue))
            self.lining_residue_ids.add(int(rec.causal_lead_residue))
            self.lining_frame_counts[int(rec.causal_lead_residue)] += 1


def aggregate_records(
    records: Iterable[GhostRecord],
) -> Dict[int, ClusterAccum]:
    out: Dict[int, ClusterAccum] = {}
    for rec in records:
        cid = int(rec.site_id)
        acc = out.get(cid)
        if acc is None:
            acc = ClusterAccum(cluster_id=cid)
            out[cid] = acc
        acc.absorb(rec)
    return out


def merge_by_centroid(
    accums: Dict[int, ClusterAccum],
    cluster_centroids: Dict[int, Tuple[float, float, float]],
    eps: float,
) -> List[List[int]]:
    """Group cluster_ids whose centroids are within eps Angstroms.

    Greedy single-link clustering. Returns groups of cluster_ids.
    """
    if eps <= 0.0:
        return [[cid] for cid in accums]
    groups: List[List[int]] = []
    used: set = set()
    cids = list(accums.keys())
    for i, ci in enumerate(cids):
        if ci in used:
            continue
        group = [ci]
        used.add(ci)
        ci_centroid = cluster_centroids.get(ci)
        if ci_centroid is None:
            groups.append(group)
            continue
        for cj in cids[i + 1 :]:
            if cj in used:
                continue
            cj_centroid = cluster_centroids.get(cj)
            if cj_centroid is None:
                continue
            dx = ci_centroid[0] - cj_centroid[0]
            dy = ci_centroid[1] - cj_centroid[1]
            dz = ci_centroid[2] - cj_centroid[2]
            if (dx * dx + dy * dy + dz * dz) <= (eps * eps):
                group.append(cj)
                used.add(cj)
        groups.append(group)
    return groups


# ---------------------------------------------------------------------------
# Centroid synthesis
#
# The firehose record does NOT carry a centroid (pos folded into the SO(3)
# spectrum). We approximate per-cluster centroids by averaging the initial
# atom positions of the cluster's lining residues' aromatic centroids if
# available, else by averaging causal-lead residue C-alpha positions from
# the topology. This stays defensive against missing data.
# ---------------------------------------------------------------------------


def estimate_cluster_centroid(
    accum: ClusterAccum,
    aromatic_map: Optional[Dict[str, Any]],
    topology_raw: Optional[Dict[str, Any]],
) -> Optional[Tuple[float, float, float]]:
    points: List[Tuple[float, float, float]] = []

    # 1) aromatic centroids whose residue_id appears as causal lead.
    if aromatic_map:
        arom_by_rid: Dict[int, Tuple[float, float, float]] = {}
        for entry in aromatic_map.get("aromatics") or []:
            try:
                rid = int(entry["residue_id"])
                arom_by_rid[rid] = (
                    float(entry["initial_cx_ang"]),
                    float(entry["initial_cy_ang"]),
                    float(entry["initial_cz_ang"]),
                )
            except (KeyError, TypeError, ValueError):
                continue
        for rid in accum.lining_residue_ids:
            pt = arom_by_rid.get(rid)
            if pt is not None:
                points.append(pt)

    # 2) Topology positions[] for the lining residues (CA atoms preferred).
    if not points and topology_raw is not None:
        positions = topology_raw.get("positions")
        ca_indices = topology_raw.get("ca_indices")
        residue_ids = topology_raw.get("residue_ids") or []
        if (
            isinstance(positions, list)
            and isinstance(ca_indices, list)
            and isinstance(residue_ids, list)
        ):
            # CA index -> residue_id map
            ca_to_rid: Dict[int, int] = {}
            for ca_idx in ca_indices:
                try:
                    ca_i = int(ca_idx)
                    if 0 <= ca_i < len(residue_ids):
                        ca_to_rid[ca_i] = int(residue_ids[ca_i])
                except (TypeError, ValueError):
                    continue
            wanted = accum.lining_residue_ids
            for ca_i, rid in ca_to_rid.items():
                if rid not in wanted:
                    continue
                try:
                    pos = positions[ca_i]
                    points.append(
                        (float(pos[0]), float(pos[1]), float(pos[2]))
                    )
                except (IndexError, TypeError, ValueError):
                    continue

    if not points:
        return None
    n = float(len(points))
    return (
        sum(p[0] for p in points) / n,
        sum(p[1] for p in points) / n,
        sum(p[2] for p in points) / n,
    )


# ---------------------------------------------------------------------------
# Site construction from merged cluster groups
# ---------------------------------------------------------------------------


@dataclass
class AggregatedSite:
    cluster_ids: List[int]
    centroid: Optional[Tuple[float, float, float]]
    lining_residues: List[Dict[str, Any]]
    driver_residue_id: Optional[int]
    site_volume: float
    coupled_voxel_count: int
    recurrence_density: float
    causality_density: float
    therm_class: str
    kcc_summary: Dict[str, float]
    aggregation_telemetry: Dict[str, Any]


def _kcc_summary_from_residues(
    residues: List[Dict[str, Any]],
    kcc_residues: Dict[int, Dict[str, Any]],
) -> Dict[str, float]:
    spike_densities: List[float] = []
    n_active = 0
    for r in residues:
        rid = r.get("residue_id")
        if rid is None:
            continue
        kr = kcc_residues.get(int(rid))
        if not kr:
            continue
        sd = kr.get("spike_density")
        if isinstance(sd, (int, float)):
            spike_densities.append(float(sd))
            if sd > 0:
                n_active += 1
    mean_spike = (sum(spike_densities) / len(spike_densities)) if spike_densities else 0.0
    return {
        "mean_max_kl": 0.0,           # filled in by caller from ghost spectrum
        "mean_spike_density": mean_spike,
        "n_residues_active": float(n_active),
    }


def _classify_therm(
    accum_group: List[ClusterAccum],
    construct_total: int,
    violation_total: int,
) -> str:
    """Heuristic therm class derived from adj_code mix and ghost activity.

    - PURE_VIOLATION: only Violation events recorded.
    - CRYPTIC:       has Construct events but at low rate (< 5 per cluster).
    - PERSISTENT:    construct events >= 5 per cluster on average.
    - UNKNOWN:       no records.
    """
    if construct_total + violation_total == 0:
        return "UNKNOWN"
    if construct_total == 0:
        return "PURE_VIOLATION"
    n_cluster = max(1, len(accum_group))
    per_cluster = construct_total / n_cluster
    if per_cluster < 5:
        return "CRYPTIC"
    return "PERSISTENT"


def build_sites(
    accums: Dict[int, ClusterAccum],
    centroids: Dict[int, Tuple[float, float, float]],
    groups: List[List[int]],
    voxel_counts_by_residue: Dict[int, int],
    voxel_cells_by_residue: Dict[int, List[Tuple[int, int, int]]],
    kcc_residues: Dict[int, Dict[str, Any]],
    topology: Topology,
    bocpd_changepoints: Dict[int, int],
    min_recurrence: int,
) -> List[AggregatedSite]:
    sites: List[AggregatedSite] = []
    for group in groups:
        members = [accums[c] for c in group if c in accums]
        if not members:
            continue
        # Recurrence: union of frames across the group's clusters.
        all_frames: set = set()
        for m in members:
            all_frames.update(m.frames)
        n_recurrence = len(all_frames)
        if n_recurrence < min_recurrence:
            continue

        # Lining residue union, with frame counts summed per residue.
        lining_counts: Dict[int, int] = defaultdict(int)
        for m in members:
            for rid, c in m.lining_frame_counts.items():
                lining_counts[rid] += c

        lining_residues = [
            {
                "residue_id": rid,
                "residue_name": topology.residue_id_to_name.get(rid, "UNK"),
                "chain_id": topology.residue_id_to_chain.get(rid, "?"),
                "n_contact_frames": int(cnt),
            }
            for rid, cnt in sorted(
                lining_counts.items(), key=lambda kv: kv[1], reverse=True
            )
        ]

        # Driver residue: most frequent causal-lead across the group.
        driver_residue_id: Optional[int] = None
        if lining_counts:
            driver_residue_id = max(lining_counts.items(), key=lambda kv: kv[1])[0]

        # Coupled voxel count for the union of lining residues.
        coupled_voxels = 0
        unique_cells: set = set()
        for rid in lining_counts:
            coupled_voxels += int(voxel_counts_by_residue.get(rid, 0))
            for cell in voxel_cells_by_residue.get(rid, []):
                unique_cells.add(cell)
        # Site "volume" = cell count (caller can multiply by voxel_edge later).
        site_volume = float(len(unique_cells))

        # Centroid: average of member-cluster centroids if available.
        cent_pts = [centroids[c] for c in group if c in centroids and centroids[c] is not None]
        if cent_pts:
            site_centroid = (
                sum(p[0] for p in cent_pts) / len(cent_pts),
                sum(p[1] for p in cent_pts) / len(cent_pts),
                sum(p[2] for p in cent_pts) / len(cent_pts),
            )
        else:
            site_centroid = None

        # KCC summary.
        construct_total = sum(m.construct_count for m in members)
        violation_total = sum(m.violation_count for m in members)
        record_total = sum(m.record_count for m in members)
        tainted_total = sum(m.tainted_count for m in members)
        all_kls: List[float] = []
        for m in members:
            all_kls.extend(m.kls)
        mean_max_kl = (max(all_kls) if all_kls else 0.0)
        kcc_summary = _kcc_summary_from_residues(lining_residues, kcc_residues)
        kcc_summary["mean_max_kl"] = float(mean_max_kl)

        # Densities (float; ranking is lex on the scalar).
        recurrence_density = (
            float(n_recurrence) / float(record_total) if record_total else 0.0
        )
        # causality_density: fraction of records that resolved a causal lead.
        n_causal = sum(len(m.causal_leads) for m in members)
        causality_density = (
            float(n_causal) / float(record_total) if record_total else 0.0
        )

        # Therm class.
        therm_class = _classify_therm(members, construct_total, violation_total)

        # BOCPD changepoint roll-up across the cluster group (telemetry lane).
        bocpd_total = sum(bocpd_changepoints.get(c, 0) for c in group)

        # Druggability score: NOT a ranking input. Reported as a derived
        # diagnostic only (lining-residue size scaled by recurrence). Keeping
        # the field present so downstream consumers don't NPE; rank uses
        # recurrence_density / KCC / lining_count, never this.
        druggability_score = float(len(lining_residues)) * recurrence_density

        site = AggregatedSite(
            cluster_ids=sorted(group),
            centroid=site_centroid,
            lining_residues=lining_residues,
            driver_residue_id=driver_residue_id,
            site_volume=site_volume,
            coupled_voxel_count=coupled_voxels,
            recurrence_density=recurrence_density,
            causality_density=causality_density,
            therm_class=therm_class,
            kcc_summary=kcc_summary,
            aggregation_telemetry={
                "n_clusters": len(members),
                "n_unique_frames": n_recurrence,
                "n_records": record_total,
                "n_construct": construct_total,
                "n_violation": violation_total,
                "n_tainted": tainted_total,
                "n_unique_voxel_cells": len(unique_cells),
                "n_bocpd_changepoints": bocpd_total,
                "druggability_score": druggability_score,
            },
        )
        sites.append(site)

    return sites


def rank_sites_lexicographic(sites: List[AggregatedSite]) -> List[AggregatedSite]:
    """Lex order: recurrence_density desc, mean_max_kl desc, lining_count desc.
    Per CLAUDE.md IMMUTABLE RULES: NO composite scores in ranking.
    """
    return sorted(
        sites,
        key=lambda s: (
            -float(s.recurrence_density),
            -float(s.kcc_summary.get("mean_max_kl", 0.0)),
            -int(len(s.lining_residues)),
        ),
    )


# ---------------------------------------------------------------------------
# Output assembly
# ---------------------------------------------------------------------------


def site_to_json(site: AggregatedSite, rank: int, idx: int) -> Dict[str, Any]:
    return {
        "site_id": f"agg_site_{idx + 1}",
        "site_rank": rank,
        "site_centroid": list(site.centroid) if site.centroid else None,
        "lining_residues": site.lining_residues,
        "driver_residue_id": site.driver_residue_id,
        "site_volume": site.site_volume,
        "druggability_score": site.aggregation_telemetry["druggability_score"],
        "kcc_summary": site.kcc_summary,
        "coupled_voxel_count": site.coupled_voxel_count,
        "recurrence_density": site.recurrence_density,
        "causality_density": site.causality_density,
        "therm_class": site.therm_class,
        "DCC_distance_to_native": None,           # requires native-pose lane
        "GT_ligand_shell_overlap_residues": None, # requires ground_truth lane
        "_provenance": {
            "source_cluster_ids": site.cluster_ids,
            "aggregation_telemetry": site.aggregation_telemetry,
        },
    }


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def _resolve_stem(run_dir: Path, topology_path: Path) -> str:
    """Pick a file stem. Prefer the topology basename without .topology.json;
    fall back to the run-dir glob of *.kcc_visualization.json.
    """
    name = topology_path.name
    for suffix in (".topology.json", ".json"):
        if name.endswith(suffix):
            stem = name[: -len(suffix)]
            return stem
    # Glob fallback.
    cand = sorted(run_dir.glob("*.kcc_visualization.json"))
    if cand:
        return cand[0].name[: -len(".kcc_visualization.json")]
    return topology_path.stem


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "PRISM-4D Null-Manifest post-hoc site reconstructor. "
            "Salvages site objects from preserved ASC/KCC/spatial-grid + "
            "ghost-firehose telemetry when the V2 12-sigma adjudicator "
            "did not trigger."
        )
    )
    parser.add_argument(
        "--run-dir",
        required=True,
        type=Path,
        help="Engine output directory (contains <stem>.kcc_visualization.json etc.)",
    )
    parser.add_argument(
        "--topology",
        required=True,
        type=Path,
        help="Topology JSON for residue_id -> name/chain resolution.",
    )
    parser.add_argument(
        "--output",
        default="cryptic_sites.json",
        type=Path,
        help="Output JSON path (default: cryptic_sites.json in CWD).",
    )
    parser.add_argument(
        "--min-cluster-recurrence",
        type=int,
        default=5,
        help="Minimum unique-frame count required to qualify a cluster as a site.",
    )
    parser.add_argument(
        "--min-coupled-voxels",
        type=int,
        default=2,
        help="Minimum coupled_spike_count for a voxel to count toward site_volume.",
    )
    parser.add_argument(
        "--clustering-eps",
        type=float,
        default=6.0,
        help="Angstrom radius for greedy centroid-merge across streams.",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Verbose logging (which inputs were loaded, cluster counts, etc.)",
    )
    args = parser.parse_args(argv)

    _setup_logging(args.verbose)

    run_dir: Path = args.run_dir
    if not run_dir.is_dir():
        log.error("--run-dir %s does not exist or is not a directory", run_dir)
        return 2
    if not args.topology.is_file():
        log.error("--topology %s is not a file", args.topology)
        return 2

    stem = _resolve_stem(run_dir, args.topology)
    log.info("run_dir=%s stem=%s", run_dir, stem)

    # ------------------------------------------------------------------
    # 1) Load all available inputs (defensive).
    # ------------------------------------------------------------------
    topology = load_topology(args.topology)
    topology_raw = _load_json(args.topology)

    kcc_path = run_dir / f"{stem}.kcc_visualization.json"
    kcc_raw = _load_json(kcc_path)
    if kcc_raw is None:
        log.warning("missing kcc_visualization at %s", kcc_path)
    kcc_residues_by_id: Dict[int, Dict[str, Any]] = {}
    if kcc_raw:
        for r in kcc_raw.get("residues") or []:
            try:
                kcc_residues_by_id[int(r["residue_id"])] = r
            except (KeyError, TypeError, ValueError):
                continue
        log.info(
            "kcc: n_residues=%d, n_consensus=%d, sites_in_file=%d",
            kcc_raw.get("n_residues", 0),
            kcc_raw.get("n_consensus", 0),
            len(kcc_raw.get("sites") or []),
        )

    binding_sites_raw = _load_json(run_dir / f"{stem}.binding_sites.json")

    # spatial grid: try both naming conventions.
    spatial_grid_raw = _load_json(run_dir / f"{stem}.spatial_grid_state.json")
    if spatial_grid_raw is None:
        spatial_grid_raw = _load_json(run_dir / f"{stem}_spatial_grid_state.json")
    if spatial_grid_raw is None:
        # last-ditch glob
        for cand in run_dir.glob("*spatial_grid*state*.json"):
            spatial_grid_raw = _load_json(cand)
            if spatial_grid_raw:
                log.info("spatial-grid resolved via glob: %s", cand.name)
                break
    if spatial_grid_raw:
        log.info(
            "spatial_grid: grid_dim=%s nonzero_voxels=%s",
            spatial_grid_raw.get("grid_dim"),
            spatial_grid_raw.get("nonzero_voxels"),
        )

    aromatic_raw = _load_json(run_dir / f"{stem}.aromatic_centroids_map.json")
    if aromatic_raw:
        log.info("aromatic_centroids: n=%d", aromatic_raw.get("n_aromatics", 0))

    # Optional sidecars.
    sidecars = discover_optional_sidecars(run_dir, stem)
    bocpd_counts = parse_bocpd_changepoints(sidecars.get("bocpd_jsonl") or [])
    if bocpd_counts:
        log.info("bocpd: changepoint counts collected for %d clusters", len(bocpd_counts))

    # Ghost firehose.
    ghost_files = discover_ghost_bins(run_dir, stem)
    if not ghost_files:
        log.warning("no ghost-tile bins found under %s — site reconstruction will be empty", run_dir)
    all_records: List[GhostRecord] = []
    for path, sidx in ghost_files:
        all_records.extend(parse_ghost_bin(path, sidx))
    log.info("ghost records parsed: %d (across %d file(s))", len(all_records), len(ghost_files))

    # ------------------------------------------------------------------
    # 2) Aggregate by (site_id, frame) -> per-cluster accumulators.
    # ------------------------------------------------------------------
    accums = aggregate_records(all_records)
    log.info("clusters before merge: %d", len(accums))

    # ------------------------------------------------------------------
    # 3) Estimate centroids and merge clusters within --clustering-eps.
    # ------------------------------------------------------------------
    centroids: Dict[int, Tuple[float, float, float]] = {}
    for cid, acc in accums.items():
        c = estimate_cluster_centroid(acc, aromatic_raw, topology_raw)
        if c is not None:
            centroids[cid] = c

    groups = merge_by_centroid(accums, centroids, args.clustering_eps)
    log.info("cluster groups after centroid merge (eps=%.2fA): %d", args.clustering_eps, len(groups))

    # ------------------------------------------------------------------
    # 4) Build site objects and rank lexicographically.
    # ------------------------------------------------------------------
    voxel_counts, voxel_cells = build_residue_voxel_index(
        spatial_grid_raw, args.min_coupled_voxels
    )
    sites = build_sites(
        accums=accums,
        centroids=centroids,
        groups=groups,
        voxel_counts_by_residue=voxel_counts,
        voxel_cells_by_residue=voxel_cells,
        kcc_residues=kcc_residues_by_id,
        topology=topology,
        bocpd_changepoints=bocpd_counts,
        min_recurrence=args.min_cluster_recurrence,
    )
    log.info("sites passing thresholds: %d (min_recurrence=%d)", len(sites), args.min_cluster_recurrence)

    sites_ranked = rank_sites_lexicographic(sites)

    # ------------------------------------------------------------------
    # 5) Emit cryptic_sites.json.
    # ------------------------------------------------------------------
    out_payload: Dict[str, Any] = {
        "schema_version": OUTPUT_SCHEMA_VERSION,
        "generated_at": _dt.datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "source_run_dir": str(run_dir.resolve()),
        "source_topology": str(args.topology.resolve()),
        "source_stem": stem,
        "n_sites": len(sites_ranked),
        "thresholds": {
            "min_cluster_recurrence": args.min_cluster_recurrence,
            "min_coupled_voxels": args.min_coupled_voxels,
            "clustering_eps_angstroms": args.clustering_eps,
        },
        "ranking_key": [
            "recurrence_density desc",
            "kcc_summary.mean_max_kl desc",
            "lining_residue_count desc",
        ],
        "ranking_policy": (
            "lexicographic only — NO composite scores per CLAUDE.md "
            "IMMUTABLE RULES §1-2"
        ),
        "inputs_present": {
            "kcc_visualization": kcc_raw is not None,
            "binding_sites": binding_sites_raw is not None,
            "spatial_grid_state": spatial_grid_raw is not None,
            "aromatic_centroids_map": aromatic_raw is not None,
            "ghost_tile_bins": [str(p) for p, _ in ghost_files],
            "bocpd_jsonl_files": [str(p) for p in sidecars.get("bocpd_jsonl") or []],
            "asc_trajectory_bins": [str(p) for p in sidecars.get("asc_trajectory_bin") or []],
            "warp_matrix_bins": [str(p) for p in sidecars.get("warp_matrix_bin") or []],
            "protocol_state_jsons": [str(p) for p in sidecars.get("protocol_state_json") or []],
        },
        "aggregation_stats": {
            "n_ghost_records_parsed": len(all_records),
            "n_clusters_observed": len(accums),
            "n_cluster_groups_after_merge": len(groups),
            "n_sites_pre_threshold": len(sites),
            "n_sites_post_threshold": len(sites_ranked),
            "n_residues_with_voxel_coverage": len(voxel_counts),
        },
        "sites": [
            site_to_json(site, rank=i + 1, idx=i)
            for i, site in enumerate(sites_ranked)
        ],
    }

    out_path: Path = args.output
    try:
        out_path.parent.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        log.error("could not create parent dir for %s: %s", out_path, exc)
        return 3
    try:
        with out_path.open("w") as fh:
            json.dump(out_payload, fh, indent=2)
    except OSError as exc:
        log.error("could not write output %s: %s", out_path, exc)
        return 4

    log.info("wrote %s (%d sites)", out_path, len(sites_ranked))
    return 0


if __name__ == "__main__":
    sys.exit(main())
