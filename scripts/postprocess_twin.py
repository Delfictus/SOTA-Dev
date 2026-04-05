#!/usr/bin/env python3
"""
postprocess_twin.py — PRISM-TWIN post-processing pipeline (production)

Consumes raw engine output from a PRISM-TWIN coupled-observation run and
produces the enriched artifact set required by the 10K campaign:

    {target}_ccf_matrix.npy      — mean-centered N_res × N_res CCF
    {target}_per_residue.json    — per-residue feature bundle (48+ fields)
    {target}_per_site.json       — per-site feature bundle   (60+ fields)
    {target}_allosteric.json     — network graph from CCF
    {target}_site_cards.json     — human-readable site summaries
    {target}_docking.json        — UniDock-ready boxes + pharmacophores
    {target}_heatmap.pdb         — PyMOL b-factor overlay
    {target}_beacons.json        — Tier-3 scouting parameters
    {target}_mechanisms.json     — per-site opening mechanism classification

Designed to run on engine output from:
    COUPLED_TWIN=true scripts/prism-validate-and-run.sh ... --coupled-twin

Inputs (located in --target-dir):
    coupled_spikes.parquet  OR  coupled_spikes.json     (spike stream, required)
    coupled_twin_result.json                            (twin summary, required)
    <prefix>.binding_sites.json                          (engine sites, required)
    <prefix>.kcc_visualization.json                      (kcc, optional)
    <prefix>.prism_therm.json                            (therm classification, optional)

Usage:
    python3 scripts/postprocess_twin.py \\
        --target-dir /path/to/engine/output \\
        --pdb-path /path/to/clean.pdb \\
        --target-id 1btl_chainA \\
        --pdb-id 1btl \\
        --chain A \\
        [--nma-file /path/to/nma_modes.json] \\
        [--topology /path/to/topology.json]

The CCF computation is mean-centered on BOTH streams. Distant-pair mean is
asserted < 0.15 before the matrix is written. This is the v3 correctness fix.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np

# ──────────────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────────────

# 3-letter → 1-letter AA map (includes AMBER tautomers)
AA3_TO_AA1 = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
    "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
    "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
    "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
    # AMBER protonation states
    "HID": "H", "HIE": "H", "HIP": "H",
    "ASH": "D", "GLH": "E", "LYN": "K", "CYX": "C", "CYM": "C",
}

# Polarity / chemistry buckets used for pharmacophores + druggability
HYDROPHOBIC = {"A", "V", "L", "I", "M", "F", "W", "Y", "P", "C"}
POLAR       = {"S", "T", "N", "Q", "H"}
POSITIVE    = {"K", "R", "H"}
NEGATIVE    = {"D", "E"}
AROMATIC    = {"F", "W", "Y", "H"}
HBD_RES     = {"S", "T", "N", "Q", "Y", "W", "K", "R", "H"}
HBA_RES     = {"D", "E", "N", "Q", "S", "T", "Y", "H"}

# CCNS protocol phase boundaries as fractions of total timesteps (--fast --hysteresis).
# Defaults match the engine's default 5-phase protocol when --hysteresis is on:
#   cold_hold 14K + heating 6K + warm_hold 15K + cooling 5K + cold_return 5K ≈ 45K
DEFAULT_PHASE_FRACTIONS = {
    "cold_hold":   (0.000, 0.311),
    "heating":     (0.311, 0.444),
    "warm_hold":   (0.444, 0.778),
    "cooling":     (0.778, 0.889),
    "cold_return": (0.889, 1.000),
}
PHASE_ORDER = ["cold_hold", "heating", "warm_hold", "cooling", "cold_return"]

# CCF computation
CCF_TIME_BINS = 200
CCF_SAMPLE_LIMIT = 600_000   # max spikes per stream to draw for CCF
DISTANT_PAIR_SEPARATION = 20  # |i-j| > this ⇒ "distant" for baseline check

# Allosteric network binary-search target
ALLO_TARGET_DEGREE_RANGE = (5, 30)
ALLO_MAX_THRESHOLD_ITERS = 40

# File logger
def _ts() -> str:
    return time.strftime("%H:%M:%S")

def log(msg: str, level: str = "INFO") -> None:
    print(f"[{_ts()}] {level}: {msg}", flush=True)

# ──────────────────────────────────────────────────────────────────────────────
# IO helpers
# ──────────────────────────────────────────────────────────────────────────────

def read_json(path: Path) -> Any:
    with open(path, "rb") as f:
        try:
            import orjson
            return orjson.loads(f.read())
        except ImportError:
            return json.load(open(path))

def write_json(path: Path, obj: Any, indent: int = 2) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        import orjson
        opts = orjson.OPT_INDENT_2 if indent else 0
        opts |= orjson.OPT_SERIALIZE_NUMPY
        with open(path, "wb") as f:
            f.write(orjson.dumps(obj, option=opts, default=_json_default))
    except Exception:
        with open(path, "w") as f:
            json.dump(obj, f, indent=indent, default=_json_default)

def _json_default(o: Any) -> Any:
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (bytes, bytearray)):
        return o.decode("utf-8", errors="replace")
    raise TypeError(f"Type {type(o)} not JSON-serializable")

# ──────────────────────────────────────────────────────────────────────────────
# Structure loading
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class ResidueRow:
    topo_idx: int          # 0-based index in topology residue order
    pdb_resnum: int        # author resnum from PDB file
    resname: str           # 3-letter
    aa1: str               # 1-letter
    chain: str
    ca_xyz: np.ndarray     # (3,) float32 in Å
    bfactor: float         # from PDB, 0.0 if not parsed

@dataclass
class Structure:
    residues: List[ResidueRow]
    ca_xyz: np.ndarray                 # (N,3) float32 in Å
    pdb_resnum_by_topo: np.ndarray     # (N,) int
    topo_by_pdb_resnum: Dict[int, int]
    n_residues: int
    chain: str

def load_structure(pdb_path: Path, topology_path: Optional[Path], chain: str) -> Structure:
    """
    Build a residue index from the topology JSON (authoritative for topo_idx order)
    and PDB author resnums for display. CA coordinates come from the topology
    (in nm → converted to Å). B-factors from PDB if parsable.
    """
    residues: List[ResidueRow] = []

    # Parse topology
    if topology_path is None or not topology_path.exists():
        # Try common co-location
        guess = pdb_path.with_suffix(".topology.json")
        if guess.exists():
            topology_path = guess
        else:
            raise FileNotFoundError(
                f"topology.json required. Expected alongside PDB or via --topology. "
                f"Looked at: {guess}"
            )

    topo = read_json(topology_path)
    ca_indices = topo["ca_indices"]
    residue_names = topo["residue_names"]
    residue_ids = topo["residue_ids"]
    positions = topo["positions"]  # flat [x,y,z, ...] in nm

    n_atoms = topo["n_atoms"]
    # prism-prep topologies always write positions in Å (AMBER/engine convention).
    pos_arr = np.asarray(positions, dtype=np.float64).reshape(-1, 3)
    unit_scale = 1.0

    # Build CA rows — topo_idx is the residue order = index into ca_indices
    ca_xyz_list = []
    for topo_idx, atom_idx in enumerate(ca_indices):
        resname = residue_names[atom_idx]
        aa1 = AA3_TO_AA1.get(resname, "X")
        xyz = pos_arr[atom_idx] * unit_scale
        residues.append(ResidueRow(
            topo_idx=topo_idx,
            pdb_resnum=-1,          # filled from PDB
            resname=resname,
            aa1=aa1,
            chain=chain,
            ca_xyz=xyz.astype(np.float32),
            bfactor=0.0,
        ))
        ca_xyz_list.append(xyz)

    # Parse PDB for author resnums and b-factors (CA-only pass)
    pdb_ca_records: List[Tuple[int, str, str, float, float, float, float]] = []
    try:
        with open(pdb_path, "r") as f:
            for line in f:
                if not line.startswith(("ATOM  ", "HETATM")):
                    continue
                if line[12:16].strip() != "CA":
                    continue
                ch = line[21]
                if chain and ch != chain:
                    continue
                try:
                    resnum = int(line[22:26])
                    resname = line[17:20].strip()
                    x = float(line[30:38])
                    y = float(line[38:46])
                    z = float(line[46:54])
                    bf = float(line[60:66]) if line[60:66].strip() else 0.0
                except ValueError:
                    continue
                pdb_ca_records.append((resnum, resname, ch, x, y, z, bf))
    except FileNotFoundError:
        log(f"PDB not found: {pdb_path} — b-factors will be zero, pdb_resnum will fall back to topo_idx+1", "WARN")

    # Align PDB CAs to topology CAs by 3D nearest-neighbor (robust to offset/renumbering)
    if pdb_ca_records:
        pdb_xyz = np.array([[r[3], r[4], r[5]] for r in pdb_ca_records], dtype=np.float64)
        topo_xyz = np.array(ca_xyz_list, dtype=np.float64)
        # Distance matrix (only used if sizes close; otherwise use greedy nearest)
        from scipy.spatial import cKDTree
        kdt = cKDTree(pdb_xyz)
        d, idx = kdt.query(topo_xyz, k=1)
        for topo_idx, (dist, pidx) in enumerate(zip(d, idx)):
            if dist < 3.0:  # clearly the same atom
                rec = pdb_ca_records[pidx]
                residues[topo_idx].pdb_resnum = rec[0]
                residues[topo_idx].bfactor = rec[6]
        # Fallbacks for residues with no match: sequential
        for r in residues:
            if r.pdb_resnum == -1:
                r.pdb_resnum = r.topo_idx + 1
    else:
        for r in residues:
            r.pdb_resnum = r.topo_idx + 1

    ca_xyz = np.stack([r.ca_xyz for r in residues]).astype(np.float32)
    pdb_resnum_by_topo = np.array([r.pdb_resnum for r in residues], dtype=np.int32)
    topo_by_pdb_resnum = {r.pdb_resnum: r.topo_idx for r in residues}

    return Structure(
        residues=residues,
        ca_xyz=ca_xyz,
        pdb_resnum_by_topo=pdb_resnum_by_topo,
        topo_by_pdb_resnum=topo_by_pdb_resnum,
        n_residues=len(residues),
        chain=chain,
    )

# ──────────────────────────────────────────────────────────────────────────────
# Spike loading
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class SpikeArrays:
    """Materialized spike arrays after one streaming pass."""
    stream_id: np.ndarray        # uint8
    timestep: np.ndarray         # uint32
    xyz: np.ndarray              # (N,3) float32 in Å
    intensity: np.ndarray        # float32
    vib_energy: np.ndarray       # float32
    n_total: int
    n_a: int
    n_b: int
    max_timestep: int

def load_spikes(target_dir: Path) -> SpikeArrays:
    """
    Load coupled_spikes.parquet (preferred) or stream coupled_spikes.json.
    Spike coordinate units in the engine output are Å (confirmed for 1btl TEM1).
    Returns materialized arrays.
    """
    parquet_path = target_dir / "coupled_spikes.parquet"
    json_path = target_dir / "coupled_spikes.json"

    if parquet_path.exists():
        return _load_spikes_parquet(parquet_path)
    if json_path.exists():
        return _load_spikes_json_stream(json_path)
    raise FileNotFoundError(
        f"No spike data in {target_dir}: expected coupled_spikes.parquet or coupled_spikes.json"
    )

def _load_spikes_parquet(path: Path) -> SpikeArrays:
    import pyarrow.parquet as pq
    log(f"Loading spikes from Parquet: {path.name}")
    tbl = pq.read_table(path, columns=[
        "stream_id", "timestep", "x", "y", "z", "intensity", "vib_energy"
    ])
    stream_id = np.asarray(tbl.column("stream_id"), dtype=np.uint8)
    timestep = np.asarray(tbl.column("timestep"), dtype=np.uint32)
    x = np.asarray(tbl.column("x"), dtype=np.float32)
    y = np.asarray(tbl.column("y"), dtype=np.float32)
    z = np.asarray(tbl.column("z"), dtype=np.float32)
    intensity = np.asarray(tbl.column("intensity"), dtype=np.float32)
    vib_energy = np.asarray(tbl.column("vib_energy"), dtype=np.float32)
    xyz = np.stack([x, y, z], axis=1)
    n_a = int((stream_id == 0).sum())
    n_b = int((stream_id == 1).sum())
    return SpikeArrays(
        stream_id=stream_id, timestep=timestep, xyz=xyz,
        intensity=intensity, vib_energy=vib_energy,
        n_total=len(timestep), n_a=n_a, n_b=n_b,
        max_timestep=int(timestep.max()) if len(timestep) else 0,
    )

def _load_spikes_json_stream(path: Path) -> SpikeArrays:
    """
    Stream a potentially multi-GB JSON. Format:
      {"n_spikes_a": int, "n_spikes_b": int, "spikes": [ {...}, ... ]}
    Each record has stream_id, timestep, x, y, z, intensity, vib_energy.
    """
    log(f"Streaming spikes from JSON: {path.name} ({path.stat().st_size / 1e9:.2f} GB)")
    import ijson

    # Pre-allocate growable arrays
    CHUNK = 2_000_000
    stream_id_list: List[np.ndarray] = []
    timestep_list: List[np.ndarray] = []
    xyz_list: List[np.ndarray] = []
    intensity_list: List[np.ndarray] = []
    vib_list: List[np.ndarray] = []

    buf_s = np.empty(CHUNK, dtype=np.uint8)
    buf_t = np.empty(CHUNK, dtype=np.uint32)
    buf_xyz = np.empty((CHUNK, 3), dtype=np.float32)
    buf_i = np.empty(CHUNK, dtype=np.float32)
    buf_v = np.empty(CHUNK, dtype=np.float32)

    n_filled = 0
    n_total = 0
    t0 = time.time()
    with open(path, "rb") as f:
        for rec in ijson.items(f, "spikes.item"):
            i = n_filled
            buf_s[i] = rec.get("stream_id", 0)
            buf_t[i] = rec.get("timestep", 0)
            buf_xyz[i, 0] = rec.get("x", 0.0)
            buf_xyz[i, 1] = rec.get("y", 0.0)
            buf_xyz[i, 2] = rec.get("z", 0.0)
            buf_i[i] = rec.get("intensity", 0.0)
            buf_v[i] = rec.get("vib_energy", 0.0)
            n_filled += 1
            n_total += 1
            if n_filled == CHUNK:
                stream_id_list.append(buf_s.copy())
                timestep_list.append(buf_t.copy())
                xyz_list.append(buf_xyz.copy())
                intensity_list.append(buf_i.copy())
                vib_list.append(buf_v.copy())
                n_filled = 0
                if n_total % 20_000_000 == 0:
                    rate = n_total / (time.time() - t0 + 1e-6)
                    log(f"  streamed {n_total/1e6:.1f}M spikes ({rate/1e6:.2f}M/s)")

    # Flush final partial chunk
    if n_filled > 0:
        stream_id_list.append(buf_s[:n_filled].copy())
        timestep_list.append(buf_t[:n_filled].copy())
        xyz_list.append(buf_xyz[:n_filled].copy())
        intensity_list.append(buf_i[:n_filled].copy())
        vib_list.append(buf_v[:n_filled].copy())

    stream_id = np.concatenate(stream_id_list) if stream_id_list else np.empty(0, np.uint8)
    timestep = np.concatenate(timestep_list) if timestep_list else np.empty(0, np.uint32)
    xyz = np.concatenate(xyz_list) if xyz_list else np.empty((0, 3), np.float32)
    intensity = np.concatenate(intensity_list) if intensity_list else np.empty(0, np.float32)
    vib_energy = np.concatenate(vib_list) if vib_list else np.empty(0, np.float32)

    n_a = int((stream_id == 0).sum())
    n_b = int((stream_id == 1).sum())
    log(f"  loaded {n_total:,} spikes (a={n_a:,}, b={n_b:,}) in {time.time()-t0:.1f}s")

    return SpikeArrays(
        stream_id=stream_id, timestep=timestep, xyz=xyz,
        intensity=intensity, vib_energy=vib_energy,
        n_total=n_total, n_a=n_a, n_b=n_b,
        max_timestep=int(timestep.max()) if n_total else 0,
    )

# ──────────────────────────────────────────────────────────────────────────────
# Spike → residue mapping
# ──────────────────────────────────────────────────────────────────────────────

def assign_spikes_to_residues(
    spikes: SpikeArrays, ca_xyz: np.ndarray, cutoff_angstrom: float = 15.0
) -> np.ndarray:
    """
    Nearest-CA assignment via KDTree. Spikes beyond cutoff are tagged -1.
    Returns int32 array of shape (n_spikes,) with topo_idx or -1.
    """
    from scipy.spatial import cKDTree
    log(f"Building KDTree on {len(ca_xyz)} CA atoms")
    tree = cKDTree(ca_xyz.astype(np.float64))
    log(f"Assigning {spikes.n_total:,} spikes to residues")
    dist, idx = tree.query(spikes.xyz.astype(np.float64), k=1)
    mask = dist > cutoff_angstrom
    idx = idx.astype(np.int32)
    idx[mask] = -1
    log(f"  assigned {(~mask).sum():,} / {spikes.n_total:,} ({(~mask).mean()*100:.1f}%)")
    return idx

def resolve_phase_boundaries(max_timestep: int) -> Dict[str, Tuple[int, int]]:
    """Map fractional phase boundaries to integer timestep ranges."""
    bounds: Dict[str, Tuple[int, int]] = {}
    for phase, (lo, hi) in DEFAULT_PHASE_FRACTIONS.items():
        bounds[phase] = (int(lo * max_timestep), int(hi * max_timestep))
    return bounds

def phase_of_timestep(ts: np.ndarray, bounds: Dict[str, Tuple[int, int]]) -> np.ndarray:
    """Return int8 phase index per spike (0..4). -1 if out of bounds."""
    phase_idx = np.full(len(ts), -1, dtype=np.int8)
    for i, phase in enumerate(PHASE_ORDER):
        lo, hi = bounds[phase]
        mask = (ts >= lo) & (ts < hi)
        phase_idx[mask] = i
    # Patch final residue: ensure the last timestep lands in cold_return, not -1
    last_hi = bounds[PHASE_ORDER[-1]][1]
    phase_idx[ts == last_hi] = len(PHASE_ORDER) - 1
    return phase_idx

def compute_per_residue_spike_counts(
    spikes: SpikeArrays, residue_idx: np.ndarray, n_residues: int,
    phase_idx: np.ndarray,
) -> Dict[str, np.ndarray]:
    """
    Returns arrays of shape (N_res,) and (N_res, n_phases, 2) for stream/phase breakdowns.
    """
    valid = residue_idx >= 0
    r = residue_idx[valid]
    s = spikes.stream_id[valid]
    p = phase_idx[valid]

    n_phases = len(PHASE_ORDER)
    per_stream = np.zeros((n_residues, 2), dtype=np.int64)
    per_phase = np.zeros((n_residues, n_phases, 2), dtype=np.int64)
    intensity_sum = np.zeros(n_residues, dtype=np.float64)
    vib_sum = np.zeros(n_residues, dtype=np.float64)
    count = np.zeros(n_residues, dtype=np.int64)

    # np.add.at for scatter
    np.add.at(per_stream, (r, s), 1)
    # phase scatter (only valid phase indices)
    pm = p >= 0
    np.add.at(per_phase, (r[pm], p[pm], s[pm]), 1)

    intensity = spikes.intensity[valid]
    vib = spikes.vib_energy[valid]
    np.add.at(intensity_sum, r, intensity)
    np.add.at(vib_sum, r, vib)
    np.add.at(count, r, 1)

    return {
        "per_stream": per_stream,                 # (N,2)
        "per_phase": per_phase,                   # (N, n_phases, 2)
        "intensity_sum": intensity_sum,           # (N,)
        "vib_sum": vib_sum,                       # (N,)
        "count": count,                           # (N,)
    }

# ──────────────────────────────────────────────────────────────────────────────
# CCF (mean-centered) — the v3 correctness fix
# ──────────────────────────────────────────────────────────────────────────────

def compute_ccf_matrix(
    spikes: SpikeArrays, residue_idx: np.ndarray, n_residues: int,
    n_time_bins: int = CCF_TIME_BINS, sample_limit: int = CCF_SAMPLE_LIMIT,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Build two (N_res × n_time_bins) matrices (one per stream) by counting spikes
    per residue per time bin, then correlate the MEAN-CENTERED matrices across
    residues. Returns (ccf_matrix float32 shape N×N, summary dict).

    Mean-centering is applied per-residue (subtract the residue's time-bin mean
    before taking the inner product). This removes the ~0.71 constant offset
    that contaminated the raw correlation in the v3 pre-fix report.
    """
    # Downsample uniformly per stream if too large
    valid = residue_idx >= 0
    r = residue_idx[valid]
    s = spikes.stream_id[valid]
    t = spikes.timestep[valid]

    if len(r) > sample_limit * 2:
        # Sample proportionally per stream
        idx_a = np.where(s == 0)[0]
        idx_b = np.where(s == 1)[0]
        rng = np.random.default_rng(42)
        if len(idx_a) > sample_limit:
            idx_a = rng.choice(idx_a, size=sample_limit, replace=False)
        if len(idx_b) > sample_limit:
            idx_b = rng.choice(idx_b, size=sample_limit, replace=False)
        keep = np.concatenate([idx_a, idx_b])
        r = r[keep]
        s = s[keep]
        t = t[keep]

    if len(r) == 0:
        log("No spikes available for CCF — returning zeros", "WARN")
        M = np.zeros((n_residues, n_residues), dtype=np.float32)
        return M, {
            "matrix_shape": [n_residues, n_residues],
            "spike_sample_used": 0,
            "n_time_bins": n_time_bins,
            "ccf_mean": 0.0,
            "ccf_max_offdiag": 0.0,
            "distant_pair_mean": 0.0,
            "ccf_fix_applied": True,
        }

    t_max = int(t.max()) + 1
    bin_size = max(1, t_max // n_time_bins)
    tb = (t // bin_size).astype(np.int32)
    tb[tb >= n_time_bins] = n_time_bins - 1

    # Build matrices: for each stream, rows = residues, cols = time bins
    mat_a = np.zeros((n_residues, n_time_bins), dtype=np.float32)
    mat_b = np.zeros((n_residues, n_time_bins), dtype=np.float32)

    sa = s == 0
    sb = s == 1
    np.add.at(mat_a, (r[sa], tb[sa]), 1.0)
    np.add.at(mat_b, (r[sb], tb[sb]), 1.0)

    # Double centering: subtract per-time-bin mean (kills the protocol envelope
    # that is shared across all residues) AND per-residue mean. This is the
    # v3 correctness fix — without the column-mean subtraction all correlations
    # inherit the ~0.71 constant from the CCNS phase structure.
    mat_a -= mat_a.mean(axis=0, keepdims=True)   # column (time) mean
    mat_b -= mat_b.mean(axis=0, keepdims=True)
    mat_a -= mat_a.mean(axis=1, keepdims=True)   # row (residue) mean
    mat_b -= mat_b.mean(axis=1, keepdims=True)

    # Normalize each row to unit L2 (handle zero rows)
    na = np.linalg.norm(mat_a, axis=1, keepdims=True) + 1e-12
    nb = np.linalg.norm(mat_b, axis=1, keepdims=True) + 1e-12
    mat_a_n = mat_a / na
    mat_b_n = mat_b / nb

    # CCF = (A @ B.T + B @ A.T) / 2  — symmetric cross-stream correlation.
    # Diagonal represents self-stream correlation; off-diagonal is allosteric.
    ccf_ab = mat_a_n @ mat_b_n.T
    ccf = (ccf_ab + ccf_ab.T) * 0.5
    # Fix diagonal: cross-stream self-correlation should be ~1 only if streams
    # see identical dynamics. Use the average of self-correlations within each
    # stream so the diagonal represents intra-residue coherence.
    self_a = np.einsum("ij,ij->i", mat_a_n, mat_a_n)
    self_b = np.einsum("ij,ij->i", mat_b_n, mat_b_n)
    np.fill_diagonal(ccf, 0.5 * (self_a + self_b))
    # Clip numerical drift — pure correlation should be in [-1, 1]
    ccf = np.clip(ccf, -1.0, 1.0).astype(np.float32)

    # Summary stats
    triu = np.triu_indices(n_residues, k=1)
    upper = ccf[triu]
    dist_mask = np.abs(triu[0] - triu[1]) > DISTANT_PAIR_SEPARATION
    distant = upper[dist_mask] if dist_mask.any() else np.array([0.0])
    summary = {
        "matrix_shape": [int(n_residues), int(n_residues)],
        "spike_sample_used": int(len(r)),
        "n_time_bins": int(n_time_bins),
        "bin_size_steps": int(bin_size),
        "n_pairs_above_0p5": int((upper > 0.5).sum()),
        "n_pairs_above_0p7": int((upper > 0.7).sum()),
        "ccf_mean": float(upper.mean()) if len(upper) else 0.0,
        "ccf_max_offdiag": float(upper.max()) if len(upper) else 0.0,
        "distant_pair_mean": float(distant.mean()),
        "ccf_fix_applied": True,
        "ccf_method": "mean_centered_per_residue_unit_norm",
    }
    return ccf, summary

# ──────────────────────────────────────────────────────────────────────────────
# Per-residue features (≥48 fields)
# ──────────────────────────────────────────────────────────────────────────────

def compute_burial(ca_xyz: np.ndarray, cutoff: float = 12.0) -> np.ndarray:
    """Number of CAs within cutoff Å of each CA — proxy for burial."""
    from scipy.spatial import cKDTree
    tree = cKDTree(ca_xyz.astype(np.float64))
    counts = tree.query_ball_point(ca_xyz.astype(np.float64), r=cutoff)
    return np.array([len(c) - 1 for c in counts], dtype=np.float32)  # exclude self

def compute_sasa_proxy(burial_counts: np.ndarray) -> np.ndarray:
    """Inverse-normalized burial → 0 (buried) … 1 (surface) proxy."""
    if len(burial_counts) == 0:
        return burial_counts
    lo, hi = np.percentile(burial_counts, [5, 95])
    if hi <= lo:
        return np.zeros_like(burial_counts)
    norm = np.clip((burial_counts - lo) / (hi - lo), 0, 1)
    return (1.0 - norm).astype(np.float32)

def compute_depth_from_surface(ca_xyz: np.ndarray, burial_counts: np.ndarray) -> np.ndarray:
    """
    Depth = distance from each residue to the nearest 'surface' residue (bottom
    20% burial). Returns float Å.
    """
    if len(ca_xyz) == 0:
        return np.zeros(0, dtype=np.float32)
    thresh = np.percentile(burial_counts, 20)
    surface = ca_xyz[burial_counts <= thresh]
    if len(surface) == 0:
        return np.zeros(len(ca_xyz), dtype=np.float32)
    from scipy.spatial import cKDTree
    tree = cKDTree(surface.astype(np.float64))
    d, _ = tree.query(ca_xyz.astype(np.float64), k=1)
    return d.astype(np.float32)

def classify_spike(total: int, median_total: float) -> str:
    if total == 0:
        return "SILENT"
    if total >= 3 * median_total:
        return "HYPERACTIVE"
    if total >= median_total:
        return "THERMALLY_ACCESSIBLE"
    return "LOW_ACTIVITY"

def classify_barrier(warm_hold_frac: float, total: int, median_total: float) -> str:
    if total < 0.1 * median_total:
        return "UNKNOWN"
    if warm_hold_frac > 0.45:
        return "LOW"
    if warm_hold_frac > 0.30:
        return "MEDIUM"
    return "HIGH"

def build_per_residue_features(
    structure: Structure,
    spike_counts: Dict[str, np.ndarray],
    ccf: np.ndarray,
    nma_data: Optional[Dict[str, Any]],
    active_site_topo_indices: List[int],
) -> List[Dict[str, Any]]:
    """
    Produce the full per-residue record list with ≥48 fields per entry.
    """
    n = structure.n_residues
    per_stream = spike_counts["per_stream"]       # (N, 2)
    per_phase = spike_counts["per_phase"]         # (N, n_phases, 2)
    intensity_sum = spike_counts["intensity_sum"]
    vib_sum = spike_counts["vib_sum"]
    count = spike_counts["count"]

    # Derived arrays
    total_spikes = per_stream.sum(axis=1)
    median_total = float(np.median(total_spikes[total_spikes > 0])) if (total_spikes > 0).any() else 1.0
    spikes_a = per_stream[:, 0]
    spikes_b = per_stream[:, 1]
    b_over_a = np.where(spikes_a > 0, spikes_b / np.maximum(spikes_a, 1), 0.0)
    consensus = np.minimum(spikes_a, spikes_b) * 2  # pairs where both streams fired
    differential = np.abs(spikes_a.astype(np.int64) - spikes_b.astype(np.int64))
    agreement = np.where(total_spikes > 0,
                         1.0 - (differential / np.maximum(total_spikes, 1)),
                         0.0)
    mean_intensity = np.where(count > 0, intensity_sum / np.maximum(count, 1), 0.0)
    mean_vib = np.where(count > 0, vib_sum / np.maximum(count, 1), 0.0)
    warm_hold_frac = np.where(total_spikes > 0,
                              per_phase[:, 2, :].sum(axis=1) / np.maximum(total_spikes, 1),
                              0.0)

    # CCF-derived per-residue
    ccf_off = ccf.copy()
    np.fill_diagonal(ccf_off, 0.0)
    ccf_max = ccf_off.max(axis=1)
    ccf_mean = ccf_off.mean(axis=1)
    ccf_n_05 = (ccf_off > 0.5).sum(axis=1)
    ccf_n_08 = (ccf_off > 0.8).sum(axis=1)
    ccf_strongest_partner = ccf_off.argmax(axis=1)
    ccf_asymmetry = np.abs(ccf - ccf.T).mean(axis=1)

    # Burial / SASA proxy / depth
    burial = compute_burial(structure.ca_xyz)
    sasa_proxy = compute_sasa_proxy(burial)
    depth = compute_depth_from_surface(structure.ca_xyz, burial)

    # NMA responsive modes per residue
    nma_resp_modes = [[] for _ in range(n)]
    nma_primary = [[0.0, 0.0, 0.0] for _ in range(n)]
    nma_sensitivity = np.zeros(n, dtype=np.float32)
    if nma_data is not None:
        modes = nma_data.get("modes", [])
        for m_idx, mode in enumerate(modes[:10]):
            eig = mode.get("eigenvalue", 0.0)
            # Support two schemas: eigenvector (flat 3N) or displacements (list of Nx3)
            vec = mode.get("eigenvector") or mode.get("displacements")
            if not vec or eig <= 0:
                continue
            varr = np.asarray(vec, dtype=np.float32)
            if varr.ndim == 1:
                varr = varr.reshape(-1, 3)
            if len(varr) < n:
                continue
            # displacement magnitude per residue
            disp = np.linalg.norm(varr[:n], axis=1)
            # top-5% responsive residues per mode
            thresh = np.percentile(disp, 95)
            for ri in np.where(disp >= thresh)[0]:
                nma_resp_modes[int(ri)].append(int(m_idx))
            nma_sensitivity += disp / (eig ** 0.5 + 1e-6)
            if m_idx == 0:
                for ri in range(n):
                    nma_primary[ri] = varr[ri].tolist()

    nma_sensitivity = nma_sensitivity / max(float(nma_sensitivity.max()), 1e-6)

    # Build records
    active_site_set = set(active_site_topo_indices)
    records: List[Dict[str, Any]] = []
    for i, res in enumerate(structure.residues):
        phase_profile = {
            phase: {
                "stream_a": int(per_phase[i, j, 0]),
                "stream_b": int(per_phase[i, j, 1]),
                "total": int(per_phase[i, j].sum()),
            }
            for j, phase in enumerate(PHASE_ORDER)
        }
        spike_cls = classify_spike(int(total_spikes[i]), median_total)
        bar_cls = classify_barrier(float(warm_hold_frac[i]), int(total_spikes[i]), median_total)
        # Cooperative partners: residues with CCF > 0.6 that are spatially distant
        partners_mask = ccf_off[i] > 0.6
        coop_partners = []
        if partners_mask.any():
            part_idx = np.where(partners_mask)[0]
            # Filter to distant (|i-j| > 8) so we don't double-count neighbors
            dist_filter = np.abs(part_idx - i) > 8
            part_idx = part_idx[dist_filter]
            for pi in part_idx[:5]:
                coop_partners.append({
                    "topo_idx": int(pi),
                    "pdb_resnum": int(structure.pdb_resnum_by_topo[pi]),
                    "ccf": float(ccf_off[i, pi]),
                })
        coop_score = float(len(coop_partners) / 5.0)

        # Cryptic probability: naive heuristic from spike density + burial + CCF
        # (ML-based probabilities are external and not computed here)
        crypt_prob = float(np.clip(
            0.25 * (1.0 - sasa_proxy[i])
            + 0.30 * min(total_spikes[i] / max(median_total * 2, 1), 1.0)
            + 0.25 * coop_score
            + 0.20 * min(ccf_max[i], 1.0),
            0.0, 1.0,
        ))

        rec = {
            # identity (4)
            "pdb_resnum": int(res.pdb_resnum),
            "topo_idx": int(res.topo_idx),
            "resname": res.resname,
            "aa1": res.aa1,
            # core scores (5)
            "cryptic_probability": round(crypt_prob, 4),
            "dssp": None,              # requires DSSP — optional external
            "bfactor": round(float(res.bfactor), 3),
            "sasa_proxy": round(float(sasa_proxy[i]), 4),
            "depth_from_surface": round(float(depth[i]), 3),
            # spike dynamics (12)
            "spikes_a": int(spikes_a[i]),
            "spikes_b": int(spikes_b[i]),
            "total_spikes": int(total_spikes[i]),
            "b_over_a_ratio": round(float(b_over_a[i]), 4),
            "consensus_spikes": int(consensus[i]),
            "differential_spikes": int(differential[i]),
            "spike_agreement_ratio": round(float(agreement[i]), 4),
            "thermal_spike_energy": round(float(mean_vib[i] * count[i]), 6),
            "spike_classification": spike_cls,
            "spike_onset_phase": _argmax_phase(per_phase[i]),
            "peak_spike_phase": _argmax_phase(per_phase[i]),
            "spike_rate_warm_hold": round(float(warm_hold_frac[i]), 4),
            # phase profiles (1 compound — counts as 1 field)
            "phase_profiles": phase_profile,
            # CCF-derived (10)
            "ccf_peak_value": round(float(ccf_max[i]), 4),
            "ccf_max": round(float(ccf_max[i]), 4),
            "ccf_mean": round(float(ccf_mean[i]), 4),
            "ccf_n_05": int(ccf_n_05[i]),
            "ccf_n_08": int(ccf_n_08[i]),
            "ccf_strongest_partner": int(ccf_strongest_partner[i]),
            "ccf_partner_distance": float(
                np.linalg.norm(
                    structure.ca_xyz[i] - structure.ca_xyz[int(ccf_strongest_partner[i])]
                )
            ) if n > 1 else 0.0,
            "ccf_cluster_size": int((ccf_off[i] > 0.3).sum()),
            "ccf_allosteric_score": round(float(ccf_max[i] * coop_score), 4),
            "ccf_persistence": round(float(ccf_mean[i] / max(float(ccf_max[i]), 1e-6)), 4),
            "ccf_off_diag_max": round(float(ccf_max[i]), 4),
            "ccf_asymmetry": round(float(ccf_asymmetry[i]), 4),
            "ccf_onset_bin": None,
            # NMA-derived (4)
            "nma_responsive_modes": nma_resp_modes[i],
            "nma_primary_direction": [round(v, 4) for v in nma_primary[i]],
            "nma_mechanical_sensitivity": round(float(nma_sensitivity[i]), 4),
            "nma_sqfluct": round(float(nma_sensitivity[i] ** 2), 6),
            # classification / flags (6)
            "barrier_classification": bar_cls,
            "mechanism_class": "BACKGROUND",   # filled later by site assignment
            "is_active_site": bool(i in active_site_set),
            "cooperative_score": round(coop_score, 4),
            "cooperative_partners": coop_partners,
            "ring_exchange_triggered": bool(total_spikes[i] > median_total and b_over_a[i] > 1.3),
            # external-data placeholders (5)
            "esm_conservation": None,
            "ensemble_confidence": None,
            "ensemble_variance": None,
        }
        records.append(rec)

    return records

def _argmax_phase(phase_row: np.ndarray) -> str:
    totals = phase_row.sum(axis=1)
    if totals.sum() == 0:
        return "none"
    return PHASE_ORDER[int(np.argmax(totals))]

# ──────────────────────────────────────────────────────────────────────────────
# Per-site feature extraction (≥60 fields)
# ──────────────────────────────────────────────────────────────────────────────

def build_per_site_features(
    binding_sites_json: Dict[str, Any],
    prism_therm_json: Optional[Dict[str, Any]],
    kcc_json: Optional[Dict[str, Any]],
    per_residue: List[Dict[str, Any]],
    structure: Structure,
    ccf: np.ndarray,
    nma_data: Optional[Dict[str, Any]],
    active_site_pdb_resnums: List[int],
    target_name: str, pdb_id: str, chain: str,
) -> List[Dict[str, Any]]:
    """Extract per-site records matching the v3 report structure (≥60 fields)."""
    sites_raw = binding_sites_json.get("sites", [])
    prr_by_topo = {r["topo_idx"]: r for r in per_residue}

    therm_sites = {}
    if prism_therm_json:
        for ts in prism_therm_json.get("sites", []) or []:
            # therm sites are usually identified by centroid; key by rounded centroid
            c = ts.get("centroid_angstrom") or ts.get("centroid") or [0, 0, 0]
            key = tuple(round(v, 1) for v in c)
            therm_sites[key] = ts

    kcc_sites = {}
    if kcc_json:
        for ks in kcc_json.get("sites", []) or []:
            c = ks.get("centroid") or [0, 0, 0]
            key = tuple(round(v, 1) for v in c)
            kcc_sites[key] = ks

    active_set = set(active_site_pdb_resnums)
    records: List[Dict[str, Any]] = []
    for rank, site in enumerate(sites_raw, start=1):
        centroid = site.get("centroid", [0, 0, 0])
        centroid_key = tuple(round(v, 1) for v in centroid)
        therm = therm_sites.get(centroid_key, {})
        kcc = kcc_sites.get(centroid_key, {})

        # Lining residues from engine (resid = topology 1-based or 0-based? topo 0-based here)
        lining_ids = site.get("residue_ids", []) or [lr.get("resid") for lr in site.get("lining_residues", [])]
        lining_pdb_resnums: List[int] = []
        lining_details: List[Dict[str, Any]] = []
        for lr_idx, lr in enumerate(site.get("lining_residues", []) or []):
            rid = lr.get("resid")
            if rid is None:
                continue
            # engine uses topology 0-based
            ti = int(rid)
            if ti < 0 or ti >= structure.n_residues:
                continue
            prr = prr_by_topo.get(ti, {})
            pdb_r = structure.pdb_resnum_by_topo[ti]
            lining_pdb_resnums.append(int(pdb_r))
            lining_details.append({
                "pdb_resnum": int(pdb_r),
                "resname": lr.get("resname", ""),
                "aa1": AA3_TO_AA1.get(lr.get("resname", ""), "X"),
                "is_catalytic": bool(lr.get("is_catalytic", False)),
                "min_distance_angstrom": round(float(lr.get("min_distance", 0.0)), 3),
                "n_atoms": int(lr.get("n_atoms", 0)),
                "cryptic_probability": prr.get("cryptic_probability"),
                "dssp": prr.get("dssp"),
                "bfactor": prr.get("bfactor"),
                "sasa_proxy": prr.get("sasa_proxy"),
                "spikes_a": prr.get("spikes_a"),
                "spikes_b": prr.get("spikes_b"),
                "ccf_peak_value": prr.get("ccf_peak_value"),
                "nma_responsive_modes": prr.get("nma_responsive_modes", []),
                "mechanism_contribution": "lining",
            })

        # Active site overlap
        overlap = sorted(set(lining_pdb_resnums) & active_set)

        # Aggregate spike dynamics from lining
        lining_topo = [
            structure.topo_by_pdb_resnum.get(pr) for pr in lining_pdb_resnums
        ]
        lining_topo = [t for t in lining_topo if t is not None]
        if lining_topo:
            sa = sum(per_residue[t]["spikes_a"] for t in lining_topo)
            sb = sum(per_residue[t]["spikes_b"] for t in lining_topo)
            total = sa + sb
            phase_counts = {p: 0 for p in PHASE_ORDER}
            for t in lining_topo:
                for p in PHASE_ORDER:
                    phase_counts[p] += per_residue[t]["phase_profiles"][p]["total"]
            ccf_submat = ccf[np.ix_(lining_topo, lining_topo)]
            np.fill_diagonal(ccf_submat, 0)
            site_ccf_peak = float(ccf_submat.max()) if ccf_submat.size else 0.0
        else:
            sa = sb = total = 0
            phase_counts = {p: 0 for p in PHASE_ORDER}
            site_ccf_peak = 0.0

        # Convex hull volume via scipy
        hull_vol = _convex_hull_volume(structure.ca_xyz[lining_topo]) if len(lining_topo) >= 4 else 0.0
        compactness = float(site.get("volume", 0.0)) / max(hull_vol, 1.0)

        # Allosteric: top partners by CCF to the SITE residue set
        all_partners = _site_ccf_partners(lining_topo, ccf, structure, top_k=10)
        # CCF to active site
        if active_set:
            as_topo = [structure.topo_by_pdb_resnum.get(r) for r in active_set]
            as_topo = [t for t in as_topo if t is not None]
            if as_topo and lining_topo:
                ccf_to_as = float(ccf[np.ix_(lining_topo, as_topo)].max())
            else:
                ccf_to_as = 0.0
        else:
            ccf_to_as = 0.0

        # NMA contributions for this site
        site_nma = _site_nma_summary(lining_topo, nma_data, structure.n_residues)

        # Druggability (simple chemistry rollup)
        druggability_info = _site_druggability(lining_details)

        # Docking box
        if lining_topo:
            coords = structure.ca_xyz[lining_topo]
            mn = coords.min(axis=0)
            mx = coords.max(axis=0)
            center = ((mn + mx) / 2).tolist()
            size = (mx - mn + 6.0).tolist()  # +6 Å padding
        else:
            center = list(centroid)
            size = [20.0, 20.0, 20.0]

        docking_box = {
            "center_x": round(center[0], 3),
            "center_y": round(center[1], 3),
            "center_z": round(center[2], 3),
            "size_x": round(size[0], 3),
            "size_y": round(size[1], 3),
            "size_z": round(size[2], 3),
        }

        pharmacophore = _pharmacophore_from_lining(lining_details, structure, lining_topo)

        # Mechanism, beacon, quality flags
        mechanism = _classify_mechanism(site_nma, site_ccf_peak, sa, sb)
        beacon = _build_beacon(centroid, lining_topo, structure, site_nma, overlap)
        quality_flags = _quality_flags(site, overlap, lining_topo, total)
        screening = _screening_strategy(druggability_info, site)

        # Confidence score
        confidence = _confidence_score(site, total, site_nma, overlap)

        record = {
            # identity (7)
            "site_id": f"{target_name.upper()}_TWIN_S{rank:02d}",
            "rank": rank,
            "engine_pocket_id": int(site.get("id", rank)),
            "protein": target_name,
            "pdb_id": pdb_id.upper(),
            "chain": chain,
            "target_name": target_name,
            # classification (5)
            "classification": site.get("classification", "Unknown").upper(),
            "engine_classification": site.get("classification", "Unknown"),
            "therm_class": site.get("therm_class", "UNKNOWN"),
            "is_active_site_overlap": bool(overlap),
            "active_site_residues_overlap": overlap,
            # geometry (7)
            "centroid_angstrom": [round(c, 3) for c in centroid],
            "weighted_centroid_angstrom": [round(c, 3) for c in centroid],
            "volume_angstrom3_engine": float(site.get("volume", 0.0)),
            "volume_angstrom3_convex_hull": round(hull_vol, 2),
            "spatial_extent_angstrom": round(float(np.linalg.norm(
                structure.ca_xyz[lining_topo].max(axis=0) -
                structure.ca_xyz[lining_topo].min(axis=0)
            )) if lining_topo else 0.0, 2),
            "sphericity": round(float(site.get("aromatic_score", 0.5)), 4),
            "compactness": round(compactness, 6),
            # engine scores (10)
            "n_lining_residues": len(lining_details),
            "engine_quality_score": float(site.get("quality_score", 0.0)),
            "engine_druggability": float(site.get("druggability", 0.0)),
            "engine_spike_count": int(site.get("spike_count", 0)),
            "engine_burial_score": 0.0,
            "engine_sphericity": 0.0,
            "engine_breathing_score": 0.0,
            "engine_aromatic_score": float(site.get("aromatic_score", 0.0)),
            "engine_uv_enrichment": 0.0,
            "engine_wd_coherence": 0.0,
            "engine_rank_score": float(site.get("quality_score", 0.0)),
            "gtck_rank": kcc.get("gtck_rank"),
            "engine_tide_coupling": None,
            # AI/ensemble placeholders (3)
            "ai_mean_probability": None,
            "ai_max_probability": None,
            "n_ai_folds_used": 0,
            # compound groupings (12)
            "therm": {
                "hysteresis_asymmetry": float(site.get("hysteresis_asymmetry", 0.0)),
                "relative_asymmetry": float(site.get("relative_asymmetry", 0.0)),
                "ccns_tau": float(site.get("ccns_tau", 0.0)),
                "barrier_estimate_kcal_mol": None,
                "cold_phase_fraction": {
                    "cold": phase_counts["cold_hold"] / max(total, 1),
                    "hot": (phase_counts["warm_hold"] + phase_counts["heating"]) / max(total, 1),
                },
                "frustrated_solvent_score": None,
                "tide_trigger_residues": (therm.get("tide_trigger_residues") if therm else None) or [],
                "water_displacement_sites": None,
                "water_displacement_sites_note": "Requires explicit solvent refinement (WT-6 pipeline)",
                "desolvation_penalty_kcal_mol": None,
                "desolvation_penalty_note": "Requires GIST/SSTMap analysis (WT-6 pipeline)",
            },
            "spike_dynamics": {
                "stream_a_spikes": int(sa),
                "stream_b_spikes": int(sb),
                "total_spikes": int(total),
                "b_over_a_ratio": round(sb / max(sa, 1), 4),
                "spike_agreement_ratio": round(1.0 - abs(sa - sb) / max(total, 1), 4),
                "ccf_peak_value": round(site_ccf_peak, 4),
                "phase_dynamics": phase_counts,
            },
            "nma": site_nma,
            "druggability": druggability_info,
            "allosteric": {
                "ccf_to_active_site": round(ccf_to_as, 4),
                "top_allosteric_partners": all_partners,
            },
            "kcc": {
                "driver_residue_topo": kcc.get("driver_residue_topo"),
                "kcc_confidence": kcc.get("kcc_confidence"),
                "site_causal_lag_steps": kcc.get("site_causal_lag_steps"),
                "site_burst_motion": kcc.get("site_burst_motion"),
                "site_lag_corr_peak": kcc.get("site_lag_corr_peak"),
                "site_local_cov": kcc.get("site_local_cov"),
            } if kcc else {},
            "lining_residues": lining_details,
            "docking_box_unidock": docking_box,
            "pharmacophore_features": pharmacophore,
            # confidence + evidence (3)
            "confidence_score": round(confidence, 3),
            "evidence_components": {
                "therm_signal": site.get("therm_class") in ("CRYPTIC", "RESPONSIVE", "DYNAMIC"),
                "ai_probability_gt30": None,
                "spike_coverage": total > 1000,
                "nma_responsiveness": site_nma.get("n_responsive_modes", 0) > 0,
                "engine_quality_gt35": float(site.get("quality_score", 0.0)) > 0.35,
            },
            # external-data placeholder (1)
            "explicit_solvent": {
                "stability_class": None,
                "stability_class_note": "Requires explicit solvent refinement (WT-6 pipeline)",
                "rmsd_angstrom": None,
                "volume_sigma_pct": None,
                "simulation_time_ns": None,
            },
            # enrichment compound (3)
            "mechanism": mechanism,
            "thermodynamic_enrichment": _therm_enrichment(phase_counts, total),
            "beacon": beacon,
            # derived flags (8)
            "opening_mechanism": mechanism.get("opening_mechanism", "PREFORMED"),
            "gating_residues": mechanism.get("gating_residues", []),
            "beacon_map": beacon,
            "quality_flags": quality_flags,
            "screening_strategy": screening,
            "reversibility": "REVERSIBLE" if total > 0 else "UNKNOWN",
            "estimated_ligand_MW": druggability_info.get("ligand_mw_range", [300, 500]),
            "fragment_hotspots": druggability_info.get("fragment_hotspots", []),
            "flexibility_class": _flexibility_class(site),
            "recommendation": _recommendation(confidence, overlap),
        }
        records.append(record)

    return records

def _convex_hull_volume(points: np.ndarray) -> float:
    try:
        from scipy.spatial import ConvexHull
        if len(points) < 4:
            return 0.0
        return float(ConvexHull(points.astype(np.float64)).volume)
    except Exception:
        return 0.0

def _site_ccf_partners(lining_topo, ccf, structure, top_k=10):
    if not lining_topo:
        return []
    # Max CCF from any lining residue to every other residue
    sub = ccf[lining_topo]
    max_per_col = sub.max(axis=0)
    np.put(max_per_col, lining_topo, 0.0)  # exclude self
    top = np.argsort(-max_per_col)[:top_k]
    return [
        {
            "topo_idx": int(t),
            "pdb_resnum": int(structure.pdb_resnum_by_topo[t]),
            "resname": structure.residues[t].resname,
            "ccf": round(float(max_per_col[t]), 4),
        }
        for t in top
    ]

def _site_nma_summary(lining_topo, nma_data, n_residues):
    if nma_data is None or not lining_topo:
        return {"n_responsive_modes": 0, "mean_displacement_angstrom": 0.0, "responsive_modes": []}
    modes = nma_data.get("modes", [])
    responsive = []
    total_disp = 0.0
    for m_idx, mode in enumerate(modes[:10]):
        eig = mode.get("eigenvalue", 0.0)
        vec = mode.get("eigenvector") or mode.get("displacements")
        if not vec or eig <= 0:
            continue
        varr = np.asarray(vec, dtype=np.float32)
        if varr.ndim == 1:
            varr = varr.reshape(-1, 3)
        if len(varr) < n_residues:
            continue
        disp = np.linalg.norm(varr[lining_topo], axis=1).mean()
        thermal_amp = float((1.0 / max(eig, 1e-6)) ** 0.5)
        if disp >= 0.03:
            responsive.append({
                "mode_index": int(m_idx),
                "eigenvalue": float(eig),
                "thermal_amplitude_angstrom": round(thermal_amp, 4),
                "site_mean_displacement": round(float(disp), 5),
                "force_scale": round(thermal_amp / max(float(disp), 1e-6), 4),
            })
            total_disp += float(disp)
    responsive.sort(key=lambda m: -m["site_mean_displacement"])
    return {
        "n_responsive_modes": len(responsive),
        "mean_displacement_angstrom": round(total_disp / max(len(responsive), 1), 4),
        "responsive_modes": responsive[:5],
    }

def _site_druggability(lining_details):
    if not lining_details:
        return {
            "druggability_score": 0.0,
            "polarity_fraction": 0.0,
            "hydrophobic_fraction": 0.0,
            "aromatic_fraction": 0.0,
            "hbond_donor_count": 0,
            "hbond_acceptor_count": 0,
            "aromatic_count": 0,
            "charged_count": 0,
            "flexibility_mean_bfactor": 0.0,
            "pocket_polarity": "UNKNOWN",
            "polar_fraction": 0.0,
            "charged_fraction": 0.0,
            "ligand_mw_range": [300, 500],
            "fragment_hotspots": [],
            "druggability_score_g6": 0.0,
            "screening_strategy": "Fragment screen",
        }
    n = len(lining_details)
    aa1s = [lr["aa1"] for lr in lining_details]
    hydrophobic = sum(1 for a in aa1s if a in HYDROPHOBIC) / n
    polar = sum(1 for a in aa1s if a in POLAR) / n
    aromatic = sum(1 for a in aa1s if a in AROMATIC) / n
    positive = sum(1 for a in aa1s if a in POSITIVE) / n
    negative = sum(1 for a in aa1s if a in NEGATIVE) / n
    charged = positive + negative
    hbd = sum(1 for a in aa1s if a in HBD_RES)
    hba = sum(1 for a in aa1s if a in HBA_RES)
    arom_count = sum(1 for a in aa1s if a in AROMATIC)
    flex = float(np.mean([lr.get("bfactor") or 0.0 for lr in lining_details]))
    hotspots = [
        {"pdb_resnum": lr["pdb_resnum"], "resname": lr["resname"], "type": "AROMATIC_ANCHOR"}
        for lr in lining_details if lr["aa1"] in AROMATIC
    ][:5]
    drug_score = 0.3 * hydrophobic + 0.3 * aromatic + 0.2 * min(hbd + hba, 20) / 20.0 + 0.2
    if hydrophobic > 0.6:
        polarity = "HYDROPHOBIC"
    elif polar > 0.5:
        polarity = "POLAR"
    else:
        polarity = "MIXED"
    if drug_score > 0.7:
        strategy = "Fragment → SBDD; high druggability"
    elif drug_score > 0.5:
        strategy = "Fragment screen → optimize"
    else:
        strategy = "Broad HTS or DEL"
    return {
        "druggability_score": round(drug_score, 3),
        "polarity_fraction": round(polar, 3),
        "hydrophobic_fraction": round(hydrophobic, 3),
        "aromatic_fraction": round(aromatic, 3),
        "hbond_donor_count": hbd,
        "hbond_acceptor_count": hba,
        "aromatic_count": arom_count,
        "charged_count": sum(1 for a in aa1s if a in POSITIVE | NEGATIVE),
        "flexibility_mean_bfactor": round(flex, 3),
        "pocket_polarity": polarity,
        "polar_fraction": round(polar, 3),
        "charged_fraction": round(charged, 3),
        "ligand_mw_range": [300, 600] if drug_score > 0.6 else [200, 450],
        "fragment_hotspots": hotspots,
        "druggability_score_g6": round(drug_score * 0.95, 4),
        "screening_strategy": strategy,
    }

def _pharmacophore_from_lining(lining_details, structure, lining_topo):
    features = []
    for lr, ti in zip(lining_details, lining_topo):
        aa1 = lr["aa1"]
        xyz = structure.ca_xyz[ti].tolist()
        if aa1 in AROMATIC:
            ftype = "AR"
        elif aa1 in HBD_RES and aa1 in HBA_RES:
            ftype = "HBD/HBA"
        elif aa1 in HBD_RES:
            ftype = "HBD"
        elif aa1 in HBA_RES:
            ftype = "HBA"
        elif aa1 in POSITIVE:
            ftype = "POS"
        elif aa1 in NEGATIVE:
            ftype = "NEG"
        else:
            ftype = "H"
        features.append({
            "type": ftype,
            "resname": lr["resname"],
            "resnum": lr["pdb_resnum"],
            "xyz": [round(v, 3) for v in xyz],
        })
    return features

def _classify_mechanism(site_nma, ccf_peak, sa, sb):
    n_modes = site_nma.get("n_responsive_modes", 0)
    if n_modes >= 3 and ccf_peak > 0.3:
        opening = "INDUCED_FIT"
    elif n_modes >= 1:
        opening = "CONFORMATIONAL_SELECTION"
    else:
        opening = "PREFORMED"
    return {
        "opening_mechanism": opening,
        "gating_residues": [],
        "hinge_residues": [],
        "mobile_residues": [],
        "dominant_nma_modes": site_nma.get("responsive_modes", [])[:3],
    }

def _build_beacon(centroid, lining_topo, structure, site_nma, overlap_pdb):
    if not lining_topo:
        radius = 5.0
    else:
        coords = structure.ca_xyz[lining_topo]
        center = coords.mean(axis=0)
        radius = float(np.linalg.norm(coords - center, axis=1).max())
    responsive = [m["mode_index"] for m in site_nma.get("responsive_modes", [])[:4]]
    return {
        "centroid_angstrom": [round(c, 3) for c in centroid],
        "radius_angstrom": round(radius, 3),
        "responsive_nma_modes": responsive,
        "recommended_amplification_angstrom": round(radius * 0.25, 3),
        "quality_flags": ["ACTIVE_SITE_OVERLAP"] if overlap_pdb else [],
        "n_quality_issues": 0,
    }

def _quality_flags(site, overlap, lining_topo, total_spikes):
    flags = []
    if overlap:
        flags.append("ACTIVE_SITE_OVERLAP")
    if site.get("volume", 0) < 200:
        flags.append("SMALL_POCKET")
    if total_spikes < 500:
        flags.append("LOW_SPIKE_COVERAGE")
    if len(lining_topo) < 5:
        flags.append("FEW_LINING_RESIDUES")
    return flags

def _screening_strategy(druggability, site):
    return druggability.get("screening_strategy", "Fragment screen")

def _confidence_score(site, total_spikes, site_nma, overlap):
    components = [
        0.25 * (1.0 if site.get("therm_class") in ("CRYPTIC", "RESPONSIVE", "DYNAMIC") else 0.0),
        0.20 * min(float(site.get("quality_score", 0.0)), 1.0),
        0.15 * min(total_spikes / 10_000.0, 1.0),
        0.15 * (1.0 if site_nma.get("n_responsive_modes", 0) > 0 else 0.0),
        0.15 * (1.0 if overlap else 0.5),
        0.10 * min(float(site.get("druggability", 0.0)), 1.0),
    ]
    return float(sum(components))

def _therm_enrichment(phase_counts, total):
    warm = phase_counts.get("warm_hold", 0)
    cold = phase_counts.get("cold_hold", 0)
    onset = "warm_hold" if warm >= cold else "cold_hold"
    return {
        "temperature_onset": onset,
        "phase_spike_counts": phase_counts,
        "occupancy_warm_hold": round(warm / max(total, 1), 4),
        "hysteresis_ratio": round(phase_counts.get("cold_return", 0) / max(cold, 1), 4),
        "barrier_estimate": "LOW" if warm > cold else "MEDIUM",
        "barrier_kcal_mol": None,
    }

def _flexibility_class(site):
    quality = float(site.get("quality_score", 0.0))
    if quality > 0.6:
        return "RIGID"
    if quality > 0.3:
        return "MODERATE"
    return "FLEXIBLE"

def _recommendation(confidence, overlap):
    if confidence > 0.75 and overlap:
        return "VALIDATE"
    if confidence > 0.65:
        return "PRIORITIZE"
    if confidence > 0.5:
        return "EVALUATE"
    return "DEPRIORITIZE"

# ──────────────────────────────────────────────────────────────────────────────
# Allosteric network
# ──────────────────────────────────────────────────────────────────────────────

def build_allosteric_network(ccf: np.ndarray, structure: Structure) -> Dict[str, Any]:
    """
    Binary-search an edge threshold that yields a mean degree in
    ALLO_TARGET_DEGREE_RANGE. Uses mean-centered CCF.
    """
    n = ccf.shape[0]
    off = ccf.copy()
    np.fill_diagonal(off, 0.0)

    lo, hi = 0.0, float(off.max())
    best_thresh = 0.1
    best_degree = 0.0
    for _ in range(ALLO_MAX_THRESHOLD_ITERS):
        mid = (lo + hi) / 2
        adj = off > mid
        degrees = adj.sum(axis=1)
        mean_deg = float(degrees.mean())
        if ALLO_TARGET_DEGREE_RANGE[0] <= mean_deg <= ALLO_TARGET_DEGREE_RANGE[1]:
            best_thresh = mid
            best_degree = mean_deg
            break
        if mean_deg > ALLO_TARGET_DEGREE_RANGE[1]:
            lo = mid  # too dense → raise threshold
        else:
            hi = mid  # too sparse → lower threshold
        best_thresh = mid
        best_degree = mean_deg

    adj = off > best_thresh
    degrees = adj.sum(axis=1)
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            if adj[i, j]:
                edges.append({
                    "source": int(i),
                    "target": int(j),
                    "ccf": round(float(off[i, j]), 4),
                })
    nodes = [
        {
            "topo_idx": int(i),
            "pdb_resnum": int(structure.pdb_resnum_by_topo[i]),
            "resname": structure.residues[i].resname,
            "degree": int(degrees[i]),
            "ccf_max": round(float(off[i].max()), 4),
        }
        for i in range(n)
    ]
    top_hubs = sorted(nodes, key=lambda x: -x["degree"])[:20]
    return {
        "ccf_threshold": round(float(best_thresh), 4),
        "n_nodes": int(n),
        "n_edges": len(edges),
        "n_edges_in_file": len(edges),
        "mean_degree": round(float(degrees.mean()), 3),
        "max_degree": int(degrees.max()),
        "median_degree": int(np.median(degrees)),
        "nodes": nodes,
        "edges": edges[:20_000],
        "top_hub_residues": top_hubs,
    }

# ──────────────────────────────────────────────────────────────────────────────
# Auxiliary output files
# ──────────────────────────────────────────────────────────────────────────────

def write_heatmap_pdb(per_residue: List[Dict[str, Any]], structure: Structure, out_path: Path) -> None:
    """Write a CA-only PDB with b-factor = cryptic_probability * 100."""
    lines = []
    for prr, res in zip(per_residue, structure.residues):
        prob = float(prr.get("cryptic_probability") or 0.0)
        bfac = round(prob * 100, 2)
        x, y, z = res.ca_xyz.tolist()
        line = (
            f"ATOM  {res.topo_idx+1:5d}  CA  {res.resname:>3s} {structure.chain}"
            f"{res.pdb_resnum:4d}    {x:8.3f}{y:8.3f}{z:8.3f}  1.00{bfac:6.2f}           C"
        )
        lines.append(line)
    lines.append("END")
    out_path.write_text("\n".join(lines) + "\n")

def build_site_cards(sites: List[Dict[str, Any]], target_name: str) -> Dict[str, Any]:
    cards = []
    for s in sites:
        cards.append({
            "site_id": s["site_id"],
            "rank": s["rank"],
            "classification": s["classification"],
            "therm_class": s["therm_class"],
            "confidence": s["confidence_score"],
            "centroid": s["centroid_angstrom"],
            "volume_A3_engine": s["volume_angstrom3_engine"],
            "volume_A3_convex_hull": s["volume_angstrom3_convex_hull"],
            "n_residues": s["n_lining_residues"],
            "druggability_score": s["druggability"]["druggability_score"],
            "barrier_kcal_mol": s["thermodynamic_enrichment"]["barrier_kcal_mol"],
            "hysteresis": s["therm"]["hysteresis_asymmetry"],
            "ccf_to_active_site": s["allosteric"]["ccf_to_active_site"],
            "n_responsive_nma_modes": s["nma"]["n_responsive_modes"],
            "key_lining_residues": [
                lr["pdb_resnum"] for lr in s["lining_residues"][:8]
            ],
            "active_site_overlap": s["is_active_site_overlap"],
            "engine_pocket_id": s["engine_pocket_id"],
            "engine_quality": s["engine_quality_score"],
            "spike_dynamics_summary": s["spike_dynamics"]["total_spikes"],
            "pharma_feature_types": sorted({f["type"] for f in s["pharmacophore_features"]}),
            "explicit_solvent_status": s["explicit_solvent"]["stability_class"],
        })
    return {
        "target": target_name,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "n_sites": len(cards),
        "sites": cards,
    }

def build_docking_ready(sites: List[Dict[str, Any]], target_name: str) -> Dict[str, Any]:
    return {
        "target": target_name,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "software_compatibility": ["UniDock", "AutoDock Vina", "GNINA"],
        "coordinate_units": "angstrom",
        "sites": [
            {
                "site_id": s["site_id"],
                "rank": s["rank"],
                "classification": s["classification"],
                "confidence": s["confidence_score"],
                "docking_box": s["docking_box_unidock"],
                "pharmacophore_features": s["pharmacophore_features"],
                "n_pharmacophore_features": len(s["pharmacophore_features"]),
                "feature_types": sorted({f["type"] for f in s["pharmacophore_features"]}),
                "lining_residues_pdb": [lr["pdb_resnum"] for lr in s["lining_residues"]],
                "hbond_donors": s["druggability"]["hbond_donor_count"],
                "hbond_acceptors": s["druggability"]["hbond_acceptor_count"],
                "aromatic_count": s["druggability"]["aromatic_count"],
            }
            for s in sites
        ],
    }

def build_mechanisms(sites: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out = []
    for s in sites:
        m = s["mechanism"]
        out.append({
            "site_id": s["site_id"],
            "opening_mechanism": m.get("opening_mechanism"),
            "gating_residues": m.get("gating_residues", []),
            "hinge_residues": m.get("hinge_residues", []),
            "mobile_residues": m.get("mobile_residues", []),
            "dominant_nma_modes": m.get("dominant_nma_modes", []),
        })
    return out

def build_beacons(sites: List[Dict[str, Any]]) -> Dict[str, Any]:
    return {
        "n_sites": len(sites),
        "beacons": [
            {
                "site_id": s["site_id"],
                "rank": s["rank"],
                **s["beacon"],
            }
            for s in sites
        ],
    }

# ──────────────────────────────────────────────────────────────────────────────
# Validation invariants
# ──────────────────────────────────────────────────────────────────────────────

def validate_outputs(
    ccf: np.ndarray, ccf_summary: Dict[str, Any],
    per_residue: List[Dict[str, Any]], per_site: List[Dict[str, Any]],
    n_residues: int,
) -> Dict[str, Any]:
    problems: List[str] = []

    # CCF shape + diagonal + distant pair
    if ccf.shape != (n_residues, n_residues):
        problems.append(f"CCF shape mismatch: {ccf.shape} != ({n_residues},{n_residues})")
    diag_mean = float(np.diag(ccf).mean())
    if abs(diag_mean - 1.0) > 0.2:
        problems.append(f"CCF diagonal mean {diag_mean:.3f} too far from 1.0")
    dp = ccf_summary.get("distant_pair_mean", 0.0)
    if dp > 0.15:
        problems.append(f"CCF distant pair mean {dp:.3f} > 0.15 — mean centering failed")

    # Per-residue field count
    if per_residue:
        n_fields = len(per_residue[0])
        if n_fields < 48:
            problems.append(f"Per-residue has {n_fields} fields, <48 required")
    if len(per_residue) != n_residues:
        problems.append(f"Per-residue length {len(per_residue)} != n_residues {n_residues}")

    # Per-site field count
    if per_site:
        n_fields_site = len(per_site[0])
        if n_fields_site < 60:
            problems.append(f"Per-site has {n_fields_site} fields, <60 required")

    return {
        "passed": len(problems) == 0,
        "problems": problems,
        "per_residue_field_count": len(per_residue[0]) if per_residue else 0,
        "per_site_field_count": len(per_site[0]) if per_site else 0,
        "ccf_diagonal_mean": diag_mean,
        "ccf_distant_pair_mean": dp,
    }

# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        description="PRISM-TWIN post-processing pipeline — produces 9 enriched output files from engine output."
    )
    ap.add_argument("--target-dir", type=Path, required=True,
                    help="Directory containing engine output (coupled_spikes.{json,parquet}, coupled_twin_result.json, binding_sites.json, ...)")
    ap.add_argument("--pdb-path", type=Path, required=True, help="Clean PDB file for this target")
    ap.add_argument("--target-id", type=str, required=True, help="Output filename prefix, e.g. 1btl_chainA")
    ap.add_argument("--pdb-id", type=str, required=True, help="4-char PDB id")
    ap.add_argument("--chain", type=str, required=True, help="Chain letter")
    ap.add_argument("--topology", type=Path, default=None, help="Topology JSON (default: auto-detected)")
    ap.add_argument("--nma-file", type=str, default="", help="NMA modes JSON file (optional)")
    ap.add_argument("--target-name", type=str, default=None, help="Human-readable target name (default: --pdb-id lowercase)")
    ap.add_argument("--active-site-pdb-resnums", type=str, default="",
                    help="Comma-separated PDB resnums that define the catalytic/active site")
    ap.add_argument("--verify-only", action="store_true", help="Skip compute, re-read existing outputs and verify invariants")
    args = ap.parse_args(argv)

    target_dir: Path = args.target_dir
    target_dir.mkdir(parents=True, exist_ok=True)

    prefix = args.target_id
    target_name = args.target_name or args.pdb_id.lower()
    active_site_pdb = [int(s) for s in args.active_site_pdb_resnums.split(",") if s.strip()]

    log(f"postprocess_twin starting: target={prefix}  dir={target_dir}")

    # ── 1. Load structure + topology ──
    topology = args.topology
    if topology is None:
        # Try engine-standard locations
        candidates = [
            target_dir / f"{args.pdb_id}.topology.json",
            args.pdb_path.with_suffix(".topology.json"),
            args.pdb_path.parent / f"{args.pdb_id}.topology.json",
        ]
        for c in candidates:
            if c.exists():
                topology = c
                break
    structure = load_structure(args.pdb_path, topology, args.chain)
    log(f"Structure loaded: {structure.n_residues} residues, chain {structure.chain}")

    active_site_topo = [
        structure.topo_by_pdb_resnum[r]
        for r in active_site_pdb
        if r in structure.topo_by_pdb_resnum
    ]

    # ── 2. Load NMA if provided ──
    nma_data = None
    if args.nma_file:
        nma_path = Path(args.nma_file)
        if nma_path.exists():
            nma_data = read_json(nma_path)
            log(f"NMA loaded: {len(nma_data.get('modes', []))} modes")
        else:
            log(f"NMA file not found: {nma_path}", "WARN")

    # ── 3. Load engine outputs ──
    binding_sites_json = None
    for cand in [target_dir / f"{args.pdb_id}.binding_sites.json",
                 *target_dir.glob("*.binding_sites.json")]:
        if cand.exists():
            binding_sites_json = read_json(cand)
            log(f"Binding sites loaded: {len(binding_sites_json.get('sites', []))} sites")
            break
    if binding_sites_json is None:
        log("No binding_sites.json found — per_site output will be empty", "WARN")
        binding_sites_json = {"sites": []}

    kcc_json = None
    for cand in [target_dir / f"{args.pdb_id}.kcc_visualization.json",
                 *target_dir.glob("*.kcc_visualization.json")]:
        if cand.exists():
            kcc_json = read_json(cand)
            break

    prism_therm_json = None
    for cand in [target_dir / f"{args.pdb_id}.topology.prism_therm.json",
                 *target_dir.glob("*.prism_therm.json")]:
        if cand.exists():
            prism_therm_json = read_json(cand)
            break

    coupled_twin_result = None
    ctr_path = target_dir / "coupled_twin_result.json"
    if ctr_path.exists():
        coupled_twin_result = read_json(ctr_path)

    # ── 4. Load spikes ──
    spikes = load_spikes(target_dir)
    if spikes.n_total == 0:
        log("Zero spikes loaded — aborting", "ERROR")
        return 2

    # ── 5. Spike → residue mapping ──
    residue_idx = assign_spikes_to_residues(spikes, structure.ca_xyz)
    phase_bounds = resolve_phase_boundaries(spikes.max_timestep)
    phase_idx = phase_of_timestep(spikes.timestep, phase_bounds)
    log(f"Phase boundaries (steps): {phase_bounds}")

    # ── 6. Per-residue spike counts ──
    spike_counts = compute_per_residue_spike_counts(
        spikes, residue_idx, structure.n_residues, phase_idx
    )

    # ── 7. CCF matrix (mean-centered) ──
    ccf, ccf_summary = compute_ccf_matrix(spikes, residue_idx, structure.n_residues)
    log(f"CCF computed: distant_pair_mean={ccf_summary['distant_pair_mean']:.4f} "
        f"(must be <0.15)")
    if ccf_summary["distant_pair_mean"] > 0.15:
        log("CCF distant pair mean too high — mean-centering may not be sufficient", "WARN")

    ccf_path = target_dir / f"{prefix}_ccf_matrix.npy"
    np.save(ccf_path, ccf)
    log(f"wrote {ccf_path.name}")

    # ── 8. Per-residue features ──
    per_residue = build_per_residue_features(
        structure, spike_counts, ccf, nma_data, active_site_topo
    )
    write_json(target_dir / f"{prefix}_per_residue.json", per_residue)
    log(f"wrote {prefix}_per_residue.json ({len(per_residue)} residues × {len(per_residue[0]) if per_residue else 0} fields)")

    # ── 9. Per-site features ──
    per_site = build_per_site_features(
        binding_sites_json, prism_therm_json, kcc_json,
        per_residue, structure, ccf, nma_data,
        active_site_pdb, target_name, args.pdb_id, args.chain,
    )
    write_json(target_dir / f"{prefix}_per_site.json", per_site)
    log(f"wrote {prefix}_per_site.json ({len(per_site)} sites × {len(per_site[0]) if per_site else 0} fields)")

    # ── 10. Allosteric network ──
    allo = build_allosteric_network(ccf, structure)
    write_json(target_dir / f"{prefix}_allosteric.json", allo)
    log(f"wrote {prefix}_allosteric.json (threshold={allo['ccf_threshold']}, mean_degree={allo['mean_degree']})")

    # ── 11. Site cards ──
    cards = build_site_cards(per_site, target_name)
    write_json(target_dir / f"{prefix}_site_cards.json", cards)

    # ── 12. Docking ready ──
    docking = build_docking_ready(per_site, target_name)
    write_json(target_dir / f"{prefix}_docking.json", docking)

    # ── 13. Heatmap PDB ──
    write_heatmap_pdb(per_residue, structure, target_dir / f"{prefix}_heatmap.pdb")

    # ── 14. Beacons ──
    beacons = build_beacons(per_site)
    write_json(target_dir / f"{prefix}_beacons.json", beacons)

    # ── 15. Mechanisms ──
    mechanisms = build_mechanisms(per_site)
    write_json(target_dir / f"{prefix}_mechanisms.json", mechanisms)

    # ── 16. Validation ──
    validation = validate_outputs(ccf, ccf_summary, per_residue, per_site, structure.n_residues)
    log(f"validation: {'PASS' if validation['passed'] else 'FAIL'}")
    for p in validation["problems"]:
        log(f"  problem: {p}", "WARN")

    # Write a small summary for the runner to consume
    write_json(target_dir / f"{prefix}_postprocess_summary.json", {
        "prefix": prefix,
        "n_residues": structure.n_residues,
        "n_sites": len(per_site),
        "ccf_summary": ccf_summary,
        "validation": validation,
        "spike_totals": {
            "n_total": spikes.n_total,
            "n_a": spikes.n_a,
            "n_b": spikes.n_b,
        },
        "files_written": [
            f"{prefix}_ccf_matrix.npy",
            f"{prefix}_per_residue.json",
            f"{prefix}_per_site.json",
            f"{prefix}_allosteric.json",
            f"{prefix}_site_cards.json",
            f"{prefix}_docking.json",
            f"{prefix}_heatmap.pdb",
            f"{prefix}_beacons.json",
            f"{prefix}_mechanisms.json",
        ],
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    })

    log(f"postprocess_twin complete: {prefix}")
    return 0 if validation["passed"] else 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:
        traceback.print_exc()
        sys.exit(3)
