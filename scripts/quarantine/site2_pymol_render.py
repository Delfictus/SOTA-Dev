#!/usr/bin/env python3
"""
Site-2 PyMOL visualization builder for the Phase-2.1 4LPK run.

What this emits
---------------
Under  <output-dir>/site2_pymol/ :

  density_all.ccp4              volumetric spike-count density (all phases)
  density_intensity.ccp4        intensity-weighted density
  density_cold_phase.ccp4       density restricted to ccns_phase == cold_hold
  density_warm_phase.ccp4       density restricted to ccns_phase == warm_hold
  density_ramp_phase.ccp4       density restricted to ccns_phase == ramp
  top_spikes.pdb                500 stratified representative spikes (HETATM),
                                encoding intensity (q), vibrational_energy (b),
                                phase (chainID), and type (resname)
  site2_session.pml             The PyMOL script — open this in PyMOL.
  site2_metadata.txt            Human-readable summary sheet.

All paths inside the emitted .pml are absolute so it works regardless of the
directory you launch PyMOL from.

Run stages
----------
  # Stage 1: preprocess (streams the 3.8 GB spike file with ijson — ~2-4 min)
  python3 scripts/quarantine/site2_pymol_render.py

  # Stage 2: open in PyMOL (GUI)
  pymol /home/diddy/Desktop/Prism4D-bio/output/4lpk_phase2.1_audit_verify/site2_pymol/site2_session.pml

Why a two-stage design
----------------------
* The spike stream is 3.8 GB.  Doing ijson parsing inside a running PyMOL
  session would block the GUI for minutes.  Pre-baking to MRC/CCP4 density
  maps + a small PDB of representative spikes keeps the PyMOL session
  responsive and re-loadable.
* The .pml is plain text and re-runnable; you can edit it without re-parsing
  the spike file.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

try:
    import ijson
except ImportError:
    sys.stderr.write("FATAL: pip install ijson\n")
    sys.exit(2)

try:
    import mrcfile
except ImportError:
    sys.stderr.write("FATAL: pip install mrcfile\n")
    sys.exit(2)


# ============================================================================
# Config defaults (CLI overrides available)
# ============================================================================

DEFAULT_ROOT = Path("/home/diddy/Desktop/Prism4D-bio/output/4lpk_phase2.1_audit_verify")
DEFAULT_PREFIX = "4lpk_clean"
DEFAULT_SITE_ID = 2
DEFAULT_RAW_PDB = Path("/home/diddy/Desktop/Prism4D-bio/4lpk_raw.pdb")
DEFAULT_VOXEL_SIZE = 1.0  # Å per voxel
DEFAULT_PADDING = 4.0     # Å pad around spike bounding box
DEFAULT_TOP_K = 500       # total representative spikes
DEFAULT_N_STRATA = 5      # intensity quintiles for stratified top-K

# Phase → chain id in the generated spike PDB
PHASE_CHAIN = {"cold_hold": "C", "ramp": "R", "warm_hold": "W"}


# ============================================================================
# Metadata extraction
# ============================================================================

def _load_json(p: Path) -> Any:
    with p.open("r") as fh:
        return json.load(fh)


@dataclass
class SiteMetadata:
    site_id: int
    centroid: Tuple[float, float, float]
    spike_count: int
    n_lining: int
    quality_score: float
    druggability_overall: float
    classification: str
    therm_class: Optional[str]
    ccns_class: Optional[str]
    is_cryptic: Optional[bool]
    ccns_tau: Optional[float]
    lining_residues: List[Dict[str, Any]]      # from binding_sites.json
    therm_top_residues: List[Dict[str, Any]]   # from prism_therm.pockets[id]
    kcc_topk_residues: List[Dict[str, Any]]    # from kcc_validation.sites[id]
    volume: Optional[float]
    ligand_centroid: Optional[Tuple[float, float, float]]


def collect_metadata(root: Path, prefix: str, site_id: int) -> SiteMetadata:
    bs = _load_json(root / f"{prefix}.binding_sites.json")
    pt = _load_json(root / f"{prefix}.topology.prism_therm.json")
    kv = _load_json(root / f"{prefix}.kcc_validation.json")

    site = next((s for s in bs.get("sites", []) if s.get("id") == site_id), None)
    if site is None:
        raise SystemExit(f"site id={site_id} not found in binding_sites.json")

    pocket = next((p for p in pt.get("pockets", []) if p.get("pocket_id") == site_id), None)
    kv_site = next((s for s in kv.get("sites", []) if s.get("site_id") == site_id), None)

    gt = root / f"{prefix}_ground_truth.json"
    lig = None
    if gt.is_file():
        gtj = _load_json(gt)
        lc = gtj.get("ligand_centroid")
        if isinstance(lc, list) and len(lc) == 3:
            lig = tuple(float(x) for x in lc)

    drug = site.get("druggability")
    drug_overall = drug if isinstance(drug, (int, float)) else (drug or {}).get("overall", 0.0)

    return SiteMetadata(
        site_id=site_id,
        centroid=tuple(float(x) for x in site["centroid"]),
        spike_count=int(site.get("spike_count", 0)),
        n_lining=len(site.get("lining_residues", [])),
        quality_score=float(site.get("quality_score", 0.0)),
        druggability_overall=float(drug_overall or 0.0),
        classification=str(site.get("classification", "Unknown")),
        therm_class=(pocket or {}).get("therm_class"),
        ccns_class=(pocket or {}).get("ccns_class"),
        is_cryptic=(pocket or {}).get("is_cryptic"),
        ccns_tau=(pocket or {}).get("ccns_tau"),
        lining_residues=site.get("lining_residues", []),
        therm_top_residues=(pocket or {}).get("top_residues", []),
        kcc_topk_residues=(kv_site or {}).get("topk_residues", []),
        volume=site.get("volume"),
        ligand_centroid=lig,
    )


# ============================================================================
# Spike-stream → density grids + top-K stratified sample
# ============================================================================

@dataclass
class DensityGrids:
    origin: np.ndarray   # (3,) grid origin in Å
    voxel: float         # voxel size Å
    shape: Tuple[int, int, int]
    all_count: np.ndarray          # (nx, ny, nz) float32
    intensity_sum: np.ndarray      # (nx, ny, nz) float32
    cold_count: np.ndarray
    ramp_count: np.ndarray
    warm_count: np.ndarray


@dataclass
class TopKSpike:
    x: float; y: float; z: float
    intensity: float
    vibrational_energy: float
    phase: str
    stype: str
    timestep: int
    stream_id: int


def stream_spikes(
    spike_path: Path,
    centroid: Tuple[float, float, float],
    padding: float,
    voxel: float,
    top_k: int,
    n_strata: int,
) -> Tuple[DensityGrids, List[TopKSpike], Dict[str, Any]]:
    """
    One-pass streaming build of:
      1. A global bounding box (first pass) — no, we do it single-pass using a
         dynamically-sized running bbox.  Instead we use two passes:
           - pass 1: running min/max of spike positions
           - pass 2: allocate grids, fill them, and maintain per-stratum
                     reservoir samples for top-K.
      The two-pass cost is acceptable (each pass ~90s for 7.9M spikes
      at modern NVMe speeds with ijson).

    Also accumulates per-field tallies for the metadata sheet.
    """

    # ----------- pass 1: bounding box -----------
    sys.stderr.write(f"[pass 1/2] scanning spike bounding box from {spike_path} ...\n")
    min_xyz = np.array([+np.inf] * 3)
    max_xyz = np.array([-np.inf] * 3)
    n_scanned = 0
    with spike_path.open("rb") as fh:
        for s in ijson.items(fh, "spikes.item"):
            try:
                p = (float(s["x"]), float(s["y"]), float(s["z"]))
            except (KeyError, TypeError, ValueError):
                continue
            min_xyz = np.minimum(min_xyz, p)
            max_xyz = np.maximum(max_xyz, p)
            n_scanned += 1
            if n_scanned % 1_000_000 == 0:
                sys.stderr.write(f"    scanned {n_scanned:,} spikes\n")
    if n_scanned == 0:
        raise SystemExit("no spikes parsed in pass 1")

    # Grid: from spike bbox with padding, snapped to voxel grid
    origin = np.floor(min_xyz - padding)
    far = np.ceil(max_xyz + padding)
    shape_f = (far - origin) / voxel
    shape = tuple(int(max(1, round(v))) for v in shape_f)
    sys.stderr.write(
        f"[pass 1/2] bbox  min={min_xyz}  max={max_xyz}  n={n_scanned:,}\n"
        f"[pass 1/2] grid  origin={origin}  shape={shape}  voxel={voxel} Å\n"
    )

    # ----------- pass 2: fill grids + stratified top-K -----------
    all_count  = np.zeros(shape, dtype=np.float32)
    intensity  = np.zeros(shape, dtype=np.float32)
    cold_count = np.zeros(shape, dtype=np.float32)
    ramp_count = np.zeros(shape, dtype=np.float32)
    warm_count = np.zeros(shape, dtype=np.float32)

    # Stratified reservoirs: one list per intensity stratum, capped at
    # top_k // n_strata.  We don't know intensity quantile boundaries a
    # priori — we estimate them from a running reservoir of the first N spikes,
    # then refine.  Simpler: split by intensity histogram after pass 2.  Even
    # simpler: keep a global "top-N by intensity" + "top-N by vibrational
    # energy" + "top-N by phase" by maintaining a min-heap per bucket.
    #
    # For med-chem utility, keeping top-K spikes per stratum by intensity is
    # fine — it produces diverse representatives.  We approximate by keeping
    # a reservoir of 5x top_k and downsampling at the end.
    import heapq
    heap_all: List[Tuple[float, int, Dict[str, Any]]] = []
    cap = top_k * 3  # oversample then pick stratified
    seq = 0

    # Per-category tallies
    phase_counter: Counter = Counter()
    type_counter: Counter = Counter()
    source_counter: Counter = Counter()
    stream_counter: Counter = Counter()
    wavelength_counter: Counter = Counter()
    intensity_running_sum = 0.0
    vib_running_sum = 0.0
    dropped_out_of_grid = 0

    sys.stderr.write("[pass 2/2] filling density grids + sampling top-K ...\n")
    with spike_path.open("rb") as fh:
        for s in ijson.items(fh, "spikes.item"):
            try:
                x = float(s["x"]); y = float(s["y"]); z = float(s["z"])
                inten = float(s.get("intensity", 0.0))
                vib = float(s.get("vibrational_energy", 0.0))
                phase = str(s.get("ccns_phase", ""))
                stype = str(s.get("type", ""))
                src = str(s.get("spike_source", ""))
                sid = int(s.get("stream_id", -1))
                wl = float(s.get("wavelength_nm", 0.0))
                ts = int(s.get("timestep", 0))
            except (KeyError, TypeError, ValueError):
                continue

            ix = int((x - origin[0]) / voxel)
            iy = int((y - origin[1]) / voxel)
            iz = int((z - origin[2]) / voxel)
            if 0 <= ix < shape[0] and 0 <= iy < shape[1] and 0 <= iz < shape[2]:
                all_count[ix, iy, iz] += 1.0
                intensity[ix, iy, iz] += inten
                if phase == "cold_hold":
                    cold_count[ix, iy, iz] += 1.0
                elif phase == "ramp":
                    ramp_count[ix, iy, iz] += 1.0
                elif phase == "warm_hold":
                    warm_count[ix, iy, iz] += 1.0
            else:
                dropped_out_of_grid += 1

            phase_counter[phase] += 1
            type_counter[stype] += 1
            source_counter[src] += 1
            stream_counter[sid] += 1
            wavelength_counter[wl] += 1
            intensity_running_sum += inten
            vib_running_sum += vib

            # Oversampled min-heap by intensity
            if len(heap_all) < cap:
                heapq.heappush(heap_all, (inten, seq, {
                    "x": x, "y": y, "z": z,
                    "intensity": inten, "vib": vib,
                    "phase": phase, "stype": stype,
                    "timestep": ts, "sid": sid,
                }))
            elif inten > heap_all[0][0]:
                heapq.heapreplace(heap_all, (inten, seq, {
                    "x": x, "y": y, "z": z,
                    "intensity": inten, "vib": vib,
                    "phase": phase, "stype": stype,
                    "timestep": ts, "sid": sid,
                }))
            seq += 1

    sys.stderr.write(f"[pass 2/2] filled; dropped {dropped_out_of_grid:,} out-of-grid spikes\n")

    grids = DensityGrids(
        origin=origin.astype(np.float64),
        voxel=voxel,
        shape=shape,
        all_count=all_count,
        intensity_sum=intensity,
        cold_count=cold_count,
        ramp_count=ramp_count,
        warm_count=warm_count,
    )

    # Stratify the oversampled top-K by intensity quantile (so the PDB shows
    # diverse representatives, not just the very hottest).
    pool = [entry for _, _, entry in heap_all]
    pool.sort(key=lambda e: e["intensity"], reverse=True)
    bucket_size = len(pool) // n_strata if n_strata > 0 else len(pool)
    per_bucket_cap = max(1, top_k // n_strata)
    top_k_spikes: List[TopKSpike] = []
    for bi in range(n_strata):
        lo = bi * bucket_size
        hi = (bi + 1) * bucket_size if bi < n_strata - 1 else len(pool)
        bucket = pool[lo:hi]
        step = max(1, len(bucket) // per_bucket_cap)
        for e in bucket[::step][:per_bucket_cap]:
            top_k_spikes.append(TopKSpike(
                x=e["x"], y=e["y"], z=e["z"],
                intensity=e["intensity"], vibrational_energy=e["vib"],
                phase=e["phase"], stype=e["stype"],
                timestep=e["timestep"], stream_id=e["sid"],
            ))
    top_k_spikes = top_k_spikes[:top_k]

    stats = {
        "n_spikes_streamed": n_scanned,
        "dropped_out_of_grid": dropped_out_of_grid,
        "phase_counts":      dict(phase_counter),
        "type_counts":       dict(type_counter.most_common(20)),
        "source_counts":     dict(source_counter),
        "stream_counts":     dict(sorted(stream_counter.items())),
        "wavelength_counts": {str(k): v for k, v in wavelength_counter.most_common()},
        "intensity_mean":    intensity_running_sum / max(1, n_scanned),
        "vib_energy_mean":   vib_running_sum / max(1, n_scanned),
        "top_k_actual":      len(top_k_spikes),
    }
    return grids, top_k_spikes, stats


# ============================================================================
# Writers: CCP4 density maps, top-K PDB, PML session
# ============================================================================

def write_ccp4(path: Path, grid: np.ndarray, origin: np.ndarray, voxel: float) -> None:
    """
    Write a grid as MRC/CCP4 with correct origin + voxel size so PyMOL's
    `load ... ccp4` places it in the same world-space as the protein.
    """
    # mrcfile stores data in (z, y, x) — transpose accordingly.
    data = np.ascontiguousarray(grid.transpose(2, 1, 0).astype(np.float32))
    with mrcfile.new(str(path), overwrite=True) as m:
        m.set_data(data)
        m.voxel_size = (voxel, voxel, voxel)
        # nstart fields define the voxel origin relative to (0,0,0); we use
        # the explicit origin header for world-space placement.
        m.header.origin.x = float(origin[0])
        m.header.origin.y = float(origin[1])
        m.header.origin.z = float(origin[2])
        m.header.nxstart = 0
        m.header.nystart = 0
        m.header.nzstart = 0
        m.update_header_from_data()
        m.update_header_stats()


def write_top_spike_pdb(path: Path, spikes: List[TopKSpike]) -> None:
    """
    Encode spikes as HETATM records.

      atom_serial : 1..N
      atom name   : "SPK"
      altLoc      : ' '
      resname     : stype[:3].upper().ljust(3)     (PHE / TYR / TRP / ...)
      chainID     : PHASE_CHAIN[phase] or 'X'
      resSeq      : sequential per spike
      x,y,z       : Å
      occupancy   : intensity normalized to [0,1]
      tempFactor  : vibrational_energy * 1e5 (amplified for visible range)
      element     : "C"
    """
    if not spikes:
        path.write_text("")
        return
    imax = max(1e-12, max(s.intensity for s in spikes))
    lines: List[str] = []
    for i, s in enumerate(spikes, start=1):
        chain = PHASE_CHAIN.get(s.phase, "X")
        resname = (s.stype or "SPK")[:3].upper().ljust(3)
        q = min(1.0, max(0.0, s.intensity / imax))
        b = s.vibrational_energy * 1.0e5
        # PDB column layout (strict) — https://www.wwpdb.org/documentation/file-format-content/format33/sect9.html
        lines.append(
            "HETATM{serial:>5d} {name:<4s}{altLoc:<1s}{resName:<3s} {chainID:<1s}{resSeq:>4d}{iCode:<1s}   "
            "{x:>8.3f}{y:>8.3f}{z:>8.3f}{occ:>6.2f}{temp:>6.2f}          {element:>2s}"
            .format(
                serial=i % 100000,
                name="SPK",
                altLoc=" ",
                resName=resname,
                chainID=chain,
                resSeq=(i % 9999) + 1,
                iCode=" ",
                x=s.x, y=s.y, z=s.z,
                occ=q, temp=b,
                element=" C",
            )
        )
    lines.append("END")
    path.write_text("\n".join(lines) + "\n")


def write_metadata_sheet(path: Path, md: SiteMetadata, stats: Dict[str, Any], raw_pdb: Path) -> None:
    lines: List[str] = []
    L = lines.append
    L(f"SITE-2 VISUALIZATION METADATA — generated by site2_pymol_render.py")
    L("=" * 72)
    L(f"  site id              : {md.site_id}")
    L(f"  centroid (Å)         : ({md.centroid[0]:.4f}, {md.centroid[1]:.4f}, {md.centroid[2]:.4f})")
    if md.ligand_centroid is not None:
        dx = md.centroid[0] - md.ligand_centroid[0]
        dy = md.centroid[1] - md.ligand_centroid[1]
        dz = md.centroid[2] - md.ligand_centroid[2]
        dcc = (dx * dx + dy * dy + dz * dz) ** 0.5
        L(f"  ligand GDP centroid  : {md.ligand_centroid} Å")
        L(f"  DCC to GDP           : {dcc:.3f} Å")
    L(f"  spike_count          : {md.spike_count:,}")
    L(f"  spikes streamed      : {stats['n_spikes_streamed']:,}  (dropped OOG: {stats['dropped_out_of_grid']:,})")
    L(f"  classification       : {md.classification}")
    L(f"  therm_class          : {md.therm_class}")
    L(f"  ccns_class           : {md.ccns_class}  (ccns_tau={md.ccns_tau})")
    L(f"  is_cryptic           : {md.is_cryptic}")
    L(f"  quality_score        : {md.quality_score:.4f}")
    L(f"  druggability_overall : {md.druggability_overall:.4f}")
    L(f"  lining residues      : {md.n_lining}")
    L(f"  protein PDB used     : {raw_pdb}")
    L("")
    L("  PHASE COUNTS (ccns_phase):")
    for k, v in sorted(stats["phase_counts"].items(), key=lambda kv: -kv[1]):
        L(f"    {k:<12s} {v:>12,d}")
    L("")
    L("  TYPE COUNTS (spike residue type, top 20):")
    for k, v in stats["type_counts"].items():
        L(f"    {k:<12s} {v:>12,d}")
    L("")
    L("  SPIKE SOURCE / STREAM / WAVELENGTH:")
    L(f"    sources: {stats['source_counts']}")
    L(f"    streams: {stats['stream_counts']}")
    L(f"    wavelengths: {stats['wavelength_counts']}")
    L("")
    L("  LINING RESIDUES (top 20 by spike_attribution_count):")
    lr_sorted = sorted(md.lining_residues, key=lambda r: -r.get("spike_attribution_count", 0))
    for r in lr_sorted[:20]:
        L(f"    resid={r.get('resid'):>4}  {r.get('resname','?'):<4s}  "
          f"chain={r.get('chain','?')}  catalytic={str(r.get('is_catalytic')):<5s}  "
          f"attrib={r.get('spike_attribution_count','?')}")
    L("")
    L("  PRISM-THERM TOP RESIDUES (role + n_causal_spikes):")
    for r in md.therm_top_residues:
        L(f"    resid={r.get('residue_id'):>4}  {r.get('residue_name','?'):<6s}  "
          f"role={r.get('role','?'):<8s}  n_causal_spikes={r.get('n_causal_spikes',0):>10,}  "
          f"kl={r.get('kl_divergence',0.0):.4f}")
    L("")
    L("  KCC VALIDATION TOP-K DRIVERS:")
    for r in md.kcc_topk_residues:
        ca = r.get("ca_position", [None] * 3)
        kcc = r.get("kcc", {})
        if isinstance(kcc, dict):
            kcc_str = " ".join(f"{k}={v:.4f}" for k, v in sorted(kcc.items())
                               if isinstance(v, (int, float)))
        elif isinstance(kcc, (int, float)):
            kcc_str = f"{kcc:.4f}"
        else:
            kcc_str = str(kcc)
        L(f"    resid={r.get('residue_id'):>4}  CA={ca}  kcc=[{kcc_str}]")
    path.write_text("\n".join(lines) + "\n")


def _grid_quantile_levels(p: Path) -> Tuple[float, float, float]:
    """
    Read a CCP4 map back and return three contour levels at the
    50th / 90th / 99th percentile of *non-zero* voxels.  These are
    the only levels that actually carve out an informative isomesh
    when the density spans many orders of magnitude.
    """
    with mrcfile.open(str(p), mode="r") as m:
        data = np.asarray(m.data, dtype=np.float32).ravel()
    nz = data[data > 0]
    if nz.size < 8:
        # Defensive fallback: spread the levels symbolically over the range
        lo = float(np.percentile(data, 50))
        md = float(np.percentile(data, 90))
        hi = float(np.percentile(data, 99))
    else:
        lo = float(np.percentile(nz, 50))
        md = float(np.percentile(nz, 90))
        hi = float(np.percentile(nz, 99))
    return (lo, md, hi)


def write_pml_session(
    path: Path,
    raw_pdb: Path,
    out_dir: Path,
    md: SiteMetadata,
    grids_files: Dict[str, Path],
    top_spikes_pdb: Path,
) -> None:
    """
    Emit a thick, self-contained .pml that a medicinal chemist can open
    directly in PyMOL and get a ready-to-analyze session with F1..F7 scenes.
    """
    c = md.centroid
    gdp = md.ligand_centroid

    # Build residue-id selection strings for lining and for each therm role
    lining_ids = sorted({r.get("resid") for r in md.lining_residues if r.get("resid") is not None})
    catalytic_ids = sorted({
        r.get("resid") for r in md.lining_residues
        if r.get("is_catalytic") and r.get("resid") is not None
    })
    role_buckets: Dict[str, List[int]] = {}
    for r in md.therm_top_residues:
        rid = r.get("residue_id")
        role = r.get("role") or "UNKNOWN"
        # prism_therm residue_id is 0-indexed; PDB resSeq is 1-indexed → +1
        if rid is not None:
            role_buckets.setdefault(role, []).append(int(rid) + 1)
    kcc_ids = [int(r["residue_id"]) + 1 for r in md.kcc_topk_residues if "residue_id" in r]

    # Note about ID offsets: binding_sites.lining_residues.resid uses PDB author
    # residue numbering (matches the raw PDB resSeq).  prism_therm.residue_id
    # and kcc_validation.residue_id are 0-indexed topology positions; we add 1
    # to align with the PDB's 1-indexed resSeq.  If your PDB uses a different
    # numbering, adjust accordingly.
    def pymol_resid_selection(ids: List[int]) -> str:
        if not ids: return "none"
        return "+".join(str(i) for i in ids)

    lining_sel = pymol_resid_selection(lining_ids)
    catalytic_sel = pymol_resid_selection(catalytic_ids)
    gateway_sel = pymol_resid_selection(role_buckets.get("GATEWAY", []))
    spectat_sel = pymol_resid_selection(role_buckets.get("SPECTAT", []))
    trigger_sel = pymol_resid_selection(role_buckets.get("TRIGGER", []))
    stabil_sel = pymol_resid_selection(role_buckets.get("STABIL", []))
    kcc_sel    = pymol_resid_selection(kcc_ids)

    header_label = f"4LPK site-{md.site_id}  therm={md.therm_class}  drug={md.druggability_overall:.2f}  q={md.quality_score:.2f}"
    dcc_str = ""
    if gdp is not None:
        dcc = ((c[0]-gdp[0])**2 + (c[1]-gdp[1])**2 + (c[2]-gdp[2])**2) ** 0.5
        dcc_str = f"  DCC={dcc:.2f}A"  # ascii-only — PyMOL CLI breaks on non-ASCII

    # Compute quantile-based contour levels from the actual maps so the
    # mesh is informative regardless of total spike count.
    lvl_all  = _grid_quantile_levels(grids_files["all"])
    lvl_int  = _grid_quantile_levels(grids_files["intensity"])
    lvl_cold = _grid_quantile_levels(grids_files["cold"])
    lvl_warm = _grid_quantile_levels(grids_files["warm"])
    lvl_ramp = _grid_quantile_levels(grids_files["ramp"])
    sys.stderr.write(
        f"[info] contour levels (p50/p90/p99 of non-zero voxels):\n"
        f"           all=  {lvl_all}\n"
        f"           int=  {lvl_int}\n"
        f"           cold= {lvl_cold}\n"
        f"           warm= {lvl_warm}\n"
        f"           ramp= {lvl_ramp}\n"
    )

    pml = f"""# =====================================================================
# PRISM-4D site-{md.site_id} visualization — 4LPK / GDP
# Auto-generated by site2_pymol_render.py
#
# Scenes:  F1 Context | F2 Density | F3 Phases | F4 Hotspots
#          F5 Mechanism | F6 Lining | F7 Full
# =====================================================================

# -- viewport / rendering setup
bg_color white
set ray_shadow, 0
set ambient, 0.20
set reflect, 0.15
set specular, 0.30
set ray_trace_mode, 1
set ray_trace_color, grey60
set cartoon_transparency, 0.45
set cartoon_fancy_helices, 1
set cartoon_smooth_loops, 1
set depth_cue, 1
set fog_start, 0.50
set hash_max, 240
set label_size, 11
set label_color, black
set label_outline_color, white

# -- load the full-atom 4LPK (raw, contains GDP HETATMs on chains A and B)
load {raw_pdb}, prot
remove chain B                # keep chain A only (matches ground-truth DCC basis)
remove solvent
remove resn HOH

# -- protein representation
hide everything
show cartoon, prot and polymer
color grey70, prot and polymer
color atomic, prot and polymer and not (name C+N+CA+O)

# -- GDP ligand
create gdp, prot and resn GDP
hide everything, gdp
show sticks, gdp
color yellow, gdp and elem C
show spheres, gdp
set sphere_scale, 0.22, gdp

# -- site-{md.site_id} centroid pseudoatom
pseudoatom site_centroid, pos=[{c[0]:.6f}, {c[1]:.6f}, {c[2]:.6f}]
show spheres, site_centroid
color orange, site_centroid
set sphere_scale, 1.2, site_centroid
label site_centroid, "site-{md.site_id} centroid"

"""

    if gdp is not None:
        pml += f"""# -- GDP centroid pseudoatom + DCC measurement
pseudoatom gdp_centroid, pos=[{gdp[0]:.6f}, {gdp[1]:.6f}, {gdp[2]:.6f}]
show spheres, gdp_centroid
color cyan, gdp_centroid
set sphere_scale, 0.9, gdp_centroid
label gdp_centroid, "GDP centroid"
distance dcc_to_gdp, site_centroid, gdp_centroid
hide labels, dcc_to_gdp
color red, dcc_to_gdp

"""

    # Density isomesh contours
    pml += f"""# -- volumetric spike-count density (all phases)
load {grids_files['all']}, dens_all
isomesh mesh_all_lo, dens_all, {lvl_all[0]:.2f}
isomesh mesh_all_md, dens_all, {lvl_all[1]:.2f}
isomesh mesh_all_hi, dens_all, {lvl_all[2]:.2f}
color yelloworange, mesh_all_lo
color orange,       mesh_all_md
color red,          mesh_all_hi
set mesh_width, 0.8, mesh_all_*

# -- intensity-weighted density (hot-spots for UV-coupled spikes)
load {grids_files['intensity']}, dens_int
isomesh mesh_int, dens_int, {lvl_int[2]:.2f}
color magenta, mesh_int
set mesh_width, 0.8, mesh_int

# -- phase-stratified density
load {grids_files['cold']}, dens_cold
load {grids_files['warm']}, dens_warm
load {grids_files['ramp']}, dens_ramp
isomesh mesh_cold, dens_cold, {lvl_cold[1]:.2f}
isomesh mesh_warm, dens_warm, {lvl_warm[1]:.2f}
isomesh mesh_ramp, dens_ramp, {lvl_ramp[1]:.2f}
color skyblue,    mesh_cold
color firebrick,  mesh_warm
color forest,     mesh_ramp
set mesh_width, 0.6, mesh_cold
set mesh_width, 0.6, mesh_warm
set mesh_width, 0.6, mesh_ramp

# -- top-K representative spikes
load {top_spikes_pdb}, spike_reps
hide everything, spike_reps
show spheres, spike_reps
set sphere_scale, 0.35, spike_reps
# color by phase via chain ID (C=cold, R=ramp, W=warm)
color marine,    spike_reps and chain C
color forest,    spike_reps and chain R
color firebrick, spike_reps and chain W
# alternative: color by intensity (occupancy)
#   spectrum q, blue_white_red, spike_reps

# -- lining residues (from binding_sites.json site id={md.site_id})
select lining, prot and chain A and resi {lining_sel}
select lining_catalytic, prot and chain A and resi {catalytic_sel}
show sticks, lining
color grey50, lining and elem C
color atomic, lining and not (name C+N+CA+O) and not elem C
show sticks, lining_catalytic
color tv_yellow, lining_catalytic and elem C

# -- prism-therm role residues (ID offset applied: 0-indexed topology → 1-indexed resSeq)
select therm_gateway, prot and chain A and resi {gateway_sel}
select therm_spectat, prot and chain A and resi {spectat_sel}
select therm_trigger, prot and chain A and resi {trigger_sel}
select therm_stabil,  prot and chain A and resi {stabil_sel}
select kcc_drivers,   prot and chain A and resi {kcc_sel}
show sticks, therm_gateway
show sticks, therm_spectat
show sticks, therm_trigger
show sticks, therm_stabil
show sticks, kcc_drivers
color forest,   therm_gateway and elem C
color slate,    therm_spectat and elem C
color hotpink,  therm_trigger and elem C
color wheat,    therm_stabil  and elem C
color purple,   kcc_drivers   and elem C

# -- residue labels on KCC drivers ONLY (3 residues — readable; the 16
#    GATEWAY residues are colored but unlabeled to keep the view legible.
#    Manually label any of them in PyMOL via:
#      label byres therm_gateway and name CA, "%s%s" % (oneletter, resi)
label (byres kcc_drivers and name CA), "%s%s (kcc)" % (oneletter, resi)

# -- orient the camera on the site
orient site_centroid
zoom site_centroid, 22

# -- title label at top of viewport
set label_color, black, gdp_centroid
set label_color, black, site_centroid

# ====================================================================
# SCENES — press F1..F7 to toggle
# ====================================================================

# F1 — Context: protein + GDP + site centroid, hide all density / drivers
disable mesh_all_*
disable mesh_int
disable mesh_cold
disable mesh_warm
disable mesh_ramp
hide sticks, therm_gateway
hide sticks, therm_spectat
hide sticks, therm_trigger
hide sticks, therm_stabil
hide sticks, kcc_drivers
hide sticks, lining
hide sticks, lining_catalytic
hide everything, spike_reps
scene F1, store, message="Context: protein + GDP + site-{md.site_id} centroid{dcc_str}"

# F2 — Density (all-phase)
enable mesh_all_lo
enable mesh_all_md
enable mesh_all_hi
scene F2, store, message="Density contours: all-phase spike counts"

# F3 — Phase stratification
disable mesh_all_*
enable mesh_cold
enable mesh_warm
enable mesh_ramp
scene F3, store, message="Phase stratification: cold=blue / warm=red / ramp=green"

# F4 — Hotspots (representative spikes)
disable mesh_cold
disable mesh_warm
disable mesh_ramp
show spheres, spike_reps
enable mesh_int
scene F4, store, message="Hotspots: intensity-weighted mesh + representative spikes"

# F5 — Mechanism (therm residue roles)
disable mesh_int
hide spheres, spike_reps
show sticks, therm_gateway
show sticks, therm_spectat
show sticks, therm_trigger
show sticks, therm_stabil
show sticks, kcc_drivers
scene F5, store, message="Mechanism: PRISM-Therm roles (GATEWAY green / SPECTAT slate / TRIGGER pink / STABIL wheat / KCC driver purple)"

# F6 — Lining residues
hide sticks, therm_gateway
hide sticks, therm_spectat
hide sticks, therm_trigger
hide sticks, therm_stabil
hide sticks, kcc_drivers
show sticks, lining
show sticks, lining_catalytic
scene F6, store, message="Lining shell: {md.n_lining} residues (yellow = catalytic)"

# F7 — Full: everything ON
enable mesh_all_md
enable mesh_cold
enable mesh_warm
show spheres, spike_reps
show sticks, therm_gateway
show sticks, kcc_drivers
scene F7, store, message="Full composite"

# Start at F2
scene F2, recall

# -- print a summary to the PyMOL console
print "============================================================"
print "  PRISM-4D site-{md.site_id} session loaded"
print "  {header_label}{dcc_str}"
print "  site centroid: ({c[0]:.3f}, {c[1]:.3f}, {c[2]:.3f})"
print "  Scenes: F1 Context | F2 Density | F3 Phases | F4 Hotspots"
print "          F5 Mechanism | F6 Lining | F7 Full"
print "  Metadata sheet: {out_dir / 'site2_metadata.txt'}"
print "============================================================"
"""
    path.write_text(pml)


# ============================================================================
# CLI
# ============================================================================

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1] if __doc__ else "")
    ap.add_argument("--root", type=Path, default=DEFAULT_ROOT, help="PRISM run output dir.")
    ap.add_argument("--prefix", default=DEFAULT_PREFIX, help="filename prefix.")
    ap.add_argument("--site-id", type=int, default=DEFAULT_SITE_ID, help="site id to render.")
    ap.add_argument("--raw-pdb", type=Path, default=DEFAULT_RAW_PDB, help="path to raw PDB (has GDP).")
    ap.add_argument("--voxel", type=float, default=DEFAULT_VOXEL_SIZE, help="density voxel size Å.")
    ap.add_argument("--padding", type=float, default=DEFAULT_PADDING, help="Å padding around spike bbox.")
    ap.add_argument("--top-k", type=int, default=DEFAULT_TOP_K, help="representative spike count.")
    ap.add_argument("--n-strata", type=int, default=DEFAULT_N_STRATA, help="intensity strata for top-K.")
    ap.add_argument(
        "--reuse-grids", action="store_true",
        help="Skip the spike-stream pass; reuse already-written CCP4 maps + top_spikes.pdb. "
             "Useful for re-emitting the .pml after fixing a bug downstream.",
    )
    args = ap.parse_args()

    root: Path = args.root.resolve()
    if not root.is_dir():
        sys.stderr.write(f"--root not a directory: {root}\n")
        return 2

    spike_path = root / f"{args.prefix}.site{args.site_id}.spike_events.json"
    if not spike_path.is_file():
        sys.stderr.write(f"spike events file not found: {spike_path}\n")
        return 2

    out_dir = root / f"site{args.site_id}_pymol"
    out_dir.mkdir(exist_ok=True)
    sys.stderr.write(f"[info] writing artifacts under: {out_dir}\n")

    md = collect_metadata(root, args.prefix, args.site_id)
    sys.stderr.write(f"[info] site id={md.site_id} centroid={md.centroid}\n")
    sys.stderr.write(f"[info] lining_residues={md.n_lining} therm_top={len(md.therm_top_residues)} kcc_topk={len(md.kcc_topk_residues)}\n")

    gmap: Dict[str, Path] = {
        "all":       out_dir / "density_all.ccp4",
        "intensity": out_dir / "density_intensity.ccp4",
        "cold":      out_dir / "density_cold_phase.ccp4",
        "warm":      out_dir / "density_warm_phase.ccp4",
        "ramp":      out_dir / "density_ramp_phase.ccp4",
    }
    top_pdb = out_dir / "top_spikes.pdb"

    if args.reuse_grids and all(p.is_file() for p in gmap.values()) and top_pdb.is_file():
        sys.stderr.write("[info] --reuse-grids: skipping the spike-stream pass\n")
        stats = {
            "n_spikes_streamed": 0, "dropped_out_of_grid": 0,
            "phase_counts": {"(skipped)": 0}, "type_counts": {"(skipped)": 0},
            "source_counts": {}, "stream_counts": {}, "wavelength_counts": {},
            "intensity_mean": 0.0, "vib_energy_mean": 0.0,
            "top_k_actual": sum(1 for _ in top_pdb.read_text().splitlines() if _.startswith("HETATM")),
        }
    else:
        grids, top_spikes, stats = stream_spikes(
            spike_path=spike_path,
            centroid=md.centroid,
            padding=args.padding,
            voxel=args.voxel,
            top_k=args.top_k,
            n_strata=args.n_strata,
        )
        write_ccp4(gmap["all"],       grids.all_count,     grids.origin, grids.voxel)
        write_ccp4(gmap["intensity"], grids.intensity_sum, grids.origin, grids.voxel)
        write_ccp4(gmap["cold"],      grids.cold_count,    grids.origin, grids.voxel)
        write_ccp4(gmap["warm"],      grids.warm_count,    grids.origin, grids.voxel)
        write_ccp4(gmap["ramp"],      grids.ramp_count,    grids.origin, grids.voxel)
        sys.stderr.write(f"[info] wrote 5 CCP4 density maps (voxel={args.voxel}Å, shape={grids.shape})\n")
        write_top_spike_pdb(top_pdb, top_spikes)
        sys.stderr.write(f"[info] wrote {len(top_spikes)} representative spikes to {top_pdb}\n")

    # Metadata sheet
    meta_txt = out_dir / f"site{args.site_id}_metadata.txt"
    write_metadata_sheet(meta_txt, md, stats, args.raw_pdb)
    sys.stderr.write(f"[info] wrote metadata sheet to {meta_txt}\n")

    # PML session
    pml_path = out_dir / f"site{args.site_id}_session.pml"
    write_pml_session(pml_path, args.raw_pdb.resolve(), out_dir, md, gmap, top_pdb)
    sys.stderr.write(f"[info] wrote PyMOL session to {pml_path}\n")

    sys.stderr.write("\n" + "=" * 68 + "\n")
    sys.stderr.write("  To open the session in PyMOL (GUI):\n")
    sys.stderr.write(f"    pymol {pml_path}\n")
    sys.stderr.write("  The chemist sees F1..F7 scenes for:\n")
    sys.stderr.write("    F1 Context | F2 Density | F3 Phases | F4 Hotspots\n")
    sys.stderr.write("    F5 Mechanism | F6 Lining | F7 Full\n")
    sys.stderr.write("=" * 68 + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
