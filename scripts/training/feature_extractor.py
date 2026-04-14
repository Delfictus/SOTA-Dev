#!/usr/bin/env python3
"""Per-residue feature extractor for PRISM-4D teacher v004 / VN-EGNN v001.

Reconstructs the feature_registry_216 spec from R2 artifacts produced by the
canonical engine run, augmented with TIDE role + n_causal_spikes and ESM-2.

Input: target name. R2 objects pulled:
  <target>_clean.pdb                        — structural source
  <target>_ground_truth.json                — ligand centroid for labels
  <target>.binding_sites.json               — Block 5 (site-mapped per-residue)
  <target>.kcc_visualization.json           — Block 3 (KCC dynamics)
  <target>.topology.prism_therm.json        — Block 4 thermo + TIDE role
  <target>.site*.spike_events.parquet|json  — Block 6 (multi-phase spikes)

Output: np.float32 matrix [N_residues, FEATURE_DIM] with residue-aligned rows.
FEATURE_DIM is printed at the top of every run — it is NOT a constant.

Per-block budget (honest accounting):
  Block 1  Structural       25   AA one-hot 20 + DSSP 3 + SASA + hydrophobicity
  Block 2  NMA              26   20 mode disp + sqfluct + stiffness + hinge + effect + sens + long_range_corr
  Perturb  NMA               5   amplitude_ratio, alignment, centroid_shift, ligand_cosine, variance_ratio
  Block 3  KCC              13   9 from kcc_visualization.json + displacement_mag + 3 zero-padded (total_steps/motion_rate/active_frac)
  Block 4  Thermo            4   transfer_entropy, causal_dg, fisher_info, kl_divergence (per-residue aggregate)
  Block 5  Site-mapped      18   per-residue max/mean over sites the residue lines
  Block 6  Multi-phase      48   per-phase spike stats (spikes in 5Å sphere, grouped by ccns_phase)
  Block 7  Dynamic ensemble 81   STUBBED — zero-padded; requires trajectory data not consistently in R2
  TIDE     Aux               5   role one-hot (4: GATEWAY/SPECTAT/STABIL./TRIGGER) + n_causal_spikes
  ESM-2                   1280   per-residue 1280-dim embedding
  ─────────────────────────────
  TOTAL    FEATURE_DIM    1505

Labels: binding = Cα within 4.5Å of ligand centroid (from ground_truth.json).

Usage:
    from feature_extractor import extract_target
    bundle = extract_target("10dc_chainA", cache_dir=Path("/workspace/cache"))
    # bundle = {X, residue_ids, labels, ligand_centroid, ca_coords, target, dim}
"""
from __future__ import annotations

import json
import math
import os
import subprocess
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# ─────────────────────────────────────────────────────────────
#  Constants
# ─────────────────────────────────────────────────────────────

R2_BUCKET = os.environ.get("R2_BUCKET", "prism-archive")
R2_PREFIX = os.environ.get("R2_PREFIX", f"r2:{R2_BUCKET}/10k-runs")

# Amino acid ordering (canonical 20, alphabetic by 3-letter code)
AA_3 = ["ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE",
        "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL"]
AA_IDX = {a: i for i, a in enumerate(AA_3)}

# Kyte-Doolittle hydrophobicity (1982)
HYDROPHOBICITY = {
    "ALA": 1.8, "ARG": -4.5, "ASN": -3.5, "ASP": -3.5, "CYS": 2.5,
    "GLN": -3.5, "GLU": -3.5, "GLY": -0.4, "HIS": -3.2, "ILE": 4.5,
    "LEU": 3.8, "LYS": -3.9, "MET": 1.9, "PHE": 2.8, "PRO": -1.6,
    "SER": -0.8, "THR": -0.7, "TRP": -0.9, "TYR": -1.3, "VAL": 4.2,
}

TIDE_ROLES = ["GATEWAY", "SPECTAT", "STABIL.", "TRIGGER"]
TIDE_ROLE_IDX = {r: i for i, r in enumerate(TIDE_ROLES)}

# Per-block dimensions
DIM_STRUCT = 25   # 20 AA + 3 DSSP + SASA + hydrophobicity
DIM_NMA = 26
DIM_PERTURB = 5
DIM_KCC = 13
DIM_THERMO = 4
DIM_SITE = 18
DIM_PHASE = 48
DIM_DYN = 81      # stubbed
DIM_TIDE_AUX = 5  # 4 role one-hot + n_causal_spikes
DIM_ESM = 1280

FEATURE_DIM = (DIM_STRUCT + DIM_NMA + DIM_PERTURB + DIM_KCC + DIM_THERMO
               + DIM_SITE + DIM_PHASE + DIM_DYN + DIM_TIDE_AUX + DIM_ESM)

BLOCK_OFFSETS = {
    "structural": (0, DIM_STRUCT),
    "nma":        (DIM_STRUCT, DIM_STRUCT + DIM_NMA),
    "perturb":    (DIM_STRUCT + DIM_NMA, DIM_STRUCT + DIM_NMA + DIM_PERTURB),
    "kcc":        (DIM_STRUCT + DIM_NMA + DIM_PERTURB, DIM_STRUCT + DIM_NMA + DIM_PERTURB + DIM_KCC),
    "thermo":     (DIM_STRUCT + DIM_NMA + DIM_PERTURB + DIM_KCC,
                   DIM_STRUCT + DIM_NMA + DIM_PERTURB + DIM_KCC + DIM_THERMO),
    "site":       (DIM_STRUCT + DIM_NMA + DIM_PERTURB + DIM_KCC + DIM_THERMO,
                   DIM_STRUCT + DIM_NMA + DIM_PERTURB + DIM_KCC + DIM_THERMO + DIM_SITE),
    "phase":      (DIM_STRUCT + DIM_NMA + DIM_PERTURB + DIM_KCC + DIM_THERMO + DIM_SITE,
                   DIM_STRUCT + DIM_NMA + DIM_PERTURB + DIM_KCC + DIM_THERMO + DIM_SITE + DIM_PHASE),
    "dynamic":    (DIM_STRUCT + DIM_NMA + DIM_PERTURB + DIM_KCC + DIM_THERMO + DIM_SITE + DIM_PHASE,
                   DIM_STRUCT + DIM_NMA + DIM_PERTURB + DIM_KCC + DIM_THERMO + DIM_SITE + DIM_PHASE + DIM_DYN),
    "tide_aux":   (DIM_STRUCT + DIM_NMA + DIM_PERTURB + DIM_KCC + DIM_THERMO + DIM_SITE + DIM_PHASE + DIM_DYN,
                   DIM_STRUCT + DIM_NMA + DIM_PERTURB + DIM_KCC + DIM_THERMO + DIM_SITE + DIM_PHASE + DIM_DYN + DIM_TIDE_AUX),
    "esm":        (FEATURE_DIM - DIM_ESM, FEATURE_DIM),
}

BINDING_LABEL_CUTOFF = 4.5  # Å from ligand centroid
N_SPIKE_PHASES = 5
SPIKE_NEAR_RADIUS = 5.0     # Å sphere around Cα for Block 6


@dataclass
class TargetBundle:
    target: str
    X: np.ndarray            # [N, FEATURE_DIM] float32
    residue_ids: np.ndarray  # [N] int32
    residue_names: List[str]
    labels: np.ndarray       # [N] int32 (0/1)
    ligand_centroid: np.ndarray  # [3] float32
    ca_coords: np.ndarray    # [N, 3] float32
    dim: int
    block_offsets: Dict[str, Tuple[int, int]]
    missing_blocks: List[str]


# ─────────────────────────────────────────────────────────────
#  R2 staging
# ─────────────────────────────────────────────────────────────

def _rclone(src: str, dst_dir: Path, timeout: int = 300) -> bool:
    dst_dir.mkdir(parents=True, exist_ok=True)
    r = subprocess.run(
        ["rclone", "copy", src, str(dst_dir), "--quiet", "--transfers", "4"],
        capture_output=True, timeout=timeout, check=False,
    )
    return r.returncode == 0


def stage_target(target: str, cache_root: Path, want_parquets: bool = True) -> Path:
    """Download the R2 artifacts we need into cache_root/<target>/. Returns the dir."""
    out = cache_root / target
    out.mkdir(parents=True, exist_ok=True)

    needed = [
        f"{target}_clean.pdb",
        f"{target}_ground_truth.json",
        f"{target}.binding_sites.json",
        f"{target}.kcc_visualization.json",
        f"{target}.topology.prism_therm.json",
    ]
    for name in needed:
        p = out / name
        if p.exists() and p.stat().st_size > 0:
            continue
        _rclone(f"{R2_PREFIX}/{target}/{name}", out, timeout=120)

    if want_parquets:
        # Pull parquets (preferred) — skip JSON spike dumps (too big)
        # rclone does a single listing; include filter on *.parquet
        subprocess.run(
            ["rclone", "copy", f"{R2_PREFIX}/{target}/",
             str(out), "--include", "*.spike_events.parquet",
             "--transfers", "8", "--quiet"],
            capture_output=True, timeout=600, check=False,
        )
    return out


# ─────────────────────────────────────────────────────────────
#  Block 1 — Structural (25)
# ─────────────────────────────────────────────────────────────

def compute_structural(pdb_path: Path) -> Tuple[np.ndarray, List[int], List[str], np.ndarray]:
    """Returns (X_struct [N,25], residue_ids [N], residue_names [N], ca_coords [N,3])."""
    # Pure-Python PDB parser — ATOM records, CA only
    residues = {}  # resid -> dict
    with open(pdb_path) as f:
        for line in f:
            if not line.startswith("ATOM"):
                continue
            name = line[12:16].strip()
            if name != "CA":
                continue
            try:
                resid = int(line[22:26])
                resname = line[17:20].strip()
                x, y, z = float(line[30:38]), float(line[38:46]), float(line[46:54])
                b = float(line[60:66])
            except (ValueError, IndexError):
                continue
            residues[resid] = {"resname": resname, "ca": (x, y, z), "b_factor": b}

    # DSSP (needed for SS + SASA). Fall back to zeros if DSSP unavailable.
    dssp = _run_dssp(pdb_path)
    for resid, entry in residues.items():
        d = dssp.get(resid, {})
        entry["ss"] = d.get("ss", "-")  # H/E/C/-
        entry["sasa"] = d.get("sasa", 0.0)

    resids_sorted = sorted(residues.keys())
    N = len(resids_sorted)
    X = np.zeros((N, DIM_STRUCT), dtype=np.float32)
    ca = np.zeros((N, 3), dtype=np.float32)
    names = []

    for i, rid in enumerate(resids_sorted):
        r = residues[rid]
        resname = r["resname"]
        names.append(resname)
        # AA one-hot (20)
        idx = AA_IDX.get(resname)
        if idx is not None:
            X[i, idx] = 1.0
        # DSSP (3): H, E, C
        ss = r["ss"]
        if ss in ("H", "G", "I"):   # helical
            X[i, 20] = 1.0
        elif ss in ("E", "B"):       # sheet
            X[i, 21] = 1.0
        else:                        # coil/other
            X[i, 22] = 1.0
        # SASA (1) — normalize to max ~220 Å² then clip
        X[i, 23] = float(r["sasa"]) / 220.0
        # Hydrophobicity (1) — Kyte-Doolittle, normalize /5
        X[i, 24] = HYDROPHOBICITY.get(resname, 0.0) / 5.0

        ca[i] = r["ca"]

    return X, resids_sorted, names, ca


def _run_dssp(pdb_path: Path) -> Dict[int, Dict[str, Any]]:
    """Try freesasa + mkdssp; fall back to empty dict if either fails."""
    out: Dict[int, Dict[str, Any]] = {}

    # 1) SASA via freesasa (Python binding if available, else CLI)
    try:
        import freesasa
        structure = freesasa.Structure(str(pdb_path))
        result = freesasa.Calc().calculate(structure)
        residue_areas = result.residueAreas()
        for chain, per_res in residue_areas.items():
            for resnum_str, area in per_res.items():
                try:
                    rid = int(resnum_str)
                    out.setdefault(rid, {})["sasa"] = float(area.total)
                except (ValueError, AttributeError):
                    pass
    except Exception:
        pass  # freesasa not installed — SASA stays 0

    # 2) DSSP (secondary structure) via mkdssp
    try:
        # Prepend HEADER if missing (mkdssp requirement)
        with open(pdb_path) as f:
            first_line = f.readline()
        needs_header = not first_line.startswith("HEADER")
        tmp_pdb = pdb_path
        if needs_header:
            tmp_pdb = pdb_path.with_suffix(".hdr.pdb")
            with open(tmp_pdb, "w") as f:
                f.write("HEADER    PROTEIN                                 01-JAN-00   XXXX              \n")
                with open(pdb_path) as src:
                    f.write(src.read())

        r = subprocess.run(
            ["mkdssp", str(tmp_pdb), "/dev/stdout"],
            capture_output=True, timeout=60, text=True, check=False,
        )
        if r.returncode == 0:
            in_data = False
            for line in r.stdout.splitlines():
                if line.startswith("  #  RESIDUE"):
                    in_data = True
                    continue
                if not in_data:
                    continue
                try:
                    resnum_str = line[5:10].strip()
                    if not resnum_str:
                        continue
                    rid = int(resnum_str)
                    ss = line[16]
                    out.setdefault(rid, {})["ss"] = ss if ss != " " else "-"
                except (ValueError, IndexError):
                    continue

        if needs_header and tmp_pdb.exists():
            tmp_pdb.unlink(missing_ok=True)
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        pass  # mkdssp not available — SS stays "-" → coil bit set

    return out


# ─────────────────────────────────────────────────────────────
#  Block 2 — NMA (26) + Perturbed NMA (5)
# ─────────────────────────────────────────────────────────────

def compute_nma(pdb_path: Path, ligand_centroid: Optional[np.ndarray],
                ca_coords: np.ndarray, residue_ids: List[int]
                ) -> Tuple[np.ndarray, np.ndarray]:
    """Returns (nma [N,26], perturbed [N,5]). ProDy-based; zeros if ProDy missing."""
    N = len(residue_ids)
    nma = np.zeros((N, DIM_NMA), dtype=np.float32)
    perturb = np.zeros((N, DIM_PERTURB), dtype=np.float32)

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            import prody
            prody.confProDy(verbosity="none")

            atoms = prody.parsePDB(str(pdb_path), subset="ca")
            if atoms is None or atoms.numAtoms() < 10:
                return nma, perturb
            anm = prody.ANM("anm")
            anm.buildHessian(atoms, cutoff=15.0)
            anm.calcModes(n_modes=20)
            vectors = anm.getArray()  # [3N, 20]
            eigenvalues = anm.getEigvals()  # [20]
            sqfluct = prody.calcSqFlucts(anm)  # [N]
            stiffness = prody.calcMechStiff(anm, atoms).mean(axis=0) if hasattr(prody, "calcMechStiff") else np.zeros(atoms.numAtoms())
            hinge = prody.calcHinges(anm, f=0.5) if hasattr(prody, "calcHinges") else []

            # Map atoms back to residue_ids
            anm_resids = atoms.getResnums()
            anm_map = {int(r): i for i, r in enumerate(anm_resids)}

            # Effectiveness / sensitivity (PRS, if available) — optional
            try:
                prs_matrix, eff, sens = prody.calcPerturbResponse(anm)
            except Exception:
                eff = np.zeros(atoms.numAtoms()); sens = np.zeros(atoms.numAtoms())

            # Long-range correlation: |corr_matrix| mean, excluding diagonal
            cross_corr = prody.calcCrossCorr(anm)  # [N,N]
            np.fill_diagonal(cross_corr, 0.0)
            long_range = np.abs(cross_corr).mean(axis=1)

            for i, rid in enumerate(residue_ids):
                ai = anm_map.get(rid)
                if ai is None:
                    continue
                # 20 mode displacements (||vec_i|| per mode)
                for m in range(min(20, vectors.shape[1])):
                    v = vectors[ai*3:(ai+1)*3, m]
                    nma[i, m] = float(np.linalg.norm(v))
                nma[i, 20] = float(sqfluct[ai]) if ai < len(sqfluct) else 0.0
                nma[i, 21] = float(stiffness[ai]) if ai < len(stiffness) else 0.0
                nma[i, 22] = 1.0 if ai in hinge else 0.0
                nma[i, 23] = float(eff[ai]) if ai < len(eff) else 0.0
                nma[i, 24] = float(sens[ai]) if ai < len(sens) else 0.0
                nma[i, 25] = float(long_range[ai]) if ai < len(long_range) else 0.0

            # Perturbed NMA — 5 features vs ligand centroid
            if ligand_centroid is not None:
                # centroid-shift: for each residue, dot(mode1_vec, (ligand - ca))
                dir_to_lig = ligand_centroid[None, :] - ca_coords  # [N,3]
                dir_norm = np.linalg.norm(dir_to_lig, axis=1, keepdims=True).clip(min=1e-6)
                dir_unit = dir_to_lig / dir_norm  # [N,3]
                amp_mode1 = np.linalg.norm(vectors[:, 0].reshape(-1, 3), axis=1)  # [n_atoms]
                amp_mode2 = np.linalg.norm(vectors[:, 1].reshape(-1, 3), axis=1)  # [n_atoms]
                mode1_vec = vectors[:, 0].reshape(-1, 3)  # [n_atoms, 3]
                for i, rid in enumerate(residue_ids):
                    ai = anm_map.get(rid)
                    if ai is None:
                        continue
                    a1 = amp_mode1[ai]
                    a2 = amp_mode2[ai] if ai < len(amp_mode2) else 0.0
                    perturb[i, 0] = float(a1 / (a2 + 1e-6))              # amplitude_ratio
                    # alignment = |mode1_vec · dir_to_lig|
                    v = mode1_vec[ai]
                    nv = np.linalg.norm(v) + 1e-6
                    perturb[i, 1] = float(abs(np.dot(v / nv, dir_unit[i])))
                    perturb[i, 2] = float(a1 * perturb[i, 1])            # centroid_shift proxy
                    perturb[i, 3] = float(np.dot(v / nv, dir_unit[i]))   # ligand_cosine (signed)
                    perturb[i, 4] = float(eigenvalues[1] / (eigenvalues[0] + 1e-6)) if len(eigenvalues) > 1 else 0.0
    except ImportError:
        pass  # prody not installed

    return nma, perturb


# ─────────────────────────────────────────────────────────────
#  Block 3 — KCC (13)
# ─────────────────────────────────────────────────────────────

KCC_ORDER = ["kcc_score", "burst_motion", "causal_lag", "direction_score",
             "lag_corr_peak", "local_cov", "motion_efficiency", "sum_motion",
             "active_causal_steps", "displacement_mag"]  # 10 populated; remaining 3 zero-padded


def compute_kcc(kcc_path: Path, residue_ids: List[int]) -> np.ndarray:
    N = len(residue_ids)
    out = np.zeros((N, DIM_KCC), dtype=np.float32)
    if not kcc_path.exists():
        return out
    try:
        d = json.load(open(kcc_path))
    except (json.JSONDecodeError, OSError):
        return out
    rid_to_idx = {r: i for i, r in enumerate(residue_ids)}
    for r in d.get("residues", []):
        rid = r.get("residue_id")
        if rid not in rid_to_idx:
            continue
        i = rid_to_idx[rid]
        for j, k in enumerate(KCC_ORDER[:-1]):   # first 9 named
            out[i, j] = float(r.get(k, 0.0) or 0.0)
        dx, dy, dz = r.get("net_dx", 0.0), r.get("net_dy", 0.0), r.get("net_dz", 0.0)
        out[i, 9] = float(math.sqrt(dx*dx + dy*dy + dz*dz))
        # out[i, 10..12] stay zero (total_steps/motion_rate/active_frac unavailable here)
    return out


# ─────────────────────────────────────────────────────────────
#  Block 4 — Thermo (4) + TIDE aux (5)
# ─────────────────────────────────────────────────────────────

def compute_thermo_and_tide(therm_path: Path, residue_ids: List[int]
                            ) -> Tuple[np.ndarray, np.ndarray]:
    """Returns (thermo [N,4], tide_aux [N,5]). TIDE aux = role one-hot (4) + n_causal_spikes (1)."""
    N = len(residue_ids)
    thermo = np.zeros((N, DIM_THERMO), dtype=np.float32)
    aux = np.zeros((N, DIM_TIDE_AUX), dtype=np.float32)
    if not therm_path.exists():
        return thermo, aux
    try:
        d = json.load(open(therm_path))
    except (json.JSONDecodeError, OSError):
        return thermo, aux

    rid_to_idx = {r: i for i, r in enumerate(residue_ids)}
    # Aggregate: residue may appear in many pockets — take MAX per feature
    agg: Dict[int, Dict[str, float]] = {}
    for p in d.get("pockets", []):
        for r in p.get("top_residues", []):
            rid = r.get("residue_id")
            if rid not in rid_to_idx:
                continue
            e = agg.setdefault(rid, {"te": 0.0, "dg": 0.0, "fi": 0.0, "kl": 0.0,
                                     "role_idx": -1, "n_spikes": 0})
            e["te"] = max(e["te"], float(r.get("transfer_entropy", 0.0) or 0.0))
            e["dg"] = max(e["dg"], float(r.get("causal_dg", 0.0) or 0.0))
            e["fi"] = max(e["fi"], float(r.get("fisher_info", 0.0) or 0.0))
            e["kl"] = max(e["kl"], float(r.get("kl_divergence", 0.0) or 0.0))
            e["n_spikes"] = max(e["n_spikes"], int(r.get("n_causal_spikes", 0) or 0))
            role = r.get("role", "")
            if role in TIDE_ROLE_IDX:
                e["role_idx"] = TIDE_ROLE_IDX[role]

    for rid, e in agg.items():
        i = rid_to_idx[rid]
        thermo[i, 0] = e["te"]
        thermo[i, 1] = e["dg"]
        thermo[i, 2] = e["fi"]
        thermo[i, 3] = e["kl"]
        if e["role_idx"] >= 0:
            aux[i, e["role_idx"]] = 1.0
        aux[i, 4] = float(e["n_spikes"])
    return thermo, aux


# ─────────────────────────────────────────────────────────────
#  Block 5 — Site-mapped (18)
# ─────────────────────────────────────────────────────────────

SITE_FIELDS = ["spike_count", "quality_score", "engine_geo", "engine_chem",
               "engine_phys", "druggability", "burial_score", "sphericity",
               "onset_score", "breathing_score", "aromatic_score",
               "frustrated_solvent_score", "source_diversity", "wd_coherence",
               "volume", "relative_asymmetry", "hysteresis_asymmetry", "ccns_tau"]


def compute_site_mapped(binding_sites_path: Path, residue_ids: List[int]) -> np.ndarray:
    N = len(residue_ids)
    out = np.zeros((N, DIM_SITE), dtype=np.float32)
    if not binding_sites_path.exists():
        return out
    try:
        d = json.load(open(binding_sites_path))
    except (json.JSONDecodeError, OSError):
        return out

    rid_to_idx = {r: i for i, r in enumerate(residue_ids)}
    # For each residue, take the MAX value across all sites in which the residue lines
    for s in d.get("sites", []):
        values = {k: float(s.get(k, 0.0) or 0.0) for k in SITE_FIELDS}
        for lr in s.get("lining_residues", []):
            rid = lr.get("residue_id") if isinstance(lr, dict) else None
            if rid is None or rid not in rid_to_idx:
                continue
            i = rid_to_idx[rid]
            for j, k in enumerate(SITE_FIELDS):
                if values[k] > out[i, j]:
                    out[i, j] = values[k]
    return out


# ─────────────────────────────────────────────────────────────
#  Block 6 — Multi-phase spike features (48)
# ─────────────────────────────────────────────────────────────

PHASE_FIELDS = ["spike_count", "intensity_mean", "intensity_std",
                "vib_energy_mean", "water_density_mean", "n_excited_mean"]
# Plus LADD/COFIRE counts and onset/return summaries = 48 total


def compute_phase_features(target_dir: Path, ca_coords: np.ndarray,
                           residue_ids: List[int]) -> np.ndarray:
    """Per-residue Block 6: 5 phases × 6 stats + per-phase LADD/COFIRE + summary =
    30 + 10 + 3 asymmetries + 2 onset/return + 3 auxiliary = 48 dims."""
    N = len(residue_ids)
    out = np.zeros((N, DIM_PHASE), dtype=np.float32)

    parquets = sorted(target_dir.glob("*.spike_events.parquet"))
    if not parquets:
        return out

    try:
        import pyarrow.parquet as pq
        import pandas as pd
    except ImportError:
        return out

    dfs = []
    for pf in parquets:
        try:
            t = pq.read_table(pf, columns=["x", "y", "z", "intensity",
                                           "ccns_phase", "timestep", "spike_source"])
            dfs.append(t.to_pandas())
        except Exception:
            continue
    if not dfs:
        return out
    all_spikes = pd.concat(dfs, ignore_index=True)

    if len(all_spikes) == 0:
        return out

    coords = all_spikes[["x", "y", "z"]].to_numpy(dtype=np.float32)
    intens = all_spikes["intensity"].to_numpy(dtype=np.float32)
    phase = all_spikes["ccns_phase"].to_numpy(dtype=np.int32)
    timestep = all_spikes["timestep"].to_numpy(dtype=np.int32)

    # Aux columns may or may not be present (older runs lack them)
    def _col(name: str) -> np.ndarray:
        if name in all_spikes.columns:
            return all_spikes[name].to_numpy(dtype=np.float32)
        return np.zeros(len(all_spikes), dtype=np.float32)
    vib = _col("vibrational_energy")
    water = _col("water_density")
    nearby = _col("n_nearby_excited")

    r2 = SPIKE_NEAR_RADIUS ** 2
    # For memory: iterate residues (N typically <500), vectorize across spikes
    for i, ca in enumerate(ca_coords):
        d2 = ((coords - ca) ** 2).sum(axis=1)
        near = d2 <= r2
        if not near.any():
            continue
        n_total = int(near.sum())
        # 5 phases × 6 stats = 30
        for ph in range(N_SPIKE_PHASES):
            mask = near & (phase == ph)
            if not mask.any():
                continue
            base = ph * 6
            out[i, base + 0] = float(mask.sum())
            out[i, base + 1] = float(intens[mask].mean())
            out[i, base + 2] = float(intens[mask].std())
            out[i, base + 3] = float(vib[mask].mean())
            out[i, base + 4] = float(water[mask].mean())
            out[i, base + 5] = float(nearby[mask].mean())
        # LADD count (using n_nearby_excited ≥ 2) and COFIRE (≥ 3) per phase = 10 dims
        for ph in range(N_SPIKE_PHASES):
            mask = near & (phase == ph)
            if not mask.any():
                continue
            out[i, 30 + ph] = float(((nearby[mask] >= 2)).sum())     # LADD
            out[i, 35 + ph] = float(((nearby[mask] >= 3)).sum())     # COFIRE
        # Asymmetry: intensity_ratio(warm_hold / cool), vib_ratio, water_ratio  (3 dims)
        m_warm = near & (phase == 2)
        m_cool = near & (phase == 4)
        if m_warm.any() and m_cool.any():
            out[i, 40] = float(intens[m_warm].mean() / (intens[m_cool].mean() + 1e-6))
            out[i, 41] = float(vib[m_warm].mean() / (vib[m_cool].mean() + 1e-6))
            out[i, 42] = float(water[m_warm].mean() / (water[m_cool].mean() + 1e-6))
        # residue_return_ratio (42 → 40 was asymmetry, so: 43 return + 44 onset_phase)
        first_ts = int(timestep[near].min())
        last_ts = int(timestep[near].max())
        ts_range = max(last_ts - first_ts, 1)
        out[i, 43] = float(n_total / ts_range)                       # return rate
        # onset_phase: which phase had first spike
        for ph in range(N_SPIKE_PHASES):
            if (near & (phase == ph)).any():
                out[i, 44] = float(ph)
                break
        # ladd_depletion_mean across last phase
        m_last = near & (phase == N_SPIKE_PHASES - 1)
        if m_last.any():
            out[i, 45] = float(nearby[m_last].mean())
        # cofire_count_near (pooled)
        out[i, 46] = float((near & (nearby >= 3)).sum())
        # ladd_onset_timestep
        ladd_mask = near & (nearby >= 2)
        if ladd_mask.any():
            out[i, 47] = float(timestep[ladd_mask].min() - first_ts)

    return out


# ─────────────────────────────────────────────────────────────
#  Block 7 — Dynamic ensemble (81, STUBBED)
# ─────────────────────────────────────────────────────────────

def compute_dynamic_ensemble(target_dir: Path, ca_coords: np.ndarray) -> np.ndarray:
    """Block 7 requires trajectory arrays (per-frame Cα, DSSP timelines, dihedrals).
    The canonical engine does not archive full trajectories to R2, so this block is
    zero-padded. Downstream: model will learn 'Block 7 absent' as a constant → no harm.
    """
    return np.zeros((ca_coords.shape[0], DIM_DYN), dtype=np.float32)


# ─────────────────────────────────────────────────────────────
#  ESM-2 (1280)
# ─────────────────────────────────────────────────────────────

_ESM_MODEL = None
_ESM_BATCH_CONVERTER = None
_ESM_DEVICE = None

def _load_esm2():
    global _ESM_MODEL, _ESM_BATCH_CONVERTER, _ESM_DEVICE
    if _ESM_MODEL is not None:
        return _ESM_MODEL, _ESM_BATCH_CONVERTER, _ESM_DEVICE
    import torch, esm
    model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
    _ESM_MODEL = model.eval()
    _ESM_BATCH_CONVERTER = alphabet.get_batch_converter()
    _ESM_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _ESM_MODEL = _ESM_MODEL.to(_ESM_DEVICE)
    return _ESM_MODEL, _ESM_BATCH_CONVERTER, _ESM_DEVICE


def compute_esm2(residue_names: List[str]) -> np.ndarray:
    """Returns [N, 1280] ESM-2 t33 per-residue embeddings."""
    aa1 = {
        "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
        "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
        "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
        "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
    }
    seq = "".join(aa1.get(n, "X") for n in residue_names)
    if not seq:
        return np.zeros((0, DIM_ESM), dtype=np.float32)

    try:
        import torch
        model, batcher, device = _load_esm2()
        _, _, tokens = batcher([("query", seq)])
        tokens = tokens.to(device)
        with torch.no_grad():
            out = model(tokens, repr_layers=[33], return_contacts=False)
        reps = out["representations"][33][0]  # [L+2, 1280] — includes BOS/EOS
        # Strip BOS (index 0) and EOS (last). Remaining length = len(seq).
        reps = reps[1:-1]
        return reps.cpu().numpy().astype(np.float32)
    except (ImportError, RuntimeError) as e:
        print(f"  ESM-2 unavailable: {e}", flush=True)
        return np.zeros((len(residue_names), DIM_ESM), dtype=np.float32)


# ─────────────────────────────────────────────────────────────
#  Labels
# ─────────────────────────────────────────────────────────────

def compute_labels(ca_coords: np.ndarray, ligand_centroid: Optional[np.ndarray]) -> np.ndarray:
    N = ca_coords.shape[0]
    if ligand_centroid is None:
        return np.zeros(N, dtype=np.int32)
    d = np.linalg.norm(ca_coords - ligand_centroid[None, :], axis=1)
    return (d <= BINDING_LABEL_CUTOFF).astype(np.int32)


# ─────────────────────────────────────────────────────────────
#  Top-level: extract one target
# ─────────────────────────────────────────────────────────────

def extract_target(target: str, cache_root: Path,
                   compute_esm: bool = True) -> Optional[TargetBundle]:
    """Full feature bundle for one target. Returns None if pipeline fails."""
    target_dir = stage_target(target, cache_root, want_parquets=True)
    pdb_path = target_dir / f"{target}_clean.pdb"
    if not pdb_path.exists():
        print(f"[{target}] no PDB in cache", flush=True)
        return None

    # Ground truth (for labels + perturbed NMA direction)
    gt_path = target_dir / f"{target}_ground_truth.json"
    ligand_centroid = None
    if gt_path.exists():
        try:
            gt = json.load(open(gt_path))
            if gt.get("valid_for_dcc_validation") and gt.get("ligand_centroid"):
                ligand_centroid = np.asarray(gt["ligand_centroid"], dtype=np.float32)
        except (json.JSONDecodeError, OSError):
            pass

    # Block 1 — Structural
    X_struct, residue_ids, residue_names, ca_coords = compute_structural(pdb_path)
    if len(residue_ids) == 0:
        print(f"[{target}] zero residues parsed", flush=True)
        return None

    # Block 2 + Perturbed NMA
    X_nma, X_perturb = compute_nma(pdb_path, ligand_centroid, ca_coords, residue_ids)

    # Block 3 — KCC
    X_kcc = compute_kcc(target_dir / f"{target}.kcc_visualization.json", residue_ids)

    # Block 4 + TIDE aux
    X_thermo, X_tide = compute_thermo_and_tide(
        target_dir / f"{target}.topology.prism_therm.json", residue_ids,
    )

    # Block 5 — Site-mapped
    X_site = compute_site_mapped(
        target_dir / f"{target}.binding_sites.json", residue_ids,
    )

    # Block 6 — Multi-phase
    X_phase = compute_phase_features(target_dir, ca_coords, residue_ids)

    # Block 7 — stubbed
    X_dyn = compute_dynamic_ensemble(target_dir, ca_coords)

    # ESM-2
    X_esm = compute_esm2(residue_names) if compute_esm else np.zeros(
        (len(residue_ids), DIM_ESM), dtype=np.float32)

    # Concat
    X = np.concatenate([X_struct, X_nma, X_perturb, X_kcc, X_thermo,
                        X_site, X_phase, X_dyn, X_tide, X_esm], axis=1)
    assert X.shape[1] == FEATURE_DIM, f"dim mismatch: got {X.shape[1]}, expected {FEATURE_DIM}"

    labels = compute_labels(ca_coords, ligand_centroid)

    missing = []
    if X_nma.sum() == 0:
        missing.append("nma")
    if X_kcc.sum() == 0:
        missing.append("kcc")
    if X_thermo.sum() == 0:
        missing.append("thermo")
    if X_site.sum() == 0:
        missing.append("site")
    if X_phase.sum() == 0:
        missing.append("phase")
    if X_esm.sum() == 0:
        missing.append("esm")

    return TargetBundle(
        target=target, X=X, residue_ids=np.asarray(residue_ids, dtype=np.int32),
        residue_names=residue_names, labels=labels,
        ligand_centroid=ligand_centroid if ligand_centroid is not None else np.zeros(3, dtype=np.float32),
        ca_coords=ca_coords, dim=FEATURE_DIM,
        block_offsets=BLOCK_OFFSETS, missing_blocks=missing,
    )


def save_bundle(bundle: TargetBundle, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    p = out_dir / f"{bundle.target}.features.npz"
    np.savez_compressed(
        p,
        X=bundle.X, residue_ids=bundle.residue_ids, labels=bundle.labels,
        ca_coords=bundle.ca_coords, ligand_centroid=bundle.ligand_centroid,
        residue_names=np.asarray(bundle.residue_names),
        missing_blocks=np.asarray(bundle.missing_blocks),
    )
    return p


def load_bundle(npz_path: Path) -> Dict[str, np.ndarray]:
    d = np.load(npz_path, allow_pickle=False)
    return {k: d[k] for k in d.files}


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("target")
    parser.add_argument("--cache-dir", default="/workspace/cache")
    parser.add_argument("--out-dir", default="/workspace/features")
    parser.add_argument("--no-esm", action="store_true", help="skip ESM-2 for quick smoke test")
    args = parser.parse_args()

    print(f"Extracting features for {args.target}")
    print(f"FEATURE_DIM = {FEATURE_DIM}")
    for block, (lo, hi) in BLOCK_OFFSETS.items():
        print(f"  {block:12s} [{lo:5d}:{hi:5d}] = {hi-lo:4d} dims")

    bundle = extract_target(args.target, Path(args.cache_dir),
                            compute_esm=not args.no_esm)
    if bundle is None:
        print("FAILED", file=sys.stderr)
        sys.exit(1)

    p = save_bundle(bundle, Path(args.out_dir))
    print(f"Saved {p}")
    print(f"  X shape: {bundle.X.shape}")
    print(f"  labels: {bundle.labels.sum()} positive / {len(bundle.labels)} total")
    print(f"  missing blocks: {bundle.missing_blocks}")
