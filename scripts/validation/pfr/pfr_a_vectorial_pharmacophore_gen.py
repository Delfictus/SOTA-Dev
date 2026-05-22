#!/usr/bin/env python3
"""
PFR Step A — Vectorial Pharmacophore Generator  (Arrow-first, contract-enforcing)
==================================================================================
EXECUTION CONTRACT:
  • ArtifactContractViolation is raised and execution halts if any canonical
    artifact is absent.  No silent JSON-dict fallbacks.
  • All spike data is loaded via PyArrow → numpy.  Zero Python dict iteration
    over spike records.
  • Per-site data is filtered via `site_id` column (Arrow exclusive assignment).
    No redundant spatial filtering loops.
  • Phase decode: from `timestep` + protocol extracted from run.log when
    run_metadata.json is absent.  wavelength_nm used directly for pharmacophore
    type (no aromatic_type enum decode required).

Vectorised voxelisation: numpy bincount/ufunc aggregation — no Python voxel loops.
"""

import argparse
import hashlib
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.ipc as ipc

# ── Paths ─────────────────────────────────────────────────────────────────────

PFR_BASE     = Path("/mnt/storage/prism-outputs/blind_validation")
PFR_OUT_BASE = PFR_BASE / "pfr_validation"
MANIFEST_PATH = PFR_OUT_BASE / "sha256_manifest_pfr.txt"

TARGETS = {
    "B01_HRAS_Q61H":        {"apo_pdb": "4L9S"},
    "B02_CDK2_allosteric":  {"apo_pdb": "1HCL"},
    "B05_TP53_R175H":       {"apo_pdb": "2OCJ"},
    "B06_cGAS":             {"apo_pdb": "4KM5"},
    "B08_CRBN":             {"apo_pdb": "4TZ4_chainC"},
    "B09_Thrombin_exosite": {"apo_pdb": "1PPB"},
}

# ── Contract ──────────────────────────────────────────────────────────────────

class ArtifactContractViolation(Exception):
    """Raised when a required canonical artifact is absent or unreadable."""


def validate_canonical_artifacts(run_dir: Path, frozen_dir: Path, stem: str) -> dict:
    """
    Enforce PRISM4D artifact contract before any processing.
    Raises ArtifactContractViolation on any missing artifact.
    Returns a dict of verified canonical paths + protocol params.
    """
    arrow_path = run_dir / f"{stem}.topology.spike_events.arrow"
    bs_path    = frozen_dir / f"{stem}.binding_sites.json"
    log_path   = run_dir / "run.log"
    meta_path  = run_dir / f"{stem}.run_metadata.json"

    errors = []
    if not arrow_path.exists():
        errors.append(f"Arrow file missing: {arrow_path}")
    if not bs_path.exists():
        errors.append(f"binding_sites.json missing: {bs_path}")
    if not log_path.exists() and not meta_path.exists():
        errors.append(
            f"Neither run_metadata.json nor run.log present in {run_dir}"
        )
    if errors:
        raise ArtifactContractViolation(
            "\n[FATAL] ARTIFACT_CONTRACT_VIOLATION:\n" +
            "\n".join(f"  {e}" for e in errors) +
            "\nPipeline state NON-CANONICAL. Processing halted.\n"
            "Implicit JSON fallback strictly forbidden."
        )

    # Extract CCNS phase protocol
    protocol = _extract_protocol(meta_path if meta_path.exists() else None,
                                  log_path if log_path.exists() else None)

    print(f"  [CONTRACT PASS] arrow={arrow_path.name}  bs={bs_path.name}")
    print(f"  [PROTOCOL] cold_hold={protocol['cold_hold_steps']}  "
          f"ramp={protocol['ramp_steps']}  warm_hold={protocol['warm_hold_steps']}")
    return {
        "arrow_path": arrow_path,
        "bs_path":    bs_path,
        "protocol":   protocol,
    }


def _extract_protocol(meta_path: Path | None, log_path: Path | None) -> dict:
    """
    Extract CCNS phase step counts.
    Priority: run_metadata.json > run.log regex.
    Raises ArtifactContractViolation if neither yields valid protocol.
    """
    if meta_path and meta_path.exists():
        meta = json.loads(meta_path.read_text())
        proto = meta.get("reference_protocol_for_json_phase_label") or {}
        if proto and "cold_hold_steps" in proto:
            return {
                "cold_hold_steps": int(proto["cold_hold_steps"]),
                "ramp_steps":      int(proto.get("ramp_steps", 0)),
                "warm_hold_steps": int(proto.get("warm_hold_steps", 0)),
                "ramp_down_steps": int(proto.get("ramp_down_steps", 0)),
                "source": "run_metadata.json",
            }

    if log_path and log_path.exists():
        # Example line: Phases: cold_hold=14000, ramp=6000, warm_hold=15000
        text = log_path.read_text(errors="replace")
        m = re.search(
            r"Phases:\s*cold_hold=(\d+),\s*ramp=(\d+),\s*warm_hold=(\d+)",
            text
        )
        if m:
            return {
                "cold_hold_steps": int(m.group(1)),
                "ramp_steps":      int(m.group(2)),
                "warm_hold_steps": int(m.group(3)),
                "ramp_down_steps": 0,
                "source": "run.log",
            }

    raise ArtifactContractViolation(
        "[FATAL] Cannot extract CCNS phase protocol from run_metadata.json or run.log"
    )


# ── Pharmacophore type from wavelength (no enum decode required) ──────────────
# Wavelength stored as float in Arrow — directly encodes the physical mechanism.

def wavelength_to_pharma_types(wl: float) -> list[int]:
    """
    Map excitation wavelength → pharmacophore type indices.
    Returns list of indices into PHARMA_TYPES.
    """
    if abs(wl - 258.0) < 2:   # PHE/BNZ: pure hydrophobic aromatic
        return [PHARMA_IDX["AROMATIC"], PHARMA_IDX["HYDROPHOBIC"]]
    if abs(wl - 274.0) < 2:   # TYR: H-bond donor + aromatic
        return [PHARMA_IDX["AROMATIC"], PHARMA_IDX["HBOND_DONOR"]]
    if abs(wl - 280.0) < 2:   # TRP: large aromatic + H-bond donor
        return [PHARMA_IDX["AROMATIC"], PHARMA_IDX["HBOND_DONOR"]]
    if abs(wl - 211.0) < 5:   # backbone n→π*: H-bond acceptor
        return [PHARMA_IDX["HBOND_DONOR"]]
    return [PHARMA_IDX["HYDROPHOBIC"]]  # LIF/UNK thermal


PHARMA_TYPES = ["AROMATIC", "HYDROPHOBIC", "HBOND_DONOR", "IONIC_NEG", "IONIC_POS", "COVALENT"]
PHARMA_IDX   = {t: i for i, t in enumerate(PHARMA_TYPES)}
N_PHARMA     = len(PHARMA_TYPES)

# Phase-weight lookup indexed by ccns_phase uint8
# 0=cold_hold, 1=heating/ramp, 2=warm_hold, 3=cooling, 4=cold_return
PHASE_WEIGHTS_BY_IDX = np.array([1.0, 0.6, 0.4, 0.7, 0.9, 0.3, 0.5], dtype=np.float32)

GRID_SPACING   = 2.0   # Å
MERGE_RADIUS   = 5.0   # Å
MIN_VOX_FEAT   = 3


# ── Vectorised Arrow loader ───────────────────────────────────────────────────

def load_site_arrays(artifacts: dict, site_id: int) -> dict | None:
    """
    Load spike data for one site_id from the Arrow file via vectorised filter.
    Returns numpy arrays: x, y, z, intensity, wavelength_nm, phase_weight.
    No Python dict iteration over individual spikes.
    """
    arrow_path = artifacts["arrow_path"]
    protocol   = artifacts["protocol"]

    with open(arrow_path, "rb") as f:
        reader = ipc.open_file(f)
        table  = reader.read_all()

    # Filter by site_id (vectorised Arrow compute)
    mask  = pc.equal(table.column("site_id"), site_id)
    sub   = table.filter(mask)
    n     = len(sub)
    if n == 0:
        return None

    def _col(name, dtype):
        return np.array(sub.column(name), dtype=dtype)

    x   = _col("x", np.float32)
    y   = _col("y", np.float32)
    z   = _col("z", np.float32)
    wl  = _col("wavelength_nm", np.float32)
    intensity = _col("intensity", np.float32)

    # Phase decode: ccns_phase uint8 → weight
    phase_u8 = _col("ccns_phase", np.uint8)
    # Clamp index to weight table length
    phase_idx = np.clip(phase_u8, 0, len(PHASE_WEIGHTS_BY_IDX) - 1)
    phase_w   = PHASE_WEIGHTS_BY_IDX[phase_idx]
    is_cold   = (phase_u8 == 0).astype(np.float32)   # cold_hold = 0

    weighted_intensity = intensity * phase_w
    return {
        "x": x, "y": y, "z": z,
        "wavelength_nm": wl,
        "weighted_intensity": weighted_intensity,
        "is_cold": is_cold,
        "n_spikes": n,
    }


# ── Vectorised voxelisation ───────────────────────────────────────────────────

def voxelize_numpy(sa: dict) -> dict:
    """
    Vectorised voxelisation: numpy integer-bin aggregation.
    No Python loops over individual spikes.

    Returns dict of voxel data arrays (vox_* prefix), all numpy arrays.
    """
    x  = sa["x"];  y  = sa["y"];  z  = sa["z"]
    wi = sa["weighted_intensity"]
    wl = sa["wavelength_nm"]
    ic = sa["is_cold"]

    # Voxel indices
    vx = (x / GRID_SPACING).astype(np.int32)
    vy = (y / GRID_SPACING).astype(np.int32)
    vz = (z / GRID_SPACING).astype(np.int32)

    # Pack (vx, vy, vz) → unique voxel id using a linear hash
    # Shift to positive range then linearise
    vx_off = vx - vx.min();  vy_off = vy - vy.min();  vz_off = vz - vz.min()
    Ny = int(vy_off.max()) + 1
    Nz = int(vz_off.max()) + 1
    linear_id = vx_off.astype(np.int64) * Ny * Nz + vy_off.astype(np.int64) * Nz + vz_off.astype(np.int64)
    unique_ids, inv = np.unique(linear_id, return_inverse=True)
    n_vox = len(unique_ids)

    # Weighted position centroid per voxel (intensity-weighted mean x/y/z)
    wi_sum  = np.bincount(inv, weights=wi,      minlength=n_vox).astype(np.float64)
    wx_sum  = np.bincount(inv, weights=x * wi,  minlength=n_vox).astype(np.float64)
    wy_sum  = np.bincount(inv, weights=y * wi,  minlength=n_vox).astype(np.float64)
    wz_sum  = np.bincount(inv, weights=z * wi,  minlength=n_vox).astype(np.float64)
    cold_sum= np.bincount(inv, weights=ic,       minlength=n_vox).astype(np.float32)
    cnt_sum = np.bincount(inv,                   minlength=n_vox).astype(np.int32)

    safe_wi = np.where(wi_sum > 0, wi_sum, 1.0)
    vox_pos = np.stack([wx_sum / safe_wi, wy_sum / safe_wi, wz_sum / safe_wi], axis=1).astype(np.float32)

    # Per-voxel per-pharmatype weighted intensity: [n_vox, N_PHARMA]
    vox_type_wi = np.zeros((n_vox, N_PHARMA), dtype=np.float32)

    # Map wavelength bins to pharma type masks (vectorised)
    WL_BINS = [258.0, 274.0, 280.0, 211.0]
    WL_TYPES = [
        [PHARMA_IDX["AROMATIC"], PHARMA_IDX["HYDROPHOBIC"]],  # 258
        [PHARMA_IDX["AROMATIC"], PHARMA_IDX["HBOND_DONOR"]],  # 274
        [PHARMA_IDX["AROMATIC"], PHARMA_IDX["HBOND_DONOR"]],  # 280
        [PHARMA_IDX["HBOND_DONOR"]],                           # 211
    ]
    for wl_val, ptypes in zip(WL_BINS, WL_TYPES):
        mask = np.abs(wl - wl_val) < 5.0
        if not mask.any():
            continue
        m_wi = wi * mask.astype(np.float32)
        for pt in ptypes:
            vox_type_wi[:, pt] += np.bincount(inv, weights=m_wi, minlength=n_vox).astype(np.float32)
    # Remaining spikes → HYDROPHOBIC
    other_mask = np.ones(len(wl), dtype=bool)
    for wl_val in WL_BINS:
        other_mask &= (np.abs(wl - wl_val) >= 5.0)
    if other_mask.any():
        vox_type_wi[:, PHARMA_IDX["HYDROPHOBIC"]] += np.bincount(
            inv, weights=wi * other_mask, minlength=n_vox).astype(np.float32)

    return {
        "vox_pos":     vox_pos,       # [n_vox, 3]
        "vox_wi":      wi_sum.astype(np.float32),   # [n_vox]
        "vox_type_wi": vox_type_wi,   # [n_vox, N_PHARMA]
        "vox_cold":    cold_sum,       # [n_vox]
        "vox_cnt":     cnt_sum,        # [n_vox]
        "n_vox":       n_vox,
        "total_wi":    float(wi_sum.sum()),
    }


# ── Spatial clustering (vectorised greedy) ────────────────────────────────────

def greedy_cluster(positions: np.ndarray, weights: np.ndarray,
                   merge_radius: float) -> list[np.ndarray]:
    """Greedy cluster: seed at highest weight, absorb neighbours within merge_radius."""
    assigned = np.zeros(len(positions), dtype=bool)
    clusters = []
    for seed in np.argsort(-weights):
        if assigned[seed]:
            continue
        d = np.linalg.norm(positions - positions[seed], axis=1)
        members = np.where((d <= merge_radius) & ~assigned)[0]
        assigned[members] = True
        clusters.append(members)
    return clusters


# ── PCA direction vector ──────────────────────────────────────────────────────

def pca_direction(positions: np.ndarray, weights: np.ndarray,
                  site_centroid: np.ndarray) -> np.ndarray:
    """Weighted PCA first component, oriented outward from site_centroid."""
    if len(positions) < 3:
        v = positions.mean(axis=0) - site_centroid
        n = np.linalg.norm(v)
        return v / n if n > 1e-9 else np.array([0.0, 0.0, 1.0])
    w = weights / (weights.sum() + 1e-9)
    mu = (positions * w[:, None]).sum(axis=0)
    centered = positions - mu
    cov = np.cov(centered.T, aweights=w + 1e-12)
    try:
        _, evecs = np.linalg.eigh(cov)
        d = evecs[:, -1].astype(np.float64)
    except np.linalg.LinAlgError:
        d = np.array([0.0, 0.0, 1.0])
    if np.dot(d, mu - site_centroid) < 0:
        d = -d
    n = np.linalg.norm(d)
    return d / n if n > 1e-9 else d


# ── Feature extraction ────────────────────────────────────────────────────────

def extract_features(vox: dict, site_centroid: np.ndarray) -> list[dict]:
    """Extract vectorial pharmacophore features from voxel arrays."""
    pos      = vox["vox_pos"]           # [n_vox, 3]
    type_wi  = vox["vox_type_wi"]       # [n_vox, N_PHARMA]
    cold     = vox["vox_cold"]          # [n_vox]
    cnt      = vox["vox_cnt"]           # [n_vox]
    total_wi = vox["total_wi"]
    sc       = site_centroid.astype(np.float64)

    features = []
    for pi, ptype in enumerate(PHARMA_TYPES):
        tw = type_wi[:, pi]
        nonzero = tw > 0
        if nonzero.sum() < MIN_VOX_FEAT:
            continue
        t_pos = pos[nonzero]
        t_wi  = tw[nonzero]

        # Threshold at median of nonzero type_wi
        thresh = np.median(t_wi)
        sig    = t_wi >= thresh
        if sig.sum() < MIN_VOX_FEAT:
            sig = np.ones(len(t_wi), dtype=bool)
        t_pos = t_pos[sig];  t_wi = t_wi[sig]

        clusters = greedy_cluster(t_pos, t_wi, MERGE_RADIUS)
        for idx_arr in clusters:
            if len(idx_arr) < MIN_VOX_FEAT:
                continue
            c_pos = t_pos[idx_arr];  c_wi = t_wi[idx_arr]
            w_sum = float(c_wi.sum())
            mu    = (c_pos * c_wi[:, None]).sum(axis=0) / w_sum
            direc = pca_direction(c_pos.astype(np.float64),
                                   c_wi.astype(np.float64), sc)

            # Phase bias: ratio of cold_hold weighted spikes in the cluster voxels
            # Re-index cold by the sig mask → approximate from original nonzero idx
            nonzero_idx = np.where(nonzero)[0]
            sig_global  = nonzero_idx[np.where(sig)[0]][idx_arr]
            cold_sum = float(cold[sig_global].sum())
            cnt_sum  = float(cnt[sig_global].sum())
            phase_bias = cold_sum / (cnt_sum + 1e-9)

            features.append({
                "type":            ptype,
                "position":        mu.tolist(),
                "direction":       direc.tolist(),
                "confidence":      w_sum / (total_wi + 1e-9),
                "phase_bias_cold": phase_bias,
                "n_voxels":        len(idx_arr),
            })

    # Normalise confidence
    total_conf = sum(f["confidence"] for f in features) or 1.0
    for f in features:
        f["confidence"] /= total_conf

    features.sort(key=lambda f: -f["confidence"])
    return features


# ── Utility ───────────────────────────────────────────────────────────────────

def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_binding_sites(bs_path: Path) -> list:
    d = json.loads(bs_path.read_text())
    return d.get("sites", d) if isinstance(d, dict) else d


# ── Per-target processing ─────────────────────────────────────────────────────

def process_target(target: str, cfg: dict, timestamp_a: str,
                   out_dir: Path, manifest_lines: list) -> list:
    apo_pdb    = cfg["apo_pdb"]
    run_dir    = PFR_BASE / target / "run"
    frozen_dir = PFR_BASE / target / "frozen"

    # Contract enforcement — raises ArtifactContractViolation on failure
    artifacts = validate_canonical_artifacts(run_dir, frozen_dir, apo_pdb)
    binding_sites = load_binding_sites(artifacts["bs_path"])

    tgt_out = out_dir / target
    tgt_out.mkdir(parents=True, exist_ok=True)
    written = []

    for site in binding_sites:
        site_id = site["id"]
        sa = load_site_arrays(artifacts, site_id)
        if sa is None or sa["n_spikes"] < 500:
            print(f"  site{site_id}: no Arrow data — skip")
            continue

        vox = voxelize_numpy(sa)
        if vox["n_vox"] < MIN_VOX_FEAT:
            print(f"  site{site_id}: too few voxels — skip")
            continue

        sc       = np.array(site["centroid"], dtype=np.float32)
        features = extract_features(vox, sc)

        pharma = {
            "target":        target,
            "site_id":       site_id,
            "site_centroid": site["centroid"],
            "therm_class":   site.get("therm_class", "UNKNOWN"),
            "quality_score": site.get("quality_score", 0.0),
            "rank_K":        site.get("rank_K"),
            "n_site_spikes": sa["n_spikes"],
            "n_voxels":      vox["n_vox"],
            "features":      features,
            "n_features":    len(features),
            "timestamp_a":   timestamp_a,
            "apo_pdb":       apo_pdb,
            "protocol_source": artifacts["protocol"]["source"],
        }

        out_path = tgt_out / f"site{site_id}_pharmacophore.json"
        out_path.write_text(json.dumps(pharma, indent=2))
        digest = sha256_file(out_path)
        manifest_lines.append(f"{digest}  {out_path}  TIMESTAMP_A={timestamp_a}")

        feat_summary = ", ".join(
            f"{f['type']}({f['confidence']:.3f})" for f in features[:4]
        )
        print(f"  site{site_id:5d}: {len(features):2d} feats  "
              f"{site.get('therm_class','?'):12s}  n={sa['n_spikes']:>7,}  "
              f"top={feat_summary}  sha256={digest[:12]}…")
        written.append(str(out_path))

    return written


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description="PFR Step A: vectorial pharmacophore generation (Arrow-first)"
    )
    ap.add_argument("--targets", nargs="*", default=list(TARGETS.keys()))
    ap.add_argument("--out-dir", default=str(PFR_OUT_BASE / "pharmacophores"))
    args = ap.parse_args()

    timestamp_a = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    PFR_OUT_BASE.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("PFR Step A — Vectorial Pharmacophore Generator (Arrow-first)")
    print("=" * 70)
    print(f"TIMESTAMP_A = {timestamp_a}")
    print()

    manifest_lines = [
        "",
        f"# ── PFR Step A — TIMESTAMP_A={timestamp_a} ──────────────────────",
        f"# Arrow-first vectorised load.  Zero JSON spike fallback.",
    ]

    all_written = []
    for target in args.targets:
        if target not in TARGETS:
            print(f"  [WARN] unknown target '{target}'")
            continue
        print(f"\n[{target}]")
        try:
            written = process_target(
                target, TARGETS[target], timestamp_a, out_dir, manifest_lines
            )
            all_written.extend(written)
        except ArtifactContractViolation as e:
            print(str(e))
            print(f"  [HALTED] {target} — contract violation")

    with open(MANIFEST_PATH, "a") as fh:
        fh.write("\n".join(manifest_lines) + "\n")

    print()
    print("=" * 70)
    print(f"Step A complete — {len(all_written)} pharmacophore JSONs written")
    print(f"TIMESTAMP_A = {timestamp_a}")
    if all_written:
        d0 = sha256_file(Path(all_written[0]))
        print(f"First output sha256: {d0[:32]}…  {all_written[0]}")


if __name__ == "__main__":
    main()
