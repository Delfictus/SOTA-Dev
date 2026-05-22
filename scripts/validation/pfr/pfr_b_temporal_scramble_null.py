#!/usr/bin/env python3
"""
PFR Step B — Temporal-Scramble Null Distribution Generator  (Arrow-first)
==========================================================================
EXECUTION CONTRACT: same as pfr_a — Arrow-first, vectorised numpy, no Python
dict loops over spike records, ArtifactContractViolation on missing artifacts.

Generates N_SCRAMBLES=1000 null pharmacophores per site by:
  1. Loading spike arrays from Arrow (same as Step A).
  2. Fixing cluster topology from Step-A reference pharmacophore.
  3. Per scramble: shuffle ccns_phase uint8 array (vectorised numpy) →
     recompute per-voxel type-weighted intensity → recompute cluster centroids.
     O(n_voxels) per scramble after the single O(n_spikes) Arrow load.

Output .npz per site:
  centroids     float32 [N_SCRAMBLES, max_clusters, 3]
  types_idx     int8    [N_SCRAMBLES, max_clusters]   (PHARMA_TYPES index)
  confidences   float32 [N_SCRAMBLES, max_clusters]
  n_features    int16   [N_SCRAMBLES]
  scramble_seeds uint64 [N_SCRAMBLES]
  ref_centroids  float32 [max_clusters, 3]            (Step-A reference)
  cluster_types  bytes array [max_clusters]            pharma type names
"""

import argparse
import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.ipc as ipc

# ── Paths ─────────────────────────────────────────────────────────────────────

PFR_BASE      = Path("/mnt/storage/prism-outputs/blind_validation")
PFR_OUT_BASE  = PFR_BASE / "pfr_validation"
PHARMA_DIR    = PFR_OUT_BASE / "pharmacophores"
MANIFEST_PATH = PFR_OUT_BASE / "sha256_manifest_pfr.txt"
SEED_LOG_PATH = PFR_OUT_BASE / "scramble_seed_log.txt"

TARGETS = {
    "B01_HRAS_Q61H":        {"apo_pdb": "4L9S"},
    "B02_CDK2_allosteric":  {"apo_pdb": "1HCL"},
    "B05_TP53_R175H":       {"apo_pdb": "2OCJ"},
    "B06_cGAS":             {"apo_pdb": "4KM5"},
    "B08_CRBN":             {"apo_pdb": "4TZ4_chainC"},
    "B09_Thrombin_exosite": {"apo_pdb": "1PPB"},
}

PHARMA_TYPES = ["AROMATIC", "HYDROPHOBIC", "HBOND_DONOR", "IONIC_NEG", "IONIC_POS", "COVALENT"]
PHARMA_IDX   = {t: i for i, t in enumerate(PHARMA_TYPES)}
N_PHARMA     = len(PHARMA_TYPES)

PHASE_WEIGHTS = np.array([1.0, 0.6, 0.4, 0.7, 0.9, 0.3, 0.5], dtype=np.float32)
GRID_SPACING  = 2.0
MERGE_RADIUS  = 5.0
MIN_VOX_FEAT  = 3
N_SCRAMBLES   = 1000

WL_BINS  = np.array([258.0, 274.0, 280.0, 211.0], dtype=np.float32)
WL_PTYPES = [
    [PHARMA_IDX["AROMATIC"], PHARMA_IDX["HYDROPHOBIC"]],
    [PHARMA_IDX["AROMATIC"], PHARMA_IDX["HBOND_DONOR"]],
    [PHARMA_IDX["AROMATIC"], PHARMA_IDX["HBOND_DONOR"]],
    [PHARMA_IDX["HBOND_DONOR"]],
]


# ── Contract ──────────────────────────────────────────────────────────────────

class ArtifactContractViolation(Exception):
    pass


def validate_and_load_site(run_dir: Path, apo_pdb: str, site_id: int) -> dict | None:
    """Load spike arrays for one site via Arrow. Contract-enforced."""
    arrow_path = run_dir / f"{apo_pdb}.topology.spike_events.arrow"
    if not arrow_path.exists():
        raise ArtifactContractViolation(f"Arrow missing: {arrow_path}")

    with open(arrow_path, "rb") as f:
        reader = ipc.open_file(f)
        table  = reader.read_all()

    sub = table.filter(pc.equal(table.column("site_id"), site_id))
    n   = len(sub)
    if n == 0:
        return None

    def _col(name, dtype):
        return np.array(sub.column(name), dtype=dtype)

    x         = _col("x",            np.float32)
    y         = _col("y",            np.float32)
    z         = _col("z",            np.float32)
    wl        = _col("wavelength_nm",np.float32)
    intensity = _col("intensity",    np.float32)
    phase_u8  = _col("ccns_phase",   np.uint8)

    return {"x": x, "y": y, "z": z, "wl": wl, "intensity": intensity,
            "phase_u8": phase_u8, "n": n}


# ── Pre-compute per-spike voxel mapping ────────────────────────────────────────

def spike_to_voxel_map(sa: dict) -> dict:
    """
    Map each spike to a voxel index; pre-compute per-pharmatype spike masks.
    Returns all arrays needed for fast scramble re-weighting.
    """
    x  = sa["x"];  y = sa["y"];  z = sa["z"]
    wl = sa["wl"]; n = sa["n"]

    vx = (x / GRID_SPACING).astype(np.int32)
    vy = (y / GRID_SPACING).astype(np.int32)
    vz = (z / GRID_SPACING).astype(np.int32)

    vx_off = vx - vx.min();  vy_off = vy - vy.min();  vz_off = vz - vz.min()
    Ny = int(vy_off.max()) + 1
    Nz = int(vz_off.max()) + 1
    linear_id = vx_off.astype(np.int64) * Ny * Nz + vy_off.astype(np.int64) * Nz + vz_off.astype(np.int64)
    _, inv = np.unique(linear_id, return_inverse=True)
    inv = inv.astype(np.int32)
    n_vox = int(inv.max()) + 1

    # Unweighted position per voxel
    cnt = np.bincount(inv, minlength=n_vox).astype(np.int32)
    cx  = np.bincount(inv, weights=x, minlength=n_vox) / np.maximum(cnt, 1)
    cy  = np.bincount(inv, weights=y, minlength=n_vox) / np.maximum(cnt, 1)
    cz  = np.bincount(inv, weights=z, minlength=n_vox) / np.maximum(cnt, 1)
    vox_pos = np.stack([cx, cy, cz], axis=1).astype(np.float32)

    # Per-pharmatype spike masks stored as index arrays
    type_spike_idx: list = []
    for pi in range(N_PHARMA):
        mask = np.zeros(n, dtype=bool)
        for wl_val, ptypes in zip(WL_BINS, WL_PTYPES):
            if pi in ptypes:
                mask |= np.abs(wl - wl_val) < 5.0
        # other → HYDROPHOBIC
        if pi == PHARMA_IDX["HYDROPHOBIC"]:
            other = np.ones(n, dtype=bool)
            for wl_val in WL_BINS:
                other &= np.abs(wl - wl_val) >= 5.0
            mask |= other
        type_spike_idx.append(np.where(mask)[0].astype(np.int32))

    return {
        "inv":             inv,           # [n_spikes] → voxel index
        "vox_pos":         vox_pos,       # [n_vox, 3]
        "n_vox":           n_vox,
        "type_spike_idx":  type_spike_idx,# [N_PHARMA] list of spike index arrays
        "intensity":       sa["intensity"],
        "phase_u8":        sa["phase_u8"],
        "n_spikes":        n,
    }


# ── Reference cluster topology (fixed across all scrambles) ───────────────────

def build_ref_clusters(vm: dict, ref_pharma: dict) -> list:
    """
    Build cluster topology from Step-A reference pharmacophore features.
    For each feature: find the nearest voxel indices within MERGE_RADIUS
    of the feature position to define the cluster's voxel membership.
    Returns list of cluster dicts.
    """
    vox_pos = vm["vox_pos"]
    clusters = []
    for feat in ref_pharma["features"]:
        ptype  = feat["type"]
        pi     = PHARMA_IDX.get(ptype, -1)
        if pi < 0:
            continue
        fpos = np.array(feat["position"], dtype=np.float32)
        dists = np.linalg.norm(vox_pos - fpos, axis=1)
        members = np.where(dists <= MERGE_RADIUS)[0].astype(np.int32)
        if len(members) < MIN_VOX_FEAT:
            # expand to nearest N voxels
            members = np.argsort(dists)[:max(MIN_VOX_FEAT, 5)].astype(np.int32)
        clusters.append({
            "ptype":       ptype,
            "pi":          pi,
            "vox_members": members,  # voxel indices in this cluster
            "ref_centroid": fpos,
        })
    return clusters


# ── Single scramble ───────────────────────────────────────────────────────────

def scramble_one(vm: dict, clusters: list, shuffled_phase: np.ndarray) -> tuple:
    """
    Compute cluster centroids under one shuffled-phase permutation.
    All operations vectorised via numpy bincount.
    Returns (types_idx, centroids, confidences, n_valid).
    """
    # Phase-weighted intensity for each spike
    pw = PHASE_WEIGHTS[np.clip(shuffled_phase, 0, len(PHASE_WEIGHTS) - 1)]
    wi = vm["intensity"] * pw                    # [n_spikes]
    inv = vm["inv"]                              # [n_spikes] → voxel idx
    n_vox = vm["n_vox"]

    # Per-voxel total weighted intensity
    vox_wi_total = np.bincount(inv, weights=wi, minlength=n_vox).astype(np.float32)
    total_wi = float(vox_wi_total.sum()) or 1.0

    # Per-type per-voxel weighted intensity (compute on demand per type)
    # Cache: indexed by pi
    type_vox_wi_cache: dict[int, np.ndarray] = {}

    max_c = len(clusters)
    types_idx   = np.full(max_c, -1, dtype=np.int8)
    centroids   = np.zeros((max_c, 3), dtype=np.float32)
    confidences = np.zeros(max_c, dtype=np.float32)

    for ci, clust in enumerate(clusters):
        pi       = clust["pi"]
        members  = clust["vox_members"]

        if pi not in type_vox_wi_cache:
            sidx = vm["type_spike_idx"][pi]
            tvw  = np.zeros(n_vox, dtype=np.float32)
            if len(sidx) > 0:
                np.add.at(tvw, inv[sidx], wi[sidx])
            type_vox_wi_cache[pi] = tvw

        tw = type_vox_wi_cache[pi][members]  # [n_members]
        tw_sum = float(tw.sum())
        if tw_sum < 1e-9:
            centroids[ci]   = clust["ref_centroid"]
            confidences[ci] = 0.0
        else:
            centroids[ci]   = (vm["vox_pos"][members] * tw[:, None]).sum(axis=0) / tw_sum
            confidences[ci] = tw_sum / total_wi

        types_idx[ci] = pi

    conf_sum = confidences.sum()
    if conf_sum > 1e-9:
        confidences /= conf_sum

    return types_idx, centroids, confidences, max_c


# ── Per-target processing ─────────────────────────────────────────────────────

def process_target(target: str, cfg: dict, timestamp_b: str,
                   out_dir: Path, manifest_lines: list, seed_lines: list) -> list:
    apo_pdb   = cfg["apo_pdb"]
    run_dir   = PFR_BASE / target / "run"
    pharma_dir = PHARMA_DIR / target
    tgt_out   = out_dir / target
    tgt_out.mkdir(parents=True, exist_ok=True)

    if not pharma_dir.exists():
        print(f"  SKIP {target}: no Step-A pharmacophores")
        return []

    base_seed = int(hashlib.sha256(target.encode()).hexdigest()[:16], 16) % (2 ** 63)
    written   = []

    for pharma_path in sorted(pharma_dir.glob("site*_pharmacophore.json")):
        ref_pharma = json.loads(pharma_path.read_text())
        site_id    = ref_pharma["site_id"]

        t0 = time.time()
        try:
            sa = validate_and_load_site(run_dir, apo_pdb, site_id)
        except ArtifactContractViolation as e:
            print(f"  site{site_id}: {e}")
            continue
        if sa is None or sa["n"] < 500:
            print(f"  site{site_id}: no spikes — skip")
            continue

        vm       = spike_to_voxel_map(sa)
        clusters = build_ref_clusters(vm, ref_pharma)
        if not clusters:
            print(f"  site{site_id}: no clusters — skip")
            continue

        n_c      = len(clusters)
        all_ti   = np.full((N_SCRAMBLES, n_c), -1, dtype=np.int8)
        all_cen  = np.zeros((N_SCRAMBLES, n_c, 3), dtype=np.float32)
        all_conf = np.zeros((N_SCRAMBLES, n_c), dtype=np.float32)
        all_seeds = np.zeros(N_SCRAMBLES, dtype=np.uint64)

        phase_orig = vm["phase_u8"].copy()
        for k in range(N_SCRAMBLES):
            seed_k = base_seed ^ (site_id * 100_000 + k)
            rng_k  = np.random.default_rng(seed_k)
            shuffled = phase_orig.copy()
            rng_k.shuffle(shuffled)
            ti, cen, conf, _ = scramble_one(vm, clusters, shuffled)
            all_ti[k]    = ti
            all_cen[k]   = cen
            all_conf[k]  = conf
            all_seeds[k] = seed_k

        elapsed = time.time() - t0
        out_path = tgt_out / f"site{site_id}_scramble_null.npz"
        np.savez_compressed(
            out_path,
            types_idx      = all_ti,
            centroids      = all_cen,
            confidences    = all_conf,
            n_features     = np.full(N_SCRAMBLES, n_c, dtype=np.int16),
            scramble_seeds = all_seeds,
            ref_centroids  = np.array([c["ref_centroid"] for c in clusters]),
            cluster_types  = np.array([c["ptype"] for c in clusters]),
            site_centroid  = np.array(ref_pharma["site_centroid"], dtype=np.float32),
            timestamp_b    = np.bytes_(timestamp_b),
        )
        digest = sha256_file(out_path)
        manifest_lines.append(f"{digest}  {out_path}  TIMESTAMP_B={timestamp_b}")
        seed_lines.append(
            f"target={target} site={site_id} n={N_SCRAMBLES} base_seed={base_seed} ts={timestamp_b}"
        )
        print(f"  site{site_id:5d}: {N_SCRAMBLES} scrambles  "
              f"{n_c} clusters  {elapsed:.1f}s  sha256={digest[:12]}…")
        written.append(str(out_path))
    return written


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="PFR Step B: temporal-scramble null (Arrow-first)")
    ap.add_argument("--targets", nargs="*", default=list(TARGETS.keys()))
    ap.add_argument("--out-dir", default=str(PFR_OUT_BASE / "scramble_nulls"))
    args = ap.parse_args()

    timestamp_b = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not PHARMA_DIR.exists() or not list(PHARMA_DIR.glob("*/*_pharmacophore.json")):
        print("ERROR: no Step-A pharmacophores found — run pfr_a first")
        raise SystemExit(1)

    print("=" * 70)
    print("PFR Step B — Temporal-Scramble Null Distribution (Arrow-first)")
    print("=" * 70)
    print(f"TIMESTAMP_B = {timestamp_b}  N_SCRAMBLES={N_SCRAMBLES}")

    manifest_lines = [
        "", f"# ── PFR Step B — TIMESTAMP_B={timestamp_b} ──────────────────────",
    ]
    seed_lines = [f"# TIMESTAMP_B={timestamp_b}"]

    all_written = []
    for target in args.targets:
        if target not in TARGETS:
            continue
        print(f"\n[{target}]")
        written = process_target(
            target, TARGETS[target], timestamp_b, out_dir,
            manifest_lines, seed_lines
        )
        all_written.extend(written)

    with open(MANIFEST_PATH, "a") as fh:
        fh.write("\n".join(manifest_lines) + "\n")

    seed_text   = "\n".join(seed_lines)
    seed_digest = hashlib.sha256(seed_text.encode()).hexdigest()
    SEED_LOG_PATH.write_text(f"# seed_log sha256={seed_digest}\n" + seed_text + "\n")
    with open(MANIFEST_PATH, "a") as fh:
        fh.write(f"{seed_digest}  {SEED_LOG_PATH}  TIMESTAMP_B={timestamp_b}\n")

    print(f"\nStep B complete — {len(all_written)} .npz files  TIMESTAMP_B={timestamp_b}")


if __name__ == "__main__":
    main()
