#!/usr/bin/env python3
"""v002 neuromorphic ensemble feature extractor — 108 per-residue channels.

Per-residue 108-dim vector saved as `channel_features` key in each
features-pct95/*_features.npz. Does NOT touch any v001 artifact — the
existing physics_216, tide_residue, temporal, esm2 keys are preserved.

Tier layout:
    Tier 1  — 3 sources × 10 feats  = 30
    Tier 2  — 5 phases  ×  6 feats  = 30
    Tier 3  — 4 continuous cols × 6 stats = 24
    Tier 4  — 12 cross-channel features
    Tier 5  —  6 cooperative-excitation features
    Tier 6  —  6 intensity-distribution features
    ─────────────────────────────────────
    TOTAL                            108

Parquet columns used: x, y, z, intensity, spike_source, ccns_phase,
timestep, stream_id, vibrational_energy, water_density, n_nearby_excited.
Arrow-only columns (background_class, group_id, phase_bits, etc.) are
skipped per the current directive.

Usage:
    python3 scripts/training/extract_channel_features.py \\
        --bundle-dir /mnt/storage/spike-audit/features-pct95 \\
        --r2-prefix r2:prism-archive/10k-runs \\
        --stage-dir /mnt/storage/spike-audit/r2stage \\
        --workers 8 --bwlimit 20M
"""
from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import subprocess
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

# ─────────────────────────────────────────────────────────────
#  Constants
# ─────────────────────────────────────────────────────────────

PHASES = ["cold_hold", "heating", "warm_hold", "cooling", "cold_return"]
PHASE_IDX = {p: i for i, p in enumerate(PHASES)}
N_PHASES = 5

# Source encoding — parquets may store string names or numeric codes.
# Per directive: 3 spike sources = UV, LIF, EFP.  Engine codes:
#   0 = LIF, 1 = EFP, 2 = UV   (values may also appear as strings)
# LADD (4) and COFIRE (5) are *events*, not channels — skipped here.
SOURCE_MAP = {
    "LIF": 0, "EFP": 1, "UV": 2,
    "0": 0, "1": 1, "2": 2,
    0: 0, 1: 1, 2: 2,
}
SOURCE_NAMES = ["LIF", "EFP", "UV"]  # column ordering for Tier 1
N_SOURCES = 3

NEAR_R = 5.0      # Å sphere around Cα
UNSAT_THR = 64.0  # intensity threshold for unsat_frac
N_TIME_BINS = 20
FEATURE_DIM = 108

TIER_OFFSETS = {
    "source":      (0,   30),   # 3 × 10
    "phase":       (30,  60),   # 5 × 6
    "continuous":  (60,  84),   # 4 × 6
    "cross":       (84,  96),   # 12
    "cooperative": (96,  102),  # 6
    "intensity":   (102, 108),  # 6
}


# ─────────────────────────────────────────────────────────────
#  R2 staging
# ─────────────────────────────────────────────────────────────

def rclone_parquets(r2_target_prefix: str, dst_dir: Path,
                    bwlimit: str) -> bool:
    dst_dir.mkdir(parents=True, exist_ok=True)
    cmd = ["rclone", "copy", r2_target_prefix, str(dst_dir),
           "--include", "*.spike_events.parquet",
           "--transfers", "8", "--quiet"]
    if bwlimit:
        cmd += ["--bwlimit", bwlimit]
    r = subprocess.run(cmd, capture_output=True, timeout=600, check=False)
    return r.returncode == 0


# ─────────────────────────────────────────────────────────────
#  Parquet loader
# ─────────────────────────────────────────────────────────────

WANT_COLS = ["x", "y", "z", "intensity", "spike_source", "ccns_phase",
             "timestep", "stream_id", "vibrational_energy",
             "water_density", "n_nearby_excited"]


def load_spikes(target_dir: Path) -> Optional[Dict[str, np.ndarray]]:
    """Read all target.site*.spike_events.parquet files, concat to arrays."""
    try:
        import pyarrow.parquet as pq
    except ImportError:
        return None

    parquets = sorted(target_dir.glob("*.spike_events.parquet"))
    if not parquets:
        return None

    chunks: Dict[str, List[np.ndarray]] = {c: [] for c in WANT_COLS}
    for pf in parquets:
        try:
            t = pq.read_table(pf, columns=[c for c in WANT_COLS])
        except Exception:
            # Fall back: read whatever's present
            try:
                t = pq.read_table(pf)
                # Filter to WANT_COLS that exist
                present = [c for c in WANT_COLS if c in t.schema.names]
                t = t.select(present)
            except Exception:
                continue
        for col in WANT_COLS:
            if col not in t.schema.names:
                chunks[col].append(np.zeros(t.num_rows, dtype=np.float32))
            else:
                arr = t.column(col).to_pandas().values
                chunks[col].append(arr)

    if not chunks["x"]:
        return None

    out: Dict[str, np.ndarray] = {}
    for col in WANT_COLS:
        arrs = chunks[col]
        if not arrs:
            continue
        cat = np.concatenate(arrs)
        if col in ("spike_source", "ccns_phase"):
            out[col] = np.asarray(cat, dtype=object)
        elif col in ("timestep", "stream_id"):
            out[col] = cat.astype(np.int64, copy=False)
        else:
            out[col] = cat.astype(np.float32, copy=False)
    return out


# ─────────────────────────────────────────────────────────────
#  Per-spike indexing  (source/phase code arrays)
# ─────────────────────────────────────────────────────────────

def encode_sources(raw) -> np.ndarray:
    """Map raw spike_source values → {0,1,2,-1}. -1 = unknown/LADD/COFIRE."""
    out = np.full(len(raw), -1, dtype=np.int8)
    for i, v in enumerate(raw):
        out[i] = SOURCE_MAP.get(v, SOURCE_MAP.get(str(v), -1))
    return out


def encode_phases(raw) -> np.ndarray:
    out = np.full(len(raw), -1, dtype=np.int8)
    for i, v in enumerate(raw):
        out[i] = PHASE_IDX.get(str(v), -1)
    return out


# ─────────────────────────────────────────────────────────────
#  Per-residue neighborhood (5Å Cα)
# ─────────────────────────────────────────────────────────────

def nearest_ca(spike_xyz: np.ndarray, ca_coords: np.ndarray) -> np.ndarray:
    """Each spike's index of nearest Cα within NEAR_R, else -1."""
    from scipy.spatial import cKDTree
    tree = cKDTree(ca_coords)
    dist, idx = tree.query(spike_xyz, k=1, distance_upper_bound=NEAR_R)
    out = idx.astype(np.int32)
    out[~np.isfinite(dist)] = -1
    out[dist >= NEAR_R] = -1
    return out


# ─────────────────────────────────────────────────────────────
#  Statistics helpers
# ─────────────────────────────────────────────────────────────

def _safe_mean(x: np.ndarray) -> float:
    return float(x.mean()) if x.size else 0.0


def _safe_std(x: np.ndarray) -> float:
    return float(x.std()) if x.size > 1 else 0.0


def _skewness(x: np.ndarray) -> float:
    if x.size < 3:
        return 0.0
    m = x.mean(); s = x.std()
    if s < 1e-9:
        return 0.0
    return float(((x - m) ** 3).mean() / (s ** 3))


def _pct(x: np.ndarray, p: float) -> float:
    return float(np.percentile(x, p)) if x.size else 0.0


def _shannon(counts: np.ndarray) -> float:
    tot = counts.sum()
    if tot <= 0:
        return 0.0
    p = counts / tot
    p = p[p > 0]
    return float(-(p * np.log(p)).sum())


# ─────────────────────────────────────────────────────────────
#  Per-residue 108-dim feature vector
# ─────────────────────────────────────────────────────────────

def compute_per_residue(ca_coords: np.ndarray, spikes: Dict[str, np.ndarray]
                         ) -> np.ndarray:
    N = ca_coords.shape[0]
    out = np.zeros((N, FEATURE_DIM), dtype=np.float32)
    if not spikes or not len(spikes.get("x", [])):
        return out

    # Stack xyz → assign each spike to nearest Cα within 5 Å
    xyz = np.column_stack([spikes["x"], spikes["y"], spikes["z"]]).astype(np.float32)
    assign = nearest_ca(xyz, ca_coords)
    valid = assign >= 0
    if not valid.any():
        return out

    # Index arrays for source/phase
    src = encode_sources(spikes["spike_source"])
    ph = encode_phases(spikes["ccns_phase"])
    inten = spikes["intensity"]
    vib = spikes["vibrational_energy"]
    wd = spikes["water_density"]
    nex = spikes["n_nearby_excited"]
    ts = spikes["timestep"]
    strm = spikes["stream_id"]

    # Residue-grouped spike indices
    idx_by_res: List[np.ndarray] = [np.array([], dtype=np.int64)] * N
    # Efficient groupby via np.argsort
    order = np.argsort(assign[valid], kind="stable")
    valid_spike_idx = np.nonzero(valid)[0][order]
    valid_res = assign[valid_spike_idx]
    # Split points
    split_at = np.nonzero(np.diff(valid_res))[0] + 1
    groups = np.split(valid_spike_idx, split_at)
    for g in groups:
        if g.size == 0:
            continue
        r = int(assign[g[0]])
        if 0 <= r < N:
            idx_by_res[r] = g

    # Timestep range for persistence binning (global)
    if valid.any():
        ts_min_global = int(ts[valid].min())
        ts_max_global = int(ts[valid].max())
    else:
        ts_min_global, ts_max_global = 0, 1
    ts_range_global = max(ts_max_global - ts_min_global, 1)
    time_edges = np.linspace(ts_min_global, ts_max_global + 1,
                              N_TIME_BINS + 1)

    # Compute per-residue blocks
    for ri in range(N):
        g = idx_by_res[ri]
        if g.size == 0:
            continue

        g_src = src[g]
        g_ph = ph[g]
        g_inten = inten[g]
        g_vib = vib[g]
        g_wd = wd[g]
        g_nex = nex[g]
        g_ts = ts[g]
        g_strm = strm[g]
        g_xyz = xyz[g]

        # ─── Tier 1 — per-source (3 × 10 = 30) ───
        for si, sname in enumerate(SOURCE_NAMES):
            sel = (g_src == si)
            n = int(sel.sum())
            base = si * 10
            if n == 0:
                continue
            out[ri, base + 0] = n
            out[ri, base + 1] = _safe_mean(g_inten[sel])
            out[ri, base + 2] = float((g_inten[sel] < UNSAT_THR).mean())
            # phase_transition_ratio (source-specific)
            warm_sel = sel & (g_ph == PHASE_IDX["warm_hold"])
            cold_sel = sel & (g_ph == PHASE_IDX["cold_hold"])
            n_warm = int(warm_sel.sum()); n_cold = int(cold_sel.sum())
            out[ri, base + 3] = n_warm / max(n_cold, 1)
            out[ri, base + 4] = n_warm / max(n, 1)
            out[ri, base + 5] = int(np.unique(g_strm[sel]).size)
            # temporal_persistence
            hist, _ = np.histogram(g_ts[sel], bins=time_edges)
            out[ri, base + 6] = float((hist > 0).sum() / N_TIME_BINS)
            out[ri, base + 7] = float(g_xyz[sel].std(axis=0).mean())
            out[ri, base + 8] = _safe_mean(g_vib[sel])
            out[ri, base + 9] = _safe_mean(g_wd[sel])

        # ─── Tier 2 — per-phase (5 × 6 = 30) ───
        base_t2 = 30
        for pi in range(N_PHASES):
            sel = (g_ph == pi)
            n = int(sel.sum())
            b = base_t2 + pi * 6
            if n == 0:
                continue
            out[ri, b + 0] = n
            out[ri, b + 1] = _safe_mean(g_inten[sel])
            # source_diversity in this phase (of 3)
            uniq_src = np.unique(g_src[sel])
            out[ri, b + 2] = int(((uniq_src >= 0) & (uniq_src < N_SOURCES)).sum())
            out[ri, b + 3] = _safe_mean(g_vib[sel])
            out[ri, b + 4] = _safe_mean(g_wd[sel])
            out[ri, b + 5] = _safe_mean(g_nex[sel])

        # ─── Tier 3 — continuous-column stats (4 × 6 = 24) ───
        base_t3 = 60
        for ci, col in enumerate([g_vib, g_wd, g_nex, g_inten]):
            b = base_t3 + ci * 6
            out[ri, b + 0] = _safe_mean(col)
            out[ri, b + 1] = _safe_std(col)
            out[ri, b + 2] = float(col.max()) if col.size else 0.0
            out[ri, b + 3] = _skewness(col)
            out[ri, b + 4] = float((col > 0).mean()) if col.size else 0.0
            out[ri, b + 5] = _pct(col, 90)

        # ─── Tier 4 — cross-channel (12) ───
        base_t4 = 84
        src_counts = np.asarray(
            [int((g_src == si).sum()) for si in range(N_SOURCES)],
            dtype=np.float64,
        )
        phase_counts = np.asarray(
            [int((g_ph == pi).sum()) for pi in range(N_PHASES)],
            dtype=np.float64,
        )
        out[ri, base_t4 + 0] = int((src_counts > 0).sum())            # channel_agreement
        out[ri, base_t4 + 1] = int(src_counts.argmax()) if src_counts.sum() else 0
        out[ri, base_t4 + 2] = _shannon(src_counts)                   # channel_entropy
        # ratios
        uv, lif, efp = src_counts[2], src_counts[0], src_counts[1]
        out[ri, base_t4 + 3] = uv / max(lif, 1.0)                     # uv_lif_ratio
        out[ri, base_t4 + 4] = uv / max(efp, 1.0)                     # uv_efp_ratio
        out[ri, base_t4 + 5] = lif / max(efp, 1.0)                    # lif_efp_ratio
        out[ri, base_t4 + 6] = int((phase_counts > 0).sum())          # phase_agreement
        out[ri, base_t4 + 7] = int(phase_counts.argmax()) if phase_counts.sum() else 0
        out[ri, base_t4 + 8] = _shannon(phase_counts)                 # phase_entropy
        n_warm = phase_counts[PHASE_IDX["warm_hold"]]
        n_cold = phase_counts[PHASE_IDX["cold_hold"]]
        out[ri, base_t4 + 9] = float(n_warm / max(n_cold, 1))         # phase_transition_ratio
        out[ri, base_t4 + 10] = (phase_counts[PHASE_IDX["cooling"]]
                                  / max(phase_counts[PHASE_IDX["heating"]], 1))
        # onset_phase: first phase with >5% of spikes
        tot = phase_counts.sum()
        if tot > 0:
            frac = phase_counts / tot
            onset = int(np.argmax(frac > 0.05)) if (frac > 0.05).any() else 0
        else:
            onset = 0
        out[ri, base_t4 + 11] = onset

        # ─── Tier 5 — cooperative excitation (6) ───
        base_t5 = 96
        out[ri, base_t5 + 0] = _safe_mean(g_nex)
        out[ri, base_t5 + 1] = float(g_nex.max()) if g_nex.size else 0.0
        out[ri, base_t5 + 2] = _safe_std(g_nex)
        out[ri, base_t5 + 3] = float((g_nex >= 5).mean()) if g_nex.size else 0.0
        # coexcitation_trend — linear slope of n_nearby_excited vs timestep
        if g_ts.size > 5 and g_nex.std() > 1e-9:
            # np.polyfit order=1 returns [slope, intercept]
            slope, _ = np.polyfit(g_ts.astype(np.float64), g_nex.astype(np.float64), 1)
            out[ri, base_t5 + 4] = float(slope)
        warm_nex = g_nex[g_ph == PHASE_IDX["warm_hold"]]
        cold_nex = g_nex[g_ph == PHASE_IDX["cold_hold"]]
        if cold_nex.size > 0 and warm_nex.size > 0:
            cold_m = max(_safe_mean(cold_nex), 1e-6)
            out[ri, base_t5 + 5] = _safe_mean(warm_nex) / cold_m

        # ─── Tier 6 — intensity distribution (6) ───
        base_t6 = 102
        out[ri, base_t6 + 0] = _pct(g_inten, 10)
        out[ri, base_t6 + 1] = _pct(g_inten, 50)
        out[ri, base_t6 + 2] = _skewness(g_inten)
        # Bimodality: variance in top vs bottom halves by intensity
        if g_inten.size >= 4:
            srt = np.sort(g_inten)
            half = srt.size // 2
            bot = srt[:half]; top = srt[half:]
            vb = float(bot.var()); vt = float(top.var())
            out[ri, base_t6 + 3] = vt / max(vb, 1e-9)
        out[ri, base_t6 + 4] = float((g_inten < np.median(g_inten)).mean()) if g_inten.size else 0.0
        if g_ts.size > 5 and g_inten.std() > 1e-9:
            slope, _ = np.polyfit(g_ts.astype(np.float64),
                                   g_inten.astype(np.float64), 1)
            out[ri, base_t6 + 5] = float(slope)

    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)


# ─────────────────────────────────────────────────────────────
#  Per-target worker
# ─────────────────────────────────────────────────────────────

def process_target(target: str, bundle_dir: Path, r2_prefix: str,
                    stage_root: Path, bwlimit: str,
                    keep_stage: bool = False) -> Dict:
    status: Dict = {"target": target, "ok": False}
    bundle_path = bundle_dir / f"{target}_features.npz"
    if not bundle_path.exists():
        status["error"] = "bundle missing"
        return status

    try:
        bundle = np.load(bundle_path, allow_pickle=False)
    except Exception as e:
        status["error"] = f"bundle load failed: {e}"
        return status

    # Idempotent: skip if channel_features already present and correct dim
    if ("channel_features" in bundle.files
            and bundle["channel_features"].ndim == 2
            and bundle["channel_features"].shape[1] == FEATURE_DIM):
        status["ok"] = True
        status["skipped_reason"] = "already has channel_features"
        return status

    coords = bundle["coords"] if "coords" in bundle.files else bundle.get("ca_coords")
    if coords is None:
        status["error"] = "no coords in bundle"
        return status

    stage_dir = stage_root / target
    try:
        if not rclone_parquets(f"{r2_prefix}/{target}/", stage_dir, bwlimit):
            status["error"] = "rclone pull failed"
            return status

        t0 = time.time()
        spikes = load_spikes(stage_dir)
        if not spikes:
            status["error"] = "no parquet spikes found"
            # Still write zeros so dim matches
            channel = np.zeros((coords.shape[0], FEATURE_DIM), dtype=np.float32)
        else:
            channel = compute_per_residue(coords.astype(np.float32), spikes)
        compute_sec = time.time() - t0

        # Re-save npz with channel_features added (preserve all other keys)
        existing = {k: bundle[k] for k in bundle.files}
        existing["channel_features"] = channel
        np.savez_compressed(bundle_path, **existing)

        status.update({
            "ok": True,
            "compute_sec": round(compute_sec, 2),
            "n_spikes_used": int(len(spikes["x"])) if spikes else 0,
            "n_residues_with_spikes": int((channel[:, 0] > 0).sum()),
        })
    except Exception as e:
        status["error"] = str(e)
        status["traceback"] = traceback.format_exc()
    finally:
        if not keep_stage and stage_dir.exists():
            shutil.rmtree(stage_dir, ignore_errors=True)
    return status


# ─────────────────────────────────────────────────────────────
#  Parallel driver
# ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle-dir", type=Path, required=True)
    parser.add_argument("--r2-prefix", default="r2:prism-archive/10k-runs")
    parser.add_argument("--stage-dir", type=Path,
                        default=Path("/mnt/storage/spike-audit/r2stage-channel"))
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--bwlimit", default="20M")
    parser.add_argument("--max", type=int, default=0, help="Cap # targets")
    parser.add_argument("--targets", default="", help="CSV subset")
    parser.add_argument("--keep-stage", action="store_true")
    args = parser.parse_args()

    args.stage_dir.mkdir(parents=True, exist_ok=True)

    npzs = sorted(args.bundle_dir.glob("*_features.npz"))
    targets = [p.stem.replace("_features", "") for p in npzs]
    if args.targets:
        wanted = set(args.targets.split(","))
        targets = [t for t in targets if t in wanted]
    if args.max:
        targets = targets[:args.max]
    print(f"Targets: {len(targets)}  workers: {args.workers}  bwlimit: {args.bwlimit}")

    n_ok = n_skip = n_fail = 0
    results: List[Dict] = []
    t0 = time.time()

    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        future_to_tgt = {
            pool.submit(process_target, t, args.bundle_dir,
                         args.r2_prefix, args.stage_dir, args.bwlimit,
                         args.keep_stage): t
            for t in targets
        }
        for i, fut in enumerate(as_completed(future_to_tgt)):
            status = fut.result()
            results.append(status)
            if status.get("skipped_reason"):
                n_skip += 1
                tag = "SKIP"
            elif status.get("ok"):
                n_ok += 1
                tag = "OK"
            else:
                n_fail += 1
                tag = f"FAIL ({status.get('error', '?')[:50]})"
            elapsed = time.time() - t0
            eta = elapsed / max(i + 1, 1) * (len(targets) - i - 1)
            print(f"  [{i+1}/{len(targets)}] {status['target']:24s} {tag}  "
                  f"elapsed={elapsed/60:.1f}m ETA={eta/60:.0f}m", flush=True)

    manifest = {
        "total": len(targets), "ok": n_ok, "skipped": n_skip, "failed": n_fail,
        "feature_dim": FEATURE_DIM,
        "tier_offsets": TIER_OFFSETS,
        "results": results,
    }
    mpath = args.bundle_dir / "channel_features_manifest.json"
    mpath.write_text(json.dumps(manifest, indent=2, default=str))
    print(f"\n  DONE: ok={n_ok} skipped={n_skip} failed={n_fail}  "
          f"time={(time.time()-t0)/60:.1f}m")
    print(f"  Manifest: {mpath}")


if __name__ == "__main__":
    main()
