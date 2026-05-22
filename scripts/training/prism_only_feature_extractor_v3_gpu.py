#!/usr/bin/env python3
"""
PRISM-only feature extractor v3 (GPU) — torch.scatter_add_ over (rid, spike) pairs.

v3 design rationale:
  v1: Python Welford loop, np.add.at sequential                ~0.25M spikes/s
  v2: bincount-based CPU vectorization                          ~0.26M spikes/s
  v3: torch.scatter_add_ on RTX 5080 (sm_120, 16 GB VRAM)      target ~25M spikes/s

The hot path is "for each (residue, value) pair, accumulate into a per-residue
bucket." That is exactly torch.scatter_add_, which runs as a fused CUDA kernel.
Closed-form moments fall out of sum/sum²/sum³/sum⁴.

Memory plan: chunk the (rid, spike_idx) pairs at 50M per chunk to fit in 16 GB
VRAM with headroom (each chunk ~ 6 features × 50M × 4B + indices ≈ 2 GB).

JSON attach functions are identical to v2.1 (operate on the post-aggregation
DataFrame on CPU — JSON parsing is negligible).
"""
from __future__ import annotations
import argparse
import json
import time
from pathlib import Path
from typing import Optional

import numpy as np
import pyarrow as pa
import pyarrow.ipc as ipc
import polars as pl
import torch


DEFAULT_N_STREAMS = 24
N_SPIKE_SOURCES = 8
N_AROMATIC_TYPES = 8
N_CCNS_PHASES = 8
N_PHASE_BITS = 32
N_BURIAL_BINS = 10
N_IPCT_BINS = 10
N_WAVELENGTH_BINS = 16
N_TIME_WINDOWS = 16
N_BG_CLASSES = 4

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def open_arrow(path: Path):
    src = pa.memory_map(str(path), "r")
    try:
        return ipc.RecordBatchFileReader(src), "file"
    except Exception:
        src.seek(0)
        return ipc.RecordBatchStreamReader(src), "stream"


def discover_n_residues(arrow_path: Path) -> int:
    reader, _ = open_arrow(arrow_path)
    max_rid = -1
    if hasattr(reader, "num_record_batches"):
        for i in range(reader.num_record_batches):
            batch = reader.get_batch(i)
            nr_col = batch.column("nearby_residues")
            try:
                arr = nr_col.values.to_numpy(zero_copy_only=False)
            except Exception:
                arr = nr_col.flatten().to_numpy(zero_copy_only=False)
            local_max = int(arr.max()) if arr.size else -1
            if local_max > max_rid:
                max_rid = local_max
    return max_rid + 1


def gpu_moments(rid_t: torch.Tensor, vals_t: torch.Tensor, n_res: int) -> dict:
    """Closed-form 4-moment statistics per residue using GPU scatter_add."""
    vals_t = vals_t.to(torch.float32)
    finite = torch.isfinite(vals_t)
    if not finite.all():
        rid_t = rid_t[finite]
        vals_t = vals_t[finite]

    v2_ = vals_t * vals_t
    v3_ = v2_ * vals_t
    v4_ = v2_ * v2_

    n = torch.zeros(n_res, dtype=torch.float64, device=rid_t.device)
    s1 = torch.zeros(n_res, dtype=torch.float64, device=rid_t.device)
    s2 = torch.zeros(n_res, dtype=torch.float64, device=rid_t.device)
    s3 = torch.zeros(n_res, dtype=torch.float64, device=rid_t.device)
    s4 = torch.zeros(n_res, dtype=torch.float64, device=rid_t.device)

    ones = torch.ones_like(vals_t, dtype=torch.float64)
    n.scatter_add_(0, rid_t, ones)
    s1.scatter_add_(0, rid_t, vals_t.to(torch.float64))
    s2.scatter_add_(0, rid_t, v2_.to(torch.float64))
    s3.scatter_add_(0, rid_t, v3_.to(torch.float64))
    s4.scatter_add_(0, rid_t, v4_.to(torch.float64))

    nz = torch.clamp(n, min=1.0)
    mean = s1 / nz
    e2 = s2 / nz
    e3 = s3 / nz
    e4 = s4 / nz
    var = torch.clamp(e2 - mean * mean, min=0.0)
    std = torch.sqrt(var)
    m3 = e3 - 3 * mean * e2 + 2 * mean.pow(3)
    m4 = e4 - 4 * mean * e3 + 6 * mean.pow(2) * e2 - 3 * mean.pow(4)
    skew = torch.where(var > 1e-12, m3 / torch.clamp(std.pow(3), min=1e-12), torch.zeros_like(mean))
    kurt = torch.where(var > 1e-12, m4 / torch.clamp(var.pow(2), min=1e-12) - 3.0, torch.zeros_like(mean))

    return {
        "n": n.to(torch.float32).cpu().numpy(),
        "mean": mean.to(torch.float32).cpu().numpy(),
        "var": var.to(torch.float32).cpu().numpy(),
        "std": std.to(torch.float32).cpu().numpy(),
        "skew": skew.to(torch.float32).cpu().numpy(),
        "kurt": kurt.to(torch.float32).cpu().numpy(),
        "max": np.zeros(n_res, dtype=np.float32),
        "min": np.zeros(n_res, dtype=np.float32),
    }


def gpu_hist_2d(rid_t: torch.Tensor, cat_t: torch.Tensor, n_res: int, n_cat: int) -> np.ndarray:
    """Per-residue × per-category histogram via single scatter_add."""
    cat_t = torch.clamp(cat_t.to(torch.long), 0, n_cat - 1)
    flat_idx = rid_t * n_cat + cat_t
    hist_flat = torch.zeros(n_res * n_cat, dtype=torch.float32, device=rid_t.device)
    hist_flat.scatter_add_(0, flat_idx, torch.ones_like(flat_idx, dtype=torch.float32))
    return hist_flat.view(n_res, n_cat).to(torch.int64).cpu().numpy()


def gpu_phase_bit_counts(rid_t: torch.Tensor, phase_bits_t: torch.Tensor, n_res: int) -> np.ndarray:
    """Per-residue × 32-bit counts. Serial per-bit loop keeps peak VRAM ~1.6 GB
    for 50M-pair chunks (vs ~13 GB for the fully broadcast variant)."""
    pb = phase_bits_t.to(torch.int64)
    out = torch.zeros(n_res, N_PHASE_BITS, dtype=torch.float32, device=rid_t.device)
    for bit in range(N_PHASE_BITS):
        bit_val = ((pb >> bit) & 1).to(torch.float32)
        out[:, bit].scatter_add_(0, rid_t, bit_val)
    return out.to(torch.int64).cpu().numpy()


def extract_features(
    arrow_path: Path,
    output_path: Path,
    binding_sites_path: Optional[Path] = None,
    kcc_path: Optional[Path] = None,
    therm_path: Optional[Path] = None,
    chunk_pairs: int = 25_000_000,
):
    t0 = time.time()
    print(f"[{time.time() - t0:6.1f}s] v3-GPU extracting from {arrow_path}", flush=True)
    print(f"[{time.time() - t0:6.1f}s]   device: {DEVICE}  size: {arrow_path.stat().st_size / 1e9:.2f} GB", flush=True)
    if DEVICE == "cuda":
        print(f"[{time.time() - t0:6.1f}s]   GPU: {torch.cuda.get_device_name(0)}  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB", flush=True)

    n_res = discover_n_residues(arrow_path)
    print(f"[{time.time() - t0:6.1f}s]   N_residues = {n_res}", flush=True)

    moments_accum = {k: None for k in ("intensity", "ve", "wd", "wdc", "burial", "nsd")}
    n_spikes_total = np.zeros(n_res, dtype=np.int64)
    n_spikes_excited = np.zeros(n_res, dtype=np.int64)
    n_spikes_per_stream = np.zeros((n_res, N_STREAMS), dtype=np.int64)
    spike_source_hist = np.zeros((n_res, N_SPIKE_SOURCES), dtype=np.int64)
    aromatic_hist = np.zeros((n_res, N_AROMATIC_TYPES), dtype=np.int64)
    ccns_hist = np.zeros((n_res, N_CCNS_PHASES), dtype=np.int64)
    bg_class_hist = np.zeros((n_res, N_BG_CLASSES), dtype=np.int64)
    burial_bin_hist = np.zeros((n_res, N_BURIAL_BINS), dtype=np.int64)
    ipct_bin_hist = np.zeros((n_res, N_IPCT_BINS), dtype=np.int64)
    wavelength_hist = np.zeros((n_res, N_WAVELENGTH_BINS), dtype=np.int64)
    time_window_hist = np.zeros((n_res, N_TIME_WINDOWS), dtype=np.int64)
    phase_bit_counts = np.zeros((n_res, N_PHASE_BITS), dtype=np.int64)

    reader, mode = open_arrow(arrow_path)
    if hasattr(reader, "num_record_batches"):
        n_batches = reader.num_record_batches
        batch_iter = (reader.get_batch(i) for i in range(n_batches))
    else:
        n_batches = None
        batch_iter = iter(reader)

    spikes_seen = 0
    for bi, batch in enumerate(batch_iter):
        sb_n = batch.num_rows
        if sb_n == 0:
            continue
        spikes_seen += sb_n
        t_batch = time.time()

        nearby_col = batch.column("nearby_residues")
        try:
            nearby_flat = nearby_col.values.to_numpy(zero_copy_only=False).astype(np.int32, copy=False)
        except Exception:
            nearby_flat = np.stack([np.asarray(x, dtype=np.int32) for x in nearby_col.to_numpy(zero_copy_only=False)]).reshape(-1)
        nearby_arr = nearby_flat.reshape(-1, 8)
        n_nearby = batch.column("n_residues").to_numpy(zero_copy_only=False).astype(np.int32)

        spike_idx = np.repeat(np.arange(sb_n, dtype=np.int32), 8)
        residue_flat = nearby_arr.reshape(-1)
        slot_idx = np.tile(np.arange(8, dtype=np.int32), sb_n)
        valid = (residue_flat >= 0) & (slot_idx < np.repeat(n_nearby, 8)) & (residue_flat < n_res)
        spike_idx = spike_idx[valid]
        rid = residue_flat[valid]

        intensity = batch.column("intensity").to_numpy(zero_copy_only=False)
        ve = batch.column("vibrational_energy").to_numpy(zero_copy_only=False)
        wd = batch.column("water_density").to_numpy(zero_copy_only=False)
        wdc = batch.column("wd_change").to_numpy(zero_copy_only=False)
        wavelength = batch.column("wavelength_nm").to_numpy(zero_copy_only=False)
        burial = batch.column("burial_score").to_numpy(zero_copy_only=False)
        stream_id = batch.column("stream_id").to_numpy(zero_copy_only=False)
        spike_source = batch.column("spike_source").to_numpy(zero_copy_only=False)
        aromatic_type = batch.column("aromatic_type").to_numpy(zero_copy_only=False)
        ccns_phase = batch.column("ccns_phase").to_numpy(zero_copy_only=False)
        phase_bits = batch.column("phase_bits").to_numpy(zero_copy_only=False).astype(np.uint32)
        n_nearby_excited = batch.column("n_nearby_excited").to_numpy(zero_copy_only=False)
        ipct = batch.column("intensity_percentile").to_numpy(zero_copy_only=False)
        bg_class = batch.column("background_class").to_numpy(zero_copy_only=False)
        nearest_site_dist = batch.column("nearest_site_dist").to_numpy(zero_copy_only=False)
        frame_index = batch.column("frame_index").to_numpy(zero_copy_only=False)
        max_frame = max(int(frame_index.max()), 1) if frame_index.size else 1

        n_pairs = rid.size
        n_chunks = (n_pairs + chunk_pairs - 1) // chunk_pairs

        for ci in range(n_chunks):
            lo = ci * chunk_pairs
            hi = min(lo + chunk_pairs, n_pairs)
            rid_chunk = rid[lo:hi]
            sidx_chunk = spike_idx[lo:hi]

            rid_t = torch.from_numpy(rid_chunk.astype(np.int64)).to(DEVICE)

            for feat_name, feat_vals in (
                ("intensity", intensity), ("ve", ve), ("wd", wd),
                ("wdc", wdc), ("burial", burial), ("nsd", nearest_site_dist),
            ):
                vals_t = torch.from_numpy(feat_vals[sidx_chunk].astype(np.float32, copy=False)).to(DEVICE)
                m = gpu_moments(rid_t, vals_t, n_res)
                if moments_accum[feat_name] is None:
                    moments_accum[feat_name] = m
                else:
                    moments_accum[feat_name] = _merge_moments(moments_accum[feat_name], m)

            n_spikes_total += np.bincount(rid_chunk, minlength=n_res)
            excited_mask = n_nearby_excited[sidx_chunk] > 0
            n_spikes_excited += np.bincount(rid_chunk[excited_mask], minlength=n_res)

            n_spikes_per_stream += gpu_hist_2d(rid_t, torch.from_numpy(stream_id[sidx_chunk].astype(np.int64)).to(DEVICE), n_res, N_STREAMS)
            spike_source_hist += gpu_hist_2d(rid_t, torch.from_numpy(spike_source[sidx_chunk].astype(np.int64)).to(DEVICE), n_res, N_SPIKE_SOURCES)
            aromatic_hist += gpu_hist_2d(rid_t, torch.from_numpy(aromatic_type[sidx_chunk].astype(np.int64)).to(DEVICE), n_res, N_AROMATIC_TYPES)
            ccns_hist += gpu_hist_2d(rid_t, torch.from_numpy(ccns_phase[sidx_chunk].astype(np.int64)).to(DEVICE), n_res, N_CCNS_PHASES)
            bg_class_hist += gpu_hist_2d(rid_t, torch.from_numpy(bg_class[sidx_chunk].astype(np.int64)).to(DEVICE), n_res, N_BG_CLASSES)

            burial_b = np.clip((burial[sidx_chunk] * N_BURIAL_BINS).astype(np.int32), 0, N_BURIAL_BINS - 1)
            burial_bin_hist += gpu_hist_2d(rid_t, torch.from_numpy(burial_b.astype(np.int64)).to(DEVICE), n_res, N_BURIAL_BINS)

            ipct_b = np.clip(ipct[sidx_chunk].astype(np.int32) // (100 // N_IPCT_BINS), 0, N_IPCT_BINS - 1)
            ipct_bin_hist += gpu_hist_2d(rid_t, torch.from_numpy(ipct_b.astype(np.int64)).to(DEVICE), n_res, N_IPCT_BINS)

            wl_safe = np.where(wavelength[sidx_chunk] > 0, wavelength[sidx_chunk], 280.0)
            wl_b = np.clip(((wl_safe - 200.0) / 50.0).astype(np.int32), 0, N_WAVELENGTH_BINS - 1)
            wavelength_hist += gpu_hist_2d(rid_t, torch.from_numpy(wl_b.astype(np.int64)).to(DEVICE), n_res, N_WAVELENGTH_BINS)

            tw_b = np.clip(
                (frame_index[sidx_chunk].astype(np.int64) * N_TIME_WINDOWS // (max_frame + 1)).astype(np.int32),
                0, N_TIME_WINDOWS - 1,
            )
            time_window_hist += gpu_hist_2d(rid_t, torch.from_numpy(tw_b.astype(np.int64)).to(DEVICE), n_res, N_TIME_WINDOWS)

            pb_t = torch.from_numpy(phase_bits[sidx_chunk].astype(np.int64)).to(DEVICE)
            phase_bit_counts += gpu_phase_bit_counts(rid_t, pb_t, n_res)

            del rid_t, pb_t
            if DEVICE == "cuda":
                torch.cuda.empty_cache()

        elapsed = time.time() - t0
        rate = spikes_seen / max(elapsed, 0.001)
        print(
            f"[{elapsed:6.1f}s] batch {bi+1}/{n_batches or '?'}  "
            f"spikes={spikes_seen:,}  chunks={n_chunks}  "
            f"rate={rate / 1e6:.2f}M/s  batch_time={time.time() - t_batch:.1f}s",
            flush=True,
        )

    print(f"[{time.time() - t0:6.1f}s] assembling dataframe", flush=True)

    cols = {"residue_id": np.arange(n_res, dtype=np.int32)}
    cols["n_spikes"] = n_spikes_total
    cols["n_spikes_excited"] = n_spikes_excited

    for feat_name, m in moments_accum.items():
        if m is None:
            for stat in ("n", "mean", "var", "std", "skew", "kurt", "max", "min"):
                cols[f"{feat_name}_{stat}"] = np.zeros(n_res, dtype=np.float32)
        else:
            for stat in ("n", "mean", "var", "std", "skew", "kurt", "max", "min"):
                cols[f"{feat_name}_{stat}"] = m[stat]

    for s in range(N_STREAMS):
        cols[f"n_spikes_stream_{s}"] = n_spikes_per_stream[:, s]
    for s in range(N_SPIKE_SOURCES):
        cols[f"src_hist_{s}"] = spike_source_hist[:, s]
    for a in range(N_AROMATIC_TYPES):
        cols[f"aromatic_hist_{a}"] = aromatic_hist[:, a]
    for c in range(N_CCNS_PHASES):
        cols[f"ccns_hist_{c}"] = ccns_hist[:, c]
    for b in range(N_BG_CLASSES):
        cols[f"bg_class_hist_{b}"] = bg_class_hist[:, b]
    for b in range(N_BURIAL_BINS):
        cols[f"burial_bin_{b}"] = burial_bin_hist[:, b]
    for b in range(N_IPCT_BINS):
        cols[f"ipct_bin_{b}"] = ipct_bin_hist[:, b]
    for b in range(N_WAVELENGTH_BINS):
        cols[f"wavelength_bin_{b}"] = wavelength_hist[:, b]
    for b in range(N_TIME_WINDOWS):
        cols[f"timewin_bin_{b}"] = time_window_hist[:, b]
    for b in range(N_PHASE_BITS):
        cols[f"phase_bit_{b}_count"] = phase_bit_counts[:, b]

    nz = np.maximum(n_spikes_total.astype(np.float32), 1.0)
    cols["excited_fraction"] = n_spikes_excited.astype(np.float32) / nz
    for s in range(N_STREAMS):
        cols[f"stream_fraction_{s}"] = n_spikes_per_stream[:, s].astype(np.float32) / nz
    cols["phase_popcount_mean"] = phase_bit_counts.sum(axis=1).astype(np.float32) / nz

    df = pl.DataFrame(cols)

    ca_positions = np.zeros((n_res, 3), dtype=np.float32)
    if kcc_path and kcc_path.exists():
        try:
            df, ca_positions = _attach_kcc(df, kcc_path, n_res)
        except Exception as e:
            print(f"[warn] kcc attach failed: {e}", flush=True)

    if binding_sites_path and binding_sites_path.exists():
        try:
            with open(binding_sites_path) as f:
                sites = json.load(f)
            df = _attach_site_metadata(df, sites, n_res, ca_positions)
        except Exception as e:
            print(f"[warn] binding_sites attach failed: {e}", flush=True)

    if therm_path and therm_path.exists():
        try:
            df = _attach_therm(df, therm_path, n_res, ca_positions)
        except Exception as e:
            print(f"[warn] therm attach failed: {e}", flush=True)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(output_path, compression="zstd", compression_level=9)
    elapsed = time.time() - t0
    size_mb = output_path.stat().st_size / 1e6
    rate = spikes_seen / max(elapsed, 0.001)
    print(
        f"[{elapsed:6.1f}s] wrote {output_path} ({size_mb:.2f} MB) — "
        f"{df.height} residues × {df.width} cols, "
        f"{spikes_seen:,} spikes @ {rate / 1e6:.2f}M/s (device={DEVICE})",
        flush=True,
    )
    return df


def _merge_moments(a: dict, b: dict) -> dict:
    n_a = a["n"].astype(np.float64)
    n_b = b["n"].astype(np.float64)
    n_ab = n_a + n_b
    nz = np.where(n_ab > 0, n_ab, 1.0)
    M2_a = a["var"] * np.maximum(a["n"] - 1, 1)
    M2_b = b["var"] * np.maximum(b["n"] - 1, 1)
    delta = b["mean"] - a["mean"]
    new_mean = np.where(n_ab > 0, (n_a * a["mean"] + n_b * b["mean"]) / nz, 0.0)
    new_M2 = M2_a + M2_b + delta * delta * n_a * n_b / nz
    new_var = np.where(n_ab > 1, new_M2 / np.maximum(n_ab - 1, 1), 0.0)
    new_std = np.sqrt(np.maximum(new_var, 0.0))
    return {
        "n": n_ab.astype(np.float32),
        "mean": new_mean.astype(np.float32),
        "var": new_var.astype(np.float32),
        "std": new_std.astype(np.float32),
        "skew": np.where(n_a > n_b, a["skew"], b["skew"]).astype(np.float32),
        "kurt": np.where(n_a > n_b, a["kurt"], b["kurt"]).astype(np.float32),
        "max": np.maximum(a["max"], b["max"]).astype(np.float32),
        "min": np.minimum(a["min"], b["min"]).astype(np.float32),
    }


def _attach_kcc(df: pl.DataFrame, kcc_path: Path, n_res: int):
    """kcc_visualization.json residues are list-position-indexed (the residue_id
    field is the PDB residue number, not the topology index).
    """
    with open(kcc_path) as f:
        kcc = json.load(f)
    residues = kcc.get("residues", []) if isinstance(kcc, dict) else []
    keys = (
        "kcc_score", "active_causal_steps", "burst_motion", "causal_lag",
        "direction_score", "lag_corr_peak", "local_cov", "motion_efficiency",
        "sum_motion", "total_steps",
    )
    cols = {k: np.zeros(n_res, dtype=np.float32) for k in keys}
    ca_pos = np.zeros((n_res, 3), dtype=np.float32)
    net_dx_norm = np.zeros(n_res, dtype=np.float32)
    pdb_resid = np.full(n_res, -1, dtype=np.int32)

    for idx, r in enumerate(residues):
        if not isinstance(r, dict) or idx >= n_res:
            continue
        for k in keys:
            v = r.get(k, 0.0)
            try:
                cols[k][idx] = float(v) if not isinstance(v, (list, dict)) else 0.0
            except (TypeError, ValueError):
                cols[k][idx] = 0.0
        cp = r.get("ca_position")
        if isinstance(cp, list) and len(cp) == 3:
            ca_pos[idx] = [float(cp[0]), float(cp[1]), float(cp[2])]
        dx, dy, dz = r.get("net_dx", 0.0), r.get("net_dy", 0.0), r.get("net_dz", 0.0)
        try:
            net_dx_norm[idx] = float(np.sqrt(float(dx)**2 + float(dy)**2 + float(dz)**2))
        except (TypeError, ValueError):
            net_dx_norm[idx] = 0.0
        try:
            pdb_resid[idx] = int(r.get("residue_id", -1))
        except (TypeError, ValueError):
            pass

    df = df.with_columns(
        [pl.Series(k, cols[k]) for k in keys]
        + [
            pl.Series("ca_x", ca_pos[:, 0]),
            pl.Series("ca_y", ca_pos[:, 1]),
            pl.Series("ca_z", ca_pos[:, 2]),
            pl.Series("net_dx_norm", net_dx_norm),
            pl.Series("pdb_resid", pdb_resid),
        ]
    )

    sites = kcc.get("sites", []) if isinstance(kcc, dict) else []
    if sites and ca_pos.any():
        site_keys = (
            "kcc_confidence", "temporal_corr", "site_burst_motion", "site_causal_lag",
            "site_direction_score", "site_lag_corr_peak", "site_local_cov",
            "site_motion_efficiency",
        )
        centroids = []
        site_scalar_vals = {k: [] for k in site_keys}
        gtck_vals = []
        driver_resid_vals = []

        for s in sites:
            c = s.get("centroid")
            if not c or len(c) != 3:
                continue
            centroids.append([float(c[0]), float(c[1]), float(c[2])])

            kcc_data = s.get("kcc", {})
            if not isinstance(kcc_data, dict):
                kcc_data = {}
            for k in site_keys:
                v = kcc_data.get(k, 0.0)
                try:
                    site_scalar_vals[k].append(float(v) if not isinstance(v, (list, dict)) else 0.0)
                except (TypeError, ValueError):
                    site_scalar_vals[k].append(0.0)

            try:
                gtck_vals.append(float(s.get("gtck_rank", 0.0)))
            except (TypeError, ValueError):
                gtck_vals.append(0.0)
            try:
                driver_resid_vals.append(int(kcc_data.get("driver_residue_id", -1)))
            except (TypeError, ValueError):
                driver_resid_vals.append(-1)

        if centroids:
            cs = np.array(centroids, dtype=np.float32)
            diff = ca_pos[:, None, :] - cs[None, :, :]
            dist2 = (diff * diff).sum(axis=2)
            nearest = dist2.argmin(axis=1)
            new_cols = [
                pl.Series("nearest_site_gtck", np.array(gtck_vals, dtype=np.float32)[nearest]),
            ]
            for k in site_keys:
                vals = np.array(site_scalar_vals[k], dtype=np.float32)
                new_cols.append(pl.Series(f"nearest_{k}", vals[nearest]))
            df = df.with_columns(new_cols)

    return df, ca_pos


def _attach_site_metadata(df: pl.DataFrame, sites, n_res: int, ca_positions: np.ndarray) -> pl.DataFrame:
    in_top_site = np.zeros(n_res, dtype=np.int8)
    nearest_site_id = np.full(n_res, -1, dtype=np.int32)
    nearest_site_dist = np.full(n_res, 1e6, dtype=np.float32)
    nearest_site_druggability = np.zeros(n_res, dtype=np.float32)
    nearest_site_classification = np.zeros(n_res, dtype=np.int8)

    site_list = []
    if isinstance(sites, dict):
        site_list = sites.get("sites", []) or sites.get("druggable_sites", []) or sites.get("all_pockets", []) or sites.get("cryptic_sites", []) or []
    elif isinstance(sites, list):
        site_list = sites

    class_map = {"druggable": 1, "Druggable": 1, "DRUGGABLE": 1, "cryptic": 2, "Cryptic": 2, "CRYPTIC": 2, "orthosteric": 3, "Orthosteric": 3, "ORTHOSTERIC": 3, "surface": 4}

    centroids, drug, cls, sids = [], [], [], []
    for site in site_list:
        if not isinstance(site, dict):
            continue
        c = site.get("centroid") or site.get("center")
        if not c or len(c) != 3:
            continue
        centroids.append([float(c[0]), float(c[1]), float(c[2])])
        drug.append(float(site.get("druggability", 0.0)))
        cls.append(class_map.get(site.get("classification", ""), 0))
        sids.append(int(site.get("site_id", -1)))

    if not centroids or not ca_positions.any():
        return df.with_columns([
            pl.Series("in_top_site", in_top_site),
            pl.Series("nearest_site_id_meta", nearest_site_id),
            pl.Series("nearest_site_dist_meta", nearest_site_dist),
            pl.Series("nearest_site_druggability", nearest_site_druggability),
            pl.Series("nearest_site_classification", nearest_site_classification),
        ])

    cs = np.array(centroids, dtype=np.float32)
    diff = ca_positions[:, None, :] - cs[None, :, :]
    dist = np.sqrt(np.maximum((diff * diff).sum(axis=2), 0.0))
    nearest = dist.argmin(axis=1)
    nearest_site_id[:] = np.array(sids, dtype=np.int32)[nearest]
    nearest_site_dist[:] = dist[np.arange(n_res), nearest]
    nearest_site_druggability[:] = np.array(drug, dtype=np.float32)[nearest]
    nearest_site_classification[:] = np.array(cls, dtype=np.int8)[nearest]
    in_top_site[nearest_site_dist < 8.0] = 1

    return df.with_columns([
        pl.Series("in_top_site", in_top_site),
        pl.Series("nearest_site_id_meta", nearest_site_id),
        pl.Series("nearest_site_dist_meta", nearest_site_dist),
        pl.Series("nearest_site_druggability", nearest_site_druggability),
        pl.Series("nearest_site_classification", nearest_site_classification),
    ])


def _attach_therm(df: pl.DataFrame, therm_path: Path, n_res: int, ca_positions: np.ndarray) -> pl.DataFrame:
    with open(therm_path) as f:
        therm = json.load(f)
    pockets = therm.get("pockets", []) if isinstance(therm, dict) else []

    therm_class = np.zeros(n_res, dtype=np.int8)
    is_cryptic = np.zeros(n_res, dtype=np.int8)
    ccns_tau = np.zeros(n_res, dtype=np.float32)
    druggability = np.zeros(n_res, dtype=np.float32)
    hysteresis_asym = np.zeros(n_res, dtype=np.float32)
    nearest_pocket_id = np.full(n_res, -1, dtype=np.int32)
    nearest_pocket_dist = np.full(n_res, 1e6, dtype=np.float32)
    max_te = np.zeros(n_res, dtype=np.float32)
    sum_te = np.zeros(n_res, dtype=np.float32)
    pocket_top_count = np.zeros(n_res, dtype=np.int32)

    cls_map = {
        "INERT": 1, "Inert": 1, "inert": 1,
        "CRYPTIC": 2, "Cryptic": 2, "cryptic": 2,
        "ORTHOSTERIC": 3, "Orthosteric": 3, "orthosteric": 3,
        "SURFACE": 4, "Surface": 4, "surface": 4,
        "DRUGGABLE": 5, "Druggable": 5, "druggable": 5,
    }

    # Build PDB→topology map AND a topology→topology pass-through. Engine conventions
    # are inconsistent: kcc.residues uses PDB resid (378-741) while prism_therm pockets'
    # top_residues use topology index (0-356). Use top_residues.residue_id directly as
    # topology index since that's the actual convention there.
    pdb_to_topo = {}
    if "pdb_resid" in df.columns:
        for topo_i, pdb_r in enumerate(df["pdb_resid"].to_numpy()):
            if 0 <= topo_i < n_res and int(pdb_r) >= 0:
                pdb_to_topo[int(pdb_r)] = topo_i

    if not pockets or not ca_positions.any():
        return df.with_columns([
            pl.Series("therm_class", therm_class),
            pl.Series("is_cryptic", is_cryptic),
            pl.Series("pocket_ccns_tau", ccns_tau),
            pl.Series("pocket_druggability", druggability),
            pl.Series("pocket_hysteresis_asym", hysteresis_asym),
            pl.Series("nearest_pocket_id", nearest_pocket_id),
            pl.Series("nearest_pocket_dist", nearest_pocket_dist),
            pl.Series("max_transfer_entropy", max_te),
            pl.Series("sum_transfer_entropy", sum_te),
            pl.Series("pocket_top_count", pocket_top_count),
        ])

    centroids, tau_v, dr_v, hy_v, cls_v, pid_v, crypt_v = [], [], [], [], [], [], []
    for p in pockets:
        c = p.get("centroid")
        if not c or len(c) != 3:
            continue
        centroids.append([float(c[0]), float(c[1]), float(c[2])])
        tau_v.append(float(p.get("ccns_tau", 0.0)))
        dr_v.append(float(p.get("druggability_score", 0.0)))
        hy_v.append(float(p.get("hysteresis_asymmetry", 0.0)))
        cls_v.append(cls_map.get(p.get("therm_class") or p.get("classification") or "", 0))
        pid_v.append(int(p.get("pocket_id", -1)))
        crypt_v.append(1 if p.get("is_cryptic") else 0)

        for tr in (p.get("top_residues") or []):
            if not isinstance(tr, dict):
                continue
            try:
                rid = int(tr.get("residue_id", -1))
                te = float(tr.get("transfer_entropy", 0.0))
            except (TypeError, ValueError):
                continue
            # prism_therm top_residues.residue_id is the TOPOLOGY index (0-based,
            # 0..n_res). NOT the PDB residue id. Use directly.
            topo_i = rid if 0 <= rid < n_res else pdb_to_topo.get(rid)
            if topo_i is None:
                continue
            if te > max_te[topo_i]:
                max_te[topo_i] = te
            sum_te[topo_i] += te
            pocket_top_count[topo_i] += 1

    cs = np.array(centroids, dtype=np.float32)
    diff = ca_positions[:, None, :] - cs[None, :, :]
    dist = np.sqrt(np.maximum((diff * diff).sum(axis=2), 0.0))
    nearest = dist.argmin(axis=1)
    therm_class[:] = np.array(cls_v, dtype=np.int8)[nearest]
    is_cryptic[:] = np.array(crypt_v, dtype=np.int8)[nearest]
    ccns_tau[:] = np.array(tau_v, dtype=np.float32)[nearest]
    druggability[:] = np.array(dr_v, dtype=np.float32)[nearest]
    hysteresis_asym[:] = np.array(hy_v, dtype=np.float32)[nearest]
    nearest_pocket_id[:] = np.array(pid_v, dtype=np.int32)[nearest]
    nearest_pocket_dist[:] = dist[np.arange(n_res), nearest]

    return df.with_columns([
        pl.Series("therm_class", therm_class),
        pl.Series("is_cryptic", is_cryptic),
        pl.Series("pocket_ccns_tau", ccns_tau),
        pl.Series("pocket_druggability", druggability),
        pl.Series("pocket_hysteresis_asym", hysteresis_asym),
        pl.Series("nearest_pocket_id", nearest_pocket_id),
        pl.Series("nearest_pocket_dist", nearest_pocket_dist),
        pl.Series("max_transfer_entropy", max_te),
        pl.Series("sum_transfer_entropy", sum_te),
        pl.Series("pocket_top_count", pocket_top_count),
    ])


def _attach_asc_consensus(df: pl.DataFrame, asc_path: Path, n_res: int) -> pl.DataFrame:
    """asc_consensus.json: consensus_residues[] with residue_id (PDB), s_pc, n_groups.

    Residues NOT in consensus_residues get default 0. Engine outputs sparse top-K only.
    """
    s_pc = np.zeros(n_res, dtype=np.float32)
    n_groups = np.zeros(n_res, dtype=np.int8)
    n_streams_total = 0
    try:
        with open(asc_path) as f:
            asc = json.load(f)
    except Exception:
        return df.with_columns([
            pl.Series("asc_s_pc", s_pc),
            pl.Series("asc_n_groups", n_groups),
            pl.Series("asc_in_consensus", n_groups),
        ])
    n_streams_total = int(asc.get("n_streams", 0))
    pdb_to_topo = {}
    if "pdb_resid" in df.columns:
        for topo_i, pdb_r in enumerate(df["pdb_resid"].to_numpy()):
            if 0 <= topo_i < n_res and int(pdb_r) >= 0:
                pdb_to_topo[int(pdb_r)] = topo_i
    for r in asc.get("consensus_residues", []):
        if not isinstance(r, dict):
            continue
        try:
            rid_raw = int(r.get("residue_id", -1))
            spc = float(r.get("s_pc", 0.0))
            ng = int(r.get("n_groups", 0))
        except (TypeError, ValueError):
            continue
        topo_i = rid_raw if 0 <= rid_raw < n_res else pdb_to_topo.get(rid_raw)
        if topo_i is None:
            continue
        s_pc[topo_i] = spc
        n_groups[topo_i] = ng
    in_cons = (s_pc > 0).astype(np.int8)
    return df.with_columns([
        pl.Series("asc_s_pc", s_pc),
        pl.Series("asc_n_groups", n_groups),
        pl.Series("asc_in_consensus", in_cons),
    ])


def _attach_gcpid_synergy(df: pl.DataFrame, gcpid_path: Path, n_res: int) -> pl.DataFrame:
    """gcpid_synergy.json: per-residue Partial Information Decomposition (Ince 2017).

    Keys per residue: n_samples, redundancy_nats, synergy_nats, synergy_fraction,
                      total_mi_nats, unique_a_nats, unique_b_nats.
    """
    keys = ("n_samples", "redundancy_nats", "synergy_nats", "synergy_fraction",
            "total_mi_nats", "unique_a_nats", "unique_b_nats")
    cols = {f"gcpid_{k}": np.zeros(n_res, dtype=np.float32) for k in keys}
    try:
        with open(gcpid_path) as f:
            g = json.load(f)
    except Exception:
        return df.with_columns([pl.Series(c, cols[c]) for c in cols])
    pdb_to_topo = {}
    if "pdb_resid" in df.columns:
        for topo_i, pdb_r in enumerate(df["pdb_resid"].to_numpy()):
            if 0 <= topo_i < n_res and int(pdb_r) >= 0:
                pdb_to_topo[int(pdb_r)] = topo_i
    for r in g.get("residues", []):
        if not isinstance(r, dict):
            continue
        try:
            rid_raw = int(r.get("residue_id", -1))
        except (TypeError, ValueError):
            continue
        topo_i = rid_raw if 0 <= rid_raw < n_res else pdb_to_topo.get(rid_raw)
        if topo_i is None:
            continue
        for k in keys:
            try:
                cols[f"gcpid_{k}"][topo_i] = float(r.get(k, 0.0))
            except (TypeError, ValueError):
                pass
    return df.with_columns([pl.Series(c, cols[c]) for c in cols])


def _attach_druggability_pdb(df: pl.DataFrame, dpdb_path: Path, n_res: int) -> pl.DataFrame:
    """druggability.pdb: per-residue druggability stored in the B-factor field of Cα atoms.

    Note: this file uses 1-indexed TOPOLOGY residue numbering (1..n_res), NOT the
    PDB author residue numbers used in kcc_visualization.json. The first Cα is
    residue 1 in the file, corresponding to topology index 0 in the Arrow data.
    Per the file REMARK: B-factor = Transfer Entropy × 100 (per-residue max).
    """
    drug = np.zeros(n_res, dtype=np.float32)
    seen = np.zeros(n_res, dtype=np.bool_)
    try:
        with open(dpdb_path) as f:
            for line in f:
                if not (line.startswith("ATOM") or line.startswith("HETATM")):
                    continue
                if line[12:16].strip() != "CA":
                    continue
                try:
                    file_resid = int(line[22:26].strip())
                    bfac = float(line[60:66].strip())
                except (ValueError, IndexError):
                    continue
                topo_i = file_resid - 1
                if 0 <= topo_i < n_res:
                    drug[topo_i] = bfac
                    seen[topo_i] = True
    except Exception:
        pass
    return df.with_columns([
        pl.Series("residue_druggability_pdb", drug),
        pl.Series("residue_druggability_seen", seen.astype(np.int8)),
    ])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arrow", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--binding-sites", type=Path, default=None)
    ap.add_argument("--kcc", type=Path, default=None)
    ap.add_argument("--therm", type=Path, default=None)
    ap.add_argument("--asc-consensus", type=Path, default=None)
    ap.add_argument("--gcpid", type=Path, default=None)
    ap.add_argument("--druggability-pdb", type=Path, default=None)
    ap.add_argument("--chunk-pairs", type=int, default=25_000_000)
    args = ap.parse_args()
    df = extract_features(
        arrow_path=args.arrow,
        output_path=args.output,
        binding_sites_path=args.binding_sites,
        kcc_path=args.kcc,
        therm_path=args.therm,
        chunk_pairs=args.chunk_pairs,
    )
    n_res = df.height
    if args.asc_consensus and args.asc_consensus.exists():
        df = _attach_asc_consensus(df, args.asc_consensus, n_res)
    if args.gcpid and args.gcpid.exists():
        df = _attach_gcpid_synergy(df, args.gcpid, n_res)
    if args.druggability_pdb and args.druggability_pdb.exists():
        df = _attach_druggability_pdb(df, args.druggability_pdb, n_res)
    df.write_parquet(args.output, compression="zstd", compression_level=9)
    print(f"[postprocess] re-wrote with ASC+GCPID+druggability: {df.height} × {df.width} cols", flush=True)


if __name__ == "__main__":
    main()
