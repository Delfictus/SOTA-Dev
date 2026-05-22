#!/usr/bin/env python3
"""
PRISM-only feature extractor v5a — unified pipeline.

Critical fixes vs v3:
  - DYNAMIC N_STREAMS (was hardcoded 8; blind_validation+named-drug targets use 20)
  - Sparse stream-id support (B10 has streams [1,2,4,6,8,9,12,13,15,18])

Integrations vs v3:
  - kcc_validation.json: top1_vs_top2_separation, validated_sites, debug stats
  - ensemble_trajectory.json: n_streams_actual, total_spikes, consensus_site_ids
  - ground_truth.json: ligand_centroid (per-residue distance), valid_for_dcc_validation
  - P2Rank residues.csv: per-residue P2Rank score/zscore/probability (auxiliary teacher)
  - All v4 enrichments inline (cross-stream entropy, phase entropy, phasor coherence)
  - Phase manifold subprocess + atlas parser inline (24 additional per-residue cols)

Output: ~350 PRISM-native per-residue columns per target.
"""
from __future__ import annotations
import argparse
import csv
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional, Dict

import numpy as np
import pyarrow as pa
import pyarrow.ipc as ipc
import polars as pl
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts" / "training"))

from prism_native_signal_decoders import (  # noqa
    decode_phz1, decode_acl1,
    attach_phasors, attach_cross_stream_stats, attach_phase_entropy,
    attach_gcpid_groups,
)
from phase_manifold_features import (  # noqa
    run_phase_manifold_ranker, parse_atlas,
)

DEFAULT_MAX_STREAMS = 24
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


def discover_dims(arrow_path: Path):
    """Discover n_residues AND n_streams from Arrow data."""
    reader, _ = open_arrow(arrow_path)
    max_rid = -1
    max_sid = -1
    for i in range(reader.num_record_batches):
        batch = reader.get_batch(i)
        nr_col = batch.column("nearby_residues")
        try:
            arr = nr_col.values.to_numpy(zero_copy_only=False)
        except Exception:
            arr = nr_col.flatten().to_numpy(zero_copy_only=False)
        if arr.size:
            local_max = int(arr.max())
            if local_max > max_rid:
                max_rid = local_max
        sid = batch.column("stream_id").to_numpy(zero_copy_only=False)
        if sid.size:
            local_smax = int(sid.max())
            if local_smax > max_sid:
                max_sid = local_smax
    return max_rid + 1, max_sid + 1


def gpu_moments(rid_t, vals_t, n_res):
    vals_t = vals_t.to(torch.float32)
    finite = torch.isfinite(vals_t)
    if not finite.all():
        rid_t = rid_t[finite]
        vals_t = vals_t[finite]
    v2_ = vals_t * vals_t
    v3_ = v2_ * vals_t
    v4_ = v2_ * v2_
    n = torch.zeros(n_res, dtype=torch.float64, device=rid_t.device)
    s1 = torch.zeros_like(n); s2 = torch.zeros_like(n); s3 = torch.zeros_like(n); s4 = torch.zeros_like(n)
    n.scatter_add_(0, rid_t, torch.ones_like(vals_t, dtype=torch.float64))
    s1.scatter_add_(0, rid_t, vals_t.to(torch.float64))
    s2.scatter_add_(0, rid_t, v2_.to(torch.float64))
    s3.scatter_add_(0, rid_t, v3_.to(torch.float64))
    s4.scatter_add_(0, rid_t, v4_.to(torch.float64))
    nz = torch.clamp(n, min=1.0)
    mean = s1 / nz
    e2 = s2 / nz; e3 = s3 / nz; e4 = s4 / nz
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
    }


def gpu_hist_2d(rid_t, cat_t, n_res, n_cat):
    cat_t = torch.clamp(cat_t.to(torch.long), 0, n_cat - 1)
    flat = rid_t * n_cat + cat_t
    hist = torch.zeros(n_res * n_cat, dtype=torch.float32, device=rid_t.device)
    hist.scatter_add_(0, flat, torch.ones_like(flat, dtype=torch.float32))
    return hist.view(n_res, n_cat).to(torch.int64).cpu().numpy()


def gpu_phase_bit_counts(rid_t, phase_bits_t, n_res):
    pb = phase_bits_t.to(torch.int64)
    out = torch.zeros(n_res, N_PHASE_BITS, dtype=torch.float32, device=rid_t.device)
    for bit in range(N_PHASE_BITS):
        bit_val = ((pb >> bit) & 1).to(torch.float32)
        out[:, bit].scatter_add_(0, rid_t, bit_val)
    return out.to(torch.int64).cpu().numpy()


def merge_moments(a, b):
    n_a = a["n"].astype(np.float64); n_b = b["n"].astype(np.float64)
    n_ab = n_a + n_b
    nz = np.where(n_ab > 0, n_ab, 1.0)
    M2_a = a["var"] * np.maximum(a["n"] - 1, 1); M2_b = b["var"] * np.maximum(b["n"] - 1, 1)
    delta = b["mean"] - a["mean"]
    new_mean = np.where(n_ab > 0, (n_a * a["mean"] + n_b * b["mean"]) / nz, 0.0)
    new_M2 = M2_a + M2_b + delta * delta * n_a * n_b / nz
    new_var = np.where(n_ab > 1, new_M2 / np.maximum(n_ab - 1, 1), 0.0)
    return {
        "n": n_ab.astype(np.float32),
        "mean": new_mean.astype(np.float32),
        "var": new_var.astype(np.float32),
        "std": np.sqrt(np.maximum(new_var, 0.0)).astype(np.float32),
        "skew": np.where(n_a > n_b, a["skew"], b["skew"]).astype(np.float32),
        "kurt": np.where(n_a > n_b, a["kurt"], b["kurt"]).astype(np.float32),
    }


def aggregate_spikes(arrow_path: Path, n_res: int, n_streams: int, chunk_pairs: int = 25_000_000):
    """GPU-accelerated per-residue aggregation from Arrow file."""
    moments_accum = {k: None for k in ("intensity", "ve", "wd", "wdc", "burial", "nsd")}
    n_spikes_total = np.zeros(n_res, dtype=np.int64)
    n_spikes_excited = np.zeros(n_res, dtype=np.int64)
    n_spikes_per_stream = np.zeros((n_res, n_streams), dtype=np.int64)
    spike_source_hist = np.zeros((n_res, N_SPIKE_SOURCES), dtype=np.int64)
    aromatic_hist = np.zeros((n_res, N_AROMATIC_TYPES), dtype=np.int64)
    ccns_hist = np.zeros((n_res, N_CCNS_PHASES), dtype=np.int64)
    bg_class_hist = np.zeros((n_res, N_BG_CLASSES), dtype=np.int64)
    burial_bin_hist = np.zeros((n_res, N_BURIAL_BINS), dtype=np.int64)
    ipct_bin_hist = np.zeros((n_res, N_IPCT_BINS), dtype=np.int64)
    wavelength_hist = np.zeros((n_res, N_WAVELENGTH_BINS), dtype=np.int64)
    time_window_hist = np.zeros((n_res, N_TIME_WINDOWS), dtype=np.int64)
    phase_bit_counts = np.zeros((n_res, N_PHASE_BITS), dtype=np.int64)

    reader, _ = open_arrow(arrow_path)
    for batch in (reader.get_batch(i) for i in range(reader.num_record_batches)):
        sb_n = batch.num_rows
        if sb_n == 0:
            continue

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
        for lo in range(0, n_pairs, chunk_pairs):
            hi = min(lo + chunk_pairs, n_pairs)
            rid_chunk = rid[lo:hi]; sidx_chunk = spike_idx[lo:hi]
            rid_t = torch.from_numpy(rid_chunk.astype(np.int64)).to(DEVICE)

            for feat_name, feat_vals in (
                ("intensity", intensity), ("ve", ve), ("wd", wd),
                ("wdc", wdc), ("burial", burial), ("nsd", nearest_site_dist),
            ):
                vals_t = torch.from_numpy(feat_vals[sidx_chunk].astype(np.float32, copy=False)).to(DEVICE)
                m = gpu_moments(rid_t, vals_t, n_res)
                moments_accum[feat_name] = m if moments_accum[feat_name] is None else merge_moments(moments_accum[feat_name], m)

            n_spikes_total += np.bincount(rid_chunk, minlength=n_res)
            excited_mask = n_nearby_excited[sidx_chunk] > 0
            n_spikes_excited += np.bincount(rid_chunk[excited_mask], minlength=n_res)

            n_spikes_per_stream += gpu_hist_2d(rid_t, torch.from_numpy(stream_id[sidx_chunk].astype(np.int64)).to(DEVICE), n_res, n_streams)
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

    return {
        "moments": moments_accum,
        "n_spikes_total": n_spikes_total,
        "n_spikes_excited": n_spikes_excited,
        "n_spikes_per_stream": n_spikes_per_stream,
        "spike_source_hist": spike_source_hist,
        "aromatic_hist": aromatic_hist,
        "ccns_hist": ccns_hist,
        "bg_class_hist": bg_class_hist,
        "burial_bin_hist": burial_bin_hist,
        "ipct_bin_hist": ipct_bin_hist,
        "wavelength_hist": wavelength_hist,
        "time_window_hist": time_window_hist,
        "phase_bit_counts": phase_bit_counts,
    }


def assemble_df(agg: dict, n_res: int, n_streams: int) -> pl.DataFrame:
    cols = {"residue_id": np.arange(n_res, dtype=np.int32)}
    cols["n_spikes"] = agg["n_spikes_total"]
    cols["n_spikes_excited"] = agg["n_spikes_excited"]
    for feat_name, m in agg["moments"].items():
        if m is None:
            for stat in ("n", "mean", "var", "std", "skew", "kurt"):
                cols[f"{feat_name}_{stat}"] = np.zeros(n_res, dtype=np.float32)
        else:
            for stat in ("n", "mean", "var", "std", "skew", "kurt"):
                cols[f"{feat_name}_{stat}"] = m[stat]
    for s in range(n_streams):
        cols[f"n_spikes_stream_{s}"] = agg["n_spikes_per_stream"][:, s]
    for s in range(N_SPIKE_SOURCES):
        cols[f"src_hist_{s}"] = agg["spike_source_hist"][:, s]
    for a in range(N_AROMATIC_TYPES):
        cols[f"aromatic_hist_{a}"] = agg["aromatic_hist"][:, a]
    for c in range(N_CCNS_PHASES):
        cols[f"ccns_hist_{c}"] = agg["ccns_hist"][:, c]
    for b in range(N_BG_CLASSES):
        cols[f"bg_class_hist_{b}"] = agg["bg_class_hist"][:, b]
    for b in range(N_BURIAL_BINS):
        cols[f"burial_bin_{b}"] = agg["burial_bin_hist"][:, b]
    for b in range(N_IPCT_BINS):
        cols[f"ipct_bin_{b}"] = agg["ipct_bin_hist"][:, b]
    for b in range(N_WAVELENGTH_BINS):
        cols[f"wavelength_bin_{b}"] = agg["wavelength_hist"][:, b]
    for b in range(N_TIME_WINDOWS):
        cols[f"timewin_bin_{b}"] = agg["time_window_hist"][:, b]
    for b in range(N_PHASE_BITS):
        cols[f"phase_bit_{b}_count"] = agg["phase_bit_counts"][:, b]
    nz = np.maximum(agg["n_spikes_total"].astype(np.float32), 1.0)
    cols["excited_fraction"] = agg["n_spikes_excited"].astype(np.float32) / nz
    for s in range(n_streams):
        cols[f"stream_fraction_{s}"] = agg["n_spikes_per_stream"][:, s].astype(np.float32) / nz
    cols["phase_popcount_mean"] = agg["phase_bit_counts"].sum(axis=1).astype(np.float32) / nz
    return pl.DataFrame(cols)


def attach_kcc(df, kcc_path, n_res):
    with open(kcc_path) as f:
        kcc = json.load(f)
    residues = kcc.get("residues", []) if isinstance(kcc, dict) else []
    keys = ("kcc_score", "active_causal_steps", "burst_motion", "causal_lag",
            "direction_score", "lag_corr_peak", "local_cov", "motion_efficiency",
            "sum_motion", "total_steps")
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
            pass
        try:
            pdb_resid[idx] = int(r.get("residue_id", -1))
        except (TypeError, ValueError):
            pass

    df = df.with_columns(
        [pl.Series(k, cols[k]) for k in keys]
        + [pl.Series("ca_x", ca_pos[:, 0]),
           pl.Series("ca_y", ca_pos[:, 1]),
           pl.Series("ca_z", ca_pos[:, 2]),
           pl.Series("net_dx_norm", net_dx_norm),
           pl.Series("pdb_resid", pdb_resid)]
    )

    sites = kcc.get("sites", []) if isinstance(kcc, dict) else []
    if sites and ca_pos.any():
        site_keys = ("kcc_confidence", "temporal_corr", "site_burst_motion", "site_causal_lag",
                     "site_direction_score", "site_lag_corr_peak", "site_local_cov", "site_motion_efficiency")
        centroids, vals = [], {k: [] for k in site_keys}
        gtck, driver = [], []
        for s in sites:
            c = s.get("centroid")
            if not c or len(c) != 3:
                continue
            centroids.append([float(c[0]), float(c[1]), float(c[2])])
            kd = s.get("kcc", {}) if isinstance(s.get("kcc"), dict) else {}
            for k in site_keys:
                v = kd.get(k, 0.0)
                try:
                    vals[k].append(float(v) if not isinstance(v, (list, dict)) else 0.0)
                except (TypeError, ValueError):
                    vals[k].append(0.0)
            try:
                gtck.append(float(s.get("gtck_rank", 0.0)))
            except (TypeError, ValueError):
                gtck.append(0.0)
            try:
                driver.append(int(kd.get("driver_residue_id", -1)))
            except (TypeError, ValueError):
                driver.append(-1)
        if centroids:
            cs = np.array(centroids, dtype=np.float32)
            diff = ca_pos[:, None, :] - cs[None, :, :]
            nearest = (diff * diff).sum(axis=2).argmin(axis=1)
            new_cols = [pl.Series("nearest_site_gtck", np.array(gtck, dtype=np.float32)[nearest])]
            for k in site_keys:
                new_cols.append(pl.Series(f"nearest_{k}", np.array(vals[k], dtype=np.float32)[nearest]))
            df = df.with_columns(new_cols)

    return df, ca_pos


def attach_therm(df, therm_path, n_res, ca_positions):
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

    cls_map = {"INERT": 1, "Cryptic": 2, "Druggable": 5, "Orthosteric": 3, "Surface": 4}
    target_stats = {
        "target_sdst_event_count": int(therm.get("sdst_event_count", 0)),
        "target_total_pockets": int(therm.get("total_pockets", 0)),
        "target_cryptic_pockets": int(therm.get("cryptic_pockets", 0)),
        "target_tide_residues_mapped": int(therm.get("tide_residues_mapped", 0)),
    }

    pdb_to_topo = {}
    if "pdb_resid" in df.columns:
        for topo_i, pdb_r in enumerate(df["pdb_resid"].to_numpy()):
            if 0 <= topo_i < n_res and int(pdb_r) >= 0:
                pdb_to_topo[int(pdb_r)] = topo_i

    if pockets and ca_positions.any():
        centroids, tau, dr, hy, cls_v, pid, crypt = [], [], [], [], [], [], []
        for p in pockets:
            c = p.get("centroid")
            if not c or len(c) != 3:
                continue
            centroids.append([float(c[0]), float(c[1]), float(c[2])])
            tau.append(float(p.get("ccns_tau", 0.0)))
            dr.append(float(p.get("druggability_score", 0.0)))
            hy.append(float(p.get("hysteresis_asymmetry", 0.0)))
            cls_v.append(cls_map.get(p.get("therm_class", ""), 0))
            pid.append(int(p.get("pocket_id", -1)))
            crypt.append(1 if p.get("is_cryptic") else 0)
            for tr in (p.get("top_residues") or []):
                if not isinstance(tr, dict):
                    continue
                try:
                    rid = int(tr.get("residue_id", -1))
                    te = float(tr.get("transfer_entropy", 0.0))
                except (TypeError, ValueError):
                    continue
                topo_i = rid if 0 <= rid < n_res else pdb_to_topo.get(rid)
                if topo_i is None:
                    continue
                if te > max_te[topo_i]:
                    max_te[topo_i] = te
                sum_te[topo_i] += te
                pocket_top_count[topo_i] += 1
        if centroids:
            cs = np.array(centroids, dtype=np.float32)
            dist = np.sqrt(np.maximum(((ca_positions[:, None, :] - cs[None, :, :]) ** 2).sum(axis=2), 0.0))
            nearest = dist.argmin(axis=1)
            therm_class[:] = np.array(cls_v, dtype=np.int8)[nearest]
            is_cryptic[:] = np.array(crypt, dtype=np.int8)[nearest]
            ccns_tau[:] = np.array(tau, dtype=np.float32)[nearest]
            druggability[:] = np.array(dr, dtype=np.float32)[nearest]
            hysteresis_asym[:] = np.array(hy, dtype=np.float32)[nearest]
            nearest_pocket_id[:] = np.array(pid, dtype=np.int32)[nearest]
            nearest_pocket_dist[:] = dist[np.arange(n_res), nearest]

    df = df.with_columns([
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
    for k, v in target_stats.items():
        df = df.with_columns(pl.Series(k, np.full(n_res, v, dtype=np.int32)))
    return df


def attach_asc_consensus(df, asc_path, n_res):
    s_pc = np.zeros(n_res, dtype=np.float32)
    n_groups = np.zeros(n_res, dtype=np.int8)
    try:
        with open(asc_path) as f:
            asc = json.load(f)
    except Exception:
        return df.with_columns([
            pl.Series("asc_s_pc", s_pc),
            pl.Series("asc_n_groups", n_groups),
            pl.Series("asc_in_consensus", n_groups),
        ])
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


def attach_gcpid(df, gcpid_path, n_res):
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


def attach_druggability_pdb(df, dpdb_path, n_res):
    drug = np.zeros(n_res, dtype=np.float32)
    seen = np.zeros(n_res, dtype=np.int8)
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
                    seen[topo_i] = 1
    except Exception:
        pass
    return df.with_columns([
        pl.Series("residue_druggability_pdb", drug),
        pl.Series("residue_druggability_seen", seen),
    ])


def attach_ground_truth(df, gt_path, n_res, ca_positions):
    """ground_truth.json: per-residue distance to ligand_centroid + target metadata."""
    dist_to_lig = np.full(n_res, -1.0, dtype=np.float32)
    in_contact_5A = np.zeros(n_res, dtype=np.int8)
    in_contact_8A = np.zeros(n_res, dtype=np.int8)
    has_gt = 0; valid_dcc = 0; lig_natoms = 0
    try:
        with open(gt_path) as f:
            gt = json.load(f)
        has_gt = 1
        valid_dcc = 1 if gt.get("valid_for_dcc_validation") else 0
        lig = gt.get("ligand") or {}
        lig_natoms = int(lig.get("n_atoms", 0))
        centroid = gt.get("ligand_centroid")
        if centroid and len(centroid) == 3 and ca_positions.any():
            c = np.array(centroid, dtype=np.float32)
            diff = ca_positions - c[None, :]
            d = np.sqrt(np.maximum((diff * diff).sum(axis=1), 0.0))
            dist_to_lig = d.astype(np.float32)
            in_contact_5A = (d < 5.0).astype(np.int8)
            in_contact_8A = (d < 8.0).astype(np.int8)
    except Exception:
        pass
    return df.with_columns([
        pl.Series("gt_dist_to_ligand", dist_to_lig),
        pl.Series("gt_in_contact_5A", in_contact_5A),
        pl.Series("gt_in_contact_8A", in_contact_8A),
        pl.Series("gt_has_ground_truth", np.full(n_res, has_gt, dtype=np.int8)),
        pl.Series("gt_valid_for_dcc", np.full(n_res, valid_dcc, dtype=np.int8)),
        pl.Series("gt_ligand_n_atoms", np.full(n_res, lig_natoms, dtype=np.int32)),
    ])


def attach_ensemble_trajectory(df, et_path, n_res):
    """ensemble_trajectory.json: per-target stream stats."""
    n_streams_actual = 0
    n_consensus_sites = 0
    total_spikes = 0
    n_streams_healthy = 0
    try:
        with open(et_path) as f:
            et = json.load(f)
        n_streams_actual = int(et.get("n_streams", 0))
        n_consensus_sites = int(et.get("n_consensus_sites", 0))
        total_spikes = int(et.get("total_spikes", 0))
        for ps in et.get("per_stream", []):
            if isinstance(ps, dict) and not ps.get("error"):
                n_streams_healthy += 1
    except Exception:
        pass
    return df.with_columns([
        pl.Series("target_n_streams_actual", np.full(n_res, n_streams_actual, dtype=np.int32)),
        pl.Series("target_n_consensus_sites", np.full(n_res, n_consensus_sites, dtype=np.int32)),
        pl.Series("target_total_spikes", np.full(n_res, total_spikes, dtype=np.int64)),
        pl.Series("target_n_streams_healthy", np.full(n_res, n_streams_healthy, dtype=np.int32)),
    ])


def attach_kcc_validation(df, kv_path, n_res):
    """kcc_validation.json: per-target validation stats + per-site topk_residues."""
    top1_vs_top2_sep = 0.0
    n_validated_sites = 0
    n_residues_tracked = 0
    n_residues_with_causal = 0
    top_k_in_validated = np.zeros(n_res, dtype=np.int8)
    try:
        with open(kv_path) as f:
            kv = json.load(f)
        gc = kv.get("global_checks") or {}
        top1_vs_top2_sep = float(gc.get("top1_vs_top2_separation", 0.0))
        n_validated_sites = int(gc.get("n_validated_sites", 0))
        dbg = kv.get("debug") or {}
        n_residues_tracked = int(dbg.get("n_residues_tracked", 0))
        n_residues_with_causal = int(dbg.get("n_residues_with_causal", 0))
        for s in kv.get("sites", []):
            for rid in (s.get("topk_residues") or []):
                try:
                    ri = int(rid)
                    if 0 <= ri < n_res:
                        top_k_in_validated[ri] = 1
                except (TypeError, ValueError):
                    continue
    except Exception:
        pass
    return df.with_columns([
        pl.Series("kv_top1_vs_top2_separation", np.full(n_res, top1_vs_top2_sep, dtype=np.float32)),
        pl.Series("kv_n_validated_sites", np.full(n_res, n_validated_sites, dtype=np.int32)),
        pl.Series("kv_n_residues_tracked", np.full(n_res, n_residues_tracked, dtype=np.int32)),
        pl.Series("kv_n_residues_with_causal", np.full(n_res, n_residues_with_causal, dtype=np.int32)),
        pl.Series("kv_in_validated_topk", top_k_in_validated),
    ])


def attach_p2rank(df, p2_path, n_res):
    """P2Rank residues.csv: per-residue score/zscore/probability (auxiliary teacher)."""
    score = np.zeros(n_res, dtype=np.float32)
    zscore = np.zeros(n_res, dtype=np.float32)
    prob = np.zeros(n_res, dtype=np.float32)
    pocket = np.zeros(n_res, dtype=np.int8)
    has_p2 = 0
    try:
        with open(p2_path) as f:
            reader = csv.reader(f)
            header = next(reader)
            header = [h.strip() for h in header]
            for row in reader:
                row = [c.strip() for c in row]
                rec = dict(zip(header, row))
                try:
                    rid = int(rec.get("residue_label", -1))
                except (TypeError, ValueError):
                    continue
                topo_i = rid - 1
                if not (0 <= topo_i < n_res):
                    continue
                try:
                    score[topo_i] = float(rec.get("score", 0.0))
                    zscore[topo_i] = float(rec.get("zscore", 0.0))
                    prob[topo_i] = float(rec.get("probability", 0.0))
                    pocket[topo_i] = int(rec.get("pocket", 0)) if (rec.get("pocket") or "0").isdigit() else 0
                except (TypeError, ValueError):
                    pass
        has_p2 = 1
    except Exception:
        pass
    return df.with_columns([
        pl.Series("p2rank_score", score),
        pl.Series("p2rank_zscore", zscore),
        pl.Series("p2rank_probability", prob),
        pl.Series("p2rank_pocket", pocket),
        pl.Series("p2rank_has_data", np.full(n_res, has_p2, dtype=np.int8)),
    ])


def attach_acl1_target_stats(df, acl_path, n_res):
    """ACL1 binary contains (chunk_idx, ratio) time series. Aggregate to per-target stats."""
    mean = 0.0; mx = 0.0; mn = 0.0; std = 0.0; n_chunks = 0
    try:
        rid_to_c = decode_acl1(acl_path)
        vals = np.array(list(rid_to_c.values()), dtype=np.float32)
        if vals.size > 0:
            mean = float(vals.mean()); mx = float(vals.max()); mn = float(vals.min())
            std = float(vals.std()); n_chunks = int(vals.size)
    except Exception:
        pass
    return df.with_columns([
        pl.Series("target_acl_mean", np.full(n_res, mean, dtype=np.float32)),
        pl.Series("target_acl_max", np.full(n_res, mx, dtype=np.float32)),
        pl.Series("target_acl_min", np.full(n_res, mn, dtype=np.float32)),
        pl.Series("target_acl_std", np.full(n_res, std, dtype=np.float32)),
        pl.Series("target_acl_n_chunks", np.full(n_res, n_chunks, dtype=np.int32)),
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
    ap.add_argument("--phasors", type=Path, default=None)
    ap.add_argument("--acl-contrast", type=Path, default=None)
    ap.add_argument("--ensemble-trajectory", type=Path, default=None)
    ap.add_argument("--kcc-validation", type=Path, default=None)
    ap.add_argument("--ground-truth", type=Path, default=None)
    ap.add_argument("--p2rank-residues", type=Path, default=None)
    ap.add_argument("--phase-manifold-outdir", type=Path, default=None,
                    help="If provided, run phase_manifold_ranker.py into this dir and parse")
    ap.add_argument("--phase-manifold-script", type=Path,
                    default=Path("scripts/phase_manifold_ranker.py"))
    ap.add_argument("--chunk-pairs", type=int, default=25_000_000)
    args = ap.parse_args()

    t0 = time.time()
    print(f"[v5] {args.arrow} → {args.output}", flush=True)
    print(f"[v5] device={DEVICE}", flush=True)

    n_res, n_streams = discover_dims(args.arrow)
    print(f"[v5] n_res={n_res}  n_streams={n_streams}", flush=True)

    agg = aggregate_spikes(args.arrow, n_res, n_streams, args.chunk_pairs)
    df = assemble_df(agg, n_res, n_streams)
    print(f"[v5] base aggregation done in {time.time()-t0:.1f}s, cols={df.width}", flush=True)

    ca_positions = np.zeros((n_res, 3), dtype=np.float32)
    if args.kcc and args.kcc.exists():
        df, ca_positions = attach_kcc(df, args.kcc, n_res)
    if args.binding_sites and args.binding_sites.exists():
        pass
    if args.therm and args.therm.exists():
        df = attach_therm(df, args.therm, n_res, ca_positions)
    if args.asc_consensus and args.asc_consensus.exists():
        df = attach_asc_consensus(df, args.asc_consensus, n_res)
    if args.gcpid and args.gcpid.exists():
        df = attach_gcpid(df, args.gcpid, n_res)
    if args.druggability_pdb and args.druggability_pdb.exists():
        df = attach_druggability_pdb(df, args.druggability_pdb, n_res)

    df = attach_cross_stream_stats(df, n_res, n_streams=n_streams)
    df = attach_phase_entropy(df, n_res)
    if args.gcpid and args.gcpid.exists():
        df = attach_gcpid_groups(df, args.gcpid, n_res)
    if args.phasors and args.phasors.exists():
        df = attach_phasors(df, args.phasors, n_res)
    if args.acl_contrast and args.acl_contrast.exists():
        df = attach_acl1_target_stats(df, args.acl_contrast, n_res)

    if args.ensemble_trajectory and args.ensemble_trajectory.exists():
        df = attach_ensemble_trajectory(df, args.ensemble_trajectory, n_res)
    if args.kcc_validation and args.kcc_validation.exists():
        df = attach_kcc_validation(df, args.kcc_validation, n_res)
    if args.ground_truth and args.ground_truth.exists():
        df = attach_ground_truth(df, args.ground_truth, n_res, ca_positions)
    if args.p2rank_residues and args.p2rank_residues.exists():
        df = attach_p2rank(df, args.p2rank_residues, n_res)

    if args.phase_manifold_outdir and args.kcc and args.binding_sites:
        outdir = args.phase_manifold_outdir
        if not (outdir / "ranked_site_manifest_atlas.json").exists():
            print(f"[v5] running phase_manifold_ranker → {outdir}", flush=True)
            ok = run_phase_manifold_ranker(args.arrow, args.binding_sites, args.kcc, outdir,
                                            args.phase_manifold_script)
            if not ok:
                print(f"[v5] phase_manifold_ranker failed", flush=True)
        pm_df = parse_atlas(outdir, n_res)
        df = df.join(pm_df.drop("residue_id") if "residue_id" in pm_df.columns else pm_df,
                     left_on="residue_id", right_index=True, how="left") if False else df.with_columns(
            [pm_df[c] for c in pm_df.columns if c != "residue_id"]
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(args.output, compression="zstd", compression_level=9)
    elapsed = time.time() - t0
    spikes_seen = int(agg["n_spikes_total"].sum())
    rate = spikes_seen / max(elapsed, 0.001)
    print(f"[v5] wrote {args.output} ({args.output.stat().st_size/1e6:.2f} MB) "
          f"— {df.height} × {df.width} cols, {spikes_seen:,} residue-pairs @ {rate/1e6:.2f}M/s "
          f"in {elapsed:.1f}s", flush=True)


if __name__ == "__main__":
    main()
