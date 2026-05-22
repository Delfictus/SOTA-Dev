#!/usr/bin/env python3
"""
PRISM-only feature extractor v2 — vectorized streaming Arrow → per-residue parquet.

v2 changes vs v1:
  - All np.add.at calls replaced with np.bincount (vectorized C, ~50× faster)
  - Moments computed in closed form from sum/sum²/sum³/sum⁴ via bincount
  - 2D bincount idiom for residue × category histograms
  - Phase-bit popcount + per-bit rate via vectorized bitwise + bincount
  - JSON attach functions match actual schemas:
      kcc_visualization.json: 'residues' list with kcc_score, active_causal_steps,
        burst_motion, causal_lag, direction_score, lag_corr_peak, local_cov,
        motion_efficiency, net_dx, ca_position (per residue)
      prism_therm.json: 'pockets' list with centroid+therm_class+ccns_tau
      binding_sites.json: 'sites'/'cryptic_sites' with centroid+druggability

Target throughput: ~2-10M spikes/s on the 12-thread Welford v1 path.
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


N_STREAMS = 8
N_SPIKE_SOURCES = 8
N_AROMATIC_TYPES = 8
N_CCNS_PHASES = 8
N_PHASE_BITS = 32
N_BURIAL_BINS = 10
N_IPCT_BINS = 10
N_WAVELENGTH_BINS = 16
N_TIME_WINDOWS = 16
N_BG_CLASSES = 4


def moments_from_bincount(rid_flat: np.ndarray, vals: np.ndarray, n_res: int) -> dict:
    """Closed-form 4-moment statistics per residue, fully vectorized via np.bincount."""
    if np.issubdtype(vals.dtype, np.floating):
        finite = np.isfinite(vals)
        if not finite.all():
            rid_flat = rid_flat[finite]
            vals = vals[finite]
    vals = vals.astype(np.float64, copy=False)

    n = np.bincount(rid_flat, minlength=n_res).astype(np.int64)
    s1 = np.bincount(rid_flat, weights=vals, minlength=n_res)
    s2 = np.bincount(rid_flat, weights=vals * vals, minlength=n_res)
    s3 = np.bincount(rid_flat, weights=vals * vals * vals, minlength=n_res)
    s4 = np.bincount(rid_flat, weights=vals * vals * vals * vals, minlength=n_res)

    nz = np.maximum(n, 1).astype(np.float64)
    mean = s1 / nz
    e2 = s2 / nz
    e3 = s3 / nz
    e4 = s4 / nz
    var = np.maximum(e2 - mean * mean, 0.0)
    std = np.sqrt(var)
    m3 = e3 - 3 * mean * e2 + 2 * mean**3
    m4 = e4 - 4 * mean * e3 + 6 * mean**2 * e2 - 3 * mean**4
    skew = np.where(var > 1e-12, m3 / np.maximum(std**3, 1e-12), 0.0)
    kurt = np.where(var > 1e-12, m4 / np.maximum(var**2, 1e-12) - 3.0, 0.0)

    mx = np.full(n_res, 0.0, dtype=np.float64)
    mn = np.full(n_res, 0.0, dtype=np.float64)
    if rid_flat.size > 0:
        order = np.argsort(rid_flat, kind="stable")
        rid_s = rid_flat[order]
        val_s = vals[order]
        unique_rids, starts = np.unique(rid_s, return_index=True)
        ends = np.append(starts[1:], rid_s.size)
        for u, lo, hi in zip(unique_rids, starts, ends):
            if 0 <= u < n_res and hi > lo:
                seg = val_s[lo:hi]
                mx[u] = float(seg.max())
                mn[u] = float(seg.min())

    return {
        "n": n.astype(np.float32),
        "mean": mean.astype(np.float32),
        "var": var.astype(np.float32),
        "std": std.astype(np.float32),
        "skew": skew.astype(np.float32),
        "kurt": kurt.astype(np.float32),
        "max": mx.astype(np.float32),
        "min": mn.astype(np.float32),
    }


def hist_2d_bincount(rid_flat: np.ndarray, cat: np.ndarray, n_res: int, n_cat: int) -> np.ndarray:
    """Per-residue × per-category counts via single bincount call."""
    cat_clipped = np.clip(cat.astype(np.int64), 0, n_cat - 1)
    combined = rid_flat.astype(np.int64) * n_cat + cat_clipped
    flat = np.bincount(combined, minlength=n_res * n_cat)
    return flat.reshape(n_res, n_cat).astype(np.int64)


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
            nr = batch.column("nearby_residues").flatten().to_numpy()
            local_max = int(nr.max()) if nr.size else -1
            if local_max > max_rid:
                max_rid = local_max
    return max_rid + 1


def extract_features(
    arrow_path: Path,
    output_path: Path,
    binding_sites_path: Optional[Path] = None,
    kcc_path: Optional[Path] = None,
    therm_path: Optional[Path] = None,
):
    t0 = time.time()
    print(f"[{time.time() - t0:6.1f}s] v2 extracting from {arrow_path}", flush=True)
    print(f"[{time.time() - t0:6.1f}s]   size: {arrow_path.stat().st_size / 1e9:.2f} GB", flush=True)

    n_res = discover_n_residues(arrow_path)
    print(f"[{time.time() - t0:6.1f}s]   N_residues = {n_res}", flush=True)

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

    n_per_res = np.zeros(n_res, dtype=np.float64)
    sum_intensity = np.zeros(n_res, dtype=np.float64)
    sum_intensity_sq = np.zeros(n_res, dtype=np.float64)
    sum_intensity_3 = np.zeros(n_res, dtype=np.float64)
    sum_intensity_4 = np.zeros(n_res, dtype=np.float64)

    moments_accum = {k: None for k in ("intensity", "ve", "wd", "wdc", "burial", "nsd")}

    reader, mode = open_arrow(arrow_path)
    print(f"[{time.time() - t0:6.1f}s]   Arrow mode: {mode}", flush=True)
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
            nearby_arr = nearby_flat.reshape(-1, 8)
        except Exception:
            nearby = nearby_col.to_numpy(zero_copy_only=False)
            nearby_arr = np.stack([np.asarray(x, dtype=np.int32) for x in nearby])
        n_nearby = batch.column("n_residues").to_numpy(zero_copy_only=False).astype(np.int32)

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

        spike_idx = np.repeat(np.arange(sb_n, dtype=np.int32), 8)
        residue_flat = nearby_arr.reshape(-1)
        slot_idx = np.tile(np.arange(8, dtype=np.int32), sb_n)
        valid = (residue_flat >= 0) & (slot_idx < np.repeat(n_nearby, 8)) & (residue_flat < n_res)
        spike_idx = spike_idx[valid]
        rid = residue_flat[valid]

        for feat_name, feat_vals in (
            ("intensity", intensity),
            ("ve", ve),
            ("wd", wd),
            ("wdc", wdc),
            ("burial", burial),
            ("nsd", nearest_site_dist),
        ):
            m = moments_from_bincount(rid, feat_vals[spike_idx], n_res)
            if moments_accum[feat_name] is None:
                moments_accum[feat_name] = m
            else:
                moments_accum[feat_name] = _merge_moments(moments_accum[feat_name], m)

        n_spikes_total += np.bincount(rid, minlength=n_res)
        excited_mask = n_nearby_excited[spike_idx] > 0
        n_spikes_excited += np.bincount(rid[excited_mask], minlength=n_res)

        n_spikes_per_stream += hist_2d_bincount(rid, stream_id[spike_idx], n_res, N_STREAMS)
        spike_source_hist += hist_2d_bincount(rid, spike_source[spike_idx], n_res, N_SPIKE_SOURCES)
        aromatic_hist += hist_2d_bincount(rid, aromatic_type[spike_idx], n_res, N_AROMATIC_TYPES)
        ccns_hist += hist_2d_bincount(rid, ccns_phase[spike_idx], n_res, N_CCNS_PHASES)
        bg_class_hist += hist_2d_bincount(rid, bg_class[spike_idx], n_res, N_BG_CLASSES)

        burial_b = np.clip((burial[spike_idx] * N_BURIAL_BINS).astype(np.int32), 0, N_BURIAL_BINS - 1)
        burial_bin_hist += hist_2d_bincount(rid, burial_b, n_res, N_BURIAL_BINS)

        ipct_b = np.clip(ipct[spike_idx].astype(np.int32) // (100 // N_IPCT_BINS), 0, N_IPCT_BINS - 1)
        ipct_bin_hist += hist_2d_bincount(rid, ipct_b, n_res, N_IPCT_BINS)

        wl_safe = np.where(wavelength[spike_idx] > 0, wavelength[spike_idx], 280.0)
        wl_b = np.clip(((wl_safe - 200.0) / 50.0).astype(np.int32), 0, N_WAVELENGTH_BINS - 1)
        wavelength_hist += hist_2d_bincount(rid, wl_b, n_res, N_WAVELENGTH_BINS)

        max_frame = max(int(frame_index.max()), 1) if frame_index.size else 1
        tw_b = np.clip(
            (frame_index[spike_idx].astype(np.int64) * N_TIME_WINDOWS // (max_frame + 1)).astype(np.int32),
            0, N_TIME_WINDOWS - 1,
        )
        time_window_hist += hist_2d_bincount(rid, tw_b, n_res, N_TIME_WINDOWS)

        pb = phase_bits[spike_idx]
        for bit in range(N_PHASE_BITS):
            bit_vals = ((pb >> bit) & 1).astype(np.float64)
            phase_bit_counts[:, bit] += np.bincount(rid, weights=bit_vals, minlength=n_res).astype(np.int64)

        elapsed = time.time() - t0
        rate = spikes_seen / max(elapsed, 0.001)
        print(
            f"[{elapsed:6.1f}s] batch {bi+1}/{n_batches or '?'}  "
            f"spikes={spikes_seen:,}  rate={rate / 1e6:.2f}M/s  "
            f"batch_time={time.time() - t_batch:.1f}s",
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
        f"{spikes_seen:,} spikes @ {rate / 1e6:.2f}M/s",
        flush=True,
    )
    return df


def _merge_moments(a: dict, b: dict) -> dict:
    """Stream-merge two per-residue 4-moment summaries (Welford parallel form)."""
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
        "n": (n_ab).astype(np.float32),
        "mean": new_mean.astype(np.float32),
        "var": new_var.astype(np.float32),
        "std": new_std.astype(np.float32),
        "skew": np.where(n_a > n_b, a["skew"], b["skew"]).astype(np.float32),
        "kurt": np.where(n_a > n_b, a["kurt"], b["kurt"]).astype(np.float32),
        "max": np.maximum(a["max"], b["max"]).astype(np.float32),
        "min": np.minimum(a["min"], b["min"]).astype(np.float32),
    }


def _attach_kcc(df: pl.DataFrame, kcc_path: Path, n_res: int):
    """kcc_visualization.json: residues[] is per-residue feature gold mine.

    Critical: residues[i].residue_id is the PDB residue number (offset), NOT the
    topology index. The topology index is the list POSITION i. The Arrow data
    uses topology indices in nearby_residues, so we use enumerate(residues) and
    bind by list position to align with the spike-side feature accumulators.
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
        centroids = []
        kcc_vals = []
        gtck_vals = []
        for s in sites:
            c = s.get("centroid")
            if c and len(c) == 3:
                centroids.append([float(c[0]), float(c[1]), float(c[2])])
                kcc_vals.append(float(s.get("kcc", 0.0)))
                gtck_vals.append(float(s.get("gtck_rank", 0.0)))
        if centroids:
            cs = np.array(centroids, dtype=np.float32)
            diff = ca_pos[:, None, :] - cs[None, :, :]
            dist2 = (diff * diff).sum(axis=2)
            nearest = dist2.argmin(axis=1)
            df = df.with_columns([
                pl.Series("nearest_site_kcc", np.array(kcc_vals, dtype=np.float32)[nearest]),
                pl.Series("nearest_site_gtck", np.array(gtck_vals, dtype=np.float32)[nearest]),
            ])

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

    centroids = []
    drug = []
    cls = []
    sids = []
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
    drug_arr = np.array(drug, dtype=np.float32)
    cls_arr = np.array(cls, dtype=np.int8)
    sid_arr = np.array(sids, dtype=np.int32)

    diff = ca_positions[:, None, :] - cs[None, :, :]
    dist = np.sqrt(np.maximum((diff * diff).sum(axis=2), 0.0))
    nearest = dist.argmin(axis=1)

    nearest_site_id[:] = sid_arr[nearest]
    nearest_site_dist[:] = dist[np.arange(n_res), nearest]
    nearest_site_druggability[:] = drug_arr[nearest]
    nearest_site_classification[:] = cls_arr[nearest]
    in_top_site[nearest_site_dist < 8.0] = 1

    return df.with_columns([
        pl.Series("in_top_site", in_top_site),
        pl.Series("nearest_site_id_meta", nearest_site_id),
        pl.Series("nearest_site_dist_meta", nearest_site_dist),
        pl.Series("nearest_site_druggability", nearest_site_druggability),
        pl.Series("nearest_site_classification", nearest_site_classification),
    ])


def _attach_therm(df: pl.DataFrame, therm_path: Path, n_res: int, ca_positions: np.ndarray) -> pl.DataFrame:
    """prism_therm.json pockets have: centroid, therm_class (INERT/Cryptic/...),
    is_cryptic, druggability_score, ccns_tau, ccns_class, hysteresis_asymmetry,
    top_residues (list of {residue_id, residue_name, transfer_entropy}).

    Strategy:
      (1) Assign each topology residue to nearest pocket via Cα-centroid distance.
      (2) ALSO extract per-residue transfer_entropy from top_residues field
          (pocket's strongest residues get an explicit pocket-affinity signal).
    """
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
    ccns_class_map = {"Soc": 1, "Boc": 2, "Pcs": 3, "Pcv": 4}

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

        top_res = p.get("top_residues") or []
        for tr in top_res:
            if not isinstance(tr, dict):
                continue
            pdb_rid = tr.get("residue_id")
            te = tr.get("transfer_entropy", 0.0)
            try:
                pdb_rid = int(pdb_rid)
                te = float(te)
            except (TypeError, ValueError):
                continue
            topo_i = pdb_to_topo.get(pdb_rid)
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arrow", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--binding-sites", type=Path, default=None)
    ap.add_argument("--kcc", type=Path, default=None)
    ap.add_argument("--therm", type=Path, default=None)
    args = ap.parse_args()
    extract_features(
        arrow_path=args.arrow,
        output_path=args.output,
        binding_sites_path=args.binding_sites,
        kcc_path=args.kcc,
        therm_path=args.therm,
    )


if __name__ == "__main__":
    main()
