#!/usr/bin/env python3
"""
PRISM-only feature extractor v1 — streaming Arrow → per-residue PRISM-native tensor.

Goal: maximize soft-label signal density per residue from the 30-column TWIN Arrow
schema without any sequence-priors (no ESM). Every spike feeds up to 8 residues via
the engine's pre-computed `nearby_residues` field — no naive nearest-Cα assignment.

Output: per-target parquet with one row per residue and ~200 PRISM-native columns.
"""
from __future__ import annotations
import argparse
import json
import sys
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


class WelfordOnline:
    """Welford 4-moment streaming aggregator, vectorized per residue.

    Maintains n, mean, M2, M3, M4 for each residue. Updates by integrating
    each (residue, spike_value) pair using the vectorized merge step.
    """

    def __init__(self, n_residues: int, dtype=np.float64):
        self.n = np.zeros(n_residues, dtype=np.int64)
        self.mean = np.zeros(n_residues, dtype=dtype)
        self.M2 = np.zeros(n_residues, dtype=dtype)
        self.M3 = np.zeros(n_residues, dtype=dtype)
        self.M4 = np.zeros(n_residues, dtype=dtype)
        self.max = np.full(n_residues, -np.inf, dtype=dtype)
        self.min = np.full(n_residues, np.inf, dtype=dtype)

    def update_batch(self, residue_ids: np.ndarray, values: np.ndarray):
        """Update per-residue moments with a batch of (residue, value) pairs.

        Uses per-element Welford. Vectorized by sorting and unique-grouping.
        """
        if residue_ids.size == 0:
            return

        order = np.argsort(residue_ids, kind="stable")
        rid_s = residue_ids[order]
        val_s = values[order].astype(self.mean.dtype, copy=False)

        unique_rids, starts = np.unique(rid_s, return_index=True)
        ends = np.append(starts[1:], rid_s.size)

        for rid, lo, hi in zip(unique_rids, starts, ends):
            if rid < 0 or rid >= self.n.size:
                continue
            chunk = val_s[lo:hi]
            self._merge_chunk(int(rid), chunk)

    def _merge_chunk(self, rid: int, chunk: np.ndarray):
        if chunk.size == 0:
            return
        if np.issubdtype(chunk.dtype, np.floating):
            mask = np.isfinite(chunk)
            if not mask.all():
                chunk = chunk[mask]
        m_n = chunk.size
        if m_n == 0:
            return
        m_mean = chunk.mean()
        delta = chunk - m_mean
        m_M2 = (delta**2).sum()
        m_M3 = (delta**3).sum()
        m_M4 = (delta**4).sum()

        n_a = self.n[rid]
        n_b = m_n
        n_ab = n_a + n_b
        if n_a == 0:
            self.n[rid] = n_b
            self.mean[rid] = m_mean
            self.M2[rid] = m_M2
            self.M3[rid] = m_M3
            self.M4[rid] = m_M4
        else:
            d = m_mean - self.mean[rid]
            d2 = d * d
            d3 = d2 * d
            d4 = d2 * d2
            new_M4 = (
                self.M4[rid] + m_M4
                + d4 * n_a * n_b * (n_a * n_a - n_a * n_b + n_b * n_b) / (n_ab**3)
                + 6 * d2 * (n_a * n_a * m_M2 + n_b * n_b * self.M2[rid]) / (n_ab**2)
                + 4 * d * (n_a * m_M3 - n_b * self.M3[rid]) / n_ab
            )
            new_M3 = (
                self.M3[rid] + m_M3
                + d3 * n_a * n_b * (n_a - n_b) / (n_ab**2)
                + 3 * d * (n_a * m_M2 - n_b * self.M2[rid]) / n_ab
            )
            new_M2 = self.M2[rid] + m_M2 + d2 * n_a * n_b / n_ab
            new_mean = self.mean[rid] + d * n_b / n_ab
            self.n[rid] = n_ab
            self.mean[rid] = new_mean
            self.M2[rid] = new_M2
            self.M3[rid] = new_M3
            self.M4[rid] = new_M4

        self.max[rid] = max(self.max[rid], float(chunk.max()))
        self.min[rid] = min(self.min[rid], float(chunk.min()))

    def finalize(self) -> dict:
        var = np.where(self.n > 1, self.M2 / np.maximum(self.n - 1, 1), 0.0)
        std = np.sqrt(np.maximum(var, 0.0))
        skew = np.where(
            (self.n > 2) & (std > 1e-12),
            (np.sqrt(self.n) * self.M3) / np.maximum(self.M2**1.5, 1e-12),
            0.0,
        )
        kurt = np.where(
            (self.n > 3) & (self.M2 > 1e-12),
            (self.n * self.M4) / np.maximum(self.M2**2, 1e-12) - 3.0,
            0.0,
        )
        return {
            "n": self.n,
            "mean": self.mean,
            "var": var,
            "std": std,
            "skew": skew,
            "kurt": kurt,
            "max": np.where(np.isinf(self.max), 0.0, self.max),
            "min": np.where(np.isinf(self.min), 0.0, self.min),
        }


def open_arrow(path: Path):
    """Open Arrow IPC file (RecordBatchFile preferred, fallback to Stream)."""
    src = pa.memory_map(str(path), "r")
    try:
        return ipc.RecordBatchFileReader(src), "file"
    except Exception:
        src.seek(0)
        return ipc.RecordBatchStreamReader(src), "stream"


def discover_n_residues(arrow_path: Path, topology_path: Optional[Path]) -> int:
    """Find N_residues by either topology JSON or a fast pass over nearby_residues."""
    if topology_path and topology_path.exists():
        with open(topology_path) as f:
            topo = json.load(f)
        for key in ("residues", "atoms_per_residue", "n_residues"):
            if key in topo:
                if key == "n_residues":
                    return int(topo[key])
                return len(topo[key])
    reader, _ = open_arrow(arrow_path)
    max_rid = -1
    if hasattr(reader, "num_record_batches"):
        for i in range(reader.num_record_batches):
            batch = reader.get_batch(i)
            nr = batch.column("nearby_residues").flatten().to_numpy()
            local_max = int(nr.max()) if nr.size else -1
            if local_max > max_rid:
                max_rid = local_max
    else:
        for batch in reader:
            nr = batch.column("nearby_residues").flatten().to_numpy()
            local_max = int(nr.max()) if nr.size else -1
            if local_max > max_rid:
                max_rid = local_max
    return max_rid + 1


def extract_features(
    arrow_path: Path,
    output_path: Path,
    topology_path: Optional[Path] = None,
    binding_sites_path: Optional[Path] = None,
    kcc_path: Optional[Path] = None,
    therm_path: Optional[Path] = None,
    batch_rows: int = 1_000_000,
    use_full_batches: bool = True,
):
    t0 = time.time()
    print(f"[{time.time() - t0:6.1f}s] extracting from {arrow_path}", flush=True)
    print(f"[{time.time() - t0:6.1f}s]   size: {arrow_path.stat().st_size / 1e9:.2f} GB", flush=True)

    print(f"[{time.time() - t0:6.1f}s] discovering residue count", flush=True)
    n_res = discover_n_residues(arrow_path, topology_path)
    print(f"[{time.time() - t0:6.1f}s]   N_residues = {n_res}", flush=True)

    intensity_w = WelfordOnline(n_res)
    ve_w = WelfordOnline(n_res)
    wd_w = WelfordOnline(n_res)
    wd_change_w = WelfordOnline(n_res)
    burial_w = WelfordOnline(n_res)
    nearest_site_dist_w = WelfordOnline(n_res)

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
    phase_popcount_sum = np.zeros(n_res, dtype=np.int64)
    phase_popcount_sq = np.zeros(n_res, dtype=np.int64)

    site_assignment_top10 = {}

    reader, mode = open_arrow(arrow_path)
    print(f"[{time.time() - t0:6.1f}s]   Arrow mode: {mode}", flush=True)
    if hasattr(reader, "num_record_batches"):
        batch_iter = (reader.get_batch(i) for i in range(reader.num_record_batches))
        n_batches = reader.num_record_batches
    else:
        batch_iter = iter(reader)
        n_batches = None

    spikes_seen = 0
    for bi, batch in enumerate(batch_iter):
        n = batch.num_rows
        if n == 0:
            continue

        if use_full_batches:
            sub_batches = [batch]
        else:
            sub_batches = []
            for off in range(0, n, batch_rows):
                sub_batches.append(batch.slice(off, min(batch_rows, n - off)))

        for sb in sub_batches:
            sb_n = sb.num_rows
            spikes_seen += sb_n

            nearby = sb.column("nearby_residues").to_numpy(zero_copy_only=False)
            nearby_arr = np.stack([np.asarray(x, dtype=np.int32) for x in nearby])

            n_nearby = sb.column("n_residues").to_numpy(zero_copy_only=False)

            intensity = sb.column("intensity").to_numpy(zero_copy_only=False)
            ve = sb.column("vibrational_energy").to_numpy(zero_copy_only=False)
            wd = sb.column("water_density").to_numpy(zero_copy_only=False)
            wd_change = sb.column("wd_change").to_numpy(zero_copy_only=False)
            wavelength = sb.column("wavelength_nm").to_numpy(zero_copy_only=False)
            burial = sb.column("burial_score").to_numpy(zero_copy_only=False)
            stream_id = sb.column("stream_id").to_numpy(zero_copy_only=False)
            spike_source = sb.column("spike_source").to_numpy(zero_copy_only=False)
            aromatic_type = sb.column("aromatic_type").to_numpy(zero_copy_only=False)
            ccns_phase = sb.column("ccns_phase").to_numpy(zero_copy_only=False)
            phase_bits = sb.column("phase_bits").to_numpy(zero_copy_only=False)
            n_nearby_excited = sb.column("n_nearby_excited").to_numpy(zero_copy_only=False)
            ipct = sb.column("intensity_percentile").to_numpy(zero_copy_only=False)
            bg_class = sb.column("background_class").to_numpy(zero_copy_only=False)
            site_id = sb.column("site_id").to_numpy(zero_copy_only=False)
            nearest_site_dist = sb.column("nearest_site_dist").to_numpy(zero_copy_only=False)
            frame_index = sb.column("frame_index").to_numpy(zero_copy_only=False)

            spike_idx = np.repeat(np.arange(sb_n, dtype=np.int32), 8)
            residue_flat = nearby_arr.reshape(-1)
            slot_idx = np.tile(np.arange(8, dtype=np.int32), sb_n)
            valid = (residue_flat >= 0) & (slot_idx < np.repeat(n_nearby, 8))
            spike_idx = spike_idx[valid]
            residue_flat = residue_flat[valid]

            intensity_w.update_batch(residue_flat, intensity[spike_idx])
            ve_w.update_batch(residue_flat, ve[spike_idx])
            wd_w.update_batch(residue_flat, wd[spike_idx])
            wd_change_w.update_batch(residue_flat, wd_change[spike_idx])
            burial_w.update_batch(residue_flat, burial[spike_idx])
            nearest_site_dist_w.update_batch(residue_flat, nearest_site_dist[spike_idx])

            np.add.at(n_spikes_total, residue_flat, 1)
            excited_mask = n_nearby_excited[spike_idx] > 0
            np.add.at(n_spikes_excited, residue_flat[excited_mask], 1)

            s_per = np.clip(stream_id[spike_idx].astype(np.int32), 0, N_STREAMS - 1)
            np.add.at(n_spikes_per_stream, (residue_flat, s_per), 1)

            sh = np.clip(spike_source[spike_idx].astype(np.int32), 0, N_SPIKE_SOURCES - 1)
            np.add.at(spike_source_hist, (residue_flat, sh), 1)

            ah = np.clip(aromatic_type[spike_idx].astype(np.int32), 0, N_AROMATIC_TYPES - 1)
            np.add.at(aromatic_hist, (residue_flat, ah), 1)

            ch = np.clip(ccns_phase[spike_idx].astype(np.int32), 0, N_CCNS_PHASES - 1)
            np.add.at(ccns_hist, (residue_flat, ch), 1)

            bh = np.clip(bg_class[spike_idx].astype(np.int32), 0, N_BG_CLASSES - 1)
            np.add.at(bg_class_hist, (residue_flat, bh), 1)

            burial_b = np.clip((burial[spike_idx] * N_BURIAL_BINS).astype(np.int32), 0, N_BURIAL_BINS - 1)
            np.add.at(burial_bin_hist, (residue_flat, burial_b), 1)

            ipct_b = np.clip(ipct[spike_idx].astype(np.int32) // (100 // N_IPCT_BINS), 0, N_IPCT_BINS - 1)
            np.add.at(ipct_bin_hist, (residue_flat, ipct_b), 1)

            wl_safe = np.where(wavelength[spike_idx] > 0, wavelength[spike_idx], 280.0)
            wl_b = np.clip(((wl_safe - 200.0) / 50.0).astype(np.int32), 0, N_WAVELENGTH_BINS - 1)
            np.add.at(wavelength_hist, (residue_flat, wl_b), 1)

            max_frame = max(int(frame_index.max()), 1) if frame_index.size else 1
            tw_b = np.clip((frame_index[spike_idx].astype(np.int64) * N_TIME_WINDOWS // (max_frame + 1)).astype(np.int32), 0, N_TIME_WINDOWS - 1)
            np.add.at(time_window_hist, (residue_flat, tw_b), 1)

            pb = phase_bits[spike_idx].astype(np.uint32)
            for bit in range(N_PHASE_BITS):
                bit_set = ((pb >> bit) & 1).astype(np.int64)
                np.add.at(phase_bit_counts[:, bit], residue_flat, bit_set)
            popcount = np.zeros_like(pb, dtype=np.int64)
            for bit in range(N_PHASE_BITS):
                popcount += ((pb >> bit) & 1).astype(np.int64)
            np.add.at(phase_popcount_sum, residue_flat, popcount)
            np.add.at(phase_popcount_sq, residue_flat, popcount * popcount)

            for sid_val in np.unique(site_id[spike_idx]):
                if sid_val < 0:
                    continue
                mask = site_id[spike_idx] == sid_val
                rids_for_site = residue_flat[mask]
                for rid in np.unique(rids_for_site):
                    key = (int(rid), int(sid_val))
                    site_assignment_top10[key] = site_assignment_top10.get(key, 0) + int((rids_for_site == rid).sum())

            elapsed = time.time() - t0
            if (bi + 1) % 1 == 0 or spikes_seen >= n:
                rate = spikes_seen / max(elapsed, 0.001)
                print(
                    f"[{elapsed:6.1f}s] batch {bi+1}/{n_batches or '?'} "
                    f"spikes={spikes_seen:,}  rate={rate / 1e6:.2f}M spikes/s",
                    flush=True,
                )

    print(f"[{time.time() - t0:6.1f}s] finalizing moments", flush=True)
    fi_intensity = intensity_w.finalize()
    fi_ve = ve_w.finalize()
    fi_wd = wd_w.finalize()
    fi_wdc = wd_change_w.finalize()
    fi_burial = burial_w.finalize()
    fi_nsd = nearest_site_dist_w.finalize()

    print(f"[{time.time() - t0:6.1f}s] building output dataframe", flush=True)
    cols = {"residue_id": np.arange(n_res, dtype=np.int32)}
    cols["n_spikes"] = n_spikes_total
    cols["n_spikes_excited"] = n_spikes_excited

    for stat in ("n", "mean", "var", "std", "skew", "kurt", "max", "min"):
        cols[f"intensity_{stat}"] = fi_intensity[stat]
        cols[f"ve_{stat}"] = fi_ve[stat]
        cols[f"wd_{stat}"] = fi_wd[stat]
        cols[f"wdc_{stat}"] = fi_wdc[stat]
        cols[f"burial_{stat}"] = fi_burial[stat]
        cols[f"nsd_{stat}"] = fi_nsd[stat]

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

    nz = np.maximum(n_spikes_total, 1)
    cols["phase_popcount_mean"] = phase_popcount_sum / nz
    cols["phase_popcount_var"] = (phase_popcount_sq / nz) - (phase_popcount_sum / nz) ** 2
    cols["excited_fraction"] = n_spikes_excited / nz

    for s in range(N_STREAMS):
        cols[f"stream_fraction_{s}"] = n_spikes_per_stream[:, s] / nz

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
    print(
        f"[{elapsed:6.1f}s] wrote {output_path} ({output_path.stat().st_size / 1e6:.2f} MB) — "
        f"{df.height} residues, {df.width} feature cols, "
        f"{spikes_seen:,} spikes processed at {spikes_seen / elapsed / 1e6:.2f}M/s",
        flush=True,
    )
    return df


def _attach_site_metadata(df: pl.DataFrame, sites, n_res: int, ca_positions: Optional[np.ndarray] = None) -> pl.DataFrame:
    """Attach per-site features. binding_sites.json has 'sites' list with per-site
    centroid + classification + druggability, and 'cryptic_sites' with cryptic-only
    versions. We assign each residue to its nearest site via Cα-centroid distance,
    inheriting the site's druggability + classification.
    """
    site_count = np.zeros(n_res, dtype=np.int32)
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

    site_centroids = []
    site_druggability = []
    site_class = []
    site_ids = []
    for site in site_list:
        if not isinstance(site, dict):
            continue
        c = site.get("centroid") or site.get("center")
        if not c or len(c) != 3:
            continue
        site_centroids.append([float(c[0]), float(c[1]), float(c[2])])
        site_druggability.append(float(site.get("druggability", 0.0)))
        site_class.append(class_map.get(site.get("classification", ""), 0))
        site_ids.append(int(site.get("site_id", -1)))
    if not site_centroids or ca_positions is None or ca_positions.shape[0] == 0:
        return df.with_columns([
            pl.Series("site_count", site_count),
            pl.Series("in_top5_site", in_top_site),
            pl.Series("nearest_site_id_meta", nearest_site_id),
            pl.Series("nearest_site_dist_meta", nearest_site_dist),
            pl.Series("nearest_site_druggability", nearest_site_druggability),
            pl.Series("nearest_site_classification", nearest_site_classification),
        ])

    centroids_arr = np.array(site_centroids, dtype=np.float32)
    drug_arr = np.array(site_druggability, dtype=np.float32)
    class_arr = np.array(site_class, dtype=np.int8)
    sid_arr = np.array(site_ids, dtype=np.int32)

    diff = ca_positions[:, None, :] - centroids_arr[None, :, :]
    dist2 = (diff * diff).sum(axis=2)
    dist = np.sqrt(np.maximum(dist2, 0.0))
    nearest = dist.argmin(axis=1)
    rng = np.arange(min(ca_positions.shape[0], n_res))
    nearest_site_id[rng] = sid_arr[nearest[: rng.size]]
    nearest_site_dist[rng] = dist[rng, nearest[: rng.size]]
    nearest_site_druggability[rng] = drug_arr[nearest[: rng.size]]
    nearest_site_classification[rng] = class_arr[nearest[: rng.size]]
    in_top_site[rng[dist[rng, nearest[: rng.size]] < 8.0]] = 1
    np.add.at(site_count, rng[dist[rng].min(axis=1) < 10.0], 1)

    return df.with_columns([
        pl.Series("site_count", site_count),
        pl.Series("in_top5_site", in_top_site),
        pl.Series("nearest_site_id_meta", nearest_site_id),
        pl.Series("nearest_site_dist_meta", nearest_site_dist),
        pl.Series("nearest_site_druggability", nearest_site_druggability),
        pl.Series("nearest_site_classification", nearest_site_classification),
    ])


def _attach_kcc(df: pl.DataFrame, kcc_path: Path, n_res: int) -> "tuple[pl.DataFrame, np.ndarray]":
    """KCC visualization residues[] is the per-residue feature gold mine: 10+ engineered
    physics-derived signals per residue. Returns updated df + ca_positions (used by
    therm attach to assign residues to pockets).
    """
    with open(kcc_path) as f:
        kcc = json.load(f)
    residues = kcc.get("residues", []) if isinstance(kcc, dict) else []

    keys = (
        "kcc_score", "active_causal_steps", "burst_motion", "causal_lag",
        "direction_score", "lag_corr_peak", "local_cov", "motion_efficiency",
    )
    cols = {k: np.zeros(n_res, dtype=np.float32) for k in keys}
    ca_pos = np.zeros((n_res, 3), dtype=np.float32)
    net_dx_norm = np.zeros(n_res, dtype=np.float32)

    for r in residues:
        if not isinstance(r, dict):
            continue
        ri = r.get("residue_id", r.get("id", r.get("index", -1)))
        try:
            ri = int(ri)
        except (TypeError, ValueError):
            continue
        if not (0 <= ri < n_res):
            continue
        for k in keys:
            v = r.get(k, 0.0)
            try:
                cols[k][ri] = float(v) if not isinstance(v, (list, dict)) else 0.0
            except (TypeError, ValueError):
                cols[k][ri] = 0.0
        cp = r.get("ca_position")
        if isinstance(cp, list) and len(cp) == 3:
            ca_pos[ri] = [float(cp[0]), float(cp[1]), float(cp[2])]
        nd = r.get("net_dx")
        if isinstance(nd, list) and len(nd) == 3:
            net_dx_norm[ri] = float(np.linalg.norm(nd))
        elif isinstance(nd, (int, float)):
            net_dx_norm[ri] = float(nd)

    df = df.with_columns(
        [pl.Series(k, cols[k]) for k in keys]
        + [
            pl.Series("ca_x", ca_pos[:, 0]),
            pl.Series("ca_y", ca_pos[:, 1]),
            pl.Series("ca_z", ca_pos[:, 2]),
            pl.Series("net_dx_norm", net_dx_norm),
        ]
    )

    sites = kcc.get("sites", []) if isinstance(kcc, dict) else []
    n_sites = len(sites)
    if n_sites > 0 and ca_pos.any():
        site_kcc = np.zeros(n_res, dtype=np.float32)
        site_gtck = np.zeros(n_res, dtype=np.float32)
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
            site_kcc = np.array(kcc_vals, dtype=np.float32)[nearest]
            site_gtck = np.array(gtck_vals, dtype=np.float32)[nearest]
        df = df.with_columns([
            pl.Series("nearest_site_kcc", site_kcc),
            pl.Series("nearest_site_gtck", site_gtck),
        ])
    return df, ca_pos


def _attach_therm(df: pl.DataFrame, therm_path: Path, n_res: int, ca_positions: Optional[np.ndarray]) -> pl.DataFrame:
    """prism_therm.json has 'pockets' list with per-pocket therm info. Assign each
    residue to nearest pocket via Cα-centroid distance.
    """
    with open(therm_path) as f:
        therm = json.load(f)
    pockets = therm.get("pockets", []) if isinstance(therm, dict) else []

    therm_class = np.zeros(n_res, dtype=np.int8)
    ccns_tau = np.zeros(n_res, dtype=np.float32)
    nearest_pocket_id = np.full(n_res, -1, dtype=np.int32)
    nearest_pocket_dist = np.full(n_res, 1e6, dtype=np.float32)

    cls_map = {"CRYPTIC": 1, "Cryptic": 1, "cryptic": 1, "ORTHOSTERIC": 2, "Orthosteric": 2, "SURFACE": 3, "Surface": 3, "DRUGGABLE": 4, "Druggable": 4}

    if not pockets or ca_positions is None or not ca_positions.any():
        return df.with_columns([
            pl.Series("therm_class", therm_class),
            pl.Series("pocket_ccns_tau", ccns_tau),
            pl.Series("nearest_pocket_id", nearest_pocket_id),
            pl.Series("nearest_pocket_dist", nearest_pocket_dist),
        ])

    pocket_centroids = []
    pocket_tau = []
    pocket_class = []
    pocket_id = []
    for p in pockets:
        c = p.get("centroid")
        if c and len(c) == 3:
            pocket_centroids.append([float(c[0]), float(c[1]), float(c[2])])
            pocket_tau.append(float(p.get("ccns_tau", 0.0)))
            pocket_class.append(cls_map.get(p.get("therm_class") or p.get("classification") or "", 0))
            pocket_id.append(int(p.get("pocket_id", -1)))

    if pocket_centroids:
        cs = np.array(pocket_centroids, dtype=np.float32)
        diff = ca_positions[:, None, :] - cs[None, :, :]
        dist2 = (diff * diff).sum(axis=2)
        dist = np.sqrt(np.maximum(dist2, 0.0))
        nearest = dist.argmin(axis=1)
        rng = np.arange(min(ca_positions.shape[0], n_res))
        valid = ca_positions[rng].any(axis=1)
        therm_class[rng[valid]] = np.array(pocket_class, dtype=np.int8)[nearest[rng[valid]]]
        ccns_tau[rng[valid]] = np.array(pocket_tau, dtype=np.float32)[nearest[rng[valid]]]
        nearest_pocket_id[rng[valid]] = np.array(pocket_id, dtype=np.int32)[nearest[rng[valid]]]
        nearest_pocket_dist[rng[valid]] = dist[rng[valid], nearest[rng[valid]]]

    return df.with_columns([
        pl.Series("therm_class", therm_class),
        pl.Series("pocket_ccns_tau", ccns_tau),
        pl.Series("nearest_pocket_id", nearest_pocket_id),
        pl.Series("nearest_pocket_dist", nearest_pocket_dist),
    ])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arrow", type=Path, required=True, help="*.topology.spike_events.arrow")
    ap.add_argument("--output", type=Path, required=True, help="output parquet path")
    ap.add_argument("--topology", type=Path, default=None)
    ap.add_argument("--binding-sites", type=Path, default=None)
    ap.add_argument("--kcc", type=Path, default=None)
    ap.add_argument("--therm", type=Path, default=None)
    ap.add_argument("--batch-rows", type=int, default=1_000_000)
    args = ap.parse_args()

    extract_features(
        arrow_path=args.arrow,
        output_path=args.output,
        topology_path=args.topology,
        binding_sites_path=args.binding_sites,
        kcc_path=args.kcc,
        therm_path=args.therm,
        batch_rows=args.batch_rows,
    )


if __name__ == "__main__":
    main()
