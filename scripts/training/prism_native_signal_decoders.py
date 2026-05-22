#!/usr/bin/env python3
"""
PRISM-native binary signal decoders + cross-stream / phase analytics.

These functions extract truly PRISM-unique signals that no other engine produces:
  - PHZ1 phasors.bin: complex-valued per-(residue,stream-group) phasors with
    pairwise coherence — interferometric measurement of phase-coupling between
    bisimulation pair groups (scout=[0,2], observer=[1,3])
  - ACL1 acl_contrast.bin: per-residue allosteric contrast scalar
  - Per-residue cross-stream entropy / dominant-stream stats from existing
    per-stream histograms
  - Per-residue phase-bit entropy / Shannon information
  - GCPID observer/scout group membership flags
"""
from __future__ import annotations
import struct
import math
import json
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import polars as pl


def decode_phz1(path: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """Returns (real, imag, count, n_streams).

    Shapes: real, imag, count = (n_residues, n_streams)
    """
    data = path.read_bytes()
    magic = data[:4]
    if magic != b"PHZ1":
        raise ValueError(f"Bad PHZ1 magic: {magic}")
    n_fields = struct.unpack_from("<I", data, 4)[0]
    n_residues = struct.unpack_from("<I", data, 8)[0]
    real = np.zeros((n_residues, n_fields), dtype=np.float64)
    imag = np.zeros((n_residues, n_fields), dtype=np.float64)
    count = np.zeros((n_residues, n_fields), dtype=np.int64)
    PHASOR_BYTES = 20
    offset = 12
    for r in range(n_residues):
        for f in range(n_fields):
            pos = offset + r * n_fields * PHASOR_BYTES + f * PHASOR_BYTES
            re, im = struct.unpack_from("<dd", data, pos)
            c = struct.unpack_from("<I", data, pos + 16)[0]
            real[r, f] = re
            imag[r, f] = im
            count[r, f] = c
    return real, imag, count, n_fields


def decode_acl1(path: Path) -> dict:
    """Returns dict: residue_id (int) → contrast (float)."""
    data = path.read_bytes()
    magic = data[:4]
    if magic != b"ACL1":
        raise ValueError(f"Bad ACL1 magic: {magic}")
    n_entries = struct.unpack_from("<I", data, 4)[0]
    offset = 12
    out = {}
    for i in range(n_entries):
        pos = offset + i * 8
        rid = struct.unpack_from("<I", data, pos)[0]
        c = struct.unpack_from("<f", data, pos + 4)[0]
        out[int(rid)] = float(c)
    return out


def attach_phasors(df: pl.DataFrame, phz_path: Path, n_res: int) -> pl.DataFrame:
    """Extract per-residue phasor features from PHZ1 binary.

    Per-stream-group features (4 groups × multiple stats) plus pairwise coherence
    across groups (the interferometric quantity). Phasor groups likely align with
    GCPID's scout=[0,2] / observer=[1,3] bisimulation pair structure.
    """
    try:
        real, imag, count, n_fields = decode_phz1(phz_path)
    except Exception:
        return df

    mag = np.sqrt(real * real + imag * imag)
    phase = np.arctan2(imag, real)

    cols = {}
    for f in range(n_fields):
        cols[f"phasor_mag_{f}"] = mag[:n_res, f].astype(np.float32) if n_res <= mag.shape[0] else np.zeros(n_res, dtype=np.float32)
        cols[f"phasor_phase_{f}"] = phase[:n_res, f].astype(np.float32) if n_res <= phase.shape[0] else np.zeros(n_res, dtype=np.float32)
        cols[f"phasor_count_{f}"] = count[:n_res, f].astype(np.int64) if n_res <= count.shape[0] else np.zeros(n_res, dtype=np.int64)

    mag_trim = mag[:n_res] if mag.shape[0] >= n_res else np.zeros((n_res, n_fields))
    real_trim = real[:n_res] if real.shape[0] >= n_res else np.zeros((n_res, n_fields))
    imag_trim = imag[:n_res] if imag.shape[0] >= n_res else np.zeros((n_res, n_fields))

    for i in range(n_fields):
        for j in range(i + 1, n_fields):
            denom = mag_trim[:, i] * mag_trim[:, j]
            cross_real = real_trim[:, i] * real_trim[:, j] + imag_trim[:, i] * imag_trim[:, j]
            cross_imag = imag_trim[:, i] * real_trim[:, j] - real_trim[:, i] * imag_trim[:, j]
            cross_mag = np.sqrt(cross_real * cross_real + cross_imag * cross_imag)
            coh = np.where(denom > 1e-12, cross_mag / np.maximum(denom, 1e-12), 0.0)
            phase_diff = np.where(denom > 1e-12, np.arctan2(cross_imag, cross_real), 0.0)
            cols[f"phasor_coherence_{i}{j}"] = coh.astype(np.float32)
            cols[f"phasor_phase_diff_{i}{j}"] = phase_diff.astype(np.float32)

    mean_mag = mag_trim.mean(axis=1)
    sum_count = count[:n_res].sum(axis=1) if count.shape[0] >= n_res else np.zeros(n_res, dtype=np.int64)
    cols["phasor_mean_mag"] = mean_mag.astype(np.float32)
    cols["phasor_total_count"] = sum_count.astype(np.int64)
    if n_fields >= 2:
        cols["phasor_scout_observer_coherence"] = (
            (cols.get("phasor_coherence_01", np.zeros(n_res)) + cols.get("phasor_coherence_23", np.zeros(n_res))) / 2
        ).astype(np.float32)

    return df.with_columns([pl.Series(k, v) for k, v in cols.items()])


def attach_acl1_contrast(df: pl.DataFrame, acl_path: Path, n_res: int) -> pl.DataFrame:
    """ACL1 binary contains a sparse list of (residue_id, contrast) entries —
    high-allosteric-contrast residues only. ID convention: topology index.
    """
    contrast = np.zeros(n_res, dtype=np.float32)
    seen = np.zeros(n_res, dtype=np.int8)
    try:
        rid_to_c = decode_acl1(acl_path)
        for rid, c in rid_to_c.items():
            if 0 <= rid < n_res:
                contrast[rid] = c
                seen[rid] = 1
    except Exception:
        pass
    return df.with_columns([
        pl.Series("acl_contrast", contrast),
        pl.Series("acl_high_contrast", seen),
    ])


def attach_cross_stream_stats(df: pl.DataFrame, n_res: int, n_streams: int = 8) -> pl.DataFrame:
    """Per-residue: how concentrated/uniform is the spike distribution across
    8 streams? Computed from existing n_spikes_stream_N columns.

    - stream_entropy: Shannon entropy of stream distribution per residue
    - stream_dominant_id: which stream produced the most spikes
    - stream_max_fraction: dominant stream's share
    - effective_n_streams: exp(entropy) — effective number of contributing streams
    - scout_observer_consensus: (mean(stream_0,2) - mean(stream_1,3)) — bisimulation pair contrast
    """
    cols_present = [f"n_spikes_stream_{s}" for s in range(n_streams) if f"n_spikes_stream_{s}" in df.columns]
    if not cols_present:
        return df

    arr = np.stack([df[c].to_numpy().astype(np.float64) for c in cols_present], axis=1)
    total = arr.sum(axis=1)
    probs = arr / np.maximum(total[:, None], 1.0)
    entropy = -np.where(probs > 0, probs * np.log(np.maximum(probs, 1e-12)), 0.0).sum(axis=1)
    dominant = arr.argmax(axis=1).astype(np.int32)
    max_frac = arr.max(axis=1) / np.maximum(total, 1.0)
    eff_streams = np.exp(entropy)

    scout_mean = np.zeros(n_res, dtype=np.float64)
    obs_mean = np.zeros(n_res, dtype=np.float64)
    n_scout = sum(1 for s in (0, 2) if f"n_spikes_stream_{s}" in df.columns)
    n_obs = sum(1 for s in (1, 3) if f"n_spikes_stream_{s}" in df.columns)
    if n_scout > 0:
        for s in (0, 2):
            if f"n_spikes_stream_{s}" in df.columns:
                scout_mean += df[f"n_spikes_stream_{s}"].to_numpy().astype(np.float64)
        scout_mean /= n_scout
    if n_obs > 0:
        for s in (1, 3):
            if f"n_spikes_stream_{s}" in df.columns:
                obs_mean += df[f"n_spikes_stream_{s}"].to_numpy().astype(np.float64)
        obs_mean /= n_obs

    pair_contrast = (scout_mean - obs_mean) / np.maximum(scout_mean + obs_mean, 1.0)

    return df.with_columns([
        pl.Series("stream_entropy", entropy.astype(np.float32)),
        pl.Series("stream_dominant_id", dominant),
        pl.Series("stream_max_fraction", max_frac.astype(np.float32)),
        pl.Series("effective_n_streams", eff_streams.astype(np.float32)),
        pl.Series("scout_mean_spikes", scout_mean.astype(np.float32)),
        pl.Series("observer_mean_spikes", obs_mean.astype(np.float32)),
        pl.Series("scout_observer_contrast", pair_contrast.astype(np.float32)),
    ])


def attach_phase_entropy(df: pl.DataFrame, n_res: int) -> pl.DataFrame:
    """Per-residue Shannon entropy over (a) the 32 phase_bit_N_count columns
    and (b) the 8 ccns_hist_N columns. Indicates how spread the phase / phase-bit
    activation is — concentrated = strong site signal, uniform = noise.
    """
    bit_cols = [f"phase_bit_{b}_count" for b in range(32) if f"phase_bit_{b}_count" in df.columns]
    if bit_cols:
        bits = np.stack([df[c].to_numpy().astype(np.float64) for c in bit_cols], axis=1)
        total = bits.sum(axis=1)
        p = bits / np.maximum(total[:, None], 1.0)
        bit_entropy = -np.where(p > 0, p * np.log(np.maximum(p, 1e-12)), 0.0).sum(axis=1)
    else:
        bit_entropy = np.zeros(n_res, dtype=np.float64)

    ccns_cols = [f"ccns_hist_{c}" for c in range(8) if f"ccns_hist_{c}" in df.columns]
    if ccns_cols:
        ccns = np.stack([df[c].to_numpy().astype(np.float64) for c in ccns_cols], axis=1)
        total = ccns.sum(axis=1)
        p = ccns / np.maximum(total[:, None], 1.0)
        ccns_entropy = -np.where(p > 0, p * np.log(np.maximum(p, 1e-12)), 0.0).sum(axis=1)
        ccns_dominant = ccns.argmax(axis=1).astype(np.int32)
        ccns_max_frac = ccns.max(axis=1) / np.maximum(total, 1.0)
    else:
        ccns_entropy = np.zeros(n_res, dtype=np.float64)
        ccns_dominant = np.zeros(n_res, dtype=np.int32)
        ccns_max_frac = np.zeros(n_res, dtype=np.float64)

    return df.with_columns([
        pl.Series("phase_bit_entropy", bit_entropy.astype(np.float32)),
        pl.Series("ccns_phase_entropy", ccns_entropy.astype(np.float32)),
        pl.Series("ccns_dominant_phase", ccns_dominant),
        pl.Series("ccns_max_fraction", ccns_max_frac.astype(np.float32)),
    ])


def attach_gcpid_groups(df: pl.DataFrame, gcpid_path: Path, n_res: int) -> pl.DataFrame:
    """Add binary indicators for membership in GCPID observer/scout groups.

    These are stream-group ids (0/2 = scout, 1/3 = observer typically). NOT
    per-residue group membership — at PRISM-4D level the groups index stream
    bundles, not individual residues. But we expose them as scalar metadata
    so the student knows the bisimulation structure.
    """
    try:
        with open(gcpid_path) as f:
            g = json.load(f)
    except Exception:
        g = {}
    obs = g.get("observer_groups", [])
    scout = g.get("scout_groups", [])
    n_obs_groups = len(obs)
    n_scout_groups = len(scout)
    n_streams_total = int(g.get("n_streams", 0))
    n_with_pid = int(g.get("n_residues_with_pid", 0))

    return df.with_columns([
        pl.Series("gcpid_n_observer_groups", np.full(n_res, n_obs_groups, dtype=np.int8)),
        pl.Series("gcpid_n_scout_groups", np.full(n_res, n_scout_groups, dtype=np.int8)),
        pl.Series("gcpid_total_streams", np.full(n_res, n_streams_total, dtype=np.int8)),
        pl.Series("gcpid_has_pid", (df["gcpid_n_samples"].to_numpy() > 0).astype(np.int8) if "gcpid_n_samples" in df.columns else np.zeros(n_res, dtype=np.int8)),
    ])


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet", type=Path, required=True, help="existing v3 parquet to enrich")
    ap.add_argument("--phasors", type=Path, default=None)
    ap.add_argument("--acl-contrast", type=Path, default=None)
    ap.add_argument("--gcpid", type=Path, default=None)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()

    df = pl.read_parquet(args.parquet)
    n = df.height
    df = attach_cross_stream_stats(df, n)
    df = attach_phase_entropy(df, n)
    if args.gcpid and args.gcpid.exists():
        df = attach_gcpid_groups(df, args.gcpid, n)
    if args.phasors and args.phasors.exists():
        df = attach_phasors(df, args.phasors, n)
    if args.acl_contrast and args.acl_contrast.exists():
        df = attach_acl1_contrast(df, args.acl_contrast, n)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(args.output, compression="zstd", compression_level=9)
    print(f"enriched: {df.height} × {df.width} cols → {args.output}")
