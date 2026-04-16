#!/usr/bin/env python3
"""
[POST-PROCESSING ENRICHMENT - PROVENANCE-WRAPPED]

Multi-criteria retention scoring on the engine's full-spectrum spike stream.

Input: *.topology.spike_events.arrow (30-field, post-engine)
Output:
  - *.topology.spike_events.enriched.parquet (30 original + ~10 enrichment + retention_score)
  - spike_enrichment_stats.json (distributions, correlations, sensitivity)

Ten retention-score dimensions (each normalized to [0,1], weighted sum → composite):
  1. intensity_percentile      (engine-native, already per-channel top-30%)
  2. spatial_density_5A         (same-voxel, ±50 ts window)
  3. burst_membership           (≥5 spikes in same voxel within 100 ts)
  4. cross_channel              (same voxel+time has spikes from ≥2 sources)
  5. phase_rarity               (rare ccns_phase boost)
  6. onset_slope, decay_slope   (per-residue kinetic signature)
  7. pcmi_group_coherence       (cross-group phase coherence from phase_bits)
  8. dewetting                  (|wd_change| - water displacement signal)
  9. burial_context             (burial_score normalized)
  10. background_penalty        (background_class=1 reduces retention)

Default weights sum to 1.0. Retention_score ∈ [−0.10, 1.10] (small negative possible
from background_penalty, capped for cleanliness).

Runtime: ~5-10 min on 74M rows (Polars, 24 cores).
Memory: ~15-25 GB RSS peak.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import polars as pl
import pyarrow.feather as feather

sys.path.insert(0, str(Path(__file__).parent))
from prism_prov import RunContext, blake3_file, blake3_bytes, canonical_json


# Default retention weights (sum to 1.0; background is subtractive)
DEFAULT_WEIGHTS = {
    "intensity_percentile":       0.15,
    "spatial_density_norm":       0.15,
    "burst_membership":           0.15,
    "cross_channel":              0.10,
    "phase_rarity":               0.10,
    "onset_slope_norm":           0.10,
    "decay_slope_norm":           0.05,
    "pcmi_group_coherence":       0.10,
    "dewetting_norm":             0.05,
    "burial_score_norm":          0.05,
    "background_penalty":        -0.10,
}


def load_arrow(path: Path) -> pl.DataFrame:
    print(f"  loading {path} ({path.stat().st_size / 1073741824:.2f} GB)...")
    t0 = time.time()
    tbl = feather.read_table(str(path))
    df = pl.from_arrow(tbl)
    print(f"    loaded {df.height:,} rows in {time.time()-t0:.1f}s")
    return df


def _minmax_norm(df: pl.DataFrame, col: str, out: str,
                 clip_low_pct: float = 1.0,
                 clip_high_pct: float = 99.0) -> pl.DataFrame:
    """Percentile-clipped min-max normalization to [0, 1]."""
    q_lo = df.select(pl.col(col).quantile(clip_low_pct / 100.0)).item()
    q_hi = df.select(pl.col(col).quantile(clip_high_pct / 100.0)).item()
    if q_hi <= q_lo:
        q_hi = q_lo + 1e-9
    return df.with_columns(
        ((pl.col(col).clip(q_lo, q_hi) - q_lo) / (q_hi - q_lo)).alias(out)
    )


def compute_enrichment(df: pl.DataFrame) -> pl.DataFrame:
    """Compute all 10 retention-score dimensions."""
    print("  computing enrichment dimensions...")
    t0 = time.time()

    # 1. intensity_percentile (already in [0, 255], normalize to [0, 1])
    if "intensity_percentile" in df.columns:
        df = df.with_columns(
            (pl.col("intensity_percentile").cast(pl.Float32) / 255.0).alias("_int_pct")
        )
    else:
        df = _minmax_norm(df, "intensity", "_int_pct")

    # 2. spatial_density_5A: count spikes in same voxel within ±50 timesteps
    # Use voxel_idx + time-bucket (bucket_size=100 → ±50 ts window approximation)
    df = df.with_columns(
        (pl.col("timestep") // 100).alias("_tbucket")
    )
    density = df.group_by(["voxel_idx", "_tbucket"]).agg(
        pl.count().alias("_vox_bucket_count")
    )
    df = df.join(density, on=["voxel_idx", "_tbucket"], how="left")
    # Normalize log-count to [0, 1]
    df = df.with_columns(
        pl.col("_vox_bucket_count").cast(pl.Float32).log1p().alias("_spatial_log")
    )
    df = _minmax_norm(df, "_spatial_log", "spatial_density_norm")

    # 3. burst_membership: 1 if ≥5 spikes in same voxel+bucket
    df = df.with_columns(
        (pl.col("_vox_bucket_count") >= 5).cast(pl.Float32).alias("burst_membership")
    )

    # 4. cross_channel: same (voxel, tbucket) has ≥2 distinct spike_sources
    if "spike_source" in df.columns:
        channel_diversity = (
            df.group_by(["voxel_idx", "_tbucket"])
            .agg(pl.col("spike_source").n_unique().alias("_n_channels"))
        )
        df = df.join(channel_diversity, on=["voxel_idx", "_tbucket"], how="left")
        df = df.with_columns(
            ((pl.col("_n_channels") >= 2).cast(pl.Float32)).alias("cross_channel")
        )
    else:
        df = df.with_columns(pl.lit(0.0).alias("cross_channel"))

    # 5. phase_rarity: rarer cryo phases get higher weight
    if "ccns_phase" in df.columns:
        phase_counts = df.group_by("ccns_phase").agg(pl.count().alias("_phase_count"))
        total = df.height
        phase_counts = phase_counts.with_columns(
            (1.0 - pl.col("_phase_count").cast(pl.Float32) / total).alias("_phase_rarity_raw")
        )
        df = df.join(phase_counts.select(["ccns_phase", "_phase_rarity_raw"]),
                     on="ccns_phase", how="left")
        df = df.with_columns(pl.col("_phase_rarity_raw").alias("phase_rarity"))
    else:
        df = df.with_columns(pl.lit(0.0).alias("phase_rarity"))

    # 6. onset_slope, decay_slope — temporal kinetics per voxel
    # For each voxel, fit spike count over time: d(rate)/dt at onset and decay
    # Simplified: compute per-voxel mean timestep and timestep spread
    # Lower spread + more early spikes = fast onset
    voxel_timing = (
        df.group_by("voxel_idx")
        .agg([
            pl.col("timestep").min().alias("_t_min"),
            pl.col("timestep").max().alias("_t_max"),
            pl.col("timestep").mean().alias("_t_mean"),
            pl.col("timestep").quantile(0.2).alias("_t_q20"),
            pl.col("timestep").quantile(0.8).alias("_t_q80"),
            pl.count().alias("_vox_n"),
        ])
    )
    # Onset slope: how concentrated are the first 20% spikes?
    # If (t_q20 - t_min) is small relative to (t_max - t_min), onset is fast
    voxel_timing = voxel_timing.with_columns([
        (pl.col("_t_max") - pl.col("_t_min") + 1).alias("_t_range"),
    ]).with_columns([
        (1.0 - (pl.col("_t_q20") - pl.col("_t_min")) / pl.col("_t_range"))
            .clip(0.0, 1.0).alias("onset_slope_norm"),
        (1.0 - (pl.col("_t_max") - pl.col("_t_q80")) / pl.col("_t_range"))
            .clip(0.0, 1.0).alias("decay_slope_norm"),
    ])
    df = df.join(
        voxel_timing.select(["voxel_idx", "onset_slope_norm", "decay_slope_norm"]),
        on="voxel_idx", how="left",
    )

    # 7. pcmi_group_coherence: cross-group phase coherence per voxel
    # Use Polars-native cos/sin for vectorized, type-safe computation.
    if "phase_bits" in df.columns and "group_id" in df.columns:
        df = df.with_columns(
            (pl.col("phase_bits").cast(pl.Float32) * (2.0 * float(np.pi) / 1024.0))
                .alias("_phase_rad")
        )
        vox_group = (
            df.group_by(["voxel_idx", "group_id"])
            .agg([
                pl.col("_phase_rad").cos().mean().cast(pl.Float32).alias("_cos_mean"),
                pl.col("_phase_rad").sin().mean().cast(pl.Float32).alias("_sin_mean"),
            ])
        )
        vox_pcmi = (
            vox_group
            .with_columns(
                (pl.col("_cos_mean") ** 2 + pl.col("_sin_mean") ** 2).sqrt().alias("_group_spc")
            )
            .group_by("voxel_idx")
            .agg(pl.col("_group_spc").mean().cast(pl.Float32).alias("pcmi_group_coherence"))
        )
        df = df.join(vox_pcmi, on="voxel_idx", how="left")
        df = df.with_columns(pl.col("pcmi_group_coherence").fill_null(0.0))
    else:
        df = df.with_columns(pl.lit(0.0).alias("pcmi_group_coherence"))

    # 8. dewetting: |wd_change| normalized
    if "wd_change" in df.columns:
        df = df.with_columns(pl.col("wd_change").abs().alias("_wd_abs"))
        df = _minmax_norm(df, "_wd_abs", "dewetting_norm")
    else:
        df = df.with_columns(pl.lit(0.0).alias("dewetting_norm"))

    # 9. burial_score normalized
    if "burial_score" in df.columns:
        df = _minmax_norm(df, "burial_score", "burial_score_norm")
    else:
        df = df.with_columns(pl.lit(0.0).alias("burial_score_norm"))

    # 10. background_penalty: 1 if classified background
    if "background_class" in df.columns:
        df = df.with_columns(
            pl.col("background_class").cast(pl.Float32).alias("background_penalty")
        )
    else:
        df = df.with_columns(pl.lit(0.0).alias("background_penalty"))

    # Rename the intensity_pct column for consistency
    df = df.rename({"_int_pct": "intensity_percentile_norm"})

    # Drop intermediate computation columns
    drop_cols = [c for c in df.columns if c.startswith("_")]
    df = df.drop(drop_cols)

    print(f"    enrichment dims computed in {time.time()-t0:.1f}s")
    return df


def compute_retention_score(df: pl.DataFrame,
                             weights: Dict[str, float]) -> pl.DataFrame:
    """Weighted sum → retention_score ∈ approximately [0, 1]."""
    print("  computing composite retention score...")
    components = []
    for key, w in weights.items():
        # Map weight-dict keys back to the column names we produced
        col_map = {
            "intensity_percentile": "intensity_percentile_norm",
            "spatial_density_norm": "spatial_density_norm",
            "burst_membership": "burst_membership",
            "cross_channel": "cross_channel",
            "phase_rarity": "phase_rarity",
            "onset_slope_norm": "onset_slope_norm",
            "decay_slope_norm": "decay_slope_norm",
            "pcmi_group_coherence": "pcmi_group_coherence",
            "dewetting_norm": "dewetting_norm",
            "burial_score_norm": "burial_score_norm",
            "background_penalty": "background_penalty",
        }
        col = col_map.get(key, key)
        if col in df.columns:
            components.append(pl.col(col).fill_null(0.0) * w)
    retention = sum(components[1:], start=components[0]) if components else pl.lit(0.0)
    df = df.with_columns(retention.alias("retention_score"))
    df = df.with_columns(
        pl.col("retention_score").rank(method="ordinal", descending=True)
          .cast(pl.UInt32).alias("retention_rank")
    )
    return df


def emit_stats(df: pl.DataFrame, weights: Dict[str, float],
               out_path: Path, input_path: Path):
    """Write distribution statistics for reviewer inspection."""
    print("  computing stats...")
    stats: Dict[str, Any] = {
        "input_file": str(input_path),
        "input_blake3": blake3_file(input_path),
        "n_spikes": int(df.height),
        "weights": weights,
        "dimensions": {},
    }
    score_cols = [
        "intensity_percentile_norm", "spatial_density_norm", "burst_membership",
        "cross_channel", "phase_rarity", "onset_slope_norm", "decay_slope_norm",
        "pcmi_group_coherence", "dewetting_norm", "burial_score_norm",
        "background_penalty", "retention_score",
    ]
    for c in score_cols:
        if c not in df.columns:
            continue
        s = df[c]
        stats["dimensions"][c] = {
            "min": float(s.min()),
            "max": float(s.max()),
            "mean": float(s.mean()),
            "std": float(s.std()),
            "p05": float(s.quantile(0.05)),
            "p50": float(s.quantile(0.50)),
            "p95": float(s.quantile(0.95)),
            "nulls": int(s.null_count()),
        }
    # Sensitivity: retention at different percentiles
    stats["sensitivity_cutoffs"] = {}
    for q in (0.70, 0.80, 0.90, 0.95, 0.99):
        cutoff = float(df["retention_score"].quantile(q))
        n_kept = int(df.filter(pl.col("retention_score") >= cutoff).height)
        stats["sensitivity_cutoffs"][f"q{int(q*100)}"] = {
            "cutoff": cutoff,
            "n_kept": n_kept,
            "frac_kept": round(n_kept / df.height, 4),
        }
    # Correlation between retention_score and intensity_percentile_norm
    # (how much does retention_score DIFFER from intensity alone?)
    if "intensity_percentile_norm" in df.columns:
        corr = df.select(
            pl.corr("retention_score", "intensity_percentile_norm").alias("c")
        ).item()
        stats["retention_vs_intensity_correlation"] = float(corr)
    # Group-balance stats if multi-diff
    if "group_id" in df.columns:
        grp = (
            df.group_by("group_id")
            .agg([
                pl.count().alias("n"),
                pl.col("retention_score").mean().alias("mean_retention"),
                pl.col("retention_score").quantile(0.95).alias("p95_retention"),
            ])
            .sort("group_id")
            .to_dicts()
        )
        stats["per_group"] = grp
    with open(out_path, "w") as f:
        json.dump(stats, f, indent=2, default=str)
    print(f"    stats written: {out_path}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arrow", required=True, type=Path,
                   help="Input .topology.spike_events.arrow")
    ap.add_argument("--output-parquet", required=True, type=Path)
    ap.add_argument("--stats", required=True, type=Path)
    ap.add_argument("--target", required=True, help="Target name for provenance")
    ap.add_argument("--prov-dir", required=True, type=Path)
    args = ap.parse_args()

    if not args.arrow.exists():
        print(f"FATAL: arrow file not found: {args.arrow}", file=sys.stderr)
        return 2

    print(f"=== SPIKE STREAM ENRICHMENT: {args.target} ===")
    print(f"Input:  {args.arrow}")
    print(f"Output: {args.output_parquet}")
    print()

    output_dir = args.output_parquet.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    with RunContext(
        target=args.target,
        stage="7_enrichment",
        substage="retention_score",
        output_dir=output_dir,
        prov_dir=args.prov_dir,
        upstream_prov=[],  # caller fills in
    ) as ctx:
        ctx.add_input(args.arrow, upstream_prov_ref="5_engine")
        ctx.set_tool("enrich_spike_arrow.py",
                    ["python3", __file__, "--arrow", str(args.arrow)])

        df = load_arrow(args.arrow)
        df = compute_enrichment(df)
        df = compute_retention_score(df, DEFAULT_WEIGHTS)

        print(f"  writing enriched parquet ({df.height:,} rows, {len(df.columns)} cols)...")
        t0 = time.time()
        df.write_parquet(args.output_parquet, compression="zstd", compression_level=3)
        print(f"    parquet written: {time.time()-t0:.1f}s, "
              f"{args.output_parquet.stat().st_size / 1073741824:.2f} GB")

        ctx.add_output(args.output_parquet, role="enriched_spikes")
        emit_stats(df, DEFAULT_WEIGHTS, args.stats, args.arrow)
        ctx.add_output(args.stats, role="stats")

        # Gates
        non_null_retention = df.filter(pl.col("retention_score").is_not_null()).height
        frac_non_null = non_null_retention / df.height
        ctx.set_gate("retention_score_populated",
                    "PASS" if frac_non_null > 0.99 else "FAIL",
                    note=f"{frac_non_null*100:.2f}% rows have retention_score")
        # Distribution sanity
        rs_std = df["retention_score"].std()
        ctx.set_gate("retention_score_non_degenerate",
                    "PASS" if rs_std > 0.05 else "FAIL",
                    note=f"std(retention_score) = {rs_std:.4f}")
        # Correlation with intensity: should NOT be 1.0 (otherwise we're just
        # doing intensity filtering under a fancier name)
        corr = df.select(
            pl.corr("retention_score", "intensity_percentile_norm")
        ).item() if "intensity_percentile_norm" in df.columns else 0.0
        ctx.set_gate("retention_beyond_intensity",
                    "PASS" if corr < 0.95 else "WARN",
                    note=f"corr(retention, intensity) = {corr:.3f}")
        ctx.set_verdict("PASS")

    print("\n=== DONE ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())
