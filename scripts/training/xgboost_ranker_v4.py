#!/usr/bin/env python3
"""XGBoost Ranker v4 — trained on pct70 full-site data (not parquet subset).

Fixes the v3 distribution mismatch. v3 was trained on pre-filtered pct95
parquets (~6 sites/target); v4 trains on ALL pct70 sites from full per-site
JSONs + binding_sites.json geometry features (~50+ sites/target).

Prerequisites:
  1. pct70 campaign complete: `post_campaign_analysis.py` has populated D1
     site_features.min_dist_to_ligand + graded_score for every pct70 target.
  2. Worker API accessible at $PRISM_API or default.
  3. xgboost, onnx, onnxmltools, onnxconverter-common installed.

Pipeline:
  1. Fetch pct70 targets from D1 (spike_percentile=70)
  2. For each target, pull site_features joined with corrected_dcc
  3. Use graded_score as label (already computed in D1)
  4. Validate feature distributions vs expected pct70 ranges
  5. LOTO evaluation (rank:ndcg)
  6. Train final model on all pct70 data
  7. Export ONNX, report SR@k

Usage:
  python3 scripts/training/xgboost_ranker_v4.py
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import xgboost as xgb

from cluster_split import target_to_cluster_map

API_BASE = os.environ.get("PRISM_API", "https://prism-feature-pipeline.is-0b9.workers.dev")
OUT_DIR = Path("/mnt/storage/spike-audit/ranker-xgb-v4")

# Features expected by pct70 data distribution. Same order as the ONNX
# model's input tensor.
FEATURE_COLS = [
    "spike_count",
    "n_streams",
    "interaction",            # spike_count * n_streams
    "unsat_frac",
    "persistence",
    "log_spike_count",
    "log_interaction",
    "spread",
    "burial_score",
    "spike_density",
    "druggability",
    "aromatic_score",
    "n_lining_residues",
    # Temporal (empirically validated: EXCELLENT 0.549 vs POOR 0.737)
    # XGBoost handles missing values natively; NaN for sites whose parquets
    # lack ccns_phase.
    "phase_transition_ratio",       # warm_hold / max(cold_hold, 1)
    "warm_hold_spike_fraction",     # warm_hold / max(total, 1)
]

# Expected pct70 feature ranges (from the 5-target pct70 validation + directive).
# Training will warn but not abort if distributions fall outside these.
EXPECTED_PCT70_RANGES = {
    "unsat_frac_mean": (0.25, 0.45),       # directive: ~30% unsaturated
    "spike_count_median": (50_000, 500_000),
    "n_sites_per_target_mean": (30, 80),   # directive: ~50/target
    "persistence_nonzero_frac": (0.1, 0.9),  # pct70 should have VARIABLE persistence
}


def _api_get(url: str, timeout: int = 60) -> Any:
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 prism4d-xgb-v4"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def fetch_pct70_dataset() -> pd.DataFrame:
    """Pull all pct70 targets + their site_features (including min_dist_to_ligand)."""
    print(f"Fetching pct70 targets from {API_BASE}/targets?spike_percentile=70")
    tdata = _api_get(f"{API_BASE}/targets?spike_percentile=70")
    targets = [t["target"] for t in tdata["targets"]]
    print(f"  {len(targets)} pct70 targets")
    if not targets:
        raise RuntimeError("No pct70 targets in D1 — has post_campaign_analysis.py run?")

    print("Fetching corrected_dcc (all grades)...")
    dcc = _api_get(f"{API_BASE}/dcc")
    dcc_map = {r["target"]: r for r in dcc["records"]}

    print("Fetching site_features per target...")
    rows = []
    for i, target in enumerate(targets):
        if i % 50 == 0:
            print(f"  [{i}/{len(targets)}]", flush=True)
        try:
            sdata = _api_get(f"{API_BASE}/site-features/{target}", timeout=30)
            for s in sdata.get("sites", []):
                r = dict(s)
                r["target"] = target
                # Join with corrected_dcc grade for filtering
                r["dcc_grade"] = dcc_map.get(target, {}).get("dcc_grade", "N/A")
                rows.append(r)
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError) as e:
            print(f"  WARN {target}: {e}")
    df = pd.DataFrame(rows)
    print(f"  Total sites: {len(df)} across {df['target'].nunique()} targets")
    return df


def validate_pct70_distribution(df: pd.DataFrame) -> List[str]:
    """Warn if feature distributions fall outside expected pct70 ranges."""
    warnings: List[str] = []
    if "unsat_frac" in df:
        m = df["unsat_frac"].dropna().mean()
        lo, hi = EXPECTED_PCT70_RANGES["unsat_frac_mean"]
        if not (lo <= m <= hi):
            warnings.append(f"unsat_frac mean={m:.3f} outside expected [{lo}, {hi}] — "
                            f"is this actually pct70 data?")
    if "spike_count" in df:
        m = df["spike_count"].dropna().median()
        lo, hi = EXPECTED_PCT70_RANGES["spike_count_median"]
        if not (lo <= m <= hi):
            warnings.append(f"spike_count median={m:,.0f} outside expected [{lo:,}, {hi:,}]")
    n_sites_per = df.groupby("target").size().mean()
    lo, hi = EXPECTED_PCT70_RANGES["n_sites_per_target_mean"]
    if not (lo <= n_sites_per <= hi):
        warnings.append(f"avg sites/target={n_sites_per:.1f} outside expected [{lo}, {hi}]")
    if "persistence" in df:
        nz = (df["persistence"].fillna(0) > 0.01).mean()
        lo, hi = EXPECTED_PCT70_RANGES["persistence_nonzero_frac"]
        if not (lo <= nz <= hi):
            warnings.append(f"persistence variance looks wrong — {nz:.1%} sites non-zero")
    return warnings


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Engineer training features. All continuous, no binning."""
    df = df.copy()
    sc = df.get("spike_count", 0).fillna(0).astype(float)
    ns = df.get("n_streams", 4).fillna(4).astype(float)
    inter = sc * ns
    df["spike_count"] = sc
    df["n_streams"] = ns
    df["interaction"] = inter
    df["log_spike_count"] = np.log1p(sc)
    df["log_interaction"] = np.log1p(inter)
    df["unsat_frac"] = df.get("unsat_frac", 0).fillna(0).astype(float)
    df["persistence"] = df.get("persistence", 0).fillna(0).astype(float)
    df["spread"] = df.get("spread", 0).fillna(0).astype(float)
    df["burial_score"] = df.get("burial", 0).fillna(0).astype(float)
    df["spike_density"] = df.get("spike_density", 0).fillna(0).astype(float)
    # Additional binding_sites.json features if present; default to 0
    for col in ("druggability", "aromatic_score", "n_lining_residues"):
        if col not in df.columns:
            df[col] = 0.0

    # Temporal features — NaN if the queue consumer hasn't computed them yet
    # (XGBoost handles native NaN via its default direction in tree splits).
    for col in ("phase_transition_ratio", "warm_hold_spike_fraction"):
        if col not in df.columns:
            df[col] = np.nan

    # Graded score = 1/(1+min_dist). NULL (training exclusion) if no min_dist.
    df["graded_score"] = 1.0 / (1.0 + df.get("min_dist_to_ligand", np.nan))
    return df


def loto_evaluate(df: pd.DataFrame, n_rounds: int = 500,
                  cluster_map: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
    """LOTO with XGBRanker rank:ndcg.

    When cluster_map is provided, each training fold excludes the entire
    sequence-identity cluster of the held-out target, not just that one
    target. This prevents homolog leakage.
    """
    df = df.dropna(subset=["graded_score"]).reset_index(drop=True)
    targets = df["target"].unique()
    cluster_mode = cluster_map is not None
    print(f"\nLOTO over {len(targets)} pct70 targets (after GT filter, "
          f"cluster_split={'ON' if cluster_mode else 'OFF'})")

    X = df[FEATURE_COLS].to_numpy(dtype=np.float32)
    y = df["graded_score"].to_numpy(dtype=np.float32)
    y_int = np.clip((y * 31).round(), 0, 31).astype(np.int32)
    tgt_arr = df["target"].to_numpy()
    site_arr = df["site_name"].to_numpy()

    tgt_cluster_arr = None
    if cluster_map is not None:
        tgt_cluster_arr = np.array([cluster_map.get(t, t) for t in tgt_arr])

    sr = {1: 0, 3: 0, 5: 0, 10: 0}
    per_target = []
    total_homologs_excluded = 0
    t0 = time.time()

    for i, tgt in enumerate(targets):
        test_mask = tgt_arr == tgt
        if cluster_map is not None:
            cluster = cluster_map.get(tgt, tgt)
            train_mask = tgt_cluster_arr != cluster
        else:
            train_mask = tgt_arr != tgt
        X_tr, y_tr = X[train_mask], y_int[train_mask]
        t_tr = tgt_arr[train_mask]
        order = np.argsort(t_tr, kind="stable")
        X_tr, y_tr, t_tr = X_tr[order], y_tr[order], t_tr[order]
        _, gs = np.unique(t_tr, return_counts=True)

        X_te = X[test_mask]
        y_te = y[test_mask]
        site_te = site_arr[test_mask]
        if len(X_tr) == 0 or len(X_te) == 0:
            continue

        model = xgb.XGBRanker(
            n_estimators=n_rounds, max_depth=6, learning_rate=0.05,
            objective="rank:ndcg", eval_metric="ndcg@1",
            tree_method="hist", n_jobs=4, verbosity=0, random_state=42,
        )
        model.fit(X_tr, y_tr, group=gs, verbose=False)

        scores = model.predict(X_te)
        ranking = np.argsort(-scores)
        gold = int(np.argmax(y_te))
        gold_pos = int(np.where(ranking == gold)[0][0]) + 1
        for k in sr:
            if gold_pos <= k:
                sr[k] += 1

        n_excluded = int((~train_mask).sum() - test_mask.sum()) if cluster_map else 0
        total_homologs_excluded += n_excluded
        per_target.append({
            "target": str(tgt),
            "n_sites": int(len(X_te)),
            "gold_site": str(site_te[gold]),
            "gold_dist": float(1.0 / y_te[gold] - 1.0) if y_te[gold] > 0 else math.inf,
            "gold_rank": gold_pos,
            "top_pred": str(site_te[ranking[0]]),
            "cluster": cluster_map.get(tgt) if cluster_map else None,
            "n_homologs_excluded": n_excluded,
        })

        if (i + 1) % 25 == 0 or (i + 1) == len(targets):
            sr1 = sr[1] / (i + 1) * 100
            print(f"  [{i+1}/{len(targets)}]  SR@1={sr1:.2f}%", flush=True)

    n = len(per_target)
    if cluster_map and n:
        n_clusters = len({cluster_map.get(t, t) for t in targets})
        print(f"\n  Cluster-aware LOTO: {n_clusters} clusters, "
              f"{total_homologs_excluded} total homolog sites excluded across all folds")
    return {
        "n_targets_evaluated": n,
        "cluster_split": cluster_map is not None,
        "n_clusters": len({cluster_map.get(t, t) for t in targets}) if cluster_map else None,
        "total_homologs_excluded": total_homologs_excluded,
        "sr1": sr[1], "sr1_pct": round(sr[1] / n * 100, 2) if n else 0.0,
        "sr3": sr[3], "sr3_pct": round(sr[3] / n * 100, 2) if n else 0.0,
        "sr5": sr[5], "sr5_pct": round(sr[5] / n * 100, 2) if n else 0.0,
        "sr10": sr[10], "sr10_pct": round(sr[10] / n * 100, 2) if n else 0.0,
        "per_target": per_target,
        "loto_duration_sec": time.time() - t0,
    }


def train_final_model(df: pd.DataFrame, n_rounds: int = 500) -> xgb.XGBRanker:
    df = df.dropna(subset=["graded_score"]).sort_values("target", kind="stable").reset_index(drop=True)
    X = df[FEATURE_COLS].to_numpy(dtype=np.float32)
    y = df["graded_score"].to_numpy(dtype=np.float32)
    y_int = np.clip((y * 31).round(), 0, 31).astype(np.int32)
    groups = df.groupby("target", sort=False).size().values
    model = xgb.XGBRanker(
        n_estimators=n_rounds, max_depth=6, learning_rate=0.05,
        objective="rank:ndcg", eval_metric="ndcg@1",
        tree_method="hist", n_jobs=4, verbosity=0, random_state=42,
    )
    model.fit(X, y_int, group=groups, verbose=False)
    return model


def export_onnx(model: xgb.XGBRanker, out_path: Path) -> bool:
    """Export XGBRanker to ONNX via the XGBRegressor wrapper (onnxmltools
    doesn't support XGBRanker directly; underlying booster is tree ensemble)."""
    try:
        from onnxmltools.convert import convert_xgboost
        from onnxconverter_common import FloatTensorType
        import onnxmltools
        # Wrap as regressor so onnxmltools accepts it
        reg = xgb.XGBRegressor(
            n_estimators=model.n_estimators, max_depth=model.max_depth,
            tree_method="hist", objective="reg:squarederror",
        )
        X_dummy = np.zeros((2, len(FEATURE_COLS)), dtype=np.float32)
        reg.fit(X_dummy, np.array([0.0, 1.0], dtype=np.float32))
        reg._Booster = model.get_booster()
        onnx_model = convert_xgboost(
            reg,
            initial_types=[("input", FloatTensorType([None, len(FEATURE_COLS)]))],
            target_opset=15,
        )
        onnxmltools.utils.save_model(onnx_model, str(out_path))
        print(f"  ONNX: {out_path} ({out_path.stat().st_size / 1024:.1f} KB)")
        return True
    except ImportError as e:
        print(f"  ONNX export skipped (missing {e.name})")
        return False
    except Exception as e:
        print(f"  ONNX export failed: {type(e).__name__}: {e}")
        return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-rounds", type=int, default=500)
    parser.add_argument("--no-loto", action="store_true")
    parser.add_argument("--out-dir", default=str(OUT_DIR))
    parser.add_argument("--bundle-dir", type=Path,
                        default=Path("/mnt/storage/spike-audit/features-pct95"))
    parser.add_argument("--min-seq-id", type=float, default=0.3,
                        help="MMseqs2 identity threshold for homolog-safe LOTO")
    parser.add_argument("--cluster-cache-path", type=Path,
                        default=Path("/mnt/storage/spike-audit/seq_clusters.json"))
    parser.add_argument("--no-cluster-split", action="store_true",
                        help="Disable cluster-aware LOTO (NOT recommended)")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = fetch_pct70_dataset()
    if df.empty:
        print("ERROR: no pct70 data available")
        sys.exit(1)

    df = build_features(df)

    # Validate distribution
    warns = validate_pct70_distribution(df)
    if warns:
        print("\n⚠ Distribution warnings:")
        for w in warns:
            print(f"  • {w}")
    else:
        print("\n✓ Feature distributions in expected pct70 ranges")

    print(f"\nDataset: {len(df)} sites, {df['target'].nunique()} targets, "
          f"{df['graded_score'].notna().sum()} with GT")

    if not args.no_loto:
        cluster_map: Optional[Dict[str, str]] = None
        if not args.no_cluster_split:
            eval_targets = df.dropna(subset=["graded_score"])["target"].unique().tolist()
            print(f"\nBuilding MMseqs2 cluster map ({args.min_seq_id*100:.0f}% id)...")
            cluster_map = target_to_cluster_map(
                bundle_dir=args.bundle_dir,
                targets=eval_targets,
                min_seq_id=args.min_seq_id,
                cache_path=args.cluster_cache_path,
            )
            n_clust = len(set(cluster_map.values()))
            print(f"  {len(cluster_map)} targets → {n_clust} clusters "
                  f"(avg {len(cluster_map)/max(n_clust,1):.1f}/cluster)")

        result = loto_evaluate(df, n_rounds=args.n_rounds, cluster_map=cluster_map)
        print(f"\n{'='*60}")
        print(f"  XGBoost v4 — LOTO Results (pct70)")
        print(f"{'='*60}")
        print(f"  Targets evaluated: {result['n_targets_evaluated']}")
        print(f"  SR@1:  {result['sr1_pct']:.2f}%  ({result['sr1']})")
        print(f"  SR@3:  {result['sr3_pct']:.2f}%  ({result['sr3']})")
        print(f"  SR@5:  {result['sr5_pct']:.2f}%  ({result['sr5']})")
        print(f"  SR@10: {result['sr10_pct']:.2f}%  ({result['sr10']})")
        with open(out_dir / "evaluation.json", "w") as f:
            json.dump(result, f, indent=2, default=str)

        # Comparison with v3
        print(f"\n{'Ranker':<30} {'SR@1':>8} {'SR@3':>8} {'SR@5':>8}")
        print("─" * 54)
        print(f"{'Tokenized v4 (pct95)':<30} {'36.42%':>8} {'81.13%':>8} {'94.70%':>8}")
        print(f"{'XGBoost v3 (pct95 parquet)':<30} {'47.83%':>8} {'85.51%':>8} {'95.94%':>8}")
        print(f"{'XGBoost v4 (pct70 full)':<30} "
              f"{result['sr1_pct']:>7.2f}% {result['sr3_pct']:>7.2f}% {result['sr5_pct']:>7.2f}%")

    # Final model + ONNX
    print("\nTraining final model on full pct70 labeled set...")
    model = train_final_model(df, n_rounds=args.n_rounds)
    model.save_model(str(out_dir / "model.json"))
    export_onnx(model, out_dir / "model.onnx")

    # Feature importance
    importance = model.get_booster().get_score(importance_type="gain")
    fmap = {f"f{i}": n for i, n in enumerate(FEATURE_COLS)}
    sorted_imp = sorted(((fmap.get(f, f), v) for f, v in importance.items()), key=lambda x: -x[1])
    print("\n  Feature importance (gain):")
    for name, val in sorted_imp:
        print(f"    {name:<25} {val:>10.2f}")
    with open(out_dir / "feature_importance.json", "w") as f:
        json.dump(dict(sorted_imp), f, indent=2)

    print(f"\n  Outputs in: {out_dir}")
    print("  To activate in engine: copy model.onnx to crates/prism-nhs/assets/xgb_ranker_v4.onnx,")
    print("  update xgb_ranker.rs EMBEDDED_ONNX path, rebuild, use --use-xgb-ranker flag.")


if __name__ == "__main__":
    main()
