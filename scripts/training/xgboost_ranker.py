#!/usr/bin/env python3
"""XGBoost Site Ranker — continuous-feature LOTO ranker.

Replaces the discrete 256-bin tokenized v4 ranker (SR@1=36.4%) with a
gradient-boosted model on continuous site features. Target: break 40% SR@1.

Data source:
  D1 `site_features` table (populated by Phase 1.3)

Features per site (same dimensions as tokenized v4, but unbinned):
  - spike_count                   (u64, log-transformed)
  - n_streams                     (int, effectively always 4)
  - unsat_frac                    (f32, from physics extraction if present)
  - spike_density                 (spikes / Å³)
  - spread                        (Å, pocket volume^1/3)
  - burial                        (mean min-distance from Cα)
  - spike_count × n_streams       (interaction, same as v4's d2)

Labels:
  - Binary: "is the best-DCC site for this target?" (top-1 per target)
  - Gold: SPIKE DCC from corrected_dcc table (< 5Å → positive)

Evaluation:
  Leave-One-Target-Out (LOTO) — train on N-1 targets' sites, predict on held-out,
  report SR@1 / SR@3 / SR@5 at the LOTO aggregate level.

Output:
  - /mnt/storage/spike-audit/ranker-xgb-v1/model.json       (XGBoost model)
  - /mnt/storage/spike-audit/ranker-xgb-v1/evaluation.json  (LOTO metrics)

Usage:
  # Pull site features from D1 Worker API:
  python3 scripts/training/xgboost_ranker.py --fetch-from-api

  # Or use a local CSV dump (same schema as D1 site_features):
  python3 scripts/training/xgboost_ranker.py --input-csv /path/to/sites.csv
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

try:
    import xgboost as xgb
    HAVE_XGB = True
except ImportError:
    HAVE_XGB = False

import pandas as pd

API_BASE = "https://prism-feature-pipeline.is-0b9.workers.dev"
OUT_DIR = Path("/mnt/storage/spike-audit/ranker-xgb-v1")

# ── Feature columns in order (must match training + inference) ──
FEATURE_COLS = [
    "log_spike_count",
    "n_streams",
    "log_interaction",      # log(spike_count * n_streams)
    "unsat_frac",
    "log_spike_density",    # log(spatial density)
    "spread",
    "burial",
]


# ─────────────────────────────────────────────────────────────
#  Data loaders
# ─────────────────────────────────────────────────────────────

def _api_get(url: str, timeout: int = 60) -> Any:
    """GET with a browser-ish User-Agent (Cloudflare 403s default Python)."""
    req = urllib.request.Request(
        url,
        headers={"User-Agent": "Mozilla/5.0 (prism4d-training)"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def fetch_all_targets_from_api() -> pd.DataFrame:
    """Pull every target's site features from the Worker API, join with
    corrected_dcc, and return a DataFrame ready for feature engineering."""
    print(f"Fetching target list from {API_BASE}/targets...")
    tdata = _api_get(f"{API_BASE}/targets")
    targets = [t["target"] for t in tdata["targets"]]
    print(f"  {len(targets)} targets available")

    print(f"Fetching corrected_dcc...")
    dcc_data = _api_get(f"{API_BASE}/dcc")
    dcc_map = {r["target"]: r for r in dcc_data["records"]}
    print(f"  {len(dcc_map)} DCC records")

    rows = []
    for i, target in enumerate(targets):
        if i % 50 == 0:
            print(f"  Fetching sites [{i}/{len(targets)}]")
        try:
            sdata = _api_get(f"{API_BASE}/site-features/{target}", timeout=30)
            dcc = dcc_map.get(target, {})
            for s in sdata.get("sites", []):
                row = dict(s)
                row["target"] = target
                row["corrected_spike_dcc"] = dcc.get("spike_dcc")
                row["corrected_spike_site"] = dcc.get("spike_site")
                row["dcc_grade"] = dcc.get("dcc_grade")
                rows.append(row)
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError) as e:
            print(f"  WARN: {target}: {e}")
            continue

    df = pd.DataFrame(rows)
    print(f"  Total sites: {len(df)}")
    return df


def load_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    print(f"Loaded {len(df)} site rows from {path}")
    return df


# ─────────────────────────────────────────────────────────────
#  Feature engineering
# ─────────────────────────────────────────────────────────────

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Engineer training features from raw site data.

    Safety-handles missing values and zeros (log(0) = -inf → clamp to 0).
    """
    df = df.copy()

    # Core features
    df["log_spike_count"] = np.log1p(df.get("spike_count", 0).astype(float))
    df["n_streams"] = df.get("n_streams", 4).fillna(4).astype(int)

    interaction = df.get("spike_count", 0).astype(float) * df["n_streams"].astype(float)
    df["log_interaction"] = np.log1p(interaction)

    df["unsat_frac"] = df.get("unsat_frac", 0).fillna(0).astype(float)

    density = df.get("spike_density", 0).astype(float).fillna(0)
    df["log_spike_density"] = np.log1p(density)

    df["spread"] = df.get("spread", 0).fillna(0).astype(float)
    df["burial"] = df.get("burial", 0).fillna(0).astype(float)

    # Label: binary positive if this site is THE best-DCC site AND grade is EXCELLENT or GOOD
    # Schemas:
    #   site_features.site_name   = "siteN"                 (e.g. "site1531")
    #   corrected_dcc.spike_site  = "<target>.siteN"        (e.g. "9qvu_chainA.site4004")
    # We match by stripping the "<target>." prefix from spike_site, or by suffix match.
    def is_positive(row) -> int:
        gold_site = row.get("corrected_spike_site")
        grade = row.get("dcc_grade")
        if not isinstance(gold_site, str) or not gold_site:
            return 0
        if grade not in ("EXCELLENT", "GOOD"):
            return 0
        my_site = str(row["site_name"])
        # Canonicalize: take the last '.'-delimited segment
        gold_suffix = gold_site.rsplit(".", 1)[-1]
        return 1 if my_site == gold_suffix else 0

    df["label"] = df.apply(is_positive, axis=1)

    return df


# ─────────────────────────────────────────────────────────────
#  LOTO evaluation
# ─────────────────────────────────────────────────────────────

def loto_evaluate(df: pd.DataFrame, n_rounds: int = 200, early_stopping: int = 20) -> Dict[str, Any]:
    """Leave-one-target-out: for each target, train on the other N-1 targets'
    sites, predict on this target's sites, record if the top-predicted site
    is the gold-standard site.

    Only targets with positive labels contribute to SR@k (we need a gold
    answer to check against). Targets with no positive label (no corrected
    spike_dcc, or grade MARGINAL/POOR) are skipped.
    """
    # Keep only targets with at least one positive label
    targets_with_gold = df[df["label"] == 1]["target"].unique()
    print(f"\nLOTO evaluation over {len(targets_with_gold)} gold targets")

    X_all = df[FEATURE_COLS].to_numpy(dtype=np.float32)
    y_all = df["label"].to_numpy(dtype=np.int32)
    targets_arr = df["target"].to_numpy()
    site_names = df["site_name"].to_numpy()

    sr_at_k = {1: 0, 3: 0, 5: 0, 10: 0}
    per_target_results = []

    t_start = time.time()
    for i, tgt in enumerate(targets_with_gold):
        train_mask = targets_arr != tgt
        test_mask = targets_arr == tgt

        X_train, y_train = X_all[train_mask], y_all[train_mask]
        X_test, y_test = X_all[test_mask], y_all[test_mask]

        if len(X_train) == 0 or len(X_test) == 0:
            continue

        # Training set: focus loss on ranking — use spw to balance positives
        n_pos = int(y_train.sum())
        n_neg = len(y_train) - n_pos
        spw = max(1.0, n_neg / max(n_pos, 1))

        model = xgb.XGBClassifier(
            n_estimators=n_rounds,
            max_depth=4,
            learning_rate=0.1,
            scale_pos_weight=spw,
            objective="binary:logistic",
            eval_metric="logloss",
            tree_method="hist",
            early_stopping_rounds=early_stopping,
            n_jobs=4,
            verbosity=0,
            random_state=42,
        )

        # Use held-out target as ersatz eval — strictly LOTO means no val,
        # but we need early stopping signal; use a mini-split from train.
        if len(X_train) > 500:
            rng = np.random.default_rng(42)
            val_idx = rng.choice(len(X_train), size=max(50, len(X_train) // 10), replace=False)
            train_idx = np.setdiff1d(np.arange(len(X_train)), val_idx)
            model.fit(
                X_train[train_idx], y_train[train_idx],
                eval_set=[(X_train[val_idx], y_train[val_idx])],
                verbose=False,
            )
        else:
            model.fit(X_train, y_train, verbose=False)

        # Predict
        proba = model.predict_proba(X_test)[:, 1]

        # Rank by proba descending
        order = np.argsort(-proba)

        # Find position of the gold site
        gold_idx = np.where(y_test == 1)[0]
        if len(gold_idx) == 0:
            continue
        gold_pos = int(np.where(order == gold_idx[0])[0][0])
        gold_rank = gold_pos + 1  # 1-indexed

        for k in sr_at_k:
            if gold_rank <= k:
                sr_at_k[k] += 1

        per_target_results.append({
            "target": str(tgt),
            "gold_site": site_names[test_mask][gold_idx[0]],
            "n_sites": len(X_test),
            "gold_rank": gold_rank,
            "top_pred_site": site_names[test_mask][order[0]],
            "top_pred_proba": float(proba[order[0]]),
        })

        if (i + 1) % 25 == 0:
            elapsed = time.time() - t_start
            rate = (i + 1) / elapsed
            eta = (len(targets_with_gold) - i - 1) / rate
            sr1_pct = sr_at_k[1] / (i + 1) * 100
            print(f"  [{i+1}/{len(targets_with_gold)}] SR@1={sr1_pct:.1f}% "
                  f"rate={rate:.1f}/s eta={eta:.0f}s")

    n = len(per_target_results)
    result = {
        "n_targets_evaluated": n,
        "sr1": sr_at_k[1],
        "sr1_pct": round(sr_at_k[1] / n * 100, 2) if n > 0 else 0.0,
        "sr3": sr_at_k[3],
        "sr3_pct": round(sr_at_k[3] / n * 100, 2) if n > 0 else 0.0,
        "sr5": sr_at_k[5],
        "sr5_pct": round(sr_at_k[5] / n * 100, 2) if n > 0 else 0.0,
        "sr10": sr_at_k[10],
        "sr10_pct": round(sr_at_k[10] / n * 100, 2) if n > 0 else 0.0,
        "per_target": per_target_results,
    }
    return result


def train_final_model(df: pd.DataFrame, n_rounds: int = 300) -> xgb.XGBClassifier:
    """Train on the full dataset (no hold-out) for the production model."""
    X = df[FEATURE_COLS].to_numpy(dtype=np.float32)
    y = df["label"].to_numpy(dtype=np.int32)
    spw = max(1.0, (len(y) - y.sum()) / max(int(y.sum()), 1))

    model = xgb.XGBClassifier(
        n_estimators=n_rounds,
        max_depth=4,
        learning_rate=0.1,
        scale_pos_weight=spw,
        objective="binary:logistic",
        eval_metric="logloss",
        tree_method="hist",
        n_jobs=4,
        verbosity=0,
        random_state=42,
    )
    model.fit(X, y, verbose=False)
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fetch-from-api", action="store_true",
                        help="Pull site data from D1 via Worker API")
    parser.add_argument("--input-csv", type=str, help="Load site data from CSV")
    parser.add_argument("--out-dir", type=str, default=str(OUT_DIR),
                        help="Output directory for model + eval JSON")
    parser.add_argument("--n-rounds", type=int, default=200,
                        help="Max XGBoost boosting rounds")
    parser.add_argument("--no-loto", action="store_true",
                        help="Skip LOTO eval, just train final model")
    args = parser.parse_args()

    if not HAVE_XGB:
        print("ERROR: xgboost is not installed. pip install --break-system-packages xgboost")
        sys.exit(1)

    # ── Data ──
    if args.fetch_from_api:
        df = fetch_all_targets_from_api()
    elif args.input_csv:
        df = load_csv(args.input_csv)
    else:
        print("ERROR: provide --fetch-from-api or --input-csv")
        sys.exit(1)

    if len(df) == 0:
        print("ERROR: no data loaded")
        sys.exit(1)

    df = build_features(df)
    print(f"\nDataset: {len(df)} sites, {df['target'].nunique()} targets, "
          f"{int(df['label'].sum())} positive labels")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── LOTO evaluation ──
    if not args.no_loto:
        eval_result = loto_evaluate(df, n_rounds=args.n_rounds)
        print(f"\n{'='*60}")
        print(f"  XGBoost Ranker v1 — LOTO Results")
        print(f"{'='*60}")
        print(f"  N targets evaluated: {eval_result['n_targets_evaluated']}")
        print(f"  SR@1:  {eval_result['sr1_pct']:.2f}%  ({eval_result['sr1']})")
        print(f"  SR@3:  {eval_result['sr3_pct']:.2f}%  ({eval_result['sr3']})")
        print(f"  SR@5:  {eval_result['sr5_pct']:.2f}%  ({eval_result['sr5']})")
        print(f"  SR@10: {eval_result['sr10_pct']:.2f}%  ({eval_result['sr10']})")

        eval_path = out_dir / "evaluation.json"
        with open(eval_path, "w") as f:
            json.dump(eval_result, f, indent=2)
        print(f"\n  Evaluation saved: {eval_path}")

        # Baseline check
        print(f"\n  Tokenized v4 baseline:  SR@1=36.42%  SR@3=81.13%  SR@5=94.70%")
        if eval_result['sr1_pct'] >= 40.0:
            print(f"  ✓ TARGET MET (SR@1 >= 40%)")
        else:
            print(f"  ✗ Below target (need SR@1 >= 40%)")

    # ── Final model ──
    print(f"\nTraining final model on full dataset...")
    final_model = train_final_model(df, n_rounds=args.n_rounds)
    model_path = out_dir / "model.json"
    final_model.save_model(str(model_path))
    print(f"  Model saved: {model_path}")

    # Feature importance
    booster = final_model.get_booster()
    fmap = {f"f{i}": name for i, name in enumerate(FEATURE_COLS)}
    importance = booster.get_score(importance_type="gain")
    sorted_imp = sorted(
        ((fmap.get(f, f), v) for f, v in importance.items()),
        key=lambda x: -x[1],
    )
    print("\n  Feature importance (gain):")
    for name, val in sorted_imp:
        print(f"    {name:<25} {val:>10.2f}")

    fi_path = out_dir / "feature_importance.json"
    with open(fi_path, "w") as f:
        json.dump({name: val for name, val in sorted_imp}, f, indent=2)
    print(f"  Feature importance saved: {fi_path}")


if __name__ == "__main__":
    main()
