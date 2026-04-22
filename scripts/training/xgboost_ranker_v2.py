#!/usr/bin/env python3
"""XGBoost Ranker v2 — enriched features + pairwise ranking objective.

Improvements over v1 (SR@1 = 6.34%):
  - Pulls features directly from cached binding_sites.json (12+ features vs 4)
  - XGBRanker with rank:pairwise objective (not classifier)
  - Computes unsat_frac proxy from site's spike_count / target max
  - Log-transforms heavy-tailed features

Feature set (12 dims per site):
  Spike volume:
    - log_spike_count
    - log_relative_spike_share  (spike_count / sum per target — v4's interaction)
  Geometry:
    - log_volume
    - burial_score
    - spread
  Chemistry:
    - druggability
    - aromatic_score
    - n_lining_residues
    - catalytic_residue_count
  Dynamics:
    - quality_score
    - onset_score
    - source_diversity
  Classification (ordinal):
    - therm_class_score  (CRYPTIC=3, RESPONSIVE=2, DYNAMIC=2, INERT=0)
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import xgboost as xgb

API_BASE = "https://prism-feature-pipeline.is-0b9.workers.dev"
CACHE = Path("/tmp/spike_count_audit")
OUT_DIR = Path("/mnt/storage/spike-audit/ranker-xgb-v2")

FEATURE_COLS = [
    "log_spike_count",
    "log_relative_spike_share",
    "log_volume",
    "burial_score",
    "spread",
    "druggability",
    "aromatic_score",
    "n_lining_residues",
    "catalytic_residue_count",
    "quality_score",
    "onset_score",
    "source_diversity",
    "therm_class_score",
]

THERM_SCORE = {
    "CRYPTIC": 3.0,
    "RESPONSIVE": 2.0,
    "DYNAMIC": 2.0,
    "INERT": 0.0,
    None: 0.5,
    "": 0.5,
}


def _api_get(url: str, timeout: int = 60) -> Any:
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 prism4d-xgb-v2"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def load_dcc_map() -> Dict[str, dict]:
    dcc = _api_get(f"{API_BASE}/dcc")
    return {r["target"]: r for r in dcc["records"]}


def build_dataset(dcc_map: Dict[str, dict]) -> pd.DataFrame:
    """Load every site from cached binding_sites.json into a DataFrame."""
    bs_files = sorted(CACHE.glob("*/*.binding_sites.json"))
    rows = []
    for bs_path in bs_files:
        target = bs_path.name.replace(".binding_sites.json", "")
        try:
            with open(bs_path) as f:
                bs = json.load(f)
        except (OSError, json.JSONDecodeError):
            continue

        sites = bs.get("sites", [])
        if not sites:
            continue

        total_spikes = sum(s.get("spike_count", 0) for s in sites)
        total_spikes = max(total_spikes, 1)

        dcc = dcc_map.get(target, {})
        gold_site = dcc.get("spike_site", "")
        gold_suffix = gold_site.rsplit(".", 1)[-1] if gold_site else ""
        grade = dcc.get("dcc_grade", "")

        for s in sites:
            sid = s.get("id")
            site_name = f"site{sid}" if sid is not None else "siteunk"
            spike_count = s.get("spike_count", 0)
            volume = s.get("volume", s.get("volume_angstrom3", 0.0)) or 0.0
            spread = volume ** (1 / 3) if volume > 0 else 0.0

            # Label: positive if this is the best-DCC site and grade is EXCELLENT/GOOD
            is_gold = 1 if (
                site_name == gold_suffix
                and grade in ("EXCELLENT", "GOOD")
            ) else 0

            rows.append({
                "target": target,
                "site_name": site_name,
                "label": is_gold,
                "dcc_grade": grade,
                # Raw / derived features
                "log_spike_count": math.log1p(spike_count),
                "log_relative_spike_share": math.log1p(spike_count / total_spikes * 1000.0),
                "log_volume": math.log1p(volume),
                "spread": spread,
                "burial_score": s.get("burial_score", 0.0) or 0.0,
                "druggability": s.get("druggability", 0.0) or 0.0,
                "aromatic_score": s.get("aromatic_score", 0.0) or 0.0,
                "n_lining_residues": len(s.get("lining_residues", [])),
                "catalytic_residue_count": s.get("catalytic_residue_count", 0) or 0,
                "quality_score": s.get("quality_score", 0.0) or 0.0,
                "onset_score": s.get("onset_score", 0.0) or 0.0,
                "source_diversity": s.get("source_diversity", 0.0) or 0.0,
                "therm_class_score": THERM_SCORE.get(s.get("therm_class"), 0.5),
            })

    df = pd.DataFrame(rows)
    return df


def loto_evaluate_pairwise(df: pd.DataFrame, n_rounds: int = 300) -> Dict[str, Any]:
    """LOTO with XGBRanker pairwise objective."""
    positive_targets = df[df["label"] == 1]["target"].unique()
    print(f"\nLOTO over {len(positive_targets)} gold targets")

    X_all = df[FEATURE_COLS].to_numpy(dtype=np.float32)
    y_all = df["label"].to_numpy(dtype=np.int32)
    targets_arr = df["target"].to_numpy()
    site_names = df["site_name"].to_numpy()

    sr = {1: 0, 3: 0, 5: 0, 10: 0}
    per_target = []

    for i, tgt in enumerate(positive_targets):
        train_mask = targets_arr != tgt
        test_mask = targets_arr == tgt

        # Group counts per target for XGBRanker
        train_targets = targets_arr[train_mask]
        train_groups = pd.Series(train_targets).value_counts().sort_index()
        # Need the grouping in the ORDER targets appear in the training data
        # Sort training data by target so groups are contiguous
        sort_idx = np.argsort(train_targets, kind="stable")
        X_train = X_all[train_mask][sort_idx]
        y_train = y_all[train_mask][sort_idx]
        # Now group sizes match the sorted order
        group_sizes = pd.Series(train_targets[sort_idx]).groupby(
            pd.Series(train_targets[sort_idx]), sort=False
        ).size().values

        X_test = X_all[test_mask]
        y_test = y_all[test_mask]

        if len(X_train) == 0 or len(X_test) == 0:
            continue

        model = xgb.XGBRanker(
            n_estimators=n_rounds,
            max_depth=5,
            learning_rate=0.08,
            objective="rank:pairwise",
            eval_metric="ndcg@1",
            tree_method="hist",
            n_jobs=4,
            verbosity=0,
            random_state=42,
        )
        model.fit(X_train, y_train, group=group_sizes, verbose=False)

        scores = model.predict(X_test)
        order = np.argsort(-scores)

        gold_idx = np.where(y_test == 1)[0]
        if len(gold_idx) == 0:
            continue
        gold_pos = int(np.where(order == gold_idx[0])[0][0]) + 1

        for k in sr:
            if gold_pos <= k:
                sr[k] += 1

        per_target.append({
            "target": str(tgt),
            "n_sites": int(len(X_test)),
            "gold_rank": gold_pos,
            "top_pred": site_names[test_mask][order[0]],
            "top_score": float(scores[order[0]]),
        })

        if (i + 1) % 25 == 0 or (i + 1) == len(positive_targets):
            sr1 = sr[1] / (i + 1) * 100
            sr3 = sr[3] / (i + 1) * 100
            print(f"  [{i+1}/{len(positive_targets)}]  SR@1={sr1:.1f}%  SR@3={sr3:.1f}%")

    n = len(per_target)
    return {
        "n_targets_evaluated": n,
        "sr1": sr[1],  "sr1_pct": round(sr[1] / n * 100, 2) if n else 0,
        "sr3": sr[3],  "sr3_pct": round(sr[3] / n * 100, 2) if n else 0,
        "sr5": sr[5],  "sr5_pct": round(sr[5] / n * 100, 2) if n else 0,
        "sr10": sr[10], "sr10_pct": round(sr[10] / n * 100, 2) if n else 0,
        "per_target": per_target,
    }


def train_final_model(df: pd.DataFrame, n_rounds: int = 300) -> xgb.XGBRanker:
    """Train on all labeled targets for production model."""
    # Include only targets with at least one positive label for group sanity
    positive_targets = set(df[df["label"] == 1]["target"].unique())
    df_labeled = df[df["target"].isin(positive_targets)].copy()
    df_labeled = df_labeled.sort_values("target", kind="stable")

    X = df_labeled[FEATURE_COLS].to_numpy(dtype=np.float32)
    y = df_labeled["label"].to_numpy(dtype=np.int32)
    groups = df_labeled.groupby("target", sort=False).size().values

    model = xgb.XGBRanker(
        n_estimators=n_rounds,
        max_depth=5,
        learning_rate=0.08,
        objective="rank:pairwise",
        eval_metric="ndcg@1",
        tree_method="hist",
        n_jobs=4,
        verbosity=0,
        random_state=42,
    )
    model.fit(X, y, group=groups, verbose=False)
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-rounds", type=int, default=300)
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading DCC from API...")
    dcc_map = load_dcc_map()
    print(f"  {len(dcc_map)} DCC records")

    print("Building dataset from cached binding_sites.json...")
    df = build_dataset(dcc_map)
    n_pos = int(df["label"].sum())
    print(f"  {len(df)} sites, {df['target'].nunique()} targets, {n_pos} positive labels")

    # LOTO
    result = loto_evaluate_pairwise(df, n_rounds=args.n_rounds)
    print(f"\n{'='*60}")
    print(f"  XGBoost Ranker v2 — LOTO Results (rank:pairwise)")
    print(f"{'='*60}")
    print(f"  N targets evaluated: {result['n_targets_evaluated']}")
    print(f"  SR@1:  {result['sr1_pct']:.2f}%  ({result['sr1']})")
    print(f"  SR@3:  {result['sr3_pct']:.2f}%  ({result['sr3']})")
    print(f"  SR@5:  {result['sr5_pct']:.2f}%  ({result['sr5']})")
    print(f"  SR@10: {result['sr10_pct']:.2f}%  ({result['sr10']})")

    with open(OUT_DIR / "evaluation.json", "w") as f:
        json.dump(result, f, indent=2)

    # ── Comparison table ──
    print(f"\n{'Ranker':<22} {'SR@1':>8} {'SR@3':>8} {'SR@5':>8}")
    print("─" * 50)
    print(f"{'Tokenized v4 (256 bins)':<22} {'36.42%':>8} {'81.13%':>8} {'94.70%':>8}")
    print(f"{'XGBoost v1 (4 feat)':<22} {'6.34%':>8} {'18.28%':>8} {'32.46%':>8}")
    print(f"{'XGBoost v2 (13 feat)':<22} "
          f"{result['sr1_pct']:>7.2f}% {result['sr3_pct']:>7.2f}% {result['sr5_pct']:>7.2f}%")

    print()
    if result["sr1_pct"] >= 40.0:
        print("  ✓ TARGET MET (SR@1 ≥ 40%)")
    elif result["sr1_pct"] >= 36.42:
        print("  ~ Above tokenized v4 but below 40% target")
    else:
        print("  ✗ Below tokenized v4 baseline")

    # ── Final model + importance ──
    print("\nTraining final model on full labeled dataset...")
    model = train_final_model(df, n_rounds=args.n_rounds)
    model.save_model(str(OUT_DIR / "model.json"))

    booster = model.get_booster()
    fmap = {f"f{i}": name for i, name in enumerate(FEATURE_COLS)}
    importance = booster.get_score(importance_type="gain")
    sorted_imp = sorted(
        ((fmap.get(f, f), v) for f, v in importance.items()),
        key=lambda x: -x[1],
    )
    print("\n  Feature importance (gain):")
    for name, val in sorted_imp:
        print(f"    {name:<30} {val:>10.2f}")
    with open(OUT_DIR / "feature_importance.json", "w") as f:
        json.dump(dict(sorted_imp), f, indent=2)

    print(f"\n  Outputs in: {OUT_DIR}")


if __name__ == "__main__":
    main()
