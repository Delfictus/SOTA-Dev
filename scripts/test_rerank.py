#!/usr/bin/env python3
"""
PRISM4D Ranking Ablation — Tests DP impact on ranking without rerunning engine.

Usage:
    python3 scripts/test_rerank.py

Reads existing binding_sites.json outputs and reranks with:
  - original: current BLP/DP correction
  - reduced_dp: DP weight halved
  - no_dp_distributed: DP disabled for distributed regime only
  - gtckl_only: raw GTCKL score (no correction layer)

Reports SR@1, SR@3, SR@10 for each mode.
"""

import json
import math
import os
from pathlib import Path

# Find the latest run directory
BENCH10_RUNS = Path("benchmarks/bench10_results/runs")
HARD_TARGETS = Path("benchmarks/hard_targets/results")

# Ground truth centroids (from bench30 + hard targets)
GT_PATH = Path("benchmarks/prism4d_bench30/ground_truth/ligand_centroids.json")
MANIFEST_PATH = Path("benchmarks/prism4d_bench30/benchmark_manifest.json")

# Hard-target ground truths
HARD_GT = {
    "1p38": [16.885, -3.551, -19.552],  # from 3HEC/STI
    "5lar": [54.0, 68.1, 16.8],          # from 5LAR/6SH
    "3mh1": [34.023, 35.443, 16.502],    # from 3HEC alignment (p38 holo)
}


def load_ground_truth():
    """Load all available ground truth centroids."""
    gt_map = {}

    # bench30
    if GT_PATH.exists() and MANIFEST_PATH.exists():
        with open(GT_PATH) as f:
            gt = json.load(f)
        with open(MANIFEST_PATH) as f:
            mdata = json.load(f)
        manifest = mdata.get("targets", mdata) if isinstance(mdata, dict) else mdata
        for m in manifest:
            if isinstance(m, dict):
                bid = str(m.get("id", ""))
                pdb = m.get("apo_pdb", "").lower()
                if bid in gt:
                    gt_map[pdb] = gt[bid].get("centroid", [0, 0, 0])

    # Hard targets override
    gt_map.update(HARD_GT)
    return gt_map


def compute_dcc(centroid, gt):
    return math.sqrt(sum((centroid[j] - gt[j]) ** 2 for j in range(3)))


def rerank(sites, mode="original"):
    """Rerank sites using different scoring modes."""
    rescored = []
    for s in sites:
        terms = s.get("ranking_terms", {})
        gtckl = terms.get("gtckl", s.get("gtckl_score", 0.0))
        dp = terms.get("dp", 0.0)
        blp = terms.get("blp", 0.0)
        regime = terms.get("regime", "local")

        if mode == "original":
            score = s.get("rank_score", 0.0)
        elif mode == "reduced_dp":
            if regime == "distributed":
                score = gtckl * (1.0 + 0.15 * blp) * (1.0 - 0.15 * dp)  # halved DP
            else:
                score = gtckl * (1.0 + 0.20 * blp)
        elif mode == "no_dp_distributed":
            if regime == "distributed":
                score = gtckl * (1.0 + 0.15 * blp)  # no DP at all
            else:
                score = gtckl * (1.0 + 0.20 * blp)
        elif mode == "gtckl_only":
            score = gtckl
        else:
            raise ValueError(f"Unknown mode: {mode}")

        rescored.append((score, s))

    rescored.sort(key=lambda x: -x[0])
    return [(sc, s) for sc, s in rescored]


def evaluate(bs_path, gt_centroid, modes):
    """Evaluate ranking for one target under multiple modes."""
    with open(bs_path) as f:
        data = json.load(f)
    sites = data if isinstance(data, list) else data.get("sites", [])

    # Add DCC to each site
    for s in sites:
        c = s.get("centroid", [0, 0, 0])
        s["_dcc"] = compute_dcc(c, gt_centroid)

    results = {}
    for mode in modes:
        ranked = rerank(sites, mode)
        if not ranked:
            results[mode] = {"top1_dcc": 999, "best_dcc": 999, "best_rank": 999}
            continue

        top1_dcc = ranked[0][1]["_dcc"]
        best = min(ranked, key=lambda x: x[1]["_dcc"])
        best_rank = ranked.index(best) + 1

        results[mode] = {
            "top1_dcc": round(top1_dcc, 1),
            "best_dcc": round(best[1]["_dcc"], 1),
            "best_rank": best_rank,
        }

    return results


def main():
    gt_map = load_ground_truth()
    modes = ["original", "reduced_dp", "no_dp_distributed", "gtckl_only"]

    # Collect all binding_sites.json from bench10 + hard targets
    bs_files = []

    # Latest bench10 run
    if BENCH10_RUNS.exists():
        runs = sorted(BENCH10_RUNS.glob("*"))
        if runs:
            latest = runs[-1]
            for p in latest.glob("*/*.binding_sites.json"):
                bs_files.append(p)

    # Hard targets
    if HARD_TARGETS.exists():
        for p in HARD_TARGETS.glob("*/*.binding_sites.json"):
            bs_files.append(p)

    if not bs_files:
        print("No binding_sites.json files found.")
        return

    print(f"Found {len(bs_files)} targets")
    print(f"Modes: {', '.join(modes)}")
    print()

    all_results = {}
    for path in sorted(bs_files):
        target = path.name.split(".")[0].lower()
        if target not in gt_map:
            print(f"  {target}: no ground truth — skipping")
            continue

        res = evaluate(path, gt_map[target], modes)
        all_results[target] = res

        print(f"{target:>6}:", end="")
        for mode in modes:
            r = res[mode]
            print(f"  {mode}={r['top1_dcc']:>5.1f}A(R{r['best_rank']})", end="")
        print()

    # Summary
    print(f"\n{'=' * 70}")
    print("SUMMARY — SR@K (success = best DCC ≤ 8Å at rank ≤ K)")
    print(f"{'=' * 70}")

    targets = list(all_results.keys())
    n = len(targets)

    fmt = f"{'Mode':<25} {'SR@1':>6} {'SR@3':>6} {'SR@5':>6} {'SR@10':>6} {'Mean DCC':>9}"
    print(fmt)
    print("-" * 60)

    for mode in modes:
        ranks = [all_results[t][mode]["best_rank"] for t in targets]
        dccs = [all_results[t][mode]["top1_dcc"] for t in targets]
        best_dccs = [all_results[t][mode]["best_dcc"] for t in targets]

        def sr(k):
            return sum(1 for i, t in enumerate(targets)
                       if ranks[i] <= k and best_dccs[i] <= 8.0) / n

        mean_dcc = sum(dccs) / n if n > 0 else 0
        print(f"{mode:<25} {sr(1):>5.0%} {sr(3):>5.0%} {sr(5):>5.0%} {sr(10):>5.0%} {mean_dcc:>8.1f}A")


if __name__ == "__main__":
    main()
