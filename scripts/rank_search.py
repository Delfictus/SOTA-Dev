#!/usr/bin/env python3
"""PRISM4D — Exhaustive Rank Fusion Search.

Tests ALL combinations of rank fusion (RF3, RF4, RF5) across consensus
results and reports which methods achieve rank 1 on the most targets.

This is the script that found RF5(q+enc+loc+p+cvar) as the optimal
consensus ranker.  Run it whenever you add new targets or change
consensus clustering to verify the ranking method still holds.

Usage:
    python3 scripts/rank_search.py

Reads consensus_sites.json from /tmp/prism_consensus/<target>/consensus/
Requires ground truth centroids defined in GT dict below.
"""
from __future__ import annotations

import json
import math
from itertools import combinations
from pathlib import Path
from typing import Any, Callable, Dict, List, Tuple


# ---------------------------------------------------------------------------
# Ground truth centroids
# ---------------------------------------------------------------------------
GT: Dict[str, List[float]] = {
    "1jwp": [6.61, 13.531, 25.526],
    "1p38": [16.885, -3.551, -19.552],
    "2hnp": [27.09, 17.46, 12.15],
}

CONSENSUS_BASE = Path("/tmp/prism_consensus")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def dist(a, b) -> float:
    return math.sqrt(sum((a[i] - b[i]) ** 2 for i in range(3)))


def rank_signal(vals: List[float], reverse: bool = True) -> List[int]:
    """Rank values. reverse=True means highest value gets rank 1."""
    order = sorted(range(len(vals)), key=lambda i: vals[i], reverse=reverse)
    ranks = [0] * len(vals)
    for r, idx in enumerate(order):
        ranks[idx] = r + 1
    return ranks


# ---------------------------------------------------------------------------
# Feature extractors (from ConsensusSite JSON)
# ---------------------------------------------------------------------------
def _get_enc(s: Dict) -> float:
    ms = s.get("member_sites", [{}])
    return float(ms[0].get("enclosure", 0.0)) if ms else 0.0


def _get_vol(s: Dict) -> float:
    ms = s.get("member_sites", [{}])
    return float(ms[0].get("volume", 0.0)) if ms else 0.0


FEATURES: Dict[str, Tuple[Callable, bool]] = {
    "q": (lambda s: float(s.get("mean_quality_score", 0.0)), True),
    "enc": (_get_enc, True),
    "loc": (lambda s: float(s.get("mean_localization", 0.0)), True),
    "rs": (lambda s: float(s.get("mean_response_sharpness", 0.0)), True),
    "cr": (lambda s: float(s.get("mean_contact_reorg", 0.0)), True),
    "p": (lambda s: float(s.get("persistence", 0.0)), True),
    "vol": (_get_vol, True),
    "ac": (lambda s: float(s.get("anchor_consistency", 0.0)), True),
    "lc": (lambda s: float(s.get("lining_consistency", 0.0)), True),
    "gvc": (lambda s: float(s.get("growth_vector_consistency", 0.0)), True),
    "cvar": (lambda s: float(s.get("centroid_variance", 0.0)), False),  # lower=better
}


# ---------------------------------------------------------------------------
# Load consensus data
# ---------------------------------------------------------------------------
def load_targets() -> Dict[str, Tuple[List[Dict], List[float]]]:
    """Load consensus sites and ground truths for all available targets."""
    targets = {}
    for name, gt in GT.items():
        path = CONSENSUS_BASE / name / "consensus" / "consensus_sites.json"
        if not path.exists():
            print(f"  SKIP {name}: {path} not found")
            continue
        with open(path) as f:
            data = json.load(f)
        sites = data.get("consensus_sites", [])
        if sites:
            targets[name] = (sites, gt)
            print(f"  {name}: {len(sites)} consensus sites")
    return targets


# ---------------------------------------------------------------------------
# Test one rank fusion method
# ---------------------------------------------------------------------------
def test_method(
    targets: Dict[str, Tuple[List[Dict], List[float]]],
    feature_names: List[str],
) -> Dict[str, Tuple[float, int]]:
    """Test a rank fusion method across all targets.

    Returns: {target_name: (top1_dcc, gt_rank)}
    """
    results = {}
    for tname, (sites, gt) in targets.items():
        n = len(sites)
        all_ranks = []
        for fname in feature_names:
            fn, rev = FEATURES[fname]
            vals = [fn(s) for s in sites]
            all_ranks.append(rank_signal(vals, reverse=rev))

        sums = [sum(all_ranks[j][i] for j in range(len(feature_names)))
                for i in range(n)]
        order = sorted(range(n), key=lambda i: sums[i])

        dccs = [dist(s["centroid_mean"], gt) for s in sites]
        top1_dcc = dccs[order[0]]
        gt_idx = min(range(n), key=lambda i: dccs[i])
        gt_rank = next(r + 1 for r, idx in enumerate(order) if idx == gt_idx)

        results[tname] = (top1_dcc, gt_rank)

    return results


# ---------------------------------------------------------------------------
# Main search
# ---------------------------------------------------------------------------
def main() -> None:
    print("Loading consensus data...")
    targets = load_targets()
    if not targets:
        print("No targets found. Run consensus first.")
        return

    n_targets = len(targets)
    fnames = list(FEATURES.keys())

    all_methods: List[Tuple[int, int, str, Dict]] = []

    # Test all RF sizes from 1 to 5
    for size in range(1, 6):
        for combo in combinations(fnames, size):
            results = test_method(targets, list(combo))
            n_r1 = sum(1 for v in results.values() if v[0] < 5.5)
            sum_gt_rank = sum(v[1] for v in results.values())
            name = "+".join(combo)
            all_methods.append((n_r1, sum_gt_rank, name, results))

    # Sort: most R1 wins first, then lowest sum of GT ranks
    all_methods.sort(key=lambda x: (-x[0], x[1]))

    # Print results
    print(f"\n{'='*80}")
    print(f"EXHAUSTIVE RANK FUSION SEARCH: {len(all_methods)} methods tested")
    print(f"{'='*80}")

    # Summary by size
    for size in range(1, 6):
        methods_at_size = [m for m in all_methods if m[2].count("+") == size - 1]
        best = methods_at_size[0] if methods_at_size else None
        if best:
            print(f"  RF{size} best: {best[0]}/{n_targets} R1  {best[2]}")

    # Top 20 overall
    print(f"\nTOP 20 METHODS:")
    print(f"{'METHOD':<30} {'R1s':<4} {'SumR':<5} ", end="")
    for t in targets:
        print(f"{t.upper():<15}", end="")
    print()
    print("-" * (50 + 15 * n_targets))

    for n_r1, sum_r, name, results in all_methods[:20]:
        print(f"{name:<30} {n_r1}/{n_targets}  {sum_r:<5} ", end="")
        for t in targets:
            dcc, gr = results.get(t, (999, 999))
            mark = "R1" if dcc < 5.5 else f"R{gr}"
            print(f"{dcc:>5.1f}A {mark:<7} ", end="")
        print()

    # Best achievable
    best = all_methods[0]
    print(f"\nBEST METHOD: {best[2]}")
    print(f"  R1 on {best[0]}/{n_targets} targets, sum GT rank = {best[1]}")
    for t in targets:
        dcc, gr = best[3].get(t, (999, 999))
        print(f"  {t.upper()}: DCC={dcc:.1f}A rank={gr}")


if __name__ == "__main__":
    main()
