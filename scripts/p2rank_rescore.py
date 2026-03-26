#!/usr/bin/env python3
"""PRISM4D — P2Rank Rescoring.

Uses P2Rank to rescore PRISM consensus sites by matching each consensus
centroid to the nearest P2Rank-predicted pocket. P2Rank's ML-based
scoring replaces the physics-only GTCKL ranking.

PRISM detects (including cryptic/allosteric sites P2Rank misses).
P2Rank ranks (50-70% SR@1 on standard benchmarks).

Usage:
    python3 scripts/p2rank_rescore.py \\
        --pdb /path/to/structure.pdb \\
        --consensus /path/to/consensus_sites.json \\
        --out /path/to/rescored_consensus.json
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


P2RANK_BIN = Path(__file__).parent.parent / "tools" / "p2rank" / "p2rank_2.5.1" / "prank"


def find_p2rank() -> str:
    if P2RANK_BIN.exists():
        return str(P2RANK_BIN)
    raise FileNotFoundError(
        f"P2Rank not found at {P2RANK_BIN}. "
        f"Download from https://github.com/rdk/p2rank"
    )


def run_p2rank(pdb_path: str) -> List[Dict[str, Any]]:
    """Run P2Rank on a PDB and return predicted pockets."""
    binary = find_p2rank()
    with tempfile.TemporaryDirectory() as tmpdir:
        cmd = [binary, "predict", "-f", pdb_path, "-o", tmpdir]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        if result.returncode != 0:
            raise RuntimeError(f"P2Rank failed: {result.stderr[-300:]}")

        # Parse predictions CSV
        pdb_name = Path(pdb_path).name
        csv_path = Path(tmpdir) / f"{pdb_name}_predictions.csv"
        if not csv_path.exists():
            return []

        pockets = []
        with open(csv_path) as f:
            lines = f.readlines()

        if len(lines) < 2:
            return []

        header = [h.strip() for h in lines[0].split(",")]
        for line in lines[1:]:
            parts = line.split(",")
            if len(parts) < len(header):
                continue
            try:
                pocket = {
                    "rank": int(parts[header.index("rank")].strip()),
                    "score": float(parts[header.index("score")].strip()),
                    "probability": float(parts[header.index("probability")].strip()),
                    "center_x": float(parts[header.index("center_x")].strip()),
                    "center_y": float(parts[header.index("center_y")].strip()),
                    "center_z": float(parts[header.index("center_z")].strip()),
                }
                pockets.append(pocket)
            except (ValueError, IndexError):
                continue

        return pockets


def _dist(a: Tuple[float, ...], b: Tuple[float, ...]) -> float:
    return math.sqrt(sum((a[i] - b[i]) ** 2 for i in range(3)))


def rescore_consensus(
    consensus_path: str,
    pdb_path: str,
    match_threshold: float = 8.0,
) -> Dict[str, Any]:
    """Rescore consensus sites using P2Rank predictions.

    For each consensus site, find the nearest P2Rank pocket within
    match_threshold. If matched, use P2Rank score. If unmatched,
    the site is a PRISM-only detection (cryptic/allosteric) that
    P2Rank missed — keep it but mark as unscored.

    Returns the consensus data with sites re-ranked by P2Rank score.
    """
    # Load consensus
    with open(consensus_path) as f:
        consensus = json.load(f)

    sites = consensus.get("consensus_sites", [])
    if not sites:
        return consensus

    # Run P2Rank
    print(f"  Running P2Rank on {pdb_path}...")
    p2rank_pockets = run_p2rank(pdb_path)
    print(f"  P2Rank found {len(p2rank_pockets)} pockets")

    # Match consensus sites to P2Rank pockets
    for site in sites:
        centroid = site.get("centroid_mean", [0, 0, 0])
        best_match = None
        best_dist = match_threshold + 1

        for pocket in p2rank_pockets:
            p2_center = (pocket["center_x"], pocket["center_y"], pocket["center_z"])
            d = _dist(tuple(centroid), p2_center)
            if d < best_dist:
                best_dist = d
                best_match = pocket

        if best_match and best_dist <= match_threshold:
            site["p2rank_score"] = best_match["score"]
            site["p2rank_probability"] = best_match["probability"]
            site["p2rank_rank"] = best_match["rank"]
            site["p2rank_distance"] = round(best_dist, 2)
            site["p2rank_matched"] = True
        else:
            site["p2rank_score"] = -1.0
            site["p2rank_probability"] = 0.0
            site["p2rank_rank"] = 999
            site["p2rank_distance"] = round(best_dist, 2) if best_match else -1
            site["p2rank_matched"] = False

    # Re-rank: P2Rank-matched sites first (by P2Rank score desc),
    # then PRISM-only sites (by original consensus order)
    matched = [s for s in sites if s.get("p2rank_matched")]
    unmatched = [s for s in sites if not s.get("p2rank_matched")]

    matched.sort(key=lambda s: s["p2rank_score"], reverse=True)

    reranked = matched + unmatched
    for i, site in enumerate(reranked):
        site["final_rank"] = i + 1

    consensus["consensus_sites"] = reranked
    consensus["p2rank_pockets"] = len(p2rank_pockets)
    consensus["p2rank_matched"] = len(matched)
    consensus["p2rank_unmatched"] = len(unmatched)

    return consensus


def main() -> None:
    parser = argparse.ArgumentParser(
        description="PRISM4D P2Rank Rescoring"
    )
    parser.add_argument("--pdb", required=True, help="Input PDB file")
    parser.add_argument("--consensus", required=True,
                        help="Path to consensus_sites.json")
    parser.add_argument("--out", required=True, help="Output rescored JSON")
    parser.add_argument("--match-threshold", type=float, default=8.0,
                        help="Max distance (A) to match consensus to P2Rank pocket")
    args = parser.parse_args()

    print(f"[P2Rank Rescore] PDB: {args.pdb}")
    result = rescore_consensus(args.consensus, args.pdb, args.match_threshold)

    with open(args.out, "w") as f:
        json.dump(result, f, indent=2)

    matched = result.get("p2rank_matched", 0)
    unmatched = result.get("p2rank_unmatched", 0)
    print(f"  Matched: {matched}, PRISM-only: {unmatched}")
    print(f"  Top-3 final ranking:")
    for site in result["consensus_sites"][:3]:
        p2s = site.get("p2rank_score", -1)
        fr = site.get("final_rank", "?")
        m = "P2R" if site.get("p2rank_matched") else "PRISM"
        print(f"    R{fr}: p2rank_score={p2s:.2f} [{m}]")
    print(f"  Saved to {args.out}")


if __name__ == "__main__":
    main()
