#!/usr/bin/env python3
"""PRISM4D — Physics-Based Pocket Ranker.

Post-processes PRISM engine output to re-rank sites using burial-weighted
physics scoring.  Replaces the G term in GTCKL with burial_score, keeping
T, K, L from the engine.

Formula:
    score = burial * (0.5 + vcs) * T * K * L

Physics rationale:
    burial  — deep pockets bind ligands, surface grooves don't
    vcs     — voxel contact score = pocket enclosure
    T       — thermodynamic signal (therm class + druggability)
    K       — kinetic signal (KCC driver quality)
    L       — localization factor

No ground truth used.  No learned weights.  Pure physical priors.

Usage:
    python3 scripts/prism_ranker.py \\
        --binding-sites /path/to/binding_sites.json \\
        [--out /path/to/reranked.json]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List


def rerank_sites(sites: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Re-rank sites using burial-weighted physics scoring."""
    for s in sites:
        burial = s.get("burial_score", 0.0)
        vcs = s.get("engine_vcs", 0.0)
        t = s.get("rank_T", 0.5)
        k = s.get("rank_K", 0.0)
        l = s.get("rank_L", 0.8)

        s["prism_rank_score"] = burial * (0.5 + vcs) * t * k * l

    sites.sort(key=lambda s: s["prism_rank_score"], reverse=True)

    for i, s in enumerate(sites):
        s["prism_rank"] = i + 1

    return sites


def main() -> None:
    parser = argparse.ArgumentParser(
        description="PRISM4D Physics-Based Pocket Ranker"
    )
    parser.add_argument("--binding-sites", required=True)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    with open(args.binding_sites) as f:
        data = json.load(f)

    is_wrapped = isinstance(data, dict) and "sites" in data
    sites = data["sites"] if is_wrapped else data

    sites = rerank_sites(sites)

    if is_wrapped:
        data["sites"] = sites

    if args.out:
        with open(args.out, "w") as f:
            json.dump(data, f, indent=2)
    else:
        for s in sites[:5]:
            print(
                f"  R{s['prism_rank']}: site {s.get('id','?')} "
                f"score={s['prism_rank_score']:.4f} "
                f"burial={s.get('burial_score',0):.3f} "
                f"vcs={s.get('engine_vcs',0):.2f}"
            )


if __name__ == "__main__":
    main()
