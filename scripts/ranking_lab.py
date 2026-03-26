#!/usr/bin/env python3
"""PRISM4D — Ranking Laboratory.

Systematic investigation of ranking formulas on blind targets.
All results logged to a matrix. No formula tuning on GT — only
evaluation. GT is used ONLY to measure DCC after ranking, never
to select or adjust the formula.

Methodology:
1. Define candidate formulas from PHYSICAL PRIORS only
2. Apply each formula to all 7 targets
3. Record SR@1, SR@3, per-target rank of GT site
4. Save full matrix to CSV for analysis

Usage:
    python3 scripts/ranking_lab.py
"""
import json, math, csv, sys
from pathlib import Path
from typing import Any, Dict, List, Callable, Tuple

# ── Ground truth (ONLY used for evaluation, never for formula selection) ──
GT = {
    "1jwp": [6.61, 13.531, 25.526],
    "1p38": [16.885, -3.551, -19.552],
    "5lar": [54.0, 68.1, 16.8],
    "2hnp": [27.09, 17.46, 12.15],
    "3l3n": [26.365, 11.222, 27.023],
    "1nna": [26.054, 17.671, 62.965],
    "2npq": [42.461, 31.126, 32.515],
}

PATHS = {
    "1jwp": "/tmp/prism_consensus/1jwp/rep_00/1jwp.binding_sites.json",
    "1p38": "/tmp/prism_consensus/1p38/rep_00/1p38.binding_sites.json",
    "2hnp": "/tmp/prism_consensus/2hnp/rep_00/2hnp.binding_sites.json",
    "5lar": "/tmp/prism_blind/5lar/5lar.binding_sites.json",
    "3l3n": "/tmp/prism_blind/3l3n/3l3n.binding_sites.json",
    "1nna": "/tmp/prism_blind/1nna/1nna.binding_sites.json",
    "2npq": "/tmp/prism_blind/2npq/2npq.binding_sites.json",
}

def dist(a, b):
    return math.sqrt(sum((a[i]-b[i])**2 for i in range(3)))

def pct_norm(vals):
    n = len(vals)
    if n <= 1: return [0.5] * n
    order = sorted(range(n), key=lambda i: vals[i])
    ranks = [0.0]*n
    for r, idx in enumerate(order):
        ranks[idx] = r / (n - 1)
    return ranks

# ── Load all target data once ──
def load_all():
    targets = {}
    for name, path in PATHS.items():
        p = Path(path)
        if not p.exists():
            continue
        with open(p) as f:
            data = json.load(f)
        sites = data.get("sites", data)
        targets[name] = sites
    return targets

# ── Candidate ranking formulas ──
# Each formula takes a site dict and optionally pre-computed percentiles.
# Returns a score (higher = better rank).
# NONE of these were designed by looking at GT results.
# They are derived from physical priors about what makes a binding pocket.

def make_formulas():
    formulas = {}

    # F0: GTCKL checkpoint (baseline)
    formulas["GTCKL"] = lambda s, **kw: s.get("rank_score", 0)

    # F1: burial-weighted (my recent attempt)
    formulas["burial*TCKL"] = lambda s, **kw: (
        s.get("burial_score", 0) * (0.5 + s.get("engine_vcs", 0))
        * s.get("rank_T", 0.5) * s.get("rank_K", 0) * s.get("rank_L", 0.8)
    )

    # F2: engine_geo * engine_chem (known good pair from earlier analysis)
    formulas["geo*chem"] = lambda s, **kw: (
        s.get("engine_geo", 0) * s.get("engine_chem", 0)
    )

    # F3: engine_geo * engine_vcs
    formulas["geo*vcs"] = lambda s, **kw: (
        s.get("engine_geo", 0) * s.get("engine_vcs", 0)
    )

    # F4: pure burial * T * K (no geometry at all)
    formulas["burial*T*K"] = lambda s, **kw: (
        s.get("burial_score", 0) * s.get("rank_T", 0.5) * s.get("rank_K", 0)
    )

    # F5: burial * chem * vcs (pocket chemistry in a deep hole)
    formulas["burial*chem*vcs"] = lambda s, **kw: (
        s.get("burial_score", 0) * s.get("engine_chem", 0)
        * (0.5 + s.get("engine_vcs", 0))
    )

    # F6: GTCKL but replace G with burial (keep everything else from engine)
    formulas["burial_G*TCKL"] = lambda s, **kw: (
        s.get("burial_score", 0) * s.get("rank_T", 0.5)
        * s.get("rank_C", 0) * s.get("rank_K", 0) * s.get("rank_L", 0.8)
    )

    # F7: geo + chem rank fusion (sum of individual ranks — non-multiplicative)
    formulas["RF2:geo+chem"] = "RANK_FUSION"

    # F8: druggability * burial (simple)
    formulas["drug*burial"] = lambda s, **kw: (
        s.get("druggability", 0) * s.get("burial_score", 0)
    )

    # F9: aromatic * burial * T (aromatic residues = binding hotspots)
    formulas["aro*burial*T"] = lambda s, **kw: (
        s.get("aromatic_score", 0) * s.get("burial_score", 0)
        * s.get("rank_T", 0.5)
    )

    # F10: max(GTCKL, burial*TCKL) — take best of both
    formulas["max(GTCKL,bTCKL)"] = lambda s, **kw: max(
        s.get("rank_score", 0),
        s.get("burial_score", 0) * (0.5 + s.get("engine_vcs", 0))
        * s.get("rank_T", 0.5) * s.get("rank_K", 0) * s.get("rank_L", 0.8)
    )

    # F11: GTCKL * (1 + burial) — burial as a boost not replacement
    formulas["GTCKL*(1+bur)"] = lambda s, **kw: (
        s.get("rank_score", 0) * (1.0 + s.get("burial_score", 0))
    )

    # F12: sqrt(GTCKL * burial_TCKL) — geometric mean
    formulas["geomean"] = lambda s, **kw: math.sqrt(max(
        s.get("rank_score", 0) * (
            s.get("burial_score", 0) * (0.5 + s.get("engine_vcs", 0))
            * s.get("rank_T", 0.5) * s.get("rank_K", 0) * s.get("rank_L", 0.8)
        ), 0))

    # F13: chem * vcs * T * K (no geometry, just pocket chemistry + physics)
    formulas["chem*vcs*T*K"] = lambda s, **kw: (
        s.get("engine_chem", 0) * (0.5 + s.get("engine_vcs", 0))
        * s.get("rank_T", 0.5) * s.get("rank_K", 0)
    )

    # F14: burial * sphericity * T * K (compact deep pockets)
    formulas["bur*sph*T*K"] = lambda s, **kw: (
        s.get("burial_score", 0) * s.get("sphericity", 0)
        * s.get("rank_T", 0.5) * s.get("rank_K", 0)
    )

    return formulas

# ── Evaluate one formula on one target ──
def evaluate(sites, gt, formula_fn, is_rank_fusion=False):
    if is_rank_fusion:
        # Rank fusion: geo + chem
        geo_vals = [s.get("engine_geo", 0) for s in sites]
        chem_vals = [s.get("engine_chem", 0) for s in sites]
        n = len(sites)
        geo_order = sorted(range(n), key=lambda i: geo_vals[i], reverse=True)
        chem_order = sorted(range(n), key=lambda i: chem_vals[i], reverse=True)
        geo_ranks = [0]*n
        chem_ranks = [0]*n
        for r, idx in enumerate(geo_order): geo_ranks[idx] = r+1
        for r, idx in enumerate(chem_order): chem_ranks[idx] = r+1
        fusion = [geo_ranks[i] + chem_ranks[i] for i in range(n)]
        ranked_idx = sorted(range(n), key=lambda i: fusion[i])
        ranked = [sites[i] for i in ranked_idx]
    else:
        for s in sites:
            s["_lab_score"] = formula_fn(s)
        ranked = sorted(sites, key=lambda s: s["_lab_score"], reverse=True)

    top1_dcc = dist(ranked[0]["centroid"], gt)
    best = min(sites, key=lambda s: dist(s["centroid"], gt))
    best_dcc = dist(best["centroid"], gt)
    # Find rank of best site
    for i, s in enumerate(ranked):
        if s.get("id") == best.get("id"):
            best_rank = i + 1
            break
    else:
        best_rank = 999

    return top1_dcc, best_dcc, best_rank

# ── Main ──
def main():
    print("Loading targets...")
    targets = load_all()
    formulas = make_formulas()
    target_names = sorted(targets.keys())
    formula_names = list(formulas.keys())

    print(f"Targets: {len(targets)}, Formulas: {len(formulas)}")

    # Results matrix
    results = []

    for fname in formula_names:
        f = formulas[fname]
        is_rf = (f == "RANK_FUSION")
        row = {"formula": fname}
        r1_count = 0
        r3_count = 0

        for tname in target_names:
            sites = [dict(s) for s in targets[tname]]  # copy
            gt = GT[tname]
            top1_dcc, best_dcc, best_rank = evaluate(
                sites, gt, f if not is_rf else None, is_rf
            )
            row[f"{tname}_top1"] = round(top1_dcc, 1)
            row[f"{tname}_rank"] = best_rank
            row[f"{tname}_best"] = round(best_dcc, 1)
            if top1_dcc < 5.5:
                r1_count += 1
            if best_rank <= 3:
                r3_count += 1

        row["SR@1"] = r1_count
        row["SR@3"] = r3_count
        results.append(row)

    # Sort by SR@1 desc, then SR@3 desc
    results.sort(key=lambda r: (-r["SR@1"], -r["SR@3"]))

    # Print
    print(f"\n{'FORMULA':<20} {'SR@1':>4} {'SR@3':>4} ", end="")
    for t in target_names:
        blind = "*" if t in ["5lar","3l3n","1nna","2npq"] else " "
        print(f" {t.upper()+blind:>6}", end="")
    print()
    print("-" * (30 + 7 * len(target_names)))

    for row in results:
        print(f"{row['formula']:<20} {row['SR@1']:>4} {row['SR@3']:>4} ", end="")
        for t in target_names:
            r = row[f"{t}_rank"]
            top1 = row[f"{t}_top1"]
            mark = f"R{r}" if top1 >= 5.5 else f"R{r}!"
            print(f" {mark:>6}", end="")
        print()

    # Save CSV
    csv_path = "scripts/ranking_matrix.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)
    print(f"\nSaved to {csv_path}")

if __name__ == "__main__":
    main()
