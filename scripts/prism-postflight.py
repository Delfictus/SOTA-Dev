#!/usr/bin/env python3
"""PRISM-4D Output Postflight Validator.

Validates engine output after a run completes. Catches missing files,
empty results, and malformed output.

Usage:
    python3 scripts/prism-postflight.py <output_dir> <prefix>

Exit 0 = PASS, Exit 1 = FAIL.
"""
import json
import os
import sys
from pathlib import Path


def postflight(output_dir, prefix):
    od = Path(output_dir)
    fails = []
    warns = []

    # Required files
    required = {
        "binding_sites": od / f"{prefix}.binding_sites.json",
        "kcc_viz": od / f"{prefix}.kcc_visualization.json",
    }
    # Reranked is optional (only if P2Rank ran)
    reranked_path = od / f"{prefix}.reranked.json"

    for name, path in required.items():
        if not path.exists():
            fails.append(f"Missing required file: {path.name}")

    if fails:
        for f in fails:
            print(f"FAIL: {f}")
        return 1

    # Validate binding_sites.json
    with open(required["binding_sites"]) as f:
        bs = json.load(f)

    sites = bs.get("sites", [])
    if not sites:
        fails.append("binding_sites.json has 0 sites")
    else:
        # Check top site has required fields
        top = sites[0]
        required_fields = ["id", "centroid", "rank_score", "druggability",
                           "classification", "lining_residues"]
        for field in required_fields:
            if field not in top:
                warns.append(f"Top site missing field: {field}")

        # Check for duplicate residue_ids across top 5
        if len(sites) >= 5:
            top5_rids = [frozenset(s.get("residue_ids", [])) for s in sites[:5]]
            if len(set(top5_rids)) == 1 and top5_rids[0]:
                warns.append("Top 5 sites have identical residue_ids — "
                             "single pocket, low diversity")

    # Validate kcc_visualization.json
    with open(required["kcc_viz"]) as f:
        viz = json.load(f)

    viz_sites = viz.get("sites", [])
    viz_residues = viz.get("residues", [])
    if not viz_sites:
        warns.append("kcc_visualization.json has 0 sites")
    if not viz_residues:
        fails.append("kcc_visualization.json has 0 residues")

    # Check reranked if present
    if reranked_path.exists():
        with open(reranked_path) as f:
            rr = json.load(f)
        rr_sites = rr.get("sites", [])
        if not rr_sites:
            warns.append("reranked.json has 0 sites")
        else:
            top_rr = rr_sites[0]
            if "rank" not in top_rr:
                warns.append("reranked.json top site missing 'rank' field")

    # Count cryptic sites and pocket anomalies
    cryptic = bs.get("cryptic_sites", [])
    all_pockets = bs.get("all_pockets", [])
    malformed = sum(1 for p in all_pockets
                    if not isinstance(p, dict) or "centroid" not in p)
    if malformed > 0:
        warns.append(f"{malformed}/{len(all_pockets)} all_pockets entries "
                     f"missing centroid")

    # Print results
    print(f"Postflight: {prefix}")
    print(f"  Sites: {len(sites)}")
    print(f"  KCC residues: {len(viz_residues)}")
    print(f"  KCC sites: {len(viz_sites)}")
    print(f"  Cryptic sites: {len(cryptic)}")
    if reranked_path.exists():
        print(f"  Reranked: {len(rr_sites)} sites")

    if sites:
        top = sites[0]
        print(f"  Top site: id={top.get('id','?')} "
              f"score={top.get('rank_score', top.get('composite_audit_score', '?'))}")
        lining = top.get("lining_residues", [])
        if lining:
            res_str = ", ".join(f"{r.get('resname','?')}{r.get('resid','?')}"
                                for r in lining[:5])
            print(f"  Top lining (first 5): {res_str}")

    for w in warns:
        print(f"  WARN: {w}")

    if fails:
        for f in fails:
            print(f"  FAIL: {f}")
        print("POSTFLIGHT: FAIL")
        return 1

    print("POSTFLIGHT: PASS")
    return 0


def main():
    if len(sys.argv) < 3:
        print("Usage: prism-postflight.py <output_dir> <prefix>", file=sys.stderr)
        sys.exit(1)
    sys.exit(postflight(sys.argv[1], sys.argv[2]))


if __name__ == "__main__":
    main()
