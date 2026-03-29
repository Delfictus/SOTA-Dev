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


def translate_residues(output_dir, prefix, chain_map_path):
    """Translate top site residues back to original chain+resnum."""
    if not os.path.exists(chain_map_path):
        return

    with open(chain_map_path) as f:
        cm = json.load(f)

    bs_path = Path(output_dir) / f"{prefix}.binding_sites.json"
    if not bs_path.exists():
        return

    with open(bs_path) as f:
        sites = json.load(f).get("sites", [])
    if not sites:
        return

    top = sites[0]
    lining = top.get("lining_residues", [])
    if not lining:
        return

    print(f"\n  Top site residues (merged → original):")
    chains_seen = set()
    for r in lining[:10]:
        rid = r.get("resid", -1)
        translated = False
        for chain in cm.get("chains", []):
            ms = chain["merged_resnum_start"]
            me = chain["merged_resnum_end"]
            if ms <= rid <= me:
                offset = rid - ms
                orig = chain["original_resnum_start"] + offset
                is_interface = len(chains_seen) > 0 and chain["original_chain"] not in chains_seen
                chains_seen.add(chain["original_chain"])
                tag = " ← INTERFACE RESIDUE" if is_interface else ""
                print(f"    residue {rid} → chain {chain['original_chain']}, "
                      f"PDB {orig}{tag}")
                translated = True
                break
        if not translated:
            print(f"    residue {rid} → NOT IN CHAIN MAP")

    if len(chains_seen) > 1:
        print(f"  INTERFACE POCKET DETECTED: residues span chains "
              f"{' and '.join(sorted(chains_seen))}")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="PRISM-4D Output Postflight")
    parser.add_argument("output_dir", help="Output directory")
    parser.add_argument("prefix", help="Output file prefix")
    parser.add_argument("--chain-map", default=None, help="Chain map JSON (for multichain)")
    args = parser.parse_args()

    result = postflight(args.output_dir, args.prefix)

    if args.chain_map:
        translate_residues(args.output_dir, args.prefix, args.chain_map)

    sys.exit(result)


if __name__ == "__main__":
    main()
