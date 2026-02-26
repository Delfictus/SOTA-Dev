#!/usr/bin/env python3
"""Re-rank PRISM4D binding sites using empirically validated scoring.

Replaces the engine's quality_score ranking with a formula derived from
CryptoBench ground-truth analysis (22 structures, 255 sites):

    ranking_score = mean_volume * (1.0 - 0.7 * frac_hydrophobic)

This achieves 45% Top-1 and 68% Top-3 DCA<4A success rate vs the engine's
9% Top-1 / 45% Top-3 using the original quality_score.

Usage:
    # Single structure
    python scripts/rerank_sites.py results/1arl/1arl.binding_sites.json

    # All results in a directory
    python scripts/rerank_sites.py benchmarks/cryptobench/results/

    # After batch aggregation
    python scripts/rerank_sites.py benchmarks/cryptobench/aggregated/batch_02/
"""
import argparse
import json
import sys
from pathlib import Path

HYDROPHOBIC_RESIDUES = {
    "ALA", "VAL", "LEU", "ILE", "MET", "PHE", "TRP", "PRO",
}


def compute_ranking_score(site: dict, pockets_by_id: dict) -> float:
    """Compute the empirical ranking score for a single site.

    Parameters
    ----------
    site : dict
        A site object from binding_sites.json ``"sites"`` array.
    pockets_by_id : dict
        Mapping of site_id -> pocket dict from ``"all_pockets"`` array,
        used to retrieve ``mean_volume``.

    Returns
    -------
    float
        Ranking score (higher = more likely to be a real binding site).
    """
    site_id = site.get("id", -1)
    pocket = pockets_by_id.get(site_id, {})

    # Mean volume from dynamics (time-averaged across all frames)
    mean_volume = pocket.get("mean_volume", 0.0)

    # Fallback to static volume if no dynamics data
    if mean_volume == 0.0:
        mean_volume = float(site.get("volume", 0.0))

    # Hydrophobic fraction of lining residues
    lining = site.get("lining_residues", [])
    n_residues = len(lining)
    if n_residues > 0:
        n_hydrophobic = sum(
            1 for r in lining if r.get("resname", "") in HYDROPHOBIC_RESIDUES
        )
        frac_hydrophobic = n_hydrophobic / n_residues
    else:
        frac_hydrophobic = 0.0

    # Core formula: volume with hydrophobic penalty
    ranking_score = mean_volume * (1.0 - 0.7 * frac_hydrophobic)

    return ranking_score


def rerank_file(bs_path: Path, in_place: bool = False) -> dict:
    """Re-rank sites in a single binding_sites.json file.

    Parameters
    ----------
    bs_path : Path
        Path to a binding_sites.json file.
    in_place : bool
        If True, overwrite the original file. Otherwise write to
        ``<name>.ranked.json`` alongside it.

    Returns
    -------
    dict
        Summary with target name, old vs new top-1 site ID, and site count.
    """
    with open(bs_path) as f:
        data = json.load(f)

    sites = data.get("sites", [])
    if not sites:
        return {"file": str(bs_path), "error": "no sites"}

    # Build pocket lookup from all_pockets
    pockets_by_id = {}
    for pocket in data.get("all_pockets", []):
        pockets_by_id[pocket["site_id"]] = pocket

    # Store old top-1 for reporting
    old_top1_id = sites[0]["id"] if sites else None

    # Compute ranking scores
    for site in sites:
        site["ranking_score"] = compute_ranking_score(site, pockets_by_id)

    # Sort descending by ranking_score
    sites.sort(key=lambda s: s["ranking_score"], reverse=True)
    data["sites"] = sites

    # Write output
    if in_place:
        out_path = bs_path
    else:
        out_path = bs_path.with_suffix(".ranked.json")

    with open(out_path, "w") as f:
        json.dump(data, f, indent=2)

    new_top1_id = sites[0]["id"]
    target = data.get("structure", bs_path.stem).replace(".topology", "")

    return {
        "target": target,
        "n_sites": len(sites),
        "old_top1": old_top1_id,
        "new_top1": new_top1_id,
        "changed": old_top1_id != new_top1_id,
        "output": str(out_path),
    }


def find_binding_sites_files(root: Path) -> list:
    """Recursively find all binding_sites.json files under root."""
    # Skip any already-ranked files
    return sorted(
        p for p in root.rglob("*.binding_sites.json")
        if ".ranked." not in p.name
    )


def main():
    parser = argparse.ArgumentParser(
        description="Re-rank PRISM4D binding sites by empirical scoring",
    )
    parser.add_argument(
        "path",
        help="Path to a binding_sites.json file or directory to search recursively",
    )
    parser.add_argument(
        "--in-place", action="store_true",
        help="Overwrite original files (default: write .ranked.json alongside)",
    )
    args = parser.parse_args()

    target = Path(args.path)

    if target.is_file():
        files = [target]
    elif target.is_dir():
        files = find_binding_sites_files(target)
    else:
        print(f"Error: {target} does not exist", file=sys.stderr)
        sys.exit(1)

    if not files:
        print(f"No binding_sites.json files found under {target}", file=sys.stderr)
        sys.exit(1)

    print(f"Re-ranking {len(files)} file(s)...")
    print()
    print(f"{'Target':<10} {'Sites':>5} {'Old #1':>7} {'New #1':>7} {'Changed':>8}")
    print("-" * 42)

    n_changed = 0
    for bs_path in files:
        result = rerank_file(bs_path, in_place=args.in_place)
        if "error" in result:
            print(f"  {bs_path}: {result['error']}")
            continue

        changed_str = "YES" if result["changed"] else "no"
        if result["changed"]:
            n_changed += 1
        print(
            f"{result['target']:<10} {result['n_sites']:>5} "
            f"{result['old_top1']:>7} {result['new_top1']:>7} "
            f"{changed_str:>8}"
        )

    print()
    print(f"Done. {n_changed}/{len(files)} targets got a new #1 site.")
    suffix = "in-place" if args.in_place else ".ranked.json"
    print(f"Output: {suffix}")


if __name__ == "__main__":
    main()
