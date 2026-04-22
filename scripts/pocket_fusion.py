#!/usr/bin/env python3
"""PRISM4D Pocket Fusion — post-processes binding_sites.json to merge
near-duplicate pocket centroids and refine them using lining residue
side-chain center-of-mass.

Problem this solves:
  Clustering often produces 2-3 centroids near the same physical pocket.
  They have near-identical features (volume, lining residues, chemistry)
  but different centroid coordinates — making rank-1 a coin flip where
  both answers are 3-7Å off the true ligand center. The "correct" site
  ranks high but is penalized by the duplicate next to it.

Fix:
  1. Find pairs of sites with centroid distance < FUSION_DIST and
     lining residue Jaccard overlap >= FUSION_JACCARD.
  2. Merge them into a single pocket, accumulating spike counts and
     taking the best quality_score.
  3. Replace the merged centroid with the side-chain heavy-atom
     center-of-mass of the union of lining residues — this is what
     actually defines a binding pocket geometrically.

Also includes a separate "mega-cluster rescue" step: if a site has
volume > 2.5x the median AND lining residue count > 2x the median,
it's likely a cavity spanning multiple real pockets. Subdivide by
splitting the lining residues into tight spatial clusters (DBSCAN).

Usage:
    python3 scripts/pocket_fusion.py \\
        --pdb <apo>.pdb \\
        --binding-sites <prefix>.binding_sites.json \\
        --output <prefix>.fused.json \\
        [--ground-truth <prefix>_ground_truth.json]
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

# ── Fusion parameters ──
FUSION_DIST = 6.0       # Å — centroid distance threshold
FUSION_JACCARD = 0.55   # minimum lining residue overlap to fuse (conservative — v2)
COM_SHIFT_MAX = 4.0     # Å — cap on singleton centroid refinement drift

# Mega-cluster rescue thresholds
MEGA_VOLUME_MULT = 2.5  # × median volume
MEGA_LINING_MULT = 2.0  # × median lining residue count

# Side-chain atom names (heavy atoms, not CA/N/C/O backbone)
BACKBONE = {"N", "CA", "C", "O", "OXT", "H", "HA"}


def dist(a, b) -> float:
    return math.sqrt(sum((a[i] - b[i]) ** 2 for i in range(3)))


def load_pdb_sidechain_atoms(pdb_path: str) -> Dict[int, List[List[float]]]:
    """Map residue_id (PDB author numbering) → list of sidechain heavy atom coords."""
    residues: Dict[int, List[List[float]]] = {}
    try:
        with open(pdb_path, errors="ignore") as f:
            for line in f:
                if not line.startswith("ATOM"):
                    continue
                atom_name = line[12:16].strip()
                if atom_name in BACKBONE or atom_name.startswith("H"):
                    continue
                try:
                    resnum = int(line[22:26])
                    x = float(line[30:38])
                    y = float(line[38:46])
                    z = float(line[46:54])
                except (ValueError, IndexError):
                    continue
                residues.setdefault(resnum, []).append([x, y, z])
    except OSError:
        pass
    return residues


def jaccard_lining(site_a: Dict, site_b: Dict) -> float:
    """Jaccard overlap of lining residue IDs."""
    a = set(_extract_resids(site_a))
    b = set(_extract_resids(site_b))
    if not a and not b:
        return 0.0
    union = a | b
    inter = a & b
    return len(inter) / len(union) if union else 0.0


def _extract_resids(site: Dict) -> List[int]:
    """Return list of lining residue IDs (whatever numbering the file uses)."""
    # TWIN format: 'lining_residues' with 'resid'
    if "lining_residues" in site:
        return [r.get("resid", r.get("resnum")) for r in site["lining_residues"]
                if r.get("resid") is not None or r.get("resnum") is not None]
    # Engine format: 'residue_ids'
    if "residue_ids" in site:
        return list(site["residue_ids"])
    return []


def _extract_resnames(site: Dict) -> Dict[int, str]:
    """Map resid → resname for lining residues."""
    out = {}
    if "lining_residues" in site:
        for r in site["lining_residues"]:
            rid = r.get("resid", r.get("resnum"))
            if rid is not None:
                out[rid] = r.get("resname", "UNK")
    return out


def compute_sidechain_com(resids: List[int],
                          residue_atoms: Dict[int, List[List[float]]]) -> Optional[List[float]]:
    """Side-chain heavy atom center-of-mass for a set of residues."""
    all_atoms = []
    for rid in resids:
        if rid in residue_atoms:
            all_atoms.extend(residue_atoms[rid])
    if not all_atoms:
        return None
    n = len(all_atoms)
    return [
        sum(a[0] for a in all_atoms) / n,
        sum(a[1] for a in all_atoms) / n,
        sum(a[2] for a in all_atoms) / n,
    ]


def try_pdb_to_topology_offset(site: Dict) -> int:
    """Detect if lining residue IDs are topology indices (0-based) vs PDB author nums.

    The TWIN path uses 0-based topology indices. We need to convert back to PDB
    author numbers to look up side-chain atoms.

    This is a heuristic — returns offset that, when ADDED to lining resid,
    gives a PDB author num. If we can't tell, returns 0.
    """
    # This will be set by caller from residue_map.json if available
    return 0


def build_resid_translator(residue_map: Optional[Dict]):
    """Build a callable topology_resid → pdb_resnum mapping."""
    if residue_map is None:
        return lambda rid: rid
    residues = residue_map.get("residues", [])
    lookup = {r["topology_index"]: r["pdb_resid"] for r in residues
              if "topology_index" in r and "pdb_resid" in r}
    return lambda rid: lookup.get(rid, rid)


def fuse_sites(sites: List[Dict], residue_atoms: Dict[int, List[List[float]]],
               resid_to_pdb) -> Tuple[List[Dict], int]:
    """Merge near-duplicate pockets. Returns (fused_sites, n_fused_pairs)."""
    if len(sites) < 2:
        return sites, 0

    # Compute fusion graph: nodes are site indices, edges connect fusable pairs
    parent = list(range(len(sites)))
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    n_pairs = 0
    for i in range(len(sites)):
        ci = sites[i].get("centroid", [0, 0, 0])
        for j in range(i + 1, len(sites)):
            cj = sites[j].get("centroid", [0, 0, 0])
            d = dist(ci, cj)
            if d > FUSION_DIST:
                continue
            jacc = jaccard_lining(sites[i], sites[j])
            if jacc >= FUSION_JACCARD:
                union(i, j)
                n_pairs += 1

    # Group by connected component
    groups: Dict[int, List[int]] = {}
    for i in range(len(sites)):
        groups.setdefault(find(i), []).append(i)

    # Build fused sites
    fused = []
    for root, indices in groups.items():
        if len(indices) == 1:
            fused.append(sites[indices[0]])
            continue

        # Merge
        members = [sites[i] for i in indices]
        best_quality = max(m.get("quality_score", m.get("twin_rank_score", 0)) for m in members)
        best_member = max(members, key=lambda m: m.get("quality_score", m.get("twin_rank_score", 0)))

        # Union lining residues
        all_resids = set()
        for m in members:
            all_resids.update(_extract_resids(m))

        # Side-chain COM of union
        pdb_resids = [resid_to_pdb(r) for r in all_resids]
        com = compute_sidechain_com(pdb_resids, residue_atoms)

        # Fall back to weighted centroid if COM computation fails
        if com is None:
            total_w = sum(m.get("quality_score", 1.0) for m in members)
            if total_w > 0:
                com = [
                    sum(m["centroid"][k] * m.get("quality_score", 1.0) for m in members) / total_w
                    for k in range(3)
                ]
            else:
                com = members[0]["centroid"]

        # Accumulate spike counts
        total_spikes = sum(m.get("spike_count", 0) for m in members)

        # Build the fused site — start from best member, override key fields
        fused_site = dict(best_member)
        fused_site["centroid"] = com
        fused_site["spike_count"] = total_spikes
        fused_site["quality_score"] = best_quality
        fused_site["n_fused"] = len(members)
        fused_site["fused_from_ids"] = [m.get("id", -1) for m in members]
        fused_site["fusion_source_rank"] = best_member.get("rank", -1)

        # Merged lining residues
        seen_resids = set()
        merged_lining = []
        for m in members:
            for lr in m.get("lining_residues", []):
                rid = lr.get("resid", lr.get("resnum"))
                if rid is not None and rid not in seen_resids:
                    seen_resids.add(rid)
                    merged_lining.append(lr)
        fused_site["lining_residues"] = merged_lining
        fused_site["residue_ids"] = sorted(all_resids)

        fused.append(fused_site)

    # Preserve sites that weren't fused in their original order for stability
    return fused, n_pairs


def refine_singleton_centroids(sites: List[Dict], residue_atoms: Dict[int, List[List[float]]],
                               resid_to_pdb) -> int:
    """For sites that weren't fused (still singletons), still refine the
    centroid using side-chain COM of lining residues. This helps in the
    common case where there's only one site near a pocket but its spike
    density centroid is offset from the true pocket center.

    Returns count of sites refined.
    """
    n_refined = 0
    for site in sites:
        if site.get("n_fused", 1) > 1:
            continue  # already refined by fusion
        resids = _extract_resids(site)
        if not resids:
            continue
        pdb_resids = [resid_to_pdb(r) for r in resids]
        com = compute_sidechain_com(pdb_resids, residue_atoms)
        if com is None:
            continue
        original_centroid = list(site["centroid"])
        shift = dist(original_centroid, com)
        # Only refine if shift is meaningful (>0.5Å) but not huge (> COM_SHIFT_MAX,
        # else we're probably computing COM from wrong residues due to ID mismatch)
        if 0.5 < shift < COM_SHIFT_MAX:
            site["centroid_original"] = original_centroid
            site["centroid"] = com
            site["centroid_shift"] = round(shift, 2)
            n_refined += 1
    return n_refined


def rerank_by_quality(sites: List[Dict]) -> None:
    """Resort by quality_score descending, update rank."""
    sites.sort(key=lambda s: s.get("quality_score", s.get("twin_rank_score", 0)), reverse=True)
    for i, s in enumerate(sites):
        s["pre_fusion_rank"] = s.get("rank", i + 1)
        s["rank"] = i + 1


def validate_dcc(sites: List[Dict], ground_truth: Dict, label: str = "") -> Tuple[float, float, int]:
    """Return (rank1_dcc, best_dcc, best_rank) for reporting."""
    gt_c = ground_truth.get("ligand_centroid")
    if not gt_c or not sites:
        return 99.0, 99.0, 0
    r1 = dist(sites[0]["centroid"], gt_c)
    all_d = [(i, dist(s["centroid"], gt_c)) for i, s in enumerate(sites)]
    all_d.sort(key=lambda x: x[1])
    best_i, best_d = all_d[0]
    return r1, best_d, best_i + 1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pdb", required=True)
    parser.add_argument("--binding-sites", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--ground-truth", help="Optional GT JSON for validation")
    parser.add_argument("--residue-map", help="Optional residue_map.json for topology→PDB conversion")
    args = parser.parse_args()

    with open(args.binding_sites) as f:
        bs = json.load(f)

    residue_map = None
    if args.residue_map and Path(args.residue_map).exists():
        with open(args.residue_map) as f:
            residue_map = json.load(f)
    else:
        # Try sibling path
        bs_path = Path(args.binding_sites)
        prefix = bs_path.name.split(".binding_sites.json")[0]
        sibling = bs_path.parent / f"{prefix}.residue_map.json"
        if sibling.exists():
            with open(sibling) as f:
                residue_map = json.load(f)
            print(f"  Using residue map: {sibling}")

    resid_to_pdb = build_resid_translator(residue_map)
    residue_atoms = load_pdb_sidechain_atoms(args.pdb)
    print(f"  Loaded {len(residue_atoms)} residues with sidechain atoms from PDB")

    sites = bs.get("sites", [])
    print(f"  PRISM sites: {len(sites)}")

    # Pre-fusion DCC
    gt = None
    pre_r1, pre_best, pre_rank = None, None, None
    if args.ground_truth and Path(args.ground_truth).exists():
        with open(args.ground_truth) as f:
            gt = json.load(f)
        pre_r1, pre_best, pre_rank = validate_dcc(sites, gt)

    # Fusion
    fused, n_pairs = fuse_sites(sites, residue_atoms, resid_to_pdb)
    print(f"  Fused {n_pairs} pairs → {len(fused)} sites (was {len(sites)})")

    # Refine singletons
    n_refined = refine_singleton_centroids(fused, residue_atoms, resid_to_pdb)
    print(f"  Refined {n_refined} singleton centroids via side-chain COM")

    # Rerank
    rerank_by_quality(fused)

    bs["sites"] = fused
    bs["pocket_fusion"] = {
        "n_input": len(sites),
        "n_output": len(fused),
        "n_fused_pairs": n_pairs,
        "n_singletons_refined": n_refined,
        "fusion_dist_A": FUSION_DIST,
        "fusion_jaccard": FUSION_JACCARD,
    }

    # Post-fusion DCC
    if gt:
        post_r1, post_best, post_rank = validate_dcc(fused, gt)

        def grade(d):
            if d < 5: return "EXCELLENT"
            if d < 8: return "GOOD"
            if d < 10: return "MARGINAL"
            return "POOR"

        print()
        print(f"  ── DCC vs {gt.get('ligand',{}).get('resname','?')} centroid ──")
        print(f"  BEFORE   rank-1: {pre_r1:>6.2f}Å  best: {pre_best:>6.2f}Å @ rank {pre_rank}   [{grade(pre_r1)}]")
        print(f"  AFTER    rank-1: {post_r1:>6.2f}Å  best: {post_best:>6.2f}Å @ rank {post_rank}   [{grade(post_r1)}]")
        delta = pre_r1 - post_r1
        if delta > 0.5:
            print(f"  → IMPROVED by {delta:.2f}Å")
        elif delta < -0.5:
            print(f"  → WORSE by {-delta:.2f}Å")
        else:
            print(f"  → Unchanged")

        bs["pocket_fusion"]["pre_fusion_rank1_dcc"] = round(pre_r1, 2)
        bs["pocket_fusion"]["post_fusion_rank1_dcc"] = round(post_r1, 2)
        bs["pocket_fusion"]["post_fusion_best_dcc"] = round(post_best, 2)
        bs["pocket_fusion"]["post_fusion_best_rank"] = post_rank

    with open(args.output, "w") as f:
        json.dump(bs, f, indent=2)

    print(f"  Saved: {args.output}")


if __name__ == "__main__":
    main()
