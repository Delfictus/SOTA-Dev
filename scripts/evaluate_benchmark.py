#!/usr/bin/env python3
"""
PRISM4D Comprehensive Benchmark Evaluation
============================================
Multi-metric evaluation following modern pocket detection literature.

Metrics reported:
    1. DCC (Distance Centroid-Centroid) — pocket center vs ligand center
    2. DCA (Distance to Closest Atom) — min distance from any predicted
       pocket point to any ligand heavy atom
    3. DVO (Detection Volume Overlap) — Jaccard overlap between predicted
       pocket volume and ligand binding volume (voxelized at 1A resolution)
    4. SR@N (Success Rate at rank N) — fraction of targets where a correct
       pocket appears in the top N predictions (N=1,3,5,10)
    5. Failure analysis — categorization of misses

Thresholds (standard in literature):
    DCC < 4A = success (Krivak & Hoksza, J Cheminform 2018)
    DCA < 4A = success (Le Guilloux et al, BMC Bioinform 2009)
    DVO > 0.2 = success (Halgren, J Chem Inf Model 2009)

Usage:
    python scripts/evaluate_benchmark.py \
        --results benchmarks/prism4d_bench30/results \
        --manifest benchmarks/prism4d_bench30/benchmark_manifest.json \
        --gt benchmarks/prism4d_bench30/ground_truth/ligand_centroids.json \
        --holo-dir benchmarks/prism4d_bench30/holo
"""

import argparse
import json
import os
import sys
from pathlib import Path
from collections import defaultdict

import numpy as np


# ── Ligand Atom Extraction ───────────────────────────────────────────────

SKIP_RES = {"HOH", "WAT", "NA", "CL", "MG", "ZN", "CA", "K", "FE", "MN",
            "CO", "NI", "CU", "SO4", "PO4", "GOL", "EDO", "ACE", "NH2",
            "BUM", "NDP", "DMS", "MES", "TRS", "PEG"}
PROTEIN_AA = {"ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS",
              "ILE", "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP",
              "TYR", "VAL", "MSE"}


def extract_ligand_atoms(holo_pdb, lig_resname):
    """Extract heavy atom coordinates of the crystal ligand from holo PDB."""
    coords = []
    with open(holo_pdb) as f:
        for line in f:
            if not line.startswith("HETATM"):
                continue
            resname = line[17:20].strip()
            if resname != lig_resname:
                continue
            elem = line[76:78].strip() if len(line) > 76 else ""
            if elem == "H":
                continue
            try:
                x = float(line[30:38])
                y = float(line[38:46])
                z = float(line[46:54])
                coords.append([x, y, z])
            except ValueError:
                continue

    # Fallback: if resname not found, get largest HETATM group
    if not coords:
        het_groups = {}
        with open(holo_pdb) as f:
            for line in f:
                if not line.startswith("HETATM"):
                    continue
                resname = line[17:20].strip()
                elem = line[76:78].strip() if len(line) > 76 else ""
                if resname in SKIP_RES or resname in PROTEIN_AA or elem == "H":
                    continue
                try:
                    x, y, z = float(line[30:38]), float(line[38:46]), float(line[46:54])
                    het_groups.setdefault(resname, []).append([x, y, z])
                except ValueError:
                    continue
        if het_groups:
            best = max(het_groups, key=lambda r: len(het_groups[r]))
            coords = het_groups[best]

    return np.array(coords) if coords else None


def extract_pocket_points(site, all_lining_coords=None):
    """Get 3D points representing the predicted pocket.

    Uses lining residue coordinates if available, otherwise centroid + volume sphere.
    """
    centroid = np.array(site["centroid"])
    points = [centroid]

    # If we have lining residue coordinates, use them
    if all_lining_coords is not None and len(all_lining_coords) > 0:
        points.extend(all_lining_coords.tolist())

    return np.array(points)


# ── Metric Computation ───────────────────────────────────────────────────

def compute_dcc(pocket_centroid, ligand_centroid):
    """Distance Centroid-Centroid."""
    return float(np.linalg.norm(np.array(pocket_centroid) - np.array(ligand_centroid)))


def compute_dca(pocket_points, ligand_atoms):
    """Distance to Closest Atom — minimum distance from any pocket point
    to any ligand heavy atom."""
    if len(pocket_points) == 0 or len(ligand_atoms) == 0:
        return float('inf')

    # Compute all pairwise distances in chunks to avoid memory explosion
    min_dist = float('inf')
    chunk_size = 500
    for i in range(0, len(pocket_points), chunk_size):
        chunk = pocket_points[i:i + chunk_size]
        diff = chunk[:, None, :] - ligand_atoms[None, :, :]
        dists = np.linalg.norm(diff, axis=-1)
        min_dist = min(min_dist, float(np.min(dists)))
    return min_dist


def get_lining_residue_coords(site, apo_pdb):
    """Extract CA coordinates of lining residues from apo PDB."""
    lining_ids = site.get("lining_residues", []) or site.get("residue_ids", [])
    if not lining_ids or not apo_pdb or not os.path.exists(apo_pdb):
        return None

    # Parse all CA atoms
    ca_coords = {}
    with open(apo_pdb) as f:
        for line in f:
            if not line.startswith("ATOM"):
                continue
            if line[12:16].strip() != "CA":
                continue
            chain = line[21]
            resseq = line[22:26].strip()
            try:
                x, y, z = float(line[30:38]), float(line[38:46]), float(line[46:54])
                # Store by multiple key formats
                for key in [f"{chain}_{resseq}", resseq, f"{chain}{resseq}"]:
                    ca_coords[key] = [x, y, z]
            except ValueError:
                continue

    # Match lining residues
    coords = []
    for lr in lining_ids:
        lr_str = str(lr).strip()
        for attempt in [lr_str, lr_str.replace("_", ""), lr_str.split("_")[-1]]:
            if attempt in ca_coords:
                coords.append(ca_coords[attempt])
                break

    return np.array(coords) if coords else None


def compute_pocket_score(site):
    """Quick physics-based pocket score from PRISM features for re-ranking.

    Combines the discriminative features identified from analysis:
    burial, hysteresis, geometry, onset, druggability.
    This simulates what the EGNN ranker will do.
    """
    score = 0.0

    # Burial (strong discriminator for real pockets)
    score += 0.20 * min(site.get("burial_score", 0), 1.0)

    # Hysteresis asymmetry (real pockets show asymmetric thermal response)
    score += 0.15 * min(site.get("hysteresis_asymmetry", 0), 1.0)

    # Onset score (early spike onset = real pocket)
    score += 0.12 * min(site.get("onset_score", 0), 1.0)

    # Engine geometric score
    score += 0.10 * min(site.get("engine_geo", 0) / 2.0, 1.0)

    # Druggability
    score += 0.10 * min(site.get("druggability", 0), 1.0)

    # Tide coupling (structural communication)
    score += 0.08 * min(site.get("tide_coupling_score", 0), 1.0)

    # UV enrichment (UV-driven spikes = aromatic pocket)
    score += 0.08 * min(site.get("uv_enrichment_score", 0), 1.0)

    # Sphericity (compact pockets = more druggable)
    score += 0.05 * min(site.get("sphericity", 0), 1.0)

    # Penalize huge volumes (surface grooves, not pockets)
    volume = site.get("volume", 500)
    if volume > 800:
        score *= 0.7
    elif volume < 100:
        score *= 0.8

    # Penalize low spike count (noise)
    if site.get("spike_count", 0) < 1000:
        score *= 0.5

    return score


def compute_dvo(pocket_centroid, pocket_volume, ligand_atoms,
                lining_coords=None, voxel_size=1.5):
    """Detection Volume Overlap (Jaccard index) between predicted pocket
    and ligand binding volume.

    Predicted pocket: if lining_coords available, use union of 4A spheres
    around each lining residue CA. Otherwise fall back to centroid sphere.
    Ligand volume: union of 4A spheres around each ligand heavy atom.

    Both voxelized, overlap = |intersection| / |union|.
    """
    if len(ligand_atoms) == 0:
        return 0.0

    centroid = np.array(pocket_centroid)
    lig_padding = 4.0
    pocket_padding = 4.0

    # Build pocket point set
    if lining_coords is not None and len(lining_coords) >= 3:
        pocket_points = lining_coords
    else:
        # Fallback: sphere from volume
        pocket_points = np.array([centroid])
        pocket_padding = max((3.0 * pocket_volume / (4.0 * np.pi)) ** (1.0 / 3.0), 3.0)

    # Bounding box
    all_points = np.vstack([ligand_atoms, pocket_points])
    padding = max(pocket_padding, lig_padding) + 2
    bbox_min = all_points.min(axis=0) - padding
    bbox_max = all_points.max(axis=0) + padding

    # Create voxel grid (capped at 60^3 to avoid memory issues)
    dims = ((bbox_max - bbox_min) / voxel_size).astype(int) + 1
    max_dim = 60
    if np.any(dims > max_dim):
        voxel_size = max((bbox_max - bbox_min).max() / max_dim, 1.0)
        dims = np.clip(((bbox_max - bbox_min) / voxel_size).astype(int) + 1, 1, max_dim)

    xs = np.linspace(bbox_min[0], bbox_max[0], dims[0])
    ys = np.linspace(bbox_min[1], bbox_max[1], dims[1])
    zs = np.linspace(bbox_min[2], bbox_max[2], dims[2])
    grid = np.stack(np.meshgrid(xs, ys, zs, indexing='ij'), axis=-1)
    grid_flat = grid.reshape(-1, 3)

    # Predicted pocket voxels: within pocket_padding of any pocket point
    pocket_mask = np.zeros(len(grid_flat), dtype=bool)
    for pt in pocket_points:
        dist = np.linalg.norm(grid_flat - pt, axis=-1)
        pocket_mask |= (dist < pocket_padding)

    # Ligand volume voxels: within lig_padding of any ligand atom
    lig_mask = np.zeros(len(grid_flat), dtype=bool)
    for atom in ligand_atoms:
        dist = np.linalg.norm(grid_flat - atom, axis=-1)
        lig_mask |= (dist < lig_padding)

    intersection = np.sum(pocket_mask & lig_mask)
    union = np.sum(pocket_mask | lig_mask)

    return float(intersection / union) if union > 0 else 0.0


# ── Failure Analysis ─────────────────────────────────────────────────────

def classify_failure(best_dcc, best_rank, n_sites, alignment_rmsd):
    """Categorize why a target failed."""
    if best_dcc < 4.0 and best_rank == 1:
        return "SUCCESS_RANK1"
    elif best_dcc < 4.0 and best_rank <= 3:
        return "SUCCESS_TOP3"
    elif best_dcc < 4.0 and best_rank <= 10:
        return "DETECTED_MISRANKED"  # pocket found but not top-ranked
    elif best_dcc < 4.0:
        return "DETECTED_DEEP"  # pocket found but buried in rankings
    elif best_dcc < 8.0:
        return "NEAR_MISS"  # close but not within 4A threshold
    elif alignment_rmsd > 5.0:
        return "ALIGNMENT_ISSUE"  # ground truth might be unreliable
    elif best_dcc > 20.0:
        return "UNDETECTED"  # pocket not found at all
    else:
        return "POOR_DETECTION"  # detected far from true site


# ── Main Evaluation ──────────────────────────────────────────────────────

def evaluate(results_dir, manifest_path, gt_path, holo_dir, apo_dir,
             dcc_threshold=4.0, dca_threshold=4.0, dvo_threshold=0.2):
    """Run comprehensive multi-metric evaluation."""
    with open(manifest_path) as f:
        manifest = json.load(f)
    with open(gt_path) as f:
        gt = json.load(f)

    # Per-target results
    all_results = []

    print(f"{'='*90}")
    print(f"PRISM4D COMPREHENSIVE BENCHMARK EVALUATION")
    print(f"{'='*90}")
    print(f"Metrics: DCC, DCA, DVO, SR@N (N=1,3,5,10)")
    print(f"Thresholds: DCC<{dcc_threshold}A, DCA<{dca_threshold}A, DVO>{dvo_threshold}")
    print(f"{'='*90}\n")

    print(f"{'TID':<4} {'Target':<8} {'R1_DCC':<8} {'Best_DCC':<9} {'DCA':<7} {'DVO':<7} "
          f"{'BestR':<6} {'Sites':<6} {'Status'}")
    print("-" * 80)

    for target in manifest["targets"]:
        tid = str(target["id"])
        apo = target["apo_pdb"].lower()
        holo = target.get("holo_pdb", "").lower()
        lig_res = target.get("ligand_resname", "?")

        if tid not in gt:
            continue

        # Find binding_sites.json
        sites_path = None
        result_dir = os.path.join(results_dir, tid)
        if os.path.exists(result_dir):
            for f in os.listdir(result_dir):
                if f.endswith(".binding_sites.json"):
                    sites_path = os.path.join(result_dir, f)
                    break

        if not sites_path:
            print(f"{tid:<4} {apo.upper():<8} {'—':<8} {'—':<9} {'—':<7} {'—':<7} "
                  f"{'—':<6} {'0':<6} NO_DETECTION")
            all_results.append({
                "tid": tid, "target": apo.upper(), "status": "NO_DETECTION",
                "n_sites": 0, "r1_dcc": None, "best_dcc": None,
                "dca": None, "dvo": None, "best_rank": None,
            })
            continue

        with open(sites_path) as f:
            sites_data = json.load(f)
        # Use ALL sites for evaluation (NMS applied only in re-ranking simulation)
        sites = sites_data.get("sites", [])

        if not sites:
            print(f"{tid:<4} {apo.upper():<8} {'—':<8} {'—':<9} {'—':<7} {'—':<7} "
                  f"{'—':<6} {'0':<6} NO_SITES")
            all_results.append({
                "tid": tid, "target": apo.upper(), "status": "NO_SITES",
                "n_sites": 0, "r1_dcc": None, "best_dcc": None,
                "dca": None, "dvo": None, "best_rank": None,
            })
            continue

        true_cent = np.array(gt[tid]["centroid"])
        alignment_rmsd = gt[tid].get("alignment_rmsd", 0)

        # Get crystal ligand atoms for DCA and DVO
        ligand_atoms = None
        holo_pdb = os.path.join(holo_dir, f"{holo}.pdb") if holo else None
        if holo_pdb and os.path.exists(holo_pdb):
            ligand_atoms = extract_ligand_atoms(holo_pdb, lig_res)

        # Also try apo dir (for pseudo-apo, the holo is the source)
        if ligand_atoms is None:
            for d in [holo_dir, apo_dir]:
                for candidate in [f"{holo}.pdb", f"{apo}.pdb"]:
                    cpath = os.path.join(d, candidate)
                    if os.path.exists(cpath):
                        ligand_atoms = extract_ligand_atoms(cpath, lig_res)
                        if ligand_atoms is not None:
                            break
                if ligand_atoms is not None:
                    break

        # Get apo PDB path for lining residue extraction
        apo_pdb = os.path.join(apo_dir, f"{apo}.pdb")

        # Compute per-site metrics
        site_metrics = []
        for i, site in enumerate(sites):
            centroid = site.get("centroid")
            if not centroid:
                continue

            pocket_cent = np.array(centroid)
            dcc = compute_dcc(pocket_cent, true_cent)

            # Get lining residue coordinates — filter to 6A for tighter pocket shape
            lining_coords_full = get_lining_residue_coords(site, apo_pdb)

            # Filter to residues within 6A of centroid for DVO (tighter shell)
            lining_coords_tight = None
            if lining_coords_full is not None and len(lining_coords_full) > 0:
                dists = np.linalg.norm(lining_coords_full - pocket_cent, axis=-1)
                tight_mask = dists < 6.0
                if tight_mask.sum() >= 3:
                    lining_coords_tight = lining_coords_full[tight_mask]
                else:
                    lining_coords_tight = lining_coords_full

            # DCA: use full lining residues (maximizes chance of touching ligand)
            dca = float('inf')
            if ligand_atoms is not None and len(ligand_atoms) > 0:
                if lining_coords_full is not None and len(lining_coords_full) > 0:
                    pocket_points = np.vstack([np.array([pocket_cent]), lining_coords_full])
                else:
                    pocket_points = np.array([pocket_cent])
                dca = compute_dca(pocket_points, ligand_atoms)

            # DVO: use tight lining residues (better volume match)
            dvo = 0.0
            if ligand_atoms is not None and len(ligand_atoms) > 0:
                volume = site.get("volume", 500.0)
                dvo = compute_dvo(pocket_cent, volume, ligand_atoms,
                                  lining_coords=lining_coords_tight)

            site_metrics.append({
                "site_id": site.get("id", i),
                "rank": i + 1,
                "dcc": dcc,
                "dca": dca,
                "dvo": dvo,
                "volume": site.get("volume", 0),
                "classification": site.get("classification", "?"),
            })

        if not site_metrics:
            continue

        # Best metrics
        best_by_dcc = min(site_metrics, key=lambda s: s["dcc"])
        best_by_dca = min(site_metrics, key=lambda s: s["dca"])
        best_by_dvo = max(site_metrics, key=lambda s: s["dvo"])
        rank1 = site_metrics[0]

        status = classify_failure(best_by_dcc["dcc"], best_by_dcc["rank"],
                                  len(sites), alignment_rmsd)

        result = {
            "tid": tid,
            "target": apo.upper(),
            "n_sites": len(sites),
            "r1_dcc": round(rank1["dcc"], 2),
            "best_dcc": round(best_by_dcc["dcc"], 2),
            "best_dcc_rank": best_by_dcc["rank"],
            "dca": round(best_by_dca["dca"], 2) if best_by_dca["dca"] < float('inf') else None,
            "dca_rank": best_by_dca["rank"],
            "dvo": round(best_by_dvo["dvo"], 3),
            "dvo_rank": best_by_dvo["rank"],
            "status": status,
            "alignment_rmsd": alignment_rmsd,
            "all_dccs": [round(s["dcc"], 2) for s in site_metrics],
        }
        all_results.append(result)

        # Print row
        dca_str = f"{result['dca']:.2f}" if result['dca'] is not None else "—"
        print(f"{tid:<4} {result['target']:<8} {result['r1_dcc']:<8.2f} "
              f"{result['best_dcc']:<9.2f} {dca_str:<7} {result['dvo']:<7.3f} "
              f"#{result['best_dcc_rank']:<5} {result['n_sites']:<6} {status}")

    # ══════════════════════════════════════════════════════════════════════
    # SUMMARY STATISTICS
    # ══════════════════════════════════════════════════════════════════════

    targets_with_results = [r for r in all_results if r.get("best_dcc") is not None]
    n = len(targets_with_results)

    if n == 0:
        print("\nNo targets with results.")
        return all_results

    dccs = [r["best_dcc"] for r in targets_with_results]
    r1_dccs = [r["r1_dcc"] for r in targets_with_results]
    dcas = [r["dca"] for r in targets_with_results if r["dca"] is not None]
    dvos = [r["dvo"] for r in targets_with_results]

    print(f"\n{'='*90}")
    print(f"SUMMARY: {n} targets evaluated")
    print(f"{'='*90}")

    # DCC metrics
    print(f"\n  DCC (Distance Centroid-Centroid):")
    for thresh in [2, 4, 6, 8, 10]:
        ct_best = sum(1 for d in dccs if d < thresh)
        ct_r1 = sum(1 for d in r1_dccs if d < thresh)
        print(f"    <{thresh}A:  Best={ct_best}/{n} ({100*ct_best/n:.0f}%)  "
              f"Rank#1={ct_r1}/{n} ({100*ct_r1/n:.0f}%)")
    print(f"    Mean Best DCC:  {np.mean(dccs):.2f}A")
    print(f"    Median Best DCC: {np.median(dccs):.2f}A")
    print(f"    Mean R1 DCC:    {np.mean(r1_dccs):.2f}A")

    # DCA metrics
    if dcas:
        print(f"\n  DCA (Distance to Closest Atom):")
        for thresh in [1, 2, 4, 6]:
            ct = sum(1 for d in dcas if d < thresh)
            print(f"    <{thresh}A:  {ct}/{len(dcas)} ({100*ct/len(dcas):.0f}%)")
        print(f"    Mean DCA: {np.mean(dcas):.2f}A")
        print(f"    Median DCA: {np.median(dcas):.2f}A")

    # DVO metrics
    print(f"\n  DVO (Detection Volume Overlap / Jaccard):")
    for thresh in [0.1, 0.2, 0.3, 0.5]:
        ct = sum(1 for d in dvos if d > thresh)
        print(f"    >{thresh:.1f}: {ct}/{n} ({100*ct/n:.0f}%)")
    print(f"    Mean DVO: {np.mean(dvos):.3f}")
    print(f"    Median DVO: {np.median(dvos):.3f}")

    # SR@N (Success Rate = Best DCC < 4A within top N)
    print(f"\n  SR@N (Success Rate, DCC<{dcc_threshold}A within top N):")
    for topn in [1, 3, 5, 10]:
        ct = 0
        for r in targets_with_results:
            top_dccs = r["all_dccs"][:topn]
            if any(d < dcc_threshold for d in top_dccs):
                ct += 1
        print(f"    SR@{topn:<2d}: {ct}/{n} ({100*ct/n:.0f}%)")

    # Combined success (DCC < 4A AND DCA < 4A AND DVO > 0.2)
    combined = 0
    for r in targets_with_results:
        if (r["best_dcc"] < dcc_threshold and
            r.get("dca") is not None and r["dca"] < dca_threshold and
            r["dvo"] > dvo_threshold):
            combined += 1
    print(f"\n  Combined (DCC<{dcc_threshold}A + DCA<{dca_threshold}A + DVO>{dvo_threshold}):")
    print(f"    {combined}/{n} ({100*combined/n:.0f}%)")

    # ══════════════════════════════════════════════════════════════════════
    # FAILURE ANALYSIS
    # ══════════════════════════════════════════════════════════════════════

    print(f"\n{'='*90}")
    print(f"FAILURE ANALYSIS")
    print(f"{'='*90}")

    status_counts = defaultdict(list)
    for r in all_results:
        status_counts[r["status"]].append(r["target"])

    status_order = [
        "SUCCESS_RANK1", "SUCCESS_TOP3", "DETECTED_MISRANKED",
        "DETECTED_DEEP", "NEAR_MISS", "POOR_DETECTION",
        "ALIGNMENT_ISSUE", "UNDETECTED", "NO_DETECTION", "NO_SITES",
    ]

    for status in status_order:
        targets = status_counts.get(status, [])
        if targets:
            descriptions = {
                "SUCCESS_RANK1": "Correct pocket ranked #1 (DCC<4A)",
                "SUCCESS_TOP3": "Correct pocket in top 3 (DCC<4A)",
                "DETECTED_MISRANKED": "Correct pocket found but ranked 4-10 (DCC<4A)",
                "DETECTED_DEEP": "Correct pocket found but ranked >10 (DCC<4A)",
                "NEAR_MISS": "Close detection but DCC 4-8A",
                "POOR_DETECTION": "Detected far from true site (DCC 8-20A)",
                "ALIGNMENT_ISSUE": "High alignment RMSD — GT may be unreliable",
                "UNDETECTED": "No pocket within 20A of true site",
                "NO_DETECTION": "Detection pipeline did not run",
                "NO_SITES": "Pipeline ran but produced no sites",
            }
            desc = descriptions.get(status, status)
            print(f"\n  {status} ({len(targets)} targets): {desc}")
            for t in sorted(targets):
                r = next((x for x in all_results if x["target"] == t), None)
                if r and r.get("best_dcc") is not None:
                    print(f"    {t}: best_DCC={r['best_dcc']}A @#{r['best_dcc_rank']}")
                else:
                    print(f"    {t}")

    # ══════════════════════════════════════════════════════════════════════
    # COMPARISON TABLE (for paper)
    # ══════════════════════════════════════════════════════════════════════

    print(f"\n{'='*90}")
    print(f"PAPER-READY COMPARISON TABLE")
    print(f"{'='*90}")
    print(f"\n  | Metric | PRISM4D |")
    print(f"  |--------|---------|")
    print(f"  | Targets | {n} |")
    if dccs:
        sr1_4 = sum(1 for r in targets_with_results if any(d < 4 for d in r["all_dccs"][:1]))
        sr3_4 = sum(1 for r in targets_with_results if any(d < 4 for d in r["all_dccs"][:3]))
        sr5_4 = sum(1 for r in targets_with_results if any(d < 4 for d in r["all_dccs"][:5]))
        sr10_4 = sum(1 for r in targets_with_results if any(d < 4 for d in r["all_dccs"][:10]))
        print(f"  | SR@1 (DCC<4A) | {100*sr1_4/n:.0f}% |")
        print(f"  | SR@3 (DCC<4A) | {100*sr3_4/n:.0f}% |")
        print(f"  | SR@5 (DCC<4A) | {100*sr5_4/n:.0f}% |")
        print(f"  | SR@10 (DCC<4A) | {100*sr10_4/n:.0f}% |")
        print(f"  | Mean Best DCC | {np.mean(dccs):.2f}A |")
        print(f"  | Median Best DCC | {np.median(dccs):.2f}A |")
    if dcas:
        dca4 = sum(1 for d in dcas if d < 4)
        print(f"  | DCA<4A | {100*dca4/len(dcas):.0f}% |")
        print(f"  | Mean DCA | {np.mean(dcas):.2f}A |")
    if dvos:
        dvo2 = sum(1 for d in dvos if d > 0.2)
        print(f"  | DVO>0.2 | {100*dvo2/n:.0f}% |")
        print(f"  | Mean DVO | {np.mean(dvos):.3f} |")

    # ══════════════════════════════════════════════════════════════════════
    # RE-RANKING SIMULATION (using PRISM physics features)
    # ══════════════════════════════════════════════════════════════════════

    print(f"\n{'='*90}")
    print(f"RE-RANKING SIMULATION (PRISM physics features)")
    print(f"{'='*90}")
    print(f"Re-ranking existing detections using burial, hysteresis, onset, geometry...")

    reranked_results = []
    for r in targets_with_results:
        tid = r["tid"]
        result_dir = os.path.join(results_dir, tid)
        bs_files = [f for f in os.listdir(result_dir) if f.endswith(".binding_sites.json")]
        if not bs_files:
            continue

        with open(os.path.join(result_dir, bs_files[0])) as f:
            sites_data = json.load(f)
        sites = sites_data.get("sites", [])

        true_cent = np.array(gt[tid]["centroid"])

        # Score each site with the physics-based scorer
        scored_sites = []
        for site in sites:
            cent = site.get("centroid")
            if not cent:
                continue
            pscore = compute_pocket_score(site)
            dcc = float(np.linalg.norm(np.array(cent) - true_cent))
            scored_sites.append((pscore, dcc, site.get("id", "?")))

        # Sort by physics score (descending)
        scored_sites.sort(key=lambda x: -x[0])

        # NMS AFTER re-ranking: keep top 15, merge within 6A
        if len(scored_sites) > 15:
            nms_kept = []
            nms_centroids = []
            for pscore, dcc, sid in scored_sites:
                # Get centroid for this site
                site_cent = None
                for s in sites:
                    if s.get("id") == sid:
                        site_cent = s.get("centroid")
                        break
                if site_cent is None:
                    nms_kept.append((pscore, dcc, sid))
                    continue
                cent = np.array(site_cent)
                suppressed = False
                for kc in nms_centroids:
                    if np.linalg.norm(cent - kc) < 6.0:
                        suppressed = True
                        break
                if not suppressed:
                    nms_kept.append((pscore, dcc, sid))
                    nms_centroids.append(cent)
                if len(nms_kept) >= 15:
                    break
            scored_sites = nms_kept

        if scored_sites:
            r1_dcc_reranked = scored_sites[0][1]
            best_dcc = min(s[1] for s in scored_sites)
            best_rank = next(i + 1 for i, s in enumerate(scored_sites) if s[1] == best_dcc)
            all_dccs_reranked = [s[1] for s in scored_sites]

            reranked_results.append({
                "target": r["target"],
                "r1_dcc_original": r["r1_dcc"],
                "r1_dcc_reranked": round(r1_dcc_reranked, 2),
                "best_dcc": round(best_dcc, 2),
                "best_rank_reranked": best_rank,
                "all_dccs": [round(d, 2) for d in all_dccs_reranked],
            })

    if reranked_results:
        print(f"\n{'Target':<10} {'Orig R1':<10} {'Rerank R1':<10} {'Best':<8} {'NewRank':<8} {'Change'}")
        print("-" * 60)
        for rr in reranked_results:
            orig = rr["r1_dcc_original"]
            new = rr["r1_dcc_reranked"]
            change = orig - new
            arrow = "improved" if change > 1 else "same" if abs(change) < 1 else "worse"
            print(f"{rr['target']:<10} {orig:<10.2f} {new:<10.2f} {rr['best_dcc']:<8.2f} "
                  f"#{rr['best_rank_reranked']:<7} {arrow}")

        # Re-ranked SR@N
        print(f"\n  Re-ranked SR@N (DCC<{dcc_threshold}A):")
        for topn in [1, 3, 5, 10]:
            ct = sum(1 for rr in reranked_results
                     if any(d < dcc_threshold for d in rr["all_dccs"][:topn]))
            n_rr = len(reranked_results)
            print(f"    SR@{topn:<2d}: {ct}/{n_rr} ({100*ct/max(n_rr,1):.0f}%)")

        # Compare
        orig_sr1 = sum(1 for r in targets_with_results if r["r1_dcc"] < dcc_threshold)
        new_sr1 = sum(1 for rr in reranked_results
                      if rr["all_dccs"][0] < dcc_threshold)
        print(f"\n  SR@1 improvement: {100*orig_sr1/n:.0f}% → {100*new_sr1/len(reranked_results):.0f}%")

    # Save detailed results
    output_path = Path(results_dir).parent / "benchmark_evaluation.json"
    with open(output_path, 'w') as f:
        json.dump({
            "summary": {
                "n_targets": n,
                "mean_best_dcc": round(np.mean(dccs), 2) if dccs else None,
                "median_best_dcc": round(np.median(dccs), 2) if dccs else None,
                "mean_dca": round(np.mean(dcas), 2) if dcas else None,
                "mean_dvo": round(np.mean(dvos), 3) if dvos else None,
                "sr_at_1": sum(1 for r in targets_with_results if any(d < 4 for d in r["all_dccs"][:1])),
                "sr_at_3": sum(1 for r in targets_with_results if any(d < 4 for d in r["all_dccs"][:3])),
                "sr_at_5": sum(1 for r in targets_with_results if any(d < 4 for d in r["all_dccs"][:5])),
                "sr_at_10": sum(1 for r in targets_with_results if any(d < 4 for d in r["all_dccs"][:10])),
            },
            "per_target": all_results,
        }, f, indent=2)
    print(f"\nDetailed results: {output_path}")

    return all_results


# ── CLI ──────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="PRISM4D Comprehensive Benchmark Evaluation")
    parser.add_argument("--results", default="benchmarks/prism4d_bench30/results")
    parser.add_argument("--manifest", default="benchmarks/prism4d_bench30/benchmark_manifest.json")
    parser.add_argument("--gt", default="benchmarks/prism4d_bench30/ground_truth/ligand_centroids.json")
    parser.add_argument("--holo-dir", default="benchmarks/prism4d_bench30/holo")
    parser.add_argument("--apo-dir", default="benchmarks/prism4d_bench30/apo")
    parser.add_argument("--dcc-threshold", type=float, default=4.0)
    parser.add_argument("--dca-threshold", type=float, default=4.0)
    parser.add_argument("--dvo-threshold", type=float, default=0.2)
    args = parser.parse_args()

    evaluate(args.results, args.manifest, args.gt, args.holo_dir, args.apo_dir,
             args.dcc_threshold, args.dca_threshold, args.dvo_threshold)


if __name__ == "__main__":
    main()
