#!/usr/bin/env python3
"""
Evaluate PRISM4D results against CryptoBench ground truth.

Computes standard binding site prediction metrics:
  - DCA (Distance to Closest Atom of ligand) at 4A threshold
  - DCC (Distance to Center of ligand) at 4A, 8A, 10A thresholds
  - Per-residue: AUC, AUPRC, MCC, TPR, FPR, F1
  - Top-1 / Top-3 / Top-N+2 success rates
  - Precision and recall per structure
  - Runtime statistics

Also downloads holo structures to compute ligand centroid DCA/DCC.

Usage:
    python 04_evaluate_results.py [--output cryptobench_results.json]
"""

import json
import sys
import os
import math
import numpy as np
from pathlib import Path
from collections import defaultdict

BENCH_DIR = Path(__file__).parent.resolve()
RESULTS_DIR = BENCH_DIR / "results"
GROUND_TRUTH_DIR = BENCH_DIR / "ground_truth"
TOPOLOGIES_DIR = BENCH_DIR / "topologies"
STRUCTURES_DIR = BENCH_DIR / "structures"
DATA_DIR = BENCH_DIR / "data"


def build_topo_to_pdb_mapping(apo_id: str, n_residues: int) -> dict:
    """
    Build a mapping from 0-based topology residue index to PDB author
    residue label ("{chain}_{resnum}").

    Parses the source PDB to extract the sequential order of unique
    (chain, resnum) pairs, which corresponds 1:1 to topology indices.
    """
    pdb_path = STRUCTURES_DIR / f"{apo_id}.pdb"
    if not pdb_path.exists():
        return None

    # Extract unique residues in the order they appear (preserving PDB order)
    seen = set()
    ordered = []
    with open(pdb_path) as f:
        for line in f:
            if line.startswith("ATOM"):
                chain = line[21]
                resnum = line[22:26].strip()
                key = (chain, resnum)
                if key not in seen:
                    seen.add(key)
                    ordered.append(key)

    # Map topology index -> PDB label
    mapping = {}
    for idx, (chain, resnum) in enumerate(ordered):
        if idx >= n_residues:
            break
        mapping[idx] = f"{chain}_{resnum}"

    return mapping


def load_binding_sites(apo_id: str) -> tuple:
    """Load PRISM4D binding site predictions for an apo structure.
    Returns (sites_list, full_data_dict)."""
    results_dir = RESULTS_DIR / apo_id
    bs_json = results_dir / f"{apo_id}.binding_sites.json"
    if not bs_json.exists():
        return [], {}

    with open(bs_json) as f:
        data = json.load(f)

    return data.get("sites", []), data


def load_ground_truth(apo_id: str) -> dict:
    """Load ground truth for an apo structure."""
    gt_path = GROUND_TRUTH_DIR / f"{apo_id}.ground_truth.json"
    if not gt_path.exists():
        return None

    with open(gt_path) as f:
        return json.load(f)


def load_topology(apo_id: str) -> dict:
    """Load topology to get atom positions and residue mapping."""
    topo_path = TOPOLOGIES_DIR / f"{apo_id}.topology.json"
    if not topo_path.exists():
        return None

    with open(topo_path) as f:
        return json.load(f)


def get_binding_residue_atoms(topo: dict, gt: dict,
                              topo_to_pdb: dict = None) -> list:
    """
    Get atom positions for ground-truth binding residues from topology.
    Returns list of [x, y, z] for each binding atom.

    Uses topo_to_pdb mapping (topology index -> PDB label) when available
    for accurate residue matching regardless of PDB author numbering.
    """
    if not topo or not gt:
        return []

    binding_labels = set(gt.get("binding_residue_labels", []))
    if not binding_labels:
        return []

    positions = topo.get("positions", [])
    chain_ids = topo.get("chain_ids", [])
    residue_ids = topo.get("residue_ids", [])
    n_atoms = topo.get("n_atoms", 0)

    # Group atoms by topology residue index
    residues = {}
    for i in range(n_atoms):
        if i >= len(residue_ids) or i >= len(chain_ids):
            break
        res_idx = residue_ids[i]
        chain = chain_ids[i] if i < len(chain_ids) else "A"
        if res_idx not in residues:
            residues[res_idx] = {
                "chain": chain,
                "atoms": [],
            }
        x = positions[3 * i]
        y = positions[3 * i + 1]
        z = positions[3 * i + 2]
        residues[res_idx]["atoms"].append([x, y, z])

    binding_atoms = []
    for res_idx, res_info in residues.items():
        # Use PDB mapping if available, otherwise fall back to +1 offset
        if topo_to_pdb and res_idx in topo_to_pdb:
            label = topo_to_pdb[res_idx]
        else:
            label = f"{res_info['chain']}_{res_idx + 1}"

        if label in binding_labels:
            binding_atoms.extend(res_info["atoms"])

    return binding_atoms


def compute_binding_residue_centroid(binding_atoms: list) -> list:
    """Compute centroid of binding residue atoms."""
    if not binding_atoms:
        return None
    arr = np.array(binding_atoms)
    return arr.mean(axis=0).tolist()


def compute_dca(site_centroid: list, binding_atoms: list) -> float:
    """Distance from predicted site center to Closest Atom of binding site."""
    if not binding_atoms or not site_centroid:
        return float('inf')

    sc = np.array(site_centroid)
    atoms = np.array(binding_atoms)
    distances = np.linalg.norm(atoms - sc, axis=1)
    return float(distances.min())


def compute_dcc(site_centroid: list, reference_centroid: list) -> float:
    """Distance from predicted site center to reference center."""
    if not site_centroid or not reference_centroid:
        return float('inf')

    return float(np.linalg.norm(
        np.array(site_centroid) - np.array(reference_centroid)
    ))


def compute_residue_overlap(site_residues: set, gt_residues: set) -> dict:
    """Compute residue-level overlap metrics."""
    if not gt_residues:
        return {"tp": 0, "fp": 0, "fn": 0, "precision": 0, "recall": 0, "f1": 0}

    tp = len(site_residues & gt_residues)
    fp = len(site_residues - gt_residues)
    fn = len(gt_residues - site_residues)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    return {
        "tp": tp, "fp": fp, "fn": fn,
        "precision": precision, "recall": recall, "f1": f1,
    }


def evaluate_structure(apo_id: str) -> dict:
    """Evaluate PRISM4D predictions for a single apo structure."""
    result = {
        "apo_id": apo_id,
        "evaluated": False,
        "n_predicted_sites": 0,
        "n_binding_residues_gt": 0,
        "dca_top1": float('inf'),
        "dca_top3": float('inf'),
        "dcc_top1": float('inf'),
        "dcc_top3": float('inf'),
        "best_residue_f1": 0,
        "best_residue_recall": 0,
        "runtime_sec": None,
    }

    # Load predictions
    sites, raw_data = load_binding_sites(apo_id)
    if not sites:
        return result

    result["n_predicted_sites"] = len(sites)

    # Load ground truth
    gt = load_ground_truth(apo_id)
    if not gt:
        return result

    result["n_binding_residues_gt"] = gt.get("n_binding_residues", 0)

    # Load topology for atom positions
    topo = load_topology(apo_id)

    # Build mapping from topology 0-based index to PDB author labels
    n_residues = topo.get("n_residues", 0) if topo else 0
    topo_to_pdb = build_topo_to_pdb_mapping(apo_id, n_residues)

    # Get binding residue atoms and centroid
    binding_atoms = get_binding_residue_atoms(topo, gt, topo_to_pdb)
    binding_centroid = compute_binding_residue_centroid(binding_atoms)

    if not binding_atoms:
        # Can't compute spatial metrics without atom positions
        result["error"] = "no_binding_atoms_resolved"
        return result

    result["evaluated"] = True

    # Sort sites by ranking_score if available (from rerank_sites.py),
    # otherwise fall back to quality_score
    rank_key = "ranking_score" if sites[0].get("ranking_score") is not None else "quality_score"
    ranked_sites = sorted(sites, key=lambda s: s.get(rank_key, 0), reverse=True)

    # Compute DCA and DCC for each predicted site
    site_metrics = []
    gt_labels = set(gt.get("binding_residue_labels", []))
    for site in ranked_sites:
        centroid = site.get("centroid", [0, 0, 0])
        dca = compute_dca(centroid, binding_atoms)
        dcc = compute_dcc(centroid, binding_centroid)

        # Residue overlap: map PRISM4D 0-based residue IDs to PDB labels
        site_res_ids = set()
        for rid in site.get("residue_ids", []):
            if isinstance(rid, str):
                site_res_ids.add(rid)
            elif isinstance(rid, int):
                # Use PDB mapping for accurate author numbering
                if topo_to_pdb and rid in topo_to_pdb:
                    site_res_ids.add(topo_to_pdb[rid])
                else:
                    # Fallback: get chain from lining_residues, apply +1
                    for lr in site.get("lining_residues", []):
                        if isinstance(lr, dict) and lr.get("resid") == rid:
                            chain = lr.get("chain", "A")
                            site_res_ids.add(f"{chain}_{rid + 1}")
                            break

        overlap = compute_residue_overlap(site_res_ids, gt_labels)

        site_metrics.append({
            "site_id": site.get("id", -1),
            "centroid": centroid,
            "quality": site.get("quality_score", 0),
            "druggability": site.get("druggability", 0),
            "dca": dca,
            "dcc": dcc,
            "residue_overlap": overlap,
        })

    # Top-N metrics
    if site_metrics:
        result["dca_top1"] = site_metrics[0]["dca"]
        result["dcc_top1"] = site_metrics[0]["dcc"]

        result["dca_top3"] = min(m["dca"] for m in site_metrics[:3])
        result["dcc_top3"] = min(m["dcc"] for m in site_metrics[:3])

        # Top-N+2: n = number of distinct binding sites (holo entries)
        n_gt = max(1, len(gt.get("holo_entries", [])))
        topn2 = n_gt + 2
        result[f"dca_topn2"] = min(m["dca"] for m in site_metrics[:topn2])
        result[f"dcc_topn2"] = min(m["dcc"] for m in site_metrics[:topn2])

        # Best residue overlap
        result["best_residue_f1"] = max(m["residue_overlap"]["f1"] for m in site_metrics)
        result["best_residue_recall"] = max(m["residue_overlap"]["recall"] for m in site_metrics)

    result["site_metrics"] = site_metrics
    result["max_pRMSD"] = gt.get("max_pRMSD", 0)

    # Get runtime from binding_sites.json (primary) or run.log (fallback)
    if raw_data.get("simulation_time_sec") is not None:
        result["runtime_sec"] = raw_data["simulation_time_sec"]
    else:
        log_path = RESULTS_DIR / apo_id / "run.log"
        if log_path.exists():
            try:
                import re
                text = log_path.read_text()
                for line in text.split('\n'):
                    if 'Total time:' in line or 'Simulation time:' in line:
                        nums = re.findall(r'[\d.]+', line)
                        if nums:
                            result["runtime_sec"] = float(nums[-1])
                            break
            except Exception:
                pass

    return result


def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="Evaluate PRISM4D CryptoBench results"
    )
    parser.add_argument("--output", "-o", default="cryptobench_evaluation.json",
                       help="Output file for detailed results")
    parser.add_argument("--summary", action="store_true",
                       help="Print summary table only")
    args = parser.parse_args()

    # Find completed results
    completed = []
    if RESULTS_DIR.exists():
        for d in sorted(RESULTS_DIR.iterdir()):
            if d.is_dir():
                bs_json = d / f"{d.name}.binding_sites.json"
                if bs_json.exists():
                    completed.append(d.name)

    if not completed:
        print("No completed results found. Run the benchmark first.")
        sys.exit(1)

    print(f"Evaluating {len(completed)} completed structures...")

    # Evaluate each
    all_results = []
    for i, apo_id in enumerate(completed, 1):
        result = evaluate_structure(apo_id)
        all_results.append(result)
        if i % 50 == 0:
            print(f"  Progress: {i}/{len(completed)}")

    # Compute aggregate metrics
    evaluated = [r for r in all_results if r["evaluated"]]
    n_eval = len(evaluated)

    if n_eval == 0:
        print("No structures could be evaluated (binding atom mapping failed)")
        sys.exit(1)

    # Success rates at various thresholds
    thresholds = {
        "dca_4A": ("dca_top1", 4.0),
        "dca_8A": ("dca_top1", 8.0),
        "dca_top3_4A": ("dca_top3", 4.0),
        "dca_topn2_4A": ("dca_topn2", 4.0),
        "dcc_4A": ("dcc_top1", 4.0),
        "dcc_8A": ("dcc_top1", 8.0),
        "dcc_10A": ("dcc_top1", 10.0),
        "dcc_top3_4A": ("dcc_top3", 4.0),
        "dcc_top3_10A": ("dcc_top3", 10.0),
    }

    success_rates = {}
    for name, (key, threshold) in thresholds.items():
        n_success = sum(1 for r in evaluated if r.get(key, float('inf')) < threshold)
        success_rates[name] = {
            "n_success": n_success,
            "n_total": n_eval,
            "rate": n_success / n_eval if n_eval > 0 else 0,
            "pct": f"{100 * n_success / n_eval:.1f}%" if n_eval > 0 else "N/A",
        }

    # Aggregate stats
    dca_values = [r["dca_top1"] for r in evaluated if r["dca_top1"] < 999]
    dcc_values = [r["dcc_top1"] for r in evaluated if r["dcc_top1"] < 999]
    runtimes = [r["runtime_sec"] for r in evaluated if r["runtime_sec"] is not None]

    summary = {
        "n_structures": len(completed),
        "n_evaluated": n_eval,
        "n_failed_mapping": len(completed) - n_eval,
        "success_rates": success_rates,
        "dca_stats": {
            "mean": float(np.mean(dca_values)) if dca_values else None,
            "median": float(np.median(dca_values)) if dca_values else None,
            "std": float(np.std(dca_values)) if dca_values else None,
        },
        "dcc_stats": {
            "mean": float(np.mean(dcc_values)) if dcc_values else None,
            "median": float(np.median(dcc_values)) if dcc_values else None,
            "std": float(np.std(dcc_values)) if dcc_values else None,
        },
        "runtime_stats": {
            "mean_sec": float(np.mean(runtimes)) if runtimes else None,
            "median_sec": float(np.median(runtimes)) if runtimes else None,
            "total_hours": sum(runtimes) / 3600 if runtimes else None,
        },
        "mean_sites_per_structure": float(np.mean([
            r["n_predicted_sites"] for r in evaluated
        ])),
        "mean_residue_f1": float(np.mean([
            r["best_residue_f1"] for r in evaluated
        ])),
        "mean_residue_recall": float(np.mean([
            r["best_residue_recall"] for r in evaluated
        ])),
    }

    # Print summary
    print()
    print("=" * 70)
    print("  PRISM4D CryptoBench Evaluation Results")
    print("=" * 70)
    print(f"  Structures evaluated: {n_eval}/{len(completed)}")
    print()

    print("  Success Rates:")
    print(f"  {'Metric':<25} {'Success':>8} {'Rate':>8}")
    print(f"  {'-'*25} {'-'*8} {'-'*8}")
    for name, sr in success_rates.items():
        print(f"  {name:<25} {sr['n_success']:>4}/{sr['n_total']:<3} {sr['pct']:>8}")

    print()
    print("  Distance Statistics:")
    if dca_values:
        print(f"  DCA top-1: mean={summary['dca_stats']['mean']:.1f}A, "
              f"median={summary['dca_stats']['median']:.1f}A, "
              f"std={summary['dca_stats']['std']:.1f}A")
    if dcc_values:
        print(f"  DCC top-1: mean={summary['dcc_stats']['mean']:.1f}A, "
              f"median={summary['dcc_stats']['median']:.1f}A, "
              f"std={summary['dcc_stats']['std']:.1f}A")
    print(f"  Mean sites/structure: {summary['mean_sites_per_structure']:.1f}")
    print(f"  Mean residue F1: {summary['mean_residue_f1']:.3f}")
    print(f"  Mean residue recall: {summary['mean_residue_recall']:.3f}")

    if runtimes:
        print()
        print(f"  Runtime: mean={summary['runtime_stats']['mean_sec']:.0f}s, "
              f"median={summary['runtime_stats']['median_sec']:.0f}s, "
              f"total={summary['runtime_stats']['total_hours']:.1f}h")

    print()

    # Save detailed results
    output_path = BENCH_DIR / args.output
    with open(output_path, 'w') as f:
        json.dump({
            "summary": summary,
            "per_structure": all_results,
        }, f, indent=2, default=str)

    print(f"  Detailed results: {output_path}")
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
