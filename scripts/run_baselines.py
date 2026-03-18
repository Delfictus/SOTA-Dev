#!/usr/bin/env python3
"""
Run P2Rank and fpocket on the BENCH60 benchmark for head-to-head comparison.

Outputs per-target DCC, DCA, DVO, SR@N for each method alongside PRISM4D.
"""

import json
import os
import subprocess
import sys
from pathlib import Path
import numpy as np

BENCH_DIR = "benchmarks/prism4d_bench30"
APO_DIR = f"{BENCH_DIR}/apo"
HOLO_DIR = f"{BENCH_DIR}/holo"
MANIFEST = f"{BENCH_DIR}/benchmark_manifest.json"
GT_PATH = f"{BENCH_DIR}/ground_truth/ligand_centroids.json"

PRANK = "tools/p2rank/p2rank_2.5.1/prank"
P2RANK_OUT = "/tmp/p2rank_bench60"
FPOCKET_OUT = "/tmp/fpocket_bench60"


def run_p2rank(apo_pdb, output_dir):
    """Run P2Rank on a single PDB."""
    os.makedirs(output_dir, exist_ok=True)
    result = subprocess.run(
        ["bash", PRANK, "predict", "-f", apo_pdb, "-o", output_dir],
        capture_output=True, text=True, timeout=120
    )
    # Parse predictions CSV
    pdb_name = Path(apo_pdb).stem
    pred_file = os.path.join(output_dir, f"{pdb_name}.pdb_predictions.csv")
    if not os.path.exists(pred_file):
        return []

    import csv
    pockets = []
    with open(pred_file) as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                pockets.append({
                    "rank": int(row["  rank"].strip()),
                    "score": float(row["   score"].strip()),
                    "centroid": [
                        float(row["   center_x"].strip()),
                        float(row["   center_y"].strip()),
                        float(row["   center_z"].strip()),
                    ],
                })
            except (ValueError, KeyError):
                continue
    return pockets


def run_fpocket(apo_pdb, output_dir):
    """Run fpocket on a single PDB."""
    os.makedirs(output_dir, exist_ok=True)
    pdb_name = Path(apo_pdb).stem

    # Copy PDB to output dir (fpocket writes next to input)
    import shutil
    local_pdb = os.path.join(output_dir, f"{pdb_name}.pdb")
    shutil.copy2(apo_pdb, local_pdb)

    result = subprocess.run(
        ["fpocket", "-f", local_pdb],
        capture_output=True, text=True, timeout=120,
        cwd=output_dir
    )

    # Parse fpocket output
    out_dir = os.path.join(output_dir, f"{pdb_name}_out")
    info_file = os.path.join(out_dir, f"{pdb_name}_info.txt")

    pockets = []
    if os.path.exists(out_dir):
        # Read pocket PDB files for centroids
        for pdb_file in sorted(Path(out_dir).glob("pocket*_atm.pdb")):
            pocket_num = int(pdb_file.stem.replace("pocket", "").replace("_atm", ""))
            coords = []
            with open(pdb_file) as f:
                for line in f:
                    if line.startswith("ATOM") or line.startswith("HETATM"):
                        try:
                            x = float(line[30:38])
                            y = float(line[38:46])
                            z = float(line[46:54])
                            coords.append([x, y, z])
                        except ValueError:
                            continue
            if coords:
                centroid = np.mean(coords, axis=0).tolist()
                pockets.append({
                    "rank": pocket_num,
                    "centroid": centroid,
                })

    # Sort by rank
    pockets.sort(key=lambda p: p["rank"])
    # Reassign ranks from 1
    for i, p in enumerate(pockets):
        p["rank"] = i + 1

    return pockets


SKIP_RES = {"HOH","WAT","NA","CL","MG","ZN","CA","K","FE","MN","CO","NI",
            "CU","SO4","PO4","GOL","EDO","ACE","NH2","BUM","NDP","DMS","MES"}
PROTEIN_AA = {"ALA","ARG","ASN","ASP","CYS","GLN","GLU","GLY","HIS","ILE",
              "LEU","LYS","MET","PHE","PRO","SER","THR","TRP","TYR","VAL","MSE"}


def get_ligand_atoms(holo_pdb, lig_resname):
    """Extract ligand heavy atom coordinates from holo PDB."""
    coords = []
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
                if resname == lig_resname:
                    coords.append([x, y, z])
            except ValueError:
                continue
    if not coords and het_groups:
        best = max(het_groups, key=lambda r: len(het_groups[r]))
        coords = het_groups[best]
    return np.array(coords) if coords else None


def compute_dca(pocket_centroids, ligand_atoms):
    """Min distance from any pocket centroid to any ligand atom."""
    if len(pocket_centroids) == 0 or ligand_atoms is None or len(ligand_atoms) == 0:
        return float('inf')
    min_dist = float('inf')
    for cent in pocket_centroids:
        dists = np.linalg.norm(ligand_atoms - cent, axis=-1)
        min_dist = min(min_dist, float(np.min(dists)))
    return min_dist


def evaluate_method(method_name, targets, gt, get_pockets_fn):
    """Evaluate a pocket detection method on all targets."""
    results = []

    for target in targets:
        tid = str(target["id"])
        apo = target["apo_pdb"].lower()
        holo = target.get("holo_pdb", "").lower()
        lig_res = target.get("ligand_resname", "?")
        apo_pdb = os.path.join(APO_DIR, f"{apo}.pdb")

        if tid not in gt or not os.path.exists(apo_pdb):
            continue

        true_cent = np.array(gt[tid]["centroid"])
        pockets = get_pockets_fn(apo_pdb, tid)

        # Get ligand atoms for DCA
        ligand_atoms = None
        holo_pdb = os.path.join(HOLO_DIR, f"{holo}.pdb")
        if os.path.exists(holo_pdb):
            ligand_atoms = get_ligand_atoms(holo_pdb, lig_res)

        if not pockets:
            results.append({
                "tid": tid, "target": apo.upper(),
                "best_dcc": None, "best_rank": None,
                "r1_dcc": None, "dca": None, "n_pockets": 0,
            })
            continue

        dccs = []
        centroids = []
        for p in pockets:
            cent = np.array(p["centroid"])
            centroids.append(cent)
            dcc = float(np.linalg.norm(cent - true_cent))
            dccs.append(dcc)

        best_dcc = min(dccs)
        best_rank = dccs.index(best_dcc) + 1
        r1_dcc = dccs[0] if dccs else None
        dca = compute_dca(centroids, ligand_atoms)

        results.append({
            "tid": tid, "target": apo.upper(),
            "best_dcc": round(best_dcc, 2),
            "best_rank": best_rank,
            "r1_dcc": round(r1_dcc, 2) if r1_dcc else None,
            "dca": round(dca, 2) if dca < float('inf') else None,
            "n_pockets": len(pockets),
            "all_dccs": [round(d, 2) for d in dccs],
        })

    return results


def print_comparison(prism_results, p2rank_results, fpocket_results):
    """Print side-by-side comparison table."""
    # Index by tid
    prism_by_tid = {r["tid"]: r for r in prism_results if r.get("best_dcc") is not None}
    p2r_by_tid = {r["tid"]: r for r in p2rank_results if r.get("best_dcc") is not None}
    fp_by_tid = {r["tid"]: r for r in fpocket_results if r.get("best_dcc") is not None}

    all_tids = sorted(set(list(prism_by_tid.keys()) + list(p2r_by_tid.keys()) + list(fp_by_tid.keys())),
                       key=lambda x: int(x))

    print(f"\n{'='*95}")
    print(f"HEAD-TO-HEAD COMPARISON: PRISM4D vs P2Rank vs fpocket")
    print(f"{'='*95}")
    print(f"{'TID':<4} {'Target':<10} {'PRISM Best':<12} {'P2Rank Best':<12} {'fpocket Best':<12} {'Winner'}")
    print("-" * 70)

    prism_wins = 0
    p2rank_wins = 0
    fpocket_wins = 0

    for tid in all_tids:
        pr = prism_by_tid.get(tid, {})
        p2 = p2r_by_tid.get(tid, {})
        fp = fp_by_tid.get(tid, {})

        target = pr.get("target", p2.get("target", fp.get("target", "?")))

        pr_dcc = pr.get("best_dcc", 999)
        p2_dcc = p2.get("best_dcc", 999)
        fp_dcc = fp.get("best_dcc", 999)

        best = min(pr_dcc, p2_dcc, fp_dcc)
        if best == pr_dcc and pr_dcc < 999:
            winner = "PRISM4D"
            prism_wins += 1
        elif best == p2_dcc and p2_dcc < 999:
            winner = "P2Rank"
            p2rank_wins += 1
        elif best == fp_dcc and fp_dcc < 999:
            winner = "fpocket"
            fpocket_wins += 1
        else:
            winner = "-"

        pr_str = f"{pr_dcc:.2f}A" if pr_dcc < 999 else "—"
        p2_str = f"{p2_dcc:.2f}A" if p2_dcc < 999 else "—"
        fp_str = f"{fp_dcc:.2f}A" if fp_dcc < 999 else "—"

        print(f"{tid:<4} {target:<10} {pr_str:<12} {p2_str:<12} {fp_str:<12} {winner}")

    print(f"\n{'='*95}")
    print(f"WINS: PRISM4D={prism_wins}  P2Rank={p2rank_wins}  fpocket={fpocket_wins}")

    # Summary metrics
    methods = {
        "PRISM4D": [r for r in prism_results if r.get("best_dcc") is not None],
        "P2Rank": [r for r in p2rank_results if r.get("best_dcc") is not None],
        "fpocket": [r for r in fpocket_results if r.get("best_dcc") is not None],
    }

    print(f"\n{'Metric':<25} {'PRISM4D':<12} {'P2Rank':<12} {'fpocket':<12}")
    print("-" * 60)

    for name, results in methods.items():
        pass  # computed below

    for metric_name, threshold, use_rank in [
        ("Best DCC <2A", 2, False), ("Best DCC <4A", 4, False),
        ("Best DCC <6A", 6, False), ("Best DCC <8A", 8, False),
        ("SR@1 (DCC<4A)", 4, True), ("SR@3 (DCC<4A)", 4, True),
        ("SR@5 (DCC<4A)", 4, True), ("SR@10 (DCC<4A)", 4, True),
    ]:
        vals = []
        for name in ["PRISM4D", "P2Rank", "fpocket"]:
            results = methods[name]
            n = len(results)
            if use_rank:
                topn = int(metric_name.split("@")[1].split(" ")[0])
                ct = sum(1 for r in results
                         if r.get("all_dccs") and any(d < threshold for d in r["all_dccs"][:topn]))
            else:
                ct = sum(1 for r in results if r["best_dcc"] < threshold)
            vals.append(f"{100*ct/max(n,1):.0f}% ({ct}/{n})")

        print(f"{metric_name:<25} {vals[0]:<12} {vals[1]:<12} {vals[2]:<12}")

    # Mean/Median DCC
    for stat_name, fn in [("Mean Best DCC", np.mean), ("Median Best DCC", np.median)]:
        vals = []
        for name in ["PRISM4D", "P2Rank", "fpocket"]:
            dccs = [r["best_dcc"] for r in methods[name]]
            vals.append(f"{fn(dccs):.2f}A" if dccs else "—")
        print(f"{stat_name:<25} {vals[0]:<12} {vals[1]:<12} {vals[2]:<12}")

    # DCA metrics
    print()
    for metric_name, threshold in [("DCA <1A", 1), ("DCA <2A", 2), ("DCA <4A", 4), ("DCA <6A", 6)]:
        vals = []
        for name in ["PRISM4D", "P2Rank", "fpocket"]:
            dcas = [r["dca"] for r in methods[name] if r.get("dca") is not None]
            ct = sum(1 for d in dcas if d < threshold)
            n = len(dcas)
            vals.append(f"{100*ct/max(n,1):.0f}% ({ct}/{n})")
        print(f"{metric_name:<25} {vals[0]:<12} {vals[1]:<12} {vals[2]:<12}")

    for stat_name, fn in [("Mean DCA", np.mean), ("Median DCA", np.median)]:
        vals = []
        for name in ["PRISM4D", "P2Rank", "fpocket"]:
            dcas = [r["dca"] for r in methods[name] if r.get("dca") is not None]
            vals.append(f"{fn(dcas):.2f}A" if dcas else "—")
        print(f"{stat_name:<25} {vals[0]:<12} {vals[1]:<12} {vals[2]:<12}")


def main():
    with open(MANIFEST) as f:
        manifest = json.load(f)
    with open(GT_PATH) as f:
        gt = json.load(f)

    targets = manifest["targets"]

    # Get PRISM results from existing detection
    print("Loading PRISM4D results...")
    prism_results = []
    for target in targets:
        tid = str(target["id"])
        apo = target["apo_pdb"].lower()
        if tid not in gt:
            continue

        result_dir = f"{BENCH_DIR}/results/{tid}"
        bs_files = [f for f in os.listdir(result_dir) if f.endswith(".binding_sites.json")] if os.path.exists(result_dir) else []

        if not bs_files:
            prism_results.append({"tid": tid, "target": apo.upper(), "best_dcc": None, "n_pockets": 0})
            continue

        with open(os.path.join(result_dir, bs_files[0])) as f:
            data = json.load(f)

        true_cent = np.array(gt[tid]["centroid"])
        sites = data.get("sites", [])
        centroids = [np.array(s["centroid"]) for s in sites if s.get("centroid")]
        dccs = [float(np.linalg.norm(c - true_cent)) for c in centroids]

        # DCA for PRISM
        holo = target.get("holo_pdb", "").lower()
        lig_res = target.get("ligand_resname", "?")
        holo_pdb = os.path.join(HOLO_DIR, f"{holo}.pdb")
        ligand_atoms = get_ligand_atoms(holo_pdb, lig_res) if os.path.exists(holo_pdb) else None
        dca = compute_dca(centroids, ligand_atoms) if centroids else float('inf')

        if dccs:
            best_dcc = min(dccs)
            prism_results.append({
                "tid": tid, "target": apo.upper(),
                "best_dcc": round(best_dcc, 2),
                "best_rank": dccs.index(best_dcc) + 1,
                "r1_dcc": round(dccs[0], 2),
                "dca": round(dca, 2) if dca < float('inf') else None,
                "n_pockets": len(sites),
                "all_dccs": [round(d, 2) for d in dccs],
            })
        else:
            prism_results.append({"tid": tid, "target": apo.upper(), "best_dcc": None, "dca": None, "n_pockets": 0})

    n_prism = sum(1 for r in prism_results if r.get("best_dcc") is not None)
    print(f"  PRISM4D: {n_prism} targets with results")

    # Run P2Rank
    print("\nRunning P2Rank on all targets...")
    os.makedirs(P2RANK_OUT, exist_ok=True)

    def get_p2rank_pockets(apo_pdb, tid):
        out_dir = os.path.join(P2RANK_OUT, Path(apo_pdb).stem)
        return run_p2rank(apo_pdb, out_dir)

    p2rank_results = evaluate_method("P2Rank", targets, gt, get_p2rank_pockets)
    n_p2r = sum(1 for r in p2rank_results if r.get("best_dcc") is not None)
    print(f"  P2Rank: {n_p2r} targets with results")

    # Run fpocket
    print("\nRunning fpocket on all targets...")
    os.makedirs(FPOCKET_OUT, exist_ok=True)

    def get_fpocket_pockets(apo_pdb, tid):
        out_dir = os.path.join(FPOCKET_OUT, Path(apo_pdb).stem)
        return run_fpocket(apo_pdb, out_dir)

    fpocket_results = evaluate_method("fpocket", targets, gt, get_fpocket_pockets)
    n_fp = sum(1 for r in fpocket_results if r.get("best_dcc") is not None)
    print(f"  fpocket: {n_fp} targets with results")

    # Print comparison
    print_comparison(prism_results, p2rank_results, fpocket_results)

    # Save results
    output = {
        "prism4d": prism_results,
        "p2rank": p2rank_results,
        "fpocket": fpocket_results,
    }
    with open(f"{BENCH_DIR}/baseline_comparison.json", "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved: {BENCH_DIR}/baseline_comparison.json")


if __name__ == "__main__":
    main()
