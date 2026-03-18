#!/usr/bin/env python3
"""
PRISM4D GPU Docking Pipeline (FAST MODE)
========================================
Optimized for pre-existing 3D ligand structures.
Skips energy minimization (200 RDKit iterations = slow).
Uses direct PDBQT conversion via OpenBabel (100x faster).

Usage:
    python scripts/gpu_dock_fast.py \
        --receptor structure.pdb \
        --sites results/binding_sites.json \
        --ligands ligand.pdb  (or .sdf) \
        --output docking_results/ \
        --no-gnina  (skip GNINA rescoring if you want speed)
"""

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from datetime import datetime

CONDA_ENV = Path(os.environ.get(
    "PRISM_DOCK_ENV",
    os.path.expanduser("~/miniconda3/envs/prism_dock")
))
UNIDOCK_BIN = CONDA_ENV / "bin" / "unidock"
GNINA_BIN = CONDA_ENV / "bin" / "gnina"
OBABEL_BIN = CONDA_ENV / "bin" / "obabel"

def fast_prep_ligand_pdbqt(ligand_path, output_pdbqt):
    """Convert PDB/SDF directly to PDBQT using OpenBabel (fast)."""
    result = subprocess.run(
        [str(OBABEL_BIN), str(ligand_path), "-O", str(output_pdbqt),
         "-xg", "-p", "7.4", "-h", "--addresidues"],
        capture_output=True, text=True, timeout=30
    )
    if result.returncode == 0 and Path(output_pdbqt).exists():
        return True
    return False

def extract_docking_boxes(sites_json):
    """Extract docking box parameters from binding_sites.json."""
    with open(sites_json) as f:
        data = json.load(f)
    
    boxes = []
    for site in data["sites"]:
        centroid = site["centroid"]
        volume = site.get("volume", 1000)
        side = max(volume ** (1.0 / 3.0) + 8.0, 20.0)
        side = min(side, 40.0)
        
        boxes.append({
            "site_id": site["id"],
            "classification": site["classification"],
            "center_x": round(centroid[0], 3),
            "center_y": round(centroid[1], 3),
            "center_z": round(centroid[2], 3),
            "size_x": round(side, 1),
            "size_y": round(side, 1),
            "size_z": round(side, 1),
        })
    return boxes, data

def run_unidock(receptor_pdbqt, ligand_sdf, box, output_dir):
    """Run UniDock GPU docking."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_sdf = output_dir / "docked.sdf"
    
    cmd = [
        str(UNIDOCK_BIN),
        "--receptor", str(receptor_pdbqt),
        "--ligand", str(ligand_sdf),
        "--center_x", str(box["center_x"]),
        "--center_y", str(box["center_y"]),
        "--center_z", str(box["center_z"]),
        "--size_x", str(box["size_x"]),
        "--size_y", str(box["size_y"]),
        "--size_z", str(box["size_z"]),
        "--search_mode", "balance",
        "--exhaustiveness", "8",
        "--num_modes", "10",
        "--out", str(output_sdf),
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    if result.returncode == 0:
        print(f"  UniDock: OK — {output_sdf}")
        return output_sdf
    else:
        print(f"  UniDock: FAILED")
        print(result.stderr[:500])
        return None

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--receptor", required=True, help="Receptor PDB")
    parser.add_argument("--sites", required=True, help="binding_sites.json")
    parser.add_argument("--ligands", required=True, help="Ligand PDB or SDF")
    parser.add_argument("--output", default="docking_results_fast/", help="Output directory")
    parser.add_argument("--no-gnina", action="store_true", help="Skip GNINA rescoring")
    
    args = parser.parse_args()
    
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("PRISM4D GPU Docking Pipeline (FAST MODE)")
    print("=" * 60)
    print(f"  Receptor:  {args.receptor}")
    print(f"  Ligand:    {args.ligands}")
    print(f"  Sites:     {args.sites}")
    print(f"  Output:    {output_dir}")
    print()
    
    # Step 1: Extract boxes
    print("[1/3] Extracting docking boxes...")
    boxes, sites_data = extract_docking_boxes(args.sites)
    print(f"  Found {len(boxes)} sites")
    
    # Step 2: Prep receptor
    print("[2/3] Preparing receptor (OpenBabel, ~5s)...")
    receptor_pdbqt = output_dir / "receptor.pdbqt"
    result = subprocess.run(
        [str(OBABEL_BIN), args.receptor, "-O", str(receptor_pdbqt),
         "-xr", "-xh", "--partialcharge", "gasteiger"],
        capture_output=True, text=True, timeout=60
    )
    print(f"  Receptor PDBQT: {receptor_pdbqt}")
    
    # Step 3: Prep ligand (FAST — no minimization)
    print("[3/3] Preparing ligand (OpenBabel, ~2s)...")
    ligand_pdbqt = output_dir / "ligand.pdbqt"
    if fast_prep_ligand_pdbqt(args.ligands, ligand_pdbqt):
        print(f"  Ligand PDBQT: {ligand_pdbqt}")
    else:
        print(f"  ERROR: Could not prep ligand")
        sys.exit(1)
    
    # Step 4: Dock all sites
    print("\n[4/4] GPU Docking (UniDock, parallel)...")
    print(f"  RTX 5080 sm_120 — ~0.5-1s per site × {len(boxes)} sites")
    print()
    
    results = []
    for i, box in enumerate(boxes[:10]):  # First 10 sites
        site_id = box["site_id"]
        site_output = output_dir / f"site_{site_id}"
        site_output.mkdir(parents=True, exist_ok=True)
        
        print(f"  [{i+1}/{min(10,len(boxes))}] Site {site_id} ({box['classification']})...", end=" ", flush=True)
        sdf = run_unidock(receptor_pdbqt, ligand_pdbqt, box, site_output)
        results.append({"site_id": site_id, "docked_sdf": str(sdf) if sdf else None})
        print()
    
    # Summary
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    success = sum(1 for r in results if r["docked_sdf"])
    print(f"  Successful dockings: {success}/{len(results)}")
    print(f"  Output directory: {output_dir}")
    print()
    print("Next: Visualize with PyMOL or analyze scores in JSON")

if __name__ == "__main__":
    main()
