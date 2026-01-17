#!/usr/bin/env python3
"""
PRISM4D Production Topology Preparation

Batch processes downloaded PDB files:
1. Sanitize (fix missing atoms, add hydrogens)
2. Generate AMBER ff14SB topology JSON for GPU kernels

Usage:
    python scripts/prepare_production_topologies.py

Dependencies:
    conda activate prism-delta
    # Requires: openmm, pdbfixer
"""

import sys
import json
import os
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import time
import traceback

# Add script directory to path
SCRIPT_DIR = Path(__file__).parent.resolve()
sys.path.insert(0, str(SCRIPT_DIR))

try:
    from stage1_sanitize import sanitize_pdb
    from stage2_topology import prepare_topology
except ImportError as e:
    print(f"ERROR: Could not import stage modules: {e}")
    print("Make sure you're in the prism-delta conda environment")
    sys.exit(1)

# Configuration
INPUT_DIR = Path("data/production/structures")
SANITIZED_DIR = Path("data/production/sanitized")
TOPOLOGY_DIR = Path("data/production/topologies_json")
MAX_ATOMS = 15000  # Skip structures larger than this (too big for GPU)
MAX_WORKERS = 4    # Parallel processing


def process_structure(pdb_id: str) -> dict:
    """Process a single structure through sanitize + topology pipeline."""
    result = {
        "pdb_id": pdb_id,
        "success": False,
        "error": None,
        "n_atoms": 0,
        "n_bonds": 0,
        "n_angles": 0,
        "n_dihedrals": 0,
    }

    pdb_path = INPUT_DIR / f"{pdb_id}.pdb"
    cif_path = INPUT_DIR / f"{pdb_id}.cif"
    sanitized_path = SANITIZED_DIR / f"{pdb_id}_sanitized.pdb"
    topology_path = TOPOLOGY_DIR / f"{pdb_id}_topology.json"

    # Skip if already processed
    if topology_path.exists():
        try:
            with open(topology_path) as f:
                topo = json.load(f)
            result["success"] = True
            result["n_atoms"] = topo.get("n_atoms", 0)
            result["n_bonds"] = len(topo.get("bonds", []))
            result["n_angles"] = len(topo.get("angles", []))
            result["n_dihedrals"] = len(topo.get("dihedrals", []))
            result["error"] = "already exists"
            return result
        except:
            pass

    try:
        # Find input file
        if pdb_path.exists():
            input_file = pdb_path
        elif cif_path.exists():
            # CIF files need special handling - skip for now
            result["error"] = "CIF format (not yet supported)"
            return result
        else:
            result["error"] = "file not found"
            return result

        # Stage 1: Sanitize
        sanitize_result = sanitize_pdb(
            str(input_file),
            str(sanitized_path),
            chain=None,  # All chains
            verbose=False
        )

        if not sanitized_path.exists():
            result["error"] = f"sanitization failed: {sanitize_result}"
            return result

        # Quick atom count check
        n_atoms_approx = sum(1 for line in open(sanitized_path) if line.startswith("ATOM"))
        if n_atoms_approx > MAX_ATOMS:
            result["error"] = f"too large ({n_atoms_approx} atoms > {MAX_ATOMS})"
            return result

        # Stage 2: Topology
        topo_result = prepare_topology(
            str(sanitized_path),
            str(topology_path),
            solvate=False,
            minimize=True,
            ph=7.0,
            verbose=False
        )

        if topology_path.exists():
            with open(topology_path) as f:
                topo = json.load(f)
            result["success"] = True
            result["n_atoms"] = topo.get("n_atoms", 0)
            result["n_bonds"] = len(topo.get("bonds", []))
            result["n_angles"] = len(topo.get("angles", []))
            result["n_dihedrals"] = len(topo.get("dihedrals", []))
        else:
            result["error"] = "topology generation failed"

    except Exception as e:
        result["error"] = str(e)[:200]

    return result


def main():
    print("=" * 70)
    print("  PRISM4D PRODUCTION TOPOLOGY PREPARATION")
    print("=" * 70)
    print()

    # Create directories
    SANITIZED_DIR.mkdir(parents=True, exist_ok=True)
    TOPOLOGY_DIR.mkdir(parents=True, exist_ok=True)

    # Load inventory
    inventory_path = INPUT_DIR / "inventory.json"
    if not inventory_path.exists():
        print("ERROR: No inventory.json found. Run download_biosecurity_dataset.py first.")
        sys.exit(1)

    with open(inventory_path) as f:
        inventory = json.load(f)

    structures = inventory["structures"]
    pdb_ids = [s["pdb_id"] for s in structures]

    print(f"Processing {len(pdb_ids)} structures...")
    print(f"  Max atoms per structure: {MAX_ATOMS}")
    print(f"  Parallel workers: {MAX_WORKERS}")
    print()
    print("-" * 70)

    start_time = time.time()
    results = []

    # Process sequentially to avoid OpenMM memory issues
    # (parallel processing causes GPU context conflicts)
    for i, pdb_id in enumerate(pdb_ids, 1):
        result = process_structure(pdb_id)
        results.append(result)

        status = "OK" if result["success"] else "SKIP"
        if result["success"]:
            print(f"  [{i:2d}/{len(pdb_ids)}] {status} {pdb_id}: {result['n_atoms']} atoms, "
                  f"{result['n_bonds']} bonds")
        else:
            print(f"  [{i:2d}/{len(pdb_ids)}] {status} {pdb_id}: {result['error']}")

    elapsed = time.time() - start_time

    print()
    print("-" * 70)
    print(f"Completed in {elapsed:.1f}s")
    print()

    # Summary
    success = [r for r in results if r["success"]]
    failed = [r for r in results if not r["success"]]
    skipped = [r for r in failed if r["error"] and ("too large" in r["error"] or "already" in r["error"])]
    errors = [r for r in failed if r not in skipped]

    print(f"Summary:")
    print(f"  Successfully prepared: {len(success)}")
    print(f"  Skipped (too large): {len([r for r in failed if 'too large' in str(r['error'])])}")
    print(f"  Skipped (existing): {len([r for r in failed if 'already' in str(r['error'])])}")
    print(f"  Errors: {len(errors)}")

    if errors:
        print()
        print("Errors:")
        for r in errors[:10]:
            print(f"  {r['pdb_id']}: {r['error']}")

    # Calculate totals
    total_atoms = sum(r["n_atoms"] for r in success)
    total_bonds = sum(r["n_bonds"] for r in success)
    total_angles = sum(r["n_angles"] for r in success)
    total_dihedrals = sum(r["n_dihedrals"] for r in success)

    print()
    print("=" * 70)
    print("  READY FOR ENSEMBLE GENERATION")
    print("=" * 70)
    print(f"  Structures: {len(success)}")
    print(f"  Total atoms: {total_atoms:,}")
    print(f"  Total bonds: {total_bonds:,}")
    print(f"  Total angles: {total_angles:,}")
    print(f"  Total dihedrals: {total_dihedrals:,}")
    print()
    print(f"  Topologies: {TOPOLOGY_DIR}/")
    print()
    print("Run ensemble generation with:")
    print("  cargo run --release -p prism-validation --bin generate-ensemble-simd -- \\")
    print(f"    --topologies {TOPOLOGY_DIR}/*.json \\")
    print("    --steps 50000 --dt 1.0 --temperature 310.0 \\")
    print("    --output-dir results/production_ensembles")
    print()

    # Save results
    results_path = TOPOLOGY_DIR / "preparation_results.json"
    with open(results_path, "w") as f:
        json.dump({
            "prepared_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "total_structures": len(pdb_ids),
            "successful": len(success),
            "total_atoms": total_atoms,
            "results": results
        }, f, indent=2)

    print(f"Results saved to: {results_path}")


if __name__ == "__main__":
    main()
