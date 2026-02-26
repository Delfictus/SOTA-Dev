#!/usr/bin/env python3
"""
CryptoBench Topology Preparation for PRISM4D Benchmarking

Processes the CryptoBench dataset (1,107 apo structures):
1. Parses dataset.json for apo PDB IDs
2. Obtains PDB files (CIF→PDB conversion or RCSB download)
3. Calls prism-prep for each structure (the official PRISM4D prep tool)
4. Extracts ground truth binding residue sets from dataset.json
5. Writes a manifest for batched nhs_rt_full runs

Usage:
    python 02_prepare_topologies.py [--test-only] [--max-atoms 15000] [--limit N]

prism-prep handles all its own dependencies internally (OpenMM, PDBFixer,
sanitization, AMBER ff14SB topology, aromatic detection, validation).
"""

import json
import sys
import os
import argparse
import time
import subprocess
import urllib.request
from pathlib import Path

# =============================================================================
# PATHS
# =============================================================================

BENCH_DIR = Path(__file__).parent.resolve()
DATA_DIR = BENCH_DIR / "data"
PRISM_ROOT = BENCH_DIR.parent.parent
PRISM_PREP = PRISM_ROOT / "scripts" / "prism-prep"

# Output directories
STRUCTURES_DIR = BENCH_DIR / "structures"       # PDB files for prism-prep input
TOPOLOGIES_DIR = BENCH_DIR / "topologies"       # prism-prep output
GROUND_TRUTH_DIR = BENCH_DIR / "ground_truth"   # Per-structure ground truth
RESULTS_DIR = BENCH_DIR / "results"             # nhs_rt_full output (created later)

MAX_ATOMS = 15000


# =============================================================================
# PDB ACQUISITION (CIF→PDB or RCSB download)
# =============================================================================

def get_pdb_file(apo_id: str) -> tuple:
    """
    Obtain a PDB file for the given apo ID.
    Tries CIF conversion first, then RCSB download.
    Returns (pdb_path, source) or (None, error_msg).
    """
    pdb_path = STRUCTURES_DIR / f"{apo_id}.pdb"

    # Already have it
    if pdb_path.exists() and pdb_path.stat().st_size > 100:
        return pdb_path, "cached"

    # Try CIF conversion
    cif_path = DATA_DIR / "cif_files" / f"{apo_id}.cif"
    if not cif_path.exists():
        # Check subdirectories (OSF may nest them)
        candidates = list((DATA_DIR / "cif_files").rglob(f"{apo_id}.cif"))
        if candidates:
            cif_path = candidates[0]

    if cif_path.exists():
        try:
            from openmm.app import PDBxFile, PDBFile
            cif = PDBxFile(str(cif_path))
            n_atoms = sum(1 for _ in cif.topology.atoms())
            if n_atoms > MAX_ATOMS:
                return None, f"too_large_{n_atoms}_atoms"
            with open(pdb_path, 'w') as f:
                PDBFile.writeFile(cif.topology, cif.positions, f)
            return pdb_path, "cif"
        except Exception as e:
            pass  # Fall through to RCSB

    # RCSB download fallback
    url = f"https://files.rcsb.org/download/{apo_id.upper()}.pdb"
    try:
        urllib.request.urlretrieve(url, str(pdb_path))
        if pdb_path.exists() and pdb_path.stat().st_size > 100:
            return pdb_path, "rcsb"
        else:
            return None, "rcsb_empty"
    except Exception as e:
        return None, f"rcsb_failed_{str(e)[:50]}"


# =============================================================================
# GROUND TRUTH EXTRACTION
# =============================================================================

def extract_ground_truth(apo_id: str, dataset: dict) -> dict:
    """
    Extract ground truth for an apo structure from CryptoBench dataset.json.

    Returns dict with binding residue labels, holo structures, pRMSD values.
    """
    entries = dataset.get(apo_id, [])
    if not entries:
        return None

    all_binding_residues = set()
    holo_entries = []
    main_holo = None

    for entry in entries:
        for res_str in entry.get("apo_pocket_selection", []):
            all_binding_residues.add(res_str)

        holo_info = {
            "holo_pdb_id": entry.get("holo_pdb_id", ""),
            "holo_chain": entry.get("holo_chain", ""),
            "apo_chain": entry.get("apo_chain", ""),
            "ligand": entry.get("ligand", ""),
            "ligand_index": entry.get("ligand_index", ""),
            "ligand_chain": entry.get("ligand_chain", ""),
            "pRMSD": entry.get("pRMSD", 0.0),
            "is_main": entry.get("is_main_holo_structure", False),
            "apo_pocket_selection": entry.get("apo_pocket_selection", []),
            "holo_pocket_selection": entry.get("holo_pocket_selection", []),
        }
        holo_entries.append(holo_info)
        if holo_info["is_main"]:
            main_holo = holo_info

    binding_residues = []
    for res_str in sorted(all_binding_residues):
        parts = res_str.split("_")
        if len(parts) == 2:
            binding_residues.append({
                "chain": parts[0],
                "resid": int(parts[1]) if parts[1].lstrip('-').isdigit() else parts[1],
                "label": res_str,
            })

    return {
        "apo_id": apo_id,
        "binding_residues": binding_residues,
        "binding_residue_labels": sorted(all_binding_residues),
        "n_binding_residues": len(binding_residues),
        "n_holo_structures": len(holo_entries),
        "main_holo": main_holo,
        "holo_entries": holo_entries,
        "max_pRMSD": max(e["pRMSD"] for e in holo_entries) if holo_entries else 0,
    }


# =============================================================================
# PER-STRUCTURE PREPARATION
# =============================================================================

def prepare_single(apo_id: str, dataset: dict) -> dict:
    """
    Prepare one structure:
      1. Get PDB file (CIF conversion or RCSB)
      2. Run prism-prep <pdb> <topology.json>
      3. Write ground truth JSON
    """
    result = {
        "apo_id": apo_id,
        "success": False,
        "n_atoms": 0,
        "error": None,
        "source": None,
        "elapsed": 0,
    }

    t0 = time.time()
    topo_path = TOPOLOGIES_DIR / f"{apo_id}.topology.json"
    gt_path = GROUND_TRUTH_DIR / f"{apo_id}.ground_truth.json"

    try:
        # Skip if already fully prepared
        if topo_path.exists() and gt_path.exists():
            result["success"] = True
            result["error"] = "already_prepared"
            result["elapsed"] = time.time() - t0
            return result

        # Step 1: Get PDB file
        pdb_path, source = get_pdb_file(apo_id)
        result["source"] = source
        if pdb_path is None:
            result["error"] = source  # source contains the error message
            result["elapsed"] = time.time() - t0
            return result

        # Quick atom count sanity check
        n_atoms_approx = sum(1 for line in open(pdb_path)
                            if line.startswith("ATOM") or line.startswith("HETATM"))
        if n_atoms_approx > MAX_ATOMS:
            result["error"] = f"too_large_{n_atoms_approx}_atoms"
            result["n_atoms"] = n_atoms_approx
            result["elapsed"] = time.time() - t0
            return result

        # Step 2: Run prism-prep (the official PRISM4D preprocessing tool)
        if not topo_path.exists():
            proc = subprocess.run(
                [
                    str(PRISM_PREP),
                    str(pdb_path.resolve()),
                    str(topo_path.resolve()),
                    "-q",
                ],
                capture_output=True,
                text=True,
                timeout=600,
            )

            if proc.returncode != 0 or not topo_path.exists():
                err = proc.stderr.strip()[-200:] if proc.stderr else ""
                out = proc.stdout.strip()[-200:] if proc.stdout else ""
                result["error"] = f"prism_prep_failed: {err or out or 'unknown'}"
                result["elapsed"] = time.time() - t0
                return result

        # Read topology for stats
        try:
            with open(topo_path) as f:
                topo = json.load(f)
            result["n_atoms"] = topo.get("n_atoms", 0)
        except Exception:
            pass

        # Step 3: Extract and write ground truth
        if not gt_path.exists():
            gt = extract_ground_truth(apo_id, dataset)
            if gt:
                with open(gt_path, 'w') as f:
                    json.dump(gt, f, indent=2)

        result["success"] = True

    except Exception as e:
        result["error"] = str(e)[:200]

    result["elapsed"] = time.time() - t0
    return result


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Prepare CryptoBench structures for PRISM4D benchmarking"
    )
    parser.add_argument("--test-only", action="store_true",
                       help="Only prepare the 222 test-set structures")
    parser.add_argument("--max-atoms", type=int, default=15000,
                       help="Skip structures with more atoms (default: 15000)")
    parser.add_argument("--limit", type=int, default=0,
                       help="Limit number of structures to process (0 = all)")
    args = parser.parse_args()

    global MAX_ATOMS
    MAX_ATOMS = args.max_atoms

    # Verify prism-prep exists
    if not PRISM_PREP.exists():
        print(f"ERROR: prism-prep not found at {PRISM_PREP}")
        sys.exit(1)

    # Quick prism-prep dependency check
    print("Checking prism-prep dependencies...")
    dep_check = subprocess.run(
        [str(PRISM_PREP), "--check-deps"],
        capture_output=True, text=True, timeout=30,
    )
    if dep_check.returncode != 0:
        print("ERROR: prism-prep dependency check failed:")
        print(dep_check.stdout)
        print(dep_check.stderr)
        print("\nFix dependencies before continuing.")
        sys.exit(1)
    # Show the check output
    for line in dep_check.stdout.strip().split('\n'):
        if line.strip():
            print(f"  {line.strip()}")
    print()

    # Load dataset
    dataset_path = DATA_DIR / "dataset.json"
    folds_path = DATA_DIR / "folds.json"

    if not dataset_path.exists():
        print(f"ERROR: {dataset_path} not found. Run 01_download_cryptobench.sh first.")
        sys.exit(1)

    print("Loading CryptoBench dataset...")
    with open(dataset_path) as f:
        dataset = json.load(f)

    # Get PDB IDs to process
    if args.test_only and folds_path.exists():
        with open(folds_path) as f:
            folds = json.load(f)
        apo_ids = sorted(folds.get("test", []))
        print(f"Test set: {len(apo_ids)} structures")
    else:
        apo_ids = sorted(dataset.keys())
        print(f"Full dataset: {len(apo_ids)} structures")

    if args.limit > 0:
        apo_ids = apo_ids[:args.limit]
        print(f"Limited to: {len(apo_ids)} structures")

    # Create output directories
    for d in [STRUCTURES_DIR, TOPOLOGIES_DIR, GROUND_TRUTH_DIR, RESULTS_DIR]:
        d.mkdir(parents=True, exist_ok=True)

    print(f"\nOutput:")
    print(f"  Structures: {STRUCTURES_DIR}")
    print(f"  Topologies: {TOPOLOGIES_DIR}")
    print(f"  Ground truth: {GROUND_TRUTH_DIR}")
    print()

    # Process structures sequentially
    start_time = time.time()
    results = []
    n_success = 0
    n_skip = 0
    n_fail = 0

    for i, apo_id in enumerate(apo_ids, 1):
        result = prepare_single(apo_id, dataset)
        results.append(result)

        if result["success"]:
            if result["error"] == "already_prepared":
                n_skip += 1
                status = "SKIP"
            else:
                n_success += 1
                status = "OK"
        else:
            n_fail += 1
            status = "FAIL"

        atoms_str = f"{result['n_atoms']:,}" if result['n_atoms'] else "?"
        elapsed_str = f"{result['elapsed']:.1f}s" if result['elapsed'] > 0.1 else ""

        if status == "FAIL":
            print(f"  [{i:4d}/{len(apo_ids)}] {status} {apo_id}: {result['error']}")
        elif status == "OK":
            print(f"  [{i:4d}/{len(apo_ids)}] {status} {apo_id}: {atoms_str} atoms {elapsed_str}")
        # SKIP = silent

        # Progress every 50
        if i % 50 == 0:
            elapsed = time.time() - start_time
            rate = i / elapsed if elapsed > 0 else 0
            eta = (len(apo_ids) - i) / rate if rate > 0 else 0
            print(f"  --- Progress: {i}/{len(apo_ids)} | "
                  f"OK:{n_success} Skip:{n_skip} Fail:{n_fail} | "
                  f"Rate:{rate:.1f}/s | ETA:{eta/60:.0f}min ---")

    total_time = time.time() - start_time

    # Summary
    print()
    print("=" * 60)
    print("  CRYPTOBENCH PREPARATION SUMMARY")
    print("=" * 60)
    print(f"  Total structures: {len(apo_ids)}")
    print(f"  Successfully prepared: {n_success}")
    print(f"  Already prepared: {n_skip}")
    print(f"  Failed: {n_fail}")
    print(f"  Total time: {total_time/60:.1f} minutes")
    print()

    # Write manifest for nhs_rt_full runs
    ready_ids = []
    for r in results:
        if r["success"]:
            topo = TOPOLOGIES_DIR / f"{r['apo_id']}.topology.json"
            if topo.exists():
                ready_ids.append(r["apo_id"])

    manifest_path = BENCH_DIR / "run_manifest.txt"
    with open(manifest_path, 'w') as f:
        for apo_id in sorted(ready_ids):
            f.write(f"{apo_id}\n")

    print(f"  Run manifest: {manifest_path} ({len(ready_ids)} structures)")

    # Write detailed results
    results_log = BENCH_DIR / "preparation_log.json"
    with open(results_log, 'w') as f:
        json.dump({
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "total": len(apo_ids),
            "success": n_success,
            "skipped": n_skip,
            "failed": n_fail,
            "ready_count": len(ready_ids),
            "elapsed_minutes": total_time / 60,
            "failures": [r for r in results if not r["success"]],
        }, f, indent=2)

    print(f"  Prep log: {results_log}")

    # List failures
    if n_fail > 0:
        print(f"\n  Failed structures ({n_fail}):")
        fail_reasons = {}
        for r in results:
            if not r["success"]:
                reason = (r["error"] or "unknown").split(":")[0]
                fail_reasons.setdefault(reason, []).append(r["apo_id"])
        for reason, ids in sorted(fail_reasons.items(), key=lambda x: -len(x[1])):
            print(f"    {reason}: {len(ids)} structures")
            if len(ids) <= 5:
                print(f"      {', '.join(ids)}")

    print()
    print("Next step: python3 03_generate_run_commands.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
