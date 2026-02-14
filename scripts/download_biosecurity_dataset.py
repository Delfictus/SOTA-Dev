#!/usr/bin/env python3
"""
PRISM4D Biosecurity Dataset Downloader

Downloads and prepares 50 priority structures for:
- Cryptic pocket detection
- Viral escape mutation prediction
- Real-time biosecurity surveillance

Usage:
    python scripts/download_biosecurity_dataset.py
"""

import json
import os
import sys
import urllib.request
import urllib.error
import gzip
import shutil
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

# Configuration
MANIFEST_PATH = "data/production/biosecurity_manifest.json"
OUTPUT_DIR = "data/production/structures"
TOPOLOGY_DIR = "data/production/topologies"
RCSB_URL = "https://files.rcsb.org/download/{pdb_id}.pdb.gz"
RCSB_CIF_URL = "https://files.rcsb.org/download/{pdb_id}.cif.gz"

def download_pdb(pdb_id: str, output_dir: Path) -> tuple[str, bool, str]:
    """Download a PDB file from RCSB."""
    pdb_path = output_dir / f"{pdb_id}.pdb"

    if pdb_path.exists():
        return pdb_id, True, "already exists"

    # Try PDB format first
    url = RCSB_URL.format(pdb_id=pdb_id)
    gz_path = output_dir / f"{pdb_id}.pdb.gz"

    try:
        urllib.request.urlretrieve(url, gz_path)
        with gzip.open(gz_path, 'rb') as f_in:
            with open(pdb_path, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)
        gz_path.unlink()
        return pdb_id, True, "downloaded"
    except urllib.error.HTTPError as e:
        # Try CIF format as fallback
        try:
            cif_url = RCSB_CIF_URL.format(pdb_id=pdb_id)
            cif_gz_path = output_dir / f"{pdb_id}.cif.gz"
            cif_path = output_dir / f"{pdb_id}.cif"
            urllib.request.urlretrieve(cif_url, cif_gz_path)
            with gzip.open(cif_gz_path, 'rb') as f_in:
                with open(cif_path, 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)
            cif_gz_path.unlink()
            return pdb_id, True, "downloaded (cif)"
        except Exception as e2:
            return pdb_id, False, str(e2)
    except Exception as e:
        return pdb_id, False, str(e)


def count_atoms_in_pdb(pdb_path: Path) -> int:
    """Count ATOM records in a PDB file."""
    count = 0
    try:
        with open(pdb_path, 'r') as f:
            for line in f:
                if line.startswith('ATOM') or line.startswith('HETATM'):
                    count += 1
    except:
        pass
    return count


def main():
    print("=" * 70)
    print("  PRISM4D BIOSECURITY DATASET DOWNLOADER")
    print("=" * 70)
    print()

    # Load manifest
    if not os.path.exists(MANIFEST_PATH):
        print(f"ERROR: Manifest not found: {MANIFEST_PATH}")
        sys.exit(1)

    with open(MANIFEST_PATH) as f:
        manifest = json.load(f)

    structures = manifest["structures"]
    print(f"Manifest loaded: {len(structures)} structures")
    print()

    # Print tier breakdown
    tiers = {}
    for s in structures:
        tier = s.get("tier", "unknown")
        tiers[tier] = tiers.get(tier, 0) + 1

    print("Dataset composition:")
    for tier, count in sorted(tiers.items()):
        desc = manifest["tiers"].get(tier, {}).get("description", "")
        print(f"  {tier}: {count} structures - {desc}")
    print()

    # Create output directories
    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    topology_dir = Path(TOPOLOGY_DIR)
    topology_dir.mkdir(parents=True, exist_ok=True)

    # Download structures
    print("Downloading structures from RCSB PDB...")
    print("-" * 70)

    pdb_ids = [s["pdb_id"] for s in structures]

    results = {"success": [], "failed": [], "skipped": []}

    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = {executor.submit(download_pdb, pdb_id, output_dir): pdb_id
                   for pdb_id in pdb_ids}

        for i, future in enumerate(as_completed(futures), 1):
            pdb_id, success, msg = future.result()
            status = "✓" if success else "✗"
            print(f"  [{i:2d}/{len(pdb_ids)}] {status} {pdb_id}: {msg}")

            if success:
                if "already" in msg:
                    results["skipped"].append(pdb_id)
                else:
                    results["success"].append(pdb_id)
            else:
                results["failed"].append(pdb_id)

    print()
    print("-" * 70)
    print(f"Download complete:")
    print(f"  Downloaded: {len(results['success'])}")
    print(f"  Skipped (existing): {len(results['skipped'])}")
    print(f"  Failed: {len(results['failed'])}")

    if results["failed"]:
        print(f"  Failed IDs: {', '.join(results['failed'])}")
    print()

    # Count atoms in downloaded structures
    print("Analyzing downloaded structures...")
    print("-" * 70)

    total_atoms = 0
    structure_info = []

    for s in structures:
        pdb_id = s["pdb_id"]
        pdb_path = output_dir / f"{pdb_id}.pdb"
        cif_path = output_dir / f"{pdb_id}.cif"

        if pdb_path.exists():
            atoms = count_atoms_in_pdb(pdb_path)
        elif cif_path.exists():
            atoms = -1  # CIF needs different parsing
        else:
            atoms = 0

        if atoms > 0:
            total_atoms += atoms
            structure_info.append({
                "pdb_id": pdb_id,
                "atoms": atoms,
                "tier": s["tier"],
                "pathogen": s["pathogen"],
                "name": s["name"]
            })

    # Sort by atom count
    structure_info.sort(key=lambda x: x["atoms"], reverse=True)

    print(f"\nStructure sizes (top 10 largest):")
    for info in structure_info[:10]:
        print(f"  {info['pdb_id']}: {info['atoms']:,} atoms - {info['name'][:40]}")

    print(f"\nTotal atoms across dataset: {total_atoms:,}")
    print()

    # Estimate processing time
    throughput = 500000  # atoms/sec from our benchmarks
    steps = manifest["sampling_config"]["steps_production"]
    snapshots = manifest["sampling_config"]["snapshots_per_structure"]

    total_atom_steps = total_atoms * steps
    est_hours = total_atom_steps / throughput / 3600

    print("=" * 70)
    print("  PROCESSING ESTIMATES")
    print("=" * 70)
    print(f"  Total structures: {len(structure_info)}")
    print(f"  Total atoms: {total_atoms:,}")
    print(f"  Steps per structure: {steps:,}")
    print(f"  Snapshots per structure: {snapshots}")
    print(f"  Total conformations: {len(structure_info) * snapshots:,}")
    print()
    print(f"  Estimated processing time: {est_hours:.1f} hours")
    print(f"  (at {throughput:,} atom-steps/sec)")
    print("=" * 70)
    print()

    # Save structure inventory
    inventory_path = output_dir / "inventory.json"
    with open(inventory_path, 'w') as f:
        json.dump({
            "download_date": time.strftime("%Y-%m-%d %H:%M:%S"),
            "total_structures": len(structure_info),
            "total_atoms": total_atoms,
            "structures": structure_info
        }, f, indent=2)

    print(f"Inventory saved to: {inventory_path}")
    print()
    print("Next steps:")
    print("  1. Convert to topology format:")
    print("     cargo run --release -p prism-io --bin pdb-to-topology -- \\")
    print(f"       --input-dir {OUTPUT_DIR} --output-dir {TOPOLOGY_DIR}")
    print()
    print("  2. Run SIMD batch ensemble generation:")
    print("     cargo run --release -p prism-validation --bin generate-ensemble-simd -- \\")
    print(f"       --topology-dir {TOPOLOGY_DIR} --steps 50000 --output-dir results/production")
    print()


if __name__ == "__main__":
    main()
