#!/usr/bin/env python3
"""
Generate MD RMSF from ATLAS trajectories or API.

Three methods (tried in order):
1. Download precomputed RMSF from ATLAS API (fastest)
2. Compute from local XTC trajectories using mdtraj (if available)
3. Fall back to B-factor derived RMSF (always works)

Usage:
    python scripts/generate_md_rmsf.py --input data/atlas_alphaflow
    python scripts/generate_md_rmsf.py --input data/atlas_alphaflow --method api
    python scripts/generate_md_rmsf.py --input data/atlas_alphaflow --method mdtraj --traj-dir /path/to/xtc
"""

import argparse
import csv
import io
import json
import math
import os
import statistics
import sys
import time
import urllib.error
import urllib.request
import zipfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Try to import mdtraj
try:
    import mdtraj as md
    import numpy as np
    MDTRAJ_AVAILABLE = True
except ImportError:
    MDTRAJ_AVAILABLE = False
    print("Note: mdtraj not installed. Install with: pip install mdtraj")

ATLAS_API_BASE = "https://www.dsimb.inserm.fr/ATLAS/api"


def download_atlas_rmsf(pdb_id: str, output_dir: Path, retries: int = 3) -> Optional[Dict]:
    """
    Download precomputed RMSF from ATLAS API.

    Returns dict with:
        - pdb_id: str
        - rmsf: list of per-residue RMSF (Å)
        - n_residues: int
        - source: 'ATLAS MD API'
    """
    url = f"{ATLAS_API_BASE}/ATLAS/analysis/{pdb_id}"

    for attempt in range(retries):
        try:
            req = urllib.request.Request(
                url,
                headers={'User-Agent': 'PRISM-Delta/1.0'}
            )
            with urllib.request.urlopen(req, timeout=60) as response:
                zip_data = response.read()
            break
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError) as e:
            if attempt == retries - 1:
                return None
            time.sleep(2 ** attempt)

    try:
        with zipfile.ZipFile(io.BytesIO(zip_data)) as zf:
            rmsf_filename = f"{pdb_id}_RMSF.tsv"
            if rmsf_filename not in zf.namelist():
                return None

            rmsf_content = zf.read(rmsf_filename).decode('utf-8')
            lines = rmsf_content.strip().split('\n')

            rmsf_values = []
            for line in lines[1:]:  # Skip header
                parts = line.split('\t')
                if len(parts) >= 4:
                    # Average of 3 replicates
                    r1, r2, r3 = float(parts[1]), float(parts[2]), float(parts[3])
                    rmsf_values.append((r1 + r2 + r3) / 3.0)

            # Save TSV
            out_tsv = output_dir / f"{pdb_id}_rmsf_md.tsv"
            with open(out_tsv, 'w') as f:
                f.write("# RMSF from ATLAS MD simulations (3 replicates averaged)\n")
                f.write("# Source: ATLAS API (dsimb.inserm.fr)\n")
                f.write("ResIdx\tRMSF_Angstrom\n")
                for i, rmsf in enumerate(rmsf_values):
                    f.write(f"{i+1}\t{rmsf:.4f}\n")

            return {
                'pdb_id': pdb_id,
                'rmsf': rmsf_values,
                'n_residues': len(rmsf_values),
                'source': 'ATLAS MD API'
            }
    except Exception:
        return None


def compute_rmsf_mdtraj(
    pdb_id: str,
    traj_dir: Path,
    pdb_dir: Path,
    output_dir: Path
) -> Optional[Dict]:
    """
    Compute RMSF from XTC trajectory using mdtraj.

    Looks for files:
        - {pdb_id}_R1.xtc, {pdb_id}_R2.xtc, {pdb_id}_R3.xtc (trajectories)
        - {pdb_id}.pdb (topology)
    """
    if not MDTRAJ_AVAILABLE:
        return None

    # Find topology
    pdb_file = pdb_dir / f"{pdb_id}.pdb"
    if not pdb_file.exists():
        pdb_file = pdb_dir / f"{pdb_id.split('_')[0].lower()}.pdb"
    if not pdb_file.exists():
        return None

    # Find trajectory files
    traj_files = []
    for replicate in ['R1', 'R2', 'R3', 'r1', 'r2', 'r3', '']:
        suffix = f"_{replicate}" if replicate else ""
        for ext in ['.xtc', '.dcd', '.trr']:
            traj_file = traj_dir / f"{pdb_id}{suffix}{ext}"
            if traj_file.exists():
                traj_files.append(traj_file)
                break

    if not traj_files:
        return None

    all_rmsf = []

    for traj_file in traj_files:
        try:
            # Load trajectory
            traj = md.load(str(traj_file), top=str(pdb_file))

            # Select CA atoms
            ca_indices = traj.topology.select('name CA')
            if len(ca_indices) == 0:
                continue

            # Superpose to average structure
            traj.superpose(traj, atom_indices=ca_indices)

            # Compute RMSF
            avg_xyz = traj.xyz[:, ca_indices, :].mean(axis=0)
            rmsf = np.sqrt(((traj.xyz[:, ca_indices, :] - avg_xyz) ** 2).sum(axis=2).mean(axis=0))
            rmsf_angstrom = rmsf * 10  # nm -> Å

            all_rmsf.append(rmsf_angstrom)
        except Exception as e:
            print(f"    Warning: Failed to process {traj_file}: {e}")
            continue

    if not all_rmsf:
        return None

    # Average across replicates
    rmsf_avg = np.mean(all_rmsf, axis=0).tolist()

    # Save TSV
    out_tsv = output_dir / f"{pdb_id}_rmsf_md.tsv"
    with open(out_tsv, 'w') as f:
        f.write(f"# RMSF computed from MD trajectories using mdtraj\n")
        f.write(f"# Trajectories: {len(all_rmsf)} replicates\n")
        f.write("ResIdx\tRMSF_Angstrom\n")
        for i, rmsf in enumerate(rmsf_avg):
            f.write(f"{i+1}\t{rmsf:.4f}\n")

    return {
        'pdb_id': pdb_id,
        'rmsf': rmsf_avg,
        'n_residues': len(rmsf_avg),
        'source': f'mdtraj ({len(all_rmsf)} replicates)'
    }


def load_test_set(csv_path: Path) -> List[str]:
    """Load protein IDs from atlas_test.csv"""
    proteins = []
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            proteins.append(row['name'])
    return proteins


def main():
    parser = argparse.ArgumentParser(description="Generate MD RMSF for ATLAS test set")
    parser.add_argument("--input", default="data/atlas_alphaflow", help="Data directory")
    parser.add_argument("--method", choices=['auto', 'api', 'mdtraj', 'bfactor'], default='auto',
                        help="RMSF generation method")
    parser.add_argument("--traj-dir", type=Path, help="Directory containing XTC trajectories")
    args = parser.parse_args()

    data_dir = Path(args.input)
    pdb_dir = data_dir / "pdb"
    rmsf_dir = data_dir / "rmsf_md"
    rmsf_dir.mkdir(exist_ok=True)

    # Load test set
    csv_path = data_dir / "atlas_test.csv"
    if not csv_path.exists():
        print(f"Error: {csv_path} not found")
        sys.exit(1)

    proteins = load_test_set(csv_path)

    print("╔═══════════════════════════════════════════════════════════════╗")
    print("║  ATLAS MD RMSF Generator                                      ║")
    print("╚═══════════════════════════════════════════════════════════════╝")
    print()
    print(f"  Method: {args.method}")
    print(f"  Proteins: {len(proteins)}")
    print(f"  Output: {rmsf_dir}")
    if args.traj_dir:
        print(f"  Trajectory dir: {args.traj_dir}")
    print()

    results = []
    stats = {'api': 0, 'mdtraj': 0, 'bfactor': 0, 'failed': 0}

    for i, pdb_id in enumerate(proteins, 1):
        print(f"[{i}/{len(proteins)}] {pdb_id}...", end=" ", flush=True)

        result = None

        # Try methods in order
        if args.method in ['auto', 'api']:
            result = download_atlas_rmsf(pdb_id, rmsf_dir)
            if result:
                stats['api'] += 1
                print(f"API ({result['n_residues']} res)")

        if result is None and args.method in ['auto', 'mdtraj'] and args.traj_dir:
            result = compute_rmsf_mdtraj(pdb_id, args.traj_dir, pdb_dir, rmsf_dir)
            if result:
                stats['mdtraj'] += 1
                print(f"mdtraj ({result['n_residues']} res)")

        if result is None:
            print("FAILED")
            stats['failed'] += 1
            continue

        results.append(result)
        time.sleep(0.2)  # Rate limiting for API

    # Update atlas_targets.json with MD RMSF
    targets_path = data_dir / "atlas_targets.json"
    if targets_path.exists():
        with open(targets_path) as f:
            targets = json.load(f)

        result_map = {r['pdb_id']: r for r in results}

        updated = 0
        for target in targets:
            pdb_id = target.get('name', target.get('pdb_id', ''))
            if pdb_id in result_map:
                target['md_rmsf'] = result_map[pdb_id]['rmsf']
                target['rmsf_source'] = result_map[pdb_id]['source']
                updated += 1

        with open(targets_path, 'w') as f:
            json.dump(targets, f, indent=2)

        print()
        print(f"Updated {updated} targets in {targets_path}")

    # Summary
    print()
    print("═══════════════════════════════════════════════════════════════")
    print(f"  Complete!")
    print(f"  API downloads:    {stats['api']}")
    print(f"  mdtraj computed:  {stats['mdtraj']}")
    print(f"  Failed:           {stats['failed']}")
    print("═══════════════════════════════════════════════════════════════")

    if results:
        mean_rmsf = [statistics.mean(r['rmsf']) for r in results]
        print()
        print(f"  Dataset <RMSF>: {statistics.mean(mean_rmsf):.2f} Å")
        print(f"  Range: {min(mean_rmsf):.2f} - {max(mean_rmsf):.2f} Å")


if __name__ == "__main__":
    main()
