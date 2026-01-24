#!/usr/bin/env python3
"""
Generate Reference RMSF from PDB B-factors

B-factors (temperature factors) capture atomic displacement and are related
to RMSF through the Debye-Waller equation:

    B = (8 * π² / 3) * RMSF²

Therefore:
    RMSF = sqrt(3 * B / (8 * π²))

This provides a crystallographic reference for dynamics validation.
While not identical to MD-derived RMSF, B-factor RMSF is:
- Experimentally derived (X-ray crystallography)
- Widely used as a dynamics proxy
- Available for all PDB structures

Reference: Radivojac et al. (2004) Protein flexibility and intrinsic disorder

Usage:
    python scripts/generate_rmsf_from_bfactors.py --input data/atlas_alphaflow
"""

import json
import math
import os
import sys
from pathlib import Path
from typing import List, Tuple, Optional
import statistics

# Physical constant for B-factor to RMSF conversion
# B = (8 * π² / 3) * RMSF²  →  RMSF = sqrt(3 * B / (8 * π²))
B_TO_RMSF_FACTOR = math.sqrt(3.0 / (8.0 * math.pi * math.pi))


def parse_pdb_bfactors(pdb_path: Path, chain: str = None) -> Tuple[List[float], List[Tuple[float, float, float]], str]:
    """
    Extract Cα B-factors and coordinates from PDB file.

    Returns:
        Tuple of (b_factors, ca_coords, chain_id)
    """
    b_factors = []
    ca_coords = []
    detected_chain = None

    with open(pdb_path, 'r') as f:
        for line in f:
            if not line.startswith('ATOM'):
                continue

            atom_name = line[12:16].strip()
            if atom_name != 'CA':
                continue

            line_chain = line[21].strip()

            # If chain specified, filter by it
            if chain and line_chain != chain:
                continue

            # Track the chain we're using
            if detected_chain is None:
                detected_chain = line_chain or 'A'
            elif line_chain != detected_chain and chain is None:
                # If no chain specified and we hit a new chain, stop
                # (use first chain only)
                continue

            try:
                x = float(line[30:38])
                y = float(line[38:46])
                z = float(line[46:54])
                b_factor = float(line[60:66])

                ca_coords.append((x, y, z))
                b_factors.append(b_factor)
            except (ValueError, IndexError):
                continue

    return b_factors, ca_coords, detected_chain or 'A'


def bfactor_to_rmsf(b_factors: List[float]) -> List[float]:
    """
    Convert B-factors to RMSF using Debye-Waller equation.

    RMSF = sqrt(3 * B / (8 * π²))

    Note: Negative B-factors (rare, indicates issues) are set to minimum positive.
    """
    # Handle negative or zero B-factors
    min_positive = min((b for b in b_factors if b > 0), default=1.0)

    rmsf_values = []
    for b in b_factors:
        if b <= 0:
            b = min_positive  # Use minimum positive value
        rmsf = B_TO_RMSF_FACTOR * math.sqrt(b)
        rmsf_values.append(rmsf)

    return rmsf_values


def normalize_rmsf(rmsf_values: List[float], target_mean: float = 1.5, target_std: float = 0.8) -> List[float]:
    """
    Optionally normalize RMSF to match expected MD distribution.

    MD simulations typically show RMSF in range 0.5-4.0 Å with mean ~1.5 Å.
    B-factor derived RMSF may have different scale.

    This normalization is OPTIONAL and should be noted in publications.
    """
    if len(rmsf_values) < 2:
        return rmsf_values

    current_mean = statistics.mean(rmsf_values)
    current_std = statistics.stdev(rmsf_values)

    if current_std < 0.01:
        return rmsf_values

    # Z-score normalization then rescale
    normalized = []
    for r in rmsf_values:
        z = (r - current_mean) / current_std
        new_r = z * target_std + target_mean
        # Clamp to reasonable range
        new_r = max(0.3, min(6.0, new_r))
        normalized.append(new_r)

    return normalized


def process_atlas_dataset(data_dir: Path, normalize: bool = False) -> None:
    """
    Process all ATLAS proteins and generate RMSF from B-factors.
    """
    pdb_dir = data_dir / "pdb"
    rmsf_dir = data_dir / "rmsf_bfactor"
    rmsf_dir.mkdir(exist_ok=True)

    # Load existing targets or CSV
    targets_path = data_dir / "atlas_targets.json"
    csv_path = data_dir / "atlas_test.csv"

    if targets_path.exists():
        with open(targets_path) as f:
            targets = json.load(f)
    else:
        targets = []

    # Build name -> target mapping
    target_map = {t.get('name', t.get('pdb_id', '')): t for t in targets}

    # Process each protein from CSV
    proteins = []
    if csv_path.exists():
        with open(csv_path) as f:
            header = f.readline()  # Skip header
            for line in f:
                if line.strip():
                    parts = line.strip().split(',')
                    if len(parts) >= 1:
                        proteins.append(parts[0])  # name column
    else:
        # Use existing targets
        proteins = [t.get('name', t.get('pdb_id', '')) for t in targets]

    print("╔═══════════════════════════════════════════════════════════════╗")
    print("║  Generating RMSF from B-factors (Debye-Waller)                ║")
    print("╚═══════════════════════════════════════════════════════════════╝")
    print()
    print(f"  Formula: RMSF = sqrt(3 * B / (8 * π²))")
    print(f"  Proteins: {len(proteins)}")
    print(f"  Normalize: {normalize}")
    print()

    updated_targets = []
    stats = {'success': 0, 'failed': 0, 'total_residues': 0}

    for name in proteins:
        # Parse name to get PDB ID and chain
        if '_' in name:
            pdb_id = name.split('_')[0]
            chain = name.split('_')[1]
        else:
            pdb_id = name
            chain = 'A'

        # Find PDB file
        pdb_path = pdb_dir / f"{name}.pdb"
        if not pdb_path.exists():
            pdb_path = pdb_dir / f"{pdb_id.lower()}.pdb"
        if not pdb_path.exists():
            pdb_path = pdb_dir / f"{pdb_id.upper()}.pdb"

        if not pdb_path.exists():
            print(f"  ✗ {name}: PDB not found")
            stats['failed'] += 1
            continue

        # Extract B-factors
        b_factors, ca_coords, detected_chain = parse_pdb_bfactors(pdb_path, chain)

        if not b_factors:
            print(f"  ✗ {name}: No Cα atoms found")
            stats['failed'] += 1
            continue

        # Convert to RMSF
        rmsf_values = bfactor_to_rmsf(b_factors)

        if normalize:
            rmsf_values = normalize_rmsf(rmsf_values)

        # Statistics
        mean_rmsf = statistics.mean(rmsf_values)
        std_rmsf = statistics.stdev(rmsf_values) if len(rmsf_values) > 1 else 0
        mean_b = statistics.mean(b_factors)

        # Save RMSF file
        rmsf_file = rmsf_dir / f"{name}_rmsf.tsv"
        with open(rmsf_file, 'w') as f:
            f.write("# RMSF derived from B-factors using Debye-Waller equation\n")
            f.write("# RMSF = sqrt(3 * B / (8 * π²))\n")
            f.write(f"# Source: {pdb_path.name}\n")
            f.write(f"# Chain: {detected_chain}\n")
            f.write(f"# Normalized: {normalize}\n")
            f.write("ResIdx\tRMSF\tB_factor\n")
            for i, (rmsf, b) in enumerate(zip(rmsf_values, b_factors)):
                f.write(f"{i+1}\t{rmsf:.4f}\t{b:.2f}\n")

        # Update target
        target = target_map.get(name, {
            'pdb_id': pdb_id.upper(),
            'chain': chain,
            'name': name,
        })
        target['n_residues'] = len(rmsf_values)
        target['md_rmsf'] = rmsf_values
        target['reference_coords'] = [[c[0], c[1], c[2]] for c in ca_coords]
        target['rmsf_source'] = 'B-factor (Debye-Waller)'
        target['mean_bfactor'] = mean_b
        target['mean_rmsf'] = mean_rmsf

        updated_targets.append(target)
        stats['success'] += 1
        stats['total_residues'] += len(rmsf_values)

        print(f"  ✓ {name}: {len(rmsf_values)} res, <B>={mean_b:.1f}, <RMSF>={mean_rmsf:.2f}±{std_rmsf:.2f} Å")

    # Save updated targets
    with open(targets_path, 'w') as f:
        json.dump(updated_targets, f, indent=2)

    print()
    print("═══════════════════════════════════════════════════════════════")
    print(f"  Complete!")
    print(f"  Processed: {stats['success']}/{len(proteins)}")
    print(f"  Total residues: {stats['total_residues']}")
    print(f"  RMSF files: {rmsf_dir}")
    print(f"  Targets JSON: {targets_path}")
    print("═══════════════════════════════════════════════════════════════")

    # Summary statistics
    if updated_targets:
        all_mean_rmsf = [t['mean_rmsf'] for t in updated_targets if 'mean_rmsf' in t]
        print()
        print(f"  Dataset <RMSF>: {statistics.mean(all_mean_rmsf):.2f} Å")
        print(f"  Range: {min(all_mean_rmsf):.2f} - {max(all_mean_rmsf):.2f} Å")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Generate RMSF from PDB B-factors")
    parser.add_argument("--input", default="data/atlas_alphaflow", help="Data directory")
    parser.add_argument("--normalize", action="store_true", help="Normalize to MD-like distribution")
    args = parser.parse_args()

    data_dir = Path(args.input)
    if not data_dir.exists():
        print(f"Error: Directory not found: {data_dir}")
        sys.exit(1)

    process_atlas_dataset(data_dir, normalize=args.normalize)


if __name__ == "__main__":
    main()
