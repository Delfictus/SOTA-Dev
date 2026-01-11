#!/usr/bin/env python3
"""
Print PRISM-Delta predictions with atomic coordinates in clean terminal format.

Usage:
    python3 print_predictions.py 2vwd_nipah_test.json
    python3 print_predictions.py 2vwd_nipah_test.json --pdb ../data/raw/2VWD.pdb
    python3 print_predictions.py 2vwd_nipah_test.json --site 1
    python3 print_predictions.py 2vwd_nipah_test.json --top 5
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple


def load_json(path: str) -> Dict[str, Any]:
    """Load JSON predictions."""
    with open(path, 'r') as f:
        return json.load(f)


def parse_pdb(pdb_path: str, chain: str = None) -> Dict[int, Dict]:
    """Parse PDB file and extract CA atom coordinates."""
    atoms = {}

    with open(pdb_path, 'r') as f:
        for line in f:
            if not line.startswith("ATOM"):
                continue

            atom_name = line[12:16].strip()
            if atom_name != "CA":
                continue

            chain_id = line[21].strip()
            if chain and chain_id != chain:
                continue

            try:
                res_num = int(line[22:26].strip())
                res_name = line[17:20].strip()
                x = float(line[30:38].strip())
                y = float(line[38:46].strip())
                z = float(line[46:54].strip())
                b_factor = float(line[60:66].strip()) if len(line) >= 66 else 0.0

                atoms[res_num] = {
                    'resn': res_name,
                    'chain': chain_id,
                    'x': x,
                    'y': y,
                    'z': z,
                    'b': b_factor
                }
            except (ValueError, IndexError):
                continue

    return atoms


def aa_3to1(three_letter: str) -> str:
    """Convert 3-letter amino acid code to 1-letter."""
    mapping = {
        'ALA': 'A', 'CYS': 'C', 'ASP': 'D', 'GLU': 'E', 'PHE': 'F',
        'GLY': 'G', 'HIS': 'H', 'ILE': 'I', 'LYS': 'K', 'LEU': 'L',
        'MET': 'M', 'ASN': 'N', 'PRO': 'P', 'GLN': 'Q', 'ARG': 'R',
        'SER': 'S', 'THR': 'T', 'VAL': 'V', 'TRP': 'W', 'TYR': 'Y',
    }
    return mapping.get(three_letter.upper(), 'X')


def print_header(predictions: Dict[str, Any]):
    """Print report header."""
    pdb_id = predictions.get('pdb_id', 'UNKNOWN')
    timestamp = predictions.get('timestamp', '')[:19]  # Truncate timestamp
    config = predictions.get('config', {})
    summary = predictions.get('summary', {})

    print()
    print("=" * 80)
    print(f"  PRISM-DELTA CRYPTIC SITE PREDICTIONS")
    print("=" * 80)
    print()
    print(f"  PDB ID:      {pdb_id}")
    print(f"  Timestamp:   {timestamp}")
    print()
    print("  Configuration:")
    print(f"    Temperature:           {config.get('hmc_temperature', 'N/A')} K")
    print(f"    Ensemble conformations: {config.get('n_ensemble_conformations', 'N/A')}")
    print(f"    GPU enabled:           {config.get('use_gpu', 'N/A')}")
    print()
    print("  Summary:")
    print(f"    Total residues:        {summary.get('total_residues', 'N/A')}")
    print(f"    Cryptic residues:      {summary.get('n_cryptic', 'N/A')}")
    print(f"    Predicted sites:       {summary.get('n_sites', 'N/A')}")
    print(f"    Threshold used:        {summary.get('threshold_used', 'N/A'):.4f}")
    print(f"    Mean cryptic score:    {summary.get('mean_cryptic_score', 0):.4f}")
    print(f"    Max cryptic score:     {summary.get('max_cryptic_score', 0):.4f}")
    print(f"    Mean escape resistance: {summary.get('mean_escape_resistance', 0):.4f}")
    print()


def print_site(site: Dict, site_num: int, pdb_atoms: Optional[Dict] = None):
    """Print a single binding site with atomic coordinates."""
    residues = site.get('residues', [])
    mean_score = site.get('mean_cryptic_score', 0)
    mean_escape = site.get('mean_escape_resistance', 0)
    druggability = site.get('druggability_score', 0)
    center = site.get('center', [0, 0, 0])
    radius = site.get('radius', 0)

    print()
    print("=" * 80)
    print(f"  SITE {site_num}: {len(residues)} residues")
    print("=" * 80)
    print()
    print(f"  Mean Cryptic Score:      {mean_score:.4f}")
    print(f"  Mean Escape Resistance:  {mean_escape:.4f}")
    print(f"  Druggability Score:      {druggability:.4f}")
    print(f"  Center of Mass:          ({center[0]:.2f}, {center[1]:.2f}, {center[2]:.2f})")
    print(f"  Radius:                  {radius:.2f} Å")
    print()

    # Table header
    print("  " + "-" * 76)
    print(f"  {'Res#':>6}  {'AA':>3}  {'Chain':>5}  {'X':>10}  {'Y':>10}  {'Z':>10}  {'Cryptic':>8}  {'EscapeR':>8}")
    print("  " + "-" * 76)

    # Sort residues by residue number
    sorted_residues = sorted(residues, key=lambda r: r['residue_num'])

    for res in sorted_residues:
        res_num = res['residue_num']
        aa = res['amino_acid']
        chain = res['chain_id']
        cscore = res['cryptic_score']
        escore = res['escape_resistance']

        # Get coordinates from PDB if available
        if pdb_atoms and res_num in pdb_atoms:
            atom = pdb_atoms[res_num]
            x, y, z = atom['x'], atom['y'], atom['z']
            print(f"  {res_num:>6}  {aa:>3}  {chain:>5}  {x:>10.3f}  {y:>10.3f}  {z:>10.3f}  {cscore:>8.4f}  {escore:>8.4f}")
        else:
            # Use center as placeholder
            print(f"  {res_num:>6}  {aa:>3}  {chain:>5}  {'N/A':>10}  {'N/A':>10}  {'N/A':>10}  {cscore:>8.4f}  {escore:>8.4f}")

    print("  " + "-" * 76)

    # Docking grid suggestion
    size = int(radius * 2 + 10)
    size = max(15, min(size, 40))

    print()
    print("  AutoDock Vina Grid:")
    print(f"    --center_x {center[0]:.2f} --center_y {center[1]:.2f} --center_z {center[2]:.2f}")
    print(f"    --size_x {size} --size_y {size} --size_z {size}")


def print_top_residues(predictions: Dict[str, Any], pdb_atoms: Optional[Dict], n: int = 20):
    """Print top N cryptic residues."""
    residue_preds = predictions.get('residue_predictions', [])

    # Sort by cryptic score
    sorted_preds = sorted(residue_preds, key=lambda r: r['cryptic_score'], reverse=True)[:n]

    print()
    print("=" * 80)
    print(f"  TOP {n} CRYPTIC RESIDUES")
    print("=" * 80)
    print()
    print("  " + "-" * 76)
    print(f"  {'Rank':>4}  {'Res#':>6}  {'AA':>3}  {'X':>10}  {'Y':>10}  {'Z':>10}  {'Cryptic':>8}  {'EscapeR':>8}")
    print("  " + "-" * 76)

    for rank, res in enumerate(sorted_preds, 1):
        res_num = res['residue_num']
        aa = res['amino_acid']
        cscore = res['cryptic_score']
        escore = res['escape_resistance']

        if pdb_atoms and res_num in pdb_atoms:
            atom = pdb_atoms[res_num]
            x, y, z = atom['x'], atom['y'], atom['z']
            print(f"  {rank:>4}  {res_num:>6}  {aa:>3}  {x:>10.3f}  {y:>10.3f}  {z:>10.3f}  {cscore:>8.4f}  {escore:>8.4f}")
        else:
            print(f"  {rank:>4}  {res_num:>6}  {aa:>3}  {'N/A':>10}  {'N/A':>10}  {'N/A':>10}  {cscore:>8.4f}  {escore:>8.4f}")

    print("  " + "-" * 76)


def print_escape_resistant(predictions: Dict[str, Any], pdb_atoms: Optional[Dict], threshold: float = 0.7):
    """Print high escape resistance residues."""
    residue_preds = predictions.get('residue_predictions', [])

    # Filter by escape resistance
    high_escape = [r for r in residue_preds if r['escape_resistance'] >= threshold]
    high_escape = sorted(high_escape, key=lambda r: r['escape_resistance'], reverse=True)

    if not high_escape:
        return

    print()
    print("=" * 80)
    print(f"  HIGH ESCAPE RESISTANCE RESIDUES (≥{threshold:.2f})")
    print("=" * 80)
    print()
    print("  " + "-" * 76)
    print(f"  {'Res#':>6}  {'AA':>3}  {'Chain':>5}  {'X':>10}  {'Y':>10}  {'Z':>10}  {'Cryptic':>8}  {'EscapeR':>8}")
    print("  " + "-" * 76)

    for res in high_escape:
        res_num = res['residue_num']
        aa = res['amino_acid']
        chain = res['chain_id']
        cscore = res['cryptic_score']
        escore = res['escape_resistance']

        if pdb_atoms and res_num in pdb_atoms:
            atom = pdb_atoms[res_num]
            x, y, z = atom['x'], atom['y'], atom['z']
            print(f"  {res_num:>6}  {aa:>3}  {chain:>5}  {x:>10.3f}  {y:>10.3f}  {z:>10.3f}  {cscore:>8.4f}  {escore:>8.4f}")
        else:
            print(f"  {res_num:>6}  {aa:>3}  {chain:>5}  {'N/A':>10}  {'N/A':>10}  {'N/A':>10}  {cscore:>8.4f}  {escore:>8.4f}")

    print("  " + "-" * 76)
    print(f"  Total: {len(high_escape)} residues")


def print_docking_grids(sites: List[Dict], top_n: int = 5):
    """Print docking grid parameters for top sites."""
    print()
    print("=" * 80)
    print("  DOCKING GRID PARAMETERS (AutoDock Vina)")
    print("=" * 80)

    for i, site in enumerate(sites[:top_n], 1):
        center = site.get('center', [0, 0, 0])
        radius = site.get('radius', 10)
        size = int(radius * 2 + 10)
        size = max(15, min(size, 40))

        print()
        print(f"  Site {i}:")
        print(f"    vina --receptor protein.pdbqt --ligand ligand.pdbqt \\")
        print(f"         --center_x {center[0]:.2f} --center_y {center[1]:.2f} --center_z {center[2]:.2f} \\")
        print(f"         --size_x {size} --size_y {size} --size_z {size}")


def print_legend():
    """Print legend for the output."""
    print()
    print("=" * 80)
    print("  LEGEND")
    print("=" * 80)
    print()
    print("  Columns:")
    print("    Res#     - Residue number from PDB")
    print("    AA       - Amino acid one-letter code")
    print("    X, Y, Z  - CA atom coordinates (Ångströms)")
    print("    Cryptic  - Cryptic site score (0-1, higher = more likely cryptic)")
    print("    EscapeR  - Escape resistance (0-1, higher = harder to escape)")
    print()
    print("  Scores:")
    print("    Cryptic > 0.6  : Strong cryptic site candidate")
    print("    Cryptic > 0.8  : Very strong cryptic site candidate")
    print("    EscapeR > 0.7  : High escape resistance (good drug target)")
    print()


def main():
    parser = argparse.ArgumentParser(
        description="Print PRISM-Delta predictions with atomic coordinates"
    )
    parser.add_argument("json_file", help="Path to JSON predictions file")
    parser.add_argument("--pdb", "-p", help="Path to PDB structure file")
    parser.add_argument("--chain", "-c", help="Chain ID to use (default: from JSON)")
    parser.add_argument("--site", "-s", type=int, help="Print only specific site number")
    parser.add_argument("--top", "-t", type=int, default=5, help="Number of top sites to show (default: 5)")
    parser.add_argument("--all-sites", "-a", action="store_true", help="Show all sites")
    parser.add_argument("--top-residues", "-r", type=int, help="Show top N residues by score")
    parser.add_argument("--escape", "-e", action="store_true", help="Show escape resistant residues")
    parser.add_argument("--no-legend", action="store_true", help="Omit legend")

    args = parser.parse_args()

    # Load predictions
    json_path = Path(args.json_file)
    if not json_path.exists():
        print(f"Error: JSON file not found: {json_path}", file=sys.stderr)
        sys.exit(1)

    predictions = load_json(json_path)
    sites = predictions.get('predicted_sites', [])

    # Get chain from predictions if not specified
    residue_preds = predictions.get('residue_predictions', [])
    chain = args.chain
    if not chain and residue_preds:
        chain = residue_preds[0].get('chain_id', 'A')

    # Load PDB if provided
    pdb_atoms = None
    if args.pdb:
        pdb_path = Path(args.pdb)
        if pdb_path.exists():
            pdb_atoms = parse_pdb(pdb_path, chain)
            if not pdb_atoms:
                print(f"Warning: No CA atoms found in {pdb_path} for chain {chain}", file=sys.stderr)
        else:
            print(f"Warning: PDB file not found: {pdb_path}", file=sys.stderr)

    # Print header
    print_header(predictions)

    # Print sites
    if args.site:
        # Specific site
        if 1 <= args.site <= len(sites):
            print_site(sites[args.site - 1], args.site, pdb_atoms)
        else:
            print(f"Error: Site {args.site} not found (have {len(sites)} sites)", file=sys.stderr)
            sys.exit(1)
    elif args.all_sites:
        # All sites
        for i, site in enumerate(sites, 1):
            print_site(site, i, pdb_atoms)
    else:
        # Top N sites
        for i, site in enumerate(sites[:args.top], 1):
            print_site(site, i, pdb_atoms)

    # Print top residues if requested
    if args.top_residues:
        print_top_residues(predictions, pdb_atoms, args.top_residues)

    # Print escape resistant residues if requested
    if args.escape:
        print_escape_resistant(predictions, pdb_atoms)

    # Print docking grids
    print_docking_grids(sites, args.top)

    # Print legend
    if not args.no_legend:
        print_legend()

    print()
    print("=" * 80)
    print("  Done!")
    print("=" * 80)
    print()


if __name__ == "__main__":
    main()
