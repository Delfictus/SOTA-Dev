#!/usr/bin/env python3
"""
Print binding site atomic coordinates from PRISM-Delta output PDB.

Usage:
    python print_binding_sites.py 6vxx_cryptic_sites.pdb
    python print_binding_sites.py 6vxx_cryptic_sites.pdb --all-atoms
    python print_binding_sites.py 6vxx_cryptic_sites.pdb --site 1
"""

import argparse
import sys
from typing import List, Tuple, Dict

# Define binding sites (same as PyMOL script)
BINDING_SITES = {
    1: {
        "name": "RBD Cryptic Pocket",
        "residues": [396, 397, 511],
        "description": "High priority - escape resistant",
        "color": "hotpink"
    },
    2: {
        "name": "NTD Druggable Pocket",
        "residues": list(range(92, 109)) + list(range(191, 206)),
        "description": "High druggability score (0.91)",
        "color": "orange"
    },
    3: {
        "name": "Fusion Peptide Region",
        "residues": list(range(816, 824)) + list(range(878, 901)),
        "description": "Known cryptic site from ground truth",
        "color": "yellow"
    },
}

ESCAPE_SITES = {
    "name": "Escape Mutation Sites",
    "residues": [417, 452, 484, 501, 614, 681],
    "description": "Known escape mutations from CoV-RDB"
}

GROUND_TRUTH = {
    "name": "Ground Truth Cryptic",
    "residues": list(range(373, 380)) + list(range(503, 510)) + list(range(816, 824)),
    "description": "Literature-validated cryptic residues"
}


def parse_pdb(pdb_path: str, chain: str = "A") -> List[Dict]:
    """Parse PDB file and extract atom information."""
    atoms = []

    with open(pdb_path, 'r') as f:
        for line in f:
            if not line.startswith("ATOM") and not line.startswith("HETATM"):
                continue

            atom_chain = line[21].strip()
            if atom_chain != chain:
                continue

            try:
                atom = {
                    "serial": int(line[6:11].strip()),
                    "name": line[12:16].strip(),
                    "resn": line[17:20].strip(),
                    "chain": atom_chain,
                    "resi": int(line[22:26].strip()),
                    "x": float(line[30:38].strip()),
                    "y": float(line[38:46].strip()),
                    "z": float(line[46:54].strip()),
                    "b": float(line[60:66].strip()) if len(line) >= 66 else 0.0,
                }
                atoms.append(atom)
            except (ValueError, IndexError):
                continue

    return atoms


def filter_atoms(atoms: List[Dict], residues: List[int], ca_only: bool = True) -> List[Dict]:
    """Filter atoms by residue numbers."""
    filtered = [a for a in atoms if a["resi"] in residues]
    if ca_only:
        filtered = [a for a in filtered if a["name"] == "CA"]
    return filtered


def compute_center(atoms: List[Dict]) -> Tuple[float, float, float]:
    """Compute center of mass from atoms."""
    if not atoms:
        return (0.0, 0.0, 0.0)
    cx = sum(a["x"] for a in atoms) / len(atoms)
    cy = sum(a["y"] for a in atoms) / len(atoms)
    cz = sum(a["z"] for a in atoms) / len(atoms)
    return (cx, cy, cz)


def print_site(site_info: Dict, atoms: List[Dict], ca_only: bool = True):
    """Print a binding site's atomic information."""
    filtered = filter_atoms(atoms, site_info["residues"], ca_only)

    print(f"\n{'='*70}")
    print(f"  {site_info['name'].upper()}")
    print(f"  {site_info['description']}")
    print(f"  Residues: {site_info['residues']}")
    print(f"{'='*70}")

    if not filtered:
        print("  No atoms found for this selection.")
        return

    atom_type = "CA atoms" if ca_only else "All atoms"
    print(f"\n  {atom_type} ({len(filtered)} total):")
    print(f"  {'Residue':<12} {'Atom':<6} {'X':>10} {'Y':>10} {'Z':>10} {'B-factor':>10}")
    print(f"  {'-'*62}")

    for a in sorted(filtered, key=lambda x: (x["resi"], x["name"])):
        print(f"  {a['chain']}{a['resn']:<4}{a['resi']:<5} {a['name']:<6} "
              f"{a['x']:>10.3f} {a['y']:>10.3f} {a['z']:>10.3f} {a['b']:>10.2f}")

    center = compute_center(filtered)
    print(f"\n  Center of Mass: {center[0]:.3f}, {center[1]:.3f}, {center[2]:.3f}")

    # Docking grid suggestion
    if len(filtered) >= 3:
        # Estimate box size from coordinate range
        xs = [a["x"] for a in filtered]
        ys = [a["y"] for a in filtered]
        zs = [a["z"] for a in filtered]
        size_x = max(xs) - min(xs) + 10  # Add 10A padding
        size_y = max(ys) - min(ys) + 10
        size_z = max(zs) - min(zs) + 10

        print(f"\n  AutoDock Vina Grid:")
        print(f"    --center_x {center[0]:.2f} --center_y {center[1]:.2f} --center_z {center[2]:.2f}")
        print(f"    --size_x {size_x:.0f} --size_y {size_y:.0f} --size_z {size_z:.0f}")


def main():
    parser = argparse.ArgumentParser(
        description="Print binding site atomic coordinates from PRISM-Delta PDB"
    )
    parser.add_argument("pdb", help="Path to PDB file (e.g., 6vxx_cryptic_sites.pdb)")
    parser.add_argument("--chain", default="A", help="Chain ID (default: A)")
    parser.add_argument("--site", type=int, choices=[1, 2, 3],
                        help="Print only specific site (1, 2, or 3)")
    parser.add_argument("--all-atoms", action="store_true",
                        help="Print all atoms (default: CA only)")
    parser.add_argument("--escape", action="store_true",
                        help="Include escape mutation sites")
    parser.add_argument("--ground-truth", action="store_true",
                        help="Include ground truth cryptic residues")

    args = parser.parse_args()

    print("=" * 70)
    print("  PRISM-DELTA BINDING SITE COORDINATES")
    print(f"  PDB: {args.pdb}")
    print(f"  Chain: {args.chain}")
    print("=" * 70)

    atoms = parse_pdb(args.pdb, args.chain)
    print(f"\n  Loaded {len(atoms)} atoms from chain {args.chain}")

    ca_only = not args.all_atoms

    # Print requested sites
    if args.site:
        print_site(BINDING_SITES[args.site], atoms, ca_only)
    else:
        for site_num, site_info in BINDING_SITES.items():
            print_site(site_info, atoms, ca_only)

    if args.escape:
        print_site(ESCAPE_SITES, atoms, ca_only)

    if args.ground_truth:
        print_site(GROUND_TRUTH, atoms, ca_only)

    print("\n" + "=" * 70)
    print("  Done!")
    print("=" * 70)


if __name__ == "__main__":
    main()
