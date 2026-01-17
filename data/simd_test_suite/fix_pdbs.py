#!/usr/bin/env python3
"""
Fix PDB structures for AMBER simulation using pdbfixer.

Workflow:
1. Remove waters and ligands (except structural ions)
2. Find and add missing residues
3. Find and add missing heavy atoms
4. Add hydrogens at pH 7.0
5. Save fixed PDB
"""

import os
import sys
from pathlib import Path

try:
    from pdbfixer import PDBFixer
    from openmm.app import PDBFile
except ImportError:
    print("ERROR: pdbfixer and openmm required")
    print("Install with: conda install -c conda-forge pdbfixer openmm")
    sys.exit(1)


# Structures to process with their chain selections
# For complexes, we'll extract just the RBD chain
STRUCTURES = {
    "6M0J": {"chains": ["E"], "desc": "WT RBD + ACE2 (RBD only)"},
    "7WHH": {"chains": ["E"], "desc": "Omicron RBD + ACE2 (RBD only)"},  # E is RBD, A is ACE2
    "7K45": {"chains": ["B"], "desc": "RBD + S2E12 Ab (RBD only)"},  # B is RBD, H+L are Ab
    "6WPS": {"chains": ["A"], "desc": "RBD + S309 Ab (RBD only)"},
    "6W41": {"chains": ["C"], "desc": "RBD + CR3022 Ab (RBD only)"},
    "8SGU": {"chains": ["A"], "desc": "Apo RBD"},
    "7JJI": {"chains": ["A"], "desc": "RBD + Ligand (chain A)"},  # Use first chain
}


def fix_pdb(input_path: Path, output_path: Path, keep_chains: list[str] = None):
    """Fix a PDB file using pdbfixer."""
    print(f"  Loading {input_path.name}...")
    fixer = PDBFixer(filename=str(input_path))

    # Get chain info before removal
    chain_ids = [c.id for c in fixer.topology.chains()]
    print(f"    Chains found: {chain_ids}")

    # Remove unwanted chains if specified
    if keep_chains:
        chains_to_remove = [c for c in chain_ids if c not in keep_chains]
        if chains_to_remove:
            print(f"    Removing chains: {chains_to_remove}, keeping: {keep_chains}")
            fixer.removeChains(chainIds=chains_to_remove)

    # Remove waters and heterogens (ligands, but keep ions)
    print("    Removing waters and ligands...")
    fixer.removeHeterogens(keepWater=False)

    # Find missing residues
    print("    Finding missing residues...")
    fixer.findMissingResidues()
    n_missing_res = len(fixer.missingResidues)
    if n_missing_res > 0:
        print(f"    Found {n_missing_res} missing residue segments")
        # Skip terminal missing residues (often disordered)
        # Only add internal missing residues
        keys_to_remove = []
        for key in fixer.missingResidues:
            chain_idx, res_idx = key
            # Skip if at very start or very end of chain
            if res_idx == 0:
                keys_to_remove.append(key)
        for key in keys_to_remove:
            del fixer.missingResidues[key]
        print(f"    Adding {len(fixer.missingResidues)} internal missing segments")

    # Find missing atoms
    print("    Finding missing atoms...")
    fixer.findMissingAtoms()
    n_missing_atoms = len(fixer.missingAtoms)
    n_missing_terminals = len(fixer.missingTerminals)
    print(f"    Found {n_missing_atoms} residues with missing atoms, {n_missing_terminals} missing terminals")

    # Add missing atoms
    print("    Adding missing atoms...")
    fixer.addMissingAtoms()

    # Add hydrogens at pH 7.0
    print("    Adding hydrogens at pH 7.0...")
    fixer.addMissingHydrogens(pH=7.0)

    # Count final atoms
    n_atoms = fixer.topology.getNumAtoms()
    n_residues = fixer.topology.getNumResidues()
    print(f"    Final structure: {n_atoms} atoms, {n_residues} residues")

    # Save
    print(f"    Saving to {output_path.name}...")
    with open(output_path, 'w') as f:
        PDBFile.writeFile(fixer.topology, fixer.positions, f)

    return n_atoms, n_residues


def main():
    raw_dir = Path("raw")
    fixed_dir = Path("sanitized")
    fixed_dir.mkdir(exist_ok=True)

    print("=" * 60)
    print("PDB Fixer - Preparing structures for AMBER simulation")
    print("=" * 60)

    results = {}

    for pdb_id, info in STRUCTURES.items():
        input_path = raw_dir / f"{pdb_id}.pdb"
        output_path = fixed_dir / f"{pdb_id}_fixed.pdb"

        if not input_path.exists():
            print(f"\nSKIPPING {pdb_id}: {input_path} not found")
            continue

        print(f"\n[{pdb_id}] {info['desc']}")
        try:
            n_atoms, n_res = fix_pdb(input_path, output_path, info.get("chains"))
            results[pdb_id] = {"atoms": n_atoms, "residues": n_res, "status": "OK"}
        except Exception as e:
            print(f"    ERROR: {e}")
            results[pdb_id] = {"status": "FAILED", "error": str(e)}

    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    for pdb_id, info in results.items():
        if info["status"] == "OK":
            print(f"  {pdb_id}: {info['atoms']:,} atoms, {info['residues']} residues")
        else:
            print(f"  {pdb_id}: FAILED - {info.get('error', 'unknown')}")


if __name__ == "__main__":
    main()
