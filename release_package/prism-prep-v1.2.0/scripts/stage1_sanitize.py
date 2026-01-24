#!/usr/bin/env python3
"""
PRISM4D Stage 1: PDB Fetch & Sanitize

This script properly prepares PDB structures for GPU-accelerated MD:
1. Fetches from RCSB or loads local file
2. Removes waters, ligands, ions
3. Replaces non-standard residues (MSE → MET, etc.)
4. Finds and adds missing heavy atoms
5. Adds terminal caps (OXT, etc.)
6. Optionally selects specific chain(s)
7. Outputs clean PDB ready for Stage 2

Usage:
    python stage1_sanitize.py 6M0J output.pdb                    # Fetch from RCSB
    python stage1_sanitize.py 6M0J output.pdb --chain E          # Single chain
    python stage1_sanitize.py input.pdb output.pdb               # Local file
    python stage1_sanitize.py input.pdb output.pdb --keep-waters # Keep waters

Dependencies:
    conda install -c conda-forge pdbfixer openmm
"""

import sys
import os
import argparse
import tempfile
import urllib.request
from pathlib import Path

try:
    from pdbfixer import PDBFixer
    from openmm import app
except ImportError:
    print("ERROR: PDBFixer/OpenMM not found.")
    print("Install with: conda install -c conda-forge pdbfixer openmm")
    sys.exit(1)


# Non-standard residue replacements
RESIDUE_REPLACEMENTS = {
    'MSE': 'MET',  # Selenomethionine → Methionine
    'SEC': 'CYS',  # Selenocysteine → Cysteine
    'PYL': 'LYS',  # Pyrrolysine → Lysine
    'HYP': 'PRO',  # Hydroxyproline → Proline
    'SEP': 'SER',  # Phosphoserine → Serine
    'TPO': 'THR',  # Phosphothreonine → Threonine
    'PTR': 'TYR',  # Phosphotyrosine → Tyrosine
    'CSO': 'CYS',  # S-hydroxycysteine → Cysteine
    'CSS': 'CYS',  # Disulfide cysteine → Cysteine
    'OCS': 'CYS',  # Cysteinesulfonic acid → Cysteine
    'MLY': 'LYS',  # N-dimethyl-lysine → Lysine
    'M3L': 'LYS',  # N-trimethyllysine → Lysine
    'HIC': 'HIS',  # Methylhistidine → Histidine
    'NEP': 'HIS',  # N1-phosphonohistidine → Histidine
}

# Standard amino acids
STANDARD_AA = {
    'ALA', 'ARG', 'ASN', 'ASP', 'CYS', 'GLN', 'GLU', 'GLY', 'HIS', 'ILE',
    'LEU', 'LYS', 'MET', 'PHE', 'PRO', 'SER', 'THR', 'TRP', 'TYR', 'VAL'
}


def fetch_pdb(pdb_id: str) -> str:
    """Fetch PDB from RCSB and return content as string."""
    url = f"https://files.rcsb.org/download/{pdb_id.upper()}.pdb"
    print(f"Fetching {pdb_id} from RCSB...")

    try:
        with urllib.request.urlopen(url, timeout=30) as response:
            content = response.read().decode('utf-8')
        print(f"  Downloaded {len(content)} bytes")
        return content
    except Exception as e:
        raise RuntimeError(f"Failed to fetch {pdb_id}: {e}")


def is_pdb_id(s: str) -> bool:
    """Check if string looks like a PDB ID (4 characters, alphanumeric)."""
    return len(s) == 4 and s.isalnum()


def sanitize_pdb(
    input_path_or_id: str,
    output_path: str,
    chain: str = None,
    keep_waters: bool = False,
    keep_heterogens: bool = False,
    verbose: bool = True
) -> dict:
    """
    Sanitize a PDB structure for GPU MD.

    Returns dict with sanitization statistics.
    """
    stats = {
        'input': input_path_or_id,
        'output': output_path,
        'chain_filter': chain,
        'waters_removed': 0,
        'heterogens_removed': 0,
        'residues_replaced': {},
        'missing_atoms_added': 0,
        'missing_terminals_added': 0,
        'final_residues': 0,
        'final_atoms': 0,
        'final_chains': [],
    }

    # Step 1: Load structure
    if is_pdb_id(input_path_or_id):
        # Fetch from RCSB
        pdb_content = fetch_pdb(input_path_or_id)
        # Write to temp file for PDBFixer
        with tempfile.NamedTemporaryFile(mode='w', suffix='.pdb', delete=False) as f:
            f.write(pdb_content)
            temp_path = f.name
        fixer = PDBFixer(filename=temp_path)
        os.unlink(temp_path)
    else:
        # Load local file
        if not os.path.exists(input_path_or_id):
            raise FileNotFoundError(f"PDB file not found: {input_path_or_id}")
        if verbose:
            print(f"Loading {input_path_or_id}...")
        fixer = PDBFixer(filename=input_path_or_id)

    # Step 2: Remove heterogens (waters, ligands, ions)
    if verbose:
        print("Removing heterogens...")

    # Count before removal
    n_residues_before = sum(1 for _ in fixer.topology.residues())

    if not keep_heterogens:
        fixer.removeHeterogens(keepWater=keep_waters)

    n_residues_after = sum(1 for _ in fixer.topology.residues())
    stats['heterogens_removed'] = n_residues_before - n_residues_after

    if verbose:
        print(f"  Removed {stats['heterogens_removed']} heterogen residues")

    # Step 3: Replace non-standard residues
    if verbose:
        print("Checking for non-standard residues...")

    fixer.findNonstandardResidues()
    if fixer.nonstandardResidues:
        for residue, replacement in fixer.nonstandardResidues:
            old_name = residue.name
            if old_name not in stats['residues_replaced']:
                stats['residues_replaced'][old_name] = 0
            stats['residues_replaced'][old_name] += 1

        if verbose:
            for name, count in stats['residues_replaced'].items():
                print(f"  Replacing {count}x {name}")

        fixer.replaceNonstandardResidues()

    # Step 4: Select chain if specified
    if chain:
        if verbose:
            print(f"Selecting chain {chain}...")

        # Get all chain IDs
        chain_ids = set()
        for c in fixer.topology.chains():
            chain_ids.add(c.id)

        if chain not in chain_ids:
            raise ValueError(f"Chain {chain} not found. Available: {sorted(chain_ids)}")

        # Remove other chains by rebuilding topology
        chains_to_remove = [c for c in fixer.topology.chains() if c.id != chain]
        for c in chains_to_remove:
            # PDBFixer doesn't have direct chain removal, so we'll filter during output
            pass

    # Step 5: Find and add missing atoms
    if verbose:
        print("Finding missing atoms...")

    fixer.findMissingResidues()
    # Don't add missing residues (gaps) - just note them
    if fixer.missingResidues:
        if verbose:
            print(f"  Note: {len(fixer.missingResidues)} chain gaps (not adding)")
        fixer.missingResidues = {}

    fixer.findMissingAtoms()
    stats['missing_atoms_added'] = sum(len(atoms) for atoms in fixer.missingAtoms.values())
    stats['missing_terminals_added'] = sum(len(atoms) for atoms in fixer.missingTerminals.values())

    if verbose:
        print(f"  Missing heavy atoms: {stats['missing_atoms_added']}")
        print(f"  Missing terminals: {stats['missing_terminals_added']}")

    fixer.addMissingAtoms()

    # Step 6: Count final structure
    stats['final_chains'] = []
    stats['final_residues'] = 0
    stats['final_atoms'] = 0

    for c in fixer.topology.chains():
        if chain and c.id != chain:
            continue
        stats['final_chains'].append(c.id)
        for r in c.residues():
            stats['final_residues'] += 1
            for a in r.atoms():
                stats['final_atoms'] += 1

    if verbose:
        print(f"Final structure: {stats['final_residues']} residues, {stats['final_atoms']} atoms")
        print(f"Chains: {stats['final_chains']}")

    # Step 7: Write output
    if verbose:
        print(f"Writing {output_path}...")

    # Filter by chain if needed
    if chain:
        # Write with chain filter
        with open(output_path, 'w') as f:
            app.PDBFile.writeFile(
                fixer.topology,
                fixer.positions,
                f,
                keepIds=True
            )

        # Re-read and filter to single chain
        filter_pdb_chain(output_path, output_path, chain)
    else:
        with open(output_path, 'w') as f:
            app.PDBFile.writeFile(
                fixer.topology,
                fixer.positions,
                f,
                keepIds=True
            )

    if verbose:
        print("Done!")

    return stats


def filter_pdb_chain(input_path: str, output_path: str, chain: str):
    """Filter PDB to single chain."""
    lines = []
    with open(input_path, 'r') as f:
        for line in f:
            if line.startswith(('ATOM', 'HETATM', 'TER')):
                if len(line) > 21 and line[21] == chain:
                    lines.append(line)
            elif line.startswith(('HEADER', 'TITLE', 'CRYST', 'END')):
                lines.append(line)

    with open(output_path, 'w') as f:
        f.writelines(lines)
        if not any(l.startswith('END') for l in lines):
            f.write('END\n')


def main():
    parser = argparse.ArgumentParser(
        description='PRISM4D Stage 1: PDB Fetch & Sanitize',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s 6M0J clean_6M0J.pdb              # Fetch and clean
  %(prog)s 6M0J clean_6M0J_E.pdb --chain E  # Single chain
  %(prog)s local.pdb clean.pdb              # Clean local file
  %(prog)s local.pdb clean.pdb --keep-waters
        """
    )

    parser.add_argument('input', help='PDB ID (4 chars) or path to local PDB file')
    parser.add_argument('output', help='Output path for sanitized PDB')
    parser.add_argument('--chain', '-c', help='Select specific chain')
    parser.add_argument('--keep-waters', '-w', action='store_true',
                        help='Keep water molecules')
    parser.add_argument('--keep-heterogens', action='store_true',
                        help='Keep all heterogens (ligands, ions, etc.)')
    parser.add_argument('--quiet', '-q', action='store_true',
                        help='Suppress output')

    args = parser.parse_args()

    try:
        stats = sanitize_pdb(
            args.input,
            args.output,
            chain=args.chain,
            keep_waters=args.keep_waters,
            keep_heterogens=args.keep_heterogens,
            verbose=not args.quiet
        )

        if not args.quiet:
            print("\n=== Sanitization Summary ===")
            print(f"Input:  {stats['input']}")
            print(f"Output: {stats['output']}")
            print(f"Chains: {stats['final_chains']}")
            print(f"Residues: {stats['final_residues']}")
            print(f"Atoms: {stats['final_atoms']}")
            if stats['residues_replaced']:
                print(f"Replaced: {stats['residues_replaced']}")

        return 0

    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1


if __name__ == '__main__':
    sys.exit(main())
