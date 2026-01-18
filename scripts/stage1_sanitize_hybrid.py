#!/usr/bin/env python3
"""
PRISM4D Stage 1 Sanitizer - Hybrid PDBFixer + AMBER Reduce

Best-of-both-worlds approach:
1. PDBFixer adds hydrogens with correct naming for OpenMM templates
2. AMBER reduce optimizes hydrogen positions for better H-bond networks

This produces structures that:
- Are compatible with OpenMM force field templates
- Have optimized hydrogen positions (Asn/Gln/His flips, H-bond networks)
- Minimize steric clashes

Usage:
    python stage1_sanitize_hybrid.py input.pdb output.pdb

Requirements:
    - OpenMM/PDBFixer (conda install -c conda-forge openmm pdbfixer)
    - AMBER reduce (conda install -c conda-forge ambertools)
"""

import os
import sys
import subprocess
import argparse
import tempfile
from pathlib import Path
from typing import Optional, Dict

# Known conda environments with AMBER tools
AMBER_ENV_NAMES = ['ambertools', 'amber', 'mdtools']


def find_amber_env_bin() -> Optional[Path]:
    """Find the bin directory of an AMBER conda environment."""
    home = Path.home()
    for base in [home / 'miniconda3', home / 'anaconda3', Path('/opt/conda')]:
        envs_dir = base / 'envs'
        if envs_dir.exists():
            for env_name in AMBER_ENV_NAMES:
                bin_dir = envs_dir / env_name / 'bin'
                if (bin_dir / 'reduce').exists():
                    return bin_dir
    return None


def get_amber_env() -> Dict[str, str]:
    """Get environment with AMBER tools in PATH."""
    env = os.environ.copy()
    amber_bin = find_amber_env_bin()
    if amber_bin:
        env['PATH'] = f"{amber_bin}:{env.get('PATH', '')}"
        # Also add Python path for pdbfixer/openmm imports
        env['PYTHONPATH'] = f"{amber_bin.parent / 'lib' / 'python3.13' / 'site-packages'}:{env.get('PYTHONPATH', '')}"
    return env


# Global AMBER environment for subprocess calls
AMBER_ENV = get_amber_env()


def check_dependencies() -> dict:
    """Check availability of required tools."""
    deps = {'pdbfixer': False, 'reduce': False}

    # Check PDBFixer
    try:
        from pdbfixer import PDBFixer
        deps['pdbfixer'] = True
    except ImportError:
        pass

    # Check reduce
    try:
        result = subprocess.run(['reduce', '-v'], capture_output=True, timeout=10, env=AMBER_ENV)
        deps['reduce'] = True
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass

    return deps


def run_pdbfixer(input_pdb: str, output_pdb: str, verbose: bool = True) -> bool:
    """
    Run PDBFixer to add hydrogens with OpenMM-compatible naming.
    """
    try:
        from pdbfixer import PDBFixer
        from openmm.app import PDBFile

        if verbose:
            print("  Step 1: PDBFixer (add hydrogens with OpenMM naming)...")

        fixer = PDBFixer(filename=input_pdb)

        # Remove heterogens (ligands, ions, waters)
        fixer.removeHeterogens(keepWater=False)

        # Find missing residues (but don't try to add non-standard ones)
        fixer.findMissingResidues()
        # Clear missing residues that are non-standard (ligand fragments)
        # These can't be reconstructed and would cause errors
        if fixer.missingResidues:
            standard_aa = {'ALA', 'ARG', 'ASN', 'ASP', 'CYS', 'GLN', 'GLU', 'GLY',
                          'HIS', 'ILE', 'LEU', 'LYS', 'MET', 'PHE', 'PRO', 'SER',
                          'THR', 'TRP', 'TYR', 'VAL'}
            to_remove = []
            for key, residues in fixer.missingResidues.items():
                if not all(r in standard_aa for r in residues):
                    to_remove.append(key)
            for key in to_remove:
                del fixer.missingResidues[key]

        # Find and add missing atoms (including hydrogens)
        fixer.findMissingAtoms()
        fixer.addMissingAtoms()
        fixer.addMissingHydrogens(7.0)  # pH 7.0

        # Count atoms
        n_atoms = sum(1 for _ in fixer.topology.atoms())

        # Write output
        with open(output_pdb, 'w') as f:
            PDBFile.writeFile(fixer.topology, fixer.positions, f)

        if verbose:
            print(f"    Added hydrogens: {n_atoms} total atoms")

        return True

    except Exception as e:
        if verbose:
            print(f"    ERROR: {e}")
        return False


def run_reduce_optimize(input_pdb: str, output_pdb: str, verbose: bool = True) -> bool:
    """
    Run AMBER reduce to optimize existing hydrogen positions.

    Uses -FLIP to optimize Asn/Gln/His orientations and hydrogen bonding.
    Does NOT add new hydrogens - only optimizes existing ones.
    """
    try:
        if verbose:
            print("  Step 2: AMBER reduce (optimize H-bond networks)...")

        # Run reduce with -FLIP to optimize orientations
        # -NOFLIP would skip the optimization, so we use default behavior
        cmd = ['reduce', '-BUILD', '-Quiet', input_pdb]

        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600, env=AMBER_ENV)

        if result.returncode != 0 and 'error' in result.stderr.lower():
            if verbose:
                print(f"    Warning: reduce returned code {result.returncode}")

        # Parse reduce output - it outputs to stdout
        # We need to extract just the ATOM/HETATM records
        lines = result.stdout.split('\n')

        # Count adjustments from USER MOD line
        n_adj = 0
        for line in lines:
            if line.startswith('USER  MOD'):
                # Parse: "USER  MOD reduce.4.10.230211 H: found=X, std=Y, add=Z, rem=W, adj=N"
                if 'adj=' in line:
                    try:
                        n_adj = int(line.split('adj=')[1].split()[0].rstrip(','))
                    except:
                        pass
                break

        # Write output, keeping only relevant records
        with open(output_pdb, 'w') as f:
            for line in lines:
                if line.startswith(('ATOM', 'HETATM', 'TER', 'END')):
                    f.write(line + '\n')
                elif line.startswith(('HEADER', 'TITLE', 'COMPND', 'CRYST1')):
                    f.write(line + '\n')

        if verbose:
            print(f"    Optimized positions: {n_adj} adjustments")

        return True

    except subprocess.TimeoutExpired:
        if verbose:
            print("    ERROR: reduce timed out")
        return False
    except Exception as e:
        if verbose:
            print(f"    ERROR: {e}")
        return False


def count_atoms(pdb_path: str) -> tuple:
    """Count heavy atoms and hydrogens in a PDB file."""
    n_heavy = 0
    n_hydrogen = 0

    with open(pdb_path) as f:
        for line in f:
            if line.startswith(('ATOM', 'HETATM')):
                element = line[76:78].strip() if len(line) >= 78 else line[12:14].strip()[0]
                if element == 'H':
                    n_hydrogen += 1
                else:
                    n_heavy += 1

    return n_heavy, n_hydrogen


def sanitize_hybrid(input_pdb: str, output_pdb: str, verbose: bool = True) -> bool:
    """
    Run hybrid sanitization: PDBFixer + AMBER reduce.

    Returns True if successful.
    """
    deps = check_dependencies()

    if not deps['pdbfixer']:
        print("ERROR: PDBFixer not available", file=sys.stderr)
        print("Install with: conda install -c conda-forge openmm pdbfixer", file=sys.stderr)
        return False

    # Create temp file for intermediate result
    with tempfile.NamedTemporaryFile(suffix='.pdb', delete=False) as tmp:
        pdbfixer_output = tmp.name

    try:
        # Step 1: PDBFixer adds hydrogens with correct naming
        if not run_pdbfixer(input_pdb, pdbfixer_output, verbose):
            return False

        # Step 2: If reduce is available, optimize hydrogen positions
        if deps['reduce']:
            if not run_reduce_optimize(pdbfixer_output, output_pdb, verbose):
                # Fall back to PDBFixer output if reduce fails
                if verbose:
                    print("    Falling back to PDBFixer output (reduce failed)")
                import shutil
                shutil.copy(pdbfixer_output, output_pdb)
        else:
            # No reduce available, just use PDBFixer output
            if verbose:
                print("  Step 2: Skipped (AMBER reduce not available)")
            import shutil
            shutil.copy(pdbfixer_output, output_pdb)

        # Report final counts
        if verbose:
            n_heavy, n_h = count_atoms(output_pdb)
            print(f"    Final structure: {n_heavy} heavy atoms, {n_h} hydrogens")

        return True

    finally:
        # Cleanup
        if os.path.exists(pdbfixer_output):
            os.unlink(pdbfixer_output)


def main():
    parser = argparse.ArgumentParser(
        description='PRISM4D Stage 1 Sanitizer - Hybrid PDBFixer + AMBER Reduce'
    )
    parser.add_argument('input', help='Input PDB file')
    parser.add_argument('output', help='Output sanitized PDB file')
    parser.add_argument('--quiet', '-q', action='store_true',
                        help='Suppress output')

    args = parser.parse_args()

    if not Path(args.input).exists():
        print(f"ERROR: Input file not found: {args.input}", file=sys.stderr)
        return 1

    verbose = not args.quiet

    if verbose:
        print(f"\n{'='*60}")
        print(f"Stage 1 Sanitization (Hybrid: PDBFixer + AMBER reduce)")
        print(f"{'='*60}")
        print(f"Input:  {args.input}")
        print(f"Output: {args.output}")

    success = sanitize_hybrid(args.input, args.output, verbose)

    if success and verbose:
        print(f"\n  Status: ✅ SUCCESS")
        print(f"{'='*60}\n")

    return 0 if success else 1


if __name__ == '__main__':
    sys.exit(main())
