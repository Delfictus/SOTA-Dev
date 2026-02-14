#!/usr/bin/env python3
"""
PRISM4D Stage 1 Sanitizer - AMBER/Reduce Version

Uses AMBER's 'reduce' tool for high-quality hydrogen placement:
- Optimizes H-bond networks
- Considers Asn/Gln/His flip states
- Minimizes steric clashes
- Better initial coordinates for MD

This produces more stable starting structures than PDBFixer alone.

Pipeline:
1. Strip existing hydrogens (reduce -Trim)
2. Add optimized hydrogens (reduce -BUILD)
3. Validate output

Usage:
    python stage1_sanitize_amber.py input.pdb output.pdb

Requirements:
    - AMBER's 'reduce' tool (conda install -c conda-forge ambertools)
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
    return env


# Global AMBER environment for subprocess calls
AMBER_ENV = get_amber_env()


def check_reduce_available() -> bool:
    """Check if reduce is available in PATH or AMBER environment."""
    try:
        result = subprocess.run(['reduce', '-v'], capture_output=True, text=True, timeout=10, env=AMBER_ENV)
        return True
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


def run_reduce(input_pdb: str, output_pdb: str, flip: bool = True, verbose: bool = True) -> bool:
    """
    Run reduce to add optimized hydrogens.

    Args:
        input_pdb: Input PDB file (can have or not have hydrogens)
        output_pdb: Output PDB with optimized hydrogens
        flip: If True, optimize Asn/Gln/His orientations
        verbose: Print progress

    Returns:
        True if successful
    """
    # Step 1: Strip existing hydrogens
    if verbose:
        print(f"  Stripping existing hydrogens...")

    with tempfile.NamedTemporaryFile(suffix='.pdb', delete=False) as tmp:
        stripped_path = tmp.name

    try:
        # reduce -Trim removes hydrogens
        cmd = ['reduce', '-Trim', '-Quiet', input_pdb]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300, env=AMBER_ENV)

        if result.returncode != 0 and 'error' in result.stderr.lower():
            print(f"  Warning: reduce -Trim returned code {result.returncode}")

        # Write stripped structure
        with open(stripped_path, 'w') as f:
            f.write(result.stdout)

        # Count atoms
        n_atoms_stripped = sum(1 for line in result.stdout.split('\n')
                               if line.startswith('ATOM') or line.startswith('HETATM'))
        if verbose:
            print(f"    Atoms after stripping: {n_atoms_stripped}")

        # Step 2: Add optimized hydrogens
        if verbose:
            print(f"  Adding optimized hydrogens (reduce -BUILD)...")

        if flip:
            cmd = ['reduce', '-BUILD', '-Quiet', stripped_path]
        else:
            cmd = ['reduce', '-NOFLIP', '-Quiet', stripped_path]

        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600, env=AMBER_ENV)

        if result.returncode != 0 and 'error' in result.stderr.lower():
            print(f"  Warning: reduce -BUILD returned code {result.returncode}")

        # Write output
        with open(output_pdb, 'w') as f:
            for line in result.stdout.split('\n'):
                # Keep ATOM, HETATM, TER, END records
                if line.startswith(('ATOM', 'HETATM', 'TER', 'END', 'HEADER', 'TITLE',
                                    'COMPND', 'SOURCE', 'CRYST1', 'REMARK')):
                    f.write(line + '\n')

        # Count final atoms
        n_atoms_final = sum(1 for line in open(output_pdb)
                           if line.startswith('ATOM') or line.startswith('HETATM'))
        n_hydrogens_added = n_atoms_final - n_atoms_stripped

        if verbose:
            print(f"    Final atoms: {n_atoms_final}")
            print(f"    Hydrogens added: {n_hydrogens_added}")

        return True

    except subprocess.TimeoutExpired:
        print("  ERROR: reduce timed out")
        return False
    except Exception as e:
        print(f"  ERROR: {e}")
        return False
    finally:
        # Cleanup
        if os.path.exists(stripped_path):
            os.unlink(stripped_path)


def count_chains(pdb_path: str) -> list:
    """Count chains in a PDB file."""
    chains = set()
    with open(pdb_path) as f:
        for line in f:
            if line.startswith('ATOM'):
                chain = line[21:22].strip() or 'A'
                chains.add(chain)
    return sorted(chains)


def main():
    parser = argparse.ArgumentParser(
        description='PRISM4D Stage 1 Sanitizer using AMBER reduce'
    )
    parser.add_argument('input', help='Input PDB file')
    parser.add_argument('output', help='Output sanitized PDB file')
    parser.add_argument('--no-flip', action='store_true',
                        help='Disable Asn/Gln/His flip optimization')
    parser.add_argument('--quiet', '-q', action='store_true',
                        help='Suppress output')

    args = parser.parse_args()

    if not Path(args.input).exists():
        print(f"ERROR: Input file not found: {args.input}", file=sys.stderr)
        return 1

    # Check reduce availability
    if not check_reduce_available():
        print("ERROR: AMBER 'reduce' tool not found.", file=sys.stderr)
        print("Install with: conda install -c conda-forge ambertools", file=sys.stderr)
        return 1

    verbose = not args.quiet

    if verbose:
        print(f"\n{'='*60}")
        print(f"Stage 1 Sanitization (AMBER reduce)")
        print(f"{'='*60}")
        print(f"Input:  {args.input}")
        print(f"Output: {args.output}")

    # Run reduce
    success = run_reduce(args.input, args.output, flip=not args.no_flip, verbose=verbose)

    if success and verbose:
        chains = count_chains(args.output)
        print(f"\n  Chains: {chains}")
        print(f"  Status: ✅ SUCCESS")
        print(f"{'='*60}\n")

    return 0 if success else 1


if __name__ == '__main__':
    sys.exit(main())
