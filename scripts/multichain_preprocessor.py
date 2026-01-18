#!/usr/bin/env python3
"""
PRISM4D Smart Multi-Chain Preprocessor

Automatically analyzes inter-chain contacts to determine optimal processing:
- Independent chains (low contacts) → Per-chain processing
- Coupled chains (high contacts) → Whole-structure processing
- Simple structures (≤2 chains) → Standard processing

Pipeline:
1. Analyze structure (chain count, glycans, inter-chain contacts)
2. Smart routing based on contact analysis:
   - Low contact density → split → process individually → recombine
   - High contact density → process as whole structure
   - Disulfide bridges → always whole structure
3. Apply appropriate glycan handling based on domain use

Usage:
    python multichain_preprocessor.py input.pdb output_topology.json --mode cryptic
    python multichain_preprocessor.py input.pdb output_topology.json --mode escape
"""

import os
import sys
import json
import argparse
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, List, Tuple, Set, Optional
from collections import defaultdict


# Import from sibling modules
SCRIPT_DIR = Path(__file__).parent
sys.path.insert(0, str(SCRIPT_DIR))

from glycan_preprocessor import detect_glycans, GLYCAN_RESIDUES
from interchain_contacts import analyze_structure as analyze_contacts, ContactAnalysisResult


def analyze_structure(pdb_path: str, verbose: bool = True) -> Dict:
    """
    Analyze a PDB file for chains, glycans, inter-chain contacts, and routing decision.

    Returns dict with:
        - chains: list of chain IDs
        - n_chains: number of chains
        - atoms_per_chain: dict of chain -> atom count
        - total_atoms: total atom count
        - glycan_atoms: number of glycan atoms
        - has_glycans: bool
        - contact_analysis: ContactAnalysisResult or None
        - routing: 'standard', 'multichain', or 'whole'
        - routing_reason: explanation for routing decision
    """
    chains = defaultdict(int)
    glycan_atoms = 0
    total_atoms = 0

    with open(pdb_path, 'r') as f:
        for line in f:
            if line.startswith('ATOM'):
                chain_id = line[21:22].strip() or 'A'
                chains[chain_id] += 1
                total_atoms += 1
            elif line.startswith('HETATM'):
                res_name = line[17:20].strip()
                if res_name in GLYCAN_RESIDUES:
                    glycan_atoms += 1

    chain_list = sorted(chains.keys())
    n_chains = len(chain_list)

    # Determine routing based on chain count and contacts
    contact_result = None
    routing = 'standard'
    routing_reason = '≤2 chains, standard processing'

    if n_chains > 2:
        # Perform inter-chain contact analysis
        contact_result = analyze_contacts(pdb_path, verbose=False)

        if contact_result.total_disulfides > 0:
            routing = 'whole'
            routing_reason = f'Inter-chain disulfide bonds detected ({contact_result.total_disulfides})'
        elif contact_result.recommendation == 'multichain':
            routing = 'multichain'
            routing_reason = f'Low contact density ({contact_result.overall_contact_density:.2f}), chains are independent'
        else:
            routing = 'whole'
            routing_reason = f'High contact density ({contact_result.overall_contact_density:.2f}), chains are coupled'

        if verbose:
            print(f"\n  Contact Analysis:")
            print(f"    Total contacts: {contact_result.total_contacts}")
            print(f"    H-bonds: {contact_result.total_hbonds}")
            print(f"    Disulfides: {contact_result.total_disulfides}")
            print(f"    Salt bridges: {contact_result.total_salt_bridges}")
            print(f"    Contact density: {contact_result.overall_contact_density:.3f}")
            print(f"    Confidence: {contact_result.confidence:.0%}")

    return {
        'chains': chain_list,
        'n_chains': n_chains,
        'atoms_per_chain': dict(chains),
        'total_atoms': total_atoms,
        'glycan_atoms': glycan_atoms,
        'has_glycans': glycan_atoms > 0,
        'contact_analysis': contact_result,
        'routing': routing,
        'routing_reason': routing_reason,
    }


def split_by_chain(pdb_path: str, output_dir: str) -> Dict[str, str]:
    """
    Split a PDB file into individual chain files.

    Returns dict of chain_id -> file_path
    """
    os.makedirs(output_dir, exist_ok=True)

    chain_lines = defaultdict(list)

    with open(pdb_path, 'r') as f:
        for line in f:
            if line.startswith('ATOM'):
                chain_id = line[21:22].strip() or 'A'
                chain_lines[chain_id].append(line)
            elif line.startswith('HETATM'):
                # Keep HETATM with their associated chain
                chain_id = line[21:22].strip() or 'A'
                res_name = line[17:20].strip()
                # Only keep non-glycan HETATM (metals, etc.)
                if res_name not in GLYCAN_RESIDUES:
                    chain_lines[chain_id].append(line)

    chain_files = {}
    pdb_name = Path(pdb_path).stem

    for chain_id, lines in chain_lines.items():
        output_path = os.path.join(output_dir, f"{pdb_name}_chain{chain_id}.pdb")
        with open(output_path, 'w') as f:
            f.writelines(lines)
            f.write("END\n")
        chain_files[chain_id] = output_path

    return chain_files


def run_command(cmd: List[str], description: str, verbose: bool = True) -> bool:
    """Run a command and return success status."""
    if verbose:
        print(f"  {description}...", end=" ", flush=True)

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=600  # 10 minute timeout
        )
        if result.returncode == 0:
            if verbose:
                print("OK")
            return True
        else:
            if verbose:
                print("FAILED")
                print(f"    Error: {result.stderr[:200]}")
            return False
    except Exception as e:
        if verbose:
            print(f"ERROR: {e}")
        return False


def check_reduce_available() -> bool:
    """Check if AMBER reduce is available."""
    try:
        result = subprocess.run(['reduce', '-v'], capture_output=True, timeout=10)
        return True
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


def process_single_chain(
    pdb_path: str,
    output_dir: str,
    chain_id: str,
    mode: str,
    use_amber: bool = False,
    verbose: bool = True
) -> Tuple[bool, str]:
    """
    Process a single chain through glycan preprocessing, Stage 1, and Stage 2.

    Args:
        use_amber: If True, use AMBER reduce for hydrogen placement (higher quality)

    Returns (success, topology_path)
    """
    base_name = Path(pdb_path).stem

    # Step 1: Glycan preprocessing (just removes glycans/waters for chain)
    preprocessed_path = os.path.join(output_dir, f"{base_name}_preprocessed.pdb")

    # For per-chain, always use cryptic mode (remove glycans) since
    # glycans span chains and we're processing individually
    cmd = [
        "python3", str(SCRIPT_DIR / "glycan_preprocessor.py"),
        pdb_path, preprocessed_path,
        "--mode", "cryptic", "-q"
    ]
    if not run_command(cmd, f"Chain {chain_id}: Glycan preprocessing", verbose):
        return False, ""

    # Step 2: Stage 1 sanitization (choose hybrid PDBFixer+reduce or PDBFixer only)
    sanitized_path = os.path.join(output_dir, f"{base_name}_sanitized.pdb")
    if use_amber:
        # Hybrid: PDBFixer adds H with correct naming, reduce optimizes positions
        cmd = [
            "python3", str(SCRIPT_DIR / "stage1_sanitize_hybrid.py"),
            preprocessed_path, sanitized_path, "-q"
        ]
        sanitizer_name = "Stage 1 (PDBFixer + reduce)"
    else:
        cmd = [
            "python3", str(SCRIPT_DIR / "stage1_sanitize.py"),
            preprocessed_path, sanitized_path
        ]
        sanitizer_name = "Stage 1 (PDBFixer)"
    if not run_command(cmd, f"Chain {chain_id}: {sanitizer_name}", verbose):
        return False, ""

    # Step 3: Stage 2 topology
    topology_path = os.path.join(output_dir, f"{base_name}_topology.json")
    cmd = [
        "python3", str(SCRIPT_DIR / "stage2_topology.py"),
        sanitized_path, topology_path
    ]
    if not run_command(cmd, f"Chain {chain_id}: Stage 2 topology", verbose):
        return False, ""

    return True, topology_path


def combine_topologies(topology_files: List[str], output_path: str, verbose: bool = True) -> bool:
    """
    Combine multiple chain topologies into a single topology.
    """
    if verbose:
        print(f"  Combining {len(topology_files)} chain topologies...", end=" ", flush=True)

    cmd = [
        "python3", str(SCRIPT_DIR / "combine_chain_topologies.py"),
        output_path
    ] + topology_files

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        if result.returncode == 0:
            if verbose:
                print("OK")
            return True
        else:
            if verbose:
                print("FAILED")
                print(f"    {result.stderr[:200]}")
            return False
    except Exception as e:
        if verbose:
            print(f"ERROR: {e}")
        return False


def process_multichain(
    pdb_path: str,
    output_topology: str,
    mode: str,
    work_dir: str = None,
    use_amber: bool = False,
    verbose: bool = True
) -> bool:
    """
    Process a multi-chain structure using per-chain pipeline.

    1. Split into individual chains
    2. Process each chain independently
    3. Recombine topologies

    Args:
        use_amber: If True, use AMBER reduce for hydrogen placement
    """
    pdb_name = Path(pdb_path).stem

    # Create work directory
    if work_dir is None:
        work_dir = tempfile.mkdtemp(prefix=f"prism_multichain_{pdb_name}_")
    os.makedirs(work_dir, exist_ok=True)

    if verbose:
        print(f"\n{'='*60}")
        print(f"Multi-Chain Pipeline: {pdb_name}")
        print(f"{'='*60}")
        print(f"Work directory: {work_dir}")
        if use_amber:
            print(f"Hydrogen placement: AMBER reduce (high quality)")
        else:
            print(f"Hydrogen placement: PDBFixer (standard)")

    # Step 1: Split by chain
    chains_dir = os.path.join(work_dir, "chains")
    chain_files = split_by_chain(pdb_path, chains_dir)

    if verbose:
        print(f"\nSplit into {len(chain_files)} chains: {', '.join(sorted(chain_files.keys()))}")

    # Step 2: Process each chain
    topology_files = []
    chain_dirs = {}

    for chain_id in sorted(chain_files.keys()):
        chain_pdb = chain_files[chain_id]
        chain_work_dir = os.path.join(work_dir, f"chain_{chain_id}")
        os.makedirs(chain_work_dir, exist_ok=True)
        chain_dirs[chain_id] = chain_work_dir

        success, topo_path = process_single_chain(
            chain_pdb, chain_work_dir, chain_id, mode, use_amber, verbose
        )

        if success:
            topology_files.append(topo_path)
        else:
            print(f"  WARNING: Chain {chain_id} failed, skipping")

    if len(topology_files) == 0:
        print("ERROR: No chains processed successfully")
        return False

    if verbose:
        print(f"\nSuccessfully processed {len(topology_files)}/{len(chain_files)} chains")

    # Step 3: Combine topologies
    if verbose:
        print(f"\nCombining topologies...")

    if not combine_topologies(topology_files, output_topology, verbose):
        return False

    # Verify output
    if os.path.exists(output_topology):
        size_mb = os.path.getsize(output_topology) / (1024 * 1024)
        with open(output_topology) as f:
            topo = json.load(f)
        if verbose:
            print(f"\n{'='*60}")
            print(f"Combined Topology: {output_topology}")
            print(f"  Atoms: {topo['n_atoms']:,}")
            print(f"  Chains: {topo['n_chains']}")
            print(f"  Bonds: {len(topo['bonds']):,}")
            print(f"  Size: {size_mb:.2f} MB")
            print(f"{'='*60}")
        return True
    else:
        print("ERROR: Output topology not created")
        return False


def process_structure(
    pdb_path: str,
    output_topology: str,
    mode: str = "auto",
    force_multichain: bool = False,
    force_whole: bool = False,
    use_amber: bool = False,
    work_dir: str = None,
    verbose: bool = True
) -> bool:
    """
    Main entry point: analyze structure and process appropriately.

    Routing logic:
    - ≤2 chains: Standard processing
    - >2 chains + low contact density: Multi-chain processing (independent chains)
    - >2 chains + high contact density: Whole-structure processing (coupled chains)
    - Disulfide bridges present: Always whole-structure

    Args:
        use_amber: If True, use AMBER reduce for high-quality hydrogen placement
    """
    # Analyze structure with contact analysis
    analysis = analyze_structure(pdb_path, verbose=verbose)

    if verbose:
        print(f"\n{'='*60}")
        print(f"Structure Analysis: {Path(pdb_path).name}")
        print(f"{'='*60}")
        print(f"  Chains: {analysis['n_chains']} ({', '.join(analysis['chains'])})")
        print(f"  Total atoms: {analysis['total_atoms']:,}")
        print(f"  Glycan atoms: {analysis['glycan_atoms']}")

    # Determine routing
    routing = analysis['routing']
    routing_reason = analysis['routing_reason']

    # Override with force flags
    if force_multichain and analysis['n_chains'] > 2:
        routing = 'multichain'
        routing_reason = 'Forced multi-chain processing'
    elif force_whole and analysis['n_chains'] > 2:
        routing = 'whole'
        routing_reason = 'Forced whole-structure processing'

    if verbose:
        routing_symbol = {'standard': '✨', 'multichain': '🔀', 'whole': '📦'}[routing]
        print(f"\n  Routing: {routing_symbol} {routing.upper()}")
        print(f"  Reason: {routing_reason}")

    # Check AMBER availability if requested
    if use_amber and not check_reduce_available():
        print("WARNING: AMBER reduce not available, falling back to PDBFixer")
        use_amber = False

    if verbose and use_amber:
        print(f"  Hydrogen placement: AMBER reduce (high quality)")

    # Execute appropriate pipeline
    if routing == 'multichain':
        if verbose:
            print(f"\n>>> Using MULTI-CHAIN pipeline (independent chains)")
        return process_multichain(pdb_path, output_topology, mode, work_dir, use_amber, verbose)
    else:
        # 'whole' or 'standard' - both use whole-structure processing
        if verbose:
            if routing == 'whole':
                print(f"\n>>> Using WHOLE-STRUCTURE pipeline (coupled chains)")
            else:
                print(f"\n>>> Using STANDARD pipeline (≤2 chains)")

        # Standard single-structure processing
        pdb_name = Path(pdb_path).stem
        if work_dir is None:
            work_dir = tempfile.mkdtemp(prefix=f"prism_{routing}_{pdb_name}_")
        os.makedirs(work_dir, exist_ok=True)

        # Glycan preprocessing
        preprocessed = os.path.join(work_dir, f"{pdb_name}_preprocessed.pdb")
        cmd = ["python3", str(SCRIPT_DIR / "glycan_preprocessor.py"),
               pdb_path, preprocessed, "--mode", mode, "-q"]
        if not run_command(cmd, "Glycan preprocessing", verbose):
            return False

        # Stage 1 sanitization (choose hybrid PDBFixer+reduce or PDBFixer only)
        sanitized = os.path.join(work_dir, f"{pdb_name}_sanitized.pdb")
        if use_amber:
            # Hybrid: PDBFixer adds H with correct naming, reduce optimizes positions
            cmd = ["python3", str(SCRIPT_DIR / "stage1_sanitize_hybrid.py"),
                   preprocessed, sanitized, "-q"]
            sanitizer_desc = "Stage 1 (PDBFixer + reduce)"
        else:
            cmd = ["python3", str(SCRIPT_DIR / "stage1_sanitize.py"),
                   preprocessed, sanitized]
            sanitizer_desc = "Stage 1 (PDBFixer)"
        if not run_command(cmd, sanitizer_desc, verbose):
            return False

        # Stage 2
        cmd = ["python3", str(SCRIPT_DIR / "stage2_topology.py"),
               sanitized, output_topology]
        if not run_command(cmd, "Stage 2 topology", verbose):
            return False

        return True


def main():
    parser = argparse.ArgumentParser(
        description='PRISM4D Smart Multi-Chain Preprocessor',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Smart routing based on inter-chain contact analysis:

  STANDARD:    ≤2 chains → whole-structure processing
  MULTICHAIN:  >2 chains + low contact density → per-chain processing
  WHOLE:       >2 chains + high contact density → whole-structure processing
               (also used when disulfide bridges detected between chains)

Examples:
  %(prog)s 5IRE.pdb 5IRE_topology.json --mode cryptic  # Will auto-route to MULTICHAIN
  %(prog)s 1HXY.pdb 1HXY_topology.json --mode cryptic  # Will auto-route to WHOLE
  %(prog)s 6LU7.pdb 6LU7_topology.json --mode cryptic  # Will auto-route to STANDARD
  %(prog)s input.pdb output.json --force-multichain    # Override to MULTICHAIN
  %(prog)s input.pdb output.json --force-whole         # Override to WHOLE
  %(prog)s input.pdb output.json --use-amber           # Use AMBER reduce for H placement
        """
    )

    parser.add_argument('input', help='Input PDB file')
    parser.add_argument('output', help='Output topology JSON file')
    parser.add_argument('--mode', '-m', choices=['cryptic', 'escape', 'auto'],
                        default='cryptic', help='Glycan handling mode (default: cryptic)')
    parser.add_argument('--force-multichain', '-f', action='store_true',
                        help='Force multi-chain processing (override contact analysis)')
    parser.add_argument('--force-whole', action='store_true',
                        help='Force whole-structure processing (override contact analysis)')
    parser.add_argument('--use-amber', '-a', action='store_true',
                        help='Use AMBER reduce for high-quality hydrogen placement')
    parser.add_argument('--work-dir', '-w', help='Working directory (default: temp)')
    parser.add_argument('--quiet', '-q', action='store_true',
                        help='Suppress output')

    args = parser.parse_args()

    if not Path(args.input).exists():
        print(f"ERROR: Input file not found: {args.input}", file=sys.stderr)
        return 1

    if args.force_multichain and args.force_whole:
        print("ERROR: Cannot use both --force-multichain and --force-whole", file=sys.stderr)
        return 1

    success = process_structure(
        args.input,
        args.output,
        mode=args.mode,
        force_multichain=args.force_multichain,
        force_whole=args.force_whole,
        use_amber=args.use_amber,
        work_dir=args.work_dir,
        verbose=not args.quiet
    )

    return 0 if success else 1


if __name__ == '__main__':
    sys.exit(main())
