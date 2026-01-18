#!/usr/bin/env python3
"""
PRISM4D Glycan-Aware Preprocessor

Handles glycosylated proteins for multiple downstream applications:
- Cryptic site detection: De-glycosylate (ASN→ASP mutation)
- Viral escape prediction: Keep glycans (GLYCAM06 force field)
- Drug binding prediction: Keep glycans (GLYCAM06 force field)
- General MD: Configurable

Glycan Types Supported:
- N-linked: NAG, MAN, BMA, FUC, GAL, SIA attached to ASN
- O-linked: NAG attached to SER/THR (future)

Usage:
    python glycan_preprocessor.py input.pdb output.pdb --mode cryptic
    python glycan_preprocessor.py input.pdb output.pdb --mode escape
    python glycan_preprocessor.py input.pdb output.pdb --mode full-glycan
"""

import sys
import re
import argparse
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Set
from collections import defaultdict


# Common glycan residue names in PDB files
GLYCAN_RESIDUES = {
    # N-acetyl hexosamines
    'NAG', 'NDG',  # N-acetyl-D-glucosamine
    'NGA', 'A2G',  # N-acetyl-D-galactosamine

    # Hexoses
    'MAN', 'BMA',  # Mannose, beta-mannose
    'GAL', 'GLA',  # Galactose
    'GLC', 'BGC',  # Glucose
    'FUC', 'FUL',  # Fucose

    # Sialic acids
    'SIA', 'SLB',  # Sialic acid

    # Other common glycans
    'XYS', 'XYP',  # Xylose
    'RIB',         # Ribose
}

# Residues that can be glycosylated
GLYCOSYLATABLE_RESIDUES = {
    'ASN',  # N-linked glycosylation
    'SER',  # O-linked glycosylation
    'THR',  # O-linked glycosylation
}


@dataclass
class GlycosylationSite:
    """Represents a detected glycosylation site."""
    chain_id: str
    residue_id: int
    residue_name: str  # ASN, SER, or THR
    glycan_residues: List[str] = field(default_factory=list)
    glycan_type: str = "N-linked"  # or "O-linked"
    sequon: Optional[str] = None  # e.g., "N-X-S" for N-linked

    def __str__(self):
        return f"{self.chain_id}:{self.residue_name}{self.residue_id} ({self.glycan_type}, {len(self.glycan_residues)} sugars)"


@dataclass
class GlycanAnalysis:
    """Complete glycan analysis of a structure."""
    pdb_path: str
    n_glycan_atoms: int
    n_glycan_residues: int
    glycosylation_sites: List[GlycosylationSite]
    glycan_chains: Set[str]
    has_glycans: bool

    def summary(self) -> str:
        lines = [
            f"=== Glycan Analysis: {Path(self.pdb_path).name} ===",
            f"Glycan atoms: {self.n_glycan_atoms}",
            f"Glycan residues: {self.n_glycan_residues}",
            f"Glycosylation sites: {len(self.glycosylation_sites)}",
        ]
        for site in self.glycosylation_sites:
            lines.append(f"  - {site}")
        return "\n".join(lines)


def detect_glycans(pdb_path: str) -> GlycanAnalysis:
    """
    Detect glycosylation sites and glycan residues in a PDB file.

    Returns GlycanAnalysis with all detected sites.
    """
    glycan_atoms = 0
    glycan_residues = set()
    glycan_chains = set()
    protein_residues = {}  # (chain, resid) -> resname
    glycan_connections = defaultdict(list)  # (chain, resid) -> list of glycan resnames

    with open(pdb_path, 'r') as f:
        for line in f:
            if line.startswith('ATOM') or line.startswith('HETATM'):
                # Parse PDB line
                record = line[0:6].strip()
                atom_name = line[12:16].strip()
                res_name = line[17:20].strip()
                chain_id = line[21:22].strip() or 'A'
                res_id = int(line[22:26].strip())
                x = float(line[30:38].strip())
                y = float(line[38:46].strip())
                z = float(line[46:54].strip())

                if res_name in GLYCAN_RESIDUES:
                    glycan_atoms += 1
                    glycan_residues.add((chain_id, res_id, res_name))
                    glycan_chains.add(chain_id)
                elif res_name in GLYCOSYLATABLE_RESIDUES:
                    protein_residues[(chain_id, res_id)] = res_name

    # Find which protein residues are glycosylated
    # by checking proximity or chain linkage
    sites = []

    # For N-linked, look for ASN residues that have glycan in same chain
    # or glycan chain that starts near ASN
    for (chain_id, res_id), res_name in protein_residues.items():
        if res_name == 'ASN':
            # Check if any glycan is connected (same chain, nearby residue number)
            connected_glycans = []
            for (g_chain, g_resid, g_resname) in glycan_residues:
                # Glycans attached to ASN typically start at a specific residue
                # Check if glycan chain matches or is a glycan-only chain
                if g_chain == chain_id or g_chain in glycan_chains:
                    connected_glycans.append(g_resname)

            if connected_glycans:
                site = GlycosylationSite(
                    chain_id=chain_id,
                    residue_id=res_id,
                    residue_name=res_name,
                    glycan_residues=list(set(connected_glycans)),
                    glycan_type="N-linked",
                    sequon="N-X-S/T"
                )
                sites.append(site)

    # Also check for O-linked glycosylation on SER/THR
    for (chain_id, res_id), res_name in protein_residues.items():
        if res_name in ('SER', 'THR'):
            connected_glycans = []
            for (g_chain, g_resid, g_resname) in glycan_residues:
                if g_chain == chain_id:
                    connected_glycans.append(g_resname)

            if connected_glycans:
                site = GlycosylationSite(
                    chain_id=chain_id,
                    residue_id=res_id,
                    residue_name=res_name,
                    glycan_residues=list(set(connected_glycans)),
                    glycan_type="O-linked",
                )
                sites.append(site)

    return GlycanAnalysis(
        pdb_path=pdb_path,
        n_glycan_atoms=glycan_atoms,
        n_glycan_residues=len(glycan_residues),
        glycosylation_sites=sites,
        glycan_chains=glycan_chains,
        has_glycans=glycan_atoms > 0
    )


def remove_glycans_only(
    pdb_path: str,
    output_path: str,
    verbose: bool = True
) -> Dict[str, int]:
    """
    Remove glycan residues without mutating protein residues.

    This is the safest approach for AMBER - the ASN residues remain intact,
    the glycan atoms are simply removed. OpenMM/AMBER will treat the ASN
    as a normal asparagine without any covalent modification.

    Returns stats dict with counts.
    """
    stats = {
        'atoms_written': 0,
        'glycan_atoms_removed': 0,
        'water_atoms_removed': 0,
        'other_removed': 0,
    }

    # Residues to remove (glycans, waters, common crystallization artifacts)
    remove_residues = GLYCAN_RESIDUES | {'HOH', 'WAT', 'GBL', 'EDO', 'GOL', 'PEG', 'SO4', 'PO4'}

    output_lines = []

    with open(pdb_path, 'r') as f:
        for line in f:
            if line.startswith('ATOM'):
                output_lines.append(line)
                stats['atoms_written'] += 1

            elif line.startswith('HETATM'):
                res_name = line[17:20].strip()

                if res_name in GLYCAN_RESIDUES:
                    stats['glycan_atoms_removed'] += 1
                    continue
                elif res_name in ('HOH', 'WAT'):
                    stats['water_atoms_removed'] += 1
                    continue
                elif res_name in remove_residues:
                    stats['other_removed'] += 1
                    continue
                else:
                    # Keep other HETATM (metals like ZN, MG, etc.)
                    output_lines.append(line)
                    stats['atoms_written'] += 1

            elif line.startswith('TER') or line.startswith('END'):
                output_lines.append(line)
            else:
                output_lines.append(line)

    with open(output_path, 'w') as f:
        f.writelines(output_lines)

    if verbose:
        print(f"Removed {stats['glycan_atoms_removed']} glycan atoms")
        print(f"Removed {stats['water_atoms_removed']} water atoms")
        if stats['other_removed']:
            print(f"Removed {stats['other_removed']} other HETATM atoms")
        print(f"Written {stats['atoms_written']} atoms to {output_path}")

    return stats


def mutate_glycosylation_sites(
    pdb_path: str,
    output_path: str,
    sites: List[GlycosylationSite],
    mutation: str = "ASP",
    remove_glycans: bool = True,
    verbose: bool = True
) -> Dict[str, int]:
    """
    Mutate glycosylation sites for de-glycosylated processing.

    NOTE: This function is deprecated for AMBER processing.
    Use remove_glycans_only() instead, which safely removes glycans
    without modifying protein residues.

    ASN → ASP mutation causes template matching issues in OpenMM.
    """
    # Redirect to the simpler, working approach
    if verbose:
        print("Note: Using glycan removal without mutation (safer for AMBER)")
    return remove_glycans_only(pdb_path, output_path, verbose)


def prepare_for_glycam(
    pdb_path: str,
    output_path: str,
    verbose: bool = True
) -> Dict[str, int]:
    """
    Prepare structure for GLYCAM force field processing.

    - Keeps all glycan residues (NAG, MAN, etc.)
    - Removes waters and ions
    - Validates glycan naming conventions

    Returns stats dict.
    """
    stats = {
        'protein_atoms': 0,
        'glycan_atoms': 0,
        'removed_atoms': 0,
    }

    output_lines = []

    with open(pdb_path, 'r') as f:
        for line in f:
            if line.startswith('ATOM'):
                output_lines.append(line)
                stats['protein_atoms'] += 1

            elif line.startswith('HETATM'):
                res_name = line[17:20].strip()

                if res_name in GLYCAN_RESIDUES:
                    output_lines.append(line)
                    stats['glycan_atoms'] += 1
                elif res_name in ('HOH', 'WAT', 'CL', 'NA', 'K', 'MG', 'CA', 'ZN'):
                    stats['removed_atoms'] += 1
                    continue
                else:
                    # Keep other HETATM (could be cofactors)
                    output_lines.append(line)

            elif line.startswith('TER') or line.startswith('END'):
                output_lines.append(line)
            else:
                output_lines.append(line)

    with open(output_path, 'w') as f:
        f.writelines(output_lines)

    if verbose:
        print(f"Prepared for GLYCAM: {stats['protein_atoms']} protein + {stats['glycan_atoms']} glycan atoms")
        print(f"Removed {stats['removed_atoms']} water/ion atoms")

    return stats


class GlycanPreprocessor:
    """
    Main preprocessor class that routes structures based on downstream task.
    """

    MODES = {
        'cryptic': 'Cryptic site detection (de-glycosylate, ASN→ASP)',
        'escape': 'Viral escape prediction (keep glycans, GLYCAM)',
        'drug': 'Drug binding prediction (keep glycans, GLYCAM)',
        'full-glycan': 'Full glycan simulation (keep all, GLYCAM)',
        'auto': 'Auto-detect best strategy based on glycan content',
    }

    def __init__(self, mode: str = 'auto', verbose: bool = True):
        if mode not in self.MODES:
            raise ValueError(f"Unknown mode: {mode}. Valid modes: {list(self.MODES.keys())}")
        self.mode = mode
        self.verbose = verbose

    def process(
        self,
        input_path: str,
        output_path: str,
        force_glycam: bool = False
    ) -> Tuple[str, GlycanAnalysis]:
        """
        Process a PDB file according to the selected mode.

        Returns (strategy_used, glycan_analysis)
        """
        # Analyze glycans
        analysis = detect_glycans(input_path)

        if self.verbose:
            print(analysis.summary())
            print()

        # Determine strategy
        if self.mode == 'auto':
            if not analysis.has_glycans:
                strategy = 'none'  # No glycans, simple pass-through
            elif analysis.n_glycan_atoms < 50:
                strategy = 'mutate'  # Few glycans, safe to mutate
            else:
                strategy = 'glycam'  # Many glycans, use GLYCAM
        elif self.mode == 'cryptic':
            strategy = 'mutate' if analysis.has_glycans else 'none'
        elif self.mode in ('escape', 'drug', 'full-glycan'):
            strategy = 'glycam' if analysis.has_glycans else 'none'
        else:
            strategy = 'none'

        # Override with force_glycam if requested
        if force_glycam and analysis.has_glycans:
            strategy = 'glycam'

        # Execute strategy
        if strategy == 'none':
            if self.verbose:
                print("Strategy: Pass-through (no glycans or glycan-agnostic mode)")
            # Just copy and clean up waters
            self._clean_copy(input_path, output_path)

        elif strategy == 'mutate':
            if self.verbose:
                print("Strategy: Remove glycans (keep protein residues intact)")
            remove_glycans_only(
                input_path,
                output_path,
                verbose=self.verbose
            )

        elif strategy == 'glycam':
            if self.verbose:
                print("Strategy: GLYCAM-ready (keeping glycans)")
            prepare_for_glycam(
                input_path,
                output_path,
                verbose=self.verbose
            )

        return strategy, analysis

    def _clean_copy(self, input_path: str, output_path: str):
        """Copy PDB, removing waters and common crystallization artifacts."""
        skip_residues = {'HOH', 'WAT', 'CL', 'NA', 'K', 'SO4', 'PO4', 'GOL', 'EDO', 'PEG'}

        with open(input_path, 'r') as f_in, open(output_path, 'w') as f_out:
            for line in f_in:
                if line.startswith('HETATM'):
                    res_name = line[17:20].strip()
                    if res_name in skip_residues:
                        continue
                f_out.write(line)


def check_glycam_available() -> Tuple[bool, str]:
    """
    Check if GLYCAM force field parameters are available.

    Returns (is_available, path_or_message)
    """
    try:
        from openmm import app

        # Try to load GLYCAM force field
        # OpenMM doesn't include GLYCAM by default, but we can check
        glycam_paths = [
            'GLYCAM_06j-1.xml',
            'glycam06.xml',
        ]

        for gpath in glycam_paths:
            try:
                ff = app.ForceField('amber14-all.xml', gpath)
                return True, gpath
            except:
                continue

        return False, "GLYCAM force field not found. Install from glycam.org or use de-glycosylation mode."

    except ImportError:
        return False, "OpenMM not available"


def main():
    parser = argparse.ArgumentParser(
        description='PRISM4D Glycan-Aware Preprocessor',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Modes:
  cryptic     - Cryptic site detection: Remove glycans, mutate ASN→ASP
  escape      - Viral escape prediction: Keep glycans for antibody epitope analysis
  drug        - Drug binding: Keep glycans for accurate binding site geometry
  full-glycan - Full simulation: Keep all glycans with GLYCAM parameters
  auto        - Auto-detect based on glycan content

Examples:
  %(prog)s input.pdb output.pdb --mode cryptic
  %(prog)s input.pdb output.pdb --mode escape --analyze-only
  %(prog)s 2VWD.pdb 2VWD_apo.pdb --mode cryptic
  %(prog)s 6VXX.pdb 6VXX_glycam.pdb --mode escape
        """
    )

    parser.add_argument('input', help='Input PDB file')
    parser.add_argument('output', help='Output PDB file', nargs='?')
    parser.add_argument('--mode', '-m', choices=list(GlycanPreprocessor.MODES.keys()),
                        default='auto', help='Processing mode (default: auto)')
    parser.add_argument('--analyze-only', '-a', action='store_true',
                        help='Only analyze glycans, do not modify')
    parser.add_argument('--check-glycam', action='store_true',
                        help='Check if GLYCAM force field is available')
    parser.add_argument('--quiet', '-q', action='store_true',
                        help='Suppress output')

    args = parser.parse_args()

    if args.check_glycam:
        available, msg = check_glycam_available()
        if available:
            print(f"GLYCAM available: {msg}")
        else:
            print(f"GLYCAM not available: {msg}")
        return 0 if available else 1

    if not Path(args.input).exists():
        print(f"ERROR: Input file not found: {args.input}", file=sys.stderr)
        return 1

    if args.analyze_only:
        analysis = detect_glycans(args.input)
        print(analysis.summary())
        return 0

    if not args.output:
        print("ERROR: Output file required (unless using --analyze-only)", file=sys.stderr)
        return 1

    preprocessor = GlycanPreprocessor(mode=args.mode, verbose=not args.quiet)
    strategy, analysis = preprocessor.process(args.input, args.output)

    if not args.quiet:
        print(f"\nProcessing complete: {strategy} strategy applied")
        print(f"Output: {args.output}")

    return 0


if __name__ == '__main__':
    sys.exit(main())
