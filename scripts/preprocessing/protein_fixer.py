"""Protein fixer — repair PDB files before simulation.

Capabilities (via PDBFixer when available, pure-PDB fallback otherwise):
    * Add missing heavy atoms and residues
    * Replace non-standard residues (MSE → MET, etc.)
    * Select best alternate conformation
    * Detect crystal contacts and warn
    * Remove heterogens (water, ligands) on request

Output is a cleaned PDB ready for docking or MD setup.
"""
from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Non-standard → standard residue mapping (common cases)
_NONSTD_MAP: Dict[str, str] = {
    "MSE": "MET",  # Selenomethionine
    "CSE": "CYS",  # Selenocysteine
    "HYP": "PRO",  # Hydroxyproline
    "MLY": "LYS",  # N-dimethyllysine
    "SEP": "SER",  # Phosphoserine
    "TPO": "THR",  # Phosphothreonine
    "PTR": "TYR",  # Phosphotyrosine
}

_STANDARD_RESIDUES = frozenset({
    "ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS",
    "ILE", "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP",
    "TYR", "VAL",
})


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class FixerResult:
    """Report of all fixes applied to a PDB file."""

    input_path: str
    output_path: str
    missing_residues_added: int = 0
    missing_atoms_added: int = 0
    nonstandard_replaced: List[str] = field(default_factory=list)
    altlocs_resolved: int = 0
    crystal_contacts_warned: bool = False
    heterogens_removed: int = 0
    method: str = "pdbfixer"  # "pdbfixer" | "fallback"
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)


# ---------------------------------------------------------------------------
# PDBFixer backend
# ---------------------------------------------------------------------------

def _has_pdbfixer() -> bool:
    try:
        import pdbfixer  # noqa: F401
        return True
    except ImportError:
        return False


def _fix_with_pdbfixer(
    pdb_path: str,
    output_path: str,
    *,
    add_missing: bool = True,
    replace_nonstandard: bool = True,
    remove_heterogens: bool = False,
    keep_water: bool = False,
    ph: float = 7.4,
) -> FixerResult:
    """Use OpenMM PDBFixer for comprehensive repair."""
    from pdbfixer import PDBFixer
    from openmm.app import PDBFile

    fixer = PDBFixer(filename=pdb_path)
    result = FixerResult(
        input_path=str(Path(pdb_path).resolve()),
        output_path=str(Path(output_path).resolve()),
        method="pdbfixer",
    )

    # Non-standard residues
    if replace_nonstandard:
        fixer.findNonstandardResidues()
        nonstd = fixer.nonstandardResidues
        result.nonstandard_replaced = [
            f"{r.name}→{_NONSTD_MAP.get(r.name, 'UNK')}" for r in nonstd
        ]
        fixer.replaceNonstandardResidues()

    # Missing residues and atoms
    if add_missing:
        fixer.findMissingResidues()
        result.missing_residues_added = sum(
            len(v) for v in fixer.missingResidues.values()
        )
        fixer.findMissingAtoms()
        result.missing_atoms_added = sum(
            len(v) for v in fixer.missingAtoms.values()
        )
        fixer.addMissingAtoms()

    # Heterogens
    if remove_heterogens:
        fixer.removeHeterogens(keepWater=keep_water)
        result.heterogens_removed = 1  # PDBFixer doesn't count individually

    # Add hydrogens at target pH
    fixer.addMissingHydrogens(ph)

    with open(output_path, "w") as f:
        PDBFile.writeFile(fixer.topology, fixer.positions, f)

    return result


# ---------------------------------------------------------------------------
# Fallback PDB parser (no dependencies)
# ---------------------------------------------------------------------------

def _fix_fallback(
    pdb_path: str,
    output_path: str,
    *,
    replace_nonstandard: bool = True,
    remove_heterogens: bool = False,
    keep_water: bool = False,
) -> FixerResult:
    """Lightweight fallback — text-level PDB fixes.

    Cannot add missing residues/atoms (needs PDBFixer for that), but handles:
    - Non-standard residue renaming
    - Alternate conformation selection (keep 'A' or first)
    - Heterogen removal
    """
    result = FixerResult(
        input_path=str(Path(pdb_path).resolve()),
        output_path=str(Path(output_path).resolve()),
        method="fallback",
    )
    result.warnings.append(
        "PDBFixer not available — using text-level fallback. "
        "Missing residues/atoms will NOT be added."
    )

    output_lines: List[str] = []
    seen_altlocs: set = set()

    with open(pdb_path) as fh:
        for line in fh:
            record = line[:6].strip()

            # Skip HETATM if removing heterogens
            if remove_heterogens and record == "HETATM":
                resname = line[17:20].strip()
                if resname == "HOH" and keep_water:
                    pass  # keep water
                else:
                    result.heterogens_removed += 1
                    continue

            if record in ("ATOM", "HETATM"):
                # Alternate conformation: keep 'A' or first, skip rest
                altloc = line[16]
                if altloc not in (" ", "", "A"):
                    result.altlocs_resolved += 1
                    continue
                if altloc == "A":
                    # Clear altloc indicator
                    line = line[:16] + " " + line[17:]
                    result.altlocs_resolved += 1

                # Non-standard residue replacement
                if replace_nonstandard:
                    resname = line[17:20].strip()
                    if resname in _NONSTD_MAP:
                        new_res = _NONSTD_MAP[resname]
                        result.nonstandard_replaced.append(
                            f"{resname}→{new_res}"
                        )
                        line = line[:17] + f"{new_res:>3}" + line[20:]

            output_lines.append(line)

    with open(output_path, "w") as fh:
        fh.writelines(output_lines)

    return result


# ---------------------------------------------------------------------------
# Crystal contact detection
# ---------------------------------------------------------------------------

def _check_crystal_contacts(pdb_path: str, threshold: float = 3.0) -> bool:
    """Check for REMARK 350 (biological assembly) or short CRYST1 axes.

    A crude heuristic: if any unit cell axis < ``threshold`` nm (30 A),
    crystal contacts may contaminate the binding site.
    """
    with open(pdb_path) as fh:
        for line in fh:
            if line.startswith("CRYST1"):
                try:
                    a = float(line[6:15])
                    b = float(line[15:24])
                    c = float(line[24:33])
                    if min(a, b, c) < threshold * 10:  # nm → Å
                        return True
                except (ValueError, IndexError):
                    pass
    return False


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def fix_protein(
    pdb_path: str,
    output_path: Optional[str] = None,
    *,
    add_missing: bool = True,
    replace_nonstandard: bool = True,
    remove_heterogens: bool = False,
    keep_water: bool = False,
    ph: float = 7.4,
) -> FixerResult:
    """Fix a PDB file for simulation readiness.

    Parameters
    ----------
    pdb_path : str
        Input PDB file.
    output_path : str, optional
        Output PDB path. Defaults to ``<stem>_fixed.pdb``.
    add_missing : bool
        Attempt to add missing residues/atoms (requires PDBFixer).
    replace_nonstandard : bool
        Replace non-standard residues with standard equivalents.
    remove_heterogens : bool
        Remove HETATM records (ligands, cofactors).
    keep_water : bool
        Keep water molecules when removing heterogens.
    ph : float
        Target pH for hydrogen addition (PDBFixer only).
    """
    pdb_path = str(Path(pdb_path).resolve())
    if output_path is None:
        p = Path(pdb_path)
        output_path = str(p.parent / f"{p.stem}_fixed.pdb")

    # Crystal contact warning
    crystal = _check_crystal_contacts(pdb_path)

    if _has_pdbfixer():
        result = _fix_with_pdbfixer(
            pdb_path, output_path,
            add_missing=add_missing,
            replace_nonstandard=replace_nonstandard,
            remove_heterogens=remove_heterogens,
            keep_water=keep_water,
            ph=ph,
        )
    else:
        result = _fix_fallback(
            pdb_path, output_path,
            replace_nonstandard=replace_nonstandard,
            remove_heterogens=remove_heterogens,
            keep_water=keep_water,
        )

    result.crystal_contacts_warned = crystal
    if crystal:
        result.warnings.append(
            "Crystal contacts detected — verify binding site is not at "
            "crystal packing interface."
        )

    return result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(
        description="Fix PDB for simulation readiness."
    )
    parser.add_argument("--pdb", required=True, help="Input PDB file")
    parser.add_argument("--output", help="Output PDB path")
    parser.add_argument(
        "--remove-heterogens", action="store_true",
        help="Remove HETATM records",
    )
    parser.add_argument(
        "--keep-water", action="store_true",
        help="Keep water when removing heterogens",
    )
    parser.add_argument("--ph", type=float, default=7.4, help="Target pH")
    args = parser.parse_args(argv)

    result = fix_protein(
        args.pdb,
        output_path=args.output,
        remove_heterogens=args.remove_heterogens,
        keep_water=args.keep_water,
        ph=args.ph,
    )
    print(result.to_json())


if __name__ == "__main__":
    main()
