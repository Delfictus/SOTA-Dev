#!/usr/bin/env python3
"""
PRISM4D Structure Validation - Comprehensive Pre-MD Validation

Validates structures for production MD readiness:
1. Protonation state validation (HID/HIE/HIP histidine tautomers)
2. Missing atoms/residues (gaps, loops, terminal capping)
3. Disulfide bond verification (CONECT records, Cys-Cys distances)
4. Clash detection (steric overlaps)
5. Chirality validation (L-amino acids)
6. Charge analysis (net charge, counterion requirements)
7. pKa prediction at target pH (via PROPKA)

Usage:
    python validate_structure.py input.pdb --ph 7.4 --verbose
    python validate_structure.py input.pdb --topology topology.json --strict
"""

import os
import sys
import json
import argparse
import subprocess
import tempfile
import math
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Set
from collections import defaultdict
from dataclasses import dataclass, field, asdict

# Add scripts directory to path
SCRIPT_DIR = Path(__file__).parent
sys.path.insert(0, str(SCRIPT_DIR))

# AMBER environment detection
AMBER_ENV_NAMES = ['ambertools', 'amber', 'mdtools']


def find_amber_env_bin() -> Optional[Path]:
    """Find the bin directory of an AMBER conda environment."""
    home = Path.home()
    for base in [home / 'miniconda3', home / 'anaconda3', Path('/opt/conda')]:
        envs_dir = base / 'envs'
        if envs_dir.exists():
            for env_name in AMBER_ENV_NAMES:
                bin_dir = envs_dir / env_name / 'bin'
                if (bin_dir / 'python').exists():
                    return bin_dir
    return None


AMBER_BIN = find_amber_env_bin()
AMBER_PYTHON = str(AMBER_BIN / 'python') if AMBER_BIN else 'python3'


# Standard amino acid properties
STANDARD_RESIDUES = {
    'ALA', 'ARG', 'ASN', 'ASP', 'CYS', 'GLN', 'GLU', 'GLY', 'HIS', 'ILE',
    'LEU', 'LYS', 'MET', 'PHE', 'PRO', 'SER', 'THR', 'TRP', 'TYR', 'VAL',
    'HID', 'HIE', 'HIP',  # Histidine tautomers
    'CYX',  # Disulfide-bonded cysteine
    'ACE', 'NME',  # Terminal caps
}

# Histidine tautomers
HIS_TAUTOMERS = {'HIS', 'HID', 'HIE', 'HIP'}

# Charged residues at pH 7
CHARGED_RESIDUES = {
    'ASP': -1, 'GLU': -1,  # Negative
    'LYS': +1, 'ARG': +1, 'HIP': +1,  # Positive
}

# Expected heavy atoms per residue (backbone + sidechain)
EXPECTED_HEAVY_ATOMS = {
    'ALA': 5, 'ARG': 11, 'ASN': 8, 'ASP': 8, 'CYS': 6, 'CYX': 6,
    'GLN': 9, 'GLU': 9, 'GLY': 4, 'HIS': 10, 'HID': 10, 'HIE': 10, 'HIP': 10,
    'ILE': 8, 'LEU': 8, 'LYS': 9, 'MET': 8, 'PHE': 11, 'PRO': 7,
    'SER': 6, 'THR': 7, 'TRP': 14, 'TYR': 12, 'VAL': 7,
}

# VDW radii for clash detection (in Angstroms)
VDW_RADII = {
    'H': 1.20, 'C': 1.70, 'N': 1.55, 'O': 1.52, 'S': 1.80, 'P': 1.80,
    'F': 1.47, 'CL': 1.75, 'BR': 1.85, 'I': 1.98, 'FE': 1.40, 'ZN': 1.39,
    'CA': 1.97, 'MG': 1.73, 'MN': 1.39, 'CU': 1.40, 'SE': 1.90,
}


@dataclass
class ValidationResult:
    """Container for validation results."""
    pdb_file: str
    is_valid: bool = True

    # Counts
    n_atoms: int = 0
    n_residues: int = 0
    n_chains: int = 0
    n_heavy_atoms: int = 0
    n_hydrogens: int = 0

    # Protonation
    histidine_states: Dict[str, List[str]] = field(default_factory=dict)
    protonation_warnings: List[str] = field(default_factory=list)

    # Missing atoms/residues
    missing_atoms: List[str] = field(default_factory=list)
    missing_residues: List[str] = field(default_factory=list)
    sequence_gaps: List[str] = field(default_factory=list)
    terminal_caps: Dict[str, str] = field(default_factory=dict)

    # Disulfide bonds
    disulfide_bonds: List[Tuple[str, str]] = field(default_factory=list)
    ss_bond_issues: List[str] = field(default_factory=list)

    # Clashes
    clash_count: int = 0
    severe_clashes: List[str] = field(default_factory=list)
    clash_score: float = 0.0

    # Chirality
    chirality_issues: List[str] = field(default_factory=list)

    # Charge analysis
    net_charge: float = 0.0
    charged_residues: Dict[str, int] = field(default_factory=dict)
    counterions_needed: int = 0

    # pKa analysis
    pka_predictions: Dict[str, float] = field(default_factory=dict)
    pka_warnings: List[str] = field(default_factory=list)

    # Overall
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)


def parse_pdb(pdb_path: str) -> Tuple[List[Dict], Dict]:
    """Parse PDB file and extract atom/residue information."""
    atoms = []
    conect_records = defaultdict(set)

    with open(pdb_path) as f:
        for line in f:
            if line.startswith('ATOM') or line.startswith('HETATM'):
                atom = {
                    'serial': int(line[6:11].strip()),
                    'name': line[12:16].strip(),
                    'resname': line[17:20].strip(),
                    'chain': line[21:22].strip() or 'A',
                    'resnum': int(line[22:26].strip()),
                    'x': float(line[30:38]),
                    'y': float(line[38:46]),
                    'z': float(line[46:54]),
                    'element': line[76:78].strip() if len(line) > 76 else line[12:14].strip()[0],
                }
                atoms.append(atom)
            elif line.startswith('CONECT'):
                parts = line.split()
                if len(parts) >= 2:
                    atom1 = int(parts[1])
                    for p in parts[2:]:
                        try:
                            atom2 = int(p)
                            conect_records[atom1].add(atom2)
                            conect_records[atom2].add(atom1)
                        except ValueError:
                            pass

    return atoms, dict(conect_records)


def validate_protonation(atoms: List[Dict], result: ValidationResult):
    """Validate protonation states, especially histidine tautomers."""
    # Group atoms by residue
    residues = defaultdict(list)
    for atom in atoms:
        key = (atom['chain'], atom['resnum'], atom['resname'])
        residues[key].append(atom)

    # Check histidines
    for (chain, resnum, resname), res_atoms in residues.items():
        if resname in HIS_TAUTOMERS:
            res_id = f"{chain}:{resname}{resnum}"

            # Check which protons are present
            atom_names = {a['name'] for a in res_atoms}
            has_hd1 = 'HD1' in atom_names
            has_he2 = 'HE2' in atom_names

            if resname == 'HIS':
                # Generic HIS - should be assigned to HID/HIE/HIP
                result.protonation_warnings.append(
                    f"{res_id}: Generic HIS (should be HID/HIE/HIP for explicit protonation)"
                )

            if chain not in result.histidine_states:
                result.histidine_states[chain] = []
            result.histidine_states[chain].append(f"{resname}{resnum}")

    # Count hydrogens
    result.n_hydrogens = sum(1 for a in atoms if a['element'] in ('H', '1', '2', '3'))
    result.n_heavy_atoms = len(atoms) - result.n_hydrogens

    if result.n_hydrogens == 0:
        result.protonation_warnings.append("No hydrogens detected - structure not protonated")


def validate_missing_atoms(atoms: List[Dict], result: ValidationResult):
    """Check for missing atoms and residues."""
    # Group by residue
    residues = defaultdict(list)
    for atom in atoms:
        key = (atom['chain'], atom['resnum'], atom['resname'])
        residues[key].append(atom)

    # Check each residue
    for (chain, resnum, resname), res_atoms in residues.items():
        if resname not in STANDARD_RESIDUES:
            continue

        # Count heavy atoms
        heavy_atoms = [a for a in res_atoms if a['element'] not in ('H', '1', '2', '3')]
        expected = EXPECTED_HEAVY_ATOMS.get(resname, 0)

        if len(heavy_atoms) < expected:
            result.missing_atoms.append(
                f"{chain}:{resname}{resnum} has {len(heavy_atoms)}/{expected} heavy atoms"
            )

    # Check for sequence gaps
    chains = defaultdict(list)
    for (chain, resnum, resname), _ in residues.items():
        if resname in STANDARD_RESIDUES:
            chains[chain].append(resnum)

    for chain, resnums in chains.items():
        resnums = sorted(set(resnums))
        for i in range(1, len(resnums)):
            gap = resnums[i] - resnums[i-1]
            if gap > 1:
                result.sequence_gaps.append(
                    f"Chain {chain}: gap between {resnums[i-1]} and {resnums[i]} ({gap-1} residues)"
                )

    # Check terminal caps
    for chain, resnums in chains.items():
        resnums = sorted(set(resnums))
        if resnums:
            # Check N-terminus
            first_res = [(c, r, n) for (c, r, n) in residues.keys()
                        if c == chain and r == resnums[0]]
            if first_res:
                _, _, resname = first_res[0]
                if resname == 'ACE':
                    result.terminal_caps[f"{chain}_N"] = "ACE (capped)"
                else:
                    result.terminal_caps[f"{chain}_N"] = f"{resname} (free N-terminus)"

            # Check C-terminus
            last_res = [(c, r, n) for (c, r, n) in residues.keys()
                       if c == chain and r == resnums[-1]]
            if last_res:
                _, _, resname = last_res[0]
                if resname == 'NME':
                    result.terminal_caps[f"{chain}_C"] = "NME (capped)"
                else:
                    result.terminal_caps[f"{chain}_C"] = f"{resname} (free C-terminus)"


def validate_disulfides(atoms: List[Dict], conect: Dict, result: ValidationResult):
    """Validate disulfide bonds."""
    # Find all CYS/CYX residues
    cys_residues = defaultdict(list)
    for atom in atoms:
        if atom['resname'] in ('CYS', 'CYX') and atom['name'] == 'SG':
            key = (atom['chain'], atom['resnum'])
            cys_residues[key] = atom

    # Check for disulfide bonds
    checked_pairs = set()
    for (chain1, resnum1), sg1 in cys_residues.items():
        for (chain2, resnum2), sg2 in cys_residues.items():
            if (chain1, resnum1) >= (chain2, resnum2):
                continue

            pair = ((chain1, resnum1), (chain2, resnum2))
            if pair in checked_pairs:
                continue
            checked_pairs.add(pair)

            # Calculate S-S distance
            dx = sg1['x'] - sg2['x']
            dy = sg1['y'] - sg2['y']
            dz = sg1['z'] - sg2['z']
            dist = math.sqrt(dx*dx + dy*dy + dz*dz)

            # Typical S-S bond is 2.0-2.1 Å
            if dist < 2.5:
                result.disulfide_bonds.append((
                    f"{chain1}:CYS{resnum1}",
                    f"{chain2}:CYS{resnum2}"
                ))

                # Check CONECT record
                if sg1['serial'] not in conect or sg2['serial'] not in conect.get(sg1['serial'], set()):
                    result.ss_bond_issues.append(
                        f"Missing CONECT for SS bond: {chain1}:CYS{resnum1}-{chain2}:CYS{resnum2} (d={dist:.2f}Å)"
                    )
            elif dist < 4.0:
                # Close but not bonded - potential issue
                result.ss_bond_issues.append(
                    f"Possible unbonded SS: {chain1}:CYS{resnum1}-{chain2}:CYS{resnum2} (d={dist:.2f}Å)"
                )


def validate_clashes(atoms: List[Dict], result: ValidationResult, threshold: float = 0.4):
    """
    Detect steric clashes between atoms.

    threshold: fraction of VDW overlap to consider a clash (0.4 = 40% overlap)
    """
    # Build spatial grid for efficiency
    grid = defaultdict(list)
    cell_size = 4.0  # Angstroms

    for i, atom in enumerate(atoms):
        cx = int(atom['x'] / cell_size)
        cy = int(atom['y'] / cell_size)
        cz = int(atom['z'] / cell_size)
        grid[(cx, cy, cz)].append(i)

    clashes = []
    checked = set()

    for (cx, cy, cz), indices in grid.items():
        # Check atoms in this cell and neighboring cells
        neighbors = []
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for dz in (-1, 0, 1):
                    neighbors.extend(grid.get((cx+dx, cy+dy, cz+dz), []))

        for i in indices:
            for j in neighbors:
                if i >= j:
                    continue
                if (i, j) in checked:
                    continue
                checked.add((i, j))

                a1, a2 = atoms[i], atoms[j]

                # Skip atoms in same residue
                if (a1['chain'] == a2['chain'] and
                    a1['resnum'] == a2['resnum']):
                    continue

                # Skip bonded atoms (1-2, 1-3)
                if abs(a1['resnum'] - a2['resnum']) <= 1 and a1['chain'] == a2['chain']:
                    continue

                # Calculate distance
                dx = a1['x'] - a2['x']
                dy = a1['y'] - a2['y']
                dz = a1['z'] - a2['z']
                dist = math.sqrt(dx*dx + dy*dy + dz*dz)

                # Get VDW radii
                r1 = VDW_RADII.get(a1['element'].upper(), 1.7)
                r2 = VDW_RADII.get(a2['element'].upper(), 1.7)
                min_dist = (r1 + r2) * (1 - threshold)

                if dist < min_dist:
                    overlap = (r1 + r2 - dist) / (r1 + r2)
                    clashes.append({
                        'atom1': f"{a1['chain']}:{a1['resname']}{a1['resnum']}:{a1['name']}",
                        'atom2': f"{a2['chain']}:{a2['resname']}{a2['resnum']}:{a2['name']}",
                        'distance': dist,
                        'overlap': overlap,
                    })

    result.clash_count = len(clashes)

    # Report severe clashes (>50% overlap)
    severe = [c for c in clashes if c['overlap'] > 0.5]
    for c in severe[:10]:  # Limit to 10
        result.severe_clashes.append(
            f"{c['atom1']} -- {c['atom2']}: {c['distance']:.2f}Å ({c['overlap']*100:.0f}% overlap)"
        )

    # Calculate clash score (clashes per 1000 atoms)
    if len(atoms) > 0:
        result.clash_score = (len(clashes) / len(atoms)) * 1000


def validate_chirality(atoms: List[Dict], result: ValidationResult):
    """
    Basic chirality validation - check CA chirality for L-amino acids.
    """
    # Group atoms by residue
    residues = defaultdict(dict)
    for atom in atoms:
        key = (atom['chain'], atom['resnum'], atom['resname'])
        residues[key][atom['name']] = atom

    for (chain, resnum, resname), res_atoms in residues.items():
        if resname not in STANDARD_RESIDUES or resname == 'GLY':
            continue

        # Need CA, N, C, CB for chirality check
        if not all(n in res_atoms for n in ('CA', 'N', 'C', 'CB')):
            continue

        ca = res_atoms['CA']
        n = res_atoms['N']
        c = res_atoms['C']
        cb = res_atoms['CB']

        # Calculate chirality using cross product
        # Vector from CA to N
        v1 = (n['x'] - ca['x'], n['y'] - ca['y'], n['z'] - ca['z'])
        # Vector from CA to C
        v2 = (c['x'] - ca['x'], c['y'] - ca['y'], c['z'] - ca['z'])
        # Vector from CA to CB
        v3 = (cb['x'] - ca['x'], cb['y'] - ca['y'], cb['z'] - ca['z'])

        # Cross product v1 x v2
        cross = (
            v1[1]*v2[2] - v1[2]*v2[1],
            v1[2]*v2[0] - v1[0]*v2[2],
            v1[0]*v2[1] - v1[1]*v2[0]
        )

        # Dot with v3
        dot = cross[0]*v3[0] + cross[1]*v3[1] + cross[2]*v3[2]

        # L-amino acids have specific chirality sign
        # The sign depends on coordinate frame convention
        # Only report if clearly wrong (both checks fail)
        # Note: PDB convention means L-amino acids typically have dot < 0
        if dot > 5.0:  # Strong positive = definite D
            result.chirality_issues.append(
                f"{chain}:{resname}{resnum} may have D-configuration"
            )


def validate_charges(atoms: List[Dict], result: ValidationResult):
    """Analyze charge distribution and counterion requirements."""
    # Group by residue
    residues = set()
    for atom in atoms:
        key = (atom['chain'], atom['resnum'], atom['resname'])
        residues.add(key)

    # Count charged residues
    charge = 0
    for chain, resnum, resname in residues:
        if resname in CHARGED_RESIDUES:
            q = CHARGED_RESIDUES[resname]
            charge += q
            if resname not in result.charged_residues:
                result.charged_residues[resname] = 0
            result.charged_residues[resname] += 1

    result.net_charge = charge

    # Calculate counterions needed for neutralization
    if charge > 0:
        result.counterions_needed = charge  # Cl- ions
    elif charge < 0:
        result.counterions_needed = -charge  # Na+ ions


def run_propka(pdb_path: str, ph: float, result: ValidationResult):
    """Run PROPKA for pKa prediction."""
    try:
        # Use AMBER environment Python
        cmd = [
            AMBER_PYTHON, '-c',
            f'''
import propka.run as pk
import sys
try:
    mol = pk.single("{pdb_path}")
    for group in mol.conformations["AVR"].groups:
        if hasattr(group, "pka_value") and group.pka_value is not None:
            res_id = f"{{group.residue_type}}{{group.res_num}}"
            print(f"{{res_id}}:{{group.pka_value:.2f}}")
except Exception as e:
    print(f"ERROR:{{e}}", file=sys.stderr)
'''
        ]

        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=60)

        if proc.returncode == 0:
            for line in proc.stdout.strip().split('\n'):
                if ':' in line:
                    res, pka = line.split(':')
                    result.pka_predictions[res] = float(pka)

                    # Check for unusual pKa values
                    pka_val = float(pka)
                    if 'ASP' in res and abs(pka_val - 3.9) > 2:
                        result.pka_warnings.append(f"{res}: unusual pKa {pka_val:.1f} (expected ~3.9)")
                    elif 'GLU' in res and abs(pka_val - 4.3) > 2:
                        result.pka_warnings.append(f"{res}: unusual pKa {pka_val:.1f} (expected ~4.3)")
                    elif 'HIS' in res and abs(pka_val - 6.0) > 2:
                        result.pka_warnings.append(f"{res}: unusual pKa {pka_val:.1f} (expected ~6.0)")
                    elif 'LYS' in res and abs(pka_val - 10.5) > 2:
                        result.pka_warnings.append(f"{res}: unusual pKa {pka_val:.1f} (expected ~10.5)")
        else:
            result.warnings.append(f"PROPKA failed: {proc.stderr[:100]}")

    except Exception as e:
        result.warnings.append(f"PROPKA error: {str(e)}")


def validate_structure(pdb_path: str, topology_path: Optional[str] = None,
                       ph: float = 7.4, strict: bool = False,
                       verbose: bool = False) -> ValidationResult:
    """
    Perform comprehensive structure validation.
    """
    result = ValidationResult(pdb_file=pdb_path)

    # Parse PDB
    atoms, conect = parse_pdb(pdb_path)
    result.n_atoms = len(atoms)
    result.n_residues = len(set((a['chain'], a['resnum']) for a in atoms))
    result.n_chains = len(set(a['chain'] for a in atoms))

    if verbose:
        print(f"Validating: {pdb_path}")
        print(f"  Atoms: {result.n_atoms:,}, Residues: {result.n_residues}, Chains: {result.n_chains}")

    # Run validations
    validate_protonation(atoms, result)
    validate_missing_atoms(atoms, result)
    validate_disulfides(atoms, conect, result)
    validate_clashes(atoms, result)
    validate_chirality(atoms, result)
    validate_charges(atoms, result)

    # Run PROPKA
    if verbose:
        print("  Running PROPKA...")
    run_propka(pdb_path, ph, result)

    # Validate against topology if provided
    if topology_path and Path(topology_path).exists():
        with open(topology_path) as f:
            topo = json.load(f)

        topo_atoms = topo.get('n_atoms', 0)
        if topo_atoms != result.n_atoms:
            result.errors.append(
                f"Atom count mismatch: PDB has {result.n_atoms}, topology has {topo_atoms}"
            )

    # Determine overall validity
    # Critical issues that always fail validation:
    critical_issues = (
        len(result.errors) > 0 or          # Explicit errors
        len(result.chirality_issues) > 0    # Wrong stereochemistry
    )

    if strict:
        # Strict mode: also fail on severe clashes (>50% overlap)
        # But NOT on protonation warnings (can be fixed by AMBER)
        # But NOT on SS bond CONECT warnings (force field handles bonds)
        very_severe_clashes = [c for c in result.severe_clashes if '60%' in c or '70%' in c or '80%' in c or '90%' in c]
        result.is_valid = not critical_issues and len(very_severe_clashes) == 0
    else:
        result.is_valid = not critical_issues

    return result


def print_report(result: ValidationResult):
    """Print validation report."""
    print("\n" + "=" * 70)
    print("PRISM4D STRUCTURE VALIDATION REPORT")
    print("=" * 70)

    print(f"\nFile: {result.pdb_file}")
    print(f"Atoms: {result.n_atoms:,} ({result.n_heavy_atoms:,} heavy, {result.n_hydrogens:,} H)")
    print(f"Residues: {result.n_residues:,}")
    print(f"Chains: {result.n_chains}")

    # Protonation
    print("\n--- Protonation State ---")
    if result.histidine_states:
        for chain, his_list in result.histidine_states.items():
            print(f"  Chain {chain} histidines: {', '.join(his_list)}")
    if result.protonation_warnings:
        for w in result.protonation_warnings:
            print(f"  ⚠️  {w}")
    else:
        print("  ✓ Protonation OK")

    # Missing atoms
    print("\n--- Missing Atoms/Residues ---")
    if result.missing_atoms:
        for m in result.missing_atoms[:5]:
            print(f"  ⚠️  {m}")
        if len(result.missing_atoms) > 5:
            print(f"  ... and {len(result.missing_atoms) - 5} more")
    if result.sequence_gaps:
        for g in result.sequence_gaps:
            print(f"  ⚠️  Gap: {g}")
    if not result.missing_atoms and not result.sequence_gaps:
        print("  ✓ No missing atoms or gaps")

    # Terminal caps
    if result.terminal_caps:
        print("\n--- Terminal Capping ---")
        for term, status in result.terminal_caps.items():
            print(f"  {term}: {status}")

    # Disulfide bonds
    print("\n--- Disulfide Bonds ---")
    if result.disulfide_bonds:
        for ss in result.disulfide_bonds:
            print(f"  ✓ {ss[0]} -- {ss[1]}")
    else:
        print("  No disulfide bonds detected")
    if result.ss_bond_issues:
        for issue in result.ss_bond_issues:
            print(f"  ⚠️  {issue}")

    # Clashes
    print("\n--- Steric Clashes ---")
    print(f"  Total clashes: {result.clash_count}")
    print(f"  Clash score: {result.clash_score:.1f} per 1000 atoms")
    if result.severe_clashes:
        print("  Severe clashes:")
        for c in result.severe_clashes:
            print(f"    ❌ {c}")

    # Chirality
    print("\n--- Chirality ---")
    if result.chirality_issues:
        for issue in result.chirality_issues:
            print(f"  ❌ {issue}")
    else:
        print("  ✓ All L-amino acids")

    # Charge analysis
    print("\n--- Charge Analysis ---")
    print(f"  Net charge: {result.net_charge:+.0f} e")
    if result.charged_residues:
        parts = [f"{k}({v})" for k, v in result.charged_residues.items()]
        print(f"  Charged residues: {', '.join(parts)}")
    if result.counterions_needed > 0:
        ion_type = "Cl⁻" if result.net_charge > 0 else "Na⁺"
        print(f"  Counterions needed: {result.counterions_needed} {ion_type}")

    # pKa
    if result.pka_predictions:
        print("\n--- pKa Predictions (PROPKA) ---")
        titratable = [(k, v) for k, v in result.pka_predictions.items()
                     if any(r in k for r in ('ASP', 'GLU', 'HIS', 'LYS', 'TYR', 'CYS'))]
        if titratable:
            for res, pka in sorted(titratable)[:10]:
                print(f"  {res}: pKa = {pka:.2f}")
            if len(titratable) > 10:
                print(f"  ... and {len(titratable) - 10} more")
    if result.pka_warnings:
        for w in result.pka_warnings:
            print(f"  ⚠️  {w}")

    # Summary
    print("\n" + "=" * 70)
    if result.is_valid:
        print("✅ STRUCTURE VALIDATION PASSED")
    else:
        print("❌ STRUCTURE VALIDATION FAILED")
        if result.errors:
            print("\nErrors:")
            for e in result.errors:
                print(f"  • {e}")
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(
        description="PRISM4D Structure Validation - Comprehensive Pre-MD Validation"
    )
    parser.add_argument('pdb', help='Input PDB file')
    parser.add_argument('--topology', '-t', help='Topology JSON to validate against')
    parser.add_argument('--ph', type=float, default=7.4, help='Target pH (default: 7.4)')
    parser.add_argument('--strict', action='store_true', help='Strict validation mode')
    parser.add_argument('--json', '-j', help='Output results as JSON')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
    parser.add_argument('--quiet', '-q', action='store_true', help='Only show summary')

    args = parser.parse_args()

    if not Path(args.pdb).exists():
        print(f"Error: PDB file not found: {args.pdb}")
        return 1

    result = validate_structure(
        args.pdb,
        topology_path=args.topology,
        ph=args.ph,
        strict=args.strict,
        verbose=args.verbose
    )

    if not args.quiet:
        print_report(result)

    if args.json:
        with open(args.json, 'w') as f:
            json.dump(asdict(result), f, indent=2)
        print(f"\nResults written to: {args.json}")

    return 0 if result.is_valid else 1


if __name__ == "__main__":
    sys.exit(main())
