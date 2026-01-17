#!/usr/bin/env python3
"""
Convert AMBER prmtop/inpcrd files to the JSON format expected by PRISM4D.

The AMBER prmtop is a text format with FLAG sections.
The inpcrd contains coordinates.
"""

import json
import math
import re
from pathlib import Path
from typing import Any


def parse_prmtop(prmtop_path: Path) -> dict:
    """Parse AMBER prmtop file into sections."""
    content = prmtop_path.read_text()

    # Find all FLAG sections
    sections = {}
    current_flag = None
    current_format = None
    current_data = []

    for line in content.split('\n'):
        if line.startswith('%FLAG'):
            if current_flag:
                sections[current_flag] = (current_format, current_data)
            current_flag = line.split()[1]
            current_data = []
        elif line.startswith('%FORMAT'):
            # Parse format like (20a4) or (5E16.8) or (10I8)
            current_format = line.strip()
        elif current_flag and not line.startswith('%'):
            current_data.append(line)

    # Save last section
    if current_flag:
        sections[current_flag] = (current_format, current_data)

    return sections


def parse_format(fmt_str: str):
    """Parse FORTRAN format string."""
    # Examples: %FORMAT(20a4), %FORMAT(5E16.8), %FORMAT(10I8)
    match = re.search(r'\((\d+)([aEeIi])(\d+)(?:\.(\d+))?\)', fmt_str)
    if not match:
        return None, None, None
    count = int(match.group(1))
    dtype = match.group(2).upper()
    width = int(match.group(3))
    return count, dtype, width


def extract_values(lines: list[str], fmt_str: str, expected_count: int = None):
    """Extract values from prmtop data lines using format string."""
    count, dtype, width = parse_format(fmt_str)
    if count is None:
        return []

    text = ''.join(lines)
    values = []
    i = 0
    while i < len(text):
        chunk = text[i:i+width]
        if chunk.strip():
            if dtype == 'I':
                values.append(int(chunk))
            elif dtype in ('E', 'F'):
                values.append(float(chunk))
            elif dtype == 'A':
                values.append(chunk.strip())
        i += width
        if expected_count and len(values) >= expected_count:
            break

    return values


def parse_inpcrd(inpcrd_path: Path) -> list[float]:
    """Parse AMBER inpcrd coordinate file."""
    lines = inpcrd_path.read_text().strip().split('\n')

    # First line is title
    # Second line is number of atoms (and optionally time)
    n_atoms = int(lines[1].split()[0])

    # Coordinates should be 3 * n_atoms values
    # AMBER uses %12.7f format (7 decimal places)
    # Use regex to match numbers with exactly 7 decimal digits

    coords = []
    for line in lines[2:]:
        # Match pattern: optional minus, digits, decimal, exactly 7 digits
        matches = re.findall(r'-?\d+\.\d{7}', line)
        for m in matches:
            try:
                coords.append(float(m))
            except ValueError:
                pass

    return coords


def convert_prmtop_to_json(prmtop_path: Path, inpcrd_path: Path, output_path: Path):
    """Convert AMBER prmtop/inpcrd to JSON topology."""

    print(f"  Parsing {prmtop_path.name}...")
    sections = parse_prmtop(prmtop_path)

    # Extract key arrays
    n_atoms = extract_values(sections['POINTERS'][1], sections['POINTERS'][0], 1)[0]
    n_types = extract_values(sections['POINTERS'][1], sections['POINTERS'][0], 31)[1]
    n_bonds_with_h = extract_values(sections['POINTERS'][1], sections['POINTERS'][0], 31)[2]
    n_bonds_without_h = extract_values(sections['POINTERS'][1], sections['POINTERS'][0], 31)[3]
    n_angles_with_h = extract_values(sections['POINTERS'][1], sections['POINTERS'][0], 31)[4]
    n_angles_without_h = extract_values(sections['POINTERS'][1], sections['POINTERS'][0], 31)[5]
    n_dihedrals_with_h = extract_values(sections['POINTERS'][1], sections['POINTERS'][0], 31)[6]
    n_dihedrals_without_h = extract_values(sections['POINTERS'][1], sections['POINTERS'][0], 31)[7]

    n_bonds = n_bonds_with_h + n_bonds_without_h
    n_angles = n_angles_with_h + n_angles_without_h
    n_dihedrals = n_dihedrals_with_h + n_dihedrals_without_h

    print(f"    Atoms: {n_atoms}, Bonds: {n_bonds}, Angles: {n_angles}, Dihedrals: {n_dihedrals}")

    # Masses
    masses = extract_values(sections['MASS'][1], sections['MASS'][0], n_atoms)

    # Charges (in AMBER electron charge units, need to convert)
    charges_raw = extract_values(sections['CHARGE'][1], sections['CHARGE'][0], n_atoms)
    # AMBER stores charges multiplied by 18.2223 (sqrt of Coulomb constant in AMBER units)
    charges = [c / 18.2223 for c in charges_raw]

    # Atom types for LJ parameters
    atom_type_indices = extract_values(sections['ATOM_TYPE_INDEX'][1], sections['ATOM_TYPE_INDEX'][0], n_atoms)

    # LJ parameters: sigma and epsilon
    # LENNARD_JONES_ACOEF and LENNARD_JONES_BCOEF contain A and B coefficients
    # V(r) = A/r^12 - B/r^6
    # Converting to sigma/epsilon: sigma = (A/B)^(1/6), epsilon = B^2/(4A)
    ntypes = n_types
    n_lj_pairs = ntypes * (ntypes + 1) // 2

    lj_a = extract_values(sections['LENNARD_JONES_ACOEF'][1], sections['LENNARD_JONES_ACOEF'][0], n_lj_pairs)
    lj_b = extract_values(sections['LENNARD_JONES_BCOEF'][1], sections['LENNARD_JONES_BCOEF'][0], n_lj_pairs)

    # LJ index: for atom types i, j -> index = (max(i,j)-1)*max(i,j)/2 + min(i,j)
    lj_index = extract_values(sections['NONBONDED_PARM_INDEX'][1], sections['NONBONDED_PARM_INDEX'][0], ntypes*ntypes)

    # Get sigma/epsilon for each atom (using self-interaction)
    lj_params = []
    for i, type_idx in enumerate(atom_type_indices):
        # Self-interaction index
        idx = lj_index[(type_idx - 1) * ntypes + (type_idx - 1)] - 1
        if idx >= 0 and idx < len(lj_a):
            A = lj_a[idx]
            B = lj_b[idx]
            if A > 0 and B > 0:
                sigma = (A / B) ** (1.0 / 6.0)
                epsilon = B ** 2 / (4.0 * A)
            else:
                sigma = 0.0
                epsilon = 0.0
        else:
            sigma = 0.0
            epsilon = 0.0
        lj_params.append({"sigma": sigma, "epsilon": epsilon})

    # Bond parameters
    bond_force_k = extract_values(sections['BOND_FORCE_CONSTANT'][1], sections['BOND_FORCE_CONSTANT'][0])
    bond_equil = extract_values(sections['BOND_EQUIL_VALUE'][1], sections['BOND_EQUIL_VALUE'][0])

    # Bond connectivity (with H and without H)
    bonds_with_h = extract_values(sections['BONDS_INC_HYDROGEN'][1], sections['BONDS_INC_HYDROGEN'][0], n_bonds_with_h * 3)
    bonds_without_h = extract_values(sections['BONDS_WITHOUT_HYDROGEN'][1], sections['BONDS_WITHOUT_HYDROGEN'][0], n_bonds_without_h * 3)

    bonds = []
    for bond_data in [bonds_with_h, bonds_without_h]:
        for b in range(0, len(bond_data), 3):
            i = bond_data[b] // 3  # AMBER stores atom indices * 3
            j = bond_data[b + 1] // 3
            param_idx = bond_data[b + 2] - 1
            bonds.append({
                "i": i,
                "j": j,
                "r0": bond_equil[param_idx],
                "k": bond_force_k[param_idx] * 2  # AMBER stores k/2
            })

    # Angle parameters
    angle_force_k = extract_values(sections['ANGLE_FORCE_CONSTANT'][1], sections['ANGLE_FORCE_CONSTANT'][0])
    angle_equil = extract_values(sections['ANGLE_EQUIL_VALUE'][1], sections['ANGLE_EQUIL_VALUE'][0])

    angles_with_h = extract_values(sections['ANGLES_INC_HYDROGEN'][1], sections['ANGLES_INC_HYDROGEN'][0], n_angles_with_h * 4)
    angles_without_h = extract_values(sections['ANGLES_WITHOUT_HYDROGEN'][1], sections['ANGLES_WITHOUT_HYDROGEN'][0], n_angles_without_h * 4)

    angles = []
    for angle_data in [angles_with_h, angles_without_h]:
        for a in range(0, len(angle_data), 4):
            i = angle_data[a] // 3
            j = angle_data[a + 1] // 3
            k = angle_data[a + 2] // 3
            param_idx = angle_data[a + 3] - 1
            angles.append({
                "i": i,
                "j": j,
                "k_idx": k,
                "theta0": angle_equil[param_idx],
                "force_k": angle_force_k[param_idx] * 2  # AMBER stores k/2
            })

    # Dihedral parameters
    dihedral_force_k = extract_values(sections['DIHEDRAL_FORCE_CONSTANT'][1], sections['DIHEDRAL_FORCE_CONSTANT'][0])
    dihedral_period = extract_values(sections['DIHEDRAL_PERIODICITY'][1], sections['DIHEDRAL_PERIODICITY'][0])
    dihedral_phase = extract_values(sections['DIHEDRAL_PHASE'][1], sections['DIHEDRAL_PHASE'][0])

    dihedrals_with_h = extract_values(sections['DIHEDRALS_INC_HYDROGEN'][1], sections['DIHEDRALS_INC_HYDROGEN'][0], n_dihedrals_with_h * 5)
    dihedrals_without_h = extract_values(sections['DIHEDRALS_WITHOUT_HYDROGEN'][1], sections['DIHEDRALS_WITHOUT_HYDROGEN'][0], n_dihedrals_without_h * 5)

    dihedrals = []
    for dihedral_data in [dihedrals_with_h, dihedrals_without_h]:
        for d in range(0, len(dihedral_data), 5):
            i = dihedral_data[d] // 3
            j = dihedral_data[d + 1] // 3
            k = abs(dihedral_data[d + 2]) // 3  # Can be negative for impropers
            l = abs(dihedral_data[d + 3]) // 3  # Can be negative for 1-4 exclusions
            param_idx = dihedral_data[d + 4] - 1
            dihedrals.append({
                "i": i,
                "j": j,
                "k_idx": k,
                "l": l,
                "periodicity": int(abs(dihedral_period[param_idx])),
                "phase": dihedral_phase[param_idx],
                "force_k": dihedral_force_k[param_idx]
            })

    # Parse coordinates
    print(f"  Parsing {inpcrd_path.name}...")
    positions = parse_inpcrd(inpcrd_path)

    # Build exclusions from bonds, angles, dihedrals (simplified: just bonded atoms)
    exclusions = [set() for _ in range(n_atoms)]
    for bond in bonds:
        exclusions[bond["i"]].add(bond["j"])
        exclusions[bond["j"]].add(bond["i"])
    for angle in angles:
        exclusions[angle["i"]].add(angle["k_idx"])
        exclusions[angle["k_idx"]].add(angle["i"])
    for dih in dihedrals:
        exclusions[dih["i"]].add(dih["l"])
        exclusions[dih["l"]].add(dih["i"])

    # Convert exclusions to sorted lists
    exclusions = [sorted(list(ex)) for ex in exclusions]

    # Build output JSON
    output = {
        "n_atoms": n_atoms,
        "positions": positions,
        "masses": masses,
        "charges": charges,
        "lj_params": lj_params,
        "bonds": bonds,
        "angles": angles,
        "dihedrals": dihedrals,
        "exclusions": exclusions,
    }

    print(f"  Writing {output_path.name}...")
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)

    print(f"    OK: {n_atoms} atoms, {len(bonds)} bonds, {len(angles)} angles, {len(dihedrals)} dihedrals")


def main():
    topo_dir = Path("topologies")
    json_dir = Path("topologies_json")
    json_dir.mkdir(exist_ok=True)

    structures = ["6M0J", "7WHH", "7K45", "6WPS", "6W41", "8SGU", "7JJI"]

    print("=" * 60)
    print("Converting AMBER prmtop/inpcrd to JSON")
    print("=" * 60)

    for pdb in structures:
        prmtop = topo_dir / f"{pdb}.prmtop"
        inpcrd = topo_dir / f"{pdb}.inpcrd"
        output = json_dir / f"{pdb}_topology.json"

        if not prmtop.exists() or not inpcrd.exists():
            print(f"\nSKIP {pdb}: Missing prmtop or inpcrd")
            continue

        print(f"\n[{pdb}]")
        try:
            convert_prmtop_to_json(prmtop, inpcrd, output)
        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    main()
