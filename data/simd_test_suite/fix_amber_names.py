#!/usr/bin/env python3
"""
Fix residue names for AMBER compatibility.

Issues to fix:
1. HIE with HD1 atom -> should be HID (delta-protonated)
2. Disulfide bonds: CYS -> CYX, remove SG hydrogens
3. Terminal residue naming (NTHR -> THR, etc.)
"""

import os
import re
import math
from pathlib import Path


def distance(p1, p2):
    """Calculate 3D distance between two points."""
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(p1, p2)))


def fix_pdb_for_amber(input_path: Path, output_path: Path):
    """Fix residue names for AMBER tleap compatibility."""

    lines = input_path.read_text().splitlines()

    # Track atoms and residues
    atom_coords = {}  # atom_key -> (x, y, z)
    residue_atoms = {}  # (chain, resid) -> {"atoms": set, "resname": str}
    sg_atoms = []  # List of (chain, resid, x, y, z) for cysteine SG atoms

    # First pass: collect atom info
    for line in lines:
        if line.startswith("ATOM") or line.startswith("HETATM"):
            atom_name = line[12:16].strip()
            chain = line[21]
            resid = line[22:26].strip()
            resname = line[17:20].strip()

            try:
                x = float(line[30:38])
                y = float(line[38:46])
                z = float(line[46:54])
            except:
                continue

            key = (chain, resid)
            if key not in residue_atoms:
                residue_atoms[key] = {"atoms": set(), "resname": resname}
            residue_atoms[key]["atoms"].add(atom_name)
            atom_coords[(chain, resid, atom_name)] = (x, y, z)

            # Track CYS SG atoms for disulfide detection
            if atom_name == "SG" and resname in ("CYS", "CYX"):
                sg_atoms.append((chain, resid, x, y, z))

    # Detect disulfide bonds (SG-SG distance < 2.5Å)
    disulfide_cys = set()  # (chain, resid) pairs that form disulfides
    for i, (c1, r1, x1, y1, z1) in enumerate(sg_atoms):
        for c2, r2, x2, y2, z2 in sg_atoms[i + 1:]:
            dist = distance((x1, y1, z1), (x2, y2, z2))
            if dist < 2.5:  # Typical S-S bond ~2.05Å
                disulfide_cys.add((c1, r1))
                disulfide_cys.add((c2, r2))
                print(f"    Disulfide bond: {c1}:{r1} - {c2}:{r2} ({dist:.2f}Å)")

    # Determine correct histidine naming based on atoms present
    his_mapping = {}
    for key, info in residue_atoms.items():
        resname = info["resname"]
        atoms = info["atoms"]

        # Handle histidine protonation
        if "HIS" in resname or "HIE" in resname or "HID" in resname or "HIP" in resname:
            has_hd1 = "HD1" in atoms
            has_he2 = "HE2" in atoms

            if has_hd1 and has_he2:
                his_mapping[key] = "HIP"
            elif has_hd1:
                his_mapping[key] = "HID"
            elif has_he2:
                his_mapping[key] = "HIE"
            else:
                his_mapping[key] = "HIS"

    # Second pass: apply fixes
    fixed_lines = []
    atoms_to_remove = set()  # (chain, resid, atom_name) to remove

    # Mark HG atoms on disulfide cysteines for removal
    for chain, resid in disulfide_cys:
        atoms_to_remove.add((chain, resid, "HG"))  # Remove SG hydrogen

    for line in lines:
        if line.startswith("ATOM") or line.startswith("HETATM"):
            atom_name = line[12:16].strip()
            chain = line[21]
            resid = line[22:26].strip()
            resname = line[17:20].strip()
            key = (chain, resid)

            # Skip atoms marked for removal
            if (chain, resid, atom_name) in atoms_to_remove:
                continue

            # Fix disulfide CYS -> CYX
            if key in disulfide_cys and resname == "CYS":
                line = line[:17] + "CYX" + line[20:]

            # Fix histidine naming
            if key in his_mapping:
                new_resname = his_mapping[key]
                line = line[:17] + f"{new_resname:>3}" + line[20:]

            # Fix N-terminal residue naming
            # OpenMM uses NXXX format, AMBER uses XXX with terminal patches
            if len(resname) == 4 and resname.startswith("N"):
                actual_resname = resname[1:]
                line = line[:17] + f"{actual_resname:>3}" + line[20:]

            # Fix C-terminal residue naming
            if len(resname) == 4 and resname.startswith("C") and resname not in ("CYS", "CYX"):
                actual_resname = resname[1:]
                line = line[:17] + f"{actual_resname:>3}" + line[20:]

            # Fix N-terminal hydrogen naming: H -> H1 for first residue
            # AMBER expects H1, H2, H3 for N-terminal NH3+
            if resid == "1" and atom_name == "H":
                # Check if this is the backbone H (bonded to N)
                # pdbfixer puts it as just "H", AMBER wants "H1"
                line = line[:12] + " H1 " + line[16:]

        fixed_lines.append(line)

    output_path.write_text("\n".join(fixed_lines) + "\n")

    return len(his_mapping), len(disulfide_cys) // 2


def main():
    sanitized_dir = Path("sanitized")

    structures = ["6M0J", "7WHH", "7K45", "6WPS", "6W41", "8SGU", "7JJI"]

    print("Fixing residue names for AMBER compatibility...")

    for pdb in structures:
        input_path = sanitized_dir / f"{pdb}_fixed.pdb"
        output_path = sanitized_dir / f"{pdb}_amber.pdb"

        if not input_path.exists():
            print(f"SKIP {pdb}: {input_path} not found")
            continue

        print(f"  [{pdb}]")
        n_his, n_ss = fix_pdb_for_amber(input_path, output_path)
        print(f"    Fixed {n_his} histidines, {n_ss} disulfide bonds -> {output_path.name}")


if __name__ == "__main__":
    main()
