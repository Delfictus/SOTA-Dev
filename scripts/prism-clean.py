#!/usr/bin/env python3
"""PRISM-4D PDB Cleaner.

Strips alternate conformers (keeps A-conformer only), retains only the
specified chain, removes HETATM records. Validates residue type diversity.

Usage:
    python3 scripts/prism-clean.py <raw.pdb> <clean.pdb> [chain]
"""
import os
import sys
from collections import OrderedDict


def clean_pdb(raw_path, clean_path, chain="A"):
    if not os.path.exists(raw_path):
        print(f"ERROR: {raw_path} not found", file=sys.stderr)
        return 1

    lines_out = []
    residues = OrderedDict()  # resnum -> resname
    prev_chain = None

    with open(raw_path) as f:
        for line in f:
            if not line.startswith("ATOM"):
                continue

            atom_chain = line[21].strip()
            if atom_chain != chain:
                continue

            # Alternate conformer: keep only A or blank
            altloc = line[16]
            if altloc not in (" ", "", "A"):
                continue

            # Clear the altloc field
            line = line[:16] + " " + line[17:]

            resname = line[17:20].strip()
            try:
                resnum = int(line[22:26])
            except ValueError:
                continue

            residues[resnum] = resname
            lines_out.append(line)

    if not lines_out:
        print(f"ERROR: No ATOM records for chain {chain} in {raw_path}", file=sys.stderr)
        return 1

    # Write clean PDB
    os.makedirs(os.path.dirname(os.path.abspath(clean_path)), exist_ok=True)
    with open(clean_path, "w") as f:
        for line in lines_out:
            f.write(line)
        f.write("END\n")

    # Residue inventory
    unique_types = set(residues.values())
    n_res = len(residues)
    print(f"Cleaned {raw_path} -> {clean_path}")
    print(f"  Chain: {chain}")
    print(f"  Residues: {n_res}")
    print(f"  Residue types: {len(unique_types)} ({', '.join(sorted(unique_types))})")
    print(f"  Atoms: {len(lines_out)}")

    if len(unique_types) < 15:
        print(f"ERROR: Only {len(unique_types)} residue types (need >=15). "
              f"Input may be corrupted or non-protein.", file=sys.stderr)
        return 1

    has_cys = "CYS" in unique_types
    print(f"  CYS present: {has_cys}")
    return 0


def main():
    if len(sys.argv) < 3:
        print("Usage: prism-clean.py <raw.pdb> <clean.pdb> [chain]", file=sys.stderr)
        sys.exit(1)

    raw = sys.argv[1]
    clean = sys.argv[2]
    chain = sys.argv[3] if len(sys.argv) > 3 else "A"
    sys.exit(clean_pdb(raw, clean, chain))


if __name__ == "__main__":
    main()
