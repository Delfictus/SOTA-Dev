#!/usr/bin/env python3
"""Superpose 6P8Y (or any holo) onto a reference apo PDB and rewrite the cache.

For each chain in the holo file matching the reference chain (default A),
collect common CA atoms by author resnum, run Bio.PDB.Superimposer (Kabsch),
apply the rotation+translation to ALL holo atoms (including HETATM/ligand),
and write the aligned PDB to the same path. Sidecar regeneration is left
to the caller (re-run prism-ground-truth.py).

usage:
    align_6p8y_to_6oim.py REF_PDB HOLO_PDB OUT_PDB [--chain A]
"""
import argparse
import sys
from pathlib import Path

import numpy as np
from Bio.PDB import PDBParser, PDBIO, Superimposer


def collect_ca_by_resnum(structure, chain_id):
    out = {}
    for model in structure:
        for chain in model:
            if chain.id != chain_id:
                continue
            for residue in chain:
                if residue.id[0] != " ":
                    continue
                if "CA" in residue:
                    out[residue.id[1]] = residue["CA"]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ref_pdb")
    ap.add_argument("holo_pdb")
    ap.add_argument("out_pdb")
    ap.add_argument("--chain", default="A")
    args = ap.parse_args()

    parser = PDBParser(QUIET=True)
    ref_struct = parser.get_structure("ref", args.ref_pdb)
    holo_struct = parser.get_structure("holo", args.holo_pdb)

    ref_ca = collect_ca_by_resnum(ref_struct, args.chain)
    holo_ca = collect_ca_by_resnum(holo_struct, args.chain)
    common = sorted(set(ref_ca) & set(holo_ca))

    if len(common) < 20:
        print(f"FAIL: only {len(common)} common CA pairs (need >=20)", file=sys.stderr)
        return 2

    ref_atoms = [ref_ca[r] for r in common]
    holo_atoms = [holo_ca[r] for r in common]

    sup = Superimposer()
    sup.set_atoms(ref_atoms, holo_atoms)
    rms_pre = float(np.linalg.norm(
        np.array([a.coord for a in ref_atoms]) -
        np.array([a.coord for a in holo_atoms])
    ) / np.sqrt(len(common)))

    # apply transform to ENTIRE holo structure (all atoms, all residues)
    sup.apply(holo_struct.get_atoms())

    print(f"common CA pairs:   {len(common)}")
    print(f"rmsd pre-align:    {rms_pre:.3f} A  (rough)")
    print(f"rmsd post-align:   {sup.rms:.3f} A")

    io = PDBIO()
    io.set_structure(holo_struct)
    io.save(args.out_pdb)
    print(f"wrote: {args.out_pdb}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
