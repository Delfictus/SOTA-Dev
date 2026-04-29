#!/usr/bin/env python3
"""
PRISM-4D numbering audit — single coordinate frame (post-MD) comparison.

Compares engine residue centroids (computed from topology JSON's
residue_to_atom_indices applied to post-MD PDB atom positions) vs
post-MD PDB CA atom positions. Both come from the same coordinate
frame, eliminating the pre-MD-vs-post-MD conflation that produced
156/156 BROKEN in the prior version.

Output:
  <output_csv>
    Header: # Numbering audit metadata
    Columns: engine_id, best_pdb_resseq, distance_A,
             alt_pdb_resseqs (within 1A), status

Status thresholds:
  CLEAN              — best match within 2.0A, no alternatives within 1.0A
  AMBIGUOUS          — best match within 2.0A, with >=1 alternative within 1.0A
  BROKEN_loose_match — match between 2.0 and 5.0A
  BROKEN_no_match    — no match within 5.0A
  BROKEN_no_atoms    — engine residue has no atoms in post-MD PDB
"""
import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np


def parse_pdb_atoms(pdb_path, chain_filter="A"):
    """Returns (atoms_by_idx, cas_by_resseq).

      atoms_by_idx: {atom_serial: {"resseq": int, "xyz": np.ndarray(3)}}
      cas_by_resseq: {resseq: np.ndarray(3)}
    Only ATOM records, alt-loc {blank, A}, matching chain_filter.
    """
    atoms = {}
    cas = {}
    with open(pdb_path) as f:
        for line in f:
            if not line.startswith("ATOM"):
                continue
            try:
                atom_index = int(line[6:11])
                atom_name = line[12:16].strip()
                alt_loc = line[16:17].strip()
                chain = line[21:22].strip()
                resseq = int(line[22:26])
                x = float(line[30:38])
                y = float(line[38:46])
                z = float(line[46:54])
            except (ValueError, IndexError):
                continue
            if alt_loc not in ("", "A"):
                continue
            if chain != chain_filter:
                continue
            atoms[atom_index] = {"resseq": resseq, "xyz": np.array([x, y, z])}
            if atom_name == "CA":
                cas[resseq] = np.array([x, y, z])
    return atoms, cas


def audit_numbering(topology_path, md_pdb_path, output_csv,
                    chain_id="A", clean_thresh=2.0,
                    ambiguous_thresh=1.0, broken_thresh=5.0):
    topo = json.load(open(topology_path))
    res_to_atoms = topo.get("residue_to_atom_indices", {})
    if not res_to_atoms:
        print(f"FATAL: residue_to_atom_indices empty in {topology_path}")
        print(f"  This target requires G4 producer-side population.")
        return False

    atoms_by_idx, cas_by_resseq = parse_pdb_atoms(md_pdb_path, chain_id)
    if not atoms_by_idx or not cas_by_resseq:
        print(f"FATAL: no atoms parsed from {md_pdb_path} (chain={chain_id})")
        return False

    rows = []
    summary = {"CLEAN": 0, "AMBIGUOUS": 0, "BROKEN": 0}

    for engine_id_str, atom_indices in res_to_atoms.items():
        engine_id = int(engine_id_str)
        atom_xyz = []
        for idx in atom_indices:
            if idx in atoms_by_idx:
                atom_xyz.append(atoms_by_idx[idx]["xyz"])
        if not atom_xyz:
            rows.append({
                "engine_id": engine_id, "best_pdb_resseq": "",
                "distance_A": "", "alt_pdb_resseqs": "",
                "status": "BROKEN_no_atoms",
            })
            summary["BROKEN"] += 1
            continue
        engine_centroid = np.mean(atom_xyz, axis=0)

        ca_distances = {
            resseq: float(np.linalg.norm(ca - engine_centroid))
            for resseq, ca in cas_by_resseq.items()
        }
        sorted_cas = sorted(ca_distances.items(), key=lambda x: x[1])
        best_resseq, best_d = sorted_cas[0]
        alts = [str(r) for r, d in sorted_cas[1:5] if d <= ambiguous_thresh]

        if best_d > broken_thresh:
            status = "BROKEN_no_match"
            summary["BROKEN"] += 1
        elif best_d <= clean_thresh and not alts:
            status = "CLEAN"
            summary["CLEAN"] += 1
        elif best_d <= clean_thresh and alts:
            status = "AMBIGUOUS"
            summary["AMBIGUOUS"] += 1
        else:
            status = "BROKEN_loose_match"
            summary["BROKEN"] += 1

        rows.append({
            "engine_id": engine_id,
            "best_pdb_resseq": best_resseq,
            "distance_A": round(best_d, 3),
            "alt_pdb_resseqs": ";".join(alts),
            "status": status,
        })

    clean_offsets = [
        r["best_pdb_resseq"] - r["engine_id"]
        for r in rows if r["status"] == "CLEAN"
    ]
    if clean_offsets:
        modal_offset = max(set(clean_offsets), key=clean_offsets.count)
        modal_count = clean_offsets.count(modal_offset)
    else:
        modal_offset = None
        modal_count = 0

    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(output_csv, "w", newline="") as f:
        f.write(f"# Numbering audit (single-frame post-MD)\n")
        f.write(f"# Topology: {topology_path}\n")
        f.write(f"# MD PDB: {md_pdb_path}\n")
        f.write(f"# Chain: {chain_id}\n")
        f.write(f"# Total engine residues: {len(rows)}\n")
        f.write(
            f"# CLEAN: {summary['CLEAN']}, "
            f"AMBIGUOUS: {summary['AMBIGUOUS']}, "
            f"BROKEN: {summary['BROKEN']}\n"
        )
        if modal_offset is not None:
            f.write(
                f"# Modal offset (CLEAN residues): "
                f"{modal_offset:+d} ({modal_count} residues)\n"
            )
        else:
            f.write("# Modal offset: undetermined (no CLEAN matches)\n")
        f.write(
            f"# Thresholds: CLEAN<={clean_thresh}A, BROKEN>{broken_thresh}A, "
            f"AMBIGUOUS_alt<={ambiguous_thresh}A\n"
        )
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "engine_id", "best_pdb_resseq", "distance_A",
                "alt_pdb_resseqs", "status",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"Audit complete: {len(rows)} engine residues")
    print(f"  CLEAN:     {summary['CLEAN']}")
    print(f"  AMBIGUOUS: {summary['AMBIGUOUS']}")
    print(f"  BROKEN:    {summary['BROKEN']}")
    if modal_offset is not None:
        print(f"  Modal offset: {modal_offset:+d} ({modal_count} residues)")
    print(f"  Output: {output_csv}")
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--topology", required=True)
    parser.add_argument("--md-pdb", required=True)
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--chain", default="A")
    args = parser.parse_args()
    ok = audit_numbering(
        args.topology, args.md_pdb, args.output_csv, chain_id=args.chain,
    )
    sys.exit(0 if ok else 1)
