#!/usr/bin/env python3
"""Extract ground truth ligand centroids from holo PDB files.

For each target in the benchmark manifest, extracts the ligand heavy atom
coordinates from the holo structure and computes the geometric centroid.
Outputs ground_truth/ligand_centroids.json for DCC evaluation.
"""

import json
import os
import sys
import numpy as np

def extract_ligand_centroid(pdb_path, ligand_id, lig_chain, multi_residue=False):
    """Extract geometric centroid of ligand heavy atoms from PDB."""
    coords = []

    with open(pdb_path) as f:
        for line in f:
            if not line.startswith("HETATM"):
                continue
            resname = line[17:20].strip()
            chain = line[21].strip()
            element = line[76:78].strip() if len(line) > 76 else ""
            atom_name = line[12:16].strip()

            # Skip hydrogens
            if element == "H" or atom_name.startswith("H"):
                continue

            if multi_residue:
                # For cyclosporin etc: accept any HETATM on the specified chain
                # that isn't solvent
                if chain == lig_chain and resname not in ("HOH", "WAT", "GOL", "EDO"):
                    try:
                        x = float(line[30:38])
                        y = float(line[38:46])
                        z = float(line[46:54])
                        coords.append([x, y, z])
                    except ValueError:
                        continue
            else:
                # Match ligand name AND chain to avoid multi-copy averaging
                if resname == ligand_id and chain == lig_chain:
                    try:
                        x = float(line[30:38])
                        y = float(line[38:46])
                        z = float(line[46:54])
                        coords.append([x, y, z])
                    except ValueError:
                        continue

    if not coords:
        return None, 0

    coords = np.array(coords)
    centroid = coords.mean(axis=0)
    return centroid.tolist(), len(coords)


def main():
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    manifest = json.load(open("benchmark_manifest.json"))
    os.makedirs("ground_truth", exist_ok=True)

    results = {}

    for t in manifest["targets"]:
        tid = t["id"]
        holo_file = t["holo_file"]
        ligand = t["ligand"]
        lig_chain = t["lig_chain"]
        multi_residue = "ligand_note" in t  # cyclosporin etc.

        centroid, n_atoms = extract_ligand_centroid(
            holo_file, ligand, lig_chain, multi_residue
        )

        if centroid is None:
            print(f"FAIL {tid}: no ligand atoms found")
            continue

        results[tid] = {
            "centroid_xyz": [round(c, 3) for c in centroid],
            "ligand": ligand,
            "n_heavy_atoms": n_atoms,
            "holo_pdb": t["holo_pdb"],
            "lig_chain": lig_chain
        }
        print(f"OK   {tid:6s}  centroid=[{centroid[0]:7.2f}, {centroid[1]:7.2f}, {centroid[2]:7.2f}]  n_atoms={n_atoms}")

    output = {
        "benchmark": manifest["name"],
        "version": manifest["version"],
        "centroid_method": "geometric mean of ligand heavy atoms",
        "coordinate_frame": "holo PDB (requires apo->holo CA superposition for DCC)",
        "targets": results
    }

    with open("ground_truth/ligand_centroids.json", "w") as f:
        json.dump(output, f, indent=2)

    print(f"\n=== {len(results)}/30 centroids extracted ===")
    print(f"Output: ground_truth/ligand_centroids.json")


if __name__ == "__main__":
    main()
