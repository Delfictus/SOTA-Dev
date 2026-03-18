import json
import os
import argparse
import glob
import numpy as np


def get_rmsd_from_coords(ref_file, probe_file):
    """Compute heavy-atom RMSD between a reference PDB and a docked SDF pose.

    Uses coordinate-based RMSD (no atom mapping) since the docked ligand
    may have different atom ordering than the crystallographic reference.
    Falls back to centroid distance if atom counts differ.
    """
    try:
        ref_coords = _parse_heavy_coords_pdb(ref_file)
        probe_coords = _parse_heavy_coords_sdf(probe_file)

        if ref_coords is None or probe_coords is None:
            return None
        if len(ref_coords) == 0 or len(probe_coords) == 0:
            return None

        # If atom counts match, compute direct RMSD
        if len(ref_coords) == len(probe_coords):
            diff = ref_coords - probe_coords
            return float(np.sqrt(np.mean(np.sum(diff ** 2, axis=1))))

        # Different atom counts — use centroid distance as proxy
        ref_cent = ref_coords.mean(axis=0)
        probe_cent = probe_coords.mean(axis=0)
        return float(np.linalg.norm(ref_cent - probe_cent))

    except Exception:
        return None


def _parse_heavy_coords_pdb(pdb_path):
    """Extract heavy atom coordinates from HETATM records in a PDB file."""
    SKIP = {"HOH", "WAT", "NA", "CL", "MG", "ZN", "CA", "K", "FE",
            "MN", "CO", "NI", "CU", "SO4", "PO4", "GOL", "EDO"}
    PROTEIN = {"ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY",
               "HIS", "ILE", "LEU", "LYS", "MET", "PHE", "PRO", "SER",
               "THR", "TRP", "TYR", "VAL", "MSE"}

    coords = []
    with open(pdb_path) as f:
        for line in f:
            if not line.startswith("HETATM"):
                continue
            resname = line[17:20].strip()
            elem = line[76:78].strip() if len(line) > 76 else ""
            if resname in SKIP or resname in PROTEIN:
                continue
            if elem == "H":
                continue
            try:
                x = float(line[30:38])
                y = float(line[38:46])
                z = float(line[46:54])
                coords.append([x, y, z])
            except ValueError:
                continue

    return np.array(coords) if coords else None


def _parse_heavy_coords_sdf(sdf_path):
    """Extract heavy atom coordinates from the first pose in an SDF file."""
    coords = []
    with open(sdf_path) as f:
        lines = f.readlines()

    # Find counts line (V2000)
    n_atoms = 0
    start = 0
    for i, line in enumerate(lines):
        if "V2000" in line:
            parts = line.split()
            try:
                n_atoms = int(parts[0])
            except (ValueError, IndexError):
                continue
            start = i + 1
            break

    if n_atoms == 0:
        return None

    for i in range(start, min(start + n_atoms, len(lines))):
        parts = lines[i].split()
        if len(parts) < 4:
            continue
        try:
            x, y, z = float(parts[0]), float(parts[1]), float(parts[2])
            elem = parts[3] if len(parts) > 3 else "?"
            if elem != "H":
                coords.append([x, y, z])
        except ValueError:
            continue

    return np.array(coords) if coords else None


def _parse_sdf_centroid(sdf_path):
    """Get heavy-atom centroid from the best pose (first) in an SDF file."""
    coords = _parse_heavy_coords_sdf(sdf_path)
    if coords is None or len(coords) == 0:
        return None
    return coords.mean(axis=0)


def main():
    parser = argparse.ArgumentParser(description="Validate docked poses against aligned ground truth")
    parser.add_argument('--results_dir', default='docking_results_full')
    parser.add_argument('--manifest', default='benchmarks/prism4d_bench30/benchmark_manifest.json')
    parser.add_argument('--gt', default='benchmarks/prism4d_bench30/ground_truth/ligand_centroids.json')
    args = parser.parse_args()

    with open(args.manifest) as f:
        data = json.load(f)
    with open(args.gt) as f:
        gt = json.load(f)

    print(f"{'Target':<10} | {'Best Site':<12} | {'DCC (A)':<10} | {'Poses':<6} | {'Quality'}")
    print("-" * 60)

    results = []
    for target in data['targets']:
        tid = str(target['id'])
        name = target['apo_pdb'].upper()

        if tid not in gt:
            print(f"{name:<10} | {'No GT':<12} | {'N/A':<10} | {'0':<6} | -")
            continue

        true_cent = np.array(gt[tid]['centroid'])

        pattern = os.path.join(args.results_dir, name.lower(), "site*", "unidock_out", "*_out.sdf")
        best_dcc = 999.9
        best_site = "?"
        n_poses = 0

        for sdf in glob.glob(pattern):
            n_poses += 1
            site_id = sdf.split("/site")[-1].split("/")[0]
            pose_cent = _parse_sdf_centroid(sdf)
            if pose_cent is not None:
                dcc = float(np.linalg.norm(pose_cent - true_cent))
                if dcc < best_dcc:
                    best_dcc = dcc
                    best_site = f"site{site_id}"

        if best_dcc < 999:
            if best_dcc < 2.0:
                quality = "ELITE"
            elif best_dcc < 4.0:
                quality = "EXCELLENT"
            elif best_dcc < 6.0:
                quality = "GOOD"
            elif best_dcc < 10.0:
                quality = "MARGINAL"
            else:
                quality = "POOR"
            print(f"{name:<10} | {best_site:<12} | {best_dcc:<10.2f} | {n_poses:<6} | {quality}")
            results.append({"target": name, "dcc": best_dcc, "quality": quality})
        else:
            print(f"{name:<10} | {'No Pose':<12} | {'N/A':<10} | {n_poses:<6} | -")

    # Summary
    if results:
        dccs = [r["dcc"] for r in results]
        n = len(dccs)
        print(f"\n{'='*60}")
        print(f"PRISM4D DOCKING SUMMARY ({n} targets with poses):")
        for thresh, label in [(2,"ELITE"),(4,"EXCELLENT"),(6,"GOOD"),(8,"MARGINAL"),(10,"FAIR")]:
            ct = sum(1 for d in dccs if d < thresh)
            print(f"  <{thresh}A ({label:>9}): {ct}/{n} ({100*ct/n:.0f}%)")
        print(f"  Mean DCC:   {np.mean(dccs):.2f} A")
        print(f"  Median DCC: {np.median(dccs):.2f} A")


if __name__ == "__main__":
    main()
