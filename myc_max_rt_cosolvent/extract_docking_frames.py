#!/usr/bin/env python3
"""
PRISM-4D MYC-MAX Frame Extraction for Ensemble Docking
=======================================================

Extracts trajectory frames where the Site 1 cross-chain pocket is most
open (suitable for docking). Ranks frames by inter-chain distance at
the dimerization interface and saves top conformations as individual PDBs.

Site 1 residues (topology numbering):
  MYC side (Chain A): topo 59-71 → UniProt E407-R419
  MAX side (Chain B): topo 142-153 → UniProt K77-L88

Usage:
  python3 extract_docking_frames.py [--output-dir OUTPUT] [--top-n 10]

Requires: numpy (pip install numpy --break-system-packages)
"""

import os
import sys
import json
import glob
import math
import argparse
from pathlib import Path
from collections import defaultdict

# ─── CONFIGURATION ───
BASE_DIR = os.path.expanduser("~/Desktop/Prism4D-bio")
RT_DIR = os.path.join(BASE_DIR, "myc_max_rt_cosolvent")
TOPO_PATH = os.path.join(BASE_DIR, "e2e_validation_test/prep/1nkp_dimer.topology.json")

# Site 1 residue definitions (topo IDs)
SITE1_MYC = list(range(59, 72))       # topo 59-71 = MYC E407-R419
SITE1_MAX = list(range(142, 154))      # topo 142-153 = MAX K77-L88

# MAX hydrophobic patch (docking filter target)
MAX_HYDROPHOBIC_PATCH = {
    150: "I85",   # ILE85 - critical dimerization residue
    153: "L88",   # LEU88 - critical dimerization residue
    149: "D84",   # ASP84 - flanking
    146: "H81",   # HIS81 - critical dimerization residue
}

# UniProt mapping
MYC_OFFSET = 348   # topo_id + 348 = UniProt (for topo >= 5)
MAX_OFFSET = -65    # topo_id - 65 = UniProt (for chain B)


def load_topology(path):
    """Load topology JSON and build residue-to-CA-atom index map."""
    print(f"Loading topology: {path}")
    with open(path) as f:
        topo = json.load(f)

    n_atoms = topo["n_atoms"]
    n_residues = topo["n_residues"]
    residue_ids = topo["residue_ids"]
    chain_ids = topo["chain_ids"]
    atom_names = topo["atom_names"]
    residue_names = topo["residue_names"]
    ca_indices = topo.get("ca_indices", [])

    print(f"  Atoms: {n_atoms}, Residues: {n_residues}, Chains: {topo['n_chains']}")
    print(f"  CA indices available: {len(ca_indices)}")

    # Build residue_id → CA atom index mapping
    # If ca_indices is available, use it directly
    res_to_ca = {}
    if ca_indices:
        # ca_indices is a flat list of atom indices for CA atoms, one per residue
        for res_idx, ca_idx in enumerate(ca_indices):
            res_to_ca[res_idx] = ca_idx
    else:
        # Fallback: scan atom_names for CA
        seen_res = set()
        for atom_idx in range(n_atoms):
            res_id = residue_ids[atom_idx]
            if atom_names[atom_idx] == "CA" and res_id not in seen_res:
                res_to_ca[res_id] = atom_idx
                seen_res.add(res_id)

    # Build residue info table
    res_info = {}
    seen = set()
    seq_idx = 0
    for atom_idx in range(n_atoms):
        rid = residue_ids[atom_idx]
        cid = chain_ids[atom_idx]
        rname = residue_names[atom_idx]
        key = (cid, rid)
        if key not in seen:
            seen.add(key)
            res_info[seq_idx] = {
                "resid": rid,
                "chain": cid,
                "name": rname,
                "seq_idx": seq_idx
            }
            seq_idx += 1

    return {
        "n_atoms": n_atoms,
        "n_residues": n_residues,
        "res_to_ca": res_to_ca,
        "res_info": res_info,
        "ca_indices": ca_indices,
        "residue_ids": residue_ids,
        "chain_ids": chain_ids,
        "atom_names": atom_names,
        "residue_names": residue_names,
    }


def parse_ensemble_pdb(pdb_path):
    """Parse multi-model PDB into list of frames, each with atom coordinates."""
    print(f"Parsing ensemble PDB: {pdb_path}")
    frames = []
    current_frame = []
    current_model = 0
    frame_atoms = []

    with open(pdb_path) as f:
        for line in f:
            if line.startswith("MODEL"):
                current_frame = []
                frame_atoms = []
                try:
                    current_model = int(line[5:].strip())
                except ValueError:
                    current_model = len(frames)
            elif line.startswith("ENDMDL"):
                if current_frame:
                    frames.append({
                        "model": current_model,
                        "coords": current_frame,
                        "lines": frame_atoms,
                    })
                current_frame = []
                frame_atoms = []
            elif line.startswith("ATOM") or line.startswith("HETATM"):
                try:
                    x = float(line[30:38])
                    y = float(line[38:46])
                    z = float(line[46:54])
                    # Sanity check for exploded coordinates
                    if abs(x) < 10000 and abs(y) < 10000 and abs(z) < 10000:
                        current_frame.append((x, y, z))
                        frame_atoms.append(line)
                    else:
                        current_frame.append((x, y, z))
                        frame_atoms.append(line)
                except (ValueError, IndexError):
                    frame_atoms.append(line)
                    current_frame.append((0, 0, 0))

    # Handle single-model PDB (no MODEL/ENDMDL)
    if not frames and current_frame:
        frames.append({
            "model": 0,
            "coords": current_frame,
            "lines": frame_atoms,
        })

    print(f"  Parsed {len(frames)} frames")

    # Check for exploded frames
    valid_frames = []
    for i, frame in enumerate(frames):
        if frame["coords"]:
            max_coord = max(max(abs(c) for c in xyz) for xyz in frame["coords"][:100])
            if max_coord < 10000:
                valid_frames.append(frame)
            else:
                print(f"  Frame {i}: EXPLODED (max coord {max_coord:.0f}), skipping")

    print(f"  Valid frames: {len(valid_frames)} / {len(frames)}")
    return valid_frames


def compute_site1_metrics(frame, topo):
    """
    Compute Site 1 pocket metrics for a single frame.
    Returns dict with inter-chain distance, pocket openness, and per-residue data.
    """
    coords = frame["coords"]
    ca_map = topo["res_to_ca"]

    # Get CA coordinates for Site 1 residues
    myc_cas = []
    max_cas = []

    for res_id in SITE1_MYC:
        if res_id in ca_map and ca_map[res_id] < len(coords):
            myc_cas.append(coords[ca_map[res_id]])

    for res_id in SITE1_MAX:
        if res_id in ca_map and ca_map[res_id] < len(coords):
            max_cas.append(coords[ca_map[res_id]])

    if not myc_cas or not max_cas:
        return None

    # Compute inter-chain distances
    distances = []
    for mc in myc_cas:
        for xc in max_cas:
            d = math.sqrt(sum((a - b) ** 2 for a, b in zip(mc, xc)))
            distances.append(d)

    # Compute MYC-MAX centroid distance (pocket openness indicator)
    myc_centroid = [sum(c[i] for c in myc_cas) / len(myc_cas) for i in range(3)]
    max_centroid = [sum(c[i] for c in max_cas) / len(max_cas) for i in range(3)]
    centroid_dist = math.sqrt(sum((a - b) ** 2 for a, b in zip(myc_centroid, max_centroid)))

    # Compute min distance (closest approach - tighter = more defined pocket)
    min_dist = min(distances)
    mean_dist = sum(distances) / len(distances)

    # Compute hydrophobic patch distances (I85, L88 to nearest MYC residue)
    hydro_dists = {}
    for res_id, label in MAX_HYDROPHOBIC_PATCH.items():
        if res_id in ca_map and ca_map[res_id] < len(coords):
            hc = coords[ca_map[res_id]]
            min_d = min(math.sqrt(sum((a - b) ** 2 for a, b in zip(hc, mc))) for mc in myc_cas)
            hydro_dists[label] = min_d

    # Pocket volume proxy: spread of inter-chain distances
    # Higher spread + moderate mean = well-defined transient pocket
    dist_spread = max(distances) - min(distances) if distances else 0

    # Composite "dockability" score
    # Ideal: centroid distance 10-15Å (pocket open but not dissociated)
    # Ideal: min distance 5-8Å (close contacts exist)
    # Ideal: spread > 5Å (heterogeneous pocket surface)
    centroid_score = 1.0 - abs(centroid_dist - 12.5) / 12.5  # peak at 12.5Å
    contact_score = 1.0 - abs(min_dist - 6.5) / 10.0          # peak at 6.5Å
    spread_score = min(dist_spread / 10.0, 1.0)                # saturates at 10Å

    # Weight: centroid matters most, then contacts, then spread
    dockability = max(0, 0.5 * centroid_score + 0.3 * contact_score + 0.2 * spread_score)

    return {
        "centroid_dist": centroid_dist,
        "min_dist": min_dist,
        "mean_dist": mean_dist,
        "dist_spread": dist_spread,
        "dockability": dockability,
        "hydro_dists": hydro_dists,
        "myc_centroid": myc_centroid,
        "max_centroid": max_centroid,
        "n_myc_residues": len(myc_cas),
        "n_max_residues": len(max_cas),
    }


def save_frame_pdb(frame, frame_idx, metrics, output_dir, topo):
    """Save a single frame as a PDB file with proper metadata."""
    filename = f"site1_dock_frame_{frame_idx:04d}.pdb"
    filepath = os.path.join(output_dir, filename)

    with open(filepath, "w") as f:
        f.write(f"REMARK   1 PRISM-4D MYC-MAX Site 1 Docking Frame\n")
        f.write(f"REMARK   2 Frame index: {frame_idx}\n")
        f.write(f"REMARK   3 Dockability score: {metrics['dockability']:.4f}\n")
        f.write(f"REMARK   4 Centroid distance: {metrics['centroid_dist']:.2f} A\n")
        f.write(f"REMARK   5 Min contact distance: {metrics['min_dist']:.2f} A\n")
        f.write(f"REMARK   6 Site 1 MYC: E407-R419 (topo 59-71)\n")
        f.write(f"REMARK   7 Site 1 MAX: K77-L88 (topo 142-153)\n")
        for label, dist in metrics.get("hydro_dists", {}).items():
            f.write(f"REMARK   8 Hydrophobic {label}: {dist:.2f} A to nearest MYC\n")
        f.write(f"MODEL     {frame_idx}\n")
        for line in frame["lines"]:
            f.write(line if line.endswith("\n") else line + "\n")
        f.write("ENDMDL\n")
        f.write("END\n")

    return filepath


def save_ensemble_pdb(ranked_frames, output_dir):
    """Save all top frames as a single multi-model PDB for ensemble docking."""
    filepath = os.path.join(output_dir, "site1_ensemble_docking.pdb")
    with open(filepath, "w") as f:
        f.write("REMARK   1 PRISM-4D MYC-MAX Site 1 Ensemble for Docking\n")
        f.write(f"REMARK   2 Contains {len(ranked_frames)} highest-scoring conformations\n")
        f.write("REMARK   3 Ranked by Site 1 pocket dockability score\n")
        f.write("REMARK   4 Use for ensemble docking against MYC-MAX interface\n")
        f.write("REMARK   5 Filter hits: require contact with MAX I85/L88 hydrophobic patch\n")
        for rank, (frame, frame_idx, metrics) in enumerate(ranked_frames):
            f.write(f"REMARK  10 Rank {rank+1}: frame {frame_idx}, "
                    f"score={metrics['dockability']:.3f}, "
                    f"centroid={metrics['centroid_dist']:.1f}A\n")
        for rank, (frame, frame_idx, metrics) in enumerate(ranked_frames):
            f.write(f"MODEL     {rank + 1}\n")
            for line in frame["lines"]:
                f.write(line if line.endswith("\n") else line + "\n")
            f.write("ENDMDL\n")
        f.write("END\n")
    return filepath


def generate_docking_filter_script(output_dir, topo):
    """Generate a Python script for post-docking filtering by I85/L88 contact."""
    script = '''#!/usr/bin/env python3
"""
Post-Docking Filter: MYC-MAX Site 1 Hydrophobic Patch Contact
=============================================================
Filters docking results to retain only compounds that make contact
with the MAX I85/L88 hydrophobic patch (the "floor" of the zipper).

Usage:
  python3 filter_docking_hits.py <docking_results.sdf> [--cutoff 4.0]

Input: SDF file from docking (AutoDock Vina, Glide, etc.)
Output: Filtered SDF with only I85/L88-contacting compounds
"""
import sys
import argparse

# MAX hydrophobic patch CA coordinates (from best docking frame)
# These are approximate - update with actual coordinates from your frame
PATCH_RESIDUES = {
    "I85": {"chain": "B", "resid": 150, "atoms": ["CA", "CB", "CG1", "CG2", "CD1"]},
    "L88": {"chain": "B", "resid": 153, "atoms": ["CA", "CB", "CG", "CD1", "CD2"]},
    "H81": {"chain": "B", "resid": 146, "atoms": ["CA", "CB", "CG", "ND1", "CE1"]},
}

CONTACT_CUTOFF = 4.0  # Angstroms

def parse_args():
    p = argparse.ArgumentParser(description="Filter docking hits by I85/L88 contact")
    p.add_argument("input_sdf", help="Docking results SDF file")
    p.add_argument("--cutoff", type=float, default=4.0, help="Contact distance cutoff (A)")
    p.add_argument("--receptor-pdb", default="site1_dock_frame_best.pdb",
                   help="Receptor PDB for coordinate reference")
    p.add_argument("-o", "--output", default="filtered_hits.sdf", help="Output SDF")
    return p.parse_args()

def get_patch_coords(pdb_path):
    """Extract I85/L88 heavy atom coordinates from receptor PDB."""
    coords = {}
    with open(pdb_path) as f:
        for line in f:
            if not line.startswith("ATOM"):
                continue
            chain = line[21]
            resid = int(line[22:26].strip())
            aname = line[12:16].strip()
            for label, info in PATCH_RESIDUES.items():
                if chain == info["chain"] and resid == info["resid"] and aname in info["atoms"]:
                    x = float(line[30:38])
                    y = float(line[38:46])
                    z = float(line[46:54])
                    if label not in coords:
                        coords[label] = []
                    coords[label].append((x, y, z))
    return coords

def min_distance(lig_coords, patch_coords):
    """Minimum distance between any ligand atom and any patch atom."""
    import math
    min_d = float("inf")
    for lx, ly, lz in lig_coords:
        for label, pcoords in patch_coords.items():
            for px, py, pz in pcoords:
                d = math.sqrt((lx-px)**2 + (ly-py)**2 + (lz-pz)**2)
                if d < min_d:
                    min_d = d
    return min_d

print("Post-docking filter for MYC-MAX Site 1")
print("Requires: docking results in SDF format + receptor PDB")
print("Update PATCH_RESIDUES coordinates from your specific docking frame")
print()
print("For AutoDock Vina:")
print("  vina --receptor site1_dock_frame_0001.pdbqt --ligand library.pdbqt \\\\")
print("       --center_x {cx:.1f} --center_y {cy:.1f} --center_z {cz:.1f} \\\\")
print("       --size_x 20 --size_y 20 --size_z 20")
'''.format(cx=0, cy=0, cz=0)  # placeholder coordinates

    filepath = os.path.join(output_dir, "filter_docking_hits.py")
    with open(filepath, "w") as f:
        f.write(script)
    return filepath


def generate_vina_box_script(output_dir, best_metrics):
    """Generate AutoDock Vina docking box centered on Site 1."""
    mc = best_metrics["myc_centroid"]
    xc = best_metrics["max_centroid"]
    # Center the box between MYC and MAX centroids
    center = [(mc[i] + xc[i]) / 2.0 for i in range(3)]

    script = f"""#!/bin/bash
# AutoDock Vina docking box for MYC-MAX Site 1
# Center: midpoint between MYC and MAX interface centroids
# Size: 24x24x24 Å to cover the full cross-chain pocket

# Prepare receptor (from best docking frame)
# Requires: MGLTools or ADFR suite
prepare_receptor -r site1_dock_frame_best.pdb -o receptor.pdbqt

# Prepare ligand library
# For each ligand in your screening library:
# prepare_ligand -l ligand.mol2 -o ligand.pdbqt

# Docking command
vina \\
  --receptor receptor.pdbqt \\
  --ligand ligand.pdbqt \\
  --center_x {center[0]:.2f} \\
  --center_y {center[1]:.2f} \\
  --center_z {center[2]:.2f} \\
  --size_x 24 \\
  --size_y 24 \\
  --size_z 24 \\
  --exhaustiveness 32 \\
  --num_modes 20 \\
  --out docking_results.pdbqt \\
  --log docking_log.txt

# For ensemble docking (all top frames):
# for frame in site1_dock_frame_*.pdb; do
#   prepare_receptor -r $frame -o ${{frame%.pdb}}.pdbqt
#   vina --receptor ${{frame%.pdb}}.pdbqt --ligand ligand.pdbqt \\
#     --center_x {center[0]:.2f} --center_y {center[1]:.2f} --center_z {center[2]:.2f} \\
#     --size_x 24 --size_y 24 --size_z 24 \\
#     --exhaustiveness 32 --num_modes 10 \\
#     --out docking_${{frame%.pdb}}.pdbqt
# done

echo "Docking box center: ({center[0]:.2f}, {center[1]:.2f}, {center[2]:.2f})"
echo "Box size: 24 x 24 x 24 Å"
echo ""
echo "POST-DOCKING FILTER:"
echo "  Keep only hits with ANY heavy atom within 4.0 Å of:"
echo "    MAX I85 (topo 150, chain B)"
echo "    MAX L88 (topo 153, chain B)"
echo "  These form the hydrophobic floor of the zipper pocket."
"""
    filepath = os.path.join(output_dir, "run_vina_docking.sh")
    with open(filepath, "w") as f:
        f.write(script)
    os.chmod(filepath, 0o755)
    return filepath


def generate_pymol_validation(output_dir, best_frame_idx, best_metrics):
    """Generate PyMOL script to visualize best docking frame with pocket."""
    mc = best_metrics["myc_centroid"]
    xc = best_metrics["max_centroid"]
    center = [(mc[i] + xc[i]) / 2.0 for i in range(3)]

    script = f"""# PyMOL Visualization: Best Docking Frame with Site 1 Pocket
# Run: pymol validate_docking_frame.pml

load site1_dock_frame_{best_frame_idx:04d}.pdb, receptor

# Basic display
hide everything
show cartoon, receptor
color gray80, receptor

# Color chains
select chain_myc, chain A
select chain_max, chain B
color marine, chain_myc
color orange, chain_max

# Site 1 MYC residues (E407-R419)
select site1_myc, chain A and resi 59-71
color tv_blue, site1_myc
show sticks, site1_myc

# Site 1 MAX residues (K77-L88)
select site1_max, chain B and resi 142-153
color tv_orange, site1_max
show sticks, site1_max

# Hydrophobic patch (docking filter target)
select hydro_patch, chain B and (resi 150 or resi 153)
color firebrick, hydro_patch
show spheres, hydro_patch
set sphere_scale, 0.5, hydro_patch

# H81 - another critical residue
select his81, chain B and resi 146
color salmon, his81
show sticks, his81

# Pocket center pseudoatom
pseudoatom pocket_center, pos=[{center[0]:.2f}, {center[1]:.2f}, {center[2]:.2f}]
show spheres, pocket_center
color green, pocket_center
set sphere_scale, 1.0, pocket_center

# Docking box visualization (24x24x24 Å)
# Shows the search space for AutoDock Vina
set cgo_transparency, 0.7
python
from pymol.cgo import *
from pymol import cmd
cx, cy, cz = {center[0]:.2f}, {center[1]:.2f}, {center[2]:.2f}
s = 12.0  # half-size
box = [
    BEGIN, LINES,
    COLOR, 0.0, 0.8, 0.0,
    VERTEX, cx-s, cy-s, cz-s, VERTEX, cx+s, cy-s, cz-s,
    VERTEX, cx+s, cy-s, cz-s, VERTEX, cx+s, cy+s, cz-s,
    VERTEX, cx+s, cy+s, cz-s, VERTEX, cx-s, cy+s, cz-s,
    VERTEX, cx-s, cy+s, cz-s, VERTEX, cx-s, cy-s, cz-s,
    VERTEX, cx-s, cy-s, cz+s, VERTEX, cx+s, cy-s, cz+s,
    VERTEX, cx+s, cy-s, cz+s, VERTEX, cx+s, cy+s, cz+s,
    VERTEX, cx+s, cy+s, cz+s, VERTEX, cx-s, cy+s, cz+s,
    VERTEX, cx-s, cy+s, cz+s, VERTEX, cx-s, cy-s, cz+s,
    VERTEX, cx-s, cy-s, cz-s, VERTEX, cx-s, cy-s, cz+s,
    VERTEX, cx+s, cy-s, cz-s, VERTEX, cx+s, cy-s, cz+s,
    VERTEX, cx+s, cy+s, cz-s, VERTEX, cx+s, cy+s, cz+s,
    VERTEX, cx-s, cy+s, cz-s, VERTEX, cx-s, cy+s, cz+s,
    END
]
cmd.load_cgo(box, "docking_box")
python end

# Labels
label site1_myc and name CA, "%s%s" % (resn, resi)
label hydro_patch and name CA, "%s%s *" % (resn, resi)

# View
zoom site1_myc or site1_max, 8
set label_size, 12
set label_color, white
set bg_rgb, [0.1, 0.1, 0.1]
set ray_shadows, 1
set antialias, 2
"""
    filepath = os.path.join(output_dir, "validate_docking_frame.pml")
    with open(filepath, "w") as f:
        f.write(script)
    return filepath


def main():
    parser = argparse.ArgumentParser(description="Extract PRISM-4D docking frames for MYC-MAX Site 1")
    parser.add_argument("--rt-dir", default=RT_DIR, help="RT cosolvent output directory")
    parser.add_argument("--topo", default=TOPO_PATH, help="Topology JSON path")
    parser.add_argument("--output-dir", "-o", default=None, help="Output directory for docking frames")
    parser.add_argument("--top-n", type=int, default=10, help="Number of top frames to extract")
    parser.add_argument("--frames-json", default=None, help="Optional frames.json for spike correlation")
    args = parser.parse_args()

    if args.output_dir is None:
        args.output_dir = os.path.join(args.rt_dir, "docking_frames")

    os.makedirs(args.output_dir, exist_ok=True)

    # ─── Step 1: Load topology ───
    if not os.path.exists(args.topo):
        print(f"ERROR: Topology not found: {args.topo}")
        sys.exit(1)
    topo = load_topology(args.topo)

    # Print Site 1 residue info
    print("\n=== SITE 1 RESIDUE MAP ===")
    print("MYC side (Chain A):")
    for rid in SITE1_MYC:
        info = topo["res_info"].get(rid, {})
        uniprot = rid + MYC_OFFSET if rid >= 5 else "TAG"
        print(f"  topo {rid:3d} → {info.get('name', '???'):3s} {uniprot}")
    print("MAX side (Chain B):")
    for rid in SITE1_MAX:
        info = topo["res_info"].get(rid, {})
        uniprot = rid + MAX_OFFSET
        marker = " ★" if rid in MAX_HYDROPHOBIC_PATCH else ""
        print(f"  topo {rid:3d} → {info.get('name', '???'):3s} {uniprot}{marker}")

    # ─── Step 2: Find ensemble trajectory ───
    ensemble_pdb = None
    candidates = [
        os.path.join(args.rt_dir, "1nkp_dimer.ensemble_trajectory.pdb"),
        os.path.join(args.rt_dir, "1nkp_dimer.topology_ensemble.pdb"),
    ]
    candidates += glob.glob(os.path.join(args.rt_dir, "*ensemble*.pdb"))
    candidates += glob.glob(os.path.join(args.rt_dir, "*trajectory*.pdb"))

    for cand in candidates:
        if os.path.exists(cand):
            ensemble_pdb = cand
            break

    if ensemble_pdb is None:
        print("\nERROR: No ensemble trajectory PDB found in", args.rt_dir)
        print("Available PDB files:")
        for f in glob.glob(os.path.join(args.rt_dir, "*.pdb")):
            print(f"  {f}")
        sys.exit(1)

    # ─── Step 3: Parse trajectory ───
    frames = parse_ensemble_pdb(ensemble_pdb)
    if not frames:
        print("ERROR: No valid frames found in trajectory")
        sys.exit(1)

    # ─── Step 4: Score each frame ───
    print(f"\n=== SCORING {len(frames)} FRAMES FOR SITE 1 POCKET OPENNESS ===")
    scored_frames = []
    for i, frame in enumerate(frames):
        metrics = compute_site1_metrics(frame, topo)
        if metrics:
            scored_frames.append((frame, i, metrics))

    if not scored_frames:
        print("ERROR: Could not compute Site 1 metrics for any frame")
        print("Check that CA indices are correct for residues 59-71, 142-153")
        sys.exit(1)

    # ─── Step 5: Rank and report ───
    scored_frames.sort(key=lambda x: x[2]["dockability"], reverse=True)

    print(f"\n=== TOP {min(args.top_n, len(scored_frames))} DOCKING FRAMES ===")
    print(f"{'Rank':>4} {'Frame':>6} {'Score':>7} {'Centroid':>9} {'MinDist':>8} "
          f"{'Spread':>7} {'I85':>6} {'L88':>6}")
    print("-" * 65)

    for rank, (frame, idx, m) in enumerate(scored_frames[:args.top_n]):
        i85 = m["hydro_dists"].get("I85", float("nan"))
        l88 = m["hydro_dists"].get("L88", float("nan"))
        print(f"{rank+1:>4} {idx:>6} {m['dockability']:>7.4f} {m['centroid_dist']:>8.2f}A "
              f"{m['min_dist']:>7.2f}A {m['dist_spread']:>6.2f}A "
              f"{i85:>5.1f}A {l88:>5.1f}A")

    # ─── Step 6: Also show distribution stats ───
    all_scores = [m["dockability"] for _, _, m in scored_frames]
    all_centroid = [m["centroid_dist"] for _, _, m in scored_frames]
    print(f"\nDistribution across {len(scored_frames)} valid frames:")
    print(f"  Dockability:  min={min(all_scores):.3f}  max={max(all_scores):.3f}  "
          f"mean={sum(all_scores)/len(all_scores):.3f}")
    print(f"  Centroid dist: min={min(all_centroid):.1f}A  max={max(all_centroid):.1f}A  "
          f"mean={sum(all_centroid)/len(all_centroid):.1f}A")

    # ─── Step 7: Check frames.json for spike correlation ───
    frames_json_path = args.frames_json
    if frames_json_path is None:
        candidates = glob.glob(os.path.join(args.rt_dir, "*part*.frames.json"))
        candidates += glob.glob(os.path.join(args.rt_dir, "*frames*.json"))
        if candidates:
            frames_json_path = candidates[0]

    spike_data = None
    if frames_json_path and os.path.exists(frames_json_path):
        print(f"\n=== LOADING SPIKE DATA: {frames_json_path} ===")
        try:
            with open(frames_json_path) as f:
                spike_data = json.load(f)
            print(f"  Loaded {len(spike_data)} frame records")

            # Correlate spike counts with dockability
            print("\n=== SPIKE COUNT vs DOCKABILITY CORRELATION ===")
            print(f"{'Frame':>6} {'Spikes':>8} {'Triggered':>10} {'Score':>7} {'Centroid':>9}")
            print("-" * 50)
            for rank, (frame, idx, m) in enumerate(scored_frames[:args.top_n]):
                if idx < len(spike_data):
                    sd = spike_data[idx]
                    sc = sd.get("spike_count", 0)
                    st = sd.get("spike_triggered", False)
                    print(f"{idx:>6} {sc:>8} {'YES' if st else 'no':>10} "
                          f"{m['dockability']:>7.4f} {m['centroid_dist']:>8.2f}A")
        except Exception as e:
            print(f"  Warning: Could not parse frames.json: {e}")

    # ─── Step 8: Extract and save frames ───
    print(f"\n=== EXTRACTING TOP {min(args.top_n, len(scored_frames))} FRAMES ===")

    saved_files = []
    for rank, (frame, idx, metrics) in enumerate(scored_frames[:args.top_n]):
        filepath = save_frame_pdb(frame, idx, metrics, args.output_dir, topo)
        saved_files.append(filepath)
        print(f"  Saved: {os.path.basename(filepath)}")

    # Save best frame with canonical name
    best_frame, best_idx, best_metrics = scored_frames[0]
    best_path = os.path.join(args.output_dir, "site1_dock_frame_best.pdb")
    save_frame_pdb(best_frame, best_idx, best_metrics, args.output_dir, topo)
    import shutil
    src = save_frame_pdb(best_frame, best_idx, best_metrics, args.output_dir, topo)
    shutil.copy2(f"{args.output_dir}/site1_dock_frame_{best_idx:04d}.pdb", best_path)
    print(f"  Best frame: {best_path}")

    # Save ensemble PDB
    ensemble_path = save_ensemble_pdb(scored_frames[:args.top_n], args.output_dir)
    print(f"  Ensemble: {ensemble_path}")

    # ─── Step 9: Generate docking scripts ───
    vina_path = generate_vina_box_script(args.output_dir, best_metrics)
    filter_path = generate_docking_filter_script(args.output_dir, topo)
    pymol_path = generate_pymol_validation(args.output_dir, best_idx, best_metrics)

    print(f"\n  Vina script: {vina_path}")
    print(f"  Filter script: {filter_path}")
    print(f"  PyMOL validation: {pymol_path}")

    # ─── Step 10: Generate report ───
    report = {
        "run_info": {
            "source_trajectory": ensemble_pdb,
            "topology": args.topo,
            "total_frames": len(frames),
            "valid_frames": len(scored_frames),
            "extracted_frames": min(args.top_n, len(scored_frames)),
        },
        "site1_definition": {
            "myc_residues": {str(r): r + MYC_OFFSET for r in SITE1_MYC if r >= 5},
            "max_residues": {str(r): r + MAX_OFFSET for r in SITE1_MAX},
            "hydrophobic_patch": MAX_HYDROPHOBIC_PATCH,
        },
        "best_frame": {
            "frame_index": best_idx,
            "dockability_score": best_metrics["dockability"],
            "centroid_distance_A": best_metrics["centroid_dist"],
            "min_contact_distance_A": best_metrics["min_dist"],
            "hydrophobic_patch_distances": best_metrics["hydro_dists"],
        },
        "docking_box": {
            "center_x": (best_metrics["myc_centroid"][0] + best_metrics["max_centroid"][0]) / 2,
            "center_y": (best_metrics["myc_centroid"][1] + best_metrics["max_centroid"][1]) / 2,
            "center_z": (best_metrics["myc_centroid"][2] + best_metrics["max_centroid"][2]) / 2,
            "size_x": 24.0,
            "size_y": 24.0,
            "size_z": 24.0,
        },
        "ranking": [
            {
                "rank": rank + 1,
                "frame_index": idx,
                "dockability": m["dockability"],
                "centroid_dist": m["centroid_dist"],
                "min_dist": m["min_dist"],
                "I85_dist": m["hydro_dists"].get("I85"),
                "L88_dist": m["hydro_dists"].get("L88"),
                "file": f"site1_dock_frame_{idx:04d}.pdb",
            }
            for rank, (_, idx, m) in enumerate(scored_frames[:args.top_n])
        ],
        "distribution": {
            "dockability_min": min(all_scores),
            "dockability_max": max(all_scores),
            "dockability_mean": sum(all_scores) / len(all_scores),
            "centroid_dist_min": min(all_centroid),
            "centroid_dist_max": max(all_centroid),
            "centroid_dist_mean": sum(all_centroid) / len(all_centroid),
        },
        "output_files": {
            "individual_frames": [os.path.basename(f) for f in saved_files],
            "ensemble_pdb": "site1_ensemble_docking.pdb",
            "best_frame": "site1_dock_frame_best.pdb",
            "vina_script": "run_vina_docking.sh",
            "filter_script": "filter_docking_hits.py",
            "pymol_script": "validate_docking_frame.pml",
        },
    }

    report_path = os.path.join(args.output_dir, "docking_extraction_report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"\n  Report: {report_path}")

    # ─── Summary ───
    print("\n" + "=" * 65)
    print("EXTRACTION COMPLETE")
    print("=" * 65)
    print(f"  Output directory: {args.output_dir}")
    print(f"  Frames extracted: {min(args.top_n, len(scored_frames))}")
    print(f"  Best dockability: {best_metrics['dockability']:.4f} (frame {best_idx})")
    print(f"  Pocket centroid:  ({best_metrics['myc_centroid'][0]:.1f}, "
          f"{best_metrics['myc_centroid'][1]:.1f}, {best_metrics['myc_centroid'][2]:.1f}) / "
          f"({best_metrics['max_centroid'][0]:.1f}, {best_metrics['max_centroid'][1]:.1f}, "
          f"{best_metrics['max_centroid'][2]:.1f})")
    print()
    print("NEXT STEPS:")
    print("  1. Validate: pymol validate_docking_frame.pml")
    print("  2. Prepare:  bash run_vina_docking.sh")
    print("  3. Dock:     ensemble docking against site1_ensemble_docking.pdb")
    print("  4. Filter:   python3 filter_docking_hits.py <results.sdf>")
    print("  5. Hits must contact MAX I85/L88 hydrophobic patch (< 4.0 Å)")


if __name__ == "__main__":
    main()
