import os, json, glob, csv, re
import numpy as np

BENCH_DIR = "benchmarks/prism4d_bench30"
RESULTS_DIR = "docking_results_full"
GT_PATH = f"{BENCH_DIR}/ground_truth/ligand_centroids.json"
MANIFEST_PATH = f"{BENCH_DIR}/benchmark_manifest.json"
OUTPUT_CSV = "training_features.csv"


def parse_sdf_poses(sdf_path):
    """Parse an SDF file, extract per-pose: Vina energy, 3D centroid, radius of gyration.

    Returns list of dicts, one per pose (separated by $$$$).
    """
    poses = []
    current_atoms = []
    current_props = {}
    in_atom_block = False
    n_atoms = 0
    atom_count = 0

    with open(sdf_path) as f:
        lines = f.readlines()

    i = 0
    while i < len(lines):
        line = lines[i]

        # Counts line (V2000): "  65 68  0  0  0  0  0  0  0  0999 V2000"
        if "V2000" in line:
            parts = line.split()
            try:
                n_atoms = int(parts[0])
            except (ValueError, IndexError):
                n_atoms = 0
            atom_count = 0
            current_atoms = []
            in_atom_block = True
            i += 1
            continue

        if in_atom_block and atom_count < n_atoms:
            # Atom line: "   17.1492   36.5092   35.3206 O ..."
            parts = line.split()
            if len(parts) >= 4:
                try:
                    x, y, z = float(parts[0]), float(parts[1]), float(parts[2])
                    elem = parts[3] if len(parts) > 3 else "?"
                    # Skip hydrogens for centroid — heavy atoms only
                    if elem != "H":
                        current_atoms.append((x, y, z))
                except ValueError:
                    pass
            atom_count += 1
            i += 1
            continue

        if in_atom_block and atom_count >= n_atoms:
            in_atom_block = False

        # Property tags
        if line.startswith("> <Uni-Dock RESULT>"):
            i += 1
            if i < len(lines):
                # "ENERGY=   -5.656  LOWER_BOUND=    0.000  UPPER_BOUND=    0.000"
                m = re.search(r'ENERGY=\s*([-\d.]+)', lines[i])
                if m:
                    current_props["vina_score"] = float(m.group(1))
            i += 1
            continue

        if line.startswith("> <CNNscore>"):
            i += 1
            if i < len(lines):
                try:
                    current_props["cnn_score"] = float(lines[i].strip())
                except ValueError:
                    pass
            i += 1
            continue

        if line.startswith("> <CNNaffinity>"):
            i += 1
            if i < len(lines):
                try:
                    current_props["cnn_affinity"] = float(lines[i].strip())
                except ValueError:
                    pass
            i += 1
            continue

        if line.startswith("> <minimizedAffinity>"):
            i += 1
            if i < len(lines):
                try:
                    current_props["minimized_affinity"] = float(lines[i].strip())
                except ValueError:
                    pass
            i += 1
            continue

        # Pose separator
        if line.startswith("$$$$"):
            if current_atoms:
                coords = np.array(current_atoms)
                centroid = coords.mean(axis=0)
                # Radius of gyration: RMS distance from centroid
                dists = np.linalg.norm(coords - centroid, axis=1)
                rgyr = np.sqrt(np.mean(dists ** 2))

                current_props["pose_cx"] = centroid[0]
                current_props["pose_cy"] = centroid[1]
                current_props["pose_cz"] = centroid[2]
                current_props["pose_rgyr"] = rgyr
                current_props["pose_n_heavy"] = len(current_atoms)

            poses.append(current_props)
            current_atoms = []
            current_props = {}
            n_atoms = 0
            atom_count = 0
            i += 1
            continue

        i += 1

    return poses


def get_best_vina_from_pdbqt(unidock_out_dir):
    """Fallback: parse PDBQT files for REMARK VINA RESULT."""
    best_score = float('inf')
    for fpath in glob.glob(f"{unidock_out_dir}/*.pdbqt"):
        with open(fpath) as f:
            for line in f:
                if line.startswith("REMARK VINA RESULT:"):
                    try:
                        score = float(line.split()[3])
                        if score < best_score:
                            best_score = score
                    except (ValueError, IndexError):
                        pass
    return best_score if best_score != float('inf') else None


def main():
    with open(MANIFEST_PATH) as f:
        manifest = json.load(f)
    with open(GT_PATH) as f:
        gt_data = json.load(f)

    fieldnames = [
        'Target', 'Site_ID', 'Classification', 'Volume_A3',
        'Vina_Score', 'CNN_Score', 'CNN_Affinity',
        'Pose_Centroid_X', 'Pose_Centroid_Y', 'Pose_Centroid_Z',
        'Pose_RGyr', 'Pose_N_Heavy',
        'Pocket_Centroid_X', 'Pocket_Centroid_Y', 'Pocket_Centroid_Z',
        'Pose_Pocket_Dist',  # distance between pose centroid and pocket centroid
        'DCC', 'Is_Hit',
    ]

    with open(OUTPUT_CSV, mode='w', newline='') as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()

        total_rows = 0
        targets_with_data = 0

        for target in manifest['targets']:
            tid = str(target['id'])
            apo = target['apo_pdb'].lower()
            target_dir = os.path.join(RESULTS_DIR, apo)

            if tid not in gt_data or not os.path.exists(target_dir):
                continue

            true_centroid = np.array(gt_data[tid]['centroid'])

            # Load site metadata
            sites_json = f"{BENCH_DIR}/results/{tid}/{apo}.binding_sites.json"
            try:
                with open(sites_json) as f:
                    sites_data = json.load(f)
            except Exception:
                continue

            site_dict = {}
            for s in sites_data.get('sites', []):
                s_id = str(s.get('name', s.get('site_id', s.get('id', '')))).replace('site', '')
                cent = s.get('centroid')
                if s_id and cent:
                    site_dict[s_id] = {
                        'centroid': cent,
                        'volume': s.get('volume', 8000.0),
                        'class': s.get('classification', 'Unknown'),
                    }

            target_had_data = False

            for site_dir in glob.glob(f"{target_dir}/site*"):
                site_id = os.path.basename(site_dir).replace('site', '')
                if site_id not in site_dict:
                    continue

                pocket_centroid = np.array(site_dict[site_id]['centroid'])
                dcc = float(np.linalg.norm(pocket_centroid - true_centroid))
                is_hit = 1 if dcc <= 5.0 else 0

                # ── Try SDF output first (new pipeline) ──
                sdf_files = sorted(glob.glob(f"{site_dir}/unidock_out/*_out.sdf"))
                rescored_files = sorted(glob.glob(f"{site_dir}/gnina_rescore/*_rescored.sdf"))

                best_row = None

                # Parse rescored SDF (has both Vina + CNN scores + 3D coords)
                source_files = rescored_files if rescored_files else sdf_files
                for sf in source_files:
                    poses = parse_sdf_poses(sf)
                    for pose in poses:
                        vina = pose.get("vina_score")
                        if vina is None:
                            continue

                        pose_cx = pose.get("pose_cx")
                        pose_cy = pose.get("pose_cy")
                        pose_cz = pose.get("pose_cz")

                        # Spatial distance: pose centroid to pocket centroid
                        if pose_cx is not None:
                            pose_cent = np.array([pose_cx, pose_cy, pose_cz])
                            pose_pocket_dist = float(np.linalg.norm(pose_cent - pocket_centroid))
                        else:
                            pose_pocket_dist = None

                        row = {
                            'Target': apo.upper(),
                            'Site_ID': site_id,
                            'Classification': site_dict[site_id]['class'],
                            'Volume_A3': round(site_dict[site_id]['volume'], 2),
                            'Vina_Score': round(vina, 3),
                            'CNN_Score': round(pose.get("cnn_score", 0), 4) if pose.get("cnn_score") is not None else '',
                            'CNN_Affinity': round(pose.get("cnn_affinity", 0), 4) if pose.get("cnn_affinity") is not None else '',
                            'Pose_Centroid_X': round(pose_cx, 3) if pose_cx is not None else '',
                            'Pose_Centroid_Y': round(pose_cy, 3) if pose_cy is not None else '',
                            'Pose_Centroid_Z': round(pose_cz, 3) if pose_cz is not None else '',
                            'Pose_RGyr': round(pose.get("pose_rgyr", 0), 3) if pose.get("pose_rgyr") is not None else '',
                            'Pose_N_Heavy': pose.get("pose_n_heavy", ''),
                            'Pocket_Centroid_X': round(pocket_centroid[0], 3),
                            'Pocket_Centroid_Y': round(pocket_centroid[1], 3),
                            'Pocket_Centroid_Z': round(pocket_centroid[2], 3),
                            'Pose_Pocket_Dist': round(pose_pocket_dist, 3) if pose_pocket_dist is not None else '',
                            'DCC': round(dcc, 2),
                            'Is_Hit': is_hit,
                        }

                        # Keep only the best Vina pose per site
                        if best_row is None or vina < best_row['Vina_Score']:
                            best_row = row

                # ── Fallback: PDBQT output (old pipeline) ──
                if best_row is None:
                    vina = get_best_vina_from_pdbqt(os.path.join(site_dir, "unidock_out"))
                    if vina is not None:
                        best_row = {
                            'Target': apo.upper(),
                            'Site_ID': site_id,
                            'Classification': site_dict[site_id]['class'],
                            'Volume_A3': round(site_dict[site_id]['volume'], 2),
                            'Vina_Score': round(vina, 3),
                            'CNN_Score': '', 'CNN_Affinity': '',
                            'Pose_Centroid_X': '', 'Pose_Centroid_Y': '', 'Pose_Centroid_Z': '',
                            'Pose_RGyr': '', 'Pose_N_Heavy': '',
                            'Pocket_Centroid_X': round(pocket_centroid[0], 3),
                            'Pocket_Centroid_Y': round(pocket_centroid[1], 3),
                            'Pocket_Centroid_Z': round(pocket_centroid[2], 3),
                            'Pose_Pocket_Dist': '',
                            'DCC': round(dcc, 2),
                            'Is_Hit': is_hit,
                        }

                if best_row:
                    writer.writerow(best_row)
                    total_rows += 1
                    target_had_data = True

            if target_had_data:
                targets_with_data += 1

    print(f"Dataset: {OUTPUT_CSV}")
    print(f"  Targets: {targets_with_data}")
    print(f"  Data points: {total_rows}")
    print(f"  Features: {len(fieldnames) - 3} (+ Target, Site_ID, Is_Hit label)")
    print(f"  Spatial features: Pose_Centroid_XYZ, Pose_Pocket_Dist, Pose_RGyr")


if __name__ == "__main__":
    main()
