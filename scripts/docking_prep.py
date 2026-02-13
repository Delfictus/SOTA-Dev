#!/usr/bin/env python3
"""
PRISM-4D Docking Preparation Pipeline
Generates everything needed for small molecule docking from PRISM outputs.

Outputs per site:
  1. Vina config file (.txt)
  2. Docking box PDB (visualization)
  3. Pharmacophore-guided fragment recommendations
  4. Comprehensive docking report (.md)
  5. Open-state receptor PDB
"""
import json
import math
import sys
import os
import glob
import numpy as np
from collections import Counter

def load_json(path):
    with open(path) as f:
        return json.load(f)

def compute_docking_box(centroid, lining_residues, spike_data, padding=4.0):
    """Compute optimal docking box from spike extent + lining residues."""
    if spike_data and 'spikes' in spike_data:
        xs = [s['x'] for s in spike_data['spikes']]
        ys = [s['y'] for s in spike_data['spikes']]
        zs = [s['z'] for s in spike_data['spikes']]
        # Use 5th-95th percentile to exclude outliers
        px = sorted(xs)
        py = sorted(ys)
        pz = sorted(zs)
        n = len(px)
        lo, hi = int(n * 0.05), int(n * 0.95)
        size_x = px[hi] - px[lo] + padding * 2
        size_y = py[hi] - py[lo] + padding * 2
        size_z = pz[hi] - pz[lo] + padding * 2
    else:
        # Fallback: cube from volume
        side = 20.0
        size_x = size_y = size_z = side

    return {
        'center_x': round(centroid[0], 3),
        'center_y': round(centroid[1], 3),
        'center_z': round(centroid[2], 3),
        'size_x': round(min(size_x, 40.0), 1),  # Vina max ~40
        'size_y': round(min(size_y, 40.0), 1),
        'size_z': round(min(size_z, 40.0), 1),
    }

def pharmacophore_profile(spike_data):
    """Analyze spike composition for fragment selection guidance."""
    if not spike_data or 'spikes' not in spike_data:
        return {}

    spikes = spike_data['spikes']
    n = len(spikes)
    types = Counter(s['type'] for s in spikes)
    sources = Counter(s['spike_source'] for s in spikes)

    # Intensity stats per type
    type_stats = {}
    for t in types:
        t_spikes = [s for s in spikes if s['type'] == t]
        intensities = [s['intensity'] for s in t_spikes]
        type_stats[t] = {
            'count': len(t_spikes),
            'pct': round(100 * len(t_spikes) / n, 1),
            'mean_intensity': round(np.mean(intensities), 2),
            'max_intensity': round(np.max(intensities), 2),
        }

    # Pocket character
    aromatic_pct = sum(types.get(t, 0) for t in ['BNZ', 'TRP', 'TYR', 'PHE']) / max(n, 1) * 100
    polar_pct = sum(types.get(t, 0) for t in ['CATION', 'ANION']) / max(n, 1) * 100
    lif_pct = types.get('UNK', 0) / max(n, 1) * 100

    # Water density at spike sites (desolvation estimate)
    water_densities = [s.get('water_density', 0) for s in spikes]
    mean_water = np.mean(water_densities) if water_densities else 0

    # Frame distribution (ensemble diversity)
    frames = Counter(s['frame_index'] for s in spikes)
    n_frames = len(frames)

    return {
        'total_spikes': n,
        'type_breakdown': type_stats,
        'source_breakdown': dict(sources),
        'aromatic_pct': round(aromatic_pct, 1),
        'polar_pct': round(polar_pct, 1),
        'water_lif_pct': round(lif_pct, 1),
        'mean_water_density': round(mean_water, 3),
        'n_frames_sampled': n_frames,
        'pocket_character': (
            'Hydrophobic' if aromatic_pct > 70 else
            'Mixed hydrophobic/polar' if aromatic_pct > 40 else
            'Polar-dominated' if polar_pct > 30 else
            'Water-mediated'
        ),
    }

def fragment_recommendations(profile, lining_residues):
    """Suggest fragment types based on pharmacophore profile."""
    recs = []
    ts = profile.get('type_breakdown', {})

    if ts.get('BNZ', {}).get('pct', 0) > 20:
        recs.append({
            'type': 'Aromatic/hydrophobic',
            'rationale': f'BNZ hotspots dominate ({ts["BNZ"]["pct"]}%) — aromatic rings, halogenated phenyls',
            'examples': 'benzene, naphthalene, indole, chlorobenzene, toluene',
            'priority': 'HIGH'
        })

    if ts.get('TYR', {}).get('pct', 0) > 2 or ts.get('TRP', {}).get('pct', 0) > 2:
        recs.append({
            'type': 'H-bond donor/acceptor',
            'rationale': f'TYR ({ts.get("TYR",{}).get("pct",0)}%) + TRP ({ts.get("TRP",{}).get("pct",0)}%) indicate H-bond network',
            'examples': 'phenol, indole, hydroxamic acid, aminopyridine',
            'priority': 'MEDIUM'
        })

    if ts.get('CATION', {}).get('pct', 0) > 0.5:
        recs.append({
            'type': 'Cation-targeting (acidic fragments)',
            'rationale': f'CATION spikes ({ts["CATION"]["pct"]}%) near LYS/ARG — acidic groups for salt bridge',
            'examples': 'carboxylic acids, tetrazole, phosphonate, sulfonamide',
            'priority': 'MEDIUM'
        })

    if ts.get('ANION', {}).get('pct', 0) > 0.5:
        recs.append({
            'type': 'Anion-targeting (basic fragments)',
            'rationale': f'ANION spikes ({ts["ANION"]["pct"]}%) near GLU/ASP — basic groups for salt bridge',
            'examples': 'amidine, guanidine, piperazine, aminopyridine',
            'priority': 'MEDIUM'
        })

    if profile.get('mean_water_density', 0) > 0.7:
        recs.append({
            'type': 'Water-displacing',
            'rationale': f'High water density ({profile["mean_water_density"]:.2f}) — structured waters to displace',
            'examples': 'cyclopropyl, gem-dimethyl, trifluoromethyl (entropy-driven displacement)',
            'priority': 'LOW'
        })

    # Cysteine-reactive if CYS in lining
    cys_residues = [r for r in lining_residues if r['resname'] == 'CYS']
    if cys_residues:
        recs.append({
            'type': 'Covalent warhead (CYS-reactive)',
            'rationale': f'CYS{cys_residues[0]["resid"]} at {cys_residues[0]["min_distance"]:.1f}Å — covalent binding possible',
            'examples': 'acrylamide, chloroacetamide, vinyl sulfonamide, α,β-unsaturated carbonyl',
            'priority': 'HIGH' if cys_residues[0]['min_distance'] < 4.0 else 'MEDIUM'
        })

    return sorted(recs, key=lambda r: {'HIGH': 0, 'MEDIUM': 1, 'LOW': 2}[r['priority']])

def find_open_frame(spike_data, centroid):
    """Find frame with highest spike density near centroid."""
    if not spike_data:
        return 0, {}
    spikes = spike_data['spikes']
    cx, cy, cz = centroid
    frame_scores = {}
    for s in spikes:
        fi = s['frame_index']
        dx = s['x'] - cx
        dy = s['y'] - cy
        dz = s['z'] - cz
        dist = math.sqrt(dx*dx + dy*dy + dz*dz)
        if dist < 6.0:
            frame_scores[fi] = frame_scores.get(fi, 0) + s['intensity']
    if not frame_scores:
        return 0, {}
    best = max(frame_scores, key=frame_scores.get)
    return best, frame_scores

def extract_frame(results_dir, structure_name, frame_idx, output_path):
    """Extract open-state frame from ensemble trajectory."""
    trajs = sorted(glob.glob(os.path.join(results_dir, f"{structure_name}_stream*.ensemble_trajectory.pdb")))
    if not trajs:
        trajs = sorted(glob.glob(os.path.join(results_dir, f"{structure_name}_stream*.ensemble*.pdb")))
    if not trajs:
        return False
    traj_path = trajs[0]  # Use stream 0

    models = []
    current = []
    in_model = False
    with open(traj_path) as f:
        for line in f:
            if line.startswith('MODEL'):
                in_model = True
                current = []
            elif line.startswith('ENDMDL'):
                models.append(current)
                in_model = False
            elif in_model:
                current.append(line)

    if frame_idx < len(models):
        with open(output_path, 'w') as f:
            f.write(f"REMARK PRISM-4D open-state receptor (frame {frame_idx})\n")
            f.write(f"REMARK Selected as highest spike-density frame near pocket centroid\n")
            f.write(f"REMARK Ready for docking — add hydrogens and convert to PDBQT\n")
            for line in models[frame_idx]:
                f.write(line)
            f.write("END\n")
        return True
    return False

def write_vina_config(box, site_id, output_path, receptor_name):
    """Write AutoDock Vina configuration file."""
    with open(output_path, 'w') as f:
        f.write(f"# PRISM-4D AutoDock Vina Configuration — Site {site_id}\n")
        f.write(f"# Generated from spike-derived docking box\n\n")
        f.write(f"receptor = {receptor_name}.pdbqt\n")
        f.write(f"ligand = ligand.pdbqt\n\n")
        f.write(f"center_x = {box['center_x']}\n")
        f.write(f"center_y = {box['center_y']}\n")
        f.write(f"center_z = {box['center_z']}\n\n")
        f.write(f"size_x = {box['size_x']}\n")
        f.write(f"size_y = {box['size_y']}\n")
        f.write(f"size_z = {box['size_z']}\n\n")
        f.write(f"exhaustiveness = 32\n")
        f.write(f"num_modes = 20\n")
        f.write(f"energy_range = 4\n")

def write_docking_report(site, profile, box, frags, open_frame, frame_scores, site_data, output_path):
    """Write comprehensive docking preparation report."""
    with open(output_path, 'w') as f:
        f.write(f"# PRISM-4D Docking Preparation Report\n")
        f.write(f"## Site {site['id']} — {site['classification']}\n\n")

        # === DOCKING BOX ===
        f.write(f"## Docking Box (AutoDock Vina)\n")
        f.write(f"| Parameter | Value |\n|---|---|\n")
        f.write(f"| Center (Å) | ({box['center_x']}, {box['center_y']}, {box['center_z']}) |\n")
        f.write(f"| Size (Å) | {box['size_x']} × {box['size_y']} × {box['size_z']} |\n")
        f.write(f"| Volume (Å³) | {site['volume']:.0f} |\n")
        f.write(f"| Druggable | {'Yes' if site['is_druggable'] else 'No'} |\n")
        f.write(f"| Quality Score | {site['quality_score']:.3f} |\n\n")

        # === POCKET CHARACTER ===
        f.write(f"## Pocket Character\n")
        f.write(f"**{profile.get('pocket_character', 'Unknown')}**\n\n")
        f.write(f"| Property | Value |\n|---|---|\n")
        f.write(f"| Aromatic content | {profile.get('aromatic_pct', 0)}% |\n")
        f.write(f"| Polar content | {profile.get('polar_pct', 0)}% |\n")
        f.write(f"| Water/LIF content | {profile.get('water_lif_pct', 0)}% |\n")
        f.write(f"| Mean water density | {profile.get('mean_water_density', 0):.3f} |\n")
        f.write(f"| Desolvation | {'Challenging' if profile.get('mean_water_density', 0) > 0.7 else 'Favorable'} |\n")
        f.write(f"| Ensemble frames | {profile.get('n_frames_sampled', 0)} |\n\n")

        # === DETECTION CHANNELS ===
        f.write(f"## Detection Channel Breakdown\n")
        f.write(f"| Source | Spikes | % |\n|---|---|---|\n")
        total = profile.get('total_spikes', 1)
        for src, count in sorted(profile.get('source_breakdown', {}).items(), key=lambda x: -x[1]):
            f.write(f"| {src} | {count:,} | {100*count/total:.1f}% |\n")
        f.write(f"| **Total** | **{total:,}** | |\n\n")

        # === PHARMACOPHORE BREAKDOWN ===
        f.write(f"## Pharmacophore Spike Profile\n")
        f.write(f"| Type | Count | % | Mean Intensity | Max Intensity |\n")
        f.write(f"|---|---|---|---|---|\n")
        for t, stats in sorted(profile.get('type_breakdown', {}).items(), key=lambda x: -x[1]['count']):
            f.write(f"| {t} | {stats['count']:,} | {stats['pct']}% | {stats['mean_intensity']} | {stats['max_intensity']} |\n")
        f.write(f"\n")

        # === FRAGMENT RECOMMENDATIONS ===
        f.write(f"## Fragment Library Recommendations\n")
        for i, frag in enumerate(frags, 1):
            f.write(f"### {i}. {frag['type']} [{frag['priority']}]\n")
            f.write(f"**Rationale:** {frag['rationale']}\n\n")
            f.write(f"**Examples:** {frag['examples']}\n\n")

        # === OPEN-STATE RECEPTOR ===
        f.write(f"## Receptor Selection\n")
        f.write(f"| Property | Value |\n|---|---|\n")
        f.write(f"| Open-state frame | {open_frame} |\n")
        if frame_scores:
            top5 = sorted(frame_scores.items(), key=lambda x: -x[1])[:5]
            f.write(f"| Top frames (by density) | {', '.join(f'{fi}({sc:.0f})' for fi, sc in top5)} |\n")
        f.write(f"| Source trajectory | Stream 0 |\n\n")

        # === LINING RESIDUES ===
        f.write(f"## Binding Site Residues ({len(site['lining_residues'])} residues)\n")
        f.write(f"| Chain | ResID | Name | Distance (Å) | Catalytic | Character |\n")
        f.write(f"|---|---|---|---|---|---|\n")
        polar_res = {'GLU', 'ASP', 'LYS', 'ARG', 'HIS', 'SER', 'THR', 'ASN', 'GLN', 'CYS'}
        aromatic_res = {'PHE', 'TRP', 'TYR', 'HIS'}
        hydrophobic_res = {'ALA', 'VAL', 'LEU', 'ILE', 'MET', 'PRO'}
        for r in site['lining_residues']:
            name = r['resname']
            char = 'Polar' if name in polar_res else 'Aromatic' if name in aromatic_res else 'Hydrophobic' if name in hydrophobic_res else 'Other'
            cat = '✓' if r.get('is_catalytic') else ''
            f.write(f"| {r['chain']} | {r['resid']} | {name} | {r['min_distance']:.1f} | {cat} | {char} |\n")

        # Residue composition summary
        res_chars = Counter()
        for r in site['lining_residues']:
            name = r['resname']
            if name in polar_res: res_chars['Polar'] += 1
            elif name in aromatic_res: res_chars['Aromatic'] += 1
            elif name in hydrophobic_res: res_chars['Hydrophobic'] += 1
            else: res_chars['Other'] += 1
        f.write(f"\n**Composition:** {dict(res_chars)}\n\n")

        # === DOCKING COMMANDS ===
        structure = site_data.get('structure', 'structure').replace('.topology', '')
        f.write(f"## Quick-Start Docking Commands\n")
        f.write(f"```bash\n")
        f.write(f"# 1. Prepare receptor (add hydrogens, compute charges)\n")
        f.write(f"obabel {structure}_open_frame{open_frame}.pdb -O receptor.pdbqt -xr\n\n")
        f.write(f"# 2. Prepare ligand\n")
        f.write(f"obabel ligand.sdf -O ligand.pdbqt --gen3d\n\n")
        f.write(f"# 3. Run Vina\n")
        f.write(f"vina --config vina_site{site['id']}.txt\n\n")
        f.write(f"# 4. Visualize in PyMOL\n")
        f.write(f"pymol @site{site['id']}_visualize.pml\n")
        f.write(f"```\n")

def main():
    if len(sys.argv) < 2:
        print("Usage: docking_prep.py <results_dir> [structure_name]")
        sys.exit(1)

    results_dir = sys.argv[1]

    # Find binding_sites.json
    bs_files = glob.glob(os.path.join(results_dir, "*.binding_sites.json"))
    if not bs_files:
        print("Error: No binding_sites.json found")
        sys.exit(1)

    bs_path = bs_files[0]
    site_data = load_json(bs_path)
    structure = site_data.get('structure', 'structure').replace('.topology', '')

    print(f"PRISM-4D Docking Preparation")
    print(f"  Structure: {structure}")
    print(f"  Sites: {site_data['binding_sites']}")
    print()

    for site in site_data['sites']:
        sid = site['id']
        centroid = site['centroid']
        print(f"  === Site {sid}: {site['classification']} ===")

        # Load spike events
        spike_path = os.path.join(results_dir, f"{structure}.site{sid}.spike_events.json")
        spike_data = None
        if os.path.exists(spike_path):
            print(f"  Loading spike events ({os.path.getsize(spike_path) / 1e6:.0f}MB)...")
            spike_data = load_json(spike_path)

        # 1. Pharmacophore profile
        profile = pharmacophore_profile(spike_data)
        print(f"  Pocket: {profile.get('pocket_character', '?')} ({profile.get('aromatic_pct',0)}% aromatic, {profile.get('polar_pct',0)}% polar)")

        # 2. Docking box
        box = compute_docking_box(centroid, site['lining_residues'], spike_data)
        print(f"  Box: {box['size_x']}×{box['size_y']}×{box['size_z']} Å at ({box['center_x']}, {box['center_y']}, {box['center_z']})")

        # 3. Fragment recommendations
        frags = fragment_recommendations(profile, site['lining_residues'])
        print(f"  Fragments: {len(frags)} recommendations")
        for frag in frags:
            print(f"    [{frag['priority']}] {frag['type']}")

        # 4. Open-state frame
        open_frame, frame_scores = find_open_frame(spike_data, centroid)
        receptor_path = os.path.join(results_dir, f"{structure}_open_frame{open_frame}.pdb")
        if extract_frame(results_dir, structure, open_frame, receptor_path):
            print(f"  Receptor: frame {open_frame} → {os.path.basename(receptor_path)}")
        else:
            print(f"  Warning: Could not extract frame {open_frame}")

        # 5. Write Vina config
        vina_path = os.path.join(results_dir, f"vina_site{sid}.txt")
        write_vina_config(box, sid, vina_path, f"{structure}_open_frame{open_frame}")
        print(f"  Vina config: {os.path.basename(vina_path)}")

        # 6. Write comprehensive report
        report_path = os.path.join(results_dir, f"site{sid}_docking_report.md")
        write_docking_report(site, profile, box, frags, open_frame, frame_scores, site_data, report_path)
        print(f"  Report: {os.path.basename(report_path)}")
        print()

    print("Done! Ready for docking.")

if __name__ == '__main__':
    main()
