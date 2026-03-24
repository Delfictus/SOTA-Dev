#!/usr/bin/env python3
"""
PRISM4D — Local Contact Reorganization Prototype

Computes per-site contact reorganization signals from ensemble trajectory PDBs.
Tests whether localized contact change distinguishes true pockets from decoys.

Usage: python3 scripts/test_contact_reorg.py
"""

import json, math, os, glob
from pathlib import Path
from collections import defaultdict

# === CONFIG ===
CONTACT_CUTOFF = 6.0  # Å — CA-CA contact threshold
SITE_RADIUS = 12.0    # Å — radius around site centroid to define "local"

# Ground truth
with open('benchmarks/prism4d_bench30/ground_truth/ligand_centroids.json') as f:
    gt_raw = json.load(f)
with open('benchmarks/prism4d_bench30/benchmark_manifest.json') as f:
    mdata = json.load(f)
manifest = mdata.get('targets', mdata) if isinstance(mdata, dict) else mdata
gt_map = {}
for m in manifest:
    if isinstance(m, dict):
        pdb = m.get('apo_pdb','').lower()
        bid = str(m.get('id',''))
        if bid in gt_raw: gt_map[pdb] = gt_raw[bid].get('centroid', [0,0,0])
gt_map.update({'1p38': [16.885,-3.551,-19.552], '5lar': [54.0,68.1,16.8]})


def parse_trajectory_ca(pdb_path, max_frames=20):
    """Parse multi-model PDB, extract CA positions per frame."""
    frames = []
    current_cas = {}
    in_model = False

    with open(pdb_path) as f:
        for line in f:
            if line.startswith("MODEL"):
                in_model = True
                current_cas = {}
            elif line.startswith("ENDMDL"):
                if current_cas:
                    frames.append(current_cas)
                in_model = False
                if len(frames) >= max_frames:
                    break
            elif line.startswith("ATOM") and line[12:16].strip() == "CA":
                chain = line[21:22].strip()
                resi = line[22:26].strip()
                key = f"{chain}:{resi}"
                x = float(line[30:38])
                y = float(line[38:46])
                z = float(line[46:54])
                current_cas[key] = (x, y, z)

    return frames


def compute_contacts(cas, cutoff=CONTACT_CUTOFF):
    """Compute contact set: pairs of residues within cutoff."""
    keys = sorted(cas.keys())
    contacts = set()
    for i in range(len(keys)):
        for j in range(i+2, len(keys)):  # skip adjacent
            p1 = cas[keys[i]]
            p2 = cas[keys[j]]
            d = math.sqrt(sum((p1[k]-p2[k])**2 for k in range(3)))
            if d < cutoff:
                contacts.add((keys[i], keys[j]))
    return contacts


def local_contacts(contacts, cas, centroid, radius=SITE_RADIUS):
    """Filter contacts to those with at least one residue near the centroid."""
    local = set()
    for (r1, r2) in contacts:
        if r1 not in cas or r2 not in cas:
            continue
        p1 = cas[r1]
        p2 = cas[r2]
        d1 = math.sqrt(sum((p1[k]-centroid[k])**2 for k in range(3)))
        d2 = math.sqrt(sum((p2[k]-centroid[k])**2 for k in range(3)))
        if d1 < radius or d2 < radius:
            local.add((r1, r2))
    return local


def contact_reorg_metrics(frames, centroid):
    """Compute contact reorganization metrics for a site across trajectory frames."""
    if len(frames) < 3:
        return None

    # Use first frame as reference, compare to later frames
    ref_contacts = compute_contacts(frames[0])
    ref_local = local_contacts(ref_contacts, frames[0], centroid)

    # Track contact changes across frames
    formed_local = []  # contacts formed locally (not in reference)
    broken_local = []  # contacts broken locally
    total_formed = []
    total_broken = []

    for i in range(1, len(frames)):
        frame_contacts = compute_contacts(frames[i])
        frame_local = local_contacts(frame_contacts, frames[i], centroid)

        # Global changes
        formed = frame_contacts - ref_contacts
        broken = ref_contacts - frame_contacts
        total_formed.append(len(formed))
        total_broken.append(len(broken))

        # Local changes (near this site)
        local_formed = local_contacts(formed, frames[i], centroid)
        local_broken = local_contacts(broken, frames[0], centroid)
        formed_local.append(len(local_formed))
        broken_local.append(len(local_broken))

    n_frames = len(frames) - 1

    # 1. Contact change density: mean local contacts formed + broken per frame
    mean_local_change = sum(f + b for f, b in zip(formed_local, broken_local)) / n_frames

    # 2. Localization ratio: fraction of total contact change happening locally
    total_change = sum(f + b for f, b in zip(total_formed, total_broken))
    local_change = sum(f + b for f, b in zip(formed_local, broken_local))
    localization_ratio = local_change / max(total_change, 1)

    # 3. Persistence: are new local contacts maintained across frames?
    # Check if contacts formed in early frames persist in late frames
    if len(frames) >= 6:
        early_formed = compute_contacts(frames[2]) - ref_contacts
        early_local = local_contacts(early_formed, frames[2], centroid)
        late_contacts = compute_contacts(frames[-1])
        persisted = early_local & late_contacts
        persistence = len(persisted) / max(len(early_local), 1)
    else:
        persistence = 0.0

    # 4. Boundary emergence: does the local contact count increase (pocket wall forming)?
    local_contact_counts = [len(local_contacts(compute_contacts(frames[i]), frames[i], centroid))
                           for i in [0, len(frames)//2, -1]]
    if local_contact_counts[0] > 0:
        boundary_growth = (local_contact_counts[-1] - local_contact_counts[0]) / local_contact_counts[0]
    else:
        boundary_growth = 0.0

    return {
        "contact_change_density": round(mean_local_change, 2),
        "localization_ratio": round(localization_ratio, 4),
        "persistence": round(persistence, 3),
        "boundary_growth": round(boundary_growth, 3),
        "n_frames": n_frames,
    }


def dcc(c, gt):
    return math.sqrt(sum((c[i]-gt[i])**2 for i in range(3)))


def main():
    targets = ['2hnp', '2npq', '3l3n', '1jwp', '1hcl', '1nna', '1p38', '5lar']

    print("=== LOCAL CONTACT REORGANIZATION PROTOTYPE ===")
    print(f"Contact cutoff: {CONTACT_CUTOFF}Å, Site radius: {SITE_RADIUS}Å\n")

    for name in targets:
        if name not in gt_map:
            continue
        gt = gt_map[name]

        # Find trajectory
        traj_paths = glob.glob(f"/tmp/prism_bench10_scratch/{name}/{name}_stream00.ensemble_trajectory.pdb")
        if not traj_paths:
            traj_paths = glob.glob(f"/tmp/prism_hard_targets/{name}/{name}_stream00.ensemble_trajectory.pdb")
        if not traj_paths:
            print(f"{name.upper()}: no trajectory found")
            continue

        print(f"{name.upper()}: loading trajectory...")
        frames = parse_trajectory_ca(traj_paths[0], max_frames=10)
        print(f"  {len(frames)} frames loaded ({len(frames[0]) if frames else 0} CAs)")

        # Load binding sites
        bs_paths = glob.glob(f"/tmp/prism_bench10_scratch/{name}/{name}.binding_sites.json")
        if not bs_paths:
            bs_paths = glob.glob(f"/tmp/prism_hard_targets/{name}/{name}.binding_sites.json")
        if not bs_paths:
            continue

        with open(bs_paths[0]) as f:
            data = json.load(f)
        sites = data if isinstance(data, list) else data.get('sites', [])

        # Find top-1 and true pocket
        ranked = sorted(sites, key=lambda s: s.get('gtck_rank', 999))
        top1 = ranked[0]
        true_site = min(sites, key=lambda s: dcc(s.get('centroid',[0,0,0]), gt))

        print(f"  Top-1: site {top1.get('id')} DCC={dcc(top1.get('centroid',[0,0,0]), gt):.1f}Å")
        print(f"  True:  site {true_site.get('id')} DCC={dcc(true_site.get('centroid',[0,0,0]), gt):.1f}Å")

        # Compute contact reorg for both
        for label, s in [("TOP1", top1), ("TRUE", true_site)]:
            centroid = s.get('centroid', [0, 0, 0])
            metrics = contact_reorg_metrics(frames, centroid)
            if metrics:
                print(f"  {label}: change_density={metrics['contact_change_density']:.1f} "
                      f"local_ratio={metrics['localization_ratio']:.3f} "
                      f"persistence={metrics['persistence']:.2f} "
                      f"boundary_growth={metrics['boundary_growth']:+.3f}")
            else:
                print(f"  {label}: insufficient frames")

        print()


if __name__ == "__main__":
    main()
