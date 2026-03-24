#!/usr/bin/env python3
"""PRISM4D DCCM Prototype: Dynamic Cross-Correlation Matrix"""
import json, math, glob
import numpy as np
from pathlib import Path

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

SITE_RADIUS = 12.0

def parse_traj(path, max_frames=15):
    frames = []
    cas = {}
    with open(path) as f:
        for line in f:
            if line.startswith("MODEL"):
                cas = {}
            elif line.startswith("ENDMDL"):
                if cas: frames.append(cas)
                if len(frames) >= max_frames: break
            elif line.startswith("ATOM") and line[12:16].strip() == "CA":
                try:
                    key = f"{line[21:22].strip()}:{line[22:26].strip()}"
                    x, y, z = float(line[30:38]), float(line[38:46]), float(line[46:54])
                    cas[key] = (x, y, z)
                except ValueError:
                    pass
    return frames

def compute_dccm(frames):
    if len(frames) < 3: return None, None
    common = set(frames[0].keys())
    for f in frames[1:]: common &= set(f.keys())
    keys = sorted(common)
    n = len(keys)
    if n < 10: return None, None
    coords = np.zeros((len(frames), n, 3))
    for fi, frame in enumerate(frames):
        for ri, key in enumerate(keys):
            coords[fi, ri] = frame[key]
    mean_pos = coords.mean(axis=0)
    delta = coords - mean_pos
    # Vectorized DCCM
    # <Δri · Δrj> = mean over frames of dot product
    cov = np.einsum('fid,fjd->ij', delta, delta) / len(frames)
    norms = np.sqrt(np.diag(cov))
    norms[norms < 1e-12] = 1e-12
    dccm = cov / np.outer(norms, norms)
    return dccm, keys

def site_metrics(dccm, keys, frames, centroid, radius=SITE_RADIUS):
    if dccm is None: return None
    n_frames = len(frames)
    # Use frame 0 positions directly (frames are aligned, avoids averaging bugs)
    mean_pos = {}
    for key in keys:
        if key in frames[0]:
            mean_pos[key] = frames[0][key]
    local_set = set()
    local_idx = []
    remote_idx = []
    for ri, key in enumerate(keys):
        if key not in mean_pos:
            remote_idx.append(ri)
            continue
        d = math.sqrt(sum((mean_pos[key][k]-centroid[k])**2 for k in range(3)))
        if d < radius:
            local_idx.append(ri)
            local_set.add(ri)
        else:
            remote_idx.append(ri)
    if len(local_idx) < 3 or len(remote_idx) < 3:
        print(f"    site_metrics: local={len(local_idx)} remote={len(remote_idx)} mean_pos={len(mean_pos)} keys={len(keys)} centroid={[round(c,1) for c in centroid]}")
        return None
    # Hub strength: mean |coupling| of local to remote
    hub = np.mean(np.abs(dccm[np.ix_(local_idx, remote_idx)]))
    # Internal coherence
    local_mat = dccm[np.ix_(local_idx, local_idx)]
    internal = np.mean(np.abs(local_mat[np.triu_indices(len(local_idx), k=1)]))
    # Coupling asymmetry
    remote_couplings = dccm[np.ix_(local_idx, remote_idx)]
    n_corr = np.sum(remote_couplings > 0.2)
    n_anti = np.sum(remote_couplings < -0.2)
    asymm = (n_corr - n_anti) / max(n_corr + n_anti, 1)
    return {"hub": round(float(hub), 4), "internal": round(float(internal), 4),
            "asymm": round(float(asymm), 3), "n_local": len(local_idx)}

def dcc(c, gt): return math.sqrt(sum((c[i]-gt[i])**2 for i in range(3)))

targets = ['2npq', '2hnp', '3l3n', '1jwp', '1hcl', '1nna', '1p38', '5lar']
print("=== DCCM: Global Dynamic Coupling ===\n")

for name in targets:
    print(f"Checking {name}... in_gt={name in gt_map}")
    if name not in gt_map: continue
    gt = gt_map[name]
    traj = glob.glob(f"/tmp/prism_bench10_scratch/{name}/{name}_stream00.ensemble_trajectory.pdb")
    if not traj: traj = glob.glob(f"/tmp/prism_hard_targets/{name}/{name}_stream00.ensemble_trajectory.pdb")
    if not traj: continue
    frames = parse_traj(traj[0], max_frames=10)
    print(f"  {len(frames)} frames, {len(frames[0]) if frames else 0} CAs")
    if len(frames) < 5: print("  SKIP: too few frames"); continue
    print(f"  Computing DCCM...", end="", flush=True)
    dccm_mat, keys = compute_dccm(frames)
    print(f" done ({len(keys) if keys else 0} residues)")
    if dccm_mat is None: print("  SKIP: DCCM failed"); continue

    bs = glob.glob(f"/tmp/prism_bench10_scratch/{name}/{name}.binding_sites.json")
    if not bs: bs = glob.glob(f"/tmp/prism_hard_targets/{name}/{name}.binding_sites.json")
    if not bs: print("  SKIP: no binding_sites.json"); continue
    print(f"  Loading sites from {bs[0]}")
    with open(bs[0]) as f:
        data = json.load(f)
    sites = data if isinstance(data, list) else data.get('sites', [])
    ranked = sorted(sites, key=lambda s: s.get('gtck_rank', 999))
    top1 = ranked[0]
    true_site = min(sites, key=lambda s: dcc(s.get('centroid',[0,0,0]), gt))

    m1 = site_metrics(dccm_mat, keys, frames, top1.get('centroid',[0,0,0]))
    m2 = site_metrics(dccm_mat, keys, frames, true_site.get('centroid',[0,0,0]))
    if not m1 or not m2: print(f"  SKIP: metrics failed (m1={m1 is not None} m2={m2 is not None})"); continue

    hub_r = m2['hub'] / max(m1['hub'], 1e-6)
    int_r = m2['internal'] / max(m1['internal'], 1e-6)
    sep = "SEPARATES" if hub_r > 1.15 else "INVERTED" if hub_r < 0.85 else "TIED"

    print(f"{name.upper()}:")
    print(f"  TOP1 (site {top1.get('id')}, DCC={dcc(top1.get('centroid',[0,0,0]),gt):.1f}A): hub={m1['hub']:.4f} internal={m1['internal']:.4f} asymm={m1['asymm']:+.3f}")
    print(f"  TRUE (site {true_site.get('id')}, DCC={dcc(true_site.get('centroid',[0,0,0]),gt):.1f}A): hub={m2['hub']:.4f} internal={m2['internal']:.4f} asymm={m2['asymm']:+.3f}")
    print(f"  → hub_ratio={hub_r:.2f} int_ratio={int_r:.2f} → {sep}")
    print()
