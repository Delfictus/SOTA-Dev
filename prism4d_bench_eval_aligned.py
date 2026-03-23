#!/usr/bin/env python3
"""
PRISM-4D Benchmark Evaluation — Kabsch-Aligned
================================================
The ground truth centroids are in the apo/holo crystal frame.
PRISM-4D pocket centroids are in the simulation frame (post-MD drift).
This script aligns simulation CAs back to the apo reference frame,
applies the same transform to pocket centroids, then computes DCC.

Requires: numpy (for Kabsch SVD)

Usage:
    python3 prism4d_bench_eval_aligned.py
    
Run from ~/Desktop/Prism4D-bio/
"""

import json, glob, os, math, sys, re
import numpy as np

# --- CONFIG ---
BASE = os.path.expanduser("~/Desktop/Prism4D-bio/benchmarks/prism4d_bench30")
RESULTS = os.path.join(BASE, "results")
GT_FILE = os.path.join(BASE, "ground_truth", "ligand_centroids.json")
MANIFEST = os.path.join(BASE, "benchmark_manifest.json")
APO_DIR = os.path.join(BASE, "apo")

def dist(a, b):
    return math.sqrt(sum((x - y)**2 for x, y in zip(a, b)))

def parse_pdb_ca(filepath):
    """Extract CA atom coordinates from a PDB file.
    Returns list of (resid, x, y, z) tuples."""
    cas = []
    with open(filepath) as f:
        for line in f:
            if (line.startswith("ATOM") or line.startswith("HETATM")):
                atom_name = line[12:16].strip()
                if atom_name == "CA":
                    try:
                        x = float(line[30:38])
                        y = float(line[38:46])
                        z = float(line[46:54])
                        resid = int(line[22:26].strip())
                        cas.append((resid, x, y, z))
                    except (ValueError, IndexError):
                        continue
    return cas

def parse_trajectory_ca(filepath, frame=-1):
    """Extract CA coordinates from an ensemble trajectory PDB.
    If frame=-1, use the last MODEL. If frame=0, use the first.
    For averaging, set frame='mean' to average across all frames."""
    models = []
    current_cas = []
    in_model = False
    
    with open(filepath) as f:
        for line in f:
            if line.startswith("MODEL"):
                in_model = True
                current_cas = []
            elif line.startswith("ENDMDL"):
                if current_cas:
                    models.append(current_cas)
                in_model = False
            elif (line.startswith("ATOM") or line.startswith("HETATM")):
                atom_name = line[12:16].strip()
                if atom_name == "CA":
                    try:
                        x = float(line[30:38])
                        y = float(line[38:46])
                        z = float(line[46:54])
                        resid = int(line[22:26].strip())
                        current_cas.append((resid, x, y, z))
                    except (ValueError, IndexError):
                        continue
    
    # Handle files without MODEL/ENDMDL
    if not models and current_cas:
        models.append(current_cas)
    
    if not models:
        return []
    
    if frame == 'mean':
        # Average across all frames
        n_atoms = len(models[0])
        avg = []
        for i in range(n_atoms):
            resid = models[0][i][0]
            mx = np.mean([m[i][1] for m in models if len(m) > i])
            my = np.mean([m[i][2] for m in models if len(m) > i])
            mz = np.mean([m[i][3] for m in models if len(m) > i])
            avg.append((resid, mx, my, mz))
        return avg
    elif frame == -1:
        return models[-1]
    else:
        return models[min(frame, len(models)-1)]

def kabsch_align(P, Q):
    """Compute optimal rotation+translation to align P onto Q.
    P, Q are Nx3 numpy arrays. Returns (R, t) such that Q ≈ R @ P + t.
    Also returns RMSD."""
    assert P.shape == Q.shape
    
    # Center both
    centroid_P = P.mean(axis=0)
    centroid_Q = Q.mean(axis=0)
    P_centered = P - centroid_P
    Q_centered = Q - centroid_Q
    
    # SVD of covariance matrix
    H = P_centered.T @ Q_centered
    U, S, Vt = np.linalg.svd(H)
    
    # Ensure right-handed coordinate system
    d = np.linalg.det(Vt.T @ U.T)
    sign_matrix = np.diag([1, 1, np.sign(d)])
    
    R = Vt.T @ sign_matrix @ U.T
    t = centroid_Q - R @ centroid_P
    
    # RMSD
    P_aligned = (R @ P.T).T + t
    rmsd = np.sqrt(np.mean(np.sum((P_aligned - Q)**2, axis=1)))
    
    return R, t, rmsd

def transform_point(point, R, t):
    """Apply rotation and translation to a 3D point."""
    p = np.array(point)
    return (R @ p + t).tolist()

def match_cas(ref_cas, sim_cas):
    """Match CA atoms by residue ID. Returns paired coordinate arrays."""
    ref_dict = {resid: (x, y, z) for resid, x, y, z in ref_cas}
    sim_dict = {resid: (x, y, z) for resid, x, y, z in sim_cas}
    
    common = sorted(set(ref_dict.keys()) & set(sim_dict.keys()))
    if len(common) < 3:
        return None, None, common
    
    ref_coords = np.array([ref_dict[r] for r in common])
    sim_coords = np.array([sim_dict[r] for r in common])
    return ref_coords, sim_coords, common

def rank_sites(sites):
    """Rank using volume * druggability (baseline).
    Override this function to test other rankers."""
    def score(s):
        v = s.get('volume', s.get('mean_volume', 0))
        d = s.get('druggability', 0)
        return v * d
    return sorted(sites, key=score, reverse=True)


# ============================================================
# MAIN
# ============================================================

# Load ground truth
with open(GT_FILE) as f:
    gt = json.load(f)

# Load manifest
manifest_targets = {}
try:
    with open(MANIFEST) as f:
        mf = json.load(f)
        for t in mf.get('targets', []):
            manifest_targets[str(t['id'])] = t
except:
    pass

print("=" * 100)
print("  PRISM-4D BENCHMARK — KABSCH-ALIGNED EVALUATION")
print("=" * 100)

# Candidate rankers to test
ranker_fns = {
    'vol*drug': lambda s: s.get('volume', 0) * s.get('druggability', 0),
    'quality': lambda s: s.get('quality_score', 0),
    'spike*drug': lambda s: s.get('spike_count', 0) * s.get('druggability', 0),
    'eng_chem*drug': lambda s: s.get('engine_chem', 0) * s.get('druggability', 0),
    'drug*burial*spher': lambda s: s.get('druggability', 0) * s.get('burial_score', 0) * s.get('sphericity', 0),
    'catalytic*drug': lambda s: s.get('catalytic_residue_count', 0) * s.get('druggability', 0),
    'eng_chem*burial': lambda s: s.get('engine_chem', 0) * s.get('burial_score', 0),
}

# Storage for all ranker results
ranker_results = {name: {'sr1_4': 0, 'sr1_5': 0, 'sr3_5': 0, 'dccs': [], 'n': 0} 
                  for name in ranker_fns}

# Per-target detail (using vol*drug as primary for reporting)
detail_results = []

# Track alignment stats
align_stats = {'success': 0, 'fail': 0, 'rmsds': []}

print(f"\n{'#':<3} {'APO':<10} {'RMSD':>6} {'nCA':>5} | ", end="")
print(f"{'DCC_raw':>8} {'DCC_aln':>8} {'best3':>7} {'best10':>7} | ", end="")
print(f"{'TAU':>5} {'DRUG':>5} {'THERM':<8} {'TYPE':<7} {'R1<5':>4}")
print("─" * 110)

for folder_id in sorted(os.listdir(RESULTS), key=lambda x: int(x) if x.isdigit() else 9999):
    folder_path = os.path.join(RESULTS, folder_id)
    if not os.path.isdir(folder_path):
        continue

    # Find files
    bs_files = glob.glob(os.path.join(folder_path, "*.binding_sites.json"))
    traj_files = sorted(glob.glob(os.path.join(folder_path, "*_stream00.ensemble_trajectory.pdb")))
    
    if not bs_files:
        continue

    gt_entry = gt.get(folder_id)
    if not gt_entry:
        continue
    gt_centroid = gt_entry['centroid']

    # Get metadata
    meta = manifest_targets.get(folder_id, {})
    apo_pdb_code = meta.get('apo_pdb', '').lower()
    
    # Find the apo PDB file
    apo_file = None
    if meta.get('apo_file'):
        candidate = os.path.join(BASE, meta['apo_file'])
        if os.path.exists(candidate):
            apo_file = candidate
    if not apo_file:
        # Try to find by pattern
        candidates = glob.glob(os.path.join(APO_DIR, f"{apo_pdb_code}*"))
        if candidates:
            apo_file = candidates[0]
    if not apo_file:
        # Try from the binding_sites filename
        bs_name = os.path.basename(bs_files[0]).replace('.binding_sites.json', '')
        candidates = glob.glob(os.path.join(APO_DIR, f"{bs_name}*"))
        if candidates:
            apo_file = candidates[0]
    
    try:
        # Load sites
        with open(bs_files[0]) as f:
            bs_data = json.load(f)
        sites = bs_data.get('sites', [])
        if not sites:
            continue

        # --- ALIGNMENT ---
        R, t, align_rmsd = None, None, None
        n_matched = 0
        
        if apo_file and traj_files:
            # Parse apo CAs (reference frame)
            ref_cas = parse_pdb_ca(apo_file)
            
            # Parse simulation CAs (use first trajectory, last frame)
            sim_cas = parse_trajectory_ca(traj_files[0], frame=-1)
            
            if ref_cas and sim_cas:
                ref_coords, sim_coords, common_res = match_cas(ref_cas, sim_cas)
                
                if ref_coords is not None and len(common_res) >= 10:
                    R, t, align_rmsd = kabsch_align(sim_coords, ref_coords)
                    n_matched = len(common_res)
                    align_stats['success'] += 1
                    align_stats['rmsds'].append(align_rmsd)
                else:
                    align_stats['fail'] += 1
            else:
                align_stats['fail'] += 1
        else:
            align_stats['fail'] += 1

        # --- COMPUTE DCC (both raw and aligned) ---
        for rname, rfn in ranker_fns.items():
            ranked = sorted(sites, key=rfn, reverse=True)
            
            dccs_raw = []
            dccs_aligned = []
            for s in ranked[:10]:
                c = s.get('centroid', [0, 0, 0])
                dccs_raw.append(dist(c, gt_centroid))
                
                if R is not None:
                    c_aligned = transform_point(c, R, t)
                    dccs_aligned.append(dist(c_aligned, gt_centroid))
                else:
                    dccs_aligned.append(dist(c, gt_centroid))
            
            dcc_use = dccs_aligned  # Use aligned DCCs
            
            rr = ranker_results[rname]
            rr['n'] += 1
            rr['dccs'].append(dcc_use[0])
            if dcc_use[0] < 4.0: rr['sr1_4'] += 1
            if dcc_use[0] < 5.0: rr['sr1_5'] += 1
            if min(dcc_use[:3]) < 5.0: rr['sr3_5'] += 1
        
        # Detail line (vol*drug ranker)
        primary_ranked = sorted(sites, key=ranker_fns['vol*drug'], reverse=True)
        raw_dcc1 = dist(primary_ranked[0]['centroid'], gt_centroid)
        
        if R is not None:
            aln_centroids = [transform_point(s['centroid'], R, t) for s in primary_ranked[:10]]
            aln_dccs = [dist(c, gt_centroid) for c in aln_centroids]
        else:
            aln_dccs = [dist(s['centroid'], gt_centroid) for s in primary_ranked[:10]]
        
        top = primary_ranked[0]
        r1_hit = "✓" if aln_dccs[0] < 5.0 else "✗"
        
        pdb_label = meta.get('apo_pdb', os.path.basename(bs_files[0]).split('.')[0])
        
        print(f"{folder_id:<3} {pdb_label:<10} {align_rmsd if align_rmsd is not None else -1:>6.2f} {n_matched:>5} | "
              f"{raw_dcc1:>8.1f} {aln_dccs[0]:>8.1f} {min(aln_dccs[:3]):>7.1f} {min(aln_dccs[:10]):>7.1f} | "
              f"{top.get('ccns_tau', 0):>5.2f} {top.get('druggability', 0):>5.2f} "
              f"{top.get('therm_class', '?'):<8} {meta.get('site_type', '?'):<7} {r1_hit:>4}")
        
        detail_results.append({
            'id': int(folder_id),
            'pdb': pdb_label,
            'align_rmsd': round(align_rmsd, 3) if align_rmsd is not None else None,
            'n_ca_matched': n_matched,
            'dcc_raw': round(raw_dcc1, 2),
            'dcc_aligned': round(aln_dccs[0], 2),
            'best3_aligned': round(min(aln_dccs[:3]), 2),
            'best10_aligned': round(min(aln_dccs[:10]), 2),
            'site_type': meta.get('site_type', '???'),
        })
        
    except Exception as e:
        print(f"{folder_id:<3} ERROR: {e}", file=sys.stderr)
        continue

# ============================================================
# SUMMARY
# ============================================================
print("\n" + "=" * 100)
print("  ALIGNMENT STATISTICS")
print("=" * 100)
print(f"  Aligned: {align_stats['success']}/{align_stats['success'] + align_stats['fail']} targets")
if align_stats['rmsds']:
    print(f"  Mean RMSD: {np.mean(align_stats['rmsds']):.3f} Å")
    print(f"  Max RMSD:  {np.max(align_stats['rmsds']):.3f} Å")
    print(f"  Min RMSD:  {np.min(align_stats['rmsds']):.3f} Å")

print("\n" + "=" * 100)
print("  RANKER COMPARISON (ALIGNED DCCs)")
print("=" * 100)
thresholds_header = f"{'RANKER':<28} {'SR@1<4':>7} {'SR@1<5':>7} {'SR@3<5':>7} {'MeanDCC':>8} {'N':>3}"
print(thresholds_header)
print("-" * 60)
for name, rr in sorted(ranker_results.items(), key=lambda x: -x[1]['sr1_5']):
    if rr['n'] > 0:
        print(f"{name:<28} {rr['sr1_4']/rr['n']*100:>6.1f}% {rr['sr1_5']/rr['n']*100:>6.1f}% "
              f"{rr['sr3_5']/rr['n']*100:>6.1f}% {np.mean(rr['dccs']):>7.2f} {rr['n']:>3}")

# Theoretical ceiling
print("\n" + "=" * 100)
print("  THEORETICAL CEILING (best pocket in any position, aligned)")
print("=" * 100)
ceil_5, ceil_8, ceil_10, n_ceil = 0, 0, 0, 0
for dr in detail_results:
    n_ceil += 1
    if dr['best10_aligned'] < 5.0: ceil_5 += 1
    if dr['best10_aligned'] < 8.0: ceil_8 += 1
    if dr['best10_aligned'] < 10.0: ceil_10 += 1
if n_ceil > 0:
    print(f"  Oracle <5Å:  {ceil_5}/{n_ceil} = {ceil_5/n_ceil*100:.1f}%")
    print(f"  Oracle <8Å:  {ceil_8}/{n_ceil} = {ceil_8/n_ceil*100:.1f}%")
    print(f"  Oracle <10Å: {ceil_10}/{n_ceil} = {ceil_10/n_ceil*100:.1f}%")

# By site type
print("\n" + "=" * 100)
print("  BREAKDOWN BY SITE TYPE (vol*drug ranker, aligned)")
print("=" * 100)
types = sorted(set(dr['site_type'] for dr in detail_results))
for st in types:
    subset = [dr for dr in detail_results if dr['site_type'] == st]
    n_st = len(subset)
    sr1_5 = sum(1 for dr in subset if dr['dcc_aligned'] < 5.0)
    sr3_5 = sum(1 for dr in subset if dr['best3_aligned'] < 5.0)
    mean_d = np.mean([dr['dcc_aligned'] for dr in subset])
    print(f"  {st:<15} N={n_st:>2}  SR@1<5Å={sr1_5/n_st*100:>5.1f}%  SR@3<5Å={sr3_5/n_st*100:>5.1f}%  MeanDCC={mean_d:.1f}")

# Save full results
out_path = os.path.join(BASE, "benchmark_evaluation_aligned.json")
with open(out_path, 'w') as f:
    json.dump({
        'alignment': {
            'n_aligned': align_stats['success'],
            'n_failed': align_stats['fail'],
            'mean_rmsd': round(float(np.mean(align_stats['rmsds'])), 3) if align_stats['rmsds'] else None,
        },
        'rankers': {name: {
            'sr1_4': round(rr['sr1_4']/max(rr['n'],1)*100, 1),
            'sr1_5': round(rr['sr1_5']/max(rr['n'],1)*100, 1),
            'sr3_5': round(rr['sr3_5']/max(rr['n'],1)*100, 1),
            'mean_dcc': round(float(np.mean(rr['dccs'])), 2) if rr['dccs'] else None,
            'n': rr['n'],
        } for name, rr in ranker_results.items()},
        'per_target': detail_results,
    }, f, indent=2, default=str)

print(f"\nResults saved to: {out_path}")
print("=" * 100)
