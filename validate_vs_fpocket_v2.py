#!/usr/bin/env python3
"""
PRISM4D vs Fpocket Validation
Compares ALL PRISM sites against ALL fpocket pockets using centroid distance.
Reports: recovery rate, new discoveries, and per-structure diagnostics.
"""

import json
import os
import sys
import glob
import warnings
import numpy as np
from Bio.PDB import PDBParser
warnings.simplefilter('ignore')

# --- CONFIG ---
FPOCKET_DIR = "e2e_validation_test/fpocket_baseline"
MATCH_THRESHOLD = 8.0  # Angstroms - pocket centroids can differ due to method

def find_latest_results():
    """Find the most recent PRISM4D results directory."""
    # Check for physics results first, then parallel, then any
    for pattern in ['results_physics_*', 'results_full_*', 'results_parallel_*', 'results_*']:
        dirs = sorted(glob.glob(f'e2e_validation_test/{pattern}'))
        if dirs:
            d = dirs[-1]
            all_json = os.path.join(d, 'all_results.json')
            if os.path.exists(all_json):
                return all_json
            # Try building it
            jsons = glob.glob(f'{d}/*/*.binding_sites.json')
            if jsons:
                return jsons
    return None

def load_prism_results(source):
    """Load PRISM results from all_results.json or individual files."""
    if isinstance(source, str) and source.endswith('.json'):
        with open(source) as f:
            return json.load(f)
    elif isinstance(source, list):
        results = []
        for jf in source:
            with open(jf) as f:
                results.append(json.load(f))
        return results
    return []

def get_fpocket_pockets(pdb_code, max_pockets=10):
    """Parse ALL fpocket pockets (not just top 3)."""
    pocket_dir = f"{FPOCKET_DIR}/{pdb_code}_out/pockets"
    if not os.path.exists(pocket_dir):
        return []

    parser = PDBParser(QUIET=True)
    pockets = []
    pocket_files = sorted(glob.glob(f"{pocket_dir}/pocket*_atm.pdb"))[:max_pockets]

    for p_file in pocket_files:
        try:
            struct = parser.get_structure("pocket", p_file)
            atoms = [a.get_coord() for a in struct.get_atoms()]
            residues = set()
            for model in struct:
                for chain in model:
                    for res in chain:
                        residues.add(res.id[1])

            if atoms:
                centroid = np.mean(atoms, axis=0)
                pockets.append({
                    "centroid": centroid,
                    "residues": residues,
                    "n_atoms": len(atoms),
                    "file": os.path.basename(p_file),
                    "rank": len(pockets) + 1,
                })
        except Exception as e:
            continue

    return pockets

def compute_distance_matrix(prism_sites, fpocket_pockets):
    """Compute all pairwise distances between PRISM and fpocket centroids."""
    if not prism_sites or not fpocket_pockets:
        return np.array([])
    
    n_prism = len(prism_sites)
    n_fp = len(fpocket_pockets)
    dist_matrix = np.zeros((n_prism, n_fp))
    
    for i, ps in enumerate(prism_sites):
        pc = np.array(ps['centroid'])
        for j, fp in enumerate(fpocket_pockets):
            dist_matrix[i, j] = np.linalg.norm(pc - fp['centroid'])
    
    return dist_matrix

def main():
    # Find results
    source = find_latest_results()
    if source is None:
        print("No PRISM4D results found.")
        sys.exit(1)
    
    if isinstance(source, str):
        print(f"Using: {source}")
    else:
        print(f"Using {len(source)} individual result files")
    
    prism_data = load_prism_results(source)
    
    # Header
    print()
    print("=" * 95)
    print(f"{'STRUCTURE':<10} {'PRISM':>6} {'FP':>4} {'MATCHED':>8} {'RECOVERY':>9} "
          f"{'BEST_DIST':>10} {'NEW_SITES':>10} {'STATUS'}")
    print("=" * 95)
    
    total_fp = 0
    total_recovered = 0
    total_prism = 0
    total_matched_prism = 0
    structure_results = []
    
    for entry in sorted(prism_data, key=lambda x: x.get('structure', '')):
        name = entry.get('structure', '?').replace('.topology', '')
        sites = entry.get('sites', [])
        n_prism = len(sites)
        
        # Get fpocket data
        fpockets = get_fpocket_pockets(name)
        n_fp = len(fpockets)
        
        if n_fp == 0:
            status = "NO_FP_DATA"
            print(f"{name:<10} {n_prism:>6} {n_fp:>4} {'--':>8} {'--':>9} "
                  f"{'--':>10} {'--':>10} {status}")
            continue
        
        # Distance matrix
        dist_matrix = compute_distance_matrix(sites, fpockets)
        
        # Greedy matching: for each fpocket pocket, find closest PRISM site
        fp_matched = set()
        prism_matched = set()
        matches = []
        best_dist = 999.0
        
        if dist_matrix.size > 0:
            # Sort all pairs by distance
            pairs = []
            for i in range(dist_matrix.shape[0]):
                for j in range(dist_matrix.shape[1]):
                    pairs.append((dist_matrix[i, j], i, j))
            pairs.sort()
            
            for dist, pi, fj in pairs:
                if dist > MATCH_THRESHOLD:
                    break
                if pi in prism_matched or fj in fp_matched:
                    continue
                prism_matched.add(pi)
                fp_matched.add(fj)
                matches.append((pi, fj, dist))
                if dist < best_dist:
                    best_dist = dist
        
        n_recovered = len(fp_matched)
        n_new = n_prism - len(prism_matched)
        recovery_pct = (n_recovered / n_fp * 100) if n_fp > 0 else 0
        
        total_fp += n_fp
        total_recovered += n_recovered
        total_prism += n_prism
        total_matched_prism += len(prism_matched)
        
        # Status
        if recovery_pct >= 60:
            status = "VALIDATED"
        elif recovery_pct >= 30:
            status = "PARTIAL"
        elif n_recovered > 0:
            status = "WEAK"
        else:
            status = "DIVERGENT"
        
        best_str = f"{best_dist:.1f}A" if best_dist < 999 else "--"
        
        print(f"{name:<10} {n_prism:>6} {n_fp:>4} {n_recovered:>8} "
              f"{recovery_pct:>8.0f}% {best_str:>10} {n_new:>10} {status}")
        
        structure_results.append({
            'name': name,
            'n_prism': n_prism,
            'n_fp': n_fp,
            'n_recovered': n_recovered,
            'recovery_pct': recovery_pct,
            'matches': matches,
            'best_dist': best_dist if best_dist < 999 else None,
            'status': status,
        })
    
    # Summary
    print("=" * 95)
    overall_recovery = (total_recovered / total_fp * 100) if total_fp > 0 else 0
    print(f"{'TOTAL':<10} {total_prism:>6} {total_fp:>4} {total_recovered:>8} "
          f"{overall_recovery:>8.0f}% {'':>10} {total_prism - total_matched_prism:>10}")
    print()
    
    # Diagnostic: show closest distances for divergent structures
    divergent = [r for r in structure_results if r['status'] == 'DIVERGENT']
    if divergent:
        print("DIAGNOSTICS FOR DIVERGENT STRUCTURES:")
        print("-" * 70)
        for r in divergent[:5]:
            name = r['name']
            fpockets = get_fpocket_pockets(name)
            entry = [e for e in prism_data if name in e.get('structure', '')][0]
            sites = entry['sites']
            
            print(f"\n  {name}:")
            print(f"    Fpocket centroids:")
            for fp in fpockets[:3]:
                c = fp['centroid']
                print(f"      FP{fp['rank']}: ({c[0]:.1f}, {c[1]:.1f}, {c[2]:.1f})  "
                      f"residues={sorted(list(fp['residues']))[:8]}...")
            print(f"    PRISM centroids (top 5 by spike count):")
            sorted_sites = sorted(sites, key=lambda s: s.get('spike_count', 0), reverse=True)[:5]
            for s in sorted_sites:
                c = s['centroid']
                # Distance to nearest fpocket
                min_d = min(np.linalg.norm(np.array(c) - fp['centroid']) for fp in fpockets) if fpockets else 999
                vol = s.get('volume', 0)
                spikes = s.get('spike_count', 0)
                drug = s.get('druggability', 0)
                rids = s.get('residue_ids', [])[:6]
                print(f"      Site {s['id']}: ({c[0]:.1f}, {c[1]:.1f}, {c[2]:.1f})  "
                      f"dist_to_fp={min_d:.1f}A  vol={vol:.0f}  spk={spikes}  "
                      f"drug={drug:.2f}  res={rids}...")
    
    # Verdict
    print()
    print("=" * 95)
    if overall_recovery >= 50:
        print(f"VERDICT: PRISM4D recovers {overall_recovery:.0f}% of fpocket sites within {MATCH_THRESHOLD}A")
    elif overall_recovery >= 20:
        print(f"VERDICT: Partial agreement ({overall_recovery:.0f}%). Likely different detection regimes.")
        print(f"  PRISM finds {total_prism - total_matched_prism} sites fpocket misses (potential cryptic sites)")
        print(f"  Fpocket finds {total_fp - total_recovered} sites PRISM misses")
    else:
        print(f"VERDICT: Low overlap ({overall_recovery:.0f}%). Root causes to investigate:")
        print(f"  1. Residue ID mapping: PRISM uses 0-indexed, fpocket uses PDB numbering")
        print(f"  2. Spike detection bias: aromatic displacement may not correlate with pocket geometry")
        print(f"  3. MD evolution: PRISM centroids from simulated positions vs static structure")
        print(f"  4. Threshold: try {MATCH_THRESHOLD*1.5:.0f}A or use residue overlap instead of centroid distance")

if __name__ == "__main__":
    main()
