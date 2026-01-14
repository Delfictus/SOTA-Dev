#!/usr/bin/env python3
"""
Analyze conformational ensemble for cryptic pockets.

This script:
1. Loads multi-MODEL PDB ensemble
2. Computes RMSF (flexibility) from MD
3. Identifies pockets that appear/disappear (cryptic sites)
4. Maps to known functional sites (ACE2 interface, antibody epitopes)

Usage:
    python scripts/analyze_ensemble_pockets.py \
        --ensemble data/ensembles/6M0J_RBD_ensemble.pdb \
        --output results/6M0J_RBD_cryptic_analysis.json
"""

import argparse
import json
import numpy as np
from pathlib import Path
from collections import defaultdict

# Known functional sites on SARS-CoV-2 RBD
# Residue numbers from 6M0J chain E
ACE2_INTERFACE_RESIDUES = [
    417, 446, 449, 453, 455, 456, 475, 476, 484, 486,
    487, 489, 490, 493, 494, 495, 496, 498, 500, 501, 502, 505
]

# Key antibody escape mutation sites
ESCAPE_MUTATION_SITES = {
    417: "K417N/T - Class 1 Ab escape",
    484: "E484K/Q - Class 2 Ab escape, immune evasion",
    501: "N501Y - ACE2 affinity, Alpha/Beta/Gamma",
    452: "L452R - Delta variant",
    478: "T478K - Delta variant",
    346: "R346K - Omicron",
    371: "S371L - Omicron",
    373: "S373P - Omicron",
    375: "S375F - Omicron",
    440: "K440N - Omicron",
    446: "G446S - Omicron",
    477: "S477N - Omicron",
    493: "Q493R - Omicron",
    496: "G496S - Omicron",
    498: "Q498R - Omicron",
    505: "Y505H - Omicron",
}


def parse_multi_model_pdb(pdb_path):
    """Parse multi-MODEL PDB into list of coordinate arrays."""
    models = []
    current_model = []
    residue_info = []  # (resname, resid, chain)
    first_model = True

    print(f"Loading ensemble: {pdb_path}")

    with open(pdb_path, 'r') as f:
        for line in f:
            if line.startswith('MODEL'):
                current_model = []
            elif line.startswith('ENDMDL'):
                if current_model:
                    models.append(np.array(current_model))
                    if first_model:
                        first_model = False
            elif line.startswith('ATOM'):
                atom_name = line[12:16].strip()
                if atom_name == 'CA':  # Only CA for RMSF
                    x = float(line[30:38])
                    y = float(line[38:46])
                    z = float(line[46:54])
                    current_model.append([x, y, z])

                    if first_model:  # First model - get residue info
                        resname = line[17:20].strip()
                        try:
                            resid = int(line[22:26])
                        except:
                            resid = len(residue_info) + 1
                        chain = line[21]
                        residue_info.append((resname, resid, chain))

    # Handle case where there's no MODEL/ENDMDL (single structure)
    if not models and current_model:
        models.append(np.array(current_model))

    print(f"  Loaded {len(models)} frames, {len(residue_info)} residues (CA atoms)")
    return models, residue_info


def compute_rmsf(models):
    """Compute RMSF from ensemble of CA coordinates."""
    if len(models) < 2:
        print("  Warning: Only 1 frame, RMSF will be zero")
        return np.zeros(len(models[0])), models[0]

    coords = np.array(models)  # (n_frames, n_residues, 3)
    mean_coords = coords.mean(axis=0)  # (n_residues, 3)

    # RMSF = sqrt(mean(|r - r_mean|^2))
    deviations = coords - mean_coords
    squared_dev = np.sum(deviations**2, axis=2)  # (n_frames, n_residues)
    rmsf = np.sqrt(squared_dev.mean(axis=0))  # (n_residues,)

    return rmsf, mean_coords


def identify_flexible_regions(rmsf, residue_info, threshold_sigma=1.5):
    """Identify regions with above-average flexibility."""
    mean_rmsf = rmsf.mean()
    std_rmsf = rmsf.std()

    if std_rmsf < 1e-6:
        print("  Warning: RMSF standard deviation near zero")
        return []

    threshold = mean_rmsf + threshold_sigma * std_rmsf

    flexible_residues = []
    for i, (resname, resid, chain) in enumerate(residue_info):
        if rmsf[i] > threshold:
            flexible_residues.append({
                'residue_id': resid,
                'residue_name': resname,
                'chain': chain,
                'rmsf': float(rmsf[i]),
                'z_score': float((rmsf[i] - mean_rmsf) / std_rmsf)
            })

    return flexible_residues


def map_to_functional_sites(flexible_residues):
    """Check if flexible residues overlap with known functional sites."""
    findings = []
    seen = set()

    for res in flexible_residues:
        resid = res['residue_id']

        if resid in seen:
            continue
        seen.add(resid)

        annotation = None
        if resid in ACE2_INTERFACE_RESIDUES:
            annotation = 'ACE2 interface'
        if resid in ESCAPE_MUTATION_SITES:
            if annotation:
                annotation += '; ' + ESCAPE_MUTATION_SITES[resid]
            else:
                annotation = ESCAPE_MUTATION_SITES[resid]

        if annotation:
            res_copy = res.copy()
            res_copy['functional_annotation'] = annotation
            findings.append(res_copy)

    return findings


def identify_cryptic_pocket_candidates(rmsf, residue_info, flexible_residues):
    """Identify potential cryptic pocket regions based on flexibility patterns."""
    candidates = []

    # Look for clusters of flexible residues
    flexible_ids = {r['residue_id'] for r in flexible_residues}

    # Find contiguous or nearby flexible regions
    sorted_flexible = sorted(flexible_residues, key=lambda x: x['residue_id'])

    current_cluster = []
    for res in sorted_flexible:
        if not current_cluster:
            current_cluster = [res]
        elif res['residue_id'] - current_cluster[-1]['residue_id'] <= 3:
            # Within 3 residues - same cluster
            current_cluster.append(res)
        else:
            # Gap - save cluster if large enough
            if len(current_cluster) >= 2:
                candidates.append({
                    'residues': current_cluster,
                    'start': current_cluster[0]['residue_id'],
                    'end': current_cluster[-1]['residue_id'],
                    'size': len(current_cluster),
                    'mean_rmsf': np.mean([r['rmsf'] for r in current_cluster]),
                    'max_rmsf': max(r['rmsf'] for r in current_cluster),
                })
            current_cluster = [res]

    # Don't forget last cluster
    if len(current_cluster) >= 2:
        candidates.append({
            'residues': current_cluster,
            'start': current_cluster[0]['residue_id'],
            'end': current_cluster[-1]['residue_id'],
            'size': len(current_cluster),
            'mean_rmsf': np.mean([r['rmsf'] for r in current_cluster]),
            'max_rmsf': max(r['rmsf'] for r in current_cluster),
        })

    return sorted(candidates, key=lambda x: -x['mean_rmsf'])


def analyze_ensemble(pdb_path, output_path, threshold_sigma=1.5):
    """Main analysis function."""
    models, residue_info = parse_multi_model_pdb(pdb_path)
    n_frames = len(models)
    n_residues = len(residue_info)

    if n_frames == 0 or n_residues == 0:
        print("ERROR: No frames or residues found!")
        return None

    # Compute RMSF
    print("Computing RMSF...")
    rmsf, mean_coords = compute_rmsf(models)

    print(f"  Mean RMSF: {rmsf.mean():.2f} A")
    print(f"  Std RMSF:  {rmsf.std():.2f} A")
    print(f"  Max RMSF:  {rmsf.max():.2f} A (residue {residue_info[rmsf.argmax()][1]})")

    # Identify flexible regions
    flexible_residues = identify_flexible_regions(rmsf, residue_info, threshold_sigma)
    print(f"  Flexible residues (>{threshold_sigma}sigma): {len(flexible_residues)}")

    # Map to functional sites
    functional_findings = map_to_functional_sites(flexible_residues)
    print(f"  Overlapping functional sites: {len(functional_findings)}")

    # Identify cryptic pocket candidates
    cryptic_candidates = identify_cryptic_pocket_candidates(rmsf, residue_info, flexible_residues)
    print(f"  Cryptic pocket candidate regions: {len(cryptic_candidates)}")

    # Build results
    results = {
        'input_file': str(pdb_path),
        'n_frames': n_frames,
        'n_residues': n_residues,
        'analysis_parameters': {
            'threshold_sigma': threshold_sigma,
        },
        'rmsf': {
            'mean': float(rmsf.mean()),
            'std': float(rmsf.std()),
            'max': float(rmsf.max()),
            'max_residue': int(residue_info[rmsf.argmax()][1]),
            'per_residue': [
                {
                    'residue_id': int(residue_info[i][1]),
                    'residue_name': residue_info[i][0],
                    'chain': residue_info[i][2],
                    'rmsf': float(rmsf[i])
                }
                for i in range(n_residues)
            ]
        },
        'flexible_regions': flexible_residues,
        'functional_site_overlap': functional_findings,
        'cryptic_pocket_candidates': [
            {
                'start_residue': c['start'],
                'end_residue': c['end'],
                'size': c['size'],
                'mean_rmsf': c['mean_rmsf'],
                'max_rmsf': c['max_rmsf'],
                'residue_ids': [r['residue_id'] for r in c['residues']],
            }
            for c in cryptic_candidates
        ],
    }

    # Report key findings
    print("\n" + "="*60)
    print("KEY FINDINGS")
    print("="*60)

    # Top flexible residues
    print("\nTop 10 Most Flexible Residues:")
    sorted_by_rmsf = sorted(enumerate(rmsf), key=lambda x: -x[1])[:10]
    for i, rmsf_val in sorted_by_rmsf:
        resname, resid, chain = residue_info[i]
        z_score = (rmsf_val - rmsf.mean()) / rmsf.std() if rmsf.std() > 0 else 0
        annotation = ""
        if resid in ACE2_INTERFACE_RESIDUES:
            annotation = " [ACE2 interface]"
        if resid in ESCAPE_MUTATION_SITES:
            annotation += f" [{ESCAPE_MUTATION_SITES[resid]}]"
        print(f"  {resname}{resid}: RMSF={rmsf_val:.2f}A (z={z_score:.1f}){annotation}")

    if functional_findings:
        print("\n⚠️  IMPORTANT: Flexible residues at functional sites:")
        for f in sorted(functional_findings, key=lambda x: -x['rmsf']):
            print(f"  {f['residue_name']}{f['residue_id']}: "
                  f"RMSF={f['rmsf']:.2f}A (z={f['z_score']:.1f}) - "
                  f"{f.get('functional_annotation', 'unknown')}")

    # Check ACE2 interface flexibility
    ace2_indices = [i for i, (_, resid, _) in enumerate(residue_info)
                    if resid in ACE2_INTERFACE_RESIDUES]
    if ace2_indices:
        ace2_rmsf = rmsf[ace2_indices]
        print(f"\nACE2 Interface Flexibility:")
        print(f"  Mean RMSF: {np.mean(ace2_rmsf):.2f} A")
        print(f"  Max RMSF:  {np.max(ace2_rmsf):.2f} A")
        print(f"  Residues covered: {len(ace2_indices)}/{len(ACE2_INTERFACE_RESIDUES)}")

    # Check escape mutation sites
    escape_indices = [(i, resid) for i, (_, resid, _) in enumerate(residue_info)
                      if resid in ESCAPE_MUTATION_SITES]
    if escape_indices:
        print(f"\nEscape Mutation Sites Flexibility:")
        escape_flexible = []
        for i, resid in escape_indices:
            z_score = (rmsf[i] - rmsf.mean()) / rmsf.std() if rmsf.std() > 0 else 0
            if z_score > threshold_sigma:
                escape_flexible.append((resid, rmsf[i], z_score))

        if escape_flexible:
            print(f"  🔴 HIGH FLEXIBILITY escape sites:")
            for resid, rmsf_val, z in sorted(escape_flexible, key=lambda x: -x[1]):
                print(f"    {resid}: RMSF={rmsf_val:.2f}A (z={z:.1f}) - {ESCAPE_MUTATION_SITES[resid]}")
        else:
            print(f"  No escape mutation sites above {threshold_sigma}sigma threshold")

    # Cryptic pocket candidates
    if cryptic_candidates:
        print(f"\n🔵 CRYPTIC POCKET CANDIDATES:")
        for i, c in enumerate(cryptic_candidates[:5], 1):  # Top 5
            print(f"  Region {i}: residues {c['start']}-{c['end']} "
                  f"({c['size']} residues, mean RMSF={c['mean_rmsf']:.2f}A)")

    # Save results
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {output_path}")

    return results


def main():
    parser = argparse.ArgumentParser(
        description='Analyze conformational ensemble for cryptic pockets'
    )
    parser.add_argument('--ensemble', required=True, help='Input ensemble PDB (multi-MODEL)')
    parser.add_argument('--output', required=True, help='Output JSON file')
    parser.add_argument('--threshold', type=float, default=1.5,
                        help='Z-score threshold for flexibility (default: 1.5)')

    args = parser.parse_args()
    analyze_ensemble(args.ensemble, args.output, args.threshold)


if __name__ == '__main__':
    main()
