#!/usr/bin/env python3
"""
PRISM-DELTA Blind Benchmark: Multi-Family Viral Glycoprotein Epitope Prediction

ZERO DATA LEAKAGE PROTOCOL:
1. PHASE 1 (BLIND): Run predictions on ANTIGEN chain only - NO ground truth access
2. PHASE 2 (LOCK): Predictions are written to file and LOCKED
3. PHASE 3 (REVEAL): Extract ground truth from ANTIBODY contacts (post-hoc)
4. PHASE 4 (EVALUATE): Compute SOTA metrics

Ground truth is NEVER used during prediction. It is derived from antibody-antigen
contacts at ≤4Å AFTER predictions are locked.
"""

import json
import subprocess
import sys
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict, Set, Tuple
import numpy as np

@dataclass
class BlindTarget:
    """A blind validation target with ground truth held out"""
    name: str
    pdb_id: str
    antigen_chain: str      # Chain to predict on (BLIND)
    antibody_chains: str    # Chains with antibody (for ground truth extraction POST-HOC)
    family: str
    description: str

# Define targets - ground truth will be extracted POST-HOC from contacts
# For structures without antibody, use literature-defined epitopes

# Literature-defined epitopes (for structures without antibody chains)
LITERATURE_EPITOPES = {
    # m102.4 epitope from Xu et al. 2013 - central cavity of Nipah G
    "2VWD": set(list(range(504, 515)) + list(range(527, 536))),
    # 1A1D-2 epitope on Dengue E DIII - from Lok et al. 2008
    "1OKE": set([305, 307, 309, 310, 311, 312, 330, 332, 333, 334, 335,
                 360, 361, 362, 386, 387, 388, 389, 390]),
}

BLIND_TARGETS = [
    BlindTarget(
        name="Nipah_G_m102.4",
        pdb_id="2VWD",
        antigen_chain="A",
        antibody_chains="",     # No Ab in structure - use literature epitope
        family="Paramyxoviridae",
        description="Nipah G protein central cavity epitope"
    ),
    BlindTarget(
        name="Ebola_GP_KZ52",
        pdb_id="3CSY",
        antigen_chain="I",      # GP1 chain
        antibody_chains="AB",   # KZ52 Fab (H=A, L=B)
        family="Filoviridae",
        description="Ebola GP base epitope (quaternary)"
    ),
    BlindTarget(
        name="SARS2_RBD_ACE2",
        pdb_id="6M0J",
        antigen_chain="E",      # Spike RBD
        antibody_chains="A",    # ACE2 (receptor as functional ground truth)
        family="Coronaviridae",
        description="SARS-CoV-2 RBD receptor binding motif"
    ),
    BlindTarget(
        name="Dengue_E_1A1D2",
        pdb_id="1OKE",
        antigen_chain="A",      # Envelope E chain A
        antibody_chains="",     # Use literature epitope
        family="Flaviviridae",
        description="Dengue E domain III epitope"
    ),
]

def parse_pdb_atoms(pdb_path: Path, chain: str) -> List[Dict]:
    """Parse CA atoms from a PDB chain"""
    atoms = []
    with open(pdb_path) as f:
        for line in f:
            if line.startswith("ATOM") and line[21] == chain:
                atom_name = line[12:16].strip()
                if atom_name == "CA":
                    try:
                        atoms.append({
                            'res_num': int(line[22:26].strip()),
                            'res_name': line[17:20].strip(),
                            'x': float(line[30:38]),
                            'y': float(line[38:46]),
                            'z': float(line[46:54]),
                        })
                    except ValueError:
                        continue
    return atoms

def parse_pdb_all_atoms(pdb_path: Path, chains: str) -> List[Dict]:
    """Parse all heavy atoms from specified chains"""
    atoms = []
    with open(pdb_path) as f:
        for line in f:
            if line.startswith("ATOM") and line[21] in chains:
                atom_name = line[12:16].strip()
                if not atom_name.startswith('H'):  # Skip hydrogens
                    try:
                        atoms.append({
                            'res_num': int(line[22:26].strip()),
                            'res_name': line[17:20].strip(),
                            'chain': line[21],
                            'atom_name': atom_name,
                            'x': float(line[30:38]),
                            'y': float(line[38:46]),
                            'z': float(line[46:54]),
                        })
                    except ValueError:
                        continue
    return atoms

def extract_epitope_from_contacts(pdb_path: Path, antigen_chain: str,
                                   antibody_chains: str, cutoff: float = 4.0) -> Set[int]:
    """
    Extract epitope residues POST-HOC from antibody-antigen contacts.
    This is the GROUND TRUTH - only called AFTER predictions are locked.

    Epitope = antigen residues with any heavy atom within cutoff Å of antibody atoms
    """
    antigen_atoms = parse_pdb_all_atoms(pdb_path, antigen_chain)
    antibody_atoms = parse_pdb_all_atoms(pdb_path, antibody_chains)

    if not antibody_atoms:
        print(f"  ⚠️ No antibody atoms found in chains '{antibody_chains}'")
        return set()

    epitope_residues = set()

    for ag_atom in antigen_atoms:
        for ab_atom in antibody_atoms:
            dx = ag_atom['x'] - ab_atom['x']
            dy = ag_atom['y'] - ab_atom['y']
            dz = ag_atom['z'] - ab_atom['z']
            dist = np.sqrt(dx*dx + dy*dy + dz*dz)

            if dist <= cutoff:
                epitope_residues.add(ag_atom['res_num'])
                break  # Found contact, move to next antigen atom

    return epitope_residues

def run_blind_prediction(pdb_path: Path, chain: str, output_path: Path) -> bool:
    """
    Run PRISM blind prediction on antigen chain.
    NO GROUND TRUTH ACCESS HERE.
    """
    cmd = [
        "cargo", "run", "--release", "-p", "prism-validation",
        "--bin", "run-blind-validation", "--",
        "--pdb", str(pdb_path),
        "--chain", chain,
        "--output", str(output_path),
        "--verbose"
    ]

    result = subprocess.run(cmd, capture_output=True, text=True,
                           cwd="/home/diddy/Desktop/PRISM4D-bio")

    if result.returncode != 0:
        print(f"  ❌ Prediction failed: {result.stderr[:500]}")
        return False

    return True

def load_predictions(output_path: Path) -> Dict:
    """Load locked predictions from file"""
    with open(output_path) as f:
        return json.load(f)

def compute_sota_metrics(predictions: Dict, epitope: Set[int]) -> Dict:
    """
    Compute SOTA ranking metrics.
    This is called AFTER predictions are locked.
    """
    residue_preds = predictions['residue_predictions']

    # Build score list sorted by score (descending)
    scored_residues = [
        (p['residue_num'], p['cryptic_score'])
        for p in residue_preds
    ]
    scored_residues.sort(key=lambda x: -x[1])  # Descending by score

    all_residues = set(p['residue_num'] for p in residue_preds)
    n_total = len(all_residues)
    n_epitope = len(epitope & all_residues)  # Only count epitope in structure

    if n_epitope == 0:
        return {'error': 'No epitope residues in structure'}

    # ROC-AUC computation
    labels = [1 if r in epitope else 0 for r, _ in scored_residues]
    scores = [s for _, s in scored_residues]
    roc_auc = compute_roc_auc(scores, labels)

    # PR-AUC computation
    pr_auc = compute_pr_auc(scores, labels)

    # Top-K precision
    top_15_hits = sum(1 for r, _ in scored_residues[:15] if r in epitope)
    top_30_hits = sum(1 for r, _ in scored_residues[:30] if r in epitope)
    top_50_hits = sum(1 for r, _ in scored_residues[:50] if r in epitope)

    precision_15 = top_15_hits / 15
    precision_30 = top_30_hits / 30
    precision_50 = top_50_hits / 50

    # Recall at different thresholds
    threshold = predictions['summary'].get('threshold_used', 0.3)
    detected = [r for r, s in scored_residues if s >= threshold and r in epitope]
    recall = len(detected) / n_epitope if n_epitope > 0 else 0

    # Site ranking - find first site containing epitope
    sites = predictions.get('predicted_sites', [])
    first_epitope_site_rank = None
    epitope_coverage_by_sites = set()

    for i, site in enumerate(sites):
        if isinstance(site['residues'][0], dict):
            site_residues = set(r['residue_num'] for r in site['residues'])
        else:
            site_residues = set(site['residues'])

        overlap = site_residues & epitope
        if overlap:
            epitope_coverage_by_sites.update(overlap)
            if first_epitope_site_rank is None:
                first_epitope_site_rank = i + 1

    return {
        'roc_auc': roc_auc,
        'pr_auc': pr_auc,
        'precision_at_15': precision_15,
        'precision_at_30': precision_30,
        'precision_at_50': precision_50,
        'recall': recall,
        'first_epitope_site_rank': first_epitope_site_rank,
        'n_sites': len(sites),
        'n_epitope': n_epitope,
        'n_detected': len(detected),
        'epitope_coverage_by_sites': len(epitope_coverage_by_sites),
    }

def compute_roc_auc(scores: List[float], labels: List[int]) -> float:
    """Compute ROC-AUC from scores and binary labels"""
    pairs = list(zip(scores, labels))
    pairs.sort(key=lambda x: -x[0])  # Sort by score descending

    n_pos = sum(labels)
    n_neg = len(labels) - n_pos

    if n_pos == 0 or n_neg == 0:
        return 0.5

    tpr_prev, fpr_prev = 0.0, 0.0
    tp, fp = 0, 0
    auc = 0.0
    prev_score = float('inf')

    for score, label in pairs:
        if score != prev_score:
            tpr = tp / n_pos
            fpr = fp / n_neg
            auc += (fpr - fpr_prev) * (tpr + tpr_prev) / 2
            tpr_prev, fpr_prev = tpr, fpr
            prev_score = score

        if label == 1:
            tp += 1
        else:
            fp += 1

    # Final point
    tpr = tp / n_pos
    fpr = fp / n_neg
    auc += (fpr - fpr_prev) * (tpr + tpr_prev) / 2

    return auc

def compute_pr_auc(scores: List[float], labels: List[int]) -> float:
    """Compute PR-AUC (Average Precision) from scores and binary labels"""
    pairs = list(zip(scores, labels))
    pairs.sort(key=lambda x: -x[0])  # Sort by score descending

    n_pos = sum(labels)
    if n_pos == 0:
        return 0.0

    tp = 0
    ap = 0.0

    for i, (score, label) in enumerate(pairs, 1):
        if label == 1:
            tp += 1
            precision = tp / i
            ap += precision

    return ap / n_pos

def main():
    print("=" * 75)
    print("  PRISM-DELTA BLIND BENCHMARK: Multi-Family Viral Glycoprotein")
    print("  ZERO DATA LEAKAGE PROTOCOL")
    print("=" * 75)

    results = []
    base_path = Path("/home/diddy/Desktop/PRISM4D-bio")

    for target in BLIND_TARGETS:
        print(f"\n{'='*75}")
        print(f"  TARGET: {target.name} ({target.family})")
        print(f"  {target.description}")
        print(f"{'='*75}")

        pdb_path = base_path / "data/raw" / f"{target.pdb_id}.pdb"
        output_path = base_path / "results" / f"{target.pdb_id.lower()}_blind_benchmark.json"

        if not pdb_path.exists():
            print(f"  ❌ PDB file not found: {pdb_path}")
            continue

        # PHASE 1: BLIND PREDICTION (NO ground truth)
        print(f"\n  📊 PHASE 1: BLIND PREDICTION (chain {target.antigen_chain})")
        print(f"     ⚠️  NO GROUND TRUTH ACCESS")

        success = run_blind_prediction(pdb_path, target.antigen_chain, output_path)
        if not success:
            continue

        # PHASE 2: LOCK PREDICTIONS
        print(f"\n  🔒 PHASE 2: PREDICTIONS LOCKED")
        predictions = load_predictions(output_path)
        n_residues = len(predictions['residue_predictions'])
        n_cryptic = predictions['summary'].get('n_cryptic_residues', 0)
        print(f"     Residues analyzed: {n_residues}")
        print(f"     Cryptic predicted: {n_cryptic}")

        # PHASE 3: EXTRACT GROUND TRUTH (POST-HOC)
        print(f"\n  🎯 PHASE 3: REVEAL GROUND TRUTH (from antibody contacts)")

        if target.antibody_chains:
            epitope = extract_epitope_from_contacts(
                pdb_path, target.antigen_chain, target.antibody_chains, cutoff=4.0
            )
            print(f"     Source: Structural contacts (≤4Å from Ab)")
            print(f"     Epitope residues: {len(epitope)}")
            if epitope:
                print(f"     Residues: {sorted(epitope)[:20]}{'...' if len(epitope) > 20 else ''}")
        elif target.pdb_id in LITERATURE_EPITOPES:
            epitope = LITERATURE_EPITOPES[target.pdb_id]
            print(f"     Source: Literature-defined epitope")
            print(f"     Epitope residues: {len(epitope)}")
            print(f"     Residues: {sorted(epitope)[:20]}{'...' if len(epitope) > 20 else ''}")
        else:
            print(f"     ⚠️ No ground truth available")
            epitope = set()

        if not epitope:
            print(f"  ⚠️ Skipping - no ground truth available")
            continue

        # PHASE 4: COMPUTE SOTA METRICS
        print(f"\n  📈 PHASE 4: SOTA METRICS")
        metrics = compute_sota_metrics(predictions, epitope)

        if 'error' in metrics:
            print(f"     ❌ {metrics['error']}")
            continue

        print(f"     ROC-AUC:           {metrics['roc_auc']:.3f}")
        print(f"     PR-AUC:            {metrics['pr_auc']:.3f}")
        print(f"     Precision@15:      {metrics['precision_at_15']:.3f} ({int(metrics['precision_at_15']*15)}/15)")
        print(f"     Precision@30:      {metrics['precision_at_30']:.3f} ({int(metrics['precision_at_30']*30)}/30)")
        print(f"     Precision@50:      {metrics['precision_at_50']:.3f} ({int(metrics['precision_at_50']*50)}/50)")
        print(f"     Recall:            {metrics['recall']:.3f} ({metrics['n_detected']}/{metrics['n_epitope']})")
        print(f"     First Epitope Site: Rank #{metrics['first_epitope_site_rank']} of {metrics['n_sites']}")

        results.append({
            'target': target.name,
            'family': target.family,
            'pdb_id': target.pdb_id,
            'n_epitope': metrics['n_epitope'],
            **metrics
        })

    # SUMMARY TABLE
    print("\n" + "=" * 75)
    print("  CROSS-FAMILY BENCHMARK SUMMARY")
    print("=" * 75)
    print(f"\n  {'Target':<25} {'ROC-AUC':<10} {'PR-AUC':<10} {'P@30':<10} {'Recall':<10} {'Site Rank':<10}")
    print("  " + "-" * 70)

    for r in results:
        site_rank = f"#{r['first_epitope_site_rank']}" if r['first_epitope_site_rank'] else "N/A"
        print(f"  {r['target']:<25} {r['roc_auc']:<10.3f} {r['pr_auc']:<10.3f} "
              f"{r['precision_at_30']:<10.3f} {r['recall']:<10.3f} {site_rank:<10}")

    # Compute averages
    if results:
        avg_roc = np.mean([r['roc_auc'] for r in results])
        avg_pr = np.mean([r['pr_auc'] for r in results])
        avg_p30 = np.mean([r['precision_at_30'] for r in results])
        avg_recall = np.mean([r['recall'] for r in results])

        print("  " + "-" * 70)
        print(f"  {'AVERAGE':<25} {avg_roc:<10.3f} {avg_pr:<10.3f} "
              f"{avg_p30:<10.3f} {avg_recall:<10.3f}")

    # SOTA comparison
    print("\n" + "=" * 75)
    print("  SOTA COMPARISON")
    print("=" * 75)
    print("""
  Method          ROC-AUC   PR-AUC    Notes
  ─────────────────────────────────────────────────────────
  ScanNet         ~0.85     ~0.40     Deep learning, requires MSA
  MaSIF           ~0.82     ~0.35     Geometric deep learning
  PocketMiner     0.87      0.17*     MD-based, cryptic sites
  ─────────────────────────────────────────────────────────
  * PocketMiner PR-AUC from cryptic site task, not epitope
""")

    if results:
        if avg_roc >= 0.85:
            print("  ✅ PRISM ROC-AUC meets SOTA threshold (≥0.85)")
        else:
            print(f"  ❌ PRISM ROC-AUC ({avg_roc:.3f}) below SOTA threshold (0.85)")

        if avg_pr >= 0.30:
            print("  ✅ PRISM PR-AUC meets practical threshold (≥0.30)")
        else:
            print(f"  ❌ PRISM PR-AUC ({avg_pr:.3f}) below practical threshold (0.30)")

    # Save results
    output_file = base_path / "results" / "blind_benchmark_sota.json"
    with open(output_file, 'w') as f:
        json.dump({
            'protocol': 'ZERO_DATA_LEAKAGE',
            'targets': results,
            'sota_thresholds': {
                'roc_auc': 0.85,
                'pr_auc': 0.30,
                'site_rank': 3
            }
        }, f, indent=2)

    print(f"\n  Results saved to: {output_file}")
    print("=" * 75)

if __name__ == "__main__":
    main()
