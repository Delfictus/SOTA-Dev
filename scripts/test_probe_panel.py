#!/usr/bin/env python3
"""
PRISM4D — Property-Matched Probe Panel Prototype (Surrogate)

Approximates probe-response selectivity from existing per-site data.
Each probe class is a feature filter on lining residue composition,
burial, polarity, and contact architecture.

NOT a docking run. Tests site selectivity via property matching.
"""

import json, math, glob, sys
from pathlib import Path
from collections import Counter

# === GROUND TRUTH ===
GROUND_TRUTH_PATH = Path("benchmarks/prism4d_bench30/ground_truth/ligand_centroids.json")
MANIFEST_PATH = Path("benchmarks/prism4d_bench30/benchmark_manifest.json")
if "pytest" in sys.modules and not (GROUND_TRUTH_PATH.is_file() and MANIFEST_PATH.is_file()):
    import pytest

    pytest.skip("requires PRISM4D benchmark ground-truth data", allow_module_level=True)

with open(GROUND_TRUTH_PATH) as f:
    gt_raw = json.load(f)
with open(MANIFEST_PATH) as f:
    mdata = json.load(f)
manifest = mdata.get('targets', mdata) if isinstance(mdata, dict) else mdata
gt_map = {}
for m in manifest:
    if isinstance(m, dict):
        pdb = m.get('apo_pdb', '').lower()
        bid = str(m.get('id', ''))
        if bid in gt_raw:
            gt_map[pdb] = gt_raw[bid].get('centroid', [0, 0, 0])
gt_map.update({'1p38': [16.885, -3.551, -19.552], '5lar': [54.0, 68.1, 16.8]})

# === RESIDUE PROPERTY TABLES ===
POLAR_ACCEPTOR = {'ASP', 'GLU', 'ASN', 'GLN', 'HIS', 'SER', 'THR', 'TYR'}
POLAR_DONOR = {'ARG', 'LYS', 'HIS', 'ASN', 'GLN', 'SER', 'THR', 'TRP', 'TYR'}
AROMATIC = {'TRP', 'TYR', 'PHE', 'HIS'}
HYDROPHOBIC = {'ALA', 'VAL', 'LEU', 'ILE', 'PHE', 'TRP', 'MET', 'PRO'}
CHARGED_NEG = {'ASP', 'GLU'}
CHARGED_POS = {'ARG', 'LYS'}
CATALYTIC = {'SER', 'HIS', 'ASP', 'CYS', 'GLU', 'LYS'}


def dcc(c, gt):
    return math.sqrt(sum((c[i] - gt[i]) ** 2 for i in range(3)))


def lining_composition(site):
    """Extract lining residue composition."""
    lining = site.get("lining_residues", [])
    resnames = [r.get("resname", "UNK") for r in lining]
    return Counter(resnames), len(lining)


def probe_a_score(site):
    """Probe A: Neutral heteroaryl acceptor — directional H-bond acceptor environment."""
    comp, n = lining_composition(site)
    if n == 0:
        return 0.0
    acceptor_frac = sum(comp.get(r, 0) for r in POLAR_ACCEPTOR) / n
    aromatic_frac = sum(comp.get(r, 0) for r in AROMATIC) / n
    burial = site.get("burial_score", 0.5)
    # Directional acceptor environment: wants acceptors + aromatics + burial
    return (0.40 * acceptor_frac + 0.30 * aromatic_frac + 0.30 * burial)


def probe_b_score(site):
    """Probe B: Weakly basic heterocycle — acidic/polar anchor environment."""
    comp, n = lining_composition(site)
    if n == 0:
        return 0.0
    neg_frac = sum(comp.get(r, 0) for r in CHARGED_NEG) / n
    polar_frac = sum(comp.get(r, 0) for r in POLAR_ACCEPTOR | POLAR_DONOR) / n
    depth = site.get("mean_burial", 3.0)
    depth_n = depth / (depth + 5.0)
    return (0.35 * neg_frac + 0.35 * polar_frac + 0.30 * depth_n)


def probe_c_score(site):
    """Probe C: Small hydrophobic aromatic — apolar packing environment."""
    comp, n = lining_composition(site)
    if n == 0:
        return 0.0
    hydro_frac = sum(comp.get(r, 0) for r in HYDROPHOBIC) / n
    aromatic_frac = sum(comp.get(r, 0) for r in AROMATIC) / n
    burial = site.get("burial_score", 0.5)
    # Hydrophobic capture: wants hydrophobic + aromatic + deep burial
    return (0.40 * hydro_frac + 0.30 * aromatic_frac + 0.30 * burial)


def probe_d_score(site):
    """Probe D: Donor/acceptor-balanced polar fragment — mixed recognition."""
    comp, n = lining_composition(site)
    if n == 0:
        return 0.0
    donor_frac = sum(comp.get(r, 0) for r in POLAR_DONOR) / n
    acceptor_frac = sum(comp.get(r, 0) for r in POLAR_ACCEPTOR) / n
    # Balance: penalize strong asymmetry
    balance = 1.0 - abs(donor_frac - acceptor_frac)
    mixed_frac = min(donor_frac, acceptor_frac) * 2  # reward overlap
    burial = site.get("burial_score", 0.5)
    return (0.35 * mixed_frac + 0.30 * balance + 0.35 * burial)


def probe_response_matrix(site):
    """Compute all probe scores for a site."""
    return {
        "A_acceptor": round(probe_a_score(site), 4),
        "B_basic": round(probe_b_score(site), 4),
        "C_hydrophobic": round(probe_c_score(site), 4),
        "D_balanced": round(probe_d_score(site), 4),
    }


def response_comparison(site_candidate, site_top1):
    """Compare probe responses: does candidate show stronger selective response?"""
    rc = probe_response_matrix(site_candidate)
    rt = probe_response_matrix(site_top1)

    wins = 0
    losses = 0
    details = {}
    for probe in ['A_acceptor', 'B_basic', 'C_hydrophobic', 'D_balanced']:
        ratio = rc[probe] / max(rt[probe], 1e-6)
        gap = rc[probe] - rt[probe]
        if ratio > 1.15 and gap > 0.02:
            wins += 1
            details[probe] = f"+{gap:.3f} ({ratio:.2f}x)"
        elif ratio < 0.85:
            losses += 1
            details[probe] = f"{gap:+.3f} ({ratio:.2f}x)"
        else:
            details[probe] = f"~tied ({ratio:.2f}x)"

    supports = wins >= 2 and losses <= 1
    return {
        "candidate_probes": rc,
        "top1_probes": rt,
        "wins": wins,
        "losses": losses,
        "supports": supports,
        "details": details,
    }


# === MAIN ===
targets = ['1jwp', '1p38', '5lar', '2hnp', '3l3n', '1nna', '1hcl', '2npq']

bs_files = {}
for d in [Path('/tmp/prism_bench10_scratch'), Path('/tmp/prism_hard_targets')]:
    if not d.exists():
        continue
    for bs in d.glob('*/*.binding_sites.json'):
        name = bs.name.split('.')[0]
        if name in gt_map:
            bs_files[name] = bs

print("=== PROPERTY-MATCHED PROBE PANEL ===")
print("Probes: A=acceptor B=basic C=hydrophobic D=balanced")
print("Comparison: candidate vs GTCKL top-1 (majority rule)\n")

results = []

for name in targets:
    if name not in bs_files:
        continue
    gt = gt_map[name]
    with open(bs_files[name]) as f:
        data = json.load(f)
    sites = data if isinstance(data, list) else data.get('sites', [])

    ranked = sorted(sites, key=lambda s: s.get('gtck_rank', 999))
    top1 = ranked[0]
    true_site = min(sites, key=lambda s: dcc(s.get('centroid', [0, 0, 0]), gt))
    already_correct = top1.get('id') == true_site.get('id')

    top1_dcc = dcc(top1.get('centroid', [0, 0, 0]), gt)
    true_dcc = dcc(true_site.get('centroid', [0, 0, 0]), gt)

    if already_correct:
        print(f"{name.upper():>5}: CORRECT (top1=true, DCC={top1_dcc:.1f}A)")
        results.append({"target": name, "status": "correct"})
        continue

    comp = response_comparison(true_site, top1)
    call = "SUPPORTS" if comp['supports'] else "NEUTRAL" if comp['wins'] >= comp['losses'] else "REJECTS"

    print(f"{name.upper():>5}: top1={top1_dcc:.1f}A true={true_dcc:.1f}A | "
          f"wins={comp['wins']} losses={comp['losses']} → {call}")
    for probe, detail in comp['details'].items():
        print(f"        {probe}: {detail}")

    results.append({
        "target": name,
        "top1_dcc": top1_dcc,
        "true_dcc": true_dcc,
        "wins": comp['wins'],
        "losses": comp['losses'],
        "call": call,
        "n_lining_true": len(true_site.get("lining_residues", [])),
        "n_lining_top1": len(top1.get("lining_residues", [])),
    })
    print()

# === SUMMARY ===
print("=== SUMMARY ===")
supports = sum(1 for r in results if r.get("call") == "SUPPORTS")
rejects = sum(1 for r in results if r.get("call") == "REJECTS")
neutral = sum(1 for r in results if r.get("call") == "NEUTRAL")
correct = sum(1 for r in results if r.get("status") == "correct")
print(f"Correct (already R1): {correct}")
print(f"SUPPORTS true pocket: {supports}")
print(f"NEUTRAL: {neutral}")
print(f"REJECTS true pocket: {rejects}")
print(f"\nProbe panel useful: {'YES' if supports >= 2 else 'NO'} ({supports}/8 targets improved)")
