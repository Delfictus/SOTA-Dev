#!/usr/bin/env python3
"""Test coherence layer: does adding collective coherence improve Therm override?"""

import json, math, glob, os

TOP_K = 10

W_THERM = {"breathing": 1.0, "hysteresis": 1.0, "onset": 0.5, "kin": 0.3, "frust": -0.5}
W_COH = {"tau_inv": 0.7, "asym": 0.7, "offset": 0.3}

OVERRIDE_RATIO = 1.5
OVERRIDE_GAP = 0.3

# Ground truth
GT_PATH = "benchmarks/prism4d_bench30/ground_truth/ligand_centroids.json"
MANIFEST_PATH = "benchmarks/prism4d_bench30/benchmark_manifest.json"
HARD_GT = {"1p38": [16.885,-3.551,-19.552], "5lar": [54.0,68.1,16.8], "3mh1": [34.023,35.443,16.502]}


def dcc(c, gt):
    return math.sqrt(sum((c[i]-gt[i])**2 for i in range(3)))

def therm_score(s):
    return (W_THERM["breathing"] * s.get("breathing_score", 0.0) +
            W_THERM["hysteresis"] * s.get("hysteresis_asymmetry", 0.0) +
            W_THERM["onset"] * s.get("onset_score", 0.0) +
            W_THERM["kin"] * s.get("kinetic_accessibility", 0.0) +
            W_THERM["frust"] * s.get("frustrated_solvent_score", 0.0))

def coherence_score(s):
    tau = s.get("ccns_tau", 1.0)
    tau_inv = 1.0 / max(tau, 1e-6)
    return (W_COH["tau_inv"] * tau_inv +
            W_COH["asym"] * s.get("relative_asymmetry", 0.0) +
            W_COH["offset"] * s.get("asymmetry_offset", 0.0))

def combined_score(s):
    return therm_score(s) + coherence_score(s)

def load_gt():
    with open(GT_PATH) as f:
        gt_raw = json.load(f)
    with open(MANIFEST_PATH) as f:
        mdata = json.load(f)
    manifest = mdata.get('targets', mdata) if isinstance(mdata, dict) else mdata
    gt_map = {}
    for m in manifest:
        if isinstance(m, dict):
            pdb = m.get('apo_pdb','').lower()
            bid = str(m.get('id',''))
            if bid in gt_raw:
                gt_map[pdb] = gt_raw[bid].get('centroid', [0,0,0])
    gt_map.update(HARD_GT)
    return gt_map


def run():
    gt_map = load_gt()
    from pathlib import Path

    # Find all binding_sites.json
    bs_files = {}
    for d in [Path('/tmp/prism_bench10_scratch'), Path('/tmp/prism_hard_targets')]:
        if not d.exists(): continue
        for bs in d.glob('*/*.binding_sites.json'):
            name = bs.name.split('.')[0]
            if name in gt_map:
                bs_files[name] = bs

    targets = ['1hcl', '1nna', '1jwp', '4ey4', '2hnp', '3l3n', '2npq', '1p38', '5lar']

    print("=== COHERENCE LAYER TEST ===")
    print(f"{'Target':>6} | {'Therm':>12} | {'Therm+Coh':>12} | {'Before':>8} {'After':>6} | Decision")
    print("-" * 75)

    for name in targets:
        if name not in bs_files: continue
        gt = gt_map[name]

        with open(bs_files[name]) as f:
            data = json.load(f)
        sites = data if isinstance(data, list) else data.get('sites', [])
        if not sites: continue

        # Add metrics
        for s in sites:
            c = s.get('centroid', [0,0,0])
            s['_dcc'] = dcc(c, gt)
            s['_therm'] = therm_score(s)
            s['_coh'] = coherence_score(s)
            s['_combo'] = combined_score(s)

        # Sort by gtck_rank
        ranked = sorted(sites, key=lambda s: s.get('gtck_rank', 999))
        top1 = ranked[0]
        best = min(sites, key=lambda s: s['_dcc'])
        best_rank = next(i+1 for i, s in enumerate(ranked) if s.get('id') == best.get('id'))

        # Therm-only override
        topk = ranked[:TOP_K]
        therm_top1 = top1['_therm']
        therm_best_in_topk = max(topk, key=lambda s: s['_therm'])
        t_ratio = therm_best_in_topk['_therm'] / therm_top1 if therm_top1 > 0.01 else 0
        t_gap = therm_best_in_topk['_therm'] - therm_top1
        therm_decision = "OVERRIDE" if t_ratio > OVERRIDE_RATIO and t_gap > OVERRIDE_GAP else "KEEP"

        # Therm+Coherence override
        combo_top1 = top1['_combo']
        combo_best_in_topk = max(topk, key=lambda s: s['_combo'])
        c_ratio = combo_best_in_topk['_combo'] / combo_top1 if combo_top1 > 0.01 else 0
        c_gap = combo_best_in_topk['_combo'] - combo_top1
        combo_decision = "OVERRIDE" if c_ratio > OVERRIDE_RATIO and c_gap > OVERRIDE_GAP else "KEEP"

        # Result
        therm_new_rank = 1 if therm_decision == "OVERRIDE" else best_rank
        combo_new_rank = 1 if combo_decision == "OVERRIDE" else best_rank

        # Check if combo promotes the TRUE pocket
        combo_promoted_dcc = combo_best_in_topk['_dcc'] if combo_decision == "OVERRIDE" else top1['_dcc']
        therm_promoted_dcc = therm_best_in_topk['_dcc'] if therm_decision == "OVERRIDE" else top1['_dcc']

        print(f"{name:>6} | r={t_ratio:.2f} g={t_gap:+.2f} {therm_decision:>4} | "
              f"r={c_ratio:.2f} g={c_gap:+.2f} {combo_decision:>4} | "
              f"R{best_rank:>2}→R{combo_new_rank} | "
              f"top1_dcc={top1['_dcc']:.1f} best={best['_dcc']:.1f} promoted={combo_promoted_dcc:.1f}")

    print()
    print("OVERRIDE = would promote a different site to #1")
    print("KEEP = no intervention")


if __name__ == "__main__":
    run()
