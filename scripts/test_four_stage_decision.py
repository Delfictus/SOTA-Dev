#!/usr/bin/env python3
"""
PRISM4D Four-Stage Decision Validation
GTCKL → Therm → Coherence → Localization

Validates that the localization gate correctly blocks false Therm overrides
(e.g., 2HNP) while preserving correct overrides (e.g., 1JWP).
"""
import json, math
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
gt_map.update({'1p38': [16.885,-3.551,-19.552], '5lar': [54.0,68.1,16.8], '3mh1': [34.023,35.443,16.502]})

def dcc(c, gt): return math.sqrt(sum((c[i]-gt[i])**2 for i in range(3)))
def therm_score(s):
    return (1.0*s.get("breathing_score",0) + 1.0*s.get("hysteresis_asymmetry",0) +
            0.5*s.get("onset_score",0) + 0.3*s.get("kinetic_accessibility",0) -
            0.5*s.get("frustrated_solvent_score",0))
def norm_asym(v): return v / (1.0 + abs(v))
def norm_offset(v, cap=5.0): return min(abs(v), cap) / cap
def coherence_ok(s):
    ts = therm_score(s)
    tau = s.get("ccns_tau", 1.0)
    asym_n = norm_asym(s.get("relative_asymmetry", 0))
    return ts > 1.5 and 1.0 < tau < 2.5 and (asym_n > 0.3 or norm_offset(s.get("asymmetry_offset",0)) > 0.3)
def loc_score(s):
    vol = s.get("volume", 500)
    burial = s.get("burial_score", 0.5)
    enclosure = s.get("druggability", 0.5)
    depth = s.get("mean_burial", 3.0)
    return (1.0*min(max(enclosure,0),1) + 1.0*(depth/(depth+5.0)) +
            0.5*min(max(burial,0),1) + 0.5*(vol/(vol+500.0)))

bs_files = {}
for d in [Path('/tmp/prism_bench10_scratch'), Path('/tmp/prism_hard_targets')]:
    if not d.exists(): continue
    for bs in d.glob('*/*.binding_sites.json'):
        name = bs.name.split('.')[0]
        if name in gt_map: bs_files[name] = bs

targets = ['1jwp', '1p38', '5lar', '2hnp', '3l3n', '4ey4', '1hcl', '1nna', '2npq']
print("=== FOUR-STAGE DECISION VALIDATION ===\n")
for name in targets:
    if name not in bs_files: continue
    gt = gt_map[name]
    with open(bs_files[name]) as f:
        data = json.load(f)
    sites = data if isinstance(data, list) else data.get('sites', [])
    for s in sites:
        s['_dcc'] = dcc(s.get('centroid',[0,0,0]), gt)
        s['_therm'] = therm_score(s)
        s['_loc'] = loc_score(s)
    ranked = sorted(sites, key=lambda s: s.get('gtck_rank', 999))
    top1 = ranked[0]
    true_site = min(sites, key=lambda s: s['_dcc'])
    true_rank = next(i+1 for i,s in enumerate(ranked) if s.get('id')==true_site.get('id'))
    topk = ranked[:10]
    therm_best = max(topk, key=lambda s: s['_therm'])
    t_ratio = therm_best['_therm'] / top1['_therm'] if top1['_therm'] > 0.01 else 0
    t_gap = therm_best['_therm'] - top1['_therm']
    therm_fires = t_ratio > 1.5 and t_gap > 0.3 and therm_best.get('id') != top1.get('id')
    coh = coherence_ok(therm_best) if therm_fires else False
    loc_ratio = therm_best['_loc'] / top1['_loc'] if top1['_loc'] > 0.01 else 0
    loc_gap = therm_best['_loc'] - top1['_loc']
    loc_confirms = loc_ratio > 1.2 and loc_gap > 0.2
    already_correct = top1.get('id') == true_site.get('id')
    if already_correct: decision = "CORRECT"
    elif therm_fires and coh and loc_confirms: decision = "OVERRIDE"
    elif therm_fires and coh: decision = "BLOCKED(loc)"
    elif therm_fires: decision = "BLOCKED(coh)"
    else: decision = "KEEP"
    print(f"{name.upper():>5}: top1={top1['_dcc']:.1f}A true={true_site['_dcc']:.1f}A@R{true_rank} "
          f"therm_r={t_ratio:.2f} coh={'Y' if coh else 'N'} loc_r={loc_ratio:.2f} → {decision}")

if __name__ == "__main__":
    pass
