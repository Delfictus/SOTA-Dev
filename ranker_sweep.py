import json, math, os, glob

# --- CONFIG ---
BASE_DIR = os.path.expanduser("~/Desktop/Prism4D-bio")
RESULTS_DIR = f"{BASE_DIR}/benchmarks/prism4d_bench30/results"
GT_FILE = f"{BASE_DIR}/benchmarks/prism4d_bench30/ground_truth/ligand_centroids.json"

def get_dist(p1, p2):
    return math.sqrt(sum((a - b)**2 for a, b in zip(p1, p2)))

with open(GT_FILE, 'r') as f:
    gt_data = json.load(f)

# Define Candidate Rankers
# Note: We test both standard and inverted (1/x) for coherence/breathing
rankers = {
    "vol*drug": lambda s: s.get('volume', 0) * s.get('druggability', 0),
    "drug/coh": lambda s: s.get('druggability', 0) / (s.get('wd_coherence', 0) + 1e-15),
    "drug*couple": lambda s: s.get('druggability', 0) * s.get('tide_coupling_score', 0),
    "1/coh": lambda s: 1.0 / (s.get('wd_coherence', 0) + 1e-15),
    "onset*drug": lambda s: s.get('onset_score', 0) * s.get('druggability', 0),
    "quality": lambda s: s.get('quality_score', 0),
    "eng_chem*burial": lambda s: s.get('engine_chem', 0) * s.get('burial_score', 0),
    "inverse_coh_drug": lambda s: s.get('druggability', 0) * (1.0 - s.get('wd_coherence', 0))
}

results = {name: {"sr1": 0, "sr3": 0, "total": 0, "dccs": []} for name in rankers}

for folder in sorted(os.listdir(RESULTS_DIR), key=lambda x: int(x) if x.isdigit() else 999):
    path = os.path.join(RESULTS_DIR, folder)
    if not os.path.isdir(path): continue
    
    bs_files = glob.glob(os.path.join(path, "*.binding_sites.json"))
    if not bs_files: continue
    
    gt = gt_data.get(folder)
    if not gt: continue
    gt_c = gt['centroid']
    
    with open(bs_files[0], 'r') as f:
        data = json.load(f)
        sites = data.get('sites', [])
        if not sites: continue
        
        for name, formula in rankers.items():
            try:
                # Rank sites based on formula
                ranked = sorted(sites, key=formula, reverse=True)
                dcc = get_dist(ranked[0]['centroid'], gt_c)
                
                results[name]["total"] += 1
                results[name]["dccs"].append(dcc)
                if dcc < 5.0: results[name]["sr1"] += 1
                if any(get_dist(s['centroid'], gt_c) < 5.0 for s in ranked[:3]):
                    results[name]["sr3"] += 1
            except: continue

print(f"\n{'RANKER':<20} | {'SR@1 (%)':<10} | {'SR@3 (%)':<10} | {'Mean DCC':<10}")
print("-" * 60)
for name, r in results.items():
    if r['total'] > 0:
        sr1 = (r['sr1'] / r['total']) * 100
        sr3 = (r['sr3'] / r['total']) * 100
        avg_dcc = sum(r['dccs']) / len(r['dccs'])
        print(f"{name:<20} | {sr1:<10.1f} | {sr3:<10.1f} | {avg_dcc:<10.2f}")
