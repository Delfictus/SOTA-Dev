import json, glob, os, math

# --- CONFIG ---
BASE_DIR = os.path.expanduser("~/Desktop/Prism4D-bio")
RESULTS_DIR = f"{BASE_DIR}/benchmarks/prism4d_bench30/results"
GT_FILE = f"{BASE_DIR}/benchmarks/prism4d_bench30/ground_truth/ligand_centroids.json"
MANIFEST_FILE = f"{BASE_DIR}/benchmarks/prism4d_bench30/benchmark_manifest.json"

def get_dist(p1, p2):
    return math.sqrt(sum((a - b)**2 for a, b in zip(p1, p2)))

# 1. Load Ground Truth
with open(GT_FILE, 'r') as f:
    gt_data = json.load(f)

# 2. Load Manifest (Fixed for your Dictionary structure)
with open(MANIFEST_FILE, 'r') as f:
    manifest = json.load(f)

print("\n" + "="*75)
print(" PRISM-4D COHERENCE AUDIT (Druggability * Coupling)")
print("="*75)
print(f"{'ID':<4} | {'HOLO':<6} | {'SITE':<6} | {'DRUG':<5} | {'COUPLE':<6} | {'DCC (Å)':<8} | {'RESULT'}")
print("-" * 75)

stats = {"hits": 0, "total": 0}
curve = {t: 0 for t in range(1, 11)}

for folder_id in sorted(os.listdir(RESULTS_DIR), key=lambda x: int(x) if x.isdigit() else 999):
    path = os.path.join(RESULTS_DIR, folder_id)
    if not os.path.isdir(path): continue
    
    # We use prism_therm because it has the exact Ranker fields we need
    therm_files = glob.glob(os.path.join(path, "*.prism_therm.json"))
    if not therm_files: continue
    
    try:
        with open(therm_files[0], 'r') as f:
            data = json.load(f)
            # Use 'pockets' as seen in your 'head' output
            pockets = data.get('pockets', [])
            
            # COHERENCE RANKER: druggability_score * tide_coupling_score
            ranked = sorted(pockets, key=lambda p: p.get('druggability_score', 0) * p.get('tide_coupling_score', 0), reverse=True)
            
            if ranked:
                # Get the Holo PDB code from the manifest using the folder ID
                # Handles dict structure like {"1": {"holo": "1pzo"}}
                manifest_entry = manifest.get(folder_id)
                if not manifest_entry: continue
                
                holo_name = manifest_entry['holo'].lower() if isinstance(manifest_entry, dict) else manifest_entry.lower()
                
                # Get Ground Truth (keyed by folder ID "1", "2", etc. or Holo name)
                gt = gt_data.get(folder_id) or gt_data.get(holo_name)
                
                if gt:
                    stats["total"] += 1
                    # Extract centroid - check if it's a list or a dict
                    gt_centroid = gt['centroid'] if isinstance(gt, dict) else gt
                    dcc = get_dist(ranked[0]['centroid'], gt_centroid)
                    
                    res = "SUCCESS" if dcc < 5.0 else "FAIL"
                    if dcc < 5.0: stats["hits"] += 1
                    for t in curve:
                        if dcc < t: curve[t] += 1

                    print(f"{folder_id:<4} | {holo_name:<6} | {ranked[0]['pocket_id']:<6} | "
                          f"{ranked[0]['druggability_score']:<5.2f} | {ranked[0]['tide_coupling_score']:<6.2f} | "
                          f"{dcc:<8.2f} | {res}")
    except Exception:
        continue

if stats["total"] > 0:
    n = stats["total"]
    print("-" * 75)
    print(f"COHERENCE RANKER SR@1: {(stats['hits']/n)*100:.1f}% ({stats['hits']}/{n})")
    print(f"SR @ 4.0Å: {(curve[4]/n)*100:.1f}%")
    print("="*75)
