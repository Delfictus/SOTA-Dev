import json, os, glob, math
from collections import defaultdict

# Known active site centers
known_sites = {
    "1btl": {"name": "TEM-1 active site", "center": (8.3, 5.9, 39.6), "criterion": 5.0, "nearest_aromatic": ("PHE72", 5.7)},
    "1w50": {"name": "BACE1 dyad", "center": (65.6, 49.8, 3.1), "criterion": 8.5, "nearest_aromatic": ("TYR199", 8.5)},
    "1maz": {"name": "BH3 groove", "center": (0.70, 12.56, 41.67), "criterion": 5.0, "nearest_aromatic": ("PHE97", 2.0)},
    "1ade": {"name": "IMP/GTP pocket", "center": (34.0, 47.0, 28.0), "criterion": 8.0, "nearest_aromatic": ("TYR269", 7.2)},
    "3k5v_atp": {"name": "ATP site", "center": (17.2, 7.8, 25.4), "criterion": 10.0, "nearest_aromatic": ("PHE401", 4.9)},
    "3k5v_allo": {"name": "Allosteric site", "center": (0.8, 26.8, 14.2), "criterion": 10.0, "nearest_aromatic": ("TYR454", 7.7)},
}

def dist(a, b):
    return math.sqrt(sum((x-y)**2 for x,y in zip(a,b)))

all_data = defaultdict(list)
# Adjust path if results are elsewhere
for f in sorted(glob.glob("/tmp/val_results/*.json")):
    try:
        target = os.path.basename(f).split("_rep")[0]
        rep = os.path.basename(f).split("_rep")[1].replace(".json","")
        with open(f) as fh:
            data = json.load(fh)
        
        # Robustly find the list of sites
        # Prioritize 'clustered_sites' or 'binding_sites' over 'sites' which might be a count
        found_sites = data.get("clustered_sites") or data.get("binding_sites")
        if not isinstance(found_sites, list):
            potential_list = data.get("sites")
            found_sites = potential_list if isinstance(potential_list, list) else []
            
        all_data[target].append({"rep": rep, "sites": found_sites})
    except Exception as e:
        print(f"Error loading {f}: {e}")

print("="*120 + "\nPRISM-4D VALIDATION: FULL METRIC MATRIX\n" + "="*120)

for target in ["1crn", "1btl", "1w50", "1maz", "1ade", "3k5v", "1l2y"]:
    reps = all_data.get(target, [])
    print(f"\nTARGET: {target} | Replicates: {len(reps)}\n" + "="*80)
    
    check_sites = {}
    if target == "3k5v":
        check_sites = {"ATP": known_sites["3k5v_atp"], "ALLO": known_sites["3k5v_allo"]}
    elif target in known_sites:
        check_sites = {"PRIMARY": known_sites[target]}

    for rd in reps:
        n_sites = len(rd['sites'])
        print(f"\n --- Replicate {rd['rep']}: {n_sites} sites detected ---")
        
        if not check_sites:
            continue
            
        for ref_n, ref_i in check_sites.items():
            print(f"    vs {ref_n} ({ref_i['name']}) @ {ref_i['center']}")
            
            scored = []
            for si in rd['sites']:
                c = si.get("centroid") or si.get("center") or [0,0,0]
                d = dist(c, ref_i["center"])
                status = "PASS" if d < ref_i["criterion"] else ("NEAR" if d < ref_i["criterion"]*1.5 else "FAIL")
                scored.append((d, si.get("spike_count", 0), si.get("quality_score", 0), si.get("druggability", {}).get("is_druggable", False), status))
            
            if scored:
                print(f"    Rank  Dist    Spikes  Quality   Druggable  Status")
                scored.sort(key=lambda x: x[0])
                for r, s in enumerate(scored[:5], 1):
                    print(f"    {r:>2}   {s[0]:>6.2f}A {s[1]:>8} {s[2]:>8.3f}   {str(s[3]):>10}  {s[4]:>6}")
            else:
                print("    No spatial site data available in this replicate.")
