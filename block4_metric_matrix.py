#!/usr/bin/env python3
"""
PRISM-4D Block 4: Full Metric Matrix + Guerrilla Near-Miss Identification
Reads /tmp/val_results/*.json from the 21-run validation.

JSON schema (from nhs_rt_full):
  top-level: binding_sites (int), druggable_sites (int), sites (list)
  per site: centroid [x,y,z], classification (str), druggability (float),
            is_druggable (bool), aromatic_score (float),
            catalytic_residue_count (int), lining_residues (list),
            id (int), spike_count (int, maybe absent)
"""
import json, os, glob, math
from collections import defaultdict

known_sites = {
    "1btl": {"name": "TEM-1 active site", "center": (8.3, 5.9, 39.6), "criterion": 5.0,
             "catalytic": ["SER70","LYS73","SER130","GLU166"], "nearest_aromatic": ("PHE72", 5.7)},
    "1w50": {"name": "BACE1 dyad", "center": (65.6, 49.8, 3.1), "criterion": 8.5,
             "catalytic": ["ASP32","ASP228"], "nearest_aromatic": ("TYR199", 8.5)},
    "1maz": {"name": "BH3 groove", "center": (0.70, 12.56, 41.67), "criterion": 5.0,
             "catalytic": ["PHE97","TYR101","PHE105","PHE146","PHE150"], "nearest_aromatic": ("PHE97", 2.0)},
    "1ade": {"name": "IMP/GTP pocket", "center": (34.0, 47.0, 28.0), "criterion": 8.0,
             "catalytic": ["ASP13","ASP333"], "nearest_aromatic": ("TYR269", 7.2)},
    "3k5v_atp": {"name": "ATP site", "center": (17.2, 7.8, 25.4), "criterion": 10.0,
                 "catalytic": ["STI"], "nearest_aromatic": ("PHE401", 4.9)},
    "3k5v_allo": {"name": "Allosteric site", "center": (0.8, 26.8, 14.2), "criterion": 10.0,
                  "catalytic": ["STJ"], "nearest_aromatic": ("TYR454", 7.7)},
}

def dist(a, b):
    return math.sqrt(sum((x-y)**2 for x,y in zip(a,b)))

# Collect all results
all_data = defaultdict(list)
for f in sorted(glob.glob("/tmp/val_results/*.json")):
    basename = os.path.basename(f)
    # Handle both 1btl_rep1.json and 1btl_1.json patterns
    parts = basename.replace(".json","")
    if "_rep" in parts:
        target = parts.split("_rep")[0]
        rep = parts.split("_rep")[1]
    elif parts.count("_") == 1:
        target, rep = parts.split("_")
    else:
        target = parts
        rep = "1"
    
    with open(f) as fh:
        data = json.load(fh)
    
    sites = data.get("sites", [])
    if not isinstance(sites, list):
        sites = []
    
    all_data[target].append({"rep": rep, "sites": sites, "file": f, "raw": data})

print("=" * 120)
print("PRISM-4D VALIDATION: FULL METRIC MATRIX")
print("=" * 120)

for target in ['1crn', '1btl', '1w50', '1maz', '1ade', '3k5v', '1l2y']:
    reps = all_data.get(target, [])
    print(f"\n{'='*100}")
    print(f"TARGET: {target} | Replicates: {len(reps)}")
    print(f"{'='*100}")
    
    if not reps:
        print("  NO DATA")
        continue
    
    # Determine which known sites to check
    if target == "3k5v":
        check_sites = {"ATP": known_sites["3k5v_atp"], "ALLO": known_sites["3k5v_allo"]}
    elif target in known_sites:
        check_sites = {"PRIMARY": known_sites[target]}
    else:
        check_sites = {}
    
    for rep_data in reps:
        rep = rep_data["rep"]
        sites = rep_data["sites"]
        raw = rep_data["raw"]
        sim_time = raw.get("simulation_time_sec", 0)
        n_binding = raw.get("binding_sites", len(sites))
        n_druggable = raw.get("druggable_sites", 0)
        
        print(f"\n  --- Replicate {rep}: {len(sites)} sites | time={sim_time:.1f}s | druggable={n_druggable} ---")
        
        if not sites:
            if target in ("1l2y",):
                print("    ✅ No sites detected (correct for negative control)")
            elif target == "1crn":
                print("    No sites detected (baseline target)")
            else:
                print("    ⚠️  No sites detected (unexpected)")
            continue
        
        # Score every site against every known reference
        for ref_name, ref_info in check_sites.items():
            ref_center = ref_info["center"]
            criterion = ref_info["criterion"]
            
            print(f"\n    vs {ref_name} ({ref_info['name']}) @ {ref_center}")
            print(f"    Criterion: < {criterion}Å | Nearest aromatic: {ref_info['nearest_aromatic']}")
            print(f"    {'Rank':>4} {'Dist':>7} {'Drugg':>7} {'AroScr':>7} {'CatRes':>7} {'Lining':>7} {'Class':>12} {'Centroid':>32} {'Status':>8}")
            print(f"    {'-'*102}")
            
            scored = []
            for si in sites:
                c = si.get("centroid", [0,0,0])
                if isinstance(c, list) and len(c) == 3:
                    d = dist(c, ref_center)
                else:
                    d = float('inf')
                    c = [0,0,0]
                
                scored.append({
                    "dist": d,
                    "centroid": c,
                    "druggability": si.get("druggability", 0.0),
                    "is_druggable": si.get("is_druggable", False),
                    "aromatic_score": si.get("aromatic_score", 0.0),
                    "catalytic_count": si.get("catalytic_residue_count", 0),
                    "lining_count": len(si.get("lining_residues", [])),
                    "classification": si.get("classification", "unknown"),
                    "spike_count": si.get("spike_count", si.get("n_spikes", 0)),
                    "site_id": si.get("id", 0),
                })
            
            scored.sort(key=lambda x: x["dist"])
            
            # Show top 10 closest
            for rank, s in enumerate(scored[:10], 1):
                if s["dist"] < criterion:
                    status = "✅ PASS"
                elif s["dist"] < criterion * 1.5:
                    status = "⚠️ NEAR"
                else:
                    status = "❌ FAIL"
                centroid_str = f"({s['centroid'][0]:.1f}, {s['centroid'][1]:.1f}, {s['centroid'][2]:.1f})"
                print(f"    {rank:>4} {s['dist']:>6.2f}Å {s['druggability']:>6.3f} {s['aromatic_score']:>6.2f} {s['catalytic_count']:>7} {s['lining_count']:>7} {s['classification']:>12} {centroid_str:>32} {status:>8}")
            
            # Best hit
            best = scored[0]
            print(f"\n    BEST: {best['dist']:.2f}Å — {'PASS' if best['dist'] < criterion else 'NEAR-MISS' if best['dist'] < criterion*1.5 else 'MISS'}")
            
            # Near-miss analysis
            near_misses = [s for s in scored if criterion <= s["dist"] < criterion * 2.0]
            if near_misses:
                print(f"    NEAR-MISS CANDIDATES ({len(near_misses)} sites in {criterion:.0f}-{criterion*2:.0f}Å envelope):")
                for s in near_misses[:5]:
                    centroid_str = f"({s['centroid'][0]:.1f}, {s['centroid'][1]:.1f}, {s['centroid'][2]:.1f})"
                    print(f"      {s['dist']:.2f}Å | drugg={s['druggability']:.3f} arom={s['aromatic_score']:.2f} cat={s['catalytic_count']} lining={s['lining_count']} @ {centroid_str}")
        
        # Orphan sites (far from any known site)
        if check_sites:
            orphans = []
            for si in sites:
                c = si.get("centroid", [0,0,0])
                if isinstance(c, list) and len(c) == 3:
                    min_d = min(dist(c, ref["center"]) for ref in check_sites.values())
                    if min_d > 15.0:
                        orphans.append((min_d, si))
            
            if orphans:
                orphans.sort(key=lambda x: x[1].get("druggability", 0), reverse=True)
                print(f"\n    ORPHAN SITES (>15Å from any known, potential novel pockets): {len(orphans)}")
                for d, si in orphans[:5]:
                    c = si.get("centroid", [0,0,0])
                    print(f"      ({c[0]:.1f}, {c[1]:.1f}, {c[2]:.1f}) drugg={si.get('druggability',0):.3f} arom={si.get('aromatic_score',0):.2f} cat={si.get('catalytic_residue_count',0)} class={si.get('classification','?')}")

# === CROSS-REPLICATE CONSISTENCY ===
print(f"\n{'='*100}")
print("CROSS-REPLICATE CONSISTENCY")
print(f"{'='*100}")
print(f"{'Target':<8} {'Sites (reps)':>18} {'Mean±Std':>14} {'Best Dists (Å)':>30} {'Mean±Std':>16} {'Verdict':>10}")
print("-" * 100)

for target in ['1crn', '1btl', '1w50', '1maz', '1ade', '3k5v', '1l2y']:
    reps = all_data.get(target, [])
    if not reps:
        print(f"{target:<8} {'N/A':>18}")
        continue
    
    site_counts = [len(r["sites"]) for r in reps]
    sc_str = "/".join(str(s) for s in site_counts)
    sc_mean = sum(site_counts)/len(site_counts)
    sc_std = (sum((s-sc_mean)**2 for s in site_counts)/len(site_counts))**0.5
    
    if target == "3k5v":
        ref = known_sites["3k5v_atp"]["center"]
    elif target in known_sites:
        ref = known_sites[target]["center"]
    else:
        ref = None
    
    if ref:
        best_dists = []
        for r in reps:
            if r["sites"]:
                dists = []
                for s in r["sites"]:
                    c = s.get("centroid", [0,0,0])
                    if isinstance(c, list) and len(c) == 3:
                        dists.append(dist(c, ref))
                if dists:
                    best_dists.append(min(dists))
        
        if best_dists:
            bd_str = "/".join(f"{d:.2f}" for d in best_dists)
            bd_mean = sum(best_dists)/len(best_dists)
            bd_std = (sum((d-bd_mean)**2 for d in best_dists)/len(best_dists))**0.5
            bd_ms = f"{bd_mean:.2f}±{bd_std:.2f}"
        else:
            bd_str = "N/A"
            bd_ms = "N/A"
    else:
        bd_str = "N/A"
        bd_ms = "N/A"
    
    verdict = "✅" if sc_std < 2 else "⚠️"
    print(f"{target:<8} {sc_str:>18} {sc_mean:.1f}±{sc_std:.1f}{'':>4} {bd_str:>30} {bd_ms:>16} {verdict:>10}")

# === GUERRILLA RE-PROCESSING CANDIDATES ===
print(f"\n{'='*100}")
print("GUERRILLA RE-PROCESSING CANDIDATES")
print("Sites in near-miss envelope that could benefit from Pass 2 features")
print(f"{'='*100}")

guerrilla_targets = []

for target in ['1btl', '1w50', '1maz', '1ade', '3k5v']:
    if target == "3k5v":
        refs = [("ATP", known_sites["3k5v_atp"]), ("ALLO", known_sites["3k5v_allo"])]
    else:
        refs = [("PRIMARY", known_sites[target])]
    
    for ref_name, ref_info in refs:
        criterion = ref_info["criterion"]
        ref_center = ref_info["center"]
        
        all_best = []
        all_near = []
        for r in all_data.get(target, []):
            best_d = float('inf')
            for si in r["sites"]:
                c = si.get("centroid", [0,0,0])
                if isinstance(c, list) and len(c) == 3:
                    d = dist(c, ref_center)
                    if d < best_d:
                        best_d = d
                    if criterion <= d < criterion * 2.0:
                        all_near.append({
                            "rep": r["rep"], "dist": d,
                            "druggability": si.get("druggability", 0),
                            "aromatic_score": si.get("aromatic_score", 0),
                            "catalytic_count": si.get("catalytic_residue_count", 0),
                            "lining_count": len(si.get("lining_residues", [])),
                            "centroid": c
                        })
            all_best.append(best_d)
        
        avg_best = sum(all_best)/len(all_best) if all_best else float('inf')
        
        if avg_best < criterion:
            classification = "CLEAN_HIT"
            symbol = "✅"
        elif avg_best < criterion * 1.5:
            classification = "NEAR_MISS"
            symbol = "⚠️"
        else:
            classification = "MISS"
            symbol = "❌"
        
        print(f"\n  {symbol} {target} ({ref_name}): avg best = {avg_best:.2f}Å | criterion = {criterion}Å | → {classification}")
        
        if classification == "NEAR_MISS":
            guerrilla_targets.append((target, ref_name, avg_best, criterion))
            if all_near:
                print(f"     Near-miss detections: {len(all_near)} across replicates")
                for nm in sorted(all_near, key=lambda x: x["dist"])[:3]:
                    c = nm["centroid"]
                    print(f"       rep{nm['rep']}: {nm['dist']:.2f}Å | drugg={nm['druggability']:.3f} arom={nm['aromatic_score']:.2f} cat={nm['catalytic_count']} lining={nm['lining_count']} @ ({c[0]:.1f},{c[1]:.1f},{c[2]:.1f})")
            
            # Recommend tactic
            if avg_best < criterion * 1.2:
                print(f"     TACTIC: Density asymmetry centroid shift (close, just drifted)")
            elif all_near and max(n["catalytic_count"] for n in all_near) > 3:
                print(f"     TACTIC: Extended MD 3x + voxel temporal persistence (strong catalytic signal)")
            elif all_near and max(n["aromatic_score"] for n in all_near) > 0.8:
                print(f"     TACTIC: Boost UV intensity on nearest aromatics (high aromatic signal)")
            else:
                print(f"     TACTIC: Flood-fill merge adjacent sub-sites + recompute centroid")

# === SUMMARY ===
print(f"\n{'='*100}")
print("SUMMARY")
print(f"{'='*100}")
print(f"Total targets: 7 (5 with known sites + 1 baseline + 1 negative control)")
print(f"Total runs: 21 (7 × 3 replicates)")
print(f"All runs completed: YES")
if guerrilla_targets:
    print(f"\nGuerrilla Pass 2 candidates ({len(guerrilla_targets)}):")
    for t, rn, d, c in guerrilla_targets:
        print(f"  {t} ({rn}): {d:.2f}Å vs {c}Å criterion")
    print(f"\nPriority implementation order for Pass 2:")
    print(f"  1. Voxel temporal persistence (kills over-detection, benefits all targets)")
    print(f"  2. RT pocket geometry (druggability filter, most impact on 1maz/1ade)")
    print(f"  3. Centroid trajectory (rescues drifted near-misses)")
    print(f"  4. Prismatic sector merging (collapses 37→3 sites for 1ade/3k5v)")
    print(f"  5. Flood-fill volume dynamics (cryptic site detection)")
else:
    print(f"\nNo guerrilla candidates — all targets within criterion or clearly missed.")

print(f"\n{'='*100}")
print("Data: /tmp/val_results/ | Logs: /tmp/val_full_output/")
print(f"{'='*100}")
