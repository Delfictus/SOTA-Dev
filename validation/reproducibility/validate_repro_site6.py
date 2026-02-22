import json, numpy as np, os

ANCHOR_RESIDUES = {"TRP242", "CYS240", "CYS278", "LYS204"}
REGION_MIN, REGION_MAX = 204, 278

results = []

for seed in range(1, 6):
    path = f"/tmp/sndc_benchmark/2gl7_repro/seed{seed}/2gl7.binding_sites.json"
    if not os.path.exists(path):
        print(f"Seed {seed}: FILE MISSING"); continue
    
    with open(path) as f:
        data = json.load(f)
    
    # Find Site 6 equivalent: site with TRP242 anchor + highest quality in ARM 2-5 region
    candidates = []
    for site in data["sites"]:
        cx, cy, cz = site["centroid"]
        lining = site.get("lining_residues", [])
        
        res_names = set()
        res_nums = []
        for r in lining:
            rname = r.get("resname", "")
            rnum = r.get("resid", 0)
            res_names.add(f"{rname}{rnum}")
            res_nums.append(rnum)
        
        if not res_nums:
            continue
        avg_resnum = np.mean(res_nums)
        anchor_hits = ANCHOR_RESIDUES & res_names
        
        # Must have TRP242 or at least 2 anchors
        if "TRP242" in res_names or len(anchor_hits) >= 2:
            candidates.append({
                "site_id": site.get("id", "?"),
                "centroid": np.array([cx, cy, cz]),
                "quality": site.get("quality_score", 0) or 0,
                "druggability": site.get("druggability", 0),
                "spikes": site.get("spike_count", 0),
                "volume": site.get("volume", 0),
                "classification": site.get("classification", "?"),
                "n_lining": len(lining),
                "anchor_hits": anchor_hits,
                "all_residues": res_names,
            })
    
    if candidates:
        best = max(candidates, key=lambda x: len(x["anchor_hits"]) * 100 + x["druggability"])
        results.append({"seed": seed, **best})
        print(f"Seed {seed}: Site {best['site_id']} | quality={best['quality']:.3f} | drug={best['druggability']:.3f} | spikes={best['spikes']} | vol={best['volume']:.0f}A³ | class={best['classification']}")
        print(f"  Anchors: {sorted(best['anchor_hits'])} ({len(best['anchor_hits'])}/4)")
        print(f"  Centroid: ({best['centroid'][0]:.1f}, {best['centroid'][1]:.1f}, {best['centroid'][2]:.1f})")
    else:
        print(f"Seed {seed}: NO MATCHING SITE FOUND")

print("\n" + "="*70)
print("REPRODUCIBILITY SUMMARY")
print("="*70)

if len(results) >= 2:
    centroids = np.array([r["centroid"] for r in results])
    mean_c = centroids.mean(axis=0)
    std_c = centroids.std(axis=0)
    dccs = []
    for i in range(len(centroids)):
        for j in range(i+1, len(centroids)):
            dccs.append(np.linalg.norm(centroids[i] - centroids[j]))
    
    drugs = [r["druggability"] for r in results]
    spikes = [r["spikes"] for r in results]
    
    all_anchors = set.intersection(*[r["anchor_hits"] for r in results])
    any_anchors = set.union(*[r["anchor_hits"] for r in results])
    
    print(f"Runs with matching site: {len(results)}/5")
    print(f"Mean centroid: ({mean_c[0]:.1f}, {mean_c[1]:.1f}, {mean_c[2]:.1f})")
    print(f"Centroid std:  ({std_c[0]:.2f}, {std_c[1]:.2f}, {std_c[2]:.2f})")
    print(f"Max pairwise DCC: {max(dccs):.2f}A")
    print(f"Mean pairwise DCC: {np.mean(dccs):.2f}A")
    print(f"Druggability: {np.mean(drugs):.3f} +/- {np.std(drugs):.3f}")
    print(f"Spikes: {np.mean(spikes):.0f} +/- {np.std(spikes):.0f}")
    print(f"Anchors in ALL runs: {sorted(all_anchors)} ({len(all_anchors)}/4)")
    print(f"Anchors in ANY run:  {sorted(any_anchors)} ({len(any_anchors)}/4)")
    
    if len(all_anchors) >= 3 and max(dccs) < 5.0:
        verdict = "★★★ HIGHLY REPRODUCIBLE"
    elif len(all_anchors) >= 2 and max(dccs) < 8.0:
        verdict = "★★ REPRODUCIBLE"
    elif len(any_anchors) >= 2:
        verdict = "★ PARTIALLY REPRODUCIBLE"
    else:
        verdict = "✗ NOT REPRODUCIBLE"
    print(f"\nVERDICT: {verdict}")
else:
    print("Insufficient data for analysis")
