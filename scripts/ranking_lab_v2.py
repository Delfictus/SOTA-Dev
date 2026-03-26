#!/usr/bin/env python3
"""PRISM4D — Ranking Lab V2: Neuromorphic Physics Ranking.

Implements the SWISH-X/TargetX-equivalent ranking using signals
PRISM already computes. No learned weights. Physics-motivated
mappings only.

Signals mapped:
  σ_density      → spike_count / volume (firing rate density)
  stream_overlap → open_frequency (fraction of streams with pocket)
  tau_score      → f(ccns_tau) = 1/tau (criticality landscape)
  therm_score    → g(hysteresis_asymmetry, therm_class)
  persistence    → onset_score (early detection = persistent)
  burial_term    → burial_score (pocket depth)
  vcs_term       → engine_vcs (enclosure)
"""
import json, math, csv, sys
from pathlib import Path

GT = {
    "1jwp": [6.61, 13.531, 25.526], "1p38": [16.885, -3.551, -19.552],
    "5lar": [54.0, 68.1, 16.8], "2hnp": [27.09, 17.46, 12.15],
    "3l3n": [26.365, 11.222, 27.023], "1nna": [26.054, 17.671, 62.965],
    "2npq": [42.461, 31.126, 32.515],
}
PATHS = {
    "1jwp": "/tmp/prism_consensus/1jwp/rep_00/1jwp.binding_sites.json",
    "1p38": "/tmp/prism_consensus/1p38/rep_00/1p38.binding_sites.json",
    "2hnp": "/tmp/prism_consensus/2hnp/rep_00/2hnp.binding_sites.json",
    "5lar": "/tmp/prism_blind/5lar/5lar.binding_sites.json",
    "3l3n": "/tmp/prism_blind/3l3n/3l3n.binding_sites.json",
    "1nna": "/tmp/prism_blind/1nna/1nna.binding_sites.json",
    "2npq": "/tmp/prism_blind/2npq/2npq.binding_sites.json",
}

def dist(a, b): return math.sqrt(sum((a[i]-b[i])**2 for i in range(3)))

def pct_norm(vals):
    n = len(vals)
    if n <= 1: return [0.5]*n
    order = sorted(range(n), key=lambda i: vals[i])
    ranks = [0.0]*n
    for r, idx in enumerate(order): ranks[idx] = r/(n-1)
    return ranks

def compute_neuromorphic_score(sites):
    """Compute neuromorphic physics ranking score per site.
    
    All terms are physics-motivated, no GT information used.
    """
    n = len(sites)
    
    # σ_density: spike firing rate per unit volume
    # High = concentrated energy deposition
    sigma = [s.get("spike_count", 0) / max(s.get("volume", 1), 1) for s in sites]
    sigma_pct = pct_norm(sigma)
    
    # Stream overlap: open_frequency or source_diversity
    # High = multiple independent detections agree
    overlap = [s.get("source_diversity", 0) for s in sites]
    overlap_pct = pct_norm(overlap)
    
    # Tau score: 1/tau — lower tau = more critical = more druggable
    # tau~1.2-1.5 is optimal (critical), tau>2 is barrier-dominated
    tau_raw = [s.get("ccns_tau", 1.5) for s in sites]
    tau_score = [1.0 / max(t, 0.5) for t in tau_raw]
    tau_pct = pct_norm(tau_score)
    
    # Therm score: hysteresis asymmetry maps to thermodynamic signal
    # High hysteresis = pocket responds to temperature perturbation
    hyst = [s.get("hysteresis_asymmetry", 0) for s in sites]
    hyst_pct = pct_norm(hyst)
    
    # Persistence: onset_score — early spike onset = persistent pocket
    onset = [s.get("onset_score", 0) for s in sites]
    onset_pct = pct_norm(onset)
    
    # Burial: pocket depth
    burial = [s.get("burial_score", 0) for s in sites]
    burial_pct = pct_norm(burial)
    
    # VCS: voxel contact score = enclosure
    vcs = [s.get("engine_vcs", 0) for s in sites]
    vcs_pct = pct_norm(vcs)
    
    # Chemistry: engine_chem = pocket chemistry quality
    chem = [s.get("engine_chem", 0) for s in sites]
    chem_pct = pct_norm(chem)
    
    # Breathing: pocket shows open/close dynamics
    breath = [s.get("breathing_score", 0) for s in sites]
    breath_pct = pct_norm(breath)
    
    # Build formulas from these percentile-normalized signals
    formulas = {}
    
    # NF1: multiplicative product of all neuromorphic signals
    formulas["neuro_full"] = [
        (0.3 + sigma_pct[i]) * (0.3 + burial_pct[i]) * (0.3 + vcs_pct[i])
        * (0.3 + tau_pct[i]) * (0.3 + hyst_pct[i])
        for i in range(n)
    ]
    
    # NF2: burial + tau + vcs (structural + critical)
    formulas["bur+tau+vcs"] = [
        burial_pct[i] + tau_pct[i] + vcs_pct[i]
        for i in range(n)
    ]
    
    # NF3: burial * tau * vcs (multiplicative, physics core)
    formulas["bur*tau*vcs"] = [
        (0.1 + burial_pct[i]) * (0.1 + tau_pct[i]) * (0.1 + vcs_pct[i])
        for i in range(n)
    ]
    
    # NF4: sigma * burial * tau (energy density in deep critical pocket)
    formulas["sig*bur*tau"] = [
        (0.1 + sigma_pct[i]) * (0.1 + burial_pct[i]) * (0.1 + tau_pct[i])
        for i in range(n)
    ]
    
    # NF5: chem * burial * vcs (chemistry in an enclosed deep pocket)
    formulas["chem*bur*vcs"] = [
        (0.1 + chem_pct[i]) * (0.1 + burial_pct[i]) * (0.1 + vcs_pct[i])
        for i in range(n)
    ]
    
    # NF6: rank fusion — sum of ranks across burial + tau + vcs + chem
    def rf_ranks(pcts):
        n = len(pcts)
        order = sorted(range(n), key=lambda i: pcts[i], reverse=True)
        ranks = [0]*n
        for r, idx in enumerate(order): ranks[idx] = r+1
        return ranks
    
    bur_r = rf_ranks(burial_pct)
    tau_r = rf_ranks(tau_pct)
    vcs_r = rf_ranks(vcs_pct)
    chem_r = rf_ranks(chem_pct)
    sig_r = rf_ranks(sigma_pct)
    hyst_r = rf_ranks(hyst_pct)
    onset_r = rf_ranks(onset_pct)
    breath_r = rf_ranks(breath_pct)
    
    formulas["RF:bur+tau+vcs+chem"] = [
        -(bur_r[i] + tau_r[i] + vcs_r[i] + chem_r[i]) for i in range(n)
    ]
    
    formulas["RF:bur+vcs+sig+hyst"] = [
        -(bur_r[i] + vcs_r[i] + sig_r[i] + hyst_r[i]) for i in range(n)
    ]
    
    formulas["RF:bur+tau+chem+onset"] = [
        -(bur_r[i] + tau_r[i] + chem_r[i] + onset_r[i]) for i in range(n)
    ]
    
    formulas["RF:bur+vcs+tau+breath"] = [
        -(bur_r[i] + vcs_r[i] + tau_r[i] + breath_r[i]) for i in range(n)
    ]
    
    formulas["RF:ALL8"] = [
        -(bur_r[i] + tau_r[i] + vcs_r[i] + chem_r[i] + 
          sig_r[i] + hyst_r[i] + onset_r[i] + breath_r[i]) for i in range(n)
    ]
    
    # NF7: GTCKL baseline for comparison
    formulas["GTCKL"] = [s.get("rank_score", 0) for s in sites]
    
    # NF8: geomean of GTCKL and burial_TCKL
    formulas["geomean"] = [
        math.sqrt(max(
            s.get("rank_score", 0) * (
                s.get("burial_score", 0) * (0.5 + s.get("engine_vcs", 0))
                * s.get("rank_T", 0.5) * s.get("rank_K", 0) * s.get("rank_L", 0.8)
            ), 0))
        for s in sites
    ]
    
    return formulas

# ── Main ──
targets = {}
for name, path in PATHS.items():
    p = Path(path)
    if not p.exists(): continue
    with open(p) as f:
        targets[name] = json.load(f).get("sites", [])

target_names = sorted(targets.keys())
print(f"Targets: {len(targets)}")

# Test all formulas
all_sites_first = list(targets.values())[0]
formula_names = list(compute_neuromorphic_score(all_sites_first).keys())

results = []
for fname in formula_names:
    row = {"formula": fname, "SR@1": 0, "SR@3": 0, "bSR@1": 0, "bSR@3": 0}
    for tname in target_names:
        sites = [dict(s) for s in targets[tname]]
        gt = GT[tname]
        formulas = compute_neuromorphic_score(sites)
        scores = formulas[fname]
        
        order = sorted(range(len(sites)), key=lambda i: scores[i], reverse=True)
        ranked = [sites[i] for i in order]
        
        top1_dcc = dist(ranked[0]["centroid"], gt)
        best = min(sites, key=lambda s: dist(s["centroid"], gt))
        br = next(i+1 for i, s in enumerate(ranked) if s["id"] == best["id"])
        
        row[f"{tname}_rank"] = br
        blind = tname in ["5lar","3l3n","1nna","2npq"]
        if top1_dcc < 5.5:
            row["SR@1"] += 1
            if blind: row["bSR@1"] += 1
        if br <= 3:
            row["SR@3"] += 1
            if blind: row["bSR@3"] += 1
    results.append(row)

results.sort(key=lambda r: (-r["SR@1"], -r["SR@3"], -r["bSR@1"], -r["bSR@3"]))

print(f"\n{'FORMULA':<25} {'SR@1':>4} {'SR@3':>4} {'bR1':>3} {'bR3':>3}", end="")
for t in target_names:
    b = "*" if t in ["5lar","3l3n","1nna","2npq"] else " "
    print(f" {t.upper()+b:>6}", end="")
print()
print("-" * (42 + 7*len(target_names)))

for row in results:
    print(f"{row['formula']:<25} {row['SR@1']:>4} {row['SR@3']:>4} {row['bSR@1']:>3} {row['bSR@3']:>3}", end="")
    for t in target_names:
        r = row[f"{t}_rank"]
        print(f"  R{r:>3}", end="")
    print()

# Save
with open("scripts/ranking_matrix_v2.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=results[0].keys())
    w.writeheader()
    w.writerows(results)
print(f"\nSaved to scripts/ranking_matrix_v2.csv")
