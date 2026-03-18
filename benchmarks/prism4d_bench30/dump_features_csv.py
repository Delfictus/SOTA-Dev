#!/usr/bin/env python3
"""
Dump all per-site features from benchmark JSON results into a single CSV.
Then run discrimination analysis: which features separate hits from misses?
"""
import json, os, sys, csv
import numpy as np

os.chdir(os.path.dirname(os.path.abspath(__file__)))

gt = json.load(open("ground_truth/ligand_centroids.json"))
manifest = json.load(open("benchmark_manifest.json"))

# ── STEP 1: Dump CSV ──
FIELDS = [
    "target_id", "apo_pdb", "site_id", "rank", "dcc",
    "quality_score", "volume", "spike_count",
    "engine_geo", "engine_chem", "engine_phys", "engine_vcs",
    "burial_score", "mean_burial", "onset_score", "source_diversity",
    "aromatic_score", "catalytic_residue_count",
    "druggability", "is_druggable",
    "sphericity", "wd_coherence", "breathing_score",
    "frustrated_solvent_score", "uv_enrichment_score",
    "ccns_tau", "hysteresis_asymmetry", "relative_asymmetry",
    "tide_coupling_score", "therm_class",
    "delta_g_sti_kcal_mol", "effective_delta_g_kcal_mol",
    "n_lining_residues",
    "asymmetry_offset", "ray_escape_ratio",
    # Derived
    "geo_intensive", "chem_intensive", "phys_intensive",
    "log_spike_count", "spike_density",
    "hit_5A", "hit_8A",
]

rows = []
for t in manifest["targets"]:
    tid = str(t["id"])
    outdir = f"results/{tid}"
    bs_files = [f for f in os.listdir(outdir) if f.endswith('.binding_sites.json')] if os.path.exists(outdir) else []
    if not bs_files or tid not in gt:
        continue

    with open(os.path.join(outdir, bs_files[0])) as f:
        data = json.load(f)
    sites = data.get("sites", [])
    lig_c = np.array(gt[tid]["centroid"])

    for rank_idx, s in enumerate(sites):
        sc = np.array(s["centroid"])
        dcc = float(np.linalg.norm(sc - lig_c))
        vol = s.get("volume", 0.0) or 1.0
        spk = s.get("spike_count", 0) or 1

        row = {
            "target_id": tid,
            "apo_pdb": t["apo_pdb"],
            "site_id": s.get("id", -1),
            "rank": rank_idx + 1,
            "dcc": round(dcc, 2),
            "quality_score": s.get("quality_score", 0),
            "volume": vol,
            "spike_count": spk,
            "engine_geo": s.get("engine_geo", 0),
            "engine_chem": s.get("engine_chem", 0),
            "engine_phys": s.get("engine_phys", 0),
            "engine_vcs": s.get("engine_vcs", 0),
            "burial_score": s.get("burial_score", 0),
            "mean_burial": s.get("mean_burial", 0),
            "onset_score": s.get("onset_score", 0),
            "source_diversity": s.get("source_diversity", 0),
            "aromatic_score": s.get("aromatic_score", 0),
            "catalytic_residue_count": s.get("catalytic_residue_count", 0),
            "druggability": s.get("druggability", 0),
            "is_druggable": 1 if s.get("is_druggable", False) else 0,
            "sphericity": s.get("sphericity", 0),
            "wd_coherence": s.get("wd_coherence", 0),
            "breathing_score": s.get("breathing_score", 0),
            "frustrated_solvent_score": s.get("frustrated_solvent_score", 0),
            "uv_enrichment_score": s.get("uv_enrichment_score", 0),
            "ccns_tau": s.get("ccns_tau", 0),
            "hysteresis_asymmetry": s.get("hysteresis_asymmetry", 0),
            "relative_asymmetry": s.get("relative_asymmetry", 0),
            "tide_coupling_score": s.get("tide_coupling_score", 0),
            "therm_class": s.get("therm_class", ""),
            "delta_g_sti_kcal_mol": s.get("delta_g_sti_kcal_mol", 0),
            "effective_delta_g_kcal_mol": s.get("effective_delta_g_kcal_mol", 0),
            "n_lining_residues": len(s.get("lining_residues", [])),
            "asymmetry_offset": s.get("asymmetry_offset", 0),
            "ray_escape_ratio": s.get("ray_escape_ratio", 0),
            # Derived: intensive = raw / ln(vol + 10)
            "geo_intensive": s.get("engine_geo", 0) / np.log(vol + 10) if vol > 0 else 0,
            "chem_intensive": s.get("engine_chem", 0) / np.log(vol + 10) if vol > 0 else 0,
            "phys_intensive": s.get("engine_phys", 0) / np.log(vol + 10) if vol > 0 else 0,
            "log_spike_count": np.log1p(spk),
            "spike_density": spk / np.log(vol + 10) if vol > 0 else 0,
            "hit_5A": 1 if dcc <= 5.0 else 0,
            "hit_8A": 1 if dcc <= 8.0 else 0,
        }
        rows.append(row)

csv_path = "site_features_all.csv"
with open(csv_path, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=FIELDS)
    writer.writeheader()
    writer.writerows(rows)

n_sites = len(rows)
n_targets = len(set(r["target_id"] for r in rows))
n_hits_5 = sum(r["hit_5A"] for r in rows)
n_hits_8 = sum(r["hit_8A"] for r in rows)
print(f"Dumped {n_sites} sites across {n_targets} targets → {csv_path}")
print(f"  Hits: {n_hits_5} sites ≤5Å, {n_hits_8} sites ≤8Å\n")

# ── STEP 2: Discrimination analysis ──
print("=" * 80)
print("FEATURE DISCRIMINATION ANALYSIS")
print("=" * 80)

NUMERIC_FEATURES = [
    "quality_score", "volume", "spike_count", "log_spike_count",
    "engine_geo", "engine_chem", "engine_phys", "engine_vcs",
    "geo_intensive", "chem_intensive", "phys_intensive",
    "burial_score", "mean_burial", "onset_score", "source_diversity",
    "aromatic_score", "catalytic_residue_count",
    "druggability", "n_lining_residues",
    "frustrated_solvent_score",
    "ccns_tau", "hysteresis_asymmetry", "relative_asymmetry",
    "tide_coupling_score",
    "asymmetry_offset", "ray_escape_ratio",
    "spike_density",
]

def auc_roc(scores, labels):
    """Manual AUC-ROC (no sklearn dependency)."""
    pairs = sorted(zip(scores, labels), key=lambda x: -x[0])
    n_pos = sum(labels)
    n_neg = len(labels) - n_pos
    if n_pos == 0 or n_neg == 0:
        return 0.5
    tp = 0
    fp = 0
    auc = 0.0
    prev_fp = 0
    prev_tp = 0
    for score, label in pairs:
        if label:
            tp += 1
        else:
            fp += 1
            auc += (tp + prev_tp) / 2.0
    return auc / (n_pos * n_neg) if (n_pos * n_neg) > 0 else 0.5

def rank_biserial(scores, labels):
    """Rank-biserial correlation: how well does the feature rank hits above misses."""
    hits = [s for s, l in zip(scores, labels) if l]
    misses = [s for s, l in zip(scores, labels) if not l]
    if not hits or not misses:
        return 0.0
    n_concordant = sum(1 for h in hits for m in misses if h > m)
    n_discordant = sum(1 for h in hits for m in misses if h < m)
    n_pairs = len(hits) * len(misses)
    return (n_concordant - n_discordant) / n_pairs if n_pairs > 0 else 0.0

# Global analysis (all sites pooled)
for threshold_label, hit_key in [("≤5Å", "hit_5A"), ("≤8Å", "hit_8A")]:
    print(f"\n{'─' * 80}")
    print(f"  GLOBAL: hit = DCC {threshold_label}")
    print(f"{'─' * 80}")
    labels = [r[hit_key] for r in rows]
    print(f"  {sum(labels)} hits / {len(labels) - sum(labels)} misses\n")
    print(f"  {'Feature':<28s} {'Hit mean':>10s} {'Miss mean':>10s} {'Δ':>8s} {'AUC':>6s} {'r_rb':>6s}")
    print(f"  {'─'*28} {'─'*10} {'─'*10} {'─'*8} {'─'*6} {'─'*6}")

    results = []
    for feat in NUMERIC_FEATURES:
        vals = [float(r[feat]) for r in rows]
        hit_vals = [v for v, l in zip(vals, labels) if l]
        miss_vals = [v for v, l in zip(vals, labels) if not l]
        if not hit_vals or not miss_vals:
            continue
        hit_mean = np.mean(hit_vals)
        miss_mean = np.mean(miss_vals)
        delta = hit_mean - miss_mean
        auc = auc_roc(vals, labels)
        rrb = rank_biserial(vals, labels)
        results.append((feat, hit_mean, miss_mean, delta, auc, rrb))

    # Sort by AUC descending
    results.sort(key=lambda x: -abs(x[4] - 0.5))
    for feat, hm, mm, d, auc, rrb in results:
        flag = " <<<" if abs(auc - 0.5) >= 0.10 else ""
        print(f"  {feat:<28s} {hm:>10.4f} {mm:>10.4f} {d:>+8.4f} {auc:>6.3f} {rrb:>+6.3f}{flag}")

# Per-target analysis: does the feature rank the best pocket above median?
print(f"\n{'=' * 80}")
print("PER-TARGET: Does feature rank correct pocket above median?")
print(f"{'=' * 80}")
print(f"  (Count of targets where best-DCC site has feature value > median of all sites)\n")

target_ids = sorted(set(r["target_id"] for r in rows), key=int)
print(f"  {'Feature':<28s} {'Above median':>14s} {'Below':>8s} {'Pct':>6s}")
print(f"  {'─'*28} {'─'*14} {'─'*8} {'─'*6}")

for feat in NUMERIC_FEATURES:
    above = 0
    below = 0
    for tid in target_ids:
        t_rows = [r for r in rows if r["target_id"] == tid]
        if len(t_rows) < 3:
            continue
        # Find best DCC site
        best_row = min(t_rows, key=lambda r: r["dcc"])
        if best_row["dcc"] > 8.0:
            continue  # No real hit for this target
        vals = [float(r[feat]) for r in t_rows]
        median_val = np.median(vals)
        best_val = float(best_row[feat])
        if best_val > median_val:
            above += 1
        else:
            below += 1
    total = above + below
    if total == 0:
        continue
    pct = above / total * 100
    flag = " <<<" if pct >= 65 else " !!!" if pct <= 35 else ""
    print(f"  {feat:<28s} {above:>6d}/{total:<6d} {below:>8d} {pct:>5.0f}%{flag}")

# Per-target detail: show rank of best-DCC pocket for each feature
print(f"\n{'=' * 80}")
print("PER-TARGET DETAIL: Rank of correct pocket by each top feature")
print(f"{'=' * 80}")

# Pick top-6 features by global AUC for detail view
labels_8 = [r["hit_8A"] for r in rows]
feat_aucs = []
for feat in NUMERIC_FEATURES:
    vals = [float(r[feat]) for r in rows]
    auc = auc_roc(vals, labels_8)
    feat_aucs.append((feat, abs(auc - 0.5)))
feat_aucs.sort(key=lambda x: -x[1])
top_feats = [f for f, _ in feat_aucs[:6]]

header = f"  {'TID':>4s} {'APO':>5s} {'BestDCC':>8s}"
for f in top_feats:
    header += f" {f[:10]:>10s}"
print(header)
print("  " + "─" * (len(header) - 2))

for tid in target_ids:
    t_rows = [r for r in rows if r["target_id"] == tid]
    if len(t_rows) < 3:
        continue
    best_row = min(t_rows, key=lambda r: r["dcc"])
    apo = t_rows[0]["apo_pdb"]
    line = f"  {tid:>4s} {apo:>5s} {best_row['dcc']:>7.1f}A"
    for feat in top_feats:
        vals = sorted([float(r[feat]) for r in t_rows], reverse=True)
        best_val = float(best_row[feat])
        # Rank (1-indexed, higher is better)
        rank = vals.index(best_val) + 1 if best_val in vals else len(vals)
        n = len(t_rows)
        marker = "*" if rank <= 3 else " "
        line += f" {rank:>4d}/{n:<4d}{marker}"
    print(line)

print(f"\nDone. Full data: {csv_path}")
