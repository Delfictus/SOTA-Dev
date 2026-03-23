#!/usr/bin/env python3
"""
PRISM4D Neuromorphic Ranker — Temporal Feature Extraction from Existing Data
=============================================================================
Extracts temporal/dynamic features from data ALREADY ON DISK:
  1. all_pockets[].volumes — 99-frame volume time series
  2. prism_therm.sites[] — phase-separated spike counts, avalanches, TIDE
  3. sites[] — existing 36 scalar features

No re-run needed. Evaluates ranking on BENCH30.
"""
import json, os, math
import numpy as np
from scipy import stats as sp_stats
from scipy.signal import find_peaks

os.chdir(os.path.dirname(os.path.abspath(__file__)))

gt = json.load(open("ground_truth/ligand_centroids.json"))
manifest = json.load(open("benchmark_manifest.json"))
EXCLUDE = {14}  # Heme polymer

# ══════════════════════════════════════════════════════════════
# 1. LOAD ALL DATA WITH TEMPORAL FEATURES
# ══════════════════════════════════════════════════════════════

def extract_volume_temporal_features(volumes):
    """Extract temporal features from per-frame volume time series."""
    vols = np.array(volumes, dtype=float)
    n = len(vols)
    if n < 3:
        return {}

    nonzero = vols[vols > 0]
    n_nonzero = len(nonzero)
    persistence = n_nonzero / n  # Fraction of frames pocket exists

    features = {
        "vol_persistence": persistence,
        "vol_n_frames": n,
    }

    if n_nonzero < 3:
        features.update({
            "vol_cv": 0, "vol_trend_slope": 0, "vol_onset_frac": 1.0,
            "vol_max_streak": 0, "vol_n_peaks": 0, "vol_mean_nonzero": 0,
            "vol_decay_rate": 0, "vol_stability": 0, "vol_fano": 0,
            "vol_late_persistence": 0, "vol_early_vs_late": 0,
        })
        return features

    # CV of non-zero volumes
    features["vol_cv"] = float(np.std(nonzero) / (np.mean(nonzero) + 1e-10))

    # Mean non-zero volume (log-scaled)
    features["vol_mean_nonzero"] = math.log1p(np.mean(nonzero))

    # Trend: linear regression slope on non-zero volumes
    nz_indices = np.where(vols > 0)[0]
    if len(nz_indices) >= 3:
        slope, _, r, _, _ = sp_stats.linregress(nz_indices, vols[nz_indices])
        features["vol_trend_slope"] = float(slope)
        features["vol_stability"] = float(r**2)  # R² of linear fit
    else:
        features["vol_trend_slope"] = 0
        features["vol_stability"] = 0

    # Onset: first non-zero frame / total frames
    first_nz = nz_indices[0] if len(nz_indices) > 0 else n
    features["vol_onset_frac"] = float(first_nz / n)

    # Max consecutive non-zero streak
    max_streak = 0
    streak = 0
    for v in vols:
        if v > 0:
            streak += 1
            max_streak = max(max_streak, streak)
        else:
            streak = 0
    features["vol_max_streak"] = max_streak / n

    # Peak detection on volume trace
    if n_nonzero >= 5:
        smoothed = np.convolve(vols, np.ones(3)/3, mode='same')
        peaks, _ = find_peaks(smoothed, distance=3)
        features["vol_n_peaks"] = len(peaks) / n
    else:
        features["vol_n_peaks"] = 0

    # Decay rate: ratio of last-quarter mean to first-quarter mean
    q1 = vols[:n//4]
    q4 = vols[3*n//4:]
    q1_mean = np.mean(q1) if len(q1) > 0 else 0
    q4_mean = np.mean(q4) if len(q4) > 0 else 0
    features["vol_decay_rate"] = float(q4_mean / (q1_mean + 1e-10))

    # Late persistence: fraction of last third that's nonzero
    last_third = vols[2*n//3:]
    features["vol_late_persistence"] = float(np.sum(last_third > 0) / (len(last_third) + 1e-10))

    # Early vs late: is the pocket more active early or late?
    first_half_nz = np.sum(vols[:n//2] > 0)
    second_half_nz = np.sum(vols[n//2:] > 0)
    features["vol_early_vs_late"] = float((first_half_nz - second_half_nz) / (n/2 + 1e-10))

    # Fano factor of binned volumes (windowed spike-count analogue)
    window = max(n // 10, 3)
    binned = [np.sum(vols[i:i+window] > 0) for i in range(0, n - window + 1, window)]
    if len(binned) >= 3:
        binned = np.array(binned, dtype=float)
        features["vol_fano"] = float(np.var(binned) / (np.mean(binned) + 1e-10))
    else:
        features["vol_fano"] = 0

    return features


def extract_therm_features(therm_site):
    """Extract features from prism_therm per-site data."""
    if not therm_site:
        return {}

    heating = therm_site.get("heating_spike_count", 0) or 0
    cooling = therm_site.get("cooling_spike_count", 0) or 0
    total = heating + cooling
    h_rate = therm_site.get("heating_spike_rate", 0) or 0
    c_rate = therm_site.get("cooling_spike_rate", 0) or 0

    features = {
        "therm_heating_frac": heating / max(total, 1),
        "therm_rate_ratio": h_rate / max(c_rate, 1e-6),
        "therm_phase_asymmetry": abs(heating - cooling) / max(total, 1),
        "therm_total_spikes": math.log1p(total),
        "therm_n_avalanches": math.log1p(therm_site.get("n_avalanches", 0) or 0),
        "therm_avalanche_rate": (therm_site.get("n_avalanches", 0) or 0) / max(total, 1),
        "therm_tau": therm_site.get("tau", 0) or 0,
        "therm_tau_optimal": 0,  # filled below
        "therm_is_soc": 1.0 if therm_site.get("ccns_classification") == "Soc" else 0.0,
        "therm_is_hysteretic": 1.0 if therm_site.get("is_hysteretic", False) else 0.0,
    }

    # SOC criticality: tau near 1.3-1.5 is optimal for binding sites
    tau = features["therm_tau"]
    if tau > 0:
        features["therm_tau_optimal"] = max(0, 1.0 - abs(tau - 1.4) / 0.3)

    # TIDE decomposition features
    tide = therm_site.get("tide_decomposition", [])
    if tide:
        te_vals = [r.get("transfer_entropy", 0) for r in tide]
        fi_vals = [r.get("fisher_info", 0) for r in tide]
        kl_vals = [r.get("kl_divergence", 0) for r in tide]
        causal_spikes = [r.get("n_causal_spikes", 0) for r in tide]

        features["tide_n_residues"] = len(tide)
        features["tide_max_te"] = max(te_vals) if te_vals else 0
        features["tide_mean_te"] = float(np.mean(te_vals)) if te_vals else 0
        features["tide_sum_te"] = sum(te_vals)
        features["tide_max_fisher"] = max(fi_vals) if fi_vals else 0
        features["tide_mean_kl"] = float(np.mean(kl_vals)) if kl_vals else 0
        features["tide_causal_spikes_total"] = math.log1p(sum(causal_spikes))
        features["tide_causal_frac"] = sum(causal_spikes) / max(total, 1)
        # Information concentration: does one residue dominate?
        if sum(te_vals) > 0:
            te_norm = np.array(te_vals) / sum(te_vals)
            features["tide_te_entropy"] = float(-np.sum(te_norm * np.log(te_norm + 1e-10)))
        else:
            features["tide_te_entropy"] = 0
    else:
        features.update({
            "tide_n_residues": 0, "tide_max_te": 0, "tide_mean_te": 0,
            "tide_sum_te": 0, "tide_max_fisher": 0, "tide_mean_kl": 0,
            "tide_causal_spikes_total": 0, "tide_causal_frac": 0,
            "tide_te_entropy": 0,
        })

    return features


def match_pocket_to_site(pocket, sites):
    """Match all_pockets entry to sites entry by centroid proximity."""
    pc = np.array(pocket["centroid"])
    best_dist = float('inf')
    best_idx = -1
    for i, s in enumerate(sites):
        sc = np.array(s["centroid"])
        d = float(np.linalg.norm(pc - sc))
        if d < best_dist:
            best_dist = d
            best_idx = i
    return best_idx if best_dist < 15.0 else -1


def match_therm_to_site(therm_sites, site_id):
    """Match prism_therm site to binding site by site_id."""
    for ts in therm_sites:
        if ts.get("site_id") == site_id:
            return ts
    return None


def load_all_targets():
    """Load all targets with full feature extraction."""
    targets = []

    for t in manifest["targets"]:
        tid = t["id"]
        if tid in EXCLUDE:
            continue
        tid_s = str(tid)
        apo = t["apo_pdb"].lower()
        bs_file = f"results/{tid_s}/{apo}.binding_sites.json"
        if not os.path.exists(bs_file) or tid_s not in gt:
            # Try alternate naming
            import glob
            alt = glob.glob(f"results/{tid_s}/*.binding_sites.json")
            if alt:
                bs_file = alt[0]
            else:
                continue

        data = json.load(open(bs_file))
        sites = data.get("sites", [])
        if len(sites) < 2:
            continue

        true_c = np.array(gt[tid_s]["centroid"])

        # Get temporal data
        all_pockets = data.get("all_pockets", [])
        prism_therm = data.get("prism_therm", {})
        therm_sites = prism_therm.get("sites", [])

        # Build pocket-to-site mapping
        pocket_vol_features = {}
        for pocket in all_pockets:
            if not pocket or "centroid" not in pocket or "volumes" not in pocket:
                continue
            matched_idx = match_pocket_to_site(pocket, sites)
            if matched_idx >= 0 and "volumes" in pocket:
                vf = extract_volume_temporal_features(pocket["volumes"])
                if matched_idx not in pocket_vol_features:
                    pocket_vol_features[matched_idx] = vf
                else:
                    # Multiple pockets map to same site — keep highest persistence
                    if vf.get("vol_persistence", 0) > pocket_vol_features[matched_idx].get("vol_persistence", 0):
                        pocket_vol_features[matched_idx] = vf

        site_features = []
        dccs = []
        for i, s in enumerate(sites):
            c = s.get("centroid")
            if not c:
                continue
            dcc = float(np.linalg.norm(np.array(c) - true_c))
            dccs.append(dcc)

            feats = {}

            # === Existing scalar features ===
            for col in ["burial_score", "onset_score", "sphericity", "uv_enrichment_score",
                        "breathing_score", "source_diversity", "wd_coherence", "quality_score",
                        "druggability", "mean_burial", "hysteresis_asymmetry", "asymmetry_offset",
                        "relative_asymmetry", "kinetic_accessibility", "frustrated_solvent_score",
                        "aromatic_score", "ray_escape_ratio", "ccns_tau", "tide_coupling_score",
                        "engine_geo", "engine_vcs", "engine_chem", "engine_phys",
                        "catalytic_residue_count"]:
                v = s.get(col, 0)
                feats[col] = float(v) if v is not None else 0.0

            # Log-transformed counts
            sc = s.get("spike_count", 0) or 0
            vol = s.get("volume", 0) or 0
            feats["log_spike_count"] = math.log10(sc + 1) if sc > 0 else 0
            feats["log_volume"] = math.log10(vol + 1) if vol > 0 else 0
            n_lin = len(s.get("lining_residues", []))
            feats["n_lining_residues"] = n_lin
            feats["log_lining"] = math.log1p(n_lin)
            feats["enclosure"] = n_lin / max(vol**0.667, 1) if vol > 0 else 0

            # Derived interactions
            feats["burial_x_lining"] = feats["burial_score"] * n_lin / 20
            feats["catalytic_density"] = feats["catalytic_residue_count"] / max(n_lin, 1)

            # Classification
            cls = s.get("classification", "")
            feats["is_cryptic_cls"] = 1.0 if cls == "Cryptic" else 0.0

            therm_cls = s.get("therm_class", "")
            feats["is_cryptic_therm"] = 1.0 if therm_cls == "CRYPTIC" else 0.0
            feats["is_dynamic_therm"] = 1.0 if therm_cls == "DYNAMIC" else 0.0

            # === NEW: Volume temporal features ===
            if i in pocket_vol_features:
                feats.update(pocket_vol_features[i])
            else:
                # Defaults for unmatched sites
                feats.update({
                    "vol_persistence": 0, "vol_cv": 0, "vol_trend_slope": 0,
                    "vol_onset_frac": 1.0, "vol_max_streak": 0, "vol_n_peaks": 0,
                    "vol_mean_nonzero": 0, "vol_decay_rate": 0, "vol_stability": 0,
                    "vol_fano": 0, "vol_late_persistence": 0, "vol_early_vs_late": 0,
                    "vol_n_frames": 0,
                })

            # === NEW: Thermodynamic temporal features ===
            site_id = s.get("id")
            therm_match = match_therm_to_site(therm_sites, site_id) if site_id else None
            therm_feats = extract_therm_features(therm_match)
            feats.update(therm_feats)

            site_features.append(feats)

        if not site_features or not dccs:
            continue

        targets.append({
            "tid": tid,
            "apo": t["apo_pdb"],
            "type": t.get("type", t.get("site_type", "orthosteric")),
            "features": site_features,
            "dccs": np.array(dccs),
            "n_sites": len(dccs),
            "best_idx": int(np.argmin(dccs)),
            "best_dcc": min(dccs),
        })

    return targets

print("Loading targets with temporal + thermodynamic features...")
targets = load_all_targets()
ALL_FEATURES = sorted(targets[0]["features"][0].keys())
print(f"Loaded {len(targets)} targets, {sum(t['n_sites'] for t in targets)} sites, {len(ALL_FEATURES)} features")

# ══════════════════════════════════════════════════════════════
# 2. SINGLE-FEATURE RANKING SCAN
# ══════════════════════════════════════════════════════════════

def evaluate_ranker(targets, score_fn, threshold=5.0):
    sr = {1: 0, 3: 0, 5: 0, 10: 0}
    top1_dccs = []
    for t in targets:
        scores = score_fn(t)
        ranking = np.argsort(-scores)
        dccs = t["dccs"]
        top1_dccs.append(dccs[ranking[0]])
        for k in sr:
            best_in_topk = min(dccs[ranking[j]] for j in range(min(k, len(ranking))))
            if best_in_topk <= threshold:
                sr[k] += 1
    n = len(targets)
    return {"SR@1": sr[1]/n, "SR@3": sr[3]/n, "SR@5": sr[5]/n, "SR@10": sr[10]/n,
            "mean_top1": np.mean(top1_dccs), "n": n}

print("\n" + "="*90)
print("SINGLE-FEATURE RANKING — ALL FEATURES (including temporal)")
print("="*90)

single_results = {}
for fn in ALL_FEATURES:
    def sf_pos(t, _fn=fn):
        return np.array([sf.get(_fn, 0) for sf in t["features"]])
    def sf_neg(t, _fn=fn):
        return -np.array([sf.get(_fn, 0) for sf in t["features"]])
    rp = evaluate_ranker(targets, sf_pos)
    rn = evaluate_ranker(targets, sf_neg)
    if rp["SR@1"] >= rn["SR@1"]:
        single_results[fn] = rp; single_results[fn]["dir"] = "+"
    else:
        single_results[fn] = rn; single_results[fn]["dir"] = "-"

sorted_sf = sorted(single_results.items(),
                   key=lambda x: (x[1]["SR@1"], x[1]["SR@3"], -x[1]["mean_top1"]),
                   reverse=True)

print(f"\n{'Feature':>40} {'Dir':>3} {'SR@1':>6} {'SR@3':>6} {'SR@5':>6} {'SR@10':>6} {'Top1':>7}")
print("-"*90)

# Categorize features
temporal_features = [f for f in ALL_FEATURES if f.startswith("vol_")]
therm_features = [f for f in ALL_FEATURES if f.startswith("therm_") or f.startswith("tide_")]
existing_features = [f for f in ALL_FEATURES if f not in temporal_features and f not in therm_features]

# Print with category markers
for fn, res in sorted_sf:
    cat = "T" if fn in temporal_features else ("H" if fn in therm_features else " ")
    print(f"{fn:>40} {cat}{res['dir']:>2} {res['SR@1']:>5.1%} {res['SR@3']:>5.1%} "
          f"{res['SR@5']:>5.1%} {res['SR@10']:>5.1%} {res['mean_top1']:>6.1f}Å")

# ══════════════════════════════════════════════════════════════
# 3. CORRELATION ANALYSIS: temporal features vs DCC
# ══════════════════════════════════════════════════════════════

print("\n" + "="*90)
print("POINT-BISERIAL CORRELATION WITH HIT@8Å — NEW TEMPORAL FEATURES")
print("="*90)

new_feats = temporal_features + therm_features
correlations = {}
for fn in new_feats:
    vals, hits = [], []
    for t in targets:
        for i, sf in enumerate(t["features"]):
            vals.append(sf.get(fn, 0))
            hits.append(1 if t["dccs"][i] <= 8.0 else 0)
    if len(set(vals)) > 1 and sum(hits) > 0:
        r, p = sp_stats.pointbiserialr(hits, vals)
        correlations[fn] = (r, p)

sorted_corr = sorted(correlations.items(), key=lambda x: abs(x[1][0]), reverse=True)
print(f"\n{'Feature':>40} {'r_pb':>8} {'p-value':>10} {'Sig':>5}")
print("-"*70)
for fn, (r, p) in sorted_corr:
    sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
    print(f"{fn:>40} {r:>+8.4f} {p:>10.2e} {sig:>5}")

# ══════════════════════════════════════════════════════════════
# 4. RANKED METHODOLOGY COMPARISON
# ══════════════════════════════════════════════════════════════

print("\n" + "="*90)
print("RANKING METHODOLOGY COMPARISON")
print("="*90)

methods = {}

# M1: Current quality_score
methods["M1: quality_score (v7)"] = evaluate_ranker(targets,
    lambda t: np.array([sf["quality_score"] for sf in t["features"]]))

# M2: Best single existing feature
best_existing = sorted_sf[0]
methods[f"M2: Best single ({best_existing[0]})"] = best_existing[1]

# M3-M5: Best single temporal/therm features
best_temp = [(fn, r) for fn, r in sorted_sf if fn in temporal_features]
best_therm = [(fn, r) for fn, r in sorted_sf if fn in therm_features]
if best_temp:
    methods[f"M3: Best temporal ({best_temp[0][0]})"] = best_temp[0][1]
if best_therm:
    methods[f"M4: Best thermo ({best_therm[0][0]})"] = best_therm[0][1]

# M5: Top-K rank fusion (proven features only)
PROVEN = [fn for fn, r in sorted_sf if r["SR@1"] > 0]
print(f"\nProven features (SR@1 > 0): {len(PROVEN)}")
for fn in PROVEN:
    cat = "[TEMPORAL]" if fn in temporal_features else "[THERMO]" if fn in therm_features else "[SCALAR]"
    print(f"  {fn:>40} {cat}")

def rank_fusion(t, features, weights=None):
    n = t["n_sites"]
    scores = np.zeros(n)
    for j, fn in enumerate(features):
        d = single_results[fn]["dir"]
        vals = np.array([sf.get(fn, 0) for sf in t["features"]])
        if d == "-": vals = -vals
        ranks = sp_stats.rankdata(vals, method='average') / n
        w = weights[j] if weights else 1.0
        scores += w * ranks
    return scores

# Rank fusion: all proven features (equal weight)
methods["M5: Rank Fusion (proven, equal)"] = evaluate_ranker(targets,
    lambda t: rank_fusion(t, PROVEN))

# Rank fusion: SR@1-weighted
sr1_weights = [single_results[fn]["SR@1"] for fn in PROVEN]
methods["M6: Rank Fusion (proven, SR@1-weighted)"] = evaluate_ranker(targets,
    lambda t: rank_fusion(t, PROVEN, sr1_weights))

# M7: Existing-only fusion (no temporal/therm)
PROVEN_EXISTING = [fn for fn in PROVEN if fn in existing_features]
methods["M7: Rank Fusion (existing only)"] = evaluate_ranker(targets,
    lambda t: rank_fusion(t, PROVEN_EXISTING))

# M8: Temporal+therm only fusion
PROVEN_NEW = [fn for fn in PROVEN if fn in temporal_features or fn in therm_features]
if PROVEN_NEW:
    methods["M8: Rank Fusion (temporal+therm only)"] = evaluate_ranker(targets,
        lambda t: rank_fusion(t, PROVEN_NEW))

# M9: Exhaustive best pair from top-12
TOP12 = [fn for fn, _ in sorted_sf[:12]]
best_pair = None
best_pair_sr1 = 0
best_pair_res = None

for i in range(len(TOP12)):
    for j in range(i+1, len(TOP12)):
        f1, f2 = TOP12[i], TOP12[j]
        res = evaluate_ranker(targets, lambda t, _f1=f1, _f2=f2: rank_fusion(t, [_f1, _f2]))
        if res["SR@1"] > best_pair_sr1 or (res["SR@1"] == best_pair_sr1 and res["mean_top1"] < (best_pair_res["mean_top1"] if best_pair_res else 999)):
            best_pair = (f1, f2)
            best_pair_sr1 = res["SR@1"]
            best_pair_res = res

methods[f"M9: Best Pair ({best_pair[0][:20]}+{best_pair[1][:20]})"] = best_pair_res

# M10: Best triple from top-10
TOP10 = [fn for fn, _ in sorted_sf[:10]]
best_triple = None
best_triple_sr1 = 0
best_triple_res = None

for i in range(len(TOP10)):
    for j in range(i+1, len(TOP10)):
        for k in range(j+1, len(TOP10)):
            f1, f2, f3 = TOP10[i], TOP10[j], TOP10[k]
            res = evaluate_ranker(targets, lambda t, _f1=f1, _f2=f2, _f3=f3: rank_fusion(t, [_f1, _f2, _f3]))
            if res["SR@1"] > best_triple_sr1 or (res["SR@1"] == best_triple_sr1 and res["mean_top1"] < (best_triple_res["mean_top1"] if best_triple_res else 999)):
                best_triple = (f1, f2, f3)
                best_triple_sr1 = res["SR@1"]
                best_triple_res = res

methods[f"M10: Best Triple ({best_triple[0][:15]}+{best_triple[1][:15]}+{best_triple[2][:15]})"] = best_triple_res

# M11: Plackett-Luce (trained on all, no CV — just to see ceiling)
def plackett_luce_full(targets, features):
    """Train PL on all targets, evaluate on all (overfitting ceiling)."""
    from scipy.optimize import minimize as sp_minimize
    n_feat = len(features)

    def loss(w):
        nll = 0
        for t in targets:
            n = t["n_sites"]
            scores = np.zeros(n)
            for j, fn in enumerate(features):
                d = single_results[fn]["dir"]
                vals = np.array([sf.get(fn, 0) for sf in t["features"]])
                if d == "-": vals = -vals
                vals = sp_stats.rankdata(vals) / n
                scores += w[j] * vals
            best = t["best_idx"]
            shifted = scores - scores.max()
            log_p = shifted[best] - np.log(np.sum(np.exp(shifted)))
            nll -= log_p
        return nll / len(targets) + 0.5 * np.sum(w**2)

    w0 = np.ones(n_feat) / n_feat
    result = sp_minimize(loss, w0, method='L-BFGS-B', options={'maxiter': 300})
    w_opt = result.x

    def scorer(t):
        n = t["n_sites"]
        scores = np.zeros(n)
        for j, fn in enumerate(features):
            d = single_results[fn]["dir"]
            vals = np.array([sf.get(fn, 0) for sf in t["features"]])
            if d == "-": vals = -vals
            vals = sp_stats.rankdata(vals) / n
            scores += w_opt[j] * vals
        return scores

    res = evaluate_ranker(targets, scorer)
    return res, w_opt

if len(PROVEN) >= 3:
    pl_res, pl_weights = plackett_luce_full(targets, PROVEN)
    methods["M11: Plackett-Luce (full, ceiling)"] = pl_res

    # Print learned weights
    print(f"\nPlackett-Luce weights (trained on proven features):")
    for fn, w in sorted(zip(PROVEN, pl_weights), key=lambda x: abs(x[1]), reverse=True):
        cat = "T" if fn in temporal_features else "H" if fn in therm_features else "S"
        print(f"  [{cat}] {fn:>40}: {w:>+8.4f}")

# ══════════════════════════════════════════════════════════════
# 5. RESULTS TABLE
# ══════════════════════════════════════════════════════════════

print("\n" + "="*100)
print("FINAL RANKING COMPARISON")
print("="*100)
print(f"{'Method':>65} {'SR@1':>6} {'SR@3':>6} {'SR@5':>6} {'SR@10':>6} {'Top1':>7}")
print("-"*100)

sorted_methods = sorted(methods.items(),
    key=lambda x: (x[1]["SR@1"], x[1]["SR@3"], -x[1]["mean_top1"]), reverse=True)

for name, res in sorted_methods:
    print(f"{name:>65} {res['SR@1']:>5.1%} {res['SR@3']:>5.1%} "
          f"{res['SR@5']:>5.1%} {res['SR@10']:>5.1%} {res['mean_top1']:>6.1f}Å")

# ══════════════════════════════════════════════════════════════
# 6. PER-TARGET BREAKDOWN FOR BEST METHOD
# ══════════════════════════════════════════════════════════════

winner_name, winner_res = sorted_methods[0]
print(f"\n{'='*90}")
print(f"WINNER: {winner_name}")
print(f"  SR@1={winner_res['SR@1']:.1%}  SR@3={winner_res['SR@3']:.1%}  Mean Top-1={winner_res['mean_top1']:.1f}Å")
print(f"{'='*90}")

print(f"\n{'TID':>4} {'APO':>5} {'Type':>12} {'N':>3} {'Top1':>7} {'Best':>7} {'BestRk':>7} {'Hit':>4}")
print("-"*65)

# Determine scoring function for winner
if "Best Pair" in winner_name:
    winner_scorer = lambda t: rank_fusion(t, list(best_pair))
elif "Best Triple" in winner_name:
    winner_scorer = lambda t: rank_fusion(t, list(best_triple))
elif "proven, equal" in winner_name:
    winner_scorer = lambda t: rank_fusion(t, PROVEN)
elif "SR@1-weighted" in winner_name:
    winner_scorer = lambda t: rank_fusion(t, PROVEN, sr1_weights)
elif "existing only" in winner_name:
    winner_scorer = lambda t: rank_fusion(t, PROVEN_EXISTING)
elif "temporal+therm" in winner_name:
    winner_scorer = lambda t: rank_fusion(t, PROVEN_NEW)
elif "Plackett" in winner_name:
    def winner_scorer(t):
        n = t["n_sites"]
        scores = np.zeros(n)
        for j, fn in enumerate(PROVEN):
            d = single_results[fn]["dir"]
            vals = np.array([sf.get(fn, 0) for sf in t["features"]])
            if d == "-": vals = -vals
            vals = sp_stats.rankdata(vals) / n
            scores += pl_weights[j] * vals
        return scores
elif "quality_score" in winner_name:
    winner_scorer = lambda t: np.array([sf["quality_score"] for sf in t["features"]])
else:
    fn = sorted_sf[0][0]
    d = single_results[fn]["dir"]
    winner_scorer = lambda t: np.array([sf.get(fn, 0) * (1 if d == "+" else -1) for sf in t["features"]])

# Compare winner vs quality_score
print(f"\n{'':>4} {'':>5} {'':>12} {'':>3} {'WINNER':>7} {'QS_v7':>7} {'Δ':>7}")
print("-"*55)
for t in targets:
    w_scores = winner_scorer(t)
    w_rank = np.argsort(-w_scores)
    w_top1 = t["dccs"][w_rank[0]]
    w_best_rank = int(np.where(w_rank == t["best_idx"])[0][0]) + 1

    qs_scores = np.array([sf["quality_score"] for sf in t["features"]])
    qs_rank = np.argsort(-qs_scores)
    qs_top1 = t["dccs"][qs_rank[0]]

    delta = qs_top1 - w_top1  # positive = winner is better
    marker = ">>>" if delta > 3 else ">>" if delta > 1 else ">" if delta > 0 else "=" if abs(delta) < 0.1 else "<"

    hit = "✓" if w_top1 <= 5.0 else "~" if w_top1 <= 8.0 else "·"
    print(f"{t['tid']:>4} {t['apo']:>5} {t['type']:>12} {t['n_sites']:>3} "
          f"{w_top1:>6.1f}Å {qs_top1:>6.1f}Å {delta:>+6.1f} {marker} {hit}")

# Summary stats
w_wins = sum(1 for t in targets
    if t["dccs"][np.argsort(-winner_scorer(t))[0]] < t["dccs"][np.argsort(-np.array([sf["quality_score"] for sf in t["features"]]))[0]])
qs_wins = sum(1 for t in targets
    if t["dccs"][np.argsort(-winner_scorer(t))[0]] > t["dccs"][np.argsort(-np.array([sf["quality_score"] for sf in t["features"]]))[0]])
ties = len(targets) - w_wins - qs_wins

print(f"\nWinner beats quality_score: {w_wins}/{len(targets)}")
print(f"quality_score beats winner: {qs_wins}/{len(targets)}")
print(f"Ties: {ties}/{len(targets)}")

# Save results
output = {
    "methods": {n: {k: float(v) if isinstance(v, (float, np.floating)) else v
                    for k, v in r.items()} for n, r in sorted_methods},
    "winner": winner_name,
    "proven_features": PROVEN,
    "best_pair": list(best_pair) if best_pair else None,
    "best_triple": list(best_triple) if best_triple else None,
    "single_feature_ranking": {fn: {"SR@1": r["SR@1"], "SR@3": r["SR@3"], "dir": r["dir"]}
                                for fn, r in sorted_sf[:25]},
    "temporal_correlations": {fn: {"r": float(r), "p": float(p)}
                              for fn, (r, p) in sorted_corr[:15]},
}
json.dump(output, open("neuromorphic_ranker_results.json", "w"), indent=2,
          default=lambda x: float(x) if isinstance(x, (float, np.floating)) else x)
print(f"\nResults saved to neuromorphic_ranker_results.json")
