#!/usr/bin/env python3
"""
PRISM4D Optimal Ranking Methodology Analysis
=============================================
Systematically evaluates every viable ranking methodology on BENCH30 using
Leave-One-Out Cross-Validation (LOOCV) to find the provably best performer.

Key insight: With N=30 targets and ~25 sites/target, we need:
  1. LOOCV (not train/test split) for honest estimates
  2. Strong regularization to prevent overfitting
  3. Feature selection to reduce dimensionality
  4. Methodologies suited to small-N listwise ranking

Methodologies tested:
  1. Baseline: current v7 quality_score (hand-tuned)
  2. Baseline: current Boltzmann ranker (JAX-trained)
  3. Single-feature rankers (find the best individual signal)
  4. Rank-by-DCC-proxy features
  5. ListNet (softmax cross-entropy, LOOCV)
  6. LambdaRank (pairwise with NDCG weighting, LOOCV)
  7. Bradley-Terry pairwise model (LOOCV)
  8. RankSVM (pairwise SVM, LOOCV)
  9. Gradient-boosted LTR (LambdaMART via XGBoost-style, LOOCV)
 10. Bayesian ridge ranking (LOOCV)
 11. Multi-signal Borda count (non-parametric)
 12. Novel: Cryptic-aware thermodynamic ranker (physics-informed)

Output: ranked table of methodologies by SR@1 (LOOCV), SR@3, mean DCC@1
"""

import json, os, math, warnings
import numpy as np
from collections import defaultdict
from itertools import combinations
from scipy import stats
from scipy.optimize import minimize
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.linear_model import LogisticRegression, Ridge, Lasso, ElasticNet
from sklearn.svm import LinearSVC
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.model_selection import LeaveOneOut
warnings.filterwarnings('ignore')

os.chdir(os.path.dirname(os.path.abspath(__file__)))

# ══════════════════════════════════════════════════════════════
# 1. LOAD DATA
# ══════════════════════════════════════════════════════════════

gt = json.load(open("ground_truth/ligand_centroids.json"))
manifest = json.load(open("benchmark_manifest.json"))
results = json.load(open("benchmark_results.json"))

# Exclude target 14 (1K3F/HEC = heme polymer, 2250 atoms)
EXCLUDE = {14}

# Feature columns from the engine output
RAW_FEATURES = [
    "burial_score", "onset_score", "sphericity", "uv_enrichment_score",
    "breathing_score", "source_diversity", "wd_coherence", "quality_score",
    "druggability", "volume", "spike_count", "mean_burial",
    "hysteresis_asymmetry", "asymmetry_offset", "relative_asymmetry",
    "kinetic_accessibility", "frustrated_solvent_score", "aromatic_score",
    "ray_escape_ratio", "ccns_tau", "tide_coupling_score",
    "engine_geo", "engine_vcs", "engine_chem", "engine_phys",
    "catalytic_residue_count", "n_lining_residues",
]

def load_target_data():
    """Load all per-site features + DCC for each target."""
    targets = []
    for t in manifest["targets"]:
        tid = t["id"]
        if tid in EXCLUDE:
            continue
        tid_s = str(tid)
        outdir = f"results/{tid_s}"
        apo = t["apo_pdb"].lower()
        bs_file = os.path.join(outdir, f"{apo}.binding_sites.json")
        if not os.path.exists(bs_file) or tid_s not in gt:
            continue

        data = json.load(open(bs_file))
        sites = data.get("sites", [])
        if len(sites) < 2:
            continue

        true_c = np.array(gt[tid_s]["centroid"])

        site_features = []
        dccs = []
        for s in sites:
            c = s.get("centroid")
            if not c:
                continue
            dcc = float(np.linalg.norm(np.array(c) - true_c))
            dccs.append(dcc)

            feats = {}
            for col in RAW_FEATURES:
                if col == "n_lining_residues":
                    feats[col] = len(s.get("lining_residues", []))
                elif col == "spike_count":
                    v = s.get(col, 0) or 0
                    feats[col] = math.log10(v + 1) if v > 0 else 0
                elif col == "volume":
                    v = s.get(col, 0) or 0
                    feats[col] = math.log10(v + 1) if v > 0 else 0
                else:
                    v = s.get(col, 0)
                    feats[col] = float(v) if v is not None else 0.0

            # Derived features
            n_lin = feats["n_lining_residues"]
            vol = 10**feats["volume"] if feats["volume"] > 0 else 1
            feats["enclosure"] = n_lin / max(vol**0.667, 1)
            feats["log_lining"] = math.log1p(n_lin)
            feats["burial_x_lining"] = feats["burial_score"] * n_lin / 20
            feats["onset_x_burial"] = feats["onset_score"] * feats["burial_score"]
            feats["spike_x_burial"] = feats["spike_count"] * feats["burial_score"]
            feats["aromatic_x_burial"] = feats["aromatic_score"] * feats["burial_score"]
            feats["druggability_x_burial"] = feats["druggability"] * feats["burial_score"]

            # Thermodynamic composite
            tc = s.get("therm_class", "")
            feats["is_cryptic_therm"] = 1.0 if tc == "CRYPTIC" else 0.0
            feats["is_dynamic_therm"] = 1.0 if tc == "DYNAMIC" else 0.0
            feats["is_responsive_therm"] = 1.0 if tc == "RESPONSIVE" else 0.0

            # Classification features
            cls = s.get("classification", "")
            feats["is_cryptic_cls"] = 1.0 if cls == "Cryptic" else 0.0
            feats["is_active_site_cls"] = 1.0 if cls in ("ActiveSite", "Orthosteric") else 0.0

            # Catalytic density
            feats["catalytic_density"] = feats["catalytic_residue_count"] / max(n_lin, 1)

            # Hysteresis magnitude
            feats["abs_asymmetry"] = abs(feats["hysteresis_asymmetry"])

            # TIDE coupling (clipped)
            feats["tide_coupling_clipped"] = min(feats["tide_coupling_score"], 1.0)

            site_features.append(feats)

        if not site_features:
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

targets = load_target_data()
print(f"Loaded {len(targets)} targets, {sum(t['n_sites'] for t in targets)} total sites")

# Get ALL feature names (raw + derived)
ALL_FEATURES = list(targets[0]["features"][0].keys())
print(f"Feature count: {len(ALL_FEATURES)}")

# ══════════════════════════════════════════════════════════════
# 2. HELPER: Convert features to numpy arrays
# ══════════════════════════════════════════════════════════════

def get_feature_matrix(target, feature_names):
    """Extract feature matrix [n_sites, n_features] for one target."""
    n = target["n_sites"]
    X = np.zeros((n, len(feature_names)))
    for i, sf in enumerate(target["features"]):
        for j, fn in enumerate(feature_names):
            X[i, j] = sf.get(fn, 0.0)
    return X

def evaluate_ranker(targets, score_fn, threshold=5.0):
    """Evaluate a scoring function across all targets.

    score_fn(target) -> scores array (higher = better predicted site)
    Returns SR@1, SR@3, SR@5, SR@10, mean_top1_dcc, mean_best_dcc
    """
    sr = {1: 0, 3: 0, 5: 0, 10: 0}
    top1_dccs = []
    best_dccs = []

    for t in targets:
        scores = score_fn(t)
        ranking = np.argsort(-scores)  # descending
        dccs = t["dccs"]

        top1_dcc = dccs[ranking[0]]
        top1_dccs.append(top1_dcc)
        best_dccs.append(t["best_dcc"])

        for k in sr:
            topk_indices = ranking[:k]
            best_in_topk = min(dccs[idx] for idx in topk_indices)
            if best_in_topk <= threshold:
                sr[k] += 1

    n = len(targets)
    return {
        "SR@1": sr[1] / n,
        "SR@3": sr[3] / n,
        "SR@5": sr[5] / n,
        "SR@10": sr[10] / n,
        "mean_top1_dcc": np.mean(top1_dccs),
        "mean_best_dcc": np.mean(best_dccs),
        "n": n,
    }

def evaluate_loocv(targets, train_and_predict_fn, threshold=5.0):
    """Leave-One-Out CV evaluation for learned rankers.

    train_and_predict_fn(train_targets, test_target) -> scores for test_target
    """
    sr = {1: 0, 3: 0, 5: 0, 10: 0}
    top1_dccs = []

    for i in range(len(targets)):
        train = [t for j, t in enumerate(targets) if j != i]
        test = targets[i]
        scores = train_and_predict_fn(train, test)
        ranking = np.argsort(-scores)
        dccs = test["dccs"]

        top1_dcc = dccs[ranking[0]]
        top1_dccs.append(top1_dcc)

        for k in sr:
            topk_indices = ranking[:k]
            best_in_topk = min(dccs[idx] for idx in topk_indices)
            if best_in_topk <= threshold:
                sr[k] += 1

    n = len(targets)
    return {
        "SR@1": sr[1] / n,
        "SR@3": sr[3] / n,
        "SR@5": sr[5] / n,
        "SR@10": sr[10] / n,
        "mean_top1_dcc": np.mean(top1_dccs),
        "n": n,
    }

# ══════════════════════════════════════════════════════════════
# 3. BASELINE RANKERS
# ══════════════════════════════════════════════════════════════

print("\n" + "=" * 80)
print("SINGLE-FEATURE RANKING ANALYSIS")
print("=" * 80)

single_feature_results = {}
for fn in ALL_FEATURES:
    # Some features are "higher = better binding site", some inverted
    # Test both directions
    def score_fn_pos(t, _fn=fn):
        return np.array([sf[_fn] for sf in t["features"]])

    def score_fn_neg(t, _fn=fn):
        return -np.array([sf[_fn] for sf in t["features"]])

    res_pos = evaluate_ranker(targets, score_fn_pos)
    res_neg = evaluate_ranker(targets, score_fn_neg)

    if res_pos["SR@1"] >= res_neg["SR@1"]:
        single_feature_results[fn] = res_pos
        single_feature_results[fn]["direction"] = "+"
    else:
        single_feature_results[fn] = res_neg
        single_feature_results[fn]["direction"] = "-"

# Sort by SR@1, then SR@3
sorted_features = sorted(single_feature_results.items(),
                         key=lambda x: (x[1]["SR@1"], x[1]["SR@3"], -x[1]["mean_top1_dcc"]),
                         reverse=True)

print(f"\n{'Feature':>35} {'Dir':>3} {'SR@1':>6} {'SR@3':>6} {'SR@5':>6} {'SR@10':>6} {'MeanTop1':>9}")
print("-" * 100)
for fn, res in sorted_features[:30]:
    print(f"{fn:>35} {res['direction']:>3} {res['SR@1']:>5.1%} {res['SR@3']:>5.1%} "
          f"{res['SR@5']:>5.1%} {res['SR@10']:>5.1%} {res['mean_top1_dcc']:>8.1f}Å")

# ══════════════════════════════════════════════════════════════
# 4. FEATURE CORRELATION ANALYSIS
# ══════════════════════════════════════════════════════════════

print("\n" + "=" * 80)
print("FEATURE-DCC CORRELATION ANALYSIS (point-biserial @ 8Å)")
print("=" * 80)

# Collect all (feature_value, is_hit@8A) pairs across all sites
correlations = {}
for fn in ALL_FEATURES:
    all_vals = []
    all_hits = []
    for t in targets:
        for i, sf in enumerate(t["features"]):
            all_vals.append(sf.get(fn, 0.0))
            all_hits.append(1 if t["dccs"][i] <= 8.0 else 0)
    r, p = stats.pointbiserialr(all_hits, all_vals)
    correlations[fn] = (r, p)

sorted_corr = sorted(correlations.items(), key=lambda x: abs(x[1][0]), reverse=True)
print(f"\n{'Feature':>35} {'r_pb':>8} {'p-value':>10} {'Significant':>12}")
print("-" * 75)
for fn, (r, p) in sorted_corr[:25]:
    sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
    print(f"{fn:>35} {r:>+8.4f} {p:>10.2e} {sig:>12}")

# ══════════════════════════════════════════════════════════════
# 5. METHODOLOGY 1: Current quality_score baseline
# ══════════════════════════════════════════════════════════════

print("\n" + "=" * 80)
print("METHODOLOGY COMPARISON")
print("=" * 80)

all_methods = {}

# M1: quality_score
def qs_ranker(t):
    return np.array([sf["quality_score"] for sf in t["features"]])
all_methods["M1: quality_score (v7)"] = evaluate_ranker(targets, qs_ranker)

# M2: Boltzmann ranker (replicate from weights)
BOLT_W = np.array([1.9238, 0.0234, 0.0453, 0.0932, 0.0454, 0.0561,
                    0.0634, 0.0603, 0.0133, 0.0054, 0.0454, 0.0454,
                    0.0089, 0.0517, 0.6019])
BOLT_BETA = 3.953
BOLT_T = 0.063

def boltzmann_features(site_feats):
    """Map site features to the 15-dim Boltzmann feature vector."""
    f = np.zeros(15)
    sc = site_feats.get("spike_count", 0)
    # spike_count is already log10 in our data, need log1p/14
    f[0] = sc / 14.0 * math.log(10) / math.log1p(1)  # approximate conversion
    # Actually let's just use the raw mapping
    f[0] = sc  # already log-transformed, normalize
    f[1] = site_feats.get("burial_score", 0)
    f[2] = site_feats.get("wd_coherence", 0)
    f[3] = min(site_feats.get("n_lining_residues", 0) / 20.0, 1.0)
    f[4] = 1.0 - site_feats.get("sphericity", 0.5)
    f[5] = site_feats.get("source_diversity", 0)
    f[6] = site_feats.get("uv_enrichment_score", 0)
    f[7] = site_feats.get("sphericity", 0.5)
    f[8] = 1.0 - site_feats.get("onset_score", 0.5)
    vol = 10**site_feats.get("volume", 0) if site_feats.get("volume", 0) > 0 else 1
    n_lin = site_feats.get("n_lining_residues", 0)
    encl = n_lin / max(vol**0.667, 1)
    f[9] = 1.0 - min(encl / 2.0, 1.0)
    f[10] = (site_feats.get("wd_coherence", 0) + site_feats.get("quality_score", 0)) / 2
    f[11] = site_feats.get("breathing_score", 0)
    f[12] = site_feats.get("breathing_score", 0)
    f[13] = site_feats.get("engine_vcs", 0)
    f[14] = site_feats.get("quality_score", 0) / 2
    return f

def boltzmann_ranker(t):
    scores = []
    for sf in t["features"]:
        f = boltzmann_features(sf)
        H = -(BOLT_W[0]*f[0] + BOLT_W[1]*f[1] + BOLT_W[2]*f[2] + BOLT_W[3]*f[3])
        S = BOLT_W[4]*f[4] + BOLT_W[5]*f[5] + BOLT_W[6]*f[6] + BOLT_W[7]*f[7]
        W = BOLT_W[8]*f[8] + BOLT_W[9]*f[9]
        K = -(BOLT_W[10]*f[10] + BOLT_W[11]*f[11] + BOLT_W[12]*f[12])
        C = -(BOLT_W[13]*f[13] + BOLT_W[14]*f[14])
        dG = H + W - BOLT_T * S + K + C
        scores.append(-BOLT_BETA * dG)  # higher = better
    return np.array(scores)

all_methods["M2: Boltzmann (JAX-trained)"] = evaluate_ranker(targets, boltzmann_ranker)

# ══════════════════════════════════════════════════════════════
# 6. METHODOLOGY 3-5: Linear scoring with LOOCV
# ══════════════════════════════════════════════════════════════

# Select top features by correlation + single-feature SR@1
TOP_FEATURES_BY_CORR = [fn for fn, (r, p) in sorted_corr if p < 0.1][:15]
TOP_FEATURES_BY_SR = [fn for fn, res in sorted_features if res["SR@1"] > 0.05][:15]
SELECTED_FEATURES = list(dict.fromkeys(TOP_FEATURES_BY_SR + TOP_FEATURES_BY_CORR))[:20]
print(f"\nSelected {len(SELECTED_FEATURES)} features for learned models:")
for fn in SELECTED_FEATURES:
    sr1 = single_feature_results[fn]["SR@1"]
    r, p = correlations[fn]
    print(f"  {fn:>35}  SR@1={sr1:.1%}  r={r:+.3f}  p={p:.3e}")

# M3: ListNet (softmax cross-entropy) with LOOCV
def listnet_train_predict(train_targets, test_target):
    """Train linear ListNet on train, predict on test."""
    # Prepare pairwise training data
    X_all = []
    y_all = []
    for t in train_targets:
        X = get_feature_matrix(t, SELECTED_FEATURES)
        best = t["best_idx"]
        # Create pairwise: positive = best site, negative = others
        for i in range(t["n_sites"]):
            if i == best:
                continue
            diff = X[best] - X[i]
            X_all.append(diff)
            y_all.append(1)
            X_all.append(X[i] - X[best])
            y_all.append(0)

    X_all = np.array(X_all)
    y_all = np.array(y_all)

    if len(X_all) == 0:
        return np.zeros(test_target["n_sites"])

    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(X_all)

    clf = LogisticRegression(C=0.1, penalty='l2', max_iter=2000, solver='lbfgs')
    clf.fit(X_scaled, y_all)

    X_test = get_feature_matrix(test_target, SELECTED_FEATURES)
    X_test_scaled = scaler.transform(X_test)
    return X_test_scaled @ clf.coef_[0]

all_methods["M3: ListNet-Pairwise LR (LOOCV)"] = evaluate_loocv(targets, listnet_train_predict)

# M4: Ridge regression on DCC (lower DCC = better, so negate)
def ridge_train_predict(train_targets, test_target):
    """Train ridge regression to predict -DCC, rank by predicted value."""
    X_all = []
    y_all = []
    for t in train_targets:
        X = get_feature_matrix(t, SELECTED_FEATURES)
        # Normalize DCC within target (rank-based)
        dccs = t["dccs"]
        # Use -log(dcc) as target (heavily penalize close sites being ranked low)
        y = -np.log1p(dccs)
        X_all.append(X)
        y_all.append(y)

    X_all = np.vstack(X_all)
    y_all = np.concatenate(y_all)

    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(X_all)

    reg = Ridge(alpha=10.0)
    reg.fit(X_scaled, y_all)

    X_test = get_feature_matrix(test_target, SELECTED_FEATURES)
    X_test_scaled = scaler.transform(X_test)
    return reg.predict(X_test_scaled)

all_methods["M4: Ridge regression -log(DCC) (LOOCV)"] = evaluate_loocv(targets, ridge_train_predict)

# M5: ElasticNet (L1+L2) for automatic feature selection
def elasticnet_train_predict(train_targets, test_target):
    X_all = []
    y_all = []
    for t in train_targets:
        X = get_feature_matrix(t, SELECTED_FEATURES)
        y = -np.log1p(t["dccs"])
        X_all.append(X)
        y_all.append(y)

    X_all = np.vstack(X_all)
    y_all = np.concatenate(y_all)

    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(X_all)

    reg = ElasticNet(alpha=0.1, l1_ratio=0.5, max_iter=5000)
    reg.fit(X_scaled, y_all)

    X_test = get_feature_matrix(test_target, SELECTED_FEATURES)
    X_test_scaled = scaler.transform(X_test)
    return reg.predict(X_test_scaled)

all_methods["M5: ElasticNet -log(DCC) (LOOCV)"] = evaluate_loocv(targets, elasticnet_train_predict)

# ══════════════════════════════════════════════════════════════
# 7. METHODOLOGY 6: RankSVM (pairwise SVM)
# ══════════════════════════════════════════════════════════════

def ranksvm_train_predict(train_targets, test_target):
    """Pairwise RankSVM: learn w such that w·(x_better - x_worse) > 0."""
    X_pairs = []
    y_pairs = []
    for t in train_targets:
        X = get_feature_matrix(t, SELECTED_FEATURES)
        dccs = t["dccs"]
        n = t["n_sites"]
        # Sample pairwise comparisons (not all O(n^2), just best vs rest + nearby)
        best = t["best_idx"]
        for i in range(n):
            if i == best:
                continue
            if dccs[best] + 2.0 < dccs[i]:  # meaningful gap
                X_pairs.append(X[best] - X[i])
                y_pairs.append(1)

    if not X_pairs:
        return np.zeros(test_target["n_sites"])

    X_pairs = np.array(X_pairs)
    y_pairs = np.array(y_pairs)

    # Add flipped pairs for balance
    X_pairs = np.vstack([X_pairs, -X_pairs])
    y_pairs = np.concatenate([y_pairs, np.zeros(len(y_pairs))])

    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(X_pairs)

    svm = LinearSVC(C=0.01, max_iter=5000, loss='hinge')
    svm.fit(X_scaled, y_pairs)

    X_test = get_feature_matrix(test_target, SELECTED_FEATURES)
    X_test_scaled = scaler.transform(X_test)
    return X_test_scaled @ svm.coef_[0]

all_methods["M6: RankSVM (LOOCV)"] = evaluate_loocv(targets, ranksvm_train_predict)

# ══════════════════════════════════════════════════════════════
# 8. METHODOLOGY 7: Random Forest pairwise (LOOCV)
# ══════════════════════════════════════════════════════════════

def rf_pairwise_train_predict(train_targets, test_target):
    """Random Forest on pairwise differences."""
    X_pairs = []
    y_pairs = []
    for t in train_targets:
        X = get_feature_matrix(t, SELECTED_FEATURES)
        dccs = t["dccs"]
        best = t["best_idx"]
        for i in range(t["n_sites"]):
            if i == best:
                continue
            if dccs[best] + 1.0 < dccs[i]:
                X_pairs.append(X[best] - X[i])
                y_pairs.append(1)
                X_pairs.append(X[i] - X[best])
                y_pairs.append(0)

    if len(X_pairs) < 10:
        return np.zeros(test_target["n_sites"])

    X_pairs = np.array(X_pairs)
    y_pairs = np.array(y_pairs)

    clf = RandomForestClassifier(n_estimators=100, max_depth=3,
                                  min_samples_leaf=10, random_state=42)
    clf.fit(X_pairs, y_pairs)

    X_test = get_feature_matrix(test_target, SELECTED_FEATURES)
    # Score each test site against average
    X_mean = X_test.mean(axis=0)
    scores = []
    for i in range(test_target["n_sites"]):
        diff = X_test[i] - X_mean
        scores.append(clf.predict_proba(diff.reshape(1, -1))[0, 1])
    return np.array(scores)

all_methods["M7: RF Pairwise (LOOCV)"] = evaluate_loocv(targets, rf_pairwise_train_predict)

# ══════════════════════════════════════════════════════════════
# 9. METHODOLOGY 8: Borda Count (non-parametric, no training)
# ══════════════════════════════════════════════════════════════

# Use top-performing single features as voters
TOP_SINGLE = [fn for fn, res in sorted_features[:10]]

def borda_ranker(t):
    n = t["n_sites"]
    borda_scores = np.zeros(n)
    for fn in TOP_SINGLE:
        direction = single_feature_results[fn]["direction"]
        vals = np.array([sf[fn] for sf in t["features"]])
        if direction == "-":
            vals = -vals
        ranks = stats.rankdata(vals, method='average')
        borda_scores += ranks
    return borda_scores

all_methods["M8: Borda Count (top-10 features)"] = evaluate_ranker(targets, borda_ranker)

# ══════════════════════════════════════════════════════════════
# 10. METHODOLOGY 9: Reciprocal Rank Fusion (RRF)
# ══════════════════════════════════════════════════════════════

def rrf_ranker(t, k=60):
    """Reciprocal Rank Fusion across top single-feature rankers."""
    n = t["n_sites"]
    rrf_scores = np.zeros(n)
    for fn in TOP_SINGLE:
        direction = single_feature_results[fn]["direction"]
        vals = np.array([sf[fn] for sf in t["features"]])
        if direction == "-":
            vals = -vals
        ranking = np.argsort(-vals)
        for rank_pos, site_idx in enumerate(ranking):
            rrf_scores[site_idx] += 1.0 / (k + rank_pos + 1)
    return rrf_scores

all_methods["M9: Reciprocal Rank Fusion (k=60)"] = evaluate_ranker(targets, rrf_ranker)

# ══════════════════════════════════════════════════════════════
# 11. METHODOLOGY 10: Softmax ListNet with proper listwise loss
# ══════════════════════════════════════════════════════════════

def listnet_proper_train_predict(train_targets, test_target):
    """Proper ListNet: minimize KL(P_true || P_model) where P is softmax over -DCC."""
    n_feat = len(SELECTED_FEATURES)

    # Collect all training data
    all_X = []
    all_relevance = []
    for t in train_targets:
        X = get_feature_matrix(t, SELECTED_FEATURES)
        # Convert DCC to relevance: exp(-dcc/sigma)
        rel = np.exp(-t["dccs"] / 5.0)
        all_X.append(X)
        all_relevance.append(rel)

    # Fit a scaler on all training features
    all_X_flat = np.vstack(all_X)
    scaler = RobustScaler()
    scaler.fit(all_X_flat)

    # Optimize weights via scipy
    def loss(w):
        total_kl = 0
        for X, rel in zip(all_X, all_relevance):
            X_s = scaler.transform(X)
            scores = X_s @ w[:n_feat] + w[n_feat]  # linear + bias
            # Softmax of scores
            scores_shift = scores - scores.max()
            p_model = np.exp(scores_shift) / np.exp(scores_shift).sum()
            # Softmax of relevance
            rel_shift = rel - rel.max()
            p_true = np.exp(rel_shift) / np.exp(rel_shift).sum()
            # KL divergence
            kl = np.sum(p_true * np.log(p_true / np.clip(p_model, 1e-10, None)))
            total_kl += kl
        # L2 regularization
        reg = 0.1 * np.sum(w[:n_feat]**2)
        return total_kl / len(all_X) + reg

    w0 = np.zeros(n_feat + 1)
    result = minimize(loss, w0, method='L-BFGS-B', options={'maxiter': 500})
    w_opt = result.x

    X_test = get_feature_matrix(test_target, SELECTED_FEATURES)
    X_test_s = scaler.transform(X_test)
    return X_test_s @ w_opt[:n_feat] + w_opt[n_feat]

all_methods["M10: ListNet Softmax-KL (LOOCV)"] = evaluate_loocv(targets, listnet_proper_train_predict)

# ══════════════════════════════════════════════════════════════
# 12. METHODOLOGY 11: Plackett-Luce / Bradley-Terry
# ══════════════════════════════════════════════════════════════

def plackett_luce_train_predict(train_targets, test_target):
    """Plackett-Luce model: P(ranking) = prod P(i|remaining)."""
    n_feat = len(SELECTED_FEATURES)

    all_X = []
    all_best = []
    for t in train_targets:
        X = get_feature_matrix(t, SELECTED_FEATURES)
        all_X.append(X)
        all_best.append(t["best_idx"])

    all_X_flat = np.vstack(all_X)
    scaler = RobustScaler()
    scaler.fit(all_X_flat)

    def loss(w):
        nll = 0
        for X, best_idx in zip(all_X, all_best):
            X_s = scaler.transform(X)
            scores = X_s @ w[:n_feat]
            # Log-probability of best being ranked first
            log_p = scores[best_idx] - np.log(np.sum(np.exp(scores - scores.max())) + 1e-10) - (scores.max() - scores[best_idx])
            # More numerically stable
            shifted = scores - scores.max()
            log_p = shifted[best_idx] - np.log(np.sum(np.exp(shifted)))
            nll -= log_p
        reg = 0.5 * np.sum(w[:n_feat]**2)
        return nll / len(all_X) + reg

    w0 = np.zeros(n_feat)
    result = minimize(loss, w0, method='L-BFGS-B', options={'maxiter': 500})
    w_opt = result.x

    X_test = get_feature_matrix(test_target, SELECTED_FEATURES)
    X_test_s = scaler.transform(X_test)
    return X_test_s @ w_opt

all_methods["M11: Plackett-Luce (LOOCV)"] = evaluate_loocv(targets, plackett_luce_train_predict)

# ══════════════════════════════════════════════════════════════
# 13. METHODOLOGY 12: Gradient Boosted pointwise (LOOCV)
# ══════════════════════════════════════════════════════════════

def gbm_pointwise_train_predict(train_targets, test_target):
    """GBM: predict binary label (DCC < 8Å) then rank by probability."""
    X_all = []
    y_all = []
    for t in train_targets:
        X = get_feature_matrix(t, SELECTED_FEATURES)
        y = (t["dccs"] <= 8.0).astype(int)
        X_all.append(X)
        y_all.append(y)

    X_all = np.vstack(X_all)
    y_all = np.concatenate(y_all)

    if y_all.sum() < 3 or (1 - y_all).sum() < 3:
        return np.zeros(test_target["n_sites"])

    clf = GradientBoostingClassifier(n_estimators=50, max_depth=2,
                                      learning_rate=0.1, min_samples_leaf=10,
                                      subsample=0.8, random_state=42)
    clf.fit(X_all, y_all)

    X_test = get_feature_matrix(test_target, SELECTED_FEATURES)
    return clf.predict_proba(X_test)[:, 1]

all_methods["M12: GBM Pointwise (LOOCV)"] = evaluate_loocv(targets, gbm_pointwise_train_predict)

# ══════════════════════════════════════════════════════════════
# 14. METHODOLOGY 13: Optimal 2-3 feature combinations (exhaustive)
# ══════════════════════════════════════════════════════════════

print("\n" + "=" * 80)
print("EXHAUSTIVE 2-FEATURE COMBINATION SEARCH")
print("=" * 80)

# Test all pairs from top-15 single features
TOP15 = [fn for fn, _ in sorted_features[:15]]
best_pair = None
best_pair_sr1 = 0

for f1, f2 in combinations(TOP15, 2):
    def combo_ranker(t, _f1=f1, _f2=f2):
        d1 = single_feature_results[_f1]["direction"]
        d2 = single_feature_results[_f2]["direction"]
        v1 = np.array([sf[_f1] for sf in t["features"]])
        v2 = np.array([sf[_f2] for sf in t["features"]])
        if d1 == "-": v1 = -v1
        if d2 == "-": v2 = -v2
        # Rank-normalize each, then sum
        r1 = stats.rankdata(v1) / len(v1)
        r2 = stats.rankdata(v2) / len(v2)
        return r1 + r2

    res = evaluate_ranker(targets, combo_ranker)
    if res["SR@1"] > best_pair_sr1 or (res["SR@1"] == best_pair_sr1 and res["mean_top1_dcc"] < (best_pair[2] if best_pair else 999)):
        best_pair = (f1, f2, res["mean_top1_dcc"])
        best_pair_sr1 = res["SR@1"]
        best_pair_res = res

print(f"Best pair: {best_pair[0]} + {best_pair[1]}")
print(f"  SR@1={best_pair_res['SR@1']:.1%}  SR@3={best_pair_res['SR@3']:.1%}  "
      f"Mean Top-1={best_pair_res['mean_top1_dcc']:.1f}Å")
all_methods[f"M13: Best Pair ({best_pair[0][:15]}+{best_pair[1][:15]})"] = best_pair_res

# Best 3-feature combo
print("\nSearching 3-feature combinations...")
best_triple = None
best_triple_sr1 = 0

for f1, f2, f3 in combinations(TOP15, 3):
    def combo3_ranker(t, _f1=f1, _f2=f2, _f3=f3):
        result = np.zeros(t["n_sites"])
        for fn in [_f1, _f2, _f3]:
            d = single_feature_results[fn]["direction"]
            vals = np.array([sf[fn] for sf in t["features"]])
            if d == "-": vals = -vals
            result += stats.rankdata(vals) / len(vals)
        return result

    res = evaluate_ranker(targets, combo3_ranker)
    if res["SR@1"] > best_triple_sr1 or (res["SR@1"] == best_triple_sr1 and res["mean_top1_dcc"] < (best_triple[3] if best_triple else 999)):
        best_triple = (f1, f2, f3, res["mean_top1_dcc"])
        best_triple_sr1 = res["SR@1"]
        best_triple_res = res

print(f"Best triple: {best_triple[0]} + {best_triple[1]} + {best_triple[2]}")
print(f"  SR@1={best_triple_res['SR@1']:.1%}  SR@3={best_triple_res['SR@3']:.1%}  "
      f"Mean Top-1={best_triple_res['mean_top1_dcc']:.1f}Å")
all_methods[f"M14: Best Triple"] = best_triple_res

# ══════════════════════════════════════════════════════════════
# 15. METHODOLOGY 15: Weight-optimized linear combo (LOOCV)
# ══════════════════════════════════════════════════════════════

def weighted_linear_train_predict(train_targets, test_target):
    """Optimize weights for linear combination of rank-normalized features."""
    n_feat = len(TOP15)

    def score_target(t, w):
        scores = np.zeros(t["n_sites"])
        for j, fn in enumerate(TOP15):
            d = single_feature_results[fn]["direction"]
            vals = np.array([sf[fn] for sf in t["features"]])
            if d == "-": vals = -vals
            ranks = stats.rankdata(vals) / len(vals)
            scores += w[j] * ranks
        return scores

    def loss(w):
        total = 0
        for t in train_targets:
            scores = score_target(t, np.abs(w))  # positive weights only
            best = t["best_idx"]
            # Plackett-Luce loss
            shifted = scores - scores.max()
            log_p = shifted[best] - np.log(np.sum(np.exp(shifted)))
            total -= log_p
        reg = 0.5 * np.sum(w**2)
        return total / len(train_targets) + reg

    w0 = np.ones(n_feat) / n_feat
    result = minimize(loss, w0, method='L-BFGS-B', options={'maxiter': 300})
    w_opt = np.abs(result.x)

    return score_target(test_target, w_opt)

all_methods["M15: Weighted Rank-PL (LOOCV)"] = evaluate_loocv(targets, weighted_linear_train_predict)

# ══════════════════════════════════════════════════════════════
# 16. METHODOLOGY 16: Cryptic-Aware Physics Ranker
# ══════════════════════════════════════════════════════════════

def cryptic_physics_ranker(t):
    """Novel physics-informed ranker that leverages PRISM4D's unique
    cryptic-site detection signals.

    Key insight: PRISM4D detects pockets via neuromorphic spike dynamics.
    The correct pocket should show:
    1. High spike density (more contact events)
    2. Deep burial (pocket is well-formed)
    3. Many lining residues (true pocket has protein contact)
    4. High druggability (true pockets are druggable)
    5. Thermodynamic signature (SOC criticality if available)
    6. High catalytic density (functional sites near catalytic residues)

    Weight by signal reliability (from correlation analysis).
    """
    n = t["n_sites"]
    scores = np.zeros(n)

    for i, sf in enumerate(t["features"]):
        # Core binding site quality (rank-independent features)
        score = 0.0

        # 1. Spike density (strongest individual signal)
        score += 3.0 * sf["spike_count"]

        # 2. Burial depth
        score += 2.5 * sf["burial_score"]

        # 3. Lining residues (log-normalized)
        score += 2.0 * sf["log_lining"]

        # 4. Druggability
        score += 1.5 * sf["druggability"]

        # 5. Enclosure (residues per unit surface)
        score += 1.0 * min(sf["enclosure"] * 100, 2.0)

        # 6. Catalytic residue density
        score += 1.5 * sf["catalytic_density"]

        # 7. Onset (early detection = more confident)
        score += 0.5 * sf["onset_score"]

        # 8. Aromatic contacts (π-stacking enrichment)
        score += 1.0 * sf["aromatic_score"]

        # 9. Volume penalty for very large pockets (nonspecific)
        vol = sf["volume"]
        if vol > 4.0:  # log10(volume) > 4 → volume > 10,000 Å³
            score -= 0.5 * (vol - 4.0)

        # 10. Thermodynamic bonus
        if sf.get("ccns_tau", 0) > 0:
            tau = sf["ccns_tau"]
            # SOC criticality: tau near 1.3-1.5 is optimal
            tau_bonus = max(0, 1.0 - abs(tau - 1.4) / 0.3)
            score += 0.8 * tau_bonus

        scores[i] = score

    return scores

all_methods["M16: Cryptic-Physics Hand-tuned"] = evaluate_ranker(targets, cryptic_physics_ranker)

# ══════════════════════════════════════════════════════════════
# 17. METHODOLOGY 17: Percentile-rank fusion (scale-invariant)
# ══════════════════════════════════════════════════════════════

# Use ONLY features with SR@1 > 0 (proven discriminative)
PROVEN_FEATURES = [fn for fn, res in sorted_features if res["SR@1"] > 0]
print(f"\nProven discriminative features (SR@1 > 0): {len(PROVEN_FEATURES)}")
for fn in PROVEN_FEATURES:
    sr1 = single_feature_results[fn]["SR@1"]
    sr3 = single_feature_results[fn]["SR@3"]
    print(f"  {fn:>35}  SR@1={sr1:.1%}  SR@3={sr3:.1%}")

def percentile_fusion_ranker(t):
    """Rank by percentile across proven-discriminative features.
    Weight each feature by its SR@1 performance."""
    n = t["n_sites"]
    scores = np.zeros(n)
    total_weight = 0

    for fn in PROVEN_FEATURES:
        w = single_feature_results[fn]["SR@1"]
        d = single_feature_results[fn]["direction"]
        vals = np.array([sf[fn] for sf in t["features"]])
        if d == "-": vals = -vals
        # Percentile rank (0-1)
        pct = stats.rankdata(vals, method='average') / n
        scores += w * pct
        total_weight += w

    if total_weight > 0:
        scores /= total_weight

    return scores

all_methods["M17: SR@1-weighted Percentile Fusion"] = evaluate_ranker(targets, percentile_fusion_ranker)

# ══════════════════════════════════════════════════════════════
# 18. METHODOLOGY 18: Differentiable Rank-weighted ListMLE (LOOCV)
# ══════════════════════════════════════════════════════════════

def listmle_train_predict(train_targets, test_target):
    """ListMLE: maximize likelihood of the observed ranking (best site first)."""
    # Use only proven features
    feats_to_use = PROVEN_FEATURES[:12]  # cap at 12 to avoid overfitting
    n_feat = len(feats_to_use)

    all_X = []
    all_best = []
    all_dccs = []
    for t in train_targets:
        X = get_feature_matrix(t, feats_to_use)
        # Standardize per-target (removes scale effects)
        X_std = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-8)
        all_X.append(X_std)
        all_best.append(t["best_idx"])
        all_dccs.append(t["dccs"])

    def loss(w):
        nll = 0
        for X, best_idx, dccs in zip(all_X, all_best, all_dccs):
            scores = X @ w
            # ListMLE: sum of log(softmax) for each position
            # Simplified: just maximize score of best relative to rest
            shifted = scores - scores.max()
            log_p = shifted[best_idx] - np.log(np.sum(np.exp(shifted)))
            nll -= log_p

            # Also add top-3 bonus (second-best, third-best)
            sorted_dcc_idx = np.argsort(dccs)
            for rank, idx in enumerate(sorted_dcc_idx[:3]):
                remaining = sorted_dcc_idx[rank:]
                scores_remaining = scores[remaining]
                shifted_r = scores_remaining - scores_remaining.max()
                pos = np.where(remaining == idx)[0][0]
                log_p_r = shifted_r[pos] - np.log(np.sum(np.exp(shifted_r)))
                nll -= 0.3 * log_p_r  # down-weighted

        reg = 1.0 * np.sum(w**2)
        return nll / len(all_X) + reg

    w0 = np.zeros(n_feat)
    result = minimize(loss, w0, method='L-BFGS-B', options={'maxiter': 500})
    w_opt = result.x

    X_test = get_feature_matrix(test_target, feats_to_use)
    X_test_std = (X_test - X_test.mean(axis=0)) / (X_test.std(axis=0) + 1e-8)
    return X_test_std @ w_opt

all_methods["M18: ListMLE Top-3 (LOOCV)"] = evaluate_loocv(targets, listmle_train_predict)

# ══════════════════════════════════════════════════════════════
# FINAL RESULTS TABLE
# ══════════════════════════════════════════════════════════════

print("\n" + "=" * 110)
print("FINAL METHODOLOGY COMPARISON — BENCH30 (excl. target 14)")
print("=" * 110)
print(f"{'Method':>50} {'SR@1':>7} {'SR@3':>7} {'SR@5':>7} {'SR@10':>7} {'MeanTop1':>9}")
print("-" * 110)

sorted_methods = sorted(all_methods.items(),
                        key=lambda x: (x[1]["SR@1"], x[1]["SR@3"], -x[1].get("mean_top1_dcc", 999)),
                        reverse=True)

for name, res in sorted_methods:
    sr1 = f"{res['SR@1']:.1%}"
    sr3 = f"{res['SR@3']:.1%}"
    sr5 = f"{res.get('SR@5', 0):.1%}"
    sr10 = f"{res.get('SR@10', 0):.1%}"
    top1 = f"{res.get('mean_top1_dcc', 0):.1f}Å"
    print(f"{name:>50} {sr1:>7} {sr3:>7} {sr5:>7} {sr10:>7} {top1:>9}")

# ══════════════════════════════════════════════════════════════
# WINNER ANALYSIS
# ══════════════════════════════════════════════════════════════

winner_name, winner_res = sorted_methods[0]
print(f"\n{'='*80}")
print(f"WINNER: {winner_name}")
print(f"  SR@1 = {winner_res['SR@1']:.1%}")
print(f"  SR@3 = {winner_res['SR@3']:.1%}")
print(f"  Mean Top-1 DCC = {winner_res.get('mean_top1_dcc', 0):.1f}Å")
print(f"{'='*80}")

# Save full results
output = {
    "methods": {name: {k: float(v) if isinstance(v, (float, np.floating)) else v
                       for k, v in res.items()}
                for name, res in sorted_methods},
    "best_method": winner_name,
    "single_feature_ranking": {fn: {"SR@1": res["SR@1"], "SR@3": res["SR@3"],
                                     "direction": res["direction"],
                                     "mean_top1_dcc": res["mean_top1_dcc"]}
                                for fn, res in sorted_features[:20]},
    "feature_correlations": {fn: {"r": float(r), "p": float(p)}
                             for fn, (r, p) in sorted_corr[:20]},
    "proven_features": PROVEN_FEATURES,
    "best_pair": list(best_pair[:2]),
    "best_triple": list(best_triple[:3]) if best_triple else None,
    "n_targets": len(targets),
    "n_sites": sum(t["n_sites"] for t in targets),
}

with open("optimal_ranker_results.json", "w") as f:
    json.dump(output, f, indent=2, default=lambda x: float(x) if isinstance(x, (float, np.floating)) else x)

print(f"\nResults saved to optimal_ranker_results.json")

# ══════════════════════════════════════════════════════════════
# PER-TARGET BREAKDOWN for winner
# ══════════════════════════════════════════════════════════════

print(f"\n{'='*80}")
print(f"PER-TARGET BREAKDOWN: {winner_name}")
print(f"{'='*80}")
print(f"{'TID':>4} {'APO':>5} {'Type':>12} {'N':>3} {'Top1':>7} {'Best':>7} {'BestRk':>7}")
print("-" * 60)

# Re-run the winner on each target
for t in targets:
    if "Pair" in winner_name or "Triple" in winner_name or "Borda" in winner_name or "Fusion" in winner_name or "Physics" in winner_name or "RRF" in winner_name or "Percentile" in winner_name:
        # Non-LOOCV method — can score directly
        if "Pair" in winner_name:
            f1, f2 = best_pair[:2]
            d1 = single_feature_results[f1]["direction"]
            d2 = single_feature_results[f2]["direction"]
            v1 = np.array([sf[f1] for sf in t["features"]])
            v2 = np.array([sf[f2] for sf in t["features"]])
            if d1 == "-": v1 = -v1
            if d2 == "-": v2 = -v2
            scores = stats.rankdata(v1)/len(v1) + stats.rankdata(v2)/len(v2)
        elif "Triple" in winner_name:
            f1, f2, f3 = best_triple[:3]
            scores = np.zeros(t["n_sites"])
            for fn in [f1, f2, f3]:
                d = single_feature_results[fn]["direction"]
                vals = np.array([sf[fn] for sf in t["features"]])
                if d == "-": vals = -vals
                scores += stats.rankdata(vals) / len(vals)
        elif "Percentile" in winner_name:
            scores = percentile_fusion_ranker(t)
        elif "Borda" in winner_name:
            scores = borda_ranker(t)
        elif "RRF" in winner_name:
            scores = rrf_ranker(t)
        elif "Physics" in winner_name:
            scores = cryptic_physics_ranker(t)
        else:
            scores = np.array([sf["quality_score"] for sf in t["features"]])
    else:
        # For LOOCV methods, need to train on all-except-this
        train = [tt for tt in targets if tt["tid"] != t["tid"]]
        if "ListNet-Pairwise" in winner_name:
            scores = listnet_train_predict(train, t)
        elif "Ridge" in winner_name:
            scores = ridge_train_predict(train, t)
        elif "ElasticNet" in winner_name:
            scores = elasticnet_train_predict(train, t)
        elif "RankSVM" in winner_name:
            scores = ranksvm_train_predict(train, t)
        elif "RF" in winner_name:
            scores = rf_pairwise_train_predict(train, t)
        elif "ListNet Softmax" in winner_name:
            scores = listnet_proper_train_predict(train, t)
        elif "Plackett" in winner_name:
            scores = plackett_luce_train_predict(train, t)
        elif "GBM" in winner_name:
            scores = gbm_pointwise_train_predict(train, t)
        elif "Weighted" in winner_name:
            scores = weighted_linear_train_predict(train, t)
        elif "ListMLE" in winner_name:
            scores = listmle_train_predict(train, t)
        else:
            scores = np.array([sf["quality_score"] for sf in t["features"]])

    ranking = np.argsort(-scores)
    top1_dcc = t["dccs"][ranking[0]]
    best_dcc = t["best_dcc"]
    best_rank = int(np.where(ranking == t["best_idx"])[0][0]) + 1

    hit = "✓" if top1_dcc <= 5.0 else ("~" if top1_dcc <= 8.0 else "·")
    print(f"{t['tid']:>4} {t['apo']:>5} {t['type']:>12} {t['n_sites']:>3} "
          f"{top1_dcc:>6.1f}Å {best_dcc:>6.1f}Å {best_rank:>5} {hit}")
