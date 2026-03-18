#!/usr/bin/env python3
"""
XGBoost Pairwise Ranker for PRISM4D pocket detection.
Trains on site_features_all.csv with leave-one-target-out cross-validation.

Objective: rank:pairwise with qid=target_id
Features: all numeric features from the CSV
Target: DCC-based relevance label
"""
import json, os, sys
import numpy as np
import csv

try:
    import xgboost as xgb
    print(f"XGBoost {xgb.__version__}")
except ImportError:
    print("ERROR: xgboost required. Install with: pip install xgboost")
    sys.exit(1)

os.chdir(os.path.dirname(os.path.abspath(__file__)))

# ── Load CSV ──
rows = []
with open("site_features_all.csv") as f:
    reader = csv.DictReader(f)
    for r in reader:
        rows.append(r)

print(f"Loaded {len(rows)} sites from site_features_all.csv")

# ── Feature columns (all numeric, exclude identifiers and labels) ──
EXCLUDE = {"target_id", "apo_pdb", "site_id", "rank", "dcc", "hit_5A", "hit_8A", "therm_class"}
FEATURE_COLS = [k for k in rows[0].keys() if k not in EXCLUDE]
print(f"Features ({len(FEATURE_COLS)}): {FEATURE_COLS}")

# ── Build arrays ──
target_ids = []
X_all = []
y_all = []  # relevance labels
dcc_all = []

for r in rows:
    target_ids.append(r["target_id"])
    feat = []
    for col in FEATURE_COLS:
        try:
            feat.append(float(r[col]))
        except (ValueError, TypeError):
            feat.append(0.0)
    X_all.append(feat)
    dcc = float(r["dcc"])
    dcc_all.append(dcc)
    # Relevance: 3=excellent (≤4Å), 2=good (≤5Å), 1=ok (≤8Å), 0=miss
    if dcc <= 4.0:
        y_all.append(3)
    elif dcc <= 5.0:
        y_all.append(2)
    elif dcc <= 8.0:
        y_all.append(1)
    else:
        y_all.append(0)

X_all = np.array(X_all, dtype=np.float32)
y_all = np.array(y_all, dtype=np.int32)
dcc_all = np.array(dcc_all, dtype=np.float32)
target_ids = np.array(target_ids)

unique_targets = sorted(set(target_ids), key=int)
n_targets = len(unique_targets)
print(f"{n_targets} targets, relevance distribution: {dict(zip(*np.unique(y_all, return_counts=True)))}")

# ── Leave-One-Target-Out Cross-Validation ──
print(f"\n{'='*80}")
print("LEAVE-ONE-TARGET-OUT CROSS-VALIDATION")
print(f"{'='*80}\n")

params = {
    "objective": "rank:pairwise",
    "eta": 0.1,
    "max_depth": 4,
    "min_child_weight": 5,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "eval_metric": "ndcg",
    "seed": 42,
    "verbosity": 0,
}

sr1_5_count = 0
sr1_8_count = 0
sr3_5_count = 0
sr3_8_count = 0
n_evaluated = 0
per_target = []

print(f"{'TID':>4s} {'APO':>5s} {'Sites':>6s} {'BestDCC':>8s} {'XGB Top1':>9s} {'XGB Top3':>9s} {'Rank':>5s} {'@1':>3s} {'@3':>3s}")
print("-" * 65)

for held_out in unique_targets:
    # Split
    train_mask = target_ids != held_out
    test_mask = target_ids == held_out

    X_train, y_train = X_all[train_mask], y_all[train_mask]
    X_test, y_test = X_all[test_mask], y_all[test_mask]
    dcc_test = dcc_all[test_mask]

    # Build qid (group sizes) for training
    train_targets = target_ids[train_mask]
    train_groups = []
    for tid in unique_targets:
        if tid == held_out:
            continue
        train_groups.append(int(np.sum(train_targets == tid)))

    # Build test group
    test_groups = [int(np.sum(test_mask))]

    dtrain = xgb.DMatrix(X_train, label=y_train)
    dtrain.set_group(train_groups)
    dtest = xgb.DMatrix(X_test)

    model = xgb.train(params, dtrain, num_boost_round=200)

    # Predict and rank
    scores = model.predict(dtest)
    ranked_indices = np.argsort(-scores)
    ranked_dcc = dcc_test[ranked_indices]

    top1_dcc = ranked_dcc[0]
    top3_dcc = min(ranked_dcc[:3]) if len(ranked_dcc) >= 3 else min(ranked_dcc)
    best_dcc = min(dcc_test)
    best_rank = int(np.where(ranked_indices == np.argmin(dcc_test))[0][0]) + 1

    # Find APO PDB for display
    test_rows = [r for r in rows if r["target_id"] == held_out]
    apo = test_rows[0]["apo_pdb"] if test_rows else "?"

    h1_5 = "Y" if top1_dcc <= 5.0 else "."
    h3_5 = "Y" if top3_dcc <= 5.0 else "."

    if top1_dcc <= 5.0: sr1_5_count += 1
    if top1_dcc <= 8.0: sr1_8_count += 1
    if top3_dcc <= 5.0: sr3_5_count += 1
    if top3_dcc <= 8.0: sr3_8_count += 1
    n_evaluated += 1

    per_target.append({
        "tid": held_out, "apo": apo, "n_sites": len(dcc_test),
        "best_dcc": best_dcc, "top1_dcc": top1_dcc, "top3_dcc": top3_dcc,
        "best_rank": best_rank,
    })

    print(f"{held_out:>4s} {apo:>5s} {len(dcc_test):>6d} {best_dcc:>7.1f}A {top1_dcc:>8.1f}A {top3_dcc:>8.1f}A {best_rank:>5d}  {h1_5}   {h3_5}")

# ── Aggregate ──
print(f"\n{'='*80}")
print("LOO-CV AGGREGATE RESULTS")
print(f"{'='*80}")
print(f"  SR@1 ≤5Å: {sr1_5_count}/{n_evaluated} ({sr1_5_count/n_evaluated*100:.0f}%)")
print(f"  SR@1 ≤8Å: {sr1_8_count}/{n_evaluated} ({sr1_8_count/n_evaluated*100:.0f}%)")
print(f"  SR@3 ≤5Å: {sr3_5_count}/{n_evaluated} ({sr3_5_count/n_evaluated*100:.0f}%)")
print(f"  SR@3 ≤8Å: {sr3_8_count}/{n_evaluated} ({sr3_8_count/n_evaluated*100:.0f}%)")
print(f"  Mean Top-1 DCC: {np.mean([p['top1_dcc'] for p in per_target]):.1f}Å")
print(f"  Median Top-1 DCC: {np.median([p['top1_dcc'] for p in per_target]):.1f}Å")

# ── Train final model on all data ──
print(f"\n{'='*80}")
print("TRAINING FINAL MODEL (all data)")
print(f"{'='*80}")

all_groups = []
for tid in unique_targets:
    all_groups.append(int(np.sum(target_ids == tid)))

dtrain_all = xgb.DMatrix(X_all, label=y_all)
dtrain_all.set_group(all_groups)
final_model = xgb.train(params, dtrain_all, num_boost_round=200)

# Feature importance
importance = final_model.get_score(importance_type="gain")
sorted_imp = sorted(importance.items(), key=lambda x: -x[1])
print(f"\nFeature importance (gain):")
for fname, gain in sorted_imp[:15]:
    # Map f0, f1, ... back to feature names
    idx = int(fname.replace("f", ""))
    real_name = FEATURE_COLS[idx] if idx < len(FEATURE_COLS) else fname
    print(f"  {real_name:<28s} {gain:>10.1f}")

# Save model
model_path = "xgb_ranker_model.json"
final_model.save_model(model_path)
print(f"\nModel saved to {model_path}")

# Save feature column order for inference
meta = {
    "feature_columns": FEATURE_COLS,
    "params": params,
    "loo_sr1_5": sr1_5_count / n_evaluated,
    "loo_sr3_5": sr3_5_count / n_evaluated,
    "loo_sr1_8": sr1_8_count / n_evaluated,
    "loo_sr3_8": sr3_8_count / n_evaluated,
    "n_targets": n_evaluated,
}
with open("xgb_ranker_meta.json", "w") as f:
    json.dump(meta, f, indent=2)
print(f"Metadata saved to xgb_ranker_meta.json")

# ── Compare vs current quality_score ranking ──
print(f"\n{'='*80}")
print("COMPARISON: XGBoost LOO-CV vs Current quality_score Ranking")
print(f"{'='*80}")

qs_sr1_5 = sum(1 for p in per_target if any(
    float(r["dcc"]) <= 5.0 and int(r["rank"]) == 1
    for r in rows if r["target_id"] == p["tid"]))
qs_sr3_5 = sum(1 for p in per_target if any(
    float(r["dcc"]) <= 5.0 and int(r["rank"]) <= 3
    for r in rows if r["target_id"] == p["tid"]))
qs_sr1_8 = sum(1 for p in per_target if any(
    float(r["dcc"]) <= 8.0 and int(r["rank"]) == 1
    for r in rows if r["target_id"] == p["tid"]))
qs_sr3_8 = sum(1 for p in per_target if any(
    float(r["dcc"]) <= 8.0 and int(r["rank"]) <= 3
    for r in rows if r["target_id"] == p["tid"]))

print(f"{'Metric':<20s} {'XGB LOO-CV':>12s} {'quality_score':>14s}")
print(f"{'-'*48}")
print(f"{'SR@1 ≤5Å':<20s} {sr1_5_count:>5d}/{n_evaluated:<5d} {qs_sr1_5:>7d}/{n_evaluated:<5d}")
print(f"{'SR@3 ≤5Å':<20s} {sr3_5_count:>5d}/{n_evaluated:<5d} {qs_sr3_5:>7d}/{n_evaluated:<5d}")
print(f"{'SR@1 ≤8Å':<20s} {sr1_8_count:>5d}/{n_evaluated:<5d} {qs_sr1_8:>7d}/{n_evaluated:<5d}")
print(f"{'SR@3 ≤8Å':<20s} {sr3_8_count:>5d}/{n_evaluated:<5d} {qs_sr3_8:>7d}/{n_evaluated:<5d}")
