#!/usr/bin/env python3
"""
Full pocket ranker: Uni-Mol 512-dim embeddings + ALL PRISM4D engine features.
Uses gradient-boosted trees (XGBoost) with LOTO CV.
Combines 3D structural representation with physics-based scoring signals.
"""
import json, os, sys
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder

BENCH_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(BENCH_DIR, "bench30_finetune.csv")
EMB_PATH = os.path.join(BENCH_DIR, "unimol_embeddings.npz")
OUT_PATH = os.path.join(BENCH_DIR, "unimol_full_ranking_results.json")

# ── Load PRISM4D features ────────────────────────────────────────────────
df = pd.read_csv(CSV_PATH)
print(f"CSV: {len(df)} sites, {df['pdb'].nunique()} targets")

# ── Load Uni-Mol embeddings ──────────────────────────────────────────────
emb_data = np.load(EMB_PATH, allow_pickle=True)
emb_cls = emb_data['cls_repr']       # (N, 512)
emb_targets = emb_data['targets']     # target names
emb_site_ids = emb_data['site_ids']   # site IDs
emb_dccs = emb_data['dccs']
print(f"Embeddings: {emb_cls.shape}")

# ── Align CSV rows with embedding rows ───────────────────────────────────
# Build lookup: (target, site_id) -> embedding index
emb_lookup = {}
for i, (t, s) in enumerate(zip(emb_targets, emb_site_ids)):
    emb_lookup[(str(t), str(s))] = i

matched_emb = []
matched_idx = []
for i, row in df.iterrows():
    key = (str(row['pdb']), str(row['site_id']))
    if key in emb_lookup:
        matched_emb.append(emb_cls[emb_lookup[key]])
        matched_idx.append(i)

df = df.loc[matched_idx].reset_index(drop=True)
emb_matrix = np.array(matched_emb)
print(f"Matched: {len(df)} sites with embeddings")

# ── Feature engineering ──────────────────────────────────────────────────
NUMERIC_COLS = [
    'burial_score', 'onset_score', 'sphericity', 'uv_enrichment_score',
    'breathing_score', 'source_diversity', 'wd_coherence', 'quality_score',
    'druggability', 'volume', 'spike_count', 'mean_burial',
    'hysteresis_asymmetry', 'asymmetry_offset', 'relative_asymmetry',
    'kinetic_accessibility', 'frustrated_solvent_score', 'aromatic_score',
    'ray_escape_ratio', 'ccns_tau', 'tide_coupling_score',
    'engine_geo', 'engine_vcs', 'engine_chem', 'engine_phys',
    'catalytic_residue_count', 'n_lining_residues',
]

# Encode categoricals
class_enc = LabelEncoder()
df['classification_enc'] = class_enc.fit_transform(df['classification'].fillna('Unknown'))
therm_enc = LabelEncoder()
df['therm_class_enc'] = therm_enc.fit_transform(df['therm_class'].fillna('Unknown'))

# Cross-features
df['burial_x_onset'] = df['burial_score'] * df['onset_score']
df['geo_x_vcs'] = df['engine_geo'] * df['engine_vcs']
df['chem_x_phys'] = df['engine_chem'] * df['engine_phys']
df['burial_x_sphericity'] = df['burial_score'] * df['sphericity']
df['quality_x_druggability'] = df['quality_score'] * df['druggability']
df['lining_x_burial'] = df['n_lining_residues'] * df['burial_score']
df['onset_x_breathing'] = df['onset_score'] * df['breathing_score']
df['engine_sum'] = df['engine_geo'] + df['engine_vcs'] + df['engine_chem'] + df['engine_phys']
df['engine_max'] = df[['engine_geo', 'engine_vcs', 'engine_chem', 'engine_phys']].max(axis=1)

EXTRA_COLS = [
    'classification_enc', 'therm_class_enc', 'is_druggable',
    'burial_x_onset', 'geo_x_vcs', 'chem_x_phys', 'burial_x_sphericity',
    'quality_x_druggability', 'lining_x_burial', 'onset_x_breathing',
    'engine_sum', 'engine_max',
]

X_prism = df[NUMERIC_COLS + EXTRA_COLS].values.astype(np.float32)
X_full = np.hstack([emb_matrix, X_prism])  # (N, 512 + 39)
y = df['label_8A'].values.astype(np.float32)
targets = df['pdb'].values

print(f"Feature matrix: {X_full.shape}")
print(f"Hits @8A: {int(y.sum())}/{len(y)}")

# ── LOTO with XGBoost ────────────────────────────────────────────────────
try:
    import xgboost as xgb
    USE_XGB = True
    print("\nUsing XGBoost")
except ImportError:
    from sklearn.ensemble import GradientBoostingClassifier
    USE_XGB = False
    print("\nUsing sklearn GradientBoosting (install xgboost for better results)")

unique_targets = sorted(set(targets))
sr1_hits = 0
sr1_total = 0
all_results = []

# Also track SR@1 for different DCC thresholds
sr1_4a = 0
sr1_8a = 0

print(f"\nLOTO Cross-Validation ({len(unique_targets)} targets)")
print(f"{'Target':<12} {'TopSite':<10} {'DCC':<8} {'Score':<8} {'@4A':<5} {'@8A':<5} {'Verdict'}")
print("-" * 65)

for held_out in unique_targets:
    train_mask = targets != held_out
    test_mask = targets == held_out

    if train_mask.sum() == 0 or test_mask.sum() == 0:
        continue

    X_train, y_train = X_full[train_mask], y[train_mask]
    X_test = X_full[test_mask]

    # Scale
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    if USE_XGB:
        scale_pos = max(1, (y_train == 0).sum() / max((y_train == 1).sum(), 1))
        model = xgb.XGBClassifier(
            n_estimators=200,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.6,
            scale_pos_weight=scale_pos,
            eval_metric='logloss',
            use_label_encoder=False,
            verbosity=0,
            random_state=42,
        )
        model.fit(X_train_s, y_train)
        scores = model.predict_proba(X_test_s)[:, 1]
    else:
        model = GradientBoostingClassifier(
            n_estimators=200, max_depth=4, learning_rate=0.05,
            subsample=0.8, random_state=42,
        )
        model.fit(X_train_s, y_train)
        scores = model.predict_proba(X_test_s)[:, 1]

    test_df = df[test_mask].copy()
    test_df['score'] = scores
    test_df = test_df.sort_values('score', ascending=False)

    top = test_df.iloc[0]
    top_dcc = top['dcc']
    is_4a = top_dcc <= 4.0
    is_8a = top_dcc <= 8.0

    if is_4a:
        sr1_4a += 1
    if is_8a:
        sr1_8a += 1
    sr1_total += 1

    verdict = "HIT4A <<<" if is_4a else ("HIT8A <<<" if is_8a else "MISS")
    print(f"{held_out:<12} {int(top['site_id']):<10} {top_dcc:<8.2f} {top['score']:<8.4f} "
          f"{int(is_4a):<5} {int(is_8a):<5} {verdict}")

    all_results.append({
        'target': held_out,
        'top_site': int(top['site_id']),
        'top_dcc': round(top_dcc, 2),
        'top_score': round(float(top['score']), 4),
        'hit_4a': bool(is_4a),
        'hit_8a': bool(is_8a),
    })

print(f"\n{'='*65}")
print(f"SR@1 @4A:  {sr1_4a}/{sr1_total} ({100*sr1_4a/max(sr1_total,1):.0f}%)")
print(f"SR@1 @8A:  {sr1_8a}/{sr1_total} ({100*sr1_8a/max(sr1_total,1):.0f}%)")

# ── Feature importance (train on all data) ───────────────────────────────
if USE_XGB:
    print(f"\n{'='*65}")
    print("TOP FEATURE IMPORTANCES (full model)")
    print(f"{'='*65}")

    scaler_full = StandardScaler()
    X_all_s = scaler_full.fit_transform(X_full)
    scale_pos = max(1, (y == 0).sum() / max((y == 1).sum(), 1))
    full_model = xgb.XGBClassifier(
        n_estimators=200, max_depth=4, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.6,
        scale_pos_weight=scale_pos, eval_metric='logloss',
        use_label_encoder=False, verbosity=0, random_state=42,
    )
    full_model.fit(X_all_s, y)

    feature_names = [f"unimol_{i}" for i in range(emb_matrix.shape[1])] + NUMERIC_COLS + EXTRA_COLS
    importances = full_model.feature_importances_
    top_idx = np.argsort(importances)[::-1][:25]
    for rank, idx in enumerate(top_idx):
        print(f"  {rank+1:>2}. {feature_names[idx]:<30} {importances[idx]:.4f}")

# Save
json.dump({
    "sr1_4a": sr1_4a,
    "sr1_8a": sr1_8a,
    "n_targets": sr1_total,
    "sr1_4a_pct": round(100 * sr1_4a / max(sr1_total, 1), 1),
    "sr1_8a_pct": round(100 * sr1_8a / max(sr1_total, 1), 1),
    "n_pockets": len(df),
    "feature_dim": X_full.shape[1],
    "results": all_results,
}, open(OUT_PATH, "w"), indent=2)
print(f"\nSaved {OUT_PATH}")
