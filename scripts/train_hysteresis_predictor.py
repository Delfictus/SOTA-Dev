#!/usr/bin/env python3
"""
PRISM-4D Hysteresis Predictor Prototype
========================================

Predicts per-site hysteresis_asymmetry (CRYPTIC character) from
structural features alone, trained on PRISM-4D spike event data.

Two models:
  Model A — full features (structure + MD-derived spike stats)
  Model B — structure-only features (no MD data)

Leave-one-target-out cross-validation across 10 targets.

Data sources:
  data/benchmark_v4/prism4d_v4_full.h5   (7 targets, 173M spikes)
  data/combined/prism4d_complete.h5       (10 targets, 270M spikes)
"""

import json
import os
import pickle
import sys
import warnings
from collections import defaultdict

import h5py
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, roc_auc_score

warnings.filterwarnings("ignore")

OUTPUT_DIR = "data/ml_prototype"

HYDROPHOBIC = {"ALA", "VAL", "ILE", "LEU", "MET", "PHE", "TRP", "PRO"}
AROMATIC = {"PHE", "TYR", "TRP", "HIS", "HID", "HIE", "HIP"}
POLAR = {"SER", "THR", "ASN", "GLN", "CYS"}
CHARGED_POS = {"ARG", "LYS", "HIS", "HID", "HIE", "HIP"}
CHARGED_NEG = {"ASP", "GLU"}

# ── STEP 1-2: Extract feature matrix ─────────────────────────────────────

def decode(val):
    if isinstance(val, (bytes, np.bytes_)):
        return val.decode("utf-8").strip()
    return str(val).strip()


def safe_float(val):
    v = float(val)
    return v if np.isfinite(v) else 0.0


def extract_lining_features(h5, target, site_id):
    """Extract residue composition features from lining residues."""
    lr_path = f"targets/{target}/binding_sites/lining_residues/site_{site_id}"
    if lr_path not in h5:
        return {
            "n_lining": 0, "n_catalytic": 0, "mean_min_dist": 0.0,
            "frac_hydrophobic": 0.0, "frac_aromatic": 0.0, "frac_polar": 0.0,
            "frac_charged_pos": 0.0, "frac_charged_neg": 0.0, "frac_glycine": 0.0,
        }

    lr = h5[lr_path]
    n = lr.shape[0]
    if n == 0:
        return {
            "n_lining": 0, "n_catalytic": 0, "mean_min_dist": 0.0,
            "frac_hydrophobic": 0.0, "frac_aromatic": 0.0, "frac_polar": 0.0,
            "frac_charged_pos": 0.0, "frac_charged_neg": 0.0, "frac_glycine": 0.0,
        }

    n_cat = sum(1 for row in lr if row["is_catalytic"])
    dists = [float(row["min_dist"]) for row in lr]
    resnames = [decode(row["resname"]) for row in lr]

    n_hydro = sum(1 for r in resnames if r in HYDROPHOBIC)
    n_arom = sum(1 for r in resnames if r in AROMATIC)
    n_polar = sum(1 for r in resnames if r in POLAR)
    n_pos = sum(1 for r in resnames if r in CHARGED_POS)
    n_neg = sum(1 for r in resnames if r in CHARGED_NEG)
    n_gly = sum(1 for r in resnames if r == "GLY")

    return {
        "n_lining": n,
        "n_catalytic": n_cat,
        "mean_min_dist": np.mean(dists) if dists else 0.0,
        "frac_hydrophobic": n_hydro / n,
        "frac_aromatic": n_arom / n,
        "frac_polar": n_polar / n,
        "frac_charged_pos": n_pos / n,
        "frac_charged_neg": n_neg / n,
        "frac_glycine": n_gly / n,
    }


def extract_spike_features(h5, target, site_id, max_sample=200000):
    """Extract spike-derived statistics for a site (MD-derived features)."""
    sp_path = f"targets/{target}/spike_events/site_{site_id}/spikes"
    if sp_path not in h5:
        return {
            "n_spikes": 0, "mean_vib_energy": 0.0, "std_vib_energy": 0.0,
            "mean_water_density": 0.0, "mean_intensity": 0.0, "std_intensity": 0.0,
            "frac_uv": 0.0, "frac_lif": 0.0, "frac_efp": 0.0,
            "temporal_span": 0, "spike_rate": 0.0,
        }

    ds = h5[sp_path]
    n = ds.shape[0]
    if n == 0:
        return {
            "n_spikes": 0, "mean_vib_energy": 0.0, "std_vib_energy": 0.0,
            "mean_water_density": 0.0, "mean_intensity": 0.0, "std_intensity": 0.0,
            "frac_uv": 0.0, "frac_lif": 0.0, "frac_efp": 0.0,
            "temporal_span": 0, "spike_rate": 0.0,
        }

    # Sample for efficiency (some sites have >1M spikes)
    sample_n = min(n, max_sample)
    idx = np.sort(np.random.choice(n, sample_n, replace=False)) if n > max_sample else slice(None)

    vib = ds["vibrational_energy"][idx]
    wd = ds["water_density"][idx]
    intensity = ds["intensity"][idx]
    sources = ds["spike_source"][idx]

    # Temporal span from full dataset (just first and last)
    ts_first = int(ds["timestep"][0])
    ts_last = int(ds["timestep"][-1])
    temporal_span = ts_last - ts_first + 1

    uv_count = np.sum(sources == b"UV")
    lif_count = np.sum(sources == b"LIF")
    efp_count = np.sum(sources == b"EFP")
    total = len(sources)

    return {
        "n_spikes": n,
        "mean_vib_energy": float(np.mean(vib)),
        "std_vib_energy": float(np.std(vib)),
        "mean_water_density": float(np.mean(wd)),
        "mean_intensity": float(np.mean(intensity)),
        "std_intensity": float(np.std(intensity)),
        "frac_uv": uv_count / total if total > 0 else 0.0,
        "frac_lif": lif_count / total if total > 0 else 0.0,
        "frac_efp": efp_count / total if total > 0 else 0.0,
        "temporal_span": temporal_span,
        "spike_rate": n / temporal_span if temporal_span > 0 else 0.0,
    }


def extract_all_features(h5_paths):
    """Extract feature matrix from one or more HDF5 files."""
    rows = []
    seen_sites = set()  # (target, site_id) to deduplicate

    for h5_path in h5_paths:
        print(f"  Reading {h5_path}...")
        with h5py.File(h5_path, "r") as h5:
            targets = list(h5["targets"].keys())
            for target in targets:
                sites_path = f"targets/{target}/binding_sites/sites"
                if sites_path not in h5:
                    continue

                sites_ds = h5[sites_path]
                for row in sites_ds:
                    site_id = int(row["id"])
                    key = (target, site_id)
                    if key in seen_sites:
                        continue
                    seen_sites.add(key)

                    # Label
                    hysteresis = safe_float(row["hysteresis_asymmetry"])
                    therm_class = decode(row["therm_class"])

                    # Structure features from binding_sites/sites
                    struct_feats = {
                        "volume": safe_float(row["volume"]),
                        "burial_score": safe_float(row["burial_score"]),
                        "mean_burial": safe_float(row["mean_burial"]),
                        "sphericity": safe_float(row["sphericity"]),
                        "druggability": safe_float(row["druggability"]),
                        "aromatic_score": safe_float(row["aromatic_score"]),
                        "ray_escape_ratio": safe_float(row["ray_escape_ratio"]),
                        "source_diversity": safe_float(row["source_diversity"]),
                        "breathing_score": safe_float(row["breathing_score"]),
                        "onset_score": safe_float(row["onset_score"]),
                        "ccns_tau": safe_float(row["ccns_tau"]),
                        "localization_score": safe_float(row["localization_score_raw"]),
                        "uv_enrichment": safe_float(row["uv_enrichment_score"]),
                        "wd_coherence": safe_float(row["wd_coherence"]),
                        "engine_geo": safe_float(row["engine_geo"]),
                        "engine_phys": safe_float(row["engine_phys"]),
                        "engine_chem": safe_float(row["engine_chem"]),
                    }

                    # Lining residue composition
                    lining_feats = extract_lining_features(h5, target, site_id)

                    # Spike-derived features (MD-only)
                    spike_feats = extract_spike_features(h5, target, site_id)

                    entry = {
                        "target": target,
                        "site_id": site_id,
                        "label": hysteresis,
                        "therm_class": therm_class,
                        **struct_feats,
                        **lining_feats,
                        **spike_feats,
                    }
                    rows.append(entry)

    return pd.DataFrame(rows)


# ── STEP 2: Build feature matrix and report ──────────────────────────────

def build_feature_matrices():
    """Build and validate feature matrices."""
    print("=" * 60)
    print("STEP 2: Building feature matrix")
    print("=" * 60)

    h5_paths = [
        "data/combined/prism4d_complete.h5",
        "data/benchmark_v4/prism4d_v4_full.h5",
    ]
    # Use complete.h5 as primary (has all 10 targets)
    # v4 as fallback for any missing data

    df = extract_all_features(h5_paths)

    # Define feature columns
    struct_cols = [
        "volume", "burial_score", "mean_burial", "sphericity", "druggability",
        "aromatic_score", "ray_escape_ratio", "source_diversity",
        "breathing_score", "onset_score", "ccns_tau", "localization_score",
        "uv_enrichment", "wd_coherence", "engine_geo", "engine_phys", "engine_chem",
        "n_lining", "n_catalytic", "mean_min_dist",
        "frac_hydrophobic", "frac_aromatic", "frac_polar",
        "frac_charged_pos", "frac_charged_neg", "frac_glycine",
    ]

    spike_cols = [
        "n_spikes", "mean_vib_energy", "std_vib_energy",
        "mean_water_density", "mean_intensity", "std_intensity",
        "frac_uv", "frac_lif", "frac_efp",
        "temporal_span", "spike_rate",
    ]

    full_cols = struct_cols + spike_cols

    print(f"\nTotal samples: {len(df)}")
    print(f"Targets: {sorted(df['target'].unique())}")
    print(f"CRYPTIC (h > 0.5): {(df['label'] > 0.5).sum()}")
    print(f"NON-CRYPTIC (h <= 0.5): {(df['label'] <= 0.5).sum()}")
    print(f"\nLabel distribution:")
    print(df["label"].describe())
    print(f"\nTherm class distribution:")
    print(df["therm_class"].value_counts())
    print(f"\nPer-target site counts:")
    print(df.groupby("target")["label"].describe().to_string())
    print()

    return df, struct_cols, spike_cols, full_cols


# ── STEP 3: Leave-one-target-out CV ──────────────────────────────────────

def train_and_evaluate(df, feature_cols, model_name, model_factory):
    """Leave-one-target-out CV for a given model and feature set."""
    targets = sorted(df["target"].unique())
    results = []

    for held_out in targets:
        train_df = df[df["target"] != held_out]
        test_df = df[df["target"] == held_out]

        X_train = train_df[feature_cols].values
        y_train = train_df["label"].values
        X_test = test_df[feature_cols].values
        y_test = test_df["label"].values

        # Train
        model = model_factory()
        model.fit(X_train, y_train)

        # Predict
        y_pred = model.predict(X_test)
        y_pred = np.clip(y_pred, 0, 1)

        # Metrics
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))

        # Pearson r (needs >1 sample and variance)
        if len(y_test) > 2 and np.std(y_test) > 0.001 and np.std(y_pred) > 0.001:
            r, _ = pearsonr(y_test, y_pred)
        else:
            r = 0.0

        # AUROC (binary: CRYPTIC = h > 0.5)
        y_bin = (y_test > 0.5).astype(int)
        if y_bin.sum() > 0 and y_bin.sum() < len(y_bin):
            auroc = roc_auc_score(y_bin, y_pred)
        else:
            auroc = float("nan")

        results.append({
            "held_out": held_out,
            "n_test": len(y_test),
            "n_cryptic_test": int(y_bin.sum()),
            "mae": mae,
            "rmse": rmse,
            "pearson_r": r,
            "auroc": auroc,
            "y_test": y_test.tolist(),
            "y_pred": y_pred.tolist(),
        })

    return results, model  # return last model for feature importance


def run_cv(df, struct_cols, spike_cols, full_cols):
    """Run all model combinations."""
    print("=" * 60)
    print("STEP 3: Leave-one-target-out cross-validation")
    print("=" * 60)

    all_results = {}

    configs = [
        ("A_GBR", "Full (MD+struct)", full_cols,
         lambda: GradientBoostingRegressor(n_estimators=200, max_depth=4,
                                           learning_rate=0.05, random_state=42)),
        ("A_RF", "Full (MD+struct)", full_cols,
         lambda: RandomForestRegressor(n_estimators=200, random_state=42)),
        ("B_GBR", "Struct-only", struct_cols,
         lambda: GradientBoostingRegressor(n_estimators=200, max_depth=4,
                                           learning_rate=0.05, random_state=42)),
        ("B_RF", "Struct-only", struct_cols,
         lambda: RandomForestRegressor(n_estimators=200, random_state=42)),
    ]

    models = {}
    for name, desc, cols, factory in configs:
        print(f"\n--- {name}: {desc} ({len(cols)} features) ---")
        results, last_model = train_and_evaluate(df, cols, name, factory)
        models[name] = last_model

        # Aggregate
        maes = [r["mae"] for r in results]
        rmses = [r["rmse"] for r in results]
        rs = [r["pearson_r"] for r in results]
        aurocs = [r["auroc"] for r in results if not np.isnan(r["auroc"])]

        print(f"  MAE:  {np.mean(maes):.3f} +/- {np.std(maes):.3f}")
        print(f"  RMSE: {np.mean(rmses):.3f} +/- {np.std(rmses):.3f}")
        print(f"  Pearson r: {np.mean(rs):.3f} +/- {np.std(rs):.3f}")
        if aurocs:
            print(f"  AUROC: {np.mean(aurocs):.3f} +/- {np.std(aurocs):.3f}  (N={len(aurocs)} folds with both classes)")
        else:
            print(f"  AUROC: N/A (no folds with both classes)")

        # Per-fold detail
        for r in results:
            auroc_str = f"{r['auroc']:.3f}" if not np.isnan(r["auroc"]) else "N/A"
            print(f"    {r['held_out']}: MAE={r['mae']:.3f} RMSE={r['rmse']:.3f} "
                  f"r={r['pearson_r']:.3f} AUROC={auroc_str} "
                  f"(n={r['n_test']}, crypt={r['n_cryptic_test']})")

        all_results[name] = {
            "description": desc,
            "n_features": len(cols),
            "features": cols,
            "mean_mae": float(np.mean(maes)),
            "std_mae": float(np.std(maes)),
            "mean_rmse": float(np.mean(rmses)),
            "std_rmse": float(np.std(rmses)),
            "mean_pearson_r": float(np.mean(rs)),
            "std_pearson_r": float(np.std(rs)),
            "mean_auroc": float(np.mean(aurocs)) if aurocs else None,
            "std_auroc": float(np.std(aurocs)) if aurocs else None,
            "n_auroc_folds": len(aurocs),
            "folds": [{k: v for k, v in r.items() if k not in ("y_test", "y_pred")} for r in results],
        }

    return all_results, models


# ── STEP 4: Feature importance ───────────────────────────────────────────

def feature_importance_analysis(models, struct_cols):
    """Print and save feature importances for Model B."""
    print("\n" + "=" * 60)
    print("STEP 4: Feature Importance (Model B — structure-only)")
    print("=" * 60)

    # Use GBR as primary (generally more informative importances)
    model = models.get("B_GBR")
    if model is None:
        print("  No B_GBR model found")
        return None

    importances = model.feature_importances_
    fi = pd.DataFrame({
        "feature": struct_cols,
        "importance": importances,
    }).sort_values("importance", ascending=False)

    print("\nFeature importances (GradientBoosting, structure-only):")
    for _, row in fi.iterrows():
        bar = "█" * int(row["importance"] * 200)
        print(f"  {row['feature']:25s} {row['importance']:.4f}  {bar}")

    # Also do RF
    rf_model = models.get("B_RF")
    if rf_model is not None:
        rf_imp = rf_model.feature_importances_
        fi["importance_rf"] = rf_imp
        print("\nRandom Forest importances:")
        fi_rf = fi.sort_values("importance_rf", ascending=False)
        for _, row in fi_rf.iterrows():
            bar = "█" * int(row["importance_rf"] * 200)
            print(f"  {row['feature']:25s} {row['importance_rf']:.4f}  {bar}")

    return fi


# ── STEP 5: Save outputs ─────────────────────────────────────────────────

def save_outputs(df, struct_cols, spike_cols, full_cols, all_results, fi, models):
    """Save all outputs."""
    print("\n" + "=" * 60)
    print("STEP 5: Saving outputs")
    print("=" * 60)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Feature matrices
    meta_cols = ["target", "site_id", "label", "therm_class"]
    df[meta_cols + full_cols].to_csv(f"{OUTPUT_DIR}/feature_matrix_full.csv", index=False)
    df[meta_cols + struct_cols].to_csv(f"{OUTPUT_DIR}/feature_matrix_struct.csv", index=False)
    print(f"  Saved feature_matrix_full.csv ({len(full_cols)} features)")
    print(f"  Saved feature_matrix_struct.csv ({len(struct_cols)} features)")

    # Best model B
    best_b = models.get("B_GBR")
    if best_b:
        with open(f"{OUTPUT_DIR}/model_b_best.pkl", "wb") as f:
            pickle.dump(best_b, f)
        print("  Saved model_b_best.pkl")

    # CV results
    with open(f"{OUTPUT_DIR}/cv_results.json", "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print("  Saved cv_results.json")

    # Feature importances
    if fi is not None:
        fi.to_csv(f"{OUTPUT_DIR}/feature_importances.csv", index=False)
        print("  Saved feature_importances.csv")


# ── STEP 6: Honest assessment ────────────────────────────────────────────

def honest_assessment(df, all_results):
    """Print honest assessment."""
    print("\n" + "=" * 60)
    print("STEP 6: PROTOTYPE ASSESSMENT")
    print("=" * 60)

    n_total = len(df)
    n_targets = df["target"].nunique()
    n_cryptic = (df["label"] > 0.5).sum()

    a_gbr = all_results.get("A_GBR", {})
    b_gbr = all_results.get("B_GBR", {})

    a_auroc = a_gbr.get("mean_auroc")
    a_auroc_std = a_gbr.get("std_auroc", 0)
    b_auroc = b_gbr.get("mean_auroc")
    b_auroc_std = b_gbr.get("std_auroc", 0)
    b_mae = b_gbr.get("mean_mae", 0)
    b_mae_std = b_gbr.get("std_mae", 0)

    print(f"N samples: {n_total} sites across {n_targets} targets")
    print(f"N CRYPTIC (h > 0.5): {n_cryptic} ({100*n_cryptic/n_total:.1f}%)")
    print()

    if a_auroc is not None:
        print(f"Model A (MD+structure) GBR AUROC: {a_auroc:.3f} +/- {a_auroc_std:.3f}")
    else:
        print("Model A (MD+structure) GBR AUROC: N/A")

    if b_auroc is not None:
        print(f"Model B (structure only) GBR AUROC: {b_auroc:.3f} +/- {b_auroc_std:.3f}")
    else:
        print("Model B (structure only) GBR AUROC: N/A")

    print(f"Model B MAE on hysteresis: {b_mae:.3f} +/- {b_mae_std:.3f}")
    print()

    # Honest caveats
    if n_total < 50:
        print("WARNING: N < 50 sites total — results are indicative only,")
        print("  not statistically robust. Need more targets to validate.")
    elif n_total < 100:
        print("CAUTION: N < 100 — moderate confidence. More targets needed.")

    if n_cryptic < 10:
        print(f"WARNING: Only {n_cryptic} CRYPTIC samples — severe class imbalance.")
        print("  AUROC estimates are unstable with so few positives.")

    if b_auroc is not None:
        if b_auroc < 0.65:
            print("Model B performs near chance — structure alone may be")
            print("  insufficient; MD features capture information not in structure.")
        elif b_auroc >= 0.65 and b_auroc < 0.85:
            print("Model B shows moderate signal in structural features.")
            print("  Feasibility is promising but needs more data to confirm.")
        elif b_auroc >= 0.85:
            print("Model B AUROC > 0.85 — strong signal in structural features.")
            print("  This supports feasibility of a fast structure-based predictor.")

    print()
    print("KEY INSIGHT: The gap between Model A and Model B quantifies")
    print("how much information the MD simulation adds beyond structure.")
    if a_auroc is not None and b_auroc is not None:
        gap = a_auroc - b_auroc
        if gap < 0.05:
            print(f"  Gap = {gap:.3f} — MD adds minimal information. Structure alone")
            print("  captures most of the cryptic signal. Fast predictor is viable.")
        elif gap < 0.15:
            print(f"  Gap = {gap:.3f} — MD adds moderate information. Structure")
            print("  captures partial signal. Hybrid approach recommended.")
        else:
            print(f"  Gap = {gap:.3f} — MD adds substantial information. Structure")
            print("  alone is insufficient. MD simulation remains essential.")


# ── Main ─────────────────────────────────────────────────────────────────

def main():
    np.random.seed(42)

    print("PRISM-4D Hysteresis Predictor Prototype")
    print("=" * 60)

    # Step 2: Build features
    df, struct_cols, spike_cols, full_cols = build_feature_matrices()

    # Step 3: Train and evaluate
    all_results, models = run_cv(df, struct_cols, spike_cols, full_cols)

    # Step 4: Feature importance
    fi = feature_importance_analysis(models, struct_cols)

    # Step 5: Save
    save_outputs(df, struct_cols, spike_cols, full_cols, all_results, fi, models)

    # Step 6: Assessment
    honest_assessment(df, all_results)


if __name__ == "__main__":
    main()
