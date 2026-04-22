#!/usr/bin/env python3
"""XGBoost Ranker v3 — graded labels + v4's exact feature pipeline.

Matches the v4 tokenized ranker's training setup precisely, then adds
XGBoost on top instead of a discrete 256-bin lookup.

Differences from XGBoost v1/v2:
  v1 (6.34% SR@1): 4 features, binary labels, XGBClassifier
  v2 (12.69% SR@1): 13 features from binding_sites.json, binary labels, XGBRanker pairwise
  v3 (target ≥40%):
      - Per-site min_dist computed from spike parquets (x,y,z → ligand centroid)
      - Graded label: 1 / (1 + min_dist)
      - v4's 4 core features (spike_count, n_streams, interaction, unsat_frac)
      - Plus continuous (spread, burial_score, spike_density) from binding_sites.json
      - Plus persistence (from v4's own extraction)
      - XGBRanker rank:ndcg (handles graded labels natively)
      - LOTO on valid targets with n_parquet_sites > 0 (matches v4's 302 targets)

Data flow:
  R2 parquets + ground_truth.json → per-site min_dist + unsat_frac + persistence
  + binding_sites.json cache         → spread, burial_score, spike_density
  → graded labels
  → XGBoost LOTO

Output:
  /mnt/storage/spike-audit/ranker-xgb-v3/{model.json, evaluation.json,
                                          features.json, model.onnx,
                                          feature_importance.json}
"""
from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pyarrow.parquet as pq
import xgboost as xgb

# ── Config ──
DCC_PATH = Path("/mnt/storage/spike-audit/dcc-recompute/corrected_dcc_results.json")
BS_CACHE = Path("/tmp/spike_count_audit")            # binding_sites.json cache
FEAT_CACHE = Path("/mnt/storage/spike-audit/ranker-xgb-v3/features")
OUT_DIR = Path("/mnt/storage/spike-audit/ranker-xgb-v3")
R2_PREFIX = "r2:prism-archive/10k-runs"

MAX_WORKERS = 16
DOWNLOAD_TIMEOUT_S = 240
EXTRACT_TIMEOUT_S = 300

FEATURE_COLS = [
    "spike_count",            # raw, not log — XGBoost handles it
    "n_streams",
    "interaction",            # spike_count * n_streams (v4's d2)
    "unsat_frac",             # v4's d3
    "persistence",            # v4 secondary feature
    "log_spike_count",        # transformed copy
    "log_interaction",
    "spread",                 # binding_sites.json
    "burial_score",           # binding_sites.json
    "spike_density",          # binding_sites.json (spike_count / volume)
    "druggability",           # binding_sites.json
    "aromatic_score",         # binding_sites.json
    "n_lining_residues",      # binding_sites.json
]


# ─────────────────────────────────────────────────────────────
#  Per-target feature extraction (from parquets + cached binding_sites)
# ─────────────────────────────────────────────────────────────

def extract_target(target: str) -> Optional[Dict[str, Any]]:
    """Extract per-site features for one target. Mirrors v4's extract step
    plus joins with cached binding_sites.json.
    Returns {"target", "sites": [...]} or None on failure.
    """
    # Cached output path
    cached = FEAT_CACHE / f"{target}.json"
    if cached.exists():
        try:
            with open(cached) as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError):
            cached.unlink(missing_ok=True)

    FEAT_CACHE.mkdir(parents=True, exist_ok=True)
    tmpdir = FEAT_CACHE / f"_tmp_{target}"
    tmpdir.mkdir(parents=True, exist_ok=True)

    try:
        # 1. Download parquets
        subprocess.run(
            ['rclone', 'copy', f'{R2_PREFIX}/{target}/', str(tmpdir),
             '--include', '*.spike_events.parquet',
             '--transfers', '4', '--quiet'],
            timeout=DOWNLOAD_TIMEOUT_S, check=False,
        )

        # 2. Download ground truth (if not cached in spike_count_audit)
        gt_path = None
        bs_gt = BS_CACHE / target / f"{target}_ground_truth.json"
        if bs_gt.exists():
            gt_path = bs_gt
        else:
            subprocess.run(
                ['rclone', 'copy', f'{R2_PREFIX}/{target}/{target}_ground_truth.json',
                 str(tmpdir), '--transfers', '1', '--quiet'],
                timeout=60, check=False,
            )
            local_gt = tmpdir / f"{target}_ground_truth.json"
            if local_gt.exists():
                gt_path = local_gt

        lig_center = None
        if gt_path:
            try:
                gt = json.load(open(gt_path))
                c = gt.get('ligand_centroid', [])
                if isinstance(c, list) and len(c) == 3:
                    lig_center = np.array(c, dtype=np.float32)
            except (json.JSONDecodeError, OSError):
                pass

        # 3. Load binding_sites.json for geometry features
        bs_path = BS_CACHE / target / f"{target}.binding_sites.json"
        bs_sites_by_id: Dict[int, Dict[str, Any]] = {}
        if bs_path.exists():
            try:
                bs = json.load(open(bs_path))
                for s in bs.get('sites', []):
                    sid = s.get('id')
                    if sid is not None:
                        bs_sites_by_id[int(sid)] = s
            except (json.JSONDecodeError, OSError):
                pass

        # 4. Compute features per site (parquet-derived + binding_sites-derived)
        out_sites: List[Dict[str, Any]] = []
        for pf in sorted(tmpdir.glob('*.spike_events.parquet')):
            site_name = pf.stem.replace('.spike_events', '')
            # Extract cluster_id from "target.siteN" suffix
            site_suffix = site_name.rsplit('.', 1)[-1]
            try:
                site_id = int(site_suffix.replace('site', ''))
            except ValueError:
                site_id = -1
            try:
                df = pq.read_table(pf).to_pandas()
            except Exception:
                continue
            n = len(df)
            if n == 0:
                continue

            spike_count = int(n)
            n_streams = int(df['stream_id'].nunique()) if 'stream_id' in df.columns else 1

            # Persistence (v4's definition)
            persistence = 0.0
            if 'timestep' in df.columns:
                ts = df['timestep'].to_numpy()
                ts_min, ts_max = ts.min(), ts.max()
                if ts_max > ts_min:
                    edges = np.linspace(ts_min, ts_max, 21)
                    persistence = float(sum(
                        1 for b in range(20)
                        if np.any((ts >= edges[b]) & (ts < edges[b + 1]))
                    ) / 20.0)

            # unsat_frac (v4's definition: fraction with intensity < 64.0)
            unsat_frac = 0.0
            if 'intensity' in df.columns:
                unsat_frac = float((df['intensity'].to_numpy() < 64.0).mean())

            # Per-site min distance to ligand (for graded label)
            min_dist = None
            if lig_center is not None and all(c in df.columns for c in ('x', 'y', 'z')):
                coords = df[['x', 'y', 'z']].to_numpy(dtype=np.float32)
                dists = np.linalg.norm(coords - lig_center, axis=1)
                min_dist = float(dists.min())

            # Merge binding_sites geometry features
            bs_site = bs_sites_by_id.get(site_id, {})
            volume = float(bs_site.get('volume', bs_site.get('volume_angstrom3', 0.0)) or 0.0)
            spread = volume ** (1 / 3) if volume > 0 else 0.0
            burial_score = float(bs_site.get('burial_score', 0.0) or 0.0)
            spike_density = spike_count / volume if volume > 0 else 0.0
            druggability = float(bs_site.get('druggability', 0.0) or 0.0)
            aromatic_score = float(bs_site.get('aromatic_score', 0.0) or 0.0)
            n_lining = len(bs_site.get('lining_residues', []))

            out_sites.append({
                'site': site_name,
                'site_id': site_id,
                'spike_count': spike_count,
                'n_streams': n_streams,
                'persistence': persistence,
                'unsat_frac': unsat_frac,
                'min_dist': min_dist,
                'spread': spread,
                'burial_score': burial_score,
                'spike_density': spike_density,
                'druggability': druggability,
                'aromatic_score': aromatic_score,
                'n_lining_residues': n_lining,
            })

        result = {'target': target, 'sites': out_sites}
        # Persist to cache so subsequent runs are fast
        with open(cached, 'w') as f:
            json.dump(result, f)
        return result

    except subprocess.TimeoutExpired:
        return None
    except Exception as e:
        print(f"  ERROR extracting {target}: {type(e).__name__}: {e}")
        return None
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def build_labeled_dataset() -> List[Dict[str, Any]]:
    """For each valid target (spike_dcc not None, n_parquet_sites > 0 per
    corrected_dcc), extract site features. Return list of target records
    with graded labels applied.
    """
    dcc_results = json.load(open(DCC_PATH))
    valid_targets = [r['target'] for r in dcc_results
                     if r.get('spike_dcc') is not None
                     and r.get('n_parquet_sites', 0) > 0]
    print(f"Valid targets per corrected_dcc: {len(valid_targets)}")

    t0 = time.time()
    extracted: List[Dict[str, Any]] = []
    completed = 0
    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futs = {pool.submit(extract_target, t): t for t in valid_targets}
        for fut in as_completed(futs):
            completed += 1
            r = fut.result()
            if r and r['sites']:
                extracted.append(r)
            if completed % 25 == 0:
                rate = completed / (time.time() - t0)
                eta = (len(valid_targets) - completed) / max(rate, 0.01) / 60.0
                print(f"  [{completed}/{len(valid_targets)}] extracted {len(extracted)} | "
                      f"{rate:.1f}/s | eta ~{eta:.0f}m")
                sys.stdout.flush()

    print(f"\nExtraction done in {(time.time()-t0)/60:.1f}m — {len(extracted)} targets with data")

    # Apply graded labels (only keep sites with min_dist)
    labeled: List[Dict[str, Any]] = []
    for td in extracted:
        sites_with_dist = [s for s in td['sites'] if s.get('min_dist') is not None]
        if not sites_with_dist:
            continue
        for s in sites_with_dist:
            s['graded_score'] = 1.0 / (1.0 + s['min_dist'])
        # Gold site = highest graded score (lowest dist)
        gold = max(sites_with_dist, key=lambda x: x['graded_score'])
        td['sites'] = sites_with_dist
        td['gold_site'] = gold['site']
        td['gold_dist'] = gold['min_dist']
        labeled.append(td)

    print(f"Labeled: {len(labeled)} targets with per-site min_dist")
    n_sites = sum(len(t['sites']) for t in labeled)
    print(f"Total sites: {n_sites}")
    n_good = sum(1 for t in labeled if t['gold_dist'] < 5.0)
    print(f"Targets with gold_dist < 5Å: {n_good}")
    return labeled


# ─────────────────────────────────────────────────────────────
#  Feature matrix build
# ─────────────────────────────────────────────────────────────

def sites_to_matrix(sites: List[Dict[str, Any]]) -> np.ndarray:
    """Build feature matrix matching FEATURE_COLS."""
    rows = []
    for s in sites:
        sc = float(s['spike_count'])
        ns = float(s['n_streams'])
        inter = sc * ns
        rows.append([
            sc,
            ns,
            inter,
            s.get('unsat_frac', 0.0),
            s.get('persistence', 0.0),
            math.log1p(sc),
            math.log1p(inter),
            s.get('spread', 0.0),
            s.get('burial_score', 0.0),
            s.get('spike_density', 0.0),
            s.get('druggability', 0.0),
            s.get('aromatic_score', 0.0),
            float(s.get('n_lining_residues', 0)),
        ])
    return np.asarray(rows, dtype=np.float32)


# ─────────────────────────────────────────────────────────────
#  LOTO evaluation
# ─────────────────────────────────────────────────────────────

def loto_evaluate(labeled: List[Dict[str, Any]], n_rounds: int = 500) -> Dict[str, Any]:
    """LOTO with XGBRanker rank:ndcg (handles graded labels natively)."""

    # Flatten all sites into parallel arrays
    all_targets: List[str] = []
    all_site_names: List[str] = []
    all_features: List[np.ndarray] = []
    all_labels: List[float] = []
    for t in labeled:
        for s in t['sites']:
            all_targets.append(t['target'])
            all_site_names.append(s['site'])
            all_labels.append(s['graded_score'])
        all_features.append(sites_to_matrix(t['sites']))
    X = np.vstack(all_features)
    y = np.asarray(all_labels, dtype=np.float32)
    targets = np.asarray(all_targets)
    site_names = np.asarray(all_site_names)

    # For graded labels, XGBoost rank:ndcg expects integer relevance scores
    # so discretize: scale [0,1] → 0..31 (5 bits)
    y_int = np.clip((y * 31).round(), 0, 31).astype(np.int32)

    unique_targets = sorted(set(all_targets))
    print(f"\nLOTO over {len(unique_targets)} targets")

    sr = {1: 0, 3: 0, 5: 0, 10: 0}
    per_target = []

    t0 = time.time()
    for i, tgt in enumerate(unique_targets):
        train_mask = targets != tgt
        test_mask = targets == tgt

        X_tr = X[train_mask]
        y_tr = y_int[train_mask]
        t_tr = targets[train_mask]

        # For XGBRanker, training data must be sorted by group
        order = np.argsort(t_tr, kind='stable')
        X_tr = X_tr[order]
        y_tr = y_tr[order]
        t_tr_sorted = t_tr[order]
        # Group sizes in order
        _, group_sizes = np.unique(t_tr_sorted, return_counts=True)
        # Ensure groups are contiguous (guaranteed by sort)

        X_te = X[test_mask]
        y_te = y[test_mask]            # continuous for SR@k evaluation
        site_te = site_names[test_mask]

        if len(X_tr) == 0 or len(X_te) == 0:
            continue

        model = xgb.XGBRanker(
            n_estimators=n_rounds,
            max_depth=6,
            learning_rate=0.05,
            objective='rank:ndcg',
            eval_metric='ndcg@1',
            tree_method='hist',
            n_jobs=4,
            verbosity=0,
            random_state=42,
        )
        model.fit(X_tr, y_tr, group=group_sizes, verbose=False)

        scores = model.predict(X_te)
        order_pred = np.argsort(-scores)

        # Gold site = max y_te (true graded score)
        gold_idx = int(np.argmax(y_te))
        gold_pos = int(np.where(order_pred == gold_idx)[0][0]) + 1

        for k in sr:
            if gold_pos <= k:
                sr[k] += 1

        per_target.append({
            'target': str(tgt),
            'n_sites': int(len(X_te)),
            'gold_site': str(site_te[gold_idx]),
            'gold_dist': float(1.0 / y_te[gold_idx] - 1.0) if y_te[gold_idx] > 0 else float('inf'),
            'gold_rank': gold_pos,
            'top_pred': str(site_te[order_pred[0]]),
        })

        if (i + 1) % 25 == 0 or (i + 1) == len(unique_targets):
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            sr1 = sr[1] / (i + 1) * 100
            print(f"  [{i+1}/{len(unique_targets)}]  SR@1={sr1:.2f}%  rate={rate:.1f}/s  "
                  f"eta={(len(unique_targets)-i-1)/max(rate,1e-3):.0f}s")
            sys.stdout.flush()

    n = len(per_target)
    return {
        'n_targets_evaluated': n,
        'sr1': sr[1],  'sr1_pct': round(sr[1] / n * 100, 2) if n else 0,
        'sr3': sr[3],  'sr3_pct': round(sr[3] / n * 100, 2) if n else 0,
        'sr5': sr[5],  'sr5_pct': round(sr[5] / n * 100, 2) if n else 0,
        'sr10': sr[10], 'sr10_pct': round(sr[10] / n * 100, 2) if n else 0,
        'per_target': per_target,
    }


def train_final_model(labeled: List[Dict[str, Any]], n_rounds: int = 500) -> xgb.XGBRanker:
    X, y, groups = [], [], []
    # Sort targets for contiguous group structure
    for t in sorted(labeled, key=lambda x: x['target']):
        sites = t['sites']
        X.append(sites_to_matrix(sites))
        y.extend([s['graded_score'] for s in sites])
        groups.append(len(sites))
    X = np.vstack(X)
    y = np.asarray(y, dtype=np.float32)
    y_int = np.clip((y * 31).round(), 0, 31).astype(np.int32)

    model = xgb.XGBRanker(
        n_estimators=n_rounds,
        max_depth=6,
        learning_rate=0.05,
        objective='rank:ndcg',
        eval_metric='ndcg@1',
        tree_method='hist',
        n_jobs=4,
        verbosity=0,
        random_state=42,
    )
    model.fit(X, y_int, group=groups, verbose=False)
    return model


def export_onnx(model: xgb.XGBRanker, out_path: Path) -> bool:
    """Export XGBoost model to ONNX via onnxmltools / skl2onnx."""
    try:
        import onnxmltools
        from onnxmltools.convert import convert_xgboost
        from onnxconverter_common import FloatTensorType
        initial_types = [('input', FloatTensorType([None, len(FEATURE_COLS)]))]
        onnx_model = convert_xgboost(model, initial_types=initial_types, target_opset=17)
        onnxmltools.utils.save_model(onnx_model, str(out_path))
        return True
    except ImportError as e:
        print(f"  ONNX export skipped (missing {e.name}). pip install onnxmltools onnxconverter-common")
        return False
    except Exception as e:
        print(f"  ONNX export failed: {type(e).__name__}: {e}")
        return False


# ─────────────────────────────────────────────────────────────
#  Driver
# ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n-rounds', type=int, default=500)
    parser.add_argument('--extract-only', action='store_true',
                        help="Just extract features, skip training")
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"{'='*70}")
    print("XGBoost Ranker v3 — graded labels + v4 features")
    print(f"{'='*70}")

    labeled = build_labeled_dataset()

    # Persist extracted features
    flat = [{'target': t['target'], 'sites': t['sites'],
             'gold_site': t.get('gold_site'), 'gold_dist': t.get('gold_dist')}
            for t in labeled]
    with open(OUT_DIR / 'features.json', 'w') as f:
        json.dump(flat, f)

    if args.extract_only:
        print("Extract-only mode — skipping training")
        return

    result = loto_evaluate(labeled, n_rounds=args.n_rounds)

    print(f"\n{'='*70}")
    print("  XGBoost Ranker v3 — LOTO Results (rank:ndcg, graded labels)")
    print(f"{'='*70}")
    print(f"  N targets evaluated: {result['n_targets_evaluated']}")
    print(f"  SR@1:  {result['sr1_pct']:.2f}%  ({result['sr1']})")
    print(f"  SR@3:  {result['sr3_pct']:.2f}%  ({result['sr3']})")
    print(f"  SR@5:  {result['sr5_pct']:.2f}%  ({result['sr5']})")
    print(f"  SR@10: {result['sr10_pct']:.2f}%  ({result['sr10']})")

    with open(OUT_DIR / 'evaluation.json', 'w') as f:
        json.dump(result, f, indent=2)

    # Comparison table
    print(f"\n{'Ranker':<30} {'SR@1':>8} {'SR@3':>8} {'SR@5':>8} {'SR@10':>8}")
    print("─" * 70)
    print(f"{'Tokenized v4 (256 bins)':<30} {'36.42%':>8} {'81.13%':>8} {'94.70%':>8} {'—':>8}")
    print(f"{'XGBoost v1 (4 feat, binary)':<30} {'6.34%':>8} {'18.28%':>8} {'32.46%':>8} {'57.84%':>8}")
    print(f"{'XGBoost v2 (13 feat, binary)':<30} {'12.69%':>8} {'27.99%':>8} {'42.54%':>8} {'69.40%':>8}")
    print(f"{'XGBoost v3 (13 feat, graded)':<30} "
          f"{result['sr1_pct']:>7.2f}% {result['sr3_pct']:>7.2f}% "
          f"{result['sr5_pct']:>7.2f}% {result['sr10_pct']:>7.2f}%")
    print()
    if result['sr1_pct'] >= 40.0:
        print("  ✓ TARGET MET (SR@1 ≥ 40%)")
    elif result['sr1_pct'] > 36.42:
        print("  ~ Beats tokenized v4 baseline but below 40% target")
    else:
        print("  ✗ Below tokenized v4 baseline")

    # Train + export final model
    print("\nTraining final model on full labeled set...")
    model = train_final_model(labeled, n_rounds=args.n_rounds)
    model.save_model(str(OUT_DIR / 'model.json'))

    booster = model.get_booster()
    fmap = {f"f{i}": name for i, name in enumerate(FEATURE_COLS)}
    importance = booster.get_score(importance_type='gain')
    sorted_imp = sorted(
        ((fmap.get(f, f), v) for f, v in importance.items()),
        key=lambda x: -x[1],
    )
    print("\n  Feature importance (gain):")
    for name, val in sorted_imp:
        print(f"    {name:<25} {val:>10.2f}")
    with open(OUT_DIR / 'feature_importance.json', 'w') as f:
        json.dump(dict(sorted_imp), f, indent=2)

    # ONNX export
    print("\nExporting ONNX model for Rust `ort` inference...")
    ok = export_onnx(model, OUT_DIR / 'model.onnx')
    if ok:
        print(f"  ✓ ONNX: {OUT_DIR / 'model.onnx'}")

    print(f"\n  Outputs in: {OUT_DIR}")


if __name__ == "__main__":
    main()
