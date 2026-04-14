#!/usr/bin/env python3
"""Temporal Tokenizer — production baseline (KMeans vocabulary).

Per-site spike events are split into K temporal windows; each window is
described by an 18-dim feature vector; the full corpus is clustered into
a vocabulary of V tokens. Each site becomes a sequence of K tokens.

Window feature design (18 dims, order fixed — downstream SpikeBERT /
SiteVQVAE assume this layout):

    spike_count                                (1)
    mean_intensity, peak_intensity, std_intensity  (3)
    burst_count  (n_nearby_excited ≥ 3)        (1)
    isi_mean, isi_std                          (2)
    channel fraction: uv, lif, efp             (3)
    phase fraction: cold_hold, heating, warm_hold, cooling, cold_return (5)
    mean_n_nearby_excited                      (1)
    mean_vibrational_energy                    (1)
    mean_water_density                         (1)
                                            ── 18 total

Modes
-----
Single-target:
    --input-root output/4obe_fixed

Corpus (auto-detected when --input-root contains target subdirs):
    --input-root /mnt/storage/prism-outputs/runs/cryptobench199

Outputs (in --output-dir, default derived from --input-root):
    temporal_vocab.npz         centroids [V,18] + z-score stats
    per_window_features.npz    [n_sites, K, 18]  raw features
    tokenized_sites.json       {site_id: [tok_0 ... tok_{K-1}]}
    corpus_stats.json          (corpus mode only) — aggregate diagnostics
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

try:
    import orjson
    HAVE_ORJSON = True
except ImportError:
    HAVE_ORJSON = False

PHASES = ["cold_hold", "heating", "warm_hold", "cooling", "cold_return"]
SOURCES = ["UV", "LIF", "EFP"]
BURST_THRESHOLD = 3

N_PER_WINDOW_FEATURES = 18   # see docstring for order


# ─────────────────────────────────────────────────────────────
#  Stream spike events
# ─────────────────────────────────────────────────────────────

def load_spikes(path: Path, sample_cap: int = 0) -> Dict[str, np.ndarray]:
    """Extract only the fields needed for temporal tokenization.

    sample_cap > 0 → uniform subsample for speed (used only in proto tests).
    """
    t0 = time.time()
    if HAVE_ORJSON:
        with open(path, "rb") as f:
            data = orjson.loads(f.read())
    else:
        with open(path) as f:
            data = json.load(f)
    spikes = data.get("spikes", []) or []
    if not spikes:
        return {"n": 0, "site_id": data.get("site_id")}

    n = len(spikes)
    ts = np.empty(n, dtype=np.int32)
    inten = np.empty(n, dtype=np.float32)
    src = np.empty(n, dtype=np.uint8)
    ph = np.empty(n, dtype=np.uint8)
    nex = np.empty(n, dtype=np.int16)
    vib = np.empty(n, dtype=np.float32)
    wd = np.empty(n, dtype=np.float32)
    src_map = {"UV": 0, "LIF": 1, "EFP": 2}
    ph_map = {p: i for i, p in enumerate(PHASES)}

    for i, sp in enumerate(spikes):
        ts[i] = sp["timestep"]
        inten[i] = sp["intensity"]
        src[i] = src_map.get(sp.get("spike_source", ""), 255)
        ph[i] = ph_map.get(sp.get("ccns_phase", ""), 255)
        nex[i] = sp.get("n_nearby_excited", 0)
        vib[i] = sp.get("vibrational_energy", 0.0)
        wd[i] = sp.get("water_density", 0.0)

    if sample_cap and n > sample_cap:
        rng = np.random.default_rng(42)
        idx = rng.choice(n, size=sample_cap, replace=False)
        idx.sort()
        ts, inten, src, ph, nex, vib, wd = (
            ts[idx], inten[idx], src[idx], ph[idx], nex[idx], vib[idx], wd[idx])
        n = sample_cap

    elapsed = time.time() - t0
    return {"n": n, "timestep": ts, "intensity": inten, "source": src,
            "phase": ph, "n_nearby_excited": nex, "vib": vib, "wd": wd,
            "site_id": data.get("site_id"), "elapsed": elapsed,
            "file_size_mb": path.stat().st_size / 1e6}


# ─────────────────────────────────────────────────────────────
#  Window features (18 dims / window / site)
# ─────────────────────────────────────────────────────────────

def window_features(arr: Dict[str, np.ndarray], n_windows: int = 32
                    ) -> Tuple[np.ndarray, int, int]:
    """Returns (feat [K,18], ts_min, ts_max)."""
    if arr.get("n", 0) == 0:
        return (np.zeros((n_windows, N_PER_WINDOW_FEATURES), dtype=np.float32),
                0, 0)
    ts = arr["timestep"]
    ts_min = int(ts.min()); ts_max = int(ts.max())
    if ts_max <= ts_min:
        ts_max = ts_min + 1
    edges = np.linspace(ts_min, ts_max + 1, n_windows + 1)
    bins = np.clip(np.digitize(ts, edges) - 1, 0, n_windows - 1)

    feats = np.zeros((n_windows, N_PER_WINDOW_FEATURES), dtype=np.float32)
    inten, src, ph, nex, vib, wd = (arr["intensity"], arr["source"],
                                      arr["phase"], arr["n_nearby_excited"],
                                      arr["vib"], arr["wd"])
    for w in range(n_windows):
        mask = bins == w
        n = int(mask.sum())
        if n == 0:
            continue
        wi = inten[mask]; wts = ts[mask]
        i = 0
        feats[w, i] = float(n); i += 1
        feats[w, i] = float(wi.mean()); i += 1
        feats[w, i] = float(wi.max()); i += 1
        feats[w, i] = float(wi.std()) if n > 1 else 0.0; i += 1
        feats[w, i] = float((nex[mask] >= BURST_THRESHOLD).sum()); i += 1
        if n > 1:
            isi = np.diff(np.sort(wts))
            feats[w, i] = float(isi.mean()); i += 1
            feats[w, i] = float(isi.std()); i += 1
        else:
            feats[w, i] = 0.0; i += 1
            feats[w, i] = 0.0; i += 1
        feats[w, i] = float((src[mask] == 0).sum()) / n; i += 1
        feats[w, i] = float((src[mask] == 1).sum()) / n; i += 1
        feats[w, i] = float((src[mask] == 2).sum()) / n; i += 1
        for phase_id in range(5):
            feats[w, i] = float((ph[mask] == phase_id).sum()) / n; i += 1
        feats[w, i] = float(nex[mask].mean()); i += 1
        feats[w, i] = float(vib[mask].mean()); i += 1
        feats[w, i] = float(wd[mask].mean()); i += 1
        assert i == N_PER_WINDOW_FEATURES
    return feats, ts_min, ts_max


# ─────────────────────────────────────────────────────────────
#  Discovery — single target vs corpus
# ─────────────────────────────────────────────────────────────

def discover_sites(input_root: Path) -> List[Tuple[str, Path]]:
    """Return list of (target_name, site_json_path).

    - If input_root directly contains *.site*.spike_events.json → single target
      (target_name derived from first matching filename prefix).
    - Otherwise input_root is treated as a corpus: every subdirectory with
      matching *.site*.spike_events.json is a target.
    """
    direct = sorted(input_root.glob("*.site*.spike_events.json"))
    if direct:
        # Use basename prefix as target name (strip trailing .siteN.*)
        stem = direct[0].name.split(".site")[0]
        return [(stem, p) for p in direct]
    out: List[Tuple[str, Path]] = []
    for sub in sorted(input_root.iterdir()):
        if not sub.is_dir():
            continue
        sites = sorted(sub.glob("*.site*.spike_events.json"))
        if not sites:
            continue
        target = sub.name
        out.extend((target, p) for p in sites)
    return out


# ─────────────────────────────────────────────────────────────
#  Corpus stats
# ─────────────────────────────────────────────────────────────

def corpus_stats(features: np.ndarray, tokens: np.ndarray,
                 vocab_size: int, empty_token: int,
                 target_by_site: List[str]) -> Dict:
    """features [S,K,F], tokens [S,K]."""
    S, K, F = features.shape
    nonzero = (features != 0).any(axis=-1)                     # [S,K]
    empty_frac = 1.0 - float(nonzero.mean())

    # Token usage across the whole corpus
    flat = tokens.ravel()
    counter = Counter(int(t) for t in flat)
    histogram = {int(t): int(counter.get(t, 0)) for t in range(vocab_size)}
    dead = [t for t in range(vocab_size) if counter.get(t, 0) == 0]
    # Entropy on non-empty tokens
    non_empty = np.array([counter.get(t, 0) for t in range(vocab_size)], dtype=np.float64)
    p = non_empty / max(non_empty.sum(), 1)
    p = p[p > 0]
    entropy = float(-(p * np.log2(p)).sum()) if p.size else 0.0

    # Per-target pooled histogram
    per_target_hist: Dict[str, Dict[int, int]] = {}
    targets_unique = sorted(set(target_by_site))
    for t in targets_unique:
        mask = [x == t for x in target_by_site]
        sub = tokens[np.asarray(mask)]
        per_target_hist[t] = {int(k): int(v) for k, v in Counter(sub.ravel()).items()}

    # Nearest-neighbor site similarity (Jaccard of token sets, top-5 pairs)
    site_token_sets = [set(tokens[i].tolist()) for i in range(S)]
    pairs = []
    # For corpus size can be very large, limit to ~1000 sites for quick diagnostics
    sample_idx = list(range(S))[:1000]
    for i in sample_idx:
        best_j = -1; best_j_jac = -1.0
        for j in sample_idx:
            if i == j: continue
            A = site_token_sets[i]; B = site_token_sets[j]
            u = A | B
            jac = len(A & B) / max(len(u), 1)
            if jac > best_j_jac:
                best_j_jac = jac; best_j = j
        if best_j >= 0:
            pairs.append((i, best_j, best_j_jac))
    pairs.sort(key=lambda x: -x[2])
    top_pairs = pairs[:5]

    return {
        "n_sites": int(S),
        "n_windows_per_site": int(K),
        "feature_dim": int(F),
        "vocab_size": int(vocab_size),
        "empty_token_id": int(empty_token),
        "empty_window_frac": empty_frac,
        "total_tokens": int(flat.size),
        "unique_tokens_used": int(len(counter)),
        "dead_tokens": dead,
        "n_dead_tokens": int(len(dead)),
        "cluster_histogram": histogram,
        "token_entropy_bits": entropy,
        "max_entropy_bits": float(np.log2(vocab_size)) if vocab_size > 1 else 0.0,
        "per_target_histogram_sample": {
            k: per_target_hist[k] for k in list(per_target_hist.keys())[:10]
        },
        "n_unique_targets": int(len(targets_unique)),
        "top5_nearest_neighbor_jaccard": [
            {"i": int(i), "j": int(j), "jaccard": float(jac)}
            for i, j, jac in top_pairs
        ],
    }


# ─────────────────────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", type=Path, required=True,
                        help="Either a single target directory OR a corpus "
                             "root containing per-target subdirs")
    parser.add_argument("--output-dir", type=Path, default=None,
                        help="Default: <input-root>/temporal_tokens")
    parser.add_argument("--window-count", type=int, default=32)
    parser.add_argument("--vocab-size", type=int, default=64,
                        help="KMeans K (baseline). Production VQ-VAE tokenizer "
                             "will supersede this — see site_vqvae.py.")
    parser.add_argument("--max-sites", type=int, default=0,
                        help="Cap sites (0=all)")
    parser.add_argument("--sample-spikes", type=int, default=0,
                        help="Uniform subsample per site for speed testing (0=off)")
    parser.add_argument("--skip-clustering", action="store_true",
                        help="Only extract per-window features; skip KMeans "
                             "(useful when features will feed a separate "
                             "learned tokenizer like SiteVQVAE).")
    args = parser.parse_args()

    out_dir = args.output_dir or (args.input_root / "temporal_tokens")
    out_dir.mkdir(parents=True, exist_ok=True)

    sites = discover_sites(args.input_root)
    if not sites:
        print(f"ERROR: no site JSON under {args.input_root}"); sys.exit(1)
    if args.max_sites:
        sites = sites[:args.max_sites]
    target_names = sorted({t for t, _ in sites})
    corpus_mode = len(target_names) > 1

    print(f"Input: {args.input_root}  mode={'CORPUS' if corpus_mode else 'SINGLE'}")
    print(f"Sites: {len(sites)}  targets: {len(target_names)}  "
          f"windows: {args.window_count}  vocab: {args.vocab_size}")

    # Extract features
    feats_list: List[np.ndarray] = []
    site_ids: List[str] = []
    site_targets: List[str] = []
    site_tsmin: List[int] = []; site_tsmax: List[int] = []
    for ti, (target, sf) in enumerate(sites):
        arr = load_spikes(sf, sample_cap=args.sample_spikes)
        print(f"    [{ti+1}/{len(sites)}] {target}/{sf.name}: "
              f"{arr.get('n', 0):,} spikes ({arr.get('file_size_mb', 0):.0f} MB) "
              f"{arr.get('elapsed', 0):.1f}s", flush=True)
        f, tmin, tmax = window_features(arr, args.window_count)
        feats_list.append(f)
        site_ids.append(f"{target}/{arr.get('site_id', sf.stem)}")
        site_targets.append(target)
        site_tsmin.append(tmin); site_tsmax.append(tmax)

    X = np.stack(feats_list, axis=0)          # [S, K, F]
    print(f"\n  feature tensor: {X.shape}  ({X.size * 4 / 1e6:.1f} MB)")

    np.savez_compressed(out_dir / "per_window_features.npz",
                         features=X,
                         site_ids=np.asarray(site_ids),
                         target_names=np.asarray(site_targets),
                         ts_min=np.asarray(site_tsmin, dtype=np.int64),
                         ts_max=np.asarray(site_tsmax, dtype=np.int64))
    print(f"  saved per-window features: {out_dir / 'per_window_features.npz'}")

    if args.skip_clustering:
        print("  --skip-clustering requested; done.")
        return

    # KMeans vocabulary
    X_flat = X.reshape(-1, N_PER_WINDOW_FEATURES)
    nonzero_mask = X_flat.any(axis=1)
    X_fit = X_flat[nonzero_mask]

    mean = X_fit.mean(axis=0).astype(np.float32)
    std = X_fit.std(axis=0).astype(np.float32)
    std[std < 1e-6] = 1.0
    X_norm = (X_fit - mean) / std

    from sklearn.cluster import MiniBatchKMeans
    k = min(args.vocab_size, X_fit.shape[0])
    print(f"  fitting MiniBatchKMeans K={k} on {X_fit.shape[0]} non-empty rows...")
    km = MiniBatchKMeans(n_clusters=k, random_state=42, n_init=10,
                          batch_size=512, max_iter=300)
    km.fit(X_norm)

    X_all_norm = (X_flat - mean) / std
    token_ids = km.predict(X_all_norm).reshape(X.shape[0], args.window_count)
    empty_mask = (~nonzero_mask).reshape(X.shape[0], args.window_count)
    empty_token = k
    token_ids = np.where(empty_mask, empty_token, token_ids)

    # Save vocabulary
    np.savez_compressed(out_dir / "temporal_vocab.npz",
                         centroids=km.cluster_centers_.astype(np.float32),
                         mean=mean, std=std,
                         vocab_size=k, empty_token_id=empty_token,
                         n_windows=args.window_count,
                         n_per_window_features=N_PER_WINDOW_FEATURES)
    # Save tokens
    tokenized = {site_ids[i]: token_ids[i].tolist() for i in range(len(site_ids))}
    (out_dir / "tokenized_sites.json").write_text(json.dumps(tokenized, indent=2))
    print(f"  saved vocabulary: {out_dir / 'temporal_vocab.npz'}")
    print(f"  saved tokenized sites: {out_dir / 'tokenized_sites.json'}")

    # Corpus stats (always — single-target mode gets degenerate but still useful)
    stats = corpus_stats(X, token_ids, vocab_size=k, empty_token=empty_token,
                          target_by_site=site_targets)
    stats["kmeans_inertia"] = float(km.inertia_)
    (out_dir / "corpus_stats.json").write_text(json.dumps(stats, indent=2, default=str))
    print(f"  saved corpus stats: {out_dir / 'corpus_stats.json'}")

    # Terse summary
    print(f"\n{'='*64}\nSUMMARY\n{'='*64}")
    print(f"  mode:                 {'CORPUS' if corpus_mode else 'SINGLE'}")
    print(f"  targets:              {stats['n_unique_targets']}")
    print(f"  sites:                {stats['n_sites']}")
    print(f"  total tokens:         {stats['total_tokens']:,}")
    print(f"  unique tokens used:   {stats['unique_tokens_used']} / {k+1}")
    print(f"  dead tokens:          {stats['n_dead_tokens']}")
    print(f"  empty-window frac:    {stats['empty_window_frac']:.2%}")
    print(f"  token entropy:        {stats['token_entropy_bits']:.3f} bits "
          f"(max {stats['max_entropy_bits']:.3f})")
    print(f"  KMeans inertia:       {km.inertia_:.1f}")
    print(f"  top-5 NN jaccard:     "
          f"{[(p['i'], p['j'], round(p['jaccard'], 2)) for p in stats['top5_nearest_neighbor_jaccard']]}")


if __name__ == "__main__":
    main()
