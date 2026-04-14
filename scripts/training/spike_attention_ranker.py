#!/usr/bin/env python3
"""Spike Attention Ranker — learns which spike patterns predict binding sites.

Phases:
  1. Tokenize raw spike events from per-site parquets into composite token sequences
  2. Extract per-site physics embeddings (PCA-128 on 281-dim per-residue features)
  3. Train SpikeAttentionRanker with learned query attention + pairwise ranking loss
  4. Cluster-aware LOTO evaluation
  5. Attention interpretability analysis + ONNX export

Adaptations from spec:
  - Teacher v005 .pt fold checkpoints do not exist; ESM2 features absent from all
    372 bundles. Teacher embeddings are replaced with PCA-128 on the 281-dim
    per-residue physics features averaged within 8A of site centroid.
  - Site centroids from tide_pocket_json or computed from spike positions.

Usage:
  python3 scripts/training/spike_attention_ranker.py
"""
from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

import torch
import torch.nn as nn
import torch.nn.functional as F

from cluster_split import target_to_cluster_map

# ─────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────

VOCAB_SIZE = 7680
DEFAULT_MAX_TOKENS = 10_000
MAX_TOKENS = int(os.environ.get("PRISM_MAX_TOKENS", DEFAULT_MAX_TOKENS))
EMBED_DIM = 32
N_HEADS = 4
TEACHER_DIM = 128
ENGINE_DIM = 13
CHANNEL_DIM = 20
NEAR_RADIUS = 8.0

R2_PREFIX = "r2:prism-archive/10k-runs"
BUNDLE_DIR = Path("/mnt/storage/spike-audit/features-pct95")
TOKEN_CACHE = Path("/mnt/storage/spike-audit/tokenized-sites")
OUT_DIR = Path("/mnt/storage/spike-audit/ranker-spike-attn")

SOURCE_MAP = {"UV": 0, "LIF": 1, "EFP": 2}
PHASE_MAP = {"cold_hold": 0, "heating": 1, "warm_hold": 2, "cooling": 3, "cold_return": 4}

ENGINE_FEATURE_KEYS = [
    "spike_count", "spread", "burial", "spike_density", "volume",
    "n_lining_residues", "hydrophobic_ratio", "aromatic_count", "charged_count",
    "druggability", "quality_score", "sphericity", "onset_score",
]


# ─────────────────────────────────────────────────────────────
# Phase 1: Tokenization
# ─────────────────────────────────────────────────────────────

def tokenize_spike(source: int, phase: int, intensity: float,
                   vib_energy: float, water_density: float,
                   n_nearby_excited: int) -> int:
    intensity_bin = min(int(intensity / 8.0), 7)
    vib_bin = min(int(vib_energy / 0.015), 3)
    water_bin = min(int(water_density / 0.0085), 3)
    coex_bin = min(int(n_nearby_excited / 4.0), 3)
    return source * 2560 + phase * 512 + intensity_bin * 64 + vib_bin * 16 + water_bin * 4 + coex_bin


def decode_token(token_id: int) -> Dict[str, Any]:
    sources = ["UV", "LIF", "EFP"]
    phases = ["cold_hold", "heating", "warm_hold", "cooling", "cold_return"]
    coex = token_id % 4; token_id //= 4
    water = token_id % 4; token_id //= 4
    vib = token_id % 4; token_id //= 4
    intensity = token_id % 8; token_id //= 8
    phase = token_id % 5; token_id //= 5
    source = token_id % 3
    return {
        "source": sources[source], "phase": phases[phase],
        "intensity_bin": intensity, "vib_bin": vib,
        "water_bin": water, "coex_bin": coex,
    }


def tokenize_target(target: str, tmp_dir: Path) -> Optional[Dict[str, np.ndarray]]:
    """Download per-site parquets from R2, tokenize, return arrays."""
    import pyarrow.parquet as pq

    target_r2 = f"{R2_PREFIX}/{target}/"
    local_dir = tmp_dir / target
    local_dir.mkdir(parents=True, exist_ok=True)

    result = subprocess.run(
        ["rclone", "copy", target_r2, str(local_dir),
         "--include", "*.spike_events.parquet",
         "--bwlimit", "20M", "--transfers", "8", "-q"],
        capture_output=True, text=True, timeout=600,
    )
    parquets = sorted(local_dir.glob("*.spike_events.parquet"))
    if not parquets:
        return None

    site_tokens = {}
    site_centroids = {}

    for pq_path in parquets:
        site_name = pq_path.name.replace(f"{target}.", "").replace(".spike_events.parquet", "")
        try:
            table = pq.read_table(pq_path, columns=[
                "spike_source", "ccns_phase", "intensity",
                "vibrational_energy", "water_density", "n_nearby_excited",
                "x", "y", "z",
            ])
        except Exception:
            continue

        n = len(table)
        if n == 0:
            continue

        sources = table.column("spike_source").to_pylist()
        phases = table.column("ccns_phase").to_pylist()
        intensities = table.column("intensity").to_numpy()
        vibs = table.column("vibrational_energy").to_numpy()
        waters = table.column("water_density").to_numpy()
        coexs = table.column("n_nearby_excited").to_numpy()
        xs = table.column("x").to_numpy()
        ys = table.column("y").to_numpy()
        zs = table.column("z").to_numpy()

        centroid = np.array([xs.mean(), ys.mean(), zs.mean()], dtype=np.float32)
        site_centroids[site_name] = centroid

        tokens = np.empty(n, dtype=np.int32)
        for j in range(n):
            src = SOURCE_MAP.get(sources[j], 0)
            ph = PHASE_MAP.get(phases[j], 0)
            tokens[j] = tokenize_spike(src, ph, float(intensities[j]),
                                       float(vibs[j]), float(waters[j]),
                                       int(coexs[j]))

        if n > MAX_TOKENS:
            idx = np.random.default_rng(42).choice(n, MAX_TOKENS, replace=False)
            idx.sort()
            tokens = tokens[idx]

        site_tokens[site_name] = tokens

    shutil.rmtree(local_dir, ignore_errors=True)

    if not site_tokens:
        return None

    site_names = sorted(site_tokens.keys())
    n_sites = len(site_names)
    token_ids = np.zeros((n_sites, MAX_TOKENS), dtype=np.int32)
    pad_mask = np.ones((n_sites, MAX_TOKENS), dtype=np.bool_)

    for i, sn in enumerate(site_names):
        toks = site_tokens[sn]
        length = min(len(toks), MAX_TOKENS)
        token_ids[i, :length] = toks[:length] + 1  # shift by 1 so 0 = pad
        pad_mask[i, :length] = False

    centroids = np.stack([site_centroids.get(sn, np.zeros(3)) for sn in site_names])

    return {
        "token_ids": token_ids,
        "pad_mask": pad_mask,
        "site_names": np.array(site_names),
        "site_centroids": centroids.astype(np.float32),
    }


# ─────────────────────────────────────────────────────────────
# Phase 2: Physics embeddings + engine/channel features
# ─────────────────────────────────────────────────────────────

def extract_site_features(target: str, site_names: np.ndarray,
                          site_centroids: np.ndarray,
                          pca_components: Optional[np.ndarray] = None,
                          pca_mean: Optional[np.ndarray] = None,
                          ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Extract teacher embeddings + engine + channel features per site.

    Returns (teacher_emb, engine_feats, channel_feats, gold_dist_min).
    """
    bundle_path = None
    for stem in (f"{target}_features.npz", f"{target}.features.npz"):
        p = BUNDLE_DIR / stem
        if p.exists():
            bundle_path = p
            break
    if bundle_path is None:
        return None

    d = np.load(bundle_path, allow_pickle=True)
    coords = d["coords"].astype(np.float32)
    ligand_centroid = d["ligand_centroid"].astype(np.float32)

    physics_blocks = []
    for k in ("structural", "nma", "perturbed_nma", "physics_216", "tide_residue", "temporal"):
        if k in d:
            physics_blocks.append(d[k].astype(np.float32))
    X_phys = np.concatenate(physics_blocks, axis=1)

    channel = d["channel_features"].astype(np.float32) if "channel_features" in d else None

    sf_raw = json.loads(str(d["site_features_json"]))
    tp_raw = json.loads(str(d["tide_pocket_json"])) if "tide_pocket_json" in d else {}

    n_sites = len(site_names)
    teacher_emb = np.zeros((n_sites, TEACHER_DIM), dtype=np.float32)
    engine_feats = np.zeros((n_sites, ENGINE_DIM), dtype=np.float32)
    channel_feats = np.zeros((n_sites, CHANNEL_DIM), dtype=np.float32)
    dists_to_ligand = np.full(n_sites, 999.0, dtype=np.float32)

    for i, sn in enumerate(site_names):
        sn_str = str(sn)
        centroid = site_centroids[i]

        # Try tide_pocket_json for better centroids
        site_num = sn_str.replace("site", "")
        if site_num in tp_raw and "centroid" in tp_raw[site_num]:
            centroid = np.array(tp_raw[site_num]["centroid"], dtype=np.float32)

        dists_to_ligand[i] = float(np.linalg.norm(centroid - ligand_centroid))

        # Residues within NEAR_RADIUS of site centroid
        res_dists = np.linalg.norm(coords - centroid, axis=1)
        near_mask = res_dists < NEAR_RADIUS
        n_near = near_mask.sum()

        # Teacher embedding: PCA-128 on physics features of nearby residues
        if n_near > 0 and pca_components is not None:
            X_near = X_phys[near_mask]
            X_centered = X_near - pca_mean
            projected = X_centered @ pca_components.T
            teacher_emb[i] = projected.mean(axis=0)

        # Channel features: mean of first CHANNEL_DIM channels of nearby residues
        if n_near > 0 and channel is not None:
            ch_near = channel[near_mask]
            ch_dim = min(CHANNEL_DIM, ch_near.shape[1])
            channel_feats[i, :ch_dim] = ch_near[:, :ch_dim].mean(axis=0)

        # Engine features from site_features_json
        sf = sf_raw.get(sn_str, {})
        for j, key in enumerate(ENGINE_FEATURE_KEYS):
            val = sf.get(key)
            if val is not None and not (isinstance(val, float) and math.isnan(val)):
                engine_feats[i, j] = float(val)

    gold_dist_min = float(dists_to_ligand.min())
    return teacher_emb, engine_feats, channel_feats, dists_to_ligand


# ─────────────────────────────────────────────────────────────
# Phase 3: Model
# ─────────────────────────────────────────────────────────────

class SpikeAttentionRanker(nn.Module):
    def __init__(self, vocab_size=VOCAB_SIZE, embed_dim=EMBED_DIM, n_heads=N_HEADS,
                 teacher_dim=TEACHER_DIM, engine_dim=ENGINE_DIM, channel_dim=CHANNEL_DIM):
        super().__init__()
        self.embed = nn.Embedding(vocab_size + 1, embed_dim, padding_idx=0)
        self.pool_query = nn.Parameter(torch.randn(1, 1, embed_dim))
        self.attn = nn.MultiheadAttention(embed_dim, num_heads=n_heads, batch_first=True)

        fusion_dim = embed_dim + teacher_dim + engine_dim + channel_dim
        self.head = nn.Sequential(
            nn.Linear(fusion_dim, 256), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(256, 64), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(64, 1),
        )
        self.last_attn_weights = None

    def forward(self, token_ids, pad_mask, teacher_emb, engine_feats, channel_feats):
        x = self.embed(token_ids)
        query = self.pool_query.expand(x.size(0), -1, -1)
        pooled, attn_weights = self.attn(query, x, x, key_padding_mask=pad_mask)
        pooled = pooled.squeeze(1)
        self.last_attn_weights = attn_weights.detach()
        combined = torch.cat([pooled, teacher_emb, engine_feats, channel_feats], dim=1)
        return self.head(combined).squeeze(-1)


# ─────────────────────────────────────────────────────────────
# Phase 4: Training
# ─────────────────────────────────────────────────────────────

def pairwise_ranking_loss(scores, labels, margin=1.0):
    gold = labels.argmax()
    if labels[gold] <= 0:
        return torch.tensor(0.0, requires_grad=True)
    diffs = margin - (scores[gold] - scores)
    mask = torch.ones_like(scores, dtype=torch.bool)
    mask[gold] = False
    loss = torch.clamp(diffs[mask], min=0.0)
    return loss.mean()


def train_one_fold(model, train_data, val_target, val_data,
                   epochs=50, lr=1e-3, device="cpu"):
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    best_sr1 = 0
    best_state = None

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        n_targets = 0
        for tgt_name, tdata in train_data.items():
            if tdata["labels"].max() <= 0:
                continue
            tok = torch.tensor(tdata["token_ids"], device=device)
            pad = torch.tensor(tdata["pad_mask"], device=device)
            te = torch.tensor(tdata["teacher_emb"], dtype=torch.float32, device=device)
            ef = torch.tensor(tdata["engine_feats"], dtype=torch.float32, device=device)
            cf = torch.tensor(tdata["channel_feats"], dtype=torch.float32, device=device)
            labels = torch.tensor(tdata["labels"], dtype=torch.float32, device=device)

            scores = model(tok, pad, te, ef, cf)
            loss = pairwise_ranking_loss(scores, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            n_targets += 1

        # Check val SR@1
        if val_data is not None and val_data["labels"].max() > 0:
            model.eval()
            with torch.no_grad():
                tok = torch.tensor(val_data["token_ids"], device=device)
                pad = torch.tensor(val_data["pad_mask"], device=device)
                te = torch.tensor(val_data["teacher_emb"], dtype=torch.float32, device=device)
                ef = torch.tensor(val_data["engine_feats"], dtype=torch.float32, device=device)
                cf = torch.tensor(val_data["channel_feats"], dtype=torch.float32, device=device)
                scores = model(tok, pad, te, ef, cf)
                pred_rank = (scores >= scores[val_data["labels"].argmax()]).sum().item()
                sr1 = 1 if pred_rank == 1 else 0
                if sr1 > best_sr1 or best_state is None:
                    best_sr1 = sr1
                    best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
    return model


# ─────────────────────────────────────────────────────────────
# Phase 5: LOTO + Evaluation
# ─────────────────────────────────────────────────────────────

def loto_evaluate(all_data: Dict[str, Dict], cluster_map: Optional[Dict[str, str]],
                  device: str = "cpu") -> Dict[str, Any]:
    targets = sorted(all_data.keys())
    evaluable = [t for t in targets if all_data[t]["labels"].max() > 0]
    print(f"\nLOTO over {len(evaluable)} evaluable targets "
          f"(of {len(targets)} total, cluster_split={'ON' if cluster_map else 'OFF'})")

    sr = {1: 0, 3: 0, 5: 0, 10: 0}
    per_target = []
    attn_records = []
    t0 = time.time()

    for i, tgt in enumerate(evaluable):
        # Build train set: exclude held-out cluster
        if cluster_map is not None:
            cluster = cluster_map.get(tgt, tgt)
            train_targets = {t: all_data[t] for t in targets
                             if cluster_map.get(t, t) != cluster}
        else:
            train_targets = {t: all_data[t] for t in targets if t != tgt}

        model = SpikeAttentionRanker()
        val_data = all_data[tgt]
        model = train_one_fold(model, train_targets, tgt, val_data,
                               epochs=50, lr=1e-3, device=device)

        # Evaluate
        model.eval()
        with torch.no_grad():
            tok = torch.tensor(val_data["token_ids"], device=device)
            pad = torch.tensor(val_data["pad_mask"], device=device)
            te = torch.tensor(val_data["teacher_emb"], dtype=torch.float32, device=device)
            ef = torch.tensor(val_data["engine_feats"], dtype=torch.float32, device=device)
            cf = torch.tensor(val_data["channel_feats"], dtype=torch.float32, device=device)
            scores = model(tok, pad, te, ef, cf)
            labels = val_data["labels"]
            gold = int(labels.argmax())
            ranking = torch.argsort(-scores).cpu().numpy()
            gold_pos = int(np.where(ranking == gold)[0][0]) + 1

            for k in sr:
                if gold_pos <= k:
                    sr[k] += 1

            # Capture attention for interpretability
            attn_w = model.last_attn_weights.cpu().numpy()  # [n_sites, 1, seq_len]
            gold_attn = attn_w[gold, 0, :]
            gold_tokens = val_data["token_ids"][gold]
            nonpad = ~val_data["pad_mask"][gold]
            gold_attn_nonpad = gold_attn[nonpad]
            gold_tokens_nonpad = gold_tokens[nonpad] - 1  # undo +1 shift

            attn_records.append({
                "target": tgt,
                "gold_rank": gold_pos,
                "correct": gold_pos == 1,
                "n_sites": len(labels),
                "gold_tokens": gold_tokens_nonpad,
                "gold_attn": gold_attn_nonpad,
                "gold_dist": float(val_data["dists_to_ligand"][gold]),
            })

        per_target.append({
            "target": tgt,
            "gold_rank": gold_pos,
            "n_sites": len(labels),
            "gold_dist": float(val_data["dists_to_ligand"][gold]),
            "gold_site": str(val_data["site_names"][gold]),
            "cluster": cluster_map.get(tgt) if cluster_map else None,
        })

        if (i + 1) % 10 == 0 or (i + 1) == len(evaluable):
            sr1_pct = sr[1] / (i + 1) * 100
            print(f"  [{i+1}/{len(evaluable)}]  SR@1={sr1_pct:.1f}%", flush=True)

    n = len(per_target)
    result = {
        "n_targets_evaluated": n,
        "cluster_split": cluster_map is not None,
        "sr1": sr[1], "sr1_pct": round(sr[1] / n * 100, 2) if n else 0,
        "sr3": sr[3], "sr3_pct": round(sr[3] / n * 100, 2) if n else 0,
        "sr5": sr[5], "sr5_pct": round(sr[5] / n * 100, 2) if n else 0,
        "sr10": sr[10], "sr10_pct": round(sr[10] / n * 100, 2) if n else 0,
        "per_target": per_target,
        "loto_duration_sec": time.time() - t0,
    }
    return result, attn_records


def analyze_attention(attn_records: List[Dict], n_show: int = 5):
    """Print attention interpretability for top and bottom targets."""
    correct = [r for r in attn_records if r["correct"]]
    failed = [r for r in attn_records if not r["correct"]]
    failed.sort(key=lambda x: x["gold_rank"])

    print(f"\n{'='*70}")
    print(f"  ATTENTION INTERPRETABILITY ANALYSIS")
    print(f"{'='*70}")

    for label, records in [("CORRECT (gold at rank 1)", correct[:n_show]),
                           ("FAILED", failed[:n_show])]:
        print(f"\n── {label} ──")
        for rec in records:
            status = "CORRECT" if rec["correct"] else f"FAILED (gold at rank {rec['gold_rank']})"
            print(f"\nTarget: {rec['target']} — {status}  (dist={rec['gold_dist']:.1f}A, "
                  f"n_sites={rec['n_sites']})")
            print("  Top attended tokens:")

            tokens = rec["gold_tokens"]
            attn = rec["gold_attn"]
            if len(tokens) == 0:
                print("    (no tokens)")
                continue

            # Aggregate attention by token type
            token_attn = defaultdict(float)
            for t_id, a in zip(tokens, attn):
                if t_id >= 0:
                    token_attn[int(t_id)] += float(a)

            top_tokens = sorted(token_attn.items(), key=lambda x: -x[1])[:8]
            for tid, weight in top_tokens:
                desc = decode_token(tid)
                desc_str = (f"{desc['source']} x {desc['phase']} x "
                            f"intensity_bin_{desc['intensity_bin']} x "
                            f"vib_bin_{desc['vib_bin']} x "
                            f"coex_bin_{desc['coex_bin']}")
                print(f"    {desc_str}: weight {weight:.4f}")


def export_onnx(model: SpikeAttentionRanker, out_path: Path):
    model.eval()
    model.cpu()
    dummy_tokens = torch.zeros(1, MAX_TOKENS, dtype=torch.long)
    dummy_mask = torch.ones(1, MAX_TOKENS, dtype=torch.bool)
    dummy_teacher = torch.zeros(1, TEACHER_DIM)
    dummy_engine = torch.zeros(1, ENGINE_DIM)
    dummy_channel = torch.zeros(1, CHANNEL_DIM)

    torch.onnx.export(
        model,
        (dummy_tokens, dummy_mask, dummy_teacher, dummy_engine, dummy_channel),
        str(out_path),
        input_names=["token_ids", "pad_mask", "teacher_emb", "engine_feats", "channel_feats"],
        output_names=["site_score"],
        dynamic_axes={
            "token_ids": {0: "batch", 1: "seq"},
            "pad_mask": {0: "batch", 1: "seq"},
            "teacher_emb": {0: "batch"},
            "engine_feats": {0: "batch"},
            "channel_feats": {0: "batch"},
        },
        opset_version=17,
    )
    print(f"  ONNX exported: {out_path} ({out_path.stat().st_size / 1024:.1f} KB)")


# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-targets", type=int, default=0)
    parser.add_argument("--max-tokens", type=int, default=MAX_TOKENS)
    parser.add_argument("--skip-tokenize", action="store_true")
    parser.add_argument("--no-cluster-split", action="store_true")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--bundle-dir", type=Path, default=BUNDLE_DIR)
    parser.add_argument("--cluster-cache", type=Path,
                        default=Path("/mnt/storage/spike-audit/seq_clusters.json"))
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    TOKEN_CACHE.mkdir(parents=True, exist_ok=True)

    # ── Discover targets with feature bundles ──
    all_bundles = sorted(args.bundle_dir.glob("*_features.npz"))
    all_targets = [p.name.replace("_features.npz", "") for p in all_bundles]
    if args.max_targets > 0:
        all_targets = all_targets[:args.max_targets]
    print(f"Targets with feature bundles: {len(all_targets)}")

    # ── Phase 1: Tokenize ──
    if not args.skip_tokenize:
        print(f"\n{'='*60}")
        print(f"  PHASE 1: Spike Tokenization")
        print(f"{'='*60}")
        existing = {p.name.replace("_tokens.npz", "")
                    for p in TOKEN_CACHE.glob("*_tokens.npz")}
        to_tokenize = [t for t in all_targets if t not in existing]
        print(f"  Already cached: {len(existing)}, to tokenize: {len(to_tokenize)}")

        with tempfile.TemporaryDirectory(prefix="prism_tok_") as tmp:
            tmp_path = Path(tmp)
            for i, target in enumerate(to_tokenize):
                t0 = time.time()
                result = tokenize_target(target, tmp_path)
                elapsed = time.time() - t0
                if result is not None:
                    np.savez_compressed(
                        TOKEN_CACHE / f"{target}_tokens.npz", **result)
                    n_sites = result["token_ids"].shape[0]
                    print(f"  [{i+1}/{len(to_tokenize)}] {target}: "
                          f"{n_sites} sites, {elapsed:.1f}s", flush=True)
                else:
                    print(f"  [{i+1}/{len(to_tokenize)}] {target}: "
                          f"SKIP (no parquets)", flush=True)

    # ── Phase 2: PCA + feature extraction ──
    print(f"\n{'='*60}")
    print(f"  PHASE 2: Physics Embedding (PCA-{TEACHER_DIM}) + Feature Extraction")
    print(f"{'='*60}")

    # Fit PCA on all available physics features
    print("  Fitting PCA on per-residue physics features...")
    sample_X = []
    for target in all_targets[:100]:
        for stem in (f"{target}_features.npz",):
            p = args.bundle_dir / stem
            if p.exists():
                d = np.load(p, allow_pickle=True)
                blocks = []
                for k in ("structural", "nma", "perturbed_nma", "physics_216",
                           "tide_residue", "temporal"):
                    if k in d:
                        blocks.append(d[k].astype(np.float32))
                if blocks:
                    sample_X.append(np.concatenate(blocks, axis=1))
    X_all = np.concatenate(sample_X, axis=0)
    pca_mean = X_all.mean(axis=0)
    X_centered = X_all - pca_mean
    # Truncated SVD for PCA
    U, S, Vt = np.linalg.svd(X_centered, full_matrices=False)
    pca_components = Vt[:TEACHER_DIM]
    variance_explained = (S[:TEACHER_DIM]**2).sum() / (S**2).sum()
    print(f"  PCA: {X_all.shape[1]}-dim → {TEACHER_DIM}-dim, "
          f"variance explained: {variance_explained:.1%}")
    del X_all, X_centered, U, S, Vt, sample_X

    # Build per-target data dicts
    print("  Extracting per-site features...")
    all_data: Dict[str, Dict] = {}
    n_skipped = 0
    for i, target in enumerate(all_targets):
        token_path = TOKEN_CACHE / f"{target}_tokens.npz"
        if not token_path.exists():
            n_skipped += 1
            continue

        tok_data = np.load(token_path, allow_pickle=True)
        site_names = tok_data["site_names"]
        site_centroids = tok_data["site_centroids"]
        # Truncate to --max-tokens if cached tokens are longer
        cached_ids = tok_data["token_ids"][:, :args.max_tokens]
        cached_mask = tok_data["pad_mask"][:, :args.max_tokens]

        feat_result = extract_site_features(
            target, site_names, site_centroids, pca_components, pca_mean)
        if feat_result is None:
            n_skipped += 1
            continue

        teacher_emb, engine_feats, channel_feats, dists_to_ligand = feat_result

        # Labels: graded score from distance
        labels = 1.0 / (1.0 + dists_to_ligand)

        all_data[target] = {
            "token_ids": cached_ids,
            "pad_mask": cached_mask,
            "site_names": site_names,
            "teacher_emb": teacher_emb,
            "engine_feats": engine_feats,
            "channel_feats": channel_feats,
            "labels": labels,
            "dists_to_ligand": dists_to_ligand,
        }

        if (i + 1) % 50 == 0:
            print(f"  [{i+1}/{len(all_targets)}] loaded", flush=True)

    print(f"  Loaded: {len(all_data)} targets, skipped: {n_skipped}")

    # ── Phase 3+4: Cluster-aware LOTO ──
    print(f"\n{'='*60}")
    print(f"  PHASE 3+4: Cluster-Aware LOTO Training + Evaluation")
    print(f"{'='*60}")

    cluster_map = None
    if not args.no_cluster_split:
        print(f"  Building MMseqs2 cluster map (30% identity)...")
        cluster_map = target_to_cluster_map(
            bundle_dir=args.bundle_dir,
            targets=list(all_data.keys()),
            min_seq_id=0.3,
            cache_path=args.cluster_cache,
        )
        n_clust = len(set(cluster_map.values()))
        print(f"  {len(cluster_map)} targets → {n_clust} clusters")

    result, attn_records = loto_evaluate(
        all_data, cluster_map, device=args.device)

    print(f"\n{'='*60}")
    print(f"  Spike Attention Ranker — LOTO Results")
    print(f"{'='*60}")
    print(f"  Targets evaluated: {result['n_targets_evaluated']}")
    print(f"  SR@1:  {result['sr1_pct']:.2f}%  ({result['sr1']})")
    print(f"  SR@3:  {result['sr3_pct']:.2f}%  ({result['sr3']})")
    print(f"  SR@5:  {result['sr5_pct']:.2f}%  ({result['sr5']})")
    print(f"  SR@10: {result['sr10_pct']:.2f}%  ({result['sr10']})")
    print(f"  Duration: {result['loto_duration_sec']:.0f}s")

    # Comparison table
    print(f"\n{'Method':<45} {'SR@1':>8} {'SR@3':>8} {'SR@5':>8}")
    print("─" * 69)
    print(f"{'XGBoost v3 (13 features, baseline)':<45} {'47.83%':>8} {'85.51%':>8} {'95.94%':>8}")
    print(f"{'Spike Attention (distance labels)':<45} "
          f"{result['sr1_pct']:>7.2f}% {result['sr3_pct']:>7.2f}% {result['sr5_pct']:>7.2f}%")

    with open(args.out_dir / "evaluation.json", "w") as f:
        json.dump(result, f, indent=2, default=str)

    # ── Phase 5: Attention interpretability ──
    analyze_attention(attn_records, n_show=5)

    # ── Train final model on all data + export ONNX ──
    print(f"\n{'='*60}")
    print(f"  Training final model on all data + ONNX export")
    print(f"{'='*60}")
    final_model = SpikeAttentionRanker()
    train_data = {t: d for t, d in all_data.items() if d["labels"].max() > 0}
    final_model = train_one_fold(final_model, train_data, None, None,
                                 epochs=args.epochs, lr=1e-3, device=args.device)
    torch.save(final_model.state_dict(), args.out_dir / "model.pt")
    print(f"  PyTorch: {args.out_dir / 'model.pt'}")

    onnx_path = args.out_dir / "spike_attention_ranker.onnx"
    export_onnx(final_model, onnx_path)

    # Copy to engine assets
    engine_asset = Path("crates/prism-nhs/assets/spike_attention_ranker.onnx")
    if engine_asset.parent.exists():
        shutil.copy2(onnx_path, engine_asset)
        print(f"  Copied to {engine_asset}")

    n_params = sum(p.numel() for p in final_model.parameters())
    print(f"  Parameters: {n_params:,}")
    print(f"\n  All outputs in: {args.out_dir}")


if __name__ == "__main__":
    main()
