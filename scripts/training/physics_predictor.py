#!/usr/bin/env python3
"""PhysicsPredictor v001 — learn to predict engine-derived features from
PDB-computable inputs.

Input  (1336 dims, all derivable without running the PRISM-4D engine):
  structural     25     AA one-hot, DSSP, SASA, hydrophobicity, B-factor
  nma            26     ProDy ANM mode displacements + global stats
  perturbed_nma   5     ligand-direction projected NMA stats
  esm2         1280     ESM-2 t33 residue embeddings

Target (333 dims, only available after an engine run):
  physics_216   216     feature_registry_216 blocks (KCC+thermo+site+phase+dyn)
  tide_residue    7     TIDE per-residue (transfer_entropy, causal_dg, fisher,
                        kl_divergence, role_trigger, role_responder, n_spikes)
  temporal        2     phase_transition_ratio, warm_hold_fraction
  channel_feat  108     v002 neuromorphic 6-tier channel features
  ─────────────────────
  TOTAL         333

Training:
  • Cluster-aware train/val split at 30% MMseqs2 seq-id (reuse existing cache)
  • Per-feature z-score normalization from train targets
  • Huber loss (robust to outliers)
  • AdamW, cosine LR, early-stop on val MSE

Outputs to --out-dir:
  physics_predictor_v001.pt        weights
  physics_predictor_v001.onnx      ONNX export (dyn batch)
  target_stats.json                {mean, std} per output dim — needed to de-norm
  evaluation.json                  val MSE, per-block breakdown, correlation

Usage (on the RunPod pod):
  python3 scripts/training/physics_predictor.py \
      --features-dir /workspace/features \
      --out-dir /workspace/models/physics_predictor_v001 \
      --cluster-cache-path /workspace/seq_clusters_30.json \
      --epochs 100 --batch-size 2048 --lr 1e-3
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, TensorDataset
    HAVE_TORCH = True
except ImportError:
    HAVE_TORCH = False

sys.path.insert(0, str(Path(__file__).parent))
from cluster_split import cluster_split_bundles  # homolog-safe splits

INPUT_BLOCKS  = ("structural", "nma", "perturbed_nma", "esm2")      # 1336 dims
TARGET_BLOCKS = ("physics_216", "tide_residue", "temporal",
                 "channel_features")                                 # 333 dims
INPUT_DIM = 25 + 26 + 5 + 1280
TARGET_DIM = 216 + 7 + 2 + 108


# ─────────────────────────────────────────────────────────────
#  Model
# ─────────────────────────────────────────────────────────────

class PhysicsPredictor(nn.Module):
    """Predict engine-derived features from PDB-computable features."""
    def __init__(self, in_dim: int = INPUT_DIM, out_dim: int = TARGET_DIM,
                 hidden: int = 512, dropout: float = 0.1):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ─────────────────────────────────────────────────────────────
#  Bundle loader
# ─────────────────────────────────────────────────────────────

def _concat_blocks(bundle, keys: Tuple[str, ...], expected_dim: int,
                   target: str) -> Optional[np.ndarray]:
    parts = []
    for k in keys:
        if k not in bundle.files:
            return None
        arr = bundle[k]
        if arr.ndim != 2:
            return None
        parts.append(arr.astype(np.float32))
    x = np.concatenate(parts, axis=1)
    if x.shape[1] != expected_dim:
        print(f"  {target}: block dim {x.shape[1]} != {expected_dim}", flush=True)
        return None
    return x


def load_pairs(features_dir: Path, targets: List[str]
               ) -> Tuple[np.ndarray, np.ndarray, List[str], List[int]]:
    X_list, Y_list, tgt_per_row, offsets = [], [], [], [0]
    skipped = 0
    for t in targets:
        for stem in (f"{t}_features.npz", f"{t}.features.npz"):
            p = features_dir / stem
            if p.exists():
                break
        else:
            skipped += 1; continue
        d = np.load(p, allow_pickle=False)
        X = _concat_blocks(d, INPUT_BLOCKS, INPUT_DIM, t)
        Y = _concat_blocks(d, TARGET_BLOCKS, TARGET_DIM, t)
        if X is None or Y is None or X.shape[0] != Y.shape[0]:
            skipped += 1; continue
        X_list.append(X); Y_list.append(Y)
        tgt_per_row.extend([t] * X.shape[0])
        offsets.append(offsets[-1] + X.shape[0])
    print(f"  Usable: {len(X_list)} / {len(targets)}  (skipped: {skipped})")
    if not X_list:
        raise RuntimeError("No usable feature bundles")
    X = np.concatenate(X_list, axis=0)
    Y = np.concatenate(Y_list, axis=0)
    return X, Y, tgt_per_row, offsets


# ─────────────────────────────────────────────────────────────
#  Stats helpers
# ─────────────────────────────────────────────────────────────

def fit_target_stats(Y: np.ndarray) -> Dict[str, np.ndarray]:
    mean = Y.mean(axis=0)
    std = Y.std(axis=0)
    std[std < 1e-6] = 1.0
    return {"mean": mean.astype(np.float32), "std": std.astype(np.float32)}


def fit_input_stats(X: np.ndarray) -> Dict[str, np.ndarray]:
    mean = X.mean(axis=0)
    std = X.std(axis=0)
    std[std < 1e-6] = 1.0
    return {"mean": mean.astype(np.float32), "std": std.astype(np.float32)}


def block_mse(pred: torch.Tensor, tgt: torch.Tensor) -> Dict[str, float]:
    """Per-feature-block MSE after normalization."""
    blocks = [
        ("physics_216",   0, 216),
        ("tide_residue",  216, 223),
        ("temporal",      223, 225),
        ("channel_feat",  225, 333),
    ]
    out = {}
    for name, lo, hi in blocks:
        se = (pred[:, lo:hi] - tgt[:, lo:hi]) ** 2
        out[name] = float(se.mean().item())
    return out


def per_block_r2(pred: np.ndarray, tgt: np.ndarray) -> Dict[str, float]:
    blocks = [("physics_216", 0, 216), ("tide_residue", 216, 223),
              ("temporal", 223, 225), ("channel_feat", 225, 333)]
    out = {}
    for name, lo, hi in blocks:
        p = pred[:, lo:hi]; t = tgt[:, lo:hi]
        ss_res = ((t - p) ** 2).sum()
        ss_tot = ((t - t.mean(axis=0)) ** 2).sum() + 1e-9
        out[name] = float(1.0 - ss_res / ss_tot)
    return out


# ─────────────────────────────────────────────────────────────
#  Training
# ─────────────────────────────────────────────────────────────

def train(X_tr: np.ndarray, Y_tr: np.ndarray, X_val: np.ndarray, Y_val: np.ndarray,
          input_stats: Dict[str, np.ndarray], target_stats: Dict[str, np.ndarray],
          epochs: int, batch_size: int, lr: float,
          device: torch.device) -> Tuple[nn.Module, Dict]:
    torch.manual_seed(42)

    Xn_tr = (X_tr - input_stats["mean"]) / input_stats["std"]
    Xn_val = (X_val - input_stats["mean"]) / input_stats["std"]
    Yn_tr = (Y_tr - target_stats["mean"]) / target_stats["std"]
    Yn_val = (Y_val - target_stats["mean"]) / target_stats["std"]

    Xt = torch.from_numpy(Xn_tr).float()
    Yt = torch.from_numpy(Yn_tr).float()
    Xv = torch.from_numpy(Xn_val).float().to(device)
    Yv = torch.from_numpy(Yn_val).float().to(device)

    model = PhysicsPredictor(in_dim=X_tr.shape[1], out_dim=Y_tr.shape[1]).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  model params: {n_params:,}  in={X_tr.shape[1]}  out={Y_tr.shape[1]}")

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    loader = DataLoader(
        TensorDataset(Xt, Yt), batch_size=batch_size, shuffle=True,
        num_workers=0, pin_memory=(device.type == "cuda"),
    )

    best_val_mse = float("inf")
    best_state = None
    patience, patience_max = 0, 10
    history = []

    for ep in range(epochs):
        model.train()
        tr_loss_sum = 0.0
        n_batches = 0
        for xb, yb in loader:
            xb, yb = xb.to(device, non_blocking=True), yb.to(device, non_blocking=True)
            opt.zero_grad()
            pred = model(xb)
            loss = F.smooth_l1_loss(pred, yb, beta=1.0)   # Huber
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            opt.step()
            tr_loss_sum += loss.item()
            n_batches += 1
        sched.step()
        tr_loss = tr_loss_sum / max(n_batches, 1)

        model.eval()
        with torch.no_grad():
            pred_val = model(Xv)
            val_mse = F.mse_loss(pred_val, Yv).item()
            val_huber = F.smooth_l1_loss(pred_val, Yv).item()
            blk = block_mse(pred_val, Yv)
        history.append({"epoch": ep + 1, "tr_huber": tr_loss,
                        "val_mse": val_mse, "val_huber": val_huber, **blk})
        if (ep + 1) % 5 == 0 or ep == 0:
            print(f"  ep {ep+1:3d}  tr_huber={tr_loss:.4f}  val_mse={val_mse:.4f}  "
                  f"val_huber={val_huber:.4f}  blk_phys={blk['physics_216']:.3f}  "
                  f"blk_chan={blk['channel_feat']:.3f}", flush=True)

        if val_mse < best_val_mse:
            best_val_mse = val_mse
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience = 0
        else:
            patience += 1
            if patience >= patience_max:
                print(f"  Early stop at epoch {ep+1}")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, {"best_val_mse": best_val_mse, "history": history,
                   "n_params": n_params}


# ─────────────────────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--features-dir", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--min-seq-id", type=float, default=0.3)
    parser.add_argument("--val-frac", type=float, default=0.15)
    parser.add_argument("--test-frac", type=float, default=0.05)
    parser.add_argument("--cluster-cache-path", type=Path,
                        default=Path("/workspace/seq_clusters_30.json"))
    args = parser.parse_args()

    if not HAVE_TORCH:
        print("ERROR: torch not installed"); sys.exit(1)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # 1) Target list = every bundle in features-dir
    all_npzs = sorted(args.features_dir.glob("*_features.npz"))
    all_targets = [p.stem.replace("_features", "") for p in all_npzs]
    print(f"Total bundles: {len(all_targets)}")

    # 2) Homolog-safe split (cluster level)
    train_names, val_names, test_names = cluster_split_bundles(
        bundle_dir=args.features_dir, targets=all_targets,
        val_frac=args.val_frac, test_frac=args.test_frac,
        min_seq_id=args.min_seq_id,
        cache_path=args.cluster_cache_path,
    )
    train_set = set(train_names); val_set = set(val_names); test_set = set(test_names)

    # 3) Load + concat features
    t0 = time.time()
    print(f"\nLoading train pairs ({len(train_names)} targets)...")
    X_tr, Y_tr, _, _ = load_pairs(args.features_dir, train_names)
    print(f"  X_tr: {X_tr.shape}  Y_tr: {Y_tr.shape}")
    print(f"\nLoading val pairs ({len(val_names)} targets)...")
    X_val, Y_val, _, _ = load_pairs(args.features_dir, val_names)
    print(f"  X_val: {X_val.shape}  Y_val: {Y_val.shape}")
    print(f"\nLoading test pairs ({len(test_names)} targets)...")
    X_test, Y_test, _, _ = load_pairs(args.features_dir, test_names)
    print(f"  X_test: {X_test.shape}  Y_test: {Y_test.shape}")
    print(f"  load time: {(time.time()-t0):.1f}s")

    # 4) Normalizer fit on train
    in_stats = fit_input_stats(X_tr)
    tgt_stats = fit_target_stats(Y_tr)
    (args.out_dir / "input_stats.json").write_text(json.dumps({
        "mean": in_stats["mean"].tolist(), "std": in_stats["std"].tolist(),
        "in_dim": int(X_tr.shape[1]),
        "blocks": {"structural": [0, 25], "nma": [25, 51],
                   "perturbed_nma": [51, 56], "esm2": [56, INPUT_DIM]},
    }))
    (args.out_dir / "target_stats.json").write_text(json.dumps({
        "mean": tgt_stats["mean"].tolist(), "std": tgt_stats["std"].tolist(),
        "out_dim": int(Y_tr.shape[1]),
        "blocks": {"physics_216": [0, 216], "tide_residue": [216, 223],
                   "temporal": [223, 225], "channel_features": [225, 333]},
    }))

    # 5) Train
    print(f"\nTraining (epochs={args.epochs}, batch={args.batch_size}, lr={args.lr})...")
    model, train_info = train(X_tr, Y_tr, X_val, Y_val, in_stats, tgt_stats,
                               args.epochs, args.batch_size, args.lr, device)

    # 6) Final eval on val + test (un-normalized → but reporting block MSE on normalized)
    model.eval()
    Xv_norm = torch.from_numpy((X_val - in_stats["mean"]) / in_stats["std"]).float().to(device)
    Yv_norm = torch.from_numpy((Y_val - tgt_stats["mean"]) / tgt_stats["std"]).float().to(device)
    Xt_norm = torch.from_numpy((X_test - in_stats["mean"]) / in_stats["std"]).float().to(device)
    Yt_norm = torch.from_numpy((Y_test - tgt_stats["mean"]) / tgt_stats["std"]).float().to(device)

    with torch.no_grad():
        val_pred = model(Xv_norm).cpu().numpy()
        test_pred = model(Xt_norm).cpu().numpy()

    # Un-normalize for R²
    val_pred_denorm = val_pred * tgt_stats["std"] + tgt_stats["mean"]
    test_pred_denorm = test_pred * tgt_stats["std"] + tgt_stats["mean"]
    val_r2 = per_block_r2(val_pred_denorm, Y_val)
    test_r2 = per_block_r2(test_pred_denorm, Y_test)

    # Per-block norm-space MSE
    val_blk_mse = block_mse(torch.from_numpy(val_pred), Yv_norm.cpu())
    test_blk_mse = block_mse(torch.from_numpy(test_pred), Yt_norm.cpu())

    # 7) Save model + eval
    torch.save(model.state_dict(), args.out_dir / "physics_predictor_v001.pt")

    eval_out = {
        "in_dim": int(X_tr.shape[1]),
        "out_dim": int(Y_tr.shape[1]),
        "n_params": train_info["n_params"],
        "best_val_mse": train_info["best_val_mse"],
        "n_train_targets": len(train_names),
        "n_val_targets": len(val_names),
        "n_test_targets": len(test_names),
        "n_train_residues": int(X_tr.shape[0]),
        "n_val_residues": int(X_val.shape[0]),
        "n_test_residues": int(X_test.shape[0]),
        "val": {"block_mse_norm": val_blk_mse, "block_r2_denorm": val_r2},
        "test": {"block_mse_norm": test_blk_mse, "block_r2_denorm": test_r2},
        "history": train_info["history"],
    }
    (args.out_dir / "evaluation.json").write_text(json.dumps(eval_out, indent=2))

    print(f"\n{'='*60}\n  PhysicsPredictor v001 — final\n{'='*60}")
    print(f"  n_train={len(train_names)} val={len(val_names)} test={len(test_names)}")
    print(f"  Best val MSE (norm-space): {train_info['best_val_mse']:.4f}")
    print(f"  VAL R² per block (denorm):  {val_r2}")
    print(f"  TEST R² per block (denorm): {test_r2}")

    # 8) ONNX export
    print("\nExporting ONNX...")
    model.cpu().eval()
    dummy = torch.randn(1, X_tr.shape[1])
    torch.onnx.export(
        model, dummy, str(args.out_dir / "physics_predictor_v001.onnx"),
        input_names=["pdb_features"], output_names=["predicted_physics"],
        dynamic_axes={"pdb_features": {0: "N"}, "predicted_physics": {0: "N"}},
        opset_version=17, do_constant_folding=True,
    )
    print(f"  Outputs in: {args.out_dir}")


if __name__ == "__main__":
    main()
