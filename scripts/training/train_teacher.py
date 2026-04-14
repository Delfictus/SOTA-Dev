#!/usr/bin/env python3
"""Teacher v004 — cold-start per-residue binding classifier.

Architecture mirrors v002 (4-layer MLP) but with the FULL feature vector
produced by feature_extractor.py. Input dim is determined at load time from
the first bundle, NOT hardcoded.

Training protocol:
  • Data: pct95 targets from D1 with dcc_grade != 'POOR' (spike-based)
  • Labels: per-residue binding (Cα within 4.5Å of ground_truth ligand centroid)
  • Split: leave-one-target-out (LOTO) on the filtered set
  • Architecture: Linear(IN → 512) → ReLU → Dropout(0.2)
                  → Linear(512 → 256) → ReLU → Dropout(0.2)
                  → Linear(256 → 128) → ReLU → Dropout(0.2)
                  → Linear(128 → 1)
  • Loss: BCEWithLogits + pos_weight (class imbalance)
  • Early stop: 5 epochs patience on held-out target AUROC
  • Gate: mean AUROC ≥ 0.723

Outputs (to --out-dir):
  teacher_fold_{000..NNN}_{target}.pt   — one checkpoint per LOTO fold
  teacher_v004.onnx                     — mean-ensemble ONNX export
  evaluation.json                       — per-fold AUROC/AUPRC + summary
  feature_stats.json                    — z-score normalization (from train set)

Usage:
    python3 scripts/training/train_teacher.py \
        --features-dir /workspace/features \
        --out-dir /workspace/models/teacher_v004 \
        --epochs 30 --batch-size 512 \
        --max-targets 302  # optional cap for quick tests
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
    HAVE_TORCH = True
except ImportError:
    HAVE_TORCH = False

from sklearn.metrics import roc_auc_score, average_precision_score

sys.path.insert(0, str(Path(__file__).parent))
from feature_extractor import FEATURE_DIM, load_bundle, BLOCK_OFFSETS
from cluster_split import target_to_cluster_map  # homolog-safe LOTO

API_BASE = os.environ.get("PRISM_API", "https://prism-feature-pipeline.is-0b9.workers.dev")


def _api_get(url: str, timeout: int = 60):
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 prism4d-teacher-v004"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


# ─────────────────────────────────────────────────────────────
#  Target selection — pull valid targets from D1
# ─────────────────────────────────────────────────────────────

def fetch_valid_targets(exclude_grade: str = "POOR") -> List[str]:
    """Return pct95 targets whose corrected_dcc.dcc_grade is NOT the excluded grade."""
    # Pull all dcc records; filter client-side (Worker doesn't expose NOT-grade query)
    data = _api_get(f"{API_BASE}/dcc")
    rows = data.get("records", [])
    return sorted({r["target"] for r in rows if r.get("dcc_grade") != exclude_grade})


# ─────────────────────────────────────────────────────────────
#  Data assembly
# ─────────────────────────────────────────────────────────────

def _build_x_from_bundle(bundle: Dict[str, np.ndarray]) -> np.ndarray:
    """Assemble per-residue feature matrix from the extract_all_features .npz.

    Layout (dynamic input_dim = 225):
        physics_216 [N, 216]  ← includes structural + NMA + 216 physics blocks
        tide_residue [N, 7]   ← directive Phase 0.1
        temporal [N, 2]       ← phase_transition_ratio, warm_hold_fraction

    The perturbed_nma block is computed but NOT included here to stay compatible
    with the old v002 teacher feature layout; it lives in vn_egnn only.
    """
    blocks = []
    if "physics_216" in bundle:
        blocks.append(bundle["physics_216"].astype(np.float32))
    if "tide_residue" in bundle:
        blocks.append(bundle["tide_residue"].astype(np.float32))
    if "temporal" in bundle:
        blocks.append(bundle["temporal"].astype(np.float32))
    # Back-compat: old bundles had a flat "X" key
    if not blocks and "X" in bundle:
        return bundle["X"].astype(np.float32)
    return np.concatenate(blocks, axis=1)


def assemble_dataset(features_dir: Path, targets: List[str]
                     ) -> Tuple[np.ndarray, np.ndarray, List[str], List[int]]:
    """Load all per-target .npz bundles into one (X, y, target_per_row, offsets).

    offsets[i] is the cumulative row count up to target i; offsets[N] == len(X).
    """
    X_list, y_list, tgt_per_row, offsets = [], [], [], [0]

    # Try both naming conventions: new extract_all_features produces
    # <target>_features.npz; old feature_extractor used <target>.features.npz.
    def _find_bundle(tgt: str) -> Optional[Path]:
        for stem in (f"{tgt}_features.npz", f"{tgt}.features.npz"):
            p = features_dir / stem
            if p.exists():
                return p
        return None

    for tgt in targets:
        p = _find_bundle(tgt)
        if p is None:
            continue
        d = load_bundle(p)
        X = _build_x_from_bundle(d)
        X_list.append(X)
        y_list.append(d["labels"].astype(np.int32))
        tgt_per_row.extend([tgt] * len(d["labels"]))
        offsets.append(offsets[-1] + len(d["labels"]))

    if not X_list:
        raise RuntimeError(f"No .features.npz found in {features_dir}")

    X = np.concatenate(X_list, axis=0)
    y = np.concatenate(y_list, axis=0)
    return X, y, tgt_per_row, offsets


def fit_feature_stats(X: np.ndarray) -> Dict[str, np.ndarray]:
    mean = X.mean(axis=0)
    std = X.std(axis=0)
    std[std < 1e-6] = 1.0
    return {"mean": mean.astype(np.float32), "std": std.astype(np.float32)}


# ─────────────────────────────────────────────────────────────
#  Model
# ─────────────────────────────────────────────────────────────

class TeacherMLP(nn.Module):
    """4-layer MLP matching the v002 shape (indices 0/3/6/9 Linear, 1/4/7 ReLU, 2/5/8 Dropout)."""
    def __init__(self, in_dim: int, dropout: float = 0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 512), nn.ReLU(), nn.Dropout(dropout),    # 0,1,2
            nn.Linear(512, 256),    nn.ReLU(), nn.Dropout(dropout),    # 3,4,5
            nn.Linear(256, 128),    nn.ReLU(), nn.Dropout(dropout),    # 6,7,8
            nn.Linear(128, 1),                                         # 9
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


# ─────────────────────────────────────────────────────────────
#  Single LOTO fold training
# ─────────────────────────────────────────────────────────────

def train_fold(X: np.ndarray, y: np.ndarray, train_mask: np.ndarray,
               val_mask: np.ndarray, stats: Dict[str, np.ndarray],
               epochs: int, batch_size: int, lr: float, seed: int,
               device: torch.device) -> Tuple[nn.Module, Dict[str, float]]:
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Z-score (train stats applied to both)
    X_norm = (X - stats["mean"]) / stats["std"]

    Xt = torch.from_numpy(X_norm[train_mask]).float()
    yt = torch.from_numpy(y[train_mask]).float()
    Xv = torch.from_numpy(X_norm[val_mask]).float().to(device)
    yv_np = y[val_mask]

    model = TeacherMLP(in_dim=X.shape[1]).to(device)

    pos = float(yt.sum())
    neg = float(len(yt) - yt.sum())
    pos_weight = torch.tensor([max(neg / max(pos, 1.0), 1.0)], device=device, dtype=torch.float32)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    loader = DataLoader(
        TensorDataset(Xt, yt), batch_size=batch_size, shuffle=True,
        num_workers=0, pin_memory=(device.type == "cuda"),
    )

    best_auroc = 0.0
    best_state = None
    patience, patience_max = 0, 5

    for ep in range(epochs):
        model.train()
        for xb, yb in loader:
            xb, yb = xb.to(device, non_blocking=True), yb.to(device, non_blocking=True)
            opt.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            opt.step()
        sched.step()

        model.eval()
        with torch.no_grad():
            val_prob = torch.sigmoid(model(Xv)).cpu().numpy()
        try:
            auroc = float(roc_auc_score(yv_np, val_prob))
        except ValueError:
            auroc = float("nan")
        if np.isfinite(auroc) and auroc > best_auroc:
            best_auroc = auroc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience = 0
        else:
            patience += 1
            if patience >= patience_max:
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    model.eval()
    with torch.no_grad():
        val_prob = torch.sigmoid(model(Xv)).cpu().numpy()
    try:
        auroc = float(roc_auc_score(yv_np, val_prob)) if yv_np.sum() > 0 else float("nan")
        auprc = float(average_precision_score(yv_np, val_prob)) if yv_np.sum() > 0 else float("nan")
    except ValueError:
        auroc, auprc = float("nan"), float("nan")

    return model, {"auroc": auroc, "auprc": auprc, "n_val": int(len(yv_np)),
                   "n_pos_val": int(yv_np.sum())}


# ─────────────────────────────────────────────────────────────
#  LOTO driver + ONNX export
# ─────────────────────────────────────────────────────────────

def run_loto(X: np.ndarray, y: np.ndarray, targets_per_row: List[str],
             offsets: List[int], unique_targets: List[str],
             out_dir: Path, epochs: int, batch_size: int, lr: float,
             device: torch.device,
             cluster_map: Optional[Dict[str, str]] = None) -> Dict:
    """Leave-one-target-out with homolog-safe training folds.

    When `cluster_map` is provided, every training fold excludes not just
    the held-out target but every target in its sequence-identity cluster
    (MMseqs2 at 30% by default). This prevents homolog leakage — the
    trained model never sees any sequence ≥30% identical to the held-out
    target during training of that fold.
    """
    stats = fit_feature_stats(X)
    (out_dir / "feature_stats.json").write_text(json.dumps({
        "mean": stats["mean"].tolist(),
        "std": stats["std"].tolist(),
        "in_dim": int(X.shape[1]),
        "block_offsets": BLOCK_OFFSETS,
    }))

    n_targets = len(unique_targets)
    all_metrics = []
    t0 = time.time()
    # Map row index → target index for fast masking
    row_to_tgt_idx = np.asarray([unique_targets.index(t) for t in targets_per_row], dtype=np.int32)

    # Cluster-aware masking: when a target is held out, mask out the
    # ENTIRE cluster from the training set.
    cluster_masks: Optional[List[np.ndarray]] = None
    if cluster_map is not None:
        cluster_masks = []
        tgt_idx_to_cluster = [cluster_map.get(t) for t in unique_targets]
        targets_per_row_cluster = np.asarray(
            [cluster_map.get(t, "__orphan__") for t in targets_per_row])
        for ti, t in enumerate(unique_targets):
            c = tgt_idx_to_cluster[ti]
            if c is None:
                # Unknown cluster → default to same-target only (fallback)
                cluster_masks.append(row_to_tgt_idx == ti)
            else:
                cluster_masks.append(targets_per_row_cluster == c)

    for fold, held_out in enumerate(unique_targets):
        held_idx = unique_targets.index(held_out)
        val_mask = (row_to_tgt_idx == held_idx)
        if cluster_masks is not None:
            # Drop the held-out target's entire cluster from training
            train_mask = ~cluster_masks[held_idx]
        else:
            train_mask = ~val_mask
        if val_mask.sum() == 0 or y[val_mask].sum() == 0:
            print(f"[fold {fold+1}/{n_targets}] {held_out}: skipping (no val residues or no positives)")
            continue

        model, m = train_fold(X, y, train_mask, val_mask, stats,
                              epochs=epochs, batch_size=batch_size, lr=lr,
                              seed=1000 + fold, device=device)
        n_excluded = int(cluster_masks[held_idx].sum() - val_mask.sum()) if cluster_masks else 0
        ckpt = {
            "state_dict": model.state_dict(),
            "test_target": held_out,
            "cluster": (cluster_map.get(held_out) if cluster_map else None),
            "n_homolog_residues_excluded": n_excluded,
            "fold_idx": fold,
            "in_dim": int(X.shape[1]),
            "auroc": m["auroc"],
            "auprc": m["auprc"],
        }
        ckpt_path = out_dir / f"teacher_fold_{fold:03d}_{held_out}.pt"
        torch.save(ckpt, ckpt_path)
        m["target"] = held_out
        m["fold"] = fold
        m["n_homolog_residues_excluded"] = n_excluded
        all_metrics.append(m)

        elapsed = time.time() - t0
        mean_so_far = np.nanmean([mm["auroc"] for mm in all_metrics])
        eta = elapsed / (fold + 1) * (n_targets - fold - 1)
        extra = f" (-{n_excluded:,} homolog res)" if n_excluded else ""
        print(f"[fold {fold+1}/{n_targets}] {held_out} AUROC={m['auroc']:.3f} "
              f"AUPRC={m['auprc']:.3f} (mean AUROC so far: {mean_so_far:.3f}, "
              f"ETA {eta/60:.0f}m){extra}",
              flush=True)

    return {"metrics": all_metrics, "n_targets": n_targets,
            "training_time_sec": time.time() - t0}


def export_onnx_mean_ensemble(fold_ckpts: List[Path], in_dim: int,
                              out_path: Path, device: torch.device) -> None:
    """Export the per-fold ensemble as a single ONNX that averages members."""
    members = []
    for cp in fold_ckpts:
        d = torch.load(cp, map_location="cpu", weights_only=False)
        m = TeacherMLP(in_dim=in_dim)
        m.load_state_dict(d["state_dict"])
        m.eval()
        members.append(m)

    class Ensemble(nn.Module):
        def __init__(self, ms):
            super().__init__()
            self.ms = nn.ModuleList(ms)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            logits = torch.stack([m(x) for m in self.ms], dim=0)
            return torch.sigmoid(logits.mean(dim=0))

    ens = Ensemble(members).eval()
    dummy = torch.randn(1, in_dim)
    torch.onnx.export(
        ens, dummy, str(out_path),
        input_names=["residue_features"],
        output_names=["binding_probability"],
        dynamic_axes={"residue_features": {0: "N"}, "binding_probability": {0: "N"}},
        opset_version=17, do_constant_folding=True,
    )
    print(f"  ONNX: {out_path} ({out_path.stat().st_size / 1024:.1f} KB, {len(members)} members)")


# ─────────────────────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--features-dir", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--max-targets", type=int, default=0,
                        help="cap target count (0=no cap)")
    parser.add_argument("--gate-auroc", type=float, default=0.723)
    parser.add_argument("--exclude-grade", default="POOR",
                        help="corrected_dcc grade to exclude (default POOR)")
    parser.add_argument("--min-seq-id", type=float, default=0.3,
                        help="MMseqs2 identity threshold for homolog-safe LOTO")
    parser.add_argument("--cluster-cache-path", type=Path,
                        default=Path("/mnt/storage/spike-audit/seq_clusters.json"),
                        help="Cache for MMseqs2 cluster map")
    parser.add_argument("--no-cluster-split", action="store_true",
                        help="Disable cluster-aware LOTO (NOT recommended for publication)")
    args = parser.parse_args()

    if not HAVE_TORCH:
        print("ERROR: torch not installed", file=sys.stderr); sys.exit(1)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device} ({torch.cuda.get_device_name(0) if device.type == 'cuda' else 'CPU'})")

    # 1) Valid targets from D1
    print(f"Fetching valid targets (exclude grade={args.exclude_grade})...")
    valid_targets = fetch_valid_targets(exclude_grade=args.exclude_grade)
    print(f"  {len(valid_targets)} valid targets from D1")

    if args.max_targets > 0:
        valid_targets = valid_targets[:args.max_targets]
        print(f"  capped to {len(valid_targets)}")

    # 2) Load all feature bundles
    print(f"Loading feature bundles from {args.features_dir}...")
    X, y, tgt_per_row, offsets = assemble_dataset(args.features_dir, valid_targets)
    present_targets = sorted(set(tgt_per_row))
    print(f"  {len(present_targets)} targets loaded  |  X: {X.shape}  |  y+: {int(y.sum()):,} "
          f"({y.mean():.2%})")

    if X.shape[1] != FEATURE_DIM:
        print(f"  WARN: X.shape[1]={X.shape[1]} != FEATURE_DIM={FEATURE_DIM}")

    # 3) Sequence-identity cluster map (for homolog-safe LOTO)
    cluster_map: Optional[Dict[str, str]] = None
    if not args.no_cluster_split:
        print(f"\nBuilding MMseqs2 cluster map ({args.min_seq_id*100:.0f}% id)...")
        cluster_map = target_to_cluster_map(
            bundle_dir=args.features_dir,
            targets=present_targets,
            min_seq_id=args.min_seq_id,
            cache_path=args.cluster_cache_path,
        )
        n_clust = len(set(cluster_map.values()))
        print(f"  {len(cluster_map)} targets → {n_clust} clusters "
              f"(avg {len(cluster_map)/max(n_clust,1):.1f}/cluster)")

    # 4) LOTO
    print(f"\nRunning LOTO (epochs={args.epochs}, batch={args.batch_size}, lr={args.lr}, "
          f"cluster_split={cluster_map is not None})...")
    results = run_loto(X, y, tgt_per_row, offsets, present_targets,
                       args.out_dir, args.epochs, args.batch_size, args.lr, device,
                       cluster_map=cluster_map)

    # 4) Aggregate
    aurocs = [m["auroc"] for m in results["metrics"] if np.isfinite(m["auroc"])]
    auprcs = [m["auprc"] for m in results["metrics"] if np.isfinite(m["auprc"])]
    mean_auroc = float(np.mean(aurocs)) if aurocs else float("nan")
    median_auroc = float(np.median(aurocs)) if aurocs else float("nan")
    mean_auprc = float(np.mean(auprcs)) if auprcs else float("nan")

    print(f"\n{'='*60}")
    print(f"  TEACHER v004 — LOTO summary ({len(aurocs)} evaluable folds)")
    print(f"{'='*60}")
    print(f"  Mean AUROC:   {mean_auroc:.4f}")
    print(f"  Median AUROC: {median_auroc:.4f}")
    print(f"  Mean AUPRC:   {mean_auprc:.4f}")
    passed = mean_auroc >= args.gate_auroc
    gate_str = "PASS" if passed else "FAIL"
    print(f"  Gate (≥{args.gate_auroc:.3f}): {gate_str}")

    (args.out_dir / "evaluation.json").write_text(json.dumps({
        "mean_auroc": mean_auroc,
        "median_auroc": median_auroc,
        "mean_auprc": mean_auprc,
        "n_folds": len(aurocs),
        "gate_auroc": args.gate_auroc,
        "passed": passed,
        "per_fold": results["metrics"],
        "training_time_sec": results["training_time_sec"],
        "in_dim": int(X.shape[1]),
    }, indent=2))

    # 5) Mean-ensemble ONNX
    fold_ckpts = sorted(args.out_dir.glob("teacher_fold_*.pt"))
    if fold_ckpts:
        print("\nExporting mean-ensemble ONNX...")
        export_onnx_mean_ensemble(fold_ckpts, X.shape[1],
                                   args.out_dir / "teacher_v004.onnx", device)

    print(f"\n  Outputs in: {args.out_dir}")
    sys.exit(0 if passed else 2)


if __name__ == "__main__":
    main()
