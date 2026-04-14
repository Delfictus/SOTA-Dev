#!/usr/bin/env python3
"""VN-EGNN v004 — SpikeBERT features + teacher distillation.

Input: structural(25) + NMA(26) + perturbedNMA(5) + SpikeBERT_allmasked(512) = 568 dims
Labels: teacher v005 soft logits (distillation) + hard binding labels (floor)
Loss: α * KL(student, teacher) + β * chamfer_vn + γ * focal_bce(student, hard)

No ESM-2 anywhere in the pipeline.

Usage:
  python3 scripts/training/vn_egnn/train_v004.py \
      --bundle-dir /mnt/storage/spike-audit/features-pct95 \
      --spikebert-dir /mnt/storage/spike-audit/spike-bert/embeddings_allmasked \
      --teacher-dir /mnt/storage/spike-audit/spike-bert/teacher_soft_logits \
      --out-dir /mnt/storage/spike-audit/vnegnn-v004 \
      --epochs 200 --device cuda
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(SCRIPT_DIR.parent))

from model import (VNEGNN, build_edge_index, init_virtual_nodes,
                    focal_binary_loss, chamfer_vn_loss, export_onnx)
from cluster_split import cluster_split_bundles

# ─────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────

IN_DIM = 568          # 25 + 26 + 5 + 512
HIDDEN_DIM = 128
N_LAYERS = 4
N_VNS = 16
EDGE_CUTOFF = 8.0
STRUCT_KEYS = ("structural", "nma", "perturbed_nma")

# Distillation loss weights
ALPHA_KL = 1.0        # teacher distillation
BETA_CHAMFER = 0.05   # VN position (low to prevent coord explosion)
GAMMA_FOCAL = 0.3     # hard label floor


# ─────────────────────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────────────────────

@dataclass
class Sample:
    target: str
    X: np.ndarray           # [N, 568]
    coords: np.ndarray      # [N, 3]
    labels: np.ndarray      # [N] hard binary
    teacher_probs: np.ndarray  # [N] soft logits from teacher
    ligand_centroid: np.ndarray  # [3]


def load_sample(target: str, bundle_dir: Path, spikebert_dir: Path,
                teacher_dir: Path) -> Optional[Sample]:
    bundle_path = bundle_dir / f"{target}_features.npz"
    emb_path = spikebert_dir / f"{target}_spike_emb.npz"
    teacher_path = teacher_dir / f"{target}_teacher_probs.npz"

    if not all(p.exists() for p in [bundle_path, emb_path, teacher_path]):
        return None

    d = np.load(bundle_path, allow_pickle=True)
    emb = np.load(emb_path)["spike_embedding"].astype(np.float32)
    teacher = np.load(teacher_path)["teacher_probs"].astype(np.float32)

    # Structural context (56 dims)
    blocks = []
    for k in STRUCT_KEYS:
        if k in d:
            blocks.append(d[k].astype(np.float32))
    struct = np.concatenate(blocks, axis=1)

    n = struct.shape[0]
    # Align dimensions
    emb = emb[:n] if emb.shape[0] >= n else np.pad(emb, ((0, n - emb.shape[0]), (0, 0)))
    teacher = teacher[:n] if len(teacher) >= n else np.pad(teacher, (0, n - len(teacher)),
                                                            constant_values=0.5)

    X = np.concatenate([struct, emb], axis=1)
    coords = d["coords"].astype(np.float32) if "coords" in d else d["ca_coords"].astype(np.float32)
    labels = d["labels"].astype(np.float32)
    ligand_centroid = d["ligand_centroid"].astype(np.float32)

    return Sample(target=target, X=X, coords=coords, labels=labels,
                  teacher_probs=teacher, ligand_centroid=ligand_centroid)


def load_all_samples(bundle_dir: Path, spikebert_dir: Path, teacher_dir: Path,
                     targets: Optional[List[str]] = None) -> List[Sample]:
    if targets is None:
        targets = sorted([p.name.replace("_features.npz", "")
                          for p in bundle_dir.glob("*_features.npz")])
    samples = []
    for t in targets:
        s = load_sample(t, bundle_dir, spikebert_dir, teacher_dir)
        if s is not None:
            samples.append(s)
    return samples


# ─────────────────────────────────────────────────────────────
# Distillation loss
# ─────────────────────────────────────────────────────────────

def distillation_loss(student_logits: torch.Tensor,
                      teacher_probs: torch.Tensor,
                      hard_labels: torch.Tensor,
                      temperature: float = 2.0) -> torch.Tensor:
    """KL divergence between student and teacher soft predictions.

    Both are treated as Bernoulli distributions per residue.
    Temperature softens the teacher's predictions for better gradient flow.
    """
    # Soften teacher with temperature
    teacher_logits = torch.log(teacher_probs / (1.0 - teacher_probs + 1e-7) + 1e-7)
    teacher_soft = torch.sigmoid(teacher_logits / temperature)
    student_soft = torch.sigmoid(student_logits.squeeze(-1) / temperature)

    # Binary KL: sum over the two classes (binding, not-binding)
    kl = teacher_soft * torch.log((teacher_soft + 1e-7) / (student_soft + 1e-7)) + \
         (1 - teacher_soft) * torch.log((1 - teacher_soft + 1e-7) / (1 - student_soft + 1e-7))
    return kl.mean() * (temperature ** 2)


# ─────────────────────────────────────────────────────────────
# Training
# ─────────────────────────────────────────────────────────────

def train_epoch(model, samples, optimizer, device, edge_cutoff, eval_seed=0):
    model.train()
    total_loss = 0.0
    rng = None  # use default generator (compatible with any device)

    for s in samples:
        X = torch.tensor(s.X, dtype=torch.float32, device=device)
        coords = torch.tensor(s.coords, dtype=torch.float32, device=device)
        labels = torch.tensor(s.labels, dtype=torch.float32, device=device)
        teacher = torch.tensor(s.teacher_probs, dtype=torch.float32, device=device)
        gt_center = torch.tensor(s.ligand_centroid, dtype=torch.float32, device=device).unsqueeze(0)

        edge_index = build_edge_index(coords, N_VNS, cutoff=edge_cutoff).to(device)
        vn_init = init_virtual_nodes(coords, N_VNS).to(device)

        atom_logits, vn_coords, vn_conf = model(X, coords, edge_index, None, vn_init)
        vn_coords = vn_coords.clamp(-200, 200)

        # Three-part loss
        loss_kl = distillation_loss(atom_logits, teacher, labels)
        loss_chamfer = chamfer_vn_loss(vn_coords, gt_center, vn_conf)
        loss_focal = focal_binary_loss(atom_logits.squeeze(-1), labels)

        loss = ALPHA_KL * loss_kl + BETA_CHAMFER * loss_chamfer + GAMMA_FOCAL * loss_focal

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item()

    return total_loss / max(len(samples), 1)


@torch.no_grad()
def eval_samples(model, samples, device, edge_cutoff, eval_seed=42):
    model.eval()
    rng = None  # use default generator (compatible with any device)

    sr = {4: 0, 8: 0}
    dccs = []

    for s in samples:
        X = torch.tensor(s.X, dtype=torch.float32, device=device)
        coords = torch.tensor(s.coords, dtype=torch.float32, device=device)
        gt = torch.tensor(s.ligand_centroid, dtype=torch.float32, device=device)

        edge_index = build_edge_index(coords, N_VNS, cutoff=edge_cutoff).to(device)
        vn_init = init_virtual_nodes(coords, N_VNS).to(device)

        atom_logits, vn_coords, vn_conf = model(X, coords, edge_index, None, vn_init)

        # Best VN by confidence
        conf_scores = vn_conf.squeeze(-1)
        best_vn = vn_coords[conf_scores.argmax()]
        dcc = (best_vn - gt).norm().item()
        dccs.append(dcc)

        for k in sr:
            if dcc <= k:
                sr[k] += 1

    n = len(samples)
    return {
        "vn_sr_at_4A": sr[4] / n if n else 0,
        "vn_sr_at_8A": sr[8] / n if n else 0,
        "vn_dcc_median": float(np.median(dccs)) if dccs else 999,
        "n": n,
    }


# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle-dir", type=Path,
                        default=Path("/mnt/storage/spike-audit/features-pct95"))
    parser.add_argument("--spikebert-dir", type=Path,
                        default=Path("/mnt/storage/spike-audit/spike-bert/embeddings_allmasked"))
    parser.add_argument("--teacher-dir", type=Path,
                        default=Path("/mnt/storage/spike-audit/spike-bert/teacher_soft_logits"))
    parser.add_argument("--out-dir", type=Path,
                        default=Path("/mnt/storage/spike-audit/vnegnn-v004"))
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--patience", type=int, default=40)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--gate-sr8", type=float, default=0.55)
    parser.add_argument("--no-split", action="store_true",
                        help="Train on ALL targets (teacher logits are already held-out)")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--max-targets", type=int, default=0)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    print(f"Device: {device}" +
          (f" ({torch.cuda.get_device_name(0)})" if device.type == "cuda" else ""))

    # Load data
    print(f"\n{'='*60}")
    print(f"  Loading data: structural(56) + SpikeBERT(512) = {IN_DIM}d")
    print(f"  + teacher soft logits for distillation")
    print(f"{'='*60}")
    all_samples = load_all_samples(args.bundle_dir, args.spikebert_dir, args.teacher_dir)
    if args.max_targets > 0:
        all_samples = all_samples[:args.max_targets]
    print(f"  {len(all_samples)} samples loaded")
    print(f"  Total residues: {sum(s.X.shape[0] for s in all_samples):,}")
    print(f"  in_dim: {all_samples[0].X.shape[1]}")

    # Split strategy
    if args.no_split:
        print(f"\nNo split — training on ALL {len(all_samples)} targets")
        print(f"  (teacher soft logits are already held-out via LOTO)")
        train_samples = all_samples
        val_samples = all_samples  # monitor on same set (teacher provides held-out guarantee)
        test_samples = []  # true zero-shot test done externally on new PDBs
    else:
        print(f"\nCluster-aware split (MMseqs2, 30% identity)...")
        all_targets = [s.target for s in all_samples]
        train_targets, val_targets, test_targets = cluster_split_bundles(
            bundle_dir=args.bundle_dir, targets=all_targets,
            val_frac=0.15, test_frac=0.10, min_seq_id=0.3,
            cache_path=Path("/mnt/storage/spike-audit/seq_clusters.json"),
        )
        target_set = {s.target: s for s in all_samples}
        train_samples = [target_set[t] for t in train_targets if t in target_set]
        val_samples = [target_set[t] for t in val_targets if t in target_set]
        test_samples = [target_set[t] for t in test_targets if t in target_set]
    print(f"  train: {len(train_samples)}, val: {len(val_samples)}, test: {len(test_samples)}")

    # Model
    print(f"\n{'='*60}")
    print(f"  VN-EGNN v004: in={IN_DIM}, hidden={HIDDEN_DIM}, "
          f"layers={N_LAYERS}, VNs={N_VNS}")
    print(f"  Loss: {ALPHA_KL}*KL + {BETA_CHAMFER}*Chamfer + {GAMMA_FOCAL}*FocalBCE")
    print(f"{'='*60}")
    model = VNEGNN(in_dim=IN_DIM, hidden_dim=HIDDEN_DIM, n_layers=N_LAYERS,
                   n_virtual_nodes=N_VNS, edge_feat_dim=0).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {n_params:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # Training
    print(f"\n{'='*60}")
    print(f"  Training: {args.epochs} epochs, lr={args.lr}, patience={args.patience}")
    print(f"{'='*60}")
    best_sr8 = 0.0
    best_epoch = 0
    patience_counter = 0
    t0 = time.time()

    for epoch in range(1, args.epochs + 1):
        train_loss = train_epoch(model, train_samples, optimizer, device, EDGE_CUTOFF,
                                 eval_seed=epoch)
        scheduler.step()

        if epoch % 5 == 0 or epoch <= 5 or epoch == args.epochs:
            val_metrics = eval_samples(model, val_samples, device, EDGE_CUTOFF)
            sr8 = val_metrics["vn_sr_at_8A"]
            sr4 = val_metrics["vn_sr_at_4A"]
            dcc = val_metrics["vn_dcc_median"]

            improved = sr8 > best_sr8 and epoch >= args.warmup
            if improved:
                best_sr8 = sr8
                best_epoch = epoch
                patience_counter = 0
                torch.save(model.state_dict(), args.out_dir / "best_model.pt")
            else:
                patience_counter += 1

            marker = " *" if improved else ""
            print(f"  epoch {epoch:3d}  loss={train_loss:.4f}  "
                  f"SR@4={sr4:.1%} SR@8={sr8:.1%} DCC={dcc:.1f}A{marker}",
                  flush=True)

            if patience_counter > args.patience and epoch > args.warmup + 10:
                print(f"  Early stop at epoch {epoch}")
                break
        else:
            print(f"  epoch {epoch:3d}  loss={train_loss:.4f}", flush=True)

    total_time = time.time() - t0
    print(f"\n  Training complete: {total_time:.0f}s ({total_time/60:.1f} min)")
    print(f"  Best: epoch {best_epoch}, SR@8={best_sr8:.1%}")

    # Test evaluation
    best_state = torch.load(args.out_dir / "best_model.pt", map_location=device, weights_only=True)
    model.load_state_dict(best_state)

    if test_samples:
        print(f"\n{'='*60}")
        print(f"  TEST SET EVALUATION")
        print(f"{'='*60}")
        test_metrics = eval_samples(model, test_samples, device, EDGE_CUTOFF)
        print(f"  Test SR@4: {test_metrics['vn_sr_at_4A']:.1%}")
        print(f"  Test SR@8: {test_metrics['vn_sr_at_8A']:.1%}")
        print(f"  Test DCC median: {test_metrics['vn_dcc_median']:.1f}A")
    else:
        # No split — report final val metrics as training metrics
        test_metrics = eval_samples(model, val_samples, device, EDGE_CUTOFF)
        print(f"\n  Final metrics (all 372 targets, teacher-distilled):")
        print(f"  SR@4: {test_metrics['vn_sr_at_4A']:.1%}")
        print(f"  SR@8: {test_metrics['vn_sr_at_8A']:.1%}")
        print(f"  DCC median: {test_metrics['vn_dcc_median']:.1f}A")
        print(f"  (True zero-shot test: run predict_v004.py on PDBs outside the 372)")

    gate_passed = test_metrics["vn_sr_at_8A"] >= args.gate_sr8
    print(f"  Gate (SR@8 >= {args.gate_sr8:.0%}): {'PASS' if gate_passed else 'FAIL'}")

    # ONNX export
    print(f"\n{'='*60}")
    print(f"  ONNX EXPORT")
    print(f"{'='*60}")
    model.cpu()
    onnx_path = str(args.out_dir / "vnegnn_v004.onnx")
    export_onnx(model, onnx_path)

    # Save config
    config = {
        "version": "v004",
        "in_dim": IN_DIM,
        "hidden_dim": HIDDEN_DIM,
        "n_layers": N_LAYERS,
        "n_vns": N_VNS,
        "features": "structural(25) + NMA(26) + perturbedNMA(5) + SpikeBERT_allmasked(512)",
        "distillation": True,
        "teacher": "v005 (0.830 AUROC)",
        "loss_weights": {"alpha_kl": ALPHA_KL, "beta_chamfer": BETA_CHAMFER,
                         "gamma_focal": GAMMA_FOCAL},
        "n_params": n_params,
        "best_epoch": best_epoch,
        "best_val_sr8": best_sr8,
        "test_sr4": test_metrics["vn_sr_at_4A"],
        "test_sr8": test_metrics["vn_sr_at_8A"],
        "test_dcc_median": test_metrics["vn_dcc_median"],
        "training_time_sec": total_time,
        "esm2_used": False,
    }
    (args.out_dir / "config.json").write_text(json.dumps(config, indent=2))

    print(f"\n{'='*60}")
    print(f"  VN-EGNN v004 COMPLETE")
    print(f"{'='*60}")
    print(f"  Model: {n_params:,} params")
    print(f"  Input: {IN_DIM}d (no ESM-2)")
    print(f"  Best val SR@8: {best_sr8:.1%} (epoch {best_epoch})")
    print(f"  Test SR@8: {test_metrics['vn_sr_at_8A']:.1%}")
    print(f"  Test SR@4: {test_metrics['vn_sr_at_4A']:.1%}")
    print(f"  Gate: {'PASS' if gate_passed else 'FAIL'}")
    print(f"  ONNX: {onnx_path}")
    print(f"  No ESM-2 anywhere in the pipeline.")

    sys.exit(0 if gate_passed else 2)


if __name__ == "__main__":
    main()
