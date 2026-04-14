#!/usr/bin/env python3
"""VN-EGNN v001 training on pct95 data.

Trains the Virtual-Node E(3)-Equivariant GNN (`model.py`) using the feature
bundles produced by `feature_extractor.py`. Inputs per target:

    npz file at {features_dir}/<target>.features.npz with keys:
      X             [N, FEATURE_DIM] float32  — full per-residue feature matrix
      ca_coords     [N, 3]                    — Cα positions
      labels        [N] int                    — binding labels (Cα within 4.5 Å)
      ligand_centroid [3]                      — for Chamfer VN loss

Training:
  • Split by TARGET (not residue): 80% train / 15% val / 5% test
  • Loss = atom focal BCE + Chamfer VN loss (+ optional teacher distillation
    if --teacher-onnx provided)
  • Edge construction: protein–protein within 8 Å + full VN↔protein bipartite
  • Early-stop on val VN-DCC median (5-epoch patience)

Outputs (to --out-dir):
  vn_egnn_v001.pt     best model weights
  vn_egnn_v001.onnx   ONNX export (dynamic N, E)
  evaluation.json     per-target VN-DCC + atom AUROC
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    HAVE_TORCH = True
except ImportError:
    HAVE_TORCH = False

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

from model import (VNEGNN, build_edge_index, init_virtual_nodes,
                   focal_binary_loss, chamfer_vn_loss, export_onnx)
from feature_extractor import load_bundle, FEATURE_DIM
from cluster_split import cluster_split_bundles  # homolog-safe splits

API_BASE = os.environ.get("PRISM_API", "https://prism-feature-pipeline.is-0b9.workers.dev")


def _api_get(url: str, timeout: int = 60):
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 prism4d-vnegnn-v001"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


# ─────────────────────────────────────────────────────────────
#  Target materialization
# ─────────────────────────────────────────────────────────────

class TargetSample:
    __slots__ = ("target", "X", "ca", "labels", "centroid")

    def __init__(self, target: str, X: np.ndarray, ca: np.ndarray,
                 labels: np.ndarray, centroid: np.ndarray):
        self.target = target
        self.X = X                # [N, FEATURE_DIM]
        self.ca = ca              # [N, 3]
        self.labels = labels      # [N]
        self.centroid = centroid  # [3]


def _build_x_from_bundle(b: Dict[str, np.ndarray]) -> np.ndarray:
    """Assemble VN-EGNN input matrix: structural+NMA+perturbed+physics+tide+temporal+esm.

    Layout (input_dim up to 25+26+5+216+7+2+1280 = 1561):
      structural [N,25] + nma [N,26] + perturbed_nma [N,5] + physics_216 [N,216]
      + tide_residue [N,7] + temporal [N,2] + esm2 [N,1280]

    esm2 is added by the RunPod Phase 2 step; absent in dev runs without GPU.
    """
    blocks = []
    for k in ("structural", "nma", "perturbed_nma", "physics_216",
              "tide_residue", "temporal", "esm2"):
        if k in b:
            blocks.append(b[k].astype(np.float32))
    if not blocks and "X" in b:
        return b["X"].astype(np.float32)
    return np.concatenate(blocks, axis=1)


def load_samples(features_dir: Path, targets: List[str],
                 min_residues: int = 30,
                 label_cutoff: Optional[float] = None) -> List[TargetSample]:
    out: List[TargetSample] = []
    for t in targets:
        for stem in (f"{t}_features.npz", f"{t}.features.npz"):
            p = features_dir / stem
            if p.exists():
                break
        else:
            continue
        d = load_bundle(p)
        X = _build_x_from_bundle(d)
        if X.shape[0] < min_residues:
            continue
        ca_key = "coords" if "coords" in d else "ca_coords"
        ca = d[ca_key].astype(np.float32)
        centroid = d["ligand_centroid"].astype(np.float32)
        if label_cutoff is not None:
            labels = (np.linalg.norm(ca - centroid, axis=1) <= label_cutoff
                      ).astype(np.float32)
        else:
            labels = d["labels"].astype(np.float32)
        out.append(TargetSample(target=t, X=X, ca=ca, labels=labels, centroid=centroid))
    return out


def fetch_valid_targets(exclude_grade: str = "POOR") -> List[str]:
    data = _api_get(f"{API_BASE}/dcc")
    return sorted({r["target"] for r in data.get("records", [])
                   if r.get("dcc_grade") != exclude_grade})


# ─────────────────────────────────────────────────────────────
#  Training
# ─────────────────────────────────────────────────────────────

def to_device(sample: TargetSample, device: torch.device) -> Dict[str, torch.Tensor]:
    return {
        "target": sample.target,
        "atom_features": torch.from_numpy(sample.X).to(device),
        "atom_coords":   torch.from_numpy(sample.ca).to(device),
        "atom_labels":   torch.from_numpy(sample.labels).to(device),
        "ligand_centroid": torch.from_numpy(sample.centroid).to(device),
    }


def train_epoch(model: VNEGNN, samples: List[Dict[str, torch.Tensor]],
                opt: torch.optim.Optimizer, cutoff: float,
                atom_w: float = 1.0, vn_w: float = 1.0) -> Dict[str, float]:
    model.train()
    total_a, total_v, n = 0.0, 0.0, 0
    for s in samples:
        edge_index = build_edge_index(s["atom_coords"],
                                      n_virtual_nodes=model.n_virtual_nodes,
                                      cutoff=cutoff)
        vn_init = init_virtual_nodes(s["atom_coords"], model.n_virtual_nodes)

        opt.zero_grad()
        atom_logits, vn_coords, vn_conf = model(
            s["atom_features"], s["atom_coords"], edge_index, None, vn_init)

        l_atom = focal_binary_loss(atom_logits.squeeze(-1), s["atom_labels"])
        l_vn = chamfer_vn_loss(vn_coords, s["ligand_centroid"].unsqueeze(0), vn_conf)
        loss = atom_w * l_atom + vn_w * l_vn
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        opt.step()

        total_a += l_atom.item()
        total_v += l_vn.item()
        n += 1
    return {"atom_loss": total_a / max(n, 1), "vn_loss": total_v / max(n, 1)}


def eval_samples(model: VNEGNN, samples: List[Dict[str, torch.Tensor]],
                 cutoff: float) -> Dict[str, Any]:
    model.eval()
    per_target = []
    with torch.no_grad():
        for s in samples:
            edge_index = build_edge_index(s["atom_coords"],
                                          n_virtual_nodes=model.n_virtual_nodes,
                                          cutoff=cutoff)
            vn_init = init_virtual_nodes(s["atom_coords"], model.n_virtual_nodes)
            atom_logits, vn_coords, vn_conf = model(
                s["atom_features"], s["atom_coords"], edge_index, None, vn_init)
            vn_dists = torch.linalg.norm(vn_coords - s["ligand_centroid"].unsqueeze(0),
                                         dim=-1)
            min_vn = float(vn_dists.min().item())

            lab = s["atom_labels"].cpu().numpy()
            prob = torch.sigmoid(atom_logits.squeeze(-1)).cpu().numpy()
            auroc = None
            if lab.sum() > 0 and lab.sum() < len(lab):
                try:
                    from sklearn.metrics import roc_auc_score
                    auroc = float(roc_auc_score(lab, prob))
                except Exception:
                    pass

            per_target.append({"target": s["target"], "min_vn_dist": min_vn,
                               "atom_auroc": auroc, "n_positive": int(lab.sum())})

    dists = [r["min_vn_dist"] for r in per_target]
    aurocs = [r["atom_auroc"] for r in per_target if r["atom_auroc"] is not None]
    return {
        "n_targets": len(per_target),
        "vn_dcc_median": float(np.median(dists)) if dists else float("nan"),
        "vn_dcc_mean": float(np.mean(dists)) if dists else float("nan"),
        "vn_sr_at_4A": sum(1 for d in dists if d < 4.0) / max(len(dists), 1),
        "vn_sr_at_8A": sum(1 for d in dists if d < 8.0) / max(len(dists), 1),
        "atom_auroc_mean": float(np.mean(aurocs)) if aurocs else float("nan"),
        "per_target": per_target,
    }


# ─────────────────────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--features-dir", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--n-layers", type=int, default=4)
    parser.add_argument("--n-vns", type=int, default=16)
    parser.add_argument("--edge-cutoff", type=float, default=8.0)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--val-fraction", type=float, default=0.15,
                        help="Fraction of CLUSTERS (not targets) in val")
    parser.add_argument("--test-fraction", type=float, default=0.05,
                        help="Fraction of CLUSTERS (not targets) in test")
    parser.add_argument("--min-seq-id", type=float, default=0.3,
                        help="MMseqs2 clustering identity threshold")
    parser.add_argument("--cluster-cache-path", type=Path,
                        default=Path("/mnt/storage/spike-audit/seq_clusters.json"),
                        help="Cache for the MMseqs2 cluster map (JSON)")
    parser.add_argument("--exclude-grade", default="POOR")
    parser.add_argument("--label-cutoff", type=float, default=None,
                        help="Relabel at Cα-to-ligand ≤ this (Å); unset = use .npz labels")
    parser.add_argument("--gate-sr8", type=float, default=0.55,
                        help="Min SR@8Å on test set")
    args = parser.parse_args()

    if not HAVE_TORCH:
        print("ERROR: torch not installed"); sys.exit(1)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # 1) Valid targets
    valid = fetch_valid_targets(exclude_grade=args.exclude_grade)
    print(f"Valid targets (D1 excl {args.exclude_grade}): {len(valid)}")

    # 2) Load feature bundles
    samples = load_samples(args.features_dir, valid, label_cutoff=args.label_cutoff)
    print(f"Usable samples: {len(samples)}")
    if len(samples) < 50:
        print("ERROR: <50 usable samples"); sys.exit(1)

    if samples[0].X.shape[1] != FEATURE_DIM:
        print(f"WARN: sample dim {samples[0].X.shape[1]} != FEATURE_DIM {FEATURE_DIM}")

    # 3) Sequence-cluster split — homologs (≥30% seq-id) kept in a single
    #    split to prevent evaluation leakage. MMseqs2 runs once; result is
    #    cached at args.cluster_cache_path.
    sample_targets = [s.target for s in samples]
    train_names, val_names, test_names = cluster_split_bundles(
        bundle_dir=args.features_dir,
        targets=sample_targets,
        val_frac=args.val_fraction,
        test_frac=args.test_fraction,
        min_seq_id=args.min_seq_id,
        cache_path=args.cluster_cache_path,
    )
    train_set, val_set, test_set = set(train_names), set(val_names), set(test_names)
    train_s = [s for s in samples if s.target in train_set]
    val_s   = [s for s in samples if s.target in val_set]
    test_s  = [s for s in samples if s.target in test_set]
    print(f"  Train: {len(train_s)}  Val: {len(val_s)}  Test: {len(test_s)}")

    # Move to device (feature tensors already built once, reused per epoch)
    train_d = [to_device(s, device) for s in train_s]
    val_d = [to_device(s, device) for s in val_s]
    test_d = [to_device(s, device) for s in test_s]

    # 4) Model
    in_dim = samples[0].X.shape[1]
    # edge_feat_dim=0 because train_epoch/eval pass edge_feat=None — the
    # EGNNLayer concat must match. Default of 1 caused a mat-mul size error.
    model = VNEGNN(in_dim=in_dim, hidden_dim=args.hidden_dim,
                   n_layers=args.n_layers, n_virtual_nodes=args.n_vns,
                   edge_feat_dim=0).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {n_params:,} params  (in_dim={in_dim})")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    best_vn_dcc = float("inf")
    best_state = None
    patience = 0
    t0 = time.time()
    epoch_rng = np.random.default_rng(42)

    for ep in range(args.epochs):
        epoch_rng.shuffle(train_d)
        tr = train_epoch(model, train_d, opt, cutoff=args.edge_cutoff)
        sched.step()
        ev = eval_samples(model, val_d, cutoff=args.edge_cutoff)
        line = (f"ep {ep+1:3d} | atom={tr['atom_loss']:.3f} vn={tr['vn_loss']:.3f} "
                f"| val VN-DCC med={ev['vn_dcc_median']:.2f}Å SR@4={ev['vn_sr_at_4A']:.1%} "
                f"SR@8={ev['vn_sr_at_8A']:.1%}")
        if np.isfinite(ev["atom_auroc_mean"]):
            line += f" atom_AUROC={ev['atom_auroc_mean']:.3f}"
        print(line, flush=True)

        if ev["vn_dcc_median"] < best_vn_dcc:
            best_vn_dcc = ev["vn_dcc_median"]
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            torch.save(best_state, args.out_dir / "vn_egnn_v001.pt")
            patience = 0
        else:
            patience += 1
            if patience >= args.patience:
                print(f"  Early stop at epoch {ep+1}")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    # 5) Final eval on test
    test_ev = eval_samples(model, test_d, cutoff=args.edge_cutoff)
    val_ev = eval_samples(model, val_d, cutoff=args.edge_cutoff)

    print(f"\n{'='*60}\n  VN-EGNN v001 — final\n{'='*60}")
    for label, ev in [("VAL", val_ev), ("TEST", test_ev)]:
        print(f"  {label}: VN-DCC med={ev['vn_dcc_median']:.2f}Å  "
              f"SR@4={ev['vn_sr_at_4A']:.1%}  SR@8={ev['vn_sr_at_8A']:.1%}  "
              f"atom_AUROC={ev['atom_auroc_mean']:.3f}")

    passed = test_ev["vn_sr_at_8A"] >= args.gate_sr8
    print(f"  Gate (test SR@8Å ≥ {args.gate_sr8:.2f}): {'PASS' if passed else 'FAIL'}")

    (args.out_dir / "evaluation.json").write_text(json.dumps({
        "in_dim": in_dim,
        "n_params": n_params,
        "training_time_sec": time.time() - t0,
        "val": val_ev, "test": test_ev,
        "best_val_vn_dcc_median": float(best_vn_dcc),
        "passed": passed,
        "gate_sr8": args.gate_sr8,
    }, indent=2, default=str))

    # 6) ONNX export (run on CPU for portability)
    print("\nExporting ONNX...")
    model.cpu().eval()
    export_onnx(model, str(args.out_dir / "vn_egnn_v001.onnx"),
                example_n_atoms=300, opset=17)

    print(f"\n  Outputs in: {args.out_dir}")
    sys.exit(0 if passed else 2)


if __name__ == "__main__":
    main()
