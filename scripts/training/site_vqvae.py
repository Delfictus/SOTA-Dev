#!/usr/bin/env python3
"""SiteVQVAE — quality-aware vector-quantized vocabulary for spike windows.

Takes the 18-dim per-window features produced by `temporal_tokenizer.py`
(output: `per_window_features.npz`) and learns a codebook whose discrete
tokens self-organize by binding-site quality. The resulting codebook is
the production vocabulary for SpikeBERT v003 — it supersedes the KMeans
baseline.

Architecture (per directive):

    encoder:      18 → 256 → 128 → latent_dim=64
    codebook:     nn.Embedding(codebook_size=1024, latent_dim=64)
    decoder:      64 → 128 → 256 → 18
    quality_head: 64 → 64 → 1          (DCC-distance regression)
    drug_head:    64 → 64 → 1          (druggability regression)

Training objective:

    L = recon + β · commit + γ · quality_mse + δ · drug_mse + ε · contrastive
                                                                      ↑ pushes EXCELLENT
                                                                        away from POOR
                                                                        in latent space

Each training row is ONE (site, window) pair — i.e. a single 18-dim vector
with its site-level quality labels replicated across all K windows of that
site. The vocabulary is shared across all (site, window) rows.

Inputs
------
    --features-npz    per_window_features.npz from temporal_tokenizer.py
                      (fields: features [S,K,18], site_ids, target_names)
    --labels-json     per-site quality labels
                      (schema: {site_key: {
                          "dcc_distance": float  (Å),
                          "dcc_grade": "EXCELLENT"|"GOOD"|"MARGINAL"|"POOR",
                          "druggability": float  (0-1),
                          "teacher_prob": float  (0-1, optional)}})

Outputs
-------
    vqvae_codebook.npz        codebook embeddings [codebook_size, latent_dim]
                              + input_stats {mean, std} [18]
    vqvae_encoder.pt          encoder + quality + druggability state_dict
    per_site_latents.npz      mean-pooled latent per site [S, latent_dim]
    per_site_tokens.json      {site_id: [tok_0 ... tok_{K-1}]}
    evaluation.json           recon MSE, codebook utilization, grade-pred
                              accuracy, contrastive margin

Ready to feed SpikeBERT v003: `vqvae_codebook.npz` is a drop-in replacement
for KMeans `temporal_vocab.npz`.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from collections import Counter
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

# ─────────────────────────────────────────────────────────────
#  Quality grade mapping
# ─────────────────────────────────────────────────────────────

GRADE_TO_FLOAT = {"EXCELLENT": 0.0, "GOOD": 0.33, "MARGINAL": 0.66, "POOR": 1.0}
GRADE_TO_INT = {"EXCELLENT": 0, "GOOD": 1, "MARGINAL": 2, "POOR": 3}


# ─────────────────────────────────────────────────────────────
#  Model
# ─────────────────────────────────────────────────────────────

class VectorQuantizer(nn.Module):
    """Straight-through VQ layer (Neural Discrete Representation, Oord 2017)."""
    def __init__(self, codebook_size: int, latent_dim: int, beta: float = 0.25):
        super().__init__()
        self.codebook = nn.Embedding(codebook_size, latent_dim)
        # Uniform init in [-1/K, 1/K] — stable for STE
        nn.init.uniform_(self.codebook.weight,
                          -1.0 / codebook_size, 1.0 / codebook_size)
        self.beta = beta
        self.codebook_size = codebook_size
        self.latent_dim = latent_dim

    def forward(self, z_e: torch.Tensor
                ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # z_e: [B, latent_dim]
        # dists to every codebook entry
        with torch.no_grad():
            e = self.codebook.weight        # [K, D]
            dists = ((z_e.unsqueeze(1) - e.unsqueeze(0)) ** 2).sum(dim=-1)  # [B, K]
            token_ids = dists.argmin(dim=-1)         # [B]
        z_q = self.codebook(token_ids)                # [B, D]
        # Commitment + codebook loss
        commit = F.mse_loss(z_e, z_q.detach())
        codebook_loss = F.mse_loss(z_q, z_e.detach())
        vq_loss = codebook_loss + self.beta * commit
        # Straight-through estimator
        z_q_st = z_e + (z_q - z_e).detach()
        return z_q_st, token_ids, vq_loss


class SiteVQVAE(nn.Module):
    def __init__(self, input_dim: int = 18, latent_dim: int = 64,
                 codebook_size: int = 1024, beta: float = 0.25):
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.codebook_size = codebook_size

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 256), nn.SiLU(),
            nn.Linear(256, 128), nn.SiLU(),
            nn.Linear(128, latent_dim),
        )
        self.vq = VectorQuantizer(codebook_size, latent_dim, beta=beta)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128), nn.SiLU(),
            nn.Linear(128, 256), nn.SiLU(),
            nn.Linear(256, input_dim),
        )
        # Quality heads — operate on the quantized latent
        self.quality_head = nn.Sequential(
            nn.Linear(latent_dim, 64), nn.SiLU(),
            nn.Linear(64, 1),
        )
        self.drug_head = nn.Sequential(
            nn.Linear(latent_dim, 64), nn.SiLU(),
            nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor):
        z_e = self.encoder(x)                          # [B, D]
        z_q, token_ids, vq_loss = self.vq(z_e)         # [B, D], [B], scalar
        x_recon = self.decoder(z_q)
        quality = self.quality_head(z_q).squeeze(-1)   # [B]
        drug = self.drug_head(z_q).squeeze(-1)
        return x_recon, token_ids, z_e, z_q, quality, drug, vq_loss


# ─────────────────────────────────────────────────────────────
#  Data
# ─────────────────────────────────────────────────────────────

def load_features(features_npz: Path) -> Dict[str, np.ndarray]:
    d = np.load(features_npz, allow_pickle=False)
    return {"features": d["features"].astype(np.float32),
            "site_ids":  d["site_ids"].astype(str),
            "target_names": d["target_names"].astype(str) if "target_names" in d.files else None}


def load_labels(labels_json: Path) -> Dict[str, Dict[str, float]]:
    """{site_key: {dcc_distance, dcc_grade, druggability, teacher_prob?}}."""
    return json.loads(labels_json.read_text())


def build_training_rows(feats: np.ndarray, site_ids: np.ndarray,
                         labels: Dict[str, Dict]) -> Dict[str, np.ndarray]:
    """Flatten [S,K,18] into [S*K, 18] with per-row labels."""
    S, K, F = feats.shape
    X = feats.reshape(S * K, F)
    # Per-row labels from per-site labels (replicate across K)
    dcc = np.zeros(S * K, dtype=np.float32)
    drug = np.zeros(S * K, dtype=np.float32)
    grade_idx = np.full(S * K, -1, dtype=np.int32)
    for s in range(S):
        sid = str(site_ids[s])
        lab = labels.get(sid, {})
        dcc_val = float(lab.get("dcc_distance", np.nan))
        if np.isnan(dcc_val):
            # Fallback: grade-derived pseudo-distance
            g = lab.get("dcc_grade", "UNKNOWN")
            dcc_val = GRADE_TO_FLOAT.get(g, 0.5) * 10.0
        drug_val = float(lab.get("druggability", 0.5))
        gi = GRADE_TO_INT.get(lab.get("dcc_grade", "UNKNOWN"), -1)
        dcc[s*K:(s+1)*K] = dcc_val
        drug[s*K:(s+1)*K] = drug_val
        grade_idx[s*K:(s+1)*K] = gi
    # Mask zero-feature rows (empty windows) out of training
    nonzero = (X != 0).any(axis=1)
    return {"X": X, "dcc": dcc, "drug": drug, "grade": grade_idx,
            "nonzero_mask": nonzero, "site_of_row": np.repeat(np.arange(S), K)}


# ─────────────────────────────────────────────────────────────
#  Losses
# ─────────────────────────────────────────────────────────────

def contrastive_quality_loss(z: torch.Tensor, grade: torch.Tensor,
                              margin: float = 2.0) -> torch.Tensor:
    """Push EXCELLENT (grade=0) away from POOR (grade=3) latents.

    z: [B, D]   grade: [B] int in {0=EXCELLENT, 3=POOR, or -1 unknown}
    """
    excel_mask = grade == 0
    poor_mask = grade == 3
    if excel_mask.sum() == 0 or poor_mask.sum() == 0:
        return torch.tensor(0.0, device=z.device, dtype=z.dtype)
    ze = z[excel_mask]; zp = z[poor_mask]
    dists = torch.cdist(ze, zp)                     # [Ne, Np]
    # Want dists >= margin; penalise (margin - d) if d < margin
    return F.relu(margin - dists).mean()


# ─────────────────────────────────────────────────────────────
#  Training
# ─────────────────────────────────────────────────────────────

def train(model: SiteVQVAE, rows: Dict[str, np.ndarray],
          input_stats: Dict[str, np.ndarray],
          epochs: int, batch_size: int, lr: float,
          lambda_recon: float, lambda_quality: float, lambda_drug: float,
          lambda_contrastive: float, contrastive_margin: float,
          device: torch.device) -> Dict:
    nonzero = rows["nonzero_mask"]
    Xn = (rows["X"][nonzero] - input_stats["mean"]) / input_stats["std"]
    dcc = rows["dcc"][nonzero]
    drug = rows["drug"][nonzero]
    grade = rows["grade"][nonzero]
    print(f"  train rows: {Xn.shape[0]:,}  "
          f"(EXCELLENT={int((grade==0).sum())}, GOOD={int((grade==1).sum())}, "
          f"MARGINAL={int((grade==2).sum())}, POOR={int((grade==3).sum())}, "
          f"UNK={int((grade==-1).sum())})")

    Xt = torch.from_numpy(Xn).float()
    dcc_t = torch.from_numpy(dcc).float()
    drug_t = torch.from_numpy(drug).float()
    grade_t = torch.from_numpy(grade).long()

    loader = DataLoader(
        TensorDataset(Xt, dcc_t, drug_t, grade_t),
        batch_size=batch_size, shuffle=True, num_workers=0,
        pin_memory=(device.type == "cuda"),
    )
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    history = []
    for ep in range(epochs):
        model.train()
        sums = {"recon": 0.0, "vq": 0.0, "quality": 0.0, "drug": 0.0,
                "contr": 0.0, "total": 0.0}
        n_batches = 0
        tokens_used = set()
        for xb, dccb, drugb, gradeb in loader:
            xb, dccb, drugb, gradeb = (xb.to(device), dccb.to(device),
                                         drugb.to(device), gradeb.to(device))
            opt.zero_grad()
            x_recon, token_ids, z_e, z_q, quality, drug_pred, vq_loss = model(xb)

            loss_recon = F.mse_loss(x_recon, xb)
            loss_quality = F.mse_loss(quality, dccb)
            loss_drug = F.mse_loss(drug_pred, drugb)
            loss_contr = contrastive_quality_loss(z_q, gradeb,
                                                   margin=contrastive_margin)
            loss = (lambda_recon * loss_recon
                    + vq_loss
                    + lambda_quality * loss_quality
                    + lambda_drug * loss_drug
                    + lambda_contrastive * loss_contr)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            sums["recon"] += loss_recon.item()
            sums["vq"] += vq_loss.item()
            sums["quality"] += loss_quality.item()
            sums["drug"] += loss_drug.item()
            sums["contr"] += loss_contr.item()
            sums["total"] += loss.item()
            n_batches += 1
            tokens_used.update(token_ids.tolist())

        sched.step()
        avgs = {k: v / max(n_batches, 1) for k, v in sums.items()}
        avgs["epoch"] = ep + 1
        avgs["tokens_used"] = len(tokens_used)
        avgs["codebook_utilization"] = len(tokens_used) / model.codebook_size
        history.append(avgs)
        if (ep + 1) % 5 == 0 or ep == 0:
            print(f"  ep {ep+1:3d}  recon={avgs['recon']:.4f}  vq={avgs['vq']:.4f}  "
                  f"quality={avgs['quality']:.4f}  drug={avgs['drug']:.4f}  "
                  f"contr={avgs['contr']:.4f}  util={avgs['codebook_utilization']:.2%}",
                  flush=True)
    return {"history": history}


# ─────────────────────────────────────────────────────────────
#  Diagnostics
# ─────────────────────────────────────────────────────────────

def diagnostics(model: SiteVQVAE, rows: Dict[str, np.ndarray],
                 input_stats: Dict[str, np.ndarray],
                 device: torch.device) -> Tuple[Dict, np.ndarray, np.ndarray]:
    """Run full inference pass; return eval metrics, per-row token IDs, per-row z_q."""
    nonzero = rows["nonzero_mask"]
    Xn = (rows["X"] - input_stats["mean"]) / input_stats["std"]
    X_all = torch.from_numpy(Xn).float().to(device)
    model.eval()
    with torch.no_grad():
        x_recon, token_ids, _, z_q, quality, drug_pred, _ = model(X_all)
    token_ids = token_ids.cpu().numpy()
    z_q_np = z_q.cpu().numpy()
    x_recon_np = x_recon.cpu().numpy()
    quality_np = quality.cpu().numpy()
    drug_np = drug_pred.cpu().numpy()

    # Codebook utilization
    used_tokens = set(int(t) for t, nz in zip(token_ids, nonzero) if nz)
    util = len(used_tokens) / model.codebook_size
    dead = model.codebook_size - len(used_tokens)

    # Reconstruction MSE (only nonzero rows)
    recon_mse = float(((x_recon_np[nonzero] - Xn[nonzero]) ** 2).mean())

    # Quality / drug MSE on nonzero
    dcc = rows["dcc"][nonzero]; drug = rows["drug"][nonzero]
    q_mse = float(((quality_np[nonzero] - dcc) ** 2).mean())
    d_mse = float(((drug_np[nonzero] - drug) ** 2).mean())

    # Contrastive separation: mean distance between EXCELLENT and POOR centroids
    grade = rows["grade"]
    excel_ids = np.where((grade == 0) & nonzero)[0]
    poor_ids = np.where((grade == 3) & nonzero)[0]
    contr_info = {"excellent_rows": int(excel_ids.size),
                  "poor_rows": int(poor_ids.size)}
    if excel_ids.size > 0 and poor_ids.size > 0:
        ze = z_q_np[excel_ids].mean(axis=0)
        zp = z_q_np[poor_ids].mean(axis=0)
        contr_info["centroid_distance"] = float(np.linalg.norm(ze - zp))
        # Per-row distance distribution
        mean_e_to_p = np.linalg.norm(
            z_q_np[excel_ids][:, None, :] - z_q_np[poor_ids][None, :, :],
            axis=-1).mean()
        contr_info["mean_pairwise_distance"] = float(mean_e_to_p)

    # Token → majority grade mapping
    token_to_grade_counter: Dict[int, Counter] = {}
    for tid, g, nz in zip(token_ids, grade, nonzero):
        if not nz or g < 0:
            continue
        token_to_grade_counter.setdefault(int(tid), Counter())[int(g)] += 1
    token_purity = []
    for tid, c in token_to_grade_counter.items():
        total = sum(c.values()); best = max(c.values())
        token_purity.append(best / total)
    mean_purity = float(np.mean(token_purity)) if token_purity else 0.0

    return {
        "recon_mse": recon_mse,
        "codebook_utilization": util,
        "dead_tokens": dead,
        "tokens_used": len(used_tokens),
        "quality_mse": q_mse,
        "drug_mse": d_mse,
        "contrastive": contr_info,
        "token_grade_purity_mean": mean_purity,
        "token_grade_purity_n": len(token_purity),
    }, token_ids, z_q_np


# ─────────────────────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--features-npz", type=Path, required=True,
                        help="Output of temporal_tokenizer.py: per_window_features.npz")
    parser.add_argument("--labels-json", type=Path, required=True,
                        help="Per-site quality labels JSON")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--latent-dim", type=int, default=64)
    parser.add_argument("--codebook-size", type=int, default=1024)
    parser.add_argument("--beta", type=float, default=0.25,
                        help="VQ commitment weight")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--lambda-recon", type=float, default=1.0)
    parser.add_argument("--lambda-quality", type=float, default=0.1)
    parser.add_argument("--lambda-drug", type=float, default=0.1)
    parser.add_argument("--lambda-contrastive", type=float, default=0.5)
    parser.add_argument("--contrastive-margin", type=float, default=2.0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if not HAVE_TORCH:
        print("ERROR: torch not installed"); sys.exit(1)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    print(f"Device: {device}")

    feats = load_features(args.features_npz)
    S, K, F = feats["features"].shape
    print(f"Input features: [S={S}, K={K}, F={F}]")

    labels = load_labels(args.labels_json)
    print(f"Labels loaded: {len(labels)} site entries")

    rows = build_training_rows(feats["features"], feats["site_ids"], labels)
    print(f"Flat rows: {rows['X'].shape}  (nonzero: {int(rows['nonzero_mask'].sum())})")

    X_fit = rows["X"][rows["nonzero_mask"]]
    input_stats = {"mean": X_fit.mean(axis=0).astype(np.float32),
                    "std":  np.maximum(X_fit.std(axis=0), 1e-6).astype(np.float32)}
    (args.output_dir / "input_stats.json").write_text(json.dumps({
        "mean": input_stats["mean"].tolist(),
        "std":  input_stats["std"].tolist(),
        "feature_dim": int(F),
    }))

    model = SiteVQVAE(input_dim=F, latent_dim=args.latent_dim,
                       codebook_size=args.codebook_size, beta=args.beta).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"\nSiteVQVAE: {n_params:,} params  "
          f"latent={args.latent_dim}  codebook={args.codebook_size}")

    t0 = time.time()
    info = train(model, rows, input_stats,
                  epochs=args.epochs, batch_size=args.batch_size, lr=args.lr,
                  lambda_recon=args.lambda_recon,
                  lambda_quality=args.lambda_quality,
                  lambda_drug=args.lambda_drug,
                  lambda_contrastive=args.lambda_contrastive,
                  contrastive_margin=args.contrastive_margin,
                  device=device)
    train_time = time.time() - t0

    diag, token_ids_flat, z_q_flat = diagnostics(model, rows, input_stats, device)
    # Reshape tokens back to [S, K]
    token_ids_sk = token_ids_flat.reshape(S, K)
    # Mean-pool latent per site
    z_q_sk = z_q_flat.reshape(S, K, args.latent_dim)
    site_latent = z_q_sk.mean(axis=1)

    # Save outputs
    np.savez_compressed(args.output_dir / "vqvae_codebook.npz",
                         codebook=model.vq.codebook.weight.detach().cpu().numpy().astype(np.float32),
                         mean=input_stats["mean"], std=input_stats["std"],
                         codebook_size=args.codebook_size,
                         latent_dim=args.latent_dim)
    torch.save({
        "encoder": model.encoder.state_dict(),
        "decoder": model.decoder.state_dict(),
        "vq": model.vq.state_dict(),
        "quality_head": model.quality_head.state_dict(),
        "drug_head": model.drug_head.state_dict(),
        "input_dim": F, "latent_dim": args.latent_dim,
        "codebook_size": args.codebook_size,
    }, args.output_dir / "vqvae_encoder.pt")
    np.savez_compressed(args.output_dir / "per_site_latents.npz",
                         latents=site_latent.astype(np.float32),
                         site_ids=feats["site_ids"])
    tokens_json = {str(feats["site_ids"][i]): token_ids_sk[i].tolist()
                   for i in range(S)}
    (args.output_dir / "per_site_tokens.json").write_text(json.dumps(tokens_json, indent=2))

    # Final eval report
    eval_out = {
        "n_sites": int(S),
        "windows_per_site": int(K),
        "feature_dim": int(F),
        "n_params": n_params,
        "train_time_sec": train_time,
        "diagnostics": diag,
        "history": info["history"],
        "config": vars(args),
    }
    (args.output_dir / "evaluation.json").write_text(
        json.dumps(eval_out, indent=2, default=str))

    # Terse final print
    print(f"\n{'='*64}\nSiteVQVAE — final diagnostics\n{'='*64}")
    print(f"  recon MSE (norm space):      {diag['recon_mse']:.4f}")
    print(f"  codebook utilization:        {diag['codebook_utilization']:.2%}  "
          f"({diag['tokens_used']}/{args.codebook_size})")
    print(f"  dead tokens:                 {diag['dead_tokens']}")
    print(f"  quality-head MSE:            {diag['quality_mse']:.4f}")
    print(f"  drug-head MSE:               {diag['drug_mse']:.4f}")
    print(f"  token→grade purity (mean):   {diag['token_grade_purity_mean']:.3f}")
    ci = diag["contrastive"]
    if "centroid_distance" in ci:
        print(f"  EXCELLENT↔POOR centroid dist: {ci['centroid_distance']:.3f}  "
              f"(pairwise mean {ci['mean_pairwise_distance']:.3f})  "
              f"[{ci['excellent_rows']} vs {ci['poor_rows']} rows]")
    else:
        print("  contrastive separation:      insufficient labels "
              f"(EXCELLENT={ci.get('excellent_rows',0)}, POOR={ci.get('poor_rows',0)})")
    print(f"  outputs in: {args.output_dir}")


if __name__ == "__main__":
    main()
