#!/usr/bin/env python3
"""
SpikeBERT v003 — proprietary PRISM-only zero-shot inference student.

Architecture (least-naive vs v002):
  - Transformer encoder with RMSNorm + SwiGLU + RoPE positional embeddings
  - Modern attention: scaled-dot-product (FlashAttention available via PyTorch)
  - 6 encoder layers, 512-dim hidden, 8 heads, 2048 FFN
  - Input: 104-dim PRISM-derivable per-residue features (no ESM)
  - 11 supervision heads, one per engine signal group

Multi-task supervision (all from engine soft labels):
  - kcc_regression (20 dims)
  - therm_classification + regression (9 dims)
  - asc_consensus prediction (3 dims)
  - gcpid_PID regression (7 dims)
  - phasors_regression (27 dims)
  - phase_manifold prediction (23 dims)
  - stream_stats regression (7 dims)
  - phase_entropy regression (5 dims)
  - druggability regression (4 dims)
  - ground_truth distance regression (where available)
  - p2rank distillation (auxiliary, when available)

Inference (true zero-shot):
  PDB → 104-dim structural input → SpikeBERT v003 → 11 prediction heads → ranked sites.
  No engine run, no spikes, no ESM, no NMA at runtime (NMA can be zero-filled like v002).
"""
from __future__ import annotations
import argparse
import json
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


INPUT_DIM = 103           # structural+NMA per-residue PDB features
# 26 (struct) + 26 (nma) + 5 (pnma) + 25 (struct_ext) + 15 (nma_ext) + 3 (ca) + 3 (sc) = 103
HIDDEN_DIM = 512
N_LAYERS = 6
N_HEADS = 8
FFN_DIM = 2048
DROPOUT = 0.1
MAX_LEN = 1024            # max protein length for positional encoding
AA_VOCAB = 22             # 20 standard AA + X (unknown) + PAD

TARGET_HEAD_DIMS = {
    "kcc": 20,
    "therm": 9,
    "asc": 3,
    "gcpid": 7,
    "phasors": 27,
    "phase_manifold": 23,
    "stream": 7,
    "phase_bit": 5,
    "druggability": 4,
    "ground_truth": 6,
    "p2rank": 5,
}

LOSS_WEIGHTS = {
    "kcc": 1.0,
    "therm": 1.0,
    "asc": 0.5,
    "gcpid": 1.0,
    "phasors": 1.0,
    "phase_manifold": 1.5,
    "stream": 0.5,
    "phase_bit": 0.5,
    "druggability": 0.7,
    "ground_truth": 2.0,
    "p2rank": 0.3,
}


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        norm = x.pow(2).mean(-1, keepdim=True).clamp(min=self.eps).rsqrt()
        return x * norm * self.scale


class SwiGLU(nn.Module):
    """SwiGLU FFN: SiLU(W1·x) ⊙ W2·x → W3"""
    def __init__(self, dim, hidden):
        super().__init__()
        self.w1 = nn.Linear(dim, hidden, bias=False)
        self.w2 = nn.Linear(dim, hidden, bias=False)
        self.w3 = nn.Linear(hidden, dim, bias=False)

    def forward(self, x):
        return self.w3(F.silu(self.w1(x)) * self.w2(x))


class RoPEAttention(nn.Module):
    """Multi-head attention with rotary position embeddings (RoPE)."""
    def __init__(self, dim, n_heads, max_len=MAX_LEN):
        super().__init__()
        assert dim % n_heads == 0
        self.dim = dim
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.qkv = nn.Linear(dim, 3 * dim, bias=False)
        self.proj = nn.Linear(dim, dim, bias=False)

        # Precompute RoPE frequencies
        freqs = 1.0 / (10000 ** (torch.arange(0, self.head_dim, 2).float() / self.head_dim))
        t = torch.arange(max_len).float()
        freqs = torch.outer(t, freqs)
        self.register_buffer("cos_cache", freqs.cos(), persistent=False)
        self.register_buffer("sin_cache", freqs.sin(), persistent=False)

    def rotate(self, x):
        x1, x2 = x[..., 0::2], x[..., 1::2]
        return torch.stack([-x2, x1], dim=-1).flatten(-2)

    def apply_rope(self, x, seq_len):
        cos = self.cos_cache[:seq_len].unsqueeze(0).unsqueeze(0)
        sin = self.sin_cache[:seq_len].unsqueeze(0).unsqueeze(0)
        cos = cos.repeat_interleave(2, dim=-1)
        sin = sin.repeat_interleave(2, dim=-1)
        return x * cos + self.rotate(x) * sin

    def forward(self, x, mask=None):
        B, L, _ = x.shape
        qkv = self.qkv(x).view(B, L, 3, self.n_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        q = self.apply_rope(q, L)
        k = self.apply_rope(k, L)

        attn = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=0.0)
        attn = attn.transpose(1, 2).contiguous().view(B, L, self.dim)
        return self.proj(attn)


class TransformerBlock(nn.Module):
    def __init__(self, dim, n_heads, ffn_dim, dropout=DROPOUT):
        super().__init__()
        self.norm1 = RMSNorm(dim)
        self.attn = RoPEAttention(dim, n_heads)
        self.norm2 = RMSNorm(dim)
        self.ffn = SwiGLU(dim, ffn_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        x = x + self.dropout(self.attn(self.norm1(x), mask=mask))
        x = x + self.dropout(self.ffn(self.norm2(x)))
        return x


class SpikeBERTv003(nn.Module):
    def __init__(self):
        super().__init__()
        self.aa_embed = nn.Embedding(AA_VOCAB, HIDDEN_DIM // 4, padding_idx=AA_VOCAB - 1)
        self.input_proj = nn.Linear(INPUT_DIM + HIDDEN_DIM // 4, HIDDEN_DIM)
        self.input_norm = RMSNorm(HIDDEN_DIM)
        self.dropout = nn.Dropout(DROPOUT)
        self.blocks = nn.ModuleList([
            TransformerBlock(HIDDEN_DIM, N_HEADS, FFN_DIM, DROPOUT)
            for _ in range(N_LAYERS)
        ])
        self.final_norm = RMSNorm(HIDDEN_DIM)
        self.heads = nn.ModuleDict({
            name: nn.Sequential(
                nn.Linear(HIDDEN_DIM, HIDDEN_DIM // 2),
                nn.SiLU(),
                nn.Linear(HIDDEN_DIM // 2, dim),
            )
            for name, dim in TARGET_HEAD_DIMS.items()
        })

    def forward(self, input_features, resname, mask=None):
        aa_emb = self.aa_embed(resname.clamp(0, AA_VOCAB - 1))
        x = torch.cat([input_features, aa_emb], dim=-1)
        x = self.input_proj(x)
        x = self.input_norm(x)
        x = self.dropout(x)
        for blk in self.blocks:
            x = blk(x, mask=mask)
        x = self.final_norm(x)
        return {name: head(x) for name, head in self.heads.items()}

    def extract_embeddings(self, input_features, resname, mask=None):
        """Returns 512-dim hidden representation per residue (for VN-EGNN v005)."""
        aa_emb = self.aa_embed(resname.clamp(0, AA_VOCAB - 1))
        x = torch.cat([input_features, aa_emb], dim=-1)
        x = self.input_proj(x)
        x = self.input_norm(x)
        for blk in self.blocks:
            x = blk(x, mask=mask)
        return self.final_norm(x)


class BundleDataset(Dataset):
    def __init__(self, bundle_dir: Path, target_names: List[str], require_structural: bool = True):
        self.bundles = []
        for name in target_names:
            p = bundle_dir / f"{name}_bundle.npz"
            if not p.exists():
                continue
            d = np.load(p)
            if require_structural and int(d.get("has_structural", 0)) == 0:
                continue
            self.bundles.append({"name": name, "path": str(p)})
        print(f"[dataset] {len(self.bundles)} bundles", file=sys.stderr)

    def __len__(self):
        return len(self.bundles)

    def __getitem__(self, idx):
        meta = self.bundles[idx]
        d = np.load(meta["path"])
        n_res = int(d["n_residues"])

        input_blocks = [
            d["input_structural"], d["input_nma"], d["input_perturbed_nma"],
            d["input_structural_ext"], d["input_nma_ext"],
            d["input_ca_xyz"], d["input_sidechain_xyz"],
        ]
        input_features = np.concatenate([a.astype(np.float32) for a in input_blocks], axis=1)

        resname = d["input_resname"].astype(np.int64)
        resname = np.where(resname < 0, AA_VOCAB - 2, resname)

        targets = {name: d[f"target_{name}"].astype(np.float32) for name in TARGET_HEAD_DIMS.keys()}
        return {
            "name": meta["name"],
            "n_res": n_res,
            "input_features": input_features,
            "resname": resname,
            "targets": targets,
        }


def collate(batch):
    B = len(batch)
    max_len = max(b["n_res"] for b in batch)
    input_dim = batch[0]["input_features"].shape[1]
    features = torch.zeros(B, max_len, input_dim, dtype=torch.float32)
    resname = torch.full((B, max_len), AA_VOCAB - 1, dtype=torch.long)
    pad_mask = torch.ones(B, max_len, dtype=torch.bool)
    targets = {name: torch.zeros(B, max_len, dim, dtype=torch.float32)
               for name, dim in TARGET_HEAD_DIMS.items()}
    names = []
    for i, b in enumerate(batch):
        n = b["n_res"]
        features[i, :n] = torch.from_numpy(b["input_features"])
        resname[i, :n] = torch.from_numpy(b["resname"])
        pad_mask[i, :n] = False
        for k, t in b["targets"].items():
            targets[k][i, :n] = torch.from_numpy(t)
        names.append(b["name"])
    return {"features": features, "resname": resname, "pad_mask": pad_mask,
            "targets": targets, "names": names}


def compute_loss(predictions: Dict[str, torch.Tensor], targets: Dict[str, torch.Tensor],
                 pad_mask: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, float]]:
    not_pad = ~pad_mask
    losses = {}
    total = 0.0
    for name, pred in predictions.items():
        tgt = targets[name]
        if name == "therm":
            cls_logits = pred[..., :1]
            cls_target = tgt[..., 0:1]
            reg_pred = pred[..., 1:]
            reg_target = tgt[..., 1:]
            cls_loss = F.l1_loss(cls_logits[not_pad], cls_target[not_pad])
            reg_loss = F.smooth_l1_loss(reg_pred[not_pad], reg_target[not_pad])
            loss = cls_loss + reg_loss
        elif name == "ground_truth":
            mask_has_gt = (tgt[..., 3:4] > 0).float()
            mask_per_residue = not_pad.unsqueeze(-1) & (mask_has_gt > 0)
            if mask_per_residue.any():
                # Only compute on residues with ground truth available
                loss = F.smooth_l1_loss(
                    pred[mask_per_residue.expand_as(pred)],
                    tgt[mask_per_residue.expand_as(tgt)],
                )
            else:
                loss = torch.tensor(0.0, device=pred.device)
        else:
            loss = F.smooth_l1_loss(pred[not_pad], tgt[not_pad])
        weight = LOSS_WEIGHTS.get(name, 1.0)
        losses[name] = float(loss.item())
        total = total + weight * loss
    return total, losses


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[v003] device={device}", file=sys.stderr)

    bundle_files = sorted(args.bundle_dir.glob("*_bundle.npz"))
    target_names = [p.stem.replace("_bundle", "") for p in bundle_files]

    ds = BundleDataset(args.bundle_dir, target_names, require_structural=True)
    n = len(ds)
    if n == 0:
        print("[v003] no bundles found", file=sys.stderr)
        return
    n_val = max(1, n // 5)
    n_train = n - n_val
    train_ds, val_ds = torch.utils.data.random_split(
        ds, [n_train, n_val], generator=torch.Generator().manual_seed(42)
    )
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate)

    model = SpikeBERTv003().to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[v003] params: {n_params/1e6:.2f}M", file=sys.stderr)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch.amp.GradScaler() if device.type == "cuda" else None

    best_val = float("inf")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    history = []

    for epoch in range(args.epochs):
        model.train()
        train_total = 0.0
        train_n = 0
        per_head = {k: 0.0 for k in TARGET_HEAD_DIMS}
        t0 = time.time()
        for batch in train_loader:
            features = batch["features"].to(device)
            resname = batch["resname"].to(device)
            pad_mask = batch["pad_mask"].to(device)
            tgt = {k: v.to(device) for k, v in batch["targets"].items()}

            optimizer.zero_grad()
            if scaler is not None:
                with torch.amp.autocast(device_type=device.type, dtype=torch.bfloat16):
                    preds = model(features, resname, mask=None)
                    loss, head_losses = compute_loss(preds, tgt, pad_mask)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                preds = model(features, resname, mask=None)
                loss, head_losses = compute_loss(preds, tgt, pad_mask)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            train_total += loss.item() * features.size(0)
            train_n += features.size(0)
            for k, v in head_losses.items():
                per_head[k] += v * features.size(0)

        scheduler.step()
        avg_train = train_total / max(train_n, 1)

        model.eval()
        val_total = 0.0
        val_n = 0
        with torch.no_grad():
            for batch in val_loader:
                features = batch["features"].to(device)
                resname = batch["resname"].to(device)
                pad_mask = batch["pad_mask"].to(device)
                tgt = {k: v.to(device) for k, v in batch["targets"].items()}
                preds = model(features, resname, mask=None)
                loss, _ = compute_loss(preds, tgt, pad_mask)
                val_total += loss.item() * features.size(0)
                val_n += features.size(0)
        avg_val = val_total / max(val_n, 1)
        elapsed = time.time() - t0

        head_str = " ".join(f"{k}={per_head[k]/max(train_n,1):.2f}" for k in TARGET_HEAD_DIMS)
        print(f"epoch {epoch+1:3d}  train={avg_train:.4f}  val={avg_val:.4f}  ({elapsed:.1f}s)", flush=True)
        history.append({"epoch": epoch + 1, "train": avg_train, "val": avg_val,
                        "per_head": {k: per_head[k]/max(train_n,1) for k in TARGET_HEAD_DIMS}})

        if avg_val < best_val:
            best_val = avg_val
            ckpt = args.output_dir / "spike_bert_v003_best.pt"
            torch.save({"model": model.state_dict(),
                        "config": {"INPUT_DIM": INPUT_DIM, "HIDDEN_DIM": HIDDEN_DIM,
                                   "N_LAYERS": N_LAYERS, "N_HEADS": N_HEADS, "FFN_DIM": FFN_DIM,
                                   "TARGET_HEAD_DIMS": TARGET_HEAD_DIMS},
                        "epoch": epoch + 1,
                        "val_loss": avg_val}, ckpt)
            print(f"  saved {ckpt} (val={avg_val:.4f})", flush=True)

    (args.output_dir / "history.json").write_text(json.dumps(history, indent=2))
    print(f"[v003] training complete. best val={best_val:.4f}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle-dir", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--lr", type=float, default=3e-4)
    args = ap.parse_args()
    train(args)


if __name__ == "__main__":
    main()
