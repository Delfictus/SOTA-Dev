#!/usr/bin/env python3
"""SpikeBERT — Masked spike-dynamics language model.

Learns the grammar of neuromorphic MD spike patterns across protein structures.
No ESM-2 dependency. Your 3.6B spike events ARE the language model.

Tokenizer: KMeans(2048) on 333-dim per-residue spike features
  channel_features(108) + physics_216(216) + tide_residue(7) + temporal(2)

Architecture: BERT-style transformer encoder
  Input: spike_token_embed(2049, 512) + structural_proj(56, 512)
  Encoder: 6 layers, 8 heads, 512 hidden, 2048 FFN
  MLM head: LayerNorm → Linear(512, 2048)

Zero-shot inference:
  Raw PDB → structural(56) → SpikeBERT(all-masked) → 512-dim/residue → VN-EGNN

Training: 372 targets × ~260 residues = 96K tokens. MLM 15% masking.
  ~1.5 hrs on H100, minutes on A100.

Usage:
  # Local smoke test
  python3 scripts/training/spike_bert.py --epochs 5 --max-targets 50

  # RunPod full training
  python3 scripts/training/spike_bert.py --epochs 200 --batch-size 32

  # Export embeddings for VN-EGNN
  python3 scripts/training/spike_bert.py --export-embeddings --checkpoint model.pt
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# ─────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────

VOCAB_SIZE = 2048
SPIKE_DIM = 333       # channel(108) + physics_216(216) + tide(7) + temporal(2)
STRUCT_DIM = 56        # structural(25) + nma(26) + perturbed_nma(5)
HIDDEN_DIM = 512
N_LAYERS = 6
N_HEADS = 8
FFN_DIM = 2048
DROPOUT = 0.1
MASK_PROB = 0.15
MASK_TOKEN_ID = VOCAB_SIZE  # 2048 = [MASK]
PAD_TOKEN_ID = VOCAB_SIZE + 1  # 2049 = [PAD]
TOTAL_VOCAB = VOCAB_SIZE + 2   # 2050

BUNDLE_DIR = Path(os.environ.get("BUNDLE_DIR", "/mnt/storage/spike-audit/features-pct95"))
OUT_DIR = Path(os.environ.get("OUT_DIR", "/mnt/storage/spike-audit/spike-bert"))

SPIKE_KEYS = ("channel_features", "physics_216", "tide_residue", "temporal")
STRUCT_KEYS = ("structural", "nma", "perturbed_nma")


# ─────────────────────────────────────────────────────────────
# Tokenizer: KMeans on 333-dim spike features
# ─────────────────────────────────────────────────────────────

def build_tokenizer(bundle_dir: Path, targets: List[str],
                    n_clusters: int = VOCAB_SIZE,
                    max_residues: int = 100_000,
                    cache_path: Optional[Path] = None,
                    ) -> np.ndarray:
    """Fit KMeans on spike features. Returns centroids [n_clusters, 333]."""
    if cache_path and cache_path.exists():
        data = np.load(cache_path)
        centroids = data["centroids"]
        print(f"[tokenizer] loaded cached centroids: {centroids.shape}")
        return centroids

    print(f"[tokenizer] collecting spike features from {len(targets)} targets...")
    all_feats = []
    for t in targets:
        p = bundle_dir / f"{t}_features.npz"
        if not p.exists():
            continue
        d = np.load(p, allow_pickle=False)
        blocks = []
        for k in SPIKE_KEYS:
            if k in d:
                blocks.append(d[k].astype(np.float32))
        if blocks:
            all_feats.append(np.concatenate(blocks, axis=1))

    X = np.concatenate(all_feats, axis=0)
    if len(X) > max_residues:
        idx = np.random.default_rng(42).choice(len(X), max_residues, replace=False)
        X = X[idx]
    print(f"[tokenizer] fitting KMeans({n_clusters}) on {X.shape} features...")

    from sklearn.cluster import MiniBatchKMeans
    km = MiniBatchKMeans(n_clusters=n_clusters, batch_size=4096,
                         random_state=42, n_init=3, max_iter=300)
    km.fit(X)
    centroids = km.cluster_centers_.astype(np.float32)
    inertia = km.inertia_

    if cache_path:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(cache_path, centroids=centroids, inertia=inertia)
        print(f"[tokenizer] cached to {cache_path}")

    print(f"[tokenizer] done. inertia={inertia:.1f}")
    return centroids


def tokenize_residues(bundle: Dict[str, np.ndarray],
                      centroids: np.ndarray) -> np.ndarray:
    """Assign each residue to nearest centroid. Returns [n_residues] int array."""
    blocks = []
    for k in SPIKE_KEYS:
        if k in bundle:
            blocks.append(bundle[k].astype(np.float32))
    X = np.concatenate(blocks, axis=1)
    # Vectorized nearest centroid
    dists = np.linalg.norm(X[:, None, :] - centroids[None, :, :], axis=2)
    return dists.argmin(axis=1).astype(np.int32)


def get_structural_features(bundle: Dict[str, np.ndarray]) -> np.ndarray:
    """Extract 56-dim structural context (always available from PDB)."""
    blocks = []
    for k in STRUCT_KEYS:
        if k in bundle:
            blocks.append(bundle[k].astype(np.float32))
    return np.concatenate(blocks, axis=1)


# ─────────────────────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────────────────────

class SpikeMLMDataset(Dataset):
    def __init__(self, bundle_dir: Path, targets: List[str],
                 centroids: np.ndarray):
        self.proteins = []
        for t in targets:
            p = bundle_dir / f"{t}_features.npz"
            if not p.exists():
                continue
            d = np.load(p, allow_pickle=True)
            tokens = tokenize_residues(d, centroids)
            struct = get_structural_features(d)
            coords = d["coords"].astype(np.float32) if "coords" in d else None
            self.proteins.append({
                "target": t,
                "tokens": tokens,
                "struct": struct,
                "coords": coords,
                "n_residues": len(tokens),
            })
        print(f"[dataset] {len(self.proteins)} proteins, "
              f"{sum(p['n_residues'] for p in self.proteins):,} residues")

    def __len__(self):
        return len(self.proteins)

    def __getitem__(self, idx):
        return self.proteins[idx]


def collate_mlm(batch, mask_prob=MASK_PROB):
    """Pad proteins to same length, apply MLM masking."""
    max_len = max(p["n_residues"] for p in batch)
    B = len(batch)

    input_ids = torch.full((B, max_len), PAD_TOKEN_ID, dtype=torch.long)
    labels = torch.full((B, max_len), -100, dtype=torch.long)
    struct_feats = torch.zeros(B, max_len, STRUCT_DIM, dtype=torch.float32)
    pad_mask = torch.ones(B, max_len, dtype=torch.bool)
    targets_list = []

    for i, p in enumerate(batch):
        n = p["n_residues"]
        tokens = torch.tensor(p["tokens"], dtype=torch.long)
        input_ids[i, :n] = tokens
        struct_feats[i, :n] = torch.tensor(p["struct"], dtype=torch.float32)
        pad_mask[i, :n] = False
        targets_list.append(p["target"])

        # MLM masking
        mask = torch.rand(n) < mask_prob
        if mask.sum() == 0:
            mask[torch.randint(n, (1,))] = True
        labels[i, :n][mask] = tokens[mask]
        # 80% [MASK], 10% random, 10% keep
        probs = torch.rand(mask.sum())
        mask_indices = mask.nonzero(as_tuple=True)[0]
        replace_mask = probs < 0.8
        random_mask = (probs >= 0.8) & (probs < 0.9)
        input_ids[i, mask_indices[replace_mask]] = MASK_TOKEN_ID
        input_ids[i, mask_indices[random_mask]] = torch.randint(VOCAB_SIZE, (random_mask.sum(),))

    return {
        "input_ids": input_ids,
        "struct_feats": struct_feats,
        "pad_mask": pad_mask,
        "labels": labels,
        "targets": targets_list,
    }


# ─────────────────────────────────────────────────────────────
# Model
# ─────────────────────────────────────────────────────────────

class SpikeBERT(nn.Module):
    def __init__(self, vocab_size=TOTAL_VOCAB, hidden_dim=HIDDEN_DIM,
                 struct_dim=STRUCT_DIM, n_layers=N_LAYERS, n_heads=N_HEADS,
                 ffn_dim=FFN_DIM, dropout=DROPOUT, max_len=2048):
        super().__init__()
        self.token_embed = nn.Embedding(vocab_size, hidden_dim, padding_idx=PAD_TOKEN_ID)
        self.struct_proj = nn.Linear(struct_dim, hidden_dim)
        self.pos_embed = nn.Embedding(max_len, hidden_dim)
        self.input_norm = nn.LayerNorm(hidden_dim)
        self.input_dropout = nn.Dropout(dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=n_heads, dim_feedforward=ffn_dim,
            dropout=dropout, activation="gelu", batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        self.mlm_norm = nn.LayerNorm(hidden_dim)
        self.mlm_head = nn.Linear(hidden_dim, VOCAB_SIZE)

        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_normal_(p)

    def forward(self, input_ids, struct_feats, pad_mask, labels=None):
        B, L = input_ids.shape
        pos = torch.arange(L, device=input_ids.device).unsqueeze(0)

        x = self.token_embed(input_ids) + self.struct_proj(struct_feats) + self.pos_embed(pos)
        x = self.input_norm(x)
        x = self.input_dropout(x)

        x = self.encoder(x, src_key_padding_mask=pad_mask)

        logits = self.mlm_head(self.mlm_norm(x))

        loss = None
        if labels is not None:
            loss = F.cross_entropy(logits.view(-1, VOCAB_SIZE), labels.view(-1),
                                   ignore_index=-100)
        return {"loss": loss, "logits": logits, "hidden_states": x}

    def extract_embeddings(self, struct_feats, pad_mask):
        """Zero-shot: all tokens masked, predict from structure alone."""
        B, L, _ = struct_feats.shape
        input_ids = torch.full((B, L), MASK_TOKEN_ID, dtype=torch.long,
                               device=struct_feats.device)
        # Set padding positions
        input_ids[pad_mask] = PAD_TOKEN_ID
        out = self.forward(input_ids, struct_feats, pad_mask)
        return out["hidden_states"]


# ─────────────────────────────────────────────────────────────
# Training
# ─────────────────────────────────────────────────────────────

def train_epoch(model, loader, optimizer, scheduler, device, epoch):
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_masked = 0
    t0 = time.time()

    for batch in loader:
        ids = batch["input_ids"].to(device)
        sf = batch["struct_feats"].to(device)
        pm = batch["pad_mask"].to(device)
        labels = batch["labels"].to(device)

        out = model(ids, sf, pm, labels)
        loss = out["loss"]

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if scheduler:
            scheduler.step()

        total_loss += loss.item() * ids.size(0)

        mask = labels != -100
        if mask.any():
            preds = out["logits"].argmax(dim=-1)
            total_correct += (preds[mask] == labels[mask]).sum().item()
            total_masked += mask.sum().item()

    n = len(loader.dataset)
    avg_loss = total_loss / max(n, 1)
    acc = total_correct / max(total_masked, 1) * 100
    elapsed = time.time() - t0
    print(f"  epoch {epoch:3d}  loss={avg_loss:.4f}  mlm_acc={acc:.1f}%  "
          f"({elapsed:.1f}s)", flush=True)
    return avg_loss, acc


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_masked = 0

    for batch in loader:
        ids = batch["input_ids"].to(device)
        sf = batch["struct_feats"].to(device)
        pm = batch["pad_mask"].to(device)
        labels = batch["labels"].to(device)

        out = model(ids, sf, pm, labels)
        total_loss += out["loss"].item() * ids.size(0)

        mask = labels != -100
        if mask.any():
            preds = out["logits"].argmax(dim=-1)
            total_correct += (preds[mask] == labels[mask]).sum().item()
            total_masked += mask.sum().item()

    n = len(loader.dataset)
    return total_loss / max(n, 1), total_correct / max(total_masked, 1) * 100


def export_all_embeddings(model, bundle_dir, targets, centroids, out_dir, device):
    """Extract 512-dim spike embeddings for every residue in every target."""
    model.eval()
    out_dir.mkdir(parents=True, exist_ok=True)

    for i, t in enumerate(targets):
        p = bundle_dir / f"{t}_features.npz"
        if not p.exists():
            continue
        d = np.load(p, allow_pickle=True)
        struct = get_structural_features(d)
        n = struct.shape[0]

        sf = torch.tensor(struct, dtype=torch.float32).unsqueeze(0).to(device)
        pm = torch.zeros(1, n, dtype=torch.bool, device=device)

        with torch.no_grad():
            emb = model.extract_embeddings(sf, pm)
        emb_np = emb.squeeze(0).cpu().numpy()

        np.savez_compressed(out_dir / f"{t}_spike_emb.npz",
                            spike_embedding=emb_np,
                            residue_ids=d.get("residue_ids", np.arange(n)))
        if (i + 1) % 50 == 0 or (i + 1) == len(targets):
            print(f"  [{i+1}/{len(targets)}] exported", flush=True)

    print(f"  embeddings saved to {out_dir}")


# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="SpikeBERT — spike dynamics language model")
    parser.add_argument("--bundle-dir", type=Path, default=BUNDLE_DIR)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--warmup-steps", type=int, default=500)
    parser.add_argument("--max-targets", type=int, default=0)
    parser.add_argument("--val-frac", type=float, default=0.1)
    parser.add_argument("--hidden-dim", type=int, default=HIDDEN_DIM)
    parser.add_argument("--n-layers", type=int, default=N_LAYERS)
    parser.add_argument("--export-embeddings", action="store_true")
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    print(f"Device: {device}" +
          (f" ({torch.cuda.get_device_name(0)})" if device.type == "cuda" else ""))

    # Discover targets
    all_bundles = sorted(args.bundle_dir.glob("*_features.npz"))
    all_targets = [p.name.replace("_features.npz", "") for p in all_bundles]
    if args.max_targets > 0:
        all_targets = all_targets[:args.max_targets]
    print(f"Targets: {len(all_targets)}")

    # Tokenizer
    print(f"\n{'='*60}")
    print(f"  TOKENIZER: KMeans({VOCAB_SIZE}) on {SPIKE_DIM}-dim spike features")
    print(f"{'='*60}")
    centroids = build_tokenizer(
        args.bundle_dir, all_targets,
        cache_path=args.out_dir / "tokenizer_centroids.npz",
    )

    # Export-only mode
    if args.export_embeddings:
        if args.checkpoint is None:
            args.checkpoint = args.out_dir / "best_model.pt"
        print(f"\nLoading checkpoint: {args.checkpoint}")
        model = SpikeBERT(hidden_dim=args.hidden_dim, n_layers=args.n_layers).to(device)
        model.load_state_dict(torch.load(args.checkpoint, map_location=device,
                                         weights_only=True))
        print(f"\nExporting {HIDDEN_DIM}-dim spike embeddings for {len(all_targets)} targets...")
        export_all_embeddings(model, args.bundle_dir, all_targets, centroids,
                              args.out_dir / "embeddings", device)
        return

    # Dataset
    print(f"\n{'='*60}")
    print(f"  DATASET")
    print(f"{'='*60}")
    dataset = SpikeMLMDataset(args.bundle_dir, all_targets, centroids)

    # Train/val split (by protein, not residue)
    n_val = max(1, int(len(dataset) * args.val_frac))
    n_train = len(dataset) - n_val
    rng = torch.Generator().manual_seed(42)
    train_ds, val_ds = torch.utils.data.random_split(dataset, [n_train, n_val], generator=rng)
    print(f"  train: {n_train} proteins, val: {n_val} proteins")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              collate_fn=collate_mlm, num_workers=0, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            collate_fn=collate_mlm, num_workers=0)

    # Model
    print(f"\n{'='*60}")
    print(f"  SpikeBERT: {args.n_layers}L, {N_HEADS}H, {args.hidden_dim}D, {FFN_DIM} FFN")
    print(f"{'='*60}")
    model = SpikeBERT(hidden_dim=args.hidden_dim, n_layers=args.n_layers).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {n_params:,}")

    if args.checkpoint and args.checkpoint.exists():
        model.load_state_dict(torch.load(args.checkpoint, map_location=device,
                                         weights_only=True))
        print(f"  Resumed from {args.checkpoint}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                   weight_decay=0.01, betas=(0.9, 0.98))
    total_steps = args.epochs * len(train_loader)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=args.lr, total_steps=max(total_steps, 1),
        pct_start=min(args.warmup_steps / max(total_steps, 1), 0.1),
        anneal_strategy="cos",
    )

    # Training
    print(f"\n{'='*60}")
    print(f"  TRAINING: {args.epochs} epochs, batch={args.batch_size}, lr={args.lr}")
    print(f"{'='*60}")
    best_val_loss = float("inf")
    best_epoch = 0
    t0 = time.time()

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_epoch(model, train_loader, optimizer,
                                            scheduler, device, epoch)
        if epoch % 5 == 0 or epoch == args.epochs:
            val_loss, val_acc = evaluate(model, val_loader, device)
            improved = val_loss < best_val_loss
            if improved:
                best_val_loss = val_loss
                best_epoch = epoch
                torch.save(model.state_dict(), args.out_dir / "best_model.pt")
            marker = " *" if improved else ""
            print(f"    val loss={val_loss:.4f}  val_acc={val_acc:.1f}%{marker}")

    total_time = time.time() - t0
    print(f"\n  Training complete: {total_time:.0f}s ({total_time/60:.1f} min)")
    print(f"  Best val loss: {best_val_loss:.4f} at epoch {best_epoch}")

    # Save final
    torch.save(model.state_dict(), args.out_dir / "final_model.pt")

    # ONNX export
    print(f"\n{'='*60}")
    print(f"  ONNX EXPORT")
    print(f"{'='*60}")
    model.eval().cpu()
    dummy_ids = torch.full((1, 200), MASK_TOKEN_ID, dtype=torch.long)
    dummy_sf = torch.randn(1, 200, STRUCT_DIM)
    dummy_pm = torch.zeros(1, 200, dtype=torch.bool)

    onnx_path = args.out_dir / "spike_bert.onnx"
    torch.onnx.export(
        model, (dummy_ids, dummy_sf, dummy_pm),
        str(onnx_path),
        input_names=["input_ids", "struct_feats", "pad_mask"],
        output_names=["logits", "hidden_states"],
        dynamic_axes={
            "input_ids": {0: "batch", 1: "seq"},
            "struct_feats": {0: "batch", 1: "seq"},
            "pad_mask": {0: "batch", 1: "seq"},
            "logits": {0: "batch", 1: "seq"},
            "hidden_states": {0: "batch", 1: "seq"},
        },
        opset_version=17,
    )
    print(f"  ONNX: {onnx_path} ({onnx_path.stat().st_size / 1024 / 1024:.1f} MB)")

    # Export embeddings for all targets
    print(f"\n{'='*60}")
    print(f"  EXPORTING {HIDDEN_DIM}-dim SPIKE EMBEDDINGS")
    print(f"{'='*60}")
    model = model.to(device)
    best_state = torch.load(args.out_dir / "best_model.pt", map_location=device,
                            weights_only=True)
    model.load_state_dict(best_state)
    export_all_embeddings(model, args.bundle_dir, all_targets, centroids,
                          args.out_dir / "embeddings", device)

    # Summary
    config = {
        "vocab_size": VOCAB_SIZE, "hidden_dim": args.hidden_dim,
        "n_layers": args.n_layers, "n_heads": N_HEADS, "ffn_dim": FFN_DIM,
        "spike_dim": SPIKE_DIM, "struct_dim": STRUCT_DIM,
        "n_params": n_params, "n_targets": len(all_targets),
        "total_residues": sum(p["n_residues"] for p in dataset.proteins),
        "epochs": args.epochs, "best_epoch": best_epoch,
        "best_val_loss": best_val_loss, "training_time_sec": total_time,
    }
    (args.out_dir / "config.json").write_text(json.dumps(config, indent=2))

    print(f"\n{'='*60}")
    print(f"  SpikeBERT TRAINING COMPLETE")
    print(f"{'='*60}")
    print(f"  Model:      {n_params:,} parameters")
    print(f"  Vocab:      {VOCAB_SIZE} spike tokens (KMeans on {SPIKE_DIM}d)")
    print(f"  Data:       {len(all_targets)} proteins, "
          f"{config['total_residues']:,} residues")
    print(f"  Best:       epoch {best_epoch}, val_loss={best_val_loss:.4f}")
    print(f"  Time:       {total_time:.0f}s")
    print(f"  Outputs:    {args.out_dir}/")
    print(f"    best_model.pt, final_model.pt, spike_bert.onnx")
    print(f"    embeddings/  ({len(all_targets)} × {HIDDEN_DIM}-dim .npz)")
    print(f"    tokenizer_centroids.npz")
    print(f"\n  Zero-shot inference stack:")
    print(f"    PDB → structural({STRUCT_DIM}d) → SpikeBERT(all-masked)")
    print(f"        → {HIDDEN_DIM}-dim spike embedding/residue → VN-EGNN")


if __name__ == "__main__":
    main()
