#!/usr/bin/env python3
"""SpikeBERT v002 — Multi-task: MLM + teacher distillation + physics regression.

Three loss heads on the same transformer encoder:
  1. MLM: predict masked spike tokens from structural context (grammar)
  2. Distillation: predict teacher v005 soft logits from hidden states (function)
  3. Physics regression: predict physics_216 from hidden states (dynamics)

All 372 targets. Teacher soft logits already held-out via LOTO.
"""
import argparse, json, math, os, sys, time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

VOCAB_SIZE = 2048
SPIKE_DIM = 333
STRUCT_DIM = 56
HIDDEN_DIM = 512
N_LAYERS = 6
N_HEADS = 8
FFN_DIM = 2048
DROPOUT = 0.15
MASK_PROB = 0.15
MASK_TOKEN_ID = VOCAB_SIZE
PAD_TOKEN_ID = VOCAB_SIZE + 1
TOTAL_VOCAB = VOCAB_SIZE + 2
PHYSICS_DIM = 216

SPIKE_KEYS = ("channel_features", "physics_216", "tide_residue", "temporal")
STRUCT_KEYS = ("structural", "nma", "perturbed_nma")

# Loss weights
W_MLM = 1.0
W_DISTILL = 2.0
W_PHYSICS = 0.5


def build_tokenizer(bundle_dir, targets, n_clusters=VOCAB_SIZE, cache_path=None):
    if cache_path and cache_path.exists():
        return np.load(cache_path)["centroids"]
    from sklearn.cluster import MiniBatchKMeans
    print(f"[tokenizer] collecting from {len(targets)} targets...")
    feats = []
    for t in targets:
        p = bundle_dir / f"{t}_features.npz"
        if not p.exists(): continue
        d = np.load(p, allow_pickle=False)
        blocks = [d[k].astype(np.float32) for k in SPIKE_KEYS if k in d]
        if blocks: feats.append(np.concatenate(blocks, axis=1))
    X = np.concatenate(feats, axis=0)
    print(f"[tokenizer] KMeans({n_clusters}) on {X.shape}...")
    km = MiniBatchKMeans(n_clusters=n_clusters, batch_size=4096, random_state=42, n_init=3)
    km.fit(X)
    if cache_path:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(cache_path, centroids=km.cluster_centers_.astype(np.float32))
    print(f"[tokenizer] done. inertia={km.inertia_:.1f}")
    return km.cluster_centers_.astype(np.float32)


def tokenize_residues(bundle, centroids):
    blocks = [bundle[k].astype(np.float32) for k in SPIKE_KEYS if k in bundle]
    X = np.concatenate(blocks, axis=1)
    dists = np.linalg.norm(X[:, None, :] - centroids[None, :, :], axis=2)
    return dists.argmin(axis=1).astype(np.int32)


class SpikeBERTv2Dataset(Dataset):
    def __init__(self, bundle_dir, teacher_dir, targets, centroids):
        self.proteins = []
        for t in targets:
            bp = bundle_dir / f"{t}_features.npz"
            tp = teacher_dir / f"{t}_teacher_probs.npz"
            if not bp.exists() or not tp.exists(): continue
            d = np.load(bp, allow_pickle=True)
            tokens = tokenize_residues(d, centroids)
            struct = np.concatenate([d[k].astype(np.float32) for k in STRUCT_KEYS if k in d], axis=1)
            physics = d["physics_216"].astype(np.float32) if "physics_216" in d else np.zeros((len(tokens), PHYSICS_DIM), dtype=np.float32)
            teacher = np.load(tp)["teacher_probs"].astype(np.float32)
            n = len(tokens)
            teacher = teacher[:n] if len(teacher) >= n else np.pad(teacher, (0, n - len(teacher)), constant_values=0.5)
            self.proteins.append({
                "target": t, "tokens": tokens, "struct": struct,
                "physics": physics, "teacher_probs": teacher, "n_residues": n,
            })
        print(f"[dataset] {len(self.proteins)} proteins, "
              f"{sum(p['n_residues'] for p in self.proteins):,} residues")

    def __len__(self): return len(self.proteins)
    def __getitem__(self, idx): return self.proteins[idx]


def collate_v2(batch):
    max_len = max(p["n_residues"] for p in batch)
    B = len(batch)
    input_ids = torch.full((B, max_len), PAD_TOKEN_ID, dtype=torch.long)
    labels_mlm = torch.full((B, max_len), -100, dtype=torch.long)
    struct_feats = torch.zeros(B, max_len, STRUCT_DIM)
    physics_targets = torch.zeros(B, max_len, PHYSICS_DIM)
    teacher_targets = torch.full((B, max_len), 0.5)
    pad_mask = torch.ones(B, max_len, dtype=torch.bool)
    lengths = []

    for i, p in enumerate(batch):
        n = p["n_residues"]
        tokens = torch.tensor(p["tokens"], dtype=torch.long)
        input_ids[i, :n] = tokens
        struct_feats[i, :n] = torch.tensor(p["struct"])
        physics_targets[i, :n] = torch.tensor(p["physics"])
        teacher_targets[i, :n] = torch.tensor(p["teacher_probs"])
        pad_mask[i, :n] = False
        lengths.append(n)
        # MLM masking
        mask = torch.rand(n) < MASK_PROB
        if mask.sum() == 0: mask[torch.randint(n, (1,))] = True
        labels_mlm[i, :n][mask] = tokens[mask]
        probs = torch.rand(mask.sum())
        midx = mask.nonzero(as_tuple=True)[0]
        input_ids[i, midx[probs < 0.8]] = MASK_TOKEN_ID
        input_ids[i, midx[(probs >= 0.8) & (probs < 0.9)]] = torch.randint(VOCAB_SIZE, ((((probs >= 0.8) & (probs < 0.9)).sum(),)))

    return {"input_ids": input_ids, "struct_feats": struct_feats, "pad_mask": pad_mask,
            "labels_mlm": labels_mlm, "physics_targets": physics_targets,
            "teacher_targets": teacher_targets, "lengths": lengths}


class SpikeBERTv2(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embed = nn.Embedding(TOTAL_VOCAB, HIDDEN_DIM, padding_idx=PAD_TOKEN_ID)
        self.struct_proj = nn.Linear(STRUCT_DIM, HIDDEN_DIM)
        self.pos_embed = nn.Embedding(2048, HIDDEN_DIM)
        self.input_norm = nn.LayerNorm(HIDDEN_DIM)
        self.input_dropout = nn.Dropout(DROPOUT)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=HIDDEN_DIM, nhead=N_HEADS, dim_feedforward=FFN_DIM,
            dropout=DROPOUT, activation="gelu", batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=N_LAYERS)
        # Head 1: MLM
        self.mlm_norm = nn.LayerNorm(HIDDEN_DIM)
        self.mlm_head = nn.Linear(HIDDEN_DIM, VOCAB_SIZE)
        # Head 2: Teacher distillation (predict per-residue binding probability)
        self.distill_head = nn.Sequential(
            nn.LayerNorm(HIDDEN_DIM), nn.Linear(HIDDEN_DIM, 128),
            nn.GELU(), nn.Linear(128, 1))
        # Head 3: Physics regression (predict physics_216)
        self.physics_head = nn.Sequential(
            nn.LayerNorm(HIDDEN_DIM), nn.Linear(HIDDEN_DIM, 256),
            nn.GELU(), nn.Linear(256, PHYSICS_DIM))
        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1: nn.init.xavier_normal_(p)

    def forward(self, input_ids, struct_feats, pad_mask):
        B, L = input_ids.shape
        pos = torch.arange(L, device=input_ids.device).unsqueeze(0)
        x = self.token_embed(input_ids) + self.struct_proj(struct_feats) + self.pos_embed(pos)
        x = self.input_dropout(self.input_norm(x))
        h = self.encoder(x, src_key_padding_mask=pad_mask)
        mlm_logits = self.mlm_head(self.mlm_norm(h))
        distill_logits = self.distill_head(h).squeeze(-1)
        physics_pred = self.physics_head(h)
        return {"hidden_states": h, "mlm_logits": mlm_logits,
                "distill_logits": distill_logits, "physics_pred": physics_pred}

    def extract_embeddings(self, struct_feats, pad_mask):
        B, L, _ = struct_feats.shape
        ids = torch.full((B, L), MASK_TOKEN_ID, dtype=torch.long, device=struct_feats.device)
        ids[pad_mask] = PAD_TOKEN_ID
        return self.forward(ids, struct_feats, pad_mask)["hidden_states"]


def train_epoch(model, loader, optimizer, scheduler, device, epoch):
    model.train()
    total_loss = 0; total_mlm_correct = 0; total_mlm_n = 0
    for batch in loader:
        ids = batch["input_ids"].to(device)
        sf = batch["struct_feats"].to(device)
        pm = batch["pad_mask"].to(device)
        mlm_labels = batch["labels_mlm"].to(device)
        teacher = batch["teacher_targets"].to(device)
        physics = batch["physics_targets"].to(device)

        out = model(ids, sf, pm)

        # Loss 1: MLM
        loss_mlm = F.cross_entropy(out["mlm_logits"].view(-1, VOCAB_SIZE),
                                    mlm_labels.view(-1), ignore_index=-100)
        # Loss 2: Teacher distillation (BCE on non-pad residues)
        non_pad = ~pm
        if non_pad.any():
            loss_distill = F.binary_cross_entropy_with_logits(
                out["distill_logits"][non_pad], teacher[non_pad])
        else:
            loss_distill = torch.tensor(0.0, device=device)

        # Loss 3: Physics regression (MSE on non-pad residues)
        if non_pad.any():
            loss_physics = F.mse_loss(out["physics_pred"][non_pad], physics[non_pad])
        else:
            loss_physics = torch.tensor(0.0, device=device)

        loss = W_MLM * loss_mlm + W_DISTILL * loss_distill + W_PHYSICS * loss_physics

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if scheduler: scheduler.step()
        total_loss += loss.item() * ids.size(0)

        mask = mlm_labels != -100
        if mask.any():
            total_mlm_correct += (out["mlm_logits"].argmax(-1)[mask] == mlm_labels[mask]).sum().item()
            total_mlm_n += mask.sum().item()

    n = len(loader.dataset)
    acc = total_mlm_correct / max(total_mlm_n, 1) * 100
    print(f"  epoch {epoch:3d}  loss={total_loss/n:.4f}  mlm_acc={acc:.1f}%  "
          f"({time.time()-t0:.0f}s)", flush=True)
    return total_loss / n, acc


def main():
    global t0
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle-dir", type=Path, default=Path("/workspace/spike_bert/features"))
    parser.add_argument("--teacher-dir", type=Path, default=Path("/workspace/spike_bert/output/teacher_soft_logits"))
    parser.add_argument("--out-dir", type=Path, default=Path("/workspace/spike_bert_v002/output"))
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--warmup-epochs", type=int, default=10)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--max-targets", type=int, default=0)
    parser.add_argument("--init-from", type=Path, default=None,
                        help="warm-start from SpikeBERT v001 checkpoint")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    print(f"Device: {device}" +
          (f" ({torch.cuda.get_device_name(0)})" if device.type == "cuda" else ""))

    targets = sorted([p.name.replace("_features.npz", "")
                       for p in args.bundle_dir.glob("*_features.npz")])
    if args.max_targets > 0: targets = targets[:args.max_targets]
    print(f"Targets: {len(targets)}")

    # Tokenizer (reuse v001 centroids if available)
    v001_cache = Path("/workspace/spike_bert/output/tokenizer_centroids.npz")
    centroids = build_tokenizer(args.bundle_dir, targets,
                                cache_path=v001_cache if v001_cache.exists() else args.out_dir / "tokenizer_centroids.npz")

    # Dataset
    dataset = SpikeBERTv2Dataset(args.bundle_dir, args.teacher_dir, targets, centroids)
    n_val = max(1, int(len(dataset) * 0.1))
    train_ds, val_ds = torch.utils.data.random_split(
        dataset, [len(dataset) - n_val, n_val], generator=torch.Generator().manual_seed(42))
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              collate_fn=collate_v2, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            collate_fn=collate_v2, num_workers=0)

    # Model
    print(f"\nSpikeBERT v002: {N_LAYERS}L, {N_HEADS}H, {HIDDEN_DIM}D")
    print(f"  Heads: MLM(w={W_MLM}) + Distill(w={W_DISTILL}) + Physics(w={W_PHYSICS})")
    model = SpikeBERTv2().to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {n_params:,}")

    # Warm-start from v001
    if args.init_from and args.init_from.exists():
        v001_state = torch.load(args.init_from, map_location=device, weights_only=True)
        missing, unexpected = model.load_state_dict(v001_state, strict=False)
        print(f"  Warm-started from {args.init_from}")
        print(f"    missing: {len(missing)}, unexpected: {len(unexpected)}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01, betas=(0.9, 0.98))
    total_steps = args.epochs * len(train_loader)
    warmup_steps = args.warmup_epochs * len(train_loader)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=args.lr, total_steps=max(total_steps, 1),
        pct_start=min(warmup_steps / max(total_steps, 1), 0.1), anneal_strategy="cos")

    # Training
    print(f"\nTraining: {args.epochs} epochs, lr={args.lr}, warmup={args.warmup_epochs}")
    best_val_loss = float("inf"); best_epoch = 0
    t0 = time.time()

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, scheduler, device, epoch)
        if epoch % 5 == 0 or epoch == args.epochs:
            model.eval()
            val_loss = 0; val_correct = 0; val_n = 0
            with torch.no_grad():
                for batch in val_loader:
                    ids = batch["input_ids"].to(device)
                    sf = batch["struct_feats"].to(device)
                    pm = batch["pad_mask"].to(device)
                    ml = batch["labels_mlm"].to(device)
                    te = batch["teacher_targets"].to(device)
                    ph = batch["physics_targets"].to(device)
                    out = model(ids, sf, pm)
                    l1 = F.cross_entropy(out["mlm_logits"].view(-1, VOCAB_SIZE), ml.view(-1), ignore_index=-100)
                    non_pad = ~pm
                    l2 = F.binary_cross_entropy_with_logits(out["distill_logits"][non_pad], te[non_pad]) if non_pad.any() else 0.0
                    l3 = F.mse_loss(out["physics_pred"][non_pad], ph[non_pad]) if non_pad.any() else 0.0
                    val_loss += (W_MLM * l1 + W_DISTILL * l2 + W_PHYSICS * l3).item() * ids.size(0)
                    mask = ml != -100
                    if mask.any():
                        val_correct += (out["mlm_logits"].argmax(-1)[mask] == ml[mask]).sum().item()
                        val_n += mask.sum().item()
            val_loss /= max(len(val_ds), 1)
            val_acc = val_correct / max(val_n, 1) * 100
            improved = val_loss < best_val_loss
            if improved:
                best_val_loss = val_loss; best_epoch = epoch
                torch.save(model.state_dict(), args.out_dir / "best_model.pt")
            print(f"    val loss={val_loss:.4f}  val_acc={val_acc:.1f}%{'  *' if improved else ''}")

    total_time = time.time() - t0
    print(f"\nTraining complete: {total_time:.0f}s ({total_time/60:.1f} min)")
    print(f"Best: epoch {best_epoch}, val_loss={best_val_loss:.4f}")
    torch.save(model.state_dict(), args.out_dir / "final_model.pt")

    # Export all-masked embeddings
    print(f"\nExporting 512-dim all-masked embeddings for {len(targets)} targets...")
    model.load_state_dict(torch.load(args.out_dir / "best_model.pt", map_location=device, weights_only=True))
    model.eval()
    emb_dir = args.out_dir / "embeddings_allmasked"
    emb_dir.mkdir(exist_ok=True)
    for i, t in enumerate(targets):
        p = args.bundle_dir / f"{t}_features.npz"
        if not p.exists(): continue
        d = np.load(p, allow_pickle=True)
        struct = np.concatenate([d[k].astype(np.float32) for k in STRUCT_KEYS if k in d], axis=1)
        n = struct.shape[0]
        sf = torch.tensor(struct).unsqueeze(0).to(device)
        pm = torch.zeros(1, n, dtype=torch.bool, device=device)
        with torch.no_grad():
            emb = model.extract_embeddings(sf, pm)
        np.savez_compressed(emb_dir / f"{t}_spike_emb.npz",
                            spike_embedding=emb.squeeze(0).cpu().numpy())
        if (i+1) % 50 == 0 or (i+1) == len(targets):
            print(f"  [{i+1}/{len(targets)}]", flush=True)

    # ONNX export
    print("\nONNX export...")
    model.cpu().eval()
    dummy_ids = torch.full((1, 200), MASK_TOKEN_ID, dtype=torch.long)
    dummy_sf = torch.randn(1, 200, STRUCT_DIM)
    dummy_pm = torch.zeros(1, 200, dtype=torch.bool)
    onnx_path = args.out_dir / "spike_bert_v002.onnx"
    torch.onnx.export(model, (dummy_ids, dummy_sf, dummy_pm), str(onnx_path),
        input_names=["input_ids", "struct_feats", "pad_mask"],
        output_names=["mlm_logits", "distill_logits", "physics_pred", "hidden_states"],
        dynamic_axes={"input_ids": {0: "batch", 1: "seq"}, "struct_feats": {0: "batch", 1: "seq"},
                       "pad_mask": {0: "batch", 1: "seq"}},
        opset_version=17)
    print(f"  ONNX: {onnx_path} ({onnx_path.stat().st_size/1024/1024:.1f} MB)")

    config = {"version": "v002", "heads": ["mlm", "distillation", "physics_regression"],
              "loss_weights": {"mlm": W_MLM, "distill": W_DISTILL, "physics": W_PHYSICS},
              "teacher": "v005 (0.830 AUROC)", "n_params": n_params,
              "best_epoch": best_epoch, "best_val_loss": best_val_loss,
              "training_time_sec": total_time, "n_targets": len(targets)}
    (args.out_dir / "config.json").write_text(json.dumps(config, indent=2))

    print(f"\nSpikeBERT v002 COMPLETE")
    print(f"  {n_params:,} params, {len(targets)} targets, {total_time:.0f}s")
    print(f"  Embeddings: {emb_dir} ({len(list(emb_dir.glob('*.npz')))} files)")
    print(f"  Next: train VN-EGNN v004 on these embeddings")


if __name__ == "__main__":
    main()
