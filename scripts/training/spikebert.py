#!/usr/bin/env python3
"""SpikeBERT v001 — hierarchical transformer for spike-inspired embeddings.

Two-level architecture tuned for small-data regime (~74K residues × 372 proteins).

Level 1 — intra-residue transformer:
    Each residue's 108-dim channel_features vector is decomposed into 6 tiers:
        source(30) + phase(30) + continuous(24) + cross(12) + cooperative(6) + intensity(6)
    Each tier → d_model projection → 6 tier-tokens
    Prepend learnable [RES_CLS]
    2-layer transformer attends across the 7 tokens
    Pool: [RES_CLS] hidden state → 128-dim per-residue signature

Level 2 — inter-residue transformer:
    Sequence of Level-1 residue signatures (one per residue in protein)
    Add PDB-context projection (struct+NMA+perturbed+ESM-2 = 1336 → d_model)
    Add sinusoidal positional encoding
    4-layer transformer → per-residue final embedding (d_model=128)

Self-supervised objectives (trained jointly):
    (a) Masked tier reconstruction — each tier masked independently with p=0.15.
        Six small heads predict the masked tier's original continuous values.
        Loss: MSE over masked tier positions.
    (b) Teacher distillation — teacher v005 binding probability is computed
        per residue (cached once per protein) and SpikeBERT predicts it from
        its final per-residue embedding via a linear sigmoid head.
        Loss: BCE between student sigmoid and teacher probability.

Combined loss: lambda_rec * tier_mse + lambda_teacher * teacher_bce

Total params ≈ 500K (2-3 orders below the training signal: 74K residues ×
6 tiers × 15% masked = ~67K tier-reconstruction gradients per epoch PLUS
74K teacher-prob gradients per epoch).

Zero-shot PDB inference (no engine required):
    tiers are replaced with learnable [MASK] embeddings (one per tier)
    Level 1 outputs a "what the tiers would be given PDB context" signature
    Level 2 outputs per-residue final embedding
    → 128-dim spike-inspired embedding per residue, ready to concat with ESM-2

Training outputs (`--out-dir`):
    spikebert_v001.pt              weights
    spikebert_v001.onnx            ONNX export (zero-shot / no tier inputs path)
    input_stats.json               PDB-context normalization
    tier_stats.json                per-tier feature z-score
    evaluation.json                val tier-MSE, teacher-AUROC, ρ
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    HAVE_TORCH = True
except ImportError:
    HAVE_TORCH = False

sys.path.insert(0, str(Path(__file__).parent))
from cluster_split import cluster_split_bundles

# ─────────────────────────────────────────────────────────────
#  Constants
# ─────────────────────────────────────────────────────────────

TIER_DIMS = {
    "source":      30,    # [0:30]
    "phase":       30,    # [30:60]
    "continuous":  24,    # [60:84]
    "cross":       12,    # [84:96]
    "cooperative": 6,     # [96:102]
    "intensity":   6,     # [102:108]
}
TIER_NAMES = list(TIER_DIMS.keys())
TIER_OFFSETS = {
    "source":      (0,   30),
    "phase":       (30,  60),
    "continuous":  (60,  84),
    "cross":       (84,  96),
    "cooperative": (96,  102),
    "intensity":   (102, 108),
}
N_TIERS = len(TIER_DIMS)            # 6
CHANNEL_DIM = sum(TIER_DIMS.values())   # 108

PDB_CONTEXT_BLOCKS = ("structural", "nma", "perturbed_nma", "esm2")
PDB_CONTEXT_DIM = 25 + 26 + 5 + 1280     # 1336


# ─────────────────────────────────────────────────────────────
#  Model
# ─────────────────────────────────────────────────────────────

class SpikeBERT(nn.Module):
    """Hierarchical residue transformer with teacher distillation head.

    Forward signature:
        tier_vecs:     dict[name] → [N, tier_dim]   (continuous, normalized)
        tier_mask:     dict[name] → [N] bool         (True = masked)
        pdb_ctx:       [N, 1336]  (normalized)
    Returns:
        residue_embeds:    [N, d_model]
        tier_reconstructions: dict[name] → [N, tier_dim]   (predicted continuous)
        teacher_logits:    [N]    (pre-sigmoid per-residue prob)
    """
    def __init__(self,
                 d_model: int = 128,
                 intra_layers: int = 2,
                 inter_layers: int = 4,
                 n_heads: int = 4,
                 ffn_mult: int = 4,
                 dropout: float = 0.1,
                 max_len: int = 2048):
        super().__init__()
        self.d_model = d_model

        # Per-tier projection (tier_dim → d_model), one per tier
        self.tier_proj = nn.ModuleDict({
            name: nn.Linear(TIER_DIMS[name], d_model) for name in TIER_NAMES
        })
        # Per-tier learnable MASK embedding (used when tier is absent/masked)
        self.tier_mask_tokens = nn.ParameterDict({
            name: nn.Parameter(torch.randn(d_model) * 0.02) for name in TIER_NAMES
        })
        # Per-tier type embedding so the model knows which tier a token represents
        self.tier_type_embed = nn.Parameter(torch.randn(N_TIERS, d_model) * 0.02)
        # [RES_CLS] token for intra-residue pooling
        self.res_cls = nn.Parameter(torch.randn(1, d_model) * 0.02)

        # PDB context projection
        self.pdb_proj = nn.Sequential(
            nn.Linear(PDB_CONTEXT_DIM, d_model),
            nn.LayerNorm(d_model),
        )

        # Level-1 transformer (attends across N_TIERS+1 tokens per residue)
        intra_layer = nn.TransformerEncoderLayer(
            d_model, n_heads, d_model * ffn_mult, dropout,
            activation="gelu", batch_first=True, norm_first=True,
        )
        self.intra = nn.TransformerEncoder(intra_layer, intra_layers)

        # Positional encoding for Level 2 (inter-residue)
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pos_enc", pe)   # [max_len, d_model]

        # Level-2 transformer (attends across residues)
        inter_layer = nn.TransformerEncoderLayer(
            d_model, n_heads, d_model * ffn_mult, dropout,
            activation="gelu", batch_first=True, norm_first=True,
        )
        self.inter = nn.TransformerEncoder(inter_layer, inter_layers)

        # Heads
        self.tier_heads = nn.ModuleDict({
            name: nn.Linear(d_model, TIER_DIMS[name]) for name in TIER_NAMES
        })
        self.teacher_head = nn.Linear(d_model, 1)

    def forward(self, tier_vecs: Dict[str, torch.Tensor],
                tier_mask: Dict[str, torch.Tensor],
                pdb_ctx: torch.Tensor
                ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], torch.Tensor]:
        N = pdb_ctx.size(0)
        device = pdb_ctx.device

        # ---- Level 1: intra-residue ----
        # Build per-residue token matrix: [N, N_TIERS+1, d_model]
        tokens = []
        for ti, name in enumerate(TIER_NAMES):
            proj = self.tier_proj[name](tier_vecs[name])            # [N, d_model]
            mask_tok = self.tier_mask_tokens[name][None, :].expand(N, -1)  # [N, d_model]
            chosen = torch.where(tier_mask[name][:, None], mask_tok, proj)
            chosen = chosen + self.tier_type_embed[ti]
            tokens.append(chosen)
        tier_tokens = torch.stack(tokens, dim=1)                    # [N, N_TIERS, d_model]

        # Prepend [RES_CLS]
        cls = self.res_cls.expand(N, 1, -1)                         # [N, 1, d_model]
        res_seq = torch.cat([cls, tier_tokens], dim=1)              # [N, N_TIERS+1, d_model]

        res_h = self.intra(res_seq)                                 # [N, N_TIERS+1, d_model]
        res_cls_h = res_h[:, 0]                                     # [N, d_model]
        tier_hidden = res_h[:, 1:]                                  # [N, N_TIERS, d_model]

        # ---- Level 2: inter-residue ----
        ctx = self.pdb_proj(pdb_ctx)                                # [N, d_model]
        seq = res_cls_h + ctx + self.pos_enc[:N]                    # [N, d_model]
        seq = seq.unsqueeze(0)                                      # [1, N, d_model]
        final = self.inter(seq).squeeze(0)                          # [N, d_model]

        # ---- Heads ----
        tier_recon = {}
        for ti, name in enumerate(TIER_NAMES):
            tier_recon[name] = self.tier_heads[name](tier_hidden[:, ti])
        teacher_logits = self.teacher_head(final).squeeze(-1)       # [N]

        return final, tier_recon, teacher_logits


# ─────────────────────────────────────────────────────────────
#  Data loading
# ─────────────────────────────────────────────────────────────

class ProteinSample:
    __slots__ = ("target", "tiers", "pdb_ctx", "teacher_probs", "labels")
    def __init__(self, target, tiers, pdb_ctx, teacher_probs, labels):
        self.target = target
        self.tiers = tiers                  # dict[name] → [N, tier_dim]
        self.pdb_ctx = pdb_ctx              # [N, PDB_CONTEXT_DIM]
        self.teacher_probs = teacher_probs  # [N] or None
        self.labels = labels                # [N] or None


def load_pdb_context(bundle) -> Optional[np.ndarray]:
    parts = []
    for k in PDB_CONTEXT_BLOCKS:
        if k not in bundle.files:
            return None
        parts.append(bundle[k].astype(np.float32))
    x = np.concatenate(parts, axis=1)
    if x.shape[1] != PDB_CONTEXT_DIM:
        return None
    return x


def load_tiers(bundle) -> Optional[Dict[str, np.ndarray]]:
    if "channel_features" not in bundle.files:
        return None
    ch = bundle["channel_features"].astype(np.float32)
    if ch.shape[1] != CHANNEL_DIM:
        return None
    out = {}
    for name, (lo, hi) in TIER_OFFSETS.items():
        out[name] = ch[:, lo:hi].copy()
    return out


def compute_teacher_probs(teacher_dir: Path, bundle, target: str,
                          device: torch.device) -> Optional[np.ndarray]:
    """Run teacher v005 ensemble on this bundle's features → per-residue sigmoid prob.

    Uses the same feature assembly as train_teacher.py's --feature-set full.
    Teacher is cached at import time via a global to avoid reloading per call.
    """
    global _TEACHER_MEMBERS, _TEACHER_STATS
    if _TEACHER_MEMBERS is None:
        _load_teacher(teacher_dir, device)
    # Assemble input: structural + nma + perturbed + physics_216 + tide + temporal + esm2 = 1561
    blocks = []
    for k in ("structural", "nma", "perturbed_nma", "physics_216",
              "tide_residue", "temporal", "esm2"):
        if k not in bundle.files:
            return None
        blocks.append(bundle[k].astype(np.float32))
    X = np.concatenate(blocks, axis=1)   # [N, 1561]
    mean = _TEACHER_STATS["mean"]; std = _TEACHER_STATS["std"]
    if X.shape[1] != len(mean):
        # dim mismatch — skip (e.g. missing block)
        return None
    Xn = (X - mean) / std
    Xt = torch.from_numpy(Xn).float().to(device)
    probs_sum = None
    with torch.no_grad():
        for m in _TEACHER_MEMBERS:
            p = torch.sigmoid(m(Xt))
            probs_sum = p if probs_sum is None else probs_sum + p
    probs = (probs_sum / len(_TEACHER_MEMBERS)).cpu().numpy().astype(np.float32)
    return probs


# Module-level teacher cache
_TEACHER_MEMBERS: Optional[List[nn.Module]] = None
_TEACHER_STATS: Optional[Dict[str, np.ndarray]] = None


def _load_teacher(teacher_dir: Path, device: torch.device) -> None:
    """Load all teacher fold checkpoints into memory as an ensemble."""
    global _TEACHER_MEMBERS, _TEACHER_STATS
    from train_teacher import TeacherMLP  # reuse the same MLP class
    stats = json.load(open(teacher_dir / "feature_stats.json"))
    _TEACHER_STATS = {
        "mean": np.asarray(stats["mean"], dtype=np.float32),
        "std":  np.asarray(stats["std"],  dtype=np.float32),
    }
    in_dim = len(_TEACHER_STATS["mean"])
    members = []
    for cp in sorted(teacher_dir.glob("teacher_fold_*.pt")):
        ck = torch.load(cp, map_location="cpu", weights_only=False)
        m = TeacherMLP(in_dim=in_dim)
        m.load_state_dict(ck["state_dict"])
        m.eval()
        members.append(m.to(device))
    if not members:
        raise RuntimeError(f"No teacher folds found in {teacher_dir}")
    _TEACHER_MEMBERS = members
    print(f"  Teacher loaded: {len(members)} fold members, in_dim={in_dim}")


def load_protein(bundle_path: Path, teacher_dir: Optional[Path],
                  device: torch.device) -> Optional[ProteinSample]:
    target = bundle_path.stem.replace("_features", "")
    try:
        bundle = np.load(bundle_path, allow_pickle=False)
    except Exception:
        return None
    tiers = load_tiers(bundle)
    pdb_ctx = load_pdb_context(bundle)
    if tiers is None or pdb_ctx is None:
        return None
    if pdb_ctx.shape[0] != tiers["source"].shape[0]:
        return None
    labels = bundle["labels"].astype(np.int32) if "labels" in bundle.files else None
    teacher_probs = None
    if teacher_dir is not None:
        teacher_probs = compute_teacher_probs(teacher_dir, bundle, target, device)
    return ProteinSample(target=target, tiers=tiers, pdb_ctx=pdb_ctx,
                          teacher_probs=teacher_probs, labels=labels)


# ─────────────────────────────────────────────────────────────
#  Normalization
# ─────────────────────────────────────────────────────────────

def fit_stats(arrays: List[np.ndarray]) -> Dict[str, np.ndarray]:
    X = np.concatenate(arrays, axis=0)
    mean = X.mean(axis=0).astype(np.float32)
    std = X.std(axis=0).astype(np.float32)
    std[std < 1e-6] = 1.0
    return {"mean": mean, "std": std}


def normalize_tiers(tiers: Dict[str, np.ndarray],
                    stats: Dict[str, Dict[str, np.ndarray]]) -> Dict[str, np.ndarray]:
    out = {}
    for name, arr in tiers.items():
        out[name] = (arr - stats[name]["mean"]) / stats[name]["std"]
    return out


# ─────────────────────────────────────────────────────────────
#  Training
# ─────────────────────────────────────────────────────────────

def masked_mse(pred: torch.Tensor, target: torch.Tensor,
               mask: torch.Tensor) -> torch.Tensor:
    if mask.sum() == 0:
        return torch.tensor(0.0, device=pred.device, dtype=pred.dtype)
    diff = (pred - target) ** 2                     # [N, D]
    loss = diff[mask].mean()
    return loss


def train_epoch(model: SpikeBERT, samples: List[ProteinSample],
                tier_stats: Dict[str, Dict[str, np.ndarray]],
                pdb_stats: Dict[str, np.ndarray],
                opt: torch.optim.Optimizer, device: torch.device,
                mask_prob: float, lambda_rec: float, lambda_teacher: float,
                rng: np.random.Generator) -> Dict[str, float]:
    model.train()
    rec_loss_sum = 0.0; teacher_loss_sum = 0.0; n_samples = 0
    rng.shuffle(samples)
    for s in samples:
        N = s.pdb_ctx.shape[0]
        # Prepare tensors
        pdb_norm = (s.pdb_ctx - pdb_stats["mean"]) / pdb_stats["std"]
        pdb_t = torch.from_numpy(pdb_norm).float().to(device)

        tier_t = {}; tier_target = {}
        for name in TIER_NAMES:
            x_norm = (s.tiers[name] - tier_stats[name]["mean"]) / tier_stats[name]["std"]
            tier_t[name] = torch.from_numpy(x_norm).float().to(device)
            tier_target[name] = tier_t[name].clone()

        # Masking — independent Bernoulli per residue per tier
        tier_mask = {}
        for name in TIER_NAMES:
            m = rng.random(N) < mask_prob
            tier_mask[name] = torch.from_numpy(m).to(device)

        # Teacher targets
        teacher_t = None
        if s.teacher_probs is not None:
            teacher_t = torch.from_numpy(s.teacher_probs).float().to(device)

        opt.zero_grad()
        _, tier_pred, teacher_logits = model(tier_t, tier_mask, pdb_t)

        rec_loss = torch.tensor(0.0, device=device)
        for name in TIER_NAMES:
            rec_loss = rec_loss + masked_mse(tier_pred[name], tier_target[name],
                                              tier_mask[name])
        rec_loss = rec_loss / N_TIERS

        teacher_loss = torch.tensor(0.0, device=device)
        if teacher_t is not None:
            teacher_loss = F.binary_cross_entropy_with_logits(teacher_logits, teacher_t)

        loss = lambda_rec * rec_loss + lambda_teacher * teacher_loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        opt.step()

        rec_loss_sum += rec_loss.item()
        teacher_loss_sum += teacher_loss.item()
        n_samples += 1
    return {"rec_loss": rec_loss_sum / max(n_samples, 1),
            "teacher_loss": teacher_loss_sum / max(n_samples, 1)}


def eval_split(model: SpikeBERT, samples: List[ProteinSample],
               tier_stats, pdb_stats, device: torch.device,
               mask_prob: float = 0.15) -> Dict:
    model.eval()
    rec_mse = {name: [] for name in TIER_NAMES}
    teacher_probs_pred = []; teacher_probs_true = []; labels_all = []
    rng = np.random.default_rng(0)
    with torch.no_grad():
        for s in samples:
            N = s.pdb_ctx.shape[0]
            pdb_norm = (s.pdb_ctx - pdb_stats["mean"]) / pdb_stats["std"]
            pdb_t = torch.from_numpy(pdb_norm).float().to(device)
            tier_t = {}
            for name in TIER_NAMES:
                x_norm = (s.tiers[name] - tier_stats[name]["mean"]) / tier_stats[name]["std"]
                tier_t[name] = torch.from_numpy(x_norm).float().to(device)
            # ZERO-SHOT MODE: all tiers masked (simulates inference with no engine output)
            tier_mask = {name: torch.ones(N, dtype=torch.bool, device=device)
                         for name in TIER_NAMES}
            _, tier_pred, teacher_logits = model(tier_t, tier_mask, pdb_t)
            for name in TIER_NAMES:
                target = tier_t[name]
                mse = ((tier_pred[name] - target) ** 2).mean().item()
                rec_mse[name].append(mse)
            if s.teacher_probs is not None:
                teacher_probs_pred.append(torch.sigmoid(teacher_logits).cpu().numpy())
                teacher_probs_true.append(s.teacher_probs)
            if s.labels is not None:
                labels_all.append(s.labels)

    # Aggregate
    agg_rec = {name: float(np.mean(v)) if v else float("nan") for name, v in rec_mse.items()}
    agg_rec_mean = float(np.mean(list(agg_rec.values())))

    teacher_mse = float("nan"); teacher_auroc = float("nan"); teacher_corr = float("nan")
    binding_auroc = float("nan")
    if teacher_probs_pred:
        p = np.concatenate(teacher_probs_pred)
        t = np.concatenate(teacher_probs_true)
        teacher_mse = float(((p - t) ** 2).mean())
        if t.std() > 1e-6:
            teacher_corr = float(np.corrcoef(p, t)[0, 1])
        if labels_all:
            lbl = np.concatenate(labels_all)
            if lbl.sum() > 0 and lbl.sum() < len(lbl):
                try:
                    from sklearn.metrics import roc_auc_score
                    binding_auroc = float(roc_auc_score(lbl, p))
                    teacher_auroc = float(roc_auc_score(lbl, t))
                except Exception:
                    pass

    return {
        "zero_shot_tier_mse": agg_rec,
        "zero_shot_tier_mse_mean": agg_rec_mean,
        "teacher_mse": teacher_mse,
        "teacher_corr_pred_vs_true": teacher_corr,
        "student_binding_auroc": binding_auroc,
        "teacher_binding_auroc": teacher_auroc,
    }


# ─────────────────────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--features-dir", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--teacher-dir", type=Path,
                        default=Path("/workspace/models/teacher_v005"))
    parser.add_argument("--cluster-cache-path", type=Path,
                        default=Path("/workspace/seq_clusters_30.json"))
    parser.add_argument("--min-seq-id", type=float, default=0.3)
    parser.add_argument("--val-frac", type=float, default=0.15)
    parser.add_argument("--test-frac", type=float, default=0.05)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--intra-layers", type=int, default=2)
    parser.add_argument("--inter-layers", type=int, default=4)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--mask-prob", type=float, default=0.15)
    parser.add_argument("--lambda-rec", type=float, default=1.0)
    parser.add_argument("--lambda-teacher", type=float, default=1.0)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--max-targets", type=int, default=0)
    args = parser.parse_args()

    if not HAVE_TORCH:
        print("ERROR: torch not installed"); sys.exit(1)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Discover bundles & cluster-aware split
    all_npzs = sorted(args.features_dir.glob("*_features.npz"))
    all_targets = [p.stem.replace("_features", "") for p in all_npzs]
    if args.max_targets:
        all_targets = all_targets[:args.max_targets]
    print(f"Total bundles: {len(all_targets)}")

    train_names, val_names, test_names = cluster_split_bundles(
        bundle_dir=args.features_dir, targets=all_targets,
        val_frac=args.val_frac, test_frac=args.test_frac,
        min_seq_id=args.min_seq_id, cache_path=args.cluster_cache_path,
    )

    # Load all samples + run teacher to cache teacher probs per residue
    print(f"\nLoading samples (train={len(train_names)} val={len(val_names)} "
          f"test={len(test_names)}) + computing teacher probs...")
    t0 = time.time()
    def _load_list(names):
        out = []
        for t in names:
            for stem in (f"{t}_features.npz", f"{t}.features.npz"):
                p = args.features_dir / stem
                if p.exists():
                    break
            else:
                continue
            s = load_protein(p, args.teacher_dir, device)
            if s is not None:
                out.append(s)
        return out
    train_s = _load_list(train_names)
    val_s = _load_list(val_names)
    test_s = _load_list(test_names)
    print(f"  usable — train={len(train_s)} val={len(val_s)} test={len(test_s)}  "
          f"load time={time.time()-t0:.1f}s")

    # Fit stats on train (per-tier + pdb context)
    tier_stats = {}
    for name in TIER_NAMES:
        tier_stats[name] = fit_stats([s.tiers[name] for s in train_s])
    pdb_stats = fit_stats([s.pdb_ctx for s in train_s])
    (args.out_dir / "tier_stats.json").write_text(json.dumps({
        name: {"mean": tier_stats[name]["mean"].tolist(),
               "std":  tier_stats[name]["std"].tolist()}
        for name in TIER_NAMES}))
    (args.out_dir / "input_stats.json").write_text(json.dumps({
        "pdb_context": {"mean": pdb_stats["mean"].tolist(),
                        "std":  pdb_stats["std"].tolist()},
        "block_offsets": {"structural": [0, 25], "nma": [25, 51],
                          "perturbed_nma": [51, 56], "esm2": [56, PDB_CONTEXT_DIM]},
    }))

    # Model
    model = SpikeBERT(d_model=args.d_model, intra_layers=args.intra_layers,
                      inter_layers=args.inter_layers, n_heads=args.n_heads).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"\nSpikeBERT: {n_params:,} params  "
          f"(d_model={args.d_model}  intra={args.intra_layers} inter={args.inter_layers})")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    rng = np.random.default_rng(42)
    history = []
    best_val = float("inf")
    best_state = None
    patience = 0

    print(f"\nTraining (epochs={args.epochs}, lr={args.lr}, "
          f"mask_prob={args.mask_prob}, λ_rec={args.lambda_rec}, "
          f"λ_teacher={args.lambda_teacher})...")
    t_start = time.time()
    for ep in range(args.epochs):
        tr = train_epoch(model, list(train_s), tier_stats, pdb_stats, opt, device,
                          args.mask_prob, args.lambda_rec, args.lambda_teacher, rng)
        sched.step()
        ev = eval_split(model, val_s, tier_stats, pdb_stats, device)
        combined_val = args.lambda_rec * ev["zero_shot_tier_mse_mean"] + \
                       args.lambda_teacher * ev["teacher_mse"]
        history.append({"epoch": ep + 1, **tr, **ev, "combined_val": combined_val})
        if (ep + 1) % 5 == 0 or ep == 0:
            print(f"  ep {ep+1:3d}  rec_tr={tr['rec_loss']:.4f}  "
                  f"tchr_tr={tr['teacher_loss']:.4f}  "
                  f"val_tier_mse={ev['zero_shot_tier_mse_mean']:.4f}  "
                  f"val_tchr_mse={ev['teacher_mse']:.4f}  "
                  f"val_tchr_corr={ev['teacher_corr_pred_vs_true']:.3f}  "
                  f"val_binding_AUROC={ev['student_binding_auroc']:.3f}", flush=True)

        if combined_val < best_val:
            best_val = combined_val
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience = 0
        else:
            patience += 1
            if patience >= args.patience:
                print(f"  Early stop at epoch {ep+1}")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    # Final eval on test + save
    test_ev = eval_split(model, test_s, tier_stats, pdb_stats, device)
    val_ev = eval_split(model, val_s, tier_stats, pdb_stats, device)

    print(f"\n{'='*64}\n  SpikeBERT v001 — final\n{'='*64}")
    print(f"  n_params: {n_params:,}")
    print(f"  best_val_combined: {best_val:.4f}")
    for label, ev in (("VAL", val_ev), ("TEST", test_ev)):
        print(f"  {label}:")
        print(f"    zero-shot tier MSE: {ev['zero_shot_tier_mse_mean']:.4f}  "
              f"(per-tier: {ev['zero_shot_tier_mse']})")
        print(f"    teacher MSE: {ev['teacher_mse']:.4f}  "
              f"corr(pred,true)={ev['teacher_corr_pred_vs_true']:.3f}")
        print(f"    binding AUROC — student={ev['student_binding_auroc']:.4f}  "
              f"teacher_oracle={ev['teacher_binding_auroc']:.4f}")

    torch.save(model.state_dict(), args.out_dir / "spikebert_v001.pt")
    (args.out_dir / "evaluation.json").write_text(json.dumps({
        "n_params": n_params,
        "best_val_combined": best_val,
        "training_time_sec": time.time() - t_start,
        "val": val_ev, "test": test_ev,
        "history": history,
        "config": vars(args),
    }, indent=2, default=str))

    print(f"\n  Outputs in: {args.out_dir}")


if __name__ == "__main__":
    main()
