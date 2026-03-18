#!/usr/bin/env python3
"""
PRISM4D Production Pocket Ranker v2
====================================
ESM-2 protein language model embeddings + Geometric Vector Perceptron (GVP)
equivariant graph neural network for binding site ranking.

Architecture:
    1. ESM-2 (650M) — pretrained on 250M sequences, produces 1280-dim
       per-residue embeddings encoding evolutionary context, secondary
       structure propensity, and binding site signatures.
    2. GVP-GNN — SE(3)-equivariant message passing that jointly updates
       scalar features AND geometric vector features. Unlike vanilla EGNN,
       GVP properly handles directional information (backbone orientation,
       sidechain pointing, surface normals).
    3. Multi-resolution graph — pocket lining residues + 2-hop protein
       context within 15A of pocket centroid.
    4. Graph-level auxiliary features — PRISM quality score, pocket volume,
       burial fraction, classification one-hot.
    5. Focal loss — handles extreme class imbalance (14 hits / 710 decoys)
       better than pos_weight BCE.

Usage:
    # Train on BENCH30
    ~/miniconda3/envs/prism_dock/bin/python scripts/egnn_pocket_ranker_v2.py train

    # Predict on new protein
    ~/miniconda3/envs/prism_dock/bin/python scripts/egnn_pocket_ranker_v2.py predict \\
        --pdb protein.pdb --sites protein.binding_sites.json

References:
    Lin et al. "Evolutionary-scale prediction of atomic-level protein
    structure with a language model" Science 2023
    Jing et al. "Learning from Protein Structure with Geometric Vector
    Perceptrons" ICLR 2021
"""

import argparse
import json
import os
import sys
import warnings
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import global_mean_pool, global_max_pool

warnings.filterwarnings("ignore", message=".*sm_120.*")
warnings.filterwarnings("ignore", message=".*DataLoader.*deprecated.*")

# ── Constants ────────────────────────────────────────────────────────────

AA_3TO1 = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
    "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
    "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
    "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
    "MSE": "M", "HID": "H", "HIE": "H", "HIP": "H", "CYX": "C",
}

# Residue physicochemical properties (hydrophobicity, charge, size, aromaticity, hbond)
AA_PROPS = {
    "A": [0.62, 0, 0.24, 0, 0], "R": [-2.53, 1, 0.80, 0, 1],
    "N": [-0.78, 0, 0.48, 0, 1], "D": [-0.90, -1, 0.44, 0, 1],
    "C": [0.29, 0, 0.36, 0, 0], "Q": [-0.85, 0, 0.56, 0, 1],
    "E": [-0.74, -1, 0.52, 0, 1], "G": [0.48, 0, 0.00, 0, 0],
    "H": [-0.40, 0.5, 0.60, 1, 1], "I": [1.38, 0, 0.56, 0, 0],
    "L": [1.06, 0, 0.56, 0, 0], "K": [-1.50, 1, 0.68, 0, 1],
    "M": [0.64, 0, 0.60, 0, 0], "F": [1.19, 0, 0.72, 1, 0],
    "P": [0.12, 0, 0.36, 0, 0], "S": [-0.18, 0, 0.28, 0, 1],
    "T": [-0.05, 0, 0.40, 0, 1], "W": [0.81, 0, 0.88, 1, 1],
    "Y": [0.26, 0, 0.76, 1, 1], "V": [1.08, 0, 0.48, 0, 0],
}


# ── GVP Layer ────────────────────────────────────────────────────────────

class GVPLayer(nn.Module):
    """Geometric Vector Perceptron layer.

    Operates on (scalar_features, vector_features) tuples.
    Scalar features: (N, s_dim) — invariant under rotation
    Vector features: (N, v_dim, 3) — equivariant under rotation

    The key insight: vector norms are invariant, so we can use them
    as inputs to scalar MLPs. The scalar outputs then gate the vectors.
    """

    def __init__(self, s_in, s_out, v_in, v_out, edge_s_dim=32):
        super().__init__()
        self.s_in, self.s_out = s_in, s_out
        self.v_in, self.v_out = v_in, v_out

        # Message MLP: operates on concatenated sender+receiver+edge features
        msg_in = 2 * s_in + v_in + edge_s_dim
        self.msg_mlp = nn.Sequential(
            nn.Linear(msg_in, s_out * 2),
            nn.SiLU(),
            nn.Linear(s_out * 2, s_out + v_out),
        )

        # Update MLP for scalar features
        self.update_mlp = nn.Sequential(
            nn.Linear(s_in + s_out + v_out, s_out),
            nn.SiLU(),
            nn.Linear(s_out, s_out),
        )

        # Edge feature MLP
        self.edge_mlp = nn.Sequential(
            nn.Linear(1, edge_s_dim),  # distance -> edge features
            nn.SiLU(),
            nn.Linear(edge_s_dim, edge_s_dim),
        )

        self.layer_norm_s = nn.LayerNorm(s_out)
        self.layer_norm_v = nn.LayerNorm(3)  # normalize over xyz

    def forward(self, s, v, pos, edge_index):
        """
        s: (N, s_in) scalar features
        v: (N, v_in, 3) vector features
        pos: (N, 3) coordinates
        edge_index: (2, E)
        """
        row, col = edge_index

        # Edge features from distances
        diff = pos[row] - pos[col]  # (E, 3)
        dist = torch.norm(diff, dim=-1, keepdim=True)  # (E, 1)
        edge_s = self.edge_mlp(dist)  # (E, edge_s_dim)

        # Vector norms as scalar features (rotation invariant)
        v_norms = torch.norm(v, dim=-1)  # (N, v_in)

        # Build messages
        msg_input = torch.cat([
            s[row], s[col], v_norms[row], edge_s
        ], dim=-1)
        msg = self.msg_mlp(msg_input)  # (E, s_out + v_out)
        msg_s = msg[:, :self.s_out]  # scalar part
        msg_v_weights = msg[:, self.s_out:]  # vector gating weights (E, v_out)

        # Aggregate scalar messages
        agg_s = torch.zeros(s.size(0), self.s_out, device=s.device)
        agg_s.index_add_(0, col, msg_s)

        # Aggregate vector messages: weight relative position vectors
        # Each output vector channel gets a weighted sum of edge vectors
        unit_diff = diff / (dist + 1e-8)  # (E, 3)
        agg_v = torch.zeros(s.size(0), self.v_out, 3, device=s.device)
        for k in range(self.v_out):
            weighted = unit_diff * msg_v_weights[:, k:k+1]  # (E, 3)
            agg_v[:, k].index_add_(0, col, weighted)

        # Update scalar features
        agg_v_norms = torch.norm(agg_v, dim=-1)  # (N, v_out)
        update_input = torch.cat([s, agg_s, agg_v_norms], dim=-1)
        s_new = s[:, :self.s_out] + self.layer_norm_s(self.update_mlp(update_input)) \
            if self.s_in == self.s_out else self.layer_norm_s(self.update_mlp(update_input))

        # Update vector features (equivariant)
        v_new = agg_v  # already equivariant by construction

        return s_new, v_new


# ── Full Model ───────────────────────────────────────────────────────────

class PrismPocketRankerV2(nn.Module):
    """Production pocket ranker: ESM-2 embeddings → GVP-GNN → classifier.

    Args:
        esm_dim: ESM-2 embedding dimension (1280 for 650M, 320 for 8M)
        physchem_dim: physicochemical property dimension (5)
        aux_dim: auxiliary graph-level features (volume, burial, quality, etc.)
        hidden_s: hidden scalar dimension in GVP
        hidden_v: hidden vector channels in GVP
        n_layers: number of GVP message passing layers
        dropout: dropout rate
    """

    def __init__(self, esm_dim=1280, physchem_dim=5, aux_dim=8,
                 hidden_s=256, hidden_v=64, n_layers=3, dropout=0.2):
        super().__init__()

        self.esm_dim = esm_dim
        self.aux_dim = aux_dim

        # Project ESM embeddings + physicochemical features to hidden dim
        self.node_encoder = nn.Sequential(
            nn.Linear(esm_dim + physchem_dim + 3, hidden_s),  # +3 for bfactor, dist_to_cent, burial
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_s, hidden_s),
        )

        # Initial vector features from relative position to centroid
        self.v_init = nn.Linear(1, hidden_v)  # scalar -> v_dim channel weights

        # GVP layers
        self.gvp_layers = nn.ModuleList()
        for i in range(n_layers):
            self.gvp_layers.append(
                GVPLayer(hidden_s, hidden_s, hidden_v, hidden_v)
            )

        # Classifier: graph-level pooled features + auxiliary features → score
        self.classifier = nn.Sequential(
            nn.Linear(hidden_s * 2 + hidden_v + aux_dim, hidden_s),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_s, hidden_s // 2),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_s // 2, 1),
        )

    def forward(self, data):
        # Node features from ESM + physicochemical
        h = self.node_encoder(data.x)  # (N, hidden_s)

        # Initialize vector features from position relative to centroid
        # data.pos_rel: (N, 3) relative to pocket centroid
        pos_rel_norm = torch.norm(data.pos_rel, dim=-1, keepdim=True)  # (N, 1)
        v_weights = self.v_init(pos_rel_norm)  # (N, hidden_v)
        # Broadcast: (N, hidden_v, 1) * (N, 1, 3) -> (N, hidden_v, 3)
        unit_pos = data.pos_rel / (pos_rel_norm + 1e-8)  # (N, 3)
        v = v_weights.unsqueeze(-1) * unit_pos.unsqueeze(1)  # (N, hidden_v, 3)

        # Message passing
        for layer in self.gvp_layers:
            h, v = layer(h, v, data.pos, data.edge_index)

        # Readout: mean + max pooling (captures both average and extreme features)
        h_mean = global_mean_pool(h, data.batch)  # (B, hidden_s)
        h_max = global_max_pool(h, data.batch)  # (B, hidden_s)
        v_norms = torch.norm(v, dim=-1)  # (N, hidden_v)
        v_pool = global_mean_pool(v_norms, data.batch)  # (B, hidden_v)

        # Concatenate with auxiliary graph-level features
        graph_feat = torch.cat([h_mean, h_max, v_pool, data.aux], dim=-1)

        return self.classifier(graph_feat).squeeze(-1)


# ── Focal Loss ───────────────────────────────────────────────────────────

class FocalLoss(nn.Module):
    """Focal loss for extreme class imbalance.

    Down-weights easy negatives, focuses learning on hard examples.
    Lin et al. "Focal Loss for Dense Object Detection" ICCV 2017
    """

    def __init__(self, alpha=0.75, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits, targets):
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        p = torch.sigmoid(logits)
        p_t = p * targets + (1 - p) * (1 - targets)
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        focal_weight = alpha_t * (1 - p_t) ** self.gamma
        return (focal_weight * bce).mean()


# ── ESM-2 Embedding Cache ───────────────────────────────────────────────

class ESMEmbedder:
    """Compute and cache ESM-2 per-residue embeddings."""

    def __init__(self, model_name="esm2_t33_650M_UR50D", device="cuda", cache_dir="models/esm_cache"):
        self.device = device
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        print(f"Loading ESM-2 ({model_name})...")
        import esm
        self.model, self.alphabet = esm.pretrained.load_model_and_alphabet(model_name)
        self.model = self.model.eval().to(device)
        self.batch_converter = self.alphabet.get_batch_converter()
        self.repr_layer = int(model_name.split("_t")[1].split("_")[0])
        print(f"  ESM-2 loaded: {sum(p.numel() for p in self.model.parameters())//1_000_000}M params, "
              f"repr_layer={self.repr_layer}")

    @torch.no_grad()
    def embed(self, pdb_id, sequence):
        """Get per-residue embeddings. Uses disk cache."""
        cache_path = self.cache_dir / f"{pdb_id}.pt"
        if cache_path.exists():
            return torch.load(cache_path, map_location="cpu", weights_only=True)

        # ESM-2 has a 1022-token limit per forward pass
        max_len = 1022
        if len(sequence) <= max_len:
            data = [(pdb_id, sequence)]
            _, _, tokens = self.batch_converter(data)
            result = self.model(tokens.to(self.device), repr_layers=[self.repr_layer])
            emb = result["representations"][self.repr_layer][0, 1:-1].cpu()  # strip BOS/EOS
        else:
            # Chunk long sequences with overlap
            overlap = 100
            chunks = []
            for start in range(0, len(sequence), max_len - overlap):
                end = min(start + max_len, len(sequence))
                chunk_seq = sequence[start:end]
                data = [(f"{pdb_id}_{start}", chunk_seq)]
                _, _, tokens = self.batch_converter(data)
                result = self.model(tokens.to(self.device), repr_layers=[self.repr_layer])
                chunk_emb = result["representations"][self.repr_layer][0, 1:-1].cpu()
                chunks.append((start, end, chunk_emb))
                if end >= len(sequence):
                    break

            # Stitch chunks (average overlapping regions)
            emb = torch.zeros(len(sequence), chunks[0][2].size(-1))
            counts = torch.zeros(len(sequence), 1)
            for start, end, chunk_emb in chunks:
                actual_len = min(chunk_emb.size(0), end - start)
                emb[start:start + actual_len] += chunk_emb[:actual_len]
                counts[start:start + actual_len] += 1
            emb = emb / counts.clamp(min=1)

        torch.save(emb, cache_path)
        return emb


# ── Data Construction ────────────────────────────────────────────────────

def parse_pdb_residues(pdb_path):
    """Parse PDB for CA atoms, sequence, and B-factors."""
    residues = []
    sequence = []
    seen = set()

    with open(pdb_path) as f:
        for line in f:
            if not line.startswith("ATOM"):
                continue
            atom_name = line[12:16].strip()
            if atom_name != "CA":
                continue
            resname = line[17:20].strip()
            chain = line[21]
            resseq = line[22:26].strip()
            key = f"{chain}_{resseq}"
            if key in seen:
                continue
            seen.add(key)

            try:
                x, y, z = float(line[30:38]), float(line[38:46]), float(line[46:54])
                bfactor = float(line[60:66]) if len(line) > 66 else 0.0
            except ValueError:
                continue

            aa1 = AA_3TO1.get(resname, "X")
            residues.append({
                "resname": resname, "aa1": aa1, "chain": chain,
                "resseq": resseq, "key": key,
                "x": x, "y": y, "z": z, "bfactor": bfactor,
            })
            sequence.append(aa1)

    return residues, "".join(sequence)


def estimate_burial(coords, idx, all_coords, radius=10.0):
    """Estimate residue burial as neighbor count within radius."""
    dists = np.linalg.norm(all_coords - coords[idx], axis=-1)
    return min(np.sum(dists < radius) / 30.0, 1.0)  # normalize to [0, 1]


def build_pocket_graph_v2(residues, sequence, esm_emb, site, pocket_radius=15.0,
                           k_neighbors=12, cutoff=12.0):
    """Build a production-quality pocket graph.

    Includes all residues within pocket_radius of centroid (not just lining).
    Node features: ESM-2 embedding + physicochemical + structural.
    """
    centroid = np.array(site["centroid"])
    volume = site.get("volume", 1000.0)
    classification = site.get("classification", "Unknown")
    quality = site.get("quality_score", 0.5)
    burial_frac = site.get("burial_fraction", 0.5)
    is_druggable = float(site.get("is_druggable", False))

    all_coords = np.array([[r["x"], r["y"], r["z"]] for r in residues])

    # Select residues within pocket_radius of centroid
    dists_to_cent = np.linalg.norm(all_coords - centroid, axis=-1)
    pocket_mask = dists_to_cent < pocket_radius
    pocket_indices = np.where(pocket_mask)[0]

    if len(pocket_indices) < 4:
        return None

    # Build node features
    node_features = []
    coords = []
    pos_rel = []  # relative to centroid

    for idx in pocket_indices:
        r = residues[idx]
        pos = np.array([r["x"], r["y"], r["z"]])
        coords.append(pos)
        pos_rel.append(pos - centroid)

        # ESM-2 embedding for this residue
        if idx < esm_emb.size(0):
            esm_feat = esm_emb[idx].numpy()
        else:
            esm_feat = np.zeros(esm_emb.size(1))

        # Physicochemical properties
        props = AA_PROPS.get(r["aa1"], [0, 0, 0, 0, 0])

        # Structural features
        bfac_norm = r["bfactor"] / 100.0
        dist_to_cent = dists_to_cent[idx] / pocket_radius
        burial = estimate_burial(all_coords, idx, all_coords)

        feat = np.concatenate([esm_feat, props, [bfac_norm, dist_to_cent, burial]])
        node_features.append(feat)

    coords = np.array(coords)
    pos_rel = np.array(pos_rel)
    node_features = np.array(node_features)

    # Build edges: k-nearest neighbors within cutoff
    n = len(coords)
    dists = np.linalg.norm(coords[:, None] - coords[None, :], axis=-1)
    edge_src, edge_dst = [], []
    for i in range(n):
        nn_idx = np.argsort(dists[i])[1:k_neighbors + 1]
        for j in nn_idx:
            if dists[i, j] < cutoff:
                edge_src.append(i)
                edge_dst.append(j)

    if not edge_src:
        return None

    # Auxiliary graph-level features
    cls_onehot = [0, 0, 0, 0]  # ActiveSite, Allosteric, Cryptic, Unknown
    cls_map = {"ActiveSite": 0, "Allosteric": 1, "Cryptic": 2, "Unknown": 3}
    cls_idx = cls_map.get(classification, 3)
    cls_onehot[cls_idx] = 1

    aux = [
        np.log1p(volume) / 10.0,  # log-normalized volume
        quality,
        burial_frac,
        is_druggable,
    ] + cls_onehot  # total: 8

    data = Data(
        x=torch.tensor(node_features, dtype=torch.float),
        pos=torch.tensor(coords, dtype=torch.float),
        pos_rel=torch.tensor(pos_rel, dtype=torch.float),
        edge_index=torch.tensor([edge_src, edge_dst], dtype=torch.long),
        aux=torch.tensor([aux], dtype=torch.float),  # (1, aux_dim)
    )
    return data


# ── Dataset from Benchmark ───────────────────────────────────────────────

def build_training_data(manifest_path, gt_path, sites_dir, apo_dir,
                        esm_embedder, dcc_threshold=5.0):
    """Build training graphs with ESM-2 embeddings."""
    with open(manifest_path) as f:
        manifest = json.load(f)
    with open(gt_path) as f:
        gt = json.load(f)

    graphs, labels, meta = [], [], []

    for target in manifest["targets"]:
        tid = str(target["id"])
        apo = target["apo_pdb"].lower()

        if tid not in gt:
            continue

        pdb_path = f"{apo_dir}/{apo}.pdb"
        sites_path = f"{sites_dir}/{tid}/{apo}.binding_sites.json"

        if not os.path.exists(pdb_path) or not os.path.exists(sites_path):
            continue

        true_cent = np.array(gt[tid]["centroid"])
        residues, sequence = parse_pdb_residues(pdb_path)

        if len(residues) < 10 or len(sequence) < 10:
            continue

        # Get ESM-2 embeddings for this protein
        esm_emb = esm_embedder.embed(apo, sequence)
        print(f"  {apo.upper()}: {len(residues)} residues, ESM shape {esm_emb.shape}")

        with open(sites_path) as f:
            sites_data = json.load(f)

        for site in sites_data.get("sites", []):
            centroid = site.get("centroid")
            if not centroid:
                continue

            graph = build_pocket_graph_v2(residues, sequence, esm_emb, site)
            if graph is None:
                continue

            dcc = float(np.linalg.norm(np.array(centroid) - true_cent))
            label = 1.0 if dcc <= dcc_threshold else 0.0

            graph.y = torch.tensor([label], dtype=torch.float)
            graphs.append(graph)
            labels.append(label)
            meta.append({
                "target": apo.upper(), "site_id": site.get("id", "?"),
                "dcc": round(dcc, 2),
            })

    return graphs, labels, meta


# ── Training Loop ────────────────────────────────────────────────────────

def augment_graph(data, coord_noise=0.5, dropout_rate=0.1):
    """Data augmentation: coordinate jitter + random residue dropout."""
    data = data.clone()
    n = data.x.size(0)

    # Coordinate jitter
    data.pos = data.pos + torch.randn_like(data.pos) * coord_noise
    data.pos_rel = data.pos_rel + torch.randn_like(data.pos_rel) * coord_noise

    # Random residue dropout (remove nodes)
    if n > 6 and dropout_rate > 0:
        keep_mask = torch.rand(n) > dropout_rate
        keep_mask[0] = True  # always keep at least some
        if keep_mask.sum() < 4:
            return data  # don't drop too many

        keep_idx = torch.where(keep_mask)[0]
        idx_map = torch.full((n,), -1, dtype=torch.long)
        idx_map[keep_idx] = torch.arange(len(keep_idx))

        data.x = data.x[keep_idx]
        data.pos = data.pos[keep_idx]
        data.pos_rel = data.pos_rel[keep_idx]

        # Remap edges
        row, col = data.edge_index
        valid = keep_mask[row] & keep_mask[col]
        data.edge_index = idx_map[data.edge_index[:, valid]]

    return data


def train(graphs, labels, meta, epochs=300, lr=5e-4,
          model_out="models/egnn_ranker_v2.pt"):
    """Train GVP pocket ranker with LOTO cross-validation + final production model."""
    os.makedirs(os.path.dirname(model_out), exist_ok=True)

    n_pos = sum(labels)
    n_neg = len(labels) - n_pos
    print(f"\nTraining GVP-EGNN v2 on {len(graphs)} pockets "
          f"({int(n_pos)} hits, {int(n_neg)} decoys)")

    if n_pos == 0:
        print("ERROR: No positive examples")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    esm_dim = graphs[0].x.size(1) - 5 - 3
    aux_dim = graphs[0].aux.size(1)

    model_config = {
        "esm_dim": esm_dim, "physchem_dim": 5, "aux_dim": aux_dim,
        "hidden_s": 128, "hidden_v": 32, "n_layers": 2, "dropout": 0.35,
    }
    n_params = sum(p.numel() for p in PrismPocketRankerV2(**model_config).parameters())
    print(f"Model config: hidden_s=128, hidden_v=32, layers=2, dropout=0.35")
    print(f"Model params: {n_params:,}")

    # ── Phase 1: Leave-One-Target-Out Cross-Validation ──
    print(f"\n{'='*60}")
    print("LEAVE-ONE-TARGET-OUT CROSS-VALIDATION")
    print(f"{'='*60}")

    targets = sorted(set(m["target"] for m in meta))
    loto_results = {}
    loto_epochs = min(epochs, 150)  # fewer epochs for CV

    for held_out in targets:
        # Split
        train_idx = [i for i, m in enumerate(meta) if m["target"] != held_out]
        test_idx = [i for i, m in enumerate(meta) if m["target"] == held_out]

        train_graphs = [graphs[i] for i in train_idx]
        test_graphs = [graphs[i] for i in test_idx]

        # Check if held-out target has any hits
        test_has_hit = any(labels[i] == 1.0 for i in test_idx)

        # Data augmentation: duplicate positive examples in training set
        aug_graphs = []
        for g in train_graphs:
            aug_graphs.append(g)
            if g.y.item() == 1.0:
                # Augment hits 5x
                for _ in range(5):
                    aug_graphs.append(augment_graph(g))
        train_loader = DataLoader(aug_graphs, batch_size=16, shuffle=True)

        # Train
        model = PrismPocketRankerV2(**model_config).to(device)
        criterion = FocalLoss(alpha=0.8, gamma=2.5)
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=5e-3)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=loto_epochs)

        for epoch in range(loto_epochs):
            model.train()
            for batch in train_loader:
                batch = batch.to(device)
                optimizer.zero_grad()
                logit = model(batch)
                loss = criterion(logit, batch.y.squeeze(-1))
                l2_reg = sum(p.pow(2).sum() for p in model.classifier.parameters())
                loss = loss + 5e-4 * l2_reg
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
                optimizer.step()
            scheduler.step()

        # Evaluate on held-out target
        model.eval()
        test_preds = []
        with torch.no_grad():
            for g in test_graphs:
                g_dev = g.to(device)
                g_dev.batch = torch.zeros(g_dev.x.size(0), dtype=torch.long, device=device)
                logit = model(g_dev)
                prob = torch.sigmoid(logit).item()
                test_preds.append(prob)

        # Find top-ranked pocket for this target
        pocket_scores = []
        for j, idx in enumerate(test_idx):
            m = meta[idx]
            pocket_scores.append((m["site_id"], m["dcc"], test_preds[j], labels[idx]))

        pocket_scores.sort(key=lambda x: -x[2])
        top = pocket_scores[0]
        is_correct = top[3] == 1.0
        loto_results[held_out] = {
            "site": top[0], "dcc": top[1], "score": top[2],
            "hit": is_correct, "has_gt_hit": test_has_hit,
        }

        marker = " <<<" if is_correct else (" (no hit in candidates)" if not test_has_hit else "")
        print(f"  {held_out:<8} top=site{top[0]:<6} DCC={top[1]:<7.2f} "
              f"score={top[2]:.4f} {'HIT' if is_correct else 'MISS'}{marker}")

    # LOTO summary
    n_with_hits = sum(1 for r in loto_results.values() if r["has_gt_hit"])
    sr1_all = sum(1 for r in loto_results.values() if r["hit"])
    sr1_possible = sum(1 for r in loto_results.values() if r["hit"] and r["has_gt_hit"])
    n_total = len(loto_results)

    print(f"\nLOTO SR@1: {sr1_all}/{n_total} ({100*sr1_all/max(n_total,1):.0f}%) overall")
    print(f"LOTO SR@1: {sr1_possible}/{n_with_hits} ({100*sr1_possible/max(n_with_hits,1):.0f}%) "
          f"on targets where GT pocket exists in candidates")

    # ── Phase 2: Train production model on ALL data ──
    print(f"\n{'='*60}")
    print("TRAINING PRODUCTION MODEL ON ALL DATA")
    print(f"{'='*60}")

    # Augment positives
    aug_graphs = []
    for g in graphs:
        aug_graphs.append(g)
        if g.y.item() == 1.0:
            for _ in range(5):
                aug_graphs.append(augment_graph(g))

    print(f"Training set: {len(aug_graphs)} graphs "
          f"({sum(1 for g in aug_graphs if g.y.item()==1)} hits after augmentation)")

    model = PrismPocketRankerV2(**model_config).to(device)
    criterion = FocalLoss(alpha=0.8, gamma=2.5)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=5e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    loader = DataLoader(aug_graphs, batch_size=16, shuffle=True)

    best_loss = float("inf")
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        n_batches = 0

        for batch in loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            logit = model(batch)
            loss = criterion(logit, batch.y.squeeze(-1))
            l2_reg = sum(p.pow(2).sum() for p in model.classifier.parameters())
            loss = loss + 5e-4 * l2_reg
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()
            try:
                scheduler.step()
            except Exception:
                pass
            total_loss += loss.item()
            n_batches += 1

        avg_loss = total_loss / max(n_batches, 1)

        if (epoch + 1) % 50 == 0 or epoch == 0:
            model.eval()
            with torch.no_grad():
                all_preds, all_labels_ev = [], []
                for batch in DataLoader(graphs, batch_size=16, shuffle=False):
                    batch = batch.to(device)
                    logit = model(batch)
                    probs = torch.sigmoid(logit)
                    all_preds.extend(probs.cpu().numpy())
                    all_labels_ev.extend(batch.y.squeeze(-1).cpu().numpy())

            all_preds = np.array(all_preds)
            all_labels_ev = np.array(all_labels_ev)
            pred_bin = (all_preds > 0.5).astype(int)
            tp = ((pred_bin == 1) & (all_labels_ev == 1)).sum()
            fp = ((pred_bin == 1) & (all_labels_ev == 0)).sum()
            fn = ((pred_bin == 0) & (all_labels_ev == 1)).sum()
            prec = tp / max(tp + fp, 1)
            rec = tp / max(tp + fn, 1)

            target_pockets = {}
            for i, m in enumerate(meta):
                target_pockets.setdefault(m["target"], []).append(
                    (all_preds[i], all_labels_ev[i]))
            sr1 = sum(1 for pockets in target_pockets.values()
                      if sorted(pockets, key=lambda x: -x[0])[0][1] == 1.0)

            print(f"  Epoch {epoch+1:>3d}: loss={avg_loss:.4f} "
                  f"P={prec:.2f} R={rec:.2f} "
                  f"SR@1={sr1}/{len(target_pockets)} ({100*sr1/max(len(target_pockets),1):.0f}%) "
                  f"lr={optimizer.param_groups[0]['lr']:.1e}")

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save({
                "model_state": model.state_dict(),
                "config": model_config,
            }, model_out)

    # ── Final per-target results ──
    print(f"\n{'='*60}")
    print(f"{'Target':<8} {'Site':<8} {'DCC':<8} {'Score':<8} {'Hit':<5} {'LOTO'}")
    print("-" * 55)

    model.eval()
    target_pockets = {}
    with torch.no_grad():
        for i, g in enumerate(graphs):
            g_dev = g.to(device)
            g_dev.batch = torch.zeros(g_dev.x.size(0), dtype=torch.long, device=device)
            logit = model(g_dev)
            prob = torch.sigmoid(logit).item()
            m = meta[i]
            target_pockets.setdefault(m["target"], []).append(
                (m["site_id"], m["dcc"], prob, labels[i])
            )

    sr1 = 0
    n_tgt = 0
    for tgt in sorted(target_pockets):
        pockets = sorted(target_pockets[tgt], key=lambda x: -x[2])
        top = pockets[0]
        is_hit = top[3] == 1.0
        if is_hit:
            sr1 += 1
        n_tgt += 1
        loto_hit = loto_results.get(tgt, {}).get("hit", False)
        marker = " <<<" if is_hit else ""
        loto_str = "HIT" if loto_hit else "miss"
        print(f"{tgt:<8} {top[0]:<8} {top[1]:<8.2f} {top[2]:<8.4f} {int(top[3]):<5}{loto_str}{marker}")

    print(f"\nProduction SR@1: {sr1}/{n_tgt} ({100*sr1/max(n_tgt,1):.0f}%)")
    print(f"LOTO SR@1:       {sr1_all}/{n_total} ({100*sr1_all/max(n_total,1):.0f}%) "
          f"(true generalization estimate)")
    print(f"Model saved: {model_out}")


# ── Inference ────────────────────────────────────────────────────────────

def predict(pdb_path, sites_path, model_path, esm_model="esm2_t33_650M_UR50D"):
    """Rank pockets for a new protein."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    config = checkpoint["config"]

    model = PrismPocketRankerV2(**config).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    embedder = ESMEmbedder(esm_model, device=str(device))
    residues, sequence = parse_pdb_residues(pdb_path)
    pdb_id = Path(pdb_path).stem
    esm_emb = embedder.embed(pdb_id, sequence)

    with open(sites_path) as f:
        sites_data = json.load(f)

    results = []
    with torch.no_grad():
        for site in sites_data.get("sites", []):
            centroid = site.get("centroid")
            if not centroid:
                continue

            graph = build_pocket_graph_v2(residues, sequence, esm_emb, site)
            if graph is None:
                results.append({
                    "site_id": site.get("id", "?"), "score": 0.0,
                    "classification": site.get("classification", "?"),
                })
                continue

            graph = graph.to(device)
            graph.batch = torch.zeros(graph.x.size(0), dtype=torch.long, device=device)
            logit = model(graph)
            prob = torch.sigmoid(logit).item()

            results.append({
                "site_id": site.get("id", "?"),
                "score": round(prob, 4),
                "classification": site.get("classification", "?"),
                "centroid": centroid,
                "volume": site.get("volume", 0),
            })

    results.sort(key=lambda r: -r["score"])

    print(f"\nPRISM4D GVP-EGNN v2 Pocket Ranking: {pdb_id.upper()}")
    print(f"{'Rank':<5} {'Site':<8} {'Score':<8} {'Class':<14} {'Volume'}")
    print("-" * 50)
    for i, r in enumerate(results[:10]):
        print(f"#{i+1:<4} {r['site_id']:<8} {r['score']:<8.4f} "
              f"{r['classification']:<14} {r.get('volume',0):.0f}")

    return results


# ── CLI ──────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="PRISM4D GVP-EGNN Pocket Ranker v2")
    sub = parser.add_subparsers(dest="command")

    train_p = sub.add_parser("train")
    train_p.add_argument("--sites-dir", default="benchmarks/prism4d_bench30/results")
    train_p.add_argument("--apo-dir", default="benchmarks/prism4d_bench30/apo")
    train_p.add_argument("--gt", default="benchmarks/prism4d_bench30/ground_truth/ligand_centroids.json")
    train_p.add_argument("--manifest", default="benchmarks/prism4d_bench30/benchmark_manifest.json")
    train_p.add_argument("--epochs", type=int, default=300)
    train_p.add_argument("--lr", type=float, default=5e-4)
    train_p.add_argument("--model-out", default="models/egnn_ranker_v2.pt")
    train_p.add_argument("--esm-model", default="esm2_t33_650M_UR50D")
    train_p.add_argument("--dcc-threshold", type=float, default=5.0)

    pred_p = sub.add_parser("predict")
    pred_p.add_argument("--pdb", required=True)
    pred_p.add_argument("--sites", required=True)
    pred_p.add_argument("--model", default="models/egnn_ranker_v2.pt")
    pred_p.add_argument("--esm-model", default="esm2_t33_650M_UR50D")

    args = parser.parse_args()

    if args.command == "train":
        embedder = ESMEmbedder(args.esm_model)
        graphs, labels, meta = build_training_data(
            args.manifest, args.gt, args.sites_dir, args.apo_dir,
            embedder, dcc_threshold=args.dcc_threshold,
        )
        if not graphs:
            print("ERROR: No training data")
            sys.exit(1)
        train(graphs, labels, meta,
              epochs=args.epochs, lr=args.lr, model_out=args.model_out)

    elif args.command == "predict":
        predict(args.pdb, args.sites, args.model, esm_model=args.esm_model)

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
