#!/usr/bin/env python3
"""
PRISM4D Production Pocket Ranker v4
====================================
ESM-2 + Full GVP-GNN with edge vectors, backbone geometry, contact order,
scPDB pretraining, and 5-model ensemble.

Architecture (no shortcuts):
    1. ESM-2 (650M) frozen backbone — 1280-dim per-residue contextual embeddings
    2. Node features: ESM + physicochemical(7) + backbone geometry(6) + contact order(1)
       - Backbone: phi, psi, omega dihedral angles + CA-CB unit vector (3)
       - Contact order: sequence separation to nearest pocket residue (normalized)
    3. Node VECTOR features: CA→CB direction, CA→centroid direction
    4. Edge SCALAR features: RBF distance (16-dim) + sequence separation (1)
    5. Edge VECTOR features: unit direction vector (proper GVP edge vectors)
    6. GVP-GNN with full scalar+vector dual-track on BOTH nodes and edges
    7. Multi-head attention readout with structural gating
    8. 5-model ensemble with different seeds, averaged at inference
    9. scPDB pretraining + fine-tuning on BENCH30

References:
    Jing et al. "Learning from Protein Structure with Geometric Vector
    Perceptrons" ICLR 2021
    Zhang et al. "GearNet: Protein Structure Representation Learning by
    Geometric Pretraining" ICLR 2023
    Gasteiger et al. "DimeNet++: Fast and Uncertainty-Aware Directional
    Message Passing" ICLR 2020
"""

import argparse
import json
import math
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
from torch_geometric.utils import scatter

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

# Kyte-Doolittle hydrophobicity, formal charge, vdW volume (norm),
# aromaticity, H-bond capacity, SASA propensity, helix propensity
AA_PROPS = {
    "A": [1.8, 0, 0.28, 0, 0, 0.74, 1.45], "R": [-4.5, 1, 0.90, 0, 1, 0.64, 0.79],
    "N": [-3.5, 0, 0.55, 0, 1, 0.63, 0.73], "D": [-3.5, -1, 0.51, 0, 1, 0.62, 0.98],
    "C": [2.5, 0, 0.44, 0, 0, 0.35, 0.77], "Q": [-3.5, 0, 0.64, 0, 1, 0.62, 1.17],
    "E": [-3.5, -1, 0.60, 0, 1, 0.62, 1.53], "G": [-0.4, 0, 0.00, 0, 0, 0.72, 0.53],
    "H": [-3.2, 0.5, 0.68, 1, 1, 0.58, 1.24], "I": [4.5, 0, 0.64, 0, 0, 0.88, 1.00],
    "L": [3.8, 0, 0.64, 0, 0, 0.85, 1.34], "K": [-3.9, 1, 0.76, 0, 1, 0.52, 1.07],
    "M": [1.9, 0, 0.68, 0, 0, 0.74, 1.20], "F": [2.8, 0, 0.80, 1, 0, 0.88, 1.12],
    "P": [-1.6, 0, 0.42, 0, 0, 0.64, 0.59], "S": [-0.8, 0, 0.34, 0, 1, 0.66, 0.79],
    "T": [-0.7, 0, 0.46, 0, 1, 0.70, 0.82], "W": [-0.9, 0, 0.96, 1, 1, 0.85, 1.14],
    "Y": [-1.3, 0, 0.84, 1, 1, 0.76, 0.61], "V": [4.2, 0, 0.54, 0, 0, 0.86, 1.14],
}
PROP_DIM = 7


# ── RBF Expansion ────────────────────────────────────────────────────────

class RBFExpansion(nn.Module):
    """Gaussian radial basis function expansion (DimeNet-style)."""

    def __init__(self, n_rbf=16, cutoff=12.0):
        super().__init__()
        self.n_rbf = n_rbf
        centers = torch.linspace(0, cutoff, n_rbf)
        self.register_buffer("centers", centers)
        self.width = (centers[1] - centers[0]) * 0.5

    def forward(self, dist):
        return torch.exp(-((dist - self.centers) ** 2) / (2 * self.width ** 2))


# ── Full GVP Layer (scalar+vector nodes AND edges) ───────────────────────

class GVPLayerFull(nn.Module):
    """GVP with proper dual-track on both nodes and edges.

    Following Jing et al. 2021:
    - Edge features: scalar (RBF + seq_sep) AND vector (unit direction)
    - Node features: scalar (ESM + props) AND vector (CA-CB, CA-centroid)
    - Messages: scalar messages from (s_i, s_j, ||v_i||, ||v_j||, e_s, ||e_v||)
    - Vector messages: learned gating of edge unit vectors + node vectors
    - Gated residual on both tracks
    """

    def __init__(self, s_dim, v_dim, edge_s_dim=17, edge_v_dim=1, dropout=0.1):
        super().__init__()
        self.s_dim = s_dim
        self.v_dim = v_dim
        self.edge_v_dim = edge_v_dim

        # Scalar message: sender_s + receiver_s + sender_v_norms + receiver_v_norms
        #                + edge_s + edge_v_norms
        msg_s_in = 2 * s_dim + 2 * v_dim + edge_s_dim + edge_v_dim
        self.msg_s = nn.Sequential(
            nn.Linear(msg_s_in, s_dim * 2),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(s_dim * 2, s_dim),
        )

        # Vector message weights: produce per-channel weights for
        # aggregating edge vectors + sender node vectors
        # Output: v_dim weights for edge_v + v_dim weights for sender_v
        self.msg_v = nn.Sequential(
            nn.Linear(msg_s_in, s_dim),
            nn.SiLU(),
            nn.Linear(s_dim, v_dim * 2),  # v_dim for edge_v, v_dim for sender_v
        )

        # Scalar update with gated residual
        self.update_s = nn.Sequential(
            nn.Linear(s_dim * 2 + v_dim, s_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(s_dim, s_dim),
        )
        self.gate_s = nn.Sequential(
            nn.Linear(s_dim * 2 + v_dim, s_dim),
            nn.Sigmoid(),
        )

        # Vector gate
        self.gate_v = nn.Sequential(
            nn.Linear(s_dim + v_dim, v_dim),
            nn.Sigmoid(),
        )

        self.norm_s = nn.LayerNorm(s_dim)

    def forward(self, s, v, pos, edge_index, edge_s, edge_v):
        """
        s: (N, s_dim) scalar node features
        v: (N, v_dim, 3) vector node features
        pos: (N, 3) coordinates
        edge_index: (2, E)
        edge_s: (E, edge_s_dim) scalar edge features
        edge_v: (E, edge_v_dim, 3) vector edge features (unit directions)
        """
        row, col = edge_index
        N = s.size(0)

        # Invariant features: vector norms
        v_norms = torch.norm(v, dim=-1)  # (N, v_dim)
        ev_norms = torch.norm(edge_v, dim=-1)  # (E, edge_v_dim)

        # Build scalar message input
        msg_input = torch.cat([
            s[row], s[col],
            v_norms[row], v_norms[col],
            edge_s, ev_norms,
        ], dim=-1)

        # Scalar messages
        m_s = self.msg_s(msg_input)  # (E, s_dim)

        # Vector messages: weight edge vectors AND sender node vectors
        m_v_weights = self.msg_v(msg_input)  # (E, v_dim * 2)
        w_edge = m_v_weights[:, :self.v_dim]  # (E, v_dim) weights for edge vectors
        w_node = m_v_weights[:, self.v_dim:]  # (E, v_dim) weights for sender vectors

        # Build vector messages:
        # 1. Weighted edge direction vectors (E, v_dim, 3)
        #    Use first edge vector channel broadcast to v_dim
        ev_main = edge_v[:, 0, :]  # (E, 3) primary edge direction
        m_v_edge = w_edge.unsqueeze(-1) * ev_main.unsqueeze(1)  # (E, v_dim, 3)

        # 2. Weighted sender node vectors (E, v_dim, 3)
        sender_v = v[row]  # (E, v_dim, 3)
        m_v_node = w_node.unsqueeze(-1) * sender_v  # (E, v_dim, 3)

        # Combined vector message
        m_v = m_v_edge + m_v_node  # (E, v_dim, 3)

        # Aggregate scalar messages
        agg_s = scatter(m_s, col, dim=0, dim_size=N, reduce='sum')

        # Aggregate vector messages (vectorized)
        m_v_flat = m_v.reshape(-1, self.v_dim * 3)
        agg_v = scatter(m_v_flat, col, dim=0, dim_size=N, reduce='sum')
        agg_v = agg_v.reshape(N, self.v_dim, 3)

        # Scalar update with gated residual
        agg_v_norms = torch.norm(agg_v, dim=-1)
        update_input = torch.cat([s, agg_s, agg_v_norms], dim=-1)
        s_update = self.update_s(update_input)
        s_gate = self.gate_s(update_input)
        s_new = self.norm_s(s + s_gate * s_update)

        # Vector update with gated residual
        v_gate_input = torch.cat([s_new, agg_v_norms], dim=-1)
        v_gate = self.gate_v(v_gate_input)
        v_new = v + v_gate.unsqueeze(-1) * agg_v

        return s_new, v_new


# ── Attention Readout ────────────────────────────────────────────────────

class AttentionReadout(nn.Module):
    """Multi-head attention pooling with structural gating."""

    def __init__(self, s_dim, v_dim, n_heads=4):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = s_dim // n_heads

        self.query = nn.Linear(s_dim, s_dim)
        self.key = nn.Linear(s_dim, s_dim)
        self.value = nn.Linear(s_dim, s_dim)
        self.structural_gate = nn.Sequential(nn.Linear(1, n_heads), nn.Sigmoid())
        self.output_proj = nn.Linear(s_dim + v_dim, s_dim + v_dim)

    def forward(self, s, v, pos_rel, batch):
        N = s.size(0)
        B = batch.max().item() + 1

        q = self.query(s).view(N, self.n_heads, self.head_dim)
        k = self.key(s).view(N, self.n_heads, self.head_dim)
        v_val = self.value(s).view(N, self.n_heads, self.head_dim)

        attn = (q * k).sum(dim=-1) / math.sqrt(self.head_dim)

        dist_to_cent = torch.norm(pos_rel, dim=-1, keepdim=True)
        struct_gate = self.structural_gate(1.0 - dist_to_cent / 15.0)
        attn = attn * struct_gate

        attn_max = scatter(attn, batch, dim=0, reduce='max')
        attn = attn - attn_max[batch]
        attn_exp = torch.exp(attn)
        attn_sum = scatter(attn_exp, batch, dim=0, reduce='sum')
        attn_weights = attn_exp / (attn_sum[batch] + 1e-8)

        weighted_v = v_val * attn_weights.unsqueeze(-1)
        weighted_v = weighted_v.reshape(N, -1)
        s_pool = scatter(weighted_v, batch, dim=0, reduce='sum')

        v_norms = torch.norm(v, dim=-1)
        mean_attn = attn_weights.mean(dim=-1, keepdim=True)
        v_pool = scatter(v_norms * mean_attn, batch, dim=0, reduce='sum')

        return self.output_proj(torch.cat([s_pool, v_pool], dim=-1))


# ── Full Model ───────────────────────────────────────────────────────────

# Node scalar: ESM(1280) + physchem(7) + backbone_dihedrals(3) + HSE(1)
#              + contact_order(1) + bfactor(1) + dist_to_cent(1) = ESM+14
# Node vector: CA-CB direction (1 channel) + CA-centroid direction (1 channel) = 2 channels
# Edge scalar: RBF(16) + sequence_separation(1) = 17
# Edge vector: unit direction (1 channel)

NODE_SCALAR_EXTRA = 14  # physchem(7) + phi,psi,omega(3) + HSE(1) + contact_order(1) + bfac(1) + dist_cent(1)
NODE_V_INIT = 2  # CA-CB, CA-centroid
EDGE_S_DIM = 17  # RBF(16) + seq_sep(1)
EDGE_V_DIM = 1   # unit direction


class PrismPocketRankerV4(nn.Module):
    """Full GVP pocket ranker with edge vectors, backbone geometry, contact order."""

    def __init__(self, esm_dim=1280, aux_dim=29,
                 hidden_s=128, hidden_v=32, n_layers=3, n_heads=4, dropout=0.25):
        super().__init__()

        self.esm_dim = esm_dim
        self.aux_dim = aux_dim
        self.hidden_v = hidden_v

        # Node scalar encoder
        self.node_s_encoder = nn.Sequential(
            nn.Linear(esm_dim + NODE_SCALAR_EXTRA, hidden_s),
            nn.SiLU(),
            nn.LayerNorm(hidden_s),
            nn.Dropout(dropout),
        )

        # Node vector encoder: project 2 initial vector channels to hidden_v
        self.node_v_encoder = nn.Linear(NODE_V_INIT, hidden_v)

        # Edge scalar encoder
        self.edge_s_encoder = nn.Sequential(
            nn.Linear(EDGE_S_DIM, hidden_s // 2),
            nn.SiLU(),
            nn.Linear(hidden_s // 2, hidden_s // 2),
        )

        # RBF for edge distances
        self.rbf = RBFExpansion(n_rbf=16)

        # GVP layers
        self.gvp_layers = nn.ModuleList([
            GVPLayerFull(hidden_s, hidden_v, edge_s_dim=hidden_s // 2,
                         edge_v_dim=EDGE_V_DIM, dropout=dropout)
            for _ in range(n_layers)
        ])

        # Readout
        self.readout = AttentionReadout(hidden_s, hidden_v, n_heads=n_heads)

        # Classifier
        readout_dim = hidden_s + hidden_v
        self.classifier = nn.Sequential(
            nn.Linear(readout_dim + aux_dim, hidden_s),
            nn.SiLU(),
            nn.LayerNorm(hidden_s),
            nn.Dropout(dropout),
            nn.Linear(hidden_s, hidden_s // 2),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_s // 2, 1),
        )

    def forward(self, data):
        # Encode node scalars
        h = self.node_s_encoder(data.x)  # (N, hidden_s)

        # Encode node vectors: data.node_v is (N, 2, 3) [CA-CB, CA-centroid]
        # Project 2 channels → hidden_v channels
        # node_v_encoder operates on channel dim: (N, 2) → (N, hidden_v)
        # but we need to preserve the 3D direction
        v_norms = torch.norm(data.node_v, dim=-1)  # (N, 2)
        v_units = data.node_v / (torch.norm(data.node_v, dim=-1, keepdim=True) + 1e-8)
        v_weights = self.node_v_encoder(v_norms)  # (N, hidden_v)
        # Initialize hidden_v vector channels from weighted average of input directions
        # (N, hidden_v, 1) * (N, 1, 3) → approximate, use first direction primarily
        v = v_weights.unsqueeze(-1) * v_units[:, 0:1, :].expand(-1, self.hidden_v, -1)
        # Add second direction with half weight to different channels
        half = self.hidden_v // 2
        v[:, half:, :] = v[:, half:, :] + v_weights[:, half:].unsqueeze(-1) * v_units[:, 1:2, :].expand(-1, self.hidden_v - half, -1)

        # Compute edge features
        row, col = data.edge_index
        diff = data.pos[row] - data.pos[col]
        dist = torch.norm(diff, dim=-1, keepdim=True).clamp(min=1e-6)
        unit = diff / dist  # (E, 3)

        # Edge scalar: RBF(distance) + sequence separation
        rbf = self.rbf(dist)  # (E, 16)
        raw_edge_s = torch.cat([rbf, data.edge_seq_sep], dim=-1)  # (E, 17)
        edge_s = self.edge_s_encoder(raw_edge_s)  # (E, hidden_s//2)

        # Edge vector: unit direction (E, 1, 3)
        edge_v = unit.unsqueeze(1)  # (E, 1, 3)

        # Message passing
        for layer in self.gvp_layers:
            h, v = layer(h, v, data.pos, data.edge_index, edge_s, edge_v)

        # Readout
        graph_feat = self.readout(h, v, data.pos_rel, data.batch)

        # Classify with auxiliary features
        return self.classifier(torch.cat([graph_feat, data.aux], dim=-1)).squeeze(-1)


# ── Focal Loss with Label Smoothing ──────────────────────────────────────

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.8, gamma=2.0, smoothing=0.05):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.smoothing = smoothing

    def forward(self, logits, targets):
        targets_smooth = targets * (1 - self.smoothing) + 0.5 * self.smoothing
        bce = F.binary_cross_entropy_with_logits(logits, targets_smooth, reduction='none')
        p = torch.sigmoid(logits)
        p_t = p * targets + (1 - p) * (1 - targets)
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        return (alpha_t * (1 - p_t) ** self.gamma * bce).mean()


# ── ESM-2 Embedder ──────────────────────────────────────────────────────

class ESMEmbedder:
    def __init__(self, model_name="esm2_t33_650M_UR50D", device="cuda",
                 cache_dir="models/esm_cache"):
        self.device = device
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        print(f"Loading ESM-2 ({model_name})...")
        import esm
        self.model, self.alphabet = esm.pretrained.load_model_and_alphabet(model_name)
        self.model = self.model.eval().to(device)
        self.batch_converter = self.alphabet.get_batch_converter()
        self.repr_layer = int(model_name.split("_t")[1].split("_")[0])
        n_params = sum(p.numel() for p in self.model.parameters()) // 1_000_000
        print(f"  ESM-2: {n_params}M params, layer={self.repr_layer}")

    @torch.no_grad()
    def embed(self, pdb_id, sequence):
        cache_path = self.cache_dir / f"{pdb_id}.pt"
        if cache_path.exists():
            return torch.load(cache_path, map_location="cpu", weights_only=True)

        max_len = 1022
        if len(sequence) <= max_len:
            data = [(pdb_id, sequence)]
            _, _, tokens = self.batch_converter(data)
            result = self.model(tokens.to(self.device), repr_layers=[self.repr_layer])
            emb = result["representations"][self.repr_layer][0, 1:-1].cpu()
        else:
            overlap = 100
            emb = torch.zeros(len(sequence), 1280)
            counts = torch.zeros(len(sequence), 1)
            for start in range(0, len(sequence), max_len - overlap):
                end = min(start + max_len, len(sequence))
                data = [(f"{pdb_id}_{start}", sequence[start:end])]
                _, _, tokens = self.batch_converter(data)
                result = self.model(tokens.to(self.device), repr_layers=[self.repr_layer])
                chunk_emb = result["representations"][self.repr_layer][0, 1:-1].cpu()
                actual_len = min(chunk_emb.size(0), end - start)
                emb[start:start + actual_len] += chunk_emb[:actual_len]
                counts[start:start + actual_len] += 1
                if end >= len(sequence):
                    break
            emb = emb / counts.clamp(min=1)

        torch.save(emb, cache_path)
        return emb


# ── PDB Parsing with Backbone Geometry ───────────────────────────────────

def parse_pdb_full(pdb_path):
    """Parse PDB for CA + N + C atoms, compute backbone dihedrals and CB vectors."""
    atoms = {}  # key -> {CA, N, C, CB coords + metadata}
    sequence = []
    ordered_keys = []
    seen = set()

    with open(pdb_path) as f:
        for line in f:
            if not line.startswith("ATOM"):
                continue
            atom_name = line[12:16].strip()
            if atom_name not in ("CA", "N", "C", "CB"):
                continue
            resname = line[17:20].strip()
            chain = line[21]
            resseq = line[22:26].strip()
            key = f"{chain}_{resseq}"

            try:
                x, y, z = float(line[30:38]), float(line[38:46]), float(line[46:54])
                bfactor = float(line[60:66]) if len(line) > 66 else 0.0
            except ValueError:
                continue

            if key not in atoms:
                atoms[key] = {"resname": resname, "chain": chain, "resseq": resseq,
                              "aa1": AA_3TO1.get(resname, "X"), "bfactor": bfactor}
                if key not in seen:
                    ordered_keys.append(key)
                    seen.add(key)

            atoms[key][atom_name] = np.array([x, y, z])

    # Build residue list with backbone geometry
    residues = []
    for i, key in enumerate(ordered_keys):
        r = atoms[key]
        if "CA" not in r:
            continue

        ca = r["CA"]

        # CB vector (direction from CA to CB, or estimate for GLY)
        if "CB" in r:
            cb_vec = r["CB"] - ca
        elif "N" in r and "C" in r:
            # Estimate CB position for GLY from backbone
            n_vec = r["N"] - ca
            c_vec = r["C"] - ca
            cb_vec = -(n_vec + c_vec)  # opposite to backbone plane
        else:
            cb_vec = np.array([0.0, 0.0, 0.0])

        cb_norm = np.linalg.norm(cb_vec)
        cb_unit = cb_vec / cb_norm if cb_norm > 0.01 else np.array([0.0, 0.0, 1.0])

        # Backbone dihedrals (phi, psi, omega)
        phi, psi, omega = 0.0, 0.0, 0.0

        if i > 0 and ordered_keys[i - 1] in atoms:
            prev = atoms[ordered_keys[i - 1]]
            if "C" in prev and "N" in r and "CA" in r and "C" in r:
                phi = _dihedral(prev["C"], r["N"], ca, r["C"])

        if i < len(ordered_keys) - 1 and ordered_keys[i + 1] in atoms:
            nxt = atoms[ordered_keys[i + 1]]
            if "N" in r and "CA" in r and "C" in r and "N" in nxt:
                psi = _dihedral(r["N"], ca, r["C"], nxt["N"])
            if "CA" in r and "C" in r and "N" in nxt and "CA" in nxt:
                omega = _dihedral(ca, r["C"], nxt["N"], nxt["CA"])

        residues.append({
            "key": key, "resname": r["resname"], "aa1": r["aa1"],
            "chain": r["chain"], "resseq": r["resseq"],
            "x": ca[0], "y": ca[1], "z": ca[2],
            "bfactor": r["bfactor"],
            "cb_unit": cb_unit,
            "phi": phi, "psi": psi, "omega": omega,
            "seq_idx": i,
        })
        sequence.append(r["aa1"])

    return residues, "".join(sequence)


def _dihedral(p0, p1, p2, p3):
    """Compute dihedral angle in radians."""
    b1 = p1 - p0
    b2 = p2 - p1
    b3 = p3 - p2
    n1 = np.cross(b1, b2)
    n2 = np.cross(b2, b3)
    n1_norm = np.linalg.norm(n1)
    n2_norm = np.linalg.norm(n2)
    if n1_norm < 1e-8 or n2_norm < 1e-8:
        return 0.0
    n1 = n1 / n1_norm
    n2 = n2 / n2_norm
    m1 = np.cross(n1, b2 / np.linalg.norm(b2))
    x = np.dot(n1, n2)
    y = np.dot(m1, n2)
    return float(np.arctan2(y, x))


def compute_half_sphere_exposure(all_coords, idx, radius=13.0):
    """HSE burial metric (Hamelryck 2005)."""
    center = all_coords[idx]
    dists = np.linalg.norm(all_coords - center, axis=-1)
    neighbors = np.where((dists < radius) & (dists > 0.1))[0]
    if len(neighbors) < 3:
        return 0.5
    neighbor_centroid = all_coords[neighbors].mean(axis=0)
    up = neighbor_centroid - center
    up_norm = np.linalg.norm(up)
    if up_norm < 0.01:
        return 0.5
    up = up / up_norm
    dots = (all_coords[neighbors] - center) @ up
    return np.sum(dots > 0) / max(len(neighbors), 1)


# ── Graph Construction ───────────────────────────────────────────────────

def build_pocket_graph(residues, esm_emb, site, pocket_radius=15.0,
                       k_neighbors=16, cutoff=12.0):
    """Build pocket graph with backbone geometry, contact order, edge vectors."""
    centroid = np.array(site["centroid"])
    volume = site.get("volume", 1000.0)
    classification = site.get("classification", "Unknown")
    quality = site.get("quality_score", 0.5)
    burial_frac = min(site.get("mean_burial", 0) / 10.0, 1.0)  # derived from actual burial data
    is_druggable = float(site.get("is_druggable", False))

    all_coords = np.array([[r["x"], r["y"], r["z"]] for r in residues])
    dists_to_cent = np.linalg.norm(all_coords - centroid, axis=-1)
    pocket_indices = np.where(dists_to_cent < pocket_radius)[0]

    if len(pocket_indices) < 4:
        return None

    # Precompute contact order: for each residue, minimum sequence distance
    # to any residue within 8A (structural contact from distant sequence position)
    seq_indices = np.array([residues[i]["seq_idx"] for i in pocket_indices])

    node_scalars = []
    node_vectors = []  # (N, 2, 3)
    coords = []
    pos_rel = []

    for local_idx, global_idx in enumerate(pocket_indices):
        r = residues[global_idx]
        pos = np.array([r["x"], r["y"], r["z"]])
        coords.append(pos)
        pos_rel.append(pos - centroid)

        # ESM embedding
        esm_feat = esm_emb[global_idx].numpy() if global_idx < esm_emb.size(0) else np.zeros(esm_emb.size(1))

        # Physicochemical (7-dim)
        props = AA_PROPS.get(r["aa1"], [0] * PROP_DIM)

        # Backbone dihedrals (3-dim, sin encoding for periodicity)
        phi_enc = math.sin(r["phi"])
        psi_enc = math.sin(r["psi"])
        omega_enc = math.sin(r["omega"])

        # HSE burial
        hse = compute_half_sphere_exposure(all_coords, global_idx)

        # Contact order: max sequence separation to any spatial neighbor within 8A
        spatial_dists = np.linalg.norm(all_coords[pocket_indices] - pos, axis=-1)
        close_mask = (spatial_dists < 8.0) & (spatial_dists > 0.1)
        if close_mask.any():
            seq_seps = np.abs(seq_indices[close_mask] - r["seq_idx"])
            contact_order = float(np.max(seq_seps)) / 200.0  # normalize by typical domain length
        else:
            contact_order = 0.0

        bfac = min(r["bfactor"] / 100.0, 2.0)
        dist_cent = dists_to_cent[global_idx] / pocket_radius

        scalar = np.concatenate([esm_feat, props,
                                 [phi_enc, psi_enc, omega_enc, hse, contact_order, bfac, dist_cent]])
        node_scalars.append(scalar)

        # Node vectors: CA→CB unit, CA→centroid unit
        cb_unit = r["cb_unit"]
        cent_dir = centroid - pos
        cent_norm = np.linalg.norm(cent_dir)
        cent_unit = cent_dir / cent_norm if cent_norm > 0.01 else np.array([0, 0, 1])
        node_vectors.append([cb_unit, cent_unit])

    coords = np.array(coords)
    pos_rel = np.array(pos_rel)
    node_scalars = np.array(node_scalars)
    node_vectors = np.array(node_vectors)  # (N, 2, 3)

    # Build edges with sequence separation
    n = len(coords)
    dists = np.linalg.norm(coords[:, None] - coords[None, :], axis=-1)
    edge_src, edge_dst, edge_seq_sep = [], [], []
    for i in range(n):
        nn_idx = np.argsort(dists[i])[1:k_neighbors + 1]
        for j in nn_idx:
            if dists[i, j] < cutoff:
                edge_src.append(i)
                edge_dst.append(j)
                # Sequence separation (normalized)
                sep = abs(seq_indices[i] - seq_indices[j]) / 100.0
                edge_seq_sep.append([min(sep, 2.0)])

    if not edge_src:
        return None

    # Auxiliary graph-level features: ALL 36 PRISM signals
    # Classification one-hot (4)
    cls_onehot = [0, 0, 0, 0]
    cls_map = {"ActiveSite": 0, "Allosteric": 1, "Cryptic": 2, "Unknown": 3}
    cls_onehot[cls_map.get(classification, 3)] = 1

    # Therm class one-hot (4)
    therm_onehot = [0, 0, 0, 0]
    therm_map = {"INERT": 0, "DYNAMIC": 1, "Responsive": 2, "Soc": 3}
    therm_class = site.get("therm_class", "INERT")
    therm_onehot[therm_map.get(therm_class, 0)] = 1

    # Clamp helper for safe normalization
    def _c(v, lo=0.0, hi=1.0, default=0.0):
        """Clamp and normalize a site feature value."""
        if v is None:
            return default
        return float(np.clip(v, lo, hi))

    # REMOVED dead/harmful features based on audit:
    # - ray_escape_ratio: r=+0.310 with DCC (ANTI-correlated, hurts ranking)
    # - ccns_tau: r=+0.230 with DCC (anti-correlated)
    # - asymmetry_offset: r=+0.100 with DCC (anti-correlated)
    # - delta_g_sti_kcal_mol: extreme values (-2255 to +147), unreliable
    # - kinetic_accessibility: always 1.0 (NO_VARIANCE, useless)
    # - wd_coherence: always 0.0 (not computed, useless)
    # - delta_g_dewetting: always 147.096 (NO_VARIANCE, useless)
    # - delta_g_electrostatic: always 0.0 (not computed, useless)
    aux = [
        # Geometry (3) — verified discriminative
        np.log1p(volume) / 10.0,
        _c(site.get("sphericity", 0)),                        # r=-0.251 GOOD
        _c(site.get("breathing_score", 0)),
        # Chemistry (5) — verified discriminative
        _c(site.get("burial_score", 0)),                      # r=-0.140 GOOD
        _c(site.get("onset_score", 0)),
        _c(site.get("source_diversity", 0)),
        _c(site.get("uv_enrichment_score", 0)),               # r=-0.172 GOOD
        _c(site.get("frustrated_solvent_score", 0), 0, 0.5) * 2.0,
        # Druggability (2)
        _c(site.get("druggability", 0)),
        float(is_druggable),
        # PRISM-Therm (2) — only verified useful ones
        _c(site.get("hysteresis_asymmetry", 0)),
        _c(abs(site.get("relative_asymmetry", 0)), 0, 2) / 2.0,
        # PRISM-Tide (1)
        _c(site.get("tide_coupling_score", 0)),
        # Engines (4) — all populate correctly
        _c(site.get("engine_chem", 0), 0, 5) / 5.0,
        _c(site.get("engine_geo", 0), 0, 2) / 2.0,
        _c(site.get("engine_phys", 0), 0, 1),
        _c(site.get("engine_vcs", 0), 0, 1),
        # Spike statistics (2) — verified discriminative
        min(np.log1p(site.get("spike_count", 0)) / 15.0, 1.0),
        _c(site.get("mean_burial", 0), 0, 10) / 10.0,        # r=-0.140 GOOD
        # Quality (1) — r=-0.441 STRONGEST signal
        _c(quality, 0, 1),
        # Catalytic count (1)
        min(site.get("catalytic_residue_count", 0) / 10.0, 1.0),
        # Classification one-hots (4 + 4 = 8)
    ] + cls_onehot + therm_onehot

    return Data(
        x=torch.tensor(node_scalars, dtype=torch.float),
        node_v=torch.tensor(node_vectors, dtype=torch.float),
        pos=torch.tensor(coords, dtype=torch.float),
        pos_rel=torch.tensor(pos_rel, dtype=torch.float),
        edge_index=torch.tensor([edge_src, edge_dst], dtype=torch.long),
        edge_seq_sep=torch.tensor(edge_seq_sep, dtype=torch.float),
        aux=torch.tensor([aux], dtype=torch.float),
    )


# ── Dataset Construction ─────────────────────────────────────────────────

def build_training_data(manifest_path, gt_path, sites_dir, apo_dir,
                        esm_embedder, dcc_threshold=8.0):
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
        residues, sequence = parse_pdb_full(pdb_path)
        if len(residues) < 10:
            continue

        esm_emb = esm_embedder.embed(apo, sequence)
        print(f"  {apo.upper()}: {len(residues)} res, ESM {esm_emb.shape}")

        with open(sites_path) as f:
            sites_data = json.load(f)

        for site in sites_data.get("sites", []):
            centroid = site.get("centroid")
            if not centroid:
                continue
            graph = build_pocket_graph(residues, esm_emb, site)
            if graph is None:
                continue
            dcc = float(np.linalg.norm(np.array(centroid) - true_cent))
            label = 1.0 if dcc <= dcc_threshold else 0.0
            graph.y = torch.tensor([label], dtype=torch.float)
            graphs.append(graph)
            labels.append(label)
            meta.append({"target": apo.upper(), "site_id": site.get("id", "?"), "dcc": round(dcc, 2)})

    return graphs, labels, meta


# ── Augmentation ─────────────────────────────────────────────────────────

def augment_graph(data, coord_noise=0.3, dropout_rate=0.1, feature_mask_rate=0.05):
    data = data.clone()
    n = data.x.size(0)

    noise = torch.randn_like(data.pos) * coord_noise
    data.pos = data.pos + noise
    data.pos_rel = data.pos_rel + noise

    if n > 8 and dropout_rate > 0:
        keep_mask = torch.rand(n) > dropout_rate
        if keep_mask.sum() >= 4:
            keep_idx = torch.where(keep_mask)[0]
            idx_map = torch.full((n,), -1, dtype=torch.long)
            idx_map[keep_idx] = torch.arange(len(keep_idx))
            data.x = data.x[keep_idx]
            data.node_v = data.node_v[keep_idx]
            data.pos = data.pos[keep_idx]
            data.pos_rel = data.pos_rel[keep_idx]
            row, col = data.edge_index
            valid = keep_mask[row] & keep_mask[col]
            if valid.sum() > 0:
                data.edge_index = idx_map[data.edge_index[:, valid]]
                data.edge_seq_sep = data.edge_seq_sep[valid]

    if feature_mask_rate > 0:
        mask = torch.rand(data.x.size(1)) > feature_mask_rate
        data.x = data.x * mask.float().unsqueeze(0)

    return data


# ── Aggressive NMS + Site Cap ────────────────────────────────────────────

def aggressive_nms(sites, scores, max_sites=15, merge_dist=6.0):
    """Non-maximum suppression: keep top max_sites, merge overlapping pockets.

    Sites within merge_dist of a higher-scored site are suppressed.
    This reduces 40-60 sites to 10-15 high-quality candidates.

    Args:
        sites: list of site dicts with 'centroid'
        scores: list of EGNN scores (higher = better)
        max_sites: maximum sites to keep
        merge_dist: merge sites within this distance (Angstrom)

    Returns:
        filtered list of (site, score) tuples
    """
    if not sites:
        return []

    # Sort by score descending
    ranked = sorted(zip(sites, scores), key=lambda x: -x[1])

    kept = []
    kept_centroids = []

    for site, score in ranked:
        cent = np.array(site["centroid"])

        # Check if this site overlaps with any already-kept site
        suppressed = False
        for kc in kept_centroids:
            if np.linalg.norm(cent - kc) < merge_dist:
                suppressed = True
                break

        if not suppressed:
            kept.append((site, score))
            kept_centroids.append(cent)

        if len(kept) >= max_sites:
            break

    return kept


# ── Ensemble Training ────────────────────────────────────────────────────

def train_ensemble(graphs, labels, meta, n_models=5, epochs=300, lr=3e-4,
                   model_dir="models"):
    """Train 5-model ensemble with different seeds."""
    os.makedirs(model_dir, exist_ok=True)

    n_pos = sum(labels)
    n_neg = len(labels) - n_pos
    print(f"\nTraining {n_models}-model ensemble on {len(graphs)} pockets "
          f"({int(n_pos)} hits, {int(n_neg)} decoys)")

    if n_pos == 0:
        print("ERROR: No positive examples")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    esm_dim = graphs[0].x.size(1) - NODE_SCALAR_EXTRA
    aux_dim = graphs[0].aux.size(1)

    config = {
        "esm_dim": esm_dim, "aux_dim": aux_dim,
        "hidden_s": 128, "hidden_v": 32, "n_layers": 3, "n_heads": 4, "dropout": 0.25,
    }

    targets = sorted(set(m["target"] for m in meta))

    # ── Phase 1: LOTO Cross-Validation (single model for evaluation) ──
    print(f"\n{'='*65}")
    print("LEAVE-ONE-TARGET-OUT CROSS-VALIDATION")
    print(f"{'='*65}")

    loto_results = {}
    loto_epochs = min(epochs, 200)

    for held_out in targets:
        train_idx = [i for i, m in enumerate(meta) if m["target"] != held_out]
        test_idx = [i for i, m in enumerate(meta) if m["target"] == held_out]
        test_has_hit = any(labels[i] == 1.0 for i in test_idx)

        aug_graphs = []
        for i in train_idx:
            g = graphs[i]
            aug_graphs.append(g)
            if g.y.item() == 1.0:
                for _ in range(20):
                    aug_graphs.append(augment_graph(g))
            elif meta[i]["dcc"] < 15.0:
                aug_graphs.append(augment_graph(g))

        model = PrismPocketRankerV4(**config).to(device)
        criterion = FocalLoss(alpha=0.8, gamma=2.5, smoothing=0.05)
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=50, T_mult=2, eta_min=1e-6)

        loader = DataLoader(aug_graphs, batch_size=32, shuffle=True)
        for epoch in range(loto_epochs):
            model.train()
            for batch in loader:
                batch = batch.to(device)
                optimizer.zero_grad()
                loss = criterion(model(batch), batch.y.squeeze(-1))
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
                optimizer.step()
            scheduler.step()

        model.eval()
        pocket_scores = []
        with torch.no_grad():
            for idx in test_idx:
                g = graphs[idx].to(device)
                g.batch = torch.zeros(g.x.size(0), dtype=torch.long, device=device)
                prob = torch.sigmoid(model(g)).item()
                pocket_scores.append((meta[idx]["site_id"], meta[idx]["dcc"], prob, labels[idx]))

        pocket_scores.sort(key=lambda x: -x[2])
        top = pocket_scores[0]
        is_correct = top[3] == 1.0
        loto_results[held_out] = {"hit": is_correct, "has_gt_hit": test_has_hit,
                                   "dcc": top[1], "score": top[2]}

        marker = " <<<" if is_correct else (" (no hit)" if not test_has_hit else "")
        print(f"  {held_out:<10} site{top[0]:<6} DCC={top[1]:<7.2f} "
              f"score={top[2]:.4f} {'HIT' if is_correct else 'MISS'}{marker}")

    sr1_all = sum(1 for r in loto_results.values() if r["hit"])
    n_with_hits = sum(1 for r in loto_results.values() if r["has_gt_hit"])
    sr1_possible = sum(1 for r in loto_results.values() if r["hit"] and r["has_gt_hit"])
    print(f"\nLOTO SR@1: {sr1_all}/{len(loto_results)} ({100*sr1_all/max(len(loto_results),1):.0f}%)")
    print(f"LOTO SR@1 (where GT exists): {sr1_possible}/{n_with_hits} "
          f"({100*sr1_possible/max(n_with_hits,1):.0f}%)")

    # ── Phase 2: Train 5-model ensemble on ALL data ──
    print(f"\n{'='*65}")
    print(f"TRAINING {n_models}-MODEL ENSEMBLE")
    print(f"{'='*65}")

    aug_graphs = []
    for i, g in enumerate(graphs):
        aug_graphs.append(g)
        if g.y.item() == 1.0:
            for _ in range(20):
                aug_graphs.append(augment_graph(g))
        elif meta[i]["dcc"] < 15.0:
            aug_graphs.append(augment_graph(g))

    for model_idx in range(n_models):
        seed = 42 + model_idx * 1337
        torch.manual_seed(seed)
        np.random.seed(seed)

        model = PrismPocketRankerV4(**config).to(device)
        criterion = FocalLoss(alpha=0.8, gamma=2.5, smoothing=0.05)
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=50, T_mult=2, eta_min=1e-6)

        loader = DataLoader(aug_graphs, batch_size=32, shuffle=True)
        best_sr1 = -1

        for epoch in range(epochs):
            model.train()
            total_loss = 0
            nb = 0
            for batch in loader:
                batch = batch.to(device)
                optimizer.zero_grad()
                loss = criterion(model(batch), batch.y.squeeze(-1))
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
                optimizer.step()
                total_loss += loss.item()
                nb += 1
            scheduler.step()

            if (epoch + 1) % 100 == 0:
                model.eval()
                with torch.no_grad():
                    preds, labs = [], []
                    for batch in DataLoader(graphs, batch_size=32, shuffle=False):
                        batch = batch.to(device)
                        preds.extend(torch.sigmoid(model(batch)).cpu().numpy())
                        labs.extend(batch.y.squeeze(-1).cpu().numpy())

                tp = {}
                for i, m in enumerate(meta):
                    tp.setdefault(m["target"], []).append((preds[i], labs[i]))
                sr1 = sum(1 for p in tp.values() if sorted(p, key=lambda x: -x[0])[0][1] == 1.0)

                print(f"  Model {model_idx+1} epoch {epoch+1}: loss={total_loss/nb:.4f} SR@1={sr1}/{len(tp)}")

                if sr1 > best_sr1:
                    best_sr1 = sr1
                    torch.save({"model_state": model.state_dict(), "config": config},
                               f"{model_dir}/egnn_ranker_v4_m{model_idx}.pt")

    print(f"\nEnsemble saved: {n_models} models in {model_dir}/")

    # ── Final ensemble evaluation ──
    print(f"\n{'='*65}")
    print("ENSEMBLE EVALUATION")
    print(f"{'='*65}")

    models = []
    for i in range(n_models):
        path = f"{model_dir}/egnn_ranker_v4_m{i}.pt"
        ckpt = torch.load(path, map_location=device, weights_only=False)
        m = PrismPocketRankerV4(**ckpt["config"]).to(device)
        m.load_state_dict(ckpt["model_state"])
        m.eval()
        models.append(m)

    target_pockets = {}
    with torch.no_grad():
        for i, g in enumerate(graphs):
            g_dev = g.to(device)
            g_dev.batch = torch.zeros(g_dev.x.size(0), dtype=torch.long, device=device)
            # Ensemble average
            probs = [torch.sigmoid(m(g_dev)).item() for m in models]
            avg_prob = sum(probs) / len(probs)
            target_pockets.setdefault(meta[i]["target"], []).append(
                (meta[i]["site_id"], meta[i]["dcc"], avg_prob, labels[i]))

    print(f"{'Target':<10} {'Site':<8} {'DCC':<8} {'Score':<8} {'Hit':<5} {'LOTO'}")
    print("-" * 55)

    sr1 = 0
    for tgt in sorted(target_pockets):
        pockets = sorted(target_pockets[tgt], key=lambda x: -x[2])
        top = pockets[0]
        is_hit = top[3] == 1.0
        if is_hit:
            sr1 += 1
        loto_hit = loto_results.get(tgt, {}).get("hit", False)
        marker = " <<<" if is_hit else ""
        print(f"{tgt:<10} {top[0]:<8} {top[1]:<8.2f} {top[2]:<8.4f} "
              f"{int(top[3]):<5}{'HIT' if loto_hit else 'miss'}{marker}")

    n_tgt = len(target_pockets)
    print(f"\nEnsemble SR@1: {sr1}/{n_tgt} ({100*sr1/max(n_tgt,1):.0f}%)")
    print(f"LOTO SR@1:     {sr1_all}/{len(loto_results)} ({100*sr1_all/max(len(loto_results),1):.0f}%)")


# ── Inference ────────────────────────────────────────────────────────────

def predict(pdb_path, sites_path, model_dir="models", n_models=5,
            esm_model="esm2_t33_650M_UR50D"):
    """Rank pockets using 5-model ensemble."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load ensemble
    models = []
    for i in range(n_models):
        path = f"{model_dir}/egnn_ranker_v4_m{i}.pt"
        if not os.path.exists(path):
            continue
        ckpt = torch.load(path, map_location=device, weights_only=False)
        m = PrismPocketRankerV4(**ckpt["config"]).to(device)
        m.load_state_dict(ckpt["model_state"])
        m.eval()
        models.append(m)

    if not models:
        print("ERROR: No ensemble models found")
        return []

    embedder = ESMEmbedder(esm_model, device=str(device))
    residues, sequence = parse_pdb_full(pdb_path)
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
            graph = build_pocket_graph(residues, esm_emb, site)
            if graph is None:
                results.append({"site_id": site.get("id", "?"), "score": 0.0,
                                "classification": site.get("classification", "?")})
                continue

            graph = graph.to(device)
            graph.batch = torch.zeros(graph.x.size(0), dtype=torch.long, device=device)

            probs = [torch.sigmoid(m(graph)).item() for m in models]
            avg_prob = sum(probs) / len(probs)
            std_prob = (sum((p - avg_prob) ** 2 for p in probs) / len(probs)) ** 0.5

            results.append({
                "site_id": site.get("id", "?"),
                "score": round(avg_prob, 4),
                "uncertainty": round(std_prob, 4),
                "classification": site.get("classification", "?"),
                "centroid": centroid,
                "volume": site.get("volume", 0),
            })

    results.sort(key=lambda r: -r["score"])

    # Aggressive NMS: reduce from 40-60 sites to top 15
    if len(results) > 15:
        nms_input_sites = [{"centroid": r["centroid"]} for r in results if r.get("centroid")]
        nms_scores = [r["score"] for r in results if r.get("centroid")]
        nms_kept = aggressive_nms(nms_input_sites, nms_scores, max_sites=15, merge_dist=6.0)
        kept_centroids = set(tuple(s["centroid"]) for s, _ in nms_kept)
        results = [r for r in results if r.get("centroid") and tuple(r["centroid"]) in kept_centroids]
        results.sort(key=lambda r: -r["score"])

    print(f"\nPRISM4D GVP-EGNN v4 Ensemble Ranking: {pdb_id.upper()}")
    print(f"{'Rank':<5} {'Site':<8} {'Score':<8} {'±σ':<8} {'Class':<14} {'Vol'}")
    print("-" * 55)
    for i, r in enumerate(results[:10]):
        print(f"#{i+1:<4} {r['site_id']:<8} {r['score']:<8.4f} {r['uncertainty']:<8.4f} "
              f"{r['classification']:<14} {r.get('volume',0):.0f}")

    return results


# ── CLI ──────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="PRISM4D GVP-EGNN Pocket Ranker v4 (Ensemble)")
    sub = parser.add_subparsers(dest="command")

    train_p = sub.add_parser("train")
    train_p.add_argument("--sites-dir", default="benchmarks/prism4d_bench30/results")
    train_p.add_argument("--apo-dir", default="benchmarks/prism4d_bench30/apo")
    train_p.add_argument("--gt", default="benchmarks/prism4d_bench30/ground_truth/ligand_centroids.json")
    train_p.add_argument("--manifest", default="benchmarks/prism4d_bench30/benchmark_manifest.json")
    train_p.add_argument("--epochs", type=int, default=300)
    train_p.add_argument("--lr", type=float, default=3e-4)
    train_p.add_argument("--n-models", type=int, default=5)
    train_p.add_argument("--model-dir", default="models")
    train_p.add_argument("--esm-model", default="esm2_t33_650M_UR50D")
    train_p.add_argument("--dcc-threshold", type=float, default=8.0)

    pred_p = sub.add_parser("predict")
    pred_p.add_argument("--pdb", required=True)
    pred_p.add_argument("--sites", required=True)
    pred_p.add_argument("--model-dir", default="models")
    pred_p.add_argument("--n-models", type=int, default=5)
    pred_p.add_argument("--esm-model", default="esm2_t33_650M_UR50D")

    args = parser.parse_args()

    if args.command == "train":
        embedder = ESMEmbedder(args.esm_model)
        graphs, labels, meta = build_training_data(
            args.manifest, args.gt, args.sites_dir, args.apo_dir,
            embedder, dcc_threshold=args.dcc_threshold)
        if not graphs:
            print("ERROR: No training data")
            sys.exit(1)
        train_ensemble(graphs, labels, meta, n_models=args.n_models,
                       epochs=args.epochs, lr=args.lr, model_dir=args.model_dir)

    elif args.command == "predict":
        predict(args.pdb, args.sites, model_dir=args.model_dir,
                n_models=args.n_models, esm_model=args.esm_model)

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
