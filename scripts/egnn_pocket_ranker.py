#!/usr/bin/env python3
"""
PRISM4D EGNN Pocket Ranker
==========================
SE(3)-equivariant graph neural network for binding site ranking.

Takes PRISM-detected pockets (with lining residues + 3D coordinates),
builds a per-pocket residue graph, and scores each pocket for
druggability. The EGNN learns spatial patterns that distinguish
real binding sites from decoys — something a flat feature vector
(Vina score, volume) cannot capture.

Architecture:
    - Nodes: pocket lining residue CA atoms
    - Node features: residue type (20-dim one-hot), B-factor, relative position
    - Edges: k-nearest neighbors within 10A cutoff
    - Message passing: E(n)-equivariant layers (coordinate-aware)
    - Readout: global mean pool → MLP → druggability score

Usage:
    # Train
    python scripts/egnn_pocket_ranker.py train \
        --sites-dir benchmarks/prism4d_bench30/results \
        --apo-dir benchmarks/prism4d_bench30/apo \
        --gt benchmarks/prism4d_bench30/ground_truth/ligand_centroids.json \
        --manifest benchmarks/prism4d_bench30/benchmark_manifest.json \
        --epochs 200

    # Predict (rank pockets for a new protein)
    python scripts/egnn_pocket_ranker.py predict \
        --pdb protein.pdb \
        --sites protein.binding_sites.json \
        --model models/egnn_ranker.pt

References:
    Satorras et al. "E(n) Equivariant Graph Neural Networks" ICML 2021
"""

import argparse
import json
import os
import sys
import numpy as np
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import global_mean_pool

# ── Residue encoding ────────────────────────────────────────────────────

AA_TYPES = [
    "ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY",
    "HIS", "ILE", "LEU", "LYS", "MET", "PHE", "PRO", "SER",
    "THR", "TRP", "TYR", "VAL",
]
AA_TO_IDX = {aa: i for i, aa in enumerate(AA_TYPES)}
N_AA = len(AA_TYPES)


# ── EGNN Layer ───────────────────────────────────────────────────────────

class EGNNLayer(nn.Module):
    """E(n) Equivariant Graph Neural Network layer.

    Updates both node features h and coordinates x equivariantly.
    Messages depend on pairwise distances (invariant) and node features.
    Coordinate updates are weighted combinations of relative positions.
    """

    def __init__(self, hidden_dim, edge_dim=1):
        super().__init__()
        self.edge_mlp = nn.Sequential(
            nn.Linear(2 * hidden_dim + edge_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        )
        self.node_mlp = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.coord_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1, bias=False),
        )
        # Attention weight for aggregation
        self.att_mlp = nn.Sequential(
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )

    def forward(self, h, x, edge_index):
        """
        h: (N, hidden_dim) node features
        x: (N, 3) coordinates
        edge_index: (2, E) edge indices
        """
        row, col = edge_index  # row -> col messages

        # Pairwise distances (invariant)
        diff = x[row] - x[col]  # (E, 3)
        dist = torch.norm(diff, dim=-1, keepdim=True)  # (E, 1)

        # Edge messages
        edge_feat = torch.cat([h[row], h[col], dist], dim=-1)
        m_ij = self.edge_mlp(edge_feat)  # (E, hidden_dim)

        # Attention-weighted aggregation
        att = self.att_mlp(m_ij)  # (E, 1)
        m_ij_att = m_ij * att

        # Aggregate messages to nodes
        agg = torch.zeros_like(h)
        agg.index_add_(0, col, m_ij_att)

        # Update node features
        h_new = h + self.node_mlp(torch.cat([h, agg], dim=-1))

        # Update coordinates (equivariant)
        coord_weights = self.coord_mlp(m_ij)  # (E, 1)
        coord_delta = torch.zeros_like(x)
        # Weight relative positions by learned coefficients
        weighted_diff = diff * coord_weights  # (E, 3)
        coord_delta.index_add_(0, col, weighted_diff)
        x_new = x + coord_delta

        return h_new, x_new


# ── Full EGNN Model ──────────────────────────────────────────────────────

class EGNNPocketRanker(nn.Module):
    """EGNN-based pocket druggability scorer.

    Input: graph of pocket lining residues with 3D coordinates
    Output: scalar druggability probability
    """

    def __init__(self, node_feat_dim=N_AA + 4, hidden_dim=128, n_layers=4, dropout=0.1):
        super().__init__()
        self.node_encoder = nn.Sequential(
            nn.Linear(node_feat_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
        )
        self.layers = nn.ModuleList([
            EGNNLayer(hidden_dim) for _ in range(n_layers)
        ])
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, data):
        h = self.node_encoder(data.x)
        x = data.pos.clone()

        for layer in self.layers:
            h, x = layer(h, x, data.edge_index)

        # Global readout: mean pool over all residues in pocket
        h_pocket = global_mean_pool(h, data.batch)

        # Druggability score
        logit = self.classifier(h_pocket).squeeze(-1)
        return logit


# ── Data construction ────────────────────────────────────────────────────

def parse_pdb_ca_atoms(pdb_path):
    """Extract CA atom coordinates and residue info from PDB."""
    residues = []
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
            try:
                x = float(line[30:38])
                y = float(line[38:46])
                z = float(line[46:54])
                bfactor = float(line[60:66]) if len(line) > 66 else 0.0
            except ValueError:
                continue
            residues.append({
                "resname": resname, "chain": chain, "resseq": resseq,
                "x": x, "y": y, "z": z, "bfactor": bfactor,
            })
    return residues


def build_pocket_graph(all_residues, lining_residue_ids, pocket_centroid,
                       pocket_volume, k_neighbors=8, cutoff=10.0):
    """Build a PyG Data object for one pocket.

    Args:
        all_residues: list of dicts from parse_pdb_ca_atoms
        lining_residue_ids: list of residue identifiers from binding_sites.json
        pocket_centroid: [x, y, z]
        pocket_volume: float
        k_neighbors: number of nearest neighbors for edges
        cutoff: max edge distance in Angstrom

    Returns:
        torch_geometric.data.Data or None
    """
    # Build residue lookup by various ID formats
    res_lookup = {}
    for r in all_residues:
        # Try multiple ID formats that PRISM might use
        keys = [
            f"{r['chain']}_{r['resseq']}",
            f"{r['resseq']}",
            f"{r['chain']}{r['resseq']}",
            f"{r['resname']}{r['resseq']}",
        ]
        for k in keys:
            res_lookup[k] = r

    # Match lining residues
    matched = []
    for lr_id in lining_residue_ids:
        lr_str = str(lr_id).strip()
        # Try the ID as-is, then various transformations
        for attempt in [lr_str, lr_str.replace("_", ""), lr_str.split("_")[-1]]:
            if attempt in res_lookup:
                matched.append(res_lookup[attempt])
                break

    if len(matched) < 4:
        # Too few residues to form a meaningful graph — fall back to
        # spatial selection: grab all CA atoms within 12A of centroid
        cent = np.array(pocket_centroid)
        for r in all_residues:
            d = np.linalg.norm(np.array([r["x"], r["y"], r["z"]]) - cent)
            if d < 12.0:
                if r not in matched:
                    matched.append(r)

    if len(matched) < 4:
        return None

    # Build node features
    coords = []
    features = []
    for r in matched:
        # One-hot residue type
        aa_idx = AA_TO_IDX.get(r["resname"], -1)
        onehot = [0.0] * N_AA
        if aa_idx >= 0:
            onehot[aa_idx] = 1.0

        # Additional features: normalized b-factor, distance to centroid,
        # pocket volume (log-scaled), relative position flag
        cent = np.array(pocket_centroid)
        pos = np.array([r["x"], r["y"], r["z"]])
        dist_to_cent = float(np.linalg.norm(pos - cent))
        log_vol = np.log1p(pocket_volume) / 10.0  # normalize
        bfac_norm = r["bfactor"] / 100.0  # rough normalization

        features.append(onehot + [bfac_norm, dist_to_cent / 20.0, log_vol, 1.0])
        coords.append([r["x"], r["y"], r["z"]])

    coords = np.array(coords)
    features = np.array(features)

    # Build edges: k-nearest neighbors within cutoff
    n = len(coords)
    dists = np.linalg.norm(coords[:, None] - coords[None, :], axis=-1)
    edge_src, edge_dst = [], []
    for i in range(n):
        # Get k nearest neighbors
        nn_idx = np.argsort(dists[i])[1:k_neighbors + 1]
        for j in nn_idx:
            if dists[i, j] < cutoff:
                edge_src.append(i)
                edge_dst.append(j)

    if not edge_src:
        return None

    data = Data(
        x=torch.tensor(features, dtype=torch.float),
        pos=torch.tensor(coords, dtype=torch.float),
        edge_index=torch.tensor([edge_src, edge_dst], dtype=torch.long),
    )
    return data


# ── Dataset construction from benchmark ──────────────────────────────────

def build_training_dataset(manifest_path, gt_path, sites_dir, apo_dir, dcc_threshold=5.0):
    """Build list of (Data, label) from the benchmark.

    Each pocket becomes one graph. Label = 1 if pocket centroid is within
    dcc_threshold of the ground truth ligand centroid, else 0.
    """
    with open(manifest_path) as f:
        manifest = json.load(f)
    with open(gt_path) as f:
        gt = json.load(f)

    graphs = []
    labels = []
    meta = []  # for tracking

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
        all_residues = parse_pdb_ca_atoms(pdb_path)

        if not all_residues:
            continue

        with open(sites_path) as f:
            sites_data = json.load(f)

        for site in sites_data.get("sites", []):
            centroid = site.get("centroid")
            if not centroid:
                continue

            lining = site.get("lining_residues", [])
            volume = site.get("volume", 1000.0)
            site_id = site.get("id", "?")

            graph = build_pocket_graph(
                all_residues, lining, centroid, volume,
            )
            if graph is None:
                continue

            dcc = float(np.linalg.norm(np.array(centroid) - true_cent))
            label = 1.0 if dcc <= dcc_threshold else 0.0

            graphs.append(graph)
            labels.append(label)
            meta.append({"target": apo.upper(), "site_id": site_id, "dcc": dcc})

    # Attach labels
    for g, label in zip(graphs, labels):
        g.y = torch.tensor([label], dtype=torch.float)

    return graphs, labels, meta


# ── Training ─────────────────────────────────────────────────────────────

def train_model(graphs, labels, meta, epochs=200, lr=1e-3, model_out="models/egnn_ranker.pt"):
    """Train the EGNN pocket ranker."""
    os.makedirs(os.path.dirname(model_out), exist_ok=True)

    n_pos = sum(labels)
    n_neg = len(labels) - n_pos
    print(f"Training EGNN on {len(graphs)} pockets ({int(n_pos)} hits, {int(n_neg)} decoys)")

    if n_pos == 0:
        print("ERROR: No positive examples. Cannot train.")
        return

    # Class weight for imbalanced data
    pos_weight = torch.tensor([n_neg / max(n_pos, 1)], dtype=torch.float)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model = EGNNPocketRanker(
        node_feat_dim=N_AA + 4,
        hidden_dim=128,
        n_layers=4,
        dropout=0.15,
    ).to(device)

    pos_weight = pos_weight.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    loader = DataLoader(graphs, batch_size=32, shuffle=True)

    best_loss = float("inf")
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        n_batches = 0

        for batch in loader:
            batch = batch.to(device)
            optimizer.zero_grad()

            logit = model(batch)
            loss = F.binary_cross_entropy_with_logits(
                logit, batch.y.squeeze(-1), pos_weight=pos_weight
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        scheduler.step()
        avg_loss = total_loss / max(n_batches, 1)

        if (epoch + 1) % 20 == 0 or epoch == 0:
            # Quick eval
            model.eval()
            with torch.no_grad():
                all_preds = []
                all_labels = []
                for batch in loader:
                    batch = batch.to(device)
                    logit = model(batch)
                    probs = torch.sigmoid(logit)
                    all_preds.extend(probs.cpu().numpy())
                    all_labels.extend(batch.y.squeeze(-1).cpu().numpy())

            all_preds = np.array(all_preds)
            all_labels = np.array(all_labels)
            pred_binary = (all_preds > 0.5).astype(int)
            tp = ((pred_binary == 1) & (all_labels == 1)).sum()
            fp = ((pred_binary == 1) & (all_labels == 0)).sum()
            fn = ((pred_binary == 0) & (all_labels == 1)).sum()
            recall = tp / max(tp + fn, 1)
            precision = tp / max(tp + fp, 1)
            print(f"  Epoch {epoch+1:>3d}: loss={avg_loss:.4f} "
                  f"P={precision:.2f} R={recall:.2f} "
                  f"lr={scheduler.get_last_lr()[0]:.1e}")

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save({
                "model_state": model.state_dict(),
                "config": {
                    "node_feat_dim": N_AA + 4,
                    "hidden_dim": 128,
                    "n_layers": 4,
                },
            }, model_out)

    print(f"\nModel saved: {model_out}")

    # Print per-target ranking quality
    print(f"\n{'Target':<8} {'Site':<8} {'DCC':<8} {'EGNN_P':<8} {'Label'}")
    print("-" * 45)
    model.eval()
    target_pockets = {}
    with torch.no_grad():
        for i, g in enumerate(graphs):
            g_dev = g.to(device)
            # Need to add batch index for single graph
            g_dev.batch = torch.zeros(g_dev.x.size(0), dtype=torch.long, device=device)
            logit = model(g_dev)
            prob = torch.sigmoid(logit).item()
            m = meta[i]
            target_pockets.setdefault(m["target"], []).append(
                (m["site_id"], m["dcc"], prob, labels[i])
            )

    sr_at_1 = 0
    n_targets = 0
    for tgt in sorted(target_pockets):
        pockets = sorted(target_pockets[tgt], key=lambda x: -x[2])  # sort by EGNN prob desc
        top = pockets[0]
        is_correct = top[3] == 1.0
        if is_correct:
            sr_at_1 += 1
        n_targets += 1
        marker = " <<<" if is_correct else ""
        print(f"{tgt:<8} {top[0]:<8} {top[1]:<8.2f} {top[2]:<8.3f} {int(top[3])}{marker}")

    print(f"\nSR@1: {sr_at_1}/{n_targets} ({100*sr_at_1/max(n_targets,1):.0f}%)")


# ── Inference ────────────────────────────────────────────────────────────

def predict_pockets(pdb_path, sites_path, model_path):
    """Rank pockets for a new protein using trained EGNN."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    config = checkpoint["config"]

    model = EGNNPocketRanker(**config).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    all_residues = parse_pdb_ca_atoms(pdb_path)
    with open(sites_path) as f:
        sites_data = json.load(f)

    results = []
    with torch.no_grad():
        for site in sites_data.get("sites", []):
            centroid = site.get("centroid")
            if not centroid:
                continue

            lining = site.get("lining_residues", [])
            volume = site.get("volume", 1000.0)
            site_id = site.get("id", "?")
            classification = site.get("classification", "?")

            graph = build_pocket_graph(all_residues, lining, centroid, volume)
            if graph is None:
                results.append({
                    "site_id": site_id, "egnn_score": 0.0,
                    "classification": classification,
                })
                continue

            graph = graph.to(device)
            graph.batch = torch.zeros(graph.x.size(0), dtype=torch.long, device=device)

            logit = model(graph)
            prob = torch.sigmoid(logit).item()

            results.append({
                "site_id": site_id,
                "egnn_score": round(prob, 4),
                "classification": classification,
                "centroid": centroid,
                "volume": volume,
                "n_lining_residues": len(lining),
            })

    # Sort by EGNN score descending
    results.sort(key=lambda r: -r["egnn_score"])

    print(f"\nEGNN Pocket Ranking for {Path(pdb_path).stem.upper()}")
    print(f"{'Rank':<5} {'Site':<8} {'EGNN':<8} {'Class':<14} {'Volume':<8} {'Residues'}")
    print("-" * 55)
    for i, r in enumerate(results[:10]):
        print(f"#{i+1:<4} {r['site_id']:<8} {r['egnn_score']:<8.3f} "
              f"{r['classification']:<14} {r.get('volume',0):<8.0f} "
              f"{r.get('n_lining_residues',0)}")

    return results


# ── CLI ──────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="PRISM4D EGNN Pocket Ranker")
    sub = parser.add_subparsers(dest="command")

    # Train
    train_p = sub.add_parser("train", help="Train EGNN on benchmark data")
    train_p.add_argument("--sites-dir", default="benchmarks/prism4d_bench30/results")
    train_p.add_argument("--apo-dir", default="benchmarks/prism4d_bench30/apo")
    train_p.add_argument("--gt", default="benchmarks/prism4d_bench30/ground_truth/ligand_centroids.json")
    train_p.add_argument("--manifest", default="benchmarks/prism4d_bench30/benchmark_manifest.json")
    train_p.add_argument("--epochs", type=int, default=200)
    train_p.add_argument("--lr", type=float, default=1e-3)
    train_p.add_argument("--model-out", default="models/egnn_ranker.pt")
    train_p.add_argument("--dcc-threshold", type=float, default=5.0)

    # Predict
    pred_p = sub.add_parser("predict", help="Rank pockets for a new protein")
    pred_p.add_argument("--pdb", required=True)
    pred_p.add_argument("--sites", required=True)
    pred_p.add_argument("--model", default="models/egnn_ranker.pt")

    args = parser.parse_args()

    if args.command == "train":
        graphs, labels, meta = build_training_dataset(
            args.manifest, args.gt, args.sites_dir, args.apo_dir,
            dcc_threshold=args.dcc_threshold,
        )
        if not graphs:
            print("ERROR: No training data generated")
            sys.exit(1)
        train_model(graphs, labels, meta,
                    epochs=args.epochs, lr=args.lr, model_out=args.model_out)

    elif args.command == "predict":
        predict_pockets(args.pdb, args.sites, args.model)

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
