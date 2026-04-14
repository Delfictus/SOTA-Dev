#!/usr/bin/env python3
"""predict_v004.py — Zero-shot binding site prediction.

PDB in → sites out. No ESM-2. No engine.

Pipeline:
  1. Parse PDB → structural(25) + NMA(26) + perturbedNMA(5) = 56 dims
  2. Load SpikeBERT, run all-masked → 512-dim spike embeddings per residue
  3. Concatenate → 568 dims
  4. Run VN-EGNN v004 → atom logits + VN positions + confidence
  5. Cluster high-confidence VNs → ranked sites
  6. Output: JSON + PyMOL .pse

Usage:
  python3 scripts/training/predict_v004.py \\
      --pdb structure.pdb \\
      --spikebert-onnx /mnt/storage/spike-audit/spike-bert/spike_bert.onnx \\
      --vnegnn-onnx /mnt/storage/spike-audit/vnegnn-v004/vnegnn_v004.onnx \\
      --output-dir results/
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

warnings.filterwarnings("ignore", category=FutureWarning)


# ─────────────────────────────────────────────────────────────
# PDB parsing → structural features (25 + 26 + 5 = 56 dims)
# ─────────────────────────────────────────────────────────────

AA_3 = ["ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE",
        "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL"]
AA_IDX = {a: i for i, a in enumerate(AA_3)}

HYDRO = {"ALA": 1.8, "ARG": -4.5, "ASN": -3.5, "ASP": -3.5, "CYS": 2.5,
         "GLN": -3.5, "GLU": -3.5, "GLY": -0.4, "HIS": -3.2, "ILE": 4.5,
         "LEU": 3.8, "LYS": -3.9, "MET": 1.9, "PHE": 2.8, "PRO": -1.6,
         "SER": -0.8, "THR": -0.7, "TRP": -0.9, "TYR": -1.3, "VAL": 4.2}

DSSP_MAP = {"H": 0, "E": 1, "C": 2}


def parse_pdb(pdb_path: str) -> Tuple[np.ndarray, np.ndarray, List[str], List[int]]:
    """Parse PDB → CA coords, structural features (25d), resnames, resids.

    Structural features per residue (25d):
      [0:20]  amino acid one-hot
      [20:23] DSSP placeholder (H/E/C one-hot, default coil)
      [23]    relative SASA placeholder (0.5 default)
      [24]    hydrophobicity
    """
    coords = []
    features = []
    resnames = []
    resids = []
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
            resid = int(line[22:26].strip())
            key = (chain, resid)
            if key in seen:
                continue
            seen.add(key)

            x = float(line[30:38])
            y = float(line[38:46])
            z = float(line[46:54])
            coords.append([x, y, z])

            feat = np.zeros(25, dtype=np.float32)
            if resname in AA_IDX:
                feat[AA_IDX[resname]] = 1.0
            feat[22] = 1.0  # default coil DSSP
            feat[23] = 0.5  # default SASA
            feat[24] = HYDRO.get(resname, 0.0) / 4.5
            features.append(feat)
            resnames.append(resname)
            resids.append(resid)

    return (np.array(coords, dtype=np.float32),
            np.array(features, dtype=np.float32),
            resnames, resids)


def compute_nma(coords: np.ndarray, n_modes: int = 13) -> Tuple[np.ndarray, np.ndarray]:
    """Approximate NMA from CA contact map (26d NMA + 5d perturbed NMA).

    Uses Gaussian Network Model (GNM) for fast eigendecomposition.
    """
    N = len(coords)
    nma_feats = np.zeros((N, 26), dtype=np.float32)
    perturb_feats = np.zeros((N, 5), dtype=np.float32)

    if N < 10:
        return nma_feats, perturb_feats

    # GNM Kirchhoff matrix
    dists = np.sqrt(((coords[:, None] - coords[None, :]) ** 2).sum(axis=2))
    contact = (dists < 7.0).astype(np.float32)
    np.fill_diagonal(contact, 0)
    K = -contact
    np.fill_diagonal(K, contact.sum(axis=1))

    try:
        eigenvalues, eigenvectors = np.linalg.eigh(K)
        # Skip first eigenvalue (zero mode)
        n_use = min(n_modes, N - 1)
        modes = eigenvectors[:, 1:1 + n_use]
        freqs = eigenvalues[1:1 + n_use]
        freqs = np.maximum(freqs, 1e-8)

        # NMA features: first 13 modes + 13 squared fluctuations
        for i in range(min(n_use, 13)):
            nma_feats[:, i] = modes[:, i]
            nma_feats[:, 13 + i] = modes[:, i] ** 2 / freqs[i]

        # Perturbed NMA: variance of lowest 5 modes under perturbation
        for i in range(min(5, n_use)):
            perturb_feats[:, i] = np.abs(modes[:, i]) * (1.0 / freqs[i])
    except np.linalg.LinAlgError:
        pass

    return nma_feats, perturb_feats


# ─────────────────────────────────────────────────────────────
# SpikeBERT inference (all-masked → 512-dim embeddings)
# ─────────────────────────────────────────────────────────────

def run_spikebert(structural_feats: np.ndarray, model_path: str) -> np.ndarray:
    """Run SpikeBERT v002 in all-masked mode → 512-dim per residue.

    Uses PyTorch .pt checkpoint directly (ONNX export has broken
    dynamic shapes in TransformerEncoder attention reshapes).
    """
    import torch
    import torch.nn as nn

    MASK_TOKEN = 2048
    PAD_TOKEN = 2049
    VOCAB = 2050
    HIDDEN = 512
    STRUCT = 56
    NLAYERS = 6
    NHEADS = 8

    class _SpikeBERT(nn.Module):
        def __init__(self):
            super().__init__()
            self.token_embed = nn.Embedding(VOCAB, HIDDEN, padding_idx=PAD_TOKEN)
            self.struct_proj = nn.Linear(STRUCT, HIDDEN)
            self.pos_embed = nn.Embedding(2048, HIDDEN)
            self.input_norm = nn.LayerNorm(HIDDEN)
            self.input_dropout = nn.Dropout(0.15)
            layer = nn.TransformerEncoderLayer(
                d_model=HIDDEN, nhead=NHEADS, dim_feedforward=2048,
                dropout=0.15, activation="gelu", batch_first=True, norm_first=True)
            self.encoder = nn.TransformerEncoder(layer, num_layers=NLAYERS)
            self.mlm_norm = nn.LayerNorm(HIDDEN)
            self.mlm_head = nn.Linear(HIDDEN, 2048)
            self.distill_head = nn.Sequential(
                nn.LayerNorm(HIDDEN), nn.Linear(HIDDEN, 128),
                nn.GELU(), nn.Linear(128, 1))
            self.physics_head = nn.Sequential(
                nn.LayerNorm(HIDDEN), nn.Linear(HIDDEN, 256),
                nn.GELU(), nn.Linear(256, 216))

    model = _SpikeBERT()
    state = torch.load(model_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    model.eval()

    N = structural_feats.shape[0]
    ids = torch.full((1, N), MASK_TOKEN, dtype=torch.long)
    sf = torch.tensor(structural_feats, dtype=torch.float32).unsqueeze(0)
    pm = torch.zeros(1, N, dtype=torch.bool)
    pos = torch.arange(N).unsqueeze(0)

    with torch.no_grad():
        x = model.token_embed(ids) + model.struct_proj(sf) + model.pos_embed(pos)
        x = model.input_dropout(model.input_norm(x))
        h = model.encoder(x, src_key_padding_mask=pm)

    return h.squeeze(0).numpy().astype(np.float32)  # [N, 512]


# ─────────────────────────────────────────────────────────────
# VN-EGNN inference
# ─────────────────────────────────────────────────────────────

def run_vnegnn(atom_features: np.ndarray, atom_coords: np.ndarray,
               onnx_path: str, n_vns: int = 16, edge_cutoff: float = 8.0
               ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run VN-EGNN → atom logits, VN coords, VN confidence."""
    import onnxruntime as ort
    import torch
    sys.path.insert(0, str(Path(__file__).resolve().parent / "vn_egnn"))
    from model import build_edge_index, init_virtual_nodes

    coords_t = torch.tensor(atom_coords, dtype=torch.float32)
    edge_index = build_edge_index(coords_t, n_vns, cutoff=edge_cutoff).numpy()
    vn_init = init_virtual_nodes(coords_t, n_vns,
                                 rng=torch.Generator().manual_seed(42)).numpy()

    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    input_names = [inp.name for inp in sess.get_inputs()]
    output_names = [out.name for out in sess.get_outputs()]

    feeds = {}
    for inp in sess.get_inputs():
        if inp.name == "atom_features":
            feeds[inp.name] = atom_features.astype(np.float32)
        elif inp.name == "atom_coords":
            feeds[inp.name] = atom_coords.astype(np.float32)
        elif inp.name == "edge_index":
            feeds[inp.name] = edge_index.astype(np.int64)
        elif inp.name == "vn_init_coords":
            feeds[inp.name] = vn_init.astype(np.float32)
        elif inp.name == "edge_feat":
            E = edge_index.shape[1]
            feeds[inp.name] = np.zeros((E, 1), dtype=np.float32)

    results = sess.run(output_names, feeds)
    atom_logits = results[0]
    vn_coords = results[1]
    vn_confidence = results[2]
    return atom_logits, vn_coords, vn_confidence


# ─────────────────────────────────────────────────────────────
# Site clustering + output
# ─────────────────────────────────────────────────────────────

def cluster_vns(vn_coords: np.ndarray, vn_confidence: np.ndarray,
                min_confidence: float = 0.3, merge_radius: float = 4.0
                ) -> List[Dict]:
    """Cluster high-confidence VNs into ranked binding sites."""
    probs = 1.0 / (1.0 + np.exp(-vn_confidence.squeeze()))
    order = np.argsort(-probs)

    sites = []
    used = set()

    for idx in order:
        if probs[idx] < min_confidence:
            continue
        if idx in used:
            continue

        center = vn_coords[idx]
        members = [idx]
        for j in order:
            if j in used or j == idx:
                continue
            if np.linalg.norm(vn_coords[j] - center) < merge_radius:
                members.append(j)
                used.add(j)
        used.add(idx)

        site_center = vn_coords[members].mean(axis=0)
        site_conf = float(probs[members].max())
        sites.append({
            "rank": len(sites) + 1,
            "centroid": site_center.tolist(),
            "confidence": round(site_conf, 4),
            "n_vns": len(members),
            "vn_indices": [int(m) for m in members],
        })

    return sites


def write_pymol_script(sites: List[Dict], pdb_path: str, out_path: str):
    """Write PyMOL .pml script to visualize predicted sites."""
    lines = [
        f'load {pdb_path}, protein',
        'hide everything',
        'show cartoon, protein',
        'color white, protein',
        '',
    ]
    colors = ["red", "blue", "green", "yellow", "magenta", "cyan",
              "orange", "pink", "purple", "brown"]

    for site in sites[:10]:
        r = site["rank"]
        x, y, z = site["centroid"]
        c = colors[(r - 1) % len(colors)]
        lines.append(f'pseudoatom site{r}, pos=[{x:.2f}, {y:.2f}, {z:.2f}]')
        lines.append(f'show spheres, site{r}')
        lines.append(f'set sphere_scale, 2.0, site{r}')
        lines.append(f'color {c}, site{r}')
        lines.append(f'label site{r}, "Site {r} ({site["confidence"]:.0%})"')
        lines.append('')

    lines.append('zoom')
    lines.append(f'# {len(sites)} predicted binding sites')
    Path(out_path).write_text('\n'.join(lines))


# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Zero-shot binding site prediction (no ESM-2, no engine)")
    parser.add_argument("--pdb", required=True, help="Input PDB file")
    parser.add_argument("--spikebert-pt", type=Path,
                        default=Path("/mnt/storage/spike-audit/spike-bert-v002/best_model.pt"))
    parser.add_argument("--vnegnn-onnx", type=Path,
                        default=Path("/mnt/storage/spike-audit/vnegnn-v004/vnegnn_v004.onnx"))
    parser.add_argument("--tokenizer", type=Path,
                        default=Path("/mnt/storage/spike-audit/spike-bert/tokenizer_centroids.npz"))
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    parser.add_argument("--min-confidence", type=float, default=0.3)
    parser.add_argument("--n-vns", type=int, default=16)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    pdb_name = Path(args.pdb).stem

    print(f"{'='*60}")
    print(f"  PRISM4D Zero-Shot Prediction v004")
    print(f"  PDB: {args.pdb}")
    print(f"  No ESM-2. No engine. SpikeBERT + VN-EGNN.")
    print(f"{'='*60}")

    # Step 1: Parse PDB → structural features
    print("\n[1/4] Parsing PDB → structural + NMA features...")
    coords, structural, resnames, resids = parse_pdb(args.pdb)
    nma, perturbed_nma = compute_nma(coords)
    struct_feats = np.concatenate([structural, nma, perturbed_nma], axis=1)
    print(f"  {len(coords)} residues, structural: {struct_feats.shape}")

    # Step 2: SpikeBERT → 512-dim spike embeddings
    print("\n[2/4] SpikeBERT v002 (all-masked, PyTorch) → 512-dim spike embeddings...")
    spike_emb = run_spikebert(struct_feats, str(args.spikebert_pt))
    print(f"  SpikeBERT output: {spike_emb.shape}")

    # Step 3: Concatenate → 568 dims
    atom_features = np.concatenate([struct_feats, spike_emb], axis=1)
    print(f"\n[3/4] Feature vector: {atom_features.shape[1]}d "
          f"(struct={struct_feats.shape[1]} + spike={spike_emb.shape[1]})")

    # Step 4: VN-EGNN → sites
    print("\n[4/4] VN-EGNN v004 → binding site prediction...")
    atom_logits, vn_coords, vn_confidence = run_vnegnn(
        atom_features, coords, str(args.vnegnn_onnx), n_vns=args.n_vns)

    atom_probs = 1.0 / (1.0 + np.exp(-atom_logits.squeeze()))
    sites = cluster_vns(vn_coords, vn_confidence,
                        min_confidence=args.min_confidence)

    # Output
    print(f"\n  Detected {len(sites)} binding sites:")
    for site in sites[:10]:
        c = site["centroid"]
        print(f"    Site {site['rank']}: [{c[0]:.1f}, {c[1]:.1f}, {c[2]:.1f}]  "
              f"confidence={site['confidence']:.1%}  VNs={site['n_vns']}")

    # Save JSON
    result = {
        "pdb": str(args.pdb),
        "n_residues": len(coords),
        "n_sites": len(sites),
        "sites": sites,
        "model": "vnegnn_v004",
        "features": "structural(25) + NMA(26) + perturbedNMA(5) + SpikeBERT(512)",
        "esm2_used": False,
        "engine_used": False,
    }
    json_path = args.output_dir / f"{pdb_name}_v004_sites.json"
    json_path.write_text(json.dumps(result, indent=2))
    print(f"\n  JSON: {json_path}")

    # PyMOL script
    pml_path = args.output_dir / f"{pdb_name}_v004_sites.pml"
    write_pymol_script(sites, str(args.pdb), str(pml_path))
    print(f"  PyMOL: {pml_path}")

    # Top binding residues
    top_residues = np.argsort(-atom_probs)[:10]
    print(f"\n  Top 10 binding residues:")
    for r in top_residues:
        print(f"    {resnames[r]}{resids[r]}: prob={atom_probs[r]:.4f}")

    print(f"\n  Pipeline: PDB → structural(56d) → SpikeBERT(512d) → VN-EGNN → {len(sites)} sites")
    print(f"  Zero ESM-2. Zero engine. Pure spike physics.")


if __name__ == "__main__":
    main()
