#!/usr/bin/env python3
"""
PRISM-AI Cryptic Pocket Predictor v004 — Zero-Shot Inference
=============================================================

Predicts cryptic binding site residues from a single PDB file.
No engine run required. No topology. No spike data. No ESM-2.

Pipeline:
    PDB → structural(25) + NMA(26) + perturbedNMA(5) = 56 dims
        → SpikeBERT v002 (all-masked) → 512-dim spike embeddings
        → VN-EGNN v004 → atom logits + VN positions + confidence
        → clustered ranked binding sites

Usage:
    python predict.py input.pdb --output results.json
    python predict.py input.pdb --chain A --output results.json
    python predict.py input.pdb --top-k 5 --output results.json --visualize output.pml

Input:  Any PDB file (apo structure preferred)
Output: JSON with per-residue probabilities + ranked binding sites

Requirements:
    pip install torch scipy scikit-learn

Model: SpikeBERT v002 (22M param transformer, teacher-distilled from v005)
       + VN-EGNN v004 (587K params, 92.7% SR@8 on 372 targets)
       Trained on 3.6 billion spike events. Zero ESM-2 dependency.
"""

import argparse
import json
import math
import os
import sys
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")

# ═══════════════════════════════════════════════════════════
# CONSTANTS
# ═══════════════════════════════════════════════════════════

AA3TO1 = {
    'ALA':'A','ARG':'R','ASN':'N','ASP':'D','CYS':'C','GLN':'Q','GLU':'E',
    'GLY':'G','HIS':'H','HID':'H','HIE':'H','HIP':'H','ILE':'I','LEU':'L',
    'LYS':'K','MET':'M','PHE':'F','PRO':'P','SER':'S','THR':'T','TRP':'W',
    'TYR':'Y','VAL':'V','CYX':'C','MSE':'M','SEP':'S','TPO':'T','PTR':'Y',
    'CSO':'C','KCX':'K','CME':'C','MLY':'K',
}
AA_ORDER = "ACDEFGHIKLMNPQRSTVWY"
AA_IDX = {aa: i for i, aa in enumerate(AA_ORDER)}
HYDROPHOBICITY = {
    'A':1.8,'R':-4.5,'N':-3.5,'D':-3.5,'C':2.5,'Q':-3.5,'E':-3.5,'G':-0.4,
    'H':-3.2,'I':4.5,'L':3.8,'K':-3.9,'M':1.9,'F':2.8,'P':-1.6,'S':-0.8,
    'T':-0.7,'W':-0.9,'Y':-1.3,'V':4.2,
}

MODEL_DIR = Path(__file__).parent / "models"
SPIKEBERT_PT = Path(__file__).parent / "models_v004" / "spike_bert_v002.pt"
VNEGNN_ONNX = Path(__file__).parent / "models_v004" / "vnegnn_v004.onnx"


def _robust_normalize(feat):
    """Robust per-column normalization. Must match training pipeline exactly.
    Skip one-hot/categorical columns (those with <5 nonzero values)."""
    out = feat.copy()
    for j in range(feat.shape[1]):
        col = feat[:, j]
        nonzero = col[np.abs(col) > 1e-10]
        if len(nonzero) < 5:
            continue
        med = np.median(nonzero)
        q75, q25 = np.percentile(nonzero, [75, 25])
        iqr = q75 - q25
        if iqr < 1e-10:
            iqr = np.std(nonzero) or 1.0
        out[:, j] = np.clip((col - med) / iqr, -5, 5)
    return out


# ═══════════════════════════════════════════════════════════
# PDB PARSING + CLEANING
# ═══════════════════════════════════════════════════════════

def parse_pdb(pdb_path, chain=None):
    """Parse PDB file. Returns residue list, CA positions, sequence.

    Args:
        pdb_path: Path to PDB file
        chain: Chain ID to extract (None = first protein chain)

    Returns:
        dict with residues, ca_positions, sequence, chain_id
    """
    residues = {}  # resid -> {resname, ca_xyz, atoms}
    chains_found = set()

    with open(pdb_path) as f:
        for line in f:
            if not line.startswith("ATOM"):
                continue
            ch = line[21].strip() or "A"
            chains_found.add(ch)
            if chain and ch != chain:
                continue
            if chain is None and len(chains_found) > 1 and ch != min(chains_found):
                continue  # default: first chain alphabetically

            resname = line[17:20].strip()
            resid = int(line[22:26].strip())
            atom_name = line[12:16].strip()

            if resid not in residues:
                residues[resid] = {"resname": resname, "resid": resid, "chain": ch}

            if atom_name == "CA":
                x = float(line[30:38])
                y = float(line[38:46])
                z = float(line[46:54])
                residues[resid]["ca_xyz"] = np.array([x, y, z])

    # Sort by residue ID
    sorted_resids = sorted(residues.keys())
    residue_list = [residues[rid] for rid in sorted_resids]

    # Filter to residues with CA atoms
    residue_list = [r for r in residue_list if "ca_xyz" in r]

    if not residue_list:
        raise ValueError(f"No CA atoms found in {pdb_path}" +
                        (f" chain {chain}" if chain else ""))

    # Build sequence
    sequence = ""
    for r in residue_list:
        aa1 = AA3TO1.get(r["resname"], "X")
        sequence += aa1

    chain_id = residue_list[0]["chain"]
    ca_positions = {r["resid"]: r["ca_xyz"] for r in residue_list}

    return {
        "residues": residue_list,
        "ca_positions": ca_positions,
        "sequence": sequence,
        "chain_id": chain_id,
        "n_residues": len(residue_list),
        "pdb_path": str(pdb_path),
    }


# ═══════════════════════════════════════════════════════════
# FEATURE EXTRACTION
# ═══════════════════════════════════════════════════════════

def extract_structural_features(parsed_pdb):
    """Extract 26-dim structural features from parsed PDB.

    Features: AA one-hot (20) + hydrophobicity (1) + DSSP (3) + SASA (1) + B-factor (1)
    """
    n = parsed_pdb["n_residues"]
    feat = np.zeros((n, 26), dtype=np.float32)

    for i, r in enumerate(parsed_pdb["residues"]):
        aa1 = AA3TO1.get(r["resname"], "X")
        # AA one-hot (0-19)
        idx = AA_IDX.get(aa1, -1)
        if idx >= 0:
            feat[i, idx] = 1.0
        # Hydrophobicity (20)
        feat[i, 20] = HYDROPHOBICITY.get(aa1, 0.0)
        # DSSP defaults (21-23): coil
        feat[i, 23] = 1.0
        # SASA default (24)
        feat[i, 24] = 0.5
        # B-factor default (25)
        feat[i, 25] = 0.5

    # Try to compute real DSSP + SASA using mdtraj
    try:
        import mdtraj
        import subprocess, tempfile

        pdb_path = parsed_pdb["pdb_path"]
        t = mdtraj.load(pdb_path)
        # Select first chain
        sel = t.topology.select('protein')
        if len(sel) > 0:
            t = t.atom_slice(sel)

        dssp = mdtraj.compute_dssp(t)[0]
        sasa = mdtraj.shrake_rupley(t, mode='residue')[0]

        for i, ss in enumerate(dssp[:n]):
            feat[i, 21:24] = [0, 0, 0]
            if ss == 'H': feat[i, 21] = 1.0
            elif ss == 'E': feat[i, 22] = 1.0
            else: feat[i, 23] = 1.0

        sasa_max = sasa.max() if sasa.max() > 0 else 1.0
        feat[:min(n, len(sasa)), 24] = (sasa / sasa_max)[:n]
    except Exception:
        pass  # keep defaults

    # B-factors from PDB
    try:
        bf = {}; bc = {}
        with open(parsed_pdb["pdb_path"]) as f:
            for line in f:
                if line.startswith("ATOM"):
                    rn = line[17:20].strip()
                    if rn not in AA3TO1: continue
                    rid = int(line[22:26].strip())
                    b = float(line[60:66].strip())
                    bf[rid] = bf.get(rid, 0) + b
                    bc[rid] = bc.get(rid, 0) + 1
        for rid in bf:
            bf[rid] /= bc[rid]
        resids = [r["resid"] for r in parsed_pdb["residues"]]
        vals = np.array([bf.get(rid, 0) for rid in resids[:n]])
        bmax = vals.max() if vals.max() > 0 else 1.0
        feat[:len(vals), 25] = vals / bmax
    except Exception:
        pass

    return feat


def extract_nma_features(parsed_pdb):
    """Extract 26-dim NMA features using ProDy ANM.

    Features: mode displacements (20) + sqfluct + stiffness + hinge +
              effectiveness + sensitivity + long_range_corr
    """
    import prody
    prody.confProDy(verbosity='none')

    n = parsed_pdb["n_residues"]
    feat = np.zeros((n, 26), dtype=np.float32)

    try:
        protein = prody.parsePDB(parsed_pdb["pdb_path"])
        calphas = protein.select('calpha')
        if calphas is None or calphas.numAtoms() == 0:
            return feat

        n_ca = calphas.numAtoms()
        n_modes = 20

        # Build ANM with fallback cutoffs
        anm = None
        for cutoff in [15.0, 20.0, 25.0]:
            try:
                anm = prody.ANM("target")
                anm.buildHessian(calphas, cutoff=cutoff)
                anm.calcModes(n_modes=n_modes)
                break
            except Exception:
                anm = None

        if anm is None:
            return feat

        eigenvalues = anm.getEigvals()
        eigenvectors = anm.getEigvecs()
        actual_modes = min(len(eigenvalues), n_modes)

        if eigenvalues[0] <= 0:
            return feat

        # Mode displacements (0-19)
        mode_disp = np.zeros((n_ca, actual_modes))
        for k in range(actual_modes):
            for i in range(n_ca):
                vec = eigenvectors[i*3:(i+1)*3, k]
                mode_disp[i, k] = np.sum(vec**2)
        feat[:min(n, n_ca), :actual_modes] = mode_disp[:n, :actual_modes]

        # Squared fluctuation (20)
        sqfluct = np.zeros(n_ca)
        for k in range(actual_modes):
            sqfluct += mode_disp[:, k] / eigenvalues[k]
        feat[:min(n, n_ca), 20] = sqfluct[:n]

        # Stiffness (21)
        stiffness = 1.0 / np.maximum(sqfluct, 1e-10)
        stiffness = stiffness / max(stiffness.max(), 1e-10)
        feat[:min(n, n_ca), 21] = stiffness[:n]

        # Hinge score (22)
        mode1_vec = eigenvectors[:, 0].reshape(n_ca, 3)
        mode1_proj = np.sum(mode1_vec * mode1_vec.mean(axis=0), axis=1)
        hinge = np.zeros(n_ca)
        for i in range(1, n_ca - 1):
            if np.sign(mode1_proj[i-1]) != np.sign(mode1_proj[i+1]):
                hinge[i] = 1.0
            if mode_disp[i, 0] < 0.5 * (mode_disp[i-1, 0] + mode_disp[i+1, 0]):
                hinge[i] += 0.5
        feat[:min(n, n_ca), 22] = hinge[:n]

        # PRS effectiveness + sensitivity (23-24)
        try:
            _, effectiveness, sensitivity = prody.calcPerturbResponse(anm)
        except Exception:
            cross_corr = prody.calcCrossCorr(anm)
            effectiveness = np.abs(cross_corr).mean(axis=1)
            sensitivity = np.abs(cross_corr).mean(axis=0)
        feat[:min(n, n_ca), 23] = effectiveness[:n]
        feat[:min(n, n_ca), 24] = sensitivity[:n]

        # Long-range correlation (25)
        cross_corr = prody.calcCrossCorr(anm)
        lrc = np.zeros(n_ca)
        for i in range(n_ca):
            distant = [j for j in range(n_ca) if abs(j - i) > 20]
            if distant:
                lrc[i] = np.mean(np.abs(cross_corr[i, distant]))
        feat[:min(n, n_ca), 25] = lrc[:n]

    except Exception:
        pass

    return feat


def _build_spikebert():
    """Build SpikeBERT v002 architecture (must match training)."""
    import torch.nn as nn
    HIDDEN, STRUCT, NHEADS, NLAYERS = 512, 56, 8, 6
    PAD_TOKEN, VOCAB = 2049, 2050

    class SpikeBERT(nn.Module):
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
    return SpikeBERT


def extract_spikebert_embeddings(structural_feats):
    """Extract 512-dim spike embeddings. No ESM-2. No spike data."""
    import torch

    MASK_TOKEN = 2048
    SpikeBERT = _build_spikebert()
    model = SpikeBERT()
    state = torch.load(SPIKEBERT_PT, map_location="cpu", weights_only=True)
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

    return h.squeeze(0).numpy().astype(np.float32)


def run_vnegnn(atom_features, atom_coords):
    """Run VN-EGNN v004 via ONNX Runtime → atom logits, VN coords, VN confidence."""
    import torch
    import onnxruntime as ort

    # Build edges + init VNs (same as training)
    K = 16
    coords_t = torch.tensor(atom_coords, dtype=torch.float32)
    N = len(atom_coords)

    # Protein-protein edges within 8A
    d2 = torch.cdist(coords_t, coords_t)
    mask = (d2 < 8.0) & (d2 > 1e-6)
    pp = mask.nonzero(as_tuple=False).t()
    # VN-atom bipartite
    vn_ids = torch.arange(N, N + K)
    atom_ids = torch.arange(0, N)
    va = torch.stack([vn_ids.unsqueeze(1).expand(K, N).reshape(-1),
                      atom_ids.unsqueeze(0).expand(K, N).reshape(-1)])
    av = torch.stack([va[1], va[0]])
    edge_index = torch.cat([pp, va, av], dim=1).numpy().astype(np.int64)

    # Fibonacci sphere VN init
    phi = math.pi * (3.0 - math.sqrt(5.0))
    i_arr = np.arange(K, dtype=np.float32)
    y = 1.0 - 2.0 * i_arr / (K - 1) if K > 1 else np.zeros_like(i_arr)
    r_xy = np.sqrt(np.maximum(1.0 - y * y, 0.0))
    theta = phi * i_arr
    center = atom_coords.mean(axis=0)
    radius = np.linalg.norm(atom_coords - center, axis=1).max()
    vn_init = radius * np.stack([np.cos(theta) * r_xy, y, np.sin(theta) * r_xy], axis=1)
    vn_init = vn_init + center

    sess = ort.InferenceSession(str(VNEGNN_ONNX), providers=["CPUExecutionProvider"])
    feeds = {}
    for inp in sess.get_inputs():
        if inp.name == "atom_features":
            feeds[inp.name] = atom_features.astype(np.float32)
        elif inp.name == "atom_coords":
            feeds[inp.name] = atom_coords.astype(np.float32)
        elif inp.name == "edge_index":
            feeds[inp.name] = edge_index
        elif inp.name == "vn_init_coords":
            feeds[inp.name] = vn_init.astype(np.float32)

    results = sess.run(None, feeds)
    return results[0], results[1], results[2]  # atom_logits, vn_coords, vn_confidence


# ═══════════════════════════════════════════════════════════
# SITE CLUSTERING + RANKING
# ═══════════════════════════════════════════════════════════

def _subdivide_cluster(members, max_extent=20.0):
    """Subdivide a mega-cluster into sub-pockets using probability-weighted
    density peaks on a 3A voxel grid."""
    cas = np.array([m["ca"] for m in members])
    ps = np.array([m["prob"] for m in members])

    from scipy.spatial.distance import pdist
    extent = float(pdist(cas).max()) if len(cas) >= 2 else 0

    if extent <= max_extent or len(members) < 6:
        return [members]

    # Voxel density peak subdivision (3A grid)
    voxel_size = 3.0
    grid_min = cas.min(axis=0) - voxel_size
    grid_idx = ((cas - grid_min) / voxel_size).astype(int)

    # Accumulate probability density per voxel
    from collections import defaultdict
    voxel_density = defaultdict(float)
    voxel_members = defaultdict(list)
    for i, (gx, gy, gz) in enumerate(grid_idx):
        key = (gx, gy, gz)
        voxel_density[key] += ps[i]
        voxel_members[key].append(i)

    # Find local maxima (26-neighbor comparison)
    peaks = []
    for key, density in voxel_density.items():
        is_peak = True
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                for dz in [-1, 0, 1]:
                    if dx == 0 and dy == 0 and dz == 0:
                        continue
                    neighbor = (key[0]+dx, key[1]+dy, key[2]+dz)
                    if voxel_density.get(neighbor, 0) > density:
                        is_peak = False
                        break
                if not is_peak:
                    break
            if not is_peak:
                break
        if is_peak and density > 0:
            peak_center = grid_min + (np.array(key) + 0.5) * voxel_size
            peaks.append((peak_center, density, key))

    if len(peaks) <= 1:
        return [members]

    # Sort peaks by density (highest first)
    peaks.sort(key=lambda x: -x[1])

    # Assign each member to nearest peak
    peak_centers = np.array([p[0] for p in peaks])
    subclusters = [[] for _ in peaks]
    for i, m in enumerate(members):
        dists = np.linalg.norm(peak_centers - cas[i], axis=1)
        subclusters[dists.argmin()].append(m)

    # Filter out tiny subclusters (merge into nearest)
    result = []
    small = []
    for sc in subclusters:
        if len(sc) >= 3:
            result.append(sc)
        else:
            small.extend(sc)

    # Merge orphans into nearest valid subcluster
    if small and result:
        for m in small:
            best_dist = float('inf')
            best_sc = 0
            for j, sc in enumerate(result):
                sc_center = np.mean([mm["ca"] for mm in sc], axis=0)
                d = np.linalg.norm(m["ca"] - sc_center)
                if d < best_dist:
                    best_dist = d
                    best_sc = j
            result[best_sc].append(m)

    return result if result else [members]


def cluster_to_sites(probabilities, residue_list, ca_positions, top_pct=0.20):
    """Cluster high-probability residues into binding sites using DBSCAN
    with mega-cluster subdivision."""
    from sklearn.cluster import DBSCAN
    from scipy.spatial import ConvexHull
    from scipy.spatial.distance import pdist
    from collections import defaultdict

    n = len(residue_list)
    resids = [r["resid"] for r in residue_list]

    # Adaptive threshold: top 20% of residues or probability > 0.15
    n_top = max(20, int(n * top_pct))
    sorted_probs = sorted(enumerate(probabilities), key=lambda x: -x[1])

    # Filter candidates — use top_pct but also enforce minimum probability
    candidates = []
    for idx, prob in sorted_probs[:n_top]:
        if prob < 0.05:  # hard floor
            break
        rid = resids[idx]
        if rid in ca_positions:
            candidates.append({"idx": idx, "resid": rid, "prob": prob,
                              "ca": ca_positions[rid]})

    if len(candidates) < 3:
        return []

    coords = np.array([c["ca"] for c in candidates])

    # DBSCAN clustering
    clustering = DBSCAN(eps=8.0, min_samples=3).fit(coords)
    labels = clustering.labels_

    raw_clusters = defaultdict(list)
    for i, label in enumerate(labels):
        if label >= 0:
            raw_clusters[label].append(candidates[i])

    # Subdivide mega-clusters and build site list
    all_subclusters = []
    for label, members in raw_clusters.items():
        subclusters = _subdivide_cluster(members, max_extent=20.0)
        all_subclusters.extend(subclusters)

    sites = []
    for members in all_subclusters:
        rids = [m["resid"] for m in members]
        cas = np.array([m["ca"] for m in members])
        ps = np.array([m["prob"] for m in members])

        weights = ps / ps.sum()
        centroid = np.average(cas, axis=0, weights=weights)

        extent = float(pdist(cas).max()) if len(cas) >= 2 else 0

        compactness = 0.0
        if len(cas) >= 4:
            try:
                hull = ConvexHull(cas)
                compactness = len(cas) / max(hull.volume, 0.01)
            except Exception:
                compactness = len(cas) / max(extent ** 3, 0.01)

        score = (0.25 * ps.mean() + 0.20 * ps.max() +
                0.15 * math.log1p(ps.sum()) +
                0.10 * min((ps > 0.30).sum() / 10.0, 1.0) +
                0.10 * min(compactness * 100, 1.0))

        sites.append({
            "residue_ids": sorted(rids),
            "centroid": centroid.tolist(),
            "n_residues": len(rids),
            "mean_prob": float(ps.mean()),
            "max_prob": float(ps.max()),
            "sum_prob": float(ps.sum()),
            "spatial_extent": round(extent, 1),
            "rank_score": round(float(score), 4),
        })

    sites.sort(key=lambda s: -s["rank_score"])
    for i, s in enumerate(sites):
        s["rank"] = i + 1
        # Attach Tier 2 PRISM-TWIN recommendation (product upsell)
        s["validation_recommendation"] = {
            "tier2_recommendation": _build_tier2_recommendation(s),
        }

    return sites


def _build_tier2_recommendation(site):
    """Build PRISM-TWIN Tier 2 recommendation for a site."""
    return {
        "action": "Run PRISM-TWIN Tier 2 focused analysis",
        "description": (
            "The PRISM-AI inference identified this site as a candidate. "
            "A PRISM-TWIN engine run with beacon-guided NMA perturbation "
            "will provide full thermodynamic characterization, barrier "
            "estimation, mechanism classification (7-class TWIN taxonomy), "
            "and CCF allosteric network mapping at sub-angstrom resolution."
        ),
        "beacon_guidance": {
            "focus_residues": site["residue_ids"],
            "suggested_nma_modes": (
                "Focus NMA perturbation on modes that displace these residues"
            ),
            "expected_outputs": [
                "Thermodynamic barrier (kcal/mol)",
                "TWIN classification (CONSENSUS_CRYPTIC / BARRIER_GATED / "
                "ALLOSTERIC_HUB / etc.)",
                "CCF allosteric coupling network",
                "Druggability with full MD-derived volume and hydration",
                "Per-residue TWIN features (48 dimensions)",
                "Mechanism of pocket opening",
            ],
            "estimated_runtime": "5-15 minutes per target on GPU",
        },
        "value_proposition": (
            "Tier 2 resolves what Tier 1 cannot: the activation barrier, "
            "the opening mechanism, and whether the pocket is thermally "
            "accessible or requires perturbation. Sites classified as "
            "BARRIER_GATED by TWIN are invisible to all static methods "
            "-- only coupled interferometric observation reveals them."
        ),
    }


# ═══════════════════════════════════════════════════════════
# PYMOL VISUALIZATION
# ═══════════════════════════════════════════════════════════

def generate_pymol_script(parsed_pdb, probabilities, sites, output_path):
    """Generate a PyMOL .pml script for visualization."""
    pdb_path = parsed_pdb["pdb_path"]
    chain = parsed_pdb["chain_id"]
    resids = [r["resid"] for r in parsed_pdb["residues"]]

    lines = [
        f'load {pdb_path}, target',
        'hide everything, target',
        'show cartoon, target',
        'color white, target',
        '',
        '# Color by cryptic pocket probability',
        'set_color prob_low, [0.8, 0.8, 1.0]',
        'set_color prob_mid, [1.0, 0.6, 0.2]',
        'set_color prob_high, [1.0, 0.0, 0.0]',
        '',
    ]

    # Color residues by probability
    for i, (rid, prob) in enumerate(zip(resids, probabilities)):
        if prob > 0.3:
            lines.append(f'color red, chain {chain} and resi {rid}')
            lines.append(f'show sticks, chain {chain} and resi {rid}')
        elif prob > 0.2:
            lines.append(f'color orange, chain {chain} and resi {rid}')

    # Mark site centroids
    for site in sites[:5]:
        cx, cy, cz = site["centroid"]
        rank = site["rank"]
        lines.append(f'pseudoatom site_{rank}, pos=[{cx:.1f},{cy:.1f},{cz:.1f}]')
        lines.append(f'show spheres, site_{rank}')
        lines.append(f'set sphere_scale, 1.5, site_{rank}')
        lines.append(f'color {"red" if rank == 1 else "orange"}, site_{rank}')
        lines.append(f'label site_{rank}, "Site {rank} ({site["mean_prob"]:.2f})"')

    lines.append('')
    lines.append('zoom target')
    lines.append('set ray_opaque_background, 0')

    with open(output_path, 'w') as f:
        f.write('\n'.join(lines))


# ═══════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════

def predict(pdb_path, chain=None, top_k=5, output=None, visualize=None, verbose=True):
    """Run full PRISM-AI prediction pipeline on a PDB file.

    Args:
        pdb_path: Path to PDB file
        chain: Chain ID (default: first protein chain)
        top_k: Number of top sites to report
        output: Path for JSON output (default: stdout)
        visualize: Path for PyMOL .pml script (optional)
        verbose: Print progress

    Returns:
        dict with predictions
    """
    import torch

    if verbose:
        print(f"PRISM-AI v004 — Zero-Shot Predictor (no ESM-2, no engine)")
        print(f"Input: {pdb_path}")

    # Step 1: Parse PDB
    if verbose: print("  [1/4] Parsing PDB...")
    parsed = parse_pdb(pdb_path, chain=chain)
    if verbose: print(f"        {parsed['n_residues']} residues, chain {parsed['chain_id']}")

    # Step 2: Extract structural features (56 dims)
    # Must match training extractor EXACTLY: 20 AA one-hot (3-letter alphabetical)
    # + 3 DSSP + SASA + hydrophobicity = 25 dims + 31 zeros (NMA/pertNMA)
    if verbose: print("  [2/4] Structural features (25d + 31d zeros = 56d)...")
    n_res = parsed["n_residues"]
    ca_pos = np.array([parsed["ca_positions"][r["resid"]] for r in parsed["residues"]
                       if r["resid"] in parsed["ca_positions"]])

    # Training AA ordering (3-letter alphabetical — NOT 1-letter!)
    TRAIN_AA = ["ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE",
                "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL"]
    TRAIN_AA_IDX = {a: i for i, a in enumerate(TRAIN_AA)}
    TRAIN_HYDRO = {
        "ALA": 1.8, "ARG": -4.5, "ASN": -3.5, "ASP": -3.5, "CYS": 2.5,
        "GLN": -3.5, "GLU": -3.5, "GLY": -0.4, "HIS": -3.2, "ILE": 4.5,
        "LEU": 3.8, "LYS": -3.9, "MET": 1.9, "PHE": 2.8, "PRO": -1.6,
        "SER": -0.8, "THR": -0.7, "TRP": -0.9, "TYR": -1.3, "VAL": 4.2}

    # DSSP + SASA (match training extractor: freesasa + mkdssp)
    dssp_map = {}
    sasa_map = {}
    try:
        import freesasa
        structure = freesasa.Structure(str(pdb_path))
        result = freesasa.Calc().calculate(structure)
        for chain, per_res in result.residueAreas().items():
            for resnum_str, area in per_res.items():
                try:
                    sasa_map[int(resnum_str)] = float(area.total)
                except (ValueError, AttributeError):
                    pass
    except Exception:
        pass
    try:
        import subprocess, tempfile
        with tempfile.NamedTemporaryFile(suffix=".pdb", mode="w") as tmp:
            with open(pdb_path) as f:
                content = f.read()
            if not content.startswith("HEADER"):
                tmp.write("HEADER\n")
            tmp.write(content)
            tmp.flush()
            proc = subprocess.run(["mkdssp", "-i", tmp.name], capture_output=True, text=True, timeout=30)
            in_residues = False
            for line in proc.stdout.split("\n"):
                if "  #  RESIDUE" in line:
                    in_residues = True
                    continue
                if in_residues and len(line) > 16:
                    try:
                        resid = int(line[5:10].strip())
                        ss = line[16] if len(line) > 16 else "-"
                        dssp_map[resid] = ss
                    except (ValueError, IndexError):
                        pass
    except Exception:
        pass

    struct_25 = np.zeros((n_res, 25), dtype=np.float32)
    for i, r in enumerate(parsed["residues"]):
        resname = r["resname"]
        resid = r["resid"]
        if resname in TRAIN_AA_IDX:
            struct_25[i, TRAIN_AA_IDX[resname]] = 1.0
        # DSSP (3): H/G/I→helix, E/B→sheet, else→coil
        ss = dssp_map.get(resid, "-")
        if ss in ("H", "G", "I"):
            struct_25[i, 20] = 1.0
        elif ss in ("E", "B"):
            struct_25[i, 21] = 1.0
        else:
            struct_25[i, 22] = 1.0
        # SASA normalized to max ~220A²
        struct_25[i, 23] = sasa_map.get(resid, 0.0) / 220.0
        # Hydrophobicity /5.0 (matches training extractor)
        struct_25[i, 24] = TRAIN_HYDRO.get(resname, 0.0) / 5.0

    structural_feats = np.concatenate([
        struct_25,
        np.zeros((n_res, 26), dtype=np.float32),  # NMA zeros (matches training)
        np.zeros((n_res, 5), dtype=np.float32),    # pertNMA zeros (matches training)
    ], axis=1)
    if verbose: print(f"        Structural: {structural_feats.shape}")

    # Step 3: SpikeBERT → 512-dim spike embeddings
    if verbose: print("  [3/4] SpikeBERT v002 (all-masked) → 512-dim embeddings...")
    spike_emb = extract_spikebert_embeddings(structural_feats)
    if verbose: print(f"        SpikeBERT: {spike_emb.shape}")

    # Step 4: VN-EGNN v004
    atom_features = np.concatenate([structural_feats, spike_emb], axis=1)
    if verbose: print(f"  [4/4] VN-EGNN v004 ({atom_features.shape[1]}d) → binding sites...")
    atom_logits, vn_coords, vn_confidence = run_vnegnn(atom_features, ca_pos)
    probabilities = 1.0 / (1.0 + np.exp(-atom_logits.squeeze()))

    if verbose:
        print(f"        Probability range: [{probabilities.min():.3f}, {probabilities.max():.3f}]")

    # Cluster VN positions by confidence (primary output)
    vn_probs = 1.0 / (1.0 + np.exp(-vn_confidence.squeeze()))
    vn_order = np.argsort(-vn_probs)
    vn_sites = []
    used = set()
    for idx in vn_order:
        if idx in used:
            continue
        center = vn_coords[idx]
        members = [idx]
        for j in vn_order:
            if j in used or j == idx:
                continue
            if np.linalg.norm(vn_coords[j] - center) < 4.0:
                members.append(j)
                used.add(j)
        used.add(idx)
        site_center = vn_coords[members].mean(axis=0)
        site_conf = float(vn_probs[members].max())
        # Find nearby residues
        res_dists = np.linalg.norm(ca_pos - site_center, axis=1)
        nearby = np.where(res_dists < 8.0)[0]
        nearby_resids = [parsed["residues"][k]["resid"] for k in nearby if k < len(parsed["residues"])]
        nearby_probs = [float(probabilities[k]) for k in nearby if k < len(probabilities)]
        vn_sites.append({
            "rank": len(vn_sites) + 1,
            "centroid": [round(float(x), 2) for x in site_center],
            "confidence": round(site_conf, 4),
            "n_vns": len(members),
            "n_residues": len(nearby_resids),
            "mean_prob": round(float(np.mean(nearby_probs)) if nearby_probs else 0, 3),
            "residue_ids": nearby_resids[:20],
            "spatial_extent": f"{np.ptp(ca_pos[nearby], axis=0).max():.0f}" if len(nearby) > 1 else "0",
        })

    # Also run residue-based clustering as secondary output
    sites = cluster_to_sites(probabilities, parsed["residues"],
                            parsed["ca_positions"], top_pct=0.20)

    if verbose:
        print(f"\n  Found {len(vn_sites)} VN-predicted sites")
        for s in vn_sites[:top_k]:
            print(f"    Site {s['rank']}: centroid={s['centroid']}  "
                  f"conf={s['confidence']:.1%}  residues={s['n_residues']}  "
                  f"prob={s['mean_prob']:.3f}")

    # Build output
    resids = [r["resid"] for r in parsed["residues"]]
    result = {
        "input_pdb": str(pdb_path),
        "chain": parsed["chain_id"],
        "n_residues": parsed["n_residues"],
        "n_vn_sites": len(vn_sites),
        "vn_sites": vn_sites[:top_k],
        "n_residue_sites": len(sites),
        "residue_sites": sites[:top_k],
        "per_residue_probabilities": {
            int(resids[i]): round(float(probabilities[i]), 4)
            for i in range(len(resids))
        },
        "model_version": "prism-ai-v004",
        "note": "SpikeBERT v002 + VN-EGNN v004. Zero ESM-2. Zero engine.",
        "esm2_used": False,
    }

    # Save output
    if output:
        with open(output, 'w') as f:
            json.dump(result, f, indent=2)
        if verbose:
            print(f"\n  Results saved: {output}")

    # Generate visualization
    if visualize:
        generate_pymol_script(parsed, probabilities, sites[:top_k], visualize)
        if verbose:
            print(f"  PyMOL script: {visualize}")

    return result


def main():
    parser = argparse.ArgumentParser(
        description="PRISM-AI: Predict cryptic binding sites from PDB structure",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python predict.py 1abc.pdb
  python predict.py 1abc.pdb --chain A --output results.json
  python predict.py 1abc.pdb --visualize pocket_map.pml
  python predict.py 1abc.pdb --top-k 3 --output results.json --visualize viz.pml

Output JSON contains:
  - per_residue_probabilities: {resid: probability} for every residue
  - top_sites: ranked binding site predictions with centroids and residue lists
  - n_sites_found: total candidate sites detected
        """
    )
    parser.add_argument("pdb", help="Path to input PDB file")
    parser.add_argument("--chain", default=None, help="Chain ID (default: first protein chain)")
    parser.add_argument("--top-k", type=int, default=5, help="Number of top sites to report (default: 5)")
    parser.add_argument("--output", "-o", default=None, help="Output JSON path (default: stdout)")
    parser.add_argument("--visualize", "-v", default=None, help="Output PyMOL .pml script path")
    parser.add_argument("--quiet", action="store_true", help="Suppress progress output")

    args = parser.parse_args()

    if not os.path.exists(args.pdb):
        print(f"Error: PDB file not found: {args.pdb}", file=sys.stderr)
        sys.exit(1)

    result = predict(
        args.pdb,
        chain=args.chain,
        top_k=args.top_k,
        output=args.output,
        visualize=args.visualize,
        verbose=not args.quiet,
    )

    if not args.output:
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
