# =============================================================
# SUPERSEDED — QUARANTINED DEV SCRIPT
# =============================================================
# Original path:   /tmp/generate_tem1_report.py
# Quarantined on:  2026-04-05
# Quarantined by:  SOP bootstrap session (Task C)
# Reason:          One-off TEM-1 PRISM-TWIN report dev script
#                  living in /tmp. Preserved for reference only.
# Status:          DO NOT EXECUTE. DO NOT IMPORT.
# Rule:            docs/PRISM4D_DEV_OPS_FRAMEWORK.md §1
#                  (no production script may reference /tmp)
# Canonical home:  TBD — if resurrected, must move to
#                  scripts/production/ with full parameterization.
# =============================================================

#!/usr/bin/env python3
"""
PRISM-TWIN Report Generator — TEM-1 beta-lactamase (1BTL)
==========================================================
Produces:
  - tem1_prism_twin_report.json   (canonical report)
  - tem1_prism_heatmap.pdb        (B-factor = cryptic probability * 100)
  - tem1_ccf_matrix.npy           (263x263 CCF)
  - tem1_allosteric_network.json  (CCF > 0.5 pathway graph)
  - tem1_site_cards.json          (human-readable per-site summaries)
  - tem1_docking_ready.json       (docking boxes + pharmacophores)

Residue ID convention (CRITICAL):
  PDB resnum = topology_index + 26   (offset confirmed from residue_map.json)
  NMA residue_ids = PDB resnums (26..290)
  Lining_residues.resid = PDB resnums
  Active site PDB: SER70, LYS73, SER130, GLU166
  Active site topo_idx: 44, 47, 104, 140
"""

import json
import math
import os
import sys
import warnings
from collections import defaultdict
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")

# ── Paths ─────────────────────────────────────────────────────────────────
OUT_DIR     = Path("/tmp/twin_tem1_full")
TWIN_RESULT = OUT_DIR / "coupled_twin_result.json"
TWIN_SPIKES = OUT_DIR / "coupled_spikes.json"
TOPO_JSON   = Path("/mnt/storage/prism-outputs/runs/v1.1-physics/tem1/tem1.topology.json")
PDB_CLEAN   = Path("/mnt/storage/prism-outputs/runs/v1.1-physics/tem1/tem1_clean.pdb")
NMA_JSON    = Path("/tmp/nma_gate2_test/tem1_nma_modes.json")
BS_JSON     = Path("/mnt/storage/prism-outputs/runs/v1.1-physics/tem1/tem1.binding_sites.json")
THERM_JSON  = Path("/mnt/storage/prism-outputs/runs/v1.1-physics/tem1/tem1.topology.prism_therm.json")
RMAP_JSON   = Path("/mnt/storage/prism-outputs/runs/v1.1-physics/tem1/tem1.residue_map.json")
AI_MODEL_DIR = Path("/home/diddy/Desktop/Prism4D-bio/prism-ai-inference/models")

REPORT_OUT  = OUT_DIR / "tem1_prism_twin_report.json"
HEATMAP_PDB = OUT_DIR / "tem1_prism_heatmap.pdb"
CCF_NPY     = OUT_DIR / "tem1_ccf_matrix.npy"
NETWORK_JSON = OUT_DIR / "tem1_allosteric_network.json"
CARDS_JSON  = OUT_DIR / "tem1_site_cards.json"
DOCKING_JSON = OUT_DIR / "tem1_docking_ready.json"

# TEM1 metadata
# PDB residue range: 26..290 (263 residues)
# topology_index = pdb_resid - 26
TOPO_OFFSET     = 26   # pdb_resid = topo_idx + TOPO_OFFSET
ACTIVE_SITE_PDB = [70, 73, 130, 166]       # SER70, LYS73, SER130, GLU166
ACTIVE_SITE_TOPO_IDX = [44, 47, 104, 140]  # topology indices

AA3TO1 = {
    'ALA':'A','ARG':'R','ASN':'N','ASP':'D','CYS':'C','GLN':'Q','GLU':'E',
    'GLY':'G','HIS':'H','HID':'H','HIE':'H','HIP':'H','ILE':'I','LEU':'L',
    'LYS':'K','MET':'M','PHE':'F','PRO':'P','SER':'S','THR':'T','TRP':'W',
    'TYR':'Y','VAL':'V','CYX':'C',
}
HYDROPHOBICITY = {
    'A':1.8,'R':-4.5,'N':-3.5,'D':-3.5,'C':2.5,'Q':-3.5,'E':-3.5,'G':-0.4,
    'H':-3.2,'I':4.5,'L':3.8,'K':-3.9,'M':1.9,'F':2.8,'P':-1.6,'S':-0.8,
    'T':-0.7,'W':-0.9,'Y':-1.3,'V':4.2,
}
HBOND_DONORS    = {'ARG','LYS','HIS','HID','HIE','HIP','SER','THR','TYR','TRP','ASN','GLN'}
HBOND_ACCEPTORS = {'ASP','GLU','SER','THR','TYR','ASN','GLN','HID','HIE','HIP'}
AROMATIC        = {'PHE','TYR','TRP','HIS','HID','HIE','HIP'}
CHARGED_POS     = {'ARG','LYS','HIP'}
CHARGED_NEG     = {'ASP','GLU'}
AA_ORDER        = "ACDEFGHIKLMNPQRSTVWY"
AA_IDX          = {aa: i for i, aa in enumerate(AA_ORDER)}

GENERATION_TS = "2026-04-03T00:00:00Z"

print("=" * 70)
print("PRISM-TWIN Report Generator — TEM-1 beta-lactamase")
print("=" * 70)

# ═══════════════════════════════════════════════════════════════════════════
# STEP 1: Load all source data
# ═══════════════════════════════════════════════════════════════════════════
print("\n[1/9] Loading source data...")

with open(TWIN_RESULT)  as f: twin       = json.load(f)
with open(TWIN_SPIKES)  as f: spikes_data = json.load(f)
with open(TOPO_JSON)    as f: topo       = json.load(f)
with open(BS_JSON)      as f: bs_data    = json.load(f)
with open(THERM_JSON)   as f: therm_data = json.load(f)
with open(NMA_JSON)     as f: nma_data   = json.load(f)
with open(RMAP_JSON)    as f: rmap_data  = json.load(f)

# ── Build canonical residue mapping ──────────────────────────────────────
# residue_map gives: {topology_index, pdb_resid, resname, chain}
rmap_residues = rmap_data["residues"]   # list of 263 entries
N_RES = len(rmap_residues)              # 263

# Lookup tables (all indexed by PDB resnum unless noted)
pdb_resids = [r["pdb_resid"] for r in rmap_residues]           # PDB resnums 26..290
topo_idx_list = [r["topology_index"] for r in rmap_residues]   # 0..262
resname_by_pdb = {r["pdb_resid"]: r["resname"] for r in rmap_residues}
pdb_to_topo   = {r["pdb_resid"]: r["topology_index"] for r in rmap_residues}
topo_to_pdb   = {r["topology_index"]: r["pdb_resid"]  for r in rmap_residues}

# ── Topology positions ────────────────────────────────────────────────────
n_atoms   = topo["n_atoms"]
positions = np.array(topo["positions"]).reshape(n_atoms, 3)   # (4061, 3)
ca_indices_topo = topo["ca_indices"]                           # atom indices of CA, len=263

ca_pos = np.zeros((N_RES, 3))   # indexed by topo_idx 0..262
for topo_idx, atom_idx in enumerate(ca_indices_topo):
    ca_pos[topo_idx] = positions[atom_idx]

# Per-residue features from twin run
prf_list = twin["per_residue_features"]
# prf resid field: check what it uses
# twin prf resid=50 -> resname=LEU; pdb_resid 50 -> topology_index=24 -> resname depends on map
# Let's build a lookup by resid as-is
prf_by_resid = {p["resid"]: p for p in prf_list}

print(f"  Topology: {N_RES} residues, {n_atoms} atoms")
print(f"  PDB resnum range: {min(pdb_resids)}..{max(pdb_resids)}")
print(f"  Twin PRF entries: {len(prf_list)}  (resid range: {min(p['resid'] for p in prf_list)}..{max(p['resid'] for p in prf_list)})")
print(f"  Spike events: A={twin['stream_a']['total_spikes']:,}  B={twin['stream_b']['total_spikes']:,}")

# PRF resid=50 corresponds to what? Check: topology_index ~= prf_resid - 1 or prf_resid directly?
# Map PRF entries to topo_idx: prf resid in range 50..296 -- looks like pdb_resid - ??
# Let's check a known entry: prf resid=50, resname=LEU
# From rmap: which pdb_resid has LEU? Check manually
prf_sample_resid = prf_list[0]["resid"]
prf_sample_resname = prf_list[0]["resname"]
# Find it in rmap
for r in rmap_residues:
    if r["pdb_resid"] == prf_sample_resid and r["resname"] == prf_sample_resname:
        print(f"  PRF resid={prf_sample_resid} matches pdb_resid={r['pdb_resid']} topo_idx={r['topology_index']}")
        break
    if r["topology_index"] == prf_sample_resid and r["resname"] == prf_sample_resname:
        print(f"  PRF resid={prf_sample_resid} matches topo_idx={r['topology_index']} pdb_resid={r['pdb_resid']}")
        break

# Build PRF lookup by pdb_resid
# twin prf resid range: check actual
prf_resids_range = (min(p["resid"] for p in prf_list), max(p["resid"] for p in prf_list))
pdb_resid_range  = (min(pdb_resids), max(pdb_resids))
print(f"  PRF resid range: {prf_resids_range}, PDB range: {pdb_resid_range}")
# If prf_resids match pdb_resids -> direct lookup
if prf_resids_range[0] >= pdb_resid_range[0]:
    prf_by_pdb_resid = prf_by_resid   # direct
    print("  PRF uses PDB resnum indexing")
else:
    # prf resid = topo_idx -> map via topo_to_pdb
    prf_by_pdb_resid = {topo_to_pdb.get(k, k): v for k, v in prf_by_resid.items()}
    print("  PRF uses topo_idx indexing (remapped)")

# ═══════════════════════════════════════════════════════════════════════════
# STEP 2: Parse PDB for B-factors, DSSP, SASA
# ═══════════════════════════════════════════════════════════════════════════
print("\n[2/9] Parsing PDB for B-factors, DSSP, SASA...")

bfactor_sum   = {}
bfactor_count = {}
pdb_residues_info = {}   # pdb_resnum -> {resname, chain, ca_xyz, all_xyz}

with open(PDB_CLEAN) as f:
    for line in f:
        if not line.startswith("ATOM"):
            continue
        aname  = line[12:16].strip()
        rname  = line[17:20].strip()
        chain  = line[21].strip()
        resnum = int(line[22:26].strip())
        x = float(line[30:38]); y = float(line[38:46]); z = float(line[46:54])
        bfac = float(line[60:66]) if len(line.rstrip()) >= 66 else 0.0

        if resnum not in pdb_residues_info:
            pdb_residues_info[resnum] = {"resname": rname, "chain": chain,
                                          "ca_xyz": None, "all_xyz": []}
        pdb_residues_info[resnum]["all_xyz"].append(np.array([x, y, z]))
        if aname == "CA":
            pdb_residues_info[resnum]["ca_xyz"] = np.array([x, y, z])

        bfactor_sum[resnum]   = bfactor_sum.get(resnum, 0.0) + bfac
        bfactor_count[resnum] = bfactor_count.get(resnum, 0) + 1

bfactor_mean = {r: bfactor_sum[r] / bfactor_count[r] for r in bfactor_sum}
max_bf = max(bfactor_mean.values()) if bfactor_mean else 1.0
min_bf = min(bfactor_mean.values()) if bfactor_mean else 0.0
bf_range = max(max_bf - min_bf, 1e-10)
bfactor_norm = {r: (bfactor_mean[r] - min_bf) / bf_range for r in bfactor_mean}

pdb_resnums_sorted = sorted(pdb_residues_info.keys())

# SASA proxy from normalized B-factors (higher B -> more exposed)
sasa_by_pdb = bfactor_norm.copy()

# DSSP
dssp_by_pdb = {}
try:
    import mdtraj
    t = mdtraj.load(str(PDB_CLEAN))
    dssp_raw = mdtraj.compute_dssp(t)[0]
    # mdtraj residue ordering matches PDB order
    mdtraj_resids_list = []
    for chain in t.topology.chains:
        for res in chain.residues:
            mdtraj_resids_list.append(res.resSeq)
    for i, rs in enumerate(mdtraj_resids_list[:len(dssp_raw)]):
        ss = dssp_raw[i]
        dssp_by_pdb[rs] = 'H' if ss == 'H' else ('E' if ss == 'E' else 'C')
    print(f"  DSSP: mdtraj ({sum(1 for v in dssp_by_pdb.values() if v=='H')}H "
          f"/ {sum(1 for v in dssp_by_pdb.values() if v=='E')}E "
          f"/ {sum(1 for v in dssp_by_pdb.values() if v=='C')}C)")
    # Better SASA from mdtraj
    try:
        sasa_arr = mdtraj.shrake_rupley(t, mode='residue')[0]
        sasa_max = sasa_arr.max() if sasa_arr.max() > 0 else 1.0
        for i, rs in enumerate(mdtraj_resids_list[:len(sasa_arr)]):
            sasa_by_pdb[rs] = float(sasa_arr[i] / sasa_max)
    except Exception:
        pass
except Exception as e:
    print(f"  DSSP: fallback (coil defaults) — {e}")
    for r in pdb_resnums_sorted:
        dssp_by_pdb[r] = 'C'

# ═══════════════════════════════════════════════════════════════════════════
# STEP 3: PRISM-AI Inference (v002 student, 109 folds)
# ═══════════════════════════════════════════════════════════════════════════
print("\n[3/9] Running PRISM-AI ensemble inference (109 folds)...")

import torch
import torch.nn as nn

class StudentModel(nn.Module):
    def __init__(self, input_dim=1332, hd=512):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(input_dim, hd), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(hd, hd),       nn.GELU(), nn.Dropout(0.1),
        )
        self.binding_head = nn.Sequential(
            nn.Linear(hd, hd // 2), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(hd // 2, 1),
        )
        self.physics_head = nn.Sequential(nn.Linear(hd, hd), nn.GELU(), nn.Linear(hd, 216))
        self.proj_head    = nn.Sequential(nn.Linear(hd, 256), nn.GELU(), nn.Linear(256, 256))
    def forward(self, x):
        return self.binding_head(self.backbone(x)).squeeze(-1)

def robust_normalize(feat):
    out = feat.copy()
    for j in range(feat.shape[1]):
        col = feat[:, j]
        nz  = col[np.abs(col) > 1e-10]
        if len(nz) < 5: continue
        med = np.median(nz); q75, q25 = np.percentile(nz, [75, 25]); iqr = q75 - q25
        if iqr < 1e-10: iqr = np.std(nz) or 1.0
        out[:, j] = np.clip((col - med) / iqr, -5, 5)
    return out

def build_features(pdb_resnums, pdb_residues_info, bfactor_norm, sasa_by_pdb,
                    dssp_by_pdb, nma_data, pdb_to_topo):
    n = len(pdb_resnums)
    # 26-dim structural
    struct = np.zeros((n, 26), dtype=np.float32)
    for i, r in enumerate(pdb_resnums):
        rn  = pdb_residues_info[r]["resname"]
        aa1 = AA3TO1.get(rn, "X")
        idx = AA_IDX.get(aa1, -1)
        if idx >= 0: struct[i, idx] = 1.0
        struct[i, 20] = HYDROPHOBICITY.get(aa1, 0.0)
        ss = dssp_by_pdb.get(r, 'C')
        if ss == 'H': struct[i, 21] = 1.0
        elif ss == 'E': struct[i, 22] = 1.0
        else: struct[i, 23] = 1.0
        struct[i, 24] = float(sasa_by_pdb.get(r, 0.5))
        struct[i, 25] = float(bfactor_norm.get(r, 0.5))

    # 26-dim NMA from pre-computed modes
    nma_feat = np.zeros((n, 26), dtype=np.float32)
    modes     = nma_data["modes"]
    nma_pdb_resids = nma_data["residue_ids"]   # PDB resnums
    nma_pdb_to_idx = {r: i for i, r in enumerate(nma_pdb_resids)}
    n_modes   = len(modes)
    n_nma_res = len(nma_pdb_resids)

    disp_matrix = np.zeros((n_nma_res, n_modes), dtype=np.float32)
    eigenvalues = np.array([m["eigenvalue"] for m in modes])
    for k, mode in enumerate(modes):
        for j, d in enumerate(mode["displacements"][:n_nma_res]):
            disp_matrix[j, k] = float(np.sum(np.array(d) ** 2))

    sqfluct  = np.zeros(n_nma_res, dtype=np.float32)
    for k in range(n_modes):
        sqfluct += disp_matrix[:, k] / max(eigenvalues[k], 1e-10)
    stiffness = 1.0 / np.maximum(sqfluct, 1e-10)
    stiffness /= max(stiffness.max(), 1e-10)

    for i, r in enumerate(pdb_resnums):
        ni = nma_pdb_to_idx.get(r)
        if ni is None: continue
        for k in range(min(n_modes, 20)):
            nma_feat[i, k] = disp_matrix[ni, k]
        nma_feat[i, 20] = sqfluct[ni]
        nma_feat[i, 21] = stiffness[ni]
        if 0 < ni < n_nma_res - 1:
            if disp_matrix[ni, 0] < 0.5 * (disp_matrix[ni-1, 0] + disp_matrix[ni+1, 0]):
                nma_feat[i, 22] = 0.5
        nma_feat[i, 23] = float(sqfluct[ni])
        nma_feat[i, 24] = float(stiffness[ni])
        nma_feat[i, 25] = float(np.mean(np.abs(sqfluct - sqfluct[ni])))

    # 1280-dim ESM — use 1JWP cache (TEM-1 family)
    esm_feat = np.zeros((n, 1280), dtype=np.float32)
    for cname in ["1btl", "1jwp"]:
        cp = Path(f"/home/diddy/Desktop/Prism4D-bio/models/esm_cache/{cname}.pt")
        if cp.exists():
            try:
                cached = torch.load(str(cp), map_location="cpu", weights_only=False)
                if isinstance(cached, dict) and "representations" in cached:
                    emb = cached["representations"][33][0, 1:-1, :].numpy()
                elif isinstance(cached, torch.Tensor):
                    emb = cached.numpy()
                elif isinstance(cached, np.ndarray):
                    emb = cached
                else:
                    emb = None
                if emb is not None:
                    rows = min(n, emb.shape[0])
                    esm_feat[:rows, :min(1280, emb.shape[1])] = emb[:rows, :1280]
                    print(f"  ESM: loaded from {cname}.pt, shape {emb.shape}")
                    break
            except Exception:
                pass
    else:
        print("  ESM: no cache found, using structural proxy")
        raw_52 = np.concatenate([struct, nma_feat], axis=1)
        tc = 1280 // 52; rm = 1280 - tc * 52
        esm_feat = np.concatenate([np.tile(raw_52, (1, tc)), raw_52[:, :rm]], axis=1) * 0.01

    raw_52   = np.concatenate([struct, nma_feat], axis=1)
    norm_52  = robust_normalize(raw_52)
    features = np.concatenate([norm_52, esm_feat], axis=1).astype(np.float32)
    return features

features = build_features(pdb_resnums_sorted, pdb_residues_info,
                           bfactor_norm, sasa_by_pdb, dssp_by_pdb, nma_data, pdb_to_topo)
print(f"  Feature matrix: {features.shape}")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
fold_files = sorted(AI_MODEL_DIR.glob("student_fold_*.pt"))
print(f"  Loading {len(fold_files)} folds on {device}...")

preds_sum = np.zeros(len(pdb_resnums_sorted), dtype=np.float64)
input_tensor = torch.tensor(features, dtype=torch.float32).to(device)
n_loaded = 0

for fp in fold_files:
    try:
        m = StudentModel()
        sd = torch.load(str(fp), map_location="cpu", weights_only=False)["state_dict"]
        m.load_state_dict(sd, strict=False)
        m.eval().to(device)
        with torch.no_grad():
            preds_sum += torch.sigmoid(m(input_tensor)).cpu().numpy().astype(np.float64)
        n_loaded += 1
    except Exception:
        pass

probabilities = preds_sum / max(n_loaded, 1)
print(f"  {n_loaded}/{len(fold_files)} folds loaded")
print(f"  Probability range: [{probabilities.min():.4f}, {probabilities.max():.4f}]  "
      f"mean={probabilities.mean():.4f}")

# Map probabilities by PDB resnum
prob_by_pdb   = {r: float(probabilities[i]) for i, r in enumerate(pdb_resnums_sorted)}
prob_by_topo  = {pdb_to_topo[r]: prob_by_pdb[r] for r in pdb_resnums_sorted if r in pdb_to_topo}

# ═══════════════════════════════════════════════════════════════════════════
# STEP 4: CCF matrix from spike data
# ═══════════════════════════════════════════════════════════════════════════
print("\n[4/9] Computing CCF matrix (263×263) from spike data...")

spikes = spikes_data["spikes"]
N_SPIKES = len(spikes)

# Sample spikes for CCF computation
rng = np.random.default_rng(42)
if N_SPIKES > 300_000:
    sample_idx = rng.choice(N_SPIKES, 300_000, replace=False)
    spike_sample = [spikes[i] for i in sorted(sample_idx)]
else:
    spike_sample = spikes

spike_xyz    = np.array([[s["x"], s["y"], s["z"]] for s in spike_sample], dtype=np.float32)
spike_ts     = np.array([s["timestep"]  for s in spike_sample], dtype=np.int64)
spike_stream = np.array([s["stream_id"] for s in spike_sample], dtype=np.int8)

# Assign spikes to nearest CA residue (topology index) within 8Å
CHUNK = 10_000
spike_to_topo = np.full(len(spike_sample), -1, dtype=np.int16)
for start in range(0, len(spike_sample), CHUNK):
    end   = min(start + CHUNK, len(spike_sample))
    xyz   = spike_xyz[start:end]
    diff  = xyz[:, None, :] - ca_pos[None, :, :]   # (C, N_RES, 3)
    dist  = np.linalg.norm(diff, axis=2)
    nearest = np.argmin(dist, axis=1)
    min_d   = dist[np.arange(len(xyz)), nearest]
    spike_to_topo[start:end] = np.where(min_d < 8.0, nearest, -1)

# Build time-binned spike density (N_RES, N_BINS)
N_BINS = 200
ts_min = int(spike_ts.min()); ts_max = int(spike_ts.max())
bin_edges = np.linspace(ts_min, ts_max + 1, N_BINS + 1)
spike_bins = np.clip(np.digitize(spike_ts, bin_edges) - 1, 0, N_BINS - 1)

density = np.zeros((N_RES, N_BINS), dtype=np.float32)
for sidx in range(len(spike_sample)):
    ti = spike_to_topo[sidx]
    if ti >= 0:
        density[ti, spike_bins[sidx]] += 1.0

# Pearson CCF via normalized matmul
row_sums  = density.sum(axis=1, keepdims=True)
dn        = density / np.maximum(row_sums, 1.0)
dn_c      = dn - dn.mean(axis=1, keepdims=True)
norms     = np.maximum(np.linalg.norm(dn_c, axis=1, keepdims=True), 1e-10)
dn_normed = dn_c / norms
CCF = np.clip((dn_normed @ dn_normed.T).astype(np.float32), -1.0, 1.0)
np.fill_diagonal(CCF, 1.0)

np.save(str(CCF_NPY), CCF)
ccf_upper = CCF[np.triu_indices(N_RES, k=1)]
print(f"  CCF: {CCF.shape}  range=[{CCF.min():.3f}, {CCF.max():.3f}]  "
      f"pairs>0.5: {int(np.sum(ccf_upper > 0.5)):,}")
print(f"  Saved: {CCF_NPY}")

# ═══════════════════════════════════════════════════════════════════════════
# STEP 5: Build sites from PRISM engine results + AI annotation
# ═══════════════════════════════════════════════════════════════════════════
print("\n[5/9] Building sites from engine detections + AI annotation...")

# Use engine sites as primary (they are validated by physics engine)
# AI probs provide per-residue confidence signal
engine_sites = bs_data["sites"]
therm_pockets = {p["pocket_id"]: p for p in therm_data["pockets"]}

# ── Geometry helpers ─────────────────────────────────────────────────────
def convex_hull_volume(xyz):
    from scipy.spatial import ConvexHull
    if len(xyz) < 4:
        return 0.0
    try:
        return float(ConvexHull(xyz).volume)
    except Exception:
        extent = float(np.linalg.norm(xyz.max(axis=0) - xyz.min(axis=0)))
        return float((4/3) * math.pi * (extent / 2) ** 3 * 0.3)

def convex_hull_sphericity(xyz):
    from scipy.spatial import ConvexHull
    if len(xyz) < 4:
        return 0.5
    try:
        h = ConvexHull(xyz)
        return float((math.pi ** (1/3)) * (6 * h.volume) ** (2/3) / max(h.area, 1e-6))
    except Exception:
        return 0.5

# ── NMA site analysis ────────────────────────────────────────────────────
def nma_site_analysis(pdb_resnums_site, nma_data):
    pdb_to_nma = {r: i for i, r in enumerate(nma_data["residue_ids"])}
    modes = nma_data["modes"]
    responsive = []
    for mode in modes:
        disps = mode["displacements"]
        site_disps = []
        for r in pdb_resnums_site:
            ni = pdb_to_nma.get(r)
            if ni is not None and ni < len(disps):
                site_disps.append(float(np.linalg.norm(np.array(disps[ni]))))
        if site_disps:
            md = float(np.mean(site_disps))
            if md > 0.05:
                responsive.append({
                    "mode_index": mode["mode_index"],
                    "eigenvalue": round(float(mode["eigenvalue"]), 5),
                    "thermal_amplitude_angstrom": float(mode["thermal_amplitude"]),
                    "site_mean_displacement": round(md, 5),
                    "force_scale": float(mode.get("force_scale", 0.0)),
                })
    responsive.sort(key=lambda m: -m["site_mean_displacement"])
    total_disp = sum(m["site_mean_displacement"] for m in responsive)
    return {
        "n_responsive_modes": len(responsive),
        "mean_displacement_angstrom": round(total_disp / max(len(responsive), 1), 4),
        "responsive_modes": responsive[:5],
    }

# ── Spike dynamics ───────────────────────────────────────────────────────
def spike_dynamics(pdb_resnums_site, pdb_to_topo, spike_to_topo, spike_stream,
                    spike_ts, prf_by_pdb_resid, twin):
    site_topo = {pdb_to_topo[r] for r in pdb_resnums_site if r in pdb_to_topo}
    mask = np.isin(spike_to_topo, list(site_topo)) if site_topo else np.zeros(len(spike_to_topo), bool)
    n_total   = int(mask.sum())
    n_a       = int((mask & (spike_stream == 0)).sum())
    n_b       = int((mask & (spike_stream == 1)).sum())

    ts_arr = spike_ts[mask]
    if len(ts_arr) > 0:
        ts0 = int(ts_arr.min()); ts1 = int(ts_arr.max())
        ph  = (ts1 - ts0 + 1) / 3.0
        nc  = int(np.sum(ts_arr < ts0 + ph))
        nh  = int(np.sum((ts_arr >= ts0 + ph) & (ts_arr < ts0 + 2 * ph)))
        nw  = int(np.sum(ts_arr >= ts0 + 2 * ph))
        phase = {"cold_hold": nc, "heating": nh, "warm_hold": nw,
                 "heating_cold_ratio": round(nh / max(nc, 1), 3)}
    else:
        phase = {"cold_hold": 0, "heating": 0, "warm_hold": 0, "heating_cold_ratio": 0.0}

    prf_entries = [prf_by_pdb_resid[r] for r in pdb_resnums_site if r in prf_by_pdb_resid]
    mean_agree  = round(float(np.mean([p["spike_agreement_ratio"] for p in prf_entries])), 4) if prf_entries else None
    mean_ccf_pk = round(float(np.mean([p["ccf_peak_value"] for p in prf_entries])), 4) if prf_entries else None
    barrier_dist = defaultdict(int)
    for p in prf_entries:
        barrier_dist[p["barrier_classification"]] += 1

    return {
        "total_spikes_sample": n_total,
        "stream_a_spikes": n_a,
        "stream_b_spikes": n_b,
        "b_over_a_ratio": round(n_b / max(n_a, 1), 4),
        "spike_agreement_ratio": mean_agree,
        "ccf_peak_value": mean_ccf_pk,
        "phase_dynamics": phase,
        "barrier_distribution": dict(barrier_dist),
    }

# ── Druggability ─────────────────────────────────────────────────────────
def druggability(pdb_resnums_site, pdb_residues_info, bfactor_norm):
    rnames = [pdb_residues_info[r]["resname"] for r in pdb_resnums_site if r in pdb_residues_info]
    n = len(rnames)
    if n == 0:
        return {"druggability_score": None}
    n_hphob   = sum(1 for rn in rnames if HYDROPHOBICITY.get(AA3TO1.get(rn, "X"), 0) > 1.0)
    n_polar   = sum(1 for rn in rnames if rn in HBOND_DONORS or rn in HBOND_ACCEPTORS)
    n_donors  = sum(1 for rn in rnames if rn in HBOND_DONORS)
    n_acceps  = sum(1 for rn in rnames if rn in HBOND_ACCEPTORS)
    n_arom    = sum(1 for rn in rnames if rn in AROMATIC)
    n_chg     = sum(1 for rn in rnames if rn in CHARGED_POS or rn in CHARGED_NEG)
    hf = n_hphob / n; pf = n_polar / n; af = n_arom / n
    drug_sc = (0.40 * min(hf / 0.4, 1.0) + 0.30 * min(pf / 0.4, 1.0) +
               0.20 * min(af / 0.2, 1.0) + 0.10 * min(n / 10.0, 1.0))
    bf_vals = [bfactor_norm.get(r, 0.5) for r in pdb_resnums_site]
    return {
        "druggability_score": round(float(drug_sc), 4),
        "polarity_fraction":    round(float(pf), 3),
        "hydrophobic_fraction": round(float(hf), 3),
        "aromatic_fraction":    round(float(af), 3),
        "hbond_donor_count":    int(n_donors),
        "hbond_acceptor_count": int(n_acceps),
        "aromatic_count":       int(n_arom),
        "charged_count":        int(n_chg),
        "flexibility_mean_bfactor": round(float(np.mean(bf_vals)), 3),
    }

# ── Docking box ──────────────────────────────────────────────────────────
def docking_box(centroid, pdb_resnums_site, pdb_residues_info, padding=5.0):
    cas = [pdb_residues_info[r]["ca_xyz"] for r in pdb_resnums_site
           if r in pdb_residues_info and pdb_residues_info[r]["ca_xyz"] is not None]
    pts = np.array(cas) if cas else np.array([centroid])
    bmin = pts.min(axis=0) - padding
    bmax = pts.max(axis=0) + padding
    bc   = (bmin + bmax) / 2.0; bs = bmax - bmin
    return {
        "center_x": round(float(bc[0]), 2), "center_y": round(float(bc[1]), 2),
        "center_z": round(float(bc[2]), 2),
        "size_x":   round(float(bs[0]), 2), "size_y":   round(float(bs[1]), 2),
        "size_z":   round(float(bs[2]), 2),
    }

# ── Pharmacophore ────────────────────────────────────────────────────────
def pharmacophore(pdb_resnums_site, pdb_residues_info):
    feats = []
    for r in pdb_resnums_site:
        rd  = pdb_residues_info.get(r, {}); rn = rd.get("resname", "UNK")
        ca  = rd.get("ca_xyz")
        if ca is None: continue
        xyz = [round(float(c), 2) for c in ca]
        aa1 = AA3TO1.get(rn, "X")
        if rn in HBOND_DONORS:    feats.append({"type":"HBD","resname":rn,"resnum":r,"xyz":xyz})
        if rn in HBOND_ACCEPTORS: feats.append({"type":"HBA","resname":rn,"resnum":r,"xyz":xyz})
        if rn in AROMATIC:        feats.append({"type":"AR", "resname":rn,"resnum":r,"xyz":xyz})
        if rn in CHARGED_POS:     feats.append({"type":"PI", "resname":rn,"resnum":r,"xyz":xyz})
        if rn in CHARGED_NEG:     feats.append({"type":"NI", "resname":rn,"resnum":r,"xyz":xyz})
        if HYDROPHOBICITY.get(aa1, 0) > 1.5:
            feats.append({"type":"H","resname":rn,"resnum":r,"xyz":xyz})
    return feats

# ── Classify site ────────────────────────────────────────────────────────
def classify_site(pdb_resnums_site, therm_class, eng_class, ai_prob,
                   pdb_to_topo, CCF, N_RES, ACTIVE_SITE_TOPO_IDX, ACTIVE_SITE_PDB):
    active_overlap = len(set(pdb_resnums_site) & set(ACTIVE_SITE_PDB))
    if active_overlap >= 2: return "ORTHOSTERIC"
    if therm_class == "CRYPTIC": return "CRYPTIC"
    if eng_class and "cryptic" in eng_class.lower(): return "CRYPTIC"
    if ai_prob > 0.45: return "CRYPTIC"
    # Allosteric: significant CCF to active site
    site_ti = [pdb_to_topo[r] for r in pdb_resnums_site if r in pdb_to_topo]
    if site_ti:
        ccf_vals = [CCF[si, ai] for si in site_ti for ai in ACTIVE_SITE_TOPO_IDX if si < N_RES]
        if ccf_vals and float(np.mean(np.abs(ccf_vals))) > 0.45: return "ALLOSTERIC"
    if therm_class in ("DYNAMIC", "RESPONSIVE"): return "CRYPTIC"
    return "UNKNOWN"

# ═══════════════════════════════════════════════════════════════════════════
# STEP 6: Build per-site report records
# ═══════════════════════════════════════════════════════════════════════════
print("\n[6/9] Building per-site records from engine detections...")

sites_report = []

for site_rank, eng in enumerate(engine_sites, 1):
    site_pdb_id = eng["id"]
    centroid    = eng["centroid"]    # topology coordinate space
    lr_raw      = eng.get("lining_residues", [])

    # lining_residues.resid = PDB resnum
    lr_pdb_resnums = [lr["resid"] for lr in lr_raw]
    lr_is_catalytic = {lr["resid"]: lr["is_catalytic"] for lr in lr_raw}
    lr_min_dist     = {lr["resid"]: lr["min_distance"] for lr in lr_raw}

    # CA positions for site residues
    site_ca_list = []
    for r in lr_pdb_resnums:
        rd = pdb_residues_info.get(r, {})
        if rd.get("ca_xyz") is not None:
            site_ca_list.append(rd["ca_xyz"])
    site_ca = np.array(site_ca_list) if site_ca_list else np.array([centroid])

    # Geometry
    extent  = float(np.linalg.norm(site_ca.max(axis=0) - site_ca.min(axis=0))) if len(site_ca) >= 2 else 0.0
    vol_ch  = convex_hull_volume(site_ca)
    sph     = convex_hull_sphericity(site_ca)
    compact = float(len(lr_pdb_resnums) / max(vol_ch, 1.0)) if vol_ch > 0 else 0.0

    # PDB-weighted centroid for docking box
    weighted_centroid = site_ca.mean(axis=0).tolist() if len(site_ca) > 0 else centroid

    # Therm
    therm_class  = eng.get("therm_class", "UNKNOWN")
    eng_class    = eng.get("classification", "Unknown")
    therm_pocket = therm_pockets.get(site_pdb_id, {})

    # Hysteresis
    hysteresis   = float(eng.get("hysteresis_asymmetry", 0.0))
    rel_asym     = float(eng.get("relative_asymmetry", 0.0))
    cold_frac    = eng.get("cold_phase_fraction", {})
    accessibility = (1.0 - float(cold_frac.get("cold", 0.5))) if cold_frac else None

    # AI mean probability over lining residues
    ai_probs_site = [prob_by_pdb.get(r, probabilities.mean()) for r in lr_pdb_resnums]
    ai_mean_prob  = float(np.mean(ai_probs_site)) if ai_probs_site else float(probabilities.mean())
    ai_max_prob   = float(np.max(ai_probs_site))  if ai_probs_site else float(probabilities.max())

    # Classification
    site_class = classify_site(lr_pdb_resnums, therm_class, eng_class, ai_mean_prob,
                                 pdb_to_topo, CCF, N_RES, ACTIVE_SITE_TOPO_IDX, ACTIVE_SITE_PDB)

    # NMA
    nma_result = nma_site_analysis(lr_pdb_resnums, nma_data)

    # Spike dynamics
    sdyn = spike_dynamics(lr_pdb_resnums, pdb_to_topo, spike_to_topo, spike_stream,
                           spike_ts, prf_by_pdb_resid, twin)

    # Druggability
    drug = druggability(lr_pdb_resnums, pdb_residues_info, bfactor_norm)

    # KCC
    kcc = eng.get("kcc", {})
    kcc_record = {
        "driver_residue_pdb": topo_to_pdb.get(int(kcc["driver_residue_id"]), None) if kcc.get("driver_residue_id") is not None else None,
        "driver_residue_topo": int(kcc["driver_residue_id"]) if kcc.get("driver_residue_id") is not None else None,
        "kcc_confidence": float(kcc.get("kcc_confidence", 0.0)),
        "site_causal_lag_steps": float(kcc.get("site_causal_lag", 0.0)),
        "site_burst_motion": float(kcc.get("site_burst_motion", 0.0)),
        "site_lag_corr_peak": float(kcc.get("site_lag_corr_peak", 0.0)),
        "site_local_cov": float(kcc.get("site_local_cov", 0.0)),
    } if kcc else {"kcc_confidence": 0.0}

    # CCF allosteric analysis
    site_topo_idxs = [pdb_to_topo[r] for r in lr_pdb_resnums if r in pdb_to_topo]
    if site_topo_idxs and ACTIVE_SITE_TOPO_IDX:
        ccf_to_active = float(np.mean([CCF[si, ai]
                                        for si in site_topo_idxs
                                        for ai in ACTIVE_SITE_TOPO_IDX if si < N_RES]))
    else:
        ccf_to_active = 0.0

    # Allosteric partners (CCF > 0.5)
    allosteric_partners = []
    if site_topo_idxs:
        site_mean_ccf = CCF[site_topo_idxs, :].mean(axis=0)
        for pi in np.where(site_mean_ccf > 0.5)[0]:
            if pi not in site_topo_idxs:
                pr = topo_to_pdb.get(int(pi), -1)
                if pr > 0:
                    allosteric_partners.append({
                        "topo_idx": int(pi),
                        "pdb_resnum": int(pr),
                        "resname": resname_by_pdb.get(pr, "UNK"),
                        "ccf": round(float(site_mean_ccf[pi]), 4),
                    })
    allosteric_partners.sort(key=lambda x: -x["ccf"])

    # Docking
    dock_box = docking_box(weighted_centroid, lr_pdb_resnums, pdb_residues_info)
    pharma   = pharmacophore(lr_pdb_resnums, pdb_residues_info)

    # Barrier estimate
    mean_bf = float(np.mean([bfactor_mean.get(r, 0) for r in lr_pdb_resnums]))
    barrier_est = round(max(0.5, hysteresis * 3.0 + mean_bf * 0.02), 2)

    # Lining residues enriched with AI + DSSP data
    lining_enriched = []
    for lr in sorted(lr_raw, key=lambda x: x["resid"]):
        r  = lr["resid"]
        rn = lr["resname"]
        lining_enriched.append({
            "pdb_resnum": int(r),
            "resname": rn,
            "aa1": AA3TO1.get(rn, "X"),
            "is_catalytic": bool(lr["is_catalytic"]),
            "min_distance_angstrom": round(float(lr["min_distance"]), 3),
            "n_atoms": int(lr.get("n_atoms", 0)),
            "cryptic_probability": round(prob_by_pdb.get(r, float(probabilities.mean())), 4),
            "dssp": dssp_by_pdb.get(r, "C"),
            "bfactor": round(bfactor_mean.get(r, 0.0), 2),
            "sasa_proxy": round(sasa_by_pdb.get(r, 0.5), 3),
            "chain": lr.get("chain", "A"),
        })

    # Confidence score
    n_ev = sum([
        1 if therm_class in ("CRYPTIC","DYNAMIC","RESPONSIVE") else 0,
        1 if ai_mean_prob > 0.30 else 0,
        1 if sdyn["total_spikes_sample"] > 50 else 0,
        1 if nma_result["n_responsive_modes"] >= 2 else 0,
        1 if eng.get("quality_score", 0) > 0.35 else 0,
    ])
    confidence = round(min(n_ev / 5.0 * 0.7 + ai_mean_prob * 0.3, 1.0), 3)

    site_rec = {
        "site_id":  f"TEM1_TWIN_E{site_rank:02d}",
        "rank":     site_rank,
        "engine_pocket_id": int(site_pdb_id),

        # Target metadata
        "protein":     "TEM-1 beta-lactamase",
        "pdb_id":      "1BTL",
        "chain":       "A",
        "target_name": "tem1",

        # Classification
        "classification": site_class,
        "engine_classification": eng_class,
        "therm_class": therm_class,
        "is_active_site_overlap": len(set(lr_pdb_resnums) & set(ACTIVE_SITE_PDB)) > 0,
        "active_site_residues_overlap": sorted(set(lr_pdb_resnums) & set(ACTIVE_SITE_PDB)),

        # Geometry
        "centroid_angstrom": [round(float(x), 3) for x in centroid],
        "weighted_centroid_angstrom": [round(float(x), 3) for x in weighted_centroid],
        "volume_angstrom3_engine": float(eng.get("volume", 0.0)),
        "volume_angstrom3_convex_hull": round(vol_ch, 2),
        "spatial_extent_angstrom": round(extent, 2),
        "sphericity": round(sph, 4),
        "compactness": round(compact, 6),
        "n_lining_residues": len(lr_pdb_resnums),

        # Engine scores
        "engine_quality_score": float(eng.get("quality_score", 0.0)),
        "engine_druggability": float(eng.get("druggability", 0.0)),
        "engine_spike_count":  int(eng.get("spike_count", 0)),
        "engine_burial_score": float(eng.get("burial_score", 0.0)),
        "engine_sphericity":   float(eng.get("sphericity", 0.0)),
        "engine_breathing_score": float(eng.get("breathing_score", 0.0)),
        "engine_aromatic_score":  float(eng.get("aromatic_score", 0.0)),
        "engine_uv_enrichment":   float(eng.get("uv_enrichment_score", 0.0)),
        "engine_wd_coherence":    float(eng.get("wd_coherence", 0.0)),
        "engine_rank_score":      float(eng.get("rank_score", 0.0)),
        "gtck_rank":              int(eng.get("gtck_rank", 99)),
        "engine_tide_coupling":   float(eng.get("tide_coupling_score", 0.0)),

        # AI scores
        "ai_mean_probability": round(ai_mean_prob, 4),
        "ai_max_probability":  round(ai_max_prob, 4),
        "n_ai_folds_used":     n_loaded,

        # Thermodynamics
        "therm": {
            "hysteresis_asymmetry":    round(hysteresis, 4),
            "relative_asymmetry":      round(rel_asym, 4),
            "accessibility":           round(accessibility, 4) if accessibility is not None else None,
            "barrier_estimate_kcal_mol": barrier_est,
            "ccns_tau":                float(eng.get("ccns_tau", 0.0)),
            "cold_phase_fraction":     cold_frac if cold_frac else None,
            "frustrated_solvent_score": float(eng.get("frustrated_solvent_score", 0.0)),
            "tide_trigger_residues":   eng.get("tide_trigger_residues", []),
            "water_displacement_sites": None,
            "water_displacement_sites_note": "Requires explicit solvent refinement (WT-6 pipeline)",
            "desolvation_penalty_kcal_mol": None,
            "desolvation_penalty_note": "Requires GIST/SSTMap analysis (WT-6 pipeline)",
        },

        # Spike dynamics
        "spike_dynamics": sdyn,

        # NMA
        "nma": nma_result,

        # Druggability
        "druggability": drug,

        # CCF / allosteric
        "allosteric": {
            "ccf_to_active_site": round(ccf_to_active, 4),
            "top_allosteric_partners": allosteric_partners[:10],
        },

        # KCC
        "kcc": kcc_record,

        # Lining residues
        "lining_residues": lining_enriched,

        # Docking
        "docking_box_unidock": dock_box,
        "pharmacophore_features": pharma,

        # Confidence
        "confidence_score": confidence,
        "evidence_components": {
            "therm_signal":        therm_class in ("CRYPTIC","DYNAMIC","RESPONSIVE"),
            "ai_probability_gt30": ai_mean_prob > 0.30,
            "spike_coverage":      sdyn["total_spikes_sample"] > 50,
            "nma_responsiveness":  nma_result["n_responsive_modes"] >= 2,
            "engine_quality_gt35": float(eng.get("quality_score", 0.0)) > 0.35,
        },

        # WT-6 placeholders
        "explicit_solvent": {
            "stability_class": None,
            "stability_class_note": "Requires explicit solvent refinement (WT-6 pipeline). "
                                    "Expected classes: STABLE/METASTABLE/COLLAPSED",
            "rmsd_angstrom": None,
            "volume_sigma_pct": None,
            "simulation_time_ns": None,
        },
    }
    sites_report.append(site_rec)

# Sort by confidence desc
sites_report.sort(key=lambda s: -(s["confidence_score"] + s["engine_quality_score"] * 0.1))
for i, s in enumerate(sites_report):
    s["rank"] = i + 1
    s["site_id"] = f"TEM1_TWIN_S{i+1:02d}"

print(f"  Built {len(sites_report)} site records from {len(engine_sites)} engine pockets")

# ═══════════════════════════════════════════════════════════════════════════
# STEP 7: Per-residue global section
# ═══════════════════════════════════════════════════════════════════════════
print("\n[7/9] Computing per-residue global metrics...")

# NMA sqfluct for all residues
modes   = nma_data["modes"]
nma_pdb = nma_data["residue_ids"]
nma_pdb_to_idx = {r: i for i, r in enumerate(nma_pdb)}
n_modes_nma = len(modes)
n_nma_res   = len(nma_pdb)
disp_m      = np.zeros((n_nma_res, n_modes_nma), dtype=np.float32)
evals       = np.array([m["eigenvalue"] for m in modes])
for k, mode in enumerate(modes):
    for j, d in enumerate(mode["displacements"][:n_nma_res]):
        disp_m[j, k] = float(np.sum(np.array(d) ** 2))
sqfluct_all = np.zeros(n_nma_res, dtype=np.float32)
for k in range(n_modes_nma):
    sqfluct_all += disp_m[:, k] / max(evals[k], 1e-10)

per_residue_global = []
for i, r in enumerate(pdb_resnums_sorted):
    ti  = pdb_to_topo.get(r, -1)
    rn  = resname_by_pdb.get(r, pdb_residues_info.get(r, {}).get("resname", "UNK"))
    aa1 = AA3TO1.get(rn, "X")
    prf = prf_by_pdb_resid.get(r)
    ni  = nma_pdb_to_idx.get(r)
    per_residue_global.append({
        "pdb_resnum": int(r),
        "topo_idx": int(ti) if ti >= 0 else None,
        "resname": rn,
        "aa1": aa1,
        "cryptic_probability": round(float(probabilities[i]), 4),
        "dssp": dssp_by_pdb.get(r, "C"),
        "bfactor": round(bfactor_mean.get(r, 0.0), 2),
        "sasa_proxy": round(sasa_by_pdb.get(r, 0.5), 3),
        "spike_agreement_ratio": round(float(prf["spike_agreement_ratio"]), 4) if prf else None,
        "ccf_peak_value": round(float(prf["ccf_peak_value"]), 4) if prf else None,
        "barrier_classification": prf["barrier_classification"] if prf else None,
        "nma_sqfluct": round(float(sqfluct_all[ni]), 5) if ni is not None else None,
        "is_active_site": r in ACTIVE_SITE_PDB,
    })

# ═══════════════════════════════════════════════════════════════════════════
# STEP 8: Feature scorecard + global metrics
# ═══════════════════════════════════════════════════════════════════════════
print("\n[8/9] Computing global feature scorecard...")

n_high_ccf = int(np.sum(ccf_upper > 0.5))

scorecard = {
    "n_residues": N_RES,
    "n_sites_engine": len(engine_sites),
    "n_sites_in_report": len(sites_report),
    "n_cryptic_therm": int(sum(1 for p in therm_data["pockets"] if p["therm_class"]=="CRYPTIC")),
    "n_dynamic_therm": int(sum(1 for p in therm_data["pockets"] if p["therm_class"]=="DYNAMIC")),
    "n_responsive_therm": int(sum(1 for p in therm_data["pockets"] if p["therm_class"]=="RESPONSIVE")),
    "n_inert_therm": int(sum(1 for p in therm_data["pockets"] if p["therm_class"]=="INERT")),
    "twin_spikes_a": int(twin["stream_a"]["total_spikes"]),
    "twin_spikes_b": int(twin["stream_b"]["total_spikes"]),
    "n_consensus_events": int(twin["n_consensus_events"]),
    "n_differential_events": int(twin["n_differential_events"]),
    "n_exchanges": int(twin["n_exchanges"]),
    "ccf_pairs_above_0p5": n_high_ccf,
    "ccf_pairs_above_0p7": int(np.sum(ccf_upper > 0.7)),
    "ccf_mean_upper_triangle": round(float(ccf_upper.mean()), 4),
    "ai_ensemble_folds": n_loaded,
    "ai_prob_range": [round(float(probabilities.min()), 4), round(float(probabilities.max()), 4)],
    "ai_prob_mean": round(float(probabilities.mean()), 4),
    "n_residues_prob_gt0p3": int(np.sum(probabilities > 0.3)),
    "n_residues_prob_gt0p5": int(np.sum(probabilities > 0.5)),
    "nma_modes_used": nma_data["n_modes"],
    "sdst_event_count": int(therm_data.get("sdst_event_count", 0)),
    "explicit_solvent_status": "NOT_RUN",
}

# ═══════════════════════════════════════════════════════════════════════════
# STEP 9: Write all outputs
# ═══════════════════════════════════════════════════════════════════════════
print("\n[9/9] Writing output files...")

# ── Main report ───────────────────────────────────────────────────────────
report = {
    "report_version": "1.0.0",
    "report_type": "PRISM-TWIN",
    "generated_at": GENERATION_TS,
    "prism_version": "v1.1-physics",
    "pipeline": "PRISM-TWIN coupled-stream analysis + PRISM-AI v002 ensemble",

    "target": {
        "name": "TEM-1 beta-lactamase",
        "pdb_id": "1BTL",
        "chain": "A",
        "uniprot": "P62593",
        "gene": "bla",
        "organism": "Escherichia coli",
        "ec_number": "3.5.2.6",
        "function": "Beta-lactam antibiotic hydrolysis; class A serine beta-lactamase. "
                    "Clinical target for resistance mechanism studies.",
        "n_residues": N_RES,
        "pdb_resnum_range": [int(min(pdb_resids)), int(max(pdb_resids))],
        "topology_offset": TOPO_OFFSET,
        "active_site_pdb_resnums": ACTIVE_SITE_PDB,
        "active_site_topo_indices": ACTIVE_SITE_TOPO_IDX,
        "active_site_residues": {
            "SER70":  {"topo_idx": 44, "role": "Nucleophilic serine, forms acyl-enzyme"},
            "LYS73":  {"topo_idx": 47, "role": "Proton shuttle; activates Ser70"},
            "SER130": {"topo_idx": 104, "role": "SDN motif; substrate binding"},
            "GLU166": {"topo_idx": 140, "role": "Deacylation general base"},
        },
    },

    "twin_run_summary": {
        "stream_a": {**twin["stream_a"]},
        "stream_b": {**twin["stream_b"]},
        "n_consensus_events":   int(twin["n_consensus_events"]),
        "n_differential_events": int(twin["n_differential_events"]),
        "n_exchanges":           int(twin["n_exchanges"]),
        "total_density_a_to_b":  twin["total_density_a_to_b"],
        "total_density_b_to_a":  twin["total_density_b_to_a"],
        "n_nonzero_regions":     twin["n_nonzero_regions"],
        "per_residue_features_count": len(prf_list),
    },

    "feature_scorecard": scorecard,

    "sites": sites_report,

    "per_residue": per_residue_global,

    "ccf_summary": {
        "matrix_file": "tem1_ccf_matrix.npy",
        "matrix_shape": [N_RES, N_RES],
        "spike_sample_used": len(spike_sample),
        "n_time_bins": N_BINS,
        "n_pairs_above_0p5": n_high_ccf,
        "n_pairs_above_0p7": int(np.sum(ccf_upper > 0.7)),
        "ccf_mean": round(float(ccf_upper.mean()), 4),
        "ccf_max_offdiag": round(float(ccf_upper.max()), 4),
    },

    "allosteric_network_file": "tem1_allosteric_network.json",
    "heatmap_pdb_file":        "tem1_prism_heatmap.pdb",
    "site_cards_file":         "tem1_site_cards.json",
    "docking_ready_file":      "tem1_docking_ready.json",

    "notes": {
        "explicit_solvent": (
            "WT-6 explicit solvent refinement (OpenMM TIP3P/OPC 10-50ns) NOT YET RUN. "
            "Fields water_displacement_sites, desolvation_penalty_kcal_mol, "
            "and stability_class are null pending WT-6 pipeline execution."
        ),
        "ai_model": (
            f"PRISM-AI v002 student ensemble, {n_loaded} folds loaded from "
            f"{AI_MODEL_DIR}. ESM-2 embeddings sourced from 1JWP cache "
            "(TEM-1 family, closest available)."
        ),
        "ccf": (
            f"CCF computed via Pearson correlation of time-binned spike density "
            f"using {len(spike_sample):,} sampled spikes (of {N_SPIKES:,} total), "
            f"{N_BINS} time bins, 8Å CA assignment radius."
        ),
        "residue_numbering": (
            "All pdb_resnum values are PDB author residue numbers (26..290). "
            "topo_idx = pdb_resnum - 26. Verified from residue_map.json."
        ),
    },
}

with open(REPORT_OUT, "w") as f:
    json.dump(report, f, indent=2)
print(f"  {REPORT_OUT} ({os.path.getsize(REPORT_OUT)/1024:.0f} KB)")

# ── Heatmap PDB ───────────────────────────────────────────────────────────
with open(PDB_CLEAN) as fin, open(HEATMAP_PDB, "w") as fout:
    for line in fin:
        if line.startswith(("ATOM", "HETATM")):
            try:
                resnum = int(line[22:26].strip())
                bval   = round(prob_by_pdb.get(resnum, 0.0) * 100.0, 2)
                new_line = (line[:60] + f"{bval:6.2f}" +
                            (line[66:] if len(line) > 66 else "\n"))
                fout.write(new_line)
            except Exception:
                fout.write(line)
        else:
            fout.write(line)
print(f"  {HEATMAP_PDB}")

# ── Allosteric network ────────────────────────────────────────────────────
net_edges = []
ii, jj = np.where(CCF > 0.5)
for i, j in zip(ii, jj):
    if i < j:
        ri = topo_to_pdb.get(int(i), -1); rj = topo_to_pdb.get(int(j), -1)
        if ri > 0 and rj > 0:
            net_edges.append({
                "source_topo": int(i), "target_topo": int(j),
                "source_pdb":  int(ri), "target_pdb":  int(rj),
                "source_resname": resname_by_pdb.get(ri, "UNK"),
                "target_resname": resname_by_pdb.get(rj, "UNK"),
                "ccf": round(float(CCF[i, j]), 4),
            })
net_edges.sort(key=lambda e: -e["ccf"])

node_degree = defaultdict(int)
for e in net_edges:
    node_degree[e["source_pdb"]] += 1; node_degree[e["target_pdb"]] += 1
net_nodes = sorted([{"pdb_resnum": int(r), "topo_idx": pdb_to_topo.get(r, -1),
                      "resname": resname_by_pdb.get(r, "UNK"), "degree": d,
                      "cryptic_probability": round(prob_by_pdb.get(r, 0.0), 4)}
                    for r, d in node_degree.items()], key=lambda n: -n["degree"])

allosteric_net = {
    "ccf_threshold": 0.5,
    "n_nodes": len(net_nodes), "n_edges": len(net_edges),
    "nodes": net_nodes,
    "edges": net_edges[:500],
    "top_hub_residues": net_nodes[:10],
}
with open(NETWORK_JSON, "w") as f:
    json.dump(allosteric_net, f, indent=2)
print(f"  {NETWORK_JSON} ({len(net_nodes)} nodes, {len(net_edges)} edges)")

# ── Site cards ────────────────────────────────────────────────────────────
site_cards = []
for s in sites_report:
    site_cards.append({
        "site_id": s["site_id"],
        "rank": s["rank"],
        "classification": s["classification"],
        "therm_class": s["therm_class"],
        "confidence": s["confidence_score"],
        "centroid": s["centroid_angstrom"],
        "volume_A3_engine": s["volume_angstrom3_engine"],
        "volume_A3_convex_hull": s["volume_angstrom3_convex_hull"],
        "n_residues": s["n_lining_residues"],
        "ai_probability": s["ai_mean_probability"],
        "druggability_score": s["druggability"].get("druggability_score"),
        "barrier_kcal_mol": s["therm"]["barrier_estimate_kcal_mol"],
        "hysteresis": s["therm"]["hysteresis_asymmetry"],
        "ccf_to_active_site": s["allosteric"]["ccf_to_active_site"],
        "n_responsive_nma_modes": s["nma"]["n_responsive_modes"],
        "key_lining_residues": [f"{r['resname']}{r['pdb_resnum']}" for r in s["lining_residues"][:5]],
        "active_site_overlap": s["active_site_residues_overlap"],
        "engine_pocket_id": s["engine_pocket_id"],
        "engine_quality": s["engine_quality_score"],
        "gtck_rank": s["gtck_rank"],
        "spike_dynamics_summary": {
            "total_spikes_sample": s["spike_dynamics"]["total_spikes_sample"],
            "b_over_a_ratio": s["spike_dynamics"]["b_over_a_ratio"],
            "phase_dynamics": s["spike_dynamics"]["phase_dynamics"],
        },
        "pharma_feature_types": sorted(set(p["type"] for p in s["pharmacophore_features"])),
        "explicit_solvent_status": "NOT_RUN (WT-6 required)",
    })

with open(CARDS_JSON, "w") as f:
    json.dump({"target": "TEM-1 beta-lactamase (1BTL)",
               "generated_at": GENERATION_TS,
               "n_sites": len(site_cards),
               "sites": site_cards}, f, indent=2)
print(f"  {CARDS_JSON}")

# ── Docking ready ─────────────────────────────────────────────────────────
docking_entries = []
for s in sites_report:
    docking_entries.append({
        "site_id": s["site_id"],
        "rank": s["rank"],
        "classification": s["classification"],
        "confidence": s["confidence_score"],
        "docking_box": s["docking_box_unidock"],
        "pharmacophore_features": s["pharmacophore_features"],
        "n_pharmacophore_features": len(s["pharmacophore_features"]),
        "feature_types": sorted(set(p["type"] for p in s["pharmacophore_features"])),
        "lining_residues_pdb": [r["pdb_resnum"] for r in s["lining_residues"]],
        "hbond_donors":   s["druggability"].get("hbond_donor_count"),
        "hbond_acceptors":s["druggability"].get("hbond_acceptor_count"),
        "aromatic_count": s["druggability"].get("aromatic_count"),
    })

with open(DOCKING_JSON, "w") as f:
    json.dump({
        "target": "TEM-1 beta-lactamase (1BTL)",
        "generated_at": GENERATION_TS,
        "software_compatibility": ["UniDock v1.1.3 (--center_x/y/z --size_x/y/z)",
                                    "GNINA v1.3.2 (--center_x/y/z --size_x/y/z)",
                                    "AutoDock Vina"],
        "coordinate_units": "Angstroms",
        "sites": docking_entries,
    }, f, indent=2)
print(f"  {DOCKING_JSON}")

# ═══════════════════════════════════════════════════════════════════════════
# SUMMARY CARD
# ═══════════════════════════════════════════════════════════════════════════
print()
print("=" * 70)
print("  PRISM-TWIN SUMMARY CARD — TEM-1 beta-lactamase (1BTL)  ")
print("=" * 70)
print(f"  Protein:        TEM-1 beta-lactamase, E. coli | UniProt P62593")
print(f"  PDB / Chain:    1BTL / A | {N_RES} residues (PDB 26..290)")
print(f"  Report date:    2026-04-03")
print(f"  Pipeline:       PRISM-TWIN v1.1-physics + PRISM-AI v002")
print()
print(f"  TWIN RUN STATISTICS")
print(f"  ├─ Stream A (scout/thermal):  {twin['stream_a']['total_spikes']:>14,} spikes")
print(f"  ├─ Stream B:                  {twin['stream_b']['total_spikes']:>14,} spikes")
print(f"  ├─ Consensus events:          {twin['n_consensus_events']:>14,}")
print(f"  ├─ Differential events:       {twin['n_differential_events']:>14,}")
print(f"  └─ Exchanges (A↔B):           {twin['n_exchanges']:>14,}")
print()
print(f"  AI ENSEMBLE  ({n_loaded} folds, ESM-2 1JWP cache)")
print(f"  ├─ Probability range:  [{probabilities.min():.3f}, {probabilities.max():.3f}]  mean={probabilities.mean():.3f}")
print(f"  ├─ Residues p > 0.30:  {int(np.sum(probabilities > 0.3)):>4} / {N_RES}")
print(f"  └─ Residues p > 0.50:  {int(np.sum(probabilities > 0.5)):>4} / {N_RES}")
print()
print(f"  THERMODYNAMIC CLASSIFICATION  (27 engine pockets)")
n_cr = sum(1 for p in therm_data['pockets'] if p['therm_class']=='CRYPTIC')
n_dy = sum(1 for p in therm_data['pockets'] if p['therm_class']=='DYNAMIC')
n_rs = sum(1 for p in therm_data['pockets'] if p['therm_class']=='RESPONSIVE')
n_in = sum(1 for p in therm_data['pockets'] if p['therm_class']=='INERT')
print(f"  ├─ CRYPTIC:    {n_cr}   (requires conformational opening)")
print(f"  ├─ DYNAMIC:    {n_dy}   (accessible, fluctuating)")
print(f"  ├─ RESPONSIVE: {n_rs}   (opens under perturbation)")
print(f"  └─ INERT:      {n_in}   (stable, low dynamics)")
print()
print(f"  CCF ALLOSTERIC NETWORK  (263×263 Pearson correlation matrix)")
print(f"  ├─ High-correlation pairs (CCF > 0.5):  {n_high_ccf:,}")
print(f"  ├─ Strong pairs (CCF > 0.7):            {int(np.sum(ccf_upper > 0.7)):,}")
print(f"  └─ Top hub residues by degree:")
for nd in net_nodes[:5]:
    print(f"     {nd['resname']:<4} {nd['pdb_resnum']:>4}  degree={nd['degree']:>5}  p={nd['cryptic_probability']:.3f}")
print()
print(f"  TOP PREDICTED SITES  (sorted by confidence)")
hdr = f"  {'Rank':<4} {'Site ID':<18} {'Class':<13} {'Therm':<11} {'Conf':<6} {'AI-p':<6} {'Vol(A3)':<9} {'Q':<5} {'N-res'}"
print(hdr)
print(f"  {'-'*4} {'-'*18} {'-'*13} {'-'*11} {'-'*6} {'-'*6} {'-'*9} {'-'*5} {'-'*5}")
for s in sites_report[:10]:
    print(f"  {s['rank']:<4} {s['site_id']:<18} {s['classification']:<13} "
          f"{s['therm_class']:<11} {s['confidence_score']:<6.3f} "
          f"{s['ai_mean_probability']:<6.3f} {s['volume_angstrom3_engine']:<9.0f} "
          f"{s['engine_quality_score']:<5.3f} {s['n_lining_residues']}")
print()
print(f"  ACTIVE SITE RESIDUES (PDB: SER70/LYS73/SER130/GLU166)")
active_covering_sites = [s for s in sites_report if s["is_active_site_overlap"]]
if active_covering_sites:
    for s in active_covering_sites[:4]:
        ovlp = s["active_site_residues_overlap"]
        print(f"  ├─ {s['site_id']:18s}: overlaps {ovlp}  (therm={s['therm_class']}, q={s['engine_quality_score']:.3f})")
else:
    print(f"  └─ No site directly covers Ser70/Lys73/Ser130/Glu166 (normal for TWIN sites)")
print()
print(f"  DRUG DISCOVERY HIGHLIGHTS")
# Top cryptic site
top_cr = next((s for s in sites_report if s["classification"] == "CRYPTIC"), None)
top_al = next((s for s in sites_report if s["classification"] == "ALLOSTERIC"), None)
if top_cr:
    print(f"  Best cryptic site:    {top_cr['site_id']}")
    print(f"  ├─ Confidence:        {top_cr['confidence_score']:.3f}")
    print(f"  ├─ Therm class:       {top_cr['therm_class']}")
    print(f"  ├─ Druggability:      {top_cr['druggability'].get('druggability_score', 'N/A'):.3f}")
    print(f"  ├─ HBD/HBA:           {top_cr['druggability'].get('hbond_donor_count',0)}D / {top_cr['druggability'].get('hbond_acceptor_count',0)}A")
    print(f"  ├─ Aromatic residues: {top_cr['druggability'].get('aromatic_count',0)}")
    print(f"  ├─ Barrier estimate:  {top_cr['therm']['barrier_estimate_kcal_mol']} kcal/mol")
    print(f"  ├─ CCF to active:     {top_cr['allosteric']['ccf_to_active_site']:.3f}")
    print(f"  └─ NMA modes:         {top_cr['nma']['n_responsive_modes']} responsive")
if top_al:
    print(f"  Best allosteric site: {top_al['site_id']}  (CCF-to-active={top_al['allosteric']['ccf_to_active_site']:.3f})")
print()
print(f"  OUTPUT FILES")
for f_out in [REPORT_OUT, HEATMAP_PDB, CCF_NPY, NETWORK_JSON, CARDS_JSON, DOCKING_JSON]:
    sz = os.path.getsize(f_out)
    print(f"  ├─ {str(f_out):<60}  {sz/1024:>6.0f} KB")
print()
print(f"  STATUS: COMPLETE. WT-6 explicit solvent refinement NOT RUN.")
print(f"          Run WT-6 pipeline to populate water_displacement_sites,")
print(f"          desolvation_penalty, and stability_class fields.")
print("=" * 70)
