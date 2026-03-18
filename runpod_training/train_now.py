#!/usr/bin/env python3
"""
Zero-arg training script. Just: python3 train_now.py
Trains GVP-GNN ensemble on scPDB data, exports to ONNX.
"""
import json, os, sys, warnings, traceback
import numpy as np
import torch
import torch.onnx

warnings.filterwarnings("ignore")

# ── ALL PATHS HARDCODED ──────────────────────────────────────────────────
MANIFEST = "/workspace/scpdb_data/manifest_cached.json"
GT = "/workspace/scpdb_data/scpdb_ground_truth.json"
SITES_DIR = "/workspace/scpdb_data/sites"
APO_DIR = "/workspace/scpdb_data/pdbs"
MODEL_DIR = "/workspace/pretrained_models"
ESM_CACHE = "/workspace/esm_cache"
ESM_MODEL = "esm2_t33_650M_UR50D"
EPOCHS = 100
DCC_THRESH = 4.0
N_MODELS = 5
LR = 3e-4

os.makedirs(MODEL_DIR, exist_ok=True)

# ── Import from v4 ───────────────────────────────────────────────────────
sys.path.insert(0, "/workspace/prism4d/runpod_training")
from egnn_pocket_ranker_v4 import (
    parse_pdb_full, build_pocket_graph, ESMEmbedder,
    train_ensemble, PrismPocketRankerV4, NODE_SCALAR_EXTRA,
)

# ── Monkey-patch torch.load to auto-delete corrupt cache files ───────────
_orig_load = torch.load
def _safe_load(*a, **k):
    try:
        return _orig_load(*a, **k)
    except Exception:
        path = str(a[0]) if a else ""
        if path.endswith(".pt") and os.path.exists(path):
            sz = os.path.getsize(path)
            print(f"  CORRUPT: {path} ({sz}B) — deleting")
            os.remove(path)
        raise
torch.load = _safe_load

# ── Build training data ──────────────────────────────────────────────────
print("Loading manifest + ground truth...")
manifest = json.load(open(MANIFEST))
gt = json.load(open(GT))
print(f"  {len(manifest['targets'])} targets, {len(gt)} ground truths")

print(f"Loading ESM-2 ({ESM_MODEL})...")
embedder = ESMEmbedder(ESM_MODEL, cache_dir=ESM_CACHE)

graphs, labels, meta = [], [], []
n_targets = len(manifest["targets"])
n_skip = 0
n_esm_fail = 0

for i, t in enumerate(manifest["targets"]):
    tid = str(t["id"])
    apo = t["apo_pdb"].lower()
    if tid not in gt:
        n_skip += 1
        continue

    pdb = f"{APO_DIR}/{apo}.pdb"
    sites = f"{SITES_DIR}/{tid}/{apo}.binding_sites.json"
    if not os.path.exists(pdb) or not os.path.exists(sites):
        n_skip += 1
        continue

    try:
        residues, seq = parse_pdb_full(pdb)
        if len(residues) < 10:
            n_skip += 1
            continue
    except Exception:
        n_skip += 1
        continue

    # ESM embedding with retry on corrupt cache
    esm_emb = None
    for attempt in range(2):
        try:
            esm_emb = embedder.embed(apo, seq)
            break
        except Exception as e:
            if attempt == 0:
                # Corrupt cache — already deleted by monkey-patch, retry
                continue
            else:
                n_esm_fail += 1
                break

    if esm_emb is None:
        continue

    true_c = np.array(gt[tid]["centroid"])
    try:
        site_data = json.load(open(sites))
    except Exception:
        continue

    for s in site_data.get("sites", []):
        c = s.get("centroid")
        if not c:
            continue
        try:
            g = build_pocket_graph(residues, esm_emb, s)
            if g is None:
                continue
            dcc = float(np.linalg.norm(np.array(c) - true_c))
            g.y = torch.tensor([1.0 if dcc <= DCC_THRESH else 0.0])
            graphs.append(g)
            labels.append(g.y.item())
            meta.append({"target": apo.upper(), "site_id": s.get("id", "?"), "dcc": round(dcc, 2)})
        except Exception:
            continue

    if (i + 1) % 200 == 0:
        print(f"  [{i+1}/{n_targets}] {len(graphs)} graphs, {int(sum(labels))} hits, {n_skip} skipped, {n_esm_fail} ESM fails")

print(f"\nDataset: {len(graphs)} graphs, {int(sum(labels))} hits, {len(graphs)-int(sum(labels))} decoys")
print(f"  Skipped: {n_skip}, ESM failures: {n_esm_fail}")

if not graphs:
    print("ERROR: No training data built. Exiting.")
    sys.exit(1)

# ── Train ensemble ───────────────────────────────────────────────────────
train_ensemble(graphs, labels, meta, n_models=N_MODELS, epochs=EPOCHS, lr=LR, model_dir=MODEL_DIR)

# ── Export best model to ONNX ────────────────────────────────────────────
print("\n" + "=" * 65)
print("EXPORTING TO ONNX")
print("=" * 65)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model 0 as the representative
ckpt_path = f"{MODEL_DIR}/egnn_ranker_v4_m0.pt"
if os.path.exists(ckpt_path):
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    config = ckpt["config"]
    model = PrismPocketRankerV4(**config)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    # Create dummy input matching graph structure
    N = 20  # dummy 20 residues
    E = 60  # dummy 60 edges
    esm_dim = config["esm_dim"]
    aux_dim = config["aux_dim"]

    from torch_geometric.data import Data, Batch

    dummy = Data(
        x=torch.randn(N, esm_dim + NODE_SCALAR_EXTRA),
        node_v=torch.randn(N, 2, 3),
        pos=torch.randn(N, 3),
        pos_rel=torch.randn(N, 3),
        edge_index=torch.randint(0, N, (2, E)),
        edge_seq_sep=torch.randn(E, 1),
        aux=torch.randn(1, aux_dim),
        batch=torch.zeros(N, dtype=torch.long),
    )

    # Can't directly ONNX export PyG models with dynamic graphs.
    # Instead, save as TorchScript + ONNX-compatible state dict.
    onnx_path = f"{MODEL_DIR}/gnn_pocket_ranker_v4.onnx"

    try:
        # Try tracing
        class OnnxWrapper(torch.nn.Module):
            """Flatten PyG Data into plain tensors for ONNX export."""
            def __init__(self, model):
                super().__init__()
                self.model = model

            def forward(self, x, node_v, pos, pos_rel, edge_index, edge_seq_sep, aux, batch):
                data = Data(
                    x=x, node_v=node_v, pos=pos, pos_rel=pos_rel,
                    edge_index=edge_index, edge_seq_sep=edge_seq_sep,
                    aux=aux, batch=batch,
                )
                return torch.sigmoid(self.model(data))

        wrapper = OnnxWrapper(model)
        wrapper.eval()

        torch.onnx.export(
            wrapper,
            (dummy.x, dummy.node_v, dummy.pos, dummy.pos_rel,
             dummy.edge_index, dummy.edge_seq_sep, dummy.aux, dummy.batch),
            onnx_path,
            input_names=["x", "node_v", "pos", "pos_rel", "edge_index", "edge_seq_sep", "aux", "batch"],
            output_names=["score"],
            dynamic_axes={
                "x": {0: "num_nodes"},
                "node_v": {0: "num_nodes"},
                "pos": {0: "num_nodes"},
                "pos_rel": {0: "num_nodes"},
                "edge_index": {1: "num_edges"},
                "edge_seq_sep": {0: "num_edges"},
                "batch": {0: "num_nodes"},
            },
            opset_version=17,
        )
        print(f"ONNX exported: {onnx_path} ({os.path.getsize(onnx_path)/1024:.0f} KB)")
    except Exception as e:
        print(f"ONNX trace export failed: {e}")
        print("Saving TorchScript instead...")
        ts_path = f"{MODEL_DIR}/gnn_pocket_ranker_v4.pt"
        torch.save({
            "model_state": ckpt["model_state"],
            "config": config,
            "format": "state_dict",
        }, ts_path)
        print(f"TorchScript state dict saved: {ts_path}")

    # Also save all 5 ensemble models as a single bundle
    bundle = {"config": config, "models": []}
    for i in range(N_MODELS):
        p = f"{MODEL_DIR}/egnn_ranker_v4_m{i}.pt"
        if os.path.exists(p):
            c = torch.load(p, map_location="cpu", weights_only=False)
            bundle["models"].append(c["model_state"])
    bundle_path = f"{MODEL_DIR}/gnn_ensemble_v4_bundle.pt"
    torch.save(bundle, bundle_path)
    print(f"Ensemble bundle: {bundle_path} ({os.path.getsize(bundle_path)/1024/1024:.1f} MB)")
else:
    print(f"ERROR: No checkpoint at {ckpt_path}")

print("\nDONE.")
