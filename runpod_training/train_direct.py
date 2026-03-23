#!/usr/bin/env python3
"""Build graphs in memory, train immediately. No dataset file needed."""
import json, os, sys, torch, numpy as np
sys.path.insert(0, '/workspace/prism4d/runpod_training')
from egnn_pocket_ranker_v4 import (
    parse_pdb_full, build_pocket_graph, ESMEmbedder, train_ensemble
)

m = json.load(open('/workspace/scpdb_data/manifest_cached.json'))
gt = json.load(open('/workspace/scpdb_data/scpdb_ground_truth.json'))

e = ESMEmbedder('esm2_t33_650M_UR50D', cache_dir='/workspace/models/esm_cache')
total = len(m['targets'])

G, L, M = [], [], []
for i, t in enumerate(m['targets']):
    tid = str(t['id'])
    apo = t['apo_pdb'].lower()
    if tid not in gt:
        continue
    p = '/workspace/scpdb_data/pdbs/' + apo + '.pdb'
    s = '/workspace/scpdb_data/sites/' + tid + '/' + apo + '.binding_sites.json'
    if not os.path.exists(p) or not os.path.exists(s):
        continue
    try:
        r, sq = parse_pdb_full(p)
        emb = e.embed(apo, sq)
        tc = np.array(gt[tid]['centroid'])
        for site in json.load(open(s)).get('sites', []):
            c = site.get('centroid')
            if not c:
                continue
            g = build_pocket_graph(r, emb, site)
            if g is None:
                continue
            dcc = float(np.linalg.norm(np.array(c) - tc))
            g.y = torch.tensor([1.0 if dcc <= 4.0 else 0.0])
            G.append(g)
            L.append(g.y.item())
            M.append({'target': apo.upper(), 'site_id': site.get('id', '?'), 'dcc': round(dcc, 2)})
    except:
        continue
    if (i + 1) % 500 == 0:
        print(f'  [{i+1}/{total}] {len(G)} graphs, {int(sum(L))} hits')

print(f'\nDataset: {len(G)} graphs, {int(sum(L))} hits')
print('Starting training...\n')
train_ensemble(G, L, M, n_models=5, epochs=100, lr=3e-4, model_dir='/workspace/models')
