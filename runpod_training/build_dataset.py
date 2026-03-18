#!/usr/bin/env python3
"""Build training dataset with checkpointing. Saves every 200 targets."""
import json, os, sys, torch, numpy as np
sys.path.insert(0, '/workspace/prism4d/runpod_training')
from egnn_pocket_ranker_v4 import parse_pdb_full, build_pocket_graph, ESMEmbedder

CKPT = '/workspace/dataset_checkpoint.pt'
OUT = '/workspace/training_dataset.pt'

m = json.load(open('/workspace/scpdb_data/manifest_cached.json'))
gt = json.load(open('/workspace/scpdb_data/scpdb_ground_truth.json'))

if os.path.exists(CKPT):
    print(f'Resuming from checkpoint...')
    ck = torch.load(CKPT, weights_only=False)
    G, L, M, done = ck['graphs'], ck['labels'], ck['meta'], ck['done']
    print(f'  {len(G)} graphs, {len(done)} targets done')
else:
    G, L, M, done = [], [], [], set()

e = ESMEmbedder('esm2_t33_650M_UR50D', cache_dir='/workspace/models/esm_cache')
total = len(m['targets'])

for i, t in enumerate(m['targets']):
    tid = str(t['id'])
    if tid in done:
        continue
    apo = t['apo_pdb'].lower()
    if tid not in gt:
        done.add(tid)
        continue
    p = '/workspace/scpdb_data/pdbs/' + apo + '.pdb'
    s = '/workspace/scpdb_data/sites/' + tid + '/' + apo + '.binding_sites.json'
    if not os.path.exists(p) or not os.path.exists(s):
        done.add(tid)
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
    except Exception as ex:
        print(f'  SKIP {apo}: {ex}')
    done.add(tid)

    if len(done) % 200 == 0:
        print(f'  [{len(done)}/{total}] {len(G)} graphs, {int(sum(L))} hits — saving checkpoint')
        torch.save({'graphs': G, 'labels': L, 'meta': M, 'done': done}, CKPT)

print(f'\nDone: {len(G)} graphs, {int(sum(L))} hits')
torch.save({'graphs': G, 'labels': L, 'meta': M}, OUT)
print(f'Saved {OUT}')
if os.path.exists(CKPT):
    os.remove(CKPT)
    print('Checkpoint cleaned up')
