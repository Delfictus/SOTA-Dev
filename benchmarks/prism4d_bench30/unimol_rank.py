#!/usr/bin/env python3
"""
Rank BENCH30/60 pockets using Uni-Mol pocket embeddings.
Extracts pocket atoms from APO PDB within lining residue cutoff,
gets Uni-Mol 512-dim representation, trains a simple classifier
on DCC labels, evaluates SR@1.
"""
import json, os, csv, sys
import numpy as np

BENCH_DIR = os.path.dirname(os.path.abspath(__file__))
MANIFEST = os.path.join(BENCH_DIR, "benchmark_manifest.json")
GT = os.path.join(BENCH_DIR, "ground_truth", "ligand_centroids.json")
RESULTS_DIR = os.path.join(BENCH_DIR, "results")
APO_DIR = os.path.join(BENCH_DIR, "apo")
DCC_THRESH = 8.0


def parse_pdb_atoms(pdb_path):
    """Parse PDB, return list of (resid, resname, chain, atom_name, element, xyz)."""
    atoms = []
    with open(pdb_path) as f:
        for line in f:
            if line.startswith(("ATOM  ", "HETATM")):
                atom_name = line[12:16].strip()
                resname = line[17:20].strip()
                chain = line[21]
                try:
                    resid = int(line[22:26])
                    x = float(line[30:38])
                    y = float(line[38:46])
                    z = float(line[46:54])
                except ValueError:
                    continue
                element = line[76:78].strip() if len(line) > 77 else atom_name[0]
                if element == "H":
                    continue
                atoms.append((resid, resname, chain, atom_name, element, np.array([x, y, z])))
    return atoms


def extract_pocket_atoms(pdb_atoms, site, radius=10.0):
    """Extract atoms within radius of pocket centroid."""
    centroid = np.array(site["centroid"])
    # Use lining residues if available, else use distance cutoff
    lining_resids = set()
    for lr in site.get("lining_residues", []):
        lining_resids.add((lr.get("resid"), lr.get("chain", "A")))

    pocket_atoms = []
    pocket_coords = []
    if lining_resids:
        for resid, resname, chain, atom_name, element, xyz in pdb_atoms:
            if (resid, chain) in lining_resids:
                pocket_atoms.append(element)
                pocket_coords.append(xyz)
    else:
        for resid, resname, chain, atom_name, element, xyz in pdb_atoms:
            if np.linalg.norm(xyz - centroid) <= radius:
                pocket_atoms.append(element)
                pocket_coords.append(xyz)

    if len(pocket_atoms) < 3:
        return None, None
    return pocket_atoms, np.array(pocket_coords, dtype=np.float32)


def main():
    from unimol_tools import UniMolRepr

    manifest = json.load(open(MANIFEST))
    gt = json.load(open(GT))

    print("Loading Uni-Mol...")
    clf = UniMolRepr(data_type='molecule', remove_hs=True)

    # Collect all pockets
    all_pockets = []  # (target_id, pdb, site_id, centroid, dcc, label, features_from_json)

    for target in manifest["targets"]:
        tid = str(target["id"])
        apo = target["apo_pdb"].lower()
        if tid not in gt:
            continue

        sites_path = os.path.join(RESULTS_DIR, tid, f"{apo}.binding_sites.json")
        pdb_path = os.path.join(APO_DIR, f"{apo}.pdb")
        if not os.path.exists(sites_path) or not os.path.exists(pdb_path):
            continue

        true_c = np.array(gt[tid]["centroid"])
        pdb_atoms = parse_pdb_atoms(pdb_path)
        sites_data = json.load(open(sites_path))

        for site in sites_data.get("sites", []):
            c = site.get("centroid")
            if not c:
                continue
            dcc = float(np.linalg.norm(np.array(c) - true_c))
            atoms, coords = extract_pocket_atoms(pdb_atoms, site)
            if atoms is None:
                continue

            all_pockets.append({
                "target": apo.upper(),
                "tid": tid,
                "site_id": site.get("id", "?"),
                "dcc": dcc,
                "label": 1.0 if dcc <= DCC_THRESH else 0.0,
                "atoms": atoms,
                "coords": coords,
                "quality": site.get("quality_score", 0),
                "burial": site.get("burial_score", 0),
                "volume": site.get("volume", 0),
                "druggability": site.get("druggability", 0),
            })

    print(f"Collected {len(all_pockets)} pockets from "
          f"{len(set(p['target'] for p in all_pockets))} targets")
    n_hits = sum(1 for p in all_pockets if p['label'] == 1.0)
    print(f"  Hits @{DCC_THRESH}A: {n_hits}")

    # Get Uni-Mol representations in batches
    print("\nGetting Uni-Mol representations...")
    BATCH = 64
    all_cls = []
    for i in range(0, len(all_pockets), BATCH):
        batch = all_pockets[i:i+BATCH]
        data = {
            'atoms': [p['atoms'] for p in batch],
            'coordinates': [p['coords'] for p in batch],
        }
        reprs = clf.get_repr(data, return_atomic_reprs=False)
        all_cls.extend(reprs)
        if (i + BATCH) % 256 == 0 or i + BATCH >= len(all_pockets):
            print(f"  [{min(i+BATCH, len(all_pockets))}/{len(all_pockets)}]")

    X = np.array(all_cls)  # (N, 512)
    y = np.array([p['label'] for p in all_pockets])
    print(f"Embeddings: {X.shape}, hits: {int(y.sum())}")

    # Add handcrafted features
    extra = np.array([[p['quality'], p['burial'], p['volume'], p['druggability']]
                      for p in all_pockets])
    X_full = np.hstack([X, extra])

    # LOTO evaluation with logistic regression
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    targets = sorted(set(p['target'] for p in all_pockets))
    target_arr = np.array([p['target'] for p in all_pockets])

    sr1_hits = 0
    sr1_total = 0
    results = []

    print(f"\nLOTO Cross-Validation ({len(targets)} targets)")
    print(f"{'Target':<10} {'Top Site':<10} {'DCC':<8} {'Score':<8} {'Hit'}")
    print("-" * 50)

    for held_out in targets:
        train_mask = target_arr != held_out
        test_mask = target_arr == held_out

        if train_mask.sum() == 0 or test_mask.sum() == 0:
            continue
        if y[train_mask].sum() == 0:
            continue

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_full[train_mask])
        X_test = scaler.transform(X_full[test_mask])

        model = LogisticRegression(max_iter=1000, class_weight='balanced', C=0.1)
        model.fit(X_train, y[train_mask])

        scores = model.predict_proba(X_test)[:, 1]
        test_indices = np.where(test_mask)[0]

        pocket_scores = []
        for j, idx in enumerate(test_indices):
            p = all_pockets[idx]
            pocket_scores.append((p['site_id'], p['dcc'], scores[j], p['label']))

        pocket_scores.sort(key=lambda x: -x[2])
        top = pocket_scores[0]
        is_hit = top[3] == 1.0
        if is_hit:
            sr1_hits += 1
        sr1_total += 1

        marker = " <<<" if is_hit else ""
        print(f"{held_out:<10} {top[0]:<10} {top[1]:<8.2f} {top[2]:<8.4f} "
              f"{'HIT' if is_hit else 'MISS'}{marker}")

        results.append({
            'target': held_out,
            'top_site': top[0],
            'top_dcc': round(top[1], 2),
            'top_score': round(top[2], 4),
            'hit': is_hit,
        })

    print(f"\n{'='*50}")
    print(f"SR@1: {sr1_hits}/{sr1_total} ({100*sr1_hits/max(sr1_total,1):.0f}%)")

    # Save results
    out_path = os.path.join(BENCH_DIR, "unimol_ranking_results.json")
    json.dump({
        "sr1": sr1_hits,
        "n_targets": sr1_total,
        "sr1_pct": round(100 * sr1_hits / max(sr1_total, 1), 1),
        "dcc_threshold": DCC_THRESH,
        "n_pockets": len(all_pockets),
        "n_hits": int(n_hits),
        "results": results,
    }, open(out_path, "w"), indent=2)
    print(f"Saved {out_path}")

    # Also save embeddings for fine-tuning
    emb_path = os.path.join(BENCH_DIR, "unimol_embeddings.npz")
    np.savez(emb_path,
             cls_repr=X,
             labels=y,
             targets=target_arr,
             site_ids=np.array([p['site_id'] for p in all_pockets]),
             dccs=np.array([p['dcc'] for p in all_pockets]))
    print(f"Saved embeddings: {emb_path}")


if __name__ == "__main__":
    main()
