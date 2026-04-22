#!/usr/bin/env python3
"""Process CryptoBank pickle into 500 engine-ready true apo-holo pairs.

CryptoBank schema (5.49M rows, 121 columns):
  Holo: rcsb_id, auth_asym_id, resolution, chain_id
  Apo:  rcsb_id_apo, auth_asym_id_apo, resolution_apo, chain_id_apo
  Ligand: ligandId, unique_ligandId, lig_atoms, lig_mw, site_centroid
  Quality: rmsd_local (pocket RMSD), crypticity (bool), cluster_id_30_apo
  UniProt: uniprot_accession_apo

Each row = one apo-holo-ligand-site combination.

Usage:
    python3 scripts/quarantine/process_cryptobank.py [--target-count 500]
"""

import argparse
import json
import os
import pickle
import re
import subprocess
import sys
import urllib.error
import urllib.request
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parent.parent.parent
BENCH_DIR = BASE / "benchmarks" / "true_apo"
DATA_DIR = BENCH_DIR / "data"
GT_DIR = BENCH_DIR / "ground_truth"
APO_DIR = BENCH_DIR / "apo_pdbs"
HOLO_DIR = BENCH_DIR / "holo_pdbs"
PICKLE_PATH = DATA_DIR / "cryptobank_dataset_12_03_2025.pkl"

NUISANCE = {
    "HOH", "DOD", "GOL", "EDO", "PEG", "PGE", "DMS", "MPD",
    "MES", "EPE", "TRS", "ACT", "FMT", "BME", "IPA",
    "SO4", "PO4", "CL", "NA", "K", "CA", "MG", "ZN",
    "NO3", "SCN", "BR", "IOD", "FE", "MN", "CO", "NI", "CU",
    "CD", "HG", "CIT", "TAR", "SUC", "MLI",
    "BOG", "LDA", "SDS", "UNX", "UNL",
    "HOH", "H2O", "WAT", "TIP", "NAG", "BMA", "MAN", "GLC", "FUC",
}


def fetch_pdb(pdb_id, dest_dir):
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    path = dest_dir / f"{pdb_id.lower()}.pdb"
    if path.exists() and path.stat().st_size > 100:
        return path
    url = f"https://files.rcsb.org/download/{pdb_id.upper()}.pdb"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "prism4d/1.0"})
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = resp.read()
        if len(data) > 100:
            path.write_bytes(data)
            return path
    except (urllib.error.HTTPError, urllib.error.URLError, OSError, TimeoutError):
        pass
    return None


def get_ligand_atoms(holo_path, ligand_code, holo_chain):
    atoms = []
    try:
        with open(holo_path, errors="ignore") as f:
            for line in f:
                if not line.startswith("HETATM"):
                    continue
                rn = line[17:20].strip()
                ch = line[21]
                if rn == ligand_code and ch == holo_chain:
                    try:
                        atoms.append([float(line[30:38]), float(line[38:46]), float(line[46:54])])
                    except (ValueError, IndexError):
                        continue
    except OSError:
        pass
    return atoms


def get_binding_residues(holo_path, lig_atoms, chain, cutoff=4.5):
    if not lig_atoms:
        return []
    lig_np = np.array(lig_atoms)
    residues = set()
    try:
        with open(holo_path, errors="ignore") as f:
            for line in f:
                if not line.startswith("ATOM"):
                    continue
                if line[21] != chain:
                    continue
                try:
                    pos = np.array([float(line[30:38]), float(line[38:46]), float(line[46:54])])
                    resnum = int(line[22:26])
                    resname = line[17:20].strip()
                except (ValueError, IndexError):
                    continue
                if np.linalg.norm(lig_np - pos, axis=1).min() <= cutoff:
                    residues.add((resnum, resname))
    except OSError:
        pass
    return sorted(residues)


def has_nucleic(pdb_path):
    NUC = {"DA", "DC", "DG", "DT", "DU", "A", "C", "G", "U", "T"}
    try:
        with open(pdb_path, errors="ignore") as f:
            for line in f:
                if line.startswith(("ATOM", "HETATM")):
                    if line[17:20].strip() in NUC:
                        return True
    except OSError:
        pass
    return False


def is_xray(pdb_path):
    try:
        with open(pdb_path, errors="ignore") as f:
            for line in f:
                if line.startswith("EXPDTA"):
                    return "X-RAY" in line.upper()
                if line.startswith("ATOM"):
                    break
    except OSError:
        pass
    return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target-count", type=int, default=500)
    parser.add_argument("--skip-download", action="store_true")
    args = parser.parse_args()

    print(f"Loading CryptoBank pickle...")
    df = pd.read_pickle(PICKLE_PATH)
    print(f"  Loaded: {len(df)} rows, {len(df.columns)} columns")

    # ── Phase 1: DataFrame-level filtering (fast) ──
    funnel = [("Raw rows", len(df))]

    # Filter: ligand not nuisance
    df = df[~df['ligandId'].isin(NUISANCE)]
    funnel.append(("Non-nuisance ligand", len(df)))

    # Filter: ligand heavy atoms >= 10
    if 'lig_atoms' in df.columns:
        df = df[pd.to_numeric(df['lig_atoms'], errors='coerce').fillna(0) >= 10]
        funnel.append(("Ligand ≥10 heavy atoms", len(df)))

    # Filter: apo resolution <= 2.0
    df = df[pd.to_numeric(df['resolution_apo'], errors='coerce').fillna(99) <= 2.0]
    funnel.append(("Apo resolution ≤2.0Å", len(df)))

    # Filter: holo resolution <= 2.5
    df = df[pd.to_numeric(df['resolution'], errors='coerce').fillna(99) <= 2.5]
    funnel.append(("Holo resolution ≤2.5Å", len(df)))

    # Filter: valid PDB IDs (4 chars)
    df = df[df['rcsb_id_apo'].astype(str).str.len() == 4]
    df = df[df['rcsb_id'].astype(str).str.len() == 4]
    funnel.append(("Valid PDB IDs", len(df)))

    # Filter: protein-only (no nucleic acid chains based on polymer_composition)
    if 'polymer_composition_apo' in df.columns:
        df = df[~df['polymer_composition_apo'].astype(str).str.contains('Nucleic|DNA|RNA', case=False, na=False)]
        funnel.append(("Protein-only apo", len(df)))

    # Filter: crypticity defined
    if 'crypticity' in df.columns:
        df = df[df['crypticity'].notna()]
        funnel.append(("Crypticity defined", len(df)))

    print(f"\n  Filtering funnel:")
    for stage, count in funnel:
        pct = 100 * count / funnel[0][1] if funnel[0][1] > 0 else 0
        print(f"    {stage:<35} {count:>10} ({pct:.1f}%)")

    # ── Phase 2: Deduplicate to one row per unique apo-holo-ligand triple ──
    # Group by (apo PDB, apo chain, holo PDB, holo chain, ligand)
    # Keep the row with highest pocket RMSD (hardest test)
    df['_apo_pdb'] = df['rcsb_id_apo'].astype(str).str.lower()
    df['_holo_pdb'] = df['rcsb_id'].astype(str).str.lower()
    df['_apo_chain'] = df['auth_asym_id_apo'].astype(str).str.strip()
    df['_holo_chain'] = df['auth_asym_id'].astype(str).str.strip()
    df['_ligand'] = df['ligandId'].astype(str).str.strip()
    df['_rmsd'] = pd.to_numeric(df['rmsd_local'], errors='coerce').fillna(0)
    df['_lig_atoms'] = pd.to_numeric(df['lig_atoms'], errors='coerce').fillna(0).astype(int)
    df['_resolution_apo'] = pd.to_numeric(df['resolution_apo'], errors='coerce').fillna(99)
    df['_resolution_holo'] = pd.to_numeric(df['resolution'], errors='coerce').fillna(99)
    df['_cryptic'] = df['crypticity'].astype(bool) if 'crypticity' in df.columns else False
    df['_uniprot'] = df['uniprot_accession_apo'].astype(str)
    df['_cluster30'] = df['cluster_id_30_apo'].astype(str) if 'cluster_id_30_apo' in df.columns else ''

    # Parse site_centroid
    def parse_centroid(val):
        if isinstance(val, (list, np.ndarray)):
            return list(val)
        if isinstance(val, str):
            try:
                return json.loads(val)
            except (json.JSONDecodeError, ValueError):
                return None
        return None

    df['_centroid'] = df['site_centroid'].apply(parse_centroid)

    # Deduplicate: one row per (apo_pdb, apo_chain, holo_pdb, ligand)
    df = df.sort_values('_rmsd', ascending=False)
    dedup_key = ['_apo_pdb', '_apo_chain', '_holo_pdb', '_ligand']
    df_dedup = df.drop_duplicates(subset=dedup_key, keep='first')
    print(f"\n  Deduplicated to {len(df_dedup)} unique apo-holo-ligand triples")

    # ── Phase 3: Redundancy removal at 30% sequence identity ──
    # Use cluster_id_30_apo to pick one representative per cluster
    # Within each cluster, keep the pair with highest pocket RMSD
    if df_dedup['_cluster30'].nunique() > 1:
        cluster_reps = df_dedup.sort_values('_rmsd', ascending=False).drop_duplicates(
            subset=['_cluster30'], keep='first')
        print(f"  30% identity clustering: {len(df_dedup)} → {len(cluster_reps)} clusters")
        df_final = cluster_reps
    else:
        # Fall back to UniProt-level dedup
        df_final = df_dedup.sort_values('_rmsd', ascending=False).drop_duplicates(
            subset=['_uniprot'], keep='first')
        print(f"  UniProt dedup: {len(df_dedup)} → {len(df_final)}")

    # ── Phase 4: Sort and select ──
    # CRYPTIC first (descending RMSD), then STANDARD
    df_final = df_final.sort_values(['_cryptic', '_rmsd'], ascending=[False, False])

    n_cryptic = df_final['_cryptic'].sum()
    n_standard = len(df_final) - n_cryptic
    print(f"\n  Available: {len(df_final)} pairs ({n_cryptic} CRYPTIC, {n_standard} STANDARD)")

    # Select target count
    target = min(args.target_count, len(df_final))
    df_selected = df_final.head(target)
    print(f"  Selected: {target} pairs")

    # ── Phase 5: Check overlap with active 380-target corpus ──
    corpus_path = Path("/mnt/storage/prism-outputs/_corpus_runner_logs/proteome_1000_gt_valid_sorted.txt")
    if corpus_path.exists():
        corpus_pdbs = set()
        with open(corpus_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    corpus_pdbs.add(line.split("_")[0].lower())

        overlap_mask = df_selected['_apo_pdb'].isin(corpus_pdbs) | df_selected['_holo_pdb'].isin(corpus_pdbs)
        n_overlap = overlap_mask.sum()
        if n_overlap > 0:
            print(f"  WARNING: {n_overlap} pairs overlap with active 380-target corpus — removing")
            df_selected = df_selected[~overlap_mask]
            # Backfill from remaining
            remaining = df_final[~df_final.index.isin(df_selected.index)]
            remaining = remaining[~remaining['_apo_pdb'].isin(corpus_pdbs) & ~remaining['_holo_pdb'].isin(corpus_pdbs)]
            need = target - len(df_selected)
            if need > 0 and len(remaining) > 0:
                df_selected = pd.concat([df_selected, remaining.head(need)])
            print(f"  After overlap removal + backfill: {len(df_selected)} pairs")
        else:
            print(f"  No overlap with active corpus")

    # ── Phase 6: Download PDBs ──
    if not args.skip_download:
        apo_ids = set(df_selected['_apo_pdb'].unique())
        holo_ids = set(df_selected['_holo_pdb'].unique())
        all_ids = apo_ids | holo_ids
        print(f"\n  Downloading {len(all_ids)} unique PDBs...")

        APO_DIR.mkdir(parents=True, exist_ok=True)
        HOLO_DIR.mkdir(parents=True, exist_ok=True)

        apo_paths = {}
        holo_paths = {}
        with ThreadPoolExecutor(max_workers=16) as pool:
            apo_futures = {pool.submit(fetch_pdb, pid, APO_DIR): pid for pid in apo_ids}
            holo_futures = {pool.submit(fetch_pdb, pid, HOLO_DIR): pid for pid in holo_ids}
            done = 0
            for fut in as_completed(list(apo_futures) + list(holo_futures)):
                done += 1
                if done % 100 == 0:
                    print(f"    {done}/{len(all_ids)} complete")
                if fut in apo_futures:
                    apo_paths[apo_futures[fut]] = fut.result()
                else:
                    holo_paths[holo_futures[fut]] = fut.result()

        apo_ok = sum(1 for v in apo_paths.values() if v)
        holo_ok = sum(1 for v in holo_paths.values() if v)
        print(f"  Downloads: {apo_ok} apo, {holo_ok} holo")

        # Filter out pairs where download failed
        valid_mask = df_selected['_apo_pdb'].map(lambda x: apo_paths.get(x) is not None) & \
                     df_selected['_holo_pdb'].map(lambda x: holo_paths.get(x) is not None)
        df_selected = df_selected[valid_mask]
        print(f"  After download validation: {len(df_selected)} pairs")

        # Validate: X-ray, no nucleic acids
        drop_idx = []
        for idx, row in df_selected.iterrows():
            apo_path = apo_paths.get(row['_apo_pdb'])
            if apo_path and not is_xray(apo_path):
                drop_idx.append(idx)
            elif apo_path and has_nucleic(apo_path):
                drop_idx.append(idx)
        if drop_idx:
            df_selected = df_selected.drop(drop_idx)
            print(f"  After X-ray/nucleic validation: {len(df_selected)} pairs")
    else:
        apo_paths = {}
        holo_paths = {}

    # ── Phase 7: Generate GT files ──
    GT_DIR.mkdir(parents=True, exist_ok=True)
    pairs_out = []

    for _, row in df_selected.iterrows():
        holo_path = holo_paths.get(row['_holo_pdb'])
        centroid = row['_centroid']

        # Get ligand atoms from holo for DCA computation
        lig_atoms = []
        binding_res = []
        if holo_path and os.path.exists(holo_path):
            lig_atoms = get_ligand_atoms(holo_path, row['_ligand'], row['_holo_chain'])
            if not lig_atoms and centroid:
                # Use centroid from CryptoBank
                pass
            binding_res = get_binding_residues(holo_path, lig_atoms, row['_holo_chain'])

        # Compute centroid from ligand atoms if not available from CryptoBank
        if not centroid and lig_atoms:
            centroid = np.mean(lig_atoms, axis=0).tolist()

        gt = {
            "apo_pdb": row['_apo_pdb'],
            "apo_chain": row['_apo_chain'],
            "holo_pdb": row['_holo_pdb'],
            "holo_chain": row['_holo_chain'],
            "ligand_code": row['_ligand'],
            "ligand_heavy_atoms": int(row['_lig_atoms']),
            "ligand_centroid": centroid,
            "binding_residues_holo": [{"resnum": r[0], "resname": r[1]} for r in binding_res],
            "n_binding_residues": len(binding_res),
            "pocket_rmsd": round(float(row['_rmsd']), 3),
            "category": "CRYPTIC" if row['_cryptic'] else "STANDARD",
            "apo_resolution": round(float(row['_resolution_apo']), 2),
            "holo_resolution": round(float(row['_resolution_holo']), 2),
            "uniprot_id": row['_uniprot'],
            "cluster_id_30": str(row['_cluster30']),
        }

        gt_path = GT_DIR / f"{row['_apo_pdb']}_{row['_apo_chain']}.ground_truth.json"
        with open(gt_path, "w") as f:
            json.dump(gt, f, indent=2)

        pairs_out.append(gt)

    print(f"\n  Generated {len(pairs_out)} GT files")

    # ── Phase 8: Build manifest ──
    cryptic = [p for p in pairs_out if p['category'] == 'CRYPTIC']
    standard = [p for p in pairs_out if p['category'] == 'STANDARD']

    manifest = {
        "pipeline": "true_apo_validation",
        "source": "CryptoBank (Zenodo 15212595, 5.49M rows)",
        "n_pairs": len(pairs_out),
        "n_cryptic": len(cryptic),
        "n_standard": len(standard),
        "pairs": [{
            "apo_pdb": p['apo_pdb'],
            "apo_chain": p['apo_chain'],
            "holo_pdb": p['holo_pdb'],
            "holo_chain": p['holo_chain'],
            "ligand_code": p['ligand_code'],
            "ligand_heavy_atoms": p['ligand_heavy_atoms'],
            "pocket_rmsd": p['pocket_rmsd'],
            "category": p['category'],
            "apo_resolution": p['apo_resolution'],
            "gt_file": f"ground_truth/{p['apo_pdb']}_{p['apo_chain']}.ground_truth.json",
        } for p in pairs_out],
    }

    manifest_path = BENCH_DIR / "true_apo_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    chain_list_path = BENCH_DIR / "true_apo_chains.txt"
    with open(chain_list_path, "w") as f:
        for p in pairs_out:
            f.write(f"{p['apo_pdb']}_chain{p['apo_chain']}\n")

    # ── Summary ──
    print(f"\n{'='*60}")
    print(f"  TRUE APO MANIFEST — FINAL SUMMARY")
    print(f"{'='*60}")
    print(f"  Total pairs:            {len(pairs_out)}")
    print(f"    CRYPTIC (≥2.0Å):      {len(cryptic)}")
    print(f"    STANDARD (<2.0Å):     {len(standard)}")
    if pairs_out:
        rmsds = [p['pocket_rmsd'] for p in pairs_out if p['pocket_rmsd'] > 0]
        if rmsds:
            print(f"  Median pocket RMSD:     {np.median(rmsds):.2f}Å")
            print(f"  Max pocket RMSD:        {max(rmsds):.2f}Å")
        resolutions = [p['apo_resolution'] for p in pairs_out if 0 < p['apo_resolution'] < 90]
        if resolutions:
            print(f"  Apo resolution range:   {min(resolutions):.2f} - {max(resolutions):.2f}Å")
        atoms = [p['ligand_heavy_atoms'] for p in pairs_out if p['ligand_heavy_atoms'] > 0]
        if atoms:
            print(f"  Avg ligand heavy atoms: {np.mean(atoms):.0f}")
    print(f"\n  Manifest: {manifest_path}")
    print(f"  Chain list: {chain_list_path}")
    print(f"  GT files: {GT_DIR}/")
    print(f"\n  Engine runs are P2 — wait for 380-target campaign to finish.")


if __name__ == "__main__":
    main()
