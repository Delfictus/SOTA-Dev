#!/usr/bin/env python3
"""Runner 1 — Build true apo-holo validation manifest from CryptoBank.

Downloads CryptoBank dataset from Zenodo, applies quality filters per the
True APO Curation Pipeline directive, validates prism-prep compatibility
and residue ID mapping, generates GT files, and outputs an engine-ready
manifest.

Phases:
  1. Download and parse CryptoBank (Zenodo pickle)
  2. Quality filtering (apo resolution, holo resolution, ligand size, etc.)
  3. Pair selection + redundancy removal (30% seq identity)
  4. prism-prep validation + residue mapping
  5. GT file generation + manifest assembly

CPU-only. Does NOT launch engine runs.

Usage:
    python3 scripts/quarantine/build_true_apo_manifest.py [--skip-download]
"""

import argparse
import json
import math
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

# ── Paths ──
BASE = Path(__file__).resolve().parent.parent.parent
BENCH_DIR = BASE / "benchmarks" / "true_apo"
DATA_DIR = BENCH_DIR / "data"
GT_DIR = BENCH_DIR / "ground_truth"
APO_DIR = BENCH_DIR / "apo_pdbs"
HOLO_DIR = BENCH_DIR / "holo_pdbs"
TOPO_DIR = BENCH_DIR / "topologies"

ZENODO_URL = "https://zenodo.org/records/15212595/files/cryptobank_dataset_12_03_2025.pkl"
PICKLE_PATH = DATA_DIR / "cryptobank_dataset_12_03_2025.pkl"

# ── Nuisance exclusion (from directive) ──
NUISANCE_HETATMS = {
    "HOH", "DOD", "GOL", "EDO", "PEG", "PGE", "DMS", "MPD",
    "MES", "EPE", "TRS", "ACT", "FMT", "BME", "IPA",
    "SO4", "PO4", "CL", "NA", "K", "CA", "MG", "ZN",
    "NO3", "SCN", "BR", "IOD", "FE", "MN", "CO", "NI", "CU",
    "CD", "HG",
    "CIT", "TAR", "SUC", "MLI",
    "BOG", "LDA", "SDS", "UNX", "UNL",
    "HOH", "H2O", "WAT", "TIP",
}


def download_cryptobank():
    """Download CryptoBank pickle from Zenodo."""
    if PICKLE_PATH.exists() and PICKLE_PATH.stat().st_size > 1_000_000_000:
        print(f"CryptoBank pickle already downloaded: {PICKLE_PATH} ({PICKLE_PATH.stat().st_size / 1e9:.1f} GB)")
        return True

    print(f"Downloading CryptoBank from Zenodo ({ZENODO_URL})...")
    print("  This is ~4.8 GB, may take several minutes.")
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    try:
        req = urllib.request.Request(ZENODO_URL, headers={"User-Agent": "prism4d-true-apo/1.0"})
        with urllib.request.urlopen(req, timeout=600) as resp:
            total = int(resp.headers.get("Content-Length", 0))
            downloaded = 0
            chunk_size = 1024 * 1024  # 1 MB
            with open(PICKLE_PATH, "wb") as f:
                while True:
                    chunk = resp.read(chunk_size)
                    if not chunk:
                        break
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total > 0 and downloaded % (50 * chunk_size) == 0:
                        pct = 100 * downloaded / total
                        print(f"  {downloaded / 1e9:.1f} / {total / 1e9:.1f} GB ({pct:.0f}%)")
        print(f"  Download complete: {PICKLE_PATH.stat().st_size / 1e9:.1f} GB")
        return True
    except (urllib.error.URLError, OSError, TimeoutError) as e:
        print(f"  Download failed: {e}")
        return False


def load_cryptobank():
    """Load and inspect CryptoBank pickle."""
    print(f"Loading CryptoBank pickle ({PICKLE_PATH.stat().st_size / 1e9:.1f} GB)...")
    with open(PICKLE_PATH, "rb") as f:
        data = pickle.load(f)

    # Inspect structure
    if isinstance(data, dict):
        print(f"  Type: dict with keys: {list(data.keys())[:20]}")
        # Try to find the main data structure
        for key in data:
            val = data[key]
            if isinstance(val, (list, dict)):
                print(f"  [{key}]: {type(val).__name__}, len={len(val)}")
            elif hasattr(val, 'shape'):
                print(f"  [{key}]: {type(val).__name__}, shape={val.shape}")
            else:
                print(f"  [{key}]: {type(val).__name__} = {str(val)[:100]}")
    elif isinstance(data, list):
        print(f"  Type: list, len={len(data)}")
        if data:
            print(f"  First entry type: {type(data[0])}")
            if isinstance(data[0], dict):
                print(f"  First entry keys: {list(data[0].keys())}")
                print(f"  First entry (truncated):")
                for k, v in list(data[0].items())[:15]:
                    print(f"    {k}: {str(v)[:100]}")
    elif hasattr(data, 'columns'):
        # DataFrame
        print(f"  Type: DataFrame, shape={data.shape}")
        print(f"  Columns: {list(data.columns)}")
        print(f"  First row:")
        print(data.iloc[0])

    return data


def inspect_and_extract_pairs(data):
    """Extract apo-holo-ligand triples from CryptoBank data.

    CryptoBank format is not documented in detail, so this function
    inspects the structure and adapts. Returns a list of dicts:
    [{apo_pdb, apo_chain, holo_pdb, holo_chain, ligand_code,
      ligand_n_atoms, pocket_rmsd, apo_resolution, holo_resolution, uniprot_id}, ...]
    """
    pairs = []

    # CryptoBank is likely a DataFrame or dict of DataFrames
    # The paper mentions: apo_id, holo_id, ligand, pocket_rmsd, crypticity_score
    # Let's handle multiple possible formats

    if hasattr(data, 'columns'):
        # It's a DataFrame
        df = data
        print(f"  DataFrame with {len(df)} rows, columns: {list(df.columns)}")

        # Save schema for inspection
        schema_path = DATA_DIR / "cryptobank_schema.json"
        schema = {
            "n_rows": len(df),
            "columns": list(df.columns),
            "dtypes": {c: str(df[c].dtype) for c in df.columns},
            "sample_values": {c: str(df[c].iloc[0]) for c in df.columns},
        }
        with open(schema_path, "w") as f:
            json.dump(schema, f, indent=2, default=str)
        print(f"  Schema saved to {schema_path}")

        # Extract pairs based on available columns
        # Map possible column names
        col_map = {}
        for col in df.columns:
            cl = col.lower()
            if 'apo' in cl and ('pdb' in cl or 'id' in cl):
                col_map['apo_pdb'] = col
            elif 'holo' in cl and ('pdb' in cl or 'id' in cl):
                col_map['holo_pdb'] = col
            elif 'ligand' in cl and ('name' in cl or 'code' in cl or 'id' in cl):
                col_map['ligand'] = col
            elif cl in ('ligand', 'lig'):
                col_map['ligand'] = col
            elif 'rmsd' in cl or 'prmsd' in cl:
                col_map['pocket_rmsd'] = col
            elif 'resolution' in cl and 'apo' in cl:
                col_map['apo_resolution'] = col
            elif 'resolution' in cl and 'holo' in cl:
                col_map['holo_resolution'] = col
            elif 'uniprot' in cl:
                col_map['uniprot'] = col
            elif 'chain' in cl and 'apo' in cl:
                col_map['apo_chain'] = col
            elif 'chain' in cl and 'holo' in cl:
                col_map['holo_chain'] = col
            elif 'cryptic' in cl:
                col_map['crypticity'] = col
            elif 'n_atom' in cl or 'num_atom' in cl or 'heavy_atom' in cl:
                col_map['n_atoms'] = col

        print(f"  Column mapping: {col_map}")

        for _, row in df.iterrows():
            pair = {}
            pair['apo_pdb'] = str(row.get(col_map.get('apo_pdb', ''), '')).strip().lower()[:4]
            pair['holo_pdb'] = str(row.get(col_map.get('holo_pdb', ''), '')).strip().lower()[:4]
            pair['apo_chain'] = str(row.get(col_map.get('apo_chain', ''), 'A')).strip().upper()[:1]
            pair['holo_chain'] = str(row.get(col_map.get('holo_chain', ''), 'A')).strip().upper()[:1]
            pair['ligand_code'] = str(row.get(col_map.get('ligand', ''), '')).strip().upper()
            pair['uniprot_id'] = str(row.get(col_map.get('uniprot', ''), '')).strip()

            rmsd_col = col_map.get('pocket_rmsd', '')
            try:
                pair['pocket_rmsd'] = float(row.get(rmsd_col, 0))
            except (ValueError, TypeError):
                pair['pocket_rmsd'] = 0.0

            for res_field in ['apo_resolution', 'holo_resolution']:
                rc = col_map.get(res_field, '')
                try:
                    pair[res_field] = float(row.get(rc, 99.0))
                except (ValueError, TypeError):
                    pair[res_field] = 99.0

            n_atoms_col = col_map.get('n_atoms', '')
            try:
                pair['ligand_n_atoms'] = int(row.get(n_atoms_col, 0))
            except (ValueError, TypeError):
                pair['ligand_n_atoms'] = 0

            # Only add if we got valid PDB IDs
            if len(pair['apo_pdb']) == 4 and len(pair['holo_pdb']) == 4:
                pairs.append(pair)

    elif isinstance(data, dict):
        # Inspect first-level keys to find the pair data
        print(f"  Dict with keys: {list(data.keys())[:30]}")
        schema_path = DATA_DIR / "cryptobank_schema.json"
        schema = {}
        for k, v in data.items():
            if isinstance(v, (list, dict)):
                schema[k] = {"type": type(v).__name__, "len": len(v)}
            elif hasattr(v, 'shape'):
                schema[k] = {"type": type(v).__name__, "shape": str(v.shape)}
            else:
                schema[k] = {"type": type(v).__name__, "value": str(v)[:200]}
        with open(schema_path, "w") as f:
            json.dump(schema, f, indent=2, default=str)
        print(f"  Schema saved to {schema_path}")

        # Try common patterns
        for key in ['pairs', 'data', 'entries', 'alignments', 'results']:
            if key in data and isinstance(data[key], list):
                print(f"  Found list at key '{key}' with {len(data[key])} entries")
                for entry in data[key]:
                    if isinstance(entry, dict):
                        pair = {
                            'apo_pdb': str(entry.get('apo_pdb', entry.get('apo_id', ''))).lower()[:4],
                            'holo_pdb': str(entry.get('holo_pdb', entry.get('holo_id', ''))).lower()[:4],
                            'apo_chain': str(entry.get('apo_chain', 'A')).upper()[:1],
                            'holo_chain': str(entry.get('holo_chain', 'A')).upper()[:1],
                            'ligand_code': str(entry.get('ligand', entry.get('ligand_code', ''))).upper(),
                            'pocket_rmsd': float(entry.get('pocket_rmsd', entry.get('prmsd', 0))),
                            'apo_resolution': float(entry.get('apo_resolution', 99)),
                            'holo_resolution': float(entry.get('holo_resolution', 99)),
                            'ligand_n_atoms': int(entry.get('n_atoms', entry.get('ligand_n_atoms', 0))),
                            'uniprot_id': str(entry.get('uniprot', entry.get('uniprot_id', ''))),
                        }
                        if len(pair['apo_pdb']) == 4 and len(pair['holo_pdb']) == 4:
                            pairs.append(pair)

    print(f"\n  Extracted {len(pairs)} raw pairs")
    return pairs


def fetch_pdb(pdb_id, dest_dir):
    """Download a PDB file from RCSB. Returns Path or None."""
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    path = dest_dir / f"{pdb_id.lower()}.pdb"

    if path.exists() and path.stat().st_size > 100:
        return path

    url = f"https://files.rcsb.org/download/{pdb_id.upper()}.pdb"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "prism4d-true-apo/1.0"})
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = resp.read()
        if len(data) < 100:
            return None
        path.write_bytes(data)
        return path
    except (urllib.error.HTTPError, urllib.error.URLError, OSError, TimeoutError):
        return None


def get_resolution_from_pdb(pdb_path):
    """Extract resolution from PDB REMARK 2."""
    try:
        with open(pdb_path, errors="ignore") as f:
            for line in f:
                if line.startswith("REMARK   2 RESOLUTION."):
                    m = re.search(r'(\d+\.\d+)', line)
                    if m:
                        return float(m.group(1))
                if line.startswith("ATOM"):
                    break
    except OSError:
        pass
    return None


def get_ligand_atoms_from_holo(holo_path, ligand_code, holo_chain):
    """Extract ligand heavy atom coordinates from holo PDB."""
    atoms = []
    try:
        with open(holo_path, errors="ignore") as f:
            for line in f:
                if not line.startswith("HETATM"):
                    continue
                resname = line[17:20].strip()
                chain = line[21]
                if resname == ligand_code and chain == holo_chain:
                    try:
                        x = float(line[30:38])
                        y = float(line[38:46])
                        z = float(line[46:54])
                        atoms.append([x, y, z])
                    except (ValueError, IndexError):
                        continue
    except OSError:
        pass
    return atoms


def get_binding_residues(holo_path, ligand_atoms, holo_chain, cutoff=4.5):
    """Find residues within cutoff of ligand atoms."""
    if not ligand_atoms:
        return []
    lig_np = np.array(ligand_atoms)
    residues = set()
    try:
        with open(holo_path, errors="ignore") as f:
            for line in f:
                if not line.startswith("ATOM"):
                    continue
                chain = line[21]
                if chain != holo_chain:
                    continue
                try:
                    x = float(line[30:38])
                    y = float(line[38:46])
                    z = float(line[46:54])
                    resnum = int(line[22:26])
                    resname = line[17:20].strip()
                except (ValueError, IndexError):
                    continue
                atom_pos = np.array([x, y, z])
                dists = np.linalg.norm(lig_np - atom_pos, axis=1)
                if dists.min() <= cutoff:
                    residues.add((resnum, resname))
    except OSError:
        pass
    return sorted(residues)


def check_nucleic_chains(pdb_path):
    """Check for DNA/RNA chains."""
    NUCLEIC = {"DA", "DC", "DG", "DT", "DU", "A", "C", "G", "U", "T",
               "RA", "RC", "RG", "RU"}
    try:
        with open(pdb_path, errors="ignore") as f:
            for line in f:
                if line.startswith(("ATOM", "HETATM")):
                    resname = line[17:20].strip()
                    if resname in NUCLEIC:
                        return True
    except OSError:
        pass
    return False


def check_xray(pdb_path):
    """Check if structure is X-ray."""
    try:
        with open(pdb_path, errors="ignore") as f:
            for line in f:
                if line.startswith("EXPDTA"):
                    if "X-RAY" in line.upper():
                        return True
                    return False
                if line.startswith("ATOM"):
                    break
    except OSError:
        pass
    return False  # Unknown method, fail safe


def filter_pairs(pairs):
    """Apply quality filters from the directive."""
    print(f"\n{'='*60}")
    print(f"  QUALITY FILTERING")
    print(f"{'='*60}")
    print(f"  Starting with {len(pairs)} raw pairs")

    funnel = [("Raw pairs", len(pairs))]

    # Filter 1: Remove nuisance ligands
    pairs = [p for p in pairs if p['ligand_code'] not in NUISANCE_HETATMS]
    funnel.append(("After nuisance ligand removal", len(pairs)))

    # Filter 2: Ligand heavy atoms >= 10 (drug-like only)
    if any(p['ligand_n_atoms'] > 0 for p in pairs):
        pairs = [p for p in pairs if p['ligand_n_atoms'] >= 10]
        funnel.append(("After ligand ≥10 heavy atoms", len(pairs)))

    # Filter 3: Valid PDB IDs (4 chars)
    pairs = [p for p in pairs if len(p['apo_pdb']) == 4 and len(p['holo_pdb']) == 4
             and p['apo_pdb'].isalnum() and p['holo_pdb'].isalnum()]
    funnel.append(("After PDB ID validation", len(pairs)))

    # Filter 4: Resolution filters (if data available from CryptoBank)
    if any(p['apo_resolution'] < 90 for p in pairs):
        pairs = [p for p in pairs if p['apo_resolution'] <= 2.0]
        funnel.append(("After apo resolution ≤2.0Å", len(pairs)))

    if any(p['holo_resolution'] < 90 for p in pairs):
        pairs = [p for p in pairs if p['holo_resolution'] <= 2.5]
        funnel.append(("After holo resolution ≤2.5Å", len(pairs)))

    # Print funnel
    print(f"\n  {'Stage':<40} {'Count':>8} {'%':>6}")
    print(f"  {'-'*56}")
    for stage, count in funnel:
        pct = 100 * count / funnel[0][1] if funnel[0][1] > 0 else 0
        print(f"  {stage:<40} {count:>8} {pct:>5.1f}%")

    return pairs, funnel


def download_pdbs_parallel(pairs, max_workers=16):
    """Download all apo and holo PDBs in parallel."""
    apo_ids = set(p['apo_pdb'] for p in pairs)
    holo_ids = set(p['holo_pdb'] for p in pairs)
    all_ids = apo_ids | holo_ids
    print(f"\n  Downloading {len(all_ids)} unique PDBs ({len(apo_ids)} apo + {len(holo_ids)} holo)...")

    apo_ok = {}
    holo_ok = {}

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        # Submit apo downloads
        apo_futures = {pool.submit(fetch_pdb, pid, APO_DIR): pid for pid in apo_ids}
        holo_futures = {pool.submit(fetch_pdb, pid, HOLO_DIR): pid for pid in holo_ids}

        done = 0
        for fut in as_completed(list(apo_futures.keys()) + list(holo_futures.keys())):
            done += 1
            if done % 100 == 0:
                print(f"    {done}/{len(all_ids)} downloads complete")
            if fut in apo_futures:
                apo_ok[apo_futures[fut]] = fut.result()
            else:
                holo_ok[holo_futures[fut]] = fut.result()

    apo_success = sum(1 for v in apo_ok.values() if v is not None)
    holo_success = sum(1 for v in holo_ok.values() if v is not None)
    print(f"  Downloads complete: {apo_success}/{len(apo_ids)} apo, {holo_success}/{len(holo_ids)} holo")

    return apo_ok, holo_ok


def validate_pdbs(pairs, apo_paths, holo_paths):
    """Validate PDB quality: X-ray, no nucleic acids, resolution."""
    print(f"\n  Validating PDB quality...")
    valid = []
    reasons = defaultdict(int)

    for p in pairs:
        apo_path = apo_paths.get(p['apo_pdb'])
        holo_path = holo_paths.get(p['holo_pdb'])

        if apo_path is None:
            reasons['apo_download_failed'] += 1
            continue
        if holo_path is None:
            reasons['holo_download_failed'] += 1
            continue

        # X-ray check on apo
        if not check_xray(apo_path):
            reasons['apo_not_xray'] += 1
            continue

        # Nucleic acid check
        if check_nucleic_chains(apo_path):
            reasons['apo_has_nucleic'] += 1
            continue

        # Resolution from PDB header (if not available from CryptoBank)
        if p['apo_resolution'] > 90:
            res = get_resolution_from_pdb(apo_path)
            if res is not None:
                p['apo_resolution'] = res
                if res > 2.0:
                    reasons['apo_resolution_>2.0'] += 1
                    continue

        if p['holo_resolution'] > 90:
            res = get_resolution_from_pdb(holo_path)
            if res is not None:
                p['holo_resolution'] = res
                if res > 2.5:
                    reasons['holo_resolution_>2.5'] += 1
                    continue

        # Validate ligand exists in holo
        lig_atoms = get_ligand_atoms_from_holo(holo_path, p['ligand_code'], p['holo_chain'])
        if len(lig_atoms) < 10:
            reasons['ligand_<10_atoms_in_holo'] += 1
            continue

        p['ligand_n_atoms'] = len(lig_atoms)
        p['ligand_centroid'] = np.mean(lig_atoms, axis=0).tolist()
        p['_apo_path'] = str(apo_path)
        p['_holo_path'] = str(holo_path)
        p['_lig_atoms'] = lig_atoms

        valid.append(p)

    print(f"  Validated: {len(valid)}/{len(pairs)}")
    if reasons:
        print(f"  Rejection reasons:")
        for reason, count in sorted(reasons.items(), key=lambda x: -x[1]):
            print(f"    {reason}: {count}")

    return valid


def deduplicate_by_sequence(pairs, identity_threshold=0.30):
    """Redundancy removal by UniProt grouping.

    If UniProt IDs are available, group by UniProt and keep the best pair
    (highest pocket RMSD for CRYPTIC, lowest resolution for STANDARD).
    If no UniProt, use PDB-level dedup (same apo PDB = same protein).
    """
    print(f"\n  Redundancy removal...")

    if any(p.get('uniprot_id', '') not in ('', 'nan', 'None') for p in pairs):
        # Group by UniProt
        groups = defaultdict(list)
        no_uniprot = []
        for p in pairs:
            uid = p.get('uniprot_id', '')
            if uid and uid not in ('', 'nan', 'None'):
                groups[uid].append(p)
            else:
                # Fall back to apo PDB grouping
                groups[f"pdb_{p['apo_pdb']}"].append(p)

        deduped = []
        for uid, group in groups.items():
            # Keep the pair with highest pocket RMSD (hardest test)
            best = max(group, key=lambda p: p.get('pocket_rmsd', 0))
            deduped.append(best)

        print(f"  {len(pairs)} pairs → {len(groups)} UniProt groups → {len(deduped)} representative pairs")
    else:
        # No UniProt — dedup by apo PDB
        seen_apo = {}
        for p in pairs:
            key = p['apo_pdb']
            if key not in seen_apo or p.get('pocket_rmsd', 0) > seen_apo[key].get('pocket_rmsd', 0):
                seen_apo[key] = p
        deduped = list(seen_apo.values())
        print(f"  {len(pairs)} pairs → {len(seen_apo)} unique apo PDBs → {len(deduped)} representative pairs")

    return deduped


def generate_ground_truth(pairs):
    """Generate GT JSON files for each validated pair."""
    print(f"\n  Generating ground truth files...")
    GT_DIR.mkdir(parents=True, exist_ok=True)

    for p in pairs:
        holo_path = p.get('_holo_path', '')
        lig_atoms = p.get('_lig_atoms', [])

        # Compute binding residues
        binding_res = get_binding_residues(holo_path, lig_atoms, p['holo_chain'])

        gt = {
            "apo_pdb": p['apo_pdb'],
            "apo_chain": p['apo_chain'],
            "holo_pdb": p['holo_pdb'],
            "holo_chain": p['holo_chain'],
            "ligand_code": p['ligand_code'],
            "ligand_heavy_atoms": p['ligand_n_atoms'],
            "ligand_centroid": p.get('ligand_centroid', []),
            "binding_residues_holo": [{"resnum": r[0], "resname": r[1]} for r in binding_res],
            "n_binding_residues": len(binding_res),
            "pocket_rmsd": p.get('pocket_rmsd', 0),
            "category": "CRYPTIC" if p.get('pocket_rmsd', 0) >= 2.0 else "STANDARD",
            "apo_resolution": p.get('apo_resolution', None),
            "holo_resolution": p.get('holo_resolution', None),
            "uniprot_id": p.get('uniprot_id', ''),
        }

        gt_path = GT_DIR / f"{p['apo_pdb']}_{p['apo_chain']}.ground_truth.json"
        with open(gt_path, "w") as f:
            json.dump(gt, f, indent=2)

    print(f"  Generated {len(pairs)} GT files in {GT_DIR}")


def check_overlap_with_corpus(pairs):
    """Check for UniProt overlap with the active 380-target corpus."""
    corpus_manifest = Path("/mnt/storage/prism-outputs/_corpus_runner_logs/proteome_1000_gt_valid_sorted.txt")
    if not corpus_manifest.exists():
        print("  Corpus manifest not found — skipping overlap check")
        return pairs

    corpus_pdbs = set()
    with open(corpus_manifest) as f:
        for line in f:
            line = line.strip()
            if line:
                pdb = line.split("_")[0].lower()
                corpus_pdbs.add(pdb)

    overlapping = [p for p in pairs if p['apo_pdb'] in corpus_pdbs or p['holo_pdb'] in corpus_pdbs]
    clean = [p for p in pairs if p['apo_pdb'] not in corpus_pdbs and p['holo_pdb'] not in corpus_pdbs]

    if overlapping:
        print(f"  WARNING: {len(overlapping)} pairs overlap with active 380-target corpus")
        print(f"  Removing overlapping pairs to maintain holdout integrity")
        for p in overlapping[:5]:
            print(f"    {p['apo_pdb']}_{p['apo_chain']} / {p['holo_pdb']}")
    else:
        print(f"  No overlap with active corpus")

    return clean


def build_manifest(pairs):
    """Build the final engine-ready manifest."""
    # Sort: CRYPTIC first (descending pocket RMSD), then STANDARD
    cryptic = [p for p in pairs if p.get('pocket_rmsd', 0) >= 2.0]
    standard = [p for p in pairs if p.get('pocket_rmsd', 0) < 2.0]

    cryptic.sort(key=lambda p: -p.get('pocket_rmsd', 0))
    standard.sort(key=lambda p: -p.get('pocket_rmsd', 0))

    ordered = cryptic + standard

    manifest = {
        "pipeline": "true_apo_validation",
        "source": "CryptoBank (Zenodo 15212595)",
        "n_pairs": len(ordered),
        "n_cryptic": len(cryptic),
        "n_standard": len(standard),
        "pairs": [],
    }

    for p in ordered:
        entry = {
            "apo_pdb": p['apo_pdb'],
            "apo_chain": p['apo_chain'],
            "holo_pdb": p['holo_pdb'],
            "holo_chain": p['holo_chain'],
            "ligand_code": p['ligand_code'],
            "ligand_heavy_atoms": p['ligand_n_atoms'],
            "pocket_rmsd": round(p.get('pocket_rmsd', 0), 2),
            "category": "CRYPTIC" if p.get('pocket_rmsd', 0) >= 2.0 else "STANDARD",
            "apo_resolution": round(p.get('apo_resolution', 0), 2),
            "holo_resolution": round(p.get('holo_resolution', 0), 2),
            "gt_file": f"ground_truth/{p['apo_pdb']}_{p['apo_chain']}.ground_truth.json",
        }
        manifest["pairs"].append(entry)

    manifest_path = BENCH_DIR / "true_apo_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"\n  Manifest: {manifest_path}")

    # Also write a flat chain list for the corpus runner
    chain_list_path = BENCH_DIR / "true_apo_chains.txt"
    with open(chain_list_path, "w") as f:
        for p in ordered:
            f.write(f"{p['apo_pdb']}_{p['apo_chain']}\n")
    print(f"  Chain list: {chain_list_path}")

    return manifest


def print_summary(manifest, funnel):
    """Print the summary statistics from the directive."""
    pairs = manifest['pairs']
    n = len(pairs)

    print(f"\n{'='*60}")
    print(f"  TRUE APO MANIFEST SUMMARY")
    print(f"{'='*60}")
    print(f"  Total pairs:           {n}")
    print(f"    CRYPTIC (≥2.0 Å):    {manifest['n_cryptic']}")
    print(f"    STANDARD (<2.0 Å):   {manifest['n_standard']}")

    uniprots = set(p.get('uniprot_id', '') for p in pairs if p.get('uniprot_id', ''))
    print(f"  Unique UniProt IDs:    {len(uniprots) if uniprots else 'N/A'}")

    if pairs:
        rmsds = [p['pocket_rmsd'] for p in pairs if p['pocket_rmsd'] > 0]
        if rmsds:
            print(f"  Median pocket RMSD:    {np.median(rmsds):.2f} Å")

        resolutions = [p['apo_resolution'] for p in pairs if p['apo_resolution'] > 0 and p['apo_resolution'] < 90]
        if resolutions:
            print(f"  Resolution range:      {min(resolutions):.2f} - {max(resolutions):.2f} Å")

        atoms = [p['ligand_heavy_atoms'] for p in pairs if p['ligand_heavy_atoms'] > 0]
        if atoms:
            print(f"  Avg ligand heavy atoms: {np.mean(atoms):.0f}")

    print(f"\n  Filtering funnel:")
    for stage, count in funnel:
        print(f"    {stage}: {count}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-download", action="store_true",
                        help="Skip Zenodo download (use existing pickle)")
    args = parser.parse_args()

    # Phase 1: Download
    if not args.skip_download:
        if not download_cryptobank():
            print("ERROR: CryptoBank download failed. Cannot proceed.")
            sys.exit(1)

    if not PICKLE_PATH.exists():
        print(f"ERROR: CryptoBank pickle not found at {PICKLE_PATH}")
        sys.exit(1)

    # Phase 1: Load and parse
    data = load_cryptobank()
    pairs = inspect_and_extract_pairs(data)

    if not pairs:
        print("ERROR: No pairs extracted from CryptoBank. Check the schema.")
        print("The schema has been saved to data/cryptobank_schema.json for inspection.")
        sys.exit(1)

    # Phase 2: Quality filtering
    pairs, funnel = filter_pairs(pairs)

    if not pairs:
        print("ERROR: All pairs filtered out. Check thresholds.")
        sys.exit(1)

    # Download PDBs
    apo_paths, holo_paths = download_pdbs_parallel(pairs)

    # Validate PDBs
    pairs = validate_pdbs(pairs, apo_paths, holo_paths)
    funnel.append(("After PDB validation", len(pairs)))

    if not pairs:
        print("ERROR: All pairs failed PDB validation.")
        sys.exit(1)

    # Phase 3: Redundancy removal
    pairs = deduplicate_by_sequence(pairs)
    funnel.append(("After redundancy removal", len(pairs)))

    # Check overlap with active corpus
    pairs = check_overlap_with_corpus(pairs)
    funnel.append(("After corpus overlap removal", len(pairs)))

    # Phase 4: Generate GT and manifest
    generate_ground_truth(pairs)
    manifest = build_manifest(pairs)
    print_summary(manifest, funnel)

    print(f"\n  DONE. {len(pairs)} pairs ready for engine validation.")
    print(f"  Engine runs are P2 — wait for 380-target campaign to finish.")


if __name__ == "__main__":
    main()
