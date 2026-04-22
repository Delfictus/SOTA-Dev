#!/usr/bin/env python3
"""Audit the proteome_1000 manifest for ground truth availability.

For each PDB ID × chain in the corpus manifest, resolves ground truth
by downloading the holo PDB from RCSB and classifying the deposit.

Categories:
  - VALID_DRUG_LIKE:    co-crystallized ligand >= 10 heavy atoms on target chain
  - VALID_METAL:        catalytic metal cofactor on target chain
  - VALID_SMALL:        small molecule (5-9 atoms) on target chain
  - VALID_OFF_CHAIN:    drug-like ligand on different chain (may still be valid)
  - PANDDA:             PanDDA fragment screen deposit (invalid for DCC)
  - TEMPLATED:          nucleic acid chains present (ligand position templated)
  - APO:                no ligands bound at all
  - NUISANCE_ONLY:      only solvents/buffers/ions
  - NO_PDB:             RCSB 404 or download failure
  - NO_LIGAND_ON_CHAIN: ligands exist but not on this chain

Output:
  - stdout: summary stats
  - proteome_1000_gt_audit.json: full per-chain classification
  - proteome_1000_gt_valid.txt: chain IDs with valid GT (new manifest)
"""

import json
import os
import sys
import re
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

# Import classification logic from the canonical ground truth resolver
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from importlib import import_module

# ── Configuration ──
CACHE_DIR = Path.home() / ".cache" / "prism4d" / "holo_pdbs"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

MANIFEST = Path("/mnt/storage/prism-outputs/_corpus_runner_logs/proteome_1000_manifest.txt")
OUTPUT_DIR = Path("/mnt/storage/prism-outputs/_corpus_runner_logs")

# Inline the classification constants (avoid import fragility)
NUISANCE_HETATMS = {
    "HOH", "H2O", "DOD", "WAT", "TIP",
    "GOL", "EDO", "DMS", "PEG", "PG4", "PGE",
    "SO4", "PO4", "ACT", "CIT", "TRS", "BME",
    "MES", "HEPES", "EPE", "FMT", "IPA", "MPD",
    "BOG", "DTT", "TLA", "MAL", "FUC", "MAN",
    "BCT", "GLC", "NAG", "BMA",
    "OXY", "PER",
}
COMMON_IONS = {"NA", "CL", "K", "BR", "I", "F", "CS", "RB", "SR"}
METAL_COFACTORS = {"ZN", "MG", "FE", "MN", "CU", "CA", "NI", "CO", "FE2", "FE3"}
NUCLEIC_RESIDUES = {"DA", "DC", "DG", "DT", "DU", "DI",
                    "A", "C", "G", "U", "T", "I",
                    "RA", "RC", "RG", "RU"}
PANDDA_KEYWORDS = ["PANDDA", "FRAGMENT SCREEN", "FRAGMENT-BASED",
                   "FRAGMENT SCREENING", "XCHEM", "X-CHEM"]
ENAMINE_ID_PATTERN = re.compile(r'\bZ[0-9]{8,12}\b')


def fetch_holo_pdb(pdb_id):
    cache_path = CACHE_DIR / f"{pdb_id.lower()}.pdb"
    if cache_path.exists() and cache_path.stat().st_size > 100:
        return cache_path
    url = f"https://files.rcsb.org/download/{pdb_id.upper()}.pdb"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "prism4d-gt-audit/1.0"})
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = resp.read()
        if len(data) < 100:
            return None
        cache_path.write_bytes(data)
        return cache_path
    except (urllib.error.HTTPError, urllib.error.URLError, OSError, TimeoutError):
        return None


def detect_pandda(holo_pdb):
    with open(holo_pdb, errors="ignore") as f:
        for line in f:
            if line.startswith("ATOM") or line.startswith("HETATM"):
                break
            if line.startswith(("HEADER", "TITLE", "COMPND", "JRNL", "REMARK", "KEYWDS")):
                upper = line.upper()
                for kw in PANDDA_KEYWORDS:
                    if kw in upper:
                        return True
                if ENAMINE_ID_PATTERN.search(line):
                    return True
    return False


def detect_nucleic_chains(holo_pdb):
    chains = set()
    with open(holo_pdb, errors="ignore") as f:
        for line in f:
            if line.startswith(("ATOM", "HETATM")):
                resname = line[17:20].strip()
                if resname in NUCLEIC_RESIDUES:
                    chains.add(line[21])
    return chains


def parse_hetatm_groups(holo_pdb):
    groups = {}
    with open(holo_pdb, errors="ignore") as f:
        for line in f:
            if not line.startswith("HETATM"):
                continue
            try:
                resname = line[17:20].strip()
                chain = line[21]
                resnum = int(line[22:26])
                x = float(line[30:38])
                y = float(line[38:46])
                z = float(line[46:54])
            except (ValueError, IndexError):
                continue
            groups.setdefault((resname, chain, resnum), []).append((x, y, z))
    return groups


def classify_ligand(resname, n_atoms):
    if resname in NUISANCE_HETATMS or resname in COMMON_IONS:
        return "nuisance"
    if resname in METAL_COFACTORS and n_atoms == 1:
        return "metal_cofactor"
    if n_atoms >= 10:
        return "drug_like"
    if n_atoms >= 5:
        return "small_molecule"
    return "tiny"


def classify_chain(pdb_id, chain, holo_path):
    """Classify a single PDB chain for GT validity."""
    if holo_path is None:
        return {"category": "NO_PDB", "ligand": None, "detail": "RCSB download failed"}

    # PanDDA check (applies to entire PDB, not per-chain)
    if detect_pandda(holo_path):
        return {"category": "PANDDA", "ligand": None, "detail": "pandda_fragment_screen_deposit"}

    # Nucleic acid check
    nuc_chains = detect_nucleic_chains(holo_path)
    if nuc_chains:
        return {"category": "TEMPLATED", "ligand": None,
                "detail": f"nucleic_chains={','.join(sorted(nuc_chains))}"}

    # Parse HETATMs
    groups = parse_hetatm_groups(holo_path)
    if not groups:
        return {"category": "APO", "ligand": None, "detail": "no_hetatm_records"}

    # Classify each group
    on_chain_drug = []
    on_chain_metal = []
    on_chain_small = []
    off_chain_drug = []
    has_any_non_nuisance = False

    for (resname, ch, resnum), atoms in groups.items():
        klass = classify_ligand(resname, len(atoms))
        if klass in ("nuisance", "tiny"):
            continue
        has_any_non_nuisance = True
        on_chain = (ch == chain)
        entry = {"resname": resname, "chain": ch, "resnum": resnum,
                 "n_atoms": len(atoms), "classification": klass}
        if klass == "drug_like":
            (on_chain_drug if on_chain else off_chain_drug).append(entry)
        elif klass == "metal_cofactor" and on_chain:
            on_chain_metal.append(entry)
        elif klass == "small_molecule" and on_chain:
            on_chain_small.append(entry)

    if not has_any_non_nuisance:
        return {"category": "NUISANCE_ONLY", "ligand": None, "detail": "only_solvents_buffers_ions"}

    if on_chain_drug:
        best = max(on_chain_drug, key=lambda e: e["n_atoms"])
        return {"category": "VALID_DRUG_LIKE", "ligand": best,
                "detail": f"{best['resname']}({best['n_atoms']} atoms)"}

    if on_chain_metal:
        return {"category": "VALID_METAL", "ligand": on_chain_metal[0],
                "detail": on_chain_metal[0]["resname"]}

    if on_chain_small:
        best = max(on_chain_small, key=lambda e: e["n_atoms"])
        return {"category": "VALID_SMALL", "ligand": best,
                "detail": f"{best['resname']}({best['n_atoms']} atoms)"}

    if off_chain_drug:
        best = max(off_chain_drug, key=lambda e: e["n_atoms"])
        return {"category": "VALID_OFF_CHAIN", "ligand": best,
                "detail": f"{best['resname']} on chain {best['chain']}({best['n_atoms']} atoms)"}

    return {"category": "NO_LIGAND_ON_CHAIN", "ligand": None,
            "detail": "non-nuisance ligands exist but not on target chain"}


def main():
    # Read manifest
    chains = []
    with open(MANIFEST) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            # Format: 10dc_chainA
            m = re.match(r'^([0-9a-z]{4})_chain([A-Z])$', line, re.IGNORECASE)
            if m:
                chains.append((m.group(1).lower(), m.group(2).upper(), line))

    print(f"Manifest: {len(chains)} chains")

    # Deduplicate PDB IDs for download
    unique_pdbs = sorted(set(pdb for pdb, _, _ in chains))
    print(f"Unique PDB IDs: {len(unique_pdbs)}")

    # Parallel download
    print("Downloading holo PDBs (cached)...")
    holo_map = {}
    with ThreadPoolExecutor(max_workers=16) as pool:
        futures = {pool.submit(fetch_holo_pdb, pdb): pdb for pdb in unique_pdbs}
        done = 0
        for fut in as_completed(futures):
            pdb = futures[fut]
            holo_map[pdb] = fut.result()
            done += 1
            if done % 100 == 0:
                cached = sum(1 for v in holo_map.values() if v is not None)
                print(f"  {done}/{len(unique_pdbs)} downloaded ({cached} cached/OK)")

    cached = sum(1 for v in holo_map.values() if v is not None)
    failed = sum(1 for v in holo_map.values() if v is None)
    print(f"Downloads complete: {cached} OK, {failed} failed")

    # Classify each chain
    print("Classifying...")
    results = []
    for pdb_id, chain, chain_id in chains:
        holo = holo_map.get(pdb_id)
        r = classify_chain(pdb_id, chain, holo)
        r["chain_id"] = chain_id
        r["pdb_id"] = pdb_id
        r["chain"] = chain
        results.append(r)

    # Tally
    categories = {}
    for r in results:
        cat = r["category"]
        categories[cat] = categories.get(cat, 0) + 1

    valid_cats = {"VALID_DRUG_LIKE", "VALID_METAL", "VALID_SMALL", "VALID_OFF_CHAIN"}
    n_valid = sum(categories.get(c, 0) for c in valid_cats)
    n_invalid = len(results) - n_valid

    print(f"\n{'='*60}")
    print(f"  MANIFEST GROUND TRUTH AUDIT")
    print(f"  Total chains: {len(results)}")
    print(f"  Valid GT:     {n_valid} ({100*n_valid/len(results):.1f}%)")
    print(f"  Invalid GT:   {n_invalid} ({100*n_invalid/len(results):.1f}%)")
    print(f"{'='*60}")
    print(f"\n  {'Category':<25} {'Count':>6} {'Pct':>6}")
    print(f"  {'-'*40}")
    for cat in sorted(categories, key=lambda c: -categories[c]):
        n = categories[cat]
        marker = " *VALID*" if cat in valid_cats else ""
        print(f"  {cat:<25} {n:>6} {100*n/len(results):>5.1f}%{marker}")

    # Write full audit JSON
    audit_path = OUTPUT_DIR / "proteome_1000_gt_audit.json"
    with open(audit_path, "w") as f:
        json.dump({
            "n_chains": len(results),
            "n_valid": n_valid,
            "n_invalid": n_invalid,
            "categories": categories,
            "chains": results,
        }, f, indent=2, default=str)
    print(f"\nFull audit: {audit_path}")

    # Write valid-only manifest
    valid_manifest = OUTPUT_DIR / "proteome_1000_gt_valid.txt"
    valid_chains = [r["chain_id"] for r in results if r["category"] in valid_cats]
    with open(valid_manifest, "w") as f:
        for c in valid_chains:
            f.write(c + "\n")
    print(f"Valid manifest: {valid_manifest} ({len(valid_chains)} chains)")

    # Write stripped invalid chains for reference
    invalid_manifest = OUTPUT_DIR / "proteome_1000_gt_invalid.txt"
    invalid_chains = [r for r in results if r["category"] not in valid_cats]
    with open(invalid_manifest, "w") as f:
        for r in invalid_chains:
            f.write(f"{r['chain_id']}\t{r['category']}\t{r.get('detail','')}\n")
    print(f"Invalid list: {invalid_manifest} ({len(invalid_chains)} chains)")


if __name__ == "__main__":
    main()
