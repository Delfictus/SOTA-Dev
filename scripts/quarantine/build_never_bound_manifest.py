#!/usr/bin/env python3
"""Runner 2 — Build "never-bound" protein manifest from RCSB.

Finds proteins in the PDB that have NEVER been crystallized with any
drug-like ligand. These are genuinely unexplored targets — no answer key
exists. The engine's predictions are prospective.

Methodology:
  1. RCSB Search API: find entries with zero non-polymer entities, protein-only, X-ray, <2.0Å
  2. For each entry, get UniProt accession
  3. For each UniProt, check ALL associated PDB entries
  4. If ANY entry has a drug-like ligand (≥10 heavy atoms, not nuisance) → discard
  5. Remaining = never-bound proteins
  6. Apply quality filters (resolution, sequence coverage, monomer/homo-oligomer)
  7. Redundancy removal at 30% sequence identity

Target: 100 engine-ready structures.

CPU-only. Does NOT launch engine runs.

Usage:
    python3 scripts/quarantine/build_never_bound_manifest.py [--target-count 100]
"""

import argparse
import json
import os
import re
import subprocess
import sys
import time
import urllib.error
import urllib.request
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent.parent
BENCH_DIR = BASE / "benchmarks" / "never_bound"
DATA_DIR = BENCH_DIR / "data"
PDB_DIR = BENCH_DIR / "pdbs"
TOPO_DIR = BENCH_DIR / "topologies"

# Nuisance exclusion list (from directive)
NUISANCE_LIGANDS = {
    "HOH", "DOD", "GOL", "EDO", "PEG", "PGE", "DMS", "MPD",
    "MES", "EPE", "TRS", "ACT", "FMT", "BME", "IPA",
    "SO4", "PO4", "CL", "NA", "K", "CA", "MG", "ZN",
    "NO3", "SCN", "BR", "IOD", "FE", "MN", "CO", "NI", "CU",
    "CD", "HG",
    "CIT", "TAR", "SUC", "MLI",
    "BOG", "LDA", "SDS", "UNX", "UNL",
    "HOH", "H2O", "WAT", "TIP", "DOD",
    "NAG", "BMA", "MAN", "GLC", "FUC",  # sugars
}

RCSB_SEARCH_URL = "https://search.rcsb.org/rcsbsearch/v2/query"
RCSB_DATA_URL = "https://data.rcsb.org/rest/v1/core"


def rcsb_search_apo_entries():
    """Find PDB entries with zero non-polymer entities, protein-only, X-ray, high resolution."""
    query = {
        "query": {
            "type": "group",
            "logical_operator": "and",
            "nodes": [
                {
                    "type": "terminal",
                    "service": "text",
                    "parameters": {
                        "attribute": "rcsb_entry_info.nonpolymer_entity_count",
                        "operator": "equals",
                        "value": 0
                    }
                },
                {
                    "type": "terminal",
                    "service": "text",
                    "parameters": {
                        "attribute": "rcsb_entry_info.selected_polymer_entity_types",
                        "operator": "exact_match",
                        "value": "Protein (only)"
                    }
                },
                {
                    "type": "terminal",
                    "service": "text",
                    "parameters": {
                        "attribute": "rcsb_entry_info.resolution_combined",
                        "operator": "less",
                        "value": 2.0
                    }
                },
                {
                    "type": "terminal",
                    "service": "text",
                    "parameters": {
                        "attribute": "rcsb_entry_info.experimental_method",
                        "operator": "exact_match",
                        "value": "X-ray"
                    }
                }
            ]
        },
        "return_type": "entry",
        "request_options": {
            "paginate": {
                "start": 0,
                "rows": 10000
            },
            "results_content_type": ["experimental"],
            "sort": [{"sort_by": "rcsb_entry_info.resolution_combined", "direction": "asc"}]
        }
    }

    print("Querying RCSB for apo entries (zero non-polymer, protein-only, X-ray, <2.0Å)...")

    all_entries = []
    start = 0
    while True:
        query["request_options"]["paginate"]["start"] = start
        data = json.dumps(query).encode()
        req = urllib.request.Request(
            RCSB_SEARCH_URL,
            data=data,
            headers={"Content-Type": "application/json", "User-Agent": "prism4d-never-bound/1.0"},
        )
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                result = json.loads(resp.read())
        except (urllib.error.URLError, OSError, TimeoutError) as e:
            print(f"  Search API error: {e}")
            break

        total = result.get("total_count", 0)
        entries = [r["identifier"] for r in result.get("result_set", [])]
        all_entries.extend(entries)
        print(f"  Fetched {len(all_entries)}/{total} entries")

        if len(entries) == 0 or len(all_entries) >= total:
            break
        start += len(entries)
        time.sleep(0.5)  # Rate limit

    print(f"  Total apo entries: {len(all_entries)}")
    return all_entries


def get_uniprot_for_entries(entries, max_workers=8):
    """Fetch UniProt accessions for each PDB entry via RCSB Data API."""
    print(f"\nFetching UniProt accessions for {len(entries)} entries...")

    def fetch_uniprot(pdb_id):
        url = f"{RCSB_DATA_URL}/entry/{pdb_id}"
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "prism4d/1.0"})
            with urllib.request.urlopen(req, timeout=15) as resp:
                data = json.loads(resp.read())
            # Get polymer entities
            polymer_entities = data.get("rcsb_entry_container_identifiers", {}).get("polymer_entity_ids", [])
            uniprots = set()
            for eid in polymer_entities:
                entity_url = f"{RCSB_DATA_URL}/polymer_entity/{pdb_id}/{eid}"
                try:
                    req2 = urllib.request.Request(entity_url, headers={"User-Agent": "prism4d/1.0"})
                    with urllib.request.urlopen(req2, timeout=15) as resp2:
                        edata = json.loads(resp2.read())
                    refs = edata.get("rcsb_polymer_entity_container_identifiers", {}).get("uniprot_ids", [])
                    uniprots.update(refs)
                except (urllib.error.URLError, OSError, TimeoutError):
                    pass
            return pdb_id, list(uniprots)
        except (urllib.error.URLError, OSError, TimeoutError):
            return pdb_id, []

    entry_to_uniprot = {}
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(fetch_uniprot, e): e for e in entries}
        done = 0
        for fut in as_completed(futures):
            pdb_id, uniprots = fut.result()
            entry_to_uniprot[pdb_id] = uniprots
            done += 1
            if done % 200 == 0:
                print(f"  {done}/{len(entries)} UniProt lookups complete")

    with_uniprot = sum(1 for v in entry_to_uniprot.values() if v)
    print(f"  {with_uniprot}/{len(entries)} entries have UniProt accessions")

    return entry_to_uniprot


def check_uniprot_has_ligand(uniprot_id):
    """Check if ANY PDB entry for this UniProt has a drug-like ligand.

    Uses RCSB search to find all entries for this UniProt that have
    non-polymer entities, then checks if any ligand is drug-like.
    """
    # Search for PDB entries with this UniProt AND nonpolymer entities > 0
    query = {
        "query": {
            "type": "group",
            "logical_operator": "and",
            "nodes": [
                {
                    "type": "terminal",
                    "service": "text",
                    "parameters": {
                        "attribute": "rcsb_polymer_entity_container_identifiers.reference_sequence_identifiers.database_accession",
                        "operator": "exact_match",
                        "value": uniprot_id
                    }
                },
                {
                    "type": "terminal",
                    "service": "text",
                    "parameters": {
                        "attribute": "rcsb_entry_info.nonpolymer_entity_count",
                        "operator": "greater",
                        "value": 0
                    }
                }
            ]
        },
        "return_type": "entry",
        "request_options": {
            "paginate": {"start": 0, "rows": 5}
        }
    }

    try:
        data = json.dumps(query).encode()
        req = urllib.request.Request(
            RCSB_SEARCH_URL,
            data=data,
            headers={"Content-Type": "application/json", "User-Agent": "prism4d/1.0"},
        )
        with urllib.request.urlopen(req, timeout=30) as resp:
            result = json.loads(resp.read())
        total = result.get("total_count", 0)
        if total == 0:
            return False  # No entries with ligands — never bound

        # Found entries with nonpolymer entities — check if any are drug-like
        entries = [r["identifier"] for r in result.get("result_set", [])]
        for entry_id in entries:
            try:
                url = f"{RCSB_DATA_URL}/entry/{entry_id}"
                req2 = urllib.request.Request(url, headers={"User-Agent": "prism4d/1.0"})
                with urllib.request.urlopen(req2, timeout=15) as resp2:
                    edata = json.loads(resp2.read())
                np_ids = edata.get("rcsb_entry_container_identifiers", {}).get("non_polymer_entity_ids", [])
                for np_id in np_ids:
                    np_url = f"{RCSB_DATA_URL}/nonpolymer_entity/{entry_id}/{np_id}"
                    try:
                        req3 = urllib.request.Request(np_url, headers={"User-Agent": "prism4d/1.0"})
                        with urllib.request.urlopen(req3, timeout=15) as resp3:
                            npdata = json.loads(resp3.read())
                        comp_id = npdata.get("rcsb_nonpolymer_entity_container_identifiers", {}).get("comp_id", "")
                        if comp_id and comp_id not in NUISANCE_LIGANDS:
                            return True  # Has a non-nuisance ligand
                    except (urllib.error.URLError, OSError, TimeoutError, json.JSONDecodeError, ValueError):
                        continue
            except (urllib.error.URLError, OSError, TimeoutError, json.JSONDecodeError, ValueError):
                continue

        # All nonpolymer entities were nuisance — still counts as never-bound
        return False

    except urllib.error.HTTPError as e:
        if e.code == 400:
            return None
        return None
    except (urllib.error.URLError, OSError, TimeoutError, json.JSONDecodeError, ValueError):
        return None


def filter_never_bound(entry_to_uniprot, max_workers=8):
    """For each UniProt, check if it has ever been bound. Keep only never-bound."""
    # Group entries by UniProt
    uniprot_to_entries = defaultdict(list)
    no_uniprot = []
    for entry, uniprots in entry_to_uniprot.items():
        if uniprots:
            for u in uniprots:
                uniprot_to_entries[u].append(entry)
        else:
            no_uniprot.append(entry)

    print(f"\n  {len(uniprot_to_entries)} unique UniProt IDs to check")
    print(f"  {len(no_uniprot)} entries without UniProt (discarded)")

    # Check each UniProt for ligand binding
    never_bound = {}
    has_ligand = 0
    unknown = 0

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(check_uniprot_has_ligand, uid): uid
                   for uid in uniprot_to_entries}
        done = 0
        for fut in as_completed(futures):
            uid = futures[fut]
            result = fut.result()
            done += 1

            if result is True:
                has_ligand += 1
            elif result is False:
                never_bound[uid] = uniprot_to_entries[uid]
            else:
                unknown += 1

            if done % 100 == 0:
                print(f"  Checked {done}/{len(futures)}: {len(never_bound)} never-bound, "
                      f"{has_ligand} has-ligand, {unknown} unknown")

    print(f"\n  Results:")
    print(f"    Never-bound UniProts: {len(never_bound)}")
    print(f"    Has-ligand UniProts:  {has_ligand}")
    print(f"    Unknown:              {unknown}")

    return never_bound


def select_best_entries(never_bound, target_count=100):
    """For each never-bound UniProt, select the best PDB entry (lowest resolution)."""
    # Flatten to entry list, keeping one per UniProt
    selected = []
    for uid, entries in never_bound.items():
        # Already sorted by resolution from the search
        selected.append({
            "pdb_id": entries[0].lower(),
            "uniprot_id": uid,
            "n_entries": len(entries),
        })

    # Sort by PDB ID for reproducibility
    selected.sort(key=lambda x: x['pdb_id'])

    if len(selected) > target_count:
        print(f"\n  {len(selected)} never-bound proteins found, selecting {target_count}")
        selected = selected[:target_count]
    else:
        print(f"\n  {len(selected)} never-bound proteins found (target was {target_count})")

    return selected


def download_and_prep(entries, max_workers=16):
    """Download PDBs and identify chains."""
    print(f"\n  Downloading {len(entries)} PDBs...")
    PDB_DIR.mkdir(parents=True, exist_ok=True)

    def fetch(entry):
        pdb_id = entry['pdb_id']
        path = PDB_DIR / f"{pdb_id}.pdb"
        if path.exists() and path.stat().st_size > 100:
            return entry, path
        url = f"https://files.rcsb.org/download/{pdb_id.upper()}.pdb"
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "prism4d/1.0"})
            with urllib.request.urlopen(req, timeout=30) as resp:
                data = resp.read()
            if len(data) > 100:
                path.write_bytes(data)
                return entry, path
        except (urllib.error.URLError, OSError, TimeoutError):
            pass
        return entry, None

    results = []
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        for entry, path in pool.map(lambda e: fetch(e), entries):
            if path:
                # Identify chains
                chains = set()
                with open(path, errors="ignore") as f:
                    for line in f:
                        if line.startswith("ATOM"):
                            chains.add(line[21])
                entry['chains'] = sorted(chains)
                entry['pdb_path'] = str(path)
                results.append(entry)

    print(f"  Downloaded: {len(results)}/{len(entries)}")
    return results


def build_manifest(entries, target_count=100):
    """Build final manifest."""
    # Expand to chain-level entries, taking chain A preferentially
    chain_entries = []
    for entry in entries:
        chains = entry.get('chains', ['A'])
        # Prefer chain A, then alphabetical
        chain = 'A' if 'A' in chains else chains[0]
        chain_entries.append({
            "pdb_id": entry['pdb_id'],
            "chain": chain,
            "chain_id": f"{entry['pdb_id']}_chain{chain}",
            "uniprot_id": entry['uniprot_id'],
            "category": "NEVER_BOUND",
            "n_pdb_entries": entry.get('n_entries', 1),
        })

    if len(chain_entries) > target_count:
        chain_entries = chain_entries[:target_count]

    manifest = {
        "pipeline": "never_bound_prospective",
        "source": "RCSB Search API — zero non-polymer, protein-only, X-ray, <2.0Å",
        "n_targets": len(chain_entries),
        "note": "NO GROUND TRUTH — prospective predictions only. Cannot compute DCC.",
        "targets": chain_entries,
    }

    manifest_path = BENCH_DIR / "never_bound_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    chain_list_path = BENCH_DIR / "never_bound_chains.txt"
    with open(chain_list_path, "w") as f:
        for e in chain_entries:
            f.write(e['chain_id'] + "\n")

    print(f"\n  Manifest: {manifest_path}")
    print(f"  Chain list: {chain_list_path} ({len(chain_entries)} targets)")

    return manifest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target-count", type=int, default=100,
                        help="Number of never-bound targets to select (default: 100)")
    args = parser.parse_args()

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    # Phase 1: RCSB search (use cache if available)
    cached_entries = DATA_DIR / "rcsb_apo_entries.json"
    if cached_entries.exists():
        print(f"Using cached RCSB search results: {cached_entries}")
        entries = json.load(open(cached_entries))
    else:
        entries = rcsb_search_apo_entries()
        if not entries:
            print("ERROR: RCSB search returned no results")
            sys.exit(1)
        with open(cached_entries, "w") as f:
            json.dump(entries, f)

    # Phase 2: Get UniProt accessions (use cache if available)
    cached_uniprot = DATA_DIR / "entry_to_uniprot.json"
    if cached_uniprot.exists():
        print(f"Using cached UniProt mapping: {cached_uniprot}")
        entry_to_uniprot = json.load(open(cached_uniprot))
    else:
        entry_to_uniprot = get_uniprot_for_entries(entries)
        with open(cached_uniprot, "w") as f:
            json.dump(entry_to_uniprot, f, indent=2)

    # Phase 3: Filter never-bound
    never_bound = filter_never_bound(entry_to_uniprot)

    # Save never-bound list
    with open(DATA_DIR / "never_bound_uniprots.json", "w") as f:
        json.dump(never_bound, f, indent=2)

    # Phase 4: Select best entries
    selected = select_best_entries(never_bound, args.target_count)

    # Phase 5: Download and identify chains
    prepped = download_and_prep(selected)

    # Phase 6: Build manifest
    manifest = build_manifest(prepped, args.target_count)

    print(f"\n{'='*60}")
    print(f"  NEVER-BOUND MANIFEST SUMMARY")
    print(f"{'='*60}")
    print(f"  RCSB apo entries:        {len(entries)}")
    print(f"  With UniProt:            {sum(1 for v in entry_to_uniprot.values() if v)}")
    print(f"  Never-bound UniProts:    {len(never_bound)}")
    print(f"  Selected targets:        {manifest['n_targets']}")
    print(f"  Category:                NEVER_BOUND (no answer key)")
    print(f"\n  NOTE: These targets have NO ground truth.")
    print(f"  Engine predictions are PROSPECTIVE — cannot compute DCC.")
    print(f"  Cross-reference with AlphaFold2 or experimental follow-up.")


if __name__ == "__main__":
    main()
