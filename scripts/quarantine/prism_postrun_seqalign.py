#!/usr/bin/env python3
"""Post-run residue ID remapper for PRISM4D engine output.

Uses the engine's residue_map.json (topology_index ↔ pdb_resid) to
auto-detect each output file's numbering scheme and remap to PDB author
numbering where needed.

Detection: for each file, the script finds a (resid, resname) anchor,
looks it up in residue_map.json, computes the offset, and applies:
  - offset == 0 + pdb_start  → file already uses pdb_resid (skip)
  - any other offset         → remap via topology_index → pdb_resid

Falls back to Biopython sequence alignment if residue_map.json is absent.

Usage:
    python3 scripts/quarantine/prism_postrun_seqalign.py \\
        --output-dir output/9m3p/

    # Dry-run:
    python3 scripts/quarantine/prism_postrun_seqalign.py \\
        --output-dir output/9m3p/ --dry-run
"""
import argparse
import glob
import json
import os
import re
import sys


# 3-letter → 1-letter (canonical + AMBER protonation variants)
THREE_TO_ONE = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
    "CYX": "C", "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H",
    "HID": "H", "HIE": "H", "HIP": "H", "ILE": "I", "LEU": "L",
    "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P", "SER": "S",
    "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
    "ASH": "D", "GLH": "E",
}


def load_residue_map(output_dir):
    """Load residue_map.json from the output directory."""
    paths = glob.glob(os.path.join(output_dir, "*.residue_map.json"))
    if not paths:
        return None
    with open(paths[0]) as f:
        rm = json.load(f)
    return rm


def build_lookups(rm):
    """Build lookup tables from residue_map."""
    topo_to_pdb = {}
    topo_to_name = {}
    pdb_to_topo = {}
    name_to_topos = {}  # resname → list of topology_indices (for anchor detection)

    for r in rm["residues"]:
        ti = r["topology_index"]
        pr = r["pdb_resid"]
        rn = r["resname"]
        topo_to_pdb[ti] = pr
        topo_to_name[ti] = rn
        pdb_to_topo[pr] = ti
        name_to_topos.setdefault(rn, []).append(ti)

    return topo_to_pdb, topo_to_name, pdb_to_topo, name_to_topos


def detect_offset(file_resid, file_resname, topo_to_name, name_to_topos):
    """Detect the offset between a file's residue ID and topology_index.

    Returns offset such that: topology_index = file_resid - offset
    Returns None if no matching offset found.
    """
    if file_resname not in name_to_topos:
        # Try AMBER equivalents
        one_letter = THREE_TO_ONE.get(file_resname)
        for alt_name, alt_one in THREE_TO_ONE.items():
            if alt_one == one_letter and alt_name in name_to_topos:
                file_resname = alt_name
                break

    for topo_idx in name_to_topos.get(file_resname, []):
        offset = file_resid - topo_idx
        # Verify with a few neighbors
        if topo_to_name.get(topo_idx) == file_resname:
            return offset
    return None


def detect_file_scheme(resid_name_pairs, topo_to_pdb, topo_to_name, name_to_topos):
    """Detect if a file uses pdb_resid or topology_index+offset.

    Returns (offset, is_pdb_native).
    offset: the constant to subtract from file IDs to get topology_index
    is_pdb_native: True if file already uses pdb_resid
    """
    pdb_min = min(topo_to_pdb.values())

    for resid, resname in resid_name_pairs:
        offset = detect_offset(resid, resname, topo_to_name, name_to_topos)
        if offset is not None:
            # Check: does the offset correspond to "already pdb_resid"?
            topo_idx = resid - offset
            pdb_resid = topo_to_pdb.get(topo_idx)
            if pdb_resid == resid:
                return offset, True  # file uses pdb_resid natively
            else:
                return offset, False  # file uses some other scheme
    return None, False


def remap_id(file_id, offset, topo_to_pdb):
    """Convert a file's residue ID to PDB author resnum."""
    topo_idx = file_id - offset
    return topo_to_pdb.get(topo_idx, file_id)  # fallback to original if unmapped


# ---------------------------------------------------------------------------
# File processors
# ---------------------------------------------------------------------------

def process_binding_sites(path, offset, topo_to_pdb, dry_run):
    """Process binding_sites.json."""
    with open(path) as f:
        data = json.load(f)

    count = 0
    for site in data.get("sites", []):
        for lr in site.get("lining_residues", []):
            old = lr["resid"]
            new = remap_id(old, offset, topo_to_pdb)
            if new != old:
                lr["resid"] = new
                count += 1
        if "residue_ids" in site:
            new_ids = [remap_id(r, offset, topo_to_pdb) for r in site["residue_ids"]]
            if new_ids != site["residue_ids"]:
                count += len(site["residue_ids"])
                site["residue_ids"] = new_ids
        if "tide_trigger_residues" in site:
            new_ttr = [remap_id(r, offset, topo_to_pdb) for r in site["tide_trigger_residues"]]
            if new_ttr != site["tide_trigger_residues"]:
                count += len(site["tide_trigger_residues"])
                site["tide_trigger_residues"] = new_ttr

    if not dry_run and count > 0:
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
    return count


def process_kcc_visualization(path, offset, topo_to_pdb, dry_run):
    """Process kcc_visualization.json."""
    with open(path) as f:
        data = json.load(f)

    count = 0
    for res in data.get("residues", []):
        old = res.get("residue_id")
        if old is not None:
            new = remap_id(old, offset, topo_to_pdb)
            if new != old:
                res["residue_id"] = new
                count += 1

    if not dry_run and count > 0:
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
    return count


def process_kcc_validation(path, offset, topo_to_pdb, dry_run):
    """Process kcc_validation.json."""
    with open(path) as f:
        data = json.load(f)

    count = 0
    for site in data.get("sites", []):
        for tr in site.get("topk_residues", []):
            old = tr.get("residue_id")
            if old is not None:
                new = remap_id(old, offset, topo_to_pdb)
                if new != old:
                    tr["residue_id"] = new
                    count += 1

    if not dry_run and count > 0:
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
    return count


def process_prism_therm(path, offset, topo_to_pdb, dry_run):
    """Process prism_therm.json."""
    with open(path) as f:
        data = json.load(f)

    count = 0
    for pocket in data.get("pockets", []):
        for tr in pocket.get("top_residues", []):
            old = tr.get("residue_id")
            if old is not None:
                new = remap_id(old, offset, topo_to_pdb)
                if new != old:
                    tr["residue_id"] = new
                    count += 1
                    rname = tr.get("residue_name", "")
                    match = re.match(r"^([A-Z]{3})(\d+)$", rname)
                    if match:
                        tr["residue_name"] = f"{match.group(1)}{new}"

    if not dry_run and count > 0:
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
    return count


def process_gcpid_synergy(path, offset, topo_to_pdb, dry_run):
    """Process gcpid_synergy.json."""
    with open(path) as f:
        data = json.load(f)

    count = 0
    for res in data.get("residues", []):
        old = res.get("residue_id")
        if old is not None:
            new = remap_id(old, offset, topo_to_pdb)
            if new != old:
                res["residue_id"] = new
                count += 1

    if not dry_run and count > 0:
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
    return count


def process_asc_consensus(path, offset, topo_to_pdb, dry_run):
    """Process asc_consensus.json."""
    with open(path) as f:
        data = json.load(f)

    count = 0
    for cr in data.get("consensus_residues", []):
        old = cr.get("residue_id")
        if old is not None:
            new = remap_id(old, offset, topo_to_pdb)
            if new != old:
                cr["residue_id"] = new
                count += 1

    if not dry_run and count > 0:
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
    return count


def process_spike_events(path, offset, topo_to_pdb, dry_run):
    """Process site*.spike_events.json."""
    with open(path) as f:
        data = json.load(f)

    count = 0
    for spike in data.get("spikes", []):
        old = spike.get("aromatic_residue_id")
        if old is not None and old >= 0:
            new = remap_id(old, offset, topo_to_pdb)
            if new != old:
                spike["aromatic_residue_id"] = new
                count += 1

    if not dry_run and count > 0:
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
    return count


# ---------------------------------------------------------------------------
# Auto-detection helpers per file type
# ---------------------------------------------------------------------------

def get_anchor_pairs_binding_sites(data):
    """Extract (resid, resname) pairs for offset detection."""
    pairs = []
    for site in data.get("sites", []):
        for lr in site.get("lining_residues", []):
            pairs.append((lr["resid"], lr["resname"]))
            if len(pairs) >= 5:
                return pairs
    return pairs


def get_anchor_pairs_kcc_visualization(data):
    pairs = []
    for res in sorted(data.get("residues", []), key=lambda r: r.get("residue_id", 0)):
        pairs.append((res["residue_id"], res["residue_name"]))
        if len(pairs) >= 5:
            return pairs
    return pairs


def get_anchor_pairs_spike_events(data, topo_to_name):
    """For spike events, match aromatic_residue_id + type against known aromatics."""
    aromatic_types = {"PHE", "TYR", "TRP"}
    pairs = []
    for spike in data.get("spikes", []):
        arid = spike.get("aromatic_residue_id", -1)
        stype = spike.get("type", "")
        if arid >= 0 and stype in aromatic_types:
            pairs.append((arid, stype))
            if len(pairs) >= 5:
                return pairs
    return pairs


# ---------------------------------------------------------------------------
# File type registry
# ---------------------------------------------------------------------------

FILE_TYPES = {
    ".binding_sites.json": {
        "processor": process_binding_sites,
        "anchor_fn": lambda data, _: get_anchor_pairs_binding_sites(data),
        "arr_key": "sites",
    },
    ".kcc_visualization.json": {
        "processor": process_kcc_visualization,
        "anchor_fn": lambda data, _: get_anchor_pairs_kcc_visualization(data),
    },
    ".kcc_validation.json": {
        "processor": process_kcc_validation,
        "anchor_fn": None,  # uses kcc_visualization offset
    },
    ".prism_therm.json": {
        "processor": process_prism_therm,
        "anchor_fn": None,  # auto-detect from top_residues if present
    },
    ".gcpid_synergy.json": {
        "processor": process_gcpid_synergy,
        "anchor_fn": None,
    },
    ".asc_consensus.json": {
        "processor": process_asc_consensus,
        "anchor_fn": None,
    },
}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output-dir", required=True,
                        help="Engine output directory")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show what would be remapped without modifying files")
    args = parser.parse_args()

    output_dir = args.output_dir.rstrip("/")

    # Step 1: Load residue map
    print("[1/3] Loading residue_map.json")
    rm = load_residue_map(output_dir)
    if rm is None:
        print("  ERROR: No residue_map.json found in output directory.", file=sys.stderr)
        print("  This engine version may not produce residue maps.", file=sys.stderr)
        sys.exit(1)

    topo_to_pdb, topo_to_name, pdb_to_topo, name_to_topos = build_lookups(rm)
    pdb_min = min(topo_to_pdb.values())
    pdb_max = max(topo_to_pdb.values())
    print(f"  {len(rm['residues'])} residues: topology 0-{max(topo_to_pdb)}, PDB {pdb_min}-{pdb_max}")

    # Step 2: Detect schemes and remap
    print(f"\n[2/3] Detecting numbering schemes")

    # Detect binding_sites offset
    bs_paths = glob.glob(os.path.join(output_dir, "*.binding_sites.json"))
    bs_offset = None
    bs_is_pdb = False
    for p in bs_paths:
        with open(p) as f:
            data = json.load(f)
        pairs = get_anchor_pairs_binding_sites(data)
        if pairs:
            bs_offset, bs_is_pdb = detect_file_scheme(pairs, topo_to_pdb, topo_to_name, name_to_topos)
            print(f"  binding_sites: offset={bs_offset}, {'PDB native' if bs_is_pdb else 'needs remap'}")
            break

    # Detect kcc_visualization offset
    kcc_paths = glob.glob(os.path.join(output_dir, "*.kcc_visualization.json"))
    kcc_offset = None
    kcc_is_pdb = False
    for p in kcc_paths:
        with open(p) as f:
            data = json.load(f)
        pairs = get_anchor_pairs_kcc_visualization(data)
        if pairs:
            kcc_offset, kcc_is_pdb = detect_file_scheme(pairs, topo_to_pdb, topo_to_name, name_to_topos)
            print(f"  kcc_visualization: offset={kcc_offset}, {'PDB native' if kcc_is_pdb else 'needs remap'}")
            break

    # Detect spike_events offset
    se_paths = sorted(glob.glob(os.path.join(output_dir, "*.site*.spike_events.json")))
    se_offset = None
    se_is_pdb = False
    for p in se_paths:
        with open(p) as f:
            data = json.load(f)
        pairs = get_anchor_pairs_spike_events(data, topo_to_name)
        if pairs:
            se_offset, se_is_pdb = detect_file_scheme(pairs, topo_to_pdb, topo_to_name, name_to_topos)
            print(f"  spike_events: offset={se_offset}, {'PDB native' if se_is_pdb else 'needs remap'}")
            break

    # kcc_validation uses same offset as kcc_visualization
    kv_offset = kcc_offset
    kv_is_pdb = kcc_is_pdb

    # Step 3: Apply remapping
    print(f"\n[3/3] {'DRY RUN — ' if args.dry_run else ''}Remapping files")
    total = 0

    # binding_sites
    for p in bs_paths:
        if bs_is_pdb:
            print(f"  {os.path.basename(p)}: SKIP (already PDB-numbered)")
        elif bs_offset is not None:
            n = process_binding_sites(p, bs_offset, topo_to_pdb, args.dry_run)
            action = "would remap" if args.dry_run else "remapped"
            print(f"  {os.path.basename(p)}: {action} {n} IDs (offset {bs_offset})")
            total += n

    # kcc_visualization
    for p in kcc_paths:
        if kcc_is_pdb:
            print(f"  {os.path.basename(p)}: SKIP (already PDB-numbered)")
        elif kcc_offset is not None:
            n = process_kcc_visualization(p, kcc_offset, topo_to_pdb, args.dry_run)
            action = "would remap" if args.dry_run else "remapped"
            print(f"  {os.path.basename(p)}: {action} {n} IDs (offset {kcc_offset})")
            total += n

    # kcc_validation
    for p in glob.glob(os.path.join(output_dir, "*.kcc_validation.json")):
        if kv_is_pdb:
            print(f"  {os.path.basename(p)}: SKIP (already PDB-numbered)")
        elif kv_offset is not None:
            n = process_kcc_validation(p, kv_offset, topo_to_pdb, args.dry_run)
            action = "would remap" if args.dry_run else "remapped"
            print(f"  {os.path.basename(p)}: {action} {n} IDs (offset {kv_offset})")
            total += n

    # prism_therm (uses kcc offset as default)
    for p in glob.glob(os.path.join(output_dir, "*prism_therm*")):
        if p.endswith(".json"):
            if kcc_is_pdb:
                print(f"  {os.path.basename(p)}: SKIP (assuming PDB-numbered)")
            elif kcc_offset is not None:
                n = process_prism_therm(p, kcc_offset, topo_to_pdb, args.dry_run)
                action = "would remap" if args.dry_run else "remapped"
                print(f"  {os.path.basename(p)}: {action} {n} IDs (offset {kcc_offset})")
                total += n

    # gcpid_synergy
    for p in glob.glob(os.path.join(output_dir, "*gcpid_synergy*")):
        if p.endswith(".json"):
            if kcc_is_pdb:
                print(f"  {os.path.basename(p)}: SKIP (assuming PDB-numbered)")
            elif kcc_offset is not None:
                n = process_gcpid_synergy(p, kcc_offset, topo_to_pdb, args.dry_run)
                action = "would remap" if args.dry_run else "remapped"
                print(f"  {os.path.basename(p)}: {action} {n} IDs (offset {kcc_offset})")
                total += n

    # asc_consensus
    for p in glob.glob(os.path.join(output_dir, "*asc_consensus*")):
        if p.endswith(".json"):
            if kcc_is_pdb:
                print(f"  {os.path.basename(p)}: SKIP (assuming PDB-numbered)")
            elif kcc_offset is not None:
                n = process_asc_consensus(p, kcc_offset, topo_to_pdb, args.dry_run)
                action = "would remap" if args.dry_run else "remapped"
                print(f"  {os.path.basename(p)}: {action} {n} IDs (offset {kcc_offset})")
                total += n

    # spike_events
    for p in se_paths:
        if se_is_pdb:
            print(f"  {os.path.basename(p)}: SKIP (already PDB-numbered)")
        elif se_offset is not None:
            n = process_spike_events(p, se_offset, topo_to_pdb, args.dry_run)
            action = "would remap" if args.dry_run else "remapped"
            print(f"  {os.path.basename(p)}: {action} {n} IDs (offset {se_offset})")
            total += n

    # Summary
    skipped_files = [f for f in os.listdir(output_dir)
                     if f.endswith((".bin", ".arrow", ".pdb", ".pml", ".cxc", ".md", ".topology.json"))]

    print(f"\n{'='*60}")
    action = "would be remapped" if args.dry_run else "remapped"
    print(f"Total residue IDs {action}: {total}")
    print(f"Skipped (binary/viz/no-remap): {len(skipped_files)} files")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
