#!/usr/bin/env python3
"""PRISM-4D Multichain PDB Merger.

Merges multiple single-chain PDBs into one file with sequential residue
numbering. Saves a chain map for translating engine output residue IDs
back to original chain + PDB residue numbers.

Usage:
    prism-merge-chains.py chainA.pdb chainB.pdb -o merged.pdb [--chain-map map.json]
"""
import argparse
import json
import os
import sys
from collections import OrderedDict
from datetime import datetime, timezone


def parse_chain_pdb(path):
    """Parse ATOM records from a PDB, return list of (line, resnum, resname, chain)."""
    records = []
    if not os.path.exists(path):
        print(f"ERROR: {path} not found", file=sys.stderr)
        sys.exit(1)

    with open(path) as f:
        for line in f:
            if not line.startswith("ATOM"):
                continue
            try:
                chain = line[21].strip() or "A"
                resnum = int(line[22:26])
                resname = line[17:20].strip()
                records.append((line, resnum, resname, chain))
            except (ValueError, IndexError):
                continue

    if not records:
        print(f"ERROR: {path} has 0 ATOM records", file=sys.stderr)
        sys.exit(1)

    return records


def main():
    parser = argparse.ArgumentParser(description="Merge multichain PDBs for PRISM-4D")
    parser.add_argument("inputs", nargs="+", help="Input PDB files (one per chain)")
    parser.add_argument("-o", "--output", required=True, help="Output merged PDB")
    parser.add_argument("--chain-map", default=None, help="Chain map JSON output path")
    args = parser.parse_args()

    if args.chain_map is None:
        base = os.path.splitext(args.output)[0]
        args.chain_map = base + ".chain_map.json"

    # Parse all input chains
    all_chains = []
    for path in args.inputs:
        records = parse_chain_pdb(path)
        # Get unique residues in order
        seen = OrderedDict()
        for line, resnum, resname, chain in records:
            seen[resnum] = (resname, chain)
        all_chains.append({
            "source_file": os.path.basename(path),
            "records": records,
            "residues": seen,
            "original_chain": records[0][3] if records else "?",
        })

    # Compute offsets — sequential numbering across chains
    next_resnum = 1
    chain_map_entries = []

    for ci, chain_data in enumerate(all_chains):
        residues = chain_data["residues"]
        orig_resnums = sorted(residues.keys())
        orig_start = orig_resnums[0]
        orig_end = orig_resnums[-1]
        n_res = len(orig_resnums)

        # Build mapping: original resnum -> merged resnum
        resnum_map = {}
        merged_start = next_resnum
        for i, orig in enumerate(orig_resnums):
            resnum_map[orig] = next_resnum + i
        merged_end = next_resnum + n_res - 1
        next_resnum = merged_end + 1

        chain_data["resnum_map"] = resnum_map
        chain_data["merged_start"] = merged_start
        chain_data["merged_end"] = merged_end

        chain_map_entries.append({
            "source_file": chain_data["source_file"],
            "original_chain": chain_data["original_chain"],
            "original_resnum_start": orig_start,
            "original_resnum_end": orig_end,
            "merged_resnum_start": merged_start,
            "merged_resnum_end": merged_end,
            "n_residues": n_res,
        })

    total_residues = next_resnum - 1

    # Write merged PDB
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    merged_resids = set()

    with open(args.output, "w") as f:
        for ci, chain_data in enumerate(all_chains):
            if ci > 0:
                f.write("TER\n")
            resnum_map = chain_data["resnum_map"]
            for line, orig_resnum, resname, chain in chain_data["records"]:
                new_resnum = resnum_map.get(orig_resnum, orig_resnum)
                merged_resids.add(new_resnum)
                # Rewrite: chain A (col 21), new resnum (cols 22-26)
                new_line = (
                    line[:21] + "A" + f"{new_resnum:>4d}" + line[26:]
                )
                f.write(new_line)
        f.write("TER\nEND\n")

    # Verify
    if len(merged_resids) != total_residues:
        print(f"ERROR: Merged PDB has {len(merged_resids)} unique residues "
              f"but expected {total_residues}", file=sys.stderr)
        sys.exit(1)

    # Save chain map
    chain_map = {
        "merged_pdb": os.path.basename(args.output),
        "created": datetime.now(timezone.utc).isoformat(),
        "chains": chain_map_entries,
        "total_residues": total_residues,
        "offset_formula": "Use merged_resnum_start/end to translate. "
                          "merged_resnum = original_resnum - original_start + merged_start",
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.chain_map)), exist_ok=True)
    with open(args.chain_map, "w") as f:
        json.dump(chain_map, f, indent=2)

    # Print summary
    print(f"Chain map for {os.path.basename(args.output)}:")
    for i, entry in enumerate(chain_map_entries):
        print(f"  [{i+1}] {entry['source_file']} (chain {entry['original_chain']}): "
              f"original {entry['original_resnum_start']}-{entry['original_resnum_end']} → "
              f"merged {entry['merged_resnum_start']}-{entry['merged_resnum_end']}  "
              f"({entry['n_residues']} residues)")
    print(f"  Total: {total_residues} residues")
    print(f"  Chain map: {args.chain_map}")


if __name__ == "__main__":
    main()
