#!/usr/bin/env python3
"""PRISM-4D Residue ID Lookup.

Translates merged topology residue IDs back to original chain + PDB
residue numbers using the chain map from prism-merge-chains.py.

Usage:
    prism-lookup-residue.py <chain_map.json> <merged_resnum> [merged_resnum ...]
    prism-lookup-residue.py <chain_map.json> --residue-file <residues.txt>
"""
import json
import os
import sys


def load_chain_map(path):
    if not os.path.exists(path):
        print(f"ERROR: {path} not found", file=sys.stderr)
        sys.exit(1)
    with open(path) as f:
        return json.load(f)


def lookup(chain_map, merged_resnum):
    """Translate a merged residue number back to original chain + resnum."""
    for chain in chain_map["chains"]:
        ms = chain["merged_resnum_start"]
        me = chain["merged_resnum_end"]
        if ms <= merged_resnum <= me:
            offset = merged_resnum - ms
            orig = chain["original_resnum_start"] + offset
            return {
                "merged": merged_resnum,
                "chain": chain["original_chain"],
                "original_resnum": orig,
                "source_file": chain["source_file"],
            }
    return None


def main():
    if len(sys.argv) < 3:
        print("Usage: prism-lookup-residue.py <chain_map.json> <resnum> [resnum ...]",
              file=sys.stderr)
        print("       prism-lookup-residue.py <chain_map.json> --residue-file <file>",
              file=sys.stderr)
        sys.exit(1)

    cm = load_chain_map(sys.argv[1])

    # Collect residue numbers
    resnums = []
    if sys.argv[2] == "--residue-file":
        if len(sys.argv) < 4:
            print("ERROR: --residue-file requires a file path", file=sys.stderr)
            sys.exit(1)
        with open(sys.argv[3]) as f:
            for line in f:
                line = line.strip()
                if line and line.isdigit():
                    resnums.append(int(line))
    else:
        for arg in sys.argv[2:]:
            try:
                resnums.append(int(arg))
            except ValueError:
                print(f"WARNING: Skipping non-integer: {arg}", file=sys.stderr)

    for rn in resnums:
        result = lookup(cm, rn)
        if result:
            print(f"merged {rn} → chain {result['chain']}, "
                  f"original PDB residue {result['original_resnum']} "
                  f"(source: {result['source_file']})")
        else:
            print(f"merged {rn} → NOT FOUND in chain map")


if __name__ == "__main__":
    main()
