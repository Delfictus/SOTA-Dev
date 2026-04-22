#!/usr/bin/env python3
"""Remap topology residue_ids to PDB author numbering via sequence alignment.

Uses Biopython Needleman-Wunsch global alignment to match the topology's amino
acid sequence against the PDB's ATOM-record sequence, then maps residue IDs
based on aligned positions.  Handles AMBER naming (HID/HIE/HIP→HIS, CYX→CYS).

Sources for the topology sequence (tried in order):
  1. lining_residues from binding_sites.json  (resid + resname pairs)
  2. Falls back to positional-index if no binding_sites supplied

Usage:
    python3 scripts/quarantine/remap_topo_resids.py \
        --pdb <pdb_file> \
        --topo <topology_json> \
        --binding-sites <binding_sites.json> \
        --output <output_json>
"""
import argparse
import json
import sys

from Bio.Align import PairwiseAligner

# 3-letter → 1-letter (canonical + AMBER variants)
THREE_TO_ONE = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
    "CYX": "C", "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H",
    "HID": "H", "HIE": "H", "HIP": "H", "ILE": "I", "LEU": "L",
    "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P", "SER": "S",
    "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
}


def extract_pdb_sequence(pdb_path):
    """Return [(resnum, resname_3letter, one_letter), ...] from ATOM records."""
    seq = []
    seen = set()
    with open(pdb_path) as f:
        for line in f:
            if line.startswith("ATOM"):
                resnum = int(line[22:26].strip())
                resname = line[17:20].strip()
                if resnum not in seen:
                    seen.add(resnum)
                    seq.append((resnum, resname, THREE_TO_ONE.get(resname, "X")))
    return seq


def extract_topo_resnames(bs_path):
    """Return {resid: (resname_3letter, one_letter)} from lining_residues."""
    with open(bs_path) as f:
        data = json.load(f)
    resmap = {}
    for site in data.get("sites", data.get("binding_sites", [])):
        for lr in site.get("lining_residues", []):
            rid = lr["resid"]
            rname = lr["resname"]
            resmap[rid] = (rname, THREE_TO_ONE.get(rname, "X"))
    return resmap


def build_topo_sequence_string(topo_resnames, n_positions):
    """Build 1-letter sequence for topology; 'X' for unknown positions."""
    seq = []
    for i in range(n_positions):
        if i in topo_resnames:
            seq.append(topo_resnames[i][1])
        else:
            seq.append("X")
    return "".join(seq)


def align_sequences(pdb_seq_1letter, topo_seq_1letter):
    """Needleman-Wunsch global alignment. Returns list of (pdb_idx, topo_idx) pairs."""
    aligner = PairwiseAligner()
    aligner.mode = "global"
    aligner.match_score = 2
    aligner.mismatch_score = -1
    # Treat X (unknown) as neutral — don't penalize or reward
    aligner.substitution_matrix = None
    aligner.open_gap_score = -5
    aligner.extend_gap_score = -0.5

    alignments = aligner.align(pdb_seq_1letter, topo_seq_1letter)
    best = alignments[0]
    return best


def build_mapping_from_alignment(alignment, pdb_seq, topo_n_positions):
    """Extract topo_idx → pdb_resnum mapping from the alignment."""
    mapping = {}
    aligned = alignment.aligned  # two arrays of (start, end) blocks

    pdb_blocks, topo_blocks = aligned
    for (ps, pe), (ts, te) in zip(pdb_blocks, topo_blocks):
        for pdb_i, topo_i in zip(range(ps, pe), range(ts, te)):
            pdb_resnum = pdb_seq[pdb_i][0]
            mapping[topo_i] = pdb_resnum

    return mapping


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pdb", required=True, help="PDB file with author numbering")
    parser.add_argument("--topo", required=True, help="Input topology JSON")
    parser.add_argument("--binding-sites", required=True,
                        help="binding_sites.json with lining_residues for topology sequence")
    parser.add_argument("--output", required=True, help="Output topology JSON")
    args = parser.parse_args()

    # --- Extract sequences ---
    pdb_seq = extract_pdb_sequence(args.pdb)
    pdb_1letter = "".join(r[2] for r in pdb_seq)
    print(f"PDB: {len(pdb_seq)} residues, range {pdb_seq[0][0]}-{pdb_seq[-1][0]}")

    topo_resnames = extract_topo_resnames(args.binding_sites)
    print(f"Topology: {len(topo_resnames)} residues with known identity (from lining_residues)")

    # Determine topology length from the topology file itself
    with open(args.topo) as f:
        topo = json.load(f)

    # Current residue_ids might already be remapped — find the actual range
    all_ids = [r["residue_id"] for r in topo["residues"]]
    topo_n = max(all_ids) + 1 if all_ids else 0

    # If topo IDs are already in PDB-resnum space (>40), reverse to 0-based first
    min_id = min(all_ids)
    if min_id > 30:
        print(f"Topology IDs appear pre-mapped (min={min_id}). Reversing to 0-based first.")
        # Build reverse map: pdb_resnum → 0-based index
        pdb_resnum_to_idx = {r[0]: i for i, r in enumerate(pdb_seq)}
        for res in topo["residues"]:
            old = res["residue_id"]
            if old in pdb_resnum_to_idx:
                res["residue_id"] = pdb_resnum_to_idx[old]
            else:
                print(f"  WARNING: residue_id {old} not found in PDB resnums", file=sys.stderr)
        all_ids = [r["residue_id"] for r in topo["residues"]]
        topo_n = max(all_ids) + 1
        print(f"  Reversed to 0-based: range {min(all_ids)}-{max(all_ids)}")

    # Build topology 1-letter sequence
    topo_1letter = build_topo_sequence_string(topo_resnames, topo_n)
    known_count = sum(1 for c in topo_1letter if c != "X")
    print(f"Topology sequence: {topo_n} positions, {known_count} with known AA, "
          f"{topo_n - known_count} unknown (X)")

    # --- Pairwise alignment ---
    print(f"\nRunning Needleman-Wunsch global alignment...")
    print(f"  PDB:  {len(pdb_1letter)} aa")
    print(f"  Topo: {len(topo_1letter)} aa")

    alignment = align_sequences(pdb_1letter, topo_1letter)
    print(f"  Alignment score: {alignment.score}")

    # --- Build mapping ---
    mapping = build_mapping_from_alignment(alignment, pdb_seq, topo_n)
    print(f"  Mapped positions: {len(mapping)} of {topo_n}")

    # --- Verify with known residues ---
    verified = 0
    mismatches = 0
    amber_equiv = 0
    for topo_id, (resname_3, resname_1) in topo_resnames.items():
        if topo_id in mapping:
            pdb_resnum = mapping[topo_id]
            # Find PDB resname at this resnum
            pdb_resname = None
            for rn, rname, _ in pdb_seq:
                if rn == pdb_resnum:
                    pdb_resname = rname
                    break
            if pdb_resname:
                pdb_1 = THREE_TO_ONE.get(pdb_resname, "X")
                topo_1 = THREE_TO_ONE.get(resname_3, "X")
                if pdb_1 == topo_1:
                    if pdb_resname != resname_3:
                        amber_equiv += 1
                    else:
                        verified += 1
                else:
                    mismatches += 1
                    print(f"  MISMATCH: topo[{topo_id}]={resname_3} → PDB[{pdb_resnum}]={pdb_resname}")

    print(f"\nSequence verification:")
    print(f"  Exact matches:     {verified}")
    print(f"  AMBER equivalents: {amber_equiv} (e.g. HID↔HIS)")
    print(f"  Mismatches:        {mismatches}")
    total_checked = verified + amber_equiv + mismatches
    print(f"  Identity:          {verified + amber_equiv}/{total_checked} "
          f"({100*(verified+amber_equiv)/total_checked:.1f}%)")

    if mismatches > 0:
        print(f"\n  WARNING: {mismatches} real mismatches found — alignment may be incorrect!")
        sys.exit(1)

    # --- Apply mapping ---
    remapped = 0
    unmapped = 0
    for res in topo["residues"]:
        old_id = res["residue_id"]
        if old_id in mapping:
            res["residue_id"] = mapping[old_id]
            remapped += 1
        else:
            print(f"  WARNING: topo residue_id {old_id} has no alignment mapping", file=sys.stderr)
            unmapped += 1

    with open(args.output, "w") as f:
        json.dump(topo, f, indent=2)

    print(f"\nResult:")
    print(f"  Remapped: {remapped}")
    print(f"  Unmapped: {unmapped}")
    print(f"  Output:   {args.output}")


if __name__ == "__main__":
    main()
