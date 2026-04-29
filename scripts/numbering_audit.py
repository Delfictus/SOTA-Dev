#!/usr/bin/env python3
"""
Numbering audit — characterize engine_id → pdb_resseq mapping for a
PRISM-4D substrate.

For each topology residue_id (engine numbering, 0-based in the JSON):
  1. Compute the centroid of the engine-mapped atoms via the topology
     `residue_to_atom_indices` map and the topology `positions` array.
  2. Find every PDB resseq within ±10 of the engine_id and grab its
     CA position (chain A, alt-loc A or blank).
  3. Pick the PDB resseq whose CA is closest to the engine-atom centroid.
  4. Classify CLEAN / AMBIGUOUS / BROKEN.

Output: CSV with columns
  engine_id, best_pdb_resseq, distance_A, alternate_pdb_resseqs, status
plus a leading summary block (commented-out lines starting with `#`)
that reports counts and the modal offset.

Pure post-processor. NO_POST_MD_LOOPS-compliant.
"""
import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


def load_topology(topology_json_path):
    with open(topology_json_path) as f:
        topo = json.load(f)
    positions = np.asarray(topo["positions"], dtype=np.float64)
    rai = {int(k): list(v) for k, v in topo["residue_to_atom_indices"].items()}
    return positions, rai


def load_pdb_ca(pdb_path, chain_filter="A"):
    """Returns dict resseq -> np.array([x,y,z]) (CA only, alt-loc A/blank)."""
    out = {}
    with open(pdb_path) as f:
        for line in f:
            if not line.startswith("ATOM"):
                continue
            try:
                atom_name = line[12:16].strip()
                alt_loc = line[16:17].strip()
                chain = line[21:22].strip()
                resseq = int(line[22:26])
                x = float(line[30:38])
                y = float(line[38:46])
                z = float(line[46:54])
            except (ValueError, IndexError):
                continue
            if atom_name != "CA":
                continue
            if alt_loc not in ("", "A"):
                continue
            if chain != chain_filter:
                continue
            out[resseq] = np.array([x, y, z])
    return out


def audit(topology_json, md_pdb, output_csv, search_radius=10):
    positions, rai = load_topology(topology_json)
    pdb_ca = load_pdb_ca(md_pdb)

    rows = []
    offset_counter = Counter()
    statuses = Counter()

    for engine_id, atom_idxs in sorted(rai.items()):
        if not atom_idxs:
            rows.append({
                "engine_id": engine_id,
                "best_pdb_resseq": None,
                "distance_A": float("nan"),
                "alternate_pdb_resseqs": "",
                "status": "BROKEN",
                "broken_reason": "no_atoms_in_topology",
            })
            statuses["BROKEN"] += 1
            continue

        coords = positions[np.asarray(atom_idxs)]
        centroid = coords.mean(axis=0)

        candidates = []
        for off in range(-search_radius, search_radius + 1):
            target = engine_id + off
            if target in pdb_ca:
                d = float(np.linalg.norm(pdb_ca[target] - centroid))
                candidates.append((d, target))
        candidates.sort()

        if not candidates:
            rows.append({
                "engine_id": engine_id,
                "best_pdb_resseq": None,
                "distance_A": float("nan"),
                "alternate_pdb_resseqs": "",
                "status": "BROKEN",
                "broken_reason": "no_pdb_ca_within_search_radius",
            })
            statuses["BROKEN"] += 1
            continue

        best_d, best_pdb = candidates[0]
        within_1A = [t for d, t in candidates[1:] if d - best_d < 1.0]

        if best_d > 5.0:
            status = "BROKEN"
            broken_reason = f"closest_ca_{best_d:.2f}A_exceeds_5A"
        elif within_1A:
            status = "AMBIGUOUS"
            broken_reason = ""
        elif best_d <= 2.0:
            status = "CLEAN"
            broken_reason = ""
        else:
            status = "AMBIGUOUS"
            broken_reason = f"closest_ca_{best_d:.2f}A_in_2-5A_band"

        statuses[status] += 1
        offset_counter[best_pdb - engine_id] += 1
        rows.append({
            "engine_id": engine_id,
            "best_pdb_resseq": best_pdb,
            "distance_A": best_d,
            "alternate_pdb_resseqs": ";".join(str(t) for t in within_1A),
            "status": status,
            "broken_reason": broken_reason,
        })

    modal_offset, modal_count = (
        offset_counter.most_common(1)[0]
        if offset_counter else (None, 0)
    )

    by_offset = defaultdict(list)
    for r in rows:
        if r["best_pdb_resseq"] is not None:
            off = r["best_pdb_resseq"] - r["engine_id"]
            by_offset[off].append(r["engine_id"])
    non_modal = sorted(
        (off, ids) for off, ids in by_offset.items()
        if off != modal_offset
    )

    summary_lines = [
        f"# topology_json     : {topology_json}",
        f"# md_pdb            : {md_pdb}",
        f"# search_radius     : ±{search_radius}",
        f"# n_engine_residues : {len(rai)}",
        f"# n_pdb_ca          : {len(pdb_ca)}",
        f"# CLEAN             : {statuses['CLEAN']}",
        f"# AMBIGUOUS         : {statuses['AMBIGUOUS']}",
        f"# BROKEN            : {statuses['BROKEN']}",
        f"# modal_offset      : {modal_offset:+d} ({modal_count} residues)"
        if modal_offset is not None else "# modal_offset      : (none)",
    ]
    if non_modal:
        summary_lines.append(
            f"# non_modal_offsets : "
            + ", ".join(
                f"{off:+d}:{len(ids)}res(eg.{ids[:3]})"
                for off, ids in non_modal[:10]
            )
        )
    summary_lines.append("#" + "-" * 80)

    headers = [
        "engine_id", "best_pdb_resseq", "distance_A",
        "alternate_pdb_resseqs", "status", "broken_reason",
    ]
    with open(output_csv, "w") as f:
        for line in summary_lines:
            f.write(line + "\n")
        f.write(",".join(headers) + "\n")
        for r in rows:
            d = r["distance_A"]
            d_str = "" if d != d else f"{d:.4f}"  # NaN handling
            f.write(",".join([
                str(r["engine_id"]),
                "" if r["best_pdb_resseq"] is None else str(r["best_pdb_resseq"]),
                d_str,
                r["alternate_pdb_resseqs"],
                r["status"],
                r["broken_reason"],
            ]) + "\n")

    print("\n".join(summary_lines))
    print(f"\nWrote {output_csv}: {len(rows)} rows")
    return statuses, modal_offset, non_modal


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--topology-json", required=True)
    ap.add_argument("--md-pdb", required=True)
    ap.add_argument("--output-csv", required=True)
    ap.add_argument("--search-radius", type=int, default=10)
    args = ap.parse_args()
    audit(args.topology_json, args.md_pdb, args.output_csv, args.search_radius)
    return 0


if __name__ == "__main__":
    sys.exit(main())
