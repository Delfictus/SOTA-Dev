#!/usr/bin/env python3
"""
fill_prism_manifold_residues.py

For each PRISM4D target, reads the kcc_visualization.json, computes:
  - prism_manifold_residues: all residues within 8.0 Å of each site centroid
  - causal_driver_residues: driver_residue_id + candidate_residue_ids (top 3), deduplicated+sorted
  - centroids: actual xyz centroid per site

Writes updated config to the output path (overwrites).
"""

import json
import math
import sys
from pathlib import Path

CUTOFF = 8.0  # Angstroms

TARGETS = {
    "KRAS_G12C": {
        "kcc_json": "/mnt/storage/prism-outputs/runs/KRAS_G12C_chainA_20260512_194818/KRAS_G12C_chainA.kcc_visualization.json",
        "sites": [4008, 4003, 7, 6, 4, 0],
        "offset": 0,
        "config_name": "KRAS_G12C"
    },
    "STING": {
        "kcc_json": "/mnt/storage/prism-outputs/runs/STING_chainA_20260512_202612/STING_chainA.kcc_visualization.json",
        "sites": [3526, 3513, 3008, 3010, 4513, 516],
        "offset": 3,
        "config_name": "STING"
    },
    "AKT1": {
        "kcc_json": "/mnt/storage/prism-outputs/runs/AKT1_chainA_20260512_203906/AKT1_chainA.kcc_visualization.json",
        "sites": [514, 5524, 6, 4001, 3526, 515, 11],
        "offset": 1,
        "config_name": "AKT1"
    },
    "MCL1": {
        "kcc_json": "/mnt/storage/prism-outputs/runs/MCL1_chainA_20260512_194006/MCL1_chainA.kcc_visualization.json",
        "sites": [3513, 3, 4003, 1],
        "offset": 170,
        "config_name": "MCL1"
    },
    "p53_Y220C": {
        "kcc_json": "/mnt/storage/prism-outputs/runs/p53_Y220C_chainA_20260512_195447/p53_Y220C_chainA.kcc_visualization.json",
        "sites": [1013, 3017, 1012, 3009, 4000],
        "offset": 95,
        "config_name": "p53_Y220C"
    },
    "TEAD3": {
        "kcc_json": "/mnt/storage/prism-outputs/runs/TEAD3_chainA_20260512_200421/TEAD3_chainA.kcc_visualization.json",
        "sites": [7],
        "offset": 218,
        "config_name": "TEAD3"
    },
    "TRPV1": {
        "kcc_json": "/mnt/storage/prism-outputs/runs/TRPV1_chainA_20260512_212518/TRPV1_chainA.kcc_visualization.json",
        "sites": [4526, 3526, 3001, 1550, 1547, 1548],
        "offset": 110,
        "config_name": "TRPV1"
    },
    "Kv31": {
        "kcc_json": "/mnt/storage/prism-outputs/runs/Kv31_chainA_primary_20260512_210836/Kv31_chainA.kcc_visualization.json",
        "sites": [523, 3520, 1534, 1531, 4002, 4542, 4543],
        "offset": 6,
        "config_name": "Kv3_1"
    },
    "GLP1R": {
        "kcc_json": "/mnt/storage/prism-outputs/runs/GLP1R_chainA_20260512_201334/GLP1R_chainA.kcc_visualization.json",
        "sites": [],
        "offset": 10,
        "config_name": "GLP1R_negative"
    }
}

CONFIG_IN  = "/home/diddy/Downloads/PRISM4D_figures_and_pymol_package/pymol/prism4d_targets_config_filled_pdbs.json"
CONFIG_OUT = "/home/diddy/Downloads/PRISM4D_figures_and_pymol_package/pymol/prism4d_targets_config_filled_pdbs.json"


def dist3(a, b):
    return math.sqrt((a[0]-b[0])**2 + (a[1]-b[1])**2 + (a[2]-b[2])**2)


def process_target(name, cfg):
    kcc_path = cfg["kcc_json"]
    offset   = cfg["offset"]
    site_ids = cfg["sites"]

    print(f"\n=== {name} ===")
    print(f"  KCC JSON: {kcc_path}")

    if not Path(kcc_path).exists():
        print(f"  ERROR: file not found: {kcc_path}", file=sys.stderr)
        return None

    with open(kcc_path) as f:
        d = json.load(f)

    residues = d["residues"]     # list of residue dicts
    sites    = d["sites"]        # list of site dicts

    # Build residue lookup: residue_id -> ca_position
    res_lookup = {}
    for r in residues:
        res_lookup[r["residue_id"]] = r["ca_position"]

    # Build site lookup: id -> site dict
    site_lookup = {s["id"]: s for s in sites}

    if not site_ids:
        print(f"  No sites to process (negative control).")
        return {
            "prism_manifold_residues": [],
            "causal_driver_residues": [],
            "centroids": []
        }

    all_manifold_ids  = set()
    all_driver_ids    = set()
    centroids_out     = []
    missing_sites     = []

    for sid in site_ids:
        if sid not in site_lookup:
            missing_sites.append(sid)
            print(f"  WARN: site {sid} not found in JSON (skipped)")
            continue

        site = site_lookup[sid]
        centroid = site["centroid"]  # [x, y, z]
        centroids_out.append({"site_id": sid, "xyz": centroid})

        # --- lining residues within 8 Å ---
        manifold_kcc_ids = []
        for r in residues:
            ca = r["ca_position"]
            if dist3(ca, centroid) <= CUTOFF:
                manifold_kcc_ids.append(r["residue_id"])
        all_manifold_ids.update(manifold_kcc_ids)

        # --- causal drivers: driver + top 3 candidates ---
        kcc = site["kcc"]
        driver = kcc.get("driver_residue_id")
        candidates = kcc.get("candidate_residue_ids", [])
        # take top 3 candidates (list is already ordered by confidence)
        top3 = candidates[:3]
        driver_ids = set(top3)
        if driver is not None:
            driver_ids.add(driver)
        all_driver_ids.update(driver_ids)

        # per-site summary
        manifold_pdb = sorted([rid + offset for rid in manifold_kcc_ids])
        driver_pdb   = sorted([rid + offset for rid in driver_ids])
        print(f"  site {sid}: centroid={[round(x,2) for x in centroid]}")
        print(f"    lining residues ({len(manifold_kcc_ids)} within {CUTOFF}Å): pdb {manifold_pdb}")
        print(f"    causal drivers (kcc_ids {sorted(driver_ids)}): pdb {driver_pdb}")

    # Convert all IDs to PDB resi strings
    manifold_pdb_sorted = sorted([str(rid + offset) for rid in all_manifold_ids])
    driver_pdb_sorted   = sorted([str(rid + offset) for rid in all_driver_ids], key=lambda x: int(x))

    print(f"  TOTAL manifold residues (all sites): {len(manifold_pdb_sorted)} residues")
    print(f"  TOTAL causal driver residues (all sites): {len(driver_pdb_sorted)} residues")

    return {
        "prism_manifold_residues": [{"chain": "A", "resi": manifold_pdb_sorted}],
        "causal_driver_residues":  [{"chain": "A", "resi": driver_pdb_sorted}],
        "centroids": centroids_out
    }


def main():
    with open(CONFIG_IN) as f:
        config = json.load(f)

    # Build lookup: config_name -> target entry
    config_by_name = {}
    for t in config["targets"]:
        config_by_name[t["name"]] = t

    for target_key, cfg in TARGETS.items():
        result = process_target(target_key, cfg)
        if result is None:
            continue

        cname = cfg["config_name"]
        if cname not in config_by_name:
            print(f"  WARN: config_name '{cname}' not found in config targets", file=sys.stderr)
            continue

        entry = config_by_name[cname]
        entry["prism_manifold_residues"] = result["prism_manifold_residues"]
        entry["causal_driver_residues"]  = result["causal_driver_residues"]
        entry["centroids"]               = result["centroids"]

    with open(CONFIG_OUT, "w") as f:
        json.dump(config, f, indent=2)
    print(f"\nWrote: {CONFIG_OUT}")


if __name__ == "__main__":
    main()
