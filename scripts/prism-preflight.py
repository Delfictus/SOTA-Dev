#!/usr/bin/env python3
"""PRISM-4D Topology Preflight Validator.

Validates a topology JSON before the engine runs. Catches corrupted
topologies, missing fields, and known per-target requirements.

Usage:
    python3 scripts/prism-preflight.py <topology.json>

Exit 0 = PASS, Exit 1 = FAIL.
"""
import json
import os
import sys

# CYS requirements registry — True = CYS required, False = CYS not needed
CYS_REGISTRY = {
    "1bzj": True,   # PTP1B catalytic Cys215
    "1r3m": True,   # RNase disulfides
    "2iyt": False,  # Shikimate Kinase — no catalytic CYS
    "3uyi": True,   # confirmed CYS in clean PDB
    "1nna": False,  # Neuraminidase
    "1jwp": False,  # TEM-1 beta-lactamase
    "1p38": True,   # p38 MAP kinase
    "2hnp": True,   # EphB2
}

REQUIRED_KEYS = [
    "n_atoms", "n_residues", "positions", "masses",
    "residue_names", "residue_ids",
]


def preflight(topo_path):
    fails = []
    warns = []

    # File exists
    if not os.path.exists(topo_path):
        print(f"FAIL: {topo_path} not found")
        return 1

    # Valid JSON
    try:
        with open(topo_path) as f:
            topo = json.load(f)
    except json.JSONDecodeError as e:
        print(f"FAIL: Invalid JSON — {e}")
        return 1

    # Required keys
    for key in REQUIRED_KEYS:
        if key not in topo:
            fails.append(f"Missing required key: {key}")

    if fails:
        for f in fails:
            print(f"FAIL: {f}")
        return 1

    n_atoms = topo["n_atoms"]
    n_residues = topo.get("n_residues", 0)
    residue_names = topo.get("residue_names", [])
    positions = topo.get("positions", [])

    # Basic counts
    if n_residues <= 0:
        fails.append(f"n_residues = {n_residues} (must be > 0)")
    if n_atoms <= 0:
        fails.append(f"n_atoms = {n_atoms} (must be > 0)")

    # Atom density
    if n_residues > 0:
        density = n_atoms / n_residues
        if density < 5:
            fails.append(f"Atom density {density:.1f} < 5 (corrupted topology?)")

    # Positions length
    if len(positions) != n_atoms * 3:
        fails.append(f"positions length {len(positions)} != n_atoms*3 ({n_atoms * 3})")

    # Residue type diversity
    unique_types = set(residue_names)
    if len(unique_types) < 15:
        fails.append(f"Only {len(unique_types)} residue types (need >=15): {sorted(unique_types)}")

    # HIS protonation check
    his_variants = {"HID", "HIE", "HIP"} & unique_types
    if not his_variants and "HIS" in unique_types:
        warns.append("HIS present but no HID/HIE/HIP — AMBER protonation may not have been assigned")

    # CYS check against registry
    source_pdb = topo.get("source_pdb", topo_path)
    target_key = None
    for key in CYS_REGISTRY:
        if key in os.path.basename(source_pdb).lower() or key in os.path.basename(topo_path).lower():
            target_key = key
            break

    has_cys = "CYS" in unique_types
    if target_key and CYS_REGISTRY.get(target_key):
        if not has_cys:
            warns.append(f"CYS REQUIRED for {target_key} but not found in topology. "
                         f"Catalytic cysteine may have been mutated during AMBER prep.")
    elif target_key is None:
        warns.append(f"Target not in CYS registry. CYS present: {has_cys}")

    # Print results
    basename = os.path.basename(topo_path)
    print(f"Preflight: {basename}")
    print(f"  n_atoms: {n_atoms}")
    print(f"  n_residues: {n_residues}")
    print(f"  residue types: {len(unique_types)}")
    print(f"  CYS: {'present' if has_cys else 'absent'}")
    print(f"  HIS variants: {sorted(his_variants) if his_variants else 'none'}")

    for w in warns:
        print(f"  WARN: {w}")

    if fails:
        for f in fails:
            print(f"  FAIL: {f}")
        print("PREFLIGHT: FAIL")
        return 1

    print("PREFLIGHT: PASS")
    return 0


def main():
    if len(sys.argv) < 2:
        print("Usage: prism-preflight.py <topology.json>", file=sys.stderr)
        sys.exit(1)
    sys.exit(preflight(sys.argv[1]))


if __name__ == "__main__":
    main()
