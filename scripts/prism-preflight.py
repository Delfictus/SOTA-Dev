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
    "4epr": True,   # KRAS G12D — Cys118, Cys185 structural CYS
    "2j1x": True,   # TP53 Y220C — Cys220 mutation site + structural CYS
    "1zg4": True,   # TEM-1 beta-lactamase — structural CYS
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

    # CHECK 2: HIS protonation — no bare "HIS", only HID/HIE/HIP
    his_variants = {"HID", "HIE", "HIP"} & unique_types
    if "HIS" in unique_types and not his_variants:
        fails.append("Bare HIS present with no HID/HIE/HIP — AMBER protonation not assigned")
    elif "HIS" in unique_types and his_variants:
        warns.append("Both bare HIS and protonated variants present — partial protonation?")

    # CHECK 4: GB radii present and non-zero for all atoms
    gb_radii = topo.get("gb_radii", [])
    if not gb_radii:
        fails.append("gb_radii array missing or empty — implicit solvent will fail")
    elif len(gb_radii) != n_atoms:
        fails.append(f"gb_radii length {len(gb_radii)} != n_atoms {n_atoms}")
    else:
        zero_radii = sum(1 for r in gb_radii if r == 0.0)
        if zero_radii > 0:
            pct = 100.0 * zero_radii / len(gb_radii)
            if pct > 5.0:
                fails.append(f"{zero_radii}/{len(gb_radii)} atoms have gb_radii=0.0 ({pct:.1f}%)")
            else:
                warns.append(f"{zero_radii}/{len(gb_radii)} atoms have gb_radii=0.0 ({pct:.1f}%)")

    # CHECK 5: CYS check against registry
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

    # CHECK 6: Net charge within reasonable bounds
    charges = topo.get("charges", [])
    if charges:
        net_charge = sum(charges)
        if abs(net_charge) > 10:
            warns.append(f"Net charge {net_charge:.1f} — |charge| > 10, verify protonation states")

    # Print results
    basename = os.path.basename(topo_path)
    print(f"Preflight: {basename}")
    print(f"  n_atoms: {n_atoms}")
    print(f"  n_residues: {n_residues}")
    print(f"  residue types: {len(unique_types)}")
    print(f"  CYS: {'present' if has_cys else 'absent'}")
    print(f"  HIS variants: {sorted(his_variants) if his_variants else 'none'}")
    print(f"  GB radii: {'present' if gb_radii else 'MISSING'} ({len(gb_radii)} values)")
    if charges:
        print(f"  Net charge: {sum(charges):.1f}")

    for w in warns:
        print(f"  WARN: {w}")

    if fails:
        for f in fails:
            print(f"  FAIL: {f}")
        print("PREFLIGHT: FAIL")
        return 1

    print("PREFLIGHT: PASS")
    return 0


def validate_chain_map(chain_map_path, n_residues):
    """Validate chain map against topology."""
    if not os.path.exists(chain_map_path):
        print(f"  WARN: Chain map {chain_map_path} not found")
        return
    with open(chain_map_path) as f:
        cm = json.load(f)
    total = cm.get("total_residues", 0)
    if total != n_residues:
        print(f"  WARN: Chain map total_residues={total} != topology n_residues={n_residues}")
    else:
        print(f"  Chain map: {total} residues across {len(cm.get('chains', []))} chains ✓")
    for entry in cm.get("chains", []):
        print(f"    {entry['source_file']} (chain {entry['original_chain']}): "
              f"merged {entry['merged_resnum_start']}-{entry['merged_resnum_end']} "
              f"({entry['n_residues']} residues)")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="PRISM-4D Topology Preflight")
    parser.add_argument("topology", help="Topology JSON file")
    parser.add_argument("--chain-map", default=None, help="Chain map JSON (for multichain)")
    args = parser.parse_args()

    result = preflight(args.topology)

    if args.chain_map and result == 0:
        with open(args.topology) as f:
            topo = json.load(f)
        validate_chain_map(args.chain_map, topo.get("n_residues", 0))

    sys.exit(result)


if __name__ == "__main__":
    main()
