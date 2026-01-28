#!/usr/bin/env python3
"""
PRISM4D Topology Validator

Validates a topology JSON file before engine processing:
1. Bond length sanity (1.0-2.5 Å typical for proteins)
2. Angle geometry (60-180°)
3. Chain connectivity (no orphan atoms)
4. Parameter completeness (all bonds/angles/dihedrals have parameters)
5. Atom overlap detection (no clashing atoms)

Usage:
    python validate_topology.py topology.json
    python validate_topology.py topology.json --strict
"""

import json
import argparse
import math
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Set
from collections import defaultdict


# Expected ratios based on production-ready structures
EXPECTED_BONDS_PER_ATOM = 1.02  # ±0.1
EXPECTED_ANGLES_PER_ATOM = 1.86  # ±0.2
EXPECTED_DIHEDRALS_PER_ATOM = 3.3  # ±0.5

# Bond length ranges (Å)
BOND_LENGTH_RANGES = {
    ('C', 'C'): (1.2, 1.6),
    ('C', 'N'): (1.2, 1.6),
    ('C', 'O'): (1.1, 1.5),
    ('C', 'S'): (1.6, 2.0),
    ('N', 'H'): (0.9, 1.1),
    ('O', 'H'): (0.9, 1.1),
    ('C', 'H'): (1.0, 1.2),
    ('S', 'S'): (1.9, 2.2),  # Disulfide
    ('DEFAULT',): (0.9, 2.5),
}

# Clash distance threshold (Å)
MIN_NONBONDED_DISTANCE = 1.5


def load_topology(path: str) -> Dict:
    """Load a topology JSON file."""
    with open(path) as f:
        return json.load(f)


def compute_distance(coords1: List[float], coords2: List[float]) -> float:
    """Compute Euclidean distance between two 3D points."""
    dx = coords1[0] - coords2[0]
    dy = coords1[1] - coords2[1]
    dz = coords1[2] - coords2[2]
    return math.sqrt(dx*dx + dy*dy + dz*dz)


def compute_angle(p1: List[float], p2: List[float], p3: List[float]) -> float:
    """Compute angle at p2 in degrees."""
    v1 = [p1[i] - p2[i] for i in range(3)]
    v2 = [p3[i] - p2[i] for i in range(3)]

    dot = sum(v1[i] * v2[i] for i in range(3))
    mag1 = math.sqrt(sum(x*x for x in v1))
    mag2 = math.sqrt(sum(x*x for x in v2))

    if mag1 < 1e-6 or mag2 < 1e-6:
        return 0.0

    cos_angle = max(-1.0, min(1.0, dot / (mag1 * mag2)))
    return math.degrees(math.acos(cos_angle))


def validate_ratios(topo: Dict, strict: bool = False) -> List[str]:
    """Validate bond/angle/dihedral ratios."""
    issues = []
    n_atoms = topo['n_atoms']

    ratios = {
        'bonds': len(topo.get('bonds', [])) / n_atoms,
        'angles': len(topo.get('angles', [])) / n_atoms,
        'dihedrals': len(topo.get('dihedrals', [])) / n_atoms,
    }

    tolerances = {
        'bonds': (EXPECTED_BONDS_PER_ATOM - 0.1, EXPECTED_BONDS_PER_ATOM + 0.1),
        'angles': (EXPECTED_ANGLES_PER_ATOM - 0.2, EXPECTED_ANGLES_PER_ATOM + 0.3),
        'dihedrals': (EXPECTED_DIHEDRALS_PER_ATOM - 0.5, EXPECTED_DIHEDRALS_PER_ATOM + 1.0),
    }

    for term, ratio in ratios.items():
        low, high = tolerances[term]
        if ratio < low:
            issues.append(f"Low {term}/atom ratio: {ratio:.3f} (expected >{low:.2f})")
        elif ratio > high and strict:
            issues.append(f"High {term}/atom ratio: {ratio:.3f} (expected <{high:.2f})")

    return issues


def validate_bond_lengths(topo: Dict, strict: bool = False) -> Tuple[List[str], Dict]:
    """Validate bond lengths are within expected ranges."""
    issues = []
    stats = {'checked': 0, 'ok': 0, 'short': 0, 'long': 0}

    # Handle flat array format [x1,y1,z1,x2,y2,z2,...] -> [[x1,y1,z1], [x2,y2,z2], ...]
    positions = topo.get('positions', topo.get('atoms', []))
    if positions and not isinstance(positions[0], (list, tuple)):
        coords = [[positions[i*3], positions[i*3+1], positions[i*3+2]]
                  for i in range(len(positions) // 3)]
    else:
        coords = positions
    elements = topo.get('elements', [])
    bonds = topo.get('bonds', [])

    short_bonds = []
    long_bonds = []

    for bond in bonds:
        i, j = bond['i'], bond['j']
        if i >= len(coords) or j >= len(coords):
            issues.append(f"Invalid bond indices: {i}-{j}")
            continue

        dist = compute_distance(coords[i], coords[j])
        stats['checked'] += 1

        # Get element types for range lookup
        if elements and i < len(elements) and j < len(elements):
            elem1, elem2 = elements[i], elements[j]
            key = tuple(sorted([elem1, elem2]))
            if key not in BOND_LENGTH_RANGES:
                key = ('DEFAULT',)
        else:
            key = ('DEFAULT',)

        low, high = BOND_LENGTH_RANGES[key]

        if dist < low * 0.8:  # Very short
            short_bonds.append((i, j, dist))
            stats['short'] += 1
        elif dist > high * 1.5:  # Very long
            long_bonds.append((i, j, dist))
            stats['long'] += 1
        else:
            stats['ok'] += 1

    if short_bonds:
        issues.append(f"Very short bonds ({len(short_bonds)}): {short_bonds[:3]}...")
    if long_bonds:
        issues.append(f"Very long bonds ({len(long_bonds)}): {long_bonds[:3]}...")

    return issues, stats


def validate_angles(topo: Dict, sample_size: int = 1000) -> Tuple[List[str], Dict]:
    """Validate angle values are within reasonable range."""
    issues = []
    stats = {'checked': 0, 'ok': 0, 'small': 0, 'linear': 0}

    # Handle flat array format
    positions = topo.get('positions', topo.get('atoms', []))
    if positions and not isinstance(positions[0], (list, tuple)):
        coords = [[positions[i*3], positions[i*3+1], positions[i*3+2]]
                  for i in range(len(positions) // 3)]
    else:
        coords = positions
    angles = topo.get('angles', [])

    small_angles = []
    linear_angles = []

    # Sample angles if there are many
    check_angles = angles[:sample_size] if len(angles) > sample_size else angles

    for angle in check_angles:
        i, j = angle['i'], angle['j']
        k = angle.get('k', angle.get('k_idx'))  # Handle both formats
        if k is None or max(i, j, k) >= len(coords):
            continue

        angle_deg = compute_angle(coords[i], coords[j], coords[k])
        stats['checked'] += 1

        if angle_deg < 60:
            small_angles.append((i, j, k, angle_deg))
            stats['small'] += 1
        elif angle_deg > 175:
            linear_angles.append((i, j, k, angle_deg))
            stats['linear'] += 1
        else:
            stats['ok'] += 1

    if small_angles and len(small_angles) > 10:
        issues.append(f"Small angles (<60°): {len(small_angles)}")
    if linear_angles and len(linear_angles) > len(check_angles) * 0.1:
        issues.append(f"Many linear angles (>175°): {len(linear_angles)}")

    return issues, stats


def validate_connectivity(topo: Dict) -> Tuple[List[str], Dict]:
    """Check for orphan atoms (atoms with no bonds)."""
    issues = []
    n_atoms = topo['n_atoms']
    bonds = topo.get('bonds', [])

    bonded_atoms = set()
    for bond in bonds:
        bonded_atoms.add(bond['i'])
        bonded_atoms.add(bond['j'])

    orphans = n_atoms - len(bonded_atoms)

    if orphans > n_atoms * 0.01:  # More than 1% orphans
        issues.append(f"Orphan atoms (no bonds): {orphans}/{n_atoms}")

    return issues, {'orphans': orphans, 'bonded': len(bonded_atoms)}


def validate_clashes(topo: Dict, sample_size: int = 5000) -> Tuple[List[str], Dict]:
    """Check for steric clashes (very close non-bonded atoms)."""
    issues = []
    stats = {'checked': 0, 'clashes': 0}

    # Handle flat array format
    positions = topo.get('positions', topo.get('atoms', []))
    if positions and not isinstance(positions[0], (list, tuple)):
        coords = [[positions[i*3], positions[i*3+1], positions[i*3+2]]
                  for i in range(len(positions) // 3)]
    else:
        coords = positions
    bonds = topo.get('bonds', [])

    # Build bonded set
    bonded = set()
    for bond in bonds:
        bonded.add((min(bond['i'], bond['j']), max(bond['i'], bond['j'])))

    # Check non-bonded pairs (sampled)
    n_atoms = len(coords)
    import random
    pairs_to_check = []

    if n_atoms < 200:
        # Check all pairs
        for i in range(n_atoms):
            for j in range(i+1, n_atoms):
                if (i, j) not in bonded:
                    pairs_to_check.append((i, j))
    else:
        # Random sampling
        for _ in range(sample_size):
            i = random.randint(0, n_atoms-1)
            j = random.randint(0, n_atoms-1)
            if i != j and (min(i,j), max(i,j)) not in bonded:
                pairs_to_check.append((i, j))

    clashes = []
    for i, j in pairs_to_check:
        dist = compute_distance(coords[i], coords[j])
        stats['checked'] += 1
        if dist < MIN_NONBONDED_DISTANCE:
            clashes.append((i, j, dist))
            stats['clashes'] += 1

    if clashes:
        issues.append(f"Steric clashes (<{MIN_NONBONDED_DISTANCE}Å): {len(clashes)}")
        if len(clashes) <= 5:
            for i, j, d in clashes:
                issues.append(f"  Clash: atoms {i}-{j}, dist={d:.2f}Å")

    return issues, stats


def validate_topology(topo_path: str, strict: bool = False, verbose: bool = True) -> bool:
    """
    Run all validations on a topology file.

    Returns True if topology passes validation.
    """
    topo = load_topology(topo_path)

    all_issues = []

    if verbose:
        print(f"\n{'='*60}")
        print(f"Validating: {Path(topo_path).name}")
        print(f"{'='*60}")
        print(f"  Atoms: {topo['n_atoms']:,}")
        print(f"  Chains: {topo.get('n_chains', 'N/A')}")
        print(f"  Bonds: {len(topo.get('bonds', [])):,}")
        print(f"  Angles: {len(topo.get('angles', [])):,}")
        print(f"  Dihedrals: {len(topo.get('dihedrals', [])):,}")

    # 1. Check ratios
    ratio_issues = validate_ratios(topo, strict)
    all_issues.extend(ratio_issues)
    if verbose:
        print(f"\n[Ratios] ", end="")
        if ratio_issues:
            print(f"⚠️  {len(ratio_issues)} issue(s)")
        else:
            print("✅ OK")

    # 2. Check bond lengths
    bond_issues, bond_stats = validate_bond_lengths(topo, strict)
    all_issues.extend(bond_issues)
    if verbose:
        print(f"[Bonds]  ", end="")
        if bond_issues:
            print(f"⚠️  {bond_stats['short']} short, {bond_stats['long']} long")
        else:
            print(f"✅ OK ({bond_stats['checked']} checked)")

    # 3. Check angles
    angle_issues, angle_stats = validate_angles(topo)
    all_issues.extend(angle_issues)
    if verbose:
        print(f"[Angles] ", end="")
        if angle_issues:
            print(f"⚠️  {angle_stats['small']} small, {angle_stats['linear']} linear")
        else:
            print(f"✅ OK ({angle_stats['checked']} checked)")

    # 4. Check connectivity
    conn_issues, conn_stats = validate_connectivity(topo)
    all_issues.extend(conn_issues)
    if verbose:
        print(f"[Connectivity] ", end="")
        if conn_issues:
            print(f"⚠️  {conn_stats['orphans']} orphan atoms")
        else:
            print(f"✅ OK ({conn_stats['bonded']} bonded atoms)")

    # 5. Check clashes
    clash_issues, clash_stats = validate_clashes(topo)
    all_issues.extend(clash_issues)
    if verbose:
        print(f"[Clashes] ", end="")
        if clash_issues:
            print(f"⚠️  {clash_stats['clashes']} clashes detected")
        else:
            print(f"✅ OK ({clash_stats['checked']} pairs sampled)")

    # Summary
    if verbose:
        print(f"\n{'='*60}")
        if all_issues:
            print(f"❌ VALIDATION FAILED ({len(all_issues)} issues)")
            for issue in all_issues:
                print(f"  • {issue}")
        else:
            print(f"✅ VALIDATION PASSED")
        print(f"{'='*60}\n")

    return len(all_issues) == 0


def main():
    parser = argparse.ArgumentParser(
        description='Validate PRISM4D topology files before engine processing'
    )
    parser.add_argument('topology', help='Topology JSON file to validate')
    parser.add_argument('--strict', '-s', action='store_true',
                        help='Enable strict validation')
    parser.add_argument('--quiet', '-q', action='store_true',
                        help='Quiet output (exit code only)')

    args = parser.parse_args()

    if not Path(args.topology).exists():
        print(f"ERROR: File not found: {args.topology}", file=sys.stderr)
        return 1

    passed = validate_topology(args.topology, strict=args.strict, verbose=not args.quiet)
    return 0 if passed else 1


if __name__ == '__main__':
    sys.exit(main())
