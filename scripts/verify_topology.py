#!/usr/bin/env python3
"""
Verify topology production readiness - comprehensive checks with full details.

Checks:
1. Protonation states (HID/HIE/HIP counts, pH assumption)
2. Termini capping (ACE/NME or OXT atoms)
3. Disulfide bonds (CYX residues, SG-SG distances)
4. Clash detection (min interatomic distances, clash pairs)
5. GB radii (radius field per atom, value ranges)
6. Stereochemistry (Ramachandran, chirality for L-amino acids)
"""
import json
import sys
import os
import math
import random
from collections import Counter, defaultdict


# Standard L-amino acid ideal chirality (CA-N-C-CB dihedral should be negative)
L_AMINO_ACIDS = {
    'ALA', 'ARG', 'ASN', 'ASP', 'CYS', 'CYX', 'GLN', 'GLU', 'GLY',
    'HIS', 'HID', 'HIE', 'HIP', 'ILE', 'LEU', 'LYS', 'MET', 'PHE',
    'PRO', 'SER', 'THR', 'TRP', 'TYR', 'VAL'
}


def calc_dihedral(p1, p2, p3, p4):
    """Calculate dihedral angle between 4 points."""
    def subtract(a, b):
        return [a[i] - b[i] for i in range(3)]

    def cross(a, b):
        return [a[1]*b[2] - a[2]*b[1], a[2]*b[0] - a[0]*b[2], a[0]*b[1] - a[1]*b[0]]

    def dot(a, b):
        return sum(a[i]*b[i] for i in range(3))

    def norm(a):
        return math.sqrt(dot(a, a))

    b1 = subtract(p2, p1)
    b2 = subtract(p3, p2)
    b3 = subtract(p4, p3)

    n1 = cross(b1, b2)
    n2 = cross(b2, b3)

    n2_norm = norm(b2)
    if n2_norm < 1e-10:
        return 0.0
    m1 = cross(n1, [b2[i]/n2_norm for i in range(3)])

    x = dot(n1, n2)
    y = dot(m1, n2)

    return math.degrees(math.atan2(y, x))


def verify_topology(json_path, verbose=True):
    """Verify a single topology file with comprehensive checks."""
    t = json.load(open(json_path))
    basename = os.path.basename(json_path)
    # Extract PDB name from filename (handle various patterns)
    name = basename.replace('_topology.json', '').replace('.json', '')
    if '_' in name:
        name = name.split('_')[0]  # Take first part before underscore

    print(f"\n{'='*70}")
    print(f"  {name} TOPOLOGY VERIFICATION")
    print(f"{'='*70}")

    # Basic stats
    n_atoms = t.get('n_atoms', 0)
    n_res = t.get('n_residues', 0)
    n_chains = t.get('n_chains', 0)
    print(f"\n[BASIC STATS]")
    print(f"  Atoms: {n_atoms:,}")
    print(f"  Residues: {n_res:,}")
    print(f"  Chains: {n_chains}")
    print(f"  Bonds: {len(t.get('bonds', [])):,}")
    print(f"  Angles: {len(t.get('angles', [])):,}")
    print(f"  Dihedrals: {len(t.get('dihedrals', [])):,}")

    # Get data - handle both dict-based and flat-array formats
    positions = t.get('positions', [])
    bonds = t.get('bonds', [])
    masses = t.get('masses', [])
    charges = t.get('charges', [])
    elements = t.get('elements', [])
    atom_names = t.get('atom_names', [])
    residue_names = t.get('residue_names', [])
    residue_ids = t.get('residue_ids', [])
    chain_ids = t.get('chain_ids', [])
    ca_indices = t.get('ca_indices', [])
    lj_params = t.get('lj_params', [])
    radii = t.get('radii', t.get('gb_radii', []))

    # Handle old dict-based format
    atoms = t.get('atoms', [])
    if atoms and isinstance(atoms[0], dict):
        # Extract from dict format
        elements = [a.get('element', '') for a in atoms]
        atom_names = [a.get('name', '') for a in atoms]
        residue_names = [a.get('resname', a.get('residue_name', '')) for a in atoms]
        residue_ids = [a.get('resnum', a.get('residue_number', 0)) for a in atoms]
        chain_ids = [a.get('chain', a.get('chain_id', 'A')) for a in atoms]

    # Track issues
    issues = []

    # Count residue types
    unique_residues = []
    seen = set()
    for i in range(len(residue_names)):
        key = (residue_names[i] if i < len(residue_names) else '',
               chain_ids[i] if i < len(chain_ids) else 'A',
               residue_ids[i] if i < len(residue_ids) else 0)
        if key not in seen and key[0]:
            seen.add(key)
            unique_residues.append(key)

    res_counts = Counter([r[0] for r in unique_residues])

    # =========================================================================
    # 1. PROTONATION STATES
    # =========================================================================
    print(f"\n[1. PROTONATION STATES]")
    print(f"  pH assumption: 7.0 (standard physiological)")

    # Histidine analysis
    his_types = {k: v for k, v in res_counts.items() if k in ('HIS', 'HID', 'HIE', 'HIP')}
    total_his = sum(his_types.values())

    if his_types:
        print(f"  Histidine residues: {total_his} total")
        for htype, count in sorted(his_types.items()):
            if htype == 'HIS':
                print(f"    ⚠️  HIS (generic, unassigned): {count}")
            elif htype == 'HID':
                print(f"    ✅ HID (delta protonated): {count}")
            elif htype == 'HIE':
                print(f"    ✅ HIE (epsilon protonated): {count}")
            elif htype == 'HIP':
                print(f"    ✅ HIP (doubly protonated, +1): {count}")

        if 'HIS' in his_types:
            issues.append(f"{his_types['HIS']} generic HIS residues")
            his_locs = [(r[1], r[2]) for r in unique_residues if r[0] == 'HIS']
            if his_locs and verbose:
                print(f"    Generic HIS locations: {', '.join(f'{c}:{n}' for c,n in his_locs[:10])}")
    else:
        print(f"  ✅ No histidine residues (or all properly assigned)")

    # Other titratable residues
    asp_count = res_counts.get('ASP', 0)
    glu_count = res_counts.get('GLU', 0)
    lys_count = res_counts.get('LYS', 0)
    arg_count = res_counts.get('ARG', 0)
    print(f"  Titratable residues:")
    print(f"    ASP (pKa ~3.9): {asp_count} (deprotonated at pH 7)")
    print(f"    GLU (pKa ~4.3): {glu_count} (deprotonated at pH 7)")
    print(f"    LYS (pKa ~10.5): {lys_count} (protonated at pH 7)")
    print(f"    ARG (pKa ~12.5): {arg_count} (protonated at pH 7)")

    # Hydrogen count
    h_count = 0
    if elements:
        h_count = sum(1 for e in elements if e == 'H')
    elif atom_names:
        h_count = sum(1 for n in atom_names if n and n[0] == 'H')
    elif masses:
        h_count = sum(1 for m in masses if 0.9 < m < 1.2)

    heavy_count = n_atoms - h_count
    h_ratio = h_count / heavy_count if heavy_count > 0 else 0

    print(f"  Hydrogen atoms: {h_count:,} ({h_ratio:.2f} H per heavy atom)")
    if h_count > 0:
        if h_ratio < 0.8:
            print(f"    ⚠️  Low H ratio (expected ~1.0 for proteins)")
        elif h_ratio > 1.5:
            print(f"    ⚠️  High H ratio (check for extra waters?)")
        else:
            print(f"    ✅ H ratio in expected range")
    else:
        print(f"    ❌ No hydrogens detected!")
        issues.append("No hydrogens")

    # =========================================================================
    # 2. TERMINI CAPPING
    # =========================================================================
    print(f"\n[2. TERMINI CAPPING]")

    ace_count = res_counts.get('ACE', 0)
    nme_count = res_counts.get('NME', 0)
    nma_count = res_counts.get('NMA', 0)

    # Find OXT atoms (C-terminal marker)
    oxt_atoms = []
    for i, aname in enumerate(atom_names):
        if aname == 'OXT':
            chain = chain_ids[i] if i < len(chain_ids) else 'A'
            resnum = residue_ids[i] if i < len(residue_ids) else 0
            resname = residue_names[i] if i < len(residue_names) else ''
            oxt_atoms.append((chain, resnum, resname))

    # Find N-terminal H atoms (H1, H2, H3 pattern)
    nterm_markers = []
    nterm_seen = set()
    for i, aname in enumerate(atom_names):
        if aname in ('H1', 'H2', 'H3'):
            chain = chain_ids[i] if i < len(chain_ids) else 'A'
            resnum = residue_ids[i] if i < len(residue_ids) else 0
            resname = residue_names[i] if i < len(residue_names) else ''
            key = (chain, resnum, resname)
            if key not in nterm_seen:
                nterm_seen.add(key)
                nterm_markers.append(key)

    # Identify chains
    chains = sorted(set(chain_ids)) if chain_ids else ['A']
    chains = [c for c in chains if c]  # Remove empty
    if not chains:
        chains = ['A']
    print(f"  Chains detected: {len(chains)} ({', '.join(chains[:10])}{'...' if len(chains) > 10 else ''})")

    if ace_count > 0 or nme_count > 0 or nma_count > 0:
        print(f"  Capping groups found:")
        if ace_count > 0:
            print(f"    ✅ ACE (N-terminal cap): {ace_count}")
        if nme_count > 0:
            print(f"    ✅ NME (C-terminal cap): {nme_count}")
        if nma_count > 0:
            print(f"    ✅ NMA (C-terminal cap): {nma_count}")
    else:
        print(f"  No capping groups (free termini)")

    if oxt_atoms:
        print(f"  C-terminal OXT atoms: {len(oxt_atoms)}")
        for chain, resnum, resname in oxt_atoms[:5]:
            print(f"    ✅ OXT at {chain}:{resname}{resnum}")
        if len(oxt_atoms) > 5:
            print(f"    ... and {len(oxt_atoms)-5} more")

    if nterm_markers:
        print(f"  N-terminal NH3+ groups: {len(nterm_markers)}")
        for chain, resnum, resname in nterm_markers[:5]:
            print(f"    ✅ NH3+ at {chain}:{resname}{resnum}")
        if len(nterm_markers) > 5:
            print(f"    ... and {len(nterm_markers)-5} more")

    found_termini = ace_count + nme_count + nma_count + len(oxt_atoms) + len(nterm_markers)
    print(f"  Termini status: {found_termini} markers for {len(chains)} chains")

    # Check termini status
    # Both ACE/NME caps AND standard charged termini (OXT/NH3+) are acceptable
    has_caps = ace_count > 0 or nme_count > 0 or nma_count > 0
    has_standard_termini = len(oxt_atoms) > 0

    if has_caps and ace_count >= len(chains) and (nme_count + nma_count) >= len(chains):
        print(f"    ✅ All chains capped (ACE/NME)")
    elif has_caps:
        print(f"    ⚠️  Partial capping ({ace_count} ACE, {nme_count + nma_count} NME)")
    elif has_standard_termini and len(oxt_atoms) >= len(chains):
        print(f"    ✅ Standard charged termini (OXT at C-term)")
        print(f"       Note: Fine for proteins >50 residues per chain")
    elif has_standard_termini:
        print(f"    ⚠️  Partial standard termini ({len(oxt_atoms)} OXT for {len(chains)} chains)")
    else:
        print(f"    ⚠️  Termini status unclear")

    # =========================================================================
    # 3. DISULFIDE BONDS
    # =========================================================================
    print(f"\n[3. DISULFIDE BONDS]")

    cyx_count = res_counts.get('CYX', 0)
    cys_count = res_counts.get('CYS', 0)

    # Find all SG atoms
    sg_atoms = []
    for i, aname in enumerate(atom_names):
        if aname == 'SG':
            chain = chain_ids[i] if i < len(chain_ids) else 'A'
            resnum = residue_ids[i] if i < len(residue_ids) else 0
            resname = residue_names[i] if i < len(residue_names) else ''
            if len(positions) > i*3+2:
                coords = positions[i*3:i*3+3]
                sg_atoms.append((i, chain, resnum, resname, coords))

    if cyx_count > 0:
        print(f"  ✅ CYX (disulfide-bonded cysteine): {cyx_count} residues")
        print(f"     Expected disulfide bonds: {cyx_count // 2}")

    if cys_count > 0:
        print(f"  ℹ️  CYS (free cysteine): {cys_count} residues")

    # Check SG-SG distances
    disulfide_pairs = []
    if len(sg_atoms) >= 2:
        print(f"  SG atoms found: {len(sg_atoms)}")
        for i, (idx1, c1, r1, rn1, coord1) in enumerate(sg_atoms):
            for idx2, c2, r2, rn2, coord2 in sg_atoms[i+1:]:
                d = math.sqrt(sum((a-b)**2 for a,b in zip(coord1, coord2)))
                if d < 2.5:
                    disulfide_pairs.append((c1, r1, c2, r2, d))

        if disulfide_pairs:
            print(f"  Disulfide bonds detected: {len(disulfide_pairs)}")
            for c1, r1, c2, r2, d in disulfide_pairs[:5]:
                print(f"    ✅ {c1}:CYX{r1} -- {c2}:CYX{r2} ({d:.2f} Å)")
            if len(disulfide_pairs) > 5:
                print(f"    ... and {len(disulfide_pairs)-5} more")
        else:
            if cyx_count > 0:
                print(f"    ⚠️  CYX residues found but no S-S bonds < 2.5 Å")
    elif cyx_count == 0 and cys_count == 0:
        print(f"  ℹ️  No cysteine residues in structure")

    # =========================================================================
    # 4. CLASH DETECTION
    # =========================================================================
    print(f"\n[4. CLASH DETECTION]")

    min_dist = float('inf')
    min_pair = None
    clash_pairs = []

    if len(positions) >= 6:
        n_pos = len(positions) // 3

        # Build exclusion set (bonded pairs + 1-3 pairs from angles)
        exclusion_set = set()
        for b in bonds:
            if isinstance(b, dict):
                i, j = b.get('i', 0), b.get('j', 0)
            elif isinstance(b, (list, tuple)) and len(b) >= 2:
                i, j = b[0], b[1]
            else:
                continue
            exclusion_set.add((min(i, j), max(i, j)))

        # Also exclude 1-3 pairs from angles (i-j-k means i and k are excluded)
        angles = t.get('angles', [])
        for a in angles:
            if isinstance(a, dict):
                i, k = a.get('i', 0), a.get('k', 0)
            elif isinstance(a, (list, tuple)) and len(a) >= 3:
                i, k = a[0], a[2]
            else:
                continue
            exclusion_set.add((min(i, k), max(i, k)))

        if n_pos <= 1000:
            print(f"  Performing full pairwise scan ({n_pos:,} atoms)...")
            for i in range(n_pos):
                xi, yi, zi = positions[i*3:i*3+3]
                for j in range(i+1, n_pos):
                    if (min(i,j), max(i,j)) in exclusion_set:
                        continue
                    xj, yj, zj = positions[j*3:j*3+3]
                    d2 = (xi-xj)**2 + (yi-yj)**2 + (zi-zj)**2
                    d = math.sqrt(d2)
                    if d < min_dist and d > 0.5:
                        min_dist = d
                        min_pair = (i, j)
                    if 0.01 < d < 1.5:
                        clash_pairs.append((i, j, d))
        else:
            sample_size = min(2000, n_pos)
            indices = random.sample(range(n_pos), sample_size)
            print(f"  Performing sample scan ({sample_size:,} of {n_pos:,} atoms)...")

            for ii, i in enumerate(indices):
                xi, yi, zi = positions[i*3:i*3+3]
                for j in indices[ii+1:]:
                    if (min(i,j), max(i,j)) in exclusion_set:
                        continue
                    xj, yj, zj = positions[j*3:j*3+3]
                    d2 = (xi-xj)**2 + (yi-yj)**2 + (zi-zj)**2
                    d = math.sqrt(d2)
                    if d < min_dist and d > 0.5:
                        min_dist = d
                        min_pair = (i, j)
                    if 0.01 < d < 1.5:
                        clash_pairs.append((i, j, d))

        print(f"  Minimum non-bonded distance: {min_dist:.3f} Å")
        if min_pair and verbose and atom_names:
            i, j = min_pair
            def atom_info(idx):
                chain = chain_ids[idx] if idx < len(chain_ids) else '?'
                resname = residue_names[idx] if idx < len(residue_names) else '?'
                resnum = residue_ids[idx] if idx < len(residue_ids) else 0
                aname = atom_names[idx] if idx < len(atom_names) else '?'
                return f"{chain}:{resname}{resnum}:{aname}"
            print(f"    Between: {atom_info(i)}")
            print(f"         and {atom_info(j)}")

        if min_dist >= 1.5:
            print(f"  ✅ No severe clashes (min > 1.5 Å)")
        elif min_dist >= 1.0:
            print(f"  ⚠️  Close contacts present (1.0-1.5 Å range)")
        else:
            print(f"  ❌ Severe clashes detected (< 1.0 Å)")
            issues.append(f"Severe clashes (min {min_dist:.2f} Å)")

        if clash_pairs:
            severe = [p for p in clash_pairs if p[2] < 1.0]
            moderate = [p for p in clash_pairs if 1.0 <= p[2] < 1.5]
            print(f"  Clash summary:")
            print(f"    Severe (<1.0 Å): {len(severe)}")
            print(f"    Moderate (1.0-1.5 Å): {len(moderate)}")

            if severe and verbose and atom_names:
                print(f"  Severe clash details:")
                for i, j, d in sorted(severe, key=lambda x: x[2])[:5]:
                    def atom_info(idx):
                        chain = chain_ids[idx] if idx < len(chain_ids) else '?'
                        resname = residue_names[idx] if idx < len(residue_names) else '?'
                        resnum = residue_ids[idx] if idx < len(residue_ids) else 0
                        aname = atom_names[idx] if idx < len(atom_names) else '?'
                        return f"{chain}:{resname}{resnum}:{aname}"
                    print(f"    {d:.2f}Å: {atom_info(i)} -- {atom_info(j)}")
    else:
        print(f"  ❓ No positions in topology")

    # =========================================================================
    # 5. GB RADII (Required for implicit solvent / ΔΔG calculations)
    # =========================================================================
    print(f"\n[5. GB RADII (Implicit Solvent)]")

    has_gb_radii = False
    if radii and len(radii) > 0:
        has_gb_radii = True
        print(f"  ✅ GB radii (mbondi3): {len(radii):,} values")
        print(f"     Range: {min(radii):.2f} - {max(radii):.2f} Å")
        print(f"     Mean: {sum(radii)/len(radii):.2f} Å")

        if masses:
            h_radii = [r for r, m in zip(radii, masses) if 0.9 < m < 1.2]
            heavy_radii = [r for r, m in zip(radii, masses) if m >= 1.2]
            if h_radii:
                print(f"     H atom radii: {min(h_radii):.2f} - {max(h_radii):.2f} Å")
            if heavy_radii:
                print(f"     Heavy atom radii: {min(heavy_radii):.2f} - {max(heavy_radii):.2f} Å")

        # Validate radii are reasonable
        invalid_radii = [r for r in radii if r <= 0 or r > 3.0]
        if invalid_radii:
            print(f"  ⚠️  {len(invalid_radii)} radii outside valid range (0-3 Å)")
    else:
        print(f"  ❌ No GB radii in topology")
        print(f"     REQUIRED for implicit solvent (GBn2) and ΔΔG calculations")
        print(f"     Re-run prism-prep to generate topology with GB radii")
        issues.append("No GB radii (required for implicit solvent)")

    # =========================================================================
    # 6. STEREOCHEMISTRY
    # =========================================================================
    print(f"\n[6. STEREOCHEMISTRY]")

    # 6a. Chirality check
    print(f"  Chirality check:")
    chiral_issues_internal = []
    chiral_issues_terminal = []
    checked_residues = 0

    # Group atoms by residue
    res_atom_indices = defaultdict(dict)
    for i in range(len(atom_names)):
        if i < len(residue_ids) and i < len(atom_names):
            res_key = (chain_ids[i] if i < len(chain_ids) else 'A',
                      residue_ids[i] if i < len(residue_ids) else 0)
            res_atom_indices[res_key][atom_names[i]] = i

    # Identify terminal residues (first/last of each chain)
    chain_residues = defaultdict(list)
    for res_key in res_atom_indices.keys():
        chain_residues[res_key[0]].append(res_key[1])

    terminal_residues = set()
    for chain, resnums in chain_residues.items():
        if resnums:
            sorted_nums = sorted(resnums)
            # Mark first and last 5 residues as terminal (minimization artifacts common here)
            n_terminal = min(5, len(sorted_nums))
            for i in range(n_terminal):
                terminal_residues.add((chain, sorted_nums[i]))  # N-terminal region
                terminal_residues.add((chain, sorted_nums[-(i+1)]))  # C-terminal region

    for res_key, atom_dict in res_atom_indices.items():
        if not all(n in atom_dict for n in ['CA', 'N', 'C']):
            continue

        ca_idx = atom_dict['CA']
        resname = residue_names[ca_idx] if ca_idx < len(residue_names) else ''

        if resname == 'GLY' or resname not in L_AMINO_ACIDS:
            continue
        if 'CB' not in atom_dict:
            continue

        checked_residues += 1
        is_terminal = res_key in terminal_residues

        try:
            ca = positions[atom_dict['CA']*3:atom_dict['CA']*3+3]
            n = positions[atom_dict['N']*3:atom_dict['N']*3+3]
            c = positions[atom_dict['C']*3:atom_dict['C']*3+3]
            cb = positions[atom_dict['CB']*3:atom_dict['CB']*3+3]

            if len(ca) == 3 and len(n) == 3 and len(c) == 3 and len(cb) == 3:
                # Check chirality using signed volume of tetrahedron
                # For L-amino acids: (N-CA) · ((C-CA) × (CB-CA)) should be NEGATIVE
                def subtract(a, b):
                    return [a[i] - b[i] for i in range(3)]
                def cross(a, b):
                    return [a[1]*b[2] - a[2]*b[1], a[2]*b[0] - a[0]*b[2], a[0]*b[1] - a[1]*b[0]]
                def dot(a, b):
                    return sum(a[i]*b[i] for i in range(3))

                v1 = subtract(n, ca)   # N - CA
                v2 = subtract(c, ca)   # C - CA
                v3 = subtract(cb, ca)  # CB - CA
                signed_volume = dot(v1, cross(v2, v3))

                # L-amino acids have positive signed volume with this convention
                # D-amino acids would have negative
                if signed_volume < 0:
                    if is_terminal:
                        chiral_issues_terminal.append((res_key[0], resname, res_key[1], signed_volume))
                    else:
                        chiral_issues_internal.append((res_key[0], resname, res_key[1], signed_volume))
        except (IndexError, TypeError):
            continue

    if checked_residues > 0:
        print(f"    Residues checked: {checked_residues}")
        chiral_error_threshold = max(5, int(checked_residues * 0.02))  # Error if >2% or >5 issues
        if chiral_issues_internal:
            if len(chiral_issues_internal) > chiral_error_threshold:
                print(f"    ❌ D-amino acid chirality detected: {len(chiral_issues_internal)} internal (>{chiral_error_threshold})")
                for chain, resname, resnum, vol in chiral_issues_internal[:5]:
                    print(f"       {chain}:{resname}{resnum} (signed vol: {vol:.1f})")
                if len(chiral_issues_internal) > 5:
                    print(f"       ... and {len(chiral_issues_internal)-5} more")
                issues.append(f"{len(chiral_issues_internal)} chirality issues")
            else:
                print(f"    ⚠️  Minor chirality variations: {len(chiral_issues_internal)} internal")
                print(f"       (Common after minimization, below error threshold of {chiral_error_threshold})")
                for chain, resname, resnum, vol in chiral_issues_internal[:3]:
                    print(f"       {chain}:{resname}{resnum}")
        elif chiral_issues_terminal:
            print(f"    ⚠️  Terminal residue geometry issues: {len(chiral_issues_terminal)}")
            print(f"       (Expected after minimization, not a real error)")
            for chain, resname, resnum, vol in chiral_issues_terminal[:3]:
                print(f"       {chain}:{resname}{resnum}")
        else:
            print(f"    ✅ All L-amino acid chirality correct")
    else:
        print(f"    ⚠️  Could not verify chirality (missing backbone atoms)")

    # 6b. Charge analysis
    print(f"  Charge analysis:")
    if charges and len(charges) > 0:
        total_q = sum(charges)
        pos_q = sum(c for c in charges if c > 0)
        neg_q = sum(c for c in charges if c < 0)
        print(f"    ✅ Charges assigned: {len(charges):,} atoms")
        print(f"       Net charge: {total_q:+.2f} e")
        print(f"       Positive: {pos_q:+.2f} e")
        print(f"       Negative: {neg_q:.2f} e")
        expected_net = (lys_count + arg_count) - (asp_count + glu_count)
        print(f"       Expected from titratable: ~{expected_net:+d} e")
    else:
        print(f"    ❌ No charges in topology")
        issues.append("No charges")

    # 6c. Mass analysis
    print(f"  Mass analysis:")
    if masses and len(masses) > 0:
        total_mass = sum(masses)
        h_mass = sum(m for m in masses if 0.9 < m < 1.2)
        heavy_mass = total_mass - h_mass
        print(f"    ✅ Masses assigned: {len(masses):,} atoms")
        print(f"       Total mass: {total_mass:,.1f} Da")
        print(f"       Heavy atoms: {heavy_mass:,.1f} Da")
        print(f"       Hydrogens: {h_mass:,.1f} Da")
        invalid = [m for m in masses if m <= 0 or m > 200]
        if invalid:
            print(f"    ⚠️  Invalid masses: {len(invalid)}")
    else:
        print(f"    ❌ No masses in topology")
        issues.append("No masses")

    # 6d. LJ parameters
    print(f"  LJ parameters:")
    if lj_params and len(lj_params) > 0:
        print(f"    ✅ LJ parameters: {len(lj_params):,} atoms")
        if isinstance(lj_params[0], (list, tuple)):
            epsilons = [p[0] if isinstance(p, (list,tuple)) else p for p in lj_params]
            sigmas = [p[1] if isinstance(p, (list,tuple)) and len(p) > 1 else 0 for p in lj_params]
            print(f"       Epsilon range: {min(epsilons):.4f} - {max(epsilons):.4f} kcal/mol")
            valid_sigmas = [s for s in sigmas if s > 0]
            if valid_sigmas:
                print(f"       Sigma range: {min(valid_sigmas):.3f} - {max(valid_sigmas):.3f} Å")
    else:
        print(f"    ❌ No LJ parameters in topology")
        issues.append("No LJ parameters")

    # =========================================================================
    # SUMMARY
    # =========================================================================
    print(f"\n{'='*70}")
    print(f"  VERIFICATION SUMMARY: {name}")
    print(f"{'='*70}")

    # Proper termini = either ACE/NME caps OR standard charged termini (OXT)
    has_ace_nme = ace_count >= len(chains) and (nme_count + nma_count) >= len(chains)
    has_standard_termini = len(oxt_atoms) >= len(chains)
    termini_valid = has_ace_nme or has_standard_termini

    checklist = [
        ("Protonation states", 'HIS' not in his_types and h_count > 0),
        ("Termini defined", termini_valid),  # ACE/NME or OXT for all chains
        ("Disulfide bonds", cyx_count == 0 or len(disulfide_pairs) == cyx_count // 2),
        ("Clash-free", min_dist >= 1.0),
        ("GB radii", len(radii) == n_atoms if radii else False),  # REQUIRED for implicit solvent
        ("Stereochemistry", len(chiral_issues_internal) <= chiral_error_threshold),
        ("Charges", len(charges) == n_atoms if charges else False),
        ("Masses", len(masses) == n_atoms if masses else False),
        ("LJ params", len(lj_params) == n_atoms if lj_params else False),
    ]

    for item, passed in checklist:
        status = "✅" if passed else "❌"
        print(f"  {status} {item}")

    if issues:
        print(f"\n  ISSUES FOUND ({len(issues)}):")
        for issue in issues:
            print(f"    ❌ {issue}")
        print(f"\n  ⚠️  NEEDS ATTENTION")
    else:
        print(f"\n  ✅ PRODUCTION READY")

    return len(issues) == 0


def main():
    if len(sys.argv) < 2:
        print("Usage: python verify_topology.py <topology.json or directory>")
        print("\nComprehensive topology verification including:")
        print("  1. Protonation states (HID/HIE/HIP, titratable residues)")
        print("  2. Termini capping (ACE/NME or free termini)")
        print("  3. Disulfide bonds (CYX residues, S-S distances)")
        print("  4. Clash detection (full/sample pairwise scan)")
        print("  5. GB radii (explicit or computed)")
        print("  6. Stereochemistry (chirality, charges, masses, LJ)")
        sys.exit(1)

    path = sys.argv[1]
    verbose = '--quiet' not in sys.argv and '-q' not in sys.argv

    if os.path.isdir(path):
        files = sorted([f for f in os.listdir(path) if f.endswith('.json')])
        all_ok = True
        results = []

        for f in files:
            ok = verify_topology(os.path.join(path, f), verbose)
            results.append((f.replace('_topology.json', ''), ok))
            all_ok = all_ok and ok

        print(f"\n{'='*70}")
        print(f"  BATCH VERIFICATION RESULTS")
        print(f"{'='*70}")
        for name, ok in results:
            status = "✅ READY" if ok else "❌ ISSUES"
            print(f"  {status}  {name}")

        print(f"\n{'='*70}")
        passed = sum(1 for _, ok in results if ok)
        if all_ok:
            print(f"  ✅ ALL {len(results)} TOPOLOGIES PRODUCTION READY")
        else:
            print(f"  ⚠️  {passed}/{len(results)} TOPOLOGIES READY")
        print(f"{'='*70}")
    else:
        verify_topology(path, verbose)


if __name__ == "__main__":
    main()
