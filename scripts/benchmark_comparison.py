#!/usr/bin/env python3
"""
PRISM4D Systematic Benchmark: PRISM vs P2Rank vs Fpocket
=========================================================
Evaluates pocket detection accuracy using DCC, DCA, and Success@N metrics.
Computes Top-1, Top-3, Top-1+2, and full success curves.

Usage:
    python scripts/benchmark_comparison.py [--run-prism] [--run-p2rank] [--run-fpocket]
    python scripts/benchmark_comparison.py --eval-only  # just compute metrics from existing results
"""

import os
import sys
import json
import subprocess
import argparse
import numpy as np
from pathlib import Path
from collections import defaultdict
import csv
import time

# ==============================================================================
# GROUND TRUTH: Apo-Holo pairs with known binding sites
# Format: "apo_target": (holo_pdb, ligand_resname, holo_chain, apo_chain)
# ==============================================================================
# Format: "apo": (holo_pdb, lig_resname, holo_protein_chain, apo_chain, [lig_chain])
# lig_chain defaults to holo_protein_chain if not specified
TARGET_PAIRS = {
    "1a4q": ("1a4r", "GNH", "B", "A"),       # Neuraminidase homodimer — GNH on chain B
    "1ade": ("1gim", "IMP", "A", "A"),        # AdSS substrate
    "1bj4": ("1bjv", "GP8", "A", "A"),        # FKBP12 PLP
    "1btl": ("1btm", "PGA", "A", "A"),        # TEM1 beta-lactamase / Penicillin G
    "1ere_chainA": ("1err", "RAL", "A", "A"), # Estrogen receptor / Raloxifene
    "1g1f": ("1g1g", "PTR", "A", "A", "B"),   # PTP1B — PTR substrate on chain B, protein chain A
    "1hhp": ("1hvr", "XK2", "A", "A"),        # HIV protease
    "1w50": ("1w51", "L01", "A", "A"),         # BACE1
    "3k5v": ("3k5u", "PFQ", "A", "A"),        # Abl kinase / Nilotinib
    "4obe_mono": ("6oim", "MOV", "A", "A"),   # KRAS G12C / Sotorasib (SII-P pocket)
}

# Map from topology name to actual PDB filename (when they differ)
PDB_NAME_MAP = {
    "1ere_chainA": "1ere_clean",
    "4obe_mono": "4obe_mono",
}

# Paths
PRISM_BINARY = "target/release/nhs_rt_full"
TOPO_DIR = Path("e2e_validation_test/prep")
PDB_CACHE = Path("/tmp/pdb_cache")
P2RANK_BIN = Path("tools/p2rank/p2rank_2.5.1/prank")
RESULTS_DIR = Path("/tmp/benchmark_results")
PRISM_OUT_DIR = RESULTS_DIR / "prism"
P2RANK_OUT_DIR = RESULTS_DIR / "p2rank"
FPOCKET_OUT_DIR = RESULTS_DIR / "fpocket"

# Post-commit (84a01cd) PRISM result directories — ONLY use these
# Fresh benchmark runs go to PRISM_OUT_DIR/{target}/
# Fallback to these validated dirs if benchmark dir doesn't have results
PRISM_VALID_DIRS = {
    "1ere_chainA": Path("1ere_monomer_out4"),
    "1bj4": Path("1bj4_megafix_out"),
    "1hhp": Path("1hhp_mono_out2"),
    "1btl": Path("results_1btl_hmr"),
    "1w50": Path("results_1w50"),
    "3k5v": Path("results_3k5v"),
    "4obe_mono": Path("results_4obe_therm2"),
}

# Some result dirs use a different filename than the target name
PRISM_FILENAME_MAP = {
    "4obe_mono": "4obe",
    "1ere_chainA": "1ere_chainA",
}

# Canonical PRISM flags (all optimizations enabled)
PRISM_FLAGS = [
    "--fast", "--hysteresis", "--multi-stream", "8",
    "--spike-percentile", "95", "--prism-therm",
    "--fused-steps", "4", "--hmr", "--adaptive-dt", "-v"
]

# Additional holo references per target for multi-reference evaluation.
# Each target can have multiple known binding sites from different holo structures.
# Format: "target": [(holo_pdb, lig_resname, holo_chain, apo_chain[, lig_chain]), ...]
ADDITIONAL_REFERENCES = {
    "4obe_mono": [
        ("4m22", "22C", "B", "A"),      # SML-8-73-1, same SII-P pocket, secondary ref
    ],
    "1ade": [
        ("1gim", "GDP", "A", "A"),      # GDP substrate site (16.8A from IMP site — distinct pocket)
    ],
}

# Number of experimentally known binding sites per target (for Top-N+2 metric)
N_KNOWN_SITES = {
    "1a4q": 1,
    "1ade": 2,    # IMP site + GDP site
    "1bj4": 1,
    "1btl": 1,
    "1ere_chainA": 1,
    "1g1f": 1,
    "1hhp": 1,
    "1w50": 1,
    "3k5v": 1,
    "4obe_mono": 1,  # SII-P pocket (sotorasib)
}

POCKETMINER_RESULTS_PATH = RESULTS_DIR / "pocketminer_results.json"

# Contact distance threshold for contacting-residue identification
CONTACT_DISTANCE = 4.5  # Angstroms — standard ligand-residue contact cutoff


def ensure_dirs():
    """Create output directories."""
    for d in [RESULTS_DIR, PRISM_OUT_DIR, P2RANK_OUT_DIR, FPOCKET_OUT_DIR, PDB_CACHE]:
        d.mkdir(parents=True, exist_ok=True)


def download_holo_pdbs():
    """Download missing holo PDB files for evaluation (primary + additional refs)."""
    all_holos = set()
    for apo, (holo, *_) in TARGET_PAIRS.items():
        all_holos.add(holo)
    for refs in ADDITIONAL_REFERENCES.values():
        for ref in refs:
            all_holos.add(ref[0])

    for holo in all_holos:
        holo_path = PDB_CACHE / f"{holo}.pdb"
        if not holo_path.exists():
            # Try case-insensitive match
            found = False
            for f in PDB_CACHE.iterdir():
                if f.stem.lower() == holo.lower() and f.suffix == '.pdb':
                    found = True
                    break
            if not found:
                print(f"  Downloading {holo}.pdb...")
                subprocess.run([
                    'wget', '-q',
                    f'https://files.rcsb.org/download/{holo}.pdb',
                    '-O', str(holo_path)
                ], check=False)


def get_apo_pdb_path(target):
    """Get path to apo PDB file (extracted from topology or prep dir)."""
    # Direct match
    pdb_path = TOPO_DIR / f"{target}.pdb"
    if pdb_path.exists():
        return pdb_path
    # Check name map
    mapped = PDB_NAME_MAP.get(target, target)
    pdb_path = TOPO_DIR / f"{mapped}.pdb"
    if pdb_path.exists():
        return pdb_path
    # Try common variants
    for suffix in ['_clean', '_mono', '']:
        base = target.split('_')[0]  # e.g., "1ere" from "1ere_chainA"
        pdb_path = TOPO_DIR / f"{base}{suffix}.pdb"
        if pdb_path.exists():
            return pdb_path
    return None


def find_holo_pdb(holo_code):
    """Find holo PDB file (case-insensitive)."""
    for f in PDB_CACHE.iterdir():
        if f.stem.lower() == holo_code.lower() and f.suffix == '.pdb':
            return f
    return PDB_CACHE / f"{holo_code}.pdb"


# ==============================================================================
# STRUCTURAL ALIGNMENT & DISTANCE COMPUTATION
# ==============================================================================

def get_ca_atoms(pdb_path, chain):
    """Extract CA atoms from PDB file."""
    cas = {}
    with open(pdb_path, 'r') as f:
        for line in f:
            if line.startswith('ATOM') and line[12:16].strip() == 'CA' and line[21] == chain:
                resi = int(line[22:26])
                xyz = np.array([float(line[30:38]), float(line[38:46]), float(line[46:54])])
                cas[resi] = xyz
    return cas


def extract_ligand(pdb_path, lig_resname, lig_chain):
    """Extract ligand coordinates from holo structure."""
    coords = []
    with open(pdb_path, 'r') as f:
        for line in f:
            if lig_resname == "PEP" and line.startswith('ATOM') and line[21] == lig_chain:
                coords.append([float(line[30:38]), float(line[38:46]), float(line[46:54])])
            elif line.startswith('HETATM') and line[17:20].strip() == lig_resname and line[21] == lig_chain:
                coords.append([float(line[30:38]), float(line[38:46]), float(line[46:54])])
    return np.array(coords) if coords else None


def kabsch_align(P, Q):
    """Kabsch alignment: find R,t that minimizes |R@P + t - Q|."""
    cp, cq = P.mean(0), Q.mean(0)
    H = (P - cp).T @ (Q - cq)
    U, S, Vt = np.linalg.svd(H)
    d = np.linalg.det(Vt.T @ U.T)
    R = Vt.T @ np.diag([1, 1, d]) @ U.T
    t = cq - R @ cp
    return R, t


def align_and_transform_ligand(apo_pdb, holo_pdb, apo_chain, holo_chain, lig_resname, lig_chain):
    """Align apo to holo and transform ligand coords into apo frame."""
    hcas = get_ca_atoms(str(holo_pdb), holo_chain)
    pcas = get_ca_atoms(str(apo_pdb), apo_chain)

    lig_coords = extract_ligand(str(holo_pdb), lig_resname, lig_chain)
    if lig_coords is None or len(lig_coords) == 0:
        return None, None

    # Find best residue offset
    best_m, best_off = 0, 0
    for off in range(-50, 51):
        m = sum(1 for r in hcas if (r + off) in pcas)
        if m > best_m:
            best_m, best_off = m, off

    if best_m < 20:
        # Fallback: use raw ligand coords
        lig_centroid = np.mean(lig_coords, axis=0)
        return lig_centroid, lig_coords

    P, Q = [], []
    for r, c in hcas.items():
        if (r + best_off) in pcas:
            P.append(c)
            Q.append(pcas[r + best_off])

    P, Q = np.array(P), np.array(Q)
    R, t = kabsch_align(P, Q)
    aligned_lig = np.array([R @ c + t for c in lig_coords])

    lig_centroid = np.mean(aligned_lig, axis=0)
    return lig_centroid, aligned_lig


def compute_dcc_dca(pocket_centroid, lig_centroid, aligned_lig):
    """Compute DCC and DCA distances."""
    dcc = np.linalg.norm(np.array(pocket_centroid) - lig_centroid)
    dca = np.min(np.linalg.norm(aligned_lig - np.array(pocket_centroid), axis=1))
    return dcc, dca


def get_contacting_residues(apo_pdb, apo_chain, aligned_lig, contact_dist=CONTACT_DISTANCE):
    """Find apo residues whose atoms are within contact_dist of the aligned ligand.
    Returns centroid of contacting CA atoms."""
    # Get all heavy atoms per residue
    residue_atoms = {}  # resi -> list of xyz
    residue_cas = {}    # resi -> CA xyz
    with open(str(apo_pdb), 'r') as f:
        for line in f:
            if line.startswith('ATOM') and line[21] == apo_chain:
                resi = int(line[22:26])
                xyz = np.array([float(line[30:38]), float(line[38:46]), float(line[46:54])])
                atom_name = line[12:16].strip()
                if atom_name[0] != 'H':  # Skip hydrogens
                    if resi not in residue_atoms:
                        residue_atoms[resi] = []
                    residue_atoms[resi].append(xyz)
                    if atom_name == 'CA':
                        residue_cas[resi] = xyz

    # Find residues with any atom within contact_dist of any ligand atom
    contacting = []
    for resi, atoms in residue_atoms.items():
        atoms_arr = np.array(atoms)
        # Min distance between any residue atom and any ligand atom
        for latom in aligned_lig:
            dists = np.linalg.norm(atoms_arr - latom, axis=1)
            if np.min(dists) <= contact_dist:
                if resi in residue_cas:
                    contacting.append(residue_cas[resi])
                break

    if not contacting:
        return None
    return np.mean(contacting, axis=0)


def compute_multi_reference_distance(pocket_centroid, lig_centroid, aligned_lig, contact_centroid):
    """Multi-reference evaluation: score against ALL legitimate holo-derived references.

    Returns dict with:
      - dcc: distance to ligand centroid (standard DCC)
      - dca: distance to closest ligand atom (standard DCA)
      - dcc_contact: distance to contacting-residue centroid
      - best_dist: minimum across all reference distances
      - hit_type: which reference was best ('lig_centroid', 'contact_centroid', 'closest_atom')
    """
    pc = np.array(pocket_centroid)

    dcc = float(np.linalg.norm(pc - lig_centroid))
    dca = float(np.min(np.linalg.norm(aligned_lig - pc, axis=1)))

    dcc_contact = float(np.linalg.norm(pc - contact_centroid)) if contact_centroid is not None else 999.0

    # Best distance across all references
    best_dist = min(dcc, dca, dcc_contact)
    if best_dist == dca:
        hit_type = 'closest_atom'
    elif best_dist == dcc_contact:
        hit_type = 'contact_centroid'
    else:
        hit_type = 'lig_centroid'

    return {
        'dcc': dcc,
        'dca': dca,
        'dcc_contact': dcc_contact,
        'best_dist': best_dist,
        'hit_type': hit_type,
    }


# ==============================================================================
# PRISM4D RUNNER & PARSER
# ==============================================================================

def run_prism(target):
    """Run PRISM4D on a single target."""
    topo = TOPO_DIR / f"{target}.topology.json"
    if not topo.exists():
        return False
    out_dir = PRISM_OUT_DIR / target
    out_dir.mkdir(parents=True, exist_ok=True)

    # Check if already completed
    bs_json = out_dir / f"{target}.binding_sites.json"
    if bs_json.exists():
        print(f"  [PRISM] {target}: already completed")
        return True

    cmd = [PRISM_BINARY, "-t", str(topo), "-o", str(out_dir)] + PRISM_FLAGS
    print(f"  [PRISM] Running {target}...")
    t0 = time.time()
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
    elapsed = time.time() - t0
    print(f"  [PRISM] {target}: completed in {elapsed:.0f}s")

    # Check for output
    if bs_json.exists():
        return True
    # Try to find binding_sites.json in subdirectory
    for f in out_dir.rglob("*.binding_sites.json"):
        return True
    return False


def parse_prism_results(target):
    """Parse PRISM binding sites, return list of (centroid, score, site_id).
    Only uses post-commit (84a01cd) results."""
    sites = []
    # Check validated post-commit directories first, then PRISM_OUT_DIR
    search_dirs = []
    if target in PRISM_VALID_DIRS:
        search_dirs.append(PRISM_VALID_DIRS[target])
    search_dirs.extend([PRISM_OUT_DIR / target, PRISM_OUT_DIR])

    for base_dir in search_dirs:
        fname = PRISM_FILENAME_MAP.get(target, target)
        bs_json = base_dir / f"{fname}.binding_sites.json"
        if not bs_json.exists():
            bs_json = base_dir / f"{target}.binding_sites.json"
        if bs_json.exists():
            with open(bs_json) as f:
                data = json.load(f)
            for s in data.get("sites", []):
                sites.append({
                    'centroid': s['centroid'],
                    'score': s.get('quality_score', s.get('druggability', 0)),
                    'id': s['id'],
                    'druggability': s.get('druggability', 0),
                    'therm_class': s.get('therm_class', ''),
                    'spike_count': s.get('spike_count', 0),
                })
            break
    # Sort by quality_score descending
    sites.sort(key=lambda s: s['score'], reverse=True)
    return sites


# ==============================================================================
# P2RANK RUNNER & PARSER
# ==============================================================================

def run_p2rank(target):
    """Run P2Rank on a single target."""
    pdb_path = get_apo_pdb_path(target)
    if pdb_path is None:
        return False
    out_dir = P2RANK_OUT_DIR / target
    out_dir.mkdir(parents=True, exist_ok=True)

    # P2Rank names output as {filename}_predictions.csv (includes .pdb extension in name)
    pred_csv = out_dir / f"{pdb_path.name}_predictions.csv"
    if pred_csv.exists():
        print(f"  [P2Rank] {target}: already completed")
        return True

    cmd = [str(P2RANK_BIN), "predict", "-f", str(pdb_path), "-o", str(out_dir)]
    print(f"  [P2Rank] Running {target}...")
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    return pred_csv.exists()


def parse_p2rank_results(target):
    """Parse P2Rank predictions, return list of (centroid, score, rank)."""
    sites = []
    pdb_path = get_apo_pdb_path(target)
    if pdb_path is None:
        return sites

    # P2Rank names files as {filename}_predictions.csv
    pred_csv = P2RANK_OUT_DIR / target / f"{pdb_path.name}_predictions.csv"
    if not pred_csv.exists():
        # Try alternate naming
        pred_csv = P2RANK_OUT_DIR / target / f"{pdb_path.stem}_predictions.csv"
    if not pred_csv.exists():
        return sites

    with open(pred_csv) as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                centroid = [
                    float(row.get('   center_x', row.get('center_x', '0')).strip()),
                    float(row.get('   center_y', row.get('center_y', '0')).strip()),
                    float(row.get('   center_z', row.get('center_z', '0')).strip()),
                ]
                score = float(row.get('   score', row.get('score', '0')).strip())
                rank = int(row.get('   rank', row.get('rank', '0')).strip())
                sites.append({
                    'centroid': centroid,
                    'score': score,
                    'id': rank,
                })
            except (ValueError, KeyError):
                continue
    # Already sorted by rank
    return sites


# ==============================================================================
# FPOCKET RUNNER & PARSER
# ==============================================================================

def run_fpocket(target):
    """Run Fpocket on a single target."""
    pdb_path = get_apo_pdb_path(target)
    if pdb_path is None:
        return False

    out_dir = FPOCKET_OUT_DIR / target
    out_dir.mkdir(parents=True, exist_ok=True)

    # Fpocket creates output in the same directory as input with _out suffix
    # We need to copy PDB to our output dir first
    local_pdb = out_dir / pdb_path.name
    if not local_pdb.exists():
        import shutil
        shutil.copy2(pdb_path, local_pdb)

    # Check if already completed
    fpocket_dir = out_dir / f"{pdb_path.stem}_out"
    if fpocket_dir.exists():
        print(f"  [Fpocket] {target}: already completed")
        return True

    print(f"  [Fpocket] Running {target}...")
    # Use conda fpocket (snap version can't access /tmp)
    fpocket_bin = "/home/diddy/miniconda3/bin/fpocket"
    if not os.path.exists(fpocket_bin):
        fpocket_bin = "fpocket"  # fallback to PATH
    result = subprocess.run(
        [fpocket_bin, "-f", str(local_pdb)],
        capture_output=True, text=True, timeout=120,
        cwd=str(out_dir)
    )
    return fpocket_dir.exists()


def parse_fpocket_results(target):
    """Parse Fpocket results, return list of (centroid, score, rank)."""
    sites = []
    pdb_path = get_apo_pdb_path(target)
    if pdb_path is None:
        return sites

    fpocket_dir = FPOCKET_OUT_DIR / target / f"{pdb_path.stem}_out"
    info_file = fpocket_dir / f"{pdb_path.stem}_info.txt"

    if not info_file.exists():
        return sites

    # Parse Fpocket info file for pocket centroids and scores
    pockets_dir = fpocket_dir / "pockets"
    if not pockets_dir.exists():
        return sites

    # Parse each pocket PDB for centroid
    pocket_files = sorted(pockets_dir.glob("pocket*_atm.pdb"))
    for i, pf in enumerate(pocket_files):
        coords = []
        with open(pf) as f:
            for line in f:
                if line.startswith('ATOM') or line.startswith('HETATM'):
                    x = float(line[30:38])
                    y = float(line[38:46])
                    z = float(line[46:54])
                    coords.append([x, y, z])
        if coords:
            centroid = np.mean(coords, axis=0).tolist()
            # Parse druggability score from info file
            score = parse_fpocket_score(info_file, i + 1)
            sites.append({
                'centroid': centroid,
                'score': score,
                'id': i + 1,
            })

    # Sort by score descending
    sites.sort(key=lambda s: s['score'], reverse=True)
    return sites


def parse_fpocket_score(info_file, pocket_num):
    """Extract druggability score for a specific pocket from fpocket info file."""
    with open(info_file) as f:
        content = f.read()

    # Find the section for this pocket
    marker = f"Pocket {pocket_num} :"
    idx = content.find(marker)
    if idx < 0:
        return 0.0

    # Look for druggability score
    section = content[idx:idx + 2000]
    for line in section.split('\n'):
        if 'Druggability Score' in line or 'Drug Score' in line:
            parts = line.split(':')
            if len(parts) >= 2:
                try:
                    return float(parts[-1].strip())
                except ValueError:
                    pass
    return 0.0


# ==============================================================================
# POCKETMINER PARSER
# ==============================================================================

def parse_pocketminer_results(target):
    """Parse PocketMiner results from pre-computed JSON."""
    sites = []
    if not POCKETMINER_RESULTS_PATH.exists():
        return sites
    with open(POCKETMINER_RESULTS_PATH) as f:
        data = json.load(f)
    for pocket in data.get(target, []):
        sites.append({
            'centroid': pocket['centroid'],
            'score': pocket['score'],
            'id': pocket['id'],
        })
    # Already sorted by score descending
    return sites


# ==============================================================================
# EVALUATION METRICS
# ==============================================================================

def evaluate_method(method_name, parse_fn, targets, all_site_refs, thresholds=None):
    """Evaluate a method across all targets using multi-reference scoring.

    For each predicted site, scores against ALL legitimate holo-derived references
    (primary + additional references). A prediction is correct if it satisfies
    the threshold against ANY reference for ANY known site.

    all_site_refs: dict target -> list of (lig_centroid, aligned_lig, contact_centroid, site_name)
    """
    if thresholds is None:
        thresholds = [4.0, 5.0, 6.0, 8.0, 10.0, 12.0, 15.0]

    results = {}
    for target in targets:
        sites = parse_fn(target)
        if not sites or target not in all_site_refs:
            results[target] = None
            continue

        refs = all_site_refs[target]
        if not refs:
            results[target] = None
            continue

        # Compute multi-reference metrics for each predicted site
        site_metrics = []
        for s in sites:
            # Evaluate against ALL known site references, take best
            best_overall = {'dcc': 999, 'dca': 999, 'dcc_contact': 999, 'best_dist': 999,
                            'hit_type': 'none', 'matched_site': 'none'}
            for lig_centroid, aligned_lig, contact_centroid, site_name in refs:
                if lig_centroid is None:
                    continue
                ref = compute_multi_reference_distance(
                    s['centroid'], lig_centroid, aligned_lig, contact_centroid
                )
                if ref['best_dist'] < best_overall['best_dist']:
                    best_overall = {**ref, 'matched_site': site_name}

            site_metrics.append({
                'id': s['id'],
                'score': s.get('score', 0),
                'dcc': best_overall['dcc'],
                'dca': best_overall['dca'],
                'dcc_contact': best_overall['dcc_contact'],
                'best_dist': best_overall['best_dist'],
                'hit_type': best_overall['hit_type'],
                'matched_site': best_overall.get('matched_site', ''),
                'centroid': s['centroid'],
            })
        results[target] = site_metrics

    return results


def compute_success_rates(results, targets, max_n=10):
    """Compute Success@N for different DCC thresholds."""
    thresholds = [4.0, 5.0, 6.0, 8.0, 10.0]

    success = {}
    for thresh in thresholds:
        success[thresh] = {}
        for n in range(1, max_n + 1):
            count = 0
            total = 0
            for target in targets:
                if results.get(target) is None:
                    continue
                total += 1
                metrics = results[target][:n]  # Top-N sites
                if any(m['dcc'] <= thresh for m in metrics):
                    count += 1
            success[thresh][n] = count / total if total > 0 else 0

    return success


def print_comparison_table(all_results, targets, all_site_refs):
    """Print comprehensive comparison table with multi-reference evaluation."""
    methods = list(all_results.keys())

    print("\n" + "=" * 120)
    print("  SYSTEMATIC BENCHMARK: Multi-Reference Pocket Detection Accuracy")
    print("  (correct if ANY holo-derived reference — ligand centroid, contact centroid, closest atom — satisfies threshold)")
    print("  Ground truth: ONLY confirmed, properly-aligned holo co-crystal structures")
    print("=" * 120)

    # Per-target results: show Top-1 best_dist and DCC
    print(f"\n{'Target':<14}", end="")
    for method in methods:
        print(f"| {method:^26}", end="")
    print(f"| {'Refs':^8}")
    print("-" * (14 + 28 * len(methods) + 10))

    valid_targets = []
    for target in targets:
        if target not in all_site_refs or not all_site_refs[target]:
            continue
        valid_targets.append(target)

        print(f"{target:<14}", end="")
        for method in methods:
            res = all_results[method].get(target)
            if res is None or len(res) == 0:
                print(f"| {'N/A':^26}", end="")
            else:
                top1 = res[0]
                # Best match in top-10
                best10 = min(m['best_dist'] for m in res[:10]) if res else 999
                status = "OK" if top1['best_dist'] <= 8.0 else "  "
                print(f"| {top1['best_dist']:5.1f} ({best10:4.1f}) [{top1['hit_type'][:4]:4s}]{status:2s}", end="")

        n_refs = len(all_site_refs.get(target, []))
        n_sites = N_KNOWN_SITES.get(target, 1)
        print(f"| {n_sites}s/{n_refs}r")

    # Multi-reference success rates
    for thresh_label, thresh in [("4A", 4.0), ("5A", 5.0), ("8A", 8.0), ("10A", 10.0)]:
        print(f"\n{'─' * 90}")
        print(f"  Success Rate @ best_dist <= {thresh_label} (multi-reference, all valid holo sites)")
        print(f"{'─' * 90}")
        print(f"  {'Metric':<20}", end="")
        for method in methods:
            print(f"  {method:>12}", end="")
        print()

        for n, label in [(1, "Top-1"), (3, "Top-3"), (None, "Top-N+2"), (10, "Top-10")]:
            print(f"  {label:<20}", end="")
            for method in methods:
                res = all_results[method]
                count = 0
                total = 0
                for target in valid_targets:
                    if res.get(target) is None:
                        continue
                    total += 1
                    if n is None:
                        # Top-N+2: N = number of known sites for this target
                        n_sites = N_KNOWN_SITES.get(target, 1)
                        k = n_sites + 2
                    else:
                        k = n
                    metrics = res[target][:k]
                    if any(m['best_dist'] <= thresh for m in metrics):
                        count += 1
                rate = count / total * 100 if total > 0 else 0
                print(f"  {rate:>10.1f}%", end="")
            print()

    # Traditional DCC-only success (for comparison with literature)
    print(f"\n{'─' * 90}")
    print(f"  Traditional Success Rate @ DCC <= 5A (ligand-centroid only)")
    print(f"{'─' * 90}")
    print(f"  {'Metric':<20}", end="")
    for method in methods:
        print(f"  {method:>12}", end="")
    print()

    for n, label in [(1, "Top-1"), (3, "Top-3"), (10, "Top-10")]:
        print(f"  {label:<20}", end="")
        for method in methods:
            res = all_results[method]
            count = 0
            total = 0
            for target in valid_targets:
                if res.get(target) is None:
                    continue
                total += 1
                metrics = res[target][:n]
                if any(m['dcc'] <= 5.0 for m in metrics):
                    count += 1
            rate = count / total * 100 if total > 0 else 0
            print(f"  {rate:>10.1f}%", end="")
        print()

    # Average metrics
    print(f"\n{'─' * 90}")
    print(f"  Average Metrics")
    print(f"{'─' * 90}")
    for metric_name, metric_key in [("Mean Top-1 best_dist", "best_dist"),
                                      ("Median Top-1 best_dist", "best_dist"),
                                      ("Mean Top-1 DCC", "dcc"),
                                      ("Median Top-1 DCC", "dcc")]:
        print(f"  {metric_name:<24}", end="")
        for method in methods:
            vals = []
            for target in valid_targets:
                res = all_results[method].get(target)
                if res and len(res) > 0:
                    vals.append(res[0][metric_key])
            if vals:
                if "Mean" in metric_name:
                    v = np.mean(vals)
                else:
                    v = np.median(vals)
                print(f"  {v:>10.1f}A", end="")
            else:
                print(f"  {'N/A':>11}", end="")
        print()

    # Detailed per-target breakdown
    print(f"\n{'─' * 90}")
    print(f"  Detailed Per-Target Breakdown")
    print(f"{'─' * 90}")
    print(f"  {'Target':<12} {'Method':<10} {'Top1-DCC':>8} {'Top1-DCA':>8} {'Top1-CCen':>9} {'Top1-Best':>9} {'Top10':>8} {'HitType':<12}")
    for target in valid_targets:
        for method in methods:
            res = all_results[method].get(target)
            if res is None or len(res) == 0:
                print(f"  {target:<12} {method:<10} {'N/A':>8}")
                continue
            top1 = res[0]
            best10 = min(m['best_dist'] for m in res[:10])
            print(f"  {target:<12} {method:<10} {top1['dcc']:>8.1f} {top1['dca']:>8.1f} "
                  f"{top1.get('dcc_contact', 999):>9.1f} {top1['best_dist']:>9.1f} "
                  f"{best10:>8.1f} {top1['hit_type']:<12}")

    print("\n" + "=" * 120)
    return valid_targets


def save_results_json(all_results, targets, lig_data, output_path):
    """Save full results to JSON for later analysis."""
    output = {
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'targets': targets,
        'methods': {},
    }
    for method, results in all_results.items():
        method_data = {}
        for target in targets:
            if results.get(target) is not None:
                method_data[target] = results[target]
        output['methods'][method] = method_data

    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2, default=lambda x: float(x) if isinstance(x, np.floating) else x)
    print(f"\nResults saved to: {output_path}")


# ==============================================================================
# MAIN
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(description="PRISM4D Systematic Benchmark")
    parser.add_argument("--run-prism", action="store_true", help="Run PRISM4D on all targets")
    parser.add_argument("--run-p2rank", action="store_true", help="Run P2Rank on all targets")
    parser.add_argument("--run-fpocket", action="store_true", help="Run Fpocket on all targets")
    parser.add_argument("--eval-only", action="store_true", help="Only evaluate existing results")
    parser.add_argument("--targets", nargs="*", help="Specific targets to evaluate")
    args = parser.parse_args()

    ensure_dirs()

    # Determine target list
    if args.targets:
        targets = args.targets
    else:
        # Use targets that have proper topologies
        targets = []
        for t in TARGET_PAIRS:
            topo = TOPO_DIR / f"{t}.topology.json"
            if topo.exists():
                targets.append(t)
        if not targets:
            print("ERROR: No targets with topologies found")
            sys.exit(1)

    print(f"Benchmark targets ({len(targets)}): {', '.join(targets)}")

    # Download missing holo PDBs
    print("\n[1] Downloading missing holo PDBs...")
    download_holo_pdbs()

    # Run methods
    if args.run_prism:
        print("\n[2a] Running PRISM4D...")
        for t in targets:
            run_prism(t)

    if args.run_p2rank:
        print("\n[2b] Running P2Rank...")
        for t in targets:
            run_p2rank(t)

    if args.run_fpocket:
        print("\n[2c] Running Fpocket...")
        for t in targets:
            run_fpocket(t)

    # Compute ligand data (alignment + transformation + contact centroids)
    # Multi-reference: primary + additional holo references per target
    print("\n[3] Computing ligand alignments + contact centroids (multi-reference)...")
    all_site_refs = {}  # target -> list of (lig_centroid, aligned_lig, contact_centroid, site_name)
    for target in targets:
        if target not in TARGET_PAIRS:
            continue
        apo_pdb = get_apo_pdb_path(target)
        if apo_pdb is None:
            print(f"  {target}: missing apo PDB")
            continue

        refs = []
        # Primary reference
        pair = TARGET_PAIRS[target]
        holo, lig_name, h_chain, a_chain = pair[:4]
        lig_chain = pair[4] if len(pair) > 4 else h_chain
        holo_pdb = find_holo_pdb(holo)

        if holo_pdb.exists():
            lig_centroid, aligned_lig = align_and_transform_ligand(
                apo_pdb, holo_pdb, a_chain, h_chain, lig_name, lig_chain
            )
            if lig_centroid is not None:
                contact_centroid = get_contacting_residues(apo_pdb, a_chain, aligned_lig)
                site_name = f"{holo.upper()}_{lig_name}"
                refs.append((lig_centroid, aligned_lig, contact_centroid, site_name))
                # Report alignment quality
                hcas = get_ca_atoms(str(holo_pdb), h_chain)
                pcas = get_ca_atoms(str(apo_pdb), a_chain)
                best_m = 0
                for off in range(-50, 51):
                    m = sum(1 for r in hcas if (r + off) in pcas)
                    if m > best_m:
                        best_m = m
                n_contact = "yes" if contact_centroid is not None else "NO"
                print(f"  {target}: [{site_name}] aligned ({len(aligned_lig)} lig atoms, "
                      f"{best_m} CA pairs, contact_cen={n_contact})")
            else:
                print(f"  {target}: ligand {lig_name} not found in {holo}")
        else:
            print(f"  {target}: holo PDB {holo} not found")

        # Additional references
        for add_ref in ADDITIONAL_REFERENCES.get(target, []):
            ah, al, ahc, aac = add_ref[:4]
            alc = add_ref[4] if len(add_ref) > 4 else ahc
            ah_pdb = find_holo_pdb(ah)
            if ah_pdb.exists():
                lc, alig = align_and_transform_ligand(apo_pdb, ah_pdb, aac, ahc, al, alc)
                if lc is not None:
                    cc = get_contacting_residues(apo_pdb, aac, alig)
                    sname = f"{ah.upper()}_{al}"
                    refs.append((lc, alig, cc, sname))
                    print(f"  {target}: [{sname}] additional ref ({len(alig)} lig atoms)")

        if refs:
            all_site_refs[target] = refs
            n_known = N_KNOWN_SITES.get(target, 1)
            if n_known > 1:
                print(f"  {target}: {len(refs)} total references across {n_known} known sites")

    # Evaluate all methods
    print("\n[4] Evaluating methods...")
    all_results = {}

    # PRISM4D — only post-commit (84a01cd) results
    prism_results = evaluate_method("PRISM4D", parse_prism_results, targets, all_site_refs)
    all_results['PRISM4D'] = prism_results

    # P2Rank
    p2rank_results = evaluate_method("P2Rank", parse_p2rank_results, targets, all_site_refs)
    all_results['P2Rank'] = p2rank_results

    # Fpocket
    fpocket_results = evaluate_method("Fpocket", parse_fpocket_results, targets, all_site_refs)
    all_results['Fpocket'] = fpocket_results

    # PocketMiner (if results available)
    if POCKETMINER_RESULTS_PATH.exists():
        pm_results = evaluate_method("PocketMiner", parse_pocketminer_results, targets, all_site_refs)
        all_results['PocketMiner'] = pm_results
    else:
        print("  [PocketMiner] No results found — skipping")

    # Print comparison
    valid_targets = print_comparison_table(all_results, targets, all_site_refs)

    # Save JSON results
    save_results_json(all_results, targets, all_site_refs, RESULTS_DIR / "benchmark_results.json")


if __name__ == "__main__":
    main()
