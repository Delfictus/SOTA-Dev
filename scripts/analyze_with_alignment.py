#!/usr/bin/env python3
"""
PRISM-4D Cryptic Pocket Analysis with Proper RMSD Alignment

This script:
1. Loads conformational ensemble
2. Performs Kabsch alignment of each frame to reference
3. Computes true RMSF (after removing rigid-body motion)
4. Identifies flexible regions and cryptic pocket candidates
5. Runs volume-based pocket detection with fpocket

Usage:
    python scripts/analyze_with_alignment.py \
        --ensemble data/ensembles/6M0J_RBD_10ns.pdb \
        --output results/6M0J_10ns_analysis \
        --target-residues "474,475,476,477,478,479,480" \
        --n-pocket-frames 20
"""

import argparse
import json
import numpy as np
from pathlib import Path
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass
import subprocess
import tempfile
import shutil

# ============================================================================
# KABSCH ALIGNMENT (Optimal RMSD Superposition)
# ============================================================================

def kabsch_align(mobile: np.ndarray, target: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute optimal rotation matrix to align mobile onto target using Kabsch algorithm.

    Args:
        mobile: (N, 3) coordinates to be rotated
        target: (N, 3) reference coordinates

    Returns:
        rotation: (3, 3) optimal rotation matrix
        translation: (3,) translation vector
        aligned: (N, 3) aligned mobile coordinates
    """
    assert mobile.shape == target.shape

    # Center both structures
    mobile_center = mobile.mean(axis=0)
    target_center = target.mean(axis=0)

    mobile_centered = mobile - mobile_center
    target_centered = target - target_center

    # Compute covariance matrix
    H = mobile_centered.T @ target_centered

    # SVD
    U, S, Vt = np.linalg.svd(H)

    # Optimal rotation (handle reflection)
    d = np.linalg.det(Vt.T @ U.T)
    correction = np.array([[1, 0, 0], [0, 1, 0], [0, 0, d]])
    rotation = Vt.T @ correction @ U.T

    # Apply transformation
    aligned = (mobile_centered @ rotation.T) + target_center

    # Translation (for completeness)
    translation = target_center - (rotation @ mobile_center)

    return rotation, translation, aligned


def compute_rmsd(coords1: np.ndarray, coords2: np.ndarray) -> float:
    """Compute RMSD between two coordinate sets."""
    diff = coords1 - coords2
    return np.sqrt(np.mean(np.sum(diff**2, axis=1)))


def align_ensemble_to_reference(
    frames: List[np.ndarray],
    reference: np.ndarray,
    align_selection: Optional[np.ndarray] = None
) -> Tuple[List[np.ndarray], List[float]]:
    """
    Align all frames in ensemble to reference structure.

    Args:
        frames: List of (N, 3) coordinate arrays
        reference: (N, 3) reference coordinates
        align_selection: Boolean mask for atoms to use in alignment (e.g., backbone only)
                        If None, use all atoms

    Returns:
        aligned_frames: List of aligned (N, 3) coordinate arrays
        rmsds: List of RMSD values after alignment
    """
    aligned_frames = []
    rmsds = []

    for frame in frames:
        if align_selection is not None:
            # Align using subset (e.g., backbone)
            mobile_subset = frame[align_selection]
            target_subset = reference[align_selection]
            rotation, translation, _ = kabsch_align(mobile_subset, target_subset)

            # Apply rotation to ALL atoms
            frame_centered = frame - frame[align_selection].mean(axis=0)
            aligned = (frame_centered @ rotation.T) + reference[align_selection].mean(axis=0)
        else:
            rotation, translation, aligned = kabsch_align(frame, reference)

        aligned_frames.append(aligned)
        rmsds.append(compute_rmsd(aligned, reference))

    return aligned_frames, rmsds


# ============================================================================
# RMSF CALCULATION (After Alignment)
# ============================================================================

def compute_rmsf_aligned(aligned_frames: List[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute RMSF from aligned ensemble.

    This is the CORRECT way to compute RMSF - after removing rigid-body motion.

    Args:
        aligned_frames: List of aligned (N, 3) coordinate arrays

    Returns:
        rmsf: (N,) RMSF per atom in Angstroms
        mean_structure: (N, 3) average structure
    """
    coords = np.array(aligned_frames)  # (n_frames, n_atoms, 3)

    # Mean structure (after alignment)
    mean_structure = coords.mean(axis=0)  # (n_atoms, 3)

    # Deviations from mean
    deviations = coords - mean_structure  # (n_frames, n_atoms, 3)

    # RMSF = sqrt(mean(|r - r_mean|^2))
    squared_deviations = np.sum(deviations**2, axis=2)  # (n_frames, n_atoms)
    mean_squared_dev = squared_deviations.mean(axis=0)  # (n_atoms,)
    rmsf = np.sqrt(mean_squared_dev)  # (n_atoms,)

    return rmsf, mean_structure


# ============================================================================
# PDB PARSING
# ============================================================================

@dataclass
class AtomRecord:
    index: int
    name: str
    resname: str
    chain: str
    resid: int
    coords: np.ndarray
    element: str
    is_ca: bool
    is_backbone: bool


def parse_pdb_frame(lines: List[str]) -> Tuple[List[AtomRecord], np.ndarray]:
    """Parse ATOM records from PDB lines."""
    atoms = []
    coords = []

    backbone_names = {'N', 'CA', 'C', 'O'}

    for line in lines:
        if line.startswith('ATOM'):
            atom_name = line[12:16].strip()
            resname = line[17:20].strip()
            chain = line[21]
            try:
                resid = int(line[22:26])
            except ValueError:
                resid = len(atoms) + 1
            x = float(line[30:38])
            y = float(line[38:46])
            z = float(line[46:54])
            element = line[76:78].strip() if len(line) > 76 else atom_name[0]

            atom = AtomRecord(
                index=len(atoms),
                name=atom_name,
                resname=resname,
                chain=chain,
                resid=resid,
                coords=np.array([x, y, z]),
                element=element,
                is_ca=(atom_name == 'CA'),
                is_backbone=(atom_name in backbone_names)
            )

            atoms.append(atom)
            coords.append([x, y, z])

    return atoms, np.array(coords)


def parse_ensemble_pdb(pdb_path: str) -> Tuple[List[AtomRecord], List[np.ndarray]]:
    """
    Parse multi-MODEL PDB file.

    Returns:
        atoms: List of AtomRecord from first frame (for metadata)
        frames: List of (N, 3) coordinate arrays
    """
    frames = []
    atoms = None
    current_lines = []

    print(f"   Parsing PDB file...")

    with open(pdb_path, 'r') as f:
        for line in f:
            if line.startswith('MODEL'):
                current_lines = []
            elif line.startswith('ENDMDL'):
                if current_lines:
                    frame_atoms, coords = parse_pdb_frame(current_lines)
                    if atoms is None:
                        atoms = frame_atoms
                    frames.append(coords)
            elif line.startswith('ATOM'):
                current_lines.append(line)

    # Handle single-model PDB (no MODEL/ENDMDL)
    if not frames and current_lines:
        atoms, coords = parse_pdb_frame(current_lines)
        frames.append(coords)

    return atoms, frames


# ============================================================================
# POCKET DETECTION WITH FPOCKET
# ============================================================================

def write_pdb_frame(atoms: List[AtomRecord], coords: np.ndarray, output_path: str):
    """Write coordinates to PDB file."""
    with open(output_path, 'w') as f:
        for i, (atom, xyz) in enumerate(zip(atoms, coords)):
            f.write(
                f"ATOM  {i+1:5d} {atom.name:^4s} {atom.resname:3s} {atom.chain}{atom.resid:4d}    "
                f"{xyz[0]:8.3f}{xyz[1]:8.3f}{xyz[2]:8.3f}  1.00  0.00          {atom.element:>2s}\n"
            )
        f.write("END\n")


def run_fpocket(pdb_path: str, output_dir: str) -> Optional[Dict]:
    """Run fpocket and parse results."""

    fpocket_path = shutil.which('fpocket')
    if fpocket_path is None:
        return None

    try:
        # Run fpocket
        result = subprocess.run(
            [fpocket_path, '-f', pdb_path],
            capture_output=True,
            text=True,
            timeout=120,
            cwd=output_dir
        )

        # Find output directory
        pdb_stem = Path(pdb_path).stem
        pocket_dir = None

        # fpocket creates output in same dir as input
        for d in Path(output_dir).glob("*_out"):
            if pdb_stem in str(d):
                pocket_dir = d
                break

        if pocket_dir is None or not pocket_dir.exists():
            return None

        # Parse pocket info
        info_file = list(pocket_dir.glob("*_info.txt"))
        if not info_file:
            return None

        pockets = parse_fpocket_output(info_file[0])

        return {
            'n_pockets': len(pockets),
            'pockets': pockets,
            'output_dir': str(pocket_dir)
        }

    except Exception as e:
        return None


def parse_fpocket_output(info_path: Path) -> List[Dict]:
    """Parse fpocket info file."""
    pockets = []
    current = None

    with open(info_path) as f:
        for line in f:
            line = line.strip()
            if line.startswith('Pocket'):
                if current:
                    pockets.append(current)
                parts = line.split()
                try:
                    pocket_id = int(parts[1].rstrip(':'))
                except (IndexError, ValueError):
                    pocket_id = len(pockets) + 1
                current = {'pocket_id': pocket_id}
            elif current and ':' in line:
                parts = line.split(':', 1)
                if len(parts) == 2:
                    key = parts[0].strip().lower().replace(' ', '_').replace('-', '_')
                    val = parts[1].strip()
                    try:
                        val = float(val) if '.' in val else int(val)
                    except ValueError:
                        pass
                    current[key] = val

    if current:
        pockets.append(current)

    return pockets


# ============================================================================
# KNOWN FUNCTIONAL SITES (SARS-CoV-2 RBD)
# ============================================================================

ESCAPE_MUTATIONS = {
    346: ("R346K", "Omicron BA.1"),
    371: ("S371L", "Omicron"),
    373: ("S373P", "Omicron"),
    375: ("S375F", "Omicron"),
    417: ("K417N", "Beta/Omicron, Class 1 Ab escape"),
    440: ("K440N", "Omicron"),
    446: ("G446S", "Omicron BA.1"),
    452: ("L452R", "Delta, Class 3 Ab escape"),
    477: ("S477N", "Omicron, ACE2 interface"),
    478: ("T478K", "Delta/Omicron"),
    484: ("E484K/A", "Beta/Gamma/Omicron, Class 2 Ab escape"),
    493: ("Q493R", "Omicron"),
    496: ("G496S", "Omicron"),
    498: ("Q498R", "Omicron"),
    501: ("N501Y", "Alpha/Beta/Gamma/Omicron, ACE2 affinity"),
    505: ("Y505H", "Omicron"),
}

ACE2_INTERFACE = [
    417, 446, 449, 453, 455, 456, 475, 476, 477, 484, 486,
    487, 489, 490, 493, 494, 495, 496, 498, 500, 501, 502, 505
]


# ============================================================================
# MAIN ANALYSIS
# ============================================================================

def main(args):
    print("=" * 70)
    print("PRISM-4D CRYPTIC POCKET ANALYSIS")
    print("With Proper RMSD Alignment (Kabsch Algorithm)")
    print("=" * 70)

    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    frames_dir = output_dir / "frames"
    frames_dir.mkdir(exist_ok=True)

    # Parse ensemble
    print(f"\n{'='*70}")
    print("STEP 1: LOADING ENSEMBLE")
    print("=" * 70)
    print(f"\n   Loading: {args.ensemble}")

    atoms, frames = parse_ensemble_pdb(args.ensemble)

    n_frames = len(frames)
    n_atoms = len(atoms)

    print(f"   Frames: {n_frames}")
    print(f"   Atoms: {n_atoms}")

    # Get residue info
    residue_ids = []
    ca_indices = []
    backbone_indices = []

    seen_resids = set()
    for i, atom in enumerate(atoms):
        if atom.is_ca and atom.resid not in seen_resids:
            residue_ids.append(atom.resid)
            ca_indices.append(i)
            seen_resids.add(atom.resid)
        if atom.is_backbone:
            backbone_indices.append(i)

    n_residues = len(residue_ids)
    print(f"   Residues: {n_residues} (IDs {min(residue_ids)}-{max(residue_ids)})")

    # Detect residue offset
    residue_offset = 0
    if min(residue_ids) < 100 and args.residue_offset:
        residue_offset = args.residue_offset
        print(f"   Applying residue offset: +{residue_offset}")

    ca_indices = np.array(ca_indices)
    backbone_indices = np.array(backbone_indices)

    # ========================================================================
    # STEP 2: KABSCH ALIGNMENT
    # ========================================================================
    print(f"\n{'='*70}")
    print("STEP 2: RMSD ALIGNMENT (Kabsch Algorithm)")
    print("=" * 70)

    # Use first frame as reference
    reference = frames[0]

    print(f"\n   Aligning {n_frames} frames to reference...")
    print(f"   Alignment atoms: {len(backbone_indices)} backbone atoms")

    # Align using backbone atoms
    backbone_mask = np.zeros(n_atoms, dtype=bool)
    backbone_mask[backbone_indices] = True

    aligned_frames, rmsds = align_ensemble_to_reference(
        frames,
        reference,
        align_selection=backbone_mask
    )

    print(f"\n   RMSD Statistics (after alignment):")
    print(f"   Mean RMSD: {np.mean(rmsds):.3f} A")
    print(f"   Std RMSD:  {np.std(rmsds):.3f} A")
    print(f"   Max RMSD:  {np.max(rmsds):.3f} A (frame {np.argmax(rmsds)})")
    print(f"   Min RMSD:  {np.min(rmsds):.3f} A (frame {np.argmin(rmsds)})")

    # ========================================================================
    # STEP 3: COMPUTE RMSF (After Alignment)
    # ========================================================================
    print(f"\n{'='*70}")
    print("STEP 3: RMSF CALCULATION (Post-Alignment)")
    print("=" * 70)

    # Extract CA coordinates for RMSF
    aligned_ca = [frame[ca_indices] for frame in aligned_frames]

    rmsf, mean_structure = compute_rmsf_aligned(aligned_ca)

    print(f"\n   RMSF Statistics (CA atoms, properly aligned):")
    print(f"   Mean RMSF: {rmsf.mean():.3f} A")
    print(f"   Std RMSF:  {rmsf.std():.3f} A")
    print(f"   Max RMSF:  {rmsf.max():.3f} A")
    print(f"   Min RMSF:  {rmsf.min():.3f} A")

    # Find most flexible residues
    max_idx = np.argmax(rmsf)
    max_resid = residue_ids[max_idx] + residue_offset
    print(f"\n   Most flexible residue: {max_resid} (RMSF = {rmsf[max_idx]:.3f} A)")

    # Sanity check
    if rmsf.mean() > 5.0:
        print("\n   WARNING: Mean RMSF > 5 A is unusually high.")
        print("   This may indicate protein unfolding or alignment issues.")
    elif rmsf.mean() < 0.3:
        print("\n   WARNING: Mean RMSF < 0.3 A is unusually low.")
        print("   This may indicate very short simulation or strong restraints.")
    else:
        print("\n   RMSF values are in expected range (0.5-3.0 A for folded protein)")

    # ========================================================================
    # STEP 4: TOP FLEXIBLE RESIDUES
    # ========================================================================
    print(f"\n{'='*70}")
    print("STEP 4: TOP FLEXIBLE RESIDUES")
    print("=" * 70)

    print("\n   Top 15 Most Flexible Residues:")
    sorted_indices = np.argsort(rmsf)[::-1]

    for i, idx in enumerate(sorted_indices[:15]):
        pdb_resid = residue_ids[idx] + residue_offset
        z_score = (rmsf[idx] - rmsf.mean()) / rmsf.std()

        # Check if it's a known functional site
        annotation = ""
        if pdb_resid in ACE2_INTERFACE:
            annotation += " [ACE2 interface]"
        if pdb_resid in ESCAPE_MUTATIONS:
            mutation, variant = ESCAPE_MUTATIONS[pdb_resid]
            annotation += f" [{mutation}]"

        status = "HIGH" if z_score > 1.5 else "MOD" if z_score > 0.5 else "LOW"
        print(f"   {i+1:2d}. {pdb_resid:4d}: RMSF={rmsf[idx]:.3f}A (z={z_score:+.2f}) {status}{annotation}")

    # ========================================================================
    # STEP 5: ESCAPE MUTATION ANALYSIS
    # ========================================================================
    print(f"\n{'='*70}")
    print("STEP 5: ESCAPE MUTATION SITE FLEXIBILITY")
    print("=" * 70)

    escape_analysis = []

    for resid, (mutation, variant) in sorted(ESCAPE_MUTATIONS.items()):
        seq_resid = resid - residue_offset
        if seq_resid in residue_ids:
            idx = residue_ids.index(seq_resid)
            res_rmsf = rmsf[idx]
            z_score = (res_rmsf - rmsf.mean()) / rmsf.std()

            escape_analysis.append({
                'residue_id': resid,
                'mutation': mutation,
                'variant': variant,
                'rmsf': float(res_rmsf),
                'z_score': float(z_score)
            })

    if escape_analysis:
        print("\n   Escape mutation sites (sorted by flexibility):")
        for e in sorted(escape_analysis, key=lambda x: -x['z_score']):
            status = "HIGH" if e['z_score'] > 1.5 else "MOD " if e['z_score'] > 0.5 else "LOW "
            print(f"   {status} {e['residue_id']} ({e['mutation']}): "
                  f"{e['rmsf']:.3f}A, z={e['z_score']:+.2f} - {e['variant']}")
    else:
        print("\n   No escape mutation sites found in structure")

    # ========================================================================
    # STEP 6: TARGET REGION ANALYSIS
    # ========================================================================
    print(f"\n{'='*70}")
    print("STEP 6: TARGET REGION ANALYSIS")
    print("=" * 70)

    target_residues = [int(x) for x in args.target_residues.split(',')]
    print(f"\n   Target residues: {target_residues}")

    target_seq_ids = [r - residue_offset for r in target_residues]
    target_indices = [i for i, rid in enumerate(residue_ids) if rid in target_seq_ids]

    if target_indices:
        target_rmsf = rmsf[target_indices]

        print(f"\n   Found {len(target_indices)} target residues:")
        for idx in target_indices:
            pdb_resid = residue_ids[idx] + residue_offset
            z_score = (rmsf[idx] - rmsf.mean()) / rmsf.std()
            status = "HIGH" if z_score > 1.5 else "MOD " if z_score > 0.5 else "LOW "
            print(f"   {status} {pdb_resid}: RMSF = {rmsf[idx]:.3f} A (z = {z_score:+.2f})")

        print(f"\n   Target region statistics:")
        print(f"   Mean RMSF: {target_rmsf.mean():.3f} A")
        print(f"   Max RMSF:  {target_rmsf.max():.3f} A")
    else:
        target_rmsf = None
        print(f"   Target residues not found in structure")
        print(f"   (Available range: {min(residue_ids)+residue_offset}-{max(residue_ids)+residue_offset})")

    # ========================================================================
    # STEP 7: POCKET DETECTION
    # ========================================================================
    print(f"\n{'='*70}")
    print("STEP 7: POCKET DETECTION (fpocket)")
    print("=" * 70)

    # Compute "openness" of target region
    if target_indices:
        openness = []
        for frame_ca in aligned_ca:
            target_coords = frame_ca[target_indices]
            ref_coords = mean_structure[target_indices]
            rmsd = np.sqrt(np.mean(np.sum((target_coords - ref_coords)**2, axis=1)))
            openness.append(rmsd)
        openness = np.array(openness)

        print(f"\n   Target region openness:")
        print(f"   Mean: {openness.mean():.3f} A")
        print(f"   Std:  {openness.std():.3f} A")
        print(f"   Max:  {openness.max():.3f} A (frame {np.argmax(openness)})")
    else:
        # Use global RMSD as openness
        openness = np.array(rmsds)

    # Select frames for pocket analysis
    n_pocket_frames = min(args.n_pocket_frames, n_frames)

    # Get most open frames
    sorted_by_openness = np.argsort(openness)[::-1]
    open_frames = sorted_by_openness[:n_pocket_frames]

    # Also get reference (most closed)
    closed_frame = sorted_by_openness[-1]

    fpocket_available = shutil.which('fpocket') is not None

    if fpocket_available:
        print(f"\n   Running fpocket on {n_pocket_frames} open frames + 1 reference...")

        pocket_results = []

        # Analyze open frames
        for i, frame_idx in enumerate(open_frames):
            frame_pdb = frames_dir / f"frame_{frame_idx:04d}_open.pdb"
            write_pdb_frame(atoms, aligned_frames[frame_idx], str(frame_pdb))

            result = run_fpocket(str(frame_pdb), str(frames_dir))

            pocket_results.append({
                'frame_idx': int(frame_idx),
                'openness': float(openness[frame_idx]),
                'state': 'open',
                'fpocket': result
            })

            if result:
                print(f"   Frame {frame_idx}: {result['n_pockets']} pockets "
                      f"(openness={openness[frame_idx]:.3f}A)")

        # Analyze reference
        ref_pdb = frames_dir / f"frame_{closed_frame:04d}_ref.pdb"
        write_pdb_frame(atoms, aligned_frames[closed_frame], str(ref_pdb))

        ref_result = run_fpocket(str(ref_pdb), str(frames_dir))

        pocket_results.append({
            'frame_idx': int(closed_frame),
            'openness': float(openness[closed_frame]),
            'state': 'reference',
            'fpocket': ref_result
        })

        if ref_result:
            print(f"   Reference frame {closed_frame}: {ref_result['n_pockets']} pockets")
    else:
        print("\n   fpocket not available. Install with: conda install -c conda-forge fpocket")
        pocket_results = []
        ref_result = None

    # ========================================================================
    # STEP 8: IDENTIFY CRYPTIC POCKETS
    # ========================================================================
    print(f"\n{'='*70}")
    print("STEP 8: CRYPTIC POCKET IDENTIFICATION")
    print("=" * 70)

    # Compare open vs closed pocket counts
    if pocket_results:
        open_pocket_counts = [r['fpocket']['n_pockets'] for r in pocket_results
                             if r['state'] == 'open' and r.get('fpocket')]
        ref_pocket_count = ref_result['n_pockets'] if ref_result else 0

        if open_pocket_counts:
            avg_open_pockets = np.mean(open_pocket_counts)
            print(f"\n   Average pockets in open frames: {avg_open_pockets:.1f}")
            print(f"   Pockets in reference frame: {ref_pocket_count}")

            if avg_open_pockets > ref_pocket_count + 1:
                print(f"\n   CRYPTIC POCKETS DETECTED!")
                print(f"   Open frames have ~{avg_open_pockets - ref_pocket_count:.0f} more pockets")
            elif avg_open_pockets > ref_pocket_count:
                print(f"\n   Potential cryptic pockets detected")
            else:
                print(f"\n   No significant pocket differences detected")
    else:
        print("\n   Pocket analysis not available")

    # ========================================================================
    # SAVE RESULTS
    # ========================================================================
    print(f"\n{'='*70}")
    print("SAVING RESULTS")
    print("=" * 70)

    results = {
        'input': {
            'ensemble': args.ensemble,
            'n_frames': n_frames,
            'n_atoms': n_atoms,
            'n_residues': n_residues,
            'residue_range': [int(min(residue_ids)), int(max(residue_ids))],
            'residue_offset': residue_offset
        },
        'alignment': {
            'method': 'Kabsch (backbone atoms)',
            'n_alignment_atoms': int(len(backbone_indices)),
            'rmsd_mean': float(np.mean(rmsds)),
            'rmsd_std': float(np.std(rmsds)),
            'rmsd_max': float(np.max(rmsds))
        },
        'rmsf': {
            'mean': float(rmsf.mean()),
            'std': float(rmsf.std()),
            'max': float(rmsf.max()),
            'max_residue': int(max_resid),
            'per_residue': [
                {'residue_id': int(residue_ids[i] + residue_offset), 'rmsf': float(rmsf[i])}
                for i in range(n_residues)
            ]
        },
        'target_region': {
            'residues': target_residues,
            'mean_rmsf': float(target_rmsf.mean()) if target_rmsf is not None else None,
            'max_rmsf': float(target_rmsf.max()) if target_rmsf is not None else None
        },
        'escape_mutations': escape_analysis,
        'pocket_analysis': pocket_results
    }

    # Save JSON
    output_json = output_dir / "analysis_results.json"
    with open(output_json, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n   Results: {output_json}")

    # Save RMSF CSV
    rmsf_csv = output_dir / "rmsf_aligned.csv"
    with open(rmsf_csv, 'w') as f:
        f.write("residue_id,rmsf,z_score\n")
        for i in range(n_residues):
            z = (rmsf[i] - rmsf.mean()) / rmsf.std()
            f.write(f"{residue_ids[i] + residue_offset},{rmsf[i]:.4f},{z:.4f}\n")
    print(f"   RMSF data: {rmsf_csv}")

    # Save RMSD time series
    rmsd_csv = output_dir / "rmsd_timeseries.csv"
    with open(rmsd_csv, 'w') as f:
        f.write("frame,rmsd,openness\n")
        for i in range(n_frames):
            f.write(f"{i},{rmsds[i]:.4f},{openness[i]:.4f}\n")
    print(f"   RMSD data: {rmsd_csv}")

    # ========================================================================
    # GENERATE PLOT
    # ========================================================================
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, 1, figsize=(12, 8))

        # RMSF plot
        ax1 = axes[0]
        pdb_residue_ids = [r + residue_offset for r in residue_ids]
        ax1.bar(pdb_residue_ids, rmsf, width=1, color='steelblue', alpha=0.7)
        threshold = rmsf.mean() + 1.5 * rmsf.std()
        ax1.axhline(threshold, color='red', linestyle='--', label=f'1.5σ threshold ({threshold:.2f} A)')
        ax1.axhline(rmsf.mean(), color='gray', linestyle=':', label=f'Mean ({rmsf.mean():.2f} A)')
        ax1.set_xlabel('Residue ID')
        ax1.set_ylabel('RMSF (A)')
        ax1.set_title('RMSF After Kabsch Alignment')
        ax1.legend()

        # Mark escape mutations
        for resid in ESCAPE_MUTATIONS:
            if resid in pdb_residue_ids:
                ax1.axvline(resid, color='orange', alpha=0.3, linewidth=2)

        # RMSD time series
        ax2 = axes[1]
        frames_x = np.arange(n_frames)
        ax2.plot(frames_x, rmsds, 'b-', alpha=0.7, label='RMSD from reference')
        ax2.plot(frames_x, openness, 'r-', alpha=0.7, label='Target region openness')
        ax2.set_xlabel('Frame')
        ax2.set_ylabel('RMSD (A)')
        ax2.set_title('Structural Deviation Over Trajectory')
        ax2.legend()

        plt.tight_layout()
        plot_path = output_dir / "analysis_plots.png"
        plt.savefig(plot_path, dpi=150)
        plt.close()
        print(f"   Plots: {plot_path}")
    except ImportError:
        print("   matplotlib not available, skipping plots")
    except Exception as e:
        print(f"   Plot generation failed: {e}")

    # ========================================================================
    # SUMMARY
    # ========================================================================
    print(f"\n{'='*70}")
    print("SUMMARY")
    print("=" * 70)

    print(f"\n   Analysis complete!")
    print(f"\n   Key findings:")
    print(f"   Mean RMSF (aligned): {rmsf.mean():.3f} A")
    print(f"   Most flexible: residue {max_resid} ({rmsf.max():.3f} A)")

    high_flex_escapes = [e for e in escape_analysis if e['z_score'] > 1.0]
    if high_flex_escapes:
        print(f"\n   High-flexibility escape mutations:")
        for e in sorted(high_flex_escapes, key=lambda x: -x['z_score'])[:5]:
            print(f"   - {e['residue_id']} ({e['mutation']}): z={e['z_score']:+.2f}")

    print(f"\n{'='*70}")


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='PRISM-4D Cryptic Pocket Analysis with Proper Alignment',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument('--ensemble', required=True,
                        help='Input ensemble PDB (multi-MODEL)')
    parser.add_argument('--output', required=True,
                        help='Output directory')
    parser.add_argument('--target-residues', default="474,475,476,477,478,479,480",
                        help='Target residues for pocket analysis (PDB numbering)')
    parser.add_argument('--n-pocket-frames', type=int, default=10,
                        help='Number of frames to analyze with fpocket')
    parser.add_argument('--residue-offset', type=int, default=332,
                        help='Offset to convert sequential to PDB numbering (default: 332 for RBD)')

    args = parser.parse_args()
    main(args)
