#!/usr/bin/env python3
"""
PRISM4D: End-to-End Cryptic Pocket Discovery Pipeline

Complete pipeline from PDB ID to cryptic site analysis:
  Stage 1: Fetch & Sanitize PDB (PDBFixer)
  Stage 2: Generate AMBER topology (OpenMM)
  Stage 3: Run GPU MD simulation (PRISM generate-ensemble)
  Stage 4: Analyze conformational dynamics (RMSF, pockets)

Usage:
    python prism_pipeline.py 6M0J results/                    # Full pipeline
    python prism_pipeline.py 6M0J results/ --chain E          # Single chain
    python prism_pipeline.py local.pdb results/               # Local file
    python prism_pipeline.py 6M0J results/ --steps 100000     # Longer MD

Dependencies:
    conda activate prism-delta
    # Requires: openmm, pdbfixer, numpy
    # Requires: PRISM generate-ensemble binary (Rust)
"""

import sys
import os
import argparse
import subprocess
import json
import time
from pathlib import Path
from datetime import datetime

# Add script directory to path for imports
SCRIPT_DIR = Path(__file__).parent.resolve()
sys.path.insert(0, str(SCRIPT_DIR))

from stage1_sanitize import sanitize_pdb, is_pdb_id
from stage2_topology import prepare_topology


def find_generate_ensemble():
    """Find the generate-ensemble binary."""
    # Check common locations
    candidates = [
        SCRIPT_DIR.parent / "target" / "release" / "generate-ensemble",
        Path.home() / "Desktop" / "PRISM-Delta-v1.0" / "target" / "release" / "generate-ensemble",
        Path.home() / "Desktop" / "PRISM4D-v1.1.0-STABLE" / "target" / "release" / "generate-ensemble",
    ]

    for path in candidates:
        if path.exists():
            return str(path)

    # Try finding in PATH
    result = subprocess.run(["which", "generate-ensemble"], capture_output=True, text=True)
    if result.returncode == 0:
        return result.stdout.strip()

    return None


def find_analyze_ensemble():
    """Find the analyze_ensemble binary (native Rust analysis)."""
    # Check common locations
    candidates = [
        SCRIPT_DIR.parent / "target" / "release" / "analyze_ensemble",
        Path.home() / "Desktop" / "PRISM-Delta-v1.0" / "target" / "release" / "analyze_ensemble",
        Path.home() / "Desktop" / "PRISM4D-v1.1.0-STABLE" / "target" / "release" / "analyze_ensemble",
    ]

    for path in candidates:
        if path.exists():
            return str(path)

    # Try finding in PATH
    result = subprocess.run(["which", "analyze_ensemble"], capture_output=True, text=True)
    if result.returncode == 0:
        return result.stdout.strip()

    return None


def run_stage4_native(
    ensemble_path: str,
    output_dir: str,
    ca_only: bool = False,
    verbose: bool = True
) -> dict:
    """
    Run native Rust analysis (analyze_ensemble binary).

    Returns dict with analysis results.
    """
    binary = find_analyze_ensemble()
    if binary is None:
        raise RuntimeError(
            "analyze_ensemble binary not found. "
            "Build with: cargo build --release -p prism-validation --bin analyze_ensemble"
        )

    output_json = Path(output_dir) / "analysis_results.json"

    if verbose:
        print("Running native Rust analysis...")
        print(f"  Binary: {binary}")
        print(f"  Mode: {'CA-only' if ca_only else 'All-atom'}")

    cmd = [
        binary,
        "--ensemble", ensemble_path,
        "--output", str(output_json),
    ]

    if ca_only:
        cmd.append("--ca-only")

    if not verbose:
        cmd.append("--quiet")

    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        print(f"Native analysis failed: {result.stderr}")
        raise RuntimeError(f"analyze_ensemble failed: {result.stderr}")

    if verbose and result.stdout:
        # Print the output (includes nice summary table)
        print(result.stdout)

    # Load and return results
    with open(output_json) as f:
        results = json.load(f)

    return results


def run_stage3_md(
    topology_path: str,
    pdb_path: str,
    output_path: str,
    steps: int = 50000,
    dt: float = 2.0,
    temperature: float = 310.0,
    save_interval: int = 500,
    equilibration: int = 10000,
    restraint_k: float = 2.0,
    verbose: bool = True
) -> dict:
    """
    Run GPU molecular dynamics using PRISM generate-ensemble.

    Args:
        restraint_k: Position restraint force constant in kcal/(mol·Å²).
                     Default 2.0 stabilizes protein fold in implicit solvent.
                     Set to 0.0 to disable restraints.

    Returns dict with MD statistics.
    """
    binary = find_generate_ensemble()
    if binary is None:
        raise RuntimeError(
            "generate-ensemble binary not found. "
            "Build with: cargo build --release --features cuda -p prism-validation --bin generate-ensemble"
        )

    if verbose:
        print(f"Running GPU MD simulation...")
        print(f"  Binary: {binary}")
        print(f"  Steps: {steps} ({steps * dt / 1000:.1f} ps)")
        print(f"  Temperature: {temperature} K")
        print(f"  Restraints: k = {restraint_k} kcal/(mol·Å²)")

    cmd = [
        binary,
        "--topology", topology_path,
        "--pdb", pdb_path,
        "--steps", str(steps),
        "--dt", str(dt),
        "--temperature", str(temperature),
        "--save-interval", str(save_interval),
        "--equilibration", str(equilibration),
        "--restraint-k", str(restraint_k),
        "--output", output_path,
    ]

    start_time = time.time()

    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True
    )

    elapsed = time.time() - start_time

    if result.returncode != 0:
        print(f"MD simulation failed:")
        print(result.stderr)
        raise RuntimeError(f"generate-ensemble failed with code {result.returncode}")

    if verbose:
        print(f"  Completed in {elapsed:.1f}s")
        # Parse output for statistics
        for line in result.stdout.split('\n'):
            if any(x in line.lower() for x in ['frame', 'energy', 'temperature', 'rmsd']):
                print(f"  {line}")

    return {
        "elapsed_seconds": elapsed,
        "steps": steps,
        "output_file": output_path,
    }


def kabsch_align(mobile: 'np.ndarray', target: 'np.ndarray') -> 'np.ndarray':
    """
    Align mobile coordinates onto target using Kabsch algorithm.
    Removes rigid-body motion (translation + rotation) for true RMSD.

    Args:
        mobile: (N, 3) coordinates to align
        target: (N, 3) reference coordinates

    Returns:
        aligned: (N, 3) aligned mobile coordinates
    """
    import numpy as np

    # Center both structures
    mobile_center = mobile.mean(axis=0)
    target_center = target.mean(axis=0)

    mobile_centered = mobile - mobile_center
    target_centered = target - target_center

    # Compute covariance matrix
    H = mobile_centered.T @ target_centered

    # SVD for optimal rotation
    U, S, Vt = np.linalg.svd(H)

    # Handle reflection (ensure proper rotation)
    d = np.linalg.det(Vt.T @ U.T)
    correction = np.array([[1, 0, 0], [0, 1, 0], [0, 0, d]])
    rotation = Vt.T @ correction @ U.T

    # Apply transformation
    aligned = (mobile_centered @ rotation.T) + target_center

    return aligned


def run_stage4_analysis(
    ensemble_path: str,
    output_dir: str,
    reference_pdb: str = None,
    verbose: bool = True
) -> dict:
    """
    Analyze conformational ensemble for cryptic pockets.
    Uses Kabsch alignment to remove rigid-body motion before RMSD/RMSF calculation.

    Returns dict with analysis results.
    """
    import numpy as np

    if verbose:
        print("Analyzing conformational ensemble...")
        print("  Using Kabsch alignment for accurate RMSD/RMSF")

    # Parse ensemble PDB
    frames = parse_ensemble_pdb(ensemble_path)
    n_frames = len(frames)
    n_atoms = len(frames[0]) if frames else 0

    if verbose:
        print(f"  Frames: {n_frames}")
        print(f"  Atoms per frame: {n_atoms}")

    if n_frames < 2:
        raise ValueError("Need at least 2 frames for analysis")

    # Get reference structure (first frame)
    ref_coords = np.array(frames[0])

    # Align all frames to reference using Kabsch algorithm
    aligned_frames = [ref_coords]  # First frame is reference
    rmsds = [0.0]  # RMSD of reference to itself

    if verbose:
        print("  Aligning frames to reference...")

    for i in range(1, n_frames):
        mobile = np.array(frames[i])
        aligned = kabsch_align(mobile, ref_coords)
        aligned_frames.append(aligned)

        # Compute RMSD after alignment
        diff = aligned - ref_coords
        rmsd = np.sqrt(np.mean(np.sum(diff ** 2, axis=1)))
        rmsds.append(rmsd)

    all_coords = np.array(aligned_frames)
    mean_rmsd = np.mean(rmsds[1:])  # Exclude reference (RMSD=0)
    std_rmsd = np.std(rmsds[1:])
    max_rmsd = np.max(rmsds)

    if verbose:
        print(f"  Mean RMSD (aligned): {mean_rmsd:.3f} +/- {std_rmsd:.3f} A")
        print(f"  Max RMSD: {max_rmsd:.3f} A")

    # Compute per-atom RMSF from aligned coordinates
    mean_coords = np.mean(all_coords, axis=0)
    deviations = all_coords - mean_coords
    rmsf = np.sqrt(np.mean(np.sum(deviations ** 2, axis=2), axis=0))

    mean_rmsf = np.mean(rmsf)
    std_rmsf = np.std(rmsf)
    max_rmsf = np.max(rmsf)

    if verbose:
        print(f"  Mean RMSF (aligned): {mean_rmsf:.3f} +/- {std_rmsf:.3f} A")
        print(f"  Max RMSF: {max_rmsf:.3f} A")

    # Find high-flexibility atoms (z-score > 1.5)
    z_scores = (rmsf - mean_rmsf) / (std_rmsf + 1e-6)
    high_flex_indices = np.where(z_scores > 1.5)[0]

    if verbose:
        print(f"  High-flexibility atoms (z>1.5): {len(high_flex_indices)}")

    # Save results
    results = {
        "n_frames": n_frames,
        "n_atoms": n_atoms,
        "alignment": "kabsch",
        "mean_rmsd": float(mean_rmsd),
        "std_rmsd": float(std_rmsd),
        "max_rmsd": float(max_rmsd),
        "mean_rmsf": float(mean_rmsf),
        "std_rmsf": float(std_rmsf),
        "max_rmsf": float(max_rmsf),
        "high_flex_count": int(len(high_flex_indices)),
        "rmsf": [float(x) for x in rmsf],
        "z_scores": [float(x) for x in z_scores],
        "rmsd_timeseries": [float(x) for x in rmsds],
    }

    # Write results
    results_path = Path(output_dir) / "analysis_results.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)

    if verbose:
        print(f"  Results saved to {results_path}")

    return results


def parse_ensemble_pdb(pdb_path: str) -> list:
    """Parse multi-MODEL PDB ensemble into list of coordinate arrays."""
    frames = []
    current_frame = []

    with open(pdb_path, 'r') as f:
        for line in f:
            if line.startswith('MODEL'):
                current_frame = []
            elif line.startswith('ENDMDL'):
                if current_frame:
                    frames.append(current_frame)
            elif line.startswith('ATOM') or line.startswith('HETATM'):
                if len(line) >= 54:
                    try:
                        x = float(line[30:38])
                        y = float(line[38:46])
                        z = float(line[46:54])
                        current_frame.append([x, y, z])
                    except ValueError:
                        pass

    # Handle single-MODEL files
    if not frames and current_frame:
        frames.append(current_frame)

    return frames


def run_pipeline(
    input_pdb: str,
    output_dir: str,
    chain: str = None,
    steps: int = 50000,
    temperature: float = 310.0,
    restraint_k: float = 2.0,
    keep_waters: bool = False,
    skip_md: bool = False,
    verbose: bool = True
) -> dict:
    """
    Run complete PRISM4D pipeline.

    Args:
        restraint_k: Position restraint force constant in kcal/(mol·Å²).
                     Default 2.0 stabilizes protein fold in implicit solvent.
                     Set to 0.0 for unrestrained dynamics.

    Returns dict with all results and file paths.
    """
    start_time = time.time()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Determine base name
    if is_pdb_id(input_pdb):
        base_name = input_pdb.upper()
    else:
        base_name = Path(input_pdb).stem

    if chain:
        base_name = f"{base_name}_{chain}"

    results = {
        "input": input_pdb,
        "output_dir": str(output_dir),
        "base_name": base_name,
        "chain": chain,
        "timestamp": datetime.now().isoformat(),
        "stages": {}
    }

    # ========== Stage 1: Sanitize ==========
    if verbose:
        print("\n" + "=" * 60)
        print("STAGE 1: PDB Sanitization")
        print("=" * 60)

    sanitized_pdb = output_dir / f"{base_name}_sanitized.pdb"

    stage1_start = time.time()
    stage1_stats = sanitize_pdb(
        input_pdb,
        str(sanitized_pdb),
        chain=chain,
        keep_waters=keep_waters,
        verbose=verbose
    )
    stage1_time = time.time() - stage1_start

    results["stages"]["sanitize"] = {
        "output": str(sanitized_pdb),
        "time_seconds": stage1_time,
        "stats": stage1_stats
    }

    # ========== Stage 2: Topology ==========
    if verbose:
        print("\n" + "=" * 60)
        print("STAGE 2: Topology Preparation")
        print("=" * 60)

    topology_json = output_dir / f"{base_name}_topology.json"

    stage2_start = time.time()
    stage2_result = prepare_topology(
        str(sanitized_pdb),
        str(topology_json),
        solvate=False,  # Implicit solvent for now
        minimize=True,
        verbose=verbose
    )
    stage2_time = time.time() - stage2_start

    results["stages"]["topology"] = {
        "output": str(topology_json),
        "time_seconds": stage2_time,
        "n_atoms": stage2_result["n_atoms"],
        "n_residues": stage2_result["n_residues"],
    }

    # ========== Stage 3: GPU MD ==========
    if not skip_md:
        if verbose:
            print("\n" + "=" * 60)
            print("STAGE 3: GPU Molecular Dynamics")
            print("=" * 60)

        ensemble_pdb = output_dir / f"{base_name}_ensemble.pdb"

        stage3_start = time.time()
        stage3_result = run_stage3_md(
            str(topology_json),
            str(sanitized_pdb),
            str(ensemble_pdb),
            steps=steps,
            temperature=temperature,
            restraint_k=restraint_k,
            verbose=verbose
        )
        stage3_time = time.time() - stage3_start

        results["stages"]["md"] = {
            "output": str(ensemble_pdb),
            "time_seconds": stage3_time,
            "steps": steps,
            "restraint_k": restraint_k,
        }

        # ========== Stage 4: Analysis ==========
        if verbose:
            print("\n" + "=" * 60)
            print("STAGE 4: Conformational Analysis")
            print("=" * 60)

        stage4_start = time.time()

        # Prefer native Rust analysis (faster, no NumPy dependency)
        native_binary = find_analyze_ensemble()
        if native_binary:
            if verbose:
                print("Using native Rust analysis (analyze_ensemble)")
            stage4_result = run_stage4_native(
                str(ensemble_pdb),
                str(output_dir),
                verbose=verbose
            )
            analysis_backend = "native-rust"
        else:
            if verbose:
                print("Native binary not found, using Python analysis")
            stage4_result = run_stage4_analysis(
                str(ensemble_pdb),
                str(output_dir),
                verbose=verbose
            )
            analysis_backend = "python"

        stage4_time = time.time() - stage4_start

        results["stages"]["analysis"] = {
            "output": str(output_dir / "analysis_results.json"),
            "time_seconds": stage4_time,
            "mean_rmsd": stage4_result["mean_rmsd"],
            "mean_rmsf": stage4_result["mean_rmsf"],
            "high_flex_count": stage4_result["high_flex_count"],
            "backend": analysis_backend,
        }
    else:
        if verbose:
            print("\n[Skipping MD and analysis stages]")

    # ========== Summary ==========
    total_time = time.time() - start_time
    results["total_time_seconds"] = total_time

    # Save pipeline results
    results_path = output_dir / f"{base_name}_pipeline_results.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)

    if verbose:
        print("\n" + "=" * 60)
        print("PIPELINE COMPLETE")
        print("=" * 60)
        print(f"Total time: {total_time:.1f}s")
        print(f"\nOutputs in {output_dir}/:")
        print(f"  - {base_name}_sanitized.pdb     (cleaned structure)")
        print(f"  - {base_name}_topology.json     (AMBER parameters)")
        if not skip_md:
            print(f"  - {base_name}_ensemble.pdb      (MD trajectory)")
            print(f"  - analysis_results.json         (RMSF analysis)")
        print(f"  - {base_name}_pipeline_results.json")

    return results


def main():
    parser = argparse.ArgumentParser(
        description='PRISM4D: End-to-End Cryptic Pocket Discovery Pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s 6M0J results/                     # SARS-CoV-2 RBD
  %(prog)s 6M0J results/ --chain E           # RBD only (no ACE2)
  %(prog)s 2VWD results/ --chain A           # Nipah G protein
  %(prog)s local.pdb results/                # Local file
  %(prog)s 6M0J results/ --steps 500000      # 1 ns simulation
  %(prog)s 6M0J results/ --restraint-k 0     # Unrestrained dynamics
  %(prog)s 6M0J results/ --skip-md           # Prep only (no GPU)

Pipeline Stages:
  1. Sanitize: Remove waters/ligands, fix atoms (PDBFixer - Python)
  2. Topology: Add hydrogens, apply AMBER ff14SB (OpenMM - Python)
  3. MD: GPU molecular dynamics with position restraints (Native Rust/CUDA)
  4. Analysis: Kabsch-aligned RMSD/RMSF (Native Rust, Python fallback)
        """
    )

    parser.add_argument('input', help='PDB ID (e.g., 6M0J) or path to local PDB file')
    parser.add_argument('output_dir', help='Output directory for results')
    parser.add_argument('--chain', '-c', help='Select specific chain')
    parser.add_argument('--steps', '-n', type=int, default=50000,
                        help='MD steps (default: 50000 = 100 ps)')
    parser.add_argument('--temperature', '-T', type=float, default=310.0,
                        help='Temperature in Kelvin (default: 310)')
    parser.add_argument('--restraint-k', '-k', type=float, default=2.0,
                        help='Position restraint k in kcal/(mol·A²) (default: 2.0, 0=disabled)')
    parser.add_argument('--keep-waters', '-w', action='store_true',
                        help='Keep crystallographic waters')
    parser.add_argument('--skip-md', action='store_true',
                        help='Skip MD simulation (stages 1-2 only)')
    parser.add_argument('--quiet', '-q', action='store_true',
                        help='Suppress output')

    args = parser.parse_args()

    try:
        results = run_pipeline(
            args.input,
            args.output_dir,
            chain=args.chain,
            steps=args.steps,
            temperature=args.temperature,
            restraint_k=args.restraint_k,
            keep_waters=args.keep_waters,
            skip_md=args.skip_md,
            verbose=not args.quiet
        )
        return 0

    except Exception as e:
        print(f"\nERROR: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())
