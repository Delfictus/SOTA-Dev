#!/usr/bin/env python3
"""
PRISM4D Cryo-UV Correlation Analysis

Correlates RMSF hotspots with water spike hotspots to identify
druggable cryptic binding sites.

Usage:
    python3 correlate_rmsf_spikes.py <results_dir> [--radius 10.0] [--output results.json]
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(
        description="Correlate RMSF and water spike hotspots"
    )
    parser.add_argument("results_dir", help="Directory containing NHS results")
    parser.add_argument(
        "--ensemble", help="Path to ensemble PDB (optional, auto-detected)"
    )
    parser.add_argument(
        "--radius", type=float, default=10.0, help="Correlation radius in Angstroms"
    )
    parser.add_argument(
        "--rmsf-threshold",
        type=float,
        default=1.0,
        help="RMSF threshold as multiple of std above mean",
    )
    parser.add_argument("--output", help="Output JSON file (default: correlated_sites.json)")
    parser.add_argument("--max-frames", type=int, default=75, help="Max frames to analyze")
    return parser.parse_args()


def find_ensemble_pdb(results_dir: Path) -> Path:
    """Find ensemble PDB in results directory or parent"""
    patterns = ["*_ensemble.pdb", "*_stable.pdb", "ensemble.pdb"]

    for pattern in patterns:
        matches = list(results_dir.glob(pattern))
        if matches:
            return matches[0]

    # Check parent directory
    parent = results_dir.parent
    for pattern in patterns:
        matches = list(parent.glob(pattern))
        if matches:
            return matches[0]

    return None


def load_spike_data(results_dir: Path) -> dict:
    """Load spike hotspot data from NHS results"""
    results_file = results_dir / "adaptive_results.json"

    if not results_file.exists():
        # Try alternative names
        for name in ["nhs_results.json", "results.json"]:
            alt = results_dir / name
            if alt.exists():
                results_file = alt
                break

    if not results_file.exists():
        raise FileNotFoundError(f"No results file found in {results_dir}")

    with open(results_file) as f:
        return json.load(f)


def get_residue_positions(pdb_path: Path, max_frames: int = 75) -> list:
    """Extract CA positions from first frame of ensemble PDB"""
    positions = []
    frame_count = 0

    with open(pdb_path) as f:
        for line in f:
            if line.startswith("MODEL"):
                if frame_count > 0:
                    break  # Only first frame for positions
            elif line.startswith("ENDMDL"):
                frame_count += 1
            elif line.startswith("ATOM") and " CA " in line:
                try:
                    x = float(line[30:38])
                    y = float(line[38:46])
                    z = float(line[46:54])
                    res_id = int(line[22:26])
                    res_name = line[17:20].strip()
                    chain = line[21]
                    positions.append({
                        "res_id": res_id,
                        "res_name": res_name,
                        "chain": chain,
                        "pos": np.array([x, y, z]),
                    })
                except ValueError:
                    continue

    return positions


def calc_rmsf(pdb_path: Path, max_frames: int = 75) -> tuple:
    """Calculate RMSF from ensemble PDB without centering"""
    models = []
    current_model = []

    with open(pdb_path) as f:
        for line in f:
            if line.startswith("MODEL"):
                current_model = []
            elif line.startswith("ENDMDL"):
                if current_model:
                    models.append(np.array(current_model))
                    if len(models) >= max_frames:
                        break
            elif line.startswith("ATOM") and " CA " in line:
                try:
                    x = float(line[30:38])
                    y = float(line[38:46])
                    z = float(line[46:54])
                    current_model.append([x, y, z])
                except ValueError:
                    continue

    if not models:
        raise ValueError("No valid frames found in ensemble PDB")

    coords = np.array(models)  # [n_frames, n_atoms, 3]
    mean_coords = coords.mean(axis=0)
    rmsf = np.sqrt(((coords - mean_coords) ** 2).sum(axis=2).mean(axis=0))

    return rmsf, mean_coords


def correlate_sites(
    residue_info: list,
    rmsf: np.ndarray,
    mean_positions: np.ndarray,
    spike_data: dict,
    radius: float,
    rmsf_threshold_mult: float,
) -> list:
    """Find residues with both high RMSF and nearby spike hotspots"""

    # Get high-RMSF residues
    rmsf_threshold = np.mean(rmsf) + rmsf_threshold_mult * np.std(rmsf)
    high_rmsf_idx = [i for i in range(len(rmsf)) if rmsf[i] > rmsf_threshold]

    print(f"RMSF threshold: {rmsf_threshold:.2f} Å")
    print(f"High-RMSF residues: {len(high_rmsf_idx)}")

    # Get spike hotspot positions
    spike_positions = []
    for hotspot in spike_data.get("mapped_hotspots", []):
        pos = hotspot.get("position_angstrom", hotspot.get("position", [0, 0, 0]))
        score = hotspot.get("spike_count", hotspot.get("count", 0))
        spike_positions.append((np.array(pos), score))

    print(f"Spike hotspots: {len(spike_positions)}")

    # Find correlations
    correlated = []

    for idx in high_rmsf_idx:
        info = residue_info[idx]
        res_pos = mean_positions[idx]

        # Find nearest spike hotspot
        min_dist = float("inf")
        best_spike_score = 0

        for spike_pos, spike_score in spike_positions:
            dist = np.linalg.norm(res_pos - spike_pos)
            if dist < min_dist:
                min_dist = dist
                best_spike_score = spike_score

        if min_dist < radius:
            correlated.append({
                "residue": f"{info['chain']}_{info['res_name']}{info['res_id']}",
                "chain": info["chain"],
                "res_name": info["res_name"],
                "res_id": info["res_id"],
                "rmsf": float(rmsf[idx]),
                "spike_score": int(best_spike_score),
                "distance": float(min_dist),
                "combined_score": float(rmsf[idx] * best_spike_score),
                "position": res_pos.tolist(),
            })

    # Sort by combined score
    correlated.sort(key=lambda x: x["combined_score"], reverse=True)

    return correlated


def print_results(correlated: list, top_n: int = 25):
    """Print formatted results table"""
    print()
    print("=" * 70)
    print("   DRUGGABLE CRYPTIC SITES (High RMSF + Water Spikes)")
    print("=" * 70)
    print()
    print(f"{'Rank':<5} {'Residue':<12} {'RMSF':<8} {'Spikes':<8} {'Dist':<8} {'Combined'}")
    print("-" * 60)

    for i, site in enumerate(correlated[:top_n], 1):
        stars = "★★★" if site["combined_score"] > 2000 else ("★★" if site["combined_score"] > 1000 else "★")
        print(
            f"{i:<5} {site['residue']:<12} {site['rmsf']:>6.2f}Å "
            f"{site['spike_score']:>6}  {site['distance']:>6.2f}Å "
            f"{site['combined_score']:>8.0f} {stars}"
        )

    print()
    print("=" * 70)
    print(f"   SUMMARY: {len(correlated)} CORRELATED CRYPTIC SITES FOUND")
    print("=" * 70)


def main():
    args = parse_args()

    results_dir = Path(args.results_dir)
    if not results_dir.exists():
        print(f"Error: Results directory not found: {results_dir}")
        sys.exit(1)

    print("=" * 70)
    print("   PRISM4D Cryo-UV Correlation Analysis")
    print("=" * 70)
    print()

    # Load spike data
    print("Loading spike data...")
    spike_data = load_spike_data(results_dir)

    # Find ensemble PDB
    if args.ensemble:
        ensemble_path = Path(args.ensemble)
    else:
        ensemble_path = find_ensemble_pdb(results_dir)

    if ensemble_path is None or not ensemble_path.exists():
        print("Error: Could not find ensemble PDB")
        print("Specify with --ensemble <path>")
        sys.exit(1)

    print(f"Using ensemble: {ensemble_path}")

    # Get residue positions and RMSF
    print("Calculating RMSF...")
    residue_info = get_residue_positions(ensemble_path, args.max_frames)
    rmsf, mean_positions = calc_rmsf(ensemble_path, args.max_frames)

    print(f"Residues analyzed: {len(residue_info)}")
    print(f"RMSF range: {rmsf.min():.2f} - {rmsf.max():.2f} Å")
    print()

    # Correlate
    print(f"Correlating with radius: {args.radius} Å")
    correlated = correlate_sites(
        residue_info,
        rmsf,
        mean_positions,
        spike_data,
        args.radius,
        args.rmsf_threshold,
    )

    # Print results
    print_results(correlated)

    # Save results
    output_path = args.output or str(results_dir / "correlated_sites.json")
    with open(output_path, "w") as f:
        json.dump({
            "parameters": {
                "correlation_radius": args.radius,
                "rmsf_threshold_mult": args.rmsf_threshold,
                "ensemble_path": str(ensemble_path),
                "results_dir": str(results_dir),
            },
            "summary": {
                "total_correlated": len(correlated),
                "high_rmsf_count": len([r for r in correlated if r["rmsf"] > 15]),
                "top_combined_score": correlated[0]["combined_score"] if correlated else 0,
            },
            "sites": correlated,
        }, f, indent=2)

    print(f"\nResults saved to: {output_path}")

    if correlated:
        print("\nTOP 10 DRUGGABLE TARGETS:")
        for i, site in enumerate(correlated[:10], 1):
            print(f"  {i}. {site['residue']}: RMSF={site['rmsf']:.1f}Å × {site['spike_score']} spikes = {site['combined_score']:.0f}")


if __name__ == "__main__":
    main()
