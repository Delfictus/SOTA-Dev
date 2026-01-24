#!/usr/bin/env python3
"""
Set up ATLAS benchmark with hybrid RMSF data:
- Real MD RMSF for proteins where available (from ATLAS API)
- B-factor derived RMSF for remaining proteins

This creates a publication-ready benchmark dataset.
"""

import json
import csv
from pathlib import Path
from typing import List, Dict, Tuple

def parse_atlas_rmsf_tsv(tsv_path: Path) -> Tuple[List[str], List[float]]:
    """Parse ATLAS RMSF TSV file and return (sequence, rmsf_avg)."""
    sequence = []
    rmsf_avg = []

    with open(tsv_path, 'r') as f:
        header = f.readline()  # Skip header
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) >= 5:
                # Format: residue_idx, aa, RMSF_R1, RMSF_R2, RMSF_R3, RMSF_avg
                sequence.append(parts[1])
                rmsf_avg.append(float(parts[5]))

    return sequence, rmsf_avg


def main():
    data_dir = Path("/home/diddy/Desktop/PRISM4D-bio/data/atlas_alphaflow")
    benchmark_dir = Path("/home/diddy/Desktop/PRISM4D-bio/data/atlas_benchmark")

    # Load existing targets (with B-factor RMSF)
    targets_path = data_dir / "atlas_targets.json"
    with open(targets_path, 'r') as f:
        targets = json.load(f)

    print("╔════════════════════════════════════════════════════════════════╗")
    print("║  ATLAS Benchmark Setup                                         ║")
    print("╚════════════════════════════════════════════════════════════════╝")
    print()
    print(f"Loaded {len(targets)} proteins from atlas_targets.json")

    # Check for real MD RMSF files
    md_rmsf_dir = benchmark_dir / "rmsf_md"
    md_rmsf_files = list(md_rmsf_dir.glob("*_RMSF.tsv")) if md_rmsf_dir.exists() else []

    print(f"Found {len(md_rmsf_files)} real MD RMSF files")

    # Build name -> target mapping
    target_map = {t['name']: t for t in targets}

    # Update targets with real MD RMSF where available
    md_updated = 0
    for rmsf_file in md_rmsf_files:
        # Parse filename: 6rrv_A_RMSF.tsv -> 6rrv_A
        name = rmsf_file.stem.replace("_RMSF", "")

        if name in target_map:
            sequence, rmsf_avg = parse_atlas_rmsf_tsv(rmsf_file)

            target = target_map[name]
            target['md_rmsf'] = rmsf_avg
            target['rmsf_source'] = 'ATLAS MD (3 replicates averaged)'
            target['md_sequence'] = sequence

            print(f"  ★ {name}: Updated with real MD RMSF ({len(rmsf_avg)} residues)")
            md_updated += 1

    # Count RMSF sources
    md_count = sum(1 for t in targets if t.get('rmsf_source', '').startswith('ATLAS MD'))
    bfactor_count = sum(1 for t in targets if 'B-factor' in t.get('rmsf_source', ''))

    print()
    print(f"RMSF sources:")
    print(f"  Real MD (ATLAS):     {md_count}")
    print(f"  B-factor derived:    {bfactor_count}")
    print(f"  Total:               {len(targets)}")

    # Save updated targets
    benchmark_targets_path = benchmark_dir / "atlas_targets.json"
    benchmark_dir.mkdir(parents=True, exist_ok=True)

    with open(benchmark_targets_path, 'w') as f:
        json.dump(targets, f, indent=2)

    print()
    print(f"Saved benchmark targets to: {benchmark_targets_path}")

    # Also update the original
    with open(targets_path, 'w') as f:
        json.dump(targets, f, indent=2)
    print(f"Updated original targets: {targets_path}")

    # Create benchmark summary
    summary = {
        'total_proteins': len(targets),
        'md_rmsf_count': md_count,
        'bfactor_rmsf_count': bfactor_count,
        'proteins': []
    }

    for t in targets:
        summary['proteins'].append({
            'name': t['name'],
            'n_residues': t['n_residues'],
            'rmsf_source': t.get('rmsf_source', 'unknown'),
            'mean_rmsf': sum(t['md_rmsf']) / len(t['md_rmsf']) if t.get('md_rmsf') else None
        })

    summary_path = benchmark_dir / "benchmark_summary.json"
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"Created benchmark summary: {summary_path}")

    # Copy test CSV
    import shutil
    test_csv_src = data_dir / "atlas_test.csv"
    test_csv_dst = benchmark_dir / "atlas_test.csv"
    if test_csv_src.exists():
        shutil.copy(test_csv_src, test_csv_dst)
        print(f"Copied test CSV: {test_csv_dst}")

    # Create symlinks to PDB files
    pdb_src = data_dir / "pdb"
    pdb_dst = benchmark_dir / "pdb"
    if pdb_src.exists() and not pdb_dst.exists():
        pdb_dst.symlink_to(pdb_src)
        print(f"Linked PDB directory: {pdb_dst} -> {pdb_src}")

    print()
    print("╔════════════════════════════════════════════════════════════════╗")
    print("║  BENCHMARK SETUP COMPLETE                                      ║")
    print("╠════════════════════════════════════════════════════════════════╣")
    print(f"║  Total proteins:          {len(targets):3d}                               ║")
    print(f"║  With real MD RMSF:       {md_count:3d}                               ║")
    print(f"║  With B-factor RMSF:      {bfactor_count:3d}                               ║")
    print("╠════════════════════════════════════════════════════════════════╣")
    print("║  Ready to run: cargo run --bin prism-atlas                     ║")
    print("╚════════════════════════════════════════════════════════════════╝")


if __name__ == "__main__":
    main()
