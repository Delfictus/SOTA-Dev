#!/usr/bin/env python3
"""
PRISM4D: Batched Multi-Structure Pipeline

Processes multiple protein structures in parallel for high throughput.
Same pipeline as prism_pipeline.py but optimized for batch processing.

Pipeline (per structure):
  Stage 1: Fetch & Sanitize PDB (PDBFixer)
  Stage 2: Generate AMBER topology (OpenMM)
  Stage 3: Run GPU MD simulation (PRISM generate-ensemble-batch)
  Stage 4: Analyze conformational dynamics (analyze_ensemble_batch)

Usage:
    # Process multiple PDB IDs
    python prism_pipeline_batch.py --pdbs 6M0J 2VWD 1AKE --output-dir results/

    # Process multiple local files
    python prism_pipeline_batch.py --files *.pdb --output-dir results/

    # From a list file (one PDB ID per line)
    python prism_pipeline_batch.py --list pdb_list.txt --output-dir results/

    # With chain selection
    python prism_pipeline_batch.py --pdbs 6M0J:E 2VWD:A --output-dir results/

Dependencies:
    conda activate prism-delta
    # Requires: openmm, pdbfixer, numpy
    # Requires: PRISM binaries (generate-ensemble-batch, analyze_ensemble_batch)
"""

import sys
import os
import argparse
import subprocess
import json
import time
from pathlib import Path
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
import multiprocessing

# Add script directory to path
SCRIPT_DIR = Path(__file__).parent.resolve()
sys.path.insert(0, str(SCRIPT_DIR))

from stage1_sanitize import sanitize_pdb, is_pdb_id
from stage2_topology import prepare_topology


def find_binary(name):
    """Find a PRISM binary by name."""
    candidates = [
        SCRIPT_DIR.parent / "target" / "release" / name,
        Path.home() / "Desktop" / "PRISM4D-v1.1.0-STABLE" / "target" / "release" / name,
        Path.home() / "Desktop" / "PRISM-Delta-v1.0" / "target" / "release" / name,
    ]

    for path in candidates:
        if path.exists():
            return str(path)

    result = subprocess.run(["which", name], capture_output=True, text=True)
    if result.returncode == 0:
        return result.stdout.strip()

    return None


def parse_pdb_spec(spec):
    """Parse PDB ID with optional chain: '6M0J' or '6M0J:E'"""
    if ':' in spec:
        pdb_id, chain = spec.split(':', 1)
        return pdb_id.strip(), chain.strip()
    return spec.strip(), None


def process_single_structure(
    pdb_spec,
    output_dir,
    steps=50000,
    temperature=310.0,
    restraint_k=2.0,
    verbose=False
):
    """
    Process a single structure through stages 1-2 (preparation).
    Returns topology path and metadata for batch MD.
    """
    pdb_input, chain = parse_pdb_spec(pdb_spec)

    # Determine base name
    if is_pdb_id(pdb_input):
        base_name = pdb_input.upper()
    else:
        base_name = Path(pdb_input).stem

    if chain:
        base_name = f"{base_name}_{chain}"

    struct_dir = output_dir / base_name
    struct_dir.mkdir(parents=True, exist_ok=True)

    result = {
        "pdb_spec": pdb_spec,
        "base_name": base_name,
        "output_dir": str(struct_dir),
        "chain": chain,
        "success": False,
        "error": None,
    }

    try:
        # Stage 1: Sanitize
        sanitized_pdb = struct_dir / f"{base_name}_sanitized.pdb"
        sanitize_pdb(
            pdb_input,
            str(sanitized_pdb),
            chain=chain,
            keep_waters=False,
            verbose=verbose
        )
        result["sanitized_pdb"] = str(sanitized_pdb)

        # Stage 2: Topology
        topology_json = struct_dir / f"{base_name}_topology.json"
        topo_result = prepare_topology(
            str(sanitized_pdb),
            str(topology_json),
            solvate=False,
            minimize=True,
            verbose=verbose
        )
        result["topology_json"] = str(topology_json)
        result["n_atoms"] = topo_result["n_atoms"]
        result["n_residues"] = topo_result["n_residues"]
        result["success"] = True

    except Exception as e:
        result["error"] = str(e)

    return result


def run_batch_md(
    topology_paths,
    output_dir,
    steps=50000,
    temperature=310.0,
    restraint_k=2.0,
    max_concurrent=2,
    verbose=True
):
    """
    Run batched MD using generate_ensemble_batch binary.
    """
    binary = find_binary("generate_ensemble_batch")

    if binary is None:
        # Fall back to sequential processing with single binary
        binary = find_binary("generate-ensemble")
        if binary is None:
            raise RuntimeError("Neither generate_ensemble_batch nor generate-ensemble found")

        if verbose:
            print("  Using sequential MD (batch binary not found)")

        # Sequential fallback
        results = []
        for topo_path in topology_paths:
            struct_dir = Path(topo_path).parent
            base_name = Path(topo_path).stem.replace("_topology", "")
            sanitized_pdb = struct_dir / f"{base_name}_sanitized.pdb"
            ensemble_pdb = struct_dir / f"{base_name}_ensemble.pdb"

            cmd = [
                binary,
                "--topology", topo_path,
                "--pdb", str(sanitized_pdb),
                "--steps", str(steps),
                "--temperature", str(temperature),
                "--restraint-k", str(restraint_k),
                "--output", str(ensemble_pdb),
            ]

            result = subprocess.run(cmd, capture_output=True, text=True)
            results.append({
                "topology": topo_path,
                "ensemble": str(ensemble_pdb),
                "success": result.returncode == 0,
                "error": result.stderr if result.returncode != 0 else None
            })

        return results

    # Use batch binary
    if verbose:
        print(f"  Using batched MD: {binary}")
        print(f"  Processing {len(topology_paths)} structures")

    cmd = [
        binary,
        "--topologies", *topology_paths,
        "--steps", str(steps),
        "--temperature", str(temperature),
        "--restraint-k", str(restraint_k),
        "--output-dir", str(output_dir),
        "--max-concurrent", str(max_concurrent),
    ]

    if not verbose:
        cmd.append("--quiet")

    result = subprocess.run(cmd, capture_output=True, text=True)

    if verbose and result.stdout:
        print(result.stdout)

    if result.returncode != 0:
        print(f"Batch MD failed: {result.stderr}")

    # Collect results
    results = []
    for topo_path in topology_paths:
        struct_dir = Path(topo_path).parent
        base_name = Path(topo_path).stem.replace("_topology", "")
        ensemble_pdb = struct_dir / f"{base_name}_ensemble.pdb"

        results.append({
            "topology": topo_path,
            "ensemble": str(ensemble_pdb),
            "success": ensemble_pdb.exists(),
        })

    return results


def run_batch_analysis(
    ensemble_paths,
    output_dir,
    verbose=True
):
    """
    Run batched analysis using analyze_ensemble_batch binary.
    """
    binary = find_binary("analyze_ensemble_batch")

    if binary is None:
        # Fall back to single analysis binary
        binary = find_binary("analyze_ensemble")
        if binary is None:
            if verbose:
                print("  Analysis binaries not found, using Python fallback")
            return None

        if verbose:
            print("  Using sequential analysis")

        results = []
        for ens_path in ensemble_paths:
            struct_dir = Path(ens_path).parent
            output_json = struct_dir / "analysis_results.json"

            cmd = [
                binary,
                "--ensemble", ens_path,
                "--output", str(output_json),
            ]

            if not verbose:
                cmd.append("--quiet")

            result = subprocess.run(cmd, capture_output=True, text=True)
            results.append({
                "ensemble": ens_path,
                "output": str(output_json),
                "success": result.returncode == 0,
            })

        return results

    # Use batch binary
    if verbose:
        print(f"  Using batched analysis: {binary}")

    cmd = [
        binary,
        "--ensembles", *ensemble_paths,
        "--output-dir", str(output_dir),
    ]

    if not verbose:
        cmd.append("--quiet")

    result = subprocess.run(cmd, capture_output=True, text=True)

    if verbose and result.stdout:
        print(result.stdout)

    return None  # Results written to individual files


def run_batch_pipeline(
    pdb_specs,
    output_dir,
    steps=50000,
    temperature=310.0,
    restraint_k=2.0,
    max_workers=4,
    max_gpu_concurrent=2,
    verbose=True
):
    """
    Run complete batched pipeline for multiple structures.
    """
    start_time = time.time()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    n_structures = len(pdb_specs)

    if verbose:
        print("=" * 70)
        print("PRISM4D BATCH PIPELINE")
        print("=" * 70)
        print(f"Structures: {n_structures}")
        print(f"Output: {output_dir}")
        print(f"MD steps: {steps}")
        print(f"Temperature: {temperature} K")
        print(f"Restraints: k = {restraint_k}")
        print()

    # ========== Stages 1-2: Parallel Preparation ==========
    if verbose:
        print("=" * 70)
        print("STAGES 1-2: Parallel Structure Preparation")
        print("=" * 70)

    prep_start = time.time()
    prep_results = []

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(
                process_single_structure,
                spec, output_dir, steps, temperature, restraint_k, False
            ): spec
            for spec in pdb_specs
        }

        for future in as_completed(futures):
            spec = futures[future]
            try:
                result = future.result()
                prep_results.append(result)
                status = "OK" if result["success"] else f"FAIL: {result['error']}"
                if verbose:
                    print(f"  {result['base_name']}: {status}")
            except Exception as e:
                if verbose:
                    print(f"  {spec}: ERROR - {e}")

    prep_time = time.time() - prep_start
    successful_preps = [r for r in prep_results if r["success"]]

    if verbose:
        print(f"\nPreparation complete: {len(successful_preps)}/{n_structures} succeeded ({prep_time:.1f}s)")

    if not successful_preps:
        print("No structures prepared successfully!")
        return None

    # ========== Stage 3: Batched GPU MD ==========
    if verbose:
        print()
        print("=" * 70)
        print("STAGE 3: Batched GPU Molecular Dynamics")
        print("=" * 70)

    topology_paths = [r["topology_json"] for r in successful_preps]

    md_start = time.time()
    md_results = run_batch_md(
        topology_paths,
        output_dir,
        steps=steps,
        temperature=temperature,
        restraint_k=restraint_k,
        max_concurrent=max_gpu_concurrent,
        verbose=verbose
    )
    md_time = time.time() - md_start

    successful_md = [r for r in md_results if r["success"]]
    if verbose:
        print(f"\nMD complete: {len(successful_md)}/{len(topology_paths)} succeeded ({md_time:.1f}s)")

    # ========== Stage 4: Batched Analysis ==========
    if successful_md:
        if verbose:
            print()
            print("=" * 70)
            print("STAGE 4: Batched Conformational Analysis")
            print("=" * 70)

        ensemble_paths = [r["ensemble"] for r in successful_md]

        analysis_start = time.time()
        run_batch_analysis(ensemble_paths, output_dir, verbose=verbose)
        analysis_time = time.time() - analysis_start

        if verbose:
            print(f"\nAnalysis complete ({analysis_time:.1f}s)")

    # ========== Summary ==========
    total_time = time.time() - start_time

    summary = {
        "timestamp": datetime.now().isoformat(),
        "n_structures": n_structures,
        "n_prepared": len(successful_preps),
        "n_md_complete": len(successful_md),
        "total_time_seconds": total_time,
        "prep_time_seconds": prep_time,
        "md_time_seconds": md_time,
        "throughput_structures_per_minute": n_structures / (total_time / 60),
        "structures": [r["base_name"] for r in successful_preps],
    }

    summary_path = output_dir / "batch_pipeline_summary.json"
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)

    if verbose:
        print()
        print("=" * 70)
        print("BATCH PIPELINE COMPLETE")
        print("=" * 70)
        print(f"Structures processed: {len(successful_md)}/{n_structures}")
        print(f"Total time: {total_time:.1f}s")
        print(f"Throughput: {summary['throughput_structures_per_minute']:.2f} structures/minute")
        print(f"\nOutputs: {output_dir}/")
        print(f"  - batch_pipeline_summary.json")
        for result in successful_preps:
            print(f"  - {result['base_name']}/")

    return summary


def main():
    parser = argparse.ArgumentParser(
        description='PRISM4D: Batched Multi-Structure Pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --pdbs 6M0J 2VWD 1AKE --output-dir results/
  %(prog)s --pdbs 6M0J:E 2VWD:A --output-dir results/   # With chains
  %(prog)s --files proteins/*.pdb --output-dir results/
  %(prog)s --list pdb_list.txt --output-dir results/
  %(prog)s --pdbs 6M0J 2VWD --steps 100000 --output-dir results/
        """
    )

    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument('--pdbs', nargs='+',
                            help='PDB IDs (e.g., 6M0J 2VWD) or with chains (6M0J:E)')
    input_group.add_argument('--files', nargs='+',
                            help='Local PDB files')
    input_group.add_argument('--list', type=str,
                            help='File containing PDB IDs (one per line)')

    parser.add_argument('--output-dir', '-o', required=True,
                        help='Output directory')
    parser.add_argument('--steps', '-n', type=int, default=50000,
                        help='MD steps per structure (default: 50000)')
    parser.add_argument('--temperature', '-T', type=float, default=310.0,
                        help='Temperature in Kelvin (default: 310)')
    parser.add_argument('--restraint-k', '-k', type=float, default=2.0,
                        help='Position restraint k (default: 2.0, 0=disabled)')
    parser.add_argument('--workers', '-w', type=int,
                        default=min(4, multiprocessing.cpu_count()),
                        help='Parallel workers for prep (default: 4)')
    parser.add_argument('--gpu-concurrent', '-g', type=int, default=2,
                        help='Max concurrent GPU simulations (default: 2)')
    parser.add_argument('--quiet', '-q', action='store_true',
                        help='Suppress output')

    args = parser.parse_args()

    # Collect PDB specs
    pdb_specs = []

    if args.pdbs:
        pdb_specs = args.pdbs
    elif args.files:
        pdb_specs = args.files
    elif args.list:
        with open(args.list) as f:
            pdb_specs = [line.strip() for line in f if line.strip() and not line.startswith('#')]

    if not pdb_specs:
        print("No structures to process!", file=sys.stderr)
        return 1

    try:
        run_batch_pipeline(
            pdb_specs,
            args.output_dir,
            steps=args.steps,
            temperature=args.temperature,
            restraint_k=args.restraint_k,
            max_workers=args.workers,
            max_gpu_concurrent=args.gpu_concurrent,
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
