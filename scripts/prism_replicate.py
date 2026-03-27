#!/usr/bin/env python3
"""PRISM4D — Replicate Runner + Consensus.

Runs N stochastic replicates of the full pipeline (engine → gating →
design layers), then builds cross-run metastable pocket consensus.

Usage:
    python3 scripts/prism_replicate.py \\
        --topology /path/to/topology.json \\
        --target-name 1btl \\
        --pdb-id 1BTL \\
        --n-replicates 5 \\
        --output-dir /tmp/prism_1btl_consensus/

This is the canonical way to run PRISM for production results.
Single runs are for debugging only.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import List, Optional


def find_engine_binary() -> str:
    """Find nhs_rt_full binary."""
    candidates = [
        Path(__file__).parent.parent / "target" / "release" / "nhs_rt_full",
        Path.home() / "Desktop" / "Prism4D-bio" / "target" / "release" / "nhs_rt_full",
    ]
    for p in candidates:
        if p.exists():
            return str(p)
    raise FileNotFoundError("nhs_rt_full binary not found")


def run_engine(
    topology: str, output_dir: str, seed: int = 42, verbose: bool = True
) -> bool:
    """Run the Rust engine once with an explicit seed."""
    binary = find_engine_binary()
    cmd = [
        binary,
        "-t", topology,
        "-o", output_dir,
        "--fast", "--hysteresis",
        "--multi-stream", "8",
        "--spike-percentile", "95",
        "--prism-therm",
        "--fused-steps", "4",
        "--hmr", "--adaptive-dt",
        "--replica-seed", str(seed),
    ]
    if verbose:
        cmd.append("-v")

    result = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)

    if result.returncode not in (0, 134, 139):
        # 134/139 = CUDA teardown segfault — output is still valid
        print(f"    Engine failed with code {result.returncode}", file=sys.stderr)
        return False
    return True


def run_canonical_pipeline(
    output_dir: str, target_name: str, pdb_id: str
) -> bool:
    """Run the canonical Python pipeline on engine output."""
    cmd = [
        sys.executable, "-m", "scripts.prism_canonical",
        "--output-dir", output_dir,
        "--target-name", target_name,
        "--pdb-id", pdb_id,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    if result.returncode != 0:
        print(f"    Pipeline failed: {result.stderr[-200:]}", file=sys.stderr)
        return False
    return True


def run_consensus(
    run_dirs: List[str], target_name: str, output_dir: str
) -> bool:
    """Run consensus layer."""
    cmd = [
        sys.executable, "-m", "scripts.consensus",
        "--run-dirs", *run_dirs,
        "--target-name", target_name,
        "--out", output_dir,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    if result.returncode != 0:
        print(f"    Consensus failed: {result.stderr[-200:]}", file=sys.stderr)
        return False
    # Print consensus output
    print(result.stdout)
    return True


def main() -> None:
    parser = argparse.ArgumentParser(
        description="PRISM4D Replicate Runner + Consensus"
    )
    parser.add_argument("--topology", required=True)
    parser.add_argument("--target-name", required=True)
    parser.add_argument("--pdb-id", default="")
    parser.add_argument("--n-replicates", type=int, default=5)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--skip-engine", action="store_true",
                        help="Skip engine runs (use existing output)")
    args = parser.parse_args()

    if not args.pdb_id:
        args.pdb_id = args.target_name.upper()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    n = args.n_replicates
    run_dirs: List[str] = []
    total_start = time.time()

    print(f"╔{'═'*58}╗")
    print(f"║  PRISM4D REPLICATE CONSENSUS: {args.target_name:<27s}║")
    print(f"║  Replicates: {n:<44d}║")
    print(f"╚{'═'*58}╝")

    # Phase 1: Run N replicates
    for i in range(n):
        run_dir = str(out / f"rep_{i:02d}")
        Path(run_dir).mkdir(parents=True, exist_ok=True)
        run_dirs.append(run_dir)

        print(f"\n[Rep {i+1}/{n}] ", end="", flush=True)

        if not args.skip_engine:
            # Each replicate gets a distinct seed: base + i*1000
            # Ensures genuinely different stochastic realizations
            seed = 42 + i * 1000
            t0 = time.time()
            print(f"Engine (seed={seed})... ", end="", flush=True)
            ok = run_engine(args.topology, run_dir, seed=seed)
            dt = time.time() - t0
            if not ok:
                print(f"FAILED ({dt:.0f}s)")
                continue
            print(f"done ({dt:.0f}s). ", end="", flush=True)

        # Check engine output exists
        bs = list(Path(run_dir).glob("*.binding_sites.json"))
        if not bs:
            print("No binding_sites.json — skipping")
            continue

        print("Pipeline... ", end="", flush=True)
        t0 = time.time()
        ok = run_canonical_pipeline(run_dir, args.target_name, args.pdb_id)
        dt = time.time() - t0
        if ok:
            print(f"done ({dt:.0f}s)")
        else:
            print(f"FAILED ({dt:.0f}s)")

    # Filter to successful runs
    valid_dirs = [
        d for d in run_dirs
        if (Path(d) / "design" / "gating_result.json").exists()
    ]

    if len(valid_dirs) < 2:
        print(f"\n❌ Only {len(valid_dirs)} valid runs. Need ≥2 for consensus.")
        sys.exit(1)

    # Phase 2: Consensus
    print(f"\n[Consensus] Clustering {len(valid_dirs)} replicates...")
    consensus_dir = str(out / "consensus")
    ok = run_consensus(valid_dirs, args.target_name, consensus_dir)

    total_time = time.time() - total_start

    if ok:
        print(f"\nTotal time: {total_time:.0f}s")
        print(f"Consensus output: {consensus_dir}/consensus_sites.json")
    else:
        print(f"\n❌ Consensus failed after {total_time:.0f}s")
        sys.exit(1)


if __name__ == "__main__":
    main()
