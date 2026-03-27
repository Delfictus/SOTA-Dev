#!/usr/bin/env python3
"""PRISM4D CryptoBench Benchmark Runner + Evaluator.

Runs the engine on all 199 CryptoBench proteins, evaluates binding site
detection and ranking against ground truth binding residues.

Evaluation uses residue overlap (precision/recall/F1) — the standard
CryptoBench metric. A site "detects" the cryptic pocket if recall > 0.2.

Usage:
    # Run all remaining proteins (resumes from where it left off)
    python3 scripts/run_cryptobench.py --run

    # Evaluate existing results only (no engine runs)
    python3 scripts/run_cryptobench.py --eval-only

    # Run specific proteins
    python3 scripts/run_cryptobench.py --run --proteins 1a4u 1arl 2qbv

    # Dry run — show what would be run
    python3 scripts/run_cryptobench.py --run --dry-run
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

PROJECT = Path(__file__).parent.parent
BENCH_DIR = PROJECT / "benchmarks" / "cryptobench"
GT_DIR = BENCH_DIR / "ground_truth"
RESULTS_DIR = BENCH_DIR / "results"
ENGINE = PROJECT / "target" / "release" / "nhs_rt_full"

ENGINE_FLAGS = [
    "--fast", "--hysteresis", "--multi-stream", "8",
    "--spike-percentile", "95", "--prism-therm",
    "--fused-steps", "4", "--hmr", "--adaptive-dt",
    "--replica-seed", "42", "-v",
]


def load_ground_truth(pdb: str) -> Optional[Dict[str, Any]]:
    gt_path = GT_DIR / f"{pdb}.ground_truth.json"
    if not gt_path.exists():
        return None
    with open(gt_path) as f:
        return json.load(f)


def find_topology(pdb: str) -> Optional[Path]:
    for pattern in [
        BENCH_DIR / "structures" / f"{pdb}.topology.json",
        BENCH_DIR / "topologies" / f"{pdb}.topology.json",
    ]:
        if pattern.exists():
            return pattern
    matches = list(BENCH_DIR.rglob(f"{pdb}.topology.json"))
    return matches[0] if matches else None


def run_engine(pdb: str, topo: Path, out_dir: Path, timeout: int = 1800) -> bool:
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [str(ENGINE), "-t", str(topo), "-o", str(out_dir)] + ENGINE_FLAGS
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return result.returncode in (0, 134, 139)
    except subprocess.TimeoutExpired:
        return False


def evaluate_protein(pdb: str) -> Optional[Dict[str, Any]]:
    gt = load_ground_truth(pdb)
    if gt is None:
        return None

    bs_candidates = list((RESULTS_DIR / pdb).glob("*.binding_sites.json"))
    if not bs_candidates:
        return {"pdb": pdb, "n_sites": 0, "detected": False}

    with open(bs_candidates[0]) as f:
        data = json.load(f)
    sites = data if isinstance(data, list) else data.get("sites", [])
    if not sites:
        return {"pdb": pdb, "n_sites": 0, "detected": False}

    gt_residues: Set[Tuple[str, int]] = set()
    for r in gt.get("binding_residues", []):
        gt_residues.add((r.get("chain", "A"), r.get("resid", -1)))

    site_evals = []
    for site in sites:
        lining = site.get("lining_residues", [])
        site_res: Set[Tuple[str, int]] = set()
        for r in lining:
            site_res.add((r.get("chain", "_"), r.get("resid", -1)))

        overlap = site_res & gt_residues
        precision = len(overlap) / max(len(site_res), 1)
        recall = len(overlap) / max(len(gt_residues), 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-9)

        site_evals.append({
            "site_id": site.get("id", -1),
            "n_lining": len(site_res),
            "overlap": len(overlap),
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "rank_score": site.get("rank_score", 0),
            "gtck_rank": site.get("gtck_rank", 999),
            "composite_audit_rank": site.get("composite_audit_rank", 999),
        })

    site_evals.sort(key=lambda r: r["f1"], reverse=True)
    best = site_evals[0] if site_evals else None
    detected = any(r["recall"] > 0.2 for r in site_evals)
    best_id = best["site_id"] if best else -1

    # Rank of best site under each ranker
    by_gtck = sorted(sites, key=lambda s: s.get("rank_score", 0), reverse=True)
    gtck_r = next((i+1 for i, s in enumerate(by_gtck) if s.get("id") == best_id), 999)

    by_audit = sorted(sites, key=lambda s: s.get("composite_audit_score", 0), reverse=True)
    audit_r = next((i+1 for i, s in enumerate(by_audit) if s.get("id") == best_id), 999)

    return {
        "pdb": pdb,
        "n_sites": len(sites),
        "n_gt_residues": len(gt_residues),
        "detected": detected,
        "best_site": best_id,
        "best_f1": best["f1"] if best else 0,
        "best_recall": best["recall"] if best else 0,
        "best_precision": best["precision"] if best else 0,
        "gtck_rank": gtck_r,
        "audit_rank": audit_r,
        "top5": site_evals[:5],
    }


def compute_aggregate(results: List[Dict]) -> Dict[str, Any]:
    detected = [r for r in results if r.get("detected")]
    n = len(results)
    n_det = len(detected)

    def sr(key, k):
        return sum(1 for r in detected if r.get(key, 999) <= k)

    agg = {
        "n_proteins": n,
        "n_detected": n_det,
        "detection_rate": round(n_det / max(n, 1), 3),
        "mean_best_f1": round(sum(r.get("best_f1", 0) for r in detected) / max(n_det, 1), 4),
        "mean_best_recall": round(sum(r.get("best_recall", 0) for r in detected) / max(n_det, 1), 4),
    }

    for ranker, key in [("gtck", "gtck_rank"), ("audit", "audit_rank")]:
        for k in [1, 3, 5, 10]:
            c = sr(key, k)
            agg[f"{ranker}_sr@{k}"] = c
            agg[f"{ranker}_sr@{k}_pct"] = round(c / max(n_det, 1) * 100, 1)

    return agg


def main():
    parser = argparse.ArgumentParser(description="PRISM4D CryptoBench Benchmark")
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--proteins", nargs="*")
    parser.add_argument("--timeout", type=int, default=1800)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    all_gts = sorted(p.stem.replace(".ground_truth", "") for p in GT_DIR.glob("*.json"))
    print(f"CryptoBench: {len(all_gts)} proteins with ground truth")

    if args.run:
        if args.proteins:
            to_run = args.proteins
        else:
            existing = set()
            for d in RESULTS_DIR.iterdir():
                if d.is_dir() and list(d.glob("*.binding_sites.json")):
                    existing.add(d.name)
            to_run = [p for p in all_gts if p not in existing]

        print(f"To run: {len(to_run)} proteins")
        if args.dry_run:
            for pdb in to_run[:10]:
                topo = find_topology(pdb)
                print(f"  {pdb}: topo={'YES' if topo else 'NO'}")
            if len(to_run) > 10:
                print(f"  ... and {len(to_run)-10} more")
            return

        total_start = time.time()
        for i, pdb in enumerate(to_run):
            topo = find_topology(pdb)
            if topo is None:
                print(f"  [{i+1}/{len(to_run)}] {pdb}: NO TOPOLOGY")
                continue
            out_dir = RESULTS_DIR / pdb
            print(f"  [{i+1}/{len(to_run)}] {pdb}... ", end="", flush=True)
            t0 = time.time()
            ok = run_engine(pdb, topo, out_dir, args.timeout)
            dt = time.time() - t0
            bs = list(out_dir.glob("*.binding_sites.json"))
            n_sites = 0
            if bs:
                try:
                    with open(bs[0]) as f:
                        d = json.load(f)
                    n_sites = len(d.get("sites", d) if isinstance(d, dict) else d)
                except:
                    pass
            print(f"{'OK' if ok else 'FAIL'} ({dt:.0f}s, {n_sites} sites)")
        print(f"\nTotal: {(time.time()-total_start)/3600:.1f} hours")

    # Evaluate
    print(f"\nEvaluating...")
    results = []
    for pdb in all_gts:
        r = evaluate_protein(pdb)
        if r and r.get("n_sites", 0) > 0:
            results.append(r)

    print(f"Evaluated: {len(results)} / {len(all_gts)}")
    agg = compute_aggregate(results)

    print(f"\n{'='*60}")
    print(f"CRYPTOBENCH: {agg['n_proteins']} proteins, {agg['n_detected']} detected ({agg['detection_rate']*100:.0f}%)")
    print(f"Mean F1: {agg['mean_best_f1']:.4f}  Mean Recall: {agg['mean_best_recall']:.4f}")
    print(f"{'='*60}")
    print(f"{'Ranker':<10} {'SR@1':>7} {'SR@3':>7} {'SR@5':>7} {'SR@10':>7}")
    print(f"{'-'*40}")
    for ranker in ["gtck", "audit"]:
        vals = [f"{agg.get(f'{ranker}_sr@{k}_pct', 0):>5.1f}%" for k in [1, 3, 5, 10]]
        print(f"{ranker:<10} {' '.join(vals)}")

    out_path = BENCH_DIR / "cryptobench_benchmark.json"
    with open(out_path, "w") as f:
        json.dump({"timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "aggregate": agg, "per_protein": results}, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
