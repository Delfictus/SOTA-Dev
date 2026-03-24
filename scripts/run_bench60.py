#!/usr/bin/env python3
"""
PRISM4D BENCH60 — Full 60-target benchmark run.

Usage:
    python3 scripts/run_bench60.py

Runs all 60 bench30 targets (30 original + 30 extended) with:
- Identical engine parameters
- KCC validation
- DCC computation against validated ground truth
- Live aggregate metrics (SR@K, Top-K curve)
- Final report

All outputs: benchmarks/bench60_results/runs/<timestamp>/
"""

import json, math, os, shutil, subprocess, sys, time, csv
from pathlib import Path
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parent.parent
BENCH30_DIR = ROOT / "benchmarks" / "prism4d_bench30"
TOPO_DIR = BENCH30_DIR / "topologies"
GT_PATH = BENCH30_DIR / "ground_truth" / "ligand_centroids.json"
MANIFEST_PATH = BENCH30_DIR / "benchmark_manifest.json"
NHS_BINARY = ROOT / "target" / "release" / "nhs_rt_full"
VALIDATION_SCRIPT = ROOT / "scripts" / "kcc_validation_v2.py"

RUN_ID = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
RESULTS_DIR = ROOT / "benchmarks" / "bench60_results" / "runs" / RUN_ID
SCRATCH_DIR = Path("/tmp/prism_bench60_scratch")

ENGINE_ARGS = [
    "--fast", "--hysteresis", "--multi-stream", "8",
    "--spike-percentile", "95", "--prism-therm",
    "--fused-steps", "4", "--hmr", "--adaptive-dt", "-v",
]

# Targets to EXCLUDE (known issues)
EXCLUDE = {14}  # 1K3F — HEC (heme, 2250 atoms, not a drug)

def log(msg, level="INFO"):
    ts = datetime.now(timezone.utc).strftime("%H:%M:%S")
    print(f"[{ts}] [{level}] {msg}", flush=True)

def run_cmd(cmd, desc, timeout=1200):
    start = time.time()
    result = subprocess.run([str(c) for c in cmd], capture_output=True, text=True,
                          timeout=timeout, cwd=str(ROOT))
    elapsed = time.time() - start
    if result.returncode != 0:
        raise RuntimeError(f"{desc} failed (exit {result.returncode}): {result.stderr[-200:]}")
    return result, elapsed

def dcc(c, gt):
    return math.sqrt(sum((c[i]-gt[i])**2 for i in range(3)))

def grade(d):
    if d <= 5: return "EXCELLENT"
    if d <= 8: return "GOOD"
    if d <= 12: return "MARGINAL"
    return "MISS"

def update_aggregate(all_metrics, output_dir):
    completed = [m for m in all_metrics if "error" not in m]
    n = len(completed)
    if n == 0: return
    def sr(k, threshold=8.0):
        return sum(1 for m in completed if m.get("best_rank",999) <= k and m.get("best_dcc",999) <= threshold) / n
    topk = {str(k): round(sr(k), 3) for k in [1,2,3,5,10]}
    top1_dccs = [m["top1_dcc"] for m in completed]
    agg = {
        "n_completed": n, "n_total": len(all_metrics),
        "sr_at_1": sr(1), "sr_at_3": sr(3), "sr_at_5": sr(5), "sr_at_10": sr(10),
        "mean_top1_dcc": round(sum(top1_dccs)/n, 2),
        "topk_curve": topk,
        "grades": {g: sum(1 for m in completed if m.get("top1_grade")==g)
                   for g in ["EXCELLENT","GOOD","MARGINAL","MISS"]},
    }
    with open(output_dir / "aggregate_metrics.json", "w") as f:
        json.dump(agg, f, indent=2)
    return agg

def main():
    log("=" * 70)
    log(f"PRISM4D BENCH60 — Full Benchmark (Run {RUN_ID})")
    log("=" * 70)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    SCRATCH_DIR.mkdir(parents=True, exist_ok=True)

    if not NHS_BINARY.exists():
        log(f"NHS binary not found: {NHS_BINARY}", "FATAL")
        sys.exit(1)

    # Load manifest and ground truth
    with open(MANIFEST_PATH) as f:
        mdata = json.load(f)
    targets = mdata.get("targets", mdata) if isinstance(mdata, dict) else mdata
    with open(GT_PATH) as f:
        gt_all = json.load(f)

    # Write config
    git_hash = "unknown"
    try:
        git_hash = subprocess.check_output(["git","rev-parse","HEAD"], cwd=str(ROOT), text=True).strip()[:12]
    except: pass
    config = {"run_id": RUN_ID, "git_commit": git_hash, "engine_args": ENGINE_ARGS,
              "n_targets": len(targets), "excluded": list(EXCLUDE)}
    with open(RESULTS_DIR / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    all_metrics = []
    pipeline_start = time.time()

    for target in targets:
        if not isinstance(target, dict): continue
        tid = target.get("id", 0)
        if tid in EXCLUDE:
            log(f"SKIP target {tid} (excluded)")
            continue

        pdb = target.get("apo_pdb", "").lower()
        name = pdb
        bid = str(tid)

        log(f"\n[{tid:>2}/{len(targets)}] {pdb.upper()} ({target.get('protein_family','?')})")

        target_start = time.time()
        target_dir = RESULTS_DIR / name
        target_dir.mkdir(parents=True, exist_ok=True)

        try:
            # Check topology
            topo = TOPO_DIR / f"{name}.topology.json"
            if not topo.exists():
                raise RuntimeError(f"No topology: {topo}")

            # Run engine
            scratch = SCRATCH_DIR / name
            scratch.mkdir(parents=True, exist_ok=True)
            bs_json = scratch / f"{name}.binding_sites.json"
            if not bs_json.exists():
                log(f"  Running engine...")
                _, elapsed = run_cmd(
                    [str(NHS_BINARY), "-t", str(topo), "-o", str(scratch)] + ENGINE_ARGS,
                    f"Engine for {pdb}", timeout=1200)
                log(f"  Engine: {elapsed:.0f}s")
            else:
                log(f"  Using cached results")

            # Copy results
            for fp in scratch.iterdir():
                if fp.is_file() and fp.suffix in (".json", ".pml", ".txt"):
                    shutil.copy2(fp, target_dir / fp.name)

            # Validation
            viz = target_dir / f"{name}.kcc_visualization.json"
            if viz.exists():
                try:
                    run_cmd([sys.executable, str(VALIDATION_SCRIPT), str(viz),
                             "--output-dir", str(target_dir)],
                            f"Validation for {pdb}", timeout=60)
                except: pass

            # DCC
            gt_entry = gt_all.get(bid, {})
            gt_centroid = gt_entry.get("centroid")
            if not gt_centroid:
                all_metrics.append({"target": pdb, "bench30_id": tid, "error": "no_gt"})
                continue

            with open(target_dir / f"{name}.binding_sites.json") as f:
                data = json.load(f)
            sites = data if isinstance(data, list) else data.get("sites", [])
            if not sites:
                all_metrics.append({"target": pdb, "bench30_id": tid, "error": "no_sites"})
                continue

            dccs = []
            for s in sites:
                c = s.get("centroid", [0,0,0])
                d = dcc(c, gt_centroid)
                dccs.append((s.get("gtck_rank",999), s.get("id","?"), d, s.get("rank_score",0)))
            dccs.sort(key=lambda x: x[0])
            top1 = dccs[0]
            best = min(dccs, key=lambda x: x[2])

            metrics = {
                "target": pdb, "bench30_id": tid,
                "protein_family": target.get("protein_family", ""),
                "site_type": target.get("site_type", ""),
                "n_sites": len(sites),
                "top1_dcc": round(top1[2], 1),
                "top1_grade": grade(top1[2]),
                "best_dcc": round(best[2], 1),
                "best_rank": best[0],
                "runtime_seconds": round(time.time() - target_start, 1),
            }
            all_metrics.append(metrics)

            log(f"  Top1={top1[2]:.1f}A ({grade(top1[2])}) Best={best[2]:.1f}A@R{best[0]} [{len(sites)} sites]")

            # Live aggregate
            agg = update_aggregate(all_metrics, RESULTS_DIR)
            if agg:
                log(f"  [LIVE] SR@1={agg['sr_at_1']:.0%} SR@3={agg['sr_at_3']:.0%} SR@10={agg['sr_at_10']:.0%} "
                    f"mean={agg['mean_top1_dcc']:.1f}A ({agg['n_completed']}/{len(targets)-len(EXCLUDE)})")

        except Exception as e:
            log(f"  ERROR: {e}", "ERROR")
            all_metrics.append({"target": pdb, "bench30_id": tid, "error": str(e),
                               "runtime_seconds": round(time.time() - target_start, 1)})

    # Final report
    total_time = time.time() - pipeline_start
    completed = [m for m in all_metrics if "error" not in m]
    final_agg = update_aggregate(all_metrics, RESULTS_DIR)

    summary = {
        "run_id": RUN_ID, "git_commit": git_hash,
        "total_time_seconds": round(total_time, 1),
        "targets": all_metrics, "summary": final_agg,
    }
    with open(RESULTS_DIR / "bench60_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    # CSV
    with open(RESULTS_DIR / "bench60_table.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Target","Family","Type","Top1_DCC","Grade","Best_DCC","Best_Rank","N_Sites","Runtime"])
        for m in all_metrics:
            if "error" in m:
                w.writerow([m["target"],"","","","ERROR","","","",m.get("runtime_seconds","")])
            else:
                w.writerow([m["target"],m.get("protein_family",""),m.get("site_type",""),
                           m["top1_dcc"],m["top1_grade"],m["best_dcc"],m["best_rank"],
                           m["n_sites"],m["runtime_seconds"]])

    log(f"\n{'=' * 70}")
    log(f"BENCH60 COMPLETE")
    log(f"{'=' * 70}")
    if final_agg:
        log(f"  SR@1:  {final_agg['sr_at_1']:.0%}")
        log(f"  SR@3:  {final_agg['sr_at_3']:.0%}")
        log(f"  SR@10: {final_agg['sr_at_10']:.0%}")
        log(f"  Mean Top-1 DCC: {final_agg['mean_top1_dcc']:.1f}A")
        log(f"  Grades: {final_agg['grades']}")
    log(f"  Time: {total_time:.0f}s ({total_time/60:.1f}min)")
    log(f"  Report: {RESULTS_DIR / 'bench60_summary.json'}")

if __name__ == "__main__":
    main()
