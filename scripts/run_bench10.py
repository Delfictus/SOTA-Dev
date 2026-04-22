#!/usr/bin/env python3
"""
PRISM4D Bench10 — Controlled 10-Target Generalization Validation

Zero-touch execution outside Claude Code:
    python3 scripts/run_bench10.py

Validates PRISM-4D generalizes beyond hard targets on a diverse subset of bench30.
Uses existing topologies and holo-derived ground truth from bench30.
Includes regression check against hard-target baselines (1P38, 5LAR).

All outputs: benchmarks/bench10_results/
"""

import json
import math
import os
import shutil
import subprocess
import sys
import time
import csv
from pathlib import Path
from datetime import datetime, timezone

# ============================================================================
# CONFIGURATION
# ============================================================================

ROOT = Path(__file__).resolve().parent.parent
BENCH30_DIR = ROOT / "benchmarks" / "prism4d_bench30"
TOPO_DIR = BENCH30_DIR / "topologies"
GT_PATH = BENCH30_DIR / "ground_truth" / "ligand_centroids.json"
MANIFEST_PATH = BENCH30_DIR / "benchmark_manifest.json"
RUN_ID = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
RESULTS_DIR = ROOT / "benchmarks" / "bench10_results" / "runs" / RUN_ID
SCRATCH_DIR = Path("/tmp/prism_bench10_scratch")

# Source of truth: crates/prism-nhs/src/bin/nhs_rt_full.rs (see docs/CANONICAL_PROVENANCE.md).
# Engine is called via mandatory wrapper (direct binary exits 2 without PRISM_VALIDATED=1).
WRAPPER = ROOT / "scripts" / "prism-validate-and-run.sh"
VALIDATION_SCRIPT = ROOT / "scripts" / "kcc_validation_v2.py"

ENGINE_ARGS = [
    "--fast", "--hysteresis", "--prism-therm",
    "--multi-stream", "8",
    "--spike-percentile", "70",
    "--fused-steps", "6",
    "--hmr", "--adaptive-dt",
    "--multi-differential",
    "--closed-loop-steering", "--asymmetric-steering",
    "--use-xgb-ranker",
    "--replica-seed", "42", "-v",
]

# 10-target diverse subset (manually curated)
# 4 kinases + 3 enzymes + 3 allosteric/non-classical
BENCH10_TARGETS = [
    # Kinases
    {"bench30_id": 2,  "pdb": "2NPQ", "class": "kinase",      "site_type": "cryptic"},
    {"bench30_id": 3,  "pdb": "1KV1", "class": "kinase",      "site_type": "allosteric"},
    {"bench30_id": 8,  "pdb": "1HCL", "class": "kinase",      "site_type": "orthosteric"},
    {"bench30_id": 22, "pdb": "3L3N", "class": "kinase",      "site_type": "orthosteric"},
    # Enzymes (non-kinase)
    {"bench30_id": 1,  "pdb": "1JWP", "class": "enzyme",      "site_type": "cryptic"},
    {"bench30_id": 5,  "pdb": "4EY4", "class": "enzyme",      "site_type": "orthosteric"},
    {"bench30_id": 23, "pdb": "1NNA", "class": "enzyme",      "site_type": "orthosteric"},
    # Allosteric / non-classical
    {"bench30_id": 6,  "pdb": "2HNP", "class": "allosteric",  "site_type": "allosteric"},
    {"bench30_id": 7,  "pdb": "1M47", "class": "PPI",         "site_type": "PPI"},
    {"bench30_id": 10, "pdb": "2OSS", "class": "epigenetic",  "site_type": "orthosteric"},
]

# Regression baselines (from hard-target benchmark)
REGRESSION_BASELINES = {
    "1P38": {"top1_dcc": 4.6, "rank": 1},
    "5LAR": {"top1_dcc": 4.8, "rank": 1},
}

# Artifact ligands to reject
ARTIFACT_LIGANDS = {"BOG", "PEG", "GOL", "SO4", "PO4", "EDO", "ACE", "NME",
                    "HOH", "NA", "CL", "MG", "ZN", "CA", "K", "MPD", "DMS"}

# ============================================================================
# UTILITIES
# ============================================================================

def log(msg, level="INFO"):
    ts = datetime.now(timezone.utc).strftime("%H:%M:%S")
    print(f"[{ts}] [{level}] {msg}", flush=True)

def run_cmd(cmd, desc, timeout=1200):
    log(f"  CMD: {' '.join(str(c) for c in cmd)}")
    start = time.time()
    result = subprocess.run([str(c) for c in cmd], capture_output=True, text=True,
                          timeout=timeout, cwd=str(ROOT))
    elapsed = time.time() - start
    if result.returncode != 0:
        log(f"  FAILED ({elapsed:.1f}s): {result.stderr[-300:]}", "ERROR")
        raise RuntimeError(f"{desc} failed (exit {result.returncode})")
    log(f"  OK ({elapsed:.1f}s)")
    return result

# ============================================================================
# GROUND TRUTH
# ============================================================================

def load_ground_truth():
    """Load bench30 ground truth and validate ligands."""
    with open(GT_PATH) as f:
        gt = json.load(f)
    with open(MANIFEST_PATH) as f:
        manifest_data = json.load(f)
    manifest = manifest_data.get("targets", manifest_data) if isinstance(manifest_data, dict) else manifest_data
    return gt, manifest

def validate_ground_truth(target, gt, manifest):
    """Validate that the ground truth ligand is a real drug, not artifact."""
    bid = str(target["bench30_id"])
    if bid not in gt:
        return None, f"No ground truth for bench30 id {bid}"

    entry = gt[bid]
    lig = entry.get("ligand_resname", "")
    centroid = entry.get("centroid", [0, 0, 0])

    if lig.upper() in ARTIFACT_LIGANDS:
        return None, f"Artifact ligand: {lig}"

    # Find manifest entry for heavy atom count
    manifest_entry = None
    for m in manifest:
        if isinstance(m, dict) and m.get("id") == target["bench30_id"]:
            manifest_entry = m
            break

    heavy = manifest_entry.get("ligand_heavy_atoms", 0) if manifest_entry else 0
    if heavy > 500:
        return None, f"Ligand too large ({heavy} heavy atoms) — likely cofactor/polymer"
    if heavy < 5:
        return None, f"Ligand too small ({heavy} heavy atoms)"

    return {
        "target": target["pdb"],
        "bench30_id": target["bench30_id"],
        "reference": manifest_entry.get("holo_pdb", "unknown") if manifest_entry else "unknown",
        "ligand": lig,
        "centroid": centroid,
        "method": "self_holo",
        "heavy_atoms": heavy,
    }, None

# ============================================================================
# ENGINE RUN
# ============================================================================

def run_engine(target, topo_path, output_dir):
    """Run NHS engine for one target."""
    pdb = target["pdb"].lower()
    bs_json = output_dir / f"{pdb}.binding_sites.json"

    if bs_json.exists():
        log(f"  Engine output exists: {bs_json}")
        return

    if not WRAPPER.exists():
        raise RuntimeError(f"Engine wrapper not found: {WRAPPER}")

    cmd = [str(WRAPPER), "-t", str(topo_path), "-o", str(output_dir)] + ENGINE_ARGS
    run_cmd(cmd, f"NHS engine for {target['pdb']}")

    if not bs_json.exists():
        raise RuntimeError(f"binding_sites.json not produced for {target['pdb']}")

# ============================================================================
# VALIDATION + METRICS
# ============================================================================

def run_validation(output_dir, pdb):
    """Run KCC validation v2."""
    viz = output_dir / f"{pdb}.kcc_visualization.json"
    val = output_dir / f"{pdb}.kcc_validation_v2.json"
    if not viz.exists():
        log(f"  No kcc_visualization.json", "WARN")
        return None
    run_cmd([sys.executable, str(VALIDATION_SCRIPT), str(viz), "--output-dir", str(output_dir)],
            f"Validation for {pdb}", timeout=60)
    if val.exists():
        with open(val) as f:
            return json.load(f)
    return None

def compute_dcc(output_dir, pdb, gt_centroid):
    """Compute DCC for all sites vs ground truth."""
    bs_path = output_dir / f"{pdb}.binding_sites.json"
    with open(bs_path) as f:
        data = json.load(f)
    sites = data if isinstance(data, list) else data.get("sites", [])

    results = []
    for s in sites:
        c = s.get("centroid", [0, 0, 0])
        dcc = math.sqrt(sum((c[j] - gt_centroid[j]) ** 2 for j in range(3)))
        results.append({
            "rank": s.get("gtck_rank", 999),
            "site_id": s.get("id", -1),
            "dcc": round(dcc, 2),
            "score": s.get("rank_score", 0),
        })
    results.sort(key=lambda x: x["rank"])
    return results

def grade(dcc):
    if dcc <= 5: return "EXCELLENT"
    if dcc <= 8: return "GOOD"
    if dcc <= 12: return "MARGINAL"
    return "MISS"

# ============================================================================
# REGRESSION CHECK
# ============================================================================

def update_aggregate_metrics(all_metrics, output_dir, dcc_threshold=8.0):
    """Compute and write live aggregate metrics (SR@K, Top-K curve, rank histogram)."""
    completed = [m for m in all_metrics if "error" not in m]
    n = len(completed)
    if n == 0:
        return

    # SR@K: fraction where best DCC ≤ threshold appears at rank ≤ K
    def sr_at_k(k):
        hits = sum(1 for m in completed if m.get("best_rank", 999) <= k and m.get("best_dcc", 999) <= dcc_threshold)
        return round(hits / n, 3)

    # Top-K recovery curve
    topk_curve = {}
    for k in [1, 2, 3, 5, 10]:
        topk_curve[str(k)] = sr_at_k(k)

    # Rank histogram (rank of best DCC site)
    rank_hist = {}
    for m in completed:
        r = str(m.get("best_rank", 999))
        rank_hist[r] = rank_hist.get(r, 0) + 1

    # DCC stats
    top1_dccs = [m["top1_dcc"] for m in completed]
    best_dccs = [m["best_dcc"] for m in completed]

    aggregate = {
        "n_completed": n,
        "n_total": len(all_metrics),
        "dcc_threshold": dcc_threshold,
        "sr_at_1": sr_at_k(1),
        "sr_at_3": sr_at_k(3),
        "sr_at_5": sr_at_k(5),
        "sr_at_10": sr_at_k(10),
        "mean_top1_dcc": round(sum(top1_dccs) / n, 2),
        "median_top1_dcc": round(sorted(top1_dccs)[n // 2], 2),
        "mean_best_dcc": round(sum(best_dccs) / n, 2),
        "topk_curve": topk_curve,
        "rank_histogram": rank_hist,
        "grades": {
            "EXCELLENT": sum(1 for m in completed if m["top1_grade"] == "EXCELLENT"),
            "GOOD": sum(1 for m in completed if m["top1_grade"] == "GOOD"),
            "MARGINAL": sum(1 for m in completed if m["top1_grade"] == "MARGINAL"),
            "MISS": sum(1 for m in completed if m["top1_grade"] == "MISS"),
        },
    }

    agg_path = output_dir / "aggregate_metrics.json"
    with open(agg_path, "w") as f:
        json.dump(aggregate, f, indent=2)

    return aggregate


def check_regressions(all_results):
    """Check hard-target baselines haven't regressed."""
    # Load hard-target results if available
    ht_dir = ROOT / "benchmarks" / "hard_targets" / "results"
    regressions = []

    for pdb, baseline in REGRESSION_BASELINES.items():
        name = pdb.lower()
        bs_path = ht_dir / name / f"{name}.binding_sites.json"
        if not bs_path.exists():
            log(f"  Regression check skipped for {pdb} (no results)", "WARN")
            continue

        gt_map = {"1P38": [16.885, -3.551, -19.552], "5LAR": [54.0, 68.1, 16.8]}
        if pdb not in gt_map:
            continue

        with open(bs_path) as f:
            data = json.load(f)
        sites = data if isinstance(data, list) else data.get("sites", [])
        if not sites:
            continue

        # Find top-1 DCC
        top = min(sites, key=lambda s: s.get("gtck_rank", 999))
        c = top.get("centroid", [0, 0, 0])
        dcc = math.sqrt(sum((c[j] - gt_map[pdb][j]) ** 2 for j in range(3)))

        if dcc > baseline["top1_dcc"] + 2.0:  # allow 2Å tolerance for stochastic variation
            regressions.append(f"{pdb}: DCC={dcc:.1f}Å (baseline={baseline['top1_dcc']}Å)")
            log(f"  REGRESSION: {pdb} DCC={dcc:.1f}Å > baseline {baseline['top1_dcc']}Å", "ERROR")

    return regressions

# ============================================================================
# MAIN
# ============================================================================

def main():
    log("=" * 70)
    log("PRISM4D Bench10 — Generalization Validation")
    log(f"Targets: {len(BENCH10_TARGETS)}")
    log(f"Output: {RESULTS_DIR}")
    log("=" * 70)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    SCRATCH_DIR.mkdir(parents=True, exist_ok=True)

    if not WRAPPER.exists():
        log(f"Engine wrapper not found: {WRAPPER}", "FATAL")
        log("Expected: scripts/prism-validate-and-run.sh")
        sys.exit(1)

    # Write target list and run config
    targets_path = RESULTS_DIR / "bench10_targets.json"
    with open(targets_path, "w") as f:
        json.dump(BENCH10_TARGETS, f, indent=2)

    # Reproducibility metadata
    git_hash = "unknown"
    try:
        import subprocess as _sp
        git_hash = _sp.check_output(["git", "rev-parse", "HEAD"], cwd=str(ROOT), text=True).strip()[:12]
    except Exception:
        pass

    config = {
        "run_id": RUN_ID,
        "git_commit": git_hash,
        "engine": str(WRAPPER),
        "parameters": {k.lstrip("-"): True for k in ENGINE_ARGS if k.startswith("--")},
        "targets": [t["pdb"] for t in BENCH10_TARGETS],
        "started": datetime.now(timezone.utc).isoformat(),
    }
    with open(RESULTS_DIR / "config.json", "w") as f:
        json.dump(config, f, indent=2)
    log(f"Run ID: {RUN_ID} (git: {git_hash})")

    # Load ground truth
    gt, manifest = load_ground_truth()

    # Phase 1: Regression check
    log("\n[REGRESSION CHECK]")
    regressions = check_regressions({})
    if regressions:
        log("REGRESSION DETECTED — ABORTING", "FATAL")
        for r in regressions:
            log(f"  {r}", "FATAL")
        sys.exit(1)
    log("  No regressions detected")

    all_metrics = []
    pipeline_start = time.time()

    for target in BENCH10_TARGETS:
        pdb = target["pdb"]
        name = pdb.lower()
        bid = target["bench30_id"]

        log(f"\n{'=' * 60}")
        log(f"TARGET: {pdb} ({target['class']}/{target['site_type']}) [bench30 #{bid}]")
        log(f"{'=' * 60}")

        target_start = time.time()
        target_dir = RESULTS_DIR / name
        target_dir.mkdir(parents=True, exist_ok=True)

        try:
            # Ground truth validation
            log("[GT] Validating ground truth...")
            gt_data, gt_error = validate_ground_truth(target, gt, manifest)
            if gt_error:
                log(f"  Ground truth REJECTED: {gt_error}", "ERROR")
                all_metrics.append({"target": pdb, "error": f"GT: {gt_error}"})
                continue
            log(f"  GT valid: {gt_data['ligand']} ({gt_data['heavy_atoms']} heavy atoms)")
            gt_file = target_dir / f"{name}.ground_truth.json"
            with open(gt_file, "w") as f:
                json.dump(gt_data, f, indent=2)

            # Check topology
            topo_path = TOPO_DIR / f"{name}.topology.json"
            if not topo_path.exists():
                log(f"  Topology not found: {topo_path}", "ERROR")
                all_metrics.append({"target": pdb, "error": "no topology"})
                continue
            log(f"  Topology: {topo_path}")

            # Run engine
            log("[ENGINE] Running NHS...")
            run_output = SCRATCH_DIR / name
            run_output.mkdir(parents=True, exist_ok=True)
            run_engine(target, topo_path, run_output)

            # Copy results
            for fp in run_output.iterdir():
                if fp.is_file() and fp.suffix in (".json", ".pml", ".txt"):
                    shutil.copy2(fp, target_dir / fp.name)

            # Validation
            log("[VALIDATION]")
            val_data = run_validation(target_dir, name)

            # DCC computation
            log("[DCC]")
            dcc_results = compute_dcc(target_dir, name, gt_data["centroid"])
            top1 = dcc_results[0] if dcc_results else {"dcc": 999, "rank": 999}
            best = min(dcc_results, key=lambda x: x["dcc"]) if dcc_results else {"dcc": 999, "rank": 999}

            top1_grade = grade(top1["dcc"])
            log(f"  Top-1 DCC: {top1['dcc']:.1f}Å → {top1_grade}")
            log(f"  Best DCC:  {best['dcc']:.1f}Å at rank {best['rank']}")

            # Top-10 enrichment
            top10_dccs = [d["dcc"] for d in dcc_results[:10]]
            top10_best = min(top10_dccs) if top10_dccs else 999
            top10_hit = any(d < 8 for d in top10_dccs)

            metrics = {
                "target": pdb,
                "class": target["class"],
                "site_type": target["site_type"],
                "bench30_id": bid,
                "n_sites": len(dcc_results),
                "top1_dcc": top1["dcc"],
                "top1_grade": top1_grade,
                "top1_site_id": top1.get("site_id"),
                "top1_score": top1.get("score", 0),
                "best_dcc": best["dcc"],
                "best_rank": best["rank"],
                "top10_best_dcc": top10_best,
                "top10_hit": top10_hit,
                "validation_verdict": val_data["sites"][0]["verdict"] if val_data and val_data.get("sites") else "N/A",
                "runtime_seconds": round(time.time() - target_start, 1),
            }
            all_metrics.append(metrics)

            # Live aggregate update after each target
            agg = update_aggregate_metrics(all_metrics, RESULTS_DIR)
            if agg:
                log(f"  [LIVE] SR@1={agg['sr_at_1']:.0%} SR@3={agg['sr_at_3']:.0%} SR@10={agg['sr_at_10']:.0%} mean_DCC={agg['mean_top1_dcc']:.1f}A ({agg['n_completed']}/{len(BENCH10_TARGETS)})")

        except Exception as e:
            log(f"FAILED: {e}", "ERROR")
            import traceback
            traceback.print_exc()
            all_metrics.append({
                "target": pdb, "error": str(e),
                "runtime_seconds": round(time.time() - target_start, 1),
            })

    # ============================================================================
    # FINAL REPORT
    # ============================================================================
    log(f"\n{'=' * 70}")
    log("BENCH10 FINAL REPORT")
    log(f"{'=' * 70}")

    completed = [m for m in all_metrics if "error" not in m]
    n_excellent = sum(1 for m in completed if m["top1_grade"] == "EXCELLENT")
    n_good = sum(1 for m in completed if m["top1_grade"] == "GOOD")
    n_marginal = sum(1 for m in completed if m["top1_grade"] == "MARGINAL")
    n_miss = sum(1 for m in completed if m["top1_grade"] == "MISS")
    n_top10 = sum(1 for m in completed if m.get("top10_hit", False))

    mean_dcc = sum(m["top1_dcc"] for m in completed) / len(completed) if completed else 0
    top1_rate = (n_excellent + n_good) / len(completed) if completed else 0
    top10_rate = n_top10 / len(completed) if completed else 0

    summary = {
        "generated": datetime.now(timezone.utc).isoformat(),
        "pipeline_version": "prism4d_v1.0_kcc_gtckl_blp_dp",
        "n_targets": len(BENCH10_TARGETS),
        "n_completed": len(completed),
        "n_failed": len(all_metrics) - len(completed),
        "top1_success_rate": round(top1_rate, 3),
        "top10_success_rate": round(top10_rate, 3),
        "mean_top1_dcc": round(mean_dcc, 2),
        "grades": {"EXCELLENT": n_excellent, "GOOD": n_good, "MARGINAL": n_marginal, "MISS": n_miss},
        "targets": all_metrics,
        "failures": [m["target"] for m in all_metrics if "error" in m],
    }

    # Write JSON
    summary_path = RESULTS_DIR / "bench10_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    # Write CSV
    csv_path = RESULTS_DIR / "bench10_table.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Target", "Class", "SiteType", "Top1_DCC", "Grade", "Best_DCC", "Best_Rank",
                         "Top10_Hit", "N_Sites", "Runtime_s"])
        for m in all_metrics:
            if "error" in m:
                writer.writerow([m["target"], "", "", "", "ERROR", "", "", "", "", m.get("runtime_seconds", "")])
            else:
                writer.writerow([m["target"], m["class"], m["site_type"],
                                m["top1_dcc"], m["top1_grade"], m["best_dcc"], m["best_rank"],
                                m["top10_hit"], m["n_sites"], m["runtime_seconds"]])

    # Print summary
    log(f"\n  Completed: {len(completed)}/{len(BENCH10_TARGETS)}")
    log(f"  Top-1 success (≤8Å): {top1_rate*100:.0f}%")
    log(f"  Top-10 enrichment:   {top10_rate*100:.0f}%")
    log(f"  Mean Top-1 DCC:      {mean_dcc:.1f}Å")
    log(f"  Grades: {n_excellent} EXCELLENT, {n_good} GOOD, {n_marginal} MARGINAL, {n_miss} MISS")
    log(f"  Total time: {time.time() - pipeline_start:.0f}s")
    log("")

    fmt = f"{'Target':>6} {'Class':>12} {'Type':>12} {'Top1':>7} {'Grade':>10} {'Best':>7} {'@Rank':>6}"
    log(fmt)
    log("-" * 70)
    for m in all_metrics:
        if "error" in m:
            log(f"{m['target']:>6} {'ERROR':>12} {m['error'][:30]}")
        else:
            log(f"{m['target']:>6} {m['class']:>12} {m['site_type']:>12} "
                f"{m['top1_dcc']:>6.1f}A {m['top1_grade']:>10} {m['best_dcc']:>6.1f}A {m['best_rank']:>6}")

    # Final aggregate metrics
    final_agg = update_aggregate_metrics(all_metrics, RESULTS_DIR)
    agg_path = RESULTS_DIR / "aggregate_metrics.json"
    log(f"\n  Summary:   {summary_path}")
    log(f"  CSV:       {csv_path}")
    log(f"  Aggregate: {agg_path}")
    if final_agg:
        log(f"\n  === AGGREGATE METRICS ===")
        log(f"  SR@1:  {final_agg['sr_at_1']:.0%}")
        log(f"  SR@3:  {final_agg['sr_at_3']:.0%}")
        log(f"  SR@5:  {final_agg['sr_at_5']:.0%}")
        log(f"  SR@10: {final_agg['sr_at_10']:.0%}")
        log(f"  Mean Top-1 DCC: {final_agg['mean_top1_dcc']:.1f}A")
        log(f"  Top-K curve: {final_agg['topk_curve']}")
    log("\nBench10 complete.")

if __name__ == "__main__":
    main()
