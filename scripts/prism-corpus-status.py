#!/usr/bin/env python3
"""PRISM-4D Corpus Run Status Report Generator.

Comprehensive hourly status report for the proteome_1000 corpus
generation campaign. Validates output integrity, spike data recording,
DCC detection quality, and R2 sync pipeline health.

Usage:
    python3 scripts/prism-corpus-status.py
    python3 scripts/prism-corpus-status.py --output ~/Desktop/report.txt
"""

import argparse
import json
import math
import os
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

WORK_DIR = Path("/mnt/storage/prism-outputs/10k-runs")
LOG_DIR = Path("/mnt/storage/prism-outputs/_corpus_runner_logs")
R2_PREFIX = "10k-runs"
R2_BUCKET = "prism-archive"
HOLO_CACHE = Path.home() / ".cache" / "prism4d" / "holo_pdbs"
VALIDATION_CACHE = Path.home() / ".cache" / "prism4d" / "status_validation"

REQUIRED_FILES = [
    ".binding_sites.json",
    ".kcc_visualization.json",
    # spike_events checked separately: .parquet (preferred) OR .arrow (legacy)
    ".topology.prism_therm.json",
    "_ground_truth.json",
    # TWIN-only (removed from required — baseline canonical runs don't produce these):
    #   .topology.gcpid_synergy.json  — requires --multi-differential
    #   .topology.asc_consensus.json  — requires --multi-differential
]

def run_cmd(cmd, timeout=30):
    try:
        r = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=timeout)
        return r.stdout.strip(), r.returncode
    except (subprocess.TimeoutExpired, OSError):
        return "", 1

def section(title):
    return f"\n{'='*70}\n  {title}\n{'='*70}\n"

def subsection(title):
    return f"\n  ── {title} ──\n"

# ─────────────────────────────────────────────────────────────────────
# Section 1: Corpus run state
# ─────────────────────────────────────────────────────────────────────
def report_run_state():
    lines = [section("1. CORPUS RUN STATE")]

    # Find the latest per-target log
    logs = sorted(LOG_DIR.glob("run_*_per_target.log"), key=os.path.getmtime, reverse=True)
    summary_logs = sorted(LOG_DIR.glob("run_*_summary.log"), key=os.path.getmtime, reverse=True)

    if not logs:
        lines.append("  No per-target logs found. Corpus runner may not have started.\n")
        return "\n".join(lines), 0, [], []

    per_target_log = logs[0]
    summary_log = summary_logs[0] if summary_logs else None

    # Parse per-target results
    results = []
    with open(per_target_log) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            target = parts[0]
            status = parts[1] if len(parts) > 1 else "?"
            results.append({"target": target, "status": status, "raw": line})

    n_ok = sum(1 for r in results if r["status"] == "OK")
    n_fail = sum(1 for r in results if "FAIL" in r["status"])
    n_skip = sum(1 for r in results if "SKIP" in r["status"])
    n_total = len(results)

    # Engine PID
    engine_out, _ = run_cmd("ps -eo pid,etime,pcpu,comm 2>/dev/null | grep nhs_rt_full | grep -v grep | head -1")
    runner_out, _ = run_cmd("ps -eo pid,etime,comm 2>/dev/null | grep prism-corpus | grep -v grep | head -1")

    # Current target in flight
    current_target = "none"
    current_dirs = sorted(WORK_DIR.glob("*/run.log"), key=os.path.getmtime, reverse=True)
    if current_dirs:
        current_target = current_dirs[0].parent.name

    # Separate 13s* PanDDA family failures from other failures
    pandda_fails = [r for r in results if "FAIL" in r["status"] and r["target"].startswith("13s")]
    other_fails = [r for r in results if "FAIL" in r["status"] and not r["target"].startswith("13s")]
    non_pandda_total = sum(1 for r in results if not r["target"].startswith("13s"))
    non_pandda_ok = sum(1 for r in results if r["status"] == "OK" and not r["target"].startswith("13s"))

    pct_complete = 100.0 * n_total / 1098
    fail_rate = 100.0 * n_fail / n_total if n_total else 0
    non_pandda_fail_rate = 100.0 * len(other_fails) / non_pandda_total if non_pandda_total else 0

    lines.append(f"  Per-target log:    {per_target_log.name}")
    lines.append(f"  Progress:          {n_total} / 1098  ({pct_complete:.1f}%)")
    lines.append(f"    OK:              {n_ok}")
    lines.append(f"    FAILED:          {n_fail}  ({fail_rate:.0f}% overall)")
    lines.append(f"      13s* PanDDA:   {len(pandda_fails)}  (RING domain homo-oligomer deadlock, task #12)")
    lines.append(f"      Other:         {len(other_fails)}  ({non_pandda_fail_rate:.0f}% non-PanDDA fail rate)")
    lines.append(f"    SKIP:            {n_skip}")
    lines.append(f"  Current target:    {current_target}")
    lines.append(f"  Engine process:    {engine_out or '(not running)'}")

    # Disk + GPU
    disk_out, _ = run_cmd("df -h /mnt/storage 2>/dev/null | tail -1")
    gpu_out, _ = run_cmd("nvidia-smi --query-gpu=memory.used,memory.free,utilization.gpu --format=csv,noheader 2>/dev/null")
    lines.append(f"  Disk:              {disk_out}")
    lines.append(f"  GPU:               {gpu_out}")

    # ETA — split into "with timeouts" and "excluding timeouts"
    engine_times_all = []
    engine_times_ok = []
    for r in results:
        m = re.search(r'engine=(\d+)s', r["raw"])
        if m:
            t = int(m.group(1))
            engine_times_all.append(t)
            if r["status"] == "OK":
                engine_times_ok.append(t)

    remaining = 1098 - n_total
    if engine_times_all:
        avg_all = sum(engine_times_all) / len(engine_times_all)
        eta_all = (remaining * avg_all) / 3600
        lines.append(f"  Avg engine (all):  {avg_all:.0f}s ({avg_all/60:.1f}m) → ETA {eta_all:.0f}h ({eta_all/24:.1f}d)")
    if engine_times_ok:
        avg_ok = sum(engine_times_ok) / len(engine_times_ok)
        eta_ok = (remaining * avg_ok) / 3600
        lines.append(f"  Avg engine (OK):   {avg_ok:.0f}s ({avg_ok/60:.1f}m) → ETA {eta_ok:.0f}h ({eta_ok/24:.1f}d)  ← real pace")
    lines.append(f"  Remaining targets: {remaining}")

    # Health score
    if non_pandda_fail_rate > 10:
        health = "RED — non-PanDDA failures detected"
    elif fail_rate > 40:
        health = "YELLOW — high overall fail rate (PanDDA block)"
    elif fail_rate > 20:
        health = "YELLOW — elevated fail rate"
    else:
        health = "GREEN"
    lines.append(f"  Health:            {health}")

    ok_targets = [r["target"] for r in results if r["status"] == "OK"]
    fail_targets = [r for r in results if "FAIL" in r["status"]]

    return "\n".join(lines), n_total, ok_targets, fail_targets

# ─────────────────────────────────────────────────────────────────────
# Section 2: R2 sync pipeline health
# ─────────────────────────────────────────────────────────────────────
def report_r2_health():
    lines = [section("2. CLOUDFLARE R2 SYNC PIPELINE")]

    # rclone connectivity
    _, rc = run_cmd("rclone lsd r2:prism-archive/ 2>/dev/null | head -1", timeout=15)
    lines.append(f"  rclone connectivity:  {'OK' if rc == 0 else 'FAILED'}")

    # Count dirs in 10k-runs
    r2_dirs, _ = run_cmd(f"rclone lsf r2:{R2_BUCKET}/{R2_PREFIX}/ --dirs-only 2>/dev/null | wc -l")
    lines.append(f"  R2 10k-runs dirs:     {r2_dirs.strip()}")

    # Total size
    r2_size, _ = run_cmd(f"rclone size r2:{R2_BUCKET}/{R2_PREFIX}/ --json 2>/dev/null", timeout=60)
    if r2_size:
        try:
            d = json.loads(r2_size)
            gb = d.get("bytes", 0) / 1e9
            count = d.get("count", 0)
            lines.append(f"  R2 total size:        {gb:.1f} GB across {count} files")
        except json.JSONDecodeError:
            lines.append(f"  R2 total size:        (parse error)")

    # Spike watcher state
    watcher_out, _ = run_cmd("ps -eo pid,etime,comm 2>/dev/null | grep spike_watcher | grep -v grep | head -1")
    lines.append(f"  Spike watcher:        {watcher_out or '(not running — OK if --emit-spike-json false)'}")

    return "\n".join(lines)

# ─────────────────────────────────────────────────────────────────────
# Section 3: Per-target output review
# ─────────────────────────────────────────────────────────────────────
def report_target_review(ok_targets):
    lines = [section("3. PER-TARGET OUTPUT INTEGRITY REVIEW")]

    if not ok_targets:
        lines.append("  No completed targets to review yet.\n")
        return "\n".join(lines), []

    # Clear validation cache — always download fresh from R2
    import shutil
    if VALIDATION_CACHE.exists():
        shutil.rmtree(VALIDATION_CACHE)
    VALIDATION_CACHE.mkdir(parents=True, exist_ok=True)

    reviewed = []
    issues = []

    # Check ALL completed targets, not a capped subset
    targets_to_check = ok_targets
    lines.append(f"  Reviewing ALL {len(targets_to_check)} completed targets (fresh download from R2)\n")

    for target in targets_to_check:
        target_issues = []

        # List files on R2
        r2_list, _ = run_cmd(f"rclone ls r2:{R2_BUCKET}/{R2_PREFIX}/{target}/ 2>/dev/null", timeout=15)
        r2_files = {}
        for fline in r2_list.split("\n"):
            fline = fline.strip()
            if not fline:
                continue
            parts = fline.split(None, 1)
            if len(parts) == 2:
                r2_files[parts[1]] = int(parts[0])

        if not r2_files:
            target_issues.append("NO_FILES_ON_R2")
            issues.append((target, target_issues))
            continue

        # Check required files
        for req in REQUIRED_FILES:
            found = any(fname.endswith(req) for fname in r2_files)
            if not found:
                target_issues.append(f"MISSING: *{req}")

        # Check spike_events file size: .parquet (preferred) or .arrow (legacy)
        spike_files = {k: v for k, v in r2_files.items()
                       if k.endswith(".spike_events.parquet") or k.endswith(".spike_events.arrow")}
        if spike_files:
            # Floor check only: catch truly empty/corrupt files (10 KB).
            # With --multi-stream 4 --spike-percentile 95, sub-5MB parquets
            # are expected for low-activity pockets — not an integrity issue.
            for fname, size in spike_files.items():
                if size < 10_000:
                    target_issues.append(f"SPIKE_EMPTY: {fname} = {size} bytes (likely corrupt)")
        else:
            target_issues.append("MISSING: spike_events.parquet (or .arrow)")

        # Check binding_sites.json has sites
        bs_files = [k for k in r2_files if k.endswith(".binding_sites.json")]
        if bs_files:
            local_bs = VALIDATION_CACHE / f"{target}_binding_sites.json"
            run_cmd(f"rclone copy r2:{R2_BUCKET}/{R2_PREFIX}/{target}/{bs_files[0]} {VALIDATION_CACHE}/ 2>/dev/null", timeout=30)
            if (VALIDATION_CACHE / bs_files[0]).exists():
                local_bs = VALIDATION_CACHE / bs_files[0]
            if local_bs.exists():
                try:
                    with open(local_bs) as f:
                        bs = json.load(f)
                    sites = bs.get("sites", [])
                    if not sites:
                        target_issues.append("BINDING_SITES_EMPTY: 0 sites")
                except (json.JSONDecodeError, OSError) as e:
                    target_issues.append(f"BINDING_SITES_CORRUPT: {e}")

        # Check gcpid_synergy.json
        gcpid_files = [k for k in r2_files if k.endswith(".gcpid_synergy.json")]
        if gcpid_files:
            run_cmd(f"rclone copy r2:{R2_BUCKET}/{R2_PREFIX}/{target}/{gcpid_files[0]} {VALIDATION_CACHE}/ 2>/dev/null", timeout=30)
            gcpid_local = VALIDATION_CACHE / gcpid_files[0]
            if gcpid_local.exists():
                try:
                    with open(gcpid_local) as f:
                        gd = json.load(f)
                    residues = gd.get("residues", [])
                    if not residues:
                        target_issues.append("GCPID_EMPTY: 0 residues")
                except (json.JSONDecodeError, OSError) as e:
                    target_issues.append(f"GCPID_CORRUPT: {e}")

        # Check run.log for POSTFLIGHT: PASS
        run_logs = [k for k in r2_files if k == "run.log"]
        if run_logs:
            run_cmd(f"rclone copy r2:{R2_BUCKET}/{R2_PREFIX}/{target}/run.log {VALIDATION_CACHE}/ 2>/dev/null", timeout=30)
            rl = VALIDATION_CACHE / "run.log"
            if rl.exists():
                content = rl.read_text(errors="ignore")
                if "POSTFLIGHT: PASS" not in content:
                    if "POSTFLIGHT: FAIL" in content:
                        target_issues.append("POSTFLIGHT_FAILED")
                    else:
                        target_issues.append("POSTFLIGHT_MISSING (run.log incomplete?)")

        total_bytes = sum(r2_files.values())
        status = "PASS" if not target_issues else "ISSUES"
        reviewed.append({
            "target": target,
            "n_files": len(r2_files),
            "total_mb": total_bytes / 1e6,
            "issues": target_issues,
            "status": status,
        })

        if target_issues:
            issues.append((target, target_issues))

    # Summary table
    lines.append(f"  {'target':<20} {'files':>5} {'MB':>8} {'status':<8} {'issues'}")
    for r in reviewed:
        issue_str = "; ".join(r["issues"]) if r["issues"] else "-"
        lines.append(f"  {r['target']:<20} {r['n_files']:>5} {r['total_mb']:>8.0f} {r['status']:<8} {issue_str}")

    n_pass = sum(1 for r in reviewed if r["status"] == "PASS")
    lines.append(f"\n  Reviewed: {len(reviewed)} | PASS: {n_pass} | ISSUES: {len(issues)}")

    return "\n".join(lines), reviewed

# ─────────────────────────────────────────────────────────────────────
# Section 4: DCC re-validation
# ─────────────────────────────────────────────────────────────────────
def _euclid(a, b):
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))

def _grade(d):
    if d < 5: return "EXCELLENT"
    if d < 8: return "GOOD"
    if d < 10: return "MARGINAL"
    return "POOR"

def report_dcc_validation(ok_targets):
    lines = [section("4. DCC DETECTION QUALITY RE-VALIDATION")]

    if not ok_targets:
        lines.append("  No completed targets to validate.\n")
        return "\n".join(lines), []

    # Cache already cleared by Section 3 — files are fresh from R2
    VALIDATION_CACHE.mkdir(parents=True, exist_ok=True)
    dcc_results = []

    # Validate ALL completed targets, not a subset
    targets_to_check = ok_targets
    lines.append(f"  Validating ALL {len(targets_to_check)} completed targets\n")

    for target in targets_to_check:
        # Download ground truth sidecar
        gt_name = f"{target}_ground_truth.json"
        run_cmd(f"rclone copy r2:{R2_BUCKET}/{R2_PREFIX}/{target}/{gt_name} {VALIDATION_CACHE}/ 2>/dev/null", timeout=15)
        gt_path = VALIDATION_CACHE / gt_name
        if not gt_path.exists():
            continue
        try:
            with open(gt_path) as f:
                gt = json.load(f)
        except (json.JSONDecodeError, OSError):
            continue

        valid = gt.get("valid_for_dcc_validation", False)
        pdb_id = gt.get("pdb_id", "?")
        skip_reason = gt.get("skip_reason")

        if not valid:
            dcc_results.append({
                "target": target,
                "pdb_id": pdb_id,
                "valid": False,
                "skip_reason": skip_reason,
            })
            continue

        ligand = gt.get("ligand", {})
        lig_centroid = gt.get("ligand_centroid")
        if not lig_centroid:
            continue

        # Download binding_sites.json
        bs_candidates = [f"{target}.binding_sites.json"]
        for bs_name in bs_candidates:
            run_cmd(f"rclone copy r2:{R2_BUCKET}/{R2_PREFIX}/{target}/{bs_name} {VALIDATION_CACHE}/ 2>/dev/null", timeout=15)
        bs_path = VALIDATION_CACHE / bs_candidates[0]
        if not bs_path.exists():
            continue
        try:
            with open(bs_path) as f:
                bs = json.load(f)
        except (json.JSONDecodeError, OSError):
            continue

        sites = bs.get("sites", [])
        if not sites:
            dcc_results.append({
                "target": target, "pdb_id": pdb_id, "valid": True,
                "ligand": ligand.get("resname", "?"),
                "n_sites": 0, "best_dcc": None, "rank1_dcc": None,
                "grade": "NO_SITES",
            })
            continue

        # Compute DCC independently
        enriched = []
        for i, site in enumerate(sites):
            sc = site.get("centroid")
            if sc is None:
                continue
            if isinstance(sc, dict):
                sc = (sc.get("x", 0), sc.get("y", 0), sc.get("z", 0))
            else:
                sc = tuple(sc[:3])
            rank = site.get("rank", i + 1)
            d = _euclid(sc, lig_centroid)
            enriched.append((rank, d))

        if not enriched:
            continue

        by_rank = sorted(enriched, key=lambda r: r[0])
        by_dcc = sorted(enriched, key=lambda r: r[1])
        best = by_dcc[0]
        rank1 = by_rank[0]

        dcc_results.append({
            "target": target, "pdb_id": pdb_id, "valid": True,
            "ligand": ligand.get("resname", "?"),
            "n_atoms": ligand.get("n_atoms", 0),
            "n_sites": len(enriched),
            "best_dcc": round(best[1], 2),
            "best_rank": best[0],
            "rank1_dcc": round(rank1[1], 2),
            "grade": _grade(best[1]),
        })

    # Report
    valid_results = [r for r in dcc_results if r.get("valid")]
    skip_results = [r for r in dcc_results if not r.get("valid")]

    if valid_results:
        lines.append(f"  {'target':<20} {'ligand':<6} {'atoms':>5} {'best_DCC':>9} {'rank':>5} {'rank1_DCC':>10} {'grade':<10}")
        for r in valid_results:
            if r.get("best_dcc") is not None:
                lines.append(
                    f"  {r['target']:<20} {r['ligand']:<6} {r.get('n_atoms',0):>5} "
                    f"{r['best_dcc']:>8.2f}A {r.get('best_rank','?'):>5} "
                    f"{r['rank1_dcc']:>9.2f}A {r['grade']:<10}"
                )
            else:
                lines.append(f"  {r['target']:<20} {r['ligand']:<6} — NO_SITES")

        best_dccs = [r["best_dcc"] for r in valid_results if r.get("best_dcc") is not None]
        if best_dccs:
            median = sorted(best_dccs)[len(best_dccs) // 2]
            n_exc = sum(1 for d in best_dccs if d < 5)
            n_good = sum(1 for d in best_dccs if 5 <= d < 8)
            n_marg = sum(1 for d in best_dccs if 8 <= d < 10)
            n_poor = sum(1 for d in best_dccs if d >= 10)
            lines.append(f"\n  DCC distribution (n={len(best_dccs)}):")
            lines.append(f"    Median best-DCC: {median:.2f}A")
            lines.append(f"    EXCELLENT (<5A): {n_exc}")
            lines.append(f"    GOOD (5-8A):     {n_good}")
            lines.append(f"    MARGINAL (8-10A): {n_marg}")
            lines.append(f"    POOR (>10A):     {n_poor}")
            pass_rate = (n_exc + n_good) / len(best_dccs) if best_dccs else 0
            lines.append(f"    Pass rate:       {pass_rate:.0%}")

    if skip_results:
        lines.append(f"\n  Skipped (no valid ground truth): {len(skip_results)}")
        for r in skip_results[:10]:
            lines.append(f"    {r['target']}: {r.get('skip_reason', '?')}")

    if not valid_results and not skip_results:
        lines.append("  No DCC data available yet.\n")

    return "\n".join(lines), dcc_results

# ─────────────────────────────────────────────────────────────────────
# Section 5: Failure analysis
# ─────────────────────────────────────────────────────────────────────
def report_failures(fail_targets):
    lines = [section("5. FAILURE ANALYSIS")]

    if not fail_targets:
        lines.append("  No failures recorded.\n")
        return "\n".join(lines)

    # Group by PDB family for pattern detection
    families = {}
    for r in fail_targets:
        pdb = r["target"].split("_")[0]
        families.setdefault(pdb, []).append(r["target"])

    pandda_count = sum(len(v) for k, v in families.items() if k.startswith("13s"))
    other_count = len(fail_targets) - pandda_count

    lines.append(f"  Total failures:    {len(fail_targets)}")
    lines.append(f"    13s* PanDDA:     {pandda_count}  (CUDA deadlock on RING domain homo-oligomers)")
    lines.append(f"    Other:           {other_count}  {'← INVESTIGATE' if other_count > 0 else '(none — engine healthy outside PanDDA block)'}")
    lines.append(f"  Distinct PDBs:     {len(families)}")
    lines.append(f"  All rc=124 (timeout): {all('rc=124' in r['raw'] for r in fail_targets)}\n")

    lines.append(f"  Per-family breakdown:")
    for pdb, targets in sorted(families.items()):
        chains = [t.split("_chain")[-1] if "_chain" in t else "?" for t in targets]
        lines.append(f"    {pdb}: chains {','.join(chains)} hung ({len(targets)} timeout{'s' if len(targets)>1 else ''})")

    if other_count > 0:
        lines.append(f"\n  Non-PanDDA failures (need investigation):")
        for r in fail_targets:
            if not r["target"].split("_")[0].startswith("13s"):
                lines.append(f"    {r['target']}: {r['raw']}")

    return "\n".join(lines)

# ─────────────────────────────────────────────────────────────────────
# Section 6: Residue ID alignment spot-check
# ─────────────────────────────────────────────────────────────────────
def report_residue_alignment(ok_targets):
    lines = [section("6. RESIDUE ID ALIGNMENT SPOT-CHECK")]

    if not ok_targets:
        lines.append("  No completed targets to check.\n")
        return "\n".join(lines)

    # Pick the most recent completed target
    target = ok_targets[-1]
    lines.append(f"  Spot-checking: {target}\n")

    # Download topology + binding_sites
    VALIDATION_CACHE.mkdir(parents=True, exist_ok=True)
    pdb = target.split("_")[0]
    topo_name = f"{target}.topology.json"
    bs_name = f"{target}.binding_sites.json"

    run_cmd(f"rclone copy r2:{R2_BUCKET}/{R2_PREFIX}/{target}/{bs_name} {VALIDATION_CACHE}/ 2>/dev/null", timeout=15)
    # Topology is in the input prefix, not output
    run_cmd(f"rclone copy r2:{R2_BUCKET}/proteome_1000/{pdb}/{topo_name} {VALIDATION_CACHE}/ 2>/dev/null", timeout=15)

    topo_path = VALIDATION_CACHE / topo_name
    bs_path = VALIDATION_CACHE / bs_name

    if not topo_path.exists() or not bs_path.exists():
        lines.append(f"  Could not download topology or binding_sites for {target}")
        return "\n".join(lines)

    try:
        with open(topo_path) as f:
            topo = json.load(f)
        with open(bs_path) as f:
            bs = json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        lines.append(f"  Parse error: {e}")
        return "\n".join(lines)

    topo_resids = set(topo.get("residue_ids", []))
    topo_resnames = topo.get("residue_names", [])
    n_topo_res = topo.get("n_residues", len(topo_resnames))

    sites = bs.get("sites", [])
    all_lining_resids = set()
    for site in sites:
        for lr in site.get("lining_residues", []):
            rid = lr.get("resid", lr.get("residue_id"))
            if rid is not None:
                all_lining_resids.add(rid)

    if topo_resids and all_lining_resids:
        orphan = all_lining_resids - topo_resids
        overlap = all_lining_resids & topo_resids
        lines.append(f"  Topology residue IDs:       {len(topo_resids)} (range {min(topo_resids)}-{max(topo_resids)})")
        lines.append(f"  Engine lining residue IDs:   {len(all_lining_resids)}")
        lines.append(f"  Overlap:                     {len(overlap)} ({100*len(overlap)/max(len(all_lining_resids),1):.0f}%)")
        if orphan:
            lines.append(f"  ORPHAN (engine IDs not in topo): {len(orphan)} — {sorted(list(orphan))[:10]}")
            lines.append(f"  STATUS: ALIGNMENT ISSUE — investigate residue ID mapping")
        else:
            lines.append(f"  STATUS: ALIGNED — all engine lining IDs exist in topology")
    else:
        lines.append(f"  Topology n_residues: {n_topo_res}")
        lines.append(f"  Lining residues from engine: {len(all_lining_resids)}")
        lines.append(f"  STATUS: INCOMPLETE — missing residue_ids in topology or no lining_residues")

    return "\n".join(lines)

# ─────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="PRISM-4D Corpus Status Report")
    parser.add_argument("--output", default=None, help="Output file path (default: ~/Desktop/prism_corpus_status_<timestamp>.txt)")
    args = parser.parse_args()

    now = datetime.now()
    ts = now.strftime("%Y%m%d_%H%M%S")

    if args.output:
        out_path = Path(args.output)
    else:
        out_path = Path.home() / "Desktop" / f"prism_corpus_status_{ts}.txt"

    report = []
    report.append("=" * 70)
    report.append(f"  PRISM-4D CORPUS GENERATION STATUS REPORT")
    report.append(f"  Generated: {now.strftime('%Y-%m-%d %H:%M:%S %Z')}")
    report.append(f"  Campaign:  proteome_1000 TWIN multi-differential")
    report.append("=" * 70)

    # Section 1
    s1, n_completed, ok_targets, fail_targets = report_run_state()
    report.append(s1)

    # Section 2
    report.append(report_r2_health())

    # Section 3
    s3, reviewed = report_target_review(ok_targets)
    report.append(s3)

    # Section 4
    s4, dcc_results = report_dcc_validation(ok_targets)
    report.append(s4)

    # Section 5
    report.append(report_failures(fail_targets))

    # Section 6
    report.append(report_residue_alignment(ok_targets))

    # Section 7: Health Score + Verdict
    report.append(section("7. HEALTH SCORE & VERDICT"))

    # Compute health dimensions
    fail_rate = (len(fail_targets) / n_completed * 100) if n_completed > 0 else 0
    n_13s_fail = sum(1 for r in fail_targets if r["target"].startswith("13s"))
    non_13s_fail_rate = ((len(fail_targets) - n_13s_fail) / max(n_completed - n_13s_fail - sum(1 for t in ok_targets if t.startswith("13s")), 1) * 100) if n_completed > 0 else 0

    valid_dcc = [r for r in dcc_results if r.get("valid") and r.get("best_dcc") is not None]
    dcc_pass_rate = (sum(1 for r in valid_dcc if r["best_dcc"] < 8) / len(valid_dcc)) if valid_dcc else None
    dcc_n = len(valid_dcc)
    dcc_stale = dcc_n < 10 and n_completed > 20

    # Color-coded status per dimension
    def color(val, green_thresh, yellow_thresh):
        if val is None: return "GRAY (insufficient data)"
        if val <= green_thresh: return "GREEN"
        if val <= yellow_thresh: return "YELLOW"
        return "RED"

    report.append(f"  ┌─────────────────────────────────────────────────────┐")
    report.append(f"  │  DIMENSION             STATUS    VALUE              │")
    report.append(f"  ├─────────────────────────────────────────────────────┤")

    # Overall failure rate
    fail_color = color(fail_rate, 10, 25)
    report.append(f"  │  Overall fail rate      {fail_color:<9} {fail_rate:.0f}%  ({len(fail_targets)}/{n_completed})    │")

    # Non-pathological failure rate (excluding known 13s* hangs)
    non13s_color = color(non_13s_fail_rate, 5, 15)
    report.append(f"  │  Non-13s* fail rate     {non13s_color:<9} {non_13s_fail_rate:.0f}%                    │")

    # DCC quality
    if dcc_pass_rate is not None:
        dcc_color = "GREEN" if dcc_pass_rate >= 0.6 else ("YELLOW" if dcc_pass_rate >= 0.3 else "RED")
    else:
        dcc_color = "GRAY"
    dcc_str = f"{dcc_pass_rate:.0%} (n={dcc_n})" if dcc_pass_rate is not None else f"n={dcc_n} (insufficient)"
    report.append(f"  │  DCC pass rate (<8A)    {dcc_color:<9} {dcc_str:<22}│")

    # DCC data freshness
    fresh_color = "YELLOW" if dcc_stale else "GREEN"
    fresh_str = f"STALE (n={dcc_n} from {n_completed} targets)" if dcc_stale else f"n={dcc_n}"
    report.append(f"  │  DCC data freshness     {fresh_color:<9} {fresh_str:<22}│")

    # Output integrity
    n_integrity_pass = sum(1 for r in reviewed if r["status"] == "PASS") if reviewed else 0
    n_integrity_total = len(reviewed) if reviewed else 0
    integ_color = "GREEN" if n_integrity_pass == n_integrity_total and n_integrity_total > 0 else "YELLOW"
    report.append(f"  │  Output integrity       {integ_color:<9} {n_integrity_pass}/{n_integrity_total} PASS            │")

    report.append(f"  └─────────────────────────────────────────────────────┘")

    # Overall verdict
    reds = sum(1 for c in [fail_color, non13s_color, dcc_color] if c == "RED")
    yellows = sum(1 for c in [fail_color, non13s_color, dcc_color, fresh_color] if c == "YELLOW")

    if n_completed == 0:
        report.append(f"\n  VERDICT: EARLY — no targets completed yet.")
    elif reds >= 2:
        report.append(f"\n  VERDICT: *** ABORT RECOMMENDED *** — {reds} RED dimensions.")
    elif reds == 1:
        report.append(f"\n  VERDICT: CAUTION — 1 RED dimension. Investigate before continuing.")
    elif yellows >= 2:
        report.append(f"\n  VERDICT: MONITOR — {yellows} YELLOW dimensions. Proceeding but watch closely.")
    elif non_13s_fail_rate == 0 and (dcc_pass_rate is None or dcc_pass_rate >= 0.5):
        report.append(f"\n  VERDICT: PROCEED — non-pathological failure rate 0%, engine healthy.")
        if dcc_stale:
            report.append(f"  NOTE: DCC data stale (n={dcc_n} from {n_completed} completed). New valid-GT")
            report.append(f"        targets expected as corpus moves past PanDDA block.")
    else:
        report.append(f"\n  VERDICT: PROCEED WITH MONITORING.")

    report.append(f"\n  Report saved to: {out_path}")
    report.append(f"  Next report: ~1 hour")

    # Write report
    full_report = "\n".join(report)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(full_report)
    print(full_report)
    return 0


if __name__ == "__main__":
    sys.exit(main())
