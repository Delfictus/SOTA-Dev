#!/usr/bin/env python3
"""R2 staging + extraction for all pct95 targets.

Pulls required engine artifacts per-target from r2:prism-archive/10k-runs/,
runs extract_all_features.extract_one(), then cleans the staging dir to
keep disk bounded.

Bandwidth cap protects the pct70 campaign which is actively uploading to R2
on the same workstation. Default cap 20M/s; reduce with --bwlimit.

Reuses already-extracted .npz files in --out-dir when --skip-existing is set.

Usage:
    python3 scripts/training/extract_from_r2.py \
        --out-dir /mnt/storage/spike-audit/features-pct95 \
        --stage-dir /mnt/storage/spike-audit/r2stage \
        --bwlimit 20M \
        --skip-existing

    # Targeted subset:
    python3 scripts/training/extract_from_r2.py \
        --targets 10dc_chainA,10dj_chainA \
        --out-dir /mnt/storage/spike-audit/features-pct95
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional

# Local imports
sys.path.insert(0, str(Path(__file__).parent))
from extract_all_features import extract_one  # type: ignore

R2_PREFIX_DEFAULT = "r2:prism-archive/10k-runs"
D1_WORKER_URL = os.environ.get("PRISM_API",
                               "https://prism-feature-pipeline.is-0b9.workers.dev")

STAGE_FILES = [
    "{target}.topology.json",
    "{target}_clean.pdb",
    "{target}.binding_sites.json",
    "{target}.kcc_visualization.json",
    "{target}.topology.prism_therm.json",
    "{target}_ground_truth.json",
    # ensemble trajectory (Block 7) — one per stream, take stream00
    "{target}_stream00.ensemble_trajectory.pdb",
]

# Parquets are a pattern match
PARQUET_GLOB = "*.spike_events.parquet"


def rclone_copy(remote_path: str, local_dir: Path,
                bwlimit: str, timeout: int = 300) -> bool:
    local_dir.mkdir(parents=True, exist_ok=True)
    cmd = ["rclone", "copy", remote_path, str(local_dir),
           "--quiet", "--transfers", "4"]
    if bwlimit:
        cmd += ["--bwlimit", bwlimit]
    try:
        r = subprocess.run(cmd, capture_output=True, timeout=timeout, check=False)
        return r.returncode == 0
    except subprocess.TimeoutExpired:
        return False


def rclone_copy_glob(remote_prefix: str, local_dir: Path,
                     include: str, bwlimit: str, timeout: int = 900) -> bool:
    local_dir.mkdir(parents=True, exist_ok=True)
    cmd = ["rclone", "copy", remote_prefix, str(local_dir),
           "--include", include, "--quiet", "--transfers", "8"]
    if bwlimit:
        cmd += ["--bwlimit", bwlimit]
    try:
        r = subprocess.run(cmd, capture_output=True, timeout=timeout, check=False)
        return r.returncode == 0
    except subprocess.TimeoutExpired:
        return False


def fetch_pct95_targets() -> List[str]:
    req = urllib.request.Request(
        f"{D1_WORKER_URL}/targets?spike_percentile=95",
        headers={"User-Agent": "Mozilla/5.0 prism4d-extract-r2"},
    )
    with urllib.request.urlopen(req, timeout=60) as r:
        data = json.loads(r.read())
    return sorted([t["target"] for t in data.get("targets", [])])


def stage_target(target: str, r2_prefix: str, stage_dir: Path,
                 bwlimit: str) -> Optional[Path]:
    """Download required files for one target. Returns the staging dir."""
    tgt_dir = stage_dir / target
    tgt_dir.mkdir(parents=True, exist_ok=True)

    r2_target_prefix = f"{r2_prefix}/{target}"
    # Pull explicit files
    for tmpl in STAGE_FILES:
        fname = tmpl.format(target=target)
        local = tgt_dir / fname
        if local.exists() and local.stat().st_size > 0:
            continue
        rclone_copy(f"{r2_target_prefix}/{fname}", tgt_dir, bwlimit, timeout=120)

    # Parquets (all of them — Block 6 iterates all)
    rclone_copy_glob(f"{r2_target_prefix}/", tgt_dir, PARQUET_GLOB,
                     bwlimit, timeout=600)

    return tgt_dir


def purge_stage(stage_dir: Path, target: str) -> None:
    tgt_dir = stage_dir / target
    if tgt_dir.exists():
        shutil.rmtree(tgt_dir, ignore_errors=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, required=True,
                        help="Where .features.npz files land")
    parser.add_argument("--stage-dir", type=Path,
                        default=Path("/mnt/storage/spike-audit/r2stage"),
                        help="Temp R2 staging dir (purged per-target)")
    parser.add_argument("--targets", default="",
                        help="Comma-separated subset (skips D1 query)")
    parser.add_argument("--max", type=int, default=0,
                        help="Cap # of targets processed (0 = all)")
    parser.add_argument("--skip-existing", action="store_true",
                        help="Skip targets with existing .features.npz")
    parser.add_argument("--bwlimit", default="20M",
                        help="rclone --bwlimit (default 20M to protect "
                             "pct70 campaign uploads)")
    parser.add_argument("--r2-prefix", default=R2_PREFIX_DEFAULT)
    parser.add_argument("--keep-stage", action="store_true",
                        help="Do not purge per-target stage dir after extraction")
    parser.add_argument("--nma-dirs", nargs="*", type=Path,
                        default=[Path("/mnt/storage/prism-outputs/ml/egnn/cryptobench"),
                                 Path("/mnt/storage/prism-outputs/ml/egnn")])
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.stage_dir.mkdir(parents=True, exist_ok=True)

    # Resolve target list
    if args.targets:
        targets = args.targets.split(",")
    else:
        targets = fetch_pct95_targets()
    if args.max > 0:
        targets = targets[:args.max]
    print(f"Targets to process: {len(targets)} (bwlimit={args.bwlimit})")

    manifest: List[Dict[str, Any]] = []
    t0 = time.time()
    n_ok = n_skip = n_fail = 0

    for i, target in enumerate(targets):
        out_path = args.out_dir / f"{target}_features.npz"
        if args.skip_existing and out_path.exists() and out_path.stat().st_size > 10_000:
            n_skip += 1
            manifest.append({"target": target, "status": "SKIPPED", "reason": "existing"})
            if (i + 1) % 25 == 0:
                print(f"  [{i+1}/{len(targets)}] skipped (existing)", flush=True)
            continue

        # Stage
        t_stage = time.time()
        stage_tgt = stage_target(target, args.r2_prefix, args.stage_dir, args.bwlimit)
        stage_sec = time.time() - t_stage

        # Extract
        try:
            status = extract_one(target, stage_tgt, args.nma_dirs, args.out_dir)
        except Exception as e:
            status = {"target": target, "ok": False, "error": str(e)}

        ext_sec = status.get("elapsed_sec", 0)
        status["stage_sec"] = round(stage_sec, 1)

        if status.get("ok"):
            n_ok += 1
        else:
            n_fail += 1
        manifest.append(status)

        if not args.keep_stage:
            purge_stage(args.stage_dir, target)

        elapsed = time.time() - t0
        eta = elapsed / max(n_ok + n_fail, 1) * (len(targets) - i - 1)
        tag = "OK" if status.get("ok") else f"FAIL ({status.get('error', '?')[:50]})"
        print(f"  [{i+1}/{len(targets)}] {target:20s} {tag}  "
              f"stage={stage_sec:5.1f}s extract={ext_sec:5.1f}s  "
              f"elapsed={elapsed/60:.1f}m ETA={eta/60:.0f}m", flush=True)

    manifest_path = args.out_dir / "extraction_r2_manifest.json"
    manifest_path.write_text(json.dumps({
        "total": len(targets),
        "ok": n_ok, "skipped": n_skip, "failed": n_fail,
        "bwlimit": args.bwlimit,
        "results": manifest,
    }, indent=2, default=str))
    print(f"\n  DONE: ok={n_ok} skipped={n_skip} failed={n_fail}  "
          f"time={(time.time()-t0)/60:.1f}m")
    print(f"  Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
