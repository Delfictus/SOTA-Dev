#!/usr/bin/env python3
"""Lightweight second-pass: add temporal features to existing .npz bundles.

For every .npz produced by the first extraction pass:

  1. Pull that target's per-site spike parquets from R2 (small — only
     the parquet files, not the JSONs or other engine outputs).
  2. Compute per-residue phase_transition_ratio + warm_hold_spike_fraction
     using 5 Å Cα radius (same logic as extract_all_features.py).
  3. Compute per-site aggregates from each parquet.
  4. Append `temporal` key to the .npz and re-save.
  5. POST per-site aggregates to the Worker at
     /site-features/{target}/temporal so D1 gets populated.

This avoids re-running the full 216-dim extraction and uses only the
spike parquets (small vs the full engine artifact set).

Usage:
    python3 scripts/training/add_temporal_to_npz.py \\
        --bundle-dir /mnt/storage/spike-audit/features-pct95 \\
        --r2-prefix r2:prism-archive/10k-runs \\
        --worker-url https://prism-feature-pipeline.is-0b9.workers.dev \\
        --bwlimit 20M

    # Skip D1 push (local-only update):
    python3 scripts/training/add_temporal_to_npz.py --no-d1-push ...

The target is inferred from the .npz filename. Spike-parquet filenames are
matched as `<target>.site<ID>.spike_events.parquet`.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from extract_all_features import compute_temporal_features  # reuse


def rclone_parquets(r2_target_prefix: str, dst_dir: Path,
                    bwlimit: str) -> bool:
    dst_dir.mkdir(parents=True, exist_ok=True)
    cmd = ["rclone", "copy", r2_target_prefix, str(dst_dir),
           "--include", "*.spike_events.parquet",
           "--transfers", "8", "--quiet"]
    if bwlimit:
        cmd += ["--bwlimit", bwlimit]
    r = subprocess.run(cmd, capture_output=True, timeout=600, check=False)
    return r.returncode == 0


def post_temporal(worker_url: str, target: str,
                  per_site: Dict[str, Dict[str, float]]) -> bool:
    """POST per-site temporal aggregates to the Worker's update endpoint."""
    payload = []
    for site_name, vals in per_site.items():
        p = vals.get("phase_transition_ratio")
        w = vals.get("warm_hold_spike_fraction")
        payload.append({
            "site_name": site_name,
            "phase_transition_ratio": None if not np.isfinite(p or 0) or p is None else float(p),
            "warm_hold_spike_fraction": None if not np.isfinite(w or 0) or w is None else float(w),
        })
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        f"{worker_url}/site-features/{target}/temporal",
        data=body,
        headers={"Content-Type": "application/json",
                 "User-Agent": "Mozilla/5.0 prism4d-temporal-pass"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            r.read()
        return True
    except Exception:
        return False


def enrich_one(npz_path: Path, r2_prefix: str, worker_url: Optional[str],
               bwlimit: str, keep_stage: bool = False) -> Dict:
    target = npz_path.stem.replace("_features", "")
    status: Dict = {"target": target, "ok": False}

    # Already has temporal? Skip.
    try:
        d = np.load(npz_path, allow_pickle=False)
    except Exception as e:
        status["error"] = f"load failed: {e}"
        return status
    if "temporal" in d.files:
        status["ok"] = True
        status["skipped_reason"] = "already has temporal"
        return status

    # Need coords + residue_ids to compute residue-level features
    coords = d["coords"] if "coords" in d.files else d.get("ca_coords")
    if coords is None:
        status["error"] = "no coords in bundle"
        return status
    residue_ids = d["residue_ids"].tolist() if "residue_ids" in d.files else list(range(coords.shape[0]))

    # Stage parquets
    stage = Path(tempfile.mkdtemp(prefix=f"temporal_{target}_"))
    try:
        ok = rclone_parquets(f"{r2_prefix}/{target}/", stage, bwlimit)
        if not ok:
            status["error"] = "rclone pull failed"
            return status

        # Grab site metadata: prefer to build minimal sites list from the
        # existing site_features_json in the bundle (so we know which site
        # IDs exist). Fall back to glob if the bundle lacks it.
        bs_sites: List[Dict] = []
        if "site_features_json" in d.files:
            try:
                sf = json.loads(str(d["site_features_json"]))
                for name in sf.keys():
                    if name.startswith("site"):
                        sid = name[4:]
                        try:
                            sid = int(sid)
                        except ValueError:
                            pass
                        bs_sites.append({"id": sid})
            except Exception:
                pass
        if not bs_sites:
            # Fall back: scan parquet filenames
            for pf in stage.glob("*.spike_events.parquet"):
                try:
                    sid_part = pf.name.split(".site", 1)[1].split(".spike_events", 1)[0]
                    bs_sites.append({"id": int(sid_part)})
                except (IndexError, ValueError):
                    continue

        t0 = time.time()
        per_res, per_site = compute_temporal_features(
            stage, coords.astype(np.float32), residue_ids, bs_sites,
        )
        compute_sec = time.time() - t0

        # Re-save npz with temporal key added
        existing = {k: d[k] for k in d.files}
        existing["temporal"] = per_res
        # Update site_features_json to include temporal per-site merged
        if "site_features_json" in existing:
            try:
                sf = json.loads(str(existing["site_features_json"]))
                for name, vals in per_site.items():
                    if name in sf:
                        sf[name].update(vals)
                    else:
                        sf[name] = dict(vals)
                existing["site_features_json"] = np.asarray(
                    json.dumps(sf, default=str))
            except Exception:
                pass
        np.savez_compressed(npz_path, **existing)

        status.update({
            "ok": True,
            "n_res": int(per_res.shape[0]),
            "n_res_with_spikes": int((per_res[:, 1] > 0).sum()),
            "n_sites": len(per_site),
            "compute_sec": round(compute_sec, 2),
        })

        # D1 push
        if worker_url and per_site:
            pushed = post_temporal(worker_url, target, per_site)
            status["d1_pushed"] = pushed
        return status
    finally:
        if not keep_stage:
            shutil.rmtree(stage, ignore_errors=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle-dir", type=Path, required=True)
    parser.add_argument("--r2-prefix", default="r2:prism-archive/10k-runs")
    parser.add_argument("--worker-url",
                        default="https://prism-feature-pipeline.is-0b9.workers.dev")
    parser.add_argument("--bwlimit", default="20M")
    parser.add_argument("--no-d1-push", action="store_true")
    parser.add_argument("--max", type=int, default=0, help="Cap # targets")
    parser.add_argument("--targets", default="", help="CSV subset")
    args = parser.parse_args()

    worker = None if args.no_d1_push else args.worker_url

    npzs = sorted(args.bundle_dir.glob("*_features.npz"))
    if args.targets:
        wanted = set(args.targets.split(","))
        npzs = [p for p in npzs if p.stem.replace("_features", "") in wanted]
    if args.max:
        npzs = npzs[:args.max]
    print(f"Enrichment targets: {len(npzs)}  bwlimit={args.bwlimit}  d1_push={worker is not None}")

    results = []
    t0 = time.time()
    n_ok = n_skip = n_fail = 0
    for i, p in enumerate(npzs):
        status = enrich_one(p, args.r2_prefix, worker, args.bwlimit)
        if status.get("skipped_reason"):
            n_skip += 1
        elif status.get("ok"):
            n_ok += 1
        else:
            n_fail += 1
        results.append(status)
        elapsed = time.time() - t0
        eta = elapsed / max(i + 1, 1) * (len(npzs) - i - 1)
        tag = "OK" if status.get("ok") else f"FAIL ({status.get('error', '?')[:50]})"
        if status.get("skipped_reason"):
            tag = "SKIP"
        print(f"  [{i+1}/{len(npzs)}] {p.stem:40s} {tag}  "
              f"elapsed={elapsed/60:.1f}m ETA={eta/60:.0f}m", flush=True)

    manifest = {"total": len(npzs), "ok": n_ok, "skipped": n_skip,
                "failed": n_fail, "results": results}
    (args.bundle_dir / "temporal_enrichment_manifest.json").write_text(
        json.dumps(manifest, indent=2, default=str))
    print(f"\n  DONE: ok={n_ok} skipped={n_skip} failed={n_fail}  "
          f"time={(time.time()-t0)/60:.1f}m")


if __name__ == "__main__":
    main()
