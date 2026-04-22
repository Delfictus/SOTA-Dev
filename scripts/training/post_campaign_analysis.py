#!/usr/bin/env python3
"""Post-campaign analysis — runs when pct70 campaign finishes.

FAST PATH (default): for each target, read binding_sites.json from R2 (small,
~300 KB each), compute **centroid-DCC** per site = |site.centroid - ligand|.
No per-site spike parquets downloaded. Total egress: ~100 MB for 372 targets.

DEEP PATH (optional, --deep-dcc <target>): for a specific target, download
per-site spike parquets and compute the stricter **spike-DCC** = min over
all spike coordinates. Used only for validation on ambiguous targets.

Rationale: binding_sites.json's per-site centroid IS spike-derived (the engine
clusters spikes and returns their centroid). For most sites, centroid-DCC
tracks spike-DCC within 1-2Å. Sites where centroid is pulled away from
ligand by a tail of distant spikes are the exception (rare — ~5% of sites).

Steps:
  1. Pull binding_sites.json + ground_truth.json for every pct70 target (~100 MB)
  2. For each site: compute centroid-DCC = |site.centroid - ligand_centroid|
  3. Best site per target = argmin centroid-DCC
  4. Grade: EXCELLENT (<5A) / GOOD / MARGINAL / POOR
  5. Generate D1 INSERTs for targets, corrected_dcc, site_features
  6. pct70 vs pct95 comparison report

Usage:
  python3 scripts/training/post_campaign_analysis.py \
      --r2-prefix 10k-runs-pct70 \
      --percentile 70

  # monitor mode — wait for campaign to finish:
  python3 scripts/training/post_campaign_analysis.py --monitor

  # deep-dive on one target using parquets:
  python3 scripts/training/post_campaign_analysis.py --deep-dcc 9ymg_chainC
"""
from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

DEFAULT_R2_PREFIX = "10k-runs-pct70"
DEFAULT_OUT = Path("/mnt/storage/spike-audit/pct70-analysis")
WORKSTATION_WRANGLER_CWD = "/home/diddy/Desktop/Prism4D-bio"
PCT95_DCC_PATH = "/mnt/storage/spike-audit/dcc-recompute/corrected_dcc_results.json"
MAX_WORKERS = 16


# ─────────────────────────────────────────────────────────────
#  Fast: centroid-DCC from binding_sites.json only
# ─────────────────────────────────────────────────────────────

def compute_target_centroid_dcc(target: str, r2_prefix: str, cache_dir: Path) -> Optional[Dict[str, Any]]:
    """Pull binding_sites.json + ground_truth.json only. Compute per-site
    centroid-DCC. No per-spike parquet downloads."""
    local = cache_dir / target
    local.mkdir(parents=True, exist_ok=True)
    try:
        # Pull both small files in one rclone pass with include patterns
        r = subprocess.run(
            ["rclone", "copy", f"r2:prism-archive/{r2_prefix}/{target}/",
             str(local),
             "--include", "*.binding_sites.json",
             "--include", "*_ground_truth.json",
             "--transfers", "2", "--quiet"],
            capture_output=True, timeout=60, check=False,
        )

        gt_path = local / f"{target}_ground_truth.json"
        bs_path = local / f"{target}.binding_sites.json"
        if not gt_path.exists() or not bs_path.exists():
            return None

        gt = json.load(open(gt_path))
        if not gt.get("valid_for_dcc_validation"):
            return {"target": target, "skip_reason": gt.get("skip_reason", "not_valid")}
        lig = np.asarray(gt["ligand_centroid"], dtype=np.float32)
        if lig.shape != (3,):
            return None

        bs = json.load(open(bs_path))
        sites = bs.get("sites", [])

        site_dccs: List[Dict[str, Any]] = []
        for s in sites:
            sid = s.get("id", -1)
            centroid = s.get("centroid")
            if not centroid or len(centroid) != 3:
                continue
            c = np.asarray(centroid, dtype=np.float32)
            dcc = float(np.linalg.norm(c - lig))
            site_dccs.append({
                "site_name": f"site{sid}",
                "site_id": int(sid),
                "centroid": centroid,
                "centroid_dcc": dcc,
                "spike_count": int(s.get("spike_count", 0) or 0),
                "volume": float(s.get("volume", 0.0) or 0.0),
                "burial_score": float(s.get("burial_score", 0.0) or 0.0),
                "druggability": float(s.get("druggability", 0.0) or 0.0),
                "aromatic_score": float(s.get("aromatic_score", 0.0) or 0.0),
                "n_lining_residues": len(s.get("lining_residues", [])),
                "quality_score": float(s.get("quality_score", 0.0) or 0.0),
                "classification": s.get("classification", ""),
                "therm_class": s.get("therm_class", ""),
                "rank": int(s.get("rank", 0) or 0),
            })

        if not site_dccs:
            return None

        best = min(site_dccs, key=lambda s: s["centroid_dcc"])
        return {
            "target": target,
            "ligand": gt.get("ligand", {}).get("resname"),
            "ligand_centroid": lig.tolist(),
            "n_sites": len(site_dccs),
            "best_site": best["site_name"],
            "best_centroid_dcc": best["centroid_dcc"],
            "sites": site_dccs,
        }
    except Exception as e:
        return {"target": target, "error": f"{type(e).__name__}: {e}"}


# ─────────────────────────────────────────────────────────────
#  Deep: per-site spike-DCC from parquets (on-demand single target)
# ─────────────────────────────────────────────────────────────

def deep_spike_dcc(target: str, r2_prefix: str, cache_dir: Path) -> Optional[Dict[str, Any]]:
    """Download all per-site parquets for ONE target, compute true spike-DCC.
    Same output shape as centroid version but with spike_dcc instead of
    centroid_dcc + spike_coord_count."""
    try:
        import pyarrow.parquet as pq
    except ImportError:
        print("  pyarrow required for deep DCC. pip install pyarrow")
        return None

    local = cache_dir / f"{target}_deep"
    local.mkdir(parents=True, exist_ok=True)
    try:
        # Pull everything needed for deep analysis
        subprocess.run(
            ["rclone", "copy", f"r2:prism-archive/{r2_prefix}/{target}/",
             str(local),
             "--include", "*.binding_sites.json",
             "--include", "*_ground_truth.json",
             "--include", "*.spike_events.parquet",
             "--transfers", "8", "--quiet"],
            timeout=600, check=False,
        )

        gt_path = local / f"{target}_ground_truth.json"
        bs_path = local / f"{target}.binding_sites.json"
        if not gt_path.exists() or not bs_path.exists():
            return None

        gt = json.load(open(gt_path))
        lig = np.asarray(gt["ligand_centroid"], dtype=np.float32)

        bs = json.load(open(bs_path))
        sites_meta = {int(s.get("id", -1)): s for s in bs.get("sites", [])}

        site_dccs = []
        for pf in sorted(local.glob("*.spike_events.parquet")):
            # File name like "<target>.siteN.spike_events.parquet"
            site_name = pf.stem.replace(".spike_events", "").rsplit(".", 1)[-1]
            try:
                sid = int(site_name.replace("site", ""))
            except ValueError:
                sid = -1
            try:
                df = pq.read_table(pf).to_pandas()
                if not all(c in df.columns for c in ("x", "y", "z")):
                    continue
                coords = df[["x", "y", "z"]].to_numpy(dtype=np.float32)
                spike_dcc = float(np.linalg.norm(coords - lig, axis=1).min())
                intensities = df["intensity"].to_numpy() if "intensity" in df.columns else np.array([])
                unsat = float((intensities < 64.0).mean()) if intensities.size else 0.0
                bs_meta = sites_meta.get(sid, {})
                site_dccs.append({
                    "site_name": site_name,
                    "site_id": sid,
                    "centroid": bs_meta.get("centroid"),
                    "centroid_dcc": float(np.linalg.norm(
                        np.asarray(bs_meta.get("centroid", [0,0,0])) - lig
                    )) if bs_meta.get("centroid") else None,
                    "spike_dcc": spike_dcc,
                    "n_spikes": len(df),
                    "unsat_frac": unsat,
                    "spike_count": int(bs_meta.get("spike_count", 0) or 0),
                    "volume": float(bs_meta.get("volume", 0.0) or 0.0),
                    "burial_score": float(bs_meta.get("burial_score", 0.0) or 0.0),
                    "druggability": float(bs_meta.get("druggability", 0.0) or 0.0),
                    "aromatic_score": float(bs_meta.get("aromatic_score", 0.0) or 0.0),
                    "n_lining_residues": len(bs_meta.get("lining_residues", [])),
                    "quality_score": float(bs_meta.get("quality_score", 0.0) or 0.0),
                })
            except Exception:
                continue

        if not site_dccs:
            return None
        best = min(site_dccs, key=lambda s: s["spike_dcc"])
        return {
            "target": target,
            "ligand": gt.get("ligand", {}).get("resname"),
            "ligand_centroid": lig.tolist(),
            "n_sites": len(site_dccs),
            "best_site": best["site_name"],
            "best_spike_dcc": best["spike_dcc"],
            "best_centroid_dcc": best.get("centroid_dcc"),
            "sites": site_dccs,
        }
    finally:
        # Delete parquets — they're huge. Keep small metadata.
        for pf in local.glob("*.spike_events.parquet"):
            pf.unlink(missing_ok=True)


# ─────────────────────────────────────────────────────────────
#  D1 SQL generation (uses centroid-DCC as the "primary" dcc metric)
# ─────────────────────────────────────────────────────────────

def grade(dcc: float) -> str:
    if dcc < 5.0: return "EXCELLENT"
    if dcc < 8.0: return "GOOD"
    if dcc < 10.0: return "MARGINAL"
    return "POOR"


def generate_d1_updates(results: List[Dict[str, Any]], out_dir: Path,
                        percentile: int = 70) -> List[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)

    def sqle(s): return "NULL" if s is None else "'" + str(s).replace("'", "''") + "'"
    def sqln(n):
        if n is None or (isinstance(n, float) and (math.isnan(n) or math.isinf(n))):
            return "NULL"
        return str(n)

    target_stmts, dcc_stmts, site_stmts = [], [], []
    for r in results:
        if "error" in r or r.get("skip_reason"):
            continue
        tgt = r["target"]
        best = r.get("best_spike_dcc") if r.get("best_spike_dcc") is not None else r["best_centroid_dcc"]
        used_metric = "spike" if r.get("best_spike_dcc") is not None else "centroid"

        target_stmts.append(
            f"INSERT OR REPLACE INTO targets (target, pdb_id, chain, spike_percentile, status) VALUES ("
            f"{sqle(tgt)}, "
            f"{sqle(tgt.split('_chain')[0].lower())}, "
            f"{sqle(tgt.split('_chain')[-1])}, "
            f"{percentile}, 'completed');"
        )

        dcc_stmts.append(
            f"INSERT OR REPLACE INTO corrected_dcc "
            f"(target, centroid_dcc, spike_dcc, spike_site, n_parquet_sites, dcc_grade) VALUES ("
            f"{sqle(tgt)}, "
            f"{sqln(r.get('best_centroid_dcc'))}, "
            f"{sqln(r.get('best_spike_dcc'))}, "
            f"{sqle(r['best_site'])}, "
            f"{r['n_sites']}, "
            f"{sqle(grade(best))});"
        )

        # v4 contract: post_campaign_analysis is a W2-class writer — it owns ONLY
        # min_dist_to_ligand, graded_score, dcc_metric_source (+ corrected_dcc).
        # Every other site_features column is W1-owned and must NOT be touched.
        # Write pattern: INSERT OR IGNORE (preserve W1 columns) + column-scoped UPDATE.
        for s in r["sites"]:
            md = s.get("spike_dcc") if s.get("spike_dcc") is not None else s.get("centroid_dcc")
            graded = 1.0 / (1.0 + md) if md is not None else None
            site_stmts.append(
                f"INSERT OR IGNORE INTO site_features (target, site_name) VALUES ("
                f"{sqle(tgt)}, {sqle(s['site_name'])});"
            )
            site_stmts.append(
                f"UPDATE site_features SET "
                f"min_dist_to_ligand = {sqln(md)}, "
                f"graded_score       = {sqln(graded)}, "
                f"dcc_metric_source  = {sqle('pct70_' + used_metric)} "
                f"WHERE target = {sqle(tgt)} AND site_name = {sqle(s['site_name'])};"
            )

    def write_batch(name, stmts, batch_size):
        files = []
        for i in range(0, len(stmts), batch_size):
            p = out_dir / f"{name}_{i // batch_size:03d}.sql"
            p.write_text("\n".join(stmts[i:i+batch_size]))
            files.append(p)
        return files

    t = write_batch("targets_pct70", target_stmts, 500)
    d = write_batch("dcc_pct70", dcc_stmts, 300)
    s = write_batch("sites_pct70", site_stmts, 200)
    print(f"SQL batches: {len(target_stmts)} targets ({len(t)}), "
          f"{len(dcc_stmts)} dcc ({len(d)}), {len(site_stmts)} sites ({len(s)})")
    return t + d + s


def load_to_d1(sql_files: List[Path]) -> int:
    loaded = 0
    for f in sql_files:
        r = subprocess.run(
            ["npx", "wrangler", "d1", "execute", "prism-features",
             "--remote", "--file", str(f)],
            cwd=WORKSTATION_WRANGLER_CWD, capture_output=True, text=True, timeout=300,
        )
        if r.returncode == 0:
            loaded += 1
    return loaded


# ─────────────────────────────────────────────────────────────
#  Comparison pct70 vs pct95
# ─────────────────────────────────────────────────────────────

def compare(results_70: List[Dict[str, Any]], pct95_path: str) -> Dict[str, Any]:
    if not Path(pct95_path).exists():
        return {"warning": f"pct95 data not found at {pct95_path}"}
    pct95 = json.load(open(pct95_path))
    pct95_map = {r["target"]: r for r in pct95}

    gd70, gd95 = {}, {}
    dccs70, dccs95 = [], []
    paired = []
    for r in results_70:
        if "error" in r or r.get("skip_reason"):
            continue
        tgt = r["target"]
        d70 = r.get("best_spike_dcc") if r.get("best_spike_dcc") is not None else r["best_centroid_dcc"]
        g70 = grade(d70)
        gd70[g70] = gd70.get(g70, 0) + 1
        dccs70.append(d70)

        p95 = pct95_map.get(tgt, {})
        d95 = p95.get("spike_dcc") if p95.get("spike_dcc") is not None else p95.get("centroid_dcc")
        if d95 is not None:
            g95 = p95.get("spike_grade") if p95.get("spike_grade") not in (None, "N/A") \
                  else p95.get("centroid_grade", "N/A")
            gd95[g95] = gd95.get(g95, 0) + 1
            dccs95.append(d95)
            paired.append({"target": tgt, "pct95": d95, "pct70": d70, "delta": d95 - d70})

    return {
        "n_pct70": sum(gd70.values()),
        "n_pct95_paired": len(paired),
        "grade_dist_pct70": gd70,
        "grade_dist_pct95": gd95,
        "median_pct70": float(np.median(dccs70)) if dccs70 else None,
        "median_pct95": float(np.median(dccs95)) if dccs95 else None,
        "n_improved": sum(1 for p in paired if p["delta"] > 0),
        "n_regressed": sum(1 for p in paired if p["delta"] < 0),
        "paired_comparisons": paired,
    }


# ─────────────────────────────────────────────────────────────
#  Driver
# ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--r2-prefix", default=DEFAULT_R2_PREFIX)
    parser.add_argument("--percentile", type=int, default=70)
    parser.add_argument("--manifest",
                        default="/mnt/storage/prism-outputs/_corpus_runner_logs/proteome_1000_pct70_372.txt")
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT))
    parser.add_argument("--pct95-dcc", default=PCT95_DCC_PATH)
    parser.add_argument("--workers", type=int, default=MAX_WORKERS)
    parser.add_argument("--monitor", action="store_true")
    parser.add_argument("--poll-interval", type=int, default=600)
    parser.add_argument("--deep-dcc", type=str,
                        help="Compute spike-DCC (parquet-based) for a single target")
    parser.add_argument("--skip-d1-load", action="store_true")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = out_dir / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Deep-DCC mode: run on ONE target, full parquet pull
    if args.deep_dcc:
        print(f"Deep spike-DCC for {args.deep_dcc}...")
        result = deep_spike_dcc(args.deep_dcc, args.r2_prefix, cache_dir)
        print(json.dumps(result, indent=2, default=str))
        return

    # Load manifest
    targets = [line.strip() for line in open(args.manifest) if line.strip()]
    print(f"Manifest: {len(targets)} targets")

    # Monitor mode — wait for campaign to complete
    if args.monitor:
        print(f"Monitor mode — polling r2:prism-archive/{args.r2_prefix}/ every {args.poll_interval}s")
        while True:
            r = subprocess.run(
                ["rclone", "lsd", f"r2:prism-archive/{args.r2_prefix}/"],
                capture_output=True, text=True, timeout=60,
            )
            n_dirs = len([l for l in r.stdout.splitlines() if l.strip()])
            print(f"  R2 dirs: {n_dirs} / {len(targets)}", flush=True)
            if n_dirs >= len(targets):
                print("✓ Campaign complete — starting analysis")
                break
            time.sleep(args.poll_interval)

    print(f"\n─── Fast pass: centroid-DCC from binding_sites.json (no parquets) ───")
    print(f"  Egress: ~{len(targets) * 0.3:.0f} MB total (vs ~100 TB for full parquets)")
    t0 = time.time()
    results = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futs = {pool.submit(compute_target_centroid_dcc, t, args.r2_prefix, cache_dir): t for t in targets}
        done = 0
        for fut in as_completed(futs):
            r = fut.result()
            if r is not None:
                results.append(r)
            done += 1
            if done % 25 == 0:
                rate = done / (time.time() - t0)
                eta = (len(targets) - done) / max(rate, 0.01) / 60
                print(f"  [{done}/{len(targets)}]  rate={rate:.1f}/s  eta~{eta:.0f}m", flush=True)
    print(f"  Fast pass: {(time.time()-t0)/60:.1f}m — {len(results)} targets")

    with open(out_dir / "pct70_site_dcc_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)

    # D1 updates
    sql_files = generate_d1_updates(results, out_dir / "sql", percentile=args.percentile)
    if not args.skip_d1_load:
        loaded = load_to_d1(sql_files)
        print(f"  Loaded {loaded}/{len(sql_files)} SQL batches into D1")

    # Comparison
    comparison = compare(results, args.pct95_dcc)
    with open(out_dir / "pct70_vs_pct95_report.json", "w") as f:
        json.dump(comparison, f, indent=2, default=str)

    print(f"\n{'='*60}")
    print("  pct70 Analysis Report (centroid-DCC metric)")
    print(f"{'='*60}")
    for g, n in sorted(comparison["grade_dist_pct70"].items()):
        print(f"  {g}: {n}")
    print(f"  Median: pct70={comparison['median_pct70']:.2f}Å  "
          f"pct95={comparison['median_pct95']:.2f}Å" if comparison.get('median_pct95') else "")
    print(f"  Paired: {comparison['n_pct95_paired']} | improved {comparison['n_improved']} | "
          f"regressed {comparison['n_regressed']}")
    print(f"\n  Outputs in: {out_dir}")
    print(f"\n  Note: this used centroid-DCC (cheap). For deep spike-DCC validation")
    print(f"  on a specific target, run: --deep-dcc <target>")


if __name__ == "__main__":
    main()
