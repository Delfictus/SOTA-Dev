#!/usr/bin/env python3
"""Differential A/B analysis: pct95 vs pct70 — directive Phase 4.

Produces the full comparison matrix specified by Phase 4.1 plus the
per-target paired analysis (4.2) and TIDE differential (4.3).

Inputs:
    Two R2 prefixes — default `10k-runs` (pct95) and `10k-runs-pct70`.
    Per-target artifacts expected on R2:
        {target}.binding_sites.json
        {target}.topology.prism_therm.json
        {target}_ground_truth.json
    Optional: extracted feature bundles from `extract_all_features.py`
        /mnt/storage/spike-audit/features-{pct95,pct70}/

Outputs:
    differential_report.json
    differential_report.md         (human-readable)
    per_target_delta.csv           (spike_dcc delta per target)

Usage:
    # After both campaigns complete:
    python3 scripts/training/differential_analysis.py \
        --out-dir /mnt/storage/spike-audit/differential/

    # Early peek (subset):
    python3 scripts/training/differential_analysis.py \
        --targets 10dc_chainA,10dj_chainA,11qe_chainA \
        --out-dir /tmp/differential_peek/
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import urllib.request
from collections import defaultdict
from pathlib import Path
from statistics import mean, median, stdev
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

R2_PCT95 = "r2:prism-archive/10k-runs"
R2_PCT70 = "r2:prism-archive/10k-runs-pct70"
D1_WORKER_URL = os.environ.get("PRISM_API",
                               "https://prism-feature-pipeline.is-0b9.workers.dev")


# ─────────────────────────────────────────────────────────────
#  Staging / R2 helpers
# ─────────────────────────────────────────────────────────────

def stage_one(target: str, prefix: str, cache_dir: Path) -> Path:
    out = cache_dir / prefix.split("/")[-1] / target
    out.mkdir(parents=True, exist_ok=True)
    needed = [
        f"{target}.binding_sites.json",
        f"{target}.topology.prism_therm.json",
        f"{target}_ground_truth.json",
    ]
    for name in needed:
        if (out / name).exists() and (out / name).stat().st_size > 0:
            continue
        subprocess.run(
            ["rclone", "copy", f"{prefix}/{target}/{name}", str(out), "--quiet"],
            capture_output=True, timeout=60, check=False,
        )
    return out


def list_targets(prefix: str) -> List[str]:
    r = subprocess.run(
        ["rclone", "lsd", prefix, "--max-depth", "1"],
        capture_output=True, text=True, timeout=120, check=False,
    )
    out = []
    for line in r.stdout.splitlines():
        parts = line.split()
        if len(parts) >= 5:
            out.append(parts[-1])
    return sorted(out)


def api_get(url: str, timeout: int = 60) -> Any:
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 prism4d-diff"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


# ─────────────────────────────────────────────────────────────
#  Per-target reads
# ─────────────────────────────────────────────────────────────

def read_therm(target_dir: Path, target: str) -> Optional[Dict[str, Any]]:
    p = target_dir / f"{target}.topology.prism_therm.json"
    if not p.exists():
        return None
    try:
        d = json.load(open(p))
    except Exception:
        return None
    pockets = d.get("pockets", []) or []
    cryptic = [p for p in pockets if p.get("is_cryptic")]
    hyst = [p.get("hysteresis_asymmetry", 0.0) or 0.0 for p in pockets]
    tide = []
    for p in pockets:
        for r in p.get("top_residues", []):
            tide.append(float(r.get("transfer_entropy", 0.0) or 0.0))
    therm_class = defaultdict(int)
    for p in pockets:
        therm_class[p.get("therm_class", "UNKNOWN")] += 1
    return {
        "n_pockets": len(pockets),
        "n_cryptic": len(cryptic),
        "hysteresis_mean": float(mean(hyst)) if hyst else 0.0,
        "hysteresis_std": float(stdev(hyst)) if len(hyst) > 1 else 0.0,
        "transfer_entropy_mean": float(mean(tide)) if tide else 0.0,
        "therm_class_counts": dict(therm_class),
        "tide_residues_mapped": int(d.get("tide_residues_mapped", 0) or 0),
        "sdst_event_count": int(d.get("sdst_event_count", 0) or 0),
    }


def read_sites(target_dir: Path, target: str) -> Optional[Dict[str, Any]]:
    p = target_dir / f"{target}.binding_sites.json"
    if not p.exists():
        return None
    try:
        d = json.load(open(p))
    except Exception:
        return None
    sites = d.get("sites", []) or []
    spike_counts = [s.get("spike_count", 0) or 0 for s in sites]
    unsat = [s.get("unsat_frac") for s in sites if s.get("unsat_frac") is not None]
    streams = [s.get("n_streams", 0) or 0 for s in sites]
    persistence = [s.get("persistence") for s in sites if s.get("persistence") is not None]
    return {
        "n_sites": len(sites),
        "spike_total": int(sum(spike_counts)),
        "spike_per_site_mean": float(mean(spike_counts)) if spike_counts else 0.0,
        "unsat_frac_mean": float(mean(unsat)) if unsat else 0.0,
        "n_streams_mean": float(mean(streams)) if streams else 0.0,
        "persistence_std": float(stdev(persistence)) if len(persistence) > 1 else 0.0,
    }


def read_corrected_dcc() -> Dict[str, Dict[str, Any]]:
    """Pull corrected_dcc records from D1 — spike-based DCC per target."""
    try:
        d = api_get(f"{D1_WORKER_URL}/dcc")
    except Exception:
        return {}
    out = {}
    for r in d.get("records", []):
        t = r.get("target")
        if not t:
            continue
        out[t] = {
            "centroid_dcc": r.get("centroid_dcc"),
            "spike_dcc": r.get("spike_dcc"),
            "dcc_grade": r.get("dcc_grade"),
            "spike_site": r.get("spike_site"),
        }
    return out


# ─────────────────────────────────────────────────────────────
#  Aggregators
# ─────────────────────────────────────────────────────────────

def dcc_bucket_stats(records: List[Dict[str, Any]]) -> Dict[str, float]:
    dccs = [r.get("spike_dcc") for r in records]
    dccs = [float(d) for d in dccs if d is not None]
    if not dccs:
        return {"n": 0, "sr_2A": 0.0, "sr_8A": 0.0, "median": None}
    n = len(dccs)
    return {
        "n": n,
        "sr_2A": sum(1 for d in dccs if d <= 2.0) / n,
        "sr_8A": sum(1 for d in dccs if d <= 8.0) / n,
        "median": float(median(dccs)),
        "mean": float(mean(dccs)),
    }


def therm_class_delta(a: Dict[str, int], b: Dict[str, int]) -> Dict[str, float]:
    """% shift in therm_class distribution between campaigns."""
    total_a = max(sum(a.values()), 1)
    total_b = max(sum(b.values()), 1)
    keys = set(a) | set(b)
    return {k: (b.get(k, 0) / total_b - a.get(k, 0) / total_a) for k in keys}


def pair_targets(targets: List[str], pct95_dir: Path, pct70_dir: Path,
                 dcc_records: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    per_target = []
    improved = regressed = stable = 0
    excellent_regressed = []
    for t in targets:
        dcc95 = dcc_records.get(t, {}).get("spike_dcc")
        # For pct70, until the D1 pipeline fully catches up, read from R2
        # ground truth + site nearest-centroid.
        dcc70 = None
        gt_p = pct70_dir / t / f"{t}_ground_truth.json"
        bs_p = pct70_dir / t / f"{t}.binding_sites.json"
        if gt_p.exists() and bs_p.exists():
            try:
                gt = json.load(open(gt_p))
                bs = json.load(open(bs_p))
                if gt.get("valid_for_dcc_validation") and gt.get("ligand_centroid"):
                    lc = np.asarray(gt["ligand_centroid"], dtype=np.float32)
                    best = None
                    for s in bs.get("sites", []):
                        c = s.get("centroid")
                        if c and len(c) == 3:
                            d = float(np.linalg.norm(np.asarray(c) - lc))
                            if best is None or d < best:
                                best = d
                    dcc70 = best
            except Exception:
                pass
        if dcc95 is None or dcc70 is None:
            continue
        delta = dcc70 - dcc95
        grade95 = dcc_records.get(t, {}).get("dcc_grade")
        cat = "STABLE"
        if delta < -2.0:
            cat = "IMPROVED"; improved += 1
        elif delta > 2.0:
            cat = "REGRESSED"; regressed += 1
            if grade95 == "EXCELLENT":
                excellent_regressed.append(t)
        else:
            stable += 1
        per_target.append({"target": t, "dcc_pct95": dcc95, "dcc_pct70": dcc70,
                           "delta": delta, "category": cat, "grade_pct95": grade95})
    return {
        "improved": improved, "regressed": regressed, "stable": stable,
        "excellent_regressed": excellent_regressed,
        "n_paired": len(per_target),
        "per_target": per_target,
    }


# ─────────────────────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pct95-prefix", default=R2_PCT95)
    parser.add_argument("--pct70-prefix", default=R2_PCT70)
    parser.add_argument("--cache-dir", type=Path,
                        default=Path("/mnt/storage/spike-audit/differential-cache"))
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--targets", default="", help="Comma-separated subset")
    parser.add_argument("--max", type=int, default=0, help="Cap # targets")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.cache_dir.mkdir(parents=True, exist_ok=True)

    # 1) Discover overlap between campaigns
    if args.targets:
        targets = args.targets.split(",")
    else:
        pct95_targets = set(list_targets(args.pct95_prefix))
        pct70_targets = set(list_targets(args.pct70_prefix))
        targets = sorted(pct95_targets & pct70_targets)
        print(f"pct95: {len(pct95_targets)}, pct70: {len(pct70_targets)}, overlap: {len(targets)}")
    if args.max:
        targets = targets[:args.max]

    # 2) Stage + read per-target data for both campaigns
    pct95_dir = args.cache_dir
    pct70_dir = args.cache_dir
    print(f"Staging {len(targets)} targets for both campaigns...")
    pct95_sites = {}; pct70_sites = {}
    pct95_therm = {}; pct70_therm = {}
    for i, t in enumerate(targets):
        d95 = stage_one(t, args.pct95_prefix, args.cache_dir)
        d70 = stage_one(t, args.pct70_prefix, args.cache_dir)
        pct95_sites[t] = read_sites(d95, t)
        pct70_sites[t] = read_sites(d70, t)
        pct95_therm[t] = read_therm(d95, t)
        pct70_therm[t] = read_therm(d70, t)
        if (i + 1) % 25 == 0:
            print(f"  staged {i+1}/{len(targets)}")

    # 3) DCC records (spike-based) from D1
    dcc = read_corrected_dcc()
    pct95_dcc = [dcc.get(t, {}) for t in targets if dcc.get(t)]

    # 4) Aggregate comparison matrix
    matrix: Dict[str, Dict[str, Any]] = {}

    # Detection quality (pct95 from D1, pct70 placeholder until DCC pipeline catches up)
    matrix["detection_quality"] = {
        "pct95": dcc_bucket_stats(pct95_dcc),
        "pct70": {"n": 0, "note": "awaiting pct70 corrected_dcc records"},
    }

    # Site density
    def site_agg(d):
        ns = [v["n_sites"] for v in d.values() if v]
        spk = [v["spike_total"] for v in d.values() if v]
        return {
            "sites_per_target_median": float(median(ns)) if ns else 0.0,
            "sites_per_target_mean": float(mean(ns)) if ns else 0.0,
            "spikes_total_mean": float(mean(spk)) if spk else 0.0,
        }
    matrix["site_density"] = {"pct95": site_agg(pct95_sites),
                              "pct70": site_agg(pct70_sites)}

    # Spike quality
    def quality_agg(d):
        us = [v["unsat_frac_mean"] for v in d.values() if v]
        ns = [v["n_streams_mean"] for v in d.values() if v]
        ps = [v["persistence_std"] for v in d.values() if v]
        return {
            "unsat_frac_mean": float(mean(us)) if us else 0.0,
            "n_streams_mean": float(mean(ns)) if ns else 0.0,
            "persistence_std_mean": float(mean(ps)) if ps else 0.0,
        }
    matrix["spike_quality"] = {"pct95": quality_agg(pct95_sites),
                                "pct70": quality_agg(pct70_sites)}

    # TIDE features
    def tide_agg(d):
        if not d:
            return {}
        crypt = [v["n_cryptic"] for v in d.values() if v]
        hyst = [v["hysteresis_mean"] for v in d.values() if v]
        te = [v["transfer_entropy_mean"] for v in d.values() if v]
        tc = defaultdict(int)
        for v in d.values():
            if v:
                for k, n in v.get("therm_class_counts", {}).items():
                    tc[k] += n
        return {
            "cryptic_pockets_mean": float(mean(crypt)) if crypt else 0.0,
            "hysteresis_asymmetry_mean": float(mean(hyst)) if hyst else 0.0,
            "hysteresis_asymmetry_std": float(stdev(hyst)) if len(hyst) > 1 else 0.0,
            "transfer_entropy_mean": float(mean(te)) if te else 0.0,
            "therm_class_counts": dict(tc),
        }
    t95 = tide_agg(pct95_therm)
    t70 = tide_agg(pct70_therm)
    matrix["tide"] = {
        "pct95": t95, "pct70": t70,
        "therm_class_distribution_delta": therm_class_delta(
            t95.get("therm_class_counts", {}),
            t70.get("therm_class_counts", {})),
        "cryptic_pocket_ratio_pct70_vs_pct95": (
            (t70.get("cryptic_pockets_mean", 0) /
             max(t95.get("cryptic_pockets_mean", 1e-9), 1e-9))
            if t95.get("cryptic_pockets_mean") else None
        ),
    }

    # 5) Per-target paired DCC analysis
    matrix["paired_analysis"] = pair_targets(targets, pct95_dir, pct70_dir, dcc)

    # 6) Ranker / teacher / VN-EGNN slots — populated by downstream runs
    matrix["ranker_performance"] = {
        "note": "tokenized_v4 / xgb_v3 / xgb_v4 SR@1 populate after training runs",
        "tokenized_v4_pct95": {"sr_at_1": 0.3642, "source": "validated earlier"},
        "xgboost_v3_pct95": {"sr_at_1": 0.4783, "source": "LOTO eval"},
    }
    matrix["teacher_auroc"] = {
        "pct95": None,
        "pct70": None,
        "note": "populated after teacher_v004 LOTO + teacher_v005 LOTO complete",
    }
    matrix["vnegnn_auroc"] = {
        "pct95": None,
        "pct70": None,
        "note": "populated after vn_egnn_v001 / vn_egnn_v002 training complete",
    }

    # 7) Write outputs
    out_json = args.out_dir / "differential_report.json"
    out_json.write_text(json.dumps(matrix, indent=2, default=str))
    print(f"  wrote {out_json}")

    # CSV of per-target deltas
    per_tgt = matrix["paired_analysis"]["per_target"]
    if per_tgt:
        csv_path = args.out_dir / "per_target_delta.csv"
        with open(csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["target", "dcc_pct95", "dcc_pct70",
                                              "delta", "category", "grade_pct95"])
            w.writeheader()
            w.writerows(per_tgt)
        print(f"  wrote {csv_path}")

    # Markdown summary
    md = args.out_dir / "differential_report.md"
    with open(md, "w") as f:
        f.write("# PRISM-4D Differential A/B Report\n\n")
        f.write(f"Targets in overlap: **{len(targets)}**\n\n")

        pa = matrix["paired_analysis"]
        f.write("## Per-target DCC delta\n")
        f.write(f"- Improved: **{pa['improved']}**  "
                f"- Regressed: **{pa['regressed']}**  "
                f"- Stable: **{pa['stable']}**  "
                f"(paired: {pa['n_paired']})\n")
        if pa["excellent_regressed"]:
            f.write(f"- EXCELLENT→ regressed targets (investigate): "
                    f"`{', '.join(pa['excellent_regressed'])}`\n")

        f.write("\n## TIDE (thermodynamic)\n")
        t = matrix["tide"]
        f.write(f"- pct95: cryptic/target={t['pct95'].get('cryptic_pockets_mean', 0):.2f}  "
                f"hyst_mean={t['pct95'].get('hysteresis_asymmetry_mean', 0):.3f}\n")
        f.write(f"- pct70: cryptic/target={t['pct70'].get('cryptic_pockets_mean', 0):.2f}  "
                f"hyst_mean={t['pct70'].get('hysteresis_asymmetry_mean', 0):.3f}\n")
        cr = t["cryptic_pocket_ratio_pct70_vs_pct95"]
        if cr is not None:
            f.write(f"- cryptic ratio pct70/pct95: **{cr:.2f}**\n")
            f.write(f"- Gate (≥1.20): {'PASS' if cr >= 1.20 else 'FAIL'}\n")

        f.write("\n## Site density\n")
        sd = matrix["site_density"]
        f.write(f"- pct95 sites/target median={sd['pct95']['sites_per_target_median']:.1f}\n")
        f.write(f"- pct70 sites/target median={sd['pct70']['sites_per_target_median']:.1f}\n")
    print(f"  wrote {md}")


if __name__ == "__main__":
    main()
