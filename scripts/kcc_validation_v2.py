#!/usr/bin/env python3
"""
KCC Validation v2 — Dual-regime post-processing (no simulation required).

Reads kcc_visualization.json, produces:
  - kcc_validation_v2.json (machine-readable, regime-aware)
  - kcc_validation_v2.pml (PyMOL, regime-aware styling)

Usage:
  python3 scripts/kcc_validation_v2.py /tmp/1p38_val/1p38.kcc_visualization.json
"""

import json
import math
import sys
import os
from datetime import datetime


def normalize_vec(v):
    mag = math.sqrt(sum(x * x for x in v))
    if mag < 1e-9:
        return None
    return [x / mag for x in v]


def dot(a, b):
    return sum(x * y for x, y in zip(a, b))


def distance(a, b):
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


def score_residue(r):
    """Composite residue score for Top-K selection."""
    me = r.get("motion_efficiency", 0)
    lc = r.get("lag_corr_peak", 0)
    lcv = r.get("local_cov", 0)
    return me * (1 + max(lc, 0)) * (1 + max(lcv, 0))


def compute_site_validation(site, all_residues, all_signal_strengths):
    """Compute dual-regime validation for one site."""
    sid = site.get("id", 0)
    rank_score = site.get("rank_score", 0)
    gtck_rank = site.get("gtck_rank", 999)
    centroid = site.get("centroid", [0, 0, 0])

    # Get Top-K residues from KCC candidate data
    kcc = site.get("kcc", {})
    cand_ids = kcc.get("candidate_residue_ids", [])

    if not cand_ids:
        # Fallback: pick top-K from all residues by score, near site centroid
        scored = [(r, score_residue(r)) for r in all_residues]
        scored.sort(key=lambda x: -x[1])
        cand_ids = [r["residue_id"] for r, _ in scored[:3]]

    # Collect residue data for candidates
    res_map = {r["residue_id"]: r for r in all_residues}
    topk = [res_map[rid] for rid in cand_ids if rid in res_map]
    K = len(topk)

    if K == 0:
        return {
            "site_id": sid, "gtck_rank": gtck_rank, "rank_score": rank_score,
            "regime": "unknown", "structural": {}, "vector": {}, "signal": {},
            "verdict": "FAIL", "reason": "no_candidates"
        }

    # === Structural metrics ===
    positions = [r["ca_position"] for r in topk]
    cx = sum(p[0] for p in positions) / K
    cy = sum(p[1] for p in positions) / K
    cz = sum(p[2] for p in positions) / K
    struct_centroid = [cx, cy, cz]

    mean_radius = sum(distance(p, struct_centroid) for p in positions) / K
    max_dist = 0.0
    for i in range(K):
        for j in range(i + 1, K):
            d = distance(positions[i], positions[j])
            if d > max_dist:
                max_dist = d

    # === Vector metrics ===
    vectors = []
    for r in topk:
        v = normalize_vec([r.get("net_dx", 0), r.get("net_dy", 0), r.get("net_dz", 0)])
        if v is not None:
            vectors.append(v)

    mean_cos = 0.0
    if len(vectors) >= 2:
        cos_vals = []
        for i in range(len(vectors)):
            for j in range(i + 1, len(vectors)):
                cos_vals.append(dot(vectors[i], vectors[j]))
        mean_cos = sum(cos_vals) / len(cos_vals)

    vector_variance = max(0.0, 1.0 - mean_cos)  # >1.0 = opposing vectors

    # === Signal metrics ===
    n_causal = sum(1 for r in topk if r.get("active_causal_steps", 0) > 0)
    causal_coverage = n_causal / K if K > 0 else 0

    signal_vals = []
    for r in topk:
        me = r.get("motion_efficiency", 0)
        lc = max(r.get("lag_corr_peak", 0), 0.0)
        signal_vals.append(me * lc)
    signal_strength = sum(signal_vals) / len(signal_vals) if signal_vals else 0

    # Distance-weighted signal
    dw_vals = []
    for r in topk:
        me = r.get("motion_efficiency", 0)
        lc = max(r.get("lag_corr_peak", 0), 0.0)
        d = distance(r["ca_position"], struct_centroid)
        dw_vals.append(me * lc / (1.0 + d))
    dw_signal = sum(dw_vals) / len(dw_vals) if dw_vals else 0

    # === Regime detection ===
    regime = "local" if mean_radius < 6.0 else "distributed"

    # === PASS/FAIL ===
    if regime == "local":
        struct_pass = mean_radius < 6.0
        vec_pass = mean_cos > 0.5
        sig_pass = causal_coverage >= 0.6
        verdict = "PASS" if (struct_pass and vec_pass and sig_pass) else "WARN" if sig_pass else "FAIL"
    else:
        # Distributed: use percentile-based checks (computed later with global context)
        sig_pass = causal_coverage >= 0.6
        vec_ok = vector_variance < 0.9
        verdict = "PASS" if (sig_pass and vec_ok) else "WARN" if sig_pass else "FAIL"

    topk_json = []
    for r in topk:
        topk_json.append({
            "residue_id": r["residue_id"],
            "residue_name": r.get("residue_name", "UNK"),
            "ca_position": r["ca_position"],
            "kcc": {
                "motion_efficiency": r.get("motion_efficiency", 0),
                "lag_corr": r.get("lag_corr_peak", 0),
                "burst": r.get("burst_motion", 0),
                "local_cov": r.get("local_cov", 0),
            }
        })

    return {
        "site_id": sid,
        "gtck_rank": gtck_rank,
        "rank_score": rank_score,
        "regime": regime,
        "topk_residues": topk_json,
        "structural": {
            "centroid": struct_centroid,
            "mean_radius": round(mean_radius, 2),
            "max_distance": round(max_dist, 2),
        },
        "vector": {
            "mean_cosine_similarity": round(mean_cos, 4),
            "vector_variance": round(vector_variance, 4),
        },
        "signal": {
            "causal_coverage": round(causal_coverage, 3),
            "signal_strength": round(signal_strength, 6),
            "distance_weighted_signal": round(dw_signal, 6),
        },
        "verdict": verdict,
    }


def refine_distributed_verdicts(val_sites):
    """Apply percentile-based checks for distributed-regime sites."""
    distributed = [s for s in val_sites if s["regime"] == "distributed"]
    if len(distributed) < 2:
        return

    sig_values = [s["signal"]["signal_strength"] for s in distributed]
    dw_values = [s["signal"]["distance_weighted_signal"] for s in distributed]
    sig_median = sorted(sig_values)[len(sig_values) // 2] if sig_values else 0
    dw_median = sorted(dw_values)[len(dw_values) // 2] if dw_values else 0

    for s in distributed:
        if s["verdict"] == "FAIL":
            continue
        sig_ok = s["signal"]["signal_strength"] >= sig_median
        dw_ok = s["signal"]["distance_weighted_signal"] >= dw_median
        # Distributed verdict: signal strength determines validity, NOT vector alignment
        if sig_ok and dw_ok:
            s["verdict"] = "PASS"
        elif sig_ok or dw_ok:
            s["verdict"] = "WARN"
        else:
            s["verdict"] = "FAIL"
        # Vector variance is annotation only — not a gate
        vec_var = s["vector"]["vector_variance"]
        if "flags" not in s:
            s["flags"] = []
        if vec_var > 1.2:
            s["flags"].append("multi-directional motion (distributed hinge/allosteric coupling)")
        elif vec_var > 0.9:
            s["flags"].append("mixed directional motion")
        else:
            s["flags"].append("coherent directional motion")


def generate_pml(val_sites, output_path):
    """Generate regime-aware PyMOL validation script."""
    with open(output_path, "w") as f:
        f.write("# PRISM4D KCC Validation v2 — Dual-Regime PyMOL Session\n")
        f.write("# Auto-generated (deterministic)\n\n")
        f.write("hide everything\nshow cartoon\ncolor gray80, all\n")
        f.write("set cartoon_transparency, 0.3\nset cgo_line_width, 3\nbg_color white\n\n")

        group_names = []
        for s in val_sites:
            sid = s["site_id"]
            regime = s["regime"]
            verdict = s["verdict"]
            rank = s.get("gtck_rank", 999)

            f.write(f"# --- Site {sid} (Rank {rank}, {regime.upper()}, {verdict}) ---\n")

            # Regime-aware residue coloring
            color = "yellow" if regime == "local" else "orange"
            f.write(f"# Regime: {regime}\n")

            # Try to select existing groups from kcc_session.pml
            group_name = f"site_{sid}_full"
            group_names.append(group_name)

            # Style for residues
            # Reference global_kcc_drivers (created by kcc_session.pml) or per-site
            driver_sel = "global_kcc_drivers" if regime == "distributed" else f"site_{sid}_local_drivers"
            f.write(f"color {color}, {driver_sel}\n")
            if regime == "distributed":
                f.write(f"set sphere_scale, 0.4, {driver_sel}\n")

            # Verdict label
            verdict_color = {"PASS": "green", "WARN": "yellow", "FAIL": "red"}[verdict]
            f.write(f"# Verdict: {verdict} (signal={s['signal']['signal_strength']:.4f})\n\n")

        # Commands via cmd.extend (reliable multi-command)
        f.write("\n# === INSPECTION COMMANDS ===\n")
        f.write("python\nfrom pymol import cmd\n")
        for i, gn in enumerate(group_names):
            f.write(f"def _inspect_{i}(self=None):\n")
            f.write(f"    cmd.disable('all')\n")
            f.write(f"    cmd.enable('{gn}')\n")
            f.write(f"    cmd.enable('kcc_vectors')\n")
            f.write(f"    cmd.show('cartoon')\n")
            f.write(f"    cmd.set('cartoon_transparency', 0.3)\n")
            f.write(f"    cmd.zoom('{gn}', 10)\n")
            f.write(f"cmd.extend('inspect_site{i}', _inspect_{i})\n")

        if len(group_names) >= 2:
            f.write(f"def _compare_top2(self=None):\n")
            f.write(f"    cmd.disable('all')\n")
            f.write(f"    cmd.enable('{group_names[0]}')\n")
            f.write(f"    cmd.enable('{group_names[1]}')\n")
            f.write(f"    cmd.enable('kcc_vectors')\n")
            f.write(f"    cmd.show('cartoon')\n")
            f.write(f"    cmd.set('cartoon_transparency', 0.3)\n")
            f.write(f"    cmd.zoom('all')\n")
            f.write(f"cmd.extend('compare_top2', _compare_top2)\n")

        f.write("python end\n")

        # Default: show top-ranked
        if group_names:
            f.write(f"\n# Default view: Rank 1\n")
            f.write(f"disable all\nenable {group_names[0]}\nenable KCC_VECTORS\n")
            f.write(f"show cartoon\nset cartoon_transparency, 0.3\nzoom {group_names[0]}, 10\n")


def main():
    if len(sys.argv) < 2:
        print("Usage: python3 kcc_validation_v2.py <kcc_visualization.json> [--output-dir <dir>]")
        sys.exit(1)

    viz_path = sys.argv[1]
    # Optional: explicit output directory
    output_dir_override = None
    if "--output-dir" in sys.argv:
        idx = sys.argv.index("--output-dir")
        if idx + 1 < len(sys.argv):
            output_dir_override = sys.argv[idx + 1]
    with open(viz_path) as f:
        viz = json.load(f)

    pdb_source = viz.get("pdb_source", "unknown")
    residues = viz.get("residues", [])
    sites = viz.get("sites", [])

    # Sort sites by rank
    sites.sort(key=lambda s: s.get("gtck_rank", s.get("rank_score", 0)))

    # Compute per-site validation (top 5 ranked)
    val_sites = []
    for site in sites[:5]:
        v = compute_site_validation(site, residues, [])
        val_sites.append(v)

    # Refine distributed verdicts with percentile context
    refine_distributed_verdicts(val_sites)

    # Global checks
    scores = [s["rank_score"] for s in val_sites if s["rank_score"] > 0]
    sep = abs(scores[0] - scores[1]) / max(scores[0], 1e-12) if len(scores) >= 2 else 0
    sig_vals = [s["signal"]["signal_strength"] for s in val_sites]
    sig_median = sorted(sig_vals)[len(sig_vals) // 2] if sig_vals else 0

    # Output paths — use --output-dir if specified, else derive from input
    base = os.path.splitext(viz_path)[0].replace(".kcc_visualization", "")
    if output_dir_override:
        basename = os.path.basename(base)
        json_out = os.path.join(output_dir_override, basename + ".kcc_validation_v2.json")
        pml_out = os.path.join(output_dir_override, basename + ".kcc_validation_v2.pml")
    else:
        json_out = base + ".kcc_validation_v2.json"
        pml_out = base + ".kcc_validation_v2.pml"

    run_id = os.path.basename(base) + "_" + datetime.utcnow().strftime("%Y%m%d_%H%M%S")

    output = {
        "pdb_source": pdb_source,
        "run_id": run_id,
        "sites": val_sites,
        "global_checks": {
            "top1_vs_top2_separation": round(sep, 4),
            "n_validated_sites": len(val_sites),
            "strong_site_has_strong_signal": sig_vals[0] >= sig_median if sig_vals else False,
            "weak_site_has_weaker_signal": sig_vals[-1] <= sig_median if len(sig_vals) >= 2 else True,
        },
        "semantics": {
            "local": "geometrically compact pocket drivers — residues cluster within 6A",
            "distributed": "non-local causal network — allosteric, hinge-driven, or interface mechanism",
            "lag_corr": "cross-correlation peak between causal activity and residue motion",
            "burst": "temporal clustering of causal events (dense vs sparse spacing)",
            "local_cov": "local covariance of motion and causality in timestep subwindows",
        }
    }

    with open(json_out, "w") as f:
        json.dump(output, f, indent=2)
    print(f"✓ Validation JSON: {json_out}")

    # Print summary
    for s in val_sites:
        regime_tag = f"[{s['regime'].upper():>11}]"
        print(f"  Rank {s.get('gtck_rank', '?'):>2}: site {s['site_id']:>5} {regime_tag} "
              f"verdict={s['verdict']:>4} "
              f"mean_rad={s['structural'].get('mean_radius', 0):>5.1f}A "
              f"cos_sim={s['vector'].get('mean_cosine_similarity', 0):>6.3f} "
              f"signal={s['signal'].get('signal_strength', 0):.4f} "
              f"coverage={s['signal'].get('causal_coverage', 0):.2f}")

    # Generate PyMOL
    generate_pml(val_sites, pml_out)
    print(f"✓ Validation PML: {pml_out}")


if __name__ == "__main__":
    main()
