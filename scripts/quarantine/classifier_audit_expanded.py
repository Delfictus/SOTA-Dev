#!/usr/bin/env python3
"""Expanded GT-valid classifier audit on CryptoBench targets.

Read-only. Writes /tmp/classifier_audit_expanded.{json,csv} + stdout table.

Uses the CryptoBench residue-overlap ground-truth (known cryptic residues
per target) as the GT criterion, since the CB benchmark lacks per-target
paired holo ligand centroids. True-site = the engine site with max residue
overlap against the published cryptic-residue list (best_site per CB
benchmark record).

Ablations:
  A current  : {drug: 0.40, therm: 0.20, spike: 0.20, tide: 0.20}  categorical THERM_MAP
  B no_therm : {drug: 0.50, therm: 0.00, spike: 0.25, tide: 0.25}  therm removed

Both ablations are re-computed here from the raw CB engine output. A_current
reproduces the v4.1 categorical therm weighting; B_no_therm reproduces the
v4.2_detector_honest weighting. This audit is self-contained — does not
depend on run_stages.py having been executed.
"""
from __future__ import annotations
import csv
import json
import math
from pathlib import Path
from statistics import median

CB_RUN_DIR = Path("/mnt/storage/prism-outputs/runs/cryptobench199")
CB_BENCH_JSON = Path("/home/diddy/Desktop/Prism4D-bio/benchmarks/cryptobench/cryptobench_benchmark.json")
OUT_JSON = Path("/tmp/classifier_audit_expanded.json")
OUT_CSV = Path("/tmp/classifier_audit_expanded.csv")

THERM_MAP = {"CRYPTIC": 1.0, "DYNAMIC": 0.7, "RESPONSIVE": 0.4, "INERT": 0.1}
TIDE_SATURATION = 20
WEIGHTS_A = {"drug": 0.40, "therm": 0.20, "spike": 0.20, "tide": 0.20}
WEIGHTS_B = {"drug": 0.50, "therm": 0.00, "spike": 0.25, "tide": 0.25}


def merge_pockets(binding_sites_json: Path) -> list[dict] | None:
    try:
        d = json.loads(binding_sites_json.read_text())
    except Exception:
        return None
    all_pockets = d.get("all_pockets") or []
    therm_sites = (d.get("prism_therm") or {}).get("sites") or []
    by_id = {}
    for p in all_pockets:
        if not p:
            continue
        pid = p.get("site_id")
        if pid is None:
            continue
        by_id[pid] = {
            "pocket_id": pid,
            "centroid": p.get("centroid"),
            "site_volume_angstrom_cubed": p.get("mean_volume"),
        }
    for t in therm_sites:
        pid = t.get("site_id")
        if pid is None:
            continue
        tide_decomp = t.get("tide_decomposition") or {}
        n_tide_trigger = 0
        for v in tide_decomp.values() if isinstance(tide_decomp, dict) else []:
            if isinstance(v, list):
                n_tide_trigger += len(v)
            elif isinstance(v, (int, float)):
                n_tide_trigger += int(v)
        row = by_id.setdefault(pid, {"pocket_id": pid})
        row.update({
            "therm_class": t.get("therm_class"),
            "druggability_score": t.get("druggability"),
            "ccns_tau": t.get("tau"),
            "hysteresis_asymmetry": t.get("asymmetry_score"),
            "relative_asymmetry": t.get("relative_asymmetry"),
            "n_spikes_attributed": (t.get("heating_spike_count") or 0) + (t.get("cooling_spike_count") or 0),
            "n_tide_residues": n_tide_trigger,
            "is_hysteretic": t.get("is_hysteretic"),
        })
    return list(by_id.values())


def compute_composite(pockets: list[dict], weights: dict, therm_mode: str) -> list[dict]:
    max_spikes = max((p.get("n_spikes_attributed") or 0) for p in pockets) or 1
    log_spike_sat = math.log(1.0 + float(max_spikes))
    log_tide_sat = math.log(1.0 + TIDE_SATURATION)

    out = []
    for p in pockets:
        drug = max(0.0, min(1.0, float(p.get("druggability_score") or 0.0)))
        if therm_mode == "categorical":
            therm = THERM_MAP.get((p.get("therm_class") or "").upper(), 0.0)
        else:
            therm = 0.0
        n_spk = float(p.get("n_spikes_attributed") or 0)
        spike = math.log(1.0 + n_spk) / log_spike_sat if log_spike_sat > 0 else 0.0
        spike = max(0.0, min(1.0, spike))
        n_tide = int(p.get("n_tide_residues") or 0)
        tide = math.log(1.0 + n_tide) / log_tide_sat
        tide = max(0.0, min(1.0, tide))
        composite = (weights["drug"] * drug + weights["therm"] * therm
                     + weights["spike"] * spike + weights["tide"] * tide)
        out.append({
            "pocket_id": p.get("pocket_id"),
            "therm_class": p.get("therm_class"),
            "composite": composite,
        })
    out.sort(key=lambda s: -s["composite"])
    for i, s in enumerate(out, 1):
        s["rank"] = i
    return out


def rank_of(ranked: list[dict], pid) -> int | None:
    for r in ranked:
        if r["pocket_id"] == pid:
            return r["rank"]
    return None


def target_bs_path(apo_pdb: str) -> Path | None:
    tgt = CB_RUN_DIR / apo_pdb
    if not tgt.exists():
        return None
    for p in tgt.glob("*_chain*.binding_sites.json"):
        return p
    return None


def audit_one(bench_record: dict) -> dict | None:
    apo = bench_record["pdb"].lower()
    bm_pid = bench_record.get("best_site")
    if bm_pid is None:
        return None
    bs = target_bs_path(apo)
    if bs is None:
        return None
    pockets = merge_pockets(bs)
    if not pockets:
        return None

    rankA = compute_composite(pockets, WEIGHTS_A, "categorical")
    rankB = compute_composite(pockets, WEIGHTS_B, "none")

    bm_rank_A = rank_of(rankA, bm_pid)
    bm_rank_B = rank_of(rankB, bm_pid)

    top1_A = rankA[0]
    top1_B = rankB[0]

    top5_best_site_in_CB = {e["site_id"] for e in (bench_record.get("top5") or [])}
    top1_A_is_true = top1_A["pocket_id"] == bm_pid
    top1_B_is_true = top1_B["pocket_id"] == bm_pid

    # FP under ablation: top-1 ≠ best_site AND top-1 class is CRYPTIC.
    # Case-insensitive: CB therm_class fields are title-cased ("Cryptic").
    top1_A_class_u = (top1_A["therm_class"] or "").upper()
    top1_B_class_u = (top1_B["therm_class"] or "").upper()
    fp_top1_A = (not top1_A_is_true) and (top1_A_class_u == "CRYPTIC")
    fp_top1_B = (not top1_B_is_true) and (top1_B_class_u == "CRYPTIC")

    detector_found = bench_record.get("detected") is True  # CB's own detection flag

    return {
        "apo_pdb": apo,
        "best_site_by_overlap": bm_pid,
        "best_f1": bench_record.get("best_f1"),
        "best_overlap": (bench_record.get("top5") or [{}])[0].get("overlap", 0) if bench_record.get("top5") else 0,
        "detector_found": detector_found,
        "bm_rank_A": bm_rank_A,
        "bm_rank_B": bm_rank_B,
        "top1_A_site": top1_A["pocket_id"],
        "top1_A_class": top1_A["therm_class"],
        "top1_A_correct": top1_A_is_true,
        "top1_B_site": top1_B["pocket_id"],
        "top1_B_class": top1_B["therm_class"],
        "top1_B_correct": top1_B_is_true,
        "fp_top1_A": fp_top1_A,
        "fp_top1_B": fp_top1_B,
        "n_pockets": len(pockets),
    }


def main():
    bench = json.loads(CB_BENCH_JSON.read_text())
    per_protein = bench["per_protein"]

    rows = []
    for rec in per_protein:
        r = audit_one(rec)
        if r is not None:
            rows.append(r)

    def sr(k: int, key: str) -> float:
        return round(sum(1 for r in rows if r[key] is not None and r[key] <= k) / len(rows), 3) if rows else 0.0

    def count_top1_correct(key: str) -> int:
        return sum(1 for r in rows if r[key])

    def count_fp(key: str) -> int:
        return sum(1 for r in rows if r[key])

    median_rank_A = median([r["bm_rank_A"] for r in rows if r["bm_rank_A"] is not None]) if rows else None
    median_rank_B = median([r["bm_rank_B"] for r in rows if r["bm_rank_B"] is not None]) if rows else None

    agg = {
        "n_targets_with_data": len(rows),
        "A_current": {
            "top1_correct": count_top1_correct("top1_A_correct"),
            "fp_top1_promotions": count_fp("fp_top1_A"),
            "SR@3": sr(3, "bm_rank_A"),
            "SR@5": sr(5, "bm_rank_A"),
            "SR@10": sr(10, "bm_rank_A"),
            "median_bm_rank": median_rank_A,
        },
        "B_no_therm": {
            "top1_correct": count_top1_correct("top1_B_correct"),
            "fp_top1_promotions": count_fp("fp_top1_B"),
            "SR@3": sr(3, "bm_rank_B"),
            "SR@5": sr(5, "bm_rank_B"),
            "SR@10": sr(10, "bm_rank_B"),
            "median_bm_rank": median_rank_B,
        },
    }

    OUT_JSON.write_text(json.dumps({"rows": rows, "aggregate": agg}, indent=2, default=str))
    with OUT_CSV.open("w") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else [])
        w.writeheader()
        for r in rows:
            w.writerow(r)

    print("=" * 110)
    print(f"EXPANDED AUDIT — {len(rows)} CryptoBench targets with binding_sites + prism_therm")
    print("=" * 110)
    print("| {:<7} | {:>6} | {:>4} | {:>4} | {:>9} | {:<10} | {:>9} | {:<10} | {:>4} | {:>4} |".format(
        "apo", "bm_pid", "rA", "rB", "top1A_pid", "top1A_cls", "top1B_pid", "top1B_cls", "fpA", "fpB"))
    print("| " + " | ".join(["-" * w for w in [7, 6, 4, 4, 9, 10, 9, 10, 4, 4]]) + " |")
    for r in rows:
        print("| {:<7} | {:>6} | {:>4} | {:>4} | {:>9} | {:<10} | {:>9} | {:<10} | {:>4} | {:>4} |".format(
            r["apo_pdb"], str(r["best_site_by_overlap"]),
            str(r["bm_rank_A"] if r["bm_rank_A"] is not None else "-"),
            str(r["bm_rank_B"] if r["bm_rank_B"] is not None else "-"),
            str(r["top1_A_site"]), r["top1_A_class"] or "-",
            str(r["top1_B_site"]), r["top1_B_class"] or "-",
            "Y" if r["fp_top1_A"] else "n",
            "Y" if r["fp_top1_B"] else "n",
        ))
    print()
    print("=" * 110)
    print("AGGREGATE (expanded panel)")
    print("=" * 110)
    print(f"  n_targets = {len(rows)}")
    print(f"  {'A_current':<14} top1_correct={agg['A_current']['top1_correct']}/{len(rows)}  fp_top1={agg['A_current']['fp_top1_promotions']}  "
          f"SR@3={agg['A_current']['SR@3']}  SR@5={agg['A_current']['SR@5']}  SR@10={agg['A_current']['SR@10']}  "
          f"median_bm_rank={agg['A_current']['median_bm_rank']}")
    print(f"  {'B_no_therm':<14} top1_correct={agg['B_no_therm']['top1_correct']}/{len(rows)}  fp_top1={agg['B_no_therm']['fp_top1_promotions']}  "
          f"SR@3={agg['B_no_therm']['SR@3']}  SR@5={agg['B_no_therm']['SR@5']}  SR@10={agg['B_no_therm']['SR@10']}  "
          f"median_bm_rank={agg['B_no_therm']['median_bm_rank']}")
    print()
    print(f"report: {OUT_JSON}")
    print(f"csv:    {OUT_CSV}")


if __name__ == "__main__":
    main()
