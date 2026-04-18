#!/usr/bin/env python3
"""B_no_therm feature-gap audit + B_no_therm truth matrix.

Pre-declared criteria (declared before execution per constitution §3):
  * WEIGHTS_B = {drug: 0.50, therm: 0.00, spike: 0.25, tide: 0.25}
  * DCC match threshold = 8.0 Å (from stage 7 default)
  * TIDE_SATURATION = 20
  * Residue convention (where applicable downstream) = one_indexed
  * GT-valid target set = WRN / MENIN / SMARCA2 / PKMYT1 (strict DCC tier only)
  * Evidence tier = A (strict spatial / DCC-based paired apo-holo)

Does NOT mix with tier B (CryptoBench residue-overlap).

Report-layer semantic split (per §12):
  raw therm_class           → report_layer_alias
  ---------------             ---------------
  CRYPTIC                    → MECHANICAL_HOTSPOT
  DYNAMIC                    → DYNAMIC
  RESPONSIVE                 → RESPONSIVE
  INERT                      → INERT

Raw engine schema unchanged. Alias applied only to report outputs.

Read-only. Writes /tmp/b_no_therm_feature_gap.json.
"""
from __future__ import annotations
import json
import math
from pathlib import Path
from statistics import mean

BASE = Path("/mnt/storage/prism-outputs/twin-10-patent")
GT_VALID = ["wrn_apo", "menin_apo", "smarca2_brd_apo", "pkmyt1_apo"]

TIDE_SATURATION = 20
DCC_MATCH_A = 8.0
WEIGHTS_B = {"drug": 0.50, "therm": 0.00, "spike": 0.25, "tide": 0.25}

REPORT_ALIAS = {
    "CRYPTIC": "MECHANICAL_HOTSPOT",
    "DYNAMIC": "DYNAMIC",
    "RESPONSIVE": "RESPONSIVE",
    "INERT": "INERT",
    None: "UNLABELED",
}

OUT = Path("/tmp/b_no_therm_feature_gap.json")

FEATURES = [
    "druggability_score",
    "site_volume_angstrom_cubed",
    "n_spikes_attributed",
    "n_tide_residues_derived",
    "ccns_tau",
    "hysteresis_asymmetry",
    "relative_asymmetry",
    "spike_score_normalized_B",
    "tide_score_normalized",
    "drug_score_normalized",
    "composite_B",
]


def alias(tc):
    return REPORT_ALIAS.get((tc or "").upper() if tc else None, "UNLABELED")


def compute_B(pockets):
    max_spikes = max((p.get("n_spikes_attributed") or 0) for p in pockets) or 1
    log_spike_sat = math.log(1.0 + float(max_spikes))
    log_tide_sat = math.log(1.0 + TIDE_SATURATION)
    for p in pockets:
        drug = max(0.0, min(1.0, float(p.get("druggability_score") or 0.0)))
        n_spk = float(p.get("n_spikes_attributed") or 0)
        spike = math.log(1.0 + n_spk) / log_spike_sat if log_spike_sat > 0 else 0.0
        spike = max(0.0, min(1.0, spike))
        n_tide = len(p.get("top_residue_ids") or [])
        tide = math.log(1.0 + n_tide) / log_tide_sat
        tide = max(0.0, min(1.0, tide))
        p["drug_score_normalized"] = drug
        p["spike_score_normalized_B"] = spike
        p["tide_score_normalized"] = tide
        p["n_tide_residues_derived"] = n_tide
        p["composite_B"] = (WEIGHTS_B["drug"] * drug + WEIGHTS_B["spike"] * spike + WEIGHTS_B["tide"] * tide)


def rank_B(pockets):
    compute_B(pockets)
    ranked = sorted(pockets, key=lambda p: -p.get("composite_B", 0))
    for i, p in enumerate(ranked, 1):
        p["rerank_rank_B"] = i
    return ranked


def dcc_for(pid, ev):
    for s in ev.get("per_site_eval", []):
        if s.get("site_id") == pid:
            return s.get("centroid_dcc_angstrom")
    if ev.get("best_match", {}).get("site_id") == pid:
        return ev["best_match"].get("centroid_dcc_angstrom")
    return None


def pocket_by_id(ranked, pid):
    for p in ranked:
        if p.get("pocket_id") == pid:
            return p
    return None


def dominant_feature_gap(best_match, top1_B):
    if best_match is None or top1_B is None:
        return None
    pairs = {
        "drug_score_normalized": top1_B["drug_score_normalized"] - best_match["drug_score_normalized"],
        "spike_score_normalized_B": top1_B["spike_score_normalized_B"] - best_match["spike_score_normalized_B"],
        "tide_score_normalized": top1_B["tide_score_normalized"] - best_match["tide_score_normalized"],
    }
    weighted = {
        "drug": WEIGHTS_B["drug"] * pairs["drug_score_normalized"],
        "spike": WEIGHTS_B["spike"] * pairs["spike_score_normalized_B"],
        "tide": WEIGHTS_B["tide"] * pairs["tide_score_normalized"],
    }
    dominant = max(weighted, key=lambda k: weighted[k])
    return {
        "raw_deltas_top1_minus_truth": pairs,
        "weighted_contributions_top1_minus_truth": weighted,
        "dominant_cause_of_B_top1_over_truth": dominant,
    }


def audit_target(target):
    rr = json.loads((BASE / target / "artifacts/6_rerank/rerank_result.json").read_text())
    ev = json.loads((BASE / target / "artifacts/7_evaluation/evaluation.json").read_text())
    pockets = rr["merged_pockets"]

    ranked = rank_B(pockets)
    top1_B = ranked[0]
    top1_B_pid = top1_B.get("pocket_id")
    top1_B_dcc = dcc_for(top1_B_pid, ev)

    bm = ev.get("best_match") or {}
    bm_pid = bm.get("site_id")
    bm_dcc = bm.get("centroid_dcc_angstrom")

    bm_pocket = pocket_by_id(ranked, bm_pid) if bm_pid is not None else None
    bm_rank_B = bm_pocket.get("rerank_rank_B") if bm_pocket is not None else None

    detector_found = (bm_dcc is not None) and (bm_dcc < DCC_MATCH_A)
    top1_B_verified = (top1_B_dcc is not None) and (top1_B_dcc < DCC_MATCH_A)
    in_top_k = {
        "top3": (bm_rank_B is not None) and (bm_rank_B <= 3),
        "top5": (bm_rank_B is not None) and (bm_rank_B <= 5),
        "top10": (bm_rank_B is not None) and (bm_rank_B <= 10),
    }

    feature_deltas = dominant_feature_gap(bm_pocket, top1_B)

    truth_row = {
        "target": target,
        "paired_holo_pdb_id": ev.get("paired_holo_pdb_id"),
        "detector_found_true_site": detector_found,
        "B_top1_verified": top1_B_verified,
        "true_site_in_top3_under_B": in_top_k["top3"],
        "true_site_in_top5_under_B": in_top_k["top5"],
        "true_site_in_top10_under_B": in_top_k["top10"],
        "true_site_therm_class_raw": bm.get("therm_class"),
        "true_site_report_alias": alias(bm.get("therm_class")),
        "top1_B_therm_class_raw": top1_B.get("therm_class"),
        "top1_B_report_alias": alias(top1_B.get("therm_class")),
        "main_feature_causing_demotion": (feature_deltas or {}).get("dominant_cause_of_B_top1_over_truth"),
        "best_match_site_id": bm_pid,
        "best_match_dcc_angstrom": round(bm_dcc, 3) if bm_dcc is not None else None,
        "best_match_rank_under_B": bm_rank_B,
        "top1_B_site_id": top1_B_pid,
        "top1_B_dcc_angstrom": round(top1_B_dcc, 3) if top1_B_dcc is not None else None,
    }

    if bm_pocket is not None and top1_B is not None:
        feat_rows = []
        for f in FEATURES:
            v_truth = bm_pocket.get(f)
            v_top1 = top1_B.get(f)
            if isinstance(v_truth, (int, float)) and isinstance(v_top1, (int, float)):
                delta = v_top1 - v_truth
            else:
                delta = None
            feat_rows.append({
                "feature": f,
                "truth_value": round(v_truth, 4) if isinstance(v_truth, (int, float)) else v_truth,
                "top1_B_value": round(v_top1, 4) if isinstance(v_top1, (int, float)) else v_top1,
                "delta_top1_minus_truth": round(delta, 4) if delta is not None else None,
            })
    else:
        feat_rows = []

    return {
        "target": target,
        "feature_rows": feat_rows,
        "feature_deltas_summary": feature_deltas,
        "truth_row": truth_row,
    }


def main():
    all_results = []
    for t in GT_VALID:
        try:
            all_results.append(audit_target(t))
        except FileNotFoundError as e:
            print(f"FAIL: {t} missing artifact: {e}")

    print("=" * 96)
    print("RAW ARTIFACT A: B_no_therm feature-gap — per target")
    print("=" * 96)
    for r in all_results:
        target = r["target"]
        tr = r["truth_row"]
        print(f"\n— {target} — paired_holo={tr['paired_holo_pdb_id']}")
        print(f"  true_site: pid={tr['best_match_site_id']} dcc={tr['best_match_dcc_angstrom']} Å "
              f"raw_class={tr['true_site_therm_class_raw']} alias={tr['true_site_report_alias']} "
              f"B_rank={tr['best_match_rank_under_B']}")
        print(f"  B_top1:    pid={tr['top1_B_site_id']} dcc={tr['top1_B_dcc_angstrom']} "
              f"raw_class={tr['top1_B_therm_class_raw']} alias={tr['top1_B_report_alias']}")
        print(f"  {'feature':<32} {'truth':>14} {'B_top1':>14} {'Δ(top1-truth)':>14}")
        print(f"  {'-'*32} {'-'*14} {'-'*14} {'-'*14}")
        for row in r["feature_rows"]:
            v_t = row["truth_value"]
            v_1 = row["top1_B_value"]
            d = row["delta_top1_minus_truth"]
            v_t_s = f"{v_t:>14.4f}" if isinstance(v_t, (int, float)) else f"{str(v_t):>14}"
            v_1_s = f"{v_1:>14.4f}" if isinstance(v_1, (int, float)) else f"{str(v_1):>14}"
            d_s = f"{d:>+14.4f}" if isinstance(d, (int, float)) else f"{str(d):>14}"
            print(f"  {row['feature']:<32} {v_t_s} {v_1_s} {d_s}")
        summ = r["feature_deltas_summary"] or {}
        if summ:
            print(f"  weighted contributions (top1 − truth, B weights):")
            for k, v in summ["weighted_contributions_top1_minus_truth"].items():
                print(f"    {k:<6} = {v:>+.4f}")
            print(f"  dominant cause of B_top1 over truth: {summ['dominant_cause_of_B_top1_over_truth']}")

    print()
    print("=" * 96)
    print("RAW ARTIFACT B: B_no_therm aggregate cause-of-demotion")
    print("=" * 96)
    causes = [r["truth_row"]["main_feature_causing_demotion"]
              for r in all_results if r["truth_row"]["main_feature_causing_demotion"]]
    from collections import Counter
    cause_counts = Counter(causes)
    print("  cause_of_demotion_count =", dict(cause_counts))
    weighted_agg = {"drug": [], "spike": [], "tide": []}
    for r in all_results:
        summ = r["feature_deltas_summary"]
        if summ:
            for k, v in summ["weighted_contributions_top1_minus_truth"].items():
                weighted_agg[k].append(v)
    print("  avg_weighted_delta_top1_minus_truth (positive = feature helped the FP outrank the true site):")
    for k, vals in weighted_agg.items():
        if vals:
            print(f"    {k:<6}: mean={mean(vals):+.4f}  min={min(vals):+.4f}  max={max(vals):+.4f}")

    under_scored_true = sum(
        1 for r in all_results if r["feature_deltas_summary"] and
        r["feature_deltas_summary"]["dominant_cause_of_B_top1_over_truth"] != "drug"
    )
    fp_wins_drug = sum(1 for r in all_results if r["feature_deltas_summary"] and
                       r["feature_deltas_summary"]["dominant_cause_of_B_top1_over_truth"] == "drug")
    fp_wins_spike = sum(1 for r in all_results if r["feature_deltas_summary"] and
                        r["feature_deltas_summary"]["dominant_cause_of_B_top1_over_truth"] == "spike")
    fp_wins_tide = sum(1 for r in all_results if r["feature_deltas_summary"] and
                       r["feature_deltas_summary"]["dominant_cause_of_B_top1_over_truth"] == "tide")
    print(f"  FP top-1 wins on DRUG: {fp_wins_drug}/{len(all_results)}")
    print(f"  FP top-1 wins on SPIKE: {fp_wins_spike}/{len(all_results)}")
    print(f"  FP top-1 wins on TIDE: {fp_wins_tide}/{len(all_results)}")

    print()
    print("=" * 96)
    print("RAW ARTIFACT C: B_no_therm truth matrix (tier A — strict DCC)")
    print("=" * 96)
    hdr = ("target", "det_found", "B_top1_verif", "≤3", "≤5", "≤10", "true_alias", "top1_B_alias", "main_cause")
    print(("| {:<18} | {:>9} | {:>12} | {:>3} | {:>3} | {:>4} | {:<17} | {:<17} | {:<10} |").format(*hdr))
    print("| " + " | ".join(["-" * w for w in [18, 9, 12, 3, 3, 4, 17, 17, 10]]) + " |")
    for r in all_results:
        tr = r["truth_row"]
        print(("| {:<18} | {:>9} | {:>12} | {:>3} | {:>3} | {:>4} | {:<17} | {:<17} | {:<10} |").format(
            tr["target"],
            "PASS" if tr["detector_found_true_site"] else "FAIL",
            "PASS" if tr["B_top1_verified"] else "FAIL",
            "Y" if tr["true_site_in_top3_under_B"] else "n",
            "Y" if tr["true_site_in_top5_under_B"] else "n",
            "Y" if tr["true_site_in_top10_under_B"] else "n",
            tr["true_site_report_alias"] or "-",
            tr["top1_B_report_alias"] or "-",
            tr["main_feature_causing_demotion"] or "-",
        ))
    print()

    OUT.write_text(json.dumps(all_results, indent=2, default=str))
    print(f"report: {OUT}")


if __name__ == "__main__":
    main()
