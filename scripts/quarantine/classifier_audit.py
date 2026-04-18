#!/usr/bin/env python3
"""Classifier/ranker failure audit + 3-way ablation on GT-valid targets.

Read-only. Writes only /tmp/classifier_audit.json plus stdout tables.
No changes to stage 6/7 or engine.

Ablations (all keep sum of weights = 1.0):
  A current  : {drug: 0.40, therm: 0.20, spike: 0.20, tide: 0.20}    therm = categorical THERM_MAP
  B no_therm : {drug: 0.50, therm: 0.00, spike: 0.25, tide: 0.25}    no therm term
  C cont_therm: {drug: 0.40, therm: 0.20, spike: 0.20, tide: 0.20}   therm = continuous from (hyst, tau, rel_asym)

No overlay. C REPLACES categorical therm; it does not double-count.
"""
from __future__ import annotations
import json
import math
import sys
from pathlib import Path
from statistics import mean, median

BASE = Path("/mnt/storage/prism-outputs/twin-10-patent")
GT_VALID_TARGETS = ["wrn_apo", "menin_apo", "smarca2_brd_apo", "pkmyt1_apo"]
GT_INVALID_TARGETS = ["polq_apo", "usp1_apo"]
NO_GT_TARGETS = ["trip12_hect_apo", "trex1_apo"]
FAILED_TARGETS = ["kras_g12d_apo"]

THERM_MAP = {"CRYPTIC": 1.0, "DYNAMIC": 0.7, "RESPONSIVE": 0.4, "INERT": 0.1}
TIDE_SATURATION = 20
DCC_MATCH_A = 8.0

OUT = Path("/tmp/classifier_audit.json")


def min_max(vals):
    v = [float(x) for x in vals if x is not None]
    if not v:
        return lambda _: 0.0
    lo, hi = min(v), max(v)
    span = hi - lo
    if span <= 1e-12:
        return lambda x: 0.5
    return lambda x: max(0.0, min(1.0, (float(x) - lo) / span))


def rerank(pockets: list[dict], weights: dict, therm_fn) -> list[dict]:
    max_spikes = max((p.get("n_spikes_attributed") or 0) for p in pockets) or 1
    log_spike_sat = math.log(1.0 + float(max_spikes))
    log_tide_sat = math.log(1.0 + TIDE_SATURATION)

    scored = []
    for p in pockets:
        drug = max(0.0, min(1.0, float(p.get("druggability_score") or 0.0)))
        therm = therm_fn(p)
        n_spk = float(p.get("n_spikes_attributed") or 0)
        spike = math.log(1.0 + n_spk) / log_spike_sat if log_spike_sat > 0 else 0.0
        spike = max(0.0, min(1.0, spike))
        n_tide = len(p.get("top_residue_ids") or [])
        tide = math.log(1.0 + n_tide) / log_tide_sat
        tide = max(0.0, min(1.0, tide))

        composite = (weights["drug"] * drug
                     + weights["therm"] * therm
                     + weights["spike"] * spike
                     + weights["tide"] * tide)
        scored.append({
            "pocket_id": p.get("pocket_id"),
            "therm_class": p.get("therm_class"),
            "engine_rank": p.get("engine_rank"),
            "composite": composite,
            "drug": drug, "therm": therm, "spike": spike, "tide": tide,
        })
    scored.sort(key=lambda s: -s["composite"])
    for i, s in enumerate(scored, 1):
        s["rank"] = i
    return scored


def load_target(target: str) -> dict:
    rr = json.loads((BASE / target / "artifacts/6_rerank/rerank_result.json").read_text())
    ev = json.loads((BASE / target / "artifacts/7_evaluation/evaluation.json").read_text())
    return {"rerank": rr, "eval": ev}


def build_continuous_therm_fn(pockets: list[dict]):
    hy_norm = min_max([p.get("hysteresis_asymmetry") for p in pockets])
    tau_norm = min_max([p.get("ccns_tau") for p in pockets])
    ra_norm = min_max([p.get("relative_asymmetry") for p in pockets])

    def fn(p):
        h = hy_norm(p.get("hysteresis_asymmetry") or 0.0)
        t = tau_norm(p.get("ccns_tau") or 0.0)
        r = ra_norm(p.get("relative_asymmetry") or 0.0)
        return (h + t + r) / 3.0
    return fn


def dcc_of(pid, ev):
    for s in ev.get("per_site_eval", []):
        if s.get("site_id") == pid:
            return s.get("centroid_dcc_angstrom")
    return None


def audit_target(target: str) -> dict:
    data = load_target(target)
    pockets = data["rerank"]["merged_pockets"]
    ev = data["eval"]
    bm = ev.get("best_match") or {}
    bm_pid = bm.get("site_id")
    bm_dcc = bm.get("centroid_dcc_angstrom")
    bm_class = bm.get("therm_class")
    bm_eng_rank = None
    bm_rerank_rank = None
    for p in pockets:
        if p.get("pocket_id") == bm_pid:
            bm_eng_rank = p.get("engine_rank")
            bm_rerank_rank = p.get("rerank_rank")
            break

    top1 = sorted(pockets, key=lambda p: -p.get("rerank_composite", 0))[0]
    top1_pid = top1.get("pocket_id")
    top1_class = top1.get("therm_class")
    top1_dcc = dcc_of(top1_pid, ev)

    detected = (bm_dcc is not None) and (bm_dcc < DCC_MATCH_A)
    if detected:
        if bm_rerank_rank == 1:
            verdict = "TRUE_SITE_TOP1"
        elif top1_class == "CRYPTIC" and (top1_dcc is None or top1_dcc > DCC_MATCH_A):
            verdict = "TOP1_FALSE_POSITIVE"
        else:
            verdict = "TRUE_SITE_FOUND_BUT_DEMOTED"
    else:
        verdict = "NOT_FOUND"

    truth = {
        "detector_found_true_site": detected,
        "reranker_improved_true_site_rank": (
            (bm_eng_rank is not None) and (bm_rerank_rank is not None) and (bm_rerank_rank < bm_eng_rank)
        ),
        "reranker_promoted_false_site_to_top1": (
            detected and (bm_rerank_rank is not None) and (bm_rerank_rank != 1)
        ),
        "typing_correct_on_true_site": (
            detected and (ev.get("expected_class") is None or ev.get("observed_class") == ev.get("expected_class"))
        ),
    }

    return {
        "target": target,
        "pdb_id": ev.get("pdb_id"),
        "paired_holo": ev.get("paired_holo_pdb_id"),
        "best_match_site_id": bm_pid,
        "best_match_engine_rank": bm_eng_rank,
        "best_match_rerank_rank": bm_rerank_rank,
        "best_match_therm_class": bm_class,
        "best_match_dcc_angstrom": round(bm_dcc, 3) if bm_dcc is not None else None,
        "top1_rerank_site_id": top1_pid,
        "top1_rerank_therm_class": top1_class,
        "top1_dcc_angstrom": round(top1_dcc, 3) if top1_dcc is not None else None,
        "verdict": verdict,
        "true_site_found": detected,
        "true_site_demoted": (detected and bm_rerank_rank is not None and bm_rerank_rank > 1),
        "false_positive_cryptic_outranked_truth": (
            detected and top1_class == "CRYPTIC" and (top1_dcc is None or top1_dcc > DCC_MATCH_A)
        ),
        "truth_matrix": truth,
        "pockets": pockets,
        "eval": ev,
    }


def ablation_pass(row: dict, label: str, weights: dict, therm_fn) -> dict:
    pockets = row["pockets"]
    ev = row["eval"]
    ranked = rerank(pockets, weights, therm_fn)
    bm_pid = row["best_match_site_id"]
    bm_dcc = row["best_match_dcc_angstrom"]
    if bm_pid is None or bm_dcc is None:
        return {}

    bm_new_rank = None
    for s in ranked:
        if s["pocket_id"] == bm_pid:
            bm_new_rank = s["rank"]
            break

    top1 = ranked[0]
    top1_dcc = dcc_of(top1["pocket_id"], ev)
    top1_match = (top1_dcc is not None) and (top1_dcc < DCC_MATCH_A)
    fp_top1 = (not top1_match) and (top1["therm_class"] == "CRYPTIC")
    demoted = (bm_new_rank is not None) and (bm_new_rank > 1)
    sr3 = (bm_new_rank is not None) and (bm_new_rank <= 3)

    return {
        "ablation": label,
        "weights": weights,
        "top1_site_id": top1["pocket_id"],
        "top1_therm_class": top1["therm_class"],
        "top1_dcc_angstrom": round(top1_dcc, 3) if top1_dcc is not None else None,
        "top1_match_within_8A": top1_match,
        "best_match_new_rank": bm_new_rank,
        "best_match_in_top3": sr3,
        "false_positive_top1": fp_top1,
        "true_site_demoted": demoted,
    }


def aggregate_counts(rows: list[dict]) -> dict:
    n_top1_fp = sum(1 for r in rows if r["verdict"] == "TOP1_FALSE_POSITIVE")
    n_demoted = sum(1 for r in rows if r["verdict"] == "TRUE_SITE_FOUND_BUT_DEMOTED")
    n_top1_true = sum(1 for r in rows if r["verdict"] == "TRUE_SITE_TOP1")
    n_not_found = sum(1 for r in rows if r["verdict"] == "NOT_FOUND")

    fp_rows = [r for r in rows if r["verdict"] in ("TOP1_FALSE_POSITIVE", "TRUE_SITE_FOUND_BUT_DEMOTED")]
    fp_class_cryptic = sum(1 for r in fp_rows if r["top1_rerank_therm_class"] == "CRYPTIC")
    frac_fp_cryptic = (fp_class_cryptic / len(fp_rows)) if fp_rows else 0.0

    true_rows = [r for r in rows if r["true_site_found"]]
    true_non_cryptic = sum(1 for r in true_rows if r["best_match_therm_class"] in ("RESPONSIVE", "DYNAMIC", "INERT"))
    frac_true_non_cryptic = (true_non_cryptic / len(true_rows)) if true_rows else 0.0

    fp_shifts = []
    for r in rows:
        if r["verdict"] == "TOP1_FALSE_POSITIVE":
            top1_pid = r["top1_rerank_site_id"]
            for p in r["pockets"]:
                if p.get("pocket_id") == top1_pid:
                    rs = p.get("rank_shift")
                    if isinstance(rs, int):
                        fp_shifts.append(rs)
                    break
    avg_fp_shift = mean(fp_shifts) if fp_shifts else None

    demotions = []
    for r in rows:
        if r["true_site_found"] and r["best_match_rerank_rank"] is not None and r["best_match_engine_rank"] is not None:
            demotions.append(r["best_match_rerank_rank"] - r["best_match_engine_rank"])
    avg_true_demotion = mean(demotions) if demotions else None

    return {
        "n_targets": len(rows),
        "TOP1_FALSE_POSITIVE": n_top1_fp,
        "TRUE_SITE_FOUND_BUT_DEMOTED": n_demoted,
        "TRUE_SITE_TOP1": n_top1_true,
        "NOT_FOUND": n_not_found,
        "fraction_fp_top1_with_CRYPTIC_class": round(frac_fp_cryptic, 3),
        "fraction_true_sites_with_non_CRYPTIC_class": round(frac_true_non_cryptic, 3),
        "avg_rank_shift_of_false_positive_cryptic_promotions": round(avg_fp_shift, 2) if avg_fp_shift is not None else None,
        "avg_rerank_demotion_of_true_sites": round(avg_true_demotion, 2) if avg_true_demotion is not None else None,
    }


def aggregate_ablation(all_ablations: list[list[dict]]) -> dict:
    # all_ablations is list of [ablation_A_row, ablation_B_row, ablation_C_row] per target
    by_label = {"A_current": [], "B_no_therm": [], "C_cont_therm": []}
    for per_target in all_ablations:
        for ab in per_target:
            by_label[ab["ablation"]].append(ab)

    agg = {}
    for label, rows in by_label.items():
        top1_correct = sum(1 for r in rows if r.get("top1_match_within_8A"))
        fp_top1 = sum(1 for r in rows if r.get("false_positive_top1"))
        true_demoted = sum(1 for r in rows if r.get("true_site_demoted"))
        ranks = [r.get("best_match_new_rank") for r in rows if r.get("best_match_new_rank") is not None]
        sr3 = sum(1 for r in rows if r.get("best_match_in_top3")) / len(rows) if rows else 0.0
        agg[label] = {
            "n_targets": len(rows),
            "top1_correct": top1_correct,
            "fp_top1_promotions": fp_top1,
            "true_site_demotions": true_demoted,
            "best_match_median_rank": median(ranks) if ranks else None,
            "best_match_ranks": ranks,
            "SR_at_3": round(sr3, 3),
        }
    return agg


def main():
    audit_rows = [audit_target(t) for t in GT_VALID_TARGETS]
    aggregate = aggregate_counts(audit_rows)

    all_ablations = []
    for row in audit_rows:
        pockets = row["pockets"]

        def cat_therm(p):
            return THERM_MAP.get((p.get("therm_class") or "").upper(), 0.0)

        cont_fn = build_continuous_therm_fn(pockets)

        abA = ablation_pass(row, "A_current", {"drug": 0.40, "therm": 0.20, "spike": 0.20, "tide": 0.20}, cat_therm)
        abB = ablation_pass(row, "B_no_therm", {"drug": 0.50, "therm": 0.00, "spike": 0.25, "tide": 0.25},
                            lambda p: 0.0)
        abC = ablation_pass(row, "C_cont_therm", {"drug": 0.40, "therm": 0.20, "spike": 0.20, "tide": 0.20}, cont_fn)
        all_ablations.append([abA, abB, abC])

    ablation_agg = aggregate_ablation(all_ablations)

    # Print raw audit table
    print("=" * 110)
    print("RAW AUDIT TABLE (GT-valid targets)")
    print("=" * 110)
    hdr = ("target", "gt_pdb", "bm_site", "bm_eng_r", "bm_rerank_r", "bm_class", "bm_dcc", "top1_site", "top1_class", "verdict")
    print("| {:<18} | {:<6} | {:>7} | {:>8} | {:>11} | {:<10} | {:>6} | {:>9} | {:<10} | {:<35} |".format(*hdr))
    print("| " + " | ".join(["-" * w for w in [18, 6, 7, 8, 11, 10, 6, 9, 10, 35]]) + " |")
    for r in audit_rows:
        print("| {:<18} | {:<6} | {:>7} | {:>8} | {:>11} | {:<10} | {:>6.2f} | {:>9} | {:<10} | {:<35} |".format(
            r["target"], r["paired_holo"] or "-",
            str(r["best_match_site_id"] or "-"),
            str(r["best_match_engine_rank"] or "-"),
            str(r["best_match_rerank_rank"] or "-"),
            r["best_match_therm_class"] or "-",
            r["best_match_dcc_angstrom"] if r["best_match_dcc_angstrom"] is not None else 0.0,
            str(r["top1_rerank_site_id"] or "-"),
            r["top1_rerank_therm_class"] or "-",
            r["verdict"],
        ))
    print()

    print("=" * 110)
    print("AGGREGATE COUNTS")
    print("=" * 110)
    for k, v in aggregate.items():
        print(f"  {k:<55} = {v}")
    print()

    print("=" * 110)
    print("PER-TARGET TRUTH MATRIX")
    print("=" * 110)
    print("| {:<18} | {:<27} | {:<31} | {:<34} | {:<28} |".format(
        "target", "detector_found_true_site", "reranker_improved_true_rank",
        "reranker_promoted_false_top1", "typing_correct_true_site"))
    print("| " + " | ".join(["-" * w for w in [18, 27, 31, 34, 28]]) + " |")
    for r in audit_rows:
        tm = r["truth_matrix"]
        print("| {:<18} | {:<27} | {:<31} | {:<34} | {:<28} |".format(
            r["target"],
            "yes" if tm["detector_found_true_site"] else "no",
            "yes" if tm["reranker_improved_true_site_rank"] else "no",
            "yes" if tm["reranker_promoted_false_site_to_top1"] else "no",
            "yes" if tm["typing_correct_on_true_site"] else "no",
        ))
    print()

    print("=" * 110)
    print("ABLATION RESULTS — stage 6 only, same engine output")
    print("=" * 110)
    print("Per-target best-match new rank + top1 outcome:")
    print("| {:<18} | {:>8} | {:>8} | {:>8} | {:>14} | {:>14} | {:>14} |".format(
        "target", "A_rank", "B_rank", "C_rank", "A_top1_match", "B_top1_match", "C_top1_match"))
    print("| " + " | ".join(["-" * w for w in [18, 8, 8, 8, 14, 14, 14]]) + " |")
    for row, abs_ in zip(audit_rows, all_ablations):
        rA, rB, rC = abs_
        print("| {:<18} | {:>8} | {:>8} | {:>8} | {:>14} | {:>14} | {:>14} |".format(
            row["target"],
            str(rA.get("best_match_new_rank") or "-"),
            str(rB.get("best_match_new_rank") or "-"),
            str(rC.get("best_match_new_rank") or "-"),
            "yes" if rA.get("top1_match_within_8A") else "no",
            "yes" if rB.get("top1_match_within_8A") else "no",
            "yes" if rC.get("top1_match_within_8A") else "no",
        ))
    print()
    print("Aggregate per ablation:")
    print("| {:<14} | {:>9} | {:>12} | {:>12} | {:>9} | {:>18} |".format(
        "ablation", "n_targets", "top1_correct", "fp_top1_prom", "SR@3", "median_bm_rank"))
    print("| " + " | ".join(["-" * w for w in [14, 9, 12, 12, 9, 18]]) + " |")
    for label in ("A_current", "B_no_therm", "C_cont_therm"):
        a = ablation_agg[label]
        print("| {:<14} | {:>9} | {:>12} | {:>12} | {:>9} | {:>18} |".format(
            label, a["n_targets"], a["top1_correct"], a["fp_top1_promotions"], a["SR_at_3"],
            str(a["best_match_median_rank"]) if a["best_match_median_rank"] is not None else "-"))
    print()

    # Recommendation — decision rule only, no prose
    reco_lines = []
    A = ablation_agg["A_current"]
    B = ablation_agg["B_no_therm"]
    C = ablation_agg["C_cont_therm"]
    reco_lines.append(f"A: top1_correct={A['top1_correct']}/{A['n_targets']}  fp_top1={A['fp_top1_promotions']}  SR@3={A['SR_at_3']}")
    reco_lines.append(f"B: top1_correct={B['top1_correct']}/{B['n_targets']}  fp_top1={B['fp_top1_promotions']}  SR@3={B['SR_at_3']}")
    reco_lines.append(f"C: top1_correct={C['top1_correct']}/{C['n_targets']}  fp_top1={C['fp_top1_promotions']}  SR@3={C['SR_at_3']}")

    def rank(ab):
        return (-ab["top1_correct"], ab["fp_top1_promotions"], -ab["SR_at_3"])
    winner_label = min(("A_current", "B_no_therm", "C_cont_therm"), key=lambda k: rank(ablation_agg[k]))
    reco_lines.append(f"RECOMMENDATION: {winner_label} (by primary top1_correct, then fewer fp_top1, then higher SR@3)")

    print("=" * 110)
    print("RECOMMENDATION (numerical ranking)")
    print("=" * 110)
    for line in reco_lines:
        print("  " + line)

    record = {
        "gt_valid_targets": GT_VALID_TARGETS,
        "gt_invalid_targets": GT_INVALID_TARGETS,
        "no_gt_targets": NO_GT_TARGETS,
        "failed_targets": FAILED_TARGETS,
        "audit": [{k: v for k, v in r.items() if k not in ("pockets", "eval")} for r in audit_rows],
        "aggregate": aggregate,
        "ablations_per_target": all_ablations,
        "ablation_aggregate": ablation_agg,
        "recommendation": winner_label,
    }
    OUT.write_text(json.dumps(record, indent=2, default=str))
    print()
    print(f"report: {OUT}")


if __name__ == "__main__":
    main()
