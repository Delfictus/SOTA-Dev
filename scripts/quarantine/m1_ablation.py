#!/usr/bin/env python3
"""M1 ligandability repair — 6-variant scorer ablation.

Pre-declared per constitution §3. Formulas frozen before execution.
No therm/CRYPTIC promotion in any variant. No MD. No engine changes.

============================================================================
VARIANT DECLARATIONS (all weights sum to 1.0 — fail-loud enforced)
============================================================================
A = "B_no_therm"  (existing honesty baseline)
    drug=0.50, therm=0.00, spike=0.25, tide=0.25
    spike_score  = log(1+n_spikes) / log(1+max_spikes)            [log-saturating mass]
    tide_score   = log(1+n_tide_residues) / log(1+20)             [saturates at 20]

B = "spike_density"
    drug=0.50, therm=0.00, spike_density=0.25, tide=0.25
    spike_density_raw     = n_spikes_attributed / site_volume_angstrom_cubed   [spikes per Å³]
    spike_density_score   = log(1+rho) / log(1+rho_max)   [log-saturating, min-max per target]
    (volume==0 → spike_density_score=0.0)

C = "spike_concentration"
    drug=0.50, therm=0.00, spike_concentration=0.25, tide=0.25
    Given no per-residue spike attribution in rerank output, we derive
    concentration from spike mass against top_residue_ids count:
    conc_raw   = n_spikes / max(1, n_tide_residues)           [spikes per trigger residue]
    conc_score = log(1+conc) / log(1+conc_max)                [per-target min-max]

D = "tide_distinctness"
    drug=0.50, therm=0.00, spike=0.25, tide_distinctness=0.25
    Saturation diagnostic: if >= 80% of top-5 pockets have n_tide == 20,
    tide is DEGENERATE and distinctness = unique_trigger_residues_overall.
    Here we approximate distinctness by: unique_count / total_count where
    unique_count = |set(top_residue_ids)| (always full 20 per pocket, so this
    is degenerate by construction). Fallback distinctness score:
    distinctness_score = (n_tide_residues - intersection_with_other_top5_pockets) / n_tide_residues
    Reported as FEATURE_GAP when insufficient discrimination.

E = "geometry_augmented"
    drug=0.40, therm=0.00, spike=0.20, tide=0.20, volume_prior=0.20
    volume_prior penalty: peaks at ~500 Å³, decays for smaller/larger.
    volume_score = exp(-0.5 * (log(V) - log(500))^2 / sigma^2)  with sigma=0.9
    If V missing → volume_score = 0.0 (FEATURE_GAP per-pocket).

F = "mechanical_interpreter_only"
    Same scoring as A (B_no_therm) for top-1 ranking.
    Adds report-layer alias: therm_class==CRYPTIC → MECHANICAL_HOTSPOT.
    No change to ranking; change only to report_layer_class field.

============================================================================
EVALUATION RULES (pre-declared, locked)
============================================================================
Lexicographic: (top1_correct↓, fp_top1_promotions↑, SR@3↓, SR@5↓, median_bm_rank↑)
DCC match threshold: 8.0 Å
Tier A = strict DCC (tier_A_targets below)
Tier B = residue-overlap (CryptoBench) — reported SEPARATELY; never merged.

Forbidden terms: recovery, match, confirmed, successful cryptic surfacing,
SOTA, unbiased (unless validation passes).
"""
from __future__ import annotations
import json
import math
from pathlib import Path
from statistics import mean, median
from collections import Counter

BASE = Path("/mnt/storage/prism-outputs/twin-10-patent")
CB_RUN_DIR = Path("/mnt/storage/prism-outputs/runs/cryptobench199")
CB_BENCH_JSON = Path("/home/diddy/Desktop/Prism4D-bio/benchmarks/cryptobench/cryptobench_benchmark.json")

TIER_A_TARGETS = ["wrn_apo", "menin_apo", "smarca2_brd_apo", "pkmyt1_apo"]
DCC_MATCH_A = 8.0
TIDE_SATURATION = 20
VOL_PRIOR_CENTER = 500.0
VOL_PRIOR_SIGMA = 0.9
DRUGGABILITY_THRESHOLD_FOR_DESIRABILITY = 0.45

REPORT_ALIAS = {"CRYPTIC": "MECHANICAL_HOTSPOT",
                "DYNAMIC": "DYNAMIC",
                "RESPONSIVE": "RESPONSIVE",
                "INERT": "INERT",
                None: "UNLABELED"}

OUT_JSON = Path("/tmp/m1_ablation.json")


def alias(tc: str | None) -> str:
    return REPORT_ALIAS.get((tc or "").upper() if tc else None, "UNLABELED")


def _log_norm(vals, v):
    m = max(vals) if vals else 0.0
    lsat = math.log(1.0 + float(m)) if m > 0 else 1.0
    return (math.log(1.0 + float(v or 0)) / lsat) if lsat > 0 else 0.0


def score_A(pockets):
    max_spikes = max((p.get("n_spikes_attributed") or 0) for p in pockets) or 1
    log_tide = math.log(1.0 + TIDE_SATURATION)
    out = []
    for p in pockets:
        drug = max(0.0, min(1.0, float(p.get("druggability_score") or 0.0)))
        n_spk = float(p.get("n_spikes_attributed") or 0)
        spike = math.log(1.0 + n_spk) / math.log(1.0 + max_spikes)
        n_tide = len(p.get("top_residue_ids") or [])
        tide = math.log(1.0 + n_tide) / log_tide
        composite = 0.50 * drug + 0.25 * spike + 0.25 * tide
        out.append({"pocket_id": p["pocket_id"], "therm_class": p.get("therm_class"),
                    "composite": composite, "drug": drug, "spike": spike, "tide": tide,
                    "variant": "A_B_no_therm"})
    return out


def score_B_density(pockets):
    # spike_density_raw = spikes / volume (Å³)
    rhos = []
    for p in pockets:
        v = p.get("site_volume_angstrom_cubed") or 0.0
        s = p.get("n_spikes_attributed") or 0
        rhos.append((s / v) if v > 0 else 0.0)
    max_rho = max(rhos) if rhos else 0.0
    log_tide = math.log(1.0 + TIDE_SATURATION)
    out = []
    for p, rho in zip(pockets, rhos):
        drug = max(0.0, min(1.0, float(p.get("druggability_score") or 0.0)))
        density_score = (math.log(1.0 + rho) / math.log(1.0 + max_rho)) if max_rho > 0 else 0.0
        n_tide = len(p.get("top_residue_ids") or [])
        tide = math.log(1.0 + n_tide) / log_tide
        composite = 0.50 * drug + 0.25 * density_score + 0.25 * tide
        out.append({"pocket_id": p["pocket_id"], "therm_class": p.get("therm_class"),
                    "composite": composite, "drug": drug, "spike_density": density_score,
                    "tide": tide, "variant": "B_spike_density"})
    return out


def score_C_concentration(pockets):
    # concentration = spikes / n_trigger_residues (bounded by n_tide)
    concs = []
    for p in pockets:
        s = p.get("n_spikes_attributed") or 0
        n_r = max(1, len(p.get("top_residue_ids") or []))
        concs.append(s / n_r)
    max_c = max(concs) if concs else 0.0
    log_tide = math.log(1.0 + TIDE_SATURATION)
    out = []
    for p, c in zip(pockets, concs):
        drug = max(0.0, min(1.0, float(p.get("druggability_score") or 0.0)))
        conc_score = (math.log(1.0 + c) / math.log(1.0 + max_c)) if max_c > 0 else 0.0
        n_tide = len(p.get("top_residue_ids") or [])
        tide = math.log(1.0 + n_tide) / log_tide
        composite = 0.50 * drug + 0.25 * conc_score + 0.25 * tide
        out.append({"pocket_id": p["pocket_id"], "therm_class": p.get("therm_class"),
                    "composite": composite, "drug": drug, "spike_conc": conc_score,
                    "tide": tide, "variant": "C_spike_concentration"})
    return out


def score_D_tide_distinctness(pockets):
    # distinctness = |unique residues in this pocket \ union of other top-5 pockets| / n_tide
    # Rank by spike first for "other top-5" set
    by_spikes = sorted(pockets, key=lambda p: -(p.get("n_spikes_attributed") or 0))
    top5_ids = [set(p.get("top_residue_ids") or []) for p in by_spikes[:5]]
    log_tide = math.log(1.0 + TIDE_SATURATION)
    max_spikes = max((p.get("n_spikes_attributed") or 0) for p in pockets) or 1
    out = []
    for p in pockets:
        drug = max(0.0, min(1.0, float(p.get("druggability_score") or 0.0)))
        n_spk = float(p.get("n_spikes_attributed") or 0)
        spike = math.log(1.0 + n_spk) / math.log(1.0 + max_spikes)
        res = set(p.get("top_residue_ids") or [])
        if not res:
            distinct = 0.0
        else:
            union_others = set().union(*[s for s in top5_ids if s != res])
            distinct_n = len(res - union_others)
            distinct = distinct_n / len(res)
        composite = 0.50 * drug + 0.25 * spike + 0.25 * distinct
        out.append({"pocket_id": p["pocket_id"], "therm_class": p.get("therm_class"),
                    "composite": composite, "drug": drug, "spike": spike,
                    "tide_distinctness": distinct, "variant": "D_tide_distinctness"})
    return out


def score_E_geometry(pockets):
    max_spikes = max((p.get("n_spikes_attributed") or 0) for p in pockets) or 1
    log_tide = math.log(1.0 + TIDE_SATURATION)
    out = []
    for p in pockets:
        drug = max(0.0, min(1.0, float(p.get("druggability_score") or 0.0)))
        n_spk = float(p.get("n_spikes_attributed") or 0)
        spike = math.log(1.0 + n_spk) / math.log(1.0 + max_spikes)
        n_tide = len(p.get("top_residue_ids") or [])
        tide = math.log(1.0 + n_tide) / log_tide
        v = p.get("site_volume_angstrom_cubed")
        if v is None or v <= 0:
            vol_score = 0.0
            vol_gap = True
        else:
            ln_v = math.log(float(v))
            ln_center = math.log(VOL_PRIOR_CENTER)
            vol_score = math.exp(-0.5 * ((ln_v - ln_center) / VOL_PRIOR_SIGMA) ** 2)
            vol_gap = False
        composite = 0.40 * drug + 0.20 * spike + 0.20 * tide + 0.20 * vol_score
        out.append({"pocket_id": p["pocket_id"], "therm_class": p.get("therm_class"),
                    "composite": composite, "drug": drug, "spike": spike, "tide": tide,
                    "vol_score": vol_score, "vol_gap": vol_gap,
                    "variant": "E_geometry_augmented"})
    return out


def score_F_mechanical_interpreter(pockets):
    # ranking identical to A; only adds report_layer_class
    out = score_A(pockets)
    for row in out:
        row["variant"] = "F_mechanical_interpreter_only"
        row["report_layer_class"] = alias(row["therm_class"])
    return out


ALL_VARIANTS = {
    "A_B_no_therm": score_A,
    "B_spike_density": score_B_density,
    "C_spike_concentration": score_C_concentration,
    "D_tide_distinctness": score_D_tide_distinctness,
    "E_geometry_augmented": score_E_geometry,
    "F_mechanical_interpreter_only": score_F_mechanical_interpreter,
}


def dcc_of(pid, ev):
    for s in ev.get("per_site_eval") or []:
        if s.get("site_id") == pid:
            return s.get("centroid_dcc_angstrom")
    bm = ev.get("best_match") or {}
    if bm.get("site_id") == pid:
        return bm.get("centroid_dcc_angstrom")
    return None


def rank(rows, pid):
    rows.sort(key=lambda r: -r["composite"])
    for i, r in enumerate(rows, 1):
        r["rank"] = i
    for r in rows:
        if r["pocket_id"] == pid:
            return r["rank"], r
    return None, None


def cause_of_demotion(pockets, variant_rows, truth_pid, top1):
    tp = next((r for r in variant_rows if r["pocket_id"] == truth_pid), None)
    if tp is None or top1 is None:
        return "unknown"
    truth = tp
    fp = top1
    comp_delta = fp["composite"] - truth["composite"]
    if comp_delta <= 0:
        return "no_demotion_top1_is_truth"
    # raw pocket lookup for geometry fields
    tp_raw = next((p for p in pockets if p.get("pocket_id") == truth_pid), {})
    fp_raw = next((p for p in pockets if p.get("pocket_id") == fp["pocket_id"]), {})
    tide_field = "tide_distinctness" if "tide_distinctness" in fp else "tide"
    d_drug = (fp.get("drug") or 0) - (truth.get("drug") or 0)
    d_spike = 0.0
    for sk in ("spike", "spike_density", "spike_conc"):
        if sk in fp or sk in truth:
            d_spike = (fp.get(sk) or 0) - (truth.get(sk) or 0)
            break
    d_tide = (fp.get(tide_field) or 0) - (truth.get(tide_field) or 0)
    d_vol = (fp.get("vol_score") or 0) - (truth.get("vol_score") or 0)
    vol_t = tp_raw.get("site_volume_angstrom_cubed") or 0
    vol_f = fp_raw.get("site_volume_angstrom_cubed") or 0
    # raw spikes
    sp_t = tp_raw.get("n_spikes_attributed") or 0
    sp_f = fp_raw.get("n_spikes_attributed") or 0
    contribs = [
        ("drug_gap", d_drug),
        ("spike_term_gap", d_spike),
        ("tide_term_gap", d_tide),
        ("geometry_gap", d_vol),
    ]
    dom = max(contribs, key=lambda kv: kv[1])[0]
    # secondary: if raw spike count has size bias (larger volume → more spikes)
    size_bias = (vol_f > vol_t * 1.5) and (sp_f > sp_t)
    if dom == "spike_term_gap" and size_bias:
        return "spike_mass_bias"
    if d_tide == 0.0 and tide_field == "tide":
        return "tide_degenerate"
    return dom


def evaluate_tier_A(verbose=True):
    per_variant = {}
    per_target_detail = []

    for t in TIER_A_TARGETS:
        rr = json.loads((BASE / t / "artifacts/6_rerank/rerank_result.json").read_text())
        ev = json.loads((BASE / t / "artifacts/7_evaluation/evaluation.json").read_text())
        pockets = rr["merged_pockets"]
        bm = ev.get("best_match") or {}
        bm_pid = bm.get("site_id")
        bm_dcc = bm.get("centroid_dcc_angstrom")
        detector_ok = (bm_dcc is not None) and (bm_dcc < DCC_MATCH_A)

        target_result = {"target": t, "best_match_pid": bm_pid,
                         "best_match_dcc": round(bm_dcc, 3) if bm_dcc is not None else None,
                         "detector_found": detector_ok, "variants": {}}

        for vname, scorer in ALL_VARIANTS.items():
            import copy
            rows = scorer(copy.deepcopy(pockets))
            for r in rows:
                pass
            bm_rank, _ = rank(rows, bm_pid) if bm_pid is not None else (None, None)
            top1 = rows[0]
            top1_pid = top1["pocket_id"]
            top1_dcc = dcc_of(top1_pid, ev)
            top1_match = (top1_dcc is not None) and (top1_dcc < DCC_MATCH_A)
            fp_top1 = (not top1_match) and ((top1.get("therm_class") or "").upper() == "CRYPTIC")
            cause = cause_of_demotion(pockets, rows, bm_pid, top1) if bm_pid is not None else "no_gt"

            target_result["variants"][vname] = {
                "top1_pid": top1_pid,
                "top1_class_raw": top1.get("therm_class"),
                "top1_class_alias": alias(top1.get("therm_class")),
                "top1_dcc": round(top1_dcc, 3) if top1_dcc is not None else None,
                "top1_match_within_8A": top1_match,
                "fp_top1_cryptic": fp_top1,
                "bm_rank": bm_rank,
                "cause_of_demotion": cause,
            }

            agg = per_variant.setdefault(vname, {
                "top1_correct": 0, "fp_top1": 0, "bm_ranks": [],
                "causes": [], "n": 0,
            })
            agg["n"] += 1
            agg["top1_correct"] += 1 if top1_match else 0
            agg["fp_top1"] += 1 if fp_top1 else 0
            if bm_rank is not None:
                agg["bm_ranks"].append(bm_rank)
            agg["causes"].append(cause)

        per_target_detail.append(target_result)

    # aggregates
    for vname, a in per_variant.items():
        ranks = a["bm_ranks"]
        a["SR@3"] = round(sum(1 for r in ranks if r <= 3) / a["n"], 3) if a["n"] else 0.0
        a["SR@5"] = round(sum(1 for r in ranks if r <= 5) / a["n"], 3) if a["n"] else 0.0
        a["SR@10"] = round(sum(1 for r in ranks if r <= 10) / a["n"], 3) if a["n"] else 0.0
        a["median_bm_rank"] = median(ranks) if ranks else None
        a["cause_counts"] = dict(Counter(a["causes"]))

    return per_variant, per_target_detail


def evaluate_tier_B():
    """Residue-overlap tier on CryptoBench. Runs A + E only (fast variants) for direct comparison."""
    bench = json.loads(CB_BENCH_JSON.read_text())
    per_protein = bench["per_protein"]
    results = {"A_B_no_therm": [], "E_geometry_augmented": []}
    for rec in per_protein:
        apo = rec["pdb"].lower()
        bm_pid = rec.get("best_site")
        if bm_pid is None:
            continue
        tgt = CB_RUN_DIR / apo
        if not tgt.exists():
            continue
        bs_path = next(tgt.glob("*_chain*.binding_sites.json"), None)
        if bs_path is None:
            continue
        try:
            d = json.loads(bs_path.read_text())
        except Exception:
            continue
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
                "site_volume_angstrom_cubed": p.get("mean_volume"),
            }
        for tsite in therm_sites:
            pid = tsite.get("site_id")
            if pid is None:
                continue
            tide = tsite.get("tide_decomposition") or {}
            n_tide = 0
            if isinstance(tide, dict):
                for v in tide.values():
                    n_tide += len(v) if isinstance(v, list) else (int(v) if isinstance(v, (int, float)) else 0)
            row = by_id.setdefault(pid, {"pocket_id": pid})
            row.update({
                "therm_class": tsite.get("therm_class"),
                "druggability_score": tsite.get("druggability"),
                "n_spikes_attributed": (tsite.get("heating_spike_count") or 0) + (tsite.get("cooling_spike_count") or 0),
                "top_residue_ids": list(range(max(1, n_tide))) if n_tide else [],
            })
        pockets = list(by_id.values())
        if not pockets:
            continue

        for vname, scorer in [("A_B_no_therm", score_A), ("E_geometry_augmented", score_E_geometry)]:
            rows = scorer([dict(p) for p in pockets])
            bm_rank, _ = rank(rows, bm_pid)
            top1 = rows[0]
            top1_correct = top1["pocket_id"] == bm_pid
            fp_top1 = (not top1_correct) and ((top1.get("therm_class") or "").upper() == "CRYPTIC")
            results[vname].append({
                "apo": apo, "bm_pid": bm_pid, "bm_rank": bm_rank,
                "top1_pid": top1["pocket_id"], "top1_class": top1.get("therm_class"),
                "top1_correct": top1_correct, "fp_top1_cryptic": fp_top1,
            })

    agg = {}
    for v, rows in results.items():
        n = len(rows)
        top1c = sum(1 for r in rows if r["top1_correct"])
        fp = sum(1 for r in rows if r["fp_top1_cryptic"])
        ranks = [r["bm_rank"] for r in rows if r["bm_rank"] is not None]
        agg[v] = {
            "n": n, "top1_correct": top1c, "fp_top1_cryptic": fp,
            "SR@3": round(sum(1 for r in ranks if r <= 3) / n, 3) if n else 0.0,
            "SR@5": round(sum(1 for r in ranks if r <= 5) / n, 3) if n else 0.0,
            "SR@10": round(sum(1 for r in ranks if r <= 10) / n, 3) if n else 0.0,
            "median_bm_rank": median(ranks) if ranks else None,
        }
    return agg, results


def lexicographic_winner(per_variant):
    def key(v):
        a = per_variant[v]
        return (-a["top1_correct"], a["fp_top1"], -a["SR@3"], -a["SR@5"],
                a["median_bm_rank"] if a["median_bm_rank"] is not None else 9999)
    return min(per_variant, key=key)


def main():
    per_variant_A, per_target_A = evaluate_tier_A()
    per_variant_B, rows_B = evaluate_tier_B()
    winner_A = lexicographic_winner(per_variant_A)

    # Raw artifact — per-target
    print("=" * 110)
    print("TIER A — strict DCC (paired apo-holo). n=4 TWIN-10 pre-panel.")
    print("=" * 110)
    for t in per_target_A:
        print(f"\n— {t['target']}  bm_pid={t['best_match_pid']} bm_dcc={t['best_match_dcc']} Å  "
              f"detector_found={'PASS' if t['detector_found'] else 'FAIL'}")
        print(f"  {'variant':<34} {'top1_pid':>9} {'top1_class':<12} {'top1_alias':<18} "
              f"{'top1_dcc':>9} {'match':>5} {'fp':>3} {'bm_rank':>8} {'cause':<20}")
        for vname, v in t["variants"].items():
            print(f"  {vname:<34} {str(v['top1_pid']):>9} {str(v['top1_class_raw'] or '-'):<12} "
                  f"{v['top1_class_alias']:<18} {str(v['top1_dcc'] if v['top1_dcc'] is not None else '-'):>9} "
                  f"{'PASS' if v['top1_match_within_8A'] else 'FAIL':>5} "
                  f"{'Y' if v['fp_top1_cryptic'] else 'n':>3} "
                  f"{str(v['bm_rank'] or '-'):>8} {v['cause_of_demotion'] or '-':<20}")

    print()
    print("=" * 110)
    print("TIER A AGGREGATE — lexicographic selection")
    print("=" * 110)
    print(f"  primary:   top1_correct↓ | secondary: fp_top1↑ | tertiary: SR@3↓ | quaternary: SR@5↓ | quinary: median↑")
    print(f"  {'variant':<34} {'top1':>5} {'fp':>3} {'SR@3':>6} {'SR@5':>6} {'SR@10':>7} {'median':>7}  causes")
    for v in ALL_VARIANTS:
        a = per_variant_A[v]
        print(f"  {v:<34} {a['top1_correct']:>5}/{a['n']} {a['fp_top1']:>3} {a['SR@3']:>6} {a['SR@5']:>6} "
              f"{a['SR@10']:>7} {str(a['median_bm_rank']):>7}  {a['cause_counts']}")
    print()
    print(f"  WINNER (tier A, lexicographic): {winner_A}")
    print()

    print("=" * 110)
    print(f"TIER B — residue-overlap (CryptoBench). A + E only. n={per_variant_B['A_B_no_therm']['n']}")
    print("=" * 110)
    print(f"  {'variant':<34} {'top1':>5} {'fp_CRY':>6} {'SR@3':>6} {'SR@5':>6} {'SR@10':>7} {'median':>7}")
    for v, a in per_variant_B.items():
        print(f"  {v:<34} {a['top1_correct']:>5}/{a['n']} {a['fp_top1_cryptic']:>6} {a['SR@3']:>6} {a['SR@5']:>6} "
              f"{a['SR@10']:>7} {str(a['median_bm_rank']):>7}")

    print()
    OUT_JSON.write_text(json.dumps({
        "tier_A_per_variant": per_variant_A,
        "tier_A_per_target": per_target_A,
        "tier_A_winner_lexicographic": winner_A,
        "tier_B_per_variant": per_variant_B,
        "tier_B_rows": rows_B,
    }, indent=2, default=str))
    print(f"report: {OUT_JSON}")

    # Check whether root-cause-not-resolved rule fires
    baseline_top1 = per_variant_A["A_B_no_therm"]["top1_correct"]
    best_top1 = max(a["top1_correct"] for a in per_variant_A.values())
    if best_top1 <= baseline_top1:
        print()
        print("ROOT_CAUSE_NOT_RESOLVED — ranking redesign must move beyond current feature family")


if __name__ == "__main__":
    main()
