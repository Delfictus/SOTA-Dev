#!/usr/bin/env python3
"""Canonical DCC audit across completed Tier-A targets.

Hard rule: exactly one canonical_dcc() function in this file. No auto-picking.

Canonical DCC definition (locked, pre-declared):
  pocket_centroid       = rerank_result.json.merged_pockets[pid].centroid_spike_weighted
                          (apo frame; spike-weighted centroid emitted by stage 6)
  ligand_centroid       = ground_truth.json.ligand_centroid_apo_frame
                          (apo frame; arithmetic mean of ligand heavy-atom coords
                          after Kabsch-aligning holo onto apo)
  alignment_mode        = "kabsch_apo_to_holo" (fixed by stage 4)
  residue_convention    = one_indexed (topology_resnum = topology_index + 1)
  centroid_definition   = "spike_weighted" (pocket side) + "heavy_atom_mean" (ligand side)
  DCC                   = Euclidean norm of (pocket_centroid - ligand_centroid)

Reference sources that feed the final Tier-A table:
  stage7           : evaluation.json per_site_eval[*].centroid_dcc_angstrom
                     (stage 7 uses binding_sites.json pocket.centroid, NOT centroid_spike_weighted)
  direct_apo_frame : per_target_completion_block.py._direct_dcc fallback
                     (uses rerank centroid_spike_weighted + ground_truth ligand_centroid_apo_frame)

This audit recomputes DCC canonically for every reported site and flags drift.

Hard outputs:
  delta > 0.5 Å         → DCC_PIPELINE_DRIFT
  verdict changes       → DCC_VERDICT_INSTABILITY
  neither               → DCC_PIPELINE_STABLE
"""
from __future__ import annotations
import copy
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from m1_ablation import (
    score_A, score_B_density, score_C_concentration,
    score_D_tide_distinctness, score_E_geometry, score_F_mechanical_interpreter,
    DCC_MATCH_A,
)

VARIANTS = {
    "A_B_no_therm": score_A,
    "B_spike_density": score_B_density,
    "C_spike_concentration": score_C_concentration,
    "D_tide_distinctness": score_D_tide_distinctness,
    "E_geometry_augmented": score_E_geometry,
    "F_mechanical_interpreter_only": score_F_mechanical_interpreter,
}

PANEL_ROOTS = [
    Path("/mnt/storage/prism-outputs/twin-10-patent"),
    Path("/mnt/storage/prism-outputs/m1-strict-dcc-panel"),
]
OUT = Path("/tmp/canonical_dcc_audit.json")


def canonical_dcc(pocket_id, pockets, gt):
    """THE ONE canonical DCC function.

    Returns (dcc_angstrom, diagnostics_dict) where diagnostics record the
    exact components used. Returns (None, {...}) if ligand centroid is
    missing in ground_truth or pocket centroid is missing in rerank.
    """
    lig_c = gt.get("ligand_centroid_apo_frame") if gt else None
    lig_n = gt.get("ligand_n_heavy_atoms") if gt else None
    lig_resname = gt.get("ligand_resname") if gt else None
    diag = {
        "ligand_resname": lig_resname or "FEATURE_GAP:ligand_resname",
        "ligand_n_heavy_atoms": lig_n if lig_n is not None else "FEATURE_GAP:ligand_n_heavy_atoms",
        "ligand_chain": gt.get("ligand_chain") if gt else "FEATURE_GAP:ligand_chain",
        "alignment_mode": (gt.get("alignment_method") if gt else None) or "FEATURE_GAP:alignment_mode",
        "residue_convention": "one_indexed",
        "centroid_definition": "spike_weighted_pocket_vs_heavy_atom_mean_ligand",
    }
    if lig_c is None or len(lig_c) != 3:
        return None, diag
    for p in pockets:
        if p.get("pocket_id") == pocket_id:
            c = p.get("centroid_spike_weighted") or p.get("centroid")
            if c is None or len(c) != 3:
                return None, diag
            return math.sqrt(sum((c[i] - lig_c[i]) ** 2 for i in range(3))), diag
    return None, diag


def verdict_from_dcc(dcc, gt_valid):
    if not gt_valid:
        return "GT_INVALID"
    if dcc is None:
        return "FAIL"
    if dcc < DCC_MATCH_A:
        return "PASS"
    if dcc > 30.0:
        return "HARD_NOT_MATCH"
    if dcc > 20.0:
        return "NOT_MATCH"
    return "FAIL"


def find_target_dir(target_key):
    for root in PANEL_ROOTS:
        p = root / target_key
        if (p / "artifacts/6_rerank/rerank_result.json").exists():
            return p
    return None


def load_artifacts(tdir):
    rr = json.loads((tdir / "artifacts/6_rerank/rerank_result.json").read_text())
    ev = json.loads((tdir / "artifacts/7_evaluation/evaluation.json").read_text())
    gt = None
    gt_dir = tdir / "artifacts/4_ground_truth"
    for f in gt_dir.glob("*_ground_truth.json"):
        try:
            g = json.loads(f.read_text())
            if g.get("error") is None and g.get("ligand_centroid_apo_frame"):
                gt = g
        except Exception:
            pass
        break
    return rr, ev, gt


def reported_dcc_from_stage7(pid, ev):
    for s in ev.get("per_site_eval") or []:
        if s.get("site_id") == pid:
            return s.get("centroid_dcc_angstrom"), "stage7"
    bm = ev.get("best_match") or {}
    if bm.get("site_id") == pid:
        return bm.get("centroid_dcc_angstrom"), "stage7_best_match"
    return None, "unavailable"


def audit_target(target_key):
    tdir = find_target_dir(target_key)
    if tdir is None:
        return None
    rr, ev, gt = load_artifacts(tdir)
    pockets = rr["merged_pockets"]
    gt_valid = gt is not None
    bm = ev.get("best_match") or {}

    rows = []

    # Best-match row
    bm_pid = bm.get("site_id")
    if bm_pid is not None:
        reported, src = reported_dcc_from_stage7(bm_pid, ev)
        canonical, diag = canonical_dcc(bm_pid, pockets, gt or {})
        delta = (abs(reported - canonical)
                 if (isinstance(reported, (int, float)) and isinstance(canonical, (int, float)))
                 else None)
        rows.append({
            "target": target_key, "role": "best_match", "variant": "-",
            "site_id": bm_pid, "dcc_source": src,
            "reported_dcc": round(reported, 3) if isinstance(reported, (int, float)) else reported,
            "canonical_dcc": round(canonical, 3) if isinstance(canonical, (int, float)) else canonical,
            "abs_delta": round(delta, 3) if delta is not None else None,
            "ligand_resname": diag.get("ligand_resname"),
            "ligand_chain": diag.get("ligand_chain"),
            "ligand_n_heavy_atoms": diag.get("ligand_n_heavy_atoms"),
            "alignment_mode": diag.get("alignment_mode"),
            "residue_convention": diag.get("residue_convention"),
            "centroid_definition": diag.get("centroid_definition"),
            "verdict_reported": verdict_from_dcc(reported if isinstance(reported, (int, float)) else None, gt_valid),
            "verdict_canonical": verdict_from_dcc(canonical, gt_valid),
        })

    # Variant top-1 rows
    for vname, scorer in VARIANTS.items():
        scored = scorer(copy.deepcopy(pockets))
        scored.sort(key=lambda r: -r["composite"])
        top1_pid = scored[0]["pocket_id"]
        reported, src = reported_dcc_from_stage7(top1_pid, ev)
        canonical, diag = canonical_dcc(top1_pid, pockets, gt or {})
        delta = (abs(reported - canonical)
                 if (isinstance(reported, (int, float)) and isinstance(canonical, (int, float)))
                 else None)
        rows.append({
            "target": target_key, "role": f"top1_{vname}", "variant": vname,
            "site_id": top1_pid, "dcc_source": src,
            "reported_dcc": round(reported, 3) if isinstance(reported, (int, float)) else reported,
            "canonical_dcc": round(canonical, 3) if isinstance(canonical, (int, float)) else canonical,
            "abs_delta": round(delta, 3) if delta is not None else None,
            "ligand_resname": diag.get("ligand_resname"),
            "ligand_chain": diag.get("ligand_chain"),
            "ligand_n_heavy_atoms": diag.get("ligand_n_heavy_atoms"),
            "alignment_mode": diag.get("alignment_mode"),
            "residue_convention": diag.get("residue_convention"),
            "centroid_definition": diag.get("centroid_definition"),
            "verdict_reported": verdict_from_dcc(reported if isinstance(reported, (int, float)) else None, gt_valid),
            "verdict_canonical": verdict_from_dcc(canonical, gt_valid),
        })

    return rows


def main():
    targets = sys.argv[1:] if len(sys.argv) > 1 else [
        "wrn_apo", "menin_apo", "smarca2_brd_apo", "pkmyt1_apo",
        "kras_g12d_apo", "m1_2nvp", "m1_1xhx",
    ]
    all_rows = []
    for t in targets:
        rows = audit_target(t)
        if rows:
            all_rows.extend(rows)

    # Print table
    hdr = ["target", "role", "site", "reported", "src", "canonical", "Δ",
           "lig", "chain", "n_heavy", "align", "v_reported", "v_canonical"]
    widths = [18, 34, 6, 10, 20, 10, 7, 6, 5, 7, 22, 16, 16]
    print("| " + " | ".join(f"{h:<{w}}" for h, w in zip(hdr, widths)) + " |")
    print("| " + " | ".join("-" * w for w in widths) + " |")
    drift_hits = []
    verdict_changes = []
    for r in all_rows:
        rep = r["reported_dcc"]
        can = r["canonical_dcc"]
        delta = r["abs_delta"]
        rep_s = f"{rep:>10.3f}" if isinstance(rep, (int, float)) else f"{str(rep):>10}"
        can_s = f"{can:>10.3f}" if isinstance(can, (int, float)) else f"{str(can):>10}"
        d_s = f"{delta:>7.3f}" if isinstance(delta, (int, float)) else f"{str(delta):>7}"
        print(
            f"| {r['target']:<18} | {r['role']:<34} | {str(r['site_id']):<6} | "
            f"{rep_s} | {str(r['dcc_source']):<20} | {can_s} | {d_s} | "
            f"{str(r['ligand_resname'])[:6]:<6} | {str(r['ligand_chain'])[:5]:<5} | "
            f"{str(r['ligand_n_heavy_atoms']):<7} | "
            f"{str(r['alignment_mode'])[:22]:<22} | "
            f"{r['verdict_reported']:<16} | {r['verdict_canonical']:<16} |"
        )
        if isinstance(delta, (int, float)) and delta > 0.5:
            drift_hits.append(r)
        if r["verdict_reported"] != r["verdict_canonical"]:
            verdict_changes.append(r)

    print()
    print(f"aggregate drift summary:")
    print(f"  total_rows                       = {len(all_rows)}")
    print(f"  rows_with_delta > 0.5 Å          = {len(drift_hits)}")
    print(f"  rows_with_verdict_change         = {len(verdict_changes)}")
    print(f"  rows_missing_canonical_dcc       = {sum(1 for r in all_rows if not isinstance(r['canonical_dcc'], (int, float)))}")
    print(f"  rows_missing_reported_dcc        = {sum(1 for r in all_rows if not isinstance(r['reported_dcc'], (int, float)))}")

    if drift_hits:
        print()
        print("DCC_PIPELINE_DRIFT")
        for r in drift_hits:
            print(f"  {r['target']} {r['role']} site={r['site_id']} reported={r['reported_dcc']} "
                  f"canonical={r['canonical_dcc']} Δ={r['abs_delta']}")
    if verdict_changes:
        print()
        print("DCC_VERDICT_INSTABILITY")
        for r in verdict_changes:
            print(f"  {r['target']} {r['role']} site={r['site_id']}  "
                  f"reported_verdict={r['verdict_reported']} → canonical_verdict={r['verdict_canonical']}")

    if not drift_hits and not verdict_changes:
        print()
        print("RECOMMENDATION: DCC_PIPELINE_STABLE")
    elif drift_hits or verdict_changes:
        print()
        print("RECOMMENDATION: DCC_PIPELINE_REQUIRES_UNIFICATION")

    OUT.write_text(json.dumps({
        "audit_rows": all_rows,
        "drift_hits": drift_hits,
        "verdict_changes": verdict_changes,
    }, indent=2, default=str))
    print()
    print(f"report: {OUT}")


if __name__ == "__main__":
    main()
