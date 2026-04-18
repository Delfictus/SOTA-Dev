#!/usr/bin/env python3
"""Emit machine-readable per-target completion blocks for the M1.1 strict-DCC panel.

Reads rerank + evaluation artifacts, recomputes all 6 variants, emits one
block per completed target with:
  - best-match site + DCC + detector_found
  - per variant {A,B,C,D,E,F}: top1_pid, top1_dcc, verdict, bm_rank, cause

No placeholders. Missing values emit FEATURE_GAP:<name>. Read-only.
"""
from __future__ import annotations
import copy
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from m1_ablation import (
    score_A, score_B_density, score_C_concentration,
    score_D_tide_distinctness, score_E_geometry, score_F_mechanical_interpreter,
    dcc_of, cause_of_demotion, alias, DCC_MATCH_A,
)
import math as _math


def _direct_dcc(pocket_id, pockets, ev):
    """Direct DCC fallback: Euclidean distance between pocket.centroid_spike_weighted
    (apo frame) and ev['best_match']['centroid']-derived ligand frame. stage 4 stores
    ligand_centroid_apo_frame under artifacts/4_ground_truth/; per_site_eval is limited
    to a subset of ranks. This fallback ensures every variant top-1 has a numeric DCC.
    """
    gt = ev.get("ground_truth_ligand_centroid_apo_frame")
    if gt is None:
        return None
    for p in pockets:
        if p.get("pocket_id") == pocket_id:
            c = p.get("centroid_spike_weighted") or p.get("centroid")
            if c is None or len(c) != 3:
                return None
            return float(_math.sqrt(sum((c[i] - gt[i]) ** 2 for i in range(3))))
    return None


def _load_gt_centroid(target_dir: Path) -> list | None:
    gt_dir = target_dir / "artifacts/4_ground_truth"
    for gt_file in gt_dir.glob("*_ground_truth.json"):
        try:
            g = json.loads(gt_file.read_text())
            return g.get("ligand_centroid_apo_frame")
        except Exception:
            continue
    return None

PANEL_ROOTS = [
    Path("/mnt/storage/prism-outputs/twin-10-patent"),
    Path("/mnt/storage/prism-outputs/m1-strict-dcc-panel"),
]
OUT_DIR = Path("/tmp/m1_1_completion_blocks")
OUT_DIR.mkdir(parents=True, exist_ok=True)

VARIANTS = {
    "A_B_no_therm": score_A,
    "B_spike_density": score_B_density,
    "C_spike_concentration": score_C_concentration,
    "D_tide_distinctness": score_D_tide_distinctness,
    "E_geometry_augmented": score_E_geometry,
    "F_mechanical_interpreter_only": score_F_mechanical_interpreter,
}


def find_target_dir(target_key: str) -> Path | None:
    for root in PANEL_ROOTS:
        p = root / target_key
        if (p / "artifacts/6_rerank/rerank_result.json").exists():
            return p
    return None


def emit_block(target_key: str) -> dict | None:
    tdir = find_target_dir(target_key)
    if tdir is None:
        return {"target_key": target_key, "state": "INCOMPLETE",
                "reason": "rerank_result.json absent"}

    rr = json.loads((tdir / "artifacts/6_rerank/rerank_result.json").read_text())
    evp = tdir / "artifacts/7_evaluation/evaluation.json"
    if not evp.exists():
        return {"target_key": target_key, "state": "INCOMPLETE",
                "reason": "evaluation.json absent"}
    ev = json.loads(evp.read_text())

    pockets = rr["merged_pockets"]
    bm = ev.get("best_match") or {}
    bm_pid = bm.get("site_id")
    bm_dcc = bm.get("centroid_dcc_angstrom")
    detector_found = (bm_dcc is not None) and (bm_dcc < DCC_MATCH_A)

    # Inject ligand_centroid_apo_frame for direct-DCC fallback
    ev["ground_truth_ligand_centroid_apo_frame"] = _load_gt_centroid(tdir)

    block = {
        "target_key": target_key,
        "state": "COMPLETE",
        "verdict_tag": ev.get("verdict_tag"),
        "paired_holo_pdb_id": ev.get("paired_holo_pdb_id"),
        "best_match_site_id": bm_pid if bm_pid is not None else "FEATURE_GAP:best_match_site_id",
        "best_match_dcc_angstrom": round(bm_dcc, 3) if bm_dcc is not None else "FEATURE_GAP:best_match_dcc",
        "best_match_therm_class_raw": bm.get("therm_class") or "FEATURE_GAP:best_match_class",
        "best_match_therm_class_alias": alias(bm.get("therm_class")),
        "detector_found_within_8A": detector_found,
    }

    for vname, scorer in VARIANTS.items():
        rows = scorer(copy.deepcopy(pockets))
        rows.sort(key=lambda r: -r["composite"])
        for i, r in enumerate(rows, 1):
            r["rank"] = i
        top1 = rows[0]
        top1_pid = top1["pocket_id"]
        top1_dcc = dcc_of(top1_pid, ev)
        if top1_dcc is None:
            top1_dcc = _direct_dcc(top1_pid, pockets, ev)
        # Derive verdict with no UNVERIFIED_NO_DCC / FEATURE_GAP tolerance on top-1.
        if ev.get("verdict_tag") == "GROUND_TRUTH_INVALID":
            top1_verdict = "GT_INVALID"
        elif ev.get("verdict_tag") == "NO_GROUND_TRUTH":
            top1_verdict = "NO_GROUND_TRUTH"
        elif top1_dcc is None:
            top1_verdict = "FAIL"  # unreachable after _direct_dcc unless gt centroid absent
        elif top1_dcc < DCC_MATCH_A:
            top1_verdict = "PASS"
        elif top1_dcc > 30.0:
            top1_verdict = "HARD_NOT_MATCH"
        elif top1_dcc > 20.0:
            top1_verdict = "NOT_MATCH"
        else:
            top1_verdict = "FAIL"
        top1_match = top1_verdict == "PASS"

        bm_rank = None
        if bm_pid is not None:
            for r in rows:
                if r["pocket_id"] == bm_pid:
                    bm_rank = r["rank"]
                    break
        cause = cause_of_demotion(pockets, rows, bm_pid, top1) if bm_pid is not None else "no_gt"

        block[f"top1_{vname}"] = {
            "site_id": top1_pid,
            "therm_class_raw": top1.get("therm_class") or "FEATURE_GAP:top1_class",
            "therm_class_alias": alias(top1.get("therm_class")),
            "dcc_angstrom": round(top1_dcc, 3) if top1_dcc is not None else None,
            "dcc_source": ("stage7" if dcc_of(top1_pid, ev) is not None
                           else ("direct_apo_frame" if top1_dcc is not None else "unavailable")),
            "verdict": top1_verdict,
            "composite": round(top1["composite"], 4),
        }
        block[f"bm_rank_{vname}"] = bm_rank if bm_rank is not None else "FEATURE_GAP:bm_rank"
        block[f"cause_{vname}"] = cause

    # Write JSON file per target
    (OUT_DIR / f"{target_key}.json").write_text(json.dumps(block, indent=2, default=str))
    return block


def main():
    tgts = sys.argv[1:] if len(sys.argv) > 1 else [
        "wrn_apo", "menin_apo", "smarca2_brd_apo", "pkmyt1_apo", "kras_g12d_apo",
    ]
    blocks = [emit_block(t) for t in tgts]
    print(json.dumps(blocks, indent=2, default=str))


if __name__ == "__main__":
    main()
