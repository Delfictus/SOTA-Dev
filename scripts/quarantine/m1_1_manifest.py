#!/usr/bin/env python3
"""M1.1 Phase 0 — strict-DCC manifest with BIN_A/B/C classification.

Read-only. Writes /tmp/m1_1_manifest.json.

Bin definitions (pre-declared):
  BIN_A_READY          : rerank_result.json + evaluation.json + DCC already computed.
  BIN_B_FIXABLE        : artifacts exist but stage-4 GT failed with deterministic fix path.
  BIN_C_REQUIRES_ENGINE: no engine output exists; engine run required.
"""
from __future__ import annotations
import json
from pathlib import Path

BASE_TWIN = Path("/mnt/storage/prism-outputs/twin-10-patent")
BASE_M1 = Path("/mnt/storage/prism-outputs/m1-strict-dcc-panel")
OUT = Path("/tmp/m1_1_manifest.json")

PANEL = [
    {"target_key": "wrn_apo",          "apo": "6yhr", "holo": "8pfo", "ligand": "YHC",   "source": "TWIN-10 done", "target_dir": str(BASE_TWIN/"wrn_apo")},
    {"target_key": "menin_apo",        "apo": "3re2", "holo": "7uj4", "ligand": "OQ4",   "source": "TWIN-10 done", "target_dir": str(BASE_TWIN/"menin_apo")},
    {"target_key": "smarca2_brd_apo",  "apo": "4qy4", "holo": "5dkc", "ligand": "5BW",   "source": "TWIN-10 done", "target_dir": str(BASE_TWIN/"smarca2_brd_apo")},
    {"target_key": "pkmyt1_apo",       "apo": "3p1a", "holo": "8d6e", "ligand": "QGI",   "source": "TWIN-10 done", "target_dir": str(BASE_TWIN/"pkmyt1_apo")},
    {"target_key": "kras_g12d_apo",    "apo": "7f0w", "holo": "7rpz", "ligand": "6IC",   "source": "TWIN-10 recovery", "target_dir": str(BASE_TWIN/"kras_g12d_apo")},
    {"target_key": "usp1_apo",         "apo": "7ay0", "holo": "9di1", "ligand": "A1A4Y", "source": "TWIN-10 recovery", "target_dir": str(BASE_TWIN/"usp1_apo")},
    {"target_key": "polq_apo",         "apo": "6xbu", "holo": "8e24", "ligand": "auto",  "source": "TWIN-10 recovery", "target_dir": str(BASE_TWIN/"polq_apo")},
    {"target_key": "m1_2nvp", "apo": "2nvp", "holo": "3qt9", "ligand": "Z4Y", "source": "blind_validation_100", "target_dir": str(BASE_M1/"m1_2nvp")},
    {"target_key": "m1_1xhx", "apo": "1xhx", "holo": "2pyj", "ligand": "DGT", "source": "blind_validation_100", "target_dir": str(BASE_M1/"m1_1xhx")},
    {"target_key": "m1_2akr", "apo": "2akr", "holo": "6ojp", "ligand": "BMA", "source": "blind_validation_100", "target_dir": str(BASE_M1/"m1_2akr")},
    {"target_key": "m1_2e3k", "apo": "2e3k", "holo": "7wmu", "ligand": "JGF", "source": "blind_validation_100", "target_dir": str(BASE_M1/"m1_2e3k")},
    {"target_key": "m1_3bjp", "apo": "3bjp", "holo": "4n9v", "ligand": "AZA", "source": "blind_validation_100", "target_dir": str(BASE_M1/"m1_3bjp")},
    {"target_key": "m1_6tyo", "apo": "6tyo", "holo": "6tyn", "ligand": "5N6", "source": "blind_validation_100", "target_dir": str(BASE_M1/"m1_6tyo")},
    {"target_key": "m1_7se6", "apo": "7se6", "holo": "7se8", "ligand": "8W1", "source": "blind_validation_100", "target_dir": str(BASE_M1/"m1_7se6")},
    {"target_key": "m1_1k47", "apo": "1k47", "holo": "3gon", "ligand": "PMV", "source": "blind_validation_100", "target_dir": str(BASE_M1/"m1_1k47")},
    {"target_key": "m1_5yj2", "apo": "5yj2", "holo": "7cug", "ligand": "MXE", "source": "blind_validation_100", "target_dir": str(BASE_M1/"m1_5yj2")},
    {"target_key": "m1_3umi", "apo": "3umi", "holo": "5buo", "ligand": "SGN", "source": "blind_validation_100", "target_dir": str(BASE_M1/"m1_3umi")},
    {"target_key": "m1_3bl7", "apo": "3bl7", "holo": "1st0", "ligand": "GTG", "source": "blind_validation_100", "target_dir": str(BASE_M1/"m1_3bl7")},
]


def classify(entry: dict) -> dict:
    tdir = Path(entry["target_dir"])
    rerank = tdir / "artifacts/6_rerank/rerank_result.json"
    evalj = tdir / "artifacts/7_evaluation/evaluation.json"
    engine_out = tdir / "artifacts/5_engine"
    gt_dir = tdir / "artifacts/4_ground_truth"

    engine_ok = rerank.exists()
    gt_files = list(gt_dir.glob("*_ground_truth.json")) if gt_dir.exists() else []
    gt_ok = False
    gt_error = None
    if gt_files:
        try:
            g = json.loads(gt_files[0].read_text())
            gt_error = g.get("error")
            gt_ok = gt_error is None and g.get("ligand_resname") is not None
        except Exception as e:
            gt_error = f"parse_fail: {e}"

    if engine_ok and gt_ok and evalj.exists():
        bin_code = "BIN_A_READY"
        blocker = "none"
        eta_min = 0
    elif engine_ok and not gt_ok:
        bin_code = "BIN_B_FIXABLE"
        blocker = f"stage4_error: {gt_error}" if gt_error else "stage4_missing"
        eta_min = 10
    else:
        bin_code = "BIN_C_REQUIRES_ENGINE"
        blocker = "engine_output_absent"
        eta_min = 60

    return {
        "bin": bin_code,
        "engine_output_exists": engine_ok,
        "gt_ready": gt_ok,
        "blocker": blocker,
        "estimated_runtime_min": eta_min,
    }


def main() -> None:
    # classify + assign execution order
    order_counter = 0
    records = []
    # BIN_A_READY first (no action), BIN_B_FIXABLE next, BIN_C_REQUIRES_ENGINE last
    tmp = []
    for e in PANEL:
        c = classify(e)
        tmp.append({**e, **c})
    priority = {"BIN_A_READY": 0, "BIN_B_FIXABLE": 1, "BIN_C_REQUIRES_ENGINE": 2}
    tmp.sort(key=lambda r: (priority[r["bin"]], r["target_key"]))
    for i, r in enumerate(tmp, 1):
        r["execution_order"] = i
        records.append(r)

    bin_counts = {"BIN_A_READY": 0, "BIN_B_FIXABLE": 0, "BIN_C_REQUIRES_ENGINE": 0}
    for r in records:
        bin_counts[r["bin"]] += 1

    # Print manifest table
    print(f"{'ord':>3} {'bin':<22} {'target_key':<22} {'apo':<6} {'holo':<6} {'ligand':<8} "
          f"{'eng':<4} {'gt':<3} {'ETA':>5}  blocker")
    print(f"{'-'*3} {'-'*22} {'-'*22} {'-'*6} {'-'*6} {'-'*8} {'-'*4} {'-'*3} {'-'*5}  {'-'*40}")
    for r in records:
        print(f"{r['execution_order']:>3} {r['bin']:<22} {r['target_key']:<22} {r['apo']:<6} {r['holo']:<6} "
              f"{r['ligand']:<8} "
              f"{'yes' if r['engine_output_exists'] else 'no':<4} "
              f"{'yes' if r['gt_ready'] else 'no':<3} "
              f"{r['estimated_runtime_min']:>5}  {r['blocker']}")
    print()
    print(f"bin_counts = {bin_counts}")
    print(f"n_total = {len(records)}")
    print(f"total_runtime_estimate_min = {sum(r['estimated_runtime_min'] for r in records)}")

    OUT.write_text(json.dumps({
        "n_total": len(records),
        "bin_counts": bin_counts,
        "panel": records,
    }, indent=2))
    print(f"manifest: {OUT}")


if __name__ == "__main__":
    main()
