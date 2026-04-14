#!/usr/bin/env python3
"""TIDE coverage gate — directive Phase 0.3 (secondary).

Verifies that ≥350 (of 372) targets have non-empty tide_residue data in
their extracted .npz feature bundle.

"Non-empty" = at least one residue has transfer_entropy > 0 OR role_trigger=1
OR role_responder=1.

For targets below threshold, prints which file is missing / malformed.

Usage:
    python3 scripts/training/validate_tide_coverage.py \
        --features-dir /mnt/storage/spike-audit/features-pct95 \
        --out /mnt/storage/spike-audit/tide_coverage_report.json \
        --gate 350
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--features-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--gate", type=int, default=350)
    args = parser.parse_args()

    npzs = sorted(args.features_dir.glob("*_features.npz"))

    results: List[Dict[str, any]] = []
    for p in npzs:
        target = p.stem.replace("_features", "")
        try:
            d = np.load(p, allow_pickle=False)
            tide = d["tide_residue"]
        except Exception as e:
            results.append({"target": target, "status": "LOAD_ERROR", "error": str(e)})
            continue
        # Non-empty test
        te = tide[:, 0]  # transfer_entropy column
        role_trigger = tide[:, 4]
        role_responder = tide[:, 5]
        n_te = int((te > 0).sum())
        n_trigger = int((role_trigger > 0).sum())
        n_responder = int((role_responder > 0).sum())
        has_any = (n_te > 0) or (n_trigger > 0) or (n_responder > 0)
        results.append({
            "target": target,
            "n_residues": int(tide.shape[0]),
            "n_te_nonzero": n_te,
            "n_trigger": n_trigger,
            "n_responder": n_responder,
            "status": "OK" if has_any else "EMPTY",
        })

    n_total = len(results)
    n_ok = sum(1 for r in results if r["status"] == "OK")
    n_empty = sum(1 for r in results if r["status"] == "EMPTY")
    n_err = sum(1 for r in results if r["status"] == "LOAD_ERROR")

    summary = {
        "total": n_total,
        "ok": n_ok,
        "empty": n_empty,
        "load_error": n_err,
        "gate_threshold": args.gate,
        "passed": n_ok >= args.gate,
        "empty_targets": [r["target"] for r in results if r["status"] == "EMPTY"],
        "error_targets": [{"target": r["target"], "error": r.get("error")}
                          for r in results if r["status"] == "LOAD_ERROR"],
        "per_target": results,
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, indent=2, default=str))

    print("="*60)
    print(f"  TIDE COVERAGE GATE — {n_total} targets")
    print("="*60)
    print(f"  OK (has TIDE data):  {n_ok}  ({n_ok/max(n_total,1):.1%})")
    print(f"  EMPTY:               {n_empty}")
    print(f"  LOAD_ERROR:          {n_err}")
    print(f"\n  Gate ≥{args.gate}: {'PASS' if summary['passed'] else 'FAIL'}")
    if summary["empty_targets"][:10]:
        print(f"  Empty (first 10): {summary['empty_targets'][:10]}")
    print(f"\n  Report: {args.out}")


if __name__ == "__main__":
    main()
