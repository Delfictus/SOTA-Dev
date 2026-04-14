#!/usr/bin/env python3
"""176-target physics reconstruction gate.

Compares physics_216 matrices produced by extract_all_features.py against
the cached /mnt/storage/prism-outputs/ml/features/{target}_features_216.npy
files. Per directive Phase 0.3, >95% must match within np.allclose(atol=1e-4,
rtol=1e-3).

Also reports per-column agreement since the cached files predate a bug-fix
in extract_216.py Block 7 (cols 212-214 = phase_persistence, temperature_
sensitivity, solvent_reorganization were zero in cached because phase_data
wasn't plumbed through). The fresh extractor produces the corrected values.

Two verdicts emitted:
  • Strict: full 216-dim np.allclose at 1e-4/1e-3
  • Pragmatic: 213-dim (excluding cols 212-214), same tolerance

Usage:
    python3 scripts/training/validate_176_gate.py \
        --fresh-dir /mnt/storage/spike-audit/features-176gate \
        --cached-dir /mnt/storage/prism-outputs/ml/features \
        --out /mnt/storage/spike-audit/176gate_report.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np


ATOL = 1e-4
RTOL = 1e-3
KNOWN_DRIFT_COLS = [212, 213, 214]  # phase_persistence, temp_sensitivity, solvent_reorg


def compare_one(cached_path: Path, fresh_path: Path) -> Dict[str, any]:
    cached = np.load(cached_path)
    fresh_npz = np.load(fresh_path)
    fresh = fresh_npz["physics_216"]

    result = {"target": cached_path.stem.replace("_features_216", ""),
              "fresh_shape": list(fresh.shape),
              "cached_shape": list(cached.shape)}

    if cached.shape != fresh.shape:
        result["status"] = "SHAPE_MISMATCH"
        return result

    # Strict full-dim
    diff = np.abs(cached - fresh)
    result["max_abs_diff"] = float(diff.max())
    result["mean_abs_diff"] = float(diff.mean())
    result["full_allclose"] = bool(np.allclose(cached, fresh, atol=ATOL, rtol=RTOL))

    # Per-column agreement
    n_bad_cols = 0
    bad_col_ids: List[int] = []
    for j in range(cached.shape[1]):
        if not np.allclose(cached[:, j], fresh[:, j], atol=ATOL, rtol=RTOL):
            n_bad_cols += 1
            bad_col_ids.append(j)
    result["n_bad_cols"] = n_bad_cols
    result["bad_col_ids"] = bad_col_ids

    # Pragmatic subset (excluding known-fixed cols)
    mask = np.ones(cached.shape[1], dtype=bool)
    mask[KNOWN_DRIFT_COLS] = False
    result["pragmatic_allclose"] = bool(
        np.allclose(cached[:, mask], fresh[:, mask], atol=ATOL, rtol=RTOL))

    if result["full_allclose"]:
        result["status"] = "MATCH"
    elif result["pragmatic_allclose"] and set(bad_col_ids).issubset(set(KNOWN_DRIFT_COLS)):
        result["status"] = "MATCH_EXCLUDING_KNOWN_FIX"
    else:
        result["status"] = "DRIFT"
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fresh-dir", type=Path, required=True)
    parser.add_argument("--cached-dir", type=Path, required=True,
                        default=Path("/mnt/storage/prism-outputs/ml/features"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--strict-gate", type=float, default=0.95,
                        help="Strict gate threshold (directive default 0.95)")
    parser.add_argument("--pragmatic-gate", type=float, default=0.95)
    args = parser.parse_args()

    cached_files = sorted(args.cached_dir.glob("*_features_216.npy"))
    print(f"Cached files: {len(cached_files)}")

    results: List[Dict[str, any]] = []
    for cf in cached_files:
        target = cf.stem.replace("_features_216", "")
        ff = args.fresh_dir / f"{target}_features.npz"
        if not ff.exists():
            results.append({"target": target, "status": "FRESH_MISSING"})
            continue
        try:
            r = compare_one(cf, ff)
        except Exception as e:
            r = {"target": target, "status": "ERROR", "error": str(e)}
        results.append(r)

    # Aggregate
    n = len(results)
    by_status: Dict[str, int] = {}
    for r in results:
        by_status[r["status"]] = by_status.get(r["status"], 0) + 1

    strict_pass = by_status.get("MATCH", 0)
    pragmatic_pass = strict_pass + by_status.get("MATCH_EXCLUDING_KNOWN_FIX", 0)
    strict_rate = strict_pass / n if n else 0.0
    pragmatic_rate = pragmatic_pass / n if n else 0.0

    # Which columns drift most often?
    col_drift_counts: Dict[int, int] = {}
    for r in results:
        for j in r.get("bad_col_ids", []):
            col_drift_counts[j] = col_drift_counts.get(j, 0) + 1
    col_drift_sorted = sorted(col_drift_counts.items(), key=lambda x: -x[1])

    summary = {
        "total": n,
        "by_status": by_status,
        "strict": {
            "pass": strict_pass,
            "rate": strict_rate,
            "gate_threshold": args.strict_gate,
            "passed": strict_rate >= args.strict_gate,
        },
        "pragmatic": {
            "pass": pragmatic_pass,
            "rate": pragmatic_rate,
            "gate_threshold": args.pragmatic_gate,
            "passed": pragmatic_rate >= args.pragmatic_gate,
            "excluded_cols": KNOWN_DRIFT_COLS,
            "note": "cols 212-214 are phase_persistence/temp_sensitivity/solvent_reorg; "
                    "cached .npy predates a bug-fix in extract_216.py Block 7.",
        },
        "most_drifted_cols": col_drift_sorted[:10],
        "per_target": results,
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, indent=2, default=str))

    # Pretty print
    print("\n" + "="*60)
    print(f"  176-TARGET PHYSICS GATE — {n} targets compared")
    print("="*60)
    for status, cnt in sorted(by_status.items(), key=lambda x: -x[1]):
        print(f"  {status:30s} {cnt:4d}  ({cnt/n:6.1%})")
    print()
    print(f"  STRICT     rate {strict_rate:6.2%}  (gate ≥{args.strict_gate:.0%}): "
          f"{'PASS' if summary['strict']['passed'] else 'FAIL'}")
    print(f"  PRAGMATIC  rate {pragmatic_rate:6.2%}  (gate ≥{args.pragmatic_gate:.0%}, "
          f"excl. cols {KNOWN_DRIFT_COLS}): "
          f"{'PASS' if summary['pragmatic']['passed'] else 'FAIL'}")
    if col_drift_sorted:
        print(f"\n  Top drifted cols: {col_drift_sorted[:5]}")
    print(f"\n  Report: {args.out}")


if __name__ == "__main__":
    main()
