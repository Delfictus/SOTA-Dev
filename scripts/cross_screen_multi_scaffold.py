#!/usr/bin/env python3
"""Prepare honest multi-scaffold cross-screen rows for top candidates."""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import polars as pl


REPO_ROOT = Path(__file__).resolve().parents[1]
TRACK_A = REPO_ROOT / "campaigns/glp1r_aleniglipron/track_a_generative"
DEFAULT_INPUT = TRACK_A / "gflownet_top_50_tripartite_profiles.parquet"
DEFAULT_OUTPUT = TRACK_A / "gflownet_top_50_cross_scaffold_screen.parquet"
DEFAULT_REPORT = TRACK_A / "gflownet_top_50_cross_scaffold_screen_report.json"
SCAFFOLDS = ("ALENI", "ORFOR", "DANU")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--candidates", type=Path, default=None, help="Alias for --input used by Epoch 016.")
    parser.add_argument("--top-n", type=int, default=None)
    parser.add_argument("--scaffold-pool", type=Path, action="append", default=None)
    parser.add_argument("--signal-grid", type=Path, default=None)
    parser.add_argument("--lock-mask", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    input_path = Path(args.candidates) if args.candidates is not None else Path(args.input)
    profiles = pl.read_parquet(input_path)
    if args.top_n is not None:
        profiles = profiles.head(int(args.top_n))
    rows = [cross_screen_row(row) for row in profiles.iter_rows(named=True)]
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    tmp_output = output.with_suffix(output.suffix + ".tmp")
    pl.DataFrame(rows).write_parquet(tmp_output)
    tmp_output.replace(output)
    report = {
        "schema_version": "PRISM.cross_scaffold_screen.v1",
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "input": str(input_path),
        "output": str(output),
        "candidate_count": len(rows),
        "n_scaffolds_positive_ge_2": sum(1 for row in rows if int(row["n_scaffolds_positive"]) >= 2),
        "scaffold_pool": [str(path) for path in (args.scaffold_pool or [])],
        "signal_grid": str(args.signal_grid) if args.signal_grid is not None else None,
        "lock_mask": str(args.lock_mask) if args.lock_mask is not None else None,
        "evidence_class": "PROJECTED_PROXY_NO_REDOCK",
        "note": (
            "This file preserves execution plumbing. L5 cross-scaffold evidence still requires "
            "candidate placement in each scaffold context before promotion."
        ),
    }
    atomic_write_json(Path(args.report), report)
    print(
        "cross_scaffold_screen_prepared "
        f"count={len(rows)} n_scaffolds_positive_ge_2={report['n_scaffolds_positive_ge_2']} output={output}"
    )
    return 0


def cross_screen_row(row: dict[str, Any]) -> dict[str, Any]:
    origin = str(row.get("scaffold_origin", row.get("router_scaffold_assignment", "UNKNOWN"))).upper()
    reward = float_value(row.get("reward", 0.0))
    lock = float_value(row.get("lock_geometry_score", 0.0))
    output = {
        "candidate_id": str(row.get("candidate_id", "")),
        "canonical_smiles": str(row.get("canonical_smiles", "")),
        "scaffold_origin": origin,
        "cross_scaffold_evidence": "PROJECTED_PROXY_NO_REDOCK",
        "cross_scaffold_score": reward,
        "n_scaffolds_positive": 1 if reward > 0.0 else 0,
    }
    for scaffold in SCAFFOLDS:
        is_origin = origin.startswith(scaffold) if origin != "UNKNOWN" else scaffold == "ALENI"
        output[f"reward_{scaffold.lower()}"] = reward if is_origin else None
        output[f"lock_geo_{scaffold.lower()}"] = lock if is_origin else None
    return output


def float_value(value: object) -> float:
    if value is None or isinstance(value, bool):
        return 0.0
    if isinstance(value, int | float | str):
        return float(value)
    return 0.0


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp_path.replace(path)


if __name__ == "__main__":
    raise SystemExit(main())
