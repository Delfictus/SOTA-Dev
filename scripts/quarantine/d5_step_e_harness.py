#!/usr/bin/env python3
"""Step-E parity + timing harness for prism_spike_watcher per-site parquet conversion.

For each target, convert ONE representative per-site spike_events.json to parquet
via:
  A_JSON   : convert_json_to_parquet (legacy)
  C_ARROW  : convert_arrow_to_parquet_per_site (new D5 path)

Compare:
  - schema (column names, dtypes)
  - content (row-multiset equivalence; decoded string columns; numeric columns)
  - timing + RSS

Fallback test: hide triad → arrow converter returns None; legacy path is invoked.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path
import re

REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO))

DEFAULT_TARGET_DIR = Path("/home/diddy/prism-working/m1-strict-dcc-panel/m1_2akr")
DEFAULT_STEM = "2akr"


def _hide_triad(eng: Path, stem: str):
    meta = eng / f"{stem}.run_metadata.json"
    hidden = eng / f"{stem}.run_metadata.json._hidden_for_step_e"
    if meta.exists():
        meta.rename(hidden)


def _restore_triad(eng: Path, stem: str):
    meta = eng / f"{stem}.run_metadata.json"
    hidden = eng / f"{stem}.run_metadata.json._hidden_for_step_e"
    if hidden.exists():
        hidden.rename(meta)


def _pick_site_json(eng: Path, stem: str) -> Path:
    files = sorted(eng.glob(f"{stem}.site*.spike_events.json"))
    if not files:
        raise SystemExit(f"no per-site JSON in {eng}")
    return files[len(files) // 2]


def run_convert(json_path: Path, mode: str):
    """Run the chosen converter on json_path. Returns parquet path + elapsed."""
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "prism_spike_watcher",
        REPO / "scripts/prism-r2-sync/prism_spike_watcher.py",
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    # Move any existing parquet aside so conversion actually runs
    pq_path = Path(str(json_path).replace(".json", ".parquet"))
    stash = pq_path.with_suffix(".parquet.stashed_for_step_e_" + mode)
    if pq_path.exists():
        pq_path.rename(stash)
    try:
        t0 = time.perf_counter()
        if mode == "C_ARROW":
            result = mod.convert_arrow_to_parquet_per_site(str(json_path), dry_run=False)
        else:
            result = mod.convert_json_to_parquet(str(json_path), dry_run=False)
        elapsed = time.perf_counter() - t0
    finally:
        pass
    # Rename written parquet to mode-specific path so both can coexist
    if result and Path(result).exists():
        mode_pq = Path(result).with_name(Path(result).stem + f"__{mode}.parquet")
        os.replace(result, str(mode_pq))
        result = str(mode_pq)
    # Restore any stashed parquet
    if stash.exists() and not pq_path.exists():
        stash.rename(pq_path)
    return result, elapsed


def schema_content_parity(a_parquet: str, c_parquet: str):
    """Compare two parquets: schema + row-multiset content."""
    import pyarrow.parquet as pq
    ta = pq.read_table(a_parquet)
    tc = pq.read_table(c_parquet)
    cols_a = ta.column_names
    cols_c = tc.column_names
    cols_common = [c for c in cols_a if c in cols_c]
    cols_only_a = [c for c in cols_a if c not in cols_c]
    cols_only_c = [c for c in cols_c if c not in cols_a]
    dtypes_a = {c: str(ta.schema.field(c).type) for c in cols_a}
    dtypes_c = {c: str(tc.schema.field(c).type) for c in cols_c}
    schema_mismatch = []
    for c in cols_common:
        if dtypes_a[c] != dtypes_c[c]:
            schema_mismatch.append({"col": c, "a": dtypes_a[c], "c": dtypes_c[c]})

    # Row count
    nrows_a = ta.num_rows
    nrows_c = tc.num_rows

    # Row-multiset hash via polars (canonical column subset)
    import polars as pl
    # Sort columns and rows for canonical comparison
    pa_df = pl.from_arrow(ta).select(sorted(cols_common))
    pc_df = pl.from_arrow(tc).select(sorted(cols_common))
    # Multiset hash: sort rows, concat + sha256
    pa_sorted = pa_df.sort(sorted(cols_common))
    pc_sorted = pc_df.sort(sorted(cols_common))
    h_a = hashlib.sha256(pa_sorted.write_csv().encode()).hexdigest()
    h_c = hashlib.sha256(pc_sorted.write_csv().encode()).hexdigest()

    verdict = "PASS"
    if cols_only_a or cols_only_c or schema_mismatch or nrows_a != nrows_c or h_a != h_c:
        verdict = "FAIL"

    return {
        "cols_a": cols_a, "cols_c": cols_c,
        "cols_only_a": cols_only_a, "cols_only_c": cols_only_c,
        "dtypes_a": dtypes_a, "dtypes_c": dtypes_c,
        "schema_mismatch": schema_mismatch,
        "nrows_a": nrows_a, "nrows_c": nrows_c,
        "row_hash_a": h_a, "row_hash_c": h_c,
        "row_multiset_equal": h_a == h_c,
        "a_parquet_size": os.path.getsize(a_parquet),
        "c_parquet_size": os.path.getsize(c_parquet),
        "verdict": verdict,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target-dir", default=str(DEFAULT_TARGET_DIR))
    ap.add_argument("--stem", default=DEFAULT_STEM)
    ap.add_argument("--mode", required=True,
                    choices=["legacy", "arrow", "parity", "fallback"])
    args = ap.parse_args()

    td = Path(args.target_dir)
    eng = td / "artifacts/5_engine"
    stem = args.stem
    spike_json = _pick_site_json(eng, stem)

    if args.mode == "fallback":
        _hide_triad(eng, stem)
        try:
            c_res, _ = run_convert(spike_json, "C_ARROW")
            fallback_arrow_blocked = (c_res is None)
            l_res, _ = run_convert(spike_json, "A_JSON")
            fallback_legacy_ok = (l_res is not None)
            print(f"FALLBACK_ARROW_BLOCKED={fallback_arrow_blocked}")
            print(f"FALLBACK_LEGACY_OK={fallback_legacy_ok}")
        finally:
            _restore_triad(eng, stem)
        return

    if args.mode == "parity":
        a_res, a_el = run_convert(spike_json, "A_JSON")
        c_res, c_el = run_convert(spike_json, "C_ARROW")
        print(f"A_ELAPSED={a_el:.3f}s  A_PARQUET={a_res}")
        print(f"C_ELAPSED={c_el:.3f}s  C_PARQUET={c_res}")
        if not a_res or not c_res:
            print(json.dumps({"verdict": "FAIL_NO_OUTPUT"}))
            return
        report = schema_content_parity(a_res, c_res)
        # Minimal print
        m = {k: v for k, v in report.items()
             if k in ("cols_a", "cols_c", "cols_only_a", "cols_only_c",
                      "schema_mismatch", "nrows_a", "nrows_c",
                      "row_hash_a", "row_hash_c", "row_multiset_equal",
                      "a_parquet_size", "c_parquet_size", "verdict")}
        print(json.dumps(m, indent=2, default=str))
        return

    # single-mode timing
    t0 = time.perf_counter()
    if args.mode == "legacy":
        _hide_triad(eng, stem)
        try:
            res, _ = run_convert(spike_json, "A_JSON")
        finally:
            _restore_triad(eng, stem)
    else:
        res, _ = run_convert(spike_json, "C_ARROW")
    wall = time.perf_counter() - t0
    print(f"MODE={args.mode}")
    print(f"TARGET={stem}")
    print(f"PARQUET={res}")
    print(f"PARQUET_BYTES={os.path.getsize(res) if res and Path(res).exists() else None}")
    print(f"INTERNAL_ELAPSED={wall:.3f}s")


if __name__ == "__main__":
    main()
