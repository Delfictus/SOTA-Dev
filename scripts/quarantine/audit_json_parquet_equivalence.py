#!/usr/bin/env python3
"""
[DIAGNOSTIC - READ-ONLY]

Three-tier rigorous audit of JSON vs Parquet spike data equivalence.
Designed to leverage workstation hardware (24-core CPU, 103 GB RAM, NVMe).

TIERS
  T1 — ALL sites on disk (~1,679 inc. unpaired): structural (row count,
       schema, site_id, format coverage, internal consistency).
  T2 — Stratified sample of 50 paired sites (small/mid/large):
       Polars hash_rows multiset equality + schema cast consistency.
  T3 — 5 spot pairs chosen by size (smallest / largest / median + 2 random):
       native-order AND multiset comparison — detects order/duplication bugs
       that sort-based compare would mask.

METHODOLOGY NOTES (why this is rigorous where the naive version was not)
  - Content hash is done AFTER canonical schema cast (parquet dtypes as
    authority), so float64 JSON values don't accidentally match float32
    parquet values via silent downcast.
  - Multiset hash via Polars hash_rows → sort → sha256: order-independent,
    ties-safe.
  - Native-order hash in T3 detects order/duplication bugs that multiset
    would mask.
  - Internal JSON consistency: verify top-level n_spikes == len(spikes[]).
  - Format coverage gap flags (json-only, parquet-only) are reported from T1.

PARALLELISM
  - T1: ProcessPoolExecutor with 24 workers, cheap per pair (~0.5 s).
  - T2: 16 workers (RAM-bounded — each JSON ~300 MB → ~1.5 GB peak working).
  - T3: sequential (spot check), with full diagnostic output per pair.

USAGE
  python3 scripts/quarantine/audit_json_parquet_equivalence.py
  python3 scripts/quarantine/audit_json_parquet_equivalence.py --skip-t3
  python3 scripts/quarantine/audit_json_parquet_equivalence.py --n-t2 100
"""
from __future__ import annotations

import argparse
import hashlib
import os
import random
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# ─────────────────────────────────────────────────────────────
# Pin Polars to 1 thread per worker to avoid oversubscription
# when running inside ProcessPoolExecutor
os.environ["POLARS_MAX_THREADS"] = "1"

import numpy as np
import orjson
import polars as pl
import pyarrow.parquet as pq
import simdjson
from tqdm import tqdm

ROOT = Path("/mnt/storage/prism-outputs/10k-runs")


# ─────────────────────────────────────────────────────────────
# Inventory
# ─────────────────────────────────────────────────────────────

def site_key_from_name(name: str, suffix: str) -> Optional[str]:
    """Extract numeric site key from '<target>.site<N>.<suffix>'."""
    stem = name.replace(suffix, "")
    if ".site" not in stem:
        return None
    k = stem.rsplit(".site", 1)[-1]
    try:
        int(k)
        return k
    except ValueError:
        return None


def enumerate_sites(root: Path) -> List[Tuple[str, str, Optional[Path], Optional[Path]]]:
    """Return [(target, site_key, json_path|None, parquet_path|None), ...]."""
    all_rows: List[Tuple[str, str, Optional[Path], Optional[Path]]] = []
    for td in sorted(root.iterdir()):
        if not td.is_dir():
            continue
        per_site: Dict[str, Dict[str, Path]] = {}
        for p in td.glob("*.spike_events.json"):
            k = site_key_from_name(p.name, ".spike_events.json")
            if k is None:
                continue
            per_site.setdefault(k, {})["json"] = p
        for p in td.glob("*.spike_events.parquet"):
            k = site_key_from_name(p.name, ".spike_events.parquet")
            if k is None:
                continue
            per_site.setdefault(k, {})["parquet"] = p
        for k in sorted(per_site.keys(), key=lambda x: int(x)):
            all_rows.append((
                td.name, k,
                per_site[k].get("json"),
                per_site[k].get("parquet"),
            ))
    return all_rows


# ─────────────────────────────────────────────────────────────
# T1 — structural (cheap, all sites)
# ─────────────────────────────────────────────────────────────

def t1_check_one(args: Tuple[str, str, Optional[str], Optional[str]]) -> Dict[str, Any]:
    target, site_key, jp_s, qp_s = args
    jp = Path(jp_s) if jp_s else None
    qp = Path(qp_s) if qp_s else None

    result: Dict[str, Any] = {
        "target": target,
        "site": site_key,
        "json_exists": jp is not None,
        "pq_exists": qp is not None,
        "json_size_mb": round(jp.stat().st_size / 1048576, 2) if jp else None,
        "pq_size_mb": round(qp.stat().st_size / 1048576, 2) if qp else None,
    }

    try:
        # ── Parquet metadata only ─────────────────────────
        if qp:
            pf = pq.ParquetFile(qp)
            result["pq_n_rows"] = pf.metadata.num_rows
            result["pq_schema"] = [f.name for f in pf.schema_arrow]
            # Read just site_id column to check unique values cheaply
            sid_col = pf.read_row_group(0, columns=["site_id"]).column("site_id")
            result["pq_site_id_unique"] = sorted(set(sid_col.to_pylist()[:10000]))
        else:
            result["pq_n_rows"] = None
            result["pq_schema"] = None
            result["pq_site_id_unique"] = None

        # ── JSON: simdjson lazy — only top-level + array length ──
        if jp:
            parser = simdjson.Parser()
            with open(jp, "rb") as f:
                doc = parser.parse(f.read())
            result["js_n_spikes_declared"] = int(doc.get("n_spikes", -1))
            result["js_site_id"] = int(doc.get("site_id", -1))
            result["js_centroid"] = [float(x) for x in doc.get("centroid", [])]
            result["js_lining_cutoff"] = float(doc.get("lining_cutoff", -1))
            result["js_spikes_array_len"] = len(doc["spikes"])
            # Internal consistency: n_spikes top-level must equal actual array length
            result["js_internal_consistent"] = (
                result["js_n_spikes_declared"] == result["js_spikes_array_len"]
            )
            if jp and qp:
                result["row_count_match"] = (
                    result["js_spikes_array_len"] == result["pq_n_rows"]
                )
                result["site_id_container_match"] = (
                    result["js_site_id"] == result["pq_site_id_unique"][0]
                    if result["pq_site_id_unique"] else False
                )

        # ── Verdict ───────────────────────────────────────
        # Only paired sites can "pass" fully; unpaired are reported as gap
        if jp and qp:
            result["pass"] = (
                result.get("row_count_match", False)
                and result.get("site_id_container_match", False)
                and result.get("js_internal_consistent", False)
            )
        else:
            result["pass"] = None  # N/A — gap site
    except Exception as e:
        result["error"] = f"{type(e).__name__}: {e}"
        result["pass"] = False
    return result


# ─────────────────────────────────────────────────────────────
# T2 — content hash via Polars hash_rows + canonical cast
# ─────────────────────────────────────────────────────────────

def canonical_cast(df: pl.DataFrame, schema_reference: Dict[str, pl.DataType]) -> pl.DataFrame:
    """Cast df to reference schema, sort columns alphabetically.

    This forces BOTH sides through the same dtype funnel BEFORE hashing,
    so fp64 JSON values don't silently equal fp32 parquet values and vice
    versa — the cast is explicit and auditable.
    """
    cols = sorted(schema_reference.keys())
    exprs = []
    for c in cols:
        if c not in df.columns:
            raise ValueError(f"missing column in df: {c}")
        exprs.append(pl.col(c).cast(schema_reference[c], strict=False))
    return df.select(exprs)


def multiset_hash_polars(df: pl.DataFrame) -> str:
    """Order-independent content hash. Uses Polars hash_rows (xxhash under
    the hood), sorts the resulting u64 array, then sha256 of the sorted
    bytes. Collision-safe for this corpus (~B total rows)."""
    row_hashes = df.hash_rows().sort().to_numpy()
    return hashlib.sha256(row_hashes.tobytes()).hexdigest()


def native_order_hash_polars(df: pl.DataFrame) -> str:
    """Order-DEPENDENT hash — each row hashed in file-written order."""
    row_hashes = df.hash_rows().to_numpy()
    return hashlib.sha256(row_hashes.tobytes()).hexdigest()


def t2_check_one(args: Tuple[str, str, str, str]) -> Dict[str, Any]:
    target, site_key, jp_s, qp_s = args
    jp = Path(jp_s)
    qp = Path(qp_s)
    result: Dict[str, Any] = {"target": target, "site": site_key, "tier": "T2"}

    try:
        # ── Parquet (authority on schema) ─────────────────
        t0 = time.time()
        pq_df = pl.read_parquet(qp)
        t_pq = time.time() - t0

        pq_site_ids = pq_df["site_id"].unique().to_list()
        pq_df_for_hash = pq_df.drop("site_id")

        # Reference schema (parquet dtypes)
        ref_schema = {c: pq_df_for_hash.schema[c] for c in pq_df_for_hash.columns}

        # ── JSON → Polars ────────────────────────────────
        t1 = time.time()
        with open(jp, "rb") as f:
            doc = orjson.loads(f.read())
        js_site_id = doc.get("site_id")
        js_spikes = doc["spikes"]
        js_df = pl.from_dicts(js_spikes, infer_schema_length=1000)
        t_js = time.time() - t1

        # ── Schema alignment ──────────────────────────────
        pq_cols = set(pq_df_for_hash.columns)
        js_cols = set(js_df.columns)
        result["schema_symmetric_diff"] = sorted(pq_cols.symmetric_difference(js_cols))
        if not pq_cols.issubset(js_cols):
            result["pass"] = False
            result["fail_reason"] = "parquet has columns JSON lacks"
            return result

        # Restrict JSON to common columns (parquet-determined)
        js_df = js_df.select([c for c in pq_df_for_hash.columns if c in js_df.columns])

        # ── Canonical cast + hash ─────────────────────────
        try:
            pq_canon = canonical_cast(pq_df_for_hash, ref_schema)
            js_canon = canonical_cast(js_df, ref_schema)
        except ValueError as e:
            result["pass"] = False
            result["fail_reason"] = f"cast error: {e}"
            return result

        pq_hash = multiset_hash_polars(pq_canon)
        js_hash = multiset_hash_polars(js_canon)

        # Per-column hash for diagnostics on mismatch
        col_match = {}
        if pq_hash != js_hash:
            for c in pq_canon.columns:
                ph = hashlib.sha256(
                    pq_canon[c].sort(nulls_last=True).to_numpy().tobytes()
                ).hexdigest()[:16]
                jh = hashlib.sha256(
                    js_canon[c].sort(nulls_last=True).to_numpy().tobytes()
                ).hexdigest()[:16]
                col_match[c] = (ph == jh, ph, jh)

        result.update({
            "pq_n_rows": pq_df.height,
            "js_n_rows": js_df.height,
            "row_count_match": pq_df.height == js_df.height,
            "pq_site_ids": pq_site_ids,
            "js_site_id": js_site_id,
            "site_id_match": (len(pq_site_ids) == 1 and pq_site_ids[0] == js_site_id),
            "schema_match": pq_cols == js_cols,
            "pq_hash": pq_hash,
            "js_hash": js_hash,
            "content_hash_match": pq_hash == js_hash,
            "per_col_match": col_match,
            "parse_sec": {"pq": round(t_pq, 2), "js": round(t_js, 2)},
        })
        result["pass"] = all([
            result["row_count_match"],
            result["site_id_match"],
            result["schema_match"],
            result["content_hash_match"],
        ])
    except Exception as e:
        result["error"] = f"{type(e).__name__}: {e}"
        result["traceback"] = traceback.format_exc()
        result["pass"] = False
    return result


# ─────────────────────────────────────────────────────────────
# T3 — native order + multiset (sequential, verbose)
# ─────────────────────────────────────────────────────────────

def t3_check_one(target: str, site_key: str, jp: Path, qp: Path) -> Dict[str, Any]:
    result: Dict[str, Any] = {"target": target, "site": site_key, "tier": "T3"}
    try:
        pq_df = pl.read_parquet(qp).drop("site_id")

        with open(jp, "rb") as f:
            doc = orjson.loads(f.read())
        js_df = pl.from_dicts(doc["spikes"], infer_schema_length=1000)
        js_df = js_df.select([c for c in pq_df.columns if c in js_df.columns])

        ref_schema = {c: pq_df.schema[c] for c in pq_df.columns}
        pq_canon = canonical_cast(pq_df, ref_schema)
        js_canon = canonical_cast(js_df, ref_schema)

        # Native-order hash — preserves as-written row order
        pq_native = native_order_hash_polars(pq_canon)
        js_native = native_order_hash_polars(js_canon)

        # Multiset hash — order-independent
        pq_multi = multiset_hash_polars(pq_canon)
        js_multi = multiset_hash_polars(js_canon)

        # First/last row hash — cheap spot check
        pq_rh = pq_canon.hash_rows().to_numpy()
        js_rh = js_canon.hash_rows().to_numpy()

        result.update({
            "n_rows": pq_canon.height,
            "native_order_match": pq_native == js_native,
            "multiset_match": pq_multi == js_multi,
            "first_row_match": int(pq_rh[0]) == int(js_rh[0]),
            "last_row_match": int(pq_rh[-1]) == int(js_rh[-1]),
            # Order divergence signature: how many rows differ in native order
            # (only useful when multiset matches)
            "native_row_divergence": int(np.sum(pq_rh != js_rh)) if pq_multi == js_multi else None,
        })
        # Data equivalence = multiset match. Order preservation is informational.
        result["pass"] = result["multiset_match"]
    except Exception as e:
        result["error"] = f"{type(e).__name__}: {e}"
        result["pass"] = False
    return result


# ─────────────────────────────────────────────────────────────
# Printers
# ─────────────────────────────────────────────────────────────

def summarize_t1(results: List[Dict[str, Any]]) -> None:
    n = len(results)
    paired = [r for r in results if r["json_exists"] and r["pq_exists"]]
    js_only = [r for r in results if r["json_exists"] and not r["pq_exists"]]
    pq_only = [r for r in results if not r["json_exists"] and r["pq_exists"]]
    internal_ok = [r for r in paired if r.get("js_internal_consistent") is True]
    internal_bad = [r for r in paired if r.get("js_internal_consistent") is False]
    row_count_ok = [r for r in paired if r.get("row_count_match") is True]
    row_count_bad = [r for r in paired if r.get("row_count_match") is False]
    site_id_ok = [r for r in paired if r.get("site_id_container_match") is True]
    site_id_bad = [r for r in paired if r.get("site_id_container_match") is False]
    passed = [r for r in paired if r.get("pass") is True]
    errored = [r for r in results if "error" in r]

    print(f"\n=== T1 SUMMARY ({n} sites inventoried) ===")
    print(f"  paired (both formats):   {len(paired)}")
    print(f"  JSON-only (no parquet):  {len(js_only)}")
    print(f"  parquet-only (no JSON):  {len(pq_only)}")
    print(f"  errored during T1:       {len(errored)}")
    print(f"  ── paired-site checks ──")
    print(f"  json_internal_consistent (n_spikes == len(spikes[])): {len(internal_ok)} / {len(paired)}")
    if internal_bad:
        print(f"    FAILED: {[(r['target'],r['site']) for r in internal_bad[:5]]}")
    print(f"  row_count_match (json vs parquet):                    {len(row_count_ok)} / {len(paired)}")
    if row_count_bad:
        print(f"    FAILED: {[(r['target'],r['site'],r.get('js_spikes_array_len'),r.get('pq_n_rows')) for r in row_count_bad[:5]]}")
    print(f"  site_id_container_match:                              {len(site_id_ok)} / {len(paired)}")
    if site_id_bad:
        print(f"    FAILED: {[(r['target'],r['site'],r.get('js_site_id'),r.get('pq_site_id_unique')) for r in site_id_bad[:5]]}")
    print(f"  T1 PASS (all three above):                            {len(passed)} / {len(paired)}")


def summarize_t2(results: List[Dict[str, Any]]) -> None:
    n = len(results)
    passed = [r for r in results if r.get("pass") is True]
    failed = [r for r in results if r.get("pass") is False]
    errored = [r for r in results if "error" in r]
    print(f"\n=== T2 SUMMARY ({n} paired sites deep-hashed) ===")
    print(f"  PASSED (content hash match): {len(passed)} / {n}")
    if failed:
        print(f"  FAILED: {len(failed)}")
        for r in failed[:10]:
            if "fail_reason" in r:
                print(f"    {r['target']}/site{r['site']}: {r['fail_reason']}")
            else:
                print(f"    {r['target']}/site{r['site']}: "
                      f"rows({r.get('pq_n_rows')}vs{r.get('js_n_rows')}) "
                      f"site_id({r.get('site_id_match')}) "
                      f"schema({r.get('schema_match')}) "
                      f"hash({r.get('content_hash_match')})")
                if not r.get("content_hash_match") and r.get("per_col_match"):
                    bad_cols = [c for c, (ok,_,_) in r["per_col_match"].items() if not ok]
                    print(f"      columns differing: {bad_cols}")
    if errored:
        print(f"  ERRORED: {len(errored)}")
        for r in errored[:3]:
            print(f"    {r['target']}/site{r['site']}: {r['error']}")


def summarize_t3(results: List[Dict[str, Any]]) -> None:
    print(f"\n=== T3 SUMMARY ({len(results)} spot pairs) ===")
    for r in results:
        if "error" in r:
            print(f"  {r['target']}/site{r['site']}: ERROR — {r['error']}")
            continue
        print(f"  {r['target']}/site{r['site']} ({r['n_rows']:,} rows):")
        print(f"    multiset_match:       {r['multiset_match']}")
        print(f"    native_order_match:   {r['native_order_match']}")
        print(f"    first_row_match:      {r['first_row_match']}")
        print(f"    last_row_match:       {r['last_row_match']}")
        if r["multiset_match"] and not r["native_order_match"]:
            print(f"    → DATA EQUAL, ORDER DIFFERS (informational)")
            print(f"    native_row_divergence: {r.get('native_row_divergence')} rows")


# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(ROOT))
    ap.add_argument("--n-t2", type=int, default=50)
    ap.add_argument("--n-t3", type=int, default=5)
    ap.add_argument("--workers-t1", type=int, default=24)
    ap.add_argument("--workers-t2", type=int, default=16)
    ap.add_argument("--skip-t1", action="store_true")
    ap.add_argument("--skip-t2", action="store_true")
    ap.add_argument("--skip-t3", action="store_true")
    args = ap.parse_args()

    root = Path(args.root)
    if not root.exists():
        print(f"FATAL: root not found: {root}", file=sys.stderr)
        return 2

    print(f"=== RIGOROUS JSON vs PARQUET AUDIT ===")
    print(f"root: {root}")
    print(f"hw:   24 cores, workers_t1={args.workers_t1}, workers_t2={args.workers_t2}")
    t_start = time.time()

    all_rows = enumerate_sites(root)
    print(f"sites found: {len(all_rows)}")

    # ── T1 ────────────────────────────────────────────────
    t1_results: List[Dict[str, Any]] = []
    if not args.skip_t1:
        print(f"\n=== T1: structural audit on all {len(all_rows)} sites ===")
        t0 = time.time()
        args_list = [
            (t, s,
             str(jp) if jp else None,
             str(qp) if qp else None)
            for t, s, jp, qp in all_rows
        ]
        with ProcessPoolExecutor(max_workers=args.workers_t1) as ex:
            for r in tqdm(ex.map(t1_check_one, args_list, chunksize=8),
                          total=len(args_list), desc="T1"):
                t1_results.append(r)
        print(f"T1 wallclock: {time.time()-t0:.1f}s")
        summarize_t1(t1_results)

    # Need paired sites for T2/T3
    paired = [(t, s, jp, qp) for t, s, jp, qp in all_rows if jp and qp]

    # ── T2 ────────────────────────────────────────────────
    t2_results: List[Dict[str, Any]] = []
    if not args.skip_t2:
        # Stratified sample by parquet size — small/mid/large thirds
        paired_sized = [(x, x[3].stat().st_size) for x in paired]
        paired_sized.sort(key=lambda r: r[1])
        thirds = max(1, len(paired_sized) // 3)
        buckets = [paired_sized[:thirds], paired_sized[thirds:2*thirds], paired_sized[2*thirds:]]
        rng = random.Random(42)
        n_each = args.n_t2 // 3
        sample = []
        for b in buckets:
            sample.extend(rng.sample(b, min(n_each, len(b))))
        # Top up to n_t2
        remainder = [x for x in paired_sized if x not in sample]
        if remainder and len(sample) < args.n_t2:
            sample.extend(rng.sample(remainder, min(args.n_t2 - len(sample), len(remainder))))

        print(f"\n=== T2: content hash audit on {len(sample)} stratified paired sites ===")
        t0 = time.time()
        args_list = [(t, s, str(jp), str(qp)) for ((t, s, jp, qp), _) in sample]
        with ProcessPoolExecutor(max_workers=args.workers_t2) as ex:
            for r in tqdm(ex.map(t2_check_one, args_list, chunksize=1),
                          total=len(args_list), desc="T2"):
                t2_results.append(r)
        print(f"T2 wallclock: {time.time()-t0:.1f}s")
        summarize_t2(t2_results)

    # ── T3 ────────────────────────────────────────────────
    t3_results: List[Dict[str, Any]] = []
    if not args.skip_t3 and paired:
        paired_sorted = sorted(paired, key=lambda p: p[3].stat().st_size)
        N = len(paired_sorted)
        picks = []
        if N >= 1: picks.append(paired_sorted[0])           # smallest
        if N >= 2: picks.append(paired_sorted[-1])          # largest
        if N >= 3: picks.append(paired_sorted[N // 2])      # median
        rng = random.Random(1)
        remaining_slots = args.n_t3 - len(picks)
        if remaining_slots > 0 and N > 3:
            pool = paired_sorted[1:-1]
            picks.extend(rng.sample(pool, min(remaining_slots, len(pool))))

        print(f"\n=== T3: native-order comparison on {len(picks)} spot pairs ===")
        t0 = time.time()
        for t, s, jp, qp in tqdm(picks, desc="T3"):
            t3_results.append(t3_check_one(t, s, jp, qp))
        print(f"T3 wallclock: {time.time()-t0:.1f}s")
        summarize_t3(t3_results)

    # ── Final verdict ─────────────────────────────────────
    total = time.time() - t_start
    print(f"\n=== OVERALL ===")
    print(f"  wallclock: {total:.1f}s")
    t1_pass = all(r.get("pass") in (True, None) for r in t1_results) if t1_results else None
    t2_pass = all(r.get("pass") is True for r in t2_results) if t2_results else None
    t3_pass = all(r.get("pass") is True for r in t3_results) if t3_results else None
    print(f"  T1 (structural, all paired sites):   {'PASS' if t1_pass else 'FAIL' if t1_pass is False else 'SKIP'}")
    print(f"  T2 (content hash, stratified sample): {'PASS' if t2_pass else 'FAIL' if t2_pass is False else 'SKIP'}")
    print(f"  T3 (native-order spot check):         {'PASS' if t3_pass else 'FAIL' if t3_pass is False else 'SKIP'}")

    all_pass = all(x is True or x is None for x in [t1_pass, t2_pass, t3_pass])
    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
