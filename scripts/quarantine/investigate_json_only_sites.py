#!/usr/bin/env python3
"""
[DIAGNOSTIC - READ-ONLY]

For every JSON-only spike_events site (JSON exists, no matching parquet),
determine whether its spike data is:
  REDUNDANT — every spike-tuple also present in some parquet in the same target
  UNIQUE    — spike-tuples not found in any target parquet (data would be lost
              if we dropped the JSON)

Also cross-references the JSON-only site_id against the target's canonical
metadata in binding_sites.json:
  - sites[] (canonical 40)
  - all_pockets[]
  - prism_therm.sites[]
  - prism_therm.global_pockets[]

Spike identity:
  Hash tuple = (timestep, frame_index, stream_id, x, y, z) cast to
  (int32, int32, int8, float32, float32, float32). Using Polars
  hash_rows for speed. Subset check = set-containment of u64 hashes.

Output:
  Per-site verdict + summary.

Runtime:
  ~1-2 min (union of ~40 parquets per target via Polars, hash_rows, set ops).
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import orjson
import polars as pl


ROOT = Path("/mnt/storage/prism-outputs/10k-runs")

# The six identity columns for spike deduplication
ID_COLS = ["timestep", "frame_index", "stream_id", "x", "y", "z"]


def find_json_only_sites(root: Path) -> List[Tuple[str, str, Path]]:
    """Return [(target, site_key, json_path)] where parquet is absent."""
    out = []
    for td in sorted(root.iterdir()):
        if not td.is_dir():
            continue
        jsons: Dict[str, Path] = {}
        pqts: Dict[str, Path] = {}
        for p in td.glob("*.spike_events.json"):
            stem = p.stem.replace(".spike_events", "")
            if ".site" not in stem:
                continue
            k = stem.rsplit(".site", 1)[-1]
            try:
                int(k)
                jsons[k] = p
            except ValueError:
                continue
        for p in td.glob("*.spike_events.parquet"):
            stem = p.stem.replace(".spike_events", "")
            if ".site" not in stem:
                continue
            k = stem.rsplit(".site", 1)[-1]
            try:
                int(k)
                pqts[k] = p
            except ValueError:
                continue
        for k, jp in jsons.items():
            if k not in pqts:
                out.append((td.name, k, jp))
    return out


def canonical_id_frame(df: pl.DataFrame) -> pl.DataFrame:
    """Project to the 6 identity columns with a canonical dtype schema."""
    casts = [
        pl.col("timestep").cast(pl.Int32),
        pl.col("frame_index").cast(pl.Int32),
        pl.col("stream_id").cast(pl.Int8),
        pl.col("x").cast(pl.Float32),
        pl.col("y").cast(pl.Float32),
        pl.col("z").cast(pl.Float32),
    ]
    return df.select(casts)


def json_spikes_to_frame(jp: Path) -> Tuple[int, int, pl.DataFrame]:
    """Return (site_id_container, n_spikes, canonical identity frame)."""
    with open(jp, "rb") as f:
        doc = orjson.loads(f.read())
    site_id = int(doc.get("site_id", -1))
    spikes = doc["spikes"]
    df = pl.from_dicts(spikes, infer_schema_length=1000)
    return site_id, len(spikes), canonical_id_frame(df)


def target_parquet_union_hashes(target_dir: Path) -> Set[int]:
    """Union of spike-identity-row hashes across ALL parquets in target."""
    union: Set[int] = set()
    for p in target_dir.glob("*.spike_events.parquet"):
        try:
            df = pl.read_parquet(p)
            # Parquets include site_id; drop it — identity is the 6-col tuple
            cols = [c for c in ID_COLS if c in df.columns]
            if len(cols) != len(ID_COLS):
                continue
            idf = canonical_id_frame(df.select(cols))
            hashes = idf.hash_rows().to_numpy()
            union.update(hashes.tolist())
        except Exception as e:
            print(f"    WARN parquet read error {p.name}: {e}", file=sys.stderr)
    return union


def load_binding_sites_refs(target_dir: Path, target: str) -> Dict[str, Set[int]]:
    """Extract referenced site/pocket IDs from binding_sites.json."""
    refs: Dict[str, Set[int]] = {
        "sites": set(),
        "all_pockets": set(),
        "prism_therm.sites": set(),
        "prism_therm.global_pockets": set(),
        "cryptic_sites": set(),
    }
    bp = target_dir / f"{target}.binding_sites.json"
    if not bp.exists():
        return refs
    try:
        with open(bp, "rb") as f:
            bs = orjson.loads(f.read())
    except Exception:
        return refs
    for s in bs.get("sites", []):
        sid = s.get("id", s.get("site_id"))
        if sid is not None:
            refs["sites"].add(int(sid))
    for s in bs.get("all_pockets", []):
        sid = s.get("site_id", s.get("id"))
        if sid is not None:
            refs["all_pockets"].add(int(sid))
    for s in bs.get("cryptic_sites", []):
        sid = s.get("site_id", s.get("id"))
        if sid is not None:
            refs["cryptic_sites"].add(int(sid))
    pt = bs.get("prism_therm", {})
    for s in pt.get("sites", []):
        sid = s.get("site_id", s.get("id"))
        if sid is not None:
            refs["prism_therm.sites"].add(int(sid))
    for s in pt.get("global_pockets", []):
        sid = s.get("site_id", s.get("id"))
        if sid is not None:
            refs["prism_therm.global_pockets"].add(int(sid))
    return refs


def main() -> int:
    t0 = time.time()
    root = ROOT
    json_only = find_json_only_sites(root)
    print(f"=== JSON-only investigation ===")
    print(f"root: {root}")
    print(f"JSON-only sites found: {len(json_only)}")
    if not json_only:
        print("nothing to do")
        return 0

    # Group by target to minimize repeated parquet reads
    by_target: Dict[str, List[Tuple[str, Path]]] = {}
    for t, k, jp in json_only:
        by_target.setdefault(t, []).append((k, jp))

    print(f"\nAffected targets: {len(by_target)}")
    for t, items in by_target.items():
        sids = [k for k, _ in items]
        print(f"  {t}: sites={sids}")

    print()
    print("=" * 78)
    print(f"{'target':<18} {'site_id':>8} {'n_spikes':>10} {'json_MB':>8} "
          f"{'overlap%':>10} {'verdict':<12} {'metadata_ref'}")
    print("=" * 78)

    results: List[Dict[str, Any]] = []
    for t, items in by_target.items():
        td = root / t
        # Compute parquet union ONCE per target
        tstart = time.time()
        pq_union = target_parquet_union_hashes(td)
        bs_refs = load_binding_sites_refs(td, t)
        load_time = time.time() - tstart

        for site_key, jp in items:
            try:
                j_site_id, n_spikes, j_id_df = json_spikes_to_frame(jp)
                j_hashes = j_id_df.hash_rows().to_numpy()
                j_set = set(j_hashes.tolist())
                # Set-containment: how many JSON spikes are also present in union
                overlap = len(j_set & pq_union)
                overlap_pct = 100.0 * overlap / len(j_set) if j_set else 0.0

                if overlap_pct >= 99.9:
                    verdict = "REDUNDANT"
                elif overlap_pct == 0.0:
                    verdict = "UNIQUE"
                else:
                    verdict = "PARTIAL"

                # Where does this site_id appear in metadata?
                refs_present = []
                for name, ids in bs_refs.items():
                    if int(site_key) in ids or j_site_id in ids:
                        refs_present.append(name)
                meta_ref = ",".join(refs_present) if refs_present else "ORPHAN"

                js_mb = jp.stat().st_size / 1048576

                results.append({
                    "target": t, "site_key": site_key,
                    "json_site_id": j_site_id, "n_spikes": n_spikes,
                    "json_mb": round(js_mb, 1),
                    "overlap_pct": round(overlap_pct, 3),
                    "verdict": verdict, "metadata_ref": meta_ref,
                })
                print(f"{t:<18} {site_key:>8} {n_spikes:>10} {js_mb:>8.1f} "
                      f"{overlap_pct:>9.2f}% {verdict:<12} {meta_ref}")
            except Exception as e:
                print(f"{t:<18} {site_key:>8} ERROR: {type(e).__name__}: {e}")
                results.append({
                    "target": t, "site_key": site_key, "error": str(e),
                })

    # Summary
    print()
    print("=" * 78)
    verdicts: Dict[str, int] = {}
    for r in results:
        v = r.get("verdict", "ERROR")
        verdicts[v] = verdicts.get(v, 0) + 1
    print("VERDICT TALLY:")
    for v, n in sorted(verdicts.items()):
        print(f"  {v}: {n}")

    unique_sites = [r for r in results if r.get("verdict") == "UNIQUE"]
    partial_sites = [r for r in results if r.get("verdict") == "PARTIAL"]
    if unique_sites:
        print(f"\nUNIQUE sites (data would be LOST if JSON dropped):")
        for r in unique_sites:
            print(f"  {r['target']}/site{r['site_key']}  "
                  f"n_spikes={r['n_spikes']}  meta={r['metadata_ref']}")
    if partial_sites:
        print(f"\nPARTIAL sites (some spikes present in parquets, some not):")
        for r in partial_sites:
            print(f"  {r['target']}/site{r['site_key']}  "
                  f"overlap={r['overlap_pct']:.2f}%  meta={r['metadata_ref']}")

    print(f"\nwallclock: {time.time()-t0:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
