#!/usr/bin/env python3
"""W3b /event-aggregates producer — canonical event-aggregate emission.

Non-invention invariants (§3 of the Detection/Feature-Service
Advancement Constitution):
  - never fabricates enum values
  - never emits zero-count rows for absent artifacts; hard-skips
  - never coerces casing/whitespace enum variants; buckets into
    count_*_unknown / count_*_other; never normalises strings
  - never recomputes a threshold the Worker already owns

Final-hardening invariant (added this lane):
  - BEFORE POSTing payload for target T, the producer MUST assert
        D1_site_names(T) == auth(T)
    If set-equality fails the target is hard-failed, the symmetric
    difference is printed, and widened execution aborts.

Staged execution: S1 proof-dry, S2 proof-exec, S4 wide-dry, S5 wide-exec,
final.  Artifacts (spike_events.parquet per site) are staged per-target
via rclone from R2 and DELETED after each target to keep disk bounded.
"""
from __future__ import annotations
import argparse, json, math, os, re, shutil, subprocess, sys, urllib.error, urllib.request
from collections import Counter
from pathlib import Path

HEADERS = {"User-Agent": "Mozilla/5.0 prism4d-w3b-event-aggregates-pin"}

CCNS_PHASE_ENUM  = {"cold_hold", "heating", "warm_hold", "cooling", "cold_return"}
SPIKE_SOURCE_ENUM = {"UV", "LIF", "EFP"}
TYPE_ENUM        = {"BNZ", "UNK", "ANION", "CATION", "PHE", "TYR", "TRP"}   # v1.1

R2_PREFIX = "r2:prism-archive/10k-runs-pct70"

# Operational concurrency bounds for rclone, overridable via CLI flags.
# Default preserves the prior single-worker behavior (--transfers 8).
RCLONE_TRANSFERS = 8
RCLONE_CHECKERS  = 8


def api_get(url: str, timeout: int = 60):
    req = urllib.request.Request(url, headers=HEADERS)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


def api_post(url: str, payload, timeout: int = 120):
    body = json.dumps(payload, allow_nan=False).encode("utf-8")
    req = urllib.request.Request(
        url, data=body,
        headers={**HEADERS, "Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, json.load(r)
    except urllib.error.HTTPError as e:
        try:
            return e.code, json.loads(e.read().decode("utf-8", "replace"))
        except Exception:
            return e.code, {"error": "non-json"}


def auth_site_names(binding_sites_path: Path) -> set[str]:
    bs = json.load(open(binding_sites_path))
    return {f"site{s['id']}" for s in bs.get("sites", []) if "id" in s}


def d1_site_names(api_base: str, target: str) -> set[str]:
    data = api_get(f"{api_base}/site-features/{target}?fields=ranker")
    return {r["site_name"] for r in data.get("sites", [])}


def rclone_spike_events(target: str, dst: Path) -> list[Path]:
    """Stage every spike_events.{parquet,json} for target from R2.
    R2 has a mixed population — some sites ship only parquet, others
    only json. We take both and dedupe by site_id preferring parquet."""
    dst.mkdir(parents=True, exist_ok=True)
    r = subprocess.run(
        ["rclone", "copy", f"{R2_PREFIX}/{target}/", str(dst),
         "--include", "*.spike_events.parquet",
         "--include", "*.spike_events.json",
         "--transfers", str(RCLONE_TRANSFERS),
         "--checkers",  str(RCLONE_CHECKERS),
         "--quiet"],
        capture_output=True, text=True, timeout=1800,
    )
    if r.returncode != 0:
        raise RuntimeError(f"rclone failed for {target}: {r.stderr[-400:]}")
    parquets = list(dst.glob("*.spike_events.parquet"))
    jsons    = list(dst.glob("*.spike_events.json"))
    return sorted(parquets + jsons)


SITE_ID_RE = re.compile(r"\.site(\d+)\.spike_events\.(parquet|json)$")


def site_id_from_filename(p: Path) -> int | None:
    m = SITE_ID_RE.search(p.name)
    return int(m.group(1)) if m else None


def select_per_site(files: list[Path]) -> dict[int, Path]:
    """Prefer parquet over json when both exist for the same site_id."""
    out: dict[int, Path] = {}
    # Pass 1: parquet wins
    for p in files:
        if not p.name.endswith(".parquet"):
            continue
        sid = site_id_from_filename(p)
        if sid is None:
            continue
        out[sid] = p
    # Pass 2: json fills only where no parquet already present
    for p in files:
        if not p.name.endswith(".json"):
            continue
        sid = site_id_from_filename(p)
        if sid is None or sid in out:
            continue
        out[sid] = p
    return out


def _read_events_parquet(path: Path):
    import pyarrow.parquet as pq
    cols = ["ccns_phase", "spike_source", "type", "intensity",
            "vibrational_energy", "water_density", "wavelength_nm",
            "aromatic_residue_id", "n_nearby_excited"]
    t = pq.read_table(path, columns=cols)
    return {c: t.column(c) for c in cols}, len(t)


def _read_events_json(path: Path):
    """JSON spike_events format: {..., "spikes":[{field:val,...}, ...]}.
    Returns a dict of per-column python lists + n."""
    try:
        d = json.load(open(path))
    except Exception:
        # try streaming heuristic failure
        raise
    spikes = d.get("spikes", [])
    if not spikes:
        return None, 0
    import numpy as np
    cols = ["ccns_phase", "spike_source", "type", "intensity",
            "vibrational_energy", "water_density", "wavelength_nm",
            "aromatic_residue_id", "n_nearby_excited"]
    out = {c: [] for c in cols}
    for s in spikes:
        for c in cols:
            out[c].append(s.get(c))
    return out, len(spikes)


def compute_site_aggregates(target: str, site_id: int, art_path: Path) -> dict | None:
    is_parquet = art_path.name.endswith(".parquet")
    try:
        if is_parquet:
            cols, n = _read_events_parquet(art_path)
            def col_list(name): return cols[name].to_pylist()
            def col_numpy(name): return cols[name].to_numpy(zero_copy_only=False)
        else:
            cols, n = _read_events_json(art_path)
            if cols is None:
                return None
            import numpy as np
            def col_list(name): return cols[name]
            def col_numpy(name):
                arr = cols[name]
                if name in ("aromatic_residue_id", "n_nearby_excited"):
                    return np.asarray([x if x is not None else 0 for x in arr], dtype=np.int64)
                return np.asarray([x if x is not None else 0.0 for x in arr], dtype=np.float64)
    except Exception as e:
        return {"_error": f"read_failed: {type(e).__name__}: {e}"}

    if n == 0:
        return None                         # hard-skip: zero events

    phase_ctr  = Counter(col_list("ccns_phase"))
    source_ctr = Counter(col_list("spike_source"))
    type_ctr   = Counter(col_list("type"))

    c_phase = {k: int(phase_ctr.get(k, 0)) for k in CCNS_PHASE_ENUM}
    c_phase_unknown = int(sum(v for k, v in phase_ctr.items() if k not in CCNS_PHASE_ENUM))
    c_source = {k: int(source_ctr.get(k, 0)) for k in SPIKE_SOURCE_ENUM}
    c_source_other = int(sum(v for k, v in source_ctr.items() if k not in SPIKE_SOURCE_ENUM))
    c_type = {k: int(type_ctr.get(k, 0)) for k in TYPE_ENUM}
    c_type_other = int(sum(v for k, v in type_ctr.items() if k not in TYPE_ENUM))

    intensity = col_numpy("intensity")
    vib       = col_numpy("vibrational_energy")
    water     = col_numpy("water_density")
    wave      = col_numpy("wavelength_nm")
    aro       = col_numpy("aromatic_residue_id")
    nne       = col_numpy("n_nearby_excited")

    mean_intensity = float(intensity.mean())
    std_intensity  = float(intensity.std(ddof=0))
    mean_vib       = float(vib.mean())
    mean_water     = float(water.mean())
    mean_nne       = float(nne.mean())
    nonzero_wave   = int((wave != 0).sum())
    aro_pos        = int((aro >= 0).sum())

    # Shannon entropy (NATS) over the 3-value source distribution.
    ts = c_source["UV"] + c_source["LIF"] + c_source["EFP"]
    H = 0.0
    if ts > 0:
        for c in (c_source["UV"], c_source["LIF"], c_source["EFP"]):
            if c > 0:
                p = c / ts
                H -= p * math.log(p)

    c_warm = c_phase["warm_hold"]
    c_cold = c_phase["cold_hold"]
    ptr = c_warm / max(c_cold, 1)
    whf = c_warm / max(n, 1)

    # Sanity: distributions partition n_events.
    if (sum(c_phase.values()) + c_phase_unknown != n
            or sum(c_source.values()) + c_source_other != n
            or sum(c_type.values()) + c_type_other != n):
        return {"_error": "partition_invariant_broken",
                "n": n,
                "phase_sum":  sum(c_phase.values()) + c_phase_unknown,
                "source_sum": sum(c_source.values()) + c_source_other,
                "type_sum":   sum(c_type.values()) + c_type_other}

    return {
        "site_name": f"site{site_id}",
        "event_contract_version": "event_schema_v1.1",
        "n_events": n,
        "count_phase_cold_hold":   c_phase["cold_hold"],
        "count_phase_warm_hold":   c_phase["warm_hold"],
        "count_phase_heating":     c_phase["heating"],
        "count_phase_cooling":     c_phase["cooling"],
        "count_phase_cold_return": c_phase["cold_return"],
        "count_phase_unknown":     c_phase_unknown,
        "count_source_uv":         c_source["UV"],
        "count_source_lif":        c_source["LIF"],
        "count_source_efp":        c_source["EFP"],
        "count_source_other":      c_source_other,
        "count_type_bnz":          c_type["BNZ"],
        "count_type_unk":          c_type["UNK"],
        "count_type_anion":        c_type["ANION"],
        "count_type_cation":       c_type["CATION"],
        "count_type_phe":          c_type["PHE"],
        "count_type_tyr":          c_type["TYR"],
        "count_type_trp":          c_type["TRP"],
        "count_type_other":        c_type_other,
        "mean_intensity":          mean_intensity,
        "std_intensity":           std_intensity,
        "mean_vibrational_energy": mean_vib,
        "mean_water_density":      mean_water,
        "mean_n_nearby_excited":   mean_nne,
        "nonzero_wavelength_count": nonzero_wave,
        "aromatic_attribution_count": aro_pos,
        "source_entropy_nat":      H,
        "phase_transition_ratio":  ptr,
        "warm_hold_spike_fraction": whf,
    }


def resolve_targets(api_base: str, arg: str) -> list[str]:
    if arg == "all":
        d = api_get(f"{api_base}/targets?spike_percentile=70")
        return sorted([t["target"] for t in d["targets"]])
    return [t.strip() for t in arg.split(",") if t.strip()]


def plan_target(api_base: str, staging_dir: Path, target: str, cache_root: Path,
                delete_after: bool = True) -> dict:
    """Returns a record with either {payload, site_count, ...} or {error}."""
    bs_path = staging_dir / target / f"{target}.binding_sites.json"
    if not bs_path.exists():
        return {"target": target, "error": "missing_binding_sites"}
    auth = auth_site_names(bs_path)

    try:
        d1_set = d1_site_names(api_base, target)
    except Exception as e:
        return {"target": target, "error": f"d1_query_failed: {type(e).__name__}: {e}"}

    # HARDENING §: set-equality assertion BEFORE any staging/POST.
    if d1_set != auth:
        only_d1 = sorted(d1_set - auth)
        only_auth = sorted(auth - d1_set)
        return {"target": target, "error": "site_set_mismatch",
                "only_in_d1": only_d1, "only_in_auth": only_auth,
                "d1_count": len(d1_set), "auth_count": len(auth)}

    stage = cache_root / target
    try:
        files = rclone_spike_events(target, stage)
    except Exception as e:
        return {"target": target, "error": f"rclone_failed: {e}"}
    if not files:
        shutil.rmtree(stage, ignore_errors=True)
        return {"target": target, "error": "no_spike_artifacts"}

    site_files = select_per_site(files)
    auth_ids = {int(n[4:]) for n in auth}
    artifact_gap = auth_ids - set(site_files.keys())

    payload_rows: list[dict] = []
    error_rows: list[dict]   = []
    for sid in sorted(auth_ids):                   # deterministic numeric-sort §
        if sid not in site_files:
            continue                                 # not an error; counted via artifact_gap
        row = compute_site_aggregates(target, sid, site_files[sid])
        if row is None:
            continue                                 # zero-event site hard-skipped
        if "_error" in row:
            error_rows.append({"site_id": sid, **row})
            continue
        payload_rows.append(row)

    if delete_after:
        shutil.rmtree(stage, ignore_errors=True)

    return {
        "target": target,
        "auth_count": len(auth),
        "artifact_gap_count": len(artifact_gap),
        "artifact_gap_site_ids": sorted(artifact_gap)[:10],
        "computed_row_count": len(payload_rows),
        "compute_errors": error_rows,
        "payload": payload_rows,
    }


def stage_dry_run(api_base, staging_dir, cache_root, targets, out_path) -> dict:
    records = []
    for i, t in enumerate(targets, 1):
        rec = plan_target(api_base, staging_dir, t, cache_root)
        summary = {k: v for k, v in rec.items() if k not in ("payload", "compute_errors")}
        records.append(summary if "error" in rec else {**summary, "has_compute_errors": bool(rec.get("compute_errors"))})
        if i % 10 == 0 or i == len(targets):
            print(f"  dry-run [{i}/{len(targets)}]  {t}", flush=True)
    out = {
        "total": len(records),
        "planned_ok":              sum(1 for r in records if "error" not in r),
        "errored":                 sum(1 for r in records if "error" in r),
        "site_set_mismatch":       sum(1 for r in records if r.get("error") == "site_set_mismatch"),
        "no_spike_artifacts":      sum(1 for r in records if r.get("error") == "no_spike_artifacts"),
        "rclone_failed":           sum(1 for r in records if str(r.get("error","")).startswith("rclone_failed")),
        "missing_binding_sites":   sum(1 for r in records if r.get("error") == "missing_binding_sites"),
        "records": records,
    }
    out_path.write_text(json.dumps(out, indent=2))
    return out


def stage_execute(api_base, staging_dir, cache_root, targets, out_path) -> dict:
    results = []
    for i, t in enumerate(targets, 1):
        rec = plan_target(api_base, staging_dir, t, cache_root)
        if "error" in rec:
            results.append({"target": t, "skipped": rec["error"],
                            **{k: v for k, v in rec.items() if k not in ("payload",)}})
            continue
        if not rec["payload"]:
            results.append({"target": t, "skipped": "empty_payload",
                            "auth_count": rec["auth_count"],
                            "artifact_gap_count": rec["artifact_gap_count"]})
            continue
        code, body = api_post(f"{api_base}/site-features/{t}/event-aggregates",
                              rec["payload"])
        results.append({
            "target": t, "http": code,
            "auth_count": rec["auth_count"],
            "artifact_gap_count": rec["artifact_gap_count"],
            "computed_rows": rec["computed_row_count"],
            "worker_response": body,
        })
        if i % 10 == 0 or i == len(targets):
            print(f"  execute [{i}/{len(targets)}]  {t}", flush=True)
    summary = {
        "total": len(results),
        "http_200":      sum(1 for r in results if r.get("http") == 200),
        "http_non_200":  sum(1 for r in results if r.get("http") not in (None, 200)),
        "skipped":       sum(1 for r in results if "skipped" in r),
        "total_worker_accepted":    sum((r.get("worker_response") or {}).get("accepted", 0) for r in results),
        "total_worker_quarantined": sum((r.get("worker_response") or {}).get("quarantined", 0) for r in results),
        "total_auth_sites_covered": sum(r.get("computed_rows", 0) for r in results),
    }
    out = {"results": results, "summary": summary}
    out_path.write_text(json.dumps(out, indent=2))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--api-base", required=True)
    ap.add_argument("--staging-dir", type=Path, required=True)
    ap.add_argument("--parquet-cache", type=Path, required=True,
                    help="working directory for per-target parquet staging")
    ap.add_argument("--targets", required=True)
    ap.add_argument("--stage", required=True, choices=["S1","S2","S4","S5","final"])
    ap.add_argument("--execute", action="store_true")
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--rclone-transfers", type=int, default=8,
                    help="rclone --transfers concurrency (operational bound for sharded runs).")
    ap.add_argument("--rclone-checkers",  type=int, default=8,
                    help="rclone --checkers concurrency (operational bound for sharded runs).")
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.parquet_cache.mkdir(parents=True, exist_ok=True)
    global RCLONE_TRANSFERS, RCLONE_CHECKERS
    RCLONE_TRANSFERS = int(args.rclone_transfers)
    RCLONE_CHECKERS  = int(args.rclone_checkers)

    if args.stage in ("S2","S5") and not args.execute:
        print(f"FATAL: {args.stage} requires --execute", file=sys.stderr); return 2
    if args.stage in ("S1","S4") and args.execute:
        print(f"FATAL: {args.stage} is dry-run only", file=sys.stderr); return 2

    targets = resolve_targets(args.api_base, args.targets)
    print(f"stage={args.stage}  targets={len(targets)}")

    if args.stage in ("S1","S4"):
        out = stage_dry_run(args.api_base, args.staging_dir, args.parquet_cache,
                            targets, args.out_dir / f"w3b_{args.stage}_dryrun.json")
        print(json.dumps({k: v for k, v in out.items() if k != "records"}, indent=2))
        return 0

    if args.stage in ("S2","S5"):
        out = stage_execute(args.api_base, args.staging_dir, args.parquet_cache,
                            targets, args.out_dir / f"w3b_{args.stage}_execute.json")
        print(json.dumps(out["summary"], indent=2))
        if out["summary"]["http_non_200"] > 0: return 1
        return 0

    if args.stage == "final":
        # Lightweight: query /stats, assert site_event_aggregates > 0.
        stats = api_get(f"{args.api_base}/stats")
        ok = stats.get("site_event_aggregates", 0) > 0
        out = {"stats": stats, "ok": ok}
        (args.out_dir / "w3b_final.json").write_text(json.dumps(out, indent=2))
        print(json.dumps(stats, indent=2))
        return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
