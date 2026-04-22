#!/usr/bin/env python3
"""W6 stale-row cleanup — staged producer.

Removes site_features rows (and cascaded child rows) whose (target, site_name)
is NOT in the authoritative current binding_sites.json site-id set.

Strictly:
  - dry-run by default (explicit --execute required)
  - HTTP-only: uses only POST /targets/:t/purge-stale and GET endpoints
  - no direct wrangler d1 execute; no SQL outside the Worker writer surface
  - child-table cascade happens inside the Worker's atomic env.DB.batch()
  - per-target sanity cap rejects any dry-run where
        len(stale_set) / len(current_D1_set) > max_delete_fraction

Canonical authoritative source (frozen in this lane):
    r2:prism-archive/10k-runs-pct70/<target>/<target>.binding_sites.json

Staging directory is expected to contain that file pre-downloaded as
    <staging-dir>/<target>/<target>.binding_sites.json

Usage — proof-set dry-run (S1):
    python3 scripts/production/stale_row_purge.py \\
        --api-base https://prism-feature-pipeline.is-0b9.workers.dev \\
        --staging-dir /home/diddy/Desktop/Prism4D-bio/.w4_backfill_stage \\
        --targets 10dc_chainA,10dj_chainA \\
        --stage S1 --out-dir /home/diddy/Desktop/Prism4D-bio/.stale_row_purge_out

Stages:
    S1  proof-set dry-run                (requires --targets <csv>)
    S2  proof-set execute                (requires --targets <csv> and --execute)
    S4  widened dry-run                  (--targets "all")
    S5  widened execute                  (--targets "all" and --execute)

(The orchestrating validator runs S3 / S6 are separate invocations of
 scripts/production/validate_v4_contract.py; see the lane's execution
 order in the user's authorization message.)
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

HEADERS = {"User-Agent": "Mozilla/5.0 prism4d-stale-row-purge"}


def api_get(url: str, timeout: int = 60) -> Any:
    req = urllib.request.Request(url, headers=HEADERS)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


def api_post(url: str, payload: Any, timeout: int = 60) -> tuple[int, Any]:
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
            return e.code, {"error": "non-json response"}


def authoritative_site_names(staging_dir: Path, target: str) -> list[str]:
    """Extract and canonicalize the authoritative site-name list from R2.

    Canonicalization per this lane's tightening:
      - sorted ascending (lexicographic on the full 'site<N>' string;
        the numeric-suffix ordering is only meaningful for equal-width
        names, but determinism is what this contract cares about, not
        numeric order)
      - deduplicated
    """
    bs_path = staging_dir / target / f"{target}.binding_sites.json"
    if not bs_path.exists():
        raise FileNotFoundError(f"missing binding_sites.json: {bs_path}")
    bs = json.load(open(bs_path))
    sites = bs.get("sites", [])
    names = set()
    for s in sites:
        sid = s.get("id")
        if sid is None:
            continue
        names.add(f"site{sid}")
    return sorted(names)


def current_d1_site_names(api_base: str, target: str) -> list[str]:
    """Complete set of site_names currently in D1 for target.

    Completeness: the Worker's GET /site-features/:t handler issues
    `SELECT ... FROM site_features WHERE target=? ORDER BY spike_count DESC`
    with no LIMIT; D1's .all() cap is 10k rows; per-target cardinality
    is observed ≤64.  The returned .count equals len(.sites).
    """
    data = api_get(f"{api_base}/site-features/{target}?fields=ranker")
    rows = data.get("sites", [])
    count_hdr = data.get("count")
    if count_hdr is not None and count_hdr != len(rows):
        raise RuntimeError(
            f"{target}: GET /site-features count header ({count_hdr}) "
            f"!= rows length ({len(rows)}); transport inconsistency"
        )
    return [r["site_name"] for r in rows]


def resolve_targets(api_base: str, arg: str) -> list[str]:
    if arg == "all":
        tdata = api_get(f"{api_base}/targets?spike_percentile=70")
        return sorted([t["target"] for t in tdata["targets"]])
    return [t.strip() for t in arg.split(",") if t.strip()]


# ──────────────────────────────────────────────────────────────────
#  Per-target dry-run computation
# ──────────────────────────────────────────────────────────────────

def compute_plan(api_base: str, staging_dir: Path, target: str,
                 max_delete_fraction: float) -> dict:
    """Returns a plan record per target.  Never calls W6."""
    auth = authoritative_site_names(staging_dir, target)
    current = current_d1_site_names(api_base, target)
    auth_set = set(auth)
    current_set = set(current)
    stale = sorted(current_set - auth_set)
    missing_in_d1 = sorted(auth_set - current_set)  # should be 0 for
                                                     # W1-reprocessed targets
    n_cur = len(current_set)
    n_auth = len(auth_set)
    sanity_frac = (len(stale) / n_cur) if n_cur > 0 else 0.0
    below_cap = sanity_frac <= max_delete_fraction
    return {
        "target": target,
        "authoritative_count": n_auth,
        "current_count": n_cur,
        "stale_count": len(stale),
        "stale_sample": stale[:10],
        "missing_in_d1_count": len(missing_in_d1),
        "missing_in_d1_sample": missing_in_d1[:10],
        "sanity_frac": round(sanity_frac, 4),
        "below_cap": below_cap,
        "auth_list_canonical": auth,   # sorted + deduplicated
    }


# ──────────────────────────────────────────────────────────────────
#  Stages
# ──────────────────────────────────────────────────────────────────

def stage_dry_run(api_base: str, staging_dir: Path, targets: list[str],
                  max_delete_fraction: float, out_path: Path) -> dict:
    plans = []
    t0 = time.time()
    for i, t in enumerate(targets, 1):
        try:
            p = compute_plan(api_base, staging_dir, t, max_delete_fraction)
        except Exception as e:
            p = {"target": t, "error": f"{type(e).__name__}: {e}"}
        plans.append(p)
        if i % 50 == 0 or i == len(targets):
            print(f"  dry-run [{i}/{len(targets)}]  t={time.time()-t0:.1f}s", flush=True)

    ok = [p for p in plans if "error" not in p]
    summary = {
        "total_targets":                 len(targets),
        "targets_ok":                    len(ok),
        "targets_with_errors":           len(plans) - len(ok),
        "total_stale_rows":              sum(p["stale_count"] for p in ok),
        "targets_with_stale_rows":       sum(1 for p in ok if p["stale_count"] > 0),
        "targets_below_cap":             sum(1 for p in ok if p["below_cap"]),
        "targets_rejected_by_sanity":    sum(1 for p in ok if not p["below_cap"]),
        "targets_with_d1_gap":           sum(1 for p in ok if p["missing_in_d1_count"] > 0),
    }
    out = {"plans": plans, "summary": summary, "max_delete_fraction": max_delete_fraction}
    out_path.write_text(json.dumps(out, indent=2))
    return out


def stage_execute(api_base: str, dry_out: dict, out_path: Path) -> dict:
    results = []
    executed = skipped = failed = 0
    for p in dry_out["plans"]:
        if "error" in p:
            skipped += 1
            results.append({"target": p["target"], "skipped": "dry_run_error"})
            continue
        if p["stale_count"] == 0:
            skipped += 1
            results.append({"target": p["target"], "skipped": "no_stale_rows"})
            continue
        if not p["below_cap"]:
            skipped += 1
            results.append({"target": p["target"], "skipped": "over_sanity_cap",
                            "sanity_frac": p["sanity_frac"]})
            continue
        code, body = api_post(
            f"{api_base}/targets/{p['target']}/purge-stale",
            {"authoritative_site_names": p["auth_list_canonical"]},
        )
        if code != 200:
            failed += 1
            results.append({"target": p["target"], "http": code, "body": body})
            continue
        executed += 1
        # Worker response shape: {target, authoritative_count, deleted{...}, post_state{...}}
        exp_post = p["authoritative_count"]
        got_post = body.get("post_state", {}).get("site_features_count")
        post_invariant_ok = (got_post == exp_post)
        results.append({
            "target": p["target"],
            "http": code,
            "deleted": body.get("deleted", {}),
            "authoritative_count": body.get("authoritative_count"),
            "post_state": body.get("post_state", {}),
            "expected_post_site_features_count": exp_post,
            "post_invariant_ok": post_invariant_ok,
        })

    summary = {
        "executed": executed, "skipped": skipped, "failed": failed,
        "total_deleted_site_features":
            sum(r.get("deleted", {}).get("site_features", 0) for r in results),
        "total_deleted_site_lining_residues":
            sum(r.get("deleted", {}).get("site_lining_residues", 0) for r in results),
        "total_deleted_site_kcc_candidates":
            sum(r.get("deleted", {}).get("site_kcc_candidates", 0) for r in results),
        "total_deleted_site_event_aggregates":
            sum(r.get("deleted", {}).get("site_event_aggregates", 0) for r in results),
        "total_deleted_quarantined_event_aggregates":
            sum(r.get("deleted", {}).get("quarantined_event_aggregates", 0) for r in results),
        "post_invariant_failures":
            sum(1 for r in results if r.get("post_invariant_ok") is False),
    }
    out = {"results": results, "summary": summary}
    out_path.write_text(json.dumps(out, indent=2))
    return out


def stage_final_state(api_base: str, staging_dir: Path, targets: list[str],
                      out_path: Path) -> dict:
    """Post-everything verification: for every target, D1 count == |auth(T)|."""
    mismatches = []
    checked = 0
    for t in targets:
        try:
            auth = authoritative_site_names(staging_dir, t)
            d1 = current_d1_site_names(api_base, t)
            checked += 1
            if set(d1) != set(auth):
                mismatches.append({
                    "target": t,
                    "d1_count": len(d1), "auth_count": len(auth),
                    "stale_remaining": len(set(d1) - set(auth)),
                    "missing_in_d1":   len(set(auth) - set(d1)),
                })
        except Exception as e:
            mismatches.append({"target": t, "error": f"{type(e).__name__}: {e}"})
    out = {"checked": checked, "mismatches": mismatches,
           "remaining_stale_rows": sum(m.get("stale_remaining", 0) for m in mismatches)}
    out_path.write_text(json.dumps(out, indent=2))
    return out


# ──────────────────────────────────────────────────────────────────
#  Driver
# ──────────────────────────────────────────────────────────────────

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--api-base", required=True)
    ap.add_argument("--staging-dir", type=Path, required=True)
    ap.add_argument("--targets", required=True,
                    help='Comma-separated list OR the literal "all"')
    ap.add_argument("--stage", required=True, choices=["S1", "S2", "S4", "S5", "final"])
    ap.add_argument("--execute", action="store_true",
                    help="Required for S2/S5. Ignored on dry-run stages.")
    ap.add_argument("--max-delete-fraction", type=float, default=0.75)
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    if args.stage in ("S2", "S5") and not args.execute:
        print(f"FATAL: stage {args.stage} requires --execute", file=sys.stderr)
        return 2
    if args.stage in ("S1", "S4") and args.execute:
        print(f"FATAL: stage {args.stage} is dry-run only (do not pass --execute)", file=sys.stderr)
        return 2

    targets = resolve_targets(args.api_base, args.targets)
    print(f"stage={args.stage}  targets={len(targets)}  "
          f"max_delete_fraction={args.max_delete_fraction}")

    if args.stage in ("S1", "S4"):
        out_name = f"stale_row_purge_{args.stage}_dryrun.json"
        out = stage_dry_run(args.api_base, args.staging_dir, targets,
                            args.max_delete_fraction, args.out_dir / out_name)
        print(json.dumps(out["summary"], indent=2))
        return 0

    if args.stage in ("S2", "S5"):
        # Rerun dry-run first (use current live state; a prior dry-run may be stale).
        print("  (re-computing dry-run to capture current live D1 state...)")
        dry = stage_dry_run(args.api_base, args.staging_dir, targets,
                            args.max_delete_fraction,
                            args.out_dir / f"stale_row_purge_{args.stage}_predryrun.json")
        print(json.dumps(dry["summary"], indent=2))
        if dry["summary"]["targets_rejected_by_sanity"] > 0:
            print("FATAL: one or more targets over sanity cap; aborting execute", file=sys.stderr)
            return 1
        out = stage_execute(args.api_base, dry,
                            args.out_dir / f"stale_row_purge_{args.stage}_execute.json")
        print(json.dumps(out["summary"], indent=2))
        if out["summary"]["failed"] or out["summary"]["post_invariant_failures"]:
            return 1
        return 0

    if args.stage == "final":
        out = stage_final_state(args.api_base, args.staging_dir, targets,
                                args.out_dir / "stale_row_purge_final_state.json")
        print(json.dumps(out, indent=2))
        return 0 if out["remaining_stale_rows"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
