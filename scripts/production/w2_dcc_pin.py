#!/usr/bin/env python3
"""W2 /dcc producer — canonical DCC emission per the v4 hardening contract.

Non-invention invariants (§3 of the Detection/Feature-Service
Advancement Constitution):
  - no fake DCC  - no fake centroid  - no fake grade
  - no fake valid_for_dcc_validation flag
  - no fake holo metadata
  - pass-through only, or producer-factual labels

Missing-centroid rule (case-matrix accepted in the W2 correction pass):
  case (a) artifact valid=TRUE AND ligand_centroid is a finite 3-vector
           → real per-site DCC + real target corrected_dcc
  case (b) artifact valid=TRUE BUT ligand_centroid unrecoverable
           → producer OVERRIDE valid=0; all numeric DCC NULL;
             skip_reason = "ligand_centroid_unrecoverable"
  case (c) artifact valid=FALSE (any reason)
           → pass-through valid=0; all numeric DCC NULL;
             skip_reason = artifact pass-through
  precondition fail: ground_truth.json missing OR unparseable OR
             lacking valid_for_dcc_validation key
           → hard-skip; no POST; counted as skipped_ground_truth_missing

Boundary contract (§4 mandate): EXCELLENT ≤5.0 < GOOD ≤8.0 < MARGINAL ≤10.0 < POOR

Value domains pinned:
  dcc_metric_source = "pct70_centroid"       (only this lane value)
  dcc_metric_used   = "centroid"
  dcc_grade         ∈ {EXCELLENT, GOOD, MARGINAL, POOR} ∪ {None}
  spike_dcc / spike_site / n_parquet_sites = NULL this lane
"""
from __future__ import annotations
import argparse, json, math, sys, urllib.error, urllib.request
from pathlib import Path

HEADERS = {"User-Agent": "Mozilla/5.0 prism4d-w2-dcc-pin"}


def api_get(url: str, timeout: int = 60):
    req = urllib.request.Request(url, headers=HEADERS)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


def api_post(url: str, payload, timeout: int = 60):
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


def grade_dcc(d):
    """mandated <=-upper-inclusive form per §4 of W2 pre-coding block"""
    if d is None:
        return None
    if d <= 5.0:  return "EXCELLENT"
    if d <= 8.0:  return "GOOD"
    if d <= 10.0: return "MARGINAL"
    return "POOR"


def valid_3vec(v) -> bool:
    if not isinstance(v, (list, tuple)) or len(v) != 3:
        return False
    for c in v:
        if not isinstance(c, (int, float)) or not math.isfinite(c):
            return False
    return True


def load_artifacts(staging_dir: Path, target: str):
    """Return (bs, gt). Raises FileNotFoundError / json.JSONDecodeError on
    precondition failure (the caller interprets those as hard-skip)."""
    bs_path = staging_dir / target / f"{target}.binding_sites.json"
    gt_path = staging_dir / target / f"{target}_ground_truth.json"
    if not bs_path.exists() or not gt_path.exists():
        raise FileNotFoundError(f"missing: {bs_path} or {gt_path}")
    bs = json.load(open(bs_path))
    gt = json.load(open(gt_path))
    return bs, gt


def build_payload(bs: dict, gt: dict):
    """Return (payload_dict, case_label).  payload_dict=None ⇒ hard-skip."""
    if "valid_for_dcc_validation" not in gt:
        return None, "artifact_no_valid_flag"

    artifact_valid = bool(gt.get("valid_for_dcc_validation"))
    ligand_centroid = gt.get("ligand_centroid")
    centroid_ok = valid_3vec(ligand_centroid)

    sites_bs = bs.get("sites", [])
    site_names = [f"site{s['id']}" for s in sites_bs if "id" in s]

    def null_site_entries():
        return [{
            "site_name": n,
            "min_dist_to_ligand": None,
            "graded_score": None,
            "dcc_metric_source": None,
        } for n in site_names]

    def pass_through_cd():
        return {
            "centroid_dcc": None,
            "spike_dcc": None,
            "spike_site": None,
            "n_parquet_sites": None,
            "dcc_grade": None,
            "ligand_centroid_x": None,
            "ligand_centroid_y": None,
            "ligand_centroid_z": None,
            "holo_source": gt.get("holo_source"),
            "is_pandda_fragment": gt.get("is_pandda_fragment"),
            "is_templated_complex": gt.get("is_templated_complex"),
            "nucleic_chains": gt.get("nucleic_chains"),
            "skip_reason": gt.get("skip_reason"),
            "valid_for_dcc_validation": False,
            "dcc_metric_used": "centroid",
        }

    # case (c): artifact FALSE
    if not artifact_valid:
        cd = pass_through_cd()
        if centroid_ok:
            cd["ligand_centroid_x"] = float(ligand_centroid[0])
            cd["ligand_centroid_y"] = float(ligand_centroid[1])
            cd["ligand_centroid_z"] = float(ligand_centroid[2])
        return {"sites": null_site_entries(), "corrected_dcc": cd}, "case_c"

    # case (b): artifact TRUE but centroid unrecoverable → producer override
    if not centroid_ok:
        cd = pass_through_cd()
        cd["skip_reason"] = "ligand_centroid_unrecoverable"   # producer-factual label
        return {"sites": null_site_entries(), "corrected_dcc": cd}, "case_b_no_centroid"

    # case (a): real DCC
    cx, cy, cz = float(ligand_centroid[0]), float(ligand_centroid[1]), float(ligand_centroid[2])
    site_entries = []
    valid_site_dccs = []
    for s in sites_bs:
        sid = s.get("id")
        if sid is None:
            continue
        name = f"site{sid}"
        sc = s.get("centroid")
        if not valid_3vec(sc):
            site_entries.append({
                "site_name": name,
                "min_dist_to_ligand": None,
                "graded_score": None,
                "dcc_metric_source": None,
            })
            continue
        md = math.sqrt((sc[0]-cx)**2 + (sc[1]-cy)**2 + (sc[2]-cz)**2)
        site_entries.append({
            "site_name": name,
            "min_dist_to_ligand": md,
            "graded_score": 1.0 / (1.0 + md),
            "dcc_metric_source": "pct70_centroid",
        })
        valid_site_dccs.append((name, md))

    if not valid_site_dccs:
        # case (b) variant — every site centroid is malformed; can't compute DCC
        cd = pass_through_cd()
        cd["skip_reason"] = "no_site_centroid_valid"
        return {"sites": site_entries, "corrected_dcc": cd}, "case_b_no_site_centroids"

    best_name, best_md = min(valid_site_dccs, key=lambda x: x[1])
    cd = {
        "centroid_dcc": best_md,
        "spike_dcc": None,
        "spike_site": None,         # §2 decision: NULL this lane
        "n_parquet_sites": None,    # §3 decision: NULL this lane
        "dcc_grade": grade_dcc(best_md),
        "ligand_centroid_x": cx,
        "ligand_centroid_y": cy,
        "ligand_centroid_z": cz,
        "holo_source": gt.get("holo_source"),
        "is_pandda_fragment": gt.get("is_pandda_fragment"),
        "is_templated_complex": gt.get("is_templated_complex"),
        "nucleic_chains": gt.get("nucleic_chains"),
        "skip_reason": gt.get("skip_reason"),
        "valid_for_dcc_validation": True,
        "dcc_metric_used": "centroid",
    }
    return {"sites": site_entries, "corrected_dcc": cd}, "case_a"


# ──────────────────────────────────────────────────────────────
#  Staged execution
# ──────────────────────────────────────────────────────────────

def resolve_targets(api_base: str, arg: str) -> list[str]:
    if arg == "all":
        d = api_get(f"{api_base}/targets?spike_percentile=70")
        return sorted([t["target"] for t in d["targets"]])
    return [t.strip() for t in arg.split(",") if t.strip()]


def build_plan(staging_dir: Path, target: str) -> dict:
    try:
        bs, gt = load_artifacts(staging_dir, target)
    except FileNotFoundError:
        return {"target": target, "case": "skipped_ground_truth_missing"}
    except json.JSONDecodeError as e:
        return {"target": target, "case": "skipped_ground_truth_unparseable",
                "error": str(e)}
    payload, case = build_payload(bs, gt)
    row = {"target": target, "case": case}
    if payload is None:
        return row
    cd = payload["corrected_dcc"]
    row.update({
        "n_sites_total":            len(payload["sites"]),
        "n_sites_with_dcc":         sum(1 for s in payload["sites"] if s["min_dist_to_ligand"] is not None),
        "centroid_dcc":             cd.get("centroid_dcc"),
        "dcc_grade":                cd.get("dcc_grade"),
        "valid_for_dcc_validation": cd.get("valid_for_dcc_validation"),
        "skip_reason":              cd.get("skip_reason"),
    })
    return row


def summarise_plans(plans: list[dict]) -> dict:
    s = {"total": len(plans), "case_a": 0, "case_b": 0, "case_c": 0,
         "skipped_ground_truth_missing": 0,
         "skipped_ground_truth_unparseable": 0,
         "artifact_no_valid_flag": 0}
    for p in plans:
        c = p.get("case", "")
        if c == "case_a":                                 s["case_a"] += 1
        elif c.startswith("case_b"):                       s["case_b"] += 1
        elif c == "case_c":                                s["case_c"] += 1
        elif c == "skipped_ground_truth_missing":          s["skipped_ground_truth_missing"] += 1
        elif c == "skipped_ground_truth_unparseable":      s["skipped_ground_truth_unparseable"] += 1
        elif c == "artifact_no_valid_flag":                s["artifact_no_valid_flag"] += 1
    return s


def stage_dry_run(api_base: str, staging_dir: Path, targets: list[str], out_path: Path) -> dict:
    plans = []
    for i, t in enumerate(targets, 1):
        plans.append(build_plan(staging_dir, t))
        if i % 50 == 0 or i == len(targets):
            print(f"  dry-run [{i}/{len(targets)}]", flush=True)
    out = {"plans": plans, "summary": summarise_plans(plans)}
    out_path.write_text(json.dumps(out, indent=2))
    return out


def stage_execute(api_base: str, staging_dir: Path, targets: list[str], out_path: Path) -> dict:
    results = []
    for i, t in enumerate(targets, 1):
        plan = build_plan(staging_dir, t)
        case = plan.get("case", "")
        if case.startswith("skipped_") or case == "artifact_no_valid_flag":
            results.append({"target": t, "case": case, "skipped": True})
            continue
        try:
            bs, gt = load_artifacts(staging_dir, t)
            payload, _ = build_payload(bs, gt)
        except Exception as e:
            results.append({"target": t, "case": case, "error": f"{type(e).__name__}: {e}"})
            continue
        code, body = api_post(f"{api_base}/site-features/{t}/dcc", payload)
        results.append({
            "target": t, "case": case, "http": code,
            "updated_sites": (body or {}).get("updated_sites"),
        })
        if i % 50 == 0 or i == len(targets):
            print(f"  execute [{i}/{len(targets)}]", flush=True)
    summary = {
        "total": len(results),
        "http_200":      sum(1 for r in results if r.get("http") == 200),
        "http_non_200":  sum(1 for r in results if r.get("http") not in (None, 200)),
        "skipped":       sum(1 for r in results if r.get("skipped")),
        "errored":       sum(1 for r in results if "error" in r),
        "case_a_posted": sum(1 for r in results if r.get("case") == "case_a" and r.get("http") == 200),
        "case_b_posted": sum(1 for r in results if r.get("case", "").startswith("case_b") and r.get("http") == 200),
        "case_c_posted": sum(1 for r in results if r.get("case") == "case_c" and r.get("http") == 200),
    }
    out = {"results": results, "summary": summary}
    out_path.write_text(json.dumps(out, indent=2))
    return out


def stage_final(api_base: str, staging_dir: Path, targets: list[str], out_path: Path) -> dict:
    """For every target: expected disposition per artifact vs. D1 reality."""
    summary = {"checked": 0, "case_a_ok": 0, "case_b_ok": 0, "case_c_ok": 0,
               "skipped_artifact_ok": 0, "issues": 0}
    issues = []
    for t in targets:
        summary["checked"] += 1
        plan = build_plan(staging_dir, t)
        case = plan.get("case", "")
        try:
            dcc = api_get(f"{api_base}/dcc/{t}")
            dcc_exists = True
        except urllib.error.HTTPError as e:
            dcc = None
            dcc_exists = (e.code != 404)
            if e.code not in (200, 404):
                issues.append({"target": t, "issue": f"GET_dcc_http={e.code}"})

        if case.startswith("skipped_") or case == "artifact_no_valid_flag":
            # no W2 POST expected; a legacy row may still be present — flag but not fail
            if dcc_exists:
                issues.append({"target": t, "issue": "legacy_or_residual_corrected_dcc_row",
                               "case": case})
            else:
                summary["skipped_artifact_ok"] += 1
            continue

        if not dcc_exists:
            issues.append({"target": t, "issue": "corrected_dcc_missing_after_w2", "case": case})
            continue

        if case == "case_a":
            ok = (dcc.get("centroid_dcc") is not None
                  and dcc.get("dcc_grade") in ("EXCELLENT","GOOD","MARGINAL","POOR")
                  and dcc.get("valid_for_dcc_validation") == 1
                  and dcc.get("dcc_metric_used") == "centroid"
                  and dcc.get("ligand_centroid_x") is not None)
            if ok: summary["case_a_ok"] += 1
            else:  issues.append({"target": t, "issue": "case_a_fields_incomplete",
                                  "dcc_seen": {k: dcc.get(k) for k in
                                               ("centroid_dcc","dcc_grade","valid_for_dcc_validation",
                                                "dcc_metric_used","ligand_centroid_x")}})
        elif case.startswith("case_b"):
            ok = (dcc.get("centroid_dcc") is None
                  and dcc.get("valid_for_dcc_validation") == 0
                  and dcc.get("skip_reason") is not None)
            if ok: summary["case_b_ok"] += 1
            else:  issues.append({"target": t, "issue": "case_b_fields_incorrect",
                                  "dcc_seen": {k: dcc.get(k) for k in
                                               ("centroid_dcc","valid_for_dcc_validation","skip_reason")}})
        elif case == "case_c":
            ok = (dcc.get("centroid_dcc") is None
                  and dcc.get("valid_for_dcc_validation") == 0)
            if ok: summary["case_c_ok"] += 1
            else:  issues.append({"target": t, "issue": "case_c_fields_incorrect"})

    summary["issues"] = len(issues)
    out = {"summary": summary, "issues": issues}
    out_path.write_text(json.dumps(out, indent=2))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--api-base", required=True)
    ap.add_argument("--staging-dir", type=Path, required=True)
    ap.add_argument("--targets", required=True)
    ap.add_argument("--stage", required=True, choices=["S1","S2","S4","S5","final"])
    ap.add_argument("--execute", action="store_true")
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    if args.stage in ("S2","S5") and not args.execute:
        print(f"FATAL: {args.stage} requires --execute", file=sys.stderr); return 2
    if args.stage in ("S1","S4") and args.execute:
        print(f"FATAL: {args.stage} is dry-run only (no --execute)", file=sys.stderr); return 2

    targets = resolve_targets(args.api_base, args.targets)
    print(f"stage={args.stage}  targets={len(targets)}")

    if args.stage in ("S1","S4"):
        out = stage_dry_run(args.api_base, args.staging_dir, targets,
                            args.out_dir / f"w2_{args.stage}_dryrun.json")
        print(json.dumps(out["summary"], indent=2))
        return 0

    if args.stage in ("S2","S5"):
        out = stage_execute(args.api_base, args.staging_dir, targets,
                            args.out_dir / f"w2_{args.stage}_execute.json")
        print(json.dumps(out["summary"], indent=2))
        return 0 if out["summary"]["http_non_200"] == 0 else 1

    if args.stage == "final":
        out = stage_final(args.api_base, args.staging_dir, targets,
                          args.out_dir / "w2_final.json")
        print(json.dumps(out["summary"], indent=2))
        return 0 if out["summary"]["issues"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
