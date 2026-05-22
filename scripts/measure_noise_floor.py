#!/usr/bin/env python3
"""Measure same-seed PRISM run noise for an explicit epsilon-budget decision.

This tool is intentionally conservative:
  * frame-count integrity is a hard requirement;
  * exact frame-hash agreement is reported, not used as the epsilon decision;
  * downstream invariance is measured from binding-site JSON plus per-residue
    feature parquet or, when explicitly requested, spike Arrow files.

It does not edit docs/cuda_determinism_audit_results.csv. Use the JSON report
as evidence before changing any CUDA audit row to accepted-as-is.
"""
from __future__ import annotations

import argparse
import glob
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np


RESIDUE_COLUMNS = ("residue_id", "res_id", "residue_idx", "residue_index", "rid")
SPIKE_RESIDUE_COLUMNS = ("residue_id", "res_id", "residue_idx", "nearest_residue_id")
SPIKE_INTENSITY_COLUMNS = ("intensity", "spike_intensity", "amplitude", "weight")
SITE_LIST_KEYS = ("sites", "binding_sites", "ranked_sites", "pockets", "candidate_sites")
SITE_SCORE_KEYS = (
    "rank_score",
    "score",
    "phase_manifold_score",
    "max_phase_manifold_score",
    "kcc_score",
    "confidence",
    "druggability",
)
THERM_KEYS = ("therm_class", "classification", "pocket_class", "site_classification")


class NoiseFloorError(RuntimeError):
    pass


@dataclass
class Site:
    key: str
    rank: int
    score: float
    residues: Tuple[int, ...]
    therm_class: str = ""


@dataclass
class RunRecord:
    name: str
    path: Path
    frame_counts: List[int] = field(default_factory=list)
    frame_hashes: List[str] = field(default_factory=list)
    all_hashes_match: Optional[bool] = None
    sites: List[Site] = field(default_factory=list)
    residue_features: Dict[str, Dict[int, float]] = field(default_factory=dict)
    artifacts: Dict[str, str] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=2, sort_keys=True)
        fh.write("\n")


def expand_paths(items: Sequence[str]) -> List[Path]:
    out: List[Path] = []
    for item in items:
        matches = [Path(p) for p in glob.glob(item)]
        out.extend(matches if matches else [Path(item)])
    seen = set()
    unique = []
    for p in out:
        rp = p.resolve()
        if rp not in seen:
            unique.append(rp)
            seen.add(rp)
    return sorted(unique)


def first_existing(candidates: Iterable[Path]) -> Optional[Path]:
    for p in candidates:
        if p and p.exists():
            return p
    return None


def find_one(root: Path, patterns: Sequence[str]) -> Optional[Path]:
    for pattern in patterns:
        hits = sorted(root.glob(pattern))
        if hits:
            return hits[0]
    return None


def resolve_relative(base: Path, value: Optional[str]) -> Optional[Path]:
    if not value:
        return None
    p = Path(value)
    return p if p.is_absolute() else base / p


def numeric(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        v = float(value)
        return v if math.isfinite(v) else default
    except Exception:
        return default


def extract_frame_audit(record: Dict[str, Any]) -> Tuple[List[int], List[str], Optional[bool]]:
    audit = record.get("frame_audit") or record.get("replica", {}).get("frame_audit") or {}
    counts = (
        audit.get("disk_count_per_stream")
        or audit.get("writer_count_per_stream")
        or audit.get("producer_count_per_stream")
        or []
    )
    hashes = (
        audit.get("disk_hash_per_stream")
        or audit.get("writer_hash_per_stream")
        or audit.get("producer_hash_per_stream")
        or []
    )
    return [int(x) for x in counts], [str(x) for x in hashes], audit.get("all_hashes_match")


def normalize_residue_list(value: Any) -> Tuple[int, ...]:
    if value is None:
        return ()
    if isinstance(value, dict):
        value = value.get("residue_ids") or value.get("residues") or value.get("ids") or []
    out: List[int] = []
    if isinstance(value, str):
        value = value.replace(",", " ").split()
    if isinstance(value, Sequence):
        for item in value:
            if isinstance(item, dict):
                item = item.get("residue_id", item.get("res_id", item.get("id")))
            try:
                out.append(int(item))
            except Exception:
                continue
    return tuple(sorted(set(out)))


def find_site_list(obj: Any) -> List[Dict[str, Any]]:
    if isinstance(obj, list):
        return [x for x in obj if isinstance(x, dict)]
    if not isinstance(obj, dict):
        return []
    for key in SITE_LIST_KEYS:
        val = obj.get(key)
        if isinstance(val, list):
            return [x for x in val if isinstance(x, dict)]
    for val in obj.values():
        nested = find_site_list(val)
        if nested:
            return nested
    return []


def parse_sites(path: Optional[Path], warnings: List[str]) -> List[Site]:
    if not path or not path.exists():
        warnings.append("binding_sites_json_missing")
        return []
    try:
        raw_sites = find_site_list(load_json(path))
    except Exception as exc:
        warnings.append(f"binding_sites_json_unreadable:{exc}")
        return []

    sites: List[Site] = []
    for idx, site in enumerate(raw_sites):
        residues = normalize_residue_list(
            site.get("residue_ids")
            or site.get("residues")
            or site.get("lining_residues")
            or site.get("hot_residues")
        )
        site_id = site.get("site_id", site.get("id", site.get("rank", idx)))
        rank = int(site.get("rank", idx + 1))
        score = 0.0
        for key in SITE_SCORE_KEYS:
            if key in site:
                score = numeric(site[key])
                break
        therm = ""
        for key in THERM_KEYS:
            if key in site and site[key] is not None:
                therm = str(site[key])
                break
        stable_key = ",".join(str(x) for x in residues) if residues else f"id:{site_id}"
        sites.append(Site(stable_key, rank, score, residues, therm))

    return sorted(sites, key=lambda s: (s.rank, -s.score, s.key))


def import_polars():
    try:
        import polars as pl  # type: ignore
    except Exception as exc:
        raise NoiseFloorError("polars is required for parquet/Arrow feature analysis") from exc
    return pl


def select_residue_column(columns: Sequence[str]) -> Optional[str]:
    for c in RESIDUE_COLUMNS:
        if c in columns:
            return c
    return None


def read_feature_parquet(path: Path, warnings: List[str]) -> Dict[str, Dict[int, float]]:
    pl = import_polars()
    try:
        df = pl.read_parquet(path)
    except Exception as exc:
        warnings.append(f"feature_parquet_unreadable:{path}:{exc}")
        return {}
    if df.height == 0:
        warnings.append(f"feature_parquet_empty:{path}")
        return {}

    residue_col = select_residue_column(df.columns)
    residue_ids = (
        df[residue_col].to_numpy().astype(np.int64)
        if residue_col
        else np.arange(df.height, dtype=np.int64)
    )

    out: Dict[str, Dict[int, float]] = {}
    for col, dtype in zip(df.columns, df.dtypes):
        if col == residue_col:
            continue
        if not dtype.is_numeric():
            continue
        vals = df[col].to_numpy()
        metric: Dict[int, float] = {}
        for rid, val in zip(residue_ids, vals):
            v = numeric(val, float("nan"))
            if math.isfinite(v):
                metric[int(rid)] = v
        if metric:
            out[col] = metric
    return out


def read_spike_arrow(path: Path, warnings: List[str]) -> Dict[str, Dict[int, float]]:
    pl = import_polars()
    try:
        try:
            df = pl.scan_ipc(path).collect()
        except Exception:
            df = pl.read_ipc(path)
    except Exception as exc:
        warnings.append(f"spike_arrow_unreadable:{path}:{exc}")
        return {}
    residue_col = next((c for c in SPIKE_RESIDUE_COLUMNS if c in df.columns), None)
    if residue_col is None:
        warnings.append(f"spike_arrow_no_residue_column:{path}")
        return {}
    intensity_col = next((c for c in SPIKE_INTENSITY_COLUMNS if c in df.columns), None)
    exprs = [pl.len().alias("spike_count")]
    if intensity_col:
        exprs.extend(
            [
                pl.col(intensity_col).mean().alias("spike_intensity_mean"),
                pl.col(intensity_col).sum().alias("spike_intensity_sum"),
            ]
        )
    grouped = df.group_by(residue_col).agg(exprs).sort(residue_col)
    rid = grouped[residue_col].to_numpy().astype(np.int64)
    out: Dict[str, Dict[int, float]] = {}
    for col in grouped.columns:
        if col == residue_col:
            continue
        vals = grouped[col].to_numpy()
        out[col] = {int(r): numeric(v) for r, v in zip(rid, vals)}
    return out


def merge_feature_maps(dst: Dict[str, Dict[int, float]], src: Dict[str, Dict[int, float]]) -> None:
    for name, metric in src.items():
        dst.setdefault(name, {}).update(metric)


def load_run(path: Path, read_spike_arrow: bool) -> RunRecord:
    root = path if path.is_dir() else path.parent
    rec = RunRecord(name=path.stem if path.is_file() else path.name, path=path)

    manifest_or_record = path if path.is_file() else first_existing(
        [
            root / "ensemble_manifest.json",
            *sorted(root.glob("ensemble_replica_*.json")),
        ]
    )
    manifest_obj: Dict[str, Any] = {}
    replica_obj: Dict[str, Any] = {}
    if manifest_or_record:
        try:
            manifest_obj = load_json(manifest_or_record)
            replicas = manifest_obj.get("replicas") or []
            if replicas:
                replica_obj = replicas[0]
            elif "replica" in manifest_obj:
                replica_obj = manifest_obj["replica"]
            rec.artifacts["manifest_or_record"] = str(manifest_or_record)
        except Exception as exc:
            rec.warnings.append(f"manifest_or_record_unreadable:{exc}")

    rec.frame_counts, rec.frame_hashes, rec.all_hashes_match = extract_frame_audit(replica_obj or manifest_obj)

    outputs = replica_obj.get("outputs", {}) if isinstance(replica_obj, dict) else {}
    binding_path = resolve_relative(root, outputs.get("binding_sites_json_relative"))
    if binding_path is None or not binding_path.exists():
        binding_path = find_one(root, ["*.binding_sites.json", "**/*.binding_sites.json"])
    if binding_path:
        rec.artifacts["binding_sites_json"] = str(binding_path)
    rec.sites = parse_sites(binding_path, rec.warnings)

    feature_path = find_one(
        root,
        [
            "*_v5.parquet",
            "**/*_v5.parquet",
            "*features*.parquet",
            "**/*features*.parquet",
        ],
    )
    if feature_path:
        rec.artifacts["feature_parquet"] = str(feature_path)
        merge_feature_maps(rec.residue_features, read_feature_parquet(feature_path, rec.warnings))

    if read_spike_arrow:
        arrow_path = resolve_relative(root, outputs.get("trajectory_arrow_relative"))
        if arrow_path is None or not arrow_path.exists():
            arrow_path = find_one(root, ["*.spike_events.arrow", "**/*.spike_events.arrow"])
        if arrow_path:
            rec.artifacts["spike_arrow"] = str(arrow_path)
            merge_feature_maps(rec.residue_features, read_spike_arrow(arrow_path, rec.warnings))
        else:
            rec.warnings.append("spike_arrow_missing")

    return rec


def safe_quantile(values: Sequence[float], q: float) -> float:
    vals = np.asarray([v for v in values if math.isfinite(v)], dtype=np.float64)
    if vals.size == 0:
        return float("nan")
    return float(np.quantile(vals, q))


def jaccard(a: Iterable[str], b: Iterable[str]) -> float:
    aa, bb = set(a), set(b)
    if not aa and not bb:
        return 1.0
    if not aa or not bb:
        return 0.0
    return len(aa & bb) / len(aa | bb)


def summarize_frames(runs: List[RunRecord]) -> Dict[str, Any]:
    baseline_counts = runs[0].frame_counts
    baseline_hashes = runs[0].frame_hashes
    count_matches = []
    hash_matches = []
    all_hash_flags = []
    for run in runs:
        count_matches.append(bool(run.frame_counts and run.frame_counts == baseline_counts))
        hash_matches.append(bool(run.frame_hashes and run.frame_hashes == baseline_hashes))
        if run.all_hashes_match is not None:
            all_hash_flags.append(bool(run.all_hashes_match))
    return {
        "streams": len(baseline_counts),
        "baseline_total_frames": int(sum(baseline_counts)) if baseline_counts else 0,
        "count_match_fraction_vs_baseline": float(np.mean(count_matches)) if count_matches else 0.0,
        "hash_exact_match_fraction_vs_baseline": float(np.mean(hash_matches)) if hash_matches else 0.0,
        "all_hashes_match_fraction_internal": float(np.mean(all_hash_flags)) if all_hash_flags else None,
        "count_mismatched_runs": [r.name for r, ok in zip(runs, count_matches) if not ok],
        "hash_mismatched_runs": [r.name for r, ok in zip(runs, hash_matches) if not ok],
    }


def summarize_sites(runs: List[RunRecord], top_k: int) -> Dict[str, Any]:
    baseline = runs[0].sites[:top_k]
    baseline_keys = [s.key for s in baseline]
    baseline_top1 = baseline_keys[0] if baseline_keys else ""

    top1_matches = []
    topk_jaccards = []
    therm_matches = []
    for run in runs:
        top = run.sites[:top_k]
        keys = [s.key for s in top]
        top1_matches.append(bool(keys and keys[0] == baseline_top1))
        topk_jaccards.append(jaccard(baseline_keys, keys))

        base_therm = {s.key: s.therm_class for s in baseline if s.therm_class}
        therm = {s.key: s.therm_class for s in top if s.therm_class}
        shared = [k for k in base_therm if k in therm]
        if shared:
            therm_matches.append(float(np.mean([base_therm[k] == therm[k] for k in shared])))

    return {
        "top_k": top_k,
        "baseline_top_keys": baseline_keys,
        "top1_agreement_fraction": float(np.mean(top1_matches)) if top1_matches else 0.0,
        "topk_jaccard_mean": float(np.mean(topk_jaccards)) if topk_jaccards else 0.0,
        "topk_jaccard_min": float(np.min(topk_jaccards)) if topk_jaccards else 0.0,
        "therm_class_agreement_fraction": float(np.mean(therm_matches)) if therm_matches else None,
        "runs_without_sites": [r.name for r in runs if not r.sites],
    }


def summarize_features(runs: List[RunRecord], spike_count_threshold: float) -> Dict[str, Any]:
    metric_names = sorted(set().union(*(r.residue_features.keys() for r in runs)))
    per_metric: Dict[str, Any] = {}
    all_rel_drifts = []
    spike_rel_drifts = []

    for metric in metric_names:
        residue_ids = sorted(set().union(*(r.residue_features.get(metric, {}).keys() for r in runs)))
        rel_drifts = []
        abs_drifts = []
        for rid in residue_ids:
            vals = [r.residue_features.get(metric, {}).get(rid, float("nan")) for r in runs]
            if not math.isfinite(vals[0]):
                continue
            finite = [v for v in vals if math.isfinite(v)]
            if len(finite) < 2:
                continue
            baseline = vals[0]
            max_abs = max(abs(v - baseline) for v in finite)
            denom = max(abs(baseline), 1.0 if "count" in metric else 1.0e-6)
            rel = max_abs / denom
            abs_drifts.append(max_abs)
            rel_drifts.append(rel)
        all_rel_drifts.extend(rel_drifts)
        if "spike_count" in metric or metric == "spike_count":
            spike_rel_drifts.extend(rel_drifts)
        per_metric[metric] = {
            "n_residues_compared": len(rel_drifts),
            "abs_drift_max": safe_quantile(abs_drifts, 1.0),
            "rel_drift_p95": safe_quantile(rel_drifts, 0.95),
            "rel_drift_p99": safe_quantile(rel_drifts, 0.99),
            "rel_drift_max": safe_quantile(rel_drifts, 1.0),
        }

    return {
        "n_metrics": len(metric_names),
        "metrics": per_metric,
        "all_feature_rel_drift_p95": safe_quantile(all_rel_drifts, 0.95),
        "all_feature_rel_drift_p99": safe_quantile(all_rel_drifts, 0.99),
        "all_feature_rel_drift_max": safe_quantile(all_rel_drifts, 1.0),
        "spike_count_rel_drift_max": safe_quantile(spike_rel_drifts, 1.0),
        "spike_count_threshold": spike_count_threshold,
        "has_spike_count_signal": bool(spike_rel_drifts),
    }


def make_decision(
    n_runs: int,
    frames: Dict[str, Any],
    sites: Dict[str, Any],
    features: Dict[str, Any],
    args: argparse.Namespace,
) -> Dict[str, Any]:
    checks = {
        "min_trials": n_runs >= args.min_trials,
        "frame_counts_match": frames["count_match_fraction_vs_baseline"] == 1.0,
        "site_signal_present": not sites["runs_without_sites"],
        "feature_signal_present": features["n_metrics"] > 0,
        "top1_site_agreement": sites["top1_agreement_fraction"] >= args.min_top1_agreement,
        "topk_jaccard": sites["topk_jaccard_min"] >= args.min_topk_jaccard,
        "therm_class_agreement": (
            sites["therm_class_agreement_fraction"] is None
            or sites["therm_class_agreement_fraction"] >= args.min_therm_agreement
        ),
        "feature_rel_drift": (
            math.isnan(features["all_feature_rel_drift_p99"])
            or features["all_feature_rel_drift_p99"] <= args.max_feature_rel_drift_p99
        ),
        "spike_count_rel_drift": (
            not features["has_spike_count_signal"]
            or features["spike_count_rel_drift_max"] <= args.max_spike_count_rel_drift
        ),
    }
    return {
        "accepted_for_epsilon_budget": all(checks.values()),
        "checks": checks,
        "note": (
            "This decision is downstream-invariance based. Exact frame hash mismatches "
            "are expected when accepting a floating-point atomic epsilon budget; frame "
            "count mismatches remain fatal."
        ),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("runs", nargs="+", help="Run directories or globs, e.g. /tmp/det_*")
    ap.add_argument("--output-json", type=Path, default=Path("noise_floor_report.json"))
    ap.add_argument("--read-spike-arrow", action="store_true", help="Read raw *.spike_events.arrow files; can be expensive")
    ap.add_argument("--top-k", type=int, default=5)
    ap.add_argument("--min-trials", type=int, default=100)
    ap.add_argument("--max-spike-count-rel-drift", type=float, default=0.001)
    ap.add_argument("--max-feature-rel-drift-p99", type=float, default=0.001)
    ap.add_argument("--min-top1-agreement", type=float, default=1.0)
    ap.add_argument("--min-topk-jaccard", type=float, default=1.0)
    ap.add_argument("--min-therm-agreement", type=float, default=1.0)
    args = ap.parse_args()

    paths = expand_paths(args.runs)
    if not paths:
        raise NoiseFloorError("no run paths matched")

    runs = [load_run(p, args.read_spike_arrow) for p in paths]
    frames = summarize_frames(runs)
    sites = summarize_sites(runs, args.top_k)
    features = summarize_features(runs, args.max_spike_count_rel_drift)
    decision = make_decision(len(runs), frames, sites, features, args)

    report = {
        "schema_version": "1.0.0",
        "n_runs": len(runs),
        "runs": [
            {
                "name": r.name,
                "path": str(r.path),
                "artifacts": r.artifacts,
                "warnings": r.warnings,
                "n_sites": len(r.sites),
                "n_feature_metrics": len(r.residue_features),
            }
            for r in runs
        ],
        "frames": frames,
        "sites": sites,
        "features": features,
        "decision": decision,
    }
    write_json(args.output_json, report)

    status = "ACCEPT" if decision["accepted_for_epsilon_budget"] else "REJECT"
    print(f"noise-floor decision: {status}")
    print(f"runs: {len(runs)}")
    print(f"frame count match: {frames['count_match_fraction_vs_baseline']:.3f}")
    print(f"top{args.top_k} jaccard min: {sites['topk_jaccard_min']:.3f}")
    print(f"feature rel drift p99: {features['all_feature_rel_drift_p99']}")
    print(f"report: {args.output_json}")
    return 0 if decision["accepted_for_epsilon_budget"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
