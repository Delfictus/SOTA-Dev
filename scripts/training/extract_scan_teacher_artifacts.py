#!/usr/bin/env python3
"""Extract high-value teacher artifacts from an existing PRISM scan run.

This is not a run wrapper. It consumes completed engine artifacts and emits
distillation-ready metadata that the current pipeline already computes or can
derive losslessly from the Arrow spike stream.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


SCHEMA = "prism_scan_teacher_artifacts_v1"
PHASE_NAMES = {
    0: "cold_hold",
    1: "ramp",
    2: "warm_hold",
    3: "cooling",
}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def load_json(path: Optional[Path]) -> Dict[str, Any]:
    if path is None or not path.exists():
        return {}
    with path.open() as f:
        data = json.load(f)
    return data if isinstance(data, dict) else {"sites": data}


def find_one(run_dir: Path, pattern: str) -> Optional[Path]:
    matches = sorted(run_dir.glob(pattern))
    return matches[0] if matches else None


def derive_stem(run_dir: Path) -> str:
    bs = find_one(run_dir, "*.binding_sites.json")
    if bs is not None:
        return bs.name[: -len(".binding_sites.json")]
    arrow = find_one(run_dir, "*.topology.spike_events.arrow")
    if arrow is not None:
        return arrow.name[: -len(".topology.spike_events.arrow")]
    return run_dir.name


def source_entry(path: Optional[Path]) -> Optional[Dict[str, Any]]:
    if path is None or not path.exists():
        return None
    return {
        "path": str(path),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def sites_from_binding(binding: Dict[str, Any]) -> List[Dict[str, Any]]:
    sites = binding.get("sites", [])
    return sites if isinstance(sites, list) else []


def candidate_inventory(binding: Dict[str, Any]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    all_pockets = binding.get("all_pockets")
    source = "binding_sites.all_pockets"
    if not isinstance(all_pockets, list) or not all_pockets:
        all_pockets = sites_from_binding(binding)
        source = "binding_sites.sites"

    for i, p in enumerate(all_pockets):
        if not isinstance(p, dict):
            continue
        rows.append({
            "candidate_id": p.get("id", p.get("cluster_id", i)),
            "source": source,
            "centroid": p.get("centroid"),
            "volume": p.get("volume", p.get("volume_angstrom3")),
            "spike_count": p.get("spike_count"),
            "quality_score": p.get("quality_score", p.get("rank_score")),
            "therm_class": p.get("therm_class"),
            "burial_score": p.get("burial_score"),
            "druggability": p.get("druggability"),
        })
    return rows


def therm_histogram(binding: Dict[str, Any]) -> Dict[str, int]:
    hist: Counter[str] = Counter()
    for site in sites_from_binding(binding):
        klass = site.get("therm_class")
        if klass:
            hist[str(klass)] += 1

    prism_therm = binding.get("prism_therm") or {}
    for key in ("sites", "pockets", "cryptic_sites"):
        for row in prism_therm.get(key, []) if isinstance(prism_therm, dict) else []:
            if isinstance(row, dict):
                klass = row.get("therm_class") or row.get("class")
                if klass:
                    hist[str(klass)] += 1
    return dict(sorted(hist.items()))


def anchor_candidate_field(binding: Dict[str, Any]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    interesting = ("anchor", "kcc", "boost")
    for site in sites_from_binding(binding):
        sid = site.get("id", site.get("cluster_id"))
        row = {"site_id": sid, "centroid": site.get("centroid"), "fields": {}}
        for key, value in site.items():
            lk = key.lower()
            if any(tok in lk for tok in interesting):
                row["fields"][key] = value
        if row["fields"]:
            out.append(row)
    return out


def _column(batch, name: str):
    idx = batch.schema.get_field_index(name)
    if idx < 0:
        return None
    return batch.column(idx)


def _column_py(batch, name: str) -> List[Any]:
    col = _column(batch, name)
    return col.to_pylist() if col is not None else []


def _column_np(batch, name: str):
    col = _column(batch, name)
    if col is None:
        return None
    return col.to_numpy(zero_copy_only=False)


def scan_arrow(arrow_path: Optional[Path], bucket_steps: int) -> Dict[str, Any]:
    if arrow_path is None or not arrow_path.exists():
        return {"available": False, "reason": "missing_arrow"}

    try:
        import pyarrow.ipc as ipc
    except ImportError:
        return {"available": False, "reason": "pyarrow_not_importable"}

    residue_counts: Counter[int] = Counter()
    aromatic_counts: Counter[int] = Counter()
    site_counts: Counter[int] = Counter()
    phase_counts: Counter[str] = Counter()
    mechanism_counts: Counter[str] = Counter()
    percentile_hist: Counter[int] = Counter()
    bucket_site_counts: Dict[int, Counter[int]] = defaultdict(Counter)
    bucket_spikes: Counter[int] = Counter()
    n_spikes = 0

    with arrow_path.open("rb") as f:
        reader = ipc.open_file(f)
        for i in range(reader.num_record_batches):
            batch = reader.get_batch(i)
            n_spikes += batch.num_rows

            nearby = _column_py(batch, "nearby_residues")
            for residues in nearby:
                if not residues:
                    continue
                valid = [int(r) for r in residues if int(r) >= 0]
                for rid in valid:
                    residue_counts[rid] += 1

            aromatic = _column_np(batch, "aromatic_residue_id")
            if aromatic is not None:
                for rid in aromatic:
                    rid_i = int(rid)
                    if rid_i >= 0:
                        aromatic_counts[rid_i] += 1
                        residue_counts[rid_i] += 1

            site_ids = _column_np(batch, "site_id")
            timesteps = _column_np(batch, "timestep")
            if site_ids is not None:
                for sid in site_ids:
                    sid_i = int(sid)
                    if sid_i >= 0:
                        site_counts[sid_i] += 1
            if site_ids is not None and timesteps is not None:
                for sid, ts in zip(site_ids, timesteps):
                    sid_i = int(sid)
                    if sid_i >= 0:
                        bucket = int(ts) // max(bucket_steps, 1)
                        bucket_site_counts[bucket][sid_i] += 1
                        bucket_spikes[bucket] += 1

            phases = _column_np(batch, "ccns_phase")
            if phases is not None:
                for ph in phases:
                    phase_counts[PHASE_NAMES.get(int(ph), str(int(ph)))] += 1

            mechanisms = _column_py(batch, "mechanism_tag")
            for mech in mechanisms:
                mechanism_counts[str(mech)] += 1

            percentiles = _column_np(batch, "intensity_percentile")
            if percentiles is not None:
                for pct in percentiles:
                    percentile_hist[int(pct)] += 1

    total_residue_hits = sum(residue_counts.values()) or 1
    per_residue = [
        {
            "residue_id": rid,
            "spike_hits": count,
            "rate": count / total_residue_hits,
            "aromatic_hits": aromatic_counts.get(rid, 0),
        }
        for rid, count in sorted(residue_counts.items())
    ]

    curve = []
    for threshold in range(0, 101, 5):
        count = sum(c for pct, c in percentile_hist.items() if pct >= threshold)
        curve.append({"threshold_percentile": threshold, "n_spikes_ge": count})

    persistence_points = []
    cumulative_sites: set[int] = set()
    cumulative_spikes = 0
    for bucket in sorted(bucket_site_counts):
        cumulative_sites.update(bucket_site_counts[bucket].keys())
        cumulative_spikes += bucket_spikes[bucket]
        persistence_points.append({
            "bucket": bucket,
            "step_start": bucket * bucket_steps,
            "active_site_count": len(bucket_site_counts[bucket]),
            "cumulative_site_count": len(cumulative_sites),
            "cumulative_assigned_spikes": cumulative_spikes,
        })

    slope = None
    if len(persistence_points) >= 2:
        xs = [p["bucket"] for p in persistence_points]
        ys = [p["cumulative_site_count"] for p in persistence_points]
        x_mean = sum(xs) / len(xs)
        y_mean = sum(ys) / len(ys)
        denom = sum((x - x_mean) ** 2 for x in xs)
        if denom > 0:
            slope = sum((x - x_mean) * (y - y_mean) for x, y in zip(xs, ys)) / denom

    return {
        "available": True,
        "n_spikes": n_spikes,
        "per_residue_spike_rate": per_residue,
        "site_spike_counts": dict(sorted(site_counts.items())),
        "phase_histogram": dict(sorted(phase_counts.items())),
        "mechanism_histogram": dict(sorted(mechanism_counts.items())),
        "spike_percentile_curve": curve,
        "persistence_vs_time": {
            "bucket_steps": bucket_steps,
            "slope_cumulative_sites_per_bucket": slope,
            "points": persistence_points,
        },
    }


def fallback_residue_rates(binding: Dict[str, Any]) -> List[Dict[str, Any]]:
    counts: Counter[int] = Counter()
    for site in sites_from_binding(binding):
        spike_count = int(site.get("spike_count") or 0)
        lining = site.get("lining_residues") or []
        residue_ids = [r.get("resid") for r in lining if isinstance(r, dict)]
        residue_ids = [int(r) for r in residue_ids if r is not None]
        if not residue_ids:
            continue
        share = max(spike_count, 1) / len(residue_ids)
        for rid in residue_ids:
            counts[rid] += share
    total = sum(counts.values()) or 1.0
    return [
        {
            "residue_id": rid,
            "spike_hits": count,
            "rate": count / total,
            "source": "binding_sites.lining_residues_proxy",
        }
        for rid, count in sorted(counts.items())
    ]


def build_artifacts(run_dir: Path, bucket_steps: int) -> Dict[str, Any]:
    stem = derive_stem(run_dir)
    binding_path = find_one(run_dir, "*.binding_sites.json")
    kcc_path = find_one(run_dir, "*.kcc_visualization.json")
    arrow_path = find_one(run_dir, "*.topology.spike_events.arrow")
    metadata_path = run_dir / f"{stem}.run_metadata.json"
    binding = load_json(binding_path)
    arrow = scan_arrow(arrow_path, bucket_steps)
    per_residue = (
        arrow["per_residue_spike_rate"]
        if arrow.get("available") else fallback_residue_rates(binding)
    )

    artifacts = {
        "schema": SCHEMA,
        "target": stem,
        "run_dir": str(run_dir),
        "sources": {
            "binding_sites": source_entry(binding_path),
            "kcc_visualization": source_entry(kcc_path),
            "spike_events_arrow": source_entry(arrow_path),
            "run_metadata": source_entry(metadata_path if metadata_path.exists() else None),
        },
        "scan_summary": {
            "n_sites": len(sites_from_binding(binding)),
            "n_candidates": len(candidate_inventory(binding)),
            "arrow_available": bool(arrow.get("available")),
            "arrow_reason": arrow.get("reason"),
            "n_spikes": arrow.get("n_spikes"),
        },
        "pre_gating_candidate_inventory": candidate_inventory(binding),
        "per_residue_spike_rate": per_residue,
        "therm_class_histogram": therm_histogram(binding),
        "anchor_candidate_field": anchor_candidate_field(binding),
        "phase_histogram": arrow.get("phase_histogram", {}),
        "mechanism_histogram": arrow.get("mechanism_histogram", {}),
        "spike_percentile_curve": arrow.get("spike_percentile_curve", []),
        "persistence_vs_time": arrow.get("persistence_vs_time"),
        "distillation_heads": {
            "per_residue_spike_rate": "regression",
            "therm_class_histogram": "multi_class_soft_target",
            "pre_gating_candidate_inventory": "candidate_set_boundary",
            "anchor_candidate_field": "contrastive_raw_vs_boosted_site_context",
            "persistence_vs_time": "temporal_trend_proxy",
            "spike_percentile_curve": "target_conditional_thresholding",
        },
    }
    return artifacts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, help="Completed PRISM output directory")
    parser.add_argument(
        "--out",
        default=None,
        help="Output JSON path. Default: <run-dir>/<stem>.teacher_scan_artifacts.json",
    )
    parser.add_argument("--bucket-steps", type=int, default=5000)
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    artifacts = build_artifacts(run_dir, args.bucket_steps)
    out = Path(args.out) if args.out else run_dir / f"{artifacts['target']}.teacher_scan_artifacts.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as f:
        json.dump(artifacts, f, indent=2, sort_keys=True)
    print(json.dumps({
        "output": str(out),
        "target": artifacts["target"],
        "n_candidates": artifacts["scan_summary"]["n_candidates"],
        "arrow_available": artifacts["scan_summary"]["arrow_available"],
    }, indent=2))


if __name__ == "__main__":
    main()
