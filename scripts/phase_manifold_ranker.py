#!/usr/bin/env python3
import argparse
import json
import math
import hashlib
from pathlib import Path
from collections import defaultdict, Counter

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.ipc as ipc
import pyarrow.compute as pc


CANONICAL_PHASES = {
    0: "cold_hold",
    1: "ramp",
    2: "warm_hold",
}

# Canonical CentroidManifold slots from the PRISM SiteManifest contract.
CENTROID_SLOTS = [
    "geometric",
    "lining",
    "driver",
    "hot_phase",
    "cold_phase",
    "burst_motion",
    "validation_structural",
    "ligand_adjacent_subcluster",
]

# Raw Arrow columns used. This intentionally excludes legacy ranker-only fields.
ARROW_COLS = [
    "spike_id",
    "stream_id",
    "group_id",
    "chunk_idx",
    "voxel_idx",
    "timestep",
    "frame_index",
    "x", "y", "z",
    "intensity",
    "spike_source",
    "aromatic_type",
    "aromatic_residue_id",
    "phase_bits",
    "n_residues",
    "nearby_residues",
    "n_nearby_excited",
    "vibrational_energy",
    "water_density",
    "wd_change",
    "wavelength_nm",
    "ccns_phase",
    "site_id",
    "nearest_site_id",
    "nearest_site_dist",
    "background_class",
    "burial_score",
    "intensity_percentile",
]


def finite_float(x, default=0.0):
    try:
        v = float(x)
        return v if math.isfinite(v) else default
    except Exception:
        return default


def safe_int(x, default=None):
    try:
        if x is None:
            return default
        return int(x)
    except Exception:
        return default


def sha16(path: Path):
    h = hashlib.sha256()
    try:
        with path.open("rb") as f:
            h.update(f.read(65536))
            if path.stat().st_size > 65536:
                f.seek(max(0, path.stat().st_size - 65536))
                h.update(f.read(65536))
        return h.hexdigest()[:16]
    except Exception:
        return None


def weighted_centroid(df, xyz=("x", "y", "z"), weight="intensity"):
    if df is None or len(df) == 0:
        return None
    w = df[weight].to_numpy(dtype=np.float64)
    if not np.isfinite(w).all() or np.sum(np.abs(w)) <= 0:
        w = np.ones(len(df), dtype=np.float64)
    coords = df[list(xyz)].to_numpy(dtype=np.float64)
    if coords.size == 0:
        return None
    c = np.average(coords, weights=w, axis=0)
    if not np.isfinite(c).all():
        return None
    return [float(c[0]), float(c[1]), float(c[2])]


def aabb_from_points(df, xyz=("x", "y", "z")):
    if df is None or len(df) == 0:
        return None
    arr = df[list(xyz)].to_numpy(dtype=np.float64)
    if arr.size == 0 or not np.isfinite(arr).any():
        return None
    mn = np.nanmin(arr, axis=0)
    mx = np.nanmax(arr, axis=0)
    if not np.isfinite(mn).all() or not np.isfinite(mx).all():
        return None
    return [float(mn[0]), float(mn[1]), float(mn[2]), float(mx[0]), float(mx[1]), float(mx[2])]


def centroid_from_residue_positions(resids, residue_kcc, weights=None):
    pts = []
    ws = []
    for resid in resids:
        r = residue_kcc.get(int(resid))
        if not r:
            continue
        pos = r.get("ca_position")
        if not isinstance(pos, list) or len(pos) != 3:
            continue
        if all(math.isfinite(float(x)) for x in pos):
            pts.append([float(pos[0]), float(pos[1]), float(pos[2])])
            ws.append(float(weights.get(int(resid), 1.0)) if isinstance(weights, dict) else 1.0)
    if not pts:
        return None, None
    arr = np.asarray(pts, dtype=np.float64)
    w = np.asarray(ws, dtype=np.float64)
    if np.sum(np.abs(w)) <= 0:
        w = np.ones(len(arr))
    c = np.average(arr, weights=w, axis=0)
    mn = np.min(arr, axis=0)
    mx = np.max(arr, axis=0)
    return [float(c[0]), float(c[1]), float(c[2])], [float(mn[0]), float(mn[1]), float(mn[2]), float(mx[0]), float(mx[1]), float(mx[2])]


def centroid_view(name, centroid, aabb, frame, support_residues, definition, provenance, available=True):
    if centroid is None:
        available = False
    return {
        "slot": name,
        "available": bool(available),
        "centroid_A": centroid,
        "aabb_A": aabb,
        "frame": int(frame) if frame is not None else None,
        "view": name,
        "support_residues": sorted({int(x) for x in support_residues if x is not None}),
        "definition": definition,
        "provenance": provenance,
        "field_completeness": {
            "centroid_A": centroid is not None,
            "aabb_A": aabb is not None,
            "support_residues": bool(support_residues),
        },
    }


def zscore_series(values):
    arr = np.asarray(values, dtype=np.float64)
    if len(arr) == 0:
        return arr
    mu = np.nanmean(arr)
    sd = np.nanstd(arr)
    if not math.isfinite(sd) or sd <= 1e-12:
        return np.zeros_like(arr)
    return (arr - mu) / sd


def load_sites(binding_sites_path: Path):
    with binding_sites_path.open("r") as f:
        data = json.load(f)
    sites = data.get("sites", [])
    by_id = {int(s["id"]): s for s in sites if "id" in s}
    return data, by_id


def load_kcc(kcc_path: Path):
    with kcc_path.open("r") as f:
        data = json.load(f)
    residue_kcc = {}
    for r in data.get("residues", []):
        rid = safe_int(r.get("residue_id"))
        if rid is not None:
            residue_kcc[rid] = r
    return data, residue_kcc


def load_arrow_filtered(arrow_path: Path, min_intensity_percentile: int, background_classes, require_site_id: bool):
    with pa.memory_map(str(arrow_path), "r") as source:
        table = ipc.open_file(source).read_all()

    present = [c for c in ARROW_COLS if c in table.column_names]
    table = table.select(present)

    mask = pc.greater_equal(table["intensity_percentile"], min_intensity_percentile)

    # Keep primary/relabel if requested; schema says 0=primary, 4=relabel_candidate.
    if "background_class" in table.column_names:
        bg_mask = None
        for bg in background_classes:
            m = pc.equal(table["background_class"], int(bg))
            bg_mask = m if bg_mask is None else pc.or_(bg_mask, m)
        mask = pc.and_(mask, bg_mask)

    if require_site_id and "site_id" in table.column_names:
        mask = pc.and_(mask, pc.greater_equal(table["site_id"], 0))

    return table.filter(mask)


def explode_residue_contacts(ft: pa.Table):
    n = ft.num_rows
    rep = pa.array(np.repeat(np.arange(n), 8))
    flat = pc.list_flatten(ft["nearby_residues"])

    cols = {"residue_id": flat}
    scalar_cols = [
        "site_id",
        "stream_id",
        "group_id",
        "chunk_idx",
        "voxel_idx",
        "timestep",
        "frame_index",
        "x", "y", "z",
        "intensity",
        "spike_source",
        "aromatic_type",
        "aromatic_residue_id",
        "phase_bits",
        "n_nearby_excited",
        "vibrational_energy",
        "water_density",
        "wd_change",
        "wavelength_nm",
        "ccns_phase",
        "background_class",
        "burial_score",
        "intensity_percentile",
    ]

    for col in scalar_cols:
        if col in ft.column_names:
            cols[col] = pc.take(ft[col], rep)

    ex = pa.Table.from_pydict(cols)
    ex = ex.filter(pc.not_equal(ex["residue_id"], -1))
    return ex.to_pandas()


def compute_residue_features(contact_df):
    df = contact_df.copy()
    for c in ("x", "y", "z"):
        df[f"w{c}"] = df[c].astype(float) * df["intensity"].astype(float)

    agg = df.groupby(["site_id", "residue_id"]).agg(
        spike_count=("intensity", "count"),
        total_energy=("intensity", "sum"),
        mean_intensity=("intensity", "mean"),
        max_intensity=("intensity", "max"),
        sum_wx=("wx", "sum"),
        sum_wy=("wy", "sum"),
        sum_wz=("wz", "sum"),
        mean_vibrational_energy=("vibrational_energy", "mean"),
        mean_water_density=("water_density", "mean"),
        mean_wd_change=("wd_change", "mean"),
        abs_wd_change=("wd_change", lambda s: float(np.mean(np.abs(s)))),
        stream_consensus=("stream_id", "nunique"),
        phase_breadth=("ccns_phase", "nunique"),
        frame_persistence=("frame_index", "nunique"),
        aromatic_hits=("aromatic_type", lambda s: int((s >= 0).sum())),
        aromatic_diversity=("aromatic_type", lambda s: int(pd.Series(s).loc[pd.Series(s) >= 0].nunique())),
        source_diversity=("spike_source", "nunique"),
        excitation_neighbors=("n_nearby_excited", "mean"),
    ).reset_index()

    agg["mean_x"] = agg["sum_wx"] / agg["total_energy"].replace(0, np.nan)
    agg["mean_y"] = agg["sum_wy"] / agg["total_energy"].replace(0, np.nan)
    agg["mean_z"] = agg["sum_wz"] / agg["total_energy"].replace(0, np.nan)
    return agg


def top_residue_ids(residue_df, n=12, by="total_energy"):
    if residue_df is None or len(residue_df) == 0:
        return []
    return [int(x) for x in residue_df.sort_values(by, ascending=False)["residue_id"].head(n).tolist()]


def rank_sites(site_rows):
    # Normalize across candidate sites.
    keys = [
        "total_energy",
        "stream_consensus",
        "phase_modulation",
        "kcc_driver_factor",
        "desolvation_factor",
        "burst_factor",
        "source_diversity",
        "manifold_completeness",
        "centroid_coherence",
    ]

    vals = {k: [r["rank_features"].get(k, 0.0) for r in site_rows] for k in keys}
    z = {k: zscore_series(vals[k]) for k in keys}

    weights = {
        "total_energy": 0.16,
        "stream_consensus": 0.14,
        "phase_modulation": 0.13,
        "kcc_driver_factor": 0.16,
        "desolvation_factor": 0.12,
        "burst_factor": 0.11,
        "source_diversity": 0.07,
        "manifold_completeness": 0.07,
        "centroid_coherence": 0.04,
    }

    for i, row in enumerate(site_rows):
        score = 0.0
        expl = {}
        for k, w in weights.items():
            contribution = float(w * z[k][i])
            expl[k] = {
                "raw": float(row["rank_features"].get(k, 0.0)),
                "z": float(z[k][i]),
                "weight": w,
                "contribution": contribution,
            }
            score += contribution

        # Penalties: avoid terminal/edge inflation and single-residue collapse.
        resids = row.get("residue_support_family", {}).get("all_region_residues", [])
        terminal_like = bool(resids and min(resids) <= 7)
        single_residue = len(resids) <= 1
        penalty = 0.0
        if terminal_like:
            penalty -= 0.35
        if single_residue:
            penalty -= 0.25

        row["final_phase_manifold_score"] = float(score + penalty)
        row["score_explanation"] = {
            "components": expl,
            "penalties": {
                "terminal_like": terminal_like,
                "single_residue": single_residue,
                "penalty_total": penalty,
            },
            "legacy_ranker_fields_used": False,
            "ligand_truth_used": False,
        }

    site_rows.sort(key=lambda r: r["final_phase_manifold_score"], reverse=True)
    for rank, row in enumerate(site_rows, 1):
        row["rank"] = rank
    return site_rows


def build_manifest_for_site(
    sid,
    site,
    site_spikes,
    site_residue_df,
    residue_kcc,
    max_streams,
    source_paths,
):
    frame_med = int(site_spikes["frame_index"].median()) if len(site_spikes) else 0

    all_support = sorted({
        int(x) for x in site_residue_df["residue_id"].tolist()
    }) if len(site_residue_df) else sorted([int(x) for x in site.get("residue_ids", []) if x is not None])

    # Raw legacy geometric center is retained, but not used as the only centroid.
    geometric_centroid = site.get("centroid")
    if not isinstance(geometric_centroid, list) or len(geometric_centroid) != 3:
        geometric_centroid = weighted_centroid(site_spikes)
    geometric_aabb = aabb_from_points(site_spikes)

    # Lining view: authoritative support from binding_sites.lining_residues ranked by spike attribution.
    lining_rows = []
    for lr in site.get("lining_residues", []):
        rid = safe_int(lr.get("resid"))
        if rid is None:
            continue
        lining_rows.append({
            "resid": rid,
            "resname": lr.get("resname"),
            "spike_attribution_count": safe_int(lr.get("spike_attribution_count"), 0) or 0,
            "min_distance": finite_float(lr.get("min_distance"), 999.0),
        })
    lining_rows = sorted(lining_rows, key=lambda r: (-r["spike_attribution_count"], r["resid"]))
    lining_resids = [r["resid"] for r in lining_rows[:12]]

    # Prefer event-derived per-residue spike centroids for lining when available.
    lining_df = site_residue_df[site_residue_df["residue_id"].isin(lining_resids)] if len(site_residue_df) else pd.DataFrame()
    if len(lining_df):
        lining_points = lining_df.rename(columns={"mean_x": "x", "mean_y": "y", "mean_z": "z"}).copy()
        lining_points["intensity"] = lining_points["total_energy"]
        lining_centroid = weighted_centroid(lining_points)
        lining_aabb = aabb_from_points(lining_points)
    else:
        weights = {r["resid"]: r["spike_attribution_count"] for r in lining_rows}
        lining_centroid, lining_aabb = centroid_from_residue_positions(lining_resids, residue_kcc, weights)

    # Driver view: site KCC driver + candidate residue ids/support/confidence when present.
    kcc = site.get("kcc", {}) or {}
    driver_resids = []
    driver_weights = {}
    drv = safe_int(kcc.get("driver_residue_id"))
    if drv is not None:
        driver_resids.append(drv)
        driver_weights[drv] = 1.0

    cand_resids = kcc.get("candidate_residue_ids") or []
    cand_weights = kcc.get("candidate_causal_weights") or []
    for i, rid_raw in enumerate(cand_resids[:12]):
        rid = safe_int(rid_raw)
        if rid is None:
            continue
        driver_resids.append(rid)
        w = finite_float(cand_weights[i], 0.0) if i < len(cand_weights) else 0.0
        driver_weights[rid] = max(driver_weights.get(rid, 0.0), w)

    # Add top KCC residues that overlap site support if site nested KCC has sparse data.
    if len(driver_resids) < 3 and all_support:
        scored = []
        for rid in all_support:
            kr = residue_kcc.get(int(rid), {})
            scored.append((finite_float(kr.get("kcc_score"), 0.0), int(rid)))
        for score, rid in sorted(scored, reverse=True)[:6]:
            driver_resids.append(rid)
            driver_weights[rid] = max(driver_weights.get(rid, 0.0), score)

    driver_resids = sorted(set(driver_resids))
    driver_centroid, driver_aabb = centroid_from_residue_positions(driver_resids, residue_kcc, driver_weights)

    # Phase views.
    cold_spikes = site_spikes[site_spikes["ccns_phase"] == 0]
    hot_spikes = site_spikes[site_spikes["ccns_phase"] == 2]
    cold_support = top_residue_ids(site_residue_df[site_residue_df["residue_id"].isin(all_support)], n=12) if len(site_residue_df) else all_support[:12]
    hot_support = top_residue_ids(site_residue_df[site_residue_df["residue_id"].isin(all_support)], n=12) if len(site_residue_df) else all_support[:12]

    # Burst view: top intensity tail within the site.
    if len(site_spikes):
        q90 = float(site_spikes["intensity"].quantile(0.90))
        burst_spikes = site_spikes[site_spikes["intensity"] >= q90]
    else:
        burst_spikes = pd.DataFrame()
    burst_support = top_residue_ids(site_residue_df, n=10, by="max_intensity") if len(site_residue_df) and "max_intensity" in site_residue_df.columns else all_support[:10]

    # Validation structural: topology/KCC CA positions over site residue IDs; no ligand truth.
    structural_centroid, structural_aabb = centroid_from_residue_positions(all_support[:32], residue_kcc)

    views = {
        "geometric": centroid_view(
            "geometric",
            [float(x) for x in geometric_centroid] if geometric_centroid is not None else None,
            geometric_aabb,
            frame_med,
            all_support,
            "raw geometric site centroid retained as a compatibility view; not used as sole site center",
            {"source": "binding_sites.centroid plus spike-event AABB"},
        ),
        "lining": centroid_view(
            "lining",
            lining_centroid,
            lining_aabb,
            frame_med,
            lining_resids,
            "centroid over lining residues ranked by spike_attribution_count, not min_distance",
            {"source": "binding_sites.lining_residues + spike attribution + event-derived residue centroids"},
        ),
        "driver": centroid_view(
            "driver",
            driver_centroid,
            driver_aabb,
            frame_med,
            driver_resids,
            "centroid over KCC/causal driver support",
            {"source": "binding_sites.kcc + kcc_visualization.residues"},
        ),
        "hot_phase": centroid_view(
            "hot_phase",
            weighted_centroid(hot_spikes),
            aabb_from_points(hot_spikes),
            int(hot_spikes["frame_index"].median()) if len(hot_spikes) else frame_med,
            hot_support,
            "intensity-weighted centroid over warm_hold ccns_phase=2 spike events",
            {"source": "spike_events.arrow ccns_phase=2"},
        ),
        "cold_phase": centroid_view(
            "cold_phase",
            weighted_centroid(cold_spikes),
            aabb_from_points(cold_spikes),
            int(cold_spikes["frame_index"].median()) if len(cold_spikes) else frame_med,
            cold_support,
            "intensity-weighted centroid over cold_hold ccns_phase=0 spike events",
            {"source": "spike_events.arrow ccns_phase=0"},
        ),
        "burst_motion": centroid_view(
            "burst_motion",
            weighted_centroid(burst_spikes),
            aabb_from_points(burst_spikes),
            int(burst_spikes["frame_index"].median()) if len(burst_spikes) else frame_med,
            burst_support,
            "centroid over top 10 percent intensity spike burst tail",
            {"source": "spike_events.arrow intensity top decile within site"},
        ),
        "validation_structural": centroid_view(
            "validation_structural",
            structural_centroid,
            structural_aabb,
            frame_med,
            all_support,
            "structural residue-support centroid from kcc_visualization CA positions; validation-ready but not ligand-derived",
            {"source": "kcc_visualization.ca_position"},
        ),
        "ligand_adjacent_subcluster": centroid_view(
            "ligand_adjacent_subcluster",
            None,
            None,
            frame_med,
            [],
            "post-hoc validation-only view; unavailable because no ligand/reference was provided to this ranker",
            {"source": "not_used_in_ranking"},
            available=False,
        ),
    }

    # Rank features.
    total_energy = float(site_spikes["intensity"].sum()) if len(site_spikes) else 0.0
    n_spikes = int(len(site_spikes))
    stream_consensus = float(site_spikes["stream_id"].nunique()) / max(1.0, float(max_streams)) if len(site_spikes) else 0.0

    phase_counts = site_spikes["ccns_phase"].value_counts().to_dict() if len(site_spikes) else {}
    cold_n = float(phase_counts.get(0, 0))
    hot_n = float(phase_counts.get(2, 0))
    ramp_n = float(phase_counts.get(1, 0))
    phase_total = max(1.0, cold_n + hot_n + ramp_n)
    phase_modulation = abs(hot_n - cold_n) / phase_total + 0.25 * (ramp_n / phase_total)

    desolvation_factor = float(np.average(np.abs(site_spikes["wd_change"]), weights=np.maximum(site_spikes["intensity"], 1e-9))) if len(site_spikes) else 0.0
    burst_factor = finite_float(kcc.get("site_burst_motion", kcc.get("burst_motion", 0.0)), 0.0)

    driver_scores = []
    for rid in driver_resids:
        kr = residue_kcc.get(int(rid), {})
        driver_scores.append(finite_float(kr.get("kcc_score"), 0.0))
    kcc_driver_factor = float(np.mean(driver_scores)) if driver_scores else 0.0

    source_diversity = float(site_spikes["spike_source"].nunique()) if len(site_spikes) else 0.0
    available_views = sum(1 for v in views.values() if v["available"])
    manifold_completeness = available_views / 8.0

    # Coherence: smaller spread across available centroid views is better, but do not force all views to coincide.
    centroids = []
    for name, v in views.items():
        if v["available"] and v["centroid_A"] is not None and name != "ligand_adjacent_subcluster":
            centroids.append(v["centroid_A"])
    if len(centroids) >= 2:
        c_arr = np.asarray(centroids, dtype=np.float64)
        center = np.mean(c_arr, axis=0)
        dispersion = float(np.mean(np.linalg.norm(c_arr - center, axis=1)))
        centroid_coherence = 1.0 / (1.0 + dispersion)
    else:
        dispersion = None
        centroid_coherence = 0.0

    rank_features = {
        "total_energy": total_energy,
        "n_spikes": n_spikes,
        "stream_consensus": stream_consensus,
        "phase_modulation": float(phase_modulation),
        "kcc_driver_factor": kcc_driver_factor,
        "desolvation_factor": desolvation_factor,
        "burst_factor": burst_factor,
        "source_diversity": source_diversity,
        "manifold_completeness": manifold_completeness,
        "centroid_coherence": centroid_coherence,
        "centroid_view_dispersion_A": dispersion,
        "legacy_ranker_fields_used": False,
    }

    residue_support_family = {
        "all_region_residues": all_support,
        "lining_or_surface_residues": lining_resids,
        "kcc_driver_residues": driver_resids,
        "hot_phase_supported_residues": hot_support,
        "cold_phase_supported_residues": cold_support,
        "burst_motion_supported_residues": burst_support,
        "validation_contact_residues": [],
    }

    therm = {
        "therm_class": site.get("therm_class"),
        "hysteresis_asymmetry": site.get("hysteresis_asymmetry"),
        "relative_asymmetry": site.get("relative_asymmetry"),
        "ccns_tau": site.get("ccns_tau"),
        "cold_phase_fraction": site.get("cold_phase_fraction"),
        "tide_coupling_score": site.get("tide_coupling_score"),
        "tide_trigger_residues": site.get("tide_trigger_residues", []),
    }

    manifest = {
        "schema_version": 1,
        "schema_kind": "prism4d_phase_manifold_site_manifest",
        "site_identity": {
            "site_uid": f"site_{int(sid):05d}",
            "site_id": int(sid),
            "cluster_id": int(sid),
            "status": "phase_manifold_ranked_candidate_not_ligand_validated",
            "classification": site.get("classification"),
            "source": "raw_engine_outputs",
        },
        "coordinate_frames": {
            "angstrom": {
                "available": True,
                "frame_name": "engine_spike_event_coordinate_frame",
                "source": "spike_events.arrow x/y/z voxel center position in Angstroms",
            }
        },
        "centroid_manifold": {
            "selected_centroid_view": "driver" if views["driver"]["available"] else "lining",
            "selection_rule": "prefer KCC driver centroid, fallback to spike-attribution lining centroid; never use ligand truth",
            "views": views,
        },
        "residue_support_family": residue_support_family,
        "evidence_blocks": {
            "raw_site": {
                "volume": site.get("volume"),
                "spike_count": site.get("spike_count"),
                "quality_score_raw": site.get("quality_score"),
                "druggability_raw": site.get("druggability"),
                "signal_preservation": site.get("signal_preservation"),
            },
            "kcc": kcc,
            "therm": therm,
            "arrow_spikes": {
                "n_site_spikes_after_filter": n_spikes,
                "total_energy_after_filter": total_energy,
                "stream_count": int(site_spikes["stream_id"].nunique()) if len(site_spikes) else 0,
                "phase_counts": {str(k): int(v) for k, v in phase_counts.items()},
                "background_classes_used": sorted([int(x) for x in site_spikes["background_class"].unique().tolist()]) if len(site_spikes) and "background_class" in site_spikes else [],
                "intensity_percentile_min": int(site_spikes["intensity_percentile"].min()) if len(site_spikes) else None,
            },
        },
        "rank_features": rank_features,
        "validation": {
            "validation_status": "not_run",
            "ranking_uses_ligand_truth": False,
            "dcc_by_centroid_view": {},
            "min_distance_by_centroid_view": {},
        },
        "limitations": [
            "ligand_adjacent_subcluster_unavailable_without_reference_ligand",
            "ranker_ignores_legacy_composite_rank_fields",
            "phase labels limited to writer values 0=cold_hold,1=ramp,2=warm_hold",
        ],
        "provenance": {
            "source_artifacts": source_paths,
            "ranker": "phase_manifold_ranker.py",
        },
    }
    return manifest


def main():
    ap = argparse.ArgumentParser(description="PRISM-4D phase-manifold-aware binding-site ranker.")
    ap.add_argument("--arrow", required=True, help="Path to <base>.spike_events.arrow")
    ap.add_argument("--binding-sites", required=True, help="Path to <base>.binding_sites.json")
    ap.add_argument("--kcc", required=True, help="Path to <base>.kcc_visualization.json")
    ap.add_argument("--outdir", required=True, help="Output directory")
    ap.add_argument("--min-intensity-percentile", type=int, default=70)
    ap.add_argument("--background-classes", default="0,4", help="Comma-separated background_class values to keep. Default 0,4.")
    ap.add_argument("--top", type=int, default=50)
    args = ap.parse_args()

    arrow_path = Path(args.arrow)
    sites_path = Path(args.binding_sites)
    kcc_path = Path(args.kcc)
    outdir = Path(args.outdir)
    manifests_dir = outdir / "site_manifests"
    manifests_dir.mkdir(parents=True, exist_ok=True)

    background_classes = [int(x.strip()) for x in args.background_classes.split(",") if x.strip() != ""]

    print("=== PRISM-4D PHASE-MANIFOLD RANKER ===")
    print(f"arrow={arrow_path}")
    print(f"binding_sites={sites_path}")
    print(f"kcc={kcc_path}")
    print(f"outdir={outdir}")
    print(f"filter: intensity_percentile >= {args.min_intensity_percentile}, background_class in {background_classes}, site_id >= 0")
    print("legacy composite rank fields: IGNORED")

    binding_data, sites_by_id = load_sites(sites_path)
    kcc_data, residue_kcc = load_kcc(kcc_path)

    ft = load_arrow_filtered(
        arrow_path,
        min_intensity_percentile=args.min_intensity_percentile,
        background_classes=background_classes,
        require_site_id=True,
    )

    if ft.num_rows == 0:
        raise SystemExit("No spike rows survived filtering. Lower --min-intensity-percentile or adjust --background-classes.")

    # Base per-site spike dataframe. Do not explode for centroid phase views.
    scalar_cols = [c for c in [
        "site_id", "stream_id", "group_id", "chunk_idx", "voxel_idx", "timestep", "frame_index",
        "x", "y", "z", "intensity", "spike_source", "aromatic_type", "aromatic_residue_id",
        "phase_bits", "n_residues", "n_nearby_excited", "vibrational_energy",
        "water_density", "wd_change", "wavelength_nm", "ccns_phase",
        "background_class", "burial_score", "intensity_percentile"
    ] if c in ft.column_names]
    site_spikes_df = ft.select(scalar_cols).to_pandas()

    contact_df = explode_residue_contacts(ft)
    residue_features = compute_residue_features(contact_df)

    max_streams = int(site_spikes_df["stream_id"].nunique()) if "stream_id" in site_spikes_df.columns else 1

    source_paths = {
        "spike_events_arrow": str(arrow_path),
        "spike_events_arrow_sha16": sha16(arrow_path),
        "binding_sites_json": str(sites_path),
        "binding_sites_json_sha16": sha16(sites_path),
        "kcc_visualization_json": str(kcc_path),
        "kcc_visualization_json_sha16": sha16(kcc_path),
    }

    site_rows = []
    for sid in sorted(site_spikes_df["site_id"].dropna().unique().tolist()):
        sid = int(sid)
        site = sites_by_id.get(sid, {"id": sid})
        s_spikes = site_spikes_df[site_spikes_df["site_id"] == sid]
        s_res = residue_features[residue_features["site_id"] == sid] if len(residue_features) else pd.DataFrame()

        manifest = build_manifest_for_site(
            sid=sid,
            site=site,
            site_spikes=s_spikes,
            site_residue_df=s_res,
            residue_kcc=residue_kcc,
            max_streams=max_streams,
            source_paths=source_paths,
        )
        site_rows.append(manifest)

    ranked = rank_sites(site_rows)

    # Write per-site manifests.
    for m in ranked:
        uid = m["site_identity"]["site_uid"]
        (manifests_dir / f"{uid}.site_manifest.json").write_text(json.dumps(m, indent=2))

    atlas = {
        "schema_version": 1,
        "schema_kind": "phase_manifold_ranked_site_manifest_atlas",
        "status": "phase_manifold_ranked_candidates_not_ligand_validated",
        "filter": {
            "min_intensity_percentile": args.min_intensity_percentile,
            "background_classes": background_classes,
            "site_id_filter": "site_id >= 0",
        },
        "ranking_rules": {
            "legacy_ranker_fields_used": False,
            "ligand_truth_used": False,
            "primary_centroid_family": CENTROID_SLOTS,
            "score_components": [
                "total_energy",
                "stream_consensus",
                "phase_modulation",
                "kcc_driver_factor",
                "desolvation_factor",
                "burst_factor",
                "source_diversity",
                "manifold_completeness",
                "centroid_coherence",
            ],
        },
        "source_artifacts": source_paths,
        "n_sites_ranked": len(ranked),
        "sites": ranked[:args.top],
    }

    (outdir / "ranked_site_manifest_atlas.json").write_text(json.dumps(atlas, indent=2))

    # Compact CSV summary.
    rows = []
    for m in ranked:
        views = m["centroid_manifold"]["views"]
        rf = m["rank_features"]
        rows.append({
            "rank": m["rank"],
            "site_id": m["site_identity"]["site_id"],
            "score": m["final_phase_manifold_score"],
            "selected_view": m["centroid_manifold"]["selected_centroid_view"],
            "status": m["site_identity"]["status"],
            "n_residues": len(m["residue_support_family"]["all_region_residues"]),
            "n_spikes": rf["n_spikes"],
            "total_energy": rf["total_energy"],
            "stream_consensus": rf["stream_consensus"],
            "phase_modulation": rf["phase_modulation"],
            "kcc_driver_factor": rf["kcc_driver_factor"],
            "desolvation_factor": rf["desolvation_factor"],
            "burst_factor": rf["burst_factor"],
            "manifold_completeness": rf["manifold_completeness"],
            "centroid_view_dispersion_A": rf["centroid_view_dispersion_A"],
            "driver_centroid_A": views["driver"]["centroid_A"],
            "lining_centroid_A": views["lining"]["centroid_A"],
            "hot_phase_centroid_A": views["hot_phase"]["centroid_A"],
            "cold_phase_centroid_A": views["cold_phase"]["centroid_A"],
            "burst_motion_centroid_A": views["burst_motion"]["centroid_A"],
            "top_driver_residues": m["residue_support_family"]["kcc_driver_residues"][:10],
            "top_lining_residues": m["residue_support_family"]["lining_or_surface_residues"][:10],
        })
    pd.DataFrame(rows).to_csv(outdir / "ranked_site_manifest_summary.csv", index=False)

    # Human-readable cards.
    cards_path = outdir / "phase_manifold_site_cards.md"
    with cards_path.open("w") as f:
        f.write("# PRISM-4D Phase-Manifold Site Cards\n\n")
        f.write("Legacy composite rank fields ignored. Ligand truth not used.\n\n")
        for m in ranked[:args.top]:
            sid = m["site_identity"]["site_id"]
            f.write(f"## Rank {m['rank']} — Site {sid}\n\n")
            f.write(f"- Score: `{m['final_phase_manifold_score']:.6f}`\n")
            f.write(f"- Selected view: `{m['centroid_manifold']['selected_centroid_view']}`\n")
            f.write(f"- Status: `{m['site_identity']['status']}`\n")
            f.write(f"- Classification: `{m['site_identity'].get('classification')}`\n")
            f.write(f"- Residues: `{m['residue_support_family']['all_region_residues'][:40]}`\n")
            f.write(f"- Driver residues: `{m['residue_support_family']['kcc_driver_residues'][:20]}`\n")
            f.write(f"- Lining residues: `{m['residue_support_family']['lining_or_surface_residues'][:20]}`\n\n")
            f.write("### Centroid views\n\n")
            for name in CENTROID_SLOTS:
                v = m["centroid_manifold"]["views"][name]
                f.write(f"- `{name}`: available={v['available']} centroid_A={v['centroid_A']} support={v['support_residues'][:12]}\n")
            f.write("\n### Score features\n\n")
            for k, v in m["rank_features"].items():
                f.write(f"- `{k}`: `{v}`\n")
            f.write("\n---\n\n")

    print(f"\nWROTE:")
    print(f"  {outdir / 'ranked_site_manifest_atlas.json'}")
    print(f"  {outdir / 'ranked_site_manifest_summary.csv'}")
    print(f"  {cards_path}")
    print(f"  {manifests_dir}/<site_uid>.site_manifest.json")
    print("\nTOP SITES:")
    for m in ranked[:min(10, len(ranked))]:
        print(
            f"rank={m['rank']:>2} site={m['site_identity']['site_id']:>5} "
            f"score={m['final_phase_manifold_score']:+.4f} "
            f"selected={m['centroid_manifold']['selected_centroid_view']} "
            f"energy={m['rank_features']['total_energy']:.2e} "
            f"streams={m['rank_features']['stream_consensus']:.2f} "
            f"kcc={m['rank_features']['kcc_driver_factor']:.4f}"
        )


if __name__ == "__main__":
    main()
