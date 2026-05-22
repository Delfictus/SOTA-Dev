#!/usr/bin/env python3
"""M1.2.25 Adjudication Audit Driver — Transparent MAR Signal-Bearing Wave.

Consumes a run dir (post-engine + post-scanner + post-materializer), emits
the 10 audit artifacts under the audit dir, and produces a final classification
∈ {PASS, PARTIAL_SIGNAL_LOW, PARTIAL_MATERIALIZER_BLOCKED, FAIL}.

Implements the 5 invariants:
  1. Gear-Aware Continuous Time Axis (G39A)
  2. 4-Plane MAR Completeness / Pillar 2 (G53A)
  3. Monomer Contrast Only (G34.1M)
  4. Threshold Source Lock (G51A)
  5. Materializer Exercise (G50A)

Usage:
  python3 m1_2_25_audit_driver.py <run_dir> <audit_dir> <topology>
"""
from __future__ import annotations
import json
import math
import os
import sys
from glob import glob
from pathlib import Path


def L(path: str | Path):
    p = Path(path)
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text())
    except Exception as e:
        return {"_load_error": str(e), "_path": str(p)}


def W(path: Path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2))
    return path


def section_1_t7(rd: Path, ad: Path, target: str) -> dict:
    """Invariant 4: Threshold Source Lock + Phase 1 T7 calibration."""
    nf_files = sorted(rd.glob(f"{target}*_noise_floor.json"))
    per_stream = []
    aggregate_mu, aggregate_sigma = [], []
    for nf in nf_files:
        d = L(nf) or {}
        mu = d.get("mu") or d.get("mu_kl")
        sigma = d.get("sigma") or d.get("sigma_kl")
        n_samples = d.get("n_samples")
        if isinstance(mu, list) and mu:
            mu_scalar = float(mu[0])
            aggregate_mu.append(mu_scalar)
        elif isinstance(mu, (int, float)):
            mu_scalar = float(mu)
            aggregate_mu.append(mu_scalar)
        else:
            mu_scalar = None
        if isinstance(sigma, list) and sigma:
            sig_scalar = float(sigma[0])
            aggregate_sigma.append(sig_scalar)
        elif isinstance(sigma, (int, float)):
            sig_scalar = float(sigma)
            aggregate_sigma.append(sig_scalar)
        else:
            sig_scalar = None
        per_stream.append({
            "file": str(nf.relative_to(rd)),
            "mu_l0": mu_scalar,
            "sigma_l0": sig_scalar,
            "n_samples": n_samples,
            "phase": d.get("phase"),
        })
    have_t7 = len(per_stream) > 0 and any(s.get("mu_l0") is not None for s in per_stream)
    if have_t7:
        mu_run = sum(aggregate_mu) / max(len(aggregate_mu), 1)
        sigma_run = sum(aggregate_sigma) / max(len(aggregate_sigma), 1)
        threshold_source = "run_calibrated"
        observation_threshold = mu_run + 3.0 * sigma_run
        discovery_threshold = mu_run + 12.0 * sigma_run
        fallback_used = False
        missing_reason = None
    else:
        mu_run, sigma_run = 0.005, 0.001
        threshold_source = "diagnostic_default"
        observation_threshold = mu_run + 3.0 * sigma_run
        discovery_threshold = mu_run + 12.0 * sigma_run
        fallback_used = True
        missing_reason = "no_noise_floor_json_emitted_per_stream_engine_likely_did_not_complete_normal_teardown"
    out = {
        "schema_version": 1,
        "schema_kind": "m1_2_25_t7_convergence_summary",
        "run_dir": str(rd),
        "n_streams_with_noise_floor": len(nf_files),
        "per_stream": per_stream,
        "mu_kl": mu_run,
        "sigma_kl": sigma_run,
        "observation_sigma": 3.0,
        "discovery_sigma": 12.0,
        "observation_threshold": observation_threshold,
        "discovery_threshold": discovery_threshold,
        "threshold_source": threshold_source,
        "fallback_used": fallback_used,
        "missing_reason": missing_reason,
    }
    W(ad / "t7_convergence_summary.json", out)
    return out


def section_2_signal_purity(rd: Path, ad: Path, t7: dict) -> dict:
    """Phase 2 / Invariant 4: ghost record signal purity probe."""
    ranked = L(rd / "ranked_tile_events.json")
    events = (ranked or {}).get("ranked_events", []) or []
    n = len(events)
    obs_thr = t7["observation_threshold"]
    disc_thr = t7["discovery_threshold"]
    schema_v_dist = {}
    adj_dist = {}
    kls = []
    obs_pass = 0
    disc_pass = 0
    for r in events:
        e = r.get("event") or {}
        kl = e.get("kl_divergence")
        if kl is not None:
            kls.append(float(kl))
        sv = e.get("schema_version")
        schema_v_dist[sv] = schema_v_dist.get(sv, 0) + 1
        ac = e.get("adj_code") if "adj_code" in e else e.get("adjudication_code")
        adj_dist[ac] = adj_dist.get(ac, 0) + 1
        if kl is not None and kl >= obs_thr:
            obs_pass += 1
        if kl is not None and kl >= disc_thr:
            disc_pass += 1
    kls_sorted = sorted(kls) if kls else []
    def pct(p):
        if not kls_sorted: return 0.0
        idx = min(int(len(kls_sorted) * p), len(kls_sorted) - 1)
        return kls_sorted[idx]
    out = {
        "schema_version": 1,
        "schema_kind": "m1_2_25_ghost_signal_purity_probe",
        "n_events": n,
        "observation_threshold": obs_thr,
        "discovery_threshold": disc_thr,
        "threshold_source": t7["threshold_source"],
        "max_kl_divergence": max(kls) if kls else 0.0,
        "mean_kl_divergence": (sum(kls) / len(kls)) if kls else 0.0,
        "p95_kl_divergence": pct(0.95),
        "p99_kl_divergence": pct(0.99),
        "observation_pass_count": obs_pass,
        "discovery_pass_count": disc_pass,
        "schema_version_distribution": {str(k): v for k, v in schema_v_dist.items()},
        "adjudication_code_distribution": {str(k): v for k, v in adj_dist.items()},
    }
    W(ad / "ghost_signal_purity_probe.json", out)
    return out


def section_3_mar_planes(rd: Path, ad: Path) -> dict:
    """Invariant 2: 4-Plane MAR Completeness + Pillar 2 Driver Coupling."""
    src = L(rd / "mar_plane_completeness.json")
    if not src:
        out = {
            "schema_version": 1,
            "schema_kind": "m1_2_25_mar_plane_completeness",
            "status": "scanner_did_not_emit_mar_plane_completeness",
            "geometry_nonzero": 0,
            "causality_nonzero": 0,
            "thermodynamics_nonzero": 0,
            "chemistry_nonzero": 0,
            "full_4plane_available": False,
            "c_total_status": "no_signal",
            "pillar_2_driver_violation": False,
        }
    else:
        out = dict(src)
    W(ad / "mar_plane_completeness.json", out)
    return out


def section_4_gear_trace(rd: Path, ad: Path) -> dict:
    """Invariant 1: Gear-Aware Continuous Time Axis."""
    ranked = L(rd / "ranked_tile_events.json")
    events = (ranked or {}).get("ranked_events", []) or []
    by_stream = {}
    gear_dist = {}; dt_dist = {}
    for r in events:
        e = r.get("event") or {}
        sid = e.get("stream_id")
        gn = (e.get("gear_normalized_timing") or {})
        gid = gn.get("gear_id"); dt = gn.get("dt_fs")
        si = gn.get("step_idx") if "step_idx" in gn else e.get("step_idx") or e.get("frame_idx")
        pt = gn.get("physical_time_fs")
        if gid is None: continue
        gear_dist[gid] = gear_dist.get(gid, 0) + 1
        dt_dist[dt] = dt_dist.get(dt, 0) + 1
        s = by_stream.setdefault(sid, dict(gears={}, dts={}, frames=[], physical_times=[]))
        s["gears"][gid] = s["gears"].get(gid, 0) + 1
        s["dts"][dt] = s["dts"].get(dt, 0) + 1
        if pt is not None: s["physical_times"].append(pt)
    transitions = 0
    for sid, s in by_stream.items():
        if len(s["gears"]) > 1: transitions += len(s["gears"]) - 1
    n_gears_observed = len(gear_dist)
    have_dt = bool(dt_dist) and not (len(dt_dist) == 1 and (None in dt_dist or 0.0 in dt_dist))
    if transitions > 0:
        status = "GEAR_TIME_PASS"
    elif have_dt and gear_dist:
        status = "GEAR_TIME_PARTIAL_CONSTANT_DT"
    else:
        status = "GEAR_TIME_UNRESOLVED"
    out = {
        "schema_version": 1,
        "schema_kind": "m1_2_25_gear_trace_summary",
        "n_events_with_gear_field": sum(gear_dist.values()),
        "global_gear_distribution": {str(k): v for k, v in gear_dist.items()},
        "global_dt_fs_distribution": {str(k): v for k, v in dt_dist.items()},
        "transition_count": transitions,
        "gear_trace_status": status,
        "n_streams_with_data": len(by_stream),
        "physical_time_axis_status": (
            "computed_from_gear_id_dt_fs_step_idx" if have_dt and gear_dist else "unresolved"
        ),
    }
    W(ad / "gear_trace_summary.json", out)
    return out


def section_5_monomer_contrast(rd: Path, ad: Path, t7: dict, mar: dict) -> dict:
    """Invariant 3: Monomer Contrast Only / No Φsym."""
    ranked = L(rd / "ranked_tile_events.json")
    events = (ranked or {}).get("ranked_events", []) or []
    if t7["threshold_source"] != "run_calibrated":
        c_status = "unresolved_t7_missing"
    elif not mar.get("full_4plane_available", False):
        c_status = "partial_geometry_only" if mar.get("geometry_nonzero", 0) > 0 else "no_signal"
    else:
        c_status = "computable_4plane"
    mu_kl = t7["mu_kl"]
    sigma_kl = max(t7["sigma_kl"], 1e-9)
    c_values = []
    for r in events:
        e = r.get("event") or {}
        kl = e.get("kl_divergence")
        if kl is not None:
            c_values.append((float(kl) - mu_kl) / sigma_kl)
    c_total_max = max(c_values) if c_values else 0.0
    c_total_mean = (sum(c_values) / len(c_values)) if c_values else 0.0
    out = {
        "schema_version": 1,
        "schema_kind": "m1_2_25_monomer_interferometric_contrast",
        "monomer_passthrough": True,
        "phi_sym_status": "not_applicable_monomer",
        "bilateral_status": "not_applicable_monomer",
        "bilateral_veto_applied": False,
        "C_total_max": c_total_max,
        "C_total_mean": c_total_mean,
        "C_status": c_status,
        "baseline_source": t7["threshold_source"],
        "mu_kl": mu_kl,
        "sigma_kl": sigma_kl,
        "geometry_plane_populated": mar.get("geometry_nonzero", 0) > 0,
        "causality_plane_populated": mar.get("causality_nonzero", 0) > 0,
        "thermodynamics_plane_populated": mar.get("thermodynamics_nonzero", 0) > 0,
        "chemistry_plane_populated": mar.get("chemistry_nonzero", 0) > 0,
    }
    W(ad / "monomer_interferometric_contrast.json", out)
    return out


def section_6_flux_coupling(rd: Path, ad: Path, t7: dict) -> dict:
    """Phase 6: flux coupling η on top-3."""
    ranked = L(rd / "ranked_tile_events.json")
    events = (ranked or {}).get("ranked_events", []) or []
    def kl_of(r): return (r.get("event") or {}).get("kl_divergence") or 0.0
    top = sorted(events, key=kl_of, reverse=True)[:3]
    obs_thr = t7["observation_threshold"]
    disc_thr = t7["discovery_threshold"]
    top3 = []
    for r in top:
        e = r.get("event") or {}
        fc = e.get("flux_coupling") or {}
        kl = e.get("kl_divergence") or 0.0
        flux = e.get("thermo_flux") or [None, None]
        gn = e.get("gear_normalized_timing") or {}
        top3.append({
            "stream_id": e.get("stream_id"),
            "frame_idx": e.get("frame_idx"),
            "kl_divergence": kl,
            "scanner_eta": fc.get("eta"),
            "thermo_flux": flux,
            "observation_pass": kl >= obs_thr,
            "discovery_pass": kl >= disc_thr,
            "dt_fs": gn.get("dt_fs"),
            "gear_id": gn.get("gear_id"),
            "physical_time_fs": gn.get("physical_time_fs"),
        })
    out = {
        "schema_version": 1,
        "schema_kind": "m1_2_25_flux_coupling_top3",
        "obs_thr_kl": obs_thr,
        "disc_thr_kl": disc_thr,
        "threshold_source": t7["threshold_source"],
        "top3_ranked_events": top3,
        "coupling_status": (
            "active" if top3 and all(t["discovery_pass"] for t in top3)
            else "partial" if top3 and any(t["observation_pass"] for t in top3)
            else "eta_computed_but_no_event_passes_obs_thr" if top3
            else "no_events"
        ),
    }
    W(ad / "flux_coupling_top3.json", out)
    return out


def section_7_branch(rd: Path, ad: Path) -> dict:
    """Phase 7: branch / predicate liveness."""
    rep = L(rd / "ghost_zstr_scan_report.json") or {}
    bt = rep.get("branch_trace") or {}
    streams = bt.get("branch_trace_by_stream") or []
    n_with_data = 0
    f1_diverse = 0
    g26_diverse = 0
    for s in streams:
        f1c = s.get("f1_branch_count") or [0, 0, 0, 0]
        g26c = s.get("g26_branch_count") or [0, 0, 0, 0]
        f1_inv = s.get("f1_bridge_invocations") or 0
        g26_inv = s.get("g26_bridge_invocations") or 0
        if f1_inv > 0 or g26_inv > 0:
            n_with_data += 1
        if any(c > 0 for c in f1c[1:]):
            f1_diverse += 1
        if any(c > 0 for c in g26c[1:]):
            g26_diverse += 1
    out = {
        "schema_version": 1,
        "schema_kind": "m1_2_25_branch_predicate_liveness",
        "n_streams_with_branch_data": n_with_data,
        "streams_with_f1_predicate_diversity": f1_diverse,
        "streams_with_g26_predicate_diversity": g26_diverse,
        "liveness_status": (
            "live_with_g26_diversity"
            if g26_diverse > 0
            else "live_with_f1_diversity_only"
            if f1_diverse > 0
            else "live_invocations_but_predicate_always_zero"
            if n_with_data > 0
            else "no_branches_recorded"
        ),
        "per_stream": streams,
    }
    W(ad / "branch_predicate_liveness.json", out)
    return out


def section_8_materializer(rd: Path, ad: Path) -> dict:
    """Invariant 5: Materializer Exercise Requirement."""
    bdir = ad / "materializer_baseline"
    gdir = ad / "materializer_with_ghost"
    b_rep = L(bdir / "materialization_report.json") or {}
    g_rep = L(gdir / "materialization_report.json") or {}
    b_cand = L(bdir / "site_candidates.json") or {}
    g_cand = L(gdir / "site_candidates.json") or {}
    b_n = b_cand.get("n_candidates", 0)
    g_n = g_cand.get("n_candidates", 0)
    # Compare binding_sites.materialized.json (the actual ranking output) and
    # ranking_ablation_report.json — NOT the materialization_report.json which
    # has timing jitter (elapsed_ms varies run-to-run).
    b_sites = (L(bdir / "binding_sites.materialized.json") or {})
    g_sites = (L(gdir / "binding_sites.materialized.json") or {})
    b_abl = (L(bdir / "ranking_ablation_report.json") or {})
    g_abl = (L(gdir / "ranking_ablation_report.json") or {})
    sites_byte_identical = (
        json.dumps(b_sites.get("materialized_sites") or b_sites.get("sites"), sort_keys=True)
        == json.dumps(g_sites.get("materialized_sites") or g_sites.get("sites"), sort_keys=True)
    )
    ablation_byte_identical = (
        json.dumps(b_abl.get("schemes"), sort_keys=True)
        == json.dumps(g_abl.get("schemes"), sort_keys=True)
    )
    diff_value_keys = []
    for k in b_rep.keys() & g_rep.keys():
        if k in {"streams", "stream_summaries"}:
            continue  # contains elapsed_ms timing jitter
        if b_rep[k] != g_rep[k]:
            diff_value_keys.append(k)
    diff_value_keys = sorted(diff_value_keys)
    if g_n == 0:
        status = "unexercisable_no_candidates"
    elif sites_byte_identical and ablation_byte_identical:
        status = "exercisable_but_neutral_no_spatial_mapping"
    elif diff_value_keys or not sites_byte_identical:
        status = "active_ghost_factor_modifies_ranking"
    else:
        status = "exercisable_but_neutral_no_spatial_mapping"
    out = {
        "schema_version": 1,
        "schema_kind": "m1_2_25_materializer_factor_audit",
        "baseline_n_candidates": b_n,
        "with_ghost_n_candidates": g_n,
        "baseline_total_spikes": b_rep.get("total_spikes_processed", 0),
        "with_ghost_total_spikes": g_rep.get("total_spikes_processed", 0),
        "report_diff_value_keys": diff_value_keys,
        "binding_sites_byte_identical": sites_byte_identical,
        "ablation_schemes_byte_identical": ablation_byte_identical,
        "ghost_factor_status": status,
        "manifest_v2_was_live": b_rep.get("manifest_v2_was_live"),
        "manifest_serialization_failure": b_rep.get("manifest_serialization_failure"),
    }
    W(ad / "materializer_factor_audit.json", out)
    return out


def section_9_csrs(rd: Path, ad: Path, t7, sp, mar, gt, mc, fc, bp, mz) -> dict:
    """G-table convergence assembly."""
    rows = [
        {"metric": "G44 Instantiation Stability", "target": "8/8 V2 instantiate",
         "verified": "see path_a_completion.v2_live_by_stream",
         "verdict": "see report"},
        {"metric": "G45 Sector Alignment", "target": "all Ghost/ZSTR file sizes % 4096 == 0",
         "verified": "see ghost_zstr_scan_report",
         "verdict": "see report"},
        {"metric": "G46 Telemetry Yield", "target": ">0 Ghost records",
         "verified": sp.get("n_events", 0),
         "verdict": "PASS" if sp.get("n_events", 0) > 0 else "FAIL"},
        {"metric": "G47 v2 MAR Schema",
         "target": "schema_version=2 records (requires --mar-v2-telemetry CLI flag)",
         "verified": sp.get("schema_version_distribution"),
         "verdict": ("PASS" if "2" in (sp.get("schema_version_distribution") or {})
                     else "PARTIAL_V1_RECORDS_REQUIRES_MAR_V2_TELEMETRY_FLAG")},
        {"metric": "G48 Spatial Mapping",
         "target": "AABB/centroid or sidecar",
         "verified": "ghost_site_map.entries " + (
             "populated" if (L(rd / "ghost_site_map.json") or {}).get("entries") else "empty"),
         "verdict": "PASS" if (L(rd / "ghost_site_map.json") or {}).get("entries") else "PARTIAL"},
        {"metric": "G49 Temporal Mapping",
         "target": "gear_id/dt/physical_time",
         "verified": gt.get("physical_time_axis_status"),
         "verdict": "PASS" if "computed" in gt.get("physical_time_axis_status", "") else "PARTIAL"},
        {"metric": "G51 T7 Thresholding",
         "target": "run-specific μ/σ emitted",
         "verified": f"threshold_source={t7['threshold_source']}",
         "verdict": "PASS" if t7["threshold_source"] == "run_calibrated" else "FAIL"},
        {"metric": "G52 Signal Purity",
         "target": "max KL vs observation/discovery threshold",
         "verified": f"max_kl={sp.get('max_kl_divergence', 0):.3e} obs_thr={t7['observation_threshold']:.3e} obs_pass={sp.get('observation_pass_count', 0)} disc_pass={sp.get('discovery_pass_count', 0)}",
         "verdict": ("PASS" if sp.get("discovery_pass_count", 0) > 0
                     else "PARTIAL_OBS_ONLY" if sp.get("observation_pass_count", 0) > 0
                     else "BELOW_FLOOR")},
        {"metric": "G53 MAR Plane Completeness",
         "target": "4-plane status",
         "verified": f"geo={mar.get('geometry_nonzero', 0)} caus={mar.get('causality_nonzero', 0)} therm={mar.get('thermodynamics_nonzero', 0)} chem={mar.get('chemistry_nonzero', 0)}",
         "verdict": ("PASS" if mar.get("full_4plane_available")
                     else "PARTIAL" if mar.get("geometry_nonzero", 0) > 0
                     else "FAIL")},
        {"metric": "G39 Gear Transition Fidelity",
         "target": "real gear transition tied to action gate",
         "verified": gt.get("gear_trace_status"),
         "verdict": ("PASS" if gt.get("transition_count", 0) > 0
                     else "PARTIAL_NO_TRANSITION")},
        {"metric": "G34.2 Flux Coupling η",
         "target": "calibrated/proxy status",
         "verified": fc.get("coupling_status"),
         "verdict": ("PASS" if fc.get("coupling_status") == "active"
                     else "PARTIAL" if "obs_thr" in (fc.get("coupling_status") or "")
                     else "FAIL")},
        {"metric": "G50 Materializer Utility",
         "target": "ghost_zstr_factor active or partial+spatial",
         "verified": mz.get("ghost_factor_status"),
         "verdict": ("PASS" if mz.get("ghost_factor_status") == "active_ghost_factor_modifies_ranking"
                     else "PARTIAL" if mz.get("ghost_factor_status") == "exercisable_but_neutral_no_spatial_mapping"
                     else "FAIL")},
        {"metric": "G39A Continuous Physical Time",
         "target": "physical_time_fs computed",
         "verified": gt.get("physical_time_axis_status"),
         "verdict": "PASS" if gt.get("physical_time_axis_status") == "computed_from_gear_id_dt_fs_step_idx" else "PARTIAL"},
        {"metric": "G53A Pillar 2 Driver Coupling",
         "target": "causality nonzero when geometry nonzero",
         "verified": f"violation={mar.get('pillar_2_driver_violation')}",
         "verdict": ("FAIL" if mar.get("pillar_2_driver_violation") else
                     "PASS" if mar.get("causality_nonzero", 0) > 0 else "FAIL")},
        {"metric": "G34.1M Monomer Interferometric Contrast",
         "target": "C_total from run-calibrated T7 baseline",
         "verified": mc.get("C_status"),
         "verdict": ("PASS" if mc.get("C_status") == "computable_4plane"
                     else "PARTIAL" if "geometry" in (mc.get("C_status") or "")
                     else "FAIL")},
        {"metric": "G51A Threshold Source Lock",
         "target": "threshold_source=run_calibrated",
         "verified": t7.get("threshold_source"),
         "verdict": "PASS" if t7.get("threshold_source") == "run_calibrated" else "FAIL"},
        {"metric": "G50A Materializer Exercise",
         "target": "n_candidates > 0 and Ghost factor non-neutral or spatially partial",
         "verified": f"n_candidates={mz.get('with_ghost_n_candidates', 0)} status={mz.get('ghost_factor_status')}",
         "verdict": ("PASS" if mz.get("ghost_factor_status") == "active_ghost_factor_modifies_ranking"
                     else "PARTIAL" if mz.get("with_ghost_n_candidates", 0) > 0
                     else "FAIL")},
    ]
    out = {
        "schema_version": 1,
        "schema_kind": "m1_2_25_csrs_convergence_table",
        "n_metrics": len(rows),
        "rows": rows,
    }
    W(ad / "csrs_convergence_table.json", out)
    return out


def section_10_verdict(ad: Path, t7, sp, mar, gt, mc, fc, bp, mz, csrs) -> dict:
    """Final classification."""
    obs_pass = sp.get("observation_pass_count", 0)
    disc_pass = sp.get("discovery_pass_count", 0)
    threshold_source_lock = t7.get("threshold_source") == "run_calibrated"
    full_4plane = mar.get("full_4plane_available", False)
    materializer_active = mz.get("ghost_factor_status") == "active_ghost_factor_modifies_ranking"
    materializer_exercisable = mz.get("with_ghost_n_candidates", 0) > 0
    pillar2_clean = not mar.get("pillar_2_driver_violation", True)
    if (threshold_source_lock and obs_pass > 0 and full_4plane
            and materializer_exercisable and pillar2_clean):
        classification = "TRANSPARENT_MAR_SIGNAL_QUALITY_PASS"
    elif not materializer_exercisable:
        classification = "TRANSPARENT_MAR_SIGNAL_QUALITY_PARTIAL_MATERIALIZER_BLOCKED"
    elif obs_pass == 0:
        classification = "TRANSPARENT_MAR_SIGNAL_QUALITY_PARTIAL_SIGNAL_LOW"
    else:
        classification = "TRANSPARENT_MAR_SIGNAL_QUALITY_PARTIAL_WITH_PROVEN_LIMITING_FACTOR"
    pass_criteria = {
        "threshold_source_run_calibrated": threshold_source_lock,
        "at_least_one_obs_pass": obs_pass > 0,
        "full_4plane_available": full_4plane,
        "materializer_exercisable": materializer_exercisable,
        "no_pillar_2_violation": pillar2_clean,
    }
    failed_criteria = [k for k, v in pass_criteria.items() if not v]
    if obs_pass == 0:
        if not full_4plane:
            limiting_factor = "signal_below_floor_due_to_missing_SO3_planes_or_weak_perturbation"
        elif not threshold_source_lock:
            limiting_factor = "signal_below_floor_due_to_threshold_miscalibration"
        else:
            limiting_factor = "signal_below_floor_due_to_weak_perturbation_thermal_equilibrium_does_not_cross_run_calibrated_floor"
    else:
        limiting_factor = None
    out = {
        "schema_version": 1,
        "schema_kind": "m1_2_25_audit_verdict",
        "classification": classification,
        "pass_criteria_evaluated": pass_criteria,
        "failed_criteria": failed_criteria,
        "limiting_factor": limiting_factor,
        "summary": {
            "n_events": sp.get("n_events", 0),
            "max_kl": sp.get("max_kl_divergence", 0.0),
            "obs_threshold": t7.get("observation_threshold"),
            "disc_threshold": t7.get("discovery_threshold"),
            "obs_pass_count": obs_pass,
            "disc_pass_count": disc_pass,
            "threshold_source": t7.get("threshold_source"),
            "geometry_nz": mar.get("geometry_nonzero", 0),
            "causality_nz": mar.get("causality_nonzero", 0),
            "thermodynamics_nz": mar.get("thermodynamics_nonzero", 0),
            "chemistry_nz": mar.get("chemistry_nonzero", 0),
            "gear_transitions": gt.get("transition_count", 0),
            "materializer_n_candidates": mz.get("with_ghost_n_candidates", 0),
            "materializer_status": mz.get("ghost_factor_status"),
        },
    }
    W(ad / "audit_verdict.json", out)
    return out


def main():
    if len(sys.argv) < 4:
        print("usage: m1_2_25_audit_driver.py <run_dir> <audit_dir> <topology_stem>", file=sys.stderr)
        sys.exit(2)
    rd = Path(sys.argv[1])
    ad = Path(sys.argv[2])
    target = sys.argv[3]
    ad.mkdir(parents=True, exist_ok=True)
    print(f"--- M1.2.25 audit driver ---", file=sys.stderr)
    print(f"run_dir: {rd}", file=sys.stderr)
    print(f"audit_dir: {ad}", file=sys.stderr)
    t7 = section_1_t7(rd, ad, target);                                    print("§1 T7 done", file=sys.stderr)
    sp = section_2_signal_purity(rd, ad, t7);                             print("§2 signal purity done", file=sys.stderr)
    mar = section_3_mar_planes(rd, ad);                                   print("§3 MAR planes done", file=sys.stderr)
    gt = section_4_gear_trace(rd, ad);                                    print("§4 gear trace done", file=sys.stderr)
    mc = section_5_monomer_contrast(rd, ad, t7, mar);                     print("§5 monomer C done", file=sys.stderr)
    fc = section_6_flux_coupling(rd, ad, t7);                             print("§6 flux coupling done", file=sys.stderr)
    bp = section_7_branch(rd, ad);                                        print("§7 branch liveness done", file=sys.stderr)
    mz = section_8_materializer(rd, ad);                                  print("§8 materializer done", file=sys.stderr)
    csrs = section_9_csrs(rd, ad, t7, sp, mar, gt, mc, fc, bp, mz);       print("§9 CSR-S done", file=sys.stderr)
    verdict = section_10_verdict(ad, t7, sp, mar, gt, mc, fc, bp, mz, csrs)
    print(f"§10 verdict: {verdict['classification']}", file=sys.stderr)
    print(json.dumps({k: verdict.get(k) for k in ("classification", "limiting_factor", "summary")}, indent=2))


if __name__ == "__main__":
    main()
