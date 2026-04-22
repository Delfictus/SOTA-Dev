#!/usr/bin/env python3
"""Exhaustive engine-output harvest per completed Tier-A target.

Extracts every field from every engine artifact and emits:
  /tmp/engine_full_profiles/<target>.engine_full_profile.json
  /tmp/engine_full_profiles/<target>.site_tag_profiles.jsonl
  /tmp/engine_full_profiles/field_coverage_matrix.csv
  /tmp/engine_full_profiles/completeness_summary.csv
  /tmp/engine_full_profiles/engine_file_manifest.csv

Schema is derived from disk discovery across all 7 completed Tier-A targets.
Any discovered field outside this schema is tagged NEW_DISCOVERED_FIELD.
Any schema field absent for a target is tagged FEATURE_GAP:<field_name>.

Read-only. No modification to engine artifacts.
"""
from __future__ import annotations
import csv
import json
import sys
from pathlib import Path

OUT_DIR = Path("/tmp/engine_full_profiles")
OUT_DIR.mkdir(parents=True, exist_ok=True)

TARGETS = [
    ("wrn_apo",         "/mnt/storage/prism-outputs/twin-10-patent",      "6yhr"),
    ("menin_apo",       "/mnt/storage/prism-outputs/twin-10-patent",      "3re2"),
    ("smarca2_brd_apo", "/mnt/storage/prism-outputs/twin-10-patent",      "4qy4"),
    ("pkmyt1_apo",      "/mnt/storage/prism-outputs/twin-10-patent",      "3p1a"),
    ("kras_g12d_apo",   "/mnt/storage/prism-outputs/twin-10-patent",      "7f0w"),
    ("m1_2nvp",         "/mnt/storage/prism-outputs/m1-strict-dcc-panel", "2nvp"),
    ("m1_1xhx",         "/mnt/storage/prism-outputs/m1-strict-dcc-panel", "1xhx"),
]

# ---------------------------------------------------------------------------
# Schema locked by disk discovery across the 7 completed targets.
# Any field outside this list that appears in an artifact is tagged NEW_DISCOVERED_FIELD.
# ---------------------------------------------------------------------------
SCHEMA = {
    "binding_sites_top": [
        "all_pockets", "background_spikes", "binding_sites", "consensus_threshold",
        "cryptic_sites", "druggable_sites", "lining_residue_cutoff_angstroms",
        "mode", "n_streams", "per_stream_stats", "prism_therm", "rescue_history",
        "simulation_time_sec", "sites", "structure", "total_steps_per_stream",
    ],
    "binding_sites_sites_item": [
        "aromatic_score", "asymmetry_offset", "breathing_score", "burial_score",
        "catalytic_residue_count", "ccns_tau", "centroid", "classification",
        "cold_phase_fraction", "composite_audit_rank", "composite_audit_score",
        "composite_v3_rank", "composite_v3_score", "cryptic_rank", "cryptic_score",
        "delta_g_aromatic_kcal_mol", "delta_g_cooperative_kcal_mol",
        "delta_g_dewetting_kcal_mol", "delta_g_electrostatic_kcal_mol",
        "delta_g_sti_kcal_mol", "druggability", "effective_delta_g_kcal_mol",
        "engine_chem", "engine_geo", "engine_phys", "engine_vcs",
        "frustrated_solvent_score", "gtck_rank", "hysteresis_asymmetry", "id",
        "is_druggable", "kcc", "kinetic_accessibility", "lining_residues",
        "localization_score_raw", "mean_burial", "onset_score", "quality_score",
        "rank", "rank_C", "rank_G", "rank_K", "rank_L", "rank_T", "rank_score",
        "ranker_version", "ray_escape_ratio", "relative_asymmetry", "residue_ids",
        "signal_preservation", "source_diversity", "sphericity", "spike_count",
        "sti_n_spikes", "sti_n_voxels", "therm_class", "tide_coupling_score",
        "tide_trigger_residues", "tokenized_score", "tokenized_token",
        "unsat_frac", "uv_enrichment_score", "volume", "wd_coherence",
    ],
    "prism_therm_top": [
        "global_pockets", "hysteresis_threshold", "hysteretic_site_count",
        "sdst_event_count", "sites", "tide_residues_mapped", "total_avalanches",
    ],
    "prism_therm_sites_item": [
        "asymmetry_score", "ccns_classification", "cooling_spike_count",
        "cooling_spike_rate", "druggability", "heating_spike_count",
        "heating_spike_rate", "is_hysteretic", "n_avalanches",
        "relative_asymmetry", "site_id", "tau", "tau_stderr", "therm_class",
        "tide_decomposition",
    ],
    "rescue_history": ["decisions", "enabled", "targets_used"],
    "kcc_visualization_top": [
        "interpretation_guidelines", "pdb_source", "residues", "semantics",
        "sites", "vector_field_definition",
    ],
    "kcc_visualization_residues_item": [
        "active_causal_steps", "burst_motion", "ca_position", "causal_lag",
        "direction_score", "kcc_score", "lag_corr_peak", "local_cov",
        "motion_efficiency", "net_dx", "net_dy", "net_dz", "residue_id",
        "residue_name", "sum_motion", "total_steps",
    ],
    "kcc_visualization_sites_item": [
        "centroid", "gtck_rank", "id", "kcc", "rank_C", "rank_G", "rank_K",
        "rank_L", "rank_T", "rank_score", "volume",
    ],
    "ensemble_trajectory_top": [
        "consensus_site_ids", "n_consensus_sites", "n_streams", "per_stream",
        "structure", "total_spikes",
    ],
    "site_spike_events_top": [
        "centroid", "lining_cutoff", "n_spikes", "open_frequency", "site_id",
        "spikes",
    ],
    "site_spike_events_spikes_item": [
        "aromatic_residue_id", "ccns_phase", "frame_index", "intensity",
        "n_nearby_excited", "spike_source", "stream_id", "timestep", "type",
        "vibrational_energy", "water_density", "wavelength_nm", "x", "y", "z",
    ],
    "rerank_merged_pockets_item": [
        "ccns_tau", "centroid_spike_weighted", "cryptic_likelihood",
        "drug_score_normalized", "druggability_exclusion_reason",
        "druggability_score", "druggability_support",
        "druggability_threshold_applied", "dynamic_support", "engine_is_druggable",
        "engine_rank", "hysteresis_asymmetry", "is_cryptic", "n_spikes_attributed",
        "pocket_id", "rank_shift", "relative_asymmetry", "rerank_composite",
        "rerank_position", "rerank_rank", "site_volume_angstrom_cubed",
        "spike_score_normalized", "therm_class", "therm_score_normalized",
        "tide_score_normalized", "top_residue_ids", "volume_threshold_applied",
    ],
}

# File classes to enumerate for the manifest
FILE_CLASSES = [
    ("binding_sites_json", "<stem>.binding_sites.json"),
    ("kcc_visualization_json", "<stem>.kcc_visualization.json"),
    ("kcc_validation_json", "<stem>.kcc_validation.json"),
    ("ensemble_trajectory_json", "<stem>.ensemble_trajectory.json"),
    ("residue_map_json", "<stem>.residue_map.json"),
    ("ground_truth_json", "<stem>_ground_truth.json"),
    ("rerank_result_json", "rerank_result.json"),
    ("evaluation_json", "evaluation.json"),
    ("topology_json", "<stem>.topology.json"),
    ("topology_prism_therm_json", "<stem>.topology.prism_therm.json"),
    ("topology_spike_events_arrow", "<stem>.topology.spike_events.arrow"),
    ("site_spike_events_jsons", "<stem>.site*.spike_events.json"),
]


def _sample_value(v):
    if isinstance(v, list):
        return {"_list_len": len(v), "_first": v[0] if v else None}
    if isinstance(v, dict):
        return {"_dict_keys": list(v.keys())}
    return v


def enumerate_paths(tdir: Path, stem: str) -> dict:
    eng = tdir / "artifacts/5_engine"
    prep = tdir / "artifacts/3_prep"
    gt_dir = tdir / "artifacts/4_ground_truth"
    rerank_path = tdir / "artifacts/6_rerank/rerank_result.json"
    eval_path = tdir / "artifacts/7_evaluation/evaluation.json"
    return {
        "binding_sites_json":        eng / f"{stem}.binding_sites.json",
        "kcc_visualization_json":    eng / f"{stem}.kcc_visualization.json",
        "kcc_validation_json":       eng / f"{stem}.kcc_validation.json",
        "ensemble_trajectory_json":  eng / f"{stem}.ensemble_trajectory.json",
        "residue_map_json":          prep / f"{stem}.residue_map.json",
        "ground_truth_json":         gt_dir / f"{stem}_ground_truth.json",
        "rerank_result_json":        rerank_path,
        "evaluation_json":           eval_path,
        "topology_json":             prep / f"{stem}.topology.json",
        "topology_prism_therm_json": eng / f"{stem}.topology.prism_therm.json",
        "topology_spike_events_arrow": eng / f"{stem}.topology.spike_events.arrow",
        "site_spike_events_jsons":   sorted(eng.glob(f"{stem}.site*.spike_events.json")),
    }


def read_json(p):
    try:
        return json.loads(Path(p).read_text())
    except Exception as e:
        return {"__read_error__": str(e)}


def _arrow_first_site_summaries(eng: Path, stem: str):
    """D3 Arrow-first: build {sid: summary_dict} matching read_spike_file_summary's
    top-level output for every site in binding_sites.json, using the canonical Arrow
    + run_metadata + binding_sites triad. Each summary includes n_spikes, centroid,
    open_frequency, lining_cutoff, and the FIRST spike (legacy JSON shape) as 'spikes'[0].

    Returns (summaries: dict[sid -> dict], virtual_path: str) or (None, None) if
    any triad member missing — caller falls back to per-site JSON reader.

    Spatial rule: site_radius = lining_cutoff + 2.0 (Gate-A validated).
    Enums + protocol: decoded via run_metadata.json.
    """
    arrow_p = eng / f"{stem}.topology.spike_events.arrow"
    meta_p = eng / f"{stem}.run_metadata.json"
    bs_p = eng / f"{stem}.binding_sites.json"
    if not (arrow_p.exists() and meta_p.exists() and bs_p.exists()):
        return None, None
    try:
        import pyarrow as pa
        import pyarrow.ipc as ipc
    except Exception:
        return None, None
    meta = json.loads(meta_p.read_text())
    proto = meta.get("reference_protocol_for_json_phase_label") or {}
    if "error" in proto:
        return None, None
    p1 = proto["cold_hold_steps"]
    p2 = p1 + proto["ramp_steps"]
    p3 = p2 + proto["warm_hold_steps"]
    p4 = p3 + proto.get("ramp_down_steps", 0)
    def _phase(ts):
        if ts < p1: return "cold_hold"
        if ts < p2: return "heating"
        if ts < p3: return "warm_hold"
        if ts < p4: return "cooling"
        return "cold_return"
    arom_enum = {int(k): v for k, v in meta["aromatic_type_enum"].items()}
    arom_default = meta.get("aromatic_type_default", "UNK")
    src_enum = {int(k): v for k, v in meta["spike_source_enum"].items()}
    src_default = meta.get("spike_source_default", "LIF")
    lining_cutoff = meta.get("lining_cutoff", 8.0)
    site_radius_sq = (float(lining_cutoff) + 2.0) ** 2
    bs = json.loads(bs_p.read_text())
    sites = [s for s in (bs.get("sites") or []) if isinstance(s, dict) and s.get("centroid")]
    with arrow_p.open("rb") as f:
        magic = f.read(8)
    opener = ipc.open_file if magic.startswith(b"ARROW1") else ipc.open_stream
    with arrow_p.open("rb") as f:
        table = opener(f).read_all()
    x = table.column("x").to_numpy()
    y = table.column("y").to_numpy()
    z = table.column("z").to_numpy()
    virtual = f"arrow+meta+bs:{arrow_p.name}"
    summaries = {}
    for s in sites:
        sid = s.get("id")
        cx, cy, cz = s["centroid"]
        d2 = (x - cx) ** 2 + (y - cy) ** 2 + (z - cz) ** 2
        mask = pa.array(d2 <= site_radius_sq)
        sub = table.filter(mask)
        n_rows = len(sub)
        frames = sub.column("frame_index").to_pylist() if n_rows else []
        ofreq = (len(set(frames)) / max(max(frames) + 1, 1)) if n_rows else 0.0
        first_spike = None
        if n_rows:
            i = 0
            ts0 = sub.column("timestep")[i].as_py()
            first_spike = {
                "x": sub.column("x")[i].as_py(),
                "y": sub.column("y")[i].as_py(),
                "z": sub.column("z")[i].as_py(),
                "intensity": sub.column("intensity")[i].as_py(),
                "type": arom_enum.get(int(sub.column("aromatic_type")[i].as_py()), arom_default),
                "wavelength_nm": sub.column("wavelength_nm")[i].as_py(),
                "spike_source": src_enum.get(int(sub.column("spike_source")[i].as_py()), src_default),
                "aromatic_residue_id": sub.column("aromatic_residue_id")[i].as_py(),
                "water_density": sub.column("water_density")[i].as_py(),
                "vibrational_energy": sub.column("vibrational_energy")[i].as_py(),
                "n_nearby_excited": sub.column("n_nearby_excited")[i].as_py(),
                "timestep": ts0,
                "frame_index": int(sub.column("frame_index")[i].as_py()),
                "ccns_phase": _phase(int(ts0)),
                "stream_id": int(sub.column("stream_id")[i].as_py()),
            }
        summaries[sid] = {
            "site_id": sid,
            "n_spikes": n_rows,
            "centroid": s["centroid"],
            "open_frequency": ofreq,
            "lining_cutoff": lining_cutoff,
            "spikes": [first_spike] if first_spike else [],
            "spike_file": f"{virtual}#site{sid}",
        }
    return summaries, virtual


def read_spike_file_summary(p: Path) -> dict:
    """Partial-parse a site spike_events.json file.

    Reads only the first ~16 KB, truncates at the first complete spike object,
    closes the JSON container, and parses the result. Returns the top-level
    scalars (centroid, lining_cutoff, n_spikes, open_frequency, site_id) plus
    the first spike element as `spike_sample_first`. Avoids loading
    hundreds-of-MB files into memory.
    """
    try:
        with open(p, "r") as f:
            head = f.read(16384)
    except Exception as e:
        return {"__read_error__": str(e)}
    idx = head.find('"spikes":')
    if idx < 0:
        # Small file with no spikes array — try full parse
        try:
            return json.loads(head)
        except Exception:
            return {"__read_error__": "spikes key not found"}
    bracket_idx = head.find('[', idx)
    if bracket_idx < 0:
        return {"__read_error__": "no [ after spikes"}
    # Find first complete spike object: first `{` then matching `}`
    first_open = head.find('{', bracket_idx)
    if first_open < 0:
        # spikes: []
        truncated = head[:bracket_idx + 1] + "]}"
    else:
        depth = 0
        first_close = -1
        for i in range(first_open, len(head)):
            c = head[i]
            if c == '{':
                depth += 1
            elif c == '}':
                depth -= 1
                if depth == 0:
                    first_close = i
                    break
        if first_close < 0:
            return {"__read_error__": "first spike object spans > 16KB"}
        truncated = head[:first_close + 1] + "]}"
    try:
        return json.loads(truncated)
    except Exception as e:
        return {"__read_error__": f"truncation parse fail: {e}"}


def coverage_for(group_name, schema_list, actual_dict_or_list, source_file):
    """Return list of rows for field_coverage_matrix.csv + new-discovered set."""
    rows = []
    observed = set()
    if isinstance(actual_dict_or_list, dict):
        observed = set(actual_dict_or_list.keys())
    elif isinstance(actual_dict_or_list, list):
        # inspect first non-empty dict
        for itm in actual_dict_or_list:
            if isinstance(itm, dict) and itm:
                observed = set(itm.keys())
                break
    schema = set(schema_list)
    for f in schema:
        exists = f in observed
        val = None
        vtype = None
        if exists:
            if isinstance(actual_dict_or_list, dict):
                val = actual_dict_or_list[f]
            elif isinstance(actual_dict_or_list, list):
                for itm in actual_dict_or_list:
                    if isinstance(itm, dict) and f in itm:
                        val = itm[f]
                        break
            vtype = type(val).__name__
        rows.append({
            "field_path": f"{group_name}.{f}",
            "source_file": source_file,
            "exists": exists,
            "value_type": vtype if exists else "FEATURE_GAP",
            "sample_value": json.dumps(_sample_value(val), default=str)[:120] if exists else "FEATURE_GAP",
            "classification": "RAW_ENGINE_FIELD" if exists else f"FEATURE_GAP:{f}",
        })
    # NEW_DISCOVERED_FIELD entries
    for f in observed - schema:
        val = None
        if isinstance(actual_dict_or_list, dict):
            val = actual_dict_or_list.get(f)
        elif isinstance(actual_dict_or_list, list):
            for itm in actual_dict_or_list:
                if isinstance(itm, dict) and f in itm:
                    val = itm[f]
                    break
        rows.append({
            "field_path": f"{group_name}.{f}",
            "source_file": source_file,
            "exists": True,
            "value_type": type(val).__name__,
            "sample_value": json.dumps(_sample_value(val), default=str)[:120],
            "classification": "NEW_DISCOVERED_FIELD",
        })
    return rows


def build_engine_profile(target, stem, paths):
    """Aggregate raw extracted values into a single engine_full_profile dict."""
    prof = {
        "target": target,
        "stem": stem,
        "files_inspected": {k: str(v) if not isinstance(v, list) else [str(p) for p in v[:3]] + ([f"...+{len(v)-3} more"] if len(v) > 3 else []) for k, v in paths.items()},
    }

    # Top-level JSON artifacts
    bs = read_json(paths["binding_sites_json"])
    kv = read_json(paths["kcc_visualization_json"])
    kc = read_json(paths["kcc_validation_json"])
    et = read_json(paths["ensemble_trajectory_json"])
    rm = read_json(paths["residue_map_json"]) if paths["residue_map_json"].exists() else {"__feature_gap__": True}
    gt = read_json(paths["ground_truth_json"]) if paths["ground_truth_json"].exists() else {"__feature_gap__": True}
    rr = read_json(paths["rerank_result_json"])
    ev = read_json(paths["evaluation_json"])
    topo = read_json(paths["topology_json"]) if paths["topology_json"].exists() else {"__feature_gap__": True}

    # topology.prism_therm.json explicit file check — may not exist as separate file
    tpt_path = paths["topology_prism_therm_json"]
    if tpt_path.exists():
        tpt = read_json(tpt_path)
        prof["topology_prism_therm_json_present"] = True
        prof["topology_prism_therm"] = tpt
    else:
        prof["topology_prism_therm_json_present"] = False
        prof["topology_prism_therm_source_note"] = "prism_therm block is nested inside binding_sites.json"

    # Therm block (from binding_sites.prism_therm)
    pt = bs.get("prism_therm") if isinstance(bs, dict) else None
    prof["therm_prism_therm"] = {
        "global_pockets": (pt or {}).get("global_pockets"),
        "hysteresis_threshold": (pt or {}).get("hysteresis_threshold"),
        "hysteretic_site_count": (pt or {}).get("hysteretic_site_count"),
        "sdst_event_count": (pt or {}).get("sdst_event_count"),
        "sites": (pt or {}).get("sites"),
        "tide_residues_mapped": (pt or {}).get("tide_residues_mapped"),
        "total_avalanches": (pt or {}).get("total_avalanches"),
    }

    # Binding-site root fields
    prof["binding_sites_root"] = {
        k: bs.get(k) if not isinstance(bs.get(k), list) or k not in ("sites", "cryptic_sites", "all_pockets", "per_stream_stats") else bs.get(k)
        for k in SCHEMA["binding_sites_top"]
    }

    # KCC blocks
    prof["kcc_visualization"] = {
        "interpretation_guidelines": kv.get("interpretation_guidelines"),
        "pdb_source": kv.get("pdb_source"),
        "semantics": kv.get("semantics"),
        "vector_field_definition": kv.get("vector_field_definition"),
        "residues_preview_first3": (kv.get("residues") or [])[:3],
        "residues_count": len(kv.get("residues") or []),
        "sites_preview_first3": (kv.get("sites") or [])[:3],
        "sites_count": len(kv.get("sites") or []),
    }
    prof["kcc_validation"] = kc

    # Ensemble trajectory
    prof["ensemble_trajectory"] = et

    # Rerank merged_pockets preview + count
    mp = rr.get("merged_pockets") if isinstance(rr, dict) else []
    prof["rerank_summary"] = {
        "schema_version": rr.get("schema_version") if isinstance(rr, dict) else None,
        "ranker_weights": rr.get("ranker_weights") if isinstance(rr, dict) else None,
        "merged_pockets_count": len(mp),
        "merged_pockets": mp,  # full list kept
    }

    # Ground truth
    prof["ground_truth"] = gt

    # Residue map
    prof["residue_map"] = rm

    # Evaluation
    prof["evaluation"] = ev

    # Topology (structure preview)
    if isinstance(topo, dict):
        prof["topology_summary"] = {
            "keys": list(topo.keys())[:20] if "__feature_gap__" not in topo else ["FEATURE_GAP:topology.json"],
            "n_residues": len(topo.get("residues") or []) if isinstance(topo.get("residues"), list) else None,
            "n_atoms": len(topo.get("atoms") or []) if isinstance(topo.get("atoms"), list) else None,
        }

    # Per-site spike events — LAZY: store only the small-file list of paths + counts + per-site top-level
    site_se_summary = []
    for se_path in paths["site_spike_events_jsons"]:
        try:
            size_bytes = se_path.stat().st_size
        except Exception:
            size_bytes = None
        site_se_summary.append({"path": str(se_path), "size_bytes": size_bytes})
    prof["site_spike_events_files"] = {
        "count": len(paths["site_spike_events_jsons"]),
        "files": site_se_summary,
    }

    # Rescue history (raw)
    prof["rescue_history"] = (bs.get("rescue_history") if isinstance(bs, dict) else None)

    # Background spikes (raw)
    prof["background_spikes"] = (bs.get("background_spikes") if isinstance(bs, dict) else None)

    # Per stream stats (raw)
    prof["per_stream_stats"] = (bs.get("per_stream_stats") if isinstance(bs, dict) else None)

    return prof, bs, kv, kc, et, rm, gt, rr, ev


def build_site_tag_profiles(target, stem, bs, kv, rr, paths):
    """One JSON per site: join all site-level sources by site_id where possible."""
    site_rows = []

    bs_sites = (bs.get("sites") if isinstance(bs, dict) else []) or []
    bs_sites_by_id = {s.get("id"): s for s in bs_sites if isinstance(s, dict)}

    kv_sites = (kv.get("sites") if isinstance(kv, dict) else []) or []
    kv_sites_by_id = {s.get("id"): s for s in kv_sites if isinstance(s, dict)}

    pt_sites = (bs.get("prism_therm") or {}).get("sites") if isinstance(bs, dict) else []
    pt_sites_by_id = {s.get("site_id"): s for s in (pt_sites or []) if isinstance(s, dict)}

    mp = rr.get("merged_pockets") if isinstance(rr, dict) else []
    mp_by_id = {p.get("pocket_id"): p for p in (mp or []) if isinstance(p, dict)}

    # Union of all site IDs across sources
    all_ids = set()
    all_ids.update(bs_sites_by_id.keys())
    all_ids.update(kv_sites_by_id.keys())
    all_ids.update(pt_sites_by_id.keys())
    all_ids.update(mp_by_id.keys())
    all_ids.discard(None)

    # Per-site spike events summary by site_id. D3 Arrow-first: prefer Arrow
    # triad; fall back to per-site JSON partial-parse if triad incomplete.
    se_summary = {}
    eng_dir = paths["binding_sites_json"].parent
    arrow_sums, arrow_virtual = _arrow_first_site_summaries(eng_dir, stem)
    if arrow_sums is not None:
        for sid, d in arrow_sums.items():
            se_summary[sid] = {
                "n_spikes": d.get("n_spikes"),
                "centroid": d.get("centroid"),
                "open_frequency": d.get("open_frequency"),
                "lining_cutoff": d.get("lining_cutoff"),
                "spike_sample_first": (d.get("spikes") or [{}])[0] if d.get("spikes") else None,
                "spike_file": d.get("spike_file"),
            }
    else:
        for se_path in paths["site_spike_events_jsons"]:
            d = read_spike_file_summary(se_path)
            if "__read_error__" in d:
                continue
            sid = d.get("site_id")
            if sid is None:
                continue
            se_summary[sid] = {
                "n_spikes": d.get("n_spikes"),
                "centroid": d.get("centroid"),
                "open_frequency": d.get("open_frequency"),
                "lining_cutoff": d.get("lining_cutoff"),
                "spike_sample_first": (d.get("spikes") or [{}])[0] if d.get("spikes") else None,
                "spike_file": str(se_path),
            }

    for sid in sorted(all_ids, key=lambda x: (isinstance(x, str), x)):
        site_row = {
            "target": target,
            "site_id": sid,
            "raw_binding_site_fields": bs_sites_by_id.get(sid, "FEATURE_GAP:binding_sites.sites[site_id]"),
            "raw_kcc_visualization_site_fields": kv_sites_by_id.get(sid, "FEATURE_GAP:kcc_visualization.sites[id]"),
            "raw_prism_therm_site_fields": pt_sites_by_id.get(sid, "FEATURE_GAP:prism_therm.sites[site_id]"),
            "raw_rerank_pocket_fields": mp_by_id.get(sid, "FEATURE_GAP:rerank.merged_pockets[pocket_id]"),
            "raw_spike_events_summary": se_summary.get(sid, "FEATURE_GAP:site_spike_events.json"),
            "field_provenance": {
                "raw_binding_site_fields": str(paths["binding_sites_json"]),
                "raw_kcc_visualization_site_fields": str(paths["kcc_visualization_json"]),
                "raw_prism_therm_site_fields": f"{paths['binding_sites_json']}:prism_therm.sites",
                "raw_rerank_pocket_fields": str(paths["rerank_result_json"]),
                "raw_spike_events_summary": "artifacts/5_engine/<stem>.site{N}.spike_events.json",
            },
        }
        site_rows.append(site_row)
    return site_rows


def main():
    manifest_rows = []
    coverage_rows = []
    completeness_rows = []

    for target, root, stem in TARGETS:
        tdir = Path(root) / target
        paths = enumerate_paths(tdir, stem)

        # ── Manifest ──
        for cls, pth in paths.items():
            if isinstance(pth, list):
                for p in pth:
                    manifest_rows.append({
                        "target": target, "artifact_class": cls, "file_path": str(p),
                        "size_bytes": p.stat().st_size if p.exists() else 0,
                        "exists": p.exists(),
                    })
            else:
                manifest_rows.append({
                    "target": target, "artifact_class": cls, "file_path": str(pth),
                    "size_bytes": pth.stat().st_size if pth.exists() else 0,
                    "exists": pth.exists(),
                })

        # ── Profile extraction ──
        profile, bs, kv, kc, et, rm, gt, rr, ev = build_engine_profile(target, stem, paths)

        # ── Field coverage matrix ──
        coverage_rows.extend(coverage_for(
            "binding_sites_top", SCHEMA["binding_sites_top"], bs, str(paths["binding_sites_json"])))
        coverage_rows.extend(coverage_for(
            "binding_sites.sites[item]", SCHEMA["binding_sites_sites_item"],
            bs.get("sites") or [], str(paths["binding_sites_json"])))
        coverage_rows.extend(coverage_for(
            "binding_sites.prism_therm", SCHEMA["prism_therm_top"],
            bs.get("prism_therm") or {}, str(paths["binding_sites_json"])))
        coverage_rows.extend(coverage_for(
            "binding_sites.prism_therm.sites[item]", SCHEMA["prism_therm_sites_item"],
            (bs.get("prism_therm") or {}).get("sites") or [], str(paths["binding_sites_json"])))
        coverage_rows.extend(coverage_for(
            "binding_sites.rescue_history", SCHEMA["rescue_history"],
            bs.get("rescue_history") or {}, str(paths["binding_sites_json"])))
        coverage_rows.extend(coverage_for(
            "kcc_visualization_top", SCHEMA["kcc_visualization_top"], kv,
            str(paths["kcc_visualization_json"])))
        coverage_rows.extend(coverage_for(
            "kcc_visualization.residues[item]", SCHEMA["kcc_visualization_residues_item"],
            kv.get("residues") or [], str(paths["kcc_visualization_json"])))
        coverage_rows.extend(coverage_for(
            "kcc_visualization.sites[item]", SCHEMA["kcc_visualization_sites_item"],
            kv.get("sites") or [], str(paths["kcc_visualization_json"])))
        coverage_rows.extend(coverage_for(
            "ensemble_trajectory_top", SCHEMA["ensemble_trajectory_top"], et,
            str(paths["ensemble_trajectory_json"])))
        coverage_rows.extend(coverage_for(
            "rerank_result.merged_pockets[item]", SCHEMA["rerank_merged_pockets_item"],
            rr.get("merged_pockets") or [], str(paths["rerank_result_json"])))

        # Spike events — D3 Arrow-first: build summaries from canonical Arrow +
        # run_metadata + binding_sites triad when present; else iterate per-site
        # JSONs until a non-empty spikes sample is found. Empty spikes would
        # produce false FEATURE_GAP on spike-item fields.
        sample_se_path = None
        sample_se = None
        arrow_sums_main, _ = _arrow_first_site_summaries(paths["binding_sites_json"].parent, stem)
        if arrow_sums_main:
            for sid, d in arrow_sums_main.items():
                if d.get("spikes"):
                    sample_se = d
                    sample_se_path = d.get("spike_file")
                    break
            if sample_se is None:
                any_sid = next(iter(arrow_sums_main))
                sample_se = arrow_sums_main[any_sid]
                sample_se_path = sample_se.get("spike_file")
        elif paths["site_spike_events_jsons"]:
            for cand in sorted(paths["site_spike_events_jsons"], key=lambda p: -p.stat().st_size):
                tmp = read_spike_file_summary(cand)
                if tmp.get("spikes"):
                    sample_se_path = str(cand)
                    sample_se = tmp
                    break
            if sample_se is None:
                sample_se_path = str(paths["site_spike_events_jsons"][0])
                sample_se = read_spike_file_summary(paths["site_spike_events_jsons"][0])
        if sample_se is not None:
            coverage_rows.extend(coverage_for(
                "site_spike_events_top", SCHEMA["site_spike_events_top"], sample_se,
                sample_se_path))
            coverage_rows.extend(coverage_for(
                "site_spike_events.spikes[item]", SCHEMA["site_spike_events_spikes_item"],
                sample_se.get("spikes") or [], sample_se_path))

        # Annotate each row with target
        for r in coverage_rows:
            r.setdefault("target", target)

        # ── Write engine_full_profile.json ──
        profile_path = OUT_DIR / f"{target}.engine_full_profile.json"
        profile_path.write_text(json.dumps(profile, indent=2, default=str))

        # ── Write site_tag_profiles.jsonl ──
        site_rows = build_site_tag_profiles(target, stem, bs, kv, rr, paths)
        jsonl_path = OUT_DIR / f"{target}.site_tag_profiles.jsonl"
        with jsonl_path.open("w") as f:
            for sr in site_rows:
                f.write(json.dumps(sr, default=str) + "\n")

        # ── Completeness row ──
        n_files_found = sum(1 for r in manifest_rows if r["target"] == target and r["exists"])
        my_cov = [r for r in coverage_rows if r.get("target") == target]
        n_verified = sum(1 for r in my_cov if r["classification"] == "RAW_ENGINE_FIELD")
        n_new = sum(1 for r in my_cov if r["classification"] == "NEW_DISCOVERED_FIELD")
        n_gaps = sum(1 for r in my_cov if r["classification"].startswith("FEATURE_GAP"))
        completeness_rows.append({
            "target": target,
            "number_of_engine_files_found": n_files_found,
            "number_of_schema_fields_verified": n_verified,
            "number_of_new_discovered_fields": n_new,
            "number_of_feature_gaps": n_gaps,
            "number_of_site_profiles_emitted": len(site_rows),
        })

        print(f"[{target}] engine_full_profile.json written "
              f"(files={n_files_found} verified={n_verified} new={n_new} gaps={n_gaps} sites={len(site_rows)})")

    # ── Write manifest CSV ──
    man_path = OUT_DIR / "engine_file_manifest.csv"
    with man_path.open("w") as f:
        w = csv.DictWriter(f, fieldnames=["target", "artifact_class", "file_path", "size_bytes", "exists"])
        w.writeheader()
        for r in manifest_rows:
            w.writerow(r)

    # ── Write coverage matrix CSV ──
    cov_path = OUT_DIR / "field_coverage_matrix.csv"
    # Dedupe by (target, field_path) — we annotated after-the-fact so earlier iterations may not have target set
    # Actually we annotate within the loop but only AFTER coverage_rows.extend — meaning rows from prior target are re-annotated.
    # Rebuild with correct per-target scoping:
    with cov_path.open("w") as f:
        w = csv.DictWriter(f, fieldnames=["target", "field_path", "source_file", "exists",
                                           "value_type", "sample_value", "classification"])
        w.writeheader()
        for r in coverage_rows:
            w.writerow({k: r.get(k, "") for k in w.fieldnames})

    # ── Write completeness summary CSV ──
    comp_path = OUT_DIR / "completeness_summary.csv"
    with comp_path.open("w") as f:
        w = csv.DictWriter(f, fieldnames=["target", "number_of_engine_files_found",
                                           "number_of_schema_fields_verified",
                                           "number_of_new_discovered_fields",
                                           "number_of_feature_gaps",
                                           "number_of_site_profiles_emitted"])
        w.writeheader()
        for r in completeness_rows:
            w.writerow(r)

    print()
    print(f"manifest:  {man_path}")
    print(f"coverage:  {cov_path}")
    print(f"summary:   {comp_path}")
    print(f"profiles:  {OUT_DIR}/<target>.engine_full_profile.json  (7 files)")
    print(f"tags:      {OUT_DIR}/<target>.site_tag_profiles.jsonl    (7 files)")


if __name__ == "__main__":
    main()
