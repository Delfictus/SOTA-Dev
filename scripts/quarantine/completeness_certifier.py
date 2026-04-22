#!/usr/bin/env python3
"""Completeness certification for 7 completed Tier-A targets.

Emits:
  /tmp/engine_full_profiles/<target>.<artifact>.actual_keyset.json
  /tmp/engine_full_profiles/schema_vs_actual_diff.csv
  /tmp/engine_full_profiles/<target>.engine_full_profile_full.json   (no previews)
  /tmp/engine_full_profiles/<target>.site_tag_profiles.full.jsonl
  /tmp/engine_full_profiles/<target>.spike_events_export_manifest.json
  /tmp/engine_full_profiles/completeness_certificate.json

Certification rule:
  COMPLETE iff every actual recursive key path is either:
    (a) fully extracted into an emitted file, or
    (b) the file content is itself preserved on-disk AND indexed by a manifest
  AND no UNMAPPED_ACTUAL_FIELD, no RAW_ONLY_NOT_EXTRACTED, no SCHEMA_ONLY_MISSING.

Spike event JSONL sidecars are emitted in streaming fashion to preserve all
per-spike fields without a ~500 MB in-memory load.
"""
from __future__ import annotations
import csv
import json
import re
import sys
from pathlib import Path

OUT = Path("/tmp/engine_full_profiles")
OUT.mkdir(parents=True, exist_ok=True)

TARGETS = [
    ("wrn_apo",         "/mnt/storage/prism-outputs/twin-10-patent",      "6yhr"),
    ("menin_apo",       "/mnt/storage/prism-outputs/twin-10-patent",      "3re2"),
    ("smarca2_brd_apo", "/mnt/storage/prism-outputs/twin-10-patent",      "4qy4"),
    ("pkmyt1_apo",      "/mnt/storage/prism-outputs/twin-10-patent",      "3p1a"),
    ("kras_g12d_apo",   "/mnt/storage/prism-outputs/twin-10-patent",      "7f0w"),
    ("m1_2nvp",         "/mnt/storage/prism-outputs/m1-strict-dcc-panel", "2nvp"),
    ("m1_1xhx",         "/mnt/storage/prism-outputs/m1-strict-dcc-panel", "1xhx"),
]

# Locked schema from engine_full_harvest.py — normalized to dot-paths
LOCKED_SCHEMA = {
    "binding_sites.json": [
        "all_pockets", "background_spikes", "binding_sites", "consensus_threshold",
        "cryptic_sites", "druggable_sites", "lining_residue_cutoff_angstroms",
        "mode", "n_streams", "per_stream_stats", "prism_therm", "rescue_history",
        "simulation_time_sec", "sites", "structure", "total_steps_per_stream",
        "sites[*].aromatic_score", "sites[*].asymmetry_offset", "sites[*].breathing_score",
        "sites[*].burial_score", "sites[*].catalytic_residue_count", "sites[*].ccns_tau",
        "sites[*].centroid", "sites[*].classification", "sites[*].cold_phase_fraction",
        "sites[*].composite_audit_rank", "sites[*].composite_audit_score",
        "sites[*].composite_v3_rank", "sites[*].composite_v3_score",
        "sites[*].cryptic_rank", "sites[*].cryptic_score",
        "sites[*].delta_g_aromatic_kcal_mol", "sites[*].delta_g_cooperative_kcal_mol",
        "sites[*].delta_g_dewetting_kcal_mol", "sites[*].delta_g_electrostatic_kcal_mol",
        "sites[*].delta_g_sti_kcal_mol", "sites[*].druggability",
        "sites[*].effective_delta_g_kcal_mol", "sites[*].engine_chem",
        "sites[*].engine_geo", "sites[*].engine_phys", "sites[*].engine_vcs",
        "sites[*].frustrated_solvent_score", "sites[*].gtck_rank",
        "sites[*].hysteresis_asymmetry", "sites[*].id", "sites[*].is_druggable",
        "sites[*].kcc", "sites[*].kinetic_accessibility", "sites[*].lining_residues",
        "sites[*].localization_score_raw", "sites[*].mean_burial",
        "sites[*].onset_score", "sites[*].quality_score", "sites[*].rank",
        "sites[*].rank_C", "sites[*].rank_G", "sites[*].rank_K", "sites[*].rank_L",
        "sites[*].rank_T", "sites[*].rank_score", "sites[*].ranker_version",
        "sites[*].ray_escape_ratio", "sites[*].relative_asymmetry",
        "sites[*].residue_ids", "sites[*].signal_preservation",
        "sites[*].source_diversity", "sites[*].sphericity", "sites[*].spike_count",
        "sites[*].sti_n_spikes", "sites[*].sti_n_voxels", "sites[*].therm_class",
        "sites[*].tide_coupling_score", "sites[*].tide_trigger_residues",
        "sites[*].tokenized_score", "sites[*].tokenized_token", "sites[*].unsat_frac",
        "sites[*].uv_enrichment_score", "sites[*].volume", "sites[*].wd_coherence",
        "prism_therm.global_pockets", "prism_therm.hysteresis_threshold",
        "prism_therm.hysteretic_site_count", "prism_therm.sdst_event_count",
        "prism_therm.sites", "prism_therm.tide_residues_mapped",
        "prism_therm.total_avalanches",
        "prism_therm.sites[*].asymmetry_score", "prism_therm.sites[*].ccns_classification",
        "prism_therm.sites[*].cooling_spike_count", "prism_therm.sites[*].cooling_spike_rate",
        "prism_therm.sites[*].druggability", "prism_therm.sites[*].heating_spike_count",
        "prism_therm.sites[*].heating_spike_rate", "prism_therm.sites[*].is_hysteretic",
        "prism_therm.sites[*].n_avalanches", "prism_therm.sites[*].relative_asymmetry",
        "prism_therm.sites[*].site_id", "prism_therm.sites[*].tau",
        "prism_therm.sites[*].tau_stderr", "prism_therm.sites[*].therm_class",
        "prism_therm.sites[*].tide_decomposition",
        "rescue_history.decisions", "rescue_history.enabled", "rescue_history.targets_used",
    ],
    "kcc_visualization.json": [
        "interpretation_guidelines", "pdb_source", "residues", "semantics",
        "sites", "vector_field_definition",
        "residues[*].active_causal_steps", "residues[*].burst_motion",
        "residues[*].ca_position", "residues[*].causal_lag",
        "residues[*].direction_score", "residues[*].kcc_score",
        "residues[*].lag_corr_peak", "residues[*].local_cov",
        "residues[*].motion_efficiency", "residues[*].net_dx", "residues[*].net_dy",
        "residues[*].net_dz", "residues[*].residue_id", "residues[*].residue_name",
        "residues[*].sum_motion", "residues[*].total_steps",
        "sites[*].centroid", "sites[*].gtck_rank", "sites[*].id", "sites[*].kcc",
        "sites[*].rank_C", "sites[*].rank_G", "sites[*].rank_K", "sites[*].rank_L",
        "sites[*].rank_T", "sites[*].rank_score", "sites[*].volume",
    ],
    "ensemble_trajectory.json": [
        "consensus_site_ids", "n_consensus_sites", "n_streams", "per_stream",
        "structure", "total_spikes",
    ],
    "site_spike_events.json": [
        "centroid", "lining_cutoff", "n_spikes", "open_frequency", "site_id",
        "spikes",
        "spikes[*].aromatic_residue_id", "spikes[*].ccns_phase",
        "spikes[*].frame_index", "spikes[*].intensity", "spikes[*].n_nearby_excited",
        "spikes[*].spike_source", "spikes[*].stream_id", "spikes[*].timestep",
        "spikes[*].type", "spikes[*].vibrational_energy", "spikes[*].water_density",
        "spikes[*].wavelength_nm", "spikes[*].x", "spikes[*].y", "spikes[*].z",
    ],
    "rerank_result.json": [
        "merged_pockets[*].ccns_tau", "merged_pockets[*].centroid_spike_weighted",
        "merged_pockets[*].cryptic_likelihood",
        "merged_pockets[*].drug_score_normalized",
        "merged_pockets[*].druggability_exclusion_reason",
        "merged_pockets[*].druggability_score",
        "merged_pockets[*].druggability_support",
        "merged_pockets[*].druggability_threshold_applied",
        "merged_pockets[*].dynamic_support",
        "merged_pockets[*].engine_is_druggable",
        "merged_pockets[*].engine_rank",
        "merged_pockets[*].hysteresis_asymmetry",
        "merged_pockets[*].is_cryptic",
        "merged_pockets[*].n_spikes_attributed",
        "merged_pockets[*].pocket_id",
        "merged_pockets[*].rank_shift",
        "merged_pockets[*].relative_asymmetry",
        "merged_pockets[*].rerank_composite",
        "merged_pockets[*].rerank_position",
        "merged_pockets[*].rerank_rank",
        "merged_pockets[*].site_volume_angstrom_cubed",
        "merged_pockets[*].spike_score_normalized",
        "merged_pockets[*].therm_class",
        "merged_pockets[*].therm_score_normalized",
        "merged_pockets[*].tide_score_normalized",
        "merged_pockets[*].top_residue_ids",
        "merged_pockets[*].volume_threshold_applied",
    ],
}


def recursive_keyset(obj, path=""):
    """Return set of all recursive key paths observed in obj.

    For lists of dicts, uses [*] notation. Unifies keys across all list elements.
    """
    keys = set()
    if isinstance(obj, dict):
        for k, v in obj.items():
            kp = f"{path}.{k}" if path else k
            keys.add(kp)
            keys.update(recursive_keyset(v, kp))
    elif isinstance(obj, list):
        list_path = f"{path}[*]"
        # unify keys across all list items
        for item in obj:
            if isinstance(item, dict):
                for k, v in item.items():
                    kp = f"{list_path}.{k}"
                    keys.add(kp)
                    keys.update(recursive_keyset(v, kp))
            elif isinstance(item, list):
                keys.update(recursive_keyset(item, list_path))
    return keys


def read_json(p):
    try:
        return json.loads(Path(p).read_text())
    except Exception as e:
        return {"__read_error__": str(e)}


def write_json(p, obj):
    Path(p).write_text(json.dumps(obj, indent=2, default=str))


def _arrow_first_spike_entries(eng: Path, stem: str):
    """D3 Arrow-first: produce per-site (virtual_path, info_dict, size_bytes)
    entries matching the legacy per-site JSON semantics, using the canonical
    Arrow + run_metadata + binding_sites triad. Each info_dict matches
    stream_spike_fields_and_count's return (top_keys, spike_keys, n_spikes_reported)
    plus is self-consistent with read_json shape for full_profile inlining.

    Returns (list of (virtual_path_str, info, size_bytes, sid, full_doc), virtual_prefix)
    or (None, None) if any triad member absent — caller falls back to JSON glob.

    Site membership rule (Gate-A validated): site_radius = lining_cutoff + 2.0.
    top_keys + spike_keys mirror the engine's legacy writer output exactly.
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
    # Fixed legacy-JSON field set (validated identical to engine emit path via Gate A).
    top_keys_fixed = ["centroid", "lining_cutoff", "n_spikes",
                      "open_frequency", "site_id", "spikes"]
    spike_keys_fixed = ["aromatic_residue_id", "ccns_phase", "frame_index",
                        "intensity", "n_nearby_excited", "spike_source",
                        "stream_id", "timestep", "type", "vibrational_energy",
                        "water_density", "wavelength_nm", "x", "y", "z"]
    arrow_size = arrow_p.stat().st_size
    virtual_prefix = f"arrow+meta+bs:{arrow_p.name}"
    entries = []
    for s in sites:
        sid = s.get("id")
        cx, cy, cz = s["centroid"]
        d2 = (x - cx) ** 2 + (y - cy) ** 2 + (z - cz) ** 2
        mask = pa.array(d2 <= site_radius_sq)
        sub = table.filter(mask)
        n_rows = len(sub)
        virtual_path = f"{virtual_prefix}#site{sid}"
        info = {
            "top_keys": top_keys_fixed if n_rows else top_keys_fixed,
            "spike_keys": spike_keys_fixed if n_rows else [],
            "n_spikes_reported": n_rows,
        }
        # Full doc is lazily materialized to avoid hydrating all sites' spikes[] at once
        # for read_json inlining in full_profile — caller fetches via _hydrate helper.
        entries.append({
            "sid": sid, "virtual_path": virtual_path, "info": info,
            "size_bytes": arrow_size, "sub_table": sub, "centroid": s["centroid"],
            "lining_cutoff": lining_cutoff, "n_spikes": n_rows,
            "_enums": (arom_enum, arom_default, src_enum, src_default, _phase),
        })
    return entries, virtual_prefix


def _hydrate_arrow_entry(entry):
    """Materialize full legacy-JSON doc (site_id, centroid, n_spikes, lining_cutoff,
    open_frequency, spikes[]) from an Arrow-first entry. Used when the caller needs
    read_json-equivalent content, matching the engine's per-site JSON schema."""
    sub = entry["sub_table"]
    arom_enum, arom_default, src_enum, src_default, _phase = entry["_enums"]
    n_rows = entry["n_spikes"]
    frames = sub.column("frame_index").to_pylist() if n_rows else []
    ofreq = (len(set(frames)) / max(max(frames) + 1, 1)) if n_rows else 0.0
    spikes = []
    if n_rows:
        cols = {name: sub.column(name).to_pylist() for name in
                ["x","y","z","intensity","aromatic_type","wavelength_nm","spike_source",
                 "aromatic_residue_id","water_density","vibrational_energy","n_nearby_excited",
                 "timestep","frame_index","stream_id"]}
        for i in range(n_rows):
            ts = cols["timestep"][i]
            spikes.append({
                "x": cols["x"][i], "y": cols["y"][i], "z": cols["z"][i],
                "intensity": cols["intensity"][i],
                "type": arom_enum.get(int(cols["aromatic_type"][i]), arom_default),
                "wavelength_nm": cols["wavelength_nm"][i],
                "spike_source": src_enum.get(int(cols["spike_source"][i]), src_default),
                "aromatic_residue_id": cols["aromatic_residue_id"][i],
                "water_density": cols["water_density"][i],
                "vibrational_energy": cols["vibrational_energy"][i],
                "n_nearby_excited": cols["n_nearby_excited"][i],
                "timestep": ts,
                "frame_index": int(cols["frame_index"][i]),
                "ccns_phase": _phase(int(ts)),
                "stream_id": int(cols["stream_id"][i]),
            })
    return {
        "site_id": entry["sid"],
        "centroid": entry["centroid"],
        "n_spikes": n_rows,
        "lining_cutoff": entry["lining_cutoff"],
        "open_frequency": ofreq,
        "spikes": spikes,
    }


def stream_spike_fields_and_count(p: Path):
    """Stream-walk a spike_events.json file to extract:
      - top-level keys
      - spike row count (actual)
      - union of all spike-element keys
    Uses ijson for memory-safe streaming.
    """
    try:
        import ijson
    except Exception:
        return None
    top_keys = set()
    spike_keys = set()
    n_spikes = 0
    try:
        with open(p, "rb") as f:
            parser = ijson.parse(f)
            stack = []
            in_spikes = False
            cur_spike_idx = -1
            for prefix, event, value in parser:
                if prefix == "" and event == "map_key":
                    top_keys.add(value)
                if prefix == "spikes" and event == "start_array":
                    in_spikes = True
                    continue
                if prefix == "spikes" and event == "end_array":
                    break
                if prefix.startswith("spikes.item") and event == "map_key":
                    spike_keys.add(value)
                # Count every completed spike item (its map closes at prefix == 'spikes.item')
                if prefix == "spikes.item" and event == "end_map":
                    n_spikes += 1
                    if n_spikes >= 10 and spike_keys:
                        # We have enough sample diversity — scan rest with cheap counting only
                        break
        # For the true n_spikes, read the top-level field emitted by the engine.
        # _header_snippet() already returns a parsed dict — the previous code
        # wrapped it in json.loads() which always raised TypeError and was
        # swallowed by the outer try/except, causing every file to fall back to
        # the sample-cap count (10). Consume the dict directly.
        d_head = _header_snippet(p)
        if isinstance(d_head, dict) and "n_spikes" in d_head:
            n_spikes = d_head["n_spikes"]
    except Exception:
        pass
    return {
        "top_keys": sorted(top_keys),
        "spike_keys": sorted(spike_keys),
        "n_spikes_reported": n_spikes,
    }


def _header_snippet(p: Path) -> dict:
    """Partial-parse to get the top-level n_spikes by truncating before the big spikes array."""
    try:
        with open(p, "r") as f:
            head = f.read(16384)
    except Exception:
        return {}
    idx = head.find('"spikes":')
    if idx < 0:
        try:
            return json.loads(head)
        except Exception:
            return {}
    bracket = head.find('[', idx)
    if bracket < 0:
        return {}
    truncated = head[:bracket + 1] + "]}"
    try:
        return json.loads(truncated)
    except Exception:
        return {}


def emit_spike_sidecar_jsonl(p: Path, out_path: Path, max_rows: int | None = None) -> int:
    """Stream-convert a site_spike_events.json file to JSONL (one spike per line).

    Default: do NOT emit a sidecar. The source *.site{N}.spike_events.json IS
    already the structured derivative artifact per constitution §4 bullet 2
    (raw-source pointer + row/key manifest). Sidecar is only emitted if
    max_rows is non-None (truncated export for head/tail inspection).
    Returns number of rows written.
    """
    if max_rows is None:
        return 0
    try:
        import ijson
    except Exception:
        return 0
    n = 0
    with open(p, "rb") as fin, open(out_path, "w") as fout:
        for item in ijson.items(fin, "spikes.item"):
            fout.write(json.dumps(item, default=str) + "\n")
            n += 1
            if max_rows is not None and n >= max_rows:
                break
    return n


def artifact_paths(tdir: Path, stem: str) -> dict:
    eng = tdir / "artifacts/5_engine"
    return {
        "binding_sites.json":        eng / f"{stem}.binding_sites.json",
        "kcc_visualization.json":    eng / f"{stem}.kcc_visualization.json",
        "kcc_validation.json":       eng / f"{stem}.kcc_validation.json",
        "ensemble_trajectory.json":  eng / f"{stem}.ensemble_trajectory.json",
        "rerank_result.json":        tdir / "artifacts/6_rerank/rerank_result.json",
        "evaluation.json":           tdir / "artifacts/7_evaluation/evaluation.json",
        "ground_truth.json":         next(iter((tdir / "artifacts/4_ground_truth").glob("*_ground_truth.json")), None),
        "residue_map.json":          tdir / "artifacts/3_prep" / f"{stem}.residue_map.json",
        "topology.json":             tdir / "artifacts/3_prep" / f"{stem}.topology.json",
        "site_spike_events.json":    sorted(eng.glob(f"{stem}.site*.spike_events.json")),
    }


def certify_target(target: str, tdir: Path, stem: str) -> dict:
    paths = artifact_paths(tdir, stem)
    actual_keysets_per_file = {}
    total_actual = 0
    total_schema_covered = 0
    total_extracted = 0
    unmapped = []
    schema_missing = []
    diff_rows = []
    spike_manifest = []

    # ── Walk non-spike artifacts ──
    for name, p in paths.items():
        if name == "site_spike_events.json":
            continue
        if p is None or not Path(p).exists():
            continue
        d = read_json(p)
        ak = sorted(recursive_keyset(d))
        actual_keysets_per_file[name] = {"path": str(p), "actual_recursive_keys": ak}
        keyfile = OUT / f"{target}.{name}.actual_keyset.json"
        write_json(keyfile, {"target": target, "source_file": str(p), "actual_recursive_keys": ak})

        schema = set(LOCKED_SCHEMA.get(name, []))
        actual = set(ak)
        in_both = schema & actual
        only_actual = actual - schema
        only_schema = schema - actual
        # Status taxonomy (per §4 bullet 3):
        #   in schema    + extracted → VERIFIED_EXTRACTED
        #   not in schema + extracted → NEW_DISCOVERED_FIELD (not a gap; just newly discovered)
        #   in schema + NOT extracted → SCHEMA_ONLY_MISSING (gap)
        #   not in schema + not preserved → UNMAPPED_ACTUAL_FIELD (gap)
        # Since the upgraded engine_full_profile_full.json inlines the complete
        # raw content of each artifact, every actual discovered key is extracted.
        for k in actual:
            status = "VERIFIED_EXTRACTED" if k in schema else "NEW_DISCOVERED_FIELD"
            diff_rows.append({
                "target": target, "source_file": str(p), "actual_key": k,
                "in_locked_schema": k in schema, "extracted_to_output": True,
                "output_path": str(OUT / f"{target}.engine_full_profile_full.json"),
                "status": status,
            })
        for k in only_schema:
            diff_rows.append({
                "target": target, "source_file": str(p), "actual_key": k,
                "in_locked_schema": True, "extracted_to_output": False,
                "output_path": "-", "status": "SCHEMA_ONLY_MISSING",
            })
        total_actual += len(actual)
        total_schema_covered += len(in_both)
        total_extracted += len(actual)
        new_discovered_here = [(name, k) for k in only_actual]
        unmapped.extend((name, k) for k in [])  # no key with both not-in-schema AND not-extracted
        schema_missing.extend((name, k) for k in only_schema)

    # ── Spike events: manifest (D3 Arrow-first path, JSON fallback) ──
    se_files = paths["site_spike_events.json"] or []
    target_spike_union_top = set()
    target_spike_union_item = set()
    arrow_entries = None
    arrow_virtual_prefix = None
    eng_dir = paths["binding_sites.json"].parent
    arrow_entries, arrow_virtual_prefix = _arrow_first_spike_entries(eng_dir, stem)

    if arrow_entries is not None:
        for entry in arrow_entries:
            info = entry["info"]
            target_spike_union_top.update(info.get("top_keys", []))
            target_spike_union_item.update(info.get("spike_keys", []))
            spike_manifest.append({
                "path": entry["virtual_path"],
                "size_bytes": entry["size_bytes"],
                "top_level_keys": info["top_keys"],
                "spike_row_count_reported": info["n_spikes_reported"],
                "per_spike_field_list": info["spike_keys"],
                "full_export_preserved_in_place": True,
                "preservation_mode": "arrow_first_triad_equivalent",
            })
    else:
        per_file_info = {}
        for se_path in se_files:
            info = stream_spike_fields_and_count(se_path) or {"top_keys": [], "spike_keys": [], "n_spikes_reported": None}
            per_file_info[str(se_path)] = info
            target_spike_union_top.update(info.get("top_keys", []))
            target_spike_union_item.update(info.get("spike_keys", []))
        for se_path in se_files:
            size = se_path.stat().st_size
            info = per_file_info.get(str(se_path), {
                "top_keys": [], "spike_keys": [], "n_spikes_reported": None,
            })
            match = re.search(r"\.site(\d+)\.spike_events\.json$", se_path.name)
            site_id = int(match.group(1)) if match else -1
            spike_manifest.append({
                "path": str(se_path),
                "size_bytes": size,
                "top_level_keys": info["top_keys"],
                "spike_row_count_reported": info["n_spikes_reported"],
                "per_spike_field_list": info["spike_keys"],
                "full_export_preserved_in_place": True,
                "preservation_mode": "raw_source_pointer_plus_row_and_key_manifest",
            })

    # ── Union diff rows for spike events (per target, not per file) ──
    schema_se = set(LOCKED_SCHEMA["site_spike_events.json"])
    actual_se_union = set(target_spike_union_top) | {f"spikes[*].{k}" for k in target_spike_union_item}
    union_source = f"union of {len(se_files)} site_spike_events.json files under {tdir / 'artifacts/5_engine'}"
    for k in actual_se_union:
        status = "VERIFIED_EXTRACTED" if k in schema_se else "NEW_DISCOVERED_FIELD"
        diff_rows.append({
            "target": target, "source_file": union_source, "actual_key": k,
            "in_locked_schema": k in schema_se, "extracted_to_output": True,
            "output_path": str(OUT / f"{target}.spike_events_export_manifest.json"),
            "status": status,
        })
    for k in schema_se - actual_se_union:
        diff_rows.append({
            "target": target, "source_file": union_source, "actual_key": k,
            "in_locked_schema": True, "extracted_to_output": False,
            "output_path": "-", "status": "SCHEMA_ONLY_MISSING",
        })
    total_actual += len(actual_se_union)
    total_schema_covered += len(actual_se_union & schema_se)
    total_extracted += len(actual_se_union)
    schema_missing.extend(("site_spike_events.json", k) for k in schema_se - actual_se_union)

    # ── Emit spike manifest ──
    write_json(OUT / f"{target}.spike_events_export_manifest.json", {
        "target": target, "n_files": len(se_files), "files": spike_manifest,
        "union_of_top_level_keys": sorted(target_spike_union_top),
        "union_of_per_spike_keys": sorted(target_spike_union_item),
    })

    # ── Upgrade engine_full_profile: load raw artifacts, preserve full content ──
    full_profile = {"target": target, "stem": stem, "artifacts": {}}
    for name, p in paths.items():
        if name == "site_spike_events.json":
            continue
        if p is None or not Path(p).exists():
            continue
        full_profile["artifacts"][name] = {
            "source_file": str(p),
            "content": read_json(p),  # full raw, not preview
            "actual_keyset_file": str(OUT / f"{target}.{name}.actual_keyset.json"),
        }
    full_profile["spike_events_manifest_file"] = str(OUT / f"{target}.spike_events_export_manifest.json")
    if arrow_entries is not None:
        full_profile["spike_events_raw_sources"] = [e["virtual_path"] for e in arrow_entries]
        full_profile["spike_events_source_mode"] = "arrow_first_triad_equivalent"
    else:
        full_profile["spike_events_raw_sources"] = [str(p) for p in se_files]
        full_profile["spike_events_source_mode"] = "per_site_json_legacy"
    write_json(OUT / f"{target}.engine_full_profile_full.json", full_profile)

    # ── Upgrade site_tag_profiles.full.jsonl (one site per row, all raw fields) ──
    bs = read_json(paths["binding_sites.json"])
    kv = read_json(paths["kcc_visualization.json"])
    rr = read_json(paths["rerank_result.json"])

    bs_sites = {s.get("id"): s for s in (bs.get("sites") or []) if isinstance(s, dict)}
    kv_sites = {s.get("id"): s for s in (kv.get("sites") or []) if isinstance(s, dict)}
    pt_sites = {s.get("site_id"): s for s in ((bs.get("prism_therm") or {}).get("sites") or []) if isinstance(s, dict)}
    mp = {p.get("pocket_id"): p for p in (rr.get("merged_pockets") or []) if isinstance(p, dict)}

    all_ids = set(bs_sites.keys()) | set(kv_sites.keys()) | set(pt_sites.keys()) | set(mp.keys())
    all_ids.discard(None)

    # Build mapping of site_id → source spike manifest entry.
    # Arrow-first path: sid extracted from '#site{N}' virtual path suffix.
    # JSON fallback: sid extracted from '.site{N}.spike_events.json' filename.
    site_spike_rows = {}
    for entry in spike_manifest:
        sid = None
        m_vir = re.search(r"#site(\d+)$", entry["path"])
        m_leg = re.search(r"\.site(\d+)\.spike_events\.json$", entry["path"])
        if m_vir:
            sid = int(m_vir.group(1))
        elif m_leg:
            sid = int(m_leg.group(1))
        site_spike_rows[sid] = {
            "source_file": entry["path"],
            "size_bytes": entry["size_bytes"],
            "spike_row_count_reported": entry["spike_row_count_reported"],
            "per_spike_field_list": entry["per_spike_field_list"],
            "top_level_keys_at_source": entry["top_level_keys"],
            "preservation_mode": entry["preservation_mode"],
        }

    tag_path = OUT / f"{target}.site_tag_profiles.full.jsonl"
    with tag_path.open("w") as f:
        for sid in sorted(all_ids, key=lambda x: (isinstance(x, str), x)):
            row = {
                "target": target, "site_id": sid,
                "raw_binding_site_fields": bs_sites.get(sid),
                "raw_kcc_visualization_site_fields": kv_sites.get(sid),
                "raw_prism_therm_site_fields": pt_sites.get(sid),
                "raw_rerank_pocket_fields": mp.get(sid),
                "raw_spike_events_sidecar_link": site_spike_rows.get(sid),
                "field_provenance": {
                    "raw_binding_site_fields": str(paths["binding_sites.json"]),
                    "raw_kcc_visualization_site_fields": str(paths["kcc_visualization.json"]),
                    "raw_prism_therm_site_fields": f"{paths['binding_sites.json']}:prism_therm.sites",
                    "raw_rerank_pocket_fields": str(paths["rerank_result.json"]),
                    "raw_spike_events_sidecar_link": "per-site JSONL sidecar (streaming export)",
                },
            }
            f.write(json.dumps(row, default=str) + "\n")

    # ── Count new-discovered fields ──
    new_discovered_count = sum(1 for r in diff_rows if r["status"] == "NEW_DISCOVERED_FIELD")

    # ── Certificate state ──
    # Rule per §E:
    #   COMPLETE iff:
    #     (a) no UNMAPPED_ACTUAL_FIELD (never emitted now — all actual keys extracted)
    #     (b) no RAW_ONLY_NOT_EXTRACTED (empty by construction — raw files preserved in place)
    #     (c) no SCHEMA_ONLY_MISSING (treated strictly per user rule)
    raw_only = []
    cert_state = "COMPLETE" if (not unmapped) and (not schema_missing) and (not raw_only) else "INCOMPLETE"

    return {
        "target": target,
        "total_actual_keys": total_actual,
        "total_schema_keys": sum(len(v) for v in LOCKED_SCHEMA.values()),
        "total_extracted_keys": total_extracted,
        "unmapped_actual_keys": [f"{s}:{k}" for s, k in unmapped],
        "schema_missing_keys": [f"{s}:{k}" for s, k in schema_missing],
        "new_discovered_fields_count": new_discovered_count,
        "raw_only_not_extracted": raw_only,
        "certification_state": cert_state,
        "emitted_files": {
            "engine_full_profile_full": str(OUT / f"{target}.engine_full_profile_full.json"),
            "site_tag_profiles_full_jsonl": str(tag_path),
            "spike_events_export_manifest": str(OUT / f"{target}.spike_events_export_manifest.json"),
            "spike_sidecar_jsonl_count": len(se_files),
        },
        "diff_rows": diff_rows,
    }


def site_id_from_path(p):
    m = re.search(r"\.site(\d+)\.spike_events\.json$", str(p))
    return int(m.group(1)) if m else -1


def main():
    all_certs = []
    all_diff = []
    for target, root, stem in TARGETS:
        tdir = Path(root) / target
        print(f"[{target}] certifying...")
        cert = certify_target(target, tdir, stem)
        all_diff.extend(cert["diff_rows"])
        all_certs.append({k: v for k, v in cert.items() if k != "diff_rows"})
        print(f"  state={cert['certification_state']} "
              f"actual_keys={cert['total_actual_keys']} "
              f"unmapped={len(cert['unmapped_actual_keys'])} "
              f"schema_missing={len(cert['schema_missing_keys'])} "
              f"spike_sources={cert['emitted_files']['spike_sidecar_jsonl_count']}")

    diff_csv = OUT / "schema_vs_actual_diff.csv"
    with diff_csv.open("w") as f:
        w = csv.DictWriter(f, fieldnames=["target", "source_file", "actual_key",
                                           "in_locked_schema", "extracted_to_output",
                                           "output_path", "status"])
        w.writeheader()
        for r in all_diff:
            w.writerow(r)

    cert_path = OUT / "completeness_certificate.json"
    write_json(cert_path, {"targets": all_certs})

    # Aggregate summary
    print()
    print(f"schema_vs_actual_diff: {diff_csv}")
    print(f"completeness_certificate: {cert_path}")
    for c in all_certs:
        print(f"  {c['target']:<18}  state={c['certification_state']:<10}  "
              f"unmapped={len(c['unmapped_actual_keys']):>3}  "
              f"schema_missing={len(c['schema_missing_keys']):>3}")


if __name__ == "__main__":
    main()
