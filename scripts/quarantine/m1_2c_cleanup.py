#!/usr/bin/env python3
"""M1.2c — final cleanup pass for the completed-target med-chem review.

Emits:
  /tmp/engine_full_profiles/open_frequency_audit.csv
  /tmp/engine_full_profiles/open_frequency_audit.json
  /tmp/engine_full_profiles/internal_medchem_review_table.v3.csv
  /tmp/engine_full_profiles/m1_2c_external_readiness_gate.json

Read-only wrt engine output; modifies v2 JSON note bodies in-place via v3 files.
"""
from __future__ import annotations
import csv
import json
import re
import statistics
from collections import Counter
from pathlib import Path

OUT = Path("/tmp/engine_full_profiles")

TARGETS = [
    ("wrn_apo", "/mnt/storage/prism-outputs/twin-10-patent/wrn_apo/artifacts/5_engine", "6yhr"),
    ("menin_apo", "/mnt/storage/prism-outputs/twin-10-patent/menin_apo/artifacts/5_engine", "3re2"),
    ("smarca2_brd_apo", "/mnt/storage/prism-outputs/twin-10-patent/smarca2_brd_apo/artifacts/5_engine", "4qy4"),
    ("pkmyt1_apo", "/mnt/storage/prism-outputs/twin-10-patent/pkmyt1_apo/artifacts/5_engine", "3p1a"),
    ("kras_g12d_apo", "/mnt/storage/prism-outputs/twin-10-patent/kras_g12d_apo/artifacts/5_engine", "7f0w"),
    ("m1_2nvp", "/mnt/storage/prism-outputs/m1-strict-dcc-panel/m1_2nvp/artifacts/5_engine", "2nvp"),
    ("m1_1xhx", "/mnt/storage/prism-outputs/m1-strict-dcc-panel/m1_1xhx/artifacts/5_engine", "1xhx"),
]


def read_open_frequency(p: Path) -> float | None:
    try:
        with p.open("r") as f:
            head = f.read(4096)
        m = re.search(r'"open_frequency"\s*:\s*([\-\d\.eE]+)', head)
        if m:
            return float(m.group(1))
    except Exception:
        return None
    return None


def _arrow_first_open_frequencies(eng: Path, stem: str):
    """D3 Arrow-first: compute {sid: open_frequency} for all sites in
    binding_sites.json using Arrow + run_metadata triad.

    Uses the Gate-A validated spatial membership rule (site_radius = lining_cutoff + 2.0)
    and the engine's open_frequency formula: unique(frame_index) / max(max_frame+1, 1).
    Returns (dict, virtual_path_prefix) or (None, None) if preconditions missing
    — caller falls back to the JSON reader.
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
    lining_cutoff = meta.get("lining_cutoff", 8.0)
    site_radius_sq = (float(lining_cutoff) + 2.0) ** 2
    bs = json.loads(bs_p.read_text())
    sites = [s for s in (bs.get("sites") or []) if isinstance(s, dict) and s.get("centroid")]
    if not sites:
        return {}, f"arrow+meta+bs:{arrow_p.name}"
    with arrow_p.open("rb") as f:
        magic = f.read(8)
    opener = ipc.open_file if magic.startswith(b"ARROW1") else ipc.open_stream
    with arrow_p.open("rb") as f:
        table = opener(f).read_all()
    x = table.column("x").to_numpy()
    y = table.column("y").to_numpy()
    z = table.column("z").to_numpy()
    frame_col_full = table.column("frame_index").to_numpy()
    out = {}
    for s in sites:
        sid = s.get("id")
        cx, cy, cz = s["centroid"]
        d2 = (x - cx) ** 2 + (y - cy) ** 2 + (z - cz) ** 2
        mask = d2 <= site_radius_sq
        frames = frame_col_full[mask]
        if frames.size:
            ofreq = float(len(set(frames.tolist())) / max(int(frames.max()) + 1, 1))
        else:
            ofreq = 0.0
        out[sid] = ofreq
    return out, f"arrow+meta+bs:{arrow_p.name}"


def open_frequency_audit():
    rows = []
    by_target = {}
    for target, engdir, stem in TARGETS:
        eng = Path(engdir)
        vals = []
        arrow_ofreqs, arrow_virtual = _arrow_first_open_frequencies(eng, stem)
        if arrow_ofreqs is not None:
            # Arrow-first path — iterate sids from binding_sites.json (source of truth)
            for sid, ofreq in sorted(arrow_ofreqs.items(), key=lambda kv: (kv[0] is None, kv[0])):
                rows.append({"target": target, "site_id": sid,
                             "file": f"{arrow_virtual}#site{sid}",
                             "open_frequency": ofreq})
                if ofreq is not None:
                    vals.append(ofreq)
        else:
            # JSON fallback — original glob-based enumeration
            files = sorted(eng.glob(f"{stem}.site*.spike_events.json"))
            for f in files:
                m = re.search(rf"{stem}\.site(\d+)\.spike_events\.json$", f.name)
                sid = int(m.group(1)) if m else -1
                ofreq = read_open_frequency(f)
                rows.append({"target": target, "site_id": sid, "file": str(f), "open_frequency": ofreq})
                if ofreq is not None:
                    vals.append(ofreq)
        by_target[target] = vals

    all_vals = [r["open_frequency"] for r in rows if r["open_frequency"] is not None]
    if all_vals:
        mn, mx = min(all_vals), max(all_vals)
        mean = statistics.fmean(all_vals)
        unique = sorted(set(all_vals))
    else:
        mn = mx = mean = None
        unique = []
    constant = (len(unique) == 1)

    csv_path = OUT / "open_frequency_audit.csv"
    with csv_path.open("w") as f:
        w = csv.DictWriter(f, fieldnames=["target", "site_id", "file", "open_frequency"])
        w.writeheader()
        for r in rows:
            w.writerow(r)

    json_path = OUT / "open_frequency_audit.json"
    json_path.write_text(json.dumps({
        "n_site_files_scanned": len(rows),
        "n_files_with_value": len(all_vals),
        "min": mn, "max": mx, "mean": mean,
        "unique_values": unique,
        "is_constant": constant,
        "verdict": "OPEN_FREQUENCY_NON_DIFFERENTIATING" if constant else "OPEN_FREQUENCY_DIFFERENTIATING",
        "per_target_summary": {
            t: {"n": len(v), "unique": sorted(set(v)),
                "min": min(v) if v else None, "max": max(v) if v else None}
            for t, v in by_target.items()
        },
    }, indent=2))
    return rows, all_vals, constant, csv_path, json_path


def strip_open_frequency_from_reasoning(notes):
    """Remove explanatory use of open_frequency from all text fields.
    Keep it in the raw_value_block."""
    keys = ["med_chem_summary", "residue_triage_note",
            "dynamic_mechanistic_note", "spatial_review_note",
            "real_world_value_note"]
    for k in keys:
        if k in notes and isinstance(notes[k], str):
            # Remove phrases like "open_frequency=1.0" anywhere, also handle "; open_frequency=..."
            notes[k] = re.sub(r"(;\s*)?open_frequency=[\-\d\.eE]+", "", notes[k])
            # Clean up accidental doubled semicolons or stray commas from removal
            notes[k] = re.sub(r";\s*;", ";", notes[k])
            notes[k] = re.sub(r",\s*,", ",", notes[k])
    return notes


def rename_mutagenesis_framing(notes, site_row, budget_label):
    """For GT25_OR_HARD_NOT_MATCH or DCC>15, replace 'mutagenesis priority'
    with 'mechanism-mapping mutagenesis' to remove chemistry-forward reading."""
    band = notes.get("action_gate_band")
    canon = notes["raw_value_block"].get("canonical_dcc_angstrom")
    canon_gt_15 = isinstance(canon, (int, float)) and canon > 15.0
    if band == "GT25_OR_HARD_NOT_MATCH" or canon_gt_15:
        if notes.get("primary_value_framing") == "mutagenesis priority":
            notes["primary_value_framing"] = "mechanism-mapping mutagenesis"
            notes["next_action"] = "design residue mutation panel (mechanism-mapping, not chemistry-forward)"
    return notes


def chemistry_usable_now(notes):
    bud = notes.get("budget_label")
    band = notes.get("action_gate_band")
    conf = notes.get("confidence_state")
    is_bm = notes["raw_value_block"].get("is_best_match_vs_ground_truth")
    if band == "LE8" and bud == "HIGH_ATTENTION" and is_bm and conf == "VERIFIED_LIGAND_SITE_SUPPORT":
        return "yes"
    if band == "GT25_OR_HARD_NOT_MATCH":
        return "no"
    return "no"


def mechanism_usable_now(notes):
    band = notes.get("action_gate_band")
    conf = notes.get("confidence_state")
    if band in ("GT25_OR_HARD_NOT_MATCH", "15to25") and conf == "MECHANISTIC_SUPPORT_ONLY":
        return "yes"
    if band in ("LE8", "8to15"):
        return "yes"
    return "no"


def external_dossier_ok(notes):
    therm_safe = notes.get("therm_fields_medchem_safe", False)
    conf = notes.get("confidence_state")
    is_bm = notes["raw_value_block"].get("is_best_match_vs_ground_truth")
    band = notes.get("action_gate_band")
    joins = notes.get("note_data_join_failures") or []
    if not is_bm:
        return "no"
    if conf != "VERIFIED_LIGAND_SITE_SUPPORT":
        return "no"
    if not therm_safe:
        return "no"
    if band != "LE8":
        return "no"
    if joins:
        return "no"
    return "yes"


def main():
    rows, all_vals, constant_panel_wide, ofreq_csv, ofreq_json = open_frequency_audit()

    # Also check constancy across the ACTUAL note-set (top-3 + best-match per target)
    note_set_vals = []
    for target in ["wrn_apo", "menin_apo", "smarca2_brd_apo", "pkmyt1_apo",
                   "kras_g12d_apo", "m1_2nvp", "m1_1xhx"]:
        src = OUT / f"{target}.site_decision_notes.v2.json"
        if not src.exists():
            continue
        v2 = json.loads(src.read_text())
        for n in v2["notes"]:
            ofq = n["raw_value_block"].get("open_frequency_from_spike_events")
            if isinstance(ofq, (int, float)):
                note_set_vals.append(ofq)
    constant_in_note_set = (len(set(note_set_vals)) == 1)
    # Strip rule: if constant within the current note set, strip from reasoning
    strip_open_freq = constant_in_note_set
    constant = constant_in_note_set  # used below

    new_csv_rows = []
    gate_records = []

    for target in ["wrn_apo", "menin_apo", "smarca2_brd_apo", "pkmyt1_apo",
                   "kras_g12d_apo", "m1_2nvp", "m1_1xhx"]:
        src = OUT / f"{target}.site_decision_notes.v2.json"
        if not src.exists():
            continue
        payload = json.loads(src.read_text())
        patched_notes = []
        for n in payload["notes"]:
            if constant:
                n = strip_open_frequency_from_reasoning(n)
            n = rename_mutagenesis_framing(n, None, n.get("budget_label"))

            chem_ok = chemistry_usable_now(n)
            mech_ok = mechanism_usable_now(n)
            ext_ok = external_dossier_ok(n)

            n["chemistry_usable_now"] = chem_ok
            n["mechanism_usable_now"] = mech_ok
            n["internal_medchem_ok"] = "yes" if (chem_ok == "yes" or mech_ok == "yes") else "no"
            n["external_dossier_ok"] = ext_ok

            raw = n["raw_value_block"]
            canon_dcc = raw.get("canonical_dcc_angstrom")
            canon_verdict = raw.get("canonical_verdict")

            new_csv_rows.append({
                "target": target,
                "site_id": n["site_id"],
                "rank": raw.get("current_reference_rank_under_B_no_therm"),
                "verified_best_match": raw.get("is_best_match_vs_ground_truth"),
                "confidence_state": n["confidence_state"],
                "therm_fields_medchem_safe": n.get("therm_fields_medchem_safe"),
                "action_gate_band": n["action_gate_band"],
                "budget_label": n["budget_label"],
                "canonical_dcc": canon_dcc if canon_dcc is not None else "UNAVAILABLE",
                "canonical_verdict": canon_verdict or "UNAVAILABLE",
                "volume": raw.get("volume_angstrom_cubed"),
                "n_spikes_attributed": raw.get("n_spikes_attributed"),
                "top_residues": str((raw.get("top_residue_ids") or [])[:8]),
                "primary_value_framing": n["primary_value_framing"],
                "next_action": n["next_action"],
                "kcc_driver_residue_id": (raw.get("kcc_driver_residue_id")
                                          if isinstance(raw.get("kcc_driver_residue_id"), int)
                                          else "FEATURE_GAP"),
                "chemistry_usable_now": chem_ok,
                "mechanism_usable_now": mech_ok,
                "internal_medchem_ok": n["internal_medchem_ok"],
                "external_dossier_ok": ext_ok,
                "specificity_pass": n.get("specificity_pass"),
                "note_data_join_failures": (";".join(n.get("note_data_join_failures") or [])
                                            or "-"),
            })
            gate_records.append({
                "target": target, "site_id": n["site_id"],
                "internal_medchem_ok": n["internal_medchem_ok"],
                "external_dossier_ok": ext_ok,
                "rationale": {
                    "band": n["action_gate_band"],
                    "budget_label": n["budget_label"],
                    "confidence_state": n["confidence_state"],
                    "therm_fields_medchem_safe": n.get("therm_fields_medchem_safe"),
                    "is_best_match": raw.get("is_best_match_vs_ground_truth"),
                    "canonical_dcc": canon_dcc,
                    "canonical_verdict": canon_verdict,
                    "note_data_join_failures": n.get("note_data_join_failures") or [],
                },
            })
            patched_notes.append(n)

        payload["notes"] = patched_notes
        payload["m1_2c_applied"] = True
        payload["open_frequency_audit_verdict"] = ("OPEN_FREQUENCY_NON_DIFFERENTIATING"
                                                    if constant else "OPEN_FREQUENCY_DIFFERENTIATING")
        out_v3 = OUT / f"{target}.site_decision_notes.v3.json"
        out_v3.write_text(json.dumps(payload, indent=2, default=str))

    csv_v3 = OUT / "internal_medchem_review_table.v3.csv"
    with csv_v3.open("w") as f:
        w = csv.DictWriter(f, fieldnames=list(new_csv_rows[0].keys()))
        w.writeheader()
        for r in new_csv_rows:
            w.writerow(r)

    gate_path = OUT / "m1_2c_external_readiness_gate.json"
    summary = {
        "open_frequency_panel_wide_verdict": ("OPEN_FREQUENCY_NON_DIFFERENTIATING"
                                              if constant_panel_wide else "OPEN_FREQUENCY_DIFFERENTIATING"),
        "open_frequency_note_set_verdict": ("OPEN_FREQUENCY_NON_DIFFERENTIATING"
                                             if constant_in_note_set else "OPEN_FREQUENCY_DIFFERENTIATING"),
        "open_frequency_unique_values_panel_wide": sorted(set(all_vals)) if all_vals else [],
        "open_frequency_unique_values_note_set": sorted(set(note_set_vals)) if note_set_vals else [],
        "stripped_from_reasoning": strip_open_freq,
        "n_sites": len(gate_records),
        "internal_medchem_ok_yes": sum(1 for r in gate_records if r["internal_medchem_ok"] == "yes"),
        "external_dossier_ok_yes": sum(1 for r in gate_records if r["external_dossier_ok"] == "yes"),
        "records": gate_records,
    }
    gate_path.write_text(json.dumps(summary, indent=2, default=str))

    print(f"open_frequency_audit_csv : {ofreq_csv}")
    print(f"open_frequency_audit_json: {ofreq_json}")
    print(f"review_table_v3          : {csv_v3}")
    print(f"external_readiness_gate  : {gate_path}")
    print()
    print(f"open_frequency panel-wide verdict: {'OPEN_FREQUENCY_NON_DIFFERENTIATING' if constant_panel_wide else 'OPEN_FREQUENCY_DIFFERENTIATING'}")
    print(f"  panel-wide unique vals         : {sorted(set(all_vals))}")
    print(f"open_frequency note-set verdict  : {'OPEN_FREQUENCY_NON_DIFFERENTIATING' if constant_in_note_set else 'OPEN_FREQUENCY_DIFFERENTIATING'}")
    print(f"  note-set unique vals           : {sorted(set(note_set_vals))}")
    print(f"stripped from reasoning          : {strip_open_freq}")
    print(f"n_sites_internal_ok      : {summary['internal_medchem_ok_yes']}/{summary['n_sites']}")
    print(f"n_sites_external_ok      : {summary['external_dossier_ok_yes']}/{summary['n_sites']}")


if __name__ == "__main__":
    main()
