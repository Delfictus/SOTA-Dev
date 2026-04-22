#!/usr/bin/env python3
"""M1.2 — med-chem decision note composer.

Reads raw values from /tmp/m1_2_raw_values.json and composes target-specific,
site-specific, value-grounded decision notes per the M1.2 contract.

All note sentences must cite specific extracted values (residue ids, spike
counts, volumes, DCC).

Output:
  /tmp/engine_full_profiles/<target>.site_decision_notes.json
  /tmp/engine_full_profiles/internal_medchem_review_table.csv
"""
from __future__ import annotations
import csv
import json
from pathlib import Path

RAW = Path("/tmp/m1_2_raw_values.json")
OUT = Path("/tmp/engine_full_profiles")


# ---------------------------------------------------------------------------
# Per-target biological context — used only to vary framing; every note body
# still injects actual extracted values, so notes are grounded, not templated.
# ---------------------------------------------------------------------------
TARGET_CTX = {
    "wrn_apo":         {"gene": "WRN",         "family": "helicase",        "holo": "8PFO / HRO761", "key_phrase": "RecQ helicase allosteric landscape"},
    "menin_apo":       {"gene": "MEN1",        "family": "scaffold",        "holo": "7UJ4 / OQ4",    "key_phrase": "Menin-MLL interaction face"},
    "smarca2_brd_apo": {"gene": "SMARCA2",     "family": "bromodomain",     "holo": "5DKC / 5BW",    "key_phrase": "SMARCA2 bromodomain acetyl-lysine pocket"},
    "pkmyt1_apo":      {"gene": "PKMYT1",      "family": "kinase",          "holo": "8D6E / QGI",    "key_phrase": "PKMYT1 ATP pocket / gatekeeper region"},
    "kras_g12d_apo":   {"gene": "KRAS-G12D",   "family": "GTPase",          "holo": "7RPZ / 6IC",    "key_phrase": "KRAS-G12D switch-II / MRTX-1133 site"},
    "m1_2nvp":         {"gene": "2NVP/3QT9",   "family": "CB pair (Z4Y)",   "holo": "3QT9 / Z4Y",    "key_phrase": "CryptoBench Z4Y pocket"},
    "m1_1xhx":         {"gene": "1XHX/2PYJ",   "family": "CB pair (DGT)",   "holo": "2PYJ / DGT",    "key_phrase": "CryptoBench DGT pocket"},
}


def f(x, nd=2):
    if isinstance(x, (int, float)):
        return f"{x:.{nd}f}"
    return str(x)


def centroid_str(c):
    if not c or len(c) != 3:
        return "FEATURE_GAP:centroid"
    return f"({c[0]:.2f}, {c[1]:.2f}, {c[2]:.2f}) Å (apo frame)"


def volume_band(v):
    if v is None:
        return "unknown volume"
    if v < 150:
        return "tight cavity (volume < 150 Å³; fragment-scale)"
    if v < 500:
        return "compact cavity (volume 150–500 Å³; drug-scale)"
    if v < 1500:
        return "medium cavity (volume 500–1500 Å³; drug/fragment blend)"
    return "large/open region (volume > 1500 Å³; likely surface-exposed)"


def phase_bias(h, c):
    if h is None or c is None:
        return "phase-bias data absent"
    if (h + c) == 0:
        return "zero phase-resolved spikes"
    ratio = h / (h + c) if (h + c) > 0 else 0.5
    if ratio > 0.65:
        return f"heating-biased ({h:,} heating vs {c:,} cooling spikes; heating fraction {ratio:.2f})"
    if ratio < 0.35:
        return f"cooling-biased ({h:,} heating vs {c:,} cooling spikes; heating fraction {ratio:.2f})"
    return f"balanced heating/cooling ({h:,} vs {c:,}; heating fraction {ratio:.2f})"


def residue_str(ids, n=8):
    if not ids:
        return "FEATURE_GAP:top_residue_ids"
    sub = ids[:n]
    more = len(ids) - len(sub)
    suffix = f" +{more} more" if more > 0 else ""
    return ", ".join(str(r) for r in sub) + suffix


def lining_str(lr, n=5):
    if not lr:
        return "FEATURE_GAP:lining_residues"
    out = []
    for r in lr[:n]:
        rid = r.get("resid", "?")
        rn = r.get("resname", "?")
        d = r.get("min_distance")
        tag = " (catalytic)" if r.get("is_catalytic") else ""
        out.append(f"{rn}{rid}{tag}@{d:.1f}Å" if isinstance(d, (int, float)) else f"{rn}{rid}{tag}")
    return "; ".join(out)


def confidence_state(site):
    v = site.get("canonical_verdict")
    is_bm = site.get("is_best_match_vs_ground_truth")
    target = site.get("target")
    # GT invalid / no GT overrides
    # (we don't have direct access here; approximate by missing DCC)
    if site.get("canonical_dcc_angstrom") is None:
        # could be GT_INVALID or unavailable eval
        return "MECHANISTIC_SUPPORT_ONLY"
    if v == "PASS":
        return "VERIFIED_LIGAND_SITE_SUPPORT" if is_bm else "VERIFIED_DETECTOR_SUPPORT_ONLY"
    if v in ("HARD_NOT_MATCH", "NOT_MATCH", "FAIL"):
        return "MECHANISTIC_SUPPORT_ONLY"
    return "LOW_PRIORITY_SIGNAL"


def real_world_framing(site):
    """Return (framing_key, justification_clause) using actual extracted values."""
    v = site.get("volume_angstrom_cubed") or 0
    drug = site.get("druggability_score") or 0
    canon = site.get("canonical_verdict")
    is_bm = site.get("is_best_match_vs_ground_truth")
    therm = (site.get("therm_class") or "").upper()
    class_ = site.get("classification") or ""
    sp_pri = site.get("primary_residue_id")
    kc_conf = site.get("kcc_confidence")
    max_rec = site.get("max_recurrence")

    vals = []
    if isinstance(v, (int, float)): vals.append(f"volume={v:.0f} Å³")
    if isinstance(drug, (int, float)): vals.append(f"druggability={drug:.2f}")
    if isinstance(kc_conf, (int, float)): vals.append(f"kcc_confidence={kc_conf:.2f}")
    if isinstance(max_rec, (int, float)): vals.append(f"max_recurrence={max_rec:.2f}")
    if sp_pri is not None: vals.append(f"primary_residue_id={sp_pri}")
    values_phrase = "; ".join(vals) if vals else "FEATURE_GAP:metrics"

    if canon == "PASS" and is_bm:
        return ("fragment-screening priority",
                f"top-1 aligns to the verified best-match site under canonical DCC; "
                f"{values_phrase}. Compact enough to support fragment-growth chemistry.")
    if canon == "PASS":
        return ("orthosteric backup / secondary pocket",
                f"canonical DCC places top-1 within the ligand-binding band, "
                f"but it is not the verified best-match; {values_phrase}.")
    if canon == "HARD_NOT_MATCH":
        if v > 1500:
            return ("deprioritize for chemistry, retain for mechanism",
                    f"canonical DCC {site.get('canonical_dcc_angstrom')} Å places this far from the verified ligand site; "
                    f"{values_phrase} indicates a large/open region, poor primary chemistry target.")
        return ("mutagenesis priority",
                f"not a chemistry target (canonical HARD_NOT_MATCH at "
                f"{site.get('canonical_dcc_angstrom')} Å) but residues dominate a local spike cluster; "
                f"{values_phrase}. Use to probe mechanism via mutation.")
    if canon == "NOT_MATCH":
        if v < 200 and drug > 0.85:
            return ("allosteric exploration candidate",
                    f"compact + druggable but away from verified site ({site.get('canonical_dcc_angstrom')} Å); "
                    f"{values_phrase}. Could be a non-orthosteric site worth independent probing.")
        return ("retain for mechanism, not chemistry",
                f"canonical DCC {site.get('canonical_dcc_angstrom')} Å outside ligand-site band; "
                f"{values_phrase}. Retain for mechanism only.")
    # FAIL (between 8–20 Å) — near-miss
    return ("secondary triage for docking",
            f"canonical DCC {site.get('canonical_dcc_angstrom')} Å is close to the ligand-site band but outside "
            f"the 8 Å cutoff; {values_phrase}. Use as a secondary site in docking triage.")


def next_action(framing_key):
    mapping = {
        "fragment-screening priority":        "include in fragment screen shortlist",
        "orthosteric backup / secondary pocket": "include as secondary site in docking triage",
        "deprioritize for chemistry, retain for mechanism": "deprioritize pending validation",
        "mutagenesis priority":               "design residue mutation panel",
        "allosteric exploration candidate":   "compare against known ligand site",
        "retain for mechanism, not chemistry": "deprioritize pending validation",
        "secondary triage for docking":       "include as secondary site in docking triage",
    }
    return mapping.get(framing_key, "inspect in structure viewer")


def med_chem_summary(site, ctx):
    v = site.get("volume_angstrom_cubed") or 0
    band = volume_band(v)
    therm = (site.get("therm_class") or "?").upper()
    canon = site.get("canonical_verdict") or "UNVERIFIED"
    canon_dcc = site.get("canonical_dcc_angstrom")
    is_bm = site.get("is_best_match_vs_ground_truth")

    # verb phrase
    if canon == "PASS" and is_bm:
        body = (f"This site matches the verified {ctx['holo']} ligand pocket "
                f"(canonical DCC {canon_dcc} Å). "
                f"Marked {therm} by the engine, classified {site.get('classification')}; "
                f"{band}. ")
        caution = f"Primary signal is structural agreement with the holo ligand; follow-up in a fragment workflow is warranted, but ligandability should still be validated in an assay before committing chemistry resource."
    elif canon == "PASS":
        body = (f"This top-1 ranked site falls within the ligand-binding DCC band ({canon_dcc} Å) "
                f"but is not the verified best-match pocket. "
                f"Engine class {therm}; {band}. ")
        caution = f"Useful as a secondary pocket for docking triage; do not treat as an independent binding-site claim without further evidence."
    elif canon in ("HARD_NOT_MATCH", "NOT_MATCH"):
        body = (f"This {therm}-labelled site sits {canon_dcc} Å from the verified {ctx['holo']} ligand pocket — "
                f"outside the ligand-binding band. "
                f"{band}. Signal is mechanistic, not ligand-validated. ")
        caution = (f"The engine's CRYPTIC/DYNAMIC label is a mechanical-response signal here, not a confirmed "
                   f"ligand pocket. Any ligandability claim must be validated independently.")
    elif canon == "FAIL":
        body = (f"Near-miss against the verified site ({canon_dcc} Å outside the 8 Å gate); "
                f"{band}; engine class {therm}. ")
        caution = f"Close enough for docking triage, not close enough to call a match. Treat as secondary."
    else:
        body = (f"Canonical DCC is unavailable for this target; "
                f"engine class {therm}; {band}. ")
        caution = f"No ligand-site verification yet. Do not use as a ligandability claim."

    return body + caution


def residue_triage(site, ctx):
    top = site.get("top_residue_ids") or []
    lining = site.get("lining_residues_top5") or []
    tide = site.get("tide_trigger_residues_top5")
    if not (top or lining):
        return "FEATURE_GAP:residue_triage_inputs"
    top_s = residue_str(top, 8)
    lining_s = lining_str(lining, 5)
    tide_s = (residue_str(tide, 5) if isinstance(tide, list) else tide) if tide else "FEATURE_GAP:tide_trigger_residues"
    pri = site.get("primary_residue_id")
    kc_drv = site.get("kcc_driver_residue_id")
    return (f"Top-attribution residues under B_no_therm: {top_s} (spike attribution ordering). "
            f"Top lining contacts (Cα-to-centroid): {lining_s}. "
            f"TIDE trigger residues (sample): {tide_s}. "
            f"signal_preservation.primary_residue_id = {pri}; kcc driver field = {kc_drv}. "
            f"For {ctx['gene']} chemistry, treat the overlap of top-attribution and primary_residue_id as the "
            f"first residues to mutate or probe; lining residues define the minimal interaction patch for a "
            f"pharmacophore sketch.")


def dynamic_note(site, ctx):
    h = site.get("heating_spike_count")
    c = site.get("cooling_spike_count")
    nsp = site.get("n_spikes_attributed")
    max_rec = site.get("max_recurrence")
    tot_rec = site.get("total_recurrence")
    conc = site.get("residue_concentration")
    kc_conf = site.get("kcc_confidence")
    mot = site.get("site_motion_efficiency")
    phase = phase_bias(h, c)
    bits = [f"n_spikes_attributed = {nsp:,}" if isinstance(nsp, int) else f"n_spikes_attributed = {nsp}"]
    bits.append(phase)
    if isinstance(max_rec, (int, float)): bits.append(f"max_recurrence = {max_rec:.3f}")
    if isinstance(tot_rec, (int, float)): bits.append(f"total_recurrence = {tot_rec:.3f}")
    if isinstance(conc, (int, float)): bits.append(f"residue_concentration = {conc:.3f}")
    if isinstance(kc_conf, (int, float)): bits.append(f"kcc_confidence = {kc_conf:.2f}")
    if isinstance(mot, (int, float)): bits.append(f"site_motion_efficiency = {mot:.3f}")
    caution = (" CAUTION: these are mechanical-response metrics (thermal protocol + causal lag), not "
               "ligand-binding evidence. Any interpretation as ligand activity must be verified independently.")
    return "; ".join(bits) + "." + caution


def spatial_note(site, ctx):
    c = centroid_str(site.get("centroid_spike_weighted"))
    v = site.get("volume_angstrom_cubed")
    ray = site.get("ray_escape_ratio")
    burial = site.get("mean_burial") or site.get("burial_score")
    sph = site.get("sphericity")
    srcd = site.get("source_diversity")
    bits = [f"centroid_spike_weighted = {c}"]
    if isinstance(v, (int, float)): bits.append(f"volume = {v:.0f} Å³")
    if isinstance(ray, (int, float)): bits.append(f"ray_escape_ratio = {ray:.3f}")
    if isinstance(burial, (int, float)): bits.append(f"burial = {burial:.3f}")
    if isinstance(sph, (int, float)): bits.append(f"sphericity = {sph:.3f}")
    if isinstance(srcd, (int, float)): bits.append(f"source_diversity = {srcd:.2f}")
    inspect_hint = ("Open the apo PDB, translate to the centroid in PyMOL/ChimeraX, and inspect whether the "
                    "lining residues form a contiguous enclosed wall or a shallow surface groove. "
                    f"Context: {ctx['key_phrase']}.")
    return "; ".join(bits) + ". " + inspect_hint


def compose_site_notes(site, ctx):
    raw = {k: v for k, v in site.items()}
    framing_key, justification = real_world_framing(site)
    notes = {
        "target": site["target"],
        "site_id": site["site_id"],
        "raw_value_block": raw,
        "med_chem_summary": med_chem_summary(site, ctx),
        "residue_triage_note": residue_triage(site, ctx),
        "dynamic_mechanistic_note": dynamic_note(site, ctx),
        "spatial_review_note": spatial_note(site, ctx),
        "real_world_value_note": f"Primary framing: {framing_key}. {justification}",
        "next_action": next_action(framing_key),
        "confidence_state": confidence_state(site),
        "primary_value_framing": framing_key,
    }
    return notes


def quality_gate(note):
    fails = []
    body = " ".join([
        note["med_chem_summary"],
        note["residue_triage_note"],
        note["dynamic_mechanistic_note"],
        note["spatial_review_note"],
        note["real_world_value_note"],
    ])
    if "FEATURE_GAP:residue_triage_inputs" in note["residue_triage_note"]:
        fails.append("missing residue_triage_inputs")
    if not note["next_action"]:
        fails.append("missing next_action")
    if "CAUTION" not in note["dynamic_mechanistic_note"] and note["confidence_state"] != "VERIFIED_LIGAND_SITE_SUPPORT":
        fails.append("missing caution")
    # unsupported ligand claim check
    banned = ["clinically actionable", "guaranteed ligandable", "proven allosteric",
              "confirmed cryptic binder", "pharma-grade winner"]
    for b in banned:
        if b.lower() in body.lower():
            fails.append(f"banned_phrase:{b}")
    return fails


def main():
    data = json.loads(RAW.read_text())
    rows = []
    failures = []
    for target, payload in data.items():
        ctx = TARGET_CTX[target]
        notes = []
        for site in payload["sites"]:
            n = compose_site_notes(site, ctx)
            q = quality_gate(n)
            if q:
                print(f"NOTE_QUALITY_FAIL:{target}:{site['site_id']}  reasons={q}")
                failures.append((target, site["site_id"], q))
            notes.append(n)
            rows.append({
                "target": target,
                "site_id": site["site_id"],
                "rank": site.get("current_reference_rank_under_B_no_therm"),
                "verified_best_match": site.get("is_best_match_vs_ground_truth"),
                "confidence_state": n["confidence_state"],
                "top_residues": str((site.get("top_residue_ids") or [])[:8]),
                "volume": site.get("volume_angstrom_cubed"),
                "n_spikes_attributed": site.get("n_spikes_attributed"),
                "heating_vs_cooling": f"{site.get('heating_spike_count')}/{site.get('cooling_spike_count')}",
                "centroid_spike_weighted": str(site.get("centroid_spike_weighted")),
                "primary_value_framing": n["primary_value_framing"],
                "next_action": n["next_action"],
            })
        out_path = OUT / f"{target}.site_decision_notes.json"
        out_path.write_text(json.dumps({
            "target": target,
            "reference_variant": "B_no_therm",
            "best_match_site_id": payload.get("best_match_site_id"),
            "evaluation_verdict_tag": payload.get("evaluation_verdict_tag"),
            "note_count": len(notes),
            "notes": notes,
        }, indent=2, default=str))

    csv_path = OUT / "internal_medchem_review_table.csv"
    with csv_path.open("w") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows:
            w.writerow(r)

    print()
    print(f"site_decision_notes: {OUT}/<target>.site_decision_notes.json (7 files)")
    print(f"review_table: {csv_path}")
    print(f"n_sites_total: {len(rows)}")
    print(f"n_failures: {len(failures)}")


if __name__ == "__main__":
    main()
