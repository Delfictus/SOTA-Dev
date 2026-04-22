#!/usr/bin/env python3
"""M1.2b — med-chem note hygiene + action gating patcher.

Reads the existing /tmp/engine_full_profiles/<target>.site_decision_notes.json
and emits <target>.site_decision_notes.v2.json after:

  1. Data-join repair:
       * pulls open_frequency from per-site *.site{N}.spike_events.json
       * pulls kcc.driver_residue_id, candidate_residue_ids, candidate_kcc_confidence
         from binding_sites.sites[*].kcc
       * removes array-typed false "driver_residue_id" (was candidate_kcc_causal_lag)
  2. Pre-fix therm quarantine on WRN/MENIN/SMARCA2/PKMYT1/KRAS:
       * therm_class, hysteresis_asymmetry, relative_asymmetry tagged
         PRE_FIX_NOT_MEDCHEM_SAFE; stripped from interpretation prose.
  3. Action gating by canonical DCC band (LE8 / 8to15 / 15to25 / GT25_OR_HARD_NOT_MATCH).
  4. Rewritten real_world_value_note (chemistry/mechanism/assay/budget).
  5. Specificity gate: must cite ≥3 actual target/site values.
  6. New output fields: therm_fields_medchem_safe, action_gate_band, budget_label,
     note_data_join_failures, specificity_pass.

Writes also internal_medchem_review_table.v2.csv.
"""
from __future__ import annotations
import csv
import json
import re
from pathlib import Path

OUT = Path("/tmp/engine_full_profiles")

PRE_FIX_THERM_QUARANTINE_TARGETS = {
    "wrn_apo", "menin_apo", "smarca2_brd_apo", "pkmyt1_apo", "kras_g12d_apo",
}

# Band → allowed primary framings
BAND_FRAMING = {
    "LE8": [
        "fragment-screening priority",
        "orthosteric backup / secondary pocket",
        "allosteric exploration candidate",
        "assay-design support",
    ],
    "8to15": [
        "secondary triage for docking",
        "mutagenesis priority",
        "assay-design support",
        "deprioritize for chemistry, retain for mechanism",
    ],
    "15to25": [
        "mutagenesis priority",
        "mechanism-only follow-up",
        "deprioritize for chemistry, retain for mechanism",
    ],
    "GT25_OR_HARD_NOT_MATCH": [
        "deprioritize for chemistry, retain for mechanism",
        "interface-mapping support",
        "mutagenesis priority",
    ],
}

BAND_NEXT_ACTION = {
    "LE8": {
        "fragment-screening priority": "include in fragment screen shortlist",
        "orthosteric backup / secondary pocket": "include as secondary site in docking triage",
        "allosteric exploration candidate": "compare against known ligand site",
        "assay-design support": "add to assay panel",
    },
    "8to15": {
        "secondary triage for docking": "include as secondary site in docking triage",
        "mutagenesis priority": "design residue mutation panel",
        "assay-design support": "add to assay panel",
        "deprioritize for chemistry, retain for mechanism": "deprioritize pending validation",
    },
    "15to25": {
        "mutagenesis priority": "design residue mutation panel",
        "mechanism-only follow-up": "deprioritize pending validation",
        "deprioritize for chemistry, retain for mechanism": "deprioritize pending validation",
    },
    "GT25_OR_HARD_NOT_MATCH": {
        "deprioritize for chemistry, retain for mechanism": "deprioritize pending validation",
        "interface-mapping support": "review interface adjacency",
        "mutagenesis priority": "design residue mutation panel",
    },
}

# Target context (for diversity in prose)
TARGET_CTX = {
    "wrn_apo":         {"gene": "WRN",     "holo": "8PFO / HRO761"},
    "menin_apo":       {"gene": "MEN1",    "holo": "7UJ4 / OQ4"},
    "smarca2_brd_apo": {"gene": "SMARCA2", "holo": "5DKC / 5BW"},
    "pkmyt1_apo":      {"gene": "PKMYT1",  "holo": "8D6E / QGI"},
    "kras_g12d_apo":   {"gene": "KRAS-G12D","holo": "7RPZ / 6IC"},
    "m1_2nvp":         {"gene": "2NVP",    "holo": "3QT9 / Z4Y"},
    "m1_1xhx":         {"gene": "1XHX",    "holo": "2PYJ / DGT"},
}


def action_gate_band(canon_dcc, canon_verdict):
    if canon_verdict == "HARD_NOT_MATCH":
        return "GT25_OR_HARD_NOT_MATCH"
    if canon_dcc is None:
        return "GT25_OR_HARD_NOT_MATCH"
    if canon_dcc <= 8.0:
        return "LE8"
    if canon_dcc <= 15.0:
        return "8to15"
    if canon_dcc <= 25.0:
        return "15to25"
    return "GT25_OR_HARD_NOT_MATCH"


def budget_label(band, is_best_match):
    if band == "GT25_OR_HARD_NOT_MATCH":
        return "MECHANISM_ONLY"
    if band == "15to25":
        return "LOW_ATTENTION"
    if band == "8to15":
        return "MEDIUM_ATTENTION" if is_best_match else "LOW_ATTENTION"
    if band == "LE8":
        return "HIGH_ATTENTION" if is_best_match else "MEDIUM_ATTENTION"
    return "MECHANISM_ONLY"


def pick_framing(band, raw, is_best_match):
    """Select one allowed primary framing for the band using actual raw values."""
    allowed = BAND_FRAMING[band]
    v = raw.get("volume_angstrom_cubed") or 0
    drug = raw.get("druggability_score") or 0
    if band == "LE8":
        if is_best_match and v < 1500:
            return "fragment-screening priority"
        if v < 1500:
            return "orthosteric backup / secondary pocket"
        return "assay-design support"
    if band == "8to15":
        if is_best_match:
            return "secondary triage for docking"
        if v > 1500:
            return "deprioritize for chemistry, retain for mechanism"
        if v < 500 and drug > 0.8:
            return "mutagenesis priority"
        return "secondary triage for docking"
    if band == "15to25":
        if v > 1500:
            return "deprioritize for chemistry, retain for mechanism"
        return "mutagenesis priority"
    # GT25_OR_HARD_NOT_MATCH
    if v > 1500:
        return "deprioritize for chemistry, retain for mechanism"
    return "mutagenesis priority"


def resolve_open_frequency(target, site_id):
    """Read per-site spike_events.json top-level open_frequency (partial parse)."""
    for root, stem in [
        ("/mnt/storage/prism-outputs/twin-10-patent/wrn_apo", "6yhr"),
        ("/mnt/storage/prism-outputs/twin-10-patent/menin_apo", "3re2"),
        ("/mnt/storage/prism-outputs/twin-10-patent/smarca2_brd_apo", "4qy4"),
        ("/mnt/storage/prism-outputs/twin-10-patent/pkmyt1_apo", "3p1a"),
        ("/mnt/storage/prism-outputs/twin-10-patent/kras_g12d_apo", "7f0w"),
        ("/mnt/storage/prism-outputs/m1-strict-dcc-panel/m1_2nvp", "2nvp"),
        ("/mnt/storage/prism-outputs/m1-strict-dcc-panel/m1_1xhx", "1xhx"),
    ]:
        base = Path(root)
        if base.name != target:
            continue
        p = base / f"artifacts/5_engine/{stem}.site{site_id}.spike_events.json"
        if not p.exists():
            return None
        try:
            with p.open("r") as f:
                head = f.read(4096)
            m = re.search(r'"open_frequency"\s*:\s*([\-\d\.eE]+)', head)
            if m:
                return float(m.group(1))
        except Exception:
            return None
        return None
    return None


def resolve_kcc_subtree(target, site_id):
    """Pull full kcc subtree from binding_sites.sites[site_id].kcc."""
    mapping = {
        "wrn_apo": ("/mnt/storage/prism-outputs/twin-10-patent/wrn_apo/artifacts/5_engine/6yhr.binding_sites.json"),
        "menin_apo": ("/mnt/storage/prism-outputs/twin-10-patent/menin_apo/artifacts/5_engine/3re2.binding_sites.json"),
        "smarca2_brd_apo": ("/mnt/storage/prism-outputs/twin-10-patent/smarca2_brd_apo/artifacts/5_engine/4qy4.binding_sites.json"),
        "pkmyt1_apo": ("/mnt/storage/prism-outputs/twin-10-patent/pkmyt1_apo/artifacts/5_engine/3p1a.binding_sites.json"),
        "kras_g12d_apo": ("/mnt/storage/prism-outputs/twin-10-patent/kras_g12d_apo/artifacts/5_engine/7f0w.binding_sites.json"),
        "m1_2nvp": ("/mnt/storage/prism-outputs/m1-strict-dcc-panel/m1_2nvp/artifacts/5_engine/2nvp.binding_sites.json"),
        "m1_1xhx": ("/mnt/storage/prism-outputs/m1-strict-dcc-panel/m1_1xhx/artifacts/5_engine/1xhx.binding_sites.json"),
    }
    p = mapping.get(target)
    if not p:
        return {}
    try:
        d = json.loads(Path(p).read_text())
        for s in d.get("sites") or []:
            if s.get("id") == site_id:
                return s.get("kcc") or {}
    except Exception:
        return {}
    return {}


def centroid_str(c):
    if not c or len(c) != 3:
        return "FEATURE_GAP:centroid"
    return f"({c[0]:.2f}, {c[1]:.2f}, {c[2]:.2f}) Å"


def residue_list_short(ids, n=8):
    if not ids:
        return "[]"
    sub = ids[:n]
    more = len(ids) - len(sub)
    return "[" + ", ".join(str(r) for r in sub) + (f", …+{more}]" if more else "]")


def build_medchem_summary(target, site, band, budget, is_bm):
    """Target-specific summary without pre-fix therm language for quarantined targets."""
    canon = site.get("canonical_dcc_angstrom")
    cv = site.get("canonical_verdict")
    v = site.get("volume_angstrom_cubed") or 0
    ctx = TARGET_CTX[target]
    volume_phrase = (
        f"compact cavity (volume {v:.0f} Å³)" if v < 500
        else f"medium cavity (volume {v:.0f} Å³)" if v < 1500
        else f"large region (volume {v:.0f} Å³; likely surface-exposed)"
    )

    if band == "LE8" and is_bm:
        body = (f"This site is spatially verified against the {ctx['holo']} ligand pocket "
                f"(canonical DCC {canon} Å). "
                f"{volume_phrase}. ")
        caution = (f"Budget label {budget}; proceed cautiously — structural match at this DCC does not "
                   f"yet guarantee chemistry tractability until assay support lands.")
    elif band == "LE8":
        body = (f"Top-ranked site under B_no_therm is within the ligand-binding DCC band "
                f"({canon} Å) but is not the verified best-match pocket. "
                f"{volume_phrase}. ")
        caution = f"Budget {budget}; use as a secondary pocket reference. Ligandability independent of best-match requires independent evidence."
    elif band == "8to15":
        body = (f"Near-miss site: canonical DCC {canon} Å is adjacent to, not within, the "
                f"{ctx['holo']} ligand-binding band. "
                f"{volume_phrase}. ")
        caution = f"Budget {budget}; suitable for secondary triage only, not primary chemistry."
    elif band == "15to25":
        body = (f"Site sits {canon} Å from the verified {ctx['holo']} ligand pocket — outside the "
                f"ligand-binding band but within the same domain envelope. "
                f"{volume_phrase}. ")
        caution = f"Budget {budget}; any signal here is mechanistic, not chemistry-ready."
    else:  # GT25_OR_HARD_NOT_MATCH
        body = (f"Site is far from the verified {ctx['holo']} ligand pocket "
                f"(canonical DCC {canon} Å; {cv or 'HARD_NOT_MATCH'}). "
                f"{volume_phrase}. ")
        caution = f"Budget {budget}; mechanism-only. No chemistry or docking action on this site."
    return body + caution


def build_residue_triage(target, site, kcc):
    top = site.get("top_residue_ids") or []
    lining = site.get("lining_residues_top5") or []
    tide = site.get("tide_trigger_residues_top5")
    driver = kcc.get("driver_residue_id")
    candidates = kcc.get("candidate_residue_ids") or []
    cand_conf = kcc.get("candidate_kcc_confidence") or []
    pri_id = site.get("primary_residue_id")
    pri_count = site.get("primary_residue_count")

    tide_s = residue_list_short(tide, 5) if isinstance(tide, list) else "FEATURE_GAP:tide_trigger_residues"
    lining_s = "; ".join([
        f"{r.get('resname','?')}{r.get('resid','?')}"
        f"{' (catalytic)' if r.get('is_catalytic') else ''}"
        f"@{r.get('min_distance',0):.1f}Å"
        for r in lining[:5]
    ]) if lining else "FEATURE_GAP:lining_residues"

    driver_s = f"driver_residue_id={driver}" if isinstance(driver, int) else "driver_residue_id=FEATURE_GAP:valid_kcc_driver_residue_id"
    cand_pairs = []
    for i, rid in enumerate(candidates[:3]):
        conf = cand_conf[i] if i < len(cand_conf) else None
        if isinstance(conf, (int, float)):
            cand_pairs.append(f"{rid}(conf={conf:.2f})")
        else:
            cand_pairs.append(f"{rid}")
    cand_s = ", ".join(cand_pairs) if cand_pairs else "FEATURE_GAP:candidate_residue_ids"

    ctx = TARGET_CTX[target]
    return (f"Top-attribution residues: {residue_list_short(top, 8)}. "
            f"Lining contacts (Cα-to-centroid): {lining_s}. "
            f"TIDE triggers (sample): {tide_s}. "
            f"KCC {driver_s}, candidate residues: {cand_s}. "
            f"signal_preservation.primary_residue_id={pri_id} (count={pri_count}). "
            f"For {ctx['gene']} triage, prioritise residues where top-attribution, KCC driver, and "
            f"primary_residue_id overlap — that is the minimum residue panel worth perturbing first.")


def build_dynamic_note(target, site, open_freq, is_therm_safe):
    h = site.get("heating_spike_count")
    c = site.get("cooling_spike_count")
    nsp = site.get("n_spikes_attributed")
    max_rec = site.get("max_recurrence")
    tot_rec = site.get("total_recurrence")
    conc = site.get("residue_concentration")
    kc_conf = site.get("kcc_confidence")
    mot = site.get("site_motion_efficiency")
    lag = site.get("site_lag_corr_peak")
    temp = site.get("temporal_corr")
    bits = []
    if isinstance(nsp, int):
        bits.append(f"n_spikes_attributed={nsp:,}")
    if isinstance(h, int) and isinstance(c, int):
        total = h + c
        if total > 0:
            frac = h / total
            bits.append(f"heating/cooling = {h:,}/{c:,} (heating fraction {frac:.2f})")
    if isinstance(max_rec, (int, float)):
        bits.append(f"max_recurrence={max_rec}")
    if isinstance(tot_rec, (int, float)):
        bits.append(f"total_recurrence={tot_rec}")
    if isinstance(conc, (int, float)):
        bits.append(f"residue_concentration={conc:.3f}")
    if isinstance(kc_conf, (int, float)):
        bits.append(f"kcc_confidence={kc_conf:.2f}")
    if isinstance(mot, (int, float)):
        bits.append(f"site_motion_efficiency={mot:.3f}")
    if isinstance(lag, (int, float)):
        bits.append(f"site_lag_corr_peak={lag:.3f}")
    if isinstance(temp, (int, float)):
        bits.append(f"temporal_corr={temp:.3f}")
    if isinstance(open_freq, (int, float)):
        bits.append(f"open_frequency={open_freq}")

    if is_therm_safe:
        frame = ("Spike-distribution and KCC/coherence signals: " + "; ".join(bits))
    else:
        frame = ("PRE_FIX_NOT_MEDCHEM_SAFE — therm_class / hysteresis_asymmetry / relative_asymmetry "
                 "suppressed. Remaining geometry, spike-distribution, KCC and signal-preservation metrics: "
                 + "; ".join(bits))
    return frame + ". CAUTION: these are mechanical-response / causal-lag metrics; do not interpret as ligand activity without independent evidence."


def build_spatial_note(target, site):
    c = centroid_str(site.get("centroid_spike_weighted"))
    v = site.get("volume_angstrom_cubed")
    ray = site.get("ray_escape_ratio")
    burial = site.get("mean_burial") or site.get("burial_score")
    sph = site.get("sphericity")
    srcd = site.get("source_diversity")
    bits = [f"centroid_spike_weighted={c}"]
    if isinstance(v, (int, float)): bits.append(f"volume={v:.0f} Å³")
    if isinstance(ray, (int, float)): bits.append(f"ray_escape_ratio={ray:.3f}")
    if isinstance(burial, (int, float)): bits.append(f"mean_burial={burial:.3f}")
    if isinstance(sph, (int, float)): bits.append(f"sphericity={sph:.3f}")
    if isinstance(srcd, (int, float)): bits.append(f"source_diversity={srcd:.2f}")
    ctx = TARGET_CTX[target]
    hint = (f"Open {ctx['gene']} apo structure, translate view to the centroid, inspect whether the "
            f"lining residues form a contiguous enclosed wall or a shallow groove open to solvent.")
    return "; ".join(bits) + ". " + hint


def build_real_world(target, site, band, budget, framing, kcc, is_bm):
    """Chemistry value + mechanism value + assay value + budget."""
    v = site.get("volume_angstrom_cubed") or 0
    drug = site.get("druggability_score") or 0
    nsp = site.get("n_spikes_attributed")
    canon = site.get("canonical_dcc_angstrom")
    pri = site.get("primary_residue_id")
    driver = kcc.get("driver_residue_id")

    if band == "LE8":
        chem = (f"Chemistry: allowable. Volume {v:.0f} Å³ + druggability {drug:.2f} + canonical DCC {canon} Å "
                f"support {framing}.")
        mech = f"Mechanism: site overlaps or is adjacent to the verified holo pocket; residue triage includes kcc_driver {driver} and primary_residue_id {pri}."
        assay = f"Assay: site {'verified' if is_bm else 'adjacent'} — supports competitive-binding assay design against the known ligand."
    elif band == "8to15":
        chem = (f"Chemistry: limited. DCC {canon} Å is outside the ligand-band — {framing} is the only defensible action.")
        mech = f"Mechanism: residue triage (driver={driver}, primary_residue={pri}, {nsp:,} spikes) useful for mutagenesis panel, not chemistry."
        assay = f"Assay: low-yield — consider orthogonal assay only if mutagenesis reveals a functional residue overlap."
    elif band == "15to25":
        chem = (f"Chemistry: not warranted. Site sits {canon} Å from the verified pocket; volume {v:.0f} Å³.")
        mech = f"Mechanism: this is where the site has value — {framing} via mutation of driver_residue {driver} / primary_residue {pri}."
        assay = f"Assay: no direct assay design from this site; use only as a mechanism control."
    else:  # GT25_OR_HARD_NOT_MATCH
        chem = (f"Chemistry: blocked. HARD_NOT_MATCH (DCC {canon} Å). Do not docking-triage this site.")
        mech = f"Mechanism: if pursued, {framing} via residues around driver={driver} / primary={pri}."
        assay = f"Assay: none directly; only retain as a mechanism-only observation."
    return f"{chem}  {mech}  {assay}  budget_label={budget}."


def specificity_count(notes, site, kcc, open_freq):
    """Count how many of the required actual values are cited anywhere in the notes."""
    body = " ".join([
        notes.get("med_chem_summary", ""),
        notes.get("residue_triage_note", ""),
        notes.get("dynamic_mechanistic_note", ""),
        notes.get("spatial_review_note", ""),
        notes.get("real_world_value_note", ""),
    ])
    hits = 0
    canon = site.get("canonical_dcc_angstrom")
    if canon is not None and f"{canon}" in body:
        hits += 1
    v = site.get("volume_angstrom_cubed") or 0
    if v and f"{v:.0f}" in body:
        hits += 1
    top = site.get("top_residue_ids") or []
    if top and str(top[0]) in body:
        hits += 1
    lining = site.get("lining_residues_top5") or []
    if lining and lining[0].get("resid") and str(lining[0]["resid"]) in body:
        hits += 1
    nsp = site.get("n_spikes_attributed")
    if nsp and f"{nsp:,}" in body:
        hits += 1
    h = site.get("heating_spike_count")
    if h and f"{h:,}" in body:
        hits += 1
    c = site.get("centroid_spike_weighted")
    if c and len(c) == 3 and f"{c[0]:.2f}" in body:
        hits += 1
    pri = site.get("primary_residue_id")
    if pri and f"primary_residue_id={pri}" in body:
        hits += 1
    kc_conf = site.get("kcc_confidence")
    if isinstance(kc_conf, (int, float)) and f"kcc_confidence={kc_conf:.2f}" in body:
        hits += 1
    if isinstance(open_freq, (int, float)) and f"open_frequency={open_freq}" in body:
        hits += 1
    mr = site.get("mean_recurrence")
    if isinstance(mr, (int, float)) and f"mean_recurrence" in body:
        hits += 1
    return hits


def main():
    all_rows = []
    data_join_fails = []
    specificity_fails = []

    for target in ["wrn_apo", "menin_apo", "smarca2_brd_apo", "pkmyt1_apo",
                   "kras_g12d_apo", "m1_2nvp", "m1_1xhx"]:
        src = OUT / f"{target}.site_decision_notes.json"
        d = json.loads(src.read_text())
        new_notes = []
        is_therm_safe = target not in PRE_FIX_THERM_QUARANTINE_TARGETS

        for old in d["notes"]:
            site_id = old["site_id"]
            raw = old["raw_value_block"]

            # ── Data-join repair ──
            kcc = resolve_kcc_subtree(target, site_id)
            open_freq = resolve_open_frequency(target, site_id)

            join_fails = []
            if open_freq is None:
                # completeness cert says it's present → genuine join fail
                join_fails.append(f"open_frequency@site{site_id}")
                data_join_fails.append(f"NOTE_DATA_JOIN_FAIL:{target}:{site_id}:open_frequency")
            if not kcc.get("driver_residue_id") or not isinstance(kcc.get("driver_residue_id"), int):
                join_fails.append(f"valid_kcc_driver_residue_id@site{site_id}")
            if not kcc.get("candidate_residue_ids"):
                join_fails.append(f"kcc_candidate_residue_ids@site{site_id}")

            # Strip pre-fix therm fields from interpretation sections (keep in raw_value_block with tag)
            raw_patched = dict(raw)
            therm_fields = ["therm_class", "hysteresis_asymmetry", "relative_asymmetry"]
            for f in therm_fields:
                if f in raw_patched and not is_therm_safe:
                    raw_patched[f"{f}__tag"] = "PRE_FIX_NOT_MEDCHEM_SAFE"
            # Replace the (invalid) kcc_driver_residue_id in raw block with canonical
            raw_patched["kcc_driver_residue_id"] = (
                kcc.get("driver_residue_id") if isinstance(kcc.get("driver_residue_id"), int)
                else "FEATURE_GAP:valid_kcc_driver_residue_id"
            )
            raw_patched["kcc_candidate_residue_ids"] = kcc.get("candidate_residue_ids") or "FEATURE_GAP:candidate_residue_ids"
            raw_patched["open_frequency_from_spike_events"] = (
                open_freq if isinstance(open_freq, (int, float))
                else "FEATURE_GAP:open_frequency_unresolved"
            )

            # ── Action gating ──
            canon_dcc = raw.get("canonical_dcc_angstrom")
            canon_verdict = raw.get("canonical_verdict")
            band = action_gate_band(canon_dcc, canon_verdict)
            is_bm = raw.get("is_best_match_vs_ground_truth") or False
            bud = budget_label(band, is_bm)
            framing = pick_framing(band, raw, is_bm)
            next_act = BAND_NEXT_ACTION[band][framing]

            # ── Compose prose ──
            notes = {
                "target": target,
                "site_id": site_id,
                "raw_value_block": raw_patched,
                "med_chem_summary": build_medchem_summary(target, raw, band, bud, is_bm),
                "residue_triage_note": build_residue_triage(target, raw, kcc),
                "dynamic_mechanistic_note": build_dynamic_note(target, raw, open_freq, is_therm_safe),
                "spatial_review_note": build_spatial_note(target, raw),
                "real_world_value_note": build_real_world(target, raw, band, bud, framing, kcc, is_bm),
                "next_action": next_act,
                "confidence_state": old.get("confidence_state", "MECHANISTIC_SUPPORT_ONLY"),
                "primary_value_framing": framing,
                "therm_fields_medchem_safe": is_therm_safe,
                "action_gate_band": band,
                "budget_label": bud,
                "note_data_join_failures": join_fails,
            }
            sc = specificity_count(notes, raw, kcc, open_freq)
            notes["specificity_value_hit_count"] = sc
            notes["specificity_pass"] = sc >= 3
            if not notes["specificity_pass"]:
                specificity_fails.append(f"NOTE_SPECIFICITY_FAIL:{target}:{site_id} (hits={sc})")

            new_notes.append(notes)
            all_rows.append({
                "target": target, "site_id": site_id,
                "rank": raw.get("current_reference_rank_under_B_no_therm"),
                "verified_best_match": is_bm,
                "confidence_state": notes["confidence_state"],
                "therm_fields_medchem_safe": is_therm_safe,
                "action_gate_band": band,
                "budget_label": bud,
                "canonical_dcc": canon_dcc,
                "volume": raw.get("volume_angstrom_cubed"),
                "n_spikes_attributed": raw.get("n_spikes_attributed"),
                "top_residues": str((raw.get("top_residue_ids") or [])[:8]),
                "primary_value_framing": framing,
                "next_action": next_act,
                "open_frequency": open_freq,
                "kcc_driver_residue_id": kcc.get("driver_residue_id") if isinstance(kcc.get("driver_residue_id"), int) else "FEATURE_GAP",
                "specificity_pass": notes["specificity_pass"],
                "note_data_join_failures": ";".join(join_fails) if join_fails else "-",
            })

        out = OUT / f"{target}.site_decision_notes.v2.json"
        out.write_text(json.dumps({
            "target": target,
            "reference_variant": "B_no_therm",
            "pre_fix_therm_quarantine": (target in PRE_FIX_THERM_QUARANTINE_TARGETS),
            "best_match_site_id": d.get("best_match_site_id"),
            "evaluation_verdict_tag": d.get("evaluation_verdict_tag"),
            "note_count": len(new_notes),
            "notes": new_notes,
        }, indent=2, default=str))
        print(f"[{target}] v2 notes written: {len(new_notes)} sites")

    csv_path = OUT / "internal_medchem_review_table.v2.csv"
    with csv_path.open("w") as f:
        w = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
        w.writeheader()
        for r in all_rows:
            w.writerow(r)

    print()
    print(f"review table v2: {csv_path}")
    print(f"n_sites={len(all_rows)}  data_join_fails={len(data_join_fails)}  specificity_fails={len(specificity_fails)}")
    print()
    if data_join_fails:
        print("NOTE_DATA_JOIN_FAIL list:")
        for s in data_join_fails:
            print(f"  {s}")
    if specificity_fails:
        print("NOTE_SPECIFICITY_FAIL list:")
        for s in specificity_fails:
            print(f"  {s}")


if __name__ == "__main__":
    main()
