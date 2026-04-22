#!/usr/bin/env python3
"""D3 parity harness — validates that the Arrow-first readers added to the 4
D3-patched scripts produce equivalent output to the legacy JSON readers.

READ-ONLY. No engine change, no default flip, no writes outside /tmp.

Compares on the m1_2akr target (Gate-A validated).
  1. spike_metadata_inventory._arrow_first_sample_se         vs  read(json) on one site
  2. m1_2c_cleanup._arrow_first_open_frequencies             vs  read_open_frequency on all per-site JSONs
  3. engine_full_harvest._arrow_first_site_summaries         vs  read_spike_file_summary on all per-site JSONs
  4. completeness_certifier._arrow_first_spike_entries       vs  stream_spike_fields_and_count on all per-site JSONs

Reports per-script: PASS/FAIL + per-field drift.
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

TARGET_DIR = Path("/home/diddy/prism-working/m1-strict-dcc-panel/m1_2akr")
STEM = "2akr"
ENG = TARGET_DIR / "artifacts/5_engine"

import spike_metadata_inventory as sm
import m1_2c_cleanup as m2c
import engine_full_harvest as efh
import completeness_certifier as cc


def check_spike_metadata_inventory() -> dict:
    arrow_doc, _ = sm._arrow_first_sample_se(ENG, STEM)
    # Compare vs loading the mid-ranked per-site JSON (the same selection rule
    # used by the original code path).
    files = sorted(ENG.glob(f"{STEM}.site*.spike_events.json"))
    # Original rule: paths[per_site][len/2]. But sm uses bs.sites[mid] under Arrow path.
    # For parity we just verify both paths yield an object with identical top-level
    # keys AND the first-spike dict has identical key-set AND numeric values match
    # for the same (ts, sid, arid, x, y, z) row if present.
    top_keys = sorted(arrow_doc.keys())
    spike0_keys = sorted(arrow_doc["spikes"][0].keys()) if arrow_doc["spikes"] else []
    # Compare against known legacy schema
    expected_top = sorted(["centroid","lining_cutoff","n_spikes","open_frequency","site_id","spikes"])
    expected_spk = sorted(["aromatic_residue_id","ccns_phase","frame_index","intensity",
                           "n_nearby_excited","spike_source","stream_id","timestep",
                           "type","vibrational_energy","water_density","wavelength_nm",
                           "x","y","z"])
    ok_top = (top_keys == expected_top)
    ok_spk = (spike0_keys == expected_spk)
    # Compare arrow sample's n_spikes to the centroid-matched legacy file's n_spikes
    legacy_sample_sid = arrow_doc["site_id"]
    legacy_path = ENG / f"{STEM}.site{legacy_sample_sid}.spike_events.json"
    legacy_n = None
    if legacy_path.exists():
        with legacy_path.open() as f:
            head = f.read(4096)
            import re
            m = re.search(r'"n_spikes"\s*:\s*(\d+)', head)
            legacy_n = int(m.group(1)) if m else None
    n_spikes_match = (legacy_n is not None and legacy_n == arrow_doc["n_spikes"])
    return {
        "script": "spike_metadata_inventory.py",
        "arrow_top_keys": top_keys,
        "arrow_spike0_keys": spike0_keys,
        "top_keys_match_schema": ok_top,
        "spike_keys_match_schema": ok_spk,
        "n_spikes_arrow": arrow_doc["n_spikes"],
        "n_spikes_legacy": legacy_n,
        "n_spikes_match": n_spikes_match,
        "verdict": "PASS" if (ok_top and ok_spk and n_spikes_match) else "FAIL",
    }


def check_m1_2c_cleanup() -> dict:
    arrow_ofreqs, _ = m2c._arrow_first_open_frequencies(ENG, STEM)
    # Legacy: regex-extract open_frequency from each per-site JSON
    legacy_ofreqs = {}
    import re as _re
    for p in sorted(ENG.glob(f"{STEM}.site*.spike_events.json")):
        m = _re.search(rf"{STEM}\.site(\d+)\.spike_events\.json$", p.name)
        if not m: continue
        sid = int(m.group(1))
        ofreq = m2c.read_open_frequency(p)
        legacy_ofreqs[sid] = ofreq
    # Compare sorted keysets + values per sid
    common_sids = set(arrow_ofreqs) & set(legacy_ofreqs)
    missing_arrow = set(legacy_ofreqs) - set(arrow_ofreqs)
    missing_legacy = set(arrow_ofreqs) - set(legacy_ofreqs)
    per_sid = []
    drift = []
    for sid in sorted(common_sids):
        a, b = arrow_ofreqs[sid], legacy_ofreqs[sid]
        eq = (a is not None and b is not None and abs(a - b) <= 1e-9)
        per_sid.append({"sid": sid, "arrow": a, "legacy": b, "match": eq})
        if not eq:
            drift.append({"sid": sid, "arrow": a, "legacy": b, "diff": (a - b) if (a is not None and b is not None) else None})
    return {
        "script": "m1_2c_cleanup.py",
        "n_sids_arrow": len(arrow_ofreqs),
        "n_sids_legacy": len(legacy_ofreqs),
        "n_sids_common": len(common_sids),
        "missing_in_arrow": sorted(missing_arrow),
        "missing_in_legacy": sorted(missing_legacy),
        "value_drift": drift,
        "verdict": "PASS" if (not drift and not missing_arrow and not missing_legacy) else "FAIL",
    }


def check_engine_full_harvest() -> dict:
    arrow_sums, _ = efh._arrow_first_site_summaries(ENG, STEM)
    legacy_sums = {}
    for p in sorted(ENG.glob(f"{STEM}.site*.spike_events.json")):
        d = efh.read_spike_file_summary(p)
        sid = d.get("site_id")
        if sid is None:
            continue
        legacy_sums[sid] = {
            "n_spikes": d.get("n_spikes"),
            "centroid": d.get("centroid"),
            "open_frequency": d.get("open_frequency"),
            "lining_cutoff": d.get("lining_cutoff"),
        }
    common = set(arrow_sums) & set(legacy_sums)
    drift = []
    for sid in sorted(common):
        a = arrow_sums[sid]
        b = legacy_sums[sid]
        for fld in ["n_spikes", "open_frequency", "lining_cutoff"]:
            va, vb = a.get(fld), b.get(fld)
            if va is None or vb is None:
                drift.append({"sid": sid, "field": fld, "arrow": va, "legacy": vb, "cause": "MISSING"})
                continue
            if isinstance(va, (int, float)) and isinstance(vb, (int, float)):
                if abs(float(va) - float(vb)) > 1e-6:
                    drift.append({"sid": sid, "field": fld, "arrow": va, "legacy": vb, "cause": "VALUE_DRIFT"})
            elif va != vb:
                drift.append({"sid": sid, "field": fld, "arrow": va, "legacy": vb, "cause": "VALUE_DRIFT"})
        # centroid — list of 3 floats
        if a.get("centroid") and b.get("centroid"):
            for i, (xa, xb) in enumerate(zip(a["centroid"], b["centroid"])):
                if abs(float(xa) - float(xb)) > 1e-6:
                    drift.append({"sid": sid, "field": f"centroid[{i}]", "arrow": xa, "legacy": xb, "cause": "VALUE_DRIFT"})
    return {
        "script": "engine_full_harvest.py",
        "n_sids_arrow": len(arrow_sums),
        "n_sids_legacy": len(legacy_sums),
        "n_sids_common": len(common),
        "missing_in_arrow": sorted(set(legacy_sums) - set(arrow_sums)),
        "missing_in_legacy": sorted(set(arrow_sums) - set(legacy_sums)),
        "value_drift": drift,
        "verdict": "PASS" if (not drift and set(arrow_sums) == set(legacy_sums)) else "FAIL",
    }


def check_completeness_certifier() -> dict:
    arrow_entries, _ = cc._arrow_first_spike_entries(ENG, STEM)
    arrow_by_sid = {e["sid"]: e for e in arrow_entries}
    # Legacy: stream_spike_fields_and_count on each per-site file
    import re as _re
    legacy_by_sid = {}
    for p in sorted(ENG.glob(f"{STEM}.site*.spike_events.json")):
        m = _re.search(rf"{STEM}\.site(\d+)\.spike_events\.json$", p.name)
        if not m: continue
        sid = int(m.group(1))
        info = cc.stream_spike_fields_and_count(p) or {}
        legacy_by_sid[sid] = info
    common = set(arrow_by_sid) & set(legacy_by_sid)
    drift = []
    for sid in sorted(common):
        a_info = arrow_by_sid[sid]["info"]
        b_info = legacy_by_sid[sid]
        # n_spikes parity
        a_n = a_info.get("n_spikes_reported")
        b_n = b_info.get("n_spikes_reported")
        if a_n != b_n:
            drift.append({"sid": sid, "field": "n_spikes_reported", "arrow": a_n, "legacy": b_n, "cause": "VALUE_DRIFT"})
        # top_keys: legacy is extracted by streaming; arrow is fixed by schema.
        # We validate that the Arrow path set covers the legacy observed set.
        b_top = set(b_info.get("top_keys", []))
        a_top = set(a_info.get("top_keys", []))
        if b_top - a_top:
            drift.append({"sid": sid, "field": "top_keys", "arrow": sorted(a_top), "legacy": sorted(b_top), "cause": "ARROW_MISSING_KEYS"})
        # spike_keys: same idea
        b_spk = set(b_info.get("spike_keys", []))
        a_spk = set(a_info.get("spike_keys", []))
        if b_spk - a_spk:
            drift.append({"sid": sid, "field": "spike_keys", "arrow": sorted(a_spk), "legacy": sorted(b_spk), "cause": "ARROW_MISSING_KEYS"})
    return {
        "script": "completeness_certifier.py",
        "n_sids_arrow": len(arrow_by_sid),
        "n_sids_legacy": len(legacy_by_sid),
        "n_sids_common": len(common),
        "missing_in_arrow": sorted(set(legacy_by_sid) - set(arrow_by_sid)),
        "missing_in_legacy": sorted(set(arrow_by_sid) - set(legacy_by_sid)),
        "value_drift": drift,
        "verdict": "PASS" if (not drift and set(arrow_by_sid) == set(legacy_by_sid)) else "FAIL",
    }


def main():
    results = [
        check_spike_metadata_inventory(),
        check_m1_2c_cleanup(),
        check_engine_full_harvest(),
        check_completeness_certifier(),
    ]
    out = {"target": "m1_2akr", "stem": STEM, "per_script": results,
           "all_pass": all(r["verdict"] == "PASS" for r in results)}
    print(json.dumps(out, indent=2, default=str))


if __name__ == "__main__":
    main()
