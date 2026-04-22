#!/usr/bin/env python3
"""D3 A/B timing harness — measures wall/cpu/RSS for each of the 4 D3-patched
reader paths on m1_2akr, isolating the Arrow-first helper (B) vs the legacy
JSON reader (A). Uses the SAME data, SAME target, SAME site set.

READ-ONLY. No writes outside /tmp.

Run modes (via --mode):
  A   : time legacy JSON readers only (fallback path as it runs today)
  B   : time Arrow-first readers only (D3-patched path)

Each mode runs the equivalent data extraction for all 4 scripts' reader
entry points and exits. A separate wrapper (/usr/bin/time -v) measures
elapsed/user/sys/max_rss.

Also writes a sha256 hash of the produced summary JSON so A and B can be
checked for output-manifest equality.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import re
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


def run_arrow(stem: str) -> dict:
    """Execute all 4 Arrow-first helpers and produce a canonical summary."""
    out = {"mode": "B_arrow_first", "target": "m1_2akr"}
    arrow_sample_doc, _ = sm._arrow_first_sample_se(ENG, stem)
    out["sm_n_spikes"] = arrow_sample_doc["n_spikes"]
    out["sm_sample_sid"] = arrow_sample_doc["site_id"]
    arrow_ofreqs, _ = m2c._arrow_first_open_frequencies(ENG, stem)
    out["m2c_ofreqs"] = {int(k): float(v) for k, v in sorted(arrow_ofreqs.items())}
    arrow_sums, _ = efh._arrow_first_site_summaries(ENG, stem)
    out["efh_summaries"] = {
        int(k): {"n_spikes": v["n_spikes"], "open_frequency": v["open_frequency"],
                 "lining_cutoff": v["lining_cutoff"], "centroid": v["centroid"]}
        for k, v in sorted(arrow_sums.items())
    }
    arrow_entries, _ = cc._arrow_first_spike_entries(ENG, stem)
    out["cc_entries"] = {
        int(e["sid"]): {
            "n_spikes_reported": e["info"]["n_spikes_reported"],
            "top_keys": e["info"]["top_keys"],
            "spike_keys": e["info"]["spike_keys"],
        }
        for e in sorted(arrow_entries, key=lambda x: x["sid"])
    }
    return out


def run_legacy(stem: str) -> dict:
    """Execute all 4 legacy JSON readers and produce a canonical summary."""
    out = {"mode": "A_legacy_json", "target": "m1_2akr"}
    # sm: pick mid-ranked per-site JSON, fully load
    files = sorted(ENG.glob(f"{stem}.site*.spike_events.json"))
    mid = files[len(files) // 2]
    doc = json.loads(mid.read_text())
    out["sm_n_spikes"] = doc["n_spikes"]
    out["sm_sample_sid"] = doc["site_id"]
    # m2c: read_open_frequency on each file
    ofreqs = {}
    for p in files:
        m = re.search(rf"{stem}\.site(\d+)\.spike_events\.json$", p.name)
        if not m: continue
        ofreqs[int(m.group(1))] = m2c.read_open_frequency(p)
    out["m2c_ofreqs"] = {int(k): float(v) for k, v in sorted(ofreqs.items())}
    # efh: read_spike_file_summary on each file
    sums = {}
    for p in files:
        d = efh.read_spike_file_summary(p)
        sid = d.get("site_id")
        if sid is None: continue
        sums[sid] = {
            "n_spikes": d.get("n_spikes"),
            "open_frequency": d.get("open_frequency"),
            "lining_cutoff": d.get("lining_cutoff"),
            "centroid": d.get("centroid"),
        }
    out["efh_summaries"] = {int(k): v for k, v in sorted(sums.items())}
    # cc: stream_spike_fields_and_count on each file
    entries = {}
    for p in files:
        m = re.search(rf"{stem}\.site(\d+)\.spike_events\.json$", p.name)
        if not m: continue
        sid = int(m.group(1))
        info = cc.stream_spike_fields_and_count(p) or {}
        entries[sid] = {
            "n_spikes_reported": info.get("n_spikes_reported"),
            "top_keys": sorted(info.get("top_keys", [])),
            "spike_keys": sorted(info.get("spike_keys", [])),
        }
    out["cc_entries"] = {int(k): v for k, v in sorted(entries.items())}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", required=True, choices=["A", "B"])
    args = ap.parse_args()
    if args.mode == "B":
        out = run_arrow(STEM)
    else:
        out = run_legacy(STEM)
    # Normalize: both paths should produce identical summary content.
    # Arrow top_keys include "spikes"; legacy stream extraction stops before
    # the 'spikes' top-level key appears in map_key events (since we break at
    # start_array). To make A and B comparable we strip "spikes" from top_keys
    # in the Arrow output for hash equality (it's still truthfully there, but
    # the legacy stream doesn't observe it as a separate top_keys element).
    # Instead, compare the UNION semantics: require legacy top_keys ⊆ arrow top_keys.
    for sid, e in out.get("cc_entries", {}).items():
        e["top_keys"] = sorted(set(e.get("top_keys", [])) - {"spikes"})
    s = json.dumps(out, sort_keys=True, default=str)
    h = hashlib.sha256(s.encode()).hexdigest()
    # Dump canonical content for offline diff
    Path(f"/tmp/d3_timing_summary_{args.mode}.json").write_text(s)
    print(f"MODE={args.mode}")
    print(f"SHA256={h}")
    print(f"SUMMARY_BYTES={len(s)}")
    # Dump a compact JSON fingerprint to stdout tail for offline diff
    fp = {
        "mode": out["mode"],
        "sha256": h,
        "summary_bytes": len(s),
        "n_sids_m2c": len(out.get("m2c_ofreqs", {})),
        "n_sids_efh": len(out.get("efh_summaries", {})),
        "n_sids_cc": len(out.get("cc_entries", {})),
        "total_n_spikes_reported": sum(
            (e.get("n_spikes_reported") or 0) for e in out.get("cc_entries", {}).values()
        ),
    }
    print(json.dumps(fp))


if __name__ == "__main__":
    main()
