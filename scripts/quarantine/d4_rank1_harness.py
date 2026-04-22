#!/usr/bin/env python3
"""D4 RANK-1 parity + timing harness.

Runs the two production hot-path scripts (spike_pharmacophore_map,
response_selectivity) against m1_2akr in two modes:

  mode=A_JSON    : Arrow-first triad hidden (run_metadata moved aside) → JSON fallback
  mode=B_ARROW   : Arrow-first triad present → Arrow-first path exercised

Reports per-script:
  - output content (canonical dict)
  - sha256 of the canonical content
  - byte size of the canonical content
  - drift vs the other mode (if both results available)

Also prints rows-sampled / per-site counts summary for manual inspection.

READ-ONLY on engine artifacts. The triad-hiding is done by a temporary
symlink/rename of the run_metadata sidecar; the rename is reverted in a
try/finally so the canonical path is never left in an inconsistent state.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO))

TARGET_DIR_DEFAULT = Path("/home/diddy/prism-working/m1-strict-dcc-panel/m1_2akr")
STEM_DEFAULT = "2akr"
# Globals set by --target argument (A/B/C share the same target in each run).
TARGET_DIR = TARGET_DIR_DEFAULT
STEM = STEM_DEFAULT
ENG = TARGET_DIR / "artifacts/5_engine"
META_PATH = ENG / f"{STEM}.run_metadata.json"
META_HIDDEN = ENG / f"{STEM}.run_metadata.json._hidden_for_d4_timing"


def _set_target(target_dir: Path, stem: str):
    global TARGET_DIR, STEM, ENG, META_PATH, META_HIDDEN
    TARGET_DIR = Path(target_dir)
    STEM = stem
    ENG = TARGET_DIR / "artifacts/5_engine"
    META_PATH = ENG / f"{STEM}.run_metadata.json"
    META_HIDDEN = ENG / f"{STEM}.run_metadata.json._hidden_for_d4_timing"


def _hide_triad():
    """Move run_metadata sidecar aside so the Arrow-first triad precondition fails."""
    if META_PATH.exists():
        META_PATH.rename(META_HIDDEN)


def _restore_triad():
    if META_HIDDEN.exists():
        META_HIDDEN.rename(META_PATH)


def run_pharmacophore(mode: str) -> dict:
    """Invoke the pharmacophore loader path on ONE representative site and
    return the canonical post-load dict content for comparison.

    mode:
      A_JSON:       hide triad → legacy json.load fallback path
      B_ARROW:      call the naive D4 helper (if it still exists in the file)
      C_D5_VIEW:    use the new SiteSpikeView layer
    """
    import scripts.spike_pharmacophore_map as ph
    from scripts.interfaces.site_spike_view import SiteSpikeView

    files = sorted(ENG.glob(f"{STEM}.site*.spike_events.json"))
    target_json = files[len(files) // 2]
    import re as _re
    m = _re.match(r"(?P<stem>.+)\.site(?P<sid>-?\d+)\.spike_events\.json$", target_json.name)
    sid = int(m.group("sid")) if m else None

    if mode == "C_D5_VIEW":
        view = SiteSpikeView.from_target_dir(TARGET_DIR, STEM)
        if view is None or sid is None or not view.has_site(sid):
            raise RuntimeError("C_D5_VIEW requested but view/site not available")
        sl = view.site(sid)
        n_spikes = sl.n_spikes()
        centroid = list(sl.centroid())
        type_stats = sl.type_intensity_stats()
        type_counts = {k: v[0] for k, v in type_stats.items()}
        voxels = sl.voxel_aggregate(grid_spacing=2.0)
        source_counts_agg = sl.source_counts()
        phase_counts_agg = sl.phase_counts()
        # Canonical dict for hashing — summarize voxel aggregation at the site level.
        data = {
            "site_id": sl.site_id(),
            "centroid": centroid,
            "n_spikes": n_spikes,
            "lining_cutoff": sl.lining_cutoff(),
            "open_frequency": sl.open_frequency(),
            "n_voxels": len(voxels),
            "total_spike_count_in_voxels": int(sum(v["spike_count"] for v in voxels.values())),
            "total_intensity_in_voxels": float(sum(v["total_intensity"] for v in voxels.values())),
            "unique_dominant_types": sorted({v["dominant_type"] for v in voxels.values()}),
        }
        source = f"d5_view:arrow+meta+bs#site{sid}"
        # For the canonical per-type stats we store via the downstream keys.
        out = {
            "source": source,
            "site_id": data["site_id"],
            "centroid": data["centroid"],
            "n_spikes": n_spikes,
            "lining_cutoff": data["lining_cutoff"],
            "open_frequency": round(data["open_frequency"], 6),
            "type_counts": type_counts,
            "source_counts": source_counts_agg,
            "phase_counts": phase_counts_agg,
            "n_voxels": data["n_voxels"],
            "voxel_total_spike_count": data["total_spike_count_in_voxels"],
            "voxel_total_intensity_round6": round(data["total_intensity_in_voxels"], 3),
        }
        return out

    # A_JSON or B_ARROW: execute script-level loader path
    arrow_doc, src = (None, None)
    if mode == "B_ARROW":
        arrow_doc, src = ph._arrow_first_site_doc_from_path(target_json)
    if arrow_doc is not None:
        data = arrow_doc
        source = f"arrow:{src}"
    else:
        with open(target_json) as f:
            data = json.load(f)
        source = f"json:{target_json.name}"
    spikes = data.get("spikes", [])
    # Also compute voxel aggregate via legacy path for apples-to-apples comparison with C.
    voxels = ph.voxelize_spikes(spikes, grid_spacing=2.0) if spikes else {}
    out = {
        "source": source,
        "site_id": data.get("site_id"),
        "centroid": data.get("centroid"),
        "n_spikes": data.get("n_spikes"),
        "lining_cutoff": data.get("lining_cutoff"),
        "open_frequency": round(float(data.get("open_frequency") or 0.0), 6),
        "type_counts": {
            t: sum(1 for s in spikes if s.get("type") == t)
            for t in {s.get("type") for s in spikes} if t is not None
        },
        "source_counts": {
            s: sum(1 for sp in spikes if sp.get("spike_source") == s)
            for s in {sp.get("spike_source") for sp in spikes} if s is not None
        },
        "phase_counts": {
            p: sum(1 for s in spikes if s.get("ccns_phase") == p)
            for p in {s.get("ccns_phase") for s in spikes} if p is not None
        },
        "n_voxels": len(voxels),
        "voxel_total_spike_count": int(sum(v["spike_count"] for v in voxels.values())),
        "voxel_total_intensity_round6": round(float(sum(v["total_intensity"] for v in voxels.values())), 3),
    }
    return out


def run_response_selectivity(mode: str) -> dict:
    """Invoke response_selectivity.evaluate_all() on all sites.

    mode:
      A_JSON:       triad hidden by _hide_triad(); legacy path runs (load_spike_events)
      B_ARROW:      evaluate_all chooses D4-style materialization (since triad present);
                    for fairness we route via evaluate_all which now uses D5 path when
                    triad is present; for B timing specifically we emulate the disproven
                    D4 behavior by calling evaluate_all with _arrow_first_lazy_builder
                    disabled (approximated by triad-hidden A path since D4 was strictly
                    slower; noted in the table as "equivalent to A for response_selectivity").
      C_D5_VIEW:    triad present; evaluate_all routes through SiteSpikeView (D5 path)
    """
    import scripts.response_selectivity as rs
    bs_p = ENG / f"{STEM}.binding_sites.json"
    data = json.loads(bs_p.read_text())
    sites = data if isinstance(data, list) else (data.get("sites") or [])
    gate = rs.ResponseSelectivityGate()

    if mode == "B_ARROW":
        # Explicitly exercise the disproven D4 naive Arrow-first path: lazy-builder
        # yields one list[dict] per site; evaluate() runs on list[dict] (not slice).
        arrow_build, arrow_sids = rs._arrow_first_lazy_builder(ENG, STEM)
        if arrow_build is None:
            raise RuntimeError("B_ARROW requested but lazy builder not available")
        arrow_sids_set = set(arrow_sids)
        results = {}
        for site in sites:
            sid = site.get("id")
            if sid in arrow_sids_set:
                spikes = arrow_build(sid)
            else:
                spikes = []
            results[sid] = gate.evaluate(site, spikes)
    else:
        # A_JSON (triad hidden) or C_D5_VIEW (triad present) both flow through
        # evaluate_all — which now routes through SiteSpikeView when triad is
        # available, or legacy load_spike_events when hidden.
        results = gate.evaluate_all(sites, str(ENG))
    out = {
        "n_sites": len(results),
        "per_site": {
            str(sid): {
                "sharpness": rp.sharpness,
                "temporal_asymmetry": rp.temporal_asymmetry,
                "energy_density": rp.energy_density,
                "n_spikes_analyzed": rp.n_spikes_analyzed,
                "gate_pass": rp.gate_pass,
            }
            for sid, rp in sorted(results.items())
        },
    }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", required=True, choices=["A", "B", "C"])
    ap.add_argument("--script", required=True, choices=["pharmacophore", "response_selectivity"])
    ap.add_argument("--target-dir", default=None)
    ap.add_argument("--stem", default=None)
    args = ap.parse_args()
    if args.target_dir and args.stem:
        _set_target(Path(args.target_dir), args.stem)
    try:
        if args.mode == "A":
            _hide_triad()
        t0 = time.perf_counter()
        if args.script == "pharmacophore":
            # A/B/C pharmacophore path
            out = run_pharmacophore("A_JSON" if args.mode == "A"
                                    else "B_ARROW" if args.mode == "B"
                                    else "C_D5_VIEW")
        else:
            out = run_response_selectivity("A_JSON" if args.mode == "A"
                                           else "B_ARROW" if args.mode == "B"
                                           else "C_D5_VIEW")
        elapsed = time.perf_counter() - t0
    finally:
        _restore_triad()

    canon = json.dumps(out, sort_keys=True, default=str)
    h = hashlib.sha256(canon.encode()).hexdigest()
    out_path = Path(f"/tmp/d4_rank1_{args.script}_{STEM}_{args.mode}.json")
    out_path.write_text(canon)
    print(f"MODE={args.mode}")
    print(f"SCRIPT={args.script}")
    print(f"SHA256={h}")
    print(f"BYTES={len(canon)}")
    print(f"INTERNAL_ELAPSED={elapsed:.3f}s")
    fp = {
        "mode": args.mode, "script": args.script,
        "sha256": h, "bytes": len(canon), "elapsed": round(elapsed, 3),
        "n_sites": out.get("n_sites"),
        "n_spikes_total": sum((v.get("n_spikes_analyzed") or 0)
                              for v in (out.get("per_site") or {}).values()) if args.script == "response_selectivity"
                          else out.get("n_spikes"),
    }
    print(json.dumps(fp))


if __name__ == "__main__":
    main()
