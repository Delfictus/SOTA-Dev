#!/usr/bin/env python3
"""Step-B parity + timing harness for train_v006 (Wave 1 consumer).

For each target, enumerate its per-site spike_events.json files and compute
18-dim temporal window features via:
  A_JSON   : legacy parse_spike_events + compute_window_features
             (500k-event cap applied by parse_spike_events)
  C_D5_VIEW: SiteSpikeView.site(sid).temporal_windows_18dim(max_events_cap=500_000)

Parity: per-site, compare matrices with tolerance.
Timing: wall/user/sys/max_rss via /usr/bin/time -v (measured externally) plus
        internal elapsed.

Fallback test: hide triad, confirm A-path still runs and produces same legacy
baseline.
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

import numpy as np
import re

DEFAULT_TARGET_DIR = Path("/home/diddy/prism-working/m1-strict-dcc-panel/m1_2akr")
DEFAULT_STEM = "2akr"


def _hide_triad(eng: Path, stem: str):
    meta = eng / f"{stem}.run_metadata.json"
    hidden = eng / f"{stem}.run_metadata.json._hidden_for_step_b"
    if meta.exists():
        meta.rename(hidden)


def _restore_triad(eng: Path, stem: str):
    meta = eng / f"{stem}.run_metadata.json"
    hidden = eng / f"{stem}.run_metadata.json._hidden_for_step_b"
    if hidden.exists():
        hidden.rename(meta)


def run_legacy(eng: Path, stem: str, n_windows: int = 32):
    """Invoke legacy parse+compute for every site JSON in eng dir.
    Returns {sid: np.ndarray(n_windows,18)}.
    """
    import scripts.training.train_v006 as tv
    out = {}
    site_files = sorted(eng.glob(f"{stem}.site*.spike_events.json"))
    for p in site_files:
        m = re.match(rf"^{stem}\.site(?P<sid>-?\d+)\.spike_events\.json$", p.name)
        if not m:
            continue
        sid = int(m.group("sid"))
        events = tv.parse_spike_events(p)
        if not events:
            out[sid] = np.zeros((n_windows, 18), dtype=np.float32)
            continue
        out[sid] = tv.compute_window_features(events, n_windows=n_windows)
    return out


def run_view(eng: Path, stem: str, n_windows: int = 32):
    """Invoke D5 view path for every site in binding_sites.
    Returns {sid: np.ndarray(n_windows,18)}.
    """
    from scripts.interfaces.site_spike_view import SiteSpikeView
    arrow_p = eng / f"{stem}.topology.spike_events.arrow"
    meta_p = eng / f"{stem}.run_metadata.json"
    bs_p = eng / f"{stem}.binding_sites.json"
    view = SiteSpikeView.from_triad(arrow_p, meta_p, bs_p, stem=stem)
    out = {}
    for sid in view.available_site_ids():
        sl = view.site(sid)
        out[sid] = sl.temporal_windows_18dim(n_windows=n_windows, max_events_cap=500_000)
    return out


def parity(legacy, view, tol_rel=1e-3, tol_abs=5e-4):
    """tolerance justification: legacy compute_window_features accumulates
    `features[w, ph_idx] += 1.0/n` in f32 across ~10k iterations per window;
    f32 ULP compounds to ~1e-3 relative. Vectorized path uses single f64
    division + f32 cast and is ~1e-7 relative. Tolerance is set to absorb
    the legacy accumulation bias; vectorized result is more accurate, not
    semantically different."""
    """Compare two {sid: (n_windows,18)} dicts. Returns summary dict."""
    keys_a = set(legacy); keys_b = set(view)
    missing_in_view = sorted(keys_a - keys_b)
    missing_in_legacy = sorted(keys_b - keys_a)
    common = sorted(keys_a & keys_b)

    per_site = {}
    n_pass = n_fail = 0
    worst_rel = 0.0
    worst_abs = 0.0
    worst_site = None
    worst_cell = None

    for sid in common:
        a = np.asarray(legacy[sid], dtype=np.float64)
        b = np.asarray(view[sid], dtype=np.float64)
        if a.shape != b.shape:
            per_site[sid] = {"shape_mismatch": (list(a.shape), list(b.shape))}
            n_fail += 1
            continue
        diff = np.abs(a - b)
        denom = np.abs(a) + 1e-12
        rel = diff / denom
        max_abs = float(diff.max())
        max_rel = float(rel.max())
        passes = (max_abs <= tol_abs) or (max_rel <= tol_rel)
        per_site[sid] = {
            "max_abs": max_abs, "max_rel": max_rel,
            "passes": passes,
            "shape": list(a.shape),
        }
        if passes:
            n_pass += 1
        else:
            n_fail += 1
        if max_rel > worst_rel:
            worst_rel = max_rel
            worst_site = sid
            worst_cell = tuple(map(int, np.unravel_index(int(rel.argmax()), rel.shape)))
        if max_abs > worst_abs:
            worst_abs = max_abs

    verdict = "PASS" if (not missing_in_view and not missing_in_legacy and n_fail == 0) else "FAIL"
    return {
        "n_common": len(common),
        "n_pass": n_pass,
        "n_fail": n_fail,
        "missing_in_view": missing_in_view,
        "missing_in_legacy": missing_in_legacy,
        "worst_rel": worst_rel,
        "worst_abs": worst_abs,
        "worst_site": worst_site,
        "worst_cell": worst_cell,
        "verdict": verdict,
        "per_site": per_site,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target-dir", default=str(DEFAULT_TARGET_DIR))
    ap.add_argument("--stem", default=DEFAULT_STEM)
    ap.add_argument("--mode", required=True, choices=["legacy", "view", "parity", "fallback"])
    ap.add_argument("--n-windows", type=int, default=32)
    args = ap.parse_args()

    td = Path(args.target_dir)
    eng = td / "artifacts/5_engine"
    stem = args.stem

    if args.mode == "fallback":
        # Hide triad, run view mode (should raise / empty), restore, confirm legacy runs
        _hide_triad(eng, stem)
        try:
            try:
                run_view(eng, stem, n_windows=args.n_windows)
                fallback_view_blocked = False
            except Exception as e:
                fallback_view_blocked = True
                print(f"view_blocked_reason: {e}")
            legacy_ok = False
            try:
                res = run_legacy(eng, stem, n_windows=args.n_windows)
                legacy_ok = len(res) > 0
            except Exception as e:
                print(f"legacy_error: {e}")
            print(f"FALLBACK_VIEW_BLOCKED={fallback_view_blocked}")
            print(f"FALLBACK_LEGACY_OK={legacy_ok}")
        finally:
            _restore_triad(eng, stem)
        return

    t0 = time.perf_counter()
    if args.mode == "legacy":
        result = run_legacy(eng, stem, n_windows=args.n_windows)
    elif args.mode == "view":
        result = run_view(eng, stem, n_windows=args.n_windows)
    elif args.mode == "parity":
        legacy_res = run_legacy(eng, stem, n_windows=args.n_windows)
        view_res = run_view(eng, stem, n_windows=args.n_windows)
        report = parity(legacy_res, view_res)
        print(json.dumps({k: v for k, v in report.items() if k != "per_site"}, indent=2, default=str))
        fails = [(sid, r) for sid, r in report["per_site"].items() if not r.get("passes")]
        for sid, r in fails[:3]:
            print(f"  FAIL site={sid} max_abs={r.get('max_abs'):.4g} max_rel={r.get('max_rel'):.4g}")
        return
    elapsed = time.perf_counter() - t0

    # Fingerprint: stack in sorted sid order → sha256 over f32 bytes
    sids_sorted = sorted(result.keys())
    stacked = np.stack([result[s] for s in sids_sorted])
    h = hashlib.sha256(stacked.tobytes()).hexdigest()
    total_bytes = stacked.nbytes
    print(f"MODE={args.mode}")
    print(f"TARGET={stem}")
    print(f"N_SITES={len(sids_sorted)}")
    print(f"STACK_SHAPE={list(stacked.shape)}")
    print(f"SHA256_F32={h}")
    print(f"BYTES={total_bytes}")
    print(f"INTERNAL_ELAPSED={elapsed:.3f}s")


if __name__ == "__main__":
    main()
