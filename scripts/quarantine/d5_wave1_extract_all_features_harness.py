#!/usr/bin/env python3
"""Step-C parity + timing harness for extract_all_features.compute_temporal_features.

For each target, run the function in two modes and compare outputs:
  A_JSON    : triad hidden → JSON fallback (legacy path)
  C_D5_VIEW : triad present → view-based read of (xyz, ccns_phase)

Parity target: the two outputs (per_res [N,2], per_site dict) must be equivalent
within documented tolerance (integer counts exact; f32 ratios ≤1e-6 relative).

Fallback test: hide triad → view open must fail, legacy JSON path must still run.
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

DEFAULT_TARGET_DIR = Path("/home/diddy/prism-working/m1-strict-dcc-panel/m1_2akr")
DEFAULT_STEM = "2akr"


def _hide_triad(eng: Path, stem: str):
    meta = eng / f"{stem}.run_metadata.json"
    hidden = eng / f"{stem}.run_metadata.json._hidden_for_step_c"
    if meta.exists():
        meta.rename(hidden)


def _restore_triad(eng: Path, stem: str):
    meta = eng / f"{stem}.run_metadata.json"
    hidden = eng / f"{stem}.run_metadata.json._hidden_for_step_c"
    if hidden.exists():
        hidden.rename(meta)


def _load_ca_and_sites(eng: Path, stem: str, target_dir: Path):
    """Build the ca_coords, resid_list, bs_sites needed by compute_temporal_features.

    Uses the topology Cα positions if available. For local parity test we derive
    ca_coords from the engine's binding_sites lining residues where possible.
    For a rigorous parity test, we only need ca_coords to be IDENTICAL across
    A and C runs — so synthesize from binding_sites once and reuse.
    """
    bs_path = eng / f"{stem}.binding_sites.json"
    bs = json.loads(bs_path.read_text())
    bs_sites = bs.get("sites") or []
    # Deterministic synthetic residue grid — ensures A/C see IDENTICAL
    # ca_coords so the per-residue aggregation (which depends on spatial
    # proximity) is apples-to-apples. Use the union of site centroids as
    # "residue positions" — each site contributes one proxy residue at its centroid.
    resid_list = []
    ca_list = []
    for i, s in enumerate(bs_sites):
        if not isinstance(s, dict) or not s.get("centroid"):
            continue
        resid_list.append(int(s.get("id", i)))
        ca_list.append(s["centroid"])
    # Add extra lattice points for broader coverage (deterministic)
    cx_mean = float(np.mean([c[0] for c in ca_list])) if ca_list else 0.0
    cy_mean = float(np.mean([c[1] for c in ca_list])) if ca_list else 0.0
    cz_mean = float(np.mean([c[2] for c in ca_list])) if ca_list else 0.0
    for dx in (-10.0, -5.0, 0.0, 5.0, 10.0):
        for dy in (-10.0, -5.0, 0.0, 5.0, 10.0):
            resid_list.append(9000 + len(resid_list))
            ca_list.append([cx_mean + dx, cy_mean + dy, cz_mean])
    ca_coords = np.asarray(ca_list, dtype=np.float32)
    return ca_coords, resid_list, bs_sites


def run_compute_temporal(eng: Path, stem: str, target_dir: Path):
    import scripts.training.extract_all_features as eaf
    ca_coords, resid_list, bs_sites = _load_ca_and_sites(eng, stem, target_dir)
    per_res, per_site = eaf.compute_temporal_features(eng, ca_coords, resid_list, bs_sites)
    return per_res, per_site


def parity(per_res_a, per_site_a, per_res_c, per_site_c,
           tol_rel=1e-6, tol_abs=1e-7):
    """Compare (per_res, per_site) tuples. Returns verdict dict."""
    shape_ok = (np.asarray(per_res_a).shape == np.asarray(per_res_c).shape)
    per_res_a = np.asarray(per_res_a, dtype=np.float64)
    per_res_c = np.asarray(per_res_c, dtype=np.float64)
    diff = np.abs(per_res_a - per_res_c)
    denom = np.abs(per_res_a) + 1e-12
    rel = diff / denom
    max_abs = float(diff.max()) if diff.size else 0.0
    max_rel = float(rel.max()) if rel.size else 0.0
    per_res_pass = (max_abs <= tol_abs) or (max_rel <= tol_rel)

    keys_a = set(per_site_a); keys_c = set(per_site_c)
    missing_in_c = sorted(keys_a - keys_c)
    missing_in_a = sorted(keys_c - keys_a)
    site_fails = []
    site_max_abs = 0.0
    site_max_rel = 0.0
    for k in sorted(keys_a & keys_c):
        for field in ("phase_transition_ratio", "warm_hold_spike_fraction"):
            va = per_site_a[k].get(field)
            vc = per_site_c[k].get(field)
            # NaN-safe equality: both NaN → pass
            if isinstance(va, float) and isinstance(vc, float) \
                    and np.isnan(va) and np.isnan(vc):
                continue
            if va is None or vc is None:
                site_fails.append((k, field, "None"))
                continue
            va_f = float(va); vc_f = float(vc)
            d = abs(va_f - vc_f)
            r = d / (abs(va_f) + 1e-12)
            site_max_abs = max(site_max_abs, d)
            site_max_rel = max(site_max_rel, r)
            if d > tol_abs and r > tol_rel:
                site_fails.append((k, field, {"a": va_f, "c": vc_f, "abs": d, "rel": r}))
    verdict = "PASS" if (shape_ok and per_res_pass and not missing_in_c
                          and not missing_in_a and not site_fails) else "FAIL"
    return {
        "shape_ok": shape_ok,
        "per_res_max_abs": max_abs,
        "per_res_max_rel": max_rel,
        "per_res_pass": per_res_pass,
        "site_max_abs": site_max_abs,
        "site_max_rel": site_max_rel,
        "missing_in_c": missing_in_c,
        "missing_in_a": missing_in_a,
        "site_fails_preview": site_fails[:5],
        "n_site_fails": len(site_fails),
        "verdict": verdict,
    }


def _fingerprint(per_res, per_site):
    res = np.asarray(per_res, dtype=np.float32)
    h1 = hashlib.sha256(res.tobytes()).hexdigest()
    per_site_canonical = {
        k: {
            "phase_transition_ratio": None if (isinstance(v["phase_transition_ratio"], float)
                                                and np.isnan(v["phase_transition_ratio"])) else v["phase_transition_ratio"],
            "warm_hold_spike_fraction": None if (isinstance(v["warm_hold_spike_fraction"], float)
                                                  and np.isnan(v["warm_hold_spike_fraction"])) else v["warm_hold_spike_fraction"],
        } for k, v in sorted(per_site.items())
    }
    h2 = hashlib.sha256(json.dumps(per_site_canonical, sort_keys=True).encode()).hexdigest()
    return h1, h2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target-dir", default=str(DEFAULT_TARGET_DIR))
    ap.add_argument("--stem", default=DEFAULT_STEM)
    ap.add_argument("--mode", required=True, choices=["legacy", "view", "parity", "fallback"])
    args = ap.parse_args()

    td = Path(args.target_dir)
    eng = td / "artifacts/5_engine"
    stem = args.stem

    if args.mode == "fallback":
        _hide_triad(eng, stem)
        try:
            try:
                run_compute_temporal(eng, stem, td)  # legacy-fallback path
                legacy_ok = True
            except Exception as e:
                legacy_ok = False
                print(f"legacy_error: {e}")
            print(f"FALLBACK_LEGACY_OK={legacy_ok}")
            # Also verify that the view itself would fail with triad hidden
            from scripts.interfaces.site_spike_view import SiteSpikeView, SiteSpikeViewError
            view_blocked = False
            try:
                SiteSpikeView.from_triad(
                    eng / f"{stem}.topology.spike_events.arrow",
                    eng / f"{stem}.run_metadata.json",
                    eng / f"{stem}.binding_sites.json",
                )
            except SiteSpikeViewError as e:
                view_blocked = True
                print(f"view_blocked_reason: {e}")
            print(f"FALLBACK_VIEW_BLOCKED={view_blocked}")
        finally:
            _restore_triad(eng, stem)
        return

    if args.mode == "parity":
        # A: triad hidden → legacy JSON fallback
        _hide_triad(eng, stem)
        try:
            per_res_a, per_site_a = run_compute_temporal(eng, stem, td)
        finally:
            _restore_triad(eng, stem)
        # C: triad present → D5 view
        per_res_c, per_site_c = run_compute_temporal(eng, stem, td)
        report = parity(per_res_a, per_site_a, per_res_c, per_site_c)
        print(json.dumps(report, indent=2, default=str))
        return

    t0 = time.perf_counter()
    if args.mode == "legacy":
        _hide_triad(eng, stem)
        try:
            per_res, per_site = run_compute_temporal(eng, stem, td)
        finally:
            _restore_triad(eng, stem)
    else:
        per_res, per_site = run_compute_temporal(eng, stem, td)
    elapsed = time.perf_counter() - t0

    h1, h2 = _fingerprint(per_res, per_site)
    print(f"MODE={args.mode}")
    print(f"TARGET={stem}")
    print(f"PER_RES_SHA256_F32={h1}")
    print(f"PER_SITE_SHA256={h2}")
    print(f"N_PER_RES_ROWS={np.asarray(per_res).shape[0]}")
    print(f"N_PER_SITE_KEYS={len(per_site)}")
    print(f"INTERNAL_ELAPSED={elapsed:.3f}s")


if __name__ == "__main__":
    main()
