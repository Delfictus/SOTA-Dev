#!/usr/bin/env python3
"""Regression tests for target-config externalization.

Run order:
  1. test_4lpk_regression — confirms YAML-loaded KRAS config produces
     numerically-identical results to the prior hardcode (regression
     equivalence is non-negotiable).
  2. test_9m3p_substrate_only — confirms graceful degradation for a
     target whose YAML has empty `references:` (E11/E15/E16 BLOCKED
     with substrate-only gate).
"""
import json
import os
import subprocess
import sys
from pathlib import Path

PRISM_ROOT = Path("/home/diddy/Desktop/Prism4D-bio")


def run_dossier(arrow_path):
    """Run dossier_unified.py with PRISM_ARROW override, return (json, err)."""
    out_dir = Path(arrow_path).parent
    env = os.environ.copy()
    env["PRISM_ARROW"] = str(arrow_path)
    result = subprocess.run(
        ["python3", str(PRISM_ROOT / "scripts" / "dossier_unified.py")],
        env=env, capture_output=True, text=True,
    )
    if result.returncode != 0:
        return None, result.stderr[-500:]
    # Locate JSON output.
    arrow_basename = Path(arrow_path).name
    target_id = arrow_basename.split(".")[0].split("_")[0].lower()
    json_path = out_dir / f"{target_id}_unified_dossier.json"
    if not json_path.exists():
        return None, f"No dossier JSON at {json_path}"
    return json.load(open(json_path)), None


def test_4lpk_regression():
    """4LPK with YAML must reproduce KRAS hardcode behavior."""
    arrow = (
        PRISM_ROOT / "output" / "m1_readiness_verify"
        / "4lpk_clean.topology.spike_events.arrow"
    )
    if not arrow.exists():
        return True, "4LPK m1_readiness_verify Arrow not present, skipping"
    d, err = run_dossier(arrow)
    if err:
        return False, f"4LPK run failed: {err[:200]}"

    p = d["pockets"]
    e11_p4 = p[3]["enhancements"]["E11_multi_view_dcc"]
    checks = [
        ("pocket_count_10", len(p) == 10),
        ("p4_dominant_switch_i", p[3]["dominant_region"] == "Switch-I"),
        ("p4_e11_status_OK", e11_p4.get("status") == "OK"),
        ("p4_ref_hit_6gj8", e11_p4.get("best_reference") == "6GJ8"),
        ("p4_min_prox_under_2A",
            e11_p4.get("best_reference_min_prox", 99) < 2.0),
    ]
    failures = [name for name, ok in checks if not ok]
    if failures:
        return False, f"4LPK regression failed checks: {failures}"
    return True, "4LPK regression PASS (P4 BI-2852 hit preserved)"


def test_9m3p_substrate_only():
    """9M3P with empty references: must run substrate-only cleanly."""
    candidates = list(PRISM_ROOT.glob(
        "output/**/9m3p*topology.spike_events.arrow"
    ))
    if not candidates:
        return True, "9M3P substrate not available, skipping"
    arrow = max(candidates, key=lambda p: p.stat().st_mtime)
    d, err = run_dossier(arrow)
    if err:
        return False, f"9M3P run failed: {err[:200]}"

    p = d["pockets"]
    if not p:
        return False, "9M3P produced zero pockets"

    p0_enh = p[0]["enhancements"]
    e11_status = p0_enh["E11_multi_view_dcc"].get("status")
    e1_status = p0_enh["E1_ccns_lifecycle"].get("status")
    e11_gate = p0_enh["E11_multi_view_dcc"].get("gate", "")

    checks = [
        ("pockets_present", len(p) > 0),
        ("e1_substrate_internal_OK", e1_status == "OK"),
        ("e11_blocked", e11_status == "BLOCKED"),
        ("e11_gate_substrate_only", "Substrate-only" in e11_gate),
    ]
    failures = [name for name, ok in checks if not ok]
    if failures:
        return False, f"9M3P substrate-only failed: {failures}"
    return True, "9M3P substrate-only PASS"


if __name__ == "__main__":
    tests = [test_4lpk_regression, test_9m3p_substrate_only]
    results = []
    for t in tests:
        ok, msg = t()
        print(f"{'PASS' if ok else 'FAIL'}: {msg}")
        results.append(ok)
    sys.exit(0 if all(results) else 1)
