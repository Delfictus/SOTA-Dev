#!/usr/bin/env python3
"""Verify a unified dossier JSON has all expected enhancement keys
populated correctly. Handles E14-E18 dict shapes."""
import json
import sys
from pathlib import Path


def verify_dossier(json_path):
    d = json.load(open(json_path))
    print(f"\n=== {Path(json_path).name} ===")
    print(f"Top-level keys: {list(d.keys())}")
    print(f"run_level_E10: "
          f"{'present' if 'run_level_E10' in d else 'MISSING'}")
    e17_key = next(
        (k for k in ("run_level_E17", "run_level_E17_persistent_homology")
         if k in d), None,
    )
    print(f"run_level_E17: {'present (' + e17_key + ')' if e17_key else 'MISSING'}")
    print(f"Pocket count: {len(d.get('pockets', []))}")

    if not d.get('pockets'):
        return False

    p1 = d['pockets'][0]
    enh = p1.get('enhancements', {})
    print(f"\nPocket 1 enhancement statuses ({len(enh)} keys):")

    expected = [
        "E1_ccns_lifecycle", "E2_wavelength", "E3_phase_bits_mi",
        "E4_first_passage", "E5_aromatic_graph", "E6_cooperative",
        "E7_source_consensus", "E8_voxel_adjacency",
        "E9_wd_directional", "E11_multi_view_dcc",
        "E12_phase_energy", "E13_md_anchor",
        "E14_md_anchor_diagnostic",
        "E15_hellinger_residue_distance",
        "E16_wasserstein_alignment",
        "E18_geodesic_centroid",
    ]

    all_ok = True
    for k in expected:
        if k not in enh:
            print(f"  {k}: MISSING")
            all_ok = False
            continue
        e = enh[k]
        if isinstance(e, dict) and "status" in e:
            print(f"  {k}: {e['status']}")
        elif isinstance(e, dict) and all(
            isinstance(v, dict) and "status" in v for v in e.values()
        ):
            statuses = [v["status"] for v in e.values()]
            ok = sum(1 for s in statuses if s == "OK")
            blocked = sum(1 for s in statuses if s == "BLOCKED")
            print(f"  {k}: per-ref {ok}/{len(statuses)} OK, "
                  f"{blocked} BLOCKED")
        else:
            shape = type(e).__name__
            n_keys = len(e) if isinstance(e, dict) else "?"
            print(f"  {k}: present (shape={shape}, n_keys={n_keys})")
    return all_ok


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("usage: verify_dossier.py <dossier.json>")
        sys.exit(1)
    ok = verify_dossier(sys.argv[1])
    sys.exit(0 if ok else 1)
