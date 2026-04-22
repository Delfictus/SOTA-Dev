#!/usr/bin/env python3
"""D5_V2 phase-2 parity + timing harness for anchor_point_map + growth_vector_map.

Runs each consumer in A (triad hidden -> JSON fallback) and C (triad present
-> D5_V2 view). Writes canonical JSON fingerprints and compares.
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

TARGET_DIR = Path("/home/diddy/prism-working/m1-strict-dcc-panel/m1_2akr")
STEM = "2akr"
ENG = TARGET_DIR / "artifacts/5_engine"
META_PATH = ENG / f"{STEM}.run_metadata.json"
META_HIDDEN = ENG / f"{STEM}.run_metadata.json._hidden_for_d5_v2_timing"


def _set_target(target_dir: Path, stem: str):
    global TARGET_DIR, STEM, ENG, META_PATH, META_HIDDEN
    TARGET_DIR = Path(target_dir)
    STEM = stem
    ENG = TARGET_DIR / "artifacts/5_engine"
    META_PATH = ENG / f"{STEM}.run_metadata.json"
    META_HIDDEN = ENG / f"{STEM}.run_metadata.json._hidden_for_d5_v2_timing"


def _hide_triad():
    if META_PATH.exists():
        META_PATH.rename(META_HIDDEN)


def _restore_triad():
    if META_HIDDEN.exists():
        META_HIDDEN.rename(META_PATH)


def run_anchor(mode: str) -> dict:
    """Invoke AnchorPointMapper.compute_all on all sites; return canonical dict."""
    import scripts.anchor_point_map as apm
    bs_p = ENG / f"{STEM}.binding_sites.json"
    data = json.loads(bs_p.read_text())
    sites = data if isinstance(data, list) else (data.get("sites") or [])
    mapper = apm.AnchorPointMapper()
    results = mapper.compute_all(sites, str(ENG))
    out = {
        "n_sites": len(results),
        "per_site": {
            str(sid): {
                "n_anchors": am.n_anchors,
                "anchor_density": am.anchor_density,
                "pocket_centroid": list(am.pocket_centroid),
                "top_anchor": (
                    {
                        "residue_id": am.anchors[0].residue_id,
                        "residue_name": am.anchors[0].residue_name,
                        "chain": am.anchors[0].chain,
                        "interaction_type": am.anchors[0].interaction_type,
                        "x": am.anchors[0].x, "y": am.anchors[0].y, "z": am.anchors[0].z,
                        "spike_intensity": am.anchors[0].spike_intensity,
                        "temporal_persistence": am.anchors[0].temporal_persistence,
                        "confidence": am.anchors[0].confidence,
                    } if am.anchors else None
                ),
            }
            for sid, am in sorted(results.items())
        },
    }
    return out


def run_growth(mode: str) -> dict:
    """Invoke GrowthVectorMapper.compute_all using anchor_maps from AnchorPointMapper.compute_all.
    This exercises the inherited speedup path (growth_vector_map does not read spikes,
    so parity depends on anchor map parity)."""
    import scripts.anchor_point_map as apm
    import scripts.growth_vector_map as gvm
    bs_p = ENG / f"{STEM}.binding_sites.json"
    data = json.loads(bs_p.read_text())
    sites = data if isinstance(data, list) else (data.get("sites") or [])
    anchor_mapper = apm.AnchorPointMapper()
    anchor_maps = anchor_mapper.compute_all(sites, str(ENG))
    growth_mapper = gvm.GrowthVectorMapper()
    growth_maps = growth_mapper.compute_all(sites, anchor_maps)
    out = {
        "n_sites": len(growth_maps),
        "per_site": {
            str(sid): {
                "n_vectors": gm.n_vectors,
                "n_sub_pockets": gm.n_sub_pockets,
                "top_vector": (
                    {
                        "source_anchor_label": gm.vectors[0].source_anchor_label,
                        "free_length": gm.vectors[0].free_length,
                        "vector_score": gm.vectors[0].vector_score,
                    } if gm.vectors else None
                ),
            }
            for sid, gm in sorted(growth_maps.items())
        },
    }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", required=True, choices=["A", "C"])
    ap.add_argument("--script", required=True, choices=["anchor", "growth"])
    ap.add_argument("--target-dir", default=None)
    ap.add_argument("--stem", default=None)
    args = ap.parse_args()
    if args.target_dir and args.stem:
        _set_target(Path(args.target_dir), args.stem)
    try:
        if args.mode == "A":
            _hide_triad()
        t0 = time.perf_counter()
        if args.script == "anchor":
            out = run_anchor("A_JSON" if args.mode == "A" else "C_D5_V2")
        else:
            out = run_growth("A_JSON" if args.mode == "A" else "C_D5_V2")
        elapsed = time.perf_counter() - t0
    finally:
        _restore_triad()
    canon = json.dumps(out, sort_keys=True, default=str)
    h = hashlib.sha256(canon.encode()).hexdigest()
    out_path = Path(f"/tmp/d5_v2_p2_{args.script}_{STEM}_{args.mode}.json")
    out_path.write_text(canon)
    print(f"MODE={args.mode}")
    print(f"SCRIPT={args.script}")
    print(f"SHA256={h}")
    print(f"BYTES={len(canon)}")
    print(f"INTERNAL_ELAPSED={elapsed:.3f}s")


if __name__ == "__main__":
    main()
