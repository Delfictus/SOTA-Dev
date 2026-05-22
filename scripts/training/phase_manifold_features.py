#!/usr/bin/env python3
"""
Phase manifold features extractor.

Wraps scripts/phase_manifold_ranker.py for batch use over many TWIN targets,
then parses the site_manifests/ output to produce a per-residue feature parquet.

For each residue:
  - max_phase_manifold_score over all sites it belongs to
  - is_kcc_driver, is_lining, is_hot_phase, is_cold_phase, is_burst_motion,
    is_validation_contact, is_all_region (binary indicators)
  - n_sites_as_driver, n_sites_as_lining, ... (integer counts)
  - top1_site_score, top5_site_score (score of the highest-ranked site
    containing this residue, if any)
  - top1_site_classification, top1_site_centroid_view
"""
from __future__ import annotations
import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import polars as pl


SUPPORT_FAMILIES = (
    "all_region_residues",
    "lining_or_surface_residues",
    "kcc_driver_residues",
    "hot_phase_supported_residues",
    "cold_phase_supported_residues",
    "burst_motion_supported_residues",
    "validation_contact_residues",
)

SHORT_NAMES = {
    "all_region_residues": "all_region",
    "lining_or_surface_residues": "lining",
    "kcc_driver_residues": "kcc_driver",
    "hot_phase_supported_residues": "hot_phase",
    "cold_phase_supported_residues": "cold_phase",
    "burst_motion_supported_residues": "burst_motion",
    "validation_contact_residues": "validation_contact",
}

CLASS_MAP = {
    "Cryptic": 1, "cryptic": 1, "CRYPTIC": 1,
    "Druggable": 2, "druggable": 2, "DRUGGABLE": 2,
    "Orthosteric": 3, "orthosteric": 3,
    "Surface": 4, "surface": 4,
    "Inert": 5, "INERT": 5, "inert": 5,
}

CENTROID_VIEW_MAP = {
    "geometric": 1, "lining": 2, "driver": 3,
    "hot_phase": 4, "cold_phase": 5, "burst_motion": 6,
    "validation_structural": 7, "ligand_adjacent_subcluster": 8,
}


def run_phase_manifold_ranker(arrow: Path, binding_sites: Path, kcc: Path, outdir: Path,
                               script_path: Path) -> bool:
    """Run phase_manifold_ranker.py as subprocess. Returns True on success."""
    outdir.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable, str(script_path),
        "--arrow", str(arrow),
        "--binding-sites", str(binding_sites),
        "--kcc", str(kcc),
        "--outdir", str(outdir),
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        if result.returncode != 0:
            print(f"phase_manifold_ranker failed (rc={result.returncode}):\n{result.stderr[-2000:]}", file=sys.stderr)
            return False
    except Exception as e:
        print(f"subprocess error: {e}", file=sys.stderr)
        return False
    return (outdir / "ranked_site_manifest_atlas.json").exists()


def parse_atlas(outdir: Path, n_res: int) -> pl.DataFrame:
    """Parse phase_manifold_ranker outputs into per-residue feature DataFrame.

    Residue IDs in support families are TOPOLOGY indices (0-based).
    """
    atlas_path = outdir / "ranked_site_manifest_atlas.json"
    manifests_dir = outdir / "site_manifests"

    if not atlas_path.exists():
        return _empty_df(n_res)

    try:
        atlas = json.loads(atlas_path.read_text())
    except Exception:
        return _empty_df(n_res)

    site_list = atlas.get("sites", []) or atlas.get("ranked_sites", []) or []
    if not site_list and manifests_dir.exists():
        site_list = [json.loads(p.read_text()) for p in sorted(manifests_dir.glob("*.json"))]

    if not site_list:
        return _empty_df(n_res)

    site_list = sorted(site_list, key=lambda m: m.get("rank", 9999))

    cols = {
        "max_phase_manifold_score": np.full(n_res, -np.inf, dtype=np.float32),
        "top1_site_score": np.zeros(n_res, dtype=np.float32),
        "top1_site_classification": np.zeros(n_res, dtype=np.int8),
        "top1_site_centroid_view": np.zeros(n_res, dtype=np.int8),
        "top1_site_rank": np.zeros(n_res, dtype=np.int16),
        "is_in_top1_phase_site": np.zeros(n_res, dtype=np.int8),
        "is_in_top5_phase_site": np.zeros(n_res, dtype=np.int8),
        "is_in_top10_phase_site": np.zeros(n_res, dtype=np.int8),
        "n_sites_containing_residue": np.zeros(n_res, dtype=np.int16),
    }
    for fam in SUPPORT_FAMILIES:
        short = SHORT_NAMES[fam]
        cols[f"is_{short}"] = np.zeros(n_res, dtype=np.int8)
        cols[f"n_sites_as_{short}"] = np.zeros(n_res, dtype=np.int16)

    for site in site_list:
        rank = int(site.get("rank", 9999))
        score = float(site.get("final_phase_manifold_score", 0.0))
        site_identity = site.get("site_identity") or {}
        classification = site_identity.get("classification") or ""
        class_id = CLASS_MAP.get(classification, 0)
        centroid_manifold = site.get("centroid_manifold") or {}
        selected_view = centroid_manifold.get("selected_centroid_view") or ""
        view_id = CENTROID_VIEW_MAP.get(selected_view, 0)
        rsf = site.get("residue_support_family") or {}
        site_residues_union = set()

        for fam in SUPPORT_FAMILIES:
            short = SHORT_NAMES[fam]
            res_list = rsf.get(fam) or []
            for r in res_list:
                try:
                    ri = int(r)
                except (TypeError, ValueError):
                    continue
                if not (0 <= ri < n_res):
                    continue
                cols[f"is_{short}"][ri] = 1
                cols[f"n_sites_as_{short}"][ri] += 1
                site_residues_union.add(ri)

        for ri in site_residues_union:
            cols["n_sites_containing_residue"][ri] += 1
            if score > cols["max_phase_manifold_score"][ri]:
                cols["max_phase_manifold_score"][ri] = score
                cols["top1_site_score"][ri] = score
                cols["top1_site_classification"][ri] = class_id
                cols["top1_site_centroid_view"][ri] = view_id
                cols["top1_site_rank"][ri] = rank
            if rank == 1:
                cols["is_in_top1_phase_site"][ri] = 1
            if rank <= 5:
                cols["is_in_top5_phase_site"][ri] = 1
            if rank <= 10:
                cols["is_in_top10_phase_site"][ri] = 1

    cols["max_phase_manifold_score"] = np.where(
        np.isinf(cols["max_phase_manifold_score"]),
        0.0,
        cols["max_phase_manifold_score"],
    ).astype(np.float32)

    cols["residue_id"] = np.arange(n_res, dtype=np.int32)
    return pl.DataFrame(cols)


def _empty_df(n_res: int) -> pl.DataFrame:
    cols = {
        "residue_id": np.arange(n_res, dtype=np.int32),
        "max_phase_manifold_score": np.zeros(n_res, dtype=np.float32),
        "top1_site_score": np.zeros(n_res, dtype=np.float32),
        "top1_site_classification": np.zeros(n_res, dtype=np.int8),
        "top1_site_centroid_view": np.zeros(n_res, dtype=np.int8),
        "top1_site_rank": np.zeros(n_res, dtype=np.int16),
        "is_in_top1_phase_site": np.zeros(n_res, dtype=np.int8),
        "is_in_top5_phase_site": np.zeros(n_res, dtype=np.int8),
        "is_in_top10_phase_site": np.zeros(n_res, dtype=np.int8),
        "n_sites_containing_residue": np.zeros(n_res, dtype=np.int16),
    }
    for fam in SUPPORT_FAMILIES:
        short = SHORT_NAMES[fam]
        cols[f"is_{short}"] = np.zeros(n_res, dtype=np.int8)
        cols[f"n_sites_as_{short}"] = np.zeros(n_res, dtype=np.int16)
    return pl.DataFrame(cols)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arrow", type=Path, required=True)
    ap.add_argument("--binding-sites", type=Path, required=True)
    ap.add_argument("--kcc", type=Path, required=True)
    ap.add_argument("--outdir", type=Path, required=True, help="phase_manifold_ranker outdir")
    ap.add_argument("--n-residues", type=int, required=True)
    ap.add_argument("--output", type=Path, required=True, help="*_phasemanifold.parquet")
    ap.add_argument("--ranker-script", type=Path,
                    default=Path("scripts/phase_manifold_ranker.py"))
    ap.add_argument("--skip-run", action="store_true",
                    help="Skip the subprocess call (assume outdir already populated)")
    args = ap.parse_args()

    if not args.skip_run:
        ok = run_phase_manifold_ranker(args.arrow, args.binding_sites, args.kcc, args.outdir,
                                       args.ranker_script)
        if not ok:
            print("phase_manifold_ranker failed; writing empty parquet", file=sys.stderr)

    df = parse_atlas(args.outdir, args.n_residues)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(args.output, compression="zstd", compression_level=9)
    print(f"wrote {args.output}: {df.height} × {df.width}")


if __name__ == "__main__":
    main()
