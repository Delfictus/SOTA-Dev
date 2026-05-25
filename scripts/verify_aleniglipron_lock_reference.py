#!/usr/bin/env python3
"""Reference audit for Aleniglipron and the current top-100 candidates against lock masks."""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, TypeAlias

import polars as pl

from lock_mask_forensics_utils import (
    Coordinate3D,
    atom_indices_in_lock_mask,
    load_lock_mask,
    load_sdf_coordinates,
    voxel_indices_for_atoms,
    z_proxy_atom_indices,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
TRACK_A = REPO_ROOT / "campaigns/glp1r_aleniglipron/track_a_generative"
DEFAULT_SCAFFOLD = TRACK_A / "ALENI-PARENT_6XOX_frame_o3a_relaxed.sdf"
DEFAULT_LOCK_MASK = TRACK_A / "lock_region_mask.json"
DEFAULT_CANDIDATES = TRACK_A / "gflownet_top_100_candidates.parquet"
DEFAULT_SURVIVORS = TRACK_A / "vspace_survivors_full_scale.parquet"
DEFAULT_OUTPUT = TRACK_A / "aleniglipron_lock_reference.json"

JsonObject: TypeAlias = dict[str, Any]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scaffold-sdf", type=Path, default=DEFAULT_SCAFFOLD)
    parser.add_argument("--lock-mask-v1", type=Path, default=DEFAULT_LOCK_MASK)
    parser.add_argument("--lock-mask-v2", type=Path, default=None)
    parser.add_argument("--candidates", type=Path, default=DEFAULT_CANDIDATES)
    parser.add_argument("--survivors", type=Path, default=DEFAULT_SURVIVORS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def atomic_write_json(path: Path, payload: JsonObject) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    tmp_path.replace(path)


def scaffold_stats(lock_mask_path: Path, scaffold_sdf: Path) -> JsonObject:
    lock_mask = load_lock_mask(lock_mask_path)
    coords = load_sdf_coordinates(scaffold_sdf)
    z_atoms = z_proxy_atom_indices(coords)
    mask_atoms = atom_indices_in_lock_mask(coords, lock_mask)
    return {
        "z_proxy_atom_count": len(z_atoms),
        "z_proxy_voxel_count": len(voxel_indices_for_atoms(coords, lock_mask.grid, z_atoms)),
        "mask_atom_count": len(mask_atoms),
        "mask_voxel_count": len(voxel_indices_for_atoms(coords, lock_mask.grid, mask_atoms)),
    }


def top100_mask_hits(lock_mask_path: Path, candidates_path: Path, survivors_path: Path) -> JsonObject:
    lock_mask = load_lock_mask(lock_mask_path)
    candidates = pl.read_parquet(candidates_path)
    survivors = (
        pl.read_parquet(survivors_path)
        .select(["canonical_smiles", "coordinates_json"])
        .unique(subset=["canonical_smiles"], keep="first")
    )
    joined = candidates.join(survivors, on="canonical_smiles", how="left")
    if joined.filter(pl.col("coordinates_json").is_null()).height > 0:
        raise RuntimeError("top100 reference audit found candidates missing coordinates_json")
    hit_count = 0
    max_hits = 0
    for row in joined.iter_rows(named=True):
        coords_json = row["coordinates_json"]
        if not isinstance(coords_json, str):
            raise TypeError("coordinates_json must be a string")
        coords_raw = json.loads(coords_json)
        coords: list[Coordinate3D] = [
            (float(item[0]), float(item[1]), float(item[2]))
            for item in coords_raw
            if isinstance(item, list) and len(item) >= 3
        ]
        hits = len(atom_indices_in_lock_mask(coords, lock_mask))
        if hits > 0:
            hit_count += 1
        if hits > max_hits:
            max_hits = hits
    return {
        "candidate_count": joined.height,
        "candidates_with_mask_hits": hit_count,
        "max_mask_hit_atoms_per_candidate": max_hits,
    }


def main() -> int:
    args = parse_args()
    v1_stats = scaffold_stats(Path(args.lock_mask_v1), Path(args.scaffold_sdf))
    v1_top100 = top100_mask_hits(Path(args.lock_mask_v1), Path(args.candidates), Path(args.survivors))

    payload: JsonObject = {
        "schema_version": "PRISM.aleniglipron_lock_reference.v1",
        "epistemic_class": "DERIVED",
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "scaffold_sdf": str(Path(args.scaffold_sdf)),
        "lock_mask_v1": {
            "path": str(Path(args.lock_mask_v1)),
            **v1_stats,
            "top100": v1_top100,
        },
    }

    v2_stats: JsonObject | None = None
    v2_top100: JsonObject | None = None
    if args.lock_mask_v2 is not None:
        v2_stats = scaffold_stats(Path(args.lock_mask_v2), Path(args.scaffold_sdf))
        v2_top100 = top100_mask_hits(Path(args.lock_mask_v2), Path(args.candidates), Path(args.survivors))
        payload["lock_mask_v2"] = {
            "path": str(Path(args.lock_mask_v2)),
            **v2_stats,
            "top100": v2_top100,
        }

    atomic_write_json(Path(args.output), payload)
    z_proxy_lock = float(v1_stats["z_proxy_atom_count"])
    residue_v1_lock = float(v1_stats["mask_atom_count"])
    residue_v2_lock = float(v2_stats["mask_atom_count"]) if v2_stats is not None else float("nan")
    print(
        "aleniglipron_reference_test "
        f"z_proxy_lock={z_proxy_lock:.1f} "
        f"residue_v1_lock={residue_v1_lock:.1f} "
        f"residue_v2_lock={residue_v2_lock:.1f} "
        f"top100_v1={v1_top100['candidates_with_mask_hits']}/{v1_top100['candidate_count']} "
        f"output={args.output}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
